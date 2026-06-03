"""Multi-year IBIS-JAX runner on GPU (RTX 4090), no F77 in the loop.

Loads GSWP3 forcing for all 117 land cells in one process, runs the JAX
timestep_body inside jax.lax.scan (compiled once, executed per month),
saving per-year state checkpoints and monthly stats.

CRITICAL: diurnal disaggregation is done with NUMPY (bit-exact to the CPU
lockstep runner global_gswp3_multiyear.py), then the precomputed hourly
forcing is passed as a (n_hours, n_cells) stack into the JIT'd scan. This
prevents ULP-level differences in jnp.sin/cos/arccos (XLA libm) from
amplifying via freeze/thaw discontinuities over 46 years (an earlier
JAX-native disagg gave ~16 K median divergence vs CPU at year 46 from this
mechanism — same amplification class as finding #25).

Projected speed: ~10 min for 50 yr × 117 cells on RTX 4090 FP64
(vs ~58 hr per CPU batch × 24 batches = much more on CPU).

Usage:
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \\
    PYTHONPATH=python-port:tests verify/.venv_gpu/bin/python -u \\
        -m verify.timestepping.pipeline.global_gswp3_gpu \\
        --year-start 1951 --year-end 2000

F77 lockstep is NOT performed here (it would require ctypes round-trips on
CPU; the F77 wrapper is also hardcoded to NPOI=5, incompatible with the
117-cell vmap). F77-JAX equivalence is already established at ULP level
by the 46-yr CPU lockstep run (see verify/STATUS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import date
from pathlib import Path

# Must be set BEFORE jax import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# CRITICAL: import verify_lsxmain FIRST. It applies two monkey patches at
# import time that are required for F77↔JAX equivalence:
#   - ibis.physics.radiation.twoset → _twoset_f77_ztab (finding #1)
#   - ibis.physics.interception.twet3 → _patched_twet3 (finding #4)
# These patches change the ALGORITHM used by timestep_body. Without this
# import, the GPU runner uses unpatched algorithms — physically valid but
# diverging from the CPU lockstep run by ~1e-5 K/month, which freeze/thaw
# discontinuities amplify to ~16 K over 46 years (observed in v2 run).
from verify.timestepping import verify_lsxmain as _vl  # noqa: F401 (side effects only)

from verify.timestepping.pipeline.global_gswp3_test import (
    build_global_cell_grid, read_gswp3_month, diurnal_disaggregate,
)
# Reuse F77-side defs for the synthetic-template index → JAX initial state
from tests.f2py.synthetic_inputs import (
    make_soil_inputs, make_snow_inputs, make_atm_inputs, make_veg_inputs,
)
from ibis.dimensions import GridConfig
from ibis.state.model_state import create_ibis_state
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body

DTIME = 3600.0
NBAND = 2
NSOILAY = 6
NSNOLAY = 3
NPFT = 12


def make_state_for_cells(cells: list):
    """Build a per-cell synthetic state for an arbitrary cell count by
    indexing the 5 hardcoded synthetic templates.

    Mirrors make_batch_state(global_gswp3_test.py) but does not assume
    npoi == 5 from the F77 wrapper.
    """
    soil_i = make_soil_inputs()
    snow_i = make_snow_inputs()
    atm_i  = make_atm_inputs()
    veg_i  = make_veg_inputs()

    soil_t_idx = np.array([c[3] for c in cells])
    veg_t_idx  = np.array([c[4] for c in cells])

    SYNTH_NTEMPLATES = 5
    def reorder(arr, idx):
        a = np.asarray(arr)
        if a.ndim >= 1 and a.shape[0] == SYNTH_NTEMPLATES:
            return a[idx]
        return a

    soil_new = {k: reorder(v, soil_t_idx) for k, v in soil_i.items()}
    snow_new = {k: reorder(v, soil_t_idx) for k, v in snow_i.items()}
    atm_new  = {k: reorder(v, soil_t_idx) for k, v in atm_i.items()}
    veg_new  = {k: reorder(v, veg_t_idx)  for k, v in veg_i.items()}
    soil_new["sand"] = np.round(soil_new["sand"])
    soil_new["clay"] = np.round(soil_new["clay"])
    return soil_new, snow_new, atm_new, veg_new


def setup_jax_state_for_n(npoi: int, soil_i, snow_i, atm_i, veg_i):
    """N-cell version of setup_jax_state from verify_lsxmain.py.

    Replaces the hardcoded NPOI=5 references with `npoi`."""
    grid = GridConfig(nlon=npoi, nlat=1, npoi=npoi, xres=1.0, yres=1.0,
                       nband=NBAND, nsoilay=NSOILAY, nsnolay=NSNOLAY, npft=NPFT)
    st = create_ibis_state(grid)

    new_soil = eqx.tree_at(
        lambda s: (s.tsoi, s.wsoi, s.wisoi, s.sand, s.clay, s.poros,
                   s.hydraul, s.suction, s.bex, s.sfield, s.swilt,
                   s.wpud, s.wipud, s.tg, s.hsoi,
                   s.albsav, s.albsan, s.z0soi),
        st.soil,
        (jnp.asarray(soil_i["tsoi"]), jnp.asarray(soil_i["wsoi"]),
         jnp.asarray(soil_i["wisoi"]),
         jnp.asarray(soil_i["sand"]), jnp.asarray(soil_i["clay"]),
         jnp.asarray(soil_i["poros"]), jnp.asarray(soil_i["hydraul"]),
         jnp.asarray(soil_i["suction"]), jnp.asarray(soil_i["bex"]),
         jnp.asarray(soil_i["sfield"]), jnp.asarray(soil_i["swilt"]),
         jnp.asarray(soil_i["wpud"]), jnp.asarray(soil_i["wipud"]),
         jnp.asarray(soil_i["tg"]), jnp.asarray(soil_i["hsoi"]),
         jnp.full((npoi,), 0.10), jnp.full((npoi,), 0.20),
         jnp.full((npoi,), 0.005)),
    )
    new_snow = eqx.tree_at(
        lambda s: (s.fi, s.tsno, s.hsno, s.ti, s.tlsub, s.hsnomin, s.fimax),
        st.snow,
        (jnp.asarray(snow_i["fi"]), jnp.asarray(snow_i["tsno"]),
         jnp.asarray(snow_i["hsno"]), jnp.asarray(soil_i["ti"]),
         jnp.asarray(veg_i["tlsub"]),
         0.10, 0.95),
    )
    new_atm = eqx.tree_at(
        lambda a: (a.ta, a.qa, a.ua, a.psurf, a.coszen,
                   a.solad, a.solai, a.fira, a.raina, a.snowa),
        st.atm,
        (jnp.asarray(atm_i["ta"]), jnp.asarray(atm_i["qa"]),
         jnp.asarray(atm_i["ua"]), jnp.asarray(atm_i["psurf"]),
         jnp.asarray(atm_i["coszen"]),
         jnp.asarray(atm_i["solad"]), jnp.asarray(atm_i["solai"]),
         jnp.asarray(atm_i["fira"]),
         jnp.asarray(atm_i["raina"]), jnp.asarray(atm_i["snowa"])),
    )
    new_can = eqx.tree_at(
        lambda c: (c.tu, c.ts, c.tl, c.t12, c.t34, c.q12, c.q34,
                   c.wliqu, c.wliqs, c.wliql, c.wsnou, c.wsnos, c.wsnol,
                   c.gs_upper, c.gs_lower,
                   c.a10scalparamu, c.a10daylightu,
                   c.a10scalparaml, c.a10daylightl),
        st.canopy,
        (jnp.asarray(veg_i["tu"]), jnp.asarray(veg_i["ts"]),
         jnp.asarray(veg_i["tl"]),
         jnp.asarray(veg_i["t12"]), jnp.asarray(veg_i["t34"]),
         jnp.asarray(veg_i["q12"]), jnp.asarray(veg_i["q34"]),
         jnp.asarray(veg_i["wliqu"]), jnp.asarray(veg_i["wliqs"]),
         jnp.asarray(veg_i["wliql"]),
         jnp.asarray(veg_i["wsnou"]), jnp.asarray(veg_i["wsnos"]),
         jnp.asarray(veg_i["wsnol"]),
         jnp.full((npoi, 2), 0.5), jnp.full((npoi, 3), 0.5),
         jnp.full((npoi,), 1.0),
         jnp.full((npoi,), 21600.0),
         jnp.full((npoi,), 1.0),
         jnp.full((npoi,), 21600.0)),
    )
    new_veg = eqx.tree_at(
        lambda v: (v.fu, v.fl, v.lai, v.sai, v.ztop, v.zbot, v.frac,
                   v.froot, v.exist),
        st.vegetation,
        (jnp.asarray(veg_i["fu"]), jnp.asarray(veg_i["fl"]),
         jnp.asarray(veg_i["lai"]), jnp.asarray(veg_i["sai"]),
         jnp.asarray(veg_i["ztop"]), jnp.asarray(veg_i["zbot"]),
         jnp.asarray(veg_i["frac"]), jnp.asarray(veg_i["froot"]),
         jnp.zeros((npoi, NPFT))),
    )
    state = eqx.tree_at(
        lambda s: (s.soil, s.snow, s.atm, s.canopy, s.vegetation),
        st,
        (new_soil, new_snow, new_atm, new_can, new_veg),
    )
    return state


# ===========================================================================
# Precompute hourly forcing in numpy (bit-exact to CPU lockstep runner)
# ===========================================================================

def precompute_hourly_forcing_numpy(day_np, lats, jday_start):
    """Build the full month of hourly atm forcing in numpy, using the EXACT
    same diurnal_disaggregate as the CPU lockstep runner. Returns a dict of
    (n_hours, n_cells) or (n_hours, n_cells, n_band) arrays.

    Bit-exact equivalence with CPU lockstep is critical: any libm difference
    (e.g. JAX-XLA sin vs numpy sin) amplifies via freeze/thaw discontinuities
    over multi-decadal runs (finding #25 mechanism).
    """
    n_days = day_np["tas"].shape[0]
    n_hours = n_days * 24
    n_cells = lats.shape[0]
    out = {
        "ta":     np.zeros((n_hours, n_cells), dtype=np.float64),
        "qa":     np.zeros((n_hours, n_cells), dtype=np.float64),
        "ua":     np.zeros((n_hours, n_cells), dtype=np.float64),
        "psurf":  np.zeros((n_hours, n_cells), dtype=np.float64),
        "coszen": np.zeros((n_hours, n_cells), dtype=np.float64),
        "solad":  np.zeros((n_hours, n_cells, 2), dtype=np.float64),
        "solai":  np.zeros((n_hours, n_cells, 2), dtype=np.float64),
        "fira":   np.zeros((n_hours, n_cells), dtype=np.float64),
        "raina":  np.zeros((n_hours, n_cells), dtype=np.float64),
        "snowa":  np.zeros((n_hours, n_cells), dtype=np.float64),
    }
    for day_idx in range(n_days):
        jday = jday_start + day_idx
        day_data = {k: v[day_idx] for k, v in day_np.items()}
        for hour in range(24):
            atm = diurnal_disaggregate(day_data, float(hour), jday, lats)
            t = day_idx * 24 + hour
            for k in ("ta", "qa", "ua", "psurf", "coszen", "fira", "raina", "snowa"):
                out[k][t] = atm[k]
            out["solad"][t] = atm["solad"]
            out["solai"][t] = atm["solai"]
    return out


def precompute_hourly_forcing_numpy_vectorized(day_np, lats, jday_start):
    """Vectorized (broadcasted) version of precompute_hourly_forcing_numpy.

    Same math, same operands, same FP ordering as the scalar version (the
    elementwise expressions are identical; numpy broadcasting preserves
    per-element FP semantics on CPU). Verified bit-exact at every cell.

    50-100× faster than the per-hour Python loop at 117 cells; enables
    intermediate (1°×1°, ~17K cells) and full-resolution (0.5°×0.5°, ~95K
    cells) global runs without the Python loop dominating wall-clock.
    """
    n_days = day_np["tas"].shape[0]
    n_hours = n_days * 24
    n_cells = lats.shape[0]

    days = np.arange(n_days)
    hours = np.arange(24, dtype=np.float64)
    jdays = (jday_start + days).astype(np.float64)  # (n_days,)

    # Solar geometry, broadcast to (n_days, 24, n_cells)
    orbit = 2.0 * np.pi * jdays / 365.2425                              # (n_days,)
    decl = (0.006918 - 0.399912 * np.cos(orbit) + 0.070257 * np.sin(orbit)
            - 0.006758 * np.cos(2 * orbit) + 0.000907 * np.sin(2 * orbit)
            - 0.002697 * np.cos(3 * orbit) + 0.001480 * np.sin(3 * orbit))  # (n_days,)
    lat_rad = np.radians(lats)                                          # (n_cells,)
    ha = (hours - 12.0) * (2 * np.pi / 24.0)                            # (24,)

    # Broadcasting shape: (n_days, 24, n_cells)
    sin_lat = np.sin(lat_rad)[None, None, :]      # (1,1,n_cells)
    cos_lat = np.cos(lat_rad)[None, None, :]
    sin_decl = np.sin(decl)[:, None, None]        # (n_days,1,1)
    cos_decl = np.cos(decl)[:, None, None]
    cos_ha = np.cos(ha)[None, :, None]            # (1,24,1)
    coszen = np.maximum(0.0, sin_lat * sin_decl + cos_lat * cos_decl * cos_ha)

    cos_h0 = np.clip(-np.tan(lat_rad) * np.tan(decl)[:, None], -1.0, 1.0)  # (n_days, n_cells)
    h0 = np.arccos(cos_h0)
    mean_cz = (h0 * np.sin(lat_rad)[None, :] * np.sin(decl)[:, None]
               + np.cos(lat_rad)[None, :] * np.cos(decl)[:, None] * np.sin(h0)) / np.pi  # (n_days, n_cells)
    mean_cz = np.maximum(mean_cz, 1e-8)[:, None, :]                     # (n_days, 1, n_cells)

    rsds = day_np["rsds"][:, None, :]                                   # (n_days, 1, n_cells)
    sw_now = np.maximum(0.0, rsds * coszen / mean_cz)                   # (n_days, 24, n_cells)
    diffuse_frac = 0.3 + 0.4 * (1.0 - coszen)
    solar_dir = sw_now * (1.0 - diffuse_frac)
    solar_dif = sw_now * diffuse_frac
    solad = np.stack([0.5 * solar_dir, 0.5 * solar_dir], axis=-1)       # (n_days, 24, n_cells, 2)
    solai = np.stack([0.5 * solar_dif, 0.5 * solar_dif], axis=-1)

    # Temperature: daily mean ± 5K with peak at 14:00 — vectorize over (day, hour, cell)
    tas = day_np["tas"][:, None, :]                                     # (n_days, 1, n_cells)
    ta_sinusoid = 5.0 * np.sin((hours - 8.0) * 2 * np.pi / 24.0)[None, :, None]
    ta = tas + ta_sinusoid                                              # (n_days, 24, n_cells)

    huss = day_np["huss"][:, None, :]
    qa = np.maximum(huss, 1e-4) + np.zeros((1, 24, 1))                  # broadcast to full shape
    wind = day_np["wind"][:, None, :]
    ua = np.maximum(wind, 0.5) + np.zeros((1, 24, 1))
    psurf = day_np["ps"][:, None, :] + np.zeros((1, 24, 1))
    fira  = day_np["rlds"][:, None, :] + np.zeros((1, 24, 1))

    pr = day_np["pr"][:, None, :]
    raina = np.where(ta >= 273.16, pr, 0.0)
    snowa = np.where(ta <  273.16, pr, 0.0)

    # Reshape (n_days, 24, ...) → (n_days*24, ...)
    def flat(x):
        return x.reshape(n_hours, *x.shape[2:])

    return {
        "ta":     np.ascontiguousarray(flat(ta),     dtype=np.float64),
        "qa":     np.ascontiguousarray(flat(qa),     dtype=np.float64),
        "ua":     np.ascontiguousarray(flat(ua),     dtype=np.float64),
        "psurf":  np.ascontiguousarray(flat(psurf),  dtype=np.float64),
        "coszen": np.ascontiguousarray(flat(coszen), dtype=np.float64),
        "solad":  np.ascontiguousarray(flat(solad),  dtype=np.float64),
        "solai":  np.ascontiguousarray(flat(solai),  dtype=np.float64),
        "fira":   np.ascontiguousarray(flat(fira),   dtype=np.float64),
        "raina":  np.ascontiguousarray(flat(raina),  dtype=np.float64),
        "snowa":  np.ascontiguousarray(flat(snowa),  dtype=np.float64),
    }


def inject_atm(state, atm):
    new_atm = eqx.tree_at(
        lambda a: (a.ta, a.qa, a.ua, a.psurf, a.coszen,
                   a.solad, a.solai, a.fira, a.raina, a.snowa),
        state.atm,
        (atm["ta"], atm["qa"], atm["ua"], atm["psurf"], atm["coszen"],
         atm["solad"], atm["solai"], atm["fira"],
         atm["raina"], atm["snowa"]),
    )
    return eqx.tree_at(lambda s: s.atm, state, new_atm)


# ===========================================================================
# Month-long scan (JIT'd; one launch per month after compile)
# ===========================================================================

@eqx.filter_jit
def run_one_month(state, consts, params, hourly):
    """Scan over n_hours = n_days*24 hourly timesteps with precomputed forcing.

    hourly: dict with each var stacked along axis 0:
            ta, qa, ua, psurf, coszen, fira, raina, snowa: (n_hours, n_cells)
            solad, solai:                                  (n_hours, n_cells, 2)
    """
    n_steps = hourly["ta"].shape[0]

    def body(carry, t):
        state, nan_c, oor_c = carry
        atm = {k: v[t] for k, v in hourly.items()}
        state = inject_atm(state, atm)
        state = timestep_body(state, consts, params, DTIME)

        nan_now = (jnp.isnan(state.canopy.tu).any()
                   | jnp.isnan(state.soil.tg).any()
                   | jnp.isnan(state.soil.wsoi).any())
        oor_now = ((state.canopy.tu < 150.0).any() | (state.canopy.tu > 360.0).any()
                   | (state.soil.tg < 150.0).any() | (state.soil.tg > 360.0).any())
        return (state, nan_c + nan_now.astype(jnp.int32),
                oor_c + oor_now.astype(jnp.int32)), None

    init = (state, jnp.int32(0), jnp.int32(0))
    (final_state, nan_count, oor_count), _ = jax.lax.scan(
        body, init, jnp.arange(n_steps))
    return final_state, nan_count, oor_count


# ===========================================================================
# NAS-resilient month reader (retry-with-backoff)
# ===========================================================================

def read_gswp3_month_with_retry(year, month, cells, max_retries=8):
    delay = 1.0
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return read_gswp3_month(year, month, cells)
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            print(f"  ! NAS read failed ({type(e).__name__}: {e}); "
                  f"retry in {delay:.0f}s (attempt {attempt}/{max_retries})",
                  flush=True)
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"GSWP3 read failed after {max_retries} retries: {last_err}")


# ===========================================================================
# Checkpointing
# ===========================================================================

def save_checkpoint(ckpt_dir, year_completed, state):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / f"gpu_state_after_{year_completed}.pkl"
    with open(p, "wb") as f:
        pickle.dump({"year_completed": year_completed, "jax_state": state}, f)
    return p


def find_latest_checkpoint(ckpt_dir, year_start, year_end):
    if not ckpt_dir.exists():
        return None, year_start
    best = None
    best_year = -1
    for p in ckpt_dir.glob("gpu_state_after_*.pkl"):
        try:
            yr = int(p.stem.split("_")[-1])
            if year_start <= yr < year_end and yr > best_year:
                best_year = yr
                best = p
        except ValueError:
            pass
    if best is None:
        return None, year_start
    return best, best_year + 1


def restore_checkpoint(ckpt_path):
    with open(ckpt_path, "rb") as f:
        d = pickle.load(f)
    return d["jax_state"]


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-start", type=int, default=1951,
                    help="First MODEL year (e.g. 1 for spin-up, 1951 for hist).")
    ap.add_argument("--year-end",   type=int, default=2000,
                    help="Last MODEL year (inclusive).")
    ap.add_argument("--forcing-cycle-start", type=int, default=None,
                    help="If set, forcing cycles within [cycle-start, cycle-end]."
                         " Otherwise forcing-year = model-year.")
    ap.add_argument("--forcing-cycle-end",   type=int, default=None,
                    help="See --forcing-cycle-start.")
    ap.add_argument("--output-root", type=str,
                    default="/shared/nas2/Hangkai/ibis/verify_outputs")
    ap.add_argument("--output-name", type=str, default=None,
                    help="Output dir name (default = gpu_<start>_<end>).")
    ap.add_argument("--no-resume", action="store_true",
                    help="Start fresh from year-start even if a checkpoint exists.")
    args = ap.parse_args()

    # Determine forcing-year function
    if args.forcing_cycle_start is not None:
        if args.forcing_cycle_end is None:
            raise ValueError("--forcing-cycle-end required with --forcing-cycle-start")
        cs, ce = args.forcing_cycle_start, args.forcing_cycle_end
        cycle_len = ce - cs + 1
        def forcing_year_for(model_year):
            # model-year-start maps to cycle-start
            return cs + ((model_year - args.year_start) % cycle_len)
        print(f"Forcing cycle: model-years {args.year_start}-{args.year_end} "
              f"→ cycling forcing {cs}-{ce} ({cycle_len}-yr period)",
              flush=True)
    else:
        forcing_year_for = lambda my: my  # noqa: E731 (1:1 mapping)

    print("=" * 80, flush=True)
    print(f"GPU multi-year run  | JAX backend: {jax.default_backend()}  "
          f"devices: {jax.devices()}", flush=True)
    print(f"Years:  {args.year_start}-{args.year_end} "
          f"({args.year_end - args.year_start + 1} yr)", flush=True)
    print("=" * 80, flush=True)

    output_name = args.output_name or f"gpu_{args.year_start}_{args.year_end}"
    output_dir = Path(args.output_root) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    stats_dir = output_dir / "per_month_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    cells = build_global_cell_grid(lat_step=10.0, lon_step=20.0)
    npoi  = len(cells)
    print(f"Cells:  {npoi}", flush=True)

    consts = get_constants()
    params = default_sim_params(npoi)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    lats_np  = np.array([c[1] for c in cells], dtype=np.float64)
    lat_rad  = jnp.asarray(np.radians(lats_np))

    # Build / restore initial state
    soil_i, snow_i, atm_i, veg_i = make_state_for_cells(cells)
    state = setup_jax_state_for_n(npoi, soil_i, snow_i, atm_i, veg_i)

    next_year = args.year_start
    if not args.no_resume:
        ckpt_path, next_year = find_latest_checkpoint(
            ckpt_dir, args.year_start, args.year_end + 1)
        if ckpt_path is not None:
            print(f"Resuming from checkpoint: {ckpt_path}  "
                  f"(next year: {next_year})", flush=True)
            state = restore_checkpoint(ckpt_path)

    if next_year > args.year_end:
        print(f"Already complete through {args.year_end}.", flush=True)
        return

    start_time = time.time()
    total_steps = 0

    for year in range(next_year, args.year_end + 1):
        yr_t0 = time.time()
        year_nan = 0
        year_oor = 0
        print(f"\n  --- year {year} ({year - args.year_start + 1}/"
              f"{args.year_end - args.year_start + 1}) ---", flush=True)
        for month in range(1, 13):
            fy = forcing_year_for(year)
            try:
                day_np = read_gswp3_month_with_retry(fy, month, cells)
            except RuntimeError as e:
                print(f"  !! FATAL: {e}", flush=True)
                save_checkpoint(ckpt_dir, year - 1, state)
                sys.exit(1)
            # jday based on forcing year so Feb 29 in leap forcing years is handled
            jday_start = (date(fy, month, 1) - date(fy, 1, 1)).days + 1
            n_days = day_np["tas"].shape[0]

            # Precompute hourly forcing with NUMPY (bit-exact to CPU lockstep)
            hourly_np = precompute_hourly_forcing_numpy(day_np, lats_np, jday_start)
            hourly_jax = {k: jnp.asarray(v) for k, v in hourly_np.items()}

            t0 = time.time()
            state, nan_c, oor_c = run_one_month(state, consts, params, hourly_jax)
            jax.block_until_ready(state.atm.ta)
            dt = time.time() - t0
            nan_c = int(nan_c); oor_c = int(oor_c)
            year_nan += nan_c; year_oor += oor_c
            total_steps += n_days * 24

            stats = {
                "year": year, "month": month, "n_days": int(n_days),
                "elapsed_s": dt, "nan_count": nan_c, "oor_count": oor_c,
                "canopy_tu_min": float(state.canopy.tu.min()),
                "canopy_tu_max": float(state.canopy.tu.max()),
                "soil_tg_min": float(state.soil.tg.min()),
                "soil_tg_max": float(state.soil.tg.max()),
            }
            with open(stats_dir / f"y{year}_m{month:02d}.json", "w") as f:
                json.dump(stats, f)
            print(f"    [{year}-{month:02d}] {dt:5.1f}s  "
                  f"NaN={nan_c}  oor={oor_c}  "
                  f"tu=[{stats['canopy_tu_min']:.1f},{stats['canopy_tu_max']:.1f}]K",
                  flush=True)

        save_checkpoint(ckpt_dir, year, state)
        yr_dt = time.time() - yr_t0
        elapsed = time.time() - start_time
        years_done = year - next_year + 1
        years_left = args.year_end - year
        eta_min = (elapsed / years_done) * years_left / 60.0
        print(f"  --- year {year} done in {yr_dt:.1f}s  "
              f"(cumulative {elapsed/60:.1f} min, ETA {eta_min:.1f} min, "
              f"year-NaN={year_nan}, year-oor={year_oor}) ---", flush=True)

    total_dt = time.time() - start_time
    print(f"\n{'='*80}", flush=True)
    print(f"DONE: {args.year_start}-{args.year_end} ({args.year_end - args.year_start + 1} yr)"
          f"  in {total_dt/60:.1f} min  ({total_steps} cell-step batches)",
          flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
