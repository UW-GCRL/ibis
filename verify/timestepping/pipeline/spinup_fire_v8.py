"""200-year spin-up on the repaired path, with fire enabled.

Everything before this was produced by runners with defects that invalidate
their vegetation and carbon state (docs/FIRE.md, 2026-08-24):

  * `biomass_update` was called instead of `dynaveg`, so NO disturbance of any
    kind was applied -- fire could not act however `isimfire` was set;
  * climate normals were never maintained, so `existence()` was never called
    and `exist` stayed at 1.000, growing all 12 PFTs in every cell;
  * `run_global` called `biomass_update` AND `dynaveg`, advancing biomass twice
    per year;
  * fire-killed carbon was emitted in full rather than routed to litter by
    combustion completeness.

**Starting state.** `spinup_carbon_v7_soilbgc` rather than a cold start: its
vegetation was grown under `exist = 1.0` and has to re-equilibrate, but it
carries 50 years of soil carbon built with `soilbgc` wired in, and soil carbon
is the slowest pool. Keeping that head start is worth more than starting the
vegetation clean, which 200 years is ample to correct.

**What to watch.** The three-year test showed biomass falling with fire both on
and off, and three years cannot say whether that settles or continues. The CSV
logs biomass, exist and disturbf every year precisely so that question has an
answer this time.
"""

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python-port"))

from verify.timestepping.pipeline.global_gswp3_test import (
    build_global_cell_grid, latlon_to_grid_idx,
)
# `global_gswp3_test` enables x64 at import. `run_spinup_gswp3` runs float32
# and soilbgc would return float64 into the lax.scan carry. Off again here.
jax.config.update("jax_enable_x64", False)

from ibis.constants import PhysicalConstants
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.io.physics_params import build_sim_params
from ibis.physics.fire import SEASONLENGTH, FireParams
from ibis.state.vegetation import VegetationState
from ibis.timestepping.loops_gswp3 import run_spinup_gswp3

CKPT_IN = ("/shared/nas2/Hangkai/ibis/verify_outputs/"
           "spinup_carbon_v7_soilbgc/checkpoints/state_latest.pkl")
OUT = Path("/shared/nas2/Hangkai/ibis/verify_outputs/spinup_fire_v8")


def migrate(st, npoi):
    """Add accumulator fields absent from checkpoints written before F1/F1b."""
    v = st.vegetation
    need = [n for n in ("nytimes", "nfirestep", "sdry",
                        "nfirestep_vpd", "sdry_vpd") if not hasattr(v, n)]
    if not need:
        return st
    f = {fd.name: getattr(v, fd.name)
         for fd in VegetationState.__dataclass_fields__.values()
         if hasattr(v, fd.name)}
    for n in need:
        f[n] = jnp.zeros(()) if n == "nytimes" else jnp.zeros((npoi,), jnp.float32)
    print(f"  migrated: added {', '.join(need)}", flush=True)
    return eqx.tree_at(lambda s: s.vegetation, st, VegetationState(**f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyears", type=int, default=200)
    ap.add_argument("--cycle", type=int, nargs=2, default=[1901, 2000],
                    help="forcing years to cycle over")
    ap.add_argument("--nsoilcall", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="years between kept checkpoints; 'latest' is always "
                         "written. 25 keeps 8 files, not the 200 that filled "
                         "250 GB across the earlier runs.")
    ap.add_argument("--isimfire", type=int, default=1)
    a = ap.parse_args()

    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"backend {jax.default_backend()} {jax.devices()}", flush=True)

    t0 = time.time()
    with open(CKPT_IN, "rb") as f:
        st = pickle.load(f)["jax_state"]
    npoi = int(np.asarray(st.carbon.totlit).shape[0])
    st = migrate(st, npoi)
    st = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, "dtype") and x.dtype == jnp.float64 else x, st)

    # Clear the fire accumulators: they belong to whatever year the checkpoint
    # stopped in, and run_year resets them anyway. Starting from stale counts
    # would put a spurious first-year burn on the record.
    v = st.vegetation
    st = eqx.tree_at(
        lambda s: (s.vegetation.nytimes, s.vegetation.nfirestep,
                   s.vegetation.sdry, s.vegetation.nfirestep_vpd,
                   s.vegetation.sdry_vpd),
        st, (jnp.zeros(()), jnp.zeros_like(v.nfirestep), jnp.zeros_like(v.sdry),
             jnp.zeros_like(v.nfirestep_vpd), jnp.zeros_like(v.sdry_vpd)))
    print(f"loaded {npoi} cells in {time.time()-t0:.0f}s", flush=True)

    cells = build_global_cell_grid(lat_step=0.5, lon_step=0.5)
    assert len(cells) == npoi, f"cell grid {len(cells)} != state {npoi}"
    lat = np.array([c[1] for c in cells], dtype=np.float64)
    lon = np.array([c[2] for c in cells], dtype=np.float64)
    idx = [latlon_to_grid_idx(la, lo) for la, lo in zip(lat, lon)]
    li = np.array([i[0] for i in idx])
    lo_i = np.array([i[1] for i in idx])

    params = build_sim_params(npoi, jnp.asarray(np.deg2rad(lat).astype(np.float32)), None)
    fire_params = FireParams()          # seasonlength + VPD + completeness
    print(f"fire: isimfire={a.isimfire} scheme={SEASONLENGTH} "
          f"vpd={fire_params.use_vpd_danger} "
          f"cc={fire_params.use_combustion_completeness}", flush=True)

    log = OUT / "spinup_log.csv"
    with open(log, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["step", "forcing_year", "biomass", "exist", "disturbf", "totalit",
             "totlit", "totcsoi", "plai", "elapsed_s"])

    t0 = time.time()
    step = {"n": 0}

    def _nan_report(s):
        """Which field, and how many cells -- one bad cell poisons every mean."""
        hits = []
        for grp in ("soil", "vegetation", "carbon", "snow", "hydrology"):
            o = getattr(s, grp, None)
            if o is None:
                continue
            for fd in type(o).__dataclass_fields__.values():
                a = getattr(o, fd.name, None)
                if a is None or not hasattr(a, "dtype"):
                    continue
                arr = np.asarray(a)
                if not np.issubdtype(arr.dtype, np.floating):
                    continue
                n = int(np.isnan(arr).sum())
                if n:
                    cells = np.unique(np.where(np.isnan(arr))[0])
                    hits.append((f"{grp}.{fd.name}", n, cells[:5].tolist()))
        return hits

    def on_year(year, s, key=None):
        step["n"] += 1
        n = step["n"]
        c, v = s.carbon, s.vegetation
        row = [n, year,
               float(jnp.mean(jnp.sum(v.biomass, axis=1))),
               float(jnp.mean(v.exist)), float(jnp.mean(v.disturbf)),
               float(jnp.mean(c.totalit)), float(jnp.mean(c.totlit)),
               float(jnp.mean(c.totcsoi)),
               float(jnp.mean(jnp.sum(v.plai, axis=1))),
               round(time.time() - t0)]
        with open(log, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        print(f"  [{n:3d}/{a.nyears}] y{year} biomass={row[2]:.3f} "
              f"exist={row[3]:.3f} disturbf={row[4]:.5f} totlit={row[6]:.3f} "
              f"totcsoi={row[7]:.3f} [{row[9]}s]", flush=True)
        if n % a.ckpt_every == 0:
            with open(OUT / "checkpoints" / f"state_step{n:03d}.pkl", "wb") as fp:
                pickle.dump({"step": n, "year": year, "jax_state": s,
                             "key": np.asarray(key) if key is not None else None}, fp)
        with open(OUT / "checkpoints" / "state_latest.pkl", "wb") as fp:
            pickle.dump({"step": n, "year": year, "jax_state": s,
                         "key": np.asarray(key) if key is not None else None}, fp)

        # A single NaN cell poisons every global mean, so the run looks dead
        # from the log while 88 483 cells are still fine. Name it the year it
        # appears, with the cell indices, so the next diagnosis starts from
        # fact rather than from a guess.
        if not np.isfinite(row[2]):
            for nm, cnt, cells in _nan_report(s)[:12]:
                print(f"    NaN {nm:30} {cnt:7d} value(s), cells {cells}",
                      flush=True)
            raise SystemExit(f"NaN at step {n} (forcing year {year})")

    y0, y1 = a.cycle
    span = y1 - y0 + 1
    years = [y0 + (i % span) for i in range(a.nyears)]
    print(f"running {a.nyears} years cycling {y0}-{y1}, "
          f"nsoilcall={a.nsoilcall}, ckpt every {a.ckpt_every}", flush=True)

    st, _ = run_spinup_gswp3(
        st, PhysicalConstants(), params, GSWP3Reader("/shared/nas2/Min/GSWP3"),
        li, lo_i, np.deg2rad(lat).astype(np.float32),
        jax.random.PRNGKey(0), 3600.0, years,
        nband=2, nsoilcall=a.nsoilcall, on_year_end=on_year,
        dynamic_vegetation=True, isimfire=a.isimfire,
        fire_scheme=SEASONLENGTH, fire_params=fire_params,
    )
    print(f"DONE {a.nyears} years in {(time.time()-t0)/60:.1f} min -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
