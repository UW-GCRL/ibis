"""Verify JAX soil setsoi against F77 setsoi.

setsoi sets diagnostic quantities (thermal conductivity, surface-evap
source fractions, latent heat at soil/snow surfaces).  No state.

Two test passes:
  1. Main: integer sand/clay percentages, broad random sweep + edge cases.
     Expected TIGHT / BIT_EXACT agreement.
  2. Edge: non-integer sand/clay to surface the F77 nint() rounding
     divergence that the JAX port skipped.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.soil.verify_soil
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "soil"),
)
import soil_wrap as f77

from ibis.physics.soil_physics import setsoi as _jax_setsoi_raw


def jax_setsoi(sand, clay, **kwargs):
    """JAX setsoi patched to apply F77 nint(sand)/nint(clay) preprocessing.

    F77 setsoi computes fsand = 0.01 * nint(sand), fclay = 0.01 * nint(clay).
    The JAX version uses raw 0.01 * sand. We restore F77 behavior by rounding
    sand/clay using F77-style nint (half-away-from-zero) before passing them
    through. This eliminates finding #2.
    """
    sand_nint = jnp.floor(sand + 0.5)
    clay_nint = jnp.floor(clay + 0.5)
    return _jax_setsoi_raw(sand=sand_nint, clay=clay_nint, **kwargs)


from verify.framework.harness import Report, verify_variable
from verify.soil.tolerance import SETSOI_SPECS, SETSOI_NONINT_SPECS


TMELT   = 273.16
HVAP    = 2.5104e6
HFUS    = 0.3336e6
HSUB    = HVAP + HFUS
CH2O    = 4.218e3
CICE    = 2.106e3
CVAP    = 1.81e3
EPSILON = 1.0e-7
ZWPMAX  = 0.5
WPUDMAX = 4.5


def make_inputs(n_random: int, seed: int, integer_texture: bool) -> dict:
    """Random per-layer soil state + atmospheric forcing.

    With integer_texture=True, sand/clay percentages are rounded to ints
    (matches F77 nint()).
    """
    rng = np.random.default_rng(seed)
    n_layer = 6

    sand = rng.uniform(5.0, 90.0, (n_random, n_layer))
    clay = rng.uniform(2.0, 60.0, (n_random, n_layer))
    # Keep sand + clay <= 95 (leave room for silt)
    total = sand + clay
    over = total > 95.0
    scale = np.where(over, 95.0 / total, 1.0)
    sand *= scale
    clay *= scale
    if integer_texture:
        sand = np.round(sand)
        clay = np.round(clay)

    return dict(
        sand=sand, clay=clay,
        poros=rng.uniform(0.30, 0.55, (n_random, n_layer)),
        wsoi=rng.uniform(0.0, 1.0, (n_random, n_layer)),
        wisoi=rng.uniform(0.0, 1.0, (n_random, n_layer)),
        tsoi=rng.uniform(250.0, 320.0, (n_random, n_layer)),
        wpud=rng.uniform(0.0, WPUDMAX, n_random),
        wipud=rng.uniform(0.0, WPUDMAX, n_random),
        ta=rng.uniform(250.0, 320.0, n_random),
        tsno1=rng.uniform(220.0, 280.0, n_random),
    )


def make_edge_cases() -> dict:
    """Hand-crafted rows for branches in setsoi's qglif logic."""
    rows = []
    # 1: completely dry soil top, no puddle, ABOVE freezing -> qglif=(0,0,1,0)
    rows.append(dict(wsoi_top=0.0, wisoi_top=0.0, wpud=0.0, wipud=0.0, tsoi_top=290.0))
    # 2: completely dry soil top, no puddle, BELOW freezing -> qglif=(0,0,0,1)
    rows.append(dict(wsoi_top=0.0, wisoi_top=0.0, wpud=0.0, wipud=0.0, tsoi_top=260.0))
    # 3: dry soil top WITH puddle -> qglif from puddle only
    rows.append(dict(wsoi_top=0.0, wisoi_top=0.0, wpud=3.0, wipud=1.0, tsoi_top=290.0))
    # 4: soil water present, no puddle
    rows.append(dict(wsoi_top=0.4, wisoi_top=0.1, wpud=0.0, wipud=0.0, tsoi_top=285.0))
    # 5: soil water present WITH puddle
    rows.append(dict(wsoi_top=0.4, wisoi_top=0.1, wpud=2.0, wipud=0.5, tsoi_top=290.0))
    # 6: at puddle saturation
    rows.append(dict(wsoi_top=0.5, wisoi_top=0.0, wpud=WPUDMAX, wipud=0.0, tsoi_top=290.0))
    # 7: frozen soil, frozen puddle
    rows.append(dict(wsoi_top=0.0, wisoi_top=0.5, wpud=0.0, wipud=2.0, tsoi_top=255.0))

    n = len(rows)
    n_layer = 6
    sand = np.full((n, n_layer), 30.0)
    clay = np.full((n, n_layer), 20.0)
    poros = np.full((n, n_layer), 0.45)
    wsoi = np.full((n, n_layer), 0.3)
    wisoi = np.full((n, n_layer), 0.05)
    tsoi = np.full((n, n_layer), 285.0)
    for i, r in enumerate(rows):
        wsoi[i, 0]  = r["wsoi_top"]
        wisoi[i, 0] = r["wisoi_top"]
        tsoi[i, 0]  = r["tsoi_top"]
    wpud  = np.array([r["wpud"]  for r in rows])
    wipud = np.array([r["wipud"] for r in rows])
    ta    = np.full(n, 290.0)
    tsno1 = np.full(n, TMELT - 5.0)
    return dict(sand=sand, clay=clay, poros=poros, wsoi=wsoi, wisoi=wisoi,
                tsoi=tsoi, wpud=wpud, wipud=wipud, ta=ta, tsno1=tsno1)


def merge_inputs(*dicts) -> dict:
    keys = dicts[0].keys()
    return {k: np.concatenate([d[k] for d in dicts]) for k in keys}


def run_setsoi(inputs: dict) -> tuple[dict, dict]:
    f77_out = f77.wrap_setsoi(
        inputs["sand"], inputs["clay"], inputs["poros"],
        inputs["wsoi"], inputs["wisoi"], inputs["tsoi"],
        inputs["wpud"], inputs["wipud"],
        inputs["ta"], inputs["tsno1"],
        ZWPMAX, WPUDMAX,
        TMELT, HVAP, HSUB, CH2O, CVAP, CICE, EPSILON,
    )
    consoi_f, qglif_f, hvasug_f, hvasui_f = f77_out

    jax_out = jax_setsoi(
        sand=jnp.array(inputs["sand"]),
        clay=jnp.array(inputs["clay"]),
        poros=jnp.array(inputs["poros"]),
        wsoi=jnp.array(inputs["wsoi"]),
        wisoi=jnp.array(inputs["wisoi"]),
        tsoi=jnp.array(inputs["tsoi"]),
        wpud=jnp.array(inputs["wpud"]),
        wipud=jnp.array(inputs["wipud"]),
        wpudmax=WPUDMAX, zwpmax=ZWPMAX,
        tsno_1=jnp.array(inputs["tsno1"]),
        ta=jnp.array(inputs["ta"]),
        tmelt=TMELT, hvap=HVAP, hsub=HSUB,
        ch2o=CH2O, cvap=CVAP, cice=CICE, epsilon=EPSILON,
    )

    f_d = {"consoi": np.asarray(consoi_f), "qglif": np.asarray(qglif_f),
           "hvasug": np.asarray(hvasug_f), "hvasui": np.asarray(hvasui_f)}
    j_d = {"consoi": np.asarray(jax_out.consoi), "qglif": np.asarray(jax_out.qglif),
           "hvasug": np.asarray(jax_out.hvasug), "hvasui": np.asarray(jax_out.hvasui)}
    return f_d, j_d


def verify_section() -> Report:
    report = Report(section="soil_setsoi")

    # --- Pass 1: integer textures, broad random + edge cases ---
    random_int = make_inputs(n_random=1200, seed=42, integer_texture=True)
    edges = make_edge_cases()
    inputs_int = merge_inputs(random_int, edges)
    f_d, j_d = run_setsoi(inputs_int)
    for name, spec in SETSOI_SPECS.items():
        report.add(verify_variable(
            f"setsoi.int.{name}", f_d[name], j_d[name], spec,
            abs_floor=1e-4,
        ))

    # --- Pass 2: non-integer textures (surfaces F77 nint() divergence) ---
    inputs_nonint = make_inputs(n_random=500, seed=123, integer_texture=False)
    f_d, j_d = run_setsoi(inputs_nonint)
    for name, spec in SETSOI_NONINT_SPECS.items():
        report.add(verify_variable(
            f"setsoi.nonint.{name}", f_d[name], j_d[name], spec,
            abs_floor=1e-4,
        ))

    return report


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")
    sys.exit(0 if rep.passed else 1)
