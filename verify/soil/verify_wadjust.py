"""Verify JAX wadjust against F77 wadjust.

wadjust is a state clamp/bound enforcer with no time evolution per se,
so we only do single-call testing across a broad set of edge cases:
  - small +ve wsoi / wisoi (below epsilon -> should clamp to 0)
  - small -ve wsoi / wisoi (round-off negatives -> should clamp to 0)
  - wsoi > 1 (oversaturated -> should clamp to 1)
  - wisoi near 1 (frozen full -> wsoi should be forced to 0)
  - random valid inputs

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.soil.verify_wadjust
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

from ibis.physics.soil_physics import wadjust as jax_wadjust

from verify.framework.harness import Report, verify_variable
from verify.soil.tolerance_soilheat import WADJUST_SPECS


RHOW    = 1000.0
DTIME   = 3600.0
EPSILON = 1.0e-7
HSOI    = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00], dtype=np.float64)


def make_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nl = 6
    # Mix of valid, slightly-negative, oversaturated, and frozen-full.
    wsoi = rng.uniform(-0.05, 1.1, (n, nl))
    wisoi = rng.uniform(-0.05, 1.05, (n, nl))
    poros = rng.uniform(0.30, 0.55, (n, nl))
    return dict(wsoi=wsoi, wisoi=wisoi, poros=poros)


def verify_section() -> Report:
    report = Report(section="wadjust")
    inputs = make_inputs(n=1500, seed=42)

    wsoi_f, wisoi_f, gadj_f = f77.wrap_wadjust(
        inputs["wsoi"], inputs["wisoi"], inputs["poros"], HSOI,
        RHOW, DTIME, EPSILON,
    )

    jax_out = jax_wadjust(
        wsoi=jnp.array(inputs["wsoi"]),
        wisoi=jnp.array(inputs["wisoi"]),
        poros=jnp.array(inputs["poros"]),
        hsoi=jnp.array(HSOI),
        rhow=RHOW, dtime=DTIME, epsilon=EPSILON,
    )
    # JAX wadjust returns a tuple (wsoi_new, wisoi_new, gadjust).
    wsoi_j, wisoi_j, gadj_j = jax_out

    report.add(verify_variable("wadjust.wsoi",    np.asarray(wsoi_f),
                                np.asarray(wsoi_j),  WADJUST_SPECS["wsoi"],
                                abs_floor=1e-6))
    report.add(verify_variable("wadjust.wisoi",   np.asarray(wisoi_f),
                                np.asarray(wisoi_j), WADJUST_SPECS["wisoi"],
                                abs_floor=1e-6))
    report.add(verify_variable("wadjust.gadjust", np.asarray(gadj_f),
                                np.asarray(gadj_j),  WADJUST_SPECS["gadjust"],
                                abs_floor=1e-8))
    return report


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")
    sys.exit(0 if rep.passed else 1)
