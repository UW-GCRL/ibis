"""Verify JAX climate (climanl/climanl2/existence) against F77.

These are all deterministic, no RNG, pure arithmetic + threshold logic.
Expected: bit-exact.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.climate.verify_climate
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
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "climate"),
)
import climate_wrap as f77

from ibis.io.parameters import default_pft_params
from ibis.physics.climate_analysis import (
    climanl as jax_climanl,
    climanl2 as jax_climanl2,
    existence as jax_existence,
)

from verify.framework.harness import Report, verify_variable
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


NDAYPM = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# Default PFT params (read from JAX defaults; same as F77 params.veg).
PFT = default_pft_params()
TminL = np.asarray(PFT.TminL, dtype=np.float64)
TminU = np.asarray(PFT.TminU, dtype=np.float64)
Twarm = np.asarray(PFT.Twarm, dtype=np.float64)
GDD   = np.asarray(PFT.GDD,   dtype=np.float64)


SPEC_BIT = ToleranceSpec("scalar", TolClass.BIT_EXACT,
                         why="Pure arithmetic / threshold; expect 0 ULP.")
SPEC_PREC = ToleranceSpec("scalar", TolClass.PRECISION,
                          abs_tol=1e-12, rel_tol=1e-14, bias_tol=1e-13)


def test_climanl(report: Report, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    xint   = rng.uniform(-30.0, 35.0, (n, 12))   # monthly mean T (C)
    deltat = rng.uniform(-5.0,  5.0,  n)

    tc_f, tw_f, tcmin_f, gdd0_f, gdd5_f, exist_f = f77.wrap_climanl(
        xint, deltat, TminL, TminU, Twarm, GDD,
    )

    tc_j, tw_j, tcmin_j, gdd0_j, gdd5_j = jax_climanl(
        jnp.asarray(xint), jnp.asarray(deltat), NDAYPM,
    )
    exist_j = jax_existence(
        jnp.asarray(np.asarray(tcmin_j)), jnp.asarray(np.asarray(tw_j)),
        jnp.asarray(np.asarray(gdd0_j)),  jnp.asarray(np.asarray(gdd5_j)),
        PFT,
    )

    pairs = [
        ("tc",    tc_f,    np.asarray(tc_j)),
        ("tw",    tw_f,    np.asarray(tw_j)),
        ("tcmin", tcmin_f, np.asarray(tcmin_j)),
        ("gdd0",  gdd0_f,  np.asarray(gdd0_j)),
        ("gdd5",  gdd5_f,  np.asarray(gdd5_j)),
        ("exist", exist_f, np.asarray(exist_j)),
    ]
    for name, fv, jv in pairs:
        spec = SPEC_BIT if name in ("tc", "tw", "exist") else SPEC_PREC
        report.add(verify_variable(
            f"climanl.{name}", np.asarray(fv), np.asarray(jv), spec,
            abs_floor=1e-6,
        ))


def test_climanl2(report: Report, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tc = rng.uniform(-30.0, 30.0, n)
    tw = rng.uniform(0.0,   40.0, n)
    tcmin = tc - 5.0
    gdd0 = rng.uniform(0.0, 3000.0, n)
    gdd5 = rng.uniform(0.0, 2500.0, n)
    tcthis = rng.uniform(-30.0, 30.0, n)
    twthis = rng.uniform(0.0,   40.0, n)
    gdd0this = rng.uniform(0.0, 3000.0, n)
    gdd5this = rng.uniform(0.0, 2500.0, n)
    deltat = rng.uniform(-5.0, 5.0, n)

    tc_f, tw_f, tcmin_f, gdd0_f, gdd5_f, exist_f = f77.wrap_climanl2(
        tc, tw, tcmin, gdd0, gdd5,
        tcthis, twthis, gdd0this, gdd5this, deltat,
        TminL, TminU, Twarm, GDD,
    )

    tc_j, tw_j, tcmin_j, gdd0_j, gdd5_j = jax_climanl2(
        jnp.asarray(tc), jnp.asarray(tw), jnp.asarray(tcmin),
        jnp.asarray(gdd0), jnp.asarray(gdd5),
        jnp.asarray(tcthis), jnp.asarray(twthis),
        jnp.asarray(gdd0this), jnp.asarray(gdd5this),
        jnp.asarray(deltat),
    )
    exist_j = jax_existence(
        jnp.asarray(np.asarray(tcmin_j)), jnp.asarray(np.asarray(tw_j)),
        jnp.asarray(np.asarray(gdd0_j)),  jnp.asarray(np.asarray(gdd5_j)),
        PFT,
    )

    pairs = [
        ("tc",    tc_f,    np.asarray(tc_j)),
        ("tw",    tw_f,    np.asarray(tw_j)),
        ("tcmin", tcmin_f, np.asarray(tcmin_j)),
        ("gdd0",  gdd0_f,  np.asarray(gdd0_j)),
        ("gdd5",  gdd5_f,  np.asarray(gdd5_j)),
        ("exist", exist_f, np.asarray(exist_j)),
    ]
    for name, fv, jv in pairs:
        spec = SPEC_BIT if name == "exist" else SPEC_PREC
        report.add(verify_variable(
            f"climanl2.{name}", np.asarray(fv), np.asarray(jv), spec,
            abs_floor=1e-6,
        ))


def test_existence(report: Report, n: int, seed: int) -> None:
    """Standalone existence: stress the threshold logic at edges."""
    rng = np.random.default_rng(seed)
    # Include values straddling each threshold.
    tcmin = rng.uniform(-60.0, 30.0, n)
    tw    = rng.uniform(0.0,   40.0, n)
    gdd0  = rng.uniform(0.0,   3000.0, n)
    gdd5  = rng.uniform(0.0,   2500.0, n)

    exist_f = np.asarray(f77.wrap_existence(
        tcmin, tw, gdd0, gdd5, TminL, TminU, Twarm, GDD,
    ))
    exist_j = np.asarray(jax_existence(
        jnp.asarray(tcmin), jnp.asarray(tw),
        jnp.asarray(gdd0), jnp.asarray(gdd5), PFT,
    ))
    report.add(verify_variable("existence.exist", exist_f, exist_j, SPEC_BIT,
                               abs_floor=1e-6))


def main() -> int:
    rep = Report(section="climate")
    test_climanl  (rep, n=500, seed=42)
    test_climanl  (rep, n=500, seed=123)
    test_climanl2 (rep, n=500, seed=99)
    test_climanl2 (rep, n=500, seed=2026)
    test_existence(rep, n=1500, seed=7)
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
