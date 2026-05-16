"""Tolerance specs for the thermo section.

Each variable's class is chosen with a reason. The Lowe (1977)
polynomial is identical in F77 and JAX, so we expect TIGHT agreement at
float64. hvapf / hsubf are linear combinations of constants and inputs,
so they should be effectively BIT_EXACT.

bias_tol catches systematic sign or scale errors that don't show up in
max-element comparisons.
"""
from __future__ import annotations

from verify.framework.tolerance_classes import ToleranceSpec, TolClass


SPECS = {
    "esat": ToleranceSpec(
        name="esat", tol_class=TolClass.TIGHT,
        bias_tol=1e-9,
        why="Same Lowe(1977) 6th-order polynomial in both. At float64, "
            "compiler reordering of FMA gives diffs around the last few ULPs.",
    ),
    "desat": ToleranceSpec(
        name="desat", tol_class=TolClass.TIGHT,
        bias_tol=1e-9,
        why="Same as esat: Lowe(1977) d(SVP)/dT polynomial, fixed coefficients.",
    ),
    "qsat": ToleranceSpec(
        name="qsat", tol_class=TolClass.TIGHT,
        bias_tol=1e-12,
        why="0.622 * e / max(p - 0.378*e, 0.622*e) -- simple ratio, should "
            "be near bit-exact.",
    ),
    "dqsat": ToleranceSpec(
        name="dqsat", tol_class=TolClass.TIGHT,
        bias_tol=1e-12,
        why="desat * q1 * (1 + q1*(1/0.622 - 1)) / esat -- multiplicative "
            "chain; expect TIGHT.",
    ),
    "hvapf": ToleranceSpec(
        name="hvapf", tol_class=TolClass.BIT_EXACT,
        abs_tol=1e-6,            # 1e-6 J/kg on values ~2.5e6 J/kg is ~4e-13 relative
        rel_tol=1e-13,
        bias_tol=1e-7,
        why="hvap + cvap*(tair-tmelt) - ch2o*(t-tmelt) -- one mul-add chain. "
            "Both sides should be bit-equivalent (within ~1 ULP of float64).",
    ),
    "hsubf": ToleranceSpec(
        name="hsubf", tol_class=TolClass.BIT_EXACT,
        abs_tol=1e-6,
        rel_tol=1e-13,
        bias_tol=1e-7,
        why="hsub + cvap*(tair-tmelt) - cice*(t-tmelt) -- same shape as hvapf.",
    ),
}
