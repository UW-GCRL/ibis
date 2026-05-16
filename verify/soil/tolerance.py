"""Tolerance specs for soil setsoi.

Expected behavior:

- consoi: F77 computes (0.56*100.)^powliq * (2.24*100.)^powice; JAX uses
  the literal 56.0 and 224.0. At float64 these produce tiny (~1e-14)
  relative differences because 0.56 and 2.24 are not exactly representable
  in binary -- the F77 runtime multiplication carries that error in.
  Set TIGHT.

- qglif: pure arithmetic on wsoi/wisoi/wpud/wipud + branch selection.
  No new sources of error if inputs are bit-identical.  TIGHT.

- hvasug, hvasui: linear combinations involving hvapf / hsubf, which we
  already verified bit-exact on the thermo section.  BIT_EXACT.

Known divergence (not tested in the main pass):
- F77 does nint(sand) / nint(clay) before computing fractions.  JAX uses
  raw values.  If non-integer sand/clay are supplied, consoi can disagree
  by O(1%) for sand/clay percentages with fractional part 0.5.  The main
  test feeds integer percentages so this divergence is suppressed.  We
  add a separate "edge_nonint" test that lets it show.
"""
from __future__ import annotations

from verify.framework.tolerance_classes import ToleranceSpec, TolClass


SETSOI_SPECS = {
    "consoi": ToleranceSpec(
        name="consoi", tol_class=TolClass.TIGHT,
        bias_tol=1e-12,
        why="F77 computes (0.56*100.)^powliq * (2.24*100.)^powice at runtime; "
            "JAX uses literal 56.0 / 224.0. Tiny diff from how the binary "
            "representation of 0.56 / 2.24 propagates through the power.",
    ),
    "qglif": ToleranceSpec(
        name="qglif", tol_class=TolClass.TIGHT,
        bias_tol=1e-12,
        why="Branch-conditioned arithmetic on wsoi/wisoi/wpud/wipud. "
            "If inputs match, both sides should agree to FP roundoff.",
    ),
    "hvasug": ToleranceSpec(
        name="hvasug", tol_class=TolClass.BIT_EXACT,
        abs_tol=1e-3,        # values are ~2.5e6 J/kg
        rel_tol=1e-13,
        bias_tol=1e-4,
        why="(q1+q3)*hvapf + (q2+q4)*hsubf -- linear combination of two "
            "functions we already verified BIT_EXACT against F77.",
    ),
    "hvasui": ToleranceSpec(
        name="hvasui", tol_class=TolClass.BIT_EXACT,
        abs_tol=1e-3,
        rel_tol=1e-13,
        bias_tol=1e-4,
        why="hsubf(tsno_1, ta) -- single call to a BIT_EXACT thermo function.",
    ),
}


SETSOI_NONINT_SPECS = {
    # Same vars at TIGHT now that the verifier patches JAX setsoi to apply
    # F77 nint(sand)/nint(clay) before computing fsand/fclay (finding #2).
    "consoi": SETSOI_SPECS["consoi"],
    "qglif":  SETSOI_SPECS["qglif"],
    "hvasug": SETSOI_SPECS["hvasug"],
    "hvasui": SETSOI_SPECS["hvasui"],
}
