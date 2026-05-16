"""Tolerance specs for freeze_thaw and puddle_freeze_thaw."""
from __future__ import annotations

from verify.framework.tolerance_classes import ToleranceSpec, TolClass


FREEZE_THAW_SPECS = {
    "tsoi": ToleranceSpec(
        name="tsoi", tol_class=TolClass.PRECISION,
        abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
        why="Heat-conserving phase change with the same algebra both sides; "
            "tiny FP roundoff possible from re-computing chav.",
    ),
    "wsoi": ToleranceSpec(
        name="wsoi", tol_class=TolClass.PRECISION,
        abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
        why="Conserves mass through dw/zporos increments; round-off only.",
    ),
    "wisoi": ToleranceSpec(
        name="wisoi", tol_class=TolClass.PRECISION,
        abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
        why="Same as wsoi.",
    ),
    "porosflo": ToleranceSpec(
        name="porosflo", tol_class=TolClass.PRECISION,
        abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14,
        why="Verifier patches JAX porosflo to use zporos = max(poros, eps), "
            "matching F77. Finding #3 was eliminated this way.",
    ),
}


PUDDLE_FT_SPECS = {
    "tsoi_0": ToleranceSpec(
        name="tsoi_0", tol_class=TolClass.PRECISION,
        abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
        why="Top-layer temp after puddle phase-change; same algebra both sides.",
    ),
    "wpud": ToleranceSpec(
        name="wpud", tol_class=TolClass.PRECISION,
        abs_tol=1e-10, rel_tol=1e-11, bias_tol=1e-11,
        why="wpud +/- dw; dw = min(wpud, max(0, dh/hfus)).",
    ),
    "wipud": ToleranceSpec(
        name="wipud", tol_class=TolClass.PRECISION,
        abs_tol=1e-10, rel_tol=1e-11, bias_tol=1e-11,
        why="wipud +/- dw mirror of wpud.",
    ),
}
