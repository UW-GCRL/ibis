"""Tolerance specs for soilheat and wadjust."""
from __future__ import annotations

from verify.framework.tolerance_classes import ToleranceSpec, TolClass


SOILHEAT_SPECS = {
    "tsoi": ToleranceSpec(
        name="tsoi", tol_class=TolClass.PRECISION,
        abs_tol=1e-9, rel_tol=1e-9,
        bias_tol=1e-9,
        why="Tridiag solve with identical algorithm both sides; round-off "
            "from ~6 layer ops per call.",
    ),
    "hflo": ToleranceSpec(
        name="hflo", tol_class=TolClass.PRECISION,
        abs_tol=1e-5, rel_tol=1e-9,
        bias_tol=1e-6,
        why="Cumulative downward heat flux; inherits tsoi error, but the "
            "(c1*tsoi - c0*otsoi)/dtime accumulation can amplify slightly.",
    ),
}


WADJUST_SPECS = {
    "wsoi": ToleranceSpec(
        name="wsoi", tol_class=TolClass.BIT_EXACT,
        abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
        why="Clamp/conditional only; no arithmetic that varies between sides.",
    ),
    "wisoi": ToleranceSpec(
        name="wisoi", tol_class=TolClass.BIT_EXACT,
        abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
        why="Clamp/conditional only.",
    ),
    "gadjust": ToleranceSpec(
        name="gadjust", tol_class=TolClass.PRECISION,
        abs_tol=1e-10, rel_tol=1e-9,
        bias_tol=1e-11,
        why="Delta of (1-wisoi)*wsoi+wisoi accumulated across layers; "
            "differences from FP order in summation.",
    ),
}
