"""Tolerance specs for snow_physics (snowheat, vadapt, snow_melt, snow_coverage)."""
from __future__ import annotations

from verify.framework.tolerance_classes import ToleranceSpec, TolClass


SNOWHEAT_SPECS = {
    "tsno":  ToleranceSpec(name="tsno",  tol_class=TolClass.PRECISION,
                            abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
                            why="Tridiag solve with identical algorithm both sides."),
    "tlsub": ToleranceSpec(name="tlsub", tol_class=TolClass.PRECISION,
                            abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
                            why="Buried-veg temp updated from tridia."),
    "sflo":  ToleranceSpec(name="sflo",  tol_class=TolClass.PRECISION,
                            abs_tol=1e-6, rel_tol=1e-9, bias_tol=1e-7,
                            why="Cumulative downward heat flux profile."),
}


VADAPT_SPECS = {
    "hcur": ToleranceSpec(name="hcur", tol_class=TolClass.PRECISION,
                           abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                           why="New thicknesses; closed-form."),
    "tcur": ToleranceSpec(name="tcur", tol_class=TolClass.PRECISION,
                           abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
                           why="Re-binned tracer; sums of products differ in last bit."),
}


SNOW_MELT_SPECS = {
    "hsno":  ToleranceSpec(name="hsno",  tol_class=TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                            why="hsno -= xm/rhos; same algebra both sides."),
    "tsno":  ToleranceSpec(name="tsno",  tol_class=TolClass.BIT_EXACT,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                            why="min(tsno, tmelt); pure clamp."),
    "zmelt": ToleranceSpec(name="zmelt", tol_class=TolClass.PRECISION,
                            abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
                            why="Sum over layers of fi*xm."),
    "zheat": ToleranceSpec(name="zheat", tol_class=TolClass.PRECISION,
                            abs_tol=1e-3, rel_tol=1e-11, bias_tol=1e-4,
                            why="Sum over layers of fi*(xh - hfus*xm); large magnitudes (~hfus*kg/m2)."),
}


SNOW_COVERAGE_SPECS = {
    "fi":   ToleranceSpec(name="fi",   tol_class=TolClass.PRECISION,
                           abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                           why="finew from sqrt closed-form."),
    "hsno": ToleranceSpec(name="hsno", tol_class=TolClass.PRECISION,
                           abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                           why="hsno *= fi_old/max(finew, epsilon)."),
}
