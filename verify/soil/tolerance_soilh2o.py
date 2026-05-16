"""Tolerance specs for soil soilh2o (implicit moisture solver + squeeze).

Expectations:

- wsoi (state output): Driven by a tridiagonal solve.  Both sides use
  identical algorithms (same RIMP, DMIN, same flux discretization).
  Differences should be tiny round-off from the order in which the
  tridiag is solved.  Set TIGHT, but allow PRECISION-class abs/rel if
  the solver's intermediate quantities (m, n exponents from nint(bex))
  introduce tiny offsets.

- wpud (state output): wpud_new = wpud + (fsqueez - fwpud) * dtime.
  Linear in fsqueez; inherits any wsoi-derived error.

- wflo (diagnostic, per layer boundary): computed from wsoi_new -
  owsoi.  Inherits wsoi error.

- fsqueez (diagnostic, per point): squeezed excess from saturating
  layers.  Often zero; when nonzero, depends on solver result.

In time-march mode we look for the FIRST STEP where the trajectories
diverge beyond tolerance, not just the worst-case point.
"""
from __future__ import annotations

from verify.framework.tolerance_classes import ToleranceSpec, TolClass


SOILH2O_SPECS = {
    "wsoi": ToleranceSpec(
        name="wsoi", tol_class=TolClass.PRECISION,
        abs_tol=1e-8, rel_tol=1e-7,
        bias_tol=1e-9,
        why="Tridiag solver with identical algorithm both sides. "
            "Round-off accumulates through ~6 layer ops per call.",
    ),
    "wpud": ToleranceSpec(
        name="wpud", tol_class=TolClass.PRECISION,
        abs_tol=1e-6, rel_tol=1e-7,
        bias_tol=1e-7,
        why="wpud_new = wpud + (fsqueez - fwpud) * dtime; inherits wsoi error.",
    ),
    "wflo": ToleranceSpec(
        name="wflo", tol_class=TolClass.PRECISION,
        abs_tol=1e-9, rel_tol=1e-6,
        bias_tol=1e-10,
        why="Diagnostic flux at layer boundaries; inherits wsoi error.",
    ),
    "fsqueez": ToleranceSpec(
        name="fsqueez", tol_class=TolClass.PRECISION,
        abs_tol=1e-7, rel_tol=1e-6,
        bias_tol=1e-9,
        why="Excess water squeezed from saturated layers; usually 0.",
    ),
}
