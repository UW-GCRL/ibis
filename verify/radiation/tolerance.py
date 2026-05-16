"""Tolerance specs for the radiation section (twoset, twostr).

With the JAX `twoset` monkey-patched to use the F77 tablemu lookup
(see verify_radiation.py / verify_radiation_full.py), the tablemu
quantization difference goes away and all outputs are at machine
precision.
"""
from __future__ import annotations

from verify.framework.tolerance_classes import ToleranceSpec, TolClass


# ----- twoset ---------------------------------------------------------
TWOSET_SPECS = {
    "omega": ToleranceSpec(
        name="omega", tol_class=TolClass.PRECISION,
        abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14,
        why="(1-y)*(rhoveg+tauveg) + y*omegasno; no tablemu dependency.",
    ),
    "betad": ToleranceSpec(
        name="betad", tol_class=TolClass.PRECISION,
        abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14,
        why="JAX twoset patched to use F77 tablemu; bit-exact.",
    ),
    "betai": ToleranceSpec(
        name="betai", tol_class=TolClass.PRECISION,
        abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14,
        why="Fixed 2/3, 1/3 weights; no tablemu involvement.",
    ),
    "avmu": ToleranceSpec(
        name="avmu", tol_class=TolClass.PRECISION,
        abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14,
        why="Set to 1.0 identically; bit-exact.",
    ),
    "gdir": ToleranceSpec(
        name="gdir", tol_class=TolClass.PRECISION,
        abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14,
        why="Closed-form in coszen, oriev, orieh; identical formula.",
    ),
}


# ----- twostr ---------------------------------------------------------
TWOSTR_SPECS = {
    "abvegd": ToleranceSpec("abvegd", TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13),
    "abvegi": ToleranceSpec("abvegi", TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13),
    "refld":  ToleranceSpec("refld",  TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13),
    "refli":  ToleranceSpec("refli",  TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13),
    "fbeldd": ToleranceSpec("fbeldd", TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13),
    "fbeldi": ToleranceSpec("fbeldi", TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                            why="Identically zero."),
    "fbelid": ToleranceSpec("fbelid", TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13),
    "fbelii": ToleranceSpec("fbelii", TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13),
}
