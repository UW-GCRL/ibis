"""Per-variable tolerance classes for JAX <-> Fortran verification.

A tolerance class encodes the *expected* precision of agreement between
the Python port and the Fortran reference for a given output variable.
Bugs that move a variable across classes are caught immediately.

Classes (loosest at bottom, tightest at top):

    BIT_EXACT   -- same algorithm, same dtype, same evaluation order.
                   Should match to ~1 ULP. Example: unit conversions,
                   pure data shuffling.
    TIGHT       -- pure math from identical formulas. Tiny differences
                   from compiler reordering of FP ops. Example: Lowe(1977)
                   polynomial evaluation at float64.
    PRECISION   -- matched-iteration solvers, same convergence. Example:
                   Newton iterations with identical inputs and tolerance.
    LOOSE       -- nonlinear physics that may iterate to slightly different
                   solutions because of compiler/algorithm differences.
                   Example: implicit canopy energy balance.
    SANITY      -- stochastic or chaotic; only order-of-magnitude agreement
                   makes sense. Example: stochastic weather generation.

Choosing too-loose lets bugs hide. Choosing too-tight produces false
failures from honest floating-point reordering. Document the *reason*
for each choice next to the variable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TolClass(Enum):
    BIT_EXACT = "bit_exact"
    TIGHT     = "tight"
    PRECISION = "precision"
    LOOSE     = "loose"
    SANITY    = "sanity"


# Default thresholds. (max_abs, max_rel) — a variable PASSES if either
# max_abs <= abs_tol OR max_rel <= rel_tol (matching Min's style so
# near-zero values are not penalized by relative error).
_DEFAULTS = {
    TolClass.BIT_EXACT: (1e-15, 1e-15),  # ~1 ULP at float64
    TolClass.TIGHT:     (1e-12, 1e-12),
    TolClass.PRECISION: (1e-8,  1e-6),
    TolClass.LOOSE:     (1e-3,  1e-3),
    TolClass.SANITY:    (None,  0.5),    # 50% relative, no absolute floor
}


@dataclass(frozen=True)
class ToleranceSpec:
    """Tolerance for a single output variable.

    abs_tol / rel_tol override the class defaults if non-None.
    bias_tol (mean signed error) is an additional invariant: a bias
    larger than this fails regardless of max error, because systematic
    offset usually indicates a real bug (whereas scatter can be honest
    FP roundoff).
    """
    name: str
    tol_class: TolClass
    abs_tol: float | None = None
    rel_tol: float | None = None
    bias_tol: float | None = None
    why: str = ""

    @property
    def effective_abs(self) -> float | None:
        if self.abs_tol is not None:
            return self.abs_tol
        return _DEFAULTS[self.tol_class][0]

    @property
    def effective_rel(self) -> float | None:
        if self.rel_tol is not None:
            return self.rel_tol
        return _DEFAULTS[self.tol_class][1]
