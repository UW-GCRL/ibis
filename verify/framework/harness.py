"""Verification harness.

The single entry point is `verify_variable(name, ref, cand, spec)`:
- computes ResidualStats
- evaluates PASS/FAIL against the ToleranceSpec
- returns a Row(spec, stats, status, fail_reasons)

`Report` aggregates rows from a whole section (e.g. thermo), prints a
table, exports the residual rows to parquet, and reports section-level
PASS/FAIL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import ResidualStats, compute
from .tolerance_classes import ToleranceSpec


@dataclass(frozen=True)
class Row:
    spec:     ToleranceSpec
    stats:    ResidualStats
    status:   str          # "PASS" | "FAIL"
    reasons:  tuple[str, ...]  # non-empty when status == "FAIL"


def verify_variable(name: str, ref: np.ndarray, cand: np.ndarray,
                    spec: ToleranceSpec, *, abs_floor: float = 1.0) -> Row:
    """Compare a single variable and judge against its tolerance spec."""
    stats = compute(ref, cand, abs_floor=abs_floor)
    reasons: list[str] = []

    if stats.n == 0:
        reasons.append("no finite elements to compare")
    else:
        # Pass condition: max_abs <= abs_tol OR max_rel <= rel_tol.
        # This is permissive on near-zero values (where max_rel is huge
        # but max_abs is tiny) without losing the relative check elsewhere.
        ok_abs = (spec.effective_abs is not None
                  and stats.max_abs <= spec.effective_abs)
        ok_rel = (spec.effective_rel is not None
                  and stats.max_rel <= spec.effective_rel)
        if not (ok_abs or ok_rel):
            reasons.append(
                f"max_abs={stats.max_abs:.3e} > {spec.effective_abs} "
                f"AND max_rel={stats.max_rel:.3e} > {spec.effective_rel}"
            )
        if spec.bias_tol is not None and abs(stats.bias) > spec.bias_tol:
            reasons.append(
                f"|bias|={abs(stats.bias):.3e} > {spec.bias_tol}"
            )
        if stats.n_nan_cand > stats.n_nan_ref:
            reasons.append(
                f"new NaNs introduced: ref has {stats.n_nan_ref}, "
                f"cand has {stats.n_nan_cand}"
            )

    status = "PASS" if not reasons else "FAIL"
    return Row(spec=spec, stats=stats, status=status,
               reasons=tuple(reasons))


@dataclass
class Report:
    section: str
    rows:    list[Row] = field(default_factory=list)
    residuals: list[dict] = field(default_factory=list)  # raw stats rows

    def add(self, row: Row) -> None:
        self.rows.append(row)
        self.residuals.append({
            "variable": row.spec.name,
            "tol_class": row.spec.tol_class.value,
            "status": row.status,
            **row.stats.as_dict(),
            "reasons": "; ".join(row.reasons),
        })

    @property
    def passed(self) -> bool:
        return all(r.status == "PASS" for r in self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.residuals)

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.section}_residuals.parquet"
        df = self.to_dataframe()
        try:
            df.to_parquet(path)
        except Exception:
            # No pyarrow / fastparquet; fall back to CSV.
            path = out_dir / f"{self.section}_residuals.csv"
            df.to_csv(path, index=False)
        return path

    def print_table(self) -> None:
        head = (f"{'STATUS':<6} {'VARIABLE':<14} {'CLASS':<10} "
                f"{'N':>6} {'MAX_ABS':>11} {'MAX_REL':>11} "
                f"{'RMSE':>11} {'BIAS':>12} {'MAX_ULP':>10}  REASON")
        print(head)
        print("-" * len(head))
        for row in self.rows:
            s = row.stats
            print(
                f"{row.status:<6} {row.spec.name:<14} "
                f"{row.spec.tol_class.value:<10} "
                f"{s.n:>6} "
                f"{s.max_abs:>11.3e} {s.max_rel:>11.3e} "
                f"{s.rmse:>11.3e} {s.bias:>+12.3e} "
                f"{s.max_ulp:>10.2f}  "
                f"{'; '.join(row.reasons)}"
            )
        print("-" * len(head))
        verdict = "ALL PASS" if self.passed else "FAILURES"
        print(f"RESULT [{self.section}]: {verdict}")
