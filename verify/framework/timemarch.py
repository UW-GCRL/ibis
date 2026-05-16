"""Time-march divergence harness.

Stateful sections (soilh2o, soilheat, snow, canopy energy balance) update
a state dict over many timesteps.  A single-call comparison only shows
step-1 divergence; we also want to know *where, how fast, and on which
variable* the trajectories drift apart over time.

This module exposes two functions:

    time_march_compare(f77_step, jax_step, init_state, n_steps,
                       var_specs, forcings=None, abs_floor=1e-6)
        Runs both step functions in lockstep starting from identical
        initial state.  Returns a long-form DataFrame:
            columns = [step, variable, tol_class, n, max_abs, max_rel,
                       rmse, bias, max_ulp, n_nan_ref, n_nan_cand]
        One row per (step, variable).

    summarize_divergence(df, var_specs)
        Reports first-failing (step, variable) and per-variable
        worst-step stats.  Returns a small DataFrame for display.

`forcings` is an optional callable `step_index -> forcings_dict` for
inputs that change per step (e.g., precipitation flux at the top of
the soil column).  The step functions receive (state, forcings) when
forcings is given, otherwise just (state).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .stats import compute
from .tolerance_classes import ToleranceSpec


StepFn = Callable[..., dict]


def time_march_compare(
    f77_step: StepFn,
    jax_step: StepFn,
    init_state: dict,
    n_steps: int,
    var_specs: dict[str, ToleranceSpec],
    forcings: Optional[Callable[[int], dict]] = None,
    abs_floor: float = 1e-6,
) -> pd.DataFrame:
    """Run f77_step and jax_step in lockstep; record per-step residuals.

    Both step functions get the SAME initial state.  Each step we collect
    residual stats for every variable in var_specs.  We never feed one
    side's output into the other -- they march independently, but seeded
    identically.

    If forcings is provided, each step gets `forcings(step)` as a second
    positional argument; otherwise just the state.
    """
    f77_state = {k: np.array(v, dtype=np.float64, copy=True)
                 for k, v in init_state.items()}
    jax_state = {k: np.array(v, dtype=np.float64, copy=True)
                 for k, v in init_state.items()}

    rows = []
    for step in range(1, n_steps + 1):
        if forcings is None:
            f77_state = f77_step(f77_state)
            jax_state = jax_step(jax_state)
        else:
            f = forcings(step)
            f77_state = f77_step(f77_state, f)
            jax_state = jax_step(jax_state, f)

        for var, spec in var_specs.items():
            if var not in f77_state or var not in jax_state:
                raise KeyError(f"variable '{var}' missing from step output")
            stats = compute(f77_state[var], jax_state[var], abs_floor=abs_floor)
            rows.append({
                "step": step,
                "variable": var,
                "tol_class": spec.tol_class.value,
                **stats.as_dict(),
            })

    return pd.DataFrame(rows)


def summarize_divergence(df: pd.DataFrame,
                         var_specs: dict[str, ToleranceSpec]) -> pd.DataFrame:
    """Reduce a long-form residual frame to per-variable summary.

    Columns of the returned frame:
      variable, tol_class, worst_step, worst_max_abs, worst_max_rel,
      worst_bias, first_fail_step, status
    """
    out = []
    for var, spec in var_specs.items():
        sub = df[df["variable"] == var]
        if sub.empty:
            continue
        # worst step by max_abs
        worst_idx = sub["max_abs"].abs().idxmax()
        worst_row = sub.loc[worst_idx]

        # first failure
        eff_abs = spec.effective_abs
        eff_rel = spec.effective_rel
        # PASS iff max_abs <= abs_tol OR max_rel <= rel_tol
        ok_abs = (eff_abs is not None) & (sub["max_abs"] <= (eff_abs or 0))
        ok_rel = (eff_rel is not None) & (sub["max_rel"] <= (eff_rel or 0))
        passed = ok_abs | ok_rel
        if spec.bias_tol is not None:
            passed = passed & (sub["bias"].abs() <= spec.bias_tol)
        failed = ~passed
        if failed.any():
            first_fail_step = int(sub.loc[failed, "step"].min())
            status = "FAIL"
        else:
            first_fail_step = None
            status = "PASS"

        out.append({
            "variable": var,
            "tol_class": spec.tol_class.value,
            "worst_step": int(worst_row["step"]),
            "worst_max_abs": float(worst_row["max_abs"]),
            "worst_max_rel": float(worst_row["max_rel"]),
            "worst_bias": float(worst_row["bias"]),
            "first_fail_step": first_fail_step,
            "status": status,
        })
    return pd.DataFrame(out)


def print_summary(summary: pd.DataFrame, n_steps: int) -> None:
    print(f"\n{'STATUS':<6} {'VARIABLE':<24} {'CLASS':<10} "
          f"{'WORST_STEP':>11} {'MAX_ABS':>12} {'MAX_REL':>12} "
          f"{'BIAS':>12} {'FIRST_FAIL':>11}")
    print("-" * 110)
    for _, r in summary.iterrows():
        # pandas turns Python None into NaN inside a column; check both.
        first_val = r["first_fail_step"]
        first = "-" if (first_val is None or pd.isna(first_val)) else str(int(first_val))
        print(f"{r['status']:<6} {r['variable']:<24} {r['tol_class']:<10} "
              f"{int(r['worst_step']):>11} "
              f"{r['worst_max_abs']:>12.3e} {r['worst_max_rel']:>12.3e} "
              f"{r['worst_bias']:>+12.3e} {first:>11}")
    print("-" * 110)
    print(f"Steps total: {n_steps}")
