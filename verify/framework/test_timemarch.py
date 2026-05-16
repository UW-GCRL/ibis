"""Self-test for time_march_compare.

Uses a trivial linear-decay iteration that we can implement identically
in pure NumPy on both sides -- if the harness mechanics work, these
should stay bit-exact for thousands of steps.

A second variant adds a deliberate small bug to one side (off-by-1e-5
in the rate) to confirm the harness flags the divergence at the right
step.

Run:
    PYTHONPATH=. verify/.venv/bin/python -m verify.framework.test_timemarch
"""
from __future__ import annotations

import numpy as np

from verify.framework.timemarch import (
    time_march_compare, summarize_divergence, print_summary,
)
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


def make_step(decay_rate: float):
    def step(state):
        return {"x": state["x"] * decay_rate}
    return step


def test_no_divergence():
    """Both sides identical -> bit-exact across all steps."""
    init = {"x": np.array([1.0, 2.0, 3.0, 4.0])}
    specs = {"x": ToleranceSpec(name="x", tol_class=TolClass.BIT_EXACT,
                                 abs_tol=1e-15, rel_tol=1e-15)}
    df = time_march_compare(
        f77_step=make_step(0.99),
        jax_step=make_step(0.99),
        init_state=init, n_steps=500, var_specs=specs,
    )
    summary = summarize_divergence(df, specs)
    print("\n=== Test 1: identical step functions ===")
    print_summary(summary, n_steps=500)
    assert summary.iloc[0]["status"] == "PASS", "identical steps should match"
    assert summary.iloc[0]["worst_max_abs"] == 0.0


def test_with_divergence():
    """Bug on one side: rate 0.99 vs 0.99001 -> error grows monotonically."""
    init = {"x": np.array([1.0, 2.0, 3.0, 4.0])}
    specs = {"x": ToleranceSpec(name="x", tol_class=TolClass.TIGHT,
                                 abs_tol=1e-3, rel_tol=1e-3)}
    df = time_march_compare(
        f77_step=make_step(0.99),
        jax_step=make_step(0.99001),
        init_state=init, n_steps=500, var_specs=specs,
    )
    summary = summarize_divergence(df, specs)
    print("\n=== Test 2: rate diff (F77 0.99 vs JAX 0.99001) ===")
    print_summary(summary, n_steps=500)
    # Confirm: error grows over time, eventually exceeds tolerance.
    by_step = df.set_index("step")["max_abs"]
    assert by_step.iloc[-1] > by_step.iloc[0], "error should grow over time"
    assert summary.iloc[0]["status"] == "FAIL", "should detect divergence"
    assert summary.iloc[0]["first_fail_step"] is not None


def test_two_variables_one_buggy():
    """One variable bit-exact, another diverges; harness flags only the bad one."""
    def step_good(rate_x, rate_y):
        def f(state):
            return {"x": state["x"] * rate_x, "y": state["y"] * rate_y}
        return f

    init = {"x": np.array([1.0, 2.0]), "y": np.array([10.0, 20.0])}
    specs = {
        "x": ToleranceSpec("x", TolClass.TIGHT, abs_tol=1e-12, rel_tol=1e-12),
        "y": ToleranceSpec("y", TolClass.TIGHT, abs_tol=1e-12, rel_tol=1e-12),
    }
    df = time_march_compare(
        f77_step=step_good(0.99, 0.95),
        jax_step=step_good(0.99, 0.95001),
        init_state=init, n_steps=200, var_specs=specs,
    )
    summary = summarize_divergence(df, specs)
    print("\n=== Test 3: only 'y' has rate diff; 'x' should PASS ===")
    print_summary(summary, n_steps=200)
    x_row = summary[summary["variable"] == "x"].iloc[0]
    y_row = summary[summary["variable"] == "y"].iloc[0]
    assert x_row["status"] == "PASS", "x should not diverge"
    assert y_row["status"] == "FAIL", "y should diverge"


if __name__ == "__main__":
    test_no_divergence()
    test_with_divergence()
    test_two_variables_one_buggy()
    print("\n=== All harness self-tests passed ===")
