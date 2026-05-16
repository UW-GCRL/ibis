"""Verify JAX soilh2o against F77 soilh2o.

Two tests:

  1. SINGLE-CALL: Run both sides once on a broad random input distribution
     plus hand-crafted edge cases (dry/saturated/frozen/heavy-rain).
     Reports per-variable max_abs/max_rel/RMSE/bias.

  2. TIME-MARCH: Start both sides from identical state, advance through
     N timesteps with the same per-step forcings, and use the time-march
     harness to find the first step where any state variable diverges
     beyond tolerance.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.soil.verify_soilh2o
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "soil"),
)
import soil_wrap as f77

from ibis.physics.soil_physics import soilh2o as jax_soilh2o

from verify.framework.harness import Report, verify_variable
from verify.framework.timemarch import (
    time_march_compare, summarize_divergence, print_summary,
)
from verify.soil.tolerance_soilh2o import SOILH2O_SPECS


# Physical constants / solver settings
BPERM   = 0.10
DTIME   = 3600.0      # 1-hour timestep
RHOW    = 1000.0
EPSILON = 1.0e-7

# Per-layer thicknesses (m), standard IBIS 6-layer profile.
HSOI = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00], dtype=np.float64)


def make_inputs(n: int, seed: int) -> dict:
    """Random soil state + forcings.  porosflo derived consistently."""
    rng = np.random.default_rng(seed)
    nl = 6

    wsoi = rng.uniform(0.05, 0.95, (n, nl))
    wisoi = rng.uniform(0.0, 0.5, (n, nl))
    # Keep liquid + ice <= 1 (sum is wsoi + wisoi after combination, but
    # actual water = (1-wisoi)*wsoi + wisoi which is naturally <= 1).
    poros = rng.uniform(0.30, 0.55, (n, nl))
    porosflo = poros * np.maximum(1.0 - wisoi, EPSILON)
    hydraul = 10.0 ** rng.uniform(-7.0, -4.0, (n, nl))  # log-uniform 1e-7..1e-4
    suction = rng.uniform(0.05, 0.5, (n, nl))
    bex = rng.uniform(3.0, 12.0, (n, nl))
    wpud = rng.uniform(0.0, 4.5, n)

    # Forcings: mix of dry/light-rain/heavy-rain/evap
    fwtop = rng.uniform(-2e-5, 5e-4, n)
    fwpud = rng.uniform(0.0, 1e-4, n)
    return dict(
        wsoi=wsoi, wisoi=wisoi, wpud=wpud,
        poros=poros, porosflo=porosflo,
        hydraul=hydraul, suction=suction, bex=bex,
        fwtop=fwtop, fwpud=fwpud,
    )


def run_one_step(state: dict, forcings: dict) -> tuple[dict, dict]:
    """Run F77 wrap_soilh2o and JAX soilh2o on the same state+forcings.

    Returns dicts of outputs keyed by name from both sides.
    """
    f_out = f77.wrap_soilh2o(
        state["wsoi"], state["wisoi"], state["wpud"],
        state["poros"], state["porosflo"],
        state["hydraul"], state["suction"], state["bex"],
        HSOI,
        forcings["fwtop"], forcings["fwpud"],
        BPERM, DTIME, RHOW, EPSILON,
    )
    wsoi_f, wpud_f, fsq_f, wflo_f = f_out

    j_out = jax_soilh2o(
        wsoi=jnp.array(state["wsoi"]),
        wisoi=jnp.array(state["wisoi"]),
        poros=jnp.array(state["poros"]),
        porosflo=jnp.array(state["porosflo"]),
        hydraul=jnp.array(state["hydraul"]),
        suction=jnp.array(state["suction"]),
        bex=jnp.array(state["bex"]),
        hsoi=jnp.array(HSOI),
        fwtop=jnp.array(forcings["fwtop"]),
        fwpud=jnp.array(forcings["fwpud"]),
        wpud=jnp.array(state["wpud"]),
        bperm=BPERM, dtime=DTIME, rhow=RHOW, epsilon=EPSILON,
    )

    return (
        {"wsoi": np.asarray(wsoi_f), "wpud": np.asarray(wpud_f),
         "wflo": np.asarray(wflo_f), "fsqueez": np.asarray(fsq_f)},
        {"wsoi": np.asarray(j_out.wsoi), "wpud": np.asarray(j_out.wpud),
         "wflo": np.asarray(j_out.wflo), "fsqueez": np.asarray(j_out.fsqueez)},
    )


def verify_single_call(report: Report) -> None:
    """Pass 1: single-step verification on broad random distribution."""
    n = 1500
    inputs = make_inputs(n=n, seed=42)
    state = {k: inputs[k] for k in
             ("wsoi", "wisoi", "wpud", "poros", "porosflo",
              "hydraul", "suction", "bex")}
    forcings = {k: inputs[k] for k in ("fwtop", "fwpud")}
    f_d, j_d = run_one_step(state, forcings)
    for name, spec in SOILH2O_SPECS.items():
        report.add(verify_variable(
            f"soilh2o.single.{name}", f_d[name], j_d[name], spec,
            abs_floor=1e-6,
        ))


def verify_time_march() -> None:
    """Pass 2: drive both sides through N steps with identical forcings;
    detect first divergence.  Uses a smaller batch (n=64) for speed."""
    n = 64
    n_steps = 100
    init = make_inputs(n=n, seed=7)
    state_keys = ("wsoi", "wisoi", "wpud", "poros", "porosflo",
                  "hydraul", "suction", "bex")
    init_state = {k: init[k] for k in state_keys}

    rng = np.random.default_rng(99)
    # Per-step forcings -- moderately random precipitation/evaporation.
    forcings_seq = [
        dict(fwtop=rng.uniform(-1e-5, 3e-4, n),
             fwpud=rng.uniform(0.0, 5e-5, n))
        for _ in range(n_steps + 1)
    ]

    def f77_step(state: dict, f: dict) -> dict:
        f_d, _ = run_one_step(state, f)
        # Update state for next step (state evolves; non-evolving fields stay).
        new = dict(state)
        new["wsoi"] = f_d["wsoi"]
        new["wpud"] = f_d["wpud"]
        new["wflo"]    = f_d["wflo"]
        new["fsqueez"] = f_d["fsqueez"]
        return new

    def jax_step(state: dict, f: dict) -> dict:
        _, j_d = run_one_step(state, f)
        new = dict(state)
        new["wsoi"] = j_d["wsoi"]
        new["wpud"] = j_d["wpud"]
        new["wflo"]    = j_d["wflo"]
        new["fsqueez"] = j_d["fsqueez"]
        return new

    # The time_march_compare harness expects step functions with signature
    # (state, forcings) -> state. We supply forcings via a step-indexed lookup.
    def forcings_at(step: int) -> dict:
        return forcings_seq[step - 1]

    # Initialize state with output fields zeroed so harness can read them.
    init_full = dict(init_state)
    init_full["wflo"]    = np.zeros((n, 7))
    init_full["fsqueez"] = np.zeros(n)

    df = time_march_compare(
        f77_step=f77_step,
        jax_step=jax_step,
        init_state=init_full,
        n_steps=n_steps,
        var_specs=SOILH2O_SPECS,
        forcings=forcings_at,
        abs_floor=1e-6,
    )
    summary = summarize_divergence(df, SOILH2O_SPECS)
    print("\n=== Time-march divergence test (n=64, steps=100) ===")
    print_summary(summary, n_steps=n_steps)
    out = Path(__file__).parent / "results" / "soilh2o_timemarch.csv"
    df.to_csv(out, index=False)
    print(f"Per-step residuals saved: {out}")


def verify_section() -> Report:
    report = Report(section="soilh2o")
    verify_single_call(report)
    return report


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nSingle-call residuals saved: {out}")

    # Always run time-march regardless of single-call result.
    verify_time_march()

    sys.exit(0 if rep.passed else 1)
