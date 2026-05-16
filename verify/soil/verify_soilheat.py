"""Verify JAX soilheat against F77 soilheat.

Single-call (1500 random inputs) + time-march (64 cells, 100 steps).
Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.soil.verify_soilheat
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

from ibis.physics.soil_physics import soilheat as jax_soilheat

from verify.framework.harness import Report, verify_variable
from verify.framework.timemarch import (
    time_march_compare, summarize_divergence, print_summary,
)
from verify.soil.tolerance_soilheat import SOILHEAT_SPECS


CH2O    = 4.218e3
CICE    = 2.106e3
RHOW    = 1000.0
DTIME   = 3600.0
HSOI    = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00], dtype=np.float64)


def make_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nl = 6
    tsoi = rng.uniform(255.0, 320.0, (n, nl))
    otsoi = tsoi + rng.uniform(-0.5, 0.5, (n, nl))   # tiny shift from current
    wsoi = rng.uniform(0.05, 0.95, (n, nl))
    owsoi = wsoi + rng.uniform(-0.05, 0.05, (n, nl))
    owsoi = np.clip(owsoi, 0.0, 1.0)
    wisoi = rng.uniform(0.0, 0.5, (n, nl))
    poros = rng.uniform(0.30, 0.55, (n, nl))
    consoi = rng.uniform(0.2, 2.5, (n, nl))
    csoi = rng.uniform(700.0, 1200.0, (n, nl))         # mineral specific heat
    rhosoi = rng.uniform(2200.0, 2900.0, (n, nl))
    # c0pud/c1pud nonzero only in the top layer
    c0pud = np.zeros((n, nl))
    c1pud = np.zeros((n, nl))
    c0pud[:, 0] = rng.uniform(0.0, 4.2e4, n)
    c1pud[:, 0] = rng.uniform(0.0, 4.2e4, n)
    fhtop = rng.uniform(-500.0, 500.0, n)              # W/m2
    wflo = rng.uniform(-1e-4, 1e-4, (n, 7))            # kg/m2/s
    return dict(tsoi=tsoi, otsoi=otsoi, owsoi=owsoi, wsoi=wsoi, wisoi=wisoi,
                poros=poros, consoi=consoi, csoi=csoi, rhosoi=rhosoi,
                c0pud=c0pud, c1pud=c1pud, fhtop=fhtop, wflo=wflo)


def run_one_step(state: dict, forcings: dict) -> tuple[dict, dict]:
    f_out = f77.wrap_soilheat(
        state["tsoi"], state["otsoi"], state["owsoi"],
        state["wsoi"], state["wisoi"],
        state["poros"], state["consoi"], state["csoi"], state["rhosoi"],
        HSOI,
        state["c0pud"], state["c1pud"],
        forcings["fhtop"], forcings["wflo"],
        CH2O, CICE, RHOW, DTIME,
    )
    tsoi_f, hflo_f = f_out

    j_out = jax_soilheat(
        tsoi=jnp.array(state["tsoi"]),
        otsoi=jnp.array(state["otsoi"]),
        owsoi=jnp.array(state["owsoi"]),
        wsoi=jnp.array(state["wsoi"]),
        wisoi=jnp.array(state["wisoi"]),
        poros=jnp.array(state["poros"]),
        consoi=jnp.array(state["consoi"]),
        csoi_mineral=jnp.array(state["csoi"]),
        rhosoi=jnp.array(state["rhosoi"]),
        hsoi=jnp.array(HSOI),
        c0pud=jnp.array(state["c0pud"]),
        c1pud=jnp.array(state["c1pud"]),
        fhtop=jnp.array(forcings["fhtop"]),
        wflo=jnp.array(forcings["wflo"]),
        ch2o=CH2O, cice=CICE, rhow=RHOW, dtime=DTIME,
    )

    f_d = {"tsoi": np.asarray(tsoi_f), "hflo": np.asarray(hflo_f)}
    j_d = {"tsoi": np.asarray(j_out.tsoi), "hflo": np.asarray(j_out.hflo)}
    return f_d, j_d


def verify_single_call(report: Report) -> None:
    n = 1500
    inputs = make_inputs(n=n, seed=42)
    state = {k: inputs[k] for k in
             ("tsoi", "otsoi", "owsoi", "wsoi", "wisoi", "poros",
              "consoi", "csoi", "rhosoi", "c0pud", "c1pud")}
    forcings = {"fhtop": inputs["fhtop"], "wflo": inputs["wflo"]}
    f_d, j_d = run_one_step(state, forcings)
    for name, spec in SOILHEAT_SPECS.items():
        report.add(verify_variable(
            f"soilheat.single.{name}", f_d[name], j_d[name], spec,
            abs_floor=1e-3,
        ))


def verify_time_march() -> None:
    n = 64
    n_steps = 100
    init = make_inputs(n=n, seed=7)
    state_keys = ("tsoi", "otsoi", "owsoi", "wsoi", "wisoi", "poros",
                  "consoi", "csoi", "rhosoi", "c0pud", "c1pud")
    init_state = {k: init[k] for k in state_keys}
    init_state["hflo"] = np.zeros((n, 7))   # so harness can read

    rng = np.random.default_rng(99)
    forcings_seq = [
        dict(fhtop=rng.uniform(-300.0, 300.0, n),
             wflo=rng.uniform(-5e-5, 5e-5, (n, 7)))
        for _ in range(n_steps + 1)
    ]

    def f77_step(state: dict, f: dict) -> dict:
        f_d, _ = run_one_step(state, f)
        new = dict(state)
        new["otsoi"] = state["tsoi"].copy()   # snapshot for next step
        new["tsoi"]  = f_d["tsoi"]
        new["hflo"]  = f_d["hflo"]
        return new

    def jax_step(state: dict, f: dict) -> dict:
        _, j_d = run_one_step(state, f)
        new = dict(state)
        new["otsoi"] = state["tsoi"].copy()
        new["tsoi"]  = j_d["tsoi"]
        new["hflo"]  = j_d["hflo"]
        return new

    def forcings_at(step: int) -> dict:
        return forcings_seq[step - 1]

    df = time_march_compare(
        f77_step=f77_step, jax_step=jax_step,
        init_state=init_state, n_steps=n_steps,
        var_specs=SOILHEAT_SPECS,
        forcings=forcings_at, abs_floor=1e-3,
    )
    summary = summarize_divergence(df, SOILHEAT_SPECS)
    print("\n=== Time-march divergence (n=64, steps=100) ===")
    print_summary(summary, n_steps=n_steps)
    out = Path(__file__).parent / "results" / "soilheat_timemarch.csv"
    df.to_csv(out, index=False)
    print(f"Per-step residuals saved: {out}")


def verify_section() -> Report:
    report = Report(section="soilheat")
    verify_single_call(report)
    return report


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nSingle-call residuals saved: {out}")
    verify_time_march()
    sys.exit(0 if rep.passed else 1)
