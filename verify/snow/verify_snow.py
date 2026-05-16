"""Verify JAX snow_physics against F77 snow.f (snowheat, vadapt, snow_melt, snow_coverage).

Single-call tests for all 4 operations, plus time-march on snowheat.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.snow.verify_snow
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
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "snow"),
)
import snow_wrap as f77

from ibis.physics.snow_physics import (
    snowheat as jax_snowheat,
    vadapt as jax_vadapt,
    snow_melt as jax_snow_melt,
    snow_coverage as jax_snow_coverage,
)

from verify.framework.harness import Report, verify_variable
from verify.framework.timemarch import (
    time_march_compare, summarize_divergence, print_summary,
)
from verify.snow.tolerance import (
    SNOWHEAT_SPECS, VADAPT_SPECS, SNOW_MELT_SPECS, SNOW_COVERAGE_SPECS,
)


CICE    = 2.106e3
HFUS    = 0.3336e6
TMELT   = 273.16
EPSILON = 1.0e-7
RHOS    = 250.0     # snow density (kg/m3)
CONSNO  = 0.20      # snow thermal conductivity (W/m/K)
CHL     = 200.0     # lower-veg specific heat (J/(area*K))
DTIME   = 3600.0
HSOI_1  = 0.10      # top soil layer thickness


# =====================================================================
# snowheat
# =====================================================================

def make_snowheat_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nsno = 3
    fi = rng.uniform(0.0, 1.0, n)
    # Make some cells with fi=0 (no snow) to exercise the has_snow branch.
    no_snow = rng.uniform(0, 1, n) < 0.15
    fi[no_snow] = 0.0
    tsno = rng.uniform(240.0, 273.16, (n, nsno))
    tlsub = rng.uniform(240.0, 280.0, n)
    hsno = rng.uniform(0.0, 0.5, (n, nsno))
    hsno[fi == 0] = 0.0
    xl = rng.uniform(0.0, 2.0, n)
    fhtop = rng.uniform(-300.0, 300.0, n)
    consoi_1 = rng.uniform(0.5, 2.5, n)
    tsoi_1 = rng.uniform(265.0, 295.0, n)
    return dict(fi=fi, tsno=tsno, tlsub=tlsub, hsno=hsno, xl=xl,
                fhtop=fhtop, consoi_1=consoi_1, tsoi_1=tsoi_1)


def run_snowheat(state: dict, forcings: dict) -> tuple[dict, dict]:
    f_out = f77.wrap_snowheat(
        state["tsno"], state["tlsub"], state["fi"], state["hsno"],
        state["xl"],
        HSOI_1, state["consoi_1"], state["tsoi_1"],
        forcings["fhtop"],
        CHL, RHOS, CONSNO, CICE, DTIME,
    )
    tsno_f, tlsub_f, sflo_f = f_out
    j_out = jax_snowheat(
        tsno=jnp.array(state["tsno"]),
        tlsub=jnp.array(state["tlsub"]),
        fhtop=jnp.array(forcings["fhtop"]),
        xl=jnp.array(state["xl"]),
        chl=CHL,
        fi=jnp.array(state["fi"]),
        hsno=jnp.array(state["hsno"]),
        rhos=RHOS, consno=CONSNO, hsoi_1=HSOI_1,
        consoi_1=jnp.array(state["consoi_1"]),
        tsoi_1=jnp.array(state["tsoi_1"]),
        dtime=DTIME, cice=CICE,
    )
    return (
        {"tsno": np.asarray(tsno_f), "tlsub": np.asarray(tlsub_f),
         "sflo": np.asarray(sflo_f)},
        {"tsno": np.asarray(j_out.tsno), "tlsub": np.asarray(j_out.tlsub),
         "sflo": np.asarray(j_out.sflo)},
    )


# =====================================================================
# vadapt
# =====================================================================

def make_vadapt_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nl = 3
    hcur = rng.uniform(0.0, 0.5, (n, nl))
    tcur = rng.uniform(240.0, 273.16, (n, nl))
    has_snow = (rng.uniform(0, 1, n) > 0.2).astype(np.float64)
    hcur[has_snow == 0] = 0.0
    return dict(hcur=hcur, tcur=tcur, has_snow=has_snow)


def run_vadapt(state: dict) -> tuple[dict, dict]:
    htop = 0.05
    hcur_f, tcur_f = f77.wrap_vadapt(
        state["hcur"], state["tcur"], htop, state["has_snow"], EPSILON,
    )
    hcur_j, tcur_j = jax_vadapt(
        hcur=jnp.array(state["hcur"]),
        tcur=jnp.array(state["tcur"]),
        htop=htop,
        has_snow=jnp.array(state["has_snow"]),
        epsilon=EPSILON,
    )
    return ({"hcur": np.asarray(hcur_f), "tcur": np.asarray(tcur_f)},
            {"hcur": np.asarray(hcur_j), "tcur": np.asarray(tcur_j)})


# =====================================================================
# snow_melt
# =====================================================================

def make_snow_melt_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nsno = 3
    fi = rng.uniform(0.0, 1.0, n)
    no_snow = rng.uniform(0, 1, n) < 0.15
    fi[no_snow] = 0.0
    tsno = rng.uniform(260.0, 280.0, (n, nsno))   # straddle tmelt
    hsno = rng.uniform(0.0, 0.5, (n, nsno))
    hsno[fi == 0] = 0.0
    return dict(fi=fi, tsno=tsno, hsno=hsno)


def run_snow_melt(state: dict) -> tuple[dict, dict]:
    hsno_f, tsno_f, zmelt_f, zheat_f = f77.wrap_snow_melt(
        state["hsno"], state["tsno"], state["fi"],
        RHOS, CICE, HFUS, TMELT,
    )
    hsno_j, tsno_j, zmelt_j, zheat_j = jax_snow_melt(
        hsno=jnp.array(state["hsno"]),
        tsno=jnp.array(state["tsno"]),
        fi=jnp.array(state["fi"]),
        rhos=RHOS, cice=CICE, hfus=HFUS, tmelt=TMELT,
    )
    return (
        {"hsno": np.asarray(hsno_f), "tsno": np.asarray(tsno_f),
         "zmelt": np.asarray(zmelt_f), "zheat": np.asarray(zheat_f)},
        {"hsno": np.asarray(hsno_j), "tsno": np.asarray(tsno_j),
         "zmelt": np.asarray(zmelt_j), "zheat": np.asarray(zheat_j)},
    )


# =====================================================================
# snow_coverage
# =====================================================================

def make_snow_coverage_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nsno = 3
    fi = rng.uniform(0.05, 1.0, n)   # avoid fi=0 (degenerate)
    hsno = rng.uniform(0.01, 0.5, (n, nsno))
    hsnoruf = rng.uniform(0.10, 0.75, n)
    return dict(fi=fi, hsno=hsno, hsnoruf=hsnoruf)


def run_snow_coverage(state: dict) -> tuple[dict, dict]:
    HSNOMIN = 0.05
    FIMAX = 1.0
    fi_f, hsno_f = f77.wrap_snow_coverage(
        state["hsno"], state["fi"], HSNOMIN, state["hsnoruf"], FIMAX, EPSILON,
    )
    fi_j, hsno_j = jax_snow_coverage(
        hsno=jnp.array(state["hsno"]),
        fi=jnp.array(state["fi"]),
        hsnomin=HSNOMIN, hsnoruf=jnp.array(state["hsnoruf"]),
        fimax=FIMAX, epsilon=EPSILON,
    )
    return ({"fi": np.asarray(fi_f), "hsno": np.asarray(hsno_f)},
            {"fi": np.asarray(fi_j), "hsno": np.asarray(hsno_j)})


# =====================================================================
# Main
# =====================================================================

def verify_section() -> Report:
    report = Report(section="snow_physics")

    # snowheat
    snowheat_in = make_snowheat_inputs(n=1500, seed=42)
    state = {k: snowheat_in[k] for k in
             ("tsno", "tlsub", "fi", "hsno", "xl", "consoi_1", "tsoi_1")}
    forcings = {"fhtop": snowheat_in["fhtop"]}
    f_d, j_d = run_snowheat(state, forcings)
    for name, spec in SNOWHEAT_SPECS.items():
        report.add(verify_variable(
            f"snowheat.{name}", f_d[name], j_d[name], spec, abs_floor=1e-3,
        ))

    # vadapt
    vadapt_in = make_vadapt_inputs(n=1500, seed=43)
    f_d, j_d = run_vadapt(vadapt_in)
    for name, spec in VADAPT_SPECS.items():
        report.add(verify_variable(
            f"vadapt.{name}", f_d[name], j_d[name], spec, abs_floor=1e-6,
        ))

    # snow_melt
    sm_in = make_snow_melt_inputs(n=1500, seed=44)
    f_d, j_d = run_snow_melt(sm_in)
    for name, spec in SNOW_MELT_SPECS.items():
        report.add(verify_variable(
            f"snow_melt.{name}", f_d[name], j_d[name], spec, abs_floor=1e-6,
        ))

    # snow_coverage
    sc_in = make_snow_coverage_inputs(n=1500, seed=45)
    f_d, j_d = run_snow_coverage(sc_in)
    for name, spec in SNOW_COVERAGE_SPECS.items():
        report.add(verify_variable(
            f"snow_coverage.{name}", f_d[name], j_d[name], spec, abs_floor=1e-6,
        ))

    return report


def verify_time_march_snowheat() -> None:
    """Time-march snowheat for 100 steps with random fhtop drift."""
    n = 64
    n_steps = 100
    init = make_snowheat_inputs(n=n, seed=7)
    state_keys = ("tsno", "tlsub", "fi", "hsno", "xl", "consoi_1", "tsoi_1")
    init_state = {k: init[k] for k in state_keys}
    init_state["sflo"] = np.zeros((n, 5))

    rng = np.random.default_rng(99)
    forcings_seq = [
        dict(fhtop=rng.uniform(-200.0, 200.0, n))
        for _ in range(n_steps + 1)
    ]

    def f77_step(state: dict, f: dict) -> dict:
        f_d, _ = run_snowheat(state, f)
        new = dict(state)
        new["tsno"]  = f_d["tsno"]
        new["tlsub"] = f_d["tlsub"]
        new["sflo"]  = f_d["sflo"]
        return new

    def jax_step(state: dict, f: dict) -> dict:
        _, j_d = run_snowheat(state, f)
        new = dict(state)
        new["tsno"]  = j_d["tsno"]
        new["tlsub"] = j_d["tlsub"]
        new["sflo"]  = j_d["sflo"]
        return new

    def forcings_at(step: int) -> dict:
        return forcings_seq[step - 1]

    df = time_march_compare(
        f77_step=f77_step, jax_step=jax_step,
        init_state=init_state, n_steps=n_steps,
        var_specs=SNOWHEAT_SPECS,
        forcings=forcings_at, abs_floor=1e-3,
    )
    summary = summarize_divergence(df, SNOWHEAT_SPECS)
    print("\n=== snowheat time-march (n=64, steps=100) ===")
    print_summary(summary, n_steps=n_steps)
    out = Path(__file__).parent / "results" / "snowheat_timemarch.csv"
    df.to_csv(out, index=False)
    print(f"Per-step residuals saved: {out}")


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nSingle-call residuals saved: {out}")
    verify_time_march_snowheat()
    sys.exit(0 if rep.passed else 1)
