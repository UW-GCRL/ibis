"""Verify JAX steph2o against F77 steph2o.

steph2o steps intercepted h2o (liquid + snow) for one canopy component
through one timestep, accounting for interception, drip/blowoff, and
phase change.

Single-call + time-march tests.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.canopy.verify_steph2o
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
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "canopy"),
)
import canopy_wrap as f77

from ibis.physics.interception import steph2o as jax_steph2o

from verify.framework.harness import Report, verify_variable
from verify.framework.timemarch import (
    time_march_compare, summarize_divergence, print_summary,
)
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


CH2O    = 4.218e3
CICE    = 2.106e3
HFUS    = 0.3336e6
TMELT   = 273.16
EPSILON = 1.0e-7
DTIME   = 3600.0


STEPH2O_SPECS = {
    "wliq":  ToleranceSpec("wliq",  TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                            why="State arithmetic; same algebra both sides."),
    "wsno":  ToleranceSpec("wsno",  TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                            why="State arithmetic."),
    "pflux": ToleranceSpec("pflux", TolClass.PRECISION,
                            abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
                            why="Heat flux accumulator; multiplications with hvap-scale values."),
    "rain":  ToleranceSpec("rain",  TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                            why="Throughfall rain rate."),
    "train": ToleranceSpec("train", TolClass.PRECISION,
                            abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
                            why="Throughfall rain T (via mix)."),
    "snow":  ToleranceSpec("snow",  TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                            why="Throughfall snow rate."),
    "tsnow": ToleranceSpec("tsnow", TolClass.PRECISION,
                            abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
                            why="Throughfall snow T (via mix)."),
}


def make_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        tveg=rng.uniform(255.0, 305.0, n),
        wliq=rng.uniform(0.0, 0.2, n),
        wsno=rng.uniform(0.0, 2.0, n),
        xai=rng.uniform(0.0, 6.0, n),
        rain=rng.uniform(0.0, 5e-4, n),
        train=rng.uniform(275.0, 295.0, n),
        snow=rng.uniform(0.0, 2e-4, n),
        tsnow=rng.uniform(260.0, 273.16, n),
    )


def run_step(state: dict, forcings: dict) -> tuple[dict, dict]:
    tdrip = 7200.0
    tblow = 14400.0
    wliqmax = 0.20
    wsnomax = 2.00
    wliqmin = 0.0010 * (DTIME / 3600.0) * (wliqmax / 0.2)
    wsnomin = 0.0010 * (DTIME / 3600.0) * (wsnomax / 2.0)

    f_out = f77.wrap_steph2o(
        state["tveg"], state["wliq"], state["wsno"], state["xai"],
        forcings["rain"], forcings["train"], forcings["snow"], forcings["tsnow"],
        tdrip, tblow, wliqmax, wsnomax, wliqmin, wsnomin,
        DTIME, CH2O, CICE, HFUS, TMELT, EPSILON,
    )
    (tveg_f, wliq_f, wsno_f, xai_f, pflux_f,
     rain_f, train_f, snow_f, tsnow_f) = f_out

    j_out = jax_steph2o(
        tveg=jnp.array(state["tveg"]),
        wliq=jnp.array(state["wliq"]),
        wsno=jnp.array(state["wsno"]),
        xai=jnp.array(state["xai"]),
        pflux=jnp.zeros(state["tveg"].shape[0]),   # pflux not in JAX signature input position; initialized
        rain=jnp.array(forcings["rain"]),
        train=jnp.array(forcings["train"]),
        snow=jnp.array(forcings["snow"]),
        tsnow=jnp.array(forcings["tsnow"]),
        tdrip=tdrip, tblow=tblow,
        wliqmax=wliqmax, wsnomax=wsnomax,
        wliqmin=wliqmin, wsnomin=wsnomin,
        dtime=DTIME, ch2o=CH2O, cice=CICE,
        tmelt=TMELT, hfus=HFUS, epsilon=EPSILON,
    )

    f_d = {
        "wliq": np.asarray(wliq_f), "wsno": np.asarray(wsno_f),
        "pflux": np.asarray(pflux_f),
        "rain": np.asarray(rain_f), "train": np.asarray(train_f),
        "snow": np.asarray(snow_f), "tsnow": np.asarray(tsnow_f),
    }
    j_d = {
        "wliq": np.asarray(j_out.wliq), "wsno": np.asarray(j_out.wsno),
        "pflux": np.asarray(j_out.pflux),
        "rain": np.asarray(j_out.rain), "train": np.asarray(j_out.train),
        "snow": np.asarray(j_out.snow), "tsnow": np.asarray(j_out.tsnow),
    }
    return f_d, j_d


def verify_section() -> Report:
    report = Report(section="steph2o")
    inputs = make_inputs(n=1500, seed=42)
    state = {k: inputs[k] for k in ("tveg", "wliq", "wsno", "xai")}
    forcings = {k: inputs[k] for k in ("rain", "train", "snow", "tsnow")}
    f_d, j_d = run_step(state, forcings)
    for name, spec in STEPH2O_SPECS.items():
        report.add(verify_variable(
            f"steph2o.{name}", f_d[name], j_d[name], spec, abs_floor=1e-6,
        ))
    return report


def verify_time_march() -> None:
    """Run 100 timesteps with steady rain/snow forcings; state evolves."""
    n = 64
    n_steps = 100
    init = make_inputs(n=n, seed=7)
    state_keys = ("tveg", "wliq", "wsno", "xai")
    init_state = {k: init[k] for k in state_keys}
    # Output fields the harness will read; init to zeros so it can read step 0.
    init_state["pflux"] = np.zeros(n)
    init_state["rain"]  = init["rain"]
    init_state["train"] = init["train"]
    init_state["snow"]  = init["snow"]
    init_state["tsnow"] = init["tsnow"]

    rng = np.random.default_rng(99)
    forcings_seq = [
        dict(
            rain=rng.uniform(0.0, 3e-4, n),
            train=rng.uniform(275.0, 290.0, n),
            snow=rng.uniform(0.0, 1e-4, n),
            tsnow=rng.uniform(263.0, 273.16, n),
        )
        for _ in range(n_steps + 1)
    ]

    def f77_step(state: dict, f: dict) -> dict:
        f_d, _ = run_step(state, f)
        new = dict(state)
        new["wliq"]  = f_d["wliq"]
        new["wsno"]  = f_d["wsno"]
        new["pflux"] = f_d["pflux"]
        new["rain"]  = f_d["rain"]
        new["train"] = f_d["train"]
        new["snow"]  = f_d["snow"]
        new["tsnow"] = f_d["tsnow"]
        return new

    def jax_step(state: dict, f: dict) -> dict:
        _, j_d = run_step(state, f)
        new = dict(state)
        new["wliq"]  = j_d["wliq"]
        new["wsno"]  = j_d["wsno"]
        new["pflux"] = j_d["pflux"]
        new["rain"]  = j_d["rain"]
        new["train"] = j_d["train"]
        new["snow"]  = j_d["snow"]
        new["tsnow"] = j_d["tsnow"]
        return new

    def forcings_at(step: int) -> dict:
        return forcings_seq[step - 1]

    df = time_march_compare(
        f77_step=f77_step, jax_step=jax_step,
        init_state=init_state, n_steps=n_steps,
        var_specs=STEPH2O_SPECS,
        forcings=forcings_at, abs_floor=1e-6,
    )
    summary = summarize_divergence(df, STEPH2O_SPECS)
    print("\n=== steph2o time-march (n=64, steps=100) ===")
    print_summary(summary, n_steps=n_steps)
    out = Path(__file__).parent / "results" / "steph2o_timemarch.csv"
    df.to_csv(out, index=False)
    print(f"Per-step residuals saved: {out}")


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nSingle-call residuals saved: {out}")
    verify_time_march()
    sys.exit(0 if rep.passed else 1)
