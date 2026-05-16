"""Verify JAX steph2o2 against F77 steph2o2.

steph2o2 is the end-of-timestep adjustment: removes evaporation from
intercepted h2o (with safety fallback to atmosphere if it goes negative)
then does final heat-conserving phase change.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.canopy.verify_steph2o2
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

from ibis.physics.interception import steph2o2 as jax_steph2o2

from verify.framework.harness import Report, verify_variable
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


CH2O    = 4.218e3
CICE    = 2.106e3
HFUS    = 0.3336e6
HVAP    = 2.5104e6
HSUB    = HVAP + HFUS
CVAP    = 1.81e3
TMELT   = 273.16
DTIME   = 3600.0
CVEG    = 200.0   # specific heat of veg component


STEPH2O2_SPECS = {
    "tveg":  ToleranceSpec("tveg",  TolClass.PRECISION,
                            abs_tol=1e-9, rel_tol=1e-11, bias_tol=1e-10,
                            why="Phase-change T update."),
    "wliq":  ToleranceSpec("wliq",  TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                            why="State arithmetic."),
    "wsno":  ToleranceSpec("wsno",  TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                            why="State arithmetic."),
    "fvapa": ToleranceSpec("fvapa", TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-11, bias_tol=1e-12,
                            why="Vapor flux to atmosphere (safety fallback for negative wliq/wsno)."),
    "fsena": ToleranceSpec("fsena", TolClass.PRECISION,
                            abs_tol=1e-6, rel_tol=1e-11, bias_tol=1e-7,
                            why="Sensible heat to atmosphere; product with hvapf/hsubf magnitudes."),
}


def make_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        tveg=rng.uniform(255.0, 305.0, n),
        wliq=rng.uniform(0.0, 0.2, n),
        wsno=rng.uniform(0.0, 2.0, n),
        fveg=rng.uniform(0.0, 1.0, n),
        xai=rng.uniform(0.0, 6.0, n),
        rliq=rng.uniform(0.0, 1.0, n),
        fvapw=rng.uniform(-1e-7, 1e-7, n),   # small evap/condensation
        ta=rng.uniform(255.0, 305.0, n),
        fvapa=rng.uniform(0.0, 1e-5, n),
        fsena=rng.uniform(-50.0, 50.0, n),
    )


def verify_section() -> Report:
    report = Report(section="steph2o2")
    inputs = make_inputs(n=1500, seed=42)

    f_out = f77.wrap_steph2o2(
        inputs["tveg"], inputs["wliq"], inputs["wsno"],
        inputs["fveg"], inputs["xai"], inputs["rliq"],
        inputs["fvapw"], CVEG,
        inputs["ta"], inputs["fvapa"], inputs["fsena"],
        DTIME, CH2O, CICE, HFUS, TMELT, HVAP, HSUB, CVAP,
    )
    tveg_f, wliq_f, wsno_f, fvapa_f, fsena_f = f_out

    j_out = jax_steph2o2(
        jnp.array(inputs["tveg"]),
        jnp.array(inputs["wliq"]),
        jnp.array(inputs["wsno"]),
        jnp.array(inputs["fveg"]),
        jnp.array(inputs["xai"]),
        jnp.array(inputs["rliq"]),
        jnp.array(inputs["fvapw"]),
        CVEG, DTIME, CH2O, CICE, HFUS, TMELT,
        jnp.array(inputs["ta"]),
        jnp.array(inputs["fvapa"]),
        jnp.array(inputs["fsena"]),
        HVAP, HSUB, CVAP,
    )
    # JAX returns (tveg, wliq, wsno, fvapa, fsena)
    tveg_j, wliq_j, wsno_j, fvapa_j, fsena_j = j_out

    pairs = [
        ("tveg",  np.asarray(tveg_f),  np.asarray(tveg_j)),
        ("wliq",  np.asarray(wliq_f),  np.asarray(wliq_j)),
        ("wsno",  np.asarray(wsno_f),  np.asarray(wsno_j)),
        ("fvapa", np.asarray(fvapa_f), np.asarray(fvapa_j)),
        ("fsena", np.asarray(fsena_f), np.asarray(fsena_j)),
    ]
    for name, f_v, j_v in pairs:
        report.add(verify_variable(
            f"steph2o2.{name}", f_v, j_v, STEPH2O2_SPECS[name],
            abs_floor=1e-6,
        ))
    return report


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")
    sys.exit(0 if rep.passed else 1)
