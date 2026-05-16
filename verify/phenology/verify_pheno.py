"""Verify JAX pheno against F77 pheno.

Phenology is deterministic — pure threshold logic + linear arithmetic, no RNG,
no iteration. Expected: bit-exact.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.phenology.verify_pheno
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
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "phenology"),
)
import phenology_wrap as f77

import equinox as eqx
from ibis.physics.phenology import pheno as jax_pheno
from ibis.state.vegetation import create_vegetation_state
from ibis.state.summary import create_summary_state

from verify.framework.harness import Report, verify_variable
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


EPSILON = 1.0e-12

SPEC_BIT = ToleranceSpec("scalar", TolClass.BIT_EXACT,
                         why="Deterministic threshold + linear math.")


def make_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        td       = rng.uniform(260.0, 305.0, n),
        a10td    = rng.uniform(255.0, 305.0, n),
        a10ancub = rng.uniform(-5e-6, 5e-6, n),
        a10ancls = rng.uniform(-5e-6, 5e-6, n),
        a10ancl4 = rng.uniform(-5e-6, 5e-6, n),
        a10ancl3 = rng.uniform(-5e-6, 5e-6, n),
        tc       = rng.uniform(-30.0, 25.0, n),
        plai     = rng.uniform(0.0, 4.0, (n, 12)),
        fu       = rng.uniform(0.1, 0.95, n),
        fl       = rng.uniform(0.1, 0.95, n),
        tempu    = rng.uniform(0.0, 1.0, n),
        templ    = rng.uniform(0.0, 1.0, n),
        agddu    = rng.uniform(0.0, 500.0, n),
        agddl    = rng.uniform(0.0, 500.0, n),
        dropu    = rng.uniform(0.1, 1.0, n),
        dropls   = rng.uniform(0.1, 1.0, n),
        dropl4   = rng.uniform(0.1, 1.0, n),
        dropl3   = rng.uniform(0.1, 1.0, n),
        zbot     = rng.uniform(0.0, 10.0, (n, 2)),
        ztop     = rng.uniform(1.0, 20.0, (n, 2)),
        lai      = rng.uniform(0.0, 8.0, (n, 2)),
        frac     = rng.uniform(0.0, 1.0, (n, 12)),
    )


def call_f77(inputs: dict) -> dict:
    out = f77.wrap_pheno(
        inputs["td"], inputs["a10td"],
        inputs["a10ancub"], inputs["a10ancls"],
        inputs["a10ancl4"], inputs["a10ancl3"],
        inputs["tc"], inputs["plai"], inputs["fu"], inputs["fl"],
        inputs["tempu"], inputs["templ"], inputs["agddu"], inputs["agddl"],
        inputs["dropu"], inputs["dropls"], inputs["dropl4"], inputs["dropl3"],
        inputs["zbot"], inputs["ztop"], inputs["lai"], inputs["frac"],
        EPSILON,
    )
    (tempu, templ, agddu, agddl,
     dropu, dropls, dropl4, dropl3,
     frac, lai, ztop, zbot) = out
    return dict(
        tempu=np.asarray(tempu), templ=np.asarray(templ),
        agddu=np.asarray(agddu), agddl=np.asarray(agddl),
        dropu=np.asarray(dropu), dropls=np.asarray(dropls),
        dropl4=np.asarray(dropl4), dropl3=np.asarray(dropl3),
        frac=np.asarray(frac), lai=np.asarray(lai),
        ztop=np.asarray(ztop), zbot=np.asarray(zbot),
    )


def call_jax(inputs: dict) -> dict:
    n = inputs["td"].shape[0]
    veg = create_vegetation_state(n, npft=12, nsoilay=6)
    summ = create_summary_state(n, npft=12)

    veg = eqx.tree_at(
        lambda v: (v.tc, v.plai, v.fu, v.fl, v.tempu, v.templ,
                   v.agddu, v.agddl, v.dropu, v.dropls, v.dropl4, v.dropl3,
                   v.lai, v.ztop, v.zbot, v.frac),
        veg,
        (jnp.asarray(inputs["tc"]),
         jnp.asarray(inputs["plai"]),
         jnp.asarray(inputs["fu"]),
         jnp.asarray(inputs["fl"]),
         jnp.asarray(inputs["tempu"]),
         jnp.asarray(inputs["templ"]),
         jnp.asarray(inputs["agddu"]),
         jnp.asarray(inputs["agddl"]),
         jnp.asarray(inputs["dropu"]),
         jnp.asarray(inputs["dropls"]),
         jnp.asarray(inputs["dropl4"]),
         jnp.asarray(inputs["dropl3"]),
         jnp.asarray(inputs["lai"]),
         jnp.asarray(inputs["ztop"]),
         jnp.asarray(inputs["zbot"]),
         jnp.asarray(inputs["frac"])),
    )
    summ = eqx.tree_at(
        lambda s: (s.a10td, s.a10ancub, s.a10ancls, s.a10ancl4, s.a10ancl3),
        summ,
        (jnp.asarray(inputs["a10td"]),
         jnp.asarray(inputs["a10ancub"]),
         jnp.asarray(inputs["a10ancls"]),
         jnp.asarray(inputs["a10ancl4"]),
         jnp.asarray(inputs["a10ancl3"])),
    )

    veg_new = jax_pheno(veg, summ, jnp.asarray(inputs["td"]), EPSILON)
    return dict(
        tempu=np.asarray(veg_new.tempu), templ=np.asarray(veg_new.templ),
        agddu=np.asarray(veg_new.agddu), agddl=np.asarray(veg_new.agddl),
        dropu=np.asarray(veg_new.dropu), dropls=np.asarray(veg_new.dropls),
        dropl4=np.asarray(veg_new.dropl4), dropl3=np.asarray(veg_new.dropl3),
        frac=np.asarray(veg_new.frac), lai=np.asarray(veg_new.lai),
        ztop=np.asarray(veg_new.ztop), zbot=np.asarray(veg_new.zbot),
    )


def verify_one(report: Report, label: str, n: int, seed: int) -> None:
    inputs = make_inputs(n, seed)
    f_out = call_f77(inputs)
    j_out = call_jax(inputs)
    for name in ("tempu", "templ", "agddu", "agddl",
                 "dropu", "dropls", "dropl4", "dropl3",
                 "frac", "lai", "ztop", "zbot"):
        report.add(verify_variable(
            f"pheno.{name}.{label}", f_out[name], j_out[name], SPEC_BIT,
            abs_floor=1e-6,
        ))


def main() -> int:
    rep = Report(section="phenology")
    verify_one(rep, "rand1", n=1500, seed=42)
    verify_one(rep, "rand2", n=1500, seed=99)
    # Edge case: cold winter (a10td below thresholds, agddu/l should reset)
    inputs = make_inputs(500, seed=7)
    inputs["a10td"] = np.full(500, 260.0)
    f_out = call_f77(inputs); j_out = call_jax(inputs)
    for name in ("agddu", "agddl", "tempu", "templ"):
        rep.add(verify_variable(f"pheno.{name}.cold", f_out[name], j_out[name],
                                SPEC_BIT, abs_floor=1e-6))
    # Edge case: warm summer + abundant photo
    inputs = make_inputs(500, seed=123)
    inputs["a10td"] = np.full(500, 300.0)
    inputs["a10ancub"] = np.full(500, 5e-6)
    f_out = call_f77(inputs); j_out = call_jax(inputs)
    for name in ("agddu", "tempu", "dropu", "lai"):
        rep.add(verify_variable(f"pheno.{name}.warm", f_out[name], j_out[name],
                                SPEC_BIT, abs_floor=1e-6))

    rep.print_table()
    rep.save(Path(__file__).parent / "results")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
