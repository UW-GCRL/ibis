"""Verify JAX dynaveg (vegetation_dynamics.py) against F77 dynaveg.

dynaveg is the annual dynamic-vegetation update: applies turnover, NPP,
disturbances, recomputes LAI/biomass/canopy structure, then calls vegmap
to classify into IBIS vegtypes. Pure deterministic arithmetic.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.vegetation.verify_dynaveg
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
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "vegetation"),
)
import vegetation_wrap as f77

import equinox as eqx
from ibis.io.parameters import default_pft_params
from ibis.physics.vegetation_dynamics import dynaveg as jax_dynaveg
from ibis.state.vegetation import create_vegetation_state
from ibis.state.summary import create_summary_state
from ibis.state.carbon import create_carbon_state

from verify.framework.harness import Report, verify_variable
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


EPSILON = 1.0e-12

# F77 compft.h declares tauleaf/tauroot/tauwood0 AND xminlai as real*4 (the
# -fdefault-real-8 compile flag does NOT promote real*4). To match F77
# bit-exactly we truncate these to float32 before passing to JAX.
_PFT_RAW = default_pft_params()
_PFT = eqx.tree_at(
    lambda p: (p.tauleaf, p.tauroot, p.tauwood0, p.xminlai),
    _PFT_RAW,
    (jnp.asarray(np.asarray(_PFT_RAW.tauleaf,  dtype=np.float32), dtype=jnp.float64),
     jnp.asarray(np.asarray(_PFT_RAW.tauroot,  dtype=np.float32), dtype=jnp.float64),
     jnp.asarray(np.asarray(_PFT_RAW.tauwood0, dtype=np.float32), dtype=jnp.float64),
     float(np.float32(_PFT_RAW.xminlai))),
)
PFT = _PFT


SPEC_BIT = ToleranceSpec("v", TolClass.PRECISION,
                          abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14)
# Some derived sums accumulate per-PFT roundoff; allow a few ULP.
SPEC_PREC = ToleranceSpec("v", TolClass.PRECISION,
                           abs_tol=1e-11, rel_tol=1e-12, bias_tol=1e-12)


def make_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        cbiol=rng.uniform(0.0, 0.5, (n, 12)),
        cbiow=rng.uniform(0.0, 20.0, (n, 12)),
        cbior=rng.uniform(0.0, 1.0, (n, 12)),
        exist=rng.choice([0.0, 1.0], size=(n, 12)).astype(np.float64),
        aynpp=rng.uniform(0.0, 0.8, (n, 12)),
        firefac=rng.uniform(0.0, 1.0, n),
        totlit=rng.uniform(0.0, 1.0, n),
        gdd0=rng.uniform(0.0, 3000.0, n),
        gdd5=rng.uniform(0.0, 2500.0, n),
        ayneetot=rng.uniform(-5.0, 5.0, n),
    )


def call_f77(inputs: dict, isimfire: int) -> dict:
    out = f77.wrap_dynaveg(
        isimfire,
        inputs["cbiol"], inputs["cbiow"], inputs["cbior"],
        inputs["exist"], inputs["aynpp"],
        inputs["firefac"], inputs["totlit"],
        np.asarray(PFT.aleaf, dtype=np.float64),
        np.asarray(PFT.aroot, dtype=np.float64),
        np.asarray(PFT.awood, dtype=np.float64),
        np.asarray(PFT.tauleaf, dtype=np.float64),
        np.asarray(PFT.tauroot, dtype=np.float64),
        np.asarray(PFT.tauwood0, dtype=np.float64),
        np.asarray(PFT.specla, dtype=np.float64),
        PFT.xminlai, PFT.woodnorm,
        inputs["gdd0"], inputs["gdd5"], inputs["ayneetot"],
    )
    (cbiol, cbiow, cbior, plai, biomass,
     falll, fallr, fallw, cdisturb,
     disturbf, disturbo,
     ayanpp, aynpp, ayneetot, ayanpptot,
     totlaiu, totlail, totbiou, totbiol,
     fu, fl, ztop, zbot, sai,
     sapfrac, vegtype0, tauwood) = out
    return dict(
        cbiol=np.asarray(cbiol), cbiow=np.asarray(cbiow), cbior=np.asarray(cbior),
        plai=np.asarray(plai), biomass=np.asarray(biomass),
        falll=np.asarray(falll), fallr=np.asarray(fallr), fallw=np.asarray(fallw),
        cdisturb=np.asarray(cdisturb),
        disturbf=np.asarray(disturbf), disturbo=np.asarray(disturbo),
        ayanpp=np.asarray(ayanpp), aynpp=np.asarray(aynpp),
        ayneetot=np.asarray(ayneetot), ayanpptot=np.asarray(ayanpptot),
        totlaiu=np.asarray(totlaiu), totlail=np.asarray(totlail),
        totbiou=np.asarray(totbiou), totbiol=np.asarray(totbiol),
        fu=np.asarray(fu), fl=np.asarray(fl),
        ztop=np.asarray(ztop), zbot=np.asarray(zbot),
        sai=np.asarray(sai),
        sapfrac=np.asarray(sapfrac), vegtype0=np.asarray(vegtype0),
        tauwood=np.asarray(tauwood),
    )


def call_jax(inputs: dict, isimfire: int) -> dict:
    n = inputs["cbiol"].shape[0]
    veg = create_vegetation_state(n, npft=12, nsoilay=6)
    summ = create_summary_state(n, npft=12)
    carb = create_carbon_state(n)

    veg = eqx.tree_at(
        lambda v: (v.cbiol, v.cbiow, v.cbior, v.exist,
                   v.firefac, v.gdd0, v.gdd5),
        veg,
        (jnp.asarray(inputs["cbiol"]),
         jnp.asarray(inputs["cbiow"]),
         jnp.asarray(inputs["cbior"]),
         jnp.asarray(inputs["exist"]),
         jnp.asarray(inputs["firefac"]),
         jnp.asarray(inputs["gdd0"]),
         jnp.asarray(inputs["gdd5"])),
    )
    summ = eqx.tree_at(
        lambda s: (s.aynpp, s.ayneetot),
        summ,
        (jnp.asarray(inputs["aynpp"]), jnp.asarray(inputs["ayneetot"])),
    )
    carb = eqx.tree_at(
        lambda c: c.totlit, carb, jnp.asarray(inputs["totlit"]),
    )

    new_veg, new_carb, new_summ = jax_dynaveg(
        veg, carb, summ, PFT, isimfire, EPSILON,
    )
    return dict(
        cbiol=np.asarray(new_veg.cbiol),
        cbiow=np.asarray(new_veg.cbiow),
        cbior=np.asarray(new_veg.cbior),
        plai=np.asarray(new_veg.plai),
        biomass=np.asarray(new_veg.biomass),
        falll=np.asarray(new_carb.falll),
        fallr=np.asarray(new_carb.fallr),
        fallw=np.asarray(new_carb.fallw),
        cdisturb=np.asarray(new_carb.cdisturb),
        disturbf=np.asarray(new_veg.disturbf),
        disturbo=np.asarray(new_veg.disturbo),
        ayanpp=np.asarray(new_summ.ayanpp),
        ayneetot=np.asarray(new_summ.ayneetot),
        ayanpptot=np.asarray(new_summ.ayanpptot),
        totlaiu=np.asarray(new_veg.totlaiu),
        totlail=np.asarray(new_veg.totlail),
        totbiou=np.asarray(new_veg.totbiou),
        totbiol=np.asarray(new_veg.totbiol_agg),
        fu=np.asarray(new_veg.fu),
        fl=np.asarray(new_veg.fl),
        ztop=np.asarray(new_veg.ztop),
        zbot=np.asarray(new_veg.zbot),
        sai=np.asarray(new_veg.sai),
        sapfrac=np.asarray(new_veg.sapfrac),
        vegtype0=np.asarray(new_veg.vegtype0),
    )


def _carbon_module():
    from ibis.state import carbon
    return carbon


from ibis.state.carbon import create_carbon_state


def verify_one(report: Report, label: str, n: int, seed: int,
               isimfire: int) -> None:
    inputs = make_inputs(n, seed)
    f_out = call_f77(inputs, isimfire)
    j_out = call_jax(inputs, isimfire)
    # Note: aynpp in F77 is mutated (multiplied by exist); ayanpp is what
    # F77 stores. JAX returns updated ayanpp but leaves aynpp untouched.
    # We compare the things both sides expose.
    fields = [
        "cbiol", "cbiow", "cbior", "plai", "biomass",
        "falll", "fallr", "fallw", "cdisturb",
        "disturbf", "disturbo",
        "ayanpp", "ayneetot", "ayanpptot",
        "totlaiu", "totlail", "totbiou", "totbiol",
        "fu", "fl",
        "ztop", "zbot", "sai",
        "sapfrac", "vegtype0",
    ]
    for name in fields:
        report.add(verify_variable(
            f"dyna.{name}.{label}", f_out[name], j_out[name],
            SPEC_PREC, abs_floor=1e-6,
        ))


def make_edge_inputs(n: int, seed: int) -> dict:
    """Inputs designed to exercise vegmap edge branches.

    Cover:
      - low cbiol + low aleaf-like NPP -> treelai < 0.5 with tree dominance
        (grass/shrub branches inside each domtree=1..5 case + domtree=0)
      - very low gdd5 -> tundra/polar desert
      - very low gdd0 -> polar desert
      - totlai < 0.4 -> desert
    """
    rng = np.random.default_rng(seed)
    # Very sparse forests with low cbiol so treelai stays below 0.5 after
    # min-LAI maintenance. Use one tree PFT slightly higher than others to
    # set domtree, but small enough that treelai stays low.
    cbiol = np.full((n, 12), 0.001)
    cbiow = np.full((n, 12), 0.5)
    cbior = np.full((n, 12), 0.01)
    # Each row: rotate which tree PFT (1..8 in 1-indexed = 0..7) is dominant
    for i in range(n):
        cbiol[i, i % 8] = 0.01     # slightly higher tree PFT
        cbiol[i, 8 + (i % 4)] += 0.005  # shrubs/grasses tiny
    return dict(
        cbiol=cbiol, cbiow=cbiow, cbior=cbior,
        exist=np.ones((n, 12)),
        aynpp=np.full((n, 12), 0.001),
        firefac=rng.uniform(0.0, 1.0, n),
        totlit=rng.uniform(0.0, 1.0, n),
        gdd0=np.concatenate([
            rng.uniform(50.0, 90.0, n // 4),         # polar desert
            rng.uniform(150.0, 300.0, n // 4),       # tundra range
            rng.uniform(400.0, 1500.0, n // 4),
            rng.uniform(1500.0, 3000.0, n - 3 * (n // 4)),
        ]),
        gdd5=np.concatenate([
            rng.uniform(0.0, 200.0, n // 4),         # tundra
            rng.uniform(200.0, 349.0, n // 4),       # tundra/polar
            rng.uniform(351.0, 1200.0, n // 4),
            rng.uniform(1200.0, 2500.0, n - 3 * (n // 4)),
        ]),
        ayneetot=rng.uniform(-5.0, 5.0, n),
    )


def main() -> int:
    rep = Report(section="vegetation_dynamics")
    verify_one(rep, "rand_nofire", n=1500, seed=42, isimfire=0)
    verify_one(rep, "rand_fire",   n=1500, seed=99, isimfire=1)
    # Edge case for vegmap branches (sparse forests + climate extremes).
    inputs = make_edge_inputs(n=400, seed=7)
    f_out = call_f77(inputs, isimfire=0)
    j_out = call_jax(inputs, isimfire=0)
    fields = ["cbiol", "biomass", "plai", "fu", "fl",
              "totlaiu", "totlail", "vegtype0", "sai", "ztop"]
    for name in fields:
        rep.add(verify_variable(
            f"dyna.{name}.edge", f_out[name], j_out[name],
            SPEC_PREC, abs_floor=1e-6,
        ))
    rep.print_table()
    rep.save(Path(__file__).parent / "results")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
