"""Verify JAX stomata against F77 stomata (Farquhar / Ball-Berry).

Both sides use the F77 default parameter values (alpha3=0.06, theta3=0.97,
F77 coefm/coefb constants from physiology.f comments). The JAX defaults
differ slightly (CMA-ES calibrated) — we override to match.

JAX's coefm/coefb arrays are ordered [ub, uc, ls, C4grass, C3grass] while
F77's separate variables `coefml3`/`coefml4` use the natural PFT order
[C3, C4]. We swap when populating JAX arrays.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.stomata.verify_stomata
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
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "stomata"),
)
import stomata_wrap as f77

from ibis.physics.stomata import StomataParams, stomata as jax_stomata

from verify.framework.harness import Report, verify_variable
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


# F77 default values (from physiology.f comments + tested via the wrapper).
F77_PARAMS = dict(
    tau15=4500.0, kc15=1.5e-4, ko15=0.25, cimax=2000.0e-6,
    o2conc=0.209,
    alpha3=0.060, theta3=0.970, beta3=0.990,
    alpha4=0.050, theta4=0.970, beta4=0.800,
    gammaub=0.0150, coefmub=10.0, coefbub=0.010, gsubmin=0.00001,
    gammauc=0.0150, coefmuc= 6.0, coefbuc=0.010, gsucmin=0.00001,
    gammals=0.0150, coefmls= 9.0, coefbls=0.010, gslsmin=0.00001,
    gammal3=0.0150, coefml3= 9.0, coefbl3=0.010, gsl3min=0.00001,
    gammal4=0.0300, coefml4= 4.0, coefbl4=0.040, gsl4min=0.00001,
    vmax_pft=np.array([65e-6, 65e-6, 40e-6, 30e-6, 30e-6, 25e-6,
                       30e-6, 30e-6, 27.5e-6, 27.5e-6, 15e-6, 25e-6],
                       dtype=np.float64),
)
EPSILON = 1.0e-7
DTIME   = 3600.0
CO2CONC = 350.0e-6


# JAX StomataParams with F77 defaults. JAX coefm/coefb order = [ub, uc, ls, C4, C3].
JAX_PARAMS = StomataParams(
    tau15=F77_PARAMS["tau15"], kc15=F77_PARAMS["kc15"], ko15=F77_PARAMS["ko15"],
    cimax=F77_PARAMS["cimax"], o2conc=F77_PARAMS["o2conc"],
    alpha3=F77_PARAMS["alpha3"], theta3=F77_PARAMS["theta3"], beta3=F77_PARAMS["beta3"],
    alpha4=F77_PARAMS["alpha4"], theta4=F77_PARAMS["theta4"], beta4=F77_PARAMS["beta4"],
    gamma=jnp.array([F77_PARAMS["gammaub"], F77_PARAMS["gammauc"],
                     F77_PARAMS["gammals"], F77_PARAMS["gammal4"],
                     F77_PARAMS["gammal3"]]),
    coefm=jnp.array([F77_PARAMS["coefmub"], F77_PARAMS["coefmuc"],
                     F77_PARAMS["coefmls"], F77_PARAMS["coefml4"],
                     F77_PARAMS["coefml3"]]),
    coefb=jnp.array([F77_PARAMS["coefbub"], F77_PARAMS["coefbuc"],
                     F77_PARAMS["coefbls"], F77_PARAMS["coefbl4"],
                     F77_PARAMS["coefbl3"]]),
    gsmin=jnp.array([F77_PARAMS["gsubmin"], F77_PARAMS["gsucmin"],
                     F77_PARAMS["gslsmin"], F77_PARAMS["gsl4min"],
                     F77_PARAMS["gsl3min"]]),
    vmax_pft=jnp.array(F77_PARAMS["vmax_pft"]),
)


SPEC = ToleranceSpec("v", TolClass.PRECISION,
                      abs_tol=1e-7, rel_tol=1e-7, bias_tol=1e-8,
                      why="Stomata is a chain of Farquhar (2 colimit solves) -> "
                          "Ball-Berry (implicit relaxation) -> canopy scaling -> "
                          "wet-fraction adjustment. Each step accumulates FP "
                          "rounding; final canopy-scaled outputs (agc/anc) and "
                          "total conductances reach ~1e-8 abs / ~1e-7 rel.")


def make_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        tu=rng.uniform(275.0, 305.0, n),
        tl=rng.uniform(275.0, 305.0, n),
        t12=rng.uniform(275.0, 305.0, n),
        t34=rng.uniform(275.0, 305.0, n),
        q12=rng.uniform(0.003, 0.020, n),
        q34=rng.uniform(0.003, 0.020, n),
        su=rng.uniform(0.05, 0.5, n),
        sl=rng.uniform(0.05, 0.5, n),
        psurf=rng.uniform(80000.0, 102000.0, n),
        topparu=rng.uniform(20.0, 800.0, n),
        topparl=rng.uniform(20.0, 400.0, n),
        termu=rng.uniform(-2.0, 2.0, (n, 7)),
        terml=rng.uniform(-2.0, 2.0, (n, 7)),
        scalcoefu=rng.uniform(0.1, 5.0, (n, 4)),
        scalcoefl=rng.uniform(0.1, 5.0, (n, 4)),
        ciub=np.full(n, 270.0e-6),
        ciuc=np.full(n, 270.0e-6),
        csub=np.full(n, 350.0e-6),
        csuc=np.full(n, 350.0e-6),
        gsub=np.full(n, 0.1),
        gsuc=np.full(n, 0.1),
        cils=np.full(n, 270.0e-6),
        cil3=np.full(n, 270.0e-6),
        cil4=np.full(n, 270.0e-6),
        csls=np.full(n, 350.0e-6),
        csl3=np.full(n, 350.0e-6),
        csl4=np.full(n, 350.0e-6),
        gsls=np.full(n, 0.1),
        gsl3=np.full(n, 0.1),
        gsl4=np.full(n, 0.1),
        fwetu=rng.uniform(0.0, 0.2, n),
        fwetl=rng.uniform(0.0, 0.2, n),
        lai=rng.uniform(0.5, 5.0, (n, 2)),
        sai=rng.uniform(0.0, 1.5, (n, 2)),
        exist=rng.choice([0.0, 1.0], size=(n, 12)).astype(np.float64),
        stresstu=rng.uniform(0.2, 1.0, n),
        stresstl=rng.uniform(0.2, 1.0, n),
        a10scalparamu=rng.uniform(0.5, 5.0, n),
        a10daylightu=rng.uniform(50.0, 500.0, n),
        a10scalparaml=rng.uniform(0.5, 5.0, n),
        a10daylightl=rng.uniform(50.0, 500.0, n),
    )


def call_f77(inp: dict) -> dict:
    out = f77.wrap_stomata(
        inp["tu"], inp["tl"], inp["t12"], inp["t34"], inp["q12"], inp["q34"],
        inp["su"], inp["sl"], inp["psurf"],
        inp["topparu"], inp["topparl"],
        inp["termu"], inp["terml"],
        inp["scalcoefu"], inp["scalcoefl"],
        inp["ciub"], inp["ciuc"], inp["csub"], inp["csuc"],
        inp["gsub"], inp["gsuc"],
        inp["cils"], inp["cil3"], inp["cil4"],
        inp["csls"], inp["csl3"], inp["csl4"],
        inp["gsls"], inp["gsl3"], inp["gsl4"],
        inp["fwetu"], inp["fwetl"],
        inp["lai"], inp["sai"], inp["exist"],
        inp["stresstu"], inp["stresstl"],
        inp["a10scalparamu"], inp["a10daylightu"],
        inp["a10scalparaml"], inp["a10daylightl"],
        CO2CONC, F77_PARAMS["o2conc"], DTIME,
        F77_PARAMS["tau15"], F77_PARAMS["kc15"], F77_PARAMS["ko15"], F77_PARAMS["cimax"],
        F77_PARAMS["alpha3"], F77_PARAMS["theta3"], F77_PARAMS["beta3"],
        F77_PARAMS["alpha4"], F77_PARAMS["theta4"], F77_PARAMS["beta4"],
        F77_PARAMS["vmax_pft"],
        F77_PARAMS["gammaub"], F77_PARAMS["coefmub"], F77_PARAMS["coefbub"], F77_PARAMS["gsubmin"],
        F77_PARAMS["gammauc"], F77_PARAMS["coefmuc"], F77_PARAMS["coefbuc"], F77_PARAMS["gsucmin"],
        F77_PARAMS["gammals"], F77_PARAMS["coefmls"], F77_PARAMS["coefbls"], F77_PARAMS["gslsmin"],
        F77_PARAMS["gammal3"], F77_PARAMS["coefml3"], F77_PARAMS["coefbl3"], F77_PARAMS["gsl3min"],
        F77_PARAMS["gammal4"], F77_PARAMS["coefml4"], F77_PARAMS["coefbl4"], F77_PARAMS["gsl4min"],
        EPSILON,
    )
    keys = ("ciub", "ciuc", "csub", "csuc", "gsub", "gsuc",
            "cils", "cil3", "cil4", "csls", "csl3", "csl4",
            "gsls", "gsl3", "gsl4",
            "agcub", "agcuc", "agcls", "agcl3", "agcl4",
            "ancub", "ancuc", "ancls", "ancl3", "ancl4",
            "totcondub", "totconduc", "totcondls", "totcondl3", "totcondl4",
            "a10scalparamu", "a10daylightu",
            "a10scalparaml", "a10daylightl")
    return {k: np.asarray(v) for k, v in zip(keys, out)}


def call_jax(inp: dict) -> dict:
    n = inp["tu"].shape[0]
    ci_upper = jnp.stack([jnp.asarray(inp["ciub"]), jnp.asarray(inp["ciuc"])], axis=1)
    gs_upper = jnp.stack([jnp.asarray(inp["gsub"]), jnp.asarray(inp["gsuc"])], axis=1)
    cs_upper = jnp.stack([jnp.asarray(inp["csub"]), jnp.asarray(inp["csuc"])], axis=1)
    # JAX ci_lower/gs_lower/cs_lower order: [shrub, C3grass, C4grass] per
    # the docstring (and stomata internals use [:,0]=ls, [:,1]=l3, [:,2]=l4).
    # The params arrays coefm/coefb are different: [ub, uc, ls, C4, C3].
    ci_lower = jnp.stack([jnp.asarray(inp["cils"]), jnp.asarray(inp["cil3"]),
                          jnp.asarray(inp["cil4"])], axis=1)
    gs_lower = jnp.stack([jnp.asarray(inp["gsls"]), jnp.asarray(inp["gsl3"]),
                          jnp.asarray(inp["gsl4"])], axis=1)
    cs_lower = jnp.stack([jnp.asarray(inp["csls"]), jnp.asarray(inp["csl3"]),
                          jnp.asarray(inp["csl4"])], axis=1)

    res = jax_stomata(
        tu=jnp.asarray(inp["tu"]), tl=jnp.asarray(inp["tl"]),
        t12=jnp.asarray(inp["t12"]), t34=jnp.asarray(inp["t34"]),
        q12=jnp.asarray(inp["q12"]), q34=jnp.asarray(inp["q34"]),
        su=jnp.asarray(inp["su"]), sl=jnp.asarray(inp["sl"]),
        psurf=jnp.asarray(inp["psurf"]),
        topparu=jnp.asarray(inp["topparu"]),
        topparl=jnp.asarray(inp["topparl"]),
        termu6=jnp.asarray(inp["termu"])[:, 5], termu7=jnp.asarray(inp["termu"])[:, 6],
        terml6=jnp.asarray(inp["terml"])[:, 5], terml7=jnp.asarray(inp["terml"])[:, 6],
        scalcoefu=jnp.asarray(inp["scalcoefu"]),
        scalcoefl=jnp.asarray(inp["scalcoefl"]),
        ci_upper=ci_upper, gs_upper=gs_upper, cs_upper=cs_upper,
        ci_lower=ci_lower, gs_lower=gs_lower, cs_lower=cs_lower,
        fwetu=jnp.asarray(inp["fwetu"]), fwetl=jnp.asarray(inp["fwetl"]),
        lai=jnp.asarray(inp["lai"]), sai=jnp.asarray(inp["sai"]),
        exist=jnp.asarray(inp["exist"]),
        stresstu=jnp.asarray(inp["stresstu"]),
        stresstl=jnp.asarray(inp["stresstl"]),
        a10scalparamu=jnp.asarray(inp["a10scalparamu"]),
        a10daylightu=jnp.asarray(inp["a10daylightu"]),
        a10scalparaml=jnp.asarray(inp["a10scalparaml"]),
        a10daylightl=jnp.asarray(inp["a10daylightl"]),
        co2conc=CO2CONC, dtime=DTIME, params=JAX_PARAMS, epsilon=EPSILON,
    )
    return dict(
        ciub=np.asarray(res.ci_upper[:, 0]),
        ciuc=np.asarray(res.ci_upper[:, 1]),
        csub=np.asarray(res.cs_upper[:, 0]),
        csuc=np.asarray(res.cs_upper[:, 1]),
        gsub=np.asarray(res.gs_upper[:, 0]),
        gsuc=np.asarray(res.gs_upper[:, 1]),
        cils=np.asarray(res.ci_lower[:, 0]),
        cil3=np.asarray(res.ci_lower[:, 1]),
        cil4=np.asarray(res.ci_lower[:, 2]),
        csls=np.asarray(res.cs_lower[:, 0]),
        csl3=np.asarray(res.cs_lower[:, 1]),
        csl4=np.asarray(res.cs_lower[:, 2]),
        gsls=np.asarray(res.gs_lower[:, 0]),
        gsl3=np.asarray(res.gs_lower[:, 1]),
        gsl4=np.asarray(res.gs_lower[:, 2]),
        agcub=np.asarray(res.ag_upper[:, 0]),
        agcuc=np.asarray(res.ag_upper[:, 1]),
        agcls=np.asarray(res.ag_lower[:, 0]),
        agcl3=np.asarray(res.ag_lower[:, 1]),
        agcl4=np.asarray(res.ag_lower[:, 2]),
        ancub=np.asarray(res.an_upper[:, 0]),
        ancuc=np.asarray(res.an_upper[:, 1]),
        ancls=np.asarray(res.an_lower[:, 0]),
        ancl3=np.asarray(res.an_lower[:, 1]),
        ancl4=np.asarray(res.an_lower[:, 2]),
        totcondub=np.asarray(res.totcondub),
        totconduc=np.asarray(res.totconduc),
        totcondls=np.asarray(res.totcondls),
        totcondl3=np.asarray(res.totcondl3),
        totcondl4=np.asarray(res.totcondl4),
        a10scalparamu=np.asarray(res.a10scalparamu),
        a10daylightu=np.asarray(res.a10daylightu),
        a10scalparaml=np.asarray(res.a10scalparaml),
        a10daylightl=np.asarray(res.a10daylightl),
    )


def verify_one(report: Report, label: str, n: int, seed: int) -> None:
    inputs = make_inputs(n, seed)
    f_out = call_f77(inputs)
    j_out = call_jax(inputs)
    fields = list(f_out.keys())
    for name in fields:
        report.add(verify_variable(
            f"stom.{name}.{label}", f_out[name], j_out[name], SPEC,
            abs_floor=1e-10,
        ))


def main() -> int:
    rep = Report(section="stomata")
    verify_one(rep, "rand1", n=500, seed=42)
    verify_one(rep, "rand2", n=500, seed=99)
    rep.print_table()
    rep.save(Path(__file__).parent / "results")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
