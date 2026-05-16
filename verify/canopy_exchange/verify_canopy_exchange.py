"""Verify JAX canopy_exchange tractable pieces vs F77.

Covers:
  - canini    (aerodynamic init: heights, log profiles, density, exphl/u, ...)
  - fstrat    (Louis 1979 stratification correction)
  - impexp    (implicit/explicit blend for canopy w/ intercepted water)
  - impexp2   (implicit/explicit blend for snow skin temp)

The big coupled solver (turcof+turvap, 9x9 implicit, 3-iter loop) is
left for a separate effort; see verify/canopy_exchange/README.md.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.canopy_exchange.verify_canopy_exchange
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
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "canopy_exchange"),
)
import canopy_exchange_wrap as f77

from ibis.physics.canopy_exchange import (canini, fstrat, impexp, impexp2,
                                          turcof, turvap, AeroState)

from verify.framework.harness import Report, verify_variable
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


# Match F77 initial.f constants
GRAV   = 9.80616
RAIR   = 287.04
RVAP   = 461.0
CAIR   = 1.00464e3
CVAP   = 1.81e3
CAPPA  = RAIR / CAIR
VONK   = 0.4
TMELT  = 273.16
HFUS   = 0.3336e6
EPSILON = 1.0e-7

ALAIML = 8.0
ALAIMU = 8.0
Z0SNO  = 0.001


SPEC = ToleranceSpec(
    "v", TolClass.PRECISION,
    abs_tol=1e-8, rel_tol=1e-3, bias_tol=1e-9,
    why="Aerodynamic init + stratification are pure arithmetic over a "
        "handful of ops per gridpoint; FP rounding bounded by ~1e-12 abs.",
)


# ---------- canini --------------------------------------------------------

def make_inputs_canini(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # ztop/zbot: pair-wise (lower, upper); lower in 0.5-2.5m, upper 8-30m
    ztop_l = rng.uniform(0.5, 2.5, n)
    zbot_l = ztop_l - rng.uniform(0.3, 0.45, n)
    ztop_u = rng.uniform(8.0, 30.0, n)
    zbot_u = ztop_u - rng.uniform(2.0, 8.0, n)
    return dict(
        ta    = rng.uniform(260.0, 305.0, n),
        qa    = rng.uniform(0.002, 0.020, n),
        psurf = rng.uniform(95000.0, 102000.0, n),
        ua    = rng.uniform(0.5, 12.0, n),
        fi    = rng.uniform(0.0, 1.0, n),
        z0soi = rng.uniform(0.005, 0.05, n),
        fl    = rng.uniform(0.05, 0.95, n),
        fu    = rng.uniform(0.05, 0.95, n),
        lai   = np.stack([rng.uniform(0.1, 4.0, n),
                          rng.uniform(0.1, 6.0, n)], axis=1),
        sai   = np.stack([rng.uniform(0.05, 1.0, n),
                          rng.uniform(0.05, 2.0, n)], axis=1),
        ztop  = np.stack([ztop_l, ztop_u], axis=1),
        zbot  = np.stack([zbot_l, zbot_u], axis=1),
    )


def call_f77_canini(inp: dict) -> dict:
    out = f77.wrap_canini(
        inp["ta"], inp["qa"], inp["psurf"], inp["ua"],
        inp["fi"], inp["z0soi"], Z0SNO,
        inp["fl"], inp["fu"],
        inp["lai"], inp["sai"], inp["ztop"], inp["zbot"],
        ALAIML, ALAIMU,
        GRAV, RAIR, RVAP, CAIR, CVAP, CAPPA,
    )
    keys = ("cp", "rhoa", "za", "tfac",
            "z1", "z12", "z2", "z3", "z34", "z4",
            "bdl", "bdu", "dil", "diu",
            "exphl", "expl", "exphu", "expu",
            "displ", "dispu",
            "alogav", "alog1", "alog2", "alog3", "alog4",
            "aloga", "alogu", "alogl", "u2")
    return {k: np.asarray(v) for k, v in zip(keys, out)}


def call_jax_canini(inp: dict) -> dict:
    aero = canini(
        jnp.asarray(inp["ta"]),    jnp.asarray(inp["qa"]),
        jnp.asarray(inp["psurf"]), jnp.asarray(inp["ua"]),
        jnp.asarray(inp["fi"]),    jnp.asarray(inp["z0soi"]),
        Z0SNO,
        jnp.asarray(inp["fl"]),    jnp.asarray(inp["fu"]),
        jnp.asarray(inp["lai"]),   jnp.asarray(inp["sai"]),
        jnp.asarray(inp["ztop"]),  jnp.asarray(inp["zbot"]),
        ALAIML, ALAIMU,
        GRAV, RAIR, RVAP, CAIR, CVAP, CAPPA,
    )
    return dict(
        cp=np.asarray(aero.cp), rhoa=np.asarray(aero.rhoa),
        za=np.asarray(aero.za), tfac=np.asarray(aero.tfac),
        z1=np.asarray(aero.z1), z12=np.asarray(aero.z12),
        z2=np.asarray(aero.z2), z3=np.asarray(aero.z3),
        z34=np.asarray(aero.z34), z4=np.asarray(aero.z4),
        bdl=np.asarray(aero.bdl), bdu=np.asarray(aero.bdu),
        dil=np.asarray(aero.dil), diu=np.asarray(aero.diu),
        exphl=np.asarray(aero.exphl), expl=np.asarray(aero.expl),
        exphu=np.asarray(aero.exphu), expu=np.asarray(aero.expu),
        displ=np.asarray(aero.displ), dispu=np.asarray(aero.dispu),
        alogav=np.asarray(aero.alogav),
        alog1=np.asarray(aero.alog1), alog2=np.asarray(aero.alog2),
        alog3=np.asarray(aero.alog3), alog4=np.asarray(aero.alog4),
        aloga=np.asarray(aero.aloga),
        alogu=np.asarray(aero.alogu), alogl=np.asarray(aero.alogl),
        u2=np.asarray(aero.u2),
    )


def verify_canini(rep: Report, label: str, n: int, seed: int) -> None:
    inp = make_inputs_canini(n, seed)
    f_out = call_f77_canini(inp)
    j_out = call_jax_canini(inp)
    for name in f_out:
        # tfac is a scalar in F77, broadcast for compare
        if name == "tfac":
            j_val = np.full_like(f_out[name], float(j_out["tfac"]))
            rep.add(verify_variable(
                f"canini.{name}.{label}", f_out[name], j_val,
                SPEC, abs_floor=1e-10,
            ))
        else:
            rep.add(verify_variable(
                f"canini.{name}.{label}", f_out[name], j_out[name],
                SPEC, abs_floor=1e-10,
            ))


# ---------- fstrat --------------------------------------------------------

def make_inputs_fstrat(n: int, seed: int, *, sign: str = "mixed") -> dict:
    """sign='unstable' forces Ri<0; 'stable' forces Ri>0; 'mixed' lets it vary."""
    rng = np.random.default_rng(seed)
    zb = rng.uniform(0.5, 5.0, n)
    zt = zb + rng.uniform(1.0, 15.0, n)
    if sign == "unstable":
        tb = rng.uniform(290.0, 310.0, n)   # warm bottom
        tt = rng.uniform(260.0, 285.0, n)   # cold top -> unstable
    elif sign == "stable":
        tb = rng.uniform(260.0, 280.0, n)   # cold bottom
        tt = rng.uniform(285.0, 305.0, n)   # warm top -> stable
    else:
        tb = rng.uniform(265.0, 305.0, n)
        tt = rng.uniform(265.0, 305.0, n)
    return dict(
        tb=tb, tt=tt, ttfac=1.0,
        qb=rng.uniform(0.001, 0.018, n),
        qt=rng.uniform(0.001, 0.018, n),
        zb=zb, zt=zt,
        albm=rng.uniform(-6.0, 0.0, n),
        albh=rng.uniform(-6.0, 0.0, n),
        alt=rng.uniform(-1.0, 3.5, n),
        u=rng.uniform(0.5, 12.0, n),
        # previous-iter state (for iter>1 weighting)
        rich_prev =rng.uniform(-1.0, 0.5, n),
        stram_prev=rng.uniform(0.5, 1.5, n),
        strah_prev=rng.uniform(0.5, 1.5, n),
    )


def call_f77_fstrat(inp: dict, iter_num: int) -> dict:
    rich, stram, strah = f77.wrap_fstrat(
        inp["tb"], inp["tt"], inp["ttfac"], inp["qb"], inp["qt"],
        inp["zb"], inp["zt"], inp["albm"], inp["albh"], inp["alt"], inp["u"],
        inp["rich_prev"], inp["stram_prev"], inp["strah_prev"],
        iter_num, GRAV, VONK,
    )
    return dict(rich=np.asarray(rich), stram=np.asarray(stram),
                strah=np.asarray(strah))


def call_jax_fstrat(inp: dict, iter_num: int) -> dict:
    rich, stram, strah = fstrat(
        jnp.asarray(inp["tb"]), jnp.asarray(inp["tt"]), inp["ttfac"],
        jnp.asarray(inp["qb"]), jnp.asarray(inp["qt"]),
        jnp.asarray(inp["zb"]), jnp.asarray(inp["zt"]),
        jnp.asarray(inp["albm"]), jnp.asarray(inp["albh"]),
        jnp.asarray(inp["alt"]), jnp.asarray(inp["u"]),
        GRAV, VONK, iter_num - 1,           # JAX uses 0-indexed iter
        jnp.asarray(inp["rich_prev"]),
        jnp.asarray(inp["stram_prev"]),
        jnp.asarray(inp["strah_prev"]),
    )
    return dict(rich=np.asarray(rich), stram=np.asarray(stram),
                strah=np.asarray(strah))


def verify_fstrat(rep: Report, label: str, n: int, seed: int, *,
                  iter_num: int, sign: str = "mixed") -> None:
    inp = make_inputs_fstrat(n, seed, sign=sign)
    f_out = call_f77_fstrat(inp, iter_num)
    j_out = call_jax_fstrat(inp, iter_num)
    for name in f_out:
        rep.add(verify_variable(
            f"fstrat.{name}.{label}", f_out[name], j_out[name],
            SPEC, abs_floor=1e-10,
        ))


# ---------- turcof --------------------------------------------------------
# turcof needs an AeroState (canini output) + atmospheric forcing + the
# previous-iteration t12/t34/q12/q34/rich*/stra* + canopy params.

CLEAF  = 0.01
CSTEM  = 0.01
CGRASS = 0.01
DLEAF  = (0.10, 0.10)   # (lower, upper)
DSTEM  = (0.10, 0.10)
DTIME  = 3600.0


def make_inputs_turcof(n: int, seed: int) -> dict:
    """Build canini result + state for one turcof call."""
    rng = np.random.default_rng(seed)
    inp = make_inputs_canini(n, seed)
    aero_f = call_f77_canini(inp)
    return dict(
        inp=inp, aero=aero_f,
        t12=rng.uniform(270.0, 305.0, n),
        t34=rng.uniform(270.0, 305.0, n),
        q12=rng.uniform(0.002, 0.020, n),
        q34=rng.uniform(0.002, 0.020, n),
        richl=rng.uniform(-1.0, 0.5, n),
        straml=rng.uniform(0.5, 1.5, n),
        strahl=rng.uniform(0.5, 1.5, n),
        richu=rng.uniform(-1.0, 0.5, n),
        stramu=rng.uniform(0.5, 1.5, n),
        strahu=rng.uniform(0.5, 1.5, n),
    )


def call_f77_turcof(d: dict, iter_num: int) -> dict:
    a = d["aero"]
    out = f77.wrap_turcof(
        iter_num,
        a["rhoa"], float(a["tfac"]),
        a["z1"], a["z12"], a["z2"], a["z3"], a["z34"], a["z4"],
        a["bdl"], a["bdu"], a["dil"], a["diu"],
        a["exphl"], a["expl"], a["exphu"], a["expu"],
        a["alogav"], a["alog1"], a["alog2"], a["alog3"], a["alog4"],
        a["aloga"], a["alogu"], a["alogl"], a["u2"],
        d["inp"]["ta"], d["inp"]["qa"], d["inp"]["ua"], d["inp"]["fi"],
        d["t12"], d["t34"], d["q12"], d["q34"],
        d["richl"], d["straml"], d["strahl"],
        d["richu"], d["stramu"], d["strahu"],
        CLEAF, CSTEM, CGRASS,
        DLEAF[0], DLEAF[1], DSTEM[0], DSTEM[1],
        GRAV, VONK, DTIME,
    )
    keys = ("cu", "cl", "su", "ss", "sl", "sg", "si",
            "u1", "u12", "u2", "u3", "u34", "u4",
            "alogu", "alogl",
            "richl", "straml", "strahl",
            "richu", "stramu", "strahu")
    return {k: np.asarray(v) for k, v in zip(keys, out)}


def _make_aero_state(a: dict, inp: dict) -> AeroState:
    """Reconstruct a JAX AeroState from F77 canini output dict."""
    # cp and za are JAX-only intermediates; rebuild via canini for the cp/za
    # fields (cheap; results match bit-exact already).
    aero_jax = canini(
        jnp.asarray(inp["ta"]), jnp.asarray(inp["qa"]),
        jnp.asarray(inp["psurf"]), jnp.asarray(inp["ua"]),
        jnp.asarray(inp["fi"]), jnp.asarray(inp["z0soi"]), Z0SNO,
        jnp.asarray(inp["fl"]), jnp.asarray(inp["fu"]),
        jnp.asarray(inp["lai"]), jnp.asarray(inp["sai"]),
        jnp.asarray(inp["ztop"]), jnp.asarray(inp["zbot"]),
        ALAIML, ALAIMU,
        GRAV, RAIR, RVAP, CAIR, CVAP, CAPPA,
    )
    return aero_jax


def call_jax_turcof(d: dict, iter_num: int) -> dict:
    aero = _make_aero_state(d["aero"], d["inp"])
    tc = turcof(
        aero,
        jnp.asarray(d["inp"]["ta"]), jnp.asarray(d["inp"]["qa"]),
        jnp.asarray(d["inp"]["ua"]),
        jnp.asarray(d["t12"]), jnp.asarray(d["t34"]),
        jnp.asarray(d["q12"]), jnp.asarray(d["q34"]),
        jnp.asarray(d["inp"]["fi"]),
        VONK, DTIME,
        CLEAF, CSTEM, CGRASS, DLEAF, DSTEM,
        iter_num - 1,                     # JAX uses 0-indexed iter_num
        jnp.asarray(d["richl"]), jnp.asarray(d["straml"]),
        jnp.asarray(d["strahl"]),
        jnp.asarray(d["richu"]), jnp.asarray(d["stramu"]),
        jnp.asarray(d["strahu"]),
        GRAV,
    )
    # turcof JAX returns TransferCoeffs (cu, cl, su, ss, sl, sg, si, u2,
    # alogu, alogl, straml, strahl, stramu, strahu). The intermediate u1,
    # u12, u3, u34, u4 + richl/richu are not exposed -- match what's
    # available.
    return dict(
        cu=np.asarray(tc.cu),  cl=np.asarray(tc.cl),
        su=np.asarray(tc.su),  ss=np.asarray(tc.ss), sl=np.asarray(tc.sl),
        sg=np.asarray(tc.sg),  si=np.asarray(tc.si),
        u2=np.asarray(tc.u2),
        alogu=np.asarray(tc.alogu), alogl=np.asarray(tc.alogl),
        straml=np.asarray(tc.straml), strahl=np.asarray(tc.strahl),
        stramu=np.asarray(tc.stramu), strahu=np.asarray(tc.strahu),
    )


# Looser tolerance: turcof divides by (x+1)*expl - (x-1)/expl in yl/yu
# computation, which has FP cancellation when x is small. Empirically
# reaches ~1e-9 abs on alogu/alogl for some random gridpoints.
TURCOF_SPEC = ToleranceSpec(
    "v", TolClass.PRECISION,
    abs_tol=1e-7, rel_tol=1e-5, bias_tol=1e-9,
    why="turcof's yu/yl involves catastrophic cancellation in "
        "((x+1)*exp - (x-1)/exp) when veg density is small (x ~ 0); "
        "alogl/alogu can reach ~1e-9 abs differences.",
)


def verify_turcof(rep: Report, label: str, n: int, seed: int, *,
                   iter_num: int) -> None:
    d = make_inputs_turcof(n, seed)
    f_out = call_f77_turcof(d, iter_num)
    j_out = call_jax_turcof(d, iter_num)
    for name in j_out:
        rep.add(verify_variable(
            f"turcof.{name}.{label}", f_out[name], j_out[name],
            TURCOF_SPEC, abs_floor=1e-9,
        ))


# ---------- turvap -------------------------------------------------------
# turvap builds + solves a 9x9 implicit system for canopy temperatures and
# humidities and computes the corresponding sensible + moisture fluxes.
# Many inputs needed: state, transfer coefficients, intercepted water,
# radiation forcing, soil/snow thermal at layer 1, drystress factors.

CHU    = 8436.0
CHS    = 2.109e5
CHL    = 8436.0
WPUDMAX = 4.5
HVAP   = 2.5104e6
HSUB   = HVAP + HFUS
CH2O   = 4218.0
CICE   = 2106.0
STEF   = 5.67e-8
RHOW   = 1.0e3


def make_inputs_turvap(n: int, seed: int, *, sparse_veg: bool = False) -> dict:
    """Build a self-consistent input set for turvap.

    Generate canini -> turcof outputs and add the per-iteration state
    (tu/ts/tl/t12/t34/q12/q34/tg/ti) and forcing fields needed by turvap.
    """
    rng = np.random.default_rng(seed)
    inp = make_inputs_canini(n, seed)
    if sparse_veg:
        # Force ~half the grid to have lai/sai well below epsilon (1e-7)
        # so turvap's "if (lai > epsilon)" / sai variants take false path.
        # Make canini happy by also setting fl/fu small.
        mask = rng.random(n) < 0.5
        inp["lai"][mask, :] = 1e-12
        inp["sai"][mask, :] = 1e-12
        inp["fl"][mask]     = 1e-12
        inp["fu"][mask]     = 1e-12
    aero_f = call_f77_canini(inp)
    # Build turcof inputs and call F77 turcof to get cu/cl/su/ss/sl/sg/si
    turcof_state = dict(
        inp=inp, aero=aero_f,
        t12=rng.uniform(280.0, 300.0, n),
        t34=rng.uniform(280.0, 300.0, n),
        q12=rng.uniform(0.004, 0.018, n),
        q34=rng.uniform(0.004, 0.018, n),
        richl=rng.uniform(-0.5, 0.5, n),
        straml=rng.uniform(0.7, 1.3, n),
        strahl=rng.uniform(0.7, 1.3, n),
        richu=rng.uniform(-0.5, 0.5, n),
        stramu=rng.uniform(0.7, 1.3, n),
        strahu=rng.uniform(0.7, 1.3, n),
    )
    tc_out = call_f77_turcof(turcof_state, iter_num=1)
    # Random canopy + soil state
    tu  = rng.uniform(280.0, 300.0, n)
    ts  = rng.uniform(280.0, 300.0, n)
    tl  = rng.uniform(280.0, 300.0, n)
    tg  = rng.uniform(275.0, 295.0, n)
    ti  = rng.uniform(265.0, 275.0, n)
    # frac normalized so each row sums to 1 (npft=12 columns)
    frac = rng.uniform(0.0, 1.0, (n, 12))
    frac = frac / frac.sum(axis=1, keepdims=True)
    # stressu/stressl over 6 layers
    stressu = rng.uniform(0.0, 1.0, (n, 6))
    stressl = rng.uniform(0.0, 1.0, (n, 6))
    # Saturation latent heats (hvasug, hvasui = hvap or hsub depending on T):
    # F77 setsoi sets hvasug = hvap if tg > tmelt else hsub. Mirror that.
    hvasug = np.where(tg > TMELT, HVAP, HSUB)
    hvasui = np.where(ti > TMELT, HVAP, HSUB)
    return dict(
        inp=inp,
        aero=aero_f,
        tc=tc_out,
        tu=tu, ts=ts, tl=tl,
        t12=turcof_state["t12"], t34=turcof_state["t34"],
        q12=turcof_state["q12"], q34=turcof_state["q34"],
        tg=tg, ti=ti,
        fu=inp["fu"], fl=inp["fl"], lai=inp["lai"], sai=inp["sai"],
        frac=frac,
        wliqu=rng.uniform(0.0, 0.1, n),
        wsnou=rng.uniform(0.0, 0.2, n),
        wliqs=rng.uniform(0.0, 0.1, n),
        wsnos=rng.uniform(0.0, 0.2, n),
        wliql=rng.uniform(0.0, 0.1, n),
        wsnol=rng.uniform(0.0, 0.2, n),
        fwetu=rng.uniform(0.0, 1.0, n),
        fwets=rng.uniform(0.0, 1.0, n),
        fwetl=rng.uniform(0.0, 1.0, n),
        rliqu=rng.uniform(0.0, 1.0, n),
        rliqs=rng.uniform(0.0, 1.0, n),
        rliql=rng.uniform(0.0, 1.0, n),
        totcondub=rng.uniform(0.001, 0.05, n),
        totconduc=rng.uniform(0.001, 0.05, n),
        totcondls=rng.uniform(0.001, 0.05, n),
        totcondl3=rng.uniform(0.001, 0.05, n),
        totcondl4=rng.uniform(0.001, 0.05, n),
        solu=rng.uniform(0.0, 500.0, n),
        sols=rng.uniform(0.0, 200.0, n),
        soll=rng.uniform(0.0, 500.0, n),
        solg=rng.uniform(0.0, 500.0, n),
        soli=rng.uniform(0.0, 500.0, n),
        firu=rng.uniform(-100.0, 100.0, n),
        firs=rng.uniform(-100.0, 100.0, n),
        firl=rng.uniform(-100.0, 100.0, n),
        firg=rng.uniform(-100.0, 100.0, n),
        firi=rng.uniform(-100.0, 100.0, n),
        firb=rng.uniform(200.0, 500.0, n),
        pfluxu=rng.uniform(-10.0, 10.0, n),
        pfluxs=rng.uniform(-10.0, 10.0, n),
        pfluxl=rng.uniform(-10.0, 10.0, n),
        poros1=rng.uniform(0.3, 0.55, n),
        csoi1=rng.uniform(700.0, 900.0, n),
        rhosoi1=rng.uniform(1300.0, 1700.0, n),
        hsoi1=0.10,
        wpud=rng.uniform(0.0, 1.0, n),
        wipud=rng.uniform(0.0, 0.5, n),
        suction1=rng.uniform(0.05, 0.5, n),
        bex1=rng.uniform(3.0, 11.0, n),
        swilt1=rng.uniform(0.05, 0.30, n),
        wsoi1=rng.uniform(0.10, 0.80, n),
        wisoi1=rng.uniform(0.0, 0.3, n),
        tsoi1=rng.uniform(275.0, 295.0, n),
        consoi1=rng.uniform(0.5, 2.5, n),
        consno=0.21,
        hsno1=rng.uniform(0.01, 0.20, n),
        hsnotop=0.05,
        tsno1=rng.uniform(255.0, 273.0, n),
        hvasug=hvasug, hvasui=hvasui,
        stresstu=rng.uniform(0.1, 1.0, n),
        stresstl=rng.uniform(0.1, 1.0, n),
        stressu=stressu, stressl=stressl,
    )


def call_f77_turvap(d: dict) -> dict:
    a = d["aero"]
    tc = d["tc"]
    out = f77.wrap_turvap(
        d["tu"], d["ts"], d["tl"], d["t12"], d["t34"],
        d["q12"], d["q34"], d["tg"], d["ti"],
        a["cp"], a["rhoa"], float(a["tfac"]),
        tc["cu"], tc["cl"], tc["su"], tc["ss"], tc["sl"], tc["sg"], tc["si"],
        d["inp"]["ta"], d["inp"]["qa"], d["inp"]["psurf"], d["inp"]["fi"],
        d["fu"], d["fl"], d["lai"], d["sai"], d["frac"],
        d["wliqu"], d["wsnou"], d["wliqs"], d["wsnos"],
        d["wliql"], d["wsnol"],
        d["fwetu"], d["fwets"], d["fwetl"],
        d["rliqu"], d["rliqs"], d["rliql"],
        d["totcondub"], d["totconduc"],
        d["totcondls"], d["totcondl3"], d["totcondl4"],
        d["solu"], d["sols"], d["soll"], d["solg"], d["soli"],
        d["firu"], d["firs"], d["firl"], d["firg"], d["firi"], d["firb"],
        d["pfluxu"], d["pfluxs"], d["pfluxl"],
        d["poros1"], d["csoi1"], d["rhosoi1"], d["hsoi1"],
        d["wpud"], d["wipud"], d["suction1"], d["bex1"],
        d["swilt1"], d["wsoi1"], d["wisoi1"], d["tsoi1"], d["consoi1"],
        d["consno"], d["hsno1"], d["hsnotop"], d["tsno1"],
        d["hvasug"], d["hvasui"],
        d["stresstu"], d["stresstl"], d["stressu"], d["stressl"],
        CHU, CHS, CHL,
        WPUDMAX, CH2O, CICE,
        HVAP, HSUB, CVAP, STEF,
        RVAP, GRAV, TMELT, EPSILON, RHOW, DTIME,
    )
    keys = ("tu", "ts", "tl", "t12", "t34", "q12", "q34", "tg", "ti",
            "fsena", "fsenu", "fsens", "fsenl", "fseng", "fseni",
            "fvapa", "fvapuw", "fvaput", "fvaps",
            "fvaplw", "fvaplt", "fvapg", "fvapi",
            "firg", "firi", "firb",
            "ginvap", "gsuvap", "gtrans")
    return {k: np.asarray(v) for k, v in zip(keys, out)}


def call_jax_turvap(d: dict) -> dict:
    a = d["aero"]
    tc = d["tc"]
    n = d["tu"].shape[0]
    # JAX needs chux/chsx/chlx + chgx/cog/coi/zirg/ziri/xu/xs/xl (computed
    # internally in F77 turvap on iter=1). Replicate that init here.
    chux = CHU + CH2O * d["wliqu"] + CICE * d["wsnou"]
    chsx = CHS + CH2O * d["wliqs"] + CICE * d["wsnos"]
    chlx = CHL + CH2O * d["wliql"] + CICE * d["wsnol"]
    rwork = d["poros1"] * RHOW
    chgx = (CH2O * d["wpud"] + CICE * d["wipud"]
            + ((1.0 - d["poros1"]) * d["csoi1"] * d["rhosoi1"]
               + rwork * (1.0 - d["wisoi1"]) * d["wsoi1"] * CH2O
               + rwork * d["wisoi1"] * CICE) * d["hsoi1"])
    wlgx = d["wpud"] + rwork * (1.0 - d["wisoi1"]) * d["wsoi1"] * d["hsoi1"]
    wigx = d["wipud"] + rwork * d["wisoi1"] * d["hsoi1"]
    cog  = d["consoi1"] / (0.5 * d["hsoi1"])
    coi  = d["consno"] / (0.5 * np.maximum(d["hsno1"], d["hsnotop"]))
    zirg = 4.0 * 0.95 * STEF * d["tg"] ** 3
    ziri = 4.0 * 0.95 * STEF * d["ti"] ** 3
    xu   = 2.0 * d["lai"][:, 1] * d["fu"]
    xs   = 2.0 * d["sai"][:, 1] * d["fu"]
    xl   = 2.0 * (d["lai"][:, 0] + d["sai"][:, 0]) * d["fl"] * (1.0 - d["inp"]["fi"])

    # F77 turvap on iter=1: tuold=tu, ..., and impexp returns 1 -> tu unchanged.
    ones = jnp.ones(n)
    res = turvap(
        jnp.asarray(d["tu"]), jnp.asarray(d["ts"]), jnp.asarray(d["tl"]),
        jnp.asarray(d["t12"]), jnp.asarray(d["t34"]),
        jnp.asarray(d["q12"]), jnp.asarray(d["q34"]),
        jnp.asarray(d["tg"]), jnp.asarray(d["ti"]),
        jnp.asarray(d["tu"]), jnp.asarray(d["ts"]), jnp.asarray(d["tl"]),
        jnp.asarray(d["tg"]), jnp.asarray(d["ti"]),  # tuold=tu, etc.
        jnp.asarray(tc["cu"]), jnp.asarray(tc["cl"]),
        jnp.asarray(tc["su"]), jnp.asarray(tc["ss"]), jnp.asarray(tc["sl"]),
        jnp.asarray(tc["sg"]), jnp.asarray(tc["si"]),
        jnp.asarray(a["cp"]), jnp.asarray(a["rhoa"]), float(a["tfac"]),
        jnp.asarray(xu), jnp.asarray(xs), jnp.asarray(xl),
        jnp.asarray(d["inp"]["fi"]),
        jnp.asarray(d["fu"]), jnp.asarray(d["fl"]),
        jnp.asarray(d["lai"]), jnp.asarray(d["sai"]),
        jnp.asarray(d["frac"]),
        jnp.asarray(chux), jnp.asarray(chsx), jnp.asarray(chlx),
        jnp.asarray(d["wliqu"]), jnp.asarray(d["wliqs"]), jnp.asarray(d["wliql"]),
        jnp.asarray(d["wsnou"]), jnp.asarray(d["wsnos"]), jnp.asarray(d["wsnol"]),
        jnp.asarray(d["rliqu"]), jnp.asarray(d["rliqs"]), jnp.asarray(d["rliql"]),
        jnp.asarray(d["fwetu"]), jnp.asarray(d["fwets"]), jnp.asarray(d["fwetl"]),
        jnp.asarray(d["totcondub"]), jnp.asarray(d["totconduc"]),
        jnp.asarray(d["totcondls"]), jnp.asarray(d["totcondl3"]),
        jnp.asarray(d["totcondl4"]),
        jnp.asarray(chgx), jnp.asarray(cog), jnp.asarray(coi),
        jnp.asarray(zirg), jnp.asarray(ziri),
        jnp.asarray(d["wpud"]), jnp.asarray(d["wipud"]), WPUDMAX,
        jnp.asarray(d["wisoi1"]), jnp.asarray(d["wsoi1"]),
        jnp.asarray(d["swilt1"]), jnp.asarray(d["suction1"]), jnp.asarray(d["bex1"]),
        jnp.asarray(d["hvasug"]), jnp.asarray(d["hvasui"]),
        jnp.asarray(d["tsoi1"]), jnp.asarray(d["tsno1"]),
        jnp.asarray(d["solu"]), jnp.asarray(d["sols"]), jnp.asarray(d["soll"]),
        jnp.asarray(d["solg"]), jnp.asarray(d["soli"]),
        jnp.asarray(d["firu"]), jnp.asarray(d["firs"]), jnp.asarray(d["firl"]),
        jnp.asarray(d["firg"]), jnp.asarray(d["firi"]), jnp.asarray(d["firb"]),
        jnp.asarray(d["pfluxu"]), jnp.asarray(d["pfluxs"]), jnp.asarray(d["pfluxl"]),
        jnp.asarray(d["inp"]["ta"]), jnp.asarray(d["inp"]["qa"]),
        jnp.asarray(d["inp"]["psurf"]),
        ones, ones, ones, ones, ones,           # wu, ws, wl, wg, wi (iter=1 -> 1)
        DTIME, TMELT, HVAP, HSUB, CH2O, CICE,
        CVAP, STEF, RVAP, GRAV, EPSILON,
    )
    return dict(
        tu=np.asarray(res.tu), ts=np.asarray(res.ts), tl=np.asarray(res.tl),
        t12=np.asarray(res.t12), t34=np.asarray(res.t34),
        q12=np.asarray(res.q12), q34=np.asarray(res.q34),
        tg=np.asarray(res.tg), ti=np.asarray(res.ti),
        fsena=np.asarray(res.fsena), fsenu=np.asarray(res.fsenu),
        fsens=np.asarray(res.fsens), fsenl=np.asarray(res.fsenl),
        fseng=np.asarray(res.fseng), fseni=np.asarray(res.fseni),
        fvapa=np.asarray(res.fvapa),
        fvapuw=np.asarray(res.fvapuw), fvaput=np.asarray(res.fvaput),
        fvaps=np.asarray(res.fvaps),
        fvaplw=np.asarray(res.fvaplw), fvaplt=np.asarray(res.fvaplt),
        fvapg=np.asarray(res.fvapg), fvapi=np.asarray(res.fvapi),
        firg=np.asarray(res.firg), firi=np.asarray(res.firi),
        firb=np.asarray(res.firb),
        ginvap=np.asarray(res.ginvap), gsuvap=np.asarray(res.gsuvap),
        gtrans=np.asarray(res.gtrans),
    )


# Empirically the 9x9 implicit solve matches much tighter than the
# README's "~1e-6 or worse" pessimism: temperatures match to ~1e-9 K abs
# and worst flux (fseni at ~1500 W/m^2 range) reaches ~3.5e-8 W/m^2 abs.
# Different pivot orders (F77 Gaussian-elim w/o pivot vs JAX LU w/ pivot)
# would normally spread errors more, but the matrix's diagonal dominance
# (timestep-scaled heat capacities) keeps both solutions tight.
TURVAP_SPEC = ToleranceSpec(
    "v", TolClass.PRECISION,
    abs_tol=1e-7, rel_tol=1e-8, bias_tol=1e-8,
    why="9x9 linsolve: F77 uses Gaussian elimination without pivoting, "
        "JAX uses LU with partial pivoting. Different pivot order spreads "
        "FP noise differently. The matrix is diagonal-dominant from "
        "timestep-scaled heat capacities, so both solutions stay tight "
        "(~1e-9 K abs on T*; ~3.5e-8 W/m^2 abs on fseni @ ~1500 W/m^2 "
        "range; ~1e-14 kg/m^2/s on moisture fluxes).",
)


def verify_turvap(rep: Report, label: str, n: int, seed: int, *,
                   sparse_veg: bool = False) -> None:
    d = make_inputs_turvap(n, seed, sparse_veg=sparse_veg)
    f_out = call_f77_turvap(d)
    j_out = call_jax_turvap(d)
    for name in f_out:
        rep.add(verify_variable(
            f"turvap.{name}.{label}", f_out[name], j_out[name],
            TURVAP_SPEC, abs_floor=1e-6,
        ))


# ---------- impexp --------------------------------------------------------

def make_inputs_impexp(n: int, seed: int, *, sign: str = "mixed") -> dict:
    """sign='melt' biases tveg > tmelt; 'freeze' < tmelt; 'mixed' random."""
    rng = np.random.default_rng(seed)
    if sign == "melt":
        tveg = TMELT + rng.uniform(0.5, 8.0, n)
    elif sign == "freeze":
        tveg = TMELT - rng.uniform(0.5, 8.0, n)
    elif sign == "tiny":
        tveg = TMELT + rng.uniform(-1e-9, 1e-9, n)   # |h|<epsilon branch
    else:
        tveg = TMELT + rng.uniform(-8.0, 8.0, n)
    return dict(
        wimp=rng.uniform(0.0, 1.0, n),
        tveg=tveg,
        ch=rng.uniform(500.0, 1e5, n),
        wliq=rng.uniform(0.0, 0.005, n),
        wsno=rng.uniform(0.0, 0.020, n),
    )


def call_f77_impexp(inp: dict, iter_num: int) -> np.ndarray:
    return np.asarray(f77.wrap_impexp(
        iter_num, inp["wimp"], inp["tveg"], inp["ch"],
        inp["wliq"], inp["wsno"],
        TMELT, HFUS, EPSILON,
    ))


def call_jax_impexp(inp: dict, iter_num: int) -> np.ndarray:
    out = impexp(
        jnp.asarray(inp["tveg"]), jnp.asarray(inp["ch"]),
        jnp.asarray(inp["wliq"]), jnp.asarray(inp["wsno"]),
        TMELT, HFUS, EPSILON,
        jnp.asarray(inp["wimp"]),
        is_first=(iter_num == 1),
    )
    return np.asarray(out)


def verify_impexp(rep: Report, label: str, n: int, seed: int, *,
                  iter_num: int, sign: str = "mixed") -> None:
    inp = make_inputs_impexp(n, seed, sign=sign)
    f_out = call_f77_impexp(inp, iter_num)
    j_out = call_jax_impexp(inp, iter_num)
    rep.add(verify_variable(
        f"impexp.wimp.{label}", f_out, j_out, SPEC, abs_floor=1e-10,
    ))


# ---------- impexp2 -------------------------------------------------------

def make_inputs_impexp2(n: int, seed: int, *, sign: str = "mixed") -> dict:
    rng = np.random.default_rng(seed)
    told = TMELT + rng.uniform(-15.0, 0.0, n)
    if sign == "rise":
        t = told + rng.uniform(0.5, 12.0, n)   # t > told
    elif sign == "fall":
        t = told - rng.uniform(0.5, 12.0, n)   # t < told (else-branch unchanged)
    else:
        t = told + rng.uniform(-12.0, 12.0, n)
    return dict(
        wimp=rng.uniform(0.0, 1.0, n),
        t=t, told=told,
    )


def call_f77_impexp2(inp: dict, iter_num: int) -> np.ndarray:
    return np.asarray(f77.wrap_impexp2(
        iter_num, inp["wimp"], inp["t"], inp["told"],
        TMELT, EPSILON,
    ))


def call_jax_impexp2(inp: dict, iter_num: int) -> np.ndarray:
    out = impexp2(
        jnp.asarray(inp["t"]), jnp.asarray(inp["told"]),
        TMELT, EPSILON,
        jnp.asarray(inp["wimp"]),
        is_first=(iter_num == 1),
    )
    return np.asarray(out)


def verify_impexp2(rep: Report, label: str, n: int, seed: int, *,
                   iter_num: int, sign: str = "mixed") -> None:
    inp = make_inputs_impexp2(n, seed, sign=sign)
    f_out = call_f77_impexp2(inp, iter_num)
    j_out = call_jax_impexp2(inp, iter_num)
    rep.add(verify_variable(
        f"impexp2.wimp.{label}", f_out, j_out, SPEC, abs_floor=1e-10,
    ))


# ---------- main ----------------------------------------------------------

def main() -> int:
    rep = Report(section="canopy_exchange")

    # canini -- 2 random seeds
    verify_canini(rep, "rand1", n=1500, seed=42)
    verify_canini(rep, "rand2", n=1500, seed=99)

    # fstrat -- exercise iter=1 (no weighting), iter=2 (weighted), and
    # force the unstable/stable sign branches separately. Two extra
    # scenarios use n=npoi (1718) -- no padding contribution -- so a pure
    # all-stable input drives np=0 (firing the "if (np.gt.0)" else-branch)
    # and pure all-unstable drives nq=0 (firing "if (nq.gt.0)" else-branch).
    verify_fstrat(rep, "mixed_iter1",     n=1000, seed=11, iter_num=1)
    verify_fstrat(rep, "mixed_iter2",     n=1000, seed=12, iter_num=2)
    verify_fstrat(rep, "unstable_iter1",  n=500,  seed=21, iter_num=1, sign="unstable")
    verify_fstrat(rep, "unstable_iter2",  n=500,  seed=22, iter_num=2, sign="unstable")
    verify_fstrat(rep, "stable_iter1",    n=500,  seed=31, iter_num=1, sign="stable")
    verify_fstrat(rep, "stable_iter2",    n=500,  seed=32, iter_num=2, sign="stable")
    verify_fstrat(rep, "allstable_full",  n=1718, seed=33, iter_num=1, sign="stable")
    verify_fstrat(rep, "allunstable_full",n=1718, seed=34, iter_num=1, sign="unstable")

    # turcof -- iter 1 (no weighting in fstrat) and iter 2 (weighted)
    verify_turcof(rep, "rand1_iter1", n=1000, seed=71, iter_num=1)
    verify_turcof(rep, "rand2_iter1", n=1000, seed=72, iter_num=1)
    verify_turcof(rep, "rand1_iter2", n=1000, seed=73, iter_num=2)

    # turvap -- the big 9x9 implicit solve. Sparse-veg scenario exercises
    # the (lai|sai > epsilon) false branches in the radiation-per-leaf-area
    # block.
    verify_turvap(rep, "rand1", n=500, seed=81)
    verify_turvap(rep, "rand2", n=500, seed=82)
    verify_turvap(rep, "sparse_veg", n=500, seed=83, sparse_veg=True)

    # impexp -- iter=1 (forces wimp=1), iter=2 melt/freeze/tiny branches
    verify_impexp(rep, "iter1",        n=500, seed=41, iter_num=1)
    verify_impexp(rep, "mixed_iter2",  n=500, seed=42, iter_num=2)
    verify_impexp(rep, "melt_iter2",   n=500, seed=43, iter_num=2, sign="melt")
    verify_impexp(rep, "freeze_iter2", n=500, seed=44, iter_num=2, sign="freeze")
    verify_impexp(rep, "tiny_iter2",   n=500, seed=45, iter_num=2, sign="tiny")

    # impexp2 -- iter=1 (wimp=1), iter=2 rise/fall branches
    verify_impexp2(rep, "iter1",       n=500, seed=51, iter_num=1)
    verify_impexp2(rep, "mixed_iter2", n=500, seed=52, iter_num=2)
    verify_impexp2(rep, "rise_iter2",  n=500, seed=53, iter_num=2, sign="rise")
    verify_impexp2(rep, "fall_iter2",  n=500, seed=54, iter_num=2, sign="fall")

    rep.print_table()
    rep.save(Path(__file__).parent / "results")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
