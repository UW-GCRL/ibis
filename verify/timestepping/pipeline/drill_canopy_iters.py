"""Per-iter drill into the canopy 3-iter loop.

Stages 1-6 of lsxmain were proven bit-exact between F77 and JAX in
drill_stage_isolate.py. The bug is in the canopy 3-iter loop (turcof +
stomata + turvap × N=3). This script:

1. Replicates the canopy iter loop from timestepping/loops.py
2. Runs N=1, 2, 3, 4, 5 iterations on JAX
3. Captures intermediate state after each iter
4. Compares each JAX-iter-N result vs F77's call_canopy() (fixed at N=3)
5. Reports the convergence behavior

Hypothesis tests:
  - If JAX-iter-1 already shows |Δ|≈5e-3 K vs F77-iter-3:
    → single-iter physics diverges. Bug in turcof or turvap.
  - If |Δ| grows from N=1 → N=3 → N=5 in JAX:
    → iter scheme is unstable / converges differently than F77.
  - If JAX|Δ| shrinks at N=3 vs N=1 then grows at N=5:
    → both converged differently to same precision floor (FP-precision).
"""
from __future__ import annotations
from datetime import date
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

from verify.timestepping.pipeline.global_gswp3_test import (
    diurnal_disaggregate, read_gswp3_month, make_batch_state,
    soil_template_for_lat,
)
from verify.timestepping.verify_lsxmain import (
    setup_f77_state, setup_jax_state, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, _canopy_intermediates
from ibis.physics.soil_physics import setsoi
from ibis.physics.interception import fwetcal, cascade
from ibis.physics.radiation import compute_radiation
from ibis.physics.stomata import drystress, stomata
from ibis.physics.canopy_exchange import (
    canini, turcof, turvap, impexp, impexp2,
)
from tests.f2py.ibis_f77_wrapper import NPOI


def grab(a, idx=0):
    return float(np.asarray(a).flat[idx])


def run_jax_canopy_n_iters(state, sets_res, fwet2_res, casc_res, rad_res,
                            stresstu_in, stresstl_in,
                            consts, params, n_iters):
    """Replica of timestepping/loops.py canopy 3-iter loop, but n_iters configurable.
    Returns: list of (iter_num, t12, t34, q12, q34, tg, ti, fseng, fsenu, fsens, fsenl, fseni)
    one tuple per iter, plus the final canopy_result.
    """
    c = consts
    cp_p = params.canopy
    sp = params.stomata_params
    soil = state.soil
    snow = state.snow
    can = state.canopy
    atm = state.atm
    veg = state.vegetation
    npoi = soil.tg.shape[0]

    aero = canini(
        atm.ta, atm.qa, atm.psurf, atm.ua,
        snow.fi, soil.z0soi, snow.z0sno,
        veg.fl, veg.fu, veg.lai, veg.sai, veg.ztop, veg.zbot,
        cp_p.alaiml, cp_p.alaimu,
        c.grav, c.rair, c.rvap, c.cair, c.cvap, c.cappa,
    )

    xu, xs, xl, chgx, wlgx, wigx, cog, coi, zirg, ziri = _canopy_intermediates(
        veg.lai, veg.sai, veg.fu, veg.fl, snow.fi,
        sets_res.consoi[:, 0], soil.hsoi[0],
        snow.consno, snow.hsno[:, 0], snow.hsnotop,
        soil.tg, snow.ti,
        c.stef, c.epsilon,
        soil.wpud, soil.wipud, soil.poros[:, 0], soil.wsoi[:, 0], soil.wisoi[:, 0],
        jnp.full_like(soil.tg, params.csoi_mineral_val),
        jnp.full_like(soil.tg, params.rhosoi_val),
        c.ch2o, c.cice, c.rhow,
    )

    wliqu = can.wliqu; wliqs = can.wliqs; wliql = can.wliql
    wsnou = can.wsnou; wsnos = can.wsnos; wsnol = can.wsnol
    # Use post-cascade water (from casc_res) — same as timestep_body does
    wliqu = casc_res.wliqu; wliqs = casc_res.wliqs; wliql = casc_res.wliql
    wsnou = casc_res.wsnou; wsnos = casc_res.wsnos; wsnol = casc_res.wsnol

    tuold, tsold, tlold = can.tu, can.ts, can.tl
    tgold, tiold = soil.tg, snow.ti

    tu, ts, tl = can.tu, can.ts, can.tl
    t12, t34 = can.t12, can.t34
    q12, q34 = can.q12, can.q34
    tg, ti = soil.tg, snow.ti

    z = jnp.zeros((npoi,))
    ones = jnp.ones((npoi,))
    richl, richu = z, z
    straml, strahl = ones, ones
    stramu, strahu = ones, ones
    wu_prev, ws_prev, wl_prev = ones, ones, ones
    wg_prev, wi_prev = ones, ones
    ci_upper, gs_upper, cs_upper = can.ci_upper, can.gs_upper, can.cs_upper
    ci_lower, gs_lower, cs_lower = can.ci_lower, can.gs_lower, can.cs_lower
    totcondub = z; totconduc = z; totcondls = z; totcondl3 = z; totcondl4 = z
    a10sp_u, a10dl_u = can.a10scalparamu, can.a10daylightu
    a10sp_l, a10dl_l = can.a10scalparaml, can.a10daylightl

    iter_logs = []
    u2_carry = aero.u2; alogu_carry = aero.alogu; alogl_carry = aero.alogl
    canopy_result = None

    for it in range(n_iters):
        is_first = (it == 0)
        aero_iter = eqx.tree_at(
            lambda a: (a.u2, a.alogu, a.alogl),
            aero,
            (aero.u2 if is_first else u2_carry,
             aero.alogu if is_first else alogu_carry,
             aero.alogl if is_first else alogl_carry),
        )

        tc = turcof(
            aero_iter, atm.ta, atm.qa, atm.ua,
            t12, t34, q12, q34, snow.fi,
            c.vonk, DTIME,
            cp_p.cleaf, cp_p.cstem, cp_p.cgrass, cp_p.dleaf, cp_p.dstem,
            it, richl, straml, strahl, richu, stramu, strahu, c.grav,
        )

        stom = stomata(
            tu, tl, t12, t34, q12, q34,
            tc.su, tc.sl, atm.psurf,
            rad_res.topparu, rad_res.topparl,
            rad_res.termu6, rad_res.termu7,
            rad_res.terml6, rad_res.terml7,
            rad_res.scalcoefu, rad_res.scalcoefl,
            ci_upper, gs_upper, cs_upper,
            ci_lower, gs_lower, cs_lower,
            fwet2_res.fwetu, fwet2_res.fwetl,
            veg.lai, veg.sai, veg.exist,
            stresstu_in, stresstl_in,
            a10sp_u, a10dl_u, a10sp_l, a10dl_l,
            atm.co2conc, DTIME, sp, c.epsilon,
        )
        ci_upper = stom.ci_upper; gs_upper = stom.gs_upper; cs_upper = stom.cs_upper
        ci_lower = stom.ci_lower; gs_lower = stom.gs_lower; cs_lower = stom.cs_lower
        totcondub = stom.totcondub; totconduc = stom.totconduc
        totcondls = stom.totcondls; totcondl3 = stom.totcondl3; totcondl4 = stom.totcondl4
        a10sp_u = stom.a10scalparamu; a10dl_u = stom.a10daylightu
        a10sp_l = stom.a10scalparaml; a10dl_l = stom.a10daylightl

        chux = cp_p.chu + c.ch2o * wliqu + c.cice * wsnou
        chsx = cp_p.chs + c.ch2o * wliqs + c.cice * wsnos
        chlx = cp_p.chl + c.ch2o * wliql + c.cice * wsnol

        wu = impexp(tu, chux, wliqu, wsnou, c.tmelt, c.hfus, c.epsilon, wu_prev, is_first)
        ws = impexp(ts, chsx, wliqs, wsnos, c.tmelt, c.hfus, c.epsilon, ws_prev, is_first)
        wl = impexp(tl, chlx, wliql, wsnol, c.tmelt, c.hfus, c.epsilon, wl_prev, is_first)
        wg = impexp(tg, chgx, wlgx, wigx, c.tmelt, c.hfus, c.epsilon, wg_prev, is_first)
        wi = impexp2(ti, tiold, c.tmelt, c.epsilon, wi_prev, is_first)

        canopy_result = turvap(
            tu, ts, tl, t12, t34, q12, q34, tg, ti,
            tuold, tsold, tlold, tgold, tiold,
            tc.cu, tc.cl, tc.su, tc.ss, tc.sl, tc.sg, tc.si,
            aero.cp, aero.rhoa, aero.tfac,
            xu, xs, xl, snow.fi, veg.fu, veg.fl,
            veg.lai, veg.sai, veg.frac,
            chux, chsx, chlx,
            wliqu, wliqs, wliql, wsnou, wsnos, wsnol,
            fwet2_res.rliqu, fwet2_res.rliqs, fwet2_res.rliql,
            fwet2_res.fwetu, fwet2_res.fwets, fwet2_res.fwetl,
            totcondub, totconduc, totcondls, totcondl3, totcondl4,
            chgx, cog, coi, zirg, ziri,
            soil.wpud, soil.wipud, params.wpudmax,
            soil.wisoi[:, 0], soil.wsoi[:, 0],
            soil.swilt[:, 0], soil.suction[:, 0], soil.bex[:, 0],
            sets_res.hvasug, sets_res.hvasui,
            soil.tsoi[:, 0], snow.tsno[:, 0],
            rad_res.solu, rad_res.sols, rad_res.soll, rad_res.solg, rad_res.soli,
            rad_res.firu, rad_res.firs, rad_res.firl, rad_res.firg, rad_res.firi, rad_res.firb,
            casc_res.pfluxu, casc_res.pfluxs, casc_res.pfluxl,
            atm.ta, atm.qa, atm.psurf,
            wu, ws, wl, wg, wi,
            DTIME, c.tmelt, c.hvap, c.hsub, c.ch2o, c.cice,
            c.cvap, c.stef, c.rvap, c.grav, c.epsilon,
        )
        tu = canopy_result.tu; ts = canopy_result.ts; tl = canopy_result.tl
        t12 = canopy_result.t12; t34 = canopy_result.t34
        q12 = canopy_result.q12; q34 = canopy_result.q34
        tg = canopy_result.tg; ti = canopy_result.ti

        straml, strahl = tc.straml, tc.strahl
        stramu, strahu = tc.stramu, tc.strahu
        u2_carry = tc.u2; alogu_carry = tc.alogu; alogl_carry = tc.alogl
        wu_prev, ws_prev, wl_prev = wu, ws, wl
        wg_prev, wi_prev = wg, wi

        iter_logs.append({
            "iter": it + 1,
            "t12": grab(t12), "t34": grab(t34),
            "q12": grab(q12), "q34": grab(q34),
            "tg": grab(tg),   "ti": grab(ti),
            "tu": grab(tu),   "tl": grab(tl),
            "fseng": grab(canopy_result.fseng),
            "fsenu": grab(canopy_result.fsenu),
            "fsens": grab(canopy_result.fsens),
            "fsenl": grab(canopy_result.fsenl),
            "fseni": grab(canopy_result.fseni),
        })
    return iter_logs, canopy_result


def main():
    target = "L-35p0_+140p0"
    lat, lon = -35.0, 140.0
    print(f"=== Canopy per-iter drill: {target} ===")
    tmpl = soil_template_for_lat(lat)
    cells = [(target, lat, lon, tmpl, tmpl)] * NPOI

    YEAR, MONTH = 1990, 6
    gswp3 = read_gswp3_month(YEAR, MONTH, cells)
    lats = np.full(NPOI, lat)
    jday = (date(YEAR, MONTH, 1) - date(YEAR, 1, 1)).days + 1
    day_data = {k: gswp3[k][0, :] for k in gswp3}
    atm_d = diurnal_disaggregate(day_data, 0.0, jday, lats)

    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    c = consts; ip = params.interception

    f77.set_atm(**atm_d)
    state = eqx.tree_at(
        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                   s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                   s.atm.raina, s.atm.snowa),
        state,
        tuple(jnp.asarray(atm_d[k]) for k in
              ("ta","qa","ua","psurf","coszen","solad","solai","fira","raina","snowa")),
    )

    sj, snj, cj, vj = state.soil, state.snow, state.canopy, state.vegetation

    # Run F77 pre-canopy stages and call_canopy
    f77.call_setsoi()
    f77.call_fwetcal()
    f77.call_radiation()
    f77.call_cascade()
    f77.call_fwetcal()
    f77.call_drystress()
    f77.call_canopy()

    fv = f77.get_veg()
    fs = f77.get_soil()
    ffx = f77.get_fluxes()
    f77_final = {
        "t12": grab(fv["t12"]), "t34": grab(fv["t34"]),
        "q12": grab(fv["q12"]), "q34": grab(fv["q34"]),
        "tg":  grab(fs["tg"]),  "ti":  grab(fv["tlsub"]),  # tlsub = snow surface
        "tu":  grab(fv["tu"]),  "tl":  grab(fv["tl"]),
        "fseng": grab(ffx["fseng"]),
        "fsenu": grab(ffx["fsenu"]),
        "fsens": grab(ffx["fsens"]),
        "fsenl": grab(ffx["fsenl"]),
        "fseni": grab(ffx["fseni"]),
    }
    print(f"\nF77 call_canopy (3-iter) final:")
    for k, v in f77_final.items():
        print(f"  {k:>8s}: {v:>14.6e}")

    # Now replicate pre-canopy in JAX
    sets_res = setsoi(
        sj.sand, sj.clay, sj.poros, sj.wsoi, sj.wisoi, sj.tsoi,
        sj.wpud, sj.wipud, params.wpudmax, params.zwpmax,
        snj.tsno[:, 0], state.atm.ta,
        c.tmelt, c.hvap, c.hsub, c.ch2o, c.cvap, c.cice, c.epsilon,
    )
    fwet = fwetcal(
        cj.tu, cj.ts, cj.tl, cj.wliqu, cj.wliqs, cj.wliql,
        cj.wsnou, cj.wsnos, cj.wsnol,
        ip.wliqumax, ip.wliqsmax, ip.wliqlmax,
        ip.wsnoumax, ip.wsnosmax, ip.wsnolmax,
        ip.fmax, c.tmelt, c.epsilon,
    )
    rad_res = compute_radiation(
        state.atm.coszen, state.atm.solad, state.atm.solai, state.atm.fira,
        vj.lai, vj.sai, vj.fu, vj.fl,
        cj.tu, cj.ts, cj.tl, sj.tg, snj.ti,
        snj.fi, snj.tsno[:, 0],
        sj.wsoi[:, 0], sj.wisoi[:, 0],
        sj.albsav, sj.albsan,
        fwet.fwetu, fwet.fwets, fwet.fwetl,
        fwet.rliqu, fwet.rliqs, fwet.rliql,
        params.veg_rad,
        c.stef, c.tmelt, c.epsilon,
    )
    casc_res = cascade(
        cj.tu, cj.ts, cj.tl, cj.t12, cj.t34,
        cj.wliqu, cj.wliqs, cj.wliql,
        cj.wsnou, cj.wsnos, cj.wsnol,
        vj.lai, vj.sai, vj.fu, vj.fl,
        state.atm.ta, state.atm.qa, state.atm.psurf,
        state.atm.raina, state.atm.snowa,
        ip, DTIME, c.tmelt, c.ch2o, c.cice, c.hfus, c.epsilon,
        c.cair, c.hvap, c.cvap,
    )
    # fwet2 uses post-cascade water
    fwet2_res = fwetcal(
        cj.tu, cj.ts, cj.tl,
        casc_res.wliqu, casc_res.wliqs, casc_res.wliql,
        casc_res.wsnou, casc_res.wsnos, casc_res.wsnol,
        ip.wliqumax, ip.wliqsmax, ip.wliqlmax,
        ip.wsnoumax, ip.wsnosmax, ip.wsnolmax,
        ip.fmax, c.tmelt, c.epsilon,
    )
    stresstl_j, stresstu_j, _, _ = drystress(
        sj.wsoi, sj.wisoi, sj.swilt, sj.sfield, vj.froot,
    )

    # Run JAX canopy with N=1,2,3,4,5 iters; compare each to F77 3-iter
    print(f"\n{'niter':>5s} | {'t12':>11s} {'t34':>11s} {'q12':>11s} {'tg':>11s}  "
          f"{'fseng':>11s}  | "
          f"{'Δt12':>10s} {'Δt34':>10s} {'Δfseng':>10s} {'Δfseni':>10s}")
    print("-" * 140)
    for n in (1, 2, 3, 4, 5):
        logs, cres = run_jax_canopy_n_iters(
            state, sets_res, fwet2_res, casc_res, rad_res,
            stresstu_j, stresstl_j, consts, params, n,
        )
        last = logs[-1]
        dt12 = abs(last["t12"] - f77_final["t12"])
        dt34 = abs(last["t34"] - f77_final["t34"])
        dfg  = abs(last["fseng"] - f77_final["fseng"])
        dfi  = abs(last["fseni"] - f77_final["fseni"])
        print(f"  N={n} | {last['t12']:>11.6f} {last['t34']:>11.6f} "
              f"{last['q12']:>11.4e} {last['tg']:>11.6f}  "
              f"{last['fseng']:>11.4e}  | "
              f"{dt12:>10.3e} {dt34:>10.3e} {dfg:>10.3e} {dfi:>10.3e}")

    # Per-iter trajectory at N=3 (the canonical case)
    print(f"\n=== Per-iter trajectory inside JAX N=3 (showing convergence) ===")
    logs, _ = run_jax_canopy_n_iters(
        state, sets_res, fwet2_res, casc_res, rad_res,
        stresstu_j, stresstl_j, consts, params, 3,
    )
    print(f"{'iter':>4s} | {'t12':>11s} {'t34':>11s} {'q12':>11s} {'tg':>11s}  "
          f"{'fseng':>11s} {'fseni':>11s}")
    print("-" * 100)
    for L in logs:
        print(f"  {L['iter']:>2d}  | {L['t12']:>11.6f} {L['t34']:>11.6f} "
              f"{L['q12']:>11.4e} {L['tg']:>11.6f}  "
              f"{L['fseng']:>11.4e} {L['fseni']:>11.4e}")
    print(f"  F77 | {f77_final['t12']:>11.6f} {f77_final['t34']:>11.6f} "
          f"{f77_final['q12']:>11.4e} {f77_final['tg']:>11.6f}  "
          f"{f77_final['fseng']:>11.4e} {f77_final['fseni']:>11.4e}")


if __name__ == "__main__":
    main()
