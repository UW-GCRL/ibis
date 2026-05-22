"""Per-iter drill of canopy 3-iter loop at d115 h07 for L+55°N +100°E.

After seeding state through d115 h06 in lockstep, at h07 we:
  - Run F77 lsxmain once (which internally runs canopy 3-iter) and snapshot final.
  - In JAX, replicate the per-iter loop with n=1,2,3,4,5 iters and snapshot
    each iter's (t12, t34, q12, q34, tu, tl, tg, fseng, fseni, fsenl).
  - Compare JAX iter-N vs F77 3-iter final.

If JAX-iter-1 already matches F77 to ~µK but JAX-iter-3 drifts to 0.23 K,
the iter scheme is unstable. If JAX-iter-1 already shows the 0.23 K diff,
single-iter physics is off (a turcof/turvap structural issue).
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
from ibis.timestepping.loops import default_sim_params, timestep_body, _canopy_intermediates
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
    """Replica of timestepping/loops.py canopy 3-iter loop with N configurable."""
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

        cres = turvap(
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
        tu = cres.tu; ts = cres.ts; tl = cres.tl
        t12 = cres.t12; t34 = cres.t34
        q12 = cres.q12; q34 = cres.q34
        tg = cres.tg; ti = cres.ti

        straml, strahl = tc.straml, tc.strahl
        stramu, strahu = tc.stramu, tc.strahu
        u2_carry = tc.u2; alogu_carry = tc.alogu; alogl_carry = tc.alogl
        wu_prev, ws_prev, wl_prev = wu, ws, wl
        wg_prev, wi_prev = wg, wi

        iter_logs.append({
            "iter": it + 1,
            "tu": grab(tu), "tl": grab(tl), "ts": grab(ts),
            "t12": grab(t12), "t34": grab(t34),
            "q12": grab(q12), "q34": grab(q34),
            "tg": grab(tg),
            "fseng": grab(cres.fseng), "fsenu": grab(cres.fsenu),
            "fsens": grab(cres.fsens), "fsenl": grab(cres.fsenl),
            "fseni": grab(cres.fseni),
            "fvapuw": grab(cres.fvapuw), "fvaps": grab(cres.fvaps),
            "fvaplw": grab(cres.fvaplw), "fvapg": grab(cres.fvapg),
        })
    return iter_logs


def main():
    target = "L+55p0_+100p0"
    lat, lon = 55.0, 100.0
    YEAR = 1990
    DRILL_DAY = 115
    DRILL_HOUR = 7
    print(f"=== Iter drill at {target} d{DRILL_DAY} h{DRILL_HOUR} ===")

    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    c = consts; ip = params.interception
    lats = np.full(NPOI, lat)

    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    print(f"  init complete, NPOI={NPOI}", flush=True)

    step = 0
    # SEED through d115 h06
    for month in [1, 2, 3, 4, 5]:
        gswp3 = read_gswp3_month(YEAR, month, cells)
        if month in (1, 3, 5):
            ndays = 31
        elif month == 2:
            ndays = 28
        else:
            ndays = 30
        for d_in_m in range(ndays):
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + d_in_m + 1
            day_data = {k: gswp3[k][d_in_m, :] for k in gswp3}
            day_global = step // 24 + 1
            for h in range(24):
                if h == 0 and day_global % 30 == 0 and day_global < DRILL_DAY:
                    print(f"  seeding: day {day_global}", flush=True)

                # Stop right BEFORE drill timestep, then do the iter drill
                if day_global == DRILL_DAY and h == DRILL_HOUR:
                    atm = diurnal_disaggregate(day_data, float(h), jday, lats)
                    state_atm = eqx.tree_at(
                        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                                   s.atm.coszen, s.atm.solad, s.atm.solai,
                                   s.atm.fira, s.atm.raina, s.atm.snowa),
                        state,
                        tuple(jnp.asarray(atm[k]) for k in
                              ("ta","qa","ua","psurf","coszen","solad","solai",
                               "fira","raina","snowa")),
                    )
                    f77.set_atm(**atm)

                    # === F77: run full lsxmain step, capture canopy outputs ===
                    f77.call_lsxmain()
                    fv = f77.get_veg(); fs = f77.get_soil(); ff = f77.get_fluxes()
                    f77_final = {
                        "tu": grab(fv["tu"]), "tl": grab(fv["tl"]), "ts": grab(fv["ts"]),
                        "t12": grab(fv["t12"]), "t34": grab(fv["t34"]),
                        "q12": grab(fv["q12"]), "q34": grab(fv["q34"]),
                        "tg":  grab(fs["tg"]),
                        "fseng": grab(ff["fseng"]), "fsenu": grab(ff["fsenu"]),
                        "fsens": grab(ff["fsens"]), "fsenl": grab(ff["fsenl"]),
                        "fseni": grab(ff["fseni"]),
                    }
                    print(f"\nF77 lsxmain (3-iter canopy) final at d{DRILL_DAY} h{DRILL_HOUR}:")
                    for k, v in f77_final.items():
                        print(f"  {k:>8s}: {v:>14.6e}")

                    # === JAX: replicate pre-canopy stages, then iter sweep ===
                    sj, snj, cj, vj = state_atm.soil, state_atm.snow, state_atm.canopy, state_atm.vegetation

                    sets_res = setsoi(
                        sj.sand, sj.clay, sj.poros, sj.wsoi, sj.wisoi, sj.tsoi,
                        sj.wpud, sj.wipud, params.wpudmax, params.zwpmax,
                        snj.tsno[:, 0], state_atm.atm.ta,
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
                        state_atm.atm.coszen, state_atm.atm.solad, state_atm.atm.solai,
                        state_atm.atm.fira,
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
                        state_atm.atm.ta, state_atm.atm.qa, state_atm.atm.psurf,
                        state_atm.atm.raina, state_atm.atm.snowa,
                        ip, DTIME, c.tmelt, c.ch2o, c.cice, c.hfus, c.epsilon,
                        c.cair, c.hvap, c.cvap,
                    )
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

                    print(f"\n{'niter':>5s} | {'tu':>11s} {'tl':>11s} {'t12':>11s} "
                          f"{'t34':>11s} {'tg':>11s} | "
                          f"{'Δtu':>10s} {'Δtl':>10s} {'Δt12':>10s} {'Δt34':>10s} {'Δfseng':>10s} {'Δfsenl':>10s}")
                    print("-" * 150)
                    for n in (1, 2, 3, 4, 5):
                        logs = run_jax_canopy_n_iters(
                            state_atm, sets_res, fwet2_res, casc_res, rad_res,
                            stresstu_j, stresstl_j,
                            consts, params, n,
                        )
                        last = logs[-1]
                        print(f"{n:>5d} | {last['tu']:>11.6e} {last['tl']:>11.6e} "
                              f"{last['t12']:>11.6e} {last['t34']:>11.6e} {last['tg']:>11.6e} | "
                              f"{last['tu']-f77_final['tu']:>+10.3e} "
                              f"{last['tl']-f77_final['tl']:>+10.3e} "
                              f"{last['t12']-f77_final['t12']:>+10.3e} "
                              f"{last['t34']-f77_final['t34']:>+10.3e} "
                              f"{last['fseng']-f77_final['fseng']:>+10.3e} "
                              f"{last['fsenl']-f77_final['fsenl']:>+10.3e}")

                    print(f"\nPer-iter trajectory for n=3 (matching production):")
                    logs = run_jax_canopy_n_iters(
                        state_atm, sets_res, fwet2_res, casc_res, rad_res,
                        stresstu_j, stresstl_j,
                        consts, params, 3,
                    )
                    print(f"{'iter':>4s}  {'tu':>11s} {'tl':>11s} {'t12':>11s} {'t34':>11s} {'fseng':>11s} {'fsenl':>11s}")
                    for log in logs:
                        print(f"{log['iter']:>4d}  {log['tu']:>11.6e} {log['tl']:>11.6e} "
                              f"{log['t12']:>11.6e} {log['t34']:>11.6e} "
                              f"{log['fseng']:>11.6e} {log['fsenl']:>11.6e}")
                    return

                # Otherwise, normal lockstep advance
                atm = diurnal_disaggregate(day_data, float(h), jday, lats)
                f77.set_atm(**atm)
                state_atm = eqx.tree_at(
                    lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                               s.atm.coszen, s.atm.solad, s.atm.solai,
                               s.atm.fira, s.atm.raina, s.atm.snowa),
                    state,
                    tuple(jnp.asarray(atm[k]) for k in
                          ("ta","qa","ua","psurf","coszen","solad","solai",
                           "fira","raina","snowa")),
                )
                f77.call_lsxmain()
                state = timestep_body(state_atm, consts, params, DTIME)
                step += 1


if __name__ == "__main__":
    main()
