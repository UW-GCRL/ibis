"""Subdivide canopy 3-iter: localize divergence to turcof, stomata, or turvap.

Strategy:
  1. Run F77 call_canopy (3 iters), read accessible iter-3 final values:
     - aerodynamic coefs (cu, cl, su, ss, sl, sg, si) ← turcof outputs
     - stomata (stresstu/l, gs_upper/lower) ← stomata outputs
     - canopy state (tu, tl, t12, t34, q12, q34, tg, ti) ← turvap outputs
  2. Run JAX 3-iter loop standalone, capture iter-3 intermediates:
     turcof outputs, stomata outputs, turvap outputs
  3. Compare aero/stomata/canopy fields between F77 and JAX iter-3

If turcof outputs (aero coefs) match → bug downstream (stomata or turvap)
If stomata outputs match too → bug in turvap (9x9 solve)
If turcof outputs differ → bug in turcof
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


def show_compare(label, results, threshold=1e-12):
    print(f"\n=== {label} ===")
    any_div = False
    for name, fv, jv, d in sorted(results, key=lambda r: -r[3]):
        flag = " 🔥" if d > 1e-6 else (" ⚠" if d > threshold else "")
        if d > threshold:
            any_div = True
        if d > 0:
            print(f"  {name:>20s}  F77={fv:>14.6e}  JAX={jv:>14.6e}  |Δ|={d:>11.3e}{flag}")
    if not any_div:
        print(f"  bit-exact across {len(results)} fields")
    return any_div


def cmp(name, fv, jv, out): out.append((name, fv, jv, abs(fv - jv)))


def main():
    target = "L-35p0_+140p0"
    lat, lon = -35.0, 140.0
    print(f"=== Canopy subroutine drill: {target} ===")
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
    cp_p = params.canopy; sp = params.stomata_params

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

    # Run F77 pre-canopy + call_canopy
    f77.call_setsoi(); f77.call_fwetcal()
    f77.call_radiation(); f77.call_cascade()
    f77.call_fwetcal(); f77.call_drystress()
    f77.call_canopy()
    f_aero = f77.get_aerodynamic()
    f_stom = f77.get_stomata()
    f_veg  = f77.get_veg()
    f_soil = f77.get_soil()
    f_flux = f77.get_fluxes()

    # Replicate pre-canopy stages in JAX
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
        vj.lai, vj.sai, vj.fu, vj.fl, cj.tu, cj.ts, cj.tl, sj.tg, snj.ti,
        snj.fi, snj.tsno[:, 0], sj.wsoi[:, 0], sj.wisoi[:, 0],
        sj.albsav, sj.albsan,
        fwet.fwetu, fwet.fwets, fwet.fwetl,
        fwet.rliqu, fwet.rliqs, fwet.rliql,
        params.veg_rad, c.stef, c.tmelt, c.epsilon,
    )
    casc_res = cascade(
        cj.tu, cj.ts, cj.tl, cj.t12, cj.t34,
        cj.wliqu, cj.wliqs, cj.wliql, cj.wsnou, cj.wsnos, cj.wsnol,
        vj.lai, vj.sai, vj.fu, vj.fl,
        state.atm.ta, state.atm.qa, state.atm.psurf,
        state.atm.raina, state.atm.snowa,
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

    # JAX 3-iter loop, capturing iter-3 turcof + stomata + turvap outputs
    aero = canini(
        state.atm.ta, state.atm.qa, state.atm.psurf, state.atm.ua,
        snj.fi, sj.z0soi, snj.z0sno,
        vj.fl, vj.fu, vj.lai, vj.sai, vj.ztop, vj.zbot,
        cp_p.alaiml, cp_p.alaimu,
        c.grav, c.rair, c.rvap, c.cair, c.cvap, c.cappa,
    )
    xu, xs, xl, chgx, wlgx, wigx, cog, coi, zirg, ziri = _canopy_intermediates(
        vj.lai, vj.sai, vj.fu, vj.fl, snj.fi,
        sets_res.consoi[:, 0], sj.hsoi[0],
        snj.consno, snj.hsno[:, 0], snj.hsnotop,
        sj.tg, snj.ti, c.stef, c.epsilon,
        sj.wpud, sj.wipud, sj.poros[:, 0], sj.wsoi[:, 0], sj.wisoi[:, 0],
        jnp.full_like(sj.tg, params.csoi_mineral_val),
        jnp.full_like(sj.tg, params.rhosoi_val),
        c.ch2o, c.cice, c.rhow,
    )

    # Post-cascade water
    wliqu = casc_res.wliqu; wliqs = casc_res.wliqs; wliql = casc_res.wliql
    wsnou = casc_res.wsnou; wsnos = casc_res.wsnos; wsnol = casc_res.wsnol

    tuold, tsold, tlold = cj.tu, cj.ts, cj.tl
    tgold, tiold = sj.tg, snj.ti
    tu, ts, tl = cj.tu, cj.ts, cj.tl
    t12, t34, q12, q34 = cj.t12, cj.t34, cj.q12, cj.q34
    tg, ti = sj.tg, snj.ti

    npoi = sj.tg.shape[0]
    z = jnp.zeros((npoi,)); ones = jnp.ones((npoi,))
    richl, richu = z, z
    straml, strahl = ones, ones
    stramu, strahu = ones, ones
    wu_prev, ws_prev, wl_prev = ones, ones, ones
    wg_prev, wi_prev = ones, ones
    ci_u, gs_u, cs_u = cj.ci_upper, cj.gs_upper, cj.cs_upper
    ci_l, gs_l, cs_l = cj.ci_lower, cj.gs_lower, cj.cs_lower
    totcondub = z; totconduc = z; totcondls = z; totcondl3 = z; totcondl4 = z
    a10sp_u, a10dl_u = cj.a10scalparamu, cj.a10daylightu
    a10sp_l, a10dl_l = cj.a10scalparaml, cj.a10daylightl
    u2_carry = aero.u2; alogu_carry = aero.alogu; alogl_carry = aero.alogl
    tc = None; stom = None; canopy_result = None

    for it in range(3):
        is_first = (it == 0)
        aero_iter = eqx.tree_at(
            lambda a: (a.u2, a.alogu, a.alogl),
            aero,
            (aero.u2 if is_first else u2_carry,
             aero.alogu if is_first else alogu_carry,
             aero.alogl if is_first else alogl_carry),
        )
        tc = turcof(
            aero_iter, state.atm.ta, state.atm.qa, state.atm.ua,
            t12, t34, q12, q34, snj.fi,
            c.vonk, DTIME, cp_p.cleaf, cp_p.cstem, cp_p.cgrass,
            cp_p.dleaf, cp_p.dstem, it,
            richl, straml, strahl, richu, stramu, strahu, c.grav,
        )
        stom = stomata(
            tu, tl, t12, t34, q12, q34, tc.su, tc.sl, state.atm.psurf,
            rad_res.topparu, rad_res.topparl,
            rad_res.termu6, rad_res.termu7,
            rad_res.terml6, rad_res.terml7,
            rad_res.scalcoefu, rad_res.scalcoefl,
            ci_u, gs_u, cs_u, ci_l, gs_l, cs_l,
            fwet2_res.fwetu, fwet2_res.fwetl,
            vj.lai, vj.sai, vj.exist, stresstu_j, stresstl_j,
            a10sp_u, a10dl_u, a10sp_l, a10dl_l,
            state.atm.co2conc, DTIME, sp, c.epsilon,
        )
        ci_u = stom.ci_upper; gs_u = stom.gs_upper; cs_u = stom.cs_upper
        ci_l = stom.ci_lower; gs_l = stom.gs_lower; cs_l = stom.cs_lower
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
            xu, xs, xl, snj.fi, vj.fu, vj.fl, vj.lai, vj.sai, vj.frac,
            chux, chsx, chlx,
            wliqu, wliqs, wliql, wsnou, wsnos, wsnol,
            fwet2_res.rliqu, fwet2_res.rliqs, fwet2_res.rliql,
            fwet2_res.fwetu, fwet2_res.fwets, fwet2_res.fwetl,
            totcondub, totconduc, totcondls, totcondl3, totcondl4,
            chgx, cog, coi, zirg, ziri,
            sj.wpud, sj.wipud, params.wpudmax,
            sj.wisoi[:, 0], sj.wsoi[:, 0],
            sj.swilt[:, 0], sj.suction[:, 0], sj.bex[:, 0],
            sets_res.hvasug, sets_res.hvasui,
            sj.tsoi[:, 0], snj.tsno[:, 0],
            rad_res.solu, rad_res.sols, rad_res.soll, rad_res.solg, rad_res.soli,
            rad_res.firu, rad_res.firs, rad_res.firl, rad_res.firg, rad_res.firi, rad_res.firb,
            casc_res.pfluxu, casc_res.pfluxs, casc_res.pfluxl,
            state.atm.ta, state.atm.qa, state.atm.psurf,
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

    # ============================================================
    # Compare F77 call_canopy() final vs JAX iter-3 final
    # ============================================================

    # turcof outputs (aerodynamic coefficients)
    r = []
    for k in ("cu","cl","su","ss","sl","sg","si"):
        cmp(f"turcof.{k}", grab(f_aero[k]), grab(getattr(tc, k)), r)
    show_compare("turcof outputs (aero conductances)", r)

    # stomata outputs
    r = []
    cmp("stom.stresstu", grab(f_stom["stresstu"]), grab(stresstu_j), r)
    cmp("stom.stresstl", grab(f_stom["stresstl"]), grab(stresstl_j), r)
    # canopy-scaled conductances if exposed
    for k in ("totcondub","totconduc","totcondls","totcondl3","totcondl4"):
        if k in f_stom:
            cmp(f"stom.{k}", grab(f_stom[k]), grab(getattr(stom, k)), r)
    show_compare("stomata outputs", r)

    # turvap outputs (canopy state + fluxes)
    r = []
    cmp("turvap.tu",  grab(f_veg["tu"]),  grab(canopy_result.tu), r)
    cmp("turvap.ts",  grab(f_veg["ts"]),  grab(canopy_result.ts), r)
    cmp("turvap.tl",  grab(f_veg["tl"]),  grab(canopy_result.tl), r)
    cmp("turvap.t12", grab(f_veg["t12"]), grab(canopy_result.t12), r)
    cmp("turvap.t34", grab(f_veg["t34"]), grab(canopy_result.t34), r)
    cmp("turvap.q12", grab(f_veg["q12"]), grab(canopy_result.q12), r)
    cmp("turvap.q34", grab(f_veg["q34"]), grab(canopy_result.q34), r)
    cmp("turvap.tg",  grab(f_soil["tg"]), grab(canopy_result.tg), r)
    for k in ("fseng","fsenu","fsens","fsenl","fseni","fvapg","fvapi"):
        cmp(f"turvap.{k}", grab(f_flux[k]), grab(getattr(canopy_result, k)), r)
    show_compare("turvap outputs", r)


if __name__ == "__main__":
    main()
