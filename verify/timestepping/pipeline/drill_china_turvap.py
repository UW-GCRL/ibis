"""Focused drill: turvap iter-1 on L+25p0_+100p0 (China temperate cell).

After findings #17 + #18, this single cell still shows ΔT ~4 mK at step 0.
turcof and stomata are bit-exact (verified by drill_canopy_subroutines).
So the divergence is inside turvap. This script:

  1. Runs F77 lsxmain (with iter-1 turvap state prints from canopy.f).
  2. Replicates JAX iter-1 turvap standalone.
  3. Compares post-solve state (tu, ts, tl, t12, t34, q12, q34, tg, ti)
     and the impexp fractions, chux, chsx, chlx, chgx that turvap uses.
"""
from __future__ import annotations
import sys, io, os, re, tempfile
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
from ibis.physics.canopy_exchange import canini, turcof, turvap, impexp, impexp2
from tests.f2py.ibis_f77_wrapper import NPOI


def parse_cell(label):
    a, b = label.split("_", 1)
    return float(a[1:].replace("p", ".")), float(b.replace("p", "."))


def parse_f77(stdout):
    out = {}
    for line in stdout.splitlines():
        m = re.match(r"\s*F77_IT1_(\w+)\s+(.*)", line)
        if m:
            tag = m.group(1)
            if tag not in out:
                out[tag] = [float(v) for v in m.group(2).split()]
    return out


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "L+25p0_+100p0"
    lat, lon = parse_cell(target)
    print(f"=== China-cell turvap iter-1 drill: {target} ===\n")

    tmpl = soil_template_for_lat(lat)
    cells = [(target, lat, lon, tmpl, tmpl)] * NPOI
    gswp3 = read_gswp3_month(1990, 6, cells)
    lats = np.full(NPOI, lat)
    jday = (date(1990, 6, 1) - date(1990, 1, 1)).days + 1
    atm_d = diurnal_disaggregate({k: gswp3[k][0, :] for k in gswp3}, 0.0, jday, lats)

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

    # F77 pre-canopy
    f77.call_setsoi(); f77.call_fwetcal()
    f77.call_radiation(); f77.call_cascade()
    f77.call_fwetcal(); f77.call_drystress()

    # Capture F77 stdout from call_canopy
    saved = os.dup(1)
    with tempfile.NamedTemporaryFile(mode='r', suffix='.txt', delete=False) as tf:
        fname = tf.name
    fd = os.open(fname, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
    try:
        os.dup2(fd, 1); os.close(fd)
        f77.call_canopy()
    finally:
        os.dup2(saved, 1); os.close(saved)
    f77_out = open(fname).read(); os.unlink(fname)
    f77_iter1 = parse_f77(f77_out)
    print(f"F77 keys captured: {sorted(f77_iter1.keys())}\n")

    # JAX replicate
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

    wliqu = casc_res.wliqu; wliqs = casc_res.wliqs; wliql = casc_res.wliql
    wsnou = casc_res.wsnou; wsnos = casc_res.wsnos; wsnol = casc_res.wsnol
    npoi = sj.tg.shape[0]
    ones = jnp.ones((npoi,)); z = jnp.zeros((npoi,))

    tc = turcof(
        aero, state.atm.ta, state.atm.qa, state.atm.ua,
        cj.t12, cj.t34, cj.q12, cj.q34, snj.fi,
        c.vonk, DTIME, cp_p.cleaf, cp_p.cstem, cp_p.cgrass,
        cp_p.dleaf, cp_p.dstem, 0,
        z, ones, ones, z, ones, ones, c.grav,
    )
    stom = stomata(
        cj.tu, cj.tl, cj.t12, cj.t34, cj.q12, cj.q34,
        tc.su, tc.sl, state.atm.psurf,
        rad_res.topparu, rad_res.topparl,
        rad_res.termu6, rad_res.termu7,
        rad_res.terml6, rad_res.terml7,
        rad_res.scalcoefu, rad_res.scalcoefl,
        cj.ci_upper, cj.gs_upper, cj.cs_upper,
        cj.ci_lower, cj.gs_lower, cj.cs_lower,
        fwet2_res.fwetu, fwet2_res.fwetl,
        vj.lai, vj.sai, vj.exist, stresstu_j, stresstl_j,
        cj.a10scalparamu, cj.a10daylightu,
        cj.a10scalparaml, cj.a10daylightl,
        state.atm.co2conc, DTIME, sp, c.epsilon,
    )

    chux = cp_p.chu + c.ch2o * wliqu + c.cice * wsnou
    chsx = cp_p.chs + c.ch2o * wliqs + c.cice * wsnos
    chlx = cp_p.chl + c.ch2o * wliql + c.cice * wsnol
    wu = impexp(cj.tu, chux, wliqu, wsnou, c.tmelt, c.hfus, c.epsilon, ones, True)
    ws = impexp(cj.ts, chsx, wliqs, wsnos, c.tmelt, c.hfus, c.epsilon, ones, True)
    wl = impexp(cj.tl, chlx, wliql, wsnol, c.tmelt, c.hfus, c.epsilon, ones, True)
    wg = impexp(sj.tg, chgx, wlgx, wigx, c.tmelt, c.hfus, c.epsilon, ones, True)
    wi = impexp2(snj.ti, snj.ti, c.tmelt, c.epsilon, ones, True)

    cr = turvap(
        cj.tu, cj.ts, cj.tl, cj.t12, cj.t34, cj.q12, cj.q34, sj.tg, snj.ti,
        cj.tu, cj.ts, cj.tl, sj.tg, snj.ti,
        tc.cu, tc.cl, tc.su, tc.ss, tc.sl, tc.sg, tc.si,
        aero.cp, aero.rhoa, aero.tfac,
        xu, xs, xl, snj.fi, vj.fu, vj.fl, vj.lai, vj.sai, vj.frac,
        chux, chsx, chlx,
        wliqu, wliqs, wliql, wsnou, wsnos, wsnol,
        fwet2_res.rliqu, fwet2_res.rliqs, fwet2_res.rliql,
        fwet2_res.fwetu, fwet2_res.fwets, fwet2_res.fwetl,
        stom.totcondub, stom.totconduc, stom.totcondls,
        stom.totcondl3, stom.totcondl4,
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

    def g(x): return float(np.asarray(x).flat[0])

    print("=== Iter-1 turvap STATE: F77 vs JAX ===")
    jax_state = [g(cr.tu), g(cr.ts), g(cr.tl), g(cr.t12), g(cr.t34),
                 g(cr.q12), g(cr.q34), g(cr.tg), g(cr.ti)]
    for nm, fv, jv in zip(("tu","ts","tl","t12","t34","q12","q34","tg","ti"),
                          f77_iter1["TURVAP_STATE"], jax_state):
        mark = " 🔥" if abs(fv-jv) > 1e-10 else ""
        print(f"  {nm:>4s}  F77={fv:>23.16e}  JAX={jv:>23.16e}  |Δ|={abs(fv-jv):>10.3e}{mark}")

    print("\n=== Iter-1 turvap OLD (chux/chsx/chlx/chgx and tuold/tsold/tlold/tgold/tiold) ===")
    jax_old = [g(cj.tu), g(cj.ts), g(cj.tl), g(sj.tg), g(snj.ti),
               g(chux), g(chsx), g(chlx), g(chgx)]
    for nm, fv, jv in zip(("tuold","tsold","tlold","tgold","tiold",
                            "chux","chsx","chlx","chgx"),
                          f77_iter1["TURVAP_OLD"], jax_old):
        mark = " 🔥" if abs(fv-jv) > 1e-10 else ""
        print(f"  {nm:>6s}  F77={fv:>23.16e}  JAX={jv:>23.16e}  |Δ|={abs(fv-jv):>10.3e}{mark}")

    print("\n=== Iter-1 turvap WET (wliq, wsno, rliq) ===")
    jax_wet = [g(wliqu), g(wliqs), g(wliql), g(wsnou), g(wsnos), g(wsnol),
               g(fwet2_res.rliqu), g(fwet2_res.rliqs), g(fwet2_res.rliql)]
    for nm, fv, jv in zip(("wliqu","wliqs","wliql","wsnou","wsnos","wsnol",
                            "rliqu","rliqs","rliql"),
                          f77_iter1["TURVAP_WET"], jax_wet):
        mark = " 🔥" if abs(fv-jv) > 1e-12 else ""
        print(f"  {nm:>6s}  F77={fv:>23.16e}  JAX={jv:>23.16e}  |Δ|={abs(fv-jv):>10.3e}{mark}")

    print("\n=== Iter-1 turvap W (impexp fractions, pflux, fwetu) ===")
    jax_w = [g(wu), g(ws), g(wl), g(wg), g(wi),
             g(casc_res.pfluxu), g(casc_res.pfluxs), g(casc_res.pfluxl),
             g(fwet2_res.fwetu)]
    for nm, fv, jv in zip(("wu","ws","wl","wg","wi","pfluxu","pfluxs","pfluxl","fwetu"),
                          f77_iter1["TURVAP_W"], jax_w):
        mark = " 🔥" if abs(fv-jv) > 1e-12 else ""
        print(f"  {nm:>6s}  F77={fv:>23.16e}  JAX={jv:>23.16e}  |Δ|={abs(fv-jv):>10.3e}{mark}")


if __name__ == "__main__":
    main()
