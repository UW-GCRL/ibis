"""Compare F77 iter-1 turcof + turvap outputs to JAX iter-1.

If iter-1 outputs are bit-exact:
  → divergence starts at iter-2+ (iter-weighting / state-feedback)
If iter-1 already differs:
  → bug is in turcof or turvap's plain FP arithmetic at iter-1

F77 is instrumented via canopy.f prints (lines 'F77_IT1_TURCOF', 'F77_IT1_STRAT',
'F77_IT1_TURVAP'). JAX replicates iter-1 standalone here.
"""
from __future__ import annotations
import sys
import io
import re
import contextlib
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
    lat_s, lon_s = label.split("_", 1)
    return float(lat_s[1:].replace("p", ".")), float(lon_s.replace("p", "."))


def parse_f77_lines(stdout):
    """Extract F77_IT1_* and F77_CANINI values from captured stdout."""
    out = {}
    for line in stdout.splitlines():
        for prefix in ("F77_IT1_", "F77_"):
            m = re.match(r"\s*" + prefix + r"(\w+)\s+(.*)", line)
            if m:
                tag = m.group(1)
                if tag not in out:  # first hit wins (canini once)
                    vals = [float(v) for v in m.group(2).split()]
                    out[tag] = vals
                break
    return out


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "L-05p0_-080p0"
    lat, lon = parse_cell(target)
    print(f"=== Iter-1 drill on {target} (lat={lat}, lon={lon}) ===")

    tmpl = soil_template_for_lat(lat)
    cells = [(target, lat, lon, tmpl, tmpl)] * NPOI

    gswp3 = read_gswp3_month(1990, 6, cells)
    lats = np.full(NPOI, lat)
    jday = (date(1990, 6, 1) - date(1990, 1, 1)).days + 1
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

    # Run F77 pre-canopy stages and canopy (with print instrumentation)
    f77.call_setsoi(); f77.call_fwetcal()
    f77.call_radiation(); f77.call_cascade()
    f77.call_fwetcal(); f77.call_drystress()

    # Capture F77 stdout
    buf = io.StringIO()
    # F77 prints go via the underlying ctypes/runtime — Python's
    # contextlib.redirect_stdout doesn't capture those. Use os-level
    # FD redirection.
    import os, tempfile
    saved = os.dup(1)
    with tempfile.NamedTemporaryFile(mode='r', suffix='.txt', delete=False) as tf:
        fname = tf.name
    fd = os.open(fname, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
    try:
        os.dup2(fd, 1)
        os.close(fd)
        f77.call_canopy()
    finally:
        os.dup2(saved, 1)
        os.close(saved)
    with open(fname) as fh:
        f77_out = fh.read()
    os.unlink(fname)
    f77_iter1 = parse_f77_lines(f77_out)
    if not f77_iter1:
        print("ERROR: no F77_IT1_* lines captured. F77 stdout:")
        print(f77_out[:500])
        return 1
    print(f"[debug] F77 keys captured: {sorted(f77_iter1.keys())}")

    # JAX: pre-canopy stages
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

    tuold, tsold, tlold = cj.tu, cj.ts, cj.tl
    tgold, tiold = sj.tg, snj.ti

    npoi = sj.tg.shape[0]
    z = jnp.zeros((npoi,)); ones = jnp.ones((npoi,))

    # === JAX iter 1 ===
    # Inline turcof to get alogl, alogu, yu, yl, u1..u4 as intermediates
    vonk = c.vonk
    # Lower-story
    xl_v = ((aero.alog4 - aero.alogav) / vonk) ** 2 * aero.bdl
    rwork = 1.0 / aero.expl
    yl_v = ((xl_v + 1) * aero.expl + (xl_v - 1) * rwork) / (
           (xl_v + 1) * aero.expl - (xl_v - 1) * rwork)
    alogl_v = aero.alog3 - vonk * jnp.sqrt(yl_v / aero.bdl)
    # Upper-story (straml=1 at iter 1)
    xu_v = ((aero.alog2 - alogl_v) / vonk) ** 2 * aero.bdu / ones
    rwork = 1.0 / aero.expu
    yu_v = ((xu_v + 1) * aero.expu + (xu_v - 1) * rwork) / (
           (xu_v + 1) * aero.expu - (xu_v - 1) * rwork)
    alogu_v = aero.alog1 - vonk * jnp.sqrt(yu_v / aero.bdu)
    # Wind speeds
    tauu_v = (state.atm.ua * vonk / (aero.aloga - alogu_v)) ** 2 * ones
    a_v = 0.5 * tauu_v * (yu_v + 1) / aero.bdu
    b_v = 0.5 * tauu_v * (yu_v - 1) / aero.bdu
    taul_v = aero.bdu * (a_v / aero.expu - b_v * aero.expu)
    cc_v = 0.5 * taul_v * (yl_v + 1) / aero.bdl
    dd_v = 0.5 * taul_v * (yl_v - 1) / aero.bdl
    _usafe = lambda v: jnp.maximum(0.01, jnp.sqrt(jnp.maximum(0.0, v)))
    u1_v = _usafe(a_v + b_v)
    u12_v = _usafe(a_v / aero.exphu + b_v * aero.exphu)
    u2_v = _usafe(a_v / aero.expu + b_v * aero.expu)
    u3_v = _usafe(cc_v + dd_v)
    u34_v = _usafe(cc_v / aero.exphl + dd_v * aero.exphl)
    u4_v = _usafe(cc_v / aero.expl + dd_v * aero.expl)

    tc = turcof(
        aero, state.atm.ta, state.atm.qa, state.atm.ua,
        cj.t12, cj.t34, cj.q12, cj.q34, snj.fi,
        c.vonk, DTIME, cp_p.cleaf, cp_p.cstem, cp_p.cgrass,
        cp_p.dleaf, cp_p.dstem, 0,    # iter_num=0 (first iter)
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
    wi = impexp2(snj.ti, tiold, c.tmelt, c.epsilon, ones, True)

    cr = turvap(
        cj.tu, cj.ts, cj.tl, cj.t12, cj.t34, cj.q12, cj.q34, sj.tg, snj.ti,
        tuold, tsold, tlold, tgold, tiold,
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

    print("\n=== Iter-1 turcof: F77 vs JAX ===")
    print(f"  {'field':>10s}  {'F77':>23s}  {'JAX':>23s}  {'|Δ|':>10s}")
    jax_turcof = [g(tc.cu), g(tc.cl), g(tc.su), g(tc.ss), g(tc.sl), g(tc.sg), g(tc.si)]
    f77_turcof = f77_iter1["TURCOF"]
    for nm, fv, jv in zip(("cu","cl","su","ss","sl","sg","si"), f77_turcof, jax_turcof):
        marker = " 🔥" if abs(fv-jv) > 1e-8 else ""
        print(f"  {nm:>10s}  {fv:>23.16e}  {jv:>23.16e}  {abs(fv-jv):>10.2e}{marker}")

    if "LOG" in f77_iter1:
        print("\n=== Iter-1 log/yu/yl: F77 vs JAX ===")
        jax_log = [g(alogu_v), g(alogl_v), g(yu_v), g(yl_v)]
        for nm, fv, jv in zip(("alogu","alogl","yu","yl"), f77_iter1["LOG"], jax_log):
            marker = " 🔥" if abs(fv-jv) > 1e-14 else ""
            print(f"  {nm:>10s}  {fv:>23.16e}  {jv:>23.16e}  {abs(fv-jv):>10.2e}{marker}")
    if "ZA_RAW" in f77_iter1:
        print("\n=== za computation: F77 vs JAX ===")
        # JAX inputs
        siga = 0.999
        pa_j = float(g(state.atm.psurf)) * siga
        rhoa_j_input = (float(c.rair) * float(g(state.atm.ta)) *
                        (1.0 + (float(c.rvap)/float(c.rair) - 1.0) * float(g(state.atm.qa))))
        rhoa_j = pa_j / rhoa_j_input
        za_raw_j = (float(g(state.atm.psurf)) - pa_j) / (rhoa_j * float(c.grav))
        jax_za = [za_raw_j, pa_j, rhoa_j, g(state.atm.ta), g(state.atm.qa), g(state.atm.psurf)]
        for nm, fv, jv in zip(("za_raw","pa","rhoa","ta","qa","psurf"), f77_iter1["ZA_RAW"], jax_za):
            marker = " 🔥" if abs(fv-jv) > 1e-13 else ""
            print(f"  {nm:>10s}  {fv:>23.16e}  {jv:>23.16e}  {abs(fv-jv):>10.2e}{marker}")
    if "ZTOP" in f77_iter1:
        ztop_j = g(state.vegetation.ztop[:, 1])
        print(f"\n=== ztop[upper]: F77={f77_iter1['ZTOP'][0]:.16e}  JAX={ztop_j:.16e}  Δ={abs(f77_iter1['ZTOP'][0]-ztop_j):.2e}")
    if "CANINI" in f77_iter1:
        print("\n=== canini: F77 vs JAX ===")
        jax_canini = [g(aero.za), g(aero.dispu), g(aero.displ), g(aero.z1), g(aero.aloga)]
        for nm, fv, jv in zip(("za","dispu","displ","z1","aloga"), f77_iter1["CANINI"], jax_canini):
            marker = " 🔥" if abs(fv-jv) > 1e-14 else ""
            print(f"  {nm:>10s}  {fv:>23.16e}  {jv:>23.16e}  {abs(fv-jv):>10.2e}{marker}")
    if "UAINP" in f77_iter1:
        print("\n=== Iter-1 ua/vonk/aloga/alogu/bdu: F77 vs JAX ===")
        jax_uainp = [g(state.atm.ua), float(c.vonk), g(aero.aloga), g(alogu_v), g(aero.bdu)]
        for nm, fv, jv in zip(("ua","vonk","aloga","alogu","bdu"), f77_iter1["UAINP"], jax_uainp):
            marker = " 🔥" if abs(fv-jv) > 1e-14 else ""
            print(f"  {nm:>10s}  {fv:>23.16e}  {jv:>23.16e}  {abs(fv-jv):>10.2e}{marker}")
    if "TAUSC" in f77_iter1:
        print("\n=== Iter-1 tauu/taul/a/b/c/d: F77 vs JAX ===")
        jax_tausc = [g(tauu_v), g(taul_v), g(a_v), g(b_v), g(cc_v), g(dd_v)]
        for nm, fv, jv in zip(("tauu","taul","a","b","c","d"), f77_iter1["TAUSC"], jax_tausc):
            marker = " 🔥" if abs(fv-jv) > 1e-14 else ""
            print(f"  {nm:>10s}  {fv:>23.16e}  {jv:>23.16e}  {abs(fv-jv):>10.2e}{marker}")
    if "U" in f77_iter1:
        print("\n=== Iter-1 u_levels: F77 vs JAX ===")
        jax_u = [g(u1_v), g(u12_v), g(u2_v), g(u3_v), g(u34_v), g(u4_v)]
        for nm, fv, jv in zip(("u1","u12","u2","u3","u34","u4"), f77_iter1["U"], jax_u):
            marker = " 🔥" if abs(fv-jv) > 1e-14 else ""
            print(f"  {nm:>10s}  {fv:>23.16e}  {jv:>23.16e}  {abs(fv-jv):>10.2e}{marker}")

    print("\n=== Iter-1 strat: F77 vs JAX ===")
    jax_strat = [g(tc.stramu), g(tc.strahu), g(tc.straml), g(tc.strahl)]
    f77_strat = f77_iter1["STRAT"]
    for nm, fv, jv in zip(("stramu","strahu","straml","strahl"), f77_strat, jax_strat):
        marker = " 🔥" if abs(fv-jv) > 1e-12 else ""
        print(f"  {nm:>10s}  {fv:>23.16e}  {jv:>23.16e}  {abs(fv-jv):>10.2e}{marker}")

    print("\n=== Iter-1 turvap: F77 vs JAX ===")
    jax_turvap = [g(cr.tu), g(cr.ts), g(cr.tl), g(cr.t12), g(cr.t34),
                  g(cr.q12), g(cr.q34), g(cr.tg), g(cr.ti)]
    f77_turvap = f77_iter1["TURVAP"]
    for nm, fv, jv in zip(("tu","ts","tl","t12","t34","q12","q34","tg","ti"),
                          f77_turvap, jax_turvap):
        marker = " 🔥" if abs(fv-jv) > 1e-10 else ""
        print(f"  {nm:>10s}  {fv:>23.16e}  {jv:>23.16e}  {abs(fv-jv):>10.2e}{marker}")


if __name__ == "__main__":
    main()
