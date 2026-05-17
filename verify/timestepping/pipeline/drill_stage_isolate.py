"""Stage-by-stage F77↔JAX divergence localization for L-35p0_+140p0.

Both codes run lsxmain in this order:
  setsoi → fwetcal → radiation → cascade → fwetcal → drystress → canopy(3-iter)

Run each F77 subroutine individually (via the wrapper) and the matching
JAX function, comparing every accessible field after each stage. The
first stage with non-zero |Δ| is the source of divergence.
"""
from __future__ import annotations
import sys
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
from ibis.timestepping.loops import default_sim_params, timestep_body
from ibis.physics.soil_physics import setsoi
from ibis.physics.interception import fwetcal, cascade
from ibis.physics.radiation import compute_radiation
from ibis.physics.stomata import drystress
from tests.f2py.ibis_f77_wrapper import NPOI


def grab(a, idx=0):
    return float(np.asarray(a).flat[idx])


def cmp(name, fv, jv, out):
    out.append((name, fv, jv, abs(fv - jv)))


def show(stage, results, threshold=1e-12):
    print(f"\n=== Stage: {stage} ===")
    any_div = False
    for name, fv, jv, d in sorted(results, key=lambda r: -r[3]):
        flag = " 🔥" if d > 1e-6 else (" ⚠" if d > threshold else "")
        if d > threshold:
            any_div = True
        if d > 0:
            print(f"  {name:>28s}  F77={fv:>14.6e}  JAX={jv:>14.6e}  |Δ|={d:>11.3e}{flag}")
    if not any_div:
        print(f"  bit-exact across {len(results)} fields (all |Δ| < {threshold:.0e})")


def main():
    target = "L-35p0_+140p0"
    lat, lon = -35.0, 140.0
    print(f"=== Stage drill: {target} (lat={lat}, lon={lon}) ===")
    tmpl = soil_template_for_lat(lat)
    cells = [(target, lat, lon, tmpl, tmpl)] * NPOI

    YEAR, MONTH = 1990, 6
    gswp3 = read_gswp3_month(YEAR, MONTH, cells)
    lats = np.full(NPOI, lat)
    jday = (date(YEAR, MONTH, 1) - date(YEAR, 1, 1)).days + 1
    day_data = {k: gswp3[k][0, :] for k in gswp3}
    atm_d = diurnal_disaggregate(day_data, 0.0, jday, lats)
    print(f"  GSWP3 forcing: ta={atm_d['ta'][0]:.3f}K  qa={atm_d['qa'][0]:.4e}  "
          f"raina={atm_d['raina'][0]:.2e}  coszen={atm_d['coszen'][0]:.3f}")

    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    c = consts
    ip = params.interception

    # Apply forcing
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

    # --- Verify pre-step state matches ---
    fs0, fn0, fv0 = f77.get_soil(), f77.get_snow(), f77.get_veg()
    pre = []
    cmp("init soil.tg",      grab(fs0["tg"]),         grab(sj.tg),         pre)
    cmp("init soil.tsoi[0]", grab(fs0["tsoi"][:,0]),  grab(sj.tsoi[:,0]),  pre)
    cmp("init soil.wsoi[0]", grab(fs0["wsoi"][:,0]),  grab(sj.wsoi[:,0]),  pre)
    cmp("init snow.fi",      grab(fn0["fi"]),         grab(snj.fi),        pre)
    cmp("init snow.tsno[0]", grab(fn0["tsno"][:,0]),  grab(snj.tsno[:,0]), pre)
    cmp("init canopy.tu",    grab(fv0["tu"]),         grab(cj.tu),         pre)
    cmp("init canopy.t12",   grab(fv0["t12"]),        grab(cj.t12),        pre)
    cmp("init canopy.t34",   grab(fv0["t34"]),        grab(cj.t34),        pre)
    cmp("init canopy.q12",   grab(fv0["q12"]),        grab(cj.q12),        pre)
    show("INITIAL STATE", pre)

    # ============================================================
    # Stage 1: setsoi
    # ============================================================
    f77.call_setsoi()
    sets = setsoi(
        sj.sand, sj.clay, sj.poros, sj.wsoi, sj.wisoi, sj.tsoi,
        sj.wpud, sj.wipud, params.wpudmax, params.zwpmax,
        snj.tsno[:, 0], state.atm.ta,
        c.tmelt, c.hvap, c.hsub, c.ch2o, c.cvap, c.cice, c.epsilon,
    )
    fs1 = f77.get_soil()
    r = []
    for k in ("consoi","hvasug","hvasui","qglif"):
        if k not in fs1: continue
        fa = np.asarray(fs1[k])
        ja = np.asarray(getattr(sets, k, None))
        if ja is None: continue
        if fa.ndim == 1:
            cmp(f"setsoi.{k}", grab(fa), grab(ja), r)
        else:
            for j in range(min(fa.shape[1], 4)):
                cmp(f"setsoi.{k}[{j}]", grab(fa[:,j]), grab(ja[:,j]), r)
    show("1. setsoi", r)

    # ============================================================
    # Stage 2: fwetcal (1st)
    # ============================================================
    f77.call_fwetcal()
    fwet = fwetcal(
        cj.tu, cj.ts, cj.tl,
        cj.wliqu, cj.wliqs, cj.wliql,
        cj.wsnou, cj.wsnos, cj.wsnol,
        ip.wliqumax, ip.wliqsmax, ip.wliqlmax,
        ip.wsnoumax, ip.wsnosmax, ip.wsnolmax,
        ip.fmax, c.tmelt, c.epsilon,
    )
    fv1 = f77.get_veg()
    r = []
    for k in ("fwetu","fwets","fwetl","rliqu","rliqs","rliql"):
        cmp(f"fwet_1.{k}", grab(fv1[k]), grab(getattr(fwet, k)), r)
    show("2. fwetcal (1st)", r)

    # ============================================================
    # Stage 3: radiation
    # ============================================================
    f77.call_radiation()
    rad = compute_radiation(
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
    f_rad = f77.get_radiation()
    r = []
    for k in sorted(f_rad.keys()):
        fa = np.asarray(f_rad[k])
        ja = getattr(rad, k, None)
        if ja is None: continue
        ja = np.asarray(ja)
        if fa.shape != ja.shape: continue
        if fa.size == NPOI:
            cmp(f"rad.{k}", grab(fa), grab(ja), r)
        elif fa.size > NPOI:
            for j in range(min(fa.shape[-1], 3)):
                cmp(f"rad.{k}[{j}]", grab(fa.ravel()[j*NPOI]), grab(ja.ravel()[j*NPOI]), r)
    show("3. radiation", r)

    # ============================================================
    # Stage 4: cascade
    # ============================================================
    f77.call_cascade()
    casc = cascade(
        cj.tu, cj.ts, cj.tl, cj.t12, cj.t34,
        cj.wliqu, cj.wliqs, cj.wliql,
        cj.wsnou, cj.wsnos, cj.wsnol,
        vj.lai, vj.sai, vj.fu, vj.fl,
        state.atm.ta, state.atm.qa, state.atm.psurf,
        state.atm.raina, state.atm.snowa,
        ip,
        DTIME, c.tmelt, c.ch2o, c.cice, c.hfus, c.epsilon,
        c.cair, c.hvap, c.cvap,
    )
    fv4 = f77.get_veg()
    fflx4 = f77.get_fluxes()
    r = []
    for k in ("wliqu","wliqs","wliql","wsnou","wsnos","wsnol"):
        jv = getattr(casc, k, None)
        if jv is not None:
            cmp(f"cascade.{k}", grab(fv4[k]), grab(jv), r)
    for k in ("raing","snowg"):
        jv = getattr(casc, k, None)
        if k in fflx4 and jv is not None:
            cmp(f"cascade.{k}", grab(fflx4[k]), grab(jv), r)
    show("4. cascade", r)

    # ============================================================
    # Stage 5: fwetcal (2nd, after cascade)
    # ============================================================
    f77.call_fwetcal()
    fwet2 = fwetcal(
        cj.tu, cj.ts, cj.tl,
        casc.wliqu, casc.wliqs, casc.wliql,
        casc.wsnou, casc.wsnos, casc.wsnol,
        ip.wliqumax, ip.wliqsmax, ip.wliqlmax,
        ip.wsnoumax, ip.wsnosmax, ip.wsnolmax,
        ip.fmax, c.tmelt, c.epsilon,
    )
    fv5 = f77.get_veg()
    r = []
    for k in ("fwetu","fwets","fwetl","rliqu","rliqs","rliql"):
        cmp(f"fwet_2.{k}", grab(fv5[k]), grab(getattr(fwet2, k)), r)
    show("5. fwetcal (2nd)", r)

    # ============================================================
    # Stage 6: drystress
    # ============================================================
    f77.call_drystress()
    stresstl_j, stresstu_j, stressl_j, stressu_j = drystress(
        sj.wsoi, sj.wisoi, sj.swilt, sj.sfield, vj.froot,
    )
    fst = f77.get_stomata()
    r = []
    cmp("dry.stresstu", grab(fst["stresstu"]), grab(stresstu_j), r)
    cmp("dry.stresstl", grab(fst["stresstl"]), grab(stresstl_j), r)
    if "stressu" in fst:
        cmp("dry.stressu[0]", grab(fst["stressu"][:,0]), grab(stressu_j[:,0]), r)
        cmp("dry.stressl[0]", grab(fst["stressl"][:,0]), grab(stressl_j[:,0]), r)
    show("6. drystress", r)

    # ============================================================
    # Stage 7: canopy (full 3-iter)
    # ============================================================
    print("\n=== Stage 7: canopy (3-iter; not subdividable from outside) ===")
    f77.call_canopy()

    # JAX equivalent: run full timestep_body from initial state with forcing
    state2 = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    state2 = eqx.tree_at(
        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                   s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                   s.atm.raina, s.atm.snowa),
        state2,
        tuple(jnp.asarray(atm_d[k]) for k in
              ("ta","qa","ua","psurf","coszen","solad","solai","fira","raina","snowa")),
    )
    st_after = timestep_body(state2, consts, params, DTIME)

    fv7 = f77.get_veg()
    fs7 = f77.get_soil()
    fn7 = f77.get_snow()
    fx7 = f77.get_fluxes()
    r = []
    for k in ("tu","ts","tl","t12","t34","q12","q34"):
        cmp(f"can.{k}", grab(fv7[k]), grab(getattr(st_after.canopy, k)), r)
    cmp("soil.tg", grab(fs7["tg"]), grab(st_after.soil.tg), r)
    cmp("snow.fi", grab(fn7["fi"]), grab(st_after.snow.fi), r)
    for k in ("fsenu","fsens","fsenl","fseng","fseni"):
        cmp(f"flx.{k}", grab(fx7[k]), grab(getattr(st_after.canopy, k)), r)
    show("7. after F77 call_canopy() / JAX full step", r)


if __name__ == "__main__":
    main()
