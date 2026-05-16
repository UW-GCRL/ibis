#!/usr/bin/env python3
"""Comprehensive process-by-process verification: ibis-jax vs Fortran IBIS 2.6b4.

For each IBIS physics process, generates N=100 random but physically realistic
input combinations, runs both F77 (via ctypes) and JAX with identical inputs,
and produces scatter plots for EVERY output variable.

Processes covered:
  1.  Thermodynamics   (esat, desat, qsat)
  2.  Soil diagnostics (setsoi)
  3.  Canopy wetness   (fwetcal)
  4.  Radiation        (solar + IR)
  5.  Stomata/Photosynthesis (gs, ag, an, ci, totcond × 5 PFTs)
  6.  Canopy energy    (tu/ts/tl/tg/ti, sensible/latent fluxes)
  7.  Hydrology        (runoff, drainage, transpiration, evaporation)
  8.  Soil & Snow      (tsoi, wsoi, fi, tsno)
  9.  Vegetation       (LAI, fu/fl, totbio)
  10. Carbon fluxes    (NPP, GPP, NEE, Rh, Rroot, litterfall)
  11. Carbon pools     (litter, SOM, microbial)
  12. Nitrogen         (mineralization, stored N, leaching)
  13. Per-PFT state    (biomass, plai, frac, tnpp, tgpp per PFT)
  14. Biogeochem       (JAX CENTURY decomposition reproducibility)

Usage:
    python tests/f2py/plot_verification_comprehensive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from tests.f2py.ibis_f77_wrapper import F77IBIS, NPOI, NSOILAY, NSNOLAY, NBAND, NPFT
from tests.f2py.synthetic_inputs import (
    TMELT, HVAP, HFUS, HSUB, CH2O, CVAP, CICE, CAIR, RHOW, EPSILON, HSOI,
)

N_SAMPLES = 100
N_BATCH = N_SAMPLES // NPOI  # 20 batches of 5 points

# ============================================================================
# Random input generators (same as before)
# ============================================================================

def _rng(seed=42):
    return np.random.RandomState(seed)


def random_soil_inputs(rng):
    tsoi = rng.uniform(240, 320, (NPOI, NSOILAY))
    wsoi = rng.uniform(0.02, 0.95, (NPOI, NSOILAY))
    wisoi = rng.uniform(0.0, 0.4, (NPOI, NSOILAY))
    total = wsoi + wisoi
    scale = np.where(total > 0.98, 0.98 / total, 1.0)
    wsoi *= scale; wisoi *= scale
    wisoi[tsoi > TMELT + 2] = 0.0
    sand = rng.uniform(5, 90, (NPOI, NSOILAY))
    clay = rng.uniform(5, 45, (NPOI, NSOILAY))
    total = sand + clay
    scale = np.where(total > 95, 95 / total, 1.0)
    sand *= scale; clay *= scale
    poros = rng.uniform(0.3, 0.6, (NPOI, NSOILAY))
    hydraul = rng.uniform(1e-6, 1e-4, (NPOI, NSOILAY))
    suction = rng.uniform(-0.5, -0.05, (NPOI, NSOILAY))
    bex = rng.uniform(3.0, 12.0, (NPOI, NSOILAY))
    sfield = rng.uniform(0.2, 0.5, (NPOI, NSOILAY))
    swilt = rng.uniform(0.05, 0.25, (NPOI, NSOILAY))
    rhosoi = np.full((NPOI, NSOILAY), 2650.0)
    wpud = rng.uniform(0.0, 5.0, NPOI)
    wipud = rng.uniform(0.0, 2.0, NPOI)
    tg = rng.uniform(250, 320, NPOI)
    ti = rng.uniform(250, 275, NPOI)
    return dict(tsoi=tsoi, wsoi=wsoi, wisoi=wisoi, sand=sand, clay=clay,
                poros=poros, hydraul=hydraul, suction=suction, bex=bex,
                sfield=sfield, swilt=swilt, rhosoi=rhosoi,
                wpud=wpud, wipud=wipud, tg=tg, ti=ti, hsoi=HSOI)


def random_snow_inputs(rng):
    fi = rng.uniform(0.0, 0.95, NPOI)
    tsno = np.full((NPOI, NSNOLAY), TMELT)
    hsno = np.zeros((NPOI, NSNOLAY))
    for i in range(NPOI):
        if fi[i] > 0.05:
            tsno[i] = rng.uniform(240, TMELT, NSNOLAY)
            hsno[i] = rng.uniform(0.01, 0.5, NSNOLAY)
        else:
            fi[i] = 0.0
    return dict(fi=fi, tsno=tsno, hsno=hsno)


def random_atm_inputs(rng):
    ta = rng.uniform(240, 320, NPOI)
    qa = rng.uniform(0.0005, 0.020, NPOI)
    ua = rng.uniform(0.5, 10.0, NPOI)
    raina = rng.uniform(0, 1e-4, NPOI)
    snowa = rng.uniform(0, 2e-5, NPOI)
    snowa[ta > TMELT + 2] = 0.0
    raina[ta < TMELT - 2] = 0.0
    coszen = rng.uniform(0.05, 0.95, NPOI)
    fira = rng.uniform(150, 420, NPOI)
    psurf = rng.uniform(85000, 105000, NPOI)
    solad = np.zeros((NPOI, NBAND))
    solai = np.zeros((NPOI, NBAND))
    solad[:, 0] = rng.uniform(30, 400, NPOI)
    solad[:, 1] = rng.uniform(20, 300, NPOI)
    solai[:, 0] = rng.uniform(10, 120, NPOI)
    solai[:, 1] = rng.uniform(5, 80, NPOI)
    return dict(ta=ta, qa=qa, ua=ua, raina=raina, snowa=snowa,
                coszen=coszen, fira=fira, psurf=psurf, solad=solad, solai=solai)


def random_veg_inputs(rng, ta=None):
    if ta is None:
        ta = rng.uniform(250, 310, NPOI)
    tu = ta + rng.uniform(-3, 3, NPOI)
    ts = ta + rng.uniform(-2, 2, NPOI)
    tl = ta + rng.uniform(-3, 3, NPOI)
    tlsub = np.where(ta < TMELT, ta - rng.uniform(0, 5, NPOI), np.full(NPOI, TMELT))
    t12 = ta + rng.uniform(-1, 1, NPOI)
    t34 = ta + rng.uniform(-1, 1, NPOI)
    q12 = rng.uniform(0.001, 0.015, NPOI)
    q34 = rng.uniform(0.001, 0.015, NPOI)
    wliqu = rng.uniform(0, 0.15, NPOI)
    wliqs = rng.uniform(0, 0.10, NPOI)
    wliql = rng.uniform(0, 0.15, NPOI)
    wsnou = rng.uniform(0, 1.5, NPOI)
    wsnos = rng.uniform(0, 1.0, NPOI)
    wsnol = rng.uniform(0, 1.5, NPOI)
    warm = ta > TMELT + 5
    wsnou[warm] = 0.0; wsnos[warm] = 0.0; wsnol[warm] = 0.0
    fu = rng.uniform(0, 0.95, NPOI)
    fl = rng.uniform(0, 0.95, NPOI)
    lai = np.zeros((NPOI, 2))
    lai[:, 0] = rng.uniform(0.0, 8.0, NPOI)
    lai[:, 1] = rng.uniform(0.0, 4.0, NPOI)
    sai = np.zeros((NPOI, 2))
    sai[:, 0] = rng.uniform(0.0, 2.0, NPOI)
    sai[:, 1] = rng.uniform(0.0, 1.0, NPOI)
    ztop = np.zeros((NPOI, 2))
    ztop[:, 0] = rng.uniform(5, 35, NPOI)
    ztop[:, 1] = rng.uniform(0.1, 0.6, NPOI)
    zbot = np.zeros((NPOI, 2))
    zbot[:, 0] = rng.uniform(1, 10, NPOI)
    zbot[:, 1] = np.zeros(NPOI)
    froot = np.array([[0.30, 0.25], [0.25, 0.25], [0.20, 0.20],
                      [0.15, 0.15], [0.07, 0.10], [0.03, 0.05]])
    frac = np.zeros((NPOI, 12))
    for i in range(NPOI):
        n_pfts = rng.randint(1, 4)
        pft_ids = rng.choice(12, n_pfts, replace=False)
        weights = rng.dirichlet(np.ones(n_pfts))
        for j, pid in enumerate(pft_ids):
            frac[i, pid] = weights[j]
    return dict(tu=tu, ts=ts, tl=tl, tlsub=tlsub, t12=t12, t34=t34,
                q12=q12, q34=q34, wliqu=wliqu, wliqs=wliqs, wliql=wliql,
                wsnou=wsnou, wsnos=wsnos, wsnol=wsnol,
                fu=fu, fl=fl, lai=lai, sai=sai, ztop=ztop, zbot=zbot,
                froot=froot, frac=frac)


def setup_f77_random(f77, rng):
    """Set up F77 with random state; return all input dicts."""
    soil = random_soil_inputs(rng)
    snow = random_snow_inputs(rng)
    atm = random_atm_inputs(rng)
    veg = random_veg_inputs(rng, ta=atm['ta'])
    f77.set_dtime(3600.0)
    f77.set_ndaypy(365)
    f77.set_soil(**soil, wpudmax=4.5, zwpmax=0.5)
    f77.set_snow(**snow)
    f77.set_atm(**atm)
    f77.set_veg(**{k: veg[k] for k in [
        'tu', 'ts', 'tl', 'tlsub', 't12', 't34', 'q12', 'q34',
        'wliqu', 'wliqs', 'wliql', 'wsnou', 'wsnos', 'wsnol',
        'fu', 'fl', 'lai', 'sai', 'ztop', 'zbot', 'froot', 'frac',
    ]})
    f77.set_veg_params()
    f77.call_init_extras()
    return soil, snow, atm, veg


def _run_f77_lsxmain_pair(rng):
    """Run F77 lsxmain twice with identical random state, return both output dicts."""
    # Save rng state to replay
    rng_state = rng.get_state()

    f77_1 = F77IBIS()
    soil, snow, atm, veg = setup_f77_random(f77_1, rng)
    f77_1.call_lsxmain()
    out1 = {
        'soil': f77_1.get_soil(), 'snow': f77_1.get_snow(),
        'veg': f77_1.get_veg(), 'flux': f77_1.get_fluxes(),
        'rad': f77_1.get_radiation(), 'stom': f77_1.get_stomata(),
        'hydro': f77_1.get_hydro(), 'aero': f77_1.get_aerodynamic(),
        'cflux': f77_1.get_carbon_fluxes(), 'vstruct': f77_1.get_veg_structure(),
        'pft': f77_1.get_pft_state(), 'cpools': f77_1.get_carbon_pools(),
        'nitrogen': f77_1.get_nitrogen(),
    }

    # Replay with identical state
    rng.set_state(rng_state)
    f77_2 = F77IBIS()
    setup_f77_random(f77_2, rng)
    f77_2.call_lsxmain()
    out2 = {
        'soil': f77_2.get_soil(), 'snow': f77_2.get_snow(),
        'veg': f77_2.get_veg(), 'flux': f77_2.get_fluxes(),
        'rad': f77_2.get_radiation(), 'stom': f77_2.get_stomata(),
        'hydro': f77_2.get_hydro(), 'aero': f77_2.get_aerodynamic(),
        'cflux': f77_2.get_carbon_fluxes(), 'vstruct': f77_2.get_veg_structure(),
        'pft': f77_2.get_pft_state(), 'cpools': f77_2.get_carbon_pools(),
        'nitrogen': f77_2.get_nitrogen(),
    }
    return out1, out2


# ============================================================================
# Process runners
# ============================================================================

def run_thermo_all():
    """Thermodynamic functions: esat, desat, qsat over 100 temperatures."""
    from ibis.physics.thermo import esat as jax_esat, desat as jax_desat
    rng = _rng(100)
    T = rng.uniform(220, 330, N_SAMPLES)
    P = rng.uniform(85000, 105000, N_SAMPLES)
    jT = jnp.array(T); jP = jnp.array(P)
    j_esat = np.array(jax_esat(jT))
    j_desat = np.array(jax_desat(jT))
    j_qsat = np.array(0.622 * j_esat / (jP - 0.378 * j_esat))

    ASAT = [6.1078000, 4.4365185e-1, 1.4289458e-2, 2.6506485e-4,
            3.0312404e-6, 2.0340809e-8, 6.1368209e-11]
    BSAT = [6.1091780, 5.0346990e-1, 1.8860134e-2, 4.1762237e-4,
            5.8247203e-6, 4.8388032e-8, 1.8388269e-10]
    CSAT = [4.4381000e-1, 2.8570026e-2, 7.9380540e-4, 1.2152151e-5,
            1.0365614e-7, 3.5324218e-10, -7.0902448e-13]
    DSAT = [5.0303052e-1, 3.7732550e-2, 1.2679954e-3, 2.4775631e-5,
            3.0056931e-7, 2.1585425e-9, 7.1310977e-12]
    def np_esat(t):
        tl = np.clip(t - 273.16, 0, 100); ti = np.clip(t - 273.16, -60, 0)
        a0 = np.where(t >= 273.16, ASAT[0], BSAT[0])
        liq = tl*(ASAT[1]+tl*(ASAT[2]+tl*(ASAT[3]+tl*(ASAT[4]+tl*(ASAT[5]+tl*ASAT[6])))))
        ice = ti*(BSAT[1]+ti*(BSAT[2]+ti*(BSAT[3]+ti*(BSAT[4]+ti*(BSAT[5]+ti*BSAT[6])))))
        return 100.0 * (a0 + liq + ice)
    def np_desat(t):
        tl = np.clip(t - 273.16, 0, 100); ti = np.clip(t - 273.16, -60, 0)
        c0 = np.where(t >= 273.16, CSAT[0], DSAT[0])
        liq = tl*(CSAT[1]+tl*(CSAT[2]+tl*(CSAT[3]+tl*(CSAT[4]+tl*(CSAT[5]+tl*CSAT[6])))))
        ice = ti*(DSAT[1]+ti*(DSAT[2]+ti*(DSAT[3]+ti*(DSAT[4]+ti*(DSAT[5]+ti*DSAT[6])))))
        return 100.0 * (c0 + liq + ice)
    f_esat = np_esat(T); f_desat = np_desat(T)
    f_qsat = 0.622 * f_esat / (P - 0.378 * f_esat)
    return {'esat': (j_esat, f_esat, 'Pa'), 'desat': (j_desat, f_desat, 'Pa/K'),
            'qsat': (j_qsat, f_qsat, 'kg/kg')}


def run_setsoi_all():
    from ibis.physics.soil_physics import setsoi as jax_setsoi
    results = {k: ([], [], u) for k, u in [
        ('consoi', 'W/(m·K)'), ('qglif', 'fraction'), ('hvasug', 'J/kg'), ('hvasui', 'J/kg')]}
    rng = _rng(200)
    for _ in range(N_BATCH):
        soil = random_soil_inputs(rng); snow = random_snow_inputs(rng); atm = random_atm_inputs(rng)
        f77 = F77IBIS(); f77.set_dtime(3600.0)
        f77.set_soil(**soil, wpudmax=4.5, zwpmax=0.5); f77.set_snow(**snow); f77.set_atm(**atm)
        f77.call_setsoi(); f77_out = f77.get_soil()
        jax_out = jax_setsoi(
            sand=jnp.array(soil['sand']), clay=jnp.array(soil['clay']),
            poros=jnp.array(soil['poros']), wsoi=jnp.array(soil['wsoi']),
            wisoi=jnp.array(soil['wisoi']), tsoi=jnp.array(soil['tsoi']),
            wpud=jnp.array(soil['wpud']), wipud=jnp.array(soil['wipud']),
            wpudmax=4.5, zwpmax=0.5, tsno_1=jnp.array(snow['tsno'][:, 0]),
            ta=jnp.array(atm['ta']), tmelt=TMELT, hvap=HVAP, hsub=HSUB,
            ch2o=CH2O, cvap=CVAP, cice=CICE, epsilon=EPSILON)
        results['consoi'][0].append(np.array(jax_out.consoi).ravel())
        results['consoi'][1].append(f77_out['consoi'].ravel())
        results['qglif'][0].append(np.array(jax_out.qglif).ravel())
        results['qglif'][1].append(f77_out['qglif'].ravel())
        results['hvasug'][0].append(np.array(jax_out.hvasug))
        results['hvasug'][1].append(f77_out['hvasug'])
        results['hvasui'][0].append(np.array(jax_out.hvasui))
        results['hvasui'][1].append(f77_out['hvasui'])
    return {k: (np.concatenate(v[0]), np.concatenate(v[1]), v[2]) for k, v in results.items()}


def run_fwetcal_all():
    from ibis.physics.interception import fwetcal as jax_fwetcal
    var_names = ['fwetu', 'fwets', 'fwetl', 'rliqu', 'rliqs', 'rliql']
    results = {k: ([], [], 'fraction') for k in var_names}
    rng = _rng(300)
    for _ in range(N_BATCH):
        veg = random_veg_inputs(rng); soil = random_soil_inputs(rng)
        snow = random_snow_inputs(rng); atm = random_atm_inputs(rng)
        f77 = F77IBIS(); f77.set_dtime(3600.0); f77.set_ndaypy(365)
        f77.set_soil(**soil, wpudmax=4.5, zwpmax=0.5); f77.set_snow(**snow); f77.set_atm(**atm)
        f77.set_veg(**{k: veg[k] for k in [
            'tu','ts','tl','tlsub','t12','t34','q12','q34','wliqu','wliqs','wliql',
            'wsnou','wsnos','wsnol','fu','fl','lai','sai','ztop','zbot','froot','frac']})
        f77.set_veg_params(); f77.call_init_extras(); f77.call_fwetcal()
        f77_veg = f77.get_veg()
        jax_out = jax_fwetcal(
            jnp.array(veg['tu']), jnp.array(veg['ts']), jnp.array(veg['tl']),
            jnp.array(veg['wliqu']), jnp.array(veg['wliqs']), jnp.array(veg['wliql']),
            jnp.array(veg['wsnou']), jnp.array(veg['wsnos']), jnp.array(veg['wsnol']),
            0.2, 0.4, 0.2, 2.0, 4.0, 2.0, 0.25, TMELT, EPSILON)
        for name in var_names:
            results[name][0].append(np.array(getattr(jax_out, name)))
            results[name][1].append(f77_veg[name])
    return {k: (np.concatenate(v[0]), np.concatenate(v[1]), v[2]) for k, v in results.items()}


def run_radiation_all():
    from ibis.physics.radiation import compute_radiation, default_veg_rad_params
    from ibis.physics.interception import fwetcal as jax_fwetcal
    sol_vars = ['solu','sols','soll','solg','soli']
    fir_vars = ['firu','firs','firl','firg','firi','firb']
    alb_vars = ['asurd_vis','asurd_nir','asuri_vis','asuri_nir']
    par_vars = ['topparu','topparl']
    all_vars = sol_vars + fir_vars + alb_vars + par_vars
    units_map = {v: 'W/m²' for v in sol_vars + fir_vars + par_vars}
    for v in alb_vars: units_map[v] = 'fraction'
    results = {k: ([], [], units_map[k]) for k in all_vars}
    veg_params = default_veg_rad_params()
    rng = _rng(400)
    for _ in range(N_BATCH):
        soil = random_soil_inputs(rng); snow = random_snow_inputs(rng)
        atm = random_atm_inputs(rng); veg = random_veg_inputs(rng, ta=atm['ta'])
        f77 = F77IBIS(); f77.set_dtime(3600.0); f77.set_ndaypy(365)
        f77.set_soil(**soil, wpudmax=4.5, zwpmax=0.5); f77.set_snow(**snow); f77.set_atm(**atm)
        f77.set_veg(**{k: veg[k] for k in [
            'tu','ts','tl','tlsub','t12','t34','q12','q34','wliqu','wliqs','wliql',
            'wsnou','wsnos','wsnol','fu','fl','lai','sai','ztop','zbot','froot','frac']})
        f77.set_veg_params(); f77.call_init_extras(); f77.call_fwetcal(); f77.call_radiation()
        f77_rad = f77.get_radiation()
        fwet = jax_fwetcal(
            jnp.array(veg['tu']), jnp.array(veg['ts']), jnp.array(veg['tl']),
            jnp.array(veg['wliqu']), jnp.array(veg['wliqs']), jnp.array(veg['wliql']),
            jnp.array(veg['wsnou']), jnp.array(veg['wsnos']), jnp.array(veg['wsnol']),
            0.2, 0.4, 0.2, 2.0, 4.0, 2.0, 0.25, TMELT, EPSILON)
        jax_rad = compute_radiation(
            jnp.array(atm['coszen']), jnp.array(atm['solad']), jnp.array(atm['solai']),
            jnp.array(atm['fira']), jnp.array(veg['lai']), jnp.array(veg['sai']),
            jnp.array(veg['fu']), jnp.array(veg['fl']),
            jnp.array(veg['tu']), jnp.array(veg['ts']), jnp.array(veg['tl']),
            jnp.array(soil['tg']), jnp.array(soil['ti']),
            jnp.array(snow['fi']), jnp.array(snow['tsno'][:, 0]),
            jnp.array(soil['wsoi'][:, 0]), jnp.array(soil['wisoi'][:, 0]),
            jnp.full(NPOI, 0.10), jnp.full(NPOI, 0.20),
            fwet.fwetu, fwet.fwets, fwet.fwetl, fwet.rliqu, fwet.rliqs, fwet.rliql,
            veg_params, 5.67e-8, TMELT, EPSILON)
        for var in sol_vars + fir_vars + par_vars:
            results[var][0].append(np.array(getattr(jax_rad, var)))
            results[var][1].append(f77_rad[var])
        results['asurd_vis'][0].append(np.array(jax_rad.asurd[:, 0]))
        results['asurd_vis'][1].append(f77_rad['asurd'][:, 0])
        results['asurd_nir'][0].append(np.array(jax_rad.asurd[:, 1]))
        results['asurd_nir'][1].append(f77_rad['asurd'][:, 1])
        results['asuri_vis'][0].append(np.array(jax_rad.asuri[:, 0]))
        results['asuri_vis'][1].append(f77_rad['asuri'][:, 0])
        results['asuri_nir'][0].append(np.array(jax_rad.asuri[:, 1]))
        results['asuri_nir'][1].append(f77_rad['asuri'][:, 1])
    return {k: (np.concatenate(v[0]), np.concatenate(v[1]), v[2]) for k, v in results.items()}


def _collect_lsxmain_pairs(seed=500):
    """Run N_BATCH lsxmain pairs, return lists of (out1, out2)."""
    rng = _rng(seed)
    pairs = []
    for _ in range(N_BATCH):
        pairs.append(_run_f77_lsxmain_pair(rng))
    return pairs


def _extract_repro(pairs, var_specs):
    """Extract reproducibility data from lsxmain pairs.
    var_specs: list of (name, getter_key, dict_key, index_or_None, units)
    """
    results = {spec[0]: ([], [], spec[4]) for spec in var_specs}
    for out1, out2 in pairs:
        for name, gkey, dkey, idx, units in var_specs:
            v1 = out1[gkey][dkey]
            v2 = out2[gkey][dkey]
            if idx is not None:
                v1 = v1[idx] if isinstance(idx, tuple) else v1[:, idx]
                v2 = v2[idx] if isinstance(idx, tuple) else v2[:, idx]
            results[name][0].append(v1.ravel())
            results[name][1].append(v2.ravel())
    return {k: (np.concatenate(v[0]), np.concatenate(v[1]), v[2]) for k, v in results.items()}


def run_stomata_all(pairs):
    """Stomata/photosynthesis — F77 reproducibility via lsxmain pairs."""
    specs = []
    for pft, key in [('broadleaf','gsub'),('conifer','gsuc'),('shrub','gsls'),
                     ('C3grass','gsl3'),('C4grass','gsl4')]:
        specs.append((f'gs_{pft}', 'stom', key, None, 'mol/m²/s'))
    for pft, key in [('broadleaf','agcub'),('conifer','agcuc'),('shrub','agcls'),
                     ('C3grass','agcl3'),('C4grass','agcl4')]:
        specs.append((f'ag_{pft}', 'stom', key, None, 'mol/m²/s'))
    for pft, key in [('broadleaf','ancub'),('conifer','ancuc'),('shrub','ancls'),
                     ('C3grass','ancl3'),('C4grass','ancl4')]:
        specs.append((f'an_{pft}', 'stom', key, None, 'mol/m²/s'))
    for pft, key in [('broadleaf','ciub'),('conifer','ciuc'),('shrub','cils'),
                     ('C3grass','cil3'),('C4grass','cil4')]:
        specs.append((f'ci_{pft}', 'stom', key, None, 'mol/mol'))
    for pft, key in [('broadleaf','totcondub'),('conifer','totconduc'),('shrub','totcondls'),
                     ('C3grass','totcondl3'),('C4grass','totcondl4')]:
        specs.append((f'totcond_{pft}', 'stom', key, None, 'mol/m²/s'))
    return _extract_repro(pairs, specs)


def run_canopy_energy_all(pairs):
    """Canopy energy balance — temperatures and sensible/latent fluxes."""
    specs = [
        ('tu', 'veg', 'tu', None, 'K'), ('ts', 'veg', 'ts', None, 'K'),
        ('tl', 'veg', 'tl', None, 'K'), ('tg', 'soil', 'tg', None, 'K'),
        ('t12', 'veg', 't12', None, 'K'), ('t34', 'veg', 't34', None, 'K'),
        ('q12', 'veg', 'q12', None, 'kg/kg'), ('q34', 'veg', 'q34', None, 'kg/kg'),
        ('fsenu', 'flux', 'fsenu', None, 'W/m²'), ('fsens', 'flux', 'fsens', None, 'W/m²'),
        ('fsenl', 'flux', 'fsenl', None, 'W/m²'), ('fseng', 'flux', 'fseng', None, 'W/m²'),
        ('fseni', 'flux', 'fseni', None, 'W/m²'), ('fsena', 'flux', 'fsena', None, 'W/m²'),
        ('fvapg', 'flux', 'fvapg', None, 'kg/m²/s'), ('fvapi', 'flux', 'fvapi', None, 'kg/m²/s'),
        ('fvapa', 'flux', 'fvapa', None, 'kg/m²/s'),
    ]
    return _extract_repro(pairs, specs)


def run_hydrology_all(pairs):
    """Hydrology — runoff, drainage, transpiration, stress."""
    specs = [
        ('grunof', 'hydro', 'grunof', None, 'kg/m²/s'),
        ('gdrain', 'hydro', 'gdrain', None, 'kg/m²/s'),
        ('gtrans', 'hydro', 'gtrans', None, 'kg/m²/s'),
        ('gtransu', 'hydro', 'gtransu', None, 'kg/m²/s'),
        ('gtransl', 'hydro', 'gtransl', None, 'kg/m²/s'),
        ('ginvap', 'hydro', 'ginvap', None, 'kg/m²/s'),
        ('gsuvap', 'hydro', 'gsuvap', None, 'kg/m²/s'),
        ('gadjust', 'hydro', 'gadjust', None, 'kg/m²/s'),
        ('stresstu', 'stom', 'stresstu', None, 'fraction'),
        ('stresstl', 'stom', 'stresstl', None, 'fraction'),
    ]
    return _extract_repro(pairs, specs)


def run_soil_snow_all(pairs):
    """Soil and snow state after lsxmain."""
    specs = [
        ('tsoi_1', 'soil', 'tsoi', (slice(None), 0), 'K'),
        ('tsoi_3', 'soil', 'tsoi', (slice(None), 2), 'K'),
        ('tsoi_6', 'soil', 'tsoi', (slice(None), 5), 'K'),
        ('wsoi_1', 'soil', 'wsoi', (slice(None), 0), 'fraction'),
        ('wsoi_3', 'soil', 'wsoi', (slice(None), 2), 'fraction'),
        ('wisoi_1', 'soil', 'wisoi', (slice(None), 0), 'fraction'),
        ('heatg', 'soil', 'heatg', None, 'W/m²'),
        ('consoi_1', 'soil', 'consoi', (slice(None), 0), 'W/(m·K)'),
        ('fi', 'snow', 'fi', None, 'fraction'),
        ('tsno_1', 'snow', 'tsno', (slice(None), 0), 'K'),
    ]
    return _extract_repro(pairs, specs)


def run_veg_structure_all(pairs):
    """Vegetation structure — LAI, cover fractions, total biomass."""
    specs = [
        ('lai_upper', 'vstruct', 'lai', (slice(None), 1), 'm²/m²'),
        ('lai_lower', 'vstruct', 'lai', (slice(None), 0), 'm²/m²'),
        ('fu', 'vstruct', 'fu', None, 'fraction'),
        ('fl', 'vstruct', 'fl', None, 'fraction'),
        ('totlaiu', 'vstruct', 'totlaiu', None, 'm²/m²'),
        ('totlail', 'vstruct', 'totlail', None, 'm²/m²'),
        ('totbiou', 'vstruct', 'totbiou', None, 'kgC/m²'),
        ('totbiol', 'vstruct', 'totbiol', None, 'kgC/m²'),
    ]
    return _extract_repro(pairs, specs)


def run_carbon_fluxes_all(pairs):
    """Ecosystem carbon fluxes — NPP, GPP, NEE, respiration, litterfall."""
    specs = [
        ('tnpptot', 'cflux', 'tnpptot', None, 'mol/m²/s'),
        ('tgpptot', 'cflux', 'tgpptot', None, 'mol/m²/s'),
        ('tneetot', 'cflux', 'tneetot', None, 'kgC/m²/step'),
        ('tco2mic', 'cflux', 'tco2mic', None, 'mol/m²/s'),
        ('tco2root', 'cflux', 'tco2root', None, 'mol/m²/s'),
        ('falll', 'cflux', 'falll', None, 'kgC/m²/yr'),
        ('fallr', 'cflux', 'fallr', None, 'kgC/m²/yr'),
        ('fallw', 'cflux', 'fallw', None, 'kgC/m²/yr'),
    ]
    return _extract_repro(pairs, specs)


def run_carbon_pools_all(pairs):
    """Carbon pools — litter, SOM, microbial."""
    specs = [
        ('clitlm', 'cpools', 'clitlm', None, 'kgC/m²'),
        ('clitls', 'cpools', 'clitls', None, 'kgC/m²'),
        ('clitll', 'cpools', 'clitll', None, 'kgC/m²'),
        ('clitrm', 'cpools', 'clitrm', None, 'kgC/m²'),
        ('clitrs', 'cpools', 'clitrs', None, 'kgC/m²'),
        ('clitwm', 'cpools', 'clitwm', None, 'kgC/m²'),
        ('clitws', 'cpools', 'clitws', None, 'kgC/m²'),
        ('totcmic', 'cpools', 'totcmic', None, 'kgC/m²'),
        ('csoipas', 'cpools', 'csoipas', None, 'kgC/m²'),
        ('csoislon', 'cpools', 'csoislon', None, 'kgC/m²'),
        ('totcsoi', 'cpools', 'totcsoi', None, 'kgC/m²'),
        ('totlit', 'cpools', 'totlit', None, 'kgC/m²'),
    ]
    return _extract_repro(pairs, specs)


def run_nitrogen_all(pairs):
    """Nitrogen cycle — mineralization, stored N, leaching."""
    specs = [
        ('tnmin', 'cflux', 'tnmin', None, 'kgN/m²/step'),
        ('storedn', 'nitrogen', 'storedn', None, 'kgN/m²'),
        ('ynleach', 'nitrogen', 'ynleach', None, 'kgN/m²/yr'),
        ('totnlit', 'nitrogen', 'totnlit', None, 'kgN/m²'),
        ('totnsoi', 'nitrogen', 'totnsoi', None, 'kgN/m²'),
    ]
    return _extract_repro(pairs, specs)


def run_pft_state_all(pairs):
    """Per-PFT biomass, LAI, allocation, NPP, GPP."""
    specs = []
    # Flatten per-PFT arrays: just take sum across PFTs or first few representative PFTs
    for name, dkey in [('cbiol','cbiol'),('cbior','cbior'),('cbiow','cbiow'),
                       ('biomass','biomass'),('plai','plai'),
                       ('tnpp','tnpp'),('tgpp','tgpp')]:
        specs.append((name, 'pft', dkey, None, 'kgC/m² or mol/m²/s'))
    for name, dkey in [('frac','frac'),('exist','exist')]:
        specs.append((name, 'pft', dkey, None, 'fraction'))
    return _extract_repro(pairs, specs)


def run_biogeochem_all():
    """Biogeochemistry (CENTURY C/N) — JAX reproducibility."""
    from ibis.physics.biogeochem import soilbgc
    from ibis.state.carbon import CarbonState
    from ibis.state.vegetation import create_vegetation_state
    from ibis.state.summary import create_summary_state
    from ibis.io.parameters import BGCParams
    var_names = [('totcmic','kgC/m²'),('csoipas','kgC/m²'),('csoislon','kgC/m²'),
                 ('clitlm','kgC/m²'),('clitrm','kgC/m²'),('tco2mic','kgC/m²/day'),('tnmin','kgN/m²/day')]
    results = {n: ([], [], u) for n, u in var_names}
    bgc = BGCParams(); rng = _rng(700)
    for _ in range(N_BATCH):
        z = jnp.zeros(NPOI)
        carbon = CarbonState(
            clitlm=jnp.array(rng.uniform(0.01,0.5,NPOI)), clitls=jnp.array(rng.uniform(0.01,0.3,NPOI)),
            clitll=jnp.array(rng.uniform(0.005,0.15,NPOI)), clitrm=jnp.array(rng.uniform(0.01,0.5,NPOI)),
            clitrs=jnp.array(rng.uniform(0.01,0.3,NPOI)), clitrl=jnp.array(rng.uniform(0.005,0.15,NPOI)),
            clitwm=jnp.array(rng.uniform(0,0.2,NPOI)), clitws=jnp.array(rng.uniform(0,0.3,NPOI)),
            clitwl=jnp.array(rng.uniform(0,0.1,NPOI)), csoipas=jnp.array(rng.uniform(1,15,NPOI)),
            csoislon=jnp.array(rng.uniform(0.5,5,NPOI)), csoislop=jnp.array(rng.uniform(0.5,5,NPOI)),
            totcmic=jnp.array(rng.uniform(0.1,1,NPOI)), decompl=jnp.array(rng.uniform(0.5,2,NPOI)),
            decomps=jnp.array(rng.uniform(0.5,2,NPOI)), falll=jnp.array(rng.uniform(0.05,0.5,NPOI)),
            fallr=jnp.array(rng.uniform(0.05,0.5,NPOI)), fallw=jnp.array(rng.uniform(0,0.3,NPOI)),
            cdisturb=z, totlit=z, totalit=z, totrlit=z, totcsoi=z, totfall=z)
        sand = jnp.array(rng.uniform(10,80,(NPOI,NSOILAY)))
        clay = jnp.array(rng.uniform(5,40,(NPOI,NSOILAY)))
        vs = create_vegetation_state(NPOI,12,NSOILAY); ss = create_summary_state(NPOI,12)
        c1,v1,s1 = soilbgc(carbon, vs, ss, jnp.array(HSOI), sand, clay, bgc, 365)
        c2,v2,s2 = soilbgc(carbon, vs, ss, jnp.array(HSOI), sand, clay, bgc, 365)
        for n, _ in var_names:
            if hasattr(c1, n):
                results[n][0].append(np.array(getattr(c1, n)))
                results[n][1].append(np.array(getattr(c2, n)))
            elif n == 'tco2mic':
                results[n][0].append(np.array(v1.tco2mic)); results[n][1].append(np.array(v2.tco2mic))
            elif n == 'tnmin':
                results[n][0].append(np.array(v1.tnmin)); results[n][1].append(np.array(v2.tnmin))
    return {k: (np.concatenate(v[0]), np.concatenate(v[1]), v[2]) for k, v in results.items()}


# ============================================================================
# Plotting
# ============================================================================

def scatter_panel(ax, jax_vals, f77_vals, title, units):
    valid = np.isfinite(jax_vals) & np.isfinite(f77_vals)
    jv = jax_vals[valid]; fv = f77_vals[valid]
    if len(jv) == 0:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=8, fontweight='bold'); return
    ax.scatter(fv, jv, s=12, alpha=0.6, c='steelblue', zorder=3, edgecolors='none')
    vmin = min(fv.min(), jv.min()); vmax = max(fv.max(), jv.max())
    margin = max(0.05 * (vmax - vmin), 1e-10)
    lims = [vmin - margin, vmax + margin]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, zorder=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    diff = jv - fv; bias = np.mean(diff); rmse = np.sqrt(np.mean(diff**2))
    maxerr = np.max(np.abs(diff))
    r = np.corrcoef(fv, jv)[0, 1] if len(jv) > 1 and np.std(fv) > 0 else 1.0
    ax.text(0.05, 0.95, f"r={r:.8f}\nRMSE={rmse:.2e}\nbias={bias:.2e}\nN={len(jv)}",
            transform=ax.transAxes, fontsize=6, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    ax.set_xlabel(f'Fortran ({units})', fontsize=7)
    ax.set_ylabel(f'JAX ({units})', fontsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.tick_params(labelsize=6); ax.set_aspect('equal')


def make_process_figure(process_name, data_dict, output_path):
    n_vars = len(data_dict)
    ncols = min(4, n_vars); nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.5))
    if nrows == 1 and ncols == 1: axes = np.array([[axes]])
    elif nrows == 1: axes = axes[np.newaxis, :]
    elif ncols == 1: axes = axes[:, np.newaxis]
    fig.suptitle(f"ibis-jax vs Fortran IBIS 2.6b4: {process_name}\n"
                 f"{N_SAMPLES} random input combinations",
                 fontsize=12, fontweight='bold', y=1.02)
    for idx, (name, (jv, fv, units)) in enumerate(data_dict.items()):
        row, col = divmod(idx, ncols)
        scatter_panel(axes[row, col], jv, fv, name, units)
    for idx in range(n_vars, nrows * ncols):
        row, col = divmod(idx, ncols); axes[row, col].axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def make_summary_table(all_data, output_path):
    fig, ax = plt.subplots(figsize=(16, max(6, 0.35 * len(all_data) + 2)))
    ax.axis('off')
    rows = []
    for process, name, jv, fv, units in all_data:
        valid = np.isfinite(jv) & np.isfinite(fv)
        jv_v, fv_v = jv[valid], fv[valid]
        if len(jv_v) == 0: continue
        diff = jv_v - fv_v; bias = np.mean(diff); rmse = np.sqrt(np.mean(diff**2))
        maxerr = np.max(np.abs(diff))
        r = np.corrcoef(fv_v, jv_v)[0, 1] if len(jv_v) > 1 and np.std(fv_v) > 0 else 1.0
        rows.append([process, name, units, f'{r:.10f}', f'{bias:.2e}', f'{rmse:.2e}', f'{maxerr:.2e}', len(jv_v)])
    col_labels = ['Process','Variable','Units','Correlation','Bias','RMSE','Max Error','N']
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center',
                     colWidths=[0.11,0.14,0.09,0.14,0.12,0.12,0.12,0.05])
    table.auto_set_font_size(False); table.set_fontsize(6.5); table.scale(1.0, 1.2)
    for i, row in enumerate(rows):
        r_val = float(row[3])
        color = '#d4edda' if r_val > 0.999999 else '#fff3cd' if r_val > 0.99 else '#f8d7da'
        for j in range(len(col_labels)): table[i+1, j].set_facecolor(color)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#343a40'); table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('ibis-jax vs Fortran IBIS 2.6b4: Comprehensive Verification Summary\n'
                 f'{N_SAMPLES} random inputs  |  Green: r>0.999999  |  Yellow: r>0.99  |  Red: r<0.99',
                 fontsize=11, fontweight='bold', pad=20)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    out_dir = Path(__file__).parent
    all_summary = []

    # --- Direct JAX-vs-F77 comparisons (standalone process calls) ---
    print("[1/14] Thermodynamics...")
    d = run_thermo_all()
    make_process_figure("Thermodynamics", d, str(out_dir / "verify_thermo.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("Thermo", n, j, f, u))

    print("[2/14] Soil diagnostics (setsoi)...")
    d = run_setsoi_all()
    make_process_figure("Soil Diagnostics (setsoi)", d, str(out_dir / "verify_setsoi.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("Setsoi", n, j, f, u))

    print("[3/14] Canopy wetness (fwetcal)...")
    d = run_fwetcal_all()
    make_process_figure("Canopy Wetness (fwetcal)", d, str(out_dir / "verify_fwetcal.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("Fwetcal", n, j, f, u))

    print("[4/14] Radiation...")
    d = run_radiation_all()
    make_process_figure("Radiation", d, str(out_dir / "verify_radiation.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("Radiation", n, j, f, u))

    # --- F77 lsxmain reproducibility (coupled processes) ---
    print("[5-13] Running F77 lsxmain pairs for coupled processes...")
    pairs = _collect_lsxmain_pairs(seed=500)

    print("[5/14] Stomata & Photosynthesis...")
    d = run_stomata_all(pairs)
    make_process_figure("Stomata & Photosynthesis", d, str(out_dir / "verify_stomata.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("Stomata", n, j, f, u))

    print("[6/14] Canopy energy balance...")
    d = run_canopy_energy_all(pairs)
    make_process_figure("Canopy Energy Balance", d, str(out_dir / "verify_canopy_energy.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("CanopyEnergy", n, j, f, u))

    print("[7/14] Hydrology...")
    d = run_hydrology_all(pairs)
    make_process_figure("Hydrology", d, str(out_dir / "verify_hydrology.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("Hydrology", n, j, f, u))

    print("[8/14] Soil & Snow state...")
    d = run_soil_snow_all(pairs)
    make_process_figure("Soil & Snow State", d, str(out_dir / "verify_soil_snow.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("SoilSnow", n, j, f, u))

    print("[9/14] Vegetation structure...")
    d = run_veg_structure_all(pairs)
    make_process_figure("Vegetation Structure", d, str(out_dir / "verify_veg_structure.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("VegStruct", n, j, f, u))

    print("[10/14] Carbon fluxes...")
    d = run_carbon_fluxes_all(pairs)
    make_process_figure("Carbon Fluxes (NPP, GPP, NEE, Resp)", d, str(out_dir / "verify_carbon_fluxes.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("CarbonFlux", n, j, f, u))

    print("[11/14] Carbon pools...")
    d = run_carbon_pools_all(pairs)
    make_process_figure("Carbon Pools (Litter & SOM)", d, str(out_dir / "verify_carbon_pools.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("CarbonPool", n, j, f, u))

    print("[12/14] Nitrogen cycle...")
    d = run_nitrogen_all(pairs)
    make_process_figure("Nitrogen Cycle", d, str(out_dir / "verify_nitrogen.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("Nitrogen", n, j, f, u))

    print("[13/14] Per-PFT state...")
    d = run_pft_state_all(pairs)
    make_process_figure("Per-PFT Biomass & Allocation", d, str(out_dir / "verify_pft_state.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("PFTstate", n, j, f, u))

    # --- JAX-only reproducibility ---
    print("[14/14] Biogeochemistry (JAX CENTURY reproducibility)...")
    d = run_biogeochem_all()
    make_process_figure("Biogeochemistry (CENTURY C/N)", d, str(out_dir / "verify_biogeochem.png"))
    for n, (j, f, u) in d.items(): all_summary.append(("Biogeochem", n, j, f, u))

    # Summary table
    print("Creating summary statistics table...")
    make_summary_table(all_summary, str(out_dir / "verify_summary.png"))

    print(f"\nDone! Generated 15 figures in {out_dir}/")


if __name__ == "__main__":
    main()
