#!/usr/bin/env python3
"""Historical transient at the FLUXNET cells, from the spin-up equilibrium.

Why this exists. `both_fix` is a 300-year spin-up that CYCLES 1901-2000
forcing at a fixed 350 ppm CO2. Its end state is equilibrated to the
20th-century mean climate -- not to the trajectory that actually produced the
ecosystem a tower measured in 2003. Validating against towers straight off a
spin-up restart therefore compares the right weather against the wrong state.

This runs the missing middle step: spin-up -> transient 1901..N with real
year-by-year GSWP3, rising CO2, and `dynaveg` so vegetation and the carbon
pools follow the historical path -- then hands on a restart for evaluation.

Why it is legitimate to do this at 205 cells instead of 95,333: IBIS has no
lateral coupling. No runoff routing, no groundwater exchange, no advection
between columns. Each land point integrates independently, so subsetting the
global equilibrium and running the transient on the subset is identical to
running it globally and subsetting afterwards -- at 1/465th the cost.

Usage:
  ibis-python verify/calibration/site_transient.py --to-year 1996
  ibis-python verify/calibration/site_transient.py --to-year 1996 \
      --eval-year0 1997 --n-eval 10
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port")

import equinox as eqx
import jax
import jax.numpy as jnp

from ibis.constants import PhysicalConstants
from ibis.dimensions import GridConfig
from ibis.io.co2 import apply_co2, co2_ppm
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.restart import load_state, save_state
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.calendar import is_leap_year
from ibis.timestepping.loops_gswp3 import (climate_stats_from_monthly,
                                           run_month_gswp3, _cast_tree_f32)
from ibis.physics.vegetation_dynamics import dynaveg, biomass_update
from ibis.physics.climate_analysis import climanl2, existence

sys.path.insert(0, f"{REPO}/verify/calibration")
from stage1_baseline import (load_sites, LAT, LON, MASK, NPOI_FULL, RESTART,
                             PARAMS, GSWP3)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

AY = ("aygpptot", "ayaet", "ayprcp", "ayco2mic", "ayco2root", "ayanpptot",
      "aynpptot", "ayneetot", "aytrunof", "aytdrain")


def reset_annual(state, npoi):
    from ibis.state.stress_accum import reset_stress_accum
    state = reset_stress_accum(state)
    from ibis.state.et_parts import reset_et_parts
    state = reset_et_parts(state)
    z = jnp.zeros((npoi,))
    return eqx.tree_at(
        lambda s: tuple(getattr(s.summary, n) for n in AY) + (s.summary.aynpp,),
        state, tuple(z for _ in AY) + (jnp.zeros((npoi, 12)),))


def one_year(state, key, year, ctx, isimveg=1):
    """One transient year: 12 months, climate normals, existence, dynaveg."""
    c = ctx
    npoi = len(c["i"])
    state = reset_annual(state, npoi)
    state = apply_co2(state, year)
    leap = is_leap_year(year)
    tmean = []
    gpp_m = []          # (12, npoi) monthly GPP, kgC m-2 month-1, from the daily diag
    gpp_d = []          # (ndays, npoi) daily GPP, kgC m-2 day-1 (phenology timing at towers)
    aet_m = []          # (12, npoi) monthly AET, mm month-1 (dd[1] is daily AET)
    ep_m = []           # (12, 4, npoi) monthly ET components from the ETParts accumulator, if attached
    _parts = lambda s_: (np.stack([np.asarray(getattr(s_.etparts, f)) for f in ("trans", "interc", "soilev", "snowsub")])
                         if getattr(s_, "etparts", None) is not None else None)
    ep_prev = _parts(state)
    # monthly NEE (+ = source) from the running annual sums, NPP per step and Rh per day, zeroed by reset_annual:
    # NEE = -(NPP - Rh), the F77 annual definition (ayneetot below) resolved by month
    _nep = lambda s_: np.asarray(s_.summary.aynpptot) - np.asarray(s_.summary.ayco2mic)
    nep_prev = _nep(state); nee_m = []
    for im in range(12):
        raw = c["reader"].read_month(year, im, c["i"], c["j"])
        tmean.append(np.nanmean(raw["tas"], axis=0) - 273.16)
        state, key, dd = run_month_gswp3(
            state, c["consts"], c["sp"], c["reader"], c["i"], c["j"],
            c["lat_rad"], year, im, key, 3600.0, leap=leap, nband=2,
            nsoilcall=1)
        # dd = (daily_gpp, daily_aet, daily_rh), each (ndays, npoi); it was
        # being discarded, which is why nothing seasonal was ever scored.
        gpp_m.append(np.asarray(dd[0]).sum(axis=0))
        gpp_d.append(np.asarray(dd[0]))
        aet_m.append(np.asarray(dd[1]).sum(axis=0))
        if ep_prev is not None:
            ep_now = _parts(state); ep_m.append(ep_now - ep_prev); ep_prev = ep_now
        nep_now = _nep(state); nee_m.append(-(nep_now - nep_prev)); nep_prev = nep_now

    tct, twt, g0t, g5t = climate_stats_from_monthly(np.stack(tmean, axis=1), leap)
    state = eqx.tree_at(
        lambda s: (s.vegetation.tcthis, s.vegetation.twthis,
                   s.vegetation.gdd0this, s.vegetation.gdd5this),
        state, (jnp.array(tct), jnp.array(twt), jnp.array(g0t), jnp.array(g5t)))

    v = state.vegetation
    tc, tw, tcmin, g0, g5 = climanl2(v.tc, v.tw, v.tcmin, v.gdd0, v.gdd5,
                                     v.tcthis, v.twthis, v.gdd0this,
                                     v.gdd5this, jnp.zeros(npoi))
    state = eqx.tree_at(
        lambda s: (s.vegetation.tc, s.vegetation.tw, s.vegetation.tcmin,
                   s.vegetation.gdd0, s.vegetation.gdd5),
        state, (tc, tw, tcmin, g0, g5))
    state = eqx.tree_at(
        lambda s: s.vegetation.exist, state,
        existence(state.vegetation.tcmin, state.vegetation.tw,
                  state.vegetation.gdd0, state.vegetation.gdd5,
                  c["pft"]))

    # NEE the F77 way, before dynaveg consumes it.
    state = eqx.tree_at(
        lambda s: s.summary.ayneetot, state,
        state.summary.aynpptot - state.summary.ayco2mic)

    diag = {n: np.asarray(getattr(state.summary, n)).copy() for n in AY}
    diag["lai"] = np.asarray(state.vegetation.plai).sum(axis=1)
    diag["gpp_monthly"] = np.stack(gpp_m, axis=0)          # (12, npoi)
    diag["gpp_daily"] = np.concatenate(gpp_d, axis=0)      # (365|366, npoi), kgC m-2 day-1
    diag["aet_monthly"] = np.stack(aet_m, axis=0)          # (12, npoi), mm
    diag["nee_monthly"] = np.stack(nee_m, axis=0)          # (12, npoi), kgC m-2 month-1, + = source; sums to -ayneetot
    if ep_m:
        diag["et_parts_monthly"] = np.stack(ep_m, axis=0)  # (12, 4, npoi): trans, interc, soilev, snowsub (mm)
        diag["et_parts"] = ep_prev                          # (4, npoi) annual

    if isimveg > 0:
        from ibis.state.stress_accum import tree_survival
        nv, nc, ns = dynaveg(state.vegetation, state.carbon, state.summary,
                             c["pft"], 1, c["consts"].epsilon,
                             fire_scheme="seasonlength",
                             tree_surv=tree_survival(state, c["pft"]))
    else:
        nv, nc, ns = biomass_update(state.vegetation, state.carbon,
                                    state.summary, c["pft"],
                                    c["consts"].epsilon)
    state = eqx.tree_at(lambda s: (s.vegetation, s.carbon, s.summary),
                        state, (nv, nc, ns))
    return state, key, diag


def build_context(df):
    i = df.i.values.astype(np.int32)
    j = df.j.values.astype(np.int32)
    lat_rad = np.deg2rad(LAT[i]).astype(np.float32)
    pdict = load_physics_params(PARAMS)
    sp, pft, _ = build_all_params(len(df), lat_rad, pdict)
    return dict(i=i, j=j, lat_rad=lat_rad, sp=sp, pft=pft,
                consts=PhysicalConstants(), reader=GSWP3Reader(GSWP3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=1901)
    ap.add_argument("--to-year", type=int, default=1996)
    ap.add_argument("--eval-year0", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=0)
    ap.add_argument("--isimveg", type=int, default=1)
    ap.add_argument("--out", default=f"{REPO}/verify_out_stage1")
    ap.add_argument("--restart-in", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_sites(1990, 2010)
    m = np.array(xr.open_dataset(MASK).landfrac.values)
    fl = np.where(m.ravel() > 0)[0]
    pos = {g: p for p, g in enumerate(fl)}
    df["pos"] = [pos.get(i * 720 + j, -1) for i, j in zip(df.i, df.j)]
    df = df[df.pos >= 0].reset_index(drop=True)
    log.info("%d site cells", len(df))
    ctx = build_context(df)
    key = jax.random.PRNGKey(0)

    tmpl = create_ibis_state(GridConfig(npoi=len(df), nbedrock=3))
    if a.restart_in:
        log.info("Resuming transient from %s", a.restart_in)
        state, _ = load_state(a.restart_in, tmpl)
    else:
        log.info("Subsetting both_fix (npoi=%d -> %d)", NPOI_FULL, len(df))
        full = load_state(RESTART, create_ibis_state(
            GridConfig(npoi=NPOI_FULL, nbedrock=3)))[0]
        sub = df.pos.values

        def cut(x):
            if isinstance(x, jnp.ndarray) and x.ndim >= 1 \
                    and x.shape[0] == NPOI_FULL:
                return x[sub]
            return x
        state = jax.tree.map(cut, full)
        state = _cast_tree_f32(state)
        state = jax.tree.map(
            lambda x: jnp.nan_to_num(x, nan=0.0) if isinstance(x, jnp.ndarray)
            and jnp.issubdtype(x.dtype, jnp.floating) else x, state)
        del full

    log.info("TRANSIENT %d-%d  (CO2 %.0f -> %.0f ppm, isimveg=%d)",
             a.from_year, a.to_year, co2_ppm(a.from_year), co2_ppm(a.to_year),
             a.isimveg)
    for yr in range(a.from_year, a.to_year + 1):
        state, key, d = one_year(state, key, yr, ctx, a.isimveg)
        if yr % 10 == 0 or yr == a.to_year:
            log.info("  %d  CO2=%.0f  GPP=%.3f  LAI=%.2f  soilC=%.2f",
                     yr, co2_ppm(yr), float(np.nanmean(d["aygpptot"])),
                     float(np.nanmean(d["lai"])),
                     float(np.nanmean(np.asarray(state.carbon.totcsoi))))
    rp = f"{a.out}/site_transient_{a.to_year}.npz"
    save_state(state, rp, {"year": a.to_year})
    log.info("Saved %s", rp)

    if a.n_eval > 0:
        rows = []
        for n in range(a.n_eval):
            yr = a.eval_year0 + n
            state, key, d = one_year(state, key, yr, ctx, a.isimveg)
            rows.append(pd.DataFrame(dict(
                year=yr, site=df.site, pft=df.pft, igbp=df.igbp,
                lat=df.lat, lon=df.lon,
                gpp_mod=d["aygpptot"], et_mod=d["ayaet"],
                rh_mod=d["ayco2mic"], ra_mod=d["ayco2root"],
                lai_mod=d["lai"], prcp_mod=d["ayprcp"],
                gpp_obs=df.gpp_obs, et_obs=df.et_obs, reco_obs=df.reco_obs)))
            log.info("  EVAL %d  GPP=%.3f  ET=%.0f", yr,
                     float(np.nanmean(d["aygpptot"])),
                     float(np.nanmean(d["ayaet"])))
        out = pd.concat(rows, ignore_index=True)
        f = f"{a.out}/site_transient_eval_{a.eval_year0}_{a.n_eval}.csv"
        out.to_csv(f, index=False)
        log.info("Wrote %s", f)


if __name__ == "__main__":
    main()
