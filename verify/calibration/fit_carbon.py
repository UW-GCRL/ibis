#!/usr/bin/env python3
"""Stage 1: joint per-PFT vmax + specla fit against FLUXCOM and MODIS.

Design follows what the diagnosis established (docs/CALIBRATION.md):

* **Magnitude comes from gridded products, never tower point values.** The
  model is 1.09x FLUXCOM at tower cells but 0.81x the tower value, and
  FLUXCOM -- trained on those towers -- is itself 0.75x them at their own
  cells. Fitting the tower gap would inflate global GPP ~1.4x while every
  site plot improved.

* **vmax and specla must move together.** GPP responds to vmax; LAI is set by
  `NPP x aleaf x tauleaf x specla` and cannot be corrected by initialisation
  (seeding decays back within 30 years). Fitting GPP alone drove LAI from
  5.01 to 8.17 in the vmax experiment, so LAI is a constraint here, not an
  afterthought.

* **Stratified global cells, not the tower network**, which is 24% temperate
  conifer against a land surface that is nothing like that.

* **Held-out cells.** A fit that only improves the cells it was fitted on is
  not a calibration.

* **Cells poleward of the exclusion latitude are dropped and declared.**
  Seeded Arctic vegetation collapses back to the `xminlai` floor and restored
  vmax does not reach it either, so leaving those cells in lets the optimiser
  compensate for a structural hole -- which is how `vmax x0.54` arose.

Usage:
  ibis-python verify/calibration/fit_carbon.py --restart <v11 transient> \
      --n-evals 200 --cells 4000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import xarray as xr

REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port")
sys.path.insert(0, f"{REPO}/verify/calibration")

import equinox as eqx
import jax
import jax.numpy as jnp

from ibis.constants import PhysicalConstants
from ibis.dimensions import GridConfig
from ibis.io.co2 import co2_molmol
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.io.physics_params import build_all_params
from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.loops_gswp3 import _cast_tree_f32

from site_transient import one_year

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MASK = "/shared/nas2/Min/HWSD/surta.nc"
GSWP3 = "/shared/nas2/Min/GSWP3"
FLUXCOM = "/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc"
MODIS_BASE = "/shared/nas2/Hangkai/ELM Edge Effects/inputdata"
NPOI_FULL = 95333
LATV = np.arange(89.75, -90, -0.5)
LONV = np.arange(-179.75, 180, 0.5)

PFT_NAMES = ("trop_bl_ever", "trop_bl_dec", "warm_bl_ever", "temp_con_ever",
             "temp_bl_dec", "boreal_con_ever", "boreal_bl_dec",
             "boreal_con_dec", "ever_shrub", "dec_shrub", "c4_grass",
             "c3_grass")

# Literature bounds, umol m-2 s-1 (scripts/calibrate_fluxnet.py)
VMAX_LO = np.array([40, 40, 25, 15, 20, 15, 20, 15, 15, 15, 8, 15]) * 1e-6
VMAX_HI = np.array([100, 100, 70, 50, 60, 40, 50, 45, 45, 45, 30, 45]) * 1e-6
SPECLA_LO = np.array([10, 10, 10, 5, 12, 5, 12, 8, 5, 10, 10, 10], float)
SPECLA_HI = np.array([30, 30, 30, 15, 35, 15, 35, 25, 18, 30, 25, 25], float)


# --------------------------------------------------------------- observations
def fluxcom_at(i, j):
    o = xr.open_dataset(FLUXCOM)
    g = np.asarray(o["gpp"].mean(dim="time")) * 365.25 / 1000.0   # -> kgC/m2/yr
    la, lo = np.asarray(o["lat"]), np.asarray(o["lon"])
    ii = np.array([np.argmin(np.abs(la - LATV[a])) for a in i])
    jj = np.array([np.argmin(np.abs(lo - LONV[b])) for b in j])
    return g[ii, jj]


def modis_at(i, j):
    ds = xr.open_dataset(f"{MODIS_BASE}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc")
    sd = xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc")
    pv = np.asarray(sd["PCT_NATVEG"].values) / 100.0
    pp = np.asarray(sd["PCT_NAT_PFT"].values) / 100.0
    nt = ds.sizes["time"]
    mon = np.zeros((12,) + pv.shape)
    for m in range(12):
        sel = [t for t in range(nt) if t % 12 == m]
        tot = np.zeros(pv.shape)
        for k in range(1, 17):
            v = f"LAI_{k}"
            if v in ds:
                tot += np.nan_to_num(ds[v].values[sel].mean(axis=0)) * pp[k]
        mon[m] = tot * pv
    g = mon.max(axis=0)
    if float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]):
        g = g[::-1, :]
    return np.nan_to_num(g[i, j])


# ------------------------------------------------------------------ parameters
def to_vec(vmax, specla):
    return np.concatenate([np.log(vmax), np.log(specla)])


def from_vec(x):
    n = len(x) // 2
    return np.exp(x[:n]), np.exp(x[n:])


def apply_params(sp, pft, vmax, specla):
    sp = eqx.tree_at(lambda p: p.stomata_params.vmax_pft, sp,
                     jnp.array(vmax, dtype=jnp.float32))
    pft = eqx.tree_at(lambda p: p.specla, pft,
                      jnp.array(specla, dtype=jnp.float32))
    return sp, pft


def bounds_penalty(vmax, specla):
    return float(
        np.sum(np.maximum(0, np.log(VMAX_LO / vmax)) ** 2) +
        np.sum(np.maximum(0, np.log(vmax / VMAX_HI)) ** 2) +
        np.sum(np.maximum(0, np.log(SPECLA_LO / specla)) ** 2) +
        np.sum(np.maximum(0, np.log(specla / SPECLA_HI)) ** 2))


# ------------------------------------------------------------------------ loss
def score(gpp, lai, gpp_obs, lai_obs, w_lai):
    """Symmetric log-ratio error; returns (total, gpp_term, lai_term)."""
    ok = np.isfinite(gpp) & np.isfinite(lai) & (gpp_obs > 0.02) & (lai_obs > 0.05)
    if ok.sum() < 20:
        return 50.0, np.nan, np.nan
    g = np.log(np.maximum(gpp[ok], 1e-4) / gpp_obs[ok])
    l = np.log(np.maximum(lai[ok], 1e-3) / lai_obs[ok])
    return float(np.mean(g ** 2) + w_lai * np.mean(l ** 2)), \
        float(np.mean(g ** 2)), float(np.mean(l ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", required=True)
    ap.add_argument("--cells", type=int, default=4000)
    ap.add_argument("--n-evals", type=int, default=200)
    ap.add_argument("--adjust-years", type=int, default=10)
    ap.add_argument("--eval-years", type=int, default=3)
    ap.add_argument("--year0", type=int, default=2001)
    ap.add_argument("--co2-year", type=int, default=2005)
    ap.add_argument("--exclude-poleward", type=float, default=68.0)
    ap.add_argument("--w-lai", type=float, default=0.5)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--out", default=f"{REPO}/verify_out_stage1/fit_carbon")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import cma
    from sample_cells import build_sample

    sel, lat, dom = build_sample(a.cells, a.restart)
    keep = np.abs(lat) < a.exclude_poleward
    log.info("sample %d cells; %d kept after |lat| < %.0f exclusion (%d dropped)",
             len(sel), keep.sum(), a.exclude_poleward, (~keep).sum())
    sel, lat, dom = sel[keep], lat[keep], dom[keep]

    m = np.array(xr.open_dataset(MASK).landfrac.values)
    fl = np.where(m.ravel() > 0)[0]
    i = (fl[sel] // 720).astype(np.int32)
    j = (fl[sel] % 720).astype(np.int32)
    lat_rad = np.deg2rad(LATV[i]).astype(np.float32)

    gpp_obs = fluxcom_at(i, j)
    lai_obs = modis_at(i, j)
    log.info("obs at sample: FLUXCOM GPP %.3f  MODIS LAI %.2f",
             np.nanmean(gpp_obs), np.nanmean(lai_obs))

    rng = np.random.default_rng(a.seed)
    fit_mask = rng.random(len(sel)) > a.holdout
    log.info("fit on %d cells, hold out %d", fit_mask.sum(), (~fit_mask).sum())

    full, _ = load_state(a.restart, create_ibis_state(
        GridConfig(npoi=NPOI_FULL, nbedrock=3)))

    def cut(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL:
            return x[sel]
        return x
    state0 = _cast_tree_f32(jax.tree.map(cut, full))
    state0 = jax.tree.map(
        lambda x: jnp.nan_to_num(x, nan=0.0) if isinstance(x, jnp.ndarray)
        and jnp.issubdtype(x.dtype, jnp.floating) else x, state0)
    state0 = eqx.tree_at(lambda s: s.atm.co2conc, state0, co2_molmol(a.co2_year))
    del full

    sp0, pft0, _ = build_all_params(len(sel), lat_rad, None)
    ctx0 = dict(i=i, j=j, lat_rad=lat_rad, sp=sp0, pft=pft0,
                consts=PhysicalConstants(), reader=GSWP3Reader(GSWP3))
    v0 = np.asarray(sp0.stomata_params.vmax_pft, dtype=np.float64)
    s0 = np.asarray(pft0.specla, dtype=np.float64)
    log.info("start vmax (umol): %s", np.round(v0 * 1e6, 1).tolist())
    log.info("start specla:      %s", np.round(s0, 1).tolist())

    hist = []
    best = [np.inf, None]
    ev = [0]

    def run(vmax, specla):
        sp, pft = apply_params(sp0, pft0, vmax, specla)
        ctx = dict(ctx0, sp=sp, pft=pft)
        st = state0
        key = jax.random.PRNGKey(0)
        for n in range(a.adjust_years):
            st, key, _ = one_year(st, key, a.year0 + (n % 5), ctx, 1)
        gs, ls = [], []
        for n in range(a.eval_years):
            st, key, d = one_year(st, key, a.year0 + (n % 5), ctx, 1)
            gs.append(d["aygpptot"]); ls.append(d["lai"])
        return np.mean(gs, axis=0), np.mean(ls, axis=0)

    def objective(x):
        t0 = time.time()
        vmax, specla = from_vec(x)
        if np.any(vmax < VMAX_LO * 0.4) or np.any(vmax > VMAX_HI * 2.5) \
                or np.any(specla < SPECLA_LO * 0.4) \
                or np.any(specla > SPECLA_HI * 2.5):
            return 100.0
        try:
            gpp, lai = run(vmax, specla)
        except Exception as e:                       # noqa: BLE001
            log.warning("  eval %d failed: %s", ev[0], str(e)[:90])
            return 50.0
        tot, gt, lt = score(gpp[fit_mask], lai[fit_mask],
                            gpp_obs[fit_mask], lai_obs[fit_mask], a.w_lai)
        tot += 0.05 * bounds_penalty(vmax, specla)
        ho, _, _ = score(gpp[~fit_mask], lai[~fit_mask],
                         gpp_obs[~fit_mask], lai_obs[~fit_mask], a.w_lai)
        ev[0] += 1
        hist.append(dict(eval=ev[0], loss=tot, gpp_term=gt, lai_term=lt,
                         holdout=ho, gpp_mean=float(np.nanmean(gpp)),
                         lai_mean=float(np.nanmean(lai))))
        if tot < best[0]:
            best[0], best[1] = tot, (vmax.copy(), specla.copy())
            log.info("  eval %3d  loss=%.4f (BEST)  gpp=%.4f lai=%.4f "
                     "holdout=%.4f  GPP=%.3f LAI=%.2f  [%.0fs]",
                     ev[0], tot, gt, lt, ho, hist[-1]["gpp_mean"],
                     hist[-1]["lai_mean"], time.time() - t0)
        elif ev[0] % 10 == 0:
            log.info("  eval %3d  loss=%.4f  holdout=%.4f  [%.0fs]",
                     ev[0], tot, ho, time.time() - t0)
        return tot

    log.info("=== baseline (starting parameters) ===")
    b = objective(to_vec(v0, s0))
    log.info("baseline loss %.4f", b)

    es = cma.CMAEvolutionStrategy(to_vec(v0, s0), a.sigma, {
        "maxfevals": a.n_evals, "seed": a.seed, "verb_disp": 0,
        "bounds": [to_vec(VMAX_LO * 0.4, SPECLA_LO * 0.4),
                   to_vec(VMAX_HI * 2.5, SPECLA_HI * 2.5)]})
    while not es.stop() and ev[0] < a.n_evals:
        xs = es.ask()
        es.tell(xs, [objective(x) for x in xs])

    vmax, specla = best[1]
    import pandas as pd
    pd.DataFrame(hist).to_csv(f"{a.out}/history.csv", index=False)
    np.savez(f"{a.out}/best.npz", vmax=vmax, specla=specla, sel=sel)
    print(f"\n{'='*76}\nBEST PARAMETERS (loss {best[0]:.4f}, baseline {b:.4f})")
    print(f"{'PFT':<18}{'vmax0':>8}{'vmax':>8}{'lo':>6}{'hi':>6}"
          f"{'specla0':>9}{'specla':>8}")
    for k, n in enumerate(PFT_NAMES):
        print(f"{n:<18}{v0[k]*1e6:>8.1f}{vmax[k]*1e6:>8.1f}"
              f"{VMAX_LO[k]*1e6:>6.0f}{VMAX_HI[k]*1e6:>6.0f}"
              f"{s0[k]:>9.1f}{specla[k]:>8.1f}")
    gpp, lai = run(vmax, specla)
    for nm, msk in (("fit", fit_mask), ("HELD OUT", ~fit_mask)):
        t, g, l = score(gpp[msk], lai[msk], gpp_obs[msk], lai_obs[msk], a.w_lai)
        print(f"  {nm:>9}: loss {t:.4f}  GPP term {g:.4f}  LAI term {l:.4f}  "
              f"model GPP {np.nanmean(gpp[msk]):.3f} vs {np.nanmean(gpp_obs[msk]):.3f}"
              f"  LAI {np.nanmean(lai[msk]):.2f} vs {np.nanmean(lai_obs[msk]):.2f}")
    print(f"\nWrote {a.out}/")


if __name__ == "__main__":
    main()
