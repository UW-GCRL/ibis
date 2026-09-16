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
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.io.physics_params import build_all_params
from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.loops_gswp3 import _cast_tree_f32

from site_transient import one_year


class CachedForcing:
    """A GSWP3Reader that serves a fixed set of years from memory.

    The fit re-runs the SAME few forcing years for every parameter
    evaluation, and reading them back off disk each time dominated the cost:
    18 minutes per evaluation at 4,331 cells, against 53 s per model-year for
    all 95,333 cells in the v11 spin-up. The difference is not compute -- it
    is fancy-indexing a compressed netCDF 12 times a year, 150 times over.

    Exposes only `read_month`, which is all `run_month_gswp3` uses.
    """

    def __init__(self, reader, years, i, j):
        self._c = {}
        for y in years:
            for m in range(12):
                self._c[(y, m)] = reader.read_month(y, m, i, j)
        n = sum(v[next(iter(v))].nbytes for v in self._c.values()) / 1e9
        log.info("cached %d months of forcing (%.2f GB)", len(self._c), n)

    def read_month(self, year, imonth, latindex=None, lonindex=None):
        return self._c[(year, imonth)]

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
def score(gpp, lai, gpp_obs, lai_obs, w_lai, area=None, w_total=1.0):
    """Magnitude-honest loss. Returns (total, gpp_term, lai_term).

    The first fit used an unweighted per-cell mean of squared log-ratios and
    reported GPP 0.83x FLUXCOM at the sample while the full domain came out
    ~1.35x (docs/CALIBRATION.md 9). A sparse cell at 0.5x counted as much as
    a rainforest cell at 1.5x, so the optimiser satisfied the many low-flux
    cells and left the few that carry the global total high.

    Two changes. (1) Each cell's GPP log-ratio is weighted by its share of the
    observed flux, area x GPP_obs -- i.e. by what it contributes to the total
    being judged. LAI is weighted by area only, since LAI comparisons are
    area means. (2) An explicit term on the log-ratio of area-integrated
    totals, so the sum itself is a target and not just an emergent property.
    """
    ok = np.isfinite(gpp) & np.isfinite(lai) & (gpp_obs > 0.02) & (lai_obs > 0.05)
    if ok.sum() < 20:
        return 50.0, np.nan, np.nan
    a = np.ones(ok.sum()) if area is None else np.asarray(area)[ok]
    g_m, g_o = np.maximum(gpp[ok], 1e-4), gpp_obs[ok]
    l_m, l_o = np.maximum(lai[ok], 1e-3), lai_obs[ok]

    wg = a * g_o; wg = wg / wg.sum()
    wl = a / a.sum()
    g_cell = float(np.sum(wg * np.log(g_m / g_o) ** 2))
    l_cell = float(np.sum(wl * np.log(l_m / l_o) ** 2))
    g_tot = float(np.log(np.sum(a * g_m) / np.sum(a * g_o)) ** 2)
    l_tot = float(np.log(np.sum(a * l_m) / np.sum(a * l_o)) ** 2)

    g_term = g_cell + w_total * g_tot
    l_term = l_cell + w_total * l_tot
    return g_term + w_lai * l_term, g_term, l_term


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", required=True)
    ap.add_argument("--cells", type=int, default=4000)
    ap.add_argument("--n-evals", type=int, default=200)
    ap.add_argument("--adjust-years", type=int, default=10)
    ap.add_argument("--eval-years", type=int, default=3)
    ap.add_argument("--year0", type=int, default=2001)
    ap.add_argument("--force-years", type=int, default=3,
                    help="distinct forcing years cycled within an evaluation")
    ap.add_argument("--exclude-poleward", type=float, default=68.0)
    ap.add_argument("--w-lai", type=float, default=0.5)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--min-cells", type=int, default=30,
                    help="PFTs dominant in fewer cells are frozen at default")
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
    area = np.cos(np.deg2rad(LATV[i]))          # relative cell area
    log.info("obs at sample: FLUXCOM GPP %.3f  MODIS LAI %.2f",
             np.nanmean(gpp_obs), np.nanmean(lai_obs))

    counts = np.bincount(dom[dom >= 0], minlength=12)
    free = counts >= a.min_cells
    log.info("PFTs free to move (>= %d dominant cells): %s", a.min_cells,
             [PFT_NAMES[k] for k in range(12) if free[k]])
    log.info("PFTs FROZEN at defaults (too few cells to identify): %s",
             [f"{PFT_NAMES[k]}({counts[k]})" for k in range(12) if not free[k]])

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
    # CO2 is NOT set here: `one_year` calls apply_co2(state, year) itself, so
    # it follows the evaluation year (2001-2005 -> 371-380 ppm). Setting it
    # here as well would have been a flag that parses and does nothing, which
    # is the defect this project has now found fourteen times.
    del full

    sp0, pft0, _ = build_all_params(len(sel), lat_rad, None)
    years = list(range(a.year0, a.year0 + a.force_years))
    cached = CachedForcing(GSWP3Reader(GSWP3), years, i, j)
    ctx0 = dict(i=i, j=j, lat_rad=lat_rad, sp=sp0, pft=pft0,
                consts=PhysicalConstants(), reader=cached)
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
            st, key, _ = one_year(st, key, years[n % len(years)], ctx, 1)
        gs, ls = [], []
        for n in range(a.eval_years):
            st, key, d = one_year(st, key, years[n % len(years)], ctx, 1)
            gs.append(d["aygpptot"]); ls.append(d["lai"])
        return np.mean(gs, axis=0), np.mean(ls, axis=0)

    def objective(x):
        t0 = time.time()
        vmax, specla = from_vec(x)
        # A parameter the sample cannot constrain must not move: with zero
        # dominant cells the loss is flat in it, so CMA-ES walks it to a
        # bound on noise. The first run did exactly that -- boreal_con_dec
        # (0 cells) went to 57 umol against a 45 ceiling.
        vmax = np.where(free, vmax, v0)
        specla = np.where(free, specla, s0)
        # Tolerance matters: `specla` defaults sit EXACTLY on two bounds
        # (30.0 and 35.0), and the log/exp round-trip in to_vec/from_vec puts
        # them a float above, which rejected the baseline itself as
        # out-of-bounds and reported loss 100.
        tol = 1.0 + 1e-9
        if np.any(vmax < VMAX_LO / tol) or np.any(vmax > VMAX_HI * tol) \
                or np.any(specla < SPECLA_LO / tol) \
                or np.any(specla > SPECLA_HI * tol):
            return 100.0
        try:
            gpp, lai = run(vmax, specla)
        except Exception as e:                       # noqa: BLE001
            log.warning("  eval %d failed: %s", ev[0], str(e)[:90])
            return 50.0
        tot, gt, lt = score(gpp[fit_mask], lai[fit_mask],
                            gpp_obs[fit_mask], lai_obs[fit_mask], a.w_lai,
                            area[fit_mask])
        ho, _, _ = score(gpp[~fit_mask], lai[~fit_mask],
                         gpp_obs[~fit_mask], lai_obs[~fit_mask], a.w_lai,
                         area[~fit_mask])
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

    x0 = np.clip(to_vec(v0, s0), to_vec(VMAX_LO, SPECLA_LO),
                 to_vec(VMAX_HI, SPECLA_HI))
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {
        "maxfevals": a.n_evals, "seed": a.seed, "verb_disp": 0,
        "bounds": [to_vec(VMAX_LO, SPECLA_LO),
                   to_vec(VMAX_HI, SPECLA_HI)]})
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
        t, g, l = score(gpp[msk], lai[msk], gpp_obs[msk], lai_obs[msk], a.w_lai,
                        area[msk])
        print(f"  {nm:>9}: loss {t:.4f}  GPP term {g:.4f}  LAI term {l:.4f}  "
              f"area-wtd GPP {np.average(gpp[msk], weights=area[msk]):.3f} vs "
              f"{np.average(gpp_obs[msk], weights=area[msk]):.3f}"
              f"  LAI {np.average(lai[msk], weights=area[msk]):.2f} vs "
              f"{np.average(lai_obs[msk], weights=area[msk]):.2f}")
    print(f"\nWrote {a.out}/")


if __name__ == "__main__":
    main()
