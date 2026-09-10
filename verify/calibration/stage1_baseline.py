#!/usr/bin/env python3
"""Stage-1 carbon calibration baseline: is `vmax x0.54` right at the towers?

Two arms, same parameters, same forcing, same 205 FLUXNET cells:

  A. the model's own LAI (2.25x MODIS at these cells)
  B. LAI prescribed to MODIS annual-maximum canopy LAI

Arm A is what the model does today. Arm B removes the LAI bias so the GPP
comparison speaks about photosynthesis rather than leaf area. The gap between
them is how much of the current GPP agreement was riding on inflated LAI --
which is the question that decides whether `vmax x0.54` survives.

Nothing is fitted here. This is the baseline the optimiser will be judged
against.

Usage:
  ibis-python verify/calibration/stage1_baseline.py --year0 1997 --nyear 8
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port")

import jax
import jax.numpy as jnp

from ibis.constants import PhysicalConstants
from ibis.dimensions import GridConfig
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.io.restart import load_state
from ibis.physics.prescribed_lai import apply_prescribed_lai
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.calendar import is_leap_year
from ibis.timestepping.loops_gswp3 import run_month_gswp3, _cast_tree_f32

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

FLUXDIR = "/shared/nas2/Min/FLUXNET2015/fluxnet_compact"
GSWP3 = "/shared/nas2/Min/GSWP3"
MASK = "/shared/nas2/Min/HWSD/surta.nc"
BOTHFIX = "/shared/nas2/Hangkai/ibis/verify_outputs/both_fix"
RESTART = f"{BOTHFIX}/restart/ibis_restart_2200.npz"
PARAMS = f"{BOTHFIX}/params.yaml"
MODIS_BASE = "/shared/nas2/Hangkai/ELM Edge Effects/inputdata"

LAT = np.arange(89.75, -90, -0.5)
LON = np.arange(-179.75, 180, 0.5)
NPOI_FULL = 95333

# IGBP -> IBIS PFT index. Latitude splits tropical from temperate.
PFT_NAMES = ("trop_bl_ever", "trop_bl_dec", "warm_bl_ever", "temp_con_ever",
             "temp_bl_dec", "boreal_con_ever", "boreal_bl_dec",
             "boreal_con_dec", "ever_shrub", "dec_shrub", "c4_grass",
             "c3_grass")
IGBP_TO_PFT = {
    "EBF": lambda la: 0 if abs(la) < 23.5 else 2,
    "DBF": lambda la: 1 if abs(la) < 23.5 else 4,
    "ENF": lambda la: 3 if abs(la) < 50 else 5,
    "DNF": lambda la: 7,
    "MF":  lambda la: 4 if abs(la) < 50 else 5,
    "CSH": lambda la: 8,
    "OSH": lambda la: 9,
    "WSA": lambda la: 10 if abs(la) < 30 else 11,
    "SAV": lambda la: 10 if abs(la) < 30 else 11,
    "GRA": lambda la: 10 if abs(la) < 30 else 11,
    "CRO": lambda la: 11,
    "WET": lambda la: 11,
}


def modis_canopy_lai(i, j):
    """MODIS annual-maximum canopy LAI, undiluted by PCT_NATVEG.

    A tower footprint sees the canopy, not the 0.5-degree mixture, so the
    natural-vegetation fraction is deliberately NOT applied. At these 205
    cells PCT_NATVEG averages 0.92, so this is an 8% choice, stated rather
    than hidden.
    """
    ds = xr.open_dataset(f"{MODIS_BASE}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc")
    sd = xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc")
    pctpft = np.asarray(sd["PCT_NAT_PFT"].values, dtype=np.float64) / 100.0
    nt = ds.sizes["time"]
    mon = np.zeros((12,) + pctpft.shape[1:])
    for m in range(12):
        sel = [t for t in range(nt) if t % 12 == m]
        tot = np.zeros(pctpft.shape[1:])
        for k in range(1, 17):
            v = f"LAI_{k}"
            if v in ds:
                tot += np.nan_to_num(ds[v].values[sel].mean(axis=0)) * pctpft[k]
        mon[m] = tot
    grid = mon.max(axis=0)
    if float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]):
        grid = grid[::-1, :]
    return np.nan_to_num(grid[i, j]).astype(np.float32)


def load_sites(year0, year1):
    """205 FLUXNET cells with tower GPP/ET/RECO averaged over `year0..year1`."""
    meta = pd.read_csv(f"{FLUXDIR}/FLUXNET_siteinfo.csv",
                       encoding="utf-8-sig").set_index("SITE_ID")
    rows = []
    for f in sorted(glob.glob(f"{FLUXDIR}/FLX_*_YY_*.csv")):
        sid = os.path.basename(f).split("_")[1]
        if sid not in meta.index:
            continue
        r = meta.loc[sid]
        if r["IGBP"] not in IGBP_TO_PFT:
            continue
        la, lo = float(r["LOCATION_LAT"]), float(r["LOCATION_LONG"])
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        d = pd.read_csv(f)
        if "TIMESTAMP" in d:
            yr = pd.to_datetime(d["TIMESTAMP"].astype(str),
                                errors="coerce").dt.year
            yr = yr.fillna(pd.to_numeric(d["TIMESTAMP"], errors="coerce"))
            d = d[(yr >= year0) & (yr <= year1)]
        if len(d) == 0:
            continue

        def m(col):
            if col not in d:
                return np.nan
            v = pd.to_numeric(d[col], errors="coerce").values
            v = v[np.isfinite(v) & (v > -9000)]
            return float(v.mean()) if v.size else np.nan

        gpp, reco, le = m("GPP_NT_VUT_REF"), m("RECO_NT_VUT_REF"), m("LE_F_MDS")
        gpp_dt = m("GPP_DT_VUT_REF")
        le_corr = m("LE_CORR")
        if not np.isfinite(gpp) or gpp <= 0:
            continue
        rows.append(dict(
            site=sid, lat=la, lon=lo, igbp=r["IGBP"],
            i=int(np.argmin(np.abs(LAT - la))),
            j=int(np.argmin(np.abs(LON - lo))),
            pft=IGBP_TO_PFT[r["IGBP"]](la),
            n_yr=len(d),
            gpp_obs=gpp / 1000.0,                      # gC/m2/yr -> kgC/m2/yr
            gpp_dt_obs=gpp_dt / 1000.0 if np.isfinite(gpp_dt) else np.nan,
            et_corr_obs=(le_corr * 86400 * 365 / 2.5e6
                         if np.isfinite(le_corr) else np.nan),
            reco_obs=reco / 1000.0 if np.isfinite(reco) else np.nan,
            et_obs=le * 86400 * 365 / 2.5e6 if np.isfinite(le) else np.nan,
        ))
    return pd.DataFrame(rows)


def run_years(state, consts, sp, reader, i, j, lat_rad, y0, ny, key):
    """Run `ny` years, returning per-year GPP/ET/Rh/Ra arrays (ny, npoi)."""
    from ibis.timestepping.loops_gswp3 import _cast_tree_f32  # noqa
    import equinox as eqx
    npoi = len(i)
    out = {k: [] for k in ("gpp", "et", "rh", "ra")}
    z = jnp.zeros((npoi,))
    z12 = jnp.zeros((npoi, 12))
    for n in range(ny):
        state = eqx.tree_at(
            lambda s: (s.summary.aygpptot, s.summary.ayaet, s.summary.ayprcp,
                       s.summary.ayco2mic, s.summary.ayco2root,
                       s.summary.ayanpptot, s.summary.aynpptot,
                       s.summary.ayneetot, s.summary.aytrunof,
                       s.summary.aytdrain, s.summary.aynpp),
            state, (z, z, z, z, z, z, z, z, z, z, z12))
        yr = y0 + n
        leap = is_leap_year(yr)
        for im in range(12):
            state, key, _ = run_month_gswp3(
                state, consts, sp, reader, i, j, lat_rad, yr, im, key,
                3600.0, leap=leap, nband=2, nsoilcall=1)
        out["gpp"].append(np.asarray(state.summary.aygpptot))
        out["et"].append(np.asarray(state.summary.ayaet))
        out["rh"].append(np.asarray(state.summary.ayco2mic))
        out["ra"].append(np.asarray(state.summary.ayco2root))
        log.info("    year %d done  GPP=%.3f  ET=%.0f", yr,
                 float(np.nanmean(out["gpp"][-1])),
                 float(np.nanmean(out["et"][-1])))
    return {k: np.array(v) for k, v in out.items()}, state, key


def report(tag, df, res, lai_used):
    gpp = np.nanmean(res["gpp"], axis=0)
    et = np.nanmean(res["et"], axis=0)
    reco = np.nanmean(res["rh"] + res["ra"], axis=0)
    d = df.copy()
    d["gpp_mod"], d["et_mod"], d["reco_mod"] = gpp, et, reco
    d["lai_used"] = lai_used
    d["r_gpp"] = d.gpp_mod / d.gpp_obs
    v = d[np.isfinite(d.r_gpp) & (d.gpp_obs > 0.05)]
    print(f"\n{'='*78}\nARM {tag}   n={len(v)}   LAI mean {np.nanmean(lai_used):.2f}")
    print(f"  GPP  model {v.gpp_mod.mean():.3f}  obs {v.gpp_obs.mean():.3f} "
          f"kgC/m2/yr   ratio mean {v.r_gpp.mean():.2f}x  median {v.r_gpp.median():.2f}x")
    dt = v[np.isfinite(v.gpp_dt_obs) & (v.gpp_dt_obs > 0.05)]
    print(f"  GPP  vs DAYTIME partitioning: obs {dt.gpp_dt_obs.mean():.3f}"
          f"   ratio median {(dt.gpp_mod/dt.gpp_dt_obs).median():.2f}x  (n={len(dt)})")
    e = v[np.isfinite(v.et_obs)]
    print(f"  ET   model {e.et_mod.mean():.0f}  obs {e.et_obs.mean():.0f} mm/yr"
          f"   ratio {(e.et_mod/e.et_obs).median():.2f}x   (n={len(e)})")
    r = v[np.isfinite(v.reco_obs)]
    print(f"  Reco model {r.reco_mod.mean():.3f}  obs {r.reco_obs.mean():.3f}"
          f"   ratio {(r.reco_mod/r.reco_obs).median():.2f}x   (n={len(r)}) "
          f"[reported, not fitted]")
    w = v[np.isfinite(v.et_obs) & (v.et_obs > 10)]
    wm = (w.gpp_mod / w.et_mod); wo = (w.gpp_obs / w.et_obs)
    print(f"  WUE  model {wm.median()*1000:.2f}  obs {wo.median()*1000:.2f} "
          f"gC/kgH2O   ratio {(wm/wo).median():.2f}x")
    ec = v[np.isfinite(v.et_corr_obs) & (v.et_corr_obs > 10)]
    if len(ec):
        print(f"  ET   vs energy-balance-corrected LE_CORR: obs "
              f"{ec.et_corr_obs.mean():.0f} mm/yr  ratio "
              f"{(ec.et_mod/ec.et_corr_obs).median():.2f}x  (n={len(ec)})")
    g = v.groupby("pft").agg(n=("site", "size"), lai=("lai_used", "mean"),
                             mod=("gpp_mod", "mean"), obs=("gpp_obs", "mean"),
                             ratio=("r_gpp", "median"))
    g.index = [PFT_NAMES[k] for k in g.index]
    print(g.round(3).to_string())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year0", type=int, default=1997)
    ap.add_argument("--nyear", type=int, default=8)
    ap.add_argument("--settle", type=int, default=1)
    ap.add_argument("--out", default=f"{REPO}/verify_out_stage1")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    y1 = a.year0 + a.nyear - 1
    log.info("Sites: tower means over %d-%d", a.year0, y1)
    df = load_sites(a.year0, y1)
    log.info("  %d sites with GPP in window", len(df))

    # map to land points
    m = np.array(xr.open_dataset(MASK).landfrac.values)
    fl = np.where(m.ravel() > 0)[0]
    pos = {g: p for p, g in enumerate(fl)}
    df["pos"] = [pos.get(i * 720 + j, -1) for i, j in zip(df.i, df.j)]
    df = df[df.pos >= 0].reset_index(drop=True)
    log.info("  %d sites on land points", len(df))
    npoi = len(df)
    i = df.i.values.astype(np.int32)
    j = df.j.values.astype(np.int32)
    lat_rad = np.deg2rad(LAT[i]).astype(np.float32)

    log.info("Loading both_fix restart (nbedrock=3) ...")
    full = load_state(RESTART, create_ibis_state(
        GridConfig(npoi=NPOI_FULL, nbedrock=3)))[0]
    sub = df.pos.values

    def cut(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL:
            return x[sub]
        return x
    st0 = jax.tree.map(cut, full)
    st0 = _cast_tree_f32(st0)
    st0 = jax.tree.map(
        lambda x: jnp.nan_to_num(x, nan=0.0)
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating)
        else x, st0)
    del full

    pdict = load_physics_params(PARAMS)
    sp, pft_params, _bgc = build_all_params(npoi, lat_rad, pdict)
    consts = PhysicalConstants()
    reader = GSWP3Reader(GSWP3)
    key = jax.random.PRNGKey(0)

    lai_model = np.asarray(st0.vegetation.plai).sum(axis=1)
    lai_modis = modis_canopy_lai(i, j)
    log.info("LAI at towers: model %.2f  MODIS canopy %.2f  ratio %.2fx",
             lai_model.mean(), lai_modis.mean(),
             np.median(lai_model / np.maximum(lai_modis, 0.01)))

    out = {}
    for tag, state in (("A_model_lai", st0),
                       ("B_modis_lai", apply_prescribed_lai(
                           st0, jnp.array(lai_modis),
                           pft_params.specla))):
        log.info("=== ARM %s ===", tag)
        s = state
        if a.settle:
            _, s, key = run_years(s, consts, sp, reader, i, j, lat_rad,
                                  a.year0 - a.settle, a.settle, key)
        res, _, key = run_years(s, consts, sp, reader, i, j, lat_rad,
                                a.year0, a.nyear, key)
        lai_used = np.asarray(state.vegetation.plai).sum(axis=1)
        out[tag] = report(tag, df, res, lai_used)
        out[tag].to_csv(f"{a.out}/stage1_{tag}.csv", index=False)

    A, B = out["A_model_lai"], out["B_modis_lai"]
    print(f"\n{'='*78}\nWHAT THE LAI CORRECTION COSTS")
    both = np.isfinite(A.r_gpp) & np.isfinite(B.r_gpp)
    print(f"  GPP/obs   arm A {A.r_gpp[both].median():.2f}x  ->  "
          f"arm B {B.r_gpp[both].median():.2f}x")
    print(f"  Written to {a.out}/")


if __name__ == "__main__":
    main()
