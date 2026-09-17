#!/usr/bin/env python3
"""Model seasonal GPP cycle against FLUXCOM's 1980-2013 monthly climatology.

Third criterion of the carbon recalibration. Nothing seasonal had been scored
before this: every FLUXCOM comparison in this project used the annual mean.

Reads a run's monthly netCDF (imonthout=1), builds the model's 12-month
climatology over all years present, and scores it cell by cell with
`seasonal_shape` (scale-free fraction-of-year distribution) and `peak_offset`
(circular months). Reported by latitude band and by the dense/sparse split
that section 10 of docs/CALIBRATION.md identified -- because if the sparse
cells' excess is a phenology problem, it will show here as leaves-on when
FLUXCOM says the season is over.

Usage:
  ibis-python verify/calibration/seasonal_baseline.py \
      --monthly-dir <run>/out --gpp-var <name in the monthly files>
"""
from __future__ import annotations
import argparse, glob, re, sys
import numpy as np, netCDF4 as nc, xarray as xr
REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port"); sys.path.insert(0, f"{REPO}/verify/calibration")
from fit_carbon import LATV, fluxcom_at, fluxcom_monthly_at, seasonal_shape, peak_offset, MASK

BANDS = [(-30, 30, "30S-30N"), (30, 50, "30-50N"), (50, 60, "50-60N"), (60, 70, "60-70N"), (70, 90, "70-90N")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly-dir", required=True)
    ap.add_argument("--gpp-var", required=True)
    ap.add_argument("--label", default="model")
    a = ap.parse_args()

    fs = sorted(glob.glob(f"{a.monthly_dir}/*_monthly_*.nc"))
    assert fs, f"no monthly files in {a.monthly_dir}"
    with nc.Dataset(fs[0]) as d:
        assert a.gpp_var in d.variables, f"{a.gpp_var} not in {list(d.variables)}"
    m = np.array(xr.open_dataset(MASK).landfrac.values) > 0
    fl = np.where(m.ravel())[0]; i = fl // 720; j = fl % 720
    area = np.cos(np.deg2rad(LATV[i]))

    acc = np.zeros((12, len(fl))); cnt = np.zeros(12)
    for f in fs:
        mo = int(re.search(r"_monthly_\d+_(\d+)\.nc$", f).group(1)) - 1
        with nc.Dataset(f) as d:
            x = np.nan_to_num(np.asarray(d[a.gpp_var][:], float)).ravel()[fl]
        acc[mo] += x; cnt[mo] += 1
    assert (cnt > 0).all(), f"missing months: {np.where(cnt == 0)[0] + 1}"
    mod_m = acc / cnt[:, None]                       # (12, n) kgC m-2 month-1
    print(f"{len(fs)} monthly files, {int(cnt.min())}-{int(cnt.max())} years per month")

    obs_m = fluxcom_monthly_at(i, j)
    gobs = fluxcom_at(i, j)
    err, ok = seasonal_shape(mod_m, obs_m)
    off = peak_offset(mod_m, obs_m)
    ratio = mod_m.sum(axis=0) / np.maximum(obs_m.sum(axis=0), 1e-9)
    w = area * gobs                                  # flux weight, as in the fit loss
    lat = LATV[i]

    def row(nm, s):
        s = s & ok
        if s.sum() < 20:
            return
        ws = w[s] / w[s].sum()
        amp_m = (mod_m[:, s].max(axis=0) / np.maximum(mod_m[:, s].mean(axis=0), 1e-9))
        amp_o = (obs_m[:, s].max(axis=0) / np.maximum(obs_m[:, s].mean(axis=0), 1e-9))
        print(f"  {nm:<26} n={s.sum():6d}  shape err {np.sum(ws*err[s]):.4f}  "
              f"peak offset mean {np.sum(ws*off[s]):+.2f} mo  |off|>=2: {100*np.mean(np.abs(off[s])>=2):4.1f}%  "
              f"amp model/obs {np.sum(ws*amp_m)/np.sum(ws*amp_o):.2f}")

    print(f"\n{a.label}: seasonal shape vs FLUXCOM (flux-weighted)")
    row("ALL", np.ones(len(fl), bool))
    for lo, hi, nm in BANDS:
        row(nm, (lat >= lo) & (lat < hi))
    print("\nby annual-GPP ratio class (section-10 split):")
    row("ratio < 0.8", ratio < 0.8)
    row("0.8 <= ratio < 1.25", (ratio >= 0.8) & (ratio < 1.25))
    row("1.25 <= ratio < 2", (ratio >= 1.25) & (ratio < 2))
    row("ratio >= 2  (sparse excess)", ratio >= 2)
    print("\nwhere in the year does the excess sit?  model/obs monthly, cells with ratio >= 2 vs < 1.25:")
    for nm, s in (("ratio >= 2", ratio >= 2), ("ratio < 1.25", ratio < 1.25)):
        s = s & ok
        r = (mod_m[:, s] * area[s]).sum(axis=1) / np.maximum((obs_m[:, s] * area[s]).sum(axis=1), 1e-9)
        print(f"  {nm:<12} " + " ".join(f"{v:5.2f}" for v in r))
    print("               " + " ".join(f"{k:>5}" for k in ("J","F","M","A","M","J","J","A","S","O","N","D")))


if __name__ == "__main__":
    main()
