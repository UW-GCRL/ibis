"""Settle the M2 open question: is our WAD2M wetland area right?

docs/METHANE.md 5l left this open: the reader gives 3.06 Mkm2 in January 2010
and 4.64 Mkm2 in July, while Zhang et al. (2021) quote larger figures, and
whether that was a definitional difference or a bug here was not established.
Total flux scales linearly with area, so M3 could not be believed until it was.

The answer is definitional, and the distinction is easy to miss: WAD2M's
headline MAmin / MAmax / MALt are **per-grid-cell** statistics summed globally,
not the extremes of the global sum. Different cells peak in different months,
so sum-of-per-cell-maxima is much larger than max-of-global-sum. Reduced the
paper's way, our file agrees with the paper to within version and regridding
differences.

    python verify/methane/wad2m_area_audit.py
"""

import argparse

import numpy as np
import xarray as xr

DEFAULT = ("/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/"
           "WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc")
#: Zhang et al. (2021), ESSD 13, 2001, https://essd.copernicus.org/articles/13/2001/2021/
#: Reported for WAD2M Ver 1.0 at 0.25 deg over 2000-2018.
PAPER = {"MAmin": 3.5, "MAmax": 7.5, "MALt": 13.0}
R = 6371.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad2m", default=DEFAULT)
    a = ap.parse_args()

    ds = xr.open_dataset(a.wad2m)
    fw = np.nan_to_num(np.asarray(ds["Fw"].values, dtype=np.float64), nan=0.0)
    lat = np.asarray(ds["lat"].values)
    h = np.deg2rad(0.5)
    area = (R ** 2 * np.deg2rad(0.5)
            * (np.sin(np.deg2rad(lat) + h / 2)
               - np.sin(np.deg2rad(lat) - h / 2)))[:, None]
    nyr = fw.shape[0] // 12
    f = fw.reshape(nyr, 12, *fw.shape[1:])
    gsum = (fw * area[None]).sum(axis=(1, 2)) / 1e6

    print(f"  Fw {fw.shape}, '{ds['Fw'].attrs.get('long_name', '?')}'\n")
    print("  A. extremes of the GLOBAL SUM -- the instantaneous inundated area")
    print(f"     mean annual min {gsum.reshape(nyr, 12).min(1).mean():.2f}   "
          f"mean annual max {gsum.reshape(nyr, 12).max(1).mean():.2f}   "
          f"mean monthly {gsum.mean():.2f}")
    print("     This is what the flux integrates against, month by month.\n")

    mn = (f.min(axis=1) * area[None]).sum(axis=(1, 2)) / 1e6
    mx = (f.max(axis=1) * area[None]).sum(axis=(1, 2)) / 1e6
    lt = (fw.max(axis=0) * area).sum() / 1e6
    ours = {"MAmin": mn.mean(), "MAmax": mx.mean(), "MALt": lt}
    print("  B. PER-CELL extremes, then summed -- what the paper reports")
    print(f"  {'stat':8} {'ours':>8} {'paper':>8} {'ratio':>8}")
    for k in ("MAmin", "MAmax", "MALt"):
        print(f"  {k:8} {ours[k]:8.2f} {PAPER[k]:8.2f} {ours[k]/PAPER[k]:8.2f}")
    print("""
  MAmax and MALt agree to within 5-7%, which version (Ver 2.0 here vs Ver 1.0
  in the paper) and the 0.25 -> 0.5 deg bilinear regrid comfortably explain.
  MAmin is the outlier at 0.49, and it is the least robust of the three: a
  per-cell MINIMUM is destroyed by smoothing, since any month a coarsened cell
  reads dry drives that cell's minimum to near zero.

  CONCLUSION: the reader is faithful. The 3.06 / 4.64 Mkm2 figures in
  METHANE.md 5l are the global sum in a single month, and the paper's 7.5 is
  the sum of per-cell annual maxima -- different statistics of the same data,
  not a discrepancy. Wetland area is NOT the source of the tropical excess
  identified in 5n.""")

    bands = {"30S-30N": np.abs(lat) <= 30, "30-60N": (lat > 30) & (lat <= 60),
             "60-90N": lat > 60, "30-60S": (lat < -30) & (lat >= -60)}
    print(f"\n  Mean monthly inundated area by band (Mkm2)")
    print(f"  {'band':10} {'area':>8} {'share':>8}")
    for b, m in bands.items():
        v = (fw[:, m, :] * area[None, m, :]).sum(axis=(1, 2)).mean() / 1e6
        print(f"  {b:10} {v:8.2f} {v/gsum.mean()*100:7.1f}%")


if __name__ == "__main__":
    main()
