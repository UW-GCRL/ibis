"""Is GSWP3 forcing biased at FLUXNET sites? The cheap test of the 1.6x GPP excess.

docs/GPP.md ends with two empirical corrections and an unexplained factor of
~1.6 in GPP. Every component of the model was verified faithful to F77, leaving
FORCING and DOMAIN as the only untested candidates. Domain is already
controlled for -- the FLUXCOM comparison uses matched cells.

This tests forcing directly, and cheaply: rather than build a tower-driven site
model (direct/diffuse partitioning, longwave, humidity conversion -- hours of
work), it compares the GSWP3 forcing itself against tower measurements at the
same locations. Shortwave and VPD are what drive GPP, so a bias in either
propagates straight into assimilation.

    python verify/forcing_vs_fluxnet.py
"""
import argparse, glob, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python-port"))

from ibis.io.gswp3_reader import GSWP3Reader  # noqa: E402

FLUX = "/shared/nas2/Min/FLUXNET2015/fluxnet_compact"
NDAYPM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def site_table():
    m = pd.read_csv(f"{FLUX}/FLUXNET_siteinfo.csv", encoding="utf-8-sig")
    return m.set_index("SITE_ID")


def tower_means(min_years=3):
    """Per-site multi-year mean of the forcing variables, QC-filtered."""
    meta = site_table()
    out = {}
    for f in sorted(glob.glob(f"{FLUX}/FLX_*_DD_*.csv")):
        sid = os.path.basename(f).split("_")[1]
        if sid not in meta.index:
            continue
        yr = re.search(r"_(\d{4})-(\d{4})_", os.path.basename(f))
        if not yr or int(yr.group(2)) - int(yr.group(1)) + 1 < min_years:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        rec = {}
        for var, qc in (("TA_F_MDS", "TA_F_MDS_QC"),
                        ("SW_IN_F_MDS", "SW_IN_F_MDS_QC"),
                        ("VPD_F_MDS", "VPD_F_MDS_QC"),
                        ("P_F", "P_F_QC")):
            if var not in df:
                continue
            v = pd.to_numeric(df[var], errors="coerce").replace(-9999, np.nan)
            if qc in df:                     # keep measured/good-gapfill only
                q = pd.to_numeric(df[qc], errors="coerce")
                v = v.where(q >= 0.8)
            if v.notna().sum() > 365 * min_years * 0.5:
                rec[var] = float(v.mean())
        if {"TA_F_MDS", "SW_IN_F_MDS"} <= set(rec):
            rec["lat"] = float(meta.loc[sid, "LOCATION_LAT"])
            rec["lon"] = float(meta.loc[sid, "LOCATION_LONG"])
            rec["igbp"] = meta.loc[sid, "IGBP"]
            rec["y0"], rec["y1"] = int(yr.group(1)), int(yr.group(2))
            out[sid] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gswp3", default="/shared/nas2/Min/GSWP3")
    ap.add_argument("--years", type=int, nargs=2, default=[1991, 2000])
    a = ap.parse_args()

    sites = tower_means()
    ids = sorted(sites)
    print(f"  {len(ids)} FLUXNET sites with >=3 yr of QC-passed TA and SW_IN")

    lat = np.array([sites[s]["lat"] for s in ids])
    lon = np.array([sites[s]["lon"] for s in ids])
    # GSWP3 grid: lat 89.75 -> -89.75, lon -179.75 -> 179.75, 0.5 deg
    li = np.rint((89.75 - lat) / 0.5).astype(int).clip(0, 359)
    lo = np.rint((lon + 179.75) / 0.5).astype(int).clip(0, 719)

    reader = GSWP3Reader(a.gswp3)
    acc = {v: np.zeros(len(ids)) for v in ("tas", "rsds", "vpd")}
    nd = 0

    def daily_vpd(tas, huss, ps):
        """VPD from DAILY values, then averaged -- never the reverse.

        `esat` is convex in temperature, so computing it from a monthly-mean
        temperature underestimates the mean of the daily values. A first pass
        here averaged first and reported GSWP3 VPD 33% below tower, which was
        mostly that error rather than a forcing bias.
        """
        T = tas - 273.15
        esat = 6.112 * np.exp(17.67 * T / (T + 243.5))              # hPa
        e = huss * ps / (0.622 + 0.378 * huss) / 100.0              # hPa
        return np.maximum(esat - e, 0.0)

    for y in range(a.years[0], a.years[1] + 1):
        for m in range(12):
            d = reader.read_month(y, m, li, lo)
            w = NDAYPM[m]
            acc["tas"] += np.nanmean(d["tas"], axis=0) * w
            acc["rsds"] += np.nanmean(d["rsds"], axis=0) * w
            if {"huss", "ps"} <= set(d):
                acc["vpd"] += np.nanmean(
                    daily_vpd(d["tas"], d["huss"], d["ps"]), axis=0) * w
            nd += w
        print(f"    {y}", flush=True)
    for v in acc:
        acc[v] /= nd
    vpd_g = acc["vpd"]

    tow_ta = np.array([sites[s]["TA_F_MDS"] for s in ids])
    tow_sw = np.array([sites[s]["SW_IN_F_MDS"] for s in ids])
    tow_vpd = np.array([sites[s].get("VPD_F_MDS", np.nan) for s in ids])

    print(f"\n  GSWP3 {a.years[0]}-{a.years[1]} vs tower multi-year means\n")
    print(f"  {'variable':16} {'GSWP3':>8} {'tower':>8} {'bias':>8} "
          f"{'bias %':>8} {'r':>6} {'n':>4}")
    for nm, g, t, unit in (("temperature C", acc["tas"] - 273.15, tow_ta, ""),
                           ("shortwave W/m2", acc["rsds"], tow_sw, ""),
                           ("VPD hPa", vpd_g, tow_vpd, "")):
        k = np.isfinite(g) & np.isfinite(t)
        if k.sum() < 5:
            continue
        b = float(np.mean(g[k] - t[k]))
        pct = b / float(np.mean(t[k])) * 100 if nm != "temperature C" else np.nan
        r = float(np.corrcoef(g[k], t[k])[0, 1])
        print(f"  {nm:16} {np.mean(g[k]):8.2f} {np.mean(t[k]):8.2f} {b:+8.2f} "
              f"{pct:+8.1f} {r:6.3f} {int(k.sum()):4d}")
    print("""
  Shortwave and VPD drive GPP. A positive shortwave bias of the right size
  would explain the GPP excess without any model defect; an unbiased forcing
  puts the excess back inside the model.""")


if __name__ == "__main__":
    main()
