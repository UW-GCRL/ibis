#!/usr/bin/env python3
"""Extract MODIS MCD15A3H LAI time series at FLUXNET 2015 sites from GEE.

Products extracted
------------------
MODIS MCD15A3H  — 500 m, 4-day composite, Terra+Aqua combined, 2002–present
                  GEE collection: MODIS/061/MCD15A3H

Output
------
output_dir/
  lai_modis_4day.csv      — raw 4-day composites (all sites, long format)
  lai_modis_monthly.csv   — monthly means (quality-filtered)
  lai_modis_annual.csv    — annual means + seasonal amplitude

Note on GLASS LAI
-----------------
GLASS LAI (BNU, 500 m, 8-day, 1981–) is not in the standard GEE public catalog
as of early 2026. Download directly from:
  http://www.glass.umd.edu/LAI/MODIS/
and use scripts/process_glass_lai.py to aggregate to site × monthly CSVs.

Usage
-----
  # Authenticate once (opens browser):
  earthengine authenticate

  python scripts/gee_lai_fluxnet.py \
      --siteinfo /shared/nas2/Min/FLUXNET2015/fluxnet_compact/FLUXNET_siteinfo.csv \
      --output   /shared/nas2/Min/FLUXNET2015/lai_gee \
      --start    2002-01-01 \
      --end      2023-12-31

Requirements
------------
  pip install earthengine-api pandas numpy tqdm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── GEE import (deferred so the script is importable without credentials) ──
try:
    import ee
except ImportError:
    raise ImportError("Run: pip install earthengine-api")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODIS_COLLECTION = "MODIS/061/MCD15A3H"
LAI_BAND         = "Lai_500m"        # stored ×10; scale 0.1 → real LAI
QC_BAND          = "FparLai_QC"      # bit-packed quality flags
LAI_SCALE        = 500               # native pixel size in metres
LAI_FILL         = 255               # fill / invalid value (after ×10 = 25.5)

# QC bits 0–1: MODLAND QC
#   00 = Good quality (main algorithm, best result)
#   01 = Other quality (main algorithm, not best result; QA may not be produced)
#   10 = Mixed clouds
#   11 = Cloud/shadow/snow; not produced
# We keep 00 and 01 (bits 0-1 ≤ 1).
QC_GOOD_MASK = 0b11   # mask for bits 0-1
QC_MAX_GOOD  = 1      # keep if (QC & 0b11) <= 1


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------

def _mask_lai(image: ee.Image) -> ee.Image:
    """Apply QC mask and scale LAI to physical units (m² m⁻²)."""
    qc     = image.select(QC_BAND)
    good   = qc.bitwiseAnd(QC_GOOD_MASK).lte(QC_MAX_GOOD)
    lai    = (image.select(LAI_BAND)
                   .updateMask(good)
                   .multiply(0.1)
                   .rename("LAI"))
    return image.addBands(lai).select("LAI").copyProperties(image, ["system:time_start"])


def build_site_fc(sites: pd.DataFrame) -> ee.FeatureCollection:
    """Convert site DataFrame to GEE FeatureCollection of point features."""
    features = []
    for _, row in sites.iterrows():
        pt = ee.Geometry.Point([float(row["LOCATION_LONG"]),
                                float(row["LOCATION_LAT"])])
        feat = ee.Feature(pt, {
            "site_id": str(row["SITE_ID"]),
            "igbp":    str(row.get("IGBP", "")),
            "lat":     float(row["LOCATION_LAT"]),
            "lon":     float(row["LOCATION_LONG"]),
        })
        features.append(feat)
    return ee.FeatureCollection(features)


def getregion_to_df(raw: list, band: str = "LAI") -> pd.DataFrame:
    """Convert ee.ImageCollection.getRegion() list output to DataFrame."""
    if len(raw) < 2:
        return pd.DataFrame()
    header = raw[0]
    rows   = raw[1:]
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={"id": "site_id",
                             "longitude": "lon",
                             "latitude":  "lat",
                             "time":      "timestamp_ms"})
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date
    df[band]   = pd.to_numeric(df[band], errors="coerce")
    # Drop fill / invalid
    df = df[df[band].notna() & (df[band] <= 10.0)]  # physical LAI ≤ 10 m²/m²
    return df[["site_id", "lat", "lon", "date", band]].copy()


def fetch_year_chunk(
    fc:         ee.FeatureCollection,
    year_start: int,
    year_end:   int,
    max_retries: int = 5,
    retry_wait:  float = 30.0,
) -> pd.DataFrame:
    """Fetch LAI for all sites over [year_start, year_end] (inclusive).

    Uses getRegion() which is efficient for point extractions at many sites.
    Returns a DataFrame with columns [site_id, lat, lon, date, LAI].
    """
    col = (ee.ImageCollection(MODIS_COLLECTION)
             .filterDate(f"{year_start}-01-01", f"{year_end+1}-01-01")
             .map(_mask_lai))

    for attempt in range(1, max_retries + 1):
        try:
            raw = col.getRegion(fc, scale=LAI_SCALE).getInfo()
            return getregion_to_df(raw)
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = retry_wait * attempt
            print(f"    GEE error (attempt {attempt}/{max_retries}): {exc}")
            print(f"    Retrying in {wait:.0f} s ...")
            time.sleep(wait)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 4-day LAI composites to monthly means."""
    df = df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly = (df.groupby(["site_id", "year", "month"])["LAI"]
                 .agg(LAI_mean="mean", LAI_std="std", n_obs="count")
                 .reset_index())
    return monthly


def to_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute annual mean LAI and seasonal amplitude from monthly data."""
    def _stats(g):
        return pd.Series({
            "LAI_annual_mean": g["LAI_mean"].mean(),
            "LAI_max":         g["LAI_mean"].max(),
            "LAI_min":         g["LAI_mean"].min(),
            "LAI_amplitude":   g["LAI_mean"].max() - g["LAI_mean"].min(),
            "n_months":        g["LAI_mean"].notna().sum(),
        })
    return (monthly[monthly["n_obs"] >= 2]     # require ≥ 2 4-day obs per month
              .groupby(["site_id", "year"])
              .apply(_stats)
              .reset_index())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract MODIS LAI at FLUXNET sites via GEE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--siteinfo", required=True,
                        help="Path to FLUXNET_siteinfo.csv")
    parser.add_argument("--output",   required=True,
                        help="Output directory for CSV files")
    parser.add_argument("--start",    default="2002-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",      default="2023-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--chunk-years", type=int, default=2,
                        help="Years per GEE request (reduce if timeouts occur)")
    parser.add_argument("--igbp",     default=None,
                        help="Comma-separated IGBP classes to filter (e.g. ENF,DBF)")
    parser.add_argument("--project",  default=None,
                        help="GEE cloud project ID (required for newer ee versions)")
    args = parser.parse_args()

    # ── Initialise GEE ──────────────────────────────────────────────────────
    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()

    # ── Load site info ───────────────────────────────────────────────────────
    sites = pd.read_csv(args.siteinfo, encoding="utf-8-sig")
    sites = sites.rename(columns=str.strip)

    # Drop rows with missing coordinates
    sites = sites.dropna(subset=["LOCATION_LAT", "LOCATION_LONG"])
    sites["LOCATION_LAT"]  = pd.to_numeric(sites["LOCATION_LAT"],  errors="coerce")
    sites["LOCATION_LONG"] = pd.to_numeric(sites["LOCATION_LONG"], errors="coerce")
    sites = sites.dropna(subset=["LOCATION_LAT", "LOCATION_LONG"])

    # Optional IGBP filter
    if args.igbp:
        keep = [s.strip() for s in args.igbp.split(",")]
        sites = sites[sites["IGBP"].isin(keep)]
        print(f"Filtered to IGBP {keep}: {len(sites)} sites")

    print(f"Sites loaded: {len(sites)}")

    # ── Build GEE FeatureCollection ──────────────────────────────────────────
    fc = build_site_fc(sites)
    print(f"GEE FeatureCollection: {fc.size().getInfo()} features")

    # ── Fetch in year chunks ─────────────────────────────────────────────────
    year_start = int(args.start[:4])
    year_end   = int(args.end[:4])
    chunk      = args.chunk_years

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_4day_path = out_dir / "lai_modis_4day.csv"
    all_chunks: list[pd.DataFrame] = []

    # Resume support: skip already-fetched years
    fetched_years: set[int] = set()
    if raw_4day_path.exists():
        existing = pd.read_csv(raw_4day_path, usecols=["date"], parse_dates=["date"])
        fetched_years = set(existing["date"].dt.year.unique())
        print(f"Resuming — already have years: {sorted(fetched_years)}")
        all_chunks.append(pd.read_csv(raw_4day_path))

    years = list(range(year_start, year_end + 1, chunk))
    for y0 in years:
        y1 = min(y0 + chunk - 1, year_end)

        # Skip if all years in this chunk are already fetched
        chunk_years = set(range(y0, y1 + 1))
        if chunk_years.issubset(fetched_years):
            print(f"  {y0}–{y1}: already fetched, skipping.")
            continue

        print(f"  Fetching {y0}–{y1} ({len(sites)} sites) ...", end=" ", flush=True)
        t0 = time.time()
        chunk_df = fetch_year_chunk(fc, y0, y1)
        elapsed  = time.time() - t0
        print(f"{len(chunk_df):,} rows in {elapsed:.1f} s")

        if not chunk_df.empty:
            all_chunks.append(chunk_df)
            # Save incrementally so progress is preserved on crash
            pd.concat(all_chunks, ignore_index=True).to_csv(raw_4day_path, index=False)

    if not all_chunks:
        print("No data retrieved. Check GEE authentication and date range.")
        return

    raw = pd.concat(all_chunks, ignore_index=True).drop_duplicates()

    # ── Merge site metadata ──────────────────────────────────────────────────
    meta = sites[["SITE_ID", "IGBP", "LOCATION_LAT", "LOCATION_LONG"]].copy()
    meta.columns = ["site_id", "igbp", "lat_site", "lon_site"]
    raw = raw.merge(meta, on="site_id", how="left")

    # ── Save raw 4-day ───────────────────────────────────────────────────────
    raw.to_csv(raw_4day_path, index=False)
    print(f"\nSaved 4-day LAI: {raw_4day_path}  ({len(raw):,} rows)")

    # ── Monthly aggregation ──────────────────────────────────────────────────
    monthly = to_monthly(raw)
    monthly = monthly.merge(meta, on="site_id", how="left")

    monthly_path = out_dir / "lai_modis_monthly.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"Saved monthly LAI: {monthly_path}  ({len(monthly):,} rows)")

    # ── Annual aggregation ───────────────────────────────────────────────────
    annual = to_annual(monthly)
    annual = annual.merge(meta, on="site_id", how="left")

    annual_path = out_dir / "lai_modis_annual.csv"
    annual.to_csv(annual_path, index=False)
    print(f"Saved annual  LAI: {annual_path}  ({len(annual):,} rows)")

    # ── Summary by IGBP ─────────────────────────────────────────────────────
    print("\n── Mean annual LAI by IGBP class ──────────────────────────")
    summary = (annual.groupby("igbp")[["LAI_annual_mean", "LAI_amplitude"]]
                     .agg(["mean", "std"])
                     .round(2))
    print(summary.to_string())

    # ── Per-site coverage report ─────────────────────────────────────────────
    coverage = (monthly.groupby("site_id")
                       .agg(n_months=("LAI_mean", "count"),
                            year_start=("year", "min"),
                            year_end=("year", "max"))
                       .reset_index()
                       .merge(meta[["site_id", "igbp"]], on="site_id"))
    coverage_path = out_dir / "lai_modis_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    print(f"\nSite coverage report: {coverage_path}")
    n_sparse = (coverage["n_months"] < 12).sum()
    if n_sparse:
        print(f"  {n_sparse} sites have < 12 months of valid LAI (cloud/snow gaps)")


if __name__ == "__main__":
    main()
