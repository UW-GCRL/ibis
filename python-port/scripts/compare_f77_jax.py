#!/usr/bin/env python3
"""Compare IBIS F77 and IBIS-JAX outputs for the India test domain.

Produces a multi-panel comparison figure with:
  - Spatial maps of soil temperature and moisture (F77 vs JAX)
  - Difference maps
  - Scatter plots (point-by-point correlation)
  - Histograms of differences
  - F77 10-year time series (monthly) with JAX year-1 overlaid

Usage:
    python scripts/compare_f77_jax.py [--output comparison.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
F77_DIR = Path("/Users/minchen/Documents/Work/IBIS/runs/india_10yr/output")
JAX_DIR = Path("/Users/minchen/Documents/Work/ibis-jax/output")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_f77_yearly(var: str, year_idx: int = 0) -> xr.Dataset:
    ds = xr.open_dataset(F77_DIR / "yearly" / f"{var}.nc", decode_times=False)
    return ds

def load_f77_monthly(var: str) -> xr.Dataset:
    ds = xr.open_dataset(F77_DIR / "monthly" / f"{var}.nc", decode_times=False)
    return ds

def load_jax_yearly(year: int = 1961) -> xr.Dataset:
    ds = xr.open_dataset(JAX_DIR / f"ibis_yearly_{year}.nc")
    return ds

def align_grids(f77_lat, f77_lon, jax_lat, jax_lon):
    """Find common latitude/longitude indices between the two grids."""
    # Both are on 0.5-degree grid; find overlapping coordinates
    lat_common = np.intersect1d(np.round(f77_lat, 2), np.round(jax_lat, 2))
    lon_common = np.intersect1d(np.round(f77_lon, 2), np.round(jax_lon, 2))

    f77_lat_idx = [np.argmin(np.abs(f77_lat - l)) for l in lat_common]
    f77_lon_idx = [np.argmin(np.abs(f77_lon - l)) for l in lon_common]
    jax_lat_idx = [np.argmin(np.abs(jax_lat - l)) for l in lat_common]
    jax_lon_idx = [np.argmin(np.abs(jax_lon - l)) for l in lon_common]

    return lat_common, lon_common, f77_lat_idx, f77_lon_idx, jax_lat_idx, jax_lon_idx


def spatial_stats(diff):
    """Compute statistics on non-NaN differences."""
    valid = diff[~np.isnan(diff)]
    if len(valid) == 0:
        return {"mean": np.nan, "rmse": np.nan, "bias": np.nan, "r": np.nan, "n": 0}
    return {
        "mean": np.mean(valid),
        "rmse": np.sqrt(np.mean(valid ** 2)),
        "bias": np.mean(valid),
        "std": np.std(valid),
        "n": len(valid),
    }


def add_map(ax, data, lat, lon, title, cmap="RdYlBu_r", vmin=None, vmax=None,
            cbar_label=""):
    """Add a filled contour map to an axes."""
    LON, LAT = np.meshgrid(lon, lat)
    im = ax.pcolormesh(LON, LAT, data, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(cbar_label, fontsize=8)
    return im


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main(output_path: str = "comparison_f77_jax.png"):

    # ---- Load data ----
    # F77 yearly (annual average, in degC for tsoi, fraction for wsoi)
    f77_tsoi_ds = load_f77_yearly("tsoi")
    f77_wsoi_ds = load_f77_yearly("wsoi")
    f77_aet_ds = load_f77_yearly("aet")
    f77_sens_ds = load_f77_yearly("sens")
    f77_npp_ds = load_f77_yearly("npp")

    f77_lat = f77_tsoi_ds.latitude.values
    f77_lon = f77_tsoi_ds.longitude.values

    # F77 year 1 data
    f77_tsoi_yr1 = f77_tsoi_ds["tsoi"].isel(time=0).values  # degC
    f77_wsoi_yr1 = f77_wsoi_ds["wsoi"].isel(time=0).values  # fraction
    f77_aet_yr1 = f77_aet_ds["aet"].isel(time=0).values     # mm/yr
    f77_sens_yr1 = f77_sens_ds["sens"].isel(time=0).values   # W/m2

    # JAX yearly (end-of-year snapshot, tsoi in K, wsoi in fraction)
    jax_ds = load_jax_yearly(1961)
    jax_lat = jax_ds.lat.values
    jax_lon = jax_ds.lon.values

    jax_tsoi_K = jax_ds["tsoi"].isel(soilay=0).values       # K, top layer
    jax_tsoi_C = jax_tsoi_K - 273.16                         # convert to degC
    jax_wsoi = jax_ds["wsoi"].isel(soilay=0).values          # fraction

    # ---- Align grids ----
    lat_c, lon_c, f77_li, f77_loi, jax_li, jax_loi = align_grids(
        f77_lat, f77_lon, jax_lat, jax_lon
    )

    # Extract common grid
    f77_tsoi_c = f77_tsoi_yr1[np.ix_(f77_li, f77_loi)]
    f77_wsoi_c = f77_wsoi_yr1[np.ix_(f77_li, f77_loi)]
    jax_tsoi_c = jax_tsoi_C[np.ix_(jax_li, jax_loi)]
    jax_wsoi_c = jax_wsoi[np.ix_(jax_li, jax_loi)]

    # Mask ocean (NaN in either)
    mask = np.isnan(f77_tsoi_c) | np.isnan(jax_tsoi_c)
    f77_tsoi_c[mask] = np.nan
    jax_tsoi_c[mask] = np.nan
    f77_wsoi_c[mask] = np.nan
    jax_wsoi_c[mask] = np.nan

    diff_tsoi = jax_tsoi_c - f77_tsoi_c
    diff_wsoi = jax_wsoi_c - f77_wsoi_c

    # ---- F77 monthly time series (domain average) ----
    f77_tsoi_mon = load_f77_monthly("tsoi")
    f77_wsoi_mon = load_f77_monthly("wsoi")
    f77_sens_mon = load_f77_monthly("sens")
    f77_rain_mon = load_f77_monthly("rain")

    nmon = f77_tsoi_mon.dims["time"]
    months = np.arange(1, nmon + 1)

    def monthly_avg(ds, var):
        data = ds[var].values
        return np.array([np.nanmean(data[t]) for t in range(data.shape[0])])

    f77_tsoi_ts = monthly_avg(f77_tsoi_mon, "tsoi")
    f77_wsoi_ts = monthly_avg(f77_wsoi_mon, "wsoi")
    f77_sens_ts = monthly_avg(f77_sens_mon, "sens")
    f77_rain_ts = monthly_avg(f77_rain_mon, "rain")

    # ---- Stats ----
    tsoi_stats = spatial_stats(diff_tsoi)
    wsoi_stats = spatial_stats(diff_wsoi)

    print("=" * 60)
    print("IBIS F77 vs JAX Comparison - India Domain, Year 1 (1961)")
    print("=" * 60)
    print(f"Note: F77 = annual average (degC), JAX = end-of-year snapshot (K->degC)")
    print(f"      F77 has vegetation (static), JAX starts bare ground")
    print()
    print(f"Soil Temperature (top layer, degC):")
    print(f"  F77: {np.nanmean(f77_tsoi_c):.2f} +/- {np.nanstd(f77_tsoi_c):.2f}")
    print(f"  JAX: {np.nanmean(jax_tsoi_c):.2f} +/- {np.nanstd(jax_tsoi_c):.2f}")
    print(f"  Bias (JAX-F77): {tsoi_stats['bias']:.2f} degC")
    print(f"  RMSE: {tsoi_stats['rmse']:.2f} degC")
    print(f"  N points: {tsoi_stats['n']}")
    print()
    print(f"Soil Moisture (top layer, fraction):")
    print(f"  F77: {np.nanmean(f77_wsoi_c):.4f} +/- {np.nanstd(f77_wsoi_c):.4f}")
    print(f"  JAX: {np.nanmean(jax_wsoi_c):.4f} +/- {np.nanstd(jax_wsoi_c):.4f}")
    print(f"  Bias (JAX-F77): {wsoi_stats['bias']:.4f}")
    print(f"  RMSE: {wsoi_stats['rmse']:.4f}")
    print()

    # ---- Correlation ----
    valid = ~(np.isnan(f77_tsoi_c.ravel()) | np.isnan(jax_tsoi_c.ravel()))
    if valid.sum() > 2:
        r_tsoi = np.corrcoef(f77_tsoi_c.ravel()[valid], jax_tsoi_c.ravel()[valid])[0, 1]
    else:
        r_tsoi = np.nan
    valid_w = ~(np.isnan(f77_wsoi_c.ravel()) | np.isnan(jax_wsoi_c.ravel()))
    if valid_w.sum() > 2:
        r_wsoi = np.corrcoef(f77_wsoi_c.ravel()[valid_w], jax_wsoi_c.ravel()[valid_w])[0, 1]
    else:
        r_wsoi = np.nan
    print(f"Spatial correlation tsoi: r = {r_tsoi:.4f}")
    print(f"Spatial correlation wsoi: r = {r_wsoi:.4f}")
    print()

    # ====================================================================
    # FIGURE 1: Spatial maps
    # ====================================================================
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 3, hspace=0.35, wspace=0.30,
                           left=0.06, right=0.94, top=0.95, bottom=0.04)

    fig.suptitle(
        "IBIS F77 vs IBIS-JAX Comparison\nIndia Domain (7.5-32.5N, 68.5-95.5E), Year 1961",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # --- Row 1: Soil Temperature maps ---
    tsoi_vmin = min(np.nanmin(f77_tsoi_c), np.nanmin(jax_tsoi_c))
    tsoi_vmax = max(np.nanmax(f77_tsoi_c), np.nanmax(jax_tsoi_c))

    ax1 = fig.add_subplot(gs[0, 0])
    add_map(ax1, f77_tsoi_c, lat_c, lon_c, "F77: Soil Temp (annual avg)",
            vmin=tsoi_vmin, vmax=tsoi_vmax, cbar_label="degC")

    ax2 = fig.add_subplot(gs[0, 1])
    add_map(ax2, jax_tsoi_c, lat_c, lon_c, "JAX: Soil Temp (end-of-year)",
            vmin=tsoi_vmin, vmax=tsoi_vmax, cbar_label="degC")

    ax3 = fig.add_subplot(gs[0, 2])
    dlim = max(abs(np.nanmin(diff_tsoi)), abs(np.nanmax(diff_tsoi)), 5)
    add_map(ax3, diff_tsoi, lat_c, lon_c,
            f"Difference (JAX - F77)\nbias={tsoi_stats['bias']:.1f}, RMSE={tsoi_stats['rmse']:.1f} degC",
            cmap="RdBu_r", vmin=-dlim, vmax=dlim, cbar_label="degC")

    # --- Row 2: Soil Moisture maps ---
    wsoi_vmin = min(np.nanmin(f77_wsoi_c), np.nanmin(jax_wsoi_c))
    wsoi_vmax = max(np.nanmax(f77_wsoi_c), np.nanmax(jax_wsoi_c))

    ax4 = fig.add_subplot(gs[1, 0])
    add_map(ax4, f77_wsoi_c, lat_c, lon_c, "F77: Soil Moisture (annual avg)",
            cmap="YlGnBu", vmin=wsoi_vmin, vmax=wsoi_vmax, cbar_label="fraction")

    ax5 = fig.add_subplot(gs[1, 1])
    add_map(ax5, jax_wsoi_c, lat_c, lon_c, "JAX: Soil Moisture (end-of-year)",
            cmap="YlGnBu", vmin=wsoi_vmin, vmax=wsoi_vmax, cbar_label="fraction")

    ax6 = fig.add_subplot(gs[1, 2])
    dlim_w = max(abs(np.nanmin(diff_wsoi)), abs(np.nanmax(diff_wsoi)), 0.1)
    add_map(ax6, diff_wsoi, lat_c, lon_c,
            f"Difference (JAX - F77)\nbias={wsoi_stats['bias']:.3f}, RMSE={wsoi_stats['rmse']:.3f}",
            cmap="BrBG", vmin=-dlim_w, vmax=dlim_w, cbar_label="fraction")

    # --- Row 3: Scatter plots ---
    ax7 = fig.add_subplot(gs[2, 0])
    f77_flat = f77_tsoi_c.ravel()
    jax_flat = jax_tsoi_c.ravel()
    valid_s = ~(np.isnan(f77_flat) | np.isnan(jax_flat))
    ax7.scatter(f77_flat[valid_s], jax_flat[valid_s], s=4, alpha=0.5, c="steelblue")
    lims = [min(f77_flat[valid_s].min(), jax_flat[valid_s].min()) - 2,
            max(f77_flat[valid_s].max(), jax_flat[valid_s].max()) + 2]
    ax7.plot(lims, lims, "k--", lw=1, label="1:1")
    ax7.set_xlim(lims)
    ax7.set_ylim(lims)
    ax7.set_xlabel("F77 Soil Temp (degC)")
    ax7.set_ylabel("JAX Soil Temp (degC)")
    ax7.set_title(f"Soil Temp: r = {r_tsoi:.3f}", fontsize=10, fontweight="bold")
    ax7.legend(fontsize=8)
    ax7.set_aspect("equal")

    ax8 = fig.add_subplot(gs[2, 1])
    f77_wf = f77_wsoi_c.ravel()
    jax_wf = jax_wsoi_c.ravel()
    valid_ws = ~(np.isnan(f77_wf) | np.isnan(jax_wf))
    ax8.scatter(f77_wf[valid_ws], jax_wf[valid_ws], s=4, alpha=0.5, c="seagreen")
    lims_w = [min(f77_wf[valid_ws].min(), jax_wf[valid_ws].min()) - 0.05,
              max(f77_wf[valid_ws].max(), jax_wf[valid_ws].max()) + 0.05]
    ax8.plot(lims_w, lims_w, "k--", lw=1, label="1:1")
    ax8.set_xlim(lims_w)
    ax8.set_ylim(lims_w)
    ax8.set_xlabel("F77 Soil Moisture (fraction)")
    ax8.set_ylabel("JAX Soil Moisture (fraction)")
    ax8.set_title(f"Soil Moisture: r = {r_wsoi:.3f}", fontsize=10, fontweight="bold")
    ax8.legend(fontsize=8)
    ax8.set_aspect("equal")

    # --- Row 3, col 3: Histograms of differences ---
    ax9 = fig.add_subplot(gs[2, 2])
    valid_dt = diff_tsoi.ravel()[~np.isnan(diff_tsoi.ravel())]
    ax9.hist(valid_dt, bins=40, color="coral", alpha=0.7, edgecolor="black", lw=0.5)
    ax9.axvline(0, color="black", ls="--", lw=1)
    ax9.axvline(tsoi_stats["bias"], color="red", ls="-", lw=2,
                label=f"bias={tsoi_stats['bias']:.1f}")
    ax9.set_xlabel("Tsoi difference (JAX - F77, degC)")
    ax9.set_ylabel("Count")
    ax9.set_title("Distribution of Tsoi Differences", fontsize=10, fontweight="bold")
    ax9.legend(fontsize=8)

    # --- Row 4: F77 monthly time series (10 years) ---
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    x_months = np.arange(nmon)

    ax10 = fig.add_subplot(gs[3, :])
    ax10.plot(x_months, f77_tsoi_ts, "b-", lw=1.2, label="F77 Tsoi (monthly avg)")
    # Mark JAX end-of-year as a horizontal band
    jax_mean_tsoi = np.nanmean(jax_tsoi_c)
    ax10.axhline(jax_mean_tsoi, color="red", ls="--", lw=2,
                 label=f"JAX Tsoi domain-avg (end yr1) = {jax_mean_tsoi:.1f} degC")
    ax10.set_xlabel("Month (10 years: 1961-1970)")
    ax10.set_ylabel("Domain-avg Soil Temp (degC)")
    ax10.set_title("F77 Monthly Soil Temperature (10-yr) with JAX Year-1 Reference",
                   fontsize=10, fontweight="bold")
    # Year ticks
    for yr in range(10):
        ax10.axvline(yr * 12, color="gray", ls=":", lw=0.5, alpha=0.5)
    xticks = [yr * 12 + 6 for yr in range(10)]
    ax10.set_xticks(xticks)
    ax10.set_xticklabels([str(1961 + yr) for yr in range(10)])
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)

    # --- Row 5: F77 monthly moisture + rain ---
    ax11 = fig.add_subplot(gs[4, :])
    ax11_r = ax11.twinx()

    l1, = ax11.plot(x_months, f77_wsoi_ts, "g-", lw=1.2, label="F77 Wsoi")
    jax_mean_wsoi = np.nanmean(jax_wsoi_c)
    l2 = ax11.axhline(jax_mean_wsoi, color="darkgreen", ls="--", lw=2,
                       label=f"JAX Wsoi domain-avg = {jax_mean_wsoi:.3f}")
    l3, = ax11_r.plot(x_months, f77_rain_ts, color="steelblue", lw=0.8, alpha=0.7,
                      label="F77 Precip")

    ax11.set_xlabel("Month (10 years: 1961-1970)")
    ax11.set_ylabel("Soil Moisture (fraction)", color="green")
    ax11_r.set_ylabel("Precip (mm/day)", color="steelblue")
    ax11.set_title("F77 Monthly Soil Moisture & Precipitation with JAX Reference",
                   fontsize=10, fontweight="bold")
    for yr in range(10):
        ax11.axvline(yr * 12, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax11.set_xticks(xticks)
    ax11.set_xticklabels([str(1961 + yr) for yr in range(10)])
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in [l1, l2, l3]]
    ax11.legend(lines, labels, fontsize=9, loc="upper left")
    ax11.grid(True, alpha=0.3)

    # --- Row 6: F77 sensible heat + annotation ---
    ax12 = fig.add_subplot(gs[5, 0:2])
    ax12.plot(x_months, f77_sens_ts, "r-", lw=1.2, label="F77 Sensible Heat")
    ax12.set_xlabel("Month")
    ax12.set_ylabel("Sensible Heat (W/m2)")
    ax12.set_title("F77 Monthly Sensible Heat Flux", fontsize=10, fontweight="bold")
    for yr in range(10):
        ax12.axvline(yr * 12, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax12.set_xticks(xticks)
    ax12.set_xticklabels([str(1961 + yr) for yr in range(10)])
    ax12.legend(fontsize=9)
    ax12.grid(True, alpha=0.3)

    # Annotation panel
    ax13 = fig.add_subplot(gs[5, 2])
    ax13.axis("off")
    notes = (
        "Comparison Notes:\n\n"
        "F77 IBIS:\n"
        "  - 10-year run (1961-1970)\n"
        "  - Static vegetation (isimveg=0)\n"
        "  - Initialized with vegetation map\n"
        "  - Annual avg / monthly avg output\n"
        "  - Soil temp in degC\n\n"
        "JAX IBIS:\n"
        "  - 1-year run (1961)\n"
        "  - Cold start (bare ground)\n"
        "  - No initial vegetation\n"
        "  - End-of-year snapshot output\n"
        "  - Soil temp in K (converted)\n\n"
        f"Tsoi: bias={tsoi_stats['bias']:.1f}C, "
        f"RMSE={tsoi_stats['rmse']:.1f}C, r={r_tsoi:.3f}\n"
        f"Wsoi: bias={wsoi_stats['bias']:.3f}, "
        f"RMSE={wsoi_stats['rmse']:.3f}, r={r_wsoi:.3f}\n"
    )
    ax13.text(0.05, 0.95, notes, transform=ax13.transAxes,
              fontsize=9, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # ---- Save ----
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {output_path}")

    # Cleanup
    f77_tsoi_ds.close()
    f77_wsoi_ds.close()
    f77_aet_ds.close()
    f77_sens_ds.close()
    f77_npp_ds.close()
    jax_ds.close()
    f77_tsoi_mon.close()
    f77_wsoi_mon.close()
    f77_sens_mon.close()
    f77_rain_mon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare F77 and JAX IBIS outputs")
    parser.add_argument("--output", "-o", default="comparison_f77_jax.png",
                        help="Output figure path")
    args = parser.parse_args()
    main(args.output)
