#!/usr/bin/env python3
"""
visualize_results.py  --  Compare F77 vs F2008 IBIS monthly output

Produces two figures saved to tests/figures/:
  fig1_seasonal_cycle.png  -- area-mean seasonal cycles, 8 variables
  fig2_spatial_maps.png    -- Jan/Jul spatial maps (4 variables × 3 rows: F77, F2008, diff)

Usage:
    python3 tests/visualize_results.py
"""

import sys, os, subprocess, importlib

# ---------------------------------------------------------------------------
# Bootstrap matplotlib into the existing venv
# ---------------------------------------------------------------------------
VENV_DIR = os.path.join(os.path.dirname(__file__), "venv")
venv_python = os.path.join(VENV_DIR, "bin", "python3")

def ensure_deps():
    missing = []
    for pkg in ("netCDF4", "numpy", "matplotlib"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing {missing} into venv ...", flush=True)
        subprocess.check_call([venv_python, "-m", "pip", "install", "--quiet"] + missing)
        os.execv(venv_python, [venv_python] + sys.argv)

ensure_deps()

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO    = os.path.dirname(os.path.dirname(__file__))
F77_DIR = os.path.join(REPO, "tests/regression/run_f77")
F08_DIR = os.path.join(REPO, "tests/regression/run_f2008")
FIG_DIR = os.path.join(REPO, "tests/figures")
os.makedirs(FIG_DIR, exist_ok=True)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# Helper: load a variable from a NetCDF file, replacing fill with NaN
# ---------------------------------------------------------------------------
def load(base_dir, relpath, varname):
    path = os.path.join(base_dir, relpath)
    with netCDF4.Dataset(path) as ds:
        v    = ds.variables[varname]
        data = np.array(v[:], dtype=float)
        fill = getattr(v, "_FillValue", 9e20)
        data[data >= 0.9 * fill] = np.nan
        lons = ds.variables["longitude"][:]
        lats = ds.variables["latitude"][:]
    return data, lats, lons   # (12, nlat, nlon)


def area_mean(data):
    """Spatial mean over non-NaN land cells (equal-weight per cell)."""
    return np.nanmean(data.reshape(data.shape[0], -1), axis=1)


# ============================================================================
# Figure 1 – Seasonal cycles
# ============================================================================
panels = [
    # (file,               varname,   label,              ylabel,       scale)
    ("output/monthly/temp.nc",    "temp",    "Temperature",     "°C",         1.0),
    ("output/monthly/rain.nc",    "rain",    "Precipitation",   "mm day⁻¹",   1.0),
    ("output/monthly/cloud.nc",   "cloud",   "Cloud cover",     "%",          1.0),
    ("output/monthly/rh.nc",      "rh",      "Relative humidity","%" ,         1.0),
    ("output/monthly/aet.nc",     "aet",     "Actual ET",       "mm day⁻¹",   1.0),
    ("output/monthly/lai.nc",     "laiu",    "LAI – upper canopy","m² m⁻²",   1.0),
    ("output/monthly/solar.nc",   "solar",   "Solar radiation", "W m⁻²",      1.0),
    ("output/monthly/npptot.nc",  "npptot",  "NPP total",       "kg m⁻² mo⁻¹",1.0),
]

fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
fig1.suptitle("IBIS 1-year simulation – India domain\nF77 vs F2008: monthly area-mean seasonal cycles",
              fontsize=13, fontweight="bold")

for ax, (relpath, varname, label, ylabel, scale) in zip(axes, panels):
    try:
        d77, lats, lons = load(F77_DIR, relpath, varname)
        d08, _,    _    = load(F08_DIR, relpath, varname)
    except Exception as e:
        ax.set_title(f"{label}\n(load error: {e})")
        continue

    m77 = area_mean(d77) * scale
    m08 = area_mean(d08) * scale

    ax.plot(range(1, 13), m77, "o-",  color="#2166ac", lw=2, ms=5, label="F77")
    ax.plot(range(1, 13), m08, "s--", color="#d6604d", lw=2, ms=5, label="F2008")

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS, fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3)
    if ax is axes[0]:
        ax.legend(fontsize=8)

    # Annotate max relative diff
    reldiff = np.nanmax(np.abs(m77 - m08) / np.where(np.abs(m77) > 1e-6, np.abs(m77), 1e-6))
    ax.text(0.98, 0.97, f"max Δrel = {reldiff:.1%}",
            ha="right", va="top", transform=ax.transAxes, fontsize=7, color="#555")

fig1.tight_layout()
out1 = os.path.join(FIG_DIR, "fig1_seasonal_cycle.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)


# ============================================================================
# Figure 2 – Spatial maps: January (month 0) and July (month 6)
# ============================================================================
map_panels = [
    ("output/monthly/temp.nc",  "temp",  "Temperature",    "°C",       "RdBu_r",   None,        None),
    ("output/monthly/rain.nc",  "rain",  "Precipitation",  "mm day⁻¹", "Blues",    0,           None),
    ("output/monthly/aet.nc",   "aet",   "Actual ET",      "mm day⁻¹", "YlGn",     0,           None),
    ("output/monthly/lai.nc",   "laiu",  "LAI (upper)",    "m² m⁻²",   "Greens",   0,           None),
]

# 3 rows (F77, F2008, Diff) × 4 variables × 2 seasons (Jan, Jul)
# Layout: 3 rows × 8 cols
nvar = len(map_panels)
ncol = nvar * 2   # each variable gets a Jan and Jul column
nrow = 3

fig2 = plt.figure(figsize=(20, 10))
gs   = GridSpec(nrow, ncol, figure=fig2,
                hspace=0.45, wspace=0.35,
                left=0.04, right=0.96, top=0.90, bottom=0.06)
fig2.suptitle("IBIS spatial maps – F77 (top) | F2008 (mid) | Difference F2008−F77 (bot)",
              fontsize=13, fontweight="bold")

season_months = [0, 6]   # January (index 0), July (index 6)
season_labels = ["January", "July"]

for vi, (relpath, varname, label, units, cmap, vmin, vmax) in enumerate(map_panels):
    try:
        d77, lats, lons = load(F77_DIR, relpath, varname)
        d08, _,    _    = load(F08_DIR, relpath, varname)
    except Exception as e:
        print(f"  skip {varname}: {e}")
        continue

    for si, (mi, slabel) in enumerate(zip(season_months, season_labels)):
        col = vi * 2 + si
        m77  = d77[mi]
        m08  = d08[mi]
        diff = m08 - m77

        # Common colour scale for F77 / F2008 panels
        all_vals = np.concatenate([m77[np.isfinite(m77)], m08[np.isfinite(m08)]])
        lo = np.nanpercentile(all_vals, 2)  if vmin is None else vmin
        hi = np.nanpercentile(all_vals, 98) if vmax is None else vmax

        extent = [lons.min(), lons.max(), lats.min(), lats.max()]

        for row_idx, (data, row_label) in enumerate([(m77, "F77"), (m08, "F2008"), (diff, "Diff")]):
            ax = fig2.add_subplot(gs[row_idx, col])

            if row_idx < 2:
                im = ax.imshow(data, origin="lower", extent=extent,
                               aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
                cb = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=6)
                cb.set_label(units, fontsize=6)
            else:
                # Diverging colourmap for difference panel
                absmax = max(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 99), 1e-9)
                im = ax.imshow(diff, origin="lower", extent=extent,
                               aspect="auto", cmap="RdBu_r",
                               vmin=-absmax, vmax=absmax)
                cb = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=6)
                cb.set_label(units, fontsize=6)

            if row_idx == 0:
                ax.set_title(f"{label}\n{slabel}", fontsize=8, fontweight="bold")
            ax.set_xlabel("Lon", fontsize=6)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nLat", fontsize=7)
            else:
                ax.set_ylabel("Lat", fontsize=6)
            ax.tick_params(labelsize=6)

out2 = os.path.join(FIG_DIR, "fig2_spatial_maps.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)


# ============================================================================
# Figure 3 – Difference summary: box-plot of per-cell differences each month
# ============================================================================
diff_vars = [
    ("output/monthly/temp.nc",  "temp",  "Temp (°C)"),
    ("output/monthly/rain.nc",  "rain",  "Rain (mm/d)"),
    ("output/monthly/aet.nc",   "aet",   "AET (mm/d)"),
    ("output/monthly/lai.nc",   "laiu",  "LAIu (m²/m²)"),
    ("output/monthly/cloud.nc", "cloud", "Cloud (%)"),
    ("output/monthly/solar.nc", "solar", "Solar (W/m²)"),
]

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
axes3 = axes3.flatten()
fig3.suptitle("F2008 − F77 monthly differences per grid cell\n(box = IQR, whiskers = 5th–95th percentile)",
              fontsize=12, fontweight="bold")

for ax, (relpath, varname, label) in zip(axes3, diff_vars):
    try:
        d77, _, _ = load(F77_DIR, relpath, varname)
        d08, _, _ = load(F08_DIR, relpath, varname)
    except Exception as e:
        ax.set_title(f"{label}\n(error)")
        continue

    diff = d08 - d77   # (12, nlat, nlon)

    # Build list of per-month arrays of valid differences
    data_per_month = []
    for m in range(12):
        vals = diff[m].ravel()
        data_per_month.append(vals[np.isfinite(vals)])

    bp = ax.boxplot(data_per_month,
                    positions=range(1, 13),
                    widths=0.6,
                    whis=[5, 95],
                    patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    boxprops=dict(facecolor="#a6cee3", alpha=0.7),
                    flierprops=dict(marker=".", markersize=1, alpha=0.3))

    ax.axhline(0, color="red", lw=0.8, ls="--")
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS, fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")

fig3.tight_layout()
out3 = os.path.join(FIG_DIR, "fig3_difference_boxplots.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig3)

print("\nAll figures written to tests/figures/")
