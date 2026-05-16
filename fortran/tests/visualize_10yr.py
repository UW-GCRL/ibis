#!/usr/bin/env python3
"""
visualize_10yr.py  --  Visualize the 10-year India domain IBIS simulation

Produces tests/figures/fig_10yr_*.png
"""

import sys, os, subprocess, importlib

VENV_DIR = os.path.join(os.path.dirname(__file__), "venv")
venv_python = os.path.join(VENV_DIR, "bin", "python3")

def ensure_deps():
    missing = []
    for p in ("netCDF4", "numpy", "matplotlib"):
        try:
            importlib.import_module(p)
        except ImportError:
            missing.append(p)
    if missing:
        subprocess.check_call([venv_python, "-m", "pip", "install", "--quiet"] + missing)
        os.execv(venv_python, [venv_python] + sys.argv)

ensure_deps()

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RUNDIR  = os.path.join(os.path.dirname(__file__), "../runs/india_10yr")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def load(relpath, varname):
    path = os.path.join(RUNDIR, relpath)
    with netCDF4.Dataset(path) as ds:
        v    = ds.variables[varname]
        data = np.array(v[:], dtype=float)
        fill = getattr(v, "_FillValue", 9e20)
        data[data >= 0.9 * fill] = np.nan
        lons = ds.variables["longitude"][:]
        lats = ds.variables["latitude"][:]
    return data, lats, lons


def area_mean(data):
    return np.nanmean(data.reshape(data.shape[0], -1), axis=1)


# ============================================================================
# Figure 1: 10-year time series of area-mean monthly climate & ecosystem vars
# ============================================================================
monthly_vars = [
    ("output/monthly/temp.nc",   "temp",   "Air temperature",  "°C"),
    ("output/monthly/rain.nc",   "rain",   "Precipitation",    "mm day⁻¹"),
    ("output/monthly/aet.nc",    "aet",    "Actual ET",        "mm day⁻¹"),
    ("output/monthly/lai.nc",    "laiu",   "LAI – upper",      "m² m⁻²"),
    ("output/monthly/solar.nc",  "solar",  "Solar radiation",  "W m⁻²"),
    ("output/monthly/npptot.nc", "npptot", "NPP total",        "kg m⁻² mo⁻¹"),
]

fig1, axes = plt.subplots(3, 2, figsize=(14, 11))
axes = axes.flatten()
fig1.suptitle("IBIS 10-year simulation – India domain (1961–1970)\nMonthly area-mean time series",
              fontsize=13, fontweight="bold")

for ax, (relpath, varname, label, ylabel) in zip(axes, monthly_vars):
    try:
        d, lats, lons = load(relpath, varname)
    except Exception as e:
        ax.set_title(f"{label}\n(error: {e})"); continue

    ts = area_mean(d)
    nt = len(ts)
    x  = np.arange(nt) / 12.0 + 1961.0

    ax.plot(x, ts, color="#1f77b4", lw=1.2, alpha=0.8)

    # Overlay 12-month rolling mean
    if nt >= 12:
        rm = np.convolve(ts, np.ones(12)/12, mode="valid")
        xrm = x[5:5+len(rm)]
        ax.plot(xrm, rm, color="#d62728", lw=2.0, label="12-mo mean")

    # Shade monsoon months (Jun-Sep = months 6-9)
    for yr in range(1961, 1971):
        ax.axvspan(yr + 5/12, yr + 9/12, alpha=0.07, color="steelblue")

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    if ax is axes[0]:
        ax.legend(fontsize=8, loc="upper right")

fig1.tight_layout()
out1 = os.path.join(FIG_DIR, "fig_10yr_timeseries.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)


# ============================================================================
# Figure 2: Yearly maps – change from year 1 to year 10
# (LAI, AET, NPP, biomass, soil carbon)
# ============================================================================
yearly_vars = [
    ("output/yearly/plai.nc",    "totlaiu",  "LAI upper",      "m² m⁻²"),
    ("output/yearly/aet.nc",     "aet",      "AET",            "mm yr⁻¹"),
    ("output/yearly/npp.nc",     "npptot",   "NPP total",      "kg m⁻² yr⁻¹"),
    ("output/yearly/biomass.nc", "totbiou",  "Biomass (upper)","kg m⁻²"),
    ("output/yearly/csoi.nc",    "totcsoi",  "Soil carbon",    "kg m⁻²"),
]

fig2, axes2 = plt.subplots(3, len(yearly_vars), figsize=(18, 9))
fig2.suptitle("IBIS 10-year simulation – Yearly output\nYear 1961 | Year 1970 | Change (1970−1961)",
              fontsize=12, fontweight="bold")

for col, (relpath, varname, label, units) in enumerate(yearly_vars):
    try:
        d, lats, lons = load(relpath, varname)
    except Exception as e:
        for row in range(3):
            axes2[row, col].set_title(f"{label}\n(error)")
        continue

    yr1   = d[0]   if d.ndim == 3 else d[0,0]
    yr10  = d[-1]  if d.ndim == 3 else d[-1,0]

    # Use ncdump shape: (nyears, nlat, nlon) or (nyears, 1, nlat, nlon)?
    # Our files are (nyears, nlat, nlon) from the nc structure
    yr1  = d[0]
    yr10 = d[-1]
    diff = yr10 - yr1

    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    all_vals = np.concatenate([yr1[np.isfinite(yr1)], yr10[np.isfinite(yr10)]])
    lo = np.nanpercentile(all_vals, 2)
    hi = np.nanpercentile(all_vals, 98)

    for row, (data, title) in enumerate([(yr1, "Year 1961"), (yr10, "Year 1970"), (diff, "Change")]):
        ax = axes2[row, col]
        if row < 2:
            im = ax.imshow(data, origin="lower", extent=ext, aspect="auto",
                           cmap="YlGn", vmin=lo, vmax=hi)
        else:
            absmax = max(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 99), 1e-9)
            im = ax.imshow(data, origin="lower", extent=ext, aspect="auto",
                           cmap="RdBu_r", vmin=-absmax, vmax=absmax)
        cb = fig2.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        cb.set_label(units, fontsize=6)
        if row == 0:
            ax.set_title(f"{label}\n{title}", fontsize=8, fontweight="bold")
        else:
            ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Lon", fontsize=6)
        if col == 0:
            ax.set_ylabel("Lat", fontsize=7)

fig2.tight_layout()
out2 = os.path.join(FIG_DIR, "fig_10yr_yearly_maps.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)


# ============================================================================
# Figure 3: Annual mean seasonal cycle — averaged over all 10 years
# ============================================================================
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
axes3 = axes3.flatten()
fig3.suptitle("IBIS 10-year simulation – Mean seasonal cycle (1961–1970 average)\nwith inter-year spread (shading = ±1 σ)",
              fontsize=12, fontweight="bold")

sc_vars = [
    ("output/monthly/temp.nc",   "temp",   "Air temperature",  "°C"),
    ("output/monthly/rain.nc",   "rain",   "Precipitation",    "mm day⁻¹"),
    ("output/monthly/aet.nc",    "aet",    "Actual ET",        "mm day⁻¹"),
    ("output/monthly/lai.nc",    "laiu",   "LAI – upper",      "m² m⁻²"),
    ("output/monthly/solar.nc",  "solar",  "Solar radiation",  "W m⁻²"),
    ("output/monthly/cloud.nc",  "cloud",  "Cloud cover",      "%"),
]

for ax, (relpath, varname, label, ylabel) in zip(axes3, sc_vars):
    try:
        d, _, _ = load(relpath, varname)
    except Exception as e:
        ax.set_title(f"{label}\n(error)"); continue

    ts = area_mean(d)          # length = 10*12 = 120
    nyears = len(ts) // 12
    ts_mat = ts[:nyears*12].reshape(nyears, 12)

    mean_sc = ts_mat.mean(axis=0)
    std_sc  = ts_mat.std(axis=0)

    mx = np.arange(1, 13)
    ax.fill_between(mx, mean_sc - std_sc, mean_sc + std_sc, alpha=0.25, color="#1f77b4")
    ax.plot(mx, mean_sc, "o-", color="#1f77b4", lw=2, ms=5)

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS, fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.97, f"n={nyears} yr", ha="right", va="top",
            transform=ax.transAxes, fontsize=7, color="#555")

fig3.tight_layout()
out3 = os.path.join(FIG_DIR, "fig_10yr_seasonal_cycle.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig3)

print("\nAll 10-year figures written to tests/figures/")
