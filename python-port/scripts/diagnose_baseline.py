#!/usr/bin/env python3
"""Comprehensive baseline diagnostic: compare model output against all ILAMB observations.

Produces:
  1. Multi-panel scatter + zonal + spatial comparison for each variable
  2. Summary statistics table
  3. Diagnostic analysis of structural biases

Usage:
    python scripts/diagnose_baseline.py --model-dir output/cal_v3 --years 1990-1999
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ILAMB_ROOT = "/home/min/ILAMB-Data"
TARGET_LAT = np.arange(89.75, -90, -0.5)
TARGET_LON = np.arange(-179.75, 180, 0.5)


# ============================================================================
# Observation loading
# ============================================================================

def _load_nc(fpath, varname=None, scale=1.0, time_slice=None):
    """Load a NetCDF, optionally time-slice, compute mean, regrid to 0.5°."""
    ds = xr.open_dataset(fpath, decode_times=False)
    if varname is None:
        skip = {'time_bnds', 'time_bounds', 'lat_bnds', 'lat_bounds',
                'lon_bnds', 'lon_bounds', 'nv', 'nb',
                'land_fraction', 'cell_area', 'area', 'time'}
        candidates = [v for v in ds.data_vars if v not in skip
                      and len(ds[v].dims) >= 2]
        varname = candidates[0]
    da = ds[varname]

    if 'time' in da.dims:
        da = da.mean(dim='time', skipna=True)

    for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old in da.dims:
            da = da.rename({old: new})

    # Ensure numeric dtype before interpolation
    if not np.issubdtype(da.dtype, np.floating):
        da = da.astype(np.float64)
    da = da.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    vals = da.values.astype(np.float64) * scale
    ds.close()
    return vals


def load_observations():
    """Load all available ILAMB benchmark datasets."""
    obs = {}

    # GPP: FLUXCOM (g/m²/day → kgC/m²/yr)
    fpath = f"{ILAMB_ROOT}/DATA/gpp/FLUXCOM/gpp.nc"
    if os.path.exists(fpath):
        ds = xr.open_dataset(fpath)
        gpp = ds['gpp'].where(ds['gpp'] >= 0).mean(dim='time', skipna=True)
        for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
            if old in gpp.dims:
                gpp = gpp.rename({old: new})
        gpp = gpp.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
        obs['gpp'] = {'data': gpp.values * 365.0 / 1000.0,
                      'units': 'kgC/m²/yr', 'label': 'GPP', 'source': 'FLUXCOM'}
        ds.close()

    # LAI: MODIS
    fpath = f"{ILAMB_ROOT}/DATA/lai/MODIS/lai_0.5x0.5.nc"
    if os.path.exists(fpath):
        obs['lai'] = {'data': _load_nc(fpath, 'lai'),
                      'units': 'm²/m²', 'label': 'LAI', 'source': 'MODIS'}

    # ET: GLEAMv3.3a (kg/m²/s → mm/yr)
    fpath = f"{ILAMB_ROOT}/DATA/evspsbl/GLEAMv3.3a/et.nc"
    if os.path.exists(fpath):
        obs['et'] = {'data': _load_nc(fpath, scale=86400 * 365),
                     'units': 'mm/yr', 'label': 'ET', 'source': 'GLEAMv3.3a'}

    # Latent heat: FLUXCOM (W/m²)
    fpath = f"{ILAMB_ROOT}/DATA/hfls/FLUXCOM/le.nc"
    if os.path.exists(fpath):
        obs['le'] = {'data': _load_nc(fpath),
                     'units': 'W/m²', 'label': 'Latent Heat', 'source': 'FLUXCOM'}

    # Sensible heat: FLUXCOM (W/m²)
    fpath = f"{ILAMB_ROOT}/DATA/hfss/FLUXCOM/sh.nc"
    if os.path.exists(fpath):
        obs['sh'] = {'data': _load_nc(fpath),
                     'units': 'W/m²', 'label': 'Sensible Heat', 'source': 'FLUXCOM'}

    # Biomass: ESACCI (Mg/ha → kgC/m²: ×0.1 for Mg/ha→kg/m², ×0.5 for biomass→C)
    fpath = f"{ILAMB_ROOT}/DATA/biomass/ESACCI/biomass.nc"
    if os.path.exists(fpath):
        obs['biomass'] = {'data': _load_nc(fpath, 'biomass', scale=0.1 * 0.5),
                          'units': 'kgC/m²', 'label': 'Biomass C', 'source': 'ESACCI'}

    # Soil carbon: HWSD (kgC/m²)
    fpath = f"{ILAMB_ROOT}/DATA/cSoil/HWSD/soilc_0.5x0.5.nc"
    if os.path.exists(fpath):
        vals = _load_nc(fpath, 'cSoilAbove1m')
        vals[vals > 1e10] = np.nan
        obs['csoil'] = {'data': vals, 'units': 'kgC/m²',
                        'label': 'Soil Carbon', 'source': 'HWSD'}

    # Ecosystem respiration: FLUXCOM (g/m²/day → kgC/m²/yr)
    fpath = f"{ILAMB_ROOT}/DATA/reco/FLUXCOM/reco.nc"
    if os.path.exists(fpath):
        ds = xr.open_dataset(fpath)
        reco = ds['reco'].where(ds['reco'] >= 0).mean(dim='time', skipna=True)
        for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
            if old in reco.dims:
                reco = reco.rename({old: new})
        reco = reco.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
        obs['reco'] = {'data': reco.values * 365.0 / 1000.0,
                       'units': 'kgC/m²/yr', 'label': 'Ecosystem Resp.', 'source': 'FLUXCOM'}
        ds.close()

    # Albedo: GEWEX
    fpath = f"{ILAMB_ROOT}/DATA/albedo/GEWEX.SRB/albedo_0.5x0.5.nc"
    if os.path.exists(fpath):
        obs['albedo'] = {'data': _load_nc(fpath, 'albedo'),
                         'units': '-', 'label': 'Albedo', 'source': 'GEWEX.SRB'}

    # Runoff: LORA (gridded, better than Dai station data)
    fpath = f"{ILAMB_ROOT}/DATA/mrro/LORA/LORA.nc"
    if os.path.exists(fpath):
        obs['runoff'] = {'data': _load_nc(fpath, scale=86400 * 365),
                         'units': 'mm/yr', 'label': 'Runoff', 'source': 'LORA'}
    else:
        fpath = f"{ILAMB_ROOT}/DATA/mrro/Dai/runoff.nc"
        if os.path.exists(fpath):
            obs['runoff'] = {'data': _load_nc(fpath, 'runoff', scale=86400 * 365),
                             'units': 'mm/yr', 'label': 'Runoff', 'source': 'Dai'}

    # Burnt area: GFED4.1S
    fpath = f"{ILAMB_ROOT}/DATA/burntArea/GFED4.1S/burntArea.nc"
    if os.path.exists(fpath):
        obs['burntarea'] = {'data': _load_nc(fpath, scale=100),  # fraction→%
                            'units': '%/yr', 'label': 'Burnt Area', 'source': 'GFED4.1S'}

    for k, v in obs.items():
        d = v['data']
        valid = np.isfinite(d) & (d != 0)
        logger.info("  %s (%s): mean=%.3f %s (%d valid pts)",
                    v['label'], v['source'],
                    np.nanmean(d[valid]) if valid.any() else np.nan,
                    v['units'], valid.sum())

    return obs


# ============================================================================
# Model output loading
# ============================================================================

def load_model(model_dir, years):
    """Load model annual/monthly means and map to observation variables."""
    model = {}

    # Try yearly files first
    accum = {}
    n = 0
    for y in years:
        fpath = os.path.join(model_dir, f"ibis_yearly_{y}.nc")
        if not os.path.exists(fpath):
            continue
        ds = xr.open_dataset(fpath)
        for v in ds.data_vars:
            arr = ds[v].values.astype(np.float64)
            if v not in accum:
                accum[v] = np.zeros_like(arr)
            accum[v] += np.nan_to_num(arr)
        n += 1
        ds.close()

    if n == 0:
        raise ValueError(f"No yearly files in {model_dir} for years {list(years)}")

    raw = {k: v / n for k, v in accum.items()}
    logger.info("Loaded %d yearly files, %d variables", n, len(raw))

    # Map to observation variables
    if 'aygpptot' in raw:
        model['gpp'] = raw['aygpptot']
    if 'laiu' in raw and 'lail' in raw:
        model['lai'] = raw['laiu'] + raw['lail']
    if 'ayaet' in raw:
        model['et'] = raw['ayaet']
        # Latent heat: ET (mm/yr) → W/m² = ET * Lv / seconds_per_year
        # Lv = 2.5e6 J/kg, 1 mm = 1 kg/m²
        model['le'] = raw['ayaet'] * 2.5e6 / (365.25 * 86400)
    if 'aytrunof' in raw:
        model['runoff'] = raw['aytrunof']
        if 'aytdrain' in raw:
            model['runoff'] = model['runoff'] + raw['aytdrain']
        # Clamp extreme runoff (NaN forcing artifacts) for cleaner comparison
        model['runoff'] = np.where(model['runoff'] > 5000, np.nan, model['runoff'])
    if 'totcsoi' in raw:
        csoil = raw['totcsoi']
        csoil[csoil == 0] = np.nan  # mask zero values (uninitialized)
        model['csoil'] = csoil
    if 'ayco2mic' in raw:
        reco = raw['ayco2mic']
        # If Rh is near zero (pre-decomp-fix), it's a bug artifact
        if np.nanmean(np.abs(reco[np.isfinite(reco)])) < 0.01:
            logger.warning("  ayco2mic near zero — decomp modifiers may be uninitialized")
        model['reco'] = reco
    if 'ayanpptot' in raw:
        model['npp'] = raw['ayanpptot']
    if 'cveg' in raw:
        model['biomass'] = raw['cveg']

    # We don't output biomass, SH, or albedo yet — flag them
    for k in model:
        d = model[k]
        valid = np.isfinite(d) & (d != 0)
        logger.info("  Model %s: mean=%.3f (%d valid pts)",
                    k, np.nanmean(d[valid]) if valid.any() else np.nan, valid.sum())

    return model, raw


# ============================================================================
# Statistics
# ============================================================================

def compute_stats(mod, obs_data):
    """Compute r, bias, RMSE, normalized RMSE for matched grid cells."""
    m = mod.ravel()
    o = obs_data.ravel()
    valid = np.isfinite(m) & np.isfinite(o) & (m != 0) & (o != 0)
    if valid.sum() < 10:
        return {'r': np.nan, 'bias': np.nan, 'rmse': np.nan,
                'nrmse': np.nan, 'n': 0, 'obs_mean': np.nan, 'mod_mean': np.nan}
    m, o = m[valid], o[valid]
    r = np.corrcoef(m, o)[0, 1]
    bias = np.mean(m - o)
    rmse = np.sqrt(np.mean((m - o) ** 2))
    nrmse = rmse / np.std(o) if np.std(o) > 0 else np.nan
    return {'r': r, 'bias': bias, 'rmse': rmse, 'nrmse': nrmse,
            'n': valid.sum(), 'obs_mean': np.mean(o), 'mod_mean': np.mean(m)}


def zonal_mean(data_2d):
    """Compute zonal mean ignoring NaN/zero."""
    d = data_2d.copy()
    d[d == 0] = np.nan
    return np.nanmean(d, axis=1)


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(mod_2d, obs_2d, label, units, outpath, vmax_override=None):
    """3-panel figure: spatial obs, spatial model, spatial diff + scatter + zonal."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    lats = TARGET_LAT

    # Auto-range
    valid_o = obs_2d[np.isfinite(obs_2d) & (obs_2d != 0)]
    valid_m = mod_2d[np.isfinite(mod_2d) & (mod_2d != 0)]
    if len(valid_o) > 0 and len(valid_m) > 0:
        vmax = vmax_override or max(np.percentile(valid_o, 98), np.percentile(valid_m, 98))
    else:
        vmax = 1.0
    vmin = 0

    # Spatial: Observed
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(obs_2d, vmin=vmin, vmax=vmax, cmap='YlGn',
                   aspect='auto', origin='upper', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=units)
    ax.set_title(f'Observed ({label})', fontsize=11, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # Spatial: Model
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(mod_2d, vmin=vmin, vmax=vmax, cmap='YlGn',
                   aspect='auto', origin='upper', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=units)
    ax.set_title(f'IBIS-JAX ({label})', fontsize=11, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # Spatial: Difference
    ax = fig.add_subplot(gs[0, 2])
    diff = mod_2d - obs_2d
    dmax = min(vmax * 0.75, np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95))
    im = ax.imshow(diff, vmin=-dmax, vmax=dmax, cmap='RdBu_r',
                   aspect='auto', origin='upper', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=units)
    ax.set_title(f'Model - Observed', fontsize=11, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # Scatter
    ax = fig.add_subplot(gs[1, 0])
    o_flat = obs_2d.ravel()
    m_flat = mod_2d.ravel()
    v = np.isfinite(o_flat) & np.isfinite(m_flat) & (o_flat != 0) & (m_flat != 0)
    if v.any():
        ax.scatter(o_flat[v], m_flat[v], s=0.3, alpha=0.2, c='green', rasterized=True)
        lo = min(0, min(np.percentile(o_flat[v], 1), np.percentile(m_flat[v], 1)))
        hi = max(np.percentile(o_flat[v], 99), np.percentile(m_flat[v], 99))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        stats = compute_stats(mod_2d, obs_2d)
        ax.text(0.05, 0.95, f"r={stats['r']:.3f}\nbias={stats['bias']:.3f}\n"
                f"RMSE={stats['rmse']:.3f}\nN={stats['n']}",
                transform=ax.transAxes, fontsize=8, va='top', fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.8))
        ax.set_xlim(lo, hi * 1.05); ax.set_ylim(lo, hi * 1.05)
    ax.set_xlabel(f'Observed ({units})')
    ax.set_ylabel(f'Model ({units})')
    ax.set_title('Scatter', fontsize=11, fontweight='bold')

    # Zonal profiles
    ax = fig.add_subplot(gs[1, 1])
    zm_obs = zonal_mean(obs_2d)
    zm_mod = zonal_mean(mod_2d)
    ax.plot(lats, zm_obs, 'k-', lw=2, label='Observed')
    ax.plot(lats, zm_mod, 'r-', lw=1.5, label='IBIS-JAX')
    ax.set_xlabel('Latitude')
    ax.set_ylabel(units)
    ax.set_title('Zonal Mean', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(90, -90)
    ax.grid(True, alpha=0.3)

    # Ratio (model/obs) by latitude band
    ax = fig.add_subplot(gs[1, 2])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where((zm_obs > 0) & np.isfinite(zm_obs) & np.isfinite(zm_mod),
                         zm_mod / zm_obs, np.nan)
    ax.plot(lats, ratio, 'b-', lw=1.5)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Model / Observed')
    ax.set_title('Zonal Ratio', fontsize=11, fontweight='bold')
    ax.set_xlim(90, -90)
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3)
    # Shade latitude bands
    for band, color, alpha in [((-10, 10), 'red', 0.1),
                                ((10, 30), 'orange', 0.08),
                                ((30, 60), 'blue', 0.08)]:
        ax.axvspan(band[1], band[0], color=color, alpha=alpha)
    ax.text(0, 2.8, 'Tropics', ha='center', fontsize=7, color='red')
    ax.text(20, 2.8, 'Subtrop', ha='center', fontsize=7, color='orange')
    ax.text(45, 2.8, 'Temperate', ha='center', fontsize=7, color='blue')

    fig.suptitle(f'{label}: Model vs Observations', fontsize=14, fontweight='bold')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved %s", outpath)


def plot_summary_table(all_stats, outpath):
    """Summary statistics table as a figure."""
    fig, ax = plt.subplots(figsize=(16, max(3, 1 + 0.6 * len(all_stats))))
    ax.axis('off')

    cols = ['Variable', 'Source', 'Units', 'Obs Mean', 'Model Mean',
            'r', 'Bias', 'RMSE', 'NRMSE', 'N pts']
    rows = []
    for name, info in all_stats.items():
        s = info['stats']
        rows.append([
            info['label'], info['source'], info['units'],
            f"{s['obs_mean']:.3f}", f"{s['mod_mean']:.3f}",
            f"{s['r']:.3f}", f"{s['bias']:.3f}",
            f"{s['rmse']:.3f}", f"{s['nrmse']:.2f}", f"{s['n']}"
        ])

    if not rows:
        return

    table = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    for j in range(len(cols)):
        table[0, j].set_facecolor('#343a40')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color-code r column (5) and NRMSE column (8)
    for i in range(len(rows)):
        r_val = float(rows[i][5])
        nrmse_val = float(rows[i][8]) if rows[i][8] != 'nan' else 999
        # r: green if > 0.7, yellow if > 0.5, red otherwise
        if r_val > 0.7:
            table[i+1, 5].set_facecolor('#d4edda')
        elif r_val > 0.5:
            table[i+1, 5].set_facecolor('#fff3cd')
        else:
            table[i+1, 5].set_facecolor('#f8d7da')
        # NRMSE: green if < 1.0, yellow if < 1.5, red otherwise
        if nrmse_val < 1.0:
            table[i+1, 8].set_facecolor('#d4edda')
        elif nrmse_val < 1.5:
            table[i+1, 8].set_facecolor('#fff3cd')
        else:
            table[i+1, 8].set_facecolor('#f8d7da')

    ax.set_title('Baseline Diagnostic Summary', fontsize=14, fontweight='bold', pad=20)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved %s", outpath)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="output/cal_v3")
    parser.add_argument("--output-dir", default="output/diagnostics_baseline")
    parser.add_argument("--years", default="1990-1999",
                        help="Year range, e.g. 1990-1999")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    y0, y1 = map(int, args.years.split('-'))
    years = range(y0, y1 + 1)

    logger.info("=== Baseline Diagnostic Evaluation ===")
    logger.info("Model dir: %s", args.model_dir)
    logger.info("Years: %d-%d", y0, y1)

    # Load
    logger.info("\nLoading observations...")
    obs = load_observations()

    logger.info("\nLoading model output...")
    model, raw = load_model(args.model_dir, years)

    # Compare each variable
    all_stats = {}
    matched = []
    for var_key in obs:
        if var_key not in model:
            logger.info("  SKIP %s: not in model output", var_key)
            continue
        matched.append(var_key)

    logger.info("\nMatched variables: %s", matched)

    for var_key in matched:
        obs_info = obs[var_key]
        obs_2d = obs_info['data'].reshape(360, 720) if obs_info['data'].size == 259200 else obs_info['data']
        mod_2d = model[var_key].reshape(360, 720) if model[var_key].size == 259200 else model[var_key]

        stats = compute_stats(mod_2d, obs_2d)
        all_stats[var_key] = {
            'stats': stats,
            'label': obs_info['label'],
            'source': obs_info['source'],
            'units': obs_info['units'],
        }

        logger.info("  %s: r=%.3f, bias=%.3f, RMSE=%.3f, NRMSE=%.2f",
                    obs_info['label'], stats['r'], stats['bias'],
                    stats['rmse'], stats['nrmse'])

        plot_comparison(mod_2d, obs_2d, obs_info['label'], obs_info['units'],
                        os.path.join(args.output_dir, f"compare_{var_key}.png"))

    # Summary table
    plot_summary_table(all_stats, os.path.join(args.output_dir, "summary_table.png"))

    # Print text summary
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE DIAGNOSTIC SUMMARY")
    logger.info("=" * 70)
    for k, v in all_stats.items():
        s = v['stats']
        grade = 'GOOD' if s['r'] > 0.7 and abs(s['nrmse']) < 1.0 else \
                'OK' if s['r'] > 0.5 else 'POOR'
        logger.info("  %-15s r=%.3f  bias=%+.3f  NRMSE=%.2f  [%s]",
                    v['label'], s['r'], s['bias'], s['nrmse'], grade)

    # Identify key issues
    logger.info("\n--- KEY ISSUES ---")
    for k, v in all_stats.items():
        s = v['stats']
        if s['r'] < 0.5:
            logger.info("  LOW CORRELATION: %s (r=%.3f)", v['label'], s['r'])
        if abs(s['bias']) > 0.3 * abs(s['obs_mean']):
            pct = 100 * s['bias'] / s['obs_mean']
            logger.info("  LARGE BIAS: %s (%+.0f%%)", v['label'], pct)
        if s['nrmse'] > 1.5:
            logger.info("  HIGH NRMSE: %s (%.2f)", v['label'], s['nrmse'])

    logger.info("\nDone! Results in %s/", args.output_dir)


if __name__ == "__main__":
    main()
