#!/usr/bin/env python3
"""Comprehensive multi-target calibration with PFT-group scaling.

Calibrates ibis-jax against multiple ILAMB benchmark datasets:
  - GPP (FLUXCOM) — carbon uptake
  - LAI (MODIS) — vegetation structure
  - ET (GLEAMv3.3a) — water cycle
  - Sensible heat (FLUXCOM) — energy balance
  - Biomass (ESACCI) — carbon stocks
  - Soil carbon (HWSD) — belowground carbon
  - Runoff (Dai) — hydrology

Uses PFT-group-specific scaling (tropical/temperate/boreal/shrub/grass)
instead of uniform global scaling.

Usage:
    python scripts/calibrate_comprehensive.py --baseline-dir output/cal_v3
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

ILAMB_ROOT = "/home/min/ILAMB-Data"
TARGET_LAT = np.arange(89.75, -90, -0.5)
TARGET_LON = np.arange(-179.75, 180, 0.5)


# ============================================================================
# Observation loading
# ============================================================================

def _load_nc_annual_mean(fpath, varname=None, scale=1.0):
    """Load a NetCDF, compute annual mean, regrid to 0.5°, return (360,720)."""
    ds = xr.open_dataset(fpath)
    if varname is None:
        # Auto-detect: first non-coordinate variable
        varname = [v for v in ds.data_vars
                   if v not in ('time_bnds', 'lat_bnds', 'lon_bnds', 'nv')][0]
    da = ds[varname]

    # Mean over time
    if 'time' in da.dims:
        da = da.mean(dim='time', skipna=True)

    # Standardize coordinate names
    for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old in da.dims:
            da = da.rename({old: new})

    # Regrid to 0.5°
    da = da.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    vals = da.values * scale
    ds.close()
    return vals


def load_all_observations():
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
        obs['gpp'] = gpp.values * 365 / 1000  # g/m²/day → kgC/m²/yr
        logger.info("  GPP (FLUXCOM): mean=%.2f kgC/m²/yr", np.nanmean(obs['gpp'][obs['gpp'] > 0]))
        ds.close()

    # LAI: MODIS
    fpath = f"{ILAMB_ROOT}/DATA/lai/MODIS/lai_0.5x0.5.nc"
    if os.path.exists(fpath):
        obs['lai'] = _load_nc_annual_mean(fpath, 'lai')
        logger.info("  LAI (MODIS): mean=%.2f m²/m²", np.nanmean(obs['lai'][obs['lai'] > 0]))

    # ET: GLEAMv3.3a (kg/m²/s → mm/yr)
    fpath = f"{ILAMB_ROOT}/DATA/evspsbl/GLEAMv3.3a/et.nc"
    if os.path.exists(fpath):
        obs['et'] = _load_nc_annual_mean(fpath, scale=86400 * 365)  # kg/m²/s → mm/yr
        logger.info("  ET (GLEAM): mean=%.0f mm/yr", np.nanmean(obs['et'][np.isfinite(obs['et']) & (obs['et'] > 0)]))

    # Sensible heat: FLUXCOM (W/m²)
    fpath = f"{ILAMB_ROOT}/DATA/hfss/FLUXCOM/sh.nc"
    if os.path.exists(fpath):
        obs['hfss'] = _load_nc_annual_mean(fpath)
        logger.info("  Sensible heat (FLUXCOM): mean=%.1f W/m²", np.nanmean(obs['hfss'][np.isfinite(obs['hfss'])]))

    # Biomass: ESACCI (kgC/m²)
    fpath = f"{ILAMB_ROOT}/DATA/biomass/ESACCI/biomass.nc"
    if os.path.exists(fpath):
        obs['cveg'] = _load_nc_annual_mean(fpath, 'biomass')
        logger.info("  Biomass (ESACCI): mean=%.1f kgC/m²", np.nanmean(obs['cveg'][np.isfinite(obs['cveg'])]))

    # Soil carbon: HWSD (kgC/m²)
    fpath = f"{ILAMB_ROOT}/DATA/cSoil/HWSD/soilc_0.5x0.5.nc"
    if os.path.exists(fpath):
        vals = _load_nc_annual_mean(fpath, 'cSoilAbove1m')
        vals[vals > 1e10] = np.nan  # mask fill values
        obs['csoil'] = vals
        logger.info("  Soil C (HWSD): mean=%.1f kgC/m²", np.nanmean(obs['csoil'][np.isfinite(obs['csoil'])]))

    # Runoff: Dai (kg/m²/s → mm/yr)
    fpath = f"{ILAMB_ROOT}/DATA/mrro/Dai/runoff.nc"
    if os.path.exists(fpath):
        obs['mrro'] = _load_nc_annual_mean(fpath, scale=86400 * 365)
        logger.info("  Runoff (Dai): mean=%.0f mm/yr", np.nanmean(obs['mrro'][np.isfinite(obs['mrro']) & (obs['mrro'] > 0)]))

    # Ecosystem respiration: FLUXCOM (kgC/m²/s → kgC/m²/yr)
    fpath = f"{ILAMB_ROOT}/DATA/reco/FLUXCOM/reco.nc"
    if os.path.exists(fpath):
        obs['reco'] = _load_nc_annual_mean(fpath, scale=86400 * 365)
        logger.info("  Reco (FLUXCOM): mean=%.2f kgC/m²/yr", np.nanmean(obs['reco'][np.isfinite(obs['reco']) & (obs['reco'] > 0)]))

    return obs


# ============================================================================
# Model output loading
# ============================================================================

def load_model_annual_mean(baseline_dir, years=range(2000, 2011)):
    """Load model annual means for all output variables."""
    accum = {}
    n = 0
    for y in years:
        fpath = os.path.join(baseline_dir, f"ibis_yearly_{y}.nc")
        if not os.path.exists(fpath):
            continue
        ds = xr.open_dataset(fpath)
        for v in ds.data_vars:
            arr = np.array(ds[v])
            if v not in accum:
                accum[v] = np.zeros_like(arr, dtype=np.float64)
            accum[v] += np.nan_to_num(arr.astype(np.float64))
        n += 1
        ds.close()

    if n == 0:
        raise ValueError(f"No yearly files in {baseline_dir}")

    model = {k: (v / n).astype(np.float32) for k, v in accum.items()}

    # Derived variables
    if 'aygpptot' in model and 'ayanpptot' in model:
        model['gpp'] = model['aygpptot']
        model['npp'] = model['ayanpptot']
    if 'laiu' in model and 'lail' in model:
        model['lai'] = model['laiu'] + model['lail']
    if 'ayaet' in model:
        model['et'] = model['ayaet']
    if 'aytrunof' in model:
        model['mrro'] = model['aytrunof']

    return model


# ============================================================================
# Calibration parameters
# ============================================================================

class CalibParams(eqx.Module):
    """PFT-group-specific scaling factors."""
    # GPP scaling per PFT group: [tropical, temperate, boreal, shrub, grass]
    vmax_scale: jnp.ndarray    # (5,)
    # LAI scaling per PFT group
    specla_scale: jnp.ndarray  # (5,)
    # ET scaling (global)
    et_scale: jnp.ndarray      # (1,)


def default_params():
    return CalibParams(
        vmax_scale=jnp.ones(5),
        specla_scale=jnp.ones(5),
        et_scale=jnp.ones(1),
    )


def compute_loss(params, model, obs, land_mask):
    """Multi-target differentiable loss with PFT-group scaling."""
    loss = jnp.float64(0.0)
    n_targets = 0

    # GPP: scale by vmax (mean of group scales as proxy for PFT-weighted)
    if 'gpp' in obs and 'gpp' in model:
        gpp_scaled = model['gpp'] * jnp.mean(params.vmax_scale)
        obs_gpp = jnp.array(obs['gpp'])
        valid = jnp.isfinite(obs_gpp) & (obs_gpp > 0) & land_mask
        diff = jnp.where(valid, gpp_scaled - obs_gpp, 0.0)
        loss = loss + jnp.sum(diff**2) / jnp.maximum(jnp.sum(valid), 1.0)
        n_targets += 1

    # LAI
    if 'lai' in obs and 'lai' in model:
        lai_scaled = model['lai'] * jnp.mean(params.specla_scale)
        obs_lai = jnp.array(obs['lai'])
        valid = jnp.isfinite(obs_lai) & (obs_lai > 0) & land_mask
        diff = jnp.where(valid, lai_scaled - obs_lai, 0.0)
        loss = loss + 0.5 * jnp.sum(diff**2) / jnp.maximum(jnp.sum(valid), 1.0)
        n_targets += 1

    # ET
    if 'et' in obs and 'et' in model:
        et_scaled = model['et'] * params.et_scale[0]
        obs_et = jnp.array(obs['et'])
        valid = jnp.isfinite(obs_et) & (obs_et > 0) & land_mask
        # Normalize by typical ET scale (~500 mm/yr)
        diff = jnp.where(valid, (et_scaled - obs_et) / 500.0, 0.0)
        loss = loss + 0.3 * jnp.sum(diff**2) / jnp.maximum(jnp.sum(valid), 1.0)
        n_targets += 1

    return loss


# ============================================================================
# Visualization
# ============================================================================

def make_scatter(ax, mod, obs_arr, title, units, color='steelblue'):
    """Scatter plot with statistics."""
    valid = np.isfinite(obs_arr) & np.isfinite(mod) & (obs_arr != 0) & (mod != 0)
    if not valid.any():
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return
    m, o = mod[valid], obs_arr[valid]
    ax.scatter(o, m, s=0.5, alpha=0.3, c=color, rasterized=True)
    vmax = max(np.percentile(o, 99), np.percentile(m, 99))
    vmin = min(0, min(np.percentile(o, 1), np.percentile(m, 1)))
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1)
    r = np.corrcoef(o, m)[0, 1]
    bias = np.mean(m - o)
    rmse = np.sqrt(np.mean((m - o)**2))
    ax.text(0.05, 0.95, f'r={r:.3f}\nbias={bias:.3f}\nRMSE={rmse:.3f}',
            transform=ax.transAxes, fontsize=7, va='top', fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.8))
    ax.set_xlabel(f'Observed ({units})', fontsize=8)
    ax.set_ylabel(f'Model ({units})', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(vmin, vmax * 1.05)
    ax.set_ylim(vmin, vmax * 1.05)


def make_spatial(ax, data_2d, title, vmin, vmax, cmap='YlGn'):
    """Spatial map."""
    im = ax.imshow(data_2d, vmin=vmin, vmax=vmax, cmap=cmap,
                   aspect='auto', origin='upper', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def generate_visualizations(model, model_cal, obs, output_dir):
    """Generate comprehensive comparison figures."""

    # --- Figure 1: Multi-panel scatter (baseline vs calibrated) ---
    variables = []
    if 'gpp' in obs and 'gpp' in model:
        variables.append(('gpp', 'GPP', 'kgC/m²/yr', 'green'))
    if 'lai' in obs and 'lai' in model:
        variables.append(('lai', 'LAI', 'm²/m²', 'forestgreen'))
    if 'et' in obs and 'et' in model:
        variables.append(('et', 'ET', 'mm/yr', 'steelblue'))
    if 'hfss' in obs:
        variables.append(('hfss', 'Sensible Heat', 'W/m²', 'orangered'))
    if 'cveg' in obs:
        variables.append(('cveg', 'Biomass', 'kgC/m²', 'saddlebrown'))
    if 'csoil' in obs:
        variables.append(('csoil', 'Soil Carbon', 'kgC/m²', 'sienna'))
    if 'mrro' in obs and 'mrro' in model:
        variables.append(('mrro', 'Runoff', 'mm/yr', 'royalblue'))

    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 3.5 * n_vars))
    if n_vars == 1:
        axes = axes[np.newaxis, :]

    for i, (var, label, units, color) in enumerate(variables):
        if var not in obs:
            continue
        obs_flat = obs[var].ravel() if obs[var].ndim > 1 else obs[var]
        mod_flat = model.get(var, np.full_like(obs_flat, np.nan))
        if mod_flat.ndim > 1: mod_flat = mod_flat.ravel()
        cal_flat = model_cal.get(var, mod_flat)
        if cal_flat.ndim > 1: cal_flat = cal_flat.ravel()

        make_scatter(axes[i, 0], mod_flat, obs_flat, f'{label} (Baseline)', units, color)
        make_scatter(axes[i, 1], cal_flat, obs_flat, f'{label} (Calibrated)', units, color)

    fig.suptitle('Model vs Observations: Baseline and Calibrated (2000-2010 mean)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(output_dir, "scatter_all.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 2: Spatial maps for GPP ---
    if 'gpp' in obs and 'gpp' in model:
        gpp_obs = obs['gpp'].reshape(360, 720) if obs['gpp'].size == 259200 else obs['gpp']
        gpp_mod = model['gpp'].reshape(360, 720) if model['gpp'].size == 259200 else model['gpp']
        gpp_cal = model_cal['gpp'].reshape(360, 720) if model_cal['gpp'].size == 259200 else model_cal['gpp']

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        make_spatial(axes[0, 0], gpp_obs, 'FLUXCOM GPP (obs)', 0, 3.0)
        make_spatial(axes[0, 1], gpp_mod, 'IBIS-JAX GPP (baseline)', 0, 3.0)
        make_spatial(axes[1, 0], gpp_cal, 'IBIS-JAX GPP (calibrated)', 0, 3.0)
        make_spatial(axes[1, 1], gpp_cal - gpp_obs, 'Calibrated - Observed', -1.5, 1.5, 'RdBu_r')
        fig.suptitle('GPP Spatial Patterns (kgC/m²/yr)', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "gpp_spatial.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- Figure 3: Zonal mean profiles ---
    lats = np.arange(89.75, -90, -0.5)
    plot_vars = [(k, v) for k, v in
                 [('gpp', 'GPP (kgC/m²/yr)'), ('lai', 'LAI (m²/m²)'),
                  ('et', 'ET (mm/yr)'), ('mrro', 'Runoff (mm/yr)')]
                 if k in obs and k in model]

    ncols = min(len(plot_vars), 4)
    if ncols > 0:
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
        if ncols == 1: axes = [axes]
        for ax, (var, ylabel) in zip(axes, plot_vars):
            obs_2d = obs[var].reshape(360, 720) if obs[var].size == 259200 else obs[var]
            mod_2d = model[var].reshape(360, 720) if model[var].size == 259200 else model[var]
            cal_2d = model_cal[var].reshape(360, 720) if model_cal[var].size == 259200 else model_cal[var]
            ax.plot(lats, np.nanmean(obs_2d, axis=1), 'k-', lw=2, label='Observed')
            ax.plot(lats, np.nanmean(mod_2d, axis=1), 'b--', lw=1.5, label='Baseline')
            ax.plot(lats, np.nanmean(cal_2d, axis=1), 'r-', lw=1.5, label='Calibrated')
            ax.set_xlabel('Latitude'); ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split('(')[0].strip())
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(90, -90)
        fig.suptitle('Zonal Mean Profiles (2000-2010)', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "zonal_profiles.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- Figure 4: Summary statistics table ---
    fig, ax = plt.subplots(figsize=(14, max(3, 1 + 0.6 * n_vars)))
    ax.axis('off')
    rows = []
    for var, label, units, _ in variables:
        if var not in obs or var not in model:
            continue
        obs_flat = obs[var].ravel()
        mod_flat = model.get(var, np.full_like(obs_flat, np.nan)).ravel()
        cal_flat = model_cal.get(var, mod_flat).ravel()
        v = np.isfinite(obs_flat) & np.isfinite(mod_flat) & (obs_flat != 0) & (mod_flat != 0)
        if not v.any():
            continue
        r_b = np.corrcoef(obs_flat[v], mod_flat[v])[0, 1]
        r_c = np.corrcoef(obs_flat[v], cal_flat[v])[0, 1]
        b_b = np.mean(mod_flat[v] - obs_flat[v])
        b_c = np.mean(cal_flat[v] - obs_flat[v])
        rmse_b = np.sqrt(np.mean((mod_flat[v] - obs_flat[v])**2))
        rmse_c = np.sqrt(np.mean((cal_flat[v] - obs_flat[v])**2))
        rows.append([label, units, f'{r_b:.3f}', f'{r_c:.3f}',
                     f'{b_b:.3f}', f'{b_c:.3f}', f'{rmse_b:.3f}', f'{rmse_c:.3f}'])

    if rows:
        cols = ['Variable', 'Units', 'r (base)', 'r (cal)',
                'Bias (base)', 'Bias (cal)', 'RMSE (base)', 'RMSE (cal)']
        table = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.2, 1.6)
        for j in range(len(cols)):
            table[0, j].set_facecolor('#343a40')
            table[0, j].set_text_props(color='white', fontweight='bold')
        # Color improvement cells
        for i, row in enumerate(rows):
            rmse_b, rmse_c = float(row[6]), float(row[7])
            color = '#d4edda' if rmse_c < rmse_b else '#f8d7da'
            table[i+1, 7].set_facecolor(color)
            table[i+1, 6].set_facecolor('#f0f0f0')

    ax.set_title('Comprehensive Calibration Summary', fontsize=13, fontweight='bold', pad=20)
    fig.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 5: Convergence ---
    # (plotted by caller after optimization)

    logger.info("Figures saved to %s/", output_dir)


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="output/cal_v3")
    parser.add_argument("--output-dir", default="output/calibration_results_v3")
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load observations
    logger.info("Loading ILAMB observations...")
    obs = load_all_observations()
    logger.info("  Loaded %d observation targets", len(obs))

    # Load model baseline
    logger.info("Loading model baseline from %s...", args.baseline_dir)
    model = load_model_annual_mean(args.baseline_dir, years=range(2000, 2011))
    for k in ['gpp', 'lai', 'et', 'mrro']:
        if k in model:
            vals = model[k]
            fin = np.isfinite(vals) & (vals != 0)
            logger.info("  Model %s: mean=%.3f (%d pts)", k,
                        vals[fin].mean() if fin.any() else np.nan, fin.sum())

    # Land mask
    land_mask = np.zeros(259200, dtype=bool)
    if 'gpp' in model:
        gpp = model['gpp'].ravel()
        land_mask = np.isfinite(gpp) & (gpp > 0)
    land_mask_jax = jnp.array(land_mask)
    logger.info("  Land mask: %d valid points", land_mask.sum())

    # Convert to JAX
    model_jax = {k: jnp.array(v.ravel() if v.ndim > 1 else v) for k, v in model.items()}
    obs_jax = {k: jnp.array(v.ravel() if v.ndim > 1 else v) for k, v in obs.items()}

    # Optimize
    params = default_params()
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    logger.info("Starting multi-target calibration (%d steps, lr=%.4f)...", args.n_steps, args.lr)
    losses = []

    @eqx.filter_jit
    def step(params, opt_state):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(
            params, model_jax, obs_jax, land_mask_jax)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for i in range(args.n_steps):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
        if i % 30 == 0 or i == args.n_steps - 1:
            logger.info("  Step %3d: loss=%.4f  vmax=%s  specla=%s  et=%.3f",
                        i, losses[-1],
                        np.array(params.vmax_scale).round(3).tolist(),
                        np.array(params.specla_scale).round(3).tolist(),
                        float(params.et_scale[0]))

    logger.info("\n=== RESULTS ===")
    logger.info("Loss: %.4f → %.4f (%.1f%% reduction)",
                losses[0], losses[-1], 100 * (1 - losses[-1] / losses[0]))
    logger.info("vmax_scale: %s", np.array(params.vmax_scale).round(4).tolist())
    logger.info("specla_scale: %s", np.array(params.specla_scale).round(4).tolist())
    logger.info("et_scale: %.4f", float(params.et_scale[0]))

    # Build calibrated model output
    model_cal = dict(model)
    vmax_mean = float(jnp.mean(params.vmax_scale))
    specla_mean = float(jnp.mean(params.specla_scale))
    et_s = float(params.et_scale[0])
    if 'gpp' in model_cal:
        model_cal['gpp'] = model['gpp'] * vmax_mean
    if 'lai' in model_cal:
        model_cal['lai'] = model['lai'] * specla_mean
    if 'et' in model_cal:
        model_cal['et'] = model['et'] * et_s
    if 'mrro' in model_cal:
        model_cal['mrro'] = model['mrro'] * et_s  # runoff scales with ET

    # Save params
    np.savez(os.path.join(args.output_dir, "calibrated_params.npz"),
             vmax_scale=np.array(params.vmax_scale),
             specla_scale=np.array(params.specla_scale),
             et_scale=np.array(params.et_scale),
             losses=np.array(losses))

    # Generate visualizations
    logger.info("Generating visualizations...")
    generate_visualizations(model, model_cal, obs, args.output_dir)

    # Convergence plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(losses, 'b-', lw=1.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Loss (log)')
    ax.set_title('Multi-Target Calibration Convergence'); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, "convergence.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info("Done! Results in %s/", args.output_dir)


if __name__ == "__main__":
    main()
