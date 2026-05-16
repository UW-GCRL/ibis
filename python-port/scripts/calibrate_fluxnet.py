#!/usr/bin/env python3
"""PFT-specific parameter calibration against FLUXNET eddy covariance sites.

Runs IBIS at ~150+ FLUXNET site locations simultaneously (npoi=N_sites),
compares model GPP, ET, Reco against tower observations, and optimizes
PFT-specific parameters using CMA-ES with PFT-grouped loss.

Usage:
    python scripts/calibrate_fluxnet.py --n-evals 400 --spinup-years 10
"""
import logging
import os
import sys
import glob
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import cma
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

from ibis.io.parameters import default_pft_params
from ibis.physics.stomata import StomataParams
from ibis.timestepping.loops_gswp3 import run_month_gswp3, _cast_tree_f32
from ibis.timestepping.calendar import is_leap_year
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.constants import PhysicalConstants
from ibis.io.physics_params import build_sim_params
from ibis.state.model_state import create_ibis_state
from ibis.dimensions import GridConfig

FLUXNET_DIR = "/shared/nas2/Min/FLUXNET2015/fluxnet_compact"
GSWP3_DIR = "/shared/nas2/Min/GSWP3"
SOIL_FILE = "/shared/nas2/Min/HWSD/soita.nc"

GSWP3_LAT = np.arange(89.75, -90, -0.5)   # 360, N→S
GSWP3_LON = np.arange(-179.75, 180, 0.5)  # 720, W→E

# IGBP → IBIS vegtype mapping
# Uses latitude to distinguish tropical vs temperate
IGBP_TO_VEGTYPE = {
    'EBF': lambda lat: 1 if abs(lat) < 25 else 3,    # tropical or temperate evergreen broadleaf
    'ENF': lambda lat: 6 if lat > 50 else 4,          # boreal or temperate conifer
    'DBF': lambda lat: 7 if lat > 50 else 5,          # boreal or temperate deciduous broadleaf
    'DNF': lambda lat: 7,                              # deciduous needleleaf → boreal deciduous
    'MF':  lambda lat: 8,                              # mixed forest
    'GRA': lambda lat: 10,                             # grassland
    'SAV': lambda lat: 9,                              # savanna
    'WSA': lambda lat: 9,                              # woody savanna
    'CRO': lambda lat: 10,                             # cropland → grassland proxy
    'OSH': lambda lat: 12,                             # open shrubland
    'CSH': lambda lat: 11,                             # closed shrubland
    'WET': lambda lat: 10,                             # wetland → grassland proxy
}

# IGBP → dominant IBIS PFT index (0-based) for parameter attribution
IGBP_TO_PFT = {
    'EBF': lambda lat: 0 if abs(lat) < 25 else 2,
    'ENF': lambda lat: 5 if lat > 50 else 3,
    'DBF': lambda lat: 6 if lat > 50 else 4,
    'DNF': lambda lat: 7,
    'MF':  lambda lat: 4,   # attribute to temperate broadleaf deciduous
    'GRA': lambda lat: 11 if abs(lat) > 30 else 10,  # C3 or C4
    'SAV': lambda lat: 10,  # C4 grass/shrub
    'WSA': lambda lat: 8,   # shrub
    'CRO': lambda lat: 11,  # C3 grass proxy
    'OSH': lambda lat: 9,   # deciduous shrub
    'CSH': lambda lat: 8,   # evergreen shrub
    'WET': lambda lat: 11,  # C3 grass proxy
}

PFT_NAMES = [
    "trop_bl_ever", "trop_bl_dec", "warm_bl_ever", "temp_con_ever",
    "temp_bl_dec", "boreal_con_ever", "boreal_bl_dec", "boreal_con_dec",
    "ever_shrub", "dec_shrub", "c4_grass", "c3_grass",
]

# PFT index → Ball-Berry group index
PFT_TO_BB = [0, 0, 0, 1, 0, 1, 0, 1, 2, 2, 3, 4]


# ============================================================================
# FLUXNET data loading
# ============================================================================

def load_fluxnet_sites():
    """Load site metadata and annual observations."""
    meta = pd.read_csv(f"{FLUXNET_DIR}/FLUXNET_siteinfo.csv", encoding='utf-8-sig')
    meta = meta.set_index('SITE_ID')

    yy_files = sorted(glob.glob(f"{FLUXNET_DIR}/FLX_*_YY_*.csv"))

    sites = []
    for f in yy_files:
        site_id = os.path.basename(f).split('_')[1]
        if site_id not in meta.index:
            continue
        row = meta.loc[site_id]
        igbp = row['IGBP']
        if igbp not in IGBP_TO_VEGTYPE:
            continue  # Skip WAT, SNO, BSV, URB

        lat = float(row['LOCATION_LAT'])
        lon = float(row['LOCATION_LONG'])
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue

        # Load annual observations
        df = pd.read_csv(f)
        gpp_vals = df['GPP_NT_VUT_REF'].values
        reco_vals = df['RECO_NT_VUT_REF'].values
        le_vals = df['LE_F_MDS'].values
        h_vals = df['H_F_MDS'].values

        # Mean over years, skip NaN
        gpp = np.nanmean(gpp_vals[np.isfinite(gpp_vals)]) if np.any(np.isfinite(gpp_vals)) else np.nan
        reco = np.nanmean(reco_vals[np.isfinite(reco_vals)]) if np.any(np.isfinite(reco_vals)) else np.nan
        le = np.nanmean(le_vals[np.isfinite(le_vals)]) if np.any(np.isfinite(le_vals)) else np.nan
        h = np.nanmean(h_vals[np.isfinite(h_vals)]) if np.any(np.isfinite(h_vals)) else np.nan

        if not np.isfinite(gpp) or gpp <= 0:
            continue  # Need at least GPP

        # Convert GPP from gC/m2/yr to kgC/m2/yr
        gpp_kgc = gpp / 1000.0
        reco_kgc = reco / 1000.0 if np.isfinite(reco) else np.nan
        # LE is in W/m2 → ET in mm/yr: ET = LE * 86400*365 / 2.5e6
        et_mm = le * 86400 * 365 / 2.5e6 if np.isfinite(le) else np.nan

        vegtype = IGBP_TO_VEGTYPE[igbp](lat)
        pft_idx = IGBP_TO_PFT[igbp](lat)

        sites.append({
            'site_id': site_id, 'lat': lat, 'lon': lon,
            'igbp': igbp, 'vegtype': vegtype, 'pft_idx': pft_idx,
            'gpp': gpp_kgc, 'reco': reco_kgc, 'et': et_mm,
            'le': le, 'h': h,
        })

    return pd.DataFrame(sites)


# ============================================================================
# Parameter encoding
# ============================================================================

def params_to_vector(vmax, specla, coefm, coefb):
    return np.concatenate([np.log(vmax), np.log(specla), np.log(coefm), np.log(coefb)])

def vector_to_params(x):
    return np.exp(x[0:12]), np.exp(x[12:24]), np.exp(x[24:29]), np.exp(x[29:34])

def default_vector():
    pft = default_pft_params()
    sp = StomataParams()
    return params_to_vector(np.array(sp.vmax_pft), np.array(pft.specla),
                            np.array(sp.coefm), np.array(sp.coefb))


# ============================================================================
# Forward model
# ============================================================================

def apply_params(base_sim_params, vmax, specla, coefm, coefb):
    return eqx.tree_at(
        lambda p: (p.stomata_params.vmax_pft,
                   p.stomata_params.coefm,
                   p.stomata_params.coefb),
        base_sim_params,
        (jnp.array(vmax, dtype=jnp.float32),
         jnp.array(coefm, dtype=jnp.float32),
         jnp.array(coefb, dtype=jnp.float32)),
    )


def run_year(state, consts, sim_params, reader, lat_idx, lon_idx, lat_rad,
             year, key, dtime=3600.0):
    leap = is_leap_year(year)
    for imonth in range(12):
        state, key, _ = run_month_gswp3(
            state, consts, sim_params, reader,
            lat_idx, lon_idx, lat_rad,
            year, imonth, key, dtime,
            leap=leap, nband=2, nsoilcall=1,
        )
    return state, key


def reset_annual(state, npoi):
    z = jnp.zeros((npoi,))
    z12 = jnp.zeros((npoi, 12))
    return eqx.tree_at(
        lambda s: (s.summary.aygpptot, s.summary.ayaet, s.summary.ayprcp,
                   s.summary.ayco2mic, s.summary.ayco2root,
                   s.summary.ayanpptot, s.summary.ayneetot,
                   s.summary.aytrunof, s.summary.aytdrain,
                   s.summary.aynpp),
        state,
        (z, z, z, z, z, z, z, z, z, z12),
    )


# ============================================================================
# Loss function with PFT grouping
# ============================================================================

def compute_pft_loss(gpp_mod, et_mod, reco_mod, sites_df):
    """Compute normalized MSE grouped by PFT."""
    loss = 0.0
    n_groups = 0

    for pft in range(12):
        mask = sites_df['pft_idx'].values == pft
        if mask.sum() < 2:
            continue

        # GPP loss
        obs_gpp = sites_df.loc[mask, 'gpp'].values
        mod_gpp = gpp_mod[mask]
        valid = np.isfinite(obs_gpp) & np.isfinite(mod_gpp) & (obs_gpp > 0.05)
        if valid.sum() >= 2:
            diff = (mod_gpp[valid] - obs_gpp[valid]) / np.maximum(obs_gpp[valid], 0.1)
            loss += np.mean(diff**2)

        # ET loss
        obs_et = sites_df.loc[mask, 'et'].values
        mod_et = et_mod[mask]
        valid = np.isfinite(obs_et) & np.isfinite(mod_et) & (obs_et > 10)
        if valid.sum() >= 2:
            diff = (mod_et[valid] - obs_et[valid]) / 500.0
            loss += 0.3 * np.mean(diff**2)

        # Reco loss
        obs_reco = sites_df.loc[mask, 'reco'].values
        mod_reco = reco_mod[mask]
        valid = np.isfinite(obs_reco) & np.isfinite(mod_reco) & (obs_reco > 0.05)
        if valid.sum() >= 2:
            diff = (mod_reco[valid] - obs_reco[valid]) / np.maximum(obs_reco[valid], 0.1)
            loss += 0.3 * np.mean(diff**2)

        n_groups += 1

    if n_groups == 0:
        return 50.0  # No valid comparisons → penalize
    return loss / n_groups


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-evals", type=int, default=400)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--spinup-years", type=int, default=10)
    parser.add_argument("--eval-year", type=int, default=2005)
    parser.add_argument("--output-dir", default="output/calibration_fluxnet")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load FLUXNET sites ---
    logger.info("Loading FLUXNET sites...")
    sites_df = load_fluxnet_sites()
    logger.info("  %d sites loaded", len(sites_df))
    logger.info("  PFT distribution:")
    for pft in range(12):
        n = (sites_df['pft_idx'] == pft).sum()
        if n > 0:
            logger.info("    PFT %2d (%s): %d sites", pft, PFT_NAMES[pft], n)

    # --- Map to GSWP3 grid ---
    npoi = len(sites_df)
    lat_idx = np.array([np.argmin(np.abs(GSWP3_LAT - lat))
                        for lat in sites_df['lat'].values])
    lon_idx = np.array([np.argmin(np.abs(GSWP3_LON - lon))
                        for lon in sites_df['lon'].values])
    lat_rad = np.deg2rad(GSWP3_LAT[lat_idx]).astype(np.float32)
    logger.info("  Mapped %d sites to GSWP3 grid", npoi)

    # --- Load soil ---
    logger.info("Loading soil properties...")
    import netCDF4 as nc
    ds = nc.Dataset(SOIL_FILE)
    sand_grid = np.array(ds.variables['sandpct'][:])
    clay_grid = np.array(ds.variables['claypct'][:])
    ds.close()
    sand = sand_grid[:, lat_idx, lon_idx].T
    clay = clay_grid[:, lat_idx, lon_idx].T
    sand = np.where(np.isfinite(sand), sand, 45.0)
    clay = np.where(np.isfinite(clay), clay, 25.0)

    from ibis.io.soil_init import compute_soil_properties
    soil_props = compute_soil_properties(sand.astype(np.float32), clay.astype(np.float32))

    # --- Load state from V7 spinup restart (already equilibrated) ---
    logger.info("Loading state from V7 spinup restart...")
    from ibis.io.restart import load_state

    # Load full global state
    _ds_lm = nc.Dataset("/shared/nas2/Min/HWSD/surta.nc")
    for _vn in ('landfrac', 'surta', 'mask', 'land'):
        if _vn in _ds_lm.variables:
            _lm = np.array(_ds_lm.variables[_vn][:], dtype=np.float32)
            break
    _ds_lm.close()
    full_land = np.where(_lm.ravel() > 0)[0]
    npoi_full = len(full_land)
    grid_full = GridConfig(npoi=npoi_full)
    template_full = create_ibis_state(grid_full)
    state_full, _ = load_state("restart/cal_v7/ibis_restart_1950.npz", template_full)

    # Map FLUXNET site grid indices to land-point indices in the full state
    # full_land contains the flattened (360*720) indices of land points
    # We need to find which position in full_land corresponds to each site's grid cell
    site_grid_flat = lat_idx * 720 + lon_idx  # flattened index for each site
    full_land_set = {idx: pos for pos, idx in enumerate(full_land)}

    sub_of_land = []
    valid_sites = []
    for i, gf in enumerate(site_grid_flat):
        if gf in full_land_set:
            sub_of_land.append(full_land_set[gf])
            valid_sites.append(i)
        else:
            logger.warning("  Site %s at grid[%d,%d] not in land mask — skipping",
                           sites_df.iloc[i]['site_id'], lat_idx[i], lon_idx[i])

    sub_of_land = np.array(sub_of_land)
    valid_sites = np.array(valid_sites)
    logger.info("  %d / %d sites matched to V7 land points", len(valid_sites), npoi)

    # Filter sites_df and indices to valid only
    sites_df = sites_df.iloc[valid_sites].reset_index(drop=True)
    lat_idx = lat_idx[valid_sites]
    lon_idx = lon_idx[valid_sites]
    lat_rad = lat_rad[valid_sites]
    npoi = len(sites_df)

    # Subsample state
    def _sub_leaf(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == npoi_full:
            return x[sub_of_land]
        return x
    state = jax.tree.map(_sub_leaf, state_full)
    state = _cast_tree_f32(state)
    state = jax.tree.map(
        lambda x: jnp.nan_to_num(x, nan=0.0) if isinstance(x, jnp.ndarray)
                  and jnp.issubdtype(x.dtype, jnp.floating) else x, state)
    del state_full, template_full

    state0 = state
    consts = PhysicalConstants()
    logger.info("State loaded: %d sites, LAI range [%.1f, %.1f]",
                npoi, float(jnp.min(state.vegetation.lai)),
                float(jnp.max(state.vegetation.lai)))

    # --- Setup ---
    base_sim_params = build_sim_params(npoi, lat_rad)
    reader = GSWP3Reader(GSWP3_DIR)
    key = jax.random.PRNGKey(42)

    # --- Physical bounds ---
    vmax_lo = np.array([40, 40, 25, 15, 20, 15, 20, 15, 15, 15, 8, 15]) * 1e-6
    vmax_hi = np.array([100, 100, 70, 50, 60, 40, 50, 45, 45, 45, 30, 45]) * 1e-6
    specla_lo = np.array([10, 10, 10, 5, 12, 5, 12, 8, 5, 10, 10, 10])
    specla_hi = np.array([30, 30, 30, 15, 35, 15, 35, 25, 18, 30, 25, 25])
    coefm_lo = np.array([6, 3, 5, 2, 5])
    coefm_hi = np.array([15, 10, 14, 8, 14])
    coefb_lo = np.array([0.005, 0.005, 0.005, 0.02, 0.005])
    coefb_hi = np.array([0.03, 0.03, 0.03, 0.08, 0.03])

    # --- Objective ---
    eval_count = [0]
    best_loss = [float('inf')]
    history = []

    def objective(x):
        vmax, specla, coefm, coefb = vector_to_params(x)

        if np.any(vmax < vmax_lo * 0.5) or np.any(vmax > vmax_hi * 2.0):
            return 100.0
        if np.any(specla < specla_lo * 0.5) or np.any(specla > specla_hi * 2.0):
            return 100.0

        sim_params = apply_params(base_sim_params, vmax, specla, coefm, coefb)
        state = state0

        try:
            # Spinup
            for yr_offset in range(args.spinup_years):
                forcing_year = 1901 + (yr_offset % 10)
                state = reset_annual(state, npoi)
                state, k = run_year(state, consts, sim_params, reader,
                                    lat_idx, lon_idx, lat_rad, forcing_year, key)

            # Evaluation year
            state = reset_annual(state, npoi)
            state, k = run_year(state, consts, sim_params, reader,
                                lat_idx, lon_idx, lat_rad, args.eval_year, key)

            gpp = np.array(state.summary.aygpptot)
            et = np.array(state.summary.ayaet)
            reco = np.array(state.summary.ayco2mic)

            # Debug: log model output stats on first eval
            if eval_count[0] == 0:
                fin_g = np.isfinite(gpp) & (gpp > 0)
                fin_e = np.isfinite(et) & (et > 0)
                logger.info("    Model GPP: %d valid, mean=%.3f", fin_g.sum(),
                            gpp[fin_g].mean() if fin_g.any() else 0)
                logger.info("    Model ET:  %d valid, mean=%.1f", fin_e.sum(),
                            et[fin_e].mean() if fin_e.any() else 0)

            loss = compute_pft_loss(gpp, et, reco, sites_df)

            # Regularization
            reg = 0.05 * (
                np.sum(np.maximum(0, np.log(vmax_lo / vmax))**2) +
                np.sum(np.maximum(0, np.log(vmax / vmax_hi))**2) +
                np.sum(np.maximum(0, np.log(specla_lo / specla))**2) +
                np.sum(np.maximum(0, np.log(specla / specla_hi))**2)
            )
            loss += reg

            if not np.isfinite(loss):
                loss = 50.0
        except Exception as e:
            logger.warning("  Eval %d failed: %s", eval_count[0], str(e)[:100])
            loss = 50.0

        eval_count[0] += 1
        history.append(loss)
        if loss < best_loss[0]:
            best_loss[0] = loss
            logger.info("  Eval %3d: loss=%.4f (BEST)  vmax_trop=%.1fµmol  "
                        "specla_trop=%.1f  coefm_bl=%.2f",
                        eval_count[0], loss, vmax[0]*1e6, specla[0], coefm[0])
        elif eval_count[0] % 10 == 0:
            logger.info("  Eval %3d: loss=%.4f  (best=%.4f)",
                        eval_count[0], loss, best_loss[0])

        return loss

    # --- Run CMA-ES ---
    x0 = default_vector()
    logger.info("\nStarting CMA-ES (%d evals, sigma=%.2f, %dyr spinup)...",
                args.n_evals, args.sigma, args.spinup_years)

    es = cma.CMAEvolutionStrategy(x0, args.sigma, {
        'maxfevals': args.n_evals,
        'verb_disp': 0,
        'bounds': [params_to_vector(vmax_lo * 0.5, specla_lo * 0.5,
                                    coefm_lo * 0.5, coefb_lo * 0.3),
                   params_to_vector(vmax_hi * 2.0, specla_hi * 2.0,
                                    coefm_hi * 2.0, coefb_hi * 3.0)],
    })

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

    # --- Results ---
    best_x = es.result.xbest
    vmax, specla, coefm, coefb = vector_to_params(best_x)

    logger.info("\n" + "=" * 60)
    logger.info("CMA-ES FLUXNET CALIBRATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Best loss: %.4f (initial: %.4f)", es.result.fbest, history[0])
    logger.info("Evaluations: %d", eval_count[0])

    logger.info("\nOptimized Vmax (µmol/m²/s):")
    for i, name in enumerate(PFT_NAMES):
        logger.info("  %-20s  %6.1f", name, vmax[i] * 1e6)
    logger.info("\nOptimized Specla (m²/kgC):")
    for i, name in enumerate(PFT_NAMES):
        logger.info("  %-20s  %5.1f", name, specla[i])
    logger.info("\nOptimized coefm:")
    BB = ["broadleaf", "conifer", "shrub", "c4_grass", "c3_grass"]
    for i, name in enumerate(BB):
        logger.info("  %-15s  %5.2f", name, coefm[i])
    logger.info("\nOptimized coefb:")
    for i, name in enumerate(BB):
        logger.info("  %-15s  %.4f", name, coefb[i])

    # --- Save ---
    np.savez(os.path.join(args.output_dir, "optimized_params.npz"),
             vmax=vmax, specla=specla, coefm=coefm, coefb=coefb,
             history=np.array(history), best_x=best_x)
    sites_df.to_csv(os.path.join(args.output_dir, "sites_used.csv"), index=False)

    # --- Run final eval and plot ---
    logger.info("\nRunning final evaluation with best params...")
    sim_params = apply_params(base_sim_params, vmax, specla, coefm, coefb)
    state = state0
    for yr_offset in range(args.spinup_years):
        forcing_year = 1901 + (yr_offset % 10)
        state = reset_annual(state, npoi)
        state, k = run_year(state, consts, sim_params, reader,
                            lat_idx, lon_idx, lat_rad, forcing_year, key)
    state = reset_annual(state, npoi)
    state, k = run_year(state, consts, sim_params, reader,
                        lat_idx, lon_idx, lat_rad, args.eval_year, key)

    gpp_mod = np.array(state.summary.aygpptot)
    et_mod = np.array(state.summary.ayaet)
    reco_mod = np.array(state.summary.ayco2mic)

    # --- Scatter plots by PFT ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pft_colors = plt.cm.tab20(np.linspace(0, 1, 12))

    for ax, (var, mod, obs_col, units) in zip(axes, [
        ('GPP', gpp_mod, 'gpp', 'kgC/m²/yr'),
        ('ET', et_mod, 'et', 'mm/yr'),
        ('Reco', reco_mod, 'reco', 'kgC/m²/yr'),
    ]):
        obs = sites_df[obs_col].values
        for pft in range(12):
            mask = (sites_df['pft_idx'].values == pft) & np.isfinite(obs) & np.isfinite(mod)
            if mask.sum() == 0:
                continue
            ax.scatter(obs[mask], mod[mask], s=20, alpha=0.7,
                       c=[pft_colors[pft]], label=PFT_NAMES[pft] if mask.sum() > 0 else None)
        valid = np.isfinite(obs) & np.isfinite(mod) & (obs != 0) & (mod != 0)
        if valid.sum() > 5:
            r = np.corrcoef(obs[valid], mod[valid])[0, 1]
            bias = np.mean(mod[valid] - obs[valid])
            lo = min(0, min(obs[valid].min(), mod[valid].min()))
            hi = max(obs[valid].max(), mod[valid].max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
            ax.text(0.05, 0.95, f'r={r:.3f}\nbias={bias:.3f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(facecolor='white', alpha=0.8))
        ax.set_xlabel(f'Observed {var} ({units})')
        ax.set_ylabel(f'Model {var} ({units})')
        ax.set_title(f'{var}: Model vs FLUXNET', fontweight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('FLUXNET Site-Level Calibration Results', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(args.output_dir, "scatter_by_pft.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Convergence
    h = np.array(history)
    h_clean = h[h < 100]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(h_clean)), h_clean, 'b.', ms=3, alpha=0.4)
    ax.plot(range(len(h_clean)), np.minimum.accumulate(h_clean), 'r-', lw=2, label='Best')
    ax.set_xlabel('Evaluation'); ax.set_ylabel('Loss')
    ax.set_title('FLUXNET CMA-ES Convergence'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, "convergence.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Per-PFT statistics table
    logger.info("\n=== Per-PFT Statistics ===")
    logger.info("%-20s %5s %8s %8s %8s", "PFT", "N", "GPP_r", "ET_r", "Reco_r")
    for pft in range(12):
        mask = sites_df['pft_idx'].values == pft
        n = mask.sum()
        if n < 2:
            continue
        def _r(mod, obs_col):
            o = sites_df.loc[mask, obs_col].values
            m = mod[mask]
            v = np.isfinite(o) & np.isfinite(m) & (o != 0) & (m != 0)
            return np.corrcoef(o[v], m[v])[0, 1] if v.sum() >= 3 else np.nan
        logger.info("%-20s %5d %8.3f %8.3f %8.3f",
                    PFT_NAMES[pft], n,
                    _r(gpp_mod, 'gpp'), _r(et_mod, 'et'), _r(reco_mod, 'reco'))

    logger.info("\nResults saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()
