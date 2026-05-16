#!/usr/bin/env python3
"""PFT parameter optimization using CMA-ES (gradient-free).

Runs full IBIS forward simulations with candidate parameter vectors,
evaluates against ILAMB observations, and uses CMA-ES to find optimal
PFT-specific parameters.

Strategy:
  - Use V7 spinup restart as starting state
  - Run 1 year with candidate parameters (full physics, GPP-ET coupled)
  - Compare GPP, LAI, ET, Reco against gridded observations
  - CMA-ES iterates to minimize multi-target loss

Trainable parameters (34 total):
  - vmax_pft[12]:  max carboxylation rate
  - specla[12]:    specific leaf area
  - coefm[5]:      Ball-Berry slope
  - coefb[5]:      Ball-Berry intercept

Usage:
    python scripts/calibrate_cmaes.py --n-evals 200
"""
import logging
import os
import sys
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import cma
import xarray as xr
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
from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.dimensions import GridConfig
import equinox as eqx

ILAMB_ROOT = "/home/min/ILAMB-Data"
TARGET_LAT = np.arange(89.75, -90, -0.5)
TARGET_LON = np.arange(-179.75, 180, 0.5)

PFT_NAMES = [
    "trop_bl_ever", "trop_bl_dec", "warm_bl_ever", "temp_con_ever",
    "temp_bl_dec", "boreal_con_ever", "boreal_bl_dec", "boreal_con_dec",
    "ever_shrub", "dec_shrub", "c4_grass", "c3_grass",
]
BB_GROUPS = ["broadleaf", "conifer", "shrub", "c4_grass", "c3_grass"]


# ============================================================================
# Parameter encoding (log-space for positivity)
# ============================================================================

def params_to_vector(vmax, specla, coefm, coefb):
    """Encode parameters as a 1D numpy vector in log-space."""
    return np.concatenate([
        np.log(vmax),      # 12
        np.log(specla),    # 12
        np.log(coefm),     # 5
        np.log(coefb),     # 5
    ])


def vector_to_params(x):
    """Decode 1D vector back to parameter arrays."""
    vmax = np.exp(x[0:12])
    specla = np.exp(x[12:24])
    coefm = np.exp(x[24:29])
    coefb = np.exp(x[29:34])
    return vmax, specla, coefm, coefb


def default_vector():
    """Get default parameter vector."""
    pft = default_pft_params()
    sp = StomataParams()
    return params_to_vector(
        np.array(sp.vmax_pft), np.array(pft.specla),
        np.array(sp.coefm), np.array(sp.coefb),
    )


# ============================================================================
# Forward model evaluation
# ============================================================================

def run_year_eval(state, consts, sim_params, reader, latindex, lonindex,
                  lat_rad, year, key, dtime=3600.0):
    """Run 1 year and return annual outputs as numpy arrays."""
    leap = is_leap_year(year)
    for imonth in range(12):
        state, key, _ = run_month_gswp3(
            state, consts, sim_params, reader,
            latindex, lonindex, lat_rad,
            year, imonth, key, dtime,
            leap=leap, nband=2, nsoilcall=1,
        )
    gpp = np.array(state.summary.aygpptot)
    lai = np.array(state.vegetation.lai[:, 0] + state.vegetation.lai[:, 1])
    et = np.array(state.summary.ayaet)
    reco = np.array(state.summary.ayco2mic)
    return gpp, lai, et, reco


def apply_params_to_sim(base_sim_params, vmax, specla, coefm, coefb):
    """Create new SimParams with updated stomata parameters."""
    new_sim = eqx.tree_at(
        lambda p: (p.stomata_params.vmax_pft,
                   p.stomata_params.coefm,
                   p.stomata_params.coefb),
        base_sim_params,
        (jnp.array(vmax, dtype=jnp.float32),
         jnp.array(coefm, dtype=jnp.float32),
         jnp.array(coefb, dtype=jnp.float32)),
    )
    return new_sim


# ============================================================================
# Loss function
# ============================================================================

def compute_loss_numpy(gpp, lai, et, reco, obs):
    """Multi-target normalized MSE loss."""
    loss = 0.0
    n = 0

    if 'gpp' in obs:
        o = obs['gpp']
        v = np.isfinite(o) & (o > 0.05) & np.isfinite(gpp) & (gpp > 0)
        if v.sum() > 5:
            diff = (gpp[v] - o[v]) / np.maximum(o[v], 0.1)
            loss += np.mean(diff**2)
            n += 1

    if 'lai' in obs:
        o = obs['lai']
        v = np.isfinite(o) & (o > 0.1) & np.isfinite(lai)
        if v.sum() > 5:
            diff = (lai[v] - o[v]) / np.maximum(o[v], 0.1)
            loss += 0.5 * np.mean(diff**2)
            n += 1

    if 'et' in obs:
        o = obs['et']
        v = np.isfinite(o) & (o > 10) & np.isfinite(et)
        if v.sum() > 5:
            diff = (et[v] - o[v]) / 500.0
            loss += 0.3 * np.mean(diff**2)
            n += 1

    if 'reco' in obs:
        o = obs['reco']
        v = np.isfinite(o) & (o > 0.05) & np.isfinite(reco)
        if v.sum() > 5:
            diff = (reco[v] - o[v]) / np.maximum(o[v], 0.1)
            loss += 0.3 * np.mean(diff**2)
            n += 1

    return loss


# ============================================================================
# Observation loading
# ============================================================================

def load_obs_at_indices(land_idx):
    """Load observations at specified grid indices."""
    obs = {}

    fpath = f"{ILAMB_ROOT}/DATA/gpp/FLUXCOM/gpp.nc"
    if os.path.exists(fpath):
        ds = xr.open_dataset(fpath)
        gpp = ds['gpp'].where(ds['gpp'] >= 0).mean(dim='time', skipna=True)
        for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
            if old in gpp.dims:
                gpp = gpp.rename({old: new})
        gpp = gpp.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
        obs['gpp'] = gpp.values.ravel()[land_idx] * 365.0 / 1000.0
        ds.close()

    fpath = f"{ILAMB_ROOT}/DATA/lai/MODIS/lai_0.5x0.5.nc"
    if os.path.exists(fpath):
        ds = xr.open_dataset(fpath)
        lai = ds['lai'].mean(dim='time', skipna=True)
        for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
            if old in lai.dims:
                lai = lai.rename({old: new})
        lai = lai.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
        obs['lai'] = lai.values.ravel()[land_idx]
        ds.close()

    fpath = f"{ILAMB_ROOT}/DATA/evspsbl/GLEAMv3.3a/et.nc"
    if os.path.exists(fpath):
        ds = xr.open_dataset(fpath, decode_times=False)
        et = ds['et'].mean(dim='time', skipna=True)
        for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
            if old in et.dims:
                et = et.rename({old: new})
        if not np.issubdtype(et.dtype, np.floating):
            et = et.astype(np.float64)
        et = et.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
        obs['et'] = et.values.ravel()[land_idx] * 86400 * 365
        ds.close()

    fpath = f"{ILAMB_ROOT}/DATA/reco/FLUXCOM/reco.nc"
    if os.path.exists(fpath):
        ds = xr.open_dataset(fpath)
        reco = ds['reco'].where(ds['reco'] >= 0).mean(dim='time', skipna=True)
        for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
            if old in reco.dims:
                reco = reco.rename({old: new})
        reco = reco.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
        obs['reco'] = reco.values.ravel()[land_idx] * 365.0 / 1000.0
        ds.close()

    return obs


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", default="restart/cal_v7/ibis_restart_1950.npz")
    parser.add_argument("--n-points", type=int, default=2000,
                        help="Number of subsampled land points")
    parser.add_argument("--n-evals", type=int, default=200,
                        help="Max function evaluations for CMA-ES")
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="Initial step size in log-space")
    parser.add_argument("--year", type=int, default=2000)
    parser.add_argument("--output-dir", default="output/calibration_cmaes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Land mask + subsample ---
    import netCDF4 as nc
    ds = nc.Dataset("/shared/nas2/Min/HWSD/surta.nc")
    for vname in ('landfrac', 'surta', 'mask', 'land'):
        if vname in ds.variables:
            land = np.array(ds.variables[vname][:], dtype=np.float32)
            break
    ds.close()

    full_land_idx = np.where(land.ravel() > 0)[0]
    npoi_full = len(full_land_idx)

    # Filter to valid points
    try:
        ds_check = xr.open_dataset("output/cal_v7/ibis_yearly_2000.nc")
        gpp_check = ds_check['aygpptot'].values.ravel()[full_land_idx]
        valid_mask = np.isfinite(gpp_check) & (gpp_check > 0)
        ds_check.close()
        valid_land = np.where(valid_mask)[0]
    except Exception:
        valid_land = np.arange(npoi_full)

    logger.info("Valid land points: %d / %d", len(valid_land), npoi_full)

    step = max(1, len(valid_land) // args.n_points)
    sub_of_valid = np.arange(0, len(valid_land), step)[:args.n_points]
    sub_of_land = valid_land[sub_of_valid]
    sub_land_idx = full_land_idx[sub_of_land]
    npoi = len(sub_land_idx)
    logger.info("Subsampled to %d land points", npoi)

    lat_idx = sub_land_idx // 720
    lon_idx = sub_land_idx % 720
    lat_rad = np.deg2rad(TARGET_LAT[lat_idx]).astype(np.float32)

    # --- Observations ---
    logger.info("Loading observations...")
    obs = load_obs_at_indices(sub_land_idx)
    for k, v in obs.items():
        valid = np.isfinite(v) & (v != 0)
        logger.info("  %s: %d valid, mean=%.3f", k, valid.sum(),
                    np.nanmean(v[valid]) if valid.any() else np.nan)

    # --- Load restart ---
    logger.info("Loading restart state...")
    grid_full = GridConfig(npoi=npoi_full)
    template_full = create_ibis_state(grid_full)
    state_full, _ = load_state(args.restart, template_full)

    def _sub_leaf(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == npoi_full:
            return x[sub_of_land]
        return x

    state0 = jax.tree.map(_sub_leaf, state_full)
    state0 = _cast_tree_f32(state0)
    state0 = jax.tree.map(
        lambda x: jnp.nan_to_num(x, nan=0.0) if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating) else x,
        state0)
    del state_full, template_full
    logger.info("State ready (%d points)", npoi)

    # --- Build base params ---
    consts = PhysicalConstants()
    base_sim_params = build_sim_params(npoi, lat_rad)
    reader = GSWP3Reader("/shared/nas2/Min/GSWP3")
    key = jax.random.PRNGKey(42)

    # --- Reset accumulators helper ---
    z = jnp.zeros((npoi,))
    z12 = jnp.zeros((npoi, 12))

    def reset_accumulators(st):
        return eqx.tree_at(
            lambda s: (s.summary.aygpptot, s.summary.ayaet, s.summary.ayprcp,
                       s.summary.ayco2mic, s.summary.ayco2root,
                       s.summary.ayanpptot, s.summary.ayneetot,
                       s.summary.aytrunof, s.summary.aytdrain,
                       s.summary.aynpp),
            st,
            (z, z, z, z, z, z, z, z, z, z12),
        )

    # --- Objective function ---
    eval_count = [0]
    best_loss = [float('inf')]
    history = []

    # Physically motivated bounds per PFT (from Kattge 2009, TRY, Bonan 2019)
    # vmax: [trop_bl_e, trop_bl_d, warm_bl_e, temp_con, temp_bl_d, bor_con,
    #         bor_bl_d, bor_con_d, ev_shrub, dec_shrub, c4_grass, c3_grass]
    vmax_lo = np.array([40, 40, 25, 15, 20, 15, 20, 15, 15, 15, 8, 15]) * 1e-6
    vmax_hi = np.array([100, 100, 70, 50, 60, 40, 50, 45, 45, 45, 30, 45]) * 1e-6
    specla_lo = np.array([10, 10, 10, 5, 12, 5, 12, 8, 5, 10, 10, 10])
    specla_hi = np.array([30, 30, 30, 15, 35, 15, 35, 25, 18, 30, 25, 25])
    coefm_lo = np.array([6, 3, 5, 2, 5])
    coefm_hi = np.array([15, 10, 14, 8, 14])
    coefb_lo = np.array([0.005, 0.005, 0.005, 0.02, 0.005])
    coefb_hi = np.array([0.03, 0.03, 0.03, 0.08, 0.03])

    def objective(x):
        vmax, specla, coefm, coefb = vector_to_params(x)

        # Strict physical bounds
        if np.any(vmax < vmax_lo * 0.5) or np.any(vmax > vmax_hi * 2.0):
            return 100.0
        if np.any(specla < specla_lo * 0.5) or np.any(specla > specla_hi * 2.0):
            return 100.0
        if np.any(coefm < coefm_lo * 0.5) or np.any(coefm > coefm_hi * 2.0):
            return 100.0
        if np.any(coefb < coefb_lo * 0.3) or np.any(coefb > coefb_hi * 3.0):
            return 100.0

        sim_params = apply_params_to_sim(base_sim_params, vmax, specla, coefm, coefb)
        state = reset_accumulators(state0)

        try:
            gpp, lai, et, reco = run_year_eval(
                state, consts, sim_params, reader,
                lat_idx, lon_idx, lat_rad,
                args.year, key,
            )
            loss = compute_loss_numpy(gpp, lai, et, reco, obs)
            if not np.isfinite(loss):
                loss = 50.0

            # Physical regularization: soft penalty for leaving literature ranges
            reg = 0.0
            reg += 0.1 * np.sum(np.maximum(0, np.log(vmax_lo / vmax))**2)
            reg += 0.1 * np.sum(np.maximum(0, np.log(vmax / vmax_hi))**2)
            reg += 0.1 * np.sum(np.maximum(0, np.log(specla_lo / specla))**2)
            reg += 0.1 * np.sum(np.maximum(0, np.log(specla / specla_hi))**2)
            reg += 0.1 * np.sum(np.maximum(0, np.log(coefm_lo / coefm))**2)
            reg += 0.1 * np.sum(np.maximum(0, np.log(coefm / coefm_hi))**2)
            loss += reg
        except Exception as e:
            logger.warning("  Eval %d failed: %s", eval_count[0], e)
            loss = 50.0

        eval_count[0] += 1
        history.append(loss)
        if loss < best_loss[0]:
            best_loss[0] = loss
            logger.info("  Eval %3d: loss=%.4f (NEW BEST)  vmax_trop=%.1fµmol  "
                        "specla_trop=%.1f  coefm_bl=%.2f",
                        eval_count[0], loss, vmax[0]*1e6, specla[0], coefm[0])
        elif eval_count[0] % 10 == 0:
            logger.info("  Eval %3d: loss=%.4f  (best=%.4f)",
                        eval_count[0], loss, best_loss[0])

        return loss

    # --- Run CMA-ES ---
    x0 = default_vector()
    logger.info("\nStarting CMA-ES optimization (max %d evals, sigma=%.2f)...",
                args.n_evals, args.sigma)
    logger.info("Initial params (34 total): %d vmax + %d specla + %d coefm + %d coefb",
                12, 12, 5, 5)

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

    result = es.result
    best_x = result.xbest
    vmax, specla, coefm, coefb = vector_to_params(best_x)

    # --- Print results ---
    logger.info("\n" + "=" * 60)
    logger.info("CMA-ES OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Best loss: %.4f (initial: %.4f)", result.fbest, history[0])
    logger.info("Evaluations: %d", eval_count[0])

    logger.info("\nOptimized Vmax (µmol/m²/s):")
    for i, name in enumerate(PFT_NAMES):
        logger.info("  %-20s  %6.1f", name, vmax[i] * 1e6)

    logger.info("\nOptimized Specla (m²/kgC):")
    for i, name in enumerate(PFT_NAMES):
        logger.info("  %-20s  %5.1f", name, specla[i])

    logger.info("\nOptimized Ball-Berry coefm:")
    for i, name in enumerate(BB_GROUPS):
        logger.info("  %-15s  %5.2f", name, coefm[i])

    logger.info("\nOptimized Ball-Berry coefb:")
    for i, name in enumerate(BB_GROUPS):
        logger.info("  %-15s  %.4f", name, coefb[i])

    # --- Save ---
    np.savez(os.path.join(args.output_dir, "optimized_params.npz"),
             vmax=vmax, specla=specla, coefm=coefm, coefb=coefb,
             history=np.array(history), best_x=best_x)

    # Convergence plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, 'b.', ms=3, alpha=0.5)
    ax.plot(np.minimum.accumulate(history), 'r-', lw=2, label='Best')
    ax.set_xlabel('Evaluation'); ax.set_ylabel('Loss')
    ax.set_title('CMA-ES Calibration Convergence')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, "convergence.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info("\nResults saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()
