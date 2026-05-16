#!/usr/bin/env python3
"""Differentiable PFT parameter calibration using JAX autodiff.

Runs the actual IBIS physics forward model with trainable PFT parameters,
computes loss against ILAMB observations, and uses gradient-based optimization.

Strategy:
  - Load a spun-up restart state
  - Subsample ~5000 land points (fits in 24 GB GPU for backprop)
  - Run 1 year of simulation with trainable params
  - Compare annual GPP, LAI, ET against gridded observations
  - Autodiff → Adam optimizer

Trainable parameters (per PFT, 12 PFTs):
  - vmax_pft[12]:  max carboxylation rate (mol CO2/m2/s)
  - specla[12]:    specific leaf area (m2/kgC)
  - coefm[5]:      Ball-Berry slope [broadleaf, conifer, shrub, C4, C3]
  - coefb[5]:      Ball-Berry intercept

Usage:
    python scripts/calibrate_diffable.py
"""
import logging
import os
import sys
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

from ibis.io.parameters import default_pft_params, PFTParams
from ibis.physics.stomata import StomataParams
from ibis.timestepping.loops import SimParams
from ibis.timestepping.loops_gswp3 import run_month_gswp3, _cast_tree_f32
from ibis.timestepping.calendar import is_leap_year
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.constants import PhysicalConstants
from ibis.io.physics_params import build_sim_params
from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.dimensions import GridConfig

ILAMB_ROOT = "/home/min/ILAMB-Data"
TARGET_LAT = np.arange(89.75, -90, -0.5)
TARGET_LON = np.arange(-179.75, 180, 0.5)


# ============================================================================
# Trainable parameter container
# ============================================================================

class TrainableParams(eqx.Module):
    """Differentiable PFT parameters for calibration."""
    # Log-space to ensure positivity: actual = exp(log_param)
    log_vmax: jnp.ndarray      # (12,) log of vmax_pft
    log_specla: jnp.ndarray    # (12,) log of specla
    log_coefm: jnp.ndarray     # (5,)  log of Ball-Berry slope
    log_coefb: jnp.ndarray     # (5,)  log of Ball-Berry intercept


def init_trainable() -> TrainableParams:
    """Initialize trainable params from current defaults."""
    pft = default_pft_params()
    sp = StomataParams()
    return TrainableParams(
        log_vmax=jnp.log(jnp.array(sp.vmax_pft)),
        log_specla=jnp.log(pft.specla),
        log_coefm=jnp.log(jnp.array(sp.coefm)),
        log_coefb=jnp.log(jnp.array(sp.coefb)),
    )


def apply_trainable(base_params: SimParams, pft_params: PFTParams,
                     tp: TrainableParams) -> tuple[SimParams, PFTParams]:
    """Inject trainable parameters into the simulation parameter objects.

    StomataParams is now an eqx.Module with jnp.ndarray fields,
    so we can use tree_at to swap in the trainable values directly.
    """
    vmax = jnp.exp(tp.log_vmax)
    specla = jnp.exp(tp.log_specla)
    coefm = jnp.exp(tp.log_coefm)
    coefb = jnp.exp(tp.log_coefb)

    # Update StomataParams via tree_at (differentiable)
    new_sim = eqx.tree_at(
        lambda p: (p.stomata_params.vmax_pft,
                   p.stomata_params.coefm,
                   p.stomata_params.coefb),
        base_params,
        (vmax, coefm, coefb),
    )

    # Update PFTParams (specla controls LAI = cbiol * specla)
    new_pft = eqx.tree_at(lambda p: p.specla, pft_params, specla)

    return new_sim, new_pft


# ============================================================================
# Observation loading (subsampled)
# ============================================================================

def load_obs_subsampled(land_idx):
    """Load observations at the subsampled land point indices."""
    obs = {}

    # GPP
    fpath = f"{ILAMB_ROOT}/DATA/gpp/FLUXCOM/gpp.nc"
    if os.path.exists(fpath):
        ds = xr.open_dataset(fpath)
        gpp = ds['gpp'].where(ds['gpp'] >= 0).mean(dim='time', skipna=True)
        for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
            if old in gpp.dims:
                gpp = gpp.rename({old: new})
        gpp = gpp.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
        gpp_2d = gpp.values * 365.0 / 1000.0  # g/m2/day → kgC/m2/yr
        obs['gpp'] = gpp_2d.ravel()[land_idx]
        ds.close()

    # LAI
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

    # ET
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
        obs['et'] = et.values.ravel()[land_idx] * 86400 * 365  # kg/m2/s → mm/yr
        ds.close()

    # Reco
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
# Forward model: run 1 year, return annual outputs
# ============================================================================

def preload_year_forcing(reader, latindex, lonindex, lat_rad, year):
    """Pre-load all 12 months of GSWP3 forcing (Python side, no JAX tracing)."""
    from ibis.io.gswp3_forcing import make_monthly_daily_weather
    from ibis.timestepping.calendar import julian_day

    leap = is_leap_year(year)
    months_xs = []
    for imonth in range(12):  # 0-based
        gswp3_month = reader.read_month(year, imonth, latindex, lonindex)
        ndays = gswp3_month['ndays']
        xstore_np = np.zeros((len(latindex), 3), dtype=np.float32)
        daily_list, rlds_list = make_monthly_daily_weather(
            gswp3_month, lat_rad, year, imonth, xstore_np,
        )
        td     = jnp.stack([dw.td     for dw in daily_list])
        tmax   = jnp.stack([dw.tmax   for dw in daily_list])
        tmin   = jnp.stack([dw.tmin   for dw in daily_list])
        cloud  = jnp.stack([dw.cloud  for dw in daily_list])
        precip = jnp.stack([dw.precip for dw in daily_list])
        psurf  = jnp.stack([dw.psurf  for dw in daily_list])
        qd     = jnp.stack([dw.qd     for dw in daily_list])
        ud     = jnp.stack([dw.ud     for dw in daily_list])
        iwet   = jnp.stack([dw.iwet   for dw in daily_list])
        rlds   = jnp.stack(rlds_list)
        jdays  = jnp.array(
            [julian_day(imonth + 1, d, leap) for d in range(ndays)],
            dtype=jnp.int32)
        months_xs.append((td, tmax, tmin, cloud, precip, psurf, qd, ud, iwet, rlds, jdays))
    return months_xs, leap


def run_year_differentiable(state, consts, sim_params, months_xs, leap, key):
    """Run 1 year using pre-loaded forcing. Fully differentiable w.r.t. sim_params.

    Uses jax.checkpoint per month to limit memory during backprop.
    """
    from ibis.timestepping.loops_gswp3 import _run_month_scan

    ndaypy = 366 if leap else 365

    for xs in months_xs:
        state = _cast_tree_f32(state)

        @jax.checkpoint
        def _month_step(state, key, xs_m):
            return _run_month_scan(state, key, consts, sim_params, xs_m,
                                   1, ndaypy, 3600.0, 2)

        state, key = _month_step(state, key, xs)

    gpp = state.summary.aygpptot
    lai = state.vegetation.lai[:, 0] + state.vegetation.lai[:, 1]
    et = state.summary.ayaet
    reco = state.summary.ayco2mic
    return state, gpp, lai, et, reco


# ============================================================================
# Loss function
# ============================================================================

def compute_loss(tp, state, consts, base_params, pft_params,
                 months_xs, leap, key, obs, n_months=1):
    """Differentiable loss: run forward model, compare with observations.

    n_months: number of months to simulate (1-12). Shorter = more stable gradients.
    """
    sim_params, new_pft = apply_trainable(base_params, pft_params, tp)

    from ibis.timestepping.loops_gswp3 import _run_month_scan

    ndaypy = 366 if leap else 365
    for i in range(min(n_months, len(months_xs))):
        xs = months_xs[i]
        state = _cast_tree_f32(state)
        state, key = _run_month_scan(state, key, consts, sim_params, xs,
                                     1, ndaypy, 3600.0, 2)

    # Use monthly accumulators scaled to annual
    scale = 12.0 / max(n_months, 1)
    gpp = state.summary.aygpptot * scale
    lai = state.vegetation.lai[:, 0] + state.vegetation.lai[:, 1]
    et = state.summary.ayaet * scale
    reco = state.summary.ayco2mic * scale

    loss = jnp.float32(0.0)
    n_targets = 0

    # GPP loss (normalized MSE)
    if 'gpp' in obs:
        obs_gpp = jnp.array(obs['gpp'], dtype=jnp.float32)
        valid = jnp.isfinite(obs_gpp) & (obs_gpp > 0.05) & jnp.isfinite(gpp) & (gpp > 0)
        diff = jnp.where(valid, (gpp - obs_gpp) / jnp.maximum(obs_gpp, 0.1), 0.0)
        loss = loss + jnp.sum(diff**2) / jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
        n_targets += 1

    # LAI loss
    if 'lai' in obs:
        obs_lai = jnp.array(obs['lai'], dtype=jnp.float32)
        valid = jnp.isfinite(obs_lai) & (obs_lai > 0.1) & jnp.isfinite(lai)
        diff = jnp.where(valid, (lai - obs_lai) / jnp.maximum(obs_lai, 0.1), 0.0)
        loss = loss + 0.5 * jnp.sum(diff**2) / jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
        n_targets += 1

    # ET loss (normalized by typical scale)
    if 'et' in obs:
        obs_et = jnp.array(obs['et'], dtype=jnp.float32)
        valid = jnp.isfinite(obs_et) & (obs_et > 10) & jnp.isfinite(et)
        diff = jnp.where(valid, (et - obs_et) / 500.0, 0.0)
        loss = loss + 0.3 * jnp.sum(diff**2) / jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
        n_targets += 1

    # Reco loss
    if 'reco' in obs:
        obs_reco = jnp.array(obs['reco'], dtype=jnp.float32)
        valid = jnp.isfinite(obs_reco) & (obs_reco > 0.05) & jnp.isfinite(reco)
        diff = jnp.where(valid, (reco - obs_reco) / jnp.maximum(obs_reco, 0.1), 0.0)
        loss = loss + 0.3 * jnp.sum(diff**2) / jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
        n_targets += 1

    # Regularization: penalize large deviations from defaults
    tp0 = init_trainable()
    reg = 0.01 * (
        jnp.sum((tp.log_vmax - tp0.log_vmax)**2) +
        jnp.sum((tp.log_specla - tp0.log_specla)**2) +
        jnp.sum((tp.log_coefm - tp0.log_coefm)**2) +
        jnp.sum((tp.log_coefb - tp0.log_coefb)**2)
    )

    return loss + reg


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", default="restart/cal_v7/ibis_restart_1950.npz")
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--year", type=int, default=2000)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output-dir", default="output/calibration_diffable")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Setup land mask and subsample ---
    import netCDF4 as nc
    ds = nc.Dataset("/shared/nas2/Min/HWSD/surta.nc")
    for vname in ('landfrac', 'surta', 'mask', 'land'):
        if vname in ds.variables:
            land = np.array(ds.variables[vname][:], dtype=np.float32)
            break
    ds.close()
    land_2d = land.ravel()

    # Full land indices
    full_land_idx = np.where(land_2d > 0)[0]
    npoi_full = len(full_land_idx)
    logger.info("Full grid: %d land points", npoi_full)

    # Filter to points that have valid (non-NaN) output in V7 run
    # Use GPP from yearly output as validity indicator
    try:
        ds_check = xr.open_dataset("output/cal_v7/ibis_yearly_2000.nc")
        gpp_check = ds_check['aygpptot'].values.ravel()[full_land_idx]
        valid_mask = np.isfinite(gpp_check) & (gpp_check > 0)
        ds_check.close()
        valid_land_of_full = np.where(valid_mask)[0]
        logger.info("  %d / %d land points have valid GPP output",
                    len(valid_land_of_full), npoi_full)
    except Exception:
        valid_land_of_full = np.arange(npoi_full)
        logger.warning("  Could not filter by valid output, using all points")

    # Subsample from valid points only
    step = max(1, len(valid_land_of_full) // args.n_points)
    sub_of_valid = np.arange(0, len(valid_land_of_full), step)[:args.n_points]
    sub_of_land = valid_land_of_full[sub_of_valid]
    sub_land_idx = full_land_idx[sub_of_land]
    npoi = len(sub_land_idx)
    logger.info("Subsampled to %d land points (step=%d)", npoi, step)

    # Lat/lon indices
    lat_idx = sub_land_idx // 720
    lon_idx = sub_land_idx % 720
    lat_rad = np.deg2rad(TARGET_LAT[lat_idx]).astype(np.float32)

    # --- Load observations at subsampled points ---
    logger.info("Loading observations...")
    obs = load_obs_subsampled(sub_land_idx)
    for k, v in obs.items():
        valid = np.isfinite(v) & (v != 0)
        logger.info("  %s: %d valid, mean=%.3f", k, valid.sum(),
                    np.nanmean(v[valid]) if valid.any() else np.nan)

    # --- Load restart state (subsampled) ---
    logger.info("Loading restart state...")
    grid = GridConfig(npoi=npoi)
    template = create_ibis_state(grid)

    # Load full restart, then subsample
    grid_full = GridConfig(npoi=npoi_full)
    template_full = create_ibis_state(grid_full)
    state_full, _meta = load_state(args.restart, template_full)

    # Subsample state: extract sub_of_land indices from every (npoi,) array
    def _sub_leaf(x):
        if isinstance(x, jnp.ndarray):
            if x.ndim >= 1 and x.shape[0] == npoi_full:
                return x[sub_of_land]
        return x

    state = jax.tree.map(_sub_leaf, state_full)
    state = _cast_tree_f32(state)
    # Replace any NaN in state with 0 to prevent NaN gradient propagation
    def _nan_to_zero(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x
    state = jax.tree.map(_nan_to_zero, state)
    del state_full, template_full
    logger.info("State subsampled to %d points", npoi)

    # --- Build simulation parameters ---
    consts = PhysicalConstants()
    sim_params = build_sim_params(npoi, lat_rad)
    pft_params = default_pft_params()

    # --- GSWP3 reader ---
    reader = GSWP3Reader("/shared/nas2/Min/GSWP3")

    # --- Reset annual accumulators ---
    z = jnp.zeros((npoi,))
    z12 = jnp.zeros((npoi, 12))
    state = eqx.tree_at(
        lambda s: (s.summary.aygpptot, s.summary.ayaet, s.summary.ayprcp,
                   s.summary.ayco2mic, s.summary.ayco2root,
                   s.summary.ayanpptot, s.summary.ayneetot,
                   s.summary.aytrunof, s.summary.aytdrain,
                   s.summary.aynpp),
        state,
        (z, z, z, z, z, z, z, z, z, z12),
    )

    # --- Initialize trainable params ---
    tp = init_trainable()
    logger.info("Initial trainable params:")
    logger.info("  vmax: %s", np.exp(np.array(tp.log_vmax)).round(8).tolist())
    logger.info("  specla: %s", np.exp(np.array(tp.log_specla)).round(2).tolist())
    logger.info("  coefm: %s", np.exp(np.array(tp.log_coefm)).round(3).tolist())
    logger.info("  coefb: %s", np.exp(np.array(tp.log_coefb)).round(4).tolist())

    # --- Pre-load forcing (Python side, outside JAX tracing) ---
    logger.info("Pre-loading GSWP3 forcing for year %d...", args.year)
    months_xs, leap = preload_year_forcing(
        reader, lat_idx, lon_idx, lat_rad, args.year)
    logger.info("  Loaded %d months of forcing", len(months_xs))

    # --- Optimizer ---
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(tp, eqx.is_array))

    # --- Optimization loop ---
    key = jax.random.PRNGKey(42)

    logger.info("\nStarting gradient-based calibration (%d steps, lr=%.4f)...",
                args.n_steps, args.lr)

    loss_fn = partial(compute_loss,
                      state=state, consts=consts, base_params=sim_params,
                      pft_params=pft_params,
                      months_xs=months_xs, leap=leap,
                      key=key, obs=obs, n_months=1)

    losses = []
    for step in range(args.n_steps):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(tp)
        # Replace NaN gradients with 0 (non-differentiable paths)
        grads = jax.tree.map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0) if isinstance(g, jnp.ndarray) else g,
            grads)
        # Clip gradients to prevent instability
        grads = jax.tree.map(
            lambda g: jnp.clip(g, -1.0, 1.0) if isinstance(g, jnp.ndarray) else g,
            grads)
        updates, opt_state = optimizer.update(grads, opt_state, tp)
        tp = eqx.apply_updates(tp, updates)
        losses.append(float(loss))

        if step % 5 == 0 or step == args.n_steps - 1:
            vmax = np.exp(np.array(tp.log_vmax))
            specla = np.exp(np.array(tp.log_specla))
            coefm = np.exp(np.array(tp.log_coefm))
            logger.info("  Step %3d: loss=%.4f  vmax_trop=%.1fµmol  specla_trop=%.1f  "
                        "coefm_bl=%.2f",
                        step, losses[-1],
                        vmax[0] * 1e6, specla[0], coefm[0])

    # --- Results ---
    logger.info("\n=== OPTIMIZED PARAMETERS ===")
    vmax = np.exp(np.array(tp.log_vmax))
    specla = np.exp(np.array(tp.log_specla))
    coefm = np.exp(np.array(tp.log_coefm))
    coefb = np.exp(np.array(tp.log_coefb))
    PFT_NAMES = [
        "trop_bl_ever", "trop_bl_dec", "warm_bl_ever", "temp_con_ever",
        "temp_bl_dec", "boreal_con_ever", "boreal_bl_dec", "boreal_con_dec",
        "ever_shrub", "dec_shrub", "c4_grass", "c3_grass",
    ]
    BB_GROUPS = ["broadleaf", "conifer", "shrub", "c4_grass", "c3_grass"]

    logger.info("\nVmax (µmol/m²/s):")
    for i, name in enumerate(PFT_NAMES):
        logger.info("  %-20s  %.1f", name, vmax[i] * 1e6)

    logger.info("\nSpecla (m²/kgC):")
    for i, name in enumerate(PFT_NAMES):
        logger.info("  %-20s  %.1f", name, specla[i])

    logger.info("\nBall-Berry coefm:")
    for i, name in enumerate(BB_GROUPS):
        logger.info("  %-15s  %.2f", name, coefm[i])

    logger.info("\nBall-Berry coefb:")
    for i, name in enumerate(BB_GROUPS):
        logger.info("  %-15s  %.4f", name, coefb[i])

    # Save
    np.savez(os.path.join(args.output_dir, "optimized_params.npz"),
             vmax=vmax, specla=specla, coefm=coefm, coefb=coefb,
             losses=np.array(losses))

    # Convergence plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, 'b-', lw=1.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title('Differentiable Calibration Convergence')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, "convergence.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info("\nLoss: %.4f → %.4f (%.1f%% reduction)",
                losses[0], losses[-1],
                100 * (1 - losses[-1] / max(losses[0], 1e-10)))
    logger.info("Results saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()
