#!/usr/bin/env python3
"""Run IBIS with optimized FLUXNET params and create per-site daily plots.

For each FLUXNET site and each variable (GPP, ET, Reco):
  - Scatter plot: daily model vs daily observed
  - Time series: daily model and observed overlaid

Usage:
    python scripts/plot_fluxnet_results.py [--spinup-years 10] [--eval-year 2005]
"""
import logging
import os
import sys
import glob as globmod
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

from scripts.calibrate_fluxnet import (
    load_fluxnet_sites, apply_params, reset_annual,
    PFT_NAMES, GSWP3_LAT, GSWP3_LON, GSWP3_DIR, SOIL_FILE,
    FLUXNET_DIR, IGBP_TO_PFT,
)
from ibis.io.parameters import default_pft_params
from ibis.physics.stomata import StomataParams
from ibis.timestepping.loops_gswp3 import run_month_gswp3, _cast_tree_f32
from ibis.timestepping.calendar import is_leap_year
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.constants import PhysicalConstants
from ibis.io.physics_params import build_sim_params
from ibis.state.model_state import create_ibis_state
from ibis.dimensions import GridConfig

CAL_DIR = "output/calibration_fluxnet"


def load_optimized_params():
    d = np.load(os.path.join(CAL_DIR, "optimized_params.npz"))
    return d['vmax'], d['specla'], d['coefm'], d['coefb']


def setup_sites_and_state():
    """Load sites, soil, and subsample global state — mirrors calibrate_fluxnet.py."""
    import netCDF4 as nc

    sites_df = load_fluxnet_sites()
    npoi = len(sites_df)

    lat_idx = np.array([np.argmin(np.abs(GSWP3_LAT - lat))
                        for lat in sites_df['lat'].values])
    lon_idx = np.array([np.argmin(np.abs(GSWP3_LON - lon))
                        for lon in sites_df['lon'].values])
    lat_rad = np.deg2rad(GSWP3_LAT[lat_idx]).astype(np.float32)

    _ds_lm = nc.Dataset("/shared/nas2/Min/HWSD/surta.nc")
    for _vn in ('landfrac', 'surta', 'mask', 'land'):
        if _vn in _ds_lm.variables:
            _lm = np.array(_ds_lm.variables[_vn][:], dtype=np.float32)
            break
    _ds_lm.close()
    full_land = np.where(_lm.ravel() > 0)[0]
    npoi_full = len(full_land)

    from ibis.io.restart import load_state
    grid_full = GridConfig(npoi=npoi_full)
    template_full = create_ibis_state(grid_full)
    state_full, _ = load_state("restart/cal_v7/ibis_restart_1950.npz", template_full)

    site_grid_flat = lat_idx * 720 + lon_idx
    full_land_set = {idx: pos for pos, idx in enumerate(full_land)}

    sub_of_land = []
    valid_sites = []
    for i, gf in enumerate(site_grid_flat):
        if gf in full_land_set:
            sub_of_land.append(full_land_set[gf])
            valid_sites.append(i)

    sub_of_land = np.array(sub_of_land)
    valid_sites = np.array(valid_sites)

    sites_df = sites_df.iloc[valid_sites].reset_index(drop=True)
    lat_idx = lat_idx[valid_sites]
    lon_idx = lon_idx[valid_sites]
    lat_rad = lat_rad[valid_sites]
    npoi = len(sites_df)

    def _sub_leaf(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == npoi_full:
            return x[sub_of_land]
        return x
    state = jax.tree.map(_sub_leaf, state_full)
    state = _cast_tree_f32(state)
    state = jax.tree.map(
        lambda x: jnp.nan_to_num(x, nan=0.0) if isinstance(x, jnp.ndarray)
                  and jnp.issubdtype(x.dtype, jnp.floating) else x, state)

    return sites_df, state, lat_idx, lon_idx, lat_rad, npoi


def run_year_daily(state, consts, sim_params, reader, lat_idx, lon_idx,
                   lat_rad, year, key, dtime=3600.0, nsoilcall=1):
    """Run one year and collect daily GPP/ET/Reco for all sites.

    Returns (state, key, daily_gpp, daily_et, daily_rh) where daily arrays
    are shape (ndays_year, npoi).
    """
    leap = is_leap_year(year)
    daily_gpp_list = []
    daily_aet_list = []
    daily_rh_list = []

    for imonth in range(12):
        state, key, daily_diag = run_month_gswp3(
            state, consts, sim_params, reader,
            lat_idx, lon_idx, lat_rad,
            year, imonth, key, dtime,
            leap=leap, nband=2, nsoilcall=nsoilcall,
        )
        # daily_diag is (daily_gpp, daily_aet, daily_rh), each (ndays_month, npoi)
        dgpp, daet, drh = daily_diag
        daily_gpp_list.append(np.array(dgpp))
        daily_aet_list.append(np.array(daet))
        daily_rh_list.append(np.array(drh))

    daily_gpp = np.concatenate(daily_gpp_list, axis=0)  # (ndays, npoi)
    daily_aet = np.concatenate(daily_aet_list, axis=0)
    daily_rh = np.concatenate(daily_rh_list, axis=0)
    return state, key, daily_gpp, daily_aet, daily_rh


def load_fluxnet_daily(site_id, year):
    """Load daily FLUXNET observations for a site and year.

    Returns DataFrame with columns: date, gpp, reco, et (or NaN if missing).
    GPP/Reco: gC/m2/day, ET: mm/day.
    """
    pattern = f"{FLUXNET_DIR}/FLX_{site_id}_FLUXNET2015_FULLSET_DD_*.csv"
    files = globmod.glob(pattern)
    if not files:
        return None

    df = pd.read_csv(files[0])
    df['date'] = pd.to_datetime(df['TIMESTAMP'])
    df = df[df['date'].dt.year == year].copy()
    if len(df) == 0:
        return None

    # GPP_NT_VUT_REF: umol CO2/m2/s → gC/m2/day: × 12e-6 × 86400
    conv_c = 12.0e-6 * 86400.0
    df['gpp'] = df['GPP_NT_VUT_REF'].where(df['GPP_NT_VUT_REF'] > -9000) * conv_c
    df['reco'] = df['RECO_NT_VUT_REF'].where(df['RECO_NT_VUT_REF'] > -9000) * conv_c
    # LE_F_MDS: W/m2 → mm/day: × 86400 / 2.5e6
    df['et'] = df['LE_F_MDS'].where(df['LE_F_MDS'] > -9000) * 86400.0 / 2.5e6

    return df[['date', 'gpp', 'reco', 'et']].reset_index(drop=True)


def plot_site(site_id, igbp, pft_idx, dates, obs_df,
              mod_gpp, mod_et, mod_rh, output_dir):
    """Create 2x3 figure: top row = time series, bottom row = scatter."""
    variables = [
        ('GPP', mod_gpp, 'gpp', 'gC/m\u00b2/day'),
        ('ET', mod_et, 'et', 'mm/day'),
        ('Reco', mod_rh, 'reco', 'gC/m\u00b2/day'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f'{site_id}  ({igbp}, PFT={PFT_NAMES[pft_idx]})',
                 fontsize=14, fontweight='bold')

    for col, (var_name, mod_daily, obs_col, units) in enumerate(variables):
        ax_ts = axes[0, col]
        ax_sc = axes[1, col]

        # Get observations aligned to model dates
        if obs_df is not None and obs_col in obs_df.columns:
            obs_dates = obs_df['date'].values
            obs_vals = obs_df[obs_col].values
        else:
            obs_dates = dates
            obs_vals = np.full(len(dates), np.nan)

        # Align: model has ndays entries starting Jan 1
        ndays_mod = len(mod_daily)
        mod_dates = dates[:ndays_mod]

        # Match obs to model dates
        if obs_df is not None and len(obs_df) > 0:
            obs_aligned = np.full(ndays_mod, np.nan)
            obs_date_set = {pd.Timestamp(d): v for d, v in zip(obs_dates, obs_vals)}
            for i, d in enumerate(mod_dates):
                ts = pd.Timestamp(d)
                if ts in obs_date_set:
                    obs_aligned[i] = obs_date_set[ts]
        else:
            obs_aligned = np.full(ndays_mod, np.nan)

        valid = np.isfinite(obs_aligned) & np.isfinite(mod_daily[:ndays_mod])

        # --- Time series ---
        ax_ts.plot(mod_dates, mod_daily[:ndays_mod], 'b-', lw=0.8, alpha=0.8, label='Model')
        if np.any(np.isfinite(obs_aligned)):
            ax_ts.plot(mod_dates, obs_aligned, 'r-', lw=0.8, alpha=0.8, label='FLUXNET')
        ax_ts.set_ylabel(f'{var_name} ({units})')
        ax_ts.set_title(f'{var_name}', fontweight='bold')
        ax_ts.legend(fontsize=8, loc='upper right')
        ax_ts.xaxis.set_major_locator(MonthLocator())
        ax_ts.xaxis.set_major_formatter(DateFormatter('%b'))
        ax_ts.grid(True, alpha=0.2)

        # --- Scatter ---
        if valid.sum() > 5:
            o = obs_aligned[valid]
            m = mod_daily[:ndays_mod][valid]
            r = np.corrcoef(o, m)[0, 1]
            bias = np.mean(m - o)
            rmse = np.sqrt(np.mean((m - o) ** 2))

            ax_sc.scatter(o, m, s=8, alpha=0.4, c='steelblue', edgecolors='none')
            lo = min(0, min(o.min(), m.min()))
            hi = max(o.max(), m.max()) * 1.05
            ax_sc.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
            ax_sc.set_xlim(lo, hi)
            ax_sc.set_ylim(lo, hi)
            ax_sc.set_aspect('equal', adjustable='box')
            ax_sc.text(0.05, 0.95,
                       f'r={r:.3f}\nbias={bias:.2f}\nRMSE={rmse:.2f}\nN={valid.sum()}',
                       transform=ax_sc.transAxes, fontsize=8, va='top',
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
        else:
            ax_sc.text(0.5, 0.5, f'No obs\n(N={valid.sum()})',
                       transform=ax_sc.transAxes, ha='center', va='center', fontsize=12)

        ax_sc.set_xlabel(f'Obs {var_name} ({units})')
        ax_sc.set_ylabel(f'Model {var_name} ({units})')
        ax_sc.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join(output_dir, f"{site_id}.png")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return outpath


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spinup-years", type=int, default=10)
    parser.add_argument("--eval-year", type=int, default=2005)
    parser.add_argument("--output-dir", default="output/calibration_fluxnet/sites")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading optimized params...")
    vmax, specla, coefm, coefb = load_optimized_params()

    logger.info("Setting up sites and state...")
    sites_df, state0, lat_idx, lon_idx, lat_rad, npoi = setup_sites_and_state()
    logger.info("  %d sites ready", npoi)

    consts = PhysicalConstants()
    base_sim_params = build_sim_params(npoi, lat_rad)
    sim_params = apply_params(base_sim_params, vmax, specla, coefm, coefb)
    reader = GSWP3Reader(GSWP3_DIR)
    key = jax.random.PRNGKey(42)

    # --- Spinup ---
    logger.info("Running %dyr spinup...", args.spinup_years)
    state = state0
    for yr_offset in range(args.spinup_years):
        forcing_year = 1901 + (yr_offset % 10)
        state = reset_annual(state, npoi)
        state, key, _, _, _ = run_year_daily(state, consts, sim_params, reader,
                                              lat_idx, lon_idx, lat_rad,
                                              forcing_year, key)
        logger.info("  Spinup year %d/%d done", yr_offset + 1, args.spinup_years)

    # --- Eval year with daily output ---
    logger.info("Running eval year %d with daily diagnostics...", args.eval_year)
    state = reset_annual(state, npoi)
    state, key, daily_gpp, daily_aet, daily_rh = run_year_daily(
        state, consts, sim_params, reader,
        lat_idx, lon_idx, lat_rad, args.eval_year, key,
    )
    # daily arrays: (ndays, npoi) in kgC/m2/day for GPP/Rh, mm/day for AET
    # Convert GPP/Rh from kgC to gC for daily comparison with FLUXNET
    daily_gpp_gc = daily_gpp * 1000.0  # kgC → gC
    daily_rh_gc = daily_rh * 1000.0

    ndays = daily_gpp.shape[0]
    year = args.eval_year
    dates = pd.date_range(f'{year}-01-01', periods=ndays, freq='D')
    logger.info("  Got %d days of daily output for %d sites", ndays, npoi)

    # --- Save daily predictions ---
    np.savez(os.path.join(args.output_dir, "daily_predictions.npz"),
             daily_gpp=daily_gpp, daily_aet=daily_aet, daily_rh=daily_rh,
             eval_year=args.eval_year)

    # --- Per-site plots ---
    logger.info("Creating per-site plots...")
    n_plotted = 0
    n_no_obs = 0
    for i in range(npoi):
        site_id = sites_df.iloc[i]['site_id']
        igbp = sites_df.iloc[i]['igbp']
        pft_idx = int(sites_df.iloc[i]['pft_idx'])

        obs_df = load_fluxnet_daily(site_id, args.eval_year)
        if obs_df is None or len(obs_df) < 30:
            n_no_obs += 1
            continue

        plot_site(
            site_id, igbp, pft_idx, dates,
            obs_df,
            daily_gpp_gc[:, i],
            daily_aet[:, i],
            daily_rh_gc[:, i],
            args.output_dir,
        )
        n_plotted += 1
        if n_plotted % 20 == 0:
            logger.info("  Plotted %d sites...", n_plotted)

    logger.info("Done! %d sites plotted, %d skipped (no obs for %d)",
                n_plotted, n_no_obs, args.eval_year)


if __name__ == "__main__":
    main()
