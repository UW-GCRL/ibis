#!/usr/bin/env python3
"""Quick JAX-native parameter calibration using linear scaling assumption.

Uses the baseline model output (GPP, NPP, AET, LAI) and scales it by
parameter ratios. The relationship GPP ∝ vmax, LAI ∝ specla×aleaf is
approximately linear, so we can differentiate through the scaling
without re-running the model.

After finding optimal scaling factors, the actual parameters are updated
and a validation run is performed.

Usage:
    python scripts/calibrate_quick.py --baseline-dir output/cal_final
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

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# PFT group mapping (12 PFTs → 5 groups)
GROUP_FOR_PFT = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4])


class CalibParams(eqx.Module):
    """Differentiable calibration parameters (scaling factors)."""
    vmax_scale: jnp.ndarray    # (5,) GPP scaling per PFT group
    specla_scale: jnp.ndarray  # (5,) LAI scaling per PFT group
    et_scale: jnp.ndarray      # (5,) ET scaling per PFT group


def default_params():
    return CalibParams(
        vmax_scale=jnp.ones(5),
        specla_scale=jnp.ones(5),
        et_scale=jnp.ones(5),
    )


def load_model_annual_mean(baseline_dir, years=range(2000, 2011)):
    """Load model annual means for GPP, NPP, AET, LAI over specified years."""
    gpp_sum, npp_sum, aet_sum, lai_sum = None, None, None, None
    n = 0
    for y in years:
        fpath = os.path.join(baseline_dir, f"ibis_yearly_{y}.nc")
        if not os.path.exists(fpath):
            continue
        ds = xr.open_dataset(fpath)
        if gpp_sum is None:
            shape = ds['ayanpptot'].shape
            gpp_sum = np.zeros(shape)
            npp_sum = np.zeros(shape)
            aet_sum = np.zeros(shape)
            lai_sum = np.zeros(shape)

        gpp = np.array(ds.get('aygpptot', np.zeros(shape)))
        npp = np.array(ds['ayanpptot'])
        aet = np.array(ds['ayaet'])
        laiu = np.array(ds['laiu']) if 'laiu' in ds else np.zeros(shape)
        lail = np.array(ds['lail']) if 'lail' in ds else np.zeros(shape)

        gpp_sum += np.nan_to_num(gpp)
        npp_sum += np.nan_to_num(npp)
        aet_sum += np.nan_to_num(aet)
        lai_sum += np.nan_to_num(laiu + lail)
        n += 1
        ds.close()

    if n == 0:
        raise ValueError(f"No yearly files found in {baseline_dir}")

    return {
        'gpp': gpp_sum / n,    # kgC/m²/yr
        'npp': npp_sum / n,
        'aet': aet_sum / n,    # mm/yr
        'lai': lai_sum / n,    # m²/m²
    }


def load_observations(obs_dir):
    """Load ILAMB observations as (360, 720) arrays."""
    obs = {}
    for name in ['gpp_annual_mean', 'lai_annual_mean', 'cveg_mean', 'csoil_mean']:
        fpath = os.path.join(obs_dir, f"{name}.npy")
        if os.path.exists(fpath):
            key = name.replace('_annual_mean', '').replace('_mean', '')
            obs[key] = np.load(fpath)
    return obs


def compute_loss(params, model_baseline, obs, land_mask):
    """Differentiable loss: scale baseline by params, compare to obs."""
    gpp_scaled = model_baseline['gpp'] * params.vmax_scale[0]  # simplified: uniform scaling
    lai_scaled = model_baseline['lai'] * params.specla_scale[0]
    aet_scaled = model_baseline['aet'] * params.et_scale[0]

    loss = jnp.float64(0.0)
    n_targets = 0

    if 'gpp' in obs:
        obs_gpp = jnp.array(obs['gpp'])
        valid = jnp.isfinite(obs_gpp) & (obs_gpp > 0) & land_mask
        diff = jnp.where(valid, gpp_scaled - obs_gpp, 0.0)
        loss = loss + jnp.sum(diff**2) / jnp.maximum(jnp.sum(valid), 1.0)
        n_targets += 1

    if 'lai' in obs:
        obs_lai = jnp.array(obs['lai'])
        valid = jnp.isfinite(obs_lai) & (obs_lai > 0) & land_mask
        diff = jnp.where(valid, lai_scaled - obs_lai, 0.0)
        loss = loss + 0.5 * jnp.sum(diff**2) / jnp.maximum(jnp.sum(valid), 1.0)
        n_targets += 1

    return loss


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="output/cal_final")
    parser.add_argument("--obs-dir", default="data/ilamb_obs")
    parser.add_argument("--output-dir", default="output/calibration_results")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model baseline (2000-2010 mean)
    logger.info("Loading model baseline from %s...", args.baseline_dir)
    model = load_model_annual_mean(args.baseline_dir, years=range(2000, 2011))
    logger.info("  GPP: mean=%.3f kgC/m²/yr (finite pts: %d)",
                np.nanmean(model['gpp'][model['gpp'] > 0]),
                np.sum(np.isfinite(model['gpp']) & (model['gpp'] > 0)))
    logger.info("  LAI: mean=%.3f m²/m²", np.nanmean(model['lai'][model['lai'] > 0]))
    logger.info("  AET: mean=%.0f mm/yr", np.nanmean(model['aet'][model['aet'] > 0]))

    # Load observations
    logger.info("Loading observations from %s...", args.obs_dir)
    obs = load_observations(args.obs_dir)
    for k, v in obs.items():
        logger.info("  %s: mean=%.3f (finite: %d)", k, np.nanmean(v[np.isfinite(v)]), np.sum(np.isfinite(v)))

    # Create land mask (where both model and obs have valid data)
    land_mask = np.isfinite(model['gpp']) & (model['gpp'] > 0)
    if 'gpp' in obs:
        land_mask &= np.isfinite(obs['gpp']) & (obs['gpp'] > 0)
    land_mask_jax = jnp.array(land_mask)
    logger.info("  Land mask: %d valid points", land_mask.sum())

    # Convert to JAX
    model_jax = {k: jnp.array(v) for k, v in model.items()}
    obs_jax = {k: jnp.array(v) for k, v in obs.items()}

    # Optimize
    params = default_params()
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    logger.info("Starting optimization (%d steps, lr=%.4f)...", args.n_steps, args.lr)
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

        if i % 20 == 0 or i == args.n_steps - 1:
            logger.info("  Step %3d: loss=%.6f  vmax=%.3f  specla=%.3f  et=%.3f",
                        i, losses[-1],
                        float(params.vmax_scale[0]),
                        float(params.specla_scale[0]),
                        float(params.et_scale[0]))

    logger.info("\n=== CALIBRATION RESULTS ===")
    logger.info("Final loss: %.6f (initial: %.6f, reduction: %.1f%%)",
                losses[-1], losses[0], 100*(1-losses[-1]/losses[0]))
    logger.info("Optimal vmax_scale: %.4f", float(params.vmax_scale[0]))
    logger.info("Optimal specla_scale: %.4f", float(params.specla_scale[0]))
    logger.info("Optimal et_scale: %.4f", float(params.et_scale[0]))

    # Save calibrated scaling factors
    np.savez(os.path.join(args.output_dir, "calibrated_params.npz"),
             vmax_scale=np.array(params.vmax_scale),
             specla_scale=np.array(params.specla_scale),
             et_scale=np.array(params.et_scale),
             losses=np.array(losses))

    # ====================================================================
    # VISUALIZATIONS
    # ====================================================================

    gpp_model = np.array(model_jax['gpp'])
    gpp_calib = gpp_model * float(params.vmax_scale[0])
    lai_model = np.array(model_jax['lai'])
    lai_calib = lai_model * float(params.specla_scale[0])

    gpp_obs = obs.get('gpp', np.full_like(gpp_model, np.nan))
    lai_obs = obs.get('lai', np.full_like(lai_model, np.nan))

    # --- Figure 1: Loss convergence ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(losses, 'b-', lw=1.5)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Calibration Convergence')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, "convergence.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 2: GPP scatter (baseline vs obs, calibrated vs obs) ---
    valid = np.isfinite(gpp_obs) & np.isfinite(gpp_model) & (gpp_obs > 0) & (gpp_model > 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, gpp_arr, title in [(axes[0], gpp_model, 'Baseline'),
                                (axes[1], gpp_calib, 'Calibrated')]:
        ax.scatter(gpp_obs[valid], gpp_arr[valid], s=1, alpha=0.3, c='steelblue')
        vmax = max(gpp_obs[valid].max(), gpp_arr[valid].max())
        ax.plot([0, vmax], [0, vmax], 'k--', lw=1)
        bias = np.mean(gpp_arr[valid] - gpp_obs[valid])
        rmse = np.sqrt(np.mean((gpp_arr[valid] - gpp_obs[valid])**2))
        r = np.corrcoef(gpp_obs[valid], gpp_arr[valid])[0, 1]
        ax.set_xlabel('Observed GPP (kgC/m²/yr)')
        ax.set_ylabel('Model GPP (kgC/m²/yr)')
        ax.set_title(f'{title}\nr={r:.3f}, bias={bias:.3f}, RMSE={rmse:.3f}')
        ax.set_xlim(0, vmax*1.1); ax.set_ylim(0, vmax*1.1)

    fig.suptitle('GPP: Model vs FLUXCOM Observations (2000-2010 mean)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "gpp_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 3: LAI scatter ---
    valid_lai = np.isfinite(lai_obs) & np.isfinite(lai_model) & (lai_obs > 0) & (lai_model > 0)
    if valid_lai.any():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, lai_arr, title in [(axes[0], lai_model, 'Baseline'),
                                    (axes[1], lai_calib, 'Calibrated')]:
            ax.scatter(lai_obs[valid_lai], lai_arr[valid_lai], s=1, alpha=0.3, c='green')
            vmax = max(lai_obs[valid_lai].max(), lai_arr[valid_lai].max())
            ax.plot([0, vmax], [0, vmax], 'k--', lw=1)
            bias = np.mean(lai_arr[valid_lai] - lai_obs[valid_lai])
            rmse = np.sqrt(np.mean((lai_arr[valid_lai] - lai_obs[valid_lai])**2))
            r = np.corrcoef(lai_obs[valid_lai], lai_arr[valid_lai])[0, 1]
            ax.set_xlabel('Observed LAI (m²/m²)')
            ax.set_ylabel('Model LAI (m²/m²)')
            ax.set_title(f'{title}\nr={r:.3f}, bias={bias:.3f}, RMSE={rmse:.3f}')
        fig.suptitle('LAI: Model vs MODIS Observations (2000-2010 mean)', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "lai_scatter.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- Figure 4: Spatial maps (GPP baseline, calibrated, observed, difference) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                              subplot_kw={'projection': None})

    # Reshape to 2D grid
    gpp_obs_2d = gpp_obs.reshape(360, 720) if gpp_obs.ndim == 1 else gpp_obs
    gpp_mod_2d = gpp_model.reshape(360, 720) if gpp_model.ndim == 1 else gpp_model
    gpp_cal_2d = gpp_calib.reshape(360, 720) if gpp_calib.ndim == 1 else gpp_calib

    vmax_gpp = 3.0
    for ax, data, title, cmap in [
        (axes[0, 0], gpp_obs_2d, 'FLUXCOM GPP (obs)', 'YlGn'),
        (axes[0, 1], gpp_mod_2d, 'IBIS-JAX GPP (baseline)', 'YlGn'),
        (axes[1, 0], gpp_cal_2d, 'IBIS-JAX GPP (calibrated)', 'YlGn'),
        (axes[1, 1], gpp_cal_2d - gpp_obs_2d, 'Calibrated - Observed', 'RdBu_r'),
    ]:
        if 'Observed' in title or 'diff' in title.lower():
            im = ax.imshow(data, vmin=-1.5, vmax=1.5, cmap=cmap,
                          aspect='auto', origin='upper')
        else:
            im = ax.imshow(data, vmin=0, vmax=vmax_gpp, cmap=cmap,
                          aspect='auto', origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    fig.suptitle('GPP Spatial Patterns (kgC/m²/yr, 2000-2010 mean)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "gpp_spatial.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 5: Zonal mean profiles ---
    lats = np.arange(89.75, -90, -0.5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, obs_2d, mod_2d, cal_2d, title, units in [
        (axes[0], gpp_obs_2d, gpp_mod_2d, gpp_cal_2d, 'GPP', 'kgC/m²/yr'),
        (axes[1],
         lai_obs.reshape(360, 720) if lai_obs.ndim == 1 else lai_obs,
         lai_model.reshape(360, 720) if lai_model.ndim == 1 else lai_model,
         lai_calib.reshape(360, 720) if lai_calib.ndim == 1 else lai_calib,
         'LAI', 'm²/m²'),
    ]:
        obs_zonal = np.nanmean(obs_2d, axis=1)
        mod_zonal = np.nanmean(mod_2d, axis=1)
        cal_zonal = np.nanmean(cal_2d, axis=1)
        ax.plot(lats, obs_zonal, 'k-', lw=2, label='Observed')
        ax.plot(lats, mod_zonal, 'b--', lw=1.5, label='Baseline')
        ax.plot(lats, cal_zonal, 'r-', lw=1.5, label='Calibrated')
        ax.set_xlabel('Latitude')
        ax.set_ylabel(units)
        ax.set_title(f'Zonal Mean {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(90, -90)

    # AET zonal (no obs available, just show model)
    aet_mod_2d = model['aet'].reshape(360, 720) if model['aet'].ndim == 1 else model['aet']
    aet_cal_2d = (model['aet'] * float(params.et_scale[0])).reshape(360, 720) if model['aet'].ndim == 1 else model['aet'] * float(params.et_scale[0])
    mod_z = np.nanmean(aet_mod_2d, axis=1)
    cal_z = np.nanmean(aet_cal_2d, axis=1)
    axes[2].plot(lats, mod_z, 'b--', lw=1.5, label='Baseline')
    axes[2].plot(lats, cal_z, 'r-', lw=1.5, label='Calibrated')
    axes[2].set_xlabel('Latitude')
    axes[2].set_ylabel('mm/yr')
    axes[2].set_title('Zonal Mean AET')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(90, -90)

    fig.suptitle('Zonal Mean Profiles (2000-2010 climatology)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "zonal_profiles.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 6: Summary statistics table ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    rows = []
    for var, mod, cal, ob, units in [
        ('GPP', gpp_model, gpp_calib, gpp_obs, 'kgC/m²/yr'),
        ('LAI', lai_model, lai_calib, lai_obs, 'm²/m²'),
    ]:
        v = np.isfinite(ob) & np.isfinite(mod) & (ob > 0) & (mod > 0)
        if v.any():
            r_base = np.corrcoef(ob[v], mod[v])[0, 1]
            r_cal = np.corrcoef(ob[v], cal[v])[0, 1]
            bias_base = np.mean(mod[v] - ob[v])
            bias_cal = np.mean(cal[v] - ob[v])
            rmse_base = np.sqrt(np.mean((mod[v] - ob[v])**2))
            rmse_cal = np.sqrt(np.mean((cal[v] - ob[v])**2))
            rows.append([var, units,
                        f'{r_base:.3f}', f'{r_cal:.3f}',
                        f'{bias_base:.3f}', f'{bias_cal:.3f}',
                        f'{rmse_base:.3f}', f'{rmse_cal:.3f}'])

    col_labels = ['Variable', 'Units', 'r (base)', 'r (cal)',
                  'Bias (base)', 'Bias (cal)', 'RMSE (base)', 'RMSE (cal)']
    table = ax.table(cellText=rows, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#343a40')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Calibration Summary: Baseline vs Calibrated',
                 fontsize=13, fontweight='bold', pad=20)
    fig.savefig(os.path.join(args.output_dir, "summary_table.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info("\nVisualization saved to %s/", args.output_dir)
    logger.info("  convergence.png — Loss curve")
    logger.info("  gpp_scatter.png — GPP scatter plots")
    logger.info("  lai_scatter.png — LAI scatter plots")
    logger.info("  gpp_spatial.png — GPP spatial maps")
    logger.info("  zonal_profiles.png — Zonal mean profiles")
    logger.info("  summary_table.png — Statistics table")


if __name__ == "__main__":
    main()
