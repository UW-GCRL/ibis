#!/usr/bin/env python3
"""Generate comprehensive comparison visualizations across calibration versions.

Compares V3 (baseline), V7 (decomp fix), V8b (CMA-ES optimized) against
ILAMB observations with multi-panel figures.
"""
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent))

ILAMB_ROOT = "/home/min/ILAMB-Data"
TARGET_LAT = np.arange(89.75, -90, -0.5)
TARGET_LON = np.arange(-179.75, 180, 0.5)
OUTPUT_DIR = "output/comparison_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VERSIONS = {
    'V3 (baseline)': 'output/cal_v3',
    'V7 (decomp fix)': 'output/cal_v7',
    'V8b (CMA-ES)': 'output/cal_v8b',
}
COLORS = {'V3 (baseline)': 'blue', 'V7 (decomp fix)': 'orange', 'V8b (CMA-ES)': 'red'}
YEARS = range(1995, 2006)


def load_obs():
    obs = {}
    # GPP
    ds = xr.open_dataset(f"{ILAMB_ROOT}/DATA/gpp/FLUXCOM/gpp.nc")
    gpp = ds['gpp'].where(ds['gpp'] >= 0).mean(dim='time', skipna=True)
    for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old in gpp.dims: gpp = gpp.rename({old: new})
    gpp = gpp.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    obs['gpp'] = {'data': gpp.values * 365.0 / 1000.0, 'units': 'kgC/m²/yr', 'label': 'GPP'}
    ds.close()

    # LAI
    ds = xr.open_dataset(f"{ILAMB_ROOT}/DATA/lai/MODIS/lai_0.5x0.5.nc")
    lai = ds['lai'].mean(dim='time', skipna=True)
    for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old in lai.dims: lai = lai.rename({old: new})
    lai = lai.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    obs['lai'] = {'data': lai.values, 'units': 'm²/m²', 'label': 'LAI'}
    ds.close()

    # ET
    ds = xr.open_dataset(f"{ILAMB_ROOT}/DATA/evspsbl/GLEAMv3.3a/et.nc", decode_times=False)
    et = ds['et'].mean(dim='time', skipna=True)
    for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old in et.dims: et = et.rename({old: new})
    if not np.issubdtype(et.dtype, np.floating): et = et.astype(np.float64)
    et = et.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    obs['et'] = {'data': et.values * 86400 * 365, 'units': 'mm/yr', 'label': 'ET'}
    ds.close()

    # LE
    ds = xr.open_dataset(f"{ILAMB_ROOT}/DATA/hfls/FLUXCOM/le.nc", decode_times=False)
    candidates = [v for v in ds.data_vars if v not in ('time_bounds','lat_bounds','lon_bounds') and len(ds[v].dims) >= 2]
    le = ds[candidates[0]].mean(dim='time', skipna=True)
    for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old in le.dims: le = le.rename({old: new})
    if not np.issubdtype(le.dtype, np.floating): le = le.astype(np.float64)
    le = le.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    obs['le'] = {'data': le.values, 'units': 'W/m²', 'label': 'Latent Heat'}
    ds.close()

    # Reco
    ds = xr.open_dataset(f"{ILAMB_ROOT}/DATA/reco/FLUXCOM/reco.nc")
    reco = ds['reco'].where(ds['reco'] >= 0).mean(dim='time', skipna=True)
    for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old in reco.dims: reco = reco.rename({old: new})
    reco = reco.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    obs['reco'] = {'data': reco.values * 365.0 / 1000.0, 'units': 'kgC/m²/yr', 'label': 'Ecosystem Resp.'}
    ds.close()

    # Biomass
    ds = xr.open_dataset(f"{ILAMB_ROOT}/DATA/biomass/ESACCI/biomass.nc", decode_times=False)
    bm = ds['biomass'].mean(dim='time', skipna=True) if 'time' in ds['biomass'].dims else ds['biomass']
    for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old in bm.dims: bm = bm.rename({old: new})
    if not np.issubdtype(bm.dtype, np.floating): bm = bm.astype(np.float64)
    bm = bm.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    obs['biomass'] = {'data': bm.values * 0.1 * 0.5, 'units': 'kgC/m²', 'label': 'Biomass C'}
    ds.close()

    # Soil C
    ds = xr.open_dataset(f"{ILAMB_ROOT}/DATA/cSoil/HWSD/soilc_0.5x0.5.nc", decode_times=False)
    sc = ds['cSoilAbove1m']
    if 'time' in sc.dims: sc = sc.mean(dim='time', skipna=True)
    for old, new in [('latitude', 'lat'), ('longitude', 'lon')]:
        if old in sc.dims: sc = sc.rename({old: new})
    if not np.issubdtype(sc.dtype, np.floating): sc = sc.astype(np.float64)
    sc = sc.interp(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    vals = sc.values.copy(); vals[vals > 1e10] = np.nan
    obs['csoil'] = {'data': vals, 'units': 'kgC/m²', 'label': 'Soil Carbon'}
    ds.close()

    return obs


def load_model(model_dir, years):
    accum = {}; n = 0
    for y in years:
        fpath = os.path.join(model_dir, f"ibis_yearly_{y}.nc")
        if not os.path.exists(fpath): continue
        ds = xr.open_dataset(fpath)
        for v in ds.data_vars:
            arr = ds[v].values.astype(np.float64)
            if v not in accum: accum[v] = np.zeros_like(arr)
            accum[v] += np.nan_to_num(arr)
        n += 1; ds.close()
    if n == 0: return {}
    raw = {k: v / n for k, v in accum.items()}
    model = {}
    if 'aygpptot' in raw: model['gpp'] = raw['aygpptot']
    if 'laiu' in raw and 'lail' in raw: model['lai'] = raw['laiu'] + raw['lail']
    if 'ayaet' in raw:
        model['et'] = raw['ayaet']
        model['le'] = raw['ayaet'] * 2.5e6 / (365.25 * 86400)
    if 'aytrunof' in raw:
        model['runoff'] = raw['aytrunof']
        if 'aytdrain' in raw: model['runoff'] += raw['aytdrain']
        model['runoff'] = np.where(model['runoff'] > 5000, np.nan, model['runoff'])
    if 'totcsoi' in raw:
        cs = raw['totcsoi']; cs[cs == 0] = np.nan; model['csoil'] = cs
    if 'ayco2mic' in raw: model['reco'] = raw['ayco2mic']
    if 'cveg' in raw: model['biomass'] = raw['cveg']
    return model


def stats(m, o):
    v = np.isfinite(m) & np.isfinite(o) & (m != 0) & (o != 0)
    if v.sum() < 10: return {'r': np.nan, 'bias': np.nan, 'rmse': np.nan}
    return {'r': np.corrcoef(m[v], o[v])[0, 1],
            'bias': np.mean(m[v] - o[v]),
            'rmse': np.sqrt(np.mean((m[v] - o[v])**2))}


def main():
    print("Loading observations...")
    obs = load_obs()

    print("Loading model versions...")
    models = {}
    for name, path in VERSIONS.items():
        models[name] = load_model(path, YEARS)
        print(f"  {name}: {len(models[name])} variables")

    lats = TARGET_LAT
    variables = ['gpp', 'lai', 'et', 'le', 'reco', 'biomass', 'csoil']

    # =========================================================================
    # Figure 1: Summary statistics table
    # =========================================================================
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis('off')
    cols = ['Variable', 'Units'] + [f'{v}\nr' for v in VERSIONS] + [f'{v}\nBias' for v in VERSIONS]
    rows = []
    for var in variables:
        if var not in obs: continue
        o = obs[var]
        row = [o['label'], o['units']]
        for vname in VERSIONS:
            m = models[vname]
            if var in m:
                s = stats(m[var].ravel(), o['data'].ravel())
                row.append(f"{s['r']:.3f}")
            else:
                row.append('—')
        for vname in VERSIONS:
            m = models[vname]
            if var in m:
                s = stats(m[var].ravel(), o['data'].ravel())
                row.append(f"{s['bias']:+.3f}")
            else:
                row.append('—')
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.1, 1.5)
    for j in range(len(cols)):
        table[0, j].set_facecolor('#343a40')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=7)
    # Color best r for each variable
    for i, row in enumerate(rows):
        r_vals = []
        for j in range(2, 2 + len(VERSIONS)):
            try: r_vals.append(float(row[j]))
            except: r_vals.append(-999)
        best_j = np.argmax(r_vals) + 2
        if r_vals[best_j - 2] > 0:
            table[i+1, best_j].set_facecolor('#d4edda')

    ax.set_title('Calibration Evolution: V3 → V7 → V8b', fontsize=14, fontweight='bold', pad=20)
    fig.savefig(f"{OUTPUT_DIR}/summary_table.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved summary_table.png")

    # =========================================================================
    # Figure 2: Zonal mean profiles (all versions on same plot)
    # =========================================================================
    plot_vars = [(v, obs[v]) for v in ['gpp', 'lai', 'et', 'reco'] if v in obs]
    fig, axes = plt.subplots(1, len(plot_vars), figsize=(5 * len(plot_vars), 4.5))
    if len(plot_vars) == 1: axes = [axes]

    for ax, (var, oinfo) in zip(axes, plot_vars):
        o2d = oinfo['data'].reshape(360, 720) if oinfo['data'].size == 259200 else oinfo['data']
        zm_obs = np.nanmean(np.where(o2d == 0, np.nan, o2d), axis=1)
        ax.plot(lats, zm_obs, 'k-', lw=2.5, label='Observed')

        for vname, color in COLORS.items():
            if var in models[vname]:
                m2d = models[vname][var]
                m2d = m2d.reshape(360, 720) if m2d.size == 259200 else m2d
                zm = np.nanmean(np.where(m2d == 0, np.nan, m2d), axis=1)
                ax.plot(lats, zm, color=color, lw=1.5, label=vname)

        ax.set_xlabel('Latitude'); ax.set_ylabel(oinfo['units'])
        ax.set_title(oinfo['label'], fontweight='bold')
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3); ax.set_xlim(90, -90)

    fig.suptitle('Zonal Mean Profiles (1995-2005)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUTPUT_DIR}/zonal_profiles.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved zonal_profiles.png")

    # =========================================================================
    # Figure 3: Scatter plots (obs vs model, all versions side by side)
    # =========================================================================
    for var in ['gpp', 'lai', 'et', 'le', 'reco']:
        if var not in obs: continue
        oinfo = obs[var]
        o_flat = oinfo['data'].ravel()

        nv = len([v for v in VERSIONS if var in models[v]])
        if nv == 0: continue
        fig, axes = plt.subplots(1, nv, figsize=(5 * nv, 5))
        if nv == 1: axes = [axes]

        idx = 0
        for vname in VERSIONS:
            if var not in models[vname]: continue
            ax = axes[idx]; idx += 1
            m_flat = models[vname][var].ravel()
            v = np.isfinite(o_flat) & np.isfinite(m_flat) & (o_flat != 0) & (m_flat != 0)
            if v.any():
                ax.scatter(o_flat[v], m_flat[v], s=0.3, alpha=0.2, c=COLORS[vname], rasterized=True)
                lo = min(0, min(np.percentile(o_flat[v], 1), np.percentile(m_flat[v], 1)))
                hi = max(np.percentile(o_flat[v], 99), np.percentile(m_flat[v], 99))
                ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
                s = stats(m_flat, o_flat)
                ax.text(0.05, 0.95, f"r={s['r']:.3f}\nbias={s['bias']:.3f}\nRMSE={s['rmse']:.3f}",
                        transform=ax.transAxes, fontsize=8, va='top', fontfamily='monospace',
                        bbox=dict(facecolor='white', alpha=0.8))
                ax.set_xlim(lo, hi * 1.05); ax.set_ylim(lo, hi * 1.05)
            ax.set_xlabel(f'Observed ({oinfo["units"]})'); ax.set_ylabel(f'Model ({oinfo["units"]})')
            ax.set_title(vname, fontweight='bold', color=COLORS[vname])

        fig.suptitle(f'{oinfo["label"]}: Model vs Observations', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{OUTPUT_DIR}/scatter_{var}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved scatter_{var}.png")

    # =========================================================================
    # Figure 4: Spatial maps for GPP (obs + best version + difference)
    # =========================================================================
    if 'gpp' in obs:
        best_name = 'V8b (CMA-ES)'
        if 'gpp' in models[best_name]:
            o2d = obs['gpp']['data'].reshape(360, 720)
            m2d = models[best_name]['gpp'].reshape(360, 720)

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            for ax, data, title, vmin, vmax, cmap in [
                (axes[0, 0], o2d, 'FLUXCOM GPP (obs)', 0, 3.0, 'YlGn'),
                (axes[0, 1], m2d, f'IBIS-JAX GPP ({best_name})', 0, 3.0, 'YlGn'),
                (axes[1, 0], m2d - o2d, 'Model - Observed', -1.5, 1.5, 'RdBu_r'),
            ]:
                im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                               aspect='auto', origin='upper', interpolation='nearest')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='kgC/m²/yr')
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xticks([]); ax.set_yticks([])

            # Zonal ratio
            ax = axes[1, 1]
            for vname, color in COLORS.items():
                if 'gpp' in models[vname]:
                    m = models[vname]['gpp'].reshape(360, 720)
                    zm_m = np.nanmean(np.where(m == 0, np.nan, m), axis=1)
                    zm_o = np.nanmean(np.where(o2d == 0, np.nan, o2d), axis=1)
                    ratio = np.where((zm_o > 0.05), zm_m / zm_o, np.nan)
                    ax.plot(lats, ratio, color=color, lw=1.5, label=vname)
            ax.axhline(1.0, color='k', ls='--', lw=1)
            ax.set_xlim(90, -90); ax.set_ylim(0, 3)
            ax.set_xlabel('Latitude'); ax.set_ylabel('Model / Observed')
            ax.set_title('GPP Zonal Ratio', fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

            fig.suptitle('GPP Spatial Comparison (kgC/m²/yr, 1995-2005)', fontsize=14, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(f"{OUTPUT_DIR}/spatial_gpp.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("  Saved spatial_gpp.png")

    # =========================================================================
    # Figure 5: Spatial maps for ET
    # =========================================================================
    if 'et' in obs:
        best_name = 'V8b (CMA-ES)'
        if 'et' in models[best_name]:
            o2d = obs['et']['data'].reshape(360, 720)
            m2d = models[best_name]['et'].reshape(360, 720)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, data, title, vmin, vmax, cmap in [
                (axes[0], o2d, 'GLEAM ET (obs)', 0, 1500, 'YlGnBu'),
                (axes[1], m2d, f'IBIS-JAX ET ({best_name})', 0, 1500, 'YlGnBu'),
                (axes[2], m2d - o2d, 'Model - Observed', -500, 500, 'RdBu_r'),
            ]:
                im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                               aspect='auto', origin='upper', interpolation='nearest')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='mm/yr')
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle('ET Spatial Comparison (mm/yr, 1995-2005)', fontsize=14, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(f"{OUTPUT_DIR}/spatial_et.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("  Saved spatial_et.png")

    # =========================================================================
    # Figure 6: Convergence of CMA-ES
    # =========================================================================
    cma_file = "output/calibration_cmaes_v2/optimized_params.npz"
    if os.path.exists(cma_file):
        r = np.load(cma_file)
        history = r['history']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history, 'b.', ms=3, alpha=0.4, label='Individual evals')
        ax.plot(np.minimum.accumulate(history), 'r-', lw=2, label='Best so far')
        ax.set_xlabel('Evaluation', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('CMA-ES Parameter Optimization (406 evaluations)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/cmaes_convergence.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved cmaes_convergence.png")

    # =========================================================================
    # Figure 7: Parameter comparison table
    # =========================================================================
    PFT_NAMES = ['Trop BL Ever', 'Trop BL Dec', 'Warm BL Ever', 'Temp Conifer',
                 'Temp BL Dec', 'Boreal Conifer', 'Boreal BL Dec', 'Boreal Con Dec',
                 'Ever Shrub', 'Dec Shrub', 'C4 Grass', 'C3 Grass']

    orig_vmax = [65, 65, 40, 30, 30, 25, 30, 30, 27.5, 27.5, 15, 25]
    cma_vmax = [32, 80, 25, 43, 48, 31.5, 18, 15, 13.3, 36, 10, 15]
    orig_specla = [25, 25, 25, 12.5, 25, 12.5, 25, 25, 12.5, 25, 20, 20]
    cma_specla = [13, 21.6, 30, 10.8, 35, 6.6, 30, 10.4, 9.7, 15.1, 15.7, 18.6]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(12)
    w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, orig_vmax, w, label='Original', color='steelblue', alpha=0.7)
    ax.bar(x + w/2, cma_vmax, w, label='CMA-ES Optimized', color='coral', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(PFT_NAMES, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Vmax (µmol/m²/s)'); ax.set_title('Vmax per PFT', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - w/2, orig_specla, w, label='Original', color='steelblue', alpha=0.7)
    ax.bar(x + w/2, cma_specla, w, label='CMA-ES Optimized', color='coral', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(PFT_NAMES, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('SLA (m²/kgC)'); ax.set_title('Specific Leaf Area per PFT', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('PFT Parameter Changes: Original vs CMA-ES Optimized', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUTPUT_DIR}/param_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved param_comparison.png")

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
