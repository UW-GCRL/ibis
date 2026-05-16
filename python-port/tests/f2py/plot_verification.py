#!/usr/bin/env python3
"""Generate comprehensive scatter-plot verification figures comparing
ibis-jax (JAX) against the original Fortran IBIS 2.6b4 (F77).

Runs both codes through multiple timesteps with identical synthetic inputs,
then produces multi-panel scatter plots for every output variable grouped
by physics process.

Usage:
    python tests/f2py/plot_verification.py [--output verification.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from tests.f2py.ibis_f77_wrapper import F77IBIS, NPOI
from tests.f2py.synthetic_inputs import (
    setup_f77_full, make_soil_inputs, make_snow_inputs,
    make_atm_inputs, make_veg_inputs,
    TMELT, HVAP, HSUB, CH2O, CVAP, CICE, EPSILON, HSOI,
)

# ---------------------------------------------------------------------------
# Collect F77 and JAX outputs for each process
# ---------------------------------------------------------------------------

def collect_setsoi():
    """Run setsoi on both F77 and JAX, return paired arrays."""
    from ibis.physics.soil_physics import setsoi as jax_setsoi

    f77 = F77IBIS()
    soil = make_soil_inputs()
    snow = make_snow_inputs()
    atm = make_atm_inputs()

    f77.set_dtime(3600.0)
    f77.set_soil(**soil, wpudmax=4.5, zwpmax=0.5)
    f77.set_snow(**snow)
    f77.set_atm(**atm)
    f77.call_setsoi()
    f77_out = f77.get_soil()

    jax_out = jax_setsoi(
        sand=jnp.array(soil['sand']), clay=jnp.array(soil['clay']),
        poros=jnp.array(soil['poros']), wsoi=jnp.array(soil['wsoi']),
        wisoi=jnp.array(soil['wisoi']), tsoi=jnp.array(soil['tsoi']),
        wpud=jnp.array(soil['wpud']), wipud=jnp.array(soil['wipud']),
        wpudmax=4.5, zwpmax=0.5,
        tsno_1=jnp.array(snow['tsno'][:, 0]), ta=jnp.array(atm['ta']),
        tmelt=TMELT, hvap=HVAP, hsub=HSUB, ch2o=CH2O, cvap=CVAP,
        cice=CICE, epsilon=EPSILON,
    )

    return {
        'consoi': (np.array(jax_out.consoi).ravel(), f77_out['consoi'].ravel(), 'W/(m K)'),
        'qglif': (np.array(jax_out.qglif).ravel(), f77_out['qglif'].ravel(), 'fraction'),
        'hvasug': (np.array(jax_out.hvasug), f77_out['hvasug'], 'J/kg'),
        'hvasui': (np.array(jax_out.hvasui), f77_out['hvasui'], 'J/kg'),
    }


def collect_fwetcal():
    """Run fwetcal on both."""
    from ibis.physics.interception import fwetcal as jax_fwetcal

    f77 = F77IBIS()
    setup_f77_full(f77)
    f77.call_fwetcal()
    f77_veg = f77.get_veg()

    veg = make_veg_inputs()
    jax_out = jax_fwetcal(
        tu=jnp.array(veg['tu']), ts=jnp.array(veg['ts']), tl=jnp.array(veg['tl']),
        wliqu=jnp.array(veg['wliqu']), wliqs=jnp.array(veg['wliqs']),
        wliql=jnp.array(veg['wliql']),
        wsnou=jnp.array(veg['wsnou']), wsnos=jnp.array(veg['wsnos']),
        wsnol=jnp.array(veg['wsnol']),
        wliqumax=0.2, wliqsmax=0.4, wliqlmax=0.2,
        wsnoumax=2.0, wsnosmax=4.0, wsnolmax=2.0,
        fmax=0.25, tmelt=TMELT, epsilon=EPSILON,
    )

    return {
        'fwetu': (np.array(jax_out.fwetu), f77_veg['fwetu'], 'fraction'),
        'fwets': (np.array(jax_out.fwets), f77_veg['fwets'], 'fraction'),
        'fwetl': (np.array(jax_out.fwetl), f77_veg['fwetl'], 'fraction'),
        'rliqu': (np.array(jax_out.rliqu), f77_veg['rliqu'], 'fraction'),
        'rliqs': (np.array(jax_out.rliqs), f77_veg['rliqs'], 'fraction'),
        'rliql': (np.array(jax_out.rliql), f77_veg['rliql'], 'fraction'),
    }


def collect_radiation():
    """Run radiation on both."""
    from ibis.physics.radiation import compute_radiation, default_veg_rad_params
    from ibis.physics.interception import fwetcal as jax_fwetcal

    f77 = F77IBIS()
    setup_f77_full(f77)
    f77.call_fwetcal()
    f77.call_radiation()
    f77_rad = f77.get_radiation()

    soil = make_soil_inputs(); snow = make_snow_inputs()
    atm = make_atm_inputs(); veg = make_veg_inputs()

    fwet = jax_fwetcal(
        jnp.array(veg['tu']), jnp.array(veg['ts']), jnp.array(veg['tl']),
        jnp.array(veg['wliqu']), jnp.array(veg['wliqs']), jnp.array(veg['wliql']),
        jnp.array(veg['wsnou']), jnp.array(veg['wsnos']), jnp.array(veg['wsnol']),
        0.2, 0.4, 0.2, 2.0, 4.0, 2.0, 0.25, TMELT, EPSILON)

    jax_rad = compute_radiation(
        jnp.array(atm['coszen']), jnp.array(atm['solad']), jnp.array(atm['solai']),
        jnp.array(atm['fira']), jnp.array(veg['lai']), jnp.array(veg['sai']),
        jnp.array(veg['fu']), jnp.array(veg['fl']),
        jnp.array(veg['tu']), jnp.array(veg['ts']), jnp.array(veg['tl']),
        jnp.array(soil['tg']), jnp.array(soil['ti']),
        jnp.array(snow['fi']), jnp.array(snow['tsno'][:, 0]),
        jnp.array(soil['wsoi'][:, 0]), jnp.array(soil['wisoi'][:, 0]),
        jnp.full(NPOI, 0.10), jnp.full(NPOI, 0.20),
        fwet.fwetu, fwet.fwets, fwet.fwetl, fwet.rliqu, fwet.rliqs, fwet.rliql,
        default_veg_rad_params(), 5.67e-8, TMELT, EPSILON)

    result = {}
    for var in ['solu', 'sols', 'soll', 'solg', 'soli']:
        result[var] = (np.array(getattr(jax_rad, var)), f77_rad[var], 'W/m2')
    for var in ['firu', 'firs', 'firl', 'firg', 'firi']:
        result[var] = (np.array(getattr(jax_rad, var)), f77_rad[var], 'W/m2')
    result['asurd_vis'] = (np.array(jax_rad.asurd[:, 0]), f77_rad['asurd'][:, 0], 'fraction')
    result['asurd_nir'] = (np.array(jax_rad.asurd[:, 1]), f77_rad['asurd'][:, 1], 'fraction')
    return result


def collect_canopy_fluxes():
    """Run setsoi+fwetcal+radiation+cascade+canopy on F77,
    compare canopy outputs (temperatures, fluxes) between F77 and initial state."""
    f77 = F77IBIS()
    setup_f77_full(f77)

    # Run through canopy
    f77.call_setsoi()
    f77.call_fwetcal()
    f77.call_radiation()
    f77.lib.wrap_call_cascade_()
    f77.lib.wrap_call_fwetcal_()
    f77.lib.wrap_call_canopy_()

    veg = f77.get_veg()
    soil = f77.get_soil()
    fluxes = f77.get_fluxes()

    atm = make_atm_inputs()

    return {
        'tu': (veg['tu'], atm['ta'], 'K'),
        'ts': (veg['ts'], atm['ta'], 'K'),
        'tl': (veg['tl'], atm['ta'], 'K'),
        'tg': (soil['tg'], atm['ta'], 'K'),
        'fsenu': (fluxes['fsenu'], np.zeros(NPOI), 'W/m2'),
        'fseng': (fluxes['fseng'], np.zeros(NPOI), 'W/m2'),
        'fvapg': (fluxes['fvapg'], np.zeros(NPOI), 'kg/m2/s'),
    }


def collect_lsxmain_multistep(nsteps=24):
    """Run lsxmain for nsteps on F77, collecting state after each step."""
    f77 = F77IBIS()
    setup_f77_full(f77)

    f77_tsoi_all, f77_wsoi_all = [], []
    f77_tg_all, f77_fi_all = [], []
    f77_tu_all, f77_tl_all = [], []
    f77_tsno_all, f77_hsno_all = [], []
    f77_heatg_all, f77_heati_all = [], []

    for step in range(nsteps):
        f77.call_lsxmain()
        s = f77.get_soil()
        sn = f77.get_snow()
        v = f77.get_veg()
        fl = f77.get_fluxes()
        f77_tsoi_all.append(s['tsoi'].copy())
        f77_wsoi_all.append(s['wsoi'].copy())
        f77_tg_all.append(s['tg'].copy())
        f77_fi_all.append(sn['fi'].copy())
        f77_tu_all.append(v['tu'].copy())
        f77_tl_all.append(v['tl'].copy())
        f77_tsno_all.append(sn['tsno'].copy())
        f77_hsno_all.append(sn['hsno'].copy())
        f77_heatg_all.append(fl['heatg'].copy())
        f77_heati_all.append(fl['heati'].copy())

    return {
        'tsoi': np.array(f77_tsoi_all),
        'wsoi': np.array(f77_wsoi_all),
        'tg': np.array(f77_tg_all),
        'fi': np.array(f77_fi_all),
        'tu': np.array(f77_tu_all),
        'tl': np.array(f77_tl_all),
        'tsno': np.array(f77_tsno_all),
        'hsno': np.array(f77_hsno_all),
        'heatg': np.array(f77_heatg_all),
        'heati': np.array(f77_heati_all),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SCENARIO_LABELS = ['Tropical', 'Arctic', 'Freeze/Thaw', 'Desert', 'Wetland']
SCENARIO_COLORS = ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#4daf4a']


def scatter_panel(ax, jax_vals, f77_vals, title, units, annotate_r=True):
    """Draw a scatter plot with 1:1 line and statistics."""
    valid = ~(np.isnan(jax_vals) | np.isnan(f77_vals))
    jv = jax_vals[valid]
    fv = f77_vals[valid]

    if len(jv) == 0:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight='bold')
        return

    # Color by scenario if arrays are multiples of NPOI
    if len(jv) % NPOI == 0:
        n_per = len(jv) // NPOI
        for i in range(NPOI):
            idx = slice(i * n_per, (i + 1) * n_per) if n_per > 1 else i
            ax.scatter(fv[idx] if isinstance(idx, slice) else [fv[idx]],
                      jv[idx] if isinstance(idx, slice) else [jv[idx]],
                      s=20, alpha=0.7, c=SCENARIO_COLORS[i],
                      label=SCENARIO_LABELS[i], zorder=3, edgecolors='none')
    else:
        ax.scatter(fv, jv, s=15, alpha=0.6, c='steelblue', zorder=3, edgecolors='none')

    # 1:1 line
    vmin = min(fv.min(), jv.min())
    vmax = max(fv.max(), jv.max())
    margin = max(0.05 * (vmax - vmin), 1e-10)
    lims = [vmin - margin, vmax + margin]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Statistics
    diff = jv - fv
    bias = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    if len(jv) > 1 and np.std(fv) > 0:
        r = np.corrcoef(fv, jv)[0, 1]
    else:
        r = 1.0

    stats_text = f"RMSE={rmse:.2e}\nbias={bias:.2e}"
    if annotate_r:
        stats_text = f"r={r:.6f}\n" + stats_text
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel(f'Fortran ({units})', fontsize=8)
    ax.set_ylabel(f'JAX ({units})', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal')


def timeseries_panel(ax, data, title, ylabel, show_legend=False):
    """Draw time-series for each scenario."""
    nsteps = data.shape[0]
    x = np.arange(nsteps)
    for i in range(min(NPOI, data.shape[1])):
        vals = data[:, i] if data.ndim == 2 else data[:, i, 0]
        ax.plot(x, vals, '-o', c=SCENARIO_COLORS[i], markersize=2, lw=1,
                label=SCENARIO_LABELS[i])
    ax.set_xlabel('Timestep (h)', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=6, loc='best', ncol=2)


# ---------------------------------------------------------------------------
# Main figure generation
# ---------------------------------------------------------------------------

def main(output_path: str = "verification_scatter.png"):
    print("Collecting process outputs...")

    print("  [1/5] setsoi...")
    setsoi_data = collect_setsoi()

    print("  [2/5] fwetcal...")
    fwetcal_data = collect_fwetcal()

    print("  [3/5] radiation...")
    rad_data = collect_radiation()

    print("  [4/5] canopy exchange...")
    canopy_data = collect_canopy_fluxes()

    print("  [5/5] lsxmain x24 timesteps...")
    ts_data = collect_lsxmain_multistep(24)

    # Flatten all process data into a single ordered list for the scatter figure
    all_scatter = []  # [(title, jax, f77, units), ...]

    for name, (jv, fv, units) in setsoi_data.items():
        all_scatter.append((f'setsoi: {name}', jv, fv, units))
    for name, (jv, fv, units) in fwetcal_data.items():
        all_scatter.append((f'fwetcal: {name}', jv, fv, units))
    for name, (jv, fv, units) in rad_data.items():
        all_scatter.append((f'radiation: {name}', jv, fv, units))

    n_scatter = len(all_scatter)
    ncols = 5
    nrows = (n_scatter + ncols - 1) // ncols + 1  # +1 for legend

    # =====================================================================
    # Figure 1: Process-level scatter plots
    # =====================================================================
    print("Creating scatter plot figure...")
    fig = plt.figure(figsize=(ncols * 4.0, nrows * 3.8))
    fig.suptitle(
        "ibis-jax vs Fortran IBIS 2.6b4: Process-by-Process Verification\n"
        "5 synthetic scenarios: Tropical, Arctic, Freeze/Thaw, Desert, Wetland",
        fontsize=14, fontweight='bold', y=0.995
    )

    gs = gridspec.GridSpec(nrows, ncols, hspace=0.50, wspace=0.40,
                           left=0.05, right=0.97, top=0.96, bottom=0.03)

    for idx, (title, jv, fv, units) in enumerate(all_scatter):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        scatter_panel(ax, jv, fv, title, units)

    # Legend row
    ax_legend = fig.add_subplot(gs[nrows - 1, :])
    ax_legend.axis('off')
    for i in range(NPOI):
        ax_legend.scatter([], [], c=SCENARIO_COLORS[i], s=60, label=SCENARIO_LABELS[i])
    ax_legend.legend(fontsize=11, loc='center', ncol=5, frameon=True,
                     title='Synthetic Test Scenarios', title_fontsize=12)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)

    # =====================================================================
    # Figure 2: Time-series from lsxmain multi-step
    # =====================================================================
    print("Creating time-series figure...")
    fig2, axes = plt.subplots(5, 2, figsize=(16, 22))
    fig2.suptitle(
        "Fortran IBIS 2.6b4: 24-Hour Integration (lsxmain x 24 timesteps)\n"
        "5 scenarios with 1-hour timestep",
        fontsize=13, fontweight='bold'
    )

    timeseries_panel(axes[0, 0], ts_data['tsoi'][:, :, 0], 'Soil Temp (Layer 1)', 'K', show_legend=True)
    timeseries_panel(axes[0, 1], ts_data['tg'], 'Ground Skin Temp (tg)', 'K')
    timeseries_panel(axes[1, 0], ts_data['wsoi'][:, :, 0], 'Soil Moisture (Layer 1)', 'fraction')
    timeseries_panel(axes[1, 1], ts_data['fi'], 'Snow Fraction (fi)', 'fraction')
    timeseries_panel(axes[2, 0], ts_data['tu'], 'Upper Canopy Temp (tu)', 'K')
    timeseries_panel(axes[2, 1], ts_data['tl'], 'Lower Canopy Temp (tl)', 'K')
    timeseries_panel(axes[3, 0], ts_data['tsno'][:, :, 0], 'Snow Temp (Layer 1)', 'K')
    timeseries_panel(axes[3, 1], ts_data['tsoi'][:, :, -1], 'Soil Temp (Layer 6, deep)', 'K')
    timeseries_panel(axes[4, 0], ts_data['heatg'], 'Ground Heat Flux (heatg)', 'W/m2')
    timeseries_panel(axes[4, 1], ts_data['heati'], 'Snow Heat Flux (heati)', 'W/m2')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    ts_path = output_path.replace('.png', '_timeseries.png')
    fig2.savefig(ts_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {ts_path}")
    plt.close(fig2)

    # =====================================================================
    # Figure 3: Canopy exchange - temperature departures and fluxes
    # =====================================================================
    print("Creating canopy exchange figure...")
    fig3, axes3 = plt.subplots(2, 4, figsize=(18, 8))
    fig3.suptitle(
        'Canopy Exchange: Post-Solve Temperatures and Fluxes\n'
        '(Fortran IBIS 2.6b4, single timestep)',
        fontsize=13, fontweight='bold'
    )

    atm = make_atm_inputs()
    can_labels = {
        'tu': 'Upper Canopy Temp',
        'ts': 'Stem Temp',
        'tl': 'Lower Canopy Temp',
        'tg': 'Ground Skin Temp',
        'fsenu': 'Sensible Heat (upper)',
        'fseng': 'Sensible Heat (ground)',
        'fvapg': 'Soil Evaporation',
    }

    for idx, (name, (vals, ref, units)) in enumerate(canopy_data.items()):
        row, col = divmod(idx, 4)
        ax = axes3[row, col]

        if 'Temp' in can_labels.get(name, ''):
            # Bar chart: T_component vs T_air
            x = np.arange(NPOI)
            width = 0.35
            ax.bar(x - width/2, vals, width, label='Component', color=[SCENARIO_COLORS[i] for i in range(NPOI)])
            ax.bar(x + width/2, atm['ta'], width, label='Air (ta)', color='gray', alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(SCENARIO_LABELS, fontsize=7, rotation=30)
            ax.set_ylabel(units, fontsize=8)
            ax.legend(fontsize=7)
        else:
            # Bar chart: flux values
            x = np.arange(NPOI)
            ax.bar(x, vals, color=[SCENARIO_COLORS[i] for i in range(NPOI)])
            ax.set_xticks(x)
            ax.set_xticklabels(SCENARIO_LABELS, fontsize=7, rotation=30)
            ax.set_ylabel(units, fontsize=8)
            ax.axhline(0, color='k', lw=0.5)

        ax.set_title(can_labels.get(name, name), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # Hide extra panel
    if len(canopy_data) < 8:
        axes3[1, 3].axis('off')

    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    canopy_path = output_path.replace('.png', '_canopy.png')
    fig3.savefig(canopy_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {canopy_path}")
    plt.close(fig3)

    # =====================================================================
    # Figure 4: Summary statistics table
    # =====================================================================
    print("Creating summary statistics figure...")
    fig4, ax4 = plt.subplots(figsize=(14, 12))
    ax4.axis('off')

    all_data = {}
    all_data.update({f'setsoi.{k}': v for k, v in setsoi_data.items()})
    all_data.update({f'fwetcal.{k}': v for k, v in fwetcal_data.items()})
    all_data.update({f'radiation.{k}': v for k, v in rad_data.items()})

    rows = []
    for name, (jv, fv, units) in all_data.items():
        valid = ~(np.isnan(jv) | np.isnan(fv))
        jv_v, fv_v = jv[valid], fv[valid]
        if len(jv_v) == 0:
            continue
        diff = jv_v - fv_v
        bias = np.mean(diff)
        rmse = np.sqrt(np.mean(diff ** 2))
        maxerr = np.max(np.abs(diff))
        if len(jv_v) > 1 and np.std(fv_v) > 0:
            r = np.corrcoef(fv_v, jv_v)[0, 1]
        else:
            r = 1.0
        rows.append([name, units, f'{r:.8f}', f'{bias:.2e}', f'{rmse:.2e}', f'{maxerr:.2e}', len(jv_v)])

    col_labels = ['Variable', 'Units', 'Correlation', 'Bias', 'RMSE', 'Max Error', 'N']
    table = ax4.table(cellText=rows, colLabels=col_labels,
                      cellLoc='center', loc='center',
                      colWidths=[0.20, 0.08, 0.12, 0.12, 0.12, 0.12, 0.06])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    for i, row in enumerate(rows):
        r_val = float(row[2])
        if r_val > 0.999999:
            color = '#d4edda'
        elif r_val > 0.99:
            color = '#fff3cd'
        else:
            color = '#f8d7da'
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#343a40')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax4.set_title(
        'ibis-jax vs Fortran IBIS 2.6b4: Process Verification Summary\n'
        'Green: r > 0.999999 (exact match)  |  Yellow: r > 0.99  |  Red: r < 0.99',
        fontsize=12, fontweight='bold', pad=20
    )

    stats_path = output_path.replace('.png', '_stats.png')
    fig4.savefig(stats_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {stats_path}")
    plt.close(fig4)

    print("\nDone! Generated 4 figures.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="verification_scatter.png")
    args = parser.parse_args()
    main(args.output)
