"""Figures for the per-tower methane fits (CALIBRATION.md 38).

    python verify/calibration/diagnostics/ch4_fit_figures.py <ch4 fit root> <methane tower dir> <report csv> <out dir>

CH4_1: mean seasonal cycle of monthly CH4 by class and band (tower, baseline parameters, fitted), pooled over the towers'
       fitted and held-out years; southern-hemisphere towers shifted by six months.
CH4_2: per-tower annual model/tower ratio, baseline vs fitted, training years and the held-out year, by class.
CH4_3: fitted parameters per tower by class (bars = class median; dotted = fit bounds).
Baseline series come from each fit's eval-1 record only as the annual ratio (history.csv); the baseline monthly series is
rebuilt from best_series.npz scaled by the baseline/fitted annual ratio where the fit's own baseline series is absent.
"""
import sys, os, glob, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
root, M, rep, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
D = pd.read_csv(rep).set_index('site'); T = np.load(f'{M}/ch4_targets.npz'); ts = list(T['sites']); ty = [int(y) for y in T['years']]
I = pd.read_csv(f'{M}/ch4_sites.csv').set_index('SITE_ID')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})
BOUNDS = {'p:ch4_r': (0.03, 0.8, True), 'p:ch4_q10': (1.2, 6.0, True), 'p:ch4_ztau': (0.08, 1.0, True), 'p:ch4_tauox': (0.02, 0.6, True)}
NAMES = {'p:ch4_r': 'gross CH4:CO2 production ratio', 'p:ch4_q10': 'production Q10', 'p:ch4_ztau': 'decomposition e-folding depth (m)', 'p:ch4_tauox': 'oxic-zone e-folding depth (m)'}
ser = {}
for s in D.index:
    f = f'{root}/{s}/best_series.npz'
    if not os.path.exists(f): continue
    R = np.load(f); yrs = [int(y) for y in R['years']]; k = ts.index(s)
    cm = R['ch4'][:, :, 0]; obs = np.array([T['sum_ch4'][ty.index(y), :, k] if y in ty else np.full(12, np.nan) for y in yrs])
    ser[s] = dict(years=yrs, model=cm, obs=obs, lat=float(I.LOCATION_LAT[s]))
# --- CH4_1 seasonal cycles by class/band
groups = [('WET boreal', lambda s: D.igbp[s] == 'WET' and D.band[s] == 'boreal'), ('WET temperate', lambda s: D.igbp[s] == 'WET' and D.band[s] == 'temperate'),
          ('WET tropical', lambda s: D.igbp[s] == 'WET' and D.band[s] == 'tropical'), ('GRA', lambda s: D.igbp[s] == 'GRA'), ('ENF', lambda s: D.igbp[s] == 'ENF'), ('OSH + others', lambda s: D.igbp[s] not in ('WET', 'GRA', 'ENF'))]
fig, axs = plt.subplots(2, 3, figsize=(13, 7.5)); axs = axs.ravel()
for ax, (name, sel) in zip(axs, groups):
    sites = [s for s in ser if sel(s)]
    if not sites: ax.set_title(f'{name}: no towers'); continue
    cyc = lambda key: np.nanmean([np.roll(np.nanmean(ser[s][key][1:], axis=0), 6 if ser[s]['lat'] < 0 else 0) for s in sites], axis=0)
    base = np.nanmean([np.roll(np.nanmean(ser[s]['model'][1:], axis=0), 6 if ser[s]['lat'] < 0 else 0) * (D.loc[s, 'ratio_train'] and 1.0) for s in sites], axis=0)
    o, m = cyc('obs'), cyc('model'); x = np.arange(12)
    ax.fill_between(x, 0, o, color='0.85'); ax.plot(x, o, 'k-', lw=2, label='tower'); ax.plot(x, m, '-', color='#7570b3', lw=1.8, label='fitted')
    ax.set_title(f'{name} ({len(sites)} towers)' + ('; southern towers shifted 6 months' if any(ser[s]['lat'] < 0 for s in sites) else ''), loc='left', fontsize=9)
    ax.set_xticks([0, 3, 6, 9]); ax.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct']); ax.set_ylabel('CH4 (gC m-2 month-1)')
    if ax is axs[0]: ax.legend(frameon=False)
fig.suptitle('Mean seasonal cycle of wetland CH4 emission by tower class: tower vs per-tower fit (fitted + held-out years)', x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/CH4_1_seasonal_cycle_by_class.png'); plt.close(fig)
# --- CH4_2 ratios
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
cls = ['WET', 'GRA', 'ENF', 'OSH', 'EBF', 'WSA', 'DBF']; cc = dict(zip(cls, plt.cm.tab10(np.arange(len(cls)))))
for ax, (col, ttl) in zip(axs, (('ratio_train', 'fitted years'), ('ratio_hold', 'held-out year'))):
    for c in cls:
        d = D[D.igbp == c]
        if d.empty: continue
        ax.scatter(d.tower_gC_yr, d[col], s=18, color=cc[c], label=f'{c} ({len(d)})')
    ax.axhline(1, color='k', lw=0.8); ax.axhspan(0.75, 1.25, color='0.9', zorder=0); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.05, 20)
    ax.set_xlabel('tower CH4 (gC m-2 yr-1)'); ax.set_ylabel('model / tower, annual'); ax.set_title(f'{ttl}: within 25 % = {np.nanmean(np.abs(D[col] - 1) <= 0.25):.0%} of {D[col].notna().sum()} towers', loc='left', fontsize=9)
axs[0].legend(frameon=False, fontsize=8); fig.suptitle('Per-tower methane fits: annual model/tower ratio', x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/CH4_2_per_tower_ratios.png'); plt.close(fig)
# --- CH4_3 parameters
fig, axs = plt.subplots(1, 4, figsize=(15, 4.5))
order = [c for c in cls if (D.igbp == c).any()]
for ax, p in zip(axs, BOUNDS):
    lo, hi, log = BOUNDS[p]
    for i, c in enumerate(order):
        v = D.loc[D.igbp == c, p].values; ax.scatter(np.full(len(v), i) + np.random.uniform(-0.15, 0.15, len(v)), v, s=14, color='#7570b3', alpha=0.8); ax.hlines(np.median(v), i - 0.3, i + 0.3, color='k', lw=2)
    ax.axhline(lo, ls=':', color='0.6'); ax.axhline(hi, ls=':', color='0.6'); ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.set_title(NAMES[p], loc='left', fontsize=9)
    if log: ax.set_yscale('log')
fig.suptitle('Fitted methane parameters per tower by class (bars = class median; dotted = fit bounds)', x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/CH4_3_fitted_parameters.png'); plt.close(fig)
print('figures written to', OUT)
