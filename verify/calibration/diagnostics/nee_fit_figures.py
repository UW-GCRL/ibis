"""Figures for the per-tower NEE fits (CALIBRATION.md 36): held-out night/day ratios S1 -> fit per tower, and the fitted values.

    python verify/calibration/diagnostics/nee_fit_figures.py <nee_fit_report.csv> <out dir>
"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
csv, OUT = sys.argv[1:3]; D = pd.read_csv(csv); plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
CLS = ['ENF', 'DBF', 'MF', 'EBF', 'GRA', 'OSH', 'CSH', 'SAV', 'WSA']; D['cls'] = pd.Categorical(D.igbp, CLS, ordered=True); D = D.sort_values(['cls', 'day_ratio_hold']).reset_index(drop=True)
C1, C2 = '#1b9e77', '#7570b3'
fig, axs = plt.subplots(1, 2, figsize=(12, 11), sharey=True); y = np.arange(len(D))
for a, tag, title in ((axs[0], 'night', 'a. night-time release (respiration)'), (axs[1], 'day', 'b. day-time uptake')):
    a.axvspan(0.8, 1.25, color='0.92', zorder=0); a.axvline(1, color='0.5', lw=0.8)
    x0, x1 = D[f'{tag}_ratio_s1_hold'].clip(0.2, 5), D[f'{tag}_ratio_hold'].clip(0.2, 5)
    for i in range(len(D)): a.plot([x0[i], x1[i]], [i, i], color='0.7', lw=1, zorder=1)
    w = lambda v: np.mean(np.abs(np.log(v.dropna())) < np.log(1.25))
    a.scatter(x0, y, s=22, color=C1, label=f'S1 set ({w(D[f"{tag}_ratio_s1_hold"]):.0%} within 25 %)', zorder=2); a.scatter(x1, y, s=26, color=C2, label=f'per-tower NEE fit ({w(D[f"{tag}_ratio_hold"]):.0%})', zorder=3)
    a.set_xscale('log'); a.set_xlim(0.2, 5); a.set_xticks([0.25, 0.5, 0.8, 1, 1.25, 2, 4]); a.set_xticklabels(['0.25', '0.5', '0.8', '1', '1.25', '2', '4']); a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel(f'{tag}-time NEE, model / tower, held-out year 2012'); a.set_title(title, loc='left', fontsize=10); a.legend(fontsize=8, frameon=False, loc='lower right')
    prev = None
    for i, c in enumerate(D.igbp):
        if c != prev: a.axhline(i - 0.5, color='0.8', lw=0.6); prev = c
axs[0].set_yticks(y); axs[0].set_yticklabels(D.site, fontsize=7); axs[0].set_ylim(-1, len(D)); axs[0].invert_yaxis()
for i, c in enumerate(D.igbp):
    if i == 0 or c != D.igbp[i - 1]: axs[1].text(5.2, i - 0.3, c, fontsize=8, va='top', color='0.3', clip_on=False)
fig.suptitle(f'Per-tower NEE fits, {len(D)} towers: night and day NEE on the held-out year, S1 set -> fit (pools from the site spin-up; som_scale, E0, rgrowth, rroot)', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/NEE_4_per_tower_fits_heldout.png', dpi=150); plt.close(fig)
# fitted values by class
P = [('p:som_scale', 'SOM pool factor (log)', (0.1, 3.0), True), ('p:rgrowth', 'growth respiration fraction', (0.10, 0.50), False), ('p:bgc_e0', 'Lloyd-Taylor E0 (K)', (150, 600), False), ('p:rroot', 'root maintenance (per yr, log)', (0.3, 5.0), True)]
F77 = {'p:som_scale': 1.0, 'p:rgrowth': 0.30, 'p:bgc_e0': 344.0, 'p:rroot': 1.25}
fig, axs = plt.subplots(1, 4, figsize=(14, 4.6)); rng = np.random.default_rng(0)
for a, (col, lab, (lo, hi), log) in zip(axs, P):
    for j, c in enumerate(CLS):
        v = D[D.igbp == c][col].dropna()
        a.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=C2, alpha=0.7)
        if len(v): a.plot([j - 0.25, j + 0.25], [v.median()] * 2, color='k', lw=2)
    a.axhline(F77[col], color='#d95f02', lw=1, ls='--', label='F77 value'); a.axhline(lo, color='0.6', lw=0.6, ls=':'); a.axhline(hi, color='0.6', lw=0.6, ls=':')
    if log: a.set_yscale('log')
    a.set_xticks(range(len(CLS))); a.set_xticklabels(CLS, fontsize=7); a.set_title(lab, loc='left', fontsize=9)
axs[0].legend(fontsize=8, frameon=False); fig.suptitle('Fitted respiration parameters per tower by class (bars = class median; dotted = fit bounds)', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/NEE_5_fitted_parameters.png', dpi=150); plt.close(fig)
print('figures written')
