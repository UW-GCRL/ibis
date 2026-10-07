"""The nine site-level figures (seasonal cycle by class, twelve example towers, pooled scatter with r by class) for GPP,
ET and NEE, with three model runs: IBIS default parameters, the per-tower GPP/ET fits (Supplementary 2, CALIBRATION.md
35) and the per-tower NEE fits (section 36), against the towers.

    python verify/calibration/diagnostics/site_level_figures_3runs.py <out dir> [nee fit folder, default nee]

Series, 2009-2012 monthly: default = site_runs/npp_default_2008-2012 (gpp, et, nee); per-tower GPP/ET fit =
site_fit/s2/<SITE>/best_series.npz (DE-Obe and US-Syv without the summary line are taken as they are; a tower without a
result is left blank for that run); per-tower NEE fit = site_fit/nee/<SITE>/best_series.npz (its nee is the (years, 12,
night/day) array, summed). Tower GPP and ET come with the default run, tower NEE from tower_nee_2008-2012.npz.
"""
import sys, os, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = sys.argv[1]; NEE_DIR = sys.argv[2] if len(sys.argv) > 2 else 'nee'; V = '/shared/nas2/Hangkai/ibis/verify_outputs'; os.makedirs(OUT, exist_ok=True)   # argv[2]: the NEE fit folder under site_fit (nee, nee_v2)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})
RUNS = [('def', 'IBIS default parameters', '#d95f02', '--', 1.4), ('s2', 'per-tower GPP/ET fit (previous)', '#1b9e77', '-', 1.6), ('nee', 'per-tower NEE fit (current)', '#7570b3', '-', 1.8)]
D = np.load(f'{V}/site_runs/npp_default_2008-2012/site_runs.npz'); sites = list(D['sites']); ns = len(sites); K = [1, 2, 3, 4]
inv = pd.read_csv('/shared/nas2/Hangkai/ibis-port/verify/calibration/site_data/site_inventory.csv').set_index('site'); igbp = {s: inv.loc[s, 'igbp'] for s in sites}; lat = {s: float(inv.loc[s, 'lat']) for s in sites}
CLS = ['ENF', 'DBF', 'MF', 'EBF', 'GRA', 'OSH', 'CSH', 'SAV', 'WSA']
CLSNAME = {'ENF': 'evergreen needleleaf forest', 'DBF': 'deciduous broadleaf forest', 'MF': 'mixed forest', 'EBF': 'evergreen broadleaf forest', 'GRA': 'grassland', 'OSH': 'open shrubland', 'CSH': 'closed shrubland', 'SAV': 'savanna', 'WSA': 'woody savanna'}
CC = dict(zip(CLS, plt.cm.tab10(np.arange(len(CLS))))); EX = ['DE-Tha', 'US-Ha1', 'FI-Hyy', 'US-PFa', 'AT-Neu', 'AU-Stp', 'FR-Pue', 'GF-Guy', 'ES-LJu', 'GL-ZaH', 'CA-TP1', 'US-Var']
TN = np.load('/shared/nas2/Hangkai/ibis_local_document/figures/site_level/tower_nee_2008-2012.npz'); assert list(TN['sites']) == sites
nan = lambda: np.full((5, 12, ns), np.nan)
mod = {'def': {'gpp': D['gpp'], 'et': D['et'], 'nee': D['nee']}, 's2': {'gpp': nan(), 'et': nan(), 'nee': nan()}, 'nee': {'gpp': nan(), 'et': nan(), 'nee': nan()}}
got = {'s2': 0, 'nee': 0}
for n, s in enumerate(sites):
    for tag in ('s2', 'nee'):
        f = f'{V}/site_fit/{NEE_DIR if tag == "nee" else tag}/{s}/best_series.npz'
        if not os.path.exists(f): continue
        R = np.load(f, allow_pickle=True); ne = R['nee']; ne = ne.sum(axis=2) if ne.ndim == 4 else ne
        if R['gpp'].shape[0] != 5: continue
        mod[tag]['gpp'][:, :, n] = R['gpp'][:, :, 0]; mod[tag]['et'][:, :, n] = R['et'][:, :, 0]; mod[tag]['nee'][:, :, n] = ne[:, :, 0]; got[tag] += 1
print('towers with a series:', got, flush=True)
obs = {'gpp': D['obs_gpp'], 'et': D['obs_et'], 'nee': TN['nee']}
VAR = {'gpp': dict(name='GPP', unit='gC m-2 month-1', lim=(0, 600), zero=True), 'et': dict(name='ET', unit='mm month-1', lim=(0, 200), zero=True), 'nee': dict(name='NEE', unit='gC m-2 month-1 (negative = uptake)', lim=(-300, 150), zero=False)}
def pooled(g, o):
    ok = np.isfinite(o) & np.isfinite(g); g, o = g[ok], o[ok]
    return dict(n=int(ok.sum()), r=np.corrcoef(g, o)[0, 1], nse=1 - ((g - o) ** 2).sum() / ((o - o.mean()) ** 2).sum(), rmse=np.sqrt(((g - o) ** 2).mean()), bias=(g - o).mean())
def annual_ratio(g, o):
    ok = np.isfinite(o) & np.isfinite(g); return g[ok].sum() / o[ok].sum() if ok.sum() >= 6 and o[ok].sum() != 0 else np.nan
summary = []
for v, X in VAR.items():
    name, unit, o = X['name'], X['unit'], obs[v]
    # 1. seasonal cycle by class
    fig, axs = plt.subplots(3, 3, figsize=(11, 8.5), sharex=True); axs = axs.ravel()
    for a, c in zip(axs, CLS):
        idx = [n for n, s in enumerate(sites) if igbp[s] == c]; mo = np.arange(1, 13)
        cyc = lambda arr: np.nanmean([np.roll(np.nanmean(arr[K, :, n], axis=0), 6 if lat[sites[n]] < 0 else 0) for n in idx], axis=0)
        t = cyc(o)
        if not X['zero']: a.axhline(0, color='0.6', lw=0.6)
        a.fill_between(mo, 0, t, color='0.9'); a.plot(mo, t, 'k-', lw=2.2, label='tower')
        for tag, lab, col, ls, lw in RUNS: a.plot(mo, cyc(mod[tag][v]), color=col, ls=ls, lw=lw, label=lab)
        nsh = sum(lat[sites[n]] < 0 for n in idx); a.set_title(f'{CLSNAME[c]} ({len(idx)} tower{"s" if len(idx) > 1 else ""})' + (f'\n{nsh} southern-hemisphere tower{"s" if nsh > 1 else ""} shifted by 6 months' if nsh else ''), fontsize=8.5, loc='left')
        a.set_xticks([1, 4, 7, 10]); a.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
        if X['zero']: a.set_ylim(bottom=0)
    axs[0].legend(fontsize=7, frameon=False, loc='upper left' if X['zero'] else 'lower left'); [a.set_ylabel(f'{name} ({unit.split(" (")[0]})') for a in axs[::3]]
    fig.suptitle(f'Mean seasonal cycle of {name} by class, 2009-2012: tower vs IBIS default vs per-tower GPP/ET fit vs per-tower NEE fit', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/{name}_1_seasonal_cycle_by_class.png', dpi=150); plt.close(fig)
    # 2. example towers
    fig, axs = plt.subplots(4, 3, figsize=(13, 11), sharex=True); axs = axs.ravel(); t = np.arange(48) / 12 + 2009
    for a, s in zip(axs, EX):
        n = sites.index(s); oo = o[K, :, n].ravel()
        if not X['zero']: a.axhline(0, color='0.6', lw=0.6)
        a.fill_between(t, 0, np.where(np.isfinite(oo), oo, 0), color='0.9'); a.plot(t, oo, 'k-', lw=1.8, label='tower')
        rat = []
        for tag, lab, col, ls, lw in RUNS:
            a.plot(t, mod[tag][v][K, :, n].ravel(), color=col, ls=ls, lw=lw, label=lab); rat.append(f'{annual_ratio(mod[tag][v][[4], :, n], o[[4], :, n]):.2f}')
        a.axvspan(2012, 2013, color='0.5', alpha=0.07)
        a.set_title(f'{s}, {CLSNAME[igbp[s]]}\n' + (f'held-out 2012, model/tower: default {rat[0]}, GPP/ET fit {rat[1]}, NEE fit {rat[2]}' if v != 'nee' else 'NEE: shaded = held-out 2012'), fontsize=8, loc='left')
        a.set_xlim(2009, 2013); a.set_xticks([2009, 2010, 2011, 2012, 2013])
        if X['zero']: a.set_ylim(bottom=0)
    axs[0].legend(fontsize=7, frameon=False); [a.set_ylabel(f'{name} ({unit.split(" (")[0]})') for a in axs[::3]]
    fig.suptitle(f'Monthly {name} at twelve towers, 2009-2012: tower, IBIS default, per-tower GPP/ET fit (previous), per-tower NEE fit (current); shaded = held-out year', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/{name}_2_example_towers.png', dpi=150); plt.close(fig)
    # 3. pooled scatter (three panels) + r by class
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.6), gridspec_kw={'width_ratios': [1, 1, 1, 1.15]}); lo, hi = X['lim']; rows = []
    for a, (tag, lab, col, ls, lw), letter in zip(axs[:3], RUNS, 'abc'):
        G = mod[tag][v][K].reshape(-1, ns); O = o[K].reshape(-1, ns)
        for c in CLS:
            idx = [n for n, s in enumerate(sites) if igbp[s] == c]; a.scatter(O[:, idx].ravel(), G[:, idx].ravel(), s=7, color=CC[c], alpha=0.6, lw=0)
        m = pooled(G.ravel(), O.ravel()); a.plot([lo, hi], [lo, hi], color='0.4', lw=0.8)
        if not X['zero']: a.axhline(0, color='0.8', lw=0.5); a.axvline(0, color='0.8', lw=0.5)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect('equal'); a.set_xlabel(f'tower {name} ({unit.split(" (")[0]})'); a.set_ylabel(f'model {name}')
        a.set_title(f'{letter}. {lab}\nr = {m["r"]:.3f}, NSE = {m["nse"]:.2f}, RMSE = {m["rmse"]:.0f}, bias = {m["bias"]:+.0f}', loc='left', fontsize=9); summary.append(dict(var=name, run=tag, **m))
    for c in CLS:
        idx = [n for n, s in enumerate(sites) if igbp[s] == c]; O = o[K][:, :, idx].ravel(); rows.append(dict(cls=c, n=len(idx), **{f'r_{tag}': pooled(mod[tag][v][K][:, :, idx].ravel(), O)['r'] for tag, *_ in RUNS}))
    Rc = pd.DataFrame(rows); a = axs[3]; y = np.arange(len(Rc)); w = 0.27
    for j, (tag, lab, col, ls, lw) in enumerate(RUNS):
        a.barh(y + (j - 1) * w, Rc[f'r_{tag}'], w, color=col if tag != 'def' else 'white', edgecolor=col, lw=1.2, label=lab)
        for i, r in Rc.iterrows(): a.text(max(r[f'r_{tag}'], 0) + 0.01, i + (j - 1) * w, f'{r[f"r_{tag}"]:.2f}', va='center', fontsize=6.5, color=col)
    a.set_yticks(y); a.set_yticklabels([f'{c} ({n})' for c, n in zip(Rc.cls, Rc.n)]); a.invert_yaxis(); a.set_xlim(min(0, Rc[[f'r_{t}' for t, *_ in RUNS]].min().min() - 0.05), 1.15); a.set_xticks([0, 0.25, 0.5, 0.75, 1])
    a.set_xlabel(f"r of monthly {name}, pooled over the class's tower-months"); a.legend(fontsize=7.5, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.02)); a.set_title('d. correlation by class (number of towers)', loc='left', fontsize=9, pad=48)
    handles = [Line2D([], [], marker='o', ls='', color=CC[c], ms=7, label=f'{c}: {CLSNAME[c]}') for c in CLS] + [Line2D([], [], color='0.4', lw=0.8, label='1:1 line')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.4, 0.0), title=f'{ns} towers, {pooled(mod["def"][v][K].ravel(), o[K].ravel())["n"]} tower-months (2009-2012), coloured by IGBP class', title_fontsize=8)
    fig.suptitle(f'Monthly {name}, model vs tower, all tower-months: IBIS default, per-tower GPP/ET fit, per-tower NEE fit', fontsize=10, x=0.01, ha='left'); fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(f'{OUT}/{name}_3_pooled_scatter_and_r_by_class.png', dpi=150); plt.close(fig)
    Rc.to_csv(f'{OUT}/{name}_r_by_class.csv', index=False)
S = pd.DataFrame(summary); S.to_csv(f'{OUT}/pooled_metrics.csv', index=False); print(S.round(3).to_string(index=False)); print('figures in', OUT)
