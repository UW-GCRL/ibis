"""Site-level state of the calibration as nine figures: for each of GPP, ET and NEE, (1) the mean seasonal cycle by class,
(2) twelve example towers month by month, (3) the pooled model-vs-tower scatter with r by class (CALIBRATION.md 34-35).

    python verify/calibration/diagnostics/site_level_figures.py <out dir>

Series: default parameters = site_runs/latfix_sunshade_2008-2012_attrsmodis (GPP, ET, with the tower observations) and
site_runs/npp_default_2008-2012 (NEE); robust S1 = site_fit/s1_robust_r1/best_series.npz; per-tower fits =
site_fit/s2/<SITE> with a finished run.log (2012 held out); tower NEE = <out dir>/tower_nee_2008-2012.npz
(NEE_VUT_REF, cached by the harness). Years 2009-2012. Southern-hemisphere towers are shifted by 6 months in the
class means. NEE is reported, not fitted (the soil carbon pools are the grid's, not the site's).
The earlier one-off figures (annual ratio per tower, metrics, per-tower fits so far, identifiability, ET per tower)
are in site_level_figures_extra.py.
"""
import sys, os, glob, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = sys.argv[1]; V = '/shared/nas2/Hangkai/ibis/verify_outputs'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})
C = {'tower': 'k', 'def': '#d95f02', 's1': '#1b9e77', 's2': '#7570b3'}
LAB = {'tower': 'tower', 'def': 'default parameters', 's1': 'robust S1 (one set for all towers)', 's2': 'per-tower fit (2012 held out)'}
D = np.load(f'{V}/site_runs/latfix_sunshade_2008-2012_attrsmodis/site_runs.npz'); S = np.load(f'{V}/site_fit/s1_robust_r1/best_series.npz', allow_pickle=True)
sites = list(D['sites']); assert sites == list(S['sites']); K = [1, 2, 3, 4]; YRS = [2009, 2010, 2011, 2012]
inv = pd.read_csv('/shared/nas2/Hangkai/ibis-port/verify/calibration/site_data/site_inventory.csv').set_index('site')
igbp = {s: inv.loc[s, 'igbp'] for s in sites}; lat = {s: float(inv.loc[s, 'lat']) for s in sites}
CLS = ['ENF', 'DBF', 'MF', 'EBF', 'GRA', 'OSH', 'CSH', 'SAV', 'WSA']
CLSNAME = {'ENF': 'evergreen needleleaf forest', 'DBF': 'deciduous broadleaf forest', 'MF': 'mixed forest', 'EBF': 'evergreen broadleaf forest', 'GRA': 'grassland', 'OSH': 'open shrubland', 'CSH': 'closed shrubland', 'SAV': 'savanna', 'WSA': 'woody savanna'}
CC = dict(zip(CLS, plt.cm.tab10(np.arange(len(CLS)))))
EX = ['DE-Tha', 'US-Ha1', 'FI-Hyy', 'US-PFa', 'AT-Neu', 'AU-Stp', 'FR-Pue', 'GF-Guy', 'ES-LJu', 'GL-ZaH', 'CA-TP1', 'US-Var']
# per-tower fits with a finished run
s2 = {}
for f in sorted(glob.glob(f'{V}/site_fit/s2/*/run.log')):
    s = f.split('/')[-2]; d = os.path.dirname(f)
    if 'BEST eval' in open(f).read() and os.path.exists(f'{d}/best_series.npz'):
        R = np.load(f'{d}/best_series.npz', allow_pickle=True); s2[s] = dict(gpp=R['gpp'][:, :, 0], et=R['et'][:, :, 0], nee=R['nee'][:, :, 0])
# variables: obs (5,12,62), model series by tag, units, axis limits
VAR = {'gpp': dict(name='GPP', unit='gC m-2 month-1', obs=D['obs_gpp'], mod={'def': D['gpp'], 's1': S['gpp']}, lim=(0, 600), zero=True),
       'et': dict(name='ET', unit='mm month-1', obs=D['obs_et'], mod={'def': D['et'], 's1': S['et']}, lim=(0, 200), zero=True)}
fn = f'{V}/site_runs/npp_default_2008-2012/site_runs.npz'; ft = f'{OUT}/tower_nee_2008-2012.npz'
if os.path.exists(fn) and 'nee' in np.load(fn).files and os.path.exists(ft):
    TN = np.load(ft); assert list(TN['sites']) == sites
    VAR['nee'] = dict(name='NEE', unit='gC m-2 month-1, negative = net uptake', obs=TN['nee'], mod={'def': np.load(fn)['nee'], 's1': S['nee']}, lim=(-300, 150), zero=False)
else: print('NEE skipped: default NEE run or tower NEE cache missing')
def pooled(g, o):
    ok = np.isfinite(o) & np.isfinite(g); g, o = g[ok], o[ok]
    return dict(n=int(ok.sum()), r=np.corrcoef(g, o)[0, 1], nse=1 - ((g - o) ** 2).sum() / ((o - o.mean()) ** 2).sum(), rmse=np.sqrt(((g - o) ** 2).mean()), bias=(g - o).mean())
def annual_ratio(g, o):
    ok = np.isfinite(o); return g[ok].sum() / o[ok].sum() if ok.sum() >= 6 and o[ok].sum() != 0 else np.nan
summary = []
for v, X in VAR.items():
    obs, mod, name, unit = X['obs'], X['mod'], X['name'], X['unit']
    # ---- 1. seasonal cycle by class
    fig, axs = plt.subplots(3, 3, figsize=(11, 8.5), sharex=True); axs = axs.ravel()
    for a, c in zip(axs, CLS):
        idx = [n for n, s in enumerate(sites) if igbp[s] == c]; cyc = {}
        for tag, arr in (('tower', obs), ('def', mod['def']), ('s1', mod['s1'])):
            mm = []
            for n in idx:
                x = np.nanmean(arr[K, :, n], axis=0); mm.append(np.roll(x, 6) if lat[sites[n]] < 0 else x)
            cyc[tag] = np.nanmean(mm, axis=0)
        mo = np.arange(1, 13)
        if not X['zero']: a.axhline(0, color='0.6', lw=0.6)
        a.fill_between(mo, 0, cyc['tower'], color='0.9'); a.plot(mo, cyc['tower'], color=C['tower'], lw=2, label=LAB['tower']); a.plot(mo, cyc['def'], color=C['def'], lw=1.5, ls='--', label=LAB['def']); a.plot(mo, cyc['s1'], color=C['s1'], lw=2, label=LAB['s1'])
        nsh = sum(lat[sites[n]] < 0 for n in idx); a.set_title(f'{CLSNAME[c]} ({len(idx)} tower{"s" if len(idx) > 1 else ""})' + (f'\n{nsh} southern-hemisphere tower{"s" if nsh > 1 else ""} shifted by 6 months' if nsh else ''), fontsize=8.5, loc='left')
        a.set_xticks([1, 4, 7, 10]); a.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
        if X['zero']: a.set_ylim(bottom=0)
    axs[0].legend(fontsize=8, frameon=False, loc='upper left' if X['zero'] else 'lower left'); [a.set_ylabel(f'{name} ({unit.split(",")[0]})') for a in axs[::3]]
    fig.suptitle(f'Mean seasonal cycle of {name} by class, 2009-2012: tower vs default parameters vs robust S1' + ('' if v != 'nee' else ' (NEE reported, not fitted)'), fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/{name}_1_seasonal_cycle_by_class.png', dpi=150); plt.close(fig)
    # ---- 2. example towers
    fig, axs = plt.subplots(4, 3, figsize=(13, 11), sharex=True); axs = axs.ravel(); t = np.arange(48) / 12 + 2009
    for a, s in zip(axs, EX):
        n = sites.index(s); o = obs[K, :, n].ravel()
        if not X['zero']: a.axhline(0, color='0.6', lw=0.6)
        a.fill_between(t, 0, np.where(np.isfinite(o), o, 0), color='0.9'); a.plot(t, o, color=C['tower'], lw=1.8, label=LAB['tower'])
        a.plot(t, mod['def'][K, :, n].ravel(), color=C['def'], lw=1.2, ls='--', label=LAB['def']); a.plot(t, mod['s1'][K, :, n].ravel(), color=C['s1'], lw=1.6, label=LAB['s1'])
        rat = f'default {annual_ratio(mod["def"][K, :, n], obs[K, :, n]):.2f}, S1 {annual_ratio(mod["s1"][K, :, n], obs[K, :, n]):.2f}'
        if s in s2:
            a.plot(t, s2[s][v][K].ravel(), color=C['s2'], lw=1.6, label=LAB['s2']); a.axvspan(2012, 2013, color=C['s2'], alpha=0.07)
            rat += f', fit {annual_ratio(s2[s][v][[4]], obs[[4], :, n]):.2f} (2012, held out)'
        a.set_title(f'{s}, {CLSNAME[igbp[s]]}\n' + (f'annual model/tower: {rat}' if v != 'nee' else 'NEE reported, not fitted'), fontsize=8, loc='left')
        a.set_xlim(2009, 2013); a.set_xticks([2009, 2010, 2011, 2012, 2013])
        if X['zero']: a.set_ylim(bottom=0)
    (axs[[i for i, s in enumerate(EX) if s in s2][0]] if any(s in s2 for s in EX) else axs[0]).legend(fontsize=7, frameon=False); [a.set_ylabel(f'{name} ({unit.split(",")[0]})') for a in axs[::3]]
    fig.suptitle(f'Monthly {name} at twelve towers, 2009-2012: tower, default parameters, robust S1, and the per-tower fit where done', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/{name}_2_example_towers.png', dpi=150); plt.close(fig)
    # ---- 3. pooled scatter and r by class
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 5.6), gridspec_kw={'width_ratios': [1, 1, 1.1]}); lo, hi = X['lim']
    for a, tag in ((axs[0], 'def'), (axs[1], 's1')):
        G = mod[tag][K].reshape(-1, len(sites)); O = obs[K].reshape(-1, len(sites))
        for c in CLS:
            idx = [n for n, s in enumerate(sites) if igbp[s] == c]; a.scatter(O[:, idx].ravel(), G[:, idx].ravel(), s=8, color=CC[c], alpha=0.6, lw=0)
        m = pooled(G.ravel(), O.ravel()); a.plot([lo, hi], [lo, hi], color='0.4', lw=0.8, label='1:1 line')
        if not X['zero']: a.axhline(0, color='0.8', lw=0.5); a.axvline(0, color='0.8', lw=0.5)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect('equal'); a.set_xlabel(f'tower {name} ({unit.split(",")[0]})'); a.set_ylabel(f'model {name} ({unit.split(",")[0]})')
        a.set_title(f'{"a" if tag == "def" else "b"}. {LAB[tag]}\nr = {m["r"]:.3f}, NSE = {m["nse"]:.2f}, RMSE = {m["rmse"]:.0f}, bias = {m["bias"]:+.0f}', loc='left', fontsize=9)
        summary.append(dict(var=name, run=tag, **m))
    handles = [Line2D([], [], marker='o', ls='', color=CC[c], ms=7, label=f'{c}: {CLSNAME[c]}') for c in CLS] + [Line2D([], [], color='0.4', lw=0.8, label='1:1 line')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.36, 0.0), title=f'{len(sites)} towers, {pooled(mod["s1"][K].ravel(), obs[K].ravel())["n"]} tower-months (2009-2012), coloured by IGBP class', title_fontsize=8)
    a = axs[2]; rows = []
    for c in CLS:
        idx = [n for n, s in enumerate(sites) if igbp[s] == c]; O = obs[K][:, :, idx].ravel()
        rows.append(dict(cls=c, n=len(idx), r_def=pooled(mod['def'][K][:, :, idx].ravel(), O)['r'], r_s1=pooled(mod['s1'][K][:, :, idx].ravel(), O)['r']))
    Rc = pd.DataFrame(rows); y = np.arange(len(Rc))
    a.barh(y - 0.2, Rc.r_def, 0.4, color='white', edgecolor=C['def'], lw=1.3, label=LAB['def']); a.barh(y + 0.2, Rc.r_s1, 0.4, color=C['s1'], label=LAB['s1'])
    for i, r in Rc.iterrows(): a.text(max(r.r_def, 0) + 0.01, i - 0.2, f'{r.r_def:.2f}', va='center', fontsize=7, color=C['def']); a.text(max(r.r_s1, 0) + 0.01, i + 0.2, f'{r.r_s1:.2f}', va='center', fontsize=7, color=C['s1'])
    a.set_yticks(y); a.set_yticklabels([f'{c} ({n})' for c, n in zip(Rc.cls, Rc.n)]); a.invert_yaxis(); a.set_xlim(min(0, Rc[['r_def', 'r_s1']].min().min() - 0.05), 1.15); a.set_xticks([0, 0.25, 0.5, 0.75, 1])
    a.set_xlabel(f'r of monthly {name}, pooled over the class\'s tower-months'); a.legend(fontsize=8, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.02)); a.set_title('c. correlation by class (number of towers)', loc='left', fontsize=9, pad=42)
    fig.suptitle(f'Monthly {name}, model vs tower, all tower-months: one parameter set for all towers' + ('' if v != 'nee' else ' (NEE reported, not fitted)'), fontsize=10, x=0.01, ha='left'); fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(f'{OUT}/{name}_3_pooled_scatter_and_r_by_class.png', dpi=150); plt.close(fig)
    Rc.to_csv(f'{OUT}/{name}_r_by_class.csv', index=False)
pd.DataFrame(summary).to_csv(f'{OUT}/pooled_metrics.csv', index=False); print(pd.DataFrame(summary).round(3).to_string(index=False)); print(f'per-tower fits finished: {len(s2)}; figures in {OUT}')
