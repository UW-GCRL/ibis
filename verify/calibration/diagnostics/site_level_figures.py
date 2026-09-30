"""Site-level state of the calibration, as figures (CALIBRATION.md 34-35): the 62-tower batch with default parameters,
the robust S1 set, and the per-tower fits done so far.

    python verify/calibration/diagnostics/site_level_figures.py <out dir>

Inputs: site_runs/latfix_sunshade_2008-2012_attrsmodis (defaults, with the tower observations), site_fit/s1_robust_r1
(S1), site_fit/s2/<SITE> (per-tower fits with a finished run.log), site_fit/s1_robust_profile/profile.csv.
Scored years 2009-2012; the per-tower fits trained on 2009-2011 with 2012 held out.
"""
import sys, os, glob, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.ticker import NullFormatter
OUT = sys.argv[1]; V = '/shared/nas2/Hangkai/ibis/verify_outputs'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})
C = {'tower': 'k', 'def': '#d95f02', 's1': '#1b9e77', 's2': '#7570b3'}
D = np.load(f'{V}/site_runs/latfix_sunshade_2008-2012_attrsmodis/site_runs.npz'); S = np.load(f'{V}/site_fit/s1_robust_r1/best_series.npz', allow_pickle=True)
sites = list(D['sites']); assert sites == list(S['sites']); ny = len(D['years']); Y = list(D['years']); K = [1, 2, 3, 4]      # 2009-2012
inv = pd.read_csv('/shared/nas2/Hangkai/ibis-port/verify/calibration/site_data/site_inventory.csv').set_index('site')
igbp = {s: inv.loc[s, 'igbp'] for s in sites}; lat = {s: float(inv.loc[s, 'lat']) for s in sites}
CLS = ['ENF', 'DBF', 'MF', 'EBF', 'GRA', 'OSH', 'CSH', 'SAV', 'WSA']; CLSNAME = {'ENF': 'evergreen needleleaf', 'DBF': 'deciduous broadleaf', 'MF': 'mixed forest', 'EBF': 'evergreen broadleaf', 'GRA': 'grassland', 'OSH': 'open shrubland', 'CSH': 'closed shrubland', 'SAV': 'savanna', 'WSA': 'woody savanna'}
obs, obs_et = D['obs_gpp'], D['obs_et']; runs = {'def': (D['gpp'], D['et']), 's1': (S['gpp'], S['et'])}
# per-tower fits with a finished run
s2 = {}
for f in sorted(glob.glob(f'{V}/site_fit/s2/*/run.log')):
    s = f.split('/')[-2]
    if 'BEST eval' in open(f).read() and os.path.exists(f'{os.path.dirname(f)}/best_series.npz'):
        R = np.load(f'{os.path.dirname(f)}/best_series.npz', allow_pickle=True); b = json.load(open(f'{os.path.dirname(f)}/best.json'))
        base = [l for l in open(f) if l.startswith('baseline loss')]; s2[s] = dict(gpp=R['gpp'][:, :, 0], et=R['et'][:, :, 0], loss=b['loss'], base=float(base[0].split()[-1]) if base else np.nan, params=b['params'])
def stats(g, o, ky):
    """model / tower annual and summer GPP, r and NSE of monthly GPP over the year indices ky."""
    g, o = g[ky], o[ky]; ok = np.isfinite(o)
    if ok.sum() < 6: return dict(annual=np.nan, r=np.nan, nse=np.nan)
    return dict(annual=g[ok].sum() / o[ok].sum(), r=np.corrcoef(g[ok], o[ok])[0, 1], nse=1 - ((g[ok] - o[ok]) ** 2).sum() / ((o[ok] - o[ok].mean()) ** 2).sum())
T = []
for n, s in enumerate(sites):
    row = dict(site=s, igbp=igbp[s])
    for tag, (g, e) in runs.items():
        for k, v in stats(g[:, :, n], obs[:, :, n], K).items(): row[f'{k}_{tag}'] = v
    if s in s2:
        for lab, ky in (('train', [1, 2, 3]), ('hold', [4])):
            for k, v in stats(s2[s]['gpp'], obs[:, :, n], ky).items(): row[f'{k}_s2_{lab}'] = v
            for k, v in stats(S['gpp'][:, :, n], obs[:, :, n], ky).items(): row[f'{k}_s1_{lab}'] = v
    T.append(row)
T = pd.DataFrame(T); T['cls'] = pd.Categorical(T.igbp, CLS, ordered=True); T = T.sort_values(['cls', 'annual_s1']).reset_index(drop=True); T.to_csv(f'{OUT}/site_level_table.csv', index=False)
within = lambda a: np.mean(np.abs(np.log(a.dropna())) < np.log(1.25))
# ---- F1: annual GPP model / tower per tower, defaults -> robust S1
fig, ax = plt.subplots(figsize=(7.5, 11)); y = np.arange(len(T))
ax.axvspan(0.8, 1.25, color='0.92', zorder=0); ax.axvline(1, color='0.5', lw=0.8)
for i, r in T.iterrows():
    ax.plot([r.annual_def, r.annual_s1], [i, i], color='0.7', lw=1, zorder=1)
ax.scatter(T.annual_def, y, s=22, facecolor='white', edgecolor=C['def'], lw=1.2, label=f'default parameters ({within(T.annual_def):.0%} within 25 %)', zorder=2)
ax.scatter(T.annual_s1, y, s=26, color=C['s1'], label=f'robust S1 set ({within(T.annual_s1):.0%} within 25 %)', zorder=3)
ax.set_yticks(y); ax.set_yticklabels(T.site, fontsize=7); ax.set_ylim(-1, len(T)); ax.invert_yaxis(); ax.set_xscale('log'); ax.set_xlim(0.25, 4)
ax.set_xticks([0.25, 0.5, 0.8, 1, 1.25, 2, 4]); ax.set_xticklabels(['0.25', '0.5', '0.8', '1', '1.25', '2', '4']); ax.xaxis.set_minor_formatter(NullFormatter())
prev = None
for i, c in enumerate(T.igbp):
    if c != prev: ax.axhline(i - 0.5, color='0.8', lw=0.6); ax.text(4.2, i - 0.3, CLSNAME[c], fontsize=8, va='top', color='0.3', clip_on=False); prev = c
ax.set_xlabel('annual GPP, model / tower (2009-2012)'); ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False, ncol=2)
ax.set_title('Annual GPP at the 62-tower batch: default IBIS parameters vs the robust S1 set', loc='left', fontsize=10, pad=22)
fig.tight_layout(); fig.savefig(f'{OUT}/F1_annual_ratio_by_tower.png', dpi=150); plt.close(fig)
# ---- F2: mean seasonal cycle by class (southern towers shifted by 6 months)
fig, axs = plt.subplots(3, 3, figsize=(11, 8.5), sharex=True); axs = axs.ravel()
for a, c in zip(axs, CLS):
    idx = [n for n, s in enumerate(sites) if igbp[s] == c]; cyc = {}
    for tag, arr in (('tower', obs), ('def', runs['def'][0]), ('s1', runs['s1'][0])):
        m = []
        for n in idx:
            v = np.nanmean(arr[K, :, n], axis=0); m.append(np.roll(v, 6) if lat[sites[n]] < 0 else v)
        cyc[tag] = np.nanmean(m, axis=0)
    m = np.arange(1, 13); a.fill_between(m, 0, cyc['tower'], color='0.9'); a.plot(m, cyc['tower'], color=C['tower'], lw=2, label='tower')
    a.plot(m, cyc['def'], color=C['def'], lw=1.5, ls='--', label='default'); a.plot(m, cyc['s1'], color=C['s1'], lw=2, label='robust S1')
    nsh = sum(lat[sites[n]] < 0 for n in idx); a.set_title(f'{CLSNAME[c]} ({len(idx)} tower{"s" if len(idx) > 1 else ""})' + (f'\n{nsh} southern-hemisphere tower{"s" if nsh > 1 else ""} shifted by 6 months' if nsh else ''), fontsize=8.5, loc='left')
    a.set_xticks([1, 4, 7, 10]); a.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct']); a.set_ylim(bottom=0)
axs[0].legend(fontsize=8, frameon=False); [a.set_ylabel('GPP (gC m-2 month-1)') for a in axs[::3]]
fig.suptitle('Mean seasonal cycle of GPP by class, 2009-2012: tower vs default parameters vs robust S1', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/F2_seasonal_cycle_by_class.png', dpi=150); plt.close(fig)
# ---- F3: NSE and r per tower by class, default -> S1
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for a, met, lim in ((axs[0], 'nse', (-1.05, 1)), (axs[1], 'r', (0, 1))):
    for j, c in enumerate(CLS):
        sub = T[T.igbp == c]; x0 = j - 0.18; x1 = j + 0.18
        for _, r in sub.iterrows():
            v0, v1 = np.clip(r[f'{met}_def'], *lim), np.clip(r[f'{met}_s1'], *lim); a.plot([x0, x1], [v0, v1], color='0.75', lw=0.7)
        a.scatter([x0] * len(sub), np.clip(sub[f'{met}_def'], *lim), s=16, facecolor='white', edgecolor=C['def'], lw=1, zorder=3)
        a.scatter([x1] * len(sub), np.clip(sub[f'{met}_s1'], *lim), s=18, color=C['s1'], zorder=4)
        a.plot([x0 - 0.1, x0 + 0.1], [sub[f'{met}_def'].median()] * 2, color=C['def'], lw=2.5); a.plot([x1 - 0.1, x1 + 0.1], [sub[f'{met}_s1'].median()] * 2, color=C['s1'], lw=2.5)
    a.set_xticks(range(len(CLS))); a.set_xticklabels(CLS); a.set_ylim(lim); a.axhline(0, color='0.6', lw=0.6) if met == 'nse' else None
    a.set_ylabel({'nse': 'Nash-Sutcliffe efficiency of monthly GPP (clipped at -1)', 'r': 'correlation r of monthly GPP'}[met])
axs[0].set_title('a. NSE per tower: default (open) -> robust S1 (filled); bars = class median', loc='left', fontsize=9); axs[1].set_title('b. correlation', loc='left', fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/F3_metrics_by_class.png', dpi=150); plt.close(fig)
# ---- F4: example towers, monthly 2009-2012
EX = ['DE-Tha', 'US-Ha1', 'FI-Hyy', 'US-PFa', 'AT-Neu', 'AU-Stp', 'FR-Pue', 'GF-Guy', 'ES-LJu', 'GL-ZaH', 'CA-TP1', 'US-Var']
fig, axs = plt.subplots(4, 3, figsize=(13, 11), sharex=True); axs = axs.ravel(); t = np.arange(48) / 12 + 2009
for a, s in zip(axs, EX):
    n = sites.index(s); o = obs[K, :, n].ravel(); a.fill_between(t, 0, np.where(np.isfinite(o), o, 0), color='0.9'); a.plot(t, o, color=C['tower'], lw=1.8, label='tower')
    a.plot(t, runs['def'][0][K, :, n].ravel(), color=C['def'], lw=1.2, ls='--', label='default'); a.plot(t, runs['s1'][0][K, :, n].ravel(), color=C['s1'], lw=1.6, label='robust S1')
    if s in s2: a.plot(t, s2[s]['gpp'][K].ravel(), color=C['s2'], lw=1.6, label='per-tower fit (2012 held out)'); a.axvspan(2012, 2013, color='#7570b3', alpha=0.07)
    r = T.set_index('site').loc[s]; a.set_title(f'{s}, {CLSNAME[igbp[s]]}\nannual model/tower: default {r.annual_def:.2f}, S1 {r.annual_s1:.2f}' + (f', fit {r.annual_s2_hold:.2f} (2012, held out)' if s in s2 and np.isfinite(r.annual_s2_hold) else ''), fontsize=8, loc='left')
    a.set_ylim(bottom=0); a.set_xlim(2009, 2013); a.set_xticks([2009, 2010, 2011, 2012, 2013])
(axs[[i for i, s in enumerate(EX) if s in s2][0]] if any(s in s2 for s in EX) else axs[0]).legend(fontsize=7, frameon=False); [a.set_ylabel('GPP (gC m-2 month-1)') for a in axs[::3]]
fig.suptitle('Monthly GPP at twelve towers, 2009-2012: tower, default parameters, robust S1, and the per-tower fit where done', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/F4_example_towers.png', dpi=150); plt.close(fig)
# ---- F5: per-tower stage so far: loss, and held-out annual ratio S1 -> fit
P = T[T.site.isin(s2)].copy()
if len(P):
    P['base'] = [s2[s]['base'] for s in P.site]; P['loss'] = [s2[s]['loss'] for s in P.site]; P = P.sort_values('base', ascending=False)
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.32 * len(P) + 2.2)); a = axs[0]; y = np.arange(len(P))
    a.barh(y - 0.18, P.base, 0.36, color=C['s1'], label='start (robust S1 set)'); a.barh(y + 0.18, P.loss, 0.36, color=C['s2'], label='after the per-tower fit')
    a.set_yticks(y); a.set_yticklabels([f'{s} ({igbp[s]})' for s in P.site], fontsize=8); a.invert_yaxis(); a.set_xlabel('objective (fit years 2009-2011)'); a.legend(fontsize=8, frameon=False); a.set_title('a. per-tower fit: objective before and after', loc='left', fontsize=9)
    a = axs[1]; a.axvspan(0.8, 1.25, color='0.92'); a.axvline(1, color='0.5', lw=0.8)
    for i, (_, r) in enumerate(P.iterrows()): a.plot([r.annual_s1_hold, r.annual_s2_hold], [i, i], color='0.7', lw=1)
    a.scatter(P.annual_s1_hold, y, s=24, color=C['s1'], label='S1 set'); a.scatter(P.annual_s2_hold, y, s=26, color=C['s2'], label='per-tower fit')
    a.set_yticks(y); a.set_yticklabels([]); a.set_ylim(len(P) - 0.5, -0.5); a.set_xscale('log'); a.set_xlim(0.25, 4); a.set_xticks([0.25, 0.5, 0.8, 1, 1.25, 2, 4]); a.set_xticklabels(['0.25', '0.5', '0.8', '1', '1.25', '2', '4']); a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel('annual GPP, model / tower, HELD-OUT year 2012'); a.legend(fontsize=8, frameon=False); a.set_title(f'b. held-out year: {within(P.annual_s1_hold):.0%} -> {within(P.annual_s2_hold):.0%} within 25 %', loc='left', fontsize=9)
    fig.suptitle(f'Per-tower fits finished so far ({len(P)} of 62), started from the robust S1 set', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/F5_per_tower_fits_so_far.png', dpi=150); plt.close(fig)
# ---- F6: identifiability from the OAT profile
Pr = pd.read_csv(f'{V}/site_fit/s1_robust_profile/profile.csv'); rows = []
for nm, g in Pr.groupby('param', sort=False):
    L = g.loss.values; i = L.argmin(); rows.append(dict(param=nm, rng=L.max() - L.min(), edge='at a bound' if i in (0, len(L) - 1) else ''))
R = pd.DataFrame(rows).sort_values('rng'); fig, a = plt.subplots(figsize=(7.5, 7)); y = np.arange(len(R))
a.barh(y, R.rng, color=[C['s2'] if e else '0.6' for e in R.edge]); a.set_yticks(y); a.set_yticklabels(R.param, fontsize=7.5); a.axvline(1, color='0.4', lw=0.8, ls=':')
a.set_xlabel('loss range along the parameter across its whole bound interval (loss at the best: 14.39)'); a.set_title('Identifiability of the S1 parameters: one-at-a-time profile over 62 towers\n(purple: the minimum sits at a bound; dotted line: 1 unit of a 14.4 loss)', loc='left', fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/F6_identifiability_profile.png', dpi=150); plt.close(fig)
print(f'{len(T)} towers; per-tower fits finished: {len(s2)}; within 25 %: default {within(T.annual_def):.0%}, S1 {within(T.annual_s1):.0%}; figures in {OUT}')
# ---- F7: pooled model vs tower monthly GPP (all tower-months 2009-2012), overall and per-class r
CC = dict(zip(CLS, plt.cm.tab10(np.arange(len(CLS)))))
def pooled(g, o):
    ok = np.isfinite(o) & np.isfinite(g); g, o = g[ok], o[ok]
    return dict(n=int(ok.sum()), r=np.corrcoef(g, o)[0, 1], nse=1 - ((g - o) ** 2).sum() / ((o - o.mean()) ** 2).sum(), rmse=np.sqrt(((g - o) ** 2).mean()), bias=(g - o).mean())
fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={'width_ratios': [1, 1, 1.1]})
for a, tag, lab in ((axs[0], 'def', 'default parameters'), (axs[1], 's1', 'robust S1 set')):
    G = runs[tag][0][K].reshape(-1, len(sites)); O = obs[K].reshape(-1, len(sites))
    for c in CLS:
        idx = [n for n, s in enumerate(sites) if igbp[s] == c]; a.scatter(O[:, idx].ravel(), G[:, idx].ravel(), s=7, color=CC[c], alpha=0.55, lw=0, label=c)
    m = pooled(G.ravel(), O.ravel()); lim = 600; a.plot([0, lim], [0, lim], color='0.4', lw=0.8); a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect('equal')
    a.set_xlabel('tower GPP (gC m-2 month-1)'); a.set_ylabel('model GPP (gC m-2 month-1)')
    a.set_title(f'{lab}: r {m["r"]:.3f}, NSE {m["nse"]:.2f}, RMSE {m["rmse"]:.0f}, bias {m["bias"]:+.0f}\n({m["n"]} tower-months, 62 towers, 2009-2012)', loc='left', fontsize=9)
axs[0].legend(fontsize=7, frameon=False, markerscale=2, loc='upper left', ncol=2)
a = axs[2]; rows = []
for c in CLS:
    idx = [n for n, s in enumerate(sites) if igbp[s] == c]; O = obs[K][:, :, idx].ravel()
    rows.append(dict(cls=c, n=len(idx), r_def=pooled(runs['def'][0][K][:, :, idx].ravel(), O)['r'], r_s1=pooled(runs['s1'][0][K][:, :, idx].ravel(), O)['r'],
                     nse_def=pooled(runs['def'][0][K][:, :, idx].ravel(), O)['nse'], nse_s1=pooled(runs['s1'][0][K][:, :, idx].ravel(), O)['nse']))
Rc = pd.DataFrame(rows); y = np.arange(len(Rc))
a.barh(y - 0.2, Rc.r_def, 0.4, color='white', edgecolor=C['def'], lw=1.2, label='default'); a.barh(y + 0.2, Rc.r_s1, 0.4, color=C['s1'], label='robust S1')
for i, r in Rc.iterrows(): a.text(1.01, i - 0.2, f'{r.r_def:.2f}', va='center', fontsize=7, color=C['def']); a.text(1.01, i + 0.2, f'{r.r_s1:.2f}', va='center', fontsize=7, color=C['s1'])
a.set_yticks(y); a.set_yticklabels([f'{c} ({n})' for c, n in zip(Rc.cls, Rc.n)]); a.invert_yaxis(); a.set_xlim(0, 1.12); a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xlabel('r of monthly GPP, pooled over the class\'s tower-months')
a.legend(fontsize=8, frameon=False, loc='lower left'); a.set_title('correlation by class (pooled)', loc='left', fontsize=9)
fig.suptitle('Monthly GPP, model vs tower, all tower-months: one parameter set for all towers (per-PFT values)', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/F7_pooled_scatter_and_r_by_class.png', dpi=150); plt.close(fig)
Rc.to_csv(f'{OUT}/r_by_class_pooled.csv', index=False); print(Rc.round(3).to_string(index=False))
