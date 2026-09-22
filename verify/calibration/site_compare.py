"""Compare parameter sets at the towers: read every site_runs/<par>_<window>/site_metrics.csv, merge per tower, summarise by class, draw the summary.

    ibis-python verify/calibration/site_compare.py <site_runs dir> <out png> [par1,par2,...]
"""
import sys, glob, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
root = sys.argv[1]; out = sys.argv[2]; pars = sys.argv[3].split(',') if len(sys.argv) > 3 else ['default', 'v12g']
frames = []
for f in glob.glob(f'{root}/*/site_metrics.csv'):
    tag = os.path.basename(os.path.dirname(f)); par = tag.split('_')[0]; win = tag[len(par) + 1:]
    if par not in pars: continue
    d = pd.read_csv(f); d['par'] = par; d['window'] = win; frames.append(d)
d = pd.concat(frames, ignore_index=True)
# prefer the _v2 rerun of a window when it exists
d['v2'] = d['window'].str.endswith('_v2'); d['base_window'] = d['window'].str.replace('_v2', '', regex=False)
d = d.sort_values('v2').drop_duplicates(['site', 'par', 'base_window'], keep='last')
if 'usable' in d: d = d[d['usable'].astype(str) != 'False']
d = d[d['gpp_tower'] > 0.1].copy(); d['lag'] = d['peak_model'] - d['peak_tower']
cls_order = ['ENF', 'DBF', 'MF', 'EBF', 'GRA', 'OSH', 'CSH', 'SAV', 'WSA']
print(f'{len(d)} tower-runs, {d["site"].nunique()} towers, parameter sets {sorted(d["par"].unique())}')
for par in pars:
    x = d[d['par'] == par]
    if len(x) == 0: continue
    g = x.groupby('igbp').agg(n=('site', 'count'), gpp_ratio=('gpp_ratio', 'median'), r=('gpp_r', 'median'), lag=('lag', 'median'), winter_model=('winter_share_model', 'median'), winter_tower=('winter_share_tower', 'median'), et_ratio=('et_ratio', 'median'), lai_model=('lai_summer_model', 'median'), lai_modis=('lai_modis_max', 'median')).reindex([c for c in cls_order if c in x['igbp'].unique()])
    print(f'\n== {par}: n {len(x)}, median GPP ratio {x["gpp_ratio"].median():.2f}, within 0.8-1.25: {((x["gpp_ratio"] > 0.8) & (x["gpp_ratio"] < 1.25)).mean():.2f}, median r {x["gpp_r"].median():.2f}, late >= 1 mo: {(x["lag"] >= 1).mean():.2f}, ET ratio {x["et_ratio"].median():.2f}'); print(g.round(2).to_string())
# figure: per-tower GPP ratio by class, one marker per parameter set
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
cols = {'default': 'tab:red', 'v12g': 'tab:blue', 'v12h': 'tab:green'}
for k, (var, lab, ref) in enumerate((('gpp_ratio', 'annual GPP model / tower (log scale)', 1.0), ('lag', 'peak month: model minus tower', 0.0), ('winter_share_model', 'cold-season share of annual GPP (model; tower in grey)', None))):
    ax = axs[k]; classes = [c for c in cls_order if c in d['igbp'].unique()]
    for ci, c in enumerate(classes):
        for par in pars:
            x = d[(d['igbp'] == c) & (d['par'] == par)]
            if len(x) == 0: continue
            jitter = (pars.index(par) - (len(pars) - 1) / 2) * 0.18
            ax.scatter(np.full(len(x), ci + jitter), x[var], s=14, color=cols.get(par, 'k'), alpha=0.7, label=par if ci == 0 else None)
        if var == 'winter_share_model':
            x = d[(d['igbp'] == c)]; ax.scatter(np.full(len(x), ci), x['winter_share_tower'], s=30, marker='_', color='0.4', label='tower' if ci == 0 else None)
    if ref is not None: ax.axhline(ref, color='k', lw=0.7)
    if var == 'gpp_ratio': ax.set_yscale('log'); ax.axhspan(0.8, 1.25, color='0.9', zorder=0)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes); ax.set_title(lab, fontsize=9.5, loc='left'); ax.legend(frameon=False, fontsize=8)
fig.suptitle(f'Towers by vegetation class: {d["site"].nunique()} towers, corrected model, tower meteorology, prescribed composition; parameter sets {", ".join(pars)}', fontsize=10, x=0.01, ha='left')
fig.tight_layout(); fig.savefig(out, dpi=140); print('figure', out)
