"""Stage-0 check of the class-median parameter set (CALIBRATION.md 37): the 62-tower site run on params_s3_classmedian.yaml
against the default run, the per-tower GPP/ET fits and the per-tower NEE fits, pooled and by class.

    python verify/calibration/diagnostics/s3_check_report.py [s3 site_runs dir] [nee fit folder]
"""
import sys, os, numpy as np, pandas as pd
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; s3dir = sys.argv[1] if len(sys.argv) > 1 else f'{V}/site_runs/s3_classmedian_2008-2012'; NEE_DIR = sys.argv[2] if len(sys.argv) > 2 else 'nee_v2'
D = np.load(f'{V}/site_runs/npp_default_2008-2012/site_runs.npz'); S3 = np.load(f'{s3dir}/site_runs.npz'); sites = list(D['sites']); ns = len(sites); K = [1, 2, 3, 4]
assert list(S3['sites']) == sites
inv = pd.read_csv('verify/calibration/site_data/site_inventory.csv').set_index('site'); igbp = {s: inv.loc[s, 'igbp'] for s in sites}
TN = np.load('/shared/nas2/Hangkai/ibis_local_document/figures/site_level/tower_nee_2008-2012.npz'); assert list(TN['sites']) == sites
nan = lambda: np.full((5, 12, ns), np.nan)
mod = {'default': {'gpp': D['gpp'], 'et': D['et'], 'nee': D['nee']}, 'class medians': {'gpp': S3['gpp'], 'et': S3['et'], 'nee': S3['nee']},
       'per-tower GPP/ET': {'gpp': nan(), 'et': nan(), 'nee': nan()}, 'per-tower NEE': {'gpp': nan(), 'et': nan(), 'nee': nan()}}
for n, s in enumerate(sites):
    for tag, d in (('per-tower GPP/ET', 's2'), ('per-tower NEE', NEE_DIR)):
        f = f'{V}/site_fit/{d}/{s}/best_series.npz'
        if not os.path.exists(f): continue
        R = np.load(f, allow_pickle=True); ne = R['nee']; ne = ne.sum(axis=2) if ne.ndim == 4 else ne
        if R['gpp'].shape[0] != 5: continue
        mod[tag]['gpp'][:, :, n] = R['gpp'][:, :, 0]; mod[tag]['et'][:, :, n] = R['et'][:, :, 0]; mod[tag]['nee'][:, :, n] = ne[:, :, 0]
obs = {'gpp': D['obs_gpp'], 'et': D['obs_et'], 'nee': TN['nee']}
def metrics(m, o):
    ok = np.isfinite(m) & np.isfinite(o); m, o = m[ok], o[ok]
    if len(m) < 10: return dict(n=len(m), r=np.nan, nse=np.nan, rmse=np.nan, bias=np.nan)
    return dict(n=len(m), r=np.corrcoef(m, o)[0, 1], nse=1 - np.sum((m - o) ** 2) / np.sum((o - o.mean()) ** 2), rmse=np.sqrt(np.mean((m - o) ** 2)), bias=np.mean(m - o))
rows = []
for v in ('gpp', 'et', 'nee'):
    for tag in mod: rows.append(dict(var=v.upper(), run=tag, **metrics(mod[tag][v][K], obs[v][K])))
T = pd.DataFrame(rows); pd.set_option('display.width', 200); print('pooled over 62 towers x 2009-2012 (gC m-2 month-1 / mm month-1):'); print(T.round(3).to_string(index=False))
print('\nr by class (towers):')
cls = sorted(set(igbp.values()), key=lambda c: -sum(1 for s in sites if igbp[s] == c))
for v in ('gpp', 'et', 'nee'):
    line = []
    for c in cls:
        idx = [n for n, s in enumerate(sites) if igbp[s] == c]
        line.append(f'{c}({len(idx)}) ' + '/'.join(f"{metrics(mod[t][v][K][:, :, idx], obs[v][K][:, :, idx])['r']:.2f}" for t in mod))
    print(f'{v.upper():4s} default/class-median/per-tower GPP-ET/per-tower NEE: ' + '  '.join(line))
# per-tower annual ratio model/tower for GPP: class medians vs per-tower fits
rat = lambda m: np.nansum(m[K], axis=(0, 1)) / np.nansum(obs['gpp'][K], axis=(0, 1))
r3, r2 = rat(mod['class medians']['gpp']), rat(mod['per-tower GPP/ET']['gpp']); ok = np.isfinite(r3) & np.isfinite(r2)
print(f'\nannual GPP model/tower, towers within 25 %: class medians {np.mean(np.abs(r3[ok] - 1) <= 0.25):.2f}, per-tower {np.mean(np.abs(r2[ok] - 1) <= 0.25):.2f}, default {np.mean(np.abs(rat(mod["default"]["gpp"])[ok] - 1) <= 0.25):.2f}')
