"""Per-tower methane fits (CALIBRATION.md 38): held-out verdict, fitted parameters, class and band medians.

    python verify/calibration/diagnostics/ch4_fit_report.py <ch4 fit root> <methane tower dir> <out csv>

Per tower: class, latitude band, years, baseline and best loss, fitted ch4_* values, annual model/tower ratio and monthly r
of the training years and of the held-out (last) year at the fitted parameters (best_series.npz), and the baseline annual
ratio from history.csv's first evaluation where the series exist. Then medians by class and, for WET, by band.
"""
import sys, os, glob, json, numpy as np, pandas as pd
root, M, out = sys.argv[1:4]
I = pd.read_csv(f'{M}/ch4_sites.csv').set_index('SITE_ID'); T = np.load(f'{M}/ch4_targets.npz'); ts = list(T['sites']); ty = [int(y) for y in T['years']]
def band(lat, mat):
    if lat >= 55 or (np.isfinite(mat) and mat < 3): return 'boreal'
    return 'tropical' if abs(lat) < 30 else 'temperate'
rows = []
for d in sorted(glob.glob(f'{root}/*/')):
    s = os.path.basename(d.rstrip('/'))
    if s not in ts or not os.path.exists(f'{d}best.json') or not os.path.exists(f'{d}best_series.npz'): continue
    b = json.load(open(f'{d}best.json')); R = np.load(f'{d}best_series.npz'); yrs = [int(y) for y in R['years']]; k = ts.index(s)
    cm = R['ch4'][:, :, 0] if 'ch4' in R else R['nee'][:, :, 0]; obs = np.array([T['sum_ch4'][ty.index(y), :, k] if y in ty else np.full(12, np.nan) for y in yrs])
    h = pd.read_csv(f'{d}history.csv') if os.path.exists(f'{d}history.csv') else None
    def stats(idx):
        m, o = cm[idx], obs[idx]; ok = np.isfinite(o)
        if ok.sum() < 4: return np.nan, np.nan, np.nan
        return float(m[ok].sum() / o[ok].sum()) if abs(o[ok].sum()) > 0 else np.nan, float(np.corrcoef(m[ok], o[ok])[0, 1]) if ok.sum() > 3 else np.nan, float(o[ok].sum() / ok.sum() * 12)
    tr = list(range(1, len(yrs) - 1)); ho = [len(yrs) - 1]                                  # first year warm-up, last held out
    rt, rr, ot = stats(tr); rh, rrh, oh = stats(ho)
    rows.append(dict(site=s, igbp=I.IGBP.get(s, '?'), band=band(float(I.LOCATION_LAT[s]), float(I.MAT[s])), lat=round(float(I.LOCATION_LAT[s]), 2), years=f'{yrs[0]}-{yrs[-1]}',
                     loss0=float(h.loss.iloc[0]) if h is not None else np.nan, loss=b['loss'], hold_loss=b.get('holdout_year', np.nan), n_eval=b.get('eval'),
                     **{f'p:{k_}': v for k_, v in b['params'].items()}, ratio_train=rt, r_train=rr, ratio_hold=rh, r_hold=rrh, tower_gC_yr=ot))
D = pd.DataFrame(rows); D.to_csv(out, index=False); pd.set_option('display.width', 250)
print(D.round(3).to_string(index=False))
pc = [c for c in D.columns if c.startswith('p:')]
print('\nmedians by class:'); print(D.groupby('igbp')[pc + ['ratio_train', 'r_train', 'ratio_hold', 'r_hold']].median().round(3).assign(n=D.groupby('igbp').size()).to_string())
print('\nWET medians by band:'); W = D[D.igbp == 'WET']; print(W.groupby('band')[pc + ['ratio_train', 'r_train', 'ratio_hold', 'r_hold']].median().round(3).assign(n=W.groupby('band').size()).to_string())
f = lambda v: f'{np.nanmean(np.abs(v - 1) <= 0.25):.2f}'
print(f'\nheld-out year, annual model/tower within 25 %: all {f(D.ratio_hold.values)} ({D.ratio_hold.notna().sum()} towers), WET {f(W.ratio_hold.values)}; median ratio {D.ratio_hold.median():.2f}, median r {D.r_hold.median():.2f}')
