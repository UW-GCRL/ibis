"""Supplementary 2 report: the per-tower fits against the S1 set and the tower (CALIBRATION.md 35).

    python verify/calibration/diagnostics/s2_report.py <s2 root> <s1 fit dir> <out.csv>

For each tower folder under <s2 root> with a best.json: model / tower ratios (summer months, annual GPP, ET where the
tower has energy-balance-corrected LE), r and NSE of monthly GPP, on the fit years and on the held-out year, for the
S1 parameters (<s1 fit dir>/best_series.npz, the 62-tower series) and the per-tower fit (best_series.npz); the fitted
parameters against their S1 values. Writes the table and prints a verdict by class.
"""
import sys, os, json, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = ['python-port', 'verify/calibration']
import site_fit
root, s1dir, out = sys.argv[1:4]
S1 = np.load(f'{s1dir}/best_series.npz', allow_pickle=True); s1_sites = list(S1['sites']); s1p = json.load(open(f'{s1dir}/best.json'))['params']
import types; Y = [int(y) for y in S1['years']]; site_fit.configure_years(types.SimpleNamespace(years=f'{Y[0]}-{Y[-1]}', hold_last=1))
def ratios(g, e, o, ky):
    """model / tower over the year indices ky: summer, annual, ET; r, NSE of monthly GPP."""
    tg, te = o['gpp'][ky], o['et'][ky]; g, e = g[ky], e[ky]; ok = np.isfinite(tg); oke = np.isfinite(te)
    if ok.sum() < 6: return dict(season=np.nan, annual=np.nan, et=np.nan, r=np.nan, nse=np.nan)
    ss = np.zeros_like(ok); ss[:, o['season']] = True; ss &= ok
    r = np.corrcoef(g[ok], tg[ok])[0, 1]; nse = 1 - ((g[ok] - tg[ok]) ** 2).sum() / ((tg[ok] - tg[ok].mean()) ** 2).sum()
    return dict(season=g[ss].sum() / tg[ss].sum(), annual=g[ok].sum() / tg[ok].sum(), et=(e[oke].sum() / te[oke].sum()) if oke.sum() >= 6 and o['et_src'] == 'LE_CORR' else np.nan, r=r, nse=nse)
rows = []
for d in sorted(glob.glob(f'{root}/*/best.json')):
    s = d.split('/')[-2]; b = json.load(open(d)); R = np.load(f'{os.path.dirname(d)}/best_series.npz', allow_pickle=True)
    o = site_fit.tower_obs(s); k1 = s1_sites.index(s); n = list(R['sites']).index(s)
    TR, HO = site_fit.TRAIN_Y, site_fit.HOLD_Y
    row = dict(site=s, igbp=site_fit.META.loc[s, 'IGBP'], loss_s1=np.nan, loss_fit=b['loss'], eval=b['eval'], n_par=len(b['params']))
    for lab, ky in (('train', TR), ('hold', HO)):
        for tag, g, e in (('s1', S1['gpp'][:, :, k1], S1['et'][:, :, k1]), ('fit', R['gpp'][:, :, n], R['et'][:, :, n])):
            for k, v in ratios(g, e, o, ky).items(): row[f'{k}_{tag}_{lab}'] = v
    for k, v in b['params'].items(): row[f'p:{k}'] = v; row[f's1:{k}'] = s1p.get(k, np.nan)
    rows.append(row)
T = pd.DataFrame(rows); T.to_csv(out, index=False); pd.set_option('display.width', 250)
print(f'{len(T)} towers; per-tower fit vs S1 on the HELD-OUT year (model / tower; within 25 % of annual GPP):')
for lab in ('train', 'hold'):
    w1 = (np.abs(np.log(T[f'annual_s1_{lab}'])) < np.log(1.25)).mean(); w2 = (np.abs(np.log(T[f'annual_fit_{lab}'])) < np.log(1.25)).mean()
    print(f'  {lab}: within 25 %  S1 {w1:.0%} -> fit {w2:.0%};  median r {T[f"r_s1_{lab}"].median():.2f} -> {T[f"r_fit_{lab}"].median():.2f};  median NSE {T[f"nse_s1_{lab}"].median():.2f} -> {T[f"nse_fit_{lab}"].median():.2f}')
cols = ['season_s1_hold', 'season_fit_hold', 'annual_s1_hold', 'annual_fit_hold', 'et_s1_hold', 'et_fit_hold', 'nse_s1_hold', 'nse_fit_hold']
G = T.groupby('igbp')[cols].median(); G['n'] = T.groupby('igbp').size(); print('\nby class, median over towers, held-out year:'); print(G.round(2).to_string())
print('\nper tower (held-out year):'); print(T[['site', 'igbp', 'loss_fit'] + cols].round(2).to_string(index=False))
