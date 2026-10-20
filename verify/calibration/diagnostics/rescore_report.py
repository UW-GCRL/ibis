"""The IBIS 3.1 per-type set at the campaign towers (CALIBRATION.md 50): the re-score baselines against each tower's own fit.

    IBIS_FLUX_DIR=<towers31>/compact python verify/calibration/diagnostics/rescore_report.py <rescore root> <own-fit roots...> --out <csv>

Per tower: the loss of the 3.1 set on the tower's own objective (rescore best.json) against the loss of the tower's own
fit (the first own-fit root holding its best.json), and the annual GPP ratio model / tower over the scored years for
both (best_series.npz against the tower's monthly GPP through site_fit.tower_obs). Medians by type (every type of the
tower's TYPE column) and by IGBP class.
"""
import sys, os, json, glob, types, numpy as np, pandas as pd
sys.path[:0] = ['python-port', 'verify/calibration']
import site_fit as SF
a = sys.argv[1:]; out = a[a.index('--out') + 1]; roots = [r for r in a if not r.startswith('--') and r != out]
rescore, own_roots = roots[0], roots[1:]
si = pd.read_csv(f"{os.environ['IBIS_FLUX_DIR']}/FLUXNET_siteinfo.csv", encoding='utf-8-sig').set_index('SITE_ID')


def gpp_ratio(d):
    z = np.load(f'{d}/best_series.npz'); yrs = [int(y) for y in z['years']]; s = str(z['sites'][0])
    SF.configure_years(types.SimpleNamespace(years=f'{yrs[0]}-{yrs[-1]}', hold_last=1))
    o = SF.tower_obs(s)['gpp']; g = z['gpp'][:, :, 0]
    ok = np.isfinite(o); ok[0] = False                                    # the first year is warm-up
    return float(g[ok].sum() / o[ok].sum()) if ok.sum() >= 12 and o[ok].sum() > 50 else np.nan


rows = []
for d in sorted(glob.glob(f'{rescore}/*/')):
    s = os.path.basename(d.rstrip('/')); b = f'{d}/best.json'
    if not os.path.exists(b) or s not in si.index: continue
    own = next((f'{r}/{s}' for r in own_roots if os.path.exists(f'{r}/{s}/best_series.npz')), None)
    rows.append(dict(site=s, igbp=si.loc[s, 'IGBP'], types=str(si.loc[s, 'TYPE']), loss_31=json.load(open(b))['loss'],
                     loss_own=json.load(open(f'{own}/best.json'))['loss'] if own else np.nan,
                     gpp_ratio_31=gpp_ratio(d), gpp_ratio_own=gpp_ratio(own) if own else np.nan))
t = pd.DataFrame(rows); t['loss_31_over_own'] = t.loss_31 / t.loss_own; t.to_csv(out, index=False)
q = lambda x: pd.Series(dict(n=len(x), loss_31=x.loss_31.median(), loss_own=x.loss_own.median(), loss_ratio=x.loss_31_over_own.median(),
                             gpp_31=x.gpp_ratio_31.median(), gpp_own=x.gpp_ratio_own.median(), within25_31=(abs(np.log(x.gpp_ratio_31)) < np.log(1.25)).mean()))
print(f'{len(t)} towers re-scored under the 3.1 set (medians; gpp = annual model / tower over the scored years; within25 = share within 0.8-1.25)\n')
print('all:'); print(q(t).round(2).to_string())
print('\nby IGBP:'); print(t.groupby('igbp').apply(q).round(2).to_string())
e = t.assign(type=t.types.str.split()).explode('type')
print('\nby type:'); print(e.groupby('type').apply(q).round(2).sort_values('n', ascending=False).to_string())
