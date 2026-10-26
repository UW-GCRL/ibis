"""Tower metrics of several shared-type-set scores against the own fits (CALIBRATION.md 57).
Sets: deck = t31_rescore (3.1 set, GRID soil pools and litterfall: no --start-state); deck_pools = the same set on the
tower pools (t31_pools); A = params_31f_cs2 per tower on the tower pools; B = params_31g5 per tower on the tower pools.
Own fit = t31_s2 (GPP, ET) and t31_nee (NEE; Reco = GPP + NEE of that stage), as in the meeting deck.
    python compare_sets.py <out csv prefix>
Metrics over the tower window, months with a tower value: amount = model sum / tower sum, timing r = monthly correlation,
NEE bias = mean(model - tower) x 12 (gC m-2 yr-1, + = too little uptake)."""
import sys, os, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
V = '/shared/nas2/Hangkai/ibis/verify_outputs/site_fit'; C = '/shared/nas2/Hangkai/ibis/inputs/sites/towers31/compact'
SETS = {'deck': 't31_rescore', 'deck_pools': 't31_rescore_pools', 'A': 't31_score_31f_cs2', 'B': 't31_score_31g5'}
out = sys.argv[1]
for a in sys.argv[2:]:                      # extra sets as name=dir, e.g. A2=t31_score_31j B2=t31_score_31k
    k, v = a.split('='); SETS[k] = v
si = pd.read_csv(f'{C}/FLUXNET_siteinfo.csv', encoding='utf-8-sig').set_index('SITE_ID')


def tower(s):
    f = glob.glob(f'{C}/FLX_{s}_FLUXNET2015_FULLSET_MM_*_31.csv')
    if not f: return None
    d = pd.read_csv(f[0]); t = pd.to_datetime(d['TIMESTAMP'].astype(str), format='%Y%m'); days = t.dt.days_in_month.values
    o = pd.DataFrame({'y': t.dt.year.values, 'm': t.dt.month.values})
    for c, k in (('GPP_NT_VUT_REF', 'gpp'), ('NEE_VUT_REF', 'nee'), ('RECO_NT_VUT_REF', 'reco')):
        o[k] = d[c].where(d[c] > -9000).values * days if c in d else np.nan
    o['et'] = d['LE_F_MDS'].where(d['LE_F_MDS'] > -9000).values * 86400 / 2.45e6 * days if 'LE_F_MDS' in d else np.nan
    return o.set_index(['y', 'm'])


def load(f):
    z = np.load(f); n = z['nee']
    n = n.sum(2)[..., 0] if n.ndim == 4 else (n[..., 0] if n.ndim == 3 else n)
    g = z['gpp'][..., 0]
    return dict(years=list(z['years']), gpp=g, et=z['et'][..., 0], nee=n, reco=g + n)


def met(m, o, k):
    ok = np.isfinite(m) & np.isfinite(o)
    if ok.sum() < 12: return {}
    m, o = m[ok], o[ok]
    if k == 'nee': return {f'{k}_bias': float((m - o).mean() * 12), f'{k}_r': float(np.corrcoef(m, o)[0, 1])}
    return {f'{k}_ratio': float(m.sum() / o.sum()) if abs(o.sum()) > 1e-6 else np.nan, f'{k}_r': float(np.corrcoef(m, o)[0, 1])}


rows = []
for s in si.index:
    if si.loc[s, 'IGBP'] in ('WET', 'SNO', 'CRO'): continue
    fo, fn = f'{V}/t31_s2/{s}/best_series.npz', f'{V}/t31_nee/{s}/best_series.npz'
    if not (os.path.exists(fo) and os.path.exists(fn)): continue
    tw = tower(s)
    if tw is None: continue
    o2, on = load(fo), load(fn); yrs = o2['years']
    obs = {k: np.array([[tw[k].get((y, mm), np.nan) for mm in range(1, 13)] for y in yrs]) for k in ('gpp', 'nee', 'et', 'reco')}
    src = {'own': dict(gpp=o2['gpp'], et=o2['et'], nee=on['nee'], reco=on['gpp'] + on['nee'])}
    for tag, d in SETS.items():
        f = f'{V}/{d}/{s}/best_series.npz'
        if os.path.exists(f) and os.path.exists(f'{V}/{d}/{s}/run.log') and 'baseline loss' in open(f'{V}/{d}/{s}/run.log').read():
            src[tag] = load(f)
    r = dict(site=s, igbp=si.loc[s, 'IGBP'], koppen=str(si.loc[s, 'KOPPEN'])[:1], types=si.loc[s, 'TYPE'],
             nee_obs=float(np.nanmean(obs['nee']) * 12), gpp_obs=float(np.nanmean(obs['gpp']) * 12))
    for tag, d in src.items():
        for k in ('gpp', 'et', 'reco', 'nee'):
            for q, v in met(d[k].ravel(), obs[k].ravel(), k).items(): r[f'{tag}_{q}'] = v
        r[f'{tag}_nee_annual'] = float(np.mean(d['nee'][np.isfinite(obs['nee'])]) * 12) if np.isfinite(obs['nee']).any() else np.nan
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f'{out}_towers.csv', index=False)
tags = ['own'] + list(SETS)
cols = ['gpp_ratio', 'gpp_r', 'et_ratio', 'reco_ratio', 'nee_bias', 'nee_r', 'nee_annual']


def summ(g):
    d = {'n': len(g)}
    for t in tags:
        for c in cols:
            k = f'{t}_{c}'
            if k in g: d[k] = g[k].median()
        if f'{t}_gpp_ratio' in g: d[f'{t}_n'] = int(g[f'{t}_gpp_ratio'].notna().sum())
    return pd.Series(d)


allm = summ(df); by = df.groupby('igbp').apply(summ)
allm.to_frame('all').T.to_csv(f'{out}_all.csv', index=False); by.to_csv(f'{out}_by_class.csv')
pd.set_option('display.width', 250)
print(f'towers {len(df)}; tower NEE median {df.nee_obs.median():.0f} gC m-2 yr-1')
tab = pd.DataFrame({t: [allm.get(f'{t}_{c}', np.nan) for c in cols] + [allm.get(f'{t}_n', np.nan)] for t in tags}, index=cols + ['n'])
print(tab.round(2).to_string())
