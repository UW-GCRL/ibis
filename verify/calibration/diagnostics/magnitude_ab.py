"""Summer magnitude, two site_run outputs against the towers (CALIBRATION.md 26): peak month, June-August, annual GPP; ET.

    python verify/calibration/diagnostics/magnitude_ab.py <site_runs dir A> <site_runs dir B> [<label A> <label B>]

Both runs must hold the same towers (site_runs.npz: gpp, et monthly, years x 12 x towers). Tower GPP is the mean of the
NT and DT partitions and ET the energy-balance-corrected LE, months with QC >= 0.5 (site_fit.tower_obs); climatology of
2009-2011. Ratios are model / tower: peak month (the model's largest month over the tower's), June-August (December-
February south of the equator), annual over the months the tower has, and annual ET.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
from site_fit import tower_obs, YEARS
from site_run import META

da, db = sys.argv[1], sys.argv[2]; la, lb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ('A', 'B')
A = np.load(os.path.join(da, 'site_runs.npz'), allow_pickle=True); B = np.load(os.path.join(db, 'site_runs.npz'), allow_pickle=True)
sites = [str(s) for s in A['sites']]; assert sites == [str(s) for s in B['sites']]
ya = [int(y) for y in A['years']]; ev = [ya.index(y) for y in (2009, 2010, 2011)]; eo = [YEARS.index(y) for y in (2009, 2010, 2011)]
rows = []
for n, s in enumerate(sites):
    o = tower_obs(s); g = np.nanmean(o['gpp'][eo], 0); e = np.nanmean(o['et'][eo], 0)
    if np.isfinite(g).sum() < 10: continue
    summer = [5, 6, 7] if META.loc[s, 'LOCATION_LAT'] >= 0 else [11, 0, 1]; ok = np.isfinite(g); oke = np.isfinite(e)
    r = dict(site=s, igbp=META.loc[s, 'IGBP'], tower_peak=np.nanmax(g))
    for lab, R in ((la, A), (lb, B)):
        m = R['gpp'][ev, :, n].mean(0) * (1000.0 if R['gpp'].max() < 5 else 1.0); em = R['et'][ev, :, n].mean(0)
        r[f'peak_{lab}'] = m.max() / np.nanmax(g); r[f'jja_{lab}'] = m[summer].sum() / max(np.nansum(g[summer]), 1e-6)
        r[f'annual_{lab}'] = m[ok].sum() / max(g[ok].sum(), 1e-6); r[f'et_{lab}'] = em[oke].sum() / max(e[oke].sum(), 1e-6) if oke.any() else np.nan
    rows.append(r)
T = pd.DataFrame(rows).set_index('site'); pd.set_option('display.width', 250)
print(f"model / tower, 2009-2011 climatology: {la} vs {lb}"); print(T.round(3).to_string())
print('\nclass medians:'); print(T.groupby('igbp')[[c for c in T.columns if c not in ('igbp', 'tower_peak')]].median().round(3).to_string())
