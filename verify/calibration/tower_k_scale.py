"""Per-tower decomposition-rate multiplier from the equilibrium soil carbon and the HWSD2 site carbon (NEE stage).

    python verify/calibration/tower_k_scale.py <spin-up dir> <hwsd profile csv> <out csv>

At the cyclic steady state every SOM pool is proportional to 1 / k_scale with all fluxes unchanged, so the multiplier
that puts the equilibrium carbon at the site's is k_scale = SOC_eq / SOC_site, clipped to [0.25, 4]. SOC_site is the
HWSD2 0-1 m stock of the share-weighted profile, or of the dominant component where the mapping unit carries a peat
component that is not the tower's. Columns: site, soc_eq, soc_site, k_scale.
"""
import sys, numpy as np, pandas as pd
spin, prof, out = sys.argv[1:4]; P = np.load(f'{spin}/pools.npz', allow_pickle=True); sites = [str(s) for s in P['sites']]
H = pd.read_csv(prof).set_index('site'); SOM = ('csoislon', 'csoislop', 'csoipas', 'totcmic'); LIT = ('clitlm', 'clitls', 'clitll', 'clitrm', 'clitrs', 'clitrl', 'clitwm', 'clitws', 'clitwl')
rows = []
for n, s in enumerate(sites):
    som = sum(float(P[f'{p}_after'][n]) for p in SOM); lit = sum(float(P[f'{p}_after'][n]) for p in LIT)
    site_soc = float(H.loc[s, 'dom_soc_1m_kgC'] if H.loc[s, 'peat_share_pct'] > 0 else H.loc[s, 'soc_1m_kgC'])
    k = float(np.clip(som / site_soc, 0.25, 4.0)) if site_soc > 0 else 1.0
    rows.append(dict(site=s, igbp=H.loc[s, 'igbp'], soc_eq=som, litter_eq=lit, soc_grid=sum(float(P[f'{p}_before'][n]) for p in SOM), soc_site=site_soc, peat=H.loc[s, 'peat_share_pct'], k_scale=k, nee_eq=float(P['nee_gC'][n]), npp_eq=float(P['npp_gC'][n])))
T = pd.DataFrame(rows); T.to_csv(out, index=False); pd.set_option('display.width', 200)
print(T.round(2).to_string(index=False)); print('\nclass medians:'); print(T.groupby('igbp')[['soc_grid', 'soc_eq', 'soc_site', 'k_scale', 'nee_eq', 'npp_eq']].median().round(2).to_string())
print('k_scale at a bound:', T[(T.k_scale <= 0.25) | (T.k_scale >= 4.0)].site.tolist())
