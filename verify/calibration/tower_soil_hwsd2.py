"""Per-tower soil profiles from HWSD2 at 1 km (Min's /shared/nas2/Min/HWSD; NEE stage, Hangkai 2026-09-30: finer soil
data to limit the 0.5-degree scale effect at the towers).

    python verify/calibration/tower_soil_hwsd2.py <out dir> [--mdb-copy /local/HWSD2.mdb]

Steps: (1) sample the soil-mapping-unit raster HWSD2.bil (30 arc-second) at each tower of the 62-tower batch; (2) read
HWSD2_LAYERS from HWSD2.mdb with the pure-Python `access_parser` (copy the 92 MB file to local disk first: the parser
opens it thousands of times); (3) share-weighted, depth-overlap-weighted sand, clay, organic carbon and bulk density on
the six IBIS layers (the rule of Min's hwsd2_to_soita.py), SOC stock to 1 m, root depth; (4) the model's 0.5-degree
texture at the tower cell (soita.nc) for comparison. Writes tower_hwsd2_smu.csv, tower_hwsd2_layers.csv,
tower_hwsd2_profile.csv. Local python3 with pandas, h5py and access_parser (pip --user) is enough; no JAX.
"""
import sys, os, shutil, numpy as np, pandas as pd
out = sys.argv[1]; os.makedirs(out, exist_ok=True); H = '/shared/nas2/Min/HWSD'
mdb = sys.argv[sys.argv.index('--mdb-copy') + 1] if '--mdb-copy' in sys.argv else f'{out}/HWSD2.mdb'
inv = pd.read_csv('verify/calibration/site_data/site_inventory.csv').set_index('site'); sites = open('verify/calibration/site_data/site_batches.txt').readline().split()[1].split(',')
# 1. soil mapping units at the towers
bil = np.memmap(f'{H}/HWSD2.bil', dtype='<u2', mode='r', shape=(21600, 43200)); rows = []
for s in sites:
    la, lo = float(inv.loc[s, 'lat']), float(inv.loc[s, 'lon']); r = int((89.9958333333333 - la) / 0.00833333333333333 + 0.5); c = int((lo + 179.995833333333) / 0.00833333333333333 + 0.5)
    win = bil[max(r - 6, 0):r + 7, max(c - 6, 0):c + 7]; rows.append(dict(site=s, lat=la, lon=lo, smu=int(bil[r, c]), n_units_11km=len(np.unique(win[win != 65535]))))
SMU = pd.DataFrame(rows); SMU.to_csv(f'{out}/tower_hwsd2_smu.csv', index=False); print('nodata towers:', int((SMU.smu == 65535).sum()), '; mixed 11-km windows:', int((SMU.n_units_11km > 1).sum()), 'of', len(SMU))
# 2. layer attributes
if not os.path.exists(mdb): shutil.copy(f'{H}/HWSD2.mdb', mdb)
from access_parser import AccessParser
L = pd.DataFrame(AccessParser(mdb).parse_table('HWSD2_LAYERS')); L = L[L['HWSD2_SMU_ID'].isin(SMU.smu)].copy()
for v in ('SAND', 'SILT', 'CLAY', 'ORG_CARBON', 'BULK', 'COARSE', 'ROOT_DEPTH', 'AWC'): L[v] = L[v].where(L[v] >= 0)          # HWSD2 codes missing layers (shallow soils, rock) as -7
L.to_csv(f'{out}/tower_hwsd2_layers.csv', index=False)
# 3. profiles on the IBIS layers
IB = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 400)]
def profile(g):
    o = {}
    for k, (t, b) in enumerate(IB):
        w = 0.0; acc = {v: 0.0 for v in ('SAND', 'CLAY', 'ORG_CARBON', 'BULK')}
        for _, r in g.iterrows():
            ov = max(0.0, min(b, r.BOTDEP) - max(t, r.TOPDEP))
            if ov <= 0 or not np.isfinite(r.SAND) or not np.isfinite(r.SHARE): continue
            ww = ov * r.SHARE; w += ww
            for v in acc: acc[v] += ww * (r[v] if np.isfinite(r[v]) else 0.0)
        o[k] = {v: acc[v] / w for v in acc} if w > 0 else (o[k - 1] if k > 0 else {v: np.nan for v in acc})
    return o
rows = []
for _, s in SMU.iterrows():
    g = L[L.HWSD2_SMU_ID == s.smu]; p = profile(g); top = p[0]
    soc = sum((p[k]['ORG_CARBON'] / 100) * p[k]['BULK'] * (b - t) * 10 for k, (t, b) in enumerate(IB[:4]))      # kgC m-2, 0-1 m
    top20 = g[g.LAYER == 'D1']; peat = float(top20[top20.ORG_CARBON >= 12].SHARE.sum()) if len(top20) else 0.0          # Histosol-like components (>= 12 % C)
    dom = g[g.SEQUENCE == g.groupby('SEQUENCE').SHARE.first().idxmax()]; pdom = profile(dom)
    rows.append(dict(site=s.site, igbp=inv.loc[s.site, 'igbp'], smu=s.smu, n_comp=g.SEQUENCE.nunique(), dom_share=g.groupby('SEQUENCE').SHARE.first().max(), peat_share_pct=peat,
                     dom_sand_L1=pdom[0]['SAND'], dom_clay_L1=pdom[0]['CLAY'], dom_oc_top_pct=pdom[0]['ORG_CARBON'], dom_soc_1m_kgC=sum((pdom[k]['ORG_CARBON'] / 100) * pdom[k]['BULK'] * (b - t) * 10 for k, (t, b) in enumerate(IB[:4])),
                     **{f'sand_L{k + 1}': p[k]['SAND'] for k in range(6)}, **{f'clay_L{k + 1}': p[k]['CLAY'] for k in range(6)},
                     oc_top_pct=top['ORG_CARBON'], bulk_top=top['BULK'], soc_1m_kgC=soc, root_depth=g.ROOT_DEPTH.dropna().iloc[0] if g.ROOT_DEPTH.notna().any() else np.nan))
T = pd.DataFrame(rows)
# 4. the model's 0.5-degree cell
try:
    import h5py
    d = h5py.File(f'{H}/soita.nc', 'r'); lat = d['lat'][:]; lon = d['lon'][:]; sand = d['sandpct'][:]; clay = d['claypct'][:]
    ij = [(np.abs(lat - inv.loc[s, 'lat']).argmin(), np.abs(lon - inv.loc[s, 'lon']).argmin()) for s in T.site]
    T['sand_L1_05deg'] = [float(sand[0, i, j]) for i, j in ij]; T['clay_L1_05deg'] = [float(clay[0, i, j]) for i, j in ij]
    print('|1-km minus 0.5-degree| top-layer sand > 10 points at', int(((T.sand_L1 - T.sand_L1_05deg).abs() > 10).sum()), 'towers, clay at', int(((T.clay_L1 - T.clay_L1_05deg).abs() > 10).sum()))
except ImportError: print('h5py missing: 0.5-degree comparison skipped')
T.to_csv(f'{out}/tower_hwsd2_profile.csv', index=False); print('wrote', f'{out}/tower_hwsd2_profile.csv')
