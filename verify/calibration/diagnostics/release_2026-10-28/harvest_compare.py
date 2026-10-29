"""Crop harvest of two protocol runs, 2001-2010 means on the cropland cells: harvest, grass (lower-canopy) NPP, lower-canopy
cover and leaf area, GPP, and the shrub NPP removed from the cropland share; by region and by cropland-share bin.
    python -I harvest_compare.py label=run_hindcast_out [label=...]"""
import sys, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/shared/nas2/Hangkai/ibis-port/verify/calibration'); from landmask import true_landfrac
runs = dict(a.split('=', 1) for a in sys.argv[1:])
LAT = np.arange(89.75, -90, -0.5); LON = np.arange(-179.75, 180, 0.5); LO, LA = np.meshgrid(LON, LAT)
area = (6371.0e3 ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * true_landfrac()
crop = np.clip(np.nan_to_num(np.load('/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz')['cropfrac']), 0, 1)
VARS = ('ayharvest', 'aynppl', 'aynppu', 'aygpptot', 'aynpptot', 'fl', 'lail', 'fu', 'aycropshrub', 'ayclearing')
def load(d):
    acc = {}; n = 0
    for y in range(2001, 2011):
        try: ds = xr.open_dataset(f'{d}/ibis_yearly_{y}.nc')
        except FileNotFoundError: continue
        for v in VARS:
            if v in ds: acc[v] = acc.get(v, 0) + np.nan_to_num(np.asarray(ds[v].values, float))
        n += 1
    return {v: a / n for v, a in acc.items()}, n
D = {k: load(d) for k, d in runs.items()}
for k, (F, n) in D.items(): print(f'{k}: {n} years, vars {sorted(F)}')
REG = {'N America': (LO < -30) & (LA > 12), 'S America': (LO < -30) & (LA <= 12), 'Europe': (LO >= -30) & (LO < 60) & (LA > 36), 'Africa': (LO >= -30) & (LO < 60) & (LA <= 36),
       'S + SE Asia': (LO >= 60) & (LA <= 30) & (LA > -10), 'E Asia + Siberia': (LO >= 60) & (LA > 30), 'Australia': (LO > 110) & (LA < -10)}
Pg = lambda x, m: float(np.nansum(np.where(m, x, 0) * area)) * 1e-12
def G(F, v): return F[v] if v in F else np.full((360, 720), np.nan)
def row(name, m):
    out = f'{name:18s} crop area {Pg(crop, m) * 1e12 / 1e12:6.2f} Mkm2 |'
    for k, (F, n) in D.items():
        h = Pg(F['ayharvest'], m); ca = float(np.nansum(np.where(m, crop, 0) * area))
        out += f' {k}: harvest {h:5.2f} PgC = {h * 1e12 / max(ca, 1) * 1e3:5.0f} gC/m2crop; nppl {Pg(G(F, "aynppl"), m) :5.2f} gpp {Pg(F["aygpptot"], m):6.1f} fl {np.nansum(np.where(m, F["fl"] * area, 0)) / np.nansum(np.where(m, area, 0)):.2f} lail {np.nansum(np.where(m, F["lail"] * area, 0)) / np.nansum(np.where(m, area, 0)):.2f}'
        if 'aycropshrub' in F: out += f' cropshrub {Pg(F["aycropshrub"], m):4.2f}'
        out += ' |'
    print(out)
land = area > 0; cm = land & (crop > 0.1)
print('\n--- totals: all land | cropland cells (share > 0.1) | by region (cropland cells)')
row('all land', land); row('cropland cells', cm)
for r, m in REG.items(): row(r, cm & m)
print('\n--- by cropland-share bin (all land): harvest and grass NPP per unit cropland area, lower cover and LAI')
for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
    m = land & (crop >= lo) & (crop < hi); row(f'share {lo:.1f}-{hi:.1f}', m)
print('\n--- harvest / grass NPP on cropland cells (the share of lower-canopy NPP that leaves as harvest):')
for k, (F, n) in D.items(): print(f'  {k}: {Pg(F["ayharvest"], cm) / max(Pg(G(F, "aynppl"), cm), 1e-9):.3f}')
