"""Water balance of a run's yearly output: P - ET - runoff - drainage by Köppen zone and globally (area-weighted on true land),
the cells creating water (residual < -50 mm/yr), and the share of land with a layer at exactly 0 over a wet layer in the last
restart (the soilh2o pump fingerprint, CALIBRATION.md 62). Read-only.
    python water_balance.py <run dir with out/ibis_yearly_*.nc and restart/> [y0 y1]"""
import sys, glob, os, re, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
from landmask import true_landfrac
from ibis.io.pft_types import zones_from_classes, ZONES
run = sys.argv[1]
yrs = sorted(int(re.search(r'(\d{4})', os.path.basename(f)).group(1)) for f in glob.glob(f'{run}/out/ibis_yearly_*.nc'))
if len(sys.argv) > 3: yrs = [y for y in yrs if int(sys.argv[2]) <= y <= int(sys.argv[3])]
lat = np.arange(89.75, -90, -0.5); LA = np.repeat(lat[:, None], 720, 1); lf = np.nan_to_num(true_landfrac())
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(LA)) * lf
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
if np.asarray(k['lat'])[0] < np.asarray(k['lat'])[-1]: c = c[::-1]
zone = zones_from_classes(c)
acc = {}
for y in yrs:
    d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc')
    for kk, n in (('P', 'ayprcp'), ('E', 'ayaet'), ('R', 'aytrunof'), ('D', 'aytdrain')): acc[kk] = acc.get(kk, 0) + np.nan_to_num(np.asarray(d[n].values, float)) / len(yrs)
m = np.isfinite(np.asarray(xr.open_dataset(f'{run}/out/ibis_yearly_{yrs[-1]}.nc')['ayprcp'].values))
res = acc['P'] - acc['E'] - acc['R'] - acc['D']
K = lambda x, mm: float((x * area * mm).sum() / 1e12)
print(f'{run}: {yrs[0]}-{yrs[-1]} means; 10^3 km3/yr: P {K(acc["P"], m):.1f} ET {K(acc["E"], m):.1f} R {K(acc["R"], m):.1f} D {K(acc["D"], m):.1f}  P-ET-R-D {K(res, m):+.2f} ({100 * K(res, m) / K(acc["P"], m):+.1f} % of P)')
print(f'  cells with residual < -50 mm/yr: {int((m & (res < -50)).sum())}; < -500: {int((m & (res < -500)).sum())}; ayaet < 0: {int((m & (acc["E"] < 0)).sum())}')
print(f"  {'zone':12s} {'P':>7s} {'ET':>7s} {'R+D':>7s} {'resid':>7s}   mm/yr")
for i, z in enumerate(ZONES):
    mm = m & (zone == i)
    if mm.sum() < 50: continue
    w = (area * mm).sum(); f = lambda x: float((x * area * mm).sum() / w)
    print(f"  {z:12s} {f(acc['P']):7.0f} {f(acc['E']):7.0f} {f(acc['R'] + acc['D']):7.0f} {f(res):+7.0f}")
rs = sorted(glob.glob(f'{run}/restart/ibis_restart_*.npz'))
if rs:
    import evaluate_gfed5 as eg
    p = rs[-1]; la, lo, *_ = eg._grid(p); st = eg._load(p); w = np.asarray(st.soil.wsoi)
    pump = (w[:, 3] == 0.0) & (w[:, 4] > 0.5); sat = w[:, 5] >= 0.999
    print(f'  last restart {os.path.basename(p)}: layer-4 exactly 0 over wet layer 5: {int(pump.sum())} cells; bottom layer saturated: {int(sat.sum())} cells')
