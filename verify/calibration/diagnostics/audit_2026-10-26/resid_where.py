"""Where does the remaining live-pool residual sit (new run, gate fixed)? Per-cell residual = dLive - (NPP - fall - cdisturb),
grouped by survival class, cropland share and zone; plus the per-position pool change in the worst cells. Read-only."""
import sys, numpy as np, xarray as xr, yaml, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg
from landmask import true_landfrac
from ibis.io.pft_types import zones_from_classes, ZONES
run, y = sys.argv[1], int(sys.argv[2])
lat, lon, *_ = eg._grid(f'{run}/restart/ibis_restart_{y}.npz')
lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj]); P = lambda x, m=None: float(np.nansum(np.asarray(x, float)[m] * area[m])) / 1e12 if m is not None else float(np.nansum(np.asarray(x, float) * area)) / 1e12
cfg = yaml.safe_load(open(f'{run}/cfg.yaml')); pp = yaml.safe_load(open(cfg['params_file']))['pft']; smin, ramp = float(pp['tree_stress_min']), float(pp['tree_stress_ramp'])
crop = np.load(cfg['forcing']['cropfrac_file']); ck = [k for k in crop.files if crop[k].ndim == 2][0]; cf = np.nan_to_num(np.asarray(crop[ck], float)); cf = cf[::-1] if 'lat' in crop.files and np.asarray(crop['lat'])[0] < 0 else cf; cfc = cf[gi, gj]
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
klat = np.asarray(k['lat'], float); kli = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int).clip(0, 359); klon = np.asarray(k['lon'], float); klo = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int).clip(0, 719)
zone = zones_from_classes(c[kli, klo])
def rec(yy):
    st = eg._load(f'{run}/restart/ibis_restart_{yy}.npz'); v, cc = st.vegetation, st.carbon
    return dict(l=np.asarray(v.cbiol), w=np.asarray(v.cbiow), r=np.asarray(v.cbior), ex=np.asarray(v.exist), fall=np.asarray(cc.falll) + np.asarray(cc.fallw) + np.asarray(cc.fallr), cd=np.asarray(cc.cdisturb))
a, b = rec(y - 1), rec(y); d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); g = lambda n: np.asarray(d[n].values, float)[gi, gj]
live = lambda r: (r['l'] + r['w'] + r['r']).sum(1)
res = (live(b) - live(a)) - (g('aynpptot') - b['fall'] - b['cd']); ts = np.clip((g('aystressu') - smin) / ramp, 0, 1)
print(f'{run.split("/")[-2]} {y}: total live residual {P(res):+.2f} PgC/yr')
for name, m in (('ts = 0', ts <= 0), ('0 < ts < 1', (ts > 0) & (ts < 0.999)), ('ts = 1', ts >= 0.999), ('cropland > 0.3', cfc > 0.3), ('cropland <= 0.3', cfc <= 0.3)):
    print(f'  {name:16s} cells {int(m.sum()):6d}  residual {P(res, m):+7.2f}')
print('  by zone:', {ZONES[i]: round(P(res, zone == i), 2) for i in range(len(ZONES)) if (zone == i).sum() > 50})
dpos = ((b['l'] + b['w'] + b['r']) - (a['l'] + a['w'] + a['r']))
exlost = (a['ex'] > 0.5) & (b['ex'] < 0.5); exgain = (a['ex'] < 0.5) & (b['ex'] > 0.5)
print(f'  positions losing existence this year: pools in them at y-1 {P((a["l"] + a["w"] + a["r"]) * exlost).sum() if False else P(((a["l"] + a["w"] + a["r"]) * exlost).sum(1)):.2f} PgC; gaining: {int(exgain.sum())} cell-positions')
print('  per-position pool change, PgC:', np.round([P(dpos[:, j]) for j in range(12)], 2))
print('  fall split: leaf %.2f wood %.2f root %.2f' % tuple(P(np.asarray(getattr(eg._load(f"{run}/restart/ibis_restart_{y}.npz").carbon, k))) for k in ('falll', 'fallw', 'fallr')))
