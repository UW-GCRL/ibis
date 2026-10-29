"""Soil organic carbon of the model (totcsoi, 2010 state) against HWSD2 (soita_hwsd2_organic.nc: orgcpct x bulkdens x
thickness, 0-1 m and 0-2 m), by latitude band, Köppen group and continent; also the model's turnover (soil C / Rh).
usage: python -I soil_ref_bands.py label=run_dir [label=run_dir ...]   (run_dir holds restart/ibis_restart_2010.npz, out/ibis_yearly_2010.nc)"""
import sys, re, ast, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'; Z = np.load(f'{S}/meeting/glob_refs.npz')['zone']
runs = dict(a.split('=', 1) for a in sys.argv[1:])
h = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/soil/soita_hwsd2_organic.nc')
thick = np.array([0.10, 0.15, 0.25, 0.50, 1.0, 2.0])   # m per layer
oc = np.asarray(h['orgcpct'].values, float) / 100.0; bd = np.asarray(h['bulkdens'].values, float) * 1000.0   # kg m-3
soc_layer = np.nan_to_num(oc * bd) * thick[:, None, None]                                                     # kgC m-2 per layer
soc1 = soc_layer[:4].sum(0); soc2 = soc_layer[:5].sum(0)
first = next(iter(runs.values()))
def load(run):
    p = f'{run}/restart/ibis_restart_2010.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p)
    lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
    area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj])
    d = xr.open_dataset(f'{run}/out/ibis_yearly_2010.nc')
    rh = np.asarray(d['ayco2mic'].values, float)[gi, gj] if 'ayco2mic' in d else np.full_like(area, np.nan)
    return dict(lat=lat, lon=lon, gi=gi, gj=gj, area=area, soil=np.asarray(st.carbon.totcsoi, float), rh=rh, lit=np.asarray(st.carbon.totlit, float) if hasattr(st.carbon, 'totlit') else None)
D = {k: load(r) for k, r in runs.items()}; g = D[next(iter(D))]; lat, lon, gi, gj, area = g['lat'], g['lon'], g['gi'], g['gj'], g['area']
zone = Z[gi, gj]; ref1 = soc1[gi, gj]; ref2 = soc2[gi, gj]; okr = np.isfinite(ref1) & (ref1 > 0)
zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
G = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'), 'temperate': zn('C_oceanic', 'D_temperate'),
     'boreal': zn('D_subarctic', 'D_extreme'), 'tundra + cold dry': zn('E_tundra', 'B_cold'), '90S-30N': lat < 30, '30-60N': (lat >= 30) & (lat < 60), '60-90N': lat >= 60, 'GLOBAL': np.ones_like(zone, bool)}
Pg = lambda x, m: float(np.nansum(x[m] * area[m])) / 1e12
print('soil C 2010, PgC: HWSD2 0-1 m | 0-2 m | per run (x HWSD 0-1 m) | model soil C / Rh (yr) | HWSD cells only; runs: ' + ', '.join(f'{k} = {r}' for k, r in runs.items()))
print(f"{'chunk':22s} {'HWSD1m':>7s} {'HWSD2m':>7s} | " + ' '.join(f'{k:>14s}' for k in D) + ' | ' + ' '.join(f'{k + " tau":>9s}' for k in D))
for name, m in G.items():
    mm = m & okr
    print(f"{name:22s} {Pg(ref1, mm):7.0f} {Pg(ref2, mm):7.0f} | " + ' '.join(f'{Pg(D[k]["soil"], mm):6.0f} ({Pg(D[k]["soil"], mm) / max(Pg(ref1, mm), 1e-9):4.2f})' for k in D)
          + ' | ' + ' '.join(f'{Pg(D[k]["soil"], mm) / max(Pg(D[k]["rh"], mm), 1e-9):9.1f}' for k in D))
print(f'\nHWSD coverage: {100 * area[okr].sum() / area.sum():.1f} % of land area')
