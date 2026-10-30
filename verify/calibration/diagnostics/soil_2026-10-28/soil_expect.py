"""Expected equilibrium soil carbon under the SOM temperature sensitivity (som_lt_e0, anchored at 25 C; CALIBRATION.md
72) from a run's 2010 state: per cell the SOM pools (slow + passive) scale by 1 / <ratio>, the ratio averaged over the
twelve monthly soil temperatures (layers 1-4, thickness-weighted) with the cell's LAI-weighted litter E0 from the
typed parameters; litter pools unchanged. By zone group and band against HWSD2 0-1 m. Read-only.
usage: python -I soil_expect.py <run dir> <params.yaml> [E0 values...]"""
import sys, re, ast, glob, numpy as np, xarray as xr, yaml, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import jax.numpy as jnp
import evaluate_gfed5 as eg; from landmask import true_landfrac
from ibis.io.pft_types import zones_from_classes, ZONES, apply_types, parse_type_overrides
from ibis.io.physics_params import build_all_params
from ibis.physics.biogeochem import som_climate_ratio
run, pfile = sys.argv[1], sys.argv[2]; E0S = [float(x) for x in sys.argv[3:]] or [400.0, 450.0, 500.0, 550.0, 600.0]
S = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work'; Z = np.load(f'{S}/meeting/glob_refs.npz')['zone']
p = f'{run}/restart/ibis_restart_2010.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p); cb = st.carbon; v = st.vegetation
lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj]); npoi = len(lat)
# the cell's litter E0 (LAI-weighted over the typed per-position lt_e0)
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
klat = np.asarray(k['lat'], float); kli = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int).clip(0, 359)
klon = np.asarray(k['lon'], float); klo = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int).clip(0, 719)
zone_idx = zones_from_classes(c[kli, klo])
y = yaml.safe_load(open(pfile)); params, pp, _ = build_all_params(npoi, np.zeros(npoi), y) if 'y' in build_all_params.__code__.co_varnames else build_all_params(npoi, np.zeros(npoi))
try:
    params, pp = apply_types(params, pp, zone_idx, parse_type_overrides(y.get('types')))
except Exception as e:
    print('apply_types failed, using base lt_e0:', e)
e0 = np.asarray(params.bgc_params.lt_e0, float); plai = np.maximum(np.asarray(v.plai, float), 0.0)
if e0.ndim == 2: tconst = np.where(plai.sum(1) > 0, (plai * e0).sum(1) / np.maximum(plai.sum(1), 1e-6), e0.mean(1))
elif e0.ndim == 1: tconst = np.where(plai.sum(1) > 0, (plai * e0[None, :]).sum(1) / np.maximum(plai.sum(1), 1e-6), e0.mean())
else: tconst = np.full(npoi, float(e0))
print(f'litter E0 per cell: mean {tconst.mean():.0f}, range {tconst.min():.0f}-{tconst.max():.0f}')
# monthly soil temperature of layers 1-4
hs = np.array([0.10, 0.15, 0.25, 0.50]); ts = []
for f in sorted(glob.glob(f'{run}/out/ibis_monthly_2010*.nc')):
    d = xr.open_dataset(f); t = np.asarray(d['tsoi'].values, float)[gi, gj, :4]; ts.append((t * hs).sum(1) / hs.sum())
ts = np.array(ts); print(f'{len(ts)} monthly soil temperatures; mean {np.nanmean(ts):.1f} K')
zone = Z[gi, gj]
h = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/soil/soita_hwsd2_organic.nc'); thick = np.array([0.10, 0.15, 0.25, 0.50, 1.0, 2.0])
soc1 = (np.nan_to_num(np.asarray(h['orgcpct'].values, float) / 100.0 * np.asarray(h['bulkdens'].values, float) * 1000.0) * thick[:, None, None])[:4].sum(0)[gi, gj]; okr = np.isfinite(soc1) & (soc1 > 0)
lit = sum(np.asarray(getattr(cb, k_), float) for k_ in ('clitlm', 'clitls', 'clitll', 'clitrm', 'clitrs', 'clitrl', 'clitwm', 'clitws', 'clitwl'))
som = np.asarray(cb.csoislon, float) + np.asarray(cb.csoislop, float) + np.asarray(cb.csoipas, float); tot = np.asarray(cb.totcsoi, float)
zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
G = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'), 'temperate': zn('C_oceanic', 'D_temperate'),
     'boreal': zn('D_subarctic', 'D_extreme'), 'tundra + cold dry': zn('E_tundra', 'B_cold'), '90S-30N': lat < 30, '30-60N': (lat >= 30) & (lat < 60), '60-90N': lat >= 60, 'GLOBAL': np.ones_like(zone, bool)}
Pg = lambda x, m: float(np.nansum(x[m] * area[m])) / 1e12
exp_ = {}
for E in E0S:
    r = np.mean([np.asarray(som_climate_ratio(jnp.asarray(t), jnp.asarray(tconst), E, 298.16)) for t in ts], 0)
    exp_[E] = lit + som / np.maximum(r, 1e-3)
print(f'\nexpected equilibrium soil C (PgC) by group: now | HWSD2 0-1 m | under som_lt_e0 = ' + ', '.join(f'{E:.0f}' for E in E0S) + '  (x HWSD2 in brackets)')
print(f"{'chunk':22s} {'now':>6s} {'HWSD':>6s} | " + ' '.join(f'{E:>13.0f}' for E in E0S))
for name, m in G.items():
    mm = m & okr
    print(f"{name:22s} {Pg(tot, mm):6.0f} {Pg(soc1, mm):6.0f} | " + ' '.join(f'{Pg(exp_[E], mm):6.0f} ({Pg(exp_[E], mm) / max(Pg(soc1, mm), 1e-9):4.2f})' for E in E0S))
print(f"\nliterature total to 1 m: 1500-2400 PgC; the model's own (all cells): now {Pg(tot, np.ones_like(zone, bool)):.0f}, " + ', '.join(f'E0 {E:.0f}: {Pg(exp_[E], np.ones_like(zone, bool)):.0f}' for E in E0S))
