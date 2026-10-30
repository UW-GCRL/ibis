"""Soil carbon through the 200-year spin-up of a protocol run, by zone group and band: the trajectory at chosen years,
the drift in the last decade of the 1x tail (constant cycled climate, so drift = disequilibrium), and an equilibrium
estimate from the 1x tail by regressing the yearly increment on the stock (dC/dt = (Ceq - C) / tau).
    python -I spinup_soil_traj.py <run>   (verify_outputs/final_<run>/spinup/out/ibis_yearly_YYYY.nc)"""
import sys, re, ast, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; V = '/shared/nas2/Hangkai/ibis/verify_outputs'; M = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work/meeting'
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
run = sys.argv[1]; ref = np.load(f'{M}/glob_refs.npz'); Z = ref['zone']; lf = ref['lf']; lat = ref['lat']
area = 6371e3 ** 2 * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((360, 720)) * np.nan_to_num(lf)
zn = lambda *nm: np.isin(Z, [ZONES.index(n) for n in nm])
G = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'), 'temperate': zn('C_oceanic', 'D_temperate'),
     'boreal': zn('D_subarctic', 'D_extreme'), 'tundra + cold dry': zn('E_tundra', 'B_cold'), '30-60N': (lat[:, None] >= 30) & (lat[:, None] < 60) & np.ones((360, 720), bool), '60-90N': (lat[:, None] >= 60) & np.ones((360, 720), bool), 'GLOBAL': np.ones((360, 720), bool)}
years = list(range(1701, 1901)); C = {g: [] for g in G}; L = {g: [] for g in G}
for y in years:
    d = xr.open_dataset(f'{V}/final_{run}/spinup/out/ibis_yearly_{y}.nc'); c = np.nan_to_num(np.asarray(d['totcsoi'].values, float)) * area; l = np.nan_to_num(np.asarray(d['totlit'].values, float)) * area if 'totlit' in d else np.zeros_like(c)
    for g, m in G.items(): C[g].append(float(c[m].sum()) / 1e12); L[g].append(float(l[m].sum()) / 1e12)
C = {g: np.array(v) for g, v in C.items()}; yrs = np.array(years)
pick = [1701, 1720, 1750, 1800, 1840, 1850, 1851, 1860, 1875, 1890, 1900]
print(f'{run}: soil C (totcsoi, PgC) through the spin-up; acceleration x10 to 1850, 1x tail 1851-1900 (cycled 1901-1930 climate, 296 ppm)')
print(f'{"chunk":22s} ' + ' '.join(f'{y:7d}' for y in pick) + ' | drift 1891-1900 PgC/yr | tail fit (1851-1900): Ceq, tau yr, filled | jump 1850->1851')
for g in G:
    c = C[g]; i = {y: list(yrs).index(y) for y in pick}
    drift = (c[-1] - c[-10]) / 9.0
    t = c[yrs >= 1851]; dc = np.diff(t); x = t[:-1]; A = np.vstack([x, np.ones_like(x)]).T; (a, b), *_ = np.linalg.lstsq(A, dc, rcond=None)   # dc = a*C + b -> tau = -1/a, Ceq = -b/a
    tau = -1.0 / a if a < 0 else float('inf'); ceq = -b / a if a < 0 else float('nan'); filled = c[-1] / ceq if np.isfinite(ceq) and ceq > 0 else float('nan')
    print(f'{g:22s} ' + ' '.join(f'{c[i[y]]:7.1f}' for y in pick) + f' | {drift:+6.2f} | {ceq:7.0f} {tau:6.0f} {filled:5.2f} | {c[list(yrs).index(1851)] - c[list(yrs).index(1850)]:+6.1f}')
