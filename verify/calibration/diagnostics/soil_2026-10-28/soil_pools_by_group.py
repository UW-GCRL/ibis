"""Soil carbon pools of a 2010 state by zone group and band: litter (leaf/root/wood x metabolic/structural/lignin),
microbial biomass, slow non-protected, slow protected, passive; with the HWSD2 0-1 m reference and the factor by which
the SOM pools would have to grow to meet it. usage: python -I soil_pools_by_group.py label=run_dir [label=run_dir ...]"""
import sys, re, ast, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'; Z = np.load(f'{S}/meeting/glob_refs.npz')['zone']
runs = dict(a.split('=', 1) for a in sys.argv[1:])
h = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/soil/soita_hwsd2_organic.nc'); thick = np.array([0.10, 0.15, 0.25, 0.50, 1.0, 2.0])
soc1 = (np.nan_to_num(np.asarray(h['orgcpct'].values, float) / 100.0 * np.asarray(h['bulkdens'].values, float) * 1000.0) * thick[:, None, None])[:4].sum(0)
POOLS = ['clitlm', 'clitls', 'clitll', 'clitrm', 'clitrs', 'clitrl', 'clitwm', 'clitws', 'clitwl', 'csoislon', 'csoislop', 'csoipas']
def load(run):
    p = f'{run}/restart/ibis_restart_2010.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p); cb = st.carbon
    lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
    area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj])
    d = xr.open_dataset(f'{run}/out/ibis_yearly_2010.nc'); rh = np.asarray(d['ayco2mic'].values, float)[gi, gj]
    pools = {k: np.asarray(getattr(cb, k), float) for k in POOLS if hasattr(cb, k)}
    return dict(lat=lat, lon=lon, gi=gi, gj=gj, area=area, tot=np.asarray(cb.totcsoi, float), rh=rh, pools=pools, fields=[f for f in dir(cb) if not f.startswith('_')])
D = {k: load(r) for k, r in runs.items()}; g = D[next(iter(D))]; lat, lon, gi, gj, area = g['lat'], g['lon'], g['gi'], g['gj'], g['area']
print('carbon state fields:', [f for f in g['fields'] if f.startswith('c') or f.startswith('tot')][:40])
zone = Z[gi, gj]; ref1 = soc1[gi, gj]; okr = np.isfinite(ref1) & (ref1 > 0)
zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
G = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'), 'temperate': zn('C_oceanic', 'D_temperate'),
     'boreal': zn('D_subarctic', 'D_extreme'), 'tundra + cold dry': zn('E_tundra', 'B_cold'), '90S-30N': lat < 30, '30-60N': (lat >= 30) & (lat < 60), '60-90N': lat >= 60, 'GLOBAL': np.ones_like(zone, bool)}
Pg = lambda x, m: float(np.nansum(x[m] * area[m])) / 1e12
for k, d in D.items():
    P = d['pools']; lit = sum(P[p] for p in POOLS[:9] if p in P); slow = P.get('csoislon', 0 * d['tot']) + P.get('csoislop', 0 * d['tot']); pas = P.get('csoipas', 0 * d['tot']); som = slow + pas
    print(f'\n===== {k}: totcsoi | litter pools | slow (non-prot + prot) | passive | HWSD2 0-1 m | factor on SOM to meet HWSD2 (= (HWSD - litter) / SOM) | turnover: total / Rh, SOM / Rh')
    print(f"{'chunk':22s} {'tot':>6s} {'litter':>7s} {'slow':>6s} {'passive':>8s} {'HWSD':>6s} {'factor':>7s} {'tau_tot':>8s} {'tau_som':>8s}")
    for name, m in G.items():
        mm = m & okr; som_g = Pg(som, mm); lit_g = Pg(lit, mm)
        print(f"{name:22s} {Pg(d['tot'], mm):6.0f} {lit_g:7.0f} {Pg(slow, mm):6.0f} {Pg(pas, mm):8.0f} {Pg(ref1, mm):6.0f} {(Pg(ref1, mm) - lit_g) / max(som_g, 1e-9):7.2f} {Pg(d['tot'], mm) / max(Pg(d['rh'], mm), 1e-9):8.1f} {som_g / max(Pg(d['rh'], mm), 1e-9):8.1f}")
