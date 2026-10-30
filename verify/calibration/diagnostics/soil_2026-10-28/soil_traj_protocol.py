"""Wood PgC by world chunk (and the humid Amazon) through the cold protocol: spin-up end 1900, transient 1950 / 2000,
hindcast 2010, for several finals. Shows how far each stock is from its equilibrium (gate_expect) at 2010. Read-only.
    python -I soil_traj_protocol.py s31n s31p"""
import sys, re, ast, numpy as np, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work'; V = '/shared/nas2/Hangkai/ibis/verify_outputs'
ref = np.load(f'{S}/meeting/glob_refs.npz'); Z = ref['zone']; W = ref['wood'] * 0.0
STAGES = [('spinup', 1800), ('spinup', 1900), ('transient', 1950), ('transient', 2000), ('hindcast', 2010)]
def load(tag, stage, y):
    p = f'{V}/final_{tag}/{stage}/restart/ibis_restart_{y}.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p)
    lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
    area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj])
    return lat, lon, gi, gj, area, np.asarray(st.carbon.totcsoi, float)
tags = sys.argv[1:]
lat, lon, gi, gj, area, _ = load(tags[0], 'hindcast', 2010); zone = Z[gi, gj]; wmap = W[gi, gj]
zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
G = {'humid tropics': zn('A_humid'), '60-90N': lat >= 60, '30-60N': (lat >= 30) & (lat < 60),
     'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'), 'temperate': zn('C_oceanic', 'D_temperate'),
     'boreal': zn('D_subarctic', 'D_extreme'), 'tundra + cold dry': zn('E_tundra', 'B_cold'), 'GLOBAL': np.ones_like(zone, bool)}
Pg = lambda x, m: float(np.nansum(x[m] * area[m])) / 1e12
print('soil C (totcsoi) and the slow + passive SOM pools PgC by chunk through the cold protocol (spin-up 1601-1900 at 296 ppm, transient 1901-2000, hindcast 2001-2010); map = Xu & Saatchi 2021')
print(f"{'chunk':24s} {'map':>6s} | " + ' | '.join(f"{t:>30s}" for t in tags)); print(f"{'':24s} {'':>6s} | " + ' | '.join(' '.join(f'{y:>5d}' for _, y in STAGES) for t in tags))
D = {t: {y: load(t, s, y)[-1] for s, y in STAGES} for t in tags}
for name, m in G.items():
    print(f"{name:24s} {Pg(wmap, m):6.1f} | " + ' | '.join(' '.join(f'{Pg(D[t][y], m):5.1f}' for _, y in STAGES) for t in tags))
print('\nten-year change 2000 -> 2010 (PgC yr-1, the soil part of the sink): ' + ', '.join(f"{t} {(Pg(D[t][2010], G['GLOBAL']) - Pg(D[t][2000], G['GLOBAL'])) / 10:+.2f}" for t in tags))
