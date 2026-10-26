"""Wood budget by chunk (PgC/yr, 2008 and 2010 mean): production 0.5 x upper-canopy NPP, the survival gate's wood kill
cbiow x (1-ts)/ts (ts = clip((aystressu - min)/ramp, 0, 1)), fire wood loss disturbf x mort_j x cbiow_j, and turnover
cbiow_j / tauwood0_j; plus the chunk's area share with ts = 0 (trees eliminated every year) and 0 < ts < 1. Read-only."""
import sys, re, ast, numpy as np, xarray as xr, yaml, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'; M = f'{S}/meeting'; V = '/shared/nas2/Hangkai/ibis/verify_outputs'
ref = np.load(f'{M}/glob_refs.npz'); glat, glon = ref['lat'], ref['lon']; W = ref['wood']; Z = ref['zone']
MORT = np.array([0.50, 0.35, 0.50, 0.35, 0.50, 0.80, 0.70, 0.75])
for tag in sys.argv[1:]:
    run = f'{V}/final_{tag}/hindcast'; y0 = yaml.safe_load(open(yaml.safe_load(open(f'{run}/cfg.yaml'))['params_file'])); pp = y0['pft']
    smin, ramp = float(pp['tree_stress_min']), float(pp['tree_stress_ramp']); names = [k for k, v in pp.items() if isinstance(v, dict) and 'tauwood0' in v][:8]
    TAU = np.array([float(pp[n]['tauwood0']) for n in names]); AW = np.array([float(pp[n]['awood']) for n in names])
    acc = {}
    for y in (2008, 2010):
        p = f'{run}/restart/ibis_restart_{y}.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p); v = st.vegetation
        lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
        area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj])
        d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); u = np.asarray(d['aystressu'].values, float)[gi, gj]; nppu = np.asarray(d['aynppu'].values, float)[gi, gj]
        ts = np.clip((u - smin) / ramp, 0, 1); cw = np.asarray(v.cbiow, float)[:, :8]; db = np.asarray(v.disturbf, float)
        wood = cw.sum(1); kill = np.where(ts > 0.05, wood * (1 - ts) / np.maximum(ts, 1e-6), 0.0)
        fire = db * (cw * MORT[None, :]).sum(1); turn = (cw / TAU[None, :]).sum(1); prod = 0.5 * np.maximum(nppu, 0)
        zone = Z[gi, gj]; wmap = W[gi, gj]
        for k, x in (('wood', wood), ('map', wmap), ('prod', prod), ('gate', kill), ('fire', fire), ('turn', turn), ('ts0', (ts <= 0).astype(float)), ('tspart', ((ts > 0) & (ts < 0.999)).astype(float)), ('one', np.ones_like(wood))):
            acc.setdefault(k, []).append(x)
    A = {k: np.mean(v, 0) for k, v in acc.items()}
    LO, LA = lon, lat
    zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
    GROUPS = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'),
              'temperate': zn(*[z for z in ZONES if z.startswith('C_oceanic') or z.startswith('D_temp') or z.startswith('D_cont')]),
              'boreal': zn(*[z for z in ZONES if 'subarctic' in z or 'boreal' in z]), 'tundra + cold dry': zn('E_tundra', 'B_cold')}
    CONT = {'N America': (LO < -30) & (LA > 12), 'S America': (LO < -30) & (LA <= 12), 'Europe': (LO >= -30) & (LO < 60) & (LA > 36),
            'Africa': (LO >= -30) & (LO < 60) & (LA <= 36), 'Asia': (LO >= 60) & ~((LO > 110) & (LA < -10)), 'Australia': (LO > 110) & (LA < -10)}
    P = lambda x, m: float(np.nansum(x[m] * area[m])) / 1e12; F = lambda x, m: float(np.nansum(x[m] * area[m]) / area[m].sum())
    print(f'\n===== {tag} (gate {smin}/{ramp}; tauwood0 {TAU.astype(int).tolist()}): PgC and PgC/yr; ts=0 and 0<ts<1 are area shares (%)')
    print(f"{'chunk':38s} {'map':>5s} {'wood':>5s} | {'prod':>5s} {'gate':>5s} {'fire':>5s} {'turn':>5s} | {'gate/prod':>9s} | {'ts=0':>5s} {'part':>5s} | {'eq':>5s} {'eq-gate':>7s}   (eq = prod / loss rate with and without the gate, PgC)")
    rows = [(f'{g} · {c}', gm & cm) for g, gm in GROUPS.items() for c, cm in CONT.items() if P(A['map'], gm & cm) >= 1.5] + [(g, gm) for g, gm in GROUPS.items()] + [('GLOBAL', np.ones_like(zone, bool))]
    for name, m in rows:
        if name in GROUPS: print('-' * 38) if name == 'humid tropics' else None
        w, pr, ga, fi, tu = (P(A[k], m) for k in ('wood', 'prod', 'gate', 'fire', 'turn')); eq = pr / max((ga + fi + tu) / max(w, 1e-9), 1e-9); eqn = pr / max((fi + tu) / max(w, 1e-9), 1e-9)
        print(f"{name:38s} {P(A['map'], m):5.1f} {w:5.1f} | {pr:5.2f} {ga:5.2f} {fi:5.2f} {tu:5.2f} | {ga / max(pr, 1e-9):9.2f} | {100 * F(A['ts0'], m):5.1f} {100 * F(A['tspart'], m):5.1f} | {eq:5.1f} {eqn:7.1f}")
