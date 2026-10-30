"""Fit the cover-limit ramp fu_max(u) = clip((u - a) / b, 0, 1) ^ p to the product's tree cover (on natural land, crop < 0.3)
against the model's annual-mean upper-canopy stress u, per zone group, on a final run's 2008/2010 state. For each
group: bins of u with area %, product tree_nat mean / p50 / p75, the tropical ramp's value, and the best fit to the
p75 and to the p50 envelopes. Read-only.   usage: python -I ramp_fit_zones.py s31n 2008 2010"""
import sys, re, ast, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work'; M = f'{S}/meeting'; V = '/shared/nas2/Hangkai/ibis/verify_outputs'
ref = np.load(f'{M}/glob_refs.npz'); Z = ref['zone']; TREE = ref['tree']; CROP = np.nan_to_num(ref['crop'])
tag = sys.argv[1]; years = [int(y) for y in sys.argv[2:]] or [2010]; run = f'{V}/final_{tag}/hindcast'
acc = []
for y in years:
    p = f'{run}/restart/ibis_restart_{y}.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p)
    lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
    area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj])
    d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); acc.append(np.asarray(d['aystressu'].values, float)[gi, gj])
u = np.mean(acc, 0); zone = Z[gi, gj]; tree = TREE[gi, gj]; crop = CROP[gi, gj]; tree_nat = np.clip(tree / np.clip(1 - crop, 0.05, 1), 0, 1)
ok = np.isfinite(u) & np.isfinite(tree) & (area > 0) & (crop < 0.3)
zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
GROUPS = {'A_humid': zn('A_humid'), 'A_savanna': zn('A_savanna'), 'B_hot': zn('B_hot'), 'tropics (A + B_hot)': zn('A_humid', 'A_savanna', 'B_hot'),
          'C zones': zn('C_med', 'C_subtrop', 'C_highland', 'C_oceanic'), 'D_temperate': zn('D_temperate'), 'D_subarctic + D_extreme': zn('D_subarctic', 'D_extreme'),
          'B_cold': zn('B_cold'), 'E_tundra': zn('E_tundra'), 'cold (D + B_cold + E_tundra)': zn('D_temperate', 'D_subarctic', 'D_extreme', 'B_cold', 'E_tundra')}
F = lambda x, m: float(np.nansum(x[m] * area[m]) / max(area[m].sum(), 1e-9))
def pct(x, m, q):
    xs, ws = x[m], area[m]; o = np.argsort(xs); c = np.cumsum(ws[o]) / ws[o].sum(); return float(xs[o][np.searchsorted(c, q / 100)])
trop = lambda x: np.clip((x - 0.30) / 0.65, 0, 1) ** 1.75
BINS = [0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
def fit(xs, ys, ws):
    best = None
    for a in np.arange(0.2, 0.8, 0.02):
        for b in np.arange(0.1, 0.7, 0.02):
            for pw in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
                f = np.clip((xs - a) / b, 0, 1) ** pw; e = float(np.sum(ws * (f - ys) ** 2) / ws.sum())
                if best is None or e < best[0]: best = (e, a, b, pw)
    return best
print(f'{tag} {years}: product tree cover on natural land vs model annual stress u, by zone group; tropical ramp = clip((u - 0.30) / 0.65)^1.75')
for g, gm in GROUPS.items():
    xs, y50, y75, ws = [], [], [], []
    print(f'\n--- {g}: {100 * area[ok & gm].sum() / area[ok].sum():.1f} % of natural land | u bin: area % | product mean p50 p75 | tropical ramp')
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = ok & gm & (u >= lo) & (u < hi)
        if m.sum() < 20: continue
        xs.append(F(u, m)); y50.append(pct(tree_nat, m, 50)); y75.append(pct(tree_nat, m, 75)); ws.append(area[m].sum())
        print(f'  u {lo:.2f}-{hi:.2f} | {100 * area[m].sum() / area[ok & gm].sum():5.1f} % | {F(tree_nat, m):.2f} {y50[-1]:.2f} {y75[-1]:.2f} | {trop(xs[-1]):.2f}')
    xs, y50, y75, ws = map(np.array, (xs, y50, y75, ws))
    for nm, ys in (('p75', y75), ('p50', y50)):
        e, a, b, pw = fit(xs, ys, ws); f = np.clip((xs - a) / b, 0, 1) ** pw
        et = float(np.sum(ws * (trop(xs) - ys) ** 2) / ws.sum())
        print(f'  fit to {nm}: a {a:.2f} b {b:.2f} p {pw}  rms {np.sqrt(e):.3f} (tropical ramp rms {np.sqrt(et):.3f}); mean cover under fit {F(np.clip((u[ok & gm] - a) / b, 0, 1) ** pw * area[ok & gm], np.ones(int((ok & gm).sum()), bool)) if False else float(np.sum(np.clip((u[ok & gm] - a) / b, 0, 1) ** pw * area[ok & gm]) / area[ok & gm].sum()):.3f} vs product {F(tree_nat, ok & gm):.3f} vs tropical ramp {float(np.sum(trop(u[ok & gm]) * area[ok & gm]) / area[ok & gm].sum()):.3f}')
