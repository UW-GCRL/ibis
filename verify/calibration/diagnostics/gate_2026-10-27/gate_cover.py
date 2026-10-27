"""Cover-limit design for the tree survival gate (CALIBRATION.md 67 -> 70): how the product's tree cover relates to the
model's annual-mean upper-canopy soil-water stress u, on natural land, so that a water-limited MAXIMUM cover
fu_max(u) = clip((u - a) / b, 0, 1) ** p can replace the yearly kill. Also the current fu-wood relation and what the
partial-gate cells hold. Read-only. usage: python -I gate_cover.py s31n [years...]"""
import sys, re, ast, numpy as np, xarray as xr, yaml, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'; M = f'{S}/meeting'; V = '/shared/nas2/Hangkai/ibis/verify_outputs'
ref = np.load(f'{M}/glob_refs.npz'); Z = ref['zone']; TREE = ref['tree']; CROP = np.nan_to_num(ref['crop']); W = ref['wood']
tag = sys.argv[1]; years = [int(y) for y in sys.argv[2:]] or [2010]
run = f'{V}/final_{tag}/hindcast'; pp = yaml.safe_load(open(yaml.safe_load(open(f'{run}/cfg.yaml'))['params_file']))['pft']
smin, ramp = float(pp['tree_stress_min']), float(pp['tree_stress_ramp']); fu_k = float(pp.get('fu_k', 1.0)); fu_min = float(pp.get('fu_min', 0.25))
acc = {}
for y in years:
    p = f'{run}/restart/ibis_restart_{y}.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p); v = st.vegetation
    lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
    area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj])
    d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); u = np.asarray(d['aystressu'].values, float)[gi, gj]
    for k, x in (('u', u), ('fu', np.asarray(v.fu, float)), ('wood', np.asarray(v.cbiow, float)[:, :8].sum(1)), ('db', np.asarray(v.disturbf, float))):
        acc.setdefault(k, []).append(x)
A = {k: np.mean(x, 0) for k, x in acc.items()}; u, fu, wood, db = A['u'], A['fu'], A['wood'], A['db']
zone = Z[gi, gj]; tree = TREE[gi, gj]; crop = CROP[gi, gj]; wmap = W[gi, gj]
ts = np.clip((u - smin) / ramp, 0, 1); nat = np.clip(1 - crop, 0.05, 1); tree_nat = np.clip(tree / nat, 0, 1)
zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
GROUPS = {'tropics (A, B_hot)': zn('A_humid', 'A_savanna', 'B_hot'), 'C zones': zn('C_med', 'C_subtrop', 'C_highland', 'C_oceanic'),
          'D zones': zn('D_temperate', 'D_subarctic', 'D_extreme', 'B_cold'), 'all but E': ~zn('E_tundra', 'E_ice', 'none')}
ok = np.isfinite(u) & np.isfinite(tree) & (area > 0) & (crop < 0.3)
F = lambda x, m: float(np.nansum(x[m] * area[m]) / max(area[m].sum(), 1e-9))
def pct(x, m, q):
    xs, ws = x[m], area[m]; o = np.argsort(xs); c = np.cumsum(ws[o]) / ws[o].sum(); return float(xs[o][np.searchsorted(c, q / 100)])
print(f'{tag} {years}: gate {smin}/{ramp}, fu_k {fu_k}, fu_min {fu_min}; natural land = crop < 0.3; tree_nat = product tree cover / (1 - crop)')
print('\n--- fu vs wood (the cover rule): wood bins, mean fu, fu(wood) = (1-exp(-w/fu_k))/(1-exp(-7.5/fu_k)) x (1 - disturb)')
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1), (1, 2), (2, 4), (4, 8), (8, 30)):
    m = ok & (wood >= lo) & (wood < hi)
    if m.sum(): print(f'  wood {lo:4.1f}-{hi:4.1f}: n {m.sum():5d}  fu {F(fu, m):.3f}  rule {F((1 - np.exp(-wood / fu_k)) / (1 - np.exp(-7.5 / fu_k)) * (1 - 0.5 * db - 0.005), m):.3f}  crop {F(crop, m):.2f}  ts {F(ts, m):.2f}')
BINS = [0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
for g, gm in GROUPS.items():
    print(f'\n--- {g}: stress bin | area %, ts | product tree_nat: mean p50 p75 p90 | model fu, wood kgC/m2, wood/fu | map wood')
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = ok & gm & (u >= lo) & (u < hi)
        if m.sum() < 20: continue
        print(f'  u {lo:.2f}-{hi:.2f} | {100 * area[m].sum() / area[ok & gm].sum():5.1f} % ts {F(ts, m):.2f} | {F(tree_nat, m):.2f} {pct(tree_nat, m, 50):.2f} {pct(tree_nat, m, 75):.2f} {pct(tree_nat, m, 90):.2f} | '
              f'{F(fu, m):.2f} {F(wood, m):5.2f} {F(wood, m) / max(F(fu, m), 1e-6):5.1f} | {F(wmap, m):5.1f}')
# fit fu_max(u) = clip((u - a)/b, 0, 1)^p to the p75 envelope of tree_nat over bins (area-weighted), all but E
gm = GROUPS['all but E']; xs, ys, ws = [], [], []
for lo, hi in zip(BINS[:-1], BINS[1:]):
    m = ok & gm & (u >= lo) & (u < hi)
    if m.sum() >= 20: xs.append(F(u, m)); ys.append(pct(tree_nat, m, 75)); ws.append(area[m].sum())
xs, ys, ws = map(np.array, (xs, ys, ws)); best = None
for a in np.arange(0.2, 0.7, 0.02):
    for b in np.arange(0.1, 0.6, 0.02):
        for pw in (0.5, 0.75, 1.0, 1.5, 2.0):
            f = np.clip((xs - a) / b, 0, 1) ** pw; e = float(np.sum(ws * (f - ys) ** 2) / ws.sum())
            if best is None or e < best[0]: best = (e, a, b, pw)
e, a, b, pw = best; print(f'\nfit to the p75 envelope (all but E): fu_max = clip((u - {a:.2f}) / {b:.2f}, 0, 1) ^ {pw}   rms {np.sqrt(e):.3f}')
fmax = np.clip((u - a) / b, 0, 1) ** pw
capped = ok & (fmax < fu) & (ts > 0); freed = ok & (ts < 1) & (ts > 0) & (fmax >= fu)
print(f'cells the limit would cap now (fu_max < fu): {100 * area[capped].sum() / area[ok].sum():.1f} % of natural land, mean fu {F(fu, capped):.2f} -> {F(fmax, capped):.2f}, product {F(tree_nat, capped):.2f}')
print(f'partial-gate cells the limit frees (fu_max >= fu, 0<ts<1): {100 * area[freed].sum() / area[ok].sum():.1f} %, fu {F(fu, freed):.2f}, limit {F(fmax, freed):.2f}, product {F(tree_nat, freed):.2f}, wood {F(wood, freed):.2f} vs map {F(wmap, freed):.2f}')
print(f'eliminated (fu_max = 0): {100 * area[ok & (fmax <= 0)].sum() / area[ok].sum():.1f} % (today ts = 0: {100 * area[ok & (ts <= 0)].sum() / area[ok].sum():.1f} %); product tree there {F(tree_nat, ok & (fmax <= 0)):.2f}')
for g, m in GROUPS.items():
    mm = ok & m; print(f'  {g:20s}: cover now {F(fu, mm):.3f}, with limit {F(np.minimum(fu, np.maximum(fu_min, fmax)), mm):.3f}, product {F(tree_nat, mm):.3f}')
