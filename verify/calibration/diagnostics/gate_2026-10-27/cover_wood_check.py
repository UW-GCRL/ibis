"""Cover and wood of several run states against the products: by Köppen group x continent (tree cover on natural land vs
the CLM product, wood PgC vs Xu & Saatchi, wood per unit cover) and, in the tropics, by annual-stress bin; plus the wood
trajectory of the partial-gate cells over the run's years. Runs are given as label=run_dir (restart/ibis_restart_YYYY.npz
and out/ibis_yearly_YYYY.nc inside). Read-only.
    python -I cover_wood_check.py YEAR label=dir [label=dir ...] [--years y1,y2,...]"""
import sys, re, ast, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'; M = f'{S}/meeting'
ref = np.load(f'{M}/glob_refs.npz'); Z = ref['zone']; TREE = ref['tree']; CROP = np.nan_to_num(ref['crop']); W = ref['wood']
args = [a for a in sys.argv[1:] if not a.startswith('--')]; year = int(args[0]); runs = dict(a.split('=', 1) for a in args[1:])
traj = [int(y) for y in next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--years=')), '').split(',') if y]
def load(run, y):
    p = f'{run}/restart/ibis_restart_{y}.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p); v = st.vegetation
    lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
    area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj])
    d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); u = np.asarray(d['aystressu'].values, float)[gi, gj]
    return dict(gi=gi, gj=gj, lat=lat, lon=lon, area=area, u=u, fu=np.asarray(v.fu, float), wood=np.asarray(v.cbiow, float)[:, :8].sum(1),
                laiu=np.asarray(v.plai, float)[:, :8].sum(1) / np.maximum(np.asarray(v.fu, float), 1e-6), nppu=np.asarray(d['aynppu'].values, float)[gi, gj])
D = {k: load(r, year) for k, r in runs.items()}; k0 = next(iter(D)); g = D[k0]; gi, gj, area, lat, lon = g['gi'], g['gj'], g['area'], g['lat'], g['lon']
zone = Z[gi, gj]; tree = TREE[gi, gj]; crop = CROP[gi, gj]; wmap = W[gi, gj]; tree_nat = np.clip(tree / np.clip(1 - crop, 0.05, 1), 0, 1)
zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
GROUPS = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'),
          'temperate': zn('C_oceanic', 'D_temperate'), 'boreal': zn('D_subarctic', 'D_extreme'), 'tundra + cold dry': zn('E_tundra', 'B_cold')}
CONT = {'N America': (lon < -30) & (lat > 12), 'S America': (lon < -30) & (lat <= 12), 'Europe': (lon >= -30) & (lon < 60) & (lat > 36),
        'Africa': (lon >= -30) & (lon < 60) & (lat <= 36), 'Asia': (lon >= 60) & ~((lon > 110) & (lat < -10)), 'Australia': (lon > 110) & (lat < -10)}
nat = (crop < 0.3) & np.isfinite(tree) & (area > 0)
P = lambda x, m: float(np.nansum(x[m] * area[m])) / 1e12; F = lambda x, m: float(np.nansum(x[m] * area[m]) / max(area[m].sum(), 1e-9))
print(f'year {year}; runs: ' + ', '.join(f'{k} = {r}' for k, r in runs.items()))
print(f"\n--- by chunk: product tree cover on natural land (crop < 0.3) | model fu per run | wood PgC: map | per run (x map) | wood per unit cover kgC/m2: map | per run")
hdr = f"{'chunk':38s} {'tree':>5s} | " + ' '.join(f'{k:>6s}' for k in D) + f" | {'map':>6s} | " + ' '.join(f'{k:>12s}' for k in D) + f" | {'map':>5s} | " + ' '.join(f'{k:>6s}' for k in D)
print(hdr)
rows = [(f'{gname} · {c}', gm & cm) for gname, gm in GROUPS.items() for c, cm in CONT.items() if P(wmap, gm & cm) >= 1.5] + [(gname, gm) for gname, gm in GROUPS.items()] + [('GLOBAL', np.ones_like(zone, bool))]
for name, m in rows:
    if name in GROUPS and name == 'humid tropics': print('-' * 38)
    mn = m & nat
    print(f"{name:38s} {F(tree_nat, mn):5.2f} | " + ' '.join(f'{F(D[k]["fu"], mn):6.2f}' for k in D) + f" | {P(wmap, m):6.1f} | "
          + ' '.join(f'{P(D[k]["wood"], m):5.1f} ({P(D[k]["wood"], m) / max(P(wmap, m), 1e-9):4.2f})' for k in D)
          + f" | {P(wmap, m) / max(P(tree, m), 1e-9):5.1f} | " + ' '.join(f'{P(D[k]["wood"], m) / max(P(D[k]["fu"], m), 1e-9):6.1f}' for k in D))
print(f"\n--- tropics (A, B_hot), natural land, by the control's annual stress u: area % | product tree | fu per run | wood kgC/m2 per run | map wood | upper LAI per cover per run")
u0 = D[k0]['u']; BINS = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]; tm = zn('A_humid', 'A_savanna', 'B_hot') & nat
for lo, hi in zip(BINS[:-1], BINS[1:]):
    m = tm & (u0 >= lo) & (u0 < hi)
    if m.sum() < 20: continue
    print(f"  u {lo:.1f}-{hi:.2f} | {100 * area[m].sum() / area[tm].sum():5.1f} % | {F(tree_nat, m):5.2f} | " + ' '.join(f'{F(D[k]["fu"], m):5.2f}' for k in D) + ' | '
          + ' '.join(f'{F(D[k]["wood"], m):5.2f}' for k in D) + f' | {F(wmap, m):5.2f} | ' + ' '.join(f'{F(D[k]["laiu"], m):4.1f}' for k in D))
if traj:
    print(f"\n--- trajectory of the partial-gate cells of the control (0 < ts < 1 at {year}, ts = clip((u - 0.5) / 0.3)): wood PgC and mean fu by year")
    ts0 = np.clip((u0 - 0.5) / 0.3, 0, 1); pm = nat & (ts0 > 0) & (ts0 < 0.999)
    print(f'  cells: {100 * area[pm].sum() / area[nat].sum():.1f} % of natural land; product tree {F(tree_nat, pm):.2f}; map wood {P(wmap, pm):.1f} PgC')
    for k, r in runs.items():
        vals = []
        for y in traj:
            try:
                d = load(r, y); vals.append(f'{y}: {P(d["wood"], pm):5.2f} PgC fu {F(d["fu"], pm):.3f}')
            except Exception as e:
                vals.append(f'{y}: n/a')
        print(f'  {k:10s} ' + ' | '.join(vals))
