"""Why a chunk's wood is off: cover vs density vs residence time. Per Köppen group x continent, for each run: map wood and
tree cover (Xu & Saatchi; tree-cover product), model wood, upper-canopy cover fu, wood per unit cover, upper-canopy NPP
(2006-2010), implied wood residence time wood / (0.5 NPP_u), fire vs GFED5, GPP vs FLUXCOM, crop fraction, and the share
of the chunk's area whose canopy is open (fu < 0.5) with its wood deficit. Read-only.    python wood_decomp.py RUN [RUN ...]"""
import sys, re, ast, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open('/shared/nas2/Hangkai/ibis-port/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'; M = f'{S}/meeting'; V = '/shared/nas2/Hangkai/ibis/verify_outputs'
ref = np.load(f'{M}/glob_refs.npz'); lat, lon = ref['lat'], ref['lon']; LO, LA = np.meshgrid(lon, lat)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(LA)) * np.nan_to_num(ref['lf']); W = ref['wood']; Z = ref['zone']; TREE = ref['tree']; CROP = np.nan_to_num(ref['crop'])
zn = lambda *names: np.isin(Z, [ZONES.index(n) for n in names])
GROUPS = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'),
          'temperate': zn(*[z for z in ZONES if z.startswith('C_oceanic') or z.startswith('D_temp') or z.startswith('D_cont')]),
          'boreal': zn(*[z for z in ZONES if 'subarctic' in z or 'boreal' in z]), 'tundra + cold dry': zn('E_tundra', 'B_cold')}
CONT = {'N America': (LO < -30) & (LA > 12), 'S America': (LO < -30) & (LA <= 12), 'Europe': (LO >= -30) & (LO < 60) & (LA > 36),
        'Africa': (LO >= -30) & (LO < 60) & (LA <= 36), 'Asia': (LO >= 60) & ~((LO > 110) & (LA < -10)), 'Australia': (LO > 110) & (LA < -10)}
land = np.isfinite(W) & (np.nan_to_num(ref['lf']) > 0)
def am(x, m): ok = m & land & np.isfinite(x); return float((x[ok] * area[ok]).sum() / area[ok].sum())
def tot(x, m): ok = m & land & np.isfinite(x); return float((x[ok] * area[ok]).sum()) / 1e12
chunks = [(f'{g} · {c}', gm & cm) for g, gm in GROUPS.items() for c, cm in CONT.items() if tot(W, gm & cm) >= 1.5]
for run in sys.argv[1:]:
    g = np.load(f'{M}/glob_{run}.npz'); wood, fu, burn, gpp = g['wood'], g['fu'], g['disturbf'], g['gpp']
    ys = [xr.open_dataset(f'{V}/final_{run}/hindcast/out/ibis_yearly_{y}.nc') for y in range(2006, 2011)]
    nppu = np.mean([d['aynppu'].values for d in ys], 0); laiu = np.mean([d['laiu'].values for d in ys], 0)
    TAU = {'humid tropics': 25, 'savanna + hot dry': 25, 'subtrop/Med/highland': 25, 'temperate': 50, 'boreal': 100, 'tundra + cold dry': 100} if run in ('s31j', 's31l') else {'humid tropics': 35, 'savanna + hot dry': 35, 'subtrop/Med/highland': 50, 'temperate': 60, 'boreal': 80, 'tundra + cold dry': 80}
    print(f'\n===== {run}: wood PgC map/model (x) | cover: product tree vs model fu | density kgC per m2 of cover: map, model | NPP_u kgC/m2/yr, implied tau = wood/(0.5 NPP_u) vs tauwood0 | fire % GFED5/model | GPP x FLUXCOM | LAI_u | crop | open cells (fu<0.5): % area, their wood share')
    print(f"{'chunk':38s} {'map':>5s} {'mod':>5s} {'x':>5s} | {'tree':>5s} {'fu':>5s} | {'d_map':>5s} {'d_mod':>5s} | {'NPPu':>5s} {'tau':>4s} {'tau0':>4s} | {'gfed':>5s} {'fire':>5s} | {'GPPx':>5s} | {'LAIu':>4s} | {'crop':>4s} | {'open%':>5s} {'wood%':>5s}")
    for name, m in chunks:
        grp = name.split(' · ')[0]; op = m & land & (fu < 0.5)
        tree_c, fu_c = am(TREE, m), am(fu, m); d_map = tot(W, m) / max(tot(TREE, m), 1e-9); d_mod = tot(wood, m) / max(tot(fu, m), 1e-9)
        npu = am(nppu, m); tau = am(wood, m) / max(0.5 * npu, 1e-9)
        print(f'{name:38s} {tot(W, m):5.1f} {tot(wood, m):5.1f} {tot(wood, m) / tot(W, m):5.2f} | {tree_c:5.2f} {fu_c:5.2f} | {d_map:5.1f} {d_mod:5.1f} | {npu:5.2f} {tau:4.0f} {TAU[grp]:4d} | {100 * am(ref["burn_gfed5"], m):5.2f} {100 * am(burn, m):5.2f} | {am(gpp, m) / am(ref["gpp"], m):5.2f} | {am(laiu, m):4.1f} | {am(CROP, m):4.2f} | {100 * area[op].sum() / area[m & land].sum():5.1f} {100 * tot(W, op) / tot(W, m):5.1f}')
