"""Expected equilibrium wood by chunk under the cover limit (params_31p) and the cover limit + tropical refit (params_31q),
from the s31n 2010 state: per cell, the new cover is fu_new = clip(fu_max(u), fu_min, 0.975) where the limit binds or frees
(ts > 0), 0 where fu_max = 0; wood production scales with cover (prod_new = awood'/awood x prod x fu_new / fu) and the loss
rate is fire (disturbf x mort_j) + 1 / tauwood0'_j per PFT, so eq = prod_new / loss. Humid cells near the cap keep today's
cover. A prediction to judge the cold protocols s31p / s31q against, not a run. Read-only.   usage: python -I gate_expect.py"""
import sys, re, ast, numpy as np, xarray as xr, yaml, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
from ibis.io.pft_types import TYPES
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'; M = f'{S}/meeting'; V = '/shared/nas2/Hangkai/ibis/verify_outputs'
ref = np.load(f'{M}/glob_refs.npz'); Z = ref['zone']; W = ref['wood']; TREE = ref['tree']; CROP = np.nan_to_num(ref['crop'])
MORT = np.array([0.50, 0.35, 0.50, 0.35, 0.50, 0.80, 0.70, 0.75])
run = f'{V}/final_s31n/hindcast'
def params(fn):
    y = yaml.safe_load(open(f'{R}/python-port/configs/params/{fn}')); pp = y['pft']; names = [k for k, v in pp.items() if isinstance(v, dict) and 'tauwood0' in v][:8]
    tau = np.array([float(pp[n]['tauwood0']) for n in names]); aw = np.array([float(pp[n]['awood']) for n in names])
    # type overrides at their parent position in their zones (awood only; tauwood0 is per parent)
    aw_type = {}   # (zone name) -> (pos -> awood)
    for t in TYPES:
        tid, zones, pos = t[0], t[3], t[4] - 1
        if pos < 8 and tid in y['types'] and 'awood' in y['types'][tid]:
            for z in zones: aw_type.setdefault(z, {})[pos] = float(y['types'][tid]['awood'])
    return dict(tau=tau, aw=aw, aw_type=aw_type, pp=pp)
P = {k: params(f) for k, f in (('s31n', 'params_31n.yaml'), ('limit', 'params_31p.yaml'), ('refit', 'params_31q.yaml'))}
acc = {}
for y in (2008, 2010):
    p = f'{run}/restart/ibis_restart_{y}.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p); v = st.vegetation
    lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
    area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj])
    d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc')
    for k, x in (('u', np.asarray(d['aystressu'].values, float)[gi, gj]), ('nppu', np.asarray(d['aynppu'].values, float)[gi, gj]),
                 ('fu', np.asarray(v.fu, float)), ('cw', np.asarray(v.cbiow, float)[:, :8]), ('db', np.asarray(v.disturbf, float))):
        acc.setdefault(k, []).append(x)
A = {k: np.mean(x, 0) for k, x in acc.items()}; u, nppu, fu, cw, db = A['u'], A['nppu'], A['fu'], A['cw'], A['db']
zone = Z[gi, gj]; wmap = W[gi, gj]; wood = cw.sum(1); share = cw / np.maximum(wood[:, None], 1e-9)
pp0 = P['s31n']['pp']; smin, ramp = float(pp0['tree_stress_min']), float(pp0['tree_stress_ramp']); ts = np.clip((u - smin) / ramp, 0, 1)
zname = np.array(ZONES)[np.clip(zone, 0, len(ZONES) - 1)]
def awood_cell(Pk):
    aw = np.broadcast_to(Pk['aw'], (len(u), 8)).copy()
    for z, d_ in Pk['aw_type'].items():
        m = zname == z
        for pos, val in d_.items(): aw[m, pos] = val
    return aw
AW = {k: awood_cell(Pk) for k, Pk in P.items()}
# today's production per PFT: awood_j x NPP_u x (PFT's share of the wood, as a proxy for its share of the upper-canopy NPP)
prod0 = AW['s31n'] * np.maximum(nppu, 0)[:, None] * share
out = {}
for k in ('s31n', 'limit', 'refit'):
    Pk = P[k]; pp = Pk['pp']
    if k == 's31n':
        fu_new = fu; kill = np.where(ts > 0.05, (1 - ts) / np.maximum(ts, 1e-6), 0.0)
    else:
        a, b, pw, fmin = float(pp['tree_cover_min']), float(pp['tree_cover_ramp']), float(pp['tree_cover_power']), float(pp.get('fu_min', 0.05))
        fmax = np.clip((u - a) / b, 0, 1) ** pw
        fu_new = np.where(fmax <= 0, 0.0, np.where(ts < 0.999, np.clip(fmax, fmin, 0.975), np.minimum(fu, np.clip(fmax, fmin, 0.975))))
        kill = np.zeros_like(fu)
    prod = prod0 * (AW[k] / np.maximum(AW['s31n'], 1e-9)) * (fu_new / np.maximum(fu, 1e-6))[:, None]
    loss = db[:, None] * MORT[None, :] + 1.0 / Pk['tau'][None, :] + kill[:, None]
    eq = np.where(fu_new[:, None] > 0, prod / loss, 0.0).sum(1)
    out[k] = dict(eq=eq, fu=fu_new, prod=prod.sum(1))
zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
GROUPS = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'),
          'temperate': zn('C_oceanic', 'D_temperate'), 'boreal': zn('D_subarctic', 'D_extreme'), 'tundra + cold dry': zn('E_tundra', 'B_cold')}
CONT = {'N America': (lon < -30) & (lat > 12), 'S America': (lon < -30) & (lat <= 12), 'Europe': (lon >= -30) & (lon < 60) & (lat > 36),
        'Africa': (lon >= -30) & (lon < 60) & (lat <= 36), 'Asia': (lon >= 60) & ~((lon > 110) & (lat < -10)), 'Australia': (lon > 110) & (lat < -10)}
Pg = lambda x, m: float(np.nansum(x[m] * area[m])) / 1e12; F = lambda x, m: float(np.nansum(x[m] * area[m]) / max(area[m].sum(), 1e-9))
print('expected equilibrium wood (PgC) from the s31n 2008/2010 state: today | eq at today\'s rules | eq under the cover limit | eq under limit + tropical refit; cover: today -> limit; production PgC/yr today -> limit -> refit')
print(f"{'chunk':38s} {'map':>6s} {'now':>6s} | {'eq_n':>6s} {'eq_lim':>6s} {'eq_ref':>6s} | {'fu':>5s} {'->':>5s} | {'prod':>5s} {'lim':>5s} {'ref':>5s}")
rows = [(f'{g} · {c}', gm & cm) for g, gm in GROUPS.items() for c, cm in CONT.items() if Pg(wmap, gm & cm) >= 1.5] + [(g, gm) for g, gm in GROUPS.items()] + [('GLOBAL', np.ones_like(zone, bool))]
for name, m in rows:
    if name == 'humid tropics': print('-' * 38)
    print(f"{name:38s} {Pg(wmap, m):6.1f} {Pg(wood, m):6.1f} | {Pg(out['s31n']['eq'], m):6.1f} {Pg(out['limit']['eq'], m):6.1f} {Pg(out['refit']['eq'], m):6.1f} | "
          f"{F(fu, m):5.2f} {F(out['limit']['fu'], m):5.2f} | {Pg(out['s31n']['prod'], m):5.2f} {Pg(out['limit']['prod'], m):5.2f} {Pg(out['refit']['prod'], m):5.2f}")
