"""Why the global run's GPP is 1.6x the towers' at the same cells: a chain of controlled runs (CALIBRATION.md 47).

    python verify/calibration/diagnostics/gpp_chain.py --params P.yaml --run RUN_DIR --out OUT_DIR [--batch 1] [--years 2008-2010]

Every arm uses the same parameters, the historical CO2 series and the state of the global run RUN_DIR at the tower
cells at the end of the year before the first; the first year is warm-up, the rest are scored. From the global run to
the tower, one ingredient at a time:
  G  the global run itself (RUN_DIR/out yearly GPP at the cell)
  F  the global vegetation frozen: composition, leaf, wood and root pools and cover (fu, fl) of the start state held,
     GSWP3 at the cell, the cell's cropland share -- what the global run does with its vegetation held
  L  F with every PFT's leaf carbon scaled so the cell's leaf area is the tower prescription's (site attributes for
     forests, else the MODIS cell maximum -- site_fit's attrs-modis rule)
  V  the tower prescription (site_run.prescribe composition, its leaf area, cover from the model's rules as in
     site_fit), GSWP3 at the cell, cropland share 1 at CRO towers else 0: the site baseline on gridded weather
  S  V on the tower's weather (TowerForcingReader): the site baseline the calibration scores
  T  the tower (mean of the night- and day-time partitions, months with NEE QC >= 0.5)
G/F: holding the vegetation; F/L: the leaf area; L/V: composition and cover (the lower canopy under trees, trees at
open sites, the cover rules); V/S: gridded vs tower weather; S/T: the fit.
"""
import sys, os, json, time, types, argparse, numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, '/shared/nas2/Hangkai/ibis-port/verify/calibration'); sys.path.insert(0, '/shared/nas2/Hangkai/ibis-port/python-port')
import jax, jax.numpy as jnp, equinox as eqx
import site_fit as SF
from fit_carbon import LATV, LONV, MASK, GSWP3, NPOI_FULL, modis_at
import site_transient; from site_transient import one_year
from site_run import prescribe, tower_mat, enforce_prescription, META, SEED_LAI, SEED_WOOD
from site_attributes import prescribe_from_attributes
from tower_forcing import TowerForcingReader
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
from ibis.io.co2 import co2_molmol

ap = argparse.ArgumentParser()
ap.add_argument('--params', required=True); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--batch', type=int, default=1); ap.add_argument('--years', default='2008-2010'); ap.add_argument('--arms', default='F,L,V,S')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
y0, y1 = (int(v) for v in a.years.split('-')); YEARS = list(range(y0, y1 + 1)); SCORED = YEARS[1:]
SF.configure_years(types.SimpleNamespace(years=a.years, hold_last=0))
line = open(f'{os.path.dirname(SF.__file__)}/site_data/site_batches.txt').read().splitlines()[a.batch - 1].split()
sites = [s for s in line[1].split(',') if s in META.index and META.loc[s, 'IGBP'] not in ('WET', 'SNO')]
site_transient.apply_co2 = lambda q, y, series=None: eqx.tree_at(lambda z: z.atm.co2conc, q, jnp.asarray(co2_molmol(y), jnp.float32))
ns = len(sites); print(f'{ns} towers, years {YEARS} (scored {SCORED})', flush=True)

m = np.array(xr.open_dataset(MASK).landfrac.values) > 0; posmap = {g: p for p, g in enumerate(np.where(m.ravel())[0])}
i = np.array([int(np.argmin(np.abs(LATV - META.loc[s, 'LOCATION_LAT']))) for s in sites], np.int32)
j = np.array([int(np.argmin(np.abs(LONV - META.loc[s, 'LOCATION_LONG']))) for s in sites], np.int32)
sel = np.array([posmap[ii * 720 + jj] for ii, jj in zip(i, j)]); lat_rad = np.deg2rad(LATV[i]).astype(np.float32)
from ibis.state.stress_accum import create_stress_accum; from ibis.state.et_parts import create_et_parts; from ibis.state.seasonal import create_seasonal
tmpl = create_ibis_state(GridConfig(npoi=NPOI_FULL, nbedrock=3))          # with the trailing accumulators the global run carries
for f_, v_ in (('stress', create_stress_accum(NPOI_FULL)), ('etparts', create_et_parts(NPOI_FULL)), ('seasonal', create_seasonal(NPOI_FULL, 5.0))):
    tmpl = eqx.tree_at(lambda s: getattr(s, f_), tmpl, v_, is_leaf=lambda x: x is None)
full, _ = load_state(f'{a.run}/restart/ibis_restart_{y0 - 1}.npz', tmpl)
cut = lambda x: x[sel] if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL else x
st0 = _cast_tree_f32(jax.tree.map(cut, full)); del full
pdict = load_physics_params(a.params)
cf_grid = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz')['cropfrac'][i, j].clip(0, 1).astype(np.float32)
cf_tower = np.array([1.0 if META.loc[s, 'IGBP'] == 'CRO' else 0.0 for s in sites], np.float32)
consts = PhysicalConstants(); g3 = GSWP3Reader(GSWP3)

def params(cf):
    sp, pft, _ = build_all_params(ns, lat_rad, pdict, cropfrac=cf if cf.any() else None); return sp, pft

# --- the tower prescription (site_fit.setup, attrs-modis leaf area)
sp_t, pft_t = params(cf_tower); specla = np.asarray(pft_t.specla)
mask_t = np.zeros((ns, 12), np.float32); cb_t = np.asarray(st0.vegetation.cbiol).copy(); cw_t = np.asarray(st0.vegetation.cbiow).copy(); cr_t = np.asarray(st0.vegetation.cbior).copy()
fu_t = np.zeros(ns, np.float32); fl_t = np.zeros(ns, np.float32); lai_t = np.zeros(ns); src = {}; modis = np.asarray(modis_at(i, j))
for n, s in enumerate(sites):
    mat = META.loc[s, 'MAT'] if 'MAT' in META else np.nan
    if not (isinstance(mat, (int, float)) and np.isfinite(mat)): mat = tower_mat(s)
    pfts, fu_t[n], fl_t[n], _ = prescribe(META.loc[s, 'IGBP'], float(META.loc[s, 'LOCATION_LAT']), float(mat) if pd.notna(mat) else None)
    for p in range(12):
        if p in pfts:
            mask_t[n, p] = 1.0
            if cb_t[n, p] * specla[p] < 0.3: cb_t[n, p] = SEED_LAI[p] / specla[p]; cw_t[n, p] = SEED_WOOD[p]; cr_t[n, p] = 0.5
        else: cb_t[n, p] = cw_t[n, p] = cr_t[n, p] = 0.0
    at = prescribe_from_attributes(s); L = None
    if at is not None and np.isfinite(at['lai_max']) and at['lai_max'] > 0 and META.loc[s, 'IGBP'] not in ('OSH', 'CSH', 'SAV', 'WSA', 'GRA', 'CRO'):
        L = float(at['lai_max']); src[s] = 'attributes'
    elif np.isfinite(modis[n]) and modis[n] > 0: L = float(modis[n]); src[s] = 'MODIS'
    if L is not None: cb_t[n] = mask_t[n] / max(mask_t[n].sum(), 1) * L / np.maximum(specla, 1e-6)
    else: src[s] = 'state'
    lai_t[n] = float((cb_t[n] * specla).sum())

# --- the global vegetation of the start state
cb_g = np.asarray(st0.vegetation.cbiol); ex_g = np.asarray(st0.vegetation.exist); fu_g = np.asarray(st0.vegetation.fu); fl_g = np.asarray(st0.vegetation.fl)
lai_g = (cb_g * specla).sum(1)

def arm(name):
    t0 = time.time(); st = st0
    if name in ('F', 'L'):
        sp, pft = params(cf_grid); cb = cb_g.copy() if name == 'F' else cb_g * (lai_t / np.maximum(lai_g, 1e-6))[:, None]
        st = eqx.tree_at(lambda s: (s.vegetation.cbiol, s.vegetation.plai), st, (jnp.asarray(cb, jnp.float32), jnp.asarray(cb * specla, jnp.float32)))
        def hold(s_):
            s_ = enforce_prescription(s_, ex_g, specla, fix_cbiol=cb, veg_params=pft)
            return eqx.tree_at(lambda q: (q.vegetation.fu, q.vegetation.fl), s_, (jnp.asarray(fu_g, jnp.float32), jnp.asarray(fl_g, jnp.float32)))
        rd = g3
    else:
        sp, pft = sp_t, pft_t
        st = eqx.tree_at(lambda s: (s.vegetation.cbiol, s.vegetation.cbiow, s.vegetation.cbior, s.vegetation.plai, s.vegetation.exist, s.vegetation.fu, s.vegetation.fl),
                         st, tuple(jnp.asarray(x, jnp.float32) for x in (cb_t, cw_t, cr_t, cb_t * specla[None, :], mask_t, fu_t, fl_t)))
        hold = lambda s_: enforce_prescription(s_, mask_t, specla, fix_cbiol=cb_t, veg_params=pft)
        rd = g3 if name == 'V' else TowerForcingReader(sites, g3)
    ctx = dict(i=i, j=j, lat_rad=lat_rad, sp=sp, pft=pft, consts=consts, reader=rd, f_inund=None); key = jax.random.PRNGKey(0)
    st, key, _ = one_year(st, key, YEARS[0], ctx, 0); st = hold(st)          # extra warm-up of the first year
    gm = {}; laiu = []; lail = []; fus = []; fls = []
    for yr in YEARS:
        st, key, d = one_year(st, key, yr, ctx, 0); st = hold(st); gm[yr] = d['gpp_monthly'] * 1000.0
        fus.append(np.asarray(st.vegetation.fu)); fls.append(np.asarray(st.vegetation.fl))
    print(f'arm {name}: {time.time() - t0:.0f} s', flush=True)
    return np.mean([gm[y].sum(0) for y in SCORED], axis=0), np.mean(fus, 0), np.mean(fls, 0)

res = {}
for nm in a.arms.split(','):
    res[nm] = arm(nm); np.savez(f'{a.out}/arm_{nm}.npz', gpp=res[nm][0], fu=res[nm][1], fl=res[nm][2], sites=np.array(sites))

# G: the global run's own yearly GPP at the cells; T: the tower
G = np.mean([xr.open_dataset(f'{a.run}/out/ibis_yearly_{y}.nc')['aygpptot'].values[i, j] for y in SCORED], axis=0) * 1000.0
T = np.full(ns, np.nan); nmon = np.zeros(ns, int)
for n, s in enumerate(sites):
    o = SF.tower_obs(s)['gpp'][1:]                     # scored years (row 0 is the warm-up year)
    ok = np.isfinite(o); nmon[n] = ok.sum()
    if ok.sum() >= 18: T[n] = np.nanmean(o, axis=0).sum() if np.isfinite(np.nanmean(o, axis=0)).all() else np.nan
df = pd.DataFrame(dict(site=sites, igbp=[META.loc[s, 'IGBP'] for s in sites], lai_src=[src[s] for s in sites], lai_tower=lai_t, lai_global=lai_g,
                       fu_global=fu_g, fl_global=fl_g, cropfrac=cf_grid, G=G, T=T, tower_months=nmon))
for nm in res: df[nm] = res[nm][0]; df[f'fu_{nm}'] = res[nm][1]; df[f'fl_{nm}'] = res[nm][2]
df.to_csv(f'{a.out}/gpp_chain.csv', index=False)
chain = [c for c in ['G', 'F', 'L', 'V', 'S', 'T'] if c in df]
med = lambda x: float(np.nanmedian(x))
print(f'\nannual GPP 2009-2010 at {ns} tower cells, gC m-2 yr-1 (medians over towers); step ratios are medians of the per-tower ratios')
def show(tag, d):
    s = f'{tag:10s} n {len(d):3d}  ' + '  '.join(f'{c} {med(d[c]):5.0f}' for c in chain) + '   |  '
    s += '  '.join(f'{p}/{q} {med(d[p] / d[q]):.2f}' for p, q in zip(chain[:-1], chain[1:])) + f'   G/T {med(d.G / d["T"]):.2f}'
    print(s)
show('all', df)
for g, d in df.groupby('igbp'):
    if len(d) >= 3: show(g, d)
