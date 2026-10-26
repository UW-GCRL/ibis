"""Where and why the humid Amazon burns in the final run: per-cell burned fraction vs GFED5, the fire terms of the hot cells
(fire season, dryness, fuel, vegetation density, cover, broadleaf share, damping), their wood against the map, and their
history through the spin-up (fu and disturbf by decade). Read-only."""
import sys, glob, numpy as np, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire']
import jax.numpy as jnp, evaluate_gfed5 as eg
from ibis.io.pft_types import zones_from_classes, ZONES
from ibis.physics.fire import FireParams, canopy_damping, vegetation_density, broadleaf_share, legacy_fuel_fraction
from scipy.stats import spearmanr
tag = sys.argv[1]; V = '/shared/nas2/Hangkai/ibis/verify_outputs'
if tag.startswith('check10_'):                      # a 10-year check run: restart/ and out/ at the top, params named after it
    run = f'{V}/{tag}'; p = f'{run}/restart/ibis_restart_2010.npz'; ptag = tag.split('_')[1]
else:
    run = f'{V}/final_{tag}'; p = f'{run}/hindcast/restart/ibis_restart_2010.npz'; ptag = tag[1:]
lat, lon, li, lo, mlat, mlon = eg._grid(p)
obs = eg.observed(eg.DEFAULT_GFED5, li, lo, mlat, mlon, range(2001, 2006))
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
klat = np.asarray(k['lat'], float); kli = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int).clip(0, 359)
klon = np.asarray(k['lon'], float); klo = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int).clip(0, 719)
zone = zones_from_classes(c[kli, klo]); hum = (lon > -82) & (lon < -34) & (lat > -20) & (lat < 13) & (zone == ZONES.index('A_humid'))
st = eg._load(p); v, cb = st.vegetation, st.carbon
d = np.asarray(v.disturbf, float); fu = np.asarray(v.fu, float); plai = np.asarray(v.plai, float)
nyt = float(np.asarray(v.nytimes)); fs = np.asarray(v.nfirestep_vpd, float) / nyt; dry = np.asarray(v.sdry_vpd, float) / np.maximum(np.asarray(v.nfirestep_vpd, float), 1)
import yaml; fp = yaml.safe_load(open(f'{R}/python-port/configs/params/params_{ptag}.yaml'))['fire']; P = FireParams(**{kk: (tuple(vv) if kk == 'closed_canopy_zones' else vv) for kk, vv in fp.items()})
vd = np.asarray(vegetation_density(jnp.asarray(plai), P)); bl = np.asarray(broadleaf_share(jnp.asarray(plai))); closed = jnp.asarray((zone == ZONES.index('A_humid')).astype(np.float32)) if P.closed_canopy_zones else None
cd = np.asarray(canopy_damping(jnp.asarray(fu), P, jnp.asarray(plai), closed=closed))
fuel = np.asarray(legacy_fuel_fraction(jnp.asarray(np.asarray(cb.totlit, float))))
wood = np.asarray(v.cbiow, float)[:, :8].sum(1)
ref = np.load('/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad/meeting/glob_refs.npz'); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int); wmap = ref['wood'][gi, gj]
aw = np.cos(np.deg2rad(lat)); m = hum & np.isfinite(d) & np.isfinite(obs)
wm = lambda x, mm: float((x[mm] * aw[mm]).sum() / aw[mm].sum())
print(f'{tag} humid Amazon: {int(m.sum())} cells; burned model {100 * wm(d, m):.2f} % vs GFED5 {100 * wm(obs, m):.2f} %; cell rank r {spearmanr(d[m], obs[m]).statistic:.2f}')
for thr in (0.05, 0.10, 0.15):
    hot = m & (d > thr); print(f'  cells burning > {100 * thr:.0f} %/yr: {int(hot.sum())} ({100 * aw[hot].sum() / aw[m].sum():.1f} % of area), they carry {100 * (d * aw)[hot].sum() / (d * aw)[m].sum():.0f} % of the zone fire; GFED5 there {100 * wm(obs, hot):.2f} %; wood {wm(wood, hot):.1f} vs map {wm(wmap, hot):.1f} kgC/m2')
quiet = m & (d < 0.02); print(f'  quiet cells (< 2 %): {int(quiet.sum())}; wood {wm(wood, quiet):.1f} vs map {wm(wmap, quiet):.1f}')
hot = m & (d > 0.10)
print(f"\n  fire terms, area means         {'hot':>7s} {'quiet':>7s}   (burn = 1 - exp(-k_b fseason dry fuel vd damp), k_b {P.k_b})")
for nm, x in (('fire season', fs), ('dryness', dry), ('fuel', fuel), ('veg density', vd), ('fu (cover)', fu), ('broadleaf share', bl), ('damping', cd), ('upper LAI', plai[:, :8].sum(1)), ('lower LAI', plai[:, 8:].sum(1)), ('wood kgC/m2', wood), ('map wood', wmap)):
    print(f'  {nm:28s} {wm(x, hot):7.2f} {wm(x, quiet):7.2f}')
# history through the spin-up: fu and disturbf in the hot cells
print('\n  hot cells through the run (area means): year  fu  disturbf  damping  upperLAI')
for y in ((2001, 2003, 2005, 2007, 2010) if tag.startswith('check10_') else (1702, 1710, 1720, 1740, 1760, 1800, 1850, 1900, 1950, 2000, 2010)):
    f = (f'{run}/restart/ibis_restart_{y}.npz' if tag.startswith('check10_') else (f'{run}/spinup/restart/ibis_restart_{y}.npz' if y <= 1900 else (f'{run}/transient/restart/ibis_restart_{y}.npz' if y <= 2000 else f'{run}/hindcast/restart/ibis_restart_{y}.npz')))
    if not glob.glob(f): continue
    s = eg._load(f); vv = s.vegetation; fu_y = np.asarray(vv.fu, float); d_y = np.asarray(vv.disturbf, float); pl = np.asarray(vv.plai, float)
    cd_y = np.asarray(canopy_damping(jnp.asarray(fu_y), P, jnp.asarray(pl), closed=closed))
    print(f'  {y}  {wm(fu_y, hot):.2f}  {100 * wm(d_y, hot):6.1f} %  {wm(cd_y, hot):.2f}  {wm(pl[:, :8].sum(1), hot):.1f}      | quiet: fu {wm(fu_y, quiet):.2f} burn {100 * wm(d_y, quiet):.1f} %')
hi = np.where(hot)[0]
if len(hi): print('\n  hot-cell centroid', round(float(np.mean(lat[hi])), 1), round(float(np.mean(lon[hi])), 1), '; lat range', round(float(lat[hi].min()), 1), round(float(lat[hi].max()), 1), 'lon', round(float(lon[hi].min()), 1), round(float(lon[hi].max()), 1))
