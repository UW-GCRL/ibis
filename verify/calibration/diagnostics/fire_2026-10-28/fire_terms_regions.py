"""Fire terms by GFED region on a run's 2003-2005 restarts, and offline what-ifs against GFED5 Norm 2003-05: the
human-ignition term F3 (HYDE population density from the CLM Li 2018 file), vegetation-density saturation, fine fuel.
Read-only.   usage: python -I fire_terms_regions.py <run dir with restart/ibis_restart_YYYY.npz> [--years 2003 2005]"""
import sys, argparse, numpy as np, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import jax.numpy as jnp, equinox as eqx, xarray as xr, yaml
import evaluate_gfed5 as eg
from ibis.physics.fire import FireParams, legacy_fuel_fraction, fine_fuel_fraction, vegetation_density, canopy_damping, human_ignition, human_suppression
from ibis.ml.fire_calibration import FireBatch, predict_burned_fraction
from ibis.io.pft_types import zones_from_classes, ZONES
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--years', type=int, nargs=2, default=[2003, 2005]); ap.add_argument('--params', default=f'{R}/python-port/configs/params/params_31r.yaml')
a = ap.parse_args(); years = range(a.years[0], a.years[1] + 1)
first = f'{a.run}/restart/ibis_restart_{years[0]}.npz'; lat, lon, li, lo, mlat, mlon = eg._grid(first)
obs = eg.observed(eg.DEFAULT_GFED5, li, lo, mlat, mlon, years)
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
klat = np.asarray(k['lat'], float); kli = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int).clip(0, 359)
klon = np.asarray(k['lon'], float); klo = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int).clip(0, 719)
zone = zones_from_classes(c[kli, klo]); Z = {n: i for i, n in enumerate(ZONES)}
REG = {'Amazon humid': (lon > -82) & (lon < -34) & (lat > -20) & (lat < 13) & (zone == Z['A_humid']),
       'Congo humid': (lon > 8) & (lon < 32) & (lat > -8) & (lat < 6) & (zone == Z['A_humid']),
       'SE Asia humid': (lon > 95) & (lon < 155) & (lat > -11) & (lat < 8) & (zone == Z['A_humid']),
       'S Am savanna': (lon > -82) & (lon < -34) & (lat > -25) & (lat < 13) & (zone == Z['A_savanna']),
       'Africa savanna': (lon > -20) & (lon < 45) & (lat > -25) & (lat < 15) & (zone == Z['A_savanna']),
       'Africa hot dry': (lon > -20) & (lon < 52) & (lat > -35) & (lat < 20) & (zone == Z['B_hot']),
       'Australia sav+dry': (lon > 110) & (lon < 155) & (lat > -30) & np.isin(zone, [Z['A_savanna'], Z['B_hot']]),
       'boreal': zone == Z['D_subarctic'], 'temperate': np.isin(zone, [Z['C_oceanic'], Z['D_temperate'], Z['C_subtrop']]), 'GLOBAL': np.ones_like(zone, bool)}
# population density (ind/km2) for the middle year
pd = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/fire/popdens/clmforc.Li_2018_SSP2_CMIP6_hdm_0.5x0.5_AVHRR_simyr1850-2100_c181205.nc')
yy = np.asarray(pd['year'].values); iy = int(np.argmin(np.abs(yy - years[len(years) // 2])))
hdm = np.asarray(pd['hdm'].values[iy], float); plat = np.asarray(pd['lat'].values, float); plon = np.asarray(pd['lon'].values, float)
pli = np.rint((lat - plat[0]) / (plat[1] - plat[0])).astype(int).clip(0, 359); plo = np.rint((lon - plon[0]) / (plon[1] - plon[0])).astype(int).clip(0, 719)
popd = np.nan_to_num(hdm[pli, plo]); print(f'population density {int(yy[iy])}: mean over land {popd.mean():.1f} ind/km2, median {np.median(popd):.1f}')
y = yaml.safe_load(open(a.params))['fire']
flag = jnp.asarray(np.isin(zone, [Z[z] for z in y.get('closed_canopy_zones', [])]).astype(np.float32))
def P(**kw):
    base = dict(use_canopy_damping=bool(y.get('use_canopy_damping', True)), damp_broadleaf_only=bool(y.get('damp_broadleaf_only', True)), fu_half=float(y['fu_half']), fu_width=float(y['fu_width']),
                k_b=float(y['k_b']), closed_canopy_zones=tuple(y.get('closed_canopy_zones', [])), closed_canopy_fu=float(y.get('closed_canopy_fu', 0.975)))
    base.update(kw); p = FireParams(**base)
    return eqx.tree_at(lambda q: q.closed_canopy_cell, p, flag, is_leaf=lambda x: x is None)
batches, dist = [], []
for yr in years:
    st = eg._load(f'{a.run}/restart/ibis_restart_{yr}.npz'); v, cb = st.vegetation, st.carbon
    batches.append(FireBatch(nfirestep=v.nfirestep_vpd, sdry=v.sdry_vpd, nytimes=v.nytimes, totlit=cb.totlit, totalit=cb.totalit, cbiol=v.cbiol, plai=v.plai, popd=jnp.asarray(popd, jnp.float32), fu=v.fu))
    dist.append(np.asarray(v.disturbf, float))
aw = np.cos(np.deg2rad(lat)); ok = np.isfinite(obs) & np.all([np.isfinite(d) for d in dist], 0)
def wm(x, m=None):
    m = ok if m is None else (ok & m); return float((np.asarray(x)[m] * aw[m]).sum() / aw[m].sum())
def predict(p): return np.mean([np.asarray(predict_burned_fraction(p, b), float) for b in batches], 0)
p0 = P(); pred0 = predict(p0); d0 = np.mean(dist, 0)
print(f'predictor vs the states\' own disturbf: ratio {wm(pred0) / wm(d0):.3f}, r {np.corrcoef(pred0[ok], d0[ok])[0, 1]:.3f}  (params {a.params.split("/")[-1]})')
# terms (means over the years)
def terms(b):
    fseason = np.asarray(b.nfirestep / jnp.maximum(b.nytimes, 1.0)); dry = np.asarray(b.sdry / jnp.maximum(b.nfirestep, 1.0))
    fuel = np.asarray(legacy_fuel_fraction(b.totlit)); vd = np.asarray(vegetation_density(b.plai, p0)); cd = np.asarray(canopy_damping(b.fu, p0, b.plai, closed=flag))
    return fseason, dry, fuel, vd, cd
T = [terms(b) for b in batches]; fseason, dry, fuel, vd, cd = [np.mean([t[i] for t in T], 0) for i in range(5)]
lai = np.mean([np.asarray(b.plai.sum(1)) for b in batches], 0); totlit = np.mean([np.asarray(b.totlit) for b in batches], 0); fuc = np.mean([np.asarray(b.fu) for b in batches], 0)
print(f"\n--- terms by region (area means over cells with GFED5): burned % GFED5 | model | fseason dry fuel vd damp | LAI litter fu popd")
print(f"{'region':20s} {'GFED5':>6s} {'model':>6s} {'ratio':>6s} | {'fseas':>5s} {'dry':>5s} {'fuel':>5s} {'vd':>5s} {'damp':>5s} | {'LAI':>5s} {'lit':>5s} {'fu':>5s} {'popd':>6s}")
for r, m in REG.items():
    print(f"{r:20s} {100 * wm(obs, m):6.2f} {100 * wm(pred0, m):6.2f} {wm(pred0, m) / max(wm(obs, m), 1e-9):6.2f} | {wm(fseason, m):5.2f} {wm(dry, m):5.2f} {wm(fuel, m):5.2f} {wm(vd, m):5.2f} {wm(cd, m):5.2f} | {wm(lai, m):5.2f} {wm(totlit, m):5.2f} {wm(fuc, m):5.2f} {wm(popd, m):6.1f}")
# African savanna by population-density class
m0 = REG['Africa savanna']
print('\n--- African savanna by population density (ind/km2): area %, GFED5, model, fseason, vd')
for lo_, hi_ in ((0, 5), (5, 20), (20, 50), (50, 100), (100, 300), (300, 1e9)):
    m = m0 & (popd >= lo_) & (popd < hi_)
    if (ok & m).sum() < 10: continue
    print(f'  {lo_:5.0f}-{hi_:5.0f}: {100 * aw[ok & m].sum() / aw[ok & m0].sum():5.1f} % | {100 * wm(obs, m):5.1f} {100 * wm(pred0, m):5.1f} | {wm(fseason, m):.2f} {wm(vd, m):.2f}')
print('--- S American savanna by population density')
m0 = REG['S Am savanna']
for lo_, hi_ in ((0, 5), (5, 20), (20, 50), (50, 100), (100, 300), (300, 1e9)):
    m = m0 & (popd >= lo_) & (popd < hi_)
    if (ok & m).sum() < 10: continue
    print(f'  {lo_:5.0f}-{hi_:5.0f}: {100 * aw[ok & m].sum() / aw[ok & m0].sum():5.1f} % | {100 * wm(obs, m):5.1f} {100 * wm(pred0, m):5.1f} | {wm(fseason, m):.2f} {wm(vd, m):.2f}')
# what-ifs
rows = [('current', pred0)]
for aig in (0.005, 0.01, 0.02, 0.05):
    for imax in (2.0, 3.0, 5.0):
        rows.append((f'human a_ig {aig} ign_max {imax}', predict(P(use_human=True, a_ig=aig, ign_max=imax))))
for lr in (2.0, 3.0):
    rows.append((f'lai_ref_vd {lr}', predict(P(lai_ref_vd=lr))))
for wg in (1.0, 2.0):
    rows.append((f'fine fuel w_grass {wg}', predict(P(use_fine_fuel=True, w_leaf_grass=wg))))
print(f"\n--- what-ifs: model / GFED5 by region (k_b unchanged)\n{'configuration':32s} " + ' '.join(f'{r[:10]:>10s}' for r in REG))
for name, pr in rows:
    print(f'{name:32s} ' + ' '.join(f'{wm(pr, m) / max(wm(obs, m), 1e-9):10.2f}' for m in REG.values()))
