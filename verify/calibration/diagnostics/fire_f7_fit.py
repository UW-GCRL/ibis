"""F7 closed-canopy fire damping: offline fit against GFED5 Norm (CALIBRATION.md 59).

    python verify/calibration/diagnostics/fire_f7_fit.py <run dir with restart/ibis_restart_YYYY.npz> --years 2003 2005

Each year-end restart of a run made WITH the annual fire-counter reset holds that year's own counters (nfirestep_vpd,
sdry_vpd, nytimes) and the vegetation the fire term reads (plai, fu, litter). For every (fu_half, fu_width) on a grid,
k_b is set by bisection so the area-weighted burned fraction equals GFED5's, and the cell skill is reported: NMSE (the
docs/FIRE.md metric), rank correlation all / burning cells, and model/GFED5 by region (humid forests and savannas, by
Köppen zone). The undamped prediction is checked against the state's own disturbf first, so the predictor is the
model's fire term.
"""
import sys, os, argparse, numpy as np, warnings
warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'
sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import jax.numpy as jnp
from scipy.stats import spearmanr
import evaluate_gfed5 as eg
from ibis.physics.fire import FireParams
from ibis.ml.fire_calibration import FireBatch, predict_burned_fraction
from ibis.io.pft_types import zones_from_classes, ZONES

ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--years', type=int, nargs=2, default=[2003, 2005])
ap.add_argument('--gfed-years', type=int, nargs=2, default=None); ap.add_argument('--out', default=''); ap.add_argument('--kb-free', action='store_true'); ap.add_argument('--all-canopies', action='store_true')
a = ap.parse_args()
years = range(a.years[0], a.years[1] + 1); gy = range(*(a.gfed_years or a.years)) if a.gfed_years else years
if a.gfed_years: gy = range(a.gfed_years[0], a.gfed_years[1] + 1)
first = f'{a.run}/restart/ibis_restart_{years[0]}.npz'
lat, lon, li, lo, mlat, mlon = eg._grid(first)
obs = eg.observed(eg.DEFAULT_GFED5, li, lo, mlat, mlon, gy)
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
klat = np.asarray(k['lat'], float); kli = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int).clip(0, 359)
klon = np.asarray(k['lon'], float); klo = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int).clip(0, 719)
zone = zones_from_classes(c[kli, klo])
Z = {n: i for i, n in enumerate(ZONES)}
REG = {'Amazon humid': (lon > -82) & (lon < -34) & (lat > -20) & (lat < 13) & (zone == Z['A_humid']),
       'Congo humid': (lon > 8) & (lon < 32) & (lat > -8) & (lat < 6) & (zone == Z['A_humid']),
       'SE Asia humid': (lon > 95) & (lon < 155) & (lat > -11) & (lat < 8) & (zone == Z['A_humid']),
       'S Am savanna': (lon > -82) & (lon < -34) & (lat > -25) & (lat < 13) & (zone == Z['A_savanna']),
       'Africa savanna': (lon > -20) & (lon < 45) & (lat > -25) & (lat < 15) & (zone == Z['A_savanna']),
       'Australia savanna': (lon > 110) & (lon < 155) & (lat > -25) & (zone == Z['A_savanna']),
       'boreal': zone == Z['D_subarctic'], 'temperate': np.isin(zone, [Z['C_oceanic'], Z['D_temperate'], Z['C_subtrop']])}
batches, dist = [], []
for y in years:
    st = eg._load(f'{a.run}/restart/ibis_restart_{y}.npz'); v, cb = st.vegetation, st.carbon
    batches.append(FireBatch(nfirestep=v.nfirestep_vpd, sdry=v.sdry_vpd, nytimes=v.nytimes, totlit=cb.totlit,
                             totalit=cb.totalit, cbiol=v.cbiol, plai=v.plai, popd=jnp.zeros_like(cb.totlit), fu=v.fu))
    dist.append(np.asarray(v.disturbf, float))
    print(f'{y}: nytimes {float(v.nytimes):.0f} (one year = 8760 or 8784), fseason mean {float(jnp.nanmean(v.nfirestep_vpd / v.nytimes)):.3f}')
aw = np.cos(np.deg2rad(lat)); ok = np.isfinite(obs) & np.all([np.isfinite(d) for d in dist], 0)
def predict(p): return np.mean([np.asarray(predict_burned_fraction(p, b), float) for b in batches], 0)
def wm(x, m=None):
    m = ok if m is None else (ok & m); return float((x[m] * aw[m]).sum() / aw[m].sum())
#: Zone-level objective: RMS of log(model/GFED5) of the zone means over the vegetated zones. Cell NMSE is dominated by the
#: African savanna cells (40 % a year in GFED5, mostly human-set grass fire the scheme has no term for) and prefers no
#: damping; the biome-scale distribution is what the forest problem is about.
ZFIT = [z for z in ZONES if z not in ('E_ice', 'D_extreme', 'none')]
def skill(pred):
    pr, ob = pred[ok], obs[ok]; nmse = float(((pr - ob) ** 2 * aw[ok]).sum() / (((ob - wm(obs)) ** 2) * aw[ok]).sum())
    b = ob > 0
    zr = [np.log(max(wm(pred, zone == Z[z]), 1e-6) / max(wm(obs, zone == Z[z]), 1e-6)) for z in ZFIT]
    return dict(total=wm(pred) / wm(obs), zlog=float(np.sqrt(np.mean(np.square(zr)))), nmse=nmse,
                rank=spearmanr(pr, ob).statistic, rank_b=spearmanr(pr[b], ob[b]).statistic,
                **{r: wm(pred, m) / max(wm(obs, m), 1e-9) for r, m in REG.items()})
p0 = FireParams()
pred0 = predict(p0); d0 = np.mean(dist, 0)
print(f'predictor check (undamped vs the states\' own disturbf): ratio {wm(pred0) / wm(d0):.3f}, r {np.corrcoef(pred0[ok], d0[ok])[0, 1]:.3f}')
print(f'GFED5 Norm {gy[0]}-{gy[-1]}: {100 * wm(obs):.2f} % per year; regions ' + ', '.join(f'{r} {100 * wm(obs, m):.1f}' for r, m in REG.items()))
def fit_kb(p):
    lo_, hi_ = 0.01, 20.0
    for _ in range(40):
        mid = (lo_ * hi_) ** 0.5; t = wm(predict(FireParams(**{**p, 'k_b': mid})))
        lo_, hi_ = (mid, hi_) if t < wm(obs) else (lo_, mid)
    return (lo_ * hi_) ** 0.5
def fit_kb_zone(p):
    """k_b minimising the zone log-RMS (golden section on log k_b): the global total is then an outcome, not a target --
    forcing it to GFED5 makes the scheme overburn every other zone to make up for the savanna fire it cannot produce."""
    import math
    a_, b_ = math.log(0.05), math.log(5.0); g = (5 ** 0.5 - 1) / 2
    f = lambda lk: skill(predict(FireParams(**{**p, 'k_b': math.exp(lk)})))['zlog']
    c_, d_ = b_ - g * (b_ - a_), a_ + g * (b_ - a_); fc, fd = f(c_), f(d_)
    for _ in range(18):
        if fc < fd: b_, d_, fd = d_, c_, fc; c_ = b_ - g * (b_ - a_); fc = f(c_)
        else: a_, c_, fc = c_, d_, fd; d_ = a_ + g * (b_ - a_); fd = f(d_)
    return math.exp((a_ + b_) / 2)
if '--kb-free' in sys.argv:
    fit_kb = fit_kb_zone; print('k_b fitted to the zone log-RMS (--kb-free)')
rows = []
s = skill(pred0); rows.append(('current (k_b 0.5, no damping)', 0.5, None, None, s))
kb = fit_kb({}); s = skill(predict(FireParams(k_b=kb))); rows.append((f'no damping, k_b to total', kb, None, None, s))
for fh in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85):
    for fw in (0.03, 0.06, 0.10):
        pp = dict(use_canopy_damping=True, fu_half=fh, fu_width=fw, damp_broadleaf_only=not a.all_canopies); kb = fit_kb(pp)
        rows.append((f'damping fu_half {fh:.2f} width {fw:.2f}', kb, fh, fw, skill(predict(FireParams(**pp, k_b=kb)))))
cols = ['total', 'zlog', 'nmse', 'rank', 'rank_b'] + list(REG)
print(f"\n{'configuration':38s} {'k_b':>6s} " + ' '.join(f'{c[:9]:>9s}' for c in cols))
for name, kb, fh, fw, s in rows:
    print(f'{name:38s} {kb:6.2f} ' + ' '.join(f'{s[c]:9.2f}' for c in cols))
best = min(rows[2:], key=lambda r: r[4]['nmse'])
print(f'\nlowest NMSE: {best[0]}  k_b {best[1]:.3f}')
best = min(rows[1:], key=lambda r: r[4]['zlog'])
print(f'lowest zone log-RMS: {best[0]}  k_b {best[1]:.3f}  (zone log-RMS {best[4]["zlog"]:.2f}, rank {best[4]["rank"]:.2f})')
print('\nzone means, % per year: zone, GFED5, current, best-zone-fit')
pb = predict(FireParams(use_canopy_damping=best[2] is not None, damp_broadleaf_only=not a.all_canopies, **({} if best[2] is None else dict(fu_half=best[2], fu_width=best[3])), k_b=best[1]))
for z in ZFIT:
    m = zone == Z[z]; print(f'  {z:12s} {100 * wm(obs, m):6.2f} {100 * wm(pred0, m):6.2f} {100 * wm(pb, m):6.2f}')
if a.out:
    np.savez_compressed(a.out, lat=lat, lon=lon, obs=obs, pred0=pred0, zone=zone,
                        rows=np.array([(r[0], r[1], r[2] or np.nan, r[3] or np.nan, *[r[4][c] for c in cols]) for r in rows], dtype=object), cols=np.array(cols))
