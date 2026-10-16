"""First estimate of the tree-establishment gate for a parameter set, against the CLM/MODIS tree share (CALIBRATION.md 47).

    python verify/calibration/diagnostics/gate_calib.py <run dir> <year> <params.yaml> [--fu-min F]

The gate (PFTParams.tree_stress_min, tree_stress_ramp; CALIBRATION.md 14) multiplies the upper-canopy pools by the
survival s = clip((annual-mean upper-canopy stress - min) / ramp, 0, 1) each year, so the equilibrium wood of a tree PFT
with turnover tau is the ungated one times (1 - e^(-1/tau)) / (1 - s e^(-1/tau)); cover follows the model's rule
fu = (1 - exp(-wood / fu_k)) / (1 - exp(-woodnorm / fu_k)), clipped to [fu_min, 0.975]. Stress (`aystressu`, written by
run_global since CALIBRATION.md 47) and wood are the run's, for the year given; the vegetation's feedback on soil water
is ignored, so this is a starting value for a global test, not a fit. Natural land only (ESA CCI cropland < 50 %);
the model's natural-share cover fu / (1 - c) against the tree share of PCT_NAT_PFT (PFTs 1-8).
"""
import sys, dataclasses, numpy as np, xarray as xr
sys.path.insert(0, 'verify/calibration'); sys.path.insert(0, 'python-port')
from landmask import true_landfrac
from ibis.state.vegetation import VegetationState
from ibis.io.physics_params import load_physics_params, build_pft_params
run, year, pfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]
pp = build_pft_params(load_physics_params(pfile))
tau = np.asarray(pp.tauwood0)[:8]; fuk, wn, fumin = float(pp.fu_k), float(pp.woodnorm), float(pp.fu_min)
fumin_run = fumin
if '--fu-min' in sys.argv: fumin = float(sys.argv[sys.argv.index('--fu-min') + 1])   # the cover floor (F77 0.25 reads treeless land as 25 % trees)
with xr.open_dataset(f'{run}/out/ibis_yearly_{year}.nc') as y: st = np.asarray(y['aystressu'].values, float)
with xr.open_dataset(f'{run}/out/ibis_monthly_{year}_07.nc') as o: ij = np.argwhere(np.isfinite(o['fu'].values))
r = np.load(f'{run}/restart/ibis_restart_{year}.npz'); tr = str(r['treedef_repr']); vo = tr[:tr.find('VegetationState')].count('*')
vf = [f.name for f in dataclasses.fields(VegetationState)]; cbiow = r[f'leaf_{vo + vf.index("cbiow")}'][:, :8]
LAT = np.arange(89.75, -90, -0.5); I, J = ij[:, 0], ij[:, 1]
lf = true_landfrac()[I, J]; A = np.cos(np.deg2rad(LAT[I])) * lf; stress = st[I, J]
sd = xr.open_dataset('/shared/nas2/Hangkai/ELM Edge Effects/inputdata/surfdata_0.5x0.5_CH_v1_all_forests.nc')
tree = sd['PCT_NAT_PFT'].values[1:9].sum(0) / 100.0
if float(sd['LATIXY'].values[0, 0]) < float(sd['LATIXY'].values[-1, 0]): tree = tree[::-1]
tree = tree[I, J]
c = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz')['cropfrac'][I, J].clip(0, 1)
ok = (lf > 0.05) & (LAT[I] > -60) & np.isfinite(tree) & np.isfinite(stress) & (c < 0.5)
cover = lambda w: np.clip((1 - np.exp(-w / fuk)) / (1 - np.exp(-wn / fuk)), fumin, 0.975)
wnat = 0.001 + np.maximum(cbiow, 0).sum(1) / np.maximum(1 - c, 0.01)        # wood density on the natural share
e = np.exp(-1.0 / tau)
def predict(tmin, ramp):
    s = np.clip((stress - tmin) / ramp, 0, 1)[:, None]
    return cover(0.001 + (np.maximum(cbiow, 0) / np.maximum(1 - c, 0.01)[:, None] * (1 - e) / (1 - s * e)).sum(1))
w = A[ok]; W = lambda x: float((x[ok] * w).sum() / w.sum())
base = np.clip((1 - np.exp(-wnat / fuk)) / (1 - np.exp(-wn / fuk)), fumin_run, 0.975)
def report(tag, f):
    ph = W(((tree < 0.15) & (f > 0.4)).astype(float)); lost = W(((tree > 0.6) & (f < 0.3)).astype(float))
    rmse = np.sqrt(W((f - tree) ** 2)); r_ = np.corrcoef(f[ok], tree[ok])[0, 1]
    by = ' '.join(f'{W(f * ((tree >= a) & (tree < b))) / max(W(((tree >= a) & (tree < b)).astype(float)), 1e-9):.2f}' for a, b in ((0, .1), (.1, .3), (.3, .6), (.6, 1.01)))
    print(f'{tag:22s} cover {W(f):.3f} (obs {W(tree):.3f})  rmse {rmse:.3f} r {r_:.2f}  phantom {100 * ph:4.1f}%  forest lost {100 * lost:4.1f}%   by obs tree class <.1/.1-.3/.3-.6/>.6: {by}')
    return rmse + 2.0 * lost                                                     # losing real forest counts double
print(f'{run} {year}: annual-mean upper-canopy stress on natural land, percentiles 5/25/50/75/95 '
      + ' '.join(f'{v:.2f}' for v in np.percentile(stress[ok], [5, 25, 50, 75, 95])))
for a_, b_ in ((0, .1), (.1, .3), (.3, .6), (.6, 1.01)):
    m = ok & (tree >= a_) & (tree < b_); print(f'  obs tree {a_:.1f}-{b_:.1f}: stress median {np.median(stress[m]):.2f} (p10 {np.percentile(stress[m], 10):.2f}, p90 {np.percentile(stress[m], 90):.2f})')
report(f'no gate (this run, floor {fumin_run:.2f})', base)
report(f'no gate, floor {fumin:.2f}', cover(wnat))
best = None
for ramp in (0.05, 0.1, 0.15, 0.2, 0.3):
    for tmin in np.arange(0.0, 1.0, 0.025):
        f = predict(tmin, ramp)
        lost = W(((tree > 0.6) & (f < 0.3)).astype(float)); score = np.sqrt(W((f - tree) ** 2)) + 2.0 * lost
        if best is None or score < best[0]: best = (score, tmin, ramp)
print(f'best (rmse + 2 x forest lost): tree_stress_min {best[1]:.3f}, tree_stress_ramp {best[2]:.2f}')
for tmin in (best[1] - 0.1, best[1] - 0.05, best[1], best[1] + 0.05, best[1] + 0.1):
    report(f'min {tmin:.3f} ramp {best[2]:.2f}', predict(tmin, best[2]))
