"""F77's annual pool update books litterfall c/tau but decays the pool by c(1-e^{-1/tau}), and adds a tau N (1-e^{-1/tau})
of growth where NEE assumes a N: exact only at equilibrium c = a tau N. Estimate of that residual on a year's state, with each
position's NPP taken as the cell NPP x its leaf-area share (the big-leaf GPP split). Compare with the measured live residual."""
import sys, numpy as np, xarray as xr, yaml, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg
from landmask import true_landfrac
from ibis.io.physics_params import PFT_NAMES
run, y = sys.argv[1], int(sys.argv[2])
lat, lon, *_ = eg._grid(f'{run}/restart/ibis_restart_{y}.npz')
lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj]); P = lambda x: float(np.nansum(np.asarray(x, float) * area)) / 1e12
pp = yaml.safe_load(open(yaml.safe_load(open(f'{run}/cfg.yaml'))['params_file']))['pft']
par = lambda k, d: np.array([float((pp.get(n, {}) or {}).get(k, d[i])) for i, n in enumerate(PFT_NAMES)])
D = dict(aleaf=[0.3]*12, aroot=[0.2]*12, awood=[0.5]*12, tauleaf=[1.0]*12, tauroot=[1.0]*12, tauwood0=[25.0]*12)
aleaf, aroot, awood = par('aleaf', D['aleaf']), par('aroot', D['aroot']), par('awood', D['awood'])
tl, tr, tw = par('tauleaf', D['tauleaf']), par('tauroot', D['tauroot']), par('tauwood0', D['tauwood0'])
sta = eg._load(f'{run}/restart/ibis_restart_{y - 1}.npz'); va = sta.vegetation
d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); npp = np.asarray(d['aynpptot'].values, float)[gi, gj]
plai = np.asarray(va.plai, float); sh = plai / np.maximum(plai.sum(1, keepdims=True), 1e-9); N = npp[:, None] * sh     # crude per-position NPP
def resid(c, a, tau):
    e = np.exp(-1.0 / tau)[None, :]; return c * (1.0 / tau[None, :] - (1 - e)) + a[None, :] * N * (tau[None, :] * (1 - e) - 1)
rl, rr, rw = resid(np.asarray(va.cbiol), aleaf, tl), resid(np.asarray(va.cbior), aroot, tr), resid(np.asarray(va.cbiow), awood, tw)
print(f'{run.split("/")[-2]} {y}: discretization residual estimate, PgC/yr: leaf {P(rl.sum(1)):+.2f}  root {P(rr.sum(1)):+.2f}  wood {P(rw.sum(1)):+.2f}  total {P((rl + rr + rw).sum(1)):+.2f}')
print('  tauleaf', np.round(tl, 2), '\n  tauroot', np.round(tr, 2))
