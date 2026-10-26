"""Old gate (unbooked kill) vs new gate (kill to litter) at the SAME cold-start spin-up years: litterfall written in the
year's restart, the live-pool residual NPP - fall - cdisturb - dLive, and the gate kill estimate post x (1-ts)/ts. Read-only.
    python gate_matched.py <old run dir> <new run dir> <year> [<year> ...]"""
import sys, numpy as np, xarray as xr, yaml, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg
from landmask import true_landfrac
old, new = sys.argv[1], sys.argv[2]; years = [int(a) for a in sys.argv[3:]]
lat, lon, *_ = eg._grid(f'{old}/restart/ibis_restart_{years[0]}.npz')
lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj]); P = lambda x: float(np.nansum(np.asarray(x, float) * area)) / 1e12
pp = yaml.safe_load(open(yaml.safe_load(open(f'{old}/spinup/cfg.yaml' if False else f'{old}/cfg.yaml'))['params_file']))['pft']; smin, ramp = float(pp['tree_stress_min']), float(pp['tree_stress_ramp'])
def rec(run, y):
    st = eg._load(f'{run}/restart/ibis_restart_{y}.npz'); v, c = st.vegetation, st.carbon
    live = np.asarray(v.cbiol).sum(1) + np.asarray(v.cbiow).sum(1) + np.asarray(v.cbior).sum(1)
    tree = np.asarray(v.cbiol)[:, :8].sum(1) + np.asarray(v.cbiow)[:, :8].sum(1) + np.asarray(v.cbior)[:, :8].sum(1)
    d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); g = lambda n: np.asarray(d[n].values, float)[gi, gj]
    ts = np.clip((g('aystressu') - smin) / ramp, 0, 1); kill = np.where(ts > 0.05, tree * (1 - ts) / np.maximum(ts, 1e-6), 0.0)
    return dict(live=live, fall=np.asarray(c.falll) + np.asarray(c.fallw) + np.asarray(c.fallr), cd=np.asarray(c.cdisturb), npp=g('aynpptot'), kill=kill, soil=np.asarray(c.totlit) + np.asarray(c.csoipas) + np.asarray(c.csoislon) + np.asarray(c.csoislop) + np.asarray(c.totcmic))
print(f"{'year':5s} {'run':4s} {'fall':>6s} {'cdist':>6s} {'NPP':>6s} {'dLive':>7s} {'live resid':>10s} {'kill est':>8s} {'dSoil':>7s}   PgC/yr")
for y in years:
    for name, run in (('old', old), ('new', new)):
        a, b = rec(run, y - 1), rec(run, y)
        dl = P(b['live'] - a['live']); res = dl - (P(b['npp']) - P(b['fall']) - P(b['cd']))
        print(f"{y:5d} {name:4s} {P(b['fall']):6.2f} {P(b['cd']):6.2f} {P(b['npp']):6.2f} {dl:+7.2f} {res:+10.2f} {P(b['kill']):8.2f} {P(b['soil'] - a['soil']):+7.2f}")
