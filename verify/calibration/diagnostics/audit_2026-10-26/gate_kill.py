"""Tree carbon removed each year by the water-stress survival gate (dynaveg: tree pools x ts, removed part not booked):
ts from the yearly aystressu and tree_stress_min/ramp, killed = post x (1 - ts) / ts from the year-end restart. Read-only."""
import sys, numpy as np, xarray as xr, yaml, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg
from landmask import true_landfrac
tag = sys.argv[1]; V = '/shared/nas2/Hangkai/ibis/verify_outputs'; run = f'{V}/final_{tag}/hindcast'
pp = yaml.safe_load(open(yaml.safe_load(open(f'{run}/cfg.yaml'))['params_file']))['pft']; smin, ramp = float(pp['tree_stress_min']), float(pp['tree_stress_ramp'])
for y in (2008, 2010):
    lat, lon, li, lo, mlat, mlon = eg._grid(f'{run}/restart/ibis_restart_{y}.npz')
    lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
    area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj]); P = lambda x: float(np.nansum(np.asarray(x, float) * area)) / 1e12
    d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); u = np.asarray(d['aystressu'].values, float)[gi, gj]
    ts = np.clip((u - smin) / ramp, 0, 1)
    st = eg._load(f'{run}/restart/ibis_restart_{y}.npz'); v = st.vegetation
    tree = np.asarray(v.cbiol)[:, :8].sum(1) + np.asarray(v.cbiow)[:, :8].sum(1) + np.asarray(v.cbior)[:, :8].sum(1)
    k = np.where(ts > 0.05, tree * (1 - ts) / np.maximum(ts, 1e-6), 0.0)
    print(f'{tag} {y}: gate {smin}/{ramp}; cells with ts<1 {100 * np.nanmean(ts < 0.999):.1f} %, ts=0 {100 * np.nanmean(ts <= 0):.1f} %; '
          f'tree C in ts<1 cells {P(np.where(ts < 0.999, tree, 0)):.1f} PgC; killed (ts>0.05 cells) {P(k):.2f} PgC/yr')
