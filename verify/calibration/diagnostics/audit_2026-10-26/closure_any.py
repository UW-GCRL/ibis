"""Carbon budget closure for any run directory with yearly restarts + yearly nc (spin-up or hindcast). Read-only.
    python closure_any.py <run dir> [max pairs]"""
import sys, os, glob, re, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg
from landmask import true_landfrac
run = sys.argv[1]; nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
yrs = sorted(int(re.search(r'(\d{4})', os.path.basename(f)).group(1)) for f in glob.glob(f'{run}/restart/ibis_restart_*.npz'))
pairs = [(a, b) for a, b in zip(yrs, yrs[1:]) if b == a + 1 and os.path.exists(f'{run}/out/ibis_yearly_{b}.nc')][-nmax:]
if not pairs: print(run, 'no consecutive restart pairs with yearly output; restarts:', yrs[-5:]); sys.exit()
lat, lon, *_ = eg._grid(f'{run}/restart/ibis_restart_{pairs[0][0]}.npz')
lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj]); P = lambda x: float(np.nansum(np.asarray(x, float) * area)) / 1e12
def pools(y):
    st = eg._load(f'{run}/restart/ibis_restart_{y}.npz'); v, c = st.vegetation, st.carbon
    live = np.asarray(v.cbiol).sum(1) + np.asarray(v.cbiow).sum(1) + np.asarray(v.cbior).sum(1)
    soil = np.asarray(c.totlit) + np.asarray(c.csoipas) + np.asarray(c.csoislon) + np.asarray(c.csoislop) + np.asarray(c.totcmic)
    return live, soil, np.asarray(c.cdisturb), np.asarray(c.falll) + np.asarray(c.fallw) + np.asarray(c.fallr)
print(f'{run.split("/")[-2]}/{run.split("/")[-1]}: pairs {pairs}')
print(' year   dC_total  NPP-Rh  cdisturb  residual | live dC vs NPP-fall-cdist (resid) | soil dC vs fall(y-1)-Rh (resid)   PgC/yr')
prev = pools(pairs[0][0])
for a, b in pairs:
    cur = pools(b); d = xr.open_dataset(f'{run}/out/ibis_yearly_{b}.nc'); g = lambda n: np.asarray(d[n].values, float)[gi, gj]
    npp, rh = P(g('aynpptot')), P(g('ayco2mic')); dL, dS = P(cur[0] - prev[0]), P(cur[1] - prev[1]); cd, fy, fp = P(cur[2]), P(cur[3]), P(prev[3])
    if 'ayrh_ch4c' in d: rh = rh + P(g('ayrh_ch4c'))     # methane_carbon_partition: the decomposed carbon is Rh as CO2 + the CH4 carbon (METHANE.md 12)
    print(f' {b}   {dL + dS:+7.2f} {npp - rh:+7.2f}  {cd:6.2f}  {dL + dS - (npp - rh - cd):+7.2f} |  {dL:+6.2f} vs {npp - fy - cd:+6.2f} ({dL - (npp - fy - cd):+6.2f})      |  {dS:+6.2f} vs {fp - rh:+6.2f} ({dS - (fp - rh):+6.2f})')
    prev = cur
