"""Second pass: is every carbon loss in cdisturb? dC vs NPP - Rh - cdisturb (cdisturb from the year's restart), plus the
background-disturbance part (0.005 x live biomass of the previous state) and the live-biomass budget alone."""
import sys, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg
from landmask import true_landfrac
tag = sys.argv[1]; V = '/shared/nas2/Hangkai/ibis/verify_outputs'; run = f'{V}/final_{tag}/hindcast'
lat, lon, li, lo, mlat, mlon = eg._grid(f'{run}/restart/ibis_restart_2005.npz')
lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj])
P = lambda x: float(np.nansum(np.asarray(x, float) * area)) / 1e12
S = {}
for y in range(2005, 2011):
    st = eg._load(f'{run}/restart/ibis_restart_{y}.npz'); v, c = st.vegetation, st.carbon
    live = np.asarray(v.cbiol).sum(1) + np.asarray(v.cbiow).sum(1) + np.asarray(v.cbior).sum(1)
    soil = np.asarray(c.totlit) + np.asarray(c.csoipas) + np.asarray(c.csoislon) + np.asarray(c.csoislop) + np.asarray(c.totcmic)
    S[y] = dict(live=live, soil=soil, cdist=np.asarray(c.cdisturb), fall=np.asarray(c.falll) + np.asarray(c.fallw) + np.asarray(c.fallr))
print(f'{tag}  PgC/yr        dC_total   NPP-Rh  cdisturb  dC-(NPP-Rh-cdist)   live_C  bg_disturb(0.005*live)  litterfall')
for y in range(2006, 2011):
    d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); g = lambda n: np.asarray(d[n].values, float)[gi, gj]
    dC = P(S[y]['live'] + S[y]['soil'] - S[y - 1]['live'] - S[y - 1]['soil']); nr = P(g('aynpptot') - g('ayco2mic')); cd = P(S[y]['cdist'])
    print(f'  {y}         {dC:+7.2f}  {nr:+7.2f}  {cd:7.2f}       {dC - (nr - cd):+7.2f}          {P(S[y]["live"]):6.0f}   {0.005 * P(S[y - 1]["live"]):6.2f}              {P(S[y]["fall"]):6.1f}')
print(f'\n{tag} compartments, PgC/yr: live dC vs NPP - fall(y) - cdisturb(y)  |  soil+litter dC vs fall(y-1) - Rh(y)')
for y in range(2006, 2011):
    d = xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc'); g = lambda n: np.asarray(d[n].values, float)[gi, gj]
    dL = P(S[y]['live'] - S[y - 1]['live']); dS = P(S[y]['soil'] - S[y - 1]['soil'])
    npp, rh = P(g('aynpptot')), P(g('ayco2mic')); fy, fp, cd = P(S[y]['fall']), P(S[y - 1]['fall']), P(S[y]['cdist'])
    print(f'  {y}  live {dL:+6.2f} vs {npp - fy - cd:+6.2f} (resid {dL - (npp - fy - cd):+6.2f})   soil {dS:+6.2f} vs {fp - rh:+6.2f} (resid {dS - (fp - rh):+6.2f})   NPP {npp:5.1f} Rh {rh:5.1f}')
