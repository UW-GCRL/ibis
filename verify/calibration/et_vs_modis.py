"""Model ET (`ayaet`) against MODIS ET (ILAMB copy of the MOD16-derived 0.5-degree product, 2000-2013).

    ibis-python verify/calibration/et_vs_modis.py <validation out dir with ibis_yearly_*.nc>

Averages the last 5 yearly files. MODIS ET is kg m-2 s-1 (-> mm yr-1) on a
S->N grid (flipped here). Natural-vegetation cells only (>= 70 %), as for the
tree fraction. Reports, area-weighted, by latitude band, CLM tree-fraction
bin and observed aridity (ET/P): the ET ratio, ET/P, the FLUXCOM GPP ratio,
and WUE = GPP/ET (gC per kg H2O) model vs observed. Stage 2 (ET) groundwork;
the v11d numbers are the baseline (CALIBRATION.md).
"""
import sys, glob, re, numpy as np, xarray as xr
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
from fit_carbon import LATV, MASK, MODIS_BASE, fluxcom_at
V = '/shared/nas2/Hangkai/ibis/verify_outputs'
ET_OBS = '/shared/nas2/Hangkai/ibis/inputs/eval/et/MODIS_et_0.5x0.5.nc'
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{V}/val_v11d/fit/out'
m = np.array(xr.open_dataset(MASK).landfrac.values) > 0
fs = sorted(glob.glob(f'{OUT}/ibis_yearly_*.nc'), key=lambda p: int(re.search(r'(\d+)\.nc$', p).group(1)))[-5:]
acc = {k: 0.0 for k in ('ayaet', 'aygpptot', 'ayprcp')}
for f in fs:
    d = xr.open_dataset(f)
    for k in acc: acc[k] = acc[k] + np.nan_to_num(np.asarray(d[k].values, float)) / len(fs)
e = xr.open_dataset(ET_OBS); et = e['et'].mean('time').values * 86400 * 365.25        # kg m-2 s-1 -> mm yr-1
if float(e.lat[0]) < float(e.lat[-1]): et = et[::-1, :]
sd = xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc")
pp = np.asarray(sd["PCT_NAT_PFT"].values); pv = np.asarray(sd["PCT_NATVEG"].values) / 100.0
if float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]): pp = pp[:, ::-1, :]; pv = pv[::-1, :]
tree_nat = pp[1:9].sum(axis=0) / 100.0
i, j = np.where(m); a = np.cos(np.deg2rad(LATV[i]))
em = acc['ayaet'][i, j]; eo = et[i, j]; g = acc['aygpptot'][i, j]; go = fluxcom_at(i, j); pr = acc['ayprcp'][i, j]; tn = tree_nat[i, j]
ok = np.isfinite(eo) & (eo > 10) & np.isfinite(go) & (go > 0.02) & np.isfinite(em) & (pv[i, j] >= 0.7)
def row(lbl, k):
    k = k & ok
    if k.sum() < 20: return
    A = a[k]; w = lambda x: float(np.sum(A * x[k]) / A.sum())
    ET_m, ET_o, G_m, G_o, P = w(em), w(eo), w(g), w(go), w(pr)
    print(f"{lbl:20s} n={k.sum():5d}  ET mod {ET_m:5.0f} obs {ET_o:5.0f} mm/yr  ratio {ET_m/ET_o:4.2f} | ET/P mod {ET_m/P:4.2f} obs {ET_o/P:4.2f} | GPP ratio {G_m/G_o:4.2f} | WUE mod {1000*G_m/ET_m:4.2f} obs {1000*G_o/ET_o:4.2f} gC/kg  ratio {(G_m/ET_m)/(G_o/ET_o):4.2f}")
print(f"{OUT}\nyears {fs[0][-7:-3]}-{fs[-1][-7:-3]} vs MODIS ET 2000-2013; scored cells (natural veg >= 70 %, ET obs > 10 mm/yr, FLUXCOM > 0.02): {ok.sum()}")
row("ALL", np.ones_like(ok))
print("-- by latitude --")
for lo, hi in ((60, 90), (45, 60), (30, 45), (15, 30), (0, 15), (-15, 0), (-30, -15), (-60, -30)): row(f"lat {lo:+d}..{hi:+d}", (LATV[i] >= lo) & (LATV[i] < hi))
print("-- by CLM tree fraction --")
for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)): row(f"tree {lo:.2f}-{hi:.2f}", (tn >= lo) & (tn < hi))
print("-- by observed aridity ET/P (P = GSWP3 forcing) --")
ar = eo / np.maximum(pr, 1.0)
for lo, hi in ((0, 0.4), (0.4, 0.7), (0.7, 0.9), (0.9, 9)): row(f"ET/P {lo:.1f}-{hi:.1f}", (ar >= lo) & (ar < hi))
