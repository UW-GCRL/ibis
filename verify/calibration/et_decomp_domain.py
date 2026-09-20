"""ET components over the domain from yearly output written with physics.et_components (CALIBRATION.md 15).

    ibis-python verify/calibration/et_decomp_domain.py <out dir with ibis_yearly_*.nc carrying aytrans/ayinterc/aysoilev/aysnowsub> [nyear=5]

Averages the last N yearly files. Natural-vegetation cells (>= 70 %) with all
five ET products; area = true land fraction. By CLM tree class and latitude
band: model ET vs the ensemble mean, the four components (mm/yr and shares),
and transpiration per unit GPP (kg H2O per gC) as the stomatal check.
"""
import sys, glob, re, numpy as np, xarray as xr, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
from fit_carbon import LATV, MASK, MODIS_BASE, fluxcom_at, et_at
from landmask import true_landfrac
OUT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
fs = sorted(glob.glob(f'{OUT}/ibis_yearly_*.nc'), key=lambda p: int(re.search(r'(\d+)\.nc$', p).group(1)))[-N:]
m = np.array(xr.open_dataset(MASK).landfrac.values) > 0; i, j = np.where(m)
lf = true_landfrac()[i, j]; a = np.cos(np.deg2rad(LATV[i])) * lf
V = ('ayaet', 'aytrans', 'ayinterc', 'aysoilev', 'aysnowsub', 'aygpptot')
acc = {k: np.zeros(len(i)) for k in V}
for f in fs:
    with xr.open_dataset(f) as d:
        assert 'aytrans' in d, f"{f} has no ET components -- run with physics.et_components: true"
        for k in V: acc[k] += np.nan_to_num(np.asarray(d[k].values, float))[i, j] / len(fs)
sd = xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc"); pp = np.asarray(sd["PCT_NAT_PFT"].values); pv = np.asarray(sd["PCT_NATVEG"].values) / 100
if float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]): pp = pp[:, ::-1, :]; pv = pv[::-1, :]
tree = (pp[1:9].sum(0) / 100)[i, j]; nat = pv[i, j] >= 0.7
eo = et_at(i, j); go = fluxcom_at(i, j)
ok = nat & np.isfinite(eo) & (eo > 10) & np.isfinite(go) & (go > 0.02) & (lf > 0.5)
print(f"{OUT}\nyears {fs[0][-7:-3]}-{fs[-1][-7:-3]}; scored cells {ok.sum()} (natural veg >= 70 %, all-five ET, land fraction > 0.5)")
def row(lbl, k):
    k = k & ok
    if k.sum() < 30: return
    w = a[k]; W = lambda x: float(np.sum(w * x[k]) / w.sum())
    et, eob = W(acc['ayaet']), W(eo); tr, ic, se, sn = (W(acc[v]) for v in ('aytrans', 'ayinterc', 'aysoilev', 'aysnowsub')); g = W(acc['aygpptot'])
    print(f"{lbl:26s} n={k.sum():5d} | ET {et:5.0f} obs {eob:5.0f} = {et/eob:4.2f} | transp {tr:5.0f} ({tr/et:4.2f}) interc {ic:5.0f} ({ic/et:4.2f}) soil {se:5.0f} ({se/et:4.2f}) snow {sn:4.0f} ({sn/et:5.2f}) | transp/GPP {tr/max(g*1000,1e-9):5.2f} kg/gC  GPP/obs {g/max(W(go),1e-9):4.2f}")
row("ALL", np.ones_like(ok))
print("-- by CLM tree fraction --")
for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)): row(f"tree {lo:.2f}-{hi:.2f}", (tree >= lo) & (tree < hi))
print("-- by latitude --")
for lo, hi in ((60, 90), (45, 60), (30, 45), (15, 30), (0, 15), (-15, 0), (-30, -15), (-60, -30)): row(f"lat {lo:+d}..{hi:+d}", (LATV[i] >= lo) & (LATV[i] < hi))
