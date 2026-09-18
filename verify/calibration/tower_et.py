"""Tower-scale ET check: FLUXNET2015 latent heat vs the model's monthly ET vs the ET ensemble, at the tower's cell.

    ibis-python verify/calibration/tower_et.py [<monthly output dir with ibis_monthly_*.nc>]

LE_F_MDS (W m-2, monthly mean) -> mm month-1 via 2.45 MJ kg-1. Sites with
>= 24 valid months. Per IGBP class: annual ET tower / model / ensemble (mean
over sites), amplitude and the month of peak relative to the tower's. Towers
see a footprint, the model a 0.5-degree cell; the ensemble at the same cell
is the fair third party (CALIBRATION.md 15).
"""
import sys, glob, re, numpy as np, pandas as pd, xarray as xr, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
from fit_carbon import LATV, LONV, MASK, NDAYS
F = '/shared/nas2/Min/FLUXNET2015/fluxnet_compact'; V = '/shared/nas2/Hangkai/ibis/verify_outputs'
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{V}/val_v12c/monthly/out'
meta = pd.read_csv(f'{F}/FLUXNET_siteinfo.csv', encoding='utf-8-sig').set_index('SITE_ID')
fs = sorted(glob.glob(f'{OUT}/ibis_monthly_*.nc')); mon = np.array([int(re.search(r'_(\d\d)\.nc$', f).group(1)) for f in fs])
E = np.zeros((12, 360, 720)); n = np.zeros(12)
for f, mm in zip(fs, mon):
    with xr.open_dataset(f) as d: E[mm - 1] += np.nan_to_num(np.asarray(d['amaet'].values, float)); n[mm - 1] += 1
E /= n[:, None, None]
ens = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/et/et_ensemble_0.5deg.npz'); EC = ens['ens_clim_mean'] * NDAYS[:, None, None]; EN = ens['ens_n']
m = np.array(xr.open_dataset(MASK).landfrac.values) > 0
rows = []
for path in sorted(glob.glob(f'{F}/FLX_*_FULLSET_MM_*.csv')):
    site = path.split('FLX_')[1].split('_')[0]
    if site not in meta.index: continue
    df = pd.read_csv(path)
    if 'LE_F_MDS' not in df: continue
    df = df[df['LE_F_MDS'] > -9000].copy()
    if len(df) < 24: continue
    ts = pd.to_datetime(df['TIMESTAMP'].astype(str).str[:10]); df['mo'] = ts.dt.month
    le = df.groupby('mo')['LE_F_MDS'].mean().reindex(range(1, 13)).values
    if np.isnan(le).any(): continue
    et_t = np.maximum(le, 0) / 2.45e6 * 86400 * NDAYS                    # mm/month
    lat = float(meta.loc[site, 'LOCATION_LAT']); lon = float(meta.loc[site, 'LOCATION_LONG'])
    i = int(np.argmin(np.abs(LATV - lat))); j = int(np.argmin(np.abs(LONV - lon)))
    if not m[i, j]: continue
    et_m = E[:, i, j]; et_e = EC[:, i, j] if EN[i, j] == 5 else np.full(12, np.nan)
    circ = lambda k: (k + 6) % 12 - 6
    rows.append(dict(site=site, veg=str(meta.loc[site, 'IGBP']), lat=lat, ann_t=et_t.sum(), ann_m=et_m.sum(), ann_e=et_e.sum(),
                     amp_t=et_t.max() / max(et_t.min(), 1), amp_m=et_m.max() / max(et_m.min(), 1),
                     off_m=circ(np.argmax(et_m) - np.argmax(et_t)), off_e=circ(np.argmax(et_e) - np.argmax(et_t)) if np.isfinite(et_e).all() else np.nan))
R = pd.DataFrame(rows); print(f"{OUT}\n{len(R)} towers with >= 24 valid LE months; ensemble at the cell where all five products cover it ({R['ann_e'].notna().sum()} sites)")
print(f"{'IGBP':6s}{'n':>4s} | annual ET mm/yr: tower  model  ensemble | model/tower  ens/tower | amplitude tower model | peak offset vs tower: model  ens")
for veg, g in R.groupby('veg'):
    if len(g) < 3: continue
    print(f"{veg:6s}{len(g):4d} | {g.ann_t.mean():17.0f} {g.ann_m.mean():6.0f} {np.nanmean(g.ann_e):9.0f} | {g.ann_m.mean()/g.ann_t.mean():10.2f} {np.nanmean(g.ann_e)/g.ann_t[g.ann_e.notna()].mean() if g.ann_e.notna().any() else np.nan:10.2f} | {min(g.amp_t.mean(),30):9.1f} {min(g.amp_m.mean(),30):5.1f} | {g.off_m.mean():+21.2f} {np.nanmean(g.off_e):+5.2f}")
print(f"{'ALL':6s}{len(R):4d} | {R.ann_t.mean():17.0f} {R.ann_m.mean():6.0f} {np.nanmean(R.ann_e):9.0f} | {R.ann_m.mean()/R.ann_t.mean():10.2f} {np.nanmean(R.ann_e)/R.ann_t[R.ann_e.notna()].mean():10.2f} | {min(R.amp_t.mean(),30):9.1f} {min(R.amp_m.mean(),30):5.1f} | {R.off_m.mean():+21.2f} {np.nanmean(R.off_e):+5.2f}")
print("\nby |lat| band: model/tower, ensemble/tower")
for lo, hi in ((0, 25), (25, 45), (45, 60), (60, 90)):
    g = R[(R.lat.abs() >= lo) & (R.lat.abs() < hi)]
    if len(g) >= 3: print(f"  {lo:2d}-{hi:2d}  n={len(g):3d}  model/tower {g.ann_m.mean()/g.ann_t.mean():.2f}  ens/tower {np.nanmean(g.ann_e)/g.ann_t[g.ann_e.notna()].mean() if g.ann_e.notna().any() else float('nan'):.2f}  tower {g.ann_t.mean():.0f} mm/yr")
