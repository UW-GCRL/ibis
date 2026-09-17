"""Gridded ET observation ensemble on the IBIS 0.5-degree grid (N->S), 2001-2009.

    ibis-python verify/calibration/et_obs.py     # builds et_ensemble_0.5deg.npz, prints the spread

Products (DATA_INVENTORY.md 4): MODIS (MOD16-derived), MOD16A2, GLEAM v3.3a,
DOLCE (hfls) and FLUXCOM LE -- all 0.5 deg; WECANN is 1 deg and left out.
Everything to mm/day: kg m-2 s-1 x 86400; W m-2 x 86400 / 2.45e6; MJ m-2 d-1
/ 2.45. Output per product: the 2001-2009 annual mean (mm/yr) and monthly
climatology (12, 360, 720) in mm/day; plus ensemble mean / min / max and the
product count. The ET stage scores against the ensemble mean and reads the
spread as the observation's uncertainty (CALIBRATION.md 15). Loader:
    e = np.load(OUT); e['ens_ann_mean'][i, j]; e['ens_clim_mean'][:, i, j]
"""
import os, warnings, numpy as np, xarray as xr
warnings.filterwarnings("ignore")
D = '/shared/nas2/Hangkai/ibis/inputs/eval/et'
OUT = f'{D}/et_ensemble_0.5deg.npz'
PROD = {'MODIS': ('MODIS_et_0.5x0.5.nc', 'et'), 'MOD16A2': ('MOD16A2_et.nc', 'et'),
        'GLEAM': ('GLEAMv3.3a_et.nc', 'et'), 'DOLCE': ('DOLCE_et.nc', 'hfls'),
        'FLUXCOM': ('FLUXCOM_le.nc', 'le')}
LV = 2.45e6                                    # J/kg, latent heat of vaporisation
def to_mmday(a, units):
    u = units.lower().replace(' ', '')
    if 'kg' in u: return a * 86400.0
    if u.startswith('w') and 'm' in u: return a * 86400.0 / LV
    if 'mj' in u: return a / 2.45
    raise ValueError(f"unknown ET units {units!r}")
out = {}
for name, (f, v) in PROD.items():
    d = xr.open_dataset(f'{D}/{f}', decode_times=True)
    tv = d['time'].values
    yrs = np.array([int(str(x)[:4]) for x in tv]); mos = np.array([int(str(x)[5:7]) for x in tv])
    sel = (yrs >= 2001) & (yrs <= 2009)
    assert sel.sum() == 108, (name, sel.sum())
    a = to_mmday(d[v].isel(time=np.where(sel)[0]).values.astype(np.float32), d[v].attrs.get('units', ''))
    if float(d.lat[0]) < float(d.lat[-1]): a = a[:, ::-1, :]                 # S->N -> N->S
    if float(d.lon[0]) >= 0.0: a = np.roll(a, a.shape[2] // 2, axis=2)        # 0..360 -> -180..180
    a = np.where(np.isfinite(a) & (a > -1.0) & (a < 50.0), a, np.nan)         # fills / garbage
    mm = mos[sel]
    clim = np.stack([np.nanmean(a[mm == k], axis=0) for k in range(1, 13)])   # (12, 360, 720) mm/day
    ann = np.nanmean(a, axis=0) * 365.25                                       # mm/yr
    out[f'{name}_clim'] = clim.astype(np.float32); out[f'{name}_ann'] = ann.astype(np.float32)
    print(f"{name:8s} {f:24s} units={d[v].attrs.get('units', '?'):14s} lat0={float(d.lat[0]):7.2f} lon0={float(d.lon[0]):8.2f}  "
          f"mean over finite cells {np.nanmean(ann):6.0f} mm/yr  finite frac {np.isfinite(ann).mean():.2f}", flush=True)
anns = np.stack([out[f'{n}_ann'] for n in PROD]); clims = np.stack([out[f'{n}_clim'] for n in PROD])
out['ens_ann_mean'] = np.nanmean(anns, axis=0); out['ens_ann_min'] = np.nanmin(anns, axis=0); out['ens_ann_max'] = np.nanmax(anns, axis=0)
out['ens_n'] = np.isfinite(anns).sum(axis=0).astype(np.int8)
out['ens_clim_mean'] = np.nanmean(clims, axis=0); out['ens_clim_min'] = np.nanmin(clims, axis=0); out['ens_clim_max'] = np.nanmax(clims, axis=0)
out['products'] = np.array(list(PROD)); out['period'] = np.array([2001, 2009])
out['lat'] = np.arange(89.75, -90, -0.5); out['lon'] = np.arange(-179.75, 180, 0.5)
np.savez_compressed(OUT, **out); print("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.0f} MB")
allok = out['ens_n'] == 5
m = out['ens_ann_mean'][allok]; sp = (out['ens_ann_max'] - out['ens_ann_min'])[allok]
w = np.cos(np.deg2rad(np.broadcast_to(out['lat'][:, None], (360, 720))[allok]))
print(f"cells with all 5 products: {allok.sum()}; area-wtd ensemble mean {np.sum(w * m) / w.sum():.0f} mm/yr; "
      f"mean (max-min) spread {np.sum(w * sp) / w.sum():.0f} mm/yr = {100 * np.sum(w * sp) / np.sum(w * m):.0f}% of the mean")
for n in PROD:
    x = out[f'{n}_ann'][allok]
    print(f"  {n:8s} / ensemble mean: {np.sum(w * x) / np.sum(w * m):.3f}   r = {np.corrcoef(x, m)[0, 1]:.3f}")
