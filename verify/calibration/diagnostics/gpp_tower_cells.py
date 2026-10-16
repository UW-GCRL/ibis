"""GPP at the towers' own grid cells: tower vs FLUXCOM vs the global run (CALIBRATION.md 47).

    python verify/calibration/diagnostics/gpp_tower_cells.py <run out dir> [--years 2001-2010] [--csv out.csv]

Tower: annual GPP from the FLUXNET-layout YY files of `inputs/sites/towers31/compact` (mean of the night- and
day-time partitions, VUT where present else CUT), over years with NEE_VUT_REF_QC >= 0.5; the tower's own years
(the global forcing ends in 2010, so model and FLUXCOM are 2001-2010 means: the comparison is of climatologies).
FLUXCOM RS+METEO and the run's `aygpptot` at the 0.5-degree cell holding the tower. Ratios are medians over towers.
"""
import sys, glob, numpy as np, pandas as pd, xarray as xr
a = sys.argv[1:]; run = a[0]
y0, y1 = (int(v) for v in (a[a.index('--years') + 1] if '--years' in a else '2001-2010').split('-'))
C = '/shared/nas2/Hangkai/ibis/inputs/sites/towers31/compact'
info = pd.read_csv(f'{C}/FLUXNET_siteinfo.csv')
gpp_m = np.mean([np.asarray(xr.open_dataset(f'{run}/ibis_yearly_{y}.nc')['aygpptot'].values, float) for y in range(y0, y1 + 1)], axis=0) * 1000.0
with xr.open_dataset(f'{run}/ibis_yearly_{y0}.nc') as d:
    mlat, mlon = d['lat'].values, d['lon'].values
with xr.open_dataset('/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc') as fc:
    ys = np.array([int(str(t)[:4]) for t in fc['time'].values]); sel = (ys >= y0) & (ys <= y1)
    gpp_f = np.nanmean(fc['gpp'].values[sel].astype(float), axis=0) * 365.25; flat, flon = fc['lat'].values, fc['lon'].values
rows = []
for _, s in info.iterrows():
    f = glob.glob(f"{C}/FLX_{s.SITE_ID}_*_YY_*.csv")
    if not f: continue
    y = pd.read_csv(f[0], na_values=[-9999, -9999.0])
    g = []
    for part in ('NT', 'DT'):
        c = f'GPP_{part}_VUT_REF' if f'GPP_{part}_VUT_REF' in y and y[f'GPP_{part}_VUT_REF'].notna().any() else f'GPP_{part}_CUT_REF'
        g.append(y[c] if c in y else pd.Series(np.nan, index=y.index))
    gy = pd.concat(g, axis=1).mean(axis=1)
    good = gy.notna() & (y.get('NEE_VUT_REF_QC', pd.Series(1.0, index=y.index)) >= 0.5)
    if good.sum() == 0: continue
    i = int(np.argmin(abs(mlat - s.LOCATION_LAT))); j = int(np.argmin(abs(mlon - s.LOCATION_LONG)))
    fi = int(np.argmin(abs(flat - s.LOCATION_LAT))); fj = int(np.argmin(abs(flon - s.LOCATION_LONG)))
    rows.append(dict(site=s.SITE_ID, igbp=s.IGBP, koppen=str(s.KOPPEN)[:1], lat=s.LOCATION_LAT, n_years=int(good.sum()),
                     tower=float(gy[good].clip(lower=0).mean()), fluxcom=float(gpp_f[fi, fj]), model=float(gpp_m[i, j])))
t = pd.DataFrame(rows)
t = t[np.isfinite(t.fluxcom) & np.isfinite(t.model) & (t.tower > 50)]
t['tower/flx'] = t.tower / t.fluxcom; t['model/flx'] = t.model / t.fluxcom; t['model/tower'] = t.model / t.tower
if '--csv' in a: t.to_csv(a[a.index('--csv') + 1], index=False)
q = lambda d: f"n {len(d):3d}  tower {d.tower.median():5.0f}  FLUXCOM {d.fluxcom.median():5.0f}  model {d.model.median():5.0f}   " \
              f"tower/FLX {d['tower/flx'].median():.2f}  model/FLX {d['model/flx'].median():.2f}  model/tower {d['model/tower'].median():.2f}"
print(f'{run}, model and FLUXCOM {y0}-{y1}; tower GPP gC m-2 yr-1 over its good years; medians over towers\n')
print('all (no WET/SNO)  ' + q(t[~t.igbp.isin(['WET', 'SNO'])]))
for k, d in t.groupby('igbp'):
    if len(d) >= 5: print(f'IGBP {k:12s} ' + q(d))
for k, d in t[~t.igbp.isin(['WET', 'SNO'])].groupby('koppen'):
    print(f'Koppen {k:10s} ' + q(d))
