"""Three hindcasts that differ only in the wetland area (hourly CH4, hourly bands): climatology / learned / observed.
Wetland CH4 by band 2001-2010 against GMB 2025, the year-to-year spread and agreement, the seasonal cycle, and the area used.
    python -I compare_area_modes.py clim=<out> learned=<out> observed=<out>"""
import sys, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/calibration']
from landmask import true_landfrac
runs = dict(a.split('=', 1) for a in sys.argv[1:]); YRS = range(2001, 2011); NDAY = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
LAT = np.arange(89.75, -90, -0.5); area = (6371e3 ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * true_landfrac()
B = {'90S-30N': LAT < 30, '30-60N': (LAT >= 30) & (LAT < 60), '60-90N': LAT >= 60, 'GLOBAL': np.ones(360, bool)}
GMB = {'90S-30N': (109.5, 65.7, 146.1), '30-60N': (32.1, 16.3, 60.1), '60-90N': (11.7, 3.6, 28.1), 'GLOBAL': (153.3, 116.1, 189.1)}
wd = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc'); FW = np.nan_to_num(np.asarray(wd['Fw'].values, float))
FW = FW[:, ::-1] if float(wd['lat'].values[0]) < float(wd['lat'].values[-1]) else FW; CLIM = FW.reshape(21, 12, 360, 720).mean(0)
CH4 = {}; AREA = {}
for k, d in runs.items():
    c = np.zeros((10, 12, 4)); a = np.zeros((10, 12, 4))
    for i, y in enumerate(YRS):
        for m in range(12):
            with xr.open_dataset(f'{d}/ibis_monthly_{y}_{m + 1:02d}.nc') as ds:
                x = np.nan_to_num(np.asarray(ds['amch4'].values, float)); f = np.nan_to_num(np.asarray(ds['finund'].values, float)) if 'finund' in ds else CLIM[m]
            for j, (b, msk) in enumerate(B.items()):
                c[i, m, j] = (x[msk] * area[msk]).sum() * NDAY[m] * 16.0 / 12.0 / 1e9; a[i, m, j] = (f[msk] * area[msk]).sum() / 1e12
    CH4[k] = c; AREA[k] = a
print('wetland CH4, Tg CH4 yr-1, 2001-2010 (hourly flux, hourly bands); in brackets the year-to-year sd of the annual total')
print(f"{'band':9s} " + ' '.join(f'{k:>18s}' for k in runs) + '   GMB 2025 BU 2000-09')
for j, b in enumerate(B):
    print(f"{b:9s} " + ' '.join(f'{CH4[k][:, :, j].sum(1).mean():10.1f} ({CH4[k][:, :, j].sum(1).std():4.2f})  ' for k in runs) + f'   {GMB[b][0]:.1f} [{GMB[b][1]:.0f}-{GMB[b][2]:.0f}]')
ks = list(runs)
if 'learned' in runs and 'observed' in runs:
    print('\nlearned vs observed: correlation of the annual anomalies of CH4 | of the wetland area; mean difference of CH4 (Tg yr-1)')
    for j, b in enumerate(B):
        cl = CH4['learned'][:, :, j].sum(1); co = CH4['observed'][:, :, j].sum(1); al = AREA['learned'][:, :, j].mean(1); ao = AREA['observed'][:, :, j].mean(1)
        print(f"  {b:9s} CH4 r {np.corrcoef(cl - cl.mean(), co - co.mean())[0, 1]:+.2f} | area r {np.corrcoef(al - al.mean(), ao - ao.mean())[0, 1]:+.2f} | CH4 learned - observed {cl.mean() - co.mean():+5.1f} | area sd learned {al.std():.3f} observed {ao.std():.3f} Mkm2")
    print('\nannual totals by year, GLOBAL (Tg): ' + ' | '.join(f'{k}: ' + ' '.join(f'{v:.0f}' for v in CH4[k][:, :, 3].sum(1)) for k in ks))
print('\nseasonal cycle, decade mean, Tg per month')
for j, b in enumerate(list(B)[:3]):
    for k in ks: print(f"  {b:9s} {k:9s} " + ' '.join(f'{v:5.1f}' for v in CH4[k][:, :, j].mean(0)))
print('\nwetland area used, decade mean (Mkm2, true land): ' + ' | '.join(f'{k}: ' + ' '.join(f'{AREA[k][:, :, j].mean():.2f}' for j in range(4)) for k in ks))
