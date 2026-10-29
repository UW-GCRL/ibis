"""Observed-month wetland area against the climatology: the two hindcasts' wetland CH4 by band, 2001-2010 -- decade means,
seasonal cycle, annual series and anomalies -- and the inundated area each used (finund in the observed run, the WAD2M
climatology in the other). Read-only.   python -I wadobs_eval.py <clim hindcast out> <observed hindcast out>"""
import sys, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/calibration']
from landmask import true_landfrac
A, B = sys.argv[1], sys.argv[2]; YRS = range(2001, 2011); NDAY = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
LAT = np.arange(89.75, -90, -0.5); area = (6371e3 ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * true_landfrac()
BANDS = {'90S-30N': LAT < 30, '30-60N': (LAT >= 30) & (LAT < 60), '60-90N': LAT >= 60, 'GLOBAL': np.ones(360, bool)}
def monthly(run, var):
    out = np.zeros((len(YRS), 12, 360, 720), np.float32)
    for i, y in enumerate(YRS):
        for m in range(12):
            with xr.open_dataset(f'{run}/ibis_monthly_{y}_{m + 1:02d}.nc') as d: out[i, m] = np.nan_to_num(np.asarray(d[var].values, np.float32)) if var in d else np.nan
    return out
ch4 = {k: monthly(r, 'amch4') for k, r in (('clim', A), ('obs', B))}       # kgC m-2 d-1
fin_obs = monthly(B, 'finund')
def tg(x):  # (yrs, 12, lat, lon) kgC/m2/d -> Tg CH4 per month per band
    return {b: np.array([[float((x[i, m][msk] * area[msk]).sum() * NDAY[m] * 16.0 / 12.0 / 1e9) for m in range(12)] for i in range(len(YRS))]) for b, msk in BANDS.items()}
T = {k: tg(v) for k, v in ch4.items()}
print('wetland CH4, Tg CH4 yr-1, 2001-2010: climatology run | observed-area run | difference; interannual sd of the annual total; r of the annual anomalies between the two runs')
for b in BANDS:
    a = T['clim'][b].sum(1); o = T['obs'][b].sum(1)
    print(f"  {b:8s} mean {a.mean():6.1f} | {o.mean():6.1f} | {o.mean() - a.mean():+5.1f}   IAV sd {a.std():4.2f} | {o.std():4.2f}   r(annual anomalies) {np.corrcoef(a - a.mean(), o - o.mean())[0, 1]:+.2f}")
print('\nannual totals (Tg), observed-area run minus climatology run, by year:')
for b in ('90S-30N', '30-60N', '60-90N', 'GLOBAL'):
    print(f"  {b:8s} " + ' '.join(f'{y}:{(T["obs"][b].sum(1) - T["clim"][b].sum(1))[i]:+5.1f}' for i, y in enumerate(YRS)))
print('\nseasonal cycle (decade mean, Tg per month): climatology run / observed-area run, max/min')
for b in ('90S-30N', '30-60N', '60-90N'):
    a = T['clim'][b].mean(0); o = T['obs'][b].mean(0)
    print(f"  {b:8s} clim " + ' '.join(f'{v:5.1f}' for v in a) + f'  max/min {a.max() / max(a.min(), 1e-6):5.1f}\n  {"":8s} obs  ' + ' '.join(f'{v:5.1f}' for v in o) + f'  max/min {o.max() / max(o.min(), 1e-6):5.1f}')
if np.isfinite(fin_obs).all():
    ar = {b: np.array([[float((fin_obs[i, m][msk] * area[msk]).sum() / 1e12) for m in range(12)] for i in range(len(YRS))]) for b, msk in BANDS.items()}
    print('\ninundated area used by the observed run (Mkm2, true land): decade mean, interannual sd of the annual mean, min and max year')
    for b in BANDS:
        y = ar[b].mean(1); print(f"  {b:8s} {y.mean():5.2f}  sd {y.std():5.3f}  min {y.min():5.2f} ({2001 + int(y.argmin())})  max {y.max():5.2f} ({2001 + int(y.argmax())})")
