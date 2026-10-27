import sys, glob, numpy as np, xarray as xr
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; S = sys.argv[1]
refs = glob.glob(f'{S}/meeting/**/glob_refs.npz', recursive=True)[0]; R = np.load(refs); lat = np.asarray(R['lat'], float); lf = np.asarray(R['lf'], float)
LAT = lat[:, None] * np.ones((1, 720)) if lat.ndim == 1 else lat
area = (111.2e3 * 0.5) ** 2 * np.cos(np.deg2rad(LAT)) * lf   # m2 of true land per cell
def grid(run, var, year=2010):
    f = sorted(glob.glob(f'{V}/final_{run}/hindcast/out/*{year}*.nc')); f = [x for x in f if 'monthly' not in x]
    d = xr.open_dataset(f[0]); g = np.asarray(d[var].values, float).squeeze()
    if float(d['lat'].values[0]) < float(d['lat'].values[-1]): g = g[::-1]
    return g
bands = [('90S-30N', -90, 30), ('30-60N', 30, 60), ('60-90N', 60, 90)]
print('soil C (totcsoi) and veg C (cveg) 2010 by band, PgC')
for run in sys.argv[2:]:
    cs, cv = grid(run, 'totcsoi'), grid(run, 'cveg')
    row = [run]
    for nm, a, b in bands:
        m = (LAT >= a) & (LAT < b)
        row.append(f'{nm} soil {np.nansum(cs[m] * area[m]) / 1e12:7.1f} veg {np.nansum(cv[m] * area[m]) / 1e12:6.1f}')
    print('  '.join(row), '| global soil', round(float(np.nansum(cs * area) / 1e12), 1))
