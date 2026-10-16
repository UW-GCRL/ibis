"""Control vs per-type leaf-area cap (CALIBRATION.md 46): one year of two global runs from the same restart.

    python verify/calibration/diagnostics/laicap_compare.py <ctl out dir> <cap out dir> [--year 2010]

Per latitude band and globally (true land north of 60 S, area-weighted): GPP vs FLUXCOM RS+METEO, ET vs the
five-product ensemble, NPP and NEE (model only), and the cell LAI fu x laiu + fl x lail (monthly means; annual mean and
annual maximum month) vs MODIS (the CLM surface-data PFT LAI that site_fit uses: sum of the PFT LAIs x their shares x
the natural-vegetation share).
"""
import sys, os, numpy as np, xarray as xr
sys.path.insert(0, 'verify/calibration'); from landmask import true_landfrac
ctl, cap = sys.argv[1], sys.argv[2]; yr = int(sys.argv[sys.argv.index('--year') + 1]) if '--year' in sys.argv else 2010
LAT = np.arange(89.75, -90, -0.5); lf = true_landfrac()
area = 6371.0e3 ** 2 * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * lf
land = (lf > 0.05) & (LAT[:, None] > -60)
bands = {'global': land, '90S-30N': land & (LAT[:, None] < 30), '30N-60N': land & (LAT[:, None] >= 30) & (LAT[:, None] < 60),
         '60N-90N': land & (LAT[:, None] >= 60)}


def run(d):
    with xr.open_dataset(f'{d}/ibis_yearly_{yr}.nc') as y:
        o = {k: np.nan_to_num(np.asarray(y[v].values, float)) for k, v in (('gpp', 'aygpptot'), ('et', 'ayaet'), ('npp', 'aynpptot'), ('nee', 'ayneetot')) if v in y}
    mon = []
    for m in range(1, 13):
        with xr.open_dataset(f'{d}/ibis_monthly_{yr}_{m:02d}.nc') as x:
            mon.append(np.nan_to_num(np.asarray(x['fu'].values * x['laiu'].values + x['fl'].values * x['lail'].values, float)))
    mon = np.array(mon); o['lai_mean'] = mon.mean(0); o['lai_max'] = mon.max(0)
    return o


A, B = run(ctl), run(cap)
with xr.open_dataset('/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc') as fc:
    ys = np.array([int(str(t)[:4]) for t in fc['time'].values])
    g = np.nanmean(np.asarray(fc['gpp'].values[ys == yr], float), axis=0) * 365.25 / 1000.0
    if float(fc['lat'].values[0]) < float(fc['lat'].values[-1]): g = g[::-1]
z = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/et/et_ensemble_0.5deg.npz', allow_pickle=True); e = z['ens_ann_mean']
if float(z['lat'][0]) < float(z['lat'][-1]): e = e[::-1]
MB = '/shared/nas2/Hangkai/ELM Edge Effects/inputdata'
with xr.open_dataset(f'{MB}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc') as ds, xr.open_dataset(f'{MB}/surfdata_0.5x0.5_CH_v1_all_forests.nc') as sd:
    pv = np.asarray(sd['PCT_NATVEG'].values) / 100.0; pp = np.asarray(sd['PCT_NAT_PFT'].values) / 100.0; nt = ds.sizes['time']
    mm = np.zeros((12,) + pv.shape)
    for m in range(12):
        sel = [t for t in range(nt) if t % 12 == m]
        mm[m] = sum(np.nan_to_num(ds[f'LAI_{k}'].values[sel].mean(axis=0)) * pp[k] for k in range(1, 17) if f'LAI_{k}' in ds) * pv
    if float(sd['LATIXY'].values[0, 0]) < float(sd['LATIXY'].values[-1, 0]): mm = mm[:, ::-1]
REF = {'gpp': g, 'et': e, 'lai_mean': mm.mean(0), 'lai_max': mm.max(0)}


def amean(f, msk):
    ok = msk & np.isfinite(f); return float((f[ok] * area[ok]).sum() / area[ok].sum())


print(f'{yr}: control {ctl}\n      cap     {cap}')
for v, unit, tot in (('gpp', 'PgC yr-1', 1e-12), ('npp', 'PgC yr-1', 1e-12), ('nee', 'PgC yr-1 (+ = sink)', 1e-12), ('et', 'mm yr-1', None),
                     ('lai_mean', 'm2 m-2', None), ('lai_max', 'm2 m-2', None)):
    if v not in A: continue
    print(f'\n{v} ({unit})' + ('  -- totals on the reference\'s land' if v in REF and tot else ''))
    for b, msk in bands.items():
        r = REF.get(v); m = msk & (np.isfinite(r) if r is not None else True)
        if tot: a, c = (float((A[v] * area)[m].sum() * tot), float((B[v] * area)[m].sum() * tot)); rr = float((np.nan_to_num(r) * area)[m].sum() * tot) if r is not None else None
        else: a, c = amean(A[v], m), amean(B[v], m); rr = amean(r, m) if r is not None else None
        s = f'  {b:8s} control {a:8.2f}  cap {c:8.2f}  ({100 * (c / a - 1) if a else 0:+5.1f}%)'
        if rr is not None: s += f'   ref {rr:8.2f}   model/ref {a / rr:5.2f} -> {c / rr:5.2f}'
        print(s)
