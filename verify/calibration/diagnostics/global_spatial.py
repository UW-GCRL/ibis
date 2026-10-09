"""Spatial comparison of a global run's GPP and ET with the gridded references (CALIBRATION.md 39): maps of model,
reference and difference, zonal means, totals and cell-wise statistics by latitude band.

    python verify/calibration/diagnostics/global_spatial.py <out dir> <yearly dir> [<yearly dir> ...] [--years 1991-2000]

GPP reference: FLUXCOM RS+METEO (g m-2 d-1 monthly, 1980-2013) averaged over the run's years. ET reference: the
five-product ensemble (`et_ensemble_0.5deg.npz`: MODIS, MOD16A2, GLEAM, DOLCE, FLUXCOM LE; 2001-2009 annual means and
the product min/max). Model: `aygpptot` (kgC m-2 yr-1) and `ayaet` (mm yr-1) of the yearly files. Statistics over true
land (ESA CCI fraction > 5 %) north of 60 S; totals area-weighted with the true land fraction.
"""
import sys, re, glob, os, numpy as np, xarray as xr, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1]; args = sys.argv[2:]; yrs = (1991, 2000)
if '--years' in args: k = args.index('--years'); yrs = tuple(int(v) for v in args[k + 1].split('-')); args = args[:k] + args[k + 2:]
os.makedirs(OUT, exist_ok=True); sys.path.insert(0, 'verify/calibration'); from landmask import true_landfrac
LAT = np.arange(89.75, -90, -0.5); R = 6371.0; lf = true_landfrac()
area = (R ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * 1.0e6 * lf
land = (lf > 0.05) & (LAT[:, None] > -60)
files = {}
for d in args:
    for f in glob.glob(f'{d}/ibis_yearly_*.nc'): files[int(re.search(r'(\d+)\.nc$', f).group(1))] = f
acc = {}; n = 0
for y in sorted(files):
    if y < yrs[0] or y > yrs[1]: continue
    with xr.open_dataset(files[y]) as d:
        for v in ('aygpptot', 'ayaet'): acc[v] = acc.get(v, 0) + np.nan_to_num(np.asarray(d[v].values, float))
    n += 1
if n == 0: raise SystemExit('no yearly files in the range')
gpp_m = acc['aygpptot'] / n; et_m = acc['ayaet'] / n
# references
with xr.open_dataset('/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc') as fc:
    t = fc['time'].values; ysel = np.array([int(str(x)[:4]) for x in t]); sel = (ysel >= yrs[0]) & (ysel <= yrs[1])
    if sel.sum() == 0: sel = (ysel >= 2001) & (ysel <= 2010)
    gpp_r = np.nanmean(np.asarray(fc['gpp'].values[sel], float), axis=0) * 365.25 / 1000.0          # g m-2 d-1 -> kgC m-2 yr-1
    if float(fc['lat'].values[0]) < float(fc['lat'].values[-1]): gpp_r = gpp_r[::-1]
z = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/et/et_ensemble_0.5deg.npz', allow_pickle=True); et_r, et_lo, et_hi = z['ens_ann_mean'], z['ens_ann_min'], z['ens_ann_max']
if float(z['lat'][0]) < float(z['lat'][-1]): et_r, et_lo, et_hi = et_r[::-1], et_lo[::-1], et_hi[::-1]
bands = {'global': land, '90S-30N': land & (LAT[:, None] < 30), '30N-60N': land & (LAT[:, None] >= 30) & (LAT[:, None] < 60), '60N-90N': land & (LAT[:, None] >= 60)}
def stats(m, r, mask, lo=None, hi=None):
    ok = mask & np.isfinite(m) & np.isfinite(r); mm, rr, a = m[ok], r[ok], area[ok]
    out = dict(model=float((mm * a).sum() / a.sum()), ref=float((rr * a).sum() / a.sum()), r=float(np.corrcoef(mm, rr)[0, 1]), ratio=float((mm * a).sum() / (rr * a).sum()))
    if lo is not None: out['within_spread'] = float(np.mean((mm >= lo[ok]) & (mm <= hi[ok])))
    return out
print(f'GPP {yrs[0]}-{yrs[1]} vs FLUXCOM (area-weighted land means, kgC m-2 yr-1; r over cells; ratio model/ref):')
for b, m in bands.items():
    s = stats(gpp_m, gpp_r, m); tot = float((gpp_m * area)[m & np.isfinite(gpp_r)].sum()) * 1e-12; totr = float((np.nan_to_num(gpp_r) * area)[m & np.isfinite(gpp_r)].sum()) * 1e-12
    print(f"  {b:8s} model {s['model']:.2f} ref {s['ref']:.2f} ratio {s['ratio']:.2f} r {s['r']:.2f}   totals on FLUXCOM land {tot:.0f} vs {totr:.0f} PgC")
print(f'\nET {yrs[0]}-{yrs[1]} vs the 5-product ensemble (mm yr-1; share of cells inside the product min-max):')
for b, m in bands.items():
    s = stats(et_m, et_r, m, et_lo, et_hi); print(f"  {b:8s} model {s['model']:.0f} ref {s['ref']:.0f} ratio {s['ratio']:.2f} r {s['r']:.2f} within spread {s['within_spread']:.2f}")
# figures
plt.rcParams.update({'font.size': 9, 'figure.dpi': 110})
def panel(ax, fld, title, vmin, vmax, cmap, unit):
    im = ax.imshow(np.where(land, fld, np.nan), extent=[-180, 180, -90, 90], vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto'); ax.set_title(title, loc='left', fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-60, 90)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=unit)
zon = lambda fld: np.nansum(np.where(land & np.isfinite(fld), fld, 0) * area, axis=1) / np.maximum(np.nansum(np.where(land & np.isfinite(fld), area, 0), axis=1), 1)
for name, m, r, lo, hi, unit, vmax, dmax, refname in (('GPP', gpp_m, gpp_r, None, None, 'kgC m-2 yr-1', 3.5, 1.5, 'FLUXCOM'), ('ET', et_m, et_r, et_lo, et_hi, 'mm yr-1', 1500, 500, 'ensemble of 5 products')):
    fig = plt.figure(figsize=(15, 8)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55])
    a1, a2, a3, a4 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]); az = fig.add_subplot(gs[:, 2])
    panel(a1, m, f'model {name}, {yrs[0]}-{yrs[1]}', 0, vmax, 'YlGn' if name == 'GPP' else 'YlGnBu', unit); panel(a2, r, f'{refname}', 0, vmax, 'YlGn' if name == 'GPP' else 'YlGnBu', unit)
    panel(a3, m - r, 'model - reference', -dmax, dmax, 'RdBu_r', unit)
    if lo is not None: panel(a4, np.where(m < lo, m - lo, np.where(m > hi, m - hi, 0.0)), 'outside the product spread (0 = inside)', -dmax, dmax, 'RdBu_r', unit)
    else: panel(a4, np.where(r > 0.05, m / r, np.nan), 'model / reference', 0, 2.5, 'PuOr_r', '-')
    az.plot(zon(m), LAT, color='#7570b3', label='model'); az.plot(zon(r), LAT, color='k', ls='--', label=refname)
    if lo is not None: az.fill_betweenx(LAT, zon(lo), zon(hi), color='0.85', label='product spread')
    az.set_ylim(-60, 90); az.set_title(f'zonal land mean ({unit})', loc='left', fontsize=9); az.legend(frameon=False, fontsize=8); az.set_ylabel('latitude')
    fig.suptitle(f'{name}: global run vs {refname}', x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/S_{name}_spatial.png'); plt.close(fig)
print('figures in', OUT)
