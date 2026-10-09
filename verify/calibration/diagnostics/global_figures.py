"""Maps and zonal means of a global run's last decade (CALIBRATION.md 39): GPP, net land flux (NPP - Rh), wetland CH4,
vegetation and soil carbon; GPP against FLUXCOM and wetland CH4 against the GMB 2025 bottom-up mean map.

    python verify/calibration/diagnostics/global_figures.py <out dir> <yearly dir> [<yearly dir> ...] [--years 2001-2010]
"""
import sys, re, glob, os, numpy as np, xarray as xr, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1]; args = sys.argv[2:]; yrs = (2001, 2010)
if '--years' in args: k = args.index('--years'); yrs = tuple(int(v) for v in args[k + 1].split('-')); args = args[:k] + args[k + 2:]
os.makedirs(OUT, exist_ok=True); sys.path.insert(0, 'verify/calibration'); from landmask import true_landfrac
LAT = np.arange(89.75, -90, -0.5); LON = np.arange(-179.75, 180, 0.5); R = 6371.0
area = (R ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * 1.0e6 * true_landfrac()
files = {}
for d in args:
    for f in glob.glob(f'{d}/ibis_yearly_*.nc'): files[int(re.search(r'(\d+)\.nc$', f).group(1))] = f
acc = {}; n = 0
for y in sorted(files):
    if y < yrs[0] or y > yrs[1]: continue
    with xr.open_dataset(files[y]) as d:
        for v in ('aygpptot', 'aynpptot', 'ayco2mic', 'cveg', 'totcsoi', 'aych4'):
            if v in d: acc[v] = acc.get(v, 0) + np.asarray(d[v].values, float)
    n += 1
if n == 0: raise SystemExit('no yearly files in range')
F = {v: a / n for v, a in acc.items()}; land = true_landfrac() > 0.05
plt.rcParams.update({'font.size': 9, 'figure.dpi': 110})
def panel(ax, fld, title, vmin, vmax, cmap, unit):
    z = np.where(land, fld, np.nan); im = ax.imshow(z, extent=[-180, 180, -90, 90], vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto'); ax.set_title(title, loc='left', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=unit)
tot = lambda v: float(np.nansum(F[v] * area)) * 1e-12
fig, axs = plt.subplots(3, 2, figsize=(14, 12)); axs = axs.ravel()
panel(axs[0], F['aygpptot'], f'GPP {yrs[0]}-{yrs[1]} mean, {tot("aygpptot"):.0f} PgC yr-1', 0, 3.5, 'YlGn', 'kgC m-2 yr-1')
panel(axs[1], F['aynpptot'] - F['ayco2mic'], f'NPP - Rh, {tot("aynpptot") - tot("ayco2mic"):+.1f} PgC yr-1 (+ = uptake)', -0.3, 0.3, 'RdBu', 'kgC m-2 yr-1')
panel(axs[2], F['cveg'], f'above-ground wood carbon, {tot("cveg"):.0f} PgC', 0, 25, 'YlOrBr', 'kgC m-2')
panel(axs[3], F['totcsoi'], f'soil carbon, {tot("totcsoi"):.0f} PgC', 0, 40, 'copper_r', 'kgC m-2')
if 'aych4' in F:
    ch4 = F['aych4'] * (16.043 / 12.011) * 1e3 / 365.25                     # kgC m-2 yr-1 -> g CH4 m-2 d-1 ... shown as mg m-2 d-1
    panel(axs[4], ch4 * 1e3, f'wetland CH4 emission (cell mean), {float(np.nansum(F["aych4"] * area)) * 1e-9 * 16.043 / 12.011:.0f} Tg CH4 yr-1', 0, 60, 'magma_r', 'mg CH4 m-2 d-1')
else: axs[4].set_title('wetland CH4: not in these files', loc='left')
G = '/shared/nas2/Hangkai/ibis/inputs/eval/mip/gmb2024/gridded/fig_maps_wetlands_anthropogenic.nc'
try:
    with xr.open_dataset(G) as g:
        v = [k for k in g.data_vars if 'wet' in k.lower()][0]; w = np.asarray(g[v].values, float); w = w[0] if w.ndim == 3 else w
        glat = np.asarray(g['lat'].values); 
        if glat[0] < glat[-1]: w = w[::-1]
        im = axs[5].imshow(w, extent=[-180, 180, -90, 90], vmin=0, vmax=60, cmap='magma_r', aspect='auto'); axs[5].set_title(f'GMB 2025 bottom-up wetland mean map ({v})', loc='left', fontsize=9); axs[5].set_xticks([]); axs[5].set_yticks([]); plt.colorbar(im, ax=axs[5], fraction=0.03, pad=0.02, label='mg CH4 m-2 d-1')
except Exception as e: axs[5].set_title(f'GMB map unavailable: {e}', loc='left', fontsize=8)
fig.suptitle(f'Global run, {yrs[0]}-{yrs[1]} means', x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/G1_maps.png'); plt.close(fig)
# zonal
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
zon = lambda fld: np.nansum(np.where(land, fld, 0) * area, axis=1) / np.maximum(np.nansum(np.where(land, area, 0), axis=1), 1)
axs[0].plot(zon(F['aygpptot']), LAT, color='#1b9e77'); axs[0].set_title('GPP (kgC m-2 yr-1, land mean)', loc='left', fontsize=9)
try:
    fc = xr.open_dataset('/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc'); vv = [k for k in fc.data_vars if 'gpp' in k.lower()][0]; f0 = np.asarray(fc[vv].values, float); f0 = np.nanmean(f0, axis=0) if f0.ndim == 3 else f0
    f0 = f0 * 365.25 / 1000.0; flat = np.asarray(fc['lat'].values)
    if flat[0] < flat[-1]: f0 = f0[::-1]
    if f0.shape == (360, 720): axs[0].plot(zon(f0), LAT, color='k', ls='--', label='FLUXCOM'); axs[0].legend(frameon=False)
except Exception as e: axs[0].text(0.02, 0.02, f'FLUXCOM n/a: {e}', transform=axs[0].transAxes, fontsize=7)
axs[1].plot(zon(F['aynpptot'] - F['ayco2mic']), LAT, color='#7570b3'); axs[1].axvline(0, color='k', lw=0.6); axs[1].set_title('NPP - Rh (kgC m-2 yr-1)', loc='left', fontsize=9)
if 'aych4' in F:
    z = np.nansum(np.where(land, F['aych4'], 0) * area, axis=1) * 1e-9 * 16.043 / 12.011
    axs[2].plot(z, LAT, color='#d95f02'); axs[2].set_title('wetland CH4 per latitude row (Tg CH4 yr-1)', loc='left', fontsize=9)
for ax in axs: ax.set_ylim(-60, 90); ax.set_ylabel('latitude')
fig.tight_layout(); fig.savefig(f'{OUT}/G2_zonal.png'); plt.close(fig); print('figures in', OUT)
