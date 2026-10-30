"""Figures: wetland area and wetland CH4 under the three area modes (climatology / learned / observed), hourly CH4 with the
hourly bands, 2001-2010; plus the release as it stood before 2026-10-29 (month-end snapshot, old bands, mis-registered WAD2M).
    python plot_wetland_modes.py <clim out> <learned out> <observed out> <old release out> <figure dir>
Writes wetland_area_series.png, wetland_area_maps.png, wetland_ch4_series.png. CPU environment (matplotlib, xarray)."""
import sys, numpy as np, xarray as xr, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/verify/calibration']
from landmask import true_landfrac
CLIM, LEARN, OBS, OLD, FIG = sys.argv[1:6]
NDAY = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0', '#fcfcfb'
COL = {'climatology': '#8a8984', 'learned': '#2a78d6', 'observed': '#eb6834', 'old': '#3d3c38'}
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 12,
                     'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'legend.frameon': False})
LAT = np.arange(89.75, -90, -0.5); lf = np.nan_to_num(true_landfrac()); AREA = (6371e3 ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * lf
BANDS = {'90°S–30°N': LAT < 30, '30–60°N': (LAT >= 30) & (LAT < 60), '60–90°N': LAT >= 60, 'Global': np.ones(360, bool)}
GMB = {'90°S–30°N': (109.5, 65.7, 146.1), '30–60°N': (32.1, 16.3, 60.1), '60–90°N': (11.7, 3.6, 28.1), 'Global': (153.3, 116.1, 189.1)}
def monthly(d, v):
    return np.array([np.nan_to_num(np.asarray(xr.open_dataset(f'{d}/ibis_monthly_{y}_{m:02d}.nc')[v].values, np.float32)) for y in range(2001, 2011) for m in range(1, 13)])
wd = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc'); FW = np.nan_to_num(np.asarray(wd['Fw'].values, np.float32))
FW = FW[:, ::-1] if float(wd['lat'].values[0]) < float(wd['lat'].values[-1]) else FW; t = wd['time'].dt.year.values
AO = FW[(t >= 2001) & (t <= 2010)]; AC = np.tile(np.clip(FW.reshape(21, 12, 360, 720).mean(0), 0, 1), (10, 1, 1)); AL = monthly(LEARN, 'finund')
land = np.isfinite(xr.open_dataset(f'{LEARN}/ibis_monthly_2001_01.nc')['fu'].values)
AO = np.where(land, AO, 0); AC = np.where(land, AC, 0)
CH = {k: monthly(d, 'amch4') for k, d in (('climatology', CLIM), ('learned', LEARN), ('observed', OBS), ('old', OLD))}
nd = np.tile(NDAY, 10)
def band_area(A, msk): return (A[:, msk] * AREA[msk]).sum((1, 2)) / 1e12
def band_ch4(C, msk): return (C[:, msk] * AREA[msk]).sum((1, 2)) * nd * 16.0 / 12.0 / 1e9
tm = 2001 + (np.arange(120) + 0.5) / 12; yrs = np.arange(2001, 2011)
# ---- figure 1: wetland area by band, monthly series and annual anomalies
fig, ax = plt.subplots(4, 2, figsize=(14, 13), gridspec_kw=dict(width_ratios=[2.4, 1]), constrained_layout=True)
for i, (b, msk) in enumerate(BANDS.items()):
    s = {'climatology': band_area(AC, msk), 'learned': band_area(AL, msk), 'observed': band_area(AO, msk)}
    a = ax[i, 0]
    for k in ('climatology', 'observed', 'learned'): a.plot(tm, s[k], color=COL[k], lw=2 if k != 'climatology' else 1.5, ls='-' if k != 'climatology' else '--', label={'observed': 'observed (WAD2M)', 'learned': 'learned', 'climatology': 'WAD2M climatology'}[k])
    a.set_title(f'{b}: wetland area, monthly', loc='left'); a.set_ylabel('million km²')
    if i == 0: a.legend(loc='upper left', ncol=3)
    a = ax[i, 1]
    for k in ('observed', 'learned'):
        y = s[k].reshape(10, 12).mean(1); a.plot(yrs, y - y.mean(), color=COL[k], lw=2, marker='o', ms=6, mec=SURF, mew=1.5, label=k)
    a.axhline(0, color=MUTED, lw=1); a.set_title('annual-mean anomaly', loc='left'); a.set_ylabel('million km²'); a.set_xticks(yrs[::3])
    r = np.corrcoef(s['observed'].reshape(10, 12).mean(1), s['learned'].reshape(10, 12).mean(1))[0, 1]
    a.text(0.02, 0.04, f'r = {r:.2f}', transform=a.transAxes, color=INK2)
    if i == 0: a.legend(loc='upper left')
fig.suptitle('Wetland area 2001–2010: observed, fixed climatology and learned', x=0.01, ha='left', fontsize=14, fontweight='bold', color=INK)
fig.savefig(f'{FIG}/wetland_area_series.png', dpi=110); plt.close(fig)
# ---- figure 2: maps
rows = LAT > -60; ext = [-180, 180, -60, 90]
def mp(a, X, title, **kw):
    im = a.imshow(np.where(land, X, np.nan)[rows], extent=ext, aspect='auto', interpolation='nearest', **kw); a.set_title(title, loc='left'); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for sp in a.spines.values(): sp.set_visible(False)
    a.set_facecolor('#eceae4'); return im
mo, ml = AO.mean(0), AL.mean(0); yo, yl = AO.reshape(10, 12, 360, 720).mean(1), AL.reshape(10, 12, 360, 720).mean(1)
so, sl = yo.std(0), yl.std(0)
ao, al = yo - yo.mean(0), yl - yl.mean(0); num = (ao * al).sum(0); den = np.sqrt((ao ** 2).sum(0) * (al ** 2).sum(0)); rmap = np.where((den > 1e-12) & (mo > 0.01), num / np.maximum(den, 1e-12), np.nan)
fig, ax = plt.subplots(2, 3, figsize=(18, 8.4), constrained_layout=True)
im = mp(ax[0, 0], np.where(mo > 0.002, mo, np.nan), 'observed wetland fraction, 2001–2010 mean', cmap='Blues', vmin=0, vmax=0.3); fig.colorbar(im, ax=ax[0, 0], shrink=0.8, label='fraction of cell')
im = mp(ax[0, 1], np.where((mo > 0.002) | (ml > 0.002), ml - mo, np.nan), 'learned minus observed, 2001–2010 mean', cmap='RdBu_r', norm=TwoSlopeNorm(0, -0.02, 0.02)); fig.colorbar(im, ax=ax[0, 1], shrink=0.8, label='fraction of cell')
im = mp(ax[0, 2], rmap, 'year-to-year correlation, learned vs observed', cmap='RdBu_r', norm=TwoSlopeNorm(0, -1, 1)); fig.colorbar(im, ax=ax[0, 2], shrink=0.8, label='r of annual means (cells > 1 % wet)')
vmax = float(np.nanpercentile(np.where(mo > 0.01, so, np.nan), 98))
im = mp(ax[1, 0], np.where(mo > 0.002, so, np.nan), 'observed: year-to-year sd of the annual mean', cmap='Oranges', vmin=0, vmax=vmax); fig.colorbar(im, ax=ax[1, 0], shrink=0.8, label='fraction of cell')
im = mp(ax[1, 1], np.where(ml > 0.002, sl, np.nan), 'learned: year-to-year sd of the annual mean', cmap='Oranges', vmin=0, vmax=vmax); fig.colorbar(im, ax=ax[1, 1], shrink=0.8, label='fraction of cell')
ratio = np.where(so > 1e-3, sl / np.maximum(so, 1e-6), np.nan)
im = mp(ax[1, 2], ratio, 'learned sd / observed sd', cmap='RdBu_r', norm=TwoSlopeNorm(1, 0, 2)); fig.colorbar(im, ax=ax[1, 2], shrink=0.8, label='ratio (cells with observed sd > 0.001)')
fig.suptitle('Wetland area in space: where the learned area follows the observations, and how strongly it varies', x=0.01, ha='left', fontsize=14, fontweight='bold', color=INK)
fig.savefig(f'{FIG}/wetland_area_maps.png', dpi=100); plt.close(fig)
# ---- figure 3: methane
LBL = {'climatology': 'climatology area', 'learned': 'learned area', 'observed': 'observed area', 'old': 'release before 29 Oct (snapshot)'}
fig, ax = plt.subplots(4, 3, figsize=(18, 13), gridspec_kw=dict(width_ratios=[2.2, 1, 1]), constrained_layout=True)
for i, (b, msk) in enumerate(BANDS.items()):
    s = {k: band_ch4(C, msk) for k, C in CH.items()}
    a = ax[i, 0]
    for k in ('climatology', 'observed', 'learned'): a.plot(tm, s[k], color=COL[k], lw=2 if k != 'climatology' else 1.5, ls='-' if k != 'climatology' else '--', label=LBL[k])
    a.set_title(f'{b}: wetland CH₄, monthly', loc='left'); a.set_ylabel('Tg CH₄ per month')
    if i == 0: a.legend(loc='upper left', ncol=3)
    a = ax[i, 1]; g = GMB[b]; a.axhspan(g[1], g[2], color='#eceae4', lw=0); a.axhline(g[0], color=INK2, lw=1.2)
    a.text(2001.1, g[0], ' GMB 2025 mean', va='bottom', color=INK2, fontsize=9)
    for k in ('climatology', 'observed', 'learned', 'old'):
        y = s[k].reshape(10, 12).sum(1); a.plot(yrs, y, color=COL[k], lw=2 if k != 'old' else 1.5, ls='-' if k not in ('climatology', 'old') else '--', marker='o' if k in ('learned', 'observed') else None, ms=5, mec=SURF, label=LBL[k])
    a.set_title('annual total (band = GMB range)', loc='left'); a.set_ylabel('Tg CH₄ per year'); a.set_xticks(yrs[::3])
    if i == 0: a.legend(loc='lower left', fontsize=9)
    a = ax[i, 2]
    for k in ('climatology', 'observed', 'learned', 'old'):
        a.plot(np.arange(1, 13), s[k].reshape(10, 12).mean(0), color=COL[k], lw=2 if k != 'old' else 1.5, ls='-' if k not in ('climatology', 'old') else '--', label=LBL[k])
    a.set_xticks(range(1, 13)); a.set_xticklabels(list('JFMAMJJASOND')); a.set_title('seasonal cycle, decade mean', loc='left'); a.set_ylabel('Tg CH₄ per month')
fig.suptitle('Wetland methane 2001–2010 under the three area modes (hourly flux, refitted bands), and the release before today', x=0.01, ha='left', fontsize=14, fontweight='bold', color=INK)
fig.savefig(f'{FIG}/wetland_ch4_series.png', dpi=110); plt.close(fig)
print('wrote', FIG)
