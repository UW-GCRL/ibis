"""Soil carbon of the protocol states against HWSD2 0-1 m (orgcpct x bulk density x layer thickness, layers 1-4 of
soita_hwsd2_organic.nc, precomputed to hwsd2_soc1m.npy by the gpu venv): maps of the 2010 soil C (totcsoi) of the first
and last run and of HWSD2, the last run minus HWSD2, the model / HWSD2 ratio by Köppen group and latitude band for every
run, and the zonal land mean.  Totals on the HWSD2-covered cells.
    python -I plot_soil.py OUT.png runA runB [runC ...]     (glob_<run>.npz of the meeting folder; first = reference, last = final)
Prints the ratio table and saves metrics_soil_<runs>.json next to the grids."""
import sys, re, ast, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
M = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad/meeting'
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open('/shared/nas2/Hangkai/ibis-port/python-port/ibis/io/pft_types.py').read()).group(1))
LBL = {'s31n': 's31n · stage final, before the soil stage', 's31r': 's31r · + zone cover limit', 's31t': 's31t · final: strong soil factors', 's31u': 's31u · mild soil factors'}
COL = {'s31n': '#2F6FB0', 's31r': '#7A8699', 's31t': '#D9822B', 's31u': '#E9A35B'}
out, runs = sys.argv[1], sys.argv[2:]
ref = dict(np.load(f'{M}/glob_refs.npz')); lat, lon, lf, Z = ref['lat'], ref['lon'], ref['lf'], ref['zone']
D = {r: np.asarray(np.load(f'{M}/glob_{r}.npz')['csoil'], float) for r in runs}
H = np.load(f'{M}/hwsd2_soc1m.npy')                                       # kgC m-2, 0-1 m, lat N->S as the meeting grids
land = (lf > 0.05) & (lat[:, None] > -60); AREA = 6371e3 ** 2 * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((360, 720)) * lf
cm = land & (H > 0.1)                                                     # HWSD2-covered cells
ext = [-180, 180, -60, 90]; rows60 = lat > -60
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 15, 'axes.titleweight': 'bold'})
def m(x): x = np.asarray(x, float); return np.where(land, x, np.nan)[rows60]
def tot(x, mk): x = np.where(mk & np.isfinite(x), x, 0); return float((x * AREA).sum() / 1e12)
def wmean(x, mk): ok = mk & np.isfinite(x); return float((x[ok] * AREA[ok]).sum() / AREA[ok].sum())
def rcorr(x, y, mk):
    ok = mk & np.isfinite(x) & np.isfinite(y); w = AREA[ok]; xm, ym = (x[ok] * w).sum() / w.sum(), (y[ok] * w).sum() / w.sum()
    return float(((x[ok] - xm) * (y[ok] - ym) * w).sum() / np.sqrt(((x[ok] - xm) ** 2 * w).sum() * ((y[ok] - ym) ** 2 * w).sum()))
def mapax(ax, x, title, **kw):
    im = ax.imshow(m(x), extent=ext, aspect='auto', **kw); ax.set_title(title, loc='left'); ax.set_facecolor('#E9ECEF'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    return im
def zrow(x):
    o = np.full(360, np.nan)
    for i in range(360):
        ok = cm[i] & np.isfinite(x[i])
        if ok.sum() >= 5: o[i] = (x[i][ok] * AREA[i][ok]).sum() / AREA[i][ok].sum()
    return o
zn = lambda *nm: np.isin(Z, [ZONES.index(n) for n in nm])
GROUPS = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop / Med / highland': zn('C_med', 'C_subtrop', 'C_highland'),
          'temperate': zn('C_oceanic', 'D_temperate'), 'boreal': zn('D_subarctic', 'D_extreme'), 'tundra + cold dry': zn('E_tundra', 'B_cold')}
BANDS = {'90°S–30°N': (-90, 30), '30–60°N': (30, 60), '60–90°N': (60, 90)}
bm = lambda lo, hi: (lat[:, None] >= lo) & (lat[:, None] < hi) & np.ones((360, 720), bool)
ra, rb = runs[0], runs[-1]
fig, ax = plt.subplots(2, 3, figsize=(19.2, 8.6), constrained_layout=True)
im = mapax(ax[0, 0], D[ra], f'{LBL.get(ra, ra)}: {tot(D[ra], cm):.0f} PgC', cmap='YlOrBr', vmin=0, vmax=40)
mapax(ax[0, 1], D[rb], f'{LBL.get(rb, rb)}: {tot(D[rb], cm):.0f} PgC', cmap='YlOrBr', vmin=0, vmax=40)
mapax(ax[0, 2], np.where(cm, H, np.nan), f'HWSD2 0–1 m: {tot(H, cm):.0f} PgC', cmap='YlOrBr', vmin=0, vmax=40)
fig.colorbar(im, ax=ax[0, :], shrink=0.85, label='soil organic C, kgC m⁻²', pad=0.01)
im2 = mapax(ax[1, 0], np.where(cm, D[rb] - H, np.nan), f'{rb} − HWSD2', cmap='RdBu_r', norm=TwoSlopeNorm(0, -20, 20)); fig.colorbar(im2, ax=ax[1, 0], shrink=0.85, label='kgC m⁻²', pad=0.01)
a = ax[1, 1]; names = list(GROUPS) + list(BANDS) + ['global']; masks = [cm & g for g in GROUPS.values()] + [cm & bm(*b) for b in BANDS.values()] + [cm]
xx = np.arange(len(names)); wdt = 0.8 / len(runs); MET = {}
for i, r in enumerate(runs):
    vals = [tot(D[r], mk) / max(tot(H, mk), 1e-9) for mk in masks]; MET[r] = dict(zip([n.replace('\n', ' ') for n in names], vals))
    a.bar(xx + (i - (len(runs) - 1) / 2) * wdt, vals, wdt, color=COL.get(r, '0.5'), label=r)
SHORT = {'humid tropics': 'humid', 'savanna + hot dry': 'savanna', 'subtrop / Med / highland': 'subtrop', 'tundra + cold dry': 'tundra'}
a.axhline(1, color='#14213D', lw=1.2); a.set_xticks(xx); a.set_xticklabels([SHORT.get(n, n) for n in names], fontsize=11, rotation=35, ha='right'); a.set_ylabel('model / HWSD2 0–1 m'); a.legend(frameon=False, fontsize=12)
a.grid(axis='y', alpha=0.3); a.set_title('soil C ratio by Köppen group and band', loc='left'); a.set_ylim(0, 1.3)
z = ax[1, 2]; z.plot(zrow(H), lat, color='#14213D', lw=2.4, label='HWSD2 0–1 m')
for r in runs: z.plot(zrow(D[r]), lat, color=COL.get(r, '0.5'), lw=2, ls='--' if r == rb else '-', label=r)
z.set_ylim(-60, 90); z.set_xlabel('zonal land mean, kgC m⁻²'); z.legend(frameon=False, fontsize=12); z.grid(alpha=0.3); z.set_title('by latitude', loc='left')
fig.savefig(out, dpi=100); plt.close(fig); print('wrote', out)
print(f"soil C 2010 (totcsoi) vs HWSD2 0-1 m on HWSD2-covered cells ({100 * AREA[cm].sum() / AREA[land].sum():.1f} % of land area); HWSD2 {tot(H, cm):.0f} PgC, {wmean(H, cm):.1f} kgC/m2")
print(f"{'chunk':26s} {'HWSD2':>7s} | " + ' '.join(f'{r:>14s}' for r in runs))
for n, mk in zip(names, masks):
    print(f"{n.replace(chr(10), ' '):26s} {tot(H, mk):7.1f} | " + ' '.join(f'{tot(D[r], mk):7.1f} ({tot(D[r], mk) / max(tot(H, mk), 1e-9):4.2f})' for r in runs))
for r in runs: MET[r]['total_PgC'] = tot(D[r], cm); MET[r]['mean_kgC_m2'] = wmean(D[r], cm); MET[r]['r'] = rcorr(D[r], H, cm)
MET['hwsd2'] = dict(total_PgC=tot(H, cm), mean_kgC_m2=wmean(H, cm))
print('area-weighted r vs HWSD2: ' + ', '.join(f'{r} {MET[r]["r"]:.2f}' for r in runs) + '; mean kgC/m2: ' + ', '.join(f'{r} {MET[r]["mean_kgC_m2"]:.1f}' for r in runs))
json.dump(MET, open(f'{M}/metrics_soil_{"_".join(runs)}.json', 'w'), indent=1, default=float); print('metrics saved')
