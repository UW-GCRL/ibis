"""Meeting figures for the global runs. python plot_global.py runA runB  (e.g. s31f s31g)."""
import sys, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
M = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad/meeting'; FIG = '/shared/nas2/Hangkai/ibis_local_document/figures/final_2026-10-26'
RA, RB = sys.argv[1], sys.argv[2]
LBL = {'s31e': 's31e (before stage 54)', 's31f': 's31f · road A, caps', 's31g': 's31g · road B, allocation', 's31j': 's31j · road A, caps + fixes', 's31k': 's31k · road B, allocation + fixes', 's31l': 's31l · road A, final (fixes + fire rule)', 's31m': 's31m · road B, final (fixes + fire rule)', 's31n': 's31n · road B, cold shrub aleaf 0.30', 's31o': 's31o · road B, cold shrub aleaf 0.25', 's31p': 's31p · road B + cover limit', 's31q': 's31q · road B + cover limit + tropical wood refit', 's31r': 's31r · road B + zone cover limit + production refit', 's31t': 's31t · final: s31r vegetation + strong soil factors', 's31u': 's31u · s31r vegetation + mild soil factors'}
C = {'ref': '#14213D', RA: '#2F6FB0', RB: '#D9822B', 's31e': '#7A8699'}
ref = dict(np.load(f'{M}/glob_refs.npz')); A_ = dict(np.load(f'{M}/glob_{RA}.npz')); B_ = dict(np.load(f'{M}/glob_{RB}.npz'))
E_ = dict(np.load(f'{M}/glob_s31e.npz')) if RA != 's31e' else None
lat, lon, lf = ref['lat'], ref['lon'], ref['lf']; land = (lf > 0.05) & (lat[:, None] > -60)
AREA = 6371e3 ** 2 * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((360, 720)) * lf
ext = [-180, 180, -60, 90]; rows60 = lat > -60
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 15, 'axes.titleweight': 'bold'})
def m(x): x = np.where(np.abs(x) > 1e5, np.nan, np.asarray(x, float)); return np.where(land, x, np.nan)[rows60]
def tot(x, s=1e15, mk=None): mk = land if mk is None else mk; x = np.where(mk & np.isfinite(x) & (np.abs(x) < 1e5), x, 0); return float((x * AREA).sum() / s)
def wmean(x, mk=None): mk = land if mk is None else mk; ok = mk & np.isfinite(x) & (np.abs(x) < 1e5); return float((x[ok] * AREA[ok]).sum() / AREA[ok].sum())
def rcorr(x, y): ok = land & np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e5) & (np.abs(y) < 1e5); w = AREA[ok]; xm, ym = (x[ok] * w).sum() / w.sum(), (y[ok] * w).sum() / w.sum(); return float(((x[ok] - xm) * (y[ok] - ym) * w).sum() / np.sqrt(((x[ok] - xm) ** 2 * w).sum() * ((y[ok] - ym) ** 2 * w).sum()))
def rmse(x, y): ok = land & np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e5) & (np.abs(y) < 1e5); return float(np.sqrt(((x[ok] - y[ok]) ** 2 * AREA[ok]).sum() / AREA[ok].sum()))
def band(x, lo, hi, s=1e15, mk=None): mk = land if mk is None else mk; b = mk & (lat[:, None] >= lo) & (lat[:, None] < hi); x = np.where(b & np.isfinite(x) & (np.abs(x) < 1e5), x, 0); return float((x * AREA).sum() / s)
def zon(x, per_deg=True):
    x = np.where(land & np.isfinite(x) & (np.abs(x) < 1e5), x, 0); return (x * AREA).sum(1) / 1e15 * 2
def mapax(ax, x, title, **kw):
    im = ax.imshow(m(x), extent=ext, aspect='auto', **kw); ax.set_title(title, loc='left'); ax.set_facecolor('#E9ECEF'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    return im
METRICS = {}
def triple(var, refkey, title, unit, cmap, vmax, dmax, fname, scale=1e15, ref_label='reference', total_unit='PgC/yr', meanmode=False):
    a, b, r = A_[var], B_[var], ref[refkey]
    fig, ax = plt.subplots(2, 3, figsize=(19.2, 8.6), constrained_layout=True)
    cm = land & np.isfinite(r) & (np.abs(r) < 1e5)                       # totals on the cells the reference covers (as the scoring does)
    T = (lambda x: wmean(x, cm)) if meanmode else (lambda x: tot(x, scale, cm))
    im = mapax(ax[0, 0], a, f'{LBL[RA]}: {T(a):.0f} {total_unit}', cmap=cmap, vmin=0, vmax=vmax); mapax(ax[0, 1], b, f'{LBL[RB]}: {T(b):.0f} {total_unit}', cmap=cmap, vmin=0, vmax=vmax)
    mapax(ax[0, 2], r, f'{ref_label}: {T(r):.0f} {total_unit}', cmap=cmap, vmin=0, vmax=vmax); fig.colorbar(im, ax=ax[0, :], shrink=0.85, label=unit, pad=0.01)
    im2 = mapax(ax[1, 0], a - r, f'{RA} − reference', cmap='RdBu_r', norm=TwoSlopeNorm(0, -dmax, dmax)); mapax(ax[1, 1], b - r, f'{RB} − reference', cmap='RdBu_r', norm=TwoSlopeNorm(0, -dmax, dmax))
    fig.colorbar(im2, ax=ax[1, :2], shrink=0.85, label=unit, pad=0.01)
    z = ax[1, 2]
    if meanmode:
        zz = lambda x: np.array([np.nansum(np.where(land[i] & np.isfinite(x[i]), x[i] * AREA[i], 0)) / max(np.nansum(np.where(land[i] & np.isfinite(x[i]), AREA[i], 0)), 1) for i in range(360)])
        lab = f'zonal land mean, {unit}'
    else: zz = zon; lab = f'zonal total, {total_unit} per degree'
    z.plot(zz(r), lat, color=C['ref'], lw=2.4, label='reference'); z.plot(zz(a), lat, color=C[RA], lw=2, label=RA); z.plot(zz(b), lat, color=C[RB], lw=2, ls='--', label=RB)
    z.set_ylim(-60, 90); z.set_xlabel(lab); z.legend(frameon=False, fontsize=13); z.grid(alpha=0.3); z.set_title('by latitude', loc='left')
    fig.savefig(f'{FIG}/{fname}.png', dpi=100); plt.close(fig)
    rows = {}
    for nm, x in ((RA, a), (RB, b)):
        bd = (lambda lo, hi: band(x, lo, hi, scale, cm) / band(r, lo, hi, scale, cm)) if not meanmode else (lambda lo, hi: wmean(x, cm & (lat[:, None] >= lo) & (lat[:, None] < hi)) / wmean(r, cm & (lat[:, None] >= lo) & (lat[:, None] < hi)))
        rows[nm] = dict(total=T(x), ratio=T(x) / T(r), r=rcorr(x, r), rmse=rmse(x, r), trop=bd(-30, 30), mid=bd(30, 60), high=bd(60, 90))
    rows['ref'] = dict(total=T(r)); METRICS[var] = rows; print('wrote', fname)
triple('gpp', 'gpp', 'GPP', 'gC m⁻² yr⁻¹', 'viridis', 3500, 1500, 'gpp_maps', ref_label='FLUXCOM RS+METEO')
triple('et', 'et', 'ET', 'mm yr⁻¹', 'YlGnBu', 1400, 500, 'et_maps', ref_label='ET ensemble (5 products)', total_unit='mm/yr', meanmode=True)
# --- GPP by latitude and season
fig = plt.figure(figsize=(19.2, 8.2), constrained_layout=True); gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1, 1])
z = fig.add_subplot(gs[:, 0])
for key, src, ls in (('ref', ref, '-'), (RA, A_, '-'), (RB, B_, '--')): z.plot(zon(src['gpp']), lat, color=C[key], lw=2.4 if key == 'ref' else 2, ls=ls, label='FLUXCOM' if key == 'ref' else LBL[key])
if E_ is not None: z.plot(zon(E_['gpp']), lat, color=C['s31e'], lw=1.5, ls=':', label=LBL['s31e'])
z.set_ylim(-60, 90); z.set_xlabel('GPP, PgC yr⁻¹ per degree latitude'); z.legend(frameon=False, fontsize=13, loc='upper right'); z.grid(alpha=0.3); z.set_title('GPP by latitude', loc='left')
MON = 'JFMAMJJASOND'
for i, (lo, hi, nm) in enumerate(((60, 90, '60–90 °N'), (30, 60, '30–60 °N'), (-30, 30, '30 °S–30 °N'), (-60, -30, '60–30 °S'))):
    ax = fig.add_subplot(gs[i // 2, 1 + i % 2]); b = land & (lat[:, None] >= lo) & (lat[:, None] < hi)
    f = lambda gm: np.array([np.nansum(np.where(b & np.isfinite(gm[k]) & (np.abs(gm[k]) < 1e5), gm[k], 0) * AREA) / 1e15 for k in range(12)])
    ax.plot(range(12), f(ref['gpp_m']), color=C['ref'], lw=2.4, label='FLUXCOM'); ax.plot(range(12), f(A_['gpp_m']), color=C[RA], lw=2, label=RA); ax.plot(range(12), f(B_['gpp_m']), color=C[RB], lw=2, ls='--', label=RB)
    ax.set_xticks(range(12)); ax.set_xticklabels(list(MON)); ax.set_title(f'{nm}: monthly GPP, PgC', loc='left'); ax.grid(alpha=0.3)
    if i == 0: ax.legend(frameon=False, fontsize=12)
    METRICS.setdefault('season', {})[nm] = {k: (f(src['gpp_m']).tolist()) for k, src in (('ref', ref), (RA, A_), (RB, B_))}
fig.savefig(f'{FIG}/gpp_zonal_season.png', dpi=100); plt.close(fig); print('wrote gpp_zonal_season')
# --- NEE (+ = uptake)
fig, ax = plt.subplots(2, 3, figsize=(19.2, 8.6), constrained_layout=True); nm = TwoSlopeNorm(0, -300, 300)
im = mapax(ax[0, 0], A_['nee'], f'{LBL[RA]}: NPP − Rh {tot(A_["nee"]):.1f} PgC/yr', cmap='BrBG', norm=nm); mapax(ax[0, 1], B_['nee'], f'{LBL[RB]}: NPP − Rh {tot(B_["nee"]):.1f} PgC/yr', cmap='BrBG', norm=nm)
mapax(ax[0, 2], ref['nee_gcb'], f'GCB2024 DGVM mean S_LAND 2014–23: {tot(ref["nee_gcb"]):.1f}', cmap='BrBG', norm=nm); fig.colorbar(im, ax=ax[0, :], shrink=0.85, label='NEE, gC m⁻² yr⁻¹ (+ = uptake)', pad=0.01)
mapax(ax[1, 0], ref['nee_fluxcom'], f'FLUXCOM NEE 2006–10 (+ uptake): {tot(ref["nee_fluxcom"]):.1f}', cmap='BrBG', norm=nm)
mapax(ax[1, 1], B_['nee'] - A_['nee'], f'{RB} − {RA}', cmap='RdBu_r', norm=TwoSlopeNorm(0, -150, 150))
z = ax[1, 2]; z.plot(zon(ref['nee_gcb']), lat, color=C['ref'], lw=2.4, label='GCB DGVM mean'); z.plot(zon(ref['nee_fluxcom']), lat, color='0.55', lw=1.6, label='FLUXCOM'); z.plot(zon(A_['nee']), lat, color=C[RA], lw=2, label=RA); z.plot(zon(B_['nee']), lat, color=C[RB], lw=2, ls='--', label=RB)
z.axvline(0, color='0.6', lw=0.8); z.set_ylim(-60, 90); z.set_xlabel('NEE, PgC yr⁻¹ per degree (+ uptake)'); z.legend(frameon=False, fontsize=12); z.grid(alpha=0.3); z.set_title('by latitude', loc='left')
fig.savefig(f'{FIG}/nee_maps.png', dpi=100); plt.close(fig); print('wrote nee_maps')
METRICS['nee'] = {RA: dict(total=tot(A_['nee']), npp=tot(A_['npp']), rh=tot(A_['rh'])), RB: dict(total=tot(B_['nee']), npp=tot(B_['npp']), rh=tot(B_['rh'])), 'gcb_pattern': tot(ref['nee_gcb']), 'fluxcom': tot(ref['nee_fluxcom'])}
# --- LAI and wood
fig, ax = plt.subplots(2, 3, figsize=(19.2, 8.6), constrained_layout=True)
im = mapax(ax[0, 0], A_['lai_peak'], f'{LBL[RA]}: peak LAI', cmap='Greens', vmin=0, vmax=7); mapax(ax[0, 1], B_['lai_peak'], f'{LBL[RB]}: peak LAI', cmap='Greens', vmin=0, vmax=7)
mapax(ax[0, 2], ref['lai_peak'], 'GLASS / GIMMS LAI4g mean, peak 2006–10', cmap='Greens', vmin=0, vmax=7); fig.colorbar(im, ax=ax[0, :], shrink=0.85, label='peak LAI, m² m⁻²', pad=0.01)
im = mapax(ax[1, 0], A_['wood'], f'{RA}: wood C {tot(A_["wood"], 1e12):.0f} PgC', cmap='YlGn', vmin=0, vmax=20); mapax(ax[1, 1], B_['wood'], f'{RB}: wood C {tot(B_["wood"], 1e12):.0f} PgC', cmap='YlGn', vmin=0, vmax=20)
mapax(ax[1, 2], ref['wood'], f'Xu & Saatchi 2021 woody C 2006–10: {tot(ref["wood"], 1e12):.0f} PgC', cmap='YlGn', vmin=0, vmax=20); fig.colorbar(im, ax=ax[1, :], shrink=0.85, label='kgC m⁻²', pad=0.01)
fig.savefig(f'{FIG}/lai_biomass.png', dpi=100); plt.close(fig); print('wrote lai_biomass')
METRICS['lai'] = {k: dict(mean=wmean(s['lai_peak']), ratio=wmean(s['lai_peak']) / wmean(ref['lai_peak']), r=rcorr(s['lai_peak'], ref['lai_peak'])) for k, s in ((RA, A_), (RB, B_))}; METRICS['lai']['ref'] = wmean(ref['lai_peak'])
METRICS['wood'] = {k: dict(total=tot(s['wood'], 1e12), r=rcorr(s['wood'], ref['wood'])) for k, s in ((RA, A_), (RB, B_))}; METRICS['wood']['ref'] = tot(ref['wood'], 1e12)
# --- tree cover and deciduous share by zone
ZN = ['A_humid', 'A_savanna', 'B_hot', 'B_cold', 'C_med', 'C_subtrop', 'C_highland', 'C_oceanic', 'D_temperate', 'D_subarctic', 'E_tundra']
ZID = {'A_humid': 0, 'A_savanna': 1, 'B_hot': 2, 'B_cold': 3, 'C_med': 4, 'C_subtrop': 5, 'C_highland': 6, 'C_oceanic': 7, 'D_temperate': 8, 'D_subarctic': 9, 'D_extreme': 10, 'E_tundra': 11}
sys.path.insert(0, '/shared/nas2/Hangkai/ibis-port/python-port')
nat = land & np.isfinite(ref['tree']) & (ref['crop'] < 0.5); W = lambda x, mk: float(np.nansum(np.where(mk, x, 0) * AREA) / max(np.nansum(np.where(mk, AREA, 0)), 1))
cov = lambda s: np.clip(s['fu'] / np.maximum(1 - ref['crop'], 0.01), 0, 1)
fig, ax = plt.subplots(1, 2, figsize=(19.2, 7.4), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.6, 1]))
xx = np.arange(len(ZN)); wdt = 0.2; srcs = [('observed (CLM / MODIS)', ref['tree'], C['ref'])] + ([('s31e', cov(E_), C['s31e'])] if E_ is not None else []) + [(RA, cov(A_), C[RA]), (RB, cov(B_), C[RB])]
CZ = {}
for i, (nm_, x, col) in enumerate(srcs):
    vals = [W(x, nat & (ref['zone'] == ZID[z])) for z in ZN]; CZ[nm_] = vals
    ax[0].bar(xx + (i - (len(srcs) - 1) / 2) * wdt, vals, wdt, color=col, label=nm_)
ax[0].set_xticks(xx); ax[0].set_xticklabels([z.replace('_', '\n') for z in ZN], fontsize=12); ax[0].set_ylabel('tree cover on natural land'); ax[0].legend(frameon=False, ncol=4, fontsize=12); ax[0].set_title('Tree cover by Köppen zone', loc='left'); ax[0].grid(axis='y', alpha=0.3)
ZD = ['C_subtrop', 'C_oceanic', 'D_temperate', 'D_subarctic']; xd = np.arange(len(ZD)); fo = nat & (ref['tree'] > 0.6) & (lat[:, None] >= 30)
dsrc = [('observed', ref['tree_dec'], ref['tree'] - ref['tree_dec'], C['ref'])] + ([('s31e', E_['wood_dec'], E_['wood_ev'], C['s31e'])] if E_ is not None else []) + [(RA, A_['wood_dec'], A_['wood_ev'], C[RA]), (RB, B_['wood_dec'], B_['wood_ev'], C[RB])]
DZ = {}
for i, (nm_, d, e, col) in enumerate(dsrc):
    vals = [W(d, fo & (ref['zone'] == ZID[z])) / max(W(d + e, fo & (ref['zone'] == ZID[z])), 1e-9) for z in ZD]; DZ[nm_] = vals
    ax[1].bar(xd + (i - (len(dsrc) - 1) / 2) * wdt, vals, wdt, color=col, label=nm_)
ax[1].set_xticks(xd); ax[1].set_xticklabels([z.replace('_', '\n') for z in ZD], fontsize=12); ax[1].set_ylabel('deciduous share of forest'); ax[1].set_title('Deciduous share of observed forest, 30–90 °N', loc='left'); ax[1].grid(axis='y', alpha=0.3)
fig.savefig(f'{FIG}/cover_zone.png', dpi=100); plt.close(fig); print('wrote cover_zone')
METRICS['cover'] = dict(zones=ZN, vals=CZ, natural={k: W(x, nat) for k, x, _ in srcs}); METRICS['decid'] = dict(zones=ZD, vals=DZ)
# --- methane
fig = plt.figure(figsize=(19.2, 8.6), constrained_layout=True); gs = fig.add_gridspec(2, 3)
for j, (key, s) in enumerate(((RA, A_), (RB, B_))):
    ax = fig.add_subplot(gs[0, j]); im = mapax(ax, np.where(s['ch4'] > 0.01, s['ch4'], np.nan), f'{LBL[key]}: {tot(s["ch4"], 1e12):.0f} Tg CH₄/yr', cmap='magma_r', vmin=0, vmax=30)
fig.colorbar(im, ax=[fig.axes[0], fig.axes[1]], shrink=0.85, label='g CH₄ m⁻² yr⁻¹', pad=0.01)
ax = fig.add_subplot(gs[0, 2]); im = mapax(ax, np.where(ref['wad'] > 0.005, ref['wad'] * 100, np.nan), 'WAD2M wetland area 2006–10 (prescribed)', cmap='Blues', vmin=0, vmax=40); fig.colorbar(im, ax=ax, shrink=0.85, label='% of cell', pad=0.01)
ax = fig.add_subplot(gs[1, :2]); zc = lambda s: np.array([np.nansum(np.where(land[i] & np.isfinite(s['ch4'][i]), s['ch4'][i] * AREA[i], 0)) for i in range(360)]) / 1e12
zA, zB = zc(A_), zc(B_); lb = lat.reshape(-1, 10).mean(1); ax.bar(lb - 1.1, zA.reshape(-1, 10).sum(1), 2.2, color=C[RA], label=RA); ax.bar(lb + 1.1, zB.reshape(-1, 10).sum(1), 2.2, color=C[RB], label=RB)
ax.set_xlim(-60, 85); ax.set_xlabel('latitude, 5-degree bands'); ax.set_ylabel('Tg CH₄ yr⁻¹'); ax.legend(frameon=False); ax.grid(axis='y', alpha=0.3); ax.set_title('wetland CH₄ by latitude', loc='left')
ax = fig.add_subplot(gs[1, 2]); G = [('90°S–30°N', 109, 66, 146), ('30–60°N', 32, 16, 60), ('60–90°N', 12, 4, 28)]; xb = np.arange(3)
bandv = lambda s: [band(s['ch4'], -90, 30, 1e12), band(s['ch4'], 30, 60, 1e12), band(s['ch4'], 60, 90, 1e12)]
ax.bar(xb - 0.27, [g[1] for g in G], 0.27, color=C['ref'], label='GMB 2025 bottom-up'); ax.errorbar(xb - 0.27, [g[1] for g in G], yerr=[[g[1] - g[2] for g in G], [g[3] - g[1] for g in G]], fmt='none', ecolor='0.4', capsize=5)
ax.bar(xb, bandv(A_), 0.27, color=C[RA], label=RA); ax.bar(xb + 0.27, bandv(B_), 0.27, color=C[RB], label=RB); ax.set_xticks(xb); ax.set_xticklabels([g[0] for g in G]); ax.set_ylabel('Tg CH₄ yr⁻¹'); ax.legend(frameon=False, fontsize=12); ax.set_title('by band vs GMB 2025', loc='left'); ax.grid(axis='y', alpha=0.3)
fig.savefig(f'{FIG}/ch4_global.png', dpi=100); plt.close(fig); print('wrote ch4_global')
METRICS['ch4'] = {RA: dict(total=tot(A_['ch4'], 1e12), bands=bandv(A_)), RB: dict(total=tot(B_['ch4'], 1e12), bands=bandv(B_))}
# --- fire: burned fraction (model disturbf from the 2010 state; GFED5 2001-05)
fig = plt.figure(figsize=(19.2, 8.6), constrained_layout=True); gs = fig.add_gridspec(2, 2)
for j, (key, s) in enumerate(((RA, A_), (RB, B_))):
    ax = fig.add_subplot(gs[0, j]); bf = s['disturbf'] * 100; im = mapax(ax, bf, f'{LBL[key]}: {wmean(s["disturbf"]) * 100:.2f} % of land per year', cmap='YlOrRd', vmin=0, vmax=20)
ax = fig.add_subplot(gs[1, 0]); mapax(ax, ref['burn_gfed5'] * 100, f'GFED5 2001–05: {wmean(ref["burn_gfed5"]) * 100:.2f} % of land per year', cmap='YlOrRd', vmin=0, vmax=20); fig.colorbar(im, ax=fig.axes, shrink=0.6, label='burned area, % of cell per year', pad=0.01)
ax = fig.add_subplot(gs[1, 1]); zz = lambda x: np.array([np.nansum(np.where(land[i] & np.isfinite(x[i]), x[i] * AREA[i], 0)) for i in range(360)]) / 1e10 * 2
ax.plot(lat, zz(ref['burn_gfed5']), color=C['ref'], lw=2.4, label='GFED5'); ax.plot(lat, zz(A_['disturbf']), color=C[RA], lw=2, label=RA); ax.plot(lat, zz(B_['disturbf']), color=C[RB], lw=2, ls='--', label=RB)
ax.set_xlim(-45, 75); ax.set_xlabel('latitude'); ax.set_ylabel('burned area, Mha yr⁻¹ per degree'); ax.legend(frameon=False); ax.grid(alpha=0.3); ax.set_title('by latitude', loc='left')
fig.savefig(f'{FIG}/fire.png', dpi=100); plt.close(fig); print('wrote fire')
from scipy.stats import spearmanr
def rk(x, y): ok = land & np.isfinite(x) & np.isfinite(y); return float(spearmanr(x[ok], y[ok]).statistic)
METRICS['fire'] = {k: dict(frac=wmean(s['disturbf']), ratio=wmean(s['disturbf']) / wmean(ref['burn_gfed5']), rank=rk(s['disturbf'], ref['burn_gfed5'])) for k, s in ((RA, A_), (RB, B_))}; METRICS['fire']['gfed5'] = wmean(ref['burn_gfed5'])
json.dump(METRICS, open(f'{M}/metrics_{RA}_{RB}.json', 'w'), indent=1, default=float); print('metrics saved')
