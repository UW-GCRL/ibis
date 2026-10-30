"""South America, final state: burned fraction and wood carbon of a reference run and the final run next to GFED5 (2001-05)
and Xu & Saatchi 2021, plus the humid-Amazon totals across the runs since the 12 Oct release. Model fire: disturbf of
the 2010 state; wood: upper-canopy cbiow of the 2010 state (amazon/<run>.npz of the nee_cycle folder, amazon_extract.py).
    python -I plot_amazon_final.py OUT.png RUN_A RUN_B"""
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
N = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work/nee_cycle'
S = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work'
out, RA, RB = sys.argv[1], sys.argv[2], sys.argv[3]
ref = np.load(f'{S}/meeting/glob_refs.npz'); lat, lon = ref['lat'], ref['lon']; LO, LA = np.meshgrid(lon, lat)
runs = {'s7_crop': '12 Oct release', 's31l': 'road A final', 's31n': 'road B stage (s31n)', 's31p': '+ uniform limit', 's31r': '+ zone limit, refit', 's31t': 'final (s31t)'}
LBL = {'s7_crop': '12 Oct release', 's31l': 'road A final (s31l)', 's31n': 'stage final (s31n)', 's31r': 's31r', 's31t': 'final (s31t)'}
COL = {'s7_crop': '#9AA6B8', 's31l': '#9AA6B8', 's31n': '#2F6FB0', 's31p': '#9AA6B8', 's31r': '#9AA6B8', 's31t': '#D9822B'}
D = {r: np.load(f'{N}/amazon/{r}.npz') for r in list(runs) + [x for x in (RA, RB) if x not in runs]}
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(LA)) * np.nan_to_num(ref['lf'])
hum = (LO > -82) & (LO < -34) & (LA > -20) & (LA < 13) & (ref['zone'] == 0)
def mean(x): ok = hum & np.isfinite(x); return float((x[ok] * area[ok]).sum() / area[ok].sum())
def tot(x): ok = hum & np.isfinite(x); return float((x[ok] * area[ok]).sum()) / 1e12
x0, x1, y0, y1 = -82, -34, -35, 13
jj = (lon >= x0) & (lon <= x1); ii = (lat >= y0) & (lat <= y1)
ext = [lon[jj].min() - 0.25, lon[jj].max() + 0.25, lat[ii].min() - 0.25, lat[ii].max() + 0.25]
sub = lambda a: a[np.ix_(ii, jj)]
land = sub(np.isfinite(ref['lf']) & (np.nan_to_num(ref['lf']) > 0))
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.titleweight': 'bold'})
fig = plt.figure(figsize=(16.64, 6.9), dpi=100, facecolor='#F7F6F2')
gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 0.07, 0.62, 1.3], left=0.005, right=0.995, top=0.94, bottom=0.08, wspace=0.06, hspace=0.18)
hum_s = sub(hum)
def mp(r, c, a, title, cmap, vmax):
    ax = fig.add_subplot(gs[r, c]); a = np.where(land, a, np.nan)
    im = ax.imshow(a, extent=ext, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest', origin='upper' if lat[0] > lat[-1] else 'lower')
    ax.contour(np.linspace(ext[0], ext[1], hum_s.shape[1]), np.linspace(ext[3], ext[2], hum_s.shape[0]) if lat[0] > lat[-1] else np.linspace(ext[2], ext[3], hum_s.shape[0]),
               hum_s.astype(float), levels=[0.5], colors='#14213D', linewidths=0.9)
    ax.set_facecolor('#E8EBEF'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, loc='left', fontsize=14)
    for s in ax.spines.values(): s.set_visible(False)
    return im
la, lb = LBL.get(RA, RA), LBL.get(RB, RB)
im1 = mp(0, 0, sub(D[RA]['disturbf']) * 100, f'{la} · burned', 'YlOrRd', 20)
mp(0, 1, sub(D[RB]['disturbf']) * 100, f'{lb} · burned', 'YlOrRd', 20)
mp(0, 2, sub(ref['burn_gfed5']) * 100, 'GFED5 · burned', 'YlOrRd', 20)
im2 = mp(1, 0, sub(D[RA]['wood']), f'{la} · wood C', 'YlGn', 20)
mp(1, 1, sub(D[RB]['wood']), f'{lb} · wood C', 'YlGn', 20)
mp(1, 2, sub(ref['wood']), 'Xu & Saatchi · wood C', 'YlGn', 20)
cb = fig.add_subplot(gs[0, 3]); fig.colorbar(im1, cax=cb); cb.set_ylabel('% of cell per year')
cb = fig.add_subplot(gs[1, 3]); fig.colorbar(im2, cax=cb); cb.set_ylabel('kgC m⁻²')
labs = list(runs.values()) + ['observed']; cols = [COL[r] for r in runs] + ['#14213D']
burn = [100 * mean(D[r]['disturbf']) for r in runs] + [100 * mean(ref['burn_gfed5'])]
wood = [tot(D[r]['wood']) for r in runs] + [tot(ref['wood'])]
ax = fig.add_subplot(gs[0, 5]); y = np.arange(len(labs))[::-1]
ax.barh(y, burn, color=cols); ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(0, 12.5)
for yi, v in zip(y, burn): ax.text(v + 0.15, yi, f'{v:.1f}', va='center', fontsize=13)
ax.set_title('Humid Amazon: burned area, % per year', loc='left'); ax.spines[['top', 'right']].set_visible(False); ax.set_facecolor('#F7F6F2')
ax = fig.add_subplot(gs[1, 5])
ax.barh(y, wood, color=cols); ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(0, 80)
for yi, v in zip(y, wood): ax.text(v + 0.8, yi, f'{v:.0f}', va='center', fontsize=13)
ax.set_title('Humid Amazon: wood carbon, PgC', loc='left'); ax.spines[['top', 'right']].set_visible(False); ax.set_facecolor('#F7F6F2')
fig.savefig(out, dpi=100, facecolor='#F7F6F2'); print('wrote', out); print(dict(zip(labs, [round(b, 2) for b in burn])), dict(zip(labs, [round(w, 1) for w in wood])))
