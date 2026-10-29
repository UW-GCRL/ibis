"""Wood carbon by world chunk: the tree wood of the 2010 state (cbiow, upper canopy) against Xu & Saatchi 2021 live biomass
carbon (2006-2010 mean), by Köppen zone group x continent. Totals (PgC), model/map ratio, cell rank correlation; a figure.
    python wood_regions.py OUT.png RUN_A RUN_B [more runs]            (runs = glob_<run>.npz of the meeting folder)"""
import sys, re, ast, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open('/shared/nas2/Hangkai/ibis-port/python-port/ibis/io/pft_types.py').read()).group(1))   # no jax import needed
def _rank(x): r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r
def spearman(x, y): return float(np.corrcoef(_rank(x), _rank(y))[0, 1])
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'; M = f'{S}/meeting'
out, runs = sys.argv[1], sys.argv[2:]
ref = np.load(f'{M}/glob_refs.npz'); lat, lon = ref['lat'], ref['lon']; LO, LA = np.meshgrid(lon, lat)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(LA)) * np.nan_to_num(ref['lf']); W = ref['wood']; Z = ref['zone']
zn = lambda *names: np.isin(Z, [ZONES.index(n) for n in names])
GROUPS = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'),
          'subtropical / Mediterranean / highland': zn('C_med', 'C_subtrop', 'C_highland'),
          'temperate': zn(*[z for z in ZONES if z.startswith('C_oceanic') or z.startswith('D_temp') or z.startswith('D_cont')]),
          'boreal': zn(*[z for z in ZONES if 'subarctic' in z or 'boreal' in z]), 'tundra + cold dry': zn('E_tundra', 'B_cold')}
assert all(m.any() for m in GROUPS.values()), [k for k, m in GROUPS.items() if not m.any()]
CONT = {'N America': (LO < -30) & (LA > 12), 'S America': (LO < -30) & (LA <= 12), 'Europe': (LO >= -30) & (LO < 60) & (LA > 36),
        'Africa': (LO >= -30) & (LO < 60) & (LA <= 36), 'Asia': (LO >= 60) & ~((LO > 110) & (LA < -10)), 'Australia': (LO > 110) & (LA < -10)}
D = {r: np.load(f'{M}/glob_{r}.npz')['wood'] for r in runs}
land = np.isfinite(W) & (np.nan_to_num(ref['lf']) > 0)
def tot(x, m): ok = m & land & np.isfinite(x); return float((x[ok] * area[ok]).sum()) / 1e12
def rank(x, m): ok = m & land & np.isfinite(x); return spearman(x[ok], W[ok]) if ok.sum() > 30 else np.nan
rows = []
print(f"{'chunk':44s} {'map PgC':>8s} " + ' '.join(f'{r:>6s} {"x":>5s} {"r":>5s}' for r in runs))
for g, gm in GROUPS.items():
    for c, cm in CONT.items():
        m = gm & cm
        if tot(W, m) < 1.5: continue
        rows.append((f'{g} · {c}', m)); print(f'{g + " · " + c:44s} {tot(W, m):8.1f} ' + ' '.join(f'{tot(D[r], m):6.1f} {tot(D[r], m) / tot(W, m):5.2f} {rank(D[r], m):+5.2f}' for r in runs))
print('-' * 44); 
for g, gm in GROUPS.items(): print(f'{g:44s} {tot(W, gm):8.1f} ' + ' '.join(f'{tot(D[r], gm):6.1f} {tot(D[r], gm) / tot(W, gm):5.2f} {rank(D[r], gm):+5.2f}' for r in runs))
for c, cm in CONT.items(): print(f'{c:44s} {tot(W, cm):8.1f} ' + ' '.join(f'{tot(D[r], cm):6.1f} {tot(D[r], cm) / tot(W, cm):5.2f} {rank(D[r], cm):+5.2f}' for r in runs))
print(f'{"GLOBAL":44s} {tot(W, land):8.1f} ' + ' '.join(f'{tot(D[r], land):6.1f} {tot(D[r], land) / tot(W, land):5.2f} {rank(D[r], land):+5.2f}' for r in runs))
# figure: ratio bars by chunk (left), model - map maps (right)
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold'}); C = ['#1D4E89', '#C8553D', '#2A9D8F', '#8D6A9F']
fig = plt.figure(figsize=(16, 7.2), dpi=110, facecolor='#F7F6F2'); gs = fig.add_gridspec(len(runs), 2, width_ratios=[1.05, 1.5], left=0.26, right=0.98, top=0.93, bottom=0.08, wspace=0.08, hspace=0.12)
ax = fig.add_subplot(gs[:, 0]); y = np.arange(len(rows))[::-1]; h = 0.8 / len(runs)
for i, r in enumerate(runs):
    ax.barh(y + (i - (len(runs) - 1) / 2) * h, [tot(D[r], m) / tot(W, m) for _, m in rows], h, color=C[i], label=r)
ax.axvline(1, color='#333', lw=1); ax.set_yticks(y); ax.set_yticklabels([f'{n}  ({tot(W, m):.0f} PgC)' for n, m in rows], fontsize=9.5); ax.set_xlabel('model wood / map live biomass carbon'); ax.set_xlim(0, max(2.2, ax.get_xlim()[1]))
ax.legend(loc='lower right', frameon=False); ax.set_facecolor('#F7F6F2'); [s.set_visible(False) for k, s in ax.spines.items() if k in ('top', 'right')]
ax.set_title('tree wood C by chunk, 2010 state vs Xu & Saatchi 2021 (2006-10)', loc='left')
for i, r in enumerate(runs):
    a = fig.add_subplot(gs[i, 1]); d = np.where(land, D[r] - W, np.nan)
    im = a.imshow(d, extent=[lon.min() - 0.25, lon.max() + 0.25, lat.min() - 0.25, lat.max() + 0.25], cmap='RdBu', vmin=-10, vmax=10, interpolation='nearest', origin='upper' if lat[0] > lat[-1] else 'lower')
    a.set_xticks([]); a.set_yticks([]); a.set_facecolor('#E8EBEF'); a.set_ylim(-58, 84); [s.set_visible(False) for s in a.spines.values()]
    a.set_title(f'{r}: model − map, kgC m⁻²  (global {tot(D[r], land):.0f} vs {tot(W, land):.0f} PgC)', loc='left', fontsize=11)
cb = fig.colorbar(im, ax=fig.axes[1:], fraction=0.025, pad=0.01); cb.set_label('kgC m⁻²')
fig.savefig(out); print('saved', out)
