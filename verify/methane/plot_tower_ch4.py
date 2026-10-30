"""Tower methane against the model at the parameters the GLOBAL model uses: each of the 48 wetland towers run once with its
wetland band's parameters (zone band, ch4_towers_zone.csv), (a) the release setting: hourly flux, the hourly-refitted bands
(site_fit/ch4_band_h), (b) the setting before 29 October: month-end snapshot, the production bands (site_fit/ch4_band_snap).
Same years, meteorology, start state and inundated fraction 1 as the fits. Monthly gC-CH4 m-2 month-1, months with a
tower value. The tower data are Chenchen's unpublished gap-filled FLUXNET-CH4 v2 set: figure and per-tower table stay local.
    python plot_tower_ch4.py <figure dir> <table out>"""
import sys, glob, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FIG, TAB = sys.argv[1], sys.argv[2]
SF = '/shared/nas2/Hangkai/ibis/verify_outputs/site_fit'; M = '/shared/nas2/Hangkai/ibis/inputs/sites/methane_towers'; L = '/shared/nas2/Hangkai/ibis_local_document/logs'
T = np.load(f'{M}/ch4_targets.npz'); tsites = list(T['sites']); tyears = [int(y) for y in T['years']]
meta = pd.read_csv(f'{M}/ch4_sites.csv').set_index('SITE_ID'); zb = pd.read_csv(f'{L}/ch4_towers_zone.csv').set_index('site')['zone_band']
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0', '#fcfcfb'; CH, CS = '#2a78d6', '#8a8984'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                     'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.8, 'legend.frameon': False})
rows = []; series = {}
for d in sorted(glob.glob(f'{SF}/ch4_band_h/*/best_series.npz')):
    s = d.split('/')[-2]; ds = f'{SF}/ch4_band_snap/{s}/best_series.npz'
    if not os.path.exists(ds) or s not in tsites: continue
    zh, zs = np.load(d), np.load(ds); yrs = [int(y) for y in zh['years']]; k = tsites.index(s)
    obs = np.array([T['sum_ch4'][tyears.index(y), :, k] for y in yrs]); mh = zh['ch4'][..., 0].astype(float); ms = zs['ch4'][..., 0].astype(float)
    ok = np.isfinite(obs); lat = float(meta.loc[s, 'LOCATION_LAT']); hold = np.zeros_like(ok); hold[-1] = True
    series[s] = dict(obs=obs, h=mh, s=ms, ok=ok, hold=hold, lat=lat, band=zb.get(s, ''))
    def stats(mod, sel):
        o, p = obs[sel], mod[sel]
        return (p.sum() / o.sum() if o.sum() > 0 else np.nan), (np.corrcoef(o, p)[0, 1] if len(o) > 3 and o.std() > 0 and p.std() > 0 else np.nan)
    for tag, mod in (('hourly', mh), ('snapshot', ms)):
        rt, rr = stats(mod, ok & ~hold); ht, hr = stats(mod, ok & hold)
        rows.append(dict(site=s, band=zb.get(s, ''), setting=tag, years=f'{yrs[0]}-{yrs[-1]}', tower_gC_yr=np.nanmean(np.where(ok, obs, np.nan)) * 12,
                         ratio_train=rt, r_train=rr, ratio_hold=ht, r_hold=hr))
D = pd.DataFrame(rows); D.to_csv(TAB, index=False)
print(f'towers with both settings: {len(series)} of 48')
print('band medians over towers (amount = model sum / tower sum over months with a tower value; r = monthly correlation):')
for b in ('tropical', 'temperate', 'boreal'):
    for tag in ('snapshot', 'hourly'):
        g = D[(D.band == b) & (D.setting == tag)]
        print(f"  {b:9s} {tag:8s} n {len(g):2d} | fit years: amount {g.ratio_train.median():.2f}, r {g.r_train.median():.2f}, within 25 % {((g.ratio_train - 1).abs() < 0.25).mean():.0%} | held-out year: amount {g.ratio_hold.median():.2f}, r {g.r_hold.median():.2f}, within 25 % {((g.ratio_hold - 1).abs() < 0.25).mean():.0%}")
fig, ax = plt.subplots(3, 3, figsize=(17, 14), constrained_layout=True)
MON = list('JFMAMJJASOND')
for i, b in enumerate(('tropical', 'temperate', 'boreal')):
    ss = [v for v in series.values() if v['band'] == b]
    a = ax[i, 0]; mx = 0
    for tag, col, ms_, lab in (('s', CS, 14, 'before 29 Oct (snapshot, production bands)'), ('h', CH, 18, 'release (hourly, refitted bands)')):
        o = np.concatenate([v['obs'][v['ok']] for v in ss]); p = np.concatenate([v[tag][v['ok']] for v in ss]); mx = max(mx, np.nanpercentile(o, 99.5), np.nanpercentile(p, 99.5))
        a.scatter(o, p, s=ms_, color=col, alpha=0.55, lw=0, label=lab)
    a.plot([0, mx], [0, mx], color=INK2, lw=1); a.set_xlim(0, mx); a.set_ylim(0, mx)
    a.set_xlabel('tower, gC-CH₄ m⁻² month⁻¹'); a.set_ylabel('model, gC-CH₄ m⁻² month⁻¹'); a.set_title(f'{b} towers (n {len(ss)}): every month', loc='left')
    if i == 0: a.legend(loc='upper left', fontsize=9)
    a = ax[i, 1]
    def cyc(v, key):   # the tower's mean monthly cycle over its months with data, shifted 6 months south of the equator
        x = np.where(v['ok'], v[key], np.nan); c = np.nanmean(x, 0); return np.roll(c, 6) if v['lat'] < 0 else c
    for key, col, lw, ls, lab in (('obs', INK, 2.4, '-', 'tower'), ('s', CS, 1.6, '--', 'before 29 Oct'), ('h', CH, 2.2, '-', 'release')):
        C = np.array([cyc(v, key) for v in ss]); med = np.nanmedian(C, 0); a.plot(range(12), med, color=col, lw=lw, ls=ls, label=lab)
        if key == 'obs': a.fill_between(range(12), np.nanpercentile(C, 25, 0), np.nanpercentile(C, 75, 0), color='#eceae4', lw=0)
    a.set_xticks(range(12)); a.set_xticklabels(MON); a.set_ylabel('gC-CH₄ m⁻² month⁻¹'); a.set_title('seasonal cycle, median tower', loc='left'); a.text(0.0, -0.13, 'shading: tower inter-quartile range; southern towers shifted 6 months', transform=a.transAxes, fontsize=9, color=INK2)
    if i == 0: a.legend(loc='upper left', fontsize=9)
    a = ax[i, 2]
    o = np.array([np.nanmean(np.where(v['ok'], v['obs'], np.nan)) * 12 for v in ss]); h = np.array([np.nanmean(np.where(v['ok'], v['h'], np.nan)) * 12 for v in ss]); s_ = np.array([np.nanmean(np.where(v['ok'], v['s'], np.nan)) * 12 for v in ss])
    mx = max(np.nanmax(o), np.nanmax(h), np.nanmax(s_)) * 1.05
    a.scatter(o, s_, s=40, color=CS, label='before 29 Oct'); a.scatter(o, h, s=46, color=CH, edgecolor=SURF, lw=1, label='release')
    a.plot([0, mx], [0, mx], color=INK2, lw=1); a.set_xlim(0, mx); a.set_ylim(0, mx); a.set_xlabel('tower, gC-CH₄ m⁻² yr⁻¹'); a.set_ylabel('model, gC-CH₄ m⁻² yr⁻¹')
    a.set_title('annual emission per tower', loc='left')
fig.suptitle('Wetland CH4 at the 48 towers with the parameters the global model uses (one set per wetland band)', x=0.01, ha='left', fontsize=14, fontweight='bold', color=INK)
fig.savefig(f'{FIG}/tower_ch4_band_params.png', dpi=100); plt.close(fig); print('wrote', f'{FIG}/tower_ch4_band_params.png')
