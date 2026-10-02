"""Day-time and night-time NEE, model vs tower, by class and tower (NEE stage, CALIBRATION.md 36).

    python verify/calibration/diagnostics/nee_daynight_report.py <site_runs dir> <targets.npz> <out dir> [label]

<site_runs dir>/site_runs.npz must carry `nee_daynight` (nyr, 12, 2, nsite: night, day; gC m-2 month-1, + = release);
<targets.npz> is tower_nee_daynight.py's output (sum_night, sum_day on the same years). Years 2009-2012. Writes
NEE_night_day_by_class.png (mean seasonal cycles, night and day, tower vs model), nee_daynight_<label>.csv (per tower:
annual night and day sums, model/tower ratios, r of the monthly night and day series) and prints the class medians.
"""
import sys, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
run, targ, OUT = sys.argv[1:4]; label = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(run); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
R = np.load(f'{run}/site_runs.npz'); T = np.load(targ); sites = list(R['sites']); assert list(T['sites']) == sites; K = [1, 2, 3, 4]
inv = pd.read_csv('/shared/nas2/Hangkai/ibis-port/verify/calibration/site_data/site_inventory.csv').set_index('site'); igbp = {s: inv.loc[s, 'igbp'] for s in sites}; lat = {s: float(inv.loc[s, 'lat']) for s in sites}
CLS = ['ENF', 'DBF', 'MF', 'EBF', 'GRA', 'OSH', 'CSH', 'SAV', 'WSA']; NAME = {'ENF': 'evergreen needleleaf', 'DBF': 'deciduous broadleaf', 'MF': 'mixed forest', 'EBF': 'evergreen broadleaf', 'GRA': 'grassland', 'OSH': 'open shrubland', 'CSH': 'closed shrubland', 'SAV': 'savanna', 'WSA': 'woody savanna'}
mn, md = R['nee_daynight'][:, :, 0, :], R['nee_daynight'][:, :, 1, :]; tn, td = T['sum_night'], T['sum_day']
def r_of(a, b):
    ok = np.isfinite(a) & np.isfinite(b); return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 6 else np.nan
rows = []
for n, s in enumerate(sites):
    okn = np.isfinite(tn[K, :, n]); okd = np.isfinite(td[K, :, n])
    if okn.sum() < 12: continue
    rows.append(dict(site=s, igbp=igbp[s], months=int(okn.sum()), tower_night=tn[K, :, n][okn].sum() / okn.sum() * 12, model_night=mn[K, :, n][okn].sum() / okn.sum() * 12,
                     tower_day=td[K, :, n][okd].sum() / okd.sum() * 12, model_day=md[K, :, n][okd].sum() / okd.sum() * 12,
                     r_night=r_of(mn[K, :, n], tn[K, :, n]), r_day=r_of(md[K, :, n], td[K, :, n])))
D = pd.DataFrame(rows); D['night_ratio'] = D.model_night / D.tower_night; D['day_ratio'] = D.model_day / D.tower_day; D['nee_tower'] = D.tower_night + D.tower_day; D['nee_model'] = D.model_night + D.model_day
D.to_csv(f'{OUT}/nee_daynight_{label}.csv', index=False); pd.set_option('display.width', 220)
G = D.groupby('igbp').agg(n=('site', 'size'), tower_night=('tower_night', 'median'), model_night=('model_night', 'median'), night_ratio=('night_ratio', 'median'), r_night=('r_night', 'median'),
                          tower_day=('tower_day', 'median'), model_day=('model_day', 'median'), day_ratio=('day_ratio', 'median'), r_day=('r_day', 'median'), nee_tower=('nee_tower', 'median'), nee_model=('nee_model', 'median'))
print(f'{label}: annual night-time release and day-time uptake (gC m-2 yr-1), model / tower, r of the monthly series; class medians'); print(G.round(2).to_string())
fig, axs = plt.subplots(3, 3, figsize=(11.5, 9), sharex=True); axs = axs.ravel()
for a, c in zip(axs, CLS):
    idx = [n for n, s in enumerate(sites) if igbp[s] == c]; cyc = {}
    for tag, arr in (('tower night', tn), ('model night', mn), ('tower day', td), ('model day', md)):
        mm = []
        for n in idx:
            v = np.nanmean(arr[K, :, n], axis=0); mm.append(np.roll(v, 6) if lat[sites[n]] < 0 else v)
        cyc[tag] = np.nanmean(mm, axis=0)
    mo = np.arange(1, 13); a.axhline(0, color='0.6', lw=0.6)
    a.plot(mo, cyc['tower night'], 'k-', lw=2, label='tower, night (respiration)'); a.plot(mo, cyc['model night'], color='#d95f02', lw=2, label='model, night')
    a.plot(mo, cyc['tower day'], 'k--', lw=2, label='tower, day'); a.plot(mo, cyc['model day'], color='#1b9e77', lw=2, label='model, day')
    nsh = sum(lat[sites[n]] < 0 for n in idx); a.set_title(f'{NAME[c]} ({len(idx)})' + (f', {nsh} southern shifted 6 mo' if nsh else ''), fontsize=8.5, loc='left'); a.set_xticks([1, 4, 7, 10]); a.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
axs[0].legend(fontsize=7, frameon=False, loc='lower left'); [a.set_ylabel('NEE (gC m-2 month-1)') for a in axs[::3]]
fig.suptitle(f'Night-time and day-time NEE by class, 2009-2012: tower (NEE_VUT_REF, NIGHT flag) vs model ({label}); + = release', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/NEE_night_day_by_class_{label}.png', dpi=150)
