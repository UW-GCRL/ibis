"""ET diagnosis, step 1 (Hangkai 2026-09-30: "ET first"): which component carries each class's ET bias.

    python verify/calibration/diagnostics/et_components_by_class.py <site_runs dir> <out dir>

Reads site_runs.npz with `et_parts` (nyr, 12, 4, nsite: transpiration, interception, soil evaporation, snow sublimation,
mm month-1; site_run.py from 2026-09-30) and the tower ET saved with it. Per class: the mean seasonal cycle of the four
model components stacked against the tower's ET (southern towers shifted 6 months), the annual component shares, and
the annual model/tower ratio. Figure ET_4_components_by_class.png and table et_components_by_class.csv in <out dir>.
"""
import sys, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
run, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True); plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
R = np.load(f'{run}/site_runs.npz'); sites = list(R['sites']); K = [1, 2, 3, 4]; P = R['et_parts']; obs = R['obs_et']; et = R['et']
inv = pd.read_csv('/shared/nas2/Hangkai/ibis-port/verify/calibration/site_data/site_inventory.csv').set_index('site'); igbp = {s: inv.loc[s, 'igbp'] for s in sites}; lat = {s: float(inv.loc[s, 'lat']) for s in sites}
CLS = ['ENF', 'DBF', 'MF', 'EBF', 'GRA', 'OSH', 'CSH', 'SAV', 'WSA']; NAME = {'ENF': 'evergreen needleleaf forest', 'DBF': 'deciduous broadleaf forest', 'MF': 'mixed forest', 'EBF': 'evergreen broadleaf forest', 'GRA': 'grassland', 'OSH': 'open shrubland', 'CSH': 'closed shrubland', 'SAV': 'savanna', 'WSA': 'woody savanna'}
PARTS = ['transpiration', 'interception', 'soil evaporation', 'snow sublimation']; PC = ['#1b9e77', '#66a61e', '#a6761d', '#7570b3']
fig, axs = plt.subplots(3, 3, figsize=(11.5, 9), sharex=True); axs = axs.ravel(); rows = []
for a, c in zip(axs, CLS):
    idx = [n for n, s in enumerate(sites) if igbp[s] == c]; comp = np.zeros((4, 12)); tow = []; mod = []
    for n in idx:
        sh = 6 if lat[sites[n]] < 0 else 0
        comp += np.roll(np.nanmean(P[K, :, :, n], axis=0).T, sh, axis=1) / len(idx)
        tow.append(np.roll(np.nanmean(obs[K, :, n], axis=0), sh)); mod.append(np.roll(np.nanmean(et[K, :, n], axis=0), sh))
    tow = np.nanmean(tow, axis=0); mod = np.nanmean(mod, axis=0); mo = np.arange(1, 13)
    a.stackplot(mo, comp, labels=PARTS, colors=PC, alpha=0.85); a.plot(mo, tow, 'k-', lw=2.2, label='tower ET'); a.plot(mo, mod, color='0.3', lw=1, ls=':', label='model ET (sum)')
    ann = comp.sum(axis=1); tot = ann.sum(); ratio = tot / np.nansum(tow)
    a.set_title(f'{NAME[c]} ({len(idx)} towers): annual model/tower {ratio:.2f}\nshares: transpiration {ann[0] / tot:.0%}, interception {ann[1] / tot:.0%}, soil evap. {ann[2] / tot:.0%}, snow {ann[3] / tot:.0%}', fontsize=8, loc='left')
    a.set_xticks([1, 4, 7, 10]); a.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct']); a.set_ylim(bottom=0)
    rows.append(dict(cls=c, n=len(idx), ratio=ratio, tower_mm=float(np.nansum(tow)), model_mm=float(tot), **{p: float(v) for p, v in zip(['trans', 'interc', 'soilev', 'snowsub'], ann)},
                     spring_ratio=float(comp[:, 2:5].sum() / np.nansum(tow[2:5])), summer_ratio=float(comp[:, 5:8].sum() / np.nansum(tow[5:8])), spring_soilev_share=float(comp[2, 2:5].sum() / comp[:, 2:5].sum())))
axs[0].legend(fontsize=7, frameon=False, loc='upper left'); [a.set_ylabel('ET (mm month-1)') for a in axs[::3]]
fig.suptitle(f'Model ET by component vs tower ET, mean seasonal cycle by class, 2009-2012 ({os.path.basename(run)})', fontsize=10, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{OUT}/ET_4_components_by_class.png', dpi=150)
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/et_components_by_class.csv', index=False); pd.set_option('display.width', 200); print(T.round(2).to_string(index=False))
