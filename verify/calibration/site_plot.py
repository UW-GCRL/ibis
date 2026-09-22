"""Figures for a site_run.py output directory (reads site_runs.npz / site_runs.json; needs matplotlib -> CPU interpreter).

    ibis-python verify/calibration/site_plot.py <out_dir>
"""
import sys, json, os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out = sys.argv[1]; z = np.load(f'{out}/site_runs.npz', allow_pickle=True); meta = json.load(open(f'{out}/site_runs.json'))
sites = [str(s) for s in z['sites']]; years = [int(y) for y in z['years']]; GM, EM, og_all, oe_all = z['gpp'], z['et'], z['obs_gpp'], z['obs_et']
for n, s in enumerate(sites):
    note = meta['notes'][s]
    og = np.nanmean(og_all[:, :, n], 0); mg = GM[:, :, n].mean(0); oe = np.nanmean(oe_all[:, :, n], 0); me = EM[:, :, n].mean(0)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    axs[0].fill_between(range(12), 0, og, color='0.85'); axs[0].plot(range(12), og, 'k', lw=1.5, label='tower GPP (night-time partitioning)'); axs[0].plot(range(12), mg, 'tab:green', lw=2, marker='o', ms=3, label='model'); axs[0].set_ylabel('GPP (gC m$^{-2}$ month$^{-1}$)')
    axs[1].fill_between(range(12), 0, np.nan_to_num(oe), color='0.85'); axs[1].plot(range(12), oe, 'k', lw=1.5, label='tower LE as ET'); axs[1].plot(range(12), me, 'tab:blue', lw=2, marker='o', ms=3, label='model'); axs[1].set_ylabel('ET (mm month$^{-1}$)')
    for ax in axs[:2]: ax.set_xticks(range(12)); ax.set_xticklabels(list('JFMAMJJASOND')); ax.legend(frameon=False, fontsize=8)
    t = np.arange(GM.shape[0] * 12); axs[2].plot(t, og_all[:, :, n].ravel(), 'k', lw=1, label='tower'); axs[2].plot(t, GM[:, :, n].ravel(), 'tab:green', lw=1, label='model')
    axs[2].set_xticks(np.arange(0, len(t), 12)); axs[2].set_xticklabels(years, rotation=90, fontsize=7); axs[2].set_ylabel('monthly GPP'); axs[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{s} ({note['igbp']} -> {'+'.join(note['pfts'])}, upper cover {note['fu']}, lower {note['fl']}); tower years {years[0]}-{years[-1]}, {meta['cycles']} spin cycle(s); {os.path.basename(meta['params'])}", fontsize=9, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(f'{out}/{s}.png', dpi=130); plt.close(fig)
print(f'{len(sites)} site figures written to {out}')
