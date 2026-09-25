"""C1 site-fit report (CALIBRATION.md 22-23): trajectory, per-tower metrics and monthly series, baseline vs best.

    python verify/calibration/diagnostics/c1_report.py <fit_dir> <baseline_dir> <best_dir> [<figure.png>]

<fit_dir> is a `site_fit.py` run (history.csv, best.json, setup.json). <baseline_dir> and <best_dir> hold best_series.npz of
two evaluations on the same towers -- normally `site_fit.py --only-baseline` replays without and with `--start best.json`,
which also carry monthly NEE (older series without NEE are accepted and the NEE rows are skipped). Prints the trajectory
summary, the parameters (start -> best, with bounds), and per tower: the loss terms of section 22 (GPP and ET misfit over
the training years), annual GPP and ET ratios, the NDJF share of GPP, and the monthly NEE RMSE; then the class medians for
the training towers, the held-out towers and the held-out year. Writes the figure if a path is given.
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
from site_fit import tower_obs, misfit, TRAIN_Y, HOLD_Y, YEARS
from fit_carbon import EXTRAS, VMAX_LO, VMAX_HI, PFT_NAMES

fit_dir, base_dir, best_dir = sys.argv[1:4]; fig_path = sys.argv[4] if len(sys.argv) > 4 else ''
h = pd.read_csv(f'{fit_dir}/history.csv'); best = json.load(open(f'{fit_dir}/best.json')); setup = json.load(open(f'{fit_dir}/setup.json'))
B0 = np.load(f'{base_dir}/best_series.npz', allow_pickle=True); B1 = np.load(f'{best_dir}/best_series.npz', allow_pickle=True)
sites = [str(s) for s in B1['sites']]; assert sites == [str(s) for s in B0['sites']], 'baseline and best series are on different towers'
train, hold = setup['train'], setup['hold']; w_et = setup.get('w_et', 0.5); obs = {s: tower_obs(s) for s in sites}
names = setup['fit']

print(f"C1 {setup['igbp']}: {len(sites)} towers ({len(train)} train, {len(hold)} held out); params {setup['params']}; {len(h)} evaluations")
k = h['loss'].idxmin(); b0 = h.iloc[0]
print(f"loss     start {b0['loss']:.3f} -> best {h['loss'][k]:.3f} (eval {int(h['eval'][k])}); held-out towers {b0['holdout_towers']:.3f} -> {h['holdout_towers'][k]:.3f}; "
      f"held-out year {b0['holdout_year']:.3f} -> {h['holdout_year'][k]:.3f}")
for q in (0.25, 0.5, 0.75, 1.0):
    n = max(1, int(round(q * len(h)))); j = h['loss'][:n].idxmin()
    print(f"  best after {n:4d} evaluations: {h['loss'][j]:.3f} (held-out towers {h['holdout_towers'][j]:.3f}, year {h['holdout_year'][j]:.3f})")
print('\nparameters (start -> best, [bounds]; * = within 2 % of a bound):')
for nm in names:
    v0, v1 = b0[nm], best['params'][nm]
    if nm.startswith('vmax:'):
        k_ = PFT_NAMES.index(nm[5:]); lo, hi = VMAX_LO[k_], VMAX_HI[k_]; txt = f"{v0 * 1e6:8.2f} -> {v1 * 1e6:8.2f} umol m-2 s-1 [{lo * 1e6:.0f}, {hi * 1e6:.0f}]"
    else: lo, hi = EXTRAS[nm][1], EXTRAS[nm][2]; txt = f"{v0:8.3f} -> {v1:8.3f}   [{lo}, {hi}]"
    edge = '*' if min(abs(v1 - lo), abs(v1 - hi)) < 0.02 * (hi - lo) else ''
    print(f"  {nm:24s} {txt} {edge}")

def per_tower(S):
    gm, em = S['gpp'], S['et']; nm_ = S['nee'] if 'nee' in S.files else None; rows = []
    for n, s in enumerate(sites):
        o = obs[s]; tr = TRAIN_Y
        g_o = np.nanmean(o['gpp'][tr], 0); g_m = gm[tr, :, n].mean(0); e_o = np.nanmean(o['et'][tr], 0); e_m = em[tr, :, n].mean(0)
        ok = np.isfinite(g_o); oke = np.isfinite(e_o)
        r = dict(site=s, set='train' if s in train else 'hold', loss_train_years=misfit(gm, em, o, n, tr, w_et), loss_hold_year=misfit(gm, em, o, n, HOLD_Y, w_et),
                 gpp_ratio=g_m[ok].sum() / max(g_o[ok].sum(), 1e-6), et_ratio=e_m[oke].sum() / max(e_o[oke].sum(), 1e-6) if oke.any() else np.nan,
                 ndjf_model=g_m[[10, 11, 0, 1]].sum() / max(g_m.sum(), 1e-6), ndjf_tower=np.nansum(g_o[[10, 11, 0, 1]]) / max(np.nansum(g_o), 1e-6))
        if nm_ is not None:
            d = nm_[tr, :, n] - o['nee'][tr]; r['nee_rmse'] = float(np.sqrt(np.nanmean(d ** 2))) if np.isfinite(d).any() else np.nan
            r['nee_bias'] = float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index('site')

P0, P1 = per_tower(B0), per_tower(B1)
pd.set_option('display.width', 250)
cols = [c for c in ('loss_train_years', 'loss_hold_year', 'gpp_ratio', 'et_ratio', 'ndjf_model', 'ndjf_tower', 'nee_rmse', 'nee_bias') if c in P1]
T = pd.concat({'baseline': P0[cols], 'best': P1[cols]}, axis=1).round(3); T.insert(0, 'set', P1['set'])
print('\nper tower (training years 2009-2011 unless noted):'); print(T.to_string())
print('\nmedians:')
for grp in ('train', 'hold'):
    m0 = P0[P0['set'] == grp][cols].median(); m1 = P1[P1['set'] == grp][cols].median()
    print(f"  {grp:5s} " + '  '.join(f"{c} {m0[c]:.3f}->{m1[c]:.3f}" for c in cols))

if fig_path:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
    has_nee = 'nee' in B1.files and 'nee' in B0.files; mo = np.arange(1, 13); n_s = len(sites)
    VARS = (('gpp', 'GPP gC m-2 mo-1'), ('et', 'ET mm mo-1')) + ((('nee', 'NEE gC m-2 mo-1'),) if has_nee else ())
    ncol = 6; nblk = -(-n_s // ncol); nv = len(VARS)
    fig, axs = plt.subplots(nblk * nv, ncol, figsize=(2.2 * ncol, 1.55 * nblk * nv + 0.9), squeeze=False)
    for n in range(n_s, nblk * ncol):
        for r in range(nv): axs[(n // ncol) * nv + r, n % ncol].set_visible(False)
    for n, s in enumerate(sites):
        o = obs[s]
        for r, (key, lab) in enumerate(VARS):
            ax = axs[(n // ncol) * nv + r, n % ncol]; ob = np.nanmean(o[key][TRAIN_Y], 0)
            ax.plot(mo, ob, 'k-o', ms=2, lw=1, label='tower'); ax.plot(mo, B0[key][TRAIN_Y, :, n].mean(0), color='tab:red', lw=1, label='baseline')
            ax.plot(mo, B1[key][TRAIN_Y, :, n].mean(0), color='tab:blue', lw=1.3, label='best')
            if key == 'nee': ax.axhline(0, color='0.7', lw=0.6)
            ax.set_xticks([1, 4, 7, 10]); ax.set_xticklabels(['J', 'A', 'J', 'O'] if r == nv - 1 else [])
            if r == 0: ax.set_title(s + ('' if s in train else ' (held out)'), fontsize=7.5)
            if n % ncol == 0: ax.set_ylabel(lab, fontsize=6.5)
    h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=3, loc='lower center')
    fig.suptitle(f"C1 {setup['igbp']}: 2009-2011 climatology, baseline vs best of {len(h)} evaluations "
                 f"(loss {b0['loss']:.2f} -> {h['loss'][k]:.2f}; held-out towers {b0['holdout_towers']:.2f} -> {h['holdout_towers'][k]:.2f})", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(fig_path, dpi=130); print('figure', fig_path)
