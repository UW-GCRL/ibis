"""Per-tower NEE fits: fitted values and the night/day verdict on the fit years and the held-out year (CALIBRATION.md 36).

    python verify/calibration/diagnostics/nee_fit_report.py <nee fit root> <targets.npz> <baseline site_runs dir> <out csv>

Each <root>/<SITE>/ with a finished run (BEST eval in run.log) contributes: the fitted som_scale, bgc_e0, rgrowth, rroot;
model / tower annual night-time release and day-time uptake, r of the monthly night and day series, on 2009-2011 (fit)
and 2012 (held out); the same for the S1 baseline (<baseline>/site_runs.npz nee_daynight) for comparison.
"""
import sys, os, glob, json, numpy as np, pandas as pd
root, targ, base, out = sys.argv[1:5]; T = np.load(targ); tsites = [str(s) for s in T['sites']]; B = np.load(f'{base}/site_runs.npz'); bsites = [str(s) for s in B['sites']]
inv = pd.read_csv('/shared/nas2/Hangkai/ibis-port/verify/calibration/site_data/site_inventory.csv').set_index('site')
TR, HO = [1, 2, 3], [4]
def stats(mn, md, tn, td, ky):
    o = {}
    for tag, m, t in (('night', mn, tn), ('day', md, td)):
        ok = np.isfinite(t[ky]); o[f'{tag}_ratio'] = (m[ky][ok].sum() / t[ky][ok].sum()) if ok.sum() >= 6 and t[ky][ok].sum() != 0 else np.nan
        o[f'r_{tag}'] = np.corrcoef(m[ky][ok], t[ky][ok])[0, 1] if ok.sum() > 6 else np.nan
    return o
rows = []
for f in sorted(glob.glob(f'{root}/*/run.log')):
    s = f.split('/')[-2]; d = os.path.dirname(f)
    if 'BEST eval' not in open(f).read() or not os.path.exists(f'{d}/best_series.npz') or s not in tsites: continue
    b = json.load(open(f'{d}/best.json')); R = np.load(f'{d}/best_series.npz', allow_pickle=True); k = tsites.index(s)
    ne = R['nee']; mn, md = ne[:, :, 0, 0], ne[:, :, 1, 0]; tn, td = T['sum_night'][:, :, k], T['sum_day'][:, :, k]
    base_line = [l for l in open(f) if l.startswith('baseline loss')]
    row = dict(site=s, igbp=inv.loc[s, 'igbp'], loss0=float(base_line[0].split()[-1]) if base_line else np.nan, loss=b['loss'], hold_loss=b.get('holdout_year', np.nan), **{f'p:{kk}': v for kk, v in b['params'].items()})
    for lab, ky in (('fit', TR), ('hold', HO)):
        for kk, v in stats(mn, md, tn, td, ky).items(): row[f'{kk}_{lab}'] = v
    if s in bsites:
        kb = bsites.index(s); bn, bd = B['nee_daynight'][:, :, 0, kb], B['nee_daynight'][:, :, 1, kb]
        for kk, v in stats(bn, bd, tn, td, HO).items(): row[f'{kk}_s1_hold'] = v
    rows.append(row)
D = pd.DataFrame(rows); D.to_csv(out, index=False); pd.set_option('display.width', 240)
if len(D):
    print(f'{len(D)} towers; held-out 2012, model / tower: S1 set -> per-tower NEE fit')
    for kk in ('night_ratio', 'day_ratio', 'r_night', 'r_day'):
        a, c = D[f'{kk}_s1_hold'], D[f'{kk}_hold']; print(f'  {kk:12s} median {a.median():.2f} -> {c.median():.2f}' + (f';  within 25 % {np.mean(np.abs(np.log(a.dropna())) < np.log(1.25)):.0%} -> {np.mean(np.abs(np.log(c.dropna())) < np.log(1.25)):.0%}' if 'ratio' in kk else ''))
    pc = [c for c in D.columns if c.startswith('p:')]; print('\nfitted values, median by class (and bound-pressing counts: som_scale <0.12 or >2.7, rgrowth <=0.11 or >=0.49, bgc_e0 <160 or >590, rroot <0.33 or >4.5):')
    G = D.groupby('igbp')[pc + ['night_ratio_hold', 'day_ratio_hold', 'r_night_hold', 'r_day_hold']].median(); G['n'] = D.groupby('igbp').size(); print(G.round(2).to_string())
    print('at bounds:', {c: int(((D[c] < lo) | (D[c] > hi)).sum()) for c, (lo, hi) in {'p:som_scale': (0.12, 2.7), 'p:rgrowth': (0.11, 0.49), 'p:bgc_e0': (160, 590), 'p:rroot': (0.33, 4.5)}.items() if c in D})
