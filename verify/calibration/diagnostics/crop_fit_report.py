"""Cropland tower fits (CALIBRATION.md 41): fitted crop/grass capacity ratio per tower, annual GPP model/tower on the
training years and the held-out year, and the medians that go into the `crop:` yaml section.

    python verify/calibration/diagnostics/crop_fit_report.py <fit root> [<fit root> ...] --out <csv> [--yaml <base.yaml> <out.yaml>]
"""
import sys, os, glob, json, numpy as np, pandas as pd
args = sys.argv[1:]; out = args[args.index('--out') + 1]; roots = [a for a in args if not a.startswith('--') and a not in (out,) and os.path.isdir(a)]
ybase = yout = None
if '--yaml' in args: k = args.index('--yaml'); ybase, yout = args[k + 1], args[k + 2]
FX = os.environ.get('IBIS_FLUX_MM_DIR', '/shared/nas2/Hangkai/ibis/inputs/sites/FLUXNET2015_FULLSET/daily_monthly_yearly')
C4 = {'US-Ne1', 'US-Ne2', 'US-Ne3', 'IT-BCi', 'US-Tw2'}                      # maize towers run as C4 grass (the fit commands' --pft)
rows = []
for root in roots:
    for d in sorted(glob.glob(f'{root}/*/')):
        s = os.path.basename(d.rstrip('/'))
        if not os.path.exists(f'{d}best.json') or not os.path.exists(f'{d}best_series.npz'): continue
        b = json.load(open(f'{d}best.json')); R = np.load(f'{d}best_series.npz'); yrs = [int(y) for y in R['years']]; gm = R['gpp'][:, :, 0]
        f = glob.glob(f'{FX}/FLX_{s}_FLUXNET2015_FULLSET_MM_*.csv')
        if not f: continue
        m = pd.read_csv(f[0]); t = pd.to_datetime(m['TIMESTAMP'].astype(str).str[:6], format='%Y%m'); m['y'], m['mo'] = t.dt.year, t.dt.month; nd = t.dt.days_in_month
        g = m['GPP_NT_VUT_REF'].where(m['GPP_NT_VUT_REF'] > -9000) * nd; obs = np.full((len(yrs), 12), np.nan)
        for (y, mo), v in zip(zip(m.y, m.mo), g):
            if y in yrs: obs[yrs.index(y), mo - 1] = v
        def st(idx):
            mm, oo = gm[idx], obs[idx]; ok = np.isfinite(oo)
            return (float(mm[ok].sum() / oo[ok].sum()), float(np.corrcoef(mm[ok], oo[ok])[0, 1])) if ok.sum() > 3 else (np.nan, np.nan)
        rt, rr = st(list(range(1, len(yrs) - 1))); rh, rrh = st([len(yrs) - 1])
        h = pd.read_csv(f'{d}history.csv') if os.path.exists(f'{d}history.csv') else None
        rows.append(dict(site=s, years=f'{yrs[0]}-{yrs[-1]}', pft='C4' if s in C4 else 'C3', loss0=float(h.loss.iloc[0]) if h is not None else np.nan, loss=b['loss'], n_eval=b.get('eval'),
                         crop_vmax_scale=b['params'].get('crop_vmax_scale_c4', b['params'].get('crop_vmax_scale_c3', np.nan)), ratio0=np.nan, ratio_train=rt, r_train=rr, ratio_hold=rh, r_hold=rrh))
D = pd.DataFrame(rows).drop_duplicates('site'); D.to_csv(out, index=False); pd.set_option('display.width', 200); print(D.round(3).to_string(index=False))
print(f"\ncrop_vmax_scale median {D.crop_vmax_scale.median():.3f} (IQR {D.crop_vmax_scale.quantile(.25):.2f}-{D.crop_vmax_scale.quantile(.75):.2f}); by pft: " + ', '.join(f'{p} {g.crop_vmax_scale.median():.2f} (n {len(g)})' for p, g in D.groupby('pft')))
print(f"held-out annual GPP model/tower: median {D.ratio_hold.median():.2f}, within 25 % {np.nanmean(np.abs(D.ratio_hold - 1) <= 0.25):.0%}; monthly r median {D.r_hold.median():.2f}")
if ybase:
    import yaml; t = yaml.safe_load(open(ybase)); g = D.groupby('pft').crop_vmax_scale.median(); c0 = dict(t.get('crop') or {}); c0.setdefault('harvest_index', 0.45)
    for pft, key in (('C3', 'vmax_scale_c3'), ('C4', 'vmax_scale_c4')):          # fitted medians only; other keys (c4_share, a PFT without towers) keep the base's
        if pft in g.index: c0[key] = float(g[pft])
    t['crop'] = c0; yaml.safe_dump(t, open(yout, 'w'), sort_keys=False); print('wrote', yout, t['crop'])
