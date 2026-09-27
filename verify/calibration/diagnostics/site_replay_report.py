"""Verdict-first report of a tower fit on held-out years (CALIBRATION.md 29): magnitude by month, timing, light response.

    python verify/calibration/diagnostics/site_replay_report.py <fit dir> <SITE> <IGBP> [<camera roi>]

Reads <fit dir>/best.json and best_series.npz (the fit window), daily_fitted_<win>.npz and daily_default_<win>.npz
(site_daily_replay.py on the held-out window). Prints model / tower for the fitted and the default parameters: peak
month, June-August, annual GPP, ET (LE_CORR where the tower has it), cold-season share, per year and as the window
mean; leaf-out and senescence dates (25 % crossings of the 15-day smoothed daily GPP, phenology_vs_phenocam.crossings)
against the tower's daily GPP and, where a camera ROI is given, PhenoCam's gcc_90 transition dates.
"""
import sys, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = ['python-port', 'verify/calibration']
import site_fit
from phenology_vs_phenocam import crossings, tower_gpp_dates, C as CAM
fd, site, igbp = sys.argv[1:4]; roi = sys.argv[4] if len(sys.argv) > 4 else ''
best = json.load(open(f'{fd}/best.json')); print(f"fit: loss {best['loss']:.2f} at eval {best['eval']}; " + ', '.join(f'{k} {v:.4g}' for k, v in best['params'].items()))
runs = {}
for f in sorted(glob.glob(f'{fd}/daily_*.npz')):
    R = np.load(f, allow_pickle=True); runs[f.split('/')[-1][6:-4]] = R
if not runs: raise SystemExit('no daily replays')
pd.set_option('display.width', 220)
for lab, R in runs.items():
    years = [int(y) for y in R['years']]; site_fit.YEARS = years; o = site_fit.tower_obs(site); n = list(R['sites']).index(site)
    g, e = R['gpp'][:, :, n], R['et'][:, :, n]; tg, te = o['gpp'], o['et']
    summer = [5, 6, 7]; cold = [0, 1, 10, 11]; rows = []
    for k, y in enumerate(years):
        ok = np.isfinite(tg[k]); oke = np.isfinite(te[k])
        if ok.sum() < 10: continue
        rows.append(dict(year=y, peak=g[k].max() / np.nanmax(tg[k]), jja=g[k][summer].sum() / np.nansum(tg[k][summer]), annual=g[k][ok].sum() / tg[k][ok].sum(),
                         et=(e[k][oke].sum() / te[k][oke].sum()) if oke.sum() >= 10 else np.nan, cold_model=g[k][cold].sum() / g[k].sum(), cold_tower=np.nansum(tg[k][cold]) / np.nansum(tg[k])))
    T = pd.DataFrame(rows).set_index('year'); print(f'\n== {lab} ({years[0]}-{years[-1]}), model / tower:'); print(T.round(2).to_string()); print('mean: ' + '  '.join(f'{c} {T[c].mean():.2f}' for c in T.columns))
    # timing
    tdates, ntow = tower_gpp_dates(site, years); acc = {'r25': [], 'f25': [], 'r50': [], 'f50': []}
    for k, y in enumerate(years):
        s = R['gpp_daily'][k, :, n] * 1000; s = s[np.isfinite(s)]
        r25, f25 = crossings(s, 0.25); r50, f50 = crossings(s, 0.50)
        if np.isfinite(r25): acc['r25'].append(r25); acc['f25'].append(f25); acc['r50'].append(r50); acc['f50'].append(f50)
    md = {k: (float(np.mean(v)) if v else np.nan) for k, v in acc.items()}
    print(f'timing (DOY, mean over years): model leaf-out 25 % {md["r25"]:.0f} / 50 % {md["r50"]:.0f}, senescence 50 % {md["f50"]:.0f} / 25 % {md["f25"]:.0f} | '
          f'tower GPP ({ntow} yr): {tdates.get("r25", np.nan):.0f} / {tdates.get("r50", np.nan):.0f}, {tdates.get("f50", np.nan):.0f} / {tdates.get("f25", np.nan):.0f}')
    if roi:
        f = f'{CAM}/{roi}_3day_transition_dates.csv'; d = pd.read_csv(f, comment='#'); d = d[d['gcc_value'] == 'gcc_90']
        cam = {'r25': [], 'f25': [], 'r50': [], 'f50': []}
        for _, x in d.iterrows():
            t25 = pd.to_datetime(x['transition_25']); t50 = pd.to_datetime(x['transition_50'])
            if t25.year not in years: continue
            if x['direction'] == 'rising': cam['r25'].append(t25.dayofyear); cam['r50'].append(t50.dayofyear)
            else: cam['f25'].append(t25.dayofyear); cam['f50'].append(t50.dayofyear)
        cy = sorted({pd.to_datetime(x).year for x in d['transition_25'] if pd.to_datetime(x).year in years})
        print(f'camera {roi} ({len(cy)} yr {cy[0] if cy else "-"}-{cy[-1] if cy else "-"}): leaf-out 25 % {np.mean(cam["r25"]) if cam["r25"] else np.nan:.0f} / 50 % {np.mean(cam["r50"]) if cam["r50"] else np.nan:.0f}, '
              f'senescence 50 % {np.mean(cam["f50"]) if cam["f50"] else np.nan:.0f} / 25 % {np.mean(cam["f25"]) if cam["f25"] else np.nan:.0f}')
