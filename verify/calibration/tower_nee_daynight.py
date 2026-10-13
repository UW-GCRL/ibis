"""Day-time and night-time NEE targets from the FLUXNET2015 half-hourly files (or the towers31_prep.py compact files) (Hangkai 2026-09-30: calibrate NEE with
the day and night parts separated; night NEE is ecosystem respiration measured directly).

    python verify/calibration/tower_nee_daynight.py <out.npz> [SITE ...]        (default: the 62-tower batch)

Per site, month and year 2008-2012 (the batch window): mean night-time NEE and mean day-time NEE (umol m-2 s-1; FLUXNET
sign, positive = release), their monthly integrals (gC m-2 month-1), the number of half-hours in each, and the share
that is measured (NEE_VUT_REF_QC == 0) rather than gap-filled. NIGHT is FLUXNET's own flag (SW_IN_POT == 0). Half-hours
with QC > 1 (poor gap-fill) are dropped. Files: the FULLSET HH/HR file under FX (tower_forcing.py).
"""
import sys, glob, numpy as np, pandas as pd
sys.path[:0] = ['verify/calibration']
from tower_forcing import FX
import os
out = sys.argv[1]; args = sys.argv[2:]
# --years A-B: the year axis (default the 2008-2012 batch window); --siteinfo FILE: every calibration-grade tower of a
# towers31_prep.py site table (CALIBRATION.md 45; window with a good held-out year and a half-hourly file)
_y = '2008-2012'
if '--years' in args: k = args.index('--years'); _y = args[k + 1]; args = args[:k] + args[k + 2:]
if '--siteinfo' in args:
    k = args.index('--siteinfo'); _si = pd.read_csv(args[k + 1]); args = args[:k] + args[k + 2:]
    args = args or [r.SITE_ID for r in _si.itertuples() if str(r.HH) == 'yes' and pd.notna(r.WIN_SCORE) and float(r.WIN_SCORE) >= 2]
sites = args or open('verify/calibration/site_data/site_batches.txt').readline().split()[1].split(',')
YEARS = list(range(int(_y.split('-')[0]), int(_y.split('-')[1]) + 1)); MOL = 12.011e-6 * 1800.0   # gC per umol m-2 s-1 over a half-hour
keys = ['nee_night', 'nee_day', 'sum_night', 'sum_day', 'n_night', 'n_day', 'meas_night', 'meas_day']
A = {k: np.full((len(YEARS), 12, len(sites)), np.nan) for k in keys}
for n, s in enumerate(sites):
    f = glob.glob(f'{FX}/FLX_{s}_FLUXNET2015_FULLSET_H[HR]_*.csv')
    if not f: print(f'{s}: no half-hourly file', flush=True); continue
    d = pd.read_csv(f[0], usecols=['TIMESTAMP_START', 'NIGHT', 'NEE_VUT_REF', 'NEE_VUT_REF_QC'], na_values=[-9999])
    t = pd.to_datetime(d['TIMESTAMP_START'].astype(str), format='%Y%m%d%H%M'); d['y'], d['m'] = t.dt.year, t.dt.month
    step = (t.iloc[1] - t.iloc[0]).total_seconds() / 1800.0                      # 1 for HH, 2 for HR files
    d = d[d.y.isin(YEARS) & (d.NEE_VUT_REF_QC <= 1) & d.NEE_VUT_REF.notna()]
    for (y, m, night), g in d.groupby(['y', 'm', 'NIGHT']):
        yi, mi, tag = YEARS.index(y), m - 1, ('night' if night == 1 else 'day')
        A[f'nee_{tag}'][yi, mi, n] = g.NEE_VUT_REF.mean(); A[f'sum_{tag}'][yi, mi, n] = g.NEE_VUT_REF.sum() * MOL * step
        A[f'n_{tag}'][yi, mi, n] = len(g) * step; A[f'meas_{tag}'][yi, mi, n] = (g.NEE_VUT_REF_QC == 0).mean()
    ok = np.isfinite(A['nee_night'][:, :, n]).sum()
    yrs = [y for y in YEARS if np.isfinite(A['nee_night'][YEARS.index(y), :, n]).any()]
    if yrs:
        yi = YEARS.index(yrs[-1])
        print(f'{s}: {ok} months with night NEE over {yrs[0]}-{yrs[-1]}; {yrs[-1]} annual night {np.nansum(A["sum_night"][yi, :, n]):+.0f} '
              f'day {np.nansum(A["sum_day"][yi, :, n]):+.0f} gC m-2', flush=True)
    else:
        print(f'{s}: no month with NEE in {YEARS[0]}-{YEARS[-1]}', flush=True)
np.savez(out, sites=np.array(sites), years=np.array(YEARS), **A); print('wrote', out)
