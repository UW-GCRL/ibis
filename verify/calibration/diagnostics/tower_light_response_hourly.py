"""Hourly light response of a tower from its FLUXNET2015 FULLSET HH/HR file (CALIBRATION.md 27).

    python verify/calibration/diagnostics/tower_light_response_hourly.py <SITE> <out_dir> [<months, e.g. 7 or 6,7,8>]

GPP is the mean of the NT and DT partitions, hours with NEE_VUT_REF_QC = 0 (measured), incident PPFD_IN (SW_IN_F x the
site's own PPFD/SW ratio where PPFD_IN is missing). Per year and for the chosen months: a rectangular hyperbola
GPP = alpha I Amax / (alpha I + Amax) fitted against incident PPFD (alpha mol CO2 per mol incident photon; Amax the
asymptote), the mean GPP in hours above 1500 umol m-2 s-1, and the mean GPP in 200-umol bins. Writes
<out_dir>/<SITE>_hourly.csv (year, doy, hour, ppfd, SW_IN_F, TA_F, VPD_F, gpp, QC) for model_hourly_lr.py, and the
per-year fits and binned means as csv.
"""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import curve_fit
FX = '/shared/nas2/Hangkai/ibis/inputs/sites/FLUXNET2015_FULLSET/halfhourly'
site, out = sys.argv[1], sys.argv[2]; months = [int(m) for m in (sys.argv[3] if len(sys.argv) > 3 else '7').split(',')]
f = glob.glob(f'{FX}/FLX_{site}_FLUXNET2015_FULLSET_H[HR]_*.csv'); assert f, f'no HH/HR file for {site} in {FX}'
cols = ['TIMESTAMP_START', 'TA_F', 'SW_IN_F', 'PPFD_IN', 'VPD_F', 'NEE_VUT_REF_QC', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF']
d = pd.read_csv(f[0], usecols=lambda c: c in cols, na_values=[-9999, -9999.0])
t = pd.to_datetime(d.TIMESTAMP_START.astype(str), format='%Y%m%d%H%M')
d['year'], d['month'], d['hour'], d['doy'] = t.dt.year, t.dt.month, t.dt.hour, t.dt.dayofyear
d['gpp'] = d[['GPP_NT_VUT_REF', 'GPP_DT_VUT_REF']].mean(axis=1)
if 'PPFD_IN' not in d: d['PPFD_IN'] = np.nan
ok = d.PPFD_IN.notna() & (d.SW_IN_F > 50); ratio = float((d.PPFD_IN[ok] / d.SW_IN_F[ok]).median()) if ok.sum() > 100 else 2.0
d['ppfd'] = d.PPFD_IN.where(d.PPFD_IN.notna(), d.SW_IN_F * ratio)
print(f'{site}: PPFD_IN present in {d.PPFD_IN.notna().mean():.0%} of hours; PPFD/SW ratio {ratio:.2f} umol J-1; half-hourly' if (t.diff().dt.total_seconds().median() < 3600) else f'{site}: PPFD_IN present in {d.PPFD_IN.notna().mean():.0%} of hours; PPFD/SW ratio {ratio:.2f} umol J-1; hourly')
rh = lambda I, a, amax: a * I * amax / (a * I + amax)
BINS = np.arange(0, 2201, 200); rows = []; binned = {}
for y in sorted(d.year.unique()):
    sel = (d.year == y) & d.month.isin(months) & (d.NEE_VUT_REF_QC == 0) & d.gpp.notna() & (d.ppfd > 5) & (d.hour >= 5) & (d.hour <= 20)
    I, G = d.ppfd[sel].values, d.gpp[sel].values
    if len(I) < 60: continue
    try: (a, amax), _ = curve_fit(rh, I, G, p0=[0.04, 30], bounds=([0, 1], [0.3, 200]))
    except Exception: a = amax = np.nan
    hi = I > 1500; ib = np.digitize(I, BINS) - 1
    rows.append(dict(year=y, n=len(I), alpha=a, amax=amax, gpp_above_1500=G[hi].mean() if hi.sum() > 5 else np.nan, gpp_p95=np.percentile(G, 95), ta=d.TA_F[sel].mean(), vpd=d.VPD_F[sel].mean()))
    binned[y] = [G[ib == b].mean() if (ib == b).sum() >= 5 else np.nan for b in range(len(BINS) - 1)]
T = pd.DataFrame(rows).set_index('year'); pd.set_option('display.width', 200)
print(f'months {months}: per-year rectangular-hyperbola fits (alpha per incident photon) and bright-hour GPP, umol m-2 s-1'); print(T.round(3).to_string())
print('medians: alpha %.4f  Amax %.1f  GPP above 1500 %.1f' % tuple(T[['alpha', 'amax', 'gpp_above_1500']].median()))
B = pd.DataFrame(binned, index=[f'{a_}-{b_}' for a_, b_ in zip(BINS[:-1], BINS[1:])]).T; print('\nbinned GPP by incident PPFD:'); print(B.round(1).to_string())
T.to_csv(f'{out}/{site}_lr_fits.csv'); B.to_csv(f'{out}/{site}_lr_binned.csv')
d[['year', 'doy', 'hour', 'ppfd', 'SW_IN_F', 'TA_F', 'VPD_F', 'gpp', 'NEE_VUT_REF_QC']].to_csv(f'{out}/{site}_hourly.csv', index=False); print('wrote', f'{out}/{site}_hourly.csv')
