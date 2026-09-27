"""Daily and monthly GPP, ET and NEE of a parameter set at one tower over a year window, after site_fit's spin-up
(CALIBRATION.md 29): held-out years, phenology timing against the tower's daily GPP and PhenoCam, daily light response.

    python verify/calibration/diagnostics/site_daily_replay.py <IGBP> <params.yaml> <y0-y1> <out.npz> <SITE> [site_fit options]

The site_fit options are the run's (`--lai SITE=v --co2 series --hourly-sw ...`); the composition and leaf area are held as
in the fit. Writes gpp_daily (years, 366, towers; kgC m-2 d-1, NaN past the year's end), gpp/et/nee monthly
(years, 12, towers; gC m-2 month-1, mm, gC), years, sites, params, lai_src.
"""
import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = ['python-port', 'verify/calibration']
import site_fit
from site_transient import one_year

igbp, yml, win, out, site = sys.argv[1:6]
a = site_fit._args(['--igbp', igbp, '--params', yml, '--fit', 'vmax:temp_con_ever', '--out', '/tmp/unused', '--years', win, '--hold-last', '0', '--sites', site] + sys.argv[6:])
C = site_fit.setup(a, spin=True, verbose=True); st, key, ctx = C.base_state, C.base_key, C.ctx0
GD, GM, EM, NM = [], [], [], []
for y in site_fit.YEARS:
    st, key, d = one_year(st, key, y, ctx, 0); st = C.hold_fn(st)
    g = np.asarray(d['gpp_daily'] if 'gpp_daily' in d else d['gpp_d']); gd = np.full((366, g.shape[1]), np.nan); gd[:len(g)] = g; GD.append(gd)
    GM.append(np.asarray(d['gpp_monthly']) * 1000); EM.append(np.asarray(d['aet_monthly'])); NM.append(np.asarray(d['nee_monthly']) * 1000)
    print(f'  {y}: GPP {float(GM[-1].sum()):.0f} gC m-2, ET {float(EM[-1].sum()):.0f} mm', flush=True)
np.savez(out, gpp_daily=np.array(GD), gpp=np.array(GM), et=np.array(EM), nee=np.array(NM), years=np.array(site_fit.YEARS), sites=np.array(C.sites),
         params=yml, lai_src=np.array([C.lai_src[s] for s in C.sites]))
print('wrote', out, flush=True)
