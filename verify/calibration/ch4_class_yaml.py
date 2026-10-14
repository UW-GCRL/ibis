"""Methane sections for a parameter yaml from the per-tower methane fits (CALIBRATION.md 38).

    python verify/calibration/ch4_class_yaml.py <ch4_fit_report.csv> <base.yaml> <out.yaml> [--class WET] [--min-band 3]

`methane:` = the medians of the fitted ch4_* parameters over the towers of --class (default WET: the wetland set is what
the inundated fraction of every grid cell emits with); `methane_bands:` = the same medians by latitude band (tropical
|lat| < 30, boreal lat >= 60, temperate between; the bands of ibis.bgc.methane.biome_methane_params) for bands with at
least --min-band towers. t_ref is not fitted and stays at the module's value. Read back through build_all_params.
"""
import sys, numpy as np, pandas as pd, yaml
sys.path[:0] = ['python-port', 'verify/calibration']
rep, base, out = sys.argv[1:4]; a = sys.argv[4:]
cls = a[a.index('--class') + 1] if '--class' in a else 'WET'; nmin = int(a[a.index('--min-band') + 1]) if '--min-band' in a else 3
FIELDS = {'p:ch4_r': 'r_ch4', 'p:ch4_q10': 'q10p', 'p:ch4_ztau': 'z_tau', 'p:ch4_tauox': 'tau_oxid', 'p:ch4_zwtmax': 'z_wt_max', 'p:ch4_tref': 't_ref'}
D = pd.read_csv(rep); W = D[D.igbp == cls]; cols = [c for c in FIELDS if c in D.columns]
if '--band-csv' in a:                    # IBIS 3.1 (CALIBRATION.md 45): bands = the wetland type of the tower's Köppen zone
    zb = pd.read_csv(a[a.index('--band-csv') + 1]).set_index('site')['zone_band']
    W = W.assign(band=[zb.get(s, b) for s, b in zip(W.site, W.band)]); print('bands from', a[a.index('--band-csv') + 1])
if W.empty: raise SystemExit(f'no {cls} towers in {rep}')
t = yaml.safe_load(open(base)); t['methane'] = {FIELDS[c]: float(W[c].median()) for c in cols}
bands = {}
for b, g in W.groupby('band'):
    if len(g) >= nmin: bands[b] = {FIELDS[c]: float(g[c].median()) for c in cols}
if bands: t['methane_bands'] = bands
yaml.safe_dump(t, open(out, 'w'), sort_keys=False)
from ibis.io.physics_params import build_all_params, load_physics_params
sp, _, _ = build_all_params(4, np.zeros(4, np.float32), load_physics_params(out))
for c in cols: assert abs(float(getattr(sp.methane, FIELDS[c])) - t['methane'][FIELDS[c]]) < 1e-6
print(f'{out}: methane {t["methane"]} ({len(W)} {cls} towers); bands ' + '; '.join(f'{b} ({len(W[W.band == b])}): ' + ', '.join(f'{k} {v:.3g}' for k, v in d.items()) for b, d in bands.items()))
