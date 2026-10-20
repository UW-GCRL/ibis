"""GSWP3 at the tower cells of a site_fit command file, for running the harness on a machine without the NAS
(CALIBRATION.md 49; served there by tower_forcing.CellCacheReader through IBIS_GSWP3_CACHE).

    python verify/calibration/gswp3_cell_cache.py <commands file> <out.npz>

Cells: each command's --sites at the 0.5-degree cell (fit_carbon.LATV / LONV, as site_fit). Years: every year of each
command's --years clipped to 1901-2010 (TowerForcingReader uses 2010's fields after 2010). Keys: '<year>_<month0>_<var>'
(ndays, ncell) float32, '<year>_<month0>_ndays', lat_i, lon_j, variables.
"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, 'verify/calibration'); sys.path.insert(0, 'python-port')
from fit_carbon import LATV, LONV, GSWP3
from ibis.io.gswp3_reader import GSWP3Reader
cmds, out = sys.argv[1], sys.argv[2]
meta = pd.read_csv('/shared/nas2/Hangkai/ibis/inputs/sites/towers31/compact/FLUXNET_siteinfo.csv', encoding='utf-8-sig').set_index('SITE_ID')
cells, years = set(), set()
for line in open(cmds):
    m = re.search(r'--sites (\S+)', line); y = re.search(r'--years (\d+)-(\d+)', line)
    if not m or not y: continue
    for s in m.group(1).split(','):
        cells.add((int(np.argmin(np.abs(LATV - meta.loc[s, 'LOCATION_LAT']))), int(np.argmin(np.abs(LONV - meta.loc[s, 'LOCATION_LONG'])))))
    years.update(min(max(v, 1901), 2010) for v in range(int(y.group(1)), int(y.group(2)) + 1))
cells = sorted(cells); I = np.array([c[0] for c in cells]); J = np.array([c[1] for c in cells])
rd = GSWP3Reader(GSWP3); res = {'lat_i': I, 'lon_j': J, 'variables': np.array(rd.variables)}
for yr in sorted(years):
    for mo in range(12):
        g = rd.read_month(yr, mo, I, J)
        for v in rd.variables: res[f'{yr}_{mo}_{v}'] = np.asarray(g[v], np.float32)
        res[f'{yr}_{mo}_ndays'] = np.array(int(g['ndays']))
    print(yr, flush=True)
np.savez_compressed(out, **res); print('wrote', out, len(cells), 'cells', sorted(years))
