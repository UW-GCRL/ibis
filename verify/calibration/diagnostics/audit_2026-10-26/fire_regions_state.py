"""Regional burned fraction (disturbf) of a restart vs GFED5 Norm 2003-05, humid forests and savannas. Read-only.
    python fire_regions_state.py <restart.npz> [<restart.npz> ...]"""
import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire']
import evaluate_gfed5 as eg
from ibis.io.pft_types import zones_from_classes, ZONES
first = sys.argv[1]; lat, lon, li, lo, mlat, mlon = eg._grid(first)
obs = eg.observed(eg.DEFAULT_GFED5, li, lo, mlat, mlon, range(2003, 2006))
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
klat = np.asarray(k['lat'], float); kli = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int).clip(0, 359)
klon = np.asarray(k['lon'], float); klo = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int).clip(0, 719)
zone = zones_from_classes(c[kli, klo]); Z = {n: i for i, n in enumerate(ZONES)}
REG = {'Amazon humid': (lon > -82) & (lon < -34) & (lat > -20) & (lat < 13) & (zone == Z['A_humid']),
       'Congo humid': (lon > 8) & (lon < 32) & (lat > -8) & (lat < 6) & (zone == Z['A_humid']),
       'SE Asia humid': (lon > 95) & (lon < 155) & (lat > -11) & (lat < 8) & (zone == Z['A_humid']),
       'S Am savanna': (lon > -82) & (lon < -34) & (lat > -25) & (lat < 13) & (zone == Z['A_savanna']),
       'Africa savanna': (lon > -20) & (lon < 45) & (lat > -25) & (lat < 15) & (zone == Z['A_savanna']),
       'boreal': zone == Z['D_subarctic'], 'temperate': np.isin(zone, [Z['C_oceanic'], Z['D_temperate'], Z['C_subtrop']]), 'global': np.ones_like(zone, bool)}
aw = np.cos(np.deg2rad(lat))
def wm(x, m): ok = m & np.isfinite(x); return float((x[ok] * aw[ok]).sum() / aw[ok].sum())
print(f"{'state':34s} " + ' '.join(f'{r[:10]:>10s}' for r in REG))
print(f"{'GFED5 Norm 2003-05, % yr-1':34s} " + ' '.join(f'{100 * wm(obs, m):10.2f}' for m in REG.values()))
for p in sys.argv[1:]:
    st = eg._load(p); d = np.asarray(st.vegetation.disturbf, float); fu = np.asarray(st.vegetation.fu, float)
    tag = '/'.join(p.split('/')[-4:-2]) + ':' + p.split('_')[-1][:4]
    print(f"{tag:34s} " + ' '.join(f'{100 * wm(d, m):10.2f}' for m in REG.values()) + f'   (Amazon fu {wm(fu, REG["Amazon humid"]):.2f})')
