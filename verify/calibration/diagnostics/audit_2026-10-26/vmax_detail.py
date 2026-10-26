import sys, yaml, numpy as np, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire']
import evaluate_gfed5 as eg
from ibis.io import pft_types as P
from ibis.io.physics_params import PFT_NAMES
run, ypath = sys.argv[1], sys.argv[2]; p = f'{run}/restart/ibis_restart_2010.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p); v = st.vegetation
plai = np.asarray(v.plai, float); ex = np.asarray(v.exist, float)
y = yaml.safe_load(open(ypath)); T = y['types']; base = y['stomata']['vmax']
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
klat = np.asarray(k['lat'], float); kli = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int).clip(0, 359)
klon = np.asarray(k['lon'], float); klo = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int).clip(0, 719)
zone = P.zones_from_classes(c[kli, klo])
for zn in ('A_humid', 'A_savanna', 'B_hot', 'D_subarctic', 'C_subtrop'):
    zi = P.ZONES.index(zn); m = zone == zi; row = P.ZONE_SLOTS[zi]
    print(f'\n{zn}: {int(m.sum())} cells')
    print('  pos  type   vmax(umol)  exist-share  leaf-area share of upper canopy')
    up = plai[m][:, :8].sum(1).sum()
    for pos in range(8):
        t = row[pos]; tid = P.IDS[t] if t >= 0 else ('OFF' if t == P.OFF else 'GENERIC')
        vmx = float(T[tid]['vmax']) * 1e6 if t >= 0 and 'vmax' in T[tid] else float(base[PFT_NAMES[pos]]) * 1e6
        print(f'  {pos:3d}  {tid:7s} {vmx:9.1f}   {np.mean(ex[m, pos] > 0.5):10.2f}   {plai[m][:, pos].sum() / max(up, 1e-9):8.3f}')
