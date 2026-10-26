"""Size of the vmax priority pick vs leaf-area mixing, by zone, from the 2010 state of a run and its yaml types.
Broadleaf big leaf: positions 0, 2, 4 (priority 0 > 2 > 4 among existing); conifer: 3 > 5. For each cell the picked vmax
vs the plai-weighted mean over the positions with leaves; summarised as the ratio weighted by broadleaf leaf area, per zone,
and the share of broadleaf leaf area that sits in positions whose own vmax differs from the picked one. Read-only."""
import sys, yaml, numpy as np, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire']
import evaluate_gfed5 as eg
from ibis.io import pft_types as P
from ibis.io.physics_params import PFT_NAMES
run, ypath = sys.argv[1], sys.argv[2]
p = f'{run}/restart/ibis_restart_2010.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p); v = st.vegetation
plai = np.asarray(v.plai, float); ex = np.asarray(v.exist, float)
y = yaml.safe_load(open(ypath)); T = y['types']; base = y['stomata']['vmax']
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
klat = np.asarray(k['lat'], float); kli = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int).clip(0, 359)
klon = np.asarray(k['lon'], float); klo = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int).clip(0, 719)
zone = P.zones_from_classes(c[kli, klo])
# per-cell vmax table from the zone slots
vm = np.zeros((len(lat), 12))
for zi, zn in enumerate(P.ZONES):
    m = zone == zi
    if not m.any(): continue
    row = P.ZONE_SLOTS[zi]
    for pos in range(12):
        t = row[pos]
        vm[m, pos] = float(T[P.IDS[t]]['vmax']) if t >= 0 and 'vmax' in T[P.IDS[t]] else float(base[PFT_NAMES[pos]])
aw = np.cos(np.deg2rad(lat))
def report(group, name):
    g = list(group); w = plai[:, g] * ex[:, g]; tot = w.sum(1)
    order = [q for q in g if q in (0, 2, 4)] if 0 in g else [q for q in g if q in (3, 5)]   # the F77 priority list
    pick = np.full(len(lat), np.nan)
    for pos in reversed(order): pick = np.where(ex[:, pos] > 0.5, vm[:, pos], pick)
    pick = np.where(np.isnan(pick), vm[:, order[-1]], pick)
    mix = np.where(tot > 1e-9, (w * vm[:, g]).sum(1) / np.maximum(tot, 1e-9), pick)
    print(f'\n{name}: leaf-area-weighted mixed/picked vmax by zone (share of the leaf area whose own vmax != picked)')
    for zi, zn in enumerate(P.ZONES):
        m = (zone == zi) & (tot > 0.05)
        if m.sum() < 20: continue
        ww = tot[m] * aw[m]; r = float((mix[m] / pick[m] * ww).sum() / ww.sum())
        diff = (np.abs(vm[:, g] - pick[:, None]) > 1e-9) & (w > 0)
        sh = float(((w * diff).sum(1)[m] * aw[m]).sum() / ww.sum())
        print(f'  {zn:12s} n {int(m.sum()):6d}  mixed/picked {r:5.2f}  share affected {sh:4.2f}')
report((0, 2, 4, 6, 1), 'broadleaf big leaf: all five positions, pick 0 > 2 > 4'); report((3, 5, 7), 'conifer big leaf: three positions, pick 3 > 5')
