"""The Amazon cells that were open at the start of the check (fu_2000 < 0.7): do they close under the new floor? fu, burn, upper LAI, wood by year. Read-only."""
import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire']
import evaluate_gfed5 as eg
from ibis.io.pft_types import zones_from_classes, ZONES
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; runs = sys.argv[1:]
p0 = f'{V}/{runs[0]}/restart/ibis_restart_2000.npz'; lat, lon, *_ = eg._grid(p0)
k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
klat = np.asarray(k['lat'], float); kli = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int).clip(0, 359)
klon = np.asarray(k['lon'], float); klo = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int).clip(0, 719)
zone = zones_from_classes(c[kli, klo]); hum = (lon > -82) & (lon < -34) & (lat > -20) & (lat < 13) & (zone == ZONES.index('A_humid'))
aw = np.cos(np.deg2rad(lat)); wm = lambda x, mm: float((x[mm] * aw[mm]).sum() / aw[mm].sum())
v0 = eg._load(p0).vegetation; fu0 = np.asarray(v0.fu, float)
groups = {'open at start (fu<0.5)': hum & (fu0 < 0.5), 'half-open (0.5-0.8)': hum & (fu0 >= 0.5) & (fu0 < 0.8), 'closed (>=0.8)': hum & (fu0 >= 0.8)}
for g, m in groups.items(): print(f'{g}: {int(m.sum())} cells, {100 * aw[m].sum() / aw[hum].sum():.1f} % of the humid Amazon')
for run in runs:
    print(f'\n== {run}: year | ' + ' | '.join(f'{g[:12]}: fu burn% LAI wood' for g in groups))
    for y in (2000, 2001, 2003, 2005, 2007, 2010):
        v = eg._load(f'{V}/{run}/restart/ibis_restart_{y}.npz').vegetation
        fu = np.asarray(v.fu, float); d = np.asarray(v.disturbf, float); pl = np.asarray(v.plai, float)[:, :8].sum(1); w = np.asarray(v.cbiow, float)[:, :8].sum(1)
        print(f'  {y} | ' + ' | '.join(f'{wm(fu, m):.2f} {100 * wm(d, m):5.1f} {wm(pl, m):4.1f} {wm(w, m):5.1f}' for m in groups.values()))
# the 2010 hot cells (> 5 %/yr) of the last run: where did they start?
v = eg._load(f'{V}/{runs[-1]}/restart/ibis_restart_2010.npz').vegetation; d = np.asarray(v.disturbf, float); hot = hum & (d > 0.05)
print(f'\n{runs[-1]} cells > 5 %/yr in 2010: {int(hot.sum())}; their fu at 2000 {wm(fu0, hot):.2f}, 2010 {wm(np.asarray(v.fu, float), hot):.2f}; centroid {np.mean(lat[hot]):.1f}N {np.mean(lon[hot]):.1f}E; fu_2000 quartiles', np.round(np.percentile(fu0[hot], [10, 50, 90]), 2))
