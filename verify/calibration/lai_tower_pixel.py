"""Leaf area at the towers' own 0.05-degree pixel against the 0.5-degree cell and the value each fit held (CALIBRATION.md 49).

    /usr/bin/python3 -I verify/calibration/lai_tower_pixel.py      (the system Python: GDAL with HDF4)

GLASS LAI V61 (0.05 deg, 8-day, 2001-2010): the climatological peak (the largest of the 46 composite-day means over
the ten years) at the pixel holding the tower, and the mean of its 3 x 3 pixels; the 0.5-degree cell peak from
inputs/eval/lai/glass_v61_0.5deg_monthly_2001-2010.npz (monthly, so slightly lower than an 8-day peak; the 8-day cell
peak is also computed from the same files for a like-for-like ratio). The held value is the "leaf area:" line of each
tower's S2 fit log (verify_outputs/site_fit/t31_s2/<site>/run.log). Output: ibis_local_document/logs/lai_tower_pixel.csv.
"""
import glob, re, csv, os, numpy as np
from osgeo import gdal
B = '/shared/nas2/Hangkai/ibis/inputs/eval/lai/raw_glass_v61'
SI = '/shared/nas2/Hangkai/ibis/inputs/sites/towers31/compact/FLUXNET_siteinfo.csv'
S2 = '/shared/nas2/Hangkai/ibis/verify_outputs/site_fit/t31_s2'
OUT = '/shared/nas2/Hangkai/ibis_local_document/logs/lai_tower_pixel.csv'
rows = list(csv.DictReader(open(SI, encoding='utf-8-sig')))
held = {}
for r in rows:
    f = f"{S2}/{r['SITE_ID']}/run.log"
    if not os.path.exists(f): continue
    for l in open(f, errors='replace'):
        m = re.match(r'leaf area: (\S+) (.*?) ([0-9.]+)\s*$', l)
        if m and m.group(1) == r['SITE_ID']: held[m.group(1)] = (m.group(2), float(m.group(3))); break
sites = [r for r in rows if r['SITE_ID'] in held]
pi = np.array([int((90.0 - float(r['LOCATION_LAT'])) / 0.05) for r in sites]).clip(1, 3598)
pj = np.array([int((float(r['LOCATION_LONG']) + 180.0) / 0.05) for r in sites]).clip(1, 7198)
ci, cj = pi // 10, pj // 10
ns = len(sites); px = np.zeros((46, ns)); p9 = np.zeros((46, ns)); cell = np.zeros((46, ns)); nn = np.zeros((46, ns))
for y in range(2001, 2011):
    for f in sorted(glob.glob(f'{B}/{y}/GLASS01B01.V61.A{y}*.hdf')):
        k = (int(re.search(r'\.A\d{4}(\d{3})\.', f).group(1)) - 1) // 8
        g = gdal.Open(f); a = g.GetRasterBand(1).ReadAsArray().astype(np.float32); g = None
        a = np.where(a <= 100, a * 0.1, np.nan)
        v = a[pi, pj]; w9 = np.nanmean(np.stack([a[pi + di, pj + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)]), 0)
        c = np.array([np.nanmean(a[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10]) for i, j in zip(ci, cj)])
        ok = np.isfinite(v) & np.isfinite(w9) & np.isfinite(c)
        px[k][ok] += v[ok]; p9[k][ok] += w9[ok]; cell[k][ok] += c[ok]; nn[k][ok] += 1
    print(y, flush=True)
peak = lambda x: np.nanmax(np.where(nn > 0, x / np.maximum(nn, 1), np.nan), axis=0)
P, P9, C = peak(px), peak(p9), peak(cell)
with open(OUT, 'w', newline='') as fo:
    w = csv.writer(fo); w.writerow(['site', 'igbp', 'type', 'held_source', 'held', 'glass_pixel_peak', 'glass_3x3_peak', 'glass_cell_peak'])
    for n, r in enumerate(sites):
        w.writerow([r['SITE_ID'], r['IGBP'], r['TYPE'], held[r['SITE_ID']][0], held[r['SITE_ID']][1], round(float(P[n]), 3), round(float(P9[n]), 3), round(float(C[n]), 3)])
print('wrote', OUT, ns, 'towers')
