#!/usr/bin/env python3
"""Global map of the 41 Köppen-zone PFTs of IBIS 3.1 (CALIBRATION.md 45) with the calibration towers.

    <ibis-python> verify/calibration/diagnostics/pft_koppen_map.py <out dir>

Each 0.5-degree cell gets ONE type, its dominant one: (1) the dominant life form of the cell among trees, shrubs,
grasses, crops and wetland -- natural forms from the MODIS-derived plant-type fractions of the CLM surface data
(PCT_NAT_PFT x PCT_NATVEG), crops from the ESA CCI cropland fraction the model uses, wetland from the WAD2M
2000-2020 mean inundated fraction; (2) within the life form, the dominant leaf type (needleleaf / broadleaf,
evergreen / deciduous; C3 / C4); (3) the Köppen zone of the cell (Beck et al. 2023, 1991-2020) -> the type, with
the zone tables of pft_koppen_design.py. Cells whose largest share is bare ground are drawn as barren, Köppen EF
and glacier cells as ice. Towers: circles = FLUXNET2015, stars = FLUXNET-CH4, filled with the colour of the type
they calibrate. C3 and C4 crops share a colour: no crop-type map is on disk.
"""
import sys, os, csv
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

R = '/shared/nas2/Hangkai/ibis-port'
sys.path.insert(0, f'{R}/verify/calibration')
import pft_koppen_design as D
from landmask import true_landfrac

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(OUT, exist_ok=True)
LAT = np.arange(89.75, -90, -0.5); LON = np.arange(-179.75, 180, 0.5)

# ---- inputs, all on the north-to-south 0.5-degree grid
sd = xr.open_dataset('/shared/nas2/Hangkai/ELM Edge Effects/inputdata/surfdata_0.5x0.5_CH_v1_all_forests.nc')
pp = np.asarray(sd['PCT_NAT_PFT'].values, float) / 100.0          # (17, lat, lon), share of the natural-veg landunit
nat = np.asarray(sd['PCT_NATVEG'].values, float) / 100.0
glac = np.asarray(sd['PCT_GLACIER'].values, float) / 100.0
if float(sd['LATIXY'].values[0, 0]) < float(sd['LATIXY'].values[-1, 0]):
    pp = pp[:, ::-1, :]; nat = nat[::-1, :]; glac = glac[::-1, :]
crop = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz')['cropfrac'].astype(float)
with xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc') as w:
    fw = np.asarray(w['Fw'].values, float)
    fw = np.where(fw < 0, 0.0, fw); wet = np.nanmean(fw, axis=0)
    if float(w['lat'].values[0]) < float(w['lat'].values[-1]): wet = wet[::-1, :]
wet = np.nan_to_num(wet)
kop = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz')['classes'][3].astype(int)
lf = true_landfrac()

# ---- cell shares by form (natural forms scaled to the cell's natural-vegetation share)
n = lambda *ix: nat * pp[list(ix)].sum(axis=0)
forms = {
    'tree_ne': n(1, 2), 'tree_nd': n(3), 'tree_be': n(4, 5), 'tree_bd': n(6, 7, 8),
    'shrub_e': n(9), 'shrub_d': n(10, 11), 'grass_c3': n(12, 13), 'grass_c4': n(14),
    'crop': crop, 'wetland': wet, 'bare': nat * pp[0],
}
life = {'tree': ['tree_ne', 'tree_nd', 'tree_be', 'tree_bd'], 'shrub': ['shrub_e', 'shrub_d'], 'grass': ['grass_c3', 'grass_c4'],
        'crop': ['crop'], 'wetland': ['wetland'], 'bare': ['bare']}
L = list(life); lshare = np.stack([sum(forms[f] for f in life[l]) for l in L])
dom_life = np.argmax(lshare, axis=0)

# ---- (form, zone) -> type: the tower-assignment tables of pft_koppen_design, by form
zone_of = np.vectorize(lambda c: D.ZONE.get(D.LEG.get(int(c), 'none'), 'none'))(kop)
def type_for(form, zone):
    ig = {'tree_ne': 'ENF', 'tree_be': 'EBF', 'tree_bd': 'DBF', 'tree_nd': 'DNF', 'shrub_e': 'OSH', 'shrub_d': 'OSH',
          'grass_c3': 'GRA', 'grass_c4': 'GRA', 'wetland': 'WET', 'crop': 'CRO'}[form]
    if form == 'tree_ne' and zone == 'B_hot': return 'T21'
    if form == 'tree_be' and zone == 'B_cold': return 'T05'
    if form == 'tree_be' and zone in ('D_subarctic', 'D_extreme', 'E_tundra'): return 'T17' if zone == 'D_subarctic' else 'T20'
    ids = D.assign(ig, zone, '')
    return ids[0] if ids else None
IDS = [p[0] for p in D.PFTS]; idx = {p: i for i, p in enumerate(IDS)}
BARREN, ICE, NONE = len(IDS), len(IDS) + 1, -1
BARE_DOMINANT = 0.6      # a cell is barren only when bare ground is above 60 % (MODIS bare share wins in sparse steppe otherwise)
WET_DOMINANT = 0.25      # a cell is wetland when its WAD2M mean inundated fraction is above 25 %
cat = np.full((360, 720), NONE, int)
land = lf > 0.05
for i, j in zip(*np.where(land)):
    z = zone_of[i, j]; c = kop[i, j]
    if c == 30 or glac[i, j] > 0.5: cat[i, j] = ICE; continue
    if wet[i, j] > WET_DOMINANT and z != 'none':
        cat[i, j] = idx[type_for('wetland', z)]; continue
    if forms['bare'][i, j] > BARE_DOMINANT or z == 'none' or lshare[:-1, i, j].sum() < 0.05: cat[i, j] = BARREN; continue
    l = L[int(np.argmax(lshare[:-1, i, j]))]                     # dominant VEGETATED life form
    f = max(life[l], key=lambda ff: forms[ff][i, j])
    t = type_for(f, z)
    cat[i, j] = idx[t] if t else BARREN

# ---- colours: hue families by climate, shades by form
PAL = {
 'T01': '#00441b', 'T02': '#238b45', 'T03': '#b5c74a', 'T04': '#4d9221', 'T05': '#c9a464',
 'T06': '#1d6996', 'T07': '#2fa4a0', 'T08': '#8fd1cf', 'T09': '#7f3b08', 'T10': '#b8732a', 'T11': '#e0b779',
 'T12': '#7fb2dc', 'T13': '#253494', 'T14': '#4f7fc4', 'T15': '#a785c4', 'T16': '#542788', 'T21': '#bcb3d9',
 'T17': '#67001f', 'T18': '#e3735e', 'T19': '#f6b89c', 'T20': '#b2182b',
 'S01': '#fdae61', 'S02': '#cdbb94', 'S03': '#a0522d', 'S04': '#fee08b', 'S05': '#f4874d', 'S06': '#efe6c8', 'S07': '#e7a23c',
 'G01': '#f6d40f', 'G02': '#fbf0a4', 'G03': '#d9b26f', 'G04': '#c4e17f', 'G05': '#e8f3b5', 'G06': '#8fcb6a', 'G07': '#5f9e5f', 'G08': '#d4ddb8',
 'W01': '#00a6b8', 'W02': '#5dc8d8', 'W03': '#a8e3ea',
 'C01': '#de77ae', 'C02': '#de77ae',
}
cols = [PAL[p] for p in IDS] + ['#cfcfcf', '#f4f4f4']
cmap = ListedColormap(cols); norm = BoundaryNorm(np.arange(-0.5, len(cols) + 0.5), len(cols))

# ---- area per type (dominant-cell area, Mkm2)
area = (6371.0 ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * lf
amk = {p: float(area[cat == idx[p]].sum()) / 1e6 for p in IDS}

# ---- towers and the type each calibrates (first listed, from the design table)
design = list(csv.DictReader(open(f'{R}/docs/pft_koppen_design.csv')))
site_type = {}
for r in design:
    for s in r['towers'].split():
        site_type.setdefault(s.split('(')[0], r['id'])
ntw = {r['id']: int(r['n_towers']) for r in design}
old = []
for r in csv.DictReader(open(f'{R}/verify/calibration/site_data/site_inventory.csv')):
    old.append((r['site'], float(r['lat']), float(r['lon'])))
seen = {s for s, _, _ in old}
for r in csv.DictReader(open('/shared/nas2/Hangkai/ibis/inputs/sites/methane_towers/ch4_sites.csv')):
    if r['SITE_ID'] not in seen: old.append((r['SITE_ID'], float(r['LOCATION_LAT']), float(r['LOCATION_LONG']))); seen.add(r['SITE_ID'])
new, nnew = [], {p: 0 for p in IDS}
for r in csv.DictReader(open('/shared/nas2/Hangkai/ibis/inputs/sites/new_sites_inventory.csv')):
    if r['status'] != 'NEW' or r['site'] in seen: continue
    t = r['pft_types'].replace(';', ' ').replace(',', ' ').split()
    if not t or int(float(r['n_years_nee50'] or 0)) < 1: continue
    seen.add(r['site']); new.append((r['site'], float(r['lat']), float(r['lon']))); site_type[r['site']] = t[0]
    for p in t:
        if p in nnew: nnew[p] += 1
fx, ch = old, new
colour = lambda s: PAL.get(site_type.get(s, ''), '#ffffff')

# ---- figure
fig = plt.figure(figsize=(22, 14.2))
ax = fig.add_axes([0.01, 0.335, 0.98, 0.64], projection=ccrs.Robinson())
ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
masked = np.ma.masked_where(cat == NONE, cat)
ax.pcolormesh(np.arange(-180, 180.5, 0.5), np.arange(90, -90.5, -0.5), masked, cmap=cmap, norm=norm,
              transform=ccrs.PlateCarree(), shading='flat', rasterized=True)
ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4, edgecolor='#444444')
ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.2, edgecolor='#888888')
ax.scatter([lo for _, _, lo in fx], [la for _, la, _ in fx], s=40, marker='o', c=[colour(s) for s, _, _ in fx],
           edgecolors='black', linewidths=0.8, transform=ccrs.PlateCarree(), zorder=5)
ax.scatter([lo for _, _, lo in ch], [la for _, la, _ in ch], s=120, marker='*', c=[colour(s) for s, _, _ in ch],
           edgecolors='black', linewidths=0.7, transform=ccrs.PlateCarree(), zorder=6)
ax.set_title('IBIS 3.1 Köppen-zone plant functional types (dominant type per 0.5° cell) and the calibration towers',
             fontsize=17, loc='left', pad=10)

# legend: one block per life form
groups = [('Trees', [p for p in IDS if p.startswith('T')]), ('Shrubs', [p for p in IDS if p.startswith('S')]),
          ('Grasses', [p for p in IDS if p.startswith('G')]), ('Wetlands', [p for p in IDS if p.startswith('W')]),
          ('Crops', ['C01'])]
name = {p[0]: p[1] for p in D.PFTS}
def label(p):
    if p == 'C01':
        b = ntw['C01'] + ntw['C02']; a = b + nnew['C01'] + nnew['C02']
        return f'C01/C02 managed crops, C3 and C4 ({b} → {a}; {amk["C01"] + amk["C02"]:.1f})'
    nm = name[p] if len(name[p]) <= 50 else name[p][:48] + '…'
    return f'{p} {nm} ({ntw[p]} → {ntw[p] + nnew[p]}; {amk[p]:.1f})'
x0 = [0.015, 0.265, 0.515, 0.765]
blocks = [(groups[0][0] + ' (1/2)', groups[0][1][:11]), (groups[0][0] + ' (2/2)', groups[0][1][11:]),
          (groups[1][0], groups[1][1]), (groups[2][0], groups[2][1])]
for k, (title, members) in enumerate(blocks):
    h = [Patch(facecolor=PAL[p], edgecolor='#555555', linewidth=0.4, label=label(p)) for p in members]
    if k == 2:   # shrubs column also carries wetlands and crops
        h += [Patch(facecolor='none', edgecolor='none', label='')]
        h += [Patch(facecolor=PAL[p], edgecolor='#555555', linewidth=0.4, label=label(p)) for p in groups[3][1] + groups[4][1]]
    if k == 3:
        h += [Patch(facecolor='none', edgecolor='none', label=''),
              Patch(facecolor='#cfcfcf', edgecolor='#555555', linewidth=0.4, label=f'barren / sparse ({float(area[cat == BARREN].sum()) / 1e6:.1f} Mkm²)'),
              Patch(facecolor='#f4f4f4', edgecolor='#555555', linewidth=0.4, label='ice'),
              Line2D([], [], marker='o', color='none', markerfacecolor='#ffffff', markeredgecolor='black', markersize=9, label=f'tower on disk before 13 Oct: FLUXNET2015, FLUXNET-CH4 ({len(fx)})'),
              Line2D([], [], marker='*', color='none', markerfacecolor='#ffffff', markeredgecolor='black', markersize=14, label=f'tower acquired 13 Oct: ICOS, FLUXNET v1.3 ({len(ch)})')]
    t = 'Trees' if k < 2 else ('Shrubs, wetlands, crops' if k == 2 else 'Grasses, other')
    lg = fig.legend(handles=h, loc='upper left', bbox_to_anchor=(x0[k], 0.33), frameon=False, fontsize=10.5,
                    title=t + ('  (towers before → after 13 Oct; Mkm² where dominant)' if k == 0 else ''), title_fontsize=12.5, alignment='left',
                    handlelength=1.6, handleheight=1.1, labelspacing=0.55)
fig.text(0.015, 0.012, 'Dominant type per cell: life form from MODIS plant-type fractions (CLM surface data), ESA CCI cropland and WAD2M wetland; '
         'leaf type within it; Köppen zone (Beck et al. 2023, 1991-2020). Barren = bare ground above 60 % of the cell; wetland = WAD2M inundated fraction above 25 %. Markers are filled with the colour of the type the tower calibrates; '
         'acquired towers are those with at least one year of 50 % NEE coverage. C3 and C4 crops share a colour (no crop-type map on disk). Cell areas use the true land fraction.', fontsize=9.5, color='#444444', wrap=True)
fig.savefig(f'{OUT}/PFT31_koppen_types_map.png', dpi=150)
fig.savefig(f'{OUT}/PFT31_koppen_types_map.pdf')
print('wrote', f'{OUT}/PFT31_koppen_types_map.png')
with open(f'{OUT}/PFT31_dominant_area.csv', 'w') as f:
    f.write('id,name,dominant_area_Mkm2,n_towers\n')
    for p in IDS: f.write(f'{p},"{name[p]}",{amk[p]:.3f},{ntw[p]}\n')
    f.write(f'barren,,{float(area[cat == BARREN].sum()) / 1e6:.3f},\nice,,{float(area[cat == ICE].sum()) / 1e6:.3f},\n')
np.savez_compressed(f'{OUT}/PFT31_dominant_type.npz', cat=cat.astype(np.int16), ids=np.array(IDS + ['barren', 'ice']), lat=LAT, lon=LON)
with open(f'{OUT}/PFT31_tower_counts.csv', 'w') as f:
    f.write('id,name,towers_before,towers_acquired,towers_after\n')
    for p in IDS: f.write(f'{p},"{name[p]}",{ntw[p]},{nnew[p]},{ntw[p] + nnew[p]}\n')
print('towers: before', len(fx), 'acquired', len(ch)); print('types with 0 towers after:', [p for p in IDS if ntw[p] + nnew[p] == 0]); print('types with < 3 after:', [(p, ntw[p] + nnew[p]) for p in IDS if ntw[p] + nnew[p] < 3])
types_mapped = sorted({IDS[c] for c in np.unique(cat) if 0 <= c < len(IDS)})
print('types with mapped extent:', len(types_mapped), '; types never dominant:', [p for p in IDS if p not in types_mapped])
