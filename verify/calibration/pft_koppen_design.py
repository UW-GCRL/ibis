#!/usr/bin/env python3
"""Köppen-zone PFT list for IBIS 3.1 and the towers that can calibrate each type (CALIBRATION.md 45).

40 types. T20 (northern open taiga conifer) was merged into T17 on 2026-10-13 (Hangkai: no towers; T17 is the
nearest type, same parent and form, adjacent zone); IDs are kept stable, so there is no T20.

    python3 verify/calibration/pft_koppen_design.py [<out csv>]

Each PFT = growth form x Köppen zone; `parent` is the IBIS 2.6b4 PFT (1-12) whose parameters a type inherits
until its own towers are fitted. Towers: the 206 FLUXNET2015 sites of site_inventory.csv, the 107 FLUXNET-CH4
sites, and the crop towers; each is assigned by its IGBP class and the Beck et al. 2023 Köppen class at 1 km
(1991-2020). Mixed forest (MF) counts under both its conifer and its deciduous type.
"""
import csv, sys, os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

KOP = '/shared/nas/Public/Climate_Biome/Koppen_Beck2023/1991_2020/koppen_geiger_0p00833333.tif'
LEG = {1:'Af',2:'Am',3:'Aw',4:'BWh',5:'BWk',6:'BSh',7:'BSk',8:'Csa',9:'Csb',10:'Csc',11:'Cwa',12:'Cwb',13:'Cwc',14:'Cfa',15:'Cfb',16:'Cfc',
       17:'Dsa',18:'Dsb',19:'Dsc',20:'Dsd',21:'Dwa',22:'Dwb',23:'Dwc',24:'Dwd',25:'Dfa',26:'Dfb',27:'Dfc',28:'Dfd',29:'ET',30:'EF'}
ZONE = {'Af':'A_humid','Am':'A_humid','Aw':'A_savanna','BWh':'B_hot','BSh':'B_hot','BWk':'B_cold','BSk':'B_cold',
        'Csa':'C_med','Csb':'C_med','Csc':'C_med','Cwa':'C_subtrop','Cfa':'C_subtrop','Cwb':'C_highland','Cwc':'C_highland',
        'Cfb':'C_oceanic','Cfc':'C_oceanic','Dfa':'D_temperate','Dfb':'D_temperate','Dwa':'D_temperate','Dwb':'D_temperate',
        'Dsa':'D_temperate','Dsb':'D_temperate','Dfc':'D_subarctic','Dwc':'D_subarctic','Dsc':'D_subarctic',
        'Dfd':'D_extreme','Dwd':'D_extreme','Dsd':'D_extreme','ET':'E_tundra','EF':'E_ice'}
# id, name, form, zones, parent IBIS PFT, canopy
PFTS = [
 ('T01','tropical humid broadleaf evergreen tree','tree_be',['A_humid'],1,'upper'),
 ('T02','tropical seasonal semi-deciduous tree','tree_bd',['A_humid','A_savanna'],2,'upper'),
 ('T03','tropical savanna tree (drought-deciduous, fire-adapted)','tree_bd',['A_savanna','B_hot'],2,'upper'),
 ('T04','tropical highland broadleaf evergreen tree','tree_be',['C_highland'],3,'upper'),
 ('T05','hot-arid woodland tree (acacia, mulga)','tree_be',['B_hot'],2,'upper'),
 ('T06','subtropical broadleaf evergreen tree','tree_be',['C_subtrop'],3,'upper'),
 ('T07','subtropical conifer (southern pines)','tree_ne',['C_subtrop','C_highland','A_humid'],4,'upper'),
 ('T08','subtropical broadleaf deciduous tree','tree_bd',['C_subtrop','C_highland'],5,'upper'),
 ('T09','Mediterranean sclerophyll evergreen tree','tree_be',['C_med'],3,'upper'),
 ('T10','Mediterranean conifer','tree_ne',['C_med'],4,'upper'),
 ('T11','Mediterranean broadleaf deciduous tree','tree_bd',['C_med'],5,'upper'),
 ('T12','oceanic temperate broadleaf deciduous tree','tree_bd',['C_oceanic'],5,'upper'),
 ('T13','oceanic temperate conifer evergreen tree','tree_ne',['C_oceanic'],4,'upper'),
 ('T14','oceanic temperate broadleaf evergreen tree','tree_be',['C_oceanic'],3,'upper'),
 ('T15','continental temperate broadleaf deciduous tree','tree_bd',['D_temperate','B_cold'],5,'upper'),
 ('T16','continental temperate conifer evergreen tree','tree_ne',['D_temperate'],4,'upper'),
 ('T17','subarctic conifer evergreen tree (closed and open taiga)','tree_ne',['D_subarctic','D_extreme','E_tundra'],6,'upper'),
 ('T18','subarctic broadleaf deciduous tree (birch, aspen)','tree_bd',['D_subarctic','D_extreme'],7,'upper'),
 ('T19','subarctic conifer deciduous tree (larch)','tree_nd',['D_subarctic','D_extreme'],8,'upper'),
 ('T21','cold-steppe conifer woodland (juniper, pinyon)','tree_ne',['B_cold'],4,'upper'),
 ('S01','hot-arid evergreen shrub','shrub_e',['B_hot'],9,'lower'),
 ('S02','cold-desert and steppe shrub (sagebrush)','shrub_d',['B_cold'],10,'lower'),
 ('S03','Mediterranean shrub (maquis, chaparral)','shrub_e',['C_med'],9,'lower'),
 ('S04','temperate deciduous shrub','shrub_d',['C_subtrop','C_oceanic','D_temperate','C_highland'],10,'lower'),
 ('S05','boreal evergreen shrub (ericaceous)','shrub_e',['D_subarctic','D_extreme'],9,'lower'),
 ('S06','tundra dwarf shrub','shrub_d',['E_tundra'],10,'lower'),
 ('S07','tropical shrub and understorey','shrub_e',['A_humid','A_savanna'],9,'lower'),
 ('G01','tropical C4 grass','grass_c4',['A_humid','A_savanna'],11,'lower'),
 ('G02','hot-arid C4 grass','grass_c4',['B_hot'],11,'lower'),
 ('G03','subtropical C4 grass','grass_c4',['C_subtrop','C_highland'],11,'lower'),
 ('G04','cold-steppe C3 grass','grass_c3',['B_cold'],12,'lower'),
 ('G05','Mediterranean annual C3 grass','grass_c3',['C_med'],12,'lower'),
 ('G06','temperate C3 grass','grass_c3',['C_oceanic','D_temperate'],12,'lower'),
 ('G07','subarctic C3 grass and forb','grass_c3',['D_subarctic','D_extreme'],12,'lower'),
 ('G08','tundra graminoid (sedge, moss)','grass_c3',['E_tundra'],12,'lower'),
 ('W01','tropical and subtropical wetland vegetation','wetland',['A_humid','A_savanna','B_hot','C_subtrop','C_highland'],11,'lower'),
 ('W02','temperate wetland graminoid','wetland',['C_med','C_oceanic','D_temperate','B_cold'],12,'lower'),
 ('W03','boreal and arctic sedge-moss wetland','wetland',['D_subarctic','D_extreme','E_tundra'],12,'lower'),
 ('C01','C3 crop (managed)','crop',['*'],12,'lower'),
 ('C02','C4 crop (managed)','crop',['*'],11,'lower'),
]
C4_CROPS = {'US-Ne1','US-Ne2','US-Ne3','IT-BCi','US-Tw2','US-ARM'}

def assign(igbp, zone, site):
    """IGBP class x zone -> list of PFT ids a tower calibrates."""
    z = zone
    if igbp == 'CRO': return ['C02' if site in C4_CROPS else 'C01']
    if igbp == 'WET':
        return ['W01'] if z in ('A_humid','A_savanna','B_hot','C_subtrop','C_highland') else ['W03'] if z in ('D_subarctic','D_extreme','E_tundra') else ['W02']
    if igbp == 'GRA':
        return {'A_humid':['G01'],'A_savanna':['G01'],'B_hot':['G02'],'C_subtrop':['G03'],'C_highland':['G03'],'B_cold':['G04'],'C_med':['G05'],
                'C_oceanic':['G06'],'D_temperate':['G06'],'D_subarctic':['G07'],'D_extreme':['G07'],'E_tundra':['G08']}.get(z, [])
    if igbp in ('OSH','CSH'):
        return {'B_hot':['S01'],'B_cold':['S02'],'C_med':['S03'],'C_subtrop':['S04'],'C_oceanic':['S04'],'D_temperate':['S04'],'C_highland':['S04'],
                'D_subarctic':['S05'],'D_extreme':['S05'],'E_tundra':['S06'],'A_humid':['S07'],'A_savanna':['S07']}.get(z, [])
    if igbp in ('SAV','WSA'):
        return {'A_humid':['T02'],'A_savanna':['T03'],'B_hot':['T03'],'C_med':['T09'],'B_cold':['T21'],'C_subtrop':['T08'],'C_highland':['T04'],
                'D_temperate':['T15'],'C_oceanic':['T12']}.get(z, [])
    ne = {'A_humid':'T07','A_savanna':'T07','C_subtrop':'T07','C_highland':'T07','C_med':'T10','C_oceanic':'T13','D_temperate':'T16',
          'D_subarctic':'T17','D_extreme':'T17','B_cold':'T21','B_hot':'T05','E_tundra':'T17'}
    be = {'A_humid':'T01','A_savanna':'T02','B_hot':'T05','C_med':'T09','C_subtrop':'T06','C_highland':'T04','C_oceanic':'T14','D_temperate':'T06'}
    bd = {'A_humid':'T02','A_savanna':'T03','B_hot':'T03','B_cold':'T15','C_med':'T11','C_subtrop':'T08','C_highland':'T08','C_oceanic':'T12',
          'D_temperate':'T15','D_subarctic':'T18','D_extreme':'T18','E_tundra':'T18'}
    if igbp == 'ENF': return [ne[z]] if z in ne else []
    if igbp == 'EBF': return [be[z]] if z in be else []
    if igbp == 'DBF': return [bd[z]] if z in bd else []
    if igbp == 'DNF': return ['T19']
    if igbp == 'MF': return [p for p in (ne.get(z), bd.get(z)) if p]
    return []

def main(out):
    img = np.array(Image.open(KOP))                      # (21600, 43200) uint8, north to south
    def kclass(lat, lon):
        i = min(int((90.0 - lat) / (1 / 120.0)), img.shape[0] - 1); j = min(int((lon + 180.0) / (1 / 120.0)), img.shape[1] - 1)
        v = int(img[i, j])
        if v == 0:                                        # coast: nearest classified pixel within 5 km
            w = img[max(i - 6, 0):i + 7, max(j - 6, 0):j + 7]; vals = w[w > 0]
            v = int(np.bincount(vals).argmax()) if vals.size else 0
        return LEG.get(v, 'none')
    towers = []
    R = os.path.dirname(os.path.abspath(__file__))
    for r in csv.DictReader(open(f'{R}/site_data/site_inventory.csv')):
        towers.append((r['site'], r['igbp'], float(r['lat']), float(r['lon']), 'FLUXNET2015', int(float(r['nyr'] or 0))))
    for r in csv.DictReader(open('/shared/nas2/Hangkai/ibis/inputs/sites/methane_towers/ch4_sites.csv')):
        towers.append((r['SITE_ID'], r['IGBP'], float(r['LOCATION_LAT']), float(r['LOCATION_LONG']), 'FLUXNET-CH4', 0))
    seen = set(); towers = [t for t in towers if not (t[0] in seen or seen.add(t[0]))]   # one entry per site (GF-Guy etc. are in both lists)
    per = {p[0]: [] for p in PFTS}; unassigned = []
    for s, ig, la, lo, src, nyr in towers:
        k = kclass(la, lo); z = ZONE.get(k, 'none'); ids = assign(ig, z, s)
        if not ids: unassigned.append((s, ig, k)); continue
        for p in ids: per[p].append(f'{s}({k})')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'name', 'form', 'canopy', 'zones', 'parent_ibis_pft', 'n_towers', 'towers'])
        for pid, name, form, zones, parent, canopy in PFTS:
            w.writerow([pid, name, form, canopy, ' '.join(zones), parent, len(per[pid]), ' '.join(per[pid])])
    print(f'{len(PFTS)} PFTs; {len(towers)} towers; {len(unassigned)} unassigned: {unassigned[:12]}')
    print(f"{'id':4s} {'n':>3s}  {'name':52s} parent  towers")
    for pid, name, form, zones, parent, canopy in PFTS:
        t = per[pid]; print(f'{pid:4s} {len(t):3d}  {name:52s} {parent:2d}      {" ".join(x.split("(")[0] for x in t[:8])}{" ..." if len(t) > 8 else ""}')
    print('types with >= 3 towers:', sum(1 for p in PFTS if len(per[p[0]]) >= 3), '; with 0:', [p[0] for p in PFTS if not per[p[0]]])

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'docs/pft_koppen_design.csv')
