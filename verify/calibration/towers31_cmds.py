#!/usr/bin/env python3
"""Per-tower fit commands of the IBIS 3.1 campaign (CALIBRATION.md 45): stage S2 (GPP and ET) for every
calibration-grade natural-vegetation tower of towers31_prep.py, in the protocol of the 62 fits of sections 31-36.

    IBIS_FLUX_DIR=<towers31>/compact python verify/calibration/towers31_cmds.py <out root> [--types T19,S06] > s2_cmds.txt

One line per tower: `SITE IGBP <site_fit options>`. Fitted set from the tower's prescribed composition (site_run.prescribe):
vmax and the Ball-Berry slope of each prescribed PFT; the acclimation constants of evergreen conifers; the upper-canopy
phenology (gdd_thresh_u, gdd_ramp_u, sen_ycrit_u) where a cold-deciduous tree is present; the lower-canopy onset
(gdd_thresh_l, tthresh_l_c) where shrubs or grass are; the drought trigger where a drought-deciduous tree, a C4 grass or a
shrub is; the soil-water stress factor always. Window = the tower's (first year warm-up, last held out). Base parameters:
the IBIS 3.0 set (params_s7_crop.yaml). Order: the thinnest types first, so partial results serve the types that need them.
"""
import sys, os, argparse
import numpy as np, pandas as pd
sys.path[:0] = ['python-port', 'verify/calibration']
from site_run import prescribe
from fit_carbon import PFT_NAMES

NATURAL = ('ENF', 'EBF', 'DBF', 'DNF', 'MF', 'SAV', 'WSA', 'OSH', 'CSH', 'GRA')
VMAX_BOUNDS = ('trop_bl_ever=27.0,60.6;trop_bl_dec=36.5,78.9;warm_bl_ever=27.0,60.6;temp_con_ever=37.8,87.2;boreal_con_ever=37.8,87.2;'
               'temp_bl_dec=36.5,78.9;boreal_bl_dec=36.5,78.9;boreal_con_dec=27.4,50.8;ever_shrub=38.3,77.4;dec_shrub=38.3,77.4;c4_grass=10,75;c3_grass=47.1,109.3')
PARAMS = 'python-port/configs/params/params_s7_crop.yaml'


def fit_set(pfts):
    f = []
    for p in pfts:
        f += [f'vmax:{PFT_NAMES[p]}', f'coefm:{PFT_NAMES[p]}']
        if p in (3, 5): f += [f'acclim_s0:{PFT_NAMES[p]}', f'acclim_smax:{PFT_NAMES[p]}']
    if any(p in (4, 6, 7) for p in pfts): f += ['gdd_thresh_u', 'gdd_ramp_u', 'sen_ycrit_u']
    if any(p >= 8 for p in pfts): f += ['gdd_thresh_l', 'tthresh_l_c']
    if any(p in (1, 8, 9, 10) for p in pfts): f += ['drop_stress_thresh']
    return f + ['stressfac']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--types', default='')
    a = ap.parse_args()
    si = pd.read_csv(f"{os.environ['IBIS_FLUX_DIR']}/FLUXNET_siteinfo.csv")
    cal = si[(si.HH == 'yes') & (si.WIN_SCORE.fillna(0) >= 2) & si.IGBP.isin(NATURAL)].copy()
    want = set(a.types.split(',')) if a.types else None
    n_type = {}
    for ts in cal.TYPE.fillna(''):
        for t in ts.split(): n_type[t] = n_type.get(t, 0) + 1
    cal['rank'] = [min(n_type.get(t, 999) for t in str(ts).split()) if str(ts).strip() else 999 for ts in cal.TYPE.fillna('')]
    cal = cal.sort_values(['rank', 'SITE_ID'])
    for r in cal.itertuples():
        types = str(r.TYPE).split()
        if want and not (set(types) & want): continue
        mat = float(r.MAT) if pd.notna(r.MAT) else None
        pfts, _, _, _ = prescribe(r.IGBP, float(r.LOCATION_LAT), mat)
        y0, y1 = int(r.WIN_Y0), int(r.WIN_Y1)
        print(f'{r.SITE_ID} {r.IGBP} --params {PARAMS} --igbp {r.IGBP} --sites {r.SITE_ID} --years {y0}-{y1} --hold-last 1 --holdout 0 '
              f'--co2 series --hourly-sw --lai-mode attrs-modis --et-corr-only --robust 25,30,1,0.15 --robust-auto --vmax-bounds "{VMAX_BOUNDS}" '
              f'--fit {",".join(fit_set(pfts))} --sigma 0.15 --n-evals 80 --popsize 6 --workers 3 --out {a.out}/{r.SITE_ID}   # {" ".join(types)}')


if __name__ == '__main__':
    main()
