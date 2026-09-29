"""Supplementary 2 (CALIBRATION.md 35): one fit per tower of the 62-tower batch, from the robust S1 parameter set (--start <best.json>).
Fitted per tower: vmax and stomatal slope of each prescribed PFT (TRY bounds as in S1), acclimation of conifer PFTs,
upper-canopy phenology at deciduous trees, lower-canopy thresholds where a lower canopy exists, drought-drop threshold
at tropical deciduous / C4 / shrub, stressfac everywhere. Writes one line per tower to stdout: SITE IGBP <site_fit options>.

    python verify/calibration/s2_commands.py <s1 best.json> <out root>
"""
import sys, numpy as np, pandas as pd
sys.path[:0] = ['python-port', 'verify/calibration']
import site_fit
from site_run import prescribe
from fit_carbon import PFT_NAMES
START = sys.argv[1]; OUTROOT = sys.argv[2]
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; Y = 'python-port/configs/params/default_seasonal_sunshade_pft.yaml'
B = "trop_bl_ever=27.0,60.6;trop_bl_dec=36.5,78.9;warm_bl_ever=27.0,60.6;temp_con_ever=37.8,87.2;boreal_con_ever=37.8,87.2;temp_bl_dec=36.5,78.9;boreal_bl_dec=36.5,78.9;boreal_con_dec=27.4,50.8;ever_shrub=38.3,77.4;dec_shrub=38.3,77.4;c4_grass=10,75;c3_grass=47.1,109.3"
OVER = {'RU-Cok': [9, 11], 'AU-How': [0, 10], 'AU-DaS': [0, 10], 'AU-Dry': [0, 10]}; LAI = {'US-Ha1': 4.7, 'GF-Guy': 6.0}
sites = open('verify/calibration/site_data/site_batches.txt').readline().split()[1].split(',')
META = site_fit.META
for s in sites:
    igbp = META.loc[s, 'IGBP']; lat = float(META.loc[s, 'LOCATION_LAT'])
    mat = META.loc[s, 'MAT'] if 'MAT' in META else np.nan
    if not (isinstance(mat, (int, float)) and np.isfinite(mat)): mat = site_fit.tower_mat(s)
    pfts, fu, fl, _ = prescribe(igbp, lat, float(mat) if pd.notna(mat) else None)
    if s in OVER: pfts = OVER[s]
    fit = []
    for p in pfts: fit += [f'vmax:{PFT_NAMES[p]}', f'coefm:{PFT_NAMES[p]}']
    for p in pfts:
        if p in (3, 5, 7): fit += [f'acclim_s0:{PFT_NAMES[p]}', f'acclim_smax:{PFT_NAMES[p]}']
    if any(p in (4, 6) for p in pfts): fit += ['gdd_thresh_u', 'gdd_ramp_u', 'sen_ycrit_u']
    if any(p >= 8 for p in pfts): fit += ['gdd_thresh_l', 'tthresh_l_c']
    if any(p in (1, 8, 9, 10) for p in pfts): fit += ['drop_stress_thresh']
    fit += ['stressfac']
    extra = (f' --lai {s}={LAI[s]}' if s in LAI else '') + (f' --pft "{s}={",".join(map(str, OVER[s]))}"' if s in OVER else '')
    print(f'{s} {igbp} --params {Y} --igbp {igbp} --sites {s} --years 2008-2012 --hold-last 1 --holdout 0 --co2 series --hourly-sw --lai-mode attrs-modis{extra} '
          f'--et-corr-only --robust 25,30,1,0.15 --vmax-bounds "{B}" --start {START} --fit {",".join(fit)} --sigma 0.15 --n-evals 80 --popsize 6 --workers 3 --out {OUTROOT}/{s}')
