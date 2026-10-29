"""Per-tower yamls: the tower's zone types (from a full 3.0 type set with `types:`) at their parent positions on that set's base."""
import sys, os, yaml, numpy as np, pandas as pd
sys.path[:0] = ['python-port', 'verify/calibration']
from ibis.io import pft_types as P
from ibis.io.physics_params import PFT_NAMES as LONG, PFT_SHORT_NAMES as SHORT
full_yaml, ydir = sys.argv[1], sys.argv[2]; towers = sys.argv[3].split(',')
si = pd.read_csv('/shared/nas2/Hangkai/ibis/inputs/sites/towers31/compact/FLUXNET_siteinfo.csv', encoding='utf-8-sig').set_index('SITE_ID')
T = yaml.safe_load(open(full_yaml))['types']; PARENT = {t[0]: t[4] - 1 for t in P.TYPES}
os.makedirs(ydir, exist_ok=True)
for s in towers:
    own = [t for t in str(si.loc[s, 'TYPE']).split() if t in T]
    y = yaml.safe_load(open(full_yaml)); y.pop('types', None); st = y['stomata']; bg = y['bgc']; ph = y['pheno']
    for key in ('coefm_pft', 'acclim_s0_pft', 'acclim_smax_pft'): st.setdefault(key, {})
    for key in ('lt_e0', 'rgrowth', 'rroot'): bg.setdefault(key, {})
    scal = {}
    for t in own:
        p = PARENT[t]; v = T[t]
        if 'vmax' in v: st['vmax'][LONG[p]] = float(v['vmax'])
        for key in ('coefm', 'acclim_s0', 'acclim_smax'):
            if key in v: st[key + '_pft'][SHORT[p]] = float(v[key])
        for key in ('lt_e0', 'rgrowth', 'rroot'):
            if key in v: bg[key][SHORT[p]] = float(v[key])
        for key in ('aleaf', 'awood', 'aroot', 'specla', 'tauleaf', 'tauroot', 'tauwood0', 'lai_max'):     # allocation stage (CALIBRATION.md 55)
            if key in v: y['pft'][LONG[p]][key] = float(v[key])
        for key in ('gdd_thresh_u', 'gdd_ramp_u', 'sen_ycrit_u', 'gdd_thresh_l', 'tthresh_l_c', 'sen_ycrit_l', 'sen_pstart_l',   # lower-canopy senescence: CALIBRATION.md 54
                    'drop_stress_thresh', 'stressfac'):
            if key in v: scal.setdefault(key, []).append(float(v[key]))
    for key, vals in scal.items(): (st if key == 'stressfac' else ph)[key] = float(np.mean(vals))
    with open(f'{ydir}/{s}.yaml', 'w') as fo:
        fo.write(f'# {s}: types {" ".join(own)} at their parent positions on {os.path.basename(full_yaml)} (viab_build.py)\n'); yaml.safe_dump(y, fo, sort_keys=False)
    print(s, own, [PARENT[t] for t in own])
