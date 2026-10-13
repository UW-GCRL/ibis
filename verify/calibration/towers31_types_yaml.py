#!/usr/bin/env python3
"""Per-type parameter file of IBIS 3.1 from the per-tower fits of the campaign (CALIBRATION.md 45.2).

    IBIS_FLUX_DIR=<towers31>/compact python verify/calibration/towers31_types_yaml.py <s2 root> <nee root> <base.yaml> <out.yaml> [table.csv]

Attribution of each fitted value at a tower (the tower runs its type at the parent's position, so the fitted PFT
position tells the canopy role):
  vmax / coefm / acclim_s0 / acclim_smax of PFT p  -> the tower's own type of that role (trees: same leaf form;
      shrubs; grasses), else the zone's type in position p (the grass or shrub of a savanna or shrubland), else none;
  gdd_thresh_u, gdd_ramp_u, sen_ycrit_u            -> the tower's tree types;
  gdd_thresh_l, tthresh_l_c                        -> the tower's shrub and grass types (own, else the zone's);
  drop_stress_thresh, stressfac, bgc_e0 (lt_e0), rgrowth, rroot -> every type of the tower (its TYPE column).
som_scale is a state multiplier of the tower runs (no parameter form). Each type takes the median over its towers;
the table gives n, median and the inter-quartile range. Types without a value keep their parent's (the base yaml).
"""
import sys, os, json
import numpy as np, pandas as pd, yaml
sys.path[:0] = ['python-port', 'verify/calibration']
from fit_carbon import PFT_NAMES
from site_run import prescribe
from ibis.io import pft_types as P

s2_root, nee_root, base, out = sys.argv[1:5]; table = sys.argv[5] if len(sys.argv) > 5 else ''
si = pd.read_csv(f"{os.environ['IBIS_FLUX_DIR']}/FLUXNET_siteinfo.csv").set_index('SITE_ID')
FORM = {t[0]: t[2] for t in P.TYPES}
CONIFER = {3, 5, 7}


def role(p):
    return 'tree' if p < 8 else ('shrub' if p < 10 else 'grass')


def type_for_position(p, own, zone):
    """The type a fitted PFT position informs at a tower with assigned types `own` in `zone`."""
    for t in own:
        f = FORM[t]
        if role(p) == 'tree' and f.startswith('tree') and ((p in CONIFER) == (f in ('tree_ne', 'tree_nd'))): return t
        if role(p) == 'shrub' and f.startswith('shrub'): return t
        if role(p) == 'grass' and f.startswith('grass') and ((p == 10) == (f == 'grass_c4')): return t
    s = P.ZONE_SLOTS[P.ZI[zone]][p] if zone in P.ZI else -1
    return P.IDS[s] if s >= 0 else None


rows = []
for stage, root in (('s2', s2_root), ('nee', nee_root)):
    if not os.path.isdir(root): continue
    for s in sorted(os.listdir(root)):
        f = f'{root}/{s}/best.json'
        if not os.path.exists(f) or s not in si.index: continue
        b = json.load(open(f)); own = str(si.loc[s, 'TYPE']).split(); zone = si.loc[s, 'ZONE']
        mat = float(si.loc[s, 'MAT']) if pd.notna(si.loc[s, 'MAT']) else None
        pfts, _, _, _ = prescribe(si.loc[s, 'IGBP'], float(si.loc[s, 'LOCATION_LAT']), mat)
        trees = [t for t in own if FORM[t].startswith('tree')]
        lower = [type_for_position(p, own, zone) for p in pfts if p >= 8]
        for k, v in b['params'].items():
            v = float(v)
            if ':' in k:
                name, pft = k.split(':'); p = PFT_NAMES.index(pft); t = type_for_position(p, own, zone)
                key = {'vmax': 'vmax', 'coefm': 'coefm', 'acclim_s0': 'acclim_s0', 'acclim_smax': 'acclim_smax'}.get(name)
                targets = [t] if (t and key) else []
            elif k in ('gdd_thresh_u', 'gdd_ramp_u', 'sen_ycrit_u'): key, targets = k, trees
            elif k in ('gdd_thresh_l', 'tthresh_l_c'): key, targets = k, [t for t in lower if t]
            elif k in ('drop_stress_thresh', 'stressfac', 'rgrowth', 'rroot'): key, targets = k, own
            elif k == 'bgc_e0': key, targets = 'lt_e0', own
            else: continue                                                     # som_scale and anything without a type form
            for t in dict.fromkeys(targets):
                rows.append(dict(stage=stage, site=s, type=t, key=key, value=v, zone=zone))
d = pd.DataFrame(rows)
if d.empty: raise SystemExit('no fitted values found')
g = d.groupby(['type', 'key'])['value']
tab = pd.DataFrame({'n': g.size(), 'median': g.median(), 'q25': g.quantile(0.25), 'q75': g.quantile(0.75)}).reset_index()
t = yaml.safe_load(open(base)) or {}
t['types'] = {}
for r in tab.itertuples():
    t['types'].setdefault(r.type, {})[r.key] = float(r.median)
t['types'] = {k: t['types'][k] for k in P.IDS if k in t['types']}
yaml.safe_dump(t, open(out, 'w'), sort_keys=False)
if table: tab.to_csv(table, index=False)
cov = tab.groupby('type')['n'].max()
print(f'{d.site.nunique()} towers, {len(d)} attributed values; types with values: {len(cov)} of {sum(1 for x in P.TYPES if not x[2] in ("wetland", "crop"))}')
print('towers per type (max over keys):', ' '.join(f'{k}:{int(cov.get(k, 0))}' for k in P.IDS if FORM[k] not in ('wetland', 'crop')))
P.parse_type_overrides(t['types']); print('wrote', out)
