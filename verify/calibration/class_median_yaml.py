"""Class-median parameter set from the per-tower fits (CALIBRATION.md 37): one yaml the global model can run.

    python verify/calibration/class_median_yaml.py <gpp/et fit root> <nee fit root> <base.yaml> <out.yaml> [table.csv]

Per-PFT entries ('vmax:<pft>', 'coefm:<pft>', 'acclim_*:<pft>') take the median over every tower that fitted that PFT,
whatever its class (a mixed-forest tower informs both its PFTs). Scalar phenology and stress entries take the median
over the towers that fitted them. The respiration values (bgc_e0, rgrowth, rroot) are fitted per tower and written
as per-PFT tables through the class -> PFT map below (the model takes (12,) vectors since CALIBRATION.md 37); classes
with mixed canopies (MF, SAV, WSA) do not map to a PFT and are left out of those medians. som_scale is a state
multiplier of the tower runs and has no yaml form; k_scale stays 1 (the per-tower HWSD2 pinning is a site device).
"""
import sys, json, glob, os, numpy as np, pandas as pd, yaml
sys.path[:0] = ['python-port', 'verify/calibration']
from fit_carbon import PFT_NAMES
from ibis.io.physics_params import build_all_params, load_physics_params

gpp_root, nee_root, base, out = sys.argv[1:5]; table = sys.argv[5] if len(sys.argv) > 5 else ''
inv = pd.read_csv('verify/calibration/site_data/site_inventory.csv').set_index('site')
CLASS_PFT = {'ENF': ['temp_con_ever', 'boreal_con_ever', 'boreal_con_dec'], 'DBF': ['temp_bl_dec', 'boreal_bl_dec'],
             'EBF': ['trop_bl_ever', 'warm_bl_ever', 'trop_bl_dec'], 'GRA': ['c3_grass', 'c4_grass'],
             'OSH': ['dec_shrub', 'ever_shrub'], 'CSH': ['dec_shrub', 'ever_shrub']}
RESP = {'bgc_e0': 'lt_e0', 'rgrowth': 'rgrowth', 'rroot': 'rroot', 'rwood': 'rwood'}
rows = []
for root, stage in ((gpp_root, 'gpp_et'), (nee_root, 'nee')):
    for f in sorted(glob.glob(f'{root}/*/best.json')):
        s = f.split('/')[-2]
        for k, v in json.load(open(f))['params'].items(): rows.append(dict(stage=stage, site=s, igbp=inv.loc[s, 'igbp'], name=k, value=float(v)))
d = pd.DataFrame(rows)
t = yaml.safe_load(open(base)); VKEYS = list(t['stomata']['vmax']); assert len(VKEYS) == len(PFT_NAMES)
PFT_SCALARS = ('cover_k', 'fl_min', 'fu_min', 'fu_k', 'tree_stress_min', 'tree_stress_ramp')
summary = []
def put(k, v, n, note=''):
    summary.append(dict(name=k, value=v, n_towers=n, note=note))
    if k.startswith('vmax:'): t['stomata']['vmax'][VKEYS[PFT_NAMES.index(k[5:])]] = v
    elif ':' in k and k.split(':')[0] in ('coefm', 'acclim_s0', 'acclim_smax'): t['stomata'].setdefault(k.split(':')[0] + '_pft', {})[k.split(':')[1]] = v
    elif k in ('wfps_opt', 'k_scale'): t.setdefault('bgc', {})[k] = v
    elif k == 'stressfac' or k in ('acclim_tau', 'acclim_s0', 'acclim_smax', 'cap_kn'): t['stomata'][k] = v
    elif k in PFT_SCALARS: t.setdefault('pft', {})[k] = v
    elif k in ('wliqumax', 'wliqsmax', 'wliqlmax', 'fmax'): t.setdefault('interception', {})[k] = v
    elif k == 'soil_beta_power': t.setdefault('canopy', {})[k] = v
    else: t.setdefault('pheno', {})[k] = v
# 1. everything but the respiration fields and som_scale: pooled median per name
for k, g in d[~d.name.isin(list(RESP) + ['som_scale'])].groupby('name'):
    put(k, float(g.value.median()), int(g.site.nunique()), 'pooled over classes ' + '/'.join(sorted(g.igbp.unique())))
# 2. respiration: per-PFT tables via the class map
for k, field in RESP.items():
    g = d[d.name == k]
    if g.empty: continue
    tab = {}
    for cls, pfts in CLASS_PFT.items():
        gc = g[g.igbp == cls] if cls != 'OSH' else g[g.igbp.isin(['OSH', 'CSH'])]
        if gc.empty: continue
        for p in pfts:
            if p in tab and cls == 'CSH': continue
            tab[p] = float(gc.value.median())
            summary.append(dict(name=f'{k}:{p}', value=tab[p], n_towers=int(gc.site.nunique()), note=f'class {cls} median'))
    t.setdefault('bgc', {})[field] = tab
    missing = [p for p in PFT_NAMES if p not in tab]
    if missing: summary.append(dict(name=f'{k}:(default)', value=float('nan'), n_towers=0, note='yaml default kept for ' + ','.join(missing)))
yaml.safe_dump(t, open(out, 'w'), sort_keys=False)
# round trip
sp, pft, _ = build_all_params(4, np.zeros(4, np.float32), load_physics_params(out))
for r in summary:
    k, v = r['name'], r['value']
    if not np.isfinite(v): continue
    if k.startswith('vmax:'): got = float(np.asarray(sp.stomata_params.vmax_pft)[PFT_NAMES.index(k[5:])])
    elif k.split(':')[0] in RESP: got = float(np.asarray(getattr(sp.bgc_params, RESP[k.split(':')[0]]))[PFT_NAMES.index(k.split(':')[1])])
    elif ':' in k: got = float(np.asarray(getattr(sp.stomata_params, k.split(':')[0] + '_pft'))[PFT_NAMES.index(k.split(':')[1])])
    elif k in ('wfps_opt', 'k_scale'): got = float(getattr(sp.bgc_params, k))
    elif k == 'stressfac' or k.startswith('acclim_') or k == 'cap_kn': got = float(getattr(sp.stomata_params, k))
    elif k in PFT_SCALARS: got = float(getattr(pft, k))
    elif k in ('wliqumax', 'wliqsmax', 'wliqlmax', 'fmax'): got = float(getattr(sp.interception, k))
    elif k == 'soil_beta_power': got = float(sp.canopy.soil_beta_power)
    else: got = float(getattr(sp.pheno, k))
    assert abs(got - v) <= 1e-6 * max(1.0, abs(v)), (k, got, v)
S = pd.DataFrame(summary)
if table: S.to_csv(table, index=False)
pd.set_option('display.width', 200); print(S.to_string(index=False)); print(f'\n{out} written; {len(S)} entries round-trip OK')
