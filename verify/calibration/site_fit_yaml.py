"""Write a site_fit result into a parameter yaml, with a round-trip check (CALIBRATION.md 22-24).

    python verify/calibration/site_fit_yaml.py <fit_dir>/best.json <base.yaml> <out.yaml>

`vmax:<pft>` entries go to stomata.vmax (keys in PFT order); EXTRAS names are routed as in finalize_fit.sh (stressfac,
acclim_* and coefm_* to stomata, cover/establishment scalars to pft, interception, soil_beta_power to canopy, anything
else to pheno). The yaml is then read back through build_all_params and every fitted value is compared with best.json.
"""
import sys, json, numpy as np, yaml
sys.path[:0] = ['python-port', 'verify/calibration']
from fit_carbon import PFT_NAMES
from ibis.io.physics_params import build_all_params, load_physics_params

best, base, out = sys.argv[1:4]
P = json.load(open(best))['params']; t = yaml.safe_load(open(base))
VKEYS = list(t['stomata']['vmax'])                     # yaml vmax keys, in PFT order
assert len(VKEYS) == len(PFT_NAMES), VKEYS
PFT_SCALARS = ('cover_k', 'fl_min', 'fu_min', 'fu_k', 'tree_stress_min', 'tree_stress_ramp')
COEFM = {'coefm_bl': 'broadleaf', 'coefm_con': 'conifer', 'coefm_shrub': 'shrub', 'coefm_c4': 'c4_grass', 'coefm_c3': 'c3_grass'}
for k, v in P.items():
    v = float(v)
    if k.startswith('vmax:'): t['stomata']['vmax'][VKEYS[PFT_NAMES.index(k[5:])]] = v
    elif ':' in k and k.split(':')[0] in ('coefm', 'acclim_s0', 'acclim_smax'): t['stomata'].setdefault(k.split(':')[0] + '_pft', {})[k.split(':')[1]] = v
    elif k == 'stressfac' or k in ('acclim_tau', 'acclim_s0', 'acclim_smax', 'cap_kn'): t['stomata'][k] = v
    elif k in COEFM: t['stomata']['ball_berry'][COEFM[k]]['coefm'] = v
    elif k in PFT_SCALARS: t.setdefault('pft', {})[k] = v
    elif k in ('wliqumax', 'wliqsmax', 'wliqlmax', 'fmax'): t.setdefault('interception', {})[k] = v
    elif k == 'soil_beta_power': t.setdefault('canopy', {})[k] = v
    else: t.setdefault('pheno', {})[k] = v
yaml.safe_dump(t, open(out, 'w'), sort_keys=False)
sp, pft, _ = build_all_params(4, np.zeros(4, np.float32), load_physics_params(out))
for k, v in P.items():
    if k.startswith('vmax:'): got = float(np.asarray(sp.stomata_params.vmax_pft)[PFT_NAMES.index(k[5:])])
    elif k == 'stressfac' or k.startswith('acclim_') or k == 'cap_kn': got = float(getattr(sp.stomata_params, k))
    elif k in COEFM: got = float(np.asarray(sp.stomata_params.coefm)[list(COEFM).index(k)])
    elif k in PFT_SCALARS: got = float(getattr(pft, k))
    elif k in ('wliqumax', 'wliqsmax', 'wliqlmax', 'fmax'): got = float(getattr(sp.interception, k))
    elif k == 'soil_beta_power': got = float(sp.canopy.soil_beta_power)
    else: got = float(getattr(sp.pheno, k))
    assert abs(got - float(v)) <= 1e-6 * max(1.0, abs(float(v))), (k, got, v)
print(f'{out} written; {len(P)} fitted values round-trip: ' + ', '.join(f'{k}={float(v):.4g}' for k, v in P.items()))
