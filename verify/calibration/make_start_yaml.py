"""Start yaml for a staged fit: a fitted yaml plus the starting values a stage needs.

    ibis-python verify/calibration/make_start_yaml.py <base.yaml> <out.yaml> [--structural-starts] [--vmax-mix] [--leaf-only] [--et-starts]

--structural-starts  stage (b1): tree gate, cover and dormancy scalars at the
                     lever-test-6 starts (tree_stress_min 0.45, ramp 0.20,
                     cover_k 0.5, fl_min 0.25, fu_min 0.10, drop_stress_thresh
                     0.5, tthresh_u_abs 0). Starts must come through the yaml:
                     fit_carbon's extras_defaults reads the current parameter
                     and the inert defaults sit outside the fit bounds.
--vmax-mix           stage (b2) on: `stomata.vmax_mix: true` and behaviour-
                     preserving starts for the four entries F77's priority
                     pick never read (CALIBRATION.md 17): trop_bl_dec <-
                     trop_bl_ever, boreal_bl_dec <- temp_bl_dec, boreal_con_dec
                     <- boreal_con_ever, dec_shrub <- ever_shrub -- the value
                     each big leaf has in fact been running at.
--leaf-only          `stomata.scale_leaf_only: true`: stems attenuate light but do
                     not assimilate (CALIBRATION.md 16, lever test 11).
--et-starts          ET stage: canopy.soil_beta_power 1.5 (F77 1.0 sits on the fit bound) and
                     interception.fmax 0.25 written explicitly (CALIBRATION.md 15).
Everything not named is carried from the base unchanged; the result is read
back through the loader and every injected value verified.
"""
import sys, yaml, numpy as np
args = [a for a in sys.argv[1:] if not a.startswith('--')]; flags = {a for a in sys.argv[1:] if a.startswith('--')}
base, out = args[0], args[1]
t = yaml.safe_load(open(base))
STRUCT_PFT = dict(tree_stress_min=0.45, tree_stress_ramp=0.20, cover_k=0.5, fl_min=0.25, fu_min=0.10)
STRUCT_PHENO = dict(drop_stress_thresh=0.5, tthresh_u_abs=0.0)
MIX_STARTS = {"tropical_broadleaf_deciduous": "tropical_broadleaf_evergreen",
              "boreal_broadleaf_deciduous": "temperate_broadleaf_deciduous",
              "boreal_conifer_deciduous": "boreal_conifer_evergreen",
              "deciduous_shrub": "evergreen_shrub"}
if '--structural-starts' in flags:
    t.setdefault('pft', {}).update(STRUCT_PFT); t.setdefault('pheno', {}).update(STRUCT_PHENO)
if '--vmax-mix' in flags:
    t.setdefault('stomata', {})['vmax_mix'] = True
if '--leaf-only' in flags:
    t.setdefault('stomata', {})['scale_leaf_only'] = True
if '--et-starts' in flags:
    t.setdefault('canopy', {})['soil_beta_power'] = 1.5
    t.setdefault('interception', {})['fmax'] = 0.25
    for dead, live in MIX_STARTS.items():
        t['stomata']['vmax'][dead] = float(t['stomata']['vmax'][live])
yaml.safe_dump(t, open(out, 'w'), sort_keys=False)
sys.path.insert(0, 'python-port')
from ibis.io.physics_params import build_all_params, load_physics_params
sp, pft, _ = build_all_params(4, np.zeros(4, np.float32), load_physics_params(out))
NAMES = ["tropical_broadleaf_evergreen", "tropical_broadleaf_deciduous", "warm_temperate_broadleaf_evergreen",
         "temperate_conifer_evergreen", "temperate_broadleaf_deciduous", "boreal_conifer_evergreen",
         "boreal_broadleaf_deciduous", "boreal_conifer_deciduous", "evergreen_shrub", "deciduous_shrub", "c4_grass", "c3_grass"]
msg = []
if '--structural-starts' in flags:
    for k, v in STRUCT_PFT.items(): assert abs(float(getattr(pft, k)) - v) < 1e-9, (k, float(getattr(pft, k)))
    for k, v in STRUCT_PHENO.items(): assert abs(float(getattr(sp.pheno, k)) - v) < 1e-9, (k, float(getattr(sp.pheno, k)))
    msg.append("structural starts verified")
if '--vmax-mix' in flags:
    assert sp.stomata_params.vmax_mix is True
    vp = np.asarray(sp.stomata_params.vmax_pft)
    for dead, live in MIX_STARTS.items():
        assert abs(vp[NAMES.index(dead)] - vp[NAMES.index(live)]) < 1e-12, (dead, vp[NAMES.index(dead)], vp[NAMES.index(live)])
    msg.append("vmax_mix on; " + ", ".join(f"{d.split('_')[0]}_{d.split('_')[-1]}={vp[NAMES.index(d)]*1e6:.1f}" for d in MIX_STARTS))
if '--leaf-only' in flags:
    assert sp.stomata_params.scale_leaf_only is True; msg.append("scale_leaf_only on")
if '--et-starts' in flags:
    assert abs(float(sp.canopy.soil_beta_power) - 1.5) < 1e-9 and abs(float(sp.interception.fmax) - 0.25) < 1e-9; msg.append("ET starts: soil_beta_power 1.5, fmax 0.25")
print(f"{out}: {'; '.join(msg) or 'base carried unchanged'}; base {base}")
