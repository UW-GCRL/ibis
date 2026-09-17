"""Stage (b1) start yaml: a fitted yaml plus starting values for the structural scalars.

    ibis-python verify/calibration/make_start_yaml.py <base.yaml> <out.yaml>

Starting values must come through the yaml: fit_carbon's extras_defaults reads
the current parameter, and the inert defaults (tree_stress_min -1, cover_k 0,
drop_stress_thresh -1, tthresh_u_abs -100) sit outside the fit bounds, so they
would be clipped to a bound as x0. Values from lever test 6 (CALIBRATION.md 14).
"""
import sys, yaml
base, out = sys.argv[1], sys.argv[2]
t = yaml.safe_load(open(base))
t.setdefault('pft', {}).update(dict(tree_stress_min=0.45, tree_stress_ramp=0.20,
                                    cover_k=0.5, fl_min=0.25, fu_min=0.10))
t.setdefault('pheno', {}).update(dict(drop_stress_thresh=0.5, tthresh_u_abs=0.0))
yaml.safe_dump(t, open(out, 'w'), sort_keys=False)
sys.path.insert(0, 'python-port')
import numpy as np
from ibis.io.physics_params import build_all_params, load_physics_params
sp, pft, _ = build_all_params(4, np.zeros(4, np.float32), load_physics_params(out))
for k, v in dict(tree_stress_min=0.45, tree_stress_ramp=0.20, cover_k=0.5, fl_min=0.25, fu_min=0.10).items():
    assert abs(float(getattr(pft, k)) - v) < 1e-9, (k, float(getattr(pft, k)))
for k, v in dict(drop_stress_thresh=0.5, tthresh_u_abs=0.0).items():
    assert abs(float(getattr(sp.pheno, k)) - v) < 1e-9, (k, float(getattr(sp.pheno, k)))
print(f"{out}: structural starts verified through the loader; vmax/specla/stressfac/pheno from {base}")
