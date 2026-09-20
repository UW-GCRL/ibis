#!/bin/bash
# After a fit completes (verify/calibration copy of the session script): params yaml (on IBIS defaults, round-trip verified),
# then a 50-yr full-domain validation from the v11 2010 state.
set -eu -o pipefail
TAG=$1; BASE=${2:-}; V=/shared/nas2/Hangkai/ibis/verify_outputs; R=/shared/nas2/Hangkai/ibis-port
cd $R; export PYTHONPATH=python-port:.
/shared/nas2/Hangkai/ibis-env/ibis-python - "$TAG" "$BASE" <<'PY' 2>&1 | grep -v Warning
import numpy as np, yaml, sys
sys.path.insert(0,'python-port')
from ibis.io.physics_params import build_all_params, load_physics_params
TAG=sys.argv[1]; V='/shared/nas2/Hangkai/ibis/verify_outputs'
NAMES=["tropical_broadleaf_evergreen","tropical_broadleaf_deciduous","warm_temperate_broadleaf_evergreen",
       "temperate_conifer_evergreen","temperate_broadleaf_deciduous","boreal_conifer_evergreen",
       "boreal_broadleaf_deciduous","boreal_conifer_deciduous","evergreen_shrub","deciduous_shrub","c4_grass","c3_grass"]
b=np.load(f'{V}/fit_carbon_{TAG}/best.npz'); vmax,specla=b['vmax'],b['specla']
sp0,pft0,bgc0=build_all_params(4,np.zeros(4,np.float32),None)
BASE=sys.argv[2] if len(sys.argv)>2 and sys.argv[2] else f'{V}/both_fix/params.yaml'   # staged fits pass the stage's start yaml
t=yaml.safe_load(open(BASE)); print('yaml base:', BASE)
for k,n in enumerate(NAMES):
    t['stomata']['vmax'][n]=float(vmax[k]); t['pft'][n]['specla']=float(specla[k])
for key in ('knb','kpb','ksb'): t['bgc'][key]=float(getattr(bgc0,key))   # undo both_fix x0.67
# fitted scalar extras (stressfac -> stomata; phenology -> pheno section)
import json
ex=json.loads(str(b['extras_json'])) if 'extras_json' in b.files else {}
PFT_SCALARS=('cover_k','fl_min','fu_min','fu_k','tree_stress_min','tree_stress_ramp')   # PFTParams, 'pft' section
COEFM={'coefm_bl':'broadleaf','coefm_con':'conifer','coefm_shrub':'shrub','coefm_c4':'c4_grass','coefm_c3':'c3_grass'}
for k,v in ex.items():
    if k=='stressfac': t['stomata']['stressfac']=float(v)
    elif k in PFT_SCALARS: t.setdefault('pft',{})[k]=float(v)
    elif k in COEFM: t['stomata']['ball_berry'][COEFM[k]]['coefm']=float(v)
    elif k in ('wliqumax','wliqsmax','wliqlmax','fmax'): t.setdefault('interception',{})[k]=float(v)
    elif k=='soil_beta_power': t.setdefault('canopy',{})[k]=float(v)
    else: t.setdefault('pheno',{})[k]=float(v)                                            # PhenoParams; unknown keys raise
print('extras carried into yaml:', ex or 'none')
out=f'{V}/fit_carbon_{TAG}/params_fit_{TAG}.yaml'; yaml.safe_dump(t,open(out,'w'),sort_keys=False)
sp,pft,bgc=build_all_params(4,np.zeros(4,np.float32),load_physics_params(out))
ev=float(np.max(np.abs(np.asarray(sp.stomata_params.vmax_pft)-vmax))); es=float(np.max(np.abs(np.asarray(pft.specla)-specla)))
assert ev<1e-9 and es<1e-4, (ev,es)
for k,v in ex.items():
    got=(float(sp.stomata_params.stressfac) if k=='stressfac' else float(getattr(pft,k)) if k in PFT_SCALARS
         else float(sp.stomata_params.coefm[list(COEFM).index(k)]) if k in COEFM else float(getattr(sp.interception,k)) if k in ('wliqumax','wliqsmax','wliqlmax','fmax')
         else float(sp.canopy.soil_beta_power) if k=='soil_beta_power' else float(getattr(sp.pheno,k)))
    assert abs(got-float(v))<1e-9, (k,got,v)
print(f"params_fit_{TAG}.yaml written; round-trip vmax {ev:.1e} specla {es:.1e}; soil-OM at defaults")
PY
cp $V/fit_carbon_$TAG/params_fit_$TAG.yaml $R/python-port/configs/params/
mkdir -p $V/val_$TAG/fit/{out,restart}; cp $V/v11_transient/restart/ibis_restart_2010.npz $V/val_$TAG/fit/restart/
sed -e "s|val_v11c/fit|val_$TAG/fit|g" -e 's|^  nrun: 25|  nrun: 50|' -e "s|^params_file:.*|params_file: $V/fit_carbon_$TAG/params_fit_$TAG.yaml|" \
    -e 's|^  transient_co2: false|  transient_co2: false\n  et_components: true|' $V/val_v11c/fit/cfg.yaml > $V/val_$TAG/fit/cfg.yaml    # ET components in the yearly files (CALIBRATION.md 15)
grep -q "et_components: true" $V/val_$TAG/fit/cfg.yaml || { echo "et_components not inserted"; exit 1; }
grep -E "nrun|params_file|output_dir" $V/val_$TAG/fit/cfg.yaml
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.15
nohup /home/hangkai/.venvs/ibis-gpu/bin/python -u python-port/scripts/run_global.py $V/val_$TAG/fit/cfg.yaml > $V/val_$TAG/fit/run.log 2>&1 &
echo "validation val_$TAG/fit launched pid $! at $(date +%H:%M) -- 50 yr, ~45 min"
