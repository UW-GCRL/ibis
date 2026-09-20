#!/bin/bash
# Closing validation of a parameter set (CALIBRATION.md): the v11 protocol under the FINAL parameters.
#
#   bash verify/calibration/final_validation.sh <TAG> <params.yaml>      (~6 h on the GPU; CIFS: no exec bit)
#
#   1. spin-up   : cold start, 1601-1900 (300 yr), 1901-1930 climate cycled, 296 ppm  (v11_spinup cfg)
#   2. transient : 1901-2000 from the 1900 state, real GSWP3 years, rising CO2, yearly output
#   3. hindcast  : 2001-2010 from the 2000 state, real years, rising CO2, MONTHLY output
#   4. scoring on the actual 2001-2010 years: pools/fluxes/bands, tree cover, ET, season, towers,
#      fire (GFED5, 2001-05 from the 2010 state), methane (GMB), and the 1901-2010 GPP trend.
# Every fit and validation so far started from a 2010 state made under the v11 default parameters
# and re-equilibrated 25-50 years on cycled 2001-2010 weather; this is the self-consistent form.
set -u -o pipefail
TAG=$1; PAR=$(readlink -f "$2"); V=/shared/nas2/Hangkai/ibis/verify_outputs; R=/shared/nas2/Hangkai/ibis-port
PY=/home/hangkai/.venvs/ibis-gpu/bin/python; CPY=/shared/nas2/Hangkai/ibis-env/ibis-python
F=$V/final_$TAG; cd $R; export PYTHONPATH=python-port:. XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.3
[ -f "$PAR" ] || { echo "no params yaml $PAR"; exit 1; }
$PY -c "import cma, jax, equinox, xarray, netCDF4, pandas, yaml" || exit 1
mkcfg() {  # mkcfg <template cfg> <out cfg> <stage dir> ; then per-stage sed on the result
  sed -e "s|$(dirname $(grep -m1 '^  output_dir:' $1 | awk '{print $2}'))|$3|g" $1 > $2
  grep -q '^params_file' $2 && sed -i "s|^params_file:.*|params_file: $PAR|" $2 || echo "params_file: $PAR" >> $2
}
mkdir -p $F/spinup/{out,restart} $F/transient/{out,restart} $F/hindcast/{out,restart}
# 1. spin-up
mkcfg $V/v11_spinup/cfg.yaml $F/spinup/cfg.yaml $F/spinup
echo "=== [$TAG] SPIN-UP 1601-1900 start $(date +%H:%M) ==="; grep -E "iyear0|nrun|irestart|co2init|spinup_year|params_file" $F/spinup/cfg.yaml | tr -s ' ' | tr '\n' ';'; echo
$PY -u python-port/scripts/run_global.py $F/spinup/cfg.yaml > $F/spinup/run.log 2>&1 || { echo "SPIN-UP FAILED"; tail -5 $F/spinup/run.log; exit 1; }
[ -f $F/spinup/restart/ibis_restart_1900.npz ] || { echo "no 1900 restart"; ls $F/spinup/restart | tail -2; exit 1; }
grep -E "gate ACTIVE|gate inactive|Physics parameters" $F/spinup/run.log | head -2
# 2. transient 1901-2000
mkcfg $V/v11_transient/cfg.yaml $F/transient/cfg.yaml $F/transient
sed -i -e 's|^  nrun: 110|  nrun: 100|' $F/transient/cfg.yaml; cp $F/spinup/restart/ibis_restart_1900.npz $F/transient/restart/
echo "=== [$TAG] TRANSIENT 1901-2000 start $(date +%H:%M) ==="
$PY -u python-port/scripts/run_global.py $F/transient/cfg.yaml > $F/transient/run.log 2>&1 || { echo "TRANSIENT FAILED"; tail -5 $F/transient/run.log; exit 1; }
[ -f $F/transient/restart/ibis_restart_2000.npz ] || { echo "no 2000 restart"; exit 1; }
# 3. hindcast 2001-2010, monthly
mkcfg $V/v11_transient/cfg.yaml $F/hindcast/cfg.yaml $F/hindcast
sed -i -e 's|^  iyear0: 1901|  iyear0: 2001|' -e 's|^  nrun: 110|  nrun: 10|' -e 's|^  irestart: 1900|  irestart: 2000|' -e 's|^  imonthout: 0|  imonthout: 1|' \
       -e 's|^  transient_co2: true|  transient_co2: true\n  et_components: true|' $F/hindcast/cfg.yaml    # ET components in the yearly files (CALIBRATION.md 15)
cp $F/transient/restart/ibis_restart_2000.npz $F/hindcast/restart/
echo "=== [$TAG] HINDCAST 2001-2010 (monthly) start $(date +%H:%M) ==="; grep -E "iyear0|nrun|irestart|imonthout" $F/hindcast/cfg.yaml | tr -s ' ' | tr '\n' ';'; echo
$PY -u python-port/scripts/run_global.py $F/hindcast/cfg.yaml > $F/hindcast/run.log 2>&1 || { echo "HINDCAST FAILED"; tail -5 $F/hindcast/run.log; exit 1; }
ST=$F/hindcast/restart/ibis_restart_2010.npz; [ -f $ST ] || { echo "no 2010 restart"; exit 1; }
# 4. scoring on 2001-2010
echo "=== [$TAG] SCORES $(date +%H:%M) ==="
$CPY verify/calibration/state_report.py --run final_$TAG=$F/hindcast/out:$ST --nyear 10 2>&1 | grep -v Warning | tail -30
$CPY verify/calibration/fu_vs_tree.py $F/hindcast/out 2>&1 | grep -E "^years|^ALL|^tree |^lat |PHANTOM|MISSING"
$CPY verify/calibration/et_vs_modis.py $F/hindcast/out 2>&1 | grep -E "^ALL|^lat|^tree "
$CPY verify/calibration/seasonal_baseline.py --monthly-dir $F/hindcast/out --gpp-var amgpptot --label "final_$TAG 2001-2010 hindcast" 2>&1 | grep -v Warning
$CPY verify/calibration/tower_phase.py $F/hindcast/out 2>&1 | grep -v Warning | tail -12
$CPY verify/calibration/tower_et.py $F/hindcast/out 2>&1 | grep -v Warning | tail -20
echo "--- 1901-2010 GPP trend (PgC/yr, decadal means from yearly output) ---"
$CPY - "$F" <<'PY' 2>&1 | grep -v Warning
import sys, glob, re, numpy as np, xarray as xr
F=sys.argv[1]; fs=sorted(glob.glob(f'{F}/transient/out/ibis_yearly_*.nc'))+sorted(glob.glob(f'{F}/hindcast/out/ibis_yearly_*.nc'))
m=xr.open_dataset('/shared/nas2/Min/HWSD/surta.nc').landfrac.values; lat=np.arange(89.75,-90,-0.5)
area=(np.cos(np.deg2rad(lat))[:,None]*np.ones((360,720)))*(0.5*111.32e3)**2*np.nan_to_num(m)
dec={}
for f in fs:
    y=int(re.search(r'(\d+)\.nc$',f).group(1))
    with xr.open_dataset(f) as d: g=np.nan_to_num(np.asarray(d['aygpptot'].values,float))
    dec.setdefault(y//10*10,[]).append(float(np.sum(g*area))*1e-12)
print("  "+"  ".join(f"{k}s:{np.mean(v):5.1f}" for k,v in sorted(dec.items())))
PY
echo "=== [$TAG] FIRE vs GFED5 (2001-05 from the 2010 state) ==="; $CPY verify/fire/evaluate_gfed5.py --state $ST 2>&1 | grep -v -E "Warning|CUDA|warn" | tail -20
echo "=== [$TAG] METHANE vs GMB ==="; $PY verify/methane/evaluate_global.py --state $ST --config all 2>&1 | grep -v -E "Warning|CUDA|warn" | tail -30
echo "=== [$TAG] FINAL VALIDATION DONE $(date +%H:%M) ==="
