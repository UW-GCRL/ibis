#!/bin/bash
# check10_31r (per-zone cover limit + C/D allocation refit + temperate residence + half tropical refit) vs the s31n hindcast and vs check10_31p/31q (CALIBRATION.md 71)
cd /shared/nas2/Hangkai/ibis-port; export JAX_PLATFORMS=cpu PYTHONPATH=python-port:.; PY=/home/hangkai/.venvs/ibis-gpu/bin/python; V=/shared/nas2/Hangkai/ibis/verify_outputs
N=/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad/nee_cycle; A=verify/calibration/diagnostics/audit_2026-10-26
P=31r; echo "##### check10_$P (tracebacks $(grep -c Traceback $V/check10_$P/run.log))"
echo "--- carbon closure"; $PY -I $A/closure_any.py $V/check10_$P 2 2>&1 | grep -v -iE "warn|Checkpoint|Land mask" | tail -3
echo "--- fire regions 2010"; $PY -I $A/fire_regions_state.py $V/check10_$P/restart/ibis_restart_2010.npz 2>&1 | grep -v -iE "warn|Checkpoint|Land mask" | tail -2
echo "--- s31n hindcast vs check10_$P, 2006-2010"; $PY -I $N/yearly_compare.py $V/final_s31n/hindcast/out $V/check10_$P/out 2006 2010 2>&1 | grep -v -i warn
echo "--- NaN cells"; $PY -I -c "
import sys, numpy as np; sys.path[:0]=['verify/fire','python-port','.']; import evaluate_gfed5 as eg
p='$V/check10_$P/restart/ibis_restart_2010.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p)
bad = ~np.isfinite(np.asarray(st.vegetation.fu)) | ~np.isfinite(np.asarray(st.carbon.totcsoi)); print('NaN cells', int(bad.sum()))" 2>&1 | grep -v -iE "warn|Checkpoint|Land mask"
echo "--- cover and wood, 2010 states (s31n, check10_31p limit, check10_31r design)"; $PY -I $N/cover_wood_check.py 2010 s31n=$V/final_s31n/hindcast limit=$V/check10_31p design=$V/check10_31r --years=2001,2004,2007,2010 2>&1 | grep -v -iE "warn|Checkpoint|Land mask"
echo "--- gate_cover-style cover by stress and zone on the design (ramp_fit_zones on check10_31r is not applicable: it reads final_*); zone cover from cover_wood_check above"
echo "=== R ANALYSIS DONE $(date '+%H:%M')"
