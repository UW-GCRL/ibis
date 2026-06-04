#!/bin/bash
# Auto-launch the real-IC 200-yr spin-up the moment the synthetic spin-up finishes.
#
# This script polls the synthetic spin-up process (PID 972844). When that
# process exits (cleanly or otherwise), it waits for GPU memory to clear,
# then launches the real-IC 200-yr spin-up using:
#   - HWSD soita.nc per-cell sand/clay
#   - ESACCI vegtype_2000 per-cell vegetation
#   - GSWP3 1901-2000 forcing cycled twice
#   - Finding #27 fix (clamp fvapg_avg ∈ [-1e-3, 1e-3])
#
# Output:  /tmp/spinup_0p5deg_200yr_real_ic.log
# State:   /shared/nas2/Hangkai/ibis/verify_outputs/spinup_200yr_0p5deg_real_ic_v2/

set -u
SYNTHETIC_PID=972844
SAFETY_TIMEOUT_HR=12   # if synthetic doesn't finish in 12 hr, abort the chain

START=$(date +%s)
echo "[chain] $(date): waiting for synthetic spin-up PID $SYNTHETIC_PID to finish..."
echo "[chain]   timeout: ${SAFETY_TIMEOUT_HR} hr"

# Poll until the process is gone
while kill -0 "$SYNTHETIC_PID" 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED_HR=$(( (NOW - START) / 3600 ))
    if [ "$ELAPSED_HR" -ge "$SAFETY_TIMEOUT_HR" ]; then
        echo "[chain] $(date): SAFETY TIMEOUT after ${SAFETY_TIMEOUT_HR}h. Aborting chain."
        exit 1
    fi
    sleep 60
done
echo "[chain] $(date): synthetic spin-up finished. Waiting 30s for GPU cleanup..."
sleep 30

# Sanity check: did synthetic complete normally? Look for DONE: in the log.
if ! grep -q "DONE:" /tmp/spinup_0p5deg_200yr.log; then
    echo "[chain] $(date): WARNING: synthetic log does NOT contain 'DONE:' marker."
    echo "[chain]   Last 5 lines of synthetic log:"
    tail -5 /tmp/spinup_0p5deg_200yr.log
    echo "[chain]   Proceeding anyway — real-IC run is independent."
fi

# Confirm GPU is free
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "[chain] $(date): GPU free memory: ${FREE_MIB} MiB"
if [ "$FREE_MIB" -lt 16000 ]; then
    echo "[chain]   WARNING: <16 GB free. Will still try to launch."
fi

# Launch the real-IC run
cd /home/hangkai/ibis-port || exit 1
echo "[chain] $(date): launching real-IC 200-yr spin-up at 0.5° resolution..."

CUDA_VISIBLE_DEVICES=0 \
JAX_ENABLE_X64=1 \
PYTHONPATH=python-port:tests \
nohup /home/hangkai/ibis-port/verify/.venv_gpu/bin/python -u \
    -m verify.timestepping.pipeline.global_gswp3_gpu \
    --year-start 1 --year-end 200 \
    --forcing-cycle-start 1901 --forcing-cycle-end 2000 \
    --lat-step 0.5 --lon-step 0.5 \
    --real-ic --real-ic-vegtype-year 2000 \
    --output-name spinup_200yr_0p5deg_real_ic_v2 \
    --no-resume \
    > /tmp/spinup_0p5deg_200yr_real_ic.log 2>&1 &

REAL_IC_PID=$!
echo "[chain] $(date): launched real-IC, PID = ${REAL_IC_PID}"
echo "[chain]   log:  /tmp/spinup_0p5deg_200yr_real_ic.log"
echo "[chain]   out:  /shared/nas2/Hangkai/ibis/verify_outputs/spinup_200yr_0p5deg_real_ic_v2/"
echo "[chain]   ETA: ~11 hours"

# Detach (script will exit; the real-IC process is independent)
disown $REAL_IC_PID 2>/dev/null || true
echo "[chain] $(date): chain detached. Real-IC run will continue independently."
