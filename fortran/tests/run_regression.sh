#!/usr/bin/env bash
# =============================================================================
# tests/run_regression.sh  --  IBIS F77 vs F2008 regression test
#
# Builds both binaries, runs each on the same 1-year India-domain test case,
# then calls compare_outputs.py to report numerical differences.
#
# Usage:
#   bash tests/run_regression.sh [--skip-build] [--skip-run-f77] [--skip-run-f2008]
#
# Options:
#   --skip-build      skip 'make ibis_f77' and 'make ibis' (use existing binaries)
#   --skip-run-f77    skip re-running the F77 binary (use existing output)
#   --skip-run-f2008  skip re-running the F2008 binary (use existing output)
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
REGDIR="$REPO/tests/regression"
IBIS_F77="$REPO/ibis_f77"
IBIS_F2008="$REPO/ibis"

SKIP_BUILD=0
SKIP_RUN_F77=0
SKIP_RUN_F2008=0

for arg in "$@"; do
  case "$arg" in
    --skip-build)      SKIP_BUILD=1 ;;
    --skip-run-f77)    SKIP_RUN_F77=1 ;;
    --skip-run-f2008)  SKIP_RUN_F2008=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# =============================================================================
# 1. Build binaries
# =============================================================================
if [ "$SKIP_BUILD" -eq 0 ]; then
  echo "=== Building F77 binary (ibis_f77) ==="
  make -C "$REPO" ibis_f77

  echo ""
  echo "=== Building F2008 binary (ibis) ==="
  make -C "$REPO" ibis
fi

echo ""
echo "F77  binary : $IBIS_F77  ($(wc -c < "$IBIS_F77") bytes)"
echo "F2008 binary: $IBIS_F2008  ($(wc -c < "$IBIS_F2008") bytes)"

# =============================================================================
# 2. Helper: prepare for a new run.
#    IBIS writes output/restart relative to its CWD ($REGDIR), so we create
#    those directories there (not inside run_f77/ / run_f2008/). After the
#    run completes, the caller moves them into the appropriate run subdir.
# =============================================================================
setup_rundir() {
  local rundir="$1"
  # Clear any leftover outputs from a previous run in the subdir
  rm -rf "$rundir/output" "$rundir/restart"
  # Clear and (re)create the dirs that IBIS will write into (relative to CWD)
  rm -rf "$REGDIR/output" "$REGDIR/restart"
  mkdir -p "$REGDIR/output/daily" \
            "$REGDIR/output/monthly" \
            "$REGDIR/output/yearly" \
            "$REGDIR/restart/odd" \
            "$REGDIR/restart/even"
}

# =============================================================================
# 3. Run F77 binary
# =============================================================================
if [ "$SKIP_RUN_F77" -eq 0 ]; then
  echo ""
  echo "=== Running F77 binary ==="
  setup_rundir "$REGDIR/run_f77"
  # IBIS reads ibis.infile, params.*, and input/ from its CWD
  (cd "$REGDIR" && "$IBIS_F77" > run_f77/ibis.log 2>&1) && \
    echo "F77 run completed" || { echo "F77 run FAILED (see $REGDIR/run_f77/ibis.log)"; exit 1; }
  # Move output/restart from CWD into run_f77/
  [ -d "$REGDIR/output"  ] && mv "$REGDIR/output"  "$REGDIR/run_f77/"
  [ -d "$REGDIR/restart" ] && mv "$REGDIR/restart" "$REGDIR/run_f77/"
  echo "F77 output files:"
  find "$REGDIR/run_f77/output" "$REGDIR/run_f77/restart" -name '*.nc' | sort
fi

# =============================================================================
# 4. Run F2008 binary
# =============================================================================
if [ "$SKIP_RUN_F2008" -eq 0 ]; then
  echo ""
  echo "=== Running F2008 binary ==="
  setup_rundir "$REGDIR/run_f2008"
  (cd "$REGDIR" && "$IBIS_F2008" > run_f2008/ibis.log 2>&1) && \
    echo "F2008 run completed" || { echo "F2008 run FAILED (see $REGDIR/run_f2008/ibis.log)"; exit 1; }
  # Move output/restart from CWD into run_f2008/
  [ -d "$REGDIR/output"  ] && mv "$REGDIR/output"  "$REGDIR/run_f2008/"
  [ -d "$REGDIR/restart" ] && mv "$REGDIR/restart" "$REGDIR/run_f2008/"
  echo "F2008 output files:"
  find "$REGDIR/run_f2008/output" "$REGDIR/run_f2008/restart" -name '*.nc' | sort
fi

# =============================================================================
# 5. Compare outputs
# =============================================================================
echo ""
echo "=== Comparing outputs ==="
python3 "$REPO/tests/compare_outputs.py" \
  "$REGDIR/run_f77" \
  "$REGDIR/run_f2008"
