#!/usr/bin/env bash
# run_global_transient.sh — run IBIS-JAX global transient simulation (1901–2010)
#
# Steps:
#   1. (Optional) run spinup if not already complete
#   2. Stage spinup year-200 restart into transient restart dir
#   3. Run transient 1901–2010
#   4. Auto-resume if interrupted (detects last completed transient year)
#
# Usage:
#   bash scripts/run_global_transient.sh [--skip-spinup] [--resume] [--platform gpu|cpu]
#
# Flags:
#   --skip-spinup   Skip spinup even if restart/spinup/ibis_restart_200.npz is absent
#                   (use if you have a pre-existing spinup restart from elsewhere)
#   --resume        Resume an interrupted transient run from the last saved restart
#   --platform X    Override JAX platform (gpu or cpu; default: auto)
#   --log-level X   Logging verbosity (DEBUG|INFO|WARNING; default: INFO)
#
# Configuration files used:
#   configs/global_spinup.yaml        — spinup (200 years, cycling 1901-1910 GSWP3)
#   configs/global_transient.yaml     — transient (1901-2010, irestart=200)
#   configs/global_transient_resume.yaml  — auto-generated when resuming
#
# Output:
#   output/spinup/   — spinup NetCDF (every 10 years)
#   output/transient/ — transient NetCDF (every year)
#   restart/spinup/  — spinup checkpoints (ibis_restart_NNN.npz)
#   restart/transient/ — transient checkpoints (ibis_restart_YYYY.npz)
#   logs/spinup.log, logs/transient.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"
RUNNER="${REPO_ROOT}/scripts/run_global.py"

SPINUP_CONFIG="${REPO_ROOT}/configs/global_spinup.yaml"
TRANSIENT_CONFIG="${REPO_ROOT}/configs/global_transient.yaml"
RESUME_CONFIG="${REPO_ROOT}/configs/global_transient_resume.yaml"

SPINUP_RESTART_DIR="${REPO_ROOT}/restart/spinup"
TRANSIENT_RESTART_DIR="${REPO_ROOT}/restart/transient"
LOG_DIR="${REPO_ROOT}/logs"

SPINUP_FINAL_YEAR=200      # irestart value in global_transient.yaml
TRANSIENT_START_YEAR=1901
TRANSIENT_END_YEAR=2010
TRANSIENT_NRUN=$(( TRANSIENT_END_YEAR - TRANSIENT_START_YEAR + 1 ))

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_SPINUP=0
RESUME=0
PLATFORM_ARG=""
LOG_LEVEL="INFO"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-spinup) SKIP_SPINUP=1 ;;
        --resume)      RESUME=1 ;;
        --platform)    PLATFORM_ARG="--platform $2"; shift ;;
        --log-level)   LOG_LEVEL="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

check_python() {
    [[ -x "${PYTHON}" ]] || die "Python not found at ${PYTHON}. Activate venv first or check path."
    "${PYTHON}" -c "import jax, equinox" 2>/dev/null \
        || die "JAX or Equinox not importable. Is the venv set up? Try: pip install -e ."
}

last_spinup_year() {
    # Return the highest NNN for ibis_restart_NNN.npz in spinup restart dir
    local latest
    latest=$(ls "${SPINUP_RESTART_DIR}"/ibis_restart_*.npz 2>/dev/null \
        | grep -oP '(?<=ibis_restart_)\d+(?=\.npz)' \
        | sort -n | tail -1)
    echo "${latest:-0}"
}

last_transient_year() {
    # Return the highest YYYY for ibis_restart_YYYY.npz in transient restart dir
    local latest
    latest=$(ls "${TRANSIENT_RESTART_DIR}"/ibis_restart_*.npz 2>/dev/null \
        | grep -oP '(?<=ibis_restart_)\d+(?=\.npz)' \
        | sort -n | tail -1)
    echo "${latest:-0}"
}

# ---------------------------------------------------------------------------
# Step 0 — preflight checks
# ---------------------------------------------------------------------------
log "=== IBIS-JAX global transient runner ==="
log "Repo root: ${REPO_ROOT}"
check_python

mkdir -p "${LOG_DIR}" "${SPINUP_RESTART_DIR}" "${TRANSIENT_RESTART_DIR}"
mkdir -p "${REPO_ROOT}/output/spinup" "${REPO_ROOT}/output/transient"

# ---------------------------------------------------------------------------
# Step 1 — Spinup
# ---------------------------------------------------------------------------
SPINUP_RESTART="${SPINUP_RESTART_DIR}/ibis_restart_${SPINUP_FINAL_YEAR}.npz"

if [[ ${SKIP_SPINUP} -eq 1 ]]; then
    log "Skipping spinup (--skip-spinup)."
    [[ -f "${SPINUP_RESTART}" ]] \
        || log "WARNING: spinup restart ${SPINUP_RESTART} not found. Stage it manually before step 2."

elif [[ -f "${SPINUP_RESTART}" ]]; then
    log "Spinup already complete (${SPINUP_RESTART} exists). Skipping."

else
    # Check for a partial spinup and resume it automatically
    LAST_SPINUP=$(last_spinup_year)
    if [[ ${LAST_SPINUP} -gt 0 && ${LAST_SPINUP} -lt ${SPINUP_FINAL_YEAR} ]]; then
        SPINUP_RESUME_YEAR=$(( LAST_SPINUP + 1 ))
        SPINUP_REMAINING=$(( SPINUP_FINAL_YEAR - LAST_SPINUP ))
        log "Partial spinup detected (reached year ${LAST_SPINUP}). Resuming from year ${SPINUP_RESUME_YEAR} ..."

        # Generate a resume config for the spinup
        SPINUP_RESUME_CONFIG="${REPO_ROOT}/configs/global_spinup_resume_auto.yaml"
        "${PYTHON}" - <<PYEOF
import yaml, re

with open("${SPINUP_CONFIG}") as f:
    cfg = yaml.safe_load(f)

cfg['time']['iyear0']    = ${SPINUP_RESUME_YEAR}
cfg['time']['nrun']      = ${SPINUP_REMAINING}
cfg['time']['irestart']  = ${LAST_SPINUP}

with open("${SPINUP_RESUME_CONFIG}", 'w') as f:
    f.write("# Auto-generated spinup resume config — do not edit by hand\n")
    yaml.dump(cfg, f, default_flow_style=False)
PYEOF
        log "Resuming spinup with config: ${SPINUP_RESUME_CONFIG}"
        log "Spinup log → ${LOG_DIR}/spinup.log"
        "${PYTHON}" "${RUNNER}" "${SPINUP_RESUME_CONFIG}" \
            --log-level "${LOG_LEVEL}" ${PLATFORM_ARG} \
            2>&1 | tee -a "${LOG_DIR}/spinup.log"

    else
        log "Starting spinup from cold (200 years, cycling 1901-1910 GSWP3) ..."
        log "Spinup log → ${LOG_DIR}/spinup.log"
        "${PYTHON}" "${RUNNER}" "${SPINUP_CONFIG}" \
            --log-level "${LOG_LEVEL}" ${PLATFORM_ARG} \
            2>&1 | tee "${LOG_DIR}/spinup.log"
    fi

    [[ -f "${SPINUP_RESTART}" ]] \
        || die "Spinup finished but ${SPINUP_RESTART} not found. Check ${LOG_DIR}/spinup.log."
    log "Spinup complete."
fi

# ---------------------------------------------------------------------------
# Step 2 — Stage spinup restart into transient restart dir
# ---------------------------------------------------------------------------
TRANSIENT_INIT_RESTART="${TRANSIENT_RESTART_DIR}/ibis_restart_${SPINUP_FINAL_YEAR}.npz"

if [[ -f "${TRANSIENT_INIT_RESTART}" ]]; then
    log "Transient init restart already staged (${TRANSIENT_INIT_RESTART})."
elif [[ -f "${SPINUP_RESTART}" ]]; then
    log "Staging spinup year-${SPINUP_FINAL_YEAR} restart → transient restart dir ..."
    cp "${SPINUP_RESTART}" "${TRANSIENT_INIT_RESTART}"
    log "Staged: ${TRANSIENT_INIT_RESTART}"
else
    die "Cannot stage restart: ${SPINUP_RESTART} not found. Run spinup first or use --skip-spinup."
fi

# ---------------------------------------------------------------------------
# Step 3 — Run (or resume) transient
# ---------------------------------------------------------------------------
if [[ ${RESUME} -eq 1 ]]; then
    LAST_TRANSIENT=$(last_transient_year)
    if [[ ${LAST_TRANSIENT} -le 0 || ${LAST_TRANSIENT} -lt ${TRANSIENT_START_YEAR} ]]; then
        log "No transient restart found (or before start year). Running from the beginning."
        RESUME=0
    elif [[ ${LAST_TRANSIENT} -ge ${TRANSIENT_END_YEAR} ]]; then
        log "Transient already complete (last restart: ${LAST_TRANSIENT}). Nothing to do."
        exit 0
    else
        RESUME_FROM_YEAR=$(( LAST_TRANSIENT + 1 ))
        RESUME_NRUN=$(( TRANSIENT_END_YEAR - LAST_TRANSIENT ))
        log "Resuming transient from year ${RESUME_FROM_YEAR} (last restart: ${LAST_TRANSIENT}) ..."
        log "Remaining: ${RESUME_NRUN} years (${RESUME_FROM_YEAR}–${TRANSIENT_END_YEAR})"

        "${PYTHON}" - <<PYEOF
import yaml

with open("${TRANSIENT_CONFIG}") as f:
    cfg = yaml.safe_load(f)

cfg['time']['iyear0']   = ${RESUME_FROM_YEAR}
cfg['time']['nrun']     = ${RESUME_NRUN}
cfg['time']['irestart'] = ${LAST_TRANSIENT}

with open("${RESUME_CONFIG}", 'w') as f:
    f.write("# Auto-generated transient resume config — do not edit by hand\n")
    yaml.dump(cfg, f, default_flow_style=False)
PYEOF
        log "Resume config written: ${RESUME_CONFIG}"
        log "Transient log → ${LOG_DIR}/transient.log"
        "${PYTHON}" "${RUNNER}" "${RESUME_CONFIG}" \
            --log-level "${LOG_LEVEL}" ${PLATFORM_ARG} \
            2>&1 | tee -a "${LOG_DIR}/transient.log"

        log "Transient simulation complete (resumed)."
        exit 0
    fi
fi

# Full transient run from year 200 restart
LAST_TRANSIENT=$(last_transient_year)
if [[ ${LAST_TRANSIENT} -ge ${TRANSIENT_END_YEAR} ]]; then
    log "Transient already complete (last restart: ${LAST_TRANSIENT}). Use --resume to re-check."
    exit 0
fi

log "Running transient ${TRANSIENT_START_YEAR}–${TRANSIENT_END_YEAR} (${TRANSIENT_NRUN} years) ..."
log "Transient log → ${LOG_DIR}/transient.log"
"${PYTHON}" "${RUNNER}" "${TRANSIENT_CONFIG}" \
    --log-level "${LOG_LEVEL}" ${PLATFORM_ARG} \
    2>&1 | tee "${LOG_DIR}/transient.log"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
LAST_TRANSIENT=$(last_transient_year)
if [[ ${LAST_TRANSIENT} -ge ${TRANSIENT_END_YEAR} ]]; then
    log "=== Transient simulation complete. ==="
    log "Output:  ${REPO_ROOT}/output/transient/"
    log "Restart: ${TRANSIENT_RESTART_DIR}/"
else
    log "WARNING: Run finished but last restart is year ${LAST_TRANSIENT} (expected ${TRANSIENT_END_YEAR})."
    log "         Check ${LOG_DIR}/transient.log for errors."
    log "         To resume: bash scripts/run_global_transient.sh --resume"
    exit 1
fi
