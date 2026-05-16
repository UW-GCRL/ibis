#!/usr/bin/env bash
# =============================================================================
# scripts/setup_run.sh  --  Create and configure an IBIS run directory
#
# Usage:
#   bash scripts/setup_run.sh <run_name> [options]
#
# Options:
#   --years   N        number of simulation years       (default: 10)
#   --start   YYYY     start year                       (default: 1961)
#   --north   LAT      northern boundary in degrees N   (default: 32.5)
#   --south   LAT      southern boundary in degrees N   (default:  7.5)
#   --west    LON      western  boundary in degrees E   (default: 68.5)
#   --east    LON      eastern  boundary in degrees E   (default: 95.5)
#   --dynveg           enable dynamic vegetation (isimveg=1)
#   --spinup           no soil-C spinup (soilcspin=0)
#   --co2     VALUE    initial CO2 in mol/mol            (default: 0.000350)
#   --daily            write daily output (idailyout=1)
#   --iyranom YYYY     year to start reading anomaly files (default: 9999 = none)
#   --nanom   N        number of years in anomaly files   (default: 124)
#
# Example — India, 30-year run with dynamic vegetation:
#   bash scripts/setup_run.sh india_dynveg --years 30 --dynveg
#
# Example — Global land, 10 years:
#   bash scripts/setup_run.sh global_10yr \
#       --north 90 --south -60 --west -180 --east 180
#
# After setup, run the model with:
#   cd runs/<run_name> && /path/to/ibis | tee ibis.log
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IBIS_INPUT="$REPO/IBIS_input"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RUN_NAME=""
YEARS=10
IYEAR0=1961
NORTH=32.5
SOUTH=7.5
WEST=68.5
EAST=95.5
ISIMVEG=0
SOILCSPIN=1
CO2=0.000350
IDAILYOUT=0
IYRANOM=9999   # 9999 = no anomalies; use CRU climatology only
NANOM=124      # years in anomaly file (1901-2024 for CRU TS4)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/setup_run.sh <run_name> [options]"
  echo "Run with --help for full usage."
  exit 1
fi

RUN_NAME="$1"; shift

while [ $# -gt 0 ]; do
  case "$1" in
    --years)   YEARS="$2";    shift 2 ;;
    --start)   IYEAR0="$2";   shift 2 ;;
    --north)   NORTH="$2";    shift 2 ;;
    --south)   SOUTH="$2";    shift 2 ;;
    --west)    WEST="$2";     shift 2 ;;
    --east)    EAST="$2";     shift 2 ;;
    --dynveg)  ISIMVEG=1;     shift   ;;
    --spinup)  SOILCSPIN=0;   shift   ;;
    --co2)     CO2="$2";      shift 2 ;;
    --daily)   IDAILYOUT=1;   shift   ;;
    --iyranom) IYRANOM="$2";  shift 2 ;;
    --nanom)   NANOM="$2";    shift 2 ;;
    --help)
      head -35 "$0" | tail -30
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

RUNDIR="$REPO/runs/$RUN_NAME"

if [ -d "$RUNDIR" ]; then
  echo "ERROR: run directory already exists: $RUNDIR"
  echo "Remove it first or choose a different name."
  exit 1
fi

echo "=== Creating run directory: $RUNDIR ==="
mkdir -p "$RUNDIR/output/daily"    \
         "$RUNDIR/output/monthly"  \
         "$RUNDIR/output/yearly"   \
         "$RUNDIR/restart/odd"     \
         "$RUNDIR/restart/even"    \
         "$RUNDIR/input"           \
         "$RUNDIR/input/anom"

# ---------------------------------------------------------------------------
# Symlink input files
# ---------------------------------------------------------------------------
echo "--- Linking input files ---"
for nc in "$IBIS_INPUT"/*.nc; do
  ln -sf "$nc" "$RUNDIR/input/$(basename "$nc")"
done
echo "    $(ls "$RUNDIR/input" | wc -l) files linked from $IBIS_INPUT"

# ---------------------------------------------------------------------------
# Symlink params files
# ---------------------------------------------------------------------------
for p in can veg soi hyd; do
  if [ -f "$REPO/params.$p" ]; then
    ln -sf "$REPO/params.$p" "$RUNDIR/params.$p"
    echo "    params.$p linked"
  else
    echo "WARNING: $REPO/params.$p not found — model may fail"
  fi
done

# ---------------------------------------------------------------------------
# Write ibis.infile
# ---------------------------------------------------------------------------
cat > "$RUNDIR/ibis.infile" <<EOF
0         ! irestart   0: not a restart run  1: restart run
${IYEAR0}      ! iyear0     initial year of simulation (don't change for restart)
${YEARS}        ! nrun       number of years in this simulation
${IYRANOM}      ! iyranom    year to start reading anomalies (9999 = never, CRU climatology only)
${NANOM}        ! nanom      number of years in anomaly files (ignored when iyranom=9999)
9999      ! iyrdaily   year to start reading daily data (9999 = use weather generator)
${SOILCSPIN}         ! soilcspin  0: no soil spinup, 1: spinup acceleration
1         ! iyearout   0: no yearly output, 1: yearly output
1         ! imonthout  0: no monthly output, 1: monthly output
${IDAILYOUT}         ! idailyout  0: no daily output, 1: daily output
${ISIMVEG}         ! isimveg    0: static veg, 1: dynamic veg, 2: dynamic veg cold start
0         ! isimfire   0: fixed fire, 1: dynamic fire
0         ! isimco2    0: fixed co2, 1: ramped co2
${CO2}  ! co2init    initial CO2 concentration in mol/mol
0.209000  ! o2init     initial O2 concentration in mol/mol
3600.     ! dtime      time step in seconds (3600 = 1-hour)
0         ! idiag      0: no diagnostic output
${NORTH}      ! snorth     northern boundary (degrees N)
${SOUTH}       ! ssouth     southern boundary (degrees N)
${WEST}      ! swest      western boundary (degrees E)
${EAST}      ! seast      eastern boundary (degrees E)
EOF

echo ""
echo "=== Run directory ready: $RUNDIR ==="
echo ""
echo "Configuration:"
printf "  Domain   : %.1f°N–%.1f°N, %.1f°E–%.1f°E\n" $NORTH $SOUTH $WEST $EAST
echo  "  Period   : $YEARS years from $IYEAR0"
echo  "  Veg mode : isimveg=$ISIMVEG  (0=static, 1=dynamic)"
echo  "  Soil spin: soilcspin=$SOILCSPIN"
echo  "  CO2      : $CO2 mol/mol"
echo  "  Daily out: idailyout=$IDAILYOUT"
echo  "  Anomalies: iyranom=$IYRANOM  nanom=$NANOM"
if [ "$IYRANOM" != "9999" ]; then
  echo ""
  echo "  NOTE: anomaly files required in $RUNDIR/input/anom/"
  echo "  Generate them with:"
  echo "    python3 $REPO/scripts/preprocess_cru_ts4.py $RUNDIR <cru_ts4_dir>"
fi
echo ""
echo "To run the model:"
echo "  cd $RUNDIR"
echo "  $REPO/ibis 2>&1 | tee ibis.log"
echo ""
echo "To monitor progress (lines written to log):"
echo "  tail -f $RUNDIR/ibis.log"
echo ""
echo "Output files will appear in:"
echo "  $RUNDIR/output/monthly/   -- monthly NetCDF"
echo "  $RUNDIR/output/yearly/    -- yearly NetCDF"
echo "  $RUNDIR/restart/          -- restart state after each year"
