"""Follow-up analyses on the completed global GSWP3 verification.

Analysis 1: hour-of-day correlation of worst diffs
  Tests hypothesis: drift concentrates at sunrise/sunset where my
  daily→hourly disaggregator's coszen transitions across 0 are abrupt.
  Sunrise/sunset hour depends on lat × julian day, so we compute
  per-cell local solar zenith trajectory and check if worst-diff
  timesteps fall near solar-noon ± transition windows.

Analysis 2: per-cell hour histogram for worst-diff cell
  For L-35p0_+140p0 (S. Australia, worst ΔT cell), plot hourly diff
  growth — if it's monotone the issue is integration; if it
  concentrates at specific hours, it's a code-path issue triggered
  at certain solar geometry.
"""
from __future__ import annotations
import csv
import glob
import collections
import math
from pathlib import Path

OUTPUT_DIR = Path("/shared/nas2/Hangkai/ibis/verify_outputs")
SUMMARY_OUT = OUTPUT_DIR / "analysis_followup.txt"


def parse_cell_label(label: str) -> tuple[float, float]:
    """Parse 'L-05p0_-080p0' → (lat=-5.0, lon=-80.0)."""
    lat_str, lon_str = label.split("_", 1)
    lat = float(lat_str[1:].replace("p", "."))
    lon = float(lon_str.replace("p", "."))
    return lat, lon


def solar_zenith_at(lat: float, lon: float, jday: int, utc_hour: float) -> float:
    """Compute cos(solar zenith) at given (lat, lon, jday, UTC hour).
    Matches the same orbital/declination formulae as diurnal_disaggregate."""
    PI = math.pi
    orbit = 2.0 * PI * jday / 365.2425
    decl = (0.006918 - 0.399912 * math.cos(orbit) + 0.070257 * math.sin(orbit)
            - 0.006758 * math.cos(2 * orbit) + 0.000907 * math.sin(2 * orbit)
            - 0.002697 * math.cos(3 * orbit) + 0.001480 * math.sin(3 * orbit))
    # Local solar hour = UTC + lon/15  (lon east = ahead of UTC)
    local_h = utc_hour + lon / 15.0
    ha = (local_h - 12.0) * (2 * PI / 24.0)
    cz = (math.sin(math.radians(lat)) * math.sin(decl)
          + math.cos(math.radians(lat)) * math.cos(decl) * math.cos(ha))
    return cz


def classify_hour(cz: float, prev_cz: float = None) -> str:
    """Classify whether 'hour' is sunrise, day, sunset, night."""
    if cz <= 0.0:
        if prev_cz is not None and prev_cz > 0.0:
            return "sunset"
        return "night"
    else:
        if prev_cz is not None and prev_cz <= 0.0:
            return "sunrise"
        return "day"


def main():
    JUNE_1990 = (1990, 6)
    jday_offset = sum([31, 28, 31, 30, 31]) + 1  # day-of-year for June 1
    YEAR, MONTH = JUNE_1990

    # ==========================================================
    # Pass 1: collect worst (cell, field) diffs + their step/hour
    # ==========================================================
    worst = {}    # (cell, field) -> (diff, step, hour, day)
    rows_by_cell = collections.defaultdict(int)
    for f in sorted(glob.glob(str(OUTPUT_DIR / "batch_*_perstep.csv"))):
        with open(f) as fh:
            r = csv.DictReader(fh)
            for row in r:
                d = float(row["abs_diff"])
                k = (row["cell_name"], row["field"])
                rows_by_cell[row["cell_name"]] += 1
                cur = worst.get(k)
                if cur is None or d > cur[0]:
                    worst[k] = (d, int(row["step"]),
                                int(row["hour"]), int(row["day"]))

    print(f"Cells with diffs > 1e-7: {len(rows_by_cell)}")
    print(f"(cell, field) pairs tracked: {len(worst)}")

    # ==========================================================
    # Analysis 1: hour-of-day distribution stratified by ΔT vs Δflux
    # ==========================================================
    state_fields = ('soil.tsoi', 'soil.wsoi', 'soil.wisoi', 'soil.tg',
                    'snow.', 'canopy.', 'stresstu', 'stresstl')
    flux_fields = ('flux.',)

    def is_flux(f):
        return any(f.startswith(p) for p in flux_fields)

    big_diff = [(c, fld, d, step, hour, day)
                for (c, fld), (d, step, hour, day) in worst.items()
                if d > 1e-3]

    print(f"\n=== {len(big_diff)} (cell, field) pairs with worst |Δ| > 1e-3 ===")

    # Solar zenith at the worst-diff timestep for each entry
    cz_distribution = {"night": 0, "sunrise": 0, "day": 0, "sunset": 0}
    cz_distribution_flux = {"night": 0, "sunrise": 0, "day": 0, "sunset": 0}
    cz_distribution_state = {"night": 0, "sunrise": 0, "day": 0, "sunset": 0}

    for c, fld, d, step, hour, day in big_diff:
        lat, lon = parse_cell_label(c)
        jday = jday_offset + day
        cz_now = solar_zenith_at(lat, lon, jday, hour)
        cz_prev = solar_zenith_at(lat, lon, jday, (hour - 1) % 24)
        cls = classify_hour(cz_now, cz_prev)
        cz_distribution[cls] += 1
        if is_flux(fld):
            cz_distribution_flux[cls] += 1
        else:
            cz_distribution_state[cls] += 1

    print(f"\n--- Distribution by local-solar phase ---")
    print(f"{'phase':>10s}   {'all':>6s}  {'state':>6s}  {'flux':>6s}")
    for ph in ("night", "sunrise", "day", "sunset"):
        print(f"  {ph:>10s}   {cz_distribution[ph]:>4d}    "
              f"{cz_distribution_state[ph]:>4d}    {cz_distribution_flux[ph]:>4d}")

    # ==========================================================
    # Analysis 2: drill into worst-diff cell (full hour trace)
    # ==========================================================
    print(f"\n=== Drill into worst-diff cell ===")
    # Find cell with highest TEMPERATURE drift (state-var, not flux)
    state_worst_cells = [(c, max((d for (cc, f), (d, *_) in worst.items()
                                   if cc == c and not is_flux(f)), default=0))
                         for c in rows_by_cell.keys()]
    state_worst_cells.sort(key=lambda x: -x[1])
    target_cell = state_worst_cells[0][0]
    target_lat, target_lon = parse_cell_label(target_cell)
    print(f"Worst-ΔT cell: {target_cell} (lat={target_lat:+6.1f}, lon={target_lon:+7.1f})")
    print(f"  Worst state-var ΔT: {state_worst_cells[0][1]:.3e}")

    # Hour-of-day histogram of all logged rows for this cell
    cell_hours = collections.defaultdict(list)
    for f in sorted(glob.glob(str(OUTPUT_DIR / "batch_*_perstep.csv"))):
        with open(f) as fh:
            r = csv.DictReader(fh)
            for row in r:
                if row["cell_name"] == target_cell:
                    d = float(row["abs_diff"])
                    h = int(row["hour"])
                    if d > 1e-5:  # only significant
                        cell_hours[h].append(d)

    print(f"  Hourly distribution of |Δ| > 1e-5 (count, max):")
    print(f"  {'hour':>5s}  {'count':>6s}  {'max |Δ|':>11s}  {'solar pos':>12s}")
    for h in range(24):
        if h not in cell_hours: continue
        ds = cell_hours[h]
        # Compute typical solar zenith at this hour, mid-month
        cz = solar_zenith_at(target_lat, target_lon,
                              jday_offset + 15, h)
        cls = ("☀️ day" if cz > 0.2 else
               "🌅 dawn/dusk" if cz > 0 else "🌙 night")
        print(f"  {h:>2d}    {len(ds):>5d}    {max(ds):>11.3e}  {cls:>12s}  cz={cz:+.2f}")

    return 0


if __name__ == "__main__":
    main()
