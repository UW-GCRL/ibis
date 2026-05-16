"""Full aggregate + problem-cell extraction + analyses 1 & 2.

After the global GSWP3 test completes, this script:
  A) Aggregates all 24 batch CSVs into a per-(cell, field) max-diff table
  B) Identifies "problem cells" with max diff > threshold; writes them
     to problem_cells.csv on NAS for cheap rerun later
  C) Analysis 1: hour-of-day correlation of worst diffs (sunrise/sunset
     hypothesis test)
  D) Analysis 2: top-3 worst cells get detailed hourly trace
"""
from __future__ import annotations
import csv
import glob
import collections
import math
from pathlib import Path

OUT = Path("/shared/nas2/Hangkai/ibis/verify_outputs")


def parse_label(label: str) -> tuple[float, float]:
    lat_str, lon_str = label.split("_", 1)
    lat = float(lat_str[1:].replace("p", "."))
    lon = float(lon_str.replace("p", "."))
    return lat, lon


def solar_zenith(lat: float, lon: float, jday: int, utc_hour: float) -> float:
    PI = math.pi
    orbit = 2.0 * PI * jday / 365.2425
    decl = (0.006918 - 0.399912 * math.cos(orbit) + 0.070257 * math.sin(orbit)
            - 0.006758 * math.cos(2 * orbit) + 0.000907 * math.sin(2 * orbit)
            - 0.002697 * math.cos(3 * orbit) + 0.001480 * math.sin(3 * orbit))
    local_h = utc_hour + lon / 15.0
    ha = (local_h - 12.0) * (2 * PI / 24.0)
    return (math.sin(math.radians(lat)) * math.sin(decl)
            + math.cos(math.radians(lat)) * math.cos(decl) * math.cos(ha))


def is_flux(fname: str) -> bool:
    return fname.startswith("flux.")


# Physically reasonable ranges for each field type (units in comments).
# Values outside these ranges = numerical instability or bug.
PHYSICAL_RANGES = {
    "soil.tsoi[:,0]":   (200.0, 340.0),    # K (-73C to +67C)
    "soil.wsoi[:,0]":   (0.0, 1.0),         # fraction
    "soil.wisoi[:,0]":  (0.0, 1.0),         # fraction
    "soil.tg":          (200.0, 340.0),     # K
    "snow.fi":          (0.0, 1.0),         # fraction
    "snow.tsno[:,0]":   (200.0, 274.0),     # K (snow can't be much above 0C)
    "snow.hsno[:,0]":   (0.0, 50.0),        # m
    "canopy.tu":        (200.0, 340.0),     # K
    "canopy.ts":        (200.0, 340.0),     # K
    "canopy.tl":        (200.0, 340.0),     # K
    "canopy.t12":       (200.0, 340.0),     # K
    "canopy.t34":       (200.0, 340.0),     # K
    "canopy.q12":       (0.0, 0.05),        # kg/kg
    "canopy.q34":       (0.0, 0.05),        # kg/kg
    "flux.fvapg":       (-2e-4, 2e-4),      # kg/m²/s (~ ±500 mm/day)
    "flux.fsenu":       (-2000.0, 2000.0),  # W/m²
    "flux.fsenl":       (-2000.0, 2000.0),  # W/m²
    "flux.fseng":       (-2000.0, 2000.0),  # W/m²
    "stresstu":         (0.0, 1.0),         # fraction
    "stresstl":         (0.0, 1.0),         # fraction
}


def main():
    jday0 = sum([31, 28, 31, 30, 31]) + 1  # June 1 = day 152
    print("=" * 72)
    print("GLOBAL GSWP3 VERIFICATION — AGGREGATE + DRILL-DOWN")
    print("=" * 72)

    # ===========================================================
    # A) Aggregate all batch CSVs + check value ranges
    # ===========================================================
    worst = collections.defaultdict(float)              # (cell, field) -> max diff
    worst_step = {}
    worst_hour = {}
    worst_day = {}
    total_rows = 0
    # Track value ranges per field across F77 and JAX values
    value_range_f77 = collections.defaultdict(lambda: [float("inf"), float("-inf")])
    value_range_jax = collections.defaultdict(lambda: [float("inf"), float("-inf")])
    out_of_range = collections.defaultdict(list)    # field -> [(cell, value, side)]

    for f in sorted(glob.glob(str(OUT / "batch_*_perstep.csv"))):
        with open(f) as fh:
            r = csv.DictReader(fh)
            for row in r:
                total_rows += 1
                d = float(row["abs_diff"])
                k = (row["cell_name"], row["field"])
                if d > worst[k]:
                    worst[k] = d
                    worst_step[k] = int(row["step"])
                    worst_hour[k] = int(row["hour"])
                    worst_day[k] = int(row["day"])

                # Value range check
                fld = row["field"]
                fv = float(row["F77"])
                jv = float(row["JAX"])
                vr_f = value_range_f77[fld]
                vr_j = value_range_jax[fld]
                vr_f[0] = min(vr_f[0], fv); vr_f[1] = max(vr_f[1], fv)
                vr_j[0] = min(vr_j[0], jv); vr_j[1] = max(vr_j[1], jv)
                if fld in PHYSICAL_RANGES:
                    lo, hi = PHYSICAL_RANGES[fld]
                    if fv < lo or fv > hi:
                        if len(out_of_range[fld]) < 5:
                            out_of_range[fld].append((row["cell_name"], fv, "F77"))
                    if jv < lo or jv > hi:
                        if len(out_of_range[fld]) < 10:
                            out_of_range[fld].append((row["cell_name"], jv, "JAX"))

    cells = sorted(set(c for c, _ in worst.keys()))
    print(f"\nA. Aggregate stats")
    print(f"   Cells verified: {len(cells)} / 117")
    print(f"   Per-step rows logged (|Δ|>1e-7): {total_rows:,}")
    print(f"   Unique (cell, field) pairs: {len(worst):,}")

    # Per-field max across all cells
    field_max = collections.defaultdict(float)
    field_owner = {}
    for (cell, field), d in worst.items():
        if d > field_max[field]:
            field_max[field] = d
            field_owner[field] = cell
    print(f"\n   Per-field worst diff across ALL cells:")
    for f, d in sorted(field_max.items(), key=lambda x: -x[1]):
        cls = "🔴" if d > 1e-2 else ("🟡" if d > 1e-4 else "✅")
        print(f"     {cls} {f:25s}: {d:.3e}   @ {field_owner[f]}")

    # ===========================================================
    # A2) Value-range sanity check
    # ===========================================================
    print(f"\n   Value-range check (across all cells, all timesteps logged):")
    print(f"   {'field':25s}  {'F77 [min, max]':>30s}  {'JAX [min, max]':>30s}  {'expected':>22s}")
    all_in_range = True
    for fld in sorted(value_range_f77.keys()):
        vf = value_range_f77[fld]
        vj = value_range_jax[fld]
        if fld in PHYSICAL_RANGES:
            lo, hi = PHYSICAL_RANGES[fld]
            exp_str = f"[{lo}, {hi}]"
            in_range = (vf[0] >= lo and vf[1] <= hi and
                        vj[0] >= lo and vj[1] <= hi)
            mark = "✅" if in_range else "🔴"
            if not in_range:
                all_in_range = False
        else:
            exp_str = "no rule"
            mark = "  "
        f_str = f"[{vf[0]:.2e}, {vf[1]:.2e}]"
        j_str = f"[{vj[0]:.2e}, {vj[1]:.2e}]"
        print(f"   {mark} {fld:23s}  {f_str:>30s}  {j_str:>30s}  {exp_str:>22s}")

    if out_of_range:
        print(f"\n   Out-of-range examples (first 5 per field):")
        for fld, examples in out_of_range.items():
            print(f"     {fld}:")
            for cell, val, side in examples[:5]:
                print(f"       {side} cell={cell} value={val:.3e}")
    else:
        print(f"\n   ✅ All values stay within physically reasonable ranges.")

    # ===========================================================
    # B) Extract problem cells → CSV on NAS
    # ===========================================================
    THRESHOLD = 1e-3
    cell_max = collections.defaultdict(float)
    cell_max_field = {}
    for (cell, field), d in worst.items():
        if d > cell_max[cell]:
            cell_max[cell] = d
            cell_max_field[cell] = field

    problem_cells = sorted([(c, d) for c, d in cell_max.items() if d > THRESHOLD],
                            key=lambda x: -x[1])
    print(f"\nB. Problem cells (worst |Δ| > {THRESHOLD:.0e})")
    print(f"   Count: {len(problem_cells)} / {len(cells)}")

    problem_csv = OUT / "problem_cells.csv"
    with open(problem_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_label", "lat", "lon", "worst_diff", "worst_field",
                    "worst_step", "worst_hour", "worst_day"])
        for cell, d in problem_cells:
            lat, lon = parse_label(cell)
            fld = cell_max_field[cell]
            k = (cell, fld)
            w.writerow([cell, lat, lon, f"{d:.6e}", fld,
                        worst_step[k], worst_hour[k], worst_day[k]])
    print(f"   Saved to: {problem_csv}")

    print(f"\n   Top 15 problem cells:")
    print(f"   {'cell':22s} {'lat':>6s} {'lon':>6s} {'|Δ|max':>11s}  {'field':20s} @step")
    for cell, d in problem_cells[:15]:
        lat, lon = parse_label(cell)
        fld = cell_max_field[cell]
        k = (cell, fld)
        print(f"   {cell:22s} {lat:>+6.1f} {lon:>+6.1f} {d:>11.3e}  {fld:20s} @{worst_step[k]} (hr {worst_hour[k]})")

    # Save FULL summary CSV
    summary_csv = OUT / "summary_199006_30d.csv"
    with open(summary_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "field", "worst_abs_diff", "step", "hour", "day"])
        for (cell, field), d in sorted(worst.items()):
            k = (cell, field)
            w.writerow([cell, field, f"{d:.6e}", worst_step[k], worst_hour[k], worst_day[k]])
    print(f"\n   Full summary CSV: {summary_csv}")

    # ===========================================================
    # C) Analysis 1: hour-of-day correlation
    # ===========================================================
    print(f"\n" + "=" * 72)
    print(f"C. Analysis 1: hour-of-day correlation of worst diffs")
    print(f"=" * 72)

    # For each (cell, field) worst case, classify by local solar phase
    # We classify into 4 phases: night, sunrise, day, sunset based on cz
    # at this hour AND prev hour
    phase_count_state = collections.Counter()
    phase_count_flux = collections.Counter()
    for (cell, field), d in worst.items():
        if d <= THRESHOLD:
            continue
        lat, lon = parse_label(cell)
        jd = jday0 + worst_day[(cell, field)]
        h = worst_hour[(cell, field)]
        cz_now = solar_zenith(lat, lon, jd, h)
        cz_prev = solar_zenith(lat, lon, jd, (h - 1) % 24)
        if cz_now <= 0 and cz_prev <= 0:
            phase = "night"
        elif cz_now > 0 and cz_prev <= 0:
            phase = "sunrise"
        elif cz_now <= 0 and cz_prev > 0:
            phase = "sunset"
        else:
            phase = "day"
        if is_flux(field):
            phase_count_flux[phase] += 1
        else:
            phase_count_state[phase] += 1

    total_state = sum(phase_count_state.values())
    total_flux = sum(phase_count_flux.values())
    # Baseline: random distribution across 24 hours per day
    # ~2 sunrise hrs, ~2 sunset hrs, ~10 day hrs, ~10 night hrs → 8% / 8% / 42% / 42%
    print(f"\n   Distribution of WORST-diff timesteps by local solar phase:")
    print(f"   {'phase':>10s}   {'state':>10s}   {'flux':>10s}   {'expected':>10s}")
    expected = {"night": 0.42, "day": 0.42, "sunrise": 0.08, "sunset": 0.08}
    for ph in ("night", "sunrise", "day", "sunset"):
        s = phase_count_state.get(ph, 0)
        f = phase_count_flux.get(ph, 0)
        s_pct = 100 * s / max(total_state, 1)
        f_pct = 100 * f / max(total_flux, 1)
        e_pct = 100 * expected[ph]
        # Highlight if observed > expected by 2x
        s_mark = "🔴" if s_pct > 2 * e_pct else "  "
        f_mark = "🔴" if f_pct > 2 * e_pct else "  "
        print(f"   {ph:>10s}   {s_mark} {s_pct:>5.1f}% ({s:>3d})   {f_mark} {f_pct:>5.1f}% ({f:>3d})   {e_pct:>5.1f}%")

    # ===========================================================
    # D) Analysis 2: detailed hourly trace for top-3 worst cells
    # ===========================================================
    print(f"\n" + "=" * 72)
    print(f"D. Analysis 2: hourly diff growth for top-3 worst cells")
    print(f"=" * 72)

    top3 = problem_cells[:3]
    for target_cell, target_d in top3:
        lat, lon = parse_label(target_cell)
        print(f"\n   ── {target_cell} (lat={lat:+5.1f}, lon={lon:+6.1f}) "
              f"worst {cell_max_field[target_cell]} = {target_d:.3e} ──")

        # Collect all rows for this cell, group by hour-of-day
        hourly_max = collections.defaultdict(float)
        hourly_count = collections.Counter()
        for f in sorted(glob.glob(str(OUT / "batch_*_perstep.csv"))):
            with open(f) as fh:
                r = csv.DictReader(fh)
                for row in r:
                    if row["cell_name"] != target_cell:
                        continue
                    d = float(row["abs_diff"])
                    h = int(row["hour"])
                    if d > hourly_max[h]:
                        hourly_max[h] = d
                    if d > 1e-5:
                        hourly_count[h] += 1

        print(f"   {'hour':>4s}  {'max |Δ|':>11s}  {'count(>1e-5)':>12s}  "
              f"{'cz(jday=167)':>13s}  phase")
        for h in range(24):
            if hourly_max.get(h, 0) < 1e-7:
                continue
            cz = solar_zenith(lat, lon, jday0 + 15, h)
            phase = ("☀️ day" if cz > 0.1 else
                     "🌅 transition" if cz > -0.1 else "🌙 night")
            print(f"   {h:>2d}    {hourly_max[h]:>11.3e}  "
                  f"{hourly_count[h]:>10d}    {cz:>+8.3f}     {phase}")

    print(f"\n{'=' * 72}")
    print(f"DONE. Outputs at {OUT}")
    print(f"  problem_cells.csv : {len(problem_cells)} cells to rerun later")
    print(f"  summary_199006_30d.csv : full per-(cell, field) summary")


if __name__ == "__main__":
    main()
