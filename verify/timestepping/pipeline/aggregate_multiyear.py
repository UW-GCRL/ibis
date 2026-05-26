"""Aggregate the per-year JSONs from a multi-year run into a final summary.

Reads all `b<NN>_year_<YYYY>_p<NN>.json` files under
`<root>/years_<start>_<end>/per_year_stats/` and produces:

  - Overall worst |Δ| per (cell, field) across the full year range
  - Per-year time-series of worst |Δ| per field (to check compounding)
  - Per-cell aggregate stats (median, p95, max) across years
  - Identification of NaN-affected cells

Usage:
    PYTHONPATH=python-port verify/.venv/bin/python -u \\
        -m verify.timestepping.pipeline.aggregate_multiyear \\
        --root /shared/nas2/Hangkai/ibis/verify_outputs/years_1951_2000 \\
        --year-start 1951 --year-end 1985
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Directory containing per_year_stats/")
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end",   type=int, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    per_year_dir = root / "per_year_stats"
    out_file = root / f"aggregate_{args.year_start}_{args.year_end}.json"

    # (cell, field) -> list of (year, batch, max_diff)
    by_cell_field = defaultdict(list)
    # field -> {year: max across all cells}
    per_year_max_field = defaultdict(dict)
    # cell -> set of years with NaN events
    cells_nan_years = defaultdict(set)

    total_nan = 0
    total_oor = 0
    n_files = 0

    for f in sorted(per_year_dir.glob("b*_year_*.json")):
        d = json.load(open(f))
        y = d["year"]
        if not (args.year_start <= y <= args.year_end):
            continue
        n_files += 1
        total_nan += d.get("nan_count", 0)
        total_oor += d.get("out_of_range_count", 0)

        for key, val in d["worst"].items():
            cell, fld = key.split("|")
            v = float(val)
            by_cell_field[(cell, fld)].append((y, d["batch"], v))
            cur = per_year_max_field[fld].get(y, 0.0)
            if v > cur:
                per_year_max_field[fld][y] = v

        if d.get("nan_count", 0) > 0:
            # Find which cells went NaN — they have worst=0 across all fields
            cells_in_batch = set(k.split("|")[0] for k in d["worst"].keys())
            for c in cells_in_batch:
                fields_zero = sum(
                    1 for k, v in d["worst"].items()
                    if k.startswith(c + "|") and float(v) == 0.0
                )
                fields_total = sum(
                    1 for k in d["worst"].keys() if k.startswith(c + "|")
                )
                # If ALL fields are 0 for this cell, it's NaN-affected
                if fields_total > 0 and fields_zero == fields_total:
                    cells_nan_years[c].add(y)

    # Compute summary
    # Overall worst per field across cells and years
    overall_worst_per_field = {}
    for (cell, fld), rows in by_cell_field.items():
        for y, b, v in rows:
            cur = overall_worst_per_field.get(fld, (0.0, "", 0))
            if v > cur[0]:
                overall_worst_per_field[fld] = (v, cell, y)

    # Per-cell aggregate (max across years per field)
    per_cell_max = defaultdict(dict)
    for (cell, fld), rows in by_cell_field.items():
        values = [v for _, _, v in rows]
        if values:
            per_cell_max[cell][fld] = float(max(values))

    # NaN-affected cells (with continuous NaN from year X to end of range)
    nan_summary = {}
    for c, years in cells_nan_years.items():
        years_sorted = sorted(years)
        nan_summary[c] = {
            "first_nan_year": years_sorted[0],
            "n_nan_years": len(years_sorted),
            "years": years_sorted,
        }

    # Print human-readable report
    n_years = args.year_end - args.year_start + 1
    print(f"\n{'='*72}")
    print(f"MULTI-YEAR AGGREGATE — {args.year_start}-{args.year_end} ({n_years} years)")
    print(f"{'='*72}")
    print(f"Per-year files: {n_files}")
    print(f"Total NaN events: {total_nan}")
    print(f"Total out-of-range warnings: {total_oor}")
    print(f"NaN-affected cells: {len(nan_summary)}")
    for c, info in nan_summary.items():
        print(f"  {c}: NaN starting {info['first_nan_year']} ({info['n_nan_years']} years affected)")

    print(f"\nWorst |Δ| per field across ALL cells × {n_years} years:")
    cats = {
        "TEMPERATURE (K)": ["soil.tg","soil.tsoi[0]","snow.tsno[0]",
                            "canopy.tu","canopy.ts","canopy.tl",
                            "canopy.t12","canopy.t34"],
        "WATER / FRAC":    ["soil.wsoi[0]","soil.wisoi[0]","snow.fi",
                            "snow.hsno[0]","stresstu","stresstl",
                            "canopy.q12","canopy.q34"],
        "FLUX (W/m^2)":    ["flux.fseng","flux.fsenu","flux.fsenl",
                            "flux.fseni","flux.fvapg"],
    }
    for cat, fields in cats.items():
        print(f"\n  {cat}")
        for fld in fields:
            if fld in overall_worst_per_field:
                v, cell, yr = overall_worst_per_field[fld]
                print(f"    {fld:22s} {v:>11.3e}  @{cell:25s} year {yr}")

    # Per-year temporal evolution (does drift compound?)
    print(f"\nYear-by-year worst |Δ| (to check for compounding):")
    key_fields = ["canopy.tu", "canopy.tl", "soil.tsoi[0]", "soil.tg",
                  "flux.fsenu", "flux.fsenl"]
    print(f"  {'year':>5s} " + " ".join(f"{f:>11s}" for f in key_fields))
    for y in range(args.year_start, args.year_end + 1):
        row = " ".join(
            f"{per_year_max_field[f].get(y, 0):>11.3e}" for f in key_fields
        )
        print(f"  {y:>5d} {row}")

    # Statistics on per-cell drift (median, p95, max across 116 healthy cells)
    healthy_cells = [c for c in per_cell_max if c not in nan_summary]
    print(f"\nGlobal distribution across {len(healthy_cells)} healthy cells, all years:")
    print(f"  {'field':22s} {'median':>11s} {'p95':>11s} {'max':>11s}")
    for fld in key_fields + ["flux.fseni", "soil.wsoi[0]"]:
        vals = np.array([per_cell_max[c].get(fld, 0.0) for c in healthy_cells])
        if len(vals) > 0:
            print(f"  {fld:22s} {np.median(vals):>11.3e} {np.percentile(vals, 95):>11.3e} {np.max(vals):>11.3e}")

    # Save full aggregate JSON
    aggregate = {
        "year_start": args.year_start,
        "year_end":   args.year_end,
        "n_years":    n_years,
        "n_per_year_files": n_files,
        "total_nan_events": total_nan,
        "total_oor_warnings": total_oor,
        "nan_affected_cells": nan_summary,
        "overall_worst_per_field": {
            f: {"value": v, "cell": c, "year": y}
            for f, (v, c, y) in overall_worst_per_field.items()
        },
        "per_year_max_field": {
            f: {str(y): v for y, v in yrs.items()}
            for f, yrs in per_year_max_field.items()
        },
        "per_cell_max": {
            c: dict(d) for c, d in per_cell_max.items()
        },
        "healthy_cells_n": len(healthy_cells),
    }
    with open(out_file, "w") as fh:
        json.dump(aggregate, fh, indent=1, default=str)
    print(f"\nAggregate JSON → {out_file}")


if __name__ == "__main__":
    main()
