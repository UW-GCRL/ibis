"""Multi-year global F77↔JAX verification using GSWP3 forcing.

Extension of global_gswp3_year.py to support a year RANGE with state
chained across year boundaries. Same alignment + physical-reasonableness
checks as the year-long version, plus:

  - Per-year summary CSV (worst |Δ| per cell-field per year)
  - Final aggregate JSON over the full year range
  - Optional state checkpointing (every N years) for crash recovery

Output: /shared/nas2/Hangkai/ibis/verify_outputs/years_<start>_<end>/

Usage:
    PYTHONPATH=python-port verify/.venv/bin/python -u -m \\
        verify.timestepping.pipeline.global_gswp3_multiyear \\
        --year-start 1951 --year-end 2000 \\
        --batch-start 0 --batch-end 6 --suffix _p0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, "/home/hangkai/ibis-port/python-port")

from verify.timestepping.pipeline.global_gswp3_test import (
    diurnal_disaggregate, read_gswp3_month, make_batch_state,
    soil_template_for_lat, build_global_cell_grid,
)
from verify.timestepping.verify_lsxmain import (
    setup_f77_state, setup_jax_state, DTIME,
)
from verify.timestepping.pipeline.global_gswp3_year import (
    TRACKED_TEMP, TRACKED_FRAC, TRACKED_FLUX, ALL_TRACKED, get_field,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body
from tests.f2py.ibis_f77_wrapper import NPOI


def run_one_year_continuing(f77, state, cells, year: int, batch_name: str,
                             consts, params, lats):
    """Run F77+JAX lockstep for one year. f77 state mutates in place;
    returns new JAX state plus per-year diff stats."""
    worst_year = {(c[0], fn): 0.0 for c in cells for fn, _, _, _ in ALL_TRACKED}
    f77_range = {(c[0], fn, m): [np.inf, -np.inf] for c in cells
                 for fn, _, _, _ in ALL_TRACKED for m in range(1, 13)}
    jax_range = {(c[0], fn, m): [np.inf, -np.inf] for c in cells
                 for fn, _, _, _ in ALL_TRACKED for m in range(1, 13)}
    nan_count = 0
    out_of_range_count = 0

    for month in range(1, 13):
        if month == 12:
            next_m = date(year + 1, 1, 1)
        else:
            next_m = date(year, month + 1, 1)
        n_days = (next_m - date(year, month, 1)).days

        t0 = time.time()
        gswp3 = read_gswp3_month(year, month, cells)
        read_t = time.time() - t0

        for day_idx in range(n_days):
            jday = (date(year, month, 1) - date(year, 1, 1)).days + day_idx + 1
            day_data = {k: gswp3[k][day_idx, :] for k in gswp3}
            for hour in range(24):
                atm = diurnal_disaggregate(day_data, float(hour), jday, lats)
                f77.set_atm(**atm)
                new_state_atm = eqx.tree_at(
                    lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                               s.atm.coszen, s.atm.solad, s.atm.solai,
                               s.atm.fira, s.atm.raina, s.atm.snowa),
                    state,
                    tuple(jnp.asarray(atm[k]) for k in
                          ("ta","qa","ua","psurf","coszen","solad","solai",
                           "fira","raina","snowa")),
                )
                f77.call_lsxmain()
                state = timestep_body(new_state_atm, consts, params, DTIME)

                f77_state = {
                    "soil": f77.get_soil(),
                    "snow": f77.get_snow(),
                    "veg":  f77.get_veg(),
                    "flux": f77.get_fluxes(),
                    "stom": f77.get_stomata(),
                }
                for fname, _u, lo, hi in ALL_TRACKED:
                    fv_arr, jv_arr = get_field(fname, f77_state, state)
                    fv = np.asarray(fv_arr); jv = np.asarray(jv_arr)
                    for ci, cell in enumerate(cells):
                        fc, jc = float(fv[ci]), float(jv[ci])
                        if np.isnan(fc) or np.isnan(jc):
                            nan_count += 1
                            continue
                        if fc < lo or fc > hi or jc < lo or jc > hi:
                            out_of_range_count += 1
                        rk = (cell[0], fname, month)
                        f77_range[rk][0] = min(f77_range[rk][0], fc)
                        f77_range[rk][1] = max(f77_range[rk][1], fc)
                        jax_range[rk][0] = min(jax_range[rk][0], jc)
                        jax_range[rk][1] = max(jax_range[rk][1], jc)
                        d = abs(fc - jc)
                        if d > worst_year[(cell[0], fname)]:
                            worst_year[(cell[0], fname)] = d

        print(f"  [{batch_name}] {year}-{month:02d} done (read {read_t:.1f}s, "
              f"NaN={nan_count}, oor={out_of_range_count})",
              flush=True)

    return state, worst_year, f77_range, jax_range, nan_count, out_of_range_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=1951)
    parser.add_argument("--year-end",   type=int, default=2000)
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-end",   type=int, default=None)
    parser.add_argument("--suffix",      type=str, default="")
    parser.add_argument("--output-root", type=str,
                        default="/shared/nas2/Hangkai/ibis/verify_outputs")
    args = parser.parse_args()

    year_start = args.year_start
    year_end   = args.year_end
    sfx        = args.suffix

    OUTPUT_DIR = Path(args.output_root) / f"years_{year_start}_{year_end}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_YEAR_DIR = OUTPUT_DIR / "per_year_stats"
    PER_YEAR_DIR.mkdir(exist_ok=True)

    CELLS = build_global_cell_grid(lat_step=10.0, lon_step=20.0)
    n_batches = (len(CELLS) + NPOI - 1) // NPOI
    bs = args.batch_start
    be = args.batch_end if args.batch_end is not None else n_batches
    n_years = year_end - year_start + 1
    print(f"MULTI-YEAR GSWP3 verification {year_start}-{year_end} ({n_years} yr), "
          f"{len(CELLS)} cells, batches [{bs}, {be}) of {n_batches}{sfx}",
          flush=True)
    print(f"Output → {OUTPUT_DIR}", flush=True)

    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    # Aggregate worst |Δ| per (cell, field) ACROSS the full year range
    all_worst = {}
    total_nan = 0
    total_oor = 0
    t_total = time.time()

    for b in range(bs, be):
        batch_cells = CELLS[b*NPOI:(b+1)*NPOI]
        while len(batch_cells) < NPOI:
            batch_cells.append(batch_cells[-1])
        batch_name = f"b{b:02d}"
        lats = np.array([c[1] for c in batch_cells])

        print(f"\n=== Batch {b}/{n_batches}: {batch_name} ===", flush=True)
        for c in batch_cells:
            print(f"    {c[0]:25s} lat={c[1]:+6.1f} lon={c[2]:+6.1f}", flush=True)

        # Initialize state ONCE per batch (year-chain starts here)
        soil_i, snow_i, atm_i, veg_i = make_batch_state(batch_cells)
        f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
        state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)

        # Roll-up per (cell, field) worst |Δ| across all years for this batch
        batch_worst = {(c[0], fn): 0.0 for c in batch_cells
                       for fn, _, _, _ in ALL_TRACKED}
        batch_nan = 0
        batch_oor = 0

        t_batch = time.time()
        for year in range(year_start, year_end + 1):
            t_yr = time.time()
            print(f"\n  --- {batch_name} year {year} ({year - year_start + 1}/{n_years}) ---",
                  flush=True)

            state, worst_y, frng_y, jrng_y, nan_y, oor_y = (
                run_one_year_continuing(
                    f77, state, batch_cells, year, batch_name,
                    consts, params, lats,
                )
            )

            # Save per-year summary for this batch
            per_year_data = {
                "year": year,
                "batch": batch_name,
                "nan_count": nan_y,
                "out_of_range_count": oor_y,
                "worst": {f"{c}|{fn}": d for (c, fn), d in worst_y.items()},
                "f77_range": {f"{c}|{fn}|{m}": v for (c, fn, m), v in frng_y.items()},
                "jax_range": {f"{c}|{fn}|{m}": v for (c, fn, m), v in jrng_y.items()},
                "elapsed_seconds": time.time() - t_yr,
            }
            with open(PER_YEAR_DIR / f"{batch_name}_year_{year}{sfx}.json", "w") as fh:
                json.dump(per_year_data, fh, indent=1, default=str)

            # Update batch-level aggregates
            for k, d in worst_y.items():
                if d > batch_worst[k]:
                    batch_worst[k] = d
            batch_nan += nan_y
            batch_oor += oor_y

            elapsed_yr = time.time() - t_yr
            print(f"  --- {batch_name} year {year} done in {elapsed_yr/60:.1f} min "
                  f"(cum {(time.time()-t_batch)/3600:.2f}h) ---",
                  flush=True)

            # Incremental save of batch-level worst (in case of interruption)
            with open(OUTPUT_DIR / f"worst_by_cell_field_{batch_name}{sfx}.csv",
                      "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["cell", "field", "worst_abs_diff"])
                for (cn, fn), d in sorted(batch_worst.items(), key=lambda x: -x[1]):
                    w.writerow([cn, fn, f"{d:.3e}"])

        # End of batch: roll up into global aggregate
        all_worst.update(batch_worst)
        total_nan += batch_nan
        total_oor += batch_oor

        t_batch = time.time() - t_batch
        print(f"\n  Batch {batch_name} TOTAL: {t_batch/3600:.2f}h, "
              f"{batch_nan} NaN, {batch_oor} oor", flush=True)

    t_total = time.time() - t_total
    print(f"\n=========================================================")
    print(f"MULTI-YEAR {year_start}-{year_end} VERIFICATION COMPLETE")
    print(f"Total wall-clock: {t_total:.0f}s ({t_total/3600:.2f}h)")
    print(f"=========================================================")
    print(f"\nNaN count:           {total_nan}")
    print(f"Out-of-range count:  {total_oor}")
    print(f"\nWorst |F77 − JAX| across all space-time-fields, by class:")
    for cls, fields in (("Temperature", TRACKED_TEMP),
                       ("Fraction/mass", TRACKED_FRAC),
                       ("Flux", TRACKED_FLUX)):
        print(f"  {cls}:")
        for fname, unit, _, _ in fields:
            cell_worst = max(((cn, d) for (cn, fn), d in all_worst.items()
                              if fn == fname), key=lambda x: x[1], default=("",0))
            print(f"    {fname:20s} {cell_worst[1]:>10.3e} {unit}  (cell {cell_worst[0]})")

    final = {
        "year_start": year_start, "year_end": year_end,
        "n_years": n_years,
        "worst": {f"{c}|{fn}": d for (c, fn), d in all_worst.items()},
        "nan_count": total_nan,
        "out_of_range_count": total_oor,
        "total_seconds": t_total,
    }
    with open(OUTPUT_DIR / f"multiyear_summary{sfx}.json", "w") as fh:
        json.dump(final, fh, indent=2, default=str)
    print(f"\nFinal summary → {OUTPUT_DIR}/multiyear_summary{sfx}.json")


if __name__ == "__main__":
    main()
