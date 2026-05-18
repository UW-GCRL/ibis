"""Year-long global F77↔JAX verification using GSWP3 forcing.

Runs the IBIS port + reference Fortran in lockstep for 12 consecutive
months (1990-01 through 1990-12) over a global cell grid. Records:

  1. Alignment — max |F77 − JAX| per (cell, field, month) for state vars
     and fluxes. Reports the worst across all space-time.
  2. Physical reasonableness — min/max of key fields per month for both
     F77 and JAX. Flags any NaN, out-of-physical-range, or anomalous
     values.
  3. Daily and monthly aggregates saved as CSV for downstream plotting.

Output: /shared/nas2/Hangkai/ibis/verify_outputs/year_1990/
"""
from __future__ import annotations

import csv
import json
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
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body
from tests.f2py.ibis_f77_wrapper import NPOI

OUTPUT_DIR = Path("/shared/nas2/Hangkai/ibis/verify_outputs/year_1990")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Track these state variables for alignment + reasonableness:
TRACKED_TEMP = [
    ("soil.tg",        "K", 200.0, 350.0),
    ("soil.tsoi[0]",   "K", 200.0, 350.0),
    ("snow.tsno[0]",   "K", 200.0, 350.0),
    ("canopy.tu",      "K", 200.0, 350.0),
    ("canopy.ts",      "K", 200.0, 350.0),
    ("canopy.tl",      "K", 200.0, 350.0),
    ("canopy.t12",     "K", 200.0, 350.0),
    ("canopy.t34",     "K", 200.0, 350.0),
]
TRACKED_FRAC = [
    ("soil.wsoi[0]",   "frac", 0.0, 1.0),
    ("soil.wisoi[0]",  "frac", 0.0, 1.0),
    ("snow.fi",        "frac", 0.0, 1.0),
    ("snow.hsno[0]",   "m",    0.0, 5.0),
    ("stresstu",       "frac", 0.0, 1.0),
    ("stresstl",       "frac", 0.0, 1.0),
    ("canopy.q12",     "kg/kg", 0.0, 0.04),
    ("canopy.q34",     "kg/kg", 0.0, 0.04),
]
TRACKED_FLUX = [
    ("flux.fseng",     "W/m^2", -500.0, 500.0),
    ("flux.fsenu",     "W/m^2", -500.0, 500.0),
    ("flux.fsenl",     "W/m^2", -500.0, 500.0),
    ("flux.fseni",     "W/m^2", -1500.0, 1500.0),
    ("flux.fvapg",     "kg/m^2/s", -1e-3, 1e-3),
]
ALL_TRACKED = TRACKED_TEMP + TRACKED_FRAC + TRACKED_FLUX


def get_field(name, f77_state, jax_state):
    """Extract a tracked field from both F77 state dict and JAX state."""
    if name == "soil.tg":          return f77_state["soil"]["tg"], jax_state.soil.tg
    if name == "soil.tsoi[0]":     return f77_state["soil"]["tsoi"][:, 0], jax_state.soil.tsoi[:, 0]
    if name == "snow.tsno[0]":     return f77_state["snow"]["tsno"][:, 0], jax_state.snow.tsno[:, 0]
    if name == "canopy.tu":        return f77_state["veg"]["tu"], jax_state.canopy.tu
    if name == "canopy.ts":        return f77_state["veg"]["ts"], jax_state.canopy.ts
    if name == "canopy.tl":        return f77_state["veg"]["tl"], jax_state.canopy.tl
    if name == "canopy.t12":       return f77_state["veg"]["t12"], jax_state.canopy.t12
    if name == "canopy.t34":       return f77_state["veg"]["t34"], jax_state.canopy.t34
    if name == "soil.wsoi[0]":     return f77_state["soil"]["wsoi"][:, 0], jax_state.soil.wsoi[:, 0]
    if name == "soil.wisoi[0]":    return f77_state["soil"]["wisoi"][:, 0], jax_state.soil.wisoi[:, 0]
    if name == "snow.fi":          return f77_state["snow"]["fi"], jax_state.snow.fi
    if name == "snow.hsno[0]":     return f77_state["snow"]["hsno"][:, 0], jax_state.snow.hsno[:, 0]
    if name == "stresstu":         return f77_state["stom"]["stresstu"], jax_state.soil.stresstu
    if name == "stresstl":         return f77_state["stom"]["stresstl"], jax_state.soil.stresstl
    if name == "canopy.q12":       return f77_state["veg"]["q12"], jax_state.canopy.q12
    if name == "canopy.q34":       return f77_state["veg"]["q34"], jax_state.canopy.q34
    if name == "flux.fseng":       return f77_state["flux"]["fseng"], jax_state.canopy.fseng
    if name == "flux.fsenu":       return f77_state["flux"]["fsenu"], jax_state.canopy.fsenu
    if name == "flux.fsenl":       return f77_state["flux"]["fsenl"], jax_state.canopy.fsenl
    if name == "flux.fseni":       return f77_state["flux"]["fseni"], jax_state.canopy.fseni
    if name == "flux.fvapg":       return f77_state["flux"]["fvapg"], jax_state.canopy.fvapg
    raise KeyError(name)


def run_batch_year(batch_name: str, cells: list, year: int):
    """Run F77+JAX lockstep for 12 months on a batch. Returns per-month stats."""
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    lats = np.array([c[1] for c in cells])

    # Per-(cell,field) worst |Δ| over the entire year
    worst_year = {(c[0], fn): 0.0 for c in cells for fn, _, _, _ in ALL_TRACKED}
    # Per-(cell,field,month) physical range tracking (min/max for F77 + JAX)
    f77_range = {(c[0], fn, m): [np.inf, -np.inf] for c in cells
                 for fn, _, _, _ in ALL_TRACKED for m in range(1, 13)}
    jax_range = {(c[0], fn, m): [np.inf, -np.inf] for c in cells
                 for fn, _, _, _ in ALL_TRACKED for m in range(1, 13)}
    nan_count = 0
    out_of_range_count = 0

    for month in range(1, 13):
        # Days in this month
        if month == 12:
            next_m = date(year + 1, 1, 1)
        else:
            next_m = date(year, month + 1, 1)
        n_days = (next_m - date(year, month, 1)).days

        # Read this month's GSWP3 for these cells
        t0 = time.time()
        gswp3 = read_gswp3_month(year, month, cells)
        # Filter to just this batch's cells (cells are already this batch's)
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
                        # Track range
                        rk = (cell[0], fname, month)
                        f77_range[rk][0] = min(f77_range[rk][0], fc)
                        f77_range[rk][1] = max(f77_range[rk][1], fc)
                        jax_range[rk][0] = min(jax_range[rk][0], jc)
                        jax_range[rk][1] = max(jax_range[rk][1], jc)
                        # Worst diff
                        d = abs(fc - jc)
                        if d > worst_year[(cell[0], fname)]:
                            worst_year[(cell[0], fname)] = d

        print(f"  [{batch_name}] {year}-{month:02d} done (read {read_t:.1f}s, "
              f"NaN={nan_count}, oor={out_of_range_count})",
              flush=True)

    return worst_year, f77_range, jax_range, nan_count, out_of_range_count


def main():
    YEAR = 1990
    # Use the same global cell grid as the 30-day test
    CELLS = build_global_cell_grid(lat_step=10.0, lon_step=20.0)
    print(f"YEAR-LONG GSWP3 verification {YEAR}, {len(CELLS)} cells, "
          f"12 months, {(len(CELLS) + NPOI - 1) // NPOI} batches", flush=True)

    n_batches = (len(CELLS) + NPOI - 1) // NPOI

    # Aggregate per-(cell, field) worst |Δ| across all batches/year
    all_worst = {}
    all_f77_range = {}
    all_jax_range = {}
    total_nan = 0
    total_oor = 0

    t_total = time.time()
    for b in range(n_batches):
        batch_cells = CELLS[b*NPOI:(b+1)*NPOI]
        if len(batch_cells) < NPOI:
            while len(batch_cells) < NPOI:
                batch_cells.append(batch_cells[-1])
        batch_name = f"b{b:02d}"
        print(f"\n=== Batch {b}/{n_batches}: {batch_name} ===", flush=True)
        for c in batch_cells:
            print(f"    {c[0]:25s} lat={c[1]:+6.1f} lon={c[2]:+6.1f}", flush=True)

        t0 = time.time()
        worst, frng, jrng, nan, oor = run_batch_year(batch_name, batch_cells, YEAR)
        elapsed = time.time() - t0
        print(f"  batch took {elapsed:.0f}s", flush=True)

        all_worst.update(worst)
        all_f77_range.update(frng)
        all_jax_range.update(jrng)
        total_nan += nan; total_oor += oor

        # Incremental save in case of interruption
        with open(OUTPUT_DIR / "worst_by_cell_field.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell", "field", "worst_abs_diff"])
            for (cn, fn), d in sorted(all_worst.items(), key=lambda x: -x[1]):
                w.writerow([cn, fn, f"{d:.3e}"])

    t_total = time.time() - t_total
    print(f"\n=========================================================")
    print(f"YEAR {YEAR} VERIFICATION COMPLETE — total {t_total:.0f}s "
          f"({t_total/3600:.2f}h)")
    print(f"=========================================================")
    print(f"\nNaN count:           {total_nan}")
    print(f"Out-of-range count:  {total_oor}")
    print(f"\nWorst |F77 − JAX| across all space-time, by field class:")
    for cls, fields in (("Temperature", TRACKED_TEMP),
                       ("Fraction/mass", TRACKED_FRAC),
                       ("Flux", TRACKED_FLUX)):
        print(f"  {cls}:")
        for fname, unit, _, _ in fields:
            cell_worst = max(((cn, d) for (cn, fn), d in all_worst.items()
                              if fn == fname), key=lambda x: x[1], default=("",0))
            print(f"    {fname:20s} {cell_worst[1]:>10.3e} {unit}  (cell {cell_worst[0]})")

    # Save full per-month range stats to JSON
    range_out = {
        "year": YEAR,
        "f77_range": {f"{c}|{fn}|{m}": v for (c, fn, m), v in all_f77_range.items()},
        "jax_range": {f"{c}|{fn}|{m}": v for (c, fn, m), v in all_jax_range.items()},
        "worst": {f"{c}|{fn}": d for (c, fn), d in all_worst.items()},
        "nan_count": total_nan,
        "out_of_range_count": total_oor,
        "total_seconds": t_total,
    }
    with open(OUTPUT_DIR / "year_summary.json", "w") as fh:
        json.dump(range_out, fh, indent=2, default=str)
    print(f"\nFull summary → {OUTPUT_DIR}/year_summary.json")


if __name__ == "__main__":
    main()
