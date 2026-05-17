"""Comprehensive month-long F77 ↔ JAX verification with real GSWP3 forcing.

Reads daily-mean atm forcing from GSWP3 (lat 0.5° × lon 0.5° global grid)
for a set of diverse global cells, disaggregates to hourly forcing, runs
F77 + JAX in lockstep for 30 days, and compares per-cell per-step.

Cells are selected to span major Köppen-Geiger climate zones:
  tropical, temperate, boreal, polar, arid, monsoon, Mediterranean,
  freeze-thaw boundary, coastal — N cells per batch, packed into NPOI=5
  batches (F77 hardcoded NPOI).

Outputs:
  - Console summary: pass/fail per batch, worst diff per (cell, field)
  - CSV files on NAS: detailed (batch, cell, step, field) → F77, JAX, diff
"""
from __future__ import annotations
import os
import sys
import csv
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from netCDF4 import Dataset

jax.config.update("jax_enable_x64", True)

# Set up paths AFTER jax config
sys.path.insert(0, "/home/hangkai/ibis-port/python-port")

from tests.f2py.ibis_f77_wrapper import F77IBIS, NPOI, NSOILAY, NSNOLAY, NPFT, NBAND
from tests.f2py.synthetic_inputs import (
    make_soil_inputs, make_snow_inputs, make_atm_inputs, make_veg_inputs,
)

# Import verifier helpers (they patch JAX before our use)
from verify.timestepping.verify_lsxmain import (
    setup_f77_state, setup_jax_state, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body

GSWP3_DIR = Path("/shared/nas2/Min/GSWP3")
OUTPUT_DIR = Path("/shared/nas2/Hangkai/ibis/verify_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Cell selection — diverse global climates
# ===========================================================================
# Each entry: (label, lat, lon, soil_template, veg_template)
# soil_template indexes into the synthetic 5-cell scenarios:
#   0 = tropical wet  (sand=30, clay=30, wet)
#   1 = arctic        (sand=60, clay=10, frozen)
#   2 = freeze-thaw   (sand=40, clay=20, partial ice)
#   3 = hot desert    (sand=90, clay= 5, dry)
#   4 = wetland       (sand=20, clay=40, saturated)
#
# veg_template same — controls fu, fl, lai, sai, ztop, frac
#
# We pick cells at realistic global locations whose climate broadly matches
# the soil/veg template. GSWP3 forcing then provides the real-world atm.

def latlon_to_grid_idx(lat: float, lon: float) -> tuple[int, int]:
    """GSWP3 grid: lat 89.75→-89.75 (N→S, 360 pts), lon -179.75→179.75 (720 pts), 0.5° step."""
    j = int(round((89.75 - lat) / 0.5))   # row
    i = int(round((lon + 179.75) / 0.5))  # col
    return max(0, min(359, j)), max(0, min(719, i))


def soil_template_for_lat(lat: float) -> int:
    """Pick a soil/veg template index (0-4) from the synthetic 5-cell grid
    based on latitude. Rough climate-zone mapping:
      0 = tropical wet      (|lat| < 15°, plus some equator-near)
      1 = arctic/boreal     (|lat| > 50°)
      2 = temperate         (25° < |lat| < 50°)
      3 = subtropical dry   (15° < |lat| < 25°)
      4 = wetland           (special-case, used for tropical low-lat with hi pr)
    """
    a = abs(lat)
    if a < 15.0:
        return 0
    if a > 50.0:
        return 1
    if a < 25.0:
        return 3
    return 2


LAND_MASK_FILE = Path("/shared/nas2/Min/HWSD/surta.nc")


def load_land_mask() -> np.ndarray:
    """Read landfrac from surta.nc. Returns bool array of shape (360, 720)
    matching GSWP3 grid (lat 89.75→-89.75, lon -179.75→179.75)."""
    ds = Dataset(str(LAND_MASK_FILE), "r")
    landfrac = np.asarray(ds.variables["landfrac"][:])
    lat = np.asarray(ds.variables["lat"][:])
    lon = np.asarray(ds.variables["lon"][:])
    ds.close()
    # Check orientation matches GSWP3 (N to S, -180 to 180)
    # surta.nc may have lat S→N or different lon convention; flip if needed
    if lat[0] < lat[-1]:  # S→N: flip
        landfrac = landfrac[::-1, :]
        lat = lat[::-1]
    if lon[0] >= 0:  # 0..360 convention: roll
        landfrac = np.roll(landfrac, landfrac.shape[1] // 2, axis=1)
    return landfrac > 0.5  # land if >50% land fraction


def build_global_cell_grid(lat_step: float = 10.0, lon_step: float = 20.0) -> list:
    """Generate global grid of cells at coarse resolution, filtering ocean
    via landfrac > 0.5 in surta.nc.

    Returns list of (label, lat, lon, soil_template_idx, veg_template_idx).
    """
    land_mask = load_land_mask()
    print(f"  Land mask: {int(land_mask.sum())} land cells out of {land_mask.size}")

    lats_sample = np.arange(-85.0, 85.0 + 1e-9, lat_step)
    lons_sample = np.arange(-180.0, 180.0 - 1e-9, lon_step)
    cells = []
    for lat in lats_sample:
        for lon in lons_sample:
            j, i = latlon_to_grid_idx(lat, lon)
            if not land_mask[j, i]:
                continue
            t = soil_template_for_lat(lat)
            label = f"L{lat:+05.1f}_{lon:+06.1f}".replace(".", "p")
            cells.append((label, float(lat), float(lon), t, t))
    return cells


# Default global grid — adjust for coverage vs runtime
# Setting lat_step=10°, lon_step=20° gives ~150 land cells (~30 batches).
# Lower step = more cells. lat_step=5°, lon_step=10° → ~600 cells (~120 batches).
CELLS = build_global_cell_grid(lat_step=10.0, lon_step=20.0)


# ===========================================================================
# GSWP3 daily reader for a month
# ===========================================================================

_REF_DATE = date(1860, 1, 1)


def _time_index_for_date(d: date) -> tuple[int, int, int]:
    """Return (decade_start, time_idx_in_file, days_in_month)."""
    decade_start = 1901 + ((d.year - 1901) // 10) * 10
    file_start = date(decade_start, 1, 1)
    idx = (d - file_start).days
    return decade_start, idx


def read_gswp3_month(year: int, month: int, cells: list) -> dict:
    """Read GSWP3 daily forcing for given month and cells.

    Returns dict with arrays of shape (ndays, ncells):
      tas, pr, rsds, rlds, huss, wind, ps (in native units).
    """
    vars_needed = ["tas", "pr", "rsds", "rlds", "huss", "wind", "ps"]
    indices = [latlon_to_grid_idx(lat, lon) for _, lat, lon, _, _ in cells]
    js = np.array([j for j, _ in indices])
    is_ = np.array([i for _, i in indices])

    start = date(year, month, 1)
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    ndays = (next_month - start).days

    decade_start, t0 = _time_index_for_date(start)
    out = {}
    for v in vars_needed:
        fname = GSWP3_DIR / f"{v}_gswp3_{decade_start}_{decade_start + 9}.nc4"
        if not fname.exists():
            raise FileNotFoundError(fname)
        ds = Dataset(str(fname), "r")
        # Read full grid for ndays then extract per cell (avoids netCDF4
        # outer-product indexing which gives shape (ndays, ncells, ncells)
        # instead of (ndays, ncells))
        grid = ds.variables[v][t0:t0 + ndays, :, :]  # (ndays, 360, 720)
        if hasattr(grid, "filled"):
            grid = grid.filled(np.nan)
        # Cast to float64: GSWP3 is stored as float32; using float32 here
        # corrupts subtractive cancellations downstream (e.g. psurf*(1-siga)
        # in canini → za off by ~2e-4 → tropical Δfseng of ~0.27 W/m²).
        grid = np.asarray(grid, dtype=np.float64)
        data = grid[:, js, is_]   # advanced indexing: (ndays, ncells)
        out[v] = data
        ds.close()
    return out


# ===========================================================================
# Daily → hourly disaggregation
# ===========================================================================

def diurnal_disaggregate(daily_forcing: dict, hour: float, jday: int,
                          lat_deg: np.ndarray) -> dict:
    """Convert daily means + lat to hourly atm forcing for one timestep.

    Inputs (all shape (ncells,) for this day):
      tas:   daily-mean ta  (K)
      pr:    daily-mean precip rate (kg/m^2/s)
      rsds:  daily-mean SW down (W/m^2)
      rlds:  daily-mean LW down (W/m^2)
      huss:  daily-mean spec humidity (kg/kg)
      wind:  daily-mean wind speed (m/s)
      ps:    surface pressure (Pa)

    Output (ncells arrays):
      ta, qa, ua, psurf, coszen, fira, solad (2-band), solai (2-band), raina, snowa
    """
    # Solar geometry
    orbit = 2.0 * np.pi * jday / 365.2425
    decl = (0.006918 - 0.399912 * np.cos(orbit) + 0.070257 * np.sin(orbit)
            - 0.006758 * np.cos(2 * orbit) + 0.000907 * np.sin(2 * orbit)
            - 0.002697 * np.cos(3 * orbit) + 0.001480 * np.sin(3 * orbit))
    lat_rad = np.radians(lat_deg)
    # Hour angle: 0 at solar noon, -π at midnight, +π at midnight
    ha = (hour - 12.0) * (2 * np.pi / 24.0)
    coszen = np.maximum(0.0, (np.sin(lat_rad) * np.sin(decl)
                              + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)))

    # SW: distribute daily-mean rsds proportional to coszen / mean_coszen_24h
    # mean_coszen_24h:  analytic integral of max(0, sin*sin + cos*cos*cos(h))
    cos_h0 = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
    h0 = np.arccos(cos_h0)
    mean_cz = (h0 * np.sin(lat_rad) * np.sin(decl)
               + np.cos(lat_rad) * np.cos(decl) * np.sin(h0)) / np.pi
    mean_cz = np.maximum(mean_cz, 1e-8)
    sw_now = daily_forcing["rsds"] * coszen / mean_cz
    sw_now = np.maximum(0.0, sw_now)
    # 50/50 VIS/NIR, direct/diffuse based on coszen
    diffuse_frac = 0.3 + 0.4 * (1.0 - coszen)  # more diffuse low sun
    solar_dir = sw_now * (1.0 - diffuse_frac)
    solar_dif = sw_now * diffuse_frac
    solad = np.zeros((len(lat_deg), 2))
    solai = np.zeros((len(lat_deg), 2))
    solad[:, 0] = 0.5 * solar_dir   # VIS direct
    solad[:, 1] = 0.5 * solar_dir   # NIR direct
    solai[:, 0] = 0.5 * solar_dif   # VIS diffuse
    solai[:, 1] = 0.5 * solar_dif   # NIR diffuse

    # Air temperature: daily mean ± 5K with peak at 14:00
    ta = daily_forcing["tas"] + 5.0 * np.sin((hour - 8.0) * 2 * np.pi / 24.0)

    # Specific humidity: constant daily
    qa = np.maximum(daily_forcing["huss"], 1e-4)

    # Wind speed: constant daily, floor at 0.5
    ua = np.maximum(daily_forcing["wind"], 0.5)

    # Surface pressure: constant
    psurf = daily_forcing["ps"]

    # LW down: constant daily
    fira = daily_forcing["rlds"]

    # Precipitation: uniform over 24h. Split by ta below/above freezing.
    raina = np.where(ta >= 273.16, daily_forcing["pr"], 0.0)
    snowa = np.where(ta < 273.16, daily_forcing["pr"], 0.0)

    # Defensive cast — even if upstream is float64, np.where() and direct
    # assignment can preserve a float32 if source array was 32-bit. JAX
    # then keeps it 32-bit through weak-type promotion, corrupting the
    # subtractive cancellation in canini (psurf - psurf*siga).
    out = {"ta": ta, "qa": qa, "ua": ua, "psurf": psurf,
           "coszen": coszen, "fira": fira,
           "solad": solad, "solai": solai,
           "raina": raina, "snowa": snowa}
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ===========================================================================
# Lockstep runner for a batch of 5 cells
# ===========================================================================

def make_batch_state(cells: list):
    """Build matched F77 and JAX initial state for 5 cells using templates."""
    soil_i = make_soil_inputs()
    snow_i = make_snow_inputs()
    atm_i  = make_atm_inputs()
    veg_i  = make_veg_inputs()
    # The synthetic inputs already have 5 cells with 5 templates; pull out
    # the per-cell templates and reorder to match our batch:
    soil_t_idx = np.array([c[3] for c in cells])
    veg_t_idx  = np.array([c[4] for c in cells])

    def reorder(arr, idx):
        a = np.asarray(arr)
        # Only reorder if first dim is NPOI (per-cell arrays); leave shape-(7,)
        # arrays like hsoi alone.
        if a.ndim >= 1 and a.shape[0] == NPOI:
            return a[idx]
        return a

    soil_new = {k: reorder(v, soil_t_idx) for k, v in soil_i.items()}
    snow_new = {k: reorder(v, soil_t_idx) for k, v in snow_i.items()}
    atm_new  = {k: reorder(v, soil_t_idx) for k, v in atm_i.items()}
    veg_new  = {k: reorder(v, veg_t_idx)  for k, v in veg_i.items()}

    # Round sand/clay
    soil_new["sand"] = np.round(soil_new["sand"])
    soil_new["clay"] = np.round(soil_new["clay"])
    return soil_new, snow_new, atm_new, veg_new


def run_batch(batch_name: str, cells: list, gswp3_data: dict, year: int,
              month: int, n_days: int, csv_dir: Path):
    """Run F77 + JAX in lockstep for n_days, comparing per-step. Write CSV."""
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    lats = np.array([c[1] for c in cells])

    # Worst-diff tracker per (cell, field)
    fields_to_track = [
        ("soil.tsoi[:,0]", lambda f, j: (f["soil"]["tsoi"][:, 0], j.soil.tsoi[:, 0])),
        ("soil.wsoi[:,0]", lambda f, j: (f["soil"]["wsoi"][:, 0], j.soil.wsoi[:, 0])),
        ("soil.wisoi[:,0]", lambda f, j: (f["soil"]["wisoi"][:, 0], j.soil.wisoi[:, 0])),
        ("soil.tg",        lambda f, j: (f["soil"]["tg"], j.soil.tg)),
        ("snow.fi",        lambda f, j: (f["snow"]["fi"], j.snow.fi)),
        ("snow.tsno[:,0]", lambda f, j: (f["snow"]["tsno"][:, 0], j.snow.tsno[:, 0])),
        ("snow.hsno[:,0]", lambda f, j: (f["snow"]["hsno"][:, 0], j.snow.hsno[:, 0])),
        ("canopy.tu",      lambda f, j: (f["veg"]["tu"], j.canopy.tu)),
        ("canopy.ts",      lambda f, j: (f["veg"]["ts"], j.canopy.ts)),
        ("canopy.tl",      lambda f, j: (f["veg"]["tl"], j.canopy.tl)),
        ("canopy.t12",     lambda f, j: (f["veg"]["t12"], j.canopy.t12)),
        ("canopy.t34",     lambda f, j: (f["veg"]["t34"], j.canopy.t34)),
        ("canopy.q12",     lambda f, j: (f["veg"]["q12"], j.canopy.q12)),
        ("canopy.q34",     lambda f, j: (f["veg"]["q34"], j.canopy.q34)),
        ("flux.fvapg",     lambda f, j: (f["flux"]["fvapg"], j.canopy.fvapg)),
        ("flux.fsenu",     lambda f, j: (f["flux"]["fsenu"], j.canopy.fsenu)),
        ("flux.fsenl",     lambda f, j: (f["flux"]["fsenl"], j.canopy.fsenl)),
        ("flux.fseng",     lambda f, j: (f["flux"]["fseng"], j.canopy.fseng)),
        ("stresstu",       lambda f, j: (f["stom"]["stresstu"], j.soil.stresstu)),
        ("stresstl",       lambda f, j: (f["stom"]["stresstl"], j.soil.stresstl)),
    ]

    worst = {(c[0], fn): 0.0 for c in cells for fn, _ in fields_to_track}
    nan_count = 0

    # Per-step CSV
    csv_path = csv_dir / f"{batch_name}_perstep.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "day", "hour", "cell_idx", "cell_name", "field",
                    "F77", "JAX", "abs_diff"])

        for step in range(n_days * 24):
            day_idx = step // 24
            hour = float(step % 24)
            jday = (date(year, month, 1) - date(year, 1, 1)).days + day_idx + 1

            day_data = {k: gswp3_data[k][day_idx, :] for k in gswp3_data}
            atm = diurnal_disaggregate(day_data, hour, jday, lats)

            f77.set_atm(**atm)
            new_state_atm = eqx.tree_at(
                lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                           s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                           s.atm.raina, s.atm.snowa),
                state,
                (jnp.asarray(atm["ta"]), jnp.asarray(atm["qa"]),
                 jnp.asarray(atm["ua"]), jnp.asarray(atm["psurf"]),
                 jnp.asarray(atm["coszen"]),
                 jnp.asarray(atm["solad"]), jnp.asarray(atm["solai"]),
                 jnp.asarray(atm["fira"]),
                 jnp.asarray(atm["raina"]), jnp.asarray(atm["snowa"])),
            )
            f77.call_lsxmain()
            state = timestep_body(new_state_atm, consts, params, DTIME)

            # Collect F77 state
            f77_state = {
                "soil": f77.get_soil(),
                "snow": f77.get_snow(),
                "veg":  f77.get_veg(),
                "flux": f77.get_fluxes(),
                "stom": f77.get_stomata(),
            }
            for fname, getter in fields_to_track:
                f_val, j_val = getter(f77_state, state)
                f_arr = np.asarray(f_val); j_arr = np.asarray(j_val)
                for ci, cell in enumerate(cells):
                    fv = float(f_arr[ci]); jv = float(j_arr[ci])
                    if np.isnan(fv) or np.isnan(jv):
                        nan_count += 1
                        continue
                    d = abs(fv - jv)
                    if d > worst[(cell[0], fname)]:
                        worst[(cell[0], fname)] = d
                    # Write per-step row only for big diffs (saves space)
                    if d > 1e-7:
                        w.writerow([step, day_idx, int(hour), ci, cell[0],
                                    fname, f"{fv:.10e}", f"{jv:.10e}",
                                    f"{d:.3e}"])

    return worst, nan_count


# ===========================================================================
# Main
# ===========================================================================

def main():
    # Test month: pick June 1990 (NH summer, has heat-wave-like conditions)
    YEAR, MONTH = 1990, 6
    N_DAYS = 30

    print(f"=" * 72)
    print(f"GLOBAL F77↔JAX VERIFICATION — GSWP3 real forcing")
    print(f"  Period: {YEAR}-{MONTH:02d}, {N_DAYS} days")
    print(f"  Cells: {len(CELLS)} across global climates")
    print(f"  Batches: {(len(CELLS) + 4) // 5} (NPOI={NPOI} per batch)")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"=" * 72)

    # Read all GSWP3 data ONCE for all cells (efficient)
    print(f"\nReading GSWP3 forcing for all {len(CELLS)} cells...")
    t0 = time.time()
    gswp3_all = read_gswp3_month(YEAR, MONTH, CELLS)
    print(f"  Read {N_DAYS} days × {len(CELLS)} cells × 7 vars in {time.time()-t0:.1f}s")

    # Validate forcing — show range per variable
    print(f"\n  Variable ranges across all cells/days:")
    for v in ["tas", "pr", "rsds", "rlds", "huss", "wind", "ps"]:
        arr = gswp3_all[v]
        print(f"    {v:6s}: [{np.nanmin(arr):.3e}, {np.nanmax(arr):.3e}]")

    # Pack into batches of NPOI=5
    n_batches = (len(CELLS) + NPOI - 1) // NPOI
    all_worst = {}
    total_nans = 0

    for b in range(n_batches):
        batch_cells = CELLS[b*NPOI:(b+1)*NPOI]
        if len(batch_cells) < NPOI:
            # Pad with repeats of last cell
            while len(batch_cells) < NPOI:
                batch_cells.append(batch_cells[-1])
        batch_name = f"batch_{b:02d}_" + "_".join(c[0][:5] for c in batch_cells[:3])
        print(f"\n--- Batch {b}: {batch_name} ---")
        for c in batch_cells:
            print(f"    {c[0]:25s} lat={c[1]:+6.1f} lon={c[2]:+6.1f}")

        # Extract this batch's GSWP3 slice
        batch_indices = [CELLS.index(c) for c in batch_cells]
        gswp3_batch = {k: v[:, batch_indices] for k, v in gswp3_all.items()}

        t0 = time.time()
        worst, nan = run_batch(batch_name, batch_cells, gswp3_batch, YEAR, MONTH,
                                N_DAYS, OUTPUT_DIR)
        elapsed = time.time() - t0
        print(f"    Ran in {elapsed:.1f}s, {nan} NaNs")
        all_worst.update(worst)
        total_nans += nan

        # Per-batch summary
        big_count = sum(1 for v in worst.values() if v > 1e-3)
        max_diff_field = max(worst.items(), key=lambda x: x[1])
        print(f"    Worst diff in batch: {max_diff_field[0][0]} {max_diff_field[0][1]} = {max_diff_field[1]:.3e}")
        print(f"    Fields with |Δ|>1e-3: {big_count} / {len(worst)}")

    # Aggregate summary report
    print(f"\n{'='*72}")
    print(f"AGGREGATE SUMMARY (all batches, {YEAR}-{MONTH:02d}, {N_DAYS} days)")
    print(f"{'='*72}")
    print(f"Total NaN occurrences: {total_nans}")

    # By-field worst diff
    field_max = {}
    for (cell, field), d in all_worst.items():
        if field not in field_max or d > field_max[field][1]:
            field_max[field] = (cell, d)
    print(f"\nPer-field worst diff (across all cells):")
    print(f"  {'field':25s} {'worst_diff':>12s}  worst_cell")
    for field in sorted(field_max.keys()):
        cell, d = field_max[field]
        marker = " 🔴 BIG" if d > 1e-3 else (" 🟡" if d > 1e-6 else " ✅")
        print(f"  {field:25s} {d:>12.3e}  {cell:25s}{marker}")

    # By-cell worst diff
    cell_max = {}
    for (cell, field), d in all_worst.items():
        if cell not in cell_max or d > cell_max[cell][1]:
            cell_max[cell] = (field, d)
    print(f"\nPer-cell worst diff (across all fields):")
    print(f"  {'cell':25s} {'worst_diff':>12s}  worst_field")
    for cell in sorted(cell_max.keys()):
        field, d = cell_max[cell]
        marker = " 🔴 BIG" if d > 1e-3 else (" 🟡" if d > 1e-6 else " ✅")
        print(f"  {cell:25s} {d:>12.3e}  {field:25s}{marker}")

    # Write summary CSV
    summary_csv = OUTPUT_DIR / f"summary_{YEAR}{MONTH:02d}_{N_DAYS}d.csv"
    with open(summary_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "field", "worst_abs_diff"])
        for (cell, field), d in sorted(all_worst.items()):
            w.writerow([cell, field, f"{d:.6e}"])
    print(f"\nSummary CSV: {summary_csv}")

    # Verdict
    big_count = sum(1 for d in all_worst.values() if d > 1e-3)
    if big_count == 0 and total_nans == 0:
        print(f"\n🎉 ALL {len(all_worst)} (cell, field) pairs PASS at |Δ|<1e-3")
    else:
        print(f"\n⚠️  {big_count}/{len(all_worst)} (cell, field) pairs FAIL at |Δ|>1e-3")
        print(f"   See {OUTPUT_DIR} for per-batch detail CSVs")

    return 0 if (big_count == 0 and total_nans == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
