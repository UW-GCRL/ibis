"""Real-data initial state builder for 0.5° global IBIS-JAX runs.

Replaces synthetic_inputs.py's 5-template approach with real per-cell
data from HWSD (soil) + ESACCI Land Cover (vegetation).

For each cell:
  - Soil sand/clay/properties from /shared/nas2/Min/HWSD/soita.nc (HWSD2 at 0.5°)
  - Vegetation type from /shared/nas2/Min/ESACCI_LC/ibis_vegtype/vegtype_frac_YYYY.nc
  - Initial moisture / temperature from sensible climatological guesses
  - Snow state = 0 (cold start, model builds up)

Drop-in for `make_state_for_cells(cells)` returning the same
(soil_i, snow_i, atm_i, veg_i) tuple consumed by `setup_jax_state_for_n`.
"""
from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

from ibis.io.soil_init import compute_soil_properties
from ibis.io.veg_init import initialize_vegetation, compute_root_distribution

NSOILAY = 6
NSNOLAY = 3
NPFT = 12
NBAND = 2

# Pre-processed inputs already at 0.5°×0.5° NS-ordered grid (lat 90→-90, lon -180→180)
HWSD_SOITA = "/shared/nas2/Min/HWSD/soita.nc"
ESACCI_VEGTYPE_DIR = "/shared/nas2/Min/ESACCI_LC/ibis_vegtype"

# Canonical IBIS vegetation product: Ramankutty & Foley 1999, Global Potential
# Vegetation Dataset (15 biomes mapping 1:1 to IBIS PFT codes). Same SAGE-
# Wisconsin team as the original Foley 1996 IBIS paper. Source:
#   https://sage.nelson.wisc.edu/data-and-models/datasets/global-potential-vegetation-dataset/
RF_VEGTYPE = "/shared/nas2/Hangkai/ibis/inputs/ramankutty_foley_1999/vegtype_0.5.nc"

# Synthetic-template defaults we keep for the few fields not in datasets
DEFAULT_TG_INITIAL = 280.0     # initial ground T (mild)
DEFAULT_WSOI_INITIAL = 0.40    # 40% saturation as IC; model equilibrates
DEFAULT_FIMAX = 0.95
DEFAULT_HSNOMIN = 0.10
DEFAULT_GS = 0.5
DEFAULT_A10SCALPARAM = 1.0
DEFAULT_A10DAYLIGHT = 21600.0


def _cells_to_grid_indices(cells: list) -> tuple[np.ndarray, np.ndarray]:
    """Convert cell (lat, lon) → (latindex, lonindex) in the 0.5° NS-ordered grid.

    Grid: lat from 89.75°N (row 0) to -89.75°S (row 359), lon from -179.75°W
    (col 0) to 179.75°E (col 719). Each cell is 0.5°.
    """
    lats = np.array([c[1] for c in cells], dtype=np.float64)
    lons = np.array([c[2] for c in cells], dtype=np.float64)
    # lat 89.75 → row 0, lat -89.75 → row 359
    latindex = np.clip(((90.0 - lats) / 0.5).astype(int), 0, 359)
    # lon -179.75 → col 0, lon 179.75 → col 719
    lonindex = np.clip(((lons + 180.0) / 0.5).astype(int), 0, 719)
    return latindex, lonindex


def _read_soil_sand_clay(latindex: np.ndarray, lonindex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read sand/clay from HWSD at the given cell indices.

    soita.nc has shape (layer=6, lat=360, lon=720). Returns sand, clay
    arrays of shape (npoi, nsoilay) in percentages.
    """
    ds = Dataset(HWSD_SOITA, mode='r')
    try:
        sand_grid = np.asarray(ds.variables['sandpct'][:], dtype=np.float64)
        clay_grid = np.asarray(ds.variables['claypct'][:], dtype=np.float64)
    finally:
        ds.close()
    # Advanced indexing: (layer, lat, lon)[layer, npoi] → (layer, npoi) then T → (npoi, layer)
    sand = sand_grid[:, latindex, lonindex].T  # (npoi, 6)
    clay = clay_grid[:, latindex, lonindex].T
    # NaN fallback (45/25 = loam)
    sand = np.where(np.isfinite(sand), sand, 45.0)
    clay = np.where(np.isfinite(clay), clay, 25.0)
    return sand, clay


def _read_vegtype(latindex: np.ndarray, lonindex: np.ndarray, year: int = 2000) -> np.ndarray:
    """Read dominant vegtype per cell from ESACCI 0.5° pre-processed maps."""
    from ibis.io.vegtype_reader import read_vegtype_year
    _, vegdom = read_vegtype_year(ESACCI_VEGTYPE_DIR, year, latindex, lonindex)
    return vegdom


def _read_vegtype_ramankutty(latindex: np.ndarray, lonindex: np.ndarray) -> np.ndarray:
    """Read dominant vegtype per cell from Ramankutty & Foley 1999.

    The CANONICAL IBIS vegetation product (same SAGE-Wisconsin team as Foley
    1996, vegtype values 1-15 match IBIS PFT codes exactly). Time-invariant
    "potential natural vegetation" — what would exist absent human land use.
    """
    ds = Dataset(RF_VEGTYPE, mode='r')
    try:
        # vegtype shape: (time=1, level=1, latitude=360, longitude=720)
        # latitude order: lat[0]=-89.75 (S→N), lon[0]=-179.75 (W→E)
        v_grid = np.asarray(ds.variables['vegtype'][0, 0, :, :], dtype=np.float64)
    finally:
        ds.close()
    # R&F file is N→S ordered (lat[0]=89.75, lat[359]=-89.75) — same as HWSD,
    # so no index flip needed.
    vegdom = v_grid[latindex, lonindex]
    # Ocean cells (NaN in R&F) → fill with 15 (polar desert/rock/ice) as safe default
    vegdom = np.where(np.isfinite(vegdom), vegdom, 15.0)
    # Clip to valid IBIS range
    vegdom = np.clip(vegdom, 1.0, 15.0)
    return vegdom.astype(np.float32)


def _initial_temperatures_by_lat(lats: np.ndarray) -> np.ndarray:
    """Climatological initial soil/canopy temperature guess (K) by latitude.

    Simple sinusoid: ~280 K at equator down to ~250 K at poles. The model
    equilibrates this in <50 yr of spin-up under real forcing.
    """
    # Linear ramp: 295 K at equator, 250 K at pole
    abs_lat = np.abs(lats)
    return 295.0 - 0.5 * abs_lat


def make_real_state_for_cells(cells: list, vegtype_year: int = 2000,
                                vegtype_source: str = "ramankutty") -> tuple[dict, dict, dict, dict]:
    """Build real-data ICs for arbitrary cell list. Returns (soil, snow, atm, veg)
    dicts matching the structure of make_state_for_cells in global_gswp3_test.py.

    Drop-in replacement for synthetic-template ICs.

    Parameters
    ----------
    cells : list of (label, lat, lon, soil_template_idx, veg_template_idx)
    vegtype_year : int
        Year for ESACCI vegtype (ignored when vegtype_source='ramankutty').
    vegtype_source : str
        Either 'ramankutty' (default, canonical IBIS Ramankutty & Foley 1999)
        or 'esacci' (Min's pre-processed ESACCI Land Cover 1992-2020).
    """
    n = len(cells)
    lats = np.array([c[1] for c in cells], dtype=np.float64)

    latindex, lonindex = _cells_to_grid_indices(cells)
    sand, clay = _read_soil_sand_clay(latindex, lonindex)  # (n, 6)

    # Soil hydraulic properties from sand/clay via IBIS texture lookup
    sp = compute_soil_properties(sand, clay)
    # Returns dict with: poros, sfield, swilt, bex, suction, hydraul, albsav,
    #                    albsan, csoi, rhosoi, z0soi, hsoi
    print(f"  Soil properties computed for {n} cells from HWSD sand/clay.")

    # Initial soil moisture: 40% saturation per layer (model equilibrates)
    wsoi_init = np.full((n, NSOILAY), DEFAULT_WSOI_INITIAL, dtype=np.float64)
    wisoi_init = np.zeros((n, NSOILAY), dtype=np.float64)

    # Initial soil + ground temperature: climatological guess by latitude
    tref = _initial_temperatures_by_lat(lats)  # (n,)
    tsoi_init = np.tile(tref[:, None], (1, NSOILAY))
    tg_init = tref.copy()
    ti_init = tref.copy()  # snow base temperature (used if no snow)

    # Build soil dict (matches keys used by setup_jax_state_for_n)
    soil_i = {
        "tsoi":   tsoi_init,
        "wsoi":   wsoi_init,
        "wisoi":  wisoi_init,
        "sand":   sand,
        "clay":   clay,
        "poros":  sp["poros"],
        "hydraul": sp["hydraul"],
        "suction": sp["suction"],
        "bex":    sp["bex"],
        "sfield": sp["sfield"],
        "swilt":  sp["swilt"],
        "wpud":   np.zeros(n),
        "wipud":  np.zeros(n),
        "tg":     tg_init,
        "hsoi":   sp["hsoi"],   # (nsoilay,)
        "ti":     ti_init,
    }

    # Snow dict — cold-start: no snow
    snow_i = {
        "fi":   np.zeros(n),
        "tsno": np.tile(tref[:, None], (1, NSNOLAY)),
        "hsno": np.zeros((n, NSNOLAY)),
    }

    # Vegetation: read dominant vegtype from canonical R&F (default) or ESACCI
    if vegtype_source == "ramankutty":
        vegdom = _read_vegtype_ramankutty(latindex, lonindex)
        print(f"  Vegetation initialized from CANONICAL Ramankutty & Foley 1999.")
    elif vegtype_source == "esacci":
        vegdom = _read_vegtype(latindex, lonindex, year=vegtype_year)
        print(f"  Vegetation initialized from ESACCI vegtype_{vegtype_year}.")
    else:
        raise ValueError(f"vegtype_source must be 'ramankutty' or 'esacci', got {vegtype_source!r}")
    # Warmest-month T guess: use lat-based estimate +10 K (rough)
    tw_guess = tref - 273.15 + 10.0
    vresult = initialize_vegetation(vegdom, tw=tw_guess, npft=NPFT, nsoilay=NSOILAY)

    # Build veg dict matching keys used by setup_jax_state_for_n
    # Note: initialize_vegetation returns lai/sai/ztop/zbot as (n, 2) [lower, upper]
    veg_i = {
        "tu":     tref + 1.0,     # upper canopy T (slight offset from ground)
        "ts":     tref + 0.5,     # stem T
        "tl":     tref + 0.5,     # lower canopy T
        "t12":    tref,
        "t34":    tref,
        "q12":    np.full(n, 0.008),
        "q34":    np.full(n, 0.008),
        "wliqu":  np.zeros(n),
        "wliqs":  np.zeros(n),
        "wliql":  np.zeros(n),
        "wsnou":  np.zeros(n),
        "wsnos":  np.zeros(n),
        "wsnol":  np.zeros(n),
        "tlsub":  tref,            # snow base
        "fu":     vresult["fu"].astype(np.float64),
        "fl":     vresult["fl"].astype(np.float64),
        "lai":    vresult["lai"].astype(np.float64),    # (n, 2)
        "sai":    vresult["sai"].astype(np.float64),    # (n, 2)
        "ztop":   vresult["ztop"].astype(np.float64),   # (n, 2)
        "zbot":   vresult["zbot"].astype(np.float64),   # (n, 2)
        "frac":   vresult.get("frac",
                              np.zeros((n, NPFT), dtype=np.float64)),  # PFT fractions
        "froot":  vresult["froot"].astype(np.float64),  # (nsoilay, 2) or (n, nsoilay)
    }

    # Atmospheric dict (just initial; will be overwritten each timestep by forcing)
    atm_i = {
        "ta":    tref,
        "qa":    np.full(n, 0.008),
        "ua":    np.full(n, 3.0),
        "psurf": np.full(n, 95000.0),
        "coszen": np.zeros(n),
        "solad": np.zeros((n, NBAND)),
        "solai": np.zeros((n, NBAND)),
        "fira":  np.full(n, 330.0),
        "raina": np.zeros(n),
        "snowa": np.zeros(n),
    }

    return soil_i, snow_i, atm_i, veg_i
