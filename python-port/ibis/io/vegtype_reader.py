"""Reader for ESACCI land-cover derived IBIS vegetation type fractions.

Files: vegtype_frac_YYYY.nc (available 1992–2020)
Grid:  360 lat (N→S, 89.75→-89.75) × 720 lon (W→E, -179.75→179.75) × 15 vegtypes
       — identical 0.5° grid to GSWP3.

Variables
---------
vegfrac          (lat, lon, vegtype) float32  — fraction of each IBIS vegtype (1-15)
vegtype_dominant (lat, lon)          float32  — dominant IBIS vegtype (1-15)

Usage
-----
    vegfrac, vegdom = read_vegtype_year(
        '/shared/nas2/Min/ESACCI_LC/ibis_vegtype', 1995,
        latindex, lonindex,
    )
    # vegfrac: (npoi, 15), vegdom: (npoi,)
"""

from __future__ import annotations

import os
import numpy as np


def read_vegtype_year(
    vegtype_dir: str,
    year: int,
    latindex: np.ndarray,
    lonindex: np.ndarray,
    year_min: int = 1992,
    year_max: int = 2020,
) -> tuple[np.ndarray, np.ndarray]:
    """Load land-point vegtype fractions and dominant type for one year.

    Parameters
    ----------
    vegtype_dir : str
        Directory containing vegtype_frac_YYYY.nc files.
    year : int
        Target year. Clamped to [year_min, year_max].
    latindex : (npoi,) int array
        Row indices into the 360-row N→S grid (0 = 89.75°N).
    lonindex : (npoi,) int array
        Column indices into the 720-column grid (0 = -179.75°E).
    year_min, year_max : int
        Available year range; out-of-range years are clamped silently.

    Returns
    -------
    vegfrac : (npoi, 15) float32
        Fraction of each IBIS vegtype (types 1–15, index 0–14).
    vegdom : (npoi,) float32
        Dominant IBIS vegtype at each land point (1–15).
    """
    import netCDF4 as nc

    year_clamped = max(year_min, min(year, year_max))
    if year_clamped != year:
        import warnings
        warnings.warn(
            f"vegtype year {year} out of range [{year_min}, {year_max}]; "
            f"using {year_clamped}.",
            stacklevel=2,
        )

    path = os.path.join(vegtype_dir, f"vegtype_frac_{year_clamped}.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ESACCI vegtype file not found: {path}")

    ds = nc.Dataset(path, mode='r')
    try:
        # vegfrac: (lat=360, lon=720, vegtype=15) float32
        vf = np.array(ds.variables['vegfrac'][:], dtype=np.float32)
        # vegtype_dominant: (lat=360, lon=720) float32
        vd = np.array(ds.variables['vegtype_dominant'][:], dtype=np.float32)
    finally:
        ds.close()

    vegfrac = vf[latindex, lonindex, :]   # (npoi, 15)
    vegdom = vd[latindex, lonindex]       # (npoi,)

    return vegfrac, vegdom


def dominant_vegtype(vegfrac: np.ndarray) -> np.ndarray:
    """Return dominant vegtype (1-based) from fraction array.

    Parameters
    ----------
    vegfrac : (npoi, 15) float32

    Returns
    -------
    (npoi,) int32 — dominant vegtype index in [1, 15]
    """
    return (np.argmax(vegfrac, axis=1) + 1).astype(np.int32)
