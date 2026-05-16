"""GSWP3 transient climate forcing reader for IBIS-JAX.

GSWP3 files are organized as decade files:
    {var}_gswp3_{start_year}_{end_year}.nc4

Each file has dimensions (time, lat, lon):
    lat : 360 values, 89.75 → -89.75  (N to S, 0.5° step)
    lon : 720 values, -179.75 → 179.75 (0.5° step)
    time: daily, in 'days since 1860-01-01'

Available variables and native units:
    tas     K           daily mean surface air temperature
    tasmax  K           daily maximum surface air temperature
    tasmin  K           daily minimum surface air temperature
    pr      kg m-2 s-1  precipitation flux
    rsds    W m-2       surface downwelling shortwave radiation
    rlds    W m-2       surface downwelling longwave radiation
    huss    kg kg-1     near-surface specific humidity
    wind    m s-1       wind speed at 10 m
    ps      Pa          surface pressure
    rhs     %           near-surface relative humidity

Usage
-----
    reader = GSWP3Reader('/shared/nas2/Min/GSWP3')
    data = reader.read_month(1980, 0, latindex, lonindex)
    # data['tas']  shape (ndays_in_month, npoi)
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All variables available in the dataset
GSWP3_VARS = ('tas', 'tasmax', 'tasmin', 'pr', 'rsds', 'rlds',
               'huss', 'wind', 'ps', 'rhs')

# Reference date for the 'days since' time axis
_REF_DATE = date(1860, 1, 1)

# Days per month for non-leap and leap years
_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_IN_MONTH_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _days_in_month(year: int, month: int) -> int:
    """Number of days in month (month 0-based)."""
    return (_DAYS_IN_MONTH_LEAP if _is_leap(year) else _DAYS_IN_MONTH)[month]


def _decade_bounds(year: int) -> tuple[int, int]:
    """Return (start, end) year of the decade file containing *year*."""
    # Files: 1901-1910, 1911-1920, ..., 2001-2010
    start = 1901 + ((year - 1901) // 10) * 10
    return start, start + 9


def _time_index(year: int, month: int, day: int) -> int:
    """Return the index within the decade file for the given date.

    Parameters
    ----------
    year, month : int
        Year and month (month 0-based).
    day : int
        Day within month (0-based).
    """
    dec_start, _ = _decade_bounds(year)
    t0 = date(dec_start, 1, 1)
    t1 = date(year, month + 1, day + 1)
    return (t1 - t0).days


def _month_time_slice(year: int, month: int) -> tuple[int, int]:
    """Return (i_start, i_end) time indices for a full month in the decade file."""
    i0 = _time_index(year, month, 0)
    ndays = _days_in_month(year, month)
    return i0, i0 + ndays


# ---------------------------------------------------------------------------
# GSWP3Reader
# ---------------------------------------------------------------------------

class GSWP3Reader:
    """Reader for GSWP3 decade NetCDF4 files.

    Parameters
    ----------
    data_dir : str
        Directory containing the GSWP3 .nc4 files.
    variables : tuple of str, optional
        Subset of variables to load. Defaults to all 10.

    Notes
    -----
    File handles are opened lazily on first access and kept open for the
    lifetime of the reader to avoid repeated open/close overhead.
    Call ``close()`` (or use as a context manager) when done.
    """

    def __init__(self, data_dir: str,
                 variables: tuple[str, ...] = GSWP3_VARS):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"GSWP3 data directory not found: {data_dir}")
        self.data_dir = data_dir
        self.variables = variables
        # Persistent handles: keyed by (var, decade_start_year)
        self._handles: dict[tuple[str, int], object] = {}

    def _decade_path(self, var: str, year: int) -> str:
        s, e = _decade_bounds(year)
        fname = f"{var}_gswp3_{s}_{e}.nc4"
        return os.path.join(self.data_dir, fname)

    def _get_handle(self, var: str, year: int):
        """Return a cached open netCDF4.Dataset for (var, decade containing year)."""
        import netCDF4 as nc
        decade_start = _decade_bounds(year)[0]
        key = (var, decade_start)
        if key not in self._handles:
            path = self._decade_path(var, year)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"GSWP3 file not found: {path}"
                )
            self._handles[key] = nc.Dataset(path, mode='r')
        return self._handles[key]

    def close(self) -> None:
        """Close all open file handles."""
        for ds in self._handles.values():
            try:
                ds.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _check_files(self, year: int) -> list[str]:
        """Return list of missing files for the given year."""
        missing = []
        for v in self.variables:
            p = self._decade_path(v, year)
            if not os.path.exists(p):
                missing.append(p)
        return missing

    def read_month(
        self,
        year: int,
        month: int,
        latindex: np.ndarray,
        lonindex: np.ndarray,
        fill_value: float = 1e20,
    ) -> dict[str, np.ndarray]:
        """Read one month of GSWP3 data for specified land points.

        Parameters
        ----------
        year : int
        month : int
            0-based month index (0 = January).
        latindex : np.ndarray, shape (npoi,)
            Row indices into the GSWP3 lat dimension (N→S, 0 = 89.75°N).
        lonindex : np.ndarray, shape (npoi,)
            Column indices into the GSWP3 lon dimension (0 = -179.75°E).
        fill_value : float
            Fill value in source files; replaced with NaN.

        Returns
        -------
        dict mapping variable name → np.ndarray of shape (ndays, npoi), float32.
        Also includes 'ndays', 'year', 'month' as metadata keys.
        """
        i0, i1 = _month_time_slice(year, month)
        ndays = i1 - i0
        result: dict[str, np.ndarray] = {}

        for var in self.variables:
            land_data_fallback = None  # reset for each variable
            # Retry once on failure to recover from stale NFS file handles
            for attempt in range(2):
                try:
                    ds = self._get_handle(var, year)
                    v = ds.variables[var]
                    # Read full spatial grid then select land points with numpy paired indexing.
                    # netCDF4 does not support paired fancy indexing on two dimensions —
                    # v[t, latindex, lonindex] would be outer-product (npoi×npoi), not paired.
                    data_grid = np.array(v[i0:i1, :, :], dtype=np.float32)  # (ndays, nlat, nlon)
                    break
                except Exception as exc:
                    if attempt == 0:
                        # Stale handle (e.g. NFS timeout) — close and reopen
                        decade_start = _decade_bounds(year)[0]
                        key = (var, decade_start)
                        if key in self._handles:
                            try:
                                self._handles[key].close()
                            except Exception:
                                pass
                            del self._handles[key]
                    else:
                        # Permanent failure — use fallback value
                        import logging
                        logger = logging.getLogger(__name__)
                        # For tasmax/tasmin: fall back to tas (mean temp) if already read
                        if var in ('tasmax', 'tasmin') and 'tas' in result:
                            logger.warning(
                                "Failed reading %s year %d month %d: %s. Using tas as fallback.",
                                var, year, month, exc)
                            data_grid = np.full((ndays, 360, 720), np.nan, dtype=np.float32)
                            # Will be replaced below with tas data
                            land_data_fallback = result['tas'].copy()
                            if var == 'tasmax':
                                land_data_fallback += 2.0  # tasmax ~ tas + 2K
                            else:
                                land_data_fallback -= 2.0  # tasmin ~ tas - 2K
                        else:
                            logger.warning(
                                "Failed reading %s year %d month %d: %s. Filling with NaN.",
                                var, year, month, exc)
                            data_grid = np.full((ndays, 360, 720), np.nan, dtype=np.float32)
                            land_data_fallback = None
                        break
            # Apply fallback for corrupted files, or normal land subsetting
            if land_data_fallback is not None:
                land_data = land_data_fallback
                land_data_fallback = None
            else:
                land_data = data_grid[:, latindex, lonindex]  # (ndays, npoi)

            # Replace fill value with NaN
            land_data[land_data >= fill_value * 0.5] = np.nan
            result[var] = land_data

        result['ndays'] = ndays
        result['year'] = year
        result['month'] = month
        return result

    def read_year(
        self,
        year: int,
        latindex: np.ndarray,
        lonindex: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Read a full year of GSWP3 data for specified land points.

        Concatenates 12 monthly reads. For global grids this may require
        ~1-2 GB of RAM; for regional subsets it's much smaller.

        Returns
        -------
        dict mapping variable → np.ndarray of shape (ndays_in_year, npoi).
        """
        months = {}
        for m in range(12):
            mdata = self.read_month(year, m, latindex, lonindex)
            for var in self.variables:
                months.setdefault(var, []).append(mdata[var])

        result = {var: np.concatenate(months[var], axis=0)
                  for var in self.variables}
        result['year'] = year
        return result

    def get_grid(self) -> dict[str, np.ndarray]:
        """Return the GSWP3 lat/lon coordinate arrays.

        Returns
        -------
        dict with 'lat' (360,) and 'lon' (720,) arrays.
        Lat runs N→S: 89.75, 89.25, ..., -89.75.
        Lon runs W→E: -179.75, -179.25, ..., 179.75.
        """
        # Use the first available variable file via persistent handle
        for var in self.variables:
            for year in (1980, 1990, 2000, 1901):
                path = self._decade_path(var, year)
                if os.path.exists(path):
                    ds = self._get_handle(var, year)
                    lat = np.array(ds.variables['lat'][:])
                    lon = np.array(ds.variables['lon'][:])
                    return {'lat': lat, 'lon': lon}
        raise RuntimeError("Cannot find any GSWP3 file to read grid from.")

    def build_land_indices(
        self,
        land_mask: np.ndarray,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build land point index arrays from a 2D land mask on the GSWP3 grid.

        Parameters
        ----------
        land_mask : np.ndarray, shape (nlat, nlon)
            Boolean or numeric mask; True/nonzero = land.
            Must be aligned with the GSWP3 grid (N→S lat, W→E lon).
        lat : np.ndarray, optional
            GSWP3 latitude array; loaded from files if not given.
        lon : np.ndarray, optional
            GSWP3 longitude array; loaded from files if not given.

        Returns
        -------
        latindex : (npoi,) int array — row indices into GSWP3 lat (N→S)
        lonindex : (npoi,) int array — col indices into GSWP3 lon
        lat_land : (npoi,) float array — latitude of each land point (degrees)
        lon_land : (npoi,) float array — longitude of each land point (degrees)
        """
        if lat is None or lon is None:
            grid = self.get_grid()
            lat = grid['lat']
            lon = grid['lon']

        land_ij = np.argwhere(land_mask > 0)
        latindex = land_ij[:, 0].astype(np.int32)
        lonindex = land_ij[:, 1].astype(np.int32)
        lat_land = lat[latindex]
        lon_land = lon[lonindex]
        return latindex, lonindex, lat_land, lon_land
