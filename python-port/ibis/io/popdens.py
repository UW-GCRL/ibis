"""Human population density input for the fire module (docs/FIRE.md stage F3).

Reads the CLM population-density stream, which is what CLM's own Li et al.
fire scheme consumes. Chosen over raw HYDE because it is already on IBIS's
720x360 0.5-degree grid, so no regridding is involved, and because it makes our
human term directly comparable to CLM's.

File: inputs/fire/popdens/clmforc.Li_2018_SSP2_CMIP6_hdm_0.5x0.5_AVHRR_*.nc
      hdm(time, lat, lon) in ind/km2, annual, 1850-2100.
See inputs/fire/README.md for provenance and the retention rationale.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ibis.io.netcdf_io import grid2vec

#: Variable name in the CLM stream file.
HDM_VAR = "hdm"


def open_popdens(path: str) -> xr.Dataset:
    """Open the population stream without decoding its non-standard time axis."""
    return xr.open_dataset(path, decode_times=False)


def available_years(path: str) -> np.ndarray:
    """Years present in the file, as integers."""
    with open_popdens(path) as ds:
        return np.asarray(ds["year"].values).astype(int)


def read_popdens_year(
    path: str,
    year: int,
    latindex: np.ndarray,
    lonindex: np.ndarray,
    model_lat: np.ndarray,
    model_lon: np.ndarray,
    clamp_to_range: bool = True,
) -> np.ndarray:
    """Population density for one year as a land-point vector.

    `model_lat` / `model_lon` are **required**, not optional. The land-point
    indices (`latindex`, `lonindex`) are derived from whatever ordering the
    *surta* land-mask file happens to use (`netcdf_io.read_input_data`), while
    this stream stores latitude south-to-north. Assuming the two agree by index
    would silently put Siberia's population in the Sahara -- exactly the class
    of silent misalignment the docs/FIRE.md audits kept turning up. Aligning on
    coordinate *values* removes the assumption.

    Parameters
    ----------
    path : str
        Path to the CLM `hdm` stream file.
    year : int
        Calendar year. The file is annual.
    latindex, lonindex : np.ndarray
        Land-point grid indices into the model grid.
    model_lat, model_lon : np.ndarray
        The model grid's latitude and longitude coordinates, as returned by
        `netcdf_io.read_input_data` (`result["lat"]`, `result["lon"]`).
    clamp_to_range : bool
        Clamp out-of-range years to the first/last available rather than
        raising. Spin-up runs commonly start before 1850.

    Returns
    -------
    np.ndarray
        `(npoi,)` population density in **ind/km2**, float64, non-negative.
    """
    with open_popdens(path) as ds:
        years = np.asarray(ds["year"].values).astype(int)

        if year < years.min() or year > years.max():
            if not clamp_to_range:
                raise ValueError(
                    f"year {year} outside {years.min()}-{years.max()}"
                )
            year = int(np.clip(year, years.min(), years.max()))

        idx = int(np.argmin(np.abs(years - year)))
        da = ds[HDM_VAR].isel(time=idx)

        # Align on coordinate values, not index order. `nearest` also copes
        # with a longitude convention mismatch of the -180/180 vs 0/360 kind
        # only if the caller has already normalised; the check below catches
        # the case where it has not.
        lon = np.asarray(model_lon, dtype=np.float64)
        file_lon = np.asarray(da["lon"].values, dtype=np.float64)
        if lon.min() < file_lon.min() - 1.0 or lon.max() > file_lon.max() + 1.0:
            raise ValueError(
                f"model longitudes [{lon.min():.1f}, {lon.max():.1f}] fall "
                f"outside the file's [{file_lon.min():.1f}, "
                f"{file_lon.max():.1f}]; normalise the convention first"
            )

        aligned = da.interp(
            lat=np.asarray(model_lat, dtype=np.float64),
            lon=lon,
            method="nearest",
            kwargs={"fill_value": None},
        )
        grid = np.asarray(aligned.values, dtype=np.float64)

    if latindex.max() >= grid.shape[0] or lonindex.max() >= grid.shape[1]:
        raise ValueError(
            f"land-point indices exceed the aligned grid {grid.shape}"
        )

    vec = grid2vec(grid, latindex, lonindex)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(vec, 0.0)
