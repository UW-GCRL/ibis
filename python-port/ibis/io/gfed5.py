"""GFED5 burned area, the calibration target for docs/FIRE.md stage F6.

GFED5 is monthly, 0.25 degree, 1997-2020, with burned area in km2 per cell and a
companion burnable-area field used as the denominator.

Files: inputs/fire/gfed5/BA/BA{YYYYMM}.nc and inputs/fire/gfed5/BurnableArea.nc
See inputs/fire/README.md for provenance.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import xarray as xr

from ibis.io.netcdf_io import grid2vec

#: GFED5 splits burned area by cause. `Total` is exactly
#: `Norm(summed over 20 land-cover types) + Peat + Defo + Crop` -- verified to
#: better than 1% on a sample month.
#:
#: **`Norm` is the default target, not `Total`.** IBIS represents neither
#: cropland, nor deforestation fire, nor peatland (docs/FIRE.md 4.4e), so
#: calibrating against `Total` would force the natural-vegetation parameters to
#: absorb fire the model structurally cannot produce. For July 2001 those
#: excluded components are ~12% of global burned area -- overwhelmingly
#: cropland -- so the choice is not cosmetic.
DEFAULT_COMPONENT = "Norm"

#: Native GFED5 resolution and the model's, as a whole-number block factor.
GFED5_SHAPE = (720, 1440)


def _burnable_area(gfed5_dir: str) -> np.ndarray:
    """Burnable area per 0.25-degree cell, km2. Note the file's typo'd name."""
    path = os.path.join(gfed5_dir, "BurnableArea.nc")
    with xr.open_dataset(path, decode_times=False) as ds:
        name = "BurableArea" if "BurableArea" in ds else "BurnableArea"
        return np.asarray(ds[name].values, dtype=np.float64)


def _block_sum(a: np.ndarray, factor: int) -> np.ndarray:
    """Conservative aggregation by summing `factor` x `factor` blocks."""
    ny, nx = a.shape
    if ny % factor or nx % factor:
        raise ValueError(f"grid {a.shape} not divisible by {factor}")
    return a.reshape(ny // factor, factor, nx // factor, factor).sum(axis=(1, 3))


def read_gfed5_annual_fraction(
    gfed5_dir: str,
    year: int,
    latindex: np.ndarray,
    lonindex: np.ndarray,
    model_lat: np.ndarray,
    model_lon: np.ndarray,
    component: str = DEFAULT_COMPONENT,
) -> np.ndarray:
    """Annual burned *fraction* on the model grid, as a land-point vector.

    Burned area and burnable area are aggregated conservatively (summed over
    2x2 blocks of 0.25-degree cells) and only then divided, so the result is a
    true area-weighted fraction rather than a mean of fractions.

    `model_lat` / `model_lon` are required for the same reason as in
    `io.popdens`: land-point indices follow the *surta* file's ordering, while
    GFED5 stores latitude south-to-north and longitude on -180..180. Aligning on
    coordinate values removes the assumption.

    Returns
    -------
    np.ndarray
        `(npoi,)` burned fraction in [0, 1] for the calendar year.
    """
    files = sorted(glob.glob(os.path.join(gfed5_dir, "BA", f"BA{year}??.nc")))
    if len(files) != 12:
        raise FileNotFoundError(
            f"expected 12 monthly files for {year}, found {len(files)} in "
            f"{gfed5_dir}/BA"
        )

    ba = np.zeros(GFED5_SHAPE, dtype=np.float64)
    for f in files:
        with xr.open_dataset(f, decode_times=False) as ds:
            if component not in ds:
                raise KeyError(f"{component!r} not in {os.path.basename(f)}; "
                               f"available: {list(ds.data_vars)}")
            v = ds[component].values
            # `Norm` carries a land-cover-type axis; sum it out.
            arr = np.asarray(v, dtype=np.float64)
            while arr.ndim > 2:
                arr = np.nansum(arr, axis=0)
            ba += np.nan_to_num(arr)

    burnable = _burnable_area(gfed5_dir)

    # Conservative 0.25 -> 0.5 degree aggregation, then divide.
    factor = GFED5_SHAPE[0] // len(model_lat)
    if factor < 1 or GFED5_SHAPE[0] % len(model_lat):
        raise ValueError(
            f"model grid {len(model_lat)}x{len(model_lon)} is not an integer "
            f"coarsening of GFED5's {GFED5_SHAPE}"
        )
    ba_c = _block_sum(ba, factor)
    burn_c = _block_sum(np.nan_to_num(burnable), factor)
    frac = np.where(burn_c > 0.0, ba_c / np.maximum(burn_c, 1e-12), 0.0)
    frac = np.clip(frac, 0.0, 1.0)

    # GFED5 is south-to-north; flip if the model runs the other way.
    with xr.open_dataset(files[0], decode_times=False) as ds:
        file_lat = np.asarray(ds["lat"].values, dtype=np.float64)
    file_lat_c = file_lat.reshape(-1, factor).mean(axis=1)
    if (file_lat_c[-1] - file_lat_c[0]) * (model_lat[-1] - model_lat[0]) < 0:
        frac = frac[::-1, :]

    if latindex.max() >= frac.shape[0] or lonindex.max() >= frac.shape[1]:
        raise ValueError(f"land-point indices exceed the aggregated grid "
                         f"{frac.shape}")
    return grid2vec(frac, latindex, lonindex)


def global_burned_area_mha(gfed5_dir: str, year: int,
                           component: str = "Total") -> float:
    """Global annual burned area in Mha -- a scalar sanity check.

    GFED5's published global mean is ~774 Mha/yr with a declining trend.
    """
    files = sorted(glob.glob(os.path.join(gfed5_dir, "BA", f"BA{year}??.nc")))
    total = 0.0
    for f in files:
        with xr.open_dataset(f, decode_times=False) as ds:
            total += float(np.nansum(ds[component].values))
    return total / 1e4          # km2 -> Mha
