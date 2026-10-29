"""WAD2M inundated-fraction reader -- `f_inund` for the methane scheme.

WAD2M (Wetland Area Dynamics for Methane Modeling, Zhang et al. 2021) gives a
monthly wetland fraction `Fw` on a 0.5 degree grid, 2000-2020.

**Why extent is prescribed rather than predicted** (docs/METHANE.md 4.1):
wetland extent is the dominant term in the global methane budget's
uncertainty. Prescribing it isolates our biogeochemistry so its skill is
measurable -- "given observed extent, our fluxes are X" is a cleaner claim than
co-tuning extent and flux against a single target.

**Grid alignment.** WAD2M labels cell *centres* (-89.75 .. 89.75); the IBIS
land mask labels the cell's lower-left corner (-90.0 .. 89.5). Both are 360x720
and both run south-to-north, so cell *i* is cell *i* -- the 0.25 degree
difference is a labelling convention, not an offset to correct. That is
asserted here rather than assumed, and was checked against geography before
being relied on: Hudson Bay Lowlands 0.61, Congo Cuvette 0.30, West Siberian
Lowland 0.26, Amazon 0.18, US Great Plains 0.004, Sahara masked. Getting this
backwards would put Siberia's wetlands in the Sahara, which is the exact class
of silent misalignment the fire audits kept turning up (docs/FIRE.md 4.4).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

#: Native shape, (lat, lon). A file that is not this is not WAD2M 0.5 degree.
WAD2M_SHAPE = (360, 720)

#: First and last year in Version 2.0.
WAD2M_YEARS = (2000, 2020)


def wad2m_in_record(year: int) -> bool:
    """True when the calendar year has an observed WAD2M field (WAD2M_YEARS, inclusive)."""
    y0, y1 = WAD2M_YEARS
    return y0 <= int(year) <= y1


def open_wad2m(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


def read_wad2m_month(
    path: str,
    year: int,
    month: int,
    latindex: np.ndarray,
    lonindex: np.ndarray,
    model_lat: np.ndarray,
    model_lon: np.ndarray,
    clamp_to_range: bool = True,
) -> np.ndarray:
    """Wetland fraction for one month as a land-point vector, in [0, 1].

    Parameters
    ----------
    year, month : int
        `month` is 1-based. Outside 2000-2020 the nearest available year is
        used when `clamp_to_range`, otherwise a ValueError is raised -- a
        spin-up cycling 1901-2000 has no satellite record to read, and silently
        returning zeros would make every pre-2000 wetland vanish.
    latindex, lonindex : np.ndarray
        Land-point indices into the model grid.
    model_lat, model_lon : np.ndarray
        **Required.** The model grid's coordinates, used to verify alignment
        rather than trusting index order.

    Returns
    -------
    (npoi,) float64 in [0, 1]. Ocean is masked in WAD2M and returns 0.0.
    """
    if not (1 <= month <= 12):
        raise ValueError(f"month must be 1-12, got {month}")
    y0, y1 = WAD2M_YEARS
    if not (y0 <= year <= y1):
        if not clamp_to_range:
            raise ValueError(
                f"WAD2M covers {y0}-{y1}; got {year}. Pass clamp_to_range=True "
                f"to use the nearest year, or prescribe a climatology.")
        year = int(np.clip(year, y0, y1))

    with open_wad2m(path) as ds:
        _check_grid(ds, model_lat, model_lon)
        idx = (year - y0) * 12 + (month - 1)
        fw = np.asarray(ds["Fw"].isel(time=idx).values, dtype=np.float64)

    fw = np.nan_to_num(fw, nan=0.0)      # WAD2M masks ocean; no wetland there
    fw = np.clip(fw, 0.0, 1.0)
    if latindex.max() >= fw.shape[0] or lonindex.max() >= fw.shape[1]:
        raise ValueError(
            f"land-point indices exceed the WAD2M grid {fw.shape}")
    return fw[latindex, lonindex]


def read_wad2m_climatology(
    path: str,
    latindex: np.ndarray,
    lonindex: np.ndarray,
    model_lat: np.ndarray,
    model_lon: np.ndarray,
    years: tuple[int, int] = WAD2M_YEARS,
) -> np.ndarray:
    """Mean seasonal cycle as (12, npoi) -- what a pre-2000 spin-up should use.

    Cycling forcing years 1901-2000 against a 2000-2020 satellite record means
    every year but the last has no observation. A climatology states that
    assumption instead of hiding it behind a clamp.
    """
    y0, y1 = years
    with open_wad2m(path) as ds:
        _check_grid(ds, model_lat, model_lon)
        fw = np.asarray(ds["Fw"].values, dtype=np.float64)
    i0 = (y0 - WAD2M_YEARS[0]) * 12
    i1 = (y1 - WAD2M_YEARS[0] + 1) * 12
    fw = np.nan_to_num(fw[i0:i1], nan=0.0)
    nyr = (i1 - i0) // 12
    clim = fw.reshape(nyr, 12, *fw.shape[1:]).mean(axis=0)
    clim = np.clip(clim, 0.0, 1.0)
    return clim[:, latindex, lonindex]



def read_wad2m_bounds(path: str, latindex: np.ndarray, lonindex: np.ndarray, model_lat: np.ndarray, model_lon: np.ndarray,
                      years: tuple[int, int] = WAD2M_YEARS) -> tuple[np.ndarray, np.ndarray]:
    """Each land point's minimum and maximum wetland fraction over the record, (npoi,) each -- the bounds of the
    dynamic wetland area (ibis.bgc.wetland_area; docs/METHANE.md 7). Fill (frozen / no data) counts as 0, as in the
    climatology, so a cell that freezes has a minimum of 0."""
    y0, y1 = years
    with open_wad2m(path) as ds:
        _check_grid(ds, model_lat, model_lon)
        fw = np.asarray(ds["Fw"].values, dtype=np.float64)
    i0 = (y0 - WAD2M_YEARS[0]) * 12; i1 = (y1 - WAD2M_YEARS[0] + 1) * 12
    fw = np.clip(np.nan_to_num(fw[i0:i1], nan=0.0), 0.0, 1.0)
    return fw.min(axis=0)[latindex, lonindex], fw.max(axis=0)[latindex, lonindex]

def _check_grid(ds, model_lat, model_lon) -> None:
    """Alignment is asserted, not assumed. See the module docstring."""
    flat = np.asarray(ds["lat"].values, dtype=np.float64)
    flon = np.asarray(ds["lon"].values, dtype=np.float64)
    if (flat.shape[0], flon.shape[0]) != WAD2M_SHAPE:
        raise ValueError(f"expected WAD2M {WAD2M_SHAPE}, got "
                         f"{(flat.shape[0], flon.shape[0])}")
    if len(model_lat) != flat.shape[0] or len(model_lon) != flon.shape[0]:
        raise ValueError(
            f"model grid {len(model_lat)}x{len(model_lon)} does not match "
            f"WAD2M {WAD2M_SHAPE}; regridding is not implemented")
    # Same direction, and offset by exactly half a cell (centre vs corner).
    if (flat[-1] - flat[0]) * (model_lat[-1] - model_lat[0]) < 0:
        raise ValueError(
            "WAD2M runs south-to-north but the model grid runs the other way; "
            "flipping is not implemented because no such grid has been seen")
    d = flat - np.asarray(model_lat, dtype=np.float64)
    if not np.allclose(d, d[0], atol=1e-6) or abs(abs(d[0]) - 0.25) > 1e-6:
        raise ValueError(
            f"expected a uniform half-cell offset between WAD2M centres and "
            f"the model grid, got offsets {d.min():.4f}..{d.max():.4f}")


def global_wetland_area_mkm2(path: str, year: int, month: int) -> float:
    """Global wetland area for one month, million km2 -- a scalar check.

    Zhang et al. (2021) report a long-term mean maximum monthly extent of
    roughly 12-13 million km2, with the annual mean well below that.
    """
    with open_wad2m(path) as ds:
        fw = np.nan_to_num(
            np.asarray(ds["Fw"].isel(
                time=(year - WAD2M_YEARS[0]) * 12 + month - 1).values), nan=0.0)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    cell = (111.32 * 0.5) ** 2 * np.cos(np.deg2rad(lat))[:, None]
    return float((fw * cell).sum() / 1e6)
