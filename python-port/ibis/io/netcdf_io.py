"""NetCDF I/O for IBIS-JAX.

Reads input climatology, soil texture, and land mask from NetCDF files
(matching the Fortran IBIS input format). Writes output in xarray format.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Float, Array


def read_input_data(
    input_dir: str,
    snorth: float, ssouth: float,
    swest: float, seast: float,
    xres: float = 0.5, yres: float = 0.5,
) -> dict:
    """Read all input data from NetCDF files.

    Reads the standard IBIS input files (surta.nc, soita.sand.nc, etc.)
    and extracts the subdomain specified by lat/lon bounds.

    Parameters
    ----------
    input_dir : str
        Directory containing input NetCDF files.
    snorth, ssouth, swest, seast : float
        Domain bounds (degrees).
    xres, yres : float
        Grid resolution (degrees).

    Returns
    -------
    dict with keys:
        'lonindex', 'latindex' : 1D arrays mapping land points to grid
        'npoi' : int, number of land points
        'nlon', 'nlat' : int, grid dimensions
        'lon', 'lat' : 1D coordinate arrays
        'sand', 'clay' : (npoi, nsoilay) soil texture
        'vegtype' : (npoi,) vegetation type
        'deltat' : (npoi,) temperature correction
        'climate' : dict of monthly climatology arrays (npoi, 12)
    """
    result = {}

    # --- Read land mask ---
    surta_path = os.path.join(input_dir, "surta.nc")
    ds_surta = xr.open_dataset(surta_path, decode_times=False)

    # Get coordinate names (may vary between files)
    lon_name = _find_coord(ds_surta, "lon")
    lat_name = _find_coord(ds_surta, "lat")

    lon = ds_surta[lon_name].values
    lat = ds_surta[lat_name].values

    # Subset to domain
    lon_mask = (lon >= swest) & (lon <= seast)
    lat_mask = (lat >= ssouth) & (lat <= snorth)

    lon_sub = lon[lon_mask]
    lat_sub = lat[lat_mask]

    # Get land mask variable
    var_name = _find_var(ds_surta, ["surta", "landfrac", "mask"])
    landmask = ds_surta[var_name].values

    # Squeeze to 2D (lat, lon)
    while landmask.ndim > 2:
        landmask = landmask[0]

    landmask_sub = landmask[np.ix_(np.where(lat_mask)[0],
                                    np.where(lon_mask)[0])]

    # Land points where mask > 0
    land_ij = np.argwhere(landmask_sub > 0)
    npoi = len(land_ij)
    latindex = land_ij[:, 0]
    lonindex = land_ij[:, 1]

    result["npoi"] = npoi
    result["nlon"] = len(lon_sub)
    result["nlat"] = len(lat_sub)
    result["lon"] = lon_sub
    result["lat"] = lat_sub
    result["lonindex"] = lonindex
    result["latindex"] = latindex

    ds_surta.close()

    # --- Read soil texture ---
    # Prefer a single combined file (soita.nc with sandpct + claypct variables);
    # fall back to separate soita.sand.nc / soita.clay.nc for backwards compatibility.
    soita_combined = os.path.join(input_dir, "soita.nc")
    if os.path.exists(soita_combined):
        sand = _read_soil_layers(input_dir, "soita.nc",
                                 ["sandpct", "sand"], latindex, lonindex,
                                 lat_mask, lon_mask)
        clay = _read_soil_layers(input_dir, "soita.nc",
                                 ["claypct", "clay"], latindex, lonindex,
                                 lat_mask, lon_mask)
    else:
        sand = _read_soil_layers(input_dir, "soita.sand.nc",
                                 ["sandpct", "sand"], latindex, lonindex,
                                 lat_mask, lon_mask)
        clay = _read_soil_layers(input_dir, "soita.clay.nc",
                                 ["claypct", "clay"], latindex, lonindex,
                                 lat_mask, lon_mask)
    result["sand"] = jnp.array(sand)
    result["clay"] = jnp.array(clay)

    # --- Read vegetation type ---
    vegtype = _read_2d_field(input_dir, "vegtype.nc",
                             ["vegtype", "veg"], latindex, lonindex,
                             lat_mask, lon_mask)
    result["vegtype"] = jnp.array(vegtype)

    # --- Read delta-T ---
    deltat = _read_2d_field(input_dir, "deltat.nc",
                            ["deltat"], latindex, lonindex,
                            lat_mask, lon_mask)
    result["deltat"] = jnp.array(deltat)

    # --- Read monthly climatology ---
    climate = {}
    clim_files = {
        "temp": ("temp.mon.nc", ["temp"]),
        "trange": ("trange.mon.nc", ["trange"]),
        "prec": ("prec.mon.nc", ["prec"]),
        "cld": ("cld.mon.nc", ["cld"]),
        "rh": ("rh.mon.nc", ["rh"]),
        "wetd": ("wetd.mon.nc", ["wetd"]),
        "wspd": ("wspd.mon.nc", ["wspd"]),
    }
    for key, (filename, varnames) in clim_files.items():
        data = _read_monthly_field(input_dir, filename, varnames,
                                   latindex, lonindex, lat_mask, lon_mask)
        climate[key] = jnp.array(data)

    result["climate"] = climate

    return result


def vec2grid(
    vec: Float[Array, "npoi"],
    latindex: np.ndarray,
    lonindex: np.ndarray,
    nlat: int, nlon: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Convert land-point vector to 2D grid array.

    Parameters
    ----------
    vec : (npoi,) or (npoi, nlevels)
        Data on land points.
    latindex, lonindex : (npoi,)
        Grid indices for each land point.
    nlat, nlon : int
        Grid dimensions.
    fill_value : float
        Value for ocean/non-land cells.

    Returns
    -------
    np.ndarray of shape (nlat, nlon) or (nlat, nlon, nlevels)
    """
    vec_np = np.asarray(vec)
    if vec_np.ndim == 1:
        grid = np.full((nlat, nlon), fill_value)
        grid[latindex, lonindex] = vec_np
    else:
        nlevels = vec_np.shape[1]
        grid = np.full((nlat, nlon, nlevels), fill_value)
        grid[latindex, lonindex, :] = vec_np
    return grid


def grid2vec(
    grid: np.ndarray,
    latindex: np.ndarray,
    lonindex: np.ndarray,
) -> np.ndarray:
    """Convert 2D grid array to land-point vector.

    Parameters
    ----------
    grid : (nlat, nlon) or (nlat, nlon, nlevels)
        Gridded data.
    latindex, lonindex : (npoi,)
        Grid indices for land points.

    Returns
    -------
    np.ndarray of shape (npoi,) or (npoi, nlevels)
    """
    if grid.ndim == 2:
        return grid[latindex, lonindex]
    else:
        return grid[latindex, lonindex, :]


def state_to_dataset(
    state,
    lon: np.ndarray,
    lat: np.ndarray,
    latindex: np.ndarray,
    lonindex: np.ndarray,
    year: int = 0,
    month: int = 0,
) -> xr.Dataset:
    """Convert IBISState fields to an xarray Dataset on the grid.

    Extracts key diagnostic variables and maps them from land-point
    vectors to the 2D grid.

    Parameters
    ----------
    state : IBISState
        Model state to convert.
    lon, lat : 1D arrays
        Coordinate arrays.
    latindex, lonindex : 1D arrays
        Land point indices.
    year, month : int
        Time indices for labeling.

    Returns
    -------
    xr.Dataset
    """
    nlat = len(lat)
    nlon = len(lon)

    ds = xr.Dataset(coords={
        "lat": ("lat", np.asarray(lat)),
        "lon": ("lon", np.asarray(lon)),
    })

    # Helper to add a variable
    def _add_var(name, vec, units="", long_name=""):
        grid = vec2grid(vec, latindex, lonindex, nlat, nlon)
        ds[name] = xr.DataArray(grid, dims=["lat", "lon"],
                                attrs={"units": units, "long_name": long_name})

    # Canopy
    _add_var("tu", state.canopy.tu, "K", "Upper canopy temperature")
    _add_var("tl", state.canopy.tl, "K", "Lower canopy temperature")
    _add_var("solu", state.canopy.solu, "W/m2", "Solar absorbed upper")
    _add_var("solg", state.canopy.solg, "W/m2", "Solar absorbed soil")

    # Soil
    _add_var("tg", state.soil.tg, "K", "Soil surface temperature")
    for k in range(state.soil.tsoi.shape[1]):
        _add_var(f"tsoi_{k}", state.soil.tsoi[:, k], "K",
                 f"Soil temperature layer {k}")
        _add_var(f"wsoi_{k}", state.soil.wsoi[:, k], "fraction",
                 f"Soil moisture layer {k}")

    # Snow
    _add_var("fi", state.snow.fi, "fraction", "Snow cover fraction")

    # Vegetation
    _add_var("fu", state.vegetation.fu, "fraction", "Upper canopy fraction")
    _add_var("fl", state.vegetation.fl, "fraction", "Lower canopy fraction")
    _add_var("laiu", state.vegetation.lai[:, 1], "m2/m2", "Upper canopy LAI")
    _add_var("lail", state.vegetation.lai[:, 0], "m2/m2", "Lower canopy LAI")
    # Above-ground biomass (sum of wood carbon across PFTs)
    import jax.numpy as jnp
    cveg = jnp.sum(state.vegetation.cbiow, axis=1)  # total wood C (kgC/m2)
    _add_var("cveg", cveg, "kgC/m2", "Above-ground biomass C")

    # Carbon — compute diagnostics from component pools to avoid stale values
    c = state.carbon
    totcsoi_computed = c.csoipas + c.csoislon + c.csoislop + c.totcmic
    totlit_computed = (c.clitlm + c.clitls + c.clitll +
                       c.clitrm + c.clitrs + c.clitrl +
                       c.clitwm + c.clitws + c.clitwl)
    _add_var("totcsoi", totcsoi_computed, "kgC/m2", "Total soil C")
    _add_var("totlit", totlit_computed, "kgC/m2", "Total litter C")
    _add_var("csoipas", c.csoipas, "kgC/m2", "Passive SOM")
    _add_var("csoislon", c.csoislon, "kgC/m2", "Slow non-protected SOM")
    _add_var("csoislop", c.csoislop, "kgC/m2", "Slow protected SOM")
    _add_var("totcmic", c.totcmic, "kgC/m2", "Microbial biomass C")

    ds.attrs["year"] = year
    ds.attrs["month"] = month

    return ds


# ---- Internal helpers ----

def _find_coord(ds: xr.Dataset, name: str) -> str:
    """Find a coordinate name matching 'lon' or 'lat'."""
    candidates = [name, name.upper(), f"{name}itude",
                  f"{name}itude".capitalize()]
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    # Fall back to first matching
    for c in ds.coords:
        if name[:3].lower() in c.lower():
            return c
    raise KeyError(f"Cannot find '{name}' coordinate in dataset")


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str:
    """Find the first matching variable name."""
    for c in candidates:
        if c in ds:
            return c
    # Try case-insensitive
    lower_map = {k.lower(): k for k in ds.data_vars}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Cannot find any of {candidates} in dataset")


def _read_2d_field(input_dir, filename, varnames,
                   latindex, lonindex, lat_mask, lon_mask):
    """Read a 2D (lat, lon) field and extract land points."""
    path = os.path.join(input_dir, filename)
    if not os.path.exists(path):
        return np.zeros(len(latindex))
    ds = xr.open_dataset(path, decode_times=False)
    var = _find_var(ds, varnames)
    data = ds[var].values
    while data.ndim > 2:
        data = data[0]
    data_sub = data[np.ix_(np.where(lat_mask)[0], np.where(lon_mask)[0])]
    ds.close()
    return data_sub[latindex, lonindex].astype(np.float32)


def _read_soil_layers(input_dir, filename, varnames,
                      latindex, lonindex, lat_mask, lon_mask):
    """Read a 3D (lat, lon, layer) soil field."""
    path = os.path.join(input_dir, filename)
    if not os.path.exists(path):
        return np.zeros((len(latindex), 6))
    ds = xr.open_dataset(path, decode_times=False)
    var = _find_var(ds, varnames)
    data = ds[var].values  # (nlayer, lat, lon) or (lat, lon, nlayer)
    if data.ndim == 4:
        data = data[0]  # Remove time dim
    # Determine layer axis
    if data.shape[0] < data.shape[1]:
        # (nlayer, lat, lon) -> (lat, lon, nlayer)
        data = np.moveaxis(data, 0, -1)
    data_sub = data[np.ix_(np.where(lat_mask)[0], np.where(lon_mask)[0])]
    ds.close()
    result = data_sub[latindex, lonindex, :]
    return result.astype(np.float32)


def _read_monthly_field(input_dir, filename, varnames,
                        latindex, lonindex, lat_mask, lon_mask):
    """Read a monthly (lat, lon, 12) climatology field."""
    path = os.path.join(input_dir, filename)
    if not os.path.exists(path):
        return np.zeros((len(latindex), 12))
    ds = xr.open_dataset(path, decode_times=False)
    var = _find_var(ds, varnames)
    data = ds[var].values  # (12, lat, lon) typically
    # Squeeze singleton dims (e.g. level=1)
    data = data.squeeze()
    if data.ndim == 2:
        # Static field, replicate for 12 months
        data_sub = data[np.ix_(np.where(lat_mask)[0], np.where(lon_mask)[0])]
        result = np.stack([data_sub[latindex, lonindex]] * 12, axis=1)
    else:
        # (month, lat, lon) -> extract and transpose
        if data.shape[0] == 12 or data.shape[0] <= 12:
            nmon = data.shape[0]
            result = np.zeros((len(latindex), nmon))
            for m in range(nmon):
                d = data[m]
                d_sub = d[np.ix_(np.where(lat_mask)[0], np.where(lon_mask)[0])]
                result[:, m] = d_sub[latindex, lonindex]
        else:
            data = np.moveaxis(data, -1, 0)  # Try other axis
            nmon = data.shape[0]
            result = np.zeros((len(latindex), min(nmon, 12)))
            for m in range(min(nmon, 12)):
                d = data[m]
                d_sub = d[np.ix_(np.where(lat_mask)[0], np.where(lon_mask)[0])]
                result[:, m] = d_sub[latindex, lonindex]
    ds.close()
    return result.astype(np.float32)
