"""Diagnostic output writer for IBIS-JAX.

Writes daily, monthly, and yearly output as NetCDF files via xarray.
Maps land-point vectors to the 2D lat/lon grid for output.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import jax.numpy as jnp
import xarray as xr

from ibis.io.netcdf_io import vec2grid


class DiagnosticWriter:
    """Writes IBIS state diagnostics to NetCDF files.

    Parameters
    ----------
    output_dir : str
        Root directory for output files.
    lon, lat : 1D arrays
        Coordinate arrays.
    latindex, lonindex : 1D int arrays
        Land point to grid mapping.
    prefix : str
        File name prefix.
    """

    def __init__(
        self,
        output_dir: str,
        lon: np.ndarray,
        lat: np.ndarray,
        latindex: np.ndarray,
        lonindex: np.ndarray,
        prefix: str = "ibis",
    ):
        self.output_dir = output_dir
        self.lon = np.asarray(lon)
        self.lat = np.asarray(lat)
        self.latindex = np.asarray(latindex)
        self.lonindex = np.asarray(lonindex)
        self.nlat = len(self.lat)
        self.nlon = len(self.lon)
        self.prefix = prefix
        os.makedirs(output_dir, exist_ok=True)

    def _to_grid(self, vec, fill_value=np.nan):
        """Convert a land-point vector to 2D grid."""
        return vec2grid(vec, self.latindex, self.lonindex,
                        self.nlat, self.nlon, fill_value)

    def _make_dataset(self) -> xr.Dataset:
        """Create a dataset with lon/lat coordinates."""
        return xr.Dataset(coords={
            "lat": ("lat", self.lat),
            "lon": ("lon", self.lon),
        })

    def _add_2d(self, ds, name, vec, units="", long_name=""):
        """Add a 2D (lat, lon) variable from a land-point vector."""
        grid = self._to_grid(vec)
        ds[name] = xr.DataArray(grid, dims=["lat", "lon"],
                                attrs={"units": units, "long_name": long_name})

    def _add_3d(self, ds, name, vec_2d, dim3_name, dim3_coords, units="", long_name=""):
        """Add a 3D (lat, lon, layer) variable."""
        vec_np = np.asarray(vec_2d)
        nlevels = vec_np.shape[1]
        grid = np.full((self.nlat, self.nlon, nlevels), np.nan)
        grid[self.latindex, self.lonindex, :] = vec_np
        ds[name] = xr.DataArray(
            grid, dims=["lat", "lon", dim3_name],
            coords={dim3_name: dim3_coords[:nlevels]},
            attrs={"units": units, "long_name": long_name},
        )

    def write_daily(self, state, year: int, jday: int) -> str:
        """Write daily diagnostic output.

        Parameters
        ----------
        state : IBISState
        year, jday : int
            Year and Julian day.

        Returns
        -------
        str : output file path
        """
        ds = self._make_dataset()
        ds.attrs["year"] = year
        ds.attrs["jday"] = jday

        # Surface energy balance
        self._add_2d(ds, "tg", state.soil.tg, "K", "Soil surface temperature")
        self._add_2d(ds, "tu", state.canopy.tu, "K", "Upper canopy temperature")
        self._add_2d(ds, "tl", state.canopy.tl, "K", "Lower canopy temperature")
        self._add_2d(ds, "t12", state.canopy.t12, "K", "Canopy air temperature")

        # Snow
        self._add_2d(ds, "fi", state.snow.fi, "fraction", "Snow cover fraction")

        # Daily accumulators
        self._add_2d(ds, "adrain", state.summary.adrain, "mm/day",
                     "Daily rainfall")
        self._add_2d(ds, "adsnow", state.summary.adsnow, "mm/day",
                     "Daily snowfall")
        self._add_2d(ds, "adaet", state.summary.adaet, "mm/day", "Daily AET")

        path = os.path.join(
            self.output_dir,
            f"{self.prefix}_daily_{year}_{jday:03d}.nc"
        )
        ds.to_netcdf(path)
        ds.close()
        return path

    def write_monthly(self, state, year: int, month: int,
                      monthly_fluxes: dict | None = None) -> str:
        """Write monthly diagnostic output.

        Parameters
        ----------
        state : IBISState
            End-of-month model state.
        year, month : int
            Calendar year and 1-based month index.
        monthly_fluxes : dict, optional
            Pre-computed monthly totals, keyed by variable name.
            Expected keys: amprcp (mm), amaet (mm), amnpptot (kgC/m2),
            amneetot (kgC/m2).  Computed in run_global.py as the difference
            between end-of-month and start-of-month annual accumulators.

        Returns
        -------
        str : output file path
        """
        ds = self._make_dataset()
        ds.attrs["year"] = year
        ds.attrs["month"] = month

        # State variables (end-of-month snapshot)
        self._add_2d(ds, "tg", state.soil.tg, "K", "Soil surface temperature")
        if getattr(state, "wetland", None) is not None:
            self._add_2d(ds, "amch4", state.wetland.flux, "kgC/m2/d",
                         "Monthly-mean wetland CH4 emission rate (carbon mass, grid-cell mean)")

        layer_coords = np.arange(state.soil.tsoi.shape[1])
        self._add_3d(ds, "tsoi", state.soil.tsoi, "soilay", layer_coords,
                     "K", "Soil temperature")
        self._add_3d(ds, "wsoi", state.soil.wsoi, "soilay", layer_coords,
                     "fraction", "Soil liquid moisture")

        self._add_2d(ds, "fu", state.vegetation.fu, "fraction",
                     "Upper canopy cover")
        self._add_2d(ds, "fl", state.vegetation.fl, "fraction",
                     "Lower canopy cover")

        if state.vegetation.lai.ndim == 2:
            self._add_2d(ds, "laiu", state.vegetation.lai[:, 1],
                         "m2/m2", "Upper canopy LAI")
            self._add_2d(ds, "lail", state.vegetation.lai[:, 0],
                         "m2/m2", "Lower canopy LAI")

        self._add_2d(ds, "totcsoi", state.carbon.totcsoi, "kgC/m2",
                     "Total soil carbon")
        self._add_2d(ds, "totlit", state.carbon.totlit, "kgC/m2",
                     "Total litter carbon")

        # Monthly flux totals (accumulator differences from run_global.py)
        if monthly_fluxes:
            for name, (arr, units, long_name) in monthly_fluxes.items():
                self._add_2d(ds, name, arr, units, long_name)

        path = os.path.join(
            self.output_dir,
            f"{self.prefix}_monthly_{year}_{month:02d}.nc"
        )
        ds.to_netcdf(path)
        ds.close()
        return path

    def write_yearly(self, state, year: int, extra: dict | None = None) -> str:
        """Write yearly diagnostic output.

        Parameters
        ----------
        state : IBISState
        year : int

        Returns
        -------
        str : output file path
        """
        ds = self._make_dataset()
        ds.attrs["year"] = year

        # Annual accumulators
        self._add_2d(ds, "aynpptot", state.summary.aynpptot, "kgC/m2/yr",
                     "Annual total NPP")
        self._add_2d(ds, "ayanpptot", state.summary.ayanpptot, "kgC/m2/yr",
                     "Annual above-ground NPP")
        self._add_2d(ds, "ayneetot", state.summary.ayneetot, "kgC/m2/yr",
                     "Annual NEE")
        self._add_2d(ds, "ayprcp", state.summary.ayprcp, "mm/yr",
                     "Annual precipitation")
        self._add_2d(ds, "ayaet", state.summary.ayaet, "mm/yr", "Annual AET")
        if getattr(state, "etparts", None) is not None:
            self._add_2d(ds, "aytrans", state.etparts.trans, "mm/yr", "Annual transpiration")
            self._add_2d(ds, "ayinterc", state.etparts.interc, "mm/yr", "Annual interception evaporation")
            self._add_2d(ds, "aysoilev", state.etparts.soilev, "mm/yr", "Annual soil evaporation")
            self._add_2d(ds, "aysnowsub", state.etparts.snowsub, "mm/yr", "Annual snow sublimation")
        self._add_2d(ds, "aygpptot", state.summary.aygpptot, "kgC/m2/yr",
                     "Annual GPP")
        self._add_2d(ds, "ayco2mic", state.summary.ayco2mic, "kgC/m2/yr",
                     "Annual heterotrophic respiration")
        self._add_2d(ds, "aytrunof", state.summary.aytrunof, "mm/yr",
                     "Annual surface runoff")
        self._add_2d(ds, "aytdrain", state.summary.aytdrain, "mm/yr",
                     "Annual drainage")
        for _name, (_vec, _unit, _long) in (extra or {}).items():               # per-run extras (cropland harvest and clearing, CALIBRATION.md 41)
            self._add_2d(ds, _name, _vec, _unit, _long)
        if getattr(state, "wetland", None) is not None:                      # wetland CH4 (docs/METHANE.md; CALIBRATION.md 38)
            self._add_2d(ds, "aych4", state.wetland.annual, "kgC/m2/yr",
                         "Annual wetland CH4 emission (carbon mass, grid-cell mean over the inundated fraction)")
            self._add_2d(ds, "zwt", state.wetland.zwt, "m", "Wetland water table depth of the last month")

        # Vegetation structure
        self._add_2d(ds, "fu", state.vegetation.fu, "fraction",
                     "Upper canopy cover")
        self._add_2d(ds, "fl", state.vegetation.fl, "fraction",
                     "Lower canopy cover")
        if state.vegetation.lai.ndim == 2:
            self._add_2d(ds, "laiu", state.vegetation.lai[:, 1],
                         "m2/m2", "Upper canopy LAI")
            self._add_2d(ds, "lail", state.vegetation.lai[:, 0],
                         "m2/m2", "Lower canopy LAI")
        # Above-ground biomass (sum of wood carbon across PFTs)
        import jax.numpy as jnp
        cveg = jnp.sum(state.vegetation.cbiow, axis=1)
        self._add_2d(ds, "cveg", cveg, "kgC/m2",
                     "Above-ground biomass C")

        # Carbon pools — compute from components to avoid stale diagnostic fields
        c = state.carbon
        totcsoi = c.csoipas + c.csoislon + c.csoislop + c.totcmic
        totlit = (c.clitlm + c.clitls + c.clitll +
                  c.clitrm + c.clitrs + c.clitrl +
                  c.clitwm + c.clitws + c.clitwl)
        self._add_2d(ds, "totcsoi", totcsoi, "kgC/m2",
                     "Total soil carbon")
        self._add_2d(ds, "totlit", totlit, "kgC/m2",
                     "Total litter carbon")

        # Soil profiles
        layer_coords = np.arange(state.soil.tsoi.shape[1])
        self._add_3d(ds, "tsoi", state.soil.tsoi, "soilay", layer_coords,
                     "K", "Soil temperature")
        self._add_3d(ds, "wsoi", state.soil.wsoi, "soilay", layer_coords,
                     "fraction", "Soil liquid moisture")

        path = os.path.join(
            self.output_dir,
            f"{self.prefix}_yearly_{year}.nc"
        )
        ds.to_netcdf(path)
        ds.close()
        return path
