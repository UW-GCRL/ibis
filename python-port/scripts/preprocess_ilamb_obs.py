#!/usr/bin/env python3
"""Pre-process ILAMB benchmark datasets into 0.5° numpy arrays for calibration.

Reads NetCDF files from ILAMB-Data and outputs annual-mean climatologies as
(360, 720) numpy arrays on the 0.5° grid that ibis-jax uses.

Usage:
    python scripts/preprocess_ilamb_obs.py
"""
import os
import numpy as np
import xarray as xr

ILAMB_DIR = "/home/min/ILAMB-Data/DATA"
OUT_DIR = "/home/min/ibis-jax/data/ilamb_obs"
os.makedirs(OUT_DIR, exist_ok=True)

# Target grid (ibis-jax 0.5°)
TARGET_LAT = np.arange(89.75, -90, -0.5)  # N→S
TARGET_LON = np.arange(-179.75, 180, 0.5)  # W→E


def regrid_to_05(da, method='nearest'):
    """Regrid a DataArray to 0.5° grid."""
    return da.interp(lat=TARGET_LAT, lon=TARGET_LON, method=method)


def process_gpp():
    """FLUXCOM GPP: monthly kgC/m²/s → annual mean kgC/m²/yr."""
    fpath = f"{ILAMB_DIR}/gpp/FLUXCOM/gpp.nc"
    if not os.path.exists(fpath):
        print("  SKIP: GPP not found")
        return
    ds = xr.open_dataset(fpath)
    # FLUXCOM GPP is typically in kgC/m²/s with time dimension
    varname = [v for v in ds.data_vars if 'gpp' in v.lower()][0]
    gpp = ds[varname]
    print(f"  GPP variable: {varname}, shape: {gpp.shape}, units: {gpp.attrs.get('units', '?')}")

    # Compute annual mean (stays in kgC/m²/s) then convert to kgC/m²/yr
    gpp_annual = gpp.mean(dim='time', skipna=True)

    # Fix coordinate names
    if 'latitude' in gpp_annual.dims:
        gpp_annual = gpp_annual.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Ensure lat/lon are standard names
    lat_name = [d for d in gpp_annual.dims if 'lat' in d.lower()][0]
    lon_name = [d for d in gpp_annual.dims if 'lon' in d.lower()][0]
    if lat_name != 'lat':
        gpp_annual = gpp_annual.rename({lat_name: 'lat'})
    if lon_name != 'lon':
        gpp_annual = gpp_annual.rename({lon_name: 'lon'})

    # Regrid to 0.5°
    gpp_05 = regrid_to_05(gpp_annual)
    gpp_kgC_yr = gpp_05.values * 86400 * 365  # kgC/m²/s → kgC/m²/yr

    np.save(f"{OUT_DIR}/gpp_annual_mean.npy", gpp_kgC_yr.astype(np.float32))
    print(f"  GPP: mean={np.nanmean(gpp_kgC_yr):.2f} kgC/m²/yr, "
          f"shape={gpp_kgC_yr.shape}, saved to {OUT_DIR}/gpp_annual_mean.npy")


def process_lai():
    """MODIS LAI: monthly m²/m² → annual mean."""
    fpath = f"{ILAMB_DIR}/lai/MODIS/lai_0.5x0.5.nc"
    if not os.path.exists(fpath):
        print("  SKIP: LAI not found")
        return
    ds = xr.open_dataset(fpath)
    varname = [v for v in ds.data_vars if 'lai' in v.lower()][0]
    lai = ds[varname]
    print(f"  LAI variable: {varname}, shape: {lai.shape}")

    lai_annual = lai.mean(dim='time', skipna=True)

    lat_name = [d for d in lai_annual.dims if 'lat' in d.lower()][0]
    lon_name = [d for d in lai_annual.dims if 'lon' in d.lower()][0]
    if lat_name != 'lat':
        lai_annual = lai_annual.rename({lat_name: 'lat'})
    if lon_name != 'lon':
        lai_annual = lai_annual.rename({lon_name: 'lon'})

    lai_05 = regrid_to_05(lai_annual)
    lai_vals = lai_05.values

    np.save(f"{OUT_DIR}/lai_annual_mean.npy", lai_vals.astype(np.float32))
    print(f"  LAI: mean={np.nanmean(lai_vals):.2f} m²/m², "
          f"shape={lai_vals.shape}, saved")


def process_csoil():
    """HWSD soil carbon: static kgC/m²."""
    fpath = f"{ILAMB_DIR}/cSoil/HWSD/soilc_0.5x0.5.nc"
    if not os.path.exists(fpath):
        print("  SKIP: cSoil not found")
        return
    ds = xr.open_dataset(fpath)
    varname = [v for v in ds.data_vars
               if any(x in v.lower() for x in ['soilc', 'csoil', 'carbon'])][0]
    csoil = ds[varname]
    print(f"  cSoil variable: {varname}, shape: {csoil.shape}")

    # May have time dimension (squeeze it)
    if 'time' in csoil.dims:
        csoil = csoil.isel(time=0)

    lat_name = [d for d in csoil.dims if 'lat' in d.lower()][0]
    lon_name = [d for d in csoil.dims if 'lon' in d.lower()][0]
    if lat_name != 'lat':
        csoil = csoil.rename({lat_name: 'lat'})
    if lon_name != 'lon':
        csoil = csoil.rename({lon_name: 'lon'})

    csoil_05 = regrid_to_05(csoil)
    csoil_vals = csoil_05.values

    np.save(f"{OUT_DIR}/csoil_mean.npy", csoil_vals.astype(np.float32))
    print(f"  cSoil: mean={np.nanmean(csoil_vals):.2f} kgC/m², "
          f"shape={csoil_vals.shape}, saved")


def process_biomass():
    """ESACCI biomass: static kgC/m²."""
    fpath = f"{ILAMB_DIR}/biomass/ESACCI/biomass.nc"
    if not os.path.exists(fpath):
        print("  SKIP: biomass not found")
        return
    ds = xr.open_dataset(fpath)
    varname = list(ds.data_vars)[0]
    biomass = ds[varname]
    print(f"  Biomass variable: {varname}, shape: {biomass.shape}, "
          f"units: {biomass.attrs.get('units', '?')}")

    if 'time' in biomass.dims:
        biomass = biomass.mean(dim='time', skipna=True)

    lat_name = [d for d in biomass.dims if 'lat' in d.lower()][0]
    lon_name = [d for d in biomass.dims if 'lon' in d.lower()][0]
    if lat_name != 'lat':
        biomass = biomass.rename({lat_name: 'lat'})
    if lon_name != 'lon':
        biomass = biomass.rename({lon_name: 'lon'})

    biomass_05 = regrid_to_05(biomass)
    biomass_vals = biomass_05.values

    np.save(f"{OUT_DIR}/cveg_mean.npy", biomass_vals.astype(np.float32))
    print(f"  Biomass: mean={np.nanmean(biomass_vals):.2f} kgC/m², "
          f"shape={biomass_vals.shape}, saved")


if __name__ == "__main__":
    print("Pre-processing ILAMB benchmark data to 0.5° numpy arrays...")
    print()
    print("[1/4] GPP (FLUXCOM)...")
    process_gpp()
    print()
    print("[2/4] LAI (MODIS)...")
    process_lai()
    print()
    print("[3/4] Soil Carbon (HWSD)...")
    process_csoil()
    print()
    print("[4/4] Biomass (ESACCI)...")
    process_biomass()
    print()
    print("Done! Files saved to:", OUT_DIR)
