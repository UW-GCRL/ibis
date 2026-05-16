#!/usr/bin/env python3
"""
scripts/preprocess_cru_ts4.py  --  Convert CRU TS4.09 monthly data to
IBIS anomaly files (temp.danom.nc, prec.danom.nc, cld.danom.nc, rh.danom.nc).

IBIS uses two forcing layers:
  1. Climatological monthly normals  (input/temp.mon.nc etc.)  - already IBIS-formatted
  2. Monthly anomalies               (input/anom/temp.danom.nc etc.) - generated here

At runtime, IBIS adds the anomaly for month M of year Y to the climatological
mean for month M.  The anomalies here are computed relative to the 1961-1990
CRU TS4 mean, so they are internally consistent.

Anomaly files produced:
  <run_dir>/input/anom/temp.danom.nc  -- temperature (°C)
  <run_dir>/input/anom/prec.danom.nc  -- precipitation (mm/day)
  <run_dir>/input/anom/cld.danom.nc   -- cloud cover (%)
  <run_dir>/input/anom/rh.danom.nc    -- relative humidity (%)

CRU TS4 inputs required (all in <cru_ts4_dir>/):
  cru_ts4.09.1901.2024.tmp.dat.nc   -- near-surface temperature (°C)
  cru_ts4.09.1901.2024.pre.dat.nc   -- precipitation (mm/month)
  cru_ts4.09.1901.2024.cld.dat.nc   -- cloud cover (%)
  cru_ts4.09.1901.2024.vap.dat.nc   -- vapour pressure (hPa)

Usage:
  python3 scripts/preprocess_cru_ts4.py <run_dir> <cru_ts4_dir>

Example:
  python3 scripts/preprocess_cru_ts4.py runs/india_cru \\
          /Users/minchen/Downloads/ibisrun/IBIS_input

After running this script, update ibis.infile in the run directory:
  iyranom = 1901   ! first year with anomalies
  nanom   = 124    ! number of years in anomaly file (1901-2024)
  iyear0  = YYYY   ! simulation start year (any year 1901-2024)
  nrun    = N      ! number of years to simulate
"""

import sys, os, subprocess, importlib

# ---------------------------------------------------------------------------
# Bootstrap dependencies into the tests/venv or system Python
# ---------------------------------------------------------------------------
REPO_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_DIR  = os.path.join(REPO_DIR, "tests", "venv")
venv_python = os.path.join(VENV_DIR, "bin", "python3")

def ensure_deps():
    missing = []
    for pkg in ("netCDF4", "numpy"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing {missing} ...", flush=True)
        subprocess.check_call([venv_python, "-m", "pip", "install", "--quiet"] + missing)
        os.execv(venv_python, [venv_python] + sys.argv)

ensure_deps()

import numpy as np
import netCDF4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRU_START_YEAR = 1901
CRU_END_YEAR   = 2024
CLM_START_YEAR = 1961   # 1961-1990 climatology reference period
CLM_END_YEAR   = 1990
FILL_VALUE     = 9.0e20

NYEARS  = CRU_END_YEAR - CRU_START_YEAR + 1      # 124
NMONTHS = NYEARS * 12                              # 1488

# Month index (0-based) for start/end of climatology
CLM_I0 = (CLM_START_YEAR - CRU_START_YEAR) * 12  # 720
CLM_I1 = (CLM_END_YEAR   - CRU_START_YEAR + 1) * 12  # 1080

# Days per calendar month (used to convert prec mm/month → mm/day)
DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                            dtype=np.float32)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def esat_hPa(T_degC):
    """Tetens saturation vapour pressure (hPa) for temperature T (°C)."""
    return 6.1078 * np.exp(17.27 * T_degC / (T_degC + 237.3))


def load_cru(path, varname):
    """Load a CRU TS4 variable; return (data, lons, lats) with NaN fill.

    CRU TS4 latitude order is S→N (-89.75 … 89.75).
    IBIS input files use N→S ( 89.75 … -89.75).
    This function flips the latitude axis so the output matches IBIS format.
    """
    print(f"  Reading {os.path.basename(path)} [{varname}] ...", flush=True)
    with netCDF4.Dataset(path) as ds:
        v    = ds.variables[varname]
        data = np.array(v[:], dtype=np.float32)   # (1488, 360, 720)  S→N
        fill = float(getattr(v, "_FillValue", 9.96921e+36))
        data[data >= 0.9 * fill] = np.nan
        lons = np.array(ds.variables["lon"][:], dtype=np.float32)
        lats = np.array(ds.variables["lat"][:], dtype=np.float32)

    # Flip latitude axis: S→N  →  N→S  (to match IBIS input convention)
    data = data[:, ::-1, :]          # (1488, 360, 720)  N→S
    lats = lats[::-1]                # 89.75 … -89.75

    print(f"    shape={data.shape}  lat {lats[0]:.2f}→{lats[-1]:.2f}"
          f"  range=[{np.nanmin(data):.3g}, {np.nanmax(data):.3g}]",
          flush=True)
    return data, lons, lats


def climatology(data_1488_360_720):
    """Compute 1961-1990 monthly climatology (12, 360, 720) from full record."""
    clm_block = data_1488_360_720[CLM_I0:CLM_I1]          # (360, 360, 720)
    clm_3d    = clm_block.reshape(CLM_END_YEAR - CLM_START_YEAR + 1, 12, 360, 720)
    return np.nanmean(clm_3d, axis=0)                      # (12, 360, 720)


def anomaly(data_1488_360_720, clm_12_360_720):
    """Subtract monthly climatology from full time series → anomaly."""
    data_4d = data_1488_360_720.reshape(NYEARS, 12, 360, 720)
    anom_4d = data_4d - clm_12_360_720                     # broadcast over years
    return anom_4d.reshape(NMONTHS, 360, 720)


def write_ibis_anom(outpath, varname, long_name, units, anom_tlatlons, lons, lats):
    """
    Write an IBIS anomaly NetCDF file.

    Layout matches the IBIS climatological files (temp.mon.nc etc.):
      dimensions: longitude=720, latitude=360, level=1, time=NMONTHS
      variable  : varname(time, level, latitude, longitude)
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ntime, nlat, nlon = anom_tlatlons.shape

    with netCDF4.Dataset(outpath, "w", format="NETCDF3_CLASSIC") as ds:
        ds.title   = f"CRU TS4.09 monthly {long_name} anomaly ({CRU_START_YEAR}-{CRU_END_YEAR})"
        ds.source  = "CRU TS4.09; anomaly relative to 1961-1990 mean"
        ds.history = "generated by scripts/preprocess_cru_ts4.py"
        ds.conventions = "NCAR-CSM"

        ds.createDimension("longitude", nlon)
        ds.createDimension("latitude",  nlat)
        ds.createDimension("level",     1)
        ds.createDimension("time",      None)   # unlimited

        vlon = ds.createVariable("longitude", "f4", ("longitude",))
        vlon.long_name = "longitude"
        vlon.units     = "degrees_east"
        vlon[:]        = lons

        vlat = ds.createVariable("latitude", "f4", ("latitude",))
        vlat.long_name = "latitude"
        vlat.units     = "degrees_north"
        vlat[:]        = lats

        vlev = ds.createVariable("level", "f4", ("level",))
        vlev.long_name = "_"
        vlev.units     = "level"
        vlev[:]        = [1.0]

        vtim = ds.createVariable("time", "f4", ("time",))
        vtim.long_name = "time"
        vtim.units     = f"months since {CRU_START_YEAR}-01-01"
        vtim[:]        = np.arange(ntime, dtype=np.float32)

        vdat = ds.createVariable(varname, "f4",
                                  ("time", "level", "latitude", "longitude"),
                                  fill_value=FILL_VALUE)
        vdat.long_name     = long_name
        vdat.units         = units
        vdat.missing_value = np.float32(FILL_VALUE)

        # Insert level dimension: (ntime,360,720) → (ntime,1,360,720)
        vdat[:] = anom_tlatlons[:, np.newaxis, :, :]

    size_mb = os.path.getsize(outpath) / 1e6
    print(f"  Wrote {outpath}  ({size_mb:.0f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    run_dir = sys.argv[1]
    cru_dir = sys.argv[2]

    anom_dir = os.path.join(run_dir, "input", "anom")

    def cru_path(suffix):
        return os.path.join(cru_dir, f"cru_ts4.09.1901.2024.{suffix}.dat.nc")

    # -----------------------------------------------------------------------
    # Temperature (°C)
    # -----------------------------------------------------------------------
    print("\n=== Temperature ===")
    tmp, lons, lats = load_cru(cru_path("tmp"), "tmp")
    clm_tmp = climatology(tmp)
    anom_tmp = anomaly(tmp, clm_tmp)
    write_ibis_anom(os.path.join(anom_dir, "temp.danom.nc"),
                    "temp", "temperature anomaly", "degC",
                    anom_tmp, lons, lats)

    # -----------------------------------------------------------------------
    # Precipitation (mm/month → mm/day)
    # -----------------------------------------------------------------------
    print("\n=== Precipitation ===")
    pre, _, _ = load_cru(cru_path("pre"), "pre")

    # Convert mm/month → mm/day (using mean days per month, ignoring leap years)
    days_tile = np.tile(DAYS_PER_MONTH, NYEARS)             # (1488,)
    pre_mmday = pre / days_tile[:, np.newaxis, np.newaxis]  # (1488, 360, 720)

    clm_pre = climatology(pre_mmday)
    anom_pre = anomaly(pre_mmday, clm_pre)
    write_ibis_anom(os.path.join(anom_dir, "prec.danom.nc"),
                    "prec", "precipitation anomaly", "mm/day",
                    anom_pre, lons, lats)
    del pre, pre_mmday  # free memory before loading next large array

    # -----------------------------------------------------------------------
    # Cloud cover (%)
    # -----------------------------------------------------------------------
    print("\n=== Cloud cover ===")
    cld, _, _ = load_cru(cru_path("cld"), "cld")
    clm_cld = climatology(cld)
    anom_cld = anomaly(cld, clm_cld)
    write_ibis_anom(os.path.join(anom_dir, "cld.danom.nc"),
                    "cld", "cloud cover anomaly", "percent",
                    anom_cld, lons, lats)
    del cld

    # -----------------------------------------------------------------------
    # Relative humidity (derived from vap + tmp)
    # RH = vapour_pressure / esat(T) * 100   (vap and esat both in hPa)
    # -----------------------------------------------------------------------
    print("\n=== Relative humidity (from vapour pressure + temperature) ===")
    vap, _, _ = load_cru(cru_path("vap"), "vap")
    rh = vap / esat_hPa(tmp) * 100.0
    rh = np.clip(rh, 0.0, 100.0)
    del vap, tmp  # free memory

    clm_rh = climatology(rh)
    anom_rh = anomaly(rh, clm_rh)
    write_ibis_anom(os.path.join(anom_dir, "rh.danom.nc"),
                    "rh", "relative humidity anomaly", "percent",
                    anom_rh, lons, lats)
    del rh

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"""
=== Done ===
Anomaly files written to: {anom_dir}/
  temp.danom.nc   temperature  (°C)
  prec.danom.nc   precipitation (mm/day)
  cld.danom.nc    cloud cover (%)
  rh.danom.nc     relative humidity (%)

Each file covers {CRU_START_YEAR}–{CRU_END_YEAR} ({NYEARS} years × 12 months = {NMONTHS} time steps).

Add/update these lines in {run_dir}/ibis.infile:
  {CRU_START_YEAR}      ! iyranom  year to start reading anomalies
  {NYEARS}        ! nanom    number of years in anomaly files

Then set iyear0 and nrun to any period within {CRU_START_YEAR}–{CRU_END_YEAR}.
""")


if __name__ == "__main__":
    main()
