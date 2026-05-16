#!/usr/bin/env python3
"""Generate a global 0.5° land mask (surta.nc) from ESACCI land cover data.

A grid cell is flagged as land if at least one IBIS vegetation type has
non-zero fraction in the ESACCI land cover map. Ocean cells have all-zero
vegfrac. This correctly excludes open ocean while including ice sheets,
tundra, and desert.

Note: GSWP3 provides data for ALL grid cells (including ocean), so it cannot
be used to derive a land mask.

Usage
-----
    python scripts/make_surta_global.py [--vegtype-dir DIR] [--out PATH]

Defaults
--------
    --vegtype-dir  /shared/nas2/Min/ESACCI_LC/ibis_vegtype
    --out          /shared/nas2/Min/HWSD/surta.nc
    --year         1992  (any available ESACCI year works; mask is near-static)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import netCDF4 as nc


def build_land_mask(vegtype_dir: str, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build land mask from ESACCI vegtype fractions.

    A cell is land if sum(vegfrac) > 0 (at least one PFT present).

    Returns
    -------
    land_mask : (360, 720) float32 — 1 = land, 0 = ocean
    lat : (360,) float64           — N to S (89.75 → -89.75)
    lon : (720,) float64           — W to E (-179.75 → 179.75)
    """
    path = os.path.join(vegtype_dir, f"vegtype_frac_{year}.nc")
    if not os.path.exists(path):
        print(f"ERROR: ESACCI vegtype file not found: {path}", file=sys.stderr)
        sys.exit(1)

    ds = nc.Dataset(path, mode='r')
    lat = np.array(ds.variables['lat'][:])
    lon = np.array(ds.variables['lon'][:])
    # vegfrac: (lat, lon, vegtype=15)
    vegfrac = np.array(ds.variables['vegfrac'][:], dtype=np.float32)
    ds.close()

    # Land = any vegtype has non-zero fraction
    land_mask = (vegfrac.sum(axis=2) > 0).astype(np.float32)
    return land_mask, lat, lon


def write_surta(out_path: str, land_mask: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                source_file: str = '') -> None:
    """Write land mask to a NetCDF4 file."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    out = nc.Dataset(out_path, mode='w', format='NETCDF4')
    out.createDimension('lat', len(lat))
    out.createDimension('lon', len(lon))

    vlat = out.createVariable('lat', 'f8', ('lat',))
    vlat[:] = lat
    vlat.units = 'degrees_north'
    vlat.long_name = 'latitude'
    vlat.axis = 'Y'

    vlon = out.createVariable('lon', 'f8', ('lon',))
    vlon[:] = lon
    vlon.units = 'degrees_east'
    vlon.long_name = 'longitude'
    vlon.axis = 'X'

    vmask = out.createVariable('landfrac', 'f4', ('lat', 'lon'),
                               zlib=True, complevel=4)
    vmask[:] = land_mask
    vmask.long_name = 'Land fraction'
    vmask.units = '1'
    vmask.comment = '1 = land, 0 = ocean/ice; derived from GSWP3 tas coverage'

    out.description = 'Global 0.5-degree land mask for IBIS-JAX'
    out.source = source_file
    out.Conventions = 'CF-1.7'
    out.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vegtype-dir',
                        default='/shared/nas2/Min/ESACCI_LC/ibis_vegtype',
                        help='Directory with ESACCI vegtype_frac_YYYY.nc files')
    parser.add_argument('--out', default='/shared/nas2/Min/HWSD/surta.nc',
                        help='Output path for surta.nc')
    parser.add_argument('--year', type=int, default=1992,
                        help='ESACCI year to use for land mask (default 1992)')
    args = parser.parse_args()

    print(f"Building land mask from ESACCI {args.vegtype_dir} (year={args.year}) ...")
    land_mask, lat, lon = build_land_mask(args.vegtype_dir, args.year)

    n_land = int(land_mask.sum())
    n_total = land_mask.size
    print(f"Land cells: {n_land:,} / {n_total:,}  ({100*n_land/n_total:.1f}%)")

    source_file = f"vegtype_frac_{args.year}.nc"
    write_surta(args.out, land_mask, lat, lon, source_file)
    print(f"Written: {args.out}")


if __name__ == '__main__':
    main()
