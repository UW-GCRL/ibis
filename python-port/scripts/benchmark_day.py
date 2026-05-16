#!/usr/bin/env python3
"""Time one simulated month and extrapolate to year/spinup.

Usage:
    python scripts/benchmark_day.py configs/global_spinup.yaml [--nmonths 2]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ibis.io.config_reader import load_config
from ibis.io.soil_init import compute_soil_properties
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.constants import PhysicalConstants
from ibis.dimensions import GridConfig
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.loops import SimParams, default_sim_params
from ibis.timestepping.loops_gswp3 import run_month_gswp3
from ibis.timestepping.calendar import is_leap_year

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("bench")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--nmonths', type=int, default=2,
                        help='Months to time (month 1 = JIT warmup, rest = steady state)')
    args = parser.parse_args()

    config = load_config(args.config)
    g = config.grid
    f = config.forcing
    t = config.time
    p = config.physics

    log.info("JAX backend: %s  devices: %s", jax.default_backend(), jax.devices())

    # ---- land mask ----
    log.info("Loading land mask from %s ...", f.land_mask_file)
    import netCDF4 as nc
    ds = nc.Dataset(f.land_mask_file)
    lat_g = np.array(ds.variables['lat'][:])
    lon_g = np.array(ds.variables['lon'][:])
    for vn in ('landfrac', 'surta', 'mask', 'land'):
        if vn in ds.variables:
            lm = np.array(ds.variables[vn][:])
            break
    ds.close()

    lat_sel = (lat_g >= g.ssouth) & (lat_g <= g.snorth)
    lon_sel = (lon_g >= g.swest)  & (lon_g <= g.seast)
    lat_sub = lat_g[lat_sel]; lon_sub = lon_g[lon_sel]
    gidx = np.where(lat_sel)[0]; gjdx = np.where(lon_sel)[0]
    mask_sub = lm[np.ix_(gidx, gjdx)]
    ij = np.argwhere(mask_sub > 0)
    latidx_sub = ij[:, 0].astype(np.int32)
    lonidx_sub = ij[:, 1].astype(np.int32)
    gswp3_latidx = gidx[latidx_sub]
    gswp3_lonidx = gjdx[lonidx_sub]
    npoi = len(gswp3_latidx)
    log.info("Land points: %d", npoi)

    # ---- state ----
    nlat = len(lat_sub); nlon = len(lon_sub)
    grid = GridConfig(nlon=nlon, nlat=nlat, npoi=npoi, xres=g.xres, yres=g.yres)
    state = create_ibis_state(grid, p.co2init, p.o2init)

    # soil
    soil_nc = nc.Dataset(f.soil_file)
    for vn in ('sandpct', 'sand'):
        if vn in soil_nc.variables:
            sand_g = np.array(soil_nc.variables[vn][:], dtype=np.float32); break
    for vn in ('claypct', 'clay'):
        if vn in soil_nc.variables:
            clay_g = np.array(soil_nc.variables[vn][:], dtype=np.float32); break
    soil_nc.close()
    sand = sand_g[:, gswp3_latidx, gswp3_lonidx].T
    clay = clay_g[:, gswp3_latidx, gswp3_lonidx].T
    sand = np.where(np.isfinite(sand), sand, 45.0)
    clay = np.where(np.isfinite(clay), clay, 25.0)
    soil_props = compute_soil_properties(sand, clay)
    state = eqx.tree_at(
        lambda s: (s.soil.poros, s.soil.sfield, s.soil.swilt, s.soil.bex,
                   s.soil.suction, s.soil.hydraul, s.soil.sand, s.soil.clay,
                   s.soil.hsoi, s.soil.albsav, s.soil.albsan, s.soil.z0soi,
                   s.soil.csoi),
        state,
        (soil_props['poros'], soil_props['sfield'], soil_props['swilt'],
         soil_props['bex'], soil_props['suction'], soil_props['hydraul'],
         soil_props['sand'], soil_props['clay'], soil_props['hsoi'],
         soil_props['albsav'], soil_props['albsan'], soil_props['z0soi'],
         soil_props['csoi']),
    )

    lat_land = lat_sub[latidx_sub].astype(np.float32)
    lat_rad = jnp.array(np.deg2rad(lat_land))
    params = default_sim_params(npoi)
    params = eqx.tree_at(lambda p: p.latitude, params, lat_rad)
    consts = PhysicalConstants()
    reader = GSWP3Reader(f.gswp3_dir)

    year = f.spinup_year_start if f.spinup_year_start > 0 else t.iyear0
    leap = is_leap_year(year)
    key = jax.random.PRNGKey(0)

    log.info("Running %d months (month 1 = JIT compilation + run) ...", args.nmonths)

    times = []
    for imonth in range(args.nmonths):
        t0 = time.perf_counter()
        state, key, _ = run_month_gswp3(
            state, consts, params, reader,
            gswp3_latidx, gswp3_lonidx, np.array(lat_rad),
            year, imonth, key, t.dtime, leap,
        )
        jax.effects_barrier()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        label = "(JIT compile+run)" if imonth == 0 else "(steady state)"
        log.info("  Month %d (%d days): %.1f s  %s",
                 imonth + 1, 28 + (imonth == 1 and leap) + [3,0,3,2,3,2,3,3,2,3,2,3][imonth],
                 elapsed, label)

    compile_time = times[0]
    steady = np.mean(times[1:]) if len(times) > 1 else times[0]

    log.info("")
    log.info("=== Timing summary ===")
    log.info("JIT compilation + month 1: %.1f s", compile_time)
    log.info("Steady-state per month:    %.1f s", steady)
    log.info("Steady-state per day:      %.2f s", steady / 30)
    log.info("")
    log.info("Extrapolations (steady-state):")
    log.info("  1 year  (12 months): %.0f s  = %.1f h", steady * 12, steady * 12 / 3600)
    log.info("  Spinup (200 years):  %.0f s  = %.1f h", steady * 12 * 200, steady * 12 * 200 / 3600)


if __name__ == '__main__':
    main()
