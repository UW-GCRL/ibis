#!/usr/bin/env python3
"""Profile IBIS-JAX physics using JAX's XLA profiler.

Runs one warm-up month to compile everything, then profiles a second month
and reports which XLA operations dominate GPU time.

Usage:
    python scripts/profile_physics.py configs/global_spinup.yaml
Output: /tmp/jax-profile/  (view at https://ui.perfetto.dev or chrome://tracing)
"""

from __future__ import annotations

import gzip, json, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
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
import netCDF4 as nc


def setup(config):
    g = config.grid; f = config.forcing; t = config.time; p = config.physics

    ds = nc.Dataset(f.land_mask_file)
    lat_g = np.array(ds.variables['lat'][:])
    lon_g = np.array(ds.variables['lon'][:])
    lm = np.array(ds.variables['landfrac'][:])
    ds.close()

    lat_sel = (lat_g >= g.ssouth) & (lat_g <= g.snorth)
    lon_sel = (lon_g >= g.swest) & (lon_g <= g.seast)
    lat_sub = lat_g[lat_sel]; lon_sub = lon_g[lon_sel]
    gidx = np.where(lat_sel)[0]; gjdx = np.where(lon_sel)[0]
    mask_sub = lm[np.ix_(gidx, gjdx)]
    ij = np.argwhere(mask_sub > 0)
    latidx_sub = ij[:, 0].astype(np.int32)
    lonidx_sub = ij[:, 1].astype(np.int32)
    gswp3_latidx = gidx[latidx_sub]
    gswp3_lonidx = gjdx[lonidx_sub]
    npoi = len(gswp3_latidx)

    nlat = len(lat_sub); nlon = len(lon_sub)
    grid = GridConfig(nlon=nlon, nlat=nlat, npoi=npoi, xres=g.xres, yres=g.yres)
    state = create_ibis_state(grid, p.co2init, p.o2init)

    soil_ds = nc.Dataset(f.soil_file)
    sand_g = np.array(soil_ds.variables['sandpct'][:], dtype=np.float32)
    clay_g = np.array(soil_ds.variables['claypct'][:], dtype=np.float32)
    soil_ds.close()
    sand = np.where(np.isfinite(sand_g[:, gswp3_latidx, gswp3_lonidx].T),
                    sand_g[:, gswp3_latidx, gswp3_lonidx].T, 45.0)
    clay = np.where(np.isfinite(clay_g[:, gswp3_latidx, gswp3_lonidx].T),
                    clay_g[:, gswp3_latidx, gswp3_lonidx].T, 25.0)
    sp = compute_soil_properties(sand, clay)
    state = eqx.tree_at(
        lambda s: (s.soil.poros, s.soil.sfield, s.soil.swilt, s.soil.bex,
                   s.soil.suction, s.soil.hydraul, s.soil.sand, s.soil.clay,
                   s.soil.hsoi, s.soil.albsav, s.soil.albsan, s.soil.z0soi, s.soil.csoi),
        state,
        (sp['poros'], sp['sfield'], sp['swilt'], sp['bex'], sp['suction'],
         sp['hydraul'], sp['sand'], sp['clay'], sp['hsoi'],
         sp['albsav'], sp['albsan'], sp['z0soi'], sp['csoi']),
    )
    lat_land = lat_sub[latidx_sub].astype(np.float32)
    lat_rad = jnp.array(np.deg2rad(lat_land))
    params = default_sim_params(npoi)
    params = eqx.tree_at(lambda p: p.latitude, params, lat_rad)
    return state, params, PhysicalConstants(), gswp3_latidx, gswp3_lonidx, lat_rad


def parse_trace(trace_dir: str) -> dict[str, float]:
    """Walk trace_dir and sum GPU kernel durations by op name."""
    candidate = None
    for root, dirs, files in os.walk(trace_dir):
        for fname in sorted(files):
            if fname.endswith('.json.gz') or fname.endswith('.json'):
                candidate = os.path.join(root, fname)
                break
        if candidate:
            break

    if candidate is None:
        return {}

    print(f"  Parsing: {candidate}")
    try:
        if candidate.endswith('.gz'):
            with gzip.open(candidate) as fh:
                trace = json.load(fh)
        else:
            with open(candidate) as fh:
                trace = json.load(fh)
    except Exception as e:
        print(f"  Parse error: {e}")
        return {}

    events = trace.get('traceEvents', [])
    gpu_times: dict[str, float] = {}

    for ev in events:
        if ev.get('ph') == 'X' and ev.get('dur', 0) > 0:
            name = ev.get('name', '')
            cat = ev.get('cat', '')
            if cat in ('Kernel', 'gpu_memcpy', 'kernel') or (
                    cat == '' and name and not name.startswith('/')):
                key = name[:70].split('$$')[0].split('__')[0]
                gpu_times[key] = gpu_times.get(key, 0.0) + ev['dur']

    return gpu_times


def main():
    config = load_config(sys.argv[1])
    print("JAX backend:", jax.default_backend())

    print("Setting up ...")
    state, params, consts, latidx, lonidx, lat_rad = setup(config)

    f = config.forcing; t = config.time
    reader = GSWP3Reader(f.gswp3_dir)
    year = f.spinup_year_start if f.spinup_year_start > 0 else t.iyear0
    leap = is_leap_year(year)
    key = jax.random.PRNGKey(0)

    # Warmup: compile the nested scan
    print("Warmup (compiling month 0) ...")
    t0 = time.perf_counter()
    state, key, _ = run_month_gswp3(state, consts, params, reader,
                                  latidx, lonidx, np.array(lat_rad),
                                  year, 0, key, t.dtime, leap)
    jax.effects_barrier()
    print(f"  {time.perf_counter()-t0:.1f}s")

    # Baseline wall time
    print("Baseline (month 1, no profiler) ...")
    t0 = time.perf_counter()
    state, key, _ = run_month_gswp3(state, consts, params, reader,
                                  latidx, lonidx, np.array(lat_rad),
                                  year, 1, key, t.dtime, leap)
    jax.effects_barrier()
    month_time = time.perf_counter() - t0
    ndays_feb = 29 if leap else 28
    print(f"  Wall time: {month_time:.2f}s  |  "
          f"per day: {month_time/ndays_feb:.3f}s  |  "
          f"per timestep: {month_time/ndays_feb/24*1000:.1f}ms")

    # Profiled run
    trace_dir = "/tmp/jax-profile"
    print(f"\nProfiling month 2 → {trace_dir} ...")
    with jax.profiler.trace(trace_dir, create_perfetto_link=False):
        state, key, _ = run_month_gswp3(state, consts, params, reader,
                                      latidx, lonidx, np.array(lat_rad),
                                      year, 1, key, t.dtime, leap)
        jax.effects_barrier()

    print("\nParsing trace ...")
    gpu_times = parse_trace(trace_dir)

    if gpu_times:
        total_gpu = sum(gpu_times.values())
        print(f"\n{'Top GPU kernels by cumulative time':<57} {'ms':>8}  {'%':>6}")
        print("-" * 74)
        for name, us in sorted(gpu_times.items(), key=lambda x: -x[1])[:25]:
            print(f"  {name:<55} {us/1000:>8.1f}  {100*us/total_gpu:>5.1f}%")
        print("-" * 74)
        print(f"  {'Total traced GPU time':<55} {total_gpu/1000:>8.1f} ms")
        print(f"  {'Wall time (month)':<55} {month_time*1000:>8.0f} ms")
    else:
        print("Could not auto-parse trace (no JSON events found).")

    print(f"\nFull trace: {trace_dir}/")
    print("Interactive view: drag the .json.gz to https://ui.perfetto.dev")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/profile_physics.py configs/global_spinup.yaml")
        sys.exit(1)
    main()
