#!/usr/bin/env python3
"""IBIS-JAX global simulation runner with GSWP3 transient forcing.

Usage
-----
    python scripts/run_global.py global_config.yaml [--log-level INFO]

Configuration
-------------
See the example config below. Key fields under 'forcing':
    forcing_type : gswp3
    gswp3_dir    : /shared/nas2/Min/GSWP3
    vegtype_dir  : /shared/nas2/Min/ESACCI_LC/ibis_vegtype
    soil_file    : /shared/nas2/Min/HWSD/soita.nc
    land_mask_file: /shared/nas2/Min/HWSD/surta.nc   # or '' to derive from GSWP3

Example YAML
------------
grid:
  snorth: 90.0
  ssouth: -90.0
  swest: -180.0
  seast: 180.0
  xres: 0.5
  yres: 0.5

time:
  iyear0: 1901
  nrun: 10
  dtime: 3600.0
  irestart: 0

physics:
  co2init: 350.0e-6
  o2init: 0.209

output:
  output_dir: /path/to/output
  restart_dir: /path/to/restart
  iyearout: 1

forcing:
  forcing_type: gswp3
  gswp3_dir: /shared/nas2/Min/GSWP3
  vegtype_dir: /shared/nas2/Min/ESACCI_LC/ibis_vegtype
  soil_file: /shared/nas2/Min/HWSD/soita.nc
  land_mask_file: /shared/nas2/Min/HWSD/surta.nc
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ibis.io.config_reader import load_config, IBISConfig
from ibis.io.restart import save_state, load_state, restart_path
from ibis.io.diagnostics import DiagnosticWriter
from ibis.io.soil_init import compute_soil_properties
from ibis.io.veg_init import initialize_vegetation
from ibis.io.vegtype_reader import read_vegtype_year
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.io.physics_params import load_physics_params, build_all_params
from ibis.constants import PhysicalConstants
from ibis.dimensions import GridConfig
from ibis.state.model_state import IBISState, create_ibis_state
from ibis.timestepping.loops import SimParams
from ibis.timestepping.loops_gswp3 import run_month_gswp3
from ibis.physics.climate_analysis import climanl, climanl2, existence
from ibis.physics.vegetation_dynamics import dynaveg
from ibis.timestepping.calendar import NDAYPM, NDAYPM_LEAP, is_leap_year

logger = logging.getLogger("ibis_global")


# ---------------------------------------------------------------------------
# Land mask helpers
# ---------------------------------------------------------------------------

def _build_land_mask_from_gswp3(gswp3_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive global land mask from GSWP3 tas coverage (no surta.nc needed).

    Returns land_mask (360, 720), lat (360,), lon (720,).
    """
    from scripts.make_surta_global import build_land_mask
    logger.info("Deriving land mask from GSWP3 tas (year=1980, month=1) ...")
    return build_land_mask(gswp3_dir, test_year=1980, test_month=1)


def _load_land_mask_from_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read surta.nc and return land_mask, lat, lon."""
    import netCDF4 as nc
    ds = nc.Dataset(path, mode='r')
    lat = np.array(ds.variables['lat'][:])
    lon = np.array(ds.variables['lon'][:])
    # Accept 'landfrac', 'surta', 'mask' — whichever exists
    for vname in ('landfrac', 'surta', 'mask', 'land'):
        if vname in ds.variables:
            land_mask = np.array(ds.variables[vname][:], dtype=np.float32)
            break
    else:
        raise KeyError(f"Cannot find land mask variable in {path}")
    ds.close()
    return land_mask, lat, lon


def _subset_land_mask(
    land_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    snorth: float, ssouth: float,
    swest: float, seast: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Restrict land mask to a lat/lon bounding box.

    Returns latindex, lonindex, lat_sub, lon_sub, land_mask_sub
    (indices are into the *sub-grid*, not the global grid).
    Also returns gswp3_latindex, gswp3_lonindex into the full 360×720 grid.
    """
    lat_sel = (lat >= ssouth) & (lat <= snorth)
    lon_sel = (lon >= swest) & (lon <= seast)

    lat_sub = lat[lat_sel]
    lon_sub = lon[lon_sel]

    global_lat_idx = np.where(lat_sel)[0]
    global_lon_idx = np.where(lon_sel)[0]

    mask_sub = land_mask[np.ix_(global_lat_idx, global_lon_idx)]
    land_ij = np.argwhere(mask_sub > 0)

    # Indices into sub-grid (for output writers that use sub-grid coords)
    latindex_sub = land_ij[:, 0].astype(np.int32)
    lonindex_sub = land_ij[:, 1].astype(np.int32)

    # Indices into global GSWP3 grid (for data reading)
    gswp3_latindex = global_lat_idx[latindex_sub]
    gswp3_lonindex = global_lon_idx[lonindex_sub]

    return latindex_sub, lonindex_sub, lat_sub, lon_sub, gswp3_latindex, gswp3_lonindex


# ---------------------------------------------------------------------------
# Soil loading
# ---------------------------------------------------------------------------

def _load_soil(soil_file: str, gswp3_latindex: np.ndarray, gswp3_lonindex: np.ndarray) -> dict:
    """Read sand/clay from soita.nc and compute soil hydraulic properties.

    soita.nc has variables sandpct and claypct with dimensions
    (layer=6, lat=360, lon=720), lat N→S to match GSWP3.
    """
    import netCDF4 as nc

    ds = nc.Dataset(soil_file, mode='r')
    # Find sand variable
    for vname in ('sandpct', 'sand'):
        if vname in ds.variables:
            sand_grid = np.array(ds.variables[vname][:], dtype=np.float32)
            break
    else:
        raise KeyError(f"Cannot find sand variable in {soil_file}")
    for vname in ('claypct', 'clay'):
        if vname in ds.variables:
            clay_grid = np.array(ds.variables[vname][:], dtype=np.float32)
            break
    else:
        raise KeyError(f"Cannot find clay variable in {soil_file}")
    ds.close()

    # sand_grid shape: (layer, lat, lon) → extract land points → (npoi, layer)
    if sand_grid.ndim == 3:
        # (layer, lat, lon)
        sand = sand_grid[:, gswp3_latindex, gswp3_lonindex].T   # (npoi, layer)
        clay = clay_grid[:, gswp3_latindex, gswp3_lonindex].T
    else:
        raise ValueError(f"Unexpected sand array shape: {sand_grid.shape}")

    # Replace NaN with defaults (45% sand, 25% clay → loam)
    sand = np.where(np.isfinite(sand), sand, 45.0)
    clay = np.where(np.isfinite(clay), clay, 25.0)

    return compute_soil_properties(sand, clay)


# ---------------------------------------------------------------------------
# Vegetation loading
# ---------------------------------------------------------------------------

def _load_vegetation(
    vegtype_dir: str,
    year: int,
    gswp3_latindex: np.ndarray,
    gswp3_lonindex: np.ndarray,
    lat_deg: np.ndarray,
    tw_override: np.ndarray | None = None,
) -> dict:
    """Load ESACCI vegtype for *year* and initialize vegetation state.

    Parameters
    ----------
    lat_deg : (npoi,) float
        Land-point latitudes in degrees (fallback for C3/C4 split).
    tw_override : (npoi,) float, optional
        Warmest-month temperature from GSWP3 climate data. If provided,
        used for C3/C4 grass split instead of the latitude proxy.
    """
    _, vegdom = read_vegtype_year(vegtype_dir, year,
                                  gswp3_latindex, gswp3_lonindex)

    if tw_override is not None:
        tw = tw_override
    else:
        # Fallback: latitude proxy (only used if GSWP3 data not available)
        tw = np.where(np.abs(lat_deg) < 23.5, 30.0, 15.0).astype(np.float32)

    return initialize_vegetation(vegdom.astype(np.float32), tw=tw)


def _apply_vegetation(state: IBISState, veg_props: dict) -> IBISState:
    """Inject vegetation properties into IBISState."""
    return eqx.tree_at(
        lambda s: (
            s.vegetation.plai, s.vegetation.cbiol, s.vegetation.cbior,
            s.vegetation.cbiow, s.vegetation.biomass, s.vegetation.exist,
            s.vegetation.vegtype0, s.vegetation.sapfrac,
            s.vegetation.fu, s.vegetation.fl,
            s.vegetation.lai, s.vegetation.sai,
            s.vegetation.ztop, s.vegetation.zbot,
            s.vegetation.totlaiu, s.vegetation.totlail,
            s.vegetation.totbiou, s.vegetation.totbiol_agg,
            s.vegetation.froot,
            s.vegetation.specla, s.vegetation.aleaf,
            s.vegetation.aroot, s.vegetation.awood,
        ),
        state,
        (
            veg_props['plai'], veg_props['cbiol'], veg_props['cbior'],
            veg_props['cbiow'], veg_props['biomass'], veg_props['exist'],
            veg_props['vegtype0'], veg_props['sapfrac'],
            veg_props['fu'], veg_props['fl'],
            veg_props['lai'], veg_props['sai'],
            veg_props['ztop'], veg_props['zbot'],
            veg_props['totlaiu'], veg_props['totlail'],
            veg_props['totbiou'], veg_props['totbiol_agg'],
            veg_props['froot'],
            veg_props['specla'], veg_props['aleaf'],
            veg_props['aroot'], veg_props['awood'],
        ),
    )


def _apply_soil(state: IBISState, soil_props: dict) -> IBISState:
    """Inject soil properties into IBISState."""
    return eqx.tree_at(
        lambda s: (
            s.soil.poros, s.soil.sfield, s.soil.swilt,
            s.soil.bex, s.soil.suction, s.soil.hydraul,
            s.soil.sand, s.soil.clay, s.soil.hsoi,
            s.soil.albsav, s.soil.albsan, s.soil.z0soi,
            s.soil.csoi,
        ),
        state,
        (
            soil_props['poros'], soil_props['sfield'], soil_props['swilt'],
            soil_props['bex'], soil_props['suction'], soil_props['hydraul'],
            soil_props['sand'], soil_props['clay'], soil_props['hsoi'],
            soil_props['albsav'], soil_props['albsan'], soil_props['z0soi'],
            soil_props['csoi'],
        ),
    )


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------

def run_global_simulation(config: IBISConfig) -> None:
    """Run a global IBIS-JAX simulation driven by GSWP3 transient forcing."""
    consts = PhysicalConstants()
    g = config.grid
    t = config.time
    p = config.physics
    out = config.output
    f = config.forcing

    if f.forcing_type != 'gswp3':
        raise ValueError(
            f"run_global.py requires forcing_type='gswp3', got '{f.forcing_type}'. "
            "Use scripts/run_ibis.py for climatology runs."
        )

    # ------------------------------------------------------------------
    # 1. Land mask
    # ------------------------------------------------------------------
    if f.land_mask_file:
        logger.info("Loading land mask from %s ...", f.land_mask_file)
        land_mask, lat_global, lon_global = _load_land_mask_from_file(f.land_mask_file)
    else:
        land_mask, lat_global, lon_global = _build_land_mask_from_gswp3(f.gswp3_dir)

    (latindex_sub, lonindex_sub, lat_sub, lon_sub,
     gswp3_latindex, gswp3_lonindex) = _subset_land_mask(
        land_mask, lat_global, lon_global,
        g.snorth, g.ssouth, g.swest, g.seast,
    )

    npoi = len(gswp3_latindex)
    nlat = len(lat_sub)
    nlon = len(lon_sub)

    logger.info("Domain: %d×%d grid, %d land points", nlat, nlon, npoi)

    # ------------------------------------------------------------------
    # 2. GSWP3 reader
    # ------------------------------------------------------------------
    reader = GSWP3Reader(f.gswp3_dir)

    # ------------------------------------------------------------------
    # 3. Soil properties
    # ------------------------------------------------------------------
    soil_file = f.soil_file or os.path.join(config.input.input_dir, 'soita.nc')
    logger.info("Loading soil from %s ...", soil_file)
    soil_props = _load_soil(soil_file, gswp3_latindex, gswp3_lonindex)

    # ------------------------------------------------------------------
    # 4. Create or load state
    # ------------------------------------------------------------------
    grid = GridConfig(nlon=nlon, nlat=nlat, npoi=npoi, xres=g.xres, yres=g.yres)

    if t.irestart > 0:
        rpath = restart_path(out.restart_dir, t.irestart)
        logger.info("Loading restart from %s", rpath)
        template = create_ibis_state(grid, p.co2init, p.o2init)
        state, meta = load_state(rpath, template)
        logger.info("Restart metadata: %s", meta)
    else:
        logger.info("Cold start ...")
        state = create_ibis_state(grid, p.co2init, p.o2init)

    # Inject soil (always, since soil is static)
    state = _apply_soil(state, soil_props)

    # ------------------------------------------------------------------
    # 5. Initialize 30-year climate normals (BEFORE vegetation, so tw is available)
    # ------------------------------------------------------------------
    # Always reseed climate normals from GSWP3 data at the start of every run.
    first_year = f.spinup_year_start if f.spinup_year_start > 0 else t.iyear0
    logger.info("  Seeding climate normals from GSWP3 year %d ...", first_year)
    leap0 = is_leap_year(first_year)
    ndaypm0 = tuple(int(d) for d in (NDAYPM_LEAP if leap0 else NDAYPM).tolist())
    xint = np.zeros((npoi, 12), dtype=np.float32)
    for _im in range(12):
        _m = reader.read_month(first_year, _im, gswp3_latindex, gswp3_lonindex)
        xint[:, _im] = np.nanmean(_m['tas'], axis=0) - 273.16  # K → °C
    tc0, tw0, tcmin0, gdd0_0, gdd5_0 = climanl(
        jnp.array(xint), jnp.zeros(npoi), ndaypm0,
    )

    # ------------------------------------------------------------------
    # 5b. Initial vegetation (cold start only)
    # ------------------------------------------------------------------
    lat_land_deg = lat_sub[latindex_sub].astype(np.float32)

    if t.irestart == 0 and f.vegtype_dir:
        logger.info("Initializing vegetation from ESACCI year %d ...", t.iyear0)
        # Use GSWP3-derived warmest-month temperature instead of latitude proxy
        tw_gswp3 = np.array(tw0, dtype=np.float32)
        veg_props = _load_vegetation(f.vegtype_dir, t.iyear0,
                                     gswp3_latindex, gswp3_lonindex, lat_land_deg,
                                     tw_override=tw_gswp3)
        state = _apply_vegetation(state, veg_props)
    state = eqx.tree_at(
        lambda s: (s.vegetation.tc, s.vegetation.tw, s.vegetation.tcmin,
                   s.vegetation.gdd0, s.vegetation.gdd5),
        state, (tc0, tw0, tcmin0, gdd0_0, gdd5_0),
    )
    logger.info("  Climate normals seeded (tc range: %.1f–%.1f °C)",
                float(jnp.nanmin(np.array(tc0))), float(jnp.nanmax(np.array(tc0))))

    if t.irestart == 0:
        # Cold start only: also seed a10td and phenology triggers to avoid
        # the model treating the first month as -273°C (absolute zero).
        a10td_init = jnp.array(np.nanmean(xint, axis=1) + 273.16)  # °C → K
        state = eqx.tree_at(
            lambda s: (s.summary.a10td, s.vegetation.tempu, s.vegetation.templ),
            state,
            (a10td_init, jnp.ones(npoi), jnp.ones(npoi)),
        )
        logger.info("  a10td seeded (%.1f–%.1f K), tempu/templ=1.0",
                    float(np.nanmin(np.array(a10td_init))),
                    float(np.nanmax(np.array(a10td_init))))

    # ------------------------------------------------------------------
    # 6. Simulation parameters (latitude in radians)
    # ------------------------------------------------------------------
    lat_rad = jnp.array(np.deg2rad(lat_land_deg))
    _pdict = load_physics_params(config.params_file) if config.params_file else None
    params, pft_params, bgc_params = build_all_params(npoi, lat_rad, _pdict)

    # Set PFT existence from (re)seeded climate normals
    exist_init = existence(
        state.vegetation.tcmin, state.vegetation.tw,
        state.vegetation.gdd0, state.vegetation.gdd5,
        pft_params,
    )
    state = eqx.tree_at(lambda s: s.vegetation.exist, state, exist_init)
    logger.info("  PFT existence set from climate normals "
                "(mean exist fraction: %.2f)",
                float(jnp.nanmean(np.array(exist_init))))

    # ------------------------------------------------------------------
    # 7. Diagnostic writer and random key
    # ------------------------------------------------------------------
    writer = DiagnosticWriter(out.output_dir, lon_sub, lat_sub,
                              latindex_sub, lonindex_sub)
    key = jax.random.PRNGKey(42)

    # Spinup configuration
    spinup = (f.spinup_year_start > 0)
    if spinup:
        span = f.spinup_year_end - f.spinup_year_start + 1
        logger.info("Spinup mode: cycling GSWP3 years %d–%d (%d-year period) for %d years",
                    f.spinup_year_start, f.spinup_year_end, span, t.nrun)

    # ------------------------------------------------------------------
    # 8. Year loop
    # ------------------------------------------------------------------
    for iyear_offset in range(t.nrun):
        year = t.iyear0 + iyear_offset
        t0 = time.time()
        logger.info("Year %d (%d/%d) ...", year, iyear_offset + 1, t.nrun)

        # Map to forcing year (cycle spinup period, or use actual year)
        if spinup:
            forcing_year = f.spinup_year_start + (iyear_offset % span)
            logger.info("  Forcing year: %d", forcing_year)
        else:
            forcing_year = year

        # Optionally update vegetation annually from ESACCI (transient only)
        if p.isimveg > 0 and f.vegtype_dir and not spinup:
            logger.info("  Updating vegetation from ESACCI year %d ...", forcing_year)
            veg_props = _load_vegetation(f.vegtype_dir, forcing_year,
                                         gswp3_latindex, gswp3_lonindex, lat_land_deg)
            state = _apply_vegetation(state, veg_props)

        # nsoilcall: number of soilbgc calls per simulated day.
        # nsoilcall > 1 fast-forwards BGC equilibration but multiplies
        # the GPU kernel size and compile time proportionally.
        # For the global domain (>50k land points) the physics kernel
        # already dominates; nsoilcall=1 keeps the kernel manageable
        # and BGC equilibrates naturally over the 200-year spinup.
        nsoilcall = 1

        # Inline month loop so we can snapshot accumulators for monthly output
        leap = is_leap_year(forcing_year)
        ndaypm_this = NDAYPM_LEAP if leap else NDAYPM
        monthly_tmean = []   # (npoi,) °C per month — for climate normal update
        for imonth in range(12):
            logger.info("  Month %d/12", imonth + 1)

            # Read raw monthly GSWP3 temperatures for climate normal tracking.
            # File handles are cached in the reader so this is a cheap slice read.
            raw_month = reader.read_month(forcing_year, imonth,
                                          gswp3_latindex, gswp3_lonindex)
            tmean_c = np.nanmean(raw_month['tas'], axis=0) - 273.16  # K → °C, (npoi,)
            monthly_tmean.append(tmean_c)

            # Snapshot annual accumulators before the month
            if out.imonthout > 0:
                prcp_before = state.summary.ayprcp
                aet_before  = state.summary.ayaet
                npp_before  = state.summary.ayanpptot
                nee_before  = state.summary.ayneetot
                gpp_before  = state.summary.aygpptot
                rh_before   = state.summary.ayco2mic
                runof_before = state.summary.aytrunof
                drain_before = state.summary.aytdrain

            state, key, _ = run_month_gswp3(
                state, consts, params, reader,
                gswp3_latindex, gswp3_lonindex, np.array(lat_rad),
                forcing_year, imonth, key, t.dtime, leap, grid.nband, nsoilcall,
            )

            # Write monthly output
            if out.imonthout > 0:
                monthly_fluxes = {
                    'amprcp':   (state.summary.ayprcp   - prcp_before, 'mm',      'Monthly precipitation'),
                    'amaet':    (state.summary.ayaet    - aet_before,  'mm',      'Monthly AET'),
                    'amnpptot': (state.summary.ayanpptot - npp_before, 'kgC/m2', 'Monthly NPP'),
                    'amneetot': (state.summary.ayneetot  - nee_before, 'kgC/m2', 'Monthly NEE'),
                    'amgpptot': (state.summary.aygpptot  - gpp_before, 'kgC/m2', 'Monthly GPP'),
                    'amco2mic': (state.summary.ayco2mic  - rh_before,  'kgC/m2', 'Monthly Rh'),
                    'amtrunof': (state.summary.aytrunof  - runof_before, 'mm',   'Monthly runoff'),
                    'amtdrain': (state.summary.aytdrain  - drain_before, 'mm',   'Monthly drainage'),
                }
                mpath = writer.write_monthly(state, year, imonth + 1, monthly_fluxes)
                logger.info("  Wrote monthly output: %s", mpath)

            # Save environment snapshot for calibration (if configured)
            env_dir = getattr(out, 'env_snapshot_dir', '')
            if env_dir and year >= getattr(out, 'env_snapshot_start', 0):
                from scripts.calibrate_jax import extract_env_snapshot
                env_snap = extract_env_snapshot(state)
                snap_path = os.path.join(env_dir, f"env_{year}_{imonth+1:02d}.npz")
                os.makedirs(env_dir, exist_ok=True)
                np.savez_compressed(snap_path,
                                    **{k: np.array(v) for k, v in env_snap.items()})
                logger.info("  Saved env snapshot: %s", snap_path)

        # ------------------------------------------------------------------
        # Annual post-processing (wired but not previously called)
        # ------------------------------------------------------------------
        # Compute this-year climate statistics from the monthly temperatures
        # accumulated above, then update the 30-year running climate means.
        xint_this = np.stack(monthly_tmean, axis=1)  # (npoi, 12) °C
        tcthis_np = np.nanmin(xint_this, axis=1)     # coldest month
        twthis_np = np.nanmax(xint_this, axis=1)     # warmest month
        gdd0this_np = np.zeros(npoi, dtype=np.float32)
        gdd5this_np = np.zeros(npoi, dtype=np.float32)
        for _im in range(12):
            ndays_m = int(ndaypm_this[_im])
            t_c = xint_this[:, _im]
            gdd0this_np += np.maximum(0.0, t_c) * ndays_m
            gdd5this_np += np.maximum(0.0, t_c - 5.0) * ndays_m

        state = eqx.tree_at(
            lambda s: (s.vegetation.tcthis, s.vegetation.twthis,
                       s.vegetation.gdd0this, s.vegetation.gdd5this),
            state,
            (jnp.array(tcthis_np), jnp.array(twthis_np),
             jnp.array(gdd0this_np), jnp.array(gdd5this_np)),
        )
        logger.info("  Climate this-year: tc_min=%.1f tw_max=%.1f "
                    "gdd0=%.0f gdd5=%.0f (domain means)",
                    float(np.nanmean(tcthis_np)), float(np.nanmean(twthis_np)),
                    float(np.nanmean(gdd0this_np)), float(np.nanmean(gdd5this_np)))

        # Update 30-year running climate means
        veg = state.vegetation
        deltat = jnp.zeros(npoi)
        tc_new, tw_new, tcmin_new, gdd0_new, gdd5_new = climanl2(
            veg.tc, veg.tw, veg.tcmin,
            veg.gdd0, veg.gdd5,
            veg.tcthis, veg.twthis,
            veg.gdd0this, veg.gdd5this,
            deltat,
        )
        state = eqx.tree_at(
            lambda s: (s.vegetation.tc, s.vegetation.tw, s.vegetation.tcmin,
                       s.vegetation.gdd0, s.vegetation.gdd5),
            state,
            (tc_new, tw_new, tcmin_new, gdd0_new, gdd5_new),
        )

        # Update PFT existence flags from bioclimatic limits
        exist_new = existence(
            state.vegetation.tcmin, state.vegetation.tw,
            state.vegetation.gdd0, state.vegetation.gdd5,
            pft_params,
        )
        state = eqx.tree_at(
            lambda s: s.vegetation.exist, state, exist_new,
        )

        # Annual biomass update: NPP → allocation → biomass → LAI → litterfall
        # ALWAYS runs to couple LAI to the carbon cycle
        from ibis.physics.vegetation_dynamics import biomass_update
        new_veg, new_carbon, new_summary = biomass_update(
            state.vegetation, state.carbon, state.summary,
            pft_params, consts.epsilon,
        )
        state = eqx.tree_at(
            lambda s: (s.vegetation, s.carbon, s.summary),
            state, (new_veg, new_carbon, new_summary),
        )
        logger.info("  biomass_update: LAI and litterfall updated from NPP")

        # Full vegetation dynamics (PFT competition, disturbance, reclassification)
        # Only with isimveg > 0
        if p.isimveg > 0:
            new_veg, new_carbon, new_summary = dynaveg(
                state.vegetation, state.carbon, state.summary,
                pft_params, p.isimfire, consts.epsilon,
            )
            state = eqx.tree_at(
                lambda s: (s.vegetation, s.carbon, s.summary),
                state, (new_veg, new_carbon, new_summary),
            )
            logger.info("  dynaveg: PFT competition and disturbance applied")

        elapsed = time.time() - t0
        logger.info("Year %d complete in %.1f s", year, elapsed)

        # Write yearly output (before resetting accumulators)
        logger.info("  Accumulators: ayprcp_max=%.1f mm  ayaet_max=%.1f mm  "
                    "ayanpptot_max=%.4f kgC/m2  ayneetot_min=%.4f kgC/m2",
                    float(jnp.max(state.summary.ayprcp)),
                    float(jnp.max(state.summary.ayaet)),
                    float(jnp.max(state.summary.ayanpptot)),
                    float(jnp.min(state.summary.ayneetot)))
        if out.iyearout > 0 and (iyear_offset + 1) % out.iyearout == 0:
            path = writer.write_yearly(state, year)
            logger.info("Wrote yearly output: %s", path)

        # Reset annual accumulators for the next year
        state = eqx.tree_at(
            lambda s: (s.summary.aynpp, s.summary.ayanpp, s.summary.ayanpptot,
                       s.summary.ayneetot,
                       s.summary.ayprcp, s.summary.ayaet,
                       s.summary.aygpptot, s.summary.ayco2mic,
                       s.summary.ayco2root,
                       s.summary.aytrunof, s.summary.aytdrain),
            state,
            (jnp.zeros_like(state.summary.aynpp),
             jnp.zeros_like(state.summary.ayanpp),
             jnp.zeros_like(state.summary.ayanpptot),
             jnp.zeros_like(state.summary.ayneetot),
             jnp.zeros_like(state.summary.ayprcp),
             jnp.zeros_like(state.summary.ayaet),
             jnp.zeros_like(state.summary.aygpptot),
             jnp.zeros_like(state.summary.ayco2mic),
             jnp.zeros_like(state.summary.ayco2root),
             jnp.zeros_like(state.summary.aytrunof),
             jnp.zeros_like(state.summary.aytdrain)),
        )

        # Save restart
        rpath = restart_path(out.restart_dir, year)
        save_state(state, rpath, metadata={"year": year})
        logger.info("Wrote restart: %s", rpath)

    logger.info("Simulation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="IBIS-JAX global runner (GSWP3 transient forcing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--platform', default=None,
                        choices=['cpu', 'gpu', 'tpu'],
                        help='JAX backend (default: auto-detect)')
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.platform:
        jax.config.update('jax_platform_name', args.platform)

    logger.info("JAX backend: %s", jax.default_backend())
    logger.info("Config: %s", args.config)

    config = load_config(args.config)
    run_global_simulation(config)


if __name__ == '__main__':
    main()
