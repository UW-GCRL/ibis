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
from ibis.physics.bedrock import BedrockParams, initialize_bedrock_temperature
from ibis.physics.prescribed_lai import apply_prescribed_lai
from ibis.io.modis_lai import read_modis_lai_max
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

def _load_soil(soil_file: str, gswp3_latindex: np.ndarray, gswp3_lonindex: np.ndarray,
               organic: bool = False, end_member: str = 'ls2008') -> dict:
    """Read sand/clay from soita.nc and compute soil hydraulic properties.

    soita.nc has variables sandpct and claypct with dimensions
    (layer=6, lat=360, lon=720), lat N→S to match GSWP3. With `organic`
    (physics.soil_organic) it also needs orgcpct and bulkdens
    (scripts/hwsd2_organic_to_nc.py), which set the organic-matter fraction
    of the solids per layer (ibis.io.soil_init); cells without HWSD2 data
    count as mineral.
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
    orgc_grid = bulk_grid = None
    if organic:
        for vname in ('orgcpct', 'bulkdens'):
            if vname not in ds.variables:
                ds.close()
                raise KeyError(f"soil_organic needs {vname} in {soil_file} (scripts/hwsd2_organic_to_nc.py)")
        orgc_grid = np.array(ds.variables['orgcpct'][:], dtype=np.float32)
        bulk_grid = np.array(ds.variables['bulkdens'][:], dtype=np.float32)
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

    orgc = bulk = None
    if organic:
        orgc = orgc_grid[:, gswp3_latindex, gswp3_lonindex].T
        bulk = bulk_grid[:, gswp3_latindex, gswp3_lonindex].T
        orgc = np.where(np.isfinite(orgc), orgc, 0.0)                      # no HWSD2 data: mineral
        bulk = np.where(np.isfinite(bulk) & (bulk > 0.0), bulk, 1.3)
    return compute_soil_properties(sand, clay, orgc=orgc, bulk=bulk, end_member=end_member)


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
            s.soil.csoi, s.soil.rhosoi,
        ),
        state,
        (
            soil_props['poros'], soil_props['sfield'], soil_props['swilt'],
            soil_props['bex'], soil_props['suction'], soil_props['hydraul'],
            soil_props['sand'], soil_props['clay'], soil_props['hsoi'],
            soil_props['albsav'], soil_props['albsan'], soil_props['z0soi'],
            soil_props['csoi'], soil_props['rhosoi'],
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
    _so_end = str(getattr(config.physics, "soil_organic_end_member", "ls2008"))
    soil_props = _load_soil(soil_file, gswp3_latindex, gswp3_lonindex,
                            organic=bool(getattr(config.physics, "soil_organic", False)), end_member=_so_end)
    if getattr(config.physics, "soil_organic", False):
        _fo = np.asarray(soil_props["forganic"])
        logger.info("Organic soil mixing ON (end-member %s): f_om top layer mean %.3f, >0.5 at %d of %d points; 1-2 m layer mean %.3f",
                    _so_end, float(_fo[:, 0].mean()), int((_fo[:, 0] > 0.5).sum()), _fo.shape[0], float(_fo[:, 4].mean()))

    # ------------------------------------------------------------------
    # 4. Create or load state
    # ------------------------------------------------------------------
    grid = GridConfig(nlon=nlon, nlat=nlat, npoi=npoi, xres=g.xres, yres=g.yres,
                      nbedrock=getattr(g, 'nbedrock', 0))

    # Tree-establishment gate (CALIBRATION.md 14). The StressAccum leaves are
    # attached only when the params activate the gate, so ungated runs keep the
    # 258-leaf restart layout. Decided from the yaml here because the state is
    # built before build_all_params runs below.
    _pd_gate = load_physics_params(config.params_file) if config.params_file else None
    _tsm = float(((_pd_gate or {}).get('pft') or {}).get('tree_stress_min', -1.0))
    _tcl = bool(((_pd_gate or {}).get('pft') or {}).get('tree_cover_limit', False))
    gate_on = _tsm >= 0.0 or _tcl          # the cover-limit form needs the accumulator too (CALIBRATION.md 70)
    et_on = bool(getattr(p, "et_components", False))
    _st_sec = ((_pd_gate or {}).get('stomata') or {}); _ph_sec = ((_pd_gate or {}).get('pheno') or {})
    seas_on = bool(_st_sec.get('acclim_on', False)) or bool(_ph_sec.get('sen_photoperiod_u', False)) \
        or bool(_ph_sec.get('sen_photoperiod_l', False))
    if seas_on:
        logger.info("Seasonal formulations ACTIVE (acclim_on=%s, sen_photoperiod_u=%s, sen_photoperiod_l=%s): "
                    "state carries SeasonalState (CALIBRATION.md 21-22)", bool(_st_sec.get('acclim_on', False)),
                    bool(_ph_sec.get('sen_photoperiod_u', False)), bool(_ph_sec.get('sen_photoperiod_l', False)))
    if et_on:
        logger.info("ET components ACCUMULATED (physics.et_components): state carries ETParts")
    if gate_on:
        logger.info("Tree-establishment gate ACTIVE (tree_stress_min=%.2f, ramp=%.2f): "
                    "state carries StressAccum; dynaveg gets tree_surv", _tsm,
                    float(_pd_gate['pft'].get('tree_stress_ramp', 0.2)))
        if _tcl:
            _pg = _pd_gate['pft']
            logger.info("  as a COVER LIMIT (tree_cover_limit): fu_max = clip((stress - %.2f) / %.2f, 0, 1) ^ %.2f, "
                        "no yearly thinning, elimination only at fu_max = 0 (CALIBRATION.md 70)",
                        float(_pg.get('tree_cover_min', 0.30)), float(_pg.get('tree_cover_ramp', 0.65)), float(_pg.get('tree_cover_power', 1.75)))
    else:
        logger.info("Tree-establishment gate inactive (tree_stress_min < 0)")

    if t.irestart > 0:
        rpath = restart_path(out.restart_dir, t.irestart)
        logger.info("Loading restart from %s", rpath)
        template = create_ibis_state(grid, p.co2init, p.o2init, with_stress=gate_on, with_etparts=et_on, with_seasonal=seas_on)
        state, meta = load_state(rpath, template)
        logger.info("Restart metadata: %s", meta)
        if state.wetland is not None:        # a methane-enabled restart: its annual sum belongs to the year it was written in
            from ibis.bgc.methane import reset_annual as _reset_ch4
            state = eqx.tree_at(lambda s: s.wetland, state, _reset_ch4(state.wetland))
    else:
        logger.info("Cold start ...")
        state = create_ibis_state(grid, p.co2init, p.o2init, with_stress=gate_on, with_etparts=et_on, with_seasonal=seas_on)

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
    # Seed the 30-year climate normals ONLY on a cold start.
    #
    # This used to run unconditionally, with the comment "always reseed at the
    # start of every run". On a cold start that is necessary -- the state
    # carries zeros. On a RESTART it overwrites normals the model spent
    # hundreds of years accumulating with values derived from a SINGLE year,
    # and `existence` reads tcmin/tw/gdd0/gdd5 directly. Measured against the
    # both_fix year-2200 checkpoint: mean |dtc| = 1.47 K (max 10.9), |dgdd5| =
    # 73 (max 774), and PFT existence flips in 6,034 cells (6.3%), of which
    # 3,173 are north of 50N -- precisely where leaf area is marginal. Since
    # climanl2 is a 30-year running mean, a restarted run needs ~30 years to
    # recover, so anything shorter carries the perturbation.
    #
    # Inverse of the defect-0 pattern this project keeps finding: a value that
    # IS written where it should not be, rather than one that never arrives.
    if t.irestart == 0:
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
    # Say which parameters are in play. Every run in this project that turned
    # out to be on the wrong parameters (both_fix's x0.54 inherited into the
    # seed test; a fit arm whose params_file had to be inferred from its GPP)
    # would have been caught at the log line, had there been one.
    if _pdict is not None:
        logger.info("Physics parameters: %s", config.params_file)
    else:
        logger.info("Physics parameters: IBIS DEFAULTS (no params_file)")
    _cropfrac = None
    if getattr(f, "cropfrac_file", ""):
        # the cropland map is on the IBIS north-to-south grid, the same as GSWP3's, so the GSWP3 row indices apply
        _cg = np.load(f.cropfrac_file)["cropfrac"]; _cropfrac = np.asarray(_cg[gswp3_latindex, gswp3_lonindex], np.float32)
        logger.info("Cropland as managed grass: %s, mean cropland fraction of the land points %.4f", f.cropfrac_file, float(_cropfrac.mean()))
    # Koppen-Geiger mask on PFT existence (CALIBRATION.md 44): a class map per 30-year period; the
    # year's period is looked up at every existence() call. None = the bioclimatic envelope alone.
    _koppen = None
    if getattr(f, "koppen_file", ""):
        from ibis.io.koppen import load_koppen, koppen_exist_mask, period_index, describe
        _koppen = load_koppen(f.koppen_file, gswp3_latindex, gswp3_lonindex)
        _m0 = koppen_exist_mask(_koppen["classes"][period_index(config.time.iyear0, _koppen["period_start"], _koppen["period_end"])])
        logger.info("Koppen mask on PFT existence: %s, periods %s; %s", f.koppen_file,
                    list(zip(_koppen["period_start"].tolist(), _koppen["period_end"].tolist())),
                    describe(_koppen["classes"][period_index(config.time.iyear0, _koppen["period_start"], _koppen["period_end"])], _m0))
    _larch_fix = bool(getattr(config.physics, "larch_gdd_fix", False))
    if _larch_fix:
        logger.info("PFT 8 (larch) existence: degree-day limit (larch_gdd_fix)")

    _types_on = bool(getattr(config.physics, "koppen_types", False))
    if _types_on and _koppen is None:
        raise ValueError("physics.koppen_types needs forcing.koppen_file (the Köppen npz of scripts/koppen_prep.py)")

    def _exist_mask_for(year):
        if _koppen is None or _types_on:      # the type map supersedes the mask (CALIBRATION.md 45.1)
            return None
        return koppen_exist_mask(_koppen["classes"][period_index(int(year), _koppen["period_start"], _koppen["period_end"])])

    _forganic = soil_props.get("forganic") if getattr(config.physics, "soil_organic", False) else None
    params, pft_params, bgc_params = build_all_params(npoi, lat_rad, _pdict, cropfrac=_cropfrac, forganic=_forganic)

    # IBIS 3.0 Köppen-zone types (CALIBRATION.md 45.1): the cell's zone maps its 12 positions to the 40 types and
    # every per-PFT parameter becomes per cell. The 12-entry base pair is kept so the map can be re-applied when a
    # year enters a new Köppen period (same shapes, so the compiled kernels are reused).
    _types_base = (params, pft_params); _types_period = None; _type_over = None; _zones = None
    if _types_on:
        from ibis.io import pft_types as _pt
        _type_over = _pt.parse_type_overrides((_pdict or {}).get("types"))
        _zones = _pt.zones_from_classes(_koppen["classes"])          # (nperiod, npoi)

    def _apply_types_for(year, params, pft_params):
        nonlocal _types_period
        if not _types_on:
            return params, pft_params
        k = period_index(int(year), _koppen["period_start"], _koppen["period_end"])
        if k == _types_period:
            return params, pft_params
        _types_period = k
        params, pft_params = _pt.apply_types(params, pft_params, _zones[k], _type_over, base=_types_base)
        logger.info("Köppen-zone types (period %d-%d): %d types with own parameters %s; land points per zone %s",
                    int(_koppen["period_start"][k]), int(_koppen["period_end"][k]),
                    len({t for v in _type_over.values() for t in v}), sorted(_type_over), _pt.type_area_summary(_zones[k]))
        if any(key in _type_over for key in ('tree_cover_min', 'tree_cover_ramp', 'tree_cover_power')):   # CALIBRATION.md 71
            _ramps = {}
            for z in _pt.ZONES:
                if z in ('none', 'E_ice'): continue
                vals = tuple(_pt.zone_scalar(z, _type_over.get(key, {}), 'all') for key in ('tree_cover_min', 'tree_cover_ramp', 'tree_cover_power'))
                if any(v is not None for v in vals): _ramps[z] = vals
            logger.info("  cover-limit ramp per zone (min, ramp, power; None = base): %s", _ramps)
        if 'som_k_scale' in _type_over:                                                                   # CALIBRATION.md 72
            _soms = {z: _pt.zone_scalar(z, _type_over['som_k_scale'], 'all') for z in _pt.ZONES if z not in ('none', 'E_ice')}
            logger.info("  SOM turnover factor per zone (som_k_scale; None = 1): %s", {z: v for z, v in _soms.items() if v is not None})
        return params, pft_params

    params, pft_params = _apply_types_for(config.time.iyear0, params, pft_params)
    if _types_on:
        logger.info("Köppen-zone types: positions per zone\n%s", _pt.describe_zones())

    # Seed thermal-only bedrock on a cold start. Downward continuation along a
    # geothermal gradient puts the new layers near equilibrium immediately;
    # a uniform guess would need centuries of spin-up to diffuse away. On a
    # restart the temperatures come from the checkpoint, so leave them alone.
    if t.irestart == 0 and state.soil.tsoi_deep is not None:
        bp = params.bedrock if params.bedrock is not None else BedrockParams()
        state = eqx.tree_at(
            lambda s: s.soil.tsoi_deep, state,
            initialize_bedrock_temperature(
                state.soil.tsoi[:, -1], state.soil.hsoi[:grid.nsoilay], bp))
        logger.info("  Seeded %d bedrock layers to %.1f m by downward continuation",
                    bp.nlayer, float(jnp.sum(state.soil.hsoi[:grid.nsoilay]))
                    + sum(bp.hsoi))

    # Prescribed LAI: load the observed field once. Applied after every annual
    # vegetation update so the model's own growth is overwritten each year.
    lai_obs = None
    lai_seed = None
    if getattr(p, "seed_lai", False):
        lai_seed = jnp.asarray(read_modis_lai_max(gswp3_latindex,
                                                  gswp3_lonindex))
        logger.info("  SEEDING leaf area from MODIS once: mean %.2f over %d "
                    "pts (then dynaveg evolves it)",
                    float(jnp.mean(lai_seed)), npoi)
        _before = float(jnp.mean(jnp.sum(state.vegetation.plai, axis=1)))
        state = apply_prescribed_lai(state, lai_seed, pft_params.specla,
                                     consts.epsilon)
        logger.info("  seeded LAI: %.2f -> %.2f (mean total). Not re-imposed "
                    "again; dynaveg evolves it from here.", _before,
                    float(jnp.mean(jnp.sum(state.vegetation.plai, axis=1))))
    if getattr(p, "prescribe_lai", False):
        lai_obs = jnp.asarray(read_modis_lai_max(gswp3_latindex, gswp3_lonindex))
        logger.info("  prescribed LAI from MODIS: mean %.2f max %.2f over %d pts",
                    float(jnp.mean(lai_obs)), float(jnp.max(lai_obs)), npoi)
        # Apply once up front too, so year 1 runs on the observed field rather
        # than whatever LAI the restart or cold start happens to carry.
        state = apply_prescribed_lai(state, lai_obs, pft_params.specla,
                                     consts.epsilon)

    # Set PFT existence from (re)seeded climate normals
    exist_init = existence(
        state.vegetation.tcmin, state.vegetation.tw,
        state.vegetation.gdd0, state.vegetation.gdd5,
        pft_params,
        exist_mask=_exist_mask_for(config.time.iyear0), larch_gdd_fix=_larch_fix,
        slot_parent=params.slot_parent, slot_gate=params.slot_gate, slot_typed=params.slot_typed,
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
    # --- wetland CH4 (docs/METHANE.md) ------------------------------------
    # `enable_methane` is read here so the flag actually does something. The
    # runner keys off `f_inund` being present; loading the climatology is what
    # the flag controls.
    ch4_clim = None
    ch4_params = None
    _wsrc = 'climatology'
    if getattr(p, "enable_methane", False):
        from ibis.io.wad2m import read_wad2m_climatology, read_wad2m_month, wad2m_in_record
        from ibis.bgc.methane import biome_methane_params
        # WAD2M runs south-to-north; the GSWP3 row indices run north-to-south. Until 2026-10-09 the GSWP3 indices were
        # used directly, which mirrored the wetland map in latitude (zero inundation north of 60 N, the Antarctic
        # rows). Index the climatology from the land points' own coordinates, as verify/methane/evaluate_global.py does.
        mlat = np.arange(-90.0, 90.0, 0.5)
        mlon = np.arange(-180.0, 180.0, 0.5)
        _lon_land_deg = lon_sub[lonindex_sub].astype(np.float32)
        _wli = np.rint((lat_land_deg - mlat[0]) / 0.5).astype(int).clip(0, 359)
        _wlo = np.rint((_lon_land_deg - mlon[0]) / 0.5).astype(int).clip(0, 719)
        ch4_clim = read_wad2m_climatology(f.wad2m_file, _wli, _wlo, mlat, mlon)
        # The VALIDATED configuration: per-biome r_ch4 and t_ref
        # (docs/METHANE.md 5q, 182 Tg/yr inside the GMB range). MethaneParams()
        # alone is the "global" configuration, which gives 458 Tg on the same
        # state -- and was silently what production used until 2026-09-04,
        # because methane_params was never passed to run_month_gswp3.
        # Base: the parameter file's `methane:` section (MethaneParams defaults without one); per-band overrides from its
        # `methane_bands:` section (CALIBRATION.md 38, the tower-class medians by band). Without either, the legacy
        # per-biome r_ch4 and t_ref tables apply, as before.
        _mb = (_pdict or {}).get('methane_bands') if _pdict else None
        _has_fit = bool(_pdict and (_pdict.get('methane') or _mb))
        # IBIS 3.0 types: the band is the wetland type of the cell's Köppen zone (W01/W02/W03; CALIBRATION.md 45), fixed at
        # the start period
        _bi = None
        if _types_on:
            _bi = _pt.wetland_band(_zones[period_index(config.time.iyear0, _koppen["period_start"], _koppen["period_end"])])
            logger.info("  CH4 bands by wetland type (Köppen zone): W01 %d, W02 %d, W03 %d land points, latitude rule %d",
                        int((_bi == 0).sum()), int((_bi == 1).sum()), int((_bi == 2).sum()), int((_bi < 0).sum()))
        ch4_params = (biome_methane_params(lat_land_deg, base=params.methane, bands=_mb, band_index=_bi) if _has_fit
                      else biome_methane_params(lat_land_deg, band_index=_bi))
        logger.info("  wetland CH4 enabled: WAD2M climatology loaded, "
                    "mean inundated fraction %.4f", float(ch4_clim.mean()))
        _wsrc = str(getattr(p, 'wetland_source', 'climatology'))
        if _wsrc not in ('climatology', 'observed', 'learned', 'observed_learned'):
            raise ValueError(f"physics.wetland_source must be 'climatology', 'observed', 'learned' or 'observed_learned', got {_wsrc!r}")
        logger.info("  wetland area source: %s%s", _wsrc, ' (WAD2M month of the calendar year inside 2000-2020, climatology outside)' if _wsrc == 'observed' else '')
        logger.info("  CH4 cadence: %s", 'hourly, averaged to the month (methane_hourly)' if getattr(p, 'methane_hourly', False) else 'end-of-month snapshot')
        logger.info("  CH4 params: %s", (f"parameter file ({'with' if _mb else 'without'} band overrides)" if _has_fit else "per-biome r_ch4 and t_ref (docs/METHANE.md 5q)"))

    # Dynamic wetland area (docs/METHANE.md 7; Min's point 4): the month's inundated fraction from the model's own
    # root-zone water as a standardised anomaly against a reference climatology, bounded by the cell's WAD2M range,
    # beta per band from the parameter file's `wetland_area:` section. Off (the default): the WAD2M climatology as before.
    _wdyn = None
    if ch4_clim is not None and getattr(p, "wetland_dynamic", False):
        from ibis.io.wad2m import read_wad2m_bounds
        from ibis.bgc import wetland_area as _wa
        if not getattr(f, "wetland_clim_file", ""):
            raise ValueError("physics.wetland_dynamic needs forcing.wetland_clim_file (verify/methane/wetland_clim_prep.py)")
        _wz = np.load(f.wetland_clim_file)
        _mu = np.asarray(_wz["mu"])[:, gswp3_latindex, gswp3_lonindex]; _sd = np.asarray(_wz["sd"])[:, gswp3_latindex, gswp3_lonindex]
        if not np.isfinite(_mu).all(): raise ValueError("wetland_clim_file has no climatology at some land points")
        _fmin, _fmax = read_wad2m_bounds(f.wad2m_file, _wli, _wlo, mlat, mlon)
        _wb = (_pdict or {}).get("wetland_area") or {}
        _bt = np.array([float(_wb.get(k, 0.0)) for k in ("tropical", "temperate", "boreal")])
        _lat = np.asarray(lat_land_deg); _band = np.where((_lat > -30.0) & (_lat < 30.0), 0, np.where(_lat >= 60.0, 2, 1))
        if _bi is not None: _band = np.where(np.asarray(_bi) >= 0, np.asarray(_bi), _band)      # the wetland type of the zone
        _wdyn = (_mu, _sd, _fmin, _fmax, _bt[_band])
        logger.info("  dynamic wetland area ON: climatology %s; beta tropical/temperate/boreal %s; cell bounds mean %.4f-%.4f",
                    f.wetland_clim_file, _bt.tolist(), float(_fmin.mean()), float(_fmax.mean()))

    # Learned wetland area (docs/METHANE.md 11): each month runs with a unit inundated fraction, so the hourly accumulator
    # integrates the flux per unit of wetland; after the month the predictors are built from the month's MEAN soil state and
    # its water totals, the hybrid fraction is computed, and the month's flux and the annual total are scaled by it (the
    # flux is linear in the fraction). The feature builder is stepped every month so its lags never break.
    _lw = _fbld = None
    if ch4_clim is not None and _wsrc in ('learned', 'observed_learned'):
        from ibis.bgc.wetland_learned import FeatureBuilder, LearnedWetland
        if not getattr(p, 'methane_hourly', False):
            raise ValueError("physics.wetland_source 'learned' / 'observed_learned' needs physics.methane_hourly: true")
        if not getattr(f, 'wetland_learned_file', ''):
            raise ValueError("physics.wetland_source 'learned' needs forcing.wetland_learned_file (verify/methane/wetland_learned_train.py --save)")
        _lw = LearnedWetland(f.wetland_learned_file, latindex_sub, lonindex_sub)
        _fbld = FeatureBuilder(latindex_sub, lonindex_sub, np.asarray(state.soil.poros), np.asarray(state.soil.hsoi), nlat=len(lat_sub), nlon=len(lon_sub))
        logger.info("  learned wetland area: %s (%d cells of the file on this run's land, %d dropped)", f.wetland_learned_file, len(_lw.pts), _lw.n_dropped)

    for iyear_offset in range(t.nrun):
        year = t.iyear0 + iyear_offset
        t0 = time.time()
        logger.info("Year %d (%d/%d) ...", year, iyear_offset + 1, t.nrun)
        params, pft_params = _apply_types_for(year, params, pft_params)       # new Köppen period: new type map
        if getattr(p, 'fire_annual_reset', True):                             # F77 stats.f:956 (CALIBRATION.md 59)
            from ibis.physics.fire import zero_fire_accumulators
            state = eqx.tree_at(lambda s: s.vegetation, state, zero_fire_accumulators(state.vegetation))

        # Map to forcing year (cycle spinup period, or use actual year)
        if spinup:
            forcing_year = f.spinup_year_start + (iyear_offset % span)
            logger.info("  Forcing year: %d", forcing_year)
        else:
            forcing_year = year

        # Historical CO2. `year`, not `forcing_year`: during a spin-up the
        # forcing cycles but calendar time does not advance, and a spin-up is
        # meant to equilibrate at one CO2 level rather than replay a
        # trajectory -- so this is only meaningful on a transient.
        if getattr(p, "transient_co2", False):
            from ibis.io.co2 import apply_co2, co2_ppm
            state = apply_co2(state, year)
            logger.info("  CO2: %.1f ppm", co2_ppm(year))

        # Optionally update vegetation annually from ESACCI (transient only)
        if p.isimveg > 0 and f.vegtype_dir and not spinup:
            logger.info("  Updating vegetation from ESACCI year %d ...", forcing_year)
            veg_props = _load_vegetation(f.vegtype_dir, forcing_year,
                                         gswp3_latindex, gswp3_lonindex, lat_land_deg)
            state = _apply_vegetation(state, veg_props)

        # nsoilcall: number of soilbgc calls per simulated day. Each call
        # advances the pools one day and applies one day of litterfall, so it
        # is a consistent clock acceleration rather than a bias.
        #
        # This was hardcoded to 1, with a comment claiming BGC "equilibrates
        # naturally over the 200-year spinup". That holds for a warm start, not
        # a cold one: the passive soil pool turns over on century to millennium
        # timescales, so 200 years at 1x leaves it far from equilibrium. It is
        # also why physics.soilcspin -- documented as "0=no spinup,
        # 1=acceleration" -- was a dead flag, read nowhere in the codebase.
        from ibis.io.config_reader import spinup_nsoilcall
        _tail = int(getattr(p, 'soilcspin_tail_years', 0))
        nsoilcall = spinup_nsoilcall(p.nsoilcall, p.soilcspin, iyear_offset, t.nrun, _tail)
        if nsoilcall == 1 and iyear_offset > 0 and spinup_nsoilcall(p.nsoilcall, p.soilcspin, iyear_offset - 1, t.nrun, _tail) > 1:
            logger.info("  soil acceleration off for the last %d years (soilcspin_tail_years)", _tail)

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
            if _lw is not None:
                _lw_before = (state.summary.ayprcp, state.summary.ayaet, state.summary.aytrunof, state.summary.aytdrain,
                              None if state.wetland is None else state.wetland.annual)
            if out.imonthout > 0:
                prcp_before = state.summary.ayprcp
                aet_before  = state.summary.ayaet
                npp_before  = state.summary.aynpptot
                gpp_before  = state.summary.aygpptot
                rh_before   = state.summary.ayco2mic
                runof_before = state.summary.aytrunof
                drain_before = state.summary.aytdrain

            # observed wetland area (docs/METHANE.md 8): the WAD2M field of this calendar month when the record has it
            _fobs = None
            if ch4_clim is not None and _wsrc == 'observed' and wad2m_in_record(year):
                _fobs = read_wad2m_month(f.wad2m_file, year, imonth + 1, _wli, _wlo, mlat, mlon, clamp_to_range=False)
            _learned_now = _lw is not None and not (_wsrc == 'observed_learned' and wad2m_in_record(year))
            if _wsrc == 'observed_learned' and ch4_clim is not None and wad2m_in_record(year):
                _fobs = read_wad2m_month(f.wad2m_file, year, imonth + 1, _wli, _wlo, mlat, mlon, clamp_to_range=False)
            _fdyn = None
            if _wdyn is not None and _fobs is None and not _learned_now:
                _so = state.soil
                _W = _wa.root_zone_water(np.asarray(_so.wsoi), np.asarray(_so.poros), np.asarray(_so.hsoi), np.asarray(_so.tsoi))
                _fdyn = _wa.dynamic_fraction(ch4_clim[imonth], _W, _wdyn[0][imonth], _wdyn[1][imonth], _wdyn[4], _wdyn[2], _wdyn[3],
                                             frozen=np.asarray(_so.tsoi)[:, 0] <= _wa.T_FREEZE)
            state, key, _dd = run_month_gswp3(
                state, consts, params, reader,
                gswp3_latindex, gswp3_lonindex, np.array(lat_rad),
                forcing_year, imonth, key, t.dtime, leap, grid.nband, nsoilcall,
                f_inund=(np.ones(len(lat_land_deg), np.float32) if _learned_now else
                         (_fobs if _fobs is not None else (_fdyn if _fdyn is not None else (None if ch4_clim is None else ch4_clim[imonth])))),
                methane_params=ch4_params,
                methane_hourly=bool(getattr(p, 'methane_hourly', False)),
                soil_means=bool(getattr(out, 'soil_means', False)) or _lw is not None,
            )
            _soilm = _dd[3] if ((getattr(out, 'soil_means', False) or _lw is not None) and len(_dd) > 3) else None
            _flearn = None
            if _lw is not None:
                _feats = _fbld.step(np.asarray(_soilm['tsoi_mean']), np.asarray(_soilm['wsoi_mean']), np.asarray(_soilm['wisoi_mean']), np.asarray(_soilm['tg_mean']),
                                    np.asarray(state.summary.ayprcp - _lw_before[0]), np.asarray(state.summary.ayaet - _lw_before[1]),
                                    np.asarray((state.summary.aytrunof - _lw_before[2]) + (state.summary.aytdrain - _lw_before[3])))
                if _learned_now:
                    _flearn = _lw.fraction(imonth, _feats, ch4_clim[imonth])
                    _wl = state.wetland; _a0 = jnp.zeros_like(_wl.annual) if _lw_before[4] is None else _lw_before[4]
                    _fj = jnp.asarray(_flearn, jnp.float32)
                    state = eqx.tree_at(lambda s: s.wetland, state, type(_wl)(flux=_wl.flux * _fj, annual=_a0 + (_wl.annual - _a0) * _fj, zwt=_wl.zwt))

            # Write monthly output
            if out.imonthout > 0:
                monthly_fluxes = {
                    'amprcp':   (state.summary.ayprcp   - prcp_before, 'mm',      'Monthly precipitation'),
                    'amaet':    (state.summary.ayaet    - aet_before,  'mm',      'Monthly AET'),
                    **({'finund': (jnp.asarray(_flearn, jnp.float32), '1', 'Inundated fraction of the month (learned: WAD2M climatology + learned anomaly)')} if _flearn is not None else
                       ({'finund': (jnp.asarray(_fobs, jnp.float32), '1', 'Inundated fraction of the month (observed WAD2M field)')} if _fobs is not None else
                       ({'finund': (jnp.asarray(_fdyn, jnp.float32), '1', 'Inundated fraction of the month (dynamic wetland area)')} if _fdyn is not None else {}))),
                    'amnpptot': (state.summary.aynpptot - npp_before, 'kgC/m2', 'Monthly NPP'),
                    # NEE is only formed at year end, so a monthly value has to
                    # be built from its two parts rather than differenced.
                    'amneetot': ((state.summary.aynpptot - npp_before)
                                 - (state.summary.ayco2mic - rh_before),
                                 'kgC/m2', 'Monthly NEE (positive = sink)'),
                    'amgpptot': (state.summary.aygpptot  - gpp_before, 'kgC/m2', 'Monthly GPP'),
                    'amco2mic': (state.summary.ayco2mic  - rh_before,  'kgC/m2', 'Monthly Rh'),
                    'amtrunof': (state.summary.aytrunof  - runof_before, 'mm',   'Monthly runoff'),
                    'amtdrain': (state.summary.aytdrain  - drain_before, 'mm',   'Monthly drainage'),
                    **({'tsoi_mean': (_soilm['tsoi_mean'], 'K', 'Soil temperature by layer, month mean over every time step'),
                        'wsoi_mean': (_soilm['wsoi_mean'], 'fraction of pore space', 'Liquid soil water by layer, month mean over every time step'),
                        'wisoi_mean': (_soilm['wisoi_mean'], 'fraction of pore space', 'Soil ice by layer, month mean over every time step'),
                        'tg_mean': (_soilm['tg_mean'], 'K', 'Soil surface temperature, month mean over every time step')} if _soilm is not None else {}),
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
        from ibis.timestepping.loops_gswp3 import climate_stats_from_monthly
        tcthis_np, twthis_np, gdd0this_np, gdd5this_np = (
            climate_stats_from_monthly(np.stack(monthly_tmean, axis=1), leap))

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
            exist_mask=_exist_mask_for(year), larch_gdd_fix=_larch_fix,
            slot_parent=params.slot_parent, slot_gate=params.slot_gate, slot_typed=params.slot_typed,
        )
        state = eqx.tree_at(
            lambda s: s.vegetation.exist, state, exist_new,
        )

        # Annual vegetation update: NPP → allocation → biomass → LAI →
        # litterfall, plus disturbance and PFT reclassification when dynamic.
        #
        # This used to call `biomass_update` unconditionally and then `dynaveg`
        # as well when isimveg > 0. They perform the SAME allocation from the
        # SAME uncleared `summary.aynpp`, so every dynamic-vegetation year
        # advanced the biomass pools twice -- measured at 1.42x the correct
        # annual change. Call one or the other, never both.
        # NEE = total NPP - Rh, formed once at year end from the two annual
        # accumulators, as F77 does (stats.f:1200). POSITIVE = carbon sink,
        # which is the sign dynaveg's `ayneetot - cdisturb` assumes.
        state = eqx.tree_at(
            lambda s: s.summary.ayneetot, state,
            state.summary.aynpptot - state.summary.ayco2mic,
        )

        from ibis.physics.vegetation_dynamics import biomass_update, dynaveg
        from ibis.state.stress_accum import tree_survival, tree_cover_max, reset_stress_accum
        from ibis.state.et_parts import reset_et_parts
        if p.isimveg > 0:
            _veg_diag = {}
            new_veg, new_carbon, new_summary = dynaveg(
                state.vegetation, state.carbon, state.summary,
                pft_params, p.isimfire, consts.epsilon,
                fire_scheme=p.fire_scheme, fire_params=params.fire,         # the yaml 'fire' section (CALIBRATION.md 59)
                tree_surv=tree_survival(state, pft_params),
                tree_cover_max=tree_cover_max(state, pft_params),
                cropfrac=params.cropfrac, crop_params=params.crop, diag=_veg_diag,
            )
        else:
            new_veg, new_carbon, new_summary = biomass_update(
                state.vegetation, state.carbon, state.summary,
                pft_params, consts.epsilon,
            )
            logger.info("  biomass_update: LAI and litterfall from NPP "
                        "(static vegetation, no disturbance)")
        state = eqx.tree_at(
            lambda s: (s.vegetation, s.carbon, s.summary),
            state, (new_veg, new_carbon, new_summary),
        )

        # Overwrite the model's grown LAI with the observed field.
        if lai_obs is not None:
            state = apply_prescribed_lai(state, lai_obs, pft_params.specla,
                                         consts.epsilon)
        if p.isimveg > 0:
            # `p.fire_scheme` used to be read nowhere: the call omitted it, so
            # every run took the legacy scheme no matter what the config said.
            # Same class of defect as the dead `firefac` (docs/FIRE.md).
            logger.info("  dynaveg: allocation, disturbance and PFT "
                        "reclassification (fire_scheme=%s)", p.fire_scheme)

            # Permafrost diagnostics. Active-layer only -- the 4 m soil column
            # cannot hold deep thermal memory; see loops_gswp3 for why.
            if getattr(p, "enable_permafrost", False):
                from ibis.bgc.permafrost import (create_permafrost_state,
                                                 default_permafrost_params,
                                                 permafrost_step)
                pf = state.permafrost
                if pf is None:
                    pf = create_permafrost_state(state.soil.tsoi.shape[0])
                pf = permafrost_step(pf, state.soil.tsoi, consts.tmelt,
                                     state.vegetation.gdd0,
                                     default_permafrost_params())
                state = eqx.tree_at(lambda s: s.permafrost, state, pf,
                                    is_leaf=lambda x: x is None)
                logger.info("  permafrost: active layer %.2f m mean, "
                            "frozen ground %.1f%% of cells",
                            float(jnp.mean(pf.active_layer)),
                            float(jnp.mean(pf.frozen_ground)) * 100.0)

            # `enable_fire_emissions` was declared in RunConfig and read
            # nowhere, so the whole F5 module -- Andreae factors, CO/CH4/N2O/
            # NOx -- was unreachable from a production run.
            # Note `veg.disturbf` falls back to IBIS's constant 0.005 when
            # isimfire=0, which is background disturbance, not fire -- so
            # reporting emissions then would be meaningless.
            if p.enable_fire_emissions and p.isimfire == 1:
                from ibis.bgc import fire_emissions_from_state
                fe = fire_emissions_from_state(state.vegetation, state.carbon)
                logger.info(
                    "  fire emissions: burned %.4f%% of land, %.4f kgC/m2, "
                    "CH4 %.4g g/m2, CO %.4g g/m2",
                    float(jnp.mean(fe.burned_fraction)) * 100.0,
                    float(jnp.mean(fe.carbon_burned)),
                    float(jnp.mean(fe.ch4)), float(jnp.mean(fe.co)),
                )

        elapsed = time.time() - t0
        logger.info("Year %d complete in %.1f s", year, elapsed)

        # Write yearly output (before resetting accumulators)
        logger.info("  Accumulators: ayprcp_max=%.1f mm  ayaet_max=%.1f mm  "
                    "aynpptot_max=%.4f kgC/m2  ayneetot_max=%.4f kgC/m2",
                    float(jnp.max(state.summary.ayprcp)),
                    float(jnp.max(state.summary.ayaet)),
                    float(jnp.max(state.summary.aynpptot)),
                    float(jnp.max(state.summary.ayneetot)))
        if out.iyearout > 0 and (iyear_offset + 1) % out.iyearout == 0:
            _extra = {}
            if p.isimveg > 0 and params.cropfrac is not None and _veg_diag:
                _extra = {"ayharvest": (_veg_diag["harvest"], "kgC/m2/yr", "Crop harvest removed from the cropland share"),
                          "ayclearing": (_veg_diag["clearing"], "kgC/m2/yr", "Tree carbon cleared to keep trees off the cropland share"),
                          "aycropshrub": (_veg_diag.get("crop_shrub_npp", jnp.zeros_like(_veg_diag["harvest"])), "kgC/m2/yr", "Shrub NPP on the cropland share, removed with the harvest (CALIBRATION.md 63)"),
                          "aycdisturb": (state.carbon.cdisturb, "kgC/m2/yr", "Carbon removed from the ecosystem this year: fire combustion, background disturbance, harvest and the cropland shrub NPP (NEE = NPP - Rh - this)")}
                logger.info("  cropland: harvest %.4f, clearing %.4f kgC/m2 (land-point means)", float(jnp.mean(_veg_diag["harvest"])), float(jnp.mean(_veg_diag["clearing"])))
            # diagnostics of the canopy structure (CALIBRATION.md 47): the tree gate's input and the NPP of each canopy
            if state.stress is not None:
                _extra["aystressu"] = (state.stress.aystressu / jnp.maximum(state.stress.nday, 1.0), "1",
                                       "Annual-mean upper-canopy soil-water stress factor (input of the tree-establishment gate)")
                from ibis.state.stress_accum import smoothed_stress as _smoothed_stress
                _extra["aystressu_smooth"] = (_smoothed_stress(state, float(getattr(pft_params, 'tree_cover_years', 5.0))), "1",
                                              "Multi-year (smoothed over tree_cover_years) upper-canopy stress this year: the cover limit's input (CALIBRATION.md 71)")
                _extra["stress_u_smooth_prev"] = (state.stress.u_smooth, "1",
                                                  "Smoothed upper-canopy stress as stored at the previous year end (0 = not yet seeded)")
            _extra["aynppu"] = (jnp.sum(state.summary.aynpp[:, :8], axis=1), "kgC/m2/yr", "Annual NPP of the upper canopy (PFTs 1-8)")
            _extra["aynppl"] = (jnp.sum(state.summary.aynpp[:, 8:], axis=1), "kgC/m2/yr", "Annual NPP of the lower canopy (PFTs 9-12)")
            path = writer.write_yearly(state, year, extra=_extra)
            logger.info("Wrote yearly output: %s", path)

        # Reset annual accumulators for the next year
        state = eqx.tree_at(
            lambda s: (s.summary.aynpp, s.summary.ayanpp, s.summary.ayanpptot,
                       s.summary.aynpptot, s.summary.ayneetot,
                       s.summary.ayprcp, s.summary.ayaet,
                       s.summary.aygpptot, s.summary.ayco2mic,
                       s.summary.ayco2root,
                       s.summary.aytrunof, s.summary.aytdrain),
            state,
            (jnp.zeros_like(state.summary.aynpp),
             jnp.zeros_like(state.summary.ayanpp),
             jnp.zeros_like(state.summary.ayanpptot),
             jnp.zeros_like(state.summary.aynpptot),
             jnp.zeros_like(state.summary.ayneetot),
             jnp.zeros_like(state.summary.ayprcp),
             jnp.zeros_like(state.summary.ayaet),
             jnp.zeros_like(state.summary.aygpptot),
             jnp.zeros_like(state.summary.ayco2mic),
             jnp.zeros_like(state.summary.ayco2root),
             jnp.zeros_like(state.summary.aytrunof),
             jnp.zeros_like(state.summary.aytdrain)),
        )
        state = reset_stress_accum(state, float(getattr(pft_params, 'tree_cover_years', 5.0)))   # after dynaveg consumed this year's mean stress; stores the smoothed stress
        state = reset_et_parts(state)
        if state.wetland is not None:        # the annual CH4 sum was written above; it accumulated across years until 2026-10-09 (CALIBRATION.md 39)
            from ibis.bgc.methane import reset_annual as _reset_ch4
            state = eqx.tree_at(lambda s: s.wetland, state, _reset_ch4(state.wetland))

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
