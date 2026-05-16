#!/usr/bin/env python3
"""IBIS-JAX command-line runner.

Usage:
    python -m scripts.run_ibis config.yaml
    python scripts/run_ibis.py config.yaml

Reads configuration from a YAML file, loads input data, runs the
simulation, and writes output/restart files.
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

from ibis.io.config_reader import load_config, IBISConfig
from ibis.io.netcdf_io import read_input_data
from ibis.io.restart import save_state, load_state, restart_path
from ibis.io.diagnostics import DiagnosticWriter
from ibis.io.soil_init import compute_soil_properties
from ibis.io.veg_init import initialize_vegetation
from ibis.constants import PhysicalConstants
from ibis.dimensions import GridConfig
from ibis.state.model_state import IBISState, create_ibis_state
from ibis.timestepping.loops import (
    SimParams, default_sim_params, run_year,
)
from ibis.physics.weather import ClimateData


logger = logging.getLogger("ibis")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_climate_data(input_data: dict) -> ClimateData:
    """Convert raw input climate arrays to ClimateData.

    Maps the keys from read_input_data (temp, rh, prec, wspd, cld, wetd, trange)
    to ClimateData fields (xint, xinq, xinprec, xinwind, xincld, xinwet, xintrng).
    """
    clim = input_data["climate"]
    npoi = clim["temp"].shape[0]
    return ClimateData(
        xint=jnp.array(clim["temp"]),
        xinq=jnp.array(clim["rh"]),
        xinprec=jnp.array(clim["prec"]),
        xinwind=jnp.array(clim["wspd"]),
        xincld=jnp.array(clim["cld"]),
        xinwet=jnp.array(clim["wetd"]),
        xintrng=jnp.array(clim["trange"]),
        xintopo=jnp.zeros((npoi,)),
    )


def run_simulation(config: IBISConfig) -> None:
    """Run the full IBIS simulation from a configuration.

    Parameters
    ----------
    config : IBISConfig
        Loaded configuration.
    """
    consts = PhysicalConstants()
    g = config.grid
    t = config.time
    p = config.physics
    inp = config.input
    out = config.output

    # --- Read input data ---
    logger.info("Reading input data from %s ...", inp.input_dir)
    input_data = read_input_data(
        inp.input_dir, g.snorth, g.ssouth, g.swest, g.seast, g.xres, g.yres,
    )
    npoi = input_data["npoi"]
    nlon = input_data["nlon"]
    nlat = input_data["nlat"]
    lon = input_data["lon"]
    lat = input_data["lat"]
    lonindex = input_data["lonindex"]
    latindex = input_data["latindex"]

    logger.info(
        "Domain: %d x %d grid, %d land points", nlon, nlat, npoi,
    )

    # --- Build grid config ---
    grid = GridConfig(
        nlon=nlon, nlat=nlat, npoi=npoi,
        xres=g.xres, yres=g.yres,
    )

    # --- Build climate data ---
    climate = build_climate_data(input_data)

    # --- Create or load state ---
    if t.irestart > 0:
        rpath = restart_path(out.restart_dir, t.irestart)
        logger.info("Loading restart from %s", rpath)
        template = create_ibis_state(grid, p.co2init, p.o2init)
        state, meta = load_state(rpath, template)
        logger.info("Restart metadata: %s", meta)
    else:
        logger.info("Cold start")
        state = create_ibis_state(grid, p.co2init, p.o2init)

    # --- Populate soil static properties from input texture data ---
    logger.info("Computing soil properties from texture data ...")
    sand_np = np.array(input_data["sand"])
    clay_np = np.array(input_data["clay"])
    soil_props = compute_soil_properties(sand_np, clay_np)

    state = eqx.tree_at(
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

    # --- Initialize vegetation from vegtype map ---
    if t.irestart == 0:
        logger.info("Initializing vegetation from type map ...")
        vegtype_np = np.array(input_data["vegtype"])
        # Use warmest-month temperature from climatology for C3/C4 split
        tw_np = np.array(climate.xint).max(axis=1)  # approx warmest month
        veg_props = initialize_vegetation(vegtype_np, tw=tw_np)

        state = eqx.tree_at(
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

    # --- Build simulation parameters (with actual latitude) ---
    lat_rad = jnp.deg2rad(jnp.array(lat[latindex].astype(np.float32)))
    params = default_sim_params(npoi)
    params = eqx.tree_at(lambda p: p.latitude, params, lat_rad)

    # --- Diagnostics writer ---
    writer = DiagnosticWriter(
        out.output_dir, lon, lat, latindex, lonindex,
    )

    # --- Random key ---
    key = jax.random.PRNGKey(42)

    # --- Year loop ---
    for iyear_offset in range(t.nrun):
        year = t.iyear0 + iyear_offset
        t0 = time.time()
        logger.info("Year %d (%d/%d) ...", year, iyear_offset + 1, t.nrun)

        state, key = run_year(
            state, consts, params, climate, key,
            t.dtime, year, grid.nband,
        )

        elapsed = time.time() - t0
        logger.info("Year %d completed in %.1f s", year, elapsed)

        # Yearly output
        if out.iyearout > 0 and (iyear_offset + 1) % out.iyearout == 0:
            path = writer.write_yearly(state, year)
            logger.info("Wrote yearly output: %s", path)

        # Restart
        rpath = restart_path(out.restart_dir, year)
        save_state(state, rpath, metadata={"year": year})
        logger.info("Wrote restart: %s", rpath)

    logger.info("Simulation complete.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IBIS-JAX: Integrated Biosphere Simulator (JAX version)",
    )
    parser.add_argument(
        "config", type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--platform", type=str, default=None,
        choices=["cpu", "gpu", "tpu"],
        help="JAX platform (default: auto-detect)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.platform:
        jax.config.update("jax_platform_name", args.platform)

    logger.info("JAX platform: %s", jax.default_backend())
    logger.info("Loading config from %s", args.config)

    config = load_config(args.config)
    run_simulation(config)


if __name__ == "__main__":
    main()
