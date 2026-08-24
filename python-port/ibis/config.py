"""Run configuration for IBIS-JAX.

Loaded from a YAML file or constructed programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ibis.dimensions import GridConfig, SOUTH_ASIA


@dataclass(frozen=True)
class RunConfig:
    """Complete run configuration."""

    # Grid
    grid: GridConfig = field(default_factory=lambda: SOUTH_ASIA)

    # Time
    dtime: float = 1800.0       # Timestep in seconds
    iyear0: int = 1980          # First simulation year
    nrun: int = 10              # Number of years to run

    # Mode flags
    irestart: bool = False
    isimveg: int = 1            # 0 = static vegetation, 1 = dynamic
    isimfire: int = 0           # 0 = no fire, 1 = fire enabled
    isimco2: int = 0            # 0 = fixed CO2, 1 = varying

    # Fire scheme selector; see ibis.physics.fire.AVAILABLE_SCHEMES and
    # docs/FIRE.md. "legacy" reproduces IBIS 2.6b4 vegetation.f:591 exactly.
    #
    # The default was "legacy", on the stated grounds that the 46-yr F77<->JAX
    # lockstep depended on it. It does not: the lockstep runs at isimfire=0,
    # and `dynaveg` selects `jnp.where(isimfire == 1, _disturbf, 0.005)`, so
    # with fire off the scheme is numerically irrelevant. Since fire has never
    # been enabled in any run, defaulting to the better scheme costs nothing
    # and means anyone who sets isimfire=1 gets rank +0.630 rather than +0.315.
    fire_scheme: str = "seasonlength"

    # Atmospheric concentrations
    co2init: float = 350.0e-6   # Initial CO2 (mol/mol)
    o2init: float = 0.209       # O2 concentration (mol/mol)

    # New biogeochemical component flags
    enable_permafrost: bool = False
    enable_wetlands: bool = False
    enable_fire_emissions: bool = False
    enable_soil_n2o: bool = False
    enable_bvoc: bool = False

    # ML component paths (None = use physics-only)
    ml_stomata: Optional[str] = None
    ml_canopy: Optional[str] = None

    # I/O paths
    input_dir: str = "input"
    output_dir: str = "output"
    restart_dir: str = "restart"


def load_config(path: str | Path) -> RunConfig:
    """Load run configuration from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Extract grid config if present
    grid_raw = raw.pop("grid", {})
    grid = GridConfig(**grid_raw) if grid_raw else SOUTH_ASIA

    return RunConfig(grid=grid, **raw)
