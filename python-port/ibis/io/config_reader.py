"""YAML configuration reader for IBIS-JAX.

Replaces the Fortran ibis.infile positional format with a structured
YAML configuration file. All simulation parameters are organized
into logical groups.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class GridSection:
    """Grid domain specification."""
    snorth: float = 32.5
    ssouth: float = 7.5
    swest: float = 68.5
    seast: float = 95.5
    xres: float = 0.5
    yres: float = 0.5
    #: Thermal-only bedrock layers below the 4 m soil column. 0 keeps the
    #: Fortran configuration; 3 gives the 12 m column. See ibis/physics/bedrock.py.
    nbedrock: int = 0


@dataclass
class TimeSection:
    """Temporal configuration."""
    iyear0: int = 1961
    nrun: int = 10
    dtime: float = 3600.0
    irestart: int = 0


@dataclass
class PhysicsSection:
    """Physics switches."""
    isimveg: int = 0       # 0=static, 1=dynamic, 2=dynamic cold start
    isimfire: int = 0      # 0=fixed, 1=dynamic
    isimco2: int = 0       # 0=fixed, 1=ramped
    co2init: float = 350.0e-6
    o2init: float = 0.209
    soilcspin: int = 1     # 0=no spinup, 1=acceleration (see nsoilcall)
    #: soilbgc calls per simulated day when soilcspin=1. Each call advances the
    #: pools one day AND applies one day of litterfall, so it is a consistent
    #: clock acceleration: nsoilcall=10 buys 10 soil-years per simulated year.
    #: Needed for a cold start -- the passive pool turns over on century to
    #: millennium timescales, so 200 years at 1x does not equilibrate it. Costs
    #: GPU kernel size and compile time proportionally.
    nsoilcall: int = 1

    #: Fire scheme selector, read by run_global when isimveg > 0. Mirrors
    #: RunConfig.fire_scheme (docs/FIRE.md). This and enable_fire_emissions
    #: were READ by scripts/run_global.py but never declared here, so every
    #: dynamic-vegetation run raised AttributeError on p.fire_scheme -- which
    #: is why the 200-year spin-ups went through verify/ rather than the
    #: production runner.
    fire_scheme: str = "seasonlength"
    #: Route fire-killed carbon to emissions rather than litter. Requires
    #: isimfire == 1.
    enable_fire_emissions: bool = False
    #: Prescribe leaf area index from MODIS instead of growing it. This is the
    #: configuration behind the published IBIS GPP of 107.5 PgC/yr (Liu et al.
    #: 2014). Carbon totals from such a run are diagnostic, not a closed
    #: budget -- see ibis/physics/prescribed_lai.py.
    prescribe_lai: bool = False


@dataclass
class OutputSection:
    """Output configuration."""
    iyearout: int = 1
    imonthout: int = 1
    idailyout: int = 0
    output_dir: str = "output"
    restart_dir: str = "restart"
    env_snapshot_dir: str = ""       # Directory for calibration env snapshots
    env_snapshot_start: int = 0      # Start year for saving snapshots (0=disabled)


@dataclass
class InputSection:
    """Input data paths."""
    input_dir: str = "input"
    param_dir: str = "."
    iyranom: int = 9999    # Year to begin reading anomalies
    nanom: int = 98
    iyrdaily: int = 9999   # Year to begin reading daily forcing


@dataclass
class ForcingSection:
    """Forcing data configuration.

    forcing_type : "climatology" or "gswp3"
        climatology — use monthly climatology files + stochastic weather generator
        gswp3       — use transient GSWP3 daily observations

    gswp3_dir : str
        Directory containing GSWP3 decade .nc4 files.
        e.g. /shared/nas2/Min/GSWP3

    vegtype_dir : str
        Directory containing ESACCI yearly vegtype files (vegtype_frac_YYYY.nc).
        e.g. /shared/nas2/Min/ESACCI_LC/ibis_vegtype
        Empty string disables transient vegetation maps.

    soil_file : str
        Path to soita.nc (combined sand+clay file).
        e.g. /shared/nas2/Min/HWSD/soita.nc
        Empty string falls back to input_dir/soita.nc.

    land_mask_file : str
        Path to surta.nc land mask.
        Empty string = derive land mask from GSWP3 tas coverage at runtime.
    """
    forcing_type: str = "climatology"
    gswp3_dir: str = ""
    vegtype_dir: str = ""
    #: WAD2M inundated-fraction file, read only when RunConfig.enable_methane
    #: is set. See docs/METHANE.md.
    wad2m_file: str = ("/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/"
                       "WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc")
    soil_file: str = ""
    land_mask_file: str = ""
    # Spinup: repeat [spinup_year_start, spinup_year_end] cyclically for nrun years.
    # Set spinup_year_start > 0 to enable. Forcing years outside GSWP3 range are
    # mapped back into [spinup_year_start, spinup_year_end] automatically.
    spinup_year_start: int = 0   # 0 = disabled
    spinup_year_end: int = 0


@dataclass
class IBISConfig:
    """Complete IBIS-JAX configuration."""
    grid: GridSection = field(default_factory=GridSection)
    time: TimeSection = field(default_factory=TimeSection)
    physics: PhysicsSection = field(default_factory=PhysicsSection)
    output: OutputSection = field(default_factory=OutputSection)
    input: InputSection = field(default_factory=InputSection)
    forcing: ForcingSection = field(default_factory=ForcingSection)
    # Optional path to a physics parameters YAML file.
    # If empty, all physics parameters use hardcoded defaults.
    params_file: str = ""


def load_config(path: str) -> IBISConfig:
    """Load IBIS configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    IBISConfig
        Populated configuration object.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    config = IBISConfig()

    if "grid" in data:
        config.grid = GridSection(**data["grid"])
    if "time" in data:
        config.time = TimeSection(**data["time"])
    if "physics" in data:
        config.physics = PhysicsSection(**data["physics"])
    if "output" in data:
        config.output = OutputSection(**data["output"])
    if "input" in data:
        config.input = InputSection(**data["input"])
    if "forcing" in data:
        config.forcing = ForcingSection(**data["forcing"])
    if "params_file" in data:
        config.params_file = str(data["params_file"])

    return config


def save_config(config: IBISConfig, path: str) -> None:
    """Save IBIS configuration to a YAML file.

    Parameters
    ----------
    config : IBISConfig
        Configuration to save.
    path : str
        Output path.
    """
    from dataclasses import asdict
    data = asdict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def default_config() -> IBISConfig:
    """Return the default IBIS configuration."""
    return IBISConfig()
