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
    #: Last years of an accelerated spin-up (soilcspin=1) run at 1x, so the fast pools settle before the transient
    #: (F77 main.f:353-391 ramps 40x down and ends with ~50 years at 1x; without it litter dropped 115 -> 103 PgC in
    #: 1901-05 of s31f, CALIBRATION.md 60). 0 = no tail (runs before 2026-10-26).
    soilcspin_tail_years: int = 50

    #: Fire scheme selector, read by run_global when isimveg > 0. Mirrors
    #: RunConfig.fire_scheme (docs/FIRE.md). This and enable_fire_emissions
    #: were READ by scripts/run_global.py but never declared here, so every
    #: dynamic-vegetation run raised AttributeError on p.fire_scheme -- which
    #: is why the 200-year spin-ups went through verify/ rather than the
    #: production runner.
    fire_scheme: str = "seasonlength"
    #: Clear the annual fire counters (nytimes, nfirestep, sdry and the VPD pair) at the start of each year, as F77
    #: does (stats.f:956). Default True since 2026-10-25 (CALIBRATION.md 59); the production runner never cleared
    #: them before, so every year's fire read the counters accumulated since the first spin-up year. False
    #: reproduces runs made before that date.
    fire_annual_reset: bool = True
    #: Route fire-killed carbon to emissions rather than litter. Requires
    #: isimfire == 1.
    enable_fire_emissions: bool = False
    #: Wetland CH4 diagnostics (docs/METHANE.md; CALIBRATION.md 38): the runner loads the WAD2M climatology
    #: (`forcing.wad2m_file`) and runs the methane module; written as aych4 / amch4.
    enable_methane: bool = False
    #: PFT 8 (larch) existence: use the degree-day limit the F77 comment intends instead of the
    #: temperature limit the F77 code compares against (ibis.physics.climate_analysis.existence).
    larch_gdd_fix: bool = False
    #: IBIS 3.1 Köppen-zone plant functional types (ibis.io.pft_types, CALIBRATION.md 45.1): each cell's zone,
    #: from forcing.koppen_file by period, maps its 12 positions to the 40 types; per-type parameters from the
    #: params-file section 'types'. Supersedes the Köppen existence mask. False = the 12 IBIS PFTs.
    koppen_types: bool = False
    #: Dynamic wetland (inundated) area from the model's own root-zone water (docs/METHANE.md 7; Min's point 4):
    #: f = clip(f_clim (1 + beta A), f_min, f_max) with A the standardised storage anomaly against the climatology of
    #: forcing.wetland_clim_file and beta per band from the parameter file's `wetland_area:` section. False: the WAD2M
    #: climatology, as before.
    wetland_dynamic: bool = False
    #: Mix organic-soil hydraulic and thermal properties by the HWSD2 organic-carbon fraction
    #: (Lawrence & Slater 2008); needs orgcpct and bulkdens in soil_file (scripts/hwsd2_organic_to_nc.py).
    soil_organic: bool | str = False
    #: Organic end-member when soil_organic is on: 'ls2008' (default), 'fibric', 'hemic' or 'sapric'.
    soil_organic_end_member: str = "ls2008"
    #: Prescribe leaf area index from MODIS instead of growing it. This is the
    #: configuration behind the published IBIS GPP of 107.5 PgC/yr (Liu et al.
    #: 2014). Carbon totals from such a run are diagnostic, not a closed
    #: budget -- see ibis/physics/prescribed_lai.py.
    prescribe_lai: bool = False
    et_components: bool = False   # accumulate transpiration / interception / soil evap / snow sublimation (CALIBRATION.md 15)
    #: Seed leaf area from MODIS ONCE at the start of the run, then let
    #: `dynaveg` evolve it. Distinct from `prescribe_lai`, which re-imposes
    #: the observed field every year.
    #:
    #: This exists because vegetation in IBIS is bistable in marginal
    #: climates. `both_fix` was a cold start, so leaf area had to bootstrap
    #: from the `xminlai` floor; verify/calibration/arctic_seed_test.py showed
    #: that sub-Arctic cells seeded to MODIS LAI stay there for 20 years
    #: (FI-Sod 1.10 -> 1.89, FI-Lom 0.93 -> 1.88) while their cold-start state
    #: sat near the floor. Mid-latitude cells instead climb back to their own
    #: attractor, and true tundra collapses back to the floor -- so this flag
    #: changes the answer only where the model has two stable branches.
    seed_lai: bool = False
    #: Follow the historical CO2 trajectory instead of holding `co2init`.
    #: `co2init` is applied once at state creation, so every run in this
    #: project -- including Min's `calibration_transient` -- held CO2 fixed
    #: while the meteorology varied. 1901-2010 CO2 rose ~296 -> 390 ppm and
    #: photosynthesis responds to it directly, so a historical run validated
    #: against towers needs this on. See ibis/io/co2.py for provenance.
    transient_co2: bool = False


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
    #: Cropland fraction per 0.5-degree cell (npz with "cropfrac" (360, 720), north to south; CALIBRATION.md 41).
    #: Empty string = no cropland treatment.
    cropfrac_file: str = ""
    #: Koppen-Geiger classes per 0.5-degree cell and period (npz from scripts/koppen_prep.py: "classes"
    #: (nperiod, 360, 720) uint8 north to south, "period_start", "period_end"; CALIBRATION.md 44).
    #: Empty string = no Koppen mask on PFT existence.
    koppen_file: str = ""
    #: Root-zone soil-water climatology (mu, sd per cell and month) of a reference hindcast, for wetland_dynamic
    #: (verify/methane/wetland_clim_prep.py).
    wetland_clim_file: str = ""
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
