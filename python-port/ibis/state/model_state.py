"""Top-level IBIS model state.

The single pytree threaded through the entire simulation.
Each physics function takes IBISState in and returns IBISState out.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx

from ibis.state.soil import SoilState, create_soil_state
from ibis.state.snow import SnowState, create_snow_state
from ibis.state.canopy import CanopyState, create_canopy_state
from ibis.state.atmosphere import AtmState, create_atm_state
from ibis.state.vegetation import VegetationState, create_vegetation_state
from ibis.state.carbon import CarbonState, create_carbon_state
from ibis.state.hydrology import HydrologyDiag, create_hydrology_diag
from ibis.state.summary import SummaryState, create_summary_state
from ibis.state.stress_accum import create_stress_accum
from ibis.state.et_parts import create_et_parts
from ibis.state.seasonal import create_seasonal
from ibis.dimensions import GridConfig


class IBISState(eqx.Module):
    """Complete model state -- the single pytree for the entire simulation.

    Optional BGC components are None when disabled, which makes them
    invisible to JAX tree operations (equinox handles Optional fields).
    """

    soil: SoilState
    snow: SnowState
    canopy: CanopyState
    atm: AtmState
    vegetation: VegetationState
    carbon: CarbonState
    hydrology: HydrologyDiag
    summary: SummaryState

    # --- Optional new BGC components (None = disabled) ---
    # These will be defined in ibis.bgc.* when implemented.
    # For now, typed as Optional[eqx.Module].
    permafrost: Optional[eqx.Module] = None
    wetland: Optional[eqx.Module] = None
    fire: Optional[eqx.Module] = None
    bvoc: Optional[eqx.Module] = None
    #: Annual-mean soil-water stress (docs/CALIBRATION.md 14). None unless a
    #: run enables the water-limited tree establishment gate.
    stress: Optional[eqx.Module] = None
    #: Annual ET components (docs/CALIBRATION.md 15). None unless a run asks
    #: for them (`physics: et_components: true`); implies `stress`.
    etparts: Optional[eqx.Module] = None
    #: Seasonal states (docs/CALIBRATION.md 21-22): photosynthetic acclimation and photoperiod-gated
    #: senescence. None unless a run enables either formulation; implies `stress` and `etparts`, so the
    #: optional trailing modules stay one nested chain for load_state (258 / 261 / 265 / 268 leaves).
    seasonal: Optional[eqx.Module] = None


def create_ibis_state(grid: GridConfig,
                      co2: float = 350.0e-6,
                      o2: float = 0.209,
                      with_stress: bool = False,
                      with_etparts: bool = False,
                      with_seasonal: bool = False) -> IBISState:
    """Create a cold-start IBISState with default initial values.

    Parameters
    ----------
    grid : GridConfig
        Grid dimensions (npoi, nsoilay, nsnolay, npft, nband).
    co2 : float
        Initial CO2 concentration (mol/mol).
    o2 : float
        O2 concentration (mol/mol).
    """
    return IBISState(
        soil=create_soil_state(grid.npoi, grid.nsoilay, grid.nbedrock),
        snow=create_snow_state(grid.npoi, grid.nsnolay),
        canopy=create_canopy_state(grid.npoi),
        atm=create_atm_state(grid.npoi, grid.nband, co2, o2),
        vegetation=create_vegetation_state(grid.npoi, grid.npft, grid.nsoilay),
        carbon=create_carbon_state(grid.npoi),
        hydrology=create_hydrology_diag(grid.npoi),
        summary=create_summary_state(grid.npoi, grid.npft),
        stress=(create_stress_accum(grid.npoi) if (with_stress or with_etparts or with_seasonal) else None),
        etparts=(create_et_parts(grid.npoi) if (with_etparts or with_seasonal) else None),
        seasonal=(create_seasonal(grid.npoi) if with_seasonal else None),
    )
