"""New biogeochemical components (permafrost, wetland CH4, fire, BVOC, etc.)."""

from ibis.bgc.wetlands import (
    WetlandState, WetlandParams, create_wetland_state,
    default_wetland_params, wetland_step,
    wetland_fraction, ch4_production, ch4_oxidation, ch4_transport,
)
from ibis.bgc.permafrost import (
    PermafrostState, PermafrostParams, create_permafrost_state,
    default_permafrost_params, permafrost_step,
    stefan_depth, diagnose_frozen_ground, thermokarst,
)
from ibis.bgc.fire_emissions import (
    FireEmissionState, create_fire_emission_state, fire_emissions,
)
from ibis.bgc.soil_gases import (
    SoilGasState, SoilGasParams, create_soil_gas_state,
    default_soil_gas_params, soil_gas_step,
)
from ibis.bgc.bvoc import (
    BVOCState, create_bvoc_state, bvoc_emissions,
    gamma_t_isop, gamma_par_isop, gamma_t_mono,
)
from ibis.bgc.nee import (
    NEEState, create_nee_state, compute_nee,
)
