"""Biogeochemical components.

bvoc, nee, soil_gases and wetlands moved to _legacy_unwired/ on 2026-08-28 --
nothing consumed them and re-exporting here implied otherwise. See
_legacy_unwired/README.md and docs/AUDIT.md.
"""

from ibis.bgc.permafrost import (
    PermafrostState, PermafrostParams, create_permafrost_state,
    default_permafrost_params, permafrost_step,
    stefan_depth, diagnose_frozen_ground, thermokarst,
)
from ibis.bgc.fire_emissions import (
    FireEmissions, FireEmissionParams, create_fire_emissions, fire_emissions,
    fire_emissions_from_state,
)
