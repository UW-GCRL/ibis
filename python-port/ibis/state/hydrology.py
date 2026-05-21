"""Hydrology diagnostic fluxes.

Ported from mod_hydrology.f90.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class HydrologyDiag(eqx.Module):
    """Hydrological fluxes (diagnostic, recomputed each timestep)."""

    ginvap: Float[Array, "npoi"]           # Interception evaporation (kg/m2/s)
    gsuvap: Float[Array, "npoi"]           # Surface evaporation (kg/m2/s)
    gtrans: Float[Array, "npoi"]           # Total transpiration (kg/m2/s)
    gtransu: Float[Array, "npoi"]          # Upper canopy transpiration
    gtransl: Float[Array, "npoi"]          # Lower canopy transpiration
    grunof: Float[Array, "npoi"]           # Surface runoff (kg/m2/s)
    gdrain: Float[Array, "npoi"]           # Drainage from lowest layer (kg/m2/s)
    gadjust: Float[Array, "npoi"]          # Water adjustment flux (kg/m2/s)
    wtot: Float[Array, "npoi"]             # Total stored water (kg/m2)
    # Diagnostic exposures (added for day-84 shock investigation)
    raing_avg: Float[Array, "npoi"]        # Areal-avg rain reaching soil (kg/m2/s)
    traing_avg: Float[Array, "npoi"]       # Areal-avg rain temperature (K)
    fhtop_soil: Float[Array, "npoi"]       # Heat flux into top of soil (W/m2)
    rainl: Float[Array, "npoi"]            # Rain to lower veg / snow (kg/m2/s, per-area)
    snowl: Float[Array, "npoi"]            # Snow to lower veg / snow (kg/m2/s, per-area)
    trainl: Float[Array, "npoi"]           # Rain temp at lower veg / snow input (K)
    tsnowl: Float[Array, "npoi"]           # Snow temp at lower veg / snow input (K)
    sflo_bot: Float[Array, "npoi"]         # Snow-bottom heat flux into soil (W/m2)
    fhtop_snow: Float[Array, "npoi"]       # Heat flux into top of snow (W/m2)


def create_hydrology_diag(npoi: int) -> HydrologyDiag:
    """Create a zero-initialised HydrologyDiag."""
    z = jnp.zeros((npoi,))
    return HydrologyDiag(
        ginvap=z, gsuvap=z, gtrans=z, gtransu=z, gtransl=z,
        grunof=z, gdrain=z, gadjust=z, wtot=z,
        raing_avg=z, traing_avg=z, fhtop_soil=z,
        rainl=z, snowl=z, trainl=z, tsnowl=z, sflo_bot=z, fhtop_snow=z,
    )
