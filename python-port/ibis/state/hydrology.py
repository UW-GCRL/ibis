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


def create_hydrology_diag(npoi: int) -> HydrologyDiag:
    """Create a zero-initialised HydrologyDiag."""
    z = jnp.zeros((npoi,))
    return HydrologyDiag(
        ginvap=z, gsuvap=z, gtrans=z, gtransu=z, gtransl=z,
        grunof=z, gdrain=z, gadjust=z, wtot=z,
    )
