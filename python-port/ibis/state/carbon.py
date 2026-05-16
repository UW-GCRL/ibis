"""Carbon pool state variables.

Ported from mod_carbon_pools.f90.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class CarbonState(eqx.Module):
    """Litter and soil organic matter pools.

    All arrays are [npoi] (per grid point), except living biomass
    which is [npoi, npft] (in VegetationState).
    """

    # --- Leaf litter (metabolic, structural, lignin) ---
    clitlm: Float[Array, "npoi"]           # Metabolic leaf litter (kgC/m2)
    clitls: Float[Array, "npoi"]           # Structural leaf litter
    clitll: Float[Array, "npoi"]           # Lignin leaf litter

    # --- Fine-root litter ---
    clitrm: Float[Array, "npoi"]           # Metabolic root litter
    clitrs: Float[Array, "npoi"]           # Structural root litter
    clitrl: Float[Array, "npoi"]           # Lignin root litter

    # --- Woody litter ---
    clitwm: Float[Array, "npoi"]           # Metabolic wood litter
    clitws: Float[Array, "npoi"]           # Structural wood litter
    clitwl: Float[Array, "npoi"]           # Lignin wood litter

    # --- Soil organic matter ---
    csoipas: Float[Array, "npoi"]          # Passive humus (kgC/m2)
    csoislon: Float[Array, "npoi"]         # Slow non-protected humus
    csoislop: Float[Array, "npoi"]         # Slow protected humus

    # --- Decomposition modifiers ---
    decompl: Float[Array, "npoi"]          # Litter decomposition rate factor
    decomps: Float[Array, "npoi"]          # SOM decomposition rate factor

    # --- Litter-fall fluxes (annual, from vegetation dynamics) ---
    falll: Float[Array, "npoi"]            # Leaf litter fall (kgC/m2/yr)
    fallr: Float[Array, "npoi"]            # Root litter fall
    fallw: Float[Array, "npoi"]            # Wood litter fall

    # --- Disturbance ---
    cdisturb: Float[Array, "npoi"]         # C lost to fire/disturbance (kgC/m2/yr)

    # --- Microbial biomass ---
    totcmic: Float[Array, "npoi"]          # Total microbial biomass C (kgC/m2)

    # --- Aggregated diagnostics ---
    totlit: Float[Array, "npoi"]           # Total litter C (kgC/m2)
    totalit: Float[Array, "npoi"]          # Above-ground litter C (kgC/m2)
    totrlit: Float[Array, "npoi"]          # Root litter C (kgC/m2)
    totcsoi: Float[Array, "npoi"]          # Total soil organic C (kgC/m2)
    totfall: Float[Array, "npoi"]          # Total litterfall (kgC/m2/yr)


def create_carbon_state(npoi: int) -> CarbonState:
    """Create a zero-initialised CarbonState."""
    z = jnp.zeros((npoi,))
    return CarbonState(
        clitlm=z, clitls=z, clitll=z,
        clitrm=z, clitrs=z, clitrl=z,
        clitwm=z, clitws=z, clitwl=z,
        csoipas=z, csoislon=z, csoislop=z,
        decompl=z, decomps=z,
        falll=z, fallr=z, fallw=z,
        cdisturb=z,
        totcmic=z,
        totlit=z, totalit=z, totrlit=z,
        totcsoi=z, totfall=z,
    )
