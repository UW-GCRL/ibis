"""Vegetation state variables.

Ported from mod_veg_dynamics.f90.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class VegetationState(eqx.Module):
    """Vegetation structure, phenology, and dynamics.

    Array shapes:
      npoi    = number of land grid points
      npft    = number of plant functional types (12)
      nsoilay = number of soil layers (for root distribution)
    """

    # --- Per-PFT biomass (prognostic) ---
    cbiol: Float[Array, "npoi npft"]       # Leaf C (kgC/m2)
    cbior: Float[Array, "npoi npft"]       # Fine-root C (kgC/m2)
    cbiow: Float[Array, "npoi npft"]       # Wood C (kgC/m2)
    plai: Float[Array, "npoi npft"]        # Plant LAI per PFT (m2/m2)
    frac: Float[Array, "npoi npft"]        # PFT area fraction
    biomass: Float[Array, "npoi npft"]     # Total biomass per PFT (kgC/m2)
    exist: Float[Array, "npoi npft"]       # PFT existence probability

    # --- Per-PFT fluxes ---
    tnpp: Float[Array, "npoi npft"]        # NPP per PFT (mol CO2/m2/s)
    tgpp: Float[Array, "npoi npft"]        # GPP per PFT (mol CO2/m2/s)

    # --- Canopy structure (2 canopy levels: lower=0, upper=1) ---
    lai: Float[Array, "npoi 2"]            # Single-sided LAI
    sai: Float[Array, "npoi 2"]            # Single-sided SAI
    ztop: Float[Array, "npoi 2"]           # Plant top height (m)
    zbot: Float[Array, "npoi 2"]           # Lowest branch height (m)
    fu: Float[Array, "npoi"]               # Upper canopy cover fraction
    fl: Float[Array, "npoi"]               # Lower canopy cover fraction

    # --- Root distribution ---
    froot: Float[Array, "nsoilay 2"]       # Root fraction per soil layer

    # --- Phenology accumulators (prognostic) ---
    tempu: Float[Array, "npoi"]            # Cold-phenology trigger, trees
    templ: Float[Array, "npoi"]            # Cold-phenology trigger, grasses
    dropu: Float[Array, "npoi"]            # Drought trigger, trees
    dropls: Float[Array, "npoi"]           # Drought trigger, shrubs
    dropl3: Float[Array, "npoi"]           # Drought trigger, C3 grass
    dropl4: Float[Array, "npoi"]           # Drought trigger, C4 grass
    agddu: Float[Array, "npoi"]            # GDD accumulator, upper canopy
    agddl: Float[Array, "npoi"]            # GDD accumulator, lower canopy

    # --- Climate accumulators (prognostic) ---
    tc: Float[Array, "npoi"]               # Coldest month temp (C)
    tw: Float[Array, "npoi"]               # Warmest month temp (C)
    gdd0: Float[Array, "npoi"]             # GDD above 0C
    gdd5: Float[Array, "npoi"]             # GDD above 5C
    tcmin: Float[Array, "npoi"]            # Coldest daily temp of year (C)

    # --- Aggregated diagnostics ---
    tnpptot: Float[Array, "npoi"]          # Total NPP (mol CO2/m2/s)
    tgpptot: Float[Array, "npoi"]          # Total GPP (mol CO2/m2/s)
    tco2mic: Float[Array, "npoi"]          # Microbial CO2 flux (mol CO2/m2/s)
    tco2root: Float[Array, "npoi"]         # Root respiration (mol CO2/m2/s)
    tneetot: Float[Array, "npoi"]          # NEE (kgC/m2/timestep)
    sapfrac: Float[Array, "npoi"]          # Sapwood fraction
    firefac: Float[Array, "npoi"]          # Fire dryness factor

    # --- Vegetation dynamics diagnostics ---
    totlaiu: Float[Array, "npoi"]          # Total upper canopy LAI
    totlail: Float[Array, "npoi"]          # Total lower canopy LAI
    totbiou: Float[Array, "npoi"]          # Total upper canopy biomass
    totbiol_agg: Float[Array, "npoi"]      # Total lower canopy biomass
    disturbf: Float[Array, "npoi"]         # Fire disturbance fraction
    disturbo: Float[Array, "npoi"]         # Other disturbance fraction
    vegtype0: Float[Array, "npoi"]         # Vegetation type classification
    tnmin: Float[Array, "npoi"]            # N mineralization rate (kgN/m2/s)

    # --- Current year climate tracking (for climanl2) ---
    gdd0this: Float[Array, "npoi"]         # This-year GDD above 0C
    gdd5this: Float[Array, "npoi"]         # This-year GDD above 5C
    tcthis: Float[Array, "npoi"]           # This-year coldest month temp (C)
    twthis: Float[Array, "npoi"]           # This-year warmest month temp (C)

    # --- PFT parameters (static, from params.veg) ---
    specla: Float[Array, "npft"]           # Specific leaf area (m2/kg)
    aleaf: Float[Array, "npft"]            # C allocation fraction to leaves
    aroot: Float[Array, "npft"]            # C allocation fraction to roots
    awood: Float[Array, "npft"]            # C allocation fraction to wood


def create_vegetation_state(npoi: int, npft: int,
                            nsoilay: int) -> VegetationState:
    """Create a zero-initialised VegetationState."""
    z = jnp.zeros((npoi,))
    return VegetationState(
        cbiol=jnp.zeros((npoi, npft)),
        cbior=jnp.zeros((npoi, npft)),
        cbiow=jnp.zeros((npoi, npft)),
        plai=jnp.zeros((npoi, npft)),
        frac=jnp.zeros((npoi, npft)),
        biomass=jnp.zeros((npoi, npft)),
        exist=jnp.ones((npoi, npft)),
        tnpp=jnp.zeros((npoi, npft)),
        tgpp=jnp.zeros((npoi, npft)),
        lai=jnp.zeros((npoi, 2)),
        sai=jnp.zeros((npoi, 2)),
        ztop=jnp.zeros((npoi, 2)),
        zbot=jnp.zeros((npoi, 2)),
        fu=z, fl=z,
        froot=jnp.zeros((nsoilay, 2)),
        tempu=z, templ=z,
        dropu=jnp.ones((npoi,)),
        dropls=jnp.ones((npoi,)),
        dropl3=jnp.ones((npoi,)),
        dropl4=jnp.ones((npoi,)),
        agddu=z, agddl=z,
        tc=z, tw=z, gdd0=z, gdd5=z, tcmin=z,
        tnpptot=z, tgpptot=z,
        tco2mic=z, tco2root=z, tneetot=z,
        sapfrac=z, firefac=z,
        totlaiu=z, totlail=z,
        totbiou=z, totbiol_agg=z,
        disturbf=z, disturbo=z,
        vegtype0=z, tnmin=z,
        gdd0this=z, gdd5this=z,
        tcthis=z, twthis=z,
        specla=jnp.zeros((npft,)),
        aleaf=jnp.zeros((npft,)),
        aroot=jnp.zeros((npft,)),
        awood=jnp.zeros((npft,)),
    )
