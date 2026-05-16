"""Soil state variables.

Ported from mod_soil_state.f90.  Contains prognostic soil temperature,
moisture, and ice; diagnostic conductivities and fluxes; static texture
and hydraulic properties.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class SoilState(eqx.Module):
    """Soil thermal, hydraulic, and surface state.

    Array shapes use named dimensions for clarity:
      npoi    = number of land grid points
      nsoilay = number of soil layers (typically 6)
    """

    # --- Prognostic (restart-required) ---
    tsoi: Float[Array, "npoi nsoilay"]     # Soil temperature (K)
    wsoi: Float[Array, "npoi nsoilay"]     # Liquid water fraction of pore space
    wisoi: Float[Array, "npoi nsoilay"]    # Ice fraction of pore space
    tg: Float[Array, "npoi"]               # Soil skin temperature (K)
    wpud: Float[Array, "npoi"]             # Puddle liquid water (kg/m2)
    wipud: Float[Array, "npoi"]            # Puddle ice (kg/m2)

    # --- Diagnostic (recomputed each timestep) ---
    consoi: Float[Array, "npoi nsoilay"]   # Thermal conductivity (W/m/K)
    csoi: Float[Array, "npoi nsoilay"]     # Volumetric heat capacity (J/m3/K)
    hflo: Float[Array, "npoi nsoilay_p1"]  # Heat flux at layer interfaces (W/m2)
    qglif: Float[Array, "npoi 4"]          # Evaporation source fractions
    heatg: Float[Array, "npoi"]            # Net heat flux into soil (W/m2)
    hvasug: Float[Array, "npoi"]           # Latent heat at soil surface (J/kg)
    z0soi: Float[Array, "npoi"]            # Roughness length of bare soil (m)
    stresstl: Float[Array, "npoi"]         # Lower-canopy soil moisture stress
    stresstu: Float[Array, "npoi"]         # Upper-canopy soil moisture stress

    # --- Per-layer diagnostics ---
    stressl: Float[Array, "npoi nsoilay"]  # Moisture stress, lower canopy
    stressu: Float[Array, "npoi nsoilay"]  # Moisture stress, upper canopy
    upsoil: Float[Array, "npoi nsoilay"]   # Water uptake, lower canopy (kg/m2/s)
    upsoiu: Float[Array, "npoi nsoilay"]   # Water uptake, upper canopy (kg/m2/s)
    porosflo: Float[Array, "npoi nsoilay"] # Effective porosity after ice

    # --- Static properties (set at initialisation, never updated) ---
    poros: Float[Array, "npoi nsoilay"]    # Porosity
    hydraul: Float[Array, "npoi nsoilay"]  # Saturated hydraulic conductivity (m/s)
    suction: Float[Array, "npoi nsoilay"]  # Saturated matric potential (m)
    bex: Float[Array, "npoi nsoilay"]      # Campbell b exponent
    sfield: Float[Array, "npoi nsoilay"]   # Field capacity (frac of pore space)
    swilt: Float[Array, "npoi nsoilay"]    # Wilting point (frac of pore space)
    sand: Float[Array, "npoi nsoilay"]     # Percent sand
    clay: Float[Array, "npoi nsoilay"]     # Percent clay
    hsoi: Float[Array, "nsoilay_p1"]       # Layer thicknesses (m)
    albsav: Float[Array, "npoi"]          # Base soil albedo, visible
    albsan: Float[Array, "npoi"]          # Base soil albedo, near-IR


def create_soil_state(npoi: int, nsoilay: int) -> SoilState:
    """Create a zero-initialised SoilState."""
    return SoilState(
        tsoi=jnp.full((npoi, nsoilay), 278.0),
        wsoi=jnp.full((npoi, nsoilay), 0.5),
        wisoi=jnp.zeros((npoi, nsoilay)),
        tg=jnp.full((npoi,), 278.0),
        wpud=jnp.zeros((npoi,)),
        wipud=jnp.zeros((npoi,)),
        consoi=jnp.zeros((npoi, nsoilay)),
        csoi=jnp.zeros((npoi, nsoilay)),
        hflo=jnp.zeros((npoi, nsoilay + 1)),
        qglif=jnp.zeros((npoi, 4)),
        heatg=jnp.zeros((npoi,)),
        hvasug=jnp.zeros((npoi,)),
        z0soi=jnp.zeros((npoi,)),
        stresstl=jnp.zeros((npoi,)),
        stresstu=jnp.zeros((npoi,)),
        stressl=jnp.zeros((npoi, nsoilay)),
        stressu=jnp.zeros((npoi, nsoilay)),
        upsoil=jnp.zeros((npoi, nsoilay)),
        upsoiu=jnp.zeros((npoi, nsoilay)),
        porosflo=jnp.zeros((npoi, nsoilay)),
        poros=jnp.zeros((npoi, nsoilay)),
        hydraul=jnp.zeros((npoi, nsoilay)),
        suction=jnp.zeros((npoi, nsoilay)),
        bex=jnp.zeros((npoi, nsoilay)),
        sfield=jnp.zeros((npoi, nsoilay)),
        swilt=jnp.zeros((npoi, nsoilay)),
        sand=jnp.zeros((npoi, nsoilay)),
        clay=jnp.zeros((npoi, nsoilay)),
        hsoi=jnp.zeros((nsoilay + 1,)),
        albsav=jnp.full((npoi,), 0.1),
        albsan=jnp.full((npoi,), 0.2),
    )
