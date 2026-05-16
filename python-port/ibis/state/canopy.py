"""Canopy state variables.

Ported from mod_canopy_params.f90 and mod_canopy_fluxes.f90.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class CanopyState(eqx.Module):
    """Canopy temperatures, intercepted water, and stomatal state.

    Array shapes:
      npoi  = number of land grid points
      nband = radiation wavebands (2: VIS, NIR)
    """

    # --- Prognostic temperatures ---
    tu: Float[Array, "npoi"]               # Upper canopy leaf temperature (K)
    ts: Float[Array, "npoi"]               # Upper canopy stem temperature (K)
    tl: Float[Array, "npoi"]               # Lower canopy temperature (K)
    t12: Float[Array, "npoi"]              # Air temperature at z12 (K)
    t34: Float[Array, "npoi"]              # Air temperature at z34 (K)
    q12: Float[Array, "npoi"]              # Specific humidity at z12 (kg/kg)
    q34: Float[Array, "npoi"]              # Specific humidity at z34 (kg/kg)

    # --- Intercepted water ---
    wliqu: Float[Array, "npoi"]            # Liquid on upper leaves (kg/m2)
    wliqs: Float[Array, "npoi"]            # Liquid on upper stems (kg/m2)
    wliql: Float[Array, "npoi"]            # Liquid on lower canopy (kg/m2)
    wsnou: Float[Array, "npoi"]            # Snow on upper leaves (kg/m2)
    wsnos: Float[Array, "npoi"]            # Snow on upper stems (kg/m2)
    wsnol: Float[Array, "npoi"]            # Snow on lower canopy (kg/m2)

    # --- Stomatal conductance per PFT group ---
    # Upper canopy: [broadleaf, conifer]
    ci_upper: Float[Array, "npoi 2"]       # Intercellular CO2 (mol/mol)
    gs_upper: Float[Array, "npoi 2"]       # Stomatal conductance (mol/m2/s)
    cs_upper: Float[Array, "npoi 2"]       # Surface CO2 (mol/mol)
    an_upper: Float[Array, "npoi 2"]       # Net photosynthesis (mol/m2/s)
    ag_upper: Float[Array, "npoi 2"]       # Gross photosynthesis (mol/m2/s)
    # Lower canopy: [shrub, C3grass, C4grass]
    ci_lower: Float[Array, "npoi 3"]       # Intercellular CO2
    gs_lower: Float[Array, "npoi 3"]       # Stomatal conductance
    cs_lower: Float[Array, "npoi 3"]       # Surface CO2
    an_lower: Float[Array, "npoi 3"]       # Net photosynthesis
    ag_lower: Float[Array, "npoi 3"]       # Gross photosynthesis

    # --- Canopy wetness fractions ---
    fwetu: Float[Array, "npoi"]            # Wetted fraction, upper leaves
    fwets: Float[Array, "npoi"]            # Wetted fraction, upper stems
    fwetl: Float[Array, "npoi"]            # Wetted fraction, lower canopy
    rliqu: Float[Array, "npoi"]            # Liquid fraction of intercepted water, upper leaves
    rliqs: Float[Array, "npoi"]            # Liquid fraction, upper stems
    rliql: Float[Array, "npoi"]            # Liquid fraction, lower canopy

    # --- Radiation fluxes (diagnostic, per-timestep) ---
    solu: Float[Array, "npoi"]             # Solar absorbed by upper leaves (W/m2)
    sols: Float[Array, "npoi"]             # Solar absorbed by upper stems
    soll: Float[Array, "npoi"]             # Solar absorbed by lower canopy
    solg: Float[Array, "npoi"]             # Solar absorbed by soil
    soli: Float[Array, "npoi"]             # Solar absorbed by snow
    firu: Float[Array, "npoi"]             # IR absorbed by upper leaves
    firs: Float[Array, "npoi"]             # IR absorbed by upper stems
    firl: Float[Array, "npoi"]             # IR absorbed by lower canopy
    firg: Float[Array, "npoi"]             # IR absorbed by soil
    firi: Float[Array, "npoi"]             # IR absorbed by snow
    firb: Float[Array, "npoi"]             # Total upward IR flux (W/m2)

    # --- Surface albedos (computed by radiation) ---
    asurd: Float[Array, "npoi nband"]      # Effective surface albedo, direct
    asuri: Float[Array, "npoi nband"]      # Effective surface albedo, diffuse

    # --- PAR and canopy scaling (computed by radiation, used by stomata) ---
    topparu: Float[Array, "npoi"]          # Top-of-canopy PAR, upper (W/m2)
    topparl: Float[Array, "npoi"]          # Top-of-canopy PAR, lower (W/m2)
    scalcoefu: Float[Array, "npoi 4"]      # PAR scaling coefficients, upper
    scalcoefl: Float[Array, "npoi 4"]      # PAR scaling coefficients, lower

    # --- Sensible / latent heat fluxes ---
    fsenu: Float[Array, "npoi"]            # Sensible heat, upper leaves (W/m2)
    fsens: Float[Array, "npoi"]            # Sensible heat, upper stems
    fsenl: Float[Array, "npoi"]            # Sensible heat, lower canopy
    fseng: Float[Array, "npoi"]            # Sensible heat, soil
    fseni: Float[Array, "npoi"]            # Sensible heat, snow
    fvapg: Float[Array, "npoi"]            # Evaporation from soil (kg/m2/s)
    fvapi: Float[Array, "npoi"]            # Evaporation from snow

    # --- Precipitation partitioning ---
    raing: Float[Array, "npoi"]            # Rainfall reaching ground (kg/m2/s)
    snowg: Float[Array, "npoi"]            # Snowfall reaching ground

    # --- 10-day running mean PAR scaling (used by stomata) ---
    a10scalparamu: Float[Array, "npoi"]    # Weighted PAR*scale, upper canopy
    a10daylightu: Float[Array, "npoi"]     # Weighted PAR, upper canopy
    a10scalparaml: Float[Array, "npoi"]    # Weighted PAR*scale, lower canopy
    a10daylightl: Float[Array, "npoi"]     # Weighted PAR, lower canopy

    # --- Aerodynamic properties ---
    rhoa: Float[Array, "npoi"]             # Air density at reference level (kg/m3)


def create_canopy_state(npoi: int) -> CanopyState:
    """Create a default-initialised CanopyState."""
    z = jnp.zeros((npoi,))
    t_init = jnp.full((npoi,), 283.0)
    return CanopyState(
        tu=t_init, ts=t_init, tl=t_init,
        t12=t_init, t34=t_init,
        q12=z, q34=z,
        wliqu=z, wliqs=z, wliql=z,
        wsnou=z, wsnos=z, wsnol=z,
        ci_upper=jnp.full((npoi, 2), 350.0e-6),
        gs_upper=jnp.full((npoi, 2), 0.01),
        cs_upper=jnp.full((npoi, 2), 350.0e-6),
        an_upper=jnp.zeros((npoi, 2)),
        ag_upper=jnp.zeros((npoi, 2)),
        ci_lower=jnp.full((npoi, 3), 350.0e-6),
        gs_lower=jnp.full((npoi, 3), 0.01),
        cs_lower=jnp.full((npoi, 3), 350.0e-6),
        an_lower=jnp.zeros((npoi, 3)),
        ag_lower=jnp.zeros((npoi, 3)),
        fwetu=z, fwets=z, fwetl=z,
        rliqu=z, rliqs=z, rliql=z,
        solu=z, sols=z, soll=z, solg=z, soli=z,
        firu=z, firs=z, firl=z, firg=z, firi=z,
        firb=z,
        asurd=jnp.zeros((npoi, 2)), asuri=jnp.zeros((npoi, 2)),
        topparu=z, topparl=z,
        scalcoefu=jnp.zeros((npoi, 4)), scalcoefl=jnp.zeros((npoi, 4)),
        fsenu=z, fsens=z, fsenl=z, fseng=z, fseni=z,
        fvapg=z, fvapi=z,
        raing=z, snowg=z,
        a10scalparamu=jnp.full((npoi,), 0.5),
        a10daylightu=jnp.full((npoi,), 100.0),
        a10scalparaml=jnp.full((npoi,), 0.5),
        a10daylightl=jnp.full((npoi,), 100.0),
        rhoa=jnp.full((npoi,), 1.225),
    )
