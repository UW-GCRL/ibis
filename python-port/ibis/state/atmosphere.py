"""Atmospheric forcing state.

Ported from mod_atm_state.f90.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


class AtmState(eqx.Module):
    """Atmospheric forcing at the current timestep.

    Array shapes:
      npoi  = number of land grid points
      nband = radiation wavebands (2)
    """

    # --- Instantaneous forcing ---
    ta: Float[Array, "npoi"]               # Air temperature (K)
    qa: Float[Array, "npoi"]               # Specific humidity (kg/kg)
    ua: Float[Array, "npoi"]               # Wind speed (m/s)
    psurf: Float[Array, "npoi"]            # Surface pressure (Pa)
    coszen: Float[Array, "npoi"]           # Cosine of solar zenith angle
    solad: Float[Array, "npoi nband"]      # Direct solar flux (W/m2)
    solai: Float[Array, "npoi nband"]      # Diffuse solar flux (W/m2)
    fira: Float[Array, "npoi"]             # Incoming longwave (W/m2)
    raina: Float[Array, "npoi"]            # Rainfall rate (kg/m2/s)
    snowa: Float[Array, "npoi"]            # Snowfall rate (kg/m2/s)

    # --- Daily values (from weather generator) ---
    td: Float[Array, "npoi"]               # Daily mean temperature (K)
    tmax: Float[Array, "npoi"]             # Daily max temperature (K)
    tmin: Float[Array, "npoi"]             # Daily min temperature (K)
    qd: Float[Array, "npoi"]              # Daily mean humidity (kg/kg)
    ud: Float[Array, "npoi"]              # Daily mean wind (m/s)
    precip: Float[Array, "npoi"]           # Daily precipitation (mm/day)
    cloud: Float[Array, "npoi"]            # Cloud fraction
    rh: Float[Array, "npoi"]               # Relative humidity (%)

    # --- Weather generator memory ---
    xstore: Float[Array, "npoi 3"]         # Autocorrelation memory
    iwet: Int[Array, "npoi"]               # Wet-day flag (0 or 1)

    # --- Atmospheric gas concentrations ---
    co2conc: float                         # CO2 (mol/mol)
    o2conc: float                          # O2 (mol/mol)


def create_atm_state(npoi: int, nband: int,
                     co2: float = 350.0e-6,
                     o2: float = 0.209) -> AtmState:
    """Create a default-initialised AtmState."""
    z = jnp.zeros((npoi,))
    return AtmState(
        ta=jnp.full((npoi,), 283.0),
        qa=jnp.full((npoi,), 0.005),
        ua=jnp.full((npoi,), 3.0),
        psurf=jnp.full((npoi,), 101325.0),
        coszen=z,
        solad=jnp.zeros((npoi, nband)),
        solai=jnp.zeros((npoi, nband)),
        fira=z,
        raina=z,
        snowa=z,
        td=jnp.full((npoi,), 283.0),
        tmax=jnp.full((npoi,), 288.0),
        tmin=jnp.full((npoi,), 278.0),
        qd=jnp.full((npoi,), 0.005),
        ud=jnp.full((npoi,), 3.0),
        precip=z,
        cloud=jnp.full((npoi,), 0.5),
        rh=jnp.full((npoi,), 70.0),
        xstore=jnp.zeros((npoi, 3)),
        iwet=jnp.zeros((npoi,), dtype=jnp.int32),
        co2conc=co2,
        o2conc=o2,
    )
