"""Snow state variables.

Ported from mod_snow_state.f90.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class SnowState(eqx.Module):
    """Snow layer state.

    Array shapes:
      npoi    = number of land grid points
      nsnolay = number of snow layers (typically 3)
    """

    # --- Prognostic ---
    fi: Float[Array, "npoi"]               # Fractional snow cover
    tsno: Float[Array, "npoi nsnolay"]     # Snow temperature (K)
    hsno: Float[Array, "npoi nsnolay"]     # Snow layer thickness (m)
    ti: Float[Array, "npoi"]               # Snow skin temperature (K)
    tlsub: Float[Array, "npoi"]            # Buried lower-veg temperature (K)

    # --- Diagnostic ---
    heati: Float[Array, "npoi"]            # Net heat flux into snow (W/m2)
    hvasui: Float[Array, "npoi"]           # Latent heat at snow surface (J/kg)

    # --- Static snow parameters ---
    z0sno: float = 0.005                   # Roughness length (m)
    rhos: float = 250.0                    # Snow density (kg/m3)
    consno: float = 0.20                   # Thermal conductivity (W/m/K)
    hsnotop: float = 0.05                  # Top layer thickness (m)
    hsnomin: float = 0.01                  # Minimum total thickness (m)
    fimin: float = 0.02                    # Minimum fractional cover
    fimax: float = 1.00                    # Maximum fractional cover


def create_snow_state(npoi: int, nsnolay: int) -> SnowState:
    """Create a zero-initialised SnowState."""
    return SnowState(
        fi=jnp.zeros((npoi,)),
        tsno=jnp.full((npoi, nsnolay), 273.16),
        hsno=jnp.zeros((npoi, nsnolay)),
        ti=jnp.full((npoi,), 273.16),
        tlsub=jnp.full((npoi,), 273.16),
        heati=jnp.zeros((npoi,)),
        hvasui=jnp.zeros((npoi,)),
    )
