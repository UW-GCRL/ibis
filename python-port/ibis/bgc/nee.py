"""Net Ecosystem Exchange (NEE) accounting for IBIS-JAX.

Aggregates carbon and greenhouse gas fluxes into net budgets:
  NEE_CO2 = -GPP + Ra + Rh + fire_CO2
  NEE_CH4 = wetland_CH4 + fire_CH4
  NEE_N2O = soil_N2O + fire_N2O
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class NEEState(eqx.Module):
    """Net ecosystem exchange diagnostics."""

    nee_co2: Float[Array, "npoi"]    # Net CO2 exchange (gC/m2/day, positive = source)
    nee_ch4: Float[Array, "npoi"]    # Net CH4 exchange (gC/m2/day)
    nee_n2o: Float[Array, "npoi"]    # Net N2O exchange (gN/m2/day)
    gpp: Float[Array, "npoi"]        # Gross primary production (gC/m2/day)
    ra: Float[Array, "npoi"]         # Autotrophic respiration (gC/m2/day)
    rh: Float[Array, "npoi"]         # Heterotrophic respiration (gC/m2/day)
    fire_co2: Float[Array, "npoi"]   # Fire CO2 emissions (gC/m2/day)
    fire_ch4: Float[Array, "npoi"]   # Fire CH4 emissions (gC/m2/day)
    fire_n2o: Float[Array, "npoi"]   # Fire N2O emissions (gN/m2/day)
    wetland_ch4: Float[Array, "npoi"]  # Wetland CH4 emissions (gC/m2/day)
    soil_n2o: Float[Array, "npoi"]   # Soil N2O emissions (gN/m2/day)


def create_nee_state(npoi: int) -> NEEState:
    """Create zero-initialised NEEState."""
    z = jnp.zeros((npoi,))
    return NEEState(
        nee_co2=z, nee_ch4=z, nee_n2o=z,
        gpp=z, ra=z, rh=z,
        fire_co2=z, fire_ch4=z, fire_n2o=z,
        wetland_ch4=z, soil_n2o=z,
    )


# ---------------------------------------------------------------------------
# NEE computation
# ---------------------------------------------------------------------------

def compute_nee(
    gpp,
    ra,
    rh,
    fire_co2=None,
    fire_ch4=None,
    fire_n2o=None,
    wetland_ch4=None,
    soil_n2o=None,
):
    """Compute net ecosystem exchange budgets.

    Parameters
    ----------
    gpp : (npoi,) gross primary production (gC/m2/day)
    ra : (npoi,) autotrophic respiration (gC/m2/day)
    rh : (npoi,) heterotrophic respiration (gC/m2/day)
    fire_co2 : (npoi,) or None — fire CO2 emissions (gC/m2/day)
    fire_ch4 : (npoi,) or None — fire CH4 emissions (gC/m2/day)
    fire_n2o : (npoi,) or None — fire N2O emissions (gN/m2/day)
    wetland_ch4 : (npoi,) or None — wetland CH4 flux (gC/m2/day)
    soil_n2o : (npoi,) or None — soil N2O flux (gN/m2/day)

    Returns
    -------
    NEEState with all flux components and net budgets
    """
    npoi = gpp.shape[0]
    z = jnp.zeros((npoi,))

    # Default missing fluxes to zero
    f_co2 = fire_co2 if fire_co2 is not None else z
    f_ch4 = fire_ch4 if fire_ch4 is not None else z
    f_n2o = fire_n2o if fire_n2o is not None else z
    w_ch4 = wetland_ch4 if wetland_ch4 is not None else z
    s_n2o = soil_n2o if soil_n2o is not None else z

    # NEE_CO2: positive = net source to atmosphere
    nee_co2 = -gpp + ra + rh + f_co2

    # NEE_CH4: wetland + fire
    nee_ch4 = w_ch4 + f_ch4

    # NEE_N2O: soil + fire
    nee_n2o = s_n2o + f_n2o

    return NEEState(
        nee_co2=nee_co2,
        nee_ch4=nee_ch4,
        nee_n2o=nee_n2o,
        gpp=gpp,
        ra=ra,
        rh=rh,
        fire_co2=f_co2,
        fire_ch4=f_ch4,
        fire_n2o=f_n2o,
        wetland_ch4=w_ch4,
        soil_n2o=s_n2o,
    )
