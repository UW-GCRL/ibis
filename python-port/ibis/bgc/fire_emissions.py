"""Fire emissions of CO2, CH4, and N2O for IBIS-JAX.

Extends the existing IBIS fire disturbance model with trace gas emission
factors by PFT, following Andreae & Merlet (2001).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FireEmissionState(eqx.Module):
    """Fire emission diagnostics."""

    fire_co2: Float[Array, "npoi"]    # CO2 emissions from fire (gC/m2/yr)
    fire_ch4: Float[Array, "npoi"]    # CH4 emissions from fire (gC/m2/yr)
    fire_n2o: Float[Array, "npoi"]    # N2O emissions from fire (gN/m2/yr)
    burned_area: Float[Array, "npoi"] # Burned fraction (0-1)


def create_fire_emission_state(npoi: int) -> FireEmissionState:
    z = jnp.zeros((npoi,))
    return FireEmissionState(fire_co2=z, fire_ch4=z, fire_n2o=z, burned_area=z)


# ---------------------------------------------------------------------------
# Emission factors (g species / kg dry matter burned)
# from Andreae & Merlet (2001), Table 1
# ---------------------------------------------------------------------------

# Indexed by vegetation type: 0=tropical, 1=boreal, 2=savanna, 3=crop
# CO2 emission factor (g CO2 / kg DM)
EF_CO2 = jnp.array([1580.0, 1569.0, 1613.0, 1515.0])
# CH4 emission factor (g CH4 / kg DM)
EF_CH4 = jnp.array([6.8, 5.7, 2.3, 2.7])
# N2O emission factor (g N2O / kg DM)
EF_N2O = jnp.array([0.20, 0.26, 0.21, 0.07])

# C fraction of dry matter
C_FRAC_DM = 0.45  # kgC per kg dry matter

# Molecular weight ratios
MW_CO2_C = 44.0 / 12.0    # g CO2 per g C
MW_CH4_C = 16.0 / 12.0    # g CH4 per g C
MW_N2O_N = 44.0 / 28.0    # g N2O per g N (for N content)


def fire_emissions(
    disturbf: Float[Array, "npoi"],
    biomass_burned: Float[Array, "npoi"],
    vegtype: Float[Array, "npoi"],
) -> FireEmissionState:
    """Compute fire emissions from burned biomass.

    Parameters
    ----------
    disturbf : (npoi,) fire disturbance fraction [0,1]
    biomass_burned : (npoi,) biomass consumed by fire (kgC/m2/yr)
    vegtype : (npoi,) vegetation type index (0-14)

    Returns
    -------
    FireEmissionState with CO2, CH4, N2O emissions
    """
    # Map vegtype to emission factor index
    # 1-5: tropical/warm trees → 0
    # 6-8: boreal trees → 1
    # 9-12: savanna/grassland → 2
    # 13-15: crop/other → 3
    ef_idx = jnp.where(vegtype <= 5, 0,
             jnp.where(vegtype <= 8, 1,
             jnp.where(vegtype <= 12, 2, 3)))

    # Convert biomass burned from kgC/m2 to kg DM/m2
    dm_burned = biomass_burned / C_FRAC_DM  # kg DM / m2

    # Emission factors for each grid point
    ef_co2 = EF_CO2[ef_idx]  # g CO2 / kg DM
    ef_ch4 = EF_CH4[ef_idx]
    ef_n2o = EF_N2O[ef_idx]

    # Emissions (convert from g species/m2 to gC/m2 or gN/m2)
    fire_co2 = dm_burned * ef_co2 / MW_CO2_C   # gC/m2/yr
    fire_ch4 = dm_burned * ef_ch4 / MW_CH4_C   # gC/m2/yr
    fire_n2o = dm_burned * ef_n2o / MW_N2O_N   # gN/m2/yr

    return FireEmissionState(
        fire_co2=fire_co2 * 1e-3,  # convert g to kg
        fire_ch4=fire_ch4 * 1e-3,
        fire_n2o=fire_n2o * 1e-3,
        burned_area=disturbf,
    )
