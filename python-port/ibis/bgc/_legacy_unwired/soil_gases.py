"""Soil N2O emissions for IBIS-JAX.

Implements nitrification + denitrification N2O following a simplified
DNDC/DayCent approach: N2O = f(NH4+, NO3-, temperature, WFPS, labile C).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SoilGasState(eqx.Module):
    """Soil gas emission state."""

    n2o_nitrif: Float[Array, "npoi"]    # N2O from nitrification (gN/m2/day)
    n2o_denitrif: Float[Array, "npoi"]  # N2O from denitrification (gN/m2/day)
    n2o_total: Float[Array, "npoi"]     # Total soil N2O (gN/m2/day)
    nh4: Float[Array, "npoi"]           # Ammonium pool (gN/m2)
    no3: Float[Array, "npoi"]           # Nitrate pool (gN/m2)


def create_soil_gas_state(npoi: int) -> SoilGasState:
    z = jnp.zeros((npoi,))
    return SoilGasState(
        n2o_nitrif=z, n2o_denitrif=z, n2o_total=z,
        nh4=jnp.full((npoi,), 0.5),  # initial NH4
        no3=jnp.full((npoi,), 0.3),  # initial NO3
    )


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class SoilGasParams(eqx.Module):
    """Soil N2O parameters."""

    # Nitrification
    k_nitrif: float = 0.05          # Base nitrification rate (1/day)
    ef_nitrif: float = 0.01         # N2O emission factor for nitrification
    q10_nitrif: float = 2.0         # Q10 for nitrification
    tref: float = 298.15            # Reference temperature (K)

    # Denitrification
    k_denitrif: float = 0.1         # Base denitrification rate (1/day)
    ef_denitrif: float = 0.02       # N2O emission factor for denitrification
    wfps_threshold: float = 0.6     # WFPS threshold for denitrification
    q10_denitrif: float = 2.5       # Q10 for denitrification

    # N mineralization input
    cn_ratio_litter: float = 50.0   # C:N ratio of litter
    min_rate: float = 0.005         # Fraction of litter C mineralised to NH4/day


def default_soil_gas_params() -> SoilGasParams:
    return SoilGasParams()


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def wfps(wsoi, poros):
    """Water-filled pore space.

    Parameters
    ----------
    wsoi : (npoi, nsoilay) liquid water fraction
    poros : (npoi, nsoilay) porosity

    Returns
    -------
    (npoi,) mean WFPS of top 3 layers
    """
    wfps_layers = wsoi / jnp.maximum(poros, 1e-6)
    return jnp.mean(jnp.clip(wfps_layers[:, :3], 0.0, 1.0), axis=1)


def nitrification(nh4, tsoi_mean, wfps_val, params: SoilGasParams):
    """Nitrification: NH4 -> NO3, with N2O byproduct.

    Parameters
    ----------
    nh4 : (npoi,) ammonium pool
    tsoi_mean : (npoi,) mean soil temperature (K)
    wfps_val : (npoi,) water-filled pore space
    params : SoilGasParams

    Returns
    -------
    n2o : (npoi,) N2O from nitrification
    nh4_consumed : (npoi,) NH4 consumed
    no3_produced : (npoi,) NO3 produced
    """
    # Temperature response
    f_temp = params.q10_nitrif ** ((tsoi_mean - params.tref) / 10.0)
    f_temp = jnp.clip(f_temp, 0.0, 10.0)

    # Moisture response: peaks at WFPS ~0.6, declines at high WFPS
    f_moist = jnp.where(
        wfps_val < 0.6,
        wfps_val / 0.6,
        jnp.maximum(1.0 - (wfps_val - 0.6) / 0.4, 0.0),
    )

    # Nitrification rate
    rate = params.k_nitrif * f_temp * f_moist
    nh4_consumed = jnp.minimum(nh4 * rate, nh4)
    no3_produced = nh4_consumed * (1.0 - params.ef_nitrif)
    n2o = nh4_consumed * params.ef_nitrif

    return n2o, nh4_consumed, no3_produced


def denitrification(no3, tsoi_mean, wfps_val, labile_c, params: SoilGasParams):
    """Denitrification: NO3 -> N2 + N2O.

    Occurs under anaerobic conditions (high WFPS).

    Parameters
    ----------
    no3 : (npoi,) nitrate pool
    tsoi_mean : (npoi,) mean soil temperature (K)
    wfps_val : (npoi,) water-filled pore space
    labile_c : (npoi,) labile carbon availability (kgC/m2)
    params : SoilGasParams

    Returns
    -------
    n2o : (npoi,) N2O from denitrification
    no3_consumed : (npoi,) NO3 consumed
    """
    # Temperature response
    f_temp = params.q10_denitrif ** ((tsoi_mean - params.tref) / 10.0)
    f_temp = jnp.clip(f_temp, 0.0, 10.0)

    # Moisture response: only active above WFPS threshold
    f_moist = jnp.clip(
        (wfps_val - params.wfps_threshold) / (1.0 - params.wfps_threshold),
        0.0, 1.0,
    )

    # Carbon limitation
    f_carbon = labile_c / (labile_c + 0.1)

    # Denitrification rate
    rate = params.k_denitrif * f_temp * f_moist * f_carbon
    no3_consumed = jnp.minimum(no3 * rate, no3)
    n2o = no3_consumed * params.ef_denitrif

    return n2o, no3_consumed


def soil_gas_step(
    sg: SoilGasState,
    tsoi, wsoi, poros,
    litter_c,
    n_mineralization,
    params: SoilGasParams,
) -> SoilGasState:
    """One daily timestep of the soil gas model.

    Parameters
    ----------
    sg : SoilGasState
    tsoi : (npoi, nsoilay) soil temperature (K)
    wsoi : (npoi, nsoilay) soil moisture fraction
    poros : (npoi, nsoilay) porosity
    litter_c : (npoi,) labile litter carbon (kgC/m2)
    n_mineralization : (npoi,) N mineralised from decomposition (gN/m2/day)
    params : SoilGasParams

    Returns
    -------
    SoilGasState — updated
    """
    # WFPS
    wfps_val = wfps(wsoi, poros)

    # Mean soil temperature (top 3 layers)
    tsoi_mean = jnp.mean(tsoi[:, :3], axis=1)

    # Add N mineralization to NH4 pool
    nh4 = sg.nh4 + n_mineralization

    # Nitrification
    n2o_nitrif, nh4_consumed, no3_produced = nitrification(
        nh4, tsoi_mean, wfps_val, params
    )
    nh4 = nh4 - nh4_consumed
    no3 = sg.no3 + no3_produced

    # Denitrification
    n2o_denitrif, no3_consumed = denitrification(
        no3, tsoi_mean, wfps_val, litter_c, params
    )
    no3 = no3 - no3_consumed

    return SoilGasState(
        n2o_nitrif=n2o_nitrif,
        n2o_denitrif=n2o_denitrif,
        n2o_total=n2o_nitrif + n2o_denitrif,
        nh4=jnp.maximum(nh4, 0.0),
        no3=jnp.maximum(no3, 0.0),
    )
