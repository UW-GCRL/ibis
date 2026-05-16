"""Prognostic wetland area and CH4 emissions for IBIS-JAX.

Implements:
  - Prognostic wetland fraction from TOPMODEL-like approach
  - CH4 production (methanogenesis with Q10 temperature dependence)
  - CH4 oxidation (methanotrophy)
  - CH4 transport: ebullition, plant-mediated, diffusion
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class WetlandState(eqx.Module):
    """Prognostic wetland state variables."""

    wetland_frac: Float[Array, "npoi"]       # Wetland area fraction [0,1]
    ch4_soil: Float[Array, "npoi"]           # Soil CH4 pool (gC/m2)
    water_table: Float[Array, "npoi"]        # Water table depth (m, negative = below surface)
    ch4_flux: Float[Array, "npoi"]           # Net CH4 emission (gC/m2/day)
    ch4_prod: Float[Array, "npoi"]           # CH4 production (gC/m2/day)
    ch4_oxid: Float[Array, "npoi"]           # CH4 oxidation (gC/m2/day)
    ch4_ebull: Float[Array, "npoi"]          # Ebullition flux (gC/m2/day)
    ch4_plant: Float[Array, "npoi"]          # Plant-mediated transport (gC/m2/day)
    ch4_diff: Float[Array, "npoi"]           # Diffusion flux (gC/m2/day)


def create_wetland_state(npoi: int) -> WetlandState:
    """Create zero-initialised WetlandState."""
    z = jnp.zeros((npoi,))
    return WetlandState(
        wetland_frac=z, ch4_soil=z, water_table=z,
        ch4_flux=z, ch4_prod=z, ch4_oxid=z,
        ch4_ebull=z, ch4_plant=z, ch4_diff=z,
    )


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class WetlandParams(eqx.Module):
    """Wetland CH4 parameters."""

    # TOPMODEL wetland fraction parameters
    fmax: float = 0.38              # Maximum wetland fraction
    cs_topo: float = 8.0            # Topographic index scale parameter
    wtd_scale: float = 0.3          # Water table depth scaling (m)

    # Methanogenesis
    q10_ch4: float = 3.0            # Q10 for CH4 production
    ch4_base: float = 0.15          # Base CH4 production rate (gC/m2/day at 25C)
    tref_ch4: float = 298.15        # Reference temperature for Q10 (K)

    # Oxidation
    vmax_oxid: float = 10.0         # Max oxidation rate (gC/m2/day)
    km_oxid: float = 5.0            # Half-saturation for oxidation (gC/m2)
    oxid_depth: float = 0.1         # Depth of aerobic zone above water table (m)

    # Transport
    ebull_threshold: float = 500.0  # CH4 threshold for ebullition (gC/m2)
    ebull_rate: float = 0.5         # Fraction released per day above threshold
    plant_transport: float = 0.4    # Fraction of production via plant transport
    diff_coeff: float = 0.05        # Diffusion coefficient (m2/day)


def default_wetland_params() -> WetlandParams:
    return WetlandParams()


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def wetland_fraction(wsoi, poros, params: WetlandParams):
    """Compute prognostic wetland area fraction.

    TOPMODEL-like approach: wetland fraction depends on soil saturation
    of the top soil layers.

    Parameters
    ----------
    wsoi : (npoi, nsoilay) — liquid water fraction of pore space
    poros : (npoi, nsoilay) — porosity
    params : WetlandParams

    Returns
    -------
    fwet : (npoi,) wetland fraction [0, fmax]
    wtd : (npoi,) estimated water table depth (m, negative below surface)
    """
    # Mean saturation of top 3 layers
    sat_top = jnp.mean(wsoi[:, :3], axis=1)

    # Wetland fraction from saturation
    fwet = params.fmax * jnp.clip(
        (sat_top - 0.5) / params.wtd_scale, 0.0, 1.0
    )

    # Estimate water table depth from saturation profile
    # Negative = below surface; 0 = at surface
    wtd = -params.wtd_scale * (1.0 - sat_top)

    return fwet, wtd


def ch4_production(tsoi, carbon_het_resp, fwet, params: WetlandParams):
    """Compute CH4 production from methanogenesis.

    Parameters
    ----------
    tsoi : (npoi, nsoilay) — soil temperature (K)
    carbon_het_resp : (npoi,) — heterotrophic respiration (gC/m2/day)
    fwet : (npoi,) — wetland fraction
    params : WetlandParams

    Returns
    -------
    prod : (npoi,) CH4 production (gC/m2/day)
    """
    # Mean temperature of saturated zone (top 3 layers)
    tsoil_mean = jnp.mean(tsoi[:, :3], axis=1)

    # Q10 temperature response
    f_temp = params.q10_ch4 ** ((tsoil_mean - params.tref_ch4) / 10.0)
    f_temp = jnp.clip(f_temp, 0.0, 20.0)

    # Production proportional to base rate * temperature * wetland fraction
    # Also scaled by heterotrophic respiration (substrate availability)
    substrate = jnp.maximum(carbon_het_resp, 0.0)
    prod = params.ch4_base * f_temp * fwet * (1.0 + substrate * 0.1)

    return prod


def ch4_oxidation(ch4_soil, fwet, water_table, params: WetlandParams):
    """Compute CH4 oxidation (methanotrophy).

    Michaelis-Menten kinetics in the aerobic zone above the water table.

    Parameters
    ----------
    ch4_soil : (npoi,) — soil CH4 pool (gC/m2)
    fwet : (npoi,) — wetland fraction
    water_table : (npoi,) — water table depth (m)
    params : WetlandParams

    Returns
    -------
    oxid : (npoi,) CH4 oxidation (gC/m2/day)
    """
    # Aerobic zone fraction (thicker when water table is deeper)
    aero_frac = jnp.clip(-water_table / params.oxid_depth, 0.0, 1.0)

    # Michaelis-Menten oxidation
    oxid = params.vmax_oxid * ch4_soil / (params.km_oxid + ch4_soil)

    # Scale by aerobic fraction and non-wetland area
    oxid = oxid * (1.0 - fwet) * aero_frac

    return oxid


def ch4_transport(ch4_soil, ch4_prod_rate, fwet, params: WetlandParams):
    """Compute CH4 transport fluxes: ebullition, plant, diffusion.

    Parameters
    ----------
    ch4_soil : (npoi,) — soil CH4 pool
    ch4_prod_rate : (npoi,) — current production rate
    fwet : (npoi,) — wetland fraction
    params : WetlandParams

    Returns
    -------
    ebull, plant, diff : (npoi,) each, transport fluxes (gC/m2/day)
    """
    # Ebullition: triggered when soil pool exceeds threshold
    excess = jnp.maximum(ch4_soil - params.ebull_threshold, 0.0)
    ebull = excess * params.ebull_rate * fwet

    # Plant-mediated transport: fraction of production
    plant = ch4_prod_rate * params.plant_transport * fwet

    # Diffusion: proportional to soil pool and gradient
    diff = ch4_soil * params.diff_coeff * fwet

    return ebull, plant, diff


def wetland_step(
    wetland: WetlandState,
    tsoi, wsoi, poros,
    carbon_het_resp,
    params: WetlandParams,
) -> WetlandState:
    """One daily timestep of the wetland CH4 model.

    Parameters
    ----------
    wetland : WetlandState
    tsoi : (npoi, nsoilay) soil temperature
    wsoi : (npoi, nsoilay) soil moisture fraction
    poros : (npoi, nsoilay) porosity
    carbon_het_resp : (npoi,) heterotrophic respiration (gC/m2/day)
    params : WetlandParams

    Returns
    -------
    WetlandState — updated
    """
    # Wetland fraction
    fwet, wtd = wetland_fraction(wsoi, poros, params)

    # Production
    prod = ch4_production(tsoi, carbon_het_resp, fwet, params)

    # Update soil pool
    ch4_pool = wetland.ch4_soil + prod

    # Oxidation
    oxid = ch4_oxidation(ch4_pool, fwet, wtd, params)
    ch4_pool = jnp.maximum(ch4_pool - oxid, 0.0)

    # Transport
    ebull, plant, diff = ch4_transport(ch4_pool, prod, fwet, params)
    total_transport = ebull + plant + diff
    ch4_pool = jnp.maximum(ch4_pool - total_transport, 0.0)

    # Net flux
    flux = ebull + plant + diff

    return WetlandState(
        wetland_frac=fwet,
        ch4_soil=ch4_pool,
        water_table=wtd,
        ch4_flux=flux,
        ch4_prod=prod,
        ch4_oxid=oxid,
        ch4_ebull=ebull,
        ch4_plant=plant,
        ch4_diff=diff,
    )
