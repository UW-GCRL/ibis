"""Permafrost dynamics for IBIS-JAX.

Implements:
  - Active layer depth via Stefan equation
  - Modified thermal/hydraulic conductivity for frozen ground
  - Thermokarst feedback (excess ice melt)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PermafrostState(eqx.Module):
    """Permafrost diagnostic state."""

    active_layer: Float[Array, "npoi"]       # Active layer depth (m)
    frozen_ground: Float[Array, "npoi"]      # Frozen ground indicator [0,1]
    excess_ice: Float[Array, "npoi"]         # Excess ice content (kg/m2)
    thaw_depth: Float[Array, "npoi"]         # Annual maximum thaw depth (m)
    talik: Float[Array, "npoi"]              # Talik indicator (unfrozen below permafrost) [0,1]


def create_permafrost_state(npoi: int) -> PermafrostState:
    """Create zero-initialised PermafrostState."""
    z = jnp.zeros((npoi,))
    return PermafrostState(
        active_layer=z, frozen_ground=z, excess_ice=z,
        thaw_depth=z, talik=z,
    )


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class PermafrostParams(eqx.Module):
    """Permafrost model parameters."""

    # Stefan equation
    nf: float = 0.5                  # Freezing n-factor (dimensionless)
    nt: float = 1.0                  # Thawing n-factor
    ksoil_thawed: float = 1.2        # Thermal conductivity, thawed (W/m/K)
    ksoil_frozen: float = 2.4        # Thermal conductivity, frozen (W/m/K)
    csoil: float = 2.5e6             # Volumetric heat capacity (J/m3/K)
    Lf: float = 3.34e5               # Latent heat of fusion (J/kg)
    rho_ice: float = 917.0           # Ice density (kg/m3)
    porosity_excess: float = 0.05    # Excess ice porosity fraction

    # Thermokarst
    subsidence_rate: float = 0.01    # Subsidence per excess ice melt (m/kgice)
    excess_ice_init: float = 50.0    # Initial excess ice (kg/m2) for permafrost points


def default_permafrost_params() -> PermafrostParams:
    return PermafrostParams()


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def stefan_depth(tsoi, tmelt, dtime_annual, params: PermafrostParams):
    """Compute active layer depth using the Stefan equation.

    Parameters
    ----------
    tsoi : (npoi, nsoilay) soil temperature (K)
    tmelt : float — freezing point (K), typically 273.16
    dtime_annual : float — annual thawing degree-day sum (K*day)
    params : PermafrostParams

    Returns
    -------
    ald : (npoi,) active layer depth (m)
    """
    # Effective thermal conductivity (using mean of thawed/frozen)
    k_eff = (params.ksoil_thawed + params.ksoil_frozen) / 2.0

    # Stefan equation: ALD = sqrt(2 * k * TDD / (rho_ice * Lf * porosity))
    tdd = jnp.maximum(dtime_annual, 0.0)
    porosity = params.porosity_excess + 0.4  # total porosity estimate
    ald = jnp.sqrt(
        2.0 * k_eff * tdd / (params.rho_ice * params.Lf * porosity + 1e-10)
    )

    return ald


def diagnose_frozen_ground(tsoi, tmelt):
    """Diagnose frozen ground from soil temperature profile.

    Parameters
    ----------
    tsoi : (npoi, nsoilay) soil temperature (K)
    tmelt : float — freezing point (K)

    Returns
    -------
    frozen : (npoi,) fraction of layers below freezing [0,1]
    active_layer : (npoi,) depth of first frozen layer proxy (m)
    """
    frozen_layers = (tsoi < tmelt).astype(jnp.float32)
    frozen_frac = jnp.mean(frozen_layers, axis=1)

    # Active layer = depth at which soil first freezes
    # Use number of unfrozen layers from top as proxy
    unfrozen_from_top = jnp.cumsum(1.0 - frozen_layers, axis=1)
    # Count consecutive unfrozen layers
    first_frozen = jnp.argmax(frozen_layers, axis=1).astype(jnp.float32)
    # If no frozen layer, active_layer = full depth
    any_frozen = jnp.any(frozen_layers > 0, axis=1).astype(jnp.float32)
    nsoilay = tsoi.shape[1]
    layer_thickness = 0.5  # approximate layer thickness (m)
    active_layer = jnp.where(
        any_frozen > 0,
        first_frozen * layer_thickness,
        nsoilay * layer_thickness,
    )

    return frozen_frac, active_layer


def thermokarst(excess_ice, active_layer, thaw_depth_prev, params: PermafrostParams):
    """Compute thermokarst subsidence from excess ice melt.

    Parameters
    ----------
    excess_ice : (npoi,) current excess ice (kg/m2)
    active_layer : (npoi,) current active layer depth (m)
    thaw_depth_prev : (npoi,) previous maximum thaw depth (m)
    params : PermafrostParams

    Returns
    -------
    new_excess_ice : (npoi,) updated excess ice
    subsidence : (npoi,) ground subsidence (m)
    """
    # Melt excess ice when active layer deepens
    deepening = jnp.maximum(active_layer - thaw_depth_prev, 0.0)
    ice_melt = deepening * params.rho_ice * params.porosity_excess
    ice_melt = jnp.minimum(ice_melt, excess_ice)

    new_excess_ice = excess_ice - ice_melt
    subsidence = ice_melt * params.subsidence_rate

    return new_excess_ice, subsidence


def permafrost_step(
    pf: PermafrostState,
    tsoi, tmelt,
    gdd0,
    params: PermafrostParams,
) -> PermafrostState:
    """One annual timestep of the permafrost model.

    Parameters
    ----------
    pf : PermafrostState
    tsoi : (npoi, nsoilay) soil temperature (K)
    tmelt : float — freezing point
    gdd0 : (npoi,) growing degree days above 0C (proxy for TDD)
    params : PermafrostParams

    Returns
    -------
    PermafrostState — updated
    """
    # Diagnose frozen ground
    frozen_frac, active_layer = diagnose_frozen_ground(tsoi, tmelt)

    # Stefan equation active layer depth
    ald_stefan = stefan_depth(tsoi, tmelt, gdd0, params)

    # Use the smaller of the two estimates
    active_layer = jnp.minimum(active_layer, ald_stefan)

    # Thermokarst
    new_excess_ice, subsidence = thermokarst(
        pf.excess_ice, active_layer, pf.thaw_depth, params
    )

    # Update maximum thaw depth
    new_thaw = jnp.maximum(pf.thaw_depth, active_layer)

    # Talik: unfrozen zone below permafrost (when active layer > some depth)
    talik = jnp.where(active_layer > 3.0, 1.0, 0.0)

    return PermafrostState(
        active_layer=active_layer,
        frozen_ground=frozen_frac,
        excess_ice=new_excess_ice,
        thaw_depth=new_thaw,
        talik=talik,
    )
