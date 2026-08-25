"""Methane scheme for IBIS-JAX -- M1: inundation and water table.

Built in tiers (docs/METHANE.md 4.0): M-lite first, and each later tier only if
evaluation shows the simpler one is systematically wrong. This module currently
holds M1 alone -- the water table diagnostic and the inundated fraction it is
paired with. Production, oxidation and transport are M2 onward.

`ibis/bgc/wetlands.py` is quarantined and must not be extended
(docs/METHANE.md 2.2); this is a separate module deliberately.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class MethaneParams(eqx.Module):
    """Parameters for the methane scheme. See docs/METHANE.md 4.7.

    Only the M1 subset is defined so far; adding a field before the tier that
    uses it invites the dead-parameter problem the fire module ran into, where
    `firefac` was computed, verified, and fed to nothing for 26 years.
    """

    #: Liquid saturation at or above which a layer counts as saturated. Below
    #: it, the layer contributes its unsaturated share to the water table
    #: depth. 0.95 rather than 1.0 because a real soil retains trapped air.
    sat_crit: float = 0.95
    #: Deepest water table the diagnostic will report, m. IBIS's soil column
    #: is 4 m, so nothing below that is resolved and reporting more would be
    #: invention.
    z_max: float = 4.0


def layer_saturation(
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
) -> Float[Array, "npoi nsoilay"]:
    """Liquid-water saturation of each layer, in [0, 1].

    `wsoi` is the liquid fraction of the pore space *not* occupied by ice, so
    the liquid share of the whole pore space is `wsoi * (1 - wisoi)`. Ice
    counts as unavailable, not as water: a frozen layer holds no water a
    methanogen can live in, and treating it as saturated would put a water
    table at the surface across the entire winter Arctic.
    """
    return jnp.clip(wsoi * (1.0 - wisoi), 0.0, 1.0)


def water_table_depth(
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    params: MethaneParams | None = None,
) -> Float[Array, "npoi"]:
    """Water table depth below the surface, m, from IBIS's own moisture profile.

    IBIS has no water table: it carries a moisture profile and nothing else
    (docs/METHANE.md 2.4). This integrates the *unsaturated* thickness from the
    top down, so a column saturated to the surface gives 0 and a bone-dry
    column gives the full depth:

        deficit = sum_k hsoi[k] * max(0, sat_crit - sat[k]) / sat_crit

    This is a diagnostic, not a prognostic water table -- it says how much
    unsaturated pore space sits above the wet zone, which is what the anaerobic
    fraction in M2 needs. It deliberately does not model lateral flow, perched
    tables, or drainage, none of which IBIS represents.

    Returns
    -------
    (npoi,) depth in metres, clipped to [0, z_max].
    """
    p = params or MethaneParams()
    sat = layer_saturation(wsoi, wisoi)
    deficit = jnp.sum(
        hsoi[None, :] * jnp.maximum(0.0, p.sat_crit - sat) / p.sat_crit, axis=1)
    return jnp.clip(deficit, 0.0, p.z_max)


def anaerobic_fraction(
    zwt: Float[Array, "npoi"],
    hsoi: Float[Array, "nsoilay"],
) -> Float[Array, "npoi nsoilay"]:
    """Fraction of each layer lying below the water table, in [0, 1].

    Linear within a layer, so a water table halfway through a layer makes it
    half anaerobic rather than snapping to 0 or 1. M2 multiplies production by
    this: methane is produced only below the water table.
    """
    top = jnp.concatenate([jnp.zeros(1), jnp.cumsum(hsoi)[:-1]])   # layer tops
    bot = jnp.cumsum(hsoi)
    below = bot[None, :] - jnp.maximum(zwt[:, None], top[None, :])
    return jnp.clip(below / hsoi[None, :], 0.0, 1.0)


def wetland_flux_area(
    f_inund: Float[Array, "npoi"],
    fixed_fraction: float = 0.0,
) -> Float[Array, "npoi"]:
    """The cell fraction methane production applies over, in [0, 1].

    Kept separate from the reader so the model never assumes a particular
    observational product, and so a sensitivity test can hold extent fixed.
    """
    return jnp.clip(jnp.maximum(f_inund, fixed_fraction), 0.0, 1.0)
