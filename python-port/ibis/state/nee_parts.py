"""Day-time and night-time NEE (NEE calibration stage, CALIBRATION.md 36).

The tower measures NEE; its night-time part is ecosystem respiration directly. The port accumulates NPP per hourly step
(stats.f sumnow) and heterotrophic respiration per day (soilbgc), so NEE at any hour is Rh/steps_per_day - NPP_step
with Rh spread evenly over the day (the model's own resolution of Rh; no diurnal cycle is imposed on it). This optional
module, hung off IBISState after `seasonal`, accumulates that NEE (kgC m-2, positive = release) separately for the
steps with the sun below the horizon (coszen <= 0, the FLUXNET NIGHT flag) and the others. A state without it keeps its
leaf count; with it, `create_ibis_state` also attaches the earlier optional modules so the trailing-leaf layouts form
one chain (258 / 262 / 266 / 269 / 271) and `load_state` migrates between them.
"""
from __future__ import annotations
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class NEEParts(eqx.Module):
    night: Float[Array, "npoi"]      # NEE over the night-time steps (kgC m-2, + = release)
    day: Float[Array, "npoi"]        # NEE over the day-time steps


FIELDS = ("night", "day")


def create_nee_parts(npoi: int) -> NEEParts:
    z = jnp.zeros((npoi,))
    return NEEParts(night=z, day=z)


def reset_nee_parts(state):
    if getattr(state, "neeparts", None) is None:
        return state
    z = jnp.zeros_like(state.neeparts.night)
    return eqx.tree_at(lambda s: (s.neeparts.night, s.neeparts.day), state, (z, z))
