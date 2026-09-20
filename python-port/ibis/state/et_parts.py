"""Annual ET components (docs/CALIBRATION.md 15, Stage 2).

The port computes every F77 evaporation component each timestep -- upper and
lower transpiration, wet-canopy (interception) evaporation, soil evaporation,
snow sublimation -- and accumulates only their sum (`ayaet`). This optional
module, hung off IBISState after `stress`, accumulates the four groups in mm
so a run or a fit can decompose its ET. A state without it keeps its leaf
count; with it, `create_ibis_state` also attaches the (inert by default)
StressAccum so the trailing-leaf layouts form one chain (258 / 261 / 265) and
`load_state` migrates between them.
"""
from __future__ import annotations
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class ETParts(eqx.Module):
    trans: Float[Array, "npoi"]      # transpiration, upper + lower canopy (mm)
    interc: Float[Array, "npoi"]     # evaporation of intercepted water, leaves + stems (mm)
    soilev: Float[Array, "npoi"]     # soil evaporation (mm)
    snowsub: Float[Array, "npoi"]    # snow sublimation (mm)


FIELDS = ("trans", "interc", "soilev", "snowsub")


def create_et_parts(npoi: int) -> ETParts:
    z = jnp.zeros((npoi,))
    return ETParts(trans=z, interc=z, soilev=z, snowsub=z)


def reset_et_parts(state):
    if getattr(state, "etparts", None) is None:
        return state
    z = jnp.zeros_like(state.etparts.trans)
    return eqx.tree_at(lambda s: (s.etparts.trans, s.etparts.interc, s.etparts.soilev, s.etparts.snowsub),
                       state, (z, z, z, z))
