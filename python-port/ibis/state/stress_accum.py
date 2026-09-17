"""Annual-mean soil-water stress accumulator (docs/CALIBRATION.md 14).

Hung off IBISState as an Optional field declared after `summary`, so a state
without it has exactly the leaf count it always had and every restart on disk
still loads. When present, `aystressu` / `aystressl` accumulate the daily
canopy stress factors (drystress, 1 = unstressed) and `nday` counts days;
`dynaveg` reads the annual mean to decide whether trees can survive.
"""
from __future__ import annotations
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class StressAccum(eqx.Module):
    aystressu: Float[Array, "npoi"]
    aystressl: Float[Array, "npoi"]
    nday: Float[Array, "npoi"]


def create_stress_accum(npoi: int) -> StressAccum:
    z = jnp.zeros((npoi,))
    return StressAccum(aystressu=z, aystressl=z, nday=z)


def reset_stress_accum(state):
    if state.stress is None:
        return state
    z = jnp.zeros_like(state.stress.nday)
    return eqx.tree_at(lambda s: (s.stress.aystressu, s.stress.aystressl, s.stress.nday),
                       state, (z, z, z))


def tree_survival(state, pft_params):
    """clip((annual-mean upper-canopy stress - tree_stress_min) / ramp, 0, 1).

    None when the accumulator is absent, so callers pass nothing to dynaveg
    and default behaviour is untouched. tree_stress_min = -1 (default) gives
    survival 1 everywhere even when the accumulator is present.
    """
    if state.stress is None:
        return None
    mean_u = state.stress.aystressu / jnp.maximum(state.stress.nday, 1.0)
    return jnp.clip((mean_u - pft_params.tree_stress_min) / pft_params.tree_stress_ramp, 0.0, 1.0)
