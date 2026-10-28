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
    #: multi-year (exponentially smoothed) annual-mean upper-canopy stress, the cover limit's input (CALIBRATION.md
    #: 71); 0 = not yet set (a fresh state or a restart written before it existed: the first year's mean seeds it)
    u_smooth: Float[Array, "npoi"]


def create_stress_accum(npoi: int) -> StressAccum:
    z = jnp.zeros((npoi,))
    return StressAccum(aystressu=z, aystressl=z, nday=z, u_smooth=z)


def smoothed_stress(state, years=5.0):
    """This year's value of the multi-year upper-canopy stress: u_smooth + (u_year - u_smooth) / years, or u_year
    where u_smooth is unset (0). The same value `reset_stress_accum` stores, so dynaveg and the state agree."""
    u_year = state.stress.aystressu / jnp.maximum(state.stress.nday, 1.0)
    us = state.stress.u_smooth
    return jnp.where(us > 0.0, us + (u_year - us) / jnp.maximum(years, 1.0), u_year)


def reset_stress_accum(state, years=5.0):
    """End of year: store the smoothed stress, zero the annual accumulators."""
    if state.stress is None:
        return state
    z = jnp.zeros_like(state.stress.nday)
    us = smoothed_stress(state, years)
    return eqx.tree_at(lambda s: (s.stress.aystressu, s.stress.aystressl, s.stress.nday, s.stress.u_smooth),
                       state, (z, z, z, us))


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


def tree_cover_max(state, pft_params):
    """Water-limited maximum upper-canopy cover (CALIBRATION.md 70): clip((annual-mean upper-canopy stress -
    tree_cover_min) / tree_cover_ramp, 0, 1) ** tree_cover_power.

    None unless pft_params.tree_cover_limit is set and the accumulator is present, so every caller without the
    switch keeps the survival-factor path (tree_survival, on the single year's mean) bit-exact.
    """
    if state.stress is None or not bool(getattr(pft_params, 'tree_cover_limit', False)):
        return None
    # the multi-year stress (tree_cover_years, CALIBRATION.md 71): one dry year moves the limit by 1 / years of the
    # dip, so forest cells are neither capped nor eliminated by a single drought
    mean_u = smoothed_stress(state, getattr(pft_params, 'tree_cover_years', 5.0))
    x = jnp.clip((mean_u - pft_params.tree_cover_min) / pft_params.tree_cover_ramp, 0.0, 1.0)
    return x ** pft_params.tree_cover_power
