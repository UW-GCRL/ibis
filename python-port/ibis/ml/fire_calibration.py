"""Gradient-based calibration of the fire scheme (docs/FIRE.md stage F6).

This is the reason the JAX port was worth building. A Li-style fire scheme has
roughly a dozen empirical parameters that every other modelling group tunes by
hand, because their models are not differentiable. IBIS-JAX is differentiable
end to end, so the parameters can be *fitted* against observed burned area
instead.

No public JAX or Python implementation of any global fire scheme was found when
this was written -- JULES, ORCHIDEE, LPJ-GUESS and LPJ-WHyMe are all
non-public and every working implementation is Fortran or C. So this is, as far
as we can tell, the first differentiable one.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from ibis.physics.fire import (
    FireParams, fine_fuel_fraction, fire_seasonlength, legacy_fuel_fraction,
)


class FireBatch(NamedTuple):
    """Everything the fire scheme needs, for a set of cells and one year.

    Held as plain arrays so calibration does not require a full model state --
    these come from a completed run, or from a synthetic experiment.
    """

    nfirestep: Float[Array, "n"]      # timesteps above d_crit
    sdry: Float[Array, "n"]           # summed dryness over those timesteps
    nytimes: Float[Array, ""]         # timesteps in the year
    totlit: Float[Array, "n"]         # total litter, kgC m-2 (legacy fuel)
    totalit: Float[Array, "n"]        # above-ground litter, kgC m-2 (F2 fuel)
    cbiol: Float[Array, "n npft"]     # live leaf carbon, kgC m-2
    plai: Float[Array, "n npft"]      # per-PFT leaf area
    popd: Float[Array, "n"]           # population density, ind/km2


def predict_burned_fraction(
    params: FireParams, batch: FireBatch
) -> Float[Array, "n"]:
    """Burned fraction from parameters and inputs -- the function being fitted."""
    fuel = (fine_fuel_fraction(batch.totalit, batch.cbiol, params)
            if params.use_fine_fuel else legacy_fuel_fraction(batch.totlit))
    popd = batch.popd if params.use_human else None
    return fire_seasonlength(
        batch.nfirestep, batch.sdry, batch.nytimes, batch.totlit, params,
        fuel=fuel, popd=popd,
    )


def fire_loss(
    params: FireParams,
    batch: FireBatch,
    observed: Float[Array, "n"],
    weights: Float[Array, "n"] | None = None,
    eps: float = 1e-4,
) -> Float[Array, ""]:
    """Mean squared error in log space, area-weighted.

    Log space because burned fraction spans orders of magnitude -- a plain MSE
    is dominated by a handful of savanna cells and learns nothing about the
    boreal zone. `eps` keeps zero-burn cells finite and sets the floor below
    which differences stop mattering.
    """
    pred = predict_burned_fraction(params, batch)
    resid = jnp.log(pred + eps) - jnp.log(observed + eps)
    if weights is None:
        return jnp.mean(resid ** 2)
    w = weights / jnp.maximum(jnp.sum(weights), 1e-12)
    return jnp.sum(w * resid ** 2)


def _trainable(params: FireParams):
    """Split FireParams into fitted leaves and everything else.

    Static fields (the `use_*` switches) are structural choices, not parameters,
    and must not be differentiated.
    """
    return eqx.partition(params, eqx.is_inexact_array_like)


class FitResult(NamedTuple):
    params: FireParams
    loss_history: list[float]


def fit_fire_params(
    params0: FireParams,
    batch: FireBatch,
    observed: Float[Array, "n"],
    steps: int = 500,
    learning_rate: float = 1e-2,
    weights: Float[Array, "n"] | None = None,
) -> FitResult:
    """Fit the fire parameters by gradient descent.

    Returns the fitted parameters and the loss trajectory, so a caller can see
    whether it actually converged rather than assuming it did.
    """
    trainable, static = _trainable(params0)
    opt = optax.adam(learning_rate)
    state = opt.init(trainable)

    def loss_fn(tr):
        return fire_loss(eqx.combine(tr, static), batch, observed, weights)

    @jax.jit
    def step(tr, state):
        loss, grads = jax.value_and_grad(loss_fn)(tr)
        updates, state = opt.update(grads, state)
        return eqx.apply_updates(tr, updates), state, loss

    history = []
    for _ in range(steps):
        trainable, state, loss = step(trainable, state)
        history.append(float(loss))

    return FitResult(params=eqx.combine(trainable, static),
                     loss_history=history)
