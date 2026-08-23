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
    balance_zeros: bool = True,
) -> Float[Array, ""]:
    """Mean squared error in log space, area-weighted.

    Log space because burned fraction spans orders of magnitude -- a plain MSE
    is dominated by a handful of savanna cells and learns nothing about the
    boreal zone. `eps` keeps zero-burn cells finite and sets the floor below
    which differences stop mattering.
    """
    pred = predict_burned_fraction(params, batch)
    resid = jnp.log(pred + eps) - jnp.log(observed + eps)

    if weights is None and balance_zeros:
        # The target is zero-inflated: in the first global fit roughly a third
        # of the 62176 retained cells never burn, and under a flat mean those
        # cells dominated -- the cheapest way to cut the loss was to predict
        # NOTHING ANYWHERE, which is what the optimiser did. Give the burning
        # and non-burning populations equal total weight so neither can drown
        # out the other.
        burns = (observed > 0.0).astype(pred.dtype)
        nb, nz = jnp.sum(burns), jnp.sum(1.0 - burns)
        weights = jnp.where(burns > 0,
                            0.5 / jnp.maximum(nb, 1.0),
                            0.5 / jnp.maximum(nz, 1.0))

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


# ---------------------------------------------------------------------------
# Constrained reparameterisation
# ---------------------------------------------------------------------------
#
# Three lessons from the global fits, in order of discovery:
#
# 1. UNCONSTRAINED FAILS. The first fit drove `k_b` NEGATIVE, inverting the
#    exponential. Sign must be enforced.
# 2. HALF-LINE FAILS TOO. Mapping through exp() bounds the sign but not the
#    magnitude, and the second fit ran `lfuel` to 50 kgC/m2 (real litter is
#    ~2.6) and the leaf weights into the thousands. Every parameter is mapped
#    to a PHYSICAL INTERVAL via a sigmoid instead.
# 3. SOME PARAMETERS ARE NOT IDENTIFIABLE and must not be fitted at all:
#
#    * `ig0` is exactly degenerate with `k_b` -- both multiply the same
#      exponent, so only their product is constrained. It is a normalisation
#      (the ignition factor where nobody lives) and is FIXED at 1.0.
#    * `d_crit` enters only through the accumulators `nfirestep`/`sdry`, which
#      are precomputed by the model run. Changing it cannot change a
#      prediction made from fixed accumulators, so its gradient is identically
#      zero here. Fitting it requires re-running the model per proposal.
#    * `mort` affects vegetation and emissions, not burned fraction (see
#      tests/test_fire_calibration.py).

#: Physically plausible interval for each fitted parameter.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "k_b":          (0.01, 20.0),
    "p":            (0.20, 5.0),
    "lfuel":        (0.01, 1.0),    # kgC m-2; real litter loads are O(0.1-3)
    "ufuel_span":   (0.05, 5.0),    # ufuel = lfuel + span, so ufuel > lfuel
    "w_leaf_woody": (0.0, 2.0),
    "w_leaf_grass": (0.0, 2.0),
    "a_ig":         (1e-4, 0.5),
    "ign_span":     (0.05, 9.0),    # ign_max = ig0 + span
}

#: Not fitted -- see the note above.
FIXED_PARAMS = ("ig0", "d_crit")

DEFAULT_FIT = ("k_b", "p")


def _to_u(v, lo, hi):
    z = jnp.clip((jnp.asarray(v, float) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    return jnp.log(z / (1.0 - z))


def _from_u(u, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(u)


def to_unconstrained(p: FireParams, fit_only=None) -> dict:
    """Map the fitted parameters into an unconstrained space."""
    names = tuple(fit_only) if fit_only is not None else DEFAULT_FIT
    u = {}
    for k in names:
        lo, hi = PARAM_BOUNDS[k]
        if k == "ufuel_span":
            u[k] = _to_u(p.ufuel - p.lfuel, lo, hi)
        elif k == "ign_span":
            u[k] = _to_u(p.ign_max - p.ig0, lo, hi)
        else:
            u[k] = _to_u(getattr(p, k), lo, hi)
    return u


def from_unconstrained(u: dict, template: FireParams) -> FireParams:
    """Map back; every value lands inside its physical interval by construction."""
    vals = {}
    for k, uu in u.items():
        lo, hi = PARAM_BOUNDS[k]
        vals[k] = _from_u(uu, lo, hi)
    out = template
    span_u = vals.pop("ufuel_span", None)
    span_i = vals.pop("ign_span", None)
    if vals:
        out = eqx.tree_at(lambda t: tuple(getattr(t, k) for k in vals),
                          out, tuple(vals.values()))
    if span_u is not None:
        out = eqx.tree_at(lambda t: t.ufuel, out, out.lfuel + span_u)
    if span_i is not None:
        out = eqx.tree_at(lambda t: t.ign_max, out, out.ig0 + span_i)
    return out


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
    balance_zeros: bool = True,
    fit_only=None,
) -> FitResult:
    """Fit the fire parameters by gradient descent.

    Returns the fitted parameters and the loss trajectory, so a caller can see
    whether it actually converged rather than assuming it did.
    """
    u = to_unconstrained(params0, fit_only)
    opt = optax.adam(learning_rate)
    state = opt.init(u)

    def loss_fn(uu):
        return fire_loss(from_unconstrained(uu, params0), batch, observed,
                         weights, balance_zeros=balance_zeros)

    @jax.jit
    def step(uu, st):
        loss, g = jax.value_and_grad(loss_fn)(uu)
        upd, st = opt.update(g, st)
        return optax.apply_updates(uu, upd), st, loss

    history = []
    for _ in range(steps):
        u, state, loss = step(u, state)
        history.append(float(loss))

    return FitResult(params=from_unconstrained(u, params0), loss_history=history)
