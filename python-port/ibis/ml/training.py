"""Differentiable training loop for hybrid physics+ML models.

Uses equinox for model management and optax for optimization.
Supports training through full simulation windows via JAX autodiff.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax


class TrainState(NamedTuple):
    """Training state container."""
    model: Any            # eqx.Module with trainable parameters
    opt_state: Any        # optax optimizer state
    step: int             # global step counter
    loss: float           # last loss value


def create_train_state(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Initialize a TrainState.

    Parameters
    ----------
    model : eqx.Module
        Model with trainable parameters.
    optimizer : optax.GradientTransformation
        Optax optimizer (e.g., optax.adam(1e-3)).

    Returns
    -------
    TrainState
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return TrainState(model=model, opt_state=opt_state, step=0, loss=0.0)


def train_step(
    train_state: TrainState,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    *loss_args,
    **loss_kwargs,
) -> TrainState:
    """Single training step: compute loss + gradients, update parameters.

    Parameters
    ----------
    train_state : TrainState
        Current training state.
    loss_fn : callable
        Function ``loss_fn(model, *loss_args, **loss_kwargs) -> scalar``.
        Must be differentiable w.r.t. the first argument (model).
    optimizer : optax.GradientTransformation
        Same optimizer used to create the train state.
    *loss_args, **loss_kwargs :
        Additional arguments forwarded to loss_fn.

    Returns
    -------
    TrainState
        Updated state with new model parameters and optimizer state.
    """
    model = train_state.model

    # Compute loss and gradients w.r.t. trainable parameters
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *loss_args, **loss_kwargs)

    # Apply optimizer update
    updates, new_opt_state = optimizer.update(
        grads, train_state.opt_state, eqx.filter(model, eqx.is_array),
    )
    new_model = eqx.apply_updates(model, updates)

    return TrainState(
        model=new_model,
        opt_state=new_opt_state,
        step=train_state.step + 1,
        loss=loss,
    )


def train_loop(
    model: eqx.Module,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    num_steps: int,
    data_iter: Any = None,
    *,
    log_every: int = 10,
    callback: Callable | None = None,
) -> TrainState:
    """Simple training loop.

    Parameters
    ----------
    model : eqx.Module
        Initial model.
    loss_fn : callable
        ``loss_fn(model, batch) -> scalar`` or ``loss_fn(model) -> scalar``
        if data_iter is None.
    optimizer : optax.GradientTransformation
        Optax optimizer.
    num_steps : int
        Number of training steps.
    data_iter : iterable, optional
        Iterator yielding batches.  If None, loss_fn is called with model only.
    log_every : int
        Print loss every N steps.
    callback : callable, optional
        ``callback(train_state)`` called every log_every steps.

    Returns
    -------
    TrainState
        Final training state.
    """
    state = create_train_state(model, optimizer)

    for step in range(num_steps):
        if data_iter is not None:
            batch = next(data_iter)
            state = train_step(state, loss_fn, optimizer, batch)
        else:
            state = train_step(state, loss_fn, optimizer)

        if (step + 1) % log_every == 0:
            if callback is not None:
                callback(state)

    return state


# ---------------------------------------------------------------------------
# Common loss functions
# ---------------------------------------------------------------------------

def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray,
             mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Mean squared error, optionally masked.

    Parameters
    ----------
    predictions, targets : arrays
        Same shape.
    mask : array, optional
        Boolean mask (True = include in loss).

    Returns
    -------
    scalar
    """
    diff_sq = (predictions - targets) ** 2
    if mask is not None:
        return jnp.sum(diff_sq * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.mean(diff_sq)


def mae_loss(predictions: jnp.ndarray, targets: jnp.ndarray,
             mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Mean absolute error, optionally masked."""
    diff_abs = jnp.abs(predictions - targets)
    if mask is not None:
        return jnp.sum(diff_abs * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.mean(diff_abs)
