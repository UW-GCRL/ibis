"""Neural and hybrid physics wrappers for IBIS-JAX.

Provides composable wrappers that let you:
  1. Replace a physics component entirely with a neural network (NeuralPhysics)
  2. Add a learned correction to existing physics (HybridPhysics)
  3. Blend two components with a learnable weight (BlendedPhysics)

All wrappers are equinox Modules (frozen dataclasses, JAX pytrees),
so they work seamlessly with jit, grad, and vmap.
"""

from __future__ import annotations

from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# Helper: simple MLP that operates on a concatenated input vector
# ---------------------------------------------------------------------------

class MLP(eqx.Module):
    """Simple multi-layer perceptron.

    Operates on flattened input features and produces an output vector.
    All layers use tanh activation except the last (linear).
    """

    layers: tuple[eqx.nn.Linear, ...]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: Sequence[int] = (64, 64),
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        sizes = [in_features] + list(hidden_sizes) + [out_features]
        self.layers = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        )

    def __call__(self, x: Float[Array, "features"]) -> Float[Array, "out"]:
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Internal: prepare features from inputs dict
# ---------------------------------------------------------------------------

def _prepare_features(
    inputs: dict[str, Any],
    input_keys: tuple[str, ...],
    input_sizes: tuple[int, ...],
) -> tuple[jnp.ndarray, bool]:
    """Build feature array from named inputs.

    Uses input_sizes to distinguish single-point from batched inputs:
      - {"x": shape (3,)} with input_sizes=(3,) → single point, features (3,)
      - {"t": shape (5,)} with input_sizes=(1,) → 5 points, features (5, 1)

    Returns (features, is_batched).
    """
    n_in = sum(input_sizes)
    parts = []
    for k, sz in zip(input_keys, input_sizes):
        v = jnp.asarray(inputs[k])
        if v.ndim == 0:
            # Scalar → single point feature
            parts.append(v.reshape(1))
        elif v.ndim == 1:
            if sz == v.shape[0]:
                # Single point with sz features
                parts.append(v)
            else:
                # Batched: (npoi,) with sz=1 → (npoi, 1)
                parts.append(v.reshape(-1, 1))
        elif v.ndim == 2:
            # Already (npoi, features)
            parts.append(v)
        else:
            parts.append(v.reshape(-1, v.shape[-1]))

    # Check if batched: any part is 2D
    is_batched = any(p.ndim == 2 for p in parts)

    if is_batched:
        # Make all parts 2D
        aligned = []
        for p in parts:
            if p.ndim == 1:
                # Broadcast single-point features across batch
                # This shouldn't normally happen in practice
                aligned.append(jnp.broadcast_to(p, (1, p.shape[0])))
            else:
                aligned.append(p)
        features = jnp.concatenate(aligned, axis=-1)
    else:
        features = jnp.concatenate(parts, axis=-1)

    return features, is_batched


def _apply_net(net, features, is_batched):
    """Apply network, vmapping if batched."""
    if is_batched:
        return jax.vmap(net)(features)
    else:
        return net(features)


def _split_outputs(
    out: jnp.ndarray,
    output_keys: tuple[str, ...],
    output_sizes: tuple[int, ...],
    is_batched: bool,
) -> dict[str, Any]:
    """Split network output into named arrays."""
    splits = jnp.split(out, jnp.cumsum(jnp.array(output_sizes[:-1])), axis=-1)
    result = {}
    for key, arr in zip(output_keys, splits):
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        result[key] = arr
    return result


# ---------------------------------------------------------------------------
# NeuralPhysics: full NN replacement
# ---------------------------------------------------------------------------

class NeuralPhysics(eqx.Module):
    """Replace a physics component entirely with a neural network.

    Maps named input arrays to named output arrays via an MLP.
    Input arrays are concatenated into a feature vector per grid point;
    the MLP output is split back into named outputs.

    Parameters
    ----------
    net : eqx.Module
        Neural network mapping (n_in,) -> (n_out,).
    input_keys : tuple of str
        Keys to extract from the inputs dict.
    input_sizes : tuple of int
        Expected per-point dimension for each input key.
    output_keys : tuple of str
        Keys for the outputs dict.
    output_sizes : tuple of int
        Size of each output.
    """

    net: eqx.Module
    input_keys: tuple[str, ...] = eqx.field(static=True)
    input_sizes: tuple[int, ...] = eqx.field(static=True)
    output_keys: tuple[str, ...] = eqx.field(static=True)
    output_sizes: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, inputs: dict[str, Any], params: Any = None) -> dict[str, Any]:
        features, batched = _prepare_features(inputs, self.input_keys, self.input_sizes)
        out = _apply_net(self.net, features, batched)
        return _split_outputs(out, self.output_keys, self.output_sizes, batched)


# ---------------------------------------------------------------------------
# HybridPhysics: physics + learnable correction
# ---------------------------------------------------------------------------

class HybridPhysics(eqx.Module):
    """Hybrid model: physics output + alpha * NN correction.

    output = physics(inputs, params) + alpha * correction_net(inputs)

    The blending weight alpha is a learnable scalar (initialized to 0),
    so the model starts as pure physics and gradually learns corrections.

    Parameters
    ----------
    physics_fn : callable
        The original physics component.
    correction_net : eqx.Module
        Neural network producing corrections.
    alpha : array, scalar
        Learnable blending weight (initialised to 0.0).
    input_keys : tuple of str
        Keys to feed to the correction network.
    input_sizes : tuple of int
        Expected per-point dimension for each input key.
    output_keys : tuple of str
        Keys that the correction applies to.
    output_sizes : tuple of int
        Size of each corrected output.
    """

    physics_fn: Any = eqx.field(static=True)
    correction_net: eqx.Module
    alpha: Float[Array, ""]
    input_keys: tuple[str, ...] = eqx.field(static=True)
    input_sizes: tuple[int, ...] = eqx.field(static=True)
    output_keys: tuple[str, ...] = eqx.field(static=True)
    output_sizes: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, inputs: dict[str, Any], params: Any = None) -> dict[str, Any]:
        # Run physics
        phys_out = self.physics_fn(inputs, params)

        # Build and apply correction
        features, batched = _prepare_features(inputs, self.input_keys, self.input_sizes)
        correction_raw = _apply_net(self.correction_net, features, batched)
        correction = _split_outputs(
            correction_raw, self.output_keys, self.output_sizes, batched
        )

        # Add scaled correction to physics output
        result = dict(phys_out)
        for key in self.output_keys:
            corr_val = correction[key]
            if key in result:
                result[key] = result[key] + self.alpha * corr_val
            else:
                result[key] = self.alpha * corr_val
        return result


# ---------------------------------------------------------------------------
# BlendedPhysics: weighted average of two components
# ---------------------------------------------------------------------------

class BlendedPhysics(eqx.Module):
    """Blend two physics components: w * comp_a + (1-w) * comp_b.

    Parameters
    ----------
    comp_a, comp_b : callable
        Two physics components producing the same output keys.
    weight_logit : array, scalar
        Learnable blending weight in [0, 1] (applied via sigmoid).
    blend_keys : tuple of str
        Output keys to blend (others taken from comp_a).
    """

    comp_a: Any
    comp_b: Any
    weight_logit: Float[Array, ""]
    blend_keys: tuple[str, ...] = eqx.field(static=True)

    def __call__(self, inputs: dict[str, Any], params: Any = None) -> dict[str, Any]:
        out_a = self.comp_a(inputs, params)
        out_b = self.comp_b(inputs, params)

        w = jax.nn.sigmoid(self.weight_logit)
        result = dict(out_a)
        for key in self.blend_keys:
            if key in out_a and key in out_b:
                result[key] = w * out_a[key] + (1.0 - w) * out_b[key]
        return result


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_neural_physics(
    input_keys: tuple[str, ...],
    input_sizes: tuple[int, ...],
    output_keys: tuple[str, ...],
    output_sizes: tuple[int, ...],
    hidden_sizes: Sequence[int] = (64, 64),
    *,
    key: jax.Array,
) -> NeuralPhysics:
    """Create a NeuralPhysics wrapper with a fresh MLP.

    Parameters
    ----------
    input_keys : tuple of str
        Names of input arrays.
    input_sizes : tuple of int
        Per-point dimension of each input (concatenated to form feature vector).
    output_keys : tuple of str
        Names of output arrays.
    output_sizes : tuple of int
        Dimension of each output.
    hidden_sizes : sequence of int
        Hidden layer sizes for the MLP.
    key : PRNGKey
        Random key for weight initialisation.
    """
    n_in = sum(input_sizes)
    n_out = sum(output_sizes)
    net = MLP(n_in, n_out, hidden_sizes, key=key)
    return NeuralPhysics(
        net=net,
        input_keys=input_keys,
        input_sizes=input_sizes,
        output_keys=output_keys,
        output_sizes=output_sizes,
    )


def make_hybrid_physics(
    physics_fn,
    input_keys: tuple[str, ...],
    input_sizes: tuple[int, ...],
    output_keys: tuple[str, ...],
    output_sizes: tuple[int, ...],
    hidden_sizes: Sequence[int] = (64, 64),
    *,
    key: jax.Array,
) -> HybridPhysics:
    """Create a HybridPhysics wrapper with a fresh correction MLP.

    Parameters
    ----------
    physics_fn : callable
        Original physics component.
    input_keys, input_sizes, output_keys, output_sizes :
        Same as make_neural_physics.
    hidden_sizes : sequence of int
        Hidden layer sizes for the correction MLP.
    key : PRNGKey
        Random key.
    """
    n_in = sum(input_sizes)
    n_out = sum(output_sizes)
    net = MLP(n_in, n_out, hidden_sizes, key=key)
    return HybridPhysics(
        physics_fn=physics_fn,
        correction_net=net,
        alpha=jnp.array(0.0),
        input_keys=input_keys,
        input_sizes=input_sizes,
        output_keys=output_keys,
        output_sizes=output_sizes,
    )
