"""PhysicsComponent protocol for IBIS-JAX hybrid physics+ML.

Defines the abstract interface that all physics modules (and their ML
replacements) must satisfy.  Works with both Equinox and Flax/NNX models.

Design principle: any function `f(state, params, ...) -> state` can be
wrapped as a PhysicsComponent.  The registry then lets you swap the
default physics for an ML surrogate at runtime.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


@runtime_checkable
class PhysicsComponent(Protocol):
    """Protocol for swappable physics components.

    Any callable (function, eqx.Module, flax.linen.Module) that maps
    (inputs, params) -> outputs satisfies this protocol.

    The exact signature of __call__ is intentionally flexible:
    components receive a dict of input arrays and return a dict of
    output arrays.  This keeps the interface uniform across very
    different physics modules (stomata, radiation, soil BGC, ...).
    """

    def __call__(
        self,
        inputs: dict[str, Any],
        params: Any = None,
    ) -> dict[str, Any]:
        """Run the component.

        Parameters
        ----------
        inputs : dict
            Named input arrays (e.g., temperature, moisture, PAR).
        params : any, optional
            Static parameters (StomataParams, BGCParams, etc.).

        Returns
        -------
        dict
            Named output arrays.
        """
        ...


class WrappedFunction(eqx.Module):
    """Wrap a plain function as a PhysicsComponent.

    Useful for wrapping existing IBIS physics functions so they can
    participate in the registry/swap system without modification.

    Example
    -------
    >>> from ibis.physics.stomata import stomata
    >>> component = WrappedFunction(stomata)
    >>> out = component(inputs, params)
    """

    fn: Any = eqx.field(static=True)

    def __call__(self, inputs: dict[str, Any], params: Any = None) -> dict[str, Any]:
        return self.fn(inputs, params)


class Identity(eqx.Module):
    """Pass-through component that returns inputs unchanged.

    Useful as a default/placeholder or for testing.
    """

    def __call__(self, inputs: dict[str, Any], params: Any = None) -> dict[str, Any]:
        return inputs
