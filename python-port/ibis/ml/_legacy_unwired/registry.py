"""Component registry for runtime physics/ML swapping.

The registry maps component names (e.g., 'stomata', 'radiation', 'soilbgc')
to callable components.  At simulation time, the timestep loop queries
the registry for each component, falling back to the default physics if
no override is registered.

Example
-------
>>> reg = ComponentRegistry()
>>> reg.register('stomata', my_neural_stomata)
>>> component = reg.get('stomata', default=physics_stomata)
>>> out = component(inputs, params)
"""

from __future__ import annotations

from typing import Any


class ComponentRegistry:
    """Runtime registry for swapping physics components with ML surrogates.

    Thread-safe for read operations (get).  Registration should happen
    before the simulation starts.

    Attributes
    ----------
    _components : dict
        Maps component name -> callable.
    """

    def __init__(self):
        self._components: dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        """Register a component (overwriting any previous entry).

        Parameters
        ----------
        name : str
            Component name (e.g., 'stomata', 'radiation').
        component : callable
            The component to register.
        """
        self._components[name] = component

    def get(self, name: str, default: Any = None) -> Any:
        """Get a registered component, or fall back to default.

        Parameters
        ----------
        name : str
            Component name.
        default : callable, optional
            Fallback if name is not registered.

        Returns
        -------
        callable
            The registered component or default.
        """
        return self._components.get(name, default)

    def unregister(self, name: str) -> None:
        """Remove a component from the registry."""
        self._components.pop(name, None)

    def list_components(self) -> list[str]:
        """Return names of all registered components."""
        return list(self._components.keys())

    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components

    def clear(self) -> None:
        """Remove all registered components."""
        self._components.clear()

    def __repr__(self) -> str:
        names = ", ".join(self._components.keys())
        return f"ComponentRegistry([{names}])"
