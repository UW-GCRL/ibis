"""Tests for state pytree creation and JAX compatibility."""

import jax
import jax.numpy as jnp
import pytest

from ibis.dimensions import GridConfig
from ibis.state.model_state import IBISState, create_ibis_state
from ibis.state.soil import create_soil_state
from ibis.state.snow import create_snow_state


class TestSoilState:
    """SoilState creation and pytree registration."""

    def test_create(self):
        soil = create_soil_state(npoi=4, nsoilay=6)
        assert soil.tsoi.shape == (4, 6)
        assert soil.wsoi.shape == (4, 6)
        assert soil.tg.shape == (4,)
        assert soil.hsoi.shape == (7,)  # nsoilay + 1

    def test_default_values(self):
        soil = create_soil_state(npoi=4, nsoilay=6)
        assert jnp.allclose(soil.tsoi, 278.0)
        assert jnp.allclose(soil.wsoi, 0.5)
        assert jnp.allclose(soil.wisoi, 0.0)

    def test_pytree_leaves(self):
        """SoilState should be a valid JAX pytree."""
        soil = create_soil_state(npoi=4, nsoilay=6)
        leaves = jax.tree.leaves(soil)
        assert len(leaves) > 0
        assert all(isinstance(l, jnp.ndarray) for l in leaves)


class TestSnowState:
    """SnowState creation."""

    def test_create(self):
        snow = create_snow_state(npoi=4, nsnolay=3)
        assert snow.fi.shape == (4,)
        assert snow.tsno.shape == (4, 3)
        assert snow.hsno.shape == (4, 3)

    def test_default_values(self):
        snow = create_snow_state(npoi=4, nsnolay=3)
        assert jnp.allclose(snow.fi, 0.0)
        assert jnp.allclose(snow.tsno, 273.16)


class TestIBISState:
    """Top-level IBISState creation and JAX operations."""

    def test_create(self, grid):
        state = create_ibis_state(grid)
        assert state.soil.tsoi.shape == (grid.npoi, grid.nsoilay)
        assert state.snow.fi.shape == (grid.npoi,)
        assert state.canopy.tu.shape == (grid.npoi,)
        assert state.atm.ta.shape == (grid.npoi,)
        assert state.vegetation.plai.shape == (grid.npoi, grid.npft)
        assert state.carbon.clitlm.shape == (grid.npoi,)
        assert state.hydrology.ginvap.shape == (grid.npoi,)

    def test_optional_bgc_none(self, grid):
        state = create_ibis_state(grid)
        assert state.permafrost is None
        assert state.wetland is None
        assert state.fire is None
        assert state.bvoc is None

    def test_pytree_compatible(self, grid):
        """IBISState must be a valid JAX pytree for jit/vmap/grad."""
        state = create_ibis_state(grid)
        leaves = jax.tree.leaves(state)
        assert len(leaves) > 0

    def test_tree_map(self, grid):
        """jax.tree.map should work on IBISState."""
        state = create_ibis_state(grid)
        # Double all array values
        doubled = jax.tree.map(lambda x: x * 2.0, state)
        assert jnp.allclose(doubled.soil.tsoi, state.soil.tsoi * 2.0)

    def test_immutable(self, grid):
        """IBISState should be frozen (equinox.Module)."""
        state = create_ibis_state(grid)
        with pytest.raises(AttributeError):
            state.soil = None

    def test_jit_passthrough(self, grid):
        """IBISState should pass through jax.jit."""
        state = create_ibis_state(grid)

        @jax.jit
        def get_mean_tsoi(s):
            return jnp.mean(s.soil.tsoi)

        result = get_mean_tsoi(state)
        assert jnp.isfinite(result)
        assert float(result) == pytest.approx(278.0, rel=1e-5)
