"""Tests for tridiagonal solver."""

import jax
import jax.numpy as jnp
import pytest

from ibis.physics.solvers import tridia_batched


class TestTridiaBatched:
    """Batched Thomas algorithm tests."""

    def test_identity_system(self):
        """Identity matrix: solution should equal rhs."""
        npoi, nlay = 4, 6
        a = jnp.zeros((npoi, nlay))
        b = jnp.ones((npoi, nlay))
        c = jnp.zeros((npoi, nlay))
        rhs = jnp.arange(nlay, dtype=jnp.float32)[None, :].repeat(npoi, axis=0)
        x = tridia_batched(a, b, c, rhs)
        assert jnp.allclose(x, rhs, atol=1e-6)

    def test_known_3x3(self):
        """Known 3x3 system: 2x=b with off-diagonals of -1."""
        # [2, -1, 0] [x0]   [1]
        # [-1, 2,-1] [x1] = [0]
        # [0, -1, 2] [x2]   [1]
        # Solution: x = [1, 1, 1]
        npoi = 2
        a = jnp.array([[0.0, -1.0, -1.0]] * npoi)
        b = jnp.array([[2.0, 2.0, 2.0]] * npoi)
        c = jnp.array([[-1.0, -1.0, 0.0]] * npoi)
        rhs = jnp.array([[1.0, 0.0, 1.0]] * npoi)
        x = tridia_batched(a, b, c, rhs)
        expected = jnp.array([[1.0, 1.0, 1.0]] * npoi)
        assert jnp.allclose(x, expected, atol=1e-5)

    def test_vs_dense_solve(self):
        """Compare to dense jnp.linalg.solve for random system."""
        npoi, nlay = 3, 5
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        # Generate diagonally dominant system
        a = 0.1 * jax.random.normal(keys[0], (npoi, nlay))
        b = 3.0 + 0.5 * jax.random.normal(keys[1], (npoi, nlay))
        c = 0.1 * jax.random.normal(keys[2], (npoi, nlay))
        rhs = jax.random.normal(keys[3], (npoi, nlay))

        x_thomas = tridia_batched(a, b, c, rhs)

        # Build dense matrix and solve for each grid point
        for j in range(npoi):
            mat = jnp.diag(b[j])
            for k in range(1, nlay):
                mat = mat.at[k, k - 1].set(a[j, k])
            for k in range(nlay - 1):
                mat = mat.at[k, k + 1].set(c[j, k])
            x_dense = jnp.linalg.solve(mat, rhs[j])
            assert jnp.allclose(x_thomas[j], x_dense, atol=1e-4), \
                f"Grid point {j}: max diff = {float(jnp.max(jnp.abs(x_thomas[j] - x_dense)))}"

    def test_6_layer(self):
        """IBIS default: 6 soil layers."""
        npoi = 8
        nlay = 6
        # Diffusion-like system
        a = jnp.full((npoi, nlay), -0.3)
        b = jnp.full((npoi, nlay), 1.6)
        c = jnp.full((npoi, nlay), -0.3)
        rhs = jnp.ones((npoi, nlay))
        x = tridia_batched(a, b, c, rhs)
        assert x.shape == (npoi, nlay)
        assert jnp.all(jnp.isfinite(x))
        assert jnp.all(x > 0)

    def test_jit_compatible(self):
        """tridia must work under jax.jit."""
        a = jnp.zeros((2, 4))
        b = jnp.ones((2, 4)) * 2
        c = jnp.zeros((2, 4))
        rhs = jnp.ones((2, 4))
        x1 = tridia_batched(a, b, c, rhs)
        x2 = jax.jit(tridia_batched)(a, b, c, rhs)
        assert jnp.allclose(x1, x2, atol=1e-6)

    def test_differentiable(self):
        """tridia must be differentiable w.r.t. rhs."""
        a = jnp.zeros((2, 4))
        b = jnp.ones((2, 4)) * 2
        c = jnp.zeros((2, 4))

        def loss(rhs):
            return tridia_batched(a, b, c, rhs).sum()

        g = jax.grad(loss)(jnp.ones((2, 4)))
        assert jnp.all(jnp.isfinite(g))

    def test_single_system(self):
        """npoi=1 should work."""
        a = jnp.array([[0.0, -1.0, -1.0]])
        b = jnp.array([[2.0, 2.0, 2.0]])
        c = jnp.array([[-1.0, -1.0, 0.0]])
        rhs = jnp.array([[1.0, 0.0, 1.0]])
        x = tridia_batched(a, b, c, rhs)
        assert x.shape == (1, 3)
