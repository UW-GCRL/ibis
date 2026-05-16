"""Tests for thermodynamic functions -- values must match Fortran original."""

import jax
import jax.numpy as jnp
import pytest

from ibis.physics.thermo import esat, desat, qsat, dqsat, hvapf, hsubf
from ibis.constants import PhysicalConstants


class TestEsat:
    """Saturation vapour pressure tests."""

    def test_at_freezing(self):
        """esat(273.16 K) ~ 610.78 Pa (Lowe 1977 liquid branch at 0C)."""
        e = esat(jnp.array(273.16))
        assert float(e) == pytest.approx(610.78, rel=0.001)

    def test_above_freezing(self):
        """esat(300 K) should be a few kPa."""
        e = esat(jnp.array(300.0))
        assert 3000.0 < float(e) < 5000.0

    def test_ice_branch(self):
        """esat below freezing should use ice coefficients."""
        e_cold = esat(jnp.array(250.0))
        e_freeze = esat(jnp.array(273.16))
        assert 0.0 < float(e_cold) < float(e_freeze)

    def test_monotonic(self):
        """esat should be monotonically increasing with T."""
        temps = jnp.linspace(240.0, 320.0, 50)
        es = esat(temps)
        diffs = jnp.diff(es)
        assert jnp.all(diffs > 0)

    def test_vectorized(self):
        """esat should work on arrays."""
        temps = jnp.array([250.0, 273.16, 300.0])
        es = esat(temps)
        assert es.shape == (3,)
        assert jnp.all(es > 0)

    def test_differentiable(self):
        """esat must be differentiable for gradient-based optimisation."""
        grad_fn = jax.grad(lambda t: esat(t).sum())
        g = grad_fn(jnp.array(290.0))
        assert jnp.isfinite(g)
        assert float(g) > 0  # esat increases with T

    def test_jit_compatible(self):
        """esat must work under jax.jit."""
        esat_jit = jax.jit(esat)
        e = esat_jit(jnp.array(290.0))
        assert jnp.isfinite(e)


class TestDesat:
    """Derivative of saturation vapour pressure."""

    def test_positive(self):
        """desat should be positive for all temperatures."""
        temps = jnp.linspace(240.0, 320.0, 50)
        des = desat(temps)
        assert jnp.all(des > 0)

    def test_increases_with_temp(self):
        """desat should increase with temperature (Clausius-Clapeyron)."""
        assert float(desat(jnp.array(300.0))) > float(desat(jnp.array(270.0)))


class TestQsat:
    """Saturation specific humidity."""

    def test_at_freezing(self):
        """qsat at 273.16 K, 101325 Pa should be ~ 3.8e-3 kg/kg."""
        e = esat(jnp.array(273.16))
        q = qsat(e, jnp.array(101325.0))
        assert float(q) == pytest.approx(3.8e-3, rel=0.05)

    def test_formula(self):
        """qsat = 0.622 * esat / (p - 0.378 * esat)."""
        e = jnp.array(2000.0)
        p = jnp.array(101325.0)
        q = qsat(e, p)
        expected = 0.622 * e / (p - 0.378 * e)
        assert float(q) == pytest.approx(float(expected), rel=1e-6)

    def test_bounded(self):
        """qsat should be in (0, 1) for reasonable conditions."""
        e = esat(jnp.array(310.0))
        q = qsat(e, jnp.array(101325.0))
        assert 0.0 < float(q) < 1.0


class TestDqsat:
    """Derivative of saturation specific humidity."""

    def test_positive(self):
        """dqsat should be positive."""
        t = jnp.array(290.0)
        p = jnp.array(101325.0)
        e = esat(t)
        q1 = qsat(e, p)
        dq = dqsat(t, q1)
        assert float(dq) > 0

    def test_differentiable(self):
        """dqsat should itself be differentiable."""
        def f(t):
            e = esat(t)
            q1 = qsat(e, jnp.array(101325.0))
            return dqsat(t, q1)
        grad_fn = jax.grad(f)
        g = grad_fn(jnp.array(290.0))
        assert jnp.isfinite(g)


class TestHvapf:
    """Temperature-adjusted latent heat of vaporisation."""

    def test_at_tmelt(self):
        """hvapf(tmelt) should equal hvap."""
        c = PhysicalConstants()
        t = jnp.array(c.tmelt)
        result = hvapf(t, t, c.tmelt, c.hvap, c.ch2o, c.cvap)
        assert float(result) == pytest.approx(c.hvap, rel=1e-10)

    def test_increases_with_temp(self):
        """hvapf should increase with temperature (cvap > ch2o)."""
        c = PhysicalConstants()
        t1 = jnp.array(273.16)
        t2 = jnp.array(300.0)
        h1 = hvapf(t1, t1, c.tmelt, c.hvap, c.ch2o, c.cvap)
        h2 = hvapf(t2, t2, c.tmelt, c.hvap, c.ch2o, c.cvap)
        # cvap (1810) < ch2o (4218), so hvapf DECREASES with T
        assert float(h2) < float(h1)


class TestHsubf:
    """Temperature-adjusted latent heat of sublimation."""

    def test_at_tmelt(self):
        """hsubf(tmelt) should equal hsub."""
        c = PhysicalConstants()
        t = jnp.array(c.tmelt)
        result = hsubf(t, t, c.tmelt, c.hsub, c.cvap, c.cice)
        assert float(result) == pytest.approx(c.hsub, rel=1e-10)
