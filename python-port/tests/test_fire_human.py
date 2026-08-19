"""Stage F3: human ignition and suppression.

People both start fires and put them out. The absence of any human term is the
leading explanation for the Africa bias and the missed post-2000 declining
burned-area trend across CMIP6 models (docs/FIRE.md 2.2 defect 3).
"""

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.physics.fire import (
    FireParams, default_fire_params, fire_seasonlength, human_ignition,
    human_suppression,
)

EPS = float(np.finfo(np.float64).eps)
NPOI = 6


# ---------------------------------------------------------------------------
# Suppression -- verbatim CLM, so pin it against CLM's own expression
# ---------------------------------------------------------------------------

def test_suppression_matches_clm_expression():
    """`lpop = 0.2 + 0.8*exp(-pi*(hdm/450)**0.5)`, CNFireLi2014Mod.F90:439-440."""
    popd = np.array([0.0, 1.0, 7.5, 63.0, 450.0, 1e4])
    want = 0.2 + 0.8 * np.exp(-np.pi * np.sqrt(popd / 450.0))
    got = np.asarray(human_suppression(jnp.asarray(popd)))
    assert np.max(np.abs(got - want)) <= 4 * EPS


def test_suppression_bounds_and_monotonicity():
    """Bounded in [0.2, 1.0] by construction -- the reason `f_min` was dropped."""
    popd = jnp.asarray(np.linspace(0.0, 5e4, 400))
    s = np.asarray(human_suppression(popd))
    assert np.all(s >= 0.2 - 4 * EPS) and np.all(s <= 1.0 + 4 * EPS)
    assert np.all(np.diff(s) <= 4 * EPS)          # more people -> more suppression
    assert float(human_suppression(jnp.array([0.0]))[0]) == pytest.approx(1.0)


def test_suppression_has_no_free_parameters():
    """The 0.2 / 0.8 / 450 constants are CLM's, so FireParams must not shift them."""
    popd = jnp.asarray([0.0, 10.0, 500.0])
    a = np.asarray(human_suppression(popd, default_fire_params()))
    b = np.asarray(human_suppression(popd, FireParams(k_b=99.0, ig0=42.0)))
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Ignition -- deliberately NOT CLM's linear form
# ---------------------------------------------------------------------------

def test_ignition_is_bounded():
    """Saturating, unlike CLM's linear term.

    CLM can afford linear-in-population ignition because its fuel term collapses
    in dense urban cells. IBIS has no such collapse, so a linear term is
    unbounded: at 1e4 ind/km2 it gave a 7x *increase* in burning, the opposite
    of the observed decline at high density.
    """
    p = default_fire_params()
    popd = jnp.asarray(np.logspace(-2, 5, 300))
    ig = np.asarray(human_ignition(popd, p))
    assert np.all(ig >= p.ig0 - 4 * EPS)
    assert np.all(ig <= p.ign_max + 4 * EPS)
    assert np.all(np.diff(ig) >= -4 * EPS)
    assert float(human_ignition(jnp.array([0.0]), p)[0]) == pytest.approx(p.ig0)


def test_fire_persists_where_nobody_lives():
    """At popd = 0 the human factor must be exactly 1 -- lightning still ignites."""
    p = default_fire_params()
    z = jnp.zeros(3)
    assert np.allclose(np.asarray(human_ignition(z, p) * human_suppression(z)),
                       1.0, atol=4 * EPS)


def test_net_human_response_is_hump_shaped_and_bounded():
    """Rises where people ignite, falls where they suppress -- and stays bounded."""
    p = default_fire_params()
    popd = jnp.asarray(np.logspace(-2, 5, 500))
    net = np.asarray(human_ignition(popd, p) * human_suppression(popd))
    assert np.all(np.isfinite(net))
    assert np.all(net > 0.0)
    assert net.max() <= p.ign_max + 4 * EPS
    # dense population must suppress relative to empty land
    assert net[-1] < net[0]


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def _acc():
    return (jnp.full((NPOI,), 60.0), jnp.full((NPOI,), 48.0), jnp.array(100.0),
            jnp.full((NPOI,), 0.5))


def test_default_leaves_fire_climate_driven():
    assert default_fire_params().use_human is False
    nfs, sd, n, tl = _acc()
    a = np.asarray(fire_seasonlength(nfs, sd, n, tl, default_fire_params()))
    b = np.asarray(fire_seasonlength(nfs, sd, n, tl, default_fire_params(),
                                     popd=None))
    assert np.array_equal(a, b)


def test_population_changes_the_answer():
    p = FireParams(use_human=True)
    nfs, sd, n, tl = _acc()
    base = np.asarray(fire_seasonlength(nfs, sd, n, tl, p))
    dense = np.asarray(fire_seasonlength(nfs, sd, n, tl, p,
                                         popd=jnp.full((NPOI,), 2000.0)))
    empty = np.asarray(fire_seasonlength(nfs, sd, n, tl, p,
                                         popd=jnp.zeros(NPOI)))
    assert np.allclose(empty, base, atol=8 * EPS)   # popd=0 -> unchanged
    assert np.all(dense < base)                     # dense -> suppressed


def test_jittable_and_differentiable():
    p = FireParams(use_human=True)
    nfs, sd, n, tl = _acc()
    popd = jnp.full((NPOI,), 30.0)
    f = lambda pd: fire_seasonlength(nfs, sd, n, tl, p, popd=pd).sum()
    assert np.isfinite(float(eqx.filter_jit(f)(popd)))
    g = jax.grad(f)(popd)
    assert np.all(np.isfinite(np.asarray(g)))
