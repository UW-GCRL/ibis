"""Stage F4: per-PFT fire mortality.

IBIS applies `disturbf` identically to every PFT, so a savanna tree with thick
bark loses the same fraction of its biomass as a thin-barked boreal conifer in
a crown fire. That makes the fire->vegetation feedback -- the whole reason fire
belongs in a DGVM -- wrong in a way no amount of tuning the burned fraction can
repair.

Precedent: FATES resolves this as crown scorch plus cambial damage scaled by a
per-PFT `bark_scaler` (`SFMainMod.F90:570`). F4 collapses that to one
coefficient per PFT, which is the light-model version of the same idea.
"""

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.physics.fire import (
    DEFAULT_PFT_MORTALITY, FireParams, default_fire_params,
)

NPFT = 12


def test_one_coefficient_per_ibis_pft():
    assert len(DEFAULT_PFT_MORTALITY) == NPFT
    assert np.asarray(default_fire_params().mort).shape == (NPFT,)


def test_mortality_is_a_fraction():
    m = np.asarray(DEFAULT_PFT_MORTALITY)
    assert np.all(m > 0.0) and np.all(m <= 1.0)


def test_grasses_are_fully_top_killed():
    """PFTs 10-11 are the C4/C3 grasses (params.veg:37-38). Any fire that
    passes removes the whole above-ground part, so mortality is 1."""
    m = DEFAULT_PFT_MORTALITY
    assert m[10] == 1.0 and m[11] == 1.0


def test_ordering_boreal_conifer_above_fire_adapted_trees():
    """The defensible part of these priors is the ORDER, not the numbers.

    Boreal conifers (5, 7) carry thin bark and support stand-replacing crown
    fire. Temperate conifers (3) and tropical drought-deciduous trees (1) are
    the fire-adapted end -- thick bark, frequent low-intensity surface fire.
    """
    m = DEFAULT_PFT_MORTALITY
    assert m[5] > m[3], "boreal conifer should die more readily than temperate"
    assert m[5] > m[1], "boreal conifer should die more readily than savanna tree"
    assert m[1] < m[0], "drought-deciduous tropical tree is more fire-adapted"
    # woody PFTs must all be strictly less vulnerable than grass
    for j in range(10):
        assert m[j] < m[10]


def test_priors_are_overridable_for_calibration():
    """F6 must be able to fit these; they are priors, not constants."""
    custom = jnp.linspace(0.1, 1.0, NPFT)
    p = FireParams(mort=custom)
    assert np.allclose(np.asarray(p.mort), np.asarray(custom))


def test_mort_is_a_traced_leaf_not_static():
    """`mort` must be a pytree leaf so gradients can reach it in F6."""
    p = default_fire_params()
    leaves = jax.tree_util.tree_leaves(p)
    assert any(getattr(l, "shape", None) == (NPFT,) for l in leaves), (
        "mort should be a traced array leaf, not a static field")

    def loss(m):
        return jnp.sum(FireParams(mort=m).mort ** 2)

    g = jax.grad(loss)(jnp.asarray(DEFAULT_PFT_MORTALITY))
    assert np.all(np.isfinite(np.asarray(g)))


def test_per_pft_mortality_is_now_the_default():
    """Switched on 2026-08-26; uniform mortality stays reachable.

    Uniform mortality kills a fire-adapted savanna tree and a thin-barked
    boreal conifer at the same rate. The v8 spin-up put burned area within 8 %
    of GFED5 with uniform mortality still on, which left mortality as the
    largest remaining structural bias rather than a tuning detail.
    """
    assert default_fire_params().use_pft_mortality is True
    assert FireParams(use_pft_mortality=False).use_pft_mortality is False


# ---------------------------------------------------------------------------
# Effect on the disturbance applied to each pool
# ---------------------------------------------------------------------------

def _applied(disturbf, disturbo, mort_j, use):
    """Mirror of the expression in dynaveg's disturbance loop."""
    return disturbf * mort_j + disturbo if use else disturbf + disturbo


def test_background_disturbance_is_not_scaled():
    """`disturbo` is not fire, so per-PFT fire resistance must not touch it."""
    disturbf, disturbo = 0.20, 0.005
    grass = _applied(disturbf, disturbo, DEFAULT_PFT_MORTALITY[10], True)
    conif = _applied(disturbf, disturbo, DEFAULT_PFT_MORTALITY[3], True)
    # the difference between PFTs comes only from the fire part
    assert (grass - conif) == pytest.approx(
        disturbf * (DEFAULT_PFT_MORTALITY[10] - DEFAULT_PFT_MORTALITY[3]))
    # and with no fire at all, every PFT sees exactly the background rate
    for j in range(NPFT):
        assert _applied(0.0, disturbo, DEFAULT_PFT_MORTALITY[j], True) == disturbo


def test_resistant_pft_loses_less_biomass():
    disturbf, disturbo = 0.30, 0.005
    keep = lambda j: 1.0 - _applied(disturbf, disturbo, DEFAULT_PFT_MORTALITY[j], True)
    assert keep(3) > keep(5)      # temperate conifer keeps more than boreal
    assert keep(1) > keep(10)     # savanna tree keeps more than grass
    assert all(0.0 <= keep(j) <= 1.0 for j in range(NPFT))
