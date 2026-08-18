"""Stage F2: fuel that can actually burn.

Two corrections over the legacy fuel term, both found by the docs/FIRE.md
audits rather than from the literature:

1. Legacy uses `totlit`, which `biogeochem.f:924` defines as "total above and
   belowground litter" -- it includes three root-litter pools that cannot burn.
   `totalit` (`:930`) is the above-ground sum. LPJmL corroborates: `firepft.c`
   burns only `agtop` litter.
2. The first draft of the design multiplied live leaf carbon by `frac`. That is
   wrong: `cbiol` is already per grid-cell area, and `frac` is a *canopy*
   fraction (comveg.h:197), not a grid fraction.
"""

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.physics.fire import (
    FireParams, default_fire_params, fine_fuel_fraction, legacy_fuel_fraction,
    fire_seasonlength,
)

EPS = float(np.finfo(np.float64).eps)
NPOI, NPFT = 4, 12
WOODY, GRASS = 3, 11        # a tree PFT and a grass PFT


def _cbiol(**kw):
    c = jnp.zeros((NPOI, NPFT))
    for idx, val in kw.items():
        c = c.at[:, int(idx[1:])].set(val)
    return c


# ---------------------------------------------------------------------------
# Root litter must not burn
# ---------------------------------------------------------------------------

def test_root_litter_is_excluded():
    """Fuel depends on above-ground litter only.

    Two cells with the same above-ground litter but very different root litter
    must give identical fuel. Under legacy they would differ, because `totlit`
    lumps them together.
    """
    totalit = jnp.full((NPOI,), 0.30)
    fuel = np.asarray(fine_fuel_fraction(totalit, jnp.zeros((NPOI, NPFT)),
                                         default_fire_params()))
    assert np.allclose(fuel, fuel[0], atol=0)      # no dependence on anything else

    # Legacy, given totlit = totalit + root litter, inflates with root carbon.
    lo = float(legacy_fuel_fraction(jnp.array([0.30]))[0])   # no roots
    hi = float(legacy_fuel_fraction(jnp.array([0.30 + 0.25]))[0])  # + roots
    assert hi >= lo, "precondition: legacy fuel is non-decreasing in totlit"
    assert float(legacy_fuel_fraction(jnp.array([0.30]))[0]) == 1.0, (
        "legacy saturates at only 0.2 kgC/m2, so root litter is often invisible "
        "there too -- the correctness issue stands regardless")


# ---------------------------------------------------------------------------
# No frac weighting; PFT grouping
# ---------------------------------------------------------------------------

def test_live_leaf_is_summed_unweighted_across_pfts():
    """cbiol is already per grid-cell area, so PFTs sum directly.

    Splitting the same total leaf carbon across many PFTs must not change the
    answer. If a `frac` weighting had been applied, it would.
    """
    p = FireParams(w_leaf_woody=0.5, w_leaf_grass=0.5)
    totalit = jnp.full((NPOI,), 0.05)

    one = _cbiol(p0=0.40)
    spread = jnp.zeros((NPOI, NPFT))
    for j in range(8):
        spread = spread.at[:, j].set(0.05)          # same 0.40 total
    a = np.asarray(fine_fuel_fraction(totalit, one, p))
    b = np.asarray(fine_fuel_fraction(totalit, spread, p))
    assert np.allclose(a, b, atol=8 * EPS)


def test_grass_leaf_counts_more_than_woody_leaf():
    p = default_fire_params()
    totalit = jnp.full((NPOI,), 0.10)
    woody = np.asarray(fine_fuel_fraction(totalit, _cbiol(p3=0.20), p))
    grass = np.asarray(fine_fuel_fraction(totalit, _cbiol(p11=0.20), p))
    assert np.all(grass > woody)


def test_pft_grouping_matches_ibis_vegmap():
    """PFTs 0-9 are woody (8-9 are shrubs), 10-11 are grasses.

    Mirrors IBIS's own grouping in `vegmap`, where shrublai = plai[8]+plai[9]
    and grasslai = plai[10]+plai[11].
    """
    p = FireParams(w_leaf_woody=0.0, w_leaf_grass=1.0)
    totalit = jnp.zeros((NPOI,))
    for j in range(10):                       # woody -> ignored at w_woody=0
        f = np.asarray(fine_fuel_fraction(totalit, _cbiol(**{f"p{j}": 1.0}), p))
        assert np.allclose(f, 0.0, atol=8 * EPS), f"PFT {j} should be woody"
    for j in (10, 11):                        # grass -> counted
        f = np.asarray(fine_fuel_fraction(totalit, _cbiol(**{f"p{j}": 1.0}), p))
        assert np.all(f > 0.0), f"PFT {j} should be grass"


# ---------------------------------------------------------------------------
# The ramp
# ---------------------------------------------------------------------------

def test_ramp_endpoints_and_monotonicity():
    p = default_fire_params()
    z = jnp.zeros((NPOI, NPFT))
    below = np.asarray(fine_fuel_fraction(jnp.full((NPOI,), p.lfuel * 0.5), z, p))
    at_lo = np.asarray(fine_fuel_fraction(jnp.full((NPOI,), p.lfuel), z, p))
    at_hi = np.asarray(fine_fuel_fraction(jnp.full((NPOI,), p.ufuel), z, p))
    above = np.asarray(fine_fuel_fraction(jnp.full((NPOI,), p.ufuel * 3), z, p))
    assert np.allclose(below, 0.0) and np.allclose(at_lo, 0.0)
    assert np.allclose(at_hi, 1.0) and np.allclose(above, 1.0)

    ladder = jnp.linspace(0.0, 1.0, 50)
    vals = np.asarray(fine_fuel_fraction(ladder, jnp.zeros((50, NPFT)), p))
    assert np.all(np.diff(vals) >= -8 * EPS)
    assert np.all((vals >= 0.0) & (vals <= 1.0))


def test_thresholds_are_in_kg_not_grams():
    """Guards the 1000x unit error the second audit caught.

    CTSM gives lfuel = 75, ufuel = 650 gC m-2. IBIS carries litter in kg_C m-2
    (comveg.h:132,137), so the stored values must be the CONVERTED ones. If
    someone pastes CLM's raw numbers back in, `f_fuel` clips to 0 everywhere
    and fire switches off globally -- silently.
    """
    p = default_fire_params()
    assert p.lfuel == pytest.approx(0.075)
    assert p.ufuel == pytest.approx(0.650)
    # A realistic IBIS litter load must land inside the ramp, not at an end.
    mid = float(fine_fuel_fraction(jnp.array([0.30]), jnp.zeros((1, NPFT)), p)[0])
    assert 0.0 < mid < 1.0


# ---------------------------------------------------------------------------
# Integration with F1, and the default
# ---------------------------------------------------------------------------

def test_default_keeps_the_legacy_fuel_term():
    """`use_fine_fuel` defaults to False, so F1 behaviour is unchanged."""
    assert default_fire_params().use_fine_fuel is False
    nfs, sd, n = jnp.full((NPOI,), 50.0), jnp.full((NPOI,), 40.0), jnp.array(100.0)
    totlit = jnp.full((NPOI,), 0.5)
    a = np.asarray(fire_seasonlength(nfs, sd, n, totlit, default_fire_params()))
    b = np.asarray(fire_seasonlength(nfs, sd, n, totlit, default_fire_params(),
                                     fuel=legacy_fuel_fraction(totlit)))
    assert np.array_equal(a, b)


def test_fine_fuel_changes_the_answer():
    """At a realistic litter load the two fuel terms must differ.

    Legacy saturates at 0.2 kgC/m2 -- it treats a modest litter layer as the
    maximum possible fuel. The ramp does not reach 1 until 0.65.
    """
    p = FireParams(use_fine_fuel=True)
    nfs, sd, n = jnp.full((NPOI,), 50.0), jnp.full((NPOI,), 40.0), jnp.array(100.0)
    totlit = totalit = jnp.full((NPOI,), 0.30)
    cbiol = jnp.zeros((NPOI, NPFT))

    legacy = np.asarray(fire_seasonlength(nfs, sd, n, totlit, p))
    fine = np.asarray(fire_seasonlength(nfs, sd, n, totlit, p,
                                        fuel=fine_fuel_fraction(totalit, cbiol, p)))
    assert np.all(fine < legacy), "the ramp should not be saturated at 0.3 kgC/m2"


def test_jittable_and_differentiable():
    p = default_fire_params()
    totalit = jnp.full((NPOI,), 0.20)
    cbiol = _cbiol(p11=0.10)
    f = lambda t: fine_fuel_fraction(t, cbiol, p).sum()
    assert np.isfinite(float(eqx.filter_jit(f)(totalit)))
    g = jax.grad(f)(totalit)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(g[0]) > 0.0     # more litter -> more fuel
