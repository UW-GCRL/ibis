"""Stage F1: fire season length replaces the annual mean of dryness.

The legacy scheme averages fuel dryness over every timestep of the year. That
average cannot tell a cell with a short, severe dry season from one that is
mildly dry all year round -- they can have identical means -- yet only the
first one burns. Fire is threshold-driven, so averaging is the wrong operator.

F1 counts the fraction of the year spent above a dryness threshold
(`fseason`) and the mean dryness within that period (`dry_mean`).
"""

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.physics.fire import (
    AVAILABLE_SCHEMES, LEGACY, SEASONLENGTH, FireParams, compute_disturbf,
    default_fire_params, fire_legacy, fire_seasonlength, update_fire_accumulators,
    update_firefac,
)

EPS = float(np.finfo(np.float64).eps)
SWILT, WISOI = 0.15, 0.0


def _accumulate(dryness_series, d_crit):
    """Drive both accumulators with a prescribed per-timestep dryness series.

    Returns (firefac, nfirestep, sdry, nsteps). Works in dryness space
    directly by inverting fueldry -> wsoi, so the test controls the signal.
    """
    nst, npoi = dryness_series.shape
    swilt = jnp.full((npoi,), SWILT)
    wisoi = jnp.zeros(npoi)
    # fueldry = -2*((w - swilt)/(1-swilt) - 0.5)  =>  w = swilt + (1-swilt)*(0.5 - d/2)
    wsoi_series = SWILT + (1.0 - SWILT) * (0.5 - dryness_series / 2.0)

    ff = jnp.zeros(npoi); nfs = jnp.zeros(npoi); sd = jnp.zeros(npoi)
    for k in range(nst):
        w = jnp.asarray(wsoi_series[k])
        ff = update_firefac(ff, jnp.array(float(k + 1)), w, wisoi, swilt)
        nfs, sd = update_fire_accumulators(nfs, sd, w, wisoi, swilt, d_crit)
    return ff, nfs, sd, float(nst)


# ---------------------------------------------------------------------------
# Reduction to legacy
# ---------------------------------------------------------------------------

def test_reduces_to_legacy_when_threshold_admits_everything():
    """With d_crit below all dryness and p == 1, F1 must recover legacy.

    fseason == 1 and dry_mean is then the all-timestep mean, i.e. firefac.
    Agreement is to ~1 ULP rather than bit-exact: legacy accumulates an
    incremental running mean while F1 divides a sum by a count, so the two
    round differently. The mathematics is identical.
    """
    rng = np.random.default_rng(0)
    dry = rng.uniform(0.0, 1.0, (200, 6))
    ff, nfs, sd, nst = _accumulate(dry, d_crit=-1.0)
    totlit = jnp.asarray(rng.uniform(0.0, 0.6, 6))

    a = np.asarray(fire_legacy(ff, totlit))
    b = np.asarray(fire_seasonlength(nfs, sd, jnp.array(nst), totlit,
                                     FireParams(d_crit=-1.0, k_b=0.5, p=1.0)))
    assert np.max(np.abs(a - b)) <= 8 * EPS


# ---------------------------------------------------------------------------
# The discrimination F1 exists to provide
# ---------------------------------------------------------------------------

def test_separates_dry_season_from_uniformly_mild_dryness():
    """Two cells, identical ANNUAL MEAN dryness, different fire regimes.

    Cell 0: dryness 0.8 for a quarter of the year, 0.0 otherwise  (mean 0.2)
    Cell 1: dryness 0.2 every timestep                            (mean 0.2)

    Legacy sees one number -- 0.2 -- and burns them identically. That is the
    defect. F1 must burn cell 0 and spare cell 1.
    """
    nst = 400
    dry = np.zeros((nst, 2))
    dry[: nst // 4, 0] = 0.8      # concentrated dry season
    dry[:, 1] = 0.2               # mild all year

    ff, nfs, sd, n = _accumulate(dry, d_crit=0.5)
    totlit = jnp.full((2,), 0.5)

    # Legacy: the two cells are indistinguishable.
    leg = np.asarray(fire_legacy(ff, totlit))
    assert leg[0] == pytest.approx(leg[1], abs=1e-9), (
        "precondition: the two cells must have equal annual-mean dryness")

    # F1: only the cell with a real dry season burns.
    new = np.asarray(fire_seasonlength(nfs, sd, jnp.array(n), totlit,
                                       FireParams(d_crit=0.5)))
    assert new[0] > 0.0
    assert new[1] == pytest.approx(0.0, abs=4 * EPS)
    assert new[0] > new[1]


def test_fseason_is_a_fraction_of_the_year():
    nst = 365
    dry = np.zeros((nst, 3))
    dry[: int(0.25 * nst), 0] = 0.9     # quarter of the year
    dry[: int(0.50 * nst), 1] = 0.9     # half
    dry[:, 2] = 0.9                     # all
    _, nfs, _, n = _accumulate(dry, d_crit=0.5)
    frac = np.asarray(nfs) / n
    assert frac[0] == pytest.approx(0.25, abs=1.0 / nst)
    assert frac[1] == pytest.approx(0.50, abs=1.0 / nst)
    assert frac[2] == pytest.approx(1.00, abs=1.0 / nst)


def test_longer_season_burns_more_at_equal_severity():
    nst = 400
    dry = np.zeros((nst, 3))
    for j, f in enumerate((0.1, 0.4, 0.9)):
        dry[: int(f * nst), j] = 0.8
    _, nfs, sd, n = _accumulate(dry, d_crit=0.5)
    out = np.asarray(fire_seasonlength(nfs, sd, jnp.array(n),
                                       jnp.full((3,), 0.5), default_fire_params()))
    assert out[0] < out[1] < out[2]


def test_bounded_and_finite():
    rng = np.random.default_rng(3)
    dry = rng.uniform(0.0, 1.0, (150, 8))
    _, nfs, sd, n = _accumulate(dry, d_crit=0.5)
    out = np.asarray(fire_seasonlength(nfs, sd, jnp.array(n),
                                       jnp.asarray(rng.uniform(0, 2.0, 8)),
                                       default_fire_params()))
    assert np.all(np.isfinite(out)) and np.all((out >= 0) & (out <= 1))


def test_no_fire_season_gives_zero():
    """A cell never above threshold must not burn, whatever the fuel load."""
    dry = np.full((100, 2), 0.1)
    _, nfs, sd, n = _accumulate(dry, d_crit=0.5)
    out = np.asarray(fire_seasonlength(nfs, sd, jnp.array(n),
                                       jnp.full((2,), 10.0), default_fire_params()))
    assert np.allclose(out, 0.0, atol=4 * EPS)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def test_scheme_registered_and_dispatches():
    assert SEASONLENGTH in AVAILABLE_SCHEMES
    dry = np.full((50, 2), 0.9)
    ff, nfs, sd, n = _accumulate(dry, d_crit=0.5)
    totlit = jnp.full((2,), 0.4)
    via = np.asarray(compute_disturbf(SEASONLENGTH, firefac=ff, totlit=totlit,
                                      nfirestep=nfs, sdry=sd, nytimes=jnp.array(n)))
    direct = np.asarray(fire_seasonlength(nfs, sd, jnp.array(n), totlit,
                                          default_fire_params()))
    assert np.array_equal(via, direct)


def test_dispatcher_requires_the_new_accumulators():
    with pytest.raises(ValueError, match="needs nfirestep"):
        compute_disturbf(SEASONLENGTH, firefac=jnp.zeros(2), totlit=jnp.ones(2))


def test_legacy_path_is_untouched():
    """F1 must not perturb the scheme F0 pinned."""
    rng = np.random.default_rng(11)
    ff = jnp.asarray(rng.uniform(0, 1, 500))
    tl = jnp.asarray(rng.uniform(0, 0.6, 500))
    a = np.asarray(compute_disturbf(LEGACY, firefac=ff, totlit=tl))
    b = np.asarray(fire_legacy(ff, tl))
    assert np.array_equal(a, b)


def test_jittable_and_differentiable():
    dry = np.full((80, 3), 0.9)
    _, nfs, sd, n = _accumulate(dry, d_crit=0.5)
    totlit = jnp.full((3,), 0.4)
    f = lambda s: fire_seasonlength(nfs, s, jnp.array(n), totlit,
                                    default_fire_params()).sum()
    assert np.isfinite(float(eqx.filter_jit(f)(sd)))
    g = jax.grad(f)(sd)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(g[0]) > 0.0   # more accumulated dryness -> more disturbance
