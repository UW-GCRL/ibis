"""Stage F0 regression tests: the legacy fire path must never change.

`docs/FIRE.md` stage F0 promises that moving the fire parameterisation into
`ibis.physics.fire` and adding a `fire_scheme` selector changes *no* numerical
behaviour. The 46-year F77<->JAX lockstep documented in `verify/STATUS.md`
rests on that promise, so it is pinned here from three independent directions:

1. Against a verbatim copy of the pre-F0 inline expression (commit 605800c,
   `vegetation_dynamics.py:40-41`) -- must be **exactly** bit-identical, since
   it is the same sequence of JAX ops.
2. Against a pure-Python `math` transcription of the Fortran
   (`fortran/ibis_2.6b4/vegetation.f:591`) -- agreement to machine precision.
   Not bit-exact: JAX's `exp` and libm's `exp` differ by up to 1 ULP.
3. Against analytically pinned values, so a future refactor cannot silently
   move the answer even if every implementation here is edited together.
"""

import math

import jax
jax.config.update("jax_enable_x64", True)  # repo convention: set before ibis imports

import jax.numpy as jnp
import numpy as np
import pytest

from ibis.config import RunConfig
from ibis.physics import vegetation_dynamics
from ibis.physics.fire import (
    SEASONLENGTH,
    AVAILABLE_SCHEMES,
    LEGACY,
    LEGACY_BURN_COEF,
    LEGACY_LITTER_REF,
    compute_disturbf,
    fire_legacy,
)

# Machine epsilon for float64; the exp() discrepancy is bounded by ~1 ULP.
EPS = float(np.finfo(np.float64).eps)


def _fire_pre_f0(firefac, totlit):
    """Verbatim copy of the pre-F0 inline implementation (commit 605800c)."""
    burn = firefac * jnp.minimum(1.0, totlit / 0.200)
    return jnp.minimum(1.0, jnp.maximum(0.0, 1.0 - jnp.exp(-0.5 * burn)))


def _fire_fortran_reference(firefac, totlit):
    """Pure-Python transcription of `vegetation.f:591`, elementwise.

    Uses only the standard library so it shares no code path with JAX
    or NumPy.
    """
    out = []
    for ff, tl in zip(firefac, totlit):
        burn = ff * min(1.0, tl / 0.200)
        disturbf = 1.0 - math.exp(-0.5 * burn)
        out.append(max(0.0, min(1.0, disturbf)))
    return np.array(out, dtype=np.float64)


@pytest.fixture(scope="module")
def sweep():
    """Randomised inputs plus edge cases, spanning the physical domain."""
    rng = np.random.default_rng(20260811)
    n = 20_000
    firefac = rng.uniform(0.0, 1.0, n)
    totlit = rng.uniform(0.0, 0.8, n)  # spans below and above the 0.2 kink

    edges = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, LEGACY_LITTER_REF],   # exactly at the saturation kink
            [1.0, 1e6],                 # far past saturation
            [1.0, 1.0],
            [1e-12, 1e-12],             # near-zero, tests the clip at 0
        ]
    )
    firefac = np.concatenate([firefac, edges[:, 0]])
    totlit = np.concatenate([totlit, edges[:, 1]])
    return firefac, totlit


# ---------------------------------------------------------------------------
# 1. The refactor itself must be bit-exact
# ---------------------------------------------------------------------------

def test_refactor_is_bit_exact(sweep):
    """fire_legacy must be bit-identical to the pre-F0 inline expression."""
    firefac, totlit = sweep
    got = np.asarray(fire_legacy(jnp.asarray(firefac), jnp.asarray(totlit)))
    want = np.asarray(_fire_pre_f0(jnp.asarray(firefac), jnp.asarray(totlit)))

    assert got.dtype == np.float64
    n_diff = int((got != want).sum())
    assert n_diff == 0, f"{n_diff}/{got.size} values changed; F0 must not alter results"


def test_wrapper_delegates_bit_exactly(sweep):
    """The retained `vegetation_dynamics.fire` wrapper must not perturb anything."""
    firefac, totlit = sweep
    via_wrapper = np.asarray(
        vegetation_dynamics.fire(jnp.asarray(firefac), jnp.asarray(totlit))
    )
    direct = np.asarray(fire_legacy(jnp.asarray(firefac), jnp.asarray(totlit)))
    assert np.array_equal(via_wrapper, direct)


# ---------------------------------------------------------------------------
# 2. Agreement with the Fortran, to machine precision
# ---------------------------------------------------------------------------

def test_matches_fortran_transcription(sweep):
    """Agreement with a stdlib transcription of vegetation.f:591.

    Bit-exactness is not claimed: JAX's exp and glibc's exp may differ by
    ~1 ULP. The bound below is 4 ULP at unit magnitude, comfortably tight
    given disturbf is O(1).
    """
    firefac, totlit = sweep
    got = np.asarray(fire_legacy(jnp.asarray(firefac), jnp.asarray(totlit)))
    want = _fire_fortran_reference(firefac, totlit)

    max_abs = float(np.max(np.abs(got - want)))
    assert max_abs <= 4 * EPS, f"max |JAX - Fortran| = {max_abs:.3e} exceeds 4 eps"


# ---------------------------------------------------------------------------
# 3. Analytically pinned values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "firefac, totlit, expected",
    [
        # below the kink: burn = firefac * totlit / 0.2
        (0.5, 0.10, 1.0 - math.exp(-0.5 * 0.5 * 0.5)),
        # exactly at the kink: burn = firefac
        (1.0, 0.20, 1.0 - math.exp(-0.5 * 1.0)),
        # past the kink: fuel term saturates at 1, burn = firefac
        (0.5, 0.30, 1.0 - math.exp(-0.5 * 0.5)),
        (1.0, 5.00, 1.0 - math.exp(-0.5 * 1.0)),
        # no dryness or no fuel -> no fire
        (0.0, 0.50, 0.0),
        (0.7, 0.00, 0.0),
    ],
)
def test_pinned_values(firefac, totlit, expected):
    got = float(fire_legacy(jnp.array([firefac]), jnp.array([totlit]))[0])
    assert got == pytest.approx(expected, abs=4 * EPS)


# ---------------------------------------------------------------------------
# 4. Structural properties the scheme must keep
# ---------------------------------------------------------------------------

def test_output_is_bounded(sweep):
    firefac, totlit = sweep
    got = np.asarray(fire_legacy(jnp.asarray(firefac), jnp.asarray(totlit)))
    assert np.all(got >= 0.0) and np.all(got <= 1.0)
    assert np.all(np.isfinite(got))


def test_fuel_term_saturates():
    """Beyond LEGACY_LITTER_REF the litter load must stop mattering."""
    ff = jnp.array([0.8, 0.8, 0.8])
    tl = jnp.array([LEGACY_LITTER_REF, 10.0 * LEGACY_LITTER_REF, 1e8])
    got = np.asarray(fire_legacy(ff, tl))
    assert np.allclose(got, got[0], rtol=0, atol=0)


def test_monotonic_in_dryness():
    ff = jnp.asarray(np.linspace(0.0, 1.0, 101))
    tl = jnp.full((101,), 0.5)
    got = np.asarray(fire_legacy(ff, tl))
    assert np.all(np.diff(got) >= 0.0)


def test_burn_coefficient_matches_fortran():
    """Guard the two magic numbers against accidental edits."""
    assert LEGACY_LITTER_REF == 0.200
    assert LEGACY_BURN_COEF == 0.5


# ---------------------------------------------------------------------------
# 5. Scheme dispatch and configuration
# ---------------------------------------------------------------------------

def test_dispatcher_matches_direct_call(sweep):
    firefac, totlit = sweep
    via_dispatch = np.asarray(
        compute_disturbf(LEGACY, firefac=jnp.asarray(firefac), totlit=jnp.asarray(totlit))
    )
    direct = np.asarray(fire_legacy(jnp.asarray(firefac), jnp.asarray(totlit)))
    assert np.array_equal(via_dispatch, direct)


def test_dispatcher_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown fire_scheme"):
        compute_disturbf(
            "spitfire", firefac=jnp.array([0.5]), totlit=jnp.array([0.3])
        )


def test_config_defaults_to_seasonlength():
    """The default scheme changed on 2026-08-24; legacy stays reachable.

    It defaulted to "legacy" on the stated grounds that the 46-yr F77<->JAX
    lockstep depended on it. It does not: the lockstep runs at isimfire=0, and
    `dynaveg` selects `jnp.where(isimfire == 1, _disturbf, 0.005)`, so with
    fire off the scheme is numerically irrelevant. Fire has never been enabled
    in any run, so nothing regresses -- and anyone setting isimfire=1 now gets
    rank +0.630 against GFED5 rather than +0.315.
    """
    assert RunConfig().fire_scheme == SEASONLENGTH
    assert LEGACY in AVAILABLE_SCHEMES
    assert SEASONLENGTH in AVAILABLE_SCHEMES


# ---------------------------------------------------------------------------
# 6. Stays usable by the calibration work planned in F6
# ---------------------------------------------------------------------------

def test_jittable_and_differentiable():
    """F6 calibrates fire parameters by gradient descent; keep the path clean."""
    jitted = jax.jit(fire_legacy)
    ff, tl = jnp.array([0.5]), jnp.array([0.3])
    assert np.array_equal(np.asarray(jitted(ff, tl)), np.asarray(fire_legacy(ff, tl)))

    grad = jax.grad(lambda f: fire_legacy(f, tl).sum())(ff)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(grad[0]) > 0.0  # drier fuel must increase disturbance
