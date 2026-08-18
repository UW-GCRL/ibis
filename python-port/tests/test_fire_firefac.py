"""Regression tests for the fuel-dryness accumulator (docs/FIRE.md defect 0).

Before 2026-08-18 the JAX port never assigned `firefac`: it was declared,
initialised to zero, read by `fire()`, and updated nowhere. F77's accumulator
in `stats.f:1174-1180` had simply not been ported, so every production run had
`firefac == 0`, hence `burn == 0` and `disturbf == 0` -- fire had no effect
even with `isimfire=1`.

Verification never caught it because `verify/vegetation/verify_dynaveg.py`
injects random `firefac` values, testing `fire()` as a function while never
exercising the chain that feeds it. The last test in this file is the one that
would have caught it: it asserts the accumulator is actually *wired in*.
"""

import math

import jax
jax.config.update("jax_enable_x64", True)  # repo convention: before ibis imports

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.constants import PhysicalConstants
from ibis.dimensions import GridConfig
from ibis.physics.fire import fueldry_from_soil, update_firefac
from ibis.state.model_state import create_ibis_state
from ibis.state.vegetation import create_vegetation_state
from ibis.timestepping.loops import default_sim_params, timestep_body

# Reuse the runnable cold-start fixtures (state/consts/params).
from tests.test_integration import consts, params, state  # noqa: F401

EPS = float(np.finfo(np.float64).eps)

GRID = GridConfig(nlon=2, nlat=2, npoi=4, xres=90.0, yres=90.0,
                  nband=2, nsoilay=6, nsnolay=3, npft=12)
DTIME = 1800.0


def _fueldry_fortran(wsoi, wisoi, swilt):
    """Pure-Python transcription of stats.f:1174-1178 (stdlib only)."""
    out = []
    for wl, wi, sw in zip(wsoi, wisoi, swilt):
        water = wi + (1.0 - wi) * wl
        waterfrac = (water - sw) / (1.0 - sw)
        out.append(max(0.0, min(1.0, -2.0 * (waterfrac - 0.5))))
    return np.array(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# fueldry_from_soil
# ---------------------------------------------------------------------------

def test_fueldry_matches_fortran_transcription():
    rng = np.random.default_rng(20260818)
    n = 20_000
    swilt = rng.uniform(0.05, 0.4, n)
    wsoi = rng.uniform(0.0, 1.0, n)
    wisoi = rng.uniform(0.0, 0.9, n)

    got = np.asarray(fueldry_from_soil(jnp.asarray(wsoi), jnp.asarray(wisoi),
                                       jnp.asarray(swilt)))
    want = _fueldry_fortran(wsoi, wisoi, swilt)
    assert got.dtype == np.float64
    assert float(np.max(np.abs(got - want))) <= 4 * EPS


def test_fueldry_bounds_and_endpoints():
    swilt = jnp.full((3,), 0.2)
    # saturated -> 0 ; at the midpoint -> 0 ; at wilting point -> 1
    wsoi = jnp.array([1.0, 0.6, 0.0])
    wisoi = jnp.zeros(3)
    fd = np.asarray(fueldry_from_soil(wsoi, wisoi, swilt))
    assert fd[0] == pytest.approx(0.0, abs=4 * EPS)
    assert fd[1] == pytest.approx(0.0, abs=4 * EPS)
    assert fd[2] == pytest.approx(1.0, abs=4 * EPS)
    assert np.all((fd >= 0.0) & (fd <= 1.0))


def test_ice_uses_fortran_water_combination():
    """Total water is wisoi + (1-wisoi)*wsoi, NOT wsoi + wisoi.

    `wsoi` is the liquid fraction of the ICE-FREE pore space (comsoi.h:41-42,
    combined at stats.f:1175). Writing `wsoi + wisoi` would look right and be
    wrong; this test pins the correct combination.
    """
    swilt = jnp.full((1,), 0.2)
    # Chosen so BOTH candidate combinations land in the unclamped range
    # (waterfrac < 0.5); otherwise both clamp to 0 and the test proves nothing.
    wsoi, wisoi = jnp.array([0.1]), jnp.array([0.3])
    got = float(fueldry_from_soil(wsoi, wisoi, swilt)[0])

    water_correct = 0.3 + (1.0 - 0.3) * 0.1          # 0.37 -> fueldry 0.575
    water_wrong = 0.1 + 0.3                          # 0.40 -> fueldry 0.500
    f = lambda w: max(0.0, min(1.0, -2.0 * ((w - 0.2) / 0.8 - 0.5)))
    assert got == pytest.approx(f(water_correct), abs=4 * EPS)
    assert got != pytest.approx(f(water_wrong), abs=1e-6)


# ---------------------------------------------------------------------------
# update_firefac
# ---------------------------------------------------------------------------

def test_first_step_returns_fueldry_exactly():
    """At nytimes == 1 the running mean must reduce to fueldry.

    This is how F77 re-initialises the annual mean without clearing firefac.
    """
    swilt, wsoi, wisoi = jnp.full((4,), 0.2), jnp.array([1.0, 0.7, 0.3, 0.0]), jnp.zeros(4)
    stale = jnp.array([0.9, 0.9, 0.9, 0.9])  # value carried over from last year
    got = np.asarray(update_firefac(stale, jnp.array(1.0), wsoi, wisoi, swilt))
    want = np.asarray(fueldry_from_soil(wsoi, wisoi, swilt))
    assert np.array_equal(got, want)


def test_running_mean_equals_arithmetic_mean():
    """N sequential updates must equal the plain mean of the N fueldry values."""
    rng = np.random.default_rng(7)
    nsteps, npoi = 60, 5
    swilt = jnp.full((npoi,), 0.25)
    wsoi_seq = rng.uniform(0.0, 1.0, (nsteps, npoi))
    wisoi = jnp.zeros(npoi)

    ff = jnp.zeros(npoi)
    for k in range(nsteps):
        ff = update_firefac(ff, jnp.array(float(k + 1)),
                            jnp.asarray(wsoi_seq[k]), wisoi, swilt)

    fd = np.stack([_fueldry_fortran(wsoi_seq[k], np.zeros(npoi),
                                    np.full(npoi, 0.25)) for k in range(nsteps)])
    assert np.max(np.abs(np.asarray(ff) - fd.mean(axis=0))) < 1e-12


def test_constant_input_is_a_fixed_point():
    swilt, wsoi, wisoi = jnp.full((3,), 0.2), jnp.zeros(3), jnp.zeros(3)  # fully dry -> 1
    ff = jnp.zeros(3)
    for k in range(1, 25):
        ff = update_firefac(ff, jnp.array(float(k)), wsoi, wisoi, swilt)
    assert np.allclose(np.asarray(ff), 1.0, atol=4 * EPS)


def test_jittable_and_differentiable():
    swilt, wsoi, wisoi = jnp.full((3,), 0.2), jnp.full((3,), 0.4), jnp.zeros(3)
    ff = jnp.zeros(3)
    j = jax.jit(update_firefac)
    assert np.allclose(np.asarray(j(ff, jnp.array(1.0), wsoi, wisoi, swilt)),
                       np.asarray(update_firefac(ff, jnp.array(1.0), wsoi, wisoi, swilt)))
    g = jax.grad(lambda w: update_firefac(ff, jnp.array(1.0), w, wisoi, swilt).sum())(wsoi)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(g[0]) < 0.0  # wetter soil -> less dryness


# ---------------------------------------------------------------------------
# The wiring test -- this is the one that would have caught defect 0
# ---------------------------------------------------------------------------

def test_state_carries_the_counter():
    veg = create_vegetation_state(GRID.npoi, GRID.npft, GRID.nsoilay)
    assert hasattr(veg, "nytimes")
    assert float(veg.nytimes) == 0.0
    assert float(jnp.max(veg.firefac)) == 0.0


def test_timestep_body_actually_updates_firefac(state, consts, params):
    """`timestep_body` must advance firefac and nytimes.

    Defect 0 was precisely that nothing did. A component can be verified
    bit-exact as a function and still be wired to nothing; this asserts the
    wiring, not the formula.

    Reuses the runnable state fixture from tests/test_integration.py -- a bare
    `create_ibis_state` leaves soil texture unset (poros == 0) and the physics
    degenerates to NaN.
    """
    # Dry the topsoil so fueldry is unambiguously positive:
    #   water = 0 + (1-0)*0.20 = 0.20
    #   waterfrac = (0.20 - 0.15) / (1 - 0.15) = 0.0588
    #   fueldry = -2*(0.0588 - 0.5) = 0.882
    state = eqx.tree_at(
        lambda s: (s.soil.wsoi, s.soil.wisoi),
        state,
        (jnp.full_like(state.soil.wsoi, 0.20),
         jnp.zeros_like(state.soil.wisoi)),
    )

    assert float(state.vegetation.nytimes) == 0.0
    assert float(jnp.max(state.vegetation.firefac)) == 0.0

    new_state = timestep_body(state, consts, params, DTIME)

    assert float(new_state.vegetation.nytimes) == 1.0
    ff = np.asarray(new_state.vegetation.firefac)
    assert np.all(np.isfinite(ff)), "firefac is not finite after one timestep"
    assert float(np.max(ff)) > 0.0, (
        "firefac stayed zero -- the stats.f accumulator is not wired in "
        "(docs/FIRE.md defect 0)"
    )

    # At nytimes == 1 firefac must equal fueldry of the post-step topsoil.
    expect = np.asarray(fueldry_from_soil(
        new_state.soil.wsoi[:, 0], new_state.soil.wisoi[:, 0],
        new_state.soil.swilt[:, 0]))
    assert np.allclose(ff, expect, rtol=0, atol=4 * EPS)

    # A second step keeps advancing the counter.
    s2 = timestep_body(new_state, consts, params, DTIME)
    assert float(s2.vegetation.nytimes) == 2.0
