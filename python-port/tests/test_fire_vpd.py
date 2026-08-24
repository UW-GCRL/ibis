"""Stage F1b: VPD-based fire danger replaces soil-moisture dryness.

Soil-moisture dryness ranks cells badly: measured against GFED5 2001-2005 on
62176 cells, Spearman rank correlation was +0.258 for the soil-moisture burn
index, and the model's most fire-prone cells sat at mean |lat| 39.8 deg -- the
subtropical dry belt -- while GFED5's sit at 10.7 deg, in tropical savanna.
Rank is parameter-free (k_b and p are monotonic), so no calibration could
repair a 29 deg displacement.

VPD multiplied by vegetation density scores +0.590 and puts the most
fire-prone cells at 11.6 deg. This is the Drueke et al. (2019) fix that took
LPJmL4-SPITFIRE from NMSE 1.80 to 0.82 over South America.
"""

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.physics.fire import (
    FireParams, default_fire_params, fire_seasonlength, flammability_vpd,
    update_fire_accumulators_vpd, vegetation_density, vpd_from_state,
)
from ibis.physics.thermo import esat

EPS = float(np.finfo(np.float64).eps)
PS = 101325.0


def test_vpd_uses_ibis_own_saturation_curve():
    """VPD must be built on thermo.esat, not a second saturation formula.

    `esat` is a transcription of the F77 statement functions in comsat.h and is
    verified bit-exact; carrying a separate Tetens curve would put two
    inconsistent thermodynamics in one model.
    """
    ta = jnp.array([280.0, 295.0, 310.0])
    qa = jnp.zeros(3)                      # dry air -> VPD == esat
    got = np.asarray(vpd_from_state(ta, qa, jnp.full((3,), PS)))
    want = np.asarray(esat(ta)) / 1000.0
    assert np.allclose(got, want, rtol=0, atol=8 * EPS)


def test_vpd_is_zero_at_saturation():
    """Saturated air has no deficit, and VPD must never go negative."""
    ta = jnp.array([290.0, 300.0])
    e_sat = np.asarray(esat(ta))
    qa = jnp.asarray(0.622 * e_sat / (PS - 0.378 * e_sat))    # exactly saturated
    v = np.asarray(vpd_from_state(ta, qa, jnp.full((2,), PS)))
    assert np.all(v >= 0.0)
    assert np.allclose(v, 0.0, atol=1e-6)


def test_vpd_ordering_matches_physics():
    """Hot+dry > hot+humid > cold."""
    ta = jnp.array([313.0, 303.0, 303.0, 283.0])
    qa = jnp.array([0.004, 0.008, 0.020, 0.006])
    v = np.asarray(vpd_from_state(ta, qa, jnp.full((4,), PS)))
    assert v[0] > v[1] > v[2]        # hot-dry > warm-dry > warm-humid
    assert v[3] < v[2]               # cold lowest


def test_flammability_bounded_and_saturating():
    p = default_fire_params()
    ta = jnp.asarray(np.linspace(270.0, 330.0, 60))
    qa = jnp.full((60,), 0.004)
    f = np.asarray(flammability_vpd(ta, qa, jnp.full((60,), PS), p))
    assert np.all((f >= 0.0) & (f <= 1.0))
    assert np.all(np.diff(f) >= -8 * EPS)          # monotone in temperature
    assert f[-1] == pytest.approx(1.0)             # saturates at vpd_ref


def test_vegetation_density_bounded_and_monotone():
    p = default_fire_params()
    plai = jnp.zeros((5, 12)).at[:, 10].set(jnp.asarray([0.0, 0.5, 2.0, 4.0, 12.0]))
    vd = np.asarray(vegetation_density(plai, p))
    assert np.all((vd >= 0.0) & (vd <= 1.0))
    assert np.all(np.diff(vd) >= 0.0)
    assert vd[0] == 0.0                      # bare ground cannot burn
    assert vd[-1] == pytest.approx(1.0)      # saturates at lai_ref_vd


def test_bare_ground_gives_no_fire_however_dry():
    """The failure VPD-alone could not fix.

    A perennially hot, bone-dry cell with no vegetation must not burn. Soil
    moisture scored such cells highest, which is why the old scheme put its
    fire in the subtropical dry belt.
    """
    p = FireParams(use_vpd_danger=True)
    n = 1
    nfs, sd = jnp.full((n,), 8000.0), jnp.full((n,), 7800.0)   # maximal fire season
    out = np.asarray(fire_seasonlength(
        nfs, sd, jnp.array(8760.0), jnp.full((n,), 0.5), p,
        vd=vegetation_density(jnp.zeros((n, 12)), p)))
    assert np.allclose(out, 0.0, atol=8 * EPS)


def test_vegetated_cell_burns_more_than_sparse_one():
    p = FireParams(use_vpd_danger=True)
    nfs, sd, nyt = jnp.full((2,), 6000.0), jnp.full((2,), 5000.0), jnp.array(8760.0)
    plai = jnp.zeros((2, 12)).at[0, 10].set(0.3).at[1, 10].set(4.0)
    out = np.asarray(fire_seasonlength(nfs, sd, nyt, jnp.full((2,), 0.5), p,
                                       vd=vegetation_density(plai, p)))
    assert out[1] > out[0]


def test_accumulator_counts_only_flammable_timesteps():
    p = default_fire_params()
    nfs, sd = jnp.zeros(3), jnp.zeros(3)
    # cell 0 humid (below threshold), 1 marginal, 2 hot-dry
    ta = jnp.array([300.0, 303.0, 315.0]); qa = jnp.array([0.020, 0.008, 0.003])
    for _ in range(10):
        nfs, sd = update_fire_accumulators_vpd(nfs, sd, ta, qa, jnp.full((3,), PS), p)
    n = np.asarray(nfs)
    assert n[0] == 0.0            # never flammable
    assert n[2] == 10.0           # always flammable
    assert np.all(np.asarray(sd) >= 0.0)


def test_vpd_danger_is_now_the_default():
    """F1b shipped off, and was switched on once measured in-model.

    Against GFED5 2001-2005 over 88 484 cells, from one live 5-year run: the
    VPD path ranks +0.630 against the soil-moisture path's +0.315, and places
    the top-2000 burning cells at |lat| 14.2 rather than 39.8, against an
    observed 10.7. The soil-moisture path stays reachable by setting the flag
    back to False, and `legacy` is untouched either way.
    """
    assert default_fire_params().use_vpd_danger is True
    assert FireParams(use_vpd_danger=False).use_vpd_danger is False


def test_vd_is_optional_and_omitting_it_changes_nothing():
    p = FireParams()
    a = np.asarray(fire_seasonlength(jnp.full((3,), 50.0), jnp.full((3,), 40.0),
                                     jnp.array(100.0), jnp.full((3,), 0.5), p))
    b = np.asarray(fire_seasonlength(jnp.full((3,), 50.0), jnp.full((3,), 40.0),
                                     jnp.array(100.0), jnp.full((3,), 0.5), p, vd=None))
    assert np.array_equal(a, b)


def test_jittable_and_differentiable():
    p = default_fire_params()
    # 295 K / qa=0.006 gives VPD ~1.7 kPa, inside the linear range. At 305 K
    # the deficit is 3.7 kPa, above vpd_ref, so flammability clips to 1 and the
    # gradient is legitimately zero -- checked separately below.
    ta, qa, ps = jnp.full((4,), 295.0), jnp.full((4,), 0.006), jnp.full((4,), PS)
    f = lambda t: flammability_vpd(t, qa, ps, p).sum()
    assert np.isfinite(float(eqx.filter_jit(f)(ta)))
    g = jax.grad(f)(ta)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(g[0]) > 0.0        # warmer air -> more flammable

    # Above vpd_ref the term saturates, so the gradient vanishes by design.
    g_sat = jax.grad(lambda t: flammability_vpd(t, qa, ps, p).sum())(
        jnp.full((4,), 310.0))
    assert float(g_sat[0]) == pytest.approx(0.0)
