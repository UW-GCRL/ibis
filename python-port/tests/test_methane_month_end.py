"""The month-end steps of the release methane (docs/METHANE.md 11, 12): a month run at unit inundated fraction is scaled
to the learned fraction exactly, because the flux is linear in it; the CH4 carbon booked out of Rh as CO2 is the month's
net emission, never more than the month's Rh."""
import numpy as np
import jax.numpy as jnp

from ibis.bgc.methane import (MethaneParams, MethaneState, ch4_carbon_from_rh, methane_flux, scale_month)


def test_flux_is_linear_in_the_inundated_fraction():
    rng = np.random.default_rng(3); n, nl = 6, 6
    hsoi = jnp.asarray([0.10, 0.15, 0.25, 0.50, 1.0, 2.0])
    rh = jnp.asarray(rng.uniform(0, 5e-3, n)); tsoi = jnp.asarray(rng.uniform(265, 305, (n, nl)))
    wsoi = jnp.asarray(rng.uniform(0.2, 1.0, (n, nl))); wisoi = jnp.asarray(rng.uniform(0, 0.2, (n, nl)))
    f = jnp.asarray([0.0, 0.01, 0.2, 0.5, 0.9, 1.0])
    one = methane_flux(jnp.ones(n), rh, tsoi, wsoi, wisoi, hsoi, MethaneParams())
    at_f = methane_flux(f, rh, tsoi, wsoi, wisoi, hsoi, MethaneParams())
    np.testing.assert_allclose(np.asarray(at_f), np.asarray(f * one), rtol=1e-6, atol=1e-12)


def test_scale_month_scales_only_the_month():
    before = jnp.asarray([1.0, 2.0, 3.0]); m = MethaneState(flux=jnp.asarray([0.1, 0.2, 0.3]), annual=before + jnp.asarray([3.0, 6.0, 9.0]),
                                                             zwt=jnp.asarray([0.1, 0.0, 0.3]))
    f = np.array([0.0, 0.5, 1.0])
    s = scale_month(m, before, f)
    np.testing.assert_allclose(np.asarray(s.annual), [1.0, 2.0 + 3.0, 3.0 + 9.0])       # earlier months untouched
    np.testing.assert_allclose(np.asarray(s.flux), [0.0, 0.1, 0.3])
    np.testing.assert_array_equal(np.asarray(s.zwt), np.asarray(m.zwt))
    first = scale_month(m, None, f)                                                       # January: nothing before
    np.testing.assert_allclose(np.asarray(first.annual), np.asarray(m.annual) * f)


def test_ch4_carbon_out_of_rh_is_capped_and_closes():
    rh0 = jnp.asarray([1.0, 1.0, 1.0, 1.0]); rh = rh0 + jnp.asarray([0.5, 0.5, 0.01, 0.0])
    w0 = jnp.asarray([0.2, 0.2, 0.2, 0.2]); w = w0 + jnp.asarray([0.1, -0.05, 0.3, 0.1])
    take = ch4_carbon_from_rh(w, w0, rh, rh0)
    np.testing.assert_allclose(np.asarray(take), [0.1, 0.0, 0.01, 0.0], rtol=1e-5, atol=1e-7)   # net uptake books nothing; capped at Rh
    rh_co2 = rh - take
    np.testing.assert_allclose(np.asarray(rh_co2 + take), np.asarray(rh))                 # Rh as CO2 + CH4 carbon = Rh
    assert np.all(np.asarray(rh_co2 - rh0) >= 0.0)
    np.testing.assert_allclose(np.asarray(ch4_carbon_from_rh(w, None, rh, rh0)), [0.3, 0.15, 0.01, 0.0], rtol=1e-6)   # January
