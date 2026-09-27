"""Observed hourly shortwave in the day loop (weather.observed_shortwave; CALIBRATION.md 27).

Where the mask is off the generator's direct and diffuse fluxes come back unchanged (bit-identical runs); where it is
on the observed total is conserved across bands and split by Spitters' hourly diffuse fraction: all diffuse under an
overcast sky or with the sun down, mostly direct under a clear noon sky. The effective solar constant is the one
`diurnal` uses."""
import numpy as np
import jax.numpy as jnp


def _gen(n):
    return jnp.full((n, 2), 100.0, jnp.float32), jnp.full((n, 2), 40.0, jnp.float32)


def test_mask_off_returns_the_generator_fluxes_exactly():
    from ibis.physics.weather import observed_shortwave
    solad_g, solai_g = _gen(3)
    sw = jnp.array([500.0, 0.0, 900.0], jnp.float32); cosz = jnp.array([0.7, 0.0, 0.9], jnp.float32)
    sd, si = observed_shortwave(sw, jnp.zeros(3, bool), cosz, jnp.int32(190), solad_g, solai_g)
    np.testing.assert_array_equal(np.asarray(sd), np.asarray(solad_g)); np.testing.assert_array_equal(np.asarray(si), np.asarray(solai_g))
    assert sd.dtype == solad_g.dtype


def test_observed_total_is_conserved_and_split_by_clearness():
    from ibis.physics.weather import observed_shortwave, effective_solar_constant
    n = 5; solad_g, solai_g = _gen(n)
    cosz = jnp.array([0.9, 0.9, 0.9, 0.3, 0.0], jnp.float32)
    s0 = float(effective_solar_constant(jnp.int32(190)))
    sw = jnp.array([0.75 * s0 * 0.9, 0.20 * s0 * 0.9, 0.45 * s0 * 0.9, 0.6 * s0 * 0.3, 5.0], jnp.float32)   # clear, overcast, broken, low sun, sun down
    sd, si = observed_shortwave(sw, jnp.ones(n, bool), cosz, jnp.int32(190), solad_g, solai_g)
    tot = np.asarray(sd + si).sum(1)
    np.testing.assert_allclose(tot, np.asarray(sw), rtol=1e-5)                       # the total is conserved
    np.testing.assert_allclose(np.asarray(sd + si)[:, 0] / np.maximum(tot, 1e-9), 0.46, rtol=1e-5)   # band split, visible share
    fd = np.asarray(si).sum(1) / np.maximum(tot, 1e-9)
    assert fd[0] < 0.3                                                               # clear noon: mostly direct
    assert fd[1] == 1.0                                                              # overcast: all diffuse
    assert 0.3 < fd[2] < 1.0                                                         # broken cloud: in between
    assert fd[3] > fd[0]                                                             # low sun: more diffuse at the same clearness class
    assert fd[4] == 1.0 and np.asarray(sd)[4].sum() == 0.0                           # sun down: all diffuse


def test_effective_solar_constant_is_the_orbital_formula():
    from ibis.physics.weather import effective_solar_constant
    for jday, lo, hi in ((3, 1410.0, 1420.0), (185, 1320.0, 1330.0)):
        s0 = float(effective_solar_constant(jnp.int32(jday)))
        assert lo < s0 < hi, (jday, s0)
