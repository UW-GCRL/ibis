"""Cold-season down-regulation of the whole leaf (StomataParams.acclim_light, frost_on; CALIBRATION.md 53)."""
import numpy as np, jax.numpy as jnp
from ibis.physics.stomata import farquhar_c3, farquhar_c4, StomataParams
from ibis.io.physics_params import load_physics_params, build_stomata_params


def test_light_rate_factor_scales_the_light_limited_rate_only():
    p = StomataParams()
    par = jnp.array([50.0, 50.0]); ci = jnp.array([2.5e-4] * 2); vmax = jnp.array([6e-5] * 2)
    g = jnp.array([4e-5] * 2); kc = jnp.array([3e-4] * 2); ko = jnp.array([0.3] * 2); rd = jnp.array([1e-6] * 2)
    ag0, _ = farquhar_c3(par, ci, vmax, g, kc, ko, 0.209, rd, p)
    ag1, _ = farquhar_c3(par, ci, vmax, g, kc, ko, 0.209, rd, p, je_fac=jnp.array([1.0, 1.0]))
    ag2, _ = farquhar_c3(par, ci, vmax, g, kc, ko, 0.209, rd, p, je_fac=jnp.array([0.2, 1.0]))
    assert np.allclose(np.asarray(ag0), np.asarray(ag1))                          # factor 1: identical
    assert float(ag2[0]) < 0.5 * float(ag0[0]) and np.isclose(float(ag2[1]), float(ag0[1]))   # low light: je-limited
    c0, _ = farquhar_c4(par, ci, vmax, rd, p); c2, _ = farquhar_c4(par, ci, vmax, rd, p, je_fac=jnp.array([0.2, 1.0]))
    assert float(c2[0]) < float(c0[0]) and np.isclose(float(c2[1]), float(c0[1]))


def test_switches_default_off_and_read_from_yaml(tmp_path):
    sp = StomataParams(); assert sp.acclim_light is False and sp.frost_on is False and (sp.frost_t0, sp.frost_t1) == (-5.0, 5.0)
    y = tmp_path / 'p.yaml'; y.write_text('stomata:\n  acclim_on: true\n  acclim_light: true\n  frost_on: true\n  frost_t0: -8.0\n  frost_t1: 8.0\n')
    st = build_stomata_params(load_physics_params(str(y)))
    assert st.acclim_light is True and st.frost_on is True and (st.frost_t0, st.frost_t1) == (-8.0, 8.0)
