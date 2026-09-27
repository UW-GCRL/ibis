"""Guards for the canopy scaling of photosynthetic capacity (CALIBRATION.md 26).

F77 scales the top leaf's rate by the light-profile integral, so a dense canopy has the capacity of ~1/k m2 of top
leaf. The switch (`stomata: cap_scale_u / cap_scale_l`, coefficient `cap_kn`) scales the capacity limits with their
own profile and co-limits at canopy level. It must be exactly F77 when the two profiles coincide, leave the light-limited
canopy alone, lift the light-saturated plateau, be bit-identical when off, and round-trip through the parameter file."""
from pathlib import Path
import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]


def _sp():
    from ibis.physics.stomata import StomataParams
    return StomataParams()


def _c3_inputs(par):
    n = len(par)
    return dict(par=jnp.asarray(par, jnp.float32), ci=jnp.full((n,), 250e-6, jnp.float32), vmax=jnp.full((n,), 37e-6, jnp.float32),
                gamstar=jnp.full((n,), 40e-6, jnp.float32), kc=jnp.full((n,), 230e-6, jnp.float32), ko=jnp.full((n,), 0.28, jnp.float32),
                o2conc=0.209, rdark=jnp.full((n,), 0.5e-6, jnp.float32))


def test_capacity_integral_limits():
    from ibis.physics.stomata import _capacity_integral
    L = jnp.array([0.0, 1e-6, 0.5, 5.0, 50.0], jnp.float32)
    f = np.asarray(_capacity_integral(L, 0.15, 1e-7))
    assert f[0] == 0.0                                                    # no leaves, no capacity
    assert abs(f[1] - 1.0) < 1e-5                                         # thin canopy: every leaf at top capacity
    assert np.all(np.diff(f[1:]) < 0)                                     # deeper canopies average lower
    assert abs(f[3] - (1 - np.exp(-0.75)) / 0.75) < 1e-6 and abs(f[4] - (1 - np.exp(-7.5)) / 7.5) < 1e-6


def test_equal_profiles_reproduce_f77_exactly():
    """Co-limitation is homogeneous of degree one: scaling every limit by the same f is F77's leaf rate x f."""
    from ibis.physics.stomata import farquhar_c3, farquhar_c4
    p = _sp(); x = _c3_inputs(np.linspace(0.0, 600.0, 25)); f = jnp.full((25,), 0.245, jnp.float32)
    ag0, an0 = farquhar_c3(x['par'], x['ci'], x['vmax'], x['gamstar'], x['kc'], x['ko'], x['o2conc'], x['rdark'], p)
    ag1, an1 = farquhar_c3(x['par'], x['ci'], x['vmax'], x['gamstar'], x['kc'], x['ko'], x['o2conc'], x['rdark'], p, f_light=f, f_cap=f)
    np.testing.assert_allclose(np.asarray(ag1), np.asarray(ag0 * f), rtol=2e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(an1), np.asarray(an0 * f), rtol=2e-6, atol=1e-12)
    g0, n0 = farquhar_c4(x['par'], x['ci'], x['vmax'], x['rdark'], p)
    g1, n1 = farquhar_c4(x['par'], x['ci'], x['vmax'], x['rdark'], p, f_light=f, f_cap=f)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g0 * f), rtol=2e-6, atol=1e-12)


def test_shallow_capacity_profile_lifts_the_plateau_not_the_low_light_rate():
    from ibis.physics.stomata import farquhar_c3
    p = _sp(); x = _c3_inputs([20.0, 400.0])                              # a dim hour and a bright one (W m-2 top PAR)
    fl = jnp.full((2,), 0.245, jnp.float32); fc = jnp.full((2,), 0.63, jnp.float32)   # probe's f_light; Kn 0.2, L 5
    ag0, _ = farquhar_c3(x['par'], x['ci'], x['vmax'], x['gamstar'], x['kc'], x['ko'], x['o2conc'], x['rdark'], p)
    ag1, _ = farquhar_c3(x['par'], x['ci'], x['vmax'], x['gamstar'], x['kc'], x['ko'], x['o2conc'], x['rdark'], p, f_light=fl, f_cap=fc)
    r = np.asarray(ag1) / np.asarray(ag0 * fl)
    assert r[0] < 1.10                                                    # light-limited canopy: nearly unchanged
    assert r[1] > 1.8                                                     # light-saturated canopy: capacity no longer ~1/k m2


def test_parameter_file_round_trip_and_default_off(tmp_path):
    import yaml
    from ibis.io.physics_params import build_all_params, load_physics_params
    base = yaml.safe_load(open(ROOT / "configs" / "params" / "default.yaml"))
    base.setdefault("stomata", {}).update(cap_scale_u=True, cap_kn=0.2)
    f = tmp_path / "p.yaml"; yaml.safe_dump(base, open(f, "w"))
    sp, _, _ = build_all_params(2, np.zeros(2, np.float32), load_physics_params(str(f)))
    st = sp.stomata_params
    assert st.cap_scale_u is True and st.cap_scale_l is False and st.cap_kn == 0.2
    sp0, _, _ = build_all_params(2, np.zeros(2, np.float32), load_physics_params(str(ROOT / "configs" / "params" / "default.yaml")))
    assert sp0.stomata_params.cap_scale_u is False and sp0.stomata_params.cap_scale_l is False


def test_switch_is_wired_into_both_canopies():
    src = (ROOT / "ibis" / "physics" / "stomata.py").read_text()
    for s in ("if params.cap_scale_u:", "if params.cap_scale_l:", "_capacity_integral(lai[:, 1], params.cap_kn", "_capacity_integral(lai[:, 0], params.cap_kn"):
        assert s in src, s
    # the canopy rates must use the ci the leaf rates used (before ball_berry_step replaces it); the sunlit/shaded
    # canopy (test_sunshade_canopy.py) uses the same pre-step ci
    for s in ("ci_ub0", "ci_uc0", "ci_ls0", "ci_l30", "ci_l40"):
        assert src.count(s) >= 2, s
    rd = (ROOT / "ibis" / "io" / "physics_params.py").read_text()
    assert "cap_scale_u=bool(" in rd and "cap_scale_l=bool(" in rd and "cap_kn=float(" in rd
