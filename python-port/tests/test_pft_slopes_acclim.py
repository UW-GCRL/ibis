"""Stomatal slope and acclimation constants per PFT (StomataParams.coefm_mix, acclim_per_pft; CALIBRATION.md 32).

With coefm_mix on and one PFT per big leaf, the slope is that PFT's entry exactly (bit-identical to the group value when
the entry equals it); changing the C4 grass entry moves only the C4 big leaf; the per-PFT extras of the fit harness set
and read the array entries; the parameter file round-trips; off is the default."""
from pathlib import Path
import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]


def _inputs(n=2):
    t = jnp.full((n,), 298.0, jnp.float32); z = jnp.zeros((n,), jnp.float32); L = jnp.full((n,), 3.0, jnp.float32); S = jnp.full((n,), 0.5, jnp.float32)
    scal = jnp.tile(jnp.array([30.0, 40.0, 5.0, 75.0], jnp.float32), (n, 1))
    exist = jnp.zeros((n, 12)).at[:, 4].set(1.0).at[:, 3].set(1.0).at[:, 8].set(1.0).at[:, 10].set(1.0).at[:, 11].set(1.0)
    frac = jnp.zeros((n, 12)).at[:, 4].set(1.0).at[:, 3].set(1.0).at[:, 8].set(1.0).at[:, 10].set(1.0).at[:, 11].set(1.0)
    return dict(tu=t, tl=t, t12=t, t34=t, q12=jnp.full((n,), 0.012), q34=jnp.full((n,), 0.012), su=jnp.full((n,), 0.05), sl=jnp.full((n,), 0.03),
                psurf=jnp.full((n,), 101325.0), topparu=jnp.full((n,), 60.0), topparl=jnp.full((n,), 20.0), termu6=jnp.full((n,), 0.6), termu7=jnp.full((n,), 0.9),
                terml6=jnp.full((n,), 0.6), terml7=jnp.full((n,), 0.9), scalcoefu=scal, scalcoefl=0.3 * scal,
                ci_upper=jnp.full((n, 2), 280.0e-6), gs_upper=jnp.full((n, 2), 0.05), cs_upper=jnp.full((n, 2), 340.0e-6),
                ci_lower=jnp.full((n, 3), 280.0e-6), gs_lower=jnp.full((n, 3), 0.05), cs_lower=jnp.full((n, 3), 340.0e-6),
                fwetu=z, fwetl=z, lai=jnp.stack([L, L], -1), sai=jnp.stack([S, S], -1), exist=exist, stresstu=jnp.ones((n,)), stresstl=jnp.ones((n,)),
                a10scalparamu=jnp.full((n,), 0.5), a10daylightu=jnp.full((n,), 100.0), a10scalparaml=jnp.full((n,), 0.5), a10daylightl=jnp.full((n,), 100.0),
                co2conc=380.0e-6, dtime=3600.0), frac


def test_one_pft_per_leaf_is_bit_identical_and_the_c4_entry_moves_only_the_c4_leaf():
    from ibis.physics.stomata import StomataParams, stomata
    x, frac = _inputs()
    off = stomata(**x, params=StomataParams(), frac=frac)
    on = stomata(**x, params=StomataParams(coefm_mix=True), frac=frac)              # coefm_pft defaults = the group values
    for k in ('gs_upper', 'gs_lower', 'ci_upper', 'ci_lower', 'totcondub', 'totcondl4'):
        np.testing.assert_array_equal(np.asarray(getattr(on, k)), np.asarray(getattr(off, k)), err_msg=k)
    p = StomataParams(coefm_mix=True, coefm_pft=StomataParams().coefm_pft.at[10].set(12.0))
    c4 = stomata(**x, params=p, frac=frac)
    assert not np.array_equal(np.asarray(c4.gs_lower[:, 2]), np.asarray(off.gs_lower[:, 2]))          # the C4 leaf moved
    np.testing.assert_array_equal(np.asarray(c4.gs_lower[:, :2]), np.asarray(off.gs_lower[:, :2]))   # shrub and C3 did not
    np.testing.assert_array_equal(np.asarray(c4.gs_upper), np.asarray(off.gs_upper))


def test_per_pft_extras_set_and_read_the_arrays():
    import sys; sys.path.insert(0, str(ROOT.parent / 'verify' / 'calibration'))
    from fit_carbon import EXTRAS, _extra_value, PFT_NAMES
    from ibis.io.physics_params import build_all_params
    sp, pft, _ = build_all_params(2, np.zeros(2, np.float32), None)
    for name, val in (('coefm:c4_grass', 12.5), ('acclim_s0:boreal_con_ever', -7.0), ('acclim_smax:temp_con_ever', 25.0)):
        assert name in EXTRAS
        sp, pft = EXTRAS[name][0](sp, pft, val)
        assert abs(float(_extra_value(sp, pft, name)) - val) < 1e-6
    assert float(sp.stomata_params.coefm_pft[PFT_NAMES.index('c4_grass')]) == 12.5 and float(sp.stomata_params.coefm_pft[PFT_NAMES.index('c3_grass')]) == 9.0


def test_parameter_file_round_trip_and_defaults():
    from ibis.io.physics_params import build_all_params, load_physics_params
    sp, _, _ = build_all_params(2, np.zeros(2, np.float32), load_physics_params(str(ROOT / "configs" / "params" / "default_seasonal_sunshade_pft.yaml")))
    st = sp.stomata_params
    assert st.coefm_mix is True and st.acclim_per_pft is True
    np.testing.assert_allclose(np.asarray(st.coefm_pft), [10, 10, 10, 6, 10, 6, 10, 6, 9, 9, 4, 9])
    np.testing.assert_allclose(np.asarray(st.acclim_s0_pft), -3.063, rtol=1e-6); np.testing.assert_allclose(np.asarray(st.acclim_smax_pft), 17.72, rtol=1e-6)
    sp0, _, _ = build_all_params(2, np.zeros(2, np.float32), load_physics_params(str(ROOT / "configs" / "params" / "default.yaml")))
    assert sp0.stomata_params.coefm_mix is False and sp0.stomata_params.acclim_per_pft is False


def test_acclimation_branch_is_wired_per_pft():
    src = (ROOT / "ibis" / "timestepping" / "loops.py").read_text()
    assert "if sp.acclim_per_pft:" in src and "_row(sp.acclim_s0_pft)" in src and "_row(sp.acclim_smax_pft)" in src   # _row: (12,) or per-cell (npoi, 12) (CALIBRATION.md 45.1)


def test_per_pft_tables_accept_short_and_long_names():
    """Per-PFT yaml tables (coefm_pft, acclim_*_pft) may use the harness's short PFT names or the long ones;
    the s1_robust yaml used short names and read back as defaults before this was handled (CALIBRATION.md 34)."""
    import numpy as np
    from ibis.io.physics_params import _pft_vector, PFT_NAMES, PFT_SHORT_NAMES
    v = np.asarray(_pft_vector({'trop_bl_ever': 10.3, 'temperate_conifer_evergreen': 4.6, 'c3_grass': 10.5}, [9.0] * 12))
    assert v[0] == 10.3 and v[3] == 4.6 and v[11] == 10.5 and v[1] == 9.0
    assert len(PFT_SHORT_NAMES) == len(PFT_NAMES) == 12
    try:
        _pft_vector({'not_a_pft': 1.0}, [9.0] * 12); assert False, 'unknown name accepted'
    except KeyError:
        pass
