"""Sunlit/shaded two-big-leaf canopy (StomataParams.sunshade_u/_l; CALIBRATION.md 27).

The two-stream's absorbed-PAR profile per unit plant area, A e^-kx + B e^-hx + C e^hx, must integrate to the flux the
two-stream says the layer absorbs; the sunlit/shaded partition must conserve light, leaf area and capacity; with uniform
light and capacity the two big leaves must equal one; the switch must change the canopy rate only when on, give nothing
without leaves, refuse to run without the beam term, and round-trip through the parameter file."""
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _two_stream(coszen, lai, sai, solad, solai, asur=0.1):
    """Upper-canopy two-stream for the VIS band with IBIS's leaf optics: the scaling coefficients, the unscattered-beam
    term, the exponents and the flux the layer absorbs (per unit ground)."""
    import ibis.physics.radiation as RD
    n = len(coszen); z = jnp.asarray(coszen, jnp.float32); L = jnp.asarray(lai, jnp.float32); S = jnp.asarray(sai, jnp.float32)
    zeros = jnp.zeros(n, jnp.float32); t = jnp.full((n,), 293.0, jnp.float32); sunlit = z > 0.0
    omega, betad, betai, avmu, gdir = RD.twoset(z, L, S, zeros, zeros, t, 0.10, 0.05, 0.0, 0.0, RD._OMEGASNO[0], 273.16, 2, sunlit,
                                                fwet_stem=zeros, rliq_stem=zeros, temp_stem=t)
    a = jnp.full((n,), asur, jnp.float32)
    ts = RD.twostr(z, L, S, a, a, omega, betad, betai, avmu, gdir, 1.0e-7, sunlit, 0)
    sd = jnp.asarray(solad, jnp.float32); si = jnp.asarray(solai, jnp.float32)
    A = ts.term[:, 0] * sd; B = ts.term[:, 1] * sd + ts.term[:, 2] * si; C = ts.term[:, 3] * sd + ts.term[:, 4] * si
    scal = jnp.stack([A, B, C, A + B + C], -1); k, h = ts.term[:, 5], ts.term[:, 6]
    dbeam = (1.0 - ts.omega) * k * sd
    absorbed = sd * ts.abvegd + si * ts.abvegi
    return scal, dbeam, k, h, L, S, absorbed


CASES = dict(coszen=[0.95, 0.6, 0.3, 0.15, 0.8], lai=[4.7, 1.0, 8.0, 3.0, 0.3], sai=[0.85, 0.2, 1.5, 0.5, 0.1],
             solad=[200.0, 300.0, 50.0, 100.0, 0.0], solai=[100.0, 50.0, 150.0, 100.0, 120.0])


def test_two_stream_profile_integrates_to_the_absorbed_flux():
    """A e^-kx + B e^-hx + C e^hx over the plant area is what the two-stream absorbs (leaves and stems)."""
    from ibis.physics.stomata import sunshade_canopy
    scal, dbeam, k, h, L, S, absorbed = _two_stream(**CASES)
    apar_sun, apar_sha, *_ = sunshade_canopy(scal, dbeam, k, h, L, S, 0.15, 1.0e-7)
    layer = np.asarray((apar_sun + apar_sha) * (L + S) / L)               # leaves' share -> whole layer
    np.testing.assert_allclose(layer, np.asarray(absorbed), rtol=2e-3, atol=1e-3)


def test_partition_conserves_light_leaf_area_and_capacity():
    from ibis.physics.stomata import sunshade_canopy
    scal, dbeam, k, h, L, S, _ = _two_stream(**CASES); kn = 0.15
    apar_sun, apar_sha, lsun, lsha, vsun, vsha = (np.asarray(x) for x in sunshade_canopy(scal, dbeam, k, h, L, S, kn, 1.0e-7))
    L = np.asarray(L)
    assert np.all(apar_sun >= 0) and np.all(apar_sha >= 0) and np.all(lsun >= 0) and np.all(lsha >= 0) and np.all(vsun >= 0) and np.all(vsha >= 0)
    np.testing.assert_allclose(lsun + lsha, L, rtol=1e-6)
    np.testing.assert_allclose(vsun + vsha, (1.0 - np.exp(-kn * L)) / kn, rtol=1e-5)
    assert np.all(apar_sun / lsun >= apar_sha / lsha * 0.999)              # a sunlit leaf never gets less than a shaded one
    assert np.all(vsun / lsun >= vsha / lsha)                               # sunlit leaves sit higher: more capacity each
    beam = np.asarray(CASES['solad']) > 0
    assert np.all((apar_sun / lsun)[beam] > 1.5 * (apar_sha / lsha)[beam])   # with a beam, much more


def test_uniform_light_and_capacity_reduce_to_one_big_leaf():
    """Only the diffuse term, with no attenuation and Kn 0: every leaf alike, so the two big leaves equal one."""
    from ibis.physics.stomata import StomataParams, farquhar_c3, sunshade_canopy, _sunshade_c3
    p = StomataParams(); n = 3
    L = jnp.array([1.0, 4.7, 8.0], jnp.float32); S = jnp.array([0.1, 0.85, 1.5], jnp.float32)
    B = jnp.array([30.0, 20.0, 10.0], jnp.float32); z = jnp.zeros(n, jnp.float32)
    scal = jnp.stack([z, B, z, B], -1); k = jnp.full((n,), 0.6, jnp.float32); h = jnp.full((n,), 1.0e-7, jnp.float32)
    ss = sunshade_canopy(scal, z, k, h, L, S, 0.0, 1.0e-7)
    ci = jnp.full((n,), 280e-6, jnp.float32); vmax = jnp.full((n,), 45e-6, jnp.float32); rdark = 0.015 * vmax
    gam = jnp.full((n,), 40e-6, jnp.float32); kc = jnp.full((n,), 300e-6, jnp.float32); ko = jnp.full((n,), 0.3, jnp.float32)
    agc, anc = _sunshade_c3(ss, ci, vmax, gam, kc, ko, p.o2conc, rdark, L, p)
    fl = L / (L + S); ag1, an1 = farquhar_c3(B * (L + S) * fl, ci, vmax * L, gam, kc, ko, p.o2conc, rdark * L, p)
    np.testing.assert_allclose(np.asarray(agc), np.asarray(ag1 / L), rtol=1e-4)
    np.testing.assert_allclose(np.asarray(anc), np.asarray(an1 / L), rtol=1e-4)


def _stomata_inputs(scal, dbeam, k, h, L, S):
    n = L.shape[0]; t = jnp.full((n,), 298.0, jnp.float32); z = jnp.zeros((n,), jnp.float32)
    exist = jnp.zeros((n, 12)).at[:, 4].set(1.0).at[:, 5].set(1.0).at[:, 9].set(1.0)   # temperate deciduous broadleaf, boreal conifer, deciduous shrub
    return dict(tu=t, tl=t, t12=t, t34=t, q12=jnp.full((n,), 0.012), q34=jnp.full((n,), 0.012), su=jnp.full((n,), 0.05), sl=jnp.full((n,), 0.03),
                psurf=jnp.full((n,), 101325.0), topparu=scal[:, 3] * L / (L + S), topparl=0.3 * scal[:, 3] * L / (L + S),
                termu6=k, termu7=h, terml6=k, terml7=h, scalcoefu=scal, scalcoefl=0.3 * scal,
                ci_upper=jnp.full((n, 2), 280.0e-6), gs_upper=jnp.full((n, 2), 0.05), cs_upper=jnp.full((n, 2), 340.0e-6),
                ci_lower=jnp.full((n, 3), 280.0e-6), gs_lower=jnp.full((n, 3), 0.05), cs_lower=jnp.full((n, 3), 340.0e-6),
                fwetu=z, fwetl=z, lai=jnp.stack([L, L], -1), sai=jnp.stack([S, S], -1), exist=exist,
                stresstu=jnp.ones((n,)), stresstl=jnp.ones((n,)), a10scalparamu=jnp.full((n,), 0.5), a10daylightu=jnp.full((n,), 100.0),
                a10scalparaml=jnp.full((n,), 0.5), a10daylightl=jnp.full((n,), 100.0), co2conc=380.0e-6, dtime=3600.0)


def test_switch_changes_the_canopy_only_when_on_and_needs_the_beam_term():
    from ibis.physics.stomata import StomataParams, stomata
    scal, dbeam, k, h, L, S, _ = _two_stream(coszen=[0.8, 0.8], lai=[4.7, 0.0], sai=[0.85, 0.85], solad=[200.0, 200.0], solai=[100.0, 100.0])
    x = _stomata_inputs(scal, dbeam, k, h, L, S)
    off = stomata(**x, params=StomataParams(), dbeamu=dbeam, dbeaml=0.3 * dbeam)
    off2 = stomata(**x, params=StomataParams())                                  # the beam term is ignored when off
    np.testing.assert_array_equal(np.asarray(off.ag_upper), np.asarray(off2.ag_upper))
    on = stomata(**x, params=StomataParams(sunshade_u=True, sunshade_l=True), dbeamu=dbeam, dbeaml=0.3 * dbeam)
    ag_off, ag_on = np.asarray(off.ag_upper[:, 0]), np.asarray(on.ag_upper[:, 0])
    assert ag_on[0] > 1.3 * ag_off[0]                                            # a dense, capacity-limited canopy gets its lower leaves back
    assert ag_on[1] == 0.0 and np.asarray(on.an_upper)[1, 0] == 0.0             # no leaves, no canopy rate
    assert np.all(np.asarray(on.ag_lower[0]) > 0.0)
    with pytest.raises(ValueError):
        stomata(**x, params=StomataParams(sunshade_u=True))


def test_parameter_file_round_trip():
    from ibis.io.physics_params import build_all_params, load_physics_params
    sp, _, _ = build_all_params(2, np.zeros(2, np.float32), load_physics_params(str(ROOT / "configs" / "params" / "default_seasonal_sunshade.yaml")))
    st = sp.stomata_params
    assert st.sunshade_u is True and st.sunshade_l is True and st.cap_scale_u is False and st.scale_leaf_only is False
    sp0, _, _ = build_all_params(2, np.zeros(2, np.float32), load_physics_params(str(ROOT / "configs" / "params" / "default.yaml")))
    assert sp0.stomata_params.sunshade_u is False and sp0.stomata_params.sunshade_l is False
