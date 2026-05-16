"""Tests for ibis.physics.stomata — Farquhar C3/C4 + Ball-Berry."""

import jax
import jax.numpy as jnp
import pytest

from ibis.physics.stomata import (
    StomataParams,
    colimit,
    farquhar_c3,
    farquhar_c4,
    ball_berry_step,
    canopy_scaling,
    drystress,
    stomata,
    _temp_dep_params,
    _select_vmax_upper,
    _total_conductance,
)
from ibis.constants import PhysicalConstants

PARAMS = StomataParams()
CONSTS = PhysicalConstants()
NPOI = 4


# ===========================================================================
# colimit
# ===========================================================================

class TestColimit:
    def test_equal_inputs(self):
        """When j1 == j2, result should be approximately j1 (for theta<1)."""
        j1 = jnp.array([10.0, 20.0])
        result = colimit(j1, j1, 0.95)
        # Result should be close to j1 but slightly less due to co-limitation
        assert jnp.all(result < j1)
        assert jnp.all(result > 0.8 * j1)

    def test_one_zero(self):
        """When one input is 0, result should be 0."""
        j1 = jnp.array([10.0])
        j2 = jnp.array([0.0])
        result = colimit(j1, j2, 0.95)
        assert jnp.allclose(result, 0.0, atol=1e-10)

    def test_theta_one(self):
        """When theta=1, result is min(j1, j2)."""
        j1 = jnp.array([10.0])
        j2 = jnp.array([5.0])
        result = colimit(j1, j2, 1.0)
        assert jnp.allclose(result, 5.0, atol=1e-6)

    def test_positive(self):
        """Result should always be positive for positive inputs."""
        j1 = jnp.array([1.0, 5.0, 0.1])
        j2 = jnp.array([2.0, 3.0, 0.5])
        result = colimit(j1, j2, 0.95)
        assert jnp.all(result > 0.0)

    def test_bounded(self):
        """Result should be bounded by min of inputs."""
        j1 = jnp.array([10.0, 3.0])
        j2 = jnp.array([5.0, 8.0])
        result = colimit(j1, j2, 0.95)
        assert jnp.all(result <= jnp.minimum(j1, j2) + 1e-10)


# ===========================================================================
# Temperature-dependent parameters
# ===========================================================================

class TestTempDep:
    def test_at_15c(self):
        """At 15C (288.16 K), kc/ko/tau should be close to 15C values."""
        t = jnp.array([288.16])
        kc, ko, tau, gamstar, tempvm_c3, tempvm_c4 = _temp_dep_params(t, PARAMS)
        assert jnp.allclose(kc, PARAMS.kc15, rtol=0.05)
        assert jnp.allclose(ko, PARAMS.ko15, rtol=0.05)
        assert jnp.allclose(tau, PARAMS.tau15, rtol=0.05)

    def test_gamstar_positive(self):
        """CO2 compensation point should be positive."""
        t = jnp.array([260.0, 280.0, 300.0, 320.0])
        _, _, _, gamstar, _, _ = _temp_dep_params(t, PARAMS)
        assert jnp.all(gamstar > 0.0)

    def test_tempvm_range(self):
        """Temperature scaling should be non-negative and finite."""
        t = jnp.array([260.0, 280.0, 300.0, 320.0])
        _, _, _, _, tempvm_c3, tempvm_c4 = _temp_dep_params(t, PARAMS)
        assert jnp.all(tempvm_c3 >= 0.0)
        assert jnp.all(jnp.isfinite(tempvm_c3))
        assert jnp.all(tempvm_c4 >= 0.0)
        assert jnp.all(jnp.isfinite(tempvm_c4))

    def test_c4_more_cold_sensitive(self):
        """C4 tempvm should be lower than C3 at cold temperatures."""
        t = jnp.array([278.0])  # 5C
        _, _, _, _, tempvm_c3, tempvm_c4 = _temp_dep_params(t, PARAMS)
        assert tempvm_c4 < tempvm_c3


# ===========================================================================
# Farquhar C3
# ===========================================================================

class TestFarquharC3:
    def test_outputs_positive(self):
        """Gross photosynthesis should be positive under light."""
        par = jnp.array([500.0])
        ci = jnp.array([250.0e-6])
        vmax = jnp.array([50.0e-6])
        gamstar = jnp.array([30.0e-6])
        kc = jnp.array([3.0e-4])
        ko = jnp.array([0.3])
        rdark = jnp.array([1.0e-6])
        ag, an = farquhar_c3(par, ci, vmax, gamstar, kc, ko, 0.209, rdark, PARAMS)
        assert jnp.all(ag > 0.0)
        assert jnp.all(an > -rdark)

    def test_dark(self):
        """In dark (PAR=0), ag should still be positive due to Rubisco."""
        par = jnp.array([0.0])
        ci = jnp.array([300.0e-6])
        vmax = jnp.array([50.0e-6])
        gamstar = jnp.array([30.0e-6])
        kc = jnp.array([3.0e-4])
        ko = jnp.array([0.3])
        rdark = jnp.array([1.0e-6])
        ag, an = farquhar_c3(par, ci, vmax, gamstar, kc, ko, 0.209, rdark, PARAMS)
        # With no light, je=0, so ag should be very small (limited by je)
        assert jnp.all(ag < rdark * 10)

    def test_light_response(self):
        """Higher PAR should give higher Ag (up to saturation)."""
        par_lo = jnp.array([100.0])
        par_hi = jnp.array([500.0])
        ci = jnp.array([300.0e-6])
        vmax = jnp.array([50.0e-6])
        gamstar = jnp.array([30.0e-6])
        kc = jnp.array([3.0e-4])
        ko = jnp.array([0.3])
        rdark = jnp.array([1.0e-6])
        ag_lo, _ = farquhar_c3(par_lo, ci, vmax, gamstar, kc, ko, 0.209, rdark, PARAMS)
        ag_hi, _ = farquhar_c3(par_hi, ci, vmax, gamstar, kc, ko, 0.209, rdark, PARAMS)
        assert ag_hi > ag_lo

    def test_co2_response(self):
        """Higher ci should give higher Ag."""
        par = jnp.array([500.0])
        ci_lo = jnp.array([200.0e-6])
        ci_hi = jnp.array([400.0e-6])
        vmax = jnp.array([50.0e-6])
        gamstar = jnp.array([30.0e-6])
        kc = jnp.array([3.0e-4])
        ko = jnp.array([0.3])
        rdark = jnp.array([1.0e-6])
        ag_lo, _ = farquhar_c3(par, ci_lo, vmax, gamstar, kc, ko, 0.209, rdark, PARAMS)
        ag_hi, _ = farquhar_c3(par, ci_hi, vmax, gamstar, kc, ko, 0.209, rdark, PARAMS)
        assert ag_hi > ag_lo


# ===========================================================================
# Farquhar C4
# ===========================================================================

class TestFarquharC4:
    def test_outputs_positive(self):
        """Gross photosynthesis should be positive under light."""
        par = jnp.array([500.0])
        ci = jnp.array([100.0e-6])
        vmax = jnp.array([15.0e-6])
        rdark = jnp.array([0.5e-6])
        ag, an = farquhar_c4(par, ci, vmax, rdark, PARAMS)
        assert jnp.all(ag > 0.0)
        assert jnp.all(an > -rdark)

    def test_co2_response(self):
        """C4 should respond to CO2 at low concentrations."""
        par = jnp.array([500.0])
        ci_lo = jnp.array([10.0e-6])
        ci_hi = jnp.array([200.0e-6])
        vmax = jnp.array([15.0e-6])
        rdark = jnp.array([0.5e-6])
        ag_lo, _ = farquhar_c4(par, ci_lo, vmax, rdark, PARAMS)
        ag_hi, _ = farquhar_c4(par, ci_hi, vmax, rdark, PARAMS)
        assert ag_hi > ag_lo


# ===========================================================================
# Ball-Berry step
# ===========================================================================

class TestBallBerry:
    def test_gs_bounded(self):
        """Stomatal conductance should be above minimum."""
        an = jnp.array([5.0e-6])
        cs = jnp.array([350.0e-6])
        gs = jnp.array([0.01])
        ci = jnp.array([300.0e-6])
        gbco2 = jnp.array([1.0])
        rh = jnp.array([0.7])
        gamstar_min = jnp.array([30.0e-6])
        cs_new, gs_new, ci_new = ball_berry_step(
            an, cs, gs, ci, 350.0e-6, gbco2, rh,
            10.0, 0.01, jnp.array([1.0]), 1e-5, gamstar_min, 2000e-6)
        assert jnp.all(gs_new >= 1e-5)

    def test_cs_bounded(self):
        """Surface CO2 should be above gamstar threshold."""
        an = jnp.array([5.0e-6])
        cs = jnp.array([350.0e-6])
        gs = jnp.array([0.01])
        ci = jnp.array([300.0e-6])
        gbco2 = jnp.array([1.0])
        rh = jnp.array([0.7])
        gamstar_min = jnp.array([30.0e-6])
        cs_new, gs_new, ci_new = ball_berry_step(
            an, cs, gs, ci, 350.0e-6, gbco2, rh,
            10.0, 0.01, jnp.array([1.0]), 1e-5, gamstar_min, 2000e-6)
        assert jnp.all(cs_new >= gamstar_min)

    def test_ci_bounded(self):
        """Intercellular CO2 should be in [gamstar, cimax]."""
        an = jnp.array([5.0e-6])
        cs = jnp.array([350.0e-6])
        gs = jnp.array([0.01])
        ci = jnp.array([300.0e-6])
        gbco2 = jnp.array([1.0])
        rh = jnp.array([0.7])
        gamstar_min = jnp.array([30.0e-6])
        cs_new, gs_new, ci_new = ball_berry_step(
            an, cs, gs, ci, 350.0e-6, gbco2, rh,
            10.0, 0.01, jnp.array([1.0]), 1e-5, gamstar_min, 2000e-6)
        assert jnp.all(ci_new >= gamstar_min)
        assert jnp.all(ci_new <= 2000e-6)


# ===========================================================================
# Canopy scaling
# ===========================================================================

class TestCanopyScaling:
    def test_no_lai(self):
        """With no LAI, scale should be 0."""
        toppar = jnp.array([500.0])
        term6 = jnp.array([0.5])
        term7 = jnp.array([0.3])
        scalcoef = jnp.array([[0.1, 0.2, 0.05, 0.5]])
        scale, _, _ = canopy_scaling(
            toppar, term6, term7, scalcoef,
            jnp.array([0.0]), jnp.array([0.0]),
            jnp.array([0.5]), jnp.array([100.0]),
            3600.0, 1e-7)
        assert jnp.allclose(scale, 0.0)

    def test_with_lai(self):
        """With LAI and daytime PAR, scale should be positive."""
        toppar = jnp.array([500.0])
        term6 = jnp.array([0.5])
        term7 = jnp.array([0.3])
        scalcoef = jnp.array([[0.1, 0.2, 0.05, 0.5]])
        scale, _, _ = canopy_scaling(
            toppar, term6, term7, scalcoef,
            jnp.array([3.0]), jnp.array([0.5]),
            jnp.array([0.5]), jnp.array([100.0]),
            3600.0, 1e-7)
        assert jnp.all(scale > 0.0)

    def test_running_mean_updates(self):
        """Running means should update during daytime."""
        toppar = jnp.array([500.0])
        term6 = jnp.array([0.5])
        term7 = jnp.array([0.3])
        scalcoef = jnp.array([[0.1, 0.2, 0.05, 0.5]])
        a10sp_old = jnp.array([0.5])
        a10dl_old = jnp.array([100.0])
        _, a10sp_new, a10dl_new = canopy_scaling(
            toppar, term6, term7, scalcoef,
            jnp.array([3.0]), jnp.array([0.5]),
            a10sp_old, a10dl_old, 3600.0, 1e-7)
        # Running means should have changed
        assert not jnp.allclose(a10sp_new, a10sp_old)


# ===========================================================================
# Drystress
# ===========================================================================

class TestDrystress:
    def test_fully_wet(self):
        """Fully wet soil should give stress near 1."""
        npoi = 2
        nsoilay = 6
        wsoi = jnp.full((npoi, nsoilay), 0.8)
        wisoi = jnp.zeros((npoi, nsoilay))
        swilt = jnp.full((npoi, nsoilay), 0.15)
        sfield = jnp.full((npoi, nsoilay), 0.45)
        froot = jnp.ones((nsoilay, 2)) / nsoilay  # uniform roots
        stresstl, stresstu, _, _ = drystress(wsoi, wisoi, swilt, sfield, froot)
        assert jnp.all(stresstl > 0.9)
        assert jnp.all(stresstu > 0.9)

    def test_at_wilting(self):
        """At wilting point, stress should be near 0."""
        npoi = 2
        nsoilay = 6
        swilt = jnp.full((npoi, nsoilay), 0.15)
        wsoi = swilt.copy()
        wisoi = jnp.zeros((npoi, nsoilay))
        sfield = jnp.full((npoi, nsoilay), 0.45)
        froot = jnp.ones((nsoilay, 2)) / nsoilay
        stresstl, stresstu, _, _ = drystress(wsoi, wisoi, swilt, sfield, froot)
        assert jnp.all(stresstl < 0.05)
        assert jnp.all(stresstu < 0.05)

    def test_bounded_01(self):
        """Stress should be in [0, 1]."""
        npoi = 3
        nsoilay = 6
        wsoi = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                           [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]])
        wisoi = jnp.zeros((npoi, nsoilay))
        swilt = jnp.full((npoi, nsoilay), 0.15)
        sfield = jnp.full((npoi, nsoilay), 0.45)
        froot = jnp.ones((nsoilay, 2)) / nsoilay
        stresstl, stresstu, _, _ = drystress(wsoi, wisoi, swilt, sfield, froot)
        assert jnp.all(stresstl >= 0.0)
        assert jnp.all(stresstl <= 1.0)
        assert jnp.all(stresstu >= 0.0)
        assert jnp.all(stresstu <= 1.0)

    def test_ice_reduces_stress(self):
        """Frozen soil (high wisoi) should reduce available water."""
        npoi = 2
        nsoilay = 6
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        swilt = jnp.full((npoi, nsoilay), 0.15)
        sfield = jnp.full((npoi, nsoilay), 0.45)
        froot = jnp.ones((nsoilay, 2)) / nsoilay
        # No ice
        stresstl_wet, _, _, _ = drystress(
            wsoi, jnp.zeros((npoi, nsoilay)), swilt, sfield, froot)
        # High ice
        stresstl_ice, _, _, _ = drystress(
            wsoi, jnp.full((npoi, nsoilay), 0.8), swilt, sfield, froot)
        assert jnp.all(stresstl_ice < stresstl_wet)


# ===========================================================================
# Vmax selection
# ===========================================================================

class TestVmaxSelection:
    def test_tropical(self):
        """With tropical PFT present, should select tropical vmax."""
        exist = jnp.zeros((1, 12))
        exist = exist.at[0, 0].set(1.0)  # PFT 0 = tropical broadleaf
        vmaxub, _ = _select_vmax_upper(exist, PARAMS)
        assert jnp.allclose(vmaxub, PARAMS.vmax_pft[0])

    def test_temperate_conifer(self):
        """With temperate conifer, should select temperate vmax."""
        exist = jnp.zeros((1, 12))
        exist = exist.at[0, 3].set(1.0)  # PFT 3 = temperate conifer
        _, vmaxuc = _select_vmax_upper(exist, PARAMS)
        assert jnp.allclose(vmaxuc, PARAMS.vmax_pft[3])

    def test_boreal_fallback(self):
        """Without temperate conifer, should fall back to boreal."""
        exist = jnp.zeros((1, 12))
        _, vmaxuc = _select_vmax_upper(exist, PARAMS)
        assert jnp.allclose(vmaxuc, PARAMS.vmax_pft[5])


# ===========================================================================
# Total conductance
# ===========================================================================

class TestTotalConductance:
    def test_positive(self):
        """Total conductance should be positive."""
        su = jnp.array([0.05, 0.1, 0.2])
        gsc = jnp.array([0.1, 0.5, 1.0])
        tc = _total_conductance(su, gsc)
        assert jnp.all(tc > 0.0)

    def test_less_than_either(self):
        """Series resistance: totcond < min(su, 0.029*gsc)."""
        su = jnp.array([0.05])
        gsc = jnp.array([0.5])
        tc = _total_conductance(su, gsc)
        assert tc < su[0]
        assert tc < 0.029 * gsc[0]


# ===========================================================================
# Full stomata function
# ===========================================================================

class TestStomata:
    @pytest.fixture
    def inputs(self):
        """Standard test inputs for the stomata function."""
        npoi = NPOI
        z = jnp.zeros((npoi,))
        t_init = jnp.full((npoi,), 298.0)
        return dict(
            tu=t_init, tl=t_init,
            t12=t_init, t34=t_init,
            q12=jnp.full((npoi,), 0.012),
            q34=jnp.full((npoi,), 0.012),
            su=jnp.full((npoi,), 0.05),
            sl=jnp.full((npoi,), 0.03),
            psurf=jnp.full((npoi,), 101325.0),
            topparu=jnp.full((npoi,), 400.0),
            topparl=jnp.full((npoi,), 200.0),
            termu6=jnp.full((npoi,), 0.5),
            termu7=jnp.full((npoi,), 0.3),
            terml6=jnp.full((npoi,), 0.5),
            terml7=jnp.full((npoi,), 0.3),
            scalcoefu=jnp.tile(jnp.array([0.1, 0.2, 0.05, 0.5]), (npoi, 1)),
            scalcoefl=jnp.tile(jnp.array([0.1, 0.2, 0.05, 0.5]), (npoi, 1)),
            ci_upper=jnp.full((npoi, 2), 300.0e-6),
            gs_upper=jnp.full((npoi, 2), 0.05),
            cs_upper=jnp.full((npoi, 2), 340.0e-6),
            ci_lower=jnp.full((npoi, 3), 300.0e-6),
            gs_lower=jnp.full((npoi, 3), 0.05),
            cs_lower=jnp.full((npoi, 3), 340.0e-6),
            fwetu=z, fwetl=z,
            lai=jnp.full((npoi, 2), 3.0),
            sai=jnp.full((npoi, 2), 0.5),
            exist=jnp.ones((npoi, 12)),
            stresstu=jnp.ones((npoi,)),
            stresstl=jnp.ones((npoi,)),
            a10scalparamu=jnp.full((npoi,), 0.5),
            a10daylightu=jnp.full((npoi,), 100.0),
            a10scalparaml=jnp.full((npoi,), 0.5),
            a10daylightl=jnp.full((npoi,), 100.0),
            co2conc=350.0e-6,
            dtime=3600.0,
            params=PARAMS,
        )

    def test_outputs_finite(self, inputs):
        """All outputs should be finite."""
        result = stomata(**inputs)
        for field in result:
            assert jnp.all(jnp.isfinite(field)), f"Non-finite in {field}"

    def test_photosynthesis_positive(self, inputs):
        """Gross photosynthesis should be positive under normal conditions."""
        result = stomata(**inputs)
        assert jnp.all(result.ag_upper >= 0.0)
        assert jnp.all(result.ag_lower >= 0.0)

    def test_ci_bounded(self, inputs):
        """Intercellular CO2 should be in valid range."""
        result = stomata(**inputs)
        assert jnp.all(result.ci_upper >= 0.0)
        assert jnp.all(result.ci_upper <= PARAMS.cimax)
        assert jnp.all(result.ci_lower >= 0.0)
        assert jnp.all(result.ci_lower <= PARAMS.cimax)

    def test_gs_positive(self, inputs):
        """Stomatal conductance should be positive."""
        result = stomata(**inputs)
        assert jnp.all(result.gs_upper > 0.0)
        assert jnp.all(result.gs_lower > 0.0)

    def test_totcond_positive(self, inputs):
        """Total conductance should be positive."""
        result = stomata(**inputs)
        assert jnp.all(result.totcondub > 0.0)
        assert jnp.all(result.totconduc > 0.0)
        assert jnp.all(result.totcondls > 0.0)
        assert jnp.all(result.totcondl3 > 0.0)
        assert jnp.all(result.totcondl4 > 0.0)

    def test_wet_canopy_reduces_photosynthesis(self, inputs):
        """Wet canopy fraction should reduce net photosynthesis."""
        result_dry = stomata(**inputs)
        wet_inputs = {**inputs, 'fwetu': jnp.full((NPOI,), 0.5),
                                'fwetl': jnp.full((NPOI,), 0.5)}
        result_wet = stomata(**wet_inputs)
        assert jnp.all(result_wet.an_upper <= result_dry.an_upper + 1e-15)
        assert jnp.all(result_wet.an_lower <= result_dry.an_lower + 1e-15)

    def test_stress_reduces_photosynthesis(self, inputs):
        """Soil moisture stress should reduce photosynthesis."""
        result_full = stomata(**inputs)
        stress_inputs = {**inputs,
                         'stresstu': jnp.full((NPOI,), 0.3),
                         'stresstl': jnp.full((NPOI,), 0.3)}
        result_stress = stomata(**stress_inputs)
        # Gross photosynthesis should be lower under stress
        assert jnp.all(result_stress.ag_upper <= result_full.ag_upper + 1e-15)

    def test_no_par_reduces_photosynthesis(self, inputs):
        """Zero PAR should give near-zero gross photosynthesis."""
        dark_inputs = {**inputs,
                       'topparu': jnp.zeros((NPOI,)),
                       'topparl': jnp.zeros((NPOI,))}
        result = stomata(**dark_inputs)
        # With no light, gross photosynthesis should be very small
        assert jnp.all(result.ag_upper < 1e-6)
        assert jnp.all(result.ag_lower < 1e-6)

    def test_jit_compatible(self, inputs):
        """Function should be JIT-compilable."""
        jitted = jax.jit(stomata, static_argnames=['params'])
        result = jitted(**inputs)
        assert jnp.all(jnp.isfinite(result.ag_upper))

    def test_differentiable(self, inputs):
        """Gradients should flow through stomata w.r.t. temperature."""
        def loss(tu_val):
            inp = {**inputs, 'tu': tu_val}
            result = stomata(**inp)
            return jnp.sum(result.ag_upper)
        grad_fn = jax.grad(loss)
        g = grad_fn(inputs['tu'])
        assert jnp.all(jnp.isfinite(g))
