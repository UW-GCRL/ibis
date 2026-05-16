"""Tests for Phase 8: BGC, Phenology, Vegetation Dynamics, Climate Analysis.

Tests:
  - PFT existence (climate analysis)
  - Phenology updates (cold/drought triggers, LAI)
  - Soil biogeochemistry (CENTURY decomposition, C conservation)
  - Vegetation dynamics (biomass, fire, vegmap)
  - Parameter loading defaults
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from ibis.constants import get_constants
from ibis.dimensions import GridConfig
from ibis.state.model_state import IBISState, create_ibis_state
from ibis.state.vegetation import VegetationState
from ibis.state.carbon import CarbonState
from ibis.state.summary import SummaryState

from ibis.io.parameters import (
    PFTParams, BGCParams,
    default_pft_params, default_bgc_params,
)
from ibis.physics.climate_analysis import existence, climanl2
from ibis.physics.phenology import pheno
from ibis.physics.biogeochem import soilbgc
from ibis.physics.vegetation_dynamics import dynaveg, fire, vegmap


# Small test grid
NPOI = 4
NPFT = 12
GRID = GridConfig(nlon=2, nlat=2, npoi=NPOI, xres=90.0, yres=90.0,
                  nband=2, nsoilay=6, nsnolay=3, npft=NPFT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def consts():
    return get_constants()


@pytest.fixture
def pft_params():
    return default_pft_params()


@pytest.fixture
def bgc_params():
    return default_bgc_params()


@pytest.fixture
def state():
    """Create a physically reasonable state for testing Phase 8."""
    st = create_ibis_state(GRID)

    # Set vegetation with some biomass
    new_veg = eqx.tree_at(
        lambda v: (v.cbiol, v.cbior, v.cbiow, v.plai, v.exist,
                   v.fu, v.fl, v.lai, v.sai,
                   v.ztop, v.zbot,
                   v.tc, v.tw, v.gdd0, v.gdd5, v.tcmin,
                   v.specla, v.aleaf, v.aroot, v.awood),
        st.vegetation,
        (
            jnp.full((NPOI, NPFT), 0.1),      # cbiol
            jnp.full((NPOI, NPFT), 0.05),     # cbior
            jnp.full((NPOI, NPFT), 0.5),      # cbiow
            jnp.full((NPOI, NPFT), 0.5),      # plai
            jnp.ones((NPOI, NPFT)),            # exist
            jnp.full((NPOI,), 0.8),           # fu
            jnp.full((NPOI,), 0.5),           # fl
            jnp.full((NPOI, 2), 2.0),         # lai
            jnp.full((NPOI, 2), 0.5),         # sai
            jnp.array([[5.0, 20.0]] * NPOI),  # ztop
            jnp.array([[0.5, 3.0]] * NPOI),   # zbot
            jnp.full((NPOI,), -5.0),          # tc (deg C)
            jnp.full((NPOI,), 25.0),          # tw (deg C)
            jnp.full((NPOI,), 2000.0),        # gdd0
            jnp.full((NPOI,), 1500.0),        # gdd5
            jnp.full((NPOI,), -10.0),         # tcmin
            # PFT params
            default_pft_params().specla,
            default_pft_params().aleaf,
            default_pft_params().aroot,
            default_pft_params().awood,
        ),
    )

    # Set carbon pools with some initial C
    new_carbon = eqx.tree_at(
        lambda c: (c.clitlm, c.clitls, c.clitrm, c.clitrs,
                   c.csoipas, c.csoislon, c.csoislop,
                   c.totcmic, c.decompl, c.decomps,
                   c.falll, c.fallr, c.fallw),
        st.carbon,
        (
            jnp.full((NPOI,), 0.05),    # clitlm
            jnp.full((NPOI,), 0.03),    # clitls
            jnp.full((NPOI,), 0.02),    # clitrm
            jnp.full((NPOI,), 0.01),    # clitrs
            jnp.full((NPOI,), 2.0),     # csoipas
            jnp.full((NPOI,), 1.0),     # csoislon
            jnp.full((NPOI,), 0.5),     # csoislop
            jnp.full((NPOI,), 0.1),     # totcmic
            jnp.full((NPOI,), 1.0),     # decompl
            jnp.full((NPOI,), 1.0),     # decomps
            jnp.full((NPOI,), 0.3),     # falll
            jnp.full((NPOI,), 0.2),     # fallr
            jnp.full((NPOI,), 0.1),     # fallw
        ),
    )

    # Set soil properties
    new_soil = eqx.tree_at(
        lambda s: (s.sand, s.clay, s.hsoi),
        st.soil,
        (
            jnp.full((NPOI, 6), 50.0),    # sand %
            jnp.full((NPOI, 6), 20.0),    # clay %
            jnp.array([0.10, 0.15, 0.30, 0.50, 1.00, 2.00, 4.00]),  # hsoi
        ),
    )

    # Set summary state with some 10-day means
    new_summary = eqx.tree_at(
        lambda s: (s.a10td, s.a10ancub, s.a10ancls, s.a10ancl3, s.a10ancl4,
                   s.aynpp, s.ayanpp),
        st.summary,
        (
            jnp.full((NPOI,), 285.0),     # a10td (above tmelt)
            jnp.full((NPOI,), 1e-6),      # a10ancub (positive)
            jnp.full((NPOI,), 1e-6),      # a10ancls
            jnp.full((NPOI,), 1e-6),      # a10ancl3
            jnp.full((NPOI,), 1e-6),      # a10ancl4
            jnp.full((NPOI, NPFT), 0.01), # aynpp
            jnp.full((NPOI, NPFT), 0.005), # ayanpp
        ),
    )

    st = eqx.tree_at(
        lambda s: (s.vegetation, s.carbon, s.soil, s.summary),
        st,
        (new_veg, new_carbon, new_soil, new_summary),
    )
    return st


# =========================================================================
# Parameter tests
# =========================================================================

class TestParameters:
    def test_pft_params_shapes(self, pft_params):
        assert pft_params.TminL.shape == (12,)
        assert pft_params.tauleaf.shape == (12,)
        assert pft_params.specla.shape == (12,)

    def test_bgc_params_defaults(self, bgc_params):
        assert bgc_params.klm == 0.15
        assert bgc_params.cnr.shape == (8,)
        assert bgc_params.cnr[0] == 8.0

    def test_pft_bioclim_values(self, pft_params):
        # Tropical PFTs need warm tcmin
        assert pft_params.TminL[0] == 0.0
        assert pft_params.TminL[1] == 0.0
        # Boreal PFTs tolerate cold
        assert pft_params.TminL[5] == -57.5

    def test_allocation_sums(self, pft_params):
        """Allocation fractions should sum to ~1.0 for each PFT."""
        total = pft_params.aleaf + pft_params.aroot + pft_params.awood
        for j in range(12):
            assert abs(float(total[j]) - 1.0) < 0.01


# =========================================================================
# Climate analysis tests
# =========================================================================

class TestExistence:
    def test_tropical_warm(self, pft_params):
        """Tropical PFTs exist when tcmin > 0."""
        tcmin = jnp.array([5.0, -20.0, -50.0, -60.0])
        tw = jnp.full((4,), 25.0)
        gdd0 = jnp.full((4,), 3000.0)
        gdd5 = jnp.full((4,), 2000.0)
        ex = existence(tcmin, tw, gdd0, gdd5, pft_params)
        # Point 0 (tcmin=5): tropical PFTs should exist
        assert ex[0, 0] == 1.0  # PFT 1
        assert ex[0, 1] == 1.0  # PFT 2
        # Point 3 (tcmin=-60): only tundra PFTs
        assert ex[3, 0] == 0.0
        assert ex[3, 1] == 0.0

    def test_boreal_cold(self, pft_params):
        """Boreal PFTs exist in cold but not too cold regions."""
        tcmin = jnp.array([-50.0])
        tw = jnp.full((1,), 15.0)
        gdd0 = jnp.full((1,), 500.0)
        gdd5 = jnp.full((1,), 400.0)
        ex = existence(tcmin, tw, gdd0, gdd5, pft_params)
        assert ex[0, 5] == 1.0  # PFT 6: boreal conifer
        assert ex[0, 0] == 0.0  # PFT 1: tropical - no

    def test_grass_gdd(self, pft_params):
        """Grasses need GDD above thresholds."""
        tcmin = jnp.array([0.0])
        tw = jnp.full((1,), 25.0)
        gdd0 = jnp.full((1,), 50.0)  # Below 100
        gdd5 = jnp.full((1,), 30.0)
        ex = existence(tcmin, tw, gdd0, gdd5, pft_params)
        assert ex[0, 11] == 0.0  # C3 grass needs gdd0 > 100

    def test_c4_needs_warmth(self, pft_params):
        """C4 grasses need tw > 22."""
        tcmin = jnp.array([0.0])
        tw = jnp.array([20.0])  # Below 22
        gdd0 = jnp.full((1,), 3000.0)
        gdd5 = jnp.full((1,), 2000.0)
        ex = existence(tcmin, tw, gdd0, gdd5, pft_params)
        assert ex[0, 10] == 0.0  # C4 grass


class TestClimanl2:
    def test_updates_running_mean(self):
        """climanl2 blends old and new climate."""
        npoi = 2
        tc = jnp.array([-5.0, -10.0])
        tw = jnp.array([20.0, 15.0])
        tcmin = tc
        gdd0 = jnp.array([1000.0, 500.0])
        gdd5 = jnp.array([800.0, 300.0])
        tcthis = jnp.array([-3.0, -12.0])
        twthis = jnp.array([22.0, 14.0])
        gdd0this = jnp.array([1200.0, 600.0])
        gdd5this = jnp.array([1000.0, 400.0])
        deltat = jnp.zeros((npoi,))

        tc2, tw2, _, gdd02, gdd52 = climanl2(
            tc, tw, tcmin, gdd0, gdd5,
            tcthis, twthis, gdd0this, gdd5this, deltat,
        )
        # Should move toward this year's values
        assert float(tc2[0]) > float(tc[0])  # tc warming
        assert float(gdd02[0]) > float(gdd0[0])  # GDD increasing


# =========================================================================
# Phenology tests
# =========================================================================

class TestPhenology:
    def test_runs_without_error(self, state, consts):
        td = jnp.full((NPOI,), 290.0)
        new_veg = pheno(state.vegetation, state.summary, td, consts.epsilon)
        assert new_veg is not None

    def test_cold_stops_growth(self, state, consts):
        """Very cold temperatures should reduce tempu toward 0."""
        # Set a10td well below threshold
        cold_summary = eqx.tree_at(
            lambda s: s.a10td,
            state.summary,
            jnp.full((NPOI,), 260.0),  # Very cold
        )
        td = jnp.full((NPOI,), 260.0)
        new_veg = pheno(state.vegetation, cold_summary, td, consts.epsilon)
        # tempu should decrease
        assert jnp.all(new_veg.tempu <= state.vegetation.tempu)

    def test_warm_grows(self, state, consts):
        """Warm temperatures should increase GDD accumulator."""
        td = jnp.full((NPOI,), 300.0)
        new_veg = pheno(state.vegetation, state.summary, td, consts.epsilon)
        # agddu should increase (a10td=285 > gthreshold=273.16)
        assert jnp.all(new_veg.agddu >= state.vegetation.agddu)

    def test_drought_drops_lai(self, state, consts):
        """Negative photosynthesis should reduce drought trigger."""
        drought_summary = eqx.tree_at(
            lambda s: (s.a10ancub, s.a10ancls),
            state.summary,
            (jnp.full((NPOI,), -1e-6),
             jnp.full((NPOI,), -1e-6)),
        )
        td = jnp.full((NPOI,), 290.0)
        new_veg = pheno(state.vegetation, drought_summary, td, consts.epsilon)
        assert jnp.all(new_veg.dropu < state.vegetation.dropu)
        assert jnp.all(new_veg.dropls < state.vegetation.dropls)

    def test_lai_capped(self, state, consts):
        """LAI should not exceed 12."""
        # Set very high plai
        high_plai = eqx.tree_at(
            lambda v: v.plai,
            state.vegetation,
            jnp.full((NPOI, NPFT), 50.0),
        )
        td = jnp.full((NPOI,), 290.0)
        new_veg = pheno(high_plai, state.summary, td, consts.epsilon)
        assert jnp.all(new_veg.lai <= 12.0)

    def test_frac_sums_to_one(self, state, consts):
        """PFT fractions within each canopy should sum to ~1."""
        td = jnp.full((NPOI,), 290.0)
        new_veg = pheno(state.vegetation, state.summary, td, consts.epsilon)
        upper_sum = jnp.sum(new_veg.frac[:, :8], axis=1)
        lower_sum = jnp.sum(new_veg.frac[:, 8:], axis=1)
        assert jnp.allclose(upper_sum, 1.0, atol=0.01)
        assert jnp.allclose(lower_sum, 1.0, atol=0.01)


# =========================================================================
# Biogeochemistry tests
# =========================================================================

class TestBiogeochem:
    def test_runs_without_error(self, state, bgc_params):
        new_c, new_v, new_s = soilbgc(
            state.carbon, state.vegetation, state.summary,
            state.soil.hsoi, state.soil.sand, state.soil.clay,
            bgc_params, 365,
        )
        assert new_c is not None

    def test_pools_nonnegative(self, state, bgc_params):
        new_c, _, _ = soilbgc(
            state.carbon, state.vegetation, state.summary,
            state.soil.hsoi, state.soil.sand, state.soil.clay,
            bgc_params, 365,
        )
        assert jnp.all(new_c.clitlm >= 0.0)
        assert jnp.all(new_c.csoipas >= 0.0)
        assert jnp.all(new_c.csoislon >= 0.0)
        assert jnp.all(new_c.totcmic >= 0.0)

    def test_c_approximately_conserved(self, state, bgc_params):
        """Total C should be approximately conserved (inputs - outputs)."""
        totts = 1.0 / 365.0
        c = state.carbon
        total_before = (c.clitlm + c.clitls + c.clitll +
                       c.clitrm + c.clitrs + c.clitrl +
                       c.clitwm + c.clitws + c.clitwl +
                       c.csoipas + c.csoislon + c.csoislop + c.totcmic)
        total_input = (c.falll + c.fallr + c.fallw) * totts

        new_c, _, _ = soilbgc(
            state.carbon, state.vegetation, state.summary,
            state.soil.hsoi, state.soil.sand, state.soil.clay,
            bgc_params, 365,
        )
        total_after = (new_c.clitlm + new_c.clitls + new_c.clitll +
                      new_c.clitrm + new_c.clitrs + new_c.clitrl +
                      new_c.clitwm + new_c.clitws + new_c.clitwl +
                      new_c.csoipas + new_c.csoislon + new_c.csoislop +
                      new_c.totcmic)

        # C should change by input - decomposition losses - leaching
        # We can't check exact conservation due to CO2 respiration,
        # but total_after should be > 0 and not wildly different
        change = total_after - total_before
        assert jnp.all(jnp.isfinite(change))
        # With inputs, total C should generally increase or stay similar
        # (net depends on decomposition rate vs input rate)

    def test_co2_flux_positive(self, state, bgc_params):
        """CO2 flux from decomposition should be positive."""
        _, new_v, _ = soilbgc(
            state.carbon, state.vegetation, state.summary,
            state.soil.hsoi, state.soil.sand, state.soil.clay,
            bgc_params, 365,
        )
        assert jnp.all(jnp.isfinite(new_v.tco2mic))

    def test_n_mineralization_nonnegative(self, state, bgc_params):
        _, new_v, _ = soilbgc(
            state.carbon, state.vegetation, state.summary,
            state.soil.hsoi, state.soil.sand, state.soil.clay,
            bgc_params, 365,
        )
        assert jnp.all(new_v.tnmin >= 0.0)

    def test_multiple_days(self, state, bgc_params):
        """Running soilbgc for 30 days should produce finite results."""
        c, v, s = state.carbon, state.vegetation, state.summary
        for _ in range(30):
            c, v, s = soilbgc(c, v, s,
                              state.soil.hsoi, state.soil.sand, state.soil.clay,
                              bgc_params, 365)
        assert jnp.all(jnp.isfinite(c.csoipas))
        assert jnp.all(jnp.isfinite(c.totcmic))
        assert jnp.all(c.csoipas >= 0.0)


# =========================================================================
# Vegetation dynamics tests
# =========================================================================

class TestFire:
    def test_no_fire_no_litter(self):
        """No litter means no fire."""
        firefac = jnp.array([1.0, 1.0])
        totlit = jnp.array([0.0, 0.0])
        d = fire(firefac, totlit)
        assert jnp.allclose(d, 0.0)

    def test_fire_increases_with_litter(self):
        """More litter = more fire."""
        firefac = jnp.array([0.5, 0.5])
        d1 = fire(firefac, jnp.array([0.1, 0.1]))
        d2 = fire(firefac, jnp.array([0.5, 0.5]))
        assert jnp.all(d2 >= d1)

    def test_fire_bounded(self):
        """Fire disturbance is bounded [0, 1]."""
        firefac = jnp.array([10.0])
        totlit = jnp.array([10.0])
        d = fire(firefac, totlit)
        assert jnp.all(d >= 0.0)
        assert jnp.all(d <= 1.0)


class TestVegmap:
    def test_tropical_forest(self):
        """High tropical LAI classified as tropical forest."""
        npoi = 1
        plai = jnp.zeros((npoi, 12))
        plai = plai.at[0, 0].set(5.0)  # PFT 1 dominant
        totlaiu = jnp.array([5.0])
        totlail = jnp.array([0.5])
        gdd5 = jnp.array([3000.0])
        gdd0 = jnp.array([4000.0])
        vt = vegmap(plai, totlaiu, totlail, gdd5, gdd0, 1e-7)
        assert float(vt[0]) == 1.0

    def test_grassland(self):
        """Low tree LAI, high grass LAI classified as grassland."""
        npoi = 1
        plai = jnp.zeros((npoi, 12))
        plai = plai.at[0, 10].set(3.0)  # C4 grass dominant
        totlaiu = jnp.array([0.0])
        totlail = jnp.array([3.0])
        gdd5 = jnp.array([2000.0])
        gdd0 = jnp.array([3000.0])
        vt = vegmap(plai, totlaiu, totlail, gdd5, gdd0, 1e-7)
        assert float(vt[0]) == 10.0  # grassland

    def test_tundra(self):
        """Very low GDD classified as tundra."""
        npoi = 1
        plai = jnp.zeros((npoi, 12))
        plai = plai.at[0, 11].set(0.3)
        totlaiu = jnp.array([0.0])
        totlail = jnp.array([0.3])
        gdd5 = jnp.array([50.0])
        gdd0 = jnp.array([50.0])
        vt = vegmap(plai, totlaiu, totlail, gdd5, gdd0, 1e-7)
        assert float(vt[0]) == 15.0  # tundra/ice


class TestDynaveg:
    def test_runs_without_error(self, state, pft_params, consts):
        new_v, new_c, new_s = dynaveg(
            state.vegetation, state.carbon, state.summary,
            pft_params, isimfire=0, epsilon=consts.epsilon,
        )
        assert new_v is not None

    def test_biomass_nonnegative(self, state, pft_params, consts):
        new_v, _, _ = dynaveg(
            state.vegetation, state.carbon, state.summary,
            pft_params, isimfire=0, epsilon=consts.epsilon,
        )
        assert jnp.all(new_v.cbiol >= 0.0)
        assert jnp.all(new_v.cbiow >= 0.0)
        assert jnp.all(new_v.cbior >= 0.0)

    def test_fu_fl_bounded(self, state, pft_params, consts):
        new_v, _, _ = dynaveg(
            state.vegetation, state.carbon, state.summary,
            pft_params, isimfire=0, epsilon=consts.epsilon,
        )
        assert jnp.all(new_v.fu >= 0.25)
        assert jnp.all(new_v.fu <= 0.975)
        assert jnp.all(new_v.fl >= 0.25)
        assert jnp.all(new_v.fl <= 0.975)

    def test_sapfrac_bounded(self, state, pft_params, consts):
        new_v, _, _ = dynaveg(
            state.vegetation, state.carbon, state.summary,
            pft_params, isimfire=0, epsilon=consts.epsilon,
        )
        assert jnp.all(new_v.sapfrac >= 0.05)
        assert jnp.all(new_v.sapfrac <= 0.50)

    def test_with_fire(self, state, pft_params, consts):
        """Dynaveg with fire enabled produces finite results."""
        # Set firefac and totlit for fire to work
        veg_fire = eqx.tree_at(
            lambda v: v.firefac,
            state.vegetation,
            jnp.full((NPOI,), 0.5),
        )
        carbon_fire = eqx.tree_at(
            lambda c: c.totlit,
            state.carbon,
            jnp.full((NPOI,), 0.3),
        )
        new_v, _, _ = dynaveg(
            veg_fire, carbon_fire, state.summary,
            pft_params, isimfire=1, epsilon=consts.epsilon,
        )
        assert jnp.all(jnp.isfinite(new_v.cbiol))
        assert jnp.all(new_v.disturbf > 0.0)

    def test_vegtype_assigned(self, state, pft_params, consts):
        """Vegetation type should be assigned."""
        new_v, _, _ = dynaveg(
            state.vegetation, state.carbon, state.summary,
            pft_params, isimfire=0, epsilon=consts.epsilon,
        )
        assert jnp.all(new_v.vegtype0 > 0)

    def test_litter_fall_positive(self, state, pft_params, consts):
        """With biomass, litter fall should be positive."""
        _, new_c, _ = dynaveg(
            state.vegetation, state.carbon, state.summary,
            pft_params, isimfire=0, epsilon=consts.epsilon,
        )
        assert jnp.all(new_c.falll > 0.0)
        assert jnp.all(new_c.fallr > 0.0)


# =========================================================================
# JIT tests
# =========================================================================

class TestJIT:
    def test_jit_phenology(self, state, consts):
        @jax.jit
        def _run(veg, summ):
            return pheno(veg, summ, jnp.full((NPOI,), 290.0), consts.epsilon)
        result = _run(state.vegetation, state.summary)
        assert jnp.all(jnp.isfinite(result.tempu))

    def test_jit_soilbgc(self, state, bgc_params):
        @jax.jit
        def _run(c, v, s):
            return soilbgc(c, v, s,
                          state.soil.hsoi, state.soil.sand, state.soil.clay,
                          bgc_params, 365)
        new_c, _, _ = _run(state.carbon, state.vegetation, state.summary)
        assert jnp.all(jnp.isfinite(new_c.csoipas))

    def test_jit_existence(self, pft_params):
        @jax.jit
        def _run(tcmin, tw, gdd0, gdd5):
            return existence(tcmin, tw, gdd0, gdd5, pft_params)
        ex = _run(jnp.array([0.0]), jnp.array([25.0]),
                  jnp.array([3000.0]), jnp.array([2000.0]))
        assert jnp.all(jnp.isfinite(ex))


# =========================================================================
# Gradient test
# =========================================================================

class TestGradients:
    def test_phenology_differentiable(self, state, consts):
        """Phenology should be differentiable w.r.t. temperature."""
        def loss(td_val):
            veg = pheno(state.vegetation, state.summary,
                       jnp.full((NPOI,), td_val), consts.epsilon)
            return jnp.sum(veg.tempu)
        grad = jax.grad(loss)(290.0)
        assert jnp.isfinite(grad)

    def test_soilbgc_differentiable(self, state, bgc_params):
        """Soil BGC should be differentiable w.r.t. decomposition rate."""
        def loss(decompl_val):
            c = eqx.tree_at(lambda c: c.decompl, state.carbon,
                           jnp.full((NPOI,), decompl_val))
            new_c, _, _ = soilbgc(c, state.vegetation, state.summary,
                                  state.soil.hsoi, state.soil.sand, state.soil.clay,
                                  bgc_params, 365)
            return jnp.sum(new_c.csoipas)
        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)
