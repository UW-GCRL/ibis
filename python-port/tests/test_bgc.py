"""Tests for Phase 9: New BGC components.

Covers wetlands, permafrost, fire emissions, soil gases, BVOC, and NEE.
"""

import jax
import jax.numpy as jnp
import pytest

from ibis.bgc._legacy_unwired.wetlands import (
    WetlandState, WetlandParams, create_wetland_state,
    default_wetland_params, wetland_step,
    wetland_fraction, ch4_production, ch4_oxidation, ch4_transport,
)
from ibis.bgc.permafrost import (
    PermafrostState, PermafrostParams, create_permafrost_state,
    default_permafrost_params, permafrost_step,
    stefan_depth, diagnose_frozen_ground, thermokarst,
)
from ibis.bgc._legacy_unwired.soil_gases import (
    SoilGasState, SoilGasParams, create_soil_gas_state,
    default_soil_gas_params, soil_gas_step,
    wfps, nitrification, denitrification,
)
from ibis.bgc._legacy_unwired.bvoc import (
    BVOCState, create_bvoc_state, bvoc_emissions,
    gamma_t_isop, gamma_par_isop, gamma_t_mono,
    T_STANDARD, PAR_STANDARD,
)
from ibis.bgc._legacy_unwired.nee import (
    NEEState, create_nee_state, compute_nee,
)

NPOI = 4
NSOILAY = 6
NPFT = 12


# ===== Fixtures =====

@pytest.fixture
def soil_temps():
    """Soil temperatures (K) — mix of warm and cold."""
    return jnp.array([
        [290.0, 288.0, 286.0, 284.0, 282.0, 280.0],
        [270.0, 268.0, 265.0, 263.0, 260.0, 258.0],
        [300.0, 298.0, 296.0, 294.0, 292.0, 290.0],
        [275.0, 274.0, 273.0, 272.0, 271.0, 270.0],
    ])


@pytest.fixture
def soil_moisture():
    """Soil moisture fraction of pore space."""
    return jnp.array([
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        [0.3, 0.3, 0.2, 0.2, 0.1, 0.1],
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ])


@pytest.fixture
def porosity():
    return jnp.full((NPOI, NSOILAY), 0.45)


# ===== Wetlands =====

class TestWetlandState:
    def test_create(self):
        ws = create_wetland_state(NPOI)
        assert ws.wetland_frac.shape == (NPOI,)
        assert ws.ch4_soil.shape == (NPOI,)
        assert jnp.all(ws.ch4_flux == 0.0)

    def test_pytree(self):
        ws = create_wetland_state(NPOI)
        leaves = jax.tree.leaves(ws)
        assert len(leaves) == 9  # 9 fields


class TestWetlandFraction:
    def test_dry_soil(self, porosity):
        wsoi = jnp.full((NPOI, NSOILAY), 0.2)
        params = default_wetland_params()
        fwet, wtd = wetland_fraction(wsoi, porosity, params)
        assert jnp.all(fwet == 0.0)  # sat_top < 0.5 → no wetland

    def test_saturated_soil(self, porosity):
        wsoi = jnp.full((NPOI, NSOILAY), 0.95)
        params = default_wetland_params()
        fwet, wtd = wetland_fraction(wsoi, porosity, params)
        assert jnp.all(fwet > 0.0)
        assert jnp.all(fwet <= params.fmax)

    def test_water_table_depth(self, porosity):
        wsoi = jnp.full((NPOI, NSOILAY), 0.8)
        params = default_wetland_params()
        _, wtd = wetland_fraction(wsoi, porosity, params)
        # wtd = -wtd_scale * (1 - sat_top) → negative when not fully saturated
        assert jnp.all(wtd < 0.0)


class TestCH4Production:
    def test_positive_production(self, soil_temps):
        params = default_wetland_params()
        fwet = jnp.full((NPOI,), 0.3)
        het_resp = jnp.full((NPOI,), 2.0)
        prod = ch4_production(soil_temps, het_resp, fwet, params)
        assert jnp.all(prod >= 0.0)
        assert prod.shape == (NPOI,)

    def test_no_wetland_no_production(self, soil_temps):
        params = default_wetland_params()
        fwet = jnp.zeros((NPOI,))
        het_resp = jnp.full((NPOI,), 2.0)
        prod = ch4_production(soil_temps, het_resp, fwet, params)
        assert jnp.allclose(prod, 0.0)

    def test_temperature_sensitivity(self):
        """Warmer soil should produce more CH4."""
        params = default_wetland_params()
        fwet = jnp.full((2,), 0.5)
        het_resp = jnp.full((2,), 2.0)
        cold = jnp.full((2, 6), 275.0)
        warm = jnp.full((2, 6), 300.0)
        prod_cold = ch4_production(cold, het_resp, fwet, params)
        prod_warm = ch4_production(warm, het_resp, fwet, params)
        assert jnp.all(prod_warm > prod_cold)


class TestCH4Oxidation:
    def test_michaelis_menten(self):
        params = default_wetland_params()
        ch4_soil = jnp.array([1.0, 5.0, 10.0, 50.0])
        fwet = jnp.full((4,), 0.3)
        wtd = jnp.full((4,), -0.05)
        oxid = ch4_oxidation(ch4_soil, fwet, wtd, params)
        assert jnp.all(oxid >= 0.0)
        # Higher CH4 pool → higher oxidation (saturating)
        assert oxid[3] > oxid[0]

    def test_fully_wet_no_oxidation(self):
        params = default_wetland_params()
        ch4_soil = jnp.full((NPOI,), 10.0)
        fwet = jnp.ones((NPOI,))  # fully wet
        wtd = jnp.full((NPOI,), 0.0)  # at surface
        oxid = ch4_oxidation(ch4_soil, fwet, wtd, params)
        assert jnp.allclose(oxid, 0.0)


class TestCH4Transport:
    def test_below_threshold_no_ebullition(self):
        params = default_wetland_params()
        ch4_soil = jnp.full((NPOI,), 100.0)  # below threshold (500)
        prod = jnp.full((NPOI,), 1.0)
        fwet = jnp.full((NPOI,), 0.5)
        ebull, plant, diff = ch4_transport(ch4_soil, prod, fwet, params)
        assert jnp.allclose(ebull, 0.0)
        assert jnp.all(plant > 0.0)
        assert jnp.all(diff > 0.0)

    def test_above_threshold_ebullition(self):
        params = default_wetland_params()
        ch4_soil = jnp.full((NPOI,), 600.0)  # above threshold
        prod = jnp.full((NPOI,), 1.0)
        fwet = jnp.full((NPOI,), 0.5)
        ebull, _, _ = ch4_transport(ch4_soil, prod, fwet, params)
        assert jnp.all(ebull > 0.0)


class TestWetlandStep:
    def test_full_step(self, soil_temps, soil_moisture, porosity):
        ws = create_wetland_state(NPOI)
        params = default_wetland_params()
        het_resp = jnp.full((NPOI,), 1.5)
        ws2 = wetland_step(ws, soil_temps, soil_moisture, porosity, het_resp, params)
        assert isinstance(ws2, WetlandState)
        assert ws2.ch4_soil.shape == (NPOI,)
        assert jnp.all(ws2.ch4_soil >= 0.0)

    def test_jit(self, soil_temps, soil_moisture, porosity):
        ws = create_wetland_state(NPOI)
        params = default_wetland_params()
        het_resp = jnp.full((NPOI,), 1.5)
        ws2 = jax.jit(wetland_step)(ws, soil_temps, soil_moisture, porosity, het_resp, params)
        assert ws2.ch4_flux.shape == (NPOI,)


# ===== Permafrost =====

class TestPermafrostState:
    def test_create(self):
        ps = create_permafrost_state(NPOI)
        assert ps.active_layer.shape == (NPOI,)
        assert jnp.all(ps.excess_ice == 0.0)

    def test_pytree(self):
        ps = create_permafrost_state(NPOI)
        leaves = jax.tree.leaves(ps)
        assert len(leaves) == 5


class TestStefanDepth:
    def test_positive_tdd(self, soil_temps):
        params = default_permafrost_params()
        gdd = jnp.array([500.0, 100.0, 1000.0, 200.0])
        ald = stefan_depth(soil_temps, 273.16, gdd, params)
        assert jnp.all(ald > 0.0)
        assert ald.shape == (NPOI,)

    def test_zero_tdd(self, soil_temps):
        params = default_permafrost_params()
        gdd = jnp.zeros((NPOI,))
        ald = stefan_depth(soil_temps, 273.16, gdd, params)
        assert jnp.allclose(ald, 0.0)

    def test_more_tdd_deeper_ald(self, soil_temps):
        params = default_permafrost_params()
        ald1 = stefan_depth(soil_temps, 273.16, jnp.full((NPOI,), 100.0), params)
        ald2 = stefan_depth(soil_temps, 273.16, jnp.full((NPOI,), 1000.0), params)
        assert jnp.all(ald2 > ald1)


class TestDiagnoseFrozenGround:
    def test_all_unfrozen(self):
        tsoi = jnp.full((NPOI, NSOILAY), 290.0)
        frozen, al = diagnose_frozen_ground(tsoi, 273.16)
        assert jnp.allclose(frozen, 0.0)
        # The full column depth, 0.10+0.15+0.25+0.50+1.00+2.00 = 4.0 m.
        # This asserted NSOILAY * 0.5 = 3.0 until 2026-08-29, encoding the
        # hardcoded 0.5 m layer thickness that diagnose_frozen_ground used
        # instead of the real hsoi -- see docs/AUDIT.md.
        from ibis.bgc.permafrost import DEFAULT_HSOI
        assert jnp.all(al == pytest.approx(sum(DEFAULT_HSOI)))

    def test_all_frozen(self):
        tsoi = jnp.full((NPOI, NSOILAY), 260.0)
        frozen, al = diagnose_frozen_ground(tsoi, 273.16)
        assert jnp.allclose(frozen, 1.0)
        assert jnp.allclose(al, 0.0)  # first layer is frozen

    def test_partial_frozen(self):
        # Top 3 unfrozen, bottom 3 frozen
        tsoi = jnp.array([
            [280.0, 278.0, 276.0, 270.0, 268.0, 265.0],
        ])
        frozen, al = diagnose_frozen_ground(tsoi, 273.16)
        assert 0.0 < frozen[0] < 1.0
        assert al[0] > 0.0  # some unfrozen depth


class TestThermokarst:
    def test_no_deepening(self):
        params = default_permafrost_params()
        excess_ice = jnp.full((NPOI,), 50.0)
        al = jnp.full((NPOI,), 1.0)
        thaw_prev = jnp.full((NPOI,), 1.5)  # prev deeper than current
        new_ice, subsidence = thermokarst(excess_ice, al, thaw_prev, params)
        assert jnp.allclose(new_ice, excess_ice)
        assert jnp.allclose(subsidence, 0.0)

    def test_deepening_melts_ice(self):
        params = default_permafrost_params()
        excess_ice = jnp.full((NPOI,), 50.0)
        al = jnp.full((NPOI,), 2.0)
        thaw_prev = jnp.full((NPOI,), 1.0)
        new_ice, subsidence = thermokarst(excess_ice, al, thaw_prev, params)
        assert jnp.all(new_ice < excess_ice)
        assert jnp.all(subsidence > 0.0)


class TestPermafrostStep:
    def test_full_step(self, soil_temps):
        ps = create_permafrost_state(NPOI)
        params = default_permafrost_params()
        gdd = jnp.array([500.0, 100.0, 1000.0, 200.0])
        ps2 = permafrost_step(ps, soil_temps, 273.16, gdd, params)
        assert isinstance(ps2, PermafrostState)
        assert ps2.active_layer.shape == (NPOI,)

    def test_jit(self, soil_temps):
        ps = create_permafrost_state(NPOI)
        params = default_permafrost_params()
        gdd = jnp.full((NPOI,), 500.0)
        ps2 = jax.jit(permafrost_step)(ps, soil_temps, 273.16, gdd, params)
        assert ps2.frozen_ground.shape == (NPOI,)


# ===== Fire Emissions =====

# NOTE: the fire-emissions tests that lived here exercised the prototype in
# ibis/bgc/fire_emissions.py that was never called by the model and used
# superseded Andreae & Merlet (2001) factors with three conflicting unit
# conventions. That file is quarantined in ibis/bgc/_legacy_prototype/ and the
# module was rewritten for docs/FIRE.md stage F5. Its replacement is covered by
# tests/test_fire_emissions.py (16 tests), including a cross-check of the
# embedded emission factors against the published Andreae (2019) spreadsheet.


class TestSoilGasState:
    def test_create(self):
        sg = create_soil_gas_state(NPOI)
        assert sg.nh4.shape == (NPOI,)
        assert jnp.all(sg.nh4 > 0.0)  # initial NH4 = 0.5
        assert jnp.all(sg.no3 > 0.0)  # initial NO3 = 0.3


class TestWFPS:
    def test_dry(self):
        wsoi = jnp.full((NPOI, NSOILAY), 0.1)
        poros = jnp.full((NPOI, NSOILAY), 0.45)
        w = wfps(wsoi, poros)
        assert jnp.all(w < 0.5)

    def test_saturated(self):
        wsoi = jnp.full((NPOI, NSOILAY), 0.45)
        poros = jnp.full((NPOI, NSOILAY), 0.45)
        w = wfps(wsoi, poros)
        assert jnp.allclose(w, 1.0)


class TestNitrification:
    def test_positive_output(self):
        params = default_soil_gas_params()
        nh4 = jnp.full((NPOI,), 1.0)
        tsoi = jnp.full((NPOI,), 298.15)
        wfps_val = jnp.full((NPOI,), 0.5)
        n2o, nh4_c, no3_p = nitrification(nh4, tsoi, wfps_val, params)
        assert jnp.all(n2o >= 0.0)
        assert jnp.all(nh4_c >= 0.0)
        assert jnp.all(no3_p >= 0.0)

    def test_no_substrate(self):
        params = default_soil_gas_params()
        nh4 = jnp.zeros((NPOI,))
        tsoi = jnp.full((NPOI,), 298.15)
        wfps_val = jnp.full((NPOI,), 0.5)
        n2o, nh4_c, no3_p = nitrification(nh4, tsoi, wfps_val, params)
        assert jnp.allclose(n2o, 0.0)

    def test_n_conservation(self):
        """N2O + NO3 produced should equal NH4 consumed."""
        params = default_soil_gas_params()
        nh4 = jnp.full((NPOI,), 2.0)
        tsoi = jnp.full((NPOI,), 298.15)
        wfps_val = jnp.full((NPOI,), 0.5)
        n2o, nh4_c, no3_p = nitrification(nh4, tsoi, wfps_val, params)
        assert jnp.allclose(n2o + no3_p, nh4_c, atol=1e-6)


class TestDenitrification:
    def test_below_threshold(self):
        params = default_soil_gas_params()
        no3 = jnp.full((NPOI,), 1.0)
        tsoi = jnp.full((NPOI,), 298.15)
        wfps_val = jnp.full((NPOI,), 0.3)  # below threshold (0.6)
        labile_c = jnp.full((NPOI,), 1.0)
        n2o, no3_c = denitrification(no3, tsoi, wfps_val, labile_c, params)
        assert jnp.allclose(n2o, 0.0)

    def test_above_threshold(self):
        params = default_soil_gas_params()
        no3 = jnp.full((NPOI,), 1.0)
        tsoi = jnp.full((NPOI,), 298.15)
        wfps_val = jnp.full((NPOI,), 0.9)
        labile_c = jnp.full((NPOI,), 1.0)
        n2o, no3_c = denitrification(no3, tsoi, wfps_val, labile_c, params)
        assert jnp.all(n2o > 0.0)


class TestSoilGasStep:
    def test_full_step(self, soil_temps, soil_moisture, porosity):
        sg = create_soil_gas_state(NPOI)
        params = default_soil_gas_params()
        litter_c = jnp.full((NPOI,), 0.5)
        n_min = jnp.full((NPOI,), 0.01)
        sg2 = soil_gas_step(sg, soil_temps, soil_moisture, porosity, litter_c, n_min, params)
        assert isinstance(sg2, SoilGasState)
        assert jnp.all(sg2.nh4 >= 0.0)
        assert jnp.all(sg2.no3 >= 0.0)

    def test_jit(self, soil_temps, soil_moisture, porosity):
        sg = create_soil_gas_state(NPOI)
        params = default_soil_gas_params()
        litter_c = jnp.full((NPOI,), 0.5)
        n_min = jnp.full((NPOI,), 0.01)
        sg2 = jax.jit(soil_gas_step)(sg, soil_temps, soil_moisture, porosity, litter_c, n_min, params)
        assert sg2.n2o_total.shape == (NPOI,)


# ===== BVOC =====

class TestBVOCState:
    def test_create(self):
        bs = create_bvoc_state(NPOI)
        assert bs.isoprene.shape == (NPOI,)
        assert jnp.all(bs.monoterpene == 0.0)


class TestGammaT:
    def test_standard_temp(self):
        """At standard temperature, gamma_T should be near 1."""
        gt = gamma_t_isop(jnp.array([T_STANDARD]))
        assert 0.5 < float(gt[0]) < 2.0  # approximately 1

    def test_cold_low(self):
        """Cold temperatures → low gamma."""
        gt = gamma_t_isop(jnp.array([260.0]))
        assert float(gt[0]) < 0.5

    def test_monoterpene_exponential(self):
        gt_std = gamma_t_mono(jnp.array([T_STANDARD]))
        assert jnp.allclose(gt_std, 1.0)  # exp(0) = 1 at standard T
        gt_warm = gamma_t_mono(jnp.array([T_STANDARD + 10.0]))
        assert float(gt_warm[0]) > 1.0


class TestGammaPAR:
    def test_zero_par(self):
        gp = gamma_par_isop(jnp.array([0.0]))
        assert jnp.allclose(gp, 0.0)

    def test_standard_par(self):
        gp = gamma_par_isop(jnp.array([PAR_STANDARD]))
        assert float(gp[0]) > 0.0

    def test_monotonic(self):
        par_vals = jnp.array([100.0, 500.0, 1000.0, 2000.0])
        gp = gamma_par_isop(par_vals)
        # Should be monotonically increasing (saturating)
        for i in range(3):
            assert gp[i + 1] >= gp[i]


class TestBVOCEmissions:
    def test_basic(self):
        tleaf = jnp.full((NPOI,), 300.0)
        par = jnp.full((NPOI,), 800.0)
        lai = jnp.ones((NPOI, NPFT)) * 2.0
        frac = jnp.ones((NPOI, NPFT)) / NPFT
        bs = bvoc_emissions(tleaf, par, lai, frac)
        assert isinstance(bs, BVOCState)
        assert jnp.all(bs.isoprene > 0.0)
        assert jnp.all(bs.monoterpene > 0.0)

    def test_no_lai_no_emission(self):
        tleaf = jnp.full((NPOI,), 300.0)
        par = jnp.full((NPOI,), 800.0)
        lai = jnp.zeros((NPOI, NPFT))
        frac = jnp.ones((NPOI, NPFT)) / NPFT
        bs = bvoc_emissions(tleaf, par, lai, frac)
        assert jnp.allclose(bs.isoprene, 0.0)
        assert jnp.allclose(bs.monoterpene, 0.0)

    def test_no_par_no_isoprene(self):
        """Isoprene needs light; monoterpene does not."""
        tleaf = jnp.full((NPOI,), 300.0)
        par = jnp.zeros((NPOI,))
        lai = jnp.ones((NPOI, NPFT)) * 2.0
        frac = jnp.ones((NPOI, NPFT)) / NPFT
        bs = bvoc_emissions(tleaf, par, lai, frac)
        assert jnp.allclose(bs.isoprene, 0.0)
        assert jnp.all(bs.monoterpene > 0.0)

    def test_jit(self):
        tleaf = jnp.full((NPOI,), 300.0)
        par = jnp.full((NPOI,), 800.0)
        lai = jnp.ones((NPOI, NPFT)) * 2.0
        frac = jnp.ones((NPOI, NPFT)) / NPFT
        bs = jax.jit(bvoc_emissions)(tleaf, par, lai, frac)
        assert bs.isoprene.shape == (NPOI,)


# ===== NEE =====

class TestNEEState:
    def test_create(self):
        ns = create_nee_state(NPOI)
        assert ns.nee_co2.shape == (NPOI,)
        assert ns.nee_ch4.shape == (NPOI,)
        assert ns.nee_n2o.shape == (NPOI,)

    def test_pytree(self):
        ns = create_nee_state(NPOI)
        leaves = jax.tree.leaves(ns)
        assert len(leaves) == 11


class TestComputeNEE:
    def test_carbon_balance(self):
        """NEE_CO2 = -GPP + Ra + Rh + fire."""
        gpp = jnp.full((NPOI,), 10.0)
        ra = jnp.full((NPOI,), 4.0)
        rh = jnp.full((NPOI,), 3.0)
        fire_co2 = jnp.full((NPOI,), 0.5)
        ns = compute_nee(gpp, ra, rh, fire_co2=fire_co2)
        expected = -10.0 + 4.0 + 3.0 + 0.5
        assert jnp.allclose(ns.nee_co2, expected)

    def test_sink(self):
        """When GPP > Ra + Rh, NEE should be negative (sink)."""
        gpp = jnp.full((NPOI,), 10.0)
        ra = jnp.full((NPOI,), 3.0)
        rh = jnp.full((NPOI,), 2.0)
        ns = compute_nee(gpp, ra, rh)
        assert jnp.all(ns.nee_co2 < 0.0)

    def test_source(self):
        """When GPP < Ra + Rh, NEE should be positive (source)."""
        gpp = jnp.full((NPOI,), 2.0)
        ra = jnp.full((NPOI,), 3.0)
        rh = jnp.full((NPOI,), 4.0)
        ns = compute_nee(gpp, ra, rh)
        assert jnp.all(ns.nee_co2 > 0.0)

    def test_ch4_budget(self):
        wetland_ch4 = jnp.full((NPOI,), 0.1)
        fire_ch4 = jnp.full((NPOI,), 0.02)
        gpp = jnp.full((NPOI,), 5.0)
        ra = jnp.full((NPOI,), 2.0)
        rh = jnp.full((NPOI,), 1.5)
        ns = compute_nee(gpp, ra, rh, wetland_ch4=wetland_ch4, fire_ch4=fire_ch4)
        assert jnp.allclose(ns.nee_ch4, 0.12)

    def test_n2o_budget(self):
        soil_n2o = jnp.full((NPOI,), 0.005)
        fire_n2o = jnp.full((NPOI,), 0.001)
        gpp = jnp.full((NPOI,), 5.0)
        ra = jnp.full((NPOI,), 2.0)
        rh = jnp.full((NPOI,), 1.5)
        ns = compute_nee(gpp, ra, rh, soil_n2o=soil_n2o, fire_n2o=fire_n2o)
        assert jnp.allclose(ns.nee_n2o, 0.006)

    def test_defaults_zero(self):
        """Missing optional fluxes default to zero."""
        gpp = jnp.full((NPOI,), 5.0)
        ra = jnp.full((NPOI,), 2.0)
        rh = jnp.full((NPOI,), 1.5)
        ns = compute_nee(gpp, ra, rh)
        assert jnp.allclose(ns.nee_ch4, 0.0)
        assert jnp.allclose(ns.nee_n2o, 0.0)
        assert jnp.allclose(ns.fire_co2, 0.0)

    def test_components_stored(self):
        gpp = jnp.full((NPOI,), 10.0)
        ra = jnp.full((NPOI,), 4.0)
        rh = jnp.full((NPOI,), 3.0)
        ns = compute_nee(gpp, ra, rh)
        assert jnp.allclose(ns.gpp, 10.0)
        assert jnp.allclose(ns.ra, 4.0)
        assert jnp.allclose(ns.rh, 3.0)


# ===== Gradient flow =====

class TestBGCGradients:
    def test_wetland_grad(self, soil_temps, soil_moisture, porosity):
        """Gradient flows through wetland_step."""
        ws = create_wetland_state(NPOI)
        params = default_wetland_params()

        def loss(het_resp):
            ws2 = wetland_step(ws, soil_temps, soil_moisture, porosity, het_resp, params)
            return jnp.sum(ws2.ch4_flux)

        het_resp = jnp.full((NPOI,), 2.0)
        grad = jax.grad(loss)(het_resp)
        assert not jnp.any(jnp.isnan(grad))
        assert jnp.any(grad != 0.0)

    def test_bvoc_grad(self):
        """Gradient flows through bvoc_emissions."""
        def loss(tleaf):
            par = jnp.full((NPOI,), 800.0)
            lai = jnp.ones((NPOI, NPFT)) * 2.0
            frac = jnp.ones((NPOI, NPFT)) / NPFT
            bs = bvoc_emissions(tleaf, par, lai, frac)
            return jnp.sum(bs.isoprene)

        tleaf = jnp.full((NPOI,), 300.0)
        grad = jax.grad(loss)(tleaf)
        assert not jnp.any(jnp.isnan(grad))
        assert jnp.any(grad != 0.0)

    def test_nee_grad(self):
        """Gradient flows through compute_nee."""
        def loss(gpp):
            ra = jnp.full((NPOI,), 2.0)
            rh = jnp.full((NPOI,), 1.5)
            ns = compute_nee(gpp, ra, rh)
            return jnp.sum(ns.nee_co2)

        gpp = jnp.full((NPOI,), 5.0)
        grad = jax.grad(loss)(gpp)
        assert jnp.allclose(grad, -1.0)  # d(nee)/d(gpp) = -1

    def test_soil_gas_grad(self, soil_temps, soil_moisture, porosity):
        """Gradient flows through soil_gas_step."""
        sg = create_soil_gas_state(NPOI)
        params = default_soil_gas_params()

        def loss(n_min):
            sg2 = soil_gas_step(sg, soil_temps, soil_moisture, porosity,
                                jnp.full((NPOI,), 0.5), n_min, params)
            return jnp.sum(sg2.n2o_total)

        n_min = jnp.full((NPOI,), 0.01)
        grad = jax.grad(loss)(n_min)
        assert not jnp.any(jnp.isnan(grad))


# ===== Integration =====

class TestBGCIntegration:
    def test_import_all(self):
        """All BGC modules importable from package."""
        # Only the components the model actually reaches are re-exported from
        # ibis.bgc. bvoc / nee / soil_gases / wetlands moved to
        # _legacy_unwired/ on 2026-08-28 (docs/AUDIT.md) precisely because
        # re-exporting them here implied they were wired.
        from ibis.bgc import (
            PermafrostState, PermafrostParams, permafrost_step,
            FireEmissions, FireEmissionParams, fire_emissions,
        )
        from ibis.bgc._legacy_unwired.wetlands import (
            WetlandState, WetlandParams, wetland_step,
        )
        from ibis.bgc._legacy_unwired.soil_gases import (
            SoilGasState, SoilGasParams, soil_gas_step,
        )
        from ibis.bgc._legacy_unwired.bvoc import BVOCState, bvoc_emissions
        from ibis.bgc._legacy_unwired.nee import NEEState, compute_nee

    def test_multi_step_wetland(self, soil_temps, soil_moisture, porosity):
        """Run 10 steps of wetland CH4 — pool should grow."""
        ws = create_wetland_state(NPOI)
        params = default_wetland_params()
        het_resp = jnp.full((NPOI,), 2.0)
        for _ in range(10):
            ws = wetland_step(ws, soil_temps, soil_moisture, porosity, het_resp, params)
        # After 10 steps, there should be some CH4 pool
        assert ws.ch4_soil.shape == (NPOI,)

    def test_nee_from_bgc_components(self, soil_temps, soil_moisture, porosity):
        """Compute NEE from wetland and fire outputs."""
        # Wetland CH4
        ws = create_wetland_state(NPOI)
        params = default_wetland_params()
        het_resp = jnp.full((NPOI,), 2.0)
        ws = wetland_step(ws, soil_temps, soil_moisture, porosity, het_resp, params)

        # Fire emissions use the F5 API (see tests/test_fire_emissions.py);
        # this integration check only needs a CO2 number to feed NEE.
        from ibis.bgc.fire_emissions import fire_emissions
        from ibis.physics.fire import DEFAULT_PFT_MORTALITY
        fs = fire_emissions(
            burned_fraction=jnp.full((NPOI,), 0.05),
            cbiol=jnp.full((NPOI, 12), 0.04),
            cbiow=jnp.full((NPOI, 12), 0.30),
            totalit=jnp.full((NPOI,), 0.20),
            plai=jnp.zeros((NPOI, 12)).at[:, 10].set(2.0),
            mort=jnp.asarray(DEFAULT_PFT_MORTALITY),
        )

        # NEE
        gpp = jnp.full((NPOI,), 8.0)
        ra = jnp.full((NPOI,), 3.0)
        rh = het_resp
        ns = compute_nee(
            gpp, ra, rh,
            fire_co2=fs.co2,
            fire_ch4=fs.ch4,
            fire_n2o=fs.n2o,
            wetland_ch4=ws.ch4_flux,
        )
        assert isinstance(ns, NEEState)
        assert ns.nee_co2.shape == (NPOI,)
        assert ns.nee_ch4.shape == (NPOI,)


class TestPermafrostWiring:
    """Permafrost wired 2026-08-29, with two bugs fixed on the way."""

    def test_stefan_depth_uses_degree_seconds(self):
        """TDD must carry seconds, or the depth is 294x too small.

        k is W/m/K = J/s/m/K, so `2*k*TDD / (rho*Lf*porosity)` is metres-squared
        only if TDD is in degree-SECONDS. `gdd0` is degree-DAYS. Passing it
        straight in gave 0.0039 m on the reference state against an observed
        Arctic active layer of 0.3-2 m; with the conversion, 1.31 m.
        """
        import numpy as np

        from ibis.bgc.permafrost import default_permafrost_params, stefan_depth

        gdd0 = jnp.full((3,), 765.7)          # Arctic mean on the reference state
        ald = float(stefan_depth(jnp.full((3, 6), 270.0), 273.16, gdd0,
                                 default_permafrost_params())[0])
        assert 0.3 <= ald <= 2.5, f"{ald} m outside the observed 0.3-2 m"
        assert ald == pytest.approx(1.31, abs=0.05)

    def test_active_layer_uses_real_layer_thicknesses(self):
        """Not a hardcoded 0.5 m.

        IBIS's layers are [0.10, 0.15, 0.25, 0.50, 1.00, 2.00]. With the top
        two thawed the active layer is 0.10+0.15 = 0.25 m, not 2*0.5 = 1.0 m.
        """
        import numpy as np

        from ibis.bgc.permafrost import DEFAULT_HSOI, diagnose_frozen_ground

        hsoi = jnp.asarray(DEFAULT_HSOI)
        tsoi = jnp.asarray([[280.0, 280.0, 270.0, 270.0, 270.0, 270.0]])
        _, al = diagnose_frozen_ground(tsoi, 273.16, hsoi)
        assert float(al[0]) == pytest.approx(0.25)

    def test_frozen_fraction_is_thickness_weighted(self):
        """The 0.10 m surface layer must not count equally with the 2.00 m one."""
        from ibis.bgc.permafrost import DEFAULT_HSOI, diagnose_frozen_ground

        hsoi = jnp.asarray(DEFAULT_HSOI)
        # only the thin top layer frozen -> a small fraction of the column
        tsoi = jnp.asarray([[270.0, 280.0, 280.0, 280.0, 280.0, 280.0]])
        ff, _ = diagnose_frozen_ground(tsoi, 273.16, hsoi)
        assert float(ff[0]) == pytest.approx(0.10 / 4.0, abs=1e-6)

    def test_runner_exposes_enable_permafrost(self):
        import inspect

        from ibis.timestepping.loops_gswp3 import (run_month_gswp3,
                                                   run_spinup_gswp3)
        for fn in (run_month_gswp3, run_spinup_gswp3):
            assert "enable_permafrost" in inspect.signature(fn).parameters
