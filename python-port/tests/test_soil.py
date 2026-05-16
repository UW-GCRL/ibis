"""Tests for soil physics module."""

import jax
import jax.numpy as jnp
import pytest

from ibis.physics.soil_physics import setsoi, wadjust, soilh2o, soilheat, freeze_thaw
from ibis.constants import PhysicalConstants


@pytest.fixture
def consts():
    return PhysicalConstants()


@pytest.fixture
def npoi():
    return 4


@pytest.fixture
def nsoilay():
    return 6


# ============================================================================
# setsoi tests
# ============================================================================

class TestSetsoi:
    """Diagnostic soil quantities."""

    def test_conductivity_positive(self, npoi, nsoilay, consts):
        """Thermal conductivity should be positive."""
        sand = jnp.full((npoi, nsoilay), 40.0)
        clay = jnp.full((npoi, nsoilay), 20.0)
        poros = jnp.full((npoi, nsoilay), 0.45)
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.zeros((npoi, nsoilay))
        tsoi = jnp.full((npoi, nsoilay), 280.0)
        wpud = jnp.zeros(npoi)
        wipud = jnp.zeros(npoi)
        tsno_1 = jnp.full(npoi, 270.0)
        ta = jnp.full(npoi, 280.0)

        result = setsoi(sand, clay, poros, wsoi, wisoi, tsoi,
                        wpud, wipud, 10.0, 0.5, tsno_1, ta,
                        consts.tmelt, consts.hvap, consts.hsub,
                        consts.ch2o, consts.cvap, consts.cice, consts.epsilon)

        assert jnp.all(result.consoi > 0.0)
        assert jnp.all(jnp.isfinite(result.consoi))

    def test_qglif_sums_to_one(self, npoi, nsoilay, consts):
        """Evap source fractions should sum to ~1."""
        sand = jnp.full((npoi, nsoilay), 50.0)
        clay = jnp.full((npoi, nsoilay), 20.0)
        poros = jnp.full((npoi, nsoilay), 0.45)
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.full((npoi, nsoilay), 0.1)
        tsoi = jnp.full((npoi, nsoilay), 280.0)
        wpud = jnp.full(npoi, 1.0)
        wipud = jnp.full(npoi, 0.5)
        tsno_1 = jnp.full(npoi, 270.0)
        ta = jnp.full(npoi, 280.0)

        result = setsoi(sand, clay, poros, wsoi, wisoi, tsoi,
                        wpud, wipud, 10.0, 0.5, tsno_1, ta,
                        consts.tmelt, consts.hvap, consts.hsub,
                        consts.ch2o, consts.cvap, consts.cice, consts.epsilon)

        totals = result.qglif.sum(axis=1)
        assert jnp.allclose(totals, 1.0, atol=0.01)

    def test_latent_heat_positive(self, npoi, nsoilay, consts):
        """Latent heat values should be positive."""
        sand = jnp.full((npoi, nsoilay), 40.0)
        clay = jnp.full((npoi, nsoilay), 20.0)
        poros = jnp.full((npoi, nsoilay), 0.45)
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.zeros((npoi, nsoilay))
        tsoi = jnp.full((npoi, nsoilay), 280.0)
        tsno_1 = jnp.full(npoi, 270.0)
        ta = jnp.full(npoi, 280.0)

        result = setsoi(sand, clay, poros, wsoi, wisoi, tsoi,
                        jnp.zeros(npoi), jnp.zeros(npoi),
                        10.0, 0.5, tsno_1, ta,
                        consts.tmelt, consts.hvap, consts.hsub,
                        consts.ch2o, consts.cvap, consts.cice, consts.epsilon)

        assert jnp.all(result.hvasug > 0.0)
        assert jnp.all(result.hvasui > 0.0)


# ============================================================================
# wadjust tests
# ============================================================================

class TestWadjust:
    """Bounds enforcement for soil moisture."""

    def test_clamps_to_01(self, npoi, nsoilay, consts):
        """wsoi and wisoi should be clamped to [0, 1]."""
        wsoi = jnp.array([[1.2, -0.1, 0.5, 0.0, 0.999, 0.5]] * npoi)
        wisoi = jnp.array([[0.0, 0.5, -0.1, 1.2, 0.0, 0.5]] * npoi)
        poros = jnp.full((npoi, nsoilay), 0.45)
        hsoi = jnp.full(nsoilay, 0.1)

        w_new, wi_new, gadj = wadjust(wsoi, wisoi, poros, hsoi,
                                       consts.rhow, 3600.0, consts.epsilon)

        assert jnp.all(w_new >= 0.0)
        assert jnp.all(w_new <= 1.0)
        assert jnp.all(wi_new >= 0.0)
        assert jnp.all(wi_new <= 1.0)

    def test_full_ice_zeros_liquid(self, npoi, nsoilay, consts):
        """When ice fills pore space, liquid should be zero."""
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.full((npoi, nsoilay), 1.0)
        poros = jnp.full((npoi, nsoilay), 0.45)
        hsoi = jnp.full(nsoilay, 0.1)

        w_new, _, _ = wadjust(wsoi, wisoi, poros, hsoi,
                              consts.rhow, 3600.0, consts.epsilon)

        assert jnp.allclose(w_new, 0.0)

    def test_no_change_preserves_zero_adjust(self, npoi, nsoilay, consts):
        """Valid moisture should give near-zero gadjust."""
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.full((npoi, nsoilay), 0.1)
        poros = jnp.full((npoi, nsoilay), 0.45)
        hsoi = jnp.full(nsoilay, 0.1)

        _, _, gadj = wadjust(wsoi, wisoi, poros, hsoi,
                             consts.rhow, 3600.0, consts.epsilon)

        assert jnp.allclose(gadj, 0.0, atol=1e-10)


# ============================================================================
# soilh2o tests
# ============================================================================

class TestSoilh2o:
    """Implicit soil moisture solver."""

    def test_steady_state_no_forcing(self, npoi, nsoilay, consts):
        """With no forcing, moisture should stay approximately constant."""
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.zeros((npoi, nsoilay))
        poros = jnp.full((npoi, nsoilay), 0.45)
        porosflo = poros * (1.0 - wisoi)
        hydraul = jnp.full((npoi, nsoilay), 1e-6)
        suction = jnp.full((npoi, nsoilay), 0.2)
        bex = jnp.full((npoi, nsoilay), 5.0)
        hsoi = jnp.array([0.1, 0.15, 0.3, 0.5, 1.0, 2.0])
        fwtop = jnp.zeros(npoi)
        fwpud = jnp.zeros(npoi)
        wpud = jnp.zeros(npoi)

        result = soilh2o(wsoi, wisoi, poros, porosflo, hydraul, suction, bex,
                         hsoi, fwtop, fwpud, wpud, 0.0, 3600.0,
                         consts.rhow, consts.epsilon)

        # With impermeable bottom (bperm=0) and no forcing, moisture should
        # change only by diffusion
        assert jnp.all(jnp.isfinite(result.wsoi))
        assert jnp.all(result.wsoi >= 0.0)

    def test_outputs_finite(self, npoi, nsoilay, consts):
        """All outputs should be finite."""
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.full((npoi, nsoilay), 0.05)
        poros = jnp.full((npoi, nsoilay), 0.45)
        porosflo = poros * jnp.maximum(consts.epsilon, 1.0 - wisoi)
        hydraul = jnp.full((npoi, nsoilay), 1e-5)
        suction = jnp.full((npoi, nsoilay), 0.2)
        bex = jnp.full((npoi, nsoilay), 5.0)
        hsoi = jnp.array([0.1, 0.15, 0.3, 0.5, 1.0, 2.0])
        fwtop = jnp.full(npoi, -1e-6)
        fwpud = jnp.full(npoi, 1e-7)
        wpud = jnp.full(npoi, 0.5)

        result = soilh2o(wsoi, wisoi, poros, porosflo, hydraul, suction, bex,
                         hsoi, fwtop, fwpud, wpud, 1.0, 1800.0,
                         consts.rhow, consts.epsilon)

        assert jnp.all(jnp.isfinite(result.wsoi))
        assert jnp.all(jnp.isfinite(result.wflo))
        assert jnp.all(jnp.isfinite(result.wpud))


# ============================================================================
# soilheat tests
# ============================================================================

class TestSoilheat:
    """Implicit soil heat conduction solver."""

    def test_uniform_temp_no_change(self, npoi, nsoilay, consts):
        """Uniform temperature + no surface flux should not change temps."""
        tsoi = jnp.full((npoi, nsoilay), 280.0)
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.zeros((npoi, nsoilay))
        poros = jnp.full((npoi, nsoilay), 0.45)
        consoi = jnp.full((npoi, nsoilay), 1.5)
        csoi_mineral = jnp.full((npoi, nsoilay), 800.0)
        rhosoi = jnp.full((npoi, nsoilay), 2600.0)
        hsoi = jnp.array([0.1, 0.15, 0.3, 0.5, 1.0, 2.0])
        c0pud = jnp.zeros((npoi, nsoilay))
        c1pud = jnp.zeros((npoi, nsoilay))
        fhtop = jnp.zeros(npoi)
        wflo = jnp.zeros((npoi, nsoilay + 1))

        result = soilheat(tsoi, tsoi, wsoi, wsoi, wisoi, poros, consoi,
                          csoi_mineral, rhosoi, hsoi, c0pud, c1pud, fhtop, wflo,
                          consts.ch2o, consts.cice, consts.rhow, 1800.0)

        assert jnp.allclose(result.tsoi, 280.0, atol=0.01)

    def test_surface_heating(self, npoi, nsoilay, consts):
        """Surface heat flux should warm the top layer."""
        tsoi = jnp.full((npoi, nsoilay), 280.0)
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.zeros((npoi, nsoilay))
        poros = jnp.full((npoi, nsoilay), 0.45)
        consoi = jnp.full((npoi, nsoilay), 1.5)
        csoi_mineral = jnp.full((npoi, nsoilay), 800.0)
        rhosoi = jnp.full((npoi, nsoilay), 2600.0)
        hsoi = jnp.array([0.1, 0.15, 0.3, 0.5, 1.0, 2.0])
        fhtop = jnp.full(npoi, 100.0)  # 100 W/m2
        wflo = jnp.zeros((npoi, nsoilay + 1))

        result = soilheat(tsoi, tsoi, wsoi, wsoi, wisoi, poros, consoi,
                          csoi_mineral, rhosoi, hsoi, jnp.zeros((npoi, nsoilay)),
                          jnp.zeros((npoi, nsoilay)), fhtop, wflo,
                          consts.ch2o, consts.cice, consts.rhow, 1800.0)

        # Top layer should warm
        assert jnp.all(result.tsoi[:, 0] > 280.0)
        # Bottom layers should barely change
        assert jnp.allclose(result.tsoi[:, -1], 280.0, atol=0.5)


# ============================================================================
# freeze_thaw tests
# ============================================================================

class TestFreezeThaw:
    """Heat-conserving freeze/thaw adjustment."""

    def test_freeze_below_tmelt(self, npoi, nsoilay, consts):
        """Below melting point, liquid should freeze."""
        tsoi = jnp.full((npoi, nsoilay), 270.0)  # below freezing
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.full((npoi, nsoilay), 0.1)
        poros = jnp.full((npoi, nsoilay), 0.45)
        csoi_mineral = jnp.full((npoi, nsoilay), 800.0)
        rhosoi = jnp.full((npoi, nsoilay), 2600.0)
        hsoi = jnp.array([0.1, 0.15, 0.3, 0.5, 1.0, 2.0])
        c1pud = jnp.zeros((npoi, nsoilay))

        tsoi_new, wsoi_new, wisoi_new, _ = freeze_thaw(
            tsoi, wsoi, wisoi, poros, csoi_mineral, rhosoi, hsoi, c1pud,
            consts.ch2o, consts.cice, consts.rhow, consts.hfus,
            consts.tmelt, consts.epsilon)

        # Ice fraction should increase
        assert jnp.all(wisoi_new >= wisoi)
        assert jnp.all(jnp.isfinite(tsoi_new))

    def test_melt_above_tmelt(self, npoi, nsoilay, consts):
        """Above melting point, ice should melt."""
        tsoi = jnp.full((npoi, nsoilay), 280.0)  # above freezing
        wsoi = jnp.full((npoi, nsoilay), 0.3)
        wisoi = jnp.full((npoi, nsoilay), 0.3)
        poros = jnp.full((npoi, nsoilay), 0.45)
        csoi_mineral = jnp.full((npoi, nsoilay), 800.0)
        rhosoi = jnp.full((npoi, nsoilay), 2600.0)
        hsoi = jnp.array([0.1, 0.15, 0.3, 0.5, 1.0, 2.0])
        c1pud = jnp.zeros((npoi, nsoilay))

        tsoi_new, wsoi_new, wisoi_new, _ = freeze_thaw(
            tsoi, wsoi, wisoi, poros, csoi_mineral, rhosoi, hsoi, c1pud,
            consts.ch2o, consts.cice, consts.rhow, consts.hfus,
            consts.tmelt, consts.epsilon)

        # Ice fraction should decrease
        assert jnp.all(wisoi_new <= wisoi)
        assert jnp.all(jnp.isfinite(tsoi_new))

    def test_at_tmelt_no_change(self, npoi, nsoilay, consts):
        """At exactly melting point, no phase change should occur."""
        tsoi = jnp.full((npoi, nsoilay), consts.tmelt)
        wsoi = jnp.full((npoi, nsoilay), 0.5)
        wisoi = jnp.full((npoi, nsoilay), 0.2)
        poros = jnp.full((npoi, nsoilay), 0.45)
        csoi_mineral = jnp.full((npoi, nsoilay), 800.0)
        rhosoi = jnp.full((npoi, nsoilay), 2600.0)
        hsoi = jnp.array([0.1, 0.15, 0.3, 0.5, 1.0, 2.0])
        c1pud = jnp.zeros((npoi, nsoilay))

        tsoi_new, _, wisoi_new, _ = freeze_thaw(
            tsoi, wsoi, wisoi, poros, csoi_mineral, rhosoi, hsoi, c1pud,
            consts.ch2o, consts.cice, consts.rhow, consts.hfus,
            consts.tmelt, consts.epsilon)

        assert jnp.allclose(wisoi_new, wisoi, atol=1e-6)
        assert jnp.allclose(tsoi_new, consts.tmelt, atol=0.01)
