"""Tests for snow physics module."""

import jax
import jax.numpy as jnp
import pytest

from ibis.physics.snow_physics import vadapt, snowheat, snow_melt, snow_coverage
from ibis.constants import PhysicalConstants


@pytest.fixture
def consts():
    return PhysicalConstants()


@pytest.fixture
def npoi():
    return 4


@pytest.fixture
def nsnolay():
    return 3


# ============================================================================
# vadapt tests
# ============================================================================

class TestVadapt:
    """Snow layer thickness re-adaptation."""

    def test_conserves_total_thickness(self, npoi, nsnolay):
        """Total thickness should be preserved."""
        hcur = jnp.array([[0.1, 0.2, 0.3]] * npoi)
        tcur = jnp.full((npoi, nsnolay), 265.0)
        has_snow = jnp.ones(npoi)
        htop = 0.05

        hnew, _ = vadapt(hcur, tcur, htop, has_snow, 1e-7)
        total_old = jnp.sum(hcur, axis=1)
        total_new = jnp.sum(hnew, axis=1)
        assert jnp.allclose(total_old, total_new, atol=1e-5)

    def test_top_layer_thickness(self, npoi, nsnolay):
        """Top layer should have prescribed thickness (if total > nlay*htop)."""
        hcur = jnp.array([[0.05, 0.3, 0.3]] * npoi)
        tcur = jnp.full((npoi, nsnolay), 265.0)
        has_snow = jnp.ones(npoi)
        htop = 0.05

        hnew, _ = vadapt(hcur, tcur, htop, has_snow, 1e-7)
        # Total = 0.65 > 3*0.05=0.15, so top should be htop
        assert jnp.allclose(hnew[:, 0], htop, atol=1e-5)

    def test_conserves_heat(self, npoi, nsnolay):
        """Temperature * thickness integral should be approximately conserved."""
        hcur = jnp.array([[0.1, 0.2, 0.3]] * npoi)
        tcur = jnp.array([[270.0, 265.0, 260.0]] * npoi)
        has_snow = jnp.ones(npoi)
        htop = 0.05

        hnew, tnew = vadapt(hcur, tcur, htop, has_snow, 1e-7)
        heat_old = jnp.sum(hcur * tcur, axis=1)
        heat_new = jnp.sum(hnew * tnew, axis=1)
        assert jnp.allclose(heat_old, heat_new, rtol=0.01)

    def test_no_snow_unchanged(self, npoi, nsnolay):
        """Points without snow should not change."""
        hcur = jnp.array([[0.1, 0.2, 0.3]] * npoi)
        tcur = jnp.full((npoi, nsnolay), 265.0)
        has_snow = jnp.zeros(npoi)  # no snow

        hnew, tnew = vadapt(hcur, tcur, 0.05, has_snow, 1e-7)
        assert jnp.allclose(hnew, hcur)
        assert jnp.allclose(tnew, tcur)


# ============================================================================
# snowheat tests
# ============================================================================

class TestSnowheat:
    """Implicit snow heat conduction."""

    def test_uniform_temp_stable(self, npoi, nsnolay, consts):
        """Uniform temperature with no flux should stay constant."""
        tsno = jnp.full((npoi, nsnolay), 265.0)
        tlsub = jnp.full(npoi, 265.0)
        fhtop = jnp.zeros(npoi)
        xl = jnp.full(npoi, 1.0)
        chl = 8436.0
        fi = jnp.full(npoi, 0.5)
        hsno = jnp.array([[0.05, 0.1, 0.1]] * npoi)
        tsoi_1 = jnp.full(npoi, 265.0)
        consoi_1 = jnp.full(npoi, 1.5)

        result = snowheat(tsno, tlsub, fhtop, xl, chl, fi, hsno,
                          250.0, 0.20, 0.1, consoi_1, tsoi_1,
                          1800.0, consts.cice)

        assert jnp.allclose(result.tsno, 265.0, atol=0.1)
        assert jnp.allclose(result.tlsub, 265.0, atol=0.1)

    def test_top_heating(self, npoi, nsnolay, consts):
        """Surface heat flux should warm the top snow layer."""
        tsno = jnp.full((npoi, nsnolay), 260.0)
        tlsub = jnp.full(npoi, 260.0)
        fhtop = jnp.full(npoi, 50.0)  # 50 W/m2
        xl = jnp.full(npoi, 1.0)
        chl = 8436.0
        fi = jnp.full(npoi, 0.5)
        hsno = jnp.array([[0.05, 0.1, 0.1]] * npoi)
        tsoi_1 = jnp.full(npoi, 275.0)
        consoi_1 = jnp.full(npoi, 1.5)

        result = snowheat(tsno, tlsub, fhtop, xl, chl, fi, hsno,
                          250.0, 0.20, 0.1, consoi_1, tsoi_1,
                          1800.0, consts.cice)

        assert jnp.all(result.tsno[:, 0] > 260.0)
        assert jnp.all(jnp.isfinite(result.tsno))

    def test_outputs_finite(self, npoi, nsnolay, consts):
        """All outputs should be finite."""
        tsno = jnp.full((npoi, nsnolay), 265.0)
        tlsub = jnp.full(npoi, 268.0)
        fhtop = jnp.full(npoi, -20.0)
        xl = jnp.full(npoi, 2.0)
        chl = 8436.0
        fi = jnp.full(npoi, 0.3)
        hsno = jnp.array([[0.05, 0.08, 0.12]] * npoi)
        tsoi_1 = jnp.full(npoi, 275.0)
        consoi_1 = jnp.full(npoi, 1.5)

        result = snowheat(tsno, tlsub, fhtop, xl, chl, fi, hsno,
                          250.0, 0.20, 0.1, consoi_1, tsoi_1,
                          1800.0, consts.cice)

        assert jnp.all(jnp.isfinite(result.tsno))
        assert jnp.all(jnp.isfinite(result.tlsub))
        assert jnp.all(jnp.isfinite(result.sflo))


# ============================================================================
# snow_melt tests
# ============================================================================

class TestSnowMelt:
    """Snow melting."""

    def test_no_melt_below_tmelt(self, npoi, nsnolay, consts):
        """Below melting point, no melting should occur."""
        hsno = jnp.array([[0.05, 0.1, 0.1]] * npoi)
        tsno = jnp.full((npoi, nsnolay), 265.0)
        fi = jnp.full(npoi, 0.5)

        hsno_new, tsno_new, zmelt, zheat = snow_melt(
            hsno, tsno, fi, 250.0, consts.cice, consts.hfus, consts.tmelt)

        assert jnp.allclose(hsno_new, hsno)
        assert jnp.allclose(zmelt, 0.0)

    def test_melt_above_tmelt(self, npoi, nsnolay, consts):
        """Above melting point, snow should melt."""
        hsno = jnp.array([[0.05, 0.1, 0.1]] * npoi)
        tsno = jnp.full((npoi, nsnolay), 280.0)  # well above freezing
        fi = jnp.full(npoi, 0.5)

        hsno_new, tsno_new, zmelt, zheat = snow_melt(
            hsno, tsno, fi, 250.0, consts.cice, consts.hfus, consts.tmelt)

        assert jnp.all(hsno_new <= hsno)
        assert jnp.all(zmelt > 0.0)
        assert jnp.all(tsno_new <= consts.tmelt)

    def test_temperature_capped(self, npoi, nsnolay, consts):
        """Post-melt temperature should not exceed tmelt."""
        hsno = jnp.array([[0.05, 0.1, 0.1]] * npoi)
        tsno = jnp.full((npoi, nsnolay), 300.0)  # extremely warm
        fi = jnp.full(npoi, 1.0)

        _, tsno_new, _, _ = snow_melt(
            hsno, tsno, fi, 250.0, consts.cice, consts.hfus, consts.tmelt)

        assert jnp.all(tsno_new <= consts.tmelt)


# ============================================================================
# snow_coverage tests
# ============================================================================

class TestSnowCoverage:
    """Fractional snow cover adjustment."""

    def test_bounded(self, npoi, nsnolay, consts):
        """fi should stay in [0, fimax]."""
        hsno = jnp.array([[0.05, 0.1, 0.1]] * npoi)
        fi = jnp.full(npoi, 0.5)
        hsnoruf = jnp.full(npoi, 0.5)

        fi_new, _ = snow_coverage(hsno, fi, 0.01, hsnoruf, 1.0, consts.epsilon)

        assert jnp.all(fi_new >= 0.0)
        assert jnp.all(fi_new <= 1.0)

    def test_total_volume_conserved(self, npoi, nsnolay, consts):
        """fi * total_thickness should be approximately conserved."""
        hsno = jnp.array([[0.05, 0.1, 0.15]] * npoi)
        fi = jnp.full(npoi, 0.4)
        hsnoruf = jnp.full(npoi, 0.5)

        vol_old = fi * jnp.sum(hsno, axis=1)
        fi_new, hsno_new = snow_coverage(hsno, fi, 0.01, hsnoruf, 1.0, consts.epsilon)
        vol_new = fi_new * jnp.sum(hsno_new, axis=1)

        assert jnp.allclose(vol_old, vol_new, rtol=0.01)
