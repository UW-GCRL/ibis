"""Tests for radiation module -- physical constraints and JIT/grad compatibility."""

import jax
import jax.numpy as jnp
import pytest

from ibis.physics.radiation import (
    solsur, twoset, twostr, solalb, solarf, irrad,
    compute_radiation, default_veg_rad_params,
    VegRadParams, TwostrResult,
)
from ibis.constants import PhysicalConstants


@pytest.fixture
def consts():
    return PhysicalConstants()


@pytest.fixture
def veg_params():
    return default_veg_rad_params()


@pytest.fixture
def npoi():
    return 8


@pytest.fixture
def sunlit_coszen(npoi):
    """coszen with mix of sunlit and dark points."""
    return jnp.array([0.5, 0.8, 0.0, 0.3, 0.0, 0.6, 0.1, 0.9])


# ============================================================================
# solsur tests
# ============================================================================

class TestSolsur:
    """Surface albedo tests."""

    def test_soil_albedo_bounded(self, sunlit_coszen, npoi, consts):
        wsoi_1 = jnp.full(npoi, 0.5)
        wisoi_1 = jnp.zeros(npoi)
        tsno_1 = jnp.full(npoi, 265.0)
        albsav = jnp.full(npoi, 0.1)
        albsan = jnp.full(npoi, 0.2)
        sunlit = sunlit_coszen > 0

        for ib in range(2):
            albsod, albsoi, albsnd, albsni = solsur(
                sunlit_coszen, wsoi_1, wisoi_1, tsno_1,
                albsav, albsan, consts.tmelt, sunlit, ib,
            )
            # Albedos should be in [0, 1] where sunlit
            sun_mask = sunlit_coszen > 0
            assert jnp.all(albsod[sun_mask] >= 0.0)
            assert jnp.all(albsod[sun_mask] <= 0.80)
            assert jnp.all(albsnd[sun_mask] >= 0.0)
            assert jnp.all(albsnd[sun_mask] <= 1.0)

    def test_snow_albedo_vis_gt_nir(self, npoi, consts):
        """Snow VIS albedo should be higher than NIR."""
        coszen = jnp.full(npoi, 0.5)
        sunlit = coszen > 0
        wsoi_1 = jnp.full(npoi, 0.3)
        wisoi_1 = jnp.zeros(npoi)
        tsno_1 = jnp.full(npoi, 260.0)
        albsav = jnp.full(npoi, 0.1)
        albsan = jnp.full(npoi, 0.2)

        _, _, snd_vis, _ = solsur(coszen, wsoi_1, wisoi_1, tsno_1,
                                   albsav, albsan, consts.tmelt, sunlit, 0)
        _, _, snd_nir, _ = solsur(coszen, wsoi_1, wisoi_1, tsno_1,
                                   albsav, albsan, consts.tmelt, sunlit, 1)
        assert jnp.all(snd_vis >= snd_nir)

    def test_dark_points_zero(self, npoi, consts):
        """Non-sunlit points should have zero albedo."""
        coszen = jnp.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        sunlit = coszen > 0
        wsoi_1 = jnp.full(npoi, 0.3)
        wisoi_1 = jnp.zeros(npoi)
        tsno_1 = jnp.full(npoi, 265.0)
        albsav = jnp.full(npoi, 0.1)
        albsan = jnp.full(npoi, 0.2)

        albsod, albsoi, albsnd, albsni = solsur(
            coszen, wsoi_1, wisoi_1, tsno_1,
            albsav, albsan, consts.tmelt, sunlit, 0,
        )
        dark = ~sunlit
        assert jnp.all(albsod[dark] == 0.0)
        assert jnp.all(albsnd[dark] == 0.0)


# ============================================================================
# twostr tests
# ============================================================================

class TestTwostr:
    """Two-stream solver tests."""

    def test_energy_conservation(self, npoi, consts, veg_params):
        """Absorbed + reflected + transmitted <= 1 (energy conservation)."""
        coszen = jnp.full(npoi, 0.5)
        sunlit = coszen > 0
        lai_iv = jnp.full(npoi, 3.0)
        sai_iv = jnp.full(npoi, 0.5)
        asurd_ib = jnp.full(npoi, 0.15)
        asuri_ib = jnp.full(npoi, 0.15)

        omega, betad, betai, avmu, gdir = twoset(
            coszen, lai_iv, sai_iv,
            jnp.zeros(npoi), jnp.zeros(npoi), jnp.full(npoi, 280.0),
            0.10, 0.07, 0.5, 0.0, 0.9, consts.tmelt, 1, sunlit,
        )
        result = twostr(
            coszen, lai_iv, sai_iv, asurd_ib, asuri_ib,
            omega, betad, betai, avmu, gdir,
            consts.epsilon, sunlit, 1,
        )
        # Direct: absorbed + reflected + transmitted_to_surface <= 1
        total_d = result.abvegd + result.refld + result.fbeldd + result.fbelid
        assert jnp.all(total_d <= 1.0 + 0.01)  # small tolerance
        assert jnp.all(total_d >= 0.0)

    def test_no_veg_transparent(self, npoi, consts):
        """With zero LAI+SAI, canopy should be transparent."""
        coszen = jnp.full(npoi, 0.6)
        sunlit = coszen > 0
        lai_iv = jnp.zeros(npoi)
        sai_iv = jnp.zeros(npoi)
        asurd_ib = jnp.full(npoi, 0.2)
        asuri_ib = jnp.full(npoi, 0.2)

        omega = jnp.full(npoi, 0.17)
        betad = jnp.full(npoi, 0.5)
        betai = jnp.full(npoi, 0.5)
        avmu = jnp.ones(npoi)
        gdir = jnp.full(npoi, 0.5)

        result = twostr(
            coszen, lai_iv, sai_iv, asurd_ib, asuri_ib,
            omega, betad, betai, avmu, gdir,
            consts.epsilon, sunlit, 1,
        )
        # No vegetation absorption
        assert jnp.allclose(result.abvegd, 0.0, atol=1e-6)
        assert jnp.allclose(result.abvegi, 0.0, atol=1e-6)

    def test_absorption_increases_with_lai(self, consts, veg_params):
        """More LAI should absorb more radiation."""
        coszen = jnp.array([0.5])
        sunlit = coszen > 0
        asurd_ib = jnp.array([0.15])
        asuri_ib = jnp.array([0.15])

        results = []
        for lai_val in [0.5, 2.0, 5.0]:
            lai_iv = jnp.array([lai_val])
            sai_iv = jnp.array([0.3])
            omega, betad, betai, avmu, gdir = twoset(
                coszen, lai_iv, sai_iv,
                jnp.zeros(1), jnp.zeros(1), jnp.full(1, 280.0),
                0.10, 0.07, 0.5, 0.0, 0.9, consts.tmelt, 1, sunlit,
            )
            r = twostr(
                coszen, lai_iv, sai_iv, asurd_ib, asuri_ib,
                omega, betad, betai, avmu, gdir,
                consts.epsilon, sunlit, 1,
            )
            results.append(float(r.abvegd[0]))

        assert results[0] < results[1] < results[2]


# ============================================================================
# irrad tests
# ============================================================================

class TestIrrad:
    """Infrared radiation tests."""

    def test_energy_balance(self, npoi, consts):
        """IR absorbed by components should roughly equal incoming - outgoing."""
        lai = jnp.full((npoi, 2), 2.0)
        sai = jnp.full((npoi, 2), 0.5)
        fu = jnp.full(npoi, 0.6)
        fl = jnp.full(npoi, 0.5)
        fi = jnp.full(npoi, 0.1)
        tu = jnp.full(npoi, 285.0)
        ts = jnp.full(npoi, 282.0)
        tl = jnp.full(npoi, 283.0)
        tg = jnp.full(npoi, 280.0)
        ti = jnp.full(npoi, 270.0)
        fira = jnp.full(npoi, 300.0)

        firu, firs, firl, firg, firi, firb = irrad(
            fira, lai, sai, fu, fl, fi,
            tu, ts, tl, tg, ti, consts.stef,
        )
        # All components should be finite
        for arr in [firu, firs, firl, firg, firi, firb]:
            assert jnp.all(jnp.isfinite(arr))

    def test_firb_positive(self, npoi, consts):
        """Outgoing IR should be positive."""
        lai = jnp.full((npoi, 2), 1.0)
        sai = jnp.full((npoi, 2), 0.2)
        fu = jnp.full(npoi, 0.5)
        fl = jnp.full(npoi, 0.5)
        fi = jnp.zeros(npoi)
        tu = jnp.full(npoi, 290.0)
        ts = jnp.full(npoi, 288.0)
        tl = jnp.full(npoi, 288.0)
        tg = jnp.full(npoi, 285.0)
        ti = jnp.full(npoi, 270.0)
        fira = jnp.full(npoi, 350.0)

        _, _, _, _, _, firb = irrad(
            fira, lai, sai, fu, fl, fi,
            tu, ts, tl, tg, ti, consts.stef,
        )
        assert jnp.all(firb > 0.0)

    def test_no_canopy_soil_only(self, npoi, consts):
        """With no canopy, soil should absorb/emit directly."""
        lai = jnp.zeros((npoi, 2))
        sai = jnp.zeros((npoi, 2))
        fu = jnp.zeros(npoi)
        fl = jnp.zeros(npoi)
        fi = jnp.zeros(npoi)
        tu = jnp.full(npoi, 280.0)
        ts = jnp.full(npoi, 280.0)
        tl = jnp.full(npoi, 280.0)
        tg = jnp.full(npoi, 285.0)
        ti = jnp.full(npoi, 270.0)
        fira = jnp.full(npoi, 300.0)

        firu, firs, firl, firg, firi, firb = irrad(
            fira, lai, sai, fu, fl, fi,
            tu, ts, tl, tg, ti, consts.stef,
        )
        # With no canopy: firu=firs=0 (emu=ems=0), firg = fira - emg*tg4
        assert jnp.allclose(firu, 0.0, atol=1e-6)
        assert jnp.allclose(firs, 0.0, atol=1e-6)


# ============================================================================
# compute_radiation integration tests
# ============================================================================

class TestComputeRadiation:
    """Integration tests for the full radiation computation."""

    def _make_inputs(self, npoi, consts, veg_params):
        """Create a consistent set of radiation inputs."""
        return dict(
            coszen=jnp.full(npoi, 0.5),
            solad=jnp.full((npoi, 2), 200.0),
            solai=jnp.full((npoi, 2), 50.0),
            fira=jnp.full(npoi, 300.0),
            lai=jnp.full((npoi, 2), 2.0),
            sai=jnp.full((npoi, 2), 0.5),
            fu=jnp.full(npoi, 0.5),
            fl=jnp.full(npoi, 0.5),
            tu=jnp.full(npoi, 285.0),
            ts=jnp.full(npoi, 282.0),
            tl=jnp.full(npoi, 283.0),
            tg=jnp.full(npoi, 280.0),
            ti=jnp.full(npoi, 270.0),
            fi=jnp.full(npoi, 0.1),
            tsno_1=jnp.full(npoi, 265.0),
            wsoi_1=jnp.full(npoi, 0.4),
            wisoi_1=jnp.zeros(npoi),
            albsav=jnp.full(npoi, 0.1),
            albsan=jnp.full(npoi, 0.2),
            fwetu=jnp.zeros(npoi),
            fwets=jnp.zeros(npoi),
            fwetl=jnp.zeros(npoi),
            rliqu=jnp.ones(npoi),
            rliqs=jnp.ones(npoi),
            rliql=jnp.ones(npoi),
            veg_params=veg_params,
            stef=consts.stef,
            tmelt=consts.tmelt,
            epsilon=consts.epsilon,
        )

    def test_all_outputs_finite(self, consts, veg_params):
        npoi = 4
        inputs = self._make_inputs(npoi, consts, veg_params)
        result = compute_radiation(**inputs)
        for field in result._fields:
            arr = getattr(result, field)
            assert jnp.all(jnp.isfinite(arr)), f"{field} has non-finite values"

    def test_solar_absorption_positive(self, consts, veg_params):
        npoi = 4
        inputs = self._make_inputs(npoi, consts, veg_params)
        result = compute_radiation(**inputs)
        assert jnp.all(result.solu >= 0.0)
        assert jnp.all(result.sols >= 0.0)
        assert jnp.all(result.soll >= 0.0)
        assert jnp.all(result.solg >= 0.0)
        assert jnp.all(result.soli >= 0.0)

    def test_albedo_bounded(self, consts, veg_params):
        npoi = 4
        inputs = self._make_inputs(npoi, consts, veg_params)
        result = compute_radiation(**inputs)
        assert jnp.all(result.asurd >= 0.0)
        assert jnp.all(result.asurd <= 1.0)
        assert jnp.all(result.asuri >= 0.0)
        assert jnp.all(result.asuri <= 1.0)

    def test_reflected_bounded_by_incoming(self, consts, veg_params):
        """Reflected solar should not exceed incoming."""
        npoi = 4
        inputs = self._make_inputs(npoi, consts, veg_params)
        result = compute_radiation(**inputs)

        total_incoming = (inputs['solad'].sum(axis=1)
                          + inputs['solai'].sum(axis=1))
        total_reflected = (result.asurd * inputs['solad']
                           + result.asuri * inputs['solai']).sum(axis=1)
        assert jnp.all(total_reflected >= 0.0)
        assert jnp.all(total_reflected <= total_incoming + 1.0)

    def test_night_zero_solar(self, consts, veg_params):
        """At night (coszen=0), all solar fluxes should be zero."""
        npoi = 4
        inputs = self._make_inputs(npoi, consts, veg_params)
        inputs['coszen'] = jnp.zeros(npoi)
        result = compute_radiation(**inputs)
        assert jnp.allclose(result.solu, 0.0)
        assert jnp.allclose(result.sols, 0.0)
        assert jnp.allclose(result.soll, 0.0)
        assert jnp.allclose(result.solg, 0.0)
        assert jnp.allclose(result.soli, 0.0)

    def test_jit_compatible(self, consts, veg_params):
        """compute_radiation must work under jax.jit."""
        npoi = 4
        inputs = self._make_inputs(npoi, consts, veg_params)
        result_eager = compute_radiation(**inputs)
        result_jit = jax.jit(compute_radiation)(**inputs)
        assert jnp.allclose(result_eager.solu, result_jit.solu, atol=1e-6)
        assert jnp.allclose(result_eager.firb, result_jit.firb, atol=1e-6)

    def test_differentiable(self, consts, veg_params):
        """Radiation must be differentiable w.r.t. coszen."""
        npoi = 4
        inputs = self._make_inputs(npoi, consts, veg_params)

        def loss_fn(coszen):
            inputs_local = dict(inputs)
            inputs_local['coszen'] = coszen
            result = compute_radiation(**inputs_local)
            return result.solg.sum()

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(inputs['coszen'])
        assert jnp.all(jnp.isfinite(g))

    def test_vectorized(self, consts, veg_params):
        """Should handle different array sizes."""
        for npoi in [1, 4, 16]:
            inputs = self._make_inputs(npoi, consts, veg_params)
            result = compute_radiation(**inputs)
            assert result.solu.shape == (npoi,)
            assert result.asurd.shape == (npoi, 2)
