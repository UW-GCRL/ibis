"""Tests for ibis.physics.weather — Richardson weather generator + diurnal."""

import jax
import jax.numpy as jnp
import pytest

from ibis.physics.weather import (
    ClimateData,
    MonthlyPrecip,
    generate_month_precip,
    daily,
    diurnal,
    compute_precip_timing,
    _interp_weights,
)

NPOI = 8
KEY = jax.random.PRNGKey(42)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def climate():
    """Standard tropical/temperate climate data."""
    npoi = NPOI
    return ClimateData(
        xint=jnp.full((npoi, 12), 20.0),        # 20 C mean
        xinq=jnp.full((npoi, 12), 70.0),         # 70% RH
        xinprec=jnp.full((npoi, 12), 3.0),       # 3 mm/day
        xinwind=jnp.full((npoi, 12), 3.5),       # 3.5 m/s
        xincld=jnp.full((npoi, 12), 50.0),       # 50% cloud
        xinwet=jnp.full((npoi, 12), 10.0),       # 10 wet days/month
        xintrng=jnp.full((npoi, 12), 8.0),       # 8 C diurnal range
        xintopo=jnp.full((npoi,), 100.0),        # 100 m elevation
    )


@pytest.fixture
def monthly_precip(climate):
    """Pre-generated monthly precipitation."""
    _, mp = generate_month_precip(KEY, climate, imonth=0)
    return mp


@pytest.fixture
def daily_weather(climate, monthly_precip):
    """Standard daily weather output."""
    xstore = jnp.zeros((NPOI, 3))
    _, dw = daily(KEY, imonth=0, iday=15, climate=climate,
                  monthly_precip=monthly_precip, xstore=xstore)
    return dw


# ===========================================================================
# Interpolation weights
# ===========================================================================

class TestInterpWeights:
    def test_first_half(self):
        """Day 1 of a 31-day month should use previous and current month."""
        it1w, it2w, dt = _interp_weights(6, 0, 31)  # July day 1
        assert it1w == 5   # June
        assert it2w == 6   # July

    def test_second_half(self):
        """Day 20 of a 31-day month should use current and next month."""
        it1w, it2w, dt = _interp_weights(6, 20, 31)  # July day 21
        assert it1w == 6   # July
        assert it2w == 7   # August

    def test_december_wraps(self):
        """December second half should wrap to January."""
        it1w, it2w, dt = _interp_weights(11, 20, 31)
        assert it2w == 0   # January

    def test_january_wraps(self):
        """January first half should wrap to December."""
        it1w, it2w, dt = _interp_weights(0, 0, 31)
        assert it1w == 11  # December


# ===========================================================================
# Monthly precipitation generation
# ===========================================================================

class TestGenerateMonthPrecip:
    def test_shapes(self, climate):
        """Output arrays should have correct shapes."""
        _, mp = generate_month_precip(KEY, climate, imonth=0)
        assert mp.iwetday.shape == (NPOI, 31)
        assert mp.precipday.shape == (NPOI, 31)

    def test_total_matches_target(self, climate):
        """Total monthly precipitation should match climatological target."""
        _, mp = generate_month_precip(KEY, climate, imonth=0)
        total = jnp.sum(mp.precipday, axis=1)
        target = climate.xinprec[:, 0] * 31  # 3 mm/day * 31 days = 93 mm
        assert jnp.allclose(total, target, rtol=0.1)

    def test_no_precip_past_month(self, climate):
        """No precipitation on days beyond month length (Feb has 28 days)."""
        _, mp = generate_month_precip(KEY, climate, imonth=1)  # February
        assert jnp.allclose(mp.precipday[:, 28:], 0.0)
        assert jnp.all(mp.iwetday[:, 28:] == 0)

    def test_precip_nonnegative(self, climate):
        """All precipitation amounts should be non-negative."""
        _, mp = generate_month_precip(KEY, climate, imonth=0)
        assert jnp.all(mp.precipday >= 0.0)

    def test_wet_days_binary(self, climate):
        """Wet day flags should be 0 or 1."""
        _, mp = generate_month_precip(KEY, climate, imonth=0)
        assert jnp.all((mp.iwetday == 0) | (mp.iwetday == 1))

    def test_dry_climate(self):
        """Zero precipitation climate should produce near-zero precip."""
        npoi = 4
        clim = ClimateData(
            xint=jnp.full((npoi, 12), 25.0),
            xinq=jnp.full((npoi, 12), 30.0),
            xinprec=jnp.zeros((npoi, 12)),         # no rain
            xinwind=jnp.full((npoi, 12), 3.0),
            xincld=jnp.full((npoi, 12), 20.0),
            xinwet=jnp.full((npoi, 12), 0.0),
            xintrng=jnp.full((npoi, 12), 12.0),
            xintopo=jnp.zeros((npoi,)),
        )
        _, mp = generate_month_precip(KEY, clim, imonth=6)
        assert jnp.allclose(jnp.sum(mp.precipday), 0.0)

    def test_reproducible(self, climate):
        """Same key should produce same results."""
        _, mp1 = generate_month_precip(KEY, climate, imonth=3)
        _, mp2 = generate_month_precip(KEY, climate, imonth=3)
        assert jnp.allclose(mp1.precipday, mp2.precipday)


# ===========================================================================
# Daily weather
# ===========================================================================

class TestDaily:
    def test_outputs_finite(self, climate, monthly_precip):
        """All daily weather outputs should be finite."""
        xstore = jnp.zeros((NPOI, 3))
        _, dw = daily(KEY, 0, 15, climate, monthly_precip, xstore)
        for name in dw._fields:
            val = getattr(dw, name)
            assert jnp.all(jnp.isfinite(val)), f"Non-finite in {name}"

    def test_temperature_ordering(self, climate, monthly_precip):
        """tmin <= td <= tmax should hold."""
        xstore = jnp.zeros((NPOI, 3))
        _, dw = daily(KEY, 0, 15, climate, monthly_precip, xstore)
        assert jnp.all(dw.tmin <= dw.tmax + 1e-5)
        assert jnp.all(dw.td >= dw.tmin - 1e-5)
        assert jnp.all(dw.td <= dw.tmax + 1e-5)

    def test_td_reasonable(self, climate, monthly_precip):
        """Daily mean temp should be within ~30 K of climatological mean."""
        xstore = jnp.zeros((NPOI, 3))
        _, dw = daily(KEY, 0, 15, climate, monthly_precip, xstore)
        # Climate is 20C = 293.16K
        assert jnp.all(dw.td > 253.0)
        assert jnp.all(dw.td < 333.0)

    def test_cloud_bounded(self, climate, monthly_precip):
        """Cloud fraction should be in [0, 1]."""
        xstore = jnp.zeros((NPOI, 3))
        _, dw = daily(KEY, 0, 15, climate, monthly_precip, xstore)
        assert jnp.all(dw.cloud >= 0.0)
        assert jnp.all(dw.cloud <= 1.0)

    def test_psurf_reasonable(self, climate, monthly_precip):
        """Surface pressure should be near sea level."""
        xstore = jnp.zeros((NPOI, 3))
        _, dw = daily(KEY, 0, 15, climate, monthly_precip, xstore)
        assert jnp.all(dw.psurf > 80000.0)
        assert jnp.all(dw.psurf < 110000.0)

    def test_qd_positive(self, climate, monthly_precip):
        """Specific humidity should be positive."""
        xstore = jnp.zeros((NPOI, 3))
        _, dw = daily(KEY, 0, 15, climate, monthly_precip, xstore)
        assert jnp.all(dw.qd > 0.0)

    def test_wind_bounded(self, climate, monthly_precip):
        """Wind should be in [2.5, 10.0] m/s."""
        xstore = jnp.zeros((NPOI, 3))
        _, dw = daily(KEY, 0, 15, climate, monthly_precip, xstore)
        assert jnp.all(dw.ud >= 2.5)
        assert jnp.all(dw.ud <= 10.0)

    def test_xstore_updated(self, climate, monthly_precip):
        """AR(1) state should be updated (not all zeros)."""
        xstore = jnp.zeros((NPOI, 3))
        _, dw = daily(KEY, 0, 15, climate, monthly_precip, xstore)
        assert not jnp.allclose(dw.xstore, 0.0)


# ===========================================================================
# Precipitation timing
# ===========================================================================

class TestPrecipTiming:
    def test_plens_positive(self):
        """Event duration should be positive."""
        precip = jnp.array([5.0, 10.0, 50.0, 100.0])
        _, plens, startp, endp = compute_precip_timing(KEY, precip, 3600.0)
        assert jnp.all(plens > 0.0)

    def test_endp_after_startp(self):
        """End time should be after start time."""
        precip = jnp.array([5.0, 10.0, 50.0, 100.0])
        _, plens, startp, endp = compute_precip_timing(KEY, precip, 3600.0)
        assert jnp.all(endp > startp)

    def test_heavy_rain_extends_event(self):
        """Heavy precipitation should result in longer event duration."""
        precip_light = jnp.array([5.0])
        precip_heavy = jnp.array([200.0])
        _, plens_l, _, _ = compute_precip_timing(KEY, precip_light, 3600.0)
        _, plens_h, _, _ = compute_precip_timing(KEY, precip_heavy, 3600.0)
        assert plens_h >= plens_l


# ===========================================================================
# Diurnal disaggregation
# ===========================================================================

class TestDiurnal:
    def test_outputs_finite(self, daily_weather):
        """All hourly outputs should be finite."""
        lat = jnp.full((NPOI,), 30.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 6 * 3600.0)
        endp = startp + plens
        _, hw = diurnal(12 * 3600.0, 172, daily_weather, lat,
                        plens, startp, endp, KEY, 3600.0)
        for name in hw._fields:
            val = getattr(hw, name)
            assert jnp.all(jnp.isfinite(val)), f"Non-finite in {name}"

    def test_coszen_bounded(self, daily_weather):
        """Cosine of zenith angle should be in [0, 1]."""
        lat = jnp.full((NPOI,), 45.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 6 * 3600.0)
        endp = startp + plens
        _, hw = diurnal(12 * 3600.0, 172, daily_weather, lat,
                        plens, startp, endp, KEY, 3600.0)
        assert jnp.all(hw.coszen >= 0.0)
        assert jnp.all(hw.coszen <= 1.0)

    def test_night_zero_solar(self, daily_weather):
        """At midnight, solar radiation should be zero."""
        lat = jnp.full((NPOI,), 45.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 6 * 3600.0)
        endp = startp + plens
        _, hw = diurnal(0.0, 172, daily_weather, lat,
                        plens, startp, endp, KEY, 3600.0)
        assert jnp.allclose(hw.solad, 0.0)
        assert jnp.allclose(hw.solai, 0.0)

    def test_noon_solar_positive(self, daily_weather):
        """At noon in summer, solar should be positive for mid-latitudes."""
        lat = jnp.full((NPOI,), 30.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 6 * 3600.0)
        endp = startp + plens
        _, hw = diurnal(12 * 3600.0, 172, daily_weather, lat,
                        plens, startp, endp, KEY, 3600.0)
        assert jnp.all(hw.solad > 0.0)
        assert jnp.all(hw.solai > 0.0)

    def test_ta_between_tmin_tmax(self, daily_weather):
        """Hourly temperature should be between tmin and tmax."""
        lat = jnp.full((NPOI,), 30.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 6 * 3600.0)
        endp = startp + plens
        _, hw = diurnal(15 * 3600.0, 172, daily_weather, lat,
                        plens, startp, endp, KEY, 3600.0)
        # Gamma can be slightly > 1, so allow small margin
        assert jnp.all(hw.ta >= daily_weather.tmin - 5.0)
        assert jnp.all(hw.ta <= daily_weather.tmax + 5.0)

    def test_fira_positive(self, daily_weather):
        """Downward longwave should be positive."""
        lat = jnp.full((NPOI,), 30.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 6 * 3600.0)
        endp = startp + plens
        _, hw = diurnal(12 * 3600.0, 172, daily_weather, lat,
                        plens, startp, endp, KEY, 3600.0)
        assert jnp.all(hw.fira > 0.0)

    def test_rain_only_during_event(self, daily_weather):
        """Rain should only occur during the precipitation event."""
        lat = jnp.full((NPOI,), 30.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 10 * 3600.0)
        endp = startp + plens
        # Before event start
        _, hw_before = diurnal(5 * 3600.0, 172, daily_weather, lat,
                               plens, startp, endp, KEY, 3600.0)
        assert jnp.allclose(hw_before.raina, 0.0)
        assert jnp.allclose(hw_before.snowa, 0.0)

    def test_wind_bounded(self, daily_weather):
        """Hourly wind should be in [2.5, 10.0] m/s."""
        lat = jnp.full((NPOI,), 30.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 6 * 3600.0)
        endp = startp + plens
        _, hw = diurnal(12 * 3600.0, 172, daily_weather, lat,
                        plens, startp, endp, KEY, 3600.0)
        assert jnp.all(hw.ua >= 2.5)
        assert jnp.all(hw.ua <= 10.0)

    def test_two_wavebands(self, daily_weather):
        """Solar output should have 2 wavebands."""
        lat = jnp.full((NPOI,), 30.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 6 * 3600.0)
        endp = startp + plens
        _, hw = diurnal(12 * 3600.0, 172, daily_weather, lat,
                        plens, startp, endp, KEY, 3600.0)
        assert hw.solad.shape == (NPOI, 2)
        assert hw.solai.shape == (NPOI, 2)

    def test_jit_compatible(self, daily_weather):
        """Diurnal should be JIT-compilable."""
        lat = jnp.full((NPOI,), 30.0)
        plens = jnp.full((NPOI,), 4 * 3600.0)
        startp = jnp.full((NPOI,), 6 * 3600.0)
        endp = startp + plens
        jitted = jax.jit(diurnal, static_argnames=['nband'])
        _, hw = jitted(12 * 3600.0, 172, daily_weather, lat,
                        plens, startp, endp, KEY, 3600.0)
        assert jnp.all(jnp.isfinite(hw.ta))
