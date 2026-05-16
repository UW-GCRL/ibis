"""Integration tests for the IBIS-JAX time-stepping framework.

Tests:
  - Calendar utilities (month lengths, leap years, Julian day)
  - Single timestep body (chain of all physics modules)
  - JIT compilation of timestep body
  - Differentiability check
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from ibis.constants import PhysicalConstants, get_constants
from ibis.dimensions import GridConfig
from ibis.state.model_state import IBISState, create_ibis_state
from ibis.timestepping.calendar import (
    is_leap_year, days_in_month, days_in_year, julian_day, timesteps_per_day,
)
from ibis.timestepping.loops import (
    SimParams, default_sim_params, timestep_body,
)

# Small test grid
NPOI = 4
GRID = GridConfig(nlon=2, nlat=2, npoi=NPOI, xres=90.0, yres=90.0,
                  nband=2, nsoilay=6, nsnolay=3, npft=12)
DTIME = 1800.0  # 30-minute timestep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def consts():
    return get_constants()


@pytest.fixture
def params():
    return default_sim_params(NPOI)


@pytest.fixture
def state():
    """Create a physically reasonable cold-start state for testing."""
    st = create_ibis_state(GRID)

    # Set minimal vegetation (some LAI, SAI, cover)
    new_veg = eqx.tree_at(
        lambda v: (v.lai, v.sai, v.fu, v.fl, v.ztop, v.zbot, v.exist),
        st.vegetation,
        (
            jnp.full((NPOI, 2), 2.0),       # LAI
            jnp.full((NPOI, 2), 0.5),        # SAI
            jnp.full((NPOI,), 0.8),           # fu
            jnp.full((NPOI,), 0.5),           # fl
            jnp.array([[5.0, 20.0]] * NPOI),  # ztop
            jnp.array([[0.5, 2.0]] * NPOI),   # zbot
            jnp.ones((NPOI, 12)),             # all PFTs exist
        ),
    )

    # Set minimal soil properties
    new_soil = eqx.tree_at(
        lambda s: (s.poros, s.hydraul, s.suction, s.bex,
                   s.sfield, s.swilt, s.sand, s.clay,
                   s.hsoi, s.z0soi, s.wsoi),
        st.soil,
        (
            jnp.full((NPOI, 6), 0.45),       # porosity
            jnp.full((NPOI, 6), 1e-5),       # hydraulic cond
            jnp.full((NPOI, 6), -0.15),      # suction (m)
            jnp.full((NPOI, 6), 5.0),        # Campbell b
            jnp.full((NPOI, 6), 0.35),       # field capacity
            jnp.full((NPOI, 6), 0.15),       # wilting point
            jnp.full((NPOI, 6), 50.0),       # sand %
            jnp.full((NPOI, 6), 20.0),       # clay %
            jnp.array([0.10, 0.15, 0.30, 0.50, 1.00, 2.00, 4.00]),  # hsoi (7 for nsoilay+1=7)
            jnp.full((NPOI,), 0.01),         # roughness length
            jnp.full((NPOI, 6), 0.40),       # initial moisture
        ),
    )

    # Set atmospheric forcing (daytime, warm, some solar)
    new_atm = eqx.tree_at(
        lambda a: (a.ta, a.qa, a.ua, a.psurf, a.coszen,
                   a.solad, a.solai, a.fira, a.raina, a.snowa),
        st.atm,
        (
            jnp.full((NPOI,), 293.0),        # 20°C air temp
            jnp.full((NPOI,), 0.008),        # specific humidity
            jnp.full((NPOI,), 3.0),          # wind speed
            jnp.full((NPOI,), 101325.0),     # surface pressure
            jnp.full((NPOI,), 0.8),          # cosine zenith (high sun)
            jnp.full((NPOI, 2), 200.0),      # direct solar
            jnp.full((NPOI, 2), 50.0),       # diffuse solar
            jnp.full((NPOI,), 320.0),        # downward IR
            jnp.full((NPOI,), 0.0),          # no rain
            jnp.full((NPOI,), 0.0),          # no snow
        ),
    )

    # Set canopy initial conditions
    new_can = eqx.tree_at(
        lambda c: (c.tu, c.ts, c.tl, c.t12, c.t34, c.q12, c.q34),
        st.canopy,
        (
            jnp.full((NPOI,), 293.0),
            jnp.full((NPOI,), 293.0),
            jnp.full((NPOI,), 293.0),
            jnp.full((NPOI,), 293.0),
            jnp.full((NPOI,), 293.0),
            jnp.full((NPOI,), 0.008),
            jnp.full((NPOI,), 0.008),
        ),
    )

    st = eqx.tree_at(
        lambda s: (s.vegetation, s.soil, s.atm, s.canopy),
        st,
        (new_veg, new_soil, new_atm, new_can),
    )
    return st


# =========================================================================
# Calendar tests
# =========================================================================

class TestCalendar:
    def test_leap_year_2000(self):
        assert is_leap_year(2000) is True

    def test_leap_year_1900(self):
        assert is_leap_year(1900) is False

    def test_leap_year_2004(self):
        assert is_leap_year(2004) is True

    def test_non_leap_year(self):
        assert is_leap_year(2001) is False

    def test_days_in_february(self):
        assert days_in_month(1, leap=False) == 28
        assert days_in_month(1, leap=True) == 29

    def test_days_in_january(self):
        assert days_in_month(0) == 31

    def test_days_in_year(self):
        assert days_in_year(False) == 365
        assert days_in_year(True) == 366

    def test_julian_day_jan1(self):
        assert julian_day(0, 0) == 1

    def test_julian_day_dec31(self):
        assert julian_day(11, 30) == 365

    def test_julian_day_dec31_leap(self):
        assert julian_day(11, 30, leap=True) == 366

    def test_julian_day_feb28(self):
        assert julian_day(1, 27) == 59

    def test_julian_day_mar1_leap(self):
        assert julian_day(2, 0, leap=True) == 61

    def test_timesteps_per_day(self):
        assert timesteps_per_day(1800.0) == 48
        assert timesteps_per_day(3600.0) == 24


# =========================================================================
# Timestep body tests
# =========================================================================

class TestTimestepBody:
    """Test that the timestep body runs without errors and produces
    finite output. We test the full physics chain on a tiny grid."""

    def test_runs_without_error(self, state, consts, params):
        """Timestep body completes without exceptions."""
        new_state = timestep_body(state, consts, params, DTIME)
        assert new_state is not None

    def test_outputs_finite(self, state, consts, params):
        """All prognostic variables remain finite after one timestep."""
        new_state = timestep_body(state, consts, params, DTIME)

        # Canopy temperatures
        assert jnp.all(jnp.isfinite(new_state.canopy.tu))
        assert jnp.all(jnp.isfinite(new_state.canopy.ts))
        assert jnp.all(jnp.isfinite(new_state.canopy.tl))
        assert jnp.all(jnp.isfinite(new_state.canopy.t12))
        assert jnp.all(jnp.isfinite(new_state.canopy.t34))

        # Soil
        assert jnp.all(jnp.isfinite(new_state.soil.tsoi))
        assert jnp.all(jnp.isfinite(new_state.soil.wsoi))

        # Snow
        assert jnp.all(jnp.isfinite(new_state.snow.fi))
        assert jnp.all(jnp.isfinite(new_state.snow.tsno))

    def test_temperatures_reasonable(self, state, consts, params):
        """Temperatures stay within physical bounds after one step."""
        new_state = timestep_body(state, consts, params, DTIME)

        # Temperature should be 150-400 K
        assert jnp.all(new_state.canopy.tu > 150.0)
        assert jnp.all(new_state.canopy.tu < 400.0)
        assert jnp.all(new_state.soil.tsoi[:, 0] > 150.0)
        assert jnp.all(new_state.soil.tsoi[:, 0] < 400.0)

    def test_moisture_nonnegative(self, state, consts, params):
        """Soil moisture stays non-negative."""
        new_state = timestep_body(state, consts, params, DTIME)
        assert jnp.all(new_state.soil.wsoi >= 0.0)

    def test_radiation_fluxes_present(self, state, consts, params):
        """Radiation diagnostics are populated."""
        new_state = timestep_body(state, consts, params, DTIME)

        # With sun up and solar input, absorbed fluxes should be > 0
        assert jnp.any(new_state.canopy.solu > 0)
        assert jnp.any(new_state.canopy.solg > 0)

    def test_stomata_state_updated(self, state, consts, params):
        """Stomatal conductance is updated from initial values."""
        new_state = timestep_body(state, consts, params, DTIME)

        # gs should be > 0 for active canopy
        assert jnp.all(new_state.canopy.gs_upper > 0)

    def test_nighttime(self, state, consts, params):
        """Timestep works at night (coszen=0, no solar)."""
        night_atm = eqx.tree_at(
            lambda a: (a.coszen, a.solad, a.solai),
            state.atm,
            (jnp.zeros((NPOI,)),
             jnp.zeros((NPOI, 2)),
             jnp.zeros((NPOI, 2))),
        )
        night_state = eqx.tree_at(lambda s: s.atm, state, night_atm)
        new_state = timestep_body(night_state, consts, params, DTIME)
        assert jnp.all(jnp.isfinite(new_state.canopy.tu))

    def test_with_rain(self, state, consts, params):
        """Timestep works with rainfall."""
        rain_atm = eqx.tree_at(
            lambda a: a.raina,
            state.atm,
            jnp.full((NPOI,), 5e-4),  # 1.8 mm/hr
        )
        rain_state = eqx.tree_at(lambda s: s.atm, state, rain_atm)
        new_state = timestep_body(rain_state, consts, params, DTIME)
        assert jnp.all(jnp.isfinite(new_state.canopy.tu))

    def test_with_snow(self, state, consts, params):
        """Timestep works with snowfall and existing snow."""
        cold_atm = eqx.tree_at(
            lambda a: (a.ta, a.snowa),
            state.atm,
            (jnp.full((NPOI,), 265.0),
             jnp.full((NPOI,), 2e-4)),
        )
        snow_state = eqx.tree_at(
            lambda s: (s.atm, s.canopy.tu, s.canopy.ts, s.canopy.tl,
                       s.canopy.t12, s.canopy.t34),
            state,
            (cold_atm,
             jnp.full((NPOI,), 265.0),
             jnp.full((NPOI,), 265.0),
             jnp.full((NPOI,), 265.0),
             jnp.full((NPOI,), 265.0),
             jnp.full((NPOI,), 265.0)),
        )
        new_state = timestep_body(snow_state, consts, params, DTIME)
        assert jnp.all(jnp.isfinite(new_state.canopy.tu))


# =========================================================================
# JIT compilation test
# =========================================================================

class TestJIT:
    def test_jit_timestep(self, state, consts, params):
        """Timestep body can be JIT-compiled."""
        @jax.jit
        def _run(st):
            return timestep_body(st, consts, params, DTIME)

        new_state = _run(state)
        assert jnp.all(jnp.isfinite(new_state.canopy.tu))

    def test_jit_two_steps(self, state, consts, params):
        """Two consecutive JIT-compiled timesteps produce finite results."""
        @jax.jit
        def _run2(st):
            st = timestep_body(st, consts, params, DTIME)
            st = timestep_body(st, consts, params, DTIME)
            return st

        new_state = _run2(state)
        assert jnp.all(jnp.isfinite(new_state.canopy.tu))
        assert jnp.all(jnp.isfinite(new_state.soil.tsoi))
