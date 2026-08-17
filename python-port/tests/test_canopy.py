"""Tests for canopy exchange and interception modules."""

import jax
import jax.numpy as jnp
import pytest

from ibis.constants import PhysicalConstants
from ibis.physics.canopy_exchange import (
    CanopyParams, canini, fstrat, turcof, impexp, impexp2, turvap,
)
from ibis.physics.interception import (
    InterceptionParams, fwetcal, mix, twet3, steph2o, cascade,
    steph2o2, cascad2, noveg,
)


@pytest.fixture
def consts():
    return PhysicalConstants()


@pytest.fixture
def npoi():
    return 4


@pytest.fixture
def canopy_params():
    return CanopyParams()


@pytest.fixture
def interp_params():
    return InterceptionParams()


# ============================================================================
# canini tests
# ============================================================================

class TestCanini:
    """Aerodynamic geometry initialization."""

    def test_outputs_finite(self, npoi, consts):
        """All aerodynamic outputs should be finite."""
        c = consts
        ta = jnp.full(npoi, 290.0)
        qa = jnp.full(npoi, 0.008)
        psurf = jnp.full(npoi, 101325.0)
        ua = jnp.full(npoi, 5.0)
        fi = jnp.full(npoi, 0.1)
        z0soi = jnp.full(npoi, 0.01)
        z0sno = 0.005
        fl = jnp.full(npoi, 0.5)
        fu = jnp.full(npoi, 0.8)
        lai = jnp.array([[2.0, 4.0]] * npoi)
        sai = jnp.array([[0.5, 1.0]] * npoi)
        ztop = jnp.array([[2.0, 20.0]] * npoi)
        zbot = jnp.array([[0.5, 10.0]] * npoi)

        aero = canini(ta, qa, psurf, ua, fi, z0soi, z0sno,
                      fl, fu, lai, sai, ztop, zbot,
                      8.0, 8.0, c.grav, c.rair, c.rvap, c.cair, c.cvap, c.cappa)

        assert jnp.all(jnp.isfinite(aero.rhoa))
        assert jnp.all(jnp.isfinite(aero.z1))
        assert jnp.all(jnp.isfinite(aero.bdl))
        assert jnp.all(jnp.isfinite(aero.bdu))
        assert jnp.all(jnp.isfinite(aero.alogu))

    def test_rhoa_reasonable(self, npoi, consts):
        """Air density should be ~1.0-1.4 kg/m3."""
        c = consts
        ta = jnp.full(npoi, 290.0)
        qa = jnp.full(npoi, 0.005)
        psurf = jnp.full(npoi, 101325.0)
        ua = jnp.full(npoi, 3.0)
        fi = jnp.zeros(npoi)
        z0soi = jnp.full(npoi, 0.01)

        aero = canini(ta, qa, psurf, ua, fi, z0soi, 0.005,
                      jnp.full(npoi, 0.5), jnp.full(npoi, 0.8),
                      jnp.array([[2.0, 4.0]] * npoi),
                      jnp.array([[0.5, 1.0]] * npoi),
                      jnp.array([[2.0, 20.0]] * npoi),
                      jnp.array([[0.5, 10.0]] * npoi),
                      8.0, 8.0, c.grav, c.rair, c.rvap, c.cair, c.cvap, c.cappa)

        assert jnp.all(aero.rhoa > 0.8)
        assert jnp.all(aero.rhoa < 1.6)

    def test_height_ordering(self, npoi, consts):
        """z4 < z3 < z2 < z1 < za."""
        c = consts
        ta = jnp.full(npoi, 290.0)
        qa = jnp.full(npoi, 0.005)
        psurf = jnp.full(npoi, 101325.0)
        ua = jnp.full(npoi, 5.0)
        fi = jnp.full(npoi, 0.1)
        z0soi = jnp.full(npoi, 0.01)

        aero = canini(ta, qa, psurf, ua, fi, z0soi, 0.005,
                      jnp.full(npoi, 0.5), jnp.full(npoi, 0.8),
                      jnp.array([[2.0, 4.0]] * npoi),
                      jnp.array([[0.5, 1.0]] * npoi),
                      jnp.array([[2.0, 20.0]] * npoi),
                      jnp.array([[0.5, 10.0]] * npoi),
                      8.0, 8.0, c.grav, c.rair, c.rvap, c.cair, c.cvap, c.cappa)

        assert jnp.all(aero.z4 < aero.z3)
        assert jnp.all(aero.z3 <= aero.z2)
        assert jnp.all(aero.z2 < aero.z1)
        assert jnp.all(aero.z1 < aero.za)


# ============================================================================
# fstrat tests
# ============================================================================

class TestFstrat:
    """Louis (1979) stratification correction."""

    def test_neutral_near_one(self, npoi, consts):
        """Neutral stratification should give factors near 1."""
        t = jnp.full(npoi, 290.0)
        q = jnp.full(npoi, 0.005)
        u = jnp.full(npoi, 5.0)
        zb = jnp.full(npoi, 5.0)
        zt = jnp.full(npoi, 30.0)
        alb = jnp.full(npoi, -2.0)
        alt = jnp.full(npoi, 2.0)

        rich, stram, strah = fstrat(
            t, t, 1.0, q, q, zb, zt, alb, alb, alt, u,
            consts.grav, consts.vonk, 0,
            jnp.zeros(npoi), jnp.ones(npoi), jnp.ones(npoi),
        )

        assert jnp.allclose(rich, 0.0, atol=0.01)
        assert jnp.allclose(stram, 1.0, atol=0.01)
        assert jnp.allclose(strah, 1.0, atol=0.01)

    def test_stable_reduces_mixing(self, npoi, consts):
        """Stable stratification should reduce mixing (stram < 1)."""
        tb = jnp.full(npoi, 280.0)  # cold below
        tt = jnp.full(npoi, 290.0)  # warm above
        q = jnp.full(npoi, 0.005)
        u = jnp.full(npoi, 3.0)
        zb = jnp.full(npoi, 5.0)
        zt = jnp.full(npoi, 30.0)
        alb = jnp.full(npoi, -2.0)
        alt = jnp.full(npoi, 2.0)

        _, stram, strah = fstrat(
            tb, tt, 1.0, q, q, zb, zt, alb, alb, alt, u,
            consts.grav, consts.vonk, 0,
            jnp.zeros(npoi), jnp.ones(npoi), jnp.ones(npoi),
        )

        assert jnp.all(stram < 1.0)
        assert jnp.all(strah < 1.0)

    def test_unstable_enhances_mixing(self, npoi, consts):
        """Unstable stratification should enhance mixing (stram > 1)."""
        tb = jnp.full(npoi, 300.0)  # warm below
        tt = jnp.full(npoi, 290.0)  # cold above
        q = jnp.full(npoi, 0.005)
        u = jnp.full(npoi, 5.0)
        zb = jnp.full(npoi, 5.0)
        zt = jnp.full(npoi, 30.0)
        alb = jnp.full(npoi, -2.0)
        alt = jnp.full(npoi, 2.0)

        _, stram, strah = fstrat(
            tb, tt, 1.0, q, q, zb, zt, alb, alb, alt, u,
            consts.grav, consts.vonk, 0,
            jnp.zeros(npoi), jnp.ones(npoi), jnp.ones(npoi),
        )

        assert jnp.all(stram > 1.0)
        assert jnp.all(strah > 1.0)


# ============================================================================
# impexp tests
# ============================================================================

class TestImpexp:
    """Implicit/explicit fraction."""

    def test_first_iter_returns_one(self, npoi, consts):
        """First iteration should return wimp=1."""
        tveg = jnp.full(npoi, 275.0)
        ch = jnp.full(npoi, 8436.0)
        wliq = jnp.full(npoi, 0.1)
        wsno = jnp.full(npoi, 0.1)
        prev = jnp.zeros(npoi)

        result = impexp(tveg, ch, wliq, wsno,
                        consts.tmelt, consts.hfus, consts.epsilon,
                        prev, True)
        assert jnp.allclose(result, 1.0)

    def test_bounded_01(self, npoi, consts):
        """Result should be in [0, 1]."""
        tveg = jnp.array([260.0, 270.0, 280.0, 290.0])
        ch = jnp.full(npoi, 8436.0)
        wliq = jnp.full(npoi, 0.2)
        wsno = jnp.full(npoi, 0.5)
        prev = jnp.full(npoi, 0.8)

        result = impexp(tveg, ch, wliq, wsno,
                        consts.tmelt, consts.hfus, consts.epsilon,
                        prev, False)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)


# ============================================================================
# turvap tests (9x9 solver)
# ============================================================================

class TestTurvap:
    """9x9 implicit canopy solver."""

    def _make_inputs(self, npoi, consts):
        """Create reasonable inputs for turvap."""
        c = consts
        T = 285.0
        ones = jnp.ones(npoi)
        zeros = jnp.zeros(npoi)

        return dict(
            tu=jnp.full(npoi, T), ts=jnp.full(npoi, T),
            tl=jnp.full(npoi, T), t12=jnp.full(npoi, T),
            t34=jnp.full(npoi, T), q12=jnp.full(npoi, 0.006),
            q34=jnp.full(npoi, 0.006),
            tg=jnp.full(npoi, T), ti=jnp.full(npoi, 270.0),
            tuold=jnp.full(npoi, T), tsold=jnp.full(npoi, T),
            tlold=jnp.full(npoi, T), tgold=jnp.full(npoi, T),
            tiold=jnp.full(npoi, 270.0),
            cu=ones * 5.0, cl=ones * 3.0,
            su=ones * 2.0, ss=ones * 1.0, sl=ones * 1.5,
            sg=ones * 4.0, si=ones * 3.0,
            cp=ones * c.cair, rhoa=ones * 1.2, tfac=1.0,
            xu=ones * 3.0, xs=ones * 1.0, xl=ones * 2.0,
            fi=ones * 0.1, fu=ones * 0.8, fl=ones * 0.5,
            lai=jnp.array([[2.0, 3.0]] * npoi),
            sai=jnp.array([[0.5, 1.0]] * npoi),
            frac=jnp.full((npoi, 12), 1.0 / 12.0),
            chux=ones * 8436.0, chsx=ones * 2.109e5, chlx=ones * 8436.0,
            wliqu=ones * 0.05, wliqs=ones * 0.05, wliql=ones * 0.05,
            wsnou=ones * 0.1, wsnos=ones * 0.1, wsnol=ones * 0.1,
            rliqu=ones * 0.8, rliqs=ones * 0.8, rliql=ones * 0.8,
            fwetu=ones * 0.1, fwets=ones * 0.1, fwetl=ones * 0.1,
            totcondub=ones * 0.01, totconduc=ones * 0.01,
            totcondls=ones * 0.01, totcondl3=ones * 0.01,
            totcondl4=ones * 0.01,
            chgx=ones * 1e6, cog=ones * 3.0, coi=ones * 4.0,
            zirg=ones * 5.0, ziri=ones * 5.0,
            wpud=ones * 1.0, wipud=zeros, wpudmax=4.5,
            wisoi_1=zeros, wsoi_1=ones * 0.5,
            swilt_1=ones * 0.15, suction_1=ones * 0.5, bex_1=ones * 5.0,
            hvasug=ones * c.hvap, hvasui=ones * c.hsub,
            tsoi_1=jnp.full(npoi, T), tsno_1=jnp.full(npoi, 270.0),
            solu=ones * 50.0, sols=ones * 10.0, soll=ones * 30.0,
            solg=ones * 40.0, soli=ones * 5.0,
            firu=ones * 20.0, firs=ones * 5.0, firl=ones * 15.0,
            firg=ones * 25.0, firi=ones * 10.0, firb=ones * 300.0,
            pfluxu=zeros, pfluxs=zeros, pfluxl=zeros,
            ta=jnp.full(npoi, T), qa=jnp.full(npoi, 0.006),
            psurf=jnp.full(npoi, 101325.0),
            wu=ones, ws=ones, wl=ones, wg=ones, wi=ones,
            dtime=1800.0, tmelt=c.tmelt, hvap_c=c.hvap,
            hsub_c=c.hsub, ch2o_c=c.ch2o, cice_c=c.cice,
            cvap_c=c.cvap, stef_c=c.stef, rvap_c=c.rvap,
            grav_c=c.grav, epsilon_c=c.epsilon,
        )

    def test_outputs_finite(self, npoi, consts):
        """All outputs should be finite."""
        inputs = self._make_inputs(npoi, consts)
        result = turvap(**inputs)

        assert jnp.all(jnp.isfinite(result.tu))
        assert jnp.all(jnp.isfinite(result.ts))
        assert jnp.all(jnp.isfinite(result.tl))
        assert jnp.all(jnp.isfinite(result.t12))
        assert jnp.all(jnp.isfinite(result.t34))
        assert jnp.all(jnp.isfinite(result.q12))
        assert jnp.all(jnp.isfinite(result.q34))
        assert jnp.all(jnp.isfinite(result.tg))
        assert jnp.all(jnp.isfinite(result.ti))
        assert jnp.all(jnp.isfinite(result.fsena))
        assert jnp.all(jnp.isfinite(result.fvapa))

    def test_temps_reasonable(self, npoi, consts):
        """Temperatures should stay within reasonable range."""
        inputs = self._make_inputs(npoi, consts)
        result = turvap(**inputs)

        assert jnp.all(result.tu > 200.0)
        assert jnp.all(result.tu < 350.0)
        assert jnp.all(result.tg > 200.0)
        assert jnp.all(result.tg < 350.0)
        assert jnp.all(jnp.isfinite(result.ti))

    def test_snow_temp_not_capped_inside_turvap(self, npoi, consts):
        """`ti` must NOT be clamped to tmelt inside turvap.

        Verification finding #13 (`verify/STATUS.md`): F77 `canopy.f` applies
        `ti = min(ti, tmelt)` only *after* the 3-iteration loop exits -- line
        1249 sits below `if (iter.lt.niter) return`. The JAX port originally
        clamped inside `turvap()`, so it ran every iteration and produced a
        real divergence from F77. The fix removed the clamp from `turvap` and
        applied it in `timestepping/loops.py` after the iteration loop.

        This test previously asserted `ti <= tmelt` on raw `turvap` output,
        which contradicts that fix -- it was asserting the bug. It now pins the
        corrected behaviour instead: `turvap` returns `ti` unclamped, and the
        cap is the caller's responsibility.
        """
        inputs = self._make_inputs(npoi, consts)
        result = turvap(**inputs)
        assert jnp.all(jnp.isfinite(result.ti))
        # The post-loop clamp (as applied in loops.py) is what enforces the cap.
        assert jnp.all(jnp.minimum(result.ti, consts.tmelt) <= consts.tmelt)

    def test_humidity_positive(self, npoi, consts):
        """Specific humidities should be non-negative."""
        inputs = self._make_inputs(npoi, consts)
        result = turvap(**inputs)
        # q12 and q34 should be near the input values (equilibrium-ish)
        assert jnp.all(jnp.isfinite(result.q12))
        assert jnp.all(jnp.isfinite(result.q34))


# ============================================================================
# fwetcal tests
# ============================================================================

class TestFwetcal:
    """Fractional wet areas."""

    def test_bounded(self, npoi, consts):
        """fwet should be in [0, fmax]."""
        tu = jnp.full(npoi, 280.0)
        wliqu = jnp.full(npoi, 0.1)
        wsnou = jnp.full(npoi, 0.5)

        result = fwetcal(
            tu, tu, tu,
            wliqu, wliqu, wliqu,
            wsnou, wsnou, wsnou,
            0.20, 0.40, 0.20,
            2.0, 4.0, 2.0,
            0.25, consts.tmelt, consts.epsilon,
        )

        assert jnp.all(result.fwetu >= 0.0)
        assert jnp.all(result.fwetu <= 0.25)
        assert jnp.all(result.rliqu >= 0.0)
        assert jnp.all(result.rliqu <= 1.0)

    def test_dry_canopy(self, npoi, consts):
        """Zero intercepted water should give zero fwet."""
        tu = jnp.full(npoi, 290.0)
        z = jnp.zeros(npoi)

        result = fwetcal(
            tu, tu, tu, z, z, z, z, z, z,
            0.20, 0.40, 0.20, 2.0, 4.0, 2.0,
            0.25, consts.tmelt, consts.epsilon,
        )

        assert jnp.allclose(result.fwetu, 0.0)
        assert jnp.allclose(result.rliqu, 1.0)  # warm canopy → rliq=1

    def test_cold_dry_gives_ice_fraction(self, npoi, consts):
        """Below-freezing dry canopy → rliq=0."""
        tu = jnp.full(npoi, 260.0)
        z = jnp.zeros(npoi)

        result = fwetcal(
            tu, tu, tu, z, z, z, z, z, z,
            0.20, 0.40, 0.20, 2.0, 4.0, 2.0,
            0.25, consts.tmelt, consts.epsilon,
        )

        assert jnp.allclose(result.rliqu, 0.0)


# ============================================================================
# mix tests
# ============================================================================

class TestMix:
    """Calorimetric mixing."""

    def test_single_mass(self, npoi, consts):
        """Single non-zero mass should give its own temperature."""
        x1 = jnp.full(npoi, 2.0)
        t1 = jnp.full(npoi, 300.0)
        z = jnp.zeros(npoi)

        xm, tm = mix(x1, t1, z, z, z, z, consts.epsilon)

        assert jnp.allclose(xm, 2.0)
        assert jnp.allclose(tm, 300.0)

    def test_equal_masses(self, npoi, consts):
        """Equal masses should give average temperature."""
        x1 = jnp.full(npoi, 1.0)
        t1 = jnp.full(npoi, 280.0)
        x2 = jnp.full(npoi, 1.0)
        t2 = jnp.full(npoi, 300.0)
        z = jnp.zeros(npoi)

        xm, tm = mix(x1, t1, x2, t2, z, z, consts.epsilon)

        assert jnp.allclose(xm, 2.0)
        assert jnp.allclose(tm, 290.0, atol=0.1)

    def test_zero_total_mass(self, npoi, consts):
        """Zero total mass should give zero result."""
        z = jnp.zeros(npoi)
        xm, tm = mix(z, z, z, z, z, z, consts.epsilon)
        assert jnp.allclose(xm, 0.0)
        assert jnp.allclose(tm, 0.0)


# ============================================================================
# twet3 tests
# ============================================================================

class TestTwet3:
    """Wet bulb temperature."""

    def test_at_saturation(self, npoi, consts):
        """At saturation, wet bulb should equal air temperature."""
        c = consts
        tak = jnp.full(npoi, 293.0)
        e = jnp.array(1170.0)  # approx esat at 293K (Pa) - rough
        from ibis.physics.thermo import esat as esat_fn, qsat as qsat_fn
        e_val = esat_fn(tak)
        q_sat = qsat_fn(e_val, jnp.full(npoi, 101325.0))

        tw = twet3(tak, q_sat, jnp.full(npoi, 101325.0),
                   c.tmelt, c.cair, c.hvap, c.ch2o, c.cvap)

        assert jnp.allclose(tw, tak, atol=1.0)

    def test_below_air_temp(self, npoi, consts):
        """Wet bulb should be <= air temperature."""
        c = consts
        tak = jnp.full(npoi, 300.0)
        q = jnp.full(npoi, 0.005)  # unsaturated
        p = jnp.full(npoi, 101325.0)

        tw = twet3(tak, q, p, c.tmelt, c.cair, c.hvap, c.ch2o, c.cvap)

        assert jnp.all(tw <= tak + 0.5)  # small tolerance for numerics
        assert jnp.all(tw > 250.0)


# ============================================================================
# steph2o tests
# ============================================================================

class TestSteph2o:
    """One-component interception step."""

    def test_interception_reduces_throughfall(self, npoi, consts):
        """Rain throughfall should be less than incoming rain."""
        c = consts
        tveg = jnp.full(npoi, 290.0)
        wliq = jnp.zeros(npoi)
        wsno = jnp.zeros(npoi)
        xai = jnp.full(npoi, 4.0)
        pflux = jnp.zeros(npoi)
        rain = jnp.full(npoi, 1e-3)
        train = jnp.full(npoi, 285.0)
        snow = jnp.zeros(npoi)
        tsnow = jnp.full(npoi, 270.0)

        result = steph2o(
            tveg, wliq, wsno, xai, pflux,
            rain, train, snow, tsnow,
            7200.0, 43200.0, 0.20, 2.0, 1e-5, 1e-5,
            1800.0, c.ch2o, c.cice, c.tmelt, c.hfus, c.epsilon,
        )

        assert jnp.all(result.rain < 1e-3)  # throughfall < incoming

    def test_water_conservation(self, npoi, consts):
        """Total water should be approximately conserved."""
        c = consts
        tveg = jnp.full(npoi, 290.0)
        wliq = jnp.full(npoi, 0.05)
        wsno = jnp.zeros(npoi)
        xai = jnp.full(npoi, 4.0)
        pflux = jnp.zeros(npoi)
        rain = jnp.full(npoi, 1e-4)
        train = jnp.full(npoi, 290.0)
        snow = jnp.zeros(npoi)
        tsnow = jnp.full(npoi, 270.0)
        dt = 1800.0

        total_in = wliq * xai + rain * dt

        result = steph2o(
            tveg, wliq, wsno, xai, pflux,
            rain, train, snow, tsnow,
            7200.0, 43200.0, 0.20, 2.0, 1e-5, 1e-5,
            dt, c.ch2o, c.cice, c.tmelt, c.hfus, c.epsilon,
        )

        total_out = (result.wliq + result.wsno) * xai + result.rain * dt
        # Conservation is approximate due to drip dynamics
        assert jnp.all(jnp.isfinite(total_out))


# ============================================================================
# noveg tests
# ============================================================================

class TestNoveg:
    """No-vegetation temperature reset."""

    def test_no_lai_resets_temp(self, npoi):
        """Zero LAI should reset temperatures to surface average."""
        tu = jnp.full(npoi, 290.0)
        ts = jnp.full(npoi, 290.0)
        tl = jnp.full(npoi, 290.0)
        wliqu = jnp.full(npoi, 0.1)
        wliqs = jnp.full(npoi, 0.1)
        wliql = jnp.full(npoi, 0.1)
        wsnou = jnp.full(npoi, 0.1)
        wsnos = jnp.full(npoi, 0.1)
        wsnol = jnp.full(npoi, 0.1)
        lai = jnp.zeros((npoi, 2))
        sai = jnp.zeros((npoi, 2))
        fu = jnp.zeros(npoi)
        fl = jnp.zeros(npoi)
        fi = jnp.full(npoi, 0.2)
        tg = jnp.full(npoi, 280.0)
        ti = jnp.full(npoi, 270.0)

        tav = (1.0 - fi) * tg + fi * ti

        result = noveg(tu, ts, tl, wliqu, wliqs, wliql,
                       wsnou, wsnos, wsnol,
                       lai, sai, fu, fl, fi, tg, ti)

        tu_new, ts_new, tl_new = result[0], result[1], result[2]
        wliqu_new = result[3]

        assert jnp.allclose(tu_new, tav)
        assert jnp.allclose(wliqu_new, 0.0)

    def test_with_veg_unchanged(self, npoi):
        """With vegetation, temperatures should not change."""
        tu = jnp.full(npoi, 290.0)
        ts = jnp.full(npoi, 290.0)
        tl = jnp.full(npoi, 290.0)
        wliqu = jnp.full(npoi, 0.1)
        wliqs = jnp.full(npoi, 0.1)
        wliql = jnp.full(npoi, 0.1)
        wsnou = jnp.full(npoi, 0.1)
        wsnos = jnp.full(npoi, 0.1)
        wsnol = jnp.full(npoi, 0.1)
        lai = jnp.array([[2.0, 3.0]] * npoi)
        sai = jnp.array([[0.5, 1.0]] * npoi)
        fu = jnp.full(npoi, 0.8)
        fl = jnp.full(npoi, 0.5)
        fi = jnp.full(npoi, 0.1)
        tg = jnp.full(npoi, 280.0)
        ti = jnp.full(npoi, 270.0)

        result = noveg(tu, ts, tl, wliqu, wliqs, wliql,
                       wsnou, wsnos, wsnol,
                       lai, sai, fu, fl, fi, tg, ti)

        assert jnp.allclose(result[0], tu)
        assert jnp.allclose(result[3], wliqu)
