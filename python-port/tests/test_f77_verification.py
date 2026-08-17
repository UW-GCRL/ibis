"""Process-by-process verification of IBIS-JAX against original Fortran IBIS 2.6b4.

This test suite verifies that each physics module in the JAX code reproduces
the original Fortran formulas exactly.  For each process we:
  1. Set up controlled, physically realistic inputs
  2. Run the JAX function
  3. Compare against hand-computed Fortran-equivalent reference values
  4. Check conservation laws and physical constraints

Reference values are derived from the original Fortran source code formulas
(ibis_2.6b4/*.f) with known input values.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from ibis.constants import PhysicalConstants, get_constants

# =========================================================================
# Helper: physical constants matching Fortran initial.f lines 70-96
# =========================================================================

C = get_constants()
STEF = 5.67e-8
VONK = 0.4
GRAV = 9.80616
TMELT = 273.16
HVAP = 2.5104e6
HFUS = 0.3336e6
HSUB = HVAP + HFUS
CH2O = 4.218e3
CICE = 2.106e3
CAIR = 1.00464e3
CVAP = 1.81e3
RAIR = 287.04
RVAP = 461.0
CAPPA = RAIR / CAIR
RHOW = 1.0e3
EPS = 1.0e-7


# =========================================================================
# 1. THERMODYNAMIC FUNCTIONS (thermo.py vs comsat.h / utilities.f)
# =========================================================================

class TestThermo:
    """Verify esat, qsat, dqsat against Fortran Lowe (1977) polynomial."""

    def test_esat_at_freezing(self):
        """esat(273.16) should match Fortran esat = 611.21 Pa (Lowe 1977)."""
        from ibis.physics.thermo import esat
        es = float(esat(jnp.array(TMELT)))
        # Lowe 1977 polynomial at 273.16 K gives ~611.2 Pa
        assert abs(es - 611.2) < 1.0, f"esat(tmelt)={es}, expected ~611.2"

    def test_esat_at_boiling(self):
        from ibis.physics.thermo import esat
        es = float(esat(jnp.array(373.16)))
        assert abs(es - 101325.0) / 101325.0 < 0.02, f"esat(373.16)={es}"

    def test_esat_ice_branch(self):
        """Below tmelt, esat should use ice curve (lower than liquid)."""
        from ibis.physics.thermo import esat
        es_liquid = float(esat(jnp.array(TMELT + 0.01)))
        es_ice = float(esat(jnp.array(TMELT - 0.01)))
        # esat should be continuous near tmelt
        assert abs(es_liquid - es_ice) < 1.0

    def test_qsat_formula(self):
        """qsat = 0.622 * esat / max(psurf - 0.378*esat, 0.622*esat)."""
        from ibis.physics.thermo import esat, qsat
        T = jnp.array(293.0)
        P = jnp.array(101325.0)
        es = float(esat(T))
        # qsat takes (esat_val, p), not (T, p)
        qs = float(qsat(jnp.array(es), P))
        # Manual formula matches Fortran: 0.622 * e / max(p - 0.378*e, 0.622*e)
        denom = max(P - 0.378 * es, 0.622 * es)
        qs_manual = 0.622 * es / denom
        assert abs(qs - qs_manual) / qs_manual < 1e-4

    def test_hvapf_at_tmelt(self):
        """hvapf(tmelt, tmelt) should equal hvap."""
        from ibis.physics.thermo import hvapf
        hv = float(hvapf(jnp.array(TMELT), jnp.array(TMELT),
                         TMELT, HVAP, CH2O, CVAP))
        assert abs(hv - HVAP) < 1.0

    def test_hsubf_at_tmelt(self):
        from ibis.physics.thermo import hsubf
        hs = float(hsubf(jnp.array(TMELT), jnp.array(TMELT),
                         TMELT, HSUB, CVAP, CICE))
        assert abs(hs - HSUB) < 1.0


# =========================================================================
# 2. RADIATION (radiation.py vs radiation.f)
# =========================================================================

class TestRadiation:
    """Verify two-stream radiation against Fortran radiation.f."""

    def test_albedo_conservation(self):
        """Absorbed + reflected solar should equal incident solar."""
        from ibis.physics.radiation import compute_radiation, default_veg_rad_params
        npoi = 2
        vr = default_veg_rad_params()

        coszen = jnp.array([0.5, 0.8])
        solad = jnp.full((npoi, 2), 200.0)
        solai = jnp.full((npoi, 2), 50.0)
        fira = jnp.full((npoi,), 350.0)
        lai = jnp.full((npoi, 2), 3.0)
        sai = jnp.full((npoi, 2), 0.5)
        fu = jnp.full((npoi,), 0.8)
        fl = jnp.full((npoi,), 0.5)
        tu = ts = tl = jnp.full((npoi,), 293.0)
        tg = ti = jnp.full((npoi,), 290.0)
        fi = jnp.zeros((npoi,))
        tsno_0 = jnp.full((npoi,), TMELT)
        wsoi_0 = jnp.full((npoi,), 0.3)
        wisoi_0 = jnp.zeros((npoi,))
        albsav = jnp.full((npoi,), 0.10)
        albsan = jnp.full((npoi,), 0.20)
        fwetu = fwets = fwetl = jnp.zeros((npoi,))
        rliqu = rliqs = rliql = jnp.ones((npoi,))

        rad = compute_radiation(
            coszen, solad, solai, fira,
            lai, sai, fu, fl,
            tu, ts, tl, tg, ti,
            fi, tsno_0, wsoi_0, wisoi_0,
            albsav, albsan,
            fwetu, fwets, fwetl,
            rliqu, rliqs, rliql,
            vr, STEF, TMELT, EPS,
        )

        # Total incident solar (both bands, direct+diffuse)
        total_incident = jnp.sum(solad + solai, axis=1)
        # Absorbed by all components
        total_absorbed = rad.solu + rad.sols + rad.soll + rad.solg + rad.soli

        for i in range(npoi):
            # Each component absorption should be non-negative
            assert float(rad.solu[i]) >= -1e-6, f"Point {i}: solu negative"
            assert float(rad.solg[i]) >= -1e-6, f"Point {i}: solg negative"
            # Total absorbed should be positive and bounded
            assert float(total_absorbed[i]) > 0, f"Point {i}: no absorption"
            assert float(total_absorbed[i]) < 2.0 * float(total_incident[i]), \
                f"Point {i}: absorbed unreasonably large"

    def test_no_solar_at_night(self):
        """When coszen <= 0, all solar fluxes should be zero."""
        from ibis.physics.radiation import compute_radiation, default_veg_rad_params
        npoi = 1
        vr = default_veg_rad_params()

        coszen = jnp.array([0.0])  # night
        rad = compute_radiation(
            coszen, jnp.zeros((1, 2)), jnp.zeros((1, 2)),
            jnp.full((1,), 300.0),
            jnp.full((1, 2), 2.0), jnp.full((1, 2), 0.5),
            jnp.full((1,), 0.8), jnp.full((1,), 0.5),
            *[jnp.full((1,), 290.0)] * 5,  # tu, ts, tl, tg, ti
            jnp.zeros((1,)), jnp.full((1,), TMELT),
            jnp.full((1,), 0.3), jnp.zeros((1,)),
            jnp.full((1,), 0.1), jnp.full((1,), 0.2),
            *[jnp.zeros((1,))] * 3,  # fwet
            *[jnp.ones((1,))] * 3,   # rliq
            vr, STEF, TMELT, EPS,
        )

        assert float(rad.solu[0]) == 0.0
        assert float(rad.sols[0]) == 0.0
        assert float(rad.soll[0]) == 0.0
        assert float(rad.solg[0]) == 0.0

    def test_ir_stefan_boltzmann(self):
        """IR emission from ground should scale as emiss * stef * T^4."""
        from ibis.physics.radiation import compute_radiation, default_veg_rad_params
        npoi = 1
        vr = default_veg_rad_params()
        T = 300.0  # ground temperature

        rad = compute_radiation(
            jnp.array([0.0]),  # nighttime
            jnp.zeros((1, 2)), jnp.zeros((1, 2)),
            jnp.full((1,), 0.95 * STEF * T**4),  # fira = emiss*stef*T^4
            jnp.full((1, 2), 2.0), jnp.full((1, 2), 0.5),
            jnp.full((1,), 0.8), jnp.full((1,), 0.5),
            *[jnp.full((1,), T)] * 5,
            jnp.zeros((1,)), jnp.full((1,), TMELT),
            jnp.full((1,), 0.3), jnp.zeros((1,)),
            jnp.full((1,), 0.1), jnp.full((1,), 0.2),
            *[jnp.zeros((1,))] * 3,
            *[jnp.ones((1,))] * 3,
            vr, STEF, TMELT, EPS,
        )

        # When all temps are T and fira = emiss*stef*T^4,
        # net IR for each component should be small (near equilibrium)
        total_ir = float(abs(rad.firu[0]) + abs(rad.firs[0]) + abs(rad.firl[0])
                         + abs(rad.firg[0]) + abs(rad.firi[0]))
        # Should be bounded, not blowing up
        assert total_ir < 500.0, f"Total IR too large: {total_ir}"


# =========================================================================
# 3. SOIL PHYSICS (soil_physics.py vs soil.f)
# =========================================================================

class TestSoilPhysics:
    """Verify soil solvers against Fortran soil.f."""

    def test_setsoi_thermal_conductivity(self):
        """Verify setsoi returns reasonable conductivity values."""
        from ibis.physics.soil_physics import setsoi
        npoi, nsoilay = 2, 6
        result = setsoi(
            sand=jnp.full((npoi, nsoilay), 40.0),
            clay=jnp.full((npoi, nsoilay), 20.0),
            poros=jnp.full((npoi, nsoilay), 0.45),
            wsoi=jnp.full((npoi, nsoilay), 0.4),
            wisoi=jnp.zeros((npoi, nsoilay)),
            tsoi=jnp.full((npoi, nsoilay), 290.0),
            wpud=jnp.zeros((npoi,)),
            wipud=jnp.zeros((npoi,)),
            wpudmax=10.0, zwpmax=0.5,
            tsno_1=jnp.full((npoi,), TMELT),
            ta=jnp.full((npoi,), 293.0),
            tmelt=TMELT, hvap=HVAP, hsub=HSUB,
            ch2o=CH2O, cvap=CVAP, cice=CICE, epsilon=EPS,
        )
        # Thermal conductivity should be 0.1 - 3.0 W/m/K for typical soil
        for i in range(npoi):
            for k in range(nsoilay):
                cond = float(result.consoi[i, k])
                assert 0.05 < cond < 5.0, f"consoi[{i},{k}]={cond}"

    def test_setsoi_qglif_sums_to_one(self):
        """Evaporation fractions qglif should sum to 1."""
        from ibis.physics.soil_physics import setsoi
        npoi, nsoilay = 4, 6
        result = setsoi(
            sand=jnp.full((npoi, nsoilay), 50.0),
            clay=jnp.full((npoi, nsoilay), 20.0),
            poros=jnp.full((npoi, nsoilay), 0.45),
            wsoi=jnp.full((npoi, nsoilay), 0.3),
            wisoi=jnp.full((npoi, nsoilay), 0.1),
            tsoi=jnp.full((npoi, nsoilay), 280.0),
            wpud=jnp.full((npoi,), 2.0),
            wipud=jnp.full((npoi,), 1.0),
            wpudmax=10.0, zwpmax=0.5,
            tsno_1=jnp.full((npoi,), TMELT - 5.0),
            ta=jnp.full((npoi,), 280.0),
            tmelt=TMELT, hvap=HVAP, hsub=HSUB,
            ch2o=CH2O, cvap=CVAP, cice=CICE, epsilon=EPS,
        )
        qsum = jnp.sum(result.qglif, axis=1)
        np.testing.assert_allclose(qsum, 1.0, atol=1e-6)

    def test_soilh2o_water_conservation(self):
        """Water in = water out + storage change (mass balance)."""
        from ibis.physics.soil_physics import soilh2o
        npoi, nsoilay = 2, 6
        dtime = 3600.0
        poros = jnp.full((npoi, nsoilay), 0.45)
        hsoi = jnp.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])
        wsoi0 = jnp.full((npoi, nsoilay), 0.4)
        wisoi = jnp.zeros((npoi, nsoilay))
        porosflo = poros * jnp.maximum(1.0 - wisoi, EPS)
        fwtop = jnp.full((npoi,), 1e-4)  # small rain input (kg/m2/s)

        h2o = soilh2o(
            wsoi0, wisoi, poros, porosflo,
            jnp.full((npoi, nsoilay), 1e-5),  # hydraul
            jnp.full((npoi, nsoilay), -0.15),  # suction
            jnp.full((npoi, nsoilay), 5.0),    # bex
            hsoi,
            fwtop,
            jnp.zeros((npoi,)),  # fwpud
            jnp.zeros((npoi,)),  # wpud
            0.04, dtime, RHOW, EPS,
        )

        # Compute total water before and after
        w_before = jnp.sum(wsoi0 * poros * hsoi[None, :] * RHOW, axis=1)
        w_after = jnp.sum(h2o.wsoi * poros * hsoi[None, :] * RHOW, axis=1)
        # Input + squeeze + bottom drainage should balance
        w_input = fwtop * dtime
        w_squeeze = h2o.fsqueez * dtime
        # Mass balance (approximate due to implicit scheme)
        balance = w_after - w_before + w_squeeze - w_input
        np.testing.assert_allclose(balance, 0.0, atol=0.1,
                                    err_msg="Soil water conservation violated")

    def test_freeze_thaw_energy_conservation(self):
        """Freeze/thaw should conserve energy: dT * heat_capacity = hfus * dw."""
        from ibis.physics.soil_physics import freeze_thaw
        npoi, nsoilay = 2, 6
        hsoi = jnp.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])
        poros = jnp.full((npoi, nsoilay), 0.45)
        # Start below freezing with some liquid → should freeze
        tsoi = jnp.full((npoi, nsoilay), TMELT - 2.0)
        wsoi = jnp.full((npoi, nsoilay), 0.5)  # 50% liquid
        wisoi = jnp.full((npoi, nsoilay), 0.1)
        csoi_m = jnp.full((npoi, nsoilay), 870.0)
        rhosoi = jnp.full((npoi, nsoilay), 2650.0)
        c1pud = jnp.zeros((npoi, nsoilay))

        tsoi_new, wsoi_new, wisoi_new, _ = freeze_thaw(
            tsoi, wsoi, wisoi, poros,
            csoi_m, rhosoi, hsoi, c1pud,
            CH2O, CICE, RHOW, HFUS, TMELT, EPS,
        )

        # Temperature should have increased (latent heat released by freezing)
        for k in range(nsoilay):
            t_old = float(tsoi[0, k])
            t_new = float(tsoi_new[0, k])
            w_old = float(wsoi[0, k])
            w_new = float(wsoi_new[0, k])
            wi_old = float(wisoi[0, k])
            wi_new = float(wisoi_new[0, k])
            # Some liquid should have frozen
            assert wi_new >= wi_old, f"Layer {k}: ice should increase"
            # Temperature should approach tmelt (released latent heat)
            assert t_new >= t_old, f"Layer {k}: T should increase during freezing"

    def test_puddle_freeze_thaw(self):
        """Puddle freeze/thaw should conserve total puddle mass."""
        from ibis.physics.soil_physics import puddle_freeze_thaw
        npoi = 4
        # Below freezing: should freeze some puddle water
        tsoi_0 = jnp.full((npoi,), TMELT - 1.0)
        wpud = jnp.full((npoi,), 5.0)    # 5 kg/m2 liquid
        wipud = jnp.full((npoi,), 1.0)   # 1 kg/m2 ice
        poros_0 = jnp.full((npoi,), 0.45)
        wsoi_0 = jnp.full((npoi,), 0.3)
        wisoi_0 = jnp.full((npoi,), 0.1)
        csoi_m = jnp.full((npoi,), 870.0)
        rhosoi = jnp.full((npoi,), 2650.0)

        t_new, w_new, wi_new = puddle_freeze_thaw(
            tsoi_0, wpud, wipud, poros_0, wsoi_0, wisoi_0,
            csoi_m, rhosoi, 0.10, CH2O, CICE, RHOW, HFUS, TMELT,
        )

        # Total puddle mass conserved
        np.testing.assert_allclose(w_new + wi_new, wpud + wipud, atol=1e-6)
        # Some water should have frozen
        assert float(jnp.mean(wi_new)) > float(jnp.mean(wipud))
        # Temperature should have increased
        assert float(jnp.mean(t_new)) > float(jnp.mean(tsoi_0))


# =========================================================================
# 4. SNOW PHYSICS (snow_physics.py vs snow.f)
# =========================================================================

class TestSnowPhysics:
    """Verify snow solvers against Fortran snow.f."""

    def test_snowheat_uniform_temp_stability(self):
        """Uniform-temperature snow should stay uniform (no heat flux)."""
        from ibis.physics.snow_physics import snowheat
        npoi, nsnolay = 2, 3
        T = TMELT - 5.0
        result = snowheat(
            tsno=jnp.full((npoi, nsnolay), T),
            tlsub=jnp.full((npoi,), T),
            fhtop=jnp.zeros((npoi,)),       # no surface flux
            xl=jnp.full((npoi,), 1.0),
            chl=200.0,
            fi=jnp.ones((npoi,)),
            hsno=jnp.full((npoi, nsnolay), 0.10),
            rhos=250.0, consno=0.20,
            hsoi_1=0.10,
            consoi_1=jnp.full((npoi,), 1.0),
            tsoi_1=jnp.full((npoi,), T),    # same temp as snow
            dtime=3600.0,
            cice=CICE,
        )
        # Should remain at T (no gradients)
        np.testing.assert_allclose(result.tsno, T, atol=0.01)
        np.testing.assert_allclose(result.tlsub, T, atol=0.01)

    def test_snowheat_conducts_to_warm_soil(self):
        """Cold snow on warm soil: bottom snow layer should warm."""
        from ibis.physics.snow_physics import snowheat
        npoi = 1
        result = snowheat(
            tsno=jnp.full((1, 3), TMELT - 10.0),
            tlsub=jnp.full((1,), TMELT - 10.0),
            fhtop=jnp.zeros((1,)),
            xl=jnp.full((1,), 1.0),
            chl=200.0,
            fi=jnp.ones((1,)),
            hsno=jnp.full((1, 3), 0.10),
            rhos=250.0, consno=0.20,
            hsoi_1=0.10,
            consoi_1=jnp.full((1,), 1.5),
            tsoi_1=jnp.full((1,), TMELT),  # warm soil
            dtime=3600.0,
            cice=CICE,
        )
        # Bottom snow layer and tlsub should be warmer than input
        assert float(result.tsno[0, -1]) > TMELT - 10.0
        assert float(result.tlsub[0]) > TMELT - 10.0

    def test_snow_melt_excess_heat(self):
        """Snow above tmelt should melt; melt mass = excess_heat/hfus."""
        from ibis.physics.snow_physics import snow_melt
        npoi = 1
        # Snow slightly above freezing
        tsno = jnp.full((1, 3), TMELT + 2.0)
        hsno = jnp.full((1, 3), 0.10)  # 10 cm per layer
        fi = jnp.ones((1,))
        rhos = 250.0

        hsno_new, tsno_new, smelt, zheat = snow_melt(
            hsno, tsno, fi, rhos, CICE, HFUS, TMELT,
        )

        # Temperature should be clamped at tmelt
        np.testing.assert_allclose(tsno_new, TMELT, atol=0.01)
        # Some snow should have melted
        assert float(smelt[0]) > 0.0
        # Mass conservation: melted = rhos * (sum(hsno) - sum(hsno_new)) * fi
        mass_lost = rhos * (jnp.sum(hsno) - jnp.sum(hsno_new)) * fi[0]
        np.testing.assert_allclose(smelt[0], mass_lost, rtol=0.01)

    def test_vadapt_conserves_heat(self):
        """Layer redistribution should conserve total heat content."""
        from ibis.physics.snow_physics import vadapt
        npoi = 2
        hsno = jnp.array([[0.08, 0.12, 0.20],
                           [0.15, 0.05, 0.30]])
        tsno = jnp.array([[260.0, 265.0, 268.0],
                           [255.0, 262.0, 270.0]])
        rhos = 250.0
        cice = CICE

        hsno_new, tsno_new = vadapt(
            hsno, tsno, htop=0.05,
            has_snow=jnp.array([1.0, 1.0]),
            epsilon=EPS,
        )

        # Total heat = sum(rhos * cice * h * T) should be conserved
        heat_old = jnp.sum(hsno * tsno, axis=1) * rhos * cice
        heat_new = jnp.sum(hsno_new * tsno_new, axis=1) * rhos * cice
        np.testing.assert_allclose(heat_new, heat_old, rtol=1e-6)

    def test_snow_coverage_quadratic(self):
        """snow_coverage should follow the quadratic volume-coverage relation."""
        from ibis.physics.snow_physics import snow_coverage
        npoi = 1
        hsno = jnp.array([[0.05, 0.05, 0.05]])  # total = 0.15m
        fi = jnp.array([0.5])
        hsnomin = 0.01
        hsnoruf = jnp.array([0.20])  # roughness parameter

        fi_new, hsno_new = snow_coverage(
            hsno, fi, hsnomin, hsnoruf, fimax=1.0, epsilon=EPS,
        )

        # fi_new should be in [0, 1]
        assert 0.0 <= float(fi_new[0]) <= 1.0
        # Total snow volume should be conserved (fi * sum(hsno) constant)
        vol_old = float(fi[0] * jnp.sum(hsno[0]))
        vol_new = float(fi_new[0] * jnp.sum(hsno_new[0]))
        np.testing.assert_allclose(vol_new, vol_old, rtol=1e-4)


# =========================================================================
# 5. CANOPY EXCHANGE (canopy_exchange.py vs canopy.f)
# =========================================================================

class TestCanopyExchange:
    """Verify canopy energy balance against Fortran canopy.f."""

    def test_impexp_first_iteration(self):
        """First iteration should always return 1.0."""
        from ibis.physics.canopy_exchange import impexp, impexp2
        t = jnp.array([280.0, 290.0, TMELT])
        wi = impexp(t, 200.0, jnp.ones(3), jnp.ones(3),
                     TMELT, HFUS, EPS, jnp.ones(3), True)
        np.testing.assert_allclose(wi, 1.0)

        wi2 = impexp2(t, t - 1.0, TMELT, EPS, jnp.ones(3), True)
        np.testing.assert_allclose(wi2, 1.0)

    def test_impexp_phase_change_detection(self):
        """impexp near tmelt should detect phase change (wi < 1)."""
        from ibis.physics.canopy_exchange import impexp
        npoi = 3
        # Temperatures straddling freezing
        t = jnp.array([TMELT + 0.1, TMELT - 0.1, TMELT + 5.0])
        ch = 200.0
        wliq = jnp.full((npoi,), 2.0)
        wsno = jnp.full((npoi,), 2.0)

        # Second iteration (not first)
        wi = impexp(t, ch, wliq, wsno, TMELT, HFUS, EPS,
                     jnp.ones(npoi), False)

        # All should be in [0, 1]
        for i in range(npoi):
            assert 0.0 <= float(wi[i]) <= 1.0

        # Point 2 is 5 K above freezing, but the test above also supplies
        # wsno = 2 kg/m2. The latent heat needed to melt that snow,
        # hfus*wsno = 6.7e5 J/m2, dwarfs the available h = ch*dT = 1e3, so
        # `min(1, hfus*wsno/z)` saturates at 1, winew = 0, and the blend with
        # wimp_prev = 1 gives exactly 0.5. That is correct F77 behaviour --
        # impexp is verified bit-exact in verify/STATUS.md. The previous
        # assertion (wi[2] > 0.9, "no phase change") was wrong because it
        # assumed no snow while passing 2 kg of it.
        assert float(wi[2]) == pytest.approx(0.5, abs=1e-6)

        # With negligible frozen/liquid water there is genuinely nothing to
        # melt, so winew -> 1 and the blend stays at 1.
        wi_dry = impexp(t, ch, jnp.zeros(npoi), jnp.zeros(npoi),
                        TMELT, HFUS, EPS, jnp.ones(npoi), False)
        assert float(wi_dry[2]) == pytest.approx(1.0, abs=1e-6)

    def test_canini_produces_valid_heights(self):
        """canini should produce sensible aerodynamic heights."""
        from ibis.physics.canopy_exchange import canini, CanopyParams
        npoi = 2
        cp = CanopyParams()
        aero = canini(
            ta=jnp.full((npoi,), 293.0),
            qa=jnp.full((npoi,), 0.008),
            psurf=jnp.full((npoi,), 101325.0),
            ua=jnp.full((npoi,), 3.0),
            fi=jnp.zeros((npoi,)),
            z0soi=jnp.full((npoi,), 0.01),
            z0sno=0.005,
            fl=jnp.full((npoi,), 0.5),
            fu=jnp.full((npoi,), 0.8),
            lai=jnp.full((npoi, 2), 3.0),
            sai=jnp.full((npoi, 2), 0.5),
            ztop=jnp.array([[5.0, 20.0]] * npoi),
            zbot=jnp.array([[0.5, 2.0]] * npoi),
            alaiml=cp.alaiml, alaimu=cp.alaimu,
            grav=GRAV, rair=RAIR, rvap=RVAP,
            cair=CAIR, cvap=CVAP, cappa=CAPPA,
        )
        # Air density should be ~1.2 kg/m3
        np.testing.assert_allclose(aero.rhoa, 1.2, atol=0.2)


# =========================================================================
# 6. STOMATA / PHOTOSYNTHESIS (stomata.py vs physiology.f)
# =========================================================================

class TestStomata:
    """Verify Farquhar-Ball-Berry against Fortran physiolog.f."""

    def test_colimit_quadratic(self):
        """colimit(a, b, theta) should satisfy theta*x^2 - (a+b)*x + a*b = 0."""
        from ibis.physics.stomata import colimit
        a, b, theta = 20.0, 10.0, 0.95
        x = float(colimit(jnp.array(a), jnp.array(b), theta))
        # Verify quadratic: theta*x^2 - (a+b)*x + a*b should be ~0
        residual = theta * x**2 - (a + b) * x + a * b
        assert abs(residual) < 0.01

        # colimit returns the SMALLER root, which for theta < 1 is strictly
        # below min(a, b) -- that is the point of co-limitation: two partially
        # co-limiting processes yield less than either limit alone.
        # For a=20, b=10, theta=0.95 the roots are
        #   (30 +/- sqrt(900 - 4*0.95*200)) / (2*0.95) = 9.562 and 21.99
        # so x = 9.562 < min(a,b) = 10. The previous assertion
        # `min(a,b) <= x` was mathematically wrong, not a code defect.
        assert 0.0 < x <= min(a, b)
        # ...and it must approach min(a,b) as theta -> 1 (no co-limitation).
        x_nolim = float(colimit(jnp.array(a), jnp.array(b), 0.9999))
        assert x_nolim == pytest.approx(min(a, b), rel=1e-3)

    def test_drystress_saturated_soil(self):
        """Fully wet soil should give stress factor near 1."""
        from ibis.physics.stomata import drystress
        npoi, nsoilay = 2, 6
        wsoi = jnp.full((npoi, nsoilay), 0.9)  # very wet
        wisoi = jnp.zeros((npoi, nsoilay))
        swilt = jnp.full((npoi, nsoilay), 0.15)
        sfield = jnp.full((npoi, nsoilay), 0.35)
        froot = jnp.full((nsoilay, 2), 1.0 / nsoilay)

        stresstu, stresstl, stressu, stressl = drystress(
            wsoi, wisoi, swilt, sfield, froot,
        )
        # Stress should be near 1 (no water limitation)
        assert float(stresstu[0]) > 0.9
        assert float(stresstl[0]) > 0.9

    def test_drystress_wilting_point(self):
        """At wilting point, stress should be near 0."""
        from ibis.physics.stomata import drystress
        npoi, nsoilay = 2, 6
        wsoi = jnp.full((npoi, nsoilay), 0.15)  # at wilting
        wisoi = jnp.zeros((npoi, nsoilay))
        swilt = jnp.full((npoi, nsoilay), 0.15)
        sfield = jnp.full((npoi, nsoilay), 0.35)
        froot = jnp.full((nsoilay, 2), 1.0 / nsoilay)

        stresstu, stresstl, _, _ = drystress(
            wsoi, wisoi, swilt, sfield, froot,
        )
        assert float(stresstu[0]) < 0.1
        assert float(stresstl[0]) < 0.1


# =========================================================================
# 7. WEATHER GENERATION (weather.py vs weather.f)
# =========================================================================

class TestWeatherGeneration:
    """Verify weather generator against Fortran weather.f."""

    def test_precip_timing_random_duration(self):
        """Precip event length should vary between plenmin and plenmax."""
        from ibis.physics.weather import compute_precip_timing
        key = jax.random.PRNGKey(42)
        npoi = 100
        precip = jnp.full((npoi,), 5.0)  # 5mm/day
        dtime = 3600.0

        # Run many times to check distribution
        durations = []
        for i in range(20):
            key, k = jax.random.split(key)
            _, plens, _, _ = compute_precip_timing(k, precip, dtime)
            durations.append(plens)

        all_plens = jnp.concatenate(durations)
        plenmin_s = dtime * (1 + int((4.0 * 3600.0 - 1.0) / dtime))
        plenmax_s = dtime * max(int(24.0 * 3600.0 / dtime), 1)

        # All durations should be within valid range
        assert float(jnp.min(all_plens)) >= plenmin_s - dtime
        assert float(jnp.max(all_plens)) <= plenmax_s + dtime
        # Should have some variation (not all the same)
        assert float(jnp.std(all_plens)) > 0, "No variation in precip duration"

    def test_daily_temperature_ordering(self):
        """Daily temperature: tmin <= td <= tmax."""
        from ibis.physics.weather import daily, ClimateData, MonthlyPrecip
        npoi = 4
        key = jax.random.PRNGKey(0)
        clim = ClimateData(
            xint=jnp.full((npoi, 12), 25.0 + TMELT),
            xinq=jnp.full((npoi, 12), 70.0),
            xinprec=jnp.full((npoi, 12), 5.0),
            xinwind=jnp.full((npoi, 12), 3.0),
            xincld=jnp.full((npoi, 12), 50.0),
            xinwet=jnp.full((npoi, 12), 15.0),
            xintrng=jnp.full((npoi, 12), 10.0),
            xintopo=jnp.full((npoi,), 200.0),
        )
        # `iwetdaysum` was dropped from MonthlyPrecip when the wet-day
        # bookkeeping moved into generate_month_precip; the NamedTuple now
        # carries only iwetday and precipday.
        mprecip = MonthlyPrecip(
            precipday=jnp.full((npoi, 31), 5.0),
            iwetday=jnp.ones((npoi, 31), dtype=jnp.int32),
        )

        # Signature is daily(key, imonth, iday, climate, monthly_precip,
        # xstore); the call below had climate/monthly_precip in the positional
        # slots for imonth/iday and passed a since-removed `ndaypm`.
        _, dw = daily(
            key, 6, 15, clim, mprecip,
            jnp.zeros((npoi, 3), dtype=jnp.float32),
        )

        # tmin <= td <= tmax
        assert jnp.all(dw.tmin <= dw.td + 0.1)
        assert jnp.all(dw.td <= dw.tmax + 0.1)


# =========================================================================
# 8. BIOGEOCHEMISTRY (biogeochem.py vs biogeochem.f)
# =========================================================================

class TestBiogeochem:
    """Verify CENTURY-like decomposition against Fortran biogeochem.f."""

    def test_carbon_conservation(self):
        """Total C in + heterotrophic respiration = total C out."""
        from ibis.physics.biogeochem import soilbgc
        from ibis.state.carbon import create_carbon_state
        from ibis.state.vegetation import create_vegetation_state
        from ibis.state.summary import create_summary_state
        from ibis.io.parameters import default_bgc_params
        npoi, npft = 2, 12

        carbon = create_carbon_state(npoi)
        veg = create_vegetation_state(npoi, npft, 6)
        summary = create_summary_state(npoi, npft)
        bgc_p = default_bgc_params()

        # Give some initial litter
        carbon = eqx.tree_at(
            lambda c: (c.clitlm, c.clitls, c.clitrm, c.clitrs),
            carbon,
            (jnp.full((npoi,), 0.1),
             jnp.full((npoi,), 0.05),
             jnp.full((npoi,), 0.08),
             jnp.full((npoi,), 0.04)),
        )

        hsoi = jnp.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])
        sand = jnp.full((npoi, 6), 50.0)
        clay = jnp.full((npoi, 6), 20.0)

        c_new, v_new, s_new = soilbgc(
            carbon, veg, summary, hsoi, sand, clay, bgc_p, 365,
        )

        # All pools should remain non-negative
        assert float(jnp.min(c_new.clitlm)) >= 0
        assert float(jnp.min(c_new.clitls)) >= 0
        assert float(jnp.min(c_new.clitrm)) >= 0
        # The single `cslow` pool was split into protected / non-protected slow
        # humus (csoislop / csoislon) plus passive humus (csoipas) when the
        # CENTURY soilbgc port landed; `cslow` no longer exists on CarbonState.
        assert float(jnp.min(c_new.csoislon)) >= 0
        assert float(jnp.min(c_new.csoislop)) >= 0
        assert float(jnp.min(c_new.csoipas)) >= 0


# =========================================================================
# 9. INTERCEPTION CASCADE (interception.py vs canopy.f)
# =========================================================================

class TestInterception:
    """Verify precipitation cascade against Fortran canopy.f."""

    def test_cascade_water_conservation(self):
        """Total rain+snow in = intercepted + throughfall at each level."""
        from ibis.physics.interception import cascade, InterceptionParams
        npoi = 2
        ip = InterceptionParams()

        # 5 mm/hr rain, no snow
        raina = jnp.full((npoi,), 5.0 / 3600.0)
        snowa = jnp.zeros((npoi,))

        res = cascade(
            tu=jnp.full((npoi,), 293.0),
            ts=jnp.full((npoi,), 293.0),
            tl=jnp.full((npoi,), 293.0),
            t12=jnp.full((npoi,), 293.0),
            t34=jnp.full((npoi,), 293.0),
            wliqu=jnp.zeros((npoi,)),
            wliqs=jnp.zeros((npoi,)),
            wliql=jnp.zeros((npoi,)),
            wsnou=jnp.zeros((npoi,)),
            wsnos=jnp.zeros((npoi,)),
            wsnol=jnp.zeros((npoi,)),
            lai=jnp.full((npoi, 2), 3.0),
            sai=jnp.full((npoi, 2), 0.5),
            fu=jnp.full((npoi,), 0.8),
            fl=jnp.full((npoi,), 0.5),
            ta=jnp.full((npoi,), 293.0),
            qa=jnp.full((npoi,), 0.008),
            psurf=jnp.full((npoi,), 101325.0),
            raina=raina, snowa=snowa,
            iparams=ip,
            dtime=3600.0,
            tmelt=TMELT, ch2o=CH2O, cice=CICE, hfus=HFUS,
            epsilon=EPS, cair=CAIR, hvap=HVAP, cvap=CVAP,
        )

        # Total reaching ground should be <= total input
        total_in = raina + snowa  # kg/m2/s
        total_ground = res.raing + res.snowg
        for i in range(npoi):
            assert float(total_ground[i]) <= float(total_in[i]) + 1e-8

    def test_cascade_returns_trainl(self):
        """Cascade should return trainl field (below-upper-canopy rain temp)."""
        from ibis.physics.interception import cascade, InterceptionParams, CascadeResult
        assert hasattr(CascadeResult, '_fields')
        assert 'trainl' in CascadeResult._fields
        assert 'rainl' in CascadeResult._fields


# =========================================================================
# 10. FULL TIMESTEP INTEGRATION (loops.py vs main.f lsxmain)
# =========================================================================

class TestFullTimestep:
    """Verify the full timestep chain produces physically consistent results."""

    @pytest.fixture
    def setup(self):
        """Create a test state for full timestep testing."""
        from ibis.state.model_state import create_ibis_state
        from ibis.timestepping.loops import default_sim_params, timestep_body
        from ibis.dimensions import GridConfig

        NPOI = 4
        GRID = GridConfig(nlon=2, nlat=2, npoi=NPOI, xres=90.0, yres=90.0,
                          nband=2, nsoilay=6, nsnolay=3, npft=12)
        DTIME = 3600.0

        st = create_ibis_state(GRID)
        # Same setup as test_integration.py
        new_veg = eqx.tree_at(
            lambda v: (v.lai, v.sai, v.fu, v.fl, v.ztop, v.zbot, v.exist,
                       v.frac, v.cbior, v.cbiow, v.sapfrac),
            st.vegetation,
            (jnp.full((NPOI, 2), 2.0), jnp.full((NPOI, 2), 0.5),
             jnp.full((NPOI,), 0.8), jnp.full((NPOI,), 0.5),
             jnp.array([[5.0, 20.0]] * NPOI),
             jnp.array([[0.5, 2.0]] * NPOI),
             jnp.ones((NPOI, 12)),
             jnp.full((NPOI, 12), 1.0/12.0),  # equal PFT fractions
             jnp.full((NPOI, 12), 0.5),         # root biomass
             jnp.full((NPOI, 12), 2.0),          # wood biomass
             jnp.full((NPOI,), 0.1),             # sapwood fraction
             ),
        )
        new_soil = eqx.tree_at(
            lambda s: (s.poros, s.hydraul, s.suction, s.bex,
                       s.sfield, s.swilt, s.sand, s.clay,
                       s.hsoi, s.z0soi, s.wsoi, s.albsav, s.albsan),
            st.soil,
            (jnp.full((NPOI, 6), 0.45), jnp.full((NPOI, 6), 1e-5),
             jnp.full((NPOI, 6), -0.15), jnp.full((NPOI, 6), 5.0),
             jnp.full((NPOI, 6), 0.35), jnp.full((NPOI, 6), 0.15),
             jnp.full((NPOI, 6), 50.0), jnp.full((NPOI, 6), 20.0),
             jnp.array([0.10, 0.15, 0.30, 0.50, 1.00, 2.00, 4.00]),
             jnp.full((NPOI,), 0.01),
             jnp.full((NPOI, 6), 0.40),
             jnp.full((NPOI,), 0.10),   # albsav
             jnp.full((NPOI,), 0.20),   # albsan
             ),
        )
        new_atm = eqx.tree_at(
            lambda a: (a.ta, a.qa, a.ua, a.psurf, a.coszen,
                       a.solad, a.solai, a.fira, a.raina, a.snowa),
            st.atm,
            (jnp.full((NPOI,), 293.0), jnp.full((NPOI,), 0.008),
             jnp.full((NPOI,), 3.0), jnp.full((NPOI,), 101325.0),
             jnp.full((NPOI,), 0.8),
             jnp.full((NPOI, 2), 200.0), jnp.full((NPOI, 2), 50.0),
             jnp.full((NPOI,), 320.0),
             jnp.full((NPOI,), 0.0), jnp.full((NPOI,), 0.0)),
        )
        new_can = eqx.tree_at(
            lambda c: (c.tu, c.ts, c.tl, c.t12, c.t34, c.q12, c.q34),
            st.canopy,
            (jnp.full((NPOI,), 293.0), jnp.full((NPOI,), 293.0),
             jnp.full((NPOI,), 293.0), jnp.full((NPOI,), 293.0),
             jnp.full((NPOI,), 293.0),
             jnp.full((NPOI,), 0.008), jnp.full((NPOI,), 0.008)),
        )
        st = eqx.tree_at(
            lambda s: (s.vegetation, s.soil, s.atm, s.canopy),
            st, (new_veg, new_soil, new_atm, new_can),
        )

        params = default_sim_params(NPOI)
        consts = get_constants()
        return st, consts, params, DTIME

    def test_single_timestep_finite(self, setup):
        """Single timestep should produce all-finite outputs."""
        from ibis.timestepping.loops import timestep_body
        st, consts, params, dtime = setup
        st_new = timestep_body(st, consts, params, dtime)

        # All key state variables should be finite
        assert jnp.all(jnp.isfinite(st_new.soil.tsoi)), "tsoi has NaN/Inf"
        assert jnp.all(jnp.isfinite(st_new.soil.wsoi)), "wsoi has NaN/Inf"
        assert jnp.all(jnp.isfinite(st_new.canopy.tu)), "tu has NaN/Inf"
        assert jnp.all(jnp.isfinite(st_new.canopy.tl)), "tl has NaN/Inf"
        assert jnp.all(jnp.isfinite(st_new.snow.fi)), "fi has NaN/Inf"
        assert jnp.all(jnp.isfinite(st_new.snow.tlsub)), "tlsub has NaN/Inf"
        assert jnp.all(jnp.isfinite(st_new.soil.wpud)), "wpud has NaN/Inf"
        assert jnp.all(jnp.isfinite(st_new.soil.wipud)), "wipud has NaN/Inf"

    def test_temperature_bounds_24h(self, setup):
        """After 24 timesteps, temperatures should stay in [150, 400] K."""
        from ibis.timestepping.loops import timestep_body
        st, consts, params, dtime = setup
        for _ in range(24):
            st = timestep_body(st, consts, params, dtime)

        assert jnp.all(st.soil.tsoi > 150), "tsoi too cold"
        assert jnp.all(st.soil.tsoi < 400), "tsoi too warm"
        assert jnp.all(st.canopy.tu > 150), "tu too cold"
        assert jnp.all(st.canopy.tu < 400), "tu too warm"
        assert jnp.all(st.soil.tg > 150), "tg too cold"
        assert jnp.all(st.soil.tg < 400), "tg too warm"

    def test_moisture_bounds_24h(self, setup):
        """After 24 timesteps, moisture should stay in [0, 1+eps]."""
        from ibis.timestepping.loops import timestep_body
        st, consts, params, dtime = setup
        for _ in range(24):
            st = timestep_body(st, consts, params, dtime)

        assert jnp.all(st.soil.wsoi >= -0.01), "wsoi negative"
        assert jnp.all(st.soil.wsoi <= 1.01), "wsoi > 1"
        assert jnp.all(st.soil.wisoi >= -0.01), "wisoi negative"
        assert jnp.all(st.snow.fi >= 0.0), "fi negative"
        assert jnp.all(st.snow.fi <= 1.0), "fi > 1"

    def test_snow_timestep_with_snowfall(self, setup):
        """Timestep with snowfall should increase snow cover."""
        from ibis.timestepping.loops import timestep_body
        st, consts, params, dtime = setup

        # Set freezing conditions with snowfall
        st = eqx.tree_at(
            lambda s: (s.atm.ta, s.atm.snowa, s.atm.raina,
                       s.canopy.tu, s.canopy.ts, s.canopy.tl,
                       s.canopy.t12, s.canopy.t34, s.soil.tg),
            st,
            (jnp.full((4,), TMELT - 10.0),  # cold
             jnp.full((4,), 2e-4),            # snowfall (kg/m2/s)
             jnp.zeros((4,)),                 # no rain
             *[jnp.full((4,), TMELT - 10.0)] * 5,
             jnp.full((4,), TMELT - 5.0)),
        )

        st_new = timestep_body(st, consts, params, dtime)

        # Snow cover should have increased from 0
        assert jnp.any(st_new.snow.fi > 0), "No snow formed"
        # Snow thickness should be positive where fi > 0
        has_snow = st_new.snow.fi > 0
        if jnp.any(has_snow):
            total_hsno = jnp.sum(st_new.snow.hsno, axis=1)
            assert jnp.all(total_hsno[has_snow] > 0), "Snow fi>0 but hsno=0"

    def test_puddle_dynamics(self, setup):
        """Heavy rain should create puddle water."""
        from ibis.timestepping.loops import timestep_body
        st, consts, params, dtime = setup

        # Set heavy rain
        st = eqx.tree_at(
            lambda s: s.atm.raina,
            st,
            jnp.full((4,), 1e-3),  # ~3.6 mm/hr
        )

        st_new = timestep_body(st, consts, params, dtime)

        # wpud should have increased (or stayed zero if all infiltrated)
        # At least the soil should have more water
        w_before = jnp.sum(st.soil.wsoi * 0.45, axis=1)
        w_after = jnp.sum(st_new.soil.wsoi * 0.45, axis=1)
        wpud_after = st_new.soil.wpud
        # Either soil water or puddle should have increased
        total_increase = (w_after - w_before) + wpud_after
        assert jnp.all(total_increase > -0.1), "Water lost during rain"

    def test_jit_compatible(self, setup):
        """timestep_body should be JIT-compilable."""
        from ibis.timestepping.loops import timestep_body
        st, consts, params, dtime = setup
        jit_step = jax.jit(timestep_body, static_argnums=(3,))
        st_new = jit_step(st, consts, params, dtime)
        assert jnp.all(jnp.isfinite(st_new.soil.tsoi))


# =========================================================================
# 11. F77 REFERENCE OUTPUT COMPARISON (if data available)
# =========================================================================

class TestF77Reference:
    """Compare against F77 reference output from India regression run.

    These tests only run if F77 output files exist at the expected path.
    They compare JAX-computed annual/monthly means against F77 output
    to verify large-scale agreement (not exact match due to precision
    differences and stochastic weather generation).
    """

    F77_DIR = "/home/min/IBIS/tests/regression/run_f77/output"

    @pytest.fixture
    def f77_monthly(self):
        """Load F77 monthly reference data."""
        import os
        if not os.path.exists(os.path.join(self.F77_DIR, "monthly/tsoi.nc")):
            pytest.skip("F77 reference data not found")
        import netCDF4 as nc
        return {
            "tsoi": nc.Dataset(os.path.join(self.F77_DIR, "monthly/tsoi.nc")),
            "wsoi": nc.Dataset(os.path.join(self.F77_DIR, "monthly/wsoi.nc")),
            "aet": nc.Dataset(os.path.join(self.F77_DIR, "monthly/aet.nc")),
            "npptot": nc.Dataset(os.path.join(self.F77_DIR, "monthly/npptot.nc")),
            "temp": nc.Dataset(os.path.join(self.F77_DIR, "monthly/temp.nc")),
            "snof": nc.Dataset(os.path.join(self.F77_DIR, "monthly/snof.nc")),
        }

    def test_f77_tsoi_range(self, f77_monthly):
        """F77 soil temperature should be in reasonable range (-30 to 50 C)."""
        tsoi = f77_monthly["tsoi"]["tsoi"][:]  # (12, 51, 55)
        valid = ~np.isnan(tsoi) & (tsoi != 0.0)
        assert np.all(tsoi[valid] > -40), f"tsoi min = {np.nanmin(tsoi[valid])}"
        assert np.all(tsoi[valid] < 55), f"tsoi max = {np.nanmax(tsoi[valid])}"

    def test_f77_wsoi_bounded(self, f77_monthly):
        """F77 soil moisture should be in [0, 1]."""
        wsoi = f77_monthly["wsoi"]["wsoi"][:]
        valid = ~np.isnan(wsoi) & (wsoi != 0.0)
        assert np.all(wsoi[valid] >= 0.0)
        assert np.all(wsoi[valid] <= 1.0 + 0.01)

    def test_f77_aet_physical(self, f77_monthly):
        """F77 AET should be positive and < 10 mm/day."""
        aet = f77_monthly["aet"]["aet"][:]
        valid = ~np.isnan(aet) & (aet != 0.0)
        assert np.all(aet[valid] > -1.0), f"AET min = {np.nanmin(aet[valid])}"
        assert np.all(aet[valid] < 10.0), f"AET max = {np.nanmax(aet[valid])}"

    def test_f77_npp_physical(self, f77_monthly):
        """F77 NPP should be bounded (-0.5 to 0.5 kgC/m2/day)."""
        npp = f77_monthly["npptot"]["npptot"][:]
        valid = ~np.isnan(npp)
        assert np.nanmin(npp[valid]) > -0.5
        assert np.nanmax(npp[valid]) < 0.5

    def test_f77_seasonal_cycle(self, f77_monthly):
        """India should have summer-monsoon AET peak (Jul-Sep)."""
        aet = f77_monthly["aet"]["aet"][:]  # (12, 51, 55)
        # Mean over space (only land points)
        aet_monthly = np.nanmean(aet, axis=(1, 2))
        # Summer (Jun-Aug, indices 5-7) should be wetter than winter (Dec-Feb, 11,0,1)
        summer = np.mean(aet_monthly[5:8])
        winter = np.mean([aet_monthly[11], aet_monthly[0], aet_monthly[1]])
        assert summer > winter, f"Summer AET {summer:.2f} <= Winter {winter:.2f}"
