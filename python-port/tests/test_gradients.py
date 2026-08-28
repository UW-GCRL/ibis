"""Gradient flow verification for IBIS-JAX (Phase 12).

Verifies that gradients propagate without NaN through:
  - Single timestep body (xfail — known NaN from non-smooth physics ops)
  - Multi-step simulation chains (xfail — same root cause)
  - ML component wrappers (should pass)
  - BGC modules standalone (should pass)
  - BGC modules through physics (xfail — inherits physics NaN)

The full physics timestep currently produces NaN gradients due to
non-smooth operations (integer indexing, hard clipping, etc.) in the
canopy solver and snow physics. Fixing this requires systematic
smoothing of the physics code, which is tracked as future work.
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from ibis.constants import get_constants
from ibis.dimensions import GridConfig
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.loops import (
    SimParams, default_sim_params, timestep_body,
)

NPOI = 2  # Small for gradient tests (faster)
GRID = GridConfig(nlon=2, nlat=1, npoi=NPOI, xres=180.0, yres=180.0,
                  nband=2, nsoilay=6, nsnolay=3, npft=12)
DTIME = 3600.0


def _make_state():
    st = create_ibis_state(GRID)
    new_veg = eqx.tree_at(
        lambda v: (v.lai, v.sai, v.fu, v.fl, v.ztop, v.zbot, v.exist),
        st.vegetation,
        (
            jnp.full((NPOI, 2), 2.0),
            jnp.full((NPOI, 2), 0.5),
            jnp.full((NPOI,), 0.8),
            jnp.full((NPOI,), 0.5),
            jnp.array([[5.0, 20.0]] * NPOI),
            jnp.array([[0.5, 2.0]] * NPOI),
            jnp.ones((NPOI, 12)),
        ),
    )
    new_soil = eqx.tree_at(
        lambda s: (s.poros, s.hydraul, s.suction, s.bex,
                   s.sfield, s.swilt, s.sand, s.clay,
                   s.hsoi, s.z0soi, s.wsoi),
        st.soil,
        (
            jnp.full((NPOI, 6), 0.45),
            jnp.full((NPOI, 6), 1e-5),
            jnp.full((NPOI, 6), -0.15),
            jnp.full((NPOI, 6), 5.0),
            jnp.full((NPOI, 6), 0.35),
            jnp.full((NPOI, 6), 0.15),
            jnp.full((NPOI, 6), 50.0),
            jnp.full((NPOI, 6), 20.0),
            jnp.array([0.10, 0.15, 0.30, 0.50, 1.00, 2.00, 4.00]),
            jnp.full((NPOI,), 0.01),
            jnp.full((NPOI, 6), 0.40),
        ),
    )
    new_atm = eqx.tree_at(
        lambda a: (a.ta, a.qa, a.ua, a.psurf, a.coszen,
                   a.solad, a.solai, a.fira, a.raina, a.snowa),
        st.atm,
        (
            jnp.full((NPOI,), 293.0),
            jnp.full((NPOI,), 0.008),
            jnp.full((NPOI,), 3.0),
            jnp.full((NPOI,), 101325.0),
            jnp.full((NPOI,), 0.8),
            jnp.full((NPOI, 2), 200.0),
            jnp.full((NPOI, 2), 50.0),
            jnp.full((NPOI,), 320.0),
            jnp.full((NPOI,), 0.0),
            jnp.full((NPOI,), 0.0),
        ),
    )
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
    return eqx.tree_at(
        lambda s: (s.vegetation, s.soil, s.atm, s.canopy),
        st,
        (new_veg, new_soil, new_atm, new_can),
    )


# ===== Single timestep gradient (xfail — physics has non-smooth ops) =====

class TestSingleStepGradient:
    """Gradient through a single physics timestep.

    Currently xfail: the full physics chain contains operations that
    produce NaN gradients (hard clipping, integer masking, etc.).
    """

    @pytest.mark.xfail(reason="Full physics timestep has non-smooth ops producing NaN grads")
    def test_grad_wrt_air_temp(self):
        state = _make_state()
        consts = get_constants()
        params = default_sim_params(NPOI)

        def loss(ta):
            new_atm = eqx.tree_at(lambda a: a.ta, state.atm, ta)
            st = eqx.tree_at(lambda s: s.atm, state, new_atm)
            st2 = timestep_body(st, consts, params, DTIME)
            return jnp.sum(st2.soil.tg)

        ta = jnp.full((NPOI,), 293.0)
        grad = jax.grad(loss)(ta)
        assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient: {grad}"

    @pytest.mark.xfail(reason="Full physics timestep has non-smooth ops producing NaN grads")
    def test_grad_wrt_solar(self):
        state = _make_state()
        consts = get_constants()
        params = default_sim_params(NPOI)

        def loss(solad):
            new_atm = eqx.tree_at(lambda a: a.solad, state.atm, solad)
            st = eqx.tree_at(lambda s: s.atm, state, new_atm)
            st2 = timestep_body(st, consts, params, DTIME)
            return jnp.sum(st2.canopy.tu)

        solad = jnp.full((NPOI, 2), 200.0)
        grad = jax.grad(loss)(solad)
        assert jnp.all(jnp.isfinite(grad))

    @pytest.mark.xfail(reason="Full physics timestep has non-smooth ops producing NaN grads")
    def test_grad_wrt_soil_moisture(self):
        state = _make_state()
        consts = get_constants()
        params = default_sim_params(NPOI)

        def loss(wsoi):
            new_soil = eqx.tree_at(lambda s: s.wsoi, state.soil, wsoi)
            st = eqx.tree_at(lambda s: s.soil, state, new_soil)
            st2 = timestep_body(st, consts, params, DTIME)
            return jnp.sum(st2.canopy.tu)

        wsoi = jnp.full((NPOI, 6), 0.4)
        grad = jax.grad(loss)(wsoi)
        assert jnp.all(jnp.isfinite(grad))

    @pytest.mark.xfail(reason="Full physics timestep has non-smooth ops producing NaN grads")
    def test_grad_wrt_lai(self):
        state = _make_state()
        consts = get_constants()
        params = default_sim_params(NPOI)

        def loss(lai):
            new_veg = eqx.tree_at(lambda v: v.lai, state.vegetation, lai)
            st = eqx.tree_at(lambda s: s.vegetation, state, new_veg)
            st2 = timestep_body(st, consts, params, DTIME)
            return jnp.sum(st2.canopy.tu) + jnp.sum(st2.soil.tg)

        lai = jnp.full((NPOI, 2), 2.0)
        grad = jax.grad(loss)(lai)
        assert jnp.all(jnp.isfinite(grad))


# ===== Multi-step gradient chain (xfail) =====

class TestMultiStepGradient:

    @pytest.mark.xfail(reason="Full physics chain produces NaN grads")
    def test_3_steps_grad_finite(self):
        state = _make_state()
        consts = get_constants()
        params = default_sim_params(NPOI)

        def loss(ta):
            new_atm = eqx.tree_at(lambda a: a.ta, state.atm, ta)
            st = eqx.tree_at(lambda s: s.atm, state, new_atm)
            for _ in range(3):
                st = timestep_body(st, consts, params, DTIME)
            return jnp.sum(st.soil.tg)

        ta = jnp.full((NPOI,), 293.0)
        grad = jax.grad(loss)(ta)
        assert jnp.all(jnp.isfinite(grad))


# ===== ML component gradient flow (should pass) =====

class TestMLGradientFlow:
    """Verify gradients flow through ML component swap."""

    def test_neural_stomata_grad(self):
        from ibis.ml._legacy_unwired.wrappers import make_neural_physics

        key = jax.random.PRNGKey(0)
        neural_stomata = make_neural_physics(
            input_keys=("ta", "qa"),
            input_sizes=(1, 1),
            output_keys=("gs",),
            output_sizes=(1,),
            hidden_sizes=[16, 8],
            key=key,
        )

        def loss(model, ta, qa):
            out = model({"ta": ta, "qa": qa})
            return jnp.sum(out["gs"])

        ta = jnp.full((4,), 293.0)
        qa = jnp.full((4,), 0.008)

        val, grad = eqx.filter_value_and_grad(loss)(neural_stomata, ta, qa)
        assert jnp.isfinite(val)
        grad_leaves = jax.tree.leaves(grad)
        for leaf in grad_leaves:
            if leaf is not None and hasattr(leaf, 'shape'):
                assert jnp.all(jnp.isfinite(leaf))

    def test_hybrid_physics_grad(self):
        from ibis.ml._legacy_unwired.wrappers import make_hybrid_physics

        def simple_physics(inputs, params=None):
            return {"y": inputs["x"] * 2.0}

        key = jax.random.PRNGKey(1)
        hybrid = make_hybrid_physics(
            physics_fn=simple_physics,
            input_keys=("x",),
            input_sizes=(1,),
            output_keys=("y",),
            output_sizes=(1,),
            hidden_sizes=[8],
            key=key,
        )

        def loss(model, x):
            out = model({"x": x})
            return jnp.sum(out["y"])

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        val, grad = eqx.filter_value_and_grad(loss)(hybrid, x)
        assert jnp.isfinite(val)


# ===== BGC module gradient flow (standalone — should pass) =====

class TestBGCGradientStandalone:
    """Gradient through BGC modules without full physics."""

    def test_wetland_grad(self):
        from ibis.bgc._legacy_unwired.wetlands import wetland_step, create_wetland_state, default_wetland_params

        ws = create_wetland_state(NPOI)
        wp = default_wetland_params()
        tsoi = jnp.full((NPOI, 6), 290.0)
        wsoi = jnp.full((NPOI, 6), 0.7)
        poros = jnp.full((NPOI, 6), 0.45)

        def loss(het_resp):
            ws2 = wetland_step(ws, tsoi, wsoi, poros, het_resp, wp)
            return jnp.sum(ws2.ch4_flux)

        het_resp = jnp.full((NPOI,), 2.0)
        grad = jax.grad(loss)(het_resp)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_bvoc_grad(self):
        from ibis.bgc._legacy_unwired.bvoc import bvoc_emissions

        def loss(tleaf):
            par = jnp.full((NPOI,), 800.0)
            lai = jnp.ones((NPOI, 12)) * 2.0
            frac = jnp.ones((NPOI, 12)) / 12.0
            bs = bvoc_emissions(tleaf, par, lai, frac)
            return jnp.sum(bs.isoprene)

        tleaf = jnp.full((NPOI,), 300.0)
        grad = jax.grad(loss)(tleaf)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_nee_grad(self):
        from ibis.bgc._legacy_unwired.nee import compute_nee

        def loss(gpp):
            ra = jnp.full((NPOI,), 2.0)
            rh = jnp.full((NPOI,), 1.5)
            ns = compute_nee(gpp, ra, rh)
            return jnp.sum(ns.nee_co2)

        gpp = jnp.full((NPOI,), 5.0)
        grad = jax.grad(loss)(gpp)
        assert jnp.allclose(grad, -1.0)  # d(nee)/d(gpp) = -1

    def test_soil_gas_grad(self):
        from ibis.bgc._legacy_unwired.soil_gases import soil_gas_step, create_soil_gas_state, default_soil_gas_params

        sg = create_soil_gas_state(NPOI)
        params = default_soil_gas_params()
        tsoi = jnp.full((NPOI, 6), 295.0)
        wsoi = jnp.full((NPOI, 6), 0.6)
        poros = jnp.full((NPOI, 6), 0.45)

        def loss(n_min):
            sg2 = soil_gas_step(sg, tsoi, wsoi, poros,
                                jnp.full((NPOI,), 0.5), n_min, params)
            return jnp.sum(sg2.n2o_total)

        n_min = jnp.full((NPOI,), 0.01)
        grad = jax.grad(loss)(n_min)
        assert jnp.all(jnp.isfinite(grad))

    def test_permafrost_grad(self):
        from ibis.bgc.permafrost import stefan_depth, default_permafrost_params

        params = default_permafrost_params()

        def loss(gdd):
            tsoi = jnp.full((NPOI, 6), 275.0)
            ald = stefan_depth(tsoi, 273.16, gdd, params)
            return jnp.sum(ald)

        gdd = jnp.full((NPOI,), 500.0)
        grad = jax.grad(loss)(gdd)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad > 0.0)  # more TDD → deeper ALD


# ===== BGC through physics (xfail — inherits physics NaN) =====

class TestBGCGradientThroughPhysics:

    @pytest.mark.xfail(reason="Full physics chain produces NaN grads")
    def test_physics_to_wetland_grad(self):
        from ibis.bgc._legacy_unwired.wetlands import wetland_step, create_wetland_state, default_wetland_params

        state = _make_state()
        consts = get_constants()
        params = default_sim_params(NPOI)
        ws = create_wetland_state(NPOI)
        wp = default_wetland_params()

        def loss(ta):
            new_atm = eqx.tree_at(lambda a: a.ta, state.atm, ta)
            st = eqx.tree_at(lambda s: s.atm, state, new_atm)
            st = timestep_body(st, consts, params, DTIME)
            het_resp = jnp.full((NPOI,), 1.0)
            ws2 = wetland_step(ws, st.soil.tsoi, st.soil.wsoi,
                               jnp.full((NPOI, 6), 0.45), het_resp, wp)
            return jnp.sum(ws2.ch4_flux)

        ta = jnp.full((NPOI,), 293.0)
        grad = jax.grad(loss)(ta)
        assert jnp.all(jnp.isfinite(grad))


# --- masked-NaN divisions (2026-08-25) --------------------------------------
# Both of these computed a NaN inside a `jnp.where` branch that was then
# discarded. Harmless for the forward value, but `jax.grad` propagates NaN
# through the untaken branch, and under `jax_debug_nans` they fire on every
# timestep and mask any real NaN behind them. Found while bisecting the
# Aug-1915 Taymyr Peninsula NaN (docs/STATUS.md).


def test_mix_is_differentiable_at_zero_mass():
    """`jnp.sign(0.0)` is 0.0, so the old denominator was exactly zero."""
    import jax
    import jax.numpy as jnp

    from ibis.physics.interception import mix

    def f(x):
        z = jnp.zeros_like(x)
        return mix(x, jnp.full_like(x, 280.0), z, jnp.full_like(x, 270.0),
                   z, z, 1e-7)[1].sum()

    g = jax.grad(f)(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(g)), g


def test_mix_value_is_unchanged_by_the_safe_denominator():
    import jax.numpy as jnp

    from ibis.physics.interception import mix

    x1 = jnp.array([0.0, 2.0, 1e-9])
    xm, tm = mix(x1, jnp.full(3, 280.0), jnp.zeros(3), jnp.full(3, 270.0),
                 jnp.zeros(3), jnp.zeros(3), 1e-7)
    assert jnp.allclose(xm, jnp.array([0.0, 2.0, 0.0]))
    assert jnp.allclose(tm, jnp.array([0.0, 280.0, 0.0]))


def test_impexp2_is_differentiable_when_temperature_does_not_change():
    """The denominator `t - told` is the same quantity the branch tests."""
    import jax
    import jax.numpy as jnp

    from ibis.physics.canopy_exchange import impexp2

    def f(t):
        return impexp2(t, jnp.array([273.16]), 273.16, 1e-7,
                       jnp.array([0.5]), jnp.array([False])).sum()

    g = jax.grad(f)(jnp.array([273.16]))
    assert jnp.all(jnp.isfinite(g)), g


def test_impexp2_still_returns_the_previous_weight_when_not_warming():
    import jax.numpy as jnp

    from ibis.physics.canopy_exchange import impexp2

    t = jnp.array([273.16, 280.0])
    told = jnp.array([273.16, 270.0])
    out = impexp2(t, told, 273.16, 1e-7, jnp.array([0.5, 0.5]),
                  jnp.array([False, False]))
    assert float(out[0]) == 0.5                     # unchanged -> carry forward
    assert 0.0 <= float(out[1]) <= 1.0
