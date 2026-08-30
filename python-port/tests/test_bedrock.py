"""Thermal-only bedrock layers below the soil column.

See ibis/physics/bedrock.py for why the 4 m Fortran column is too shallow.
"""

import jax.numpy as jnp
import numpy as np

from ibis.physics.soil_physics import soilheat
from ibis.physics.bedrock import (
    BedrockParams, soilheat_with_bedrock, initialize_bedrock_temperature,
)

HSOI = jnp.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])
NPOI, NSOI = 4, 6


def _args(npoi=NPOI, n=NSOI, seed=0, wflo_scale=1e-5):
    rng = np.random.default_rng(seed)
    u = lambda *sh, lo=0.0, hi=1.0: jnp.asarray(rng.uniform(lo, hi, sh))
    return dict(
        tsoi=u(npoi, n, lo=265, hi=295), otsoi=u(npoi, n, lo=265, hi=295),
        owsoi=u(npoi, n), wsoi=u(npoi, n), wisoi=u(npoi, n, hi=0.3),
        poros=u(npoi, n, lo=0.3, hi=0.5), consoi=u(npoi, n, lo=0.5, hi=2.0),
        csoi_mineral=u(npoi, n, lo=800, hi=900),
        rhosoi=u(npoi, n, lo=2000, hi=2700), hsoi=HSOI[:n],
        c0pud=u(npoi, n, hi=10.0), c1pud=u(npoi, n, hi=10.0),
        fhtop=u(npoi, lo=-50, hi=50), wflo=u(npoi, n + 1, hi=wflo_scale),
        ch2o=4218.0, cice=2106.0, rhow=1000.0, dtime=3600.0,
    )


def _quiescent(a):
    """Strip terms that make the step non-conservative.

    c0pud != c1pud or owsoi != wsoi make the scheme's old- and new-timestep heat
    capacities differ, which is a real source/sink. Tests that isolate
    conduction across the seam must not carry one.
    """
    a = dict(a)
    a["owsoi"] = a["wsoi"]
    a["otsoi"] = a["tsoi"]
    a["c0pud"] = jnp.zeros_like(a["c0pud"])
    a["c1pud"] = jnp.zeros_like(a["c1pud"])
    return a


def _call(a, deep, p):
    return soilheat_with_bedrock(
        a["tsoi"], deep, a["otsoi"], deep, a["owsoi"], a["wsoi"], a["wisoi"],
        a["poros"], a["consoi"], a["csoi_mineral"], a["rhosoi"], HSOI,
        a["c0pud"], a["c1pud"], a["fhtop"], a["wflo"],
        a["ch2o"], a["cice"], a["rhow"], a["dtime"], p)


def test_n_hydro_default_is_unchanged():
    """The default path must be bit-identical to the Fortran behaviour."""
    a = _args()
    assert jnp.all(soilheat(**a).tsoi == soilheat(**a, n_hydro=NSOI).tsoi)


def test_bedrock_matches_manual_extension():
    """The wrapper must equal a hand-assembled extended column."""
    a, p = _args(), BedrockParams()
    nb = p.nlayer
    deep = jnp.full((NPOI, nb), 280.0)
    z = jnp.zeros((NPOI, nb))
    cat = lambda x, y: jnp.concatenate([x, y], axis=1)
    manual = soilheat(
        tsoi=cat(a["tsoi"], deep), otsoi=cat(a["otsoi"], deep),
        owsoi=cat(a["owsoi"], z), wsoi=cat(a["wsoi"], z), wisoi=cat(a["wisoi"], z),
        poros=cat(a["poros"], z),
        consoi=cat(a["consoi"], jnp.full((NPOI, nb), p.consoi)),
        csoi_mineral=cat(a["csoi_mineral"], jnp.full((NPOI, nb), p.csoi)),
        rhosoi=cat(a["rhosoi"], jnp.full((NPOI, nb), p.rhosoi)),
        hsoi=jnp.concatenate([HSOI, jnp.asarray(p.hsoi)]),
        c0pud=cat(a["c0pud"], z), c1pud=cat(a["c1pud"], z), fhtop=a["fhtop"],
        wflo=jnp.concatenate([a["wflo"], z], axis=1),
        ch2o=a["ch2o"], cice=a["cice"], rhow=a["rhow"], dtime=a["dtime"],
        n_hydro=NSOI,
    )
    res, deep_new = _call(a, deep, p)
    assert jnp.array_equal(res.tsoi, manual.tsoi[:, :NSOI])
    assert jnp.array_equal(deep_new, manual.tsoi[:, NSOI:])
    assert res.tsoi.shape == (NPOI, NSOI)      # callers see soil layers only
    assert res.hflo.shape == (NPOI, NSOI + 1)


def test_drainage_heat_does_not_enter_bedrock():
    """Drainage must leave the domain at the soil base, not warm the rock."""
    a = _args(wflo_scale=0.0)
    T0 = 280.0
    a["fhtop"] = jnp.zeros((NPOI,))
    a["tsoi"] = jnp.full((NPOI, NSOI), T0)
    a["otsoi"] = jnp.full((NPOI, NSOI), T0)
    a["wflo"] = jnp.zeros((NPOI, NSOI + 1)).at[:, NSOI].set(1e-4)
    a = _quiescent(a)
    p = BedrockParams()
    deep = jnp.full((NPOI, p.nlayer), T0)
    _, deep_new = _call(a, deep, p)
    assert jnp.allclose(deep_new, T0, atol=1e-6)


def test_bedrock_conducts_heat_downward():
    """A warm soil column must warm the rock below it (the seam is not sealed)."""
    a = _args(wflo_scale=0.0)
    a["fhtop"] = jnp.zeros((NPOI,))
    a["tsoi"] = jnp.full((NPOI, NSOI), 290.0)
    a["otsoi"] = a["tsoi"]
    a["wflo"] = jnp.zeros((NPOI, NSOI + 1))
    a = _quiescent(a)
    p = BedrockParams()
    deep = jnp.full((NPOI, p.nlayer), 270.0)
    _, deep_new = _call(a, deep, p)
    assert deep_new[0, 0] > 270.0                      # top of rock warms
    assert deep_new[0, 0] > deep_new[0, -1]            # gradient points down


def test_energy_is_conserved_across_the_seam():
    """With no drainage, stored energy must change by exactly fhtop*dtime."""
    a = _args(wflo_scale=0.0)
    a["wflo"] = jnp.zeros((NPOI, NSOI + 1))
    a = _quiescent(a)
    p = BedrockParams()
    deep = jnp.full((NPOI, p.nlayer), 279.0)
    res, deep_new = _call(a, deep, p)
    cv_s = ((1 - a["poros"]) * a["csoi_mineral"] * a["rhosoi"]
            + a["poros"] * (1 - a["wisoi"]) * a["ch2o"] * a["rhow"] * a["wsoi"]
            + a["poros"] * a["wisoi"] * a["cice"] * a["rhow"])
    dE = jnp.sum(cv_s * HSOI[None, :] * (res.tsoi - a["tsoi"]), axis=1)
    dE += jnp.sum(p.volumetric_heat_capacity
                  * jnp.asarray(p.hsoi)[None, :] * (deep_new - deep), axis=1)
    # Tolerance is set by float32, not by the scheme: stored energy is ~5e8
    # J/m2 while the step changes it by ~5e4, so differencing loses four digits.
    # test_energy_is_conserved_exactly_in_float64 pins the actual accuracy.
    assert jnp.allclose(dE, a["fhtop"] * a["dtime"], rtol=2e-2)


def test_downward_continuation_seeds_a_geothermal_profile():
    p = BedrockParams()
    tb = jnp.array([275.0, 260.0])
    deep = initialize_bedrock_temperature(tb, HSOI, p)
    assert deep.shape == (2, p.nlayer)
    assert jnp.all(jnp.diff(deep, axis=1) > 0)          # warms with depth
    assert jnp.all(deep[:, 0] > tb)                     # below the soil base
    # Soil nodes sit at [.05 .175 .375 .75 1.5 3.0] m, so with (2, 2, 4) m of
    # rock the bedrock nodes are at 5, 7 and 10 m -- i.e. 2, 4 and 7 m below
    # the deepest soil node.
    expect = tb[:, None] + p.dtdz * jnp.array([2.0, 4.0, 7.0])[None, :]
    assert jnp.allclose(deep, expect)


def test_deeper_column_stops_the_base_node_over_responding():
    """The point of the change: the 4 m zero-flux base traps heat.

    Drive layer 1 with a step and integrate. The sealed column has nowhere to
    put the heat, so its deepest node ends warmer than one backed by bedrock.
    """
    n = NSOI
    f = lambda v: jnp.full((1, n), v)
    p = BedrockParams()
    T = jnp.full((1, n), 280.0).at[:, 0].set(300.0)
    common = dict(owsoi=f(0.0), wsoi=f(0.0), wisoi=f(0.0), poros=f(0.0),
                  consoi=f(2.0), csoi_mineral=f(880.0), rhosoi=f(2650.0),
                  c0pud=f(0.0), c1pud=f(0.0), fhtop=jnp.zeros((1,)),
                  wflo=jnp.zeros((1, n + 1)), ch2o=4218.0, cice=2106.0,
                  rhow=1000.0, dtime=86400.0)
    Ta = Tb = T
    deep = jnp.full((1, p.nlayer), 280.0)
    a = dict(common, hsoi=HSOI)
    for _ in range(400):
        Ta = soilheat(tsoi=Ta, otsoi=Ta, **a).tsoi
        rb, deep = soilheat_with_bedrock(
            Tb, deep, Tb, deep, common["owsoi"], common["wsoi"],
            common["wisoi"], common["poros"], common["consoi"],
            common["csoi_mineral"], common["rhosoi"], HSOI, common["c0pud"],
            common["c1pud"], common["fhtop"], common["wflo"], common["ch2o"],
            common["cice"], common["rhow"], common["dtime"], p)
        Tb = rb.tsoi
    assert Ta[0, -1] > 280.0 and Tb[0, -1] > 280.0
    assert Ta[0, -1] > Tb[0, -1]


def test_energy_is_conserved_exactly_in_float64():
    """The seam must not leak energy. Run in float64 to see the true error."""
    import jax
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        a = _quiescent(_args(wflo_scale=0.0))
        a = {k: (jnp.asarray(v, dtype=jnp.float64)
                 if isinstance(v, jnp.ndarray) else v) for k, v in a.items()}
        a["wflo"] = jnp.zeros((NPOI, NSOI + 1), dtype=jnp.float64)
        hsoi = jnp.asarray(HSOI, dtype=jnp.float64)
        p = BedrockParams()
        deep = jnp.full((NPOI, p.nlayer), 279.0, dtype=jnp.float64)
        res, deep_new = soilheat_with_bedrock(
            a["tsoi"], deep, a["otsoi"], deep, a["owsoi"], a["wsoi"],
            a["wisoi"], a["poros"], a["consoi"], a["csoi_mineral"],
            a["rhosoi"], hsoi, a["c0pud"], a["c1pud"], a["fhtop"], a["wflo"],
            a["ch2o"], a["cice"], a["rhow"], a["dtime"], p)
        cv_s = ((1 - a["poros"]) * a["csoi_mineral"] * a["rhosoi"]
                + a["poros"] * (1 - a["wisoi"]) * a["ch2o"] * a["rhow"] * a["wsoi"]
                + a["poros"] * a["wisoi"] * a["cice"] * a["rhow"])
        dE = jnp.sum(cv_s * hsoi[None, :] * (res.tsoi - a["tsoi"]), axis=1)
        dE += jnp.sum(p.volumetric_heat_capacity
                      * jnp.asarray(p.hsoi, dtype=jnp.float64)[None, :]
                      * (deep_new - deep), axis=1)
        assert jnp.allclose(dE, a["fhtop"] * a["dtime"], rtol=1e-10)
    finally:
        jax.config.update("jax_enable_x64", was)


# ---------------------------------------------------------------------------
# End-to-end: the column must actually be live inside a timestep
# ---------------------------------------------------------------------------

def _smoke_state(nbedrock):
    """Minimal physically-plausible state, mirroring tests/test_integration.py."""
    import equinox as eqx
    from ibis.dimensions import GridConfig
    from ibis.state.model_state import create_ibis_state

    n = 4
    grid = GridConfig(nlon=2, nlat=2, npoi=n, xres=90.0, yres=90.0,
                      nband=2, nsoilay=6, nbedrock=nbedrock, nsnolay=3, npft=12)
    st = create_ibis_state(grid)
    veg = eqx.tree_at(
        lambda v: (v.lai, v.sai, v.fu, v.fl, v.ztop, v.zbot, v.exist), st.vegetation,
        (jnp.full((n, 2), 2.0), jnp.full((n, 2), 0.5), jnp.full((n,), 0.8),
         jnp.full((n,), 0.5), jnp.array([[5.0, 20.0]] * n),
         jnp.array([[0.5, 2.0]] * n), jnp.ones((n, 12))))
    soil = eqx.tree_at(
        lambda s: (s.poros, s.hydraul, s.suction, s.bex, s.sfield, s.swilt,
                   s.sand, s.clay, s.hsoi, s.z0soi, s.wsoi), st.soil,
        (jnp.full((n, 6), 0.45), jnp.full((n, 6), 1e-5), jnp.full((n, 6), -0.15),
         jnp.full((n, 6), 5.0), jnp.full((n, 6), 0.35), jnp.full((n, 6), 0.15),
         jnp.full((n, 6), 50.0), jnp.full((n, 6), 20.0),
         jnp.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00, 4.00]),
         jnp.full((n,), 0.01), jnp.full((n, 6), 0.40)))
    atm = eqx.tree_at(
        lambda a: (a.ta, a.qa, a.ua, a.psurf, a.coszen, a.solad, a.solai,
                   a.fira, a.raina, a.snowa), st.atm,
        (jnp.full((n,), 293.0), jnp.full((n,), 0.008), jnp.full((n,), 3.0),
         jnp.full((n,), 101325.0), jnp.full((n,), 0.8), jnp.full((n, 2), 200.0),
         jnp.full((n, 2), 50.0), jnp.full((n,), 320.0), jnp.zeros((n,)),
         jnp.zeros((n,))))
    can = eqx.tree_at(
        lambda c: (c.tu, c.ts, c.tl, c.t12, c.t34, c.q12, c.q34), st.canopy,
        tuple([jnp.full((n,), 293.0)] * 5 + [jnp.full((n,), 0.008)] * 2))
    return eqx.tree_at(lambda s: (s.vegetation, s.soil, s.atm, s.canopy), st,
                       (veg, soil, atm, can))


def _run(nbedrock, nstep=48, t_deep=None):
    import equinox as eqx
    from ibis.constants import get_constants
    from ibis.timestepping.loops import default_sim_params, timestep_body
    st = _smoke_state(nbedrock)
    if t_deep is not None and st.soil.tsoi_deep is not None:
        st = eqx.tree_at(lambda s: s.soil.tsoi_deep, st,
                         jnp.full_like(st.soil.tsoi_deep, t_deep))
    c, p = get_constants(), default_sim_params(4)
    for _ in range(nstep):
        st = timestep_body(st, c, p, 1800.0)
    return st


def test_timestep_runs_with_bedrock_and_stays_finite():
    st = _run(3)
    assert st.soil.tsoi_deep.shape == (4, 3)
    assert jnp.all(jnp.isfinite(st.soil.tsoi))
    assert jnp.all(jnp.isfinite(st.soil.tsoi_deep))
    assert jnp.all((st.soil.tsoi > 200.0) & (st.soil.tsoi < 350.0))


def test_bedrock_is_live_not_inert():
    """The seam must exchange heat inside a real timestep.

    Guards against the layers being allocated but silently bypassed. A surface
    signal needs months to reach 4 m, so a short run cannot show this; instead
    start the rock 15 K colder than the soil, which drives conduction at once.
    """
    cold_rock = _run(3, t_deep=263.0)
    sealed = _run(0)
    assert sealed.soil.tsoi_deep is None
    # the rock warms towards the soil above it
    assert jnp.all(cold_rock.soil.tsoi_deep[:, 0] > 263.0)
    # and the deepest soil node loses heat into it, which the sealed column cannot
    assert jnp.all(cold_rock.soil.tsoi[:, -1] < sealed.soil.tsoi[:, -1])


def test_bedrock_does_not_perturb_the_surface_over_a_day():
    """Adding rock at 4-12 m must not change surface fluxes on daily timescales.

    The annual wave is what the deep column is for; the diurnal signal never
    reaches the seam. A surface difference would mean the change leaked upward.
    """
    a, b = _run(3), _run(0)
    assert jnp.allclose(a.soil.tsoi[:, 0], b.soil.tsoi[:, 0], atol=1e-5)
    assert jnp.allclose(a.soil.tg, b.soil.tg, atol=1e-5)


def test_layer_count_mismatch_is_an_error():
    """GridConfig.nbedrock and SimParams.bedrock must agree, loudly."""
    import pytest
    a = _quiescent(_args())
    p = BedrockParams()                       # 3 layers
    deep = jnp.full((NPOI, 2), 280.0)         # but state carries 2
    with pytest.raises(ValueError, match="bedrock layers"):
        _call(a, deep, p)
