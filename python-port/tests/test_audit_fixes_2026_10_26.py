"""CALIBRATION.md 62-63: the soilh2o linearisation weights, exact litterfall booking, shrub NPP on the cropland share,
leafless stands under the fire damping, and the spin-up tail. Each fix sits behind a switch whose off state is F77."""
from __future__ import annotations

import pathlib

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.io.parameters import default_pft_params
from ibis.physics.fire import FireParams, broadleaf_share
from ibis.physics.soil_physics import soilh2o, wadjust
from ibis.physics.vegetation_dynamics import biomass_update, dynaveg
from ibis.state.carbon import create_carbon_state
from ibis.state.summary import create_summary_state
from ibis.state.vegetation import create_vegetation_state


# ---------------------------------------------------------------- soil water
HSOI = jnp.asarray([0.10, 0.15, 0.25, 0.50, 1.00, 2.00], jnp.float32)


def _column(wsoi):
    n = len(wsoi); w = jnp.asarray(wsoi, jnp.float32)
    k = lambda v: jnp.full((n, 6), v, jnp.float32)
    return w, k(0.0), k(0.46), k(0.46), k(3.667e-6), k(0.11), k(4.5)


def _pump(wsoi, consistent, nsteps=240):
    w, wisoi, poros, porosflo, hyd, suc, bex = _column(wsoi); n = w.shape[0]
    eps, rhow, dt = 1e-7, 1000.0, 3600.0
    store = lambda x: np.asarray(jnp.sum(HSOI[None, :] * poros * rhow * x, 1))
    s0 = store(w); drain = np.zeros(n); created = np.zeros(n)
    for _ in range(nsteps):
        h = soilh2o(w, wisoi, poros, porosflo, hyd, suc, bex, HSOI, jnp.zeros(n), jnp.zeros(n), jnp.zeros(n),
                    0.04, dt, rhow, eps, consistent_weights=consistent)
        drain += np.asarray(h.wflo[:, -1]) * dt
        w, _, gadj = wadjust(h.wsoi, wisoi, poros, HSOI, rhow, dt, eps); created += np.asarray(gadj) * dt
    return drain, created, store(w) - s0, np.asarray(w)


DRY_OVER_WET = [[0.0, 0.0, 0.0, 0.0, 0.711, 0.864], [0.3, 0.3, 0.35, 0.0, 0.711, 0.864]]
UNIFORM = [[0.2] * 6, [0.7] * 6]


def test_f77_weights_pump_water_from_a_dry_layer_over_a_wet_one():
    drain, created, ds, w = _pump(DRY_OVER_WET, False)
    assert np.all(drain > 15.0) and np.all(created > 15.0)          # ~690 mm/yr at 240 h -> ~19 mm
    np.testing.assert_allclose(created, drain + ds, atol=0.05)       # the floor supplies what drains
    assert np.all(w[:, 3] == 0.0)                                    # the wadjust fingerprint


def test_consistent_weights_create_no_water_and_drain_the_store():
    drain, created, ds, w = _pump(DRY_OVER_WET, True)
    assert np.all(np.abs(created) < 1e-3)
    np.testing.assert_allclose(ds, -drain, atol=0.05)
    assert np.all(w[:, 4] < 0.711) and np.all(drain > 10.0)


def test_switch_is_inert_on_uniform_columns_and_off_by_default():
    a = _pump(UNIFORM, False); b = _pump(UNIFORM, True)
    np.testing.assert_allclose(a[3][0], b[3][0], rtol=1e-5)          # dry uniform: identical
    np.testing.assert_allclose(a[3][1], b[3][1], rtol=2e-3)          # wet uniform: drains, gradients form, within 0.2 %
    np.testing.assert_allclose(a[0], b[0], rtol=1e-2, atol=1e-6)
    assert np.all(np.abs(a[1]) < 1e-3) and np.all(np.abs(b[1]) < 1e-3)   # no water created either way
    from ibis.timestepping.loops import SimParams
    assert SimParams.__dataclass_fields__['h2o_consistent_weights'].default is False


# ----------------------------------------------------------- pool accounting
def _set(obj, **kw):
    for k, v in kw.items():
        obj = eqx.tree_at(lambda o, _k=k: getattr(o, _k), obj, v)
    return obj


def _state(npoi=5, npft=12, npp=0.6):
    veg = create_vegetation_state(npoi, npft, 6); carbon = create_carbon_state(npoi); summary = create_summary_state(npoi, npft)
    rng = np.random.default_rng(0)
    veg = _set(veg, cbiol=jnp.asarray(rng.uniform(0.02, 0.8, (npoi, npft)), jnp.float32), cbiow=jnp.asarray(rng.uniform(0.1, 9.0, (npoi, npft)), jnp.float32),
               cbior=jnp.asarray(rng.uniform(0.02, 0.8, (npoi, npft)), jnp.float32), plai=jnp.full((npoi, npft), 1.0), exist=jnp.ones((npoi, npft)))
    summary = _set(summary, aynpp=jnp.asarray(rng.uniform(0.05, npp, (npoi, npft)), jnp.float32))
    return veg, carbon, summary


def _closure(fn, pp, **kw):
    veg, carbon, summary = _state()
    nv, nc, ns = fn(veg, carbon, summary, pp, *([1, 1e-12] if fn is dynaveg else [1e-12]), **kw)
    live0 = np.asarray(veg.cbiol + veg.cbiow + veg.cbior).sum(1); live1 = np.asarray(nv.cbiol + nv.cbiow + nv.cbior).sum(1)
    npp = np.asarray(summary.aynpp).sum(1); fall = np.asarray(nc.falll + nc.fallw + nc.fallr); cd = np.asarray(nc.cdisturb)
    return live1 - live0, npp - fall - cd


@pytest.mark.parametrize('fn', [biomass_update, dynaveg])
def test_exact_litterfall_closes_the_live_pool_budget(fn):
    pp = default_pft_params()
    assert pp.exact_litterfall is False
    d_f77, b_f77 = _closure(fn, pp)
    d_ex, b_ex = _closure(fn, eqx.tree_at(lambda p: p.exact_litterfall, pp, True))
    np.testing.assert_allclose(d_ex, d_f77, rtol=1e-6)               # the pool trajectory is unchanged
    assert np.max(np.abs(d_f77 - b_f77)) > 0.05                      # F77 booking does not close off equilibrium
    np.testing.assert_allclose(d_ex, b_ex, atol=2e-4)                # exact booking closes


def test_exact_litterfall_equals_f77_at_equilibrium():
    pp = eqx.tree_at(lambda p: p.exact_litterfall, default_pft_params(), True)
    veg, carbon, summary = _state()
    a, tl, tr, tw = (np.asarray(getattr(pp, k)) for k in ('aleaf', 'tauleaf', 'tauroot', 'tauwood0'))
    npp = np.asarray(summary.aynpp)
    veg = _set(veg, cbiol=jnp.asarray(a * tl * npp, jnp.float32), cbior=jnp.asarray(np.asarray(pp.aroot) * tr * npp, jnp.float32),
               cbiow=jnp.asarray(np.asarray(pp.awood) * tw * npp, jnp.float32))
    f77 = biomass_update(veg, carbon, summary, default_pft_params(), 1e-12)[1]
    ex = biomass_update(veg, carbon, summary, pp, 1e-12)[1]
    for k in ('falll', 'fallr', 'fallw'):
        np.testing.assert_allclose(np.asarray(getattr(ex, k)), np.asarray(getattr(f77, k)), rtol=2e-4)


def test_shrub_npp_on_the_cropland_share_is_booked():
    veg, carbon, summary = _state(); crop = jnp.asarray([0.0, 0.2, 0.5, 0.8, 1.0], jnp.float32)
    pp = eqx.tree_at(lambda p: p.exact_litterfall, default_pft_params(), True)
    d, b = _closure(dynaveg, pp, cropfrac=crop)
    np.testing.assert_allclose(d, b, atol=2e-4)                      # closes with cropland too
    diag = {}; dynaveg(veg, carbon, summary, pp, 1, 1e-12, cropfrac=crop, diag=diag)
    shrub = (np.asarray(summary.aynpp)[:, 8:10].sum(1)) * np.asarray(crop)
    np.testing.assert_allclose(np.asarray(diag['crop_shrub_npp']), shrub, rtol=1e-5, atol=1e-7)


# ------------------------------------------------------------ fire, spin-up
def test_leafless_stand_gets_no_closed_canopy_damping():
    plai = jnp.zeros((2, 12)).at[1, 0].set(2.0)
    np.testing.assert_allclose(np.asarray(broadleaf_share(plai)), [0.0, 1.0])
    from ibis.physics.fire import canopy_damping
    d = canopy_damping(jnp.asarray([0.95, 0.95]), FireParams(use_canopy_damping=True), plai)
    assert float(d[0]) == pytest.approx(1.0) and float(d[1]) < 0.3


def test_spinup_tail_and_tower_harness_wiring():
    from ibis.io.config_reader import PhysicsSection
    assert PhysicsSection().soilcspin_tail_years == 50
    root = pathlib.Path(__file__).resolve().parents[1]
    assert 'soilcspin_tail_years' in (root / 'scripts' / 'run_global.py').read_text()
    st = (root.parent / 'verify' / 'calibration' / 'site_transient.py').read_text()
    assert 'zero_fire_accumulators(state.vegetation)' in st and 'fire_params=c["sp"].fire' in st


def test_snow_constants_are_f77s():
    """initial.f:409-416: hsnomin = max(0.15, nsnolay hsnotop), fimin = 2e-5 (dtime/1800)(0.15/hsnomin), z0sno = 0.0005."""
    from ibis.state.snow import create_snow_state
    s = create_snow_state(3, 3)
    assert s.hsnomin == pytest.approx(0.15) and s.fimin == pytest.approx(4e-5) and s.z0sno == pytest.approx(5e-4)
    assert s.hsnomin >= 3 * s.hsnotop - 1e-9                            # vadapt's requirement


def test_spinup_tail_keeps_short_checks_accelerated():
    from ibis.io.config_reader import spinup_nsoilcall
    assert [spinup_nsoilcall(10, 1, i, 200, 50) for i in (0, 149, 150, 199)] == [10, 10, 1, 1]
    assert [spinup_nsoilcall(10, 1, i, 30, 50) for i in (0, 29)] == [10, 10]
    assert spinup_nsoilcall(10, 0, 0, 200, 50) == 1 and spinup_nsoilcall(1, 1, 0, 200, 50) == 1


def test_exact_booking_closes_with_negative_npp_dead_pfts_and_cropland():
    """The review's two cases: negative NPP (paid from the pools) and exist = 0 PFTs with positive NPP (to litter)."""
    pp = eqx.tree_at(lambda p: p.exact_litterfall, default_pft_params(), True)
    veg, carbon, summary = _state()
    rng = np.random.default_rng(1); npp = np.asarray(summary.aynpp).copy(); npp[:, [1, 5]] = -0.08
    ex = np.ones((5, 12), np.float32); ex[1:3, :8] = 0.0
    veg = _set(veg, exist=jnp.asarray(ex)); summary = _set(summary, aynpp=jnp.asarray(npp, jnp.float32))
    crop = jnp.asarray([0.0, 0.3, 0.0, 0.6, 0.0], jnp.float32)
    for fn, args, kw in ((biomass_update, [1e-12], {}), (dynaveg, [1, 1e-12], {'cropfrac': crop})):
        nv, nc, _ = fn(veg, carbon, summary, pp, *args, **kw)
        d = np.asarray(nv.cbiol + nv.cbiow + nv.cbior).sum(1) - np.asarray(veg.cbiol + veg.cbiow + veg.cbior).sum(1)
        b = npp.sum(1) - np.asarray(nc.falll + nc.fallw + nc.fallr) - np.asarray(nc.cdisturb)   # against the RAW total, as NEE counts it
        np.testing.assert_allclose(d, b, atol=3e-4)


def test_harvest_never_exceeds_the_grass_leaf_litter_under_exact_booking():
    from ibis.io.parameters import CropParams
    pp = eqx.tree_at(lambda p: p.exact_litterfall, default_pft_params(), True)
    veg, carbon, summary = _state()
    veg = _set(veg, cbiol=jnp.asarray(np.asarray(veg.cbiol) * np.r_[np.ones(10), 6.0, 6.0][None, :], jnp.float32))   # leaf pools far above equilibrium
    crop = jnp.full((5,), 0.9, jnp.float32); diag = {}
    nv, nc, _ = dynaveg(veg, carbon, summary, pp, 1, 1e-12, cropfrac=crop, crop_params=CropParams(harvest_index=0.9), diag=diag)
    assert np.all(np.asarray(nc.falll) >= -1e-7)
    d = np.asarray(nv.cbiol + nv.cbiow + nv.cbior).sum(1) - np.asarray(veg.cbiol + veg.cbiow + veg.cbior).sum(1)
    b = np.asarray(summary.aynpp).sum(1) - np.asarray(nc.falll + nc.fallw + nc.fallr) - np.asarray(nc.cdisturb)
    np.testing.assert_allclose(d, b, atol=3e-4)
