"""F7 (CALIBRATION.md 59): closed-canopy fire damping, the annual reset of the fire counters in the production runner,
and the parameter-file 'fire' section."""
from __future__ import annotations

import pathlib

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.io.parameters import default_pft_params
from ibis.io.physics_params import build_fire_params
from ibis.ml.fire_calibration import FireBatch, predict_burned_fraction
from ibis.physics.fire import (
    SEASONLENGTH, FireParams, canopy_damping, zero_fire_accumulators,
)
from ibis.physics.vegetation_dynamics import dynaveg
from ibis.state.carbon import create_carbon_state
from ibis.state.summary import create_summary_state
from ibis.state.vegetation import create_vegetation_state


def _set(obj, **kw):
    for k, v in kw.items():
        obj = eqx.tree_at(lambda o, _k=k: getattr(o, _k), obj, v)
    return obj


BL = jnp.asarray([1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1.])     # broadleaf upper positions + lower canopy
CON = jnp.asarray([0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1.])    # conifer upper positions + lower canopy


def _burning_state(fu, npoi=4, npft=12, form=None):
    veg = create_vegetation_state(npoi, npft, 6)
    carbon = create_carbon_state(npoi)
    summary = create_summary_state(npoi, npft)
    nyt = 17520.0
    veg = _set(veg, nytimes=jnp.asarray(nyt),
               nfirestep_vpd=jnp.full((npoi,), 0.45 * nyt), sdry_vpd=jnp.full((npoi,), 0.45 * nyt * 0.8),
               nfirestep=jnp.full((npoi,), 0.45 * nyt), sdry=jnp.full((npoi,), 0.45 * nyt * 0.8),
               cbiol=jnp.full((npoi, npft), 0.15), cbiow=jnp.full((npoi, npft), 1.20),
               cbior=jnp.full((npoi, npft), 0.10), plai=jnp.full((npoi, npft), 0.50) * (BL if form is None else form)[None, :],
               exist=jnp.ones((npoi, npft)), fu=jnp.asarray(fu, jnp.float32))
    carbon = _set(carbon, totlit=jnp.full((npoi,), 1.0), totalit=jnp.full((npoi,), 0.6))
    return veg, carbon, summary


def test_damping_is_a_decreasing_logistic_in_tree_cover():
    p = FireParams()
    fu = jnp.linspace(0.0, 1.0, 101)
    d = np.asarray(canopy_damping(fu, p))
    assert np.all(np.diff(d) < 0)
    assert d[0] > 0.99 and d[-1] < 0.01
    assert float(canopy_damping(jnp.asarray([p.fu_half]), p)[0]) == pytest.approx(0.5, abs=1e-6)


def test_off_by_default_and_off_changes_nothing_in_dynaveg():
    assert FireParams().use_canopy_damping is False
    veg, carbon, summary = _burning_state([0.1, 0.5, 0.8, 0.95])
    a = dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12, fire_scheme=SEASONLENGTH)[0].disturbf
    b = dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12, fire_scheme=SEASONLENGTH,
                fire_params=FireParams())[0].disturbf
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert float(jnp.max(a)) > 0.05                     # the fixture burns


def test_damping_cuts_fire_under_closed_canopy_only():
    fu = [0.1, 0.5, 0.8, 0.95]
    veg, carbon, summary = _burning_state(fu)
    off = np.asarray(dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12, fire_scheme=SEASONLENGTH,
                             fire_params=FireParams())[0].disturbf)
    on = np.asarray(dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12, fire_scheme=SEASONLENGTH,
                            fire_params=FireParams(use_canopy_damping=True))[0].disturbf)
    assert on[0] == pytest.approx(off[0], rel=0.02)    # open land: unchanged
    assert on[2] < 0.2 * off[2] and on[3] < 0.05 * off[3]
    assert np.all(on <= off + 1e-7)


def test_closed_conifer_stand_keeps_its_fire_unless_all_canopies_are_damped():
    fu = [0.9, 0.9, 0.9, 0.9]
    veg, carbon, summary = _burning_state(fu, form=CON)
    run = lambda fp: np.asarray(dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12, fire_scheme=SEASONLENGTH,
                                        fire_params=fp)[0].disturbf)
    off = run(FireParams()); bl_only = run(FireParams(use_canopy_damping=True))
    every = run(FireParams(use_canopy_damping=True, damp_broadleaf_only=False))
    np.testing.assert_allclose(bl_only, off, rtol=1e-6)
    assert np.all(every < 0.1 * off)


def test_offline_predictor_applies_the_same_damping():
    n = 3
    batch = FireBatch(nfirestep=jnp.full((n,), 4000.0), sdry=jnp.full((n,), 2800.0), nytimes=jnp.asarray(8760.0),
                      totlit=jnp.full((n,), 1.0), totalit=jnp.full((n,), 0.6), cbiol=jnp.full((n, 12), 0.1),
                      plai=jnp.full((n, 12), 0.5) * BL[None, :], popd=jnp.zeros((n,)), fu=jnp.asarray([0.2, 0.6, 0.9]))
    p0, p1 = FireParams(), FireParams(use_canopy_damping=True)
    b0, b1 = np.asarray(predict_burned_fraction(p0, batch)), np.asarray(predict_burned_fraction(p1, batch))
    assert b1[0] == pytest.approx(b0[0], rel=0.02) and b1[2] < 0.1 * b0[2]
    with pytest.raises(ValueError):
        predict_burned_fraction(p1, batch._replace(fu=None))


def test_per_cell_mortality_from_the_types_equals_the_per_pft_vector():
    """apply_types gathers FireParams.mort to (npoi, 12); dynaveg must accept it (the 2026-10-25 rerun crashed on it)."""
    veg, carbon, summary = _burning_state([0.2, 0.4, 0.6, 0.8])
    p1 = FireParams()
    p2 = eqx.tree_at(lambda q: q.mort, p1, jnp.tile(p1.mort[None, :], (4, 1)))
    a = dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12, fire_scheme=SEASONLENGTH, fire_params=p1)
    b = dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12, fire_scheme=SEASONLENGTH, fire_params=p2)
    for x, y in ((a[0].cbiow, b[0].cbiow), (a[0].fu, b[0].fu), (a[2].ayneetot, b[2].ayneetot)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6)


def test_zero_fire_accumulators_clears_the_five_counters_only():
    veg, _, _ = _burning_state([0.3, 0.3, 0.3, 0.3])
    z = zero_fire_accumulators(veg)
    for k in ('nytimes', 'nfirestep', 'sdry', 'nfirestep_vpd', 'sdry_vpd'):
        assert float(jnp.max(jnp.abs(getattr(z, k)))) == 0.0
    np.testing.assert_array_equal(np.asarray(z.cbiow), np.asarray(veg.cbiow))
    np.testing.assert_array_equal(np.asarray(z.fu), np.asarray(veg.fu))


def test_run_global_resets_the_counters_each_year_and_passes_fire_params():
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_global.py").read_text()
    assert "zero_fire_accumulators(state.vegetation)" in src
    assert "fire_params=params.fire" in src
    from ibis.io.config_reader import PhysicsSection
    assert PhysicsSection().fire_annual_reset is True


def test_fire_section_of_the_parameter_file():
    assert build_fire_params({}) == FireParams()
    p = build_fire_params({'fire': {'k_b': 1.5, 'use_canopy_damping': True, 'fu_half': 0.55}})
    assert p.k_b == 1.5 and p.use_canopy_damping is True and p.fu_half == 0.55
    assert build_fire_params({'fire': {'damp_broadleaf_only': False}}).damp_broadleaf_only is False
    with pytest.raises(KeyError):
        build_fire_params({'fire': {'kb': 1.0}})


def test_closed_canopy_climate_damps_whatever_the_cover():
    """CALIBRATION.md 65: in a closed-canopy climate zone the damping is that of a closed canopy, so an open cell
    (the cold-start fire trap) is damped too; elsewhere the cover decides."""
    from ibis.physics.fire import canopy_damping
    p = FireParams(use_canopy_damping=True)
    fu = jnp.asarray([0.2, 0.2, 0.95]); plai = jnp.full((3, 12), 0.5) * BL[None, :]
    closed = jnp.asarray([1.0, 0.0, 0.0])
    d = np.asarray(canopy_damping(fu, p, plai, closed=closed))
    d_closed = float(canopy_damping(jnp.asarray([p.closed_canopy_fu]), p, plai[:1])[0])
    assert d[0] == pytest.approx(d_closed) and d[1] > 0.99 and d[2] < 0.05            # flagged open cell = closed; unflagged open cell undamped
    np.testing.assert_allclose(np.asarray(canopy_damping(fu, p, plai)), np.asarray(canopy_damping(fu, p, plai, closed=None)))


def test_closed_canopy_zones_reach_the_cells_through_apply_types():
    from ibis.io.physics_params import build_fire_params, build_all_params
    from ibis.io.pft_types import apply_types, ZONES
    p = build_fire_params({'fire': {'closed_canopy_zones': ['A_humid'], 'use_canopy_damping': True}})
    assert p.closed_canopy_zones == ('A_humid',) and p.closed_canopy_cell is None
    with pytest.raises(KeyError):
        bad, badp, _ = build_all_params(2, jnp.zeros(2), {'fire': {'closed_canopy_zones': ['A_wet']}}); apply_types(bad, badp, jnp.asarray([0, 1]))
    params, pft, _ = build_all_params(4, jnp.zeros(4), {'fire': {'closed_canopy_zones': ['A_humid'], 'use_canopy_damping': True}})
    zone = jnp.asarray([ZONES.index('A_humid'), ZONES.index('A_savanna'), ZONES.index('D_temperate'), ZONES.index('A_humid')])
    sp, _ = apply_types(params, pft, zone)
    np.testing.assert_array_equal(np.asarray(sp.fire.closed_canopy_cell), [1.0, 0.0, 0.0, 1.0])
    assert sp.fire.closed_canopy_zones == ('A_humid',)


def test_dynaveg_damps_an_open_cell_in_a_closed_canopy_climate():
    veg, carbon, summary = _burning_state([0.3, 0.3, 0.3, 0.3])
    p0 = FireParams(use_canopy_damping=True)
    p1 = eqx.tree_at(lambda q: q.closed_canopy_cell, p0, jnp.asarray([1.0, 0.0, 1.0, 0.0]), is_leaf=lambda x: x is None)
    a = np.asarray(dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12, fire_scheme=SEASONLENGTH, fire_params=p0)[0].disturbf)
    b = np.asarray(dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12, fire_scheme=SEASONLENGTH, fire_params=p1)[0].disturbf)
    assert b[0] < 0.15 * a[0] and b[2] < 0.15 * a[2] and b[1] == pytest.approx(a[1]) and b[3] == pytest.approx(a[3])
