"""CALIBRATION.md 70: the cover-limit form of the tree survival gate. With tree_cover_limit the annual-mean stress sets
a water-limited MAXIMUM upper-canopy cover; the standing pools are never thinned, trees are eliminated (to litter) only
where that maximum is 0, and fu may not exceed it. Off (default) the survival-factor path is bit-exact."""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ibis.io.parameters import default_pft_params
from ibis.physics.vegetation_dynamics import dynaveg
from ibis.state.carbon import create_carbon_state
from ibis.state.summary import create_summary_state
from ibis.state.vegetation import create_vegetation_state


def _set(obj, **kw):
    for k, v in kw.items():
        obj = eqx.tree_at(lambda o, _k=k: getattr(o, _k), obj, v)
    return obj


def _state(npoi=4, npft=12):
    veg = create_vegetation_state(npoi, npft, 6)
    carbon = create_carbon_state(npoi)
    summary = create_summary_state(npoi, npft)
    veg = _set(veg, cbiol=jnp.full((npoi, npft), 0.4), cbiow=jnp.full((npoi, npft), 3.0),
               cbior=jnp.full((npoi, npft), 0.5), plai=jnp.full((npoi, npft), 1.0), exist=jnp.ones((npoi, npft)))
    summary = _set(summary, aynpp=jnp.full((npoi, npft), 0.3))
    return veg, carbon, summary


def _run(pp, ts=None, cmax=None):
    veg, carbon, summary = _state()
    kw = {}
    if ts is not None:
        kw['tree_surv'] = jnp.asarray(ts, jnp.float32)
    if cmax is not None:
        kw['tree_cover_max'] = jnp.asarray(cmax, jnp.float32)
    nv, nc, _ = dynaveg(veg, carbon, summary, pp, 0, 1e-12, **kw)
    live = np.asarray(nv.cbiol + nv.cbiow + nv.cbior)[:, :8].sum(1)
    fall = np.asarray(nc.falll + nc.fallw + nc.fallr)
    return nv, live, fall, np.asarray(nc.cdisturb)


def test_cover_limit_off_by_default_and_the_gate_path_is_bit_exact():
    pp = default_pft_params()
    assert pp.tree_cover_limit is False and (pp.tree_cover_min, pp.tree_cover_ramp, pp.tree_cover_power) == (0.30, 0.65, 1.75)
    from ibis.state.stress_accum import tree_cover_max
    from ibis.state.model_state import create_ibis_state
    from ibis.dimensions import GridConfig
    st = create_ibis_state(GridConfig(npoi=4, nbedrock=3), with_stress=True)
    assert tree_cover_max(st, pp) is None                                   # switch off: nothing passed to dynaveg
    nv1, l1, f1, d1 = _run(pp, ts=[1.0, 0.7, 0.4, 0.1])
    nv2, l2, f2, d2 = _run(pp, ts=[1.0, 0.7, 0.4, 0.1], cmax=None)
    np.testing.assert_array_equal(np.asarray(nv1.fu), np.asarray(nv2.fu)); np.testing.assert_array_equal(l1, l2)


def test_cover_limit_caps_cover_without_thinning_and_eliminates_at_zero():
    pp = eqx.tree_at(lambda p: (p.tree_cover_limit, p.fu_min), default_pft_params(), (True, 0.05))   # fu_min below the test caps
    ts = [1.0, 0.7, 0.4, 0.0]; cmax = [1.0, 0.5, 0.2, 0.0]
    nv_gate, l_gate, f_gate, d_gate = _run(pp, ts=ts)                        # the old path for comparison
    nv, live, fall, dist = _run(pp, ts=ts, cmax=cmax)
    fu = np.asarray(nv.fu)
    assert fu[0] == np.asarray(nv_gate.fu)[0]                                # fu_max 1: unchanged
    assert abs(fu[1] - 0.5) < 1e-6 and abs(fu[2] - 0.2) < 1e-6              # capped at the maximum cover
    assert live[1] == live[0] and live[2] == live[0]                         # no thinning in the partial range
    assert live[1] > l_gate[1] and live[2] > l_gate[2]                        # where the gate killed 30 / 60 %
    assert live[3] == 0.0 and fall[3] > fall[0]                              # eliminated where fu_max = 0, to litter
    np.testing.assert_allclose(live + fall + dist, (live + fall + dist)[0], rtol=1e-5)   # carbon conserved


def test_cover_limit_blocks_seeding_only_where_the_maximum_is_zero():
    pp = eqx.tree_at(lambda p: p.tree_cover_limit, default_pft_params(), True)
    veg, carbon, summary = _state()
    veg = _set(veg, cbiol=jnp.zeros((4, 12)), cbiow=jnp.zeros((4, 12)), cbior=jnp.zeros((4, 12)))
    nv, _, _ = dynaveg(veg, carbon, summary, pp, 0, 1e-12, tree_surv=jnp.asarray([1.0, 0.5, 0.5, 0.0], jnp.float32),
                       tree_cover_max=jnp.asarray([1.0, 0.3, 0.0, 0.0], jnp.float32))
    seeded = np.asarray(nv.cbiol)[:, :8].sum(1) > 0
    assert seeded.tolist() == [True, True, False, False]


def test_cover_limit_ramp_and_wiring():
    """fu_max = clip((u - min) / ramp, 0, 1) ** power from the accumulator; both production callers and the site
    harness pass it next to the survival factor; the yaml loader reads the four keys."""
    from pathlib import Path
    from ibis.state.stress_accum import tree_cover_max
    from ibis.state.model_state import create_ibis_state
    from ibis.dimensions import GridConfig
    pp = eqx.tree_at(lambda p: p.tree_cover_limit, default_pft_params(), True)
    st = create_ibis_state(GridConfig(npoi=3, nbedrock=3), with_stress=True)
    st = eqx.tree_at(lambda s: (s.stress.aystressu, s.stress.nday), st,
                     (jnp.asarray([0.2, 0.6, 1.0]) * 100.0, jnp.full((3,), 100.0)))
    got = np.asarray(tree_cover_max(st, pp))
    want = np.clip((np.array([0.2, 0.6, 1.0]) - 0.30) / 0.65, 0, 1) ** 1.75
    np.testing.assert_allclose(got, want, rtol=1e-5)
    root = Path(__file__).resolve().parents[1]
    for rel in ("scripts/run_global.py", "ibis/timestepping/loops_gswp3.py"):
        txt = (root / rel).read_text()
        assert "tree_cover_max=tree_cover_max(state, pft_params)," in txt, rel
    st_ = (root.parent / "verify" / "calibration" / "site_transient.py").read_text()
    assert "tree_cover_max=tree_cover_max(state, c[\"pft\"])" in st_
    pl = (root / "ibis" / "io" / "physics_params.py").read_text()
    for k in ("tree_cover_limit", "tree_cover_min", "tree_cover_ramp", "tree_cover_power"):
        assert f"'{k}'" in pl, k


def test_cover_limit_ramp_per_zone_through_the_types():
    """CALIBRATION.md 71: a zone's tree types may carry tree_cover_min / ramp / power; apply_types turns them into
    per-cell arrays (zones without values keep the base ramp) and tree_cover_max uses them."""
    from ibis.io import pft_types as P
    from ibis.io.physics_params import build_all_params
    from ibis.state.stress_accum import tree_cover_max
    from ibis.state.model_state import create_ibis_state
    from ibis.dimensions import GridConfig
    params, pp, _ = build_all_params(3, np.zeros(3))
    z = np.array([P.ZI['D_temperate'], P.ZI['A_humid'], P.ZI['B_hot']])
    over = P.parse_type_overrides({'T16': {'tree_cover_min': 0.72, 'tree_cover_ramp': 0.18, 'tree_cover_power': 1.0},
                                   'T01': {'tree_cover_min': 0.34, 'tree_cover_ramp': 0.32, 'tree_cover_power': 2.5}})
    _, pf = P.apply_types(params, pp, z, over)
    np.testing.assert_allclose(np.asarray(pf.tree_cover_min), [0.72, 0.34, 0.30], rtol=1e-6)      # B_hot: the base
    np.testing.assert_allclose(np.asarray(pf.tree_cover_ramp), [0.18, 0.32, 0.65], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(pf.tree_cover_power), [1.0, 2.5, 1.75], rtol=1e-6)
    pf = eqx.tree_at(lambda p: p.tree_cover_limit, pf, True)
    st = create_ibis_state(GridConfig(npoi=3, nbedrock=3), with_stress=True)
    st = eqx.tree_at(lambda s: (s.stress.aystressu, s.stress.nday), st, (jnp.asarray([0.81, 0.70, 0.81]) * 50.0, jnp.full((3,), 50.0)))
    got = np.asarray(tree_cover_max(st, pf))
    want = [0.5, 1.0, ((0.81 - 0.30) / 0.65) ** 1.75]
    np.testing.assert_allclose(got, want, rtol=1e-4)
    # the scalar path is untouched: no overrides -> scalars stay scalars
    _, pf0 = P.apply_types(params, pp, z, {})
    assert np.ndim(np.asarray(pf0.tree_cover_min)) == 0
