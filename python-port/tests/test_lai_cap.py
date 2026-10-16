"""Per-type maximum leaf area (docs/CALIBRATION.md 46): off by default, an infinite cap changes nothing, a finite
cap holds each canopy layer's LAI per unit cover to it, and the leaf carbon it removes stays in the plant."""
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
from ibis.io.parameters import CropParams, default_pft_params
from ibis.physics.vegetation_dynamics import dynaveg, biomass_update
from ibis.state.model_state import create_ibis_state
from ibis.dimensions import GridConfig


def _state(npoi=5):
    st = create_ibis_state(GridConfig(npoi=npoi, nbedrock=0))
    k = np.random.default_rng(3)
    cbiol = jnp.asarray(k.uniform(0.0, 0.6, (npoi, 12)), dtype=jnp.float32)
    cbiow = jnp.asarray(k.uniform(0.0, 6.0, (npoi, 12)), dtype=jnp.float32).at[:, 10:].set(0.0)
    cbior = jnp.asarray(k.uniform(0.0, 0.5, (npoi, 12)), dtype=jnp.float32)
    aynpp = jnp.asarray(k.uniform(0.0, 0.6, (npoi, 12)), dtype=jnp.float32)
    return eqx.tree_at(lambda s: (s.vegetation.cbiol, s.vegetation.cbiow, s.vegetation.cbior, s.vegetation.exist, s.summary.aynpp),
                       st, (cbiol, cbiow, cbior, jnp.ones((npoi, 12)), aynpp))


def _pp(cap=None, **kw):
    pp = default_pft_params()
    if cap is not None:
        pp = eqx.tree_at(lambda p: p.lai_max, pp, jnp.broadcast_to(jnp.asarray(cap, jnp.float32), (12,)),
                         is_leaf=lambda x: x is None)
    return eqx.tree_at(lambda p: tuple(getattr(p, f) for f in kw), pp, tuple(kw.values())) if kw else pp


def _dyn(st, pp, cf=None, cp=None):
    return dynaveg(st.vegetation, st.carbon, st.summary, pp, 0, 1e-7, cropfrac=cf, crop_params=cp)


def _same(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def test_off_by_default_and_infinite_cap_is_identical():
    st = _state()
    assert default_pft_params().lai_max is None
    cf = jnp.array([0.0, 0.3, 0.0, 0.8, 0.0])
    assert _same(_dyn(st, _pp()), _dyn(st, _pp(np.inf)))
    assert _same(_dyn(st, _pp(), cf, CropParams()), _dyn(st, _pp(np.inf), cf, CropParams()))
    a = biomass_update(st.vegetation, st.carbon, st.summary, _pp(), 1e-7)
    b = biomass_update(st.vegetation, st.carbon, st.summary, _pp(np.inf), 1e-7)
    assert _same(a, b)


def _layer_lai(v):
    return np.asarray(v.totlaiu / v.fu), np.asarray(v.totlail / v.fl)


def test_cap_binds_and_conserves_plant_carbon():
    st = _state()
    for kw in ({}, {'cover_k': 0.5}):                                    # F77 linear lower cover, and the saturating one
        v0, c0, _ = _dyn(st, _pp(**kw))
        v1, c1, _ = _dyn(st, _pp(2.0, **kw))
        lu0, ll0 = _layer_lai(v0); lu1, ll1 = _layer_lai(v1)
        assert (lu0 > 2.5).any() and (ll0 > 2.5).any()                   # the test state does exceed the cap
        assert (lu1 <= 2.0 * (1 + 1e-4)).all() and (ll1 <= 2.0 * (1 + 1e-4)).all()
        assert (np.maximum(lu0, 2.0) - lu1 > -1e-4).all()                # never raises a layer's LAI
        tot0 = np.asarray((v0.cbiol + v0.cbiow + v0.cbior).sum(axis=1)); tot1 = np.asarray((v1.cbiol + v1.cbiow + v1.cbior).sum(axis=1))
        assert np.allclose(tot0, tot1, rtol=1e-5)                         # the removed leaf carbon went to wood and roots
        assert np.allclose(np.asarray(c0.falll), np.asarray(c1.falll))    # this year's litter is untouched
        g = 11; assert float(v1.cbiow[:, g].sum()) == 0.0                 # C3 grass has no wood: its excess goes to roots
        assert (np.asarray(v1.fu) >= np.asarray(v0.fu) - 1e-7).all()     # wood can only grow, so does the upper cover
    b0 = biomass_update(st.vegetation, st.carbon, st.summary, _pp(), 1e-7)[0]
    b1 = biomass_update(st.vegetation, st.carbon, st.summary, _pp(2.0), 1e-7)[0]
    lu, ll = _layer_lai(b1)
    assert (lu <= 2.0 * (1 + 1e-4)).all() and (ll <= 2.0 * (1 + 1e-4)).all() and (np.asarray(b1.lai) <= 2.0 * (1 + 1e-4)).all()
    assert np.allclose(np.asarray(b0.biomass.sum(axis=1)), np.asarray(b1.biomass.sum(axis=1)), rtol=1e-5)


def test_layer_cap_is_the_leaf_area_weighted_mean_and_crops_blend():
    st = _state()
    cap = np.array([2.0] * 4 + [6.0] * 4 + [1.5] * 4, dtype=np.float32)
    v0, _, _ = _dyn(st, _pp()); v1, _, _ = _dyn(st, _pp(cap))
    p0 = np.asarray(v0.plai)
    want = (p0[:, :8] * cap[:8]).sum(1) / p0[:, :8].sum(1)               # the weights are the pre-cap leaf areas
    lu1 = np.asarray(v1.totlaiu / v1.fu)
    bind = np.asarray(v0.totlaiu / v0.fu) > want
    assert bind.any() and np.allclose(lu1[bind], want[bind], rtol=1e-3)
    cf = jnp.array([0.0, 1.0, 0.0, 1.0, 0.5])
    va, _, _ = _dyn(st, _pp(1.5), cf, CropParams())                      # crops take the grass cap
    vb, _, _ = _dyn(st, _pp(1.5), cf, CropParams(lai_max=4.0))          # ... unless they have their own
    lla = np.asarray(va.totlail / va.fl); llb = np.asarray(vb.totlail / vb.fl)
    assert (lla <= 1.5 * (1 + 1e-4)).all() and llb[1] > lla[1] and llb[1] <= 4.0 * (1 + 1e-4)
    assert np.allclose(lla[[0, 2]], llb[[0, 2]])                          # no cropland: no change


def test_yaml_and_types():
    from ibis.io.physics_params import build_pft_params
    from ibis.io import pft_types as T
    assert build_pft_params({'pft': {'tropical_broadleaf_evergreen': {'specla': 13.0}}}).lai_max is None
    pp = build_pft_params({'pft': {'tropical_broadleaf_evergreen': {'lai_max': 7.0}, 'c3_grass': {'lai_max': 3.0}}})
    assert float(pp.lai_max[0]) == 7.0 and float(pp.lai_max[11]) == 3.0 and np.isinf(np.asarray(pp.lai_max[1:11])).all()
    from ibis.io.physics_params import build_all_params
    zi = np.array([T.ZI[z] for z in ('A_humid', 'C_med', 'D_subarctic', 'B_hot')])
    sp, _, _ = build_all_params(len(zi), np.zeros(len(zi)))
    _, pf = T.apply_types(sp, default_pft_params(), zi)
    assert pf.lai_max is None                                             # nobody set it: stays off
    tid = T.TYPES[0][0]
    _, pf = T.apply_types(sp, default_pft_params(), zi, T.parse_type_overrides({tid: {'lai_max': 5.5}}))
    m = np.asarray(T.slot_map(zi)) == 0
    assert np.asarray(pf.lai_max).shape == (len(zi), 12) and np.allclose(np.asarray(pf.lai_max)[m], 5.5)
    assert np.isinf(np.asarray(pf.lai_max)[~m]).all()
