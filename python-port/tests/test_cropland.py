"""Cropland as managed grass (docs/CALIBRATION.md 41): the tree cap, the harvest, the daily capacity factor, and
the F77 path being untouched when there is no cropland."""
import numpy as np, jax, jax.numpy as jnp, pytest
from ibis.io.parameters import CropParams, default_pft_params
from ibis.physics.vegetation_dynamics import dynaveg
from ibis.state.model_state import create_ibis_state
from ibis.dimensions import GridConfig


def _state(npoi=4):
    st = create_ibis_state(GridConfig(npoi=npoi, nbedrock=0))
    v = st.vegetation
    cbiol = jnp.full((npoi, 12), 0.05); cbiow = jnp.zeros((npoi, 12)).at[:, 4].set(8.0); cbior = jnp.full((npoi, 12), 0.05)
    cbiol = cbiol.at[:, 11].set(0.4)                                           # a C3 grass understorey with a real leaf pool
    aynpp = jnp.full((npoi, 12), 0.05).at[:, 4].set(0.4)
    st = jax.tree.map(lambda x: x, st)
    import equinox as eqx
    st = eqx.tree_at(lambda s: (s.vegetation.cbiol, s.vegetation.cbiow, s.vegetation.cbior, s.vegetation.exist, s.summary.aynpp),
                     st, (cbiol, cbiow, cbior, jnp.ones((npoi, 12)), aynpp))
    return st


def _run(st, cropfrac, cp=None, diag=None):
    return dynaveg(st.vegetation, st.carbon, st.summary, default_pft_params(), 0, 1e-7, cropfrac=cropfrac, crop_params=cp, diag=diag)


def test_no_cropland_is_the_old_path():
    st = _state()
    v0, c0, s0 = _run(st, None); v1, c1, s1 = _run(st, jnp.zeros(4))
    assert np.allclose(np.asarray(v0.cbiow), np.asarray(v1.cbiow)) and np.allclose(np.asarray(v0.fu), np.asarray(v1.fu))
    assert np.allclose(np.asarray(s0.ayneetot), np.asarray(s1.ayneetot))


def test_tree_cover_is_the_natural_share_times_the_natural_cover():
    st = _state(); pp = default_pft_params()
    cf = jnp.array([0.0, 0.5, 0.9, 0.99]); diag = {}
    v, c, s = _run(st, cf, diag=diag)
    v0, _, _ = _run(st, None)
    fu = np.asarray(v.fu); fu0 = np.asarray(v0.fu); wood = np.asarray(v.cbiow[:, :8].sum(axis=1)); wood0 = np.asarray(v0.cbiow[:, :8].sum(axis=1))
    assert np.allclose(wood, wood0)                                             # no clearing: the tree pools are untouched
    shrub = np.asarray(v.cbiol[:, 8:10].sum(axis=1)); shrub0 = np.asarray(v0.cbiol[:, 8:10].sum(axis=1))
    assert shrub[0] == shrub0[0] and shrub[3] < shrub0[3]                      # shrubs grow only on the natural share
    assert abs(fu[0] - fu0[0]) < 1e-6                                            # no cropland: the old cover
    assert fu[1] <= 0.5 + 1e-3 and fu[2] <= max(0.1, float(pp.fu_min)) + 1e-3   # cover within the non-cropland share (fu_min is the F77 floor)
    assert float(np.asarray(diag["clearing"]).sum()) == 0.0


def test_harvest_removes_the_cropland_share_of_grass_turnover_from_litter():
    st = _state(); pp = default_pft_params(); cp = CropParams(harvest_index=0.5)
    cf = jnp.array([0.0, 1.0, 0.0, 1.0]); diag = {}
    v0, c0, s0 = _run(st, jnp.zeros(4), cp)
    v1, c1, s1 = _run(st, cf, cp, diag=diag)
    h = np.asarray(diag["harvest"]); expected = 0.5 * float(st.vegetation.cbiol[0, 11]) / float(pp.tauleaf[11]) + 0.5 * float(st.vegetation.cbiol[0, 10]) / float(pp.tauleaf[10])
    assert h[0] == 0.0 and abs(h[1] - expected) < 1e-6
    # the harvested carbon leaves the ecosystem: NEE lower by the harvest (+ the clearing) on the cropland cells
    d = np.asarray(s0.ayneetot - s1.ayneetot); assert d[0] == 0.0 and d[1] >= h[1] - 2e-3     # minus a little: the shrubs on the cropland cell shrink, so their background disturbance does too


def test_daily_capacity_factor():
    from ibis.timestepping.loops import crop_vfac
    class P: pass
    p = P(); p.cropfrac = jnp.array([0.0, 0.5, 1.0]); p.crop = CropParams(vmax_scale_c3=2.0, vmax_scale_c4=3.0)
    g = np.asarray(crop_vfac(p, None, 12))
    assert g.shape == (3, 12) and np.allclose(g[:, :10], 1.0) and np.allclose(g[:, 11], [1.0, 1.5, 2.0]) and np.allclose(g[:, 10], [1.0, 2.0, 3.0])
    p.cropfrac = jnp.array([0.0, 1.0]); p.crop = CropParams(vmax_scale_c3=0.4, vmax_scale_c4=1.2, c4_share=0.25)     # the global setting: both grasses get the crop mix
    g = np.asarray(crop_vfac(p, None, 12)); mix = 0.75 * 0.4 + 0.25 * 1.2
    assert np.allclose(g[1, 10], mix) and np.allclose(g[1, 11], mix) and np.allclose(g[0, 10:], 1.0)
    p.cropfrac = None; assert crop_vfac(p, None, 12) is None
