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


def test_tree_cap_keeps_trees_off_the_cropland_share():
    st = _state(); pp = default_pft_params()
    cf = jnp.array([0.0, 0.5, 0.9, 1.0]); diag = {}
    v, c, s = _run(st, cf, diag=diag)
    fu = np.asarray(v.fu); wood = np.asarray(v.cbiow[:, :8].sum(axis=1))
    assert fu[0] > 0.9                                                         # a full forest cell keeps its canopy
    assert fu[1] <= 0.5 + 1e-3 and fu[2] <= max(0.1, float(pp.fu_min)) + 1e-3  # trees within the non-cropland share (fu_min floor is the F77 cover floor)
    assert wood[3] < 1e-3                                                      # full cropland: no trees
    assert np.asarray(diag["clearing"])[3] > np.asarray(diag["clearing"])[1] > np.asarray(diag["clearing"])[0] == 0.0


def test_harvest_removes_the_cropland_share_of_grass_turnover_from_litter():
    st = _state(); pp = default_pft_params(); cp = CropParams(harvest_index=0.5)
    cf = jnp.array([0.0, 1.0, 0.0, 1.0]); diag = {}
    v0, c0, s0 = _run(st, jnp.zeros(4), cp)
    v1, c1, s1 = _run(st, cf, cp, diag=diag)
    h = np.asarray(diag["harvest"]); expected = 0.5 * float(st.vegetation.cbiol[0, 11]) / float(pp.tauleaf[11]) + 0.5 * float(st.vegetation.cbiol[0, 10]) / float(pp.tauleaf[10])
    assert h[0] == 0.0 and abs(h[1] - expected) < 1e-6
    # the harvested carbon leaves the ecosystem: NEE lower by the harvest (+ the clearing) on the cropland cells
    d = np.asarray(s0.ayneetot - s1.ayneetot); assert d[0] == 0.0 and d[1] >= h[1] - 1e-6


def test_daily_capacity_factor():
    from ibis.timestepping.loops import crop_vfac
    class P: pass
    p = P(); p.cropfrac = jnp.array([0.0, 0.5, 1.0]); p.crop = CropParams(vmax_scale=2.0)
    g = np.asarray(crop_vfac(p, None, 12))
    assert g.shape == (3, 12) and np.allclose(g[:, :10], 1.0) and np.allclose(g[:, 11], [1.0, 1.5, 2.0]) and np.allclose(g[:, 10], [1.0, 1.5, 2.0])
    p.cropfrac = None; assert crop_vfac(p, None, 12) is None
