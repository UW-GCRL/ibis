"""CALIBRATION.md 60-61: the water-stress survival gate must conserve carbon -- the tree carbon it removes goes to
litter (tree_kill_to_litter, default True); False reproduces the earlier runs, where it left the model unbooked."""
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


def _budget(veg, carbon, summary, ts, pp):
    nv, nc, _ = dynaveg(veg, carbon, summary, pp, 0, 1e-12, tree_surv=jnp.asarray(ts, jnp.float32))
    live = np.asarray(nv.cbiol + nv.cbiow + nv.cbior).sum(1)
    fall = np.asarray(nc.falll + nc.fallw + nc.fallr)
    return live, fall, np.asarray(nc.cdisturb)


def test_gate_conserves_carbon_by_default():
    pp = default_pft_params()
    assert pp.tree_kill_to_litter is True
    veg, carbon, summary = _state()
    l1, f1, d1 = _budget(veg, carbon, summary, [1.0, 1.0, 1.0, 1.0], pp)
    l2, f2, d2 = _budget(veg, carbon, summary, [1.0, 0.7, 0.4, 0.1], pp)
    np.testing.assert_allclose(l2 + f2 + d2, l1 + f1 + d1, rtol=1e-5)    # live + litterfall + disturbance unchanged
    assert np.all(np.diff(f2) > 0) and np.all(l2[1:] < l1[1:])            # the dead trees went to litter


def test_switch_off_reproduces_the_unbooked_removal():
    pp = eqx.tree_at(lambda p: p.tree_kill_to_litter, default_pft_params(), False)
    veg, carbon, summary = _state()
    l1, f1, d1 = _budget(veg, carbon, summary, [1.0, 1.0, 1.0, 1.0], pp)
    l2, f2, d2 = _budget(veg, carbon, summary, [1.0, 0.7, 0.4, 0.1], pp)
    np.testing.assert_allclose(f2, f1, rtol=1e-6)                          # litterfall ignores the gate
    assert np.all((l2 + f2 + d2)[1:] < (l1 + f1 + d1)[1:])                 # carbon lost
