"""Stage F6: gradient-based calibration of the fire scheme.

Every other global fire model tunes its dozen-odd empirical parameters by hand,
because those models are not differentiable. IBIS-JAX is, so they can be fitted.
These tests pin the machinery: gradients reach the parameters, structural
switches are excluded, and a synthetic fit recovers a known answer.
"""

import os

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.ml.fire_calibration import (
    FIXED_PARAMS, PARAM_BOUNDS, FireBatch, _trainable, fire_loss,
    fit_fire_params, from_unconstrained, predict_burned_fraction,
    to_unconstrained,
)
from ibis.physics.fire import FireParams

GFED5_DIR = "/shared/nas2/Hangkai/ibis/inputs/fire/gfed5"


def _batch(n=96, seed=0):
    rng = np.random.default_rng(seed)
    return FireBatch(
        nfirestep=jnp.asarray(rng.uniform(0, 250, n)),
        sdry=jnp.asarray(rng.uniform(0, 200, n)),
        nytimes=jnp.array(365.0),
        totlit=jnp.asarray(rng.uniform(0.05, 0.8, n)),
        totalit=jnp.asarray(rng.uniform(0.05, 0.6, n)),
        cbiol=jnp.asarray(rng.uniform(0.0, 0.10, (n, 12))),
        plai=jnp.asarray(rng.uniform(0.0, 3.0, (n, 12))),
        popd=jnp.asarray(rng.uniform(0.0, 500.0, n)),
    )


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_gradients_reach_every_trainable_parameter():
    b = _batch()
    p = FireParams(use_fine_fuel=True, use_human=True)
    tr, static = _trainable(p)
    obs = jnp.asarray(np.random.default_rng(1).uniform(0, 0.3, 96))

    g = jax.grad(lambda t: fire_loss(eqx.combine(t, static), b, obs))(tr)
    leaves = jax.tree_util.tree_leaves(g)
    assert len(leaves) >= 10
    assert all(np.all(np.isfinite(np.asarray(x))) for x in leaves)
    # the headline knobs must actually move the loss
    assert abs(float(g.k_b)) > 0.0
    assert abs(float(g.p)) > 0.0


def test_structural_switches_are_not_differentiated():
    """`use_*` flags are structural choices, not parameters."""
    tr, static = _trainable(FireParams(use_fine_fuel=True, use_human=True))
    flat = jax.tree_util.tree_leaves(tr)
    assert not any(isinstance(x, bool) for x in flat)
    assert static.use_fine_fuel is True and static.use_human is True


def test_mortality_is_not_identifiable_from_burned_area():
    """`mort` has zero gradient here, and that is correct.

    Per-PFT mortality controls what fire does to vegetation and how much carbon
    is emitted -- not how much of the cell burns. So it cannot be fitted against
    burned area alone; it needs a vegetation or emissions target. Recorded as a
    test so the zero is not later mistaken for a broken gradient.
    """
    b = _batch()
    p = FireParams(use_fine_fuel=True, use_human=True)
    tr, static = _trainable(p)
    obs = jnp.asarray(np.random.default_rng(2).uniform(0, 0.3, 96))
    g = jax.grad(lambda t: fire_loss(eqx.combine(t, static), b, obs))(tr)
    assert np.allclose(np.asarray(g.mort), 0.0)


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

def test_loss_is_zero_for_a_perfect_prediction():
    b = _batch()
    p = FireParams(use_fine_fuel=True, use_human=True)
    obs = predict_burned_fraction(p, b)
    assert float(fire_loss(p, b, obs)) == pytest.approx(0.0, abs=1e-18)


def test_loss_handles_zero_burn_cells():
    """Log space needs a floor; cells that never burn must stay finite."""
    b = _batch()
    obs = jnp.zeros(96)
    val = float(fire_loss(FireParams(), b, obs))
    assert np.isfinite(val)


def test_weights_are_normalised():
    b = _batch()
    obs = jnp.asarray(np.random.default_rng(3).uniform(0, 0.3, 96))
    p = FireParams()
    a = float(fire_loss(p, b, obs, weights=jnp.ones(96)))
    c = float(fire_loss(p, b, obs, weights=jnp.full((96,), 7.0)))
    assert a == pytest.approx(c, rel=1e-12)
    assert a == pytest.approx(float(fire_loss(p, b, obs)), rel=1e-12)


# ---------------------------------------------------------------------------
# It actually fits
# ---------------------------------------------------------------------------

def test_recovers_a_known_parameter_from_synthetic_data():
    """Generate observations from a known k_b, then fit it back."""
    b = _batch(seed=4)
    truth = FireParams(k_b=1.7)
    obs = predict_burned_fraction(truth, b)

    def loss_kb(k):
        return fire_loss(FireParams(k_b=k), b, obs)

    k = 0.5
    opt_state = None
    import optax
    opt = optax.adam(5e-2)
    opt_state = opt.init(k)
    for _ in range(600):
        gval = jax.grad(loss_kb)(k)
        upd, opt_state = opt.update(gval, opt_state)
        k = float(optax.apply_updates(k, upd))

    assert k == pytest.approx(1.7, rel=0.02), f"recovered k_b = {k}"


def test_fit_reduces_the_loss_substantially():
    b = _batch(seed=5)
    truth = FireParams(k_b=1.2, p=1.6, use_fine_fuel=True)
    obs = predict_burned_fraction(truth, b)

    start = FireParams(k_b=0.4, p=0.8, use_fine_fuel=True)
    res = fit_fire_params(start, b, obs, steps=400, learning_rate=2e-2)

    assert len(res.loss_history) == 400
    assert np.all(np.isfinite(res.loss_history))
    assert res.loss_history[-1] < res.loss_history[0] / 50.0, (
        f"loss {res.loss_history[0]:.4g} -> {res.loss_history[-1]:.4g}")
    # and the fitted parameters predict the observations well
    pred = np.asarray(predict_burned_fraction(res.params, b))
    assert np.allclose(pred, np.asarray(obs), atol=2e-3)


def test_fit_leaves_structural_switches_alone():
    b = _batch(seed=6)
    start = FireParams(use_fine_fuel=True, use_human=False)
    obs = predict_burned_fraction(start, b)
    res = fit_fire_params(start, b, obs, steps=20)
    assert res.params.use_fine_fuel is True
    assert res.params.use_human is False


# ---------------------------------------------------------------------------
# The real target
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(GFED5_DIR), reason="GFED5 not present")
def test_gfed5_annual_totals_are_plausible():
    """GFED5's published global mean is ~774 Mha/yr, with a declining trend."""
    from ibis.io.gfed5 import global_burned_area_mha
    tot_2015 = global_burned_area_mha(GFED5_DIR, 2015, "Total")
    nat_2015 = global_burned_area_mha(GFED5_DIR, 2015, "Norm")
    assert 500.0 < tot_2015 < 1100.0
    assert nat_2015 < tot_2015, "natural-vegetation fire is a subset of the total"
    # cropland/deforestation/peat are ~10-15% of global burned area
    assert 0.80 < nat_2015 / tot_2015 < 0.95


@pytest.mark.skipif(not os.path.isdir(GFED5_DIR), reason="GFED5 not present")
def test_gfed5_fraction_is_bounded_and_conservative():
    from ibis.io.gfed5 import read_gfed5_annual_fraction
    mlat = np.arange(-89.75, 90.0, 0.5)
    mlon = np.arange(-179.75, 180.0, 0.5)
    li = np.arange(60, 300, 7)
    lo = np.arange(100, 700, 17)[: len(li)]
    f = read_gfed5_annual_fraction(GFED5_DIR, 2015, li, lo, mlat, mlon)
    assert f.shape == li.shape
    assert np.all(np.isfinite(f))
    assert np.all((f >= 0.0) & (f <= 1.0))


# --- global-total anchor (added after the 2026-08-24 global fit) -------------


def _skewed_batch(n=400):
    """A right-skewed target, which is what makes the log loss misbehave."""
    import numpy as np
    rng = np.random.default_rng(0)
    nyt = 17520.0
    fs = rng.uniform(0.0, 0.6, n)
    batch = FireBatch(
        nfirestep=jnp.asarray(fs * nyt), sdry=jnp.asarray(fs * nyt * 0.7),
        nytimes=jnp.asarray(nyt), totlit=jnp.asarray(rng.uniform(0.1, 2.0, n)),
        totalit=jnp.asarray(rng.uniform(0.05, 1.0, n)),
        cbiol=jnp.asarray(rng.uniform(0.0, 1.0, (n, 12))),
        plai=jnp.asarray(rng.uniform(0.0, 5.0, (n, 12))),
        popd=jnp.asarray(rng.uniform(0.0, 50.0, n)),
    )
    return batch


def test_total_weight_zero_recovers_the_pure_pattern_loss():
    batch = _skewed_batch()
    p = FireParams()
    obs = predict_burned_fraction(FireParams(k_b=1.3), batch)
    anchored = fire_loss(p, batch, obs, total_weight=0.0)
    # The anchor is additive and non-negative, so it can only raise the loss.
    assert float(fire_loss(p, batch, obs, total_weight=2.0)) >= float(anchored)


def test_anchor_vanishes_when_the_global_total_is_right():
    """A prediction with the correct area-mean pays no anchor penalty."""
    batch = _skewed_batch()
    p = FireParams()
    obs = predict_burned_fraction(p, batch)          # exact match
    assert float(fire_loss(p, batch, obs, total_weight=5.0)) == pytest.approx(
        float(fire_loss(p, batch, obs, total_weight=0.0)), abs=1e-12)


def test_anchor_penalises_a_global_magnitude_collapse():
    """The defect it exists for: predicting near-nothing must cost something.

    Without the anchor, log-space MSE is minimised at the geometric mean of a
    right-skewed target, which drove the 2001-2005 global fit to 0.34% burned
    against an observed 3.00%. See docs/FIRE.md.
    """
    batch = _skewed_batch()
    obs = predict_burned_fraction(FireParams(k_b=1.0), batch)
    collapsed = FireParams(k_b=0.01)
    lo = fire_loss(collapsed, batch, obs, total_weight=0.0)
    hi = fire_loss(collapsed, batch, obs, total_weight=3.0)
    assert float(hi) > float(lo) + 0.1


def test_vpd_danger_applies_vegetation_density():
    """With VPD danger on, VD must enter the prediction -- lai_ref_vd matters."""
    # Sparse canopy on purpose: the default fixture sums to ~30 LAI over its 12
    # PFTs, which saturates vegetation density at 1.0 for any plausible
    # lai_ref_vd and would hide the dependence entirely.
    batch = _skewed_batch()._replace(plai=_skewed_batch().plai * 0.02)
    assert float(jnp.max(jnp.sum(batch.plai, axis=1))) < 1.5
    a = predict_burned_fraction(
        FireParams(use_vpd_danger=True, lai_ref_vd=2.0), batch)
    b = predict_burned_fraction(
        FireParams(use_vpd_danger=True, lai_ref_vd=10.0), batch)
    assert not jnp.allclose(a, b)
    # A lower saturation LAI means VD saturates sooner, so burning is higher.
    assert float(jnp.mean(a)) > float(jnp.mean(b))


def test_lai_ref_vd_is_fittable_but_vpd_ref_is_not():
    """vpd_ref/d_crit_vpd act only through accumulators the run precomputes."""
    assert "lai_ref_vd" in PARAM_BOUNDS
    for k in ("vpd_ref", "d_crit_vpd", "d_crit"):
        assert k in FIXED_PARAMS
        assert k not in PARAM_BOUNDS


def test_ign_max_is_reached_only_through_the_span_reparameterisation():
    """A direct ign_max bound would let the fit put it below ig0."""
    assert "ign_span" in PARAM_BOUNDS
    assert "ign_max" not in PARAM_BOUNDS
    p = from_unconstrained(to_unconstrained(FireParams(), ("ign_span",)),
                           FireParams())
    assert float(p.ign_max) > float(p.ig0)
