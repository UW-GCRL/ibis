"""Tests for Phase 10: ML framework — protocol, wrappers, registry, training."""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import pytest

from ibis.ml.base import PhysicsComponent, WrappedFunction, Identity
from ibis.ml.wrappers import (
    MLP, NeuralPhysics, HybridPhysics, BlendedPhysics,
    make_neural_physics, make_hybrid_physics,
)
from ibis.ml.registry import ComponentRegistry
from ibis.ml.training import (
    TrainState, create_train_state, train_step, train_loop,
    mse_loss, mae_loss,
)


# ===================================================================
# Protocol tests
# ===================================================================

class TestProtocol:
    """Test the PhysicsComponent protocol."""

    def test_identity_satisfies_protocol(self):
        comp = Identity()
        assert isinstance(comp, PhysicsComponent)

    def test_wrapped_function_satisfies_protocol(self):
        def my_func(inputs, params=None):
            return {"out": inputs["x"] * 2}
        comp = WrappedFunction(my_func)
        assert isinstance(comp, PhysicsComponent)

    def test_wrapped_function_call(self):
        def my_func(inputs, params=None):
            return {"out": inputs["x"] * 2}
        comp = WrappedFunction(my_func)
        result = comp({"x": jnp.array(3.0)})
        assert result["out"] == 6.0

    def test_identity_passthrough(self):
        inp = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        comp = Identity()
        out = comp(inp)
        assert out["a"] == 1.0
        assert out["b"] == 2.0


# ===================================================================
# MLP tests
# ===================================================================

class TestMLP:
    """Test the MLP building block."""

    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        mlp = MLP(4, 2, hidden_sizes=(8, 8), key=key)
        x = jnp.ones(4)
        y = mlp(x)
        assert y.shape == (2,)

    def test_batched_vmap(self):
        key = jax.random.PRNGKey(1)
        mlp = MLP(3, 1, hidden_sizes=(16,), key=key)
        batch = jnp.ones((10, 3))
        y = jax.vmap(mlp)(batch)
        assert y.shape == (10, 1)

    def test_differentiable(self):
        key = jax.random.PRNGKey(2)
        mlp = MLP(2, 1, hidden_sizes=(8,), key=key)
        x = jnp.array([1.0, 2.0])

        def loss(model):
            return jnp.sum(model(x) ** 2)

        grads = eqx.filter_grad(loss)(mlp)
        # Check that gradients exist for all layers
        for layer_grad in grads.layers:
            assert jnp.any(layer_grad.weight != 0.0)

    def test_jit_compatible(self):
        key = jax.random.PRNGKey(3)
        mlp = MLP(3, 2, hidden_sizes=(8,), key=key)
        x = jnp.ones(3)

        @jax.jit
        def run(model, x):
            return model(x)

        y = run(mlp, x)
        assert y.shape == (2,)
        assert jnp.all(jnp.isfinite(y))


# ===================================================================
# NeuralPhysics tests
# ===================================================================

class TestNeuralPhysics:
    """Test NeuralPhysics wrapper."""

    def test_create_and_call(self):
        key = jax.random.PRNGKey(10)
        comp = make_neural_physics(
            input_keys=("temp", "moisture"),
            input_sizes=(1, 1),
            output_keys=("flux",),
            output_sizes=(1,),
            hidden_sizes=(16, 16),
            key=key,
        )
        inputs = {"temp": jnp.array(300.0), "moisture": jnp.array(0.5)}
        out = comp(inputs)
        assert "flux" in out
        assert jnp.isfinite(out["flux"])

    def test_batched(self):
        key = jax.random.PRNGKey(11)
        npoi = 5
        comp = make_neural_physics(
            input_keys=("t", "q"),
            input_sizes=(1, 1),
            output_keys=("gs", "an"),
            output_sizes=(1, 1),
            hidden_sizes=(8,),
            key=key,
        )
        inputs = {
            "t": jnp.ones(npoi) * 300.0,
            "q": jnp.ones(npoi) * 0.01,
        }
        out = comp(inputs)
        assert out["gs"].shape == (npoi,)
        assert out["an"].shape == (npoi,)

    def test_multi_dim_output(self):
        key = jax.random.PRNGKey(12)
        comp = make_neural_physics(
            input_keys=("x",),
            input_sizes=(3,),
            output_keys=("y",),
            output_sizes=(4,),
            hidden_sizes=(8,),
            key=key,
        )
        inputs = {"x": jnp.ones(3)}
        out = comp(inputs)
        assert out["y"].shape == (4,)

    def test_satisfies_protocol(self):
        key = jax.random.PRNGKey(13)
        comp = make_neural_physics(
            input_keys=("x",), input_sizes=(1,),
            output_keys=("y",), output_sizes=(1,),
            key=key,
        )
        assert isinstance(comp, PhysicsComponent)

    def test_differentiable(self):
        key = jax.random.PRNGKey(14)
        comp = make_neural_physics(
            input_keys=("x",), input_sizes=(2,),
            output_keys=("y",), output_sizes=(1,),
            hidden_sizes=(8,),
            key=key,
        )
        inputs = {"x": jnp.array([1.0, 2.0])}

        def loss(model):
            return jnp.sum(model(inputs)["y"] ** 2)

        loss_val, grads = eqx.filter_value_and_grad(loss)(comp)
        assert jnp.isfinite(loss_val)
        # Gradients should flow through the net
        grad_leaves = jax.tree.leaves(grads.net)
        has_nonzero = any(jnp.any(g != 0.0) for g in grad_leaves if g is not None)
        assert has_nonzero


# ===================================================================
# HybridPhysics tests
# ===================================================================

class TestHybridPhysics:
    """Test HybridPhysics wrapper."""

    def _simple_physics(self, inputs, params=None):
        return {"flux": inputs["t"] * 0.01}

    def test_alpha_zero_equals_physics(self):
        """With alpha=0, hybrid output should equal physics output."""
        key = jax.random.PRNGKey(20)
        hybrid = make_hybrid_physics(
            physics_fn=self._simple_physics,
            input_keys=("t",), input_sizes=(1,),
            output_keys=("flux",), output_sizes=(1,),
            hidden_sizes=(8,),
            key=key,
        )
        # alpha is initialised to 0
        assert hybrid.alpha == 0.0

        inputs = {"t": jnp.array(300.0)}
        phys_out = self._simple_physics(inputs)
        hybrid_out = hybrid(inputs)
        np.testing.assert_allclose(
            float(hybrid_out["flux"]), float(phys_out["flux"]), atol=1e-6
        )

    def test_nonzero_alpha_adds_correction(self):
        key = jax.random.PRNGKey(21)
        hybrid = make_hybrid_physics(
            physics_fn=self._simple_physics,
            input_keys=("t",), input_sizes=(1,),
            output_keys=("flux",), output_sizes=(1,),
            hidden_sizes=(8,),
            key=key,
        )
        # Set alpha to 1.0
        hybrid = eqx.tree_at(lambda m: m.alpha, hybrid, jnp.array(1.0))

        inputs = {"t": jnp.array(300.0)}
        phys_out = self._simple_physics(inputs)
        hybrid_out = hybrid(inputs)

        # Should differ from pure physics by the NN correction
        assert hybrid_out["flux"] != phys_out["flux"]

    def test_batched(self):
        key = jax.random.PRNGKey(22)
        npoi = 4
        hybrid = make_hybrid_physics(
            physics_fn=self._simple_physics,
            input_keys=("t",), input_sizes=(1,),
            output_keys=("flux",), output_sizes=(1,),
            hidden_sizes=(8,),
            key=key,
        )
        hybrid = eqx.tree_at(lambda m: m.alpha, hybrid, jnp.array(0.5))

        inputs = {"t": jnp.ones(npoi) * 300.0}
        out = hybrid(inputs)
        assert out["flux"].shape == (npoi,)

    def test_gradient_through_alpha(self):
        key = jax.random.PRNGKey(23)
        hybrid = make_hybrid_physics(
            physics_fn=self._simple_physics,
            input_keys=("t",), input_sizes=(1,),
            output_keys=("flux",), output_sizes=(1,),
            hidden_sizes=(8,),
            key=key,
        )
        hybrid = eqx.tree_at(lambda m: m.alpha, hybrid, jnp.array(0.1))

        inputs = {"t": jnp.array(300.0)}
        target = jnp.array(5.0)

        def loss(model):
            out = model(inputs)
            return (out["flux"] - target) ** 2

        loss_val, grads = eqx.filter_value_and_grad(loss)(hybrid)
        assert jnp.isfinite(loss_val)
        # Alpha gradient should be non-zero
        assert grads.alpha != 0.0


# ===================================================================
# BlendedPhysics tests
# ===================================================================

class TestBlendedPhysics:
    """Test BlendedPhysics wrapper."""

    def test_extreme_weights(self):
        def comp_a(inputs, params=None):
            return {"y": jnp.array(10.0)}

        def comp_b(inputs, params=None):
            return {"y": jnp.array(20.0)}

        # Large positive logit → weight ≈ 1 → comp_a dominates
        blend = BlendedPhysics(
            comp_a=comp_a, comp_b=comp_b,
            weight_logit=jnp.array(100.0),
            blend_keys=("y",),
        )
        out = blend({})
        np.testing.assert_allclose(float(out["y"]), 10.0, atol=1e-3)

        # Large negative logit → weight ≈ 0 → comp_b dominates
        blend2 = BlendedPhysics(
            comp_a=comp_a, comp_b=comp_b,
            weight_logit=jnp.array(-100.0),
            blend_keys=("y",),
        )
        out2 = blend2({})
        np.testing.assert_allclose(float(out2["y"]), 20.0, atol=1e-3)

    def test_equal_weight(self):
        def comp_a(inputs, params=None):
            return {"y": jnp.array(0.0)}

        def comp_b(inputs, params=None):
            return {"y": jnp.array(10.0)}

        # logit=0 → sigmoid=0.5 → average
        blend = BlendedPhysics(
            comp_a=comp_a, comp_b=comp_b,
            weight_logit=jnp.array(0.0),
            blend_keys=("y",),
        )
        out = blend({})
        np.testing.assert_allclose(float(out["y"]), 5.0, atol=1e-5)


# ===================================================================
# Registry tests
# ===================================================================

class TestRegistry:
    """Test ComponentRegistry."""

    def test_register_and_get(self):
        reg = ComponentRegistry()
        comp = Identity()
        reg.register("stomata", comp)
        assert reg.get("stomata") is comp

    def test_get_default(self):
        reg = ComponentRegistry()
        default = Identity()
        assert reg.get("stomata", default) is default

    def test_has(self):
        reg = ComponentRegistry()
        assert not reg.has("stomata")
        reg.register("stomata", Identity())
        assert reg.has("stomata")

    def test_unregister(self):
        reg = ComponentRegistry()
        reg.register("stomata", Identity())
        reg.unregister("stomata")
        assert not reg.has("stomata")

    def test_list_components(self):
        reg = ComponentRegistry()
        reg.register("stomata", Identity())
        reg.register("radiation", Identity())
        names = reg.list_components()
        assert set(names) == {"stomata", "radiation"}

    def test_clear(self):
        reg = ComponentRegistry()
        reg.register("a", Identity())
        reg.register("b", Identity())
        reg.clear()
        assert reg.list_components() == []

    def test_overwrite(self):
        reg = ComponentRegistry()
        comp1 = Identity()
        comp2 = Identity()
        reg.register("x", comp1)
        reg.register("x", comp2)
        assert reg.get("x") is comp2

    def test_repr(self):
        reg = ComponentRegistry()
        reg.register("stomata", Identity())
        assert "stomata" in repr(reg)


# ===================================================================
# Training tests
# ===================================================================

class TestTraining:
    """Test differentiable training loop."""

    def test_create_train_state(self):
        key = jax.random.PRNGKey(40)
        model = MLP(2, 1, hidden_sizes=(8,), key=key)
        optimizer = optax.adam(1e-3)
        state = create_train_state(model, optimizer)
        assert state.step == 0
        assert state.loss == 0.0

    def test_single_train_step(self):
        key = jax.random.PRNGKey(41)
        model = MLP(2, 1, hidden_sizes=(8,), key=key)
        optimizer = optax.adam(1e-3)
        state = create_train_state(model, optimizer)

        x = jnp.array([1.0, 2.0])
        target = jnp.array([5.0])

        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - target) ** 2)

        new_state = train_step(state, loss_fn, optimizer)
        assert new_state.step == 1
        assert jnp.isfinite(new_state.loss)

        # Parameters should have changed
        old_leaves = jax.tree.leaves(state.model)
        new_leaves = jax.tree.leaves(new_state.model)
        any_changed = any(
            not jnp.array_equal(o, n)
            for o, n in zip(old_leaves, new_leaves)
        )
        assert any_changed

    def test_loss_decreases(self):
        key = jax.random.PRNGKey(42)
        model = MLP(2, 1, hidden_sizes=(16,), key=key)
        optimizer = optax.adam(1e-2)

        x = jnp.array([1.0, 2.0])
        target = jnp.array([3.0])

        def loss_fn(model):
            return jnp.mean((model(x) - target) ** 2)

        state = create_train_state(model, optimizer)
        losses = []
        for _ in range(50):
            state = train_step(state, loss_fn, optimizer)
            losses.append(float(state.loss))

        # Loss should decrease overall
        assert losses[-1] < losses[0]

    def test_train_loop(self):
        key = jax.random.PRNGKey(43)
        model = MLP(2, 1, hidden_sizes=(8,), key=key)
        optimizer = optax.adam(1e-2)

        x = jnp.array([1.0, 2.0])
        target = jnp.array([3.0])

        def loss_fn(model):
            return jnp.mean((model(x) - target) ** 2)

        final_state = train_loop(model, loss_fn, optimizer, num_steps=20)
        assert final_state.step == 20
        assert jnp.isfinite(final_state.loss)

    def test_mse_loss(self):
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.0, 2.0, 3.0])
        assert mse_loss(pred, target) == 0.0

        pred2 = jnp.array([2.0, 3.0, 4.0])
        assert mse_loss(pred2, target) == 1.0

    def test_mse_loss_masked(self):
        pred = jnp.array([1.0, 10.0, 3.0])
        target = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([1.0, 0.0, 1.0])
        loss = mse_loss(pred, target, mask)
        assert loss == 0.0  # masked out the mismatch

    def test_mae_loss(self):
        pred = jnp.array([1.0, 3.0, 5.0])
        target = jnp.array([2.0, 3.0, 4.0])
        loss = mae_loss(pred, target)
        np.testing.assert_allclose(float(loss), 2.0 / 3.0, atol=1e-6)


# ===================================================================
# Integration: train a NeuralPhysics to fit a target function
# ===================================================================

class TestMLIntegration:
    """End-to-end: train NN to approximate a simple physics function."""

    def test_neural_learns_linear(self):
        """Train NeuralPhysics to learn y = 2*x + 1."""
        key = jax.random.PRNGKey(50)
        comp = make_neural_physics(
            input_keys=("x",), input_sizes=(1,),
            output_keys=("y",), output_sizes=(1,),
            hidden_sizes=(32, 32),
            key=key,
        )
        optimizer = optax.adam(1e-3)

        # Training data — batched
        xs = jnp.linspace(-1, 1, 20)
        ys_target = 2.0 * xs + 1.0

        def loss_fn(model):
            pred = model({"x": xs})["y"]  # (20,)
            return jnp.mean((pred - ys_target) ** 2)

        state = create_train_state(comp, optimizer)
        for _ in range(300):
            state = train_step(state, loss_fn, optimizer)

        # Verify learned function is reasonable
        final_model = state.model
        test_xs = jnp.array([0.5, -0.5])
        pred = final_model({"x": test_xs})["y"]
        expected = 2.0 * test_xs + 1.0
        np.testing.assert_allclose(np.asarray(pred), np.asarray(expected), atol=0.5)

    def test_hybrid_correction(self):
        """Train HybridPhysics correction to fix a biased physics model."""
        # Physics model with known bias: f(x) = x (should be 2x)
        def biased_physics(inputs, params=None):
            return {"y": inputs["x"]}

        key = jax.random.PRNGKey(51)
        hybrid = make_hybrid_physics(
            physics_fn=biased_physics,
            input_keys=("x",), input_sizes=(1,),
            output_keys=("y",), output_sizes=(1,),
            hidden_sizes=(32, 32),
            key=key,
        )
        # Start with alpha=1
        hybrid = eqx.tree_at(lambda m: m.alpha, hybrid, jnp.array(1.0))

        optimizer = optax.adam(1e-3)

        # Target: y = 2x
        xs = jnp.linspace(-1, 1, 20)

        def loss_fn(model):
            total = jnp.array(0.0)
            for i in range(len(xs)):
                pred = model({"x": xs[i:i+1]})["y"]
                target = 2.0 * xs[i]
                total = total + (pred[0] - target) ** 2
            return total / len(xs)

        state = create_train_state(hybrid, optimizer)
        for _ in range(200):
            state = train_step(state, loss_fn, optimizer)

        # The correction net should have learned the bias (≈ x)
        final = state.model
        test_x = jnp.array([0.5])
        pred = final({"x": test_x})["y"][0]
        expected = 2.0 * 0.5
        np.testing.assert_allclose(float(pred), expected, atol=0.5)

    def test_registry_swap_and_run(self):
        """Register an ML component and run it via the registry."""
        reg = ComponentRegistry()

        # Default physics
        def physics_stomata(inputs, params=None):
            return {"gs": inputs["t"] * 0.001}

        # ML override
        key = jax.random.PRNGKey(52)
        nn_stomata = make_neural_physics(
            input_keys=("t",), input_sizes=(1,),
            output_keys=("gs",), output_sizes=(1,),
            hidden_sizes=(8,),
            key=key,
        )
        reg.register("stomata", nn_stomata)

        # Use registry
        component = reg.get("stomata", default=physics_stomata)
        assert component is nn_stomata

        inputs = {"t": jnp.array(300.0)}
        out = component(inputs)
        assert "gs" in out
        assert jnp.isfinite(out["gs"])

        # Unregister → falls back to physics
        reg.unregister("stomata")
        component2 = reg.get("stomata", default=physics_stomata)
        assert component2 is physics_stomata
        out2 = component2(inputs)
        np.testing.assert_allclose(float(out2["gs"]), 0.3, atol=1e-6)


# ===================================================================
# JIT + gradient tests
# ===================================================================

class TestJITandGrad:
    """Verify JIT compilation and gradient flow through ML components."""

    def test_jit_neural_physics(self):
        key = jax.random.PRNGKey(60)
        comp = make_neural_physics(
            input_keys=("x",), input_sizes=(2,),
            output_keys=("y",), output_sizes=(1,),
            hidden_sizes=(8,),
            key=key,
        )
        inputs = {"x": jnp.array([1.0, 2.0])}

        @jax.jit
        def run(model, inp):
            return model(inp)

        out = run(comp, inputs)
        assert jnp.isfinite(out["y"])

    def test_jit_hybrid_physics(self):
        def phys(inputs, params=None):
            return {"y": inputs["x"][0] * 2.0}

        key = jax.random.PRNGKey(61)
        hybrid = make_hybrid_physics(
            physics_fn=phys,
            input_keys=("x",), input_sizes=(2,),
            output_keys=("y",), output_sizes=(1,),
            hidden_sizes=(8,),
            key=key,
        )

        @jax.jit
        def run(model, inp):
            return model(inp)

        out = run(hybrid, {"x": jnp.array([1.0, 2.0])})
        assert jnp.isfinite(out["y"])

    def test_grad_through_neural_physics(self):
        key = jax.random.PRNGKey(62)
        comp = make_neural_physics(
            input_keys=("x",), input_sizes=(3,),
            output_keys=("y",), output_sizes=(1,),
            hidden_sizes=(16,),
            key=key,
        )

        def loss(model):
            return jnp.sum(model({"x": jnp.ones(3)})["y"] ** 2)

        grads = eqx.filter_grad(loss)(comp)
        grad_leaves = jax.tree.leaves(grads.net)
        has_grad = any(jnp.any(g != 0.0) for g in grad_leaves if g is not None)
        assert has_grad

    def test_grad_through_hybrid_physics(self):
        def phys(inputs, params=None):
            return {"y": jnp.sum(inputs["x"])}

        key = jax.random.PRNGKey(63)
        hybrid = make_hybrid_physics(
            physics_fn=phys,
            input_keys=("x",), input_sizes=(3,),
            output_keys=("y",), output_sizes=(1,),
            hidden_sizes=(16,),
            key=key,
        )
        hybrid = eqx.tree_at(lambda m: m.alpha, hybrid, jnp.array(0.5))

        def loss(model):
            return jnp.sum(model({"x": jnp.ones(3)})["y"] ** 2)

        grads = eqx.filter_grad(loss)(hybrid)
        # Both alpha and correction net should have gradients
        assert grads.alpha != 0.0
        grad_leaves = jax.tree.leaves(grads.correction_net)
        has_grad = any(jnp.any(g != 0.0) for g in grad_leaves if g is not None)
        assert has_grad

    def test_vmap_neural_physics(self):
        """Verify vmap works for batched evaluation."""
        key = jax.random.PRNGKey(64)
        comp = make_neural_physics(
            input_keys=("x",), input_sizes=(2,),
            output_keys=("y",), output_sizes=(1,),
            hidden_sizes=(8,),
            key=key,
        )
        # Batched input
        batch_inputs = {"x": jnp.ones((10, 2))}
        out = comp(batch_inputs)
        assert out["y"].shape == (10,)
        assert jnp.all(jnp.isfinite(out["y"]))
