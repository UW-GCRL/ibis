"""Hybrid physics+ML framework for IBIS-JAX."""

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
