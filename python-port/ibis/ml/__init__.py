"""Hybrid physics+ML framework for IBIS-JAX."""

# The component-swapping framework (base / wrappers / registry) moved to
# _legacy_unwired/ on 2026-08-28: nothing consumed it, and re-exporting it here
# implied it was wired. See _legacy_unwired/README.md and docs/AUDIT.md.
from ibis.ml.training import (
    TrainState, create_train_state, train_step, train_loop,
    mse_loss, mae_loss,
)
