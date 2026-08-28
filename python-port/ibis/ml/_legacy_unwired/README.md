# Unwired: the hybrid physics/ML component-swapping framework

`base.py` (PhysicsComponent protocol), `registry.py` (runtime component
registry) and `wrappers.py` (NeuralPhysics / HybridPhysics / BlendedPhysics)
were written to let an ML surrogate replace a physics component at runtime.

**Nothing consumes them.** Moved here 2026-08-28 by the audit in
docs/AUDIT.md, which found them reachable only from tests. The `ml_stomata`
and `ml_canopy` config fields that would have selected a surrogate are read
nowhere either.

They are quarantined rather than deleted because the design is sound and a
hybrid physics-ML methane or stomata model is a plausible future direction
(see docs/METHANE.md 5f on KGML-style approaches). But leaving them importable
from `ibis.ml` invited the assumption that they work end to end, which is the
exact failure this codebase kept hitting.

**Note `ibis/ml/training.py` and `ibis/ml/fire_calibration.py` stayed put** --
both have real callers. Gradient-based calibration of *physical parameters*
(the M7 plan) needs neither a registry nor a surrogate; it differentiates the
physics directly.

To revive: move the files back, restore the re-exports in `ibis/ml/__init__.py`,
wire `ml_stomata`/`ml_canopy` in the timestep, and remove the entries from
`KNOWN_DEAD_MODULES` in `tests/test_no_dead_code.py`.
