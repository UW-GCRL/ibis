# Full lsxmain time-march — verification scaffold

**Status**: PENDING. Requires all preceding sections completed first.

## Scope

F77 `main.f` chain: one full timestep of IBIS, composing every physics
module verified individually (radiation, soil, snow, canopy, stomata,
canopy_exchange, biogeochem, weather, climate, phenology, vegetation
dynamics, etc.).

JAX: `timestepping/loops.py` — `step()` function that orchestrates the
full timestep.

## Why this is the heaviest

- Composes every other module → tolerance is the union of all module
  tolerances; expect LOOSE somewhere.
- Time-march: 100+ timesteps with state carrying between → tiny per-step
  diffs can amplify over many steps.
- Common-block state at thousands of fields → wrapper boundary is
  enormous.

## Recommended path

1. Verify each subsystem individually first (canopy_exchange, biogeochem
   still pending).
2. Build a `wrap_step` that calls F77 main.f's per-timestep block.
3. Mirror in Python by calling loops.step().
4. Time-march for 100 timesteps with framework's `time_march_compare`
   to detect first-divergence step per variable.

## Estimated effort: 5–8 hours (after canopy_exchange + biogeochem done)
