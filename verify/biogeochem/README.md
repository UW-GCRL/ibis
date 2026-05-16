# biogeochem (CENTURY soil C/N) — verification scaffold

**Status**: PENDING.

## Scope

F77 `biogeochem.f` — single subroutine `soilbgc` (1046 lines). CENTURY-like
decomposition of leaf/root/wood litter into 11 SOM pools with N
mineralization. Called once per day.

JAX: `physics/biogeochem.py` (~270 lines) — single function `soilbgc`.

## Why this is heavy but tractable

- **No RNG, no iterative solver** — pure deterministic arithmetic.
- ~20 SOM pool variables (clitlm/clitls/clitll/clitrm/clitrs/clitrl/clitwm/
  clitws/clitwl/csoislop/csoislon/csoipas + N forms) all read/written by
  soilbgc via common blocks (comveg/comsum).
- Many BGC rate constants from compft (k_litl, k_lits, k_litr, k_woods,
  k_woodl, kslow, kpass, cn_*, fbpom, fbsom, decomp_factor_*).

## Recommended path

- Standard wrapper pattern: copy biogeochem.f → biogeochem_subset.f, build
  wrap_soilbgc that seeds ~25 input fields and reads ~30 outputs.
- Expected: PRECISION (machine precision); pure arithmetic should match
  to ~1e-13 abs.

## Estimated effort: 2–3 hours
