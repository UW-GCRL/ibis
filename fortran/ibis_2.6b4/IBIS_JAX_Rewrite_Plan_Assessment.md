# Assessment of IBIS JAX Rewrite Plan

## Context
Reviewing the rewrite plan at `/home/min/IBIS/ibis_2.6b4/IBIS_JAX_Rewrite_Plan.md` against the actual Fortran codebase (~30K lines, 17 source files, 16 common block headers, ~500+ state variables per gridcell) and available datasets.

## Available Datasets

All on the same 0.5-degree global grid (720x360) — no regridding needed.

| Dataset | Location | Contents |
|---|---|---|
| GSWP3 forcing | `/shared/nas2/Min/GSWP3/` | 10 daily variables (tas, tasmin, tasmax, pr, huss, rhs, ps, rsds, rlds, wind), 1901-2010. Sub-daily version to be acquired later. |
| ESA CCI vegetation | `/shared/nas2/Min/ESACCI_LC/ibis_vegtype/` | 15 IBIS PFT fractions + dominant type, annual maps 1992-2020 |
| HWSD soil | `/shared/nas2/Min/HWSD/soita.nc` | Sand/clay % for 6 depth layers (0-400cm) |
| Land mask | `/shared/nas2/Min/HWSD/surta.nc` | Land fraction from ESA CCI 1992 |

**Key decision:** float32 is sufficient (no need for `jax_enable_x64`).

## Overall Verdict

**This is a strong rewrite plan.** The behavior-first philosophy and checkpoint-gated validation ladder are exactly right. Most Fortran-to-X rewrites fail by trying to modernize and translate simultaneously — this plan avoids that.

## Strengths

1. **Behavior-first staging** (Phase A/B/C: parity -> modernize -> enhance) is correct
2. **Checkpoint ladder** (function -> module -> interface -> runtime horizons) is the plan's best feature — addresses the #1 failure mode of scientific code ports
3. **Forcing abstraction** correctly separates legacy weather generation from canonical interface
4. **Accurate codebase assessment** matches the actual code structure
5. **Existing test infrastructure** (regression tests, phased unit tests) provides a foundation for Phase 0

## Forcing Strategy Update

### Sub-daily GSWP3 (once acquired) enables direct forcing

With sub-daily GSWP3, the forcing provider maps almost directly to IBIS's canonical needs:

| GSWP3 variable | IBIS forcing field | Processing needed |
|---|---|---|
| `rsds` | `solad(2)`, `solai(2)` | Split total SW → visible/NIR (0.46/0.54), then direct/diffuse via Spitters method + zenith angle |
| `rlds` | `fira` | Direct use — replaces internal LW calculation |
| `tas` | `ta` | Direct use |
| `huss` | `qa` | Direct use |
| `ps` | `psurf` | Direct use — replaces elevation-based derivation |
| `wind` | `ua` | Direct use |
| `pr` + `tas` | `raina`, `snowa` | Partition by temperature threshold (2.5°C) |
| computed | `coszen` | From orbital mechanics + time |
| prescribed | `co2conc`, `o2conc` | Annual constants, not from GSWP3 |

**Implication for plan:** The legacy weather generator (`daily()` + `diurnal()` in weather.f) becomes entirely optional. Phase 2 should prioritize the GSWP3 forcing provider, not the legacy path.

### Remaining processing for GSWP3 provider
- SW waveband split (visible/NIR) — fixed ratio
- Direct/diffuse split — Spitters-type clearness index method
- Rain/snow partitioning — temperature threshold
- `coszen` computation — orbital mechanics
- Cloud fraction (if needed by any kernel) — derive from clearness index

## Concerns and Recommended Plan Amendments

### 1. Subroutine I/O audit — add to Phase 0

The plan describes process kernels as pure functions `(params, static, state, forcing, context) -> (new_state, diagnostics)`, but the actual code communicates through ~16 common blocks mutated in-place. Without knowing which variables each subroutine reads and writes, you can't define function signatures or write interface checkpoints.

**Recommended action:** For each subroutine in `lsxmain` and the daily/annual loops, document which common-block variables are read vs. written. Can be partially automated with static analysis.

### 2. The `com1d.h` intermediate context — acknowledge as a data category

`com1d.h` has ~150 intermediate variables (aerodynamic coefficients, flux partitions, solar redistribution) written by one subroutine and consumed by the next within `lsxmain`. These are not prognostic state, not diagnostics, not forcing — they're **intermediate coupling variables**.

**Recommended action:** Add an explicit "intermediates" or "coupling context" data category to the target architecture. This is just honest plumbing for what common blocks do implicitly.

### 3. Canopy solve is highest-risk kernel — derisk early

`canopy.f` (2,397 lines) contains the implicit energy/moisture balance: 3 fixed iterations interleaving `stomata`, `turcof`, `turvap`, coupling radiation, aerodynamics, stomatal conductance, and leaf energy balance. This is the hardest piece to port while keeping JIT-compatible.

**Recommended action:** Consider prototyping the canopy kernel early (even before the full Phase 4 sequence) to identify JAX-specific issues before they block the critical path.

### 4. Adjust Phase 2: GSWP3 provider first, legacy weather generator later

Original plan puts legacy weather generation in Phase 2 and GSWP3 in Phase 9. With sub-daily GSWP3 as the primary forcing pathway:

**Recommended Phase 2 reordering:**
1. Implement static input loading (soil, vegetation, land mask from HWSD/ESA CCI)
2. Implement GSWP3 sub-daily forcing provider (primary path)
3. Define canonical forcing object
4. Defer legacy weather generator to Phase 9 or optional compatibility mode

### 5. Move `comveg.h` schema definition earlier

`comveg.h` is the largest header (249 lines, ~100+ variables including per-PFT arrays for biomass, carbon, phenology). Its structure gates Phases 4-7.

**Recommended action:** Define the vegetation state schema in Phase 1 alongside utility/solver porting, not incrementally during later phases.

### 6. Separate `biogeochem.f` from `vegetation.f`

The plan groups them in Phase 6, but:
- Biogeochem is called **daily** (with spin-up acceleration)
- Vegetation dynamics are **annual**
- Biogeochem spin-up is the primary performance target

**Recommended action:** Port biogeochem in Phase 6a, vegetation dynamics in Phase 7 (with fire and restart).

### 7. JAX-specific decisions to make early

| Decision | Recommendation |
|---|---|
| Precision | float32 throughout (confirmed) |
| State containers | Choose between equinox modules, plain pytrees, or NamedTuples before Phase 1 |
| Tridiagonal solver | Custom Thomas algorithm via `lax.scan` or `jax.scipy.linalg.solve_banded` — no built-in batched solver |
| Time loop | `lax.scan` for hourly loop; `jax.checkpoint` (rematerialization) if differentiability through long runs is needed |
| JIT boundary | Decide which kernels must be JIT-compatible vs. which can run in Python (I/O, forcing ingest, annual ecology likely Python; hourly physics JIT) |

### 8. Tolerance calibration for float32

The plan's tolerance classes were implicitly designed for float64. With float32:
- **function-tight:** ~1e-6 relative (not 1e-12)
- **step-tight:** ~1e-5 relative
- **accumulation-tight:** ~1e-4 for daily sums over 24 steps
- **annual-tight:** ~1e-3 for annual sums over 8,760 steps

Accumulator drift is the main risk — worth monitoring but acceptable for a land surface model.

## Revised Phase Summary

| Phase | Scope | Key change from original |
|---|---|---|
| 0 | Legacy baseline, validation harness, **subroutine I/O audit** | Added I/O audit |
| 1 | Utilities, solvers, state/param schemas, **`comveg.h` schema** | Schema definition pulled earlier |
| 2 | Static input loading (HWSD/ESA CCI), **GSWP3 forcing provider**, canonical forcing object | GSWP3 replaces legacy weather as primary |
| 3 | Radiation | Unchanged |
| 4 | Canopy / interception / physiology (**highest risk — consider early prototype**) | Flagged as critical path |
| 5 | Snow and soil physics | Unchanged |
| 6 | Climate memory, stats, phenology, **soil biogeochem (separated)** | Biogeochem split out |
| 7 | Annual vegetation dynamics, fire, **restart** | Vegetation dynamics moved here |
| 8 | Full legacy-parity approval | Unchanged |
| 9 | Modernization, **legacy weather generator (optional)**, external forcing adapters | Legacy weather deferred here |

## Verification

- Run GSWP3 provider against a small regional subset (~100-500 gridcells)
- Compare canonical forcing output against Fortran `diurnal()` output for same daily inputs
- Checkpoint ladder proceeds as originally specified, with float32 tolerances
- Regression tests at `/home/min/IBIS/tests/` provide existing infrastructure to build on
