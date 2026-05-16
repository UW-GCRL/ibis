# IBIS Python + JAX Rewrite Plan

## Scope

This document captures a code-grounded architecture and migration plan for rewriting the original IBIS model in `/home/min/IBIS/ibis_2.6b4` into Python + JAX.

Primary goals:

1. Preserve original scientific behavior first.
2. Build a differentiable, calibration-ready core where practical.
3. Redesign the software for extensibility.
4. Improve performance, especially spin-up.
5. Support both modern external sub-daily forcing and the original internal weather generation pathway.
6. Enforce dense checkpoint-based parity validation throughout the migration.
7. Keep the rewritten model ready for WIEMIP-style factorial and protocol-driven development, including future addition of missing land processes without destabilizing the core model architecture.

---

## Most Important Files in the Original Codebase

- `main.f`: top-level simulation driver and time loop orchestration.
- `main.f` (`lsxmain`): fixed land-surface process sequence.
- `io.f`: static input reading, anomaly reading, restart reading/writing, and daily/monthly/yearly output.
- `weather.f`: daily weather generation and sub-daily forcing disaggregation.
- `climate.f`: climate analysis and PFT existence logic.
- `canopy.f`: canopy aerodynamic, interception, and implicit energy/moisture solve.
- `radiation.f`: shortwave and longwave transfer.
- `soil.f`: soil hydrology and soil temperature.
- `snow.f`: snow physics.
- `physiology.f`: stomata and photosynthesis.
- `vegetation.f`: phenology, annual vegetation dynamics, fire, vegetation mapping.
- `biogeochem.f`: soil carbon and nitrogen cycling; current spin-up acceleration target.
- `initial.f`: cold start, restart, and initialization.
- `readpars.f`: parameter file loading.
- `compar.h`, `comatm.h`, `combcs.h`, `comsoi.h`, `comsno.h`, `comveg.h`, `comsum.h`, `com1d.h`, `comwork.h`: shared common-block state and scratch architecture.
- `math.f`: RNG and shared numerical solvers.
- `utilities.f`: array utilities, including land-only vector packing/unpacking.

Parameter files currently used by the code:

- `params.veg` in `ibis_2.6b4`
- `params.can` in `/home/min/IBIS`
- `params.soi` in `/home/min/IBIS`

Available external datasets identified for the rewrite:

- GSWP3 forcing at `/shared/nas2/Min/GSWP3/`
- ESA CCI vegetation products at `/shared/nas2/Min/ESACCI_LC/ibis_vegtype/`
- HWSD soil texture at `/shared/nas2/Min/HWSD/soita.nc`
- land mask at `/shared/nas2/Min/HWSD/surta.nc`

Assessment input indicates these data are already on the same 0.5-degree global grid as IBIS (`720 x 360`), so a first rewrite pass can assume no regridding is required. The currently available GSWP3 archive is daily; sub-daily GSWP3 is planned for later acquisition.

Protocol-driven external drivers now also need to be anticipated from `WIEMIP_1PCT_Protocol_20250330.md`, including:

- stable 6-hourly climate forcing recycled from the first 20 years of a provided stable dataset
- dynamic 6-hourly climate forcing from multiple PRIME-pattern datasets
- prescribed 1pctCO2 concentration time series from file
- fixed and transient nitrogen deposition drivers
- fixed-1850 land-use treatment and disabled wood harvest for protocol runs

---

## Codebase Understanding

### Observations

- The code is organized by science area, but the effective architecture is driven by global common blocks rather than explicit module boundaries.
- `main.f` controls the full run sequence:
  - read configuration from `ibis.infile`
  - read parameter files
  - read static inputs and climatology
  - initialize state or restart
  - loop over year -> month -> day -> sub-daily timestep
  - call soil biogeochemistry daily
  - call land-surface physics every sub-daily step
  - accumulate diagnostics
  - run annual vegetation dynamics
  - write outputs and restart files
- The land-surface core in `lsxmain` is a hard-coded process chain:
  - `setsoi`
  - `fwetcal`
  - `solset`
  - `solsur`, `solalb`, `solarf`
  - `irrad`
  - `cascade`
  - `fwetcal`
  - `canopy`
  - `cascad2`
  - `noveg`
  - `snow`
  - `soilctl`
- State is spread across many common blocks:
  - atmospheric forcing and weather-generator memory
  - static climatology and anomaly-adjusted climate
  - soil, snow, and canopy state
  - vegetation biomass, LAI, phenology, and carbon pools
  - time accumulators and diagnostics
  - large scratch arrays and land-grid packing buffers
- The runtime forcing path is layered:
  - static monthly climatology from `readit`
  - optional monthly anomalies from `rdanom`
  - optional daily anomalies from `rdday`
  - daily weather synthesis in `daily`
  - sub-daily disaggregation in `diurnal`
- The current spin-up acceleration is only for soil biogeochemistry:
  - `soilbgc` is called multiple times per day in early years
  - `spincons=40`, `spinfrac=0.75`, `eqyears=50`, `nspinsoil=iyear0+150`
- Array sizing is compile-time in `compar.h`, including `npoi`, which tightly couples input subset, grid definition, and compilation.
- I/O is extensive and fragmented, especially restart and time-aggregated outputs.
- `io.f` repeatedly converts between full-grid arrays and land-only vectors using `arr2vec` and `vec2arr`.
- The current checkout is not fully self-contained for running:
  - parameter files are split across directories
  - runtime `input`, `output`, and `restart` directories are not present in the inspected tree
- There is some documentation / code drift:
  - `main.f` prints version `2.6b3`
  - `makefile` says `2.6b4`
  - `READMEnet.txt` references `ibisnetcdf`, but the make targets are `ibis` and `ibisabs`
- There is existing test infrastructure in `/home/min/IBIS/tests`, which should be incorporated into the rewrite validation harness rather than replaced.
- The `tests/` directory also contains a modular Fortran 2008 refactor effort, with `test_phase1.f90`, `test_phase2.f90`, and `test_phase3.f90` checking modules such as `mod_precision`, `mod_constants`, `mod_bgc_params`, `mod_thermo_funcs`, and multiple explicit state modules.

### Inferences

- The existing science decomposition is still useful as the basis for the rewrite.
- The land-only vector strategy already aligns well with JAX-oriented batched column execution.
- The codebase has accumulated multiple generations of patches and partial refactors, so parity must be based on actual runtime behavior, not only on developer intent or comments.
- Some extension pathways appear incomplete or dormant, so the rewrite should not assume all legacy structures are fully active.
- The Fortran 2008 modularization work in `/home/min/IBIS/tests` is complementary to the JAX rewrite: it is useful for extracting schemas, constants, and fixtures, but it should not be treated as the production end-state unless the project explicitly chooses to maintain a modern Fortran branch in parallel.

---

## Current Architecture Assessment

### Strengths

- Major science domains are already separated into recognizable source files.
- State variables are explicit and named.
- Key numerical kernels are centralized:
  - canopy implicit solve
  - tridiagonal soil solves
  - annual biomass updates
- The weather-generation pathway and the land-surface physics pathway are conceptually separable, even if not cleanly implemented.

### Weaknesses

- Shared mutable global state dominates the architecture.
- Science logic, configuration, I/O, diagnostics, and experiment control are tightly interwoven.
- Runtime shapes are effectively compile-time.
- The forcing pathway is embedded in the model rather than abstracted.
- Restart and output logic are large and fragmented.
- Validation infrastructure is minimal.
- There are hidden process couplings through accumulator fields and climate-memory fields.

### Risks of a Direct JAX Port

- A literal port would preserve global-state coupling and make the new code hard to test and extend.
- JAX would be awkward if hidden state and shape assumptions are retained.
- Differentiability would be impaired by embedded randomness, categorical logic, thresholds, and legacy weather generation.
- Without an early parity harness, it would be difficult to distinguish intended improvements from regressions.
- Without a protocol-aware experiment layer, WIEMIP-style factorial runs would force experiment logic into the science kernels and recreate the same coupling problems in a new language.

---

## Target Architecture for the Rewrite

### Core Principles

1. Preserve original behavior first in a legacy-compatible mode.
2. Make all core data flow explicit.
3. Separate prognostic state, diagnostics, parameters, static inputs, and forcing.
4. Keep the physics core independent of I/O.
5. Support both site-scale and grid-scale execution from the same core.
6. Make validation, restart, diagnostics, and experiment reproducibility first-class.
7. Separate meteorology, atmospheric composition, deposition, land-use / disturbance drivers, and process toggles so future WIEMIP process additions remain modular.

### Recommended Package Structure

- `ibis_jax/config`
- `ibis_jax/domain`
- `ibis_jax/static`
- `ibis_jax/params`
- `ibis_jax/state`
- `ibis_jax/forcing`
- `ibis_jax/processes`
- `ibis_jax/driver`
- `ibis_jax/diagnostics`
- `ibis_jax/io`
- `ibis_jax/validation`
- `ibis_jax/calibration`

### Data Structure Separation

#### Configuration

Experiment configuration, feature flags, compatibility flags, run controls, calendar, output controls, forcing mode, and spin-up controls.

This should be extended to an explicit `ExperimentSpec` layer for protocol-driven runs, including:

- experiment id such as `spin`, `ctrl`, `bgc`, `cou`, `rad`, `bgc_ndep`, `cou_ndep`
- calendar choice, including `365_day` / `noleap`
- climate forcing source and pattern id
- atmospheric composition driver source
- explicit `CO2` baseline and override fields, so WIEMIP runs can force `280 ppm` even though the legacy `ibis.infile` example currently sets `co2init = 0.000350`
- nitrogen deposition mode
- land-use and harvest mode
- process toggles for fire, dynamic vegetation, nitrogen limitation, permafrost, wetlands, inland waters, BVOC, and other future factorials
- parent restart lineage and branch point metadata
- stable-climate cycling controls, including cycle length, phase handling, and whether branch runs inherit the forcing-cycle position from the parent restart
- output profile selection, including protocol-specific naming and metadata rules

#### Static Inputs / Grid

- land mask
- longitude / latitude
- land index mapping
- cell area
- topography
- soil texture
- climatology
- fixed vegetation map
- fixed land-use / land-cover reference inputs when required by protocol experiments
- dataset metadata

#### Parameters

Nested parameter groups:

- canopy / aerodynamic
- physiology
- snow
- soil physical
- vegetation structural
- biogeochem
- legacy compatibility overrides

#### Dynamic State

Nested prognostic state groups:

- atmosphere memory for legacy weather generator
- snow state
- soil hydrology state
- soil thermal state
- canopy interception and canopy temperatures
- vegetation biomass / LAI / phenology memory
- soil litter / carbon / nitrogen pools
- climate-memory state used by vegetation

The state design should also reserve extension points for future WIEMIP-relevant developments:

- mineral nitrogen state and plant-available nitrogen state
- freeze-thaw / permafrost state beyond the current soil ice and temperature fields
- water-table / inundation state for wetlands
- routed water / lateral export state for inland-water coupling
- disturbance state, if fire and other disturbance schemes are upgraded

#### Intermediate Coupling Context

The rewrite should explicitly represent transient coupling variables that are neither prognostic state nor durable diagnostics.

This category is required because `com1d.h` contains a large set of short-lived interface variables written by one process and consumed by the next inside `lsxmain`, including:

- aerodynamic coefficients
- near-surface exchange terms
- radiation redistribution terms
- flux partitions
- canopy-layer helper quantities

In the Python + JAX design, these should live in an explicit `CouplingContext` or `Intermediates` PyTree passed between adjacent kernels during a timestep.

#### Diagnostics

Derived non-prognostic outputs:

- fluxes
- albedos
- canopy conductance
- photosynthesis
- time accumulators
- annual budgets
- conservation metrics
- protocol reporting variables and metadata-ready output bundles

At minimum, the framework should compute explicit balance diagnostics for:

- water: precipitation input versus evapotranspiration, runoff, drainage, and total storage change
- energy: net radiation versus sensible, latent, ground, and resolved storage tendency terms
- carbon: GPP, NPP, autotrophic respiration, heterotrophic respiration, disturbance losses, leaching, and storage change
- nitrogen, once interactive nitrogen is added: deposition and fixation versus plant uptake, leaching, gaseous loss if represented, and storage change

#### Process Kernels

Pure functions of the form:

`new_state, diagnostics = process(params, static, state, forcing, context)`

Target process modules:

- climate analysis
- legacy weather generation
- radiation
- canopy / interception
- physiology
- snow
- soil hydrology
- soil thermal
- phenology
- soil biogeochemistry
- annual vegetation dynamics
- reducers / diagnostics

#### Driver / Time Integration

Explicit daily and sub-daily drivers:

- initialization driver
- restart driver
- yearly / monthly / daily orchestration
- sub-daily physics loop
- annual ecology loop
- spin-up workflow
- restart branching for experiment trees such as `SPIN -> CTRL/BGC/COU/RAD`

#### I/O

- static input loading
- forcing ingest
- baseline snapshot loading
- restart / checkpoint read/write
- output writers

For WIEMIP specifically, the I/O layer should also include:

- staged download and checksum manifests for Wasabi S3 forcing assets
- a variable-request crosswalk between WIEMIP requested outputs and internal diagnostics
- protocol-compliant NetCDF writing with naming, compression, fill values, and metadata checks

### Process Coupling Model

Use explicit sequencing at first, not a plugin registry by default.

During parity mode, keep a fixed call order matching `main.f` and `lsxmain`.

After parity is stable, add optional process registration or pluggable extensions around the stable baseline sequence.

### Early JAX Structural Decisions

These choices should be made before substantial porting starts:

- use plain nested PyTrees or frozen dataclass-based PyTrees for state and parameter containers, rather than introducing a heavy framework dependency early
- default to `float32` for the main rewrite and parity workflow, with an optional `float64` debug mode retained as a safety valve for difficult numerical investigations
- implement the sub-daily timestep loop with `lax.scan`
- keep I/O, dataset ingest, and experiment orchestration outside JIT
- target the hourly or sub-daily physical kernels for JIT first
- implement the tridiagonal soil solver as an explicit JAX-compatible Thomas algorithm, likely via `lax.scan` and batching across land columns
- treat `jax.checkpoint` / rematerialization as a required design element, not an optional optimization, for any later calibration workflow that differentiates through long `lax.scan` integrations

---

## Forcing Strategy

### Hard Requirement

The rewritten model must support:

1. externally provided sub-daily forcing
2. legacy internal weather generation
3. protocol-driven atmospheric composition and deposition forcing
4. reproducible experiment branching with noleap calendars and fixed-year ancillary drivers

without making the core land-surface code messy.

### Canonical Forcing Interface

All forcing providers should emit the same canonical sub-daily forcing object with fields such as:

- air temperature
- specific humidity
- wind speed
- surface pressure
- rainfall rate
- snowfall rate
- downwelling longwave
- direct and diffuse shortwave, or a canonical broadband form plus split logic
- solar zenith cosine
- optional cloud fraction and relative humidity

In addition, the driver layer should expose separate time-varying forcing channels that are not meteorology:

- atmospheric composition forcing, especially prescribed `CO2`
- deposition forcing, especially nitrogen deposition
- land-use / management forcing
- disturbance-driver forcing such as lightning or population if those are added later

### Forcing Providers

The architecture should prioritize externally provided forcing as the long-term primary pathway, while still preserving the original IBIS weather generator as a compatibility provider.

#### Legacy Weather Provider

Implements the legacy path based on:

- `readit`
- `rdanom`
- `rdday`
- `daily`
- `diurnal`

This preserves the original IBIS weather-generation mode as an optional forcing pathway.

#### External Sub-daily Provider

Drives the model directly from sub-daily datasets.

Adapters should be implemented for:

- WIEMIP CRUJRA-based 6-hourly protocol forcing
- GSWP3
- CRU-NCEP style products

In practice, the first external implementation should be designed around the available GSWP3 archive and the identified static datasets:

- `/shared/nas2/Min/GSWP3/` for forcing
- `/shared/nas2/Min/HWSD/soita.nc` for soil texture
- `/shared/nas2/Min/HWSD/surta.nc` for land mask
- `/shared/nas2/Min/ESACCI_LC/ibis_vegtype/` for vegetation structure and PFT fractions

Because the available GSWP3 archive is currently daily rather than sub-daily, the forcing layer should be built so that:

- the canonical forcing schema is external-forcing-first
- daily GSWP3 bridge preprocessing can be implemented immediately
- WIEMIP 6-hourly datasets can serve as the first production external sub-daily pathway
- true sub-daily GSWP3 can slot in later without changing the physics core
- the legacy weather generator remains available for strict parity experiments and for cases where only daily boundary data are available

### Dedicated WIEMIP Forcing Adapter

The WIEMIP protocol is not a GSWP3 run mode. It requires its own dedicated forcing adapter over the provided CRUJRA-based 6-hourly fields distributed via Wasabi S3.

The first production external adapter should therefore be a WIEMIP-specific provider that ingests:

- `tmp`
- `dlwrf`
- `dswrf`
- `pre`
- `pres`
- `spfh`
- `wind`

and maps them to the canonical forcing object with:

- temperature from `tmp`
- longwave from `dlwrf`
- shortwave from `dswrf`, plus required broadband-to-model partitioning
- precipitation rate from `pre`
- pressure from `pres`
- humidity from `spfh`
- wind speed from `wind`
- rain / snow partitioning derived from temperature
- solar geometry such as `coszen` computed from time and location

### Design Rules

- The land-surface physics core must not care whether forcing came from an external dataset or the legacy generator.
- Unit harmonization, interpolation, phase partitioning, and dataset-specific quirks belong in forcing adapters or preprocessors, not in the science kernels.
- Daily variables needed by phenology or soil biogeochemistry should be computed by explicit reducers from sub-daily forcing or sub-daily model output.
- Meteorological forcing must remain separate from experiment drivers such as `CO2`, nitrogen deposition, and land-use controls, because WIEMIP changes them independently across `CTRL`, `BGC`, `COU`, `RAD`, and factorial runs.

### Practical External Forcing Issues to Handle

- calendars and leap years
- time zone / timestamp conventions
- precipitation accumulation versus rate
- air temperature units
- humidity variable types and conversions
- wind reference height differences
- pressure units
- shortwave / longwave availability
- direct versus diffuse radiation availability
- visible / near-IR partitioning
- rain / snow partitioning
- land mask and grid alignment
- missing data policy
- interpolation onto model timestep

### Practical GSWP3 Mapping Notes

When direct sub-daily GSWP3 is available, the provider should map close to the IBIS boundary needs:

- `tas` -> air temperature
- `huss` -> specific humidity
- `ps` -> surface pressure
- `wind` -> wind speed
- `rlds` -> downwelling longwave
- `pr` + temperature thresholding -> rain / snow partition
- `rsds` -> shortwave, followed by visible / near-IR split and direct / diffuse partitioning
- computed astronomy -> solar zenith cosine

If cloud fraction is needed by a kernel, it should be derived within the forcing adapter from radiation diagnostics rather than inserted into the physics core as dataset-specific logic.

### WIEMIP Climate and Driver Mapping Notes

The WIEMIP protocol imposes several forcing and experiment-control requirements that should be designed into the new architecture early:

- use a noleap / `365_day` calendar for all protocol experiments
- support a stable 6-hourly climate mode whose first 20 years are recycled for long spin-up and control runs
- support dynamic 6-hourly climate datasets for `COU` and optional `RAD`
- support prescribed annual or time-series `CO2` input from file rather than a hard-coded internal formula
- support both fixed and transient nitrogen deposition drivers
- keep land-use and wood-harvest controls outside the core land model, even if the first WIEMIP implementation uses fixed-1850 settings

The current legacy example configuration needs explicit override in WIEMIP mode:

- `ibis.infile` currently sets `co2init = 0.000350`, or `350 ppm`
- WIEMIP pre-industrial experiments must instead use `280 ppm`
- the rewrite should therefore treat atmospheric composition as an experiment driver, not a carry-over from legacy local input files

### Stable-Climate Recycling Mechanics

The WIEMIP stable forcing path should be handled explicitly in the driver, not informally in preprocessing.

Recommended mechanics:

- define the stable dataset cycle length as `20 years x 365 days = 7300 days`
- during spin-up and `CTRL`, advance simulation time monotonically but compute forcing indices modulo the 20-year stable cycle
- store or deterministically reconstruct the forcing-cycle phase in restart metadata so branch runs preserve the correct climate phase
- make branch behavior explicit in `ExperimentSpec`, with WIEMIP defaults preserving the parent forcing-cycle position at the branch point
- keep the forcing clock entirely on a `365_day` calendar with no leap days

This avoids hidden calendar drift during centuries of spin-up and ensures `CTRL` and `BGC` can branch cleanly from the same spun-up state.

### Daily-to-Subdaily Bridge for External Daily Forcing

There is an important sequencing gap between the currently available daily GSWP3 archive and the sub-daily forcing required by the physics core.

The plan should make the bridge explicit:

- use the WIEMIP-provided 6-hourly stable and dynamic datasets as the first production external sub-daily pathway for protocol experiments
- implement a provisional deterministic daily-to-subdaily adapter for local daily GSWP3, rather than implying that daily GSWP3 is already a parity-grade sub-daily product
- keep strict parity validation anchored on the legacy weather path and on direct sub-daily external datasets, not on the provisional daily GSWP3 bridge

Where practical, the deterministic bridge should reuse non-stochastic pieces of the legacy sub-daily forcing logic, especially:

- zenith-weighted shortwave distribution
- deterministic temperature interpolation over the diurnal cycle
- configurable precipitation timing or windowing rules
- explicit humidity and wind interpolation policies

This bridge should be treated as an adapter-layer approximation until true sub-daily GSWP3 is available.

---

## Differentiability Strategy

### Likely Good Early Targets

- radiation
- canopy physics
- photosynthesis and stomata
- soil hydrology and soil temperature
- snow thermodynamics
- biomass pool updates
- much of soil biogeochemistry

### Likely Nondifferentiable or Problematic Components

- random weather generation
- wet/dry day logic
- random precipitation event timing
- categorical texture classification
- PFT existence logic
- vegetation type mapping
- fire disturbance
- restart and I/O pathways

### Recommendation

- Keep the core physical and biogeochemical kernels differentiable where practical.
- Keep forcing providers, categorical mapping, and I/O outside gradient paths.
- Preserve hard thresholds initially in legacy compatibility mode.
- Only introduce smoothing after faithful parity is achieved and only in explicitly opt-in modes.

---

## Performance and Spin-up Strategy

### Likely Bottlenecks

- canopy iterative solve with fixed 3 iterations
- dense canopy linear system solves
- soil hydrology and temperature tridiagonal solves
- repeated sub-daily reductions and accumulators
- fragmented restart and output I/O
- brute-force repeated `soilbgc` calls during spin-up

The canopy kernel deserves special treatment as the highest-risk scientific and implementation component. `canopy.f` is 2,397 lines and couples radiation, stomata, turbulence, and leaf energy balance through a fixed-iteration implicit solve. It should be derisked with an isolated prototype and checkpoint harness before the full canopy phase is considered complete.

### Low-risk Optimizations

- keep land-only arrays throughout the core computation
- precompute static geometry and index maps
- use JAX `jit` on stable kernels
- drive long loops with `lax.scan`
- make diagnostics optional and configurable
- reduce output fragmentation

### Medium-effort Optimizations

- use batched JAX linear algebra for canopy and soil solvers
- separate sub-daily physics from daily/annual ecology
- compute only required diagnostics for a given experiment
- chunk large domains for memory-efficient execution

### High-payoff Ideas

- batch over sites, ensembles, or parameter sets
- accelerate long scans on GPU
- use `jax.checkpoint` / rematerialization systematically for differentiable long runs
- replace brute-force spin-up with staged or convergence-based workflows after parity

### Spin-up Strategy

#### Phase 1

Reproduce the current spin-up exactly, including:

- `spincons`
- `spinfrac`
- `eqyears`
- `nspinsoil`
- counting only the final spin cycle for selected annual totals

#### Phase 2

After parity:

- develop faster soil equilibrium strategies
- consider partially decoupled BGC spin-up
- use convergence criteria instead of fixed years where scientifically acceptable

For WIEMIP specifically, the spin-up system should also support:

- long stable-climate recycling without calendar drift
- branchable restart generation for `CTRL`, `BGC`, `COU`, and factorial descendants
- optional process-specific spin-up strategies when new slow pools are added, especially permafrost carbon, wetland water-table memory, and inland-water storage

---

## Modern Earth System Model Design Ideas Most Valuable Here

These are especially valuable given the observed IBIS codebase:

- explicit separation of prognostic state, diagnostics, parameters, forcing, and static metadata
- cleaner process coupling interfaces
- restart / checkpoint infrastructure as a coherent subsystem
- budget and conservation diagnostics built into the framework
- configuration-driven experiments
- science kernels separated from I/O
- reproducible experiment manifests
- benchmark cases and standard validation harnesses
- same core for site-scale and grid-scale runs
- calibration / data-assimilation readiness
- easier ensemble execution
- better observability and debug diagnostics

These should be introduced carefully:

- preserve parity first
- then layer architectural improvements without changing baseline behavior

---

## WIEMIP Implications and Coupling-Aware Extension Strategy

The WIEMIP protocol changes the rewrite problem in two important ways:

1. it requires a protocol-aware experiment system, not just a translated land model
2. it raises the chance that additional processes will be added soon after baseline porting, so extension architecture matters immediately

### Protocol Requirements That Affect Architecture

From `WIEMIP_1PCT_Protocol_20250330.md`, the rewrite should explicitly support:

- `SPIN`, `CTRL`, `BGC`, `COU`, and optionally `RAD`
- factorial process on/off experiments
- optional transient nitrogen deposition experiments
- multiple climate-pattern ensembles
- fixed-1850 land use and disabled wood harvest for protocol runs
- 150-year noleap simulations
- 6-hourly forcing
- restart branching from a common spun-up state
- netCDF4 output with protocol-controlled naming, metadata, and frequency

These are not just output settings. They affect forcing, restart lineage, driver separation, and process toggles.

### Output Compliance Must Start Early

WIEMIP output compliance should not be deferred to the end of the project.

The plan should explicitly include an early crosswalk from the WIEMIP variable-request spreadsheet to:

- existing IBIS diagnostics and output streams
- derived quantities that can be computed from internal fluxes or states
- variables that are currently missing and need new diagnostics or science support

The WIEMIP output subsystem should enforce:

- naming convention of the form `<MODEL>_<experiment>_<variable>_<frequency>_<resolution>.nc`
- consistent global grid and dimension ordering
- CF-style metadata and correct units
- compression enabled
- missing-data fill value of `-99999`
- experiment and process-toggle labels such as `noFire` or interaction tags where required

This crosswalk should begin in Phase 0 and be refined in Phase 1, not left until production time.

### What the Current IBIS Code Appears to Already Have

Observed in the inspected repository:

- a noleap calendar structure based on fixed monthly day counts in `main.f`
- dynamic vegetation via `pheno` and `dynaveg`
- a fire pathway via `isimfire`, `firefac`, and `fire`
- soil carbon and nitrogen pools in `biogeochem.f` and `comveg.h`
- soil thermal and freeze-state physics through `soil.f` and `snow.f`

### What Appears Weak, Partial, or Missing

Observed or strongly suggested by source inspection:

- fire exists, but `READMEnet.txt` explicitly says the developers were not satisfied with the dynamic fire scheme and recommend not using it
- the nitrogen pathway is partial: `soilbgc` computes soil and litter nitrogen dynamics and `tnmin`, but there is no clear evidence of plant nitrogen uptake or a fully interactive plant nitrogen limitation loop
- nitrogen deposition is currently internal and empirical, computed from precipitation in `biogeochem.f`, not read from an external transient deposition dataset
- the current `co2` routine in `physiology.f` simply returns `co2init`, so WIEMIP-style prescribed `1pctCO2` time-series forcing is not yet represented
- no clear evidence was found for explicit permafrost, wetlands, BVOC, inland waters, LUH-style land use, wood harvest, lightning, or population modules in the current source tree

Inference:

- the current IBIS code provides a useful base for vegetation, soil, snow, canopy, and soil C/N work, but several WIEMIP-priority factorial processes would need real model development rather than simple translation

### Coupling-Aware Design Rule

Future process modules should not be added as side effects inside existing vegetation or soil routines. Instead, each new process should declare:

- the state it owns
- the forcing and parameters it reads
- the fluxes or tendency terms it exports
- the budgets it affects
- the timescale it runs on
- the process toggles that activate it

This is especially important because the WIEMIP-priority additions couple across different timescales.

### Couplings That Must Be Designed Explicitly

#### Interactive Nitrogen

Current code evidence:

- `biogeochem.f` contains litter and soil nitrogen pools, mineralization calculations, and internally generated deposition / fixation terms
- `storedn` is accumulated, but no clear downstream plant-uptake usage was found in the scanned code

For the rewrite, interactive nitrogen should be designed as a true coupling among:

- atmospheric deposition forcing
- biological fixation
- soil mineral nitrogen state
- microbial immobilization and mineralization
- plant nitrogen uptake
- photosynthesis and stomatal limitation
- allocation, litterfall, and biomass turnover
- leaching
- fire nitrogen losses if fire remains interactive

Architectural recommendation:

- create a dedicated `nitrogen` process layer and state group rather than leaving nitrogen implicit inside `soilbgc`
- separate deposition forcing from internal transformations
- keep plant nitrogen limitation as an explicit interface between physiology / allocation and soil biogeochemistry

#### Permafrost

No explicit permafrost module was found, but the existing code already has strong prerequisites:

- snow physics
- soil temperature
- soil ice and liquid water

Permafrost should therefore be designed as a coupling extension across:

- snow insulation
- soil thermal diffusion and freeze-thaw
- hydrologic permeability and runoff / drainage
- decomposition and carbon turnover
- vegetation establishment and survival at cold sites
- fire fuel availability and post-fire recovery
- wetland / inundation behavior if wetlands are added

Architectural recommendation:

- do not add permafrost logic as special cases inside generic soil thermal code
- instead define a frozen-ground or permafrost extension module that reads soil thermal and hydrology state and returns permeability, active-layer diagnostics, and decomposition modifiers

#### Wetlands

No clear wetland or water-table module was found in the current repository scan.

If wetlands are added, they should couple to:

- surface and subsurface hydrology
- snowmelt and runoff timing
- soil temperature and oxygen availability
- anaerobic versus aerobic decomposition pathways
- vegetation composition
- fire suppression and post-drought flammability shifts

Architectural recommendation:

- represent wetlands through an explicit water-table / inundation state and hydrology coupling interface
- keep methane or anaerobic chemistry extensions downstream of that state, rather than embedding them directly in generic soil C code

#### Inland Waters

No clear inland-water routing or storage module was found.

Inland waters should be treated as a lateral-export extension, not as a patch inside the column physics:

- runoff and drainage generated by soil hydrology feed lateral transport
- dissolved or particulate carbon and nitrogen exports branch from land budgets
- inland-water storage, processing, and outgassing can evolve on a separate routing graph

Architectural recommendation:

- keep the core land column one-dimensional
- expose runoff, drainage, and export-ready tracers through diagnostics or flux bundles
- let inland-water modules operate as an optional downstream component driven by those fluxes

#### BVOC

No clear BVOC implementation was found.

BVOC should be designed as a mostly canopy-diagnostic extension that couples to:

- absorbed radiation
- leaf temperature
- PFT identity and biomass / leaf area
- phenology
- water and heat stress

Architectural recommendation:

- keep BVOC emission factors and chemistry outside the core photosynthesis solver
- use canopy diagnostics as inputs to a separate `bvoc` module so BVOC can be enabled or disabled without disturbing carbon balance kernels

#### Fire Redesign

Fire exists in the legacy code, but the current implementation appears too weak to be the long-term architecture for WIEMIP factorial work.

Current couplings already visible in the code:

- `stats.f` derives `firefac` from upper-soil moisture
- `dynaveg` applies annual disturbance and optional `fire`
- carbon losses are recorded in `cdisturb`

A stronger fire design should couple explicitly to:

- fuel loads from vegetation biomass and litter pools
- fuel moisture from soil, snow, and canopy states
- meteorological fire weather from forcing
- ignition drivers, potentially including lightning and population
- carbon and nitrogen emissions
- vegetation mortality and post-fire succession

Architectural recommendation:

- separate fire weather, ignition, spread / burn fraction, and emissions / mortality into explicit components or substeps
- preserve legacy fire as a compatibility mode, but design the new interfaces so a better fire model can replace it cleanly

#### Land Use and Harvest

The WIEMIP protocol fixes land use at 1850 and disables wood harvest, so these are not first-wave science additions for the protocol runs. But the architecture should still reserve a clean place for them because they couple strongly to vegetation and carbon bookkeeping.

Architectural recommendation:

- keep land-use transitions and harvest outside natural-vegetation dynamics
- separate natural PFT composition from managed land-cover fractions in the long-term state design

### Recommended Extension Order After Parity

For WIEMIP readiness, the most valuable controlled modernization path is:

1. protocol-aware experiment manager and output system
2. prescribed `CO2` and nitrogen deposition driver support
3. fire redesign or at least fire-interface cleanup
4. interactive nitrogen coupling between plant and soil systems
5. permafrost extension
6. wetlands
7. inland waters
8. BVOC

This order follows both the WIEMIP priorities and the strongest coupling dependencies in the current codebase.

### Factorial Re-tuning Strategy

The WIEMIP protocol explicitly asks groups to re-tune process-off factorial runs to remain comparable to the baseline.

This should be built into the plan as a controlled calibration workflow:

- define a small approved parameter subset for each factorial family, such as mortality, turnover, or disturbance scalars
- optimize factorial variants against baseline stocks and fluxes, not against an unrelated external target
- keep re-tuning separate from parity mode; parity is against legacy behavior, while factorial re-tuning is against the chosen WIEMIP baseline configuration
- use differentiable calibration where the relevant kernels are ready, but allow simpler bounded tuning methods if a full gradient path is not yet reliable

Given the June 17, 2026 deadline, factorial re-tuning should be prioritized for the highest-value near-term experiments only, especially the baseline `Main` configuration and any first `noFire` or nitrogen-related variants that are scientifically defensible.

### Deadline-Oriented Delivery Scope

As of April 10, 2026, there are roughly 68 days until the WIEMIP submission deadline of June 17, 2026.

That timeline is too short for the full Phase 0-10 vision plus multiple new science modules. The plan should therefore distinguish:

- first-submission scope
- post-submission scope

Recommended first-submission scope:

- required baseline `SPIN`, `CTRL`, `BGC`, and `COU`
- the `Main` process configuration first
- protocol-compliant outputs and restart lineage
- optional `RAD` only if baseline production is already stable
- optional `noFire` factorial if baseline production is complete early enough
- optional `BGC-ndep` and `COU-ndep` only if nitrogen forcing support and scientifically meaningful nitrogen coupling are ready in time

Recommended post-submission scope:

- permafrost
- wetlands
- inland waters
- BVOC
- major fire-science redesign beyond what is needed for initial factorial work

The main architectural plan remains valuable, but those later modules should be treated as post-June 17, 2026 development unless the schedule changes materially.

---

## Strict Checkpoint-Based Validation Workflow

Validation is not a final step. It is the migration workflow.

The rewrite should follow this gated ladder:

`function green -> module green -> interface green -> one-step green -> one-day green -> 10-day green -> one-month green -> seasonal green -> annual green -> restart green -> spin-up green -> modernization allowed`

No downstream work should proceed until the current checkpoint level passes.

### 1. Function-level Checkpoints

Each rewritten function receives a checkpoint definition containing:

- exact legacy inputs
- expected touched outputs
- fields that must remain unchanged
- units
- shapes
- invariants
- tolerance class
- required golden cases

Examples:

- math helpers
- array pack/unpack helpers
- saturation functions
- soil and canopy helper kernels
- weather-generator helper logic
- small numerical solvers

Function checkpoints should compare:

- direct return values for pure helpers
- pre-call and post-call state snapshots for stateful routines

Progress rule:

- A function is not complete until its checkpoint passes on all required golden cases.

### 2. Module / Process-level Checkpoints

After all functions in a process area are green, the whole process receives a module checkpoint.

Required process checkpoints:

- static input and forcing ingestion
- climate analysis / PFT existence
- radiation
- canopy / interception / physiology
- snow
- soil hydrology
- soil temperature
- phenology
- soil biogeochemistry
- annual vegetation dynamics / fire / vegetation mapping
- stats and time accumulators
- initialization and restart

Progress rule:

- A module is not complete until all of its functions are green and the module checkpoint passes.

### 3. Coupling / Interface Checkpoints

Because the original code has hidden coupling through common blocks, explicit interface checkpoints are required between modules.

Critical interface checkpoints:

- forcing provider -> land-surface forcing object
- radiation -> canopy
- canopy -> snow / soil
- stats -> phenology
- stats -> soil biogeochemistry
- climate-memory -> vegetation dynamics
- restart write -> restart read

Each interface checkpoint should verify:

- exact field set and names
- units
- touched versus untouched fields
- no accidental state drift across boundaries

Progress rule:

- No adjacent module pair is complete until their coupling checkpoint passes.

### 4. Runtime Horizon Checkpoints

The rewrite should include staged integration checkpoints:

- one-step checkpoint
- one-day checkpoint
- 10-day checkpoint
- one-month checkpoint
- seasonal checkpoint
- annual checkpoint
- restart / continuation checkpoint
- spin-up checkpoint

#### One-step

One sub-daily land-surface step with frozen forcing and frozen pre-state.

#### One-day

One full day including daily forcing generation, daily BGC, and all sub-daily land-surface steps.

#### 10-day

Needed because running means drive phenology.

#### One-month

Captures daily accumulation behavior and anomaly / climatology transitions.

#### Seasonal

Checks regime behavior through wet/dry or snow/no-snow transitions.

#### Annual

Checks annual budgets, annual climate-memory updates, annual vegetation updates, and restart behavior.

#### Spin-up

Checks the legacy accelerated soil-BGC workflow over:

- early acceleration phase
- taper phase
- post-acceleration phase
- long-run pool trajectories and quasi-equilibrium behavior

Progress rule:

- The next horizon cannot begin until the previous horizon passes.

### 5. Golden Cases, Baseline Outputs, Intermediate-State Snapshots, and Comparison Harnesses

#### Golden Test Cases

At minimum maintain:

- helper-function fixture cases
- single-column deterministic cases
- tiny multi-point subset cases
- legacy weather-generator cases with fixed seed
- restart roundtrip cases
- annual vegetation update cases
- soil spin-up cases
- later, external sub-daily forcing cases for GSWP3 and CRU-NCEP

#### Baseline Outputs

For each golden case, store:

- exact legacy inputs
- exact parameter files
- selected output files
- selected state trajectories
- restart files where relevant
- build metadata for the reference Fortran run

#### Intermediate-state Snapshots

Capture snapshots:

- before and after each target function where practical
- at module boundaries
- at key time checkpoints
- at restart write/read boundaries

These snapshots are essential for localizing mismatches.

#### Automated Comparison Harness

The harness should:

- run the legacy reference
- run the rewritten code
- align fields through a common registry
- compare touched fields
- verify untouched fields remain unchanged
- apply field-specific tolerance rules
- emit pass/fail reports and quantitative difference summaries

The harness should also compute and enforce minimum required balance checks:

- water balance residual
- energy balance residual
- carbon balance residual
- nitrogen balance residual once interactive nitrogen is implemented

These should be treated as checkpoint criteria, not optional diagnostics, because conservation failures localize kernel and coupling errors quickly.

### 6. Tolerance Rules

Use strict tolerance classes during parity mode, but calibrate them for a `float32` default implementation. Retain an optional `float64` debug mode only for targeted investigations where the source of a mismatch is unclear.

Suggested classes:

- exact: integers, flags, indices, land ordering, dates, PFT ids, categorical vegetation outputs
- function-tight: helper math and solver kernels, typically around `rtol=1e-6`
- step-tight: single-step continuous states and fluxes, typically around `rtol=1e-5`
- accumulation-tight: daily / monthly reductions, typically around `rtol=1e-4`
- annual-tight: annual totals and annual state summaries, typically around `rtol=1e-3`
- spin-up-trajectory: long-horizon pool and equilibrium trajectory comparisons

Any tolerance relaxation should require:

- an explicit written reason
- named affected variables
- named affected checkpoints

### 7. Hard Gating Rules

The migration should be blocked by failed checkpoints.

Rules:

- do not move from a function to the next function if the current function checkpoint fails
- do not move from a module to the next module if the module or adjacent interface checkpoint fails
- do not move from one runtime horizon to the next if the current horizon fails
- do not apply modernization or performance redesign until the affected parity ladder is green

For any modernization or performance change:

- rerun affected function checkpoints
- rerun affected module checkpoint
- rerun adjacent interface checkpoints
- rerun shortest relevant runtime horizons
- rerun annual and spin-up checkpoints when core physics or ecology are touched

### 8. GitHub Checkpoint Discipline

Key project checkpoints should also be version-control checkpoints.

The plan should require pushing the work to a GitHub repository at agreed milestone states so the project always has a reproducible remote history, backup, and rollback point.

For this project, the designated rewrite repository should be:

- `https://github.com/minchen-wisc/ibis-jax.git`

The original IBIS repository or local source tree should remain the reference implementation and source-inspection baseline, but rewrite milestones, branches, and tags should be published in `ibis-jax`.

Recommended repository discipline:

- use short-lived feature branches for active work
- merge only after the relevant checkpoint chain is green
- push milestone branches or merges immediately after each approved checkpoint
- tag major scientific states so they can be rerun later without ambiguity
- keep validation reports, manifests, and experiment metadata linked to the corresponding commit or tag
- record which legacy IBIS source revision or local source snapshot was used as the comparison baseline for each major `ibis-jax` milestone

Recommended push and tag milestones:

- after Phase 0 baseline harness and legacy reference cases are working
- after Phase 1 state, parameter, and solver schemas are stable
- after Phase 2 forcing and driver ingestion is working for the first approved pathway
- after Phase 5 once the sub-daily physical core is green on the required checkpoints
- after Phase 6a once daily biogeochemistry and spin-up checkpoints are green
- after Phase 8 full legacy-parity approval
- after Phase 9 WIEMIP protocol production readiness
- at the exact commit used for any WIEMIP submission runs

Suggested GitHub tag pattern:

- `baseline-harness-v1`
- `schema-v1`
- `forcing-v1`
- `physics-core-v1`
- `bgc-spinup-v1`
- `legacy-parity-v1`
- `wiemip-prod-v1`
- `wiemip-submission-2026-06-17`

These repository checkpoints should be treated as part of the workflow, not as optional housekeeping after the science work is done.

---

## Migration Strategy: Preserve Behavior First, Then Modernize

### Phase A: Parity-preserving translation

- preserve process order
- preserve effective legacy behavior
- preserve restart semantics
- preserve legacy weather generation as an optional provider
- preserve current field naming and meaning where useful
- use strict checkpoint-based validation throughout
- use the Fortran 2008 modular tests and modules as supporting fixtures and schema references where helpful, while treating Python + JAX as the production rewrite target

### Phase B: Controlled modernization

- replace common blocks with explicit PyTrees / typed state objects
- separate science kernels from I/O
- clean restart and output architecture
- clean forcing architecture
- improve diagnostics and experiment configuration

### Phase C: Optional enhancements

- improve differentiability of selected pathways
- use `jax.checkpoint` / rematerialization as a prerequisite whenever gradients through long time integrations are required
- improve spin-up algorithms
- add pluggable process extension mechanisms
- add ensemble and calibration workflows

---

## Revised Roadmap with Embedded Checkpoints

### WIEMIP Critical-Path Schedule

For the June 17, 2026 WIEMIP submission, the architectural roadmap needs an explicit calendar plan.

Recommended aggressive critical path:

- April 10 to April 18, 2026: Phase 0 baseline setup, S3 access, output crosswalk, traceability matrix, and checkpoint harness
- April 19 to April 26, 2026: Phase 1 schemas, constants, utilities, solver scaffolding, and experiment-driver definitions
- April 27 to May 8, 2026: Phase 2 WIEMIP forcing adapter, stable-climate cycling, prescribed `CO2` support, nitrogen-deposition driver support, and protocol-output skeleton
- May 9 to May 24, 2026: Phases 3 to 5 for the sub-daily physical core needed for baseline runs
- May 25 to June 2, 2026: Phase 6a and Phase 7 for daily BGC, annual vegetation, restart, and baseline integration tests
- June 3 to June 7, 2026: Phase 8 parity approval for the minimum WIEMIP submission path
- June 8 to June 12, 2026: Phase 9 protocol production plumbing, baseline run launch, and output QA
- June 13 to June 17, 2026: reruns if needed, final diagnostics, metadata checks, packaging, README preparation, and submission

This is an aggressive schedule. It is appropriate only for the first-submission scope described above, not for the full long-term feature set.

A hard go or no-go decision should be built in:

- by May 9, 2026, decide whether the WIEMIP submission will be generated from the rewritten engine or whether a legacy or modern-Fortran contingency path is needed for the June 17, 2026 deadline

That contingency decision protects the scientific submission from architectural overreach.

### Phase 0: Legacy Baseline and Validation Harness

- assemble runnable reference cases
- gather all required parameter and runtime input assets
- obtain Wasabi S3 credentials and stage the WIEMIP forcing assets needed for the first production path
- freeze golden test cases
- define tolerance registry
- build snapshot and comparison harness
- perform a subroutine I/O audit for the routines in `lsxmain`, daily ecology, annual ecology, restart, and forcing generation
- use the existing `/home/min/IBIS/tests` infrastructure where it is helpful, rather than creating a completely separate testing stack
- build the WIEMIP variable-request crosswalk early and identify missing diagnostics before the production deadline is near
- push the completed baseline harness state to GitHub and tag it before substantial translation begins

Gate:

- no rewrite starts until the harness and baseline cases are working

### Phase 1: Utilities, Solvers, and State Registry

- port utility helpers
- port numerical helpers and solvers
- define explicit state / parameter / forcing schemas
- define the large vegetation schema from `comveg.h` early rather than leaving it implicit until late phases
- define the `CouplingContext` / intermediate state schema based on `com1d.h`
- make the early JAX structural choices up front: PyTree container style, precision policy, tridiagonal solver approach, and initial JIT boundaries
- define the experiment-driver schema early, including meteorology, atmospheric composition, deposition, and process-toggle controls
- push the first stable schema and solver milestone to GitHub once the Phase 1 checkpoint chain is green

Checkpoint policy:

- validate after each function
- then validate module-level utility / solver checkpoint

### Phase 2: Forcing and Input Path

- implement static input loading for the identified HWSD, ESA CCI, and land-mask datasets
- define canonical forcing objects
- implement the external-forcing-first ingestion path around WIEMIP 6-hourly and GSWP3-style fields
- implement the first working external sub-daily adapters for WIEMIP stable and dynamic 6-hourly datasets
- implement a dedicated WIEMIP forcing adapter for the protocol variables `tmp`, `dlwrf`, `dswrf`, `pre`, `pres`, `spfh`, and `wind`
- implement daily GSWP3 preprocessing immediately, with an explicit deterministic daily-to-subdaily bridge until true sub-daily GSWP3 is available
- implement legacy anomaly ingestion and the legacy weather provider as a compatibility path required for strict parity runs
- add prescribed-driver support for file-based `CO2` and nitrogen deposition inputs
- design the forcing system so WIEMIP stable-climate recycling and pattern-based branching are driver-layer features, not science-kernel features
- push the first approved forcing milestone to GitHub once the external and legacy forcing checkpoints are green

Checkpoint policy:

- function checkpoint after each rewritten function
- forcing module checkpoint
- forcing-to-physics interface checkpoint
- one-day and one-month forcing-only comparison cases
- external-forcing ingest checks against the identified shared datasets
- legacy weather-generator checks for parity cases
- protocol-driver checks for noleap calendars, stable-climate cycling, and prescribed `CO2` time series ingestion
- adapter-level checks for the provisional daily-to-subdaily GSWP3 bridge, clearly separated from strict parity checkpoints

### Phase 3: Radiation

- port radiation functions in legacy order

Checkpoint policy:

- function checkpoint after each function
- radiation module checkpoint
- radiation-to-canopy interface checkpoint
- one-step checkpoint

### Phase 3b: Canopy Risk-Reduction Prototype

- build an isolated canopy prototype around frozen legacy snapshots
- test JAX compatibility of the fixed-iteration implicit canopy solve
- confirm the planned solver and PyTree structure can support this kernel cleanly before the full canopy port proceeds

Checkpoint policy:

- canopy prototype must pass focused function and one-step checks before the full canopy module phase proceeds

### Phase 4: Canopy / Interception / Physiology

- port canopy support functions
- port `stomata`
- port `turcof`
- port `turvap`
- port `canopy`

Checkpoint policy:

- validate after each function
- canopy module checkpoint
- coupling checkpoint to soil / snow
- one-step and one-day checkpoints

### Phase 5: Snow and Soil Physics

- port snow module
- port soil hydrology and thermal modules
- push the first full sub-daily physics-core milestone to GitHub once the Phase 5 checkpoint chain is green

Checkpoint policy:

- validate after each function
- module checkpoints
- canopy-to-snow / canopy-to-soil interface checkpoints
- one-step, one-day, and short-run checkpoints

### Phase 6a: Climate Memory, Stats, Phenology, and Soil Biogeochemistry

- port climate analysis
- port reducers / accumulators
- port phenology
- port soil biogeochemistry

This phase is intentionally separated from annual vegetation dynamics so the daily `soilbgc` loop and spin-up behavior can be validated and optimized before the annual vegetation and disturbance layer is ported.

- push the first biogeochemistry and spin-up milestone to GitHub once the Phase 6a checkpoint chain is green

Checkpoint policy:

- validate after each function
- module checkpoints
- stats-to-phenology checkpoint
- stats-to-bgc checkpoint
- 10-day, one-month, seasonal, and annual checkpoints

### Phase 7: Annual Vegetation Dynamics and Restart

- port annual vegetation dynamics
- port fire and vegetation mapping
- port initialization and restart

Checkpoint policy:

- validate after each function
- module checkpoints
- restart roundtrip checkpoint
- annual and restart continuation checkpoints

### Phase 8: Full Legacy-Parity Approval

- run the full checkpoint ladder on the approved golden cases
- certify legacy-compatibility mode
- push and tag the legacy-parity-approved state in GitHub immediately after approval

Gate:

- no modernization begins until the full parity ladder is green

### Phase 9: Protocol-Ready Infrastructure

- add a WIEMIP-aware experiment manager for `SPIN`, `CTRL`, `BGC`, `COU`, `RAD`, factorial, and `ndep` variants
- add branch-aware restart lineage handling
- add protocol-compliant output writing, metadata, naming, and frequency control
- productionize the already-implemented external forcing pathways inside the protocol experiment system, rather than introducing them for the first time here
- push and tag the WIEMIP production-ready state in GitHub before any submission-scale runs are launched

Checkpoint policy:

- every protocol feature reruns the affected checkpoint chain
- branch and restart lineage tests are mandatory before factorial production use

### Phase 10: Controlled Modernization and Missing-Process Development

- introduce cleaner architecture and JAX performance features
- extend and harden external forcing adapters beyond the already-working Phase 2 providers, including true sub-daily GSWP3 when available and CRU-NCEP-style products
- improve restart and experiment management
- add missing WIEMIP-relevant processes in a coupling-aware order:
  - upgraded fire interfaces and optionally improved fire science
  - prescribed and interactive nitrogen coupling
  - permafrost
  - wetlands
  - inland waters
  - BVOC
- later improve differentiability and spin-up

Checkpoint policy:

- every redesign reruns the affected checkpoint chain
- every major post-parity redesign that passes its checkpoint chain should be pushed to GitHub as a new tagged milestone

---

## Immediate Next Steps

1. Assemble a runnable legacy reference workspace with:
   - `params.can`
   - `params.soi`
   - `params.veg`
   - representative `input/`, `restart/`, and anomaly datasets
   - WIEMIP Wasabi S3 credentials and a staged local mirror of the first required forcing and driver assets
2. Define the first small golden cases:
   - one-column deterministic case
   - tiny multi-point subset case
   - legacy weather-generator case with fixed seed
   - restart case
   - soil spin-up case
3. Build a machine-readable inventory of:
   - state fields
   - parameters
   - forcing variables
   - process inputs / outputs
   - read / write sets for each target subroutine
   - intermediate coupling variables from `com1d.h`
4. Finalize the early implementation decisions:
   - default precision policy
   - PyTree container approach
   - tridiagonal solver strategy
   - initial JIT boundaries
5. Implement the checkpoint harness before starting the actual port.
6. Build a WIEMIP traceability matrix mapping:
   - protocol experiments
   - required external drivers
   - required outputs
   - current IBIS capabilities
   - clearly missing or weak processes
7. Build a WIEMIP variable-request compliance matrix mapping requested outputs to:
   - existing IBIS outputs
   - derived diagnostics that can be computed in the rewrite
   - still-missing variables that need new diagnostics or science support
8. Decide explicitly how external daily forcing is allowed to reach the sub-daily core before true sub-daily GSWP3 is available:
   - direct WIEMIP 6-hourly forcing for protocol runs
   - provisional deterministic disaggregation for daily GSWP3
   - legacy weather path for strict parity baselines
9. Use `https://github.com/minchen-wisc/ibis-jax.git` as the designated rewrite repository, and agree on:
   - default branch and branch-protection rules
   - milestone branch naming
   - tag naming for scientific checkpoints
   - where validation reports and manifests will live in the repo
10. Set a deadline-protection milestone:
   - by May 9, 2026, choose whether the June 17, 2026 WIEMIP submission will run from the rewritten engine or from a contingency legacy or modern-Fortran path
11. Start the rewrite with utilities, solvers, schemas, the forcing abstraction, and the experiment-driver layer, not with a full-core translation.
