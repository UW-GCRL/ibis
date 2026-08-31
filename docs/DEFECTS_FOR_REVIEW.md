# Defects found in IBIS-JAX, 2026-08-12 → 2026-08-27

**Audience: anyone using the archived spin-ups or building on `main`.**

Written while adding the fire and methane modules. Several of these predate that
work and affect results already on disk. Every number below was read from the
checkpoint files, not from memory — the script is `verify/` history in the
`fire-test` branch.

---

## 1. If you read nothing else

**Do not use these archived spin-ups for anything involving vegetation or
carbon.** Measured directly from `state_latest.pkl` / `gpu_state_after_200.pkl`:

| run | `totlit` | `totcsoi` | `exist` | `gdd5` | NaN values |
|---|---|---|---|---|---|
| `spinup_200yr_0p5deg_v4_n4_carbon_monthly` | 0.0000 | 0.0000 | 1.000 | 0.0 | **140** |
| `spinup_200yr_0p5deg_v5_wsoi_floor` | 0.0000 | 0.0000 | 1.000 | 0.0 | **140** |
| `spinup_200yr_0p5deg_v6_flux_clamp` | 0.0000 | 0.0000 | 1.000 | 0.0 | 0 |
| `spinup_200yr_0p5deg_cycle_1901_2000` | 0.0000 | 0.0000 | **0.000** | 0.0 | 0 |
| `spinup_carbon_v7_soilbgc` | 3.0219 | 18.7174 | 1.000 | 0.0 | 0 |

Read that table as three separate failures:

* **Zero litter and zero soil carbon after 200 simulated years** in all four
  200-year runs. Not small — exactly 0.
* **`exist = 1.000`** means every one of the 12 PFTs was permitted in every
  grid cell, tropical broadleaf evergreen in Siberia included.
  `cycle_1901_2000` has `exist = 0.000`, the opposite failure: no PFT permitted
  anywhere.
* **`gdd5 = 0.0`** everywhere: the climate normals were never computed, which
  is *why* `exist` is wrong.

`v4` and `v5` additionally contain **140 NaN values each** (cell 71935).

A replacement spin-up is available: `spinup_fire_v8`, 200 years, 0 NaN,
converged, validated against GFED5. See §5.

---

## 2. Defects, worst first

### 2.1 The carbon chain was never complete (affects all archived runs)

IBIS's carbon cycle needs `biomass_update` (annual litterfall) **and**
`soilbgc` (daily decomposition). Neither runner had both:

* `verify/.../global_gswp3_gpu.py` — which produced the archived 200-year runs
  — called `biomass_update` annually but never `soilbgc`;
* `run_year_gswp3` called `soilbgc` daily but never `biomass_update`, so it
  decomposed a litterfall flux that was identically zero.

Result: `totlit == totalit == totcsoi == 0` exactly. **Fixed** —
`run_spinup_gswp3` in `ibis/timestepping/loops_gswp3.py` contains both.

### 2.2 Climate normals were never maintained (affects all archived runs)

`climanl` / `climanl2` / `existence` were never called by the spin-up runner,
so `tc`, `tw`, `tcmin`, `gdd0`, `gdd5` stayed at their zero initialisation and
`exist` was never set from bioclimatic limits. Vegetation therefore grew with
no biogeographic constraint at all.

**Fixed** — the runner now does `annual_climate_stats → climanl2 → existence →
dynaveg`, seeding with `climanl` on the first year when the normals are unset
(a 30-year running mean started from zero is wrong for three decades).
Measured effect: `exist` goes from 1.000 to a stable 0.341.

### 2.3 The spin-up runner never applied disturbance

`run_spinup_gswp3` called `biomass_update`, whose own docstring says it *"does
NOT apply disturbances (fire, background mortality)"*, and never called
`dynaveg`. So every result from that path had **no fire feedback of any kind**,
whatever `isimfire` was set to. **Fixed.**

### 2.4 `run_global.py` advanced biomass twice per year

It called `biomass_update` *and* `dynaveg`. Both perform the same allocation
from the same uncleared `summary.aynpp`, so each dynamic-vegetation year
advanced the pools twice — measured at **1.42×** the correct annual change on a
controlled fixture. IBIS 2.6b4 has no such split; `biomass_update` is a
port-invented subset of `dynaveg`. **Fixed**: exactly one runs per year.

### 2.5 A NaN that has recurred three times

Archived `v4`/`v5` (cell 71935), and twice more during this work (cells 69933
and 85004). Previous mitigations `v5_wsoi_floor` and `v6_flux_clamp` each
treated a symptom.

Root cause, bisected to the timestep: when the surface is 100 % dry with no
puddle, `setsoi` follows F77 `soil.f:135-160` and assigns **all** ground
evaporation to the puddle. F77's own comment concedes this *"could lead to
negative puddle depths"* and assumes it cannot matter because *"evaporation is
likely to be very small or less than zero (condensation)"*. That assumption is
written for dew. Under strong evaporative demand on bone-dry soil, `wpud` goes
to −3e−3 kg/m², and the next timestep reads the negative puddle back as a
**spurious condensation flux**:

```
step  ta      solad    tg       tsoi0     heatg      wpud
   2  278.54    0.00   274.953  275.576    -3.32   -0.003026
   3  278.23    0.00   273.582  284.248  +338.83    0.000000
```

+339 W/m² into the ground at night with zero insolation. Repeated, it drove the
thermal solve to 324.7 K on CPU and NaN on GPU.

**Fixed**: floor `wpud`/`wipud` at zero. Verified on the exact failing day —
GPU was NaN and CPU 324.7 K; both now return 278.6180 K, agreeing to four
decimals.

### 2.6 Three dead configuration flags

Declared in `RunConfig`, read nowhere:

* `fire_scheme` — the `dynaveg` call omitted the argument, so **every run took
  the legacy scheme regardless of the config**;
* `enable_fire_emissions` — the whole trace-gas module was unreachable;
* (and `firefac` was computed by nothing before that, so the legacy fire scheme
  could not have fired anyway).

**Fixed**, all wired.

### 2.7 Soil solid-phase properties were computed, stored, and never read

`io/soil_init.py` computes the specific heat and density from F77
`initial.f:715-724` (Campbell & Norman 1998), and `run_global.py` stored `csoi`
into the state. The physics read neither, substituting flat constants that drop
the organic correction; `SoilState` had no `rhosoi` field at all. Three
separate copies of the constants existed, two of them wrong. **Fixed**, all
derived from one `FORGANIC`. Effect: +0.69 % on solid-phase heat capacity.

### 2.8 The PRNG key was never checkpointed

The stochastic weather generator draws from it, so **no checkpoint was
reproducibly resumable** — restarting gave a cell entirely different weather.
This is why the NaN went undiagnosed for three rounds. **Fixed**: checkpoints
carry the key, which is what made §2.5 bisectable.

### 2.9 Two masked-NaN divisions break `jax.grad`

`interception.mix` (`jnp.sign(0.0) == 0.0` makes the denominator exactly zero
when canopy masses cancel — routine for a dry canopy) and
`canopy_exchange.impexp2` (divides by the quantity its own condition tests).
Forward values were unaffected, but **gradients through these functions were
NaN**, which matters for any calibration work. **Fixed**; gradients now finite.

---

## 3. What is *not* wrong

Two things that look alarming and are correct, recorded so nobody re-opens them:

* **`wsoi == 0` in 37 % of cells** is frozen ground, not aridity. Antarctica is
  98.4 % of the −90…−60 band and Arctic permafrost 72.4 % of +60…+90; between
  them it is 0–5 %. 89.4 % of dry cells are ice-rich and **not one is
  ice-free**. The Sahara reads `wsoi` 0.085–0.161.
* **Soil above 310 K** in ~3 500 cells in July: 97.1 % lie between 15–40 °N,
  none poleward of 50°, hottest in the Rub' al Khali at 45.4 °C. Realistic.

---

## 4. Still open

* **WAD2M global wetland area** through our reader is 3.06 Mkm² (Jan 2010) /
  4.64 (Jul). Zhang et al. (2021) quote larger figures. Whether that is a
  definitional difference (inundated-only vs including non-inundated peatland)
  or an error on our side is **not established**, and methane flux scales
  linearly with it.
* **F6 fire calibration produced a null result** — the a-priori parameters beat
  the fitted ones on global rank and total burned area. Shipped defaults are a
  priori. Consistent with FireMIP (Hantson 2020): structure dominates
  parameters.
* Methane M2–M7 not started.

---

## 5. Replacement spin-up

`spinup_fire_v8` — 200 years cycling 1901–2000, fire on, seasonlength + VPD +
combustion completeness + per-PFT mortality. 200/200 steps, **0 NaN**,
converged (biomass 3.269 ± 0.011 over the last 50 years).

Against GFED5 `Norm` 2001–2005, 88 484 cells:

```
area-weighted burned fraction   model 4.232%   GFED5 4.615%   x0.92
rank correlation                all +0.749     burning +0.410
NMSE                            0.723
mean |lat| of top-2000 cells    model 11.9     GFED5 10.7
area-weighted live biomass      4.364 kgC/m2   (obs. range ~3.5-5)
```

Its `state_latest.pkl` carries the PRNG key, so it is reproducibly resumable.

---

## 6. Where the code is

Branch **`fire-test`** (35 commits). `main` is untouched at `605800c`. Each
defect above has its own commit with the evidence in the message. Test suite:
614 passing.


---

## Annual accumulators never reset, and dead productivity diagnostics

**Found 2026-08-30, while auditing Rh for the methane tropical excess
(METHANE.md 5p). Both affect reported output, not the simulation.**

### 1. Seven annual accumulators accumulated over the whole run

`SummaryState`'s `ay*` fields are annual totals -- `summary.py` and
`io/diagnostics.py` both label them `/yr`. The year-end block in
`loops_gswp3.py` cleared only the three `dynaveg` consumes, its own comment
saying "the annual accumulators the update just consumed". The rest were never
cleared. Measured in the 200-year `spinup_bedrock_v9_3lay` checkpoint:

| field | value after 200 yr | plausible for one year |
|---|---|---|
| `ayprcp` | 3.22e6 mm | ~3 000 mm |
| `ayaet` | 8.97e5 mm | ~1 000 mm |
| `aytrunof` | 1.98e7 mm | ~300 mm |
| `ayco2mic` | 921.9 kgC/m2 | ~1-5 kgC/m2 |

`ayco2root`, `aytdrain` and `ayneetot` were in the same state.

**Impact is confined to netCDF diagnostics.** No physics reads these:
`dynaveg` takes only `aynpp`, `ayanpptot` and `ayneetot`, and the daily
diagnostics difference against a value saved at the start of the day. Any
analysis that read `ayco2mic`, `ayprcp`, `ayaet` or `aytrunof` out of the
netCDF output as a per-year quantity was reading a run-cumulative total.

**Fixed:** the year-end block now clears every accumulated `ay*` field.
`ayanpp` is deliberately excluded -- `dynaveg` derives it from zeros and
overwrites it wholesale each year, so clearing it afterwards would destroy the
value just written. `tests/test_spinup_dynaveg_wiring.py::
test_all_annual_accumulators_are_reset_each_year` guards the whole set, so a
new accumulator cannot quietly join the unreset ones.

**Still open:** `ayanpptot` is accumulated during the year, then overwritten by
`dynaveg`, then cleared by the year-end block -- so it reads 0 in every
checkpoint. That predates this fix and is left alone; the three-way write wants
deciding rather than patching.

### 2. `tgpp`, `tnpp`, `tgpptot`, `tnpptot`, `tco2root` are dead

All exactly zero in the 200-year checkpoint, and **never assigned anywhere in
production code** -- `tgpp`, `tnpp`, `tgpptot` and `tnpptot` have no references
at all outside the `VegetationState` definition; `tco2root` is read twice and
written never. Same class as the dead `firefac` and the three dead config flags
in FIRE.md, but these are state fields, which `tests/test_no_dead_code.py` does
not cover -- it checks modules and config flags.

**Consequence:** the model has no usable GPP or NPP diagnostic. Rh could not be
cross-checked against the model's own productivity when auditing it, which is
the check that would have said whether an Rh of 80.2 PgC/yr reflects a
decomposition problem or a productivity one. Not fixed here: wiring them means
deciding where GPP/NPP are summed, which is a change to the physics path rather
than to reporting.


---

## The global grid duplicated 75% of its cells

**Found 2026-08-31. Affects the resolution and cost of every global run; does
NOT affect the global totals already reported. Fixed.**

`build_global_cell_grid` sampled latitudes and longitudes on cell BOUNDARIES --
`np.arange(-85.0, 85.0, 0.5)`, i.e. X.0 and X.5 -- while `latlon_to_grid_idx`
rounds against a grid whose centres are at X.25 / X.75:

    j = int(round((89.75 - lat) / 0.5))

Every sample therefore evaluated `round()` on an exact .5, and Python rounds
half to EVEN, collapsing adjacent samples in pairs:

    lat -85.0 -> 349.50 -> 350        lat -84.0 -> 347.50 -> 348
    lat -84.5 -> 348.50 -> 348        lat -83.5 -> 346.50 -> 346

In both axes, so four rows per cell.

| | before | after |
|---|---|---|
| rows | 88 484 | 95 333 |
| distinct cells simulated | **22 301** | **95 333** |
| duplicate rows | 66 183 (74.8%) | 0 |
| area integrated | 164.0 Mkm2 | 165.0 Mkm2 |

### What it cost

* **4x the computation for no extra information.** 21 941 cells were simulated
  four times over.
* **Effective resolution was 1 deg** while every run, document and file name
  said 0.5 deg.
* **77% of the mask's land cells were never simulated** -- their climate was
  represented by a neighbour.
* **The four copies were not even consistent.** Each row carries its own
  sampled latitude into the solar zenith calculation while sharing GSWP3
  forcing, so copies of a single cell drifted apart by 1 K on average and up to
  **21.6 K** over the 200-year spin-up (`tsoi[:,0]`; `ayco2mic` spread up to
  266.9, `totcsoi` up to 122.1).

### What it did NOT cost

Checked rather than assumed, because the natural fear is that every global
number is 4x too large:

* **Global totals are not inflated.** Each duplicate carried its own 0.5 deg
  cell area, and four of those sum to the area of the four cells they stood
  for: 164.0 Mkm2 integrated against 165.0 Mkm2 of land in the mask. Rh,
  CH4, burned area and permafrost extent were computed over the right total
  area -- at 1 deg resolution.
* **Geography is intact.** corr(|lat|, deep soil T) = **-0.907**, with 300.9 K
  in the tropics and 247.4 K poleward of 75 deg, so the row-to-cell
  correspondence never broke and every latitude-band result stands.

### Fixed

Rows now sample cell centres, giving exactly one row per land cell, with an
explicit `seen` set so a coarser `lat_step` stays an honest subsample.
`tests/test_grid_no_duplicates.py` guards it at 0.5, 1 and 2 deg and checks
that the 0.5 deg grid has exactly one row per land cell.

**Cost of the fix: +8% rows for 4.3x the distinct cells.**

### Consequence: existing checkpoints cannot be loaded

`npoi` changes 88 484 -> 95 333, so every checkpoint in `verify_outputs/`
(including `spinup_fire_v8` and both `spinup_bedrock_v9` arms) is on the old
grid. `spinup_*.py` asserts `len(cells) == npoi` and will now fail loudly on
them, which is the correct behaviour. Re-running requires either a cold start
or remapping each new cell onto the old duplicate that covered it.

### Separately: the domain is larger than Earth's land

`surta.nc`'s `landfrac` is binary and its area-weighted sum is **165.0 Mkm2**,
against Earth's ~149 Mkm2 of land (~130 ice-free). Our global totals are over
that 165 Mkm2 domain. This is independent of the duplication bug and is not
fixed here -- it needs the mask itself examined -- but it means any per-area
comparison against an observational product must account for an 11% larger
domain. It directly weakens METHANE.md 5p: correcting Rh for the domain gives
72 rather than 80.2 PgC/yr against a published 49-58.


---

## The carbon cycle is inflated, and the cause is GPP

**Found 2026-08-31 from the cold-start spin-up (`configs/spinup_cold_v10.yaml`),
300 years on the corrected grid. This is a physics finding, not a reporting one.**

The run converged cleanly -- soil carbon drift +0.017 %/yr over the last 50
years, **0 NaN across all 300 years** -- so the numbers below are equilibrium
values, not a transient.

| flux | IBIS | published | total | per-area |
|---|---|---|---|---|
| **GPP** | **219.7 PgC/yr** | 120-130 | **1.76x** | **1.39x** |
| Rh | 89.6 PgC/yr | 49-58 | 1.67x | 1.33x |

Two separate factors, and they should not be conflated:

* **per-area**: GPP is 1.39x too high per square metre. This is a physics
  problem in photosynthesis.
* **domain**: our land mask is 164 Mkm2 against ~130 Mkm2 of ice-free land,
  a further 1.26x on any total.

**Rh follows GPP.** At equilibrium Rh is set by carbon input, so a 1.39x
photosynthesis gives a 1.33x decomposition -- the two ratios agree. METHANE.md
5p framed this as "Rh is ~1.5x too high" and pointed at decomposition; that was
looking at the symptom. **soilbgc is faithfully respiring what it is handed.**

This also explains the methane magnitude. CH4 scales linearly with Rh in
M-lite+, we are 1.6x high on CH4, and Rh is 1.67x high. **No methane parameter
needs adjusting for magnitude** -- and fitting one would have buried a
photosynthesis error inside `r_ch4`.

### Fixing GPP is now the highest-value work in the model

It propagates to NPP, biomass, litterfall, soil carbon, Rh, fire fuel load and
methane. Every one of those was tuned or evaluated against observations while
GPP was 1.4x high per unit area.

### Two diagnostics that obstructed this, one fixed

**Both fixed 2026-08-31, following F77 rather than inventing a convention.**

**`ayneetot` was structurally wrong, twice over.** Only the `-NPP` half was
ever accumulated (`loops_gswp3.py`: `sm.ayneetot - npp_step`); nothing added
Rh, so it reported **-106 PgC/yr** for a converged spin-up. And that sign was
inverted against `dynaveg`'s `ayneetot - cdisturb` (copied faithfully from
`vegetation.f:481`), which made **biomass burning increase the apparent sink**
instead of reducing it.

F77 forms it once at year end (`stats.f:1200`):

    ayneetot(i) = aynpptot(i) - ayco2mic(i)

POSITIVE = sink. The port now does the same, in both `run_spinup_gswp3` and
`run_global`'s own year loop, which makes `- cdisturb` correct.

**`aynpptot` and `ayanpptot` are now separate fields, as in F77.**
`comsum.h:117-118` declares them distinctly -- "annual above-ground npp" and
"annual total npp". The port had the timestepper accumulating *total* NPP into
`ayanpptot` while `dynaveg` overwrote the same field with the *above-ground*
sum, so whichever wrote last won. The netCDF is written after `dynaveg`, so the
published field was silently ANPP. Now the timestepper owns `aynpptot` (total)
and `dynaveg` owns `ayanpptot` (above-ground); both are published, and
`run_global`'s monthly NPP/NEE diagnostics read the right one.

Verified on a 3-year run:

| ratio | measured | expected |
|---|---|---|
| ANPP / NPP | **0.654** | 0.5-0.7 |
| NPP / GPP | **0.464** | 0.45-0.50 |
| NEE sign | **+46.2 PgC/yr** (sink) | positive for a young cold start |

NEE also reconciles: NPP - Rh = 52.3, less ~6 PgC/yr of fire, against the
reported 46.2.

### A number this corrects

The entry above reported "NPP 75.8 PgC/yr, 1.32x published". **That was ANPP**,
read from the colliding field, and is withdrawn. `NPP/GPP` correspondingly read
0.345 rather than the correct 0.464 -- the model's internal partitioning was
never wrong, only the field it was read from.

Equilibrium *total* NPP is not recoverable from the v10 run, whose output
predates the `aynpptot` field. **The GPP and Rh findings are unaffected**:
`aygpptot` and `ayco2mic` were always correct, so GPP 219.7 PgC/yr (1.76x) and
Rh 89.6 PgC/yr (1.67x) stand, and GPP remains the root cause.
