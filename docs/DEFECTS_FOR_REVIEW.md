# Defects found in IBIS-JAX, 2026-08-12 → 2026-08-27

> Documentation edition, 2 November 2026: dated release results remain historical. Descriptions of later corrections refer to the reviewed development implementation; this publication does not update model code on main. See [the edition note](technote/EARLY_EDITION.md).


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

**Note (2026-11-01, technote review D9):** `annual_climate_stats` took the year's statistics from monthly means, not
from the daily 10-day running mean and the daily degree-days of IBIS 2.6b4. The size of that difference, and the daily
rule restored, are in the entry D9 of the last section.

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

**Correction (2026-11-01, technote review D73):** the fix was made in the checkpoints of the August spin-up runner
(§5); the restart files of the release driver do not carry the key. `run_global.py:1080` and `run_ibis.py:220` save
the state with the year as the only metadata, and `run_global.py:620` creates `PRNGKey(42)` after a restart is loaded
(`:400-404`). The release restart of 2010 (`final_rel30/hindcast/restart/ibis_restart_2010.npz`) holds 272 state
arrays, `treedef_repr`, `num_leaves` and `meta_year`, and no key (listed 2026-11-01). So every stage of the release
protocol starts its weather at key 42, a run made in two parts draws other weather than the same run made in one, and
an interrupted run cannot be resumed with its own weather (technote 17.6). IBIS 2.6b4 also reseeds at the start of
each run. Storing the key in the restart would change the weather of every restarted run; it is left for a later
version.

### 2.9 Two masked-NaN divisions break `jax.grad`

`interception.mix` (`jnp.sign(0.0) == 0.0` makes the denominator exactly zero
when canopy masses cancel — routine for a dry canopy) and
`canopy_exchange.impexp2` (divides by the quantity its own condition tests).
Forward values were unaffected, but **gradients through these functions were
NaN**, which matters for any calibration work. **Fixed**; gradients now finite.

**Note (2026-11-01, technote review D64):** finite for these two routines only. The gradients of the whole time step
are not numbers: `tests/test_gradients.py` marks the single-step gradients with respect to air temperature, solar
radiation, soil moisture and leaf area, and the 3-step gradient, as expected failures, and with `--runxfail` all five
fail with all-NaN gradients (technote 17.11). No calibration used gradients.

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
  linearly with it. **Update 2026-10-30:** a registration error in our reader (3 % low and smeared) was found
  and fixed (CALIBRATION 81, below); whether the remaining gap to Zhang et al. is definitional is still open.
* **F6 fire calibration produced a null result** — the a-priori parameters beat
  the fitted ones on global rank and total burned area. Shipped defaults are a
  priori. Consistent with FireMIP (Hantson 2020): structure dominates
  parameters. **Superseded:** the fire parameters were refitted on a zone objective
  (CALIBRATION 65-66); the release sits at its optimum.
* Methane: built, calibrated and in the release (METHANE.md 1); only M4 (microbial dormancy) is not built.

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
614 passing. (The August state. The IBIS 3.0 release is on `main`, where the full suite passes; CALIBRATION 85.)


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
| **NPP (total)** | **98.7 PgC/yr** | 55-60 | **1.72x** | **1.36x** |
| Rh | 89.6 PgC/yr | 49-58 | 1.67x | 1.33x |

**The whole cycle is scaled up by a single, nearly uniform factor** -- 1.76x,
1.72x, 1.67x -- while the ratios BETWEEN the fluxes are right:

| ratio | IBIS | expected |
|---|---|---|
| NPP / GPP | **0.450** | 0.45-0.50 |
| Rh / NPP | **0.907** | <1 at equilibrium, remainder to fire |
| NEE | **+1.6 PgC/yr** | ~0 at equilibrium |

That is the signature of ONE upstream error, not several independent ones. The
carbon allocation, the autotrophic respiration fraction and the decomposition
are all doing the right thing with too much carbon.

(`ANPP/NPP = 0.768` sits above the 0.5-0.7 usually quoted, but that range is
strongly biome-dependent and is a much weaker constraint than the others; it is
noted, not treated as a second finding.)

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

v10 was re-run with the corrected diagnostics (2026-08-31 12:17-16:35, 300
years, 0 NaN, soil carbon drift +0.017 %/yr). Equilibrium total NPP is
**98.7 PgC/yr, 1.72x published**, and NEE now reads **+1.6 PgC/yr** against the
-106 the broken accumulator reported -- the check that the fix works. The
carbon state reproduced the archived run almost exactly, as it must, since the
fixes were to diagnostics and not to physics.

## One NaN cell in every v11 state (open; absent from the release run)

The release run's 2000 and 2010 restarts hold no non-finite value (`final_rel30`, every float array, 2026-10-30).

**Found 2026-09-11, diagnosed 2026-09-16.** Cell index 8028, **68.75N
-26.75E** -- the Greenland ice-sheet margin. Present in `v11_spinup` (year
1900), `v11_transient` (2010) and every state descended from them; **absent
from `both_fix`** at the same cell.

**What is NaN.** 133 of 258 leaves at that cell and no other. The first
ones are soil: `tsoi`, `wsoi`, `wisoi`, `tg`, `wpud`, `wipud` -- temperature,
liquid water, ice, ground temperature and both puddle stores. Soil texture
is ordinary and identical to the west neighbour (sand 40 / clay 20). So this
is the freeze-thaw / puddle hydrology path failing at an ice-margin cell
under default parameters, not a vegetation or texture problem. Same class as
2.5 above.

**Why both_fix escaped it.** Not established. `both_fix` ran `vmax x0.54`
and 350 ppm from cold; v11 ran defaults and 296 ppm. Higher productivity at
the margin plausibly pushes the hydrology into the failing branch, but that
is a hypothesis.

**Consequence.** It has cost four reruns: `state_report.py`,
`evaluate_gfed5.py` and `evaluate_global.py` each aggregated without masking
and returned NaN for a whole domain on the strength of one cell. All three
now drop non-finite cells and say so. IBIS has no lateral coupling, so it
cannot spread.

**Not fixed at the source.** A one-cell NaN in 95,333 is tolerable for the
calibration; it is not tolerable in a released model. Localising the first
non-finite operation (run the cell alone from the 1900 restart with
`jax.debug` checks) is the next step when someone picks this up.

## The physics ran at sea-level pressure on every cell (found 2026-09-20 as "negative ET", cause found and FIXED 2026-09-21)

**Symptom.** Negative monthly total evapotranspiration (`amaet`) for whole
seasons. In the joint-fit closing hindcast (`final_v12g/hindcast`, 120
months of 2001-2010) 33,169 of the 95,333 cells have at least one negative
month and 18.3 % of all cell-months are negative; 21,792 cells go below
-3 mm/month. The worst are the Himalayan monsoon slopes (27-33 N,
77-94 E): -175 to -205 mm in July-August and a **negative annual mean**
(-50 to -93 mm/month averaged over 120 months, negative in 87-120 of the
120 months), with a full forest canopy (`fu` 0.98), wet soil, and a
ground temperature of 17-20 C. The first report (2026-09-20, on the (b3)
sample) had seen -80 to -146 mm/month near 29-35 N and guessed at the
ground-humidity factor, the puddle, or saturated forcing humidity.

**Cause.** The GSWP3 driver reads surface pressure (`ps`) into
`DailyWeather.psurf` and `diurnal` uses it to cap the hourly humidity at
0.99 of saturation, exactly as F77 does. The hourly atmosphere update in
`loops_gswp3._step_body` then wrote `ta, qa, ua, coszen, solad, solai,
fira, raina, snowa` into `atm` -- and **not `psurf`**. `atm.psurf`
therefore kept its creation value, 101325 Pa, on every cell for the
whole run; every saved state carries it (`leaf_110` of every restart is
101325 everywhere). The canopy and soil physics (`turvap`, `stomata`,
`twet3`, the fire VPD) compute saturation at that pressure. At the worst
cell (31.25 N, 78.75 E, GSWP3 `ps` 583 hPa) the August air is 13.7 C with
13.5 g/kg of water: 78 % relative humidity at 583 hPa, **136 % at
1013 hPa**, dew point 10.3 C at the true pressure and 18.9 C at sea
level. The surface condensed water and warmed until its own saturation
matched the air's humidity -- the model's monthly ground temperature
there, 19.6 C, is the sea-level dew point. The condensation heat (about
190 W/m2 for -200 mm/month) was carried off as sensible heat to the
colder air. This is not a saturation-cap or puddle question: the
humidity the physics was handed was built at one pressure and judged at
another.

**Extent beyond the negative cells.** Saturation specific humidity scales
as 1/p, so every cell above sea level ran with saturation too low by the
factor p_true/1013: 3 % at 300 m, 11 % at 1000 m, 17 % at 1500 m, 40 %
at 4000 m. The leaf saw a relative humidity too high by the same factor
(Ball-Berry conductance is proportional to it), the VPD too low, the
air density (p/RT) too high by the same factor in every aerodynamic
conductance. Plateaus and mountain ranges -- Tibet, Mongolia, the Great
Basin and Rockies, the Andes, the East African and southern African
plateaus, Anatolia and Iran, the Brazilian shield, Greenland and
Antarctica -- ran with wider stomata and stronger turbulent exchange
than the forcing describes. Every archived run, fit and validation since
the GSWP3 driver was written carries this, and the fitted stomatal and
stress parameters partly compensate for it. Sea-level cells (the
majority of the tropics and the boreal plain) are affected only by the
weather's own +-2 % pressure variation, which is now also represented.

**Fix (2026-09-21).** `loops_gswp3._step_body` writes `dw.psurf`
(broadcast to the cell vector) into `atm.psurf` alongside the other
hourly forcing. The weather-generator driver (`loops.run_day`, used by
`run_ibis.py` and the benchmarks) had the same omission -- F77's
`daily` computes a topography-derived pressure into `comatm` and the
port's `weather.daily` computes the same value into
`DailyWeather.psurf`, which its hourly update then dropped -- and gets
the same one-line fix. The creation value 101325 Pa remains only as the
initial value before the first hourly step. Guard:
`test_hourly_atmosphere_carries_the_forcing_surface_pressure`. Probe
(`verify/calibration/diagnostics/psurf_probe.py`): the six worst cells
plus plateau and sea-level controls run two years on the joint-fit
parameters -- results in CALIBRATION.md 16. Stage (d), the refit on the
corrected phenology clock, was held back until this fix was in and is
the first fit on the corrected pressure; the closing protocol re-spins
from cold, so no saved state needs migrating (the pressure is
overwritten on the first hourly step). The calibration's ET guard
(`fit_carbon.ET_MIN_RATIO`, which excluded cells with model ET below a
fifth of the observation) stays as a safety net and should now exclude
few cells.

## Ten-day running means updated with the per-timestep weight (found 2026-09-20, FIXED)

`loops_gswp3._day_body` updates `a10td` and the four `a10anc*` running means
once per day but used F77's per-timestep weight `exp(-1/(10*niter_day))`
= exp(-1/240) for hourly steps. F77 (`stats.f` sumday) applies that weight
at every timestep, 24 times a day, so one day's decay is exp(-1/10). The
port's memory was therefore 240 days, not 10: the annual temperature cycle
in `a10td` was attenuated to 24 % of its amplitude and lagged 77 days
(January `a10td` +7 C on steppe cells whose January air temperature is
-3 C), so `pheno`'s cold triggers for the lower canopy rarely fired,
grasses and deciduous shrubs kept 3-4 m2/m2 of displayed leaf area through
a -7 C winter (CALIBRATION.md 16, lever test 15), and the `a10anc*`
drought triggers averaged over eight months. Fixed to the daily-equivalent
weight exp(-1/10) (guarded). `canopy_scaling`'s `a10scalparam` runs per
timestep with the per-timestep weight and was correct. Consequence: every
calibration stage through v12g ran on a model with a 240-day phenology
clock; the seasonal criterion (mid-latitude amplitude 0.6, tropical phase
lag) and any fitted phenology scalar (`tthresh_*`, `gdd_*`, `drop_*`) are
re-read on the corrected model. Old-code runs are not reproducible with
the fix in; the checkpoints are unaffected (no state field changed).


## The weather generator received latitude in radians and read it as degrees (found 2026-09-27)

**Symptom.** At Harvard Forest (42.5 N) the generator's shortwave integrated to 193 W m-2 in January against the
71 W m-2 of the daily forcing, 197 against 65 in December and 218 against 255 in July; its daylight ran 06-18 h in
every month. At 60 N a June day had 13 h of light instead of 19, a December day 11 h instead of 5, and the noon
sun stood at cos z = 0.92 the whole year.

**Cause.** `SimParams.latitude` holds radians (`build_all_params(npoi, lat_rad, ...)`; `daylength_hours` reads it as
radians). `weather.diurnal` converts from degrees, as F77 does (`xlat = latscale(jj) * pi / 180.0`). Both day loops
(`loops_gswp3._run_day_inner`, `loops._run_day`) passed `params.latitude` straight through, so every cell's solar
geometry was that of latitude |lat| = (lat in radians) degrees, i.e. within 1.6 degrees of the equator. The cloud
fraction that the daily forcing is inverted into (`rsds_to_cloud`, radians, correct) then met a diurnal cycle of the
wrong shape and length, and the daily shortwave the model received was not the forcing's.

**Exposure.** Every run of the JAX port through the day loops: all GSWP3 global runs (spin-ups, transients,
validations, the calibration stages), all site runs at the towers, and the tests that inspect them. The unit tests of
`diurnal` pass degrees and were green. The equator's insolation is far too strong in the mid- and high-latitude
winter and too weak in their summer; the "mid-latitude winter GPP 4-5x FLUXCOM, June-July 0.6x" residual
(CALIBRATION.md 16, 20) and the towers' cold-season GPP share of 0.12-0.16 against 0.01-0.06 (section 20, "the one
robust class-wide error") are candidates for this cause and must be re-read on the corrected model.

**Fix.** `loops_gswp3.latitude_degrees(params)` at the generator call, the same conversion in `loops.py`;
`tests/test_solar_latitude.py` guards the convention at both call sites and the day lengths. The correction changes
every result; no earlier run is reproducible against it.

## Runoff and drainage diagnostics (found 2026-10-16, fixed)

The surface-runoff diagnostic `HydrologyDiag.grunof` added the squeeze-out of a saturated column (`fsqueez`) although
`soilh2o` already returns that water to the puddle; F77 `soil.f` keeps it in the puddle (line 845) and never adds it to
`grunof` (lines 256-280). The same water was therefore counted as runoff every timestep it re-infiltrated. The bottom
drainage `gdrain` was hard-wired to zero, while F77 reports `wflo(nsoilay+1)` (line 387). Diagnostics only -- no physics
reads either field -- but every runoff and drainage output before this date is wrong: in the 2001-2010 hindcast of
IBIS 3.0 the wetland cells reported 2,182 mm/yr of runoff on 1,278 mm/yr of rain and 713 of evapotranspiration (8x
P - ET in the Amazon), all land 60S-60N 1,309 mm/yr of runoff on 969 of rain, and zero drainage everywhere. Found by the
WAD2M regression (`logs/wad2m_storage_regression_2026-10-16.md`). Fixed in `loops.py`; `tests/test_water_balance.py`
closes the soil-plus-puddle water balance of a timestep with the stored fluxes (it failed by 0.55 and 6.1 mm per step
before the fix).

## The tower harness ignored `--f-inund` outside the tower-water-table mode (found 2026-10-29, fixed)

`verify/calibration/site_fit.py`: the loop applying the explicit `--f-inund SITE=value` overrides, and the log line of the
inundated fraction, sat inside the `--wtd-mode tower` branch since commit dd7af6d (2026-10-11, section 42), so in the
default mode the override was silently ignored. Impact: none on any result. Every override on a non-wetland tower ran in
the 10 October methane batches (r3-r5), before the defect; every later command passed it only for wetland towers, where
the harness sets the fraction to 1 regardless (checked over all command files, 2026-10-29). Fixed by moving both
statements out of the branch, guarded by `FIN is not None`.

## WAD2M (and GFED5) registration: half-to-even rounding put 75 % of land points on a neighbouring cell (found 2026-10-29, fixed)

The runner mapped each land point to the WAD2M grid as `np.rint((lat - (-90)) / 0.5)` (and the same for longitude), on the
corner grid the readers check against. A half-degree cell centre (x.25 / x.75) is exactly half-way between two corner
indices, and `np.rint` rounds half to even, so every other row and every other column landed one cell off: 49.8 % of rows,
50.0 % of columns, 74.9 % of land points read a neighbouring WAD2M cell. Introduced 2026-10-09 with the fix of the
latitude mirror ("index the climatology from the land points' own coordinates, as evaluate_global.py does"), which copied
the rounding from the evaluation scripts. Effect on the prescribed wetland area (annual mean, true land): 90S-30N 1.993
instead of 2.071 Mkm2 (-3.8 %), 30-60N 0.944 instead of 0.959 (-1.5 %), 60-90N 0.377 instead of 0.390 (-3.5 %), global
3.313 instead of 3.420 (-3.1 %); cell correlation of the read map with the correct one 0.71 in the tropics, 0.77 globally.
So every global wetland CH4 number from 2026-10-09 to 2026-10-29 (the 28 October candidate's 170 Tg, CALIBRATION 73, the hourly runs; the release's 152 is on the corrected reader)
carries a ~3 % low and spatially smeared wetland area. The tower fits are not affected (wetland towers run at an inundated
fraction of 1). The same expression sat in the GFED5 fire evaluator (`verify/fire/evaluate_gfed5.py`, `_grid`), so the
observed burned-area map of every fire score (protocol logs, `glob_refs.npz` burn_gfed5, the fire-objective work) was read
from a neighbouring cell for 75 % of land points; and in `site_fit.py` (WAD2M at non-wetland towers), the methane band
diagnostic and four older methane scripts. Fix: `ibis.io.wad2m.wad2m_cell_index` (floor, the cell that contains the point)
in the runner, floor in the scripts; `tests/test_wad2m_registration.py` (every centre maps to its own cell, any point to
the cell that contains it, the file's centre equals the point). Methane hindcasts rerun; fire re-scored (CALIBRATION 81).

## Texture classes on the class borders differed from IBIS 2.6b4 (found 2026-10-30, fixed; the release run not rerun)

`ibis/io/soil_init.py` gave a soil layer the FIRST class polygon whose interior held its (sand, clay) point, by plain ray
casting, after rounding the percentages half to even. F77 `textcls` and `inpoly` (`initial.f` 771-1074) test every polygon
with the border points inside and keep the LAST match, after `nint` rounding. Against the F77 functions compiled as they
are, the port gave another class to 315 of the 5,151 whole-percent points of the triangle (93 on interior class borders;
99 on the no-silt edge, which the port's ray casting left outside every class and gave silt loam; 123 with no sand or no
clay, which F77 gives silt loam), and to 13,147 of the 571,998 soil layers of the land cells of the HWSD2 soil file (2.3 %): 9,135 cells (9.6 %) in at least one
layer, 2,483 in the top layer, all of them interior-border cases (the rounding rule alone accounts for 9). The moves
(F77 class to port class, layers): loam to sandy loam at 52 % sand 6,939, clay
loam to sandy clay loam at 45 % sand 3,117, silt loam to loam 1,911, loamy sand to sand 450, sandy loam to loamy sand
372, clay to sandy clay 300, four smaller ones 58. At the two largest borders the F77 class is the USDA one (sandy loam
and sandy clay loam need more than 52 and 45 % sand). In a changed layer the porosity moved by up to 0.066 and the field
capacity by -0.015 to +0.082 m3 m-3. Found while drawing the texture figure of the technical note (Figure 2.4).

Fixed: `_inpoly` is F77 `inpoly`, `_build_texture_lookup` keeps the last match, `texture_class` rounds as `nint`; the
saturated soil albedo now rounds the top layer's sand and clay as `nint` too (`initial.f:646`; it rounded half to even,
1 point of sand or clay in 401 top layers, at most 0.00035 of visible and 0.0007 of near-infrared albedo). Two departures from F77, kept, neither touching a
land layer of the soil file (rounded sand 12-91 %, clay 3-70 %, sum at most 97 %): F77 gives silt loam to every
point with no sand or no clay (`if (msand .gt. 0.0 .and. mclay .gt. 0.0)`), the port classifies them like any other point;
a pair that rounds above 100 % keeps its sand and loses the excess clay (F77: silt loam). `tests/test_soil_texture_classes.py` compiles the F77 functions and compares all 5,151 points.

**The release run (`final_rel30`) and every calibration used the earlier classes.** The rerun with the corrected classes is
deferred (Hangkai 2026-10-30: revise and document, "do not rerun right now"). The learned wetland area reads the model's
porosity (its root-zone water feature) and was trained on a hindcast with the earlier classes; whether to retrain it is a
question for that rerun.

## The technote review (2026-11-01): defects and unrecorded departures

Found while writing the technical note of IBIS 3.0 (`docs/technote/`) and reviewed chapter by chapter (CALIBRATION 87);
every finding below was decided by Hangkai on 2026-11-01. Evidence and numbers are those of the review (`merged.json` in
`technote_review_2026-11-01/` on the lab NAS, which also gives the options; technote sections as cited); line numbers are
those of the release code `e54e342`. "Inherited": IBIS 2.6b4 does the same. "Fixed on technote-review-fixes" names the
commit of the implementation pass (all merged into that branch on 2026-11-01, head c8b4d21; CALIBRATION 87,
Implementation); the tower analysis, the refits and a final protocol run follow. No fix below has been through a tower or
protocol run yet: the release run `final_rel30` has none of them. The texture-class entry above came from the same review.

### D1. Growth respiration charged on negative net assimilation (departure; fixed, refit to follow)

`loops_gswp3.py:211, 217, 223, 228, 233` computed NPP = (1 - rgrowth)(A - R) whatever the sign of A - R, while the
comment at `:177` cites `stats.f`, which applies (1 - rgrowth) only where tnpp > 0 (`stats.f:253`, "growth respiration
is only applied to positive carbon gains"). In the port since the initial commit 4ca4481, with no switch and no record.
A repeat of 2010 from the release's 2009 restart (903 cells in three 1-degree strips) charges 0.063 kg C m-2 yr-1 on
deficits, 9.4 % of NPP and 3.9 % of GPP: NPP is about 9 % higher than under the IBIS 2.6b4 rule (10-14 % at 10S-10N,
3-5 % north of 55N) in every run, and the NEE-stage fits of rgrowth and rroot (CALIBRATION 36) partly absorbed it.
Technote 11.2.3, 11.12.3. **Decision (b): the IBIS 2.6b4 rule, rgrowth and rroot refitted. Fixed on
technote-review-fixes, commit 42a885c** (`npp_after_growth_respiration`, `tests/test_growth_respiration.py`); the switch
`bgc.rgrowth_positive_only` (default true; false gives the release's rule) came with 307d0f8. The refit follows.

### D2. The decomposition factor averages the soil over 0-1 m; IBIS 2.6b4 over 0-0.25 m (unrecorded departure; kept)

`loops_gswp3.py:539-550` averages tsoi, wsoi and wisoi over layers 1-4 by thickness (the comment at `:497` says
"root-profile weighted"); `stats.f:444-456` averages layers 1-2 for `decomps` (`:595-624`). Since 4ca4481 and in no
record; `verify_soilbgc.py:111-112` passes decompl and decomps as random inputs, so the factor itself was never compared
with the Fortran. First-order equilibrium estimate from the 120 monthly states of 2001-2010: the layers-1-2 factor is
1.126 times the layers-1-4 factor (cells 5-95 %: 0.75-2.25), which would put the slow pools at x0.876 globally (2072 to
about 1815 PgC; boreal x0.73, savanna and hot dry x1.12). The zone factors `som_k_scale` (CALIBRATION 72) were set on the
0-1 m average. Technote 12.2.3, 12.9 item 4. **Decision (a): kept as an IBIS 3.0 difference, documented;** the comment
and the verify test are in commit 8416c20 (`decomposition_factor`, tested against the litter and soil blocks of `stats.f`
compiled verbatim: agreement 2.3e-6, the port's float32 constants).

### D3. Day and night NEE targets are partial-month sums, the model's are full-month sums (tower harness; fixed)

`tower_nee_daynight.py:33, 36` sums the half-hours with `NEE_VUT_REF_QC` <= 1, and `site_fit.py:56-62` fits those sums,
while the model sums every step of the month (`loops_gswp3.py:276-281`). In `tower_nee_daynight_t31.npz` (478 towers,
24,447 tower-months) the kept half-hours cover a median 0.960 of the month, less than 0.9 in 28.1 % and less than 0.75 in
13.9 % of the tower-months. In those months the night (respiration) and day targets are too small in magnitude, which
biases the NEE-stage fits of rgrowth, rroot and E0 at 358 towers by an unmeasured amount; the likeliest common cause of
the pile-up of rgrowth at its 0.10 floor (D4). Technote 16.3, 16.14 item 3. **Decision (a): night and day means per kept
half-hour against the model's night and day means, all tower-months kept. Fixed on technote-review-fixes, commit
63fd7c0:** the targets are the mean NEE of the kept half-hours of each month's night and day, with their counts; the
model's means come from step counters (`NEEParts`), and `site_fit --target nee` scores the means, each month weighted by
the part's length in it. The night flag the targets are split by is being recomputed from the solar clock (D34). Used in
the NEE refit.

### D8. On cropland cells the maximum tree cover caps the cell cover, not the natural-share cover it was fitted to (kept)

`vegetation_dynamics.py:711-720` (`_fu_of`): fu = (1 - c) phi(W/(1 - c)) (1 - D), then min(fu, tree_cover_max) on the
cell cover, while the ramp was fitted to the natural-share cover min[c_tree/max(1 - f_crop, 0.05), 1] on cells below 0.3
cropland (CALIBRATION 70-71). In 2010 the natural-share cover exceeds the cap on 19,016 of 47,304 cropland cells; capping
the natural share instead would lower their tree-cover area from 17.34 to 14.59 Mkm2 (-2.75, 6 % of the global 44.33).
Technote (11.23), 15.4. **Decision (a): kept as it is, documented** (with D22 and D48, in the same function).

### D9. The year's climate statistics from monthly means; IBIS 2.6b4 uses daily statistics (fixed; revised floors selected)

`run_global.py:883-885` called `climate_stats_from_monthly` (`loops_gswp3.py:803-836`): Tc and Tw were the extremes of the
monthly means and the degree-days came from the `climanl` interpolation, where IBIS 2.6b4 (`weather.f:909-915`) takes the
extremes of the 10-day running mean and sums daily degree-days. Section 2.2 recorded the method, not its size: over GSWP3
1901-1910 the daily rules give Tc 2.3 K lower (median; p5-p95 0.4-4.2 K), Tw 1.4 K higher and GDD5 21 % higher in the
10,604 cells with 300-900 C d; existence differs in 6,398 of 95,333 cells, almost all admitting more under the daily rule
(positions 6-8 in about 2,000 cells each, the C4 grass in 2,685); the leaf-drop threshold max(0, Tc + 5) differs by more
than 0.01 K on 12,874 of the 36,418 cells with a cold-deciduous position, higher under the monthly rule (median +1.21 K).
The degree-day floors (900 for position 4, 600 for 6-8) were tuned on the monthly statistics. Technote 10.12 item 5,
11.3.1, 11.12.3 item 3. **Decision (b): IBIS 2.6b4's daily statistics, the degree-day floors re-tuned against observed
tree cover. Fixed on technote-review-fixes, commit 307d0f8** (`pheno.climate_stats_daily`, default true; false gives the
release's monthly statistics). Update, 2026-11-02: the preliminary-run comparison is complete. The selected revised
preset floors are 975 °C d for position 4 and 700 °C d for positions 6–8 (`c6698fd`); 900/600 remain the historical
release values. See CALIBRATION section 87 and technote 11.3.1.

### D10. Ice-sheet snowpacks: water lost to single precision and noise temperatures beneath (fixed)

`vadapt` (`snow_physics.py:58-68`) re-divides the float32 total thickness every step, and the heat flux into the soil is
deduced by subtraction (`snow_physics.py:211-221`; `loops.py:852-854`). On 31 Dec 2001, 26,437 cells had packs thicker
than 10 m (largest 1,770.8 m); they gained 2,716 Gt/yr over 2002-2010, and their storage residual is 174 Gt/yr in
magnitude. In a column test float32 kept between 66.8 and 133.7 kg m-2 of 78 kg m-2 of snowfall (none at 1,500 m), where
float64 kept 78.000; the soil flux beneath 1,081 flagged cells is -1.21 to +1.10 W m-2 (p10-p90) in float32 against a
float64 median of -0.017 W m-2, and in 2001 those cells kept the 2-4 m layer above 273.16 K in every monthly mean. No
vegetated cell is affected. Technote 6.8.2, 7.11, 8.14. **Decision (b): fix the single-precision problems only** (the
snow layers updated incrementally; the snow-to-soil flux from the snow-soil conductance), the packs stay uncapped as in
IBIS 2.6b4 (with D17). **Fixed on technote-review-fixes, commit 31a4f0c:** for packs of 10 m or more `vadapt` re-divides
the layers only when the top layer has moved by half its thickness, and the flux into the soil is the conductance of the
buried layer to the soil (`tests/test_snow_deep_pack.py`); thinner packs are computed as before.

### D11. Frozen soil triggered the lower-canopy drought dormancy every winter (IBIS 3.0 trigger; fixed)

`phenology.py:193-207` dropped the lower-canopy drought factors while s_l < `drop_stress_thresh`, with no temperature
test, and the stress factor counts liquid water only (`stomata.py:545-547`). At the end of 2000 the factors were at their
floor on 52 % (D_temperate), 97 % (D_subarctic), 100 % (D_extreme) and 75 % (E_tundra) of the area with deciduous lower
leaf area, 98.6-100 % of it with ice in layer 1; green-up then waited for the thaw and the 14-day recovery. CALIBRATION 18
had fixed the same confusion for the tree-cover accumulators only. Technote 10.6. **Decision (b): the trigger gated by
the growing-season test (`stress_accum_tmin`). Fixed on technote-review-fixes, commit 307d0f8** (`pheno.drop_stress_gate`,
default true); the phenology parameters are refitted in the GPP/ET stage.

### D13. The methane production temperature is a mean to 4 m, against the module's own decomposition profile (fixed; band refit to follow)

`methane.py:196-230` weights each thawed layer below z_w by its thickness over all six layers to 4 m, while
`methane.py:279-297` has decomposition falling as exp(-z/z_tau) with z_tau 0.70-0.77 m, and the docstring (`:180-182`)
says it should not be a whole-column mean. In the NH cells of the release hindcast 77 % (W02) and 72 % (W03) of the
weight lies below 1 m; weighting by exp(-z/z_tau) offline gives tropical +1.3 %, temperate +16 % and boreal +16 % emission
before any refit. Methane is diagnostic; carbon, water and vegetation are unaffected. Technote 14.3, Figure 14.5.
**Decision (b): the production temperature weighted by the substrate profile over the thawed part below z_w, the band
parameters refitted at the 48 towers. Fixed on technote-review-fixes, commit 13b23d5** (`methane.t_anaer_substrate`,
default true; false gives the release's rule); the band refit follows.

### D17. Puddle ice is not capped and grows without bound on the ice sheets (inherited; kept)

The overflow removes only liquid (`loops.py:909-911`; `soil.f:256-259`), and the puddle's heat capacity is added to that of
the top 0.1 m layer (`loops.py:903, 980, 1039, 1069`). At the end of 2010, 6,505 cells hold more than 100 kg m-2 of
puddle ice (largest store 1.42e5 kg m-2), whose heat capacity is a median 27 times (up to about 1,600 times) that of the
top layer; meltwater there never runs off, and puddle storage enters the all-land storage gain (18 mm/yr, 2001-2010).
Vegetated cells are probably unaffected. Technote 7.3. **Decision (with D10): kept uncapped as in IBIS 2.6b4,
documented.**

### D18. The temperate-grass (G06) timing fit was not in the release file (fixed for the refits)

CALIBRATION 54-55 say the G06 timing of `params_31f.yaml` (tthresh_l_c 2.0, gdd_thresh_l 700, sen_ycrit_l 200,
sen_pstart_l 15) stands; the release's `types.G06` has 3.0496 and 58.696 and no sen_ycrit_l or sen_pstart_l (the base 300
K d and 14 h). By the section-54 emulator the C3 grass at the temperate grassland towers greens about 51 days early and
senesces about 26 days late (not re-measured). Technote 10.8. **Decision (b): the G06 timing in the parameter file, the
phenology refitted. Fixed on technote-review-fixes, commit 307d0f8** (`params_32a_base.yaml`, the base of the refits;
the release file is unchanged); CALIBRATION 55 carries the correction.

### D22. The sapwood fraction divides by the cell's tree wood, not the natural-share wood (kept)

`vegetation_dynamics.py:753-754` take the canopy top from the natural-share wood, but `:767-768` compute sapfrac =
clip(sapvolume rho / W, 0.05, 0.5) with W the cell's wood. In 2010, on cells with more than 0.5 kg C m-2 of tree wood, the
median sapfrac is 0.066 without cropland and 0.326 in the cropland bin 0.6-1 (27.8 % of that bin at the cap 0.5), against
0.063-0.065 with the natural-share wood; the wood maintenance respiration of cropland cells at 15 C is 0.450 against 0.338
Pg C yr-1 (+33 %). Technote (11.28), 15.4. **Decision (a): kept, documented.**

### D24. Dew cannot form on a dry leaf or stem (inherited; variant built, tested in the analysis phase)

`canopy_exchange.py:587-589, 600-602, 606-608` cap the wet fraction at 0.8 (w_liq + w_sno)/max[dt (q* - q_air), eps],
which is 0 for an empty store under condensation; `canopy.f:936-938` is the same, although its `fwetcal` comment
(`canopy.f:1704-1706`) provides for "initial dew formation on dry surface". The vapour goes to the soil or stays in the
canopy air. Not quantified globally. Technote 5.6.3. **Decision (with D5, D6, D7, D14): dew on dry surfaces is a variant
of the hourly energy-balance test at the towers, adopted only if it improves hourly H and LE without worsening monthly
GPP and ET.** The variant is `canopy.dew_on_dry` (commit 8416c20; default 0, the IBIS 2.6b4 form).

### D25. In hyperarid cells most of the ground evaporation is water that no store supplied (source fixed in the revised preset)

Release balance 2001-2010: 99 cells of 60S-60N (16.25-27.75N, 28.75-42.25E) have a residual above 5 mm/yr (largest 12.3),
a median 92 % of their soil evaporation (median 7.5 mm/yr). Candidate sources, not separated: case 4 of `setsoi` followed
by the IBIS 3.0 puddle floor (`loops.py:974`), the `wadjust` bounds (`gadjust`, written nowhere) and the |E_g| guard
(`loops.py:875-876`). Small globally (band residual -0.07 mm/yr). Technote 7.11. **Decision (b): the conservation outputs
of D70, one diagnostic year to separate the sources, then a fix of the source.** The outputs are in (commit e6c70cb,
`physics.water_balance`). Update, 2026-11-02: the diagnostic year and source investigation are complete. The
`soil.evap_needs_store` change (`b2d2cca`) prevents ground evaporation without an available store when enabled; it is
true in `params_32a_base.yaml` and false in the function fallback. The remaining adjustment terms and residual limits
are documented in technote 7.11 and 7.13; this does not claim exact closure in every cell.

### D26. The two-stream solution is wrong near K = h (inherited; fixed)

`radiation.py:211-213` perturbs p by +/-0.1 % where |p - q| < 0.001 p, while s2 (`:233`), t1 (`:286`) and term 6 (`:291`)
keep the unperturbed K; the Fortran is the same (`radiation.f:563-564, 584, 663/671, 668/676`). Against an exact solution
the code gives T^mu 0.0152 for 0.1835 (lower NIR, L 4, S 1, albedo 0.3, mu 0.9270). 0.27 % of the sunlit hourly steps
fall in such a window, where up to 0.18 of the direct NIR beam moves between the lower canopy and the soil; energy is
conserved. Technote 4.4. **Decision (b): k = p/avmu after the perturbation, tested against the exact reference of
`make_ch04.py`. Fixed on technote-review-fixes, commit 8416c20** (bit-identical outside the window).

### D27. The GPP split among the positions of a big leaf ignored the acclimation factor (fixed)

`loops_gswp3.py:199-205` called `capacity_shares` with the unacclimated capacities, while the big leaf runs at vmax_pft
x vfac (`stomata.py:826-828`). In spring and autumn, while the acclimation factor of the evergreen position of a mixed
shrub (9 with 10) or conifer (6 with 8) big leaf is below 1, that position received more of the big leaf's assimilation
than its effective capacity, shifting NPP from the deciduous shrub to the evergreen (and from larch to T17). 4,494 of
15,266 cells with shrub leaf area above 0.05 have both shrub positions above 10 %; the size of the shift was not measured.
Technote 9.8. **Decision (b). Fixed on technote-review-fixes, commit 307d0f8** (`loops.acclimation_vfac`, used by the time
step and by the split).

### D28. The bound on ground evaporation limits the soil water, not the latent heat or the ET output (fixed)

`loops.py:671-672` forms the ground heat flux with the unbounded fvapg; `loops.py:875-876` bounds (1 - f_i) fvapg to
+/-1e-3 kg m-2 s-1 for the puddle and soil water only, and the output (`canopy_exchange.py:795`; `loops_gswp3.py:255,
271`) is unbounded. Where the bound acts, the energy balance and the ET output carry water that does not leave the soil;
how often it acts in the release run was not counted. Technote 5.6.5, 6.9, 7.13 item 3. **Decision (b): count it (D70),
then use the bounded rate in the heat flux and the output. Fixed on technote-review-fixes, commit e6c70cb**
(`bounded_ground_evaporation`; the water-balance module counts the steps on which the bound acts).

### D29. With fire off, the default fire scheme still changes the annual update (fixed)

`vegetation_dynamics.py:614` sets the IBIS 2.6b4 background disturbance 0.005 yr-1 when isimfire != 1, but the
per-position mortality (`:622`) and the combustion completeness (`:637`) test only `fire_scheme`, so the 0.005 is scaled
by m_j and split between combustion and litter, and the covers take the leaf-area-weighted mortality. `check_fireoff.py`
(isimfire 0): cdisturb 0.0867 against 0.1498 kg C m-2 yr-1 under the legacy scheme (-42 %); in the F77-harness run legacy
matches the Fortran to 8.9e-16. A fire-off run under the default configuration is therefore not IBIS 2.6b4 without fire,
although `config.py:38-43`, `FIRE.md:829-833` and `loops_gswp3.py:989-990` call fire off numerically irrelevant or
background only. No effect on the release (isimfire 1). Technote 13.1. **Decision (b): both tests gated on isimfire ==
1, with a test that the season-length scheme with fire off equals legacy. Fixed on technote-review-fixes, commit
e6c70cb.**

### D34. Tower runs lose the shortwave of steps whose hour-start sun is down, and the tower clock is not solar time (fixed)

`weather.py:474` sends observed shortwave to diffuse where coszen <= 0, but `compute_radiation` masks those points
(`radiation.py:675`) and zeroes their absorbed fluxes; the tower shortwave (`tower_forcing.py:47-61`) is on local standard
time, while mu is local solar time at the hour start. US-Ha1 2001-2010: 12,417 of 87,648 hours, 0.45 % of the shortwave
(at most 77 W m-2). Global runs are unaffected. Technote 4.1, 4.11. **Decision (b): tower timestamps in local apparent
solar time in the tower reader, before the tower refits; with D33 the sun and the weather generator are evaluated at
mid-step (the UTC day of the forcing kept, documented). Fixed on technote-review-fixes, commits 63fd7c0 and 8416c20:**
`tower_forcing.py` moves the record to local apparent solar time and gives each step the record's mean over its solar
hour, the clock taken from the site's UTC offset and checked against the record's potential radiation;
`weather.sun_midstep` (default true) puts the sun and the generator at mid-step. A follow-up that recomputes the night
flag from the solar clock is in progress: the flag ran one hour ahead at 9 Australian FLUXNET v1.3 towers.

### D35. A methane section without `methane_bands` is silently overridden by the legacy per-biome rates (fixed)

`run_global.py:659-669` calls `biome_methane_params` whenever either section exists, and with no bands
`methane.py:560-563` replaces r_ch4 and t_ref by the per-biome tables (0.15 / 0.32 / 0.18; 299.6 / 283.0 / 277.9 K),
while the log says "without band overrides" (`:677`). No effect on the release (both sections present). Technote 14.6.
**Decision (b): the methane section used as given, the legacy tables only when neither section exists. Fixed on
technote-review-fixes, commit 13b23d5** (`methane_params_for_run`).

### D36. The trace-gas emissions are not the carbon the fire removes (fixed)

`fire_emissions.py:160-163` burns half of the above-ground litter on the burned fraction, which no litter pool loses, and
evaluates the live part on the post-fire pools with the default mortality (`:203-204`) and the `FireEmissionParams`
defaults (`run_global.py:1007` passes neither the per-cell mortality nor the file's values; `FireParams.cc_litter` is read
nowhere); the burned fraction and the emissions are not written. Release hindcast 2001-2010: the diagnostic burns 2.17
Pg C yr-1 (0.83 of it litter) against the 1.38 the fire removes (+58 %), so the release's emission totals (7,767 Tg CO2,
472 CO, 23.8 CH4 a year) cannot be reconciled with NEE; RELEASE_3.0 called them "conserved". The carbon balance is
unaffected. Technote 13.5. **Decision (b): the emissions computed from the combusted carbon per position, the burned
fraction and the emissions written yearly. Fixed on technote-review-fixes, commit e6c70cb** (each position's combusted
carbon with its own emission factors, in agreement with cdisturb; the yearly file carries the burned fraction, the
combusted carbon and the five species); the RELEASE_3.0 wording corrected 2026-11-01.

### D37. The metabolic fraction of wood litter is negative (inherited; kept)

`biogeochem.py:74-77` (= `biogeochem.f:316-335`) with cnwood 200 and cnr[7] 250 gives lignin 0.4369, structural 0.5731
and metabolic -0.0100 of the wood litter, against the Fortran comment's "97 % structural". Carbon is conserved: the
structural and lignin pools receive 101 % of the wood litterfall and Rh is lowered by 0.6 % of it (0.06-0.07 PgC/yr).
Technote 12.1, 12.3. **Decision: kept as in IBIS 2.6b4, documented.**

### D38. In the transient, 2,703 learned-area cells use the methane band of one period and the learned model of another (fixed)

`run_global.py:663-665` takes each cell's methane band from the Köppen period of the run's first year (1901-1930 for the
transient), while the learned wetland model (`wetland_learned.py:112`) uses the band stored in its file (1991-2020 for all
49,545 file cells); 2,703 of these differ. Transient methane and its carbon partition only. Technote 14.6, 14.7.2.
**Decision (b): the band of the file's cells taken from the file in every run. Fixed on technote-review-fixes,
commit 13b23d5** (`LearnedWetland.band_index`; the hindcast's bands are unchanged).

### D39. The root-zone water of the learned wetland area omits the ice factor (fixed; retraining to follow)

`wetland_learned.py:72` and `wetland_area.py:163` sum wsoi poros hsoi; the liquid volume is poros (1 - wisoi) wsoi hsoi
(technote Table 14.3). The release is self-consistent (the model was trained with the same definition). **Decision (b): W
as the liquid volume, the learned model retrained on the final run's hindcast. Fixed on technote-review-fixes, commit
13b23d5** (also in `root_zone_water` of the dynamic area); the retraining follows the final run. Until then the
release's file, trained with the earlier W, draws a warning (new files carry a mark).

### D41. The C4 capacity bounds of the tower fits were converted with the C3 temperature factor (fixed)

`site_fit.py:281` uses the C3 factor for every position, so the C4 bounds were 6.5-49.0 instead of 5.9-44.1 umol m-2 s-1
at 15 C; 12 of the 55 C4 capacity fits of `t31_s2_merged_relax` lie above 44.1, all above the type medians, so the release
C4 capacities would not change (the slopes and stress factors of those fits could). Recorded in CALIBRATION 60 and 63.
Technote 16.5, 16.14 item 2. **Decision (b): the C4 factor for the C4 position, in the GPP/ET refit. Fixed on
technote-review-fixes, commit 63fd7c0** (`site_fit.vmax_bounds_15c`: 5.9-44.1 umol m-2 s-1 at 15 C for the C4 position).

### D42, D43. Two inherited details of the canopy radiation (kept)

D42: the upper-canopy snow factor on omega_sno ramps over 0.1 K (`radiation.py:137-138`; `radiation.f:825`), the lower
over 1 K (`:143`; `radiation.f:843-844`), where the Fortran comment (`:816`) says 1 K for both: up to about +0.02 albedo
for snow-laden upper canopies within 1 K of melting. D43: the diffuse light scattered from the beam is reflected with the
direct albedo in the solution (`radiation.py:225-229`) but absorbed with the diffuse albedo in the bookkeeping (`:273`):
up to a few per cent of the direct NIR beam moves between the upper canopy and the ground over snow at low sun; energy is
conserved. Technote 4.3, 4.4, 4.7. **Decision: both kept as in IBIS 2.6b4, documented.**

### D44. Two ET diagnostics differ from IBIS 2.6b4 without a record (fixed)

The port accumulates aet = (ginvap + gsuvap + gtrans) dt (`loops_gswp3.py:255-256`) and discards the `cascad2` deficit of
negative canopy stores (`loops.py:616-617`), which IBIS 2.6b4 includes through -fvapa (`stats.f:406, 730, 1001`;
`canopy.f:2239-2245`); and it books snow sublimation with the end-of-step snow cover (`loops_gswp3.py:266`) where gsuvap
used the start-of-step cover. Diagnostics only: one year at 150 cells north of 30N gives sublimation 1.84 against 1.81
mm/yr; in 2002, 31 land cells have a negative annual soil evaporation (to -21.6 mm/yr; not attributed). Technote 5.7,
5.8, 5.11, 8.13. **Decision (b): the step's sublimation stored and accumulated, the deficit removed from ginvap (or the
corrected total written) and counted. Fixed on technote-review-fixes, commit e6c70cb** (`et_parts_step`; the
water-balance module counts the canopy overshoots).

### D45. Cells with a tiny positive leaf area got a decomposition E0 of about zero (fixed)

`loops_gswp3.py:503-504` divided the leaf-area-weighted E0 by max(leaf area, 1e-6), so a total leaf area between 0 and
1e-6 scaled E0 down. In the 2010 restart 136 cells (122 E_tundra, 14 E_ice, 61.75-82.75N), 88 % of them with a total leaf
area below 1e-40 (subnormal float32 leaf carbon), had E0 of at most 2.35e-7 K, so f_T = 1 at every temperature and their
soil decomposed about 11 times faster than their neighbours' in frozen periods (soil carbon 0.015 PgC). Technote 12.2.1,
12.9 item 1. **Decision (b). Fixed on technote-review-fixes, commit 307d0f8** (`cell_lt_e0`: below 1e-6 of total leaf
area, the positions' mean E0, as with no leaves).

### D47. Larch (position 8) assimilated in the conifer big leaf but transpired with the broadleaf conductance (fixed)

`canopy_exchange.py:590-594` weighted frac[:, 7] with the broadleaf conductance, as `canopy.f:942-950`, while GPP, NPP and
the capacity split put position 8 in the conifer leaf (`loops_gswp3.py:213-218, 242-246`; `stomata.py:570`); IBIS 2.6b4
puts it in the broadleaf leaf for carbon as well (`stats.f:96-97`). Recorded only in CALIBRATION 17. Position 8 holds 0.05
% of the upper-canopy leaf area at the end of 2010. Technote 5.6.3, 5.11 item 7, 9.11 item 7, 11.12.3 item 2. **Decision
(b). Fixed on technote-review-fixes, commit 307d0f8** (transpiration with the conifer conductance, as its carbon; the
conifer assignment of CALIBRATION 17 remains a departure from IBIS 2.6b4).

### D48. The cover floor keeps a productive tree canopy on near-total cropland (kept)

`vegetation_dynamics.py:720` applies the floor 0.05 after the (1 - c) factor: in 2010, 940 cells (c 0.950-1.000; 2.21
Mkm2) have fu > 1 - c, and their upper-canopy NPP is 0.108 Pg C yr-1, about 0.055 of it above the natural share, against
a harvest of 0.072 on the same cells (0.13 % of the global upper-canopy NPP). Technote 15.4. **Decision (a): kept,
documented** (with D8 and D22).

### D51. The top-layer heat flux uses the temperature after the transpiration adjustment; IBIS 2.6b4 the one before (unrecorded departure; fixed)

`loops.py:1051-1057` forms fhtop_soil after the adjustment (6.10); `soil.f:323-325` forms fhtop in loop 100, before loop
120 (`:349-372`) applies it. Ordinarily below 0.01 W m-2, about 0.25 W m-2 in a hand estimate for an extreme case.
Technote 6.10 item 3. **Decision (b): the start-of-step temperature, as `soil.f`, confirmed with a lockstep check. Fixed
on technote-review-fixes, commit e6c70cb** (with D84). The harness re-run with this order as a local probe (bed0ae7;
`verify/STATUS.md`, Harness settings) agreed to 2e-11 or better in the single-step, 24-step and four of the 7-day
comparisons.

### D52. With a dry top layer and a puddle, the evaporation shares sum to a_p (inherited; fixed)

`soil_physics.py:103-107` (= `soil.f:136-141`) sets q3 = a_p Pl/(Pl + Pi) and q4 = a_p Pi/(Pl + Pi) in case 3, so
(1 - a_p) E_g comes from no store and the ground latent heat is scaled by a_p, against the Fortran comment's "all soil
evap to the puddles". 12 cells at the end of 2000 and 16 at the end of 2010, all at 77.25-77.75S, 160.25-164.25E.
Technote 7.4. **Decision (b): q3 = Pl/(Pl + Pi), q4 = Pi/(Pl + Pi). Fixed on technote-review-fixes, commit
e6c70cb.**

### D53. Ground snowfall beyond rho_s H per step is discarded (inherited; fixed)

`loops.py:745-746` (= `snow.f:138-139`) bounds the new snow cover by 1 - f_i, and the rest of the step's ground snowfall
goes neither to the pack nor to the soil. It takes more than 37.5 kg m-2 of ground snowfall in one hour; not counted.
Technote 8.5. **Decision (b): the excess added to the pack, with a counter. Fixed on technote-review-fixes, commit
e6c70cb** (the water-balance module counts it).

### D54. The cold-start soil, ground and bedrock temperature is 278.0 K; IBIS 2.6b4 278.13 K (unrecorded departure; fixed)

`ibis/state/soil.py:76, 79, 109` against `initial.f:628, 630`. No effect after the 200-year spin-up (the deepest bedrock
drifted 0.014 K over 1870-1900 outside the thick-snow cells). Technote 6.7, 6.10 item 5. **Decision (b): 278.13 K. Fixed
on technote-review-fixes, commit e6c70cb.**

### D55. `apply_prescribed_lai` resets the lower cover with the IBIS 2.6b4 rule (fixed)

`prescribed_lai.py:68` sets fl = max(0.25, min(0.975, totlail)) after every annual update (`run_global.py:593-600,
964-966`), ignoring `pft.cover_k` and `pft.fl_min` (0.5 and 0.05 in the release; `vegetation_dynamics.py:722-726`). Off in
the release. **Decision (b): the annual update's cover rule, with a test. Fixed on
technote-review-fixes, commit e6c70cb.**

### D56. `stomata.frost_on` had no effect (fixed)

Its ramp reads `atm.td` (`loops.py:305-309`), which the daily loops never update (`loops_gswp3.py:142-151`;
`loops.py:1353-1362`): td is 283.0 K in all 95,333 cells of the 2010 release restart, so with frost_t1 = 5 C the ramp is 1
everywhere; `tests/test_cold_season.py` checks only that the parameters load. Off in the release. Technote 9.3.3.
**Decision (b): the ramp reads the day's mean temperature. Fixed on technote-review-fixes, commit e6c70cb** (the day
loops write the day's mean air temperature into `atm.td`); CALIBRATION 53 carries a note.

### D57. `stomata.acclim_light` would scale the grasses' light-limited rate by the cropland factor (fixed)

`stomata.py:832-836` takes the grasses' light factors from vp_eff, which includes the cropland factor (0.495 at full
cropland), although the grasses do not acclimate. No configuration turns acclim_light on. Technote 9.3.3.
**Decision (b): the light factor from the acclimation and the frost only, without the cropland factor. Fixed on
technote-review-fixes, commit e6c70cb.**

### D58. `stomata.scale_leaf_only` rests on a wrong premise (comment fixed)

The comment at `stomata.py:100-106` and CALIBRATION 16 say IBIS 2.6b4 credits the stems' PAR to the leaves'
photosynthesis, but the top-leaf light already carries the leaves' share L/(L+S) (`radiation.py:514`); what the big-leaf
scaling counts for the stems is capacity, and the switch removes that and applies L/(L+S) to the light a second time. Off
in the release and inert with the sunlit/shaded canopy (CALIBRATION 51). Technote 9.6.2. **Decision (batch 1): the code
comment corrected. Fixed on technote-review-fixes, commit e6c70cb;** CALIBRATION 16 carries the correction.

### D59, D60. Two kept details

D59 (inherited): the big-leaf running means `a10scalparam` and `a10daylight` are updated in each of the three canopy
iterations, as `canopy.f:52-66`, so their memory is 80 daylight steps instead of 240; unused with the sunlit/shaded canopy
of the release. D60: `stress_accum.py:34` uses a smoothed stress of 0 as "unset", so a cell whose smoothed value is exactly
0 takes the next year's mean unsmoothed (26,677 cells at 0 in 2010, all with a yearly stress of 0; no effect found).
Technote 9.6.2. **Decision: both kept, documented; D60's sentinel to change at the next restart-format change.**

### D61. A frozen anaerobic zone still produced methane through the top-layer temperature (fixed)

`methane.py:226-228` falls back to the top-layer temperature when no thawed soil lies below z_w, and `methane_flux`
(`:371-382`) does not use the thawed fraction, contrary to the docstring (`:185-187`). Under 0.1 % of each band's emission
on the 2001-2010 month-mean states. Technote 14.3. **Decision (b): no production when the thawed anaerobic thickness is
zero. Fixed on technote-review-fixes, commit 13b23d5.**

### D68. Puddle capacity 10 kg m-2 and base permeability 0.04, against IBIS 2.6b4's 4.5 and 0.10 (unrecorded departure; kept)

`fortran/params.soi:46-47` (read by `readpars.f:224-225`) has bperm 0.10 and wpudmax 4.5; the port's defaults
(`loops.py:92, 94`; `physics_params.py:478, 480`), repeated in the release file (`params_31s_ch4h.yaml:296, 298`), are 10
and 0.04, both since the initial commit 4ca4481 (`git log -S`), with no recorded reason. For the same bottom-layer state
the drainage is 0.4 times IBIS 2.6b4's; over 60S-60N the release drains 203 mm/yr, 58 % of runoff plus drainage. Technote
7.13 item 6, 17.9.1 item 4. **Decision: kept as IBIS 3.0 parameters, since the calibration rests on them; the untraced
origin is recorded in CALIBRATION 87.**

### D69. The surface flux and temperature bounds take their limits from unrecorded environment variables (fixed)

`loops.py:666` reads `IBIS_FLUX_MAX` (default 2000) and `:1144` `IBIS_T_MAX_DEV` (default 40) at trace time; no script,
cfg.yaml or parameter file sets them, the cfg.yaml and run.log of `final_rel30` do not record them, and nothing counts when
the bounds act (in a synthetic re-run of 31 October they did not act). Technote 5.10, 17.4. **Decision (batch 2): log the
bound limits. Fixed on technote-review-fixes, commit e6c70cb** (`run_global` logs both limits).

### D70. No water-conservation diagnostic (fixed)

`gadjust` is computed (`loops.py:1100-1117`) but written to no output, and wtot is hard-zero (`loops.py:1230`;
`state/hydrology.py:24`), unlike IBIS 2.6b4's wtot/aydwtot (`stats.f`); the IBIS 3.0 floors (root uptake, puddle,
`soilh2o`, `puddle_freeze_thaw`) and the top-layer ice floor (`:936`) are booked nowhere. Non-conservation can be found
only offline from the restarts. Technote 7.9, 7.11, 7.13 item 4. **Decision (b): a yearly output of the adjustments
(gadjust, the floors and the unextracted transpiration), a wtot/aydwtot storage diagnostic, and counters of the
evaporation bound and of discarded snow. Fixed on technote-review-fixes, commit e6c70cb:** `hydrology.wtot` is
filled each step as `stats.f` does, and the optional `WaterBalance` module (`physics.water_balance`, default false)
writes `ayadjust` with its per-source terms, `aydwtot`, `wtot` and the counters to the yearly file. Follow-up `75d8935`
also books rain and melt dropped below the mixing threshold before reaching the soil. See technote 7.11 for the
diagnostic sign convention, units and remaining residual.

### D71. `load_state` can misalign a restart written with the other `grid.nbedrock`, without an error (fixed)

`restart.py:65-150` assigns the arrays by position and accepts leaf-count differences equal to the offsets of the
optional modules, and `soil.tsoi_deep` changes the count by one in the middle of the tree. The release restart of 2010
(272 leaves, nbedrock 3) loads without error into nbedrock-0 states for all five module chains, with `snow.tsno` receiving
a (95333,) array; BEDROCK.md said the load would fail, and `scripts/convert_restart.py` has no bedrock handling. No effect
on the release (all three stages ran nbedrock 3). Technote 6.6.1. **Decision (b): a structure check in `load_state`. Fixed
on technote-review-fixes, commit e6c70cb:** `load_state` compares the number of arrays of each non-optional module in the
file's saved structure with the template's, so a restart of the other `grid.nbedrock` raises a ValueError naming
SoilState; BEDROCK.md corrected.

### D77. The spin-up's yearly Rh and NEE reflect one soil call of n, and the files do not say so (fixed)

`loops_gswp3.py:576-583` adds only the last call's tco2mic after the `nsoilcall` calls of a day; with nsoilcall 10 (the
spin-up of `final_rel30`, 1701-1850) the yearly Rh and NEE are not carbon-consistent with the change of the pools. No
effect on the transient or the hindcast. **Decision (b): an `nsoilcall` attribute on the yearly files. Fixed on
technote-review-fixes, commit e6c70cb** (with a note in the accelerated years).

### D80. The release file sets two solid-phase keys that the release never reads (fixed for the refits)

`params_31s_ch4h.yaml:300-301` sets `soil.rhosoi_val` 2650 and `soil.csoi_mineral_val` 870, while every cell and layer of
the release restarts carries 2636.5 and 880.5 (the 1 % organic fraction of IBIS 2.6b4; `soil_init.py:114-118`) and
`_solid_phase` (`loops.py:190-215`) reads the keys only where the state has no values. CALIBRATION 60 lists them as dead.
Technote 6.2.2. **Fixed on technote-review-fixes, commit 307d0f8:** `params_32a_base.yaml` drops them; the release file
keeps them, unread.

### D84-D97. Small departures, comments and docstrings (batch 4)

- D84 (unrecorded departure): the puddle's phase change runs before the second water bounds (`loops.py:1097-1125`), where
  `soil.f` runs it after them; an effect of order 1e-7. Fixed (e6c70cb), with D51.
- D85: the optical parameters are cast to float32 when read from the parameter file (`physics_params.py:200-207`), 1.5e-8
  relative in float64 runs. Fixed (8416c20): the run's float type.
- D86 (inherited): the "no vegetation" zeroing of `twostr` never triggers (`radiation.py:279-281`; `radiation.f:638-639`).
  Kept.
- D87 (unrecorded departure): `snow_coverage` gives a pack thinner than 1e-10 m the minimum thickness
  (`snow_physics.py:276-278`), which `snow.f` does only in its snowfall block; no cell of the 2001 state reaches it. Fixed
  (e6c70cb; the substitution dropped).
- D88: the hourly methane substrate is the previous day's Rh, while the partition cap uses the month's
  (`loops_gswp3.py:312, 578-583`); 27 cells clipped in 2001. Kept.
- D89: `soil.zwpmax` does not reach the puddle cover of the surface humidity (`canopy_exchange.py:535` has 0.5, as
  `canopy.f:816`). Fixed (e6c70cb; bit-identical at 0.5).
- D90 (inherited): the lower-canopy sensible heat diagnostic `fsenl` is computed against T_12, not T_34
  (`canopy_exchange.py:762`; `canopy.f:1279`); read nowhere. Fixed (e6c70cb).
- D91-D97: wrong or stale comments and docstrings: cropland clearing (`ayclearing` is always 0), fire (`fueldry`,
  `d_crit`, the legacy scheme called the default, `enable_fire_emissions`), the methane modules, phenology (`sumnow`, the
  linear steps), state units and periods (`tnmin`, the run-long leaching sums), `twoset`, and the canopy and constants
  code. Fixed: D93 in 13b23d5, the others in 8416c20.
