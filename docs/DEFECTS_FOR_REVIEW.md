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

## One NaN cell in every v11 state (open)

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
