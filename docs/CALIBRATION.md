# Calibration

Started 2026-09-09, after the fire and methane modules were re-validated on the
corrected carbon cycle. Order set by Hangkai: carbon, then ET, then the smaller
parts. Methane is evaluated against FLUXNET-CH4 (Chenchen's processing).

## 0. What already existed, and what had to be fixed before fitting

`scripts/calibrate_fluxnet.py` (Min) already had the right architecture: all
sites in one `npoi` vector, so a parameter evaluation is a single model run,
driving CMA-ES over per-PFT `vmax`/`specla` and Ball-Berry `coefm`/`coefb`
with literature bounds. 202 sites load against current code.

Four things had to be fixed or ruled out first.

**The restart would not load.** "258 leaves but template has 257". `both_fix`
was written with `nbedrock: 3` (docs/BEDROCK.md); a template built with
`nbedrock=3` matches at 258 leaves with zero shape mismatches.

**The site-to-cell mapping had to be proved, not assumed.** `surta.nc` is N->S
(89.75 first), matching the harness's `GSWP3_LAT`, and `nland = 95,333` equals
the restart `npoi`. `argwhere` (run_global) and `where(ravel())` (harness) give
identical row-major order. Geography: corr(|lat|, deep soil T) = **-0.851**
over all land and **-0.891** at the 205 sites; southernmost sites return
Australian, northernmost Svalbard.

**`specla` is dead in that harness.** It appears only in
`vegetation_dynamics` (`plai = cbiol * specla`), and `calibrate_fluxnet.py`
never calls `dynaveg` -- 12 of its 38 CMA-ES parameters had no effect on the
loss. Twelfth instance of the defect-0 pattern, first one in calibration code.
`verify/calibration/site_transient.py` does call `dynaveg`, so `specla` is
reachable there.

**`reco = ayco2mic` is Rh, not Reco**, and fitting Rh needs centuries of
soil-carbon re-equilibration per evaluation. Reco is reported, never fitted.

## 1. Towers cannot set the GPP magnitude

At the 131 matched cells the model is **1.09x FLUXCOM** (own LAI) / 1.04x
(MODIS LAI), consistent with 1.01x globally -- but **0.81x the tower value**.
FLUXCOM at the same cells is **0.75x the tower value**:

| IGBP | n | tower | FLUXCOM | ratio |
|---|---|---|---|---|
| ENF | 49 | 1.35 | 0.89 | 0.64x |
| DBF | 24 | 1.53 | 1.04 | 0.69x |
| EBF | 15 | 2.29 | 1.64 | 0.76x |
| GRA | 37 | 0.92 | 0.74 | 0.81x |

A gridded product trained on these towers is also well below them at their own
cells: the gap is 0.5-degree representativeness. Fitting it away would have
inflated global GPP ~1.4x while every site plot appeared to improve. Towers
constrain **scale-insensitive** quantities -- WUE, seasonality, interannual
variability, PFT ordering. Magnitude stays anchored to the gridded products.

FLUXCOM is `g m-2 day-1`. Read as `kgC m-2 yr-1` it gives a nonsense 2.03x.
Second time that unit has bitten; `x365.25/1000` both times.

## 2. The missing transient, and why it mattered less than expected

`both_fix` CYCLES 1901-2000 forcing at a fixed 350 ppm, so its end state is
equilibrated to the 20th-century mean, not to the trajectory a tower measured.
Hangkai asked for the step between spin-up and validation; it was missing.

`site_transient.py` runs 1901-1996 with real GSWP3, rising CO2 and `dynaveg`,
then evaluates 1997-2006. Legitimate at 171 cells rather than 95,333 because
IBIS has **no lateral coupling** -- no routing, no groundwater exchange -- so
each column integrates independently.

Result: soil carbon drifted **-1.5% over 86 years**, and on the same 131 sites
GPP/FLUXCOM went **1.09x -> 1.10x**. The spin-up CO2 inconsistency is
immaterial. Worth adding for correctness; it was not hiding anything.

There was also **no transient CO2 anywhere in the model** -- `co2init` is a
scalar applied once at state creation, so even `calibration_transient` held
296 ppm. `ibis/io/co2.py` adds a published trajectory with its provenance and
limits stated.

## 3. The residual is cold-structured, not aridity-structured

Recorded because it was claimed the other way first. Binning model/FLUXCOM by
precipitation put the worst band (<400 mm) at 0.26x, and the transient
appeared to fix it (0.26 -> 1.00). It did not: that was a **sample artifact**,
two runs drawing different site sets. On common sites, 0.26 -> 0.29.

The low-ratio "dry" cells are CA-Man, CA-NS1-NS7, US-Prr, RU-Che, RU-Cok,
US-Atq, US-Ivo, GL-ZaF, GL-NuF -- all 55-75N. Their precipitation is low
*because they are cold*. The binning conflated the two.

Globally, from `both_fix` year 2200:

| band | IBIS LAI | MODIS | ratio | GPP | NPP | cveg |
|---|---|---|---|---|---|---|
| 30S-30N | 3.51 | 1.83 | 1.91x | 1.295 | 0.576 | 3.67 |
| 30-50N | 3.20 | 1.36 | 2.35x | 0.730 | 0.366 | 3.44 |
| 50-60N | 2.20 | 2.16 | 1.02x | 0.500 | 0.278 | 2.59 |
| 60-70N | 0.46 | 1.40 | **0.33x** | 0.112 | 0.063 | 0.63 |
| 70-90N | 0.03 | 0.21 | **0.13x** | 0.006 | 0.003 | 0.03 |

**2x too leafy in the tropics and mid-latitudes, 3-8x too bare above 60N.**
The global mean is right because the errors cancel.

Not the bioclimatic filter: at 60-70N `exist` permits evergreen shrub,
deciduous shrub and C3 grass in **96%** of cells and boreal conifer in 59%,
but their leaf area is 0.09, ~0, 0.08 and 0.02. The PFTs are allowed and do
not grow -- the low-LAI -> low-GPP -> low-NPP -> low-LAI attractor named in
docs/GPP.md.

## 4. `vmax x0.54` put nine of twelve PFTs below physical minimum

Against the calibration harness's own literature bounds:

| PFT | current | IBIS default | range | |
|---|---|---|---|---|
| tropical_broadleaf_evergreen | 35.1 | 65.0 | 40-100 | below |
| warm_temperate_bl_evergreen | 21.6 | 40.0 | 25-70 | below |
| temperate_broadleaf_deciduous | 16.2 | 30.0 | 20-60 | below |
| **boreal_conifer_evergreen** | **13.5** | 25.0 | **15-40** | **below** |
| boreal_broadleaf_deciduous | 16.2 | 30.0 | 20-50 | below |
| evergreen / deciduous shrub | 14.9 | 27.5 | 15-45 | below |
| c3_grass | 13.5 | 25.0 | 15-45 | below |

The scalar did not merely fudge the global mean; it pushed most of the model's
photosynthetic capacity outside plausible physiology, and it did so uniformly
-- suppressing high-latitude productivity that was already 3-8x too low in
order to offset tropical/temperate leaf area that is 2x too high.

### Consequence for the Arctic methane and soil-carbon deficits

NPP at 70-90N is **0.003 kgC m-2 yr-1**. At equilibrium `pool = input x
residence time`, so even a millennial residence time yields almost nothing --
which is why Arctic soil carbon is 4.33 against NCSCD's ~55, and CH4 north of
60N is 1.0% against a target 8%. docs/METHANE.md 5q attributed this to missing
cryoturbation, vertical resolution and deep carbon. Those gaps are real, but
underneath them is a more elementary one: **there is almost no carbon input to
decompose, because the model grows almost no vegetation there.** The "~6x
lower Arctic Rh per unit area" is downstream of this.

## 5. LAI is a weak lever on GPP

Measured elasticity `dlnGPP/dlnLAI` between the two prescribed-LAI arms:

| band | n | LAI a | LAI b | elasticity |
|---|---|---|---|---|
| <30N | 15 | 5.31 | 2.89 | **0.04** |
| 30-50N | 81 | 5.23 | 2.45 | 0.09 |
| 50-60N | 29 | 4.14 | 2.60 | 0.10 |
| all | 128 | 4.95 | 2.54 | **0.09** |

Beer-Lambert `1-exp(-0.5 L)` implies 0.22 (LAI 5) to 0.58 (LAI 2) over the
same range. The formula itself is faithful: `canopy_scaling` gives
`scale = (1-exp(-extpar(lai+sai)))/(extpar*lai)`, and multiplied by `lai` this
is `A_top (1-exp(-k(L+S)))/k`, the correct big-leaf integral. The weak
response is a *consequence* of an optically thick canopy -- stem area also
intercepts, and `extpar(L+S)` is large enough that absorbed light is nearly
saturated.

**So correcting LAI will not correct GPP**, and the two biases have to be
fixed with different parameters: `vmax` for GPP, `specla` for LAI. That also
means the 2x LAI excess contributes little to the GPP excess, so the 1.55x
gap docs/GPP.md left open is not an LAI artifact.

## Where this leaves the 1.55x problem

It is now localised rather than solved. IBIS with physically plausible `vmax`
overpredicts GPP in warm and temperate regions and underpredicts it above
60N; a single global scalar traded one against the other and landed the mean
on FLUXCOM. Restoring `vmax` globally is not available: `both_fix` measured
`dlnGPP/dlnVmax = 0.75`, so undoing x0.54 returns ~211 PgC/yr against a
published 120-130.

## Status

* Stage 1 (carbon) -- diagnosis complete, fitting not started.
* Stage 2 (ET) -- not started. Model ET is 1.22x tower and WUE 0.75x, so the
  error is in the coupling; `vmax` and `coefm` will need fitting jointly
  rather than carbon-then-ET.
* Stage 3 (methane) -- 459 AmeriFlux CH4 site-years, of which only **51 (11%)**
  fall within GSWP3's 2010 cutoff, and only 25 of 170 north of 60N. The CH4
  sites carry their own gap-filled meteorology, so they will be driven from
  tower forcing rather than GSWP3. 43 EuroFlux sites additionally available.

## 6. The bare Arctic is two different problems

At the dead high-latitude cells `plai` sat at **exactly 0.010** per PFT for
evergreen shrub, deciduous shrub and C3 grass -- `xminlai`, the floor the code
permits -- with leaf carbon 0.0022 kgC m-2. Not "low vegetation": the minimum
and nothing more. All three PFTs pass `exist`, so the bioclimatic filter is
not the gate.

`verify/calibration/arctic_seed_test.py` seeded every cell with MODIS
annual-maximum LAI, changed **no parameters**, and ran 20 years with
`dynaveg`:

| site | lat | seeded | before | after 20 yr | GPP final | GPP obs |
|---|---|---|---|---|---|---|
| FI-Sod | 67.4 | 2.04 | 1.10 | **1.89** | 0.453 | 0.616 |
| FI-Lom | 68.0 | 2.06 | 0.93 | **1.88** | 0.486 | 0.455 |
| US-Ivo | 68.5 | 1.07 | 0.030 | 0.057 | 0.009 | 0.290 |
| RU-Che | 68.6 | 1.08 | 0.036 | 0.115 | 0.023 | 0.306 |
| US-Atq | 70.5 | 0.86 | 0.030 | 0.049 | 0.008 | 0.190 |
| RU-Cok | 70.8 | 0.88 | 0.030 | 0.060 | 0.010 | 0.381 |
| GL-ZaH/ZaF | 74.5 | 0.12 | 0.030 | **0.030** | 0.001 | 0.08 / 0.36 |

**Sub-Arctic cells are bistable and the cold start chose the wrong branch.**
FI-Sod and FI-Lom hold ~1.88 for twenty years from a seed of ~2.0, against a
cold-start state near 1.0, with GPP landing near observed. Nothing changed but
the initial condition. `both_fix` was a cold start (its own cfg says so), so
leaf area had to bootstrap from `xminlai`; warm cells escaped, these did not.

**True tundra and Greenland collapse back to the floor.** Seeded LAI decays to
0.03-0.115 within twenty years and GPP to 0.001-0.023 against an observed
0.19-0.38. There the bare state IS the model's equilibrium, and the `vmax`
experiment showed restoring photosynthetic capacity does not reach it either
(RU-Cok 0.00 -> 0.10 against 0.38; GL-ZaF 0.00 -> 0.00). **No calibration of
`vmax` or `specla` will fix the Arctic carbon input.** That is the origin of
the Arctic soil-carbon and CH4 deficits, and it is a structural limitation to
declare rather than a target to fit.

The aggregate verdict the script printed -- "partially retained", Arctic mean
LAI 1.027 -> 0.514 -- averages these two opposite behaviours and should not be
quoted on its own.

### The mid-latitude LAI excess is an attractor too

Seeding pushed all-cell mean LAI from 4.95 to 2.28 (MODIS); it climbed back to
4.49 within four years and sat at 4.60 at year 20. Since equilibrium LAI is
`NPP x aleaf x tauleaf x specla`, forcing the state does not move the
attractor. So **LAI can only be corrected through `specla` or allocation,
never through initialisation** -- and `specla` is reachable in the
`site_transient` harness because it calls `dynaveg`.

### `seed_lai`

`run_global` gained `seed_lai`, which applies the observed field ONCE and then
lets `dynaveg` evolve it -- distinct from `prescribe_lai`, which re-imposes it
every year. Guarded by two reachability tests in
`tests/test_no_dead_wiring.py`, because thirteen flags, parameters and state
fields in this project have now been found declared, documented, read and
never reaching the physics.

## 7. Two corrections to findings recorded earlier in this document

**Reco was never compared.** Section 0 said the fix for Min's `reco =
ayco2mic` was `ayco2mic + ayco2root`. `tco2root` is **never written** --
`ayco2root` is exactly 0.000000 in all 1710 evaluation rows -- so the fix
changed nothing and the reported "Reco 0.52x" was Rh against tower Reco.
Rh/Reco is *expected* to be ~0.5, so that figure indicates nothing wrong and
is withdrawn. The model has no autotrophic respiration diagnostic, so Reco
cannot currently be evaluated at all. Thirteenth instance of defect-0, and the
first to defeat one of my own corrections.

**`agddl` growing without bound is NOT a port defect.** It reaches 3,364,903
at DK-Sor and 2,121,381 at FI-Hyy against an annual scale of ~2,000, because
it resets only when the 10-day mean falls below -5 C, which never happens in
maritime climates. Checked against `vegetation.f:100-105`: IBIS 2.6b4 does
exactly the same. It is also functionally harmless, since
`templ = min(1, (agddl-150)/50)` saturates at 1 above `agddl = 200` and
float32 carries 3.4e6 without difficulty. Recorded as checked, not fixed.

## 8. The global test refutes section 6's bistability, and finds a restart defect

Section 6 concluded from two tower cells that sub-Arctic vegetation is
bistable. Run at all 95,333 cells for 30 years from `both_fix` year 2200, with
a control arm on identical forcing, that does not hold:

| cell | 2200 | control+30 | seed+5 | seed+30 |
|---|---|---|---|---|
| FI-Sod | 1.02 | 0.74 | 0.88 | **0.75** |
| FI-Lom | 0.96 | 0.67 | 0.63 | **0.67** |
| FI-Hyy | 4.45 | 1.54 | 2.44 | **1.53** |
| RU-Che | 0.02 | 0.01 | 0.23 | **0.02** |

Seeding gives a transient bump that decays back to the control within 30
years, in every band (60-70N: control 0.41, seeded 0.42) and with GPP
identical between arms to three decimals. **Withdrawn: there is no
exploitable bistability, and `seed_lai` does not change the equilibrium.**

**The tower result was a different experiment, not a wrong one.** The tower
run used 1997-2006 forcing with transient CO2 (~370 ppm); the global run
cycles 1901-2000 at a fixed 350 ppm. What looked like bistability is the
attractor being **higher under recent climate and CO2**. That matters for the
LAI comparison itself: MODIS LAI is a 2000-2020 product, `both_fix` is
equilibrated to 1901-2000 at 350 ppm, so part of the "Arctic LAI deficit
against MODIS" in section 3 is a temporal mismatch rather than a model error.
Comparing them requires a transient to the present -- which is what Hangkai
asked for at the outset.

Also corrected: section 6 reported `both_fix` vegetation drifting -13.6%.
The 2100-2200 trajectory is stationary (LAI 1.77-1.81, GPP 0.535-0.557); that
figure came from comparing one year against a high year.

### The restart clobbered the 30-year climate normals

`run_global` wrote `tc`, `tw`, `tcmin`, `gdd0`, `gdd5` unconditionally at the
start of **every** run, from a SINGLE forcing year, over normals the model had
accumulated for 300 years. `existence` reads them directly. Against the
year-2200 checkpoint:

| normal | restart | 1901 reseed | mean abs diff | max |
|---|---|---|---|---|
| tc / tcmin | -14.38 | -15.24 | **1.47 K** | 10.9 |
| tw | 9.27 | 8.55 | 1.10 K | 7.7 |
| gdd0 | 3438.2 | 3400.0 | 82.4 | 774.8 |
| gdd5 | 2531.6 | 2500.5 | 73.4 | 773.5 |

**PFT existence flipped in 6,034 cells (6.3%), 3,173 of them north of 50N** --
exactly where leaf area is marginal. `climanl2` is a 30-year running mean, so
a restarted run needed ~30 years to recover and anything shorter carried the
perturbation. This affected the v9/v10 restart chain the fire and methane
validations were built on. It explains the step-down between `both_fix` year
2200 and the continuation arms.

Inverse of the defect-0 pattern: a value that IS written where it should not
be, rather than one that never arrives.

**Fixed** -- gated on `irestart == 0`, guarded by
`test_climate_normals_not_reseeded_on_restart`. Verified empirically: after one
restarted year the normals move by 0.048 K (tc) and 2.4 GDD, i.e. 1/30 of the
reseed perturbation, which is the correct one-year `climanl2` update.

Neither the seed-vs-control comparison (both arms carried the same
perturbation) nor the `site_transient` results (that harness never reseeded)
are invalidated by this.

### Consequence for the calibration state

The cleanest path is a cold start followed by a transient to the present with
rising CO2, not a restart-and-continue. That supersedes `both_fix`, which the
fire and methane validations rest on, so it is a decision about scope rather
than a detail.

## 9. The fit at full domain: pattern largely fixed, magnitude not

`fit_carbon_v11c` (152 CMA-ES evaluations, 4,331 cells, 9 free PFTs, 3
frozen) gave loss 1.7166 -> 1.0200 fitted and 1.9012 -> 1.2206 held out.
Three PFTs pinned at the lower `vmax` bound: tropical broadleaf evergreen
40.1 (floor 40), warm-temperate evergreen 26.2 (25), C4 grass 8.2 (8).

Run on all 95,333 cells for 25 years from the v11 2010 state, against a
default-parameter control on identical forcing (`val_v11c/`):

| quantity | control | **fit** | observed |
|---|---|---|---|
| GPP PgC/yr | 234.1 | **169.7** | 120-130 |
| NPP | 105.4 | 80.2 | 55-60 |
| Rh | 94.5 | 78.7 | 49-58 |
| ET mm/yr | 525.0 | 523.3 | 500-550 |
| veg C PgC | 804.5 | 669.7 | 450-650 |
| soil C kgC/m2 | 13.74 | 13.01 | 10-15 |
| litter | 1.13 | 0.93 | 0.4-0.8 |

By band (ratio to observation, control -> fit):

| band | LAI/MODIS | GPP/FLUXCOM |
|---|---|---|
| 30S-30N | 2.85 -> **1.53** | 1.53 -> **1.11** |
| 30-50N | 4.03 -> 1.89 | 2.29 -> **1.60** |
| 50-60N | 2.32 -> 1.35 | 1.97 -> 1.47 |
| 60-70N | 1.56 -> **1.07** | 2.15 -> 1.51 |
| 70-90N | 1.49 -> 0.62 | 2.47 -> **0.94** |

### What it fixed

The spatial pattern. Before the fit GPP ran 1.5-2.5x FLUXCOM in every band;
after it the tropics are at 1.11x and the far Arctic at 0.94x, and LAI at
60-70N is 1.07x MODIS. The two ends of the latitude range -- the places where
`vmax x0.54` had produced errors of opposite sign -- are now close to right
with a single parameter set and no fudge.

### What it did not fix

**Magnitude.** Global GPP is 170 PgC/yr against 120-130: 1.31-1.41x. The
excess is now concentrated in **30-70N at ~1.5x**, where the fit moved
`temp_con_ever` 30 -> 19.3, `temp_bl_dec` 30 -> 22.7 and `c3_grass` 25 -> 16.2
and still could not close it. `temp_con_ever` was dominant in only 40 sampled
cells, so it is weakly constrained.

**ET did not move.** 525.0 -> 523.3 mm/yr while GPP fell 27%. So the fit made
WUE *worse* (GPP/ET fell 27%), and the section-1 finding that model WUE is
0.75x observed at towers is now sharper: transpiration in this model is not
governed by the same lever as photosynthesis. Stage 2 cannot be "then fit
ET" -- the coupling itself is the problem.

### Why the sample and the domain disagree

At the sample the fit reported GPP 0.83x FLUXCOM; at the domain it is ~1.35x.
The loss is a per-cell mean of squared log-ratios, so a sparse cell at 0.5x
counts as much as a rainforest cell at 1.5x, and the optimiser satisfied the
many low-GPP cells at the expense of the few high-GPP ones that carry the
global total. A magnitude-honest fit needs the loss weighted by observed flux
(or an explicit total-GPP term). This is a design error in `fit_carbon.py`,
not a property of the model.

### Not converged

LAI in the fit arm fell to 1.85 at year 1 (the `specla` change is immediate)
and climbed to 2.32 by year 25, still accelerating, as biomass re-equilibrates.
The fit judged LAI after only 8 adjustment years. `val_v11c/fit_ext` extends
the arm 50 years; the LAI and vegetation-carbon rows above will move.

### Addendum: the continuation converged, and it changes the fit window

`val_v11c/fit_ext` extended the fit arm 50 years (to 2085):

| quantity | fit_2035 | **fit_2085** | observed |
|---|---|---|---|
| GPP PgC/yr | 169.7 | 172.8 | 120-130 |
| Rh | 78.7 | 74.7 | 49-58 |
| **veg C PgC** | 669.7 | **602.0** | 450-650 -- **in range** |
| **soil C kgC/m2** | 13.01 | **12.55** | 10-15 -- in range |
| litter | 0.93 | 0.87 | 0.4-0.8 |
| LAI mean | 2.32 | **2.49** | |

LAI plateaued at 2.47-2.51 from ~2046 on (the 0.04 oscillation is the
ten-year forcing cycle). Vegetation carbon fell into range as the woody pools
finished re-equilibrating. Every carbon **pool** is now inside its observed
range; the **fluxes** remain ~1.35x.

**Consequence for the fit.** Equilibrium LAI is 2.49; at 8 adjustment years,
where `fit_carbon` judged it, LAI was ~1.9 -- 76% of the plateau. The
optimiser therefore saw a 1.8x LAI excess against MODIS when the true excess
is 2.4x, and under-corrected `specla`. The corrected-loss re-fit
(`fit_carbon_v11d`) was launched with the same 8-year window; killed at 6 s
and relaunched with **25 adjustment years**, where LAI has reached 96% of its
plateau (2.39). Per-evaluation cost roughly triples; 150 evaluations is
~8 h rather than ~3.5.

By band at 2085, GPP/FLUXCOM: 1.13 / 1.63 / 1.53 / 1.58 / 0.86. The 30-70N
excess is unchanged from 2035 -- it is not a transient.

## 10. Re-fit with the corrected loss and a 25-year window (`fit_carbon_v11d`)

152 evaluations, same 4,331 cells and frozen set as v11c; flux-weighted loss
with a total term (section 9); 25 adjustment years (section 9 addendum).

Loss **1.8787 -> 0.7039** fitted; held-out reported below. Baseline differs
from v11c's 1.7166 because the loss definition changed -- the two are not
comparable numbers.

| PFT | vmax0 | **vmax** | bounds | specla0 | **specla** |
|---|---|---|---|---|---|
| trop_bl_ever | 65.0 | **40.1** (floor) | 40-100 | 13.0 | 10.1 |
| trop_bl_dec | 65.0 | 80.5 | 40-100 | 21.6 | **11.5** |
| warm_bl_ever | 40.0 | **25.4** (floor) | 25-70 | 30.0 | 19.4 |
| temp_con_ever | 30.0 | 32.4 | 15-50 | 10.8 | 9.4 |
| temp_bl_dec | 30.0 | 26.9 | 20-60 | 35.0 | **14.5** |
| boreal_bl_dec | 30.0 | 46.5 | 20-50 | 30.0 | 34.1 |
| ever_shrub | 27.5 | 28.0 | 15-45 | 9.7 | 7.5 |
| c4_grass | 15.0 | 15.4 | 8-30 | 15.7 | 10.2 |
| c3_grass | 25.0 | 18.4 | 15-45 | 18.6 | 15.6 |
| boreal_con_ever / boreal_con_dec / dec_shrub | frozen | | | frozen | |

Sample-level result (unweighted cell means as printed; the loss itself is
flux-weighted):

    fit:      loss 0.7039  GPP term 0.3097  LAI term 0.7883  GPP 0.714 vs 0.787  LAI 1.57 vs 1.05
    held out: loss 0.9289  GPP term 0.4721  LAI term 0.9135  GPP 0.702 vs 0.741  LAI 1.56 vs 1.06

Held-out/fitted ratio 1.32, against 1.20 for v11c -- the weighted loss
generalises slightly less well, as expected when a few high-flux cells carry
more of it.



**What changed against v11c.** With the loss no longer fooled by sparse cells,
`temp_con_ever` went *up* (19.3 -> 32.4) and C4 grass returned to its default
(8.2 -> 15.4) -- v11c had pushed both down to satisfy cell counts rather than
flux. The work of lowering GPP moved onto `specla`: temperate broadleaf
35.0 -> 14.5, tropical deciduous 21.6 -> 11.5. Two PFTs remain pinned at the
vmax floor -- tropical and warm-temperate evergreen -- so the section-9
conclusion stands for those: their excess is not a vmax problem.

**Not yet evidence.** Sample agreement did not survive the full domain last
time. `val_v11d/fit` (50 yr, all cells) is the test; then the 1901-2010
trend check (`trend_v11d/`), then fire and methane re-validation, all
chained.

Operational note: the validation's first launch pointed at
`fit_carbon_v11c/params_fit_v11d.yaml` -- a sed that swapped the filename and
left the directory. `load_physics_params` raised FileNotFoundError rather than
falling back to defaults, and the `Physics parameters:` log line added the day
before confirmed the relaunch. Both guards did their job; the generator now
rewrites the whole `params_file:` line.

### v11d at full domain: worse on magnitude than v11c

`val_v11d/fit`, 50 years from the v11 2010 state, last 5 years:

| quantity | control | fit_v11c (2085) | **fit_v11d (2060)** | observed |
|---|---|---|---|---|
| GPP PgC/yr | 234.1 | 172.8 | **182.1** | 120-130 |
| NPP | 105.4 | 82.3 | 84.8 | 55-60 |
| Rh | 94.5 | 74.7 | 78.6 | 49-58 |
| ET mm/yr | 525.0 | 525.7 | 527.1 | 500-550 |
| veg C PgC | 804.5 | 602.0 | 633.7 | 450-650 |
| soil C kgC/m2 | 13.74 | 12.55 | 12.97 | 10-15 |
| LAI mean | 4.35 | 2.49 | **2.15** | |

| band | LAI/MODIS v11c -> v11d | GPP/FLUXCOM v11c -> v11d |
|---|---|---|
| 30S-30N | 1.67 -> **1.30** | 1.13 -> 1.18 |
| 30-50N | 2.01 -> 1.97 | 1.63 -> 1.77 |
| 50-60N | 1.44 -> 1.39 | 1.53 -> 1.69 |
| 60-70N | 1.13 -> 1.08 | 1.58 -> 1.82 |
| 70-90N | 0.57 -> 0.54 | 0.86 -> 1.07 |

LAI improved everywhere; GPP got **worse in every band**. Global GPP rose
172.8 -> 182.1. The corrected loss did not deliver the magnitude it was built
to deliver.

### Why -- established so far

**Not the sample.** Band-area shares of the 4,331 fitted cells match the
domain to within 1% (0.474 / 0.225 / 0.135 / 0.113 against 0.473 / 0.234 /
0.130 / 0.121); FLUXCOM and MODIS area means within 4%.

**Not the parameters.** Leaf-by-leaf diff of the fit's parameter tree against
the domain run's: two leaves differ, `rhosoi_val` 2636.5 vs 2650 and
`csoi_mineral_val` 880.5 vs 870 -- both ~1%, both carried in from the
`both_fix` yaml used as the template. Nothing that moves GPP 40%.

**The flux-weighted loss did what it was built for.** At the fitted cells in
the domain run, cells above the 90th percentile of observed GPP (37% of the
flux) sit at **0.92x**; cells below the median (11% of the flux) sit at
**2.59x**; median per-cell ratio 1.60x. High-productivity cells are right;
the 182 PgC/yr excess is carried by the many moderate-and-low-flux cells --
and a per-PFT parameter cannot separate a sparse cell from a dense one of the
same PFT. This is a within-PFT, climate-structured pattern, and it is the
same signal as the two PFTs pinned at the `vmax` floor.

**The harnesses disagree by ~25%.** `score()` applied to the domain run at
the fit's own cells gives 0.8734 against the 0.7039 the fit reported; both
the GPP and LAI totals read 1.375x at the domain, where the fit's GPP term
implies ~1.1x. CO2 (373 vs 390 ppm, elasticity 0.63) explains ~3%; the
evaluation window (28 vs 50 yr) ~2% by the v11c trajectory.

**Harness test: they agree.** `one_year` on the fitted cells from the domain
run's own 2060 restart, at its 389.8 ppm, forcing year 2001, against the
domain run's 2001-forcing year: area-weighted GPP **1.3518 vs 1.3519**, median
per-cell ratio 1.000, p10-p90 0.979-1.021. No wiring difference. The two
code paths are the same model.

**Resolved by replay: there was no gap, only an inference error.** Replaying
the fit's exact protocol on its own cells reproduced its logged numbers to the
digit (yr 26-28: GPP 0.714, LAI 1.57). The flux-weighted ratio the fit
actually saw was **GPP 1.297x, LAI 1.295x** -- not the ~1.1x inferred above
by assuming its per-cell term equalled the domain's, and not the 0.91x its
unweighted printout showed. Continuing the same state under the domain
protocol (2001-10, 389.8 ppm) to year 50 gave **1.380x / 1.389x** against the
domain run's 1.375x / 1.377x. The 1.30 -> 1.38 step is CO2 (373 -> 390 ppm,
~2.7%), forcing years (-1.3% for the 2001-03 triplet), and LAI still rising
(1.295 -> 1.389). Fully accounted.

The "~19% unexplained" two paragraphs up was a decomposition by assumption
where a measurement was available. Left in place as a record.

**What the fit actually hit.** It converged with GPP at 1.30x and could not
go lower. Lowering `vmax` helps both loss terms, so the optimiser should have
driven every `vmax` to its floor -- and it did for tropical and warm-temperate
evergreen. It raised the others (`temp_con_ever` 30 -> 32.4, `boreal_bl_dec`
30 -> 46.5, `trop_bl_dec` 65 -> 80.5) because the productive cells of those
PFTs sat at 0.92x and the flux weighting told it to lift them -- which lifted
the sparse cells of the same PFT that were already at 2.6x. **Raising the
productive cells and lowering the sparse cells of one PFT require opposite
moves on the same parameter.** The fit is at a genuine optimum of a
parameterisation that cannot express the answer. That is the finding, and it
is why the next step is a process question -- what makes a sparse cell of a
given PFT 2.6x too productive when a dense one is right -- not another fit.

`--force-years` defaults to 10 regardless; a three-year window is a poor
climate sample even at 1.3%.

Whatever the harness answer, the calibration has reached what `vmax` and
`specla` can do: pools in range, spatial pattern largely fixed, high-flux cells
on FLUXCOM, and a ~1.4x residual living in low-productivity cells that these
parameters do not reach. Where that residual actually lives -- water stress,
sparse-canopy light use, the stomatal coupling that left ET unmoved -- is the
next question, and it is a process question, not a parameter one.

## 11. The 1901-2010 GPP trend on the fitted parameters

Section 8 flagged a risk: IBIS has no nitrogen limitation and
`dlnGPP/dlnCO2 = 0.63`, so parameters fitted to 2010 might be too low for
1901 and the historical trend too steep. `trend_v11d/`: 30 years re-
equilibrating to `params_fit_v11d` at 296 ppm, then 1901-2010 with real
forcing and rising CO2.

| decade | GPP (domain area-wtd, kgC m-2 yr-1) |
|---|---|
| 1901-1910 | 0.909 |
| 1931-1940 | 0.946 |
| 1961-1970 | 0.977 |
| 1981-1990 | 1.032 |
| 2001-2010 | 1.098 |

**+21%** over the century, monotonic. Published: Campbell et al. (2017,
Nature, from COS) +31 +/- 5%; TRENDY ensemble ~20-30%. IBIS sits at the low
end of the range. The unconstrained CO2 response is not too strong, and the
fitted parameters do not produce an implausible 1901 state. The concern in
section 8 is closed.

(The level is still ~1.4x FLUXCOM throughout -- section 10. Trend and level
are separate questions and this settles only the trend.)

## 12. Fire and methane on the calibrated state

Both modules were validated on `both_fix` (FIRE.md 9, METHANE.md 5q). v11d
has no `vmax x0.54`, no soil-OM `x0.67`, a vegetated Arctic, and 634 rather
than 481 PgC of vegetation. Re-run with **no fire or methane parameter
changed** -- the same discipline that made 5q clean.

### Fire (`evaluate_gfed5.py`, 2001-2005, 95,333 cells)

| metric | biased fuel | both_fix | **v11d** | GFED5 |
|---|---|---|---|---|
| burned fraction | 4.23% | 3.59% | **3.10%** | 4.60% |
| rank, all cells | 0.749 | 0.742 | **0.759** | -- |
| rank, burning cells | 0.410 | 0.376 | 0.365 | -- |
| NMSE | 0.715 | 0.754 | 0.814 | -- |
| mean \|lat\|, top 2000 | 11.9 | 10.9 | 10.1 | 10.8 |

Burned area fell with the fuel load and sits **inside the 2.38-4.60%
product spread** (MCD64A1 2.38, GFED4s 2.86, GFED5 4.60). All-cell rank is
the best of any state. Top-2000 latitude is now 0.7 deg too tropical where
both_fix was 0.1 deg off; the Arctic filling in moved fuel poleward less than
it moved GPP. Fire remains a consequence of the carbon state, not a tuned
quantity.

`evaluate_gfed5.report()` now drops non-finite cells: the one NaN cell in v11
states had turned the burned fraction and the all-cell rank into NaN on the
first pass.

### Methane (`evaluate_global.py --config all`, forcing year 2005)

| config | both_fix Tg | **v11d Tg** | both_fix split | **v11d split** (30S-30N / 30-60N / 60-90N) | shape both_fix -> v11d | peak |
|---|---|---|---|---|---|---|
| GMB target | 158 | | 70 / 21 / 8 | | | 7-8 |
| global | 271.1 | 307.0 | 92.5 / 5.4 / 0.4 | 84.6 / 9.2 / 4.6 | 45.7 -> 29.8 | 3 -> 7 |
| biome | 328.3 | 375.6 | 93.6 / 3.9 / 0.4 | 84.7 / 8.3 / 5.2 | 48.2 -> 30.2 | 3 -> 5 |
| biome_q10 | 412.8 | 449.0 | 97.0 / 1.6 / 0.1 | 93.9 / 3.2 / 1.7 | 54.3 -> 47.9 | 3 -> 3 |
| **biome_noq10** | 283.1 | 373.2 | 87.5 / 8.1 / 1.2 | **68.1 / 17.0 / 12.4** | 37.3 -> **10.3** | 3 -> 6 |
| biome_tref | **182.0** | 250.9 | 72.7 / 19.8 / **1.0** | 53.9 / 30.6 / **10.3** | **11.0** -> 28.0 | 6 -> **7** |

**The high-latitude deficit was never structural.** METHANE.md 5q put the
1.0% Arctic share (target 8%) down to missing cryoturbation, vertical
resolution and deep carbon. On v11d, with no methane parameter changed,
`biome_tref` gives **10.3%** and every configuration's Arctic share rose
4-12x. The deficit was `vmax x0.54` starving the Arctic of carbon input
(section 4). The methane module was reporting the carbon state faithfully.

**Magnitude is the Rh ratio.** 182.0 -> 250.9 Tg is x1.38; Rh 57.9 -> 78.6
PgC/yr is x1.36. Linear, as 5p predicted. The methane excess IS the
section-10 carbon-flux excess. Fitting `r_CH4` to close it would be the
double-count 5m refused; nothing to fit here.

**The ranking inverted again.** `biome_tref` won on both_fix; on v11d
`biome_noq10` -- Q10 = 1, temperature entering only through Rh -- lands on
the latitudinal target (shape 10.3), while `biome_tref` overshoots the Arctic
(10.3 vs 8) and undershoots the tropics (53.9 vs 70) but keeps the right
seasonal peak (7). Second time a configuration has been ranked on a biased
carbon state and re-ranked on a corrected one (5q was the first). The
production choice should now be made on FLUXNET-CH4 towers (M3; 108 sites,
31 north of 60N), not on a third global re-rank.

`evaluate_global.py` now masks the NaN cell in tref weighting and flux sums.

## 13. Criterion 3: the seasonal cycle, scored for the first time

The carbon recalibration now targets global total, spatial pattern and
seasonal pattern. Every FLUXCOM comparison before this used the annual mean.
`seasonal_baseline.py` on a 10-year monthly-output run of the v11d state
(`val_v11d/monthly`), against FLUXCOM's 1980-2013 monthly climatology,
flux-weighted, 51,353 cells:

| | shape err | peak offset | \|off\| >= 2 mo | amplitude model/obs |
|---|---|---|---|---|
| ALL | 0.0165 | +0.59 mo | 34% | **0.80** |
| 30S-30N | 0.0071 | +0.75 | 65% (flat cycles; peak ill-defined) | 0.92 |
| 30-50N | 0.0405 | -0.14 | 28% | **0.62** |
| 50-60N | 0.0434 | +0.58 | 10% | **0.62** |
| 60-70N | 0.0524 | +0.57 | 2% | 0.66 |
| 70-90N | 0.0835 | +0.15 | 0% | 0.68 |

Amplitude is max/mean of the 12-month cycle. **North of 30N the model's
cycle is too flat by a third and mildly late.**

By the section-10 annual-ratio class:

| class | n | shape err | amplitude |
|---|---|---|---|
| ratio < 0.8 | 5,340 | 0.0091 | 1.03 |
| 0.8-1.25 (well fitted) | 13,861 | 0.0091 | 0.88 |
| 1.25-2 | 16,570 | 0.0253 | 0.70 |
| **>= 2 (sparse excess)** | 15,582 | **0.0364** | **0.66** |

### Where in the year the sparse excess sits

Model/obs by month, cells with annual ratio >= 2 against well-fitted cells:

| | J | F | M | A | M | J | J | A | S | O | N | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ratio >= 2 | 2.93 | 2.92 | 3.32 | 3.41 | 2.77 | **2.08** | **1.95** | 2.37 | 3.65 | **4.79** | **4.46** | 3.35 |
| < 1.25 | 0.96 | 0.92 | 0.93 | 0.97 | 0.95 | 0.88 | 0.85 | 0.88 | 0.97 | 1.01 | 1.03 | 1.01 |

**The excess is smallest at the growing-season peak (~2x) and largest in
autumn (4.5-4.8x) and spring (3.3-3.4x).** Sparse cells keep photosynthesising
after FLUXCOM says the season is over. That is a phenology signature, not a
capacity one -- capacity would inflate every month alike -- and it is the
first thing in this calibration that distinguishes a sparse cell from a dense
one of the same PFT (section 10's unreachable residual).

### What it implicates, and what will be freed

Sparse cells are grass- and shrub-dominated, so the lower-canopy phenology
governs them. In `pheno()` (now `PhenoParams`, section 12 of this doc /
commit 7fd608b): grasses drop leaves only when the 10-day mean temperature
falls below **0 C** (`tthresh_l_c`); their GDD accumulator resets only below
**-5 C** (`gthresh_l_c`), so in mild climates it never resets and `templ`
sits at 1 all year (F77-faithful, vegetation.f:100-105); the drought trigger
fires only when 10-day net assimilation goes **negative**, and even then
retains **10%** leaf area (`drop_floor`) with a 15-day e-folding (`ddays`).
Each of those keeps sparse canopies active into autumn. For the flattened
forest bands: the tree cold offset (`tthresh_u_offset`, leaves drop below
tc + 5) and budburst GDD (`gdd_thresh_u`).

Stage (a) of the re-fit frees exactly these -- `stressfac`, `tthresh_l_c`,
`gthresh_l_c`, `gdd_thresh_l`, `drop_floor`, `ddays`, `tthresh_u_offset`,
`gdd_thresh_u` -- with `vmax`/`specla` **held at v11d**, so the question
"how much of the spatial and seasonal error is phenology and stress, given
the v11d capacity" gets a clean answer before a joint refinement.

`--w-season` is raised to 10: shape errors are O(0.02-0.04) against a GPP
term of O(0.3-0.4), and at the earlier 2.0 the third criterion would have
contributed ~5% of the loss.

Sample check: `seasonal_baseline_sample.py` on the fit's 4,331 cells via
`one_year` reproduces the full-domain table almost to the digit (ALL shape
0.0170 vs 0.0165, 30-50N amplitude 0.62 vs 0.62, sparse-cell October ratio
4.9 vs 4.8). That validates the harness's monthly path -- which stage (a)'s
seasonal term scores -- and shows the sample is representative for
seasonality as well as for annual pattern.

Also noted: even well-fitted cells run 0.85-0.88x in June-August -- the peak
is slightly low everywhere. That is a capacity-at-peak question (temperature
response of Vmax, or light), separate from the shoulder-season excess, and is
left for the joint stage.

## 14. Stage (a) cannot reach the autumn excess: the levers have no effect

Stage (a) ran 40 evaluations with the season term flat at 0.0172-0.0178 in
every new best while LAI fell 0.794 -> 0.658 and GPP 0.300 -> 0.283. Rather
than wait for 120, the levers were pushed to their bound extremes directly on
60 sparse-excess cells (annual ratio >= 2, |lat| < 66, mean 41N, mean ratio
3.6x), four years each, monthly model/FLUXCOM ratio:

| lever at extreme | annual | Oct | Nov |
|---|---|---|---|
| default (v11d) | 2.37 | 4.22 | 4.42 |
| `tthresh_l_c` +8, `gthresh_l_c` +2 | 2.36 | 4.22 | 4.42 |
| `gdd_thresh_l` 300 | 2.37 | 4.22 | 4.42 |
| `drop_floor` 0.02, `ddays` 5 | 2.37 | 4.22 | 4.42 |
| `tthresh_u_offset` 12, `gdd_thresh_u` 250 | 2.25 | 4.09 | 4.27 |
| `stressfac` -2.5 (max stress) | 2.26 | 4.01 | 4.17 |
| `stressfac` -10 (min stress) | 2.44 | 4.36 | 4.60 |
| all phenology at extremes | 2.22 | 4.08 | 4.25 |

**The lower-canopy phenology thresholds have zero effect** -- identical to
two decimals -- and the drought floor and e-folding have zero effect, which
means the drought trigger (`a10ancls < 0`) never fires in these cells. Tree
cold and `stressfac` each move ~5%. Everything together takes the autumn
excess from 4.4x to 4.25x.

So the season term was flat because nothing in the vector controls the
process. Section 13 read the autumn signature as "phenology" and implicated
the grass thresholds; that inference was wrong about *which* mechanism. The
excess sits in something these cells' GPP does not pass through -- the next
diagnostic asks which PFT carries their GPP and why it has no autumn
cut-off. The leading suspect is evergreen shrub (PFT 8): the sample's
second-largest dominant class (735 cells), no cold phenology by definition,
and a drought trigger that evidently never fires.

The fit runs to completion regardless: its LAI reduction is real progress on
criteria 1-2 and its parameters are written only at the end. Its result is
not evidence about criterion 3.

### Which mechanism -- the composition and trigger diagnostic

Same 60 cells, v11d 2060 state, one year of 2001 forcing, read month by
month.

**Who holds the leaf area.** A mix, not one PFT: boreal broadleaf deciduous
22% of LAI, C4 grass 17% (dominant in 16 cells), tropical deciduous 16.5%,
evergreen shrub 15.5% (dominant in 17), C3 grass 9%, warm-temperate
evergreen 8%. Total LAI **2.85** in cells FLUXCOM puts at ~0.4 kgC m-2 yr-1.
The set spans 46S-66N, 47 NH / 13 SH: hot cells carry the C4 grass and
tropical deciduous, cold cells the boreal deciduous and shrub -- which is
why the *grass cold* thresholds could not act (they bite only where it is
cold, and there the grass is minor).

**The drought trigger never fires.** `a10ancls` and `a10ancl3` (10-day mean
net assimilation, shrubs and C3 grass) are positive in every cell in every
month; `dropls` and `dropl3` sit at 1.000 all year. A leafed canopy in
daylight essentially never has ten days of negative net assimilation, so
IBIS 2.6b4's drought phenology -- the only phenology that applies to C4
grass, tropical deciduous, deciduous shrub -- is unreachable in practice.
That is why `drop_floor` and `ddays` had zero effect.

**The model sees the dry season; the canopy ignores it.**

| mon | stresstl | frac tl<0.5 | wsoi L1 | prcp mm | GPP model | GPP obs | ratio |
|---|---|---|---|---|---|---|---|
| Jan | 0.315 | 0.72 | 0.266 | 32 | 0.047 | 0.016 | 2.98 |
| Apr | 0.546 | 0.43 | 0.393 | 48 | 0.089 | 0.023 | 3.84 |
| Jul | **0.702** | 0.28 | 0.461 | 77 | 0.116 | 0.073 | **1.59** |
| Oct | 0.437 | 0.60 | 0.297 | 38 | 0.082 | 0.017 | 4.81 |
| Nov | 0.405 | 0.60 | 0.240 | 36 | 0.065 | 0.012 | **5.34** |
| Dec | **0.285** | 0.75 | 0.193 | 24 | 0.056 | 0.014 | 4.16 |

Soil-water stress (drystress factor, 1 = unstressed) falls to **0.29-0.34**
in Nov-Feb with 60-75% of cells below 0.5, and top-layer soil water halves --
exactly the months of the largest GPP excess. Yet GPP in those months runs at
half its summer value. Stress reaches stomata (and `stressfac` at its
extreme throttled 5%), but nothing turns it into leaf loss: **IBIS has no
functioning dry-season dormancy.** That one gap accounts for the sparse-cell
annual excess (section 10), the 3-4x winter GPP, and the 0.62x seasonal
amplitude at mid-latitudes, together.

**The fix is a trigger, not a parameter.** `PhenoParams.drop_stress_thresh`:
the drought-deciduous PFTs also drop leaves while their canopy's soil-water
stress is below the threshold (`soil.stresstu` for the upper canopy,
`soil.stresstl` for the lower), regrowing when it recovers, through the
existing `ddfac` / `drop_floor` machinery. Default -1.0 is unreachable, so
default behaviour is bit-identical and F77-faithful; evergreen shrub keeps no
phenology, as in F77. Its effect on these cells is measured below.

### Lever test 2: the stress trigger fires, and it does not matter either

Same 60 cells, four years, `drop_stress_thresh` at three values (the stress
table above says 0.7 should trigger in most cell-months October-April):

| case | annual | Oct | Nov | Dec |
|---|---|---|---|---|
| default (inert) | 2.37 | 4.22 | 4.42 | 3.26 |
| threshold 0.3 | 2.36 | 4.19 | 4.39 | 3.24 |
| threshold 0.5 | 2.34 | 4.16 | 4.38 | 3.21 |
| threshold 0.7 | 2.29 | 4.06 | 4.24 | 3.09 |
| 0.7 + drop_floor 0.02, ddays 10 | 2.34 | 4.16 | 4.50 | 3.22 |

Dropping the leaves of **every drought-deciduous PFT in the mix** -- C4 and
C3 grass, tropical deciduous, deciduous shrub, ~43% of the leaf area -- takes
3% off the annual GPP of these cells. So those PFTs do not carry the excess.
It must sit in the PFTs no trigger touches: evergreen shrub (no phenology by
construction), the evergreens, and boreal broadleaf deciduous -- 22% of LAI,
the largest share -- whose `tempu` stayed at 0.5-0.66 through winter because
its cold threshold is `tc + 5`, relative to the local coldest month rather
than absolute. A canopy-type decomposition of assimilation is running to
name the carrier. The trigger stays in the code: it is correct, inert by
default, and will matter wherever the excess is drought-deciduous -- just not
here.

### Canopy-type decomposition: it is a regime, not a PFT

Same 60 cells, month by month, share of assimilation by canopy type and the
displayed leaf area behind it:

| | upper broadleaf | upper conifer | shrub | C3 | C4 |
|---|---|---|---|---|---|
| share of assimilation, all months | 46-51% | 0% | 15-24% | 5-6% | 22-32% |
| displayed LAI, Jan / Jul | 1.37 / 1.41 | 0.14 / 0.14 | 0.55 / 0.55 | 0.72 / 0.74 (grass) | |
| phenology state, Jan / Jul | tempu 0.56 / 0.63 | | dropls 1.00 / 1.00 | templ 0.64 / 0.73 | dropl4 1.00 / 1.00 |

**Nothing in the canopy has a seasonal cycle.** That is the 0.62x amplitude
directly. The cold-deciduous trees are perpetually half-on -- `tempu` 0.50-0.66
in every month, July included -- because their drop threshold is `tc + 5`,
relative to the local coldest month: in these mild-winter cells it sits
inside the winter temperature range, and the GDD accumulator never resets
(0 C is never reached) so budburst is instantaneous whenever the 10-day mean
ticks back above it. Half-leafed in winter, half-leafed in summer. Grass is
the same story one layer down (`templ` 0.60-0.82). Evergreen shrub is flat
by construction.

**With the stress trigger at 0.7 the trigger fires** -- `dropl4` 0.29-0.67,
grass LAI 0.72 -> 0.17-0.41, tropical-deciduous `dropu` 0.26-0.66, ~0.75 of
2.85 LAI removed in the dry months -- **and GPP falls 10%** (Jan 0.047 ->
0.042). The remaining canopy absorbs the light. This is section 5 at the
process level: at `dlnGPP/dlnLAI ~ 0.09` a canopy of LAI 2-3 is optically
saturated, and no phenology can make GPP seasonal until leaf area is low
enough for GPP to feel it.

**So the three criteria are one problem.** These cells carry LAI 2.85 where
the land is sparse; above roughly LAI 1.5 the model's light interception
makes GPP indifferent to leaves; the year-round GPP feeds the NPP that
sustains the LAI. Criterion 3 (seasonal) cannot be met until criterion 2's
leaf area is fixed, and leaf area cannot be fixed by phenology while GPP is
saturated. The lever has to act on the saturation itself -- how much of the
canopy's light the leaves (and stems) intercept in sparse vegetation -- or
on the leaf-area attractor below it. Quantified next.

### The regime, quantified

Same 60 cells:

* **Leaf area:** model `plai` 2.85 against MODIS annual-max **1.11** (2.4x,
  median). MODIS puts 17% of these cells below 0.5, 23% at 0.5-1, 55% at 1-2,
  5% above 2 -- moderate vegetation, not desert. The model is over by the
  global LAI factor here, not by an extreme one.
* **Stems are a minor interceptor:** SAI upper 0.39 (0.25 x LAI), lower 0.06
  (0.05 x LAI). Optical load upper 2.11, lower 1.35. The stem hypothesis is
  out.
* **Canopy structure:** upper canopy LAI 1.72 over 59% of the cell (`fu`),
  lower 1.28 over 81% (`fl`).
* **Elasticity dlnGPP/dlnLAI between the default and the stress-trigger
  run:** 0.35 in Jan-Feb, 0.19-0.24 in Mar-Apr, 0.05-0.14 in May-Oct, 0.28
  in Nov. Light-limited when the sun is low, saturated when high. Section 5's
  global 0.09 is the growing-season value.

The trigger cut the **lower** canopy's optical load 1.44 -> ~0.9 and GPP fell
10% -- partly saturation, but mostly because it never touched the **upper
broadleaf canopy: half of all assimilation, LAI 1.72, displayed 1.37-1.41 in
every month.** That canopy is evergreen broadleaf plus cold-deciduous trees
held at half-leaf by a relative threshold. The lever the process needs is an
absolute leaf-drop temperature for the upper canopy: real temperate
deciduous senescence sits near a 10-day mean of 5-10 C wherever the tree
grows, not at "local coldest month + 5". `PhenoParams.tthresh_u_abs` adds it
(threshold = max(relative, absolute); default -100 C never binds). Tested
next on the same cells, alone and with the stress trigger and a GDD reset
that actually resets in mild winters.

### Lever test 3: an absolute leaf-drop temperature fixes spring, not autumn

| case | annual | Apr | Oct | Nov |
|---|---|---|---|---|
| default | 2.37 | 2.96 | 4.22 | 4.42 |
| `tthresh_u_abs` 5 C | 2.31 | 2.71 | 4.19 | 4.40 |
| `tthresh_u_abs` 8 C | 2.23 | 2.57 | 4.18 | 4.37 |
| + `gthresh_u_c` 5 (GDD resets) | 2.23 | 2.57 | 4.18 | 4.37 |
| + stress trigger 0.7 | 2.14 | 2.51 | 4.02 | 4.18 |
| + grass threshold 8 C | 2.12 | 2.44 | 4.02 | 4.18 |

Trees leaf out later, so spring falls (April 2.96 -> 2.44); autumn does not
move (October-November 4.0-4.4x under every combination). The 10-day mean
in these cells is 8-9 C in October, so an 8 C threshold barely binds until
late November, and what remains is evergreen -- shrub, broadleaf evergreen,
conifer -- plus a canopy amount that is 2.4x reality.

**Six lever tests agree: phenology, however parameterised, moves these cells
<= 10% annually and <= 5% in autumn.** The excess is not seasonality of the
leaves; it is how much canopy the model paints on a sparse cell. IBIS sets
lower-canopy cover `fl = clip(totlail, 0.25, 0.975)` -- 1.28 of grass LAI
becomes 81-97% cover -- and `fu` from wood biomass reads 0.59 here. GPP
scales linearly with cover, which light saturation cannot defeat, and a
cover that saturates with LAI would leave dense cells untouched while
halving sparse ones: the within-PFT separation section 10 said no per-PFT
parameter could give. Tested next.

`tthresh_u_abs` stays (correct, inert by default, and it is the right fix for
spring leaf-out where it matters); the pooled monthly tables above mix 47 NH
with 13 SH cells and are split by hemisphere from here.

### Lever test 4: cover is a real lever; the hemisphere split; the upper canopy remains

Split 47 NH / 13 SH (the pooled tables above mixed SH winter into NH autumn):

| default (F77 cover) | J | F | M | A | M | J | J | A | S | O | N | D | annual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NH | 6.5 | 5.2 | 3.9 | 3.0 | 2.1 | **1.44** | **1.31** | 1.7 | 2.8 | 4.6 | 6.4 | 7.2 | 2.31 |
| SH | 1.4 | 1.5 | 1.9 | 2.9 | 5.0 | **10.6** | **10.9** | 6.3 | 3.8 | 3.2 | 2.6 | 1.8 | 2.56 |

Both hemispheres: growing season 1.3-1.8x, dormant season 5-11x. **The
model has almost no dormant season in these cells.** (Dormant-season ratios
sit on FLUXCOM denominators of ~0.01 kgC m-2 mo-1; the robust targets are the
annual excess and the flat amplitude, not the 7x itself.)

`PFTParams.cover_k` (fl = 1 - exp(-k x totlail); 0 keeps F77's fl = totlail),
`fl_min`, `fu_min` (F77 floors 0.25). Inert defaults, bit-identical, guarded,
both update paths.

| case | fl | fu | plai | NH annual | NH Jul | NH Dec |
|---|---|---|---|---|---|---|
| default | 0.69 | 0.60 | 2.11 | 2.31 | 1.31 | 7.15 |
| cover_k 0.5 | 0.31 | 0.60 | 1.80 | 1.99 | 1.07 | 6.62 |
| cover_k 0.7 | 0.40 | 0.60 | 1.90 | 2.10 | 1.15 | 6.84 |
| cover_k 0.5 + floors 0.05 | 0.23 | 0.55 | 1.78 | 1.98 | 1.06 | 6.60 |
| + tthresh_u_abs 8 + stress 0.7 | 0.23 | 0.55 | 1.59 | **1.76** | **0.96** | 6.23 |

Cover acts as predicted: lower-canopy cover 0.69 -> 0.31, `plai` follows
(2.11 -> 1.80, the feedback runs the right way), NH annual -14%. With every
phenology lever added the NH growing season lands at **0.96-0.98** -- right --
and the annual excess is 1.76x. What remains is the dormant season, and its
carrier is unchanged: **`fu` did not move** (0.60 -> 0.55). Upper-canopy cover
comes from wood, `fu = (1 - exp(-wood)) / (1 - exp(-woodnorm))`, which puts
trees carrying 0.9 kgC m-2 of wood over 60% of a sparse cell where real tree
cover is 5-20%. That canopy carries half the assimilation and none of the
levers so far reach it. A wood-to-cover scale for `fu` is the next lever, and
it must be tested on dense cells as well as sparse ones.

### Lever test 5: cover on sparse AND dense cells -- arrangement cancels

`PFTParams.fu_k` (fu = (1 - exp(-wood/k)) / (1 - exp(-woodnorm/k)); 1.0 =
F77). Inert default, bit-identical, guarded. Tested on the 60 sparse cells
and on 60 dense well-fitted cells (annual ratio 0.8-1.25, top-quartile
FLUXCOM GPP; mean lat 12.6 -- tropical forest):

| case | sparse: fl / fu / plai | NH annual | dense: fl / fu / plai | annual |
|---|---|---|---|---|
| default (F77) | 0.69 / 0.60 / 2.11 | 2.31 | 0.28 / 0.95 / 3.23 | 0.98 |
| fu_k 3 | 0.70 / 0.52 / 2.09 | 2.27 | 0.28 / 0.89 / 3.19 | 0.96 |
| fu_k 6 | 0.70 / 0.49 / 2.08 | 2.25 | 0.28 / 0.84 / 3.15 | 0.93 |
| cover_k 0.5 + fu_k 3 | 0.31 / 0.52 / 1.77 | 1.94 | 0.26 / 0.89 / 3.16 | 0.95 |
| + floors 0.05 | 0.24 / 0.43 / 1.74 | 1.92 | 0.06 / 0.89 / 3.14 | 0.94 |
| + tthresh_u_abs 8 + stress 0.7 | 0.24 / 0.43 / 1.57 | **1.72** | 0.06 / 0.89 / 3.05 | **0.91** |

**Upper-canopy cover is a weak lever (-3%)**, and the reason corrects an
earlier claim in this section: cover is not linear in GPP. When cover falls
the same leaves crowd into less ground (`lai = avglai / fu`), and in an
unsaturated canopy GPP ~ cover x f(LAI/cover) ~ total leaf area -- the
arrangement cancels. Cover bites only where it crowds leaves into
saturation, which is what the lower canopy did (-14%, with `plai` falling as
feedback). Dense cells lose 2-7% under the same levers, so they are not free.

**Every structural lever tested acts on leaf display or arrangement. None
acts on the amount of leaf area the vegetation model grows**, and in the
unsaturated regime GPP tracks that amount. The sparse cells carry 2.4x MODIS
leaf area; the model's GPP follows. The amount is set by the allocation
attractor (LAI = NPP x aleaf x tauleaf x specla) and is per-PFT -- and the
stage-(a) fit already shows the within-PFT trade-off with names: it *raised*
`vmax` for tropical deciduous (65 -> 80.5) and boreal deciduous (30 -> 46.5)
because their dense cells read 0.92x, worsening the sparse cells of the same
PFTs.

Before designing around that: 80% of the annual excess is off-season GPP
judged against FLUXCOM values of ~0.013 kgC m-2 mo-1 -- a machine-learning
product at its noise floor (the clipped negative winters are that noise),
known to be weak in dry systems. The semi-arid FLUXNET2015 towers test that
target directly; result below.

### The off-season target is real: 19 semi-arid towers

FLUXNET2015 monthly `GPP_NT_VUT_REF` climatologies at 19 semi-arid sites,
against FLUXCOM and the v11d model at the same 0.5-degree cells (off-season
= the 8 lowest tower months):

| | annual kgC m-2 | off-season kgC m-2 mo-1 |
|---|---|---|
| tower (point, productive patch) | 0.52 | 0.025 |
| FLUXCOM (cell) | 0.46  (0.87x tower) | 0.022  (**0.88x tower**) |
| model (cell) | 0.94  (**1.81x tower**) | 0.055  (**2.19x tower**) |

FLUXCOM's near-zero dormant season is what the towers measure; the 0.87-0.88
is the usual point-vs-cell representativeness, not a data floor. The model
exceeds the *towers* -- which normally read above the cell mean -- by 1.8x
annually and 2.2x in the dormant season. Worst where the land is sparsest:
ES-LJu shrubland 9.2x (23.9x off-season), US-Cop grassland 7.2x, US-SRC
shrubland 4.2x, US-Whs 3.4x, CN-Du2 steppe 3.2x. A few are right or low
(AU-DaS 1.02, AU-Cpr 0.73, SD-Dem 0.50). **The sparse-cell excess is
confirmed on independent site data; FLUXCOM is not the problem.**

The next check has ground truth: US-Whs, ES-LJu, US-Cop are open shrub and
grass with no trees. If the model carries an upper canopy there (`fu` 0.60,
half of assimilation in the sparse set), half the excess is a tree layer
that should not exist -- an establishment error in `dynaveg`, which knows
no water limit on where trees grow.

### Ground truth at the tower cells: phantom trees and an over-painted understory

Model canopy cover against the MODIS-derived CLM surfdata fractions at the
same 0.5-degree cells (20 semi-arid towers):

| site | class | MODIS tree / shrub+grass / bare | model fu / fl | upper-canopy PFT holding LAI |
|---|---|---|---|---|
| US-SRM/SRC/SRG (AZ) | WSA/OSH/GRA | 0.08 / 0.61 / 0.26 | **0.25** / **0.98** | none (fu at floor) |
| US-Whs (AZ) | OSH | 0.00 / 0.62 / 0.34 | 0.25 / 0.86 | none |
| US-Cop (UT) | GRA | 0.00 / 0.51 / 0.41 | **0.62** / 0.98 | warm-temp evergreen 0.52 |
| ES-LJu (Spain) | OSH | **0.03** / 0.48 / 0.34 | **0.84** / 0.98 | **tropical BL deciduous 0.90** |
| IT-Noe (Italy) | CSH | 0.14 / 0.52 / 0.03 | **0.97** / 0.98 | **tropical BL deciduous 2.34** |
| US-Ton, US-Var (CA) | WSA/GRA | 0.15 / 0.59 / 0.05 | **0.92** / 0.98 | **tropical BL deciduous 2.47** |
| CN-Du2 (Mongolia) | GRA | **0.00** / 0.55 / 0.34 | **0.92** / 0.98 | **temperate conifer 1.10** |
| CN-Cng, CN-Sw2 | GRA | 0.01 / 0.4 / 0.3-0.6 | 0.62-0.64 / 0.98 | little leaf, 1 kgC wood |
| US-Me6 (OR) | ENF | 0.90 / 0.04 / 0.06 | 0.98 / 0.25 | evergreen 3.9 (right) |
| **mean of 20** | | **tree 0.13 / shrub+grass 0.50** | **fu 0.57 / fl 0.89** | |

Two failure modes, both structural, both F77-faithful:

**Phantom trees.** In Mediterranean and steppe cells `dynaveg` establishes a
dense tree canopy where there is none -- and the PFT is wrong: **tropical
broadleaf deciduous at 37-41N** (Spain, Italy, California), **temperate
conifer in Mongolian steppe**. The tropical PFTs' only cold limit is
`TminL = 0 C`: wherever the coldest month averages above freezing they are
allowed, and with the highest `vmax` they win the NPP-share competition.
Neither `existence()` nor `dynaveg` asks whether there is water for a tree.
(The stage-(a) fit raised trBLd `vmax` 65 -> 80.5 for its real savanna cells
and thereby fed the phantom Mediterranean trees -- the within-PFT trade-off
of section 10, now with a map.) The upper canopy carries half of sparse-cell
assimilation with `tempu` dithering at 0.5-0.66, so this is the largest
single component of the off-season excess.

**Over-painted understory.** Where the model rightly has no trees (Arizona,
Australia, Sudan) `fu` sits at its 0.25 floor with ~0.03 leaf area -- harmless
-- but grass/shrub cover is painted at **0.98** where MODIS says 0.3-0.6 with
0.2-0.5 bare ground: F77's `fl = totlail`. `cover_k` (above) is the lever for
this half.

What a tree-establishment fix could deliver is bounded next by removing the
upper canopy outright in the 60 sparse cells.

### The global extent: 80% of the excess is in open-canopy land; forests are right

v11d state, all cells with FLUXCOM, stratified by MODIS/CLM tree cover:

| MODIS/CLM tree cover | cells | area Mkm2 | model/FLUXCOM | excess PgC | share of global excess | model fu |
|---|---|---|---|---|---|---|
| < 0.10 **and model fu >= 0.5, upper LAI >= 0.5 (phantom trees)** | 6,001 | 14.7 | **2.02x** | 10.7 | **27%** | 0.90 |
| < 0.10 (all treeless) | 19,138 | 40.7 | 1.95x | 16.6 | 42% | 0.48 |
| 0.10-0.40 (savanna, woodland) | 12,859 | 29.4 | 1.54x | 14.9 | 38% | 0.71 |
| >= 0.40 (forest) | 18,807 | 43.7 | **1.11x** | 7.9 | 20% | 0.85 |
| **all** | 50,804 | 128.5 | **1.34x** | **39.4** | | |

Phantom-tree cells by band: 30-50N **3,177**, 0-30N 732, 0-30S 967, 50-70N
640, 30-60S 469. Held by tropical broadleaf deciduous (2,756), warm-temperate
broadleaf evergreen (1,778), temperate conifer (652), boreal deciduous (414).

**Forests are nearly right (1.11x). Eighty percent of the global excess is
in cells with under 40% tree cover, and phantom trees alone are 27%.** The
1.34x is not a photosynthesis-parameter problem: where the vegetation
structure is right, the canopy physiology is close. It is a **vegetation-
structure problem in open systems** -- trees established with no water
limit, understory painted at 98% cover, no dormancy -- and each of those is
a few lines in `dynaveg` / `existence` / `pheno`, not a module.

This also explains why every parameter fit converged where it did: `vmax`
and `specla` are per PFT, and the same PFT is right in its forest and wrong
in the open cells it should not occupy. No parameter can express "no tree
here"; the structure has to.

### The ceiling: sparse cells with no upper canopy

Same 60 cells, trees removed and prevented from re-establishing, years 3-5:

| | fu | fl | upper LAI | lower LAI | NH annual | NH Jul | NH Dec | SH annual |
|---|---|---|---|---|---|---|---|---|
| default (v11d) | 0.60 | 0.67 | 1.14 | 0.90 | 2.30 | 1.30 | 7.19 | 2.58 |
| **no upper canopy** | 0.25 | 0.91 | 0.00 | **1.78** | **1.47** | **0.89** | **3.98** | **1.92** |

Removing the trees takes NH annual **-36%** and puts the NH growing season at
0.89-0.95 -- even though the understory compensates (grass LAI doubles with
the light freed, `fl` 0.67 -> 0.91). The winter that remains (3.6-4.0x) is the
over-painted, non-dormant understory: `cover_k` (-14%) and the stress
trigger act on that half. Together the two structural fixes plausibly bring
open-canopy cells near 1.2x while forests (1.11x) are left alone.

### Stage (b), redesigned

Not another parameter fit on the same structure. Two structural changes,
each inert by default and each already measured on the 60 cells:

1. **Water-limited tree establishment.** A survival factor on upper-canopy
   PFTs from annual-mean soil-water stress (`aystressu`, new accumulator),
   with no seeding where trees cannot survive. Calibrated directly against
   the MODIS/CLM tree fraction at the tower cells (fu 0.57 -> ~0.13) and
   validated globally on the same field -- a fourth criterion the carbon
   data alone could not give.
2. **Understory cover and dormancy.** `cover_k`, `fl_min`,
   `drop_stress_thresh`, `tthresh_u_abs` -- already built.

Then a joint fit of the structural scalars with `vmax`/`specla` on the
three-criterion loss. The stage-(a) result (running) is the parameter-only
baseline that fit is judged against.

### Built: the water-limited tree establishment gate (commit f98e9c5)

`StressAccum` (annual sums of `soil.stresstu` / `stresstl` and a day count)
hung off `IBISState` as an Optional declared after every other field, so a
state without it keeps its 258 leaves and every restart on disk still
loads; with it, `load_state` fills the three trailing leaves from the
template and says so. Accumulated daily inside the scan body, reset with
the annual accumulators. `dynaveg(tree_surv=...)`: survival =
clip((annual-mean upper-canopy stress - `tree_stress_min`) /
`tree_stress_ramp`, 0, 1) multiplies the upper-canopy pools and blocks
seeding where survival is 0. `tree_stress_min = -1` (default) never binds.

Proofs: bit-identical without the accumulator; the v11 2010 checkpoint loads
into a with-accumulator template; 14 guards pass (exact +3 leaves, threading
of `tree_surv` from the callers into `dynaveg`).

Build note, because it is a hygiene finding: the first patch used a
multi-line non-greedy regex for the `dynaveg` signature, which ends
`) -> tuple[...]:`; the match ran into the body and rewrote
`for j in range(npft):` as `range(npft, tree_surv=None)` -- valid Python,
broken semantics, caught only by the behavioural probe. Exact-string
anchors from here.

### The tree fraction, globally: the excess sits where the phantom trees are

`fu` from the v11d validation (years 2056-60) against the CLM surfdata tree
fraction (PFTs 1-8 of `PCT_NAT_PFT`; the no-crop `all_forests` variant, so a
potential-vegetation field like IBIS's own), on the 65,698 cells with >= 70 %
natural vegetation -- 89 % of land area, the rest ice and lakes.
Area-weighted (`verify/calibration/fu_vs_tree.py`):

| CLM tree fraction | share of natural land | fu (model) | tree (CLM) | GPP mod/obs | share of global excess |
|---|---|---|---|---|---|
| < 0.05 | 37 % | 0.40 | 0.01 | **2.01** | 28 % |
| 0.05-0.15 | 11 % | 0.62 | 0.09 | 1.77 | 16 % |
| 0.15-0.30 | 11 % | 0.75 | 0.22 | 1.54 | 18 % |
| 0.30-0.50 | 10 % | 0.81 | 0.39 | 1.44 | 14 % |
| 0.50-0.70 | 9 % | 0.83 | 0.59 | 1.33 | 11 % |
| > 0.70 | 23 % | 0.92 | 0.91 | **1.02** | 12 % |

Overall `fu` 0.66 against 0.34 (RMSE 0.46, r 0.58). The GPP ratio falls
monotonically with the observed tree fraction, from 2.0x on treeless land to
1.02x in forests, and the cover error falls with it. The phantom cells
(tree < 0.15, `fu` > 0.40) are 16 % of natural land at `fu` 0.84 against
0.04, run 1.94x and hold **31 % of the global excess**; the converse
(tree > 0.5, `fu` < 0.30) is 2 % of land and 2 % of the excess. By latitude
the worst bands are 30-45N (`fu` 0.65 vs 0.18, 1.83x, 23 % of the excess),
45-60N (0.76 vs 0.45, 1.64x, 18 %) and 15-30S (0.60 vs 0.25, 1.69x, 17 %);
the tropics agree on both fields (1.01-1.08x). The 0.25 floor on `fu` is
visible in the first row: treeless land cannot read below it unless
`fu_min` moves.

So the tree fraction enters the stage-(b) loss as a fourth term
(`--w-tree`, `tree_at`): the area-weighted mean |fu - tree| over cells with
natural vegetation, **0.365 at v11d** against a GPP term of 0.3-0.4, so
`--w-tree 1` gives the fourth criterion the weight of the first two. Without it a joint fit could satisfy GPP through `vmax`
instead of through the trees -- the degeneracy that produced phantom trees
with low `vmax` in the first place.

Lever test 6 runs the gate at thresholds 0.45 / 0.55 / 0.65, alone and with
`cover_k` and the stress trigger, on the sparse, dense and tower sets -- the
tower set reporting `fu` against the MODIS/CLM tree fraction cell by cell,
which is the gate's direct calibration target.

Result (years 4-6 of a 6-year cycle; sparse = 60 cells with ratio >= 2,
dense = 60 well-fitted high-flux cells, towers = 20 FLUXNET cells, 16
open/dry and 4 forest):

| `tree_stress_min` | sparse fu | sparse NH annual | NH Jul | NH Dec | dense fu | dense NH / SH | tower fu | phantom tower sites (obs <= 0.15, model > 0.4) | forest sites lost |
|---|---|---|---|---|---|---|---|---|---|
| inert (v11d) | 0.60 | 2.30 | 1.30 | 7.18 | 0.95 | 0.99 / 0.97 | 0.70 | 9 of 16 | 0 |
| 0.45 | 0.42 | 2.13 | 1.21 | 6.50 | 0.86 | 0.99 / 0.95 | 0.43 | 1 (US-Ton 0.94) | 0 |
| 0.55 | 0.39 | 2.06 | 1.18 | 6.30 | 0.76 | 1.00 / 0.94 | 0.40 | 1 (US-Ton 0.44) | 0 (BR-Sa1 0.86) |
| 0.65 | 0.36 | 2.00 | 1.14 | 6.09 | 0.65 | 0.98 / 0.92 | 0.36 | 0 | **1: BR-Sa1 0.25 (obs 0.90)** |
| 0.55 + `cover_k` 0.5 + stress trigger 0.7 | 0.40 | **1.60** | **0.85** | 5.34 | 0.84 | 0.95 / 0.91 | 0.43 | 1 (US-Ton 0.89) | 0 |

Four readings.

1. **The gate separates the towers.** At 0.45 every one of the 15 dry and
   open sites falls to the 0.25 floor -- CN-Cng 0.96 -> 0.25, CN-Du2 0.98,
   ES-LJu 0.97, IT-Noe 0.98, ZA-Kru 0.98, US-Cop 0.79, AU-DaS 0.88 -- while
   US-Me6, US-Ha1, BR-Sa1 and FI-Hyy keep 0.93-0.98. At 0.65 the gate kills
   Santarem (BR-Sa1, a dry-season Amazon forest): the threshold's ceiling.
   The dense tropical cells lose cover from 0.55 up (fu 0.95 -> 0.76) with
   no GPP change, grass filling the light. The safe starting value is
   **0.45**; `tree_stress_ramp` then sets how sharp the edge is.
2. **The sparse-cell excess is only partly trees.** Alone the gate takes NH
   annual 2.30 -> 2.00-2.13 (-7 to -13 %); the understory compensates
   (lower LAI 0.89 -> 1.28, `fl` 0.67 -> 0.79), as in the no-tree
   experiment. With `cover_k` and the stress trigger it reaches **1.60**
   and a growing season of 0.85-1.12. Winter stays 5-6x: the non-dormant
   understory, criterion 3's remaining problem.
3. **The levers interact.** The stress trigger *raises* tree survival at
   US-Ton (0.44 -> 0.89): leaf drop cuts transpiration, soil water
   recovers, the annual-mean stress climbs back over the threshold. The
   structural scalars must be fitted together, not one after another.
4. **The floor.** All 15 open sites sit at exactly 0.25 -- the tower mean
   cannot fall below 0.43 against an observed 0.25 unless `fu_min` is free.

### Found before it could bite: the production path was gateless

The gate was threaded into `site_transient.one_year` (the fit harness) and
guarded there -- and only there. `run_global.py` and `loops_gswp3.py`, which
run every validation, never created the accumulator and called `dynaveg`
without `tree_surv`; `run_global.py` did not reset the accumulator either.
A stage-(b) validation would have silently scored a gateless model against
a gated fit -- defect 0 once more, this time in the instrumentation around
the model. Fixed: the state carries `StressAccum` when the yaml's
`tree_stress_min >= 0` (the run log says "Tree-establishment gate ACTIVE"
or "inactive", so a log states which model ran), both production `dynaveg`
calls receive `tree_surv`, the reset follows the consumption, and
`load_state` drops the three trailing leaves when a gated restart meets an
ungated template, so the fire, methane and report tools keep working on
gated states. Guards: the wiring and the reset order in both files, and a
261 -> 258 -> 261 checkpoint round-trip. The end-to-end check is
`gate_probe`: two gated years (`tree_stress_min` 0.45, `fu_min` 0.10) from
the v11 2010 state through `run_global.py`, scored with `fu_vs_tree.py`,
run as soon as the GPU frees.

### Stage (b1): structural scalars, PFT blocks frozen

PFT blocks frozen at the stage-(a) result; seven structural scalars fitted
on the four-criterion loss (`--w-tree 1`, `--w-season 10`, `--w-lai 0.5`):
`tree_stress_min` (start 0.45), `tree_stress_ramp` (0.20), `cover_k`
(0.5), `fl_min` (0.25), `fu_min` (0.10), `drop_stress_thresh` (0.5),
`tthresh_u_abs` (0 C). Starting values enter through the init yaml (`pft`
and `pheno` sections), because `extras_defaults` reads the current
parameter and the inert defaults sit outside the bounds. 4000 cells, 25
adjust + 3 eval years, ~150 evaluations overnight. Judged against stage (a)
on the same loss, then validated over the domain: `fu_vs_tree.py` on the
50-year run (the v11d table above is the baseline), `state_report.py`,
`seasonal_baseline.py`. Stage (b2) is the joint refinement with
`vmax`/`specla`, on which the tree term keeps the phantom trees from
returning.

## 15. Stage 2 groundwork: the ET error is the same phantom-tree pattern

Before ET is calibrated it needs a target and a baseline. Six gridded
products from the public ILAMB collection are now on disk
(`DATA_INVENTORY.md` 4: MODIS/MOD16, MOD16A2, GLEAM v3.3a, DOLCE, FLUXCOM
LE, WECANN LE), so the ET stage can use their spread as its uncertainty
rather than one product's bias. The baseline below is v11d (years 2056-60)
against MODIS ET (2000-13) on natural-vegetation cells with FLUXCOM GPP,
area-weighted (`verify/calibration/et_vs_modis.py`, reusable on any
validation run):

| | ET mod / obs (mm/yr) | ET ratio | ET/P mod vs obs | GPP ratio | WUE mod vs obs (gC/kg) | WUE ratio |
|---|---|---|---|---|---|---|
| all natural land | 629 / 583 | **1.08** | 0.62 vs 0.58 | 1.34 | 2.20 vs 1.77 | **1.24** |
| tree < 0.05 | 401 / 294 | **1.36** | 0.80 vs 0.58 | 2.06 | 2.05 vs 1.36 | 1.51 |
| tree 0.05-0.15 | 548 / 441 | 1.24 | 0.72 vs 0.58 | 1.77 | 2.09 vs 1.47 | 1.42 |
| tree 0.15-0.30 | 657 / 569 | 1.15 | 0.70 vs 0.61 | 1.54 | 2.19 vs 1.64 | 1.33 |
| tree 0.30-0.50 | 698 / 641 | 1.09 | 0.66 vs 0.60 | 1.44 | 2.28 vs 1.73 | 1.32 |
| tree 0.50-0.70 | 708 / 709 | 1.00 | 0.59 vs 0.59 | 1.33 | 2.36 vs 1.77 | 1.33 |
| tree > 0.70 | 830 / 890 | **0.93** | 0.52 vs 0.55 | 1.01 | 2.23 vs 2.05 | 1.09 |

Three readings.

1. **The global ET total is nearly right (1.08x) while GPP is 1.34x**, so
   WUE runs 1.24x high. The excess ET is where the excess GPP is: treeless
   land evaporates 80 % of its precipitation against 58 % observed and runs
   1.36x; forests run 0.93x. By latitude the leaking bands are the
   phantom-tree bands -- 30-45N 1.18x, 15-30S 1.38x, 30-60S 1.36x -- while
   the tropics (0.97-1.10x) and the boreal zone (0.99-1.05x) are close.
2. **So the tree gate is the first ET lever too.** Stage (b1) removes the
   trees whose transpiration makes steppes evaporate like forests; ET must
   be re-baselined after it, and ET-specific parameters fitted only to what
   remains. What will remain by the look of it: the boreal WUE (60-90N GPP
   1.79x on ET 0.99x -- a GPP problem, not ET) and the forest WUE 1.09x.
3. **The aridity extremes are data-limited.** In the most water-limited
   class MODIS ET exceeds precipitation (ET/P 1.06 against a physical
   ceiling near 1 without irrigation), the known MOD16 arid-land bias, and
   there the model reads 0.77x; in the wettest class the model reads 1.51x
   on a small observed ET. Neither extreme is a calibration target on one
   product -- the ensemble spread decides.

Model ET is `ayaet` (mm/yr) yearly and `amaet` monthly; the fit harness's
`one_year` already returns `ayaet` and the daily AET series, so the ET fit
is the carbon harness with `fluxcom_at` -> an ET loader and the seasonal
term on monthly ET.

