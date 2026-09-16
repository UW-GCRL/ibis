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

**The forcing window is NOT the gap -- correcting the paragraph this replaced.**
In the domain run at those cells the 2001-2003 triplet gives GPP 1.0438
against 1.0578 for 2001-2010: **-1.3%**. 2002 and 2003 are drought years
but the effect on the cycled mean is small. CO2 (~3%) + window (~2%) +
forcing (~1.3%) is ~6%; **~19% of the fit-vs-domain gap is unexplained**,
and the harness test above says it is not the code path. What remains is the
trajectory itself: 28 years from the 2010 state at ~373 ppm against 50 years
at 389.8. A replay of the fit's exact protocol on its own cells, extended to
the domain's protocol, is running to isolate it. `--force-years` defaults to
10 regardless -- a three-year window is a poor climate sample even at 1.3%.

Whatever the harness answer, the calibration has reached what `vmax` and
`specla` can do: pools in range, spatial pattern largely fixed, high-flux cells
on FLUXCOM, and a ~1.4x residual living in low-productivity cells that these
parameters do not reach. Where that residual actually lives -- water stress,
sparse-canopy light use, the stomatal coupling that left ET unmoved -- is the
next question, and it is a process question, not a parameter one.
