# Calibration

Started 2026-09-09, after the fire and methane modules were re-validated on the
corrected carbon cycle. Order set by Hangkai: carbon, then ET, then the smaller
parts. Methane is evaluated against FLUXNET-CH4 (Chenchen's processing).


## State of the calibration (updated 2026-09-21 14:45; phenology clock fixed -- all parameters to be refit on the corrected model)

Three criteria for carbon (the user's): global total, spatial pattern,
seasonal pattern. A fourth was added as the means to the second: tree
cover against the CLM tree fraction. ET follows.

| version | what changed | GPP PgC/yr (obs 120-130) | GPP by band 30-50N / 60-70N / tropics | tree cover, natural land (obs 0.34) | seasonal amplitude (obs 1) | section |
|---|---|---|---|---|---|---|
| ctl (defaults) | -- | 234 (192) | | | | 9 |
| v11d | per-PFT vmax/specla fit | 182 (151) | 1.77 / 1.82 / 1.18 | 0.66 | | 10-12 |
| v12a, stage (a) | + stressfac, phenology scalars | 176 (145) | 1.64 / 1.70 / 1.16 | 0.67 | 0.80 | 13-14 |
| v12c, stage (b1) | + tree-establishment gate, cover, dormancy trigger | 164 (134) | 1.54 / **1.02** / 1.10 | **0.53** | 0.78 | 14 |
| corrected canopy, 2-yr probe | + `vmax_mix`, `scale_leaf_only`, no refit | 131 | 0.87 / 1.24 / 0.92 | 0.55 | | 16-18 |
| v12d, stage (b2) | joint refit on the corrected canopy, tree weight 3 | 148 (120) | 0.98 / 1.12 / 0.98 | 0.29 (forests lost) | 0.81 | 18 |
| **v12e, stage (b3)** | growing-season survival stress, joint refit | 164 (**132**) | 1.40 / 1.03 / 1.13 | **0.355** (r 0.66) | 0.82 | 18 |
| **closing protocol on (b3)** | spin-up + 1901-2010 transient + hindcast, same parameters | 161 (**130**) | 1.38 / 0.98 / 1.10 | 0.323 (r 0.63) | 0.82 | 19 |
| v12f, stage (b4) | gate fixed 0.80/0.10, total weighted, land + representative weights | 149 (**118**) | 1.25 / 1.05 / 1.01 | 0.272 (forests 0.62) | 0.81 | 18 |
| v12g, joint carbon + ET | from (b3), gate free, (b4) scoring, ET terms, soil_beta_power | 148 (**118**) | 1.27 / 1.05 / 0.99 | 0.312 (forests 0.67) | 0.80 | 15, 18 |

GPP column: binary-mask total as quoted in the sections, and in brackets
the total on TRUE land (section 18, "The mask"), which is the number to
compare with 120-130. Earlier rows: ctl 192, v11d 151, v12a 145 on true land.

Structural findings, in the order they were forced by the data:

1. **Phantom trees** (14): the model grows forest where the observation has
   steppe -- nothing limited tree establishment by water. Fixed by a
   survival gate on annual-mean soil-water stress, calibrated against the
   CLM tree fraction. The boreal band went from 1.8x to 1.02x GPP.
2. **The seasonal term never moved** (13-16) because the canopy's GPP was
   nearly independent of its leaf area (dlnGPP/dlnLAI = 0.08): F77's canopy
   scaling credits stem-absorbed PAR to leaf photosynthesis, a flat +25 %
   for the upper canopy and the reason leaf drop, phenology and LAI could
   not act. `scale_leaf_only` (16). Alone it takes 168 -> 131 PgC.
3. **Four of twelve vmax entries were dead** (17): each canopy big leaf took
   ONE PFT's vmax by existence priority. `vmax_mix` weights by leaf area.
4. **The phenology clock ran 24x too slow** (16): the port applied F77's
   per-timestep ten-day-mean weight once a day, giving `a10td` and the
   `a10anc*` drought triggers a 240-day memory -- the annual temperature
   cycle attenuated to 24 % with a 77-day lag, so the lower canopy never
   met winter and the dormancy triggers never met a dry season. Every fit
   in this document ran on it. Fixed 2026-09-20 22:55.
5. **Observation caveats** (15-16): the ET products spread 38 % per cell
   (score pattern and season, not the total); FLUXCOM's savanna phase is
   confirmed by towers but its dry-season depth is 0.6x theirs (hold the
   model to the phase).

Every switch is off by default and bit-for-bit F77; every fitted
configuration from (b2) on has both canopy switches on. ET (Stage 2) is
prepared: six products, a five-product ensemble, a baseline (15), and the
ET terms in the fit harness; it is re-baselined once (b2) lands.

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
`gate_probe`: two gated years (`tree_stress_min` 0.45, `fu_min` 0.10,
`cover_k` 0.5) from the v11 2010 state through `run_global.py`, scored with
`fu_vs_tree.py`. **Passed** (21:21): the log states the gate is active and
the 258-leaf restart migrated into the 261-leaf template; after two years
natural-land `fu` is 0.66 -> 0.52 (obs 0.34; MAD 0.365 -> 0.288), treeless
land 0.40 -> 0.21 with its GPP 2.01x -> 1.62x, and the global positive GPP
excess is down 21 %. Two trade-offs the fit has to settle are already
visible -- forests 0.92 -> 0.85 against 0.91, and "missing trees" (tree >
0.5, fu < 0.3) up from 2 % to 5 % of natural land -- which is what the tree
term is for. The phantom cells that remain (10 % of land, fu 0.88 where the
observation is 0.05) have wet soils: their treelessness is cold- not
water-limited, which is `existence`'s job rather than the gate's; to be
read on the 50-year validation.

### Stage (a) result: -11 %, and three levers on their bounds

120 evaluations, loss **0.8717 -> 0.7761** (hold-out 1.10 -> 1.03); GPP
term 0.300 -> 0.276, LAI term 0.794 -> 0.654, seasonal term 0.0175 ->
0.0173 (flat, as section 13 predicted). The eight scalars, default ->
fitted [bounds]:

| scalar | default | fitted | bounds | |
|---|---|---|---|---|
| `stressfac` | -5.0 | **-2.50** | [-10, -2.5] | **on the bound: maximum water-stress response** |
| `tthresh_l_c` | 0 | **+7.99** | [-5, 8] | **on the bound: grasses dormant 8 C warmer** |
| `tthresh_u_offset` | 5 | 11.7 | [0, 12] | near the bound: trees drop leaves at tc + 11.7 |
| `gthresh_l_c` | -5 | -1.13 | [-12, 2] | |
| `gdd_thresh_l` | 150 | 99.6 | [50, 300] | |
| `gdd_thresh_u` | 100 | 68.9 | [50, 250] | |
| `drop_floor` | 0.10 | 0.27 | [0.02, 0.5] | |
| `ddays` | 15 | 11.1 | [5, 60] | |

Read together: the optimiser asked for the strongest water stress it was
allowed and the earliest dormancy it was allowed, and bought 11 % with them
-- almost all of it LAI. These are compensations for the structural excess
(phantom trees, over-painted understory), not physiology; the fitted
values are not carried into stage (b1) as fixed truths.

### Stage (a) over the domain: -3.6 % GPP, the season untouched, the Arctic thinner

50-year validation from the v11 2010 state on `params_fit_v12a.yaml`
(`val_v12a`, years 2056-60), then 10 monthly years:

| | v11d | **v12a** | observed |
|---|---|---|---|
| GPP PgC/yr | 182.1 | **175.6** | 120-130 |
| NPP PgC/yr | 84.8 | 81.6 | 55-60 |
| Rh PgC/yr | 78.6 | 76.6 | 49-58 |
| ET mm/yr | 527 | 524 | 500-550 |
| veg C PgC | 634 | 618 | 450-650 |
| soil C kgC/m2 | 13.0 | 12.6 | 10-15 |
| LAI mean | 2.15 | 1.99 | |
| GPP 30-50N / 50-60N / 60-70N vs FLUXCOM | 1.77 / 1.69 / 1.82 | **1.64 / 1.60 / 1.70** | |
| GPP 30S-30N | 1.18 | 1.16 | |
| LAI 30-50N vs MODIS | 1.97 | 1.73 | |
| LAI 70-90N vs MODIS | 0.56 | **0.46** | |
| seasonal amplitude model/obs, all / 30-50N | | **0.80 / 0.61** | 1 |
| shape error, flux-weighted | | 0.0164 | |
| fu vs CLM tree (natural land), MAD | 0.661 / 0.365 | 0.665 / 0.370 | 0.335 |
| ET vs MODIS ET, WUE ratio | 1.08 / 1.24 | 1.07 / 1.21 | 1 / 1 |

The 11 % on the sample is 3.6 % of global GPP (1.46x -> 1.40x), bought
with the levers on their bounds. The seasonal cycle is unchanged: amplitude
0.80 of observed, 0.61 at 30-50N; the sparse-excess class still runs
2.8-3.3x through winter and 4.6x in October. Tree cover did not move --
stage (a) had no cover lever -- and neither did ET or WUE. One regression:
Arctic LAI (70-90N) fell from 0.56x to 0.46x MODIS, the price of the cold
thresholds pushed toward their bounds. Criteria: total 1.40x, spatial
1.6-1.7x mid/high latitudes against 1.16x tropics, seasonal amplitude 0.80
-- none met; the parameter-only route is exhausted, which is what stage (b)
is for.

### Stage (b1): structural scalars on neutral physiology, PFT blocks frozen at v11d

Start from **v11d** (F77 phenology, `stressfac` -5), not from the stage-(a)
values: a maximal `stressfac` already suppresses dry-cell GPP, so on top of
it the gate would be found weaker than the tree fraction wants. Eight
scalars on the four-criterion loss (`--w-tree 1`, `--w-season 10`,
`--w-lai 0.5`): `tree_stress_min` (start 0.45), `tree_stress_ramp` (0.20),
`cover_k` (0.5), `fl_min` (0.25), `fu_min` (0.10), `drop_stress_thresh`
(0.5), `tthresh_u_abs` (0 C) -- and `stressfac`, because `drystress`
shapes the very `stresstu` the gate accumulates, so the two are coupled.
Starting values enter through the init yaml (`make_start_yaml.py`; `pft`
and `pheno` sections), because `extras_defaults` reads the current
parameter and the inert defaults sit outside the bounds. 4000 cells, 25
adjust + 3 eval years, 150 evaluations overnight. Judged on the same sample
against both 0.8717 (v11d) and 0.7761 (stage a): structure against
compensation. Then validated over the domain -- `fu_vs_tree.py` on the
50-year run (the v11d table above is the baseline), `state_report.py`,
`seasonal_baseline.py`, `et_vs_modis.py`. Stage (b2) is the joint
refinement of phenology, `stressfac`, `vmax`/`specla` on the fixed
structure, on which the tree term keeps the phantom trees from returning.

Launched 22:00 on 2026-09-17 (`fit_carbon_v12c`; log confirms `CLM tree
0.30 (3058 cells)` at the sample, `StressAccum attached`, 8 extras, PFT
blocks frozen). Baseline at the starting values: **1.1968** = GPP 0.2767 +
0.5 x LAI 0.8755 + 10 x season 0.0175 + 1 x tree 0.307. Two readings before
a single optimiser step: the GPP term at the starting structure on v11d
physiology already equals stage (a)'s best after 120 evaluations (0.2762) --
the structure gives at once what the levers gave at their bounds -- and the
three carbon terms sum to 0.8895, so the fit must come in under 0.7761 to
beat compensation. The LAI term is worse than v11d's 0.794: cells whose
trees the 0.45 gate removed are now too thin (the "missing trees" the probe
showed), which `tree_stress_min`/`ramp` and the tree term settle between
them. ~150 evaluations at ~205 s; `chain_v12c.sh` validates and scores at
the end (`fu_vs_tree`, `state_report`, `seasonal_baseline`).

### Stage (b1) result: 1.1968 -> 1.0855; the gate survives weakened, the season ignores everything

150 evaluations (22:00 -> 06:40, 197-210 s each). Terms, start -> fitted:
GPP 0.2767 -> **0.2452**, LAI 0.8755 -> **0.6773**, season 0.0175 -> 0.0174,
tree 0.307 -> **0.327**; hold-out 1.464 -> 1.317. Carbon terms 0.758
against stage (a)'s 0.776 -- better, on a structurally saner model.

| scalar | start | fitted | bounds | |
|---|---|---|---|---|
| `stressfac` | -5.0 | **-8.27** | [-10, -2.5] | LESS stress -- the opposite of stage (a)'s -2.5 |
| `tree_stress_min` | 0.45 | 0.365 | [0.1, 0.9] | gate kept, weakened |
| `tree_stress_ramp` | 0.20 | 0.136 | [0.05, 0.5] | sharper edge |
| `cover_k` | 0.5 | 0.27 | [0.2, 2.0] | understory cover thinned hard (fl = 1 - exp(-0.27 LAI)) |
| `fl_min` | 0.25 | 0.249 | [0.01, 0.25] | at the ceiling = default |
| `fu_min` | 0.10 | **0.042** | [0.01, 0.25] | treeless land can read treeless |
| `drop_stress_thresh` | 0.5 | **0.051** | [0.05, 0.85] | **on the floor: drought dormancy switched off** |
| `tthresh_u_abs` | 0 C | **14.9 C** | [-5, 15] | **on the ceiling: deciduous trees dormant below 15 C** |

Four readings.

1. **The tree weight did not bind.** The tree term ended above its start:
   the LAI term (0.5 x 0.68 = 0.34) is the same size as the tree term
   (0.33), and where MODIS wants LAI ~1.1 on semi-arid land the cheapest
   way to get it is trees. The optimiser bought LAI with cover the tree
   fraction says is not there. `--w-tree 1` was a guess; the data say 3.
2. **`stressfac` and the gate are one lever.** Stage (a) drove `stressfac`
   to maximum stress to kill dry-cell GPP; with the gate available the
   optimiser reversed it to -8.3 (less stress) and let the gate do the
   killing. Since `stresstu` is what the gate accumulates, the pair
   (`stressfac`, `tree_stress_min`) is a single coordinate: 0.365 at -8.3 is
   not comparable to 0.45 at -5.
3. **Two more scalars on bounds, both compensations.** Drought dormancy
   was rejected outright (trigger at its floor never fires); the absolute
   upper-canopy threshold went to its 15 C ceiling, shortening deciduous
   trees' season -- the same move stage (a) made with `tthresh_u_offset`.
   Whatever the mid-latitude excess is, the optimiser keeps reaching for
   "fewer leaf-days on trees" and is stopped only by the bounds I set.
4. **The seasonal term has now ignored every lever on two fits** (0.0175 ->
   0.0173 -> 0.0174). It is flux-weighted, so it is carried by the tropics
   -- section 13's +0.78-month phase lag on 63 % of tropical cells -- and
   nothing in stages (a) or (b1) touches tropical phenology. Criterion 3
   needs its own diagnosis (tropical dry-season response) rather than more
   weight on a term these levers cannot move.

Decision waits on the domain validation (`chain_v12c`): the three criteria
are judged there, not on the sample.

### Stage (b1) over the domain: the boreal band is exact, the steppes are half-fixed, the understory is pinned

50-year validation from the v11 2010 state on `params_fit_v12c.yaml`
(`val_v12c`, years 2056-60; gate active, logged), scored with
`fu_vs_tree.py` and `et_vs_modis.py` (pools and the seasonal score follow
from the chain):

| | v11d | v12a (stage a) | **v12c (stage b1)** | observed |
|---|---|---|---|---|
| fu, natural land / MAD tree term | 0.661 / 0.365 | 0.665 / 0.370 | **0.530 / 0.280** | 0.335 / 0 |
| fu where tree < 0.05 (37 % of natural land) | 0.40 | 0.42 | **0.18** | 0.01 |
| GPP there | 2.01 | 1.85 | **1.53** | 1 |
| GPP 60-90N / ET / WUE | 1.79 / 0.99 / 1.81 | 1.67 / 0.99 / 1.69 | **0.99 / 0.96 / 1.04** | 1 |
| GPP 45-60N | 1.64 | 1.52 | 1.36 | |
| GPP 30-45N (fu 0.47 vs 0.18) | 1.85 | 1.73 | 1.58 | |
| GPP 15-30S / 30-60S | 1.69 / 2.04 | 1.63 / 1.94 | 1.47 / 1.77 | |
| GPP tropics (0-15N, 15S-0) | 1.08 / 1.01 | 1.06 / 1.00 | 1.00 / 0.99 | |
| forests (tree > 0.7): fu / GPP | 0.92 / 1.02 | 0.91 / 0.99 | 0.88 / 0.98 | 0.91 / 1 |
| GPP, natural land (ET-scored cells) / WUE | 1.34 / 1.24 | 1.29 / 1.21 | **1.19 / 1.12** | 1 / 1 |
| ET, natural land | 1.08 | 1.07 | 1.07 | 1 (spread 0.91-1.11) |
| phantom cells (tree < 0.15, fu > 0.4), share of natural land | 15.9 % | 16.5 % | 10.9 % | |
| missing-tree cells (tree > 0.5, fu < 0.3) | 2.2 % | 2.3 % | 3.4 % (GPP 0.69) | |
| global positive GPP excess (area-weighted, rel. v11d) | 1.00 | 0.90 | **0.70** | |
| **GPP PgC/yr** (state_report, last 5 yr) | 182.1 | 175.6 | **163.6** | 120-130 |
| NPP / Rh PgC/yr | 84.8 / 78.6 | 81.6 / 76.6 | 77.4 / 73.1 | 55-60 / 49-58 |
| ET mm/yr | 527 | 524 | 523 | 500-550 |
| veg C PgC / soil C kgC m-2 / litter kgC m-2 | 634 / 13.0 / 0.92 | 618 / 12.6 / 0.88 | 570 / 12.0 / **0.72** | 450-650 / 10-15 / 0.4-0.8 |
| GPP by band 30S-30N / 30-50N / 50-60N / 60-70N / 70-90N | 1.18 / 1.77 / 1.69 / 1.82 / 1.07 | 1.16 / 1.64 / 1.60 / 1.70 / 0.95 | 1.10 / 1.54 / 1.34 / **1.02** / **0.69** | 1 |
| LAI by band vs MODIS, same bands | 1.30 / 1.97 / 1.39 / 1.08 / 0.56 | 1.27 / 1.73 / 1.21 / 0.96 / 0.46 | 1.23 / 1.61 / **1.01** / **0.55** / 0.38 | 1 |
| seasonal amplitude model/obs, all / 30-50N; shape error | | 0.80 / 0.61; 0.0164 | 0.78 / 0.60; 0.0167 | 1 |

Four readings (the fourth added with the pools and the season).

1. **The boreal and Arctic band is now right on all three fields** -- GPP
   0.99, ET 0.96, WUE 1.04, fu 0.24 against 0.25 -- where stage (a) had
   1.67x GPP and 1.69x WUE. Removing the trees that could not survive there
   did what no parameter could.
2. **The steppes are half-fixed.** 30-45N still carries fu 0.47 against
   0.18 and GPP 1.58x; 15-60S 1.5-1.8x. The remaining phantom cells (11 %
   of natural land, fu 0.86 where the observation is 0.05) hold 29 % of
   the remaining excess. The gate weakened to 0.365 under a tree weight
   that did not bind (section 14); a heavier weight is the obvious next
   move, with a watch on the missing-tree side (3.4 % of land now at GPP
   0.69).
3. **The understory cover is pinned.** `fl` reads 0.25 -- its floor -- in
   every band and every tree class: with `cover_k` 0.27, `1 - exp(-0.27
   LAI)` is below 0.25 for any LAI under 1.06, so the fitted understory has
   no cover variation at all. That is how the optimiser met "understory
   over-painted" with two scalars; it is not how grass cover works.
   Stage (b2) needs `cover_k` and `fl_min` re-bounded so cover can vary
   (or the cover-from-LAI form replaced by something with a slope).
4. **Criterion 1 moved from 1.40x to 1.31x; criterion 3 did not move; and
   the boreal band shows the model's indifference to leaf area from the
   other side.** 60-70N now has the right GPP (1.02x) with half the leaves
   (LAI 0.55x MODIS) and 70-90N has fallen to 0.69x: the gate thinned the
   boreal forest and the flux hardly noticed, which is the dlnGPP/dlnLAI of
   0.08 measured in sections 14 and 16 read the other way round. Litter is
   in its observed range for the first time; the pools stay in theirs. The
   seasonal amplitude is 0.78 as before. The next stage cannot be another
   fit on this canopy: section 16 finds the canopy optically black, and
   section 17 the vmax table half dead; both are structural and both come
   before (b2).

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

**The ensemble** (`verify/calibration/et_obs.py` ->
`inputs/eval/et/et_ensemble_0.5deg.npz`: 2001-09 annual means and monthly
climatologies of MODIS, MOD16A2, GLEAM v3.3a, DOLCE and FLUXCOM LE on the
IBIS grid, mm/day, with the ensemble mean / min / max and product count).
On the 47,008 cells all five cover, the products agree on pattern (each
r = 0.97 against the ensemble mean) and disagree on level -- DOLCE 0.91x,
MODIS and MOD16A2 0.97x (near-duplicates), GLEAM 1.05x, FLUXCOM 1.11x of
the mean -- with a per-cell max-min spread of 220 mm/yr, **38 % of the
mean**. The model's 1.08x global ET sits inside that band: ET's total is
already within observational uncertainty, the pattern is not. Rules for the
ET stage that follow: score on the all-five cells (coverage differs, so the
mean elsewhere depends on which products cover a cell), take the ensemble
mean as the target and the spread as its uncertainty, and put the weight on
the spatial and seasonal terms rather than the total.

**Tower scale** (`verify/calibration/tower_et.py`; 194 FLUXNET2015 towers
with >= 24 valid months of `LE_F_MDS`, model = the (b1) monthly output at
the tower's cell, ensemble at the same cell where all five products cover
it): model / tower **1.13**, ensemble / tower **0.99** -- the gridded
ensemble agrees with the towers on average, which makes it a sound target
and puts the model's 1.07x-1.13x on the model. By class the model runs
DBF 1.26, EBF 1.22, open shrub 1.34, woody savanna 1.20, ENF 1.12, grass
1.15, crops 1.01 and wetlands **0.83** (the one class it under-evaporates);
by latitude 1.17 (0-25), 1.07 (25-45), 1.17 (45-60), 1.37 (60-90) against
an ensemble that also exceeds the high-latitude towers (1.12, 1.17), so
part of the poleward excess is the towers' energy-balance closure. The
model's seasonal ET amplitude is 9 against the towers' 15: too flat, like
GPP. Baseline on the F77 canopy; the corrected canopy moved domain ET by
only 2 % (section 18), so this is close to the Stage-2 starting point.

**ET lever test 1** (`et_lever1.py`; 70 fit-sample cells in three classes,
(b3) parameters, 6 years cycled on 2001-03 from the v11 state -- so the
absolute levels below carry the dry-year and transition bias of section 10
and only the responses count): the Ball-Berry slope `coefm`, the obvious
"ET at fixed GPP" lever, is not one. All five slopes x0.7 moves ET by -7 %
(forest), -4 % (open), -3 % (boreal) and GPP by -10 %, -6 %, -14 %; WUE
barely changes (forest 0.87 -> 0.84). dlnET/dln(coefm) ~ 0.2 with a GPP
side-effect of ~0.3: the coupled scheme fixes WUE, and the stomatal
conductance is not what limits ET in these canopies. The seasonal ET
amplitude, on the other hand, is wrong in opposite directions -- forests
0.66 of observed, open land 2.3x (the model's dry-season ET collapses where
the products keep evaporating), boreal a fraction (the model's winter ET
floor is above the products') -- and no stomatal lever touches that.

So Stage 2 does not start with a fit. It starts with a decomposition of
model ET into transpiration, soil evaporation, canopy interception and
snow sublimation by class and month, against the ensemble's total and
GLEAM's components where they can be had, to find which component carries
each class's error; the levers are then the soil-evaporation resistance,
interception capacity and the winter/sublimation path, not `coefm`.
The port computes the components F77 does but accumulated only the total (`ayaet`). **Built:** `ETParts` (`ibis/state/et_parts.py`) -- transpiration, interception evaporation (leaves and stems), soil evaporation, snow sublimation, in mm -- an optional trailing state module switched on by `physics: et_components: true`; `one_year` returns them monthly; the yearly file carries `aytrans` / `ayinterc` / `aysoilev` / `aysnowsub`. Bit-identical physics with or without it, closure to the total to 1e-6.

**First decomposition** (`et_decomp1.py`; the same 70 cells, (b3)
parameters, 6 years on 2001-03 from the v11 state -- transient-biased, so
shares and seasonal shapes are the reading, not levels):

| class | ET model / obs | transpiration | interception | soil evaporation | snow |
|---|---|---|---|---|---|
| forest (tree > 0.7) | 773 / 923 (0.84) | 315 (41 %) | 258 (33 %) | **195 (25 %)** | 4 |
| open (tree < 0.3) | 551 / 659 (0.84) | 277 (50 %) | 50 (9 %) | **225 (41 %)** | -1 |
| boreal (> 50 N) | 398 / 395 (1.01) | **87 (22 %)** | 87 (22 %) | **212 (53 %)** | 12 |

Three things stand out. Soil evaporation carries a quarter of forest ET,
half of boreal ET, and 4-6 mm/month in boreal winter -- under a closed
canopy and over frozen or snow-covered ground it should be small; that
is where the model puts the ET the products do not have (boreal Dec-Feb
model 9-12 mm/month against 2-4). Boreal transpiration is 22 % of ET where
a forest's should be half or more -- and boreal winter GPP runs 5-10x
FLUXCOM in these cells (evergreens assimilating at sub-zero temperatures;
small in mm but a seasonal-shape error). Forest summer ET is short
(Jun-Aug 64-75 against 94-99) exactly where forest GPP is short (0.59-0.66
in the same months): the coupled scheme carries the carbon deficit into the
water, so forest ET is the carbon stage's residual, not an ET lever. In
open land the amplitude error (section above, 2.3x) is soil evaporation
that runs at potential when the surface is wet and collapses when it dries
-- the model has no soil-surface resistance between the two. The domain
decomposition (`et_decomp_domain.py`, on a hindcast written with
`et_components`) follows; the ET levers it points to are the soil-surface
resistance and the snow-cover / frozen-ground evaporation path, then
interception capacity.

**The formulation behind the soil evaporation** -- corrected the same
evening. A first reading of `setsoi` (the `qglif` weights, which only
partition liquid / ice / puddle) and `turcof` (`cfi`, purely aerodynamic)
led to the claim that IBIS has no soil-moisture control on bare-soil
evaporation. It does: `turvap` (F77 `canopy.f` 792-860, ported line for
line) scales the soil surface humidity by `qgfac0 = alpha = betas * hfac +
(1 - betas) * q34/qg`, with `hfac = exp(psi g / (R_v T_g))` the
matric-potential humidity of the top layer and `betas` a wetness "beta",
`betaw = (w_top - swilt) / (1 - swilt)` corrected for the surface energy
balance (`betafac`). So the control exists; the question is its strength.
Two features make it weak on dry ground: the beta reaches zero only at the
wilting point, whereas a real soil surface dries far below it (CLM and
Noah add a dry-surface-layer or soil resistance that grows as the top
centimetres dry), and the top layer is 10 cm thick, so the surface store
that feeds evaporation at near-saturation humidity is large. Whether that
is what keeps open-land soil evaporation at a flat 15-24 mm/month through
the dry season is a measurement, not a reading: the domain decomposition
first, then a lever test on the 70 cells with the beta's zero moved toward
air-dry and with a Sellers-type soil resistance added to `1/sg` -- both
off by default, bit-identical, as with every structural switch so far.

**The domain decomposition** (`et_decomp_domain.py` on the (b3) closing
run's 2001-2010 hindcast rerun with `et_components`; 44,993 natural-land
cells with all five ET products; true-land area weights):

| class | ET model / ensemble | transpiration | interception | soil evaporation | snow | GPP / obs |
|---|---|---|---|---|---|---|
| **all natural land** | 590 / 594 = **0.99** | 270 (**0.46**) | 142 (0.24) | 174 (0.30) | 4 | 1.15 |
| tree < 0.05 | 383 / 336 = 1.14 | 158 (0.41) | 47 (0.12) | 174 (0.45) | 4 | 1.55 |
| tree 0.15-0.50 | 611-645 / 599-654 = 0.99-1.02 | 0.48-0.50 | 0.20-0.23 | 0.27-0.32 | | 1.28-1.32 |
| tree > 0.70 | 798 / 873 = 0.91 | 361 (0.45) | **275 (0.34)** | 156 (0.20) | 5 | 0.90 |
| 60-90 N | 304 / 279 = 1.09 | **38 (0.12)** | 30 (0.10) | **216 (0.71)** | 21 (0.07) | 0.92 |
| 45-60 N | 411 / 390 = 1.05 | 120 (0.29) | 83 (0.20) | 198 (0.48) | 10 | 1.09 |
| 30-45 N | 469 / 445 = 1.06 | 0.48 | 0.17 | 0.34 | | 1.47 |
| **15-30 N** | 571 / 683 = **0.84** | 0.55 | 0.19 | 0.26 | | 1.18 |
| 15 S-15 N | 865-978 / 882-1078 = 0.91-0.98 | 0.48-0.50 | 0.32-0.33 | 0.17-0.20 | | 0.97-1.00 |
| **15-60 S** | 545-594 / 442-530 = **1.12-1.23** | 0.52-0.56 | 0.19-0.20 | 0.25-0.28 | | 1.56-1.86 |

Transpiration per unit GPP is 0.22-0.26 kg per gC in every class -- the
coupled scheme fixes it -- so the ET pattern is the GPP pattern damped by
the other two components. Readings, and what the ET stage is therefore
for:

1. **Criterion 1 for ET is met** (0.99 on natural land, before any ET
   parameter has been touched) and the pattern is within 10 % except the
   subtropical belt 15-30 N (0.84 -- the Sahel, India, Mexico: GPP 1.18x
   yet ET short) and the southern mid-latitudes (1.12-1.23, where GPP is
   1.6-1.9x). These two are the pattern targets.
2. **The partition is where the structure is wrong.** Transpiration is
   46 % of ET globally against 57-64 % from isotope and flux syntheses
   (Good et al. 2015, Wei et al. 2017, Lian et al. 2018 -- the CMIP-class
   underestimate, which IBIS shares); 45 % in forests, 41 % on open land.
   Interception carries 34 % of forest ET (upper end of the observed
   10-30 %). And the boreal zone is the extreme case: **71 % soil
   evaporation, 12 % transpiration** at 60-90 N -- with half the observed
   LAI (0.52x, section 19), the ground receives the radiation the canopy
   should, and the soil-evaporation control lets it evaporate. So the ET
   levers are the strength of the soil-evaporation control (a dry-surface
   resistance, Sellers 1992) and the interception capacity, with the
   boreal LAI deficit belonging to the carbon side; the partition
   (T/ET ~ 0.6, boreal ~ 0.5) is the ET stage's fourth criterion, a weak
   one, since no gridded product on disk resolves components.
3. The seasonal ET criterion is scored on the ensemble's monthly
   climatology (`--w-et-season`), with the tower finding kept in mind: the
   model's ET seasonal amplitude at 194 towers is 9 against 15 observed.

**ET lever test 2 -- the soil-evaporation control strength**
(`soil_beta_power`, built: `turvap`'s wetness beta raised to a power `p`,
`p` = 1 bit-identical, threaded from `CanopyParams`, fittable as an extra;
`et_lever2.py`, same 70 transient-biased cells, responses not levels):

| class | p = 1 (F77): ET ratio / T/ET / Ei/ET / Es/ET / GPP / ET amp | p = 3 | p = 5 |
|---|---|---|---|
| forest | 0.84 / 0.41 / 0.33 / 0.25 / 0.73 / 0.66 | 0.77 / 0.47 / 0.39 / **0.14** / 0.75 / 0.64 | 0.75 / 0.49 / 0.41 / 0.10 / 0.75 / 0.65 |
| open | 0.84 / 0.50 / 0.09 / 0.41 / 1.01 / 2.26 | 0.80 / **0.69** / 0.15 / **0.17** / **1.21** / **1.43** | 0.77 / 0.74 / 0.17 / 0.10 / 1.25 / 1.14 |
| boreal | 1.01 / 0.22 / 0.22 / 0.53 / 0.79 / 0.24 | 0.78 / 0.31 / 0.31 / 0.34 / 0.82 / 0.23 | 0.69 / 0.36 / 0.36 / 0.23 / 0.82 / 0.22 |

The lever is strong and it acts on the right things: at p = 3 the open
land's soil evaporation falls from 41 % to 17 % of ET, transpiration rises
to 69 %, and the seasonal amplitude error (2.3x) halves (1.4x); boreal soil
evaporation falls from 53 % to 34 %. Two consequences bind it to the
carbon fit. Total ET falls in every class (the saved soil water is only
partly transpired), so on the domain -- treeless land 1.14x, boreal 1.09x,
forests 0.91x -- it helps two classes and hurts the forests; and the water
it saves feeds grass GPP (open land 1.01 -> 1.21x), on land whose GPP is
already 1.55x observed. So `soil_beta_power` cannot be fitted on ET alone.

**ET lever test 3 -- interception** (`et_lever3.py`, same cells): the
water-holding capacities of leaves and stems halved or doubled move the
interception share by at most 0.01 and total ET by under 1 % -- with a
two-hour drip time the store is never the limit; what sets the 33-41 % of
forest ET that is interception is the wetting-drying dynamics of the
rain, not a parameter. `fmax`, the maximum wetted fraction, moves Ei/ET by
only +-0.02 between 0.15 and 0.40 but moves GPP by -+6 %, because wet
leaves do not assimilate: it is a GPP lever wearing an ET label, and it
stays out of the ET fit. The interception EXTRAS remain available; the
ET stage's structural lever is the soil-evaporation control alone, with
the boreal LAI on the carbon side.

**A trap found at the first joint evaluation.** Its loss was 10.77, of
which the ET-shape term was 8.07 -- 46 times the GPP-shape term. The
cause is a model anomaly, not a shape: 1,139 of the 2,129 all-five sample
cells have a month of *negative* ET and a few subtropical cells reach
-80..-146 mm/month (`DEFECTS_FOR_REVIEW.md`), which turns
fraction-of-year shapes and log-ratios into noise (median cell 0.009,
mean 0.58). The ET terms now clip months at zero and exclude cells whose
model annual ET is under a fifth of the observed (count reported); on the
(b3) output the shape term is then 0.009-0.018, the GPP term's size. The
fit was stopped and relaunched with the guard.

Relaunched 00:43 on 2026-09-20 (`fit_carbon_v12g`). Baseline **2.4407**:
GPP 0.799 (of which ~0.55 is the weighted total -- the (b3) parameters read
~1.16x on this sample), LAI 0.609, season 0.174, tree 0.705, **ET 0.065,
ET shape 0.089** -- the ET terms now the size of their carbon
counterparts. The mark to beat on the carbon terms alone is (b4)'s 1.37;
180 evaluations, ~10 h.

**Result: 2.4407 -> 1.4984** (best at eval 179, the last generation). Terms:
GPP 0.211 (of which the weighted total ~0.02: the sample total at ~1.03x),
LAI 0.512, season 0.0172, tree 0.231 (the lowest of any fit), ET 0.0705,
ET shape 0.0097; hold-out 1.547. On the carbon terms alone, 1.331 --
better than (b4)'s 1.370 with the gate fitted rather than fixed.

| scalar | start (b3) | fitted | |
|---|---|---|---|
| `tree_stress_min` / `ramp` | 0.630 / 0.159 | **0.759 / 0.137** | between (b3)'s 0.63 and (b4)'s fixed 0.80 |
| `stressfac` | -6.53 | **-7.67** | less stress -- moving together with the gate, as they must |
| `soil_beta_power` | 1.5 | **1.31** | a mild strengthening of the soil-evaporation control (1 = F77) |
| `cover_k` | 0.428 | 0.203 (floor) | understory cover thinned to the bound |
| `drop_stress_thresh` | 0.134 | 0.277 | dormancy trigger partly live |
| `tthresh_u_abs` | 12.5 C | 8.9 C | |
| `fl_min`, `fu_min` | 0.229, 0.033 | 0.242, 0.063 | |

| PFT | vmax start -> fitted (umol) | specla | |
|---|---|---|---|
| trop_bl_ever | 52.1 -> 46.7 | 14.9 -> 14.0 | |
| trop_bl_dec | **52.1** -> 46.0 | 11.2 -> 11.3 | start should have been 40.3 (below) |
| warm_bl_ever | 28.0 -> 26.3 | 18.2 -> 15.8 | |
| temp_con_ever | 17.6 -> 19.1 | 14.9 -> 13.5 | |
| temp_bl_dec | 58.3 -> 59.9 | 14.1 -> 12.2 | at the 60 ceiling |
| boreal_bl_dec | **58.3** -> **49.9** | 30.1 -> 21.6 | start should have been 20.3 (below) |
| ever_shrub | 35.7 -> 35.6 | 8.0 -> 8.7 | |
| dec_shrub | **35.7** (frozen) | | should have been 28.0 (below) |
| c4_grass / c3_grass | 18.2 -> 21.5 / 17.0 -> 16.2 | 10.2 -> 11.6 / 11.3 -> 10.0 | |
| boreal_con_ever, boreal_con_dec | 25.0 (frozen) | | |

**A provenance flaw.** The start yaml carried three vmax values that were
not (b3)'s: `trop_bl_dec` 52.1, `boreal_bl_dec` 58.3 and `dec_shrub` 35.7
-- the `--vmax-mix` dead-to-live mapping, which a patch to
`make_start_yaml.py` had slid under the `--et-starts` flag (fixed and
guarded). The optimiser re-fitted the first two (`boreal_bl_dec` settled at
49.9, 2.5x (b3)'s 20.3 -- whether the 60-70 N band wanted that is read on
the domain) and could not move the frozen `dec_shrub`, so the finalized
yaml carries 35.7 where (b3) had 28.0 -- a small effect (shrub LAI is
small and the shrub big leaf mixes the two) but a value no fit chose. The
result stands as a fit from a different start; the flaw is noted, and the
pool stage starts from a corrected base.

Final report on the scored cells, model and observation on the same mask:
**GPP 0.991 (fit) / 0.990 (held out), ET 0.971 / 0.975, LAI 1.66 vs 1.57 /
1.59 vs 1.59, fu 0.25 vs 0.30**; 40 + 16 negative-ET cells excluded from the
ET terms and counted. Every term agrees between fit and hold-out to a few
hundredths -- nothing over-fitted. The domain validation (`val_v12g`, with
ET components) decides whether this is the closing state for carbon and
ET together.

**The ET stage is therefore a joint fit**, not a separate one: the (b4)
structure with the carbon terms as they stand plus `--w-et` and
`--w-et-season`, `soil_beta_power` added to the structural extras,
`vmax`/`specla` free -- one loss carrying the three
carbon criteria, the tree fraction, the ET total and pattern, and the ET
season, on the representative land-weighted sample. Launched after the
(b4) decision on whichever carbon state wins.

### The joint fit over the domain: the carbon terms decided everything, ET moved nowhere

50-year validation (`val_v12g`, years 2056-60, with ET components), against
the two states it descends from:

| | (b3) v12e | (b4) v12f | **joint v12g** | observed |
|---|---|---|---|---|
| GPP PgC/yr, true land / on FLUXCOM land | 132.1 / 128.9 | 118.4 / 115.3 | **118.3 / 115.0 (1.06x)** | 120-130 / 109 |
| NPP true land; veg C true land | 59.6; 281 | 51.1; 207 | 52.3; 225 | 55-60; 450-650 |
| GPP by band 30S-30N / 30-50N / 50-60N / 60-70N / 70-90N | 1.13 / 1.40 / 1.10 / 1.03 / 1.10 | 1.01 / 1.25 / 0.99 / 1.05 / 1.21 | **0.99 / 1.27 / 1.03 / 1.05 / 1.26** | 1 |
| fu natural land / r; forests fu / GPP | 0.355 / 0.66; 0.77 / 0.93 | 0.272 / 0.58; 0.62 / 0.83 | 0.312 / 0.61; 0.67 / 0.85 | 0.335; 0.91 / 1 |
| phantom / missing-tree cells | 3.7 / 7.1 % | 2.1 / 12.9 % | 2.6 / 11.2 % | |
| ET vs MODIS / vs ensemble; WUE | 1.04 / 0.99; 1.13 | 1.02 / 0.99; 1.03 | **1.01 / 0.98; 1.03** | 1 |
| ET by band 15-30N / 30-45N / 15-60S; treeless | 0.87 / 1.16 / 1.29-1.32; 1.33 | 0.87 / 1.16 / 1.29-1.32; 1.32 | 0.87 / 1.15 / 1.27-1.32; 1.30 | 1 |
| T/ET; boreal soil-evaporation share | 0.46; 0.71 | 0.42; 0.70 | 0.45; 0.68 | 0.57-0.64; -- |
| seasonal amplitude all / tropics / 30-50N | 0.82 / 0.96 / 0.62 | 0.81 / 0.95 / 0.62 | 0.80 / 0.94 / 0.62 | 1 |

Two conclusions, one for each stage.

**Carbon.** The joint fit is (b4) with the gate fitted: the same total
(118 on true land, 1.06x FLUXCOM on the land it covers, the bands within
5 % at high latitudes and in the tropics) and forests between (b4)'s and
(b3)'s. Three states now bracket the trade the data leave open -- (b3)
holds the forests and the pools and runs 1.18x on covered land; (b4) and
the joint fit sit at 1.06x and give up a fifth of the forest cover and
the biomass with it. Criterion 1 is met by either reading; criterion 2's
residual is the steppe and grassland belt at 1.3-1.6x (treeless land 1.47x
with the trees gone: the grass itself over-produces) and the forests at
0.85-0.93; criterion 3's residual is the mid-latitude amplitude at 0.6,
untouched by every stage since (a).

**ET.** At weight 1 the ET terms decided nothing: total, pattern and
partition are the same to two decimals as in the two carbon states, and
`soil_beta_power` settled at a mild 1.31. That is not a failure of the
fit: the ET total was already right (0.98-1.01), and the decomposition
(above) had shown that transpiration per unit GPP is fixed by the coupled
scheme, so the ET pattern -- 15-30 N short, the southern mid-latitudes
and treeless land 1.3x high -- *is* the GPP pattern (15-60 S GPP 1.4-1.6x,
treeless 1.5x) in water. The ET stage's remaining error is therefore the
carbon stage's remaining error: the steppe and grassland excess. The
partition (T/ET 0.45, boreal soil evaporation 68 %) is the one ET-specific
residual, and its lever exists (`soil_beta_power`; the joint fit did not
need it at these weights). A dedicated ET pass, if wanted, is a fit with
the ET weights raised on the state that has the steppes right -- which
does not exist yet.

**Closing protocol on the joint result** (`final_validation.sh v12g`,
launched 12:15 on 2026-09-20, ~6 h): the self-consistent 2010 state with
ET components, fire, methane and the trend, to stand beside the (b3)
closing run (section 19) as the two candidate states.

Model ET is `ayaet` (mm/yr) yearly and `amaet` monthly; the fit harness's
`one_year` already returns `ayaet` and the daily AET series, so the ET fit
is the carbon harness with `fluxcom_at` -> an ET loader and the seasonal
term on monthly ET -- built: `et_at`, `et_monthly_at`, `--w-et`,
`--w-et-season`, and the Ball-Berry slope per canopy group as extras
(`coefm_bl/con/shrub/c4/c3`), the lever that moves ET at fixed GPP.

## 16. Criterion 3 diagnosed: the tropical dry season does not end

The seasonal term is flux-weighted, so the tropics carry it, and two fits
have not moved it. Composite monthly cycles of the v12a 10-year monthly
output against FLUXCOM on the 12,476 tropical land cells (|lat| < 23.5,
FLUXCOM > 0.5 kgC m-2 yr-1), each cell aligned to its own wettest month
and classed by the observed seasonal amplitude (`tropical_phase.py`):

| class | cells | model/obs annual | GPP peak after rain peak: obs / model | model/obs by month from wettest (0 ... 11) |
|---|---|---|---|---|
| rainforest (amp < 1.5) | 3369 | **0.87** | -0.6 / +1.0 mo | 1.00 1.04 1.09 1.10 1.07 0.99 0.94 0.94 0.95 0.94 0.96 0.99 |
| seasonal (1.5-3) | 2370 | 1.02 | -0.5 / +1.6 mo | 0.96 1.02 1.10 1.17 **1.20** 1.16 1.04 0.92 0.84 0.85 0.88 0.92 |
| savanna / dry (>= 3) | 6737 | 1.28 | +0.1 / **+1.9 mo** | 0.83 0.88 1.00 1.17 1.42 **1.61** 1.48 1.21 0.98 0.89 0.86 0.85 |

Observed savanna GPP follows the rain almost instantly (best lag +0.4
month, r = 0.93) and collapses in the dry season (amplitude 8.4x); the
model follows *soil moisture* -- peak 1.9 months after the rain, best lag
+1.3, amplitude 5.9x -- so it is low at the wet peak (0.83x) and **1.4-1.6x
observed for the four months after the rains stop**. The seasonal class
shows the same shape one notch weaker. Rainforest phase is right; its
annual GPP is the one tropical number that is *low* (0.87x), which the
sample's flux weighting hides behind the savanna excess. By continent the
lag is everywhere: S America +0.9 month (84 % of cells off by >= 2), Africa
+1.1, Asia/Australia +1.4.

This is the drought deciduousness that section 14 found inert: the F77
trigger on 10-day net assimilation never fires, so tropical deciduous
trees and savanna grasses keep their leaves and transpire deep soil water
through the dry season. It also says why the (b1) optimiser put
`drop_stress_thresh` on its floor: a global CMA-ES on a flux-weighted loss
cannot see a lever whose payoff is confined to the tropical dry season
when the same lever costs GPP wherever temperate soils dry in summer. The
lever has to be designed and tested against this composite directly --
lever test 7 -- and if it works, given a form that acts on the tropical
PFTs' dry season without firing on temperate summers (e.g. a duration
condition, or the trigger restricted to the drought-deciduous PFTs).

Consequences for the criteria: the tropical dry season is where criterion
3 lives; fixing it also removes part of the savanna annual excess
(criterion 1) and, if the rainforest deficit (0.87x) is then visible, the
tropical `vmax` in stage (b2) has a clean target.

### Lever test 7: the drought trigger fires and the dry season stays

60 savanna, 40 seasonal and 30 rainforest cells from the fit sample, the
(b1) structure (`params_fit_v12c.yaml`), 6 years cycled, the soil-water
trigger `drop_stress_thresh` at 0.05 (off, as fitted) / 0.3 / 0.5 / 0.7,
composites aligned to each cell's wettest month:

| savanna, months from wettest | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | amp | dropu / dropl4 | fu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| observed shape | .156 | .155 | .130 | .090 | .056 | .036 | .028 | .031 | .042 | .063 | .089 | .124 | 5.5 | | |
| trigger off, model/obs | 0.78 | 0.80 | 0.96 | 1.19 | 1.53 | **1.82** | 1.77 | 1.39 | 1.02 | 0.87 | 0.89 | 0.83 | 2.9 | 1.00 / 0.99 | 0.68 |
| trigger 0.5 | 0.82 | 0.84 | 0.99 | 1.18 | 1.50 | 1.70 | 1.58 | 1.22 | 0.92 | 0.84 | 0.91 | 0.86 | 3.4 | 0.77 / 0.65 | 0.74 |
| trigger 0.7 | 0.83 | 0.85 | 1.00 | 1.17 | 1.48 | 1.68 | 1.59 | 1.25 | 0.93 | 0.82 | 0.90 | 0.85 | 3.4 | 0.58 / 0.59 | 0.77 |
| 0.5 + `drop_floor` 0.02 | 0.83 | 0.85 | 1.00 | 1.18 | 1.48 | 1.68 | 1.52 | 1.17 | 0.89 | 0.83 | 0.91 | 0.87 | 3.6 | 0.76 / 0.62 | 0.77 |

The trigger fires -- the drought-deciduous trees and C4 grasses lose 40 %
of their leaf display -- and the dry-season excess falls by a tenth. The
seasonal class does not move (1.24 -> 1.25 at month 4); the rainforest
class does not move (0.89x annual). So the leaves that drop are not the
leaves carrying the dry-season GPP. What does not drop is the *evergreen*
upper canopy, and `fu` in these savanna cells is 0.68 -- rising to 0.77
under the trigger, through the same feedback as US-Ton (less leaf, less
transpiration, less stress, more tree survival). **Hypothesis, to be
tested by decomposition:** the savanna dry season is tropical evergreen
trees (PFT 1) standing where savannas have grass and deciduous trees,
drawing deep soil water through the dry season; the annual-mean gate cannot
reach them because the wet season lifts their mean stress above the
threshold. If so the lever is the gate's stress *measure* (the dry-season
minimum rather than the annual mean, or the evergreen-vs-deciduous
competition), not the phenology trigger.

### Decomposition: the persistence is in every plant, so it is in the soil

Same 60 savanna cells, (b1) structure, counterfactual removals (PFT pools
zeroed and `existence` blocked), years 4-6 (`savanna_decomp.py`):

| case | LAI by PFT | annual model/obs | model/obs shape, months 0-1 / 4-7 from wettest | fu |
|---|---|---|---|---|
| as fitted | trop. deciduous 1.75, C4 grass 0.33, evergreen 0.01 | 1.06 | 0.78 0.80 / 1.53 1.82 1.77 1.39 | 0.68 |
| no evergreen trees (PFT 1) | deciduous 1.27, C4 0.37 | 0.80 | 0.76 0.77 / 1.52 1.89 1.90 1.49 | 0.71 |
| no tropical deciduous trees (PFT 2) | C4 grass 1.51, shrubs 0.15 | 0.61 | 0.73 0.75 / 1.51 1.84 1.84 1.45 | 0.06 |
| no upper canopy | C4 grass 1.58 | 0.58 | 0.73 0.75 / 1.51 1.84 1.84 1.45 | 0.04 |
| no lower canopy | deciduous 1.92 | 1.01 | 0.74 0.76 / 1.63 2.01 1.94 1.49 | 0.79 |

The evergreen hypothesis is wrong: these cells are tropical *deciduous*
trees over sparse C4 grass. And the finding is stronger than any
composition story: **every counterfactual has the same wrong seasonality**
-- trees alone, grass alone, both -- late peak at 0.73-0.78x, 1.5-2.0x
through the four months after the rains. Whatever holds GPP up in the dry
season holds it up for every plant in the cell, which puts it below the
vegetation: the soil water the plants draw on. That also says why leaf
drop in lever test 7 barely mattered -- the plants that kept leaves had
water to use them -- and it makes the next test an independent one: model
ET against the ET ensemble through the same composite year. If model ET
persists where the products fall, the memory is in the soil column (depth,
drainage, root profile, or the stress function), not in phenology.

Two side notes. Grass alone reaches only 0.58x the observed GPP (C4 `vmax`
15.4 umol, the low end of its bounds): a stage-(b2) target once the
seasonality is right. Removing the 0.01-LAI evergreen PFT cost 25 % of
GPP in every month (deciduous LAI 1.75 -> 1.27) -- not understood, noted.

### The independent check: ET says the soil is fine and the leaves are not

Same 60 cells, v12a 10-year monthly output, model ET against the
five-product ET ensemble, soil water by layer (`savanna_water.py`):

| months from wettest | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| precipitation mm/mo | 266 | 197 | 130 | 73 | 31 | 14 | 15 | 23 | 41 | 68 | 120 | 195 |
| ET model mm/mo | 103 | 103 | 100 | 82 | 60 | 44 | 36 | 38 | 44 | 60 | 82 | 97 |
| ET ensemble mean | 96 | 99 | 92 | 74 | 53 | 38 | 32 | 31 | 37 | 49 | 65 | 83 |
| ET model / ensemble | 1.08 | 1.04 | 1.08 | 1.11 | 1.14 | 1.16 | 1.14 | 1.22 | 1.22 | 1.23 | 1.26 | 1.17 |
| GPP model / FLUXCOM (shape) | 0.84 | 0.89 | 1.00 | 1.15 | 1.37 | 1.61 | 1.59 | 1.35 | 1.02 | 0.82 | 0.82 | 0.83 |
| soil water, layers 1-3 (frac. sat.) | .78 | .76 | .67 | .57 | .50 | .44 | .42 | .42 | .44 | .49 | .58 | .70 |
| layers 5-6 | .82 | .84 | .84 | .82 | .80 | .78 | .76 | .76 | .75 | .75 | .75 | .77 |

The soil-water hypothesis fails the independent test. Model ET has the
*same* seasonal shape as the observed ET (trough at months 6-7 in both;
amplitude 2.9 against 3.2), running 8-26 % high throughout; the top three
layers dry from 0.78 to 0.42 of saturation and only the deep layers stay
wet, as deep soils do. What differs is the *observations' own* ratio:
observed GPP collapses far more than observed ET (amplitude 5.5 against
3.2) -- the plants stop assimilating while the surface keeps evaporating --
whereas model GPP tracks model ET (2.9 against 2.9; WUE flat through the
year). That is leaf-area control of GPP in reality and its absence in the
model, and it explains lever test 7: dropping 40 % of the leaves cost 10 %
of GPP because the fitted canopy is light-saturated -- `trop_bl_dec` at
vmax 80.5 umol over LAI 1.75, the dlnGPP/dlnLAI of 0.09 measured in section
14. A canopy that gets its GPP from leaf capacity rather than leaf area
cannot have its seasons driven by leaf phenology, however well the
phenology fires. Lever test 8 measures the LAI-sensitivity directly (LAI
x0.5, x2) and whether trading capacity for area (tropical vmax x0.5 with
LAI x2) restores the amplitude.

### Lever test 8 and the stress the plants feel

Same cells, (b1) structure (`lever_test8.py`, `savanna_stress.py`):

| case | annual model/obs | amplitude | GPP vs as-fitted, wet peak / dry trough |
|---|---|---|---|
| as fitted | 1.06 | 2.9 | 1.00 / 1.00 |
| LAI x0.5 or x2 at start | 1.06 | 2.9 | 1.00 / 1.00 -- LAI re-equilibrates to 2.1 within the adjust years; the test measured nothing |
| tropical vmax x0.5 | **0.59** | 2.3 | 0.52 / 0.66 -- GPP is proportional to vmax: capacity-limited, and the shape gets worse |
| `stressfac` -2.5 | 1.00 | 3.0 | 0.95 / 0.93 -- no change of shape |

The plants are stressed after all -- section 16's earlier reading of the
soil column used the wrong wilting point. `swilt` here is 0.37-0.45 of
saturation, so the dry-season topsoil at 0.42 is plant-available water
0.10-0.18, and the root-weighted stress factor (roots: lower canopy 92 %
in the top three layers, upper canopy 72 %) runs, by month from the
wettest:

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| awc, layers 1-3 | .94 | .89 | .72 | .52 | .35 | .20 | .15 | .13 | .21 | .35 | .61 | .85 |
| stress, lower canopy, `stressfac` -8.27 | .99 | .99 | .91 | .74 | .61 | .46 | .42 | .41 | .52 | .66 | .87 | .97 |
| upper canopy, -8.27 | .98 | .98 | .92 | .80 | .69 | .57 | .52 | .51 | .58 | .68 | .85 | .96 |
| lower canopy, -2.5 | .97 | .95 | .82 | .63 | .47 | .31 | .27 | .25 | .36 | .50 | .74 | .92 |

A stress cycle of 0.99 -> 0.41 is what gives the model its amplitude of
2.9; the observed 5.5 needs the leaves to go as well, and with stress at
0.4-0.5 for four months the trigger at 0.7 in lever test 7 should have
fired for all of them. Whether the displayed LAI actually fell, and what
GPP did month by month, is measured next (lever test 9) rather than read
off a year-end `dropu`.

### Lever test 9: the leaves go, the GPP stays -- dlnGPP/dlnLAI = 0.08

Same cells, monthly state recorded inside `one_year` (`lever_test9.py`):

| trigger 0.7, months from wettest | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| displayed LAI, upper (off: 2.08 all year) | 2.08 | 2.09 | 2.00 | 1.78 | 1.47 | 1.04 | 0.76 | **0.61** | 0.78 | 1.07 | 1.56 | 1.92 |
| displayed LAI, lower (off: 1.13-1.26) | 0.96 | 0.92 | 0.69 | 0.37 | 0.27 | **0.21** | 0.23 | 0.30 | 0.29 | 0.43 | 0.65 | 0.82 |
| dropu / dropl4 | .96/.97 | .96/.96 | .87/.84 | .70/.66 | .53/.49 | .34/.33 | .26/.25 | **.22/.22** | .28/.28 | .40/.41 | .66/.67 | .86/.89 |
| stress upper (off: 0.36 at month 7) | .94 | .95 | .88 | .79 | .70 | .59 | .51 | .49 | .56 | .65 | .78 | .90 |
| GPP model/obs, trigger off | 0.78 | 0.80 | 0.96 | 1.19 | 1.53 | 1.82 | 1.77 | 1.39 | 1.02 | 0.87 | 0.89 | 0.83 |
| GPP model/obs, trigger 0.7 | 0.83 | 0.85 | 1.00 | 1.17 | 1.48 | **1.68** | **1.59** | 1.25 | 0.93 | 0.82 | 0.90 | 0.85 |

The phenology lever works as a lever on leaf area: the trigger fires when
the stress says so, the deciduous trees lose 71 % of their displayed LAI
and the grass 83 %, with the observed timing. And the canopy loses 10 % of
its GPP. dlnGPP/dlnLAI = 0.08 -- the number section 14 measured on
temperate cells, now on tropical ones with the leaf area varied by the
model's own phenology. Per-leaf assimilation must have tripled to hold GPP
up; the light per leaf can rise 37 % at most and the stress relief is 36 %,
so neither explains it. The observed savanna amplitude (5.5) is
unreachable by any phenology while the canopy is this indifferent to its
leaf area; the same indifference is why every fit has reached for vmax
instead of LAI. Whether it is IBIS's own canopy formulation or a port
defect in the canopy scaling is measured next at the timestep level (lever
test 11: one month from one state with only the displayed LAI scaled).

### Lever test 11: the canopy is optically black -- and the stems photosynthesise

One dry-season month from one saved state, 15 savanna cells in that month,
only `plai` scaled (`pheno` rebuilds the displayed `lai` from it daily;
`sai` is set annually and so stays at 0.56) -- `lever_test11b.py`:

| plai x | 0.1 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| month GPP, ratio to x1 | **0.579** | 0.777 | 0.907 | 1 | 1.014 |
| dlnGPP/dlnLAI | 0.24 | 0.18 | 0.14 | | 0.02 |
| ET, ratio | 0.62 | 0.76 | 0.88 | 1 | 1.12 |

(The first version scaled `veg.lai` and got 1.000 everywhere: the state's
`lai` is not a control, `pheno` overwrites it each morning.)

Doubling the leaf area adds 1.4 %; removing 90 % of it leaves 58 % of the
GPP. F77's canopy scaling (`physiology.f` 436-492, ported line for line)
scales the top-leaf rate by (1 - exp(-k(L+S))) / kL and the accounting
multiplies back by L, so canopy GPP is proportional to 1 - exp(-k(L+S)):
the PAR absorbed by **leaves and stems** divided by the **leaf** area --
every photon the stems absorb is credited to leaf photosynthesis. With S
held at 0.56, 1 - exp(-k(fL+S)) reproduces the whole curve at k = 1.0
(f = 0.1: 0.578 predicted, 0.579 measured). Two consequences. With leaves
on, a flat over-credit of S/L -- 25 % for the upper canopy, whose SAI is
0.25 LAI -- everywhere, all the time. With leaves dropping, the stems keep
assimilating: at L = 0.22 they carry most of the canopy. That is why lever
test 9's 71 % leaf drop cost 10 % of GPP, why no phenology lever has ever
moved the seasonal term, why the fits reach for vmax and not LAI, and why
the boreal band now has the right GPP from half the leaves.

The reference treatment is CLM's: stems attenuate light, only leaf area
assimilates. Here that is one denominator -- k(L+S) instead of kL, i.e.
the old scale x L/(L+S), an exact identity -- implemented as
`StomataParams.scale_leaf_only` (static, default False = F77 bit-for-bit;
`stomata: scale_leaf_only: true`; `test_vmax_mix.py` pins both). The
separate fact that k is about 1.0 -- the canopy 94 % absorbing at LAI 2.2
where a clumped savanna canopy absorbs perhaps 60 % -- is IBIS's
unclumped two-stream and is left alone for now; the stem credit is a
defect, the extinction a modelling choice.

What follows: the causal chain trigger -> leaf area -> GPP is tested with
the correction on (lever test 12), and the domain-wide size of removing
the stem credit is read on a two-year run. Every fit from here runs with
both `vmax_mix` and `scale_leaf_only` on, and the calibration is redone on
the corrected canopy: the fitted vmax carried a 25 % stem credit.

### Lever test 12: with the stems out of the photosynthesis, the trigger reaches the flux

Same 60 savanna cells, (b1) structure, `vmax_mix` on with behaviour-
preserving starts (`lever_test12.py`):

| case | annual model/obs | amplitude (obs 5.5) | model/obs shape, months 0-1 / 4-7 from wettest | dry-season GPP vs as-fitted (month 7) |
|---|---|---|---|---|
| as fitted (F77 canopy) | 1.06 | 2.9 | 0.78 0.80 / 1.53 1.82 1.77 1.39 | 1.00 |
| `scale_leaf_only` | 0.89 | 2.7 | 0.76 0.78 / 1.53 1.88 1.86 1.46 | 0.89 |
| leaf-only + trigger 0.7 | 0.84 | **3.8** | 0.84 0.86 / 1.49 **1.67 1.48 1.12** | **0.67** |
| leaf-only + trigger 0.9 | 0.77 | 4.1 | 0.87 0.89 / 1.45 1.54 1.39 1.08 | 0.57 |
| leaf-only + trigger 0.7 + `stressfac` -5 | 0.81 | 4.0 | 0.86 0.88 / 1.46 1.60 1.40 1.08 | 0.62 |

The same trigger that bought a tenth of the dry-season excess on the F77
canopy (lever test 7) now takes a third of it: the chain trigger -> leaf
area -> GPP is live. The correction alone costs the savanna 17 % of its
annual GPP (the stem credit), and the trigger at 0.9 already over-fires
into the wet season (annual 0.77), so the remaining shape error -- wet
peak at 0.85, months 4-5 still 1.5-1.7x, the peak two months after the
rain -- is now a matter of magnitude (the (b2) refit of tropical vmax) and
of the k = 1 extinction, not of a lever that cannot act. Whether the
two-month phase lag is all the model's is also an observation-side
question: FLUXCOM's savanna phase rests on its remote-sensing drivers, and
that is checked against tower data before the phase is fitted.

### The observation side: towers confirm FLUXCOM's phase and dispute its dry-season depth

Fourteen FLUXNET2015 savanna / woody-savanna / grassland / shrub towers
with |lat| < 30 and monthly `GPP_NT_VUT_REF` and `P_F` (ten Australian:
AU-ASM, Ade, DaP, DaS, Dry, Emr, How, RDF, Stp, TTE; plus CG-Tch, PA-SPs,
SD-Dem, US-KS2), each aligned to its own wettest month; FLUXCOM and the
model (v12a) read at the tower's cell (`tower_phase.py`):

| | GPP peak minus rain peak (months, mean of sites) | composite amplitude (peak / dry-season month) | dry-season months 5-7, fraction of annual |
|---|---|---|---|
| towers | **+0.21** | 2.9 | 0.048-0.051 |
| FLUXCOM at the cells | +0.21 | 5.8 | 0.029-0.031 |
| model at the cells | **+1.14** | 3.6 | 0.042-0.053 |

Two readings. The **phase** is FLUXCOM's: tower GPP peaks with the rain,
and the model's lag is the model's own. The **depth** is not: at these
towers FLUXCOM's dry-season GPP is 60-64 % of the towers' (FLUXCOM's
remote-sensing drivers see the leafless canopy; the towers see the
understory and the soil still assimilating), so the composite amplitude
is 2.9 at the towers against 5.8 in FLUXCOM, with the model's 3.6 between
them. Against the towers the model's dry season is about right and its
error is timing -- too low in months 2-3, too high in 4-6; against FLUXCOM
it is a factor two too flat. With ten of fourteen sites Australian this is
not the last word, but it is enough to say the tropical seasonal
amplitude in FLUXCOM carries real uncertainty and should not be fitted to
the last digit: the seasonal term's weight stays where it is, and the
savanna phase (a month of lag), not the depth, is the part of criterion 3
to hold the model to.

### Criterion 3's other half: the mid-latitude winter, and the grass that never goes dormant

The tropics' phase lag (above) is one half of the seasonal residual; the
other is the 30-60 N amplitude at 0.6, unmoved by every stage since (a).
On the joint fit's monthly output, 16,801 cells at 30-60 N with FLUXCOM >
0.2, flux-weighted (`midlat_season.py`):

| | J | F | M | A | M | J | J | A | S | O | N | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model / FLUXCOM, all 30-60 N | **4.3** | 3.3 | 2.3 | 1.4 | 0.83 | 0.61 | 0.60 | 0.76 | 1.3 | 2.6 | **4.5** | **4.9** |
| model GPP, kgC m-2 mo-1 | .042 | .045 | .071 | .096 | .114 | .114 | .117 | .114 | .110 | .099 | .069 | .046 |
| FLUXCOM | .010 | .014 | .031 | .068 | .137 | .186 | .196 | .151 | .083 | .038 | .015 | .009 |
| displayed LAI, all | 4.6 | 4.6 | 4.8 | 4.8 | 4.8 | 4.7 | 4.6 | 4.5 | 4.7 | 4.8 | 4.8 | 4.6 |
| C4-grass-dominant cells (4,156): model / obs | 2.9 | 2.3 | 1.8 | 1.2 | 1.0 | 0.83 | 0.73 | 0.80 | 1.3 | 2.3 | 3.4 | 3.3 |
| their displayed LAI | 5.6 | 5.8 | 6.0 | 5.9 | 5.6 | 5.2 | 4.7 | 4.5 | 5.1 | 5.4 | 5.8 | 5.7 |
| deciduous-shrub-dominant (1,786): model / obs | 2.7 | 2.2 | 1.6 | 1.0 | 0.59 | 0.47 | 0.48 | 0.60 | 1.0 | 2.0 | 3.3 | 3.2 |
| their displayed LAI | 5.5 | 5.7 | 6.4 | 6.5 | 6.5 | 6.5 | 6.5 | 6.4 | 6.4 | 6.5 | 6.5 | 5.7 |

The mid-latitude model has **no winter**: 4-5x the observed GPP in
November-February, 0.6x in June-July, and a displayed leaf area that does
not change through the year -- deciduous shrubs at LAI 5.5-6.5 in January,
grasses at 5.6. Two mechanisms, both structural, both now identified.

1. **The lower canopy goes dormant only below 0 C.** `pheno`'s cold trigger
   for grasses and shrubs is `a10td < tthresh_l_c + tmelt` with
   `tthresh_l_c` = 0 (F77), an absolute threshold at freezing: wherever
   the ten-day mean stays above 0 C -- most of 30-45 N, all winter -- the
   understory keeps its leaves and assimilates. Stage (a) had found
   exactly this and pushed `tthresh_l_c` to its +8 C bound (section 14);
   I dropped it when stage (b1) restarted from v11d and never put it
   back among the extras. It goes back in the next fit, with
   `gdd_thresh_l`.
2. **C3 grass is extinct.** At the joint fit's state C3 grass is the
   dominant PFT in **zero** cells on Earth; C4 grass dominates 4,156 cells
   at 30-60 N, 1,727 of them at 40-60 N (the Kazakh and Mongolian steppes,
   Ukraine: C4 LAI 1.6, C3 0.01). The lower-canopy competition is by NPP,
   the fitted C4 vmax (21.5) beats C3's (16.2, at its floor) wherever C4
   is allowed to exist, and the existence rule allows it on cool steppes.
   Real C3 steppes at 45-55 N are being simulated as C4 grassland -- with
   C4's temperature response and without dormancy -- which is part of the
   1.35x at 30-50 N and the winter excess. The fix is on the existence
   side (a C4 climate envelope: warm-season temperature *and* the cold
   season) or on the competition side; a lever test decides.

Together with the tropical phase lag (the dormancy trigger that reaches
the flux only on the corrected canopy), these are criterion 3 in full:
the mid-latitude half is a winter the model does not have.

### Lever test 13: neither the C4 envelope nor the grass threshold reaches the winter -- the shrubs do

Sixty C4-dominated steppe cells at 40-55 N (warmest month 23.7 C, coldest
-7.0 C; FLUXCOM 0.52 kgC m-2 yr-1, winter GPP ~0), the joint-fit
parameters, 8 years cycled, years 6-8 (`lever_test13.py`):

| case | dominant PFT: C4 / C3 / shrub (of 60) | annual | Nov-Feb model/obs | Jun-Jul | amplitude (obs 47) | Dec displayed LAI |
|---|---|---|---|---|---|---|
| as fitted (`Twarm` 22, lower-canopy threshold 0 C) | 56 / 0 / 4 | 0.98 | **5.0** | 0.58 | 5.7 | 3.3 |
| `Twarm` 24 | 23 / 0 / **33** | 0.90 | 5.6 | 0.47 | 4.0 | 3.6 |
| `Twarm` 26 (C4 excluded) | 7 / 0 / **49** | 0.84 | 5.7 | 0.40 | 3.4 | 3.8 |
| `tthresh_l_c` +5 C | 52 / 0 / 6 | 0.95 | 5.1 | 0.56 | 4.9 | 3.5 |
| `tthresh_l_c` +8 C | 35 / 0 / 22 | 0.92 | 5.4 | 0.50 | 4.1 | 3.9 |
| `Twarm` 26 + `tthresh_l_c` +8 | 7 / 0 / 49 | 0.83 | 5.8 | 0.39 | 3.2 | 4.4 |

Both levers made the winter worse, and the reason is the same in every
row: **C3 grass wins nowhere** -- when C4 is excluded or the grasses are
made dormant earlier, the lower canopy goes to the **evergreen shrub**
(PFT 9: vmax 35.7 after the fits, no cold phenology at all, existence
wherever gdd0 > 100), whose leaf area then stands at 3.5-4.4 through a
-7 C winter and assimilates in every mild spell. The mid-latitude winter
is therefore a lower-canopy competition problem, not a threshold: the
real steppe is C3 grass because of cold tolerance and disturbance, and
IBIS's lower canopy decides by NPP among a C3 grass whose vmax the fits
have pushed to its floor (it dominates nowhere, so no loss can see it),
a C4 grass allowed north to 55 N by a warm-month rule alone, and an
evergreen shrub with a fitted vmax of 35.7 and no winter. Lever test 14
removes the evergreen classes to attribute the winter GPP before any fix
is designed.

### Lever test 14: the evergreens are innocent; the deciduous lower canopy itself assimilates in winter

Same 60 cells, PFT classes removed (pools zeroed, existence blocked),
years 6-8 (`lever_test14.py`):

| case | dominant: C4 / C3 / evergreen shrub / deciduous shrub | annual | Nov-Feb | Jun-Jul |
|---|---|---|---|---|
| as fitted | 56 / 0 / 1 / 3 | 0.98 | 4.97 | 0.58 |
| no evergreen shrub | 56 / 0 / 0 / 4 | 0.99 | 4.95 | 0.59 |
| no evergreen conifers | 56 / 0 / 1 / 3 | 0.98 | 4.98 | 0.58 |
| no evergreens at all | 56 / 0 / 0 / 4 | 0.99 | 4.96 | 0.59 |
| no C4 grass | 0 / 0 / 1 / **55** | 0.83 | **5.83** | 0.39 |
| no C4, no evergreens | 0 / 0 / 0 / 56 | 0.83 | 5.88 | 0.39 |

Removing every evergreen PFT changes the winter by nothing (4.97 -> 4.96),
so section 16's previous paragraph read the wrong culprit: the "shrub" that
took the steppe in lever test 13 was the **deciduous** shrub (PFT 10), and
with it the winter excess grew (5.83). The winter GPP is produced by the
deciduous lower canopy -- C4 grass or deciduous shrub -- which means
either its cold phenology (`templ`, the shared trigger of grasses and
deciduous shrubs) is not shutting it, or a floor keeps assimilating leaf
area on the ground after it does. (A reading caveat for lever tests 13
and 14: the "December LAI" columns are end-of-year states after `dynaveg`
has reset the display to the structural LAI, and say nothing about
winter.) Lever test 15 records the displayed LAI, `templ`, `tempu`, the
ten-day temperature and GPP at every month-end inside the year to settle
it.

### Lever test 15: the ten-day temperature never saw winter -- a port defect in the phenology clock

Month-end state inside the year on the same 60 steppe cells, joint-fit
parameters (`lever_test15.py`):

| as fitted, slow clock | J | F | M | A | M | J | J | A | S | O | N | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GSWP3 air temperature 2002 (C) | -2.7 | | | 10.0 | | | 24.9 | | | 10.0 | | |
| **`a10td` (C)** | **8.3** | 7.1 | 6.7 | 7.1 | 8.3 | 9.7 | 11.5 | 12.9 | 13.4 | 13.0 | 11.9 | 9.9 |
| `templ` | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| displayed LAI, lower | 3.2 | 3.6 | 4.4 | 4.5 | 4.3 | 4.3 | 3.8 | 3.4 | 3.7 | 4.1 | 4.5 | 3.3 |
| GPP model/obs | 4.5 | 3.3 | 2.0 | 1.1 | 0.77 | 0.61 | 0.55 | 0.66 | 1.4 | 3.2 | 7.1 | 5.0 |

The ten-day mean temperature the phenology reads was **+8 C in January on
cells whose January air is -3 C** and 13 C in September: an annual cycle
of 7 C against the weather's 28, peaking in September. `pheno`'s cold
trigger (`a10td < 0 C`) could not fire; `templ` sat at 1.00 all year and
the grass kept 3-4 m2/m2 of displayed leaf area through a -7 C winter.

The cause is in the port, not in IBIS. F77 (`stats.f`, `sumday`) updates
the ten-day means at *every timestep* with `zweight = exp(-dtime/(10 d))`
-- 24 applications a day for hourly steps, one day's decay exp(-1/10), a
ten-day memory. The port (`loops_gswp3._day_body`) updates them *once a
day* with that same per-timestep weight, exp(-1/240): a **240-day**
memory, which attenuates an annual cycle to 24 % of its amplitude and
lags it 77 days -- exactly the +8 C January. Every short-window
verification passed, because the per-day difference is 0.4 %; the effect
takes a season to show. The same weight fed the `a10anc*` drought
triggers, which therefore averaged eight months of assimilation and could
not see a dry season either -- part of why section 16's dormancy lever
was so hard to make act. `canopy_scaling`'s `a10scalparam` runs per
timestep and was right. Fixed to the daily-equivalent weight (guarded;
`DEFECTS_FOR_REVIEW.md`).

| as fitted, **fixed clock** | J | F | M | A | M | J | J | A | S | O | N | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `a10td` (C) | **-3.1** | -1.9 | 4.3 | 11.7 | 17.8 | 21.6 | **24.3** | 21.8 | 15.8 | 8.5 | 2.6 | -4.0 |
| `templ` | 0.32 | 0.40 | 0.70 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.89 | 0.35 |
| `tempu` | 0.00 | 0.00 | 0.24 | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.72 | 0.21 | 0.00 |
| displayed LAI, lower | 1.8 | 2.1 | 3.2 | 4.2 | 4.0 | 4.0 | 3.6 | 3.2 | 3.5 | 3.9 | 3.9 | 1.9 |
| GPP model/obs | 3.9 | 2.8 | 1.8 | 1.2 | 0.88 | 0.70 | 0.63 | 0.70 | 1.5 | 3.4 | 6.8 | 4.7 |
| with `tthresh_l_c` +8 C: LAI lower / GPP model/obs | 0.9 / 3.7 | 0.9 / 2.4 | 1.7 / 1.4 | 3.5 / 0.95 | | | | | | 3.0 / 3.5 | 1.7 / 6.3 | 0.9 / 4.4 |

The seasons now reach the phenology: the upper canopy is bare from
December to February, the grass at a third of its leaf area. And the
winter GPP is still 3-4x observed with a displayed lower LAI of 0.9-2.1 --
so the phenology was one of two problems. What remains is physiological:
a lower canopy at LAI 0.9 assimilating 11-14 g C m-2 a month at a ten-day
mean of -3 C. Lever test 17 measures GPP per unit displayed leaf area by
month and takes the lower canopy out altogether to attribute it (the
canopy floor `fl_min`, the C4 temperature response, the k = 1
extinction are the candidates). Every calibration stage through v12g ran
on the slow clock; the joint-fit parameters are being re-validated on the
corrected model (`val_v12g_clock`) before anything is closed.

### Lever test 17: the lower canopy carries the winter, at a third of its summer rate per leaf, at -3 C

Fixed clock, same 60 steppe cells, years 6-8 (`lever_test17.py`):

| case | annual | Nov-Feb model/obs | Jun-Jul | displayed LAI Jan (lower / upper) | GPP per unit displayed LAI, Jan / Jul (gC m-2 leaf mo-1) |
|---|---|---|---|---|---|
| as fitted, fixed clock | 1.04 | **4.5** | 0.66 | 1.8 / 0.4 | **5.2** / 16.2 |
| `fl_min` 0.05 | 0.67 | 2.7 | 0.43 | 1.9 / 0.7 | 2.5 / 8.0 |
| `tthresh_l_c` +8 C | 1.00 | 4.2 | 0.66 | 0.9 / 0.5 | 7.9 / 16.3 |
| C4 vmax x0.5 | 0.83 | 4.1 | 0.51 | 1.6 / 0.6 | 5.0 / 13.2 |
| **no lower canopy** | 0.46 | **1.4** | 0.31 | 0 / 1.0 | 3.2 / 19.2 |

Three readings. (1) The winter GPP is the lower canopy's: without it,
Nov-Feb falls from 4.5x to 1.4x (and summer to 0.31x -- the grass is also
most of the summer). (2) Cutting its leaf area does not cut its winter
GPP: at LAI 0.9 (`tthresh_l_c` +8) it assimilates *more per leaf* (7.9)
than at 1.8 (5.2) -- the light-saturated canopy of section 16 again, on
its own scale. (3) The rate itself is the finding: **5-8 gC per m2 of
leaf per month at a ten-day mean of -3 C**, a third to half of the 16 it
does in July at 24 C. `fl_min` acts only because it scales the whole
canopy's cover (summer falls with winter, 0.67 annual); it is not a
winter lever. So the mid-latitude winter is physiological: a C4 grass
photosynthesising at freezing on days that are individually below zero.
The candidates, in order: the C4 temperature function `tempvm_c4` (F77's
`3.0**((T-15)/10) / ((1+exp(0.3(13-T))) (1+exp(0.3(T-36))))`) at 0 C and
below -- checked next -- and, behind it, a C4 grass on steppes that are
C3 in reality (the existence rule of section 16), which the fixed clock
does not change.

### Lever test 16: the tropical dry season is not the clock's

The same savanna / seasonal / rainforest composites as lever tests 7 and
12, on the fixed clock with the joint-fit parameters (`lever_test16.py`):

| savanna, months from wettest | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | annual | amplitude (obs 5.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trigger 0.277 (as fitted) | 0.74 | 0.76 | 0.94 | 1.17 | 1.50 | **1.86** | **1.90** | 1.44 | 1.08 | 0.94 | 0.94 | 0.83 | 0.78 | 2.7 |
| trigger 0.7 | 0.77 | 0.79 | 0.97 | 1.18 | 1.47 | 1.79 | 1.81 | 1.34 | 0.99 | 0.92 | 0.94 | 0.84 | 0.84 | 3.1 |

The corrected clock changes the savanna dry season by nothing (compare
lever test 12's 1.67 / 1.48 at months 5-6 with the trigger at 0.7 on the
(b3) state): in the tropics the ten-day mean was never far from the daily
temperature, and the drought trigger reads soil stress, not `a10td`. The
tropical phase lag is therefore its own problem, as section 16 first
read it: the leaf area drops (`dropu` 0.76 at 0.7) and the flux barely
follows, on a canopy whose k = 1 extinction leaves it light-saturated at
the leaf area that remains. Two side readings: the joint fit's tropical
`vmax` was tuned on the slow clock and reads 0.78x (savanna) and 0.77x
(rainforest) on the fixed one over these 6 transient years, and the
seasonal class's wet-season excess (1.28 at months 4-5) is the same shape
one notch weaker.

### Lever test 18: the winter GPP comes from days at 0-10 C, at half the summer rate

Year-6 daily GPP on the 60 steppe cells (fixed clock, joint-fit
parameters) binned by the day's mean air temperature (`lever_test18.py`;
a sampling caveat: the forcing reader hands back daily means, and the
script's hourly reshape kept only 12 days per cell per month, so the
counts are a sample and the rates the reading):

| daily mean air T | share of days | GPP on those days, gC m-2 d-1 | share of annual GPP | share of Nov-Feb GPP |
|---|---|---|---|---|
| below -5 C | 13 % | 0.00-0.03 | 0 | 1 % |
| -5 .. 0 C | 11 % | **0.45** | 3 % | **19 %** |
| 0 .. +5 C | 13 % | **0.91** | 8 % | **42 %** |
| +5 .. +10 C | 13 % | 1.48 | 13 % | 36 % |
| +10 .. +20 C | 27 % | 1.8-2.2 | 39 % | 1 % |
| above 20 C | 23 % | 2.30 | 36 % | 0 |

The winter excess is not a phenology-clock artefact and not a mild-day
artefact in the ordinary sense: on days whose mean air temperature is
0-5 C the grass assimilates 0.9 gC m-2 -- 40 % of its rate on a 25 C
summer day -- and on days at -5 to 0 C still 0.45. The leaf temperature
function says a C4 leaf at 0-5 C runs at 1-8 % of its 25 C rate
(`tempvm_c4` 0.009 at 0 C, 0.077 at 5 C; C3 0.06 and 0.32), so either
the leaves are far warmer than the air on those days -- winter sun on a
low canopy in the port's turbulence scheme -- or the rate is not coming
through `tempvm` at all. The next probe is at the timestep: leaf
temperature against air temperature, `tempvm`, PAR and the assimilation
rate, hour by hour on a January day on these cells. That is a physics
investigation (the canopy energy balance in winter), the fourth found by
the seasonal criterion after the stem credit, the dead vmax entries and
the clock.

### Lever test 19: the winter leaf is honest -- and the summer soil is frozen

Hour by hour on 15 January and 15 July 2003, twenty of the steppe cells,
fixed clock, joint-fit parameters, driving `timestep_body` directly with
the model's own diurnal forcing (`lever_test19.py`):

| 15 January (air -3.7 C, tmax 0.4) | 06 | 09 | 12 | 14 | 17 | 21 |
|---|---|---|---|---|---|---|
| air / lower-canopy leaf T (C) | -7.7 / -7.5 | -4.6 / -1.3 | -0.6 / **3.6** | 0.4 / **4.2** | -1.6 / 0.1 | -5.5 / -5.1 |
| `tempvm_c4`(leaf) | 0.007 | 0.059 | 0.177 | 0.193 | 0.084 | 0.016 |
| PAR at the lower canopy (W m-2) | 0 | 91 | 118 | 107 | 48 | 0 |
| C4 net assimilation (umol m-2 leaf s-1) | 0 | 0.61 | 1.39 | 1.45 | 0.46 | 0 |

Daily gross assimilation **0.45 gC per m2 of leaf** (July: 5.6), or ~0.26
gC m-2 ground per day at the cell's LAI 1.16 and cover 0.25 -- 0.008
kgC m-2 for the month against FLUXCOM's 0.003. So the winter leaf is
honest: warmed 3-4 C above the air by the midday sun, it assimilates at
a twelfth of its July rate through the temperature function as written;
the "4-5x" mid-latitude winter is a *small absolute* excess (0.005-0.01
kgC m-2 mo-1) on an observation near zero, which is why the flux-weighted
seasonal term never moved on it, and the amplitude criterion at 30-60 N
(0.6) is carried by the **summer deficit** (June-July 0.6-0.7x), not the
winter. That reorders the mid-latitude problem: the grass under-produces
in summer and stays slightly on in winter; the clock fix has put the
phenology right and what remains is the summer -- the steppe canopy's
light use, and the C4-for-C3 substitution of section 16.

The "second finding" was the probe's, not the model's -- corrected the
same hour. The July day in `lever_test19.py` was run from the state the
January day had left (no intervening months), so its soil column, snow
and canopy temperatures were January's: the 1-3 C "July" soil surface and
the 3-9 C night-time leaf were a January soil under a July sky. The
monthly output of the same cells (`val_v12g/monthly`) has the top soil
layer at 25.9 C in July, `tg` 21.7 C, no summer ice; the 2060 restart
carries 19 % ice in the top layer only on 31 December. The January
numbers above stand (that day followed five full years and the January of
the sixth). Lesson recorded: a single-day probe must start from a state
at that date.

### The joint-fit parameters on the corrected clock: the season improves, the boreal vmax is now wrong

50-year validation of `params_fit_v12g.yaml` with the phenology clock fixed
(`val_v12g_clock`, years 2056-60), against the same parameters on the slow
clock (`val_v12g`):

| | slow clock | **fixed clock** | observed |
|---|---|---|---|
| GPP PgC/yr, true land / on FLUXCOM land | 118.3 / 115.0 | **123.6 / 120.3** (1.10x) | 120-130 / 109 |
| NPP true land; veg C true land | 52.3; 225 | 55.6; 264 | 55-60; 450-650 |
| GPP by band 30S-30N / 30-50N / 50-60N / 60-70N / 70-90N | 0.99 / 1.27 / 1.03 / 1.05 / 1.26 | 0.99 / 1.33 / **1.38 / 1.59 / 1.34** | 1 |
| LAI by band, same | 1.30 / 1.35 / 0.82 / 0.60 / 0.75 | 1.30 / 1.13 / **0.49 / 0.35** / 0.64 | 1 |
| fu natural land / r | 0.312 / 0.61 | **0.333 / 0.60** | 0.335 |
| fu 60-90N / 45-60N | 0.25 / 0.53 | 0.37 / 0.57 | 0.25 / 0.45 |
| seasonal amplitude all / 30-50N / 50-60N / 60-70N / 70-90N | 0.80 / 0.61 / 0.58 / 0.63 / 0.75 | **0.83 / 0.65 / 0.71 / 0.79 / 0.78** | 1 |
| shape error, flux-weighted | 0.0169 | **0.0161** (lowest of any state) | |
| 30-60N composite: model/obs Jan / Jul; LAI Jan / Jul | 4.3 / 0.60; 4.6 / 4.6 | 4.2 / **0.75**; **3.5 / 4.6** | 1 / 1 |

The clock fix does what lever test 15 said it would: the mid-latitude and
boreal canopies now have a seasonal cycle of leaf area (30-60 N LAI 3.5 in
January against 4.6 in July, where it was flat) and the amplitude improves
in every band north of 30 -- the seasonal criterion's best reading so far.
The small-absolute winter excess is unchanged (4.2x on ~0.01 kgC m-2).
And the boreal GPP has gone from right to 1.4-1.6x with **half the leaf
area** (60-70 N LAI 0.35x MODIS, GPP 1.59x): on the slow clock the boreal
evergreens and deciduous PFTs carried displayed leaves through a
phenology that never turned off, and their `vmax` was fitted to that; on
the corrected clock the same `vmax` on a shorter, sharper season -- and
`boreal_bl_dec` at 49.9 from the provenance flaw -- over-produces per
leaf while the canopy is thinner. Vegetation carbon rose (225 -> 264) with
the trees returning at high latitude (fu 60-90 N 0.37 against 0.25: too
many now).

So every parameter fitted so far was fitted to a model with the wrong
seasonal clock, and the next fit is a refit on the corrected model rather
than a refinement -- stage (d): the joint fit's scoring and structure,
`tthresh_l_c` and `gdd_thresh_l` back among the extras (the grass
threshold matters again now that `a10td` reaches it), the `vmax_mix`
starts corrected (`dec_shrub` 28.0, `boreal_bl_dec` 20.3), and the
C3-versus-C4 existence question (section 16) carried as a lever test
before it enters the structure. One start value is set by hand: C3
grass `vmax` 25.0 (the F77 default) instead of the 16.2 the earlier fits
had walked it to. C3 is dominant in no sample cell, so `--min-cells 30`
freezes it at its start; an unidentifiable parameter belongs at the
default, not at a bound the optimiser reached on noise (lever test 21
below shows the choice does not decide the steppe season either way).

### Lever test 20: bounding C4 out of the cool steppes hands them to shrubs, not to C3 grass

Fixed clock, the 60 C4-dominated 40-55 N steppe cells (coldest-month mean
-7 C, 10-90 % -15 to 0 C), joint-fit parameters, a cold-season bound
added to the C4 existence rule (`lever_test20.py`):

| C4 allowed where | dominant: C4 / C3 / evergreen shrub / deciduous shrub | annual | Jun-Jul | amplitude (obs 47) |
|---|---|---|---|---|
| `tw` > 22 C only (F77, as fitted) | 53 / 0 / 1 / 3 | 1.04 | 0.66 | 7.1 |
| ... and `tc` > -5 C | 24 / 0 / 4 / 28 | 0.98 | 0.58 | 5.9 |
| ... and `tc` > 0 C | 7 / 0 / 4 / 44 | 0.95 | 0.53 | 4.9 |
| ... and `tc` > +5 C | 0 / 0 / 4 / 51 | 0.91 | 0.50 | 4.6 |

C3 grass wins **nowhere** whatever is done to C4; the cells go to the
deciduous shrub and the summer deficit deepens (0.66 -> 0.50). The C3
grass has a fitted `vmax` of 16.2 -- its floor -- because it dominated no
sample cell in any stage and no loss could ever see it: a PFT that cannot
establish cannot be fitted, the same trap as the dead `vmax` entries of
section 17, one level up (there the entries were never read; here the PFT
is read but never present). The existence rule is not the lever; the
competition is, and it is decided by NPP with a C3 `vmax` that the fits
starved. Lever test 21 gives C3 the F77 default (25) and above, seeds it,
and asks whether it takes the cool steppes and what that does to their
summer.

### Lever test 21: C3 grass does not fix the steppe summer either -- the grass type is not the lever

Same cells, fixed clock, C3 grass seeded (LAI 0.3) and given `vmax` 16.2
(the fitted floor), 25 (F77), 30, 35 against C4's 21.5 (`lever_test21.py`):

| C3 vmax | dominant: C4 / C3 | annual | Jun-Jul | amplitude | LAI C4 / C3 |
|---|---|---|---|---|---|
| 16.2 | 51 / 0 | 1.04 | 0.67 | 6.9 | 0.98 / 0.02 |
| 25.0 | 51 / 2 | 1.01 | 0.64 | 6.7 | 0.89 / 0.11 |
| 30.0 | 49 / 7 | 1.00 | 0.62 | 6.4 | 0.80 / 0.20 |
| 35.0 | 45 / 11 | 1.00 | 0.61 | 6.1 | 0.69 / 0.33 |

Even at a `vmax` above C4's, C3 takes a sixth of the cells and the
**summer deficit does not move** (0.61-0.67 whichever grass grows). The
C3-for-C4 question of section 16 is closed as a lever: the steppe's
midsummer under-production is not the grass type. What the two grass
tests leave is the canopy itself -- a lower canopy at LAI ~1.0 and cover
0.25 on cells whose observed GPP is 0.52 kgC m-2 yr-1, with k ~ 1
extinction and a quarter of the ground under leaves: the light the cell
receives in June-July falls mostly on bare soil. Lever test 22 varies the
cover at fixed leaf area (`fl_min` 0.25 -> 1.0, `cover_k`) and reads the
July light partition between lower canopy, upper canopy and soil.

### Lever test 22: cover scales the whole year -- the steppe season is a timing problem

Same cells, fixed clock, the lower-canopy cover floor raised at fixed
leaf carbon (`fl_min` 0.25 as fitted -> 1.0; `cover_k` 1.0 =
`fl = 1 - exp(-LAI)`), `lever_test22.py`:

| case | annual | Jun-Jul | Nov-Feb | amplitude (obs 47) | July cover / displayed lower LAI |
|---|---|---|---|---|---|
| as fitted (`fl_min` 0.25, `cover_k` 0.20) | 1.04 | 0.66 | 4.5 | 7.1 | 0.25 / 3.6 |
| `fl_min` 0.50 | 1.49 | 0.92 | 7.2 | 6.8 | 0.50 / 2.4 |
| `fl_min` 0.75 | 1.88 | 1.10 | 9.7 | 6.4 | 0.75 / 1.9 |
| `fl_min` 1.0 | 2.21 | 1.24 | 12.2 | 5.9 | 1.00 / 1.6 |
| `cover_k` 1.0 | 2.04 | 1.18 | 10.8 | 6.2 | 0.78 / 1.6 |

Spreading the same leaf area over more ground doubles the light the
grass absorbs (a quarter of the cell under leaves at LAI 3.6 absorbs ~24 %
of the incoming light; the same leaves at cover 1 absorb ~50 %) and GPP
follows -- in **every month alike**. The summer reaches the observation
only when the winter is twelve times it. The amplitude moves from 7 to 6
against an observed 47. So neither the grass type (tests 20-21) nor the
canopy geometry (this test) is the steppe lever: the model's steppe cycle
is too flat because its **timing** is wrong. Month by month (as fitted):
March 1.75x and April 1.15x (leaf-out a month early), June-July 0.66x,
September 1.45x, October 3.4x, November 6.8x (senescence a month or two
late). The observed GPP collapses in September-October at daily means of
+5 to +10 C; the model's lower canopy senesces when the ten-day mean
falls below `tthresh_l_c` = 0 C, which on these cells is November. Stage
(a) had fitted that threshold to +8 C and stage (b1) dropped it; stage
(d) carries `tthresh_l_c` and `gdd_thresh_l` (leaf-out) among its
extras, and the fixed clock now lets them act. The light-partition
columns of the test read zero (the month-end state is a night step);
the numbers above rest on the GPP response, which is the quantity of
interest.

### The second forcing defect: the physics ran at sea-level pressure on every cell (found and fixed 2026-09-21)

The open negative-ET defect (section 15) was localised while stage (d)
waited for the GPU. In the joint-fit closing hindcast 18 % of all
cell-months have negative total ET and 33,169 cells have at least one
negative month; the worst are Himalayan monsoon slopes with -175 to
-205 mm in July-August and a **negative annual mean** under a full forest
canopy and a ground at 17-20 C. The forcing at the worst cell (31.25 N,
78.75 E) is ordinary: August air 13.7 C, 13.5 g/kg, surface pressure
583 hPa, relative humidity 78 %. Read at sea-level pressure the same
humidity is 136 % with a dew point of 18.9 C -- and the model's ground
temperature there, 19.6 C, is that dew point: the surface condensed and
warmed until its own saturation met the air. The GSWP3 driver had read
`ps` into the daily weather and used it in `diurnal` to cap the hourly
humidity, but the hourly atmosphere update never wrote `atm.psurf`, which
stayed at its creation value 101325 Pa on every cell of every run since
the driver was written (every restart's `leaf_110` is 101325 throughout).
Saturation humidity scales as 1/p, so every cell above sea level ran with
the leaf's relative humidity too high (3 % at 300 m, 11 % at 1000 m,
17 % at 1500 m, 40 % at 4000 m) and the air density too high by the same
factor: wider stomata and stronger turbulent exchange on every plateau.
DEFECTS_FOR_REVIEW.md "sea-level pressure" has the full account; the fix
is one line in `loops_gswp3._step_body` (guard
`test_hourly_atmosphere_carries_the_forcing_surface_pressure`).

Probe (`diagnostics/psurf_probe.py`): the six worst cells and seven
controls, two years from the 2010 transient state on the joint-fit
parameters, against the same cells in the (unfixed) closing hindcast:

| cell | pressure now seen (hPa) | ET, fixed (mm/yr; Aug) | ET, hindcast (mm/yr; Aug) |
|---|---|---|---|
| 31.2 N 78.8 E Himalaya | 576 | 682; 90 | -775; -161 |
| 31.2 N 79.2 E | 553 | 776; 112 | -1107; -150 |
| 27.8 N 92.2 E | 598 | 597; 73 | -1112; -169 |
| 28.8 N 93.8 E | 613 | 723; 76 | -1000; -155 |
| 32.8 N 77.2 E | 568 | 538; 76 | -941; -108 |
| 27.8 N 90.8 E | 628 | 697; 91 | -767; -135 |
| Tibet 33 N 90 E (4800 m) | 557 | 516 | 199 |
| Mongolia 47 N 105 E (1300 m) | 876 | 183 | 245 |
| Great Basin 40 N 116 W (1600 m) | 815 | 179 | 238 |
| Kenya 0 N 37 E (1700 m) | 810 | 492 | 634 |
| Amazon 3 S 60 W (sea level) | 1004 | 1310 | 1328 |
| Kazakh steppe 50 N 70 E (300 m) | 993 | 390 | 329 |
| Iowa 42 N 93 W (300 m) | 976 | 732 | 647 |

Exposure (GSWP3 July-2005 pressure, true-land-area weights): the
physics' saturation humidity was low by a factor 1013/p averaging 1.12
over land (1.07 weighted by FLUXCOM GPP); 28 % of the land, carrying 15 %
of GPP, lies above ~1000 m with a mean error of 31 %; the 30-45 N band --
Tibet, Iran and Anatolia, the Great Basin, the Mexican plateau -- has
46 % of its land above 1000 m and an area-mean factor of 1.18, and it is
the band where every stage has read GPP 1.25-1.54x (section 18). The
tropics and the boreal plain average 1.07.

Every month on the six worst cells is now positive (17-112 mm), with
transpiration and interception the largest terms, as a wet monsoon forest
should have. The controls do NOT move one way: the sea-level Amazon is
unchanged (1328 -> 1311), Mongolia, the Great Basin and Kenya lose a
fifth to a quarter (245 -> 183, 238 -> 179, 634 -> 492), and **Tibet
rises from 199 to 516 mm/yr**, two thirds of it soil evaporation (351 of
516 mm). "Narrower stomata" therefore explains at most the three
declines; the Tibetan rise is the removal of the same condensation that
dominated the Himalayan cells (its hindcast ET was negative from November
to April). And none of these before/after pairs is an attribution: the
"before" is the 2001-2010 hindcast from a cold spin-up, the "after" two
years from the v11 2010 state, so state and years differ as well as the
pressure. GPP on the probe cells is also far from the observation either
way (Himalaya #2 5.8x, Tibet 2.9x, Kazakh 2.3x, Iowa 2.0x, Kenya 0.04x
FLUXCOM after two years from an un-equilibrated state): a positive ET
says nothing about the ecosystem on that cell. The controlled
comparison that separates the two fixes from state and years -- same
cells, same forcing years, same start state, same parameters, no fix /
clock only / pressure only / both -- is the next entry. The domain-scale consequences
-- ET and GPP on plateaus and mountain ranges, the ET pattern of section
15 (15-30 N 0.84, which is largely the Tibetan-Himalayan and Mexican
plateaus), and the fitted stomatal and stress parameters that
compensated -- are read from stage (d) and its validation, the first runs
on both corrected forcings.

### Controlled attribution of the two forcing fixes (2026-09-22, `diagnostics/controlled_4way.py`)

Same 123 cells in seven groups (30 C4 steppe 40-55 N, 20 boreal forest,
20 temperate forest, 20 tropical savanna, 20 rainforest, the 6 Himalayan
condensation cells, 7 plateau and sea-level controls), same 2001-2005
forcing, same 2010 start state, same parameters (defaults, and the joint
fit); four model variants built from the current source by substituting
the two fixed lines back: no fix / clock only / pressure only / both. Mean
of years 4-5. Output `diagnostics/controlled_4way_pass1.txt`; figures in
the local document folder (F9, F10).

- **Pressure fix alone**: the six Himalayan cells go from -280..-877 mm/yr
  of ET to +539..+855 and Tibet from -53 to 540, while Mongolia, the Great
  Basin, Kenya, Kazakhstan, Iowa and the Amazon change by 0-5 %. The
  "plateau ET down 20 %" of the earlier before/after probe was state and
  years, not the pressure. GPP falls with elevation by the stomatal
  humidity mechanism: -49 % at 1600 m (Great Basin), -40 % at 1700 m
  (Kenya), -27 % at 1300 m (Mongolia), 0-7 % near sea level; by group
  (defaults) steppe 2.39 -> 2.26, savanna 1.61 -> 1.52, boreal 1.70 ->
  1.66, rainforest unchanged. ET rises where the surface had condensed
  (boreal group 1.03 -> 1.19, temperate 0.87 -> 1.01).
- **Clock fix alone**: `a10td` January/July on the steppes 8.0/11.0 ->
  -5.0/23.5 C, boreal -1.0/2.1 -> -12.7/15.5, temperate 1.2/4.5 ->
  -15.9/18.6. Temperate-forest amplitude ratio 0.04 -> 0.41 (joint fit
  0.09 -> 0.86); with the joint-fit parameters the temperate annual GPP
  goes 0.41 -> 0.98 and the boreal 0.93 -> 1.44 (the source of the
  fixed-clock validation's boreal 1.6x). Steppe Nov-Feb excess 16.1 ->
  12.2 (joint fit 5.4 -> 4.3): smaller, still a non-dormant grass.
  Tropics unchanged.
- **Both fixes, defaults, by group** (annual / Jun-Jul / Nov-Feb GPP
  ratio; ET ratio): steppe 2.04 / 1.17 / 11.7; 1.37. Boreal 1.66 / 1.09
  / 137; 1.20. Temperate 1.01 / 0.66 / 10.9; 1.04. Savanna 1.53 / 1.49 /
  1.49; 1.28. Rainforest 1.12 / 1.09 / 1.20; 0.95. The fixed-composition
  pass (section 20) separates the vegetation the model grows from the
  physiology of what should be there.

## 17. Dead parameters in the vmax table: the upper canopy's vmax is chosen by existence, not composition

`stomata._select_vmax_upper` (F77 `stomata.f` design): the upper canopy is
two big leaves, broadleaf and conifer, each with ONE vmax picked by
existence priority -- broadleaf: tropical evergreen (PFT 1) if it can
exist, else warm-temperate evergreen (3), else temperate deciduous (5);
conifer: temperate evergreen (4) else boreal evergreen (6). Two
consequences the calibration has been living with:

1. **Wherever tropical evergreen can exist the whole broadleaf canopy
   photosynthesises at its vmax**, whatever actually grows there. In the
   savanna cells the canopy is tropical *deciduous* at LAI 1.75, the
   evergreen has LAI 0.01, and the canopy runs at PFT 1's 40 umol. Remove
   PFT 1 and the canopy falls to warm-temperate's 25: the unexplained 25 %
   GPP loss in the decomposition above.
2. **The fitted vmax of trop_bl_dec (80.5), boreal_bl_dec (46.5),
   boreal_con_dec (30.0) and dec_shrub (27.5) are never used** -- the lower
   canopy's shrub big leaf reads `vmax_pft[8]` (evergreen shrub) for both
   shrub PFTs, the grasses their own. Four of the twelve "per-PFT"
   parameters the v11c/v11d/(b) fits carried were dead; CMA-ES walked them
   on noise (trop_bl_dec to 80.5 against a 100 ceiling, for nothing). The
   same class as the fifteen defect-0 findings, inside the parameter table
   this time.

The remedy is a composition-weighted canopy vmax -- broadleaf
vmax = sum(frac_j vmax_j) over the broadleaf PFTs present, conifer
likewise, shrubs likewise, with `frac` the LAI share pheno already
computes -- behind an explicit switch (`vmax_mix`), off by default so F77
behaviour is kept, on in every fitted configuration; plus a guard that
every entry of `vmax_pft` reaches GPP. Until it is in, the vmax vector has
eight live entries, not twelve, and trop_bl_dec's 80.5 means nothing.

**Implemented (commit 68cff90).** `StomataParams.vmax_mix` (static, default
False = F77, bit-for-bit -- `test_vmax_mix.py` pins the priority pick and
the unchanged `test_stomata.py` suite passes): when on, each big leaf takes
the displayed-LAI-weighted mean of vmax over its PFTs (broadleaf {1,2,3,5,7},
conifer {4,6,8}, shrub {9,10}, 1-based), with the priority pick where no
leaves exist; `stomata()` takes `frac`, the single call site passes
`veg.frac`, the yaml key is `stomata: vmax_mix: true`. Guards: one-hot
canopies return each PFT's own vmax, so all twelve entries reach GPP.

Two rules for every fit from here. (1) `vmax_mix: true` in the start yaml.
(2) The four newly-live entries start at the value their big leaf has in
fact been running at, so switching the scheme on changes nothing until the
optimiser moves them: trop_bl_dec <- trop_bl_ever's fitted 40.1 (savannas
have been running at it), boreal_bl_dec <- temp_bl_dec's 26.9 (the
priority fallback where tropical and warm-temperate cannot exist),
boreal_con_dec <- boreal_con_ever's 25.0, dec_shrub <- ever_shrub's 28.0.
Their fitted v11d values (80.5, 46.5, 30.0, 27.5) are discarded as the
noise they were.

**Proved end to end (lever test 10, 60 savanna cells, (b1) structure, 6
years):** with the switch off, `trop_bl_dec` vmax x1.5 changes GPP by
0.0000 in every month -- the entry is behaviourally dead; with the switch
on and the behaviour-preserving starts, GPP is again identical to four
digits -- the start rule is exact where the priority pick coincides with
the dominant PFT; with the switch on and x1.5, GPP rises 25-30 % (annual
1.06 -> 1.33 of observed, LAI 1.76 -> 2.14). Where the priority pick and
the dominant PFT differ (temperate deciduous forest with a rare
warm-temperate evergreen present: 25.4 picked, 26.9 dominant) switching on
changes GPP by the vmax ratio, ~6 %; the domain-wide size of that jump is
read on the first `vmax_mix` validation.

## 18. The corrected canopy over the domain: the total lands in range before any refit

Two-year runs from the v11 2010 state with the (b1) structure -- identical
except for the canopy switches -- against the 50-year (b1) validation
(`leafonly_probe`, `state_report`, `fu_vs_tree`):

| | (b1) structure, F77 canopy (2 yr) | **(b1) structure, `vmax_mix` + `scale_leaf_only` (2 yr)** | (b1) 50-yr validation | observed |
|---|---|---|---|---|
| GPP PgC/yr | 167.7 | **130.6** | 163.7 | 120-130 |
| NPP PgC/yr | 76.2 | 53.4 | 77.3 | 55-60 |
| ET mm/yr | 504 | 494 | 526 | 500-550 |
| LAI mean | 2.63 | 2.31 | 1.84 | |
| GPP by band 30S-30N / 30-50N / 50-60N / 60-70N | 1.10 / 1.52 / 1.52 / 1.64 | **0.92 / 0.87 / 0.92 / 1.24** | 1.10 / 1.55 / 1.30 / 1.01 | 1 |
| GPP by tree class: < 0.05 / 0.15-0.30 / 0.5-0.7 / > 0.7 | 1.62 / 1.40 / 1.27 / 0.98 (gate probe) | **1.30 / 1.12 / 0.96 / 0.73** | 1.53 / 1.35 / 1.24 / 0.98 | 1 |
| fu natural land / MAD | 0.52 / 0.288 | 0.55 / 0.302 | 0.53 / 0.280 | 0.335 |

Removing the stem credit takes 22 % off global GPP with nothing else
changed, and the total lands inside the observed range with the vmax
values the fits had tuned *with* the credit. The pattern turns over: the
upper-canopy-dominated forests, which carried the 25 % credit, fall to
0.73x, while treeless land, which carried 5 %, stays at 1.30x -- so the
first thing a refit will do is raise the tree vmax and lower the grass
vmax, which is what a per-PFT vmax should have been doing all along and
could not while four entries were dead and the canopy indifferent to
leaf area. Two-year numbers are transients (LAI is still falling toward
the corrected canopy's lower NPP; pools are the v11 pools), which is why
the fit and not the probe sets the values.

Stage (b2), launched 08:30 on 2026-09-18 (`fit_carbon_v12d`): both
switches on; joint `vmax`/`specla` (24, all live) plus the eight
structural scalars, PFT blocks free; start from the (b1) yaml with the
newly-live vmax at their behaviour-preserving values, `cover_k` back at
0.5 so understory cover can vary, `tthresh_u_abs` and the dormancy
trigger reset to neutral (0 C, 0.5); `--w-tree 3` so the tree term binds;
4000 cells, 25 + 3 years, 180 evaluations (~10 h). `chain_v12d.sh`
validates and scores at the end -- pools, bands, tree cover, ET, season.

### Stage (b2) result: 1.8464 -> 1.5620; the best carbon terms yet, at the lowest tree term yet

180 evaluations (08:26 -> 19:27; 187-212 s each, one slow stretch around
eval 155). Best at eval 117: GPP 0.2427, LAI 0.6148, season 0.0177, tree
0.278, hold-out 1.738. On the three carbon terms that every stage shares
(GPP + 0.5 LAI + 10 season, same sample): v11d 0.872, stage (a) 0.776,
stage (b1) 0.758, **stage (b2) 0.727** -- and the tree term, which (b1)
could not hold (0.327), is 0.278 under the weight of 3.

| scalar | start | fitted | bounds | |
|---|---|---|---|---|
| `tree_stress_min` / `ramp` | 0.365 / 0.136 | **0.713 / 0.100** | [0.1, 0.9] / [0.05, 0.5] | a much stronger, sharper gate: trees survive only on soils wet 71 % of the year |
| `fu_min` | 0.042 | 0.065 | [0.01, 0.25] | |
| `cover_k` / `fl_min` | 0.5 / 0.249 | 0.546 / 0.250 | | understory cover as before; `fl_min` still at its ceiling |
| `drop_stress_thresh` | 0.5 | **0.748** | [0.05, 0.85] | the dormancy trigger is live and used, now that leaf drop reaches the flux |
| `tthresh_u_abs` | 0 C | 5.2 C | [-5, 15] | a moderate value, not the ceiling stage (b1) took |
| `stressfac` | -8.27 | -8.11 | [-10, -2.5] | unchanged |

| PFT | vmax start -> fitted (umol) | specla start -> fitted | |
|---|---|---|---|
| trop_bl_ever | 40.1 -> 41.1 | 10.1 -> 13.7 | |
| trop_bl_dec | 40.1 -> 40.5 | 11.5 -> 12.1 | live for the first time; stays at the evergreen's value |
| warm_bl_ever | 25.4 -> 30.1 | 19.4 -> 19.9 | |
| temp_con_ever | 32.4 -> 29.2 | 9.4 -> 12.0 | |
| temp_bl_dec | 26.9 -> **57.6** | 14.5 -> 12.0 | the forests that lost the 25 % stem credit; near its 60 ceiling |
| boreal_con_ever, boreal_con_dec, dec_shrub | 25.0, 25.0, 28.0 (frozen: too few dominant cells) | | |
| boreal_bl_dec | 26.9 -> 20.7 | 34.1 -> 31.3 | live for the first time |
| ever_shrub | 28.0 -> 33.4 | 7.5 -> 8.4 | |
| c4_grass / c3_grass | 15.4 -> 14.5 / 18.4 -> 21.1 | 10.2 -> 10.5 / 15.6 -> 13.6 | |

Read together: with the stem credit gone the optimiser did what a
per-PFT vmax was supposed to do -- raised the temperate deciduous forest,
left the tropical broadleaf canopy where the priority pick had in fact been
running it, trimmed the grasses -- and, with a tree weight that binds,
pushed the gate hard enough to matter and used the dormancy trigger it
had switched off on the old canopy. The domain validation (`chain_v12d`)
says whether this is the state to close the carbon stage on.

### Stage (b2) over the domain: the total comes down, the forests come down with it

50-year validation (`val_v12d`, years 2056-60), against the (b1) validation:

| | v12c (b1) | **v12d (b2)** | observed |
|---|---|---|---|
| GPP PgC/yr | 163.6 | **147.7** | 120-130 |
| NPP / Rh PgC/yr | 77.4 / 73.1 | 67.5 / 63.6 | 55-60 / 49-58 |
| ET mm/yr | 523 | 504 | 500-550 |
| **veg C PgC** | 570 | **304** | **450-650** |
| soil C / litter | 12.0 / 0.72 | 11.6 / 0.59 | 10-15 / 0.4-0.8 |
| GPP by band 30S-30N / 30-50N / 50-60N / 60-70N / 70-90N | 1.10 / 1.54 / 1.30 / 1.01 / 0.69 | **0.98** / 1.39 / 1.21 / 1.12 / 1.08 | 1 |
| LAI by band, same | 1.23 / 1.61 / 1.01 / 0.55 / 0.38 | 1.08 / 1.45 / 0.91 / 0.62 / 0.61 | 1 |
| fu natural land / MAD / r | 0.530 / 0.280 / 0.61 | **0.289 / 0.241 / 0.47** | 0.335 / 0 / 1 |
| fu where tree < 0.05 / 0.15-0.30 / > 0.70 | 0.18 / 0.63 / 0.88 | **0.10 / 0.25 / 0.58** | 0.01 / 0.22 / 0.91 |
| fu 60-90N / 45-60N / 30-45N | 0.24 / 0.67 / 0.47 | **0.09 / 0.24** / 0.27 | 0.25 / 0.45 / 0.18 |
| phantom cells / missing-tree cells, share of natural land | 10.9 % / 3.4 % | 2.4 % / **14.1 %** | |
| seasonal amplitude all / tropics / 30-50N | 0.78 / 0.90 / 0.60 | 0.81 / 0.96 / 0.60 | 1 |
| ET natural land / WUE | 1.07 / 1.12 | 1.03 / 1.04 | 1 / 1 |

The tropics are right on every field (GPP 0.98, LAI 1.08, phase improved
to 0.96 amplitude), the phantom steppe trees are mostly gone (2.4 % of land
from 11 %), ET and WUE are within a few percent -- and **the gate at 0.71 has
removed a third of the world's forest**: forest cover 0.58 against 0.91,
the boreal forest reduced to fu 0.09 against 0.25, "missing trees" on 14 %
of natural land, vegetation carbon at 304 PgC against 450-650. The
optimiser was paid for it: on the sample the mid-latitude phantoms it
killed outweighed the boreal forests it killed in the MAD, and the missing
forests' GPP is replaced by grass at 0.95x -- the canopy's indifference to
what grows the leaves, one more time.

The criterion, not the weight, is at fault. `drystress` counts only liquid
water (`wsoi (1 - wisoi)`), so frozen boreal soils read as fully stressed
for half the year and the annual-mean stress of a healthy boreal forest
sits below any threshold that removes steppe trees. At 0.365 (b1) this was
masked; at 0.713 it is the whole story. The survival stress has to be the
growing-season mean, not the calendar mean -- the (b3) fix, measured
first (below) and then refit. v12d is not a state to close the carbon
stage on; its structural scalars and vmax carry the missing forests.

### Fire and methane on the (b2) state (informative, not decisive: the forests are missing)

`post_v12d`: fire (GFED5, 2001-05 from the 2060 state) burned area
**2.45 %** against GFED5 4.60 % (product spread 2.38-4.60; v11d 3.10 %),
rank correlation +0.750 all / +0.324 burning, NMSE 0.860, mean |lat| of
the top-2000 cells 8.9 against 10.8 -- less fuel with a third of the forest
and its litter gone. Methane (GMB): `biome_tref` **189.7 Tg/yr** (v11d 251;
GMB 158 [119-203]), split 60 / 29 / 6 % against 70 / 21 / 8, shape error
19.6; `biome_noq10` 284 Tg with the best shape (12.4). Magnitude tracks Rh
(63.6 PgC on v12d against 73.1 on v12c), as section 12 said it would. Both
move again with (b3), which is where they are weighed.

### Measured: the annual mean cannot tell a boreal forest from a steppe; the growing-season mean can

From the v12d monthly climatology (`wsoi`, `tsoi`, `tg`), the root-weighted
upper-canopy stress factor with liquid water only (frozen layers read as
dry, as `drystress` does), at the (b2) `stressfac` -8.11; growing-season
months = monthly mean ground temperature above 5 C:

| class (natural cells) | n | frozen months | growing months | stress: annual mean | growing-season mean | cells the 0.71 gate kills: annual / growing-season |
|---|---|---|---|---|---|---|
| boreal forest 55-70N, tree > 0.5 | 5199 | 4.8 | 4.7 | **0.63** | **0.96** | **76 % / 0 %** |
| temperate forest 35-55N, tree > 0.5 | 4417 | 3.0 | 6.4 | 0.77 | 0.96 | 47 % / 1 % |
| tropical forest, tree > 0.7 | 4905 | 0 | 12 | 0.94 | 0.94 | 5 % / 5 % |
| savanna, 0.15 < tree < 0.5 | 5364 | 0 | 12 | 0.81 | 0.81 | 16 % / 16 % |
| phantom steppe 30-50N, tree < 0.1 | 8852 | 1.7 | 7.5 | 0.62 | **0.69** | 66 % / 46 % |
| S-hemisphere steppe, tree < 0.1 | 3780 | 0 | 11.5 | 0.74 | 0.74 | 33 % / 33 % |

On the annual mean a boreal forest (0.63) and a steppe (0.62) are the same
number, so no threshold can keep one and remove the other -- which is what
(b2) found out the expensive way. On the growing-season mean the forests
sit at 0.94-0.96 and the steppes at 0.69-0.74: a gate near 0.80-0.85 with
the ramp keeps every forest and removes most of the phantom trees. That is
the (b3) change: the survival stress accumulates only on days whose mean
temperature exceeds `stress_accum_tmin` (5 C; the default -100 counts
every day and is bit-identical), the gate and the PFT blocks are refit
from the (b2) values with the gate restarted at 0.80 / 0.10.

### Stage (b3): the same fit on the growing-season criterion

Launched 20:54 on 2026-09-18 (`fit_carbon_v12e`; the first attempt failed
on a missing exec bit and launched nothing -- caught within a minute, kit
relaunched through `bash`). Everything as (b2) -- corrected canopy, joint
`vmax`/`specla` plus the eight structural scalars, `--w-tree 3`, 180
evaluations -- with one change of physics and two of starting point: the
survival stress accumulates on days above 5 C (`stress_accum_tmin`), the
gate restarts at 0.80 / 0.10 (between the steppes' 0.69-0.74 and the
forests' 0.94-0.96 on that criterion), and the PFT blocks start from the
(b2) values, which the fit is expected to walk back where they compensated
for missing forests (`temp_bl_dec` 57.6). Judged on the sample against
(b2)'s 1.5620 and, on the domain, on whether the forests return (fu > 0.7
class, boreal band, vegetation carbon) while the phantoms stay gone. Then
the closing protocol (section 19) on whichever of (b2)/(b3) the validation
prefers.

Baseline at the (b3) starting point, 21:28: **1.4527** -- below (b2)'s
fitted best of 1.5620 before a single optimiser step. GPP 0.2425 and LAI
0.616 are (b2)'s numbers to the third digit; the seasonal term 0.0180; the
tree term **0.278 -> 0.241** from the change of criterion and the 0.80
restart alone. The criterion was the fault, as the measurement said.

### Stage (b3) result: 1.4527 -> 1.4155; the gate settles where the optimiser wants it, not where the steppes are

180 evaluations (20:54 -> 08:00). Best at eval 175: GPP **0.2223** (the
lowest of any fit), LAI 0.606, season 0.0173, tree 0.239 (= the start),
hold-out 1.634. Carbon terms 0.698 (b2: 0.727, b1: 0.758, a: 0.776, v11d:
0.872).

| scalar | start | fitted | |
|---|---|---|---|
| `tree_stress_min` / `ramp` | 0.80 / 0.10 | **0.630 / 0.159** | forests (growing-season stress 0.94-0.96) fully survive; 30-50N steppes (0.69) survive at ~0.4, S-hemisphere steppes (0.74) at ~0.7 |
| `drop_stress_thresh` | 0.748 | **0.134** | the dormancy trigger switched off again |
| `tthresh_u_abs` | 5.2 C | **12.5 C** | the temperature leaf-drop preferred instead, as in (a) and (b1) |
| `stressfac` | -8.11 | -6.53 | |
| `cover_k` / `fl_min` / `fu_min` | 0.546 / 0.250 / 0.065 | 0.428 / 0.229 / 0.033 | `fl_min` off its ceiling for the first time |

| PFT | vmax (b2) -> (b3) | specla | |
|---|---|---|---|
| trop_bl_ever | 41.1 -> **52.1** | 13.7 -> 14.9 | |
| trop_bl_dec | 40.5 -> 40.3 | 12.1 -> 11.2 | |
| warm_bl_ever | 30.1 -> 28.0 | 19.9 -> 18.2 | |
| temp_con_ever | 29.2 -> **17.6** | 12.0 -> 14.9 | near its 15 floor |
| temp_bl_dec | 57.6 -> 58.3 | 12.0 -> 14.1 | at its 60 ceiling |
| boreal_bl_dec | 20.7 -> 20.3 | 31.3 -> 30.1 | |
| ever_shrub | 33.4 -> 35.7 | 8.4 -> 8.0 | |
| c4_grass / c3_grass | 14.5 -> 18.2 / 21.1 -> 17.0 | 10.5 -> 10.2 / 13.6 -> 11.3 | |
| boreal_con_ever, boreal_con_dec, dec_shrub | frozen (too few dominant cells) | | |

With the criterion able to tell forest from steppe, the optimiser chose
not to remove the steppe trees fully: the tree term did not move from its
start while GPP and season improved, so at a tree weight of 3 the phantom
trees are still worth more to the loss as GPP and LAI than they cost as
cover. The domain validation says how much of the forest is back and how
much of the steppe forest returned with it; the closing decision follows
it.

### Stage (b3) over the domain: the forests are back, the mean cover is right -- and the total went back up

50-year validation (`val_v12e`):

| | v12c (b1) | v12d (b2) | **v12e (b3)** | observed |
|---|---|---|---|---|
| GPP PgC/yr | 163.6 | 147.7 | **163.6** | 120-130 |
| NPP / Rh | 77.4 / 73.1 | 67.5 / 63.6 | 73.7 / 67.9 | 55-60 / 49-58 |
| ET mm/yr | 523 | 504 | 510 | 500-550 |
| veg C PgC | 570 | 304 | **361** | 450-650 |
| soil C / litter | 12.0 / 0.72 | 11.6 / 0.59 | 11.6 / 0.59 | 10-15 / 0.4-0.8 |
| GPP by band 30S-30N / 30-50N / 50-60N / 60-70N / 70-90N | 1.10 / 1.54 / 1.30 / 1.01 / 0.69 | 0.98 / 1.39 / 1.21 / 1.12 / 1.08 | **1.13** / 1.40 / 1.10 / **1.03** / 1.10 | 1 |
| LAI by band, same | 1.23 / 1.61 / 1.01 / 0.55 / 0.38 | 1.08 / 1.45 / 0.91 / 0.62 / 0.61 | 1.46 / 1.52 / 0.90 / 0.58 / 0.60 | 1 |
| fu natural land / MAD / r | 0.530 / 0.280 / 0.61 | 0.289 / 0.241 / 0.47 | **0.355 / 0.210 / 0.66** | 0.335 / 0 / 1 |
| fu where tree < 0.05 / 0.15-0.30 / > 0.70 | 0.18 / 0.63 / 0.88 | 0.10 / 0.25 / 0.58 | 0.08 / 0.31 / **0.77** | 0.01 / 0.22 / 0.91 |
| fu 60-90N / 45-60N / 30-45N | 0.24 / 0.67 / 0.47 | 0.09 / 0.24 / 0.27 | **0.25** / 0.55 / 0.28 | 0.25 / 0.45 / 0.18 |
| phantom / missing-tree cells, share of natural land | 10.9 / 3.4 % | 2.4 / 14.1 % | **3.7 / 7.1 %** | |
| seasonal amplitude all / tropics / 30-50N | 0.78 / 0.90 / 0.60 | 0.81 / 0.96 / 0.60 | 0.82 / 0.96 / 0.62 | 1 |
| ET natural land / WUE | 1.07 / 1.12 | 1.03 / 1.04 | 1.04 / 1.13 | 1 / 1 |
| **sample, area-weighted GPP model / obs (fit report)** | | 0.89 | **0.95** | -- domain 1.31 |

Three readings.

1. **The structure is right for the first time.** The growing-season
   criterion brought the forests back (fu > 0.7 class 0.58 -> 0.77, boreal
   0.09 -> 0.25 = observed) while keeping most phantoms out (3.7 % of land);
   mean natural-land cover 0.355 against 0.335, the pattern correlation the
   best of any state. Boreal GPP 1.03, ET 1.04, WUE 1.13.
2. **The total went back to 1.31x**, carried by the tropics (1.13x, with
   `trop_bl_ever` vmax raised to 52) and the steppes (1.4-1.8x, where the
   grass itself over-produces: treeless land at fu 0.08 still runs 1.57x).
   Vegetation carbon (361) is low with a third of the forest cover's biomass
   still missing and `temp_con_ever` cut to 17.6.
3. **The fit never saw criterion 1.** On the fit sample the model reads
   0.95x of observed GPP while the domain reads 1.31x. The sample is
   stratified by dominant PFT for identifiability, so the rare PFTs are
   over-represented and the many low-flux steppe and grassland cells that
   carry the excess are under-represented; the "total" term is the sample's
   total, not the domain's. Every stage has optimised a total that was not
   the one being judged. The remedy is inverse-probability weighting -- each
   sampled cell weighted by its area times the ratio of domain to sample
   counts in its stratum -- so that the sample's totals and patterns
   estimate the domain's; its first test is whether the (b3) parameters then
   read ~1.3x on the sample.

### The mask: every PgC total so far was on 165 Mkm2 of "land"

Looking for the 28 PgC of model GPP in cells FLUXCOM does not cover led to
the mask. `surta.nc` is `landfrac = 1` wherever `vegtype_frac_1992.nc` has
any vegetation fraction: 95,333 cells, all exactly 1.0, **165 Mkm2 against
Earth's 149**. The vegtype fractions themselves sum to the true land
fraction of each cell (ESA CCI, 300 m to 0.5 degree): 49,682 cells are fully
land, **45,651 are fractional**, and open-water cells with a speck of island
-- (4.75 N, 120.25 E) in the Sulu Sea, land fraction 0.00; the Molucca Sea
at 0.01-0.04 -- were made 100 % land, forced with 1,700 mm of rain, and grew
rainforest: 23.6 PgC of model GPP in no-FLUXCOM cells with some MODIS
vegetation, plus 4 PgC in the deserts (Sahara LAI 0.23, Antarctica 0.01 --
the model is bare where it should be). The model's per-m2 fluxes are not
at fault; the area weights were. `landmask.true_landfrac()` recovers the
fraction (143.7 Mkm2 on the mask); `state_report` now reports on the binary
mask, on true land, and on FLUXCOM-covered land against FLUXCOM's own total
(108.9 PgC on 113.8 Mkm2); the fit's cell weights carry it too.

Restated, last 5 years of each 50-year validation:

| | ctl | v11d | v12a (a) | v12c (b1) | v12d (b2) | **v12e (b3)** | observed |
|---|---|---|---|---|---|---|---|
| GPP PgC/yr, binary mask (as quoted so far) | 234 | 182 | 176 | 164 | 148 | 164 | |
| **GPP PgC/yr, true land** | 192 | 151 | 145 | 134 | 120 | **132** | 120-130 |
| GPP, FLUXCOM-covered land | 188 | 148 | 142 | 131 | 117 | 129 | FLUXCOM 109 |
| NPP PgC/yr, true land | 86 | 70 | 67 | 63 | 55 | **60** | 55-60 |
| veg C PgC, true land | 630 | 498 | 488 | 447 | 235 | **281** | 450-650 |

Criterion 1 on true land: (b3) at 132 sits just above the literature range
and 1.18x FLUXCOM on the land both cover -- the excess that remains is the
steppes' (1.4-1.8x) and the tropics' (1.13x), which sections 14-18 have
mapped. The pool that is now plainly wrong is vegetation carbon, 281
against 450-650: with the tree cover right for the first time, the model
carries too little biomass per unit of forest -- allocation and turnover,
a pool question outside the three flux criteria, and the first item for
the stage after ET.

### The harness and the validation agree; the total was under-weighted all along

Replay of the (b3) parameters through the fit harness on the fit sample,
28 years, against the run_global validation at the same cells
(`replay_v12e.py`): year 1 1.054 vs 1.088, year 10 1.100 vs 1.099, year 20
1.129 vs 1.130, year 28 **1.152 vs 1.161** -- 1-3 % apart throughout, the
harness cycling 2001-2010's CO2 (371-390 ppm) against the validation's
fixed 389.9, with the dry 2001-2003 forcing years visibly the low ones. The
two evaluation paths are consistent; the 0.95 the fit had printed was a
report line averaging the model over every fit cell and the observation
over cells with data (fixed: the report now prints the ratio on the scored
cells).

So the fit did see a 1.13-1.15x total. It did not act on it because of the
loss's arithmetic: the GPP term is `per-cell log-ratio^2 (flux-weighted) +
w_total x ln(total ratio)^2` with `w_total` = 1 since section 9, and
ln(1.15)^2 = 0.02 against a per-cell term of ~0.22. A 15 % global bias was
worth a tenth of the pattern scatter; criterion 1 was never pulled on.
`--w-total` (new; 25 makes a 10 % bias cost as much as the pattern term)
joins the fixed gate and the land/representative weights in stage (b4).

### Stage (b4): the gate fixed where the measurement put it, the total weighted, the areas right

Launched 10:09 on 2026-09-19 (`fit_carbon_v12f`), sharing the GPU with the
closing protocol running on (b3). From the (b3) parameters, three changes
of scoring and one of structure, no change of physics: the tree gate is
**fixed at 0.80 / 0.10** on the growing-season criterion (section 18's
measurement: forests 0.94-0.96, steppes 0.69-0.74) rather than fitted --
twice the optimiser weakened it to keep steppe trees for their GPP and LAI;
`--w-total 25`, so a 10 % global bias costs what the pattern does;
cell weights carry the true land fraction (499 of the 4,331 sample cells
are under half land) and the inverse-probability stratum weights. Fitted:
`vmax`/`specla` (24) and six structural scalars (`stressfac`, `cover_k`,
`fl_min`, `fu_min`, `drop_stress_thresh`, `tthresh_u_abs`); 180
evaluations. Baseline **1.6860**: GPP 0.402 (of which ~0.18 is the total
term at ~1.09x), LAI 0.769, season 0.0168, tree 0.244; the fixed gate's
accumulator attached through the new path and logged. Judged on the
domain by the true-land total, the steppes' GPP, forest and boreal cover,
and vegetation carbon.

### Stage (b4) result: 1.6860 -> 1.3698; the total term did the work

180 evaluations (10:09 -> 22:41, GPU shared with the closing run until
19:14). Best at eval 29 and never beaten: GPP 0.2154 (from 0.402 -- the
weighted total term collapsing as the sample total went from ~1.09x to
~1.0x), LAI 0.5168 (from 0.769; the lowest of any fit), season 0.0167,
tree 0.243 (the gate is fixed, so this is the pattern the gate leaves),
hold-out 1.458.

| scalar | start | fitted | |
|---|---|---|---|
| `cover_k` | 0.428 | 0.273 | understory cover thinned again |
| `drop_stress_thresh` | 0.134 | 0.086 | dormancy trigger off, as in (b1) and (b3) |
| `tthresh_u_abs` | 12.5 C | 12.1 C | |
| `stressfac`, `fl_min`, `fu_min` | -6.53, 0.229, 0.033 | -6.46, 0.247, 0.040 | unchanged in effect |

| PFT | vmax (b3) -> (b4) | specla (b3) -> (b4) | |
|---|---|---|---|
| trop_bl_ever | 52.1 -> **47.9** | 14.9 -> **11.6** | the total term pulled the tropics back |
| trop_bl_dec | 40.3 -> 41.5 | 11.2 -> 13.3 | |
| warm_bl_ever | 28.0 -> 29.7 | 18.2 -> 18.4 | |
| temp_con_ever | 17.6 -> 21.4 | 14.9 -> 13.5 | |
| temp_bl_dec | 58.3 -> 55.9 | 14.1 -> 12.0 | |
| ever_shrub | 35.7 -> 38.9 | 8.0 -> 7.1 | |
| c4_grass / c3_grass | 18.2 -> **21.8** / 17.0 -> 16.4 | 10.2 -> 10.0 / 11.3 -> 10.2 | |
| boreal_con_ever, boreal_con_dec, dec_shrub, boreal_bl_dec (vmax) | frozen / unmoved | | |

With the gate fixed and the total weighted, the optimiser moved capacity
rather than structure: the tropical evergreen canopy down (vmax and
specla), C4 grass up.

Final report, on the scored cells with the same mask for model and
observation (the line added after section 18's 0.95-vs-1.16 confusion):
**GPP model/obs 0.997 on the fit cells, 0.995 held out; ET 0.980 / 0.983
on the all-five cells; LAI 1.61 vs 1.57.** On the representative,
land-weighted sample the carbon total, the leaf area and the ET total all
sit at unity, with the tree term (fu 0.22 vs 0.30) the fixed gate's
residual. The domain validation and the closing protocol, both running
overnight, say whether the domain follows -- a true-land total near
120-125 PgC would -- and judge (b4) against (b3) on forest and boreal
cover, the steppes and the pools.

### Stage (b4) over the domain, against (b3): the total at the forests' expense again

50-year validations (years 2056-60; true-land totals):

| | (b3) v12e | **(b4) v12f** | observed |
|---|---|---|---|
| GPP PgC/yr, true land / on FLUXCOM-covered land | 132.1 / 128.9 (1.18x) | **118.4 / 115.3 (1.06x)** | 120-130 / 109 |
| NPP true land | 59.6 | 51.1 | 55-60 |
| GPP by band 30S-30N / 30-50N / 50-60N / 60-70N / 70-90N | 1.13 / 1.40 / 1.10 / 1.03 / 1.10 | **1.01 / 1.25 / 0.99 / 1.05 / 1.21** | 1 |
| ET (vs MODIS) / WUE; ET vs ensemble | 1.04 / 1.13; 0.99 | 1.02 / **1.03**; 0.99 | 1 |
| fu natural land / MAD / r | **0.355 / 0.210 / 0.66** | 0.272 / 0.227 / 0.58 | 0.335 / 0 / 1 |
| forests (> 0.7): fu / GPP | **0.77 / 0.93** | 0.62 / 0.83 | 0.91 / 1 |
| boreal 60-90N fu; 45-60N fu | 0.25; 0.55 | 0.22; 0.47 | 0.25; 0.45 |
| phantom / missing-tree cells | 3.7 / **7.1 %** | 2.1 / 12.9 % | |
| veg C PgC true land; soil C; litter | **281**; 11.6; 0.59 | 207; 11.1; 0.49 | 450-650; 10-15; 0.4-0.8 |
| T/ET (components) | 0.46 | 0.42 | 0.57-0.64 |
| seasonal amplitude all / tropics / 30-50N | 0.82 / 0.96 / 0.62 | 0.81 / 0.95 / 0.62 | 1 |

(b4) is the best state yet on criterion 1 and on the band pattern -- and
it got there the way (b2) did, in a milder form: the gate fixed at 0.80
took forest cover from 0.77 to 0.62 of the observed 0.91, doubled the
missing-tree land, and cut vegetation carbon to 207. The measurement that
put the threshold at 0.80 (section 18) was made at `stressfac` -8.1; the
fit ran at -6.5, a stronger stress function, and the two are one
coordinate -- fixing one while the other moved was the mistake. On the
sample (b4) read GPP 0.997 and ET 0.980 against the domain's 1.06x and
0.99: with the land and representative weights the sample is now honest.

**Decision.** The joint carbon + ET fit starts from **(b3)** (forests,
pools, cover pattern) with the **gate free** again and **(b4)'s scoring**
(total weighted 25, land and representative weights), the ET terms on,
and `soil_beta_power` among the extras. The closing protocol on (b4) is
stopped: it would document a state not going forward and halve the joint
fit's speed for six hours; the (b4) 50-year validation above is its record.

## 19. What is scored, and the closing protocol

Asked on 2026-09-18: are the numbers spin-up results, or after spin-up?
Neither, exactly -- the configurations, verified:

| step | run | forcing | CO2 | what it produces |
|---|---|---|---|---|
| spin-up (`v11_spinup`) | cold start, 1601-1900 (300 yr), IBIS default parameters | 1901-1930 GSWP3 cycled | 296 ppm fixed | a pre-industrial equilibrium, labelled 1900 |
| historical transient (`v11_transient`) | 1901-2010 from the 1900 state | real GSWP3, year by year | rising, 1901-2010 | `ibis_restart_2010.npz` -- the start of every fit and validation |
| a fit evaluation (`fit_carbon`) | from the 2010 state, 25 adjust + 3 scored years, 4000 cells | 2001-2010 cycled | the cycled year's | the candidate parameters at a modern-climate quasi-equilibrium |
| a validation (`val_v12x/fit`) | from the 2010 state, 50 yr (last 5 scored) + 10 monthly | 2001-2010 cycled | 389.9 ppm fixed | the fitted parameters at a modern-climate quasi-equilibrium |

So every number in sections 8-18 is "after spin-up and after the
historical transient", re-equilibrated 25-50 years to modern weather under
the parameters being judged, and compared with 2001-2013 observational
climatologies -- the right comparison for the three criteria, which are all
climatological. Two caveats follow. The 2010 start state was made under the
v11 default parameters, so the slow soil pool in every table is inherited
(50 years re-equilibrate vegetation, litter and fast soil carbon, not that).
And nothing is a year-by-year hindcast; the one transient test was the
1901-2010 trend on v11d (section 11).

**Closing protocol** (`verify/calibration/final_validation.sh <TAG>
<params.yaml>`, ~6 h GPU, once per final parameter set): the v11 protocol
under the final parameters -- cold spin-up 1601-1900, transient 1901-2000,
monthly hindcast 2001-2010 -- scored on the actual 2001-2010 years: pools,
fluxes and bands (`state_report`), tree cover (`fu_vs_tree`), ET
(`et_vs_modis`), season (`seasonal_baseline`), the savanna and ET tower
checks, fire (GFED5, 2001-05 from the 2010 state), methane (GMB), and the
1901-2010 GPP trend by decade. Its 2010 state is self-consistent and is the
one Stage 2 (ET) and the module re-validations should start from. To be
run when the carbon structure is final -- after (b2), or after a (b3) if
the validation tonight says one more structural pass is needed.

### Closing protocol on (b3): the self-consistent state, scored on 2001-2010

`final_validation.sh v12e`, 09:51-19:14 on 2026-09-19 (sharing the GPU
with the (b4) fit): cold spin-up 1601-1900 on the (b3) parameters,
transient 1901-2000, monthly hindcast 2001-2010, everything scored on the
ten hindcast years -- the first numbers in this document from a state made
entirely under the parameters being judged.

| | closing (b3), 2001-2010 | (b3) cycled validation, 2056-60 | observed |
|---|---|---|---|
| **GPP PgC/yr, true land** | **129.8** | 132.1 | 120-130 |
| GPP, FLUXCOM-covered land | 126.6 (1.16x) | 128.9 | FLUXCOM 109 |
| NPP true land / Rh true land (est.) | **57.6** / ~52.8 | 59.6 / ~59 | 55-60 / 49-58 |
| ET mm/yr | 504 | 510 | 500-550 |
| veg C PgC, true land | **219** | 281 | 450-650 |
| **soil C kgC m-2** | **6.9** | 11.6 (inherited from the v11 state) | 10-15 |
| litter kgC m-2 | 0.49 | 0.59 | 0.4-0.8 |
| GPP by band 30S-30N / 30-50N / 50-60N / 60-70N / 70-90N | 1.10 / 1.38 / 1.09 / **0.98 / 0.99** | 1.13 / 1.40 / 1.10 / 1.03 / 1.10 | 1 |
| LAI by band, same | 1.42 / 1.49 / 0.87 / 0.52 / 0.52 | 1.46 / 1.52 / 0.90 / 0.58 / 0.60 | 1 |
| fu natural land / MAD / r | 0.323 / 0.215 / 0.63 | 0.355 / 0.210 / 0.66 | 0.335 / 0 / 1 |
| fu forests (> 0.7) / boreal 60-90N | 0.70 / 0.21 | 0.77 / 0.25 | 0.91 / 0.25 |
| phantom / missing-tree cells | 3.4 % / 8.7 % | 3.7 % / 7.1 % | |
| ET natural land / WUE; towers model/tower (ensemble/tower) | 1.03 / 1.14; 1.11 (0.99) | 1.04 / 1.13 | 1 |
| seasonal amplitude all / tropics / 30-50N; shape error | 0.82 / 0.97 / 0.62; 0.0167 | 0.82 / 0.96 / 0.62 | 1 |
| savanna towers: amplitude tower / FLUXCOM / model; phase lag model | 10.4 / 12.2 / **10.6**; +1.43 mo | | tower +0.21 |
| GPP trend, 1900s -> 2000s decadal means | 135.5 -> 161.1 (**+19 %**) | v11d +21 % | |
| fire: burned area (GFED5 4.60, spread 2.38-4.60) / rank / NMSE | 2.49 % / +0.747 / 0.858 | 2.55 % | |
| methane `biome_tref`: Tg / split 30S-30N, 30-60N, 60-90N / peak / shape | **167.7 / 70.7, 20.5, 3.0 / 7 / 6.3** | 199 / 65, 25, 5 / 7 / 12.9 | 158 [119-203] / 70, 21, 8 / 7-8 |

Readings.

1. **The three flux criteria are met or nearly met on a self-consistent
   state.** Total 130 on true land (the cycled validation's 132 held);
   the high latitudes are exact (60-90N GPP 0.98-0.99, ET 0.98, WUE 1.02);
   the tropics 1.10x; the 30-50N steppes 1.38x remain criterion 2's
   residual. Season 0.82 overall, 0.97 in the tropics -- and at the savanna
   towers the model's amplitude now equals the towers' (10.6 vs 10.4) where
   FLUXCOM overshoots (12.2); the phase lag (+1.4 months) is what is left of
   criterion 3.
2. **The pools are the open problem, and the cold spin-up shows it
   plainly.** Vegetation carbon 219 PgC against 450-650, soil carbon 6.9
   kgC m-2 against 10-15 -- the cycled validations had carried the v11
   state's soil (11.6); equilibrated on the corrected canopy's lower inputs
   the soil settles well below range while Rh (~53) is inside it, so the
   decomposition rates are too fast for the inputs (the both_fix x0.67 on
   `knb/kpb/ksb`, undone at v11, is back on the table) and biomass per unit
   forest is too small (allocation / wood turnover). Neither is a flux
   criterion; both are the stage after ET.
3. **The modules held.** Methane `biome_tref` at 167.7 Tg with the split
   70.7 / 20.5 / 3.0 against 70 / 21 / 8 and shape error **6.3** -- the best
   methane result of the project (the Arctic 3.0 vs 8 is the one deficit);
   fire inside the product spread with the pattern kept.
4. The 1901-2010 GPP trend is +19 %, as v11d's was, so the corrected
   canopy did not change the CO2 response.

`final_v12e/hindcast/restart/ibis_restart_2010.npz` is the self-consistent
2010 state of (b3).

### Closing protocol on the joint fit (v12g), beside (b3)

`final_validation.sh v12g`, 12:09-22:52 on 2026-09-20 (the hindcast with
ET components); both columns are cold spin-up -> transient -> 2001-2010
hindcast under the parameters named:

| | closing (b3) | **closing (joint, v12g)** | observed |
|---|---|---|---|
| GPP PgC/yr, true land / on FLUXCOM land | 129.8 / 126.6 (1.16x) | **115.6 / 112.5 (1.03x)** | 120-130 / 109 |
| NPP true land / Rh | 57.6 / 60.6 | 50.3 / 53.6 | 55-60 / 49-58 |
| ET mm/yr; ET natural land vs MODIS; WUE | 504; 1.03; 1.14 | 491; **1.00; 1.03** | 500-550; 1; 1 |
| veg C true land; soil C; litter | 219; 6.9; 0.49 | **163; 6.3; 0.40** | 450-650; 10-15; 0.4-0.8 |
| GPP by band 30S-30N / 30-50N / 50-60N / 60-70N / 70-90N | 1.10 / 1.38 / 1.09 / 0.98 / 0.99 | **0.96 / 1.26 / 1.00 / 1.00 / 1.17** | 1 |
| fu natural land / r; forests fu / GPP | 0.323 / 0.63; 0.70 / 0.94 | 0.290 / 0.57; 0.61 / 0.84 | 0.335; 0.91 / 1 |
| phantom / missing-tree cells | 3.4 / 8.7 % | 2.6 / 12.5 % | |
| seasonal amplitude all / tropics / 30-50N | 0.82 / 0.97 / 0.62 | 0.80 / 0.94 / 0.61 | 1 |
| savanna towers: model amplitude / phase lag | 10.6 / +1.4 mo | 7.5 / +1.2 mo | 10.4 / +0.2 |
| ET at 194 towers, model/tower (ensemble/tower 0.99) | 1.11 | ~1.08 | 1 |

The joint state has the total (1.03x FLUXCOM on covered land; 116 PgC on
true land, at the bottom of the literature range), the ET total and WUE
at unity and four of five bands within 4 % -- and pays for it with a third
of the forest cover and half the vegetation carbon (163 PgC). (b3) keeps
more forest and biomass and runs 1.16x. Neither is final: the phenology
clock defect found the same evening (section 16) means both were fitted
on a model whose seasons lagged the weather by 77 days, and the seasonal
criterion and the pools are re-read on the corrected model before the
carbon+ET stage is closed.

## 20. Site-first calibration (pivot of 2026-09-22)

Hangkai's decision after the review of the record and a discussion with a
labmate: calibrate at flux towers first, then the spatial pattern and the
totals while keeping the sites good; carbon, then ET, then methane. The
global refit queued as stage (d) is withdrawn. The plan with its open
questions is `/shared/nas2/Hangkai/ibis_local_document/SITE_CALIBRATION_PLAN.md`;
the running record with figures is `IBIS_work_record_2026-09-22.md` in the
same folder.

**Tools** (all under `verify/calibration/`): `tower_forcing.py`
(FLUXNET2015 daily tower meteorology in the GSWP3 reader contract; the
diurnal range from the half-hourly files; gaps and years after 2010 from
GSWP3 at the tower's cell), `site_run.py` (prescribed composition from
the IGBP class via `prescribe()`, existence held to the prescription,
spin-up on cycled tower years with dynamic vegetation off, one evaluation
pass, monthly GPP and ET against the tower, summer LAI against the MODIS
maximum; `--recompute` from a saved run), `site_plot.py`,
`site_compare.py` (per-class summary and figure across parameter sets).
Site inventory and batches: `site_inventory.py` in the scratchpad ->
`site_inventory.csv`, `site_batches.txt` (114 natural-vegetation towers
with three or more years; common-year batches 2008-2012 (62), 2001-2005
(14), 2005-2009 (8), 2010-2014 (5), 2002-2006 (4); 47 towers need
three-year windows).

**S0, the corrected-model reference (done 2026-09-22):** the fixed model
under IBIS defaults and under the joint-fit parameters, 50 years plus 10
monthly years each (`val_base_default`, `val_base_v12g`; report
`val_report_baseline.txt`). Defaults: GPP true land 183 PgC (1.5x), NPP
82, ET 568, vegetation carbon 580 (in range), soil carbon 13.8 (in
range), boreal LAI = MODIS; LAI 2.8-2.9x MODIS in the tropics and
mid-latitudes; shape error 0.0172, amplitude 0.63-0.67 at 30-70 N;
phantom trees on 15 % of the area; forest ET 57 % transpiration, 37 %
interception, 6 % soil. Joint fit on the fixed model: GPP 114, ET 549,
vegetation carbon 219, boreal LAI 0.35x with boreal GPP 1.54x; shape
error 0.0157, amplitude 0.84; interception share unchanged (36 %). Each
set gets its own fitted targets and the other's wrong.

**Closing protocol on the defaults (`final_base_default`, done 2026-09-23
12:46; cold spin-up 1601-1900, transient 1901-2000, monthly hindcast
2001-2010, scored 2006-2010):** GPP true land 177.5 PgC (1.4x; bands
tropics 1.40, 30-50 N 2.02, 50-60 N 1.88, 60-70 N 2.23), NPP 78.8, Rh
84.5, ET 564, vegetation C true land 471 (in range), soil C 14.0 (in
range), litter 1.01 (high); LAI 2.8x MODIS south of 50 N, 0.86x at 60-70
N; tree cover 0.663 vs CLM 0.335 (phantom forest on 15 % of the area,
GPP 2.2x there, 25 % of the excess); ET 1.16 of the ensemble (treeless
1.48; 30-45 N 1.34; tropics 1.02), and at 194 towers model/tower 1.24
where ensemble/tower is 0.99; seasonal shape 0.0173, amplitude 0.85
(tropics 1.00, 30-70 N 0.62-0.66), savanna GPP peak +1.3 months after the
rain (towers +0.2); GPP trend 1900s -> 2000s +18 %; **burned area 4.62 %
vs GFED5 4.60 % (1.01x, rank r 0.76)**; methane biome_tref 268.9 Tg
(GMB 158, range 119-203; split 61/26/6.5 vs 70/21/8; shape 15.9).
Against the two fitted closings ((b3): GPP 129.8, veg C 219, soil C 6.9,
fire 2.49 %, methane 167.7 / shape 6.3; joint: GPP 115.6, veg C 163):
the defaults hold the pools and the fire, the fitted sets held GPP. The
site-first calibration starts from the defaults.

**S1.3, the site baseline (done 2026-09-22):** 90 towers with the joint
fit, 63 with defaults (two batches rerunning), tower meteorology,
prescribed composition, two spin cycles, evaluated on the batch window.
Defaults: median GPP ratio 1.04 but 37 % of towers within 0.8-1.25;
forests 0.84-0.88 with model summer LAI twice the MODIS maximum,
grasslands 1.53, shrublands 2.25, semi-arid towers 1.5-9x. Joint fit:
median 0.91, 32 % within 0.8-1.25; leaf area brought to MODIS, needleleaf
0.73, grasslands 0.78, shrublands 1.33, deciduous 0.97. Invariant between
the two sets and therefore not a vmax/SLA question: the needleleaf and
mixed forests peak a month late; the cold-season share of GPP is
0.12-0.20 against 0.01-0.07 at the towers (evergreen conifers
photosynthesising at 50-60 gC m-2 month-1 in winter at Tharandt against
15-25; C4 grass at Walnut Gulch producing 15-25 in a dry winter where the
tower is at zero); forest ET 1.1-1.7x the tower LE. Tropical seasonality
has no skill (Guyaflux r 0.05; Australian savannas two to four months
off). Order for S1.4: phenology timing and cold-season dormancy,
interception, then leaf area against capacity (SLA, allocation, vmax),
then water limitation at dry sites.

**S1.3 CORRECTED (2026-09-23 23:05) -- harness defect, and a wrong LAI
reference.** Two errors invalidated the model side of the S1.3 numbers
above. (1) `site_transient.one_year` recomputes `exist` from climate at
year end BEFORE `biomass_update`, so non-prescribed PFTs gained carbon
every year and their leaf area entered `avglaiu` regardless of `exist`
(evergreen PFTs are summed unweighted there): prescribed deciduous stands
ran EVERGREEN at mild towers. `site_run.enforce_prescription()` now clears
the carbon and `plai` of non-prescribed PFTs each year (commit ef0e019);
verified at IT-Col (displayed LAI was stuck at 5.10 with tempu = 0 from
October to May; now 0). (2) Model leaf area at a tower was compared with
MODIS at the tower's 0.5-deg CELL. The towers' own canopies carry 1.69x
the cell's satellite LAI, so the model reads 2.20x the cell but only
1.23x the site's measured maximum; by class against the site value:
DBF 1.05, ENF 0.80, MF 0.45, GRA 1.75, SAV 1.57, CSH 2.15. Forests do
not carry too much leaf.

Corrected results, 62 towers, defaults, composition held:
- **Dynamic carbon: the prescribed stand DIES at 12 of 61 towers**
  (DK-Sor, BE-Bra, FR-Fon, NL-Loo, RU-Fyo, IT-Col, IT-Ren, IT-SRo,
  IT-Ro2, FR-Pue, AU-Tum, CA-Gro) -- default allocation/turnover cannot
  sustain the type those sites carry; the invaders had hidden it. New
  `stand_died` flag. GRA 1.59 -> 1.08 and CSH 2.04 -> 1.23 once the
  composition is held (their excess was the invaders); forests unchanged.
- **Fixed vegetation** (`--fixed-lai attrs`: leaf carbon held at the site
  attribute LAI_Max, phenology still modulates the displayed canopy; the
  review's fixed-vegetation experiment): GPP model/tower DBF 0.75, ENF
  0.81, EBF 0.85, GRA 0.87, MF 1.00, SAV 1.00-1.28, OSH 2.12, all 0.85;
  monthly r 0.86; ET 1.15 (DBF 1.62, MF 1.35, ENF 0.99); peak lag +1 at
  ENF and MF, 0 elsewhere. **Cold-season share 0.12-0.16 vs tower
  0.01-0.06, unchanged by either correction, in every class.**
- Order for S1.4 revised: cold-season rate first, then forest capacity
  (the model UNDER-produces 15-25 % with the right canopy), then
  semi-arid water stress (OSH 2.1x), then allocation/turnover for the
  stands that cannot stand up.

**Phenology timing, quantified (2026-09-23; `phenology_vs_phenocam.py` on
`site_runs/default_phenocam_2008-2012`):** 23 PhenoCam-matched towers
rerun on defaults with daily GPP; transition dates = 25 % of the annual
amplitude of the 15-day-smoothed curve, rising and falling, for the model
GPP, the tower GPP (FULLSET daily) and the camera greenness (network
transition dates, gcc_90). Summer-growing towers with a camera (n = 14):
model leaf-out 30 d before the camera and 31 d before the tower GPP
onset; senescence 21 d after the camera and 26 d after the tower GPP;
tower GPP and camera agree within 3 d. Deciduous broadleaf (4): -34/+35
vs camera, -43/+44 vs tower GPP. Grassland meadows (3): -44/+40.
Evergreen needleleaf (4): -18/-6 (their winter error is a rate, not a
date). Mixed (2): +2/+22. Winter-growing semi-arid and Mediterranean
towers excluded (the crossing search does not apply; 6) plus 2 with an
untimeable tower season. Same sign and size as Kucharik et al. 2006 at
Harvard, Willow Creek and Park Falls. S1.4 target: leaf-out ~40 d later,
senescence ~40 d earlier at deciduous and grass towers -> `gdd_thresh_u/l`
and ramps, `tthresh_u_abs`, `tthresh_l_c`, the drop triggers.
Data: `inputs/sites/PhenoCam/` (29 towers), `inputs/sites/
FLUXNET2015_FULLSET/`.

## 21. Two formulation gaps the site stage exposed (2026-09-24)

The screening and the corrected site baseline separate our errors into two
kinds. Capacity, leaf area, water stress and the spring degree-day
threshold are parameter errors and the fit can reach them. Two are not,
and the literature names both, with published equations and constants.
The plan for Hangkai and their labmate to choose from is
`/shared/nas2/Hangkai/ibis_local_document/REVISED_PLAN_2026-09-24.md`.

**(a) No cold-season down-regulation of photosynthesis.** Our cold-season
GPP share is 0.12-0.16 against 0.01-0.06 at the towers, in every class,
and no phenology parameter moves it at the evergreen conifers at all.
Evergreen needleleaf forests down-regulate photosynthesis completely in
winter while staying green; CLM5 shows the same overestimate in both its
prescribed-LAI and prognostic modes. The standard representation is a
state of acclimation: a first-order delay of temperature with a time
constant near 8 days (200 h) multiplying the photosynthetic capacity
(Makela et al. 2004, Tree Physiology 24, 369; Kolari et al.;
Gea-Izquierdo et al. 2010). Related: thermal acclimation of the Farquhar
parameters (Kattge & Knorr 2007) implemented in JULES vn5.6 (Oliver et
al. 2022, GMD 15, 5567: tropical GPP -5..-10 %, high-latitude +20 %), and
photoperiod control of capacity (Bauerle et al. 2012, PNAS 109, 8612:
-4 PgC/yr globally). IBIS has a constant per-PFT vmax with an
instantaneous temperature response and no memory.

**(b) Senescence has no photoperiod gate.** The upper-canopy rule is a
pure threshold on the ten-day mean temperature, floored at 0 C; the
sweep showed it cannot pull autumn back by more than 12 days, against a
target of about 40. The standard replacement is Delpierre et al. (2009):
senescence accumulates only when day length is below a threshold, at a
rate set by the temperature deficit (refined by Lang et al. 2019, Zani et
al. 2020, the DP3 model 2025). Our spring rule is likewise the oldest
form, a bare growing-degree-day threshold with no chilling requirement,
which is why leaf-out is a month early. For scale, a seven-model European
evaluation found growing-season starts ~0.6 months late and ends ~0.5
months early (Biogeosciences 18, 2405, 2021), citing a 235 gC m-2 yr-1
GPP error at deciduous sites from a two-week spring bias; our biases are
larger and of the opposite sign.

**A method correction on the way through.** Model leaf area at a tower was
compared with MODIS at the tower's 0.5-degree CELL. Tower canopies carry
1.69x their cell's satellite LAI (site-attribute dataset, ESSD 17, 117,
2025), so the model reads 2.20x the cell but 1.23x the site. Published
representativeness work says the same: mean absolute vegetation-index
bias grows from 4.3 % at site scale to 12.1 % at land-model grid scale
(Chu et al. 2021, AFM 301-302, 108350; Sun et al. 2024, Sci. Data).

## 22. Stage A: the two formulations in the model, and what they do at the towers (2026-09-25)

Both formulations of section 21 are in the model (commit `0b3c9b3`), each behind a static switch that is off
by default. `configs/params/default_seasonal.yaml` is `default.yaml` with both switched on.

**(a) Acclimation of photosynthetic capacity** (`stomata: acclim_on`). This is the state of acclimation of
Makela et al. (2004) in PRELES form. Once a day, S <- S + (T - S)/tau, with T the daily mean air temperature.
The model multiplies vmax, and with it dark respiration, by f_S = clip((S - S0)/Smax, 0, 1) for the PFTs in
`acclim_pfts`: temperate and boreal evergreen conifers and the evergreen shrub (3, 5, 8), which stay green
through a frozen winter. The defaults are the Rpreles calibration to ten Scandinavian conifer sites
(tau 10.93 d, S0 -3.063 C, Smax 17.72 C).

**(b) Photoperiod-gated senescence** (`pheno: sen_photoperiod_u / _l`). This has the structure of Delpierre et
al. (2009) with x = 1, y = 0. While day length is decreasing and below Pstart, the canopy accumulates cold
degree-days below Tb. Once the sum reaches Ycrit, its leaves fall as if the ten-day temperature had crossed the
drop threshold, and the degree-day sum for the next leaf-out is held at zero. The sum resets while day length
is not decreasing, i.e. from the winter to the summer solstice. The starting values of 14 h, 20 C and 300 C d
are chosen, not published.

**Where they live.** Both states are in a new optional `SeasonalState` (acclim_x, ssen_u, ssen_l). It extends
the chain of optional trailing modules (StressAccum -> ETParts -> SeasonalState), so a state carries 258, 261,
265 or 268 leaves. `load_state` now checks both ends of a migration against that chain instead of a bare limit.
The fitted constants are traced floats and are `fit_carbon.EXTRAS` entries (acclim_tau/s0/smax,
sen_pstart/tb/ycrit_u/l). `finalize_fit.sh` carries them back into yaml. `run_global` attaches the module when
either switch is on.

**Verification.**
- Unit tests: 7 in `tests/test_seasonal_formulations.py`. Full suite: 703 passed.
- Bit-identity: in the real GSWP3 loop, attaching the module with both switches off changes 0 state fields and
  no monthly GPP.
- Senescence on: Harvard Forest November GPP falls from 143 to 6 gC m-2.

**The conifer A/B.** The 17 ENF towers of the 2008-2012 batch were run with composition prescribed and leaf area
held at the site's measured maximum (`site_fit.py --only-baseline`), with and without the formulations. Monthly
GPP climatology, 2009-2011, gC m-2 month-1:

| tower | | J | F | M | A | M | J | J | A | S | O | N | D | cold share (NDJF) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CZ-BK1 | tower | 30 | 19 | 59 | 235 | 243 | 285 | 300 | 296 | 220 | 127 | 54 | 7 | 0.06 |
| | off | 23 | 28 | 73 | 140 | 141 | 152 | 161 | 169 | 157 | 122 | 96 | 33 | 0.14 |
| | on | 1 | 1 | 14 | 83 | 107 | 139 | 159 | 167 | 146 | 82 | 48 | 8 | 0.06 |
| IT-Ren | tower | 5 | 11 | 41 | 161 | 246 | 288 | 301 | 251 | 206 | 130 | 58 | 7 | 0.05 |
| | off | 46 | 43 | 73 | 110 | 133 | 139 | 158 | 166 | 154 | 124 | 88 | 35 | 0.17 |
| | on | 1 | 2 | 7 | 42 | 77 | 104 | 141 | 149 | 123 | 69 | 30 | 4 | 0.05 |
| US-NR1 | tower | 0 | 1 | 1 | 15 | 95 | 174 | 196 | 190 | 117 | 58 | 6 | 0 | 0.01 |
| | off | 14 | 10 | 23 | 30 | 68 | 99 | 114 | 113 | 93 | 68 | 51 | 21 | 0.14 |
| | on | 0 | 0 | 1 | 4 | 22 | 63 | 98 | 100 | 72 | 35 | 10 | 0 | 0.02 |

Over all 17 towers, the median cold-season share is 0.050 at the towers, 0.153 without the formulations and
0.063 with them. The site loss of section 20 falls from 29.98 to 21.69 on the training towers, from 29.31 to
20.43 on the held-out towers, and from 29.91 to 18.85 on the held-out year.

The winter excess that no parameter could move (section 21a) is gone at every conifer tower from Finland to
Italy. What remains is capacity:
- Summer GPP is about half the tower value at most sites (DE-Tha June 150 vs 346, IT-Lav 137 vs 353).
- With the default acclimation constants, spring now comes late at the warmer sites (CZ-BK1 April 83 vs 235,
  DE-Obe 74 vs 183).

Both are what the stage C1 fit is for: conifer vmax, and the three acclimation constants.

**Two harness defects found on the way.**
1. `tests/test_forcing_invariants.py` had only ever passed alone. In a full run, the fire test modules switch
   x64 on at import, and the float64 parameter tables promote the soil-BGC carry. The test now runs in production
   precision (commit `54aa5d6`).
2. eqx.filter_jit keys its cache on which parameter leaves are arrays. A spin-up on the parameters as built,
   followed by fitted evaluations, therefore compiled every month kernel twice: 1784 s for the first evaluation
   above against 265 s once compiled, on the GPU for 17 towers. `fit_carbon.trace_extras` now makes every EXTRAS
   entry an array before the first run (commit `cb7e51e`). The site tools also use a persistent XLA compilation
   cache.

**The stem-credit correction at the same towers (added 05:55).**
- The site stage has run on `default.yaml`, which is the F77 canopy: both corrections of sections 16-17
  (`scale_leaf_only`, `vmax_mix`) are off, although section 17 set them as rules for every fit. The approved
  plan names neither.
- At the ENF towers only `scale_leaf_only` can matter, because the stands are single-PFT. Two CPU baselines
  isolate it (`$V/site_fit/ab_cpu_seasonal`, `ab_cpu_seasonal_leafonly`, both with the formulations of this
  section on):
  - The CPU baseline reproduces the GPU one to 3e-5 (21.6947 vs 21.6941).
  - Removing the credit lowers GPP by 17 % (median x0.83, range 0.81-0.88). The median ratio to the towers
    goes from 0.51 to 0.43.
  - The loss goes from 21.69 to 26.28, and the monthly NEE RMSE from 65 to 73 gC m-2.
- So the stem credit was not hiding the summer deficit; the physically correct canopy makes it larger.
- The conifers produce about half the tower GPP with realistic leaf area and ET close to the towers' (ratio
  0.85-1.0). The model's water-use efficiency is therefore about half the towers'.
- The C1 fit (F77 canopy, as in the plan) will show whether vmax within its bounds can close this. If it presses
  the upper bound, the deficit is in canopy light use rather than capacity.
- Per-tower ratios: `$V/site_fit/leafonly_AB_ratios_2026-09-25.txt`.
- Whether the site stage should carry the two corrections is for Hangkai to decide.

**Where the forest deficit sits: the summer light response (added 06:05).** June-August daily GPP was binned by the
tower's own incoming shortwave. The fixed-vegetation run on defaults (`site_runs/default_2008-2012_fixedlai`, tower
meteorology) was compared with the towers' daily GPP (mean of NT and DT, NEE QC >= 0.8), using
`diagnostics/light_response.py`:

| | model/tower, SW < 150 W m-2 | model/tower, SW >= 250 | slope, gC m-2 d-1 per 100 W m-2 (model / tower) |
|---|---|---|---|
| ENF, 17 towers | 0.74 | 0.46 | 0.22 / 1.69 |
| DBF, 12 towers | 0.65 | 0.46 | 0.63 / 2.13 |

The model's summer canopy is flat at 4-5.5 gC m-2 d-1 from the dimmest to the brightest days. The towers rise from
3-4 to 10-12. On the dimmest days the model matches the towers or exceeds them.

The deficit is therefore the high-light plateau, not absorption or quantum efficiency. The plateau could come from
the capacity of the scaled big leaf (vmax times the canopy scaling), from a stomatal limit, or from a stress factor
that binds on bright days. It is the same "light-saturated canopy" that lever tests 16 and 21 met in the tropics
and on the steppes. The next process probe is an hourly one at a humid conifer tower: absorbed PAR, the limiting
rate (light, Rubisco or sink), ci, gs, the stress factors and the scaling factor.

Figures are in the local document (F12, F13). The tables are `$V/site_runs/light_response_{ENF,DBF}.txt`.

**What sets the plateau: an hourly probe inside the canopy (added 06:25).**
`diagnostics/canopy_plateau_probe.py` wraps `stomata` in the production month (a `jax.debug.callback`; no change to
the physics). It was run in July 2010 at the 17 ENF towers: prescribed stands, fixed leaf area, formulations of this
section on, F77 canopy. It records the conifer big leaf's inputs and limiting rates every hour; the last of the
three canopy iterations is kept. Raw record: `$V/site_fit/plateau_probe_ENF_201007.npz`. Medians over daylight
hours:

| top PAR | vmax (T, stress) | ci | leaf T | soil stress | wet fraction | limit: light / Rubisco / sink | top-leaf Ag |
|---|---|---|---|---|---|---|---|
| 126 W m-2 | 37 umol m-2 s-1 | 224 ppm | 22.7 C | 0.995 | 0.04 | 7 % / 87 % / 7 % | 8.7 umol m-2 s-1 |

| extinction k | LAI / SAI | scale (1 - e^-k(L+S))/(kL) | effective leaf area, scale x L | model GPP | tower GPP |
|---|---|---|---|---|---|
| 2.2 (daytime mean) | 4.1 / 0.85 | 0.245 | **1.00** | 4.6 gC m-2 d-1 | 9.4 gC m-2 d-1 |

The top leaf is Rubisco-limited in almost every daylight hour. Water stress and wet leaves play no part.

The effective leaf area is 1.0 at every tower, with LAI anywhere from 2.5 to 7.3. The F77 big leaf computes the
top-of-canopy leaf rate and multiplies it by the light-profile integral. The canopy's capacity therefore declines
with the instantaneous PAR extinction coefficient, 1-2 at midday and more at low sun. A dense canopy then has the
capacity of about 1/k square metres of top leaf, independent of its leaf area. (This is also why removing 90 % of
plai left 58 % of GPP in lever test 11b.)

Observed within-canopy gradients of photosynthetic capacity are far shallower. Lloyd et al. (2010), from numerous
forest canopies: Kn = exp(0.00963 Vcmax25 - 2.43), i.e. 0.13-0.23 for the Kattge et al. (2009) values. CLM4 used
Kn = 0.11. Bonan et al. (2012, JGR-Biogeosciences 117, G02026) used a multilayer canopy to show how strongly
canopy GPP depends on this scaling:
- With a shallow Kn, CLM4's two-leaf canopy gives too much shaded-leaf photosynthesis. CLM4 compensated by
  reducing Vcmax below the TRY values.
- A multilayer canopy with the observed Vcmax matched the towers as well or better.

IBIS errs the other way: its implied Kn equals k, 5-10x the observed gradient. So the site fit can close the
summer deficit only with an effective vmax well above leaf measurements, and the canopy light response stays
flat.

This is a third formulation gap next to section 21 (a) and (b), for Hangkai to decide. The candidates, lightest
first:
1. Scale the capacity-limited rates (jc, js) with a fixed Kn (Lloyd 2010), and the light-limited rate with the
   absorbed-light integral, co-limiting at canopy level.
2. A sunlit/shaded two-leaf canopy (de Pury & Farquhar 1997) with that Kn, using the two-stream direct/diffuse
   fluxes IBIS already computes.

Either must be checked against the tower light-response curves, not only totals, given Bonan's warning about
shaded-leaf overshoot.

## 23. Stage B: the screening redone on the corrected harness (2026-09-25)

**Setup.** 28 towers of the 2008-2012 batch (up to 4 per class) were run on `default_seasonal.yaml` (both Stage A
formulations on), with composition prescribed and leaf carbon held at the site-attribute LAI (`--fixed-lai attrs`).
There were 52 one-at-a-time perturbations: the 32 of section 20 and the 20 of the new constants
(`--set seasonal`). Each is one 5-year cycle from a shared base, scored on 2010-2012. The sweep ran as three GPU
processes (`$V/site_runs/stageB_standard_a`, `_b`, `stageB_seasonal`); their bases agree to the last bit. Ranking:
`diagnostics/stageB_rank.py` -> `$V/site_runs/stageB_ranking.txt`.

**Base by class** (model/tower unless stated):

| class | GPP | Jun-Jul GPP | NDJF share (model) | leaf-out DOY | senescence DOY | ET | summer LAI |
|---|---|---|---|---|---|---|---|
| DBF | 0.63 | 0.44 | 0.013 | 96 | 285 | 1.62 | 5.7 |
| ENF | 0.55 | 0.48 | 0.079 | 92 | 318 | 1.07 | 4.5 |
| MF | 0.92 | 0.64 | 0.062 | 89 | 314 | 1.50 | 6.5 |
| EBF | 1.05 | 0.99 | 0.334 | 86 | 229 | 1.32 | 3.6 |
| GRA | 1.16 | 3.07* | 0.097 | 76 | 257 | 1.48 | 3.7 |
| OSH | 2.64 | 2.92 | 0.213 | 169 | 271 | 1.36 | 2.1 |
| SAV / WSA | 1.07 / 1.18 | 0.45 / 0.96 | 0.49 / 0.29 | | | 1.20 / 1.24 | 1.7 / 2.4 |

*US-Var is a winter-growing grassland; its near-zero summer GPP makes the Jun-Jul ratio meaningless for the class.

**What moves what, per forest class** (class-mean change):

- **ENF.**
  - Live: vmax x1.4 raises the GPP ratio by 0.19. `acclim_smax` x1.5 / x0.6 changes it by -0.15 / +0.12.
    `acclim_s0` +/-3 C changes it by -0.10 / +0.08 and moves both season edges by about 15 d.
  - Dead at these towers (|dGPP| < 0.01): `acclim_tau`, `stressfac`, `fmax`.
  - The running C1 ENF fit carries `acclim_tau` and `stressfac`. Expect them to be poorly determined; their
    fitted values should not be read as findings.
- **DBF.**
  - Live: vmax x1.4 gives +0.21.
  - Leaf-out: `gdd_thresh_u` +150 / +300 / +600 C d delays it by 14 / 25 / 47 d. `gthresh_u_c` +5 C delays it by
    23 d.
  - Senescence: `sen_ycrit_u` x2 / x0.5 moves it by +18 / -14 d. `sen_tb_u` -4 / +4 C moves it by +16 / -14 d.
    These are the same lever for practical purposes. `sen_pstart_u` is weaker.
  - Water stress is inert at these towers.
  - The leaf-out threshold had a fit bound of 250 C d (default 100). The PhenoCam timing (-30 d) needs about
    +300, so the bound is now 1000 (commit `ff0e39d`).
- **MF.** Live: vmax (+0.30), then the acclimation constants. Phenology is secondary.
- **GRA / OSH.** As in section 20: vmax, `soil_beta_power` (+0.25 GRA, +0.79 OSH), and the lower-canopy thresholds
  (`tthresh_l_c` +10 C: -0.66 GRA, leaf-out +55 d).

**For C1.**
- ENF is running with capacity plus acclimation.
- DBF was launched at 07:50 on vmax (`temp_bl_dec`, the only live deciduous vmax under the F77 priority rule),
  `gdd_thresh_u` and `sen_ycrit_u`, with `sen_tb_u` and `sen_pstart_u` fixed to avoid the Tb/Ycrit degeneracy.
  Run: 150 evaluations, 4 workers, `$V/site_fit/C1_DBF`.
- In both classes, vmax will absorb the canopy-scaling gap of section 22. Its fitted value is an effective
  canopy capacity, not a leaf trait, until that gap is decided.

## 24. C1 ENF: the fixed-vegetation fit of the evergreen conifers (2026-09-25)

**Verdict: not an acceptable fit.** Summer GPP at the temperate conifer towers stays at 55-70 % of the towers'
(DE-Tha about 200 vs 350 gC m-2 month-1), and annual GPP at 0.73-0.80. The loss fell because the baseline was
poor, not because the model reproduces the towers. Capacity went to its bound and the magnitude did not close; that
is the structural limit of section 22. The fitted values are not to be used. The C1 fits are repeated after the
canopy-capacity scaling is addressed. (Verdict added 13:00 after Hangkai's review of the figures: "the magnitude is
off by 50 %". The first write-up below led with the loss reduction and should not have.)

**Setup.**
- 17 ENF towers of the 2008-2012 batch: 11 training towers and 6 held out (seed 1). 2008 is warm-up, 2009-2011
  train, and 2012 is held out at every tower.
- `default_seasonal.yaml` (A1 and A2 on, F77 canopy). Composition prescribed, leaf carbon held at the site-attribute
  LAI where one exists (else the grid state's).
- Loss of section 22: monthly GPP (mean of NT and DT) and LE_CORR, w_et 0.5, QC >= 0.5.
- CMA-ES over six parameters, 209 evaluations, sigma 0.2, population 8, 4 GPU workers
  (`$V/site_fit/C1_ENF`).
- The replays (`C1_ENF_replay_baseline`, `C1_ENF_replay_best`) reproduce the fit's first and best losses exactly
  (21.6739, 15.3899) and add monthly NEE. Report: `diagnostics/c1_report.py` -> `C1_ENF_report.txt/.png`.

**Result.**

| | baseline | best (eval 202) |
|---|---|---|
| loss, training towers | 21.67 | 15.39 (-29 %) |
| held-out towers | 20.43 | 15.25 (-25 %) |
| held-out year 2012 | 18.89 | 13.56 (-28 %) |
| median GPP ratio, train / held out | 0.51 / 0.57 | 0.73 / 0.80 |
| median NDJF share, train (towers 0.055) | 0.063 | 0.071 |
| median ET ratio, train / held out | 0.84 / 0.94 | 0.88 / 0.98 |
| median monthly NEE RMSE, train / held out (gC m-2) | 65 / 70 | 52 / 53 (not fitted) |

The improvement holds on the held-out towers and the held-out year about as well as on the training towers, so
the fit generalises within the class. The best was still improving slowly at 209 evaluations (15.64 at 157), so the
optimum is not sharply resolved.

**Parameters.**

| parameter | start | best | bound |
|---|---|---|---|
| vmax temp_con_ever | 30.0 | 40.3 umol m-2 s-1 | [15, 50] |
| vmax boreal_con_ever | 25.0 | 39.6 | [15, 40], at the bound |
| acclim_s0 | -3.06 C | -4.20 C | |
| acclim_smax | 17.7 C | 20.0 C | |
| acclim_tau | 10.9 d | 2.9 d | inert here (section 23) |
| stressfac | -5.0 | -4.5 | inert here (section 23) |

- **The summer deficit is not closed.** Boreal conifer capacity sits at its bound and GPP stops at three quarters
  of the towers'. Temperate summer peaks stay far below (DE-Tha about 200 vs 350 gC m-2 month-1; IT-Lav 220 vs
  355). This is the canopy-scaling gap of section 22 showing through the fit: the fitted vmax is an effective
  canopy capacity, 30-60 % above its start, and it should not go to the domain as a leaf trait until that gap is
  decided.
- **The acclimation constants moved modestly** and keep the winter right.
- **The two inert parameters wandered.** A 2.9 d tau is not a finding.
- **Per tower:** CA-TP1 overshoots (1.21), a young plantation run with the grid state's LAI 4.0 for want of a site
  value. Towers without site LAI should get a measured value before C2.
- **NEE stays far off,** a near-neutral summer against tower sinks of 100-250 gC m-2 month-1. There is too little
  GPP, and the respiration comes from the grid cell's soil pools (IT-Lav bias +170). NEE belongs to C2, as planned.

## 25. C1 DBF: the fixed-vegetation fit of the deciduous forests (2026-09-25)

**Verdict: timing improved; magnitude not acceptable.** The summer magnitude is far off, as at the conifers:
- The peak month of model GPP is at a median 0.56 of the towers' (0.36-0.79; DE-Lnf 144 vs 405, US-Oho 205 vs 418,
  DK-Sor 239 vs 462 gC m-2 month-1).
- June-August is at 0.61, and annual GPP at 0.65 (training) and 0.75 (held out). The fit did not move these (0.65
  -> 0.65).
- The loss reduction is almost all timing. The fitted values are not to be used for magnitude. They are repeated
  after the canopy scaling is addressed.

**Setup.**
- 12 DBF towers: 8 training, 4 held out.
- `default_seasonal.yaml`, fixed leaf area, loss of section 22.
- CMA-ES over `vmax:temp_bl_dec`, `gdd_thresh_u` and `sen_ycrit_u`: 153 evaluations, population 8, 4 GPU workers
  (`$V/site_fit/C1_DBF`).
- The replays reproduce the first and best losses exactly (36.4487, 23.4981).

**Result.**

| | baseline | best (eval 144) |
|---|---|---|
| loss, training / held-out towers / held-out year | 36.45 / 87.49 / 70.06 | 23.50 / 49.23 / 40.69 |
| vmax temp_bl_dec | 30.0 | 36.6 umol m-2 s-1 (bound 60, not binding) |
| gdd_thresh_u | 100 | 310 C d (above the old bound 250) |
| sen_ycrit_u | 300 | 169 C d |
| leaf-out vs PhenoCam (4) / tower GPP (11) | -26 / -28 d | -9 / -10 d |
| senescence vs PhenoCam (4) / tower GPP (11) | +4 / +7 d | -10 / -8 d |

The timing columns come from `site_run.py --fixed-lai attrs` runs with the baseline and the fitted yaml
(`site_runs/C1_DBF_timing_{baseline,best}`, `phenology_vs_phenocam.py`). Neither PhenoCam nor the daily dates were
in the loss.

- Leaf-out comes within about 10 days.
- Senescence now overshoots by about a week early.
- vmax stayed low. The canopy's flat light response means more capacity inflates the spring and autumn shoulders,
  the months of small tower GPP and small sigma, before it lifts the summer. This is the same structural limit as
  in section 24, seen from the loss side.
- IT-Ro2 (Mediterranean oak) collapses in summer with ET, a model drought that none of these parameters touch.

## 26. The canopy-capacity scaling: implemented as a switch, first tower test (2026-09-25)

**Why.** Hangkai's review of the C1 figures: "the magnitude is off by 50 %". This is right, and no parameter
closes it (sections 22, 24, 25). The structural cause, measured by the probe of section 22: F77 scales the top
leaf's rate by the light-profile integral, so a forest canopy has the capacity of about 1 m2 of top leaf.

**What (commit `d2e65d4`).** `stomata: cap_scale_u / cap_scale_l` (static switches, off = F77, bit-identical;
full suite 708 passed; `tests/test_canopy_capacity.py`):
- The Rubisco and sink limits (C4: the CO2 limit) and dark respiration are scaled by
  f_cap = (1 - e^-Kn L)/(Kn L).
- The light limit keeps F77's f_light.
- The canopy rate is the co-limitation of the scaled limits.
- With f_cap = f_light this is F77 exactly.
- `cap_kn` defaults to 0.15 (Lloyd et al. 2010: 0.13-0.23), is traced and fittable.

**First test.** The 29 ENF and DBF towers of the 2008-2012 batch were run with fixed leaf area, both Stage A
formulations on and default parameters, with the switch off and on for the upper canopy:
`site_runs/capscale_AB_{off,capu}`, `diagnostics/magnitude_ab.py` -> `capscale_AB_magnitude.txt`, and
`light_response.py`. Class medians, model/tower:

| | ENF off | ENF on | DBF off | DBF on |
|---|---|---|---|---|
| peak month | 0.53 | 1.20 | 0.45 | 1.03 |
| June-August | 0.49 | 1.19 | 0.50 | 1.06 |
| annual | 0.50 | 1.28 | 0.65 | 1.40 |
| ET | 0.91 | 1.01 | 1.03 | 1.19 |
| summer days SW < 150 / >= 250 W m-2 | 0.67 / 0.45 | 1.24 / 1.19 | 0.65 / 0.46 | 1.11 / 1.10 |
| light-response slope, gC m-2 d-1 per 100 W m-2 (towers 1.69 / 2.13) | 0.20 | 1.60 | 0.64 | 3.11 |

**Reading.**
- **The diagnosis is confirmed.** Given the capacity of its leaves, the canopy reaches the towers' summer magnitude
  with default capacity, and at the conifers the light response acquires the towers' slope.
- **It overshoots in two ways, so it is not adopted as it stands.**
  1. On dim days (daily SW < 75 W m-2) the model makes 7-8.5 gC m-2 d-1 against the towers' 4-5 (DE-Tha, FI-Hyy,
     DE-Hai, DK-Sor). The capacity limit no longer hides the light-limited canopy rate, and co-limiting at canopy
     level credits the whole canopy with the top leaf's light efficiency. This is the big-leaf overestimate of
     de Pury & Farquhar (1997) and the shaded-leaf warning of Bonan et al. (2012). F77's stem credit (f_light
     divides by kL, about +25 %) adds to it.
  2. The deciduous season becomes a box. Full GPP is reached in April and held to October, because the default
     leaf-out, four weeks early, is no longer capped.
- **A refit of vmax and phenology can fix the second, but not the first:** the light-limited rate does not depend
  on vmax.
- **Next test:** the same with `scale_leaf_only` (the stem credit removed; with capacity scaled separately it
  lowers only the light-limited part). Then, if the dim-day excess remains, a sunlit/shaded two-leaf canopy
  (option 2 of section 22).

**Second test: capacity scaling plus `scale_leaf_only` (added 14:00).** The same 29 towers and defaults, with the
stem credit also removed (`site_runs/capscale_AB_capu_leafonly`). Class medians, model/tower:

| | ENF | DBF |
|---|---|---|
| peak month | 1.06 | 0.92 |
| June-August | 1.04 | 0.93 |
| annual | 1.22 | 1.33 (shoulders: default leaf-out) |
| ET | 1.00 | 1.18 |
| summer days SW < 150 / >= 250 W m-2 | 1.03 / 1.07 | 0.91 / 0.99 |
| light-response slope (towers 1.69 / 2.13) | 1.80 | 3.19 |

- At default capacity, the conifer summer matches the towers in magnitude and in light response.
- Per tower, 7 of 17 ENF towers follow the towers' light response closely.
- The departures are site-specific:
  - Too high at CA-TP1 (a young plantation on the grid state's LAI), CH-Dav, IT-Ren and US-NR1.
  - Too low at the summer-dry US-Me2 and IT-SRo.
- The dimmest bin (SW < 75) stays 1.5-2.5x the towers at several towers. There are few days in it, but the
  residual is systematic; watch it.
- This configuration is `configs/params/default_seasonal_canopy.yaml` (commit `0ccea86`). The C1 fits are
  repeated on it from 14:05: `$V/site_fit/C1b_ENF` (vmax x2, acclim_s0, acclim_smax; Kn held at 0.15) and
  `C1b_DBF` (vmax, gdd_thresh_u, sen_ycrit_u). They are judged first on the peak-month and June-August ratios
  and the light response, then on the loss.

**The hourly probe at the deciduous towers (added 14:15).** Same probe as section 22, with the fitted C1 DBF parameters
(`$V/site_fit/C1_DBF/params_C1_DBF_best.yaml`, F77 canopy), July 2010, 12 DBF towers
(`$V/site_fit/plateau_probe_DBF_201007.{npz,txt}`). Daylight medians:

| quantity | value |
|---|---|
| top leaf limit: light / Rubisco / sink | 12 / 82 / 6 % of hours |
| vmax at leaf T | 47.5 umol m-2 s-1 |
| ci | 282 ppm |
| soil stress | 0.94 |
| wet fraction | 0.03 |
| top-leaf Ag | 13.6 umol m-2 s-1 |
| k | 2.2 |
| LAI | 5.4 (range 4.1-8.6) |
| effective leaf area | **1.01 (range 0.97-1.03)** |
| GPP, model / tower | 7.5 / 12.0 gC m-2 d-1 |

The effective leaf area is the same at every tower whatever the LAI: US-Ha1 at LAI 8.6 has 1.006, FR-Fon at LAI 4.1
has 0.967. This confirms that the deciduous deficit of section 25 is the canopy scaling, not leaf area, water
(except IT-Ro2, stress 0.32) or wet leaves.

**Order (Hangkai, 14:20).** "Currently we do not need to run globally, just fit for the site point first." The global
with/without test of the canopy switch is deferred. The site fits (C1b ENF and DBF, then the remaining classes) come
first, and they are checked at held-out towers and years only.

## 27. The canopy scaling traced to its sources: k, the capacity profile, vmax (2026-09-26)

**Order (Hangkai, 2026-09-26).** "Check all the related variables, and double check how others solve this issue online, like
the Beer's law, the k factor, can we change it? Also check the vcmax, is it wrong? Dig into it rather than change the
parameters without any clues." Full record with figures: `ibis_local_document/CANOPY_SCALING_INVESTIGATION_2026-09-26.md`.

**The two Harvard fits (1994-2002, LAI 4.7, CO2 series; `$V/site_fit/Ha1_{A_original,B_corrected}`)** both reach a
monthly loss of 18 (from 47.4 and 206.7) by compensation: A raises vmax to 44.3e-6 (68 umol m-2 s-1 at 25 C, 1.5x TRY)
with the stomatal slope at 15; B lowers it to 23.5e-6 (36, 0.6x TRY) with the slope at 5.8 (ci 240 ppm). Replayed hourly
on July 2010 (`diag_US-Ha1/{f77,canopy}_{default,fitHa1}_L47.npz`, `model_hourly_lr.py`) both give alpha 0.030 against
the tower's 0.065-0.068 and 25-26 umol m-2 s-1 in bright hours against 29 (21-yr July mean) to 37 (2010).

**What the F77 canopy does (physiology.f, comment block at the scaling):** "an(x) is proportional to apar(x)"; the top-leaf
rate is multiplied by (1 - e^-k(L+S))/k with k = extpar, the two-stream extinction of THIS HOUR's absorbed PAR. The light-
limited part is Beer's law exactly (alpha x the leaves' share of APAR); the capacity-limited part gets 1/k top leaves:
1.4 at noon (k 0.70), < 1 at low sun (k 1.7 at 07/17 h, clipped at 10 at 06/18 h), 1.00 on the daylight average at LAI
4.8 and 1.01 at LAI 8.6. Sellers (1992), whom IBIS-2 (Kucharik 2000) follows, argued with the TIME-MEAN light profile;
IBIS-1 integrated sunlit and shaded leaves. Every other variable of the big leaf is ordinary in July (hourly record,
`model_hourly_lr.py`): top leaf Rubisco-limited 80 % of daylight hours, vmax(T) 45 = 30 x 1.52 x 0.988 (no stress), leaf
T 25 C, ci 289 ppm, wet fraction 0.04, gs 0.22.

**k.** Not a free parameter (two-stream with chifuz 0, rhoveg 0.10, tauveg 0.07): beam 0.5/cos z, diffuse ~0.9, blend
0.70 at noon, 0.79 light-weighted. Measured daily k for broadleaf forests 0.59 (Zhang et al. 2014 meta-analysis; PnET at
Harvard 0.58). Lowering k for light is wrong for radiation and insufficient for capacity: canopy capacity
(1 - e^-Kn L)/Kn at LAI 4.7 is 0.99 top leaves at Kn 1.0, 1.4 at 0.7 (time-mean), 1.8 at 0.5, 2.5 at 0.3, 3.4 at 0.15.
Observed Kn 0.13-0.23 (Lloyd et al. 2010: loge kV = 0.00963 Vmax(0) - 2.43; "kP should always be substantially less than
kI"); de Pury & Farquhar 1997 kn 0.713 per canopy (0.18 per LAI); JULES kn 0.78 per canopy; CLM4 0.11, CLM4.5 Lloyd's
relation. Capacity needs its own coefficient.

**The reference (`diagnostics/canopy_reference.py`):** the record's hourly inputs (tower PPFD, big-leaf T, ci, vmax(T,
stress), L, S) integrated with IBIS's own leaf model in a 60-layer sunlit/shaded canopy (de Pury & Farquhar light model,
kb 0.5/sin(elev), kd 0.78, scattering 0.15, diffuse fraction from the clearness index, stems attenuate, leaves assimilate,
V0 e^-Kn x). July 2010 at LAI 4.8, bright hours (> 1500 umol m-2 s-1) / tower 2010 36.8 / tower 21-yr 30:

| canopy | alpha | bright | dim (< 500) | bright / 21-yr tower |
|---|---|---|---|---|
| F77 record | 0.023 | 17.2 | 4.3 | 0.57 |
| sunlit/shaded multilayer, Kn 0.15, IBIS vmax | 0.059 | 24.1 | 7.7 | 0.80 |
| two-big-leaf sunlit/shaded closed form | 0.062 | 24.5 | 7.9 | 0.82 |
| multilayer, layer-mean light (no split) | 0.047 | 32.6 | 7.9 | 1.09 |
| one big leaf, Kn 0.15, canopy-level co-limitation (= cap_scale) | 0.058 | 39.5 | 8.5 | 1.32 |
| big leaf, time-mean k 0.79 (Sellers) / Kn 0.5 | 0.089 / 0.076 | 15.2 / 22.2 | 7.5 / 8.1 | 0.51 / 0.74 |
| reference + TRY vmax (57.7) / + Bernacchi T / both | 0.057 / 0.050 / 0.049 | 27.9 / 25.8 / 30.0 | 8.0 / 7.5 / 7.8 | 0.93 / 0.86 / 1.00 |

Typical year (2001 record; tower alpha 0.065, bright 28.6): F77 0.60, reference 0.87, reference + TRY vmax 1.01
(bins 200-1800: 0.96, 0.99, 0.92, 1.04, 1.04, 1.04, 1.06, 0.99), + Bernacchi 0.99, cap_scale 1.40.

**Readings.**
1. The sunlit/shaded canopy with IBIS's leaves has the tower's low-light response (bins 200-600 within 10-15 %) and
   80-87 % of its bright-hour rate; the closed-form two-big-leaf is within 2 % of the multilayer.
2. `cap_scale` (section 26) has the right capacity total and the wrong integration: co-limiting once at canopy level
   (+20 %) on top of the layer-mean light (+35 %) gives +64 % over the reference at LAI 4.7, +97 % at 8.6 -- de Pury &
   Farquhar's big-leaf error (20 % at LAI 4, 45 % at 6; "no distribution of nitrogen or capacity that allows the big-leaf
   model to work directly"). The B fit hid it with vmax 36 and slope 5.8. `scale_leaf_only` also discounts the LIGHT-
   limited rate for stems a second time (topparu already carries L/(L+S)): x0.74-0.80 here.
3. vmax: 30e-6 at 15 C = 46 at 25 C vs TRY 57.7 (temperate deciduous; Kattge 2009; JULES9 57.3) -> 0.80. The temperature
   function exp(3500 (1/288 - 1/T)) is 29 kJ mol-1 (Q10 1.5) vs 65.3 measured (Bernacchi 2001) / 71.5 with deactivation and
   acclimation (Kattge & Knorr 2007): 1.21x at 30 C vs 1.40-1.54x; 0.65x at 15 C vs 0.39-0.40x. kc, ko, gamma* match Collatz
   at 25 C. Whole table (IBIS at 25 C / TRY): tropical evergreen 99.5 / 41.2 (2.4x; Yuan et al. 2014 inverted -50 % for EBF
   and +24..61 % for every other type at 62 towers), warm-temperate evergreen 61 / 61, temperate conifer 46 / 54-62,
   boreal conifer 38 / 54-62, shrubs 42 / 50-62, C4 23 / 32-52, C3 38 / 51-78.
4. Neither vmax nor k is the main gap; the canopy integration is. With the integration right, TRY vmax (and the measured
   temperature response) puts the July light response on the 21-year tower mean at every light level.

**Proposal (decisions pending with Hangkai, section 8 of the local record):** sunlit/shaded two-big-leaf canopy as a
switch (bit-identical off) reproducing `canopy_reference.py` at Harvard; Kn per PFT from Lloyd's relation; TRY vmax
defaults as a second parameter file; temperature-response switch (IBIS / Bernacchi / Kattge-Knorr); then refit Harvard
within TRY bounds and judge on the hourly light response. `cap_scale` and `scale_leaf_only` would be superseded.

**Tools.** `diagnostics/tower_light_response_hourly.py <SITE> <out>` (per-year alpha/Amax, 200-umol bins, hourly csv),
`diagnostics/model_hourly_lr.py <record> <hourly.csv>` (every variable of the big leaf by hour; July extract),
`diagnostics/canopy_reference.py <extract> <lat> [LAI SAI] [--vmax-scan]`, `site_diag_year.py` now passes extra
`site_fit` options (`--lai`, `--co2`). Tower facts (US-Ha1 July 1992-2012, NEE QC 0, NT/DT mean; the two partitions
agree within 5 %): alpha median 0.065 (0.050-0.082), Amax 40 (33-59), GPP above 1500 umol m-2 s-1 28.6 (23.7-37.5;
2010 the highest); apparent yield at 100-400 umol m-2 s-1 0.050 per absorbed photon = IBIS's leaf-level 0.055.

**Implemented (2026-09-27, Hangkai: "Yes, try that first!"): `stomata: sunshade_u / sunshade_l`.** The two-stream's
absorbed-PAR profile per unit plant area, A e^-kx + B e^-hx + C e^hx (its integral over the plant area equals the flux
the two-stream absorbs: `tests/test_sunshade_canopy.py`), is split into the unscattered beam, (1 - omega) k I_beam e^-kx
(`radiation.RadiationResult.dbeamu/dbeaml`, new), taken by the sunlit plant area e^-kx, and the diffuse and scattered
field taken by all leaves; capacity per unit leaf area falls as e^-Kn x (`cap_kn`); leaves are the share L/(L+S) of the
plant area and the only assimilating surface; each big leaf is integrated in one leaf-model call on its totals (the
leaf model is homogeneous of degree one), dark respiration follows the capacity profile, the Ball-Berry state relaxes
on the canopy-mean rate, the canopy conductance is unchanged. Off = bit-identical (full suite). Parameter file
`configs/params/default_seasonal_sunshade.yaml` (both canopies on, Kn 0.15). `site_diag_year.py` now also records the
model's own hourly visible direct/diffuse radiation; `canopy_reference.py --model-light` integrates the reference under
that light.

**Harvard check (LAI 4.7, IBIS vmax, July 2010 / 2001; `diag_US-Ha1/sunshade_default_L47_{2010,2001}.npz`):**

| July daily GPP, gC m-2 d-1 | 2010 | 2001 |
|---|---|---|
| tower | 13.8 | 12.4 |
| IBIS as ported (F77 canopy) | 6.4 | 6.4 |
| **sunlit/shaded canopy in the model** | **8.65** | **8.85** |
| reference two-big-leaf, the model's own light, x (1 - fwet) | 9.30 | - |
| reference two-big-leaf, the model's own light | 9.66 | - |
| reference two-big-leaf, the tower's hourly light | 10.95 | 11.4 |

- The wiring is exact: the canopy rate is reproduced to 4 decimals from the recorded sunlit/shaded partition with the
  reference's leaf model. Bright hours in the model 22.5-23.3 against 24.5-25.4 in the reference.
- Under identical light the model is 7 % below de Pury's light model (the two-stream with IBIS's leaf optics, omega
  0.17, gives the shaded leaves less) and a further 4 % for wet leaves (fwet 0.07 in dim hours). Both are IBIS's own
  physics, kept.
- The largest piece, 13 %, is the light itself: the site harness feeds daily tower weather and the weather generator
  spreads it into a smooth diurnal cycle with a PAR-weighted diffuse fraction of 0.36 (Spitters on the tower's hourly
  clearness: 0.44). A sunlit/shaded canopy is sensitive to this; the F77 canopy was not. Hourly tower radiation into the
  site harness is the next structural item to decide.
- ET: June-August AET +5 % relative to the F77 canopy (2001: 75/90/100 mm against the tower's 81/92/78).
- Monthly GPP 2010 May-Sep: 260/254/268/274/266 gC against the tower's 305/475/451/298/209: the summer plateau is
  still 0.6 of the tower with IBIS's vmax and the generator's light; the shoulders are high (default phenology).

**Hourly tower shortwave in the site harness (2026-09-27, Hangkai: "do that next").** `TowerForcingReader(use_hourly_sw=True)`
(`site_fit --hourly-sw`, `site_run --hourly-sw`) returns each day's 24 hourly shortwave values from the tower's HH/HR file
(SW_IN_F_MDS, SW_IN_F where missing; the mean of the two intervals around each hour, so the value is the instant the
generator's step represents; days with a gap keep the generator); `run_month_gswp3` carries them as two more scan inputs
under a STATIC flag and `weather.observed_shortwave` replaces the generator's direct and diffuse fluxes with the
observed total split by Spitters et al. (1986) hourly diffuse fraction and the generator's band fractions. Off is the
original graph (a mask-off replay reproduced the earlier record to the bit only after the flag was made static; a
`where` on the same arrays changed XLA's fusion at the 1e-5 level, which a two-year spin-up amplified).

**Found on the way: the weather generator's latitude (DEFECTS_FOR_REVIEW.md "solar latitude").** The generator's
daylight at Harvard ran 06-18 h in every month and its shortwave was 2.7x the forcing in January, 0.85x in July.
`SimParams.latitude` is radians, `diurnal` converts from degrees (F77), both day loops passed the radians through:
every cell in every run had the equator's solar geometry. Fixed at both calls (`latitude_degrees`), guarded by
`tests/test_solar_latitude.py`. Harvard, sunlit/shaded canopy, IBIS vmax, 2010:

| | day length Jun / Dec | generator SW / forcing, Jan..Dec | GPP Jun-Aug / tower | July bright hours / tower (2010; 21-yr) |
|---|---|---|---|---|
| before (generator) | 13 / 11 h | 2.7 .. 0.85 .. 3.0 | 0.65 | 0.61; 0.75 |
| generator, latitude fixed | 15 / 9 h | 1.02-1.07 every month | 0.74 | 0.60; 0.74 |
| hourly tower shortwave, latitude fixed | 17 / 11 h (tower's own) | 1.00 | 0.76 | 0.63; 0.77 |

- With the geometry right, the generator's daylight is within 3 % of the tower's hourly light for June-August GPP:
  most of the "13 % generator effect" of the previous entry was the latitude.
- The hourly path gives the model the tower's own light response: alpha 0.066 against the tower's 0.068, the 200-600
  bins at 0.87-0.89 of the 21-year mean; the plateau, 23 against 30 (21-year) and 37 (2010), is capacity.
- The annual total looks better than it is (0.92-0.95 of the tower) because the default phenology holds April at 237-254
  and October at 146-154 gC against the tower's 90 and 80.
- Every earlier site and global result was produced on the equator's insolation. The class-wide cold-season excess of
  section 20 and the mid-latitude amplitude of section 16 have to be re-read on the corrected model before anything
  else is concluded from them.

## 28. The 62-tower baseline on the corrected solar geometry (2026-09-27)

**Order (Hangkai).** "The 62-tower baseline rerun on the corrected model first, then continue the Harvard site."
Runs: `site_runs/latfix_default_2008-2012_fixedlai` (the exact configuration of `default_2008-2012_fixedlai`,
section 20: `default.yaml`, leaf area held at the site attributes, 2 spin cycles, generator daylight) and
`site_runs/latfix_sunshade_2008-2012_fixedlai` (`default_seasonal_sunshade.yaml`: Stage A acclimation and photoperiod
senescence, sunlit/shaded canopy, Kn 0.15, IBIS vmax); 62 towers, 2008-2012, GPU, 43 min each. Metrics
`site_metrics.csv` (cold-season share, r, peak lag) and `diagnostics/magnitude_ab.py` (peak, June-August, annual, ET;
LE_CORR where the tower has it). Class medians, model / tower:

| class (n) | cold-season share: before / lat fixed / tower | r: before / lat fixed | JJA: before / lat fixed / **+ two-leaf** | peak: + two-leaf | annual: + two-leaf | ET: lat fixed / + two-leaf |
|---|---|---|---|---|---|---|
| ENF (17) | 0.16 / 0.06 / 0.05 | 0.92 / 0.99 | 0.54 / 0.63 / **0.96** | 0.97 | 0.97 | 0.82 / 0.86 |
| DBF (12) | 0.12 / 0.04 / 0.01 | 0.81 / 0.90 | 0.49 / 0.58 / **0.84** | 0.84 | 1.05 | 0.93 / 0.97 |
| MF (6) | 0.16 / 0.06 / 0.03 | 0.86 / 0.97 | 0.60 / 0.71 / **1.10** | 1.11 | 1.20 | 0.94 / 1.03 |
| GRA (13) | 0.15 / 0.05 / 0.06 | 0.89 / 0.94 | 0.58 / 0.66 / **1.06** | 0.96 | 1.10 | 0.84 / 0.85 |
| EBF (3) | 0.31 / 0.20 / 0.22 | 0.27 / 0.90 | 0.92 / 0.94 / **1.03** | 1.05 | 0.98 | 0.98 / 0.99 |
| WSA (3) | 0.22 / 0.21 / 0.18 | 0.77 / 0.88 | 0.88 / 0.93 / **1.01** | 0.98 | 1.34 | 0.95 / 0.96 |
| SAV (2) | 0.15 / 0.14 / 0.30 | 0.85 / 0.85 | 1.23 / 1.29 / **1.40** | 1.32 | 1.07 | 0.94 / 0.95 |
| OSH (4) | 0.26 / 0.22 / 0.10 | 0.78 / 0.90 | 3.7 / 4.0 / **6.1** | 4.1 | 5.0 | 1.41 / 1.42 |
| all (61) | median GPP ratio 0.85 / 0.80 / 1.10; within 0.8-1.25: 34 % / 25 % / 52 % | 0.86 / 0.95 / 0.93 | | | | |

**Readings, verdict first.**
1. **The class-wide cold-season excess was the latitude bug.** With the geometry right and nothing else changed, the
   cold-season share of GPP falls to the towers' in every temperate class (ENF 0.06 vs 0.05, DBF 0.04 vs 0.01, GRA
   0.05 vs 0.06, MF 0.06 vs 0.03), the monthly correlation rises from 0.86 to 0.95 and the conifer peak lag from +1
   month to 0. Section 20's "one robust class-wide error" and the winter side of section 16 are closed by the fix,
   not by a parameter.
2. **The summer deficit is the canopy, as section 27 said.** Light alone lifts June-August from 0.5-0.6 to 0.6-0.7 of
   the towers; the sunlit/shaded canopy at IBIS's default vmax puts ENF at 0.96, GRA at 1.06, EBF at 1.03, WSA at
   1.01 and MF at 1.10. Deciduous broadleaf stays at 0.84, Harvard's 0.76 among them: the temperate deciduous vmax (46
   at 25 C against TRY's 57.7) is the remaining lever there.
3. **The dry classes are now the largest error and are not a light or canopy problem.** Open shrubland is 6x the
   towers in summer, savanna 1.4x, with ET 1.4x at the shrublands: water stress and the lower canopy at semi-arid
   sites (US-SRC/SRG/SRM/Whs/Wkg, AU-DaS/Dry/How) is the next class-wide item after the deciduous vmax.
4. The annual ratios above 1 in the new baseline (DBF 1.05, MF 1.20, GRA 1.10, WSA 1.34) come with summer ratios at
   or below 1: the default phenology's long season (section 25) is now the deciduous classes' visible error.
5. ET moved from 0.95-1.12 to 0.82-0.93 with the latitude fix at the forests and grasslands and back to 0.85-1.03
   with the two-leaf canopy; the earlier "forest ET 1.2-1.6x" (section 20) was also the equatorial light.

**Consequence for the record.** Every result in sections 14-26 that depends on the daily loop's light was produced
under the equator's insolation: the S1.3 baselines, both sensitivity sweeps, the Stage B ranking, the C1/C1b fits and the
Harvard fits of section 27. The findings that survive are the ones diagnosed on hourly records against the tower's own
light or by construction of the code (the phenology clock, the surface pressure, the prescription hold, the canopy
capacity profile); the numbers do not. Stage A's acclimation and photoperiod senescence stay in as formulations; their
fitted constants are to be re-read. The local record of 09-26 is superseded on its Harvard numbers by section 27's
corrected table.

## 29. Harvard on the corrected model: the fit within the TRY range (2026-09-27)

**Order (Hangkai).** "Fit vmax within the TRY range" (temperate deciduous Vcmax25 57.7 +- 21.2 umol m-2 s-1, Kattge et
al. 2009 as tabulated by Ma et al. 2022 ESSD; `site_fit --vmax-bounds 36.5,78.9`, converted to IBIS's 15 C basis with its
own temperature function: 23.9-51.6e-6). Configuration: `default_seasonal_sunshade.yaml`, LAI 4.7, CO2 series,
`--hourly-sw`, corrected latitude; fitted vmax:temp_bl_dec, gdd_thresh_u, gdd_ramp_u, sen_ycrit_u, coefm_bl on 1994-2002
(monthly GPP + 0.5 ET); 127 evaluations, 367 s each, 3 CPU workers (`$V/site_fit/Ha1_C_sunshade_hourly`). Held out:
2003-2012 (`daily_{fitted,default}_2003-2012.npz`, `diagnostics/site_daily_replay.py`; `site_replay_report.py`); the
hourly year 2010 (`diag_US-Ha1/sunshade_fitC_hourly_L47_2010.npz`).

**Result.** Loss 91.99 -> 10.44 (held-out years: 49.7 -> 18.0). vmax 31.8e-6 = 48.6 at 25 C, inside the range and
close to IBIS's 45.9, the optimiser having explored 36-68; leaf-out 533 degree-days, ramp 108, senescence Ycrit 211,
stomatal slope 10.7 (default 10). Held-out 2003-2012, model / tower:

| | peak month | June-August | annual | ET (LE_F_MDS, uncorrected) | leaf-out 25 / 50 % (DOY) | senescence 50 / 25 % |
|---|---|---|---|---|---|---|
| tower GPP | | | | | 130 / 148 (camera 120 / 129) | 259 / 282 (camera 282 / 291) |
| default parameters | 0.79 | 0.97 | 1.03 | 1.41 | 102 / 107 | 282 / 295 |
| **fitted** | **0.84** | **1.02** | **0.83** | 1.37 | 143 / 147 | 278 / 288 |

July 2010 hourly light response, fitted: alpha 0.068 (tower 0.068); plateau 24.6 against 30 (21-year July mean, 0.82)
and 36.8 (2010, 0.67). 2003-2012 daily climatology: plateau (July-August) 10.5 against 11.7 (0.90); July 0.85.

**Readings, verdict first.**
1. **The summer is at 0.85-0.90 of the tower and the fit did not buy more with vmax.** With the light response's slope
   exact and the plateau 10-18 % low, raising vmax above 49 would also raise April-May and ET; the monthly loss
   preferred a later leaf-out. The remaining plateau gap is the leaf temperature response and the July of 2010 (an
   exceptional year), not something a bounded vmax reaches.
2. **The shape of the spring is the largest remaining error, and no phenology parameter can fix it.** The forest's GPP
   rises over 40 days (25 / 50 / 90 % of the plateau on DOY 124 / 141 / 164) while its greenness completes leaf-out in
   about ten (camera 120 -> 129): the leaves are there by mid-May, their photosynthetic capacity matures until mid-June.
   IBIS switches on the full capacity with the leaf area: the model's LAI goes from 10 to 90 % in eight days (DOY
   130-138 fitted, 88-93 default) and its GPP with it (25 / 50 / 90 % on 141 / 147 / 153). The optimiser can only slide
   that step: it put it in the middle of the tower's ramp, three weeks later than the camera's leaf-out, and lost April-
   May (annual 0.83 with the summer at 1.0). The autumn has the same signature: the tower's plateau starts to fall on
   DOY 225 and is halved on 271; the model holds until 243 and halves on 283 (leaf area falls 283-295) -- capacity
   declines with leaf age before the leaves drop. Wilson, Baldocchi & Hanson (2001, PCE 24:571) measured exactly this at
   a deciduous forest (Vcmax rising for a month after leaf-out, declining from late summer) and showed a model without it
   misses spring and autumn NEE; ORCHIDEE carries leaf-age classes with a capacity efficiency for the purpose.
3. **ET at Harvard cannot be fitted to LE_F_MDS.** The tower's uncorrected LE gives 300-480 mm yr-1 against 840-1630 mm
   of precipitation (2010: 199, a fill artefact); there is no LE_CORR at US-Ha1. The model's 475-611 is the plausible
   range for this forest. The 0.5 ET term in the loss pulled on a target that is 30 % low; it should be off at this
   tower or replaced by another ET reference.
4. Timing (the fitted step at DOY 143-147 against the camera's 120-129 and senescence 278-288 against 282-291) is not
   a phenology-parameter result to keep: it is the compensation for reading 2.

**Next (proposed, one decision): leaf-age development of photosynthetic capacity** as a switch -- vmax x f_age, f_age
rising from about 0.3 at leaf emergence to 1 over ~30 days and declining after ~100 days of leaf age (Wilson et al. 2001
shapes; ORCHIDEE's efficiency by age class), the phenology dates then re-fitted against the camera. F8 in the local
figures folder (`figures/canopy/F8_harvard_fit_TRY.png`).

## 30. Step 2: one calibration per PFT at its typical tower (2026-09-28, in progress)

Configuration as at Harvard (section 29): `default_seasonal_sunshade.yaml`, hourly tower light, corrected latitude,
CO2 series, vmax within the PFT's TRY range (Kattge et al. 2009 via Chen et al. 2022: ENF 62.5 +- 24.7, EBF 43.8 +- 16.8,
DBF 57.7 +- 21.2, shrubs 57.9 +- 19.6, C3 grass 78.2 +- 31.1; C4 grass has no TRY value, 30-75 assumed), 120 evaluations,
loss monthly GPP + 0.5 ET (0 where the tower has no LE_CORR). `site_fit --pft` overrides the IGBP rule at ZM-Mon
(tropical deciduous) and RU-Cok (deciduous shrub); towers outside the 2008-2012 batch run on their own years. Runs
`$V/site_fit/pft_<SITE>`, held-out replays `daily_{fitted,default}_<win>.npz`, `site_replay_report.py`.

**Batch 1 (verdict on held-out years, model / tower):**

| site, PFT | fit years; held out | loss | vmax at 25 C (range) | other fitted | JJA default -> fitted | peak | annual | ET | timing |
|---|---|---|---|---|---|---|---|---|---|
| DE-Tha, temperate conifer | 2004-12; 1997-2003 | 2.90 -> 2.53 | 58.1 (37.8-87.2) | acclim s0 -3.2, smax 21.3, slope 3.5 (bound 3) | 0.97 -> 0.95 | 0.92 | 0.97 | 1.12 -> 1.01 | at the tower's within days |
| FI-Hyy, boreal conifer | 2004-12; 1997-2003 | 6.47 -> 5.47 | **37.9 (lower bound)** | s0 -9.1, **smax 30 (bound)**, slope 7.5 | 1.06 -> 1.02 | 0.99 | 1.04 | 0.63 -> 0.67 (LE_CORR) | at the tower's |
| FR-Pue, holm oak | 2004-12; 2000-2003 | 133 -> 19.1 | **27.0 (lower bound)** | **slope 3.1 (bound), stressfac -3.3** | 1.51 -> 1.03 | 1.75 -> 0.75 | 1.75 -> 0.98 | 1.23 -> 0.85 | season shape wrong both ways |
| DE-Gri, C3 grass | 2005-13; 2004-2005 | 53.9 -> 21.9 | **47.1 (lower bound)** | **slope 3.1 (bound)**, gdd_thresh_l 300 (bound), tthresh_l_c 4.1 | 1.35 -> 1.13 | 1.01 | 1.03 | 0.54 -> 0.40 (LE_CORR) | leaf-out at the tower's; senescence 50 % 276 vs 202 |

Readings: the two conifers are already the towers' at default and the fit changes little (DE-Tha slope 3.5 at the
bound buys ET 1.12 -> 1.01; FI-Hyy's ET at 0.65 of LE_CORR is the open item there). FR-Pue and DE-Gri are structural
fits: capacity to the bottom of the range and the stomatal slope to its bound to hold summer GPP and ET down -- the
Mediterranean drought response (a 1.5x summer with defaults) and the grassland's ET at 0.4-0.5 of the tower with a
matching GPP are not parameter problems. Batch 2 (GF-Guy, IT-Noe, ZM-Mon, CA-Oas) and 3 (RU-Cok, AU-Stp) running.

**Batches 2-3 (all ten sites done 09:24; held-out replays 11:55-13:10):**

| site, PFT | fit years; held out | loss | vmax at 25 C (range) | other fitted | peak default -> fitted | season / annual | ET | note |
|---|---|---|---|---|---|---|---|---|
| GF-Guy, tropical evergreen | 2005-13; 2013-14 | 2.17 -> 2.03 | **60.6 (upper bound)** | slope 8.4, stressfac -7.4 | 1.27 -> 0.99 | annual 1.27 -> 1.03 | 0.85 (LE_F_MDS) | a flat tropical year; fine |
| IT-Noe, evergreen shrub | 2005-13; 2013-14 | 88 -> 62 | **38.9 (lower bound)** | **slope 3.1 (bound)**, stressfac -4.4 | 1.92 -> 1.64 | JJA 1.56 -> 1.77; annual 1.89 -> 1.75 | 1.00 -> 0.93 | summer drought: the model does not stop in July-August; structural |
| ZM-Mon, tropical deciduous | 2001-09 (22 tower months); no held-out data | 32.2 -> 31.0 | 40.0 | drop_stress 0.16, drop_floor 0.21, slope 6.8 | 0.43 (in-sample) | annual 0.61; model 60-100 gC/mo all year, tower 235 wet / 12 dry | - | drought-deciduous phenology inert: the fit moved nothing; structural |
| CA-Oas, boreal deciduous | 2001-09; 1997-2000 | 27.5 -> 2.8 | 50.4 | leaf-out 266 gdd, ramp 150 (bound), Ycrit 127, slope 7.2 | 0.82 -> 0.80 | JJA 0.85 -> 0.84; annual 1.05 -> 0.86 | 1.17 -> 1.05 (LE_F_MDS) | timing fixed (leaf-out 140 vs tower 141); summer 0.84 like Harvard |
| RU-Cok, shrub tundra | 2004-12 (29 tower months); 2013-14 (2 months) | 2.76 -> 2.51 | 40.0 | gdd_thresh_l 50 (bound), tthresh_l_c -0.4, slope 9.3 | 0.92 (in-sample) | annual 0.86; June 60 vs 71, July 118 vs 127 | 0.62 | small fluxes; starts a month late in the held-out spring |
| AU-Stp, C4 grass | 2008-14 (all years, in-sample) | 253 -> 55 | **30.0 (lower bound)** | **slope 15.9 (upper bound)**, stressfac -2.5, drop_stress 0.85 (bound) | 1.98 -> 1.48 | wet season 3.7 -> 2.6; annual 3.0 -> 1.6; dry season 40-50 gC/mo against 1-13 | 1.1 | the semi-arid excess of the 62-tower baseline: no dry-season shutdown; structural |

**Step 2 verdict.** Seven of the ten PFTs are at or near the towers with parameters inside their measured ranges: both
conifers, tropical and boreal deciduous timing, tropical evergreen, shrub tundra, temperate deciduous at 0.85-0.90 of
the summer (Harvard, CA-Oas). Three are not parameter problems: the Mediterranean shrub and holm oak (IT-Noe, FR-Pue),
the tropical C4 grassland (AU-Stp) and the miombo (ZM-Mon) all fail on the dry season -- the model keeps
photosynthesising through drought (IT-Noe July-August, AU-Stp May-November, ZM-Mon June-October) and the optimiser can only
lower capacity and conductance to their bounds. The soil-water stress and drought phenology are the next structural item
before a joint fit (step 3), which would otherwise inherit these bounds. Secondary: ET at 0.4-0.65 of LE_CORR at the
grassland (DE-Gri) and the boreal pine (FI-Hyy); leaf-out one month late in the tundra's held-out spring.

**Why IT-Noe and AU-Stp are bad (Hangkai's question, 2026-09-28; diagnostic years `site_fit/diag_{IT-Noe_fitted_2013,
AU-Stp_fitted_2012}.npz`, fitted parameters).** Not the drought response: at both towers the model's ET matches the
tower month by month (IT-Noe 409 vs 468 mm yr-1, AU-Stp 598 vs 588) and the soil-water stress does act (IT-Noe stress
0.2-0.3 in July-September with GPP 71-112 against the tower's 48-88; AU-Stp stress 0.1-0.4 and displayed LAI 0.12 from
May to December). The excess is in the WET months, where the leaves are unstressed and the canopy makes about twice the
carbon per unit water the tower does:
- IT-Noe: leaf area held all year at the site-attribute maximum 3.08 (MODIS at the cell 1.3), stress 1.0 through May,
  shrub vmax(T) 20-31 umol, top-leaf gross 8-10 umol -> 237-274 gC/mo in April-May against the tower's 137-170, and
  111-183 in the mild winter against 56-95. The model treats the maquis as a dense, productive shrub canopy; the fit
  ran the evergreen-shrub vmax to the bottom of the TRY range (38.9) and the stomatal slope to its bound and still has
  1.75x the annual. Candidates: the leaf area (3.0 vs 1.3) and the sclerophyll leaves' capacity, which TRY's shrub
  class (57.9 +- 19.6) does not represent; to be tested by holding LAI at the MODIS value.
- AU-Stp: the phenology does drop the canopy (LAI 1.15 in March -> 0.12 dry season), but a month too early -- the model
  peaks with the rain (January-March) while the tower's grass peaks in March-April, after it -- and the wet-season leaves
  are too productive: LAI 1.15 with C4 vmax(T) 34 umol gives 305 gC/mo and 168 mm ET in March against the tower's 113
  gC and 107 mm. The fit put C4 vmax at the assumed floor (30 at 25 C; no TRY value) and the stomatal slope at its
  ceiling; the tower implies a leaf capacity of 15-20. The dry-season floor LAI 0.12 keeps 9-15 gC/mo alive against 1-7.

## 31. Leaf area held at the MODIS value at non-forest towers (2026-09-28)

**Order (Hangkai).** "Hold the leaf area at the MODIS site value for the non-forest classes (shrublands, savannas,
grasslands) instead of the attribute maximum, and rerun that baseline." `site_run --fixed-lai attrs-modis`: OSH, CSH,
SAV, WSA and GRA towers at the MODIS annual maximum of the cell (`fit_carbon.modis_at`), forests at the site-attribute
maximum as before. `site_runs/latfix_sunshade_2008-2012_attrsmodis` against `latfix_sunshade_2008-2012_fixedlai`
(section 28), class medians, model / tower:

| class (n) | leaf area held | peak | June-August | annual | ET |
|---|---|---|---|---|---|
| OSH (4) | attribute 2.1-2.5 -> MODIS 0.1-0.8 | 4.15 -> 1.10 | 6.1 -> 1.8 | 5.0 -> 1.85 | 1.42 -> 1.32 |
| CSH (1, IT-Noe) | 3.0 -> 1.3 | 1.24 -> 0.78 | 1.08 -> 0.79 | 1.41 -> 0.94 | 0.87 -> 0.84 |
| SAV (2) | -> 1.3-1.6 | 1.32 -> 1.13 | 1.40 -> 1.21 | 1.07 -> 0.96 | 0.95 -> 0.91 |
| WSA (3) | -> 0.8-1.8 | 0.98 -> 0.82 | 1.01 -> 0.95 | 1.34 -> 1.26 | 0.96 -> 0.92 |
| GRA (13) | -> 0.6-3.1 | 0.96 -> 0.90 | 1.06 -> 1.03 | 1.10 -> 1.24 | 0.85 -> 0.86 |
| forests | unchanged | | | | |
| all 61 | | median GPP ratio 1.10 -> 1.07; within 0.8-1.25: 52 -> 51 %; r 0.93 -> 0.94 | | | |

The dry-site excess of section 28 was the canopy the harness gave those towers: the open shrublands go from 5x to 1.85x
the annual and the maquis to 0.94. What is left at OSH (1.8x, four semi-arid US and Spanish towers with MODIS LAI
0.1-0.8) and at the grasslands' shoulders (annual 1.24 with the summer at 1.03) is the next thing to read at those
towers. The MODIS cell maximum understates a few sites (ES-Amo 0.1, GL-ZaH 0.1); a site-level LAI would be better where
one exists. This configuration is the site baseline for step 3.

**Refits at the MODIS leaf area (`pft_{IT-Noe,AU-Stp}_modislai`, held-out / in-sample as before):**
- IT-Noe (LAI 1.3): loss 15.1 -> 8.8; vmax 38.3 (bottom of the TRY range), slope 3.0 (bound), stressfac -2.9. Held out
  2013-14: peak 0.89, June-August 1.12, annual 1.03, ET 0.88. Solved to the tower's uncertainty; the two bounds say the
  sclerophyll leaves are still less productive than TRY's shrub class.
- AU-Stp (LAI 0.8, C4 range opened to 10-75): loss 123 -> 20.5; vmax 10.0 at 25 C (the floor; IBIS default 23, CLM 52),
  slope 12.2, stressfac -3.4, drop_stress 0.50. In-sample 2008-14: peak 0.82, wet season 1.20, annual 0.98, ET 1.04.
  The timing is still a month early (model 76-86 gC in January-March, tower peaks in March-April at 108-113; April 39
  vs 108), and the very low vmax is compensating it: with leaves in January when the tower has none, only a weak leaf
  keeps the wet-season total right. The grass green-up lag (leaf growth after the rain, not with it) is the structural
  item at the tropical grasslands; the C4 capacity should be re-fitted once it is in.

## 32. Step 3: the joint fit across the typical towers (2026-09-28/29)

**Setup (Hangkai: "do that then" -- three starts, 300 evaluations, a profile).** Nine typical towers that share 2008-2012
(DE-Tha, FI-Hyy, FR-Pue, GF-Guy, IT-Noe, RU-Cok, AU-Stp, DE-Gri, US-Ha1; CA-Oas and ZM-Mon end before 2008), 2012 held
out, leaf area and composition as in step 2, hourly tower light, ET term at the six towers with LE_CORR
(`--et-sites`), 22 parameters: vmax of the nine PFTs present within their TRY ranges (`--vmax-bounds pft=lo,hi;...`),
five stomatal slopes, the two acclimation constants, Harvard's three phenology dates, stressfac, the lower canopy's two
thresholds. CMA-ES population 12, 300 evaluations, 4 workers, seeds 1-3 (`$V/site_fit/joint_typical_seed{1,2,3}`,
131 s per evaluation for the nine towers in one batch); `--profile 5` around the best start (`seed3/profile5`).

**Convergence and identifiability.** Losses 13.8 / 13.2 / 12.4 from the baseline 40.8 (held-out 2012: 16.2 / 21.1 /
13.0); best evaluations 291 / 192 / 246 with the last generation still 40 % above the best: not converged at 300.
The three starts agree (within 10 %) on the warm-temperate evergreen vmax (27-28, the lower bound), the conifer vmax
(54-65), C4 (10-11, the floor) and C3 grass (48-49, the floor), the conifer and C3 slopes (3-5), the leaf-out ramp and
stressfac; they disagree by a factor of 2 or more on the temperate deciduous, shrub vmax, the shrub slope, both
acclimation constants, Harvard's leaf-out and senescence and the lower canopy's thresholds. The profile says the same:
identified in the interior are the conifer slope, the acclimation pair, Harvard's leaf-out and senescence and the
grass cold threshold; on a bound are the tropical evergreen (high), warm evergreen (low), both conifers (HIGH, 87 --
against acclimation, see below), C4 and C3 (low), the broadleaf and C4 slopes (high), the C3 slope (low) and stressfac
(-2.5, least stress); flat across their whole ranges are the temperate deciduous vmax, both shrub vmax, the shrub
slope, the leaf-out ramp and the lower canopy's leaf-out threshold.

**Per tower (seed 3, 2009-2012, model / tower) against the per-site fits of section 30:**

| tower | joint: peak / season / annual / ET | per-site: season / annual |
|---|---|---|
| DE-Tha | 0.77 / 0.78 / 0.79 / 0.90 | 0.95 / 0.97 |
| FI-Hyy | 0.83 / 0.91 / 0.84 / 0.74 | 1.02 / 1.04 |
| GF-Guy | 0.85 / 0.84 / 0.79 / 0.92 | 1.00 / 1.03 |
| FR-Pue | 1.26 / 1.26 / 1.38 / 1.26 | 1.03 / 0.98 |
| RU-Cok | 0.80 / 0.66 / 0.62 / 0.61 | 0.86 / 0.86 |
| US-Ha1 | 0.97 / 1.23 / 0.98 / 1.28 | 1.02 / 0.83 |
| DE-Gri | 0.94 / 1.02 / 1.03 / 0.62 | 1.13 / 1.03 |
| IT-Noe | 0.78 / 1.05 / 0.93 / 0.86 | 1.12 / 1.03 |
| AU-Stp | 0.78 / 1.35 / 1.18 / 1.08 | 1.20 / 0.98 |

**Reading.** With one tower per PFT, the PFT-specific parameters of a joint fit can only reproduce the per-site fits;
what the joint fit adds is the parameters shared across PFTs, and those are where it loses: the broadleaf stomatal slope
is shared by the holm oak, which wants 3, and Harvard, which wants 11-16 (FR-Pue 0.98 -> 1.38); the two acclimation
constants are shared by the boreal and the temperate conifer, which want opposite things (DE-Tha 0.97 -> 0.79, FI-Hyy
1.04 -> 0.84, with the conifer vmax pushed to the top of TRY to compensate the stronger acclimation); the shrub slope
and tundra thresholds are flat. The step-3 result is therefore not a parameter set but a list of what may not be
shared: stomatal slope per PFT rather than per leaf-form group, acclimation per conifer PFT. Stressfac and the grass
cold threshold are the shared parameters that are identified (-2.5 to -4; about 4-5 C).

**Consequence for the plan.** The PFT parameter set to carry forward is the per-site set of section 30-31 for the
PFT-specific parameters, with the shared parameters at the joint values that were identified. A joint fit becomes
meaningful at Supplementary 1, where each PFT has several towers (17 conifers, 12 deciduous, 13 grasslands in the
2008-2012 batch alone) and the PFT parameters are constrained by more than one site; it should carry per-PFT slopes and
acclimation, three starts and the profile as here.

## 33. Supplementary 1: the joint fit over the 62-tower batch (2026-09-29/30)

**Setup (Hangkai: "do that as what we planned").** `default_seasonal_sunshade_pft.yaml` (new: stomatal slope and
acclimation constants per PFT, `coefm_mix` / `acclim_per_pft`, bit-identical at the group values; full suite 722), the
62 towers of 2008-2012, non-forest towers at the MODIS leaf area (`--lai-mode attrs-modis`), ET term only where LE_CORR
exists (`--et-corr-only`), composition overrides for the eucalypt savannas (AU-How/DaS/Dry: tropical evergreen + C4)
and RU-Cok; 29 parameters (vmax and slope of the nine PFTs present, acclimation of the two conifers, Harvard's
phenology, stressfac, the lower canopy's thresholds); seeds 1-3, 400 evaluations, population 14, 4 workers; 114 s per
evaluation for the 62 towers in one batch. Runs `$V/site_fit/s1_batch62_seed{1,2,3}`, profile `seed1/profile5`.

**Convergence.** All three seeds reach the same loss, 19.8 / 19.9 / 20.1 from 47.1 (2012 held out: 22.1 / 22.0 /
20.6), and agree within 10 % on the C4 (10, the floor) and C3 (48, the floor) vmax, the boreal conifer vmax (39-46),
the conifer slope (3.3-3.6), stressfac (-2.5, the bound = least stress) and the grass cold threshold (6-8 C); they
disagree by 30 % or more on the tropical, temperate-conifer and shrub vmax and most slopes, the acclimation and the
phenology constants. The profile has the tropical, temperate conifer and both grass vmax preferring their LOWER bounds,
both acclimation strengths at their maximum, and the shrub parameters flat.

**Verdict, 2009-2012, class medians of model / tower (defaults | joint):** ENF annual 1.00 | 0.83, summer 1.01 | 0.86;
DBF 1.09 | 0.82, 0.83 | 0.84, April 3.4x | 1.0; GRA 1.21 | 0.81, 1.11 | 1.06; EBF 0.96 | 0.72; MF 1.18 | 0.93; OSH 1.69 |
0.65; SAV 0.89 | 0.59; WSA 1.23 | 0.67. Monthly r 0.90 -> 0.93 and NSE 0.64 -> 0.74 (towers above 0.5: 69 -> 72 %),
but towers within 25 % of the annual: 59 -> 52 %. The fit improved the timing everywhere and lowered the magnitude at
every class but the deciduous and grassland summers.

**Why (`logs/s1_batch62_seed1_vs_defaults.csv`).** The objective is the mean over towers of a squared, sigma-normalised
monthly misfit. Twelve towers carried 61 % of the default loss (US-PFa 235, US-WCr 197, AU-Stp 165, US-MMS 154, US-Oho
144, IT-MBo 115, FR-Pue 110, ...): the deciduous April (a month early, 3.4x), the grassland shoulders and one mixed
forest. The fit serves them: DBF mean loss 79 -> 18, GRA 54 -> 24, MF 61 -> 30. The conifers, already at the towers,
got WORSE at 12 of 17 (12.2 -> 12.9) because the temperate-conifer PFT is also the conifer half of the mixed forests,
whose largest tower (US-PFa) pulls its vmax down and its acclimation up. Squared loss plus a mean over towers lets a
few towers set a PFT's parameters against the majority of its own class.

**Reading.** The S1 set is better on shape and worse on magnitude; as PFT defaults it carries the outlier towers'
demands. The remedy is in the objective, not the model: a per-tower contribution bounded so that no tower can dominate
(a Huber or capped misfit, or a median across towers), and the magnitude judged as the verdict does (annual and
summer ratios). Then the per-site step (Supplementary 2) is where each tower's own deviation belongs.

## 34. Supplementary 1 refitted with the bounded objective (2026-09-30; branch `calibration-s1`)

**Order (Hangkai).** "Refit with a bounded per-tower contribution and the magnitude ratios in the objective, and save the
results as a new branch." `site_fit --robust 25,30,1,0.15`: each month's normalised squared misfit capped at 25, the
tower's annual and summer GPP ratios added as (ln R / 0.15)^2 (capped), each tower's total capped at 30 (section 33's
US-PFa contributed 235 before). Same 62 towers, parameters and settings as section 33; seeds 1-3, stopped by Hangkai's
NAS schedule after 225-239 of 400 evaluations (`$V/site_fit/s1_robust_seed{1,2,3}`); best seed 2, loss 14.39 from 16.73
(seeds 14.6 / 14.4 / 14.7). Parameter file `configs/params/s1_robust_2026-09-30.yaml` (seed 2). No profile was run.

**Class medians, 2009-2012, model / tower: defaults | plain S1 (section 33) | robust S1**

| class (n) | summer | annual | ET | NSE |
|---|---|---|---|---|
| ENF (17) | 1.01 / 0.86 / **0.98** | 1.00 / 0.83 / **0.99** | 0.86 / 0.77 / 0.78 | 0.91 / 0.85 / 0.87 |
| DBF (12) | 0.83 / 0.84 / **0.90** | 1.09 / 0.82 / **0.93** | 1.09 / 0.83 / 1.05 | 0.69 / 0.88 / 0.83 |
| GRA (13) | 1.11 / 1.06 / 1.19 | 1.21 / 0.81 / **0.97** | 0.86 / 0.88 / 0.87 | 0.61 / 0.59 / 0.69 |
| MF (5) | 1.06 / 0.94 / 1.06 | 1.18 / 0.93 / 1.18 | 1.10 / 0.90 / 0.97 | 0.84 / 0.80 / 0.84 |
| EBF (3) | 1.03 / 0.76 / 0.91 | 0.96 / 0.72 / 0.91 | 0.96 / 0.96 / 0.92 | poor in all three |
| OSH (5) | 1.82 / 0.80 / 1.15 | 1.69 / 0.65 / **0.97** | 0.95 / 0.93 / 0.94 | ~0 in all three |
| SAV (2), WSA (3) | 1.59 / 1.12 / 1.53; 0.98 / 0.62 / 0.75 | 0.89 / 0.59 / 0.83; 1.23 / 0.67 / 0.87 | | |
| all 61 | towers within 25 % of the annual: 59 / 52 / **72 %**; median NSE 0.64 / 0.74 / **0.77**; r 0.90 / 0.93 / 0.93 | | | |

**Parameters (three seeds, vmax at 25 C).** The vmax agree across seeds within 17 % for every PFT but the boreal conifer
(40-53): tropical evergreen 58, warm-temperate evergreen 55-59, temperate conifer 43-51, temperate deciduous 46-52,
evergreen shrub 44, deciduous shrub 50-56, C4 13-16, C3 48-50 -- all inside their TRY ranges, none at a bound except C3
near its floor. The slopes, the acclimation strengths, Harvard's phenology and the lower canopy's thresholds still
differ between seeds by 50-200 %: not identified by these towers (the profile of section 33 said the same).

**Reading.** With no tower able to dominate, the fit keeps the conifers at the towers (0.98-0.99), takes the deciduous
summer from 0.83 to 0.90 and the annual from 1.09 to 0.93, the grasslands from 1.21 to 0.97, the open shrublands from
1.69 to 0.97, and 72 % of the towers land within 25 % of their annual GPP. This is the PFT parameter set to carry to
Supplementary 2 and the trend test; the unidentified parameters stay at their seed-2 values but should be read as
defaults, not fitted values.

**Stage 2 (evening of 2026-09-29; `$V/site_fit/s1_robust_r{1,2,3}`, `s1_robust_profile`).** The NAS stayed up, so the
three seeds were restarted from the seed-2 best (sigma 0.1, 200 evaluations each) and the one-at-a-time profile was
run around it (7 points across each parameter's bounds, `--profile 7`). Losses 14.39 -> 14.13 / 14.19 / 14.16
(seeds 1-3): 1.8 % more, and the parameter file `s1_robust_2026-09-30.yaml` now carries the seed-1 restart
(`logs/s1_robust_r1_best.json`). The move between the two bests is the identifiability finding, not the loss: for that
1.8 % the tropical evergreen slope went 10.3 -> 5.4, the evergreen shrub slope 15.5 -> 8.8, the upper-canopy GDD ramp
150 -> 82, the drought-drop threshold 0.28 -> 0.10, the boreal acclimation ceiling 24.5 -> 30 (its bound), while the
vmax values moved by 5-10 % (all inside their TRY ranges, tropical evergreen at its top 60.6). The profile
(`logs/s1_robust_profile.csv`) says the same: the widest loss range along any single parameter's whole bound interval is
2.8 (temperate-conifer acclimation s0), then C4 vmax 2.0, the drought-drop threshold 1.9, the upper GDD threshold 1.7;
21 of the 29 parameters move the loss by less than 1.0 across their full range, and gdd_thresh_l, gdd_ramp_u,
vmax:dec_shrub and the shrub slopes by less than 0.15. At a bound at the profile's minimum: C3 vmax (low), C4 slope
(low), boreal conifer slope (low), boreal acclimation ceiling (high), tropical and warm-temperate evergreen vmax (high),
stressfac (high, -2.5). Reading: with the capped objective the surface is flat, the 62 towers identify the conifer
acclimation, the C4 capacity, the drought drop and the deciduous leaf-out threshold and little else; everything else
in this file is a defensible default, and the per-tower stage (section 35) is where the rest is decided tower by tower.

**Parameter-file defect found on the way.** The per-PFT tables in the yaml (`coefm_pft`, `acclim_s0_pft`,
`acclim_smax_pft`) were written with the harness's short PFT names and read by the long names, so only the two grass
slopes of the nine were applied when the file was read back; fixed in `physics_params.py` (both spellings accepted,
unknown names raise; test `test_per_pft_tables_accept_short_and_long_names`). No run had used the file before the fix.
 Open as before: ET at the conifers (0.78) and grasslands (0.87), the deciduous and grass
spring shape, the evergreen broadleaf seasonality (r 0.47).

## 35. Supplementary 2: one fit per tower from the robust S1 set (2026-09-30 to 10-02; branch `calibration-s1`)

**Setup.** Each of the 62 towers of the 2008-2012 batch fitted on its own (`verify/calibration/s2_commands.py` writes the
command per tower): start = the robust S1 best (section 34, `s1_robust_r1/best.json`), fitted = vmax and stomatal slope
of each prescribed PFT (TRY bounds), acclimation s0/smax of conifer PFTs, upper-canopy phenology (gdd_thresh_u,
gdd_ramp_u, sen_ycrit_u) where a deciduous tree is present, lower-canopy thresholds (gdd_thresh_l, tthresh_l_c) where a
lower canopy exists, drop_stress_thresh at tropical deciduous / C4 / shrub, stressfac everywhere (4-10 parameters per
tower); 2009-2011 fitted, 2012 held out; CMA-ES 60-80 evaluations, popsize 6, 3 workers, sigma 0.15; the robust
objective WITHOUT the per-tower cap (`--robust 25,1e9,1,0.15`). Runs `$V/site_fit/s2/<SITE>`; report
`diagnostics/s2_report.py` -> `ibis_local_document/logs/s2_per_tower_report.{csv,txt}`; figures
`figures/site_level/{GPP,ET,NEE}_2_example_towers.png` (purple) and `extra/F5_per_tower_fits_so_far.png`.

**Two operational defects on the way (details in the memory notes).** (1) The per-tower cap of the robust objective
makes the surface flat for a tower whose loss sits at the cap: AU-Stp returned 30.000 at every point, CMA-ES stopped
after one generation on "flat fitness" and reported the start as the fit -- the cap is for joint fits only; seven towers
of the S1 result sat at it. (2) Running two launchers at once put 16 fits x 16 GB against 188 GB and thrashed the box;
one throttled launcher (max 8 alive, 4-core taskset slots) at ~1.3 h per tower was the working configuration. The NAS
outage of 09-30 13:20-16:00 and the SHR expansion that followed (file opens 5-18 s for a day) interrupted the last
towers; US-Syv has no tower data in 2008-2011 (FLUXNET2015 gap) and was refitted on 2002-2008 (`s2_ownyears/US-Syv`: loss 11.9 -> 3.25 on 2003-2007; 2008 held out carries no data, so it has no held-out verdict and is not in the 61-tower table);
DE-Obe lost its summary line to the outage but its best.json/best_series are intact.

**Verdict (61 towers, model / tower, held-out year 2012; S1 set -> per-tower fit):**

| | S1 set | per-tower fit |
|---|---|---|
| towers within 25 % of annual GPP, fit years 2009-2011 | 74 % | 97 % |
| towers within 25 %, held-out 2012 | 64 % | 80 % |
| median r / NSE of monthly GPP, held-out 2012 | 0.95 / 0.74 | 0.96 / 0.88 |
| ENF (17) annual / NSE | 1.03 / 0.87 | 1.04 / 0.94 |
| DBF (12) annual / NSE | 0.95 / 0.78 | 1.01 / 0.90 |
| MF (5) annual / NSE | 1.11 / 0.92 | 1.04 / 0.91 |
| GRA (13) annual / NSE | 1.12 / 0.78 | 1.07 / 0.84 |
| EBF (3) annual / NSE | 0.94 / -6.5 | 0.99 / 0.47 |
| OSH (5) summer / NSE | 2.23 / -0.54 | 1.97 / -0.15 |
| SAV+WSA (5) annual / NSE | 0.81-0.90 / 0.45-0.54 | 0.92-0.95 / 0.65-0.73 |
| ET, ENF / GRA / OSH | 0.84 / 0.83 / 1.18 | 0.86 / 0.81 / 1.26 |

**Reading.** Per-tower parameters transfer to an unseen year at the forests and grasslands (80 % of towers within 25 %,
NSE 0.88), and they rescue the evergreen broadleaf seasonality that the per-PFT set could not (FR-Pue NSE -6.5 -> 0.62,
GF-Guy stays negative: the flat tropical year has no seasonal variance to explain). They do NOT fix: the open shrublands
(summer 2.0x with any parameters; ES-LJu 2.8x), US-PFa (1.7x, tall-tower footprint), US-Var (2.0x), AU-Stp (the fit
undershoots 0.50 in 2012 after overshooting in the fit years: the green-up timing, not capacity), and ET anywhere
(conifers 0.86, grasslands 0.81, shrublands 1.26 -- parameters that fix GPP leave ET where it was). Those are the
structural list for the ET and NEE stages (sections 36-). The per-tower parameter records (`s2/<SITE>/best.json`) are
the input to the gap-filled spatial parameter map, not yet built.

## 36. NEE stage: day-time and night-time NEE at the towers (2026-10-02, in progress; branch `calibration-s1`)

**Order (Hangkai, 09-30).** Calibrate on the tower NEE with the day and night parts separated, so that both are
captured; use Min's finer soil data to limit the scale effect. (His "FLUXCOM climate" item is still unanswered; the
towers keep their own half-hourly meteorology.)

**Targets.** `verify/calibration/tower_nee_daynight.py`: from the FLUXNET2015 half-hourly files, NEE_VUT_REF with
QC <= 1 split by FLUXNET's NIGHT flag (SW_IN_POT = 0), monthly means and sums (gC m-2 month-1, + = release), the
measured share per month; `ibis_local_document/data/tower_nee_daynight_2008-2012.npz`, 62 towers, 40 with all 48
months. Only 20-35 % of the night half-hours are measured (u* filtering); night NEE therefore carries the gap-filler's
temperature response, and the day-time term is the independent check.

**Soil data.** HWSD2 at 1 km (`/shared/nas2/Min/HWSD`, `tower_soil_hwsd2.py`, `data/tower_hwsd2_*.csv`): every tower
on a mapping unit, 44 of 62 in mixed 11-km neighbourhoods. The texture scale effect is small: the 1-km unit differs
from the 0.5-degree cell by > 10 points of sand or clay at 7 towers only (AU-How, AU-Stp, DE-Hai, DE-Lnf, ES-LJu,
FI-Hyy, RU-Cok), median 2 points, because the 0.5-degree file is aggregated from the same units. What the 1-km data
adds is a site soil-carbon prior (organic carbon and bulk density per layer: mineral soils 5-27 kgC m-2 to 1 m, class
medians 10-17); HWSD2 codes missing deep layers as -7 (masked) and 10 towers share their unit with a peat component
that is not the tower's (flagged, dominant-component values alongside).

**Diagnosis (S1 set, 2009-2012; `data/tower_model_carbon_balance_s1.csv`, `tower_model_soil_pools_2010.csv`).**
Model annual NEE at the forests is a small sink (DBF -43, ENF -90 gC m-2 yr-1) where the towers are large ones (-416,
-290): the grid pools are near equilibrium (Rh/NPP 0.89-0.93) while the towers are aggrading. To give the tower NEE at
the model NPP, model Rh must be x0.55 (DBF) and x0.75 (ENF): a per-site disequilibrium factor (Carvalhais et al. 2008).
Evergreen broadleaf is the opposite (model -674 vs -222; Rh/NPP 0.50), open shrubland has Rh/NPP 2.3 (grid pools far
above what the site NPP sustains). The grid-cell soil carbon at the tower cells is 1.3x the HWSD2 site carbon at the
median and > 2x at 17 temperate-forest towers. First day/night split at DE-Tha 2010 (S1 set): night-time release
+592 vs tower +528, day-time uptake -501 vs -1154 with GPP within 3 %: the night respiration is close, the excess is
day-time respiration -- growth respiration charged in step with assimilation (0.30 of GPP - Rm) and a heterotrophic
rate with no diurnal cycle.

**Machinery (commits 7cbf0ee, 80cd5af, 5792efa).** (1) Optional state module `NEEParts` (chain 258/261/265/268/270):
NEE per hourly step = Rh (daily rate, spread evenly) - NPP_step, accumulated for coszen <= 0 and the rest; the site
harness carries it as monthly `nee_daynight`; night + day = monthly NEE within 2 gC m-2 month-1 (the one-day lag of
the daily Rh). (2) `site_pool_spinup.py`: the tower's pools at the cyclic steady state under its own forcing and
per-tower parameters, by accelerated decomposition (slow-pool rates x20 / x200 / x2000, pools scaled back; exact for
the linear cascade). (3) Respiration constants that F77 hard-codes are now `BGCParams` fields with F77 defaults
(bit-identical): `lt_e0` (Lloyd-Taylor E0, 344 K), `wfps_opt` (60 %), `rgrowth` (0.30); extras `bgc_e0`, `wfps_opt`,
`rgrowth`, and `som_scale` (a state scaling of the three SOM pools at the start of every evaluation, the
disequilibrium factor). (4) `site_fit --target nee --nee-targets <npz> [--start-state <spin-up state>]`: objective =
night-time + w_et x day-time normalised monthly misfit (sigma = max(15 %, 5 gC), robust caps and annual-ratio terms as
in section 34). Smoke test at DE-Tha: baseline loss 24.2 with the per-tower GPP parameters. Full suite 752 passed.


**Day/night baseline, 62 towers, S1 set (`site_runs/daynight_s1robust_2008-2012`, `diagnostics/nee_daynight_report.py`,
`figures/site_level/NEE_night_day_by_class_s1robust.png`, `logs/nee_daynight_s1robust.csv`; class medians of annual
sums, model / tower, r of the monthly series):**

| class (n) | night-time release: tower / model / ratio / r | day-time uptake: tower / model / ratio / r |
|---|---|---|
| ENF (17) | 341 / 398 / 1.00 / 0.79 | -806 / -614 / 0.73 / 0.87 |
| DBF (12) | 391 / 375 / 1.05 / 0.78 | -871 / -422 / 0.60 / 0.87 |
| MF (6) | 424 / 474 / 0.83 / 0.79 | -712 / -551 / 1.17 / 0.94 |
| GRA (13) | 411 / 416 / 0.85 / 0.81 | -391 / -383 / 0.70 / 0.80 |
| EBF (3) | 823 / 407 / 0.50 / 0.29 | -1357 / -1076 / 1.09 / -0.02 |
| OSH (5) | 50 / 103 / 1.74 / 0.60 | -66 / +2 / 0.01 / 0.71 |
| SAV (2) | 454 / 404 / 0.95 / 0.47 | -661 / -322 / 0.49 / 0.58 |
| WSA (3) | 335 / 337 / 1.01 / -0.31 | -462 / -379 / 0.48 / 0.67 |

**Reading.** At the temperate and boreal forests and the grasslands the model's night-time NEE -- the respiration the
tower measures directly -- is at the tower in size and in seasonal shape (ratios 0.85-1.05, r ~0.8). The missing sink
is entirely in the day-time: summer day-time uptake 0.60-0.73 of the tower's with GPP within 10 %. So the model's
respiration is too high during the day and right at night, which is not what the pools do (they respire the same per
hour day and night in the model). The day-time excess is autotrophic: growth respiration charged in step with
assimilation (0.30 of GPP - Rm, all during the day) and maintenance respiration at day-time temperatures. The per-tower
fits therefore need rgrowth (and the maintenance constants) as much as som_scale; som_scale alone would break the
night match. Evergreen broadleaf is the one class where night respiration is too LOW (0.50: pools too small, and the
seasonality absent); open shrubland has no net day-time uptake at all (the model cannot photosynthesise against its
respiration there); the savannas' day-time uptake is half the tower's.


**The litterfall artefact (10-02 evening).** The first joint spin-up (`pool_spinup_s1_62`, grid litterfall) reached
"equilibrium" with NEE -960 gC m-2 yr-1 at BE-Bra and -900 at the evergreen broadleaf towers. The reason is the
harness, not the soil: with fixed vegetation (isimveg = 0) `dynaveg` never runs, so `carbon.falll/fallr/fallw` keep the
GRID restart's values in every site run made so far. The soil therefore equilibrates to the grid cell's litterfall
while NPP is the tower's, and model NEE = NPP_site - litterfall_grid at equilibrium: the "model sink too small / pools
near equilibrium" reading above and the EBF "sink" of -674 are that difference, not pool disequilibrium. The spin-up
now closes the carbon balance (falll, fallr, fallw = cycle-mean per-PFT NPP x PFT allocation fractions, re-set after
every cycle; the NEE fits inherit the closed litterfall with the state), and is being rerun (`pool_spinup_v2_*`). Correction the same evening: with isimveg = 0 `one_year` calls `biomass_update` every year, which allocates NPP into the
leaf, wood and root pools (relaxation toward a tau NPP over tauleaf / tauwood0 / tauroot) and RECOMPUTES litterfall from
the pools, so the grid litterfall is not frozen but relaxes over the wood turnover time, and the -500 to -900 gC
"equilibrium NEE" of the first closed spin-up (BE-Bra, FR-Fon, NL-Loo, AU-Tum, DK-Sor) is wood accumulation toward
awood tauwood NPP; setting the litterfall rates directly is overwritten. The closure that holds sets the wood and root
pools of the prescribed PFTs to their steady state for the cycle-mean NPP after every cycle (cbiol is held), so the
litterfall biomass_update computes equals the allocated NPP and the vegetation is stationary in the NEE fits
(`pool_spinup_v3_*`). That was not enough either: `diagnostics/pool_budget_cycle.py` showed the soil budget closing
exactly (Rh is its residual) while the INPUTS collapsed within a cycle -- `one_year` recomputes `exist` from the
climate before the biomass update, so where the climate rules out the prescribed PFT (NL-Loo's temperate conifer, the
evergreen broadleaf towers, many high-NPP forests) its NPP is dropped from the allocation, the root pool decays within
a year and litterfall falls with it (NL-Loo inputs 1005 -> 655 gC m-2 yr-1 while NPP rose to 1235). The hold
(`enforce_prescription`) therefore closes the stand's carbon every year: wood and root pools at their steady state for
the year's NPP of the prescribed PFTs, litterfall = allocated NPP (one-year lag; cycle means close to zero NEE at
equilibrium; `pool_spinup_v7_*`). The day/night reading stands (both parts carry the same Rh); the disequilibrium factors and the
rate multipliers are to be redone on the vegetation-closed spin-up.

**Closed spin-up (`pool_spinup_v7_62`, 10-03 03:47; `data/tower_pool_equilibrium_v7.csv`).** With the stand closed every
year the 62 towers are at equilibrium: class-median NEE -6 (DBF), -4 (ENF), -3 (MF), -18 (GRA), -28 (EBF) gC m-2 yr-1,
median |NEE| 19, 41 of 62 within +-30; the exceptions are grasslands with large year-to-year NPP swings (the one-year
litterfall lag: AU-DaP -69, AU-Stp +70, US-Var +65), DE-Lnf -114 and US-Oho +209 (still converging, drift -1 kgC per
cycle). Equilibrium soil carbon against the HWSD2 site carbon gives the per-tower decomposition-rate multiplier: forests
2.0-2.6 (DBF 42 vs 13 kgC m-2, ENF 39 vs 17, MF 48 vs 21), grasslands 1.8, evergreen broadleaf 2.2, and below 1 at the
shrublands and savannas (OSH 0.25 at the bound: the model's NPP there, 77 gC m-2 yr-1, cannot sustain the site's 7 kgC
of soil carbon at any rate -- a productivity problem, not a soil one). Nine towers sit at a multiplier bound. The
eighth spin-up runs with these multipliers and the 61 per-tower NEE fits start from its state. With them (`pool_spinup_v8_62`, 05:04) the equilibrium soil carbon is at the site's (median ratio 1.01, class medians 0.88-1.01; seven towers off by more than 25 %, six of them at a multiplier bound) and the equilibrium NEE has median |14| gC m-2 yr-1. The 61 per-tower NEE fits (som_scale, bgc_e0, rgrowth, rroot; 48 evaluations; 2012 held out) launched from that state at 05:04 (`$V/site_fit/nee/<SITE>`).
 First NEE fit at DE-Tha on the unclosed state (loss 24.4 ->
1.9; held-out 2012 night 1.10, day 0.89, NEE -509 vs -696; r 0.89 / 0.97) pressed som_scale (0.11) and rgrowth (0.10)
against their bounds: the maintenance constants rwood/rroot are parameters now (extras), and the per-tower set is
som_scale, bgc_e0, rgrowth, rroot with k_scale fixed per tower by the HWSD2 carbon.

**Per-tower NEE fits (10-03, all 61 by 21:44; `$V/site_fit/nee/<SITE>`, `diagnostics/nee_fit_report.py` ->
`logs/nee_fit_report.csv`). Start = the per-tower GPP/ET parameters (section 35) and the equilibrated pools of the
eighth spin-up (per-tower rate multiplier from the HWSD2 carbon); fitted som_scale, bgc_e0, rgrowth, rroot (4
parameters) to night-time + day-time monthly NEE on 2009-2011, 48 evaluations; 2012 held out. Memory on the shared
box (the methane ANN queues of another session, 30-40 GB) forced the launcher to 6 fits at a time with a 40 GB guard and
5 minutes between launches (two incidents at 0 GB free before that).**

| held-out 2012, model / tower (61 towers) | S1 set | per-tower NEE fit |
|---|---|---|
| night-time release, median ratio / within 25 % | 1.01 / 47 % | 1.05 / 58 % |
| day-time uptake, median ratio / within 25 % | 0.67 / 22 % | 0.95 / 60 % |
| r of the monthly night / day series | 0.85 / 0.87 | 0.87 / 0.83 |
| ENF (17) night / day / r_day | 1.08 / 0.73 / 0.87 | 1.05 / 0.94 / 0.95 |
| DBF (11) night / day | 1.05 / 0.60 | 1.08 / 0.97 |
| MF (5) night / day | 0.83 / 1.17 | 1.03 / 0.92 |
| GRA (13) night / day | 0.85 / 0.70 | 0.91 / 1.03 |
| EBF (3) night / day | 0.50 / 1.09 | 1.00 / 1.24 |
| OSH (5) night / day | 1.74 / 0.01 | 1.64 / 0.66 |

**Fitted values (class medians):** som_scale ENF 0.38, DBF 0.61, MF 1.5, GRA 1.3, OSH 0.16 (bound), CSH 0.10 (bound);
rgrowth 0.17-0.38 (F77 0.30; 18 of 60 at a bound, mostly the floor 0.10); bgc_e0 200-350 K (F77 344; 9 at a bound);
rroot 0.3-1.0 per year (F77 1.25; 4 at a bound). 14 of 60 press the som_scale bound.

**Reading.** The day-time uptake, which the per-PFT set missed by a third, is brought to the tower on the unseen year
(0.67 -> 0.95; within 25 %: 22 -> 60 %) at forests and grasslands, while the night-time release stays at the tower
(1.05) and its shape improves slightly (r 0.85 -> 0.87); the day-time shape does not improve (r 0.87 -> 0.83). The
levers that did it are the ones the baseline predicted: growth respiration down from 0.30 toward 0.10-0.20 and the
soil pools scaled, with E0 and root maintenance secondary. That 18 towers want growth respiration at the 0.10 floor
says the model charges too much autotrophic respiration in daylight for a structural reason -- growth respiration
booked in step with assimilation, all by day -- rather than a mis-set constant; a growth-respiration term paid over the
whole day (and night) from the day's assimilate is the structural fix to test next. The open shrublands (night 1.64,
day 0.66), the one closed shrubland and the woody savannas remain the productivity problem of sections 33-35: no
respiration parameter can create day-time uptake the canopy does not produce.

**Plan.** Per tower: pools from the spin-up (HWSD2 carbon as the check), then fit som_scale, bgc_e0, rgrowth
(wfps_opt where the moisture signal exists) to night-time NEE with day-time NEE weighted, 2012 held out; verdict by
model/tower night release and day uptake, r of both series, by class.

### 36.1 Correction (10-04 to 10-05): the NEE fits reran on the per-tower GPP/ET base (`site_fit/nee_v2`)

`site_fit.py --start best.json` used to take ONLY the fitted names from the start file; every other parameter came from
`--params`. The 61 NEE fits above passed the section-35 best.json as `--start` with the default yaml as `--params`,
so they ran on the DEFAULT photosynthesis and phenology (their GPP equals the default run: pooled r 0.863 vs 0.862),
and the section-36 table compared respiration fitted on the wrong GPP. (The section-35 fits had the same defect
against the section-34 start: the S1 names outside their fit list -- conifer acclimation, lower-canopy phenology,
drought drop -- sat at the yaml values. Their series are self-consistent, so yaml + their best.json reproduces them.)

Fix: `setup` applies every entry of the start file to the base parameters (vmax and EXTRAS setters), then re-decodes
the fitted names as the CMA-ES start; verified on US-Ha1 (six entries, all equal). The 61 fits reran 10-04 20:22 to
10-05 07:02 (`$S/nee_v2_chain.sh`, local venv on CPU because the NAS-hosted environment imported at minutes per
package that night; identical jax 0.11.0 / equinox 0.13.8 / numpy 2.5.2). The rerun GPP sits on the section-35
series to a median 1.1 % (5 towers above 5 %, AT-Neu 20 %: the pool spin-up's soil state, not parameters).
Report `logs/nee_fit_report_v2.csv`, figures `figures/nee_fit_v2/`, three-run site figures
`figures/site_level_3runs_v2/` (`diagnostics/site_level_figures_3runs.py <out> nee_v2`).

| monthly, 62 towers, 2009-2012 (all months) | default | per-tower GPP/ET fit | per-tower NEE fit (wrong base) | per-tower NEE fit v2 |
|---|---|---|---|---|
| GPP r / NSE / RMSE / bias | 0.862 / 0.73 / 63 / +7 | 0.939 / 0.88 / 42 / +1 | 0.863 / 0.73 / 63 / +9 | 0.933 / 0.87 / 44 / -1 |
| ET r / NSE / RMSE / bias | 0.802 / 0.57 / 23 / +2 | 0.831 / 0.65 / 21 / +1 | 0.812 / 0.60 / 23 / +2 | 0.831 / 0.65 / 21 / +1 |
| NEE r / NSE / RMSE / bias | 0.469 / 0.12 / 61 / +11 | 0.580 / 0.24 / 57 / +18 | 0.638 / 0.38 / 52 / +6 | 0.725 / 0.49 / 47 / +8 |

NEE r by class, default -> GPP/ET fit -> NEE fit v2: ENF 0.50 -> 0.60 -> 0.78 (17), DBF 0.49 -> 0.72 -> 0.84 (12),
MF 0.39 -> 0.52 -> 0.89 (6), GRA 0.58 -> 0.55 -> 0.64 (13), OSH 0.44 -> 0.47 -> 0.72 (5), EBF 0.34 -> 0.33 -> 0.22 (3),
SAV 0.25 -> 0.30 -> 0.28 (2), WSA 0.35 -> 0.35 -> 0.45 (3).

Held-out 2012, model / tower, 61 towers (S1 set -> v2): night ratio median 1.02 -> 1.07, within 25 % 47 -> 70 %;
day ratio median 0.68 -> 0.89, within 25 % 23 -> 69 %; r night 0.85 -> 0.84, r day 0.88 -> 0.89.

Fitted values (median, bound-pressing of 61): som_scale 0.53 (0.1-3 log; 9 at a bound, no class pattern: it acts as
a heterotrophic-respiration multiplier, since the pools leave the spun-up equilibrium the moment it is applied);
rgrowth 0.17 (0.10-0.50; 11 at the floor, 3 at the ceiling; forests 0.15-0.17, i.e. half the F77 0.30);
bgc_e0 309 K (150-600; ENF 370 with four at 550-600, DBF 220 with three at 150-160, WSA 156 at the floor);
rroot 0.89 per yr (0.3-5 log; 6 at the floor). Reading: the night-time target is matched by any som_scale x E0
combination (trade-off, hence the scatter), while the day-time target is reached by cutting growth respiration,
which the model books in daylight (NPP_step = GPP - maintenance - rgrowth x (GPP - maintenance), hourly); the floor
pressure and the 0.89 day ratio point at the structural test queued in section 36 (growth respiration over 24 h),
not at wider bounds.

## 37. One set per PFT from the per-tower fits: the class-median parameter file (2026-10-10)

Hangkai (10-09): calibrate methane at the flux towers per class on the latest parameters, then a 200-year spin-up and
the historical transient, and compare carbon, fire and methane with the MIP products. The global model carries one
value per PFT, so the first step is to turn the per-tower fits (sections 35, 36.1) into such a set and measure what is
lost.

**Model change.** `BGCParams.rgrowth / rroot / rwood / lt_e0` accept a (12,) per-PFT vector as well as a scalar (yaml
tables keyed by PFT name); the canopy respiration uses the PFT's own value, and the Lloyd-Taylor E0 of the soil is the
cell's leaf-area-weighted mean over its PFTs (the soil has one temperature function; a stand of several PFTs gets the
blend). Scalars behave as before; the site fits still move scalars. 723 tests pass.

**The file** (`verify/calibration/class_median_yaml.py` -> `python-port/configs/params/params_s3_classmedian.yaml`,
table `logs/params_s3_classmedian_table.csv`, 67 entries, read back and checked): per-PFT photosynthesis entries
(vmax, stomatal slope, conifer acclimation) are the median over every tower that fitted that PFT, whatever its class
(mixed-forest towers inform both their PFTs); the scalar phenology and stress entries are medians over the towers that
fitted them (gdd_thresh_u 413 over 19 DBF/MF/WSA towers, gdd_thresh_l 74 over 24 towers, stressfac -3.34 over 61);
the respiration values of section 36.1 are class medians mapped to PFTs (ENF -> the three conifers, DBF -> the two
deciduous broadleaf, EBF -> the tropical/warm evergreens and the tropical deciduous, GRA -> both grasses, OSH+CSH ->
both shrubs): rgrowth 0.15-0.21 in forests and grass, 0.37 shrubs; rroot 0.78 ENF, 1.06 DBF, 1.64 EBF, 1.69 GRA,
0.67 shrubs; E0 369 ENF, 222 DBF, 312 EBF, 309 GRA, 306 shrubs. som_scale has no yaml form (a multiplier on a tower's
starting pools) and is dropped; k_scale stays 1 (the HWSD2 pinning was a site device; the global spin-up finds its own
soil carbon).

**Verdict at the 62 towers** (site run, same settings as the default run, grid pools; `diagnostics/s3_check_report.py`):

| monthly, 62 towers, 2009-2012 | default | robust S1 (section 34, one set) | class medians (one set) | per-tower fits |
|---|---|---|---|---|
| GPP r / NSE / bias | 0.862 / 0.73 / +7 | 0.894 / 0.79 / +2 | 0.883 / 0.78 / -5 | 0.939 / 0.88 / +1 |
| ET r / NSE / bias | 0.802 / 0.57 / +2 | 0.806 / 0.61 / 0 | 0.807 / 0.62 / 0 | 0.831 / 0.65 / +1 |
| NEE r / NSE / bias | 0.469 / 0.12 / +11 | 0.557 / 0.22 / +18 | 0.587 / 0.18 / +24 | 0.725 / 0.49 / +8 |
| annual GPP within 25 % of the tower | 58 % | - | 66 % | 90 % |

Reading: a single set per PFT recovers about a quarter of the per-tower GPP gain, whichever way it is built (the joint
S1 fit and the class medians land in the same place); the rest is tower-specific (leaf area, site vmax). The
class-median file is preferred because it carries the respiration values. The NEE bias here (+24) is the grid pools
being out of equilibrium with the towers' NPP once som_scale is gone; it is not a property the global run inherits,
because the spin-up equilibrates the pools (the per-tower NEE scores of 36.1 came with equilibrated pools and
som_scale). Still open from section 36: growth respiration booked in daylight; soil-evaporation excess in ET.

## 38. Methane at the flux towers: harness, pools, per-tower fits (2026-10-10, in progress)

Hangkai (10-09): calibrate the methane module at the methane towers, one parameter set per tower class, on the carbon
set of section 37; then the global run and the budget comparison (section 39).

**Data.** Chenchen's gap-filled sets (AmeriFlux 66 sites, EuroFlux 44; random-forest gap-fill of FCH4 at the USTAR50
threshold with FCH4_F_UNCERTAINTY; budgets in gC-CH4 m-2 yr-1; half-hourly FLUXNET-format meteorology with ERA fill;
NEE/GPP/LE partitioning). `verify/calibration/ch4_tower_prep.py` writes them in the FLUXNET2015 file layout the site
harness reads (`/shared/nas2/Hangkai/ibis/inputs/sites/methane_towers/{compact,halfhourly,monthly}`; selected with
`IBIS_FLUX_DIR`, `IBIS_FLUX_HH_DIR`, `IBIS_FLUX_MM_DIR`) and `ch4_targets.npz`: monthly CH4 sums (nmol m-2 s-1 x 1800 s
x 12.011e-9), the gap-fill sigma summed in quadrature, the measured share, months with >= 20 days. 107 sites with
coordinates and a budget: 60 WET, 16 CRO (no crop PFT; not fitted), 15 GRA, 9 ENF, 3 OSH, 2 EBF, 1 WSA, 1 SNO.
FLUXNET-CH4 itself is only metadata on the NAS.

**Model and harness.** `SimParams.methane` carries `MethaneParams` (yaml `methane:`), `run_month_gswp3` uses it when no
explicit set is passed, and fit_carbon has the extras `ch4_r` (r_ch4, 0.03-0.8 log), `ch4_q10` (q10p, 1.2-6 log),
`ch4_ztau` (z_tau, 0.08-1.0 log), `ch4_tauox` (tau_oxid, 0.02-0.6 log), `ch4_tref`, `ch4_zwtmax`. site_fit
`--target ch4 --ch4-targets`: the methane module runs with an inundated fraction of 1 at WET towers (the tower stands
in the wetland) and the WAD2M climatology of the 0.5-degree cell elsewhere (`--f-inund` overrides); the monthly flux
(`ch4_monthly`, kgC-CH4 m-2 month-1 from `MethaneState.flux` x days) is the series scored, with the robust objective of
the other targets (sigma = max(gap-fill sigma, 15 %, 0.05 gC)). Vegetation at WET towers: C3 sedge (temperate), shrub +
C3 sedge (lat >= 55 or MAT < 3), C4 grass (tropical, MAT > 18); leaf area from the MODIS cell maximum (WET added to the
attrs-modis classes). Forested swamps are not identified (no site classification beyond IGBP) and keep the sedge
prescription -- a known weakness.

**Pools.** `site_pool_spinup.py` on the 85 fitted sites (section 37 yaml, 2016-2021 cycled, k_scale 1; GPU 75 min):
median equilibrium NEE -13 gC m-2 yr-1 at WET, NPP 600; 16 sites keep |NEE| > 40 (tropical PE-QFR +211, US-ONA +201).
Soil carbon after the spin-up is 32 kgC m-2 at the WET median (grid pools 21), still far below peat stocks; the methane
production depends on Rh and its depth distribution, not on the stock, so this is acceptable for the fit and wrong for
a peat-carbon comparison.

**Fits.** Per tower, own years (the last <= 6 with CH4 months; first year warm-up, last held out), 4 parameters
(ch4_r, ch4_q10, ch4_ztau, ch4_tauox), 48 evaluations, pools from the spin-up (`--start-state`), CPU (4 cores, 3
workers, ~1.5 h per fit beside the ANN queue) plus a GPU launcher from the other end of the list. **Lesson (the AU-Stp
cap again):** the per-month cap of 25 (5 sigma) makes the objective flat wherever the starting model is more than 5
sigma from the tower every month -- near-zero grassland fluxes against a model that emits from the cell's WAD2M
fraction, or US-Cms where the model gives a tenth of the tower -- and CMA-ES then never leaves the start (5 of the
first 22 fits). Fits started after 11:34 use a cap of 400 (20 sigma); the first-batch flat fits are rerun with it.
Results: section 38.1 (pending).

### 38.1 Per-tower methane fits: verdict (2026-10-10)

Run history, for the record: the first pass (10-09 05:05 to 10-10 03:16) on CPU and GPU, then a wide-cap rerun of the
four flat fits, an inundated rerun of the eight strongly emitting non-wetland towers, and a repaired-forcing rerun of two
towers; the first versions of those reruns had lost `--start-state` through a generator slip and were discarded (the
valid batch is `site_fit/ch4_r5`). Two towers (US-HB4, US-Sne) had NaN pool states because their meteorology carried
sensor faults (a 25,640 mm half-hour of rain; longwave of -23,729 to 68,394 W m-2); `ch4_tower_prep.py` now turns
out-of-range meteorology into gaps and the two were re-spun. 19 towers with |budget| < 0.5 gC m-2 yr-1 were not fitted:
they cannot constrain a production ratio, and several are known wetlands with a budget of exactly zero in the EuroFlux
set (Ballynahone Bog, CZ-wet, AT-Nsd, AT-PsM, NO-Ikr, the four UK sites) -- a unit or scaling fault in that branch of
the gap-fill pipeline, reported to Chenchen. Four towers had only two usable years and were skipped. Report
`logs/ch4_fit_report.csv`, figures `figures/ch4_fit/` (`diagnostics/ch4_fit_report.py`, `ch4_fit_figures.py`).

**Wetland towers (45 informative, fitted ch4_r, ch4_q10, ch4_ztau, ch4_tauox; held-out = the last year):**

| band | n | r_ch4 median (IQR) | Q10 | z_tau | tau_oxid | held-out model/tower median | within 25 % | held-out monthly r |
|---|---|---|---|---|---|---|---|---|
| boreal (lat >= 60) | 13 | 0.17 (0.10-0.22) | 1.6 | 0.76 | 0.16 | 0.84 | 54 % | 0.66 |
| temperate | 25 | 0.11 (0.07-0.18) | 2.5 | 0.63 | 0.18 | 0.82 | 40 % | 0.83 |
| tropical (|lat| < 30) | 7 | 0.15 (0.11-0.17) | 1.4 | 0.66 | 0.09 | 0.95 | 29 % | 0.03 |

The gross production ratio comes out at 0.11-0.17, below the module's 0.23 and the legacy band table (0.15 / 0.32 /
0.18): the model emits too much per unit of heterotrophic respiration once its water table and oxidation are what they
are. The oxic-zone depth is unchanged at the median (0.16-0.18 against 0.10) and the decomposition depth deepens to
0.63-0.76 m (11 of 45 at the 1 m bound: the fits want more of Rh below the water table). Seasonality is where the
structure shows (figures/ch4_fit/CH4_1): boreal towers match the summer peak but the model emits nothing from November
to April where the towers keep 0.4 gC m-2 month-1 (frozen soil carries no methanogenesis in the module; shoulder-season
and under-ice emission are missing); temperate towers get a flatter peak than measured (2.0 against 3.2 gC m-2 month-1)
with Q10 pushed to 2.5 and two at the bound, i.e. temperature alone cannot produce the observed amplitude -- the
water-table seasonality of a real marsh is held at a constant inundated fraction of 1; tropical towers have no fitted
seasonality at all (r 0.03) for the same reason: their cycle is hydrological (flood pulse), and the harness holds the
tower inundated all year. Magnitude is right to within 25 % at only 43 % of the towers on the held-out year, median
0.84, so the per-class set is a scale correction, not a seasonal one.

**Non-wetland methane towers.** Grassland and forest towers that emit strongly (US-Snf 25, US-Sne 41, US-NGC 3, CA-SCC
6, BR-Npw 14, CH-Cha 2, RU-Fy2 2 gC m-2 yr-1) are wet sites classed by their vegetation; with the cell's WAD2M fraction
the model cannot emit there and the fit pushes r_ch4 to its ceiling, so they were refitted fully inundated. The class
medians (GRA 0.23, ENF 0.07, OSH 0.50, 9 / 5 / 2 towers) are not used for the global run: the inundated fraction of a
grid cell emits with the wetland set, and the upland sink the model lacks is what those towers mostly show (US-Ho1,
US-ICh, FI-Hyy, DE-Msr budgets of -0.2 to 0.04).

**Global translation.** `ch4_class_yaml.py` writes `methane:` (the 47-tower WET medians: r_ch4 0.142, Q10 2.06, z_tau
0.71, tau_oxid 0.158) and `methane_bands:` (the medians by band above) into
`python-port/configs/params/params_s4_classmedian_ch4.yaml`; run_global applies the band values by latitude with t_ref
unchanged (295.15 K, as the tower fits held it). The 1991-2010 methane rerun on that file is section 39.

## 39. The global run on the calibrated set, against the budgets and the gridded products (2026-10-10)

Protocol (`verify/calibration/final_validation.sh`, `SPINUP_YEARS=200 METHANE=1`): cold start 1701-1900 on
`params_s3_classmedian.yaml` with 1901-1930 GSWP3 cycled at 296 ppm (5 h on the GPU), transient 1901-2000 with real
years and rising CO2 (7 h beside the methane fits), then 1991-2000 and 2001-2010 again with monthly output and the
fitted methane sections of `params_s4_classmedian_ch4.yaml` (`$S/hindcast_rerun.sh`). Outputs under
`$V/final_s4_classmedian/{spinup,transient,hindcast_ch4_1991,hindcast_ch4}`. Two driver defects were found and fixed
on the way (METHANE.md 5s): the wetland map had been read mirrored in latitude, and the yearly CH4 field accumulated
across years. Comparison: `diagnostics/mip_compare.py` (`logs/mip_compare_s4_ch4_1991-2010.csv`),
`global_spatial.py` and `global_figures.py` (`figures/global_s4/`, `figures/global_s4_ch4/`). Forcing on the NAS ends in
2010, so "the latest decades" are 1991-2010.

**Carbon.** Land sink as the change of stored carbon (wood + litter + soil; NPP - Rh also carries the fire and
disturbance loss): +1.9 PgC yr-1 in the 1990s and +1.9 in the 2000s against GCB 2024's 2.50 and 2.83 (+-0.5) and the
TRENDY-v13 multi-model mean of 2.83 (range 1.4-4.4; the TRENDY IBIS gives 1.99). GPP 183-189 PgC yr-1 on true land:
1.70 x FLUXCOM (109 PgC on the land it covers) and 1.4-1.6 x the TRENDY models (115-135). The excess is regional
(figures/global_s4/S_GPP_spatial): 1.47 x in the tropics and south, 2.2 x in 30-60 N and 3.6 x north of 60; the
difference map puts it on the US Midwest, Europe, eastern China and India -- cropland that this run grows as natural
forest and grass (no land use) -- and on the boreal belt, where the class-median set raised GPP north of 60 from 6.6 to
11.9 PgC yr-1 (LAI 0.5 to 1.3). At the 62 towers the same set is within 25 % of the measured GPP at two thirds of the
sites, so the global excess is land cover and the boreal zone, not the photosynthesis of the calibrated classes. Wood
carbon 434 PgC (above-ground; the default run had 591), soil carbon 2445 PgC (default 1873; the respiration set slowed
the soil). NPP - Rh exceeds the stored-carbon change by 7-9 PgC yr-1: that is what the model loses to fire and
disturbance, four times the fire products -- a finding for the fire stage.

**ET** (2001-2010 against the five-product ensemble, figures/global_s4/S_ET_spatial): land mean 525 against 508 mm yr-1
(1.03 x), cell-wise r 0.93, 54 % of cells inside the product spread; 1.08 x in the tropics and south with the excess in
the semi-arid belts (Sahel, India, Australia, Chaco, Cerrado: the soil-evaporation excess of the ET stage), 0.99 x in
30-60 N, 0.75 x north of 60.

**Wetland methane** (Tg CH4 yr-1, 2001-2010; GMB 2025 bottom-up wetlands 2000-2009 with the model range):

| | global | 90S-30N | 30N-60N | 60N-90N |
|---|---|---|---|---|
| GMB 2025 | 153 (116-189) | 109 (66-146) | 32 (16-60) | 12 (4-28) |
| mirrored map, legacy bands (the transient) | 71 | 68 | 3 | 0 |
| corrected map, legacy bands (first hindcast) | 201 | 121 | 60 | 20 |
| corrected map, tower-fitted bands (final) | 151 | 126 | 11 | 13 |

The fitted set puts the global total and the tropical and boreal bands inside the budget's range; the temperate band
drops from twice the budget to a third of it, because the 25 temperate towers (Sacramento Delta marshes, European fens
and bogs, the Great Lakes) fit a production ratio of 0.11 against the legacy 0.32, and the band also holds the West
Siberian lowland south of 60 N where the map's emission is highest. The split is therefore 84 / 7 / 9 % against the
budget's 71 / 21 / 8 %. The seasonal structure of 38.1 (no winter emission in the boreal, constant inundation at the
towers) is carried into these numbers unchanged. 1991-2000 gives 139 Tg (117 / 11 / 12).

**Fire.** Burned area is scored by `verify/fire/evaluate_gfed5.py` on the 2010 state as in the v11 protocol; not rerun
here (the fire parameters are unchanged since section 12).

**What this run settles and what it leaves.** Settled: the calibrated carbon set gives the budget's land sink and the
products' ET pattern on a global grid, the methane module reproduces the budget's total once the map and the production
ratio are right, and the harness can carry a tower-calibrated set through a 300-year run in a day. Left, in the order
the evidence suggests: (1) seasonal inundation at the methane towers and in the global map's use (the temperate and
tropical seasonality, and the temperate band's magnitude); (2) a cropland or managed-land treatment for GPP; (3) the
boreal productivity under the class-median set; (4) the fire and disturbance carbon loss; (5) from section 36, growth
respiration over 24 h and the soil-evaporation excess. Open with Hangkai: TRENDY per-model fields (data policy), and
extending the forcing past 2010.

## 40. The boreal check: the set is right at the towers, the dynamic vegetation is not (2026-10-11)

Section 39 found GPP north of 60 N doubled under the class-median set (6.6 to 11.9 PgC yr-1) and 3.6 x FLUXCOM. Two
diagnostics settle where that comes from.

At the towers north of 55 N, with prescribed vegetation and leaf area, the class-median set is as good as the default
or better (annual GPP model / tower, 2009-2012): DK-Sor 0.82, FI-Hyy 0.82, FI-Sod 1.05, RU-Fyo 0.90, RU-Cok 0.84,
GL-ZaH 0.43 (the default 1.01 / 0.96 / 1.29 / 1.03 / 0.77 / 0.40; the per-tower fits 0.95-1.11). The set is not the
problem where the vegetation is held.

In the global run the vegetation is dynamic, and that is where the doubling happens (2001-2010 means by band, default
-> class-median, ILAMB reference): 60-70 N tree cover 0.39 -> 0.57, GPP 0.53 -> 0.88 kgC m-2 yr-1 (ILAMB 0.61), LAI
2.19 -> 1.78 (ILAMB 0.98); north of 70 N GPP 0.07 -> 0.39 (ILAMB 0.14) with the displayed LAI 0.41 -> 0.09. Lower
growth respiration (0.16 against 0.30) and the lower-canopy phenology medians (gdd_thresh_l 74 pooled over 24
grassland, shrubland and savanna towers; drop_stress_thresh 0.14; tthresh_l_c 2.4) let trees expand north and the
tundra grass photosynthesise through a longer season, neither of which a prescribed-vegetation tower fit can see; the
one tundra tower in the set, GL-ZaH, is the only tower where the class median fails (0.43), because its own fit (1.11)
wanted different lower-canopy thresholds from the temperate grasslands it was pooled with. Verdict: no defect in the
set; a limitation of calibrating a dynamic-vegetation model at fixed-vegetation towers, and a pooled phenology that
treats tundra as temperate grassland. The remedy is lower-canopy phenology by PFT or by climate class (the model holds
one set of thresholds for the whole lower canopy) and a tree-establishment check against the ESA CCI tree cover in
the boreal zone; both recorded for section 43, not done here. GL-ZaH is also 0.40 in the default run and 0.10 LAI at
the tower: the Arctic grassland is under-productive in every configuration.

## 41. Cropland as managed grass, calibrated at the cropland towers (2026-10-11)

**Why.** Section 39's GPP excess over FLUXCOM (76 PgC yr-1 in 2001-2010) is a third on cells with more than 30 %
cropland (ratio 1.8-2.1 there), where this model, which has no land use, had grown forest: tree cover 0.8 in cells
that are more than 60 % cropland. IBIS 2.6b4 maps the ESA CCI cropland classes to grassland at initialisation
(`process_esacci_lc.py`) and the dynamic vegetation then plants whatever the climate allows. The cropland fraction of
each cell, aggregated from the ESA CCI 300 m map of 2000 (classes 10/11/12/20 counted whole, the 30/40 mosaics at half;
`inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz`), is 21.7 Mkm2 of true land.

**Treatment** (`CropParams`, `SimParams.cropfrac`; commit 652fc52 and 9da8342). (1) Trees may cover at most the
non-cropland share of a cell: in `dynaveg`, where the cover the wood would give exceeds 1 - cropfrac, the tree pools
are scaled to the wood that gives exactly that cover (the inverse of the fu formula) and the removed carbon is a
clearing flux booked like a disturbance. (2) The cropland share of the grass leaf turnover leaves the field as harvest
(`harvest_index` 0.45) instead of becoming litter; both fluxes are written (`ayharvest`, `ayclearing`). (3) On the
cropland share the grasses photosynthesise with their vmax times `vmax_scale_c3` (PFT 11) or `vmax_scale_c4` (PFT 10),
through the same per-cell factor the conifer acclimation uses (`crop_vfac`). Without cropland the code path is the
old one (test_cropland.py). The driver takes `forcing.cropfrac_file`; the cropland map is on the model's own
north-to-south grid, so the GSWP3 row indices apply (unlike WAD2M, METHANE.md 5s).

**Tower fits.** FLUXNET2015 has 28 cropland towers, 20 with usable years; 16 with at least three years were fitted
(own years, last <= 6, last held out; `--fit crop_vmax_scale_c3|c4,stressfac`, the stress factor a nuisance parameter
because pycma cannot run a one-dimensional problem; 36 evaluations; Mead on 2007-2012 because 2013 has no usable
tower GPP). Cropland towers run as C3 grass (maize towers US-Ne1/2/3, IT-BCi as C4), fully cropland, leaf area at the
MODIS cell maximum. Report `logs/crop_fit_report.csv`.

| | C3 crops (12 towers) | C4 maize (4 towers) |
|---|---|---|
| crop / grass capacity, median (IQR) | 0.33 (0.30-0.35; 8 at the 0.30 floor) | 1.16 (0.79-1.42) |
| held-out annual GPP model/tower, median | 0.76 | 1.01 |
| held-out monthly r, median | 0.55 | 0.77 |

All 16: held-out annual ratio median 0.99, within 25 % at 62 %, monthly r 0.54. Reading: maize behaves like a C4
grass with a slightly higher capacity and the Mead fits are good (r 0.77-0.80). The C3 crops (wheat, barley, rapeseed,
sugar beet, rice) want a third of the natural grass's capacity, and eight of twelve press the floor: the model's C3
grass grows from the first warm days to the first frost, whereas a crop has a sowing date, a harvest in late summer and
bare soil after it, so the fit cuts the capacity to bring the annual sum down rather than the season in -- the held-out
r of 0.5-0.6 is that mismatch. Rice (US-Twt) fails outright (r -0.5): a flooded field is not a grassland. A crop
phenology (sowing by growing-degree days, harvest date, bare fallow) is the structural step; until then the C3 value
is a magnitude correction. Written to `params_s5_crop.yaml` (`crop: harvest_index 0.45, vmax_scale_c3 0.33,
vmax_scale_c4 1.16`); the global run with the cropland map is section 43.

## 42. The tower's water table in the methane fit: a test, not an adoption (2026-10-11)

**Why.** Section 38.1 found the methane seasonality structural at the towers: a constant inundated fraction of 1 and
the module's own water-table diagnostic cannot produce the temperate marshes' summer peak or the tropical flood pulse.
The one hydrological observation the towers carry is the water table (FLUXNET `WTD`, m, positive above the surface;
24 of the 60 wetland towers have one, 31 sites with at least six months).

**What was built.** `ch4_tower_prep.py` writes the monthly mean water table (`wtd_ch4`) and the inundated share of
the month (`finund_ch4`) into the targets; `methane_flux` / `methane_step` take an observed water-table depth
(`zwt_obs`, m below the surface, NaN = the module's own) that replaces the diagnosed depth inside the wetland; the
site fit's `--wtd-mode tower` passes it month by month (year-specific where recorded). The inundated share was tried
first and rejected on inspection: a bog with its water table 10-25 cm below the surface all year (CA-SCB) is
"never inundated" by that rule and would emit nothing, whereas the module's own treatment of depth is what the
observation should inform. 23 towers were refitted (`site_fit/ch4_wtd`, `ch4_wtd_gpu`; 20 with results, three too short).

**Result, paired with the constant-water-table fits of 38.1 at the same 20 towers** (`logs/ch4_fit_report_wtd.csv`):

| held-out year | model's water table | tower's water table | better / worse with the tower's |
|---|---|---|---|
| annual model / tower, median | 0.73 | 0.76 | 11 / 7 towers |
| monthly r, median | 0.66 | 0.50 | 12 / 5 towers |
| training-years monthly r, median | 0.55 | 0.50 | 4 / 9 towers |

The tower's water table helps where the record is a real wetland water table (CA-SCB bog 0.60 -> 0.73 r, US-ALQ,
US-Esm, US-Elm Everglades 0.03 -> 0.33, CA-DBB) and breaks the fit where the record's datum is not the surface: the
tidal marsh US-HB1 reports 0.6-4.2 m "above the surface" all year and gives a held-out ratio of 125, CA-DB2 and US-Tw4
lose their seasonality altogether. The medians do not move in one direction, the fitted production ratios scatter
more (temperate 0.09 against 0.11), and the boreal and tropical bands keep two and three towers. **Verdict: not
adopted.** The constant-water-table fits of 38.1 remain the parameter set, and the global run uses the module's own
water table. What the test did settle: where a wetland's water table is known and referenced to the surface, the
module's response to it is right (the seasonality follows), so the global seasonality problem is the inundation and
water-table input, not the production or oxidation formulation. A tower-by-tower screening of the WTD datum (or the
FLUXNET-CH4 v2 release's harmonised WTD) is the way to use this data; the code path stays.

## 43. The global run with cropland (2026-10-11/12)

**First attempt, wrong by construction (`final_s5_crop`, 13:08-19:56).** The cropland rule of section 41 capped the
cell's upper-canopy cover at the non-cropland share and scaled the tree pools down to the wood that gives that cover.
IBIS's cover rule saturates (fu = 0.95 at 3 kgC m-2 of wood with fu_k = 1), so a cell with 5 % cropland and a forest of
10 kgC m-2 had its wood cut to 3 kgC m-2 every year: a clearing flux of 12.5 PgC yr-1, wood carbon 434 -> 206 PgC, the
stored-carbon sink +1.9 -> +0.5 PgC yr-1, and GPP UP (189 -> 199 PgC yr-1 on true land) because the grass took the
cleared share at full productivity -- the cropland cells' ratio to FLUXCOM did not move (2.1). Harvest came out at
1.3 PgC yr-1. Kept as the record of a formulation that cannot work in a single-column cover model: the cover of a
few percent of cropland is not representable as a cap on a saturating cover.

**The rule that replaced it** (commit ed8fba5): cropland is a sub-grid share. The trees keep their natural wood
density on the non-cropland share (1 - c) and the cell's tree cover is (1 - c) times the cover their density gives,
`fu = (1 - c) x fu(wood / (1 - c))`; their GPP scales with their cover, so the pools equilibrate on their share with no
clearing. The crops are the grasses on the share c with the fitted capacity and the harvest of section 41. Tests:
no cropland reproduces the old path; wood is untouched; cover stays within the non-cropland share.

**Second run (`final_s6_crop`, 20:14-02:34).** The sub-grid rule behaves: no clearing, harvest 1.27 PgC yr-1, wood
carbon 349 PgC, the 2001-2010 stored-carbon sink +1.64 PgC yr-1 (GCB 2.78; numbers corrected 2026-10-12, section 44), wetland CH4 154 Tg (130 / 11 / 13 by
band), ET 1.04 x the ensemble. But GPP rose again (207 PgC yr-1 on true land) and the cropland cells kept their ratio
to FLUXCOM (2.15 in cells over 60 % cropland) with the tree cover now correctly at 0.26. The 2010 restart shows why:
in those cells the lower canopy is C4 grass (mean leaf area 3.4; 5.1 in the tropics, 0.6 C3 grass) and shrubs (2.4
at 30-60 N). The model's grass competition puts C4 grass wherever the warmest month exceeds 22 C and the shrubs take
the rest, so the "crop" ran as C4 grass at 1.16 x and as shrubs, and the C3 crop capacity of 0.33 that the European
and American towers asked for was applied to a grass that was hardly there. The fault is in letting the natural
competition decide what grows on cropland.

**The rule that replaced it** (commit 7450f75): on the cropland share the shrubs' NPP is scaled by (1 - c), so they
too keep to the natural share, and the crop capacity is a mix of the C3 and C4 values, (1 - q) x 0.33 + q x 1.16,
applied to both grasses, with q the C4 share of the cropland (`CropParams.c4_share`; -1 keeps the per-PFT rule the
tower fits used; the global file `params_s7_crop.yaml` sets 0.2, about the harvested-area share of maize, sorghum,
millet and sugar cane). What is still not represented: a crop's leaf area and season (the grass LAI on cropland is
dynamic, and a crop field is bare for part of the year) and a map of which crop grows where.

**Third run (`final_s7_crop`, 02:49-09:08) -- the result** (`logs/mip_compare_s7_2001-2010.csv`, figures
`figures/global_s7/`):

| 2001-2010 | s4 (no cropland, section 39) | s7 (cropland as managed grass) | reference |
|---|---|---|---|
| land sink, stored-carbon change, PgC yr-1 | +2.0 | +1.6 | GCB 2.8 +- 0.7; TRENDY 1.4-4.4 |
| harvest, PgC yr-1 | - | 1.36 | FAO crop NPP removed ~1-2 |
| GPP on true land, PgC yr-1 | 189 | 195 | TRENDY 115-135 |
| GPP / FLUXCOM, cells over 60 % cropland | 2.10 | 1.84 | - |
| GPP / FLUXCOM, cells 30-60 % cropland | 2.13 | 2.05 | - |
| GPP / FLUXCOM, cells under 10 % cropland | 1.61 | 1.61 | - |
| tree cover, cells over 60 % cropland | 0.78 | 0.26 | - |
| wood carbon / soil carbon, PgC | 434 / 2445 | 349 / 2416 | 450-650 (all vegetation) / 1500-2400 |
| ET / ensemble (cell r) | 1.03 (0.93) | 1.04 (0.93) | - |
| wetland CH4, Tg yr-1 (90S-30N / 30-60N / 60-90N) | 151 (126 / 11 / 13) | 153 (129 / 11 / 13) | 153 (109 / 32 / 12), range 116-189 |

Reading. The trees now sit on their share (tree cover 0.26 where cropland is more than 60 %, against 0.78 before),
the harvest is of the right order, the sink falls (+2.0 -> +1.6 against GCB 2.8) because the harvest removes carbon that
used to accumulate as forest on farmland, and methane and ET are unchanged. The GPP excess on cropland cells falls from
2.1 to 1.8 times FLUXCOM -- a gain, but a modest one, and the global GPP goes up rather than down (189 -> 195)
because the grass that fills the cropland share grows a dynamic leaf area of 3-4 m2 m-2 that no capacity factor of
0.33 holds down: lower capacity, lower NPP, lower leaf carbon, but the canopy closes anyway on a grass that has no
season. The cropland treatment is therefore a partial answer: it fixes the land cover and the carbon accounting
(trees, harvest), not the productivity, because a crop is a plant with a sowing date, a harvest date and bare soil
in between, and the model's grass has none of these. The structural step is a crop phenology and leaf-area rule on
the cropland share (sowing by growing-degree days, a prescribed peak leaf area, harvest and fallow); with it the
capacity factor fitted at the towers (0.33 for C3) would no longer be asked to carry the season. Recorded for the
next stage. Everything else the section-39 run showed stands: the sink, ET and methane inside or near their
budgets; GPP 1.5 x FLUXCOM in the tropics (where FLUXCOM itself sits 25 % below the towers), 2.3 x in the temperate
zone and 3.6 x north of 60 (section 40).

## 44. Min's four points: the PFT envelope (Köppen mask), organic soils, cropland, dynamic wetland area (2026-10-12, in progress)

Min's comments on the IBIS 3.0 report, and what was done with each.

**1. The PFT envelope.** IBIS decides which of its 12 PFTs may exist in a cell from four temperature
normals only (absolute minimum, warmest month, GDD0, GDD5; `climate_analysis.existence`, table in the
README of 2026-10-12). Nothing in the envelope says how dry a cell is, which is why trees had to be
held off steppes with the survival gate (section 14) and why the calibrated set grows them onto the
tundra (section 40). Two findings while summarising it:

- *Larch is broken in the original Fortran.* `climate.f:398-400` compares `gdd5` with `TminL(4)`
  (-45) where the comment above it says `GDD(4)` (1200); the comparison is never true, so PFT 8 exists
  only where the coldest month averages below -45 C. The port reproduced it. `physics.larch_gdd_fix`
  (default off, Fortran-identical) uses the degree-day limit the comment intends.
- *Köppen-Geiger supplies the missing moisture axis.* `ibis.io.koppen` maps each of the 30 classes
  of Beck et al. (2023, Sci. Data 10:724; 0.5-degree maps per 30-year period, on the NAS under
  `/shared/nas/Public/Climate_Biome/Koppen_Beck2023`) to the upper-canopy PFTs it allows, and
  `existence()` multiplies the mask into its envelope; the lower canopy is never masked. The table is
  in the module docstring: A keeps the tropical trees (Aw the drought-deciduous one only), B is
  treeless except the hot steppe (acacia savanna), Cs the Mediterranean set, Cw/Cfa the subtropical
  set, Cfb/Cfc and D the temperate-boreal set, E treeless. `forcing.koppen_file` points at the npz of
  `scripts/koppen_prep.py` (periods 1901-1930 ... 1991-2020; the run picks the period of each year,
  the spin-up takes the first). Decided with Hangkai 2026-10-12: prescribe first (this), derive the
  class from the model's own normals later if it pays.

**2. Soils.** The 0.5-degree texture file already is HWSD2 (Min's `hwsd2_to_soita.py`, March 2026),
but only sand and clay were read; organic carbon and bulk density were discarded, and IBIS's
hydraulic and thermal properties were texture-only with a constant 1 % organic correction on the
solids. `scripts/hwsd2_organic_to_nc.py` puts ORG_CARBON and BULK on the six IBIS layers with Min's
rule (SHARE x depth-overlap per mapping unit, pixel-count mean per cell) into
`inputs/soil/soita_hwsd2_organic.nc` (with sand and clay recomputed alongside), and
`physics.soil_organic` mixes the Lawrence & Slater (2008) organic end-member into porosity,
Campbell b, saturated suction and conductivity, field capacity, wilting point, the solids' density
and heat capacity (`soil_init`) and the dry thermal conductivity (`setsoi`) by
f_om = min(1, OC/100 x 1.724 x BD / 130 kg m-3). Off by default and bit-identical when off.

**3. Cropland.** Left as it is (section 41-43); Chris's crop model will replace the managed grass on
the cropland share later. No change.

**4. Dynamic wetland area.** The methane module takes the WAD2M fraction as a 2000-2020 monthly
climatology, identical every year. Min wants a switch, off by default, that scales it with
precipitation for the years without observations. Before any code: a literature review
(GIEMS/SWAMPS/WAD2M interannual variability, the precipitation-scaled extents of the methane budget
studies, the TOPMODEL family) and a regression of WAD2M 2000-2010 on the model's precipitation per
band, to see how much of the variance precipitation explains. Both in progress; results below.

**Experiment design.** All from the s7 1900 state (section 43) through the 1901-2000 transient and
the 2001-2010 hindcast (`final_validation.sh` with `SKIP_SPINUP=1`, env `KOPPEN`, `LARCH_FIX`,
`SOIL_ORGANIC`/`SOIL_FILE`), so each differs from s7 in one switch and the 110 years let the
vegetation and the soil water re-equilibrate: k1 = Köppen mask; k2 = Köppen + larch fix; o1 =
organic soils. Verdicts on the section-43 quantities (sink, GPP by band vs FLUXCOM, ET by band,
tree cover by band vs the reference, methane, fire), model/obs ratios first.

**Correction to sections 39-43 (2026-10-12).** `mip_compare.py` grouped years by calendar decade, so for a 2001-2010
record its last "decade" row was the single year 2010, and its summary line reported that row. The s7 numbers in
section 43's summary and table, in the IBIS 3.0 report and in the README were read from it: GPP 201, wood 356, soil
2418 and the sink were year-2010 values. The 2001-2010 means (script fixed: decades labelled 2001-2010, period means,
first-year dC/dt from the year before the range) are GPP 195 PgC yr-1, stored-carbon sink +1.64 (s4 without cropland
+1.95; the IBIS default run +2.77; GCB 2001-2010 2.78 +- 0.69), wood 349 and soil 2416 PgC, CH4 154 Tg (130 / 11 /
13). Two verdicts change: the sink is BELOW the Global Carbon Budget, not inside it, and the cropland treatment moved
it away from the budget rather than toward it (the harvest removes carbon that used to accumulate as forest). The s5
and s6 numbers quoted earlier in section 43 came from the same row and carry the same year-2010 bias. The band
ratios (GPP / FLUXCOM, ET / ensemble), the methane and fire scores and everything at the towers are unaffected.

**Results, k1 and o1 (2026-10-13 00:30; k2 running).** Both from the s7 1900 state, 1901-2000 transient and
2001-2010 hindcast; verdicts as model/reference ratios (`$S/k44_summary.py`; s7 numbers are the corrected
2001-2010 means).

| 2001-2010 | s7 | k1 Köppen mask | o1 organic soils (L&S 2008, linear k_sat) |
|---|---|---|---|
| GPP, PgC yr-1 (x FLUXCOM, cell r) | 195 (1.75, 0.85) | 187 (1.68, 0.85) | 198 (1.77, 0.78: NaN cells) |
| GPP / FLUXCOM by Köppen group A / B / C / D / ET | 1.35 / 1.91 / 2.11 / 2.44 / 4.92 | 1.31 / 1.75 / 2.00 / 2.37 / 4.42 | 1.31 / 2.13 / 2.14 / 2.56 / 5.05 |
| tree cover (fu) by group A / B / C / D / ET | 0.71 / 0.34 / 0.64 / 0.65 / 0.42 | 0.72 / 0.28 / 0.54 / 0.64 / 0.25 | 0.67 / 0.34 / 0.63 / 0.69 / 0.42 |
| fu 30-60 S (reference 0.23) / phantom-tree area | 0.63 / 10.8 % | 0.34 / 6.3 % | 0.63 / 11.8 % |
| fu 60-90 N (reference 0.25); GPP ratio there | 0.53; 3.58 | 0.51; 3.58 | 0.59; 3.78 |
| stored-carbon sink, PgC yr-1 (GCB 2.78) | +1.64 | +1.43 | +3.36 (soil C still rising 2.3 PgC yr-1) |
| wood / soil / litter carbon, PgC | 349 / 2416 / 150 | 296 / 2369 / 130 | 370 / 2608 / 240 |
| ET / ensemble (cell r) | 1.04 (0.93) | 1.04 (0.93) | 1.03 (0.87) |
| wetland CH4, Tg yr-1 | 154 | 152 | blew up (one cell 8e6) |
| burned area x GFED5 | 0.99 | 0.97 | 1.07 |
| cells gone NaN by 2010 | 0 | 0 | 1,259 (309 after the first year) |

*k1.* The mask does what it was built to do and nothing more. Where it removes trees (B, C, ET) the cover falls
to or toward the reference: the 30-60 S steppes from 0.63 to 0.34 against 0.23, the phantom-tree area from 10.8
to 6.3 % of land, the tundra from 0.42 to 0.25. GPP falls 4 % (195 -> 187) and the FLUXCOM ratio from 1.75 to
1.68, with the pattern unchanged (r 0.85). It does not touch the two large excesses: the D climates (30 % of
land, ratio 2.44 -> 2.37, cover 0.65 -> 0.64) because every temperate and boreal tree is allowed there, so the
taiga keeps a closed canopy (fu 0.5 against 0.25 north of 60) by competition rather than by envelope -- the
section-40 finding in a different light: the boreal excess is a density problem inside Dfc/Dfd, not a treeline
problem; and the tundra GPP ratio stays 4.4 with no trees at all, so there it is the lower canopy (shrubs and
grasses, fl 0.74) and the reference itself. The sink moves further from the budget (+1.64 -> +1.43) because
less wood accumulates. Verdict: adopt as the default envelope (it removes real errors and costs nothing), but it
is not the GPP fix; that stays with the boreal tree density and the cropland season.

*o1.* Not adoptable as run, for three reasons in order. (1) NaN: 309 cells after the first year, 1,259 (1.3 % of
land) by 2010, mostly tropical and subtropical with ordinary organic fractions (f_om 0.3, not the peat cells).
Cause found: the linear mix of saturated conductivity gave a cell with a third organic matter a k_sat a hundred
times its clay value (1e-4 m s-1 is 360 mm h-1), which the hourly soil-water solver cannot take. Lawrence &
Slater mix k_sat by percolation theory (series below f_om 0.5, a connected fraction in parallel above), which
the first run did not; `organic_ksat` now does, and a one-year smoke test (`smoke_o1p`) says whether the NaN
goes away. (2) Disequilibrium: the 1900 state's soil carbon was in balance with mineral properties; the wetter,
colder organic column slows decomposition and the soil pool gained 230 PgC over the transient and is still
rising 2.3 PgC yr-1 in 2010, so the +3.36 sink is the pool filling, not a verdict -- o1 needs its own 200-year
spin-up before the sink or the soil carbon mean anything. (3) Where it stays finite it does not help: the boreal
GPP ratio rises (3.58 -> 3.78, cover 0.53 -> 0.59), ET is unchanged, and the methane module blew up in a few
cells on the way to NaN. The organic fraction itself is large: f_om above 0.9 in 14 % of the land's top layer
(boreal and peat soils), mean 0.31, as the CLM definition gives with HWSD2's 0-20 cm organic carbon.

*o1, second look (03:30).* With `organic_ksat` (percolation mix) the one-year smoke run `smoke_o1p` still lost
5 cells (309 before) and the methane module still exploded in 3 (one cell 7,000 kgC m-2 yr-1). The monthly
files show what fails: the soil HEAT solver. In a dry organic column (Tibet, 33.75 N 80.25 E, f_om 0.91 in the
top layer; the Sudd, 8.75 N 30 E, f_om 0.8 through the column) layer temperatures reach hundreds then thousands
of degrees within a month (2821, 4467, -3769 C) while the water solver is still finite; everything downstream
(Rh, Q10, methane) follows and the cell is NaN the month after. The organic solids of the Lawrence & Slater
end-member carry a volumetric heat capacity five times smaller than mineral soil ((1 - 0.9) x 1920 x 1300
= 2.5e5 J m-3 K-1 against 1.25e6) and a dry conductivity of 0.05 W m-1 K-1, and IBIS's hourly soil-heat step
with its explicit freeze-thaw exchange is not stable for a dry layer of that capacity. The conductivity change
alone is not the problem, the capacity is. GPP and ET also moved +6 % in the first year, so the hydraulic
part is not small either. A hemic-end-member run (`smoke_o1h`) tests whether the slower drainage changes the
picture; the thermal failure needs a solver guard (a floor on the solids' heat capacity, or sub-stepping of the
soil column where it is small) before organic soils can be evaluated at all. Recorded; parked behind the
new-PFT stage unless Min wants it first.

*o1, third look (03:40): the hemic end-member is stable.* `smoke_o1h` (percolation k_sat, Letts et al. 2000
hemic: porosity 0.88, b 6.1, k_sat 2e-6 m s-1) ran its year with 0 NaN cells (309 for the first o1, 5 with the
percolation mix alone), a sane methane field (max 61 gC m-2 yr-1 against s7's 74) and GPP +2.9 %, ET +3.4 %.
That reorders the diagnosis above: the fibric drainage (b 2.7, k_sat 1e-4) dried the organic layers to zero
water, and it is a DRY organic layer whose heat capacity is too small for the heat step; the hemic curve holds
the water and the column stays well-posed. So no solver guard is needed for the hemic configuration, and that is
the one to evaluate: o2 = organic soils (hemic) with its own 200-year spin-up, beside s7b = s7 with its own
200-year spin-up, because s7's spin-up was inherited from s5 and the soil-carbon comparison needs the same
protocol on both sides. Queued behind k2.

**Result, k2 (06:10): the larch fix is inert from a restart.** k2 (Köppen mask + `larch_gdd_fix`) is k1 to four
digits on every score (GPP 186.97 vs 186.98 PgC yr-1, wood 295.8 vs 295.9, sink +1.43 both, every band ratio
equal). The 2010 restarts say why: with the fix larch MAY exist at 12,372 points instead of 308 (the same count
as the two other boreal trees, as the rule intends), but after 110 years its wood is 0.2 TgC against 50 PgC for
the boreal evergreen conifer, and not one point carries more than 0.1 kgC m-2 of it. A PFT that starts at zero
biomass cannot establish in `dynaveg`: the upper-canopy NPP is shared by leaf-area fraction, the seed leaf
(`xminlai`) is a negligible share, and the exponential never takes off. Two consequences. (1) The Fortran bug is
real but inert in any run that starts from an existing state; the fix is the right rule and costs nothing, so it
goes into the default together with the mask, but its effect can only show in a cold-start spin-up where the
land-cover map seeds larch. (2) For the new-PFT stage this is the structural item that comes first: a new type
needs a seed from the land-cover map at cold start or an establishment rule (a minimum share of the canopy NPP,
or a biomass floor where it may exist), or it will never appear in a run started from today's states.

Verdict on the envelope (k1 = k2): adopt the Köppen mask and the larch rule as the default envelope -- they
remove the steppe, Mediterranean and tundra phantom trees at no cost -- and carry the taiga density and the
tundra lower canopy to the new-PFT stage. Hindcast scoring of k2 was re-run outside the launcher
(`$S/final_validation_from_hindcast.sh`) after the launcher was patched while k2 was executing it.

**Results, s7b and o2 (2026-10-13 13:50): organic soils run clean and make the soil carbon worse.** Both
cold-start 200-year spin-ups (1701-1900), 1901-2000 transient, 2001-2010 hindcast, the full protocol.

| 2001-2010 | s7 | s7b (s7, fresh spin-up) | o2 organic soils, hemic, percolation k_sat |
|---|---|---|---|
| GPP, PgC yr-1 | 195.3 | 195.3 | 195.7 |
| stored-carbon sink, PgC yr-1 (GCB 2.78) | +1.64 | +1.64 | +1.13 |
| wood / soil / litter carbon, PgC (soil 1500-2400 to 1 m) | 349 / 2416 / 150 | 349 / 2416 / 150 | 344 / 3110 / 242 |
| soil-carbon drift 2001-2010, PgC yr-1 | +0.44 | +0.44 | +0.45 |
| GPP / FLUXCOM 60-90 N; tree cover there (ref 0.25) | 3.58; 0.53 | 3.58; 0.53 | 3.60; 0.49 |
| tree cover all natural land (ref 0.335); phantom area | 0.556; 10.8 % | 0.556; 10.8 % | 0.542; 9.7 % |
| ET / products by band 60-90 N / 45-60 / 30-45 | 0.65 / 0.87 / 1.21 | same | 0.66 / 0.89 / 1.23 |
| wetland CH4, Tg yr-1 (bands 90S-30N / 30-60N / 60-90N) | 154 (130 / 11 / 13) | same | 154 (128 / 12 / 14) |
| burned area x GFED5 | 0.99 | 0.99 | 1.00 |
| cells gone NaN by 2010 | 0 | 0 | 0 |

*s7b reproduces s7 to every printed digit.* So the s7 spin-up was its own (the note in section 43 that it was
inherited was wrong), and a 310-year run is bit-reproducible across launches -- worth knowing before any
ensemble work.

*o2.* With the hemic end-member and the percolation conductivity the organic column is numerically sound: no
NaN in 310 years, methane bounded (max 85 gC m-2 yr-1 against s7's 74). It changes almost nothing that is
scored -- GPP, its pattern, tree cover, ET, fire and methane are within 2 % of s7 -- and the one thing it
changes a lot it changes the wrong way: the equilibrium soil carbon is 3,110 PgC against 2,416 (literature
1,500-2,400 to 1 m), litter 242 against 150, because the wetter, colder organic column decomposes slower, and
the sink falls from +1.64 to +1.13 PgC yr-1. The two small gains (tree cover 0.556 -> 0.542, phantom area 10.8
-> 9.7 %) do not pay for that. Verdict: not adopted. The switch stays (`physics.soil_organic`,
`soil_organic_end_member: hemic` is the configuration that runs), and the reason it does not help is itself a
finding: IBIS's soil-carbon equilibrium is already at the top of the literature range with mineral soils, so
anything that slows decomposition pushes it out; the soil respiration side (section 36, the growth-respiration
and Rh structure) has to move first. Min's item 2 is closed with that record.

## 45. IBIS 3.1: Köppen-zone PFTs and recalibration on tower NEE (2026-10-13, design)

Hangkai's decision (2026-10-13, after Min's comments and sections 44): regenerate the PFT list from the Köppen
mask, 30-40 or more types, then recalibrate the parameter set on the towers' NEE. This supersedes the four-PFT
proposal of 2026-10-12 and the Köppen mask of section 44 as such: the mask becomes the definition of the types.

**The list** (`verify/calibration/pft_koppen_design.py` -> `docs/pft_koppen_design.csv`): each PFT is a growth
form in a Köppen zone. Twelve zones -- A humid (Af, Am), A savanna (Aw), B hot (BWh, BSh), B cold (BWk, BSk),
C Mediterranean (Cs), C subtropical (Cwa, Cfa), C tropical highland (Cwb, Cwc), C oceanic (Cfb, Cfc), D
temperate (Dfa/b, Dwa/b, Dsa/b), D subarctic (Dfc, Dwc, Dsc), D extreme (Dfd, Dwd, Dsd), E tundra (ET) -- and
nine forms (needleleaf evergreen, broadleaf evergreen, broadleaf deciduous, needleleaf deciduous, evergreen
shrub, deciduous shrub, C3 grass, C4 grass, wetland vegetation), pruned to the combinations that exist, plus the
two managed crops: 41 types, 21 trees, 7 shrubs, 8 grasses, 3 wetland types, 2 crops. Each carries a `parent`
among the 12 IBIS PFTs whose parameters it inherits until its own towers are fitted, and the canopy it lives in
(the sunlit/shaded big-leaf structure stays: trees in the upper canopy, the rest in the lower).

**Towers.** 295 distinct towers (206 FLUXNET2015, 107 FLUXNET-CH4, 2 unassigned snow sites), assigned by IGBP
class and the 1-km Köppen class of the site: 28 of the 41 types have three or more towers, 7 have one or two,
and 6 have none (tropical highland evergreen tree, subarctic broadleaf deciduous, larch, open taiga conifer,
tundra dwarf shrub, tropical shrub). The large classes are the temperate wetlands (43), the continental conifer
(30), C3 crops (29), boreal wetlands (23), temperate C3 grass (21), subarctic conifer (20), continental
deciduous (20). Types without towers keep their parent's fitted parameters and only their envelope and cover
rules differ -- that is stated in the table, not hidden.

**What has to be built, in order.**
1. *Establishment* (section 44, k2): a PFT at zero biomass never grows in `dynaveg`, so every type needs a seed
   -- at cold start from the land-cover map crossed with the Köppen class (ESA CCI gives the form, Köppen the
   zone), and an establishment rule (a minimum share of canopy NPP where the type may exist) so the vegetation
   can move when the climate does.
2. *npft generalised*: the 12 is wired through the existence rules, every per-PFT parameter array, fire
   mortality, the phenology and canopy groups, the vegetation-type map, restarts and the site harness. The
   generalisation keeps the 12-PFT path bit-identical (the F77 tests) and keys the new types by `parent` and
   form.
3. *Existence by type*: the Köppen zone list of each type replaces the temperature envelope as the first gate
   (the envelope stays as the second, by form), the mask of section 44 retires.
4. *Calibration on NEE*: per-tower fits with the NEE harness of section 36 (night-time respiration and
   day-time uptake, held-out year), grouped by type; class medians per type; parent medians where a type has
   no towers; then the GPP/ET verdicts of sections 37 and 39 and the global protocol.
Expected effort: one to two weeks for 1-3, one week of fits for 4 (about 300 towers at the section-36 cost,
six at a time), two days of global runs; three to four weeks to a scored IBIS 3.1.

**Why the GPP is high: it is leaf area, not physiology (2026-10-13 16:00).** Decomposition of the s7 2001-2010
excess over FLUXCOM into leaf area (annual mean of the monthly cover-weighted displayed LAI against the MODIS
annual mean of the ILAMB collection), light use per unit leaf area, and growing-season length (months with GPP
above 20 % of the annual maximum, model against FLUXCOM's monthly climatology):

| region | GPP ratio | LAI ratio (model / MODIS, m2 m-2) | GPP per leaf area | season, months model / ref |
|---|---|---|---|---|
| all land | 1.75 | 2.81 (4.33 / 1.54) | 0.62 | 8.6 / 7.4 |
| tropics | 1.47 | 2.35 (4.72 / 2.01) | 0.62 | 10.3 / 9.3 |
| 30-60 N | 2.29 | 3.56 (4.25 / 1.19) | 0.64 | 7.3 / 6.1 |
| 60-90 N | 3.58 | 2.86 (2.75 / 0.96) | 1.25 | 4.2 / 3.8 |
| Köppen A / B / C / D / ET | 1.35 / 1.91 / 2.11 / 2.44 / 4.92 | 2.07 / 2.83 / 4.21 / 2.99 / 7.51 | 0.65 / 0.67 / 0.50 / 0.81 / 0.65 | |
| cropland > 50 % | 1.89 | 4.29 (4.72 / 1.10) | 0.44 | 9.1 / 7.8 |

The model carries 2.8 times the observed leaf area and photosynthesises 0.62 times the observed amount per unit
of it; the excess GPP is the excess canopy, damped by its own shading. Temperate climates hold the extreme (LAI
7.8 against 1.9, two stacked full canopies: tree cover 0.64 with a grass understorey at 0.80), the tundra the
largest ratio (2.5 against 0.34, a lower canopy that nothing limits), the cropland share a grass of 4.7. Only
north of 60 does the rate per leaf also run high (1.25, where FLUXCOM is weakest), with a season 10-25 % too
long everywhere outside the tropics. The towers did not see this because every fit held the leaf carbon at the
site-attribute or MODIS leaf area (`--fixed-lai attrs`, `--lai-mode attrs-modis`, sections 31-36): the
physiology was calibrated under observed canopies; the model's own canopy generation (leaf allocation and
turnover, the lower-canopy LAI, the cover rules, the two-canopy stacking) was never under test. So the answer
to "is it the framework" is yes: the vegetation dynamics, not the parameters of photosynthesis. For IBIS 3.1
this fixes the calibration design: NEE at towers calibrates the physiology and respiration of each type; the
leaf area, cover and season of each type must be fitted against MODIS LAI, tree cover and the seasonal cycle
by type and zone, or the global GPP stays where it is however well the towers fit.

**Towers acquired (2026-10-13 evening).** Hangkai: fetch the towers that exist but are not on disk before the
type list is decided. ICOS Warm Winter 2020, the ICOS L2 archive and the FLUXNET v1.3 product (open through the
FLUXNET Shuttle, CC-BY-4.0, including 140 AmeriFlux and NEON Alaska sites) add 344 sites not on disk, 313 of them
vegetated with at least one year of 50 % NEE coverage (`inputs/sites/new_sites_inventory.csv`, DATA_INVENTORY 6).
Towers per type before -> after, under the assignment rules of `pft_koppen_design.py` unchanged:
larch T19 0 -> 12 (RU-SkP, RU-Ege, JP-Tef, JP-Fhk ...), birch-aspen T18 0 -> 1, open taiga T20 0 -> 0 (FI-Ken,
FI-Var, SE-Svb, US-xDJ, CA-HPC are Köppen Dfc and go to T17), tundra dwarf shrub S06 0 -> 1, tropical highland T04
0 -> 1 (CN-Ash), tropical shrub S07 0 -> 1, boreal wetlands W03 22 -> 40, closed taiga T17 18 -> 28, continental
deciduous T15 20 -> 41, subtropical conifer T07 4 -> 29, Mediterranean sclerophyll T09 5 -> 16, cold-steppe grass
G04 4 -> 11, steppe shrub S02 1 -> 10, juniper-pinyon T21 1 -> 7; full table `figures/pft31/PFT31_tower_counts.csv`.
Six types stay under three towers: T04 1, T05 2, T18 1, T20 0, S06 1, S07 1 -- the decision on the treeline,
openness and boundary rules and on the merges (2026-10-13) is open with Hangkai.

Map: `verify/calibration/diagnostics/pft_koppen_map.py` -> `ibis_local_document/figures/pft31/PFT31_koppen_types_map.png`
(dominant type per cell; barren = bare ground > 60 %, wetland = WAD2M > 25 %; towers on disk before 13 October as
circles, acquired as stars, filled with the colour of the type they calibrate). The dominant-type areas
(`PFT31_dominant_area.csv`) put 40 of 41 types on the map; C4 crops cannot be separated without a crop-type map.

**The list is final: 40 types (2026-10-13, Hangkai).** T20 (northern open taiga conifer, no tower) is merged into
its nearest type, T17 (subarctic conifer evergreen tree, now "closed and open taiga": same parent PFT 6, same form,
zones D_subarctic + D_extreme + E_tundra). The other thin types stay separate and are calibrated on their own
towers: T04 tropical highland evergreen tree (1, CN-Ash), T05 hot-arid woodland tree (2), T18 subarctic
birch-aspen (1, US-Rpf), S06 tundra dwarf shrub (1, GL-Dsk), S07 tropical shrub (1, ID-PaB). No treeline,
openness or boundary rule is applied and no other merge. IDs are kept stable (there is no T20). Caveat recorded,
not acted on: a type fitted on one tower has no between-tower spread, so its held-out year is its only check.
Towers per type after the acquisition: `figures/pft31/PFT31_tower_counts.csv`; map redrawn with 40 types.

### 45.1 Build design (2026-10-13)

**Types live in IBIS's 12 positions, not in a 40-wide state.** Every cell has one Köppen zone, and within one
zone the tree types are at most three broadleaf and two conifer, the shrubs one, the grasses one. So each cell
maps each of its types to one of the 12 IBIS positions, and every per-PFT parameter becomes a per-cell array
(npoi, 12) gathered from a 40-row type table. The state arrays, restarts, fire and mortality tables and the
Fortran-identical path stay as they are; switched off (no type map), every cell takes the parent values and the
model is bit-identical. Outputs per position convert back to the 40 types through the same map.

Placement rule: a type takes its parent's position (T01 -> 1, T17 -> 6, T19 -> 8, ...); trees keep their leaf
form (broadleaf positions 1, 2, 3, 5, 7; conifer 4, 6, 8, the groups of the big-leaf photosynthesis); if the
parent's position is taken, the type takes the first free position of its leaf form. Two zones need that: A
savanna (T02 and T03 are both drought-deciduous, T03 goes to position 1) and B hot (T03 and T05, T05 to position
1). Lower canopy: the zone's shrub type takes position 9 (evergreen) or 10 (deciduous) by its habit, the zone's
grass takes 11 (C4) or 12 (C3); the other shrub and grass positions keep the IBIS 3.0 generic shrub and grass with
their own envelope, so C4 grass still grows in the hot-summer temperate climates and deciduous shrubs in the
boreal zone. Tree positions with no type in the zone are off. Wetland types W01-W03 are not positions: they
are the methane parameter sets by zone (replacing the latitude bands); crop types C01/C02 are the crop capacity
factors on the cropland share.

**What becomes per type.** vmax, specla, tauleaf, tauroot, tauwood0, aleaf/aroot/awood, rgrowth, rroot, rwood,
lt_e0, the Ball-Berry slope and the acclimation parameters, and the leaf habit (evergreen, drought-deciduous,
cold-deciduous), which phenology reads from the type instead of from the position. Existence: the zone gate
(the type map) times the type's parent envelope.

**Competition inside a big leaf.** All types in one big leaf (the conifers of a cell, say) get the same net
assimilation per unit leaf area, so their own vmax hardly matters to how they compete -- the reason larch never
established (k2). New switch `stomata.gpp_split_by_capacity`: each big leaf's GPP is split among its positions by
displayed leaf area x own vmax instead of by leaf area alone. With the mixed vmax (`vmax_mix`) as the leaf-area
weighted mean, the shares still sum to one, so the canopy total is unchanged and only its division moves. Off by
default.

**Establishment.** Every position that may exist keeps the minimum leaf carbon (`xminlai`), as now; at cold start
the land-cover initialisation fills positions by parent and the type map relabels them, so a type starts where its
parent had biomass. Whether the capacity split is enough for a seeded type to take hold is tested before any
further rule is added.

**Calibration.** (1) New towers to the FLUXNET2015 layout the harness reads (as `ch4_tower_prep.py` does for the
methane towers). (2) Per-tower NEE fits on every tower with three or more good years, each tower run as its zone's
cell with its type in its position, under the corrected respiration of section 36; class medians per type; a
type's parent median where it has no fit. (3) Leaf area, cover and season per type against MODIS LAI, tree cover
and the seasonal cycle by zone (the GPP finding above). (4) The global protocol and the section-43 scores.

### 45.2 The tower campaign (2026-10-13)

`towers31_prep.py` put every tower on disk into the FLUXNET2015 layout the harness reads
(`inputs/sites/towers31/compact`, with `FLUXNET_siteinfo.csv`; NAS paths, no symbolic links on the share): 605
towers, 478 calibration-grade (a window with a good held-out year and at least one good fitted year, and a
half-hourly file), 358 of them natural vegetation. Record per tower: the source with the most good NEE years (a good
year has >= 240 daily values and a mean NEE_VUT_REF_QC >= 0.5; 240 because high-latitude records leave the winter
unfilled -- RU-SkP), ties to the newer processing (FLUXNET v1.3 291 towers, FLUXNET2015 64, ICOS 3). Window: five
years (four or three for short records) within 1996-2024, the latest with the most good years (145 end in 2024).
CO2 knots 2021-2024 added (Mauna Loa annual means). Calibration-grade towers per type: T01 11, T02 4, T03 12, T04 1,
T05 1, T06 9, T07 26, T08 21, T09 14, T10 16, T11 4, T12 9, T13 9, T14 6, T15 36, T16 37, T17 24, T18 1, T19 8,
T21 7, S01 7, S02 9, S03 4, S04 7, S05 10, S06 1, S07 1, G01 4, G02 4, G03 11, G04 11, G05 5, G06 35, G07 9, G08 7.

Protocol, per tower, in the harness of sections 31-36 (a type runs at its parent's position at a single tower,
which is the same physics as at its zone position -- same leaf group, same habit -- so the fits need no type
machinery; the type enters when the per-tower values are grouped): (S2) GPP and ET, `towers31_cmds.py`: vmax and
slope of the prescribed PFTs, acclimation of evergreen conifers, upper or lower phenology where present, drought
trigger, stress factor; IBIS 3.0 set as base; TRY vmax bounds; robust objective 25,30 with `--robust-auto` (caps
lifted to 400,1e9 where the start is already at the tower cap: CN-Ash starts at 30.0, 118 uncapped); 80 evaluations.
Leaf area: attributes where a tower has the file, else the MODIS cell maximum (not the model's own state, which is
the inflated leaf area of the GPP finding). (P) pool spin-up per window group from the S2 parameters. (NEE) night and
day NEE (`tower_nee_daynight.py --years 1996-2024 --siteinfo`) on the S2 start and the pools: som_scale, bgc_e0,
rgrowth, rroot, 48 evaluations. Then per-type medians -> the `types:` section, plus per-zone phenology and stress
from the same fits. Launched 2026-10-13 18:55, at most seven fits at a time (4-core slots, nice 5), thinnest types
first; S2 is expected to take 2-3 days, the NEE stage 1-2 more.

**Wetland and crop types (2026-10-13).** W01/W02/W03 are the methane parameter sets, now assigned by the wetland type
of the cell's Köppen zone instead of by latitude when the types are on (`biome_methane_params(band_index=)`,
`pft_types.wetland_band`; unclassified cells keep the latitude rule). The 66 per-tower methane fits regrouped by zone
(`logs/ch4_towers_zone.csv`, `ch4_class_yaml.py --band-csv`): 12 towers change band (five subtropical sites --
the Everglades and similar -- go from temperate to tropical, seven between temperate and boreal); wetland medians
W01 (12 towers) / W02 (24) / W03 (12): boreal r_ch4 0.143 (latitude bands 0.170), temperate 0.126 (0.113), tropical
on 12 towers instead of 10. C01/C02 are the crop capacity factors: 46 calibration-grade crop towers (16 before),
fitted with the section-41 protocol after the NEE stage (`$S/t31_crop_cmds.txt`); maize at US-Ne1/2/3 and IT-BCi as
before, every other crop tower as C3 -- the new towers' crop types are not on disk, so rotations with maize (CN-Yuc,
KE-Mkt, SN-Nkr and others) are an open item.

**First hour of the campaign (2026-10-13 19:00-19:40).** Seven fits running, all moving except one. ID-PaB (the only
tropical-shrub tower) returned NaN from its first evaluation: not the model (a forward run is clean) but the target --
the v1.3 record has no variable-u* GPP partition there. A scan found 68 towers where FLUXNET v1.3 leaves the
variable-u* partitions (GPP_NT_VUT_REF, GPP_DT_VUT_REF) empty while the constant-u* ones exist (DE-Lnf, US-NR1,
RU-SkP, CH-Dav ...); `towers31_prep.fill_gpp` fills the VUT columns from the CUT twins (alternative estimates of the
same flux, Pastorello et al. 2020), applied in place to the written files and recorded in the site table
(`GPP_FILL_MM`). Two towers have no GPP partition at all in their window (BR-Ji3, ES-Amo): placeholder results skip
their S2 fit and send them to the NEE stage from the base set. ID-PaB, rescued by the fill, was rerun in the spare
slot after four orphaned workers of the killed fit were stopped (they held its log and its cores).

**The NaN fits, found (2026-10-14 00:45).** Five S2 fits (ID-PaB, CD-Ygb, ES-LJu, PA-SPs, PR-xGU) ended after a few
evaluations with loss NaN. Not the model and not the target: `tower_obs` set each month's GPP sigma as
np.maximum(|NT - DT| / 2, 10 % of GPP, 5 g), and at towers where FLUXNET v1.3 gives only one partition (DT only at
PA-SPs, GF-Guy ...) the NT - DT spread is NaN, which np.maximum propagates into every sigma and the loss; the ET
sigma had the same flaw where LE_F_MDS is missing (that one silently dropped the ET term). np.fmax now. Fits with
both partitions -- every FLUXNET2015 tower and every fit that had finished -- are unchanged. The five were rerun
(outputs of the NaN runs kept in `t31_s2/_nan_runs`). AU-DaP is not a bug: its loss moved between 10.8 and 22.9
and none beat the start, which sits near the lower vmax bound -- the tower wants less C4 capacity than TRY allows.

**Tower leaf area in the campaign: a known bias, kept (2026-10-16, Hangkai).** At the 69 towers with a ground maximum
LAI (site attributes), the MODIS peak LAI of the tower's own PFT is 0.72x the ground value (ENF 0.62, CRO 0.41, DBF
0.77, GRA 0.80, SAV 0.82; `logs/modis_vs_ground_lai.csv`), and the value `site_fit` holds at towers without an attribute
file (`fit_carbon.modis_at`: the cell maximum times the natural-vegetation share) is 0.60x. 328 of the 358 campaign fits
use it, so their fitted capacity (vmax, slope) is expected to be biased high, roughly 20-30 %, compensating for too few
leaves. Options put to Hangkai: pause and rerun those towers with ground-corrected leaf area (end ~21 Oct) or finish as
is (~19 Oct). Decision: finish as is. Consequence carried into the leaf-area stage: when leaf area is raised toward
ground-corrected values, the per-type capacity must be re-checked at the towers (re-scored with the corrected leaf area)
and corrected where it overshoots. Literature review of how other models determine and calibrate LAI launched the same
day (`$S/lai_review.md`).

**GPP stage end and the NEE start (2026-10-17/18).** The launcher printed "S2 CHAIN DONE" at 08:56 with six fits still
running (its command loop was piped, so `wait` ran in a subshell), and the pool stage started on their intermediate
best.json (written as a fit improves). Caught at the first window group (1996-2000, not one of the six); the stage was
stopped, the launcher fixed (no pipe, a final wait on its own fits, a tower done only when its log has the final "BEST
eval" line, towers running in another launcher skipped) and the pools resumed at 10:11 after the last fit. 358 S2
results: 356 fitted, BR-Ji3 and ES-Amo the placeholders. On Hangkai's word (10-18) the NEE fits started at 02:50 for the
191 towers whose pools were ready instead of after all 52 groups, and both remaining stages use 8 slots (cores 0-31).
The first NEE fit failed at its start value: the s7 class-median set holds rgrowth, rroot and lt_e0 per PFT (12,), and
the NEE harness expected scalars (`float(array)`), and its setter would have replaced the vector by a scalar. Fixed: the
start value is the tower's own PFT's value, and a fitted scalar is spread over the vector (`fit_carbon._set`), the
scalar semantics of section 36. CN-Ash then started from its class medians (bgc_e0 311.7, rgrowth 0.201, rroot 1.643),
baseline loss 82.2, ~90 s per evaluation.

## 46. A per-type maximum leaf area (2026-10-16, Hangkai: "to keep IBIS slim")

**Why.** The GPP excess (195 PgC yr-1, 1.75x FLUXCOM, section 44) is mostly leaf area: s7's cell LAI is 2.8x MODIS at the seasonal peak
and 3.3x on the annual mean (2010, true land north of 60 S, `diagnostics/laicap_compare.py`), while GPP per unit LAI is
0.62x. IBIS has no limit on leaf area: the leaf pool relaxes to aleaf x tauleaf x NPP, so a productive cell keeps
adding leaves. Most big-leaf DGVMs cap it (ORCHIDEE `lai_max`, JULES `Lmax`, JSBACH, CLM crops; review in
`logs/lai_review_2026-10-16.md`), and uncapped models overestimate LAI (CLM4/4.5 "chronic overestimation", Noah-MP).
Of the options from the review (cap; cap + understorey light rule; pipe-model allometry), Hangkai chose the cap alone.

**Rule** (`vegetation_dynamics.cap_leaf_area`, called by `dynaveg` and `biomass_update` after the minimum-LAI rule).
A layer's LAI per unit of its cover -- upper sum(plai[:8]) / fu, lower sum(plai[8:]) / fl, the canopy LAI the
photosynthesis sees -- may not exceed the leaf-area-weighted mean `lai_max` of its PFTs; the layer's leaf pools are
scaled down to it. The removed leaf carbon goes to the same PFT's wood and roots in its awood:aroot ratio (roots for
grasses), so plant carbon and this year's litterfall are unchanged (ORCHIDEE redirects leaf allocation to sapwood and
roots the same way). fu comes from wood and is held fixed (the moved carbon can only raise it); fl rises with the
lower LAI, so the lower limit is the largest L with L <= cap x fl(L), found by bisection. The cap acts on the year's
leaf pool, so it bounds the seasonal peak; phenology still shapes the season below it. On the cropland share of the
lower canopy the cap is `crop.lai_max` (cell cap (1 - c) x grass cap + c x crop cap). Off unless a params file sets
`lai_max` (bit-identical: 26 dynaveg/biomass_update cases against the committed code, fire, survival, cropland and
both cover rules); per type through `types.<id>.lai_max` (a type without one keeps its parent's). Tests:
`tests/test_lai_cap.py`.

**Values** (`verify/calibration/laicap_yaml.py`, written into `configs/params/params_s7_laicap.yaml` = s7 + caps): the
midpoint of ORCHIDEE's lai_max and JULES's Lmax for the matching PFT -- the generous end of the published caps, as
MODIS reads low over needleleaf, crops and dense canopies (Hangkai: "MODIS might underestimate"):

| PFT | ORCHIDEE | JULES | cap | s7 canopy LAI, median / p90 (2010 restart, dominant PFT) |
|---|---|---|---|---|
| tropical broadleaf evergreen | 7 | 9 | 8.0 | 7.5 / 10.4 |
| tropical broadleaf deciduous | 5 | 7 | 6.0 | 6.0 / 8.7 |
| warm-temperate broadleaf evergreen | 4 | 7 | 5.5 | 7.7 / 9.4 |
| temperate conifer evergreen | 5 | 7 | 6.0 | 6.1 / 7.8 |
| temperate broadleaf deciduous | 5 | 7 | 6.0 | 3.8 / 5.5 |
| boreal conifer evergreen | 3.5 | 7 | 5.25 | 5.2 / 6.4 |
| boreal broadleaf deciduous | 4 | 7 | 5.5 | 2.7 / 3.8 |
| boreal conifer deciduous | 3 | 6 | 4.5 | (none dominant) |
| evergreen / deciduous shrub | -- | 4 | 4.0 | 1.5 / 3.3, 1.4 / 2.3 (lower canopy) |
| C4 grass | 2 | 3 | 2.5 | 2.2 / 5.6 (lower canopy) |
| C3 grass | 2.5 | 3 | 2.75 | 4.8 / 8.0 (lower canopy) |
| crops (cropland share) | 5 | -- | 5.0 | (CLM5 crops 5-7) |

Offline on the s7 2010 state (the cap applied once to the pools, F77 lower cover): it binds in 31 % of cells in the
upper canopy and 40 % in the lower; the cell peak LAI goes 5.39 -> 3.96 (-27 %). Most of the cut is the C3-grass lower
canopy (median 4.8 -> 2.75, including grass under forests: boreal broadleaf cells 6.2 -> 4.5 with the tree canopy
untouched) and the warm-temperate evergreens (7.7 -> 5.5). Even then the model would sit near 2.3x MODIS at the peak:
part of the remaining gap is the two stacked canopies (fl is 0.98 under most forests), which the cap does not touch.

**Global test (2026-10-16).** Two arms from the s7 2008 restart, 2009-2010, monthly output
(`verify_outputs/laicap_smoke/{cap,ctl}`, comparison `compare_2010.txt`); the control is needed because a restart
from 2008 already differs from the continuous s7 hindcast in its first month. The cap acts at the end of 2009, so
2010 is the first capped year. In the capped run's end-of-2009 state no cell exceeds its cap in either layer (with
the cropland blend), and plant carbon is unchanged.

| 2010 | control | cap | reference |
|---|---|---|---|
| GPP, PgC yr-1 (x FLUXCOM) | 196.0 (1.78) | 190.3 (1.73) | 110.0 |
| GPP 90S-30N / 30-60N / 60-90N (x FLUXCOM) | 1.55 / 2.32 / 3.74 | 1.52 / 2.23 / 3.33 | |
| cell LAI, peak month (x MODIS) | 4.89 (2.80) | 3.94 (2.26) | 1.74 |
| cell LAI, annual mean (x MODIS) | 3.91 (3.30) | 3.24 (2.74) | 1.18 |
| cell LAI peak, 60-90N (x MODIS) | 4.98 (4.01) | 3.08 (2.48) | 1.24 |
| ET, mm yr-1 (x ensemble) | 540 (1.06) | 547 (1.08) | 508 |
| NPP, PgC yr-1 | 77.3 | 71.8 | |
| NEE, PgC yr-1 | 5.7 | 1.0 (first-year transient) | |

**Verdict: the cap does what it is meant to do for leaf area, not for GPP.** Leaf area falls 17-20 % (38 % at the
peak north of 60 N), but GPP only 2.9 %: the canopies are light-saturated. The GPP-weighted cell LAI is 7.0 in the
control and 5.6 with the cap, and a crude absorption index, cover x (1 - exp(-0.5 LAI)) with the upper canopy
shading the lower, falls 3.1 % globally and 14 % north of 60 N, matching the GPP changes band by band (-2.9 %,
-11 %). Leaf area is a GPP lever only below LAI ~3, i.e. near the MODIS values, which the literature caps do not
reach. Side effects: NPP falls more than GPP (-7 %) because the moved carbon raises wood and root maintenance
respiration; the 2010 NEE drop is the first-year transient of that and is not a budget number (no spin-up with the
cap); ET is +1 %.

## 47. Why the global GPP is 1.75x FLUXCOM: the diagnosis (2026-10-16, Hangkai: "dig deeper")

**The target.** FLUXCOM RS+METEO (110 PgC yr-1 on our land) is the low end of the defensible range, about 110-160 for
2001-2010: SIF-GOSIF 135, NIRv 147, COS 157 +- 8.5, 18O 150-175; ISIMIP2a models 106-134, CMIP6 84-152 (review with
DOIs: `logs/gpp_estimates_review_2026-10-16.md`). A FLUXCOM low bias accounts for at most ~35-45 PgC, mostly tropical;
s7's 195 is ~35 above 160, so the model is high whichever reference is used.

**At the towers' own cells** (`diagnostics/gpp_tower_cells.py`, `logs/gpp_tower_cells_s7.csv`; 435 natural towers,
model and FLUXCOM 2001-2010 climatologies against each tower's own years): tower / FLUXCOM 1.43 (the towers stand in
more productive patches than their 0.5-degree cell, section 1), global run / FLUXCOM 2.32, global run / tower 1.62. The
site model with the same set reproduces the towers (section 39), so the excess at the tower cells is not the
parameters but the difference between the global configuration and the tower configuration.

**The controlled chain** (`diagnostics/gpp_chain.py`, `verify_outputs/gpp_chain/b1`; the 62 towers of the 2008-2012
batch, s7 parameters, CO2 series, the s7 state at the end of 2007, 2008 warm-up, 2009-2010 scored; one ingredient at a
time from the global run to the tower; medians of the per-tower step ratios):

| step | what changes | ratio |
|---|---|---|
| G/F | the global vegetation frozen (composition, pools, cover) | 1.00 (the harness reproduces the global run) |
| F/L | leaf area scaled from the global run's (cell LAI 7.8) to the towers' prescription (2.7: site attributes for forests, else the MODIS cell maximum) | **1.41** |
| L/V | composition and cover to the tower prescription (no grass layer under trees, no trees at open sites) | 1.11 |
| V/S | GSWP3 at the cell -> the tower's weather | 1.04 |
| S/T | the s7 site model -> the tower | 0.89 |
| G/T | | 1.46 |

The elasticity dlnGPP/dlnLAI between F and L is 0.29: leaf area is a weak lever at the top of the range (the cap,
section 46: -20 % LAI for -3 % GPP) but not over the whole gap between the global leaf area and the leaf area the towers
were fitted at. By class: DBF F/L 1.45, L/V 1.28; ENF 1.24, 1.14; GRA 1.50, 1.02; OSH 2.02, 1.17.

**Over the globe, by observed land type** (CLM/MODIS PFT shares, ESA CCI cropland; 2001-2010; total excess 82 PgC):

| observed | area | model / FLUXCOM | share of the excess | model tree cover | observed tree share |
|---|---|---|---|---|---|
| forest (trees >= 60 %) | 27 % | 1.39 | 25 % | 0.86 | 0.87 |
| mixed (10-60 %) | 26 % | 1.97 | 35 % | 0.66 | 0.32 |
| grassland | 9 % | 2.37 | 10 % | 0.48 | 0.02 |
| shrubland | 7 % | 5.45 | 7 % | 0.40 | 0.03 |
| bare / sparse | 15 % | 2.84 | 5 % | 0.34 | 0.03 |
| cropland (>= 50 %) | 16 % | 1.89 | 18 % | 0.29 | 0.13 |

In real forests the tree cover is right and the ratio (1.39) is the towers' own over FLUXCOM (1.43). Three quarters of
the excess is where the model grows trees the observation does not have. **s7 runs F77's cover rules**: no water limit
on tree establishment (`tree_stress_min` -1), a 25 % cover floor in each canopy (`fu_min`, `fl_min`), the linear lower
cover (`cover_k` 0). The global calibration had fixed this (section 14: the gate; stages b3/b4: GPP 118-132 PgC on true
land with tree cover 0.27-0.36); the site-first calibration (section 20) started again from the defaults, and towers
with prescribed vegetation cannot see tree establishment or cover. Section 40 had found the same in the boreal zone.

**Season.** 30-50 N: model / FLUXCOM 14 in December-January, 2-4 in spring and autumn, 1.4-1.6 in summer; November-March
holds 21 % of the model's GPP against 7 % and 31 % of the band's excess. Tropics: 1.3-1.5 in every month.

**Leaf area, revisited.** The equilibrium leaf pool is aleaf x tauleaf x NPP, so leaf area and GPP feed each other.
Split of the peak-LAI excess (s7 2001-2010, `diagnostics/lai_npp_decomp.py`; CUE_obs 0.5 assumed): LAI 2.72x MODIS =
GPP 1.75 x CUE 0.76 x leaf area per NPP 2.04 globally. Leaf area per unit GPP (assumption-free) is 1.55x MODIS and
1.12x MODIS / 0.72 (the MODIS/ground ratio at the towers). The towers were fitted near the MODIS level (section 45.2),
so against the model's own calibration the leaf parameters (aleaf, tauleaf, specla -- all still F77's: 25 / 12.5 /
20 m2 kgC-1, never calibrated because they are dead at fixed-leaf-area towers, section 0) carry ~1.5x too much leaf per
unit GPP.

**Decision (Hangkai, 2026-10-16): keep dynamic vegetation and calibrate the tree gate against the observed tree cover**
(not a prescribed land-cover map); leaf-area calibration after it.

**Gate, first estimate** (`diagnostics/gate_calib.py` on a one-year s7 run that writes the gate's input, `aystressu`,
and the NPP of each canopy, `aynppu`/`aynppl`, now in run_global's yearly files; `verify_outputs/gate_test/diag`).
Annual-mean upper-canopy stress separates the observed classes (median 0.23 where trees < 10 %, 0.48 at 10-30 %, 0.61 at
30-60 %, 0.82 in forest) but overlaps (p90 of treeless land 0.48, p10 of forest 0.42). With the wood response of the
gate (equilibrium wood x (1 - e^(-1/tau)) / (1 - s e^(-1/tau))) and the cover rule, the score rmse + 2 x forest lost
picks tree_stress_min 0.225, ramp 0.30; with the floor fu_min 0.05 the natural-land cover goes 0.648 -> 0.457 (obs
0.334), phantom area (obs < 0.15, model > 0.4) 13.0 -> 5.0 %, forest lost (obs > 0.6, model < 0.3) 1.0 -> 3.0 %. The
soil-water feedback is not in this estimate. Global test: `verify_outputs/gate_test/gate10` (`params_s7_gate.yaml` =
s7 + the three values), 2001-2010 from the s7 2000 restart, i.e. the s7 hindcast's own start, so the hindcast is the
control (its first month reproduced bit for bit in GPP, LAI and soil state).

**Gate, global test (2026-10-16 18:16; `gate_test/gate10/compare_2006-2010.txt`, `diagnostics/gate_compare.py`; means
2006-2010 against the s7 hindcast).** The gate does what it is fitted for: natural-land tree cover 0.70 -> 0.52 (obs
0.39), phantom area 15.2 -> 6.2 %, forest lost 1.1 -> 3.2 %, r 0.60 -> 0.65; tree cover on observed grassland 0.59 ->
0.29, shrubland 0.42 -> 0.14, bare land 0.36 -> 0.09 (obs 0.02-0.03). **GPP does not follow: 193.9 -> 191.7 PgC yr-1
(-1 %)**; grassland 2.42 -> 2.34 x FLUXCOM, shrubland 5.39 -> 5.38, bare land 2.76 -> 2.52, mixed 2.00 -> 2.00, cropland
1.92 -> 1.83, north of 60 N 3.64 -> 3.77; ET and LAI unchanged. Where the trees went, the lower canopy took the land and
photosynthesises as much (lower-canopy share of global NPP 0.42 in 2010, 0.16 in observed forest). So the excess
outside forests is not the tree life form but the productivity of whatever grows there -- the same reading as the
chain (leaf area 1.41, composition and cover 1.11). The grasses carry the most leaf per unit NPP of any PFT (aleaf x
tauleaf x specla 13.5 for C3 grass, 11.25 C4, against 7.5 for the trees) and F77's lower cover saturates (fl = LAI,
clipped at 0.975), so a modest dryland NPP grows a full, dense grass canopy. Next: the leaf-area calibration, grasses
and shrubs first. The gate is kept as the tree-cover fix (a realism gain at no GPP cost); its 3 % forest loss is the
price of the stress overlap and can be revisited per type when the 3.1 types carry their own stress.

## 48. The leaf-area stage: references and where the excess sits (2026-10-17)

**Reference (Hangkai, 2026-10-16/17).** The MODIS/ground factor 0.72 of section 45.2 is not a basis for a global
correction: it is a median over 64 towers (IQR 0.51-0.95; 20 ENF, no EBF) of a plot maximum against a 0.5-degree cell,
so it mixes representativeness with retrieval error, while MODIS C6 is nearly unbiased against upscaled true-LAI maps
(2.28 vs 2.31, Yan et al. 2016) with class-specific low biases (needleleaf clumping, pasture/crops, saturation). The
references are therefore two products as they are, their mean the target and their difference the tolerance:
**GLASS LAI V61** (MODIS-based, 0.05 deg 8-day, glass.hku.hk, public) and **GIMMS LAI4g** (Cao et al. 2023, doi
10.5194/essd-15-4877-2023; AVHRR NDVI with networks trained on 3.6 million Landsat LAI samples; ILAMB's 0.5-degree
monthly copy). The reprocessed MODIS C6.1 LAI (Lin et al. 2023) needs a registration form and was not used. ILAMB's
"MODIS" LAI is NCAR's 2014 Collection-5 climatology, the lineage of the CLM surface data used as the reference until
now. `verify/calibration/lai_products_prep.py` (system Python, GDAL with HDF4) writes
`inputs/eval/lai/{glass_v61,gimms_lai4g}_0.5deg_monthly_2001-2010.npz` (north-up, model grid). Checks: Amazon and
Congo 5-6, Sahara 0, boreal peak 3-4; GLASS-GIMMS cell correlation 0.98 (annual mean); GIMMS leaves deserts and ice
empty (86 % of land), GLASS alone is used there. On common cells the products are above the old reference: annual mean
/ peak month 1.46 / 2.28 (GLASS), 1.63 / 2.60 (GIMMS), 1.37 / 2.02 (CLM Collection 5).

**The model against them** (the gate run of section 47, 2006-2010; cell LAI fu x laiu + fl x lail):

| region | peak LAI x obs | annual mean x obs | GPP x FLUXCOM | leaf per unit GPP x obs (peak) |
|---|---|---|---|---|
| global | 2.28 (GLASS 2.43, GIMMS 2.15) | 2.82 | 1.75 | 1.30 |
| 90S-30N | 2.01 | 2.27 | 1.52 | 1.33 |
| 30-60 N | 2.48 | 4.04 | 2.29 | 1.09 |
| 60-90 N | 3.30 | 4.59 | 3.77 | 0.87 |
| observed forest | 1.73 | 2.22 | 1.41 | 1.23 |
| observed mixed | 2.53 | 3.30 | 2.00 | 1.26 |
| observed grassland | 2.77 | 3.34 | 2.34 | 1.18 |
| observed shrubland | 4.63 | 4.58 | 5.38 | 0.86 |
| observed bare / sparse | 4.66 | 3.78 | 2.52 | 1.85 |
| observed cropland | 2.67 | 3.80 | 1.83 | 1.45 |

Three parts: the leaf parameters (leaf per unit GPP 1.3x globally, 1.1-1.5x by type), the GPP the leaves are grown from
(1.75x), and the season (annual mean / peak excess 1.6x at 30-90 N: leaves out too long). With dlnGPP/dlnLAI ~0.29
(section 47) and LAI proportional to the leaf-per-NPP factor k times NPP, LAI scales as k^1.41 and GPP as k^0.41:
correcting k by the observed 1.3 takes LAI 2.28 -> ~1.6x and GPP 1.75 -> ~1.58x; pushing LAI all the way to the
products would need k 1.8x below what the products imply -- the leaves compensating for productivity per leaf.

**Decision (Hangkai, 2026-10-17): match the observed leaf area per unit GPP**, not the leaf area itself.

**Factors** (`diagnostics/leaf_per_gpp.py` on the gate run, 2006-2010; peak-month cell LAI against the GLASS/GIMMS
mean, annual GPP against FLUXCOM; cells where one PFT holds >= 50 % of the leaf area, cropland < 30 %): model / observed
leaf per unit GPP = tropical broadleaf evergreen 1.33 (4109 cells), tropical broadleaf deciduous 1.18 (3618),
warm-temperate broadleaf evergreen 1.22 (1820), temperate conifer 0.85 (1722), boreal conifer 1.05 (5721), C4 grass
1.23 (6201), C3 grass 0.91 (11608); deciduous shrub 1.47 on 50 cells only; temperate and boreal broadleaf deciduous,
larch and evergreen shrub dominate too few cells (0-11) and stay as they are; cropland 1.46 (on the grass PFTs; left for
the crop model). C3 grass is not too leafy for its GPP: its leaf area (3.35x) follows its GPP (3.67x).

**Which parameter carries each factor.** The equilibrium leaf area is aleaf x tauleaf x specla x NPP, and specla or
tauleaf scale it linearly without touching the carbon allocation (aleaf would move carbon to wood and roots). Trait
review (`logs/leaf_trait_ranges_2026-10-17.md`: TRY, Kattge et al. 2011, which CLM5 and LPJmL reuse; GLOPNET, Wright et
al. 2004; JSBACH): each factor goes through the parameter that it moves TOWARD its published range.

| PFT | factor | parameter | change | trait range |
|---|---|---|---|---|
| tropical BE | 1.33 | specla | 25 -> 18.8 | TRY 19.7 (12.7-30.5), GLOPNET 18.0 |
| tropical BD | 1.18 | tauleaf | 1.0 -> 0.85 | specla would fall under TRY's 21.7 (TRY 32.1) |
| warm-temperate BE | 1.22 | specla | 25 -> 20.5 | TRY 19.7, GLOPNET 15.4, LPJmL 15.0 |
| temperate conifer | 0.85 | tauleaf | 2.0 -> 2.35 | needle lifespan TRY 3.3 (2.0-5.4); specla would exceed TRY's 14.1 |
| boreal conifer | 1.05 | specla | 12.5 -> 11.9 | TRY 10.4 (7.7-14.1) |
| C4 grass | 1.23 | tauleaf | 1.25 -> 1.02 | leaf lifespan TRY 0.14, GLOPNET 0.16; specla would move away from TRY's 40 |
| C3 grass | 0.91 | specla | 20 -> 22.0 | TRY 41.9 (26-66) |

F77's grasses are far from the traits on both specla (20 against ~40) and lifespan (1.25-1.5 against 0.14-0.5), in
opposite directions: IBIS's single yearly leaf pool cannot represent sub-annual grass turnover, so only the product is
constrained, by the observations. Tree aleaf 0.30 is at the top of the evidence (0.22 global forests, Litton et al.
2007; 0.34 canopy incl. reproduction in the tropics, Malhi et al. 2011), left as it is. Parameter file
`configs/params/params_s7_gate_leaf.yaml` (= params_s7_gate.yaml + the seven changes); global test
`verify_outputs/gate_test/gate_leaf10`, 2001-2010 from the s7 2000 restart, control the gate run.

**Global test of the leaf step (2026-10-17 06:02; `gate_test/gate_leaf10/{compare_vs_gate,leaf_per_gpp}_2006-2010.txt`,
control the gate run).** The calibration converges: leaf per unit GPP, model / observed, tropical BE 1.33 -> 1.17,
tropical BD 1.18 -> 1.05, warm-temperate BE 1.22 -> 1.07, temperate conifer 0.85 -> 0.96, boreal conifer 1.05 -> 1.02,
C4 1.23 -> 1.06, C3 0.91 -> 1.02 (tropical BE wants a second step). **GPP barely follows: 191.7 -> 189.0 PgC yr-1
(1.75 -> 1.73 x FLUXCOM)**; tropics 1.52 -> 1.48, 30-60 N 2.29 -> 2.29, north of 60 N 3.77 -> 3.80; peak LAI -14 % in
the tropics, +6 % north of 60 N (C3 grass and the temperate conifers needed more leaf); ET 1.04 -> 1.05; NPP 73.7 ->
74.6; tree cover unchanged. The estimate of 1.58x in the decision above was wrong: it assumed one global 1.3x cut, but
the factors have both signs (the largest PFT by area, C3 grass, went up) and where leaf area fell, in the tropics, the
canopy is light-saturated (-14 % LAI, -2.3 % GPP). With tree cover (section 47) and leaf per unit GPP now consistent
with the observations, the remaining 1.7x is GPP per unit leaf area and the season: leaf area and GPP feed each other,
and neither fix moved the level the loop settles at.

**What sets the level (2026-10-17).** The leaf area the 62 calibration fits held, against the GLASS/GIMMS cell peak:
1.26x below the products at the 41 towers held at the Collection-5 MODIS cell maximum (held 1.8, products 2.6), 0.73x at
the 21 forest towers held at their site attribute (4.2 vs 3.2); a capacity re-check against the products' leaf area would
move tower GPP by only about +7 / -9 % (elasticity 0.29), with opposite signs -- not the lever. At the 54 towers with >= 18
good months in 2009-2010 (`gpp_chain/b1`): tower / FLUXCOM 1.43 (1.30 MODIS-held, 1.68 attribute-held), site model with
the tower's vegetation on GSWP3 (V) / FLUXCOM 1.33, global run / V 1.48, global run / FLUXCOM 2.22. The fits matched plot
GPP with the cell's leaf area, so the model carries tower-level GPP per unit leaf area (1.3x the grid's), and the dynamic
vegetation, growing leaves in proportion to NPP with dlnGPP/dlnLAI 0.29, amplifies it: 1.33^(1/(1-0.29)) = 1.50, the
1.48 the global run adds. Bringing GPP per unit leaf area to the grid level would let the loop settle ~1.5x lower
(global ~1.73 -> ~1.15x FLUXCOM, leaf area with it) -- an estimate for a global test. The open choice is the global GPP
level the spatial stage targets (section 47: defensible 110-160 PgC yr-1, FLUXCOM at the low end).

## 49. Toward a reasonable GPP without a target (2026-10-19, Hangkai: "No target but try our best to make it more reasonable")

No capacity scaling to a chosen total: structural causes are fixed on physical grounds and the result is judged against
FLUXCOM's pattern and the published range (110-160 PgC yr-1, section 47).

**Lower-canopy cover (`gate_test/gate_leaf_cover10`, `params_s7_gate_leaf_cover.yaml` = the leaf step + `cover_k` 0.5,
`fl_min` 0.05; 2001-2010 from the s7 2000 restart; `compare_vs_gate_leaf_2006-2010.txt`).** F77 sets the lower cover
fl = LAI (clipped to [0.25, 0.975]), so any grass layer with LAI > 1 covers the whole cell; cover_k 0.5 is LPJ's foliar
projective cover, 1 - exp(-0.5 LAI) (Sitch et al. 2003), and the floor drops from 25 % to 5 % as the tree floor did
(section 47). Against the leaf step, 2006-2010: **GPP 189.0 -> 175.9 PgC yr-1 (1.73 -> 1.61 x FLUXCOM)**; tropics 1.48 ->
1.36, 30-60 N 2.29 -> 2.17, north of 60 N 3.80 -> 3.71; excess over FLUXCOM 80 -> 67 PgC (forest 21.5 -> 18.3, mixed
28.2 -> 23.4, grassland 7.5 -> 6.2, shrubland 5.8 -> 5.4, bare 3.3 -> 2.6, cropland 13.5 -> 10.8); ET unchanged (1.05 x
the ensemble); peak LAI 2.61 -> 2.42 x MODIS; lower-canopy share of NPP 0.41 -> 0.35 (0.15 -> 0.11 in observed forest);
tree cover 0.519 -> 0.531 (phantom 6.3 -> 6.8 %: trees gain a little where the grass thinned). The largest single step so
far; the steps since s7 (2006-2010): 193.9 (s7) -> 191.7 (gate) -> 189.0 (leaf) -> 175.9 (cover).

**The towers' own leaf area (`verify/calibration/lai_tower_pixel.py`, `logs/lai_tower_pixel.csv`; GLASS V61 8-day
climatological peak 2001-2010 at the 0.05-degree pixel holding each of 355 towers).** Pixel / 0.5-degree cell 1.08
(median; 68 % of towers above 1; DBF 1.20, EBF 1.24, DNF 1.41, ENF 1.10, MF 1.08, GRA 1.02, SAV 1.01, OSH 0.88): towers
sit in somewhat leafier spots than their cells, mostly in deciduous and tropical forests. Pixel / the leaf area each S2
fit held 1.22 (1.26 at the 325 towers held at the Collection-5 MODIS cell maximum, 0.76 at the 29 held at the site
attribute). This corrects section 48's reading: plot-versus-cell leaf area explains only ~7 % (1.26^0.29) of the 1.3x
GPP per leaf area the model carries, not most of it; the rest of the tower / FLUXCOM gap (1.43) is FLUXCOM's documented
low bias, plot productivity beyond leaf area, or both, which these data cannot separate. What it does supply is a
physically grounded capacity correction: refitting capacity at the tower pixel's leaf area (+22 % over what was held)
would lower GPP per leaf by ~7 %, ~10 % through the leaf-GPP feedback -- to be done per type on the 3.1 set.

**Capacity from the tower pixel's leaf area, approximate test (`gate_test/glc_cap10`, `params_s7_glc_cap.yaml`,
`diagnostics/capacity_pixel.py`; `compare_vs_cover_2006-2010.txt`).** vmax x (pixel / held)^(-0.29/0.75), median per
PFT: tropical BE x0.98, tropical BD 0.94, warm-temperate BE 0.91, temperate conifer 0.93, temperate BD 0.87, boreal
conifer 0.94, larch 0.81, evergreen shrub 0.95, C4 0.93, C3 0.90 (boreal BD and deciduous shrub unchanged). Against the
cover run: **GPP 175.9 -> 171.2 PgC yr-1 (1.61 -> 1.57 x FLUXCOM)**; tropics 1.36 -> 1.34, 30-60 N 2.17 -> 2.09, north of
60 N 3.71 -> 3.56; excess 67 -> 62 PgC; ET, tree cover unchanged. The estimate of ~10 % was again too high (the leaf
step: -10 % estimated, -1.4 % measured): the effective dlnGPP/dlnVmax in this state is ~0.37, half of section 4's 0.75,
and the leaf-GPP amplification hardly shows. Only measured effects are reported from here. Where the 62 PgC stand:
north of 60 N 3.56x (tundra and boreal: the pooled lower-canopy phenology of section 40, which the 3.1 tundra types
replace), 30-60 N 2.09x (the season: November-March 21 % of the GPP against 7 %), shrubland 4.86x, cropland 1.58x (the
crop model), tropics 1.34x (within FLUXCOM's documented tropical low bias; COS-based estimates are 25-30 % higher).
Since s7 (2006-2010): 193.9 -> 191.7 (gate) -> 189.0 (leaf) -> 175.9 (cover) -> 171.2 (capacity), -12 %.

**The cover rule at the towers (`gpp_chain/cover_{f77cover,lpjcover}`; the chain's V and S arms at the 62 calibration
towers, 2009-2010, params_s7_gate_leaf vs params_s7_gate_leaf_cover).** The site harness recomputes fl each year
(biomass_update), so unlike the gate and the leaf step the cover rule does reach the tower runs (an earlier statement
that none of the three fixes does was wrong for this one). Site model on tower weather, LPJ / F77 cover: forests 1.00
(no grass layer at forest towers); grassland 0.96 (dense European meadows 0.99, the arid US and Australian grasslands
0.80-0.91), shrubland 0.87, savanna 0.90, woody savanna 0.91; open classes together 0.93. Site / tower, F77 -> LPJ:
shrubland 1.26 -> 1.11 (closer), grassland 1.05 -> 1.04, savanna 0.80 -> 0.70 and woody savanna 1.01 -> 0.86 (further
below), open classes 1.02 -> 0.95, all towers 0.89 -> 0.87. The rule removes 9-20 % of GPP at sparse towers and nothing
in forests and dense grasslands.

**Decision (Hangkai, 2026-10-19): adopt the LPJ cover in IBIS 3.1 and refit the open types' capacity under it.** The
3.1 base is the gate + leaf step + LPJ cover (`params_s7_gate_leaf_cover.yaml`; the approximate capacity test of
`params_s7_glc_cap.yaml` is not adopted -- capacity comes from the refits). The refit (`$S/t31_stage4.sh`, queued after
the crop stage, 8 slots): the GPP/ET fits of the 148 open towers (GRA 86, OSH 34, SAV 15, WSA 8, CSH 5) again, same
fitted set as S2, base `params_s7_crop_lpjcover.yaml` (= s7 + cover_k 0.5, fl_min 0.05), warm-started from the S2
best.json, at the tower's GLASS 0.05-degree pixel peak leaf area (`--lai`, 147 towers; logs/lai_tower_pixel.csv), 48
evaluations, outputs `site_fit/t31_open_lpj`. The first 3.1 global run uses the open types' S2 capacity until the
refit lands; the NEE respiration fits of the open towers were made at the S2 GPP and may want a re-check after it.

**The fits were I/O-bound (2026-10-19/20).** The site harness read GSWP3 through the global reader, which decompresses
the whole 360 x 720 grid of each of the nine variables for every month of every evaluation to keep one cell (~17 GB
per five-year evaluation; the grid-cell harness had solved this with `fit_carbon.CachedForcing` in September, the site
harness never used it). `gswp3_cell_cache.py` extracts the campaign's 307 cells for 1996-2010 once (55 MB,
`inputs/sites/towers31/gswp3_cache/all_towers_ready.npz`); `tower_forcing.CellCacheReader` serves it with the reader's
contract (and falls back to the global files for a month or cell it lacks); site_fit uses the cache when the file exists
(`IBIS_GSWP3_LIVE` forces the files, `IBIS_GSWP3_CACHE` names another cache). Same values, same baseline loss (GL-Dsk
14.6346 both ways). NEE fits 68 -> 20 min (median), memory per fit 18.4 -> 15.3 GB. The reading of "the CPUs are
saturated" (10-19) was decompression, not model compute.

**A second machine (eridu, the lab's Mac Studio: M2 Ultra, 16P + 8E cores, 128 GB; Hangkai 10-19).** A mirror of the
inputs the harness needs under ~/nas2 (code with the NAS prefix rewritten, tower files, S2 best.json, the restart, the
cell cache; 2 GB), a pinned Python 3.12 / JAX 0.11 environment, no core pinning, 4-5 fits at a time under caffeinate.
Validated on GL-Dsk (baseline 14.6347 vs 14.6346 here, float32 noise). It ran the 147 open-type refits in 3.5 h
(22:20-01:51, 0 failed; memory peak 17.9 GB per fit, >= 61 GB free) and then the 46 crop fits under the LPJ cover
(Hangkai 10-20: "yes"; `params_s7_crop_lpjcover.yaml`, 5 at a time). Results are copied to the NAS with rsync
--inplace --no-perms --no-times: the CIFS share refuses rsync's temp files, which left the first pull loop writing
empty folders all night (caught at 02:00, recopied and verified). Refit vs S2 capacity (vmax ratio, median): GRA 1.00,
CSH 1.01, OSH 1.02, WSA 1.06, SAV 1.09 -- the pixel leaf area and the LPJ cover roughly cancel at most open towers;
savannas gain capacity under the thinner cover.

**Crop fits under the LPJ cover: bound-pinned, not adopted (2026-10-20 03:10; `logs/crop_fit_report_t31.csv`,
`params_31_base_crop.yaml` as the record).** The 46 crop towers refitted under `params_s7_crop_lpjcover.yaml` (same
protocol as section 41, leaf area the MODIS cell maximum): C3 median 0.30 with 25 of 42 towers at the floor 0.3 (s7: 0.33
from 16 towers; the fallow season the managed grass does not have), C4 3.00 with all four maize towers at the cap and
still 0.58-0.83 x their GPP (s7: 1.16): under the thinner cover at a crop LAI of ~1.7 the capacity cannot reach maize.
A section with both medians at their bounds would raise the global cropland capacity 0.49 -> 0.84 x grass, so the first
3.1 run keeps s7's crop scales (0.329 / 1.158, already run under the LPJ cover in the cover test: cropland 1.65 x
FLUXCOM). Next for crops, if wanted before IBIScrop: the refit at the towers' GLASS pixel leaf area (Corn Belt pixels
3-4 against the cell maximum 1.7), as done for the open types.

