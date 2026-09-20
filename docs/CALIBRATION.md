# Calibration

Started 2026-09-09, after the fire and methane modules were re-validated on the
corrected carbon cycle. Order set by Hangkai: carbon, then ET, then the smaller
parts. Methane is evaluated against FLUXNET-CH4 (Chenchen's processing).


## State of the calibration (updated 2026-09-20 12:20; joint carbon + ET fit validated; closing protocol on it running)

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
4. **Observation caveats** (15-16): the ET products spread 38 % per cell
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
2010 state: the ET stage's start, unless (b4) supersedes it.

