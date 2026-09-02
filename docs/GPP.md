# The GPP overestimate

**Status 2026-08-31: diagnosed, one dead parameter fixed, not yet solved.**

## The finding

The 300-year cold-start reference run (`spinup_cold_v10`) gives an equilibrium
carbon cycle scaled up by a single nearly uniform factor:

| flux | IBIS | published | ratio |
|---|---|---|---|
| GPP | 219.7 PgC/yr | 120-130 | 1.76x |
| NPP (total) | 98.7 PgC/yr | 55-60 | 1.72x |
| Rh | 89.6 PgC/yr | 49-58 | 1.67x |

The ratios BETWEEN fluxes are right -- NPP/GPP = 0.450 (observed 0.45-0.50),
Rh/NPP = 0.907, NEE = +1.6 PgC/yr against ~0 expected. Allocation, autotrophic
respiration and decomposition are all doing the right thing with too much
carbon, which is the signature of one upstream error rather than several.

## Where the bias lives

Against FLUXCOM (`verify/gpp_vs_fluxcom.py`; FLUXCOM sanity-checks at
114.5 PgC/yr on matched cells against its published ~120), model GPP is
**1.58-1.62x** high. The bias is **strongly aridity-dependent**:

| precip mm/yr | model LAI | GPP bias |
|---|---|---|
| 0-200 | 0.12 | 1.36x |
| 200-400 | 1.26 | **2.00x** |
| 400-700 | 2.91 | 1.93x |
| 700-1000 | 5.00 | 1.86x |
| 1000-1500 | 6.72 | 1.55x |
| 1500-2500 | 8.04 | 1.40x |
| 2500+ | **8.74** | 1.24x |

Worst where it is dry, closest where it is wet. Two things follow:

* **This is not a uniform photosynthesis scaling.** A global `vmax` error would
  bias all bands equally.
* **LAI is inflated in step with NPP, not independently.** See below -- an
  earlier draft of this document claimed LAI was 3.4x too high by comparing
  our area-weighted vegetated-cell mean of 4.11 against a MODIS *global land*
  mean of ~1.2, which includes barren ground. That comparison was not
  like-for-like and is withdrawn.

The aridity signature is the documented LSM failure mode: models overestimate
GPP in seasonally dry systems because soil-moisture stress is under-represented
(Green et al. 2025, GCB; Harper et al. 2021, GMD, for JULES).

## Published IBIS reports about half our GPP

Liu, D., Cai, W., Xia, J., Dong, W., Zhou, G., Chen, Y., Zhang, H. & Yuan, W.
(2014), *PLoS ONE* 9(11), e110407, doi:10.1371/journal.pone.0110407 --
validating IBIS against 62 eddy covariance sites:

> "Global mean GPP estimated by the IBIS model was **107.50 +- 1.37 Pg C
> year-1** (mean value +- standard deviation) across the vegetated area for the
> period 2000-2006"

and describes IBIS as having *"underestimated GPP for the majority of PFTs"*.
Our 219.7 PgC/yr, or 185.0 on FLUXCOM-matched cells, is roughly **2x** that.

**And they prescribed LAI.** From the methods, verbatim:

> "To decrease model uncertainty, we used the satellite-based leaf area index
> (LAI) from the Moderate Resolution Imaging Spectroradiometer (MODIS) as a
> model input."

So their 107.5 PgC/yr is IBIS running on **exactly the MODIS LAI field our
model overshoots by 2.7x**. IBIS did not compute LAI in that study at all.

That makes the comparison decisive rather than suggestive:

| configuration | LAI | global GPP |
|---|---|---|
| Liu et al. 2014 | MODIS, prescribed | 107.5 PgC/yr |
| this port, 300-yr spin-up | prognostic, equilibrated | 219.7 PgC/yr |
| our LAI vs that same MODIS field | -- | **2.7x** |

**The entire difference between published IBIS and ours is the leaf area.**
Photosynthesis, allocation and turnover are all faithful; what the port does
that the published evaluation never exercised is let IBIS choose its own LAI.

Note also that the paper reports IBIS performing worst for **evergreen
broadleaf forests and savannas**, attributing it to *"inadequate soil moisture
simulation and phenology algorithms"* -- independent support for the semi-arid
water-stress residual characterised above.

The paper reports no LAI bias; it mentions LAI only as an input. The 2.7x
figure is this project's, from `verify/lai_vs_modis.py`.

## It is not a port defect

Checked against F77 rather than assumed:

* `stats.f:118` -- `tgpp(i,1) = frac(i,1) * agcub(i) * lai(i,2) * fu(i)`.
  Identical to the port's `ag_ub * frac * lai_u * fu`.
* `physiology.f:989-1016` -- `stressfac = -5.0`, `znorm = 1 - exp(stressfac)`,
  `awc = (wsoi*(1-wisoi) - swilt)/(sfield - swilt)`,
  `zwilt = (1 - exp(stressfac*awc))/znorm`, root-weighted. The port's
  `drystress` matches line for line, including `stressfac = -5.0`.

So the overestimate is **inherited IBIS behaviour**, not something the port
introduced.

## What was fixed: stressfac was dead

`timestep_body` called `drystress()` without passing `stressfac`, so the
hardcoded `-5.0` always won and `StomataParams.stressfac` -- declared, loaded
from YAML, settable by any user -- was populated and never read. Proof:
`stressfac = -5` and `-10` produced byte-identical output.

The implication is worse than a dead knob. **The CMA-ES calibration that
produced the current `vmax` and `specla` would have measured `stressfac` as
having zero effect** and compensated through other parameters. Several of those
carry manual caps in the source (`vmax` 121.8 -> 80, 4.5 -> 10, 11 -> 15;
`specla` 39.4 -> 35, 33.4 -> 30), which is what compensating for a missing
degree of freedom looks like.

**The sign is counter-intuitive and easy to get backwards** --
`zwilt = (1-exp(sf*awc))/(1-exp(sf))`, so a MORE negative `stressfac`
saturates faster and applies LESS stress. At `awc = 0.3` the shipped `-5.0`
applies only 22% stress; `-2.0` gives 48%, `-1.0` gives 59%. There is a test
pinning this.

## What did NOT work: stressfac alone

1-year sensitivity from the v10 restart, LAI held fixed:

| precip | sf=-5 (base) | sf=-3 | sf=-2 | sf=-1 |
|---|---|---|---|---|
| 0-200 | 1.36x | 1.26x | 1.19x | **1.10x** |
| 200-400 | 2.00x | 1.91x | 1.84x | 1.76x |
| 400-700 | 1.93x | 1.87x | 1.83x | 1.77x |
| 1000-1500 | 1.55x | 1.53x | 1.52x | 1.49x |
| 2500+ | 1.24x | 1.24x | 1.23x | 1.23x |
| **global** | **1.58x** | 1.56x | 1.53x | **1.51x** |

Driving `stressfac` to the strongest defensible value moves global GPP only
1.58x -> 1.51x. It genuinely fixes the driest bin (1.36 -> 1.10) and does
nothing in the wettest. **Water stress alone cannot close a 1.6x gap.**

### ...and it does not compound at equilibrium either

The sweep above holds LAI fixed. Stronger stress should also shrink LAI, which
would compound, so a 60-year run at `stressfac = -1` was run from the v10
restart with LAI free to adjust. It does not:

| precip mm/yr | base (sf=-5, 300 yr) | sf=-1, 60 yr |
|---|---|---|
| 0-200 | 0.99x | **0.85x** |
| 200-400 | 2.51x | 2.28x |
| 400-700 | 2.09x | 1.98x |
| 700-1000 | 1.87x | 1.78x |
| 1000-1500 | 1.66x | 1.62x |
| 1500-2500 | 1.34x | 1.31x |
| 2500+ | 1.21x | 1.19x |
| **global** | **1.61x** | **1.56x** |

A 3% improvement after 60 years, no better than the direct 1-year response. And
it **overshoots the driest bin**: 0-200 mm/yr is already right at 0.99x and
`stressfac = -1` drives it to 0.85x, 15% too LOW. So `stressfac` should be left
at its F77 value of -5.0; the fix was to make it live, not to retune it.

**The cheap lever is exhausted.** Water stress cannot close a 1.6x gap.

## Open: what to do next
1. **The parameter set is NOT the obvious culprit** -- checked, 2026-08-31.
   IBIS's own defaults live in `fortran/params.can` (found one directory above
   the `ibis_2.6b4/` tree, which lacks it):

   | parameter | ours | IBIS params.can | verdict |
   |---|---|---|---|
   | `vmax_pft` | 65, 65, 40, 30, 30, 25, 30, 30, 27.5, 27.5, 15, 25 (e-6) | identical | **already IBIS's** |
   | `tauleaf` | 1.01, 1.0, 1.0, 2.0, 1.0, 2.5, ... | identical | match |
   | `aleaf`/`aroot`/`awood` | 0.30/0.20/0.50 ... | identical | match |
   | `specla` | 13.0, 21.6, 30.0, 10.8, 35.0, 6.6 ... | 25, 25, 25, 12.5, 25, 12.5 ... | ours are LOWER |

   Two things follow. **The CMA-ES `vmax` never reached the physics**:
   `PFTParams.vmax_pft` is read only by `scripts/calibrate_*.py`, while
   `stomata` consumes `StomataParams.vmax_pft`, whose default is IBIS's set.
   The model has been running IBIS's own vmax all along. And **`specla` points
   the wrong way** -- ours are roughly half IBIS's for several PFTs, and since
   `plai = cbiol * specla`, reverting to IBIS defaults would raise LAI and make
   the bias worse. The tuning was already compensating in the right direction.

   Also verified: `vmax_pft` is defined at **15 C** (`params.can` line 92), and
   the port's `tempvm` references `3.47e-03 = 1/288.18 K`, so there is no
   reference-temperature mix-up -- a hypothesis that would have explained a
   uniform 2x neatly, and does not hold.

2. **`fu`/`fl` and the canopy light scaling also match F77.** `dynaveg`'s
   `fu = (1-exp(-wood))/(1-exp(-woodnorm))`, the `(1 - disturbf - disturbo)`
   term and the `max(0.25, min(0.975, .))` clamp all match `vegetation.f:525-530`.
   `canopy_scaling` reproduces `physiology.f:418`'s
   `(1 - exp(-extpar*xai)) / (extpar*lai)` exactly, so GPP does saturate with
   LAI rather than growing linearly. Every component checked is faithful.

## The decomposition that matters

Dividing the GPP bias by the LAI bias separates "too much leaf" from "too much
photosynthesis per leaf":

| precip mm/yr | LAI bias | GPP bias | GPP per unit LAI |
|---|---|---|---|
| 200-400 | 1.57x | 2.00x | **1.27x** |
| 400-700 | 1.62x | 1.93x | 1.19x |
| 700-1000 | 1.67x | 1.86x | 1.11x |
| 1000-1500 | 1.68x | 1.55x | 0.92x |
| 1500-2500 | 1.61x | 1.34x | 0.83x |
| 2500+ | 1.46x | 1.21x | **0.83x** |

**Photosynthesis per unit leaf area is roughly right** -- slightly low in the
wet tropics, ~1.2-1.3x high in semi-arid, which is the water-stress residual
already characterised above. **LAI is uniformly 1.46-1.68x high in every
band.** If LAI were right, GPP would be close to right.

So this is a vegetation-carbon problem, not a photosynthesis problem.

## LAI against real MODIS data: 2.7x too high

**Unblocked 2026-08-31.** `verify/lai_vs_modis.py` builds a grid-cell total LAI
from the MODIS PFT LAI that ships with CLM's 0.5 deg surface dataset (Peter
Lawrence, NCAR), weighted by the matching surfdata's cover fractions:

    LAI_total = sum_i LAI_i * PCT_NAT_PFT_i/100 * PCT_NATVEG/100

Both files are on disk under `ELM Edge Effects/inputdata/`, already on our grid.

| | model | MODIS | ratio |
|---|---|---|---|
| **global** | **4.53** | **1.69** | **2.69x** |
| 200-400 mm/yr | 1.68 | 0.77 | 2.18x |
| 700-1000 | 5.26 | 1.71 | 3.08x |
| 1000-1500 | 6.94 | 2.20 | 3.15x |
| 2500+ | 8.81 | 3.48 | 2.53x |
| 60-90N | 1.44 | 1.10 | 1.31x |
| 30-60N | 4.75 | 1.55 | 3.08x |

Restricted to cells >=90% natural vegetation. Robust to that restriction:
2.69x at >=90%, 2.73x at >=50%, 2.75x over all land, so cropland (which the
MODIS total scores as zero LAI and IBIS vegetates) is not driving it.

**This supersedes the 1.46-1.68x figure above**, which came from biome-typical
LAI values typed into a diagnostic script. The real bias is substantially
larger.

### Three comparability traps, all of which had to be fixed first

1. **Seasonality.** IBIS carries leaf drop in `frac`
   (`frac = plai * dropu / avglaiu`, `vegetation.f:150`), NOT in `lai`, so
   `laiu`/`lail` are FULL-LEAF values. They must be compared against MODIS
   annual *maximum*, not December or annual mean. A December comparison
   reported 8.6x at 30-60N purely from deciduous dormancy.
2. **Grid orientation.** CLM surface datasets run SOUTH->NORTH
   (`LATIXY` -89.75 -> 89.75); IBIS output runs NORTH->SOUTH. Unflipped, the
   comparison is latitude-mirrored -- it matched Arctic model cells against
   Antarctic MODIS cells and reported MODIS LAI = 0.00 at 60-90N.
3. **Cropland.** Handled by the `--min-natveg` filter above.

### The MODIS product validates

Checked before relying on it, since the whole diagnosis now rests on it:

| check | result |
|---|---|
| Amazon box (0-10S, 55-70W) | LAI **5.44**, against an observed ~5-6 |
| its PCT_NATVEG / bare soil | 97% / 0% -- no dilution there |
| global max LAI per grid cell | 6.30 -- physically sensible ceiling |
| bare-soil fraction over land | 27% |

### What it implies

GPP is 1.6x high while LAI is 2.7x high, so GPP per unit leaf area is about
**0.6x** -- the model is producing *less* photosynthesis per leaf than
observations imply. The canopy light saturation is working; that is precisely
why GPP is "only" 1.6x when LAI is 2.7x.

**The target is now unambiguous: LAI, not photosynthesis.**

Verified directly from the year-2200 restart:

* `plai = cbiol * specla` holds **exactly** (max relative error 0.00e+00), so
  the conversion is not the bug.
* mean leaf carbon `cbiol` = **0.147 kgC/m2**, mean total `plai` = 2.95
  (unweighted; 4.1-4.5 area-weighted, which is what meets MODIS).
* implied mean `specla` = **20.0 m2/kgC**.

### The fork, resolved: allocation is self-consistent

`verify/leaf_carbon_balance.py` runs a year forward from the restart and reads
`summary.aynpp` before `dynaveg` consumes it (`run_global` resets the
accumulators at line 700 and writes the restart at line 724, so a restart
always shows zero).

| quantity | value |
|---|---|
| annual `aynpp` summed over PFTs | 98.0 PgC/yr |
| `cbiol` actual | 0.1291 kgC/m2 |
| `cbiol` predicted = `aleaf * aynpp * tauleaf` | 0.1301 kgC/m2 |
| **ratio actual / predicted** | **0.992** |

**The leaf pool is exactly what allocation and turnover predict.** Nothing is
feeding or retaining leaf carbon beyond what `aleaf` and `tauleaf` allow, so
there is no upstream bug for a `specla` change to bury.

### Which means the bias is IBIS's dynamic-vegetation attractor

Putting the chain together: photosynthesis per unit leaf is right, the leaf
pool follows allocation exactly, `aleaf`/`tauleaf` match F77, and `specla` is
already below F77's values. The whole system is internally consistent -- it
simply equilibrates to a state 2.7x too leafy, and GPP/NPP/Rh follow.

That also reconciles the published IBIS figure, and more directly than a
spin-up argument would: **Liu et al. prescribed MODIS LAI** (see above), so
their run reports GPP at observed leaf area by construction. Ours lets LAI
reach IBIS's own attractor, which overshoots by 2.7x. Note too that with F77's
`specla` (25 vs our 13-20) IBIS's equilibrium LAI would be *higher* than ours,
not lower -- so this is inherited IBIS behaviour that the published evaluation
simply never exercised.

### So calibrating specla is now defensible -- with one caveat

`specla` is the one free parameter mapping leaf carbon to leaf area, and we now
have a validated per-cell LAI target. Because the leaf pool has been shown
self-consistent, fitting `specla` is setting a parameter against data rather
than hiding a defect.

### The projection: correcting LAI should also correct GPP

Worth computing before spending compute, since it decides whether this is even
the right lever. GPP goes as `(1 - exp(-k*LAI))/k` through `canopy_scaling`, so
moving LAI from the model's value to MODIS's scales GPP by
`(1-exp(-k*L_obs))/(1-exp(-k*L_now))`:

| band | LAI now | LAI obs | GPP now | GPP after (k=0.4) | (k=0.6) | (k=0.8) |
|---|---|---|---|---|---|---|
| 200-400 | 1.68 | 0.77 | 2.00x | 1.08x | 1.17x | 1.24x |
| 400-700 | 3.35 | 1.45 | 1.93x | 1.15x | 1.29x | 1.42x |
| 700-1000 | 5.26 | 1.71 | 1.86x | 1.05x | 1.25x | 1.41x |
| 1000-1500 | 6.94 | 2.20 | 1.55x | 0.97x | 1.15x | 1.29x |
| 1500-2500 | 8.11 | 2.72 | 1.34x | 0.92x | 1.09x | 1.19x |
| 2500+ | 8.81 | 3.48 | 1.21x | 0.94x | 1.07x | 1.14x |
| **mean** | | | **1.65x** | **1.02x** | **1.17x** | **1.28x** |

**This projection was WRONG and is retained only as a record.** It was tested
directly (below) rather than by measuring `extpar`, and the measured GPP
response to an LAI change is far smaller than any column above.

### Tested: specla x0.63, 60-year spin-up

| | LAI vs MODIS | GPP vs FLUXCOM |
|---|---|---|
| baseline | 2.69x | 1.61x |
| `specla` x 0.63 | **1.67x** | **1.55x** |

Two results, one expected and one not:

* **LAI responds almost exactly linearly to `specla`** -- 2.69 x 0.63 = 1.69
  predicted, 1.67 measured. There is NO feedback amplification, contrary to the
  caveat recorded above. `specla` is a clean, proportional lever on LAI.
* **GPP barely moves** -- 1.61x to 1.55x for a 37% cut in leaf area.

The second kills the projection. The canopy is **light-saturated**, so GPP is
nearly independent of LAI: `GPP ~ (1 - exp(-k(L+S)))/k`, and the measured
response implies an effective `k` near 1.0, at which `k*L = 4.5` and the
exponential is spent. Correcting LAI the whole way to MODIS would take GPP to
roughly **1.3x**, not 1.0x.

**So LAI and GPP are not one lever after all.** Fixing LAI fixes LAI, and buys
perhaps a fifth of the GPP error. With GPP saturated, `GPP ~ ag * fu / k`, so
the remaining bias sits in per-leaf assimilation `ag`, the canopy fraction `fu`
(which has a hard 0.25 floor), or the extinction `k` -- not in leaf area.

A uniform `specla` scale also breaks the spatial pattern: 60-90N drops to 0.37x
(now too LOW) while 60-30S stays at 3.40x. Any real `specla` calibration has to
be per-PFT, not a single factor.

**The caveat is the feedback.** Lowering `specla` lowers LAI, which lowers GPP,
NPP, `cbiol` and LAI again. The equilibrium response is therefore LARGER than
the direct one, so the naive scaling (2.68 -> 1.69 needs specla x0.63, mean
20 -> 12.6 m2/kgC) overshoots. The calibration must be run against full
spin-ups, not one-year evaluations -- which is exactly the mistake the previous
CMA-ES setup could have made.

## The decisive test: prescribe MODIS LAI

`physics.prescribe_lai` sets LAI to the observed field exactly
(`ibis/physics/prescribed_lai.py`), which is the configuration Liu et al. used.
10-year run from the v10 restart.

| run | LAI vs MODIS | GPP vs FLUXCOM |
|---|---|---|
| baseline | 2.69x | 1.63x |
| `specla` x 0.63 | 1.67x | 1.56x |
| **prescribed MODIS LAI** | **1.00x** | **1.46x** |

The prescription lands exactly (ratio 1.00). **With leaf area exactly right,
GPP is still 46% too high.**

### Decomposition

| | |
|---|---|
| total GPP bias | 1.63x |
| with LAI set exactly to MODIS | 1.46x |
| **attributable to excess leaf area** | **1.12x -- about 10% of the error** |
| residual, per unit leaf area | 1.46x |

**Leaf area is not the main cause.** It accounts for roughly a tenth of the
GPP bias; the rest is per-leaf assimilation. This overturns the conclusion
recorded above that "the target is unambiguously LAI, not photosynthesis" --
that inference came from dividing bias ratios, and the direct experiment
contradicts it. LAI at 2.7x is still a real defect worth fixing on its own
terms; it is simply not what makes GPP high.

### The residual is aridity-structured

With LAI exactly correct, by precipitation band:

| precip mm/yr | GPP bias |
|---|---|
| 0-200 | 0.81x |
| **200-400** | **1.94x** |
| 400-700 | 1.74x |
| 700-1000 | 1.62x |
| 1000-1500 | 1.53x |
| 2500+ | 1.21x |

The same semi-arid signature the aridity analysis found at the very start,
undiminished by fixing leaf area. The driest band is now slightly LOW (0.81x).
So the dominant remaining cause is water limitation on assimilation -- and
`stressfac`, the obvious knob, was already shown to move the global figure only
1.58x -> 1.51x at its strongest defensible value.

### Same LAI, same model, 1.55x apart

Liu et al. get 107.5 PgC/yr with prescribed MODIS LAI. This port gets
**166.9 PgC/yr with the same prescribed MODIS LAI** -- a factor of **1.55**
that leaf area, `vmax`, the GPP formula, the temperature reference, the canopy
scaling, `fu`/`fl`, turnover and allocation have all been individually verified
NOT to explain.

What has not been ruled out, and is where the next work belongs:

1. **Forcing.** They used their own meteorology; we use GSWP3. Radiation and
   VPD differences propagate straight into assimilation.
2. **Domain.** Their "vegetated area" versus our 165 Mkm2 land mask, which is
   already known to exceed Earth's ice-free land by ~27%.
3. **Soil water supply.** `awc = (wsoi(1-wisoi) - swilt)/(sfield - swilt)`
   depends on the soil hydraulic properties from `soita.nc`. If the profile
   holds too much plant-available water, `stressfac` cannot bite however it is
   shaped -- which would explain why retuning it barely moved anything.

Option 3 is the cheapest to check and the best fit to the aridity signature.

### Preliminary: option 3 looks WEAK

Computing `awc` from the year-2200 restart exactly as `drystress` does, and the
`stresstu` it implies at `stressfac = -5`:

| precip mm/yr | root-weighted `awc` | `stresstu` |
|---|---|---|
| 0-200 | 0.023 | 0.076 |
| **200-400** | **0.053** | **0.135** |
| 400-700 | 0.169 | 0.340 |
| 700-1000 | 0.437 | 0.670 |
| 2500+ | 0.881 | 0.959 |

Water stress is already severe where the GPP bias is worst -- `vmax` cut to
13.5% of nominal in the 200-400 mm band, which still runs 1.94x high with LAI
prescribed. The soil is not holding too much plant-available water, so the
"stressfac cannot bite" explanation does not survive, and this also explains why
retuning `stressfac` moved so little: the stress term is already near its floor
there.

**Treat as preliminary.** `wsoi` here is a year-END snapshot, not an annual
mean, while GPP is a growing-season integral -- the same snapshot trap that has
produced several wrong answers in this project. Settling it needs `awc`
accumulated over a simulated year, which is a small addition to the daily
diagnostics rather than a new run.

That leaves **forcing**, **domain**, and -- from the literature search below --
**canopy structure** as candidates for the 1.55x gap against published IBIS at
identical prescribed LAI.

## The leading candidate: IBIS is a one-big-leaf model

**IBIS does not separate sunlit from shaded leaves.** `canopy_scaling` computes
a single scale factor `(1 - exp(-k(L+S)))/(kL)` and applies it to one `ag` per
canopy layer. The `sunlit` variable in `radiation.py` is a daylight mask (is the
sun above the horizon), not a leaf partition.

This is a documented, quantified structural bias:

> "About half of state-of-the-art earth system models use a 'two-big-leaf'
> scheme that partitions canopies into direct and diffusely illuminated
> fractions **to reduce high bias of GPP simulated by one-big-leaf models**."

> "A big-leaf model assumes a canopy to be a single big leaf without a shaded
> leaf component... **It will overestimate canopy-level GPP when the shaded
> leaf contribution is less than the average (i.e., low LAI)**."

The mechanism is Jensen's inequality: leaf photosynthesis saturates with light,
so applying the canopy-MEAN light to a single leaf overestimates the integral
over a canopy with a light gradient. Reported magnitude at low-GPP sites is a
mean prediction error of **1.09 +- 0.12 gC m-2 d-1** -- against 1.0-2.2
gC m-2 d-1 of actual GPP in our driest vegetated bands, i.e. a 50-100% error,
which is the size of ours.

### It predicts our residual almost exactly

Big-leaf theory says the overestimate is largest at LOW LAI. With LAI
prescribed to MODIS, so leaf area is exactly right and cannot confound:

| MODIS LAI | GPP bias |
|---|---|
| 0.77 | 1.94x |
| 1.45 | 1.74x |
| 1.71 | 1.62x |
| 2.20 | 1.53x |
| 2.72 | 1.29x |
| 3.48 | 1.21x |

**r(LAI, bias) = -0.985**, monotonic. (The 0-200 mm band at LAI 0.11 and 0.81x
is excluded -- essentially unvegetated, a different regime.)

### The honest caveat

LAI covaries with precipitation, temperature and radiation, so a water-stress
explanation produces the same correlation. This is a strong *consistency*
result, not proof. What raises it above the alternatives is that the mechanism
is structural and specific, IBIS's code matches the failing design exactly, the
published error magnitude matches ours, and the competing water-stress
explanation has already been weakened twice -- `stressfac` retuning barely
moved the total, and `stresstu` is already at 0.135 in the worst band.

### REFUTED before implementation -- the sign is backwards

Checked the arithmetic before writing any code. IBIS computes

    agc = A(I_top) * scale,   scale = (1 - exp(-k*L)) / (k*L)

i.e. assimilation at TOP-OF-CANOPY light multiplied by the MEAN LIGHT fraction.
Because `A` saturates with light, that **underestimates** the true canopy
integral `mean_x A(I_top * exp(-k*x))`, and the underestimate grows with LAI:

| LAI | IBIS scheme / true canopy integral |
|---|---|
| 0.5 | 0.91x |
| 1 | 0.84x |
| 2 | 0.73x |
| 4 | 0.61x |
| 8 | 0.54x |

**So the big-leaf scaling is not causing the high bias -- at high LAI it is
masking one.** Replacing it with a two-leaf or quadrature scheme would RAISE
GPP by ~1.2x at low LAI and ~1.5x at high LAI, making the bias worse everywhere
and inverting its gradient.

The `r = -0.985` correlation is real, but its cause is the opposite of what the
big-leaf literature describes: our bias falls with LAI because the scaling
scheme's own underestimate grows with LAI, partially cancelling a roughly
uniform overestimate underneath.

**Which implies the underlying per-leaf overestimate is LARGER than measured.**
Correcting the canopy integration would leave a roughly uniform ~2-2.5x
per-leaf excess rather than the 1.2-1.9x gradient we observe.

Two readings, and the distinction matters:

1. IBIS's `vmax` values are EFFECTIVE parameters, tuned within a scheme that
   under-integrates the canopy, so scheme and parameter are only meaningful
   together. Changing one without the other breaks a calibration that the F77
   model depends on. This is the more likely reading, and it is a reason NOT to
   touch the canopy scheme.
2. The scheme is simply wrong and `vmax` should be re-derived with it. That is
   a research project, not a fix, and it would invalidate every parameter
   inherited from F77.

Either way, the canopy scheme is the wrong place to intervene, and the 1.55x
gap against published IBIS at identical prescribed LAI remains unexplained by
anything internal to the model.

## References (canopy structure)

- Big-leaf vs two-leaf LUE comparisons: https://www.sciencedirect.com/science/article/abs/pii/S0168192321004470
- Two-big-leaf advantages at ecosystem level: https://www.sciencedirect.com/science/article/pii/S0168192322003720
- Bridging big-leaf and two-leaf for MODIS GPP: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023JG007737
- Global sunlit/shaded GPP dataset: https://www.nature.com/articles/s41597-022-01309-2

## Superseded: where it was blocked

At equilibrium `LAI = NPP * aleaf * tauleaf * specla`. `aleaf` and `tauleaf`
match F77 exactly and `specla` is already biased LOW relative to F77, so on
parameters alone our LAI should sit BELOW IBIS's, not 1.6x above it. Nothing in
the checked chain accounts for that.

**The blocker is that the LAI reference values used above are my own
biome-typical estimates, not a dataset.** Given that three separate
apples-to-oranges reference comparisons have already been caught in this
project, the 1.46-1.68x LAI bias should not be acted on until it is confirmed
against a real global LAI product. `/shared/nas2` has only per-PFT forest LAI
(`MODISPFTLAI_0.5x0.5_*.nc`), which needs PFT cover weights we do not have.

**Next step: obtain a global gridded LAI climatology** (MODIS MOD15A2 or
GIMMS/GLASS LAI at 0.5 deg) and repeat the per-band comparison. That either
confirms LAI as the single remaining target or removes it, and it is the one
piece of evidence the diagnosis now turns on.
3. ~~**LAI needs a ceiling.**~~ **Dropped -- LAI is a symptom, not a second
   cause.** Compared per-biome against typical observed values rather than
   against a global land mean, LAI is high by a nearly constant factor:

   | precip mm/yr | model | typical | ratio |
   |---|---|---|---|
   | 200-400 | 1.26 | 0.8 semi-arid shrub | 1.57x |
   | 400-700 | 2.91 | 1.8 grass/savanna | 1.62x |
   | 700-1000 | 5.00 | 3.0 woodland | 1.67x |
   | 1000-1500 | 6.72 | 4.0 temperate forest | 1.68x |
   | 1500-2500 | 8.04 | 5.0 moist forest | 1.61x |
   | 2500+ | 8.74 | 6.0 tropical rainforest | 1.46x |

   That is the NPP inflation (1.72x) passed through
   `LAI = NPP * aleaf * tauleaf * specla`, and `aleaf`/`aroot` match F77
   exactly. F77 itself has only a minimum (`xminlai`), no maximum, so capping
   LAI would both deviate from IBIS and treat the symptom.

## References

- Green, J. K. et al. (2025), *Global Change Biology* 31, e70098. https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.70098
- Harper, A. B. et al. (2021), *Geosci. Model Dev.* 14, 3269. https://gmd.copernicus.org/articles/14/3269/2021/
- FLUXCOM RS+METEO: Jung, M. et al. (2019). Data at `/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc`
