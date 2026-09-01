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

Yuan et al. (2014), PLOS One 9(11), e110407, validating IBIS against 62 eddy
covariance sites:

> "Global mean GPP estimated by the IBIS model was **107.50 +- 1.37 Pg C
> year-1** (mean value +- standard deviation) across the vegetated area for the
> period 2000-2006"

and describes IBIS as having *"underestimated GPP for the majority of PFTs"*.
Our 219.7 PgC/yr, or 185.0 on FLUXCOM-matched cells, is roughly **2x** that.

**Read this as suggestive, not conclusive.** Their run is 2000-2006 under a
different configuration; ours is a 300-year dynamic-vegetation equilibrium
spin-up. Spin-up length and whether vegetation is prognostic or prescribed both
move GPP substantially, so this is not a like-for-like comparison. What it does
say is that IBIS is *capable* of a realistic global GPP, so a 1.6-1.8x
overestimate is not simply what the model does.

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

2. **What remains.** With the formula, `vmax`, the temperature reference,
   turnover and allocation all matching F77, and `specla` already biased
   downward, the excess has to enter through the LAI/NPP feedback itself
   (`GPP -> NPP -> cbiol -> LAI -> GPP`) or through the canopy fractions
   `fu`/`fl` that multiply it. Those are the untested terms.
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
