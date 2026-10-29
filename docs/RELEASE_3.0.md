# IBIS 3.0: the calibrated model of 28 October 2026

The short description of the IBIS 3.0 release state: what the model is, which parameter file is the release and where it
came from, what was calibrated against what, how the release run is reproduced and scored, what it scores, and what it
does not do. The long form is the calibration log, [`CALIBRATION.md`](CALIBRATION.md) (74 sections, each stage with its
baseline, lever test and verdict); the section numbers below point there. Naming: this is IBIS 3.0; the run tags and
file names of the October stage (s31*, t31_*, params_31*) are working labels, not a version.

## 1. What IBIS 3.0 is

IBIS 2.6b4 (Foley et al. 1996; Kucharik et al. 2000) rewritten in JAX and verified against the Fortran at machine
precision, with the Fortran's own defects and the port's fixed ([`DEFECTS_FOR_REVIEW.md`](DEFECTS_FOR_REVIEW.md),
CALIBRATION 59-63), and with these additions, each behind a switch that is off by default and bit-identical to the Fortran
when off: a sunlit/shaded two-big-leaf canopy with capacity declining through the canopy and acclimation; phenology as
parameters (photoperiod senescence, a soil-stress drought trigger); forty Köppen-zone plant types that live in the twelve
PFT slots, each square taking the types of its climate zone; a water-limited maximum tree cover; a fire scheme driven by
the length of the fire season with per-type mortality and conserved emissions (CO2, CO, CH4, N2O); a wetland methane
module (one scalar form) on WAD2M inundation with the water table from the model's own soil moisture; cropland as managed
grass on the ESA CCI cropland share with harvest; three thermal bedrock layers under the 4 m column; per-type respiration
and allocation; and the diagnostics the towers measure (ET parts, day and night NEE, fire emissions, methane, harvest).
The global run has 95,333 half-degree land cells; the 310-year protocol takes about seven hours on one RTX 4090.

## 2. The release parameter file and its provenance

The release is `python-port/configs/params/params_31s_soil_strong.yaml`, run as the protocol tagged `s31t` on 28 October
2026 (`verify_outputs/final_s31t/`). Newest change first:

| Step | File | What it added | Log |
|---|---|---|---|
| soil-carbon stage | `params_31s_soil_strong.yaml` | slow- and passive-pool turnover scaled per zone group on the zones' tree types (`som_k_scale`); the stronger member of a two-member bracket run through the full protocol | 72 |
| cover limit and production | `params_31r.yaml` | the water-limited maximum tree cover with a ramp per zone (the stress gate retired); the C/D-zone allocation refit (awood 0.35, the leaf cut going to roots); shorter temperate wood residence (45 / 40 yr); the tropical wood refit at half strength | 70-71 |
| allocation stage, road B | `params_31n.yaml` | leaf area set through the allocation and leaf-mass coefficients, no caps; the cold shrub kept (aleaf 0.30) | 54-69 |
| the type set | carried in the above | 40 climate-zone types, each the median of its towers' fits from the 356-tower campaign; methane bands from the 45 FLUXNET-CH4 towers; cropland from 16 cropland towers | 45-57, 38, 41 |
| 12 October release (`main`) | `params_s7_crop.yaml` | the class-median set of the 62-tower fits, with methane and cropland | 37-43 |

Switches on in the release run: `koppen_types` (with the Köppen mask `koppen_file`), `tree_cover_limit`, `enable_methane`,
`enable_fire_emissions`, cropland (`cropfrac_file`). Organic-soil hydrology is off. The run is cold: no state is inherited
from an earlier run.

## 3. What was calibrated against what

Towers first, then the globe; no global total was ever a term in a loss function.

| Process | Parameters | Data | How | Log |
|---|---|---|---|---|
| photosynthesis, stomata, phenology, respiration, allocation | per type: capacity and acclimation, Ball-Berry, leaf traits, allocation and turnover, respiration | 356 flux towers (FLUXNET2015, ICOS, AmeriFlux, NEON, regional networks): monthly GPP, ET and NEE on the tower's own meteorology, three to five years, one held out | CMA-ES per tower on a bounded objective; each type is the median of its towers; every tower re-scored under the shared set | 34-36, 45, 50-57 |
| wetland methane | rate, Q10, anaerobic depth scale and oxidation scale per latitude band | 45 FLUXNET-CH4 towers, held-out year | per-tower fits, band medians; the bands kept at the final (the decomposition of 73) | 38, 42, 73 |
| cropland | crop capacity factor, harvest | 16 cropland towers | per-tower fits | 41 |
| fire | season, danger, fuel, mortality, damping | GFED5 burned fraction 2001-05 by zone | a zone objective on amount and pattern; the release sits at its optimum | 59, 65-66 |
| tree cover | maximum cover against growing-season water stress, per zone | CLM / MODIS tree cover on natural land | the product's cover as a function of the model's own stress, fitted by zone; phantom and lost forest as the check | 70-71 |
| wood | C/D-zone allocation, temperate residence, tropical turnover | Xu & Saatchi 2021 live biomass by zone group and continent | bounded refit, verified in the full protocol | 67, 70-71 |
| soil carbon | slow/passive turnover per zone group | HWSD2 organic carbon 0-1 m by zone group and latitude band | a two-member bracket through the full protocol; the strong member chosen | 72 |

Everything else (surface exchange, hydrology, snow, the soil thermal column, the CENTURY pools, the two-stream canopy
radiation) is the verified IBIS 2.6b4 physics.

## 4. Reproducing the release run and its scores

```bash
cd ibis-port; export PYTHONPATH=python-port:.
SPINUP_YEARS=200 METHANE=1 KOPPEN_TYPES=1 \
  KOPPEN=<koppen_beck2023_0.5deg.npz> CROPFRAC=<cropfrac_esacci_2000_05deg.npz> \
  bash verify/calibration/final_validation.sh s31t python-port/configs/params/params_31s_soil_strong.yaml
```

Stages: cold spin-up 1701-1900 (1901-1930 climate cycled, 296 ppm, soil carbon accelerated then released), transient
1901-2000 (GSWP3 years, rising CO2, yearly output), hindcast 2001-2010 (monthly output; methane and fire emissions
written). Outputs land under `verify_outputs/final_s31t/{spinup,transient,hindcast}/{out,restart}`; the protocol log
carries the built-in scoring: pools, fluxes and latitude bands against GCB 2024 and GMB 2025, tree cover against the
product, ET against the five-product ensemble, the seasonal cycle, GPP at the 435 tower cells, fire against GFED5
(2001-05 from the 2010 state), the 1901-2010 GPP trend. The chain that follows (`verify/calibration/diagnostics/`):
`gate_compare.py` against the previous run, `leaf_per_gpp.py`, `lai_npp_decomp.py`, `gpp_tower_cells.py`,
`mip_compare.py` (1991-2010 by decade against GCB, TRENDY v13 and GMB), `global_spatial.py` (needs matplotlib: run it
in the CPU environment). The release figures and tables come from `verify/calibration/diagnostics/release_2026-10-28/`
(its README gives the order); their outputs are in the reading folder `figures/final_2026-10-26/` with
`tables/scorecard_s31n_s31t.txt`, where every headline number is listed with its source.

## 5. What it scores

2001-2010 means unless stated; pools from the 2010 state.

| Quantity | IBIS 3.0 | Reference | Note |
|---|---|---|---|
| GPP, PgC yr-1 | 147 | FLUXCOM 109 on its land (144 model); published 110-160 | cell r 0.90; by band 1.25 / 1.49 / 1.82 (tropics / 30-60 N / 60-90 N); FLUXCOM is 0.75 of the towers at the tower cells |
| GPP at 435 tower cells, model / tower | 1.14 | 1 | Köppen A 0.96, B 1.30, C 1.18, D 1.05, E 1.04 |
| GPP trend, 1900s to 2000s | +21 % | +31 ± 5 % (Campbell et al. 2017); 20-30 % (TRENDY) | |
| ET, mm yr-1 | 529 | five-product ensemble 508 | 1.04; cell r 0.94; 55 % of cells inside the product spread; by band 1.08 / 0.99 / 0.80 |
| land sink (change of stored carbon), PgC yr-1 | 2.91 | GCB 2024 2.78 ± 0.69; TRENDY v13 mean 2.83 (1.4-4.4) | NPP - Rh before fire and harvest 8.1; carbon closure to 0.1 PgC yr-1, the soil residual being leaching |
| crop harvest, PgC yr-1 | 0.67 | about 1-2 | cropland as managed grass |
| wetland CH4, Tg CH4 yr-1 | 170 | GMB 2025 bottom-up 153 (116-189), 2000-09 | by band 152 / 12 / 6 against 109 / 32 / 12 |
| burned area, % of land yr-1 | 2.42 | GFED5 4.60 (products 2.4-4.6) | 0.53; rank r 0.76 (burning cells 0.43); African savanna 0.24 |
| wood carbon, PgC | 388 | Xu & Saatchi 2021: 362 | 1.07; humid tropics 1.01 (Amazon 0.98, Congo 0.63), savanna 0.71, temperate 1.70, boreal 1.10, tundra 0.60; rank r 0.70 |
| soil carbon, PgC | 1982 | HWSD2 0-1 m 2793 (98 % of land); literature 1500-2400 | 0.71; 90 S-30 N 0.82, 30-60 N 0.80, 60-90 N 0.47; boreal 0.59; cell r 0.51 |
| tree cover on natural land | 0.450 | CLM / MODIS 0.388 | phantom forest 3.6 % of natural land, lost forest 1.9 % |
| peak leaf area, land mean | 2.33 | GLASS / GIMMS LAI4g 2.12 | 1.10; cell r 0.79 |

## 6. What it does not do

1. **Burned area is half of GFED5**, a quarter in the African savannas: the burning there is largely lit by people and
   the scheme has no agricultural or lightning ignition term. The fire parameters sit at the optimum of the zone
   objective, so a further fire change would trade burned area for wood (71).
2. **Temperate wood is 1.7 times the biomass map** (Europe 1.9, the eastern United States 1.7, Australia 1.6): those
   forests are young and managed, and the model grows old growth over the 310-year protocol with no land-use history. The Congo is
   0.63 and humid Asia 1.36 of the map (67, 71).
3. **The methane split by band follows the towers, not the wetland-model ensemble**: 1.39 times the ensemble in the
   tropics, 0.36 in 30-60 N, 0.53 north of 60 N. The band parameters fitted at the towers carry the towers' own rates into
   the globe; rescaling a band to the ensemble would be tuning to a global total. The module has no frozen-season
   production (about 3 Tg yr-1 in the north at the towers), and the inundation input is a climatology, so the tropical
   towers' seasonal cycle is not reproduced (held-out r 0.03) (38, 73).
4. **GPP north of 30 N in spring and autumn** is 1.5-3 times FLUXCOM while July is close; annual GPP north of 60 N is
   1.8 times FLUXCOM, where FLUXCOM is also lowest against the towers and the campaign has eight polar towers (47-49, 69).
5. **Soil carbon is 0.71 of the HWSD2 top metre**: the cold zones fall short (boreal 0.59, north of 60 N 0.47). The soil is
   at equilibrium by the end of the spin-up (drift under 0.1 PgC yr-1 per zone, 77), so the gap is the equilibrium stock
   under the model's turnover and inputs, and the boreal HWSD2 stock is largely peat and permafrost carbon that a
   mineral-soil decomposition scheme has no process to build (72, 77).
6. **Tree cover is 0.450 against 0.388** on natural land, phantom forest 3.6 % (71).
7. **Cropland is a managed grass**: no crop phenology, sowing or harvest date; harvest 0.67 PgC yr-1 (41).
8. **Carried from the 12 October release and not re-evaluated**: the ET partition (transpiration share), growth respiration
   booked in daylight, the unclumped two-stream canopy, no vertically resolved decomposition, forcing ending in 2010.

## 7. Where things are

- Run: `verify_outputs/final_s31t/` (spin-up, transient, hindcast, `score/`); the reference run of the stage before the
  soil stage is `final_s31n/`, the cover-limit run `final_s31r/` (vegetation bit-identical to the release).
- Figures and tables: `ibis_local_document/figures/final_2026-10-26/` (`tables/scorecard_s31n_s31t.txt`,
  `tables/metrics_s31n_s31t.json`, `tables/mip_compare_s31t.txt`, `tables/gpp_tower_cells_s31t.txt`,
  `tables/ch4_bands_s31t.txt`, `tables/soil_figure_s31n_s31r_s31t.txt`, `tables/global_spatial_s31t.txt`).
- Log: [`CALIBRATION.md`](CALIBRATION.md) 70-74 for the final stages; [`FIRE.md`](FIRE.md), [`METHANE.md`](METHANE.md),
  [`BEDROCK.md`](BEDROCK.md), [`GPP.md`](GPP.md) for the modules; [`RUNNING.md`](RUNNING.md) and
  [`ENVIRONMENT.md`](ENVIRONMENT.md) for the machine.
