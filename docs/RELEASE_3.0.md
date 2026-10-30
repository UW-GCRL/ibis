# IBIS 3.0: the calibrated model of 30 October 2026

The short description of the IBIS 3.0 release state: what the model is, which parameter file is the release and where it
came from, what was calibrated against what, how the release run is reproduced and scored, what it scores, and what it
does not do. The long form is the calibration log, [`CALIBRATION.md`](CALIBRATION.md) (85 sections, each stage with its
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
module (one scalar form) evaluated every time step, with the water table from the model's own soil moisture and a
dynamic wetland area (the WAD2M seasonal climatology plus a learned year-to-year anomaly from the model's own water); cropland as managed
grass on the ESA CCI cropland share with harvest; three thermal bedrock layers under the 4 m column; per-type respiration
and allocation; and the diagnostics the towers measure (ET parts, day and night NEE, fire emissions, methane, harvest).
The global run has 95,333 half-degree land cells; the protocol (200-year spin-up, 1901-2010) takes about eight and a half hours on one RTX 4090.

## 2. The release parameter file and its provenance

The release is `python-port/configs/params/params_31s_ch4h.yaml`, run through the whole protocol from a cold start with
every release switch on (30 October 2026, run tag `rel30`; the command is in section 4). That run equals the one assembled
before it (the protocol `s31t` of 28 October and its 2001-2010 hindcast rerun with hourly methane and the learned wetland
area) in every carbon, water, vegetation and methane total to the printed digits; the one difference is the carbon
partition, which books 0.11 PgC yr-1 of the respiration as CH4 instead of CO2 (CALIBRATION 84). Newest change first:

| Step | File | What it added | Log |
|---|---|---|---|
| methane at the hourly cadence | `params_31s_ch4h.yaml` | the methane band parameters refitted at the 48 Fluxnet-Methane_V2.0 wetland towers with the flux evaluated every time step (the earlier fits used the month-end state); used with `methane_hourly` and the learned wetland area (`wetland_learned_v2_s31t.npz`) | 80-84 |
| soil-carbon stage | `params_31s_soil_strong.yaml` | slow- and passive-pool turnover scaled per zone group on the zones' tree types (`som_k_scale`); the stronger member of a two-member bracket run through the full protocol | 72 |
| cover limit and production | `params_31r.yaml` | the water-limited maximum tree cover with a ramp per zone (the stress gate retired); the C/D-zone allocation refit (awood 0.35, the leaf cut going to roots); shorter temperate wood residence (45 / 40 yr); the tropical wood refit at half strength | 70-71 |
| allocation stage, road B | `params_31n.yaml` | leaf area set through the allocation and leaf-mass coefficients, no caps; the cold shrub kept (aleaf 0.30) | 54-69 |
| the type set | carried in the above | 40 climate-zone types, each the median of its towers' fits from the 356-tower campaign; methane bands from 45 Fluxnet-Methane_V2.0 wetland towers (refitted at 48 in the step above); cropland from 16 cropland towers | 45-57, 38, 41 |
| 12 October release (the previous `main`) | `params_s7_crop.yaml` | the class-median set of the 62-tower fits, with methane and cropland | 37-43 |

Switches on in the release run: `koppen_types` (with the Köppen mask `koppen_file`), `tree_cover_limit`, `enable_methane`
with `methane_hourly`, `methane_carbon_partition` (the net CH4 carbon taken out of the respiration booked as CO2) and
`wetland_source: learned` (`wetland_learned_file`), `enable_fire_emissions`, cropland
(`cropfrac_file`). Organic-soil hydrology is off. The run is cold: no state is inherited
from an earlier run.

Corrections that are not switches change results against port runs made before 25-26 October; each carries a setting or
note that restores the old behaviour, and without them the carbon budget does not close (CALIBRATION 59-63, 81): the
fire counters cleared every year as F77 does (`fire_annual_reset`) and the per-type fire parameters reaching the fire
scheme (59); the last 50 spin-up years at the true soil rate (`soilcspin_tail_years`, 60); the tree carbon removed by the
cover limit booked to litter (`tree_kill_to_litter`, 61); F77's snow constants (63; a restart written before carries the
old ones and takes F77's on loading); the cropland harvest and the shrub NPP of the cropland share booked as disturbance
(63); WAD2M and GFED5 read on the right cells (81).

## 3. What was calibrated against what

Towers first, then the globe; no global total was ever a term in a loss function.

| Process | Parameters | Data | How | Log |
|---|---|---|---|---|
| photosynthesis, stomata, phenology, respiration, allocation | per type: capacity and acclimation, Ball-Berry, leaf traits, allocation and turnover, respiration | 356 flux towers (FLUXNET2015, ICOS, AmeriFlux, NEON, regional networks): monthly GPP, ET and NEE on the tower's own meteorology, three to five years, one held out | CMA-ES per tower on a bounded objective; each type is the median of its towers; every tower re-scored under the shared set | 34-36, 45, 50-57 |
| wetland methane | rate, Q10, anaerobic depth scale and oxidation scale per wetland band | 48 wetland towers of Fluxnet-Methane_V2.0, held-out year | per-tower fits with the flux evaluated every time step, band medians by Köppen wetland type | 38, 42, 73, 80 |
| wetland area | the learned anomaly model (trees per band, monotone in water) | WAD2M v2.0 monthly fraction 2001-2010 | trained on a reference hindcast's month-mean soil state and water totals; held out by year | METHANE 8-11, 81 |
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
  METHANE_HOURLY=1 METHANE_CARBON_PARTITION=1 WETLAND_SOURCE=learned WETLAND_LEARNED_FILE=<wetland_learned_v2_s31t.npz> \
  bash verify/calibration/final_validation.sh rel30 python-port/configs/params/params_31s_ch4h.yaml
```

Stages: cold spin-up 1701-1900 (1901-1930 climate cycled, 296 ppm, soil carbon accelerated then released), transient
1901-2000 (GSWP3 years, rising CO2, yearly output), hindcast 2001-2010 (monthly output; methane and fire emissions
written). The stage templates are `verify/calibration/protocol/{spinup,transient}.yaml` (input paths of the lab NAS;
[`DATA_INVENTORY.md`](DATA_INVENTORY.md) lists every source); the environment variables `IBIS_OUT`, `PY` and `CPY` set the
output root and the run (GPU) and scoring (CPU) interpreters. Outputs land under
`$IBIS_OUT/final_<TAG>/{spinup,transient,hindcast}/{out,restart}`; the protocol log
carries the built-in scoring: pools, fluxes and latitude bands against GCB 2024 and GMB 2025, tree cover against the
product, ET against the five-product ensemble, the seasonal cycle, GPP at the 435 tower cells, fire against GFED5
(2001-05 from the 2010 state), the 1901-2010 GPP trend. The chain that follows (`verify/calibration/diagnostics/`):
`gate_compare.py` against the previous run, `leaf_per_gpp.py`, `lai_npp_decomp.py`, `gpp_tower_cells.py`,
`mip_compare.py` (1991-2010 by decade against GCB, TRENDY v13 and GMB), `global_spatial.py` (needs matplotlib: run it
in the CPU environment). The numbers in section 5 come from that chain on the release run (`mip_compare.txt`,
`global_spatial.txt`, `gpp_tower_cells.txt` and the vegetation comparison in its `score/` folder); the README figures from
`global_figures.py` and `global_spatial.py`; the wood, soil and fire figures of the release stage from
`verify/calibration/diagnostics/release_2026-10-28/` (its README gives the order).

## 5. What it scores

2001-2010 means unless stated; pools from the 2010 state.

| Quantity | IBIS 3.0 | Reference | Note |
|---|---|---|---|
| GPP, PgC yr-1 | 148 | FLUXCOM 109 on its land (144 model); published 110-160 | cell r 0.90; by band 1.25 / 1.49 / 1.82 (tropics / 30-60 N / 60-90 N); FLUXCOM is 0.70 of the towers at the tower cells |
| GPP at 435 tower cells, model / tower | 1.14 | 1 | Köppen A 0.96, B 1.30, C 1.18, D 1.05, E 1.04 |
| GPP trend, 1900s to 2000s | +21 % | +31 ± 5 % (Campbell et al. 2017); 20-30 % (TRENDY) | |
| ET, mm yr-1 | 529 | five-product ensemble 508 | 1.04; cell r 0.94; 55 % of cells inside the product spread; by band 1.08 / 0.99 / 0.80 |
| land sink (change of stored carbon), PgC yr-1 | 2.91 | GCB 2024 2.78 ± 0.69; TRENDY v13 mean 2.83 (1.4-4.4) | NPP - Rh before fire and harvest 8.1 (8.2 with Rh counted as CO2 only: the 0.11 PgC of wetland CH4 carbon); carbon closure to 0.1 PgC yr-1, the soil residual being leaching |
| crop harvest, PgC yr-1 | 0.67 | about 1-2 | cropland as managed grass |
| wetland CH4, Tg CH4 yr-1 | 152 | GMB 2025 bottom-up 153 (116-189), 2000-09 | by band 135 / 11 / 7 against 109 / 32 / 12; hourly flux on the learned wetland area (observed WAD2M area: 156; fixed climatology: 151) |
| burned area, % of land yr-1 | 2.42 | GFED5 4.62 (products 2.4-4.6) | 0.52; rank r 0.78 (burning cells 0.44), on the corrected GFED5 registration (81); African savanna 0.24 |
| wood carbon, PgC | 388 | Xu & Saatchi 2021: 362 | 1.07; humid tropics 1.01 (Amazon 0.98, Congo 0.66), savanna 0.71, temperate 1.70, boreal 1.11, tundra 0.60; rank r 0.70 |
| soil carbon, PgC | 1982 | HWSD2 0-1 m 2793 (98 % of land); literature 1500-2400 | 0.71; 90 S-30 N 0.82, 30-60 N 0.80, 60-90 N 0.47; boreal 0.59; cell r 0.51 |
| tree cover on natural land | 0.450 | CLM / MODIS 0.388 | phantom forest 3.6 % of natural land, lost forest 1.9 % |
| peak leaf area, land mean | 2.33 | GLASS / GIMMS LAI4g 2.12 | 1.10; cell r 0.79 |

## 6. What it does not do

1. **Burned area is half of GFED5**, a quarter in the African savannas: the burning there is largely lit by people and
   the scheme has no agricultural or lightning ignition term. The fire parameters sit at the optimum of the zone
   objective, so a further fire change would trade burned area for wood (71).
2. **Temperate wood is 1.7 times the biomass map** (temperate Europe 2.0, Asia 1.5, North America 1.35, Australia 2.4): those
   forests are young and managed, and the model grows old growth over the 300-year protocol with no land-use history. The humid African
   tropics (the Congo basin) are 0.66 and humid Asia 1.36 of the map (67, 71).
3. **The methane split by band follows the towers, not the wetland-model ensemble**: 1.23 times the ensemble in the
   tropics, 0.33 in 30-60 N, 0.58 north of 60 N. The band parameters fitted at the towers carry the towers' own rates into
   the globe; rescaling a band to the ensemble would be tuning to a global total. The module has no frozen-season
   production (about 3 Tg yr-1 in the north at the towers). The learned wetland area follows the observed year-to-year
   variation (r 0.90 tropics, 0.48 temperate, 0.58 boreal, held out) at damped amplitude (a quarter of the observed
   swings in the tropics); a prognostic extent needs river routing (38, 73, 80-84; METHANE 8-12).
4. **GPP north of 30 N in spring and autumn** is 1.5-3 times FLUXCOM while July is close; annual GPP north of 60 N is
   1.8 times FLUXCOM, where FLUXCOM is also lowest against the towers and the campaign has eight polar towers (47-49, 69).
5. **Soil carbon is 0.71 of the HWSD2 top metre**: the cold zones fall short (boreal 0.59, north of 60 N 0.47). The soil is
   at equilibrium by the end of the spin-up (drift at most 0.13 PgC yr-1 per zone group, 77), so the gap is the equilibrium stock
   under the model's turnover and inputs, and the boreal HWSD2 stock is largely peat and permafrost carbon that a
   mineral-soil decomposition scheme has no process to build (72, 77).
6. **Tree cover is 0.450 against 0.388** on natural land, phantom forest 3.6 % (71).
7. **Cropland is a managed grass**: no crop phenology, sowing or harvest date; harvest 0.67 PgC yr-1 (41).
8. **Carried from the 12 October release and not re-evaluated**: the ET partition (transpiration share), growth respiration
   booked in daylight, the unclumped two-stream canopy, no vertically resolved decomposition, forcing ending in 2010.

## 7. Where things are

- Runs (lab NAS, not in the repository): the release `verify_outputs/final_rel30/` (spin-up, transient, hindcast,
  `score/`). The log's earlier names: `final_s31t/` with the methane hindcasts `final_s31t_ch4h_learned_r2/` (and
  `_clim_r2`, `_obs_r2`), equal to the release but for the partition; `final_s31n/`, the stage before the soil stage;
  `final_s31r/`, the cover-limit run (vegetation bit-identical to the release).
- Figures and tables of the log (lab NAS, not in the repository): `ibis_local_document/figures/final_2026-10-26/` (`tables/scorecard_s31n_s31t.txt`,
  `tables/metrics_s31n_s31t.json`, `tables/mip_compare_s31t.txt`, `tables/gpp_tower_cells_s31t.txt`,
  `tables/ch4_bands_s31t.txt`, `tables/soil_figure_s31n_s31r_s31t.txt`, `tables/global_spatial_s31t.txt`,
  `tables/wetland_modes_compare_r2.txt`, `tables/wetland_learned_vs_observed_breakdown.txt`,
  `tables/fire_rescored_registration_fix.txt`; wetland figures in `wetland_2026-10-29/`).
- Log: [`CALIBRATION.md`](CALIBRATION.md) 70-85 for the final stages and the pre-release review; [`FIRE.md`](FIRE.md), [`METHANE.md`](METHANE.md),
  [`BEDROCK.md`](BEDROCK.md), [`GPP.md`](GPP.md) for the modules; [`RUNNING.md`](RUNNING.md) and
  [`ENVIRONMENT.md`](ENVIRONMENT.md) for the machine.
