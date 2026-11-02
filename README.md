# IBIS 3.0

IBIS 3.0 is the Integrated Biosphere Simulator (IBIS 2.6b4; Foley et al. 1996, Kucharik et al. 2000) rewritten in JAX, checked against the Fortran in routine and time-step tests (with the remaining differences documented in technote Chapter 17), corrected where the original or the port was wrong, extended with a sunlit/shaded canopy, forty climate-zone plant types, a water-limited tree cover, fire, wetland methane, cropland and a deeper soil column, and calibrated from the flux towers outward to the globe. The global protocol at 0.5° (200-year spin-up, 1901–2010) takes about eight and a half hours on one GPU and a per-tower calibration under an hour; the gradients of the time step are at present not finite ([technote](docs/technote/) 17.11).

The release state (30 October 2026) is the parameter set [`params_31s_ch4h.yaml`](python-port/configs/params/params_31s_ch4h.yaml) on branch `main`, run with hourly wetland methane, its carbon taken out of the respiration CO2, and the learned wetland area (`methane_hourly`, `methane_carbon_partition`, `wetland_source: learned`). [`docs/RELEASE_3.0.md`](docs/RELEASE_3.0.md) describes it in short: what was calibrated against what, the file's provenance, how to reproduce the run and its scores, and the limitations. [`docs/CALIBRATION.md`](docs/CALIBRATION.md) is the full record, including the subsequent review in sections 86–87.1. The previous release on `main` was that of 12 October 2026 ([`params_s7_crop.yaml`](python-port/configs/params/params_s7_crop.yaml)); the calibration that led from it to this one is the history of branch `calibration-s1`.

**Early technote edition, 2 November 2026.** The [technical note](docs/technote/README.md) has been edited for clearer explanations while retaining its scientific content. It includes the completed diagnostic comparisons and distinguishes them from dated release results. This publication changes documentation only; the model code on main remains the 30 October release. See the [edition note](docs/technote/EARLY_EDITION.md) for its scope and Chapter 16 for the remaining scientific work.

## Results at a glance (30 October 2026)

**Towers.** The tower campaign fitted 356 flux towers (FLUXNET2015, ICOS, AmeriFlux, NEON and regional networks) one by one on their own meteorology, with one year held out, built one parameter set per climate-zone plant type from the medians of each type's towers, and re-scored every tower under two of the shared type sets: that of 20 October 2026 (`params_31a.yaml`, the set of the first global run of the types, which differs from the release in 113 tower-derived type values) and that of 26 October (`params_31k.yaml`). At the release date, the release set itself had not been re-scored at the towers; it differs from the 26 October set in the cold-shrub allocation, the cover-limit, wood, soil and fire stages and the methane bands ([technote](docs/technote/) Table 16.5 and section 16.14). Annual model/tower ratios, medians over the towers with a scored record (in brackets the share of towers within 25 % of the tower):

| Quantity | per-tower fits | shared type set of 20 October (`params_31a`) | shared type set of 26 October (`params_31k`) |
|---|---|---|---|
| GPP, 331 towers | 0.97 (80 %) | 0.98 (50 %) | 1.01 (53 %) |
| ET, 356 towers | 1.06 (63 %) | 1.02 (61 %) | 1.03 (62 %) |
| Ecosystem respiration, 331 towers | 1.01 (78 %) | 1.11 (44 %) | 1.13 (43 %) |
| Monthly correlation, GPP / ET / NEE | 0.88 / 0.89 / 0.60 | 0.87 / 0.89 / 0.55 | 0.87 / 0.89 / 0.52 |

The 12 October comparison at the original 62 FLUXNET2015 towers, 2009–2012 ("held-out" = a year the fit never saw; "within 25 %" = share of towers whose annual model/tower ratio lies between 0.75 and 1.25):

| Quantity | IBIS 2.6b4 default | IBIS 3.0, one set per PFT | IBIS 3.0, per-tower fits | Target |
|---|---|---|---|---|
| GPP, monthly r / NSE | 0.86 / 0.73 | 0.88 / 0.78 | 0.94 / 0.88 | 1 / 1 |
| GPP, annual within 25 % of tower | 58 % | 66 % | 90 % | 100 % |
| ET, monthly r / NSE | 0.80 / 0.57 | 0.81 / 0.62 | 0.83 / 0.65 | 1 / 1 |
| NEE, monthly r / NSE | 0.47 / 0.12 | – | 0.73 / 0.49 | 1 / 1 |
| NEE day-time uptake, held-out year, model/tower (within 25 %) | 0.68 (23 %) | – | 0.89 (69 %) | 1 |
| NEE night-time release, held-out year, model/tower | 1.02 | – | 1.07 | 1 |
| Wetland CH4, held-out model/tower: boreal / temperate / tropical (45 Fluxnet-Methane_V2.0 towers) | – | – | 0.84 / 0.82 / 0.95 | 1 |
| Wetland CH4, held-out monthly r: boreal / temperate / tropical | – | – | 0.66 / 0.83 / 0.03 | 1 |
| Wetland CH4, the release (flux every time step, 48 towers), held-out model/tower: boreal / temperate / tropical; middle column at the band parameters the global run uses | – | 0.99 / 0.79 / 1.51 | 0.82 / 0.69 / 0.78 | 1 |
| Wetland CH4, the release, held-out monthly r: boreal / temperate / tropical | – | 0.75 / 0.64 / 0.33 | 0.74 / 0.83 / 0.30 | 1 |
| Cropland GPP, held-out model/tower (r): C3 crops / maize (16 towers) | – | – | 0.76 (0.55) / 1.01 (0.77) | 1 |

**Global, 2001–2010** — means on true land area after the cold 200-year spin-up and the 1901–2000 transient with rising CO2; pools from the 2010 state. The 12 October column is the previous release; the final is this one.

| Quantity | IBIS 2.6b4 default | IBIS 3.0, 12 October | IBIS 3.0, final | Reference |
|---|---|---|---|---|
| Land sink (change of stored carbon), PgC yr⁻¹ | +2.8 | +1.6 | **+2.91** | GCB 2024: 2.78 ± 0.69; TRENDY v13 mean 2.83, range 1.4–4.4 |
| Crop harvest, PgC yr⁻¹ | – | 1.4 | **0.67** | about 1–2 |
| GPP, PgC yr⁻¹ | 178 | 195 (1.75 × FLUXCOM, cell r 0.85) | **148** (1.32 × FLUXCOM on its land, cell r 0.90) | published 110–160; FLUXCOM 109 on its land (0.70 × the towers) |
| GPP / FLUXCOM: tropics / 30–60N / north of 60 | – | 1.52 / 2.29 / 3.58 | **1.25 / 1.49 / 1.82** | 1 |
| GPP at the 435 tower cells, model / tower | – | – | **1.14** (Köppen A 0.96, B 1.30, C 1.18, D 1.05, E 1.04) | 1 |
| GPP trend, 1900s to 2000s | – | +18 % | **+21 %** | +31 ± 5 % (Campbell et al. 2017); 20–30 % (TRENDY) |
| ET, mm yr⁻¹ (cell r; cells inside the product spread) | 1.03 × | 527 (1.04 ×; 0.93; 54 %) | **529** (1.04 ×; 0.94; 55 %) | 508 (five-product ensemble) |
| ET / ensemble: tropics / 30–60N / north of 60 | – | 1.08 / 1.00 / 0.75 | **1.08 / 0.99 / 0.80** | 1 |
| Wetland CH4, Tg yr⁻¹ | 269 | 154 | **152** | GMB 2025 bottom-up: 153 (116–189) |
| CH4 by band 90S–30N / 30–60N / 60–90N | – | 130 / 11 / 13 | **135 / 11 / 7** | 109 / 32 / 12 |
| Burned area, % yr⁻¹ (rank r vs GFED5, all / burning cells) | 4.62 | 4.55 (+0.76 / +0.42) | **2.42** (+0.78 / +0.44) | 4.62 (products 2.4–4.6; the final column re-scored on the corrected GFED5 registration, CALIBRATION 81) |
| Wood carbon (above-ground), PgC | 471 | 349 | **388** (1.07 × the map; temperate 1.70, savanna 0.71, humid tropics 1.01) | Xu & Saatchi 2021: 362 |
| Soil carbon, PgC | 1938 | 2416 | **1982** (0.71 × HWSD2 0–1 m; 30–60N 0.80, north of 60 0.47) | HWSD2 0–1 m: 2793; literature 1500–2400 |
| Tree cover on natural land | – | – | **0.450** (phantom forest 3.6 %, lost forest 1.9 %) | 0.388 (CLM / MODIS) |
| Peak leaf area, land mean | – | 2.3 × | **2.33** (1.10 ×, cell r 0.79) | GLASS / GIMMS LAI4g 2.12 |
| Permafrost extent, Mkm² | – | 23.2 | not re-scored | 21.8 (Obu et al. 2019) |

**In one line:** at the towers the shared type sets re-scored before the release (20 and 26 October; the release set itself had not yet been re-scored) keep GPP and ET within a few percent of the towers at the median (0.98-1.03), put ecosystem respiration at 1.11-1.13, and lose spread; globally the land sink, ET, wetland methane, wood and soil carbon sit on their budgets, maps or product ranges, GPP is inside the published range and 1.3 times the lowest product, and the two quantities still outside are burned area, at half of GFED5 because the African savanna fires are lit by people, and the spring and autumn photosynthesis north of 30 N. The limitations are listed in full below and in [`docs/RELEASE_3.0.md`](docs/RELEASE_3.0.md).

### The final global run

![Figure 1. The final global run, 2001–2010 means: GPP, net flux (NPP − Rh), above-ground wood carbon, soil carbon, and wetland methane beside the Global Methane Budget 2025 bottom-up map.](docs/figures/readme/G1_maps.png)

*Figure 1. The release run, 2001–2010 means: GPP (148 PgC yr⁻¹), net flux (NPP − Rh with Rh as CO2 only, +8.2 PgC yr⁻¹, positive = uptake), above-ground wood carbon (387 PgC), soil carbon (1980 PgC), and the model's wetland methane (152 Tg yr⁻¹, hourly flux on the learned wetland area) beside the Global Methane Budget 2025 bottom-up wetland map on the same colour scale.*

![Figure 2. Zonal land means: GPP against FLUXCOM, NPP − Rh, and wetland methane per latitude row.](docs/figures/readme/G2_zonal.png)

*Figure 2. Zonal land means of the same run: GPP beside FLUXCOM, the net flux, and wetland methane per latitude row (Tg CH4 yr⁻¹).*

### Wetland methane: hourly flux, learned wetland area

Wetland CH4 is computed every time step from the hourly soil temperature, water and ice and the day's heterotrophic
respiration, with the band parameters refitted at the 48 Fluxnet-Methane_V2.0 wetland towers at that cadence. The methane
carbon is taken out of the respiration booked as CO2, so the emitted carbon is counted once (0.11 PgC yr⁻¹). The wetland area is dynamic,
seasonally and from year to year: the WAD2M seasonal climatology plus an anomaly from a learned model (gradient-boosted
trees per wetland band, monotone in water) of the month's inundation from the model's own month-mean soil water and
water totals and static topography. Held out by year, the learned area beats the fixed climatology against WAD2M in
every band (monthly R² 0.56 / 0.48 / 0.81 against 0.52 / 0.46 / 0.80, tropical / temperate / boreal) and follows the
year-to-year variation (r of the annual area 0.90 / 0.48 / 0.58), with damped amplitude (a quarter of the observed
swings in the tropics). Details: [`docs/METHANE.md`](docs/METHANE.md) 8-12, [`docs/CALIBRATION.md`](docs/CALIBRATION.md) 80-84.

Wetland CH4, Tg CH4 yr⁻¹, 2001–2010, under three wetland-area sources (hourly flux, refitted bands):

| Band | fixed climatology | learned (the release) | observed WAD2M month | GMB 2025 bottom-up, 2000–2009 |
|---|---|---|---|---|
| 90S–30N | 134.3 | **134.7** | 138.3 | 109.5 (66–146) |
| 30–60N | 10.5 | **10.6** | 10.5 | 32.1 (16–60) |
| 60–90N | 6.7 | **6.8** | 6.7 | 11.7 (4–28) |
| Global | 151.5 | **152.1** | 155.6 | 153.3 (116–189) |

The three runs are compared on one extraction (cell areas on the model's land fraction); the scorecard's true-land
totals, used in the table above, give the release 152.5 (135.0 / 10.7 / 6.8).

Learned against observed wetland area, and the CH4 each gives (learned / observed − 1):

| Band | area, decade mean | area, range over years | methane, range over years |
|---|---|---|---|
| tropics | −1.7 % | −3.5 to +2.0 % | −0.7 to −5.0 % |
| 30–60°N | +0.3 % | −4.3 to +3.6 % | −0.3 to +3.0 % |
| 60–90°N | +1.1 % | −0.8 to +3.3 % | −1.0 to +1.5 % |
| global | −0.8 % | −2.9 to +1.3 % | −0.6 to −4.4 % |

The annual CH4 anomalies of the learned and the observed run correlate at 0.98; the decade-mean area maps at cell r
0.999. The learned run is the release setting because it is the like-for-like protocol for the budget's numbers above,
which are the models' own (prognostic) wetland extents; the observed source remains a switch for diagnostic runs.

![Figure 3. GPP against FLUXCOM: model, reference, difference, ratio and zonal means.](docs/figures/readme/S_GPP_spatial.png)

*Figure 3. GPP against FLUXCOM: model, reference, model − reference, model / reference, and zonal land means. The pattern matches (cell r 0.90); the excess is north of 30 N (1.5 ×) and largest north of 60 N (1.8 ×), where FLUXCOM is also lowest against the towers.*

![Figure 4. ET against the five-product ensemble: model, reference, difference, cells outside the product spread, and zonal means.](docs/figures/readme/S_ET_spatial.png)

*Figure 4. ET against the ensemble of five satellite- and tower-based products (MODIS ET, MOD16A2, GLEAM v3.3a, DOLCE, FLUXCOM LE; from the ILAMB collection): model, ensemble mean, model − reference, and the part of the difference that lies outside the spread of the products (55 % of cells are inside it); the tropics are 8 % high and the far north 20 % low.*

## What changed from IBIS 2.6b4

| Component | IBIS 2.6b4 | IBIS 3.0 | Kind |
|---|---|---|---|
| Numerics | Fortran 77, serial, hourly | JAX on GPU, float32 with float64 verification, 900 s sub-steps (the time step's gradients not yet finite) | port |
| Reference and numerical corrections | real*4 physiology constants; bare-cell NaN; wet-day chain; the soil-water pump of `soilh2o` | corrections and scope documented in Chapter 17; the old wilting-point overflow was a synthetic-soil test artifact, not an established Fortran soil defect | corrections |
| Forcing in the port | — | surface pressure, phenology clock, solar latitude, grid duplication, normals on restart, accumulators | port fixes |
| Canopy photosynthesis | one big leaf, the stems' area counted in the leaves' capacity, vmax by existence priority | sunlit/shaded two big leaves, capacity e^(−Kn x), vmax mixed by leaf area, acclimation | new |
| Phenology | hard-coded thresholds, temperature-only senescence, inert drought trigger | parameters; photoperiod senescence; soil-stress drought trigger; absolute cold threshold | new |
| Plant types | 12 PFTs, one parameter set each | 40 Köppen-zone types in the 12 slots, each square taking its zone's types; a Köppen mask on existence | new |
| Cover and establishment | linear grass cover, 25 % floors, no water limit on trees | saturating cover, floors as parameters, a water-limited maximum tree cover per zone | new |
| Fire | annual-mean dryness, never switched on in the port | fire-season length, VPD × density danger, fine fuel, human terms, per-type mortality, closed-canopy damping, emissions, combustion completeness | new |
| Methane | none | one-scalar wetland module evaluated every time step; wetland area = WAD2M seasonal climatology + a learned year-to-year anomaly from the model's own water; per-band parameters fitted at towers at the hourly cadence | new |
| Cropland | none (crops become forest) | managed grass on a sub-grid share: harvest, crop capacity factor, calibrated at cropland towers | new |
| Soil column | 4 m, zero flux | +8 m bedrock, permafrost diagnostic | new |
| Respiration and allocation | hard-coded constants | per-type parameters, NEE by day and night, ET components; slow/passive soil turnover per zone group | parameters |
| Calibration | none shipped | tower-first: 356 towers one by one, then one set per type; 48 Fluxnet-Methane_V2.0 wetland towers and 16 cropland towers; then cover, wood and soil carbon against the gridded products inside the full protocol | calibration |

Most new processes sit behind a switch that is off by default; the details of the fire scheme are on by default (technote Table 13.1) and some additions act through parameter values (technote 10.6). The science is written down in `docs/`:

| Document | What it holds |
|---|---|
| [`docs/RELEASE_3.0.md`](docs/RELEASE_3.0.md) | the release state in short: provenance of the parameter file, what was calibrated against what, how to reproduce and score the run, the scorecard, the limitations |
| [`docs/CALIBRATION.md`](docs/CALIBRATION.md) | the calibration log, sections 1–85: every stage, its baseline, its lever tests and its verdict |
| [`docs/GPP.md`](docs/GPP.md) | the GPP overestimate and the canopy rebuild |
| [`docs/FIRE.md`](docs/FIRE.md) | fire module design and build log |
| [`docs/METHANE.md`](docs/METHANE.md) | methane module design and build log |
| [`docs/BEDROCK.md`](docs/BEDROCK.md) | the thermal bedrock layers |
| [`docs/DEFECTS_FOR_REVIEW.md`](docs/DEFECTS_FOR_REVIEW.md) | defects found in the port and the Fortran |
| [`docs/DATA_INVENTORY.md`](docs/DATA_INVENTORY.md) | every input, evaluation product and archived run, with its path |
| [`docs/RUNNING.md`](docs/RUNNING.md), [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | how to run, environments, pinned versions |
| [`verify/STATUS.md`](verify/STATUS.md) | the F77 ↔ JAX verification findings |

## Known limitations

1. **Burned area is half of GFED5**, a quarter in the African savannas, where the fires are lit by people and the scheme has no agricultural or lightning ignition term. The fire parameters sit at the optimum of the zone objective.
2. **Temperate wood is 1.7 times the biomass map** (temperate Europe 2.0, Asia 1.5, North America 1.35): young, managed forests against a model that grows old growth with no land-use history. The humid African tropics (the Congo basin) are 0.66 of the map.
3. **Wetland methane by band follows the towers, not the wetland-model ensemble**: 1.23 times the ensemble in the tropics, 0.33 and 0.58 in the two northern bands; frozen soil is excluded from the active substrate, although thawed subsurface layers can still support winter production; the learned wetland area's year-to-year swings are a quarter of the observed in the tropics.
4. **GPP north of 30 N in spring and autumn** is 1.5–3 times FLUXCOM while July is close; north of 60 N the annual GPP is 1.8 times FLUXCOM with eight polar towers behind the types.
5. **Soil carbon is 0.71 of the HWSD2 top metre**, the cold zones short of it (boreal 0.59, north of 60 N 0.47). The soil is at equilibrium by the end of the spin-up (drift at most 0.13 PgC yr⁻¹ per zone group), so this is the equilibrium stock under the model's turnover and inputs; the HWSD2 top metre in the boreal zones is largely peat and permafrost carbon that a mineral-soil decomposition scheme does not accumulate.
6. **Tree cover is 0.45 against 0.39** on natural land (phantom forest 3.6 %).
7. **Cropland** is a managed grass: no crop phenology, sowing or harvest date; harvest 0.67 PgC yr⁻¹.
8. **Other limits and subsequent diagnostics.** The canopy remains unclumped, decomposition has no vertical resolution, and forcing ends in 2010. The ET partition has since been examined in the 624-run comparison (technote 16.14.2), without validation against observed transpiration. Growth respiration follows net carbon gain rather than daylight: the release also reduced deficits; revised code charges only positive gain (technote 11.2.3).

## Repository layout

| Directory | What it contains |
|---|---|
| [`python-port/`](python-port/) | the model: `ibis/` (physics, bgc, timestepping, io), `scripts/run_global.py` (the global runner), `configs/params/` (parameter sets), `tests/` (800+ tests) |
| [`verify/`](verify/) | `calibration/` (tower harness, site fits, global protocol, diagnostics, the release figure scripts), `timestepping/` (F77 ↔ JAX lockstep verification), `STATUS.md` |
| [`fortran/`](fortran/) | the original IBIS 2.6b4 (Fortran 77) and a Fortran 2008 rewrite |
| [`docs/`](docs/) | the science log (table above) and the README figures |
| [`configs/`](configs/) | spin-up configurations of earlier stages |

Parameter sets in [`python-port/configs/params/`](python-port/configs/params/): `params_31s_ch4h.yaml` (**the IBIS 3.0 release**: the type set with the cover limit, the production refit, the soil-carbon factors and the methane bands refitted at the hourly cadence), its chain `params_31s_soil_strong.yaml` (the same without the hourly methane bands), `params_31r.yaml` (cover limit and production) and `params_31n.yaml` (the allocation stage), and `params_s7_crop.yaml` (the 12 October release on `main`; class medians of the 62-tower fits with methane and cropland). The provenance of each step is in [`docs/RELEASE_3.0.md`](docs/RELEASE_3.0.md) section 2.

## Quick start

The inputs (GSWP3 forcing, HWSD soils, ESACCI land cover, WAD2M inundation, the cropland map, FLUXNET2015 and Fluxnet-Methane_V2.0 towers, and the evaluation products) are not in the repository; [`docs/DATA_INVENTORY.md`](docs/DATA_INVENTORY.md) lists each with its source. The calibration harness and `final_validation.sh` carry the paths of our machine.

```bash
# environment (CPU; add jax[cuda12] for the GPU) — pinned versions in docs/ENVIRONMENT.md
python3 -m venv .venv && source .venv/bin/activate
pip install -e "python-port[wetland]" cma pandas pytest     # [wetland]: xgboost + scipy, for the learned wetland area

# tests (~10 min)
cd python-port && PYTHONPATH=. python -m pytest tests/ -q && cd ..

# a global run from a configuration (spin-up, transient or hindcast)
PYTHONPATH=python-port:. python python-port/scripts/run_global.py <cfg.yaml>

# the IBIS 3.0 protocol as the release ran it: cold 200-yr spin-up, 1901-2000 transient, 2001-2010 hindcast, scoring against the budgets
SPINUP_YEARS=200 METHANE=1 KOPPEN_TYPES=1 KOPPEN=<koppen_npz> CROPFRAC=<cropfrac_npz> \
  METHANE_HOURLY=1 METHANE_CARBON_PARTITION=1 WETLAND_SOURCE=learned WETLAND_LEARNED_FILE=<wetland_learned_v2_s31t.npz> \
  bash verify/calibration/final_validation.sh rel30 python-port/configs/params/params_31s_ch4h.yaml
# (stage templates: verify/calibration/protocol/; IBIS_OUT, PY and CPY set the output root and the GPU / CPU interpreters)

# a per-tower fit (CMA-ES) at the evergreen-needleleaf towers
PYTHONPATH=python-port:. python verify/calibration/site_fit.py --igbp ENF \
  --params python-port/configs/params/params_31s_ch4h.yaml \
  --fit vmax:temp_con_ever,acclim_tau,acclim_s0 --n-evals 200 --workers 4 --out <dir>
```

See [`docs/RUNNING.md`](docs/RUNNING.md) for the reference state, the common tasks and the gotchas.

## Port verification against the Fortran

The port was compared with IBIS 2.6b4 (F77) using shared inputs, first routine by routine and then hour by hour.
The May–June 2026 time-step comparisons used fixed vegetation and simplified test soils. They show close agreement
in many tested conditions, with larger differences at some hours. They do not establish bit-for-bit equivalence,
complete model correctness or validation of the processes added in IBIS 3.0.

The current interpretation, including the revised test settings and unresolved differences, is in
[Chapter 17](docs/technote/17_numerics_verification.md), especially sections 17.9–17.12. The historical numerical
tables below are retained for provenance; they are not results from a rerun of the current code.

<details>
<summary><b>Historical verification results, early GPU benchmarks and numerical safeguards</b></summary>

### Historical time-step comparisons

The following values were reported in the early verification summary. Their aggregation differs among rows:
short tests report maxima over the sampled cells and steps, while the multi-year rows also report distributions.
ΔT denotes temperature differences in K and ΔW water or snow fractions. A finite run is not necessarily an
accurate or physically valid run, and a nonzero difference is not bit-exact agreement.

| Test | Cells | Reported ΔT (K) | Reported ΔW (fraction) | Scope |
|---|---|---|---|---|
| Single-step base | 5 | 5.8×10⁻⁶ | 1.6×10⁻¹⁵ | historical comparison; see qualifications below |
| 24-step base, 1 day hourly | 5 | 5.9×10⁻⁶ | 1.4×10⁻¹¹ | historical comparison; see qualifications below |
| 24-step perturbed (10 random seeds + seed 42) | 5 × 11 seeds | 9.5×10⁻⁶ | 2.0×10⁻¹⁰ | historical comparison; see qualifications below |
| 7-day diurnal base | 5 | 1.2×10⁻⁵ | 1.5×10⁻⁷ | historical comparison; see qualifications below |
| 7-day cold snap (Ta −20 K) | 5 | 7.9×10⁻⁶ | 4.7×10⁻⁸ | historical comparison; see qualifications below |
| 7-day heat wave (Ta +15 K) | 5 | 4.4×10⁻⁵ | 1.9×10⁻⁷ | historical comparison; see qualifications below |
| 7-day monsoon (10× rain days 3–4) | 5 | 1.1×10⁻³ | 2.0×10⁻⁵ | historical comparison; see qualifications below |
| 7-day polar night | 5 | 9.6×10⁻⁸ | 8.0×10⁻¹¹ | historical comparison; see qualifications below |
| 7-day freeze-thaw oscillation | 5 | 1.5×10⁻⁵ | 1.3×10⁻⁷ | historical comparison; see qualifications below |
| **30-day diurnal base (720 timesteps)** | **5** | **3.1×10⁻⁴** | **6.8×10⁻⁶** | historical comparison; see qualifications below |
| **Global GSWP3, real forcing (single timestep)** | **~120** | **1.1×10⁻¹³** | **<1×10⁻⁹** | historical comparison; see qualifications below |
| **Global GSWP3, full year 1990 (8760 timesteps × 117 cells, findings #20 + #24 applied)** | **117** | **7.4×10⁻¹** (worst cell L+55°N +040°E canopy.tl) | **3.1×10⁻³** (worst cell L+55°N +100°E soil.wsoi[0]) | historical comparison; see qualifications below |
| **Global GSWP3, 35-year multi-decadal 1951-1985 (306,600 timesteps × 117 cells, WMO climate normal length)** | **116 finite + 1 NaN cell** | **4.5×10⁻¹** (worst single cell-year canopy.tu, 1951 cold-start) — median **1.25×10⁻³**, p95 **3.3×10⁻²** | **4.5×10⁻³** soil.wsoi[0] | historical comparison; see qualifications below |
| **Global GSWP3, 46-year multi-decadal 1951-1996 post-finding-#26 (~48M cell-step comparisons in total)** | **117 finite, no NaN** | **2.1×10⁻¹** (1985 freeze/thaw single-spike canopy.tu) — **median 3 µK**, **p99 20 mK** | **6.5×10⁻³** soil.wsoi[0] | historical comparison; see qualifications below |

The 46-year row's 0.21 K value is retained as an earlier reported event in 1985, not as the largest difference over
the complete record. The later audit of the same archived outputs gives a largest upper-canopy difference of
0.45 K in 1951 and a largest lower-canopy difference of 0.87 K. The upper-canopy median and 99th percentile over
5,382 cell-years are 2.6 µK and 17 mK, rounded in the early summary to 3 µK and 20 mK. These are distributions of
each cell-year's largest hourly difference; they are not errors of annual means. See Chapter 17, Table 17.8.

The old test soils had a negative saturated suction that kept capillary diffusion at its floor in both codes.
The settings also differed from IBIS 2.6b4's own soil, snow and vegetation parameters. Since November 2026 the
verification harness uses those original settings. Revised one-step checks then agree to about 10⁻¹² K in the
compared temperatures; the lower-canopy sensible-heat diagnostic now differs by design. The 46-year run has not
been repeated with the revised harness or current code.

### What the differences do and do not show

Small numerical differences can grow when freezing or melting sends the two calculations through different
branches. A diagnostic at 55°N, 100°E on 25 April of the separate 1990 comparison tested one such event: starting
the port from the Fortran state just before that hour reduced the next-step differences to at most 2 × 10⁻⁹ in
the recorded quantities. This supports accumulated rounding as the cause of that event. It does not isolate a
particular math library or fused operation, or show that every discrepancy has the same cause.

Two later cases remain unexplained: a spring canopy event at 55°N, 40°E and the fourth-day event in the synthetic
monsoon test. Giving the port the Fortran state immediately before those events does not remove their differences.
Chapter 17.9.3 and its developer notes distinguish these cases from the rounding-sensitive cases.

### The 46-year comparison, 1951–1996

The two codes ran continuously at 117 cells, about 48 million cell-step comparisons, with vegetation fixed and
the old synthetic surface settings. Neither produced a non-finite value. The temperature differences did not show
steadily increasing drift across those years, but this observation is limited to that configuration.

The original statement that there were no out-of-range events was incorrect. The archived files contain
1,117,261 step-level exceedances of the harness's plausibility windows; the flagged cell-months occurred in both
codes. These include Antarctic temperatures below 200 K and large sensible-heat diagnostics, some at snow-free
surfaces where the snow flux does not enter the energy balance. They are described in Chapter 17.9.4.

The run stopped during 1997 after an input-file read failure, recorded for `rsds_gswp3_1991_2000.nc4`.
The completed 46 years are evidence about numerical agreement under the test conditions, not proof of general
equivalence or of the correctness of a future global simulation.

### Historical GPU benchmark (RTX 4090, double precision)

The early benchmark compared ways of running the same ported time step on an RTX 4090:

| Configuration | ms / timestep (FP64) | Speedup vs no-JIT |
|---|---|---|
| no-JIT, single step (eager) | 602 | 1× (baseline) |
| JIT'd single step | 4.66 | 130× |
| JIT + `lax.scan` over 24 h | 1.42 | 423× |
| JIT + `lax.scan` over 720 h (1 month) | 0.87 | **691×** |

The record reports 1.3 µs per cell-step at 1,000 cells. It projected about 9 minutes for a 50-year, 117-cell
GPU-only calculation, against about 58 hours for the CPU lockstep workflow, a roughly 400-fold wall-clock ratio.
That was a projection between different workflows, not a measured speedup of the full IBIS 3.0 protocol.
The GPU workflow omits the in-loop Fortran comparison, which runs on the CPU.

These benchmarks used the historical `verify/.venv_gpu/` environment with `jax[cuda12]==0.10.0` and
[bench_gpu_timestep.py](verify/timestepping/pipeline/bench_gpu_timestep.py). Use
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the current pinned environments, and Chapter 17.8 for the tested
scope of deterministic GPU reductions.

### Early global GPU spin-ups (June 2026)

The historical physics runner was also exercised at 0.5° resolution with 88,484 land cells. This was a larger
numerical-stability test, distinct from the later release protocol and its 95,333-cell land mask. The runner is
[global_gswp3_gpu.py](verify/timestepping/pipeline/global_gswp3_gpu.py), with forcing-cycle options and the
initialization helpers in [real_state_builder.py](verify/timestepping/pipeline/real_state_builder.py).

| Run | Resolution | Cells | Wall-clock | Per-year (steady-state) |
|---|---|---|---|---|
| **0.5° × 200-yr (synthetic ICs)** | 0.5° | 88,484 | ~11 hr on RTX 4090 | ~195 sec/yr |
| **0.5° × 200-yr (real ICs)** | 0.5° | 88,484 | ~11 hr | ~195 sec/yr |

Both recorded runs cycled the 1901–2000 forcing twice, starting from cold-start initial conditions. The early
record reports that the resulting approximately 155 billion cell-steps per run stayed finite after safeguards
were applied. Finite output alone does not verify energy and water conservation, realism or equivalence with the
Fortran at every global cell. These runs were not repeated for the documentation review.

### Ground-evaporation safeguard (finding 27)

The higher-resolution tests exposed four Yemen-highland cells near 13.5–14°N, 44–44.5°E that the 117-cell grid had
not sampled. The historical trace records an unstable canopy calculation producing ground evaporation near
0.26 kg m⁻² s⁻¹, about 940 mm h⁻¹, followed by nonphysical stores and non-finite values.

The port bounds cell-mean ground evaporation to ±10⁻³ kg m⁻² s⁻¹, equivalent to ±3.6 mm h⁻¹. This is a numerical
safeguard, not a universal physical upper limit. The current `bounded_ground_evaporation` function in
[loops.py](python-port/ibis/timestepping/loops.py) supplies the same bounded rate to the water calculation,
ground latent heat and ET output; the release bounded only the water. Chapter 17.4 describes the bounds and
their diagnostic counts. They can make the two codes differ when active.

This should not be grouped with finding 26 as an established inherited-soil defect. Finding 26's overflow came
from the negative suction of the old synthetic harness; its classification as an inherited bug was withdrawn.
The guard remains in both codes. The bare-canopy singularity of finding 20 is a separate case, discussed below.

### Historical input-based initialization example

The early runner's `--real-ic` option replaces its synthetic latitude-band templates with soil texture from
HWSD and vegetation classes from ESA CCI. The associated initial states are computed by
[soil_init.py](python-port/ibis/io/soil_init.py) and [veg_init.py](python-port/ibis/io/veg_init.py).
The following Yakutsk example (62°N, 130°E) compares the two initializations; it is not an independent validation
against measured canopy structure or soil properties.

| Field | Synthetic template | Real (HWSD + ESACCI) | Interpretation of the input-based initialization |
|---|---|---|---|
| `fu` (upper canopy cover) | 0.10 | **0.97** | high initialized canopy cover |
| `lai` (upper) | 0.5 | **4.1** | initialized leaf area |
| Soil sand | template avg | **52%** | input texture |
| Soil clay | template avg | **13%** | — |
| Wilting point | 0.15 (template) | **0.21** (texture-derived) | — |

The synthetic option was retained for comparisons using the historical harness. The current production driver
and release initialization are described in [Chapter 2](docs/technote/02_surface_characterization.md).

### Historical flux differences in the 46-year run

For each cell, the early flux summary took the largest hourly absolute difference across all 46 years, then
reported the distribution over cells. This aggregation differs from the distribution over cell-years in
Chapter 17, Table 17.8, so their percentiles should not be compared directly.

| Flux | Max abs Δ (W/m²) | p95 across cells | Median Δ across cells | Worst cell · year |
|---|---|---|---|---|
| flux.fseng (ground sensible heat) | 7.49 | 0.34 W/m² | 30 mW/m² | Yukon L+65°N -120°E · 1983 |
| flux.fseni (sensible heat into snow) | 6.97 | 3.59 W/m² | 340 mW/m² | L+35°N -80°E · 1969 |
| flux.fsenu (upper canopy sensible heat) | 12.12 | 0.45 W/m² | 15 mW/m² | Russia L+55°N +40°E · 1951 |
| flux.fsenl (lower canopy sensible heat) | 12.50 | 0.63 W/m² | 15 mW/m² | Siberia L+55°N +160°E · 1987 |

Temperature differences can produce larger flux differences through the transfer coefficients, particularly
near phase changes. The table alone does not establish the cause of each extreme or a general observational
noise threshold. The earlier 1990-only comparison reported a lower-canopy sensible-heat maximum of 21.6 W m⁻²;
it belongs to a separate trajectory and is not the maximum of this 46-year trajectory. Also, the current code's
lower-canopy heat diagnostic uses the lower canopy's own air temperature, a deliberate correction that must be
accounted for in any new Fortran comparison.

### Bare-canopy singularity (finding 20) and remaining work

In an earlier one-year test, three bare arid cells developed non-finite values in both codes. The investigation
traced this to a singular canopy system with vanishing vegetation area. A bare-canopy branch was added to both
the port and the reference Fortran; the repeated 117-cell year then stayed finite. This outcome establishes the
effect of that correction in the tested cases, not that all numerical issues are closed.

The detailed record is [verify/STATUS.md](verify/STATUS.md). For its later corrections, the harness scope and the
unresolved canopy cases, use [Chapter 17](docs/technote/17_numerics_verification.md). New scientific processes,
driver behavior, calibration and numerical repeatability require their own checks in addition to the historical
Fortran comparisons.

</details>

## Citation

A paper describing IBIS 3.0 is in preparation; until it appears, please cite this repository and the original model:

- Foley, J.A. et al. (1996), "An integrated biosphere model of land surface processes, terrestrial carbon balance, and vegetation dynamics", _Global Biogeochemical Cycles_ 10(4): 603–628.
- Kucharik, C.J. et al. (2000), "Testing the performance of a dynamic global ecosystem model: water balance, carbon balance, and vegetation structure", _Global Biogeochemical Cycles_ 14(3): 795–825.

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Hangkai You · hyou34@wisc.edu · University of Wisconsin–Madison · [UW-GCRL](https://github.com/UW-GCRL)
- Min Chen · min.chen@wisc.edu · University of Wisconsin–Madison · [UW-GCRL](https://github.com/UW-GCRL)
