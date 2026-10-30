# Calibration diagnostics (one-off scripts and their output)

The lever tests and decompositions behind `docs/CALIBRATION.md` sections
14-18, archived as run. They are one-off diagnostics, not tools: paths are
hard-coded to the nas2 layout, each imports the calibration harness
(`fit_carbon.py`, `site_transient.py`) from the repo, and most run 60-cell
subsets on the CPU environment (`ibis-python`) in 5-25 minutes. The `.txt`
files are the outputs the sections quote.

| script | question | section |
|---|---|---|
| `lever_test.py` .. `lever_test5.py` | phenology and stress levers on 60 sparse-excess cells | 14 |
| `notree_test.py`, `canopy_decomp.py`, `composition_test.py`, `regime_test.py`, `stress_test.py`, `../vmax_experiment.py`, `tower_offseason.py` | where the sparse-cell excess sits (canopy, PFT, season) | 14 |
| `fu_vs_clm_v11d.py` | tree cover vs CLM tree fraction, globally (became `fu_vs_tree.py`) | 14 |
| `lever_test6.py` | the tree-establishment gate at 0.45 / 0.55 / 0.65 on sparse, dense and tower cells | 14 |
| `tropical_phase.py` | the tropical GPP phase lag by seasonality class | 16 |
| `lever_test7.py` | the soil-water drought trigger against the savanna composite | 16 |
| `savanna_decomp.py` | dry-season GPP by canopy component (counterfactual removals) | 16 |
| `savanna_water.py`, `savanna_stress.py` | soil water by layer, ET vs the ensemble, the stress the plants feel | 16 |
| `lever_test8.py` | LAI x0.5 / x2 (re-equilibrates: measured nothing), vmax x0.5, stressfac | 16 |
| `lever_test9.py` | monthly displayed LAI, drop factors and stress under the trigger | 16 |
| `lever_test10.py` | `vmax_mix` end to end: dead entry off, live entry on | 17 |
| `lever_test11.py`, `lever_test11b.py` | GPP vs LAI at one month (11: scaled `lai`, erased daily -- measured nothing; 11b: scaled `plai`) | 16 |
| `lever_test12.py` | trigger + `scale_leaf_only`: the causal chain restored | 16 |
| `lever_test20.py` | bounding C4 out of the cool steppes: cells go to deciduous shrubs, never C3 | 16 |
| `lever_test21.py` | C3 grass seeded at vmax 16-35: takes 0-11 of 60 steppe cells, summer deficit unmoved | 16 |
| `lever_test22.py` | lower-canopy cover 0.25-1.0 at fixed leaf carbon: scales every month alike | 16 |
| `neg_et_scan.py` | negative monthly ET in a hindcast: extent, bands, months, worst cells | 15-16 |
| `psurf_probe.py` | surface-pressure fix: worst cells + plateau/sea-level controls, 2 yr | 16 |

Reusable tools graduated out of here: `fu_vs_tree.py`, `et_vs_modis.py`,
`et_obs.py`, `tower_phase.py`, `tower_et.py`, `make_start_yaml.py`,
`finalize_fit.sh` (all in `verify/calibration/`).

Note (2026-09-21): `lever_test10.py` and `lever_test12.py` flip
`stomata_params.vmax_mix` / `scale_leaf_only` with `eqx.tree_at`. Those two
fields are now declared `eqx.field(static=True)` (plain `jax.jit` traced them
as leaves and failed on the Python `if`), and `tree_at` cannot modify a static
field. To re-run either script, rebuild the params from a yaml with the switch
set (`stomata: vmax_mix: true`) or use `dataclasses.replace` on the
`StomataParams` object instead.

## Later stages (CALIBRATION.md 19-75)

The same rule holds: one-off diagnostics archived as run, paths hard-coded, the `.txt` files are the outputs the sections
quote. Where a stage grew its own folder, the folder is dated by the day it opened.

| script or folder | question | section |
|---|---|---|
| `lever_test13.py` .. `lever_test19.py` | the mid-latitude winter and the C4 steppes: climate envelope, PFT removals, month by month, GPP per leaf, daily temperature, the hourly energy balance | 16 |
| `stageB_rank.py`, `c1_report.py`, `canopy_plateau_probe.py`, `light_response.py`, `site_diag_year.py`, `site_diag_plot.py`, `magnitude_ab.py`, `canopy_reference.py`, `model_hourly_lr.py`, `tower_light_response_hourly.py`, `site_replay_report.py`, `site_daily_replay.py`, `site_ab_monthly.py` | the site-fit stage: ranking the perturbations, the summer plateau, the canopy light response, the held-out verdicts | 22-29 |
| `controlled_4way.py` (+ `_pass1/2.txt`) | the controlled comparison of the review: same cells, same years, same start | review 2026-09-22 |
| `et_components_by_class.py`, `et_decomp1.py`, `et_lever1.py` .. `et_lever3.py` (+ `.txt`), `midlat_season.py`, `tropical_phase.py` | the ET stage: which component carries each class's bias, the Ball-Berry, soil-beta and interception levers; the 30-60 N amplitude and the tropical phase lag | 30-33 |
| `site_level_figures.py`, `site_level_figures_extra.py`, `site_level_figures_3runs.py`, `s2_report.py` | the site-level state as figures; the per-tower fits against the S1 set | 34-35 |
| `nee_fit_report.py`, `nee_fit_figures.py`, `nee_daynight_report.py`, `pool_budget_cycle.py` | the NEE stage: night/day verdicts, the tower carbon budget | 36 |
| `s3_check_report.py`, `ch4_fit_report.py`, `ch4_fit_figures.py`, `crop_fit_report.py` | class medians; the methane and cropland tower fits | 37-38, 41 |
| `global_figures.py`, `global_spatial.py`, `mip_compare.py`, `wad2m_precip_regression.py`, `wad2m_storage_regression.py` | the global run against the budgets and the gridded products (the three are also the scoring chain's) | 39, 43-44 |
| `pft_koppen_map.py`, `rescore_report.py`, `capacity_pixel.py`, `laicap_compare.py`, `gpp_chain.py`, `gpp_tower_cells.py`, `lai_npp_decomp.py`, `leaf_per_gpp.py`, `gate_calib.py`, `gate_compare.py` | the Köppen-zone types, the re-score, the leaf-area stage and the first tree gate | 45-57 |
| `fire_f7_fit.py`, `fire_2026-10-28/` | the closed-canopy fire damping; the fire terms by region and the zone objective | 59, 65-66, 71 |
| `audit_2026-10-26/` | the audit after the fire defects: carbon closure, water balance, wood by chunk, the Amazon hot spot, cold boxes, soil bands, the vmax mix | 60-68 |
| `gate_2026-10-27/` | the survival gate as a cover limit: the cover-stress curve, the ramp per zone, the expected equilibrium, cover and wood checks, elimination by year | 70-71 |
| `soil_2026-10-28/` | the soil-carbon stage: the parameter build, the expectation, soil C against HWSD2, the pools, the trajectory | 72 |
| `methane_2026-10-28/` | the methane bands on the final state: the factor decomposition against GMB 2025 | 73 |
| `release_2026-10-28/` | the release figure set and scorecard (its README gives the order) | 74-75 |
