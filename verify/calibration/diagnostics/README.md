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
| `notree_test.py`, `canopy_decomp.py`, `composition_test.py`, `regime_test.py`, `stress_test.py`, `vmax_experiment.py`, `tower_offseason.py` | where the sparse-cell excess sits (canopy, PFT, season) | 14 |
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
