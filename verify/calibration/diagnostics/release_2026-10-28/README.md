# Release figures and tables of the IBIS 3.0 final state (s31t, 2026-10-28)

The scripts that produced the final figure set and the scorecard of CALIBRATION.md 74 (and the verdict tables of 67-73),
copied from the working folders as they ran. They read the protocol outputs under `verify_outputs/final_<tag>/` and the
reference products listed in DATA_INVENTORY.md, and they carry the paths of our machine: `M` (the folder that holds the
extracted 0.5-degree grids `glob_<tag>.npz`, `glob_refs.npz`, `hwsd2_soc1m.npy`) and `FIG` (the output folder) are set at
the top of each file; change them before running elsewhere.

Order of the pipeline (python -I; the gpu venv for the extractions, an environment with matplotlib for the plots):

1. `extract_global.py refs` once, then `extract_global.py <tag>`: 2006-2010 hindcast means and the 2010 pools on the grid.
2. `plot_global.py <reference tag> <final tag>`: gpp_maps, et_maps, gpp_zonal_season, nee_maps, lai_biomass, cover_zone,
   ch4_global, fire, and `metrics_<a>_<b>.json`.
3. `plot_soil.py OUT.png <tags...>`: soil C against HWSD2 0-1 m (`hwsd2_soc1m.npy` = orgcpct x bulk density x thickness,
   layers 1-4 of `soita_hwsd2_organic.nc`, as `soil_ref_bands.py` builds it).
4. `amazon_extract.py <tag>` then `plot_amazon_final.py OUT.png <reference> <final>`.
5. `wood_regions.py OUT.png <tags...>`: wood by Köppen group and continent against Xu & Saatchi 2021.
6. The verdict tables: `soil_ref_bands.py`, `soil_pools_by_group.py`, `soil_traj_protocol.py` (72); `cover_wood_check.py`,
   `wood_budget_chunks.py`, `elimination_by_year.py`, `wood_traj_protocol.py` (70-71); `../methane_2026-10-28/ch4_bands_production.py` (73).

Outputs of the release: the reading folder `ibis_local_document/figures/final_2026-10-26/` (figures, `tables/`, `scripts/`),
with `tables/scorecard_s31n_s31t.txt` holding every headline number and its source.
