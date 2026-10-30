# Figures and tables of the IBIS 3.0 final stage (s31t, 2026-10-28)

The scripts that produced the final figure set and the scorecard of CALIBRATION.md 74, copied from the working folders
as they ran. They read the protocol outputs under `verify_outputs/final_<tag>/` and the reference products listed in
DATA_INVENTORY.md, and they carry the paths of our machine: `M` (the folder holding the extracted 0.5-degree grids
`glob_<tag>.npz`, `glob_refs.npz`, `hwsd2_soc1m.npy`; on the lab NAS `verify_outputs/diag_work/meeting`) and `FIG` (the
output folder) are set at the top of each file; change them before running elsewhere. The gpu venv runs the extractions; the plots need matplotlib (the CPU environment).

Order (python -I):

1. `extract_global.py refs` once, then `extract_global.py <tag>`: 2006-2010 hindcast means and the 2010 pools on the grid.
2. `plot_global.py <reference tag> <final tag>`: gpp_maps, et_maps, gpp_zonal_season, nee_maps, lai_biomass, cover_zone,
   ch4_global, fire, and `metrics_<a>_<b>.json`.
3. `plot_soil.py OUT.png <tags...>`: soil C against HWSD2 0-1 m (`hwsd2_soc1m.npy` = orgcpct x bulk density x thickness,
   layers 1-4 of `soita_hwsd2_organic.nc`, the construction of `../soil_2026-10-28/soil_ref_bands.py`).
4. `amazon_extract.py <tag>`, then `plot_amazon_final.py OUT.png <reference> <final>`.
5. The wood-by-region figure and table: `../audit_2026-10-26/wood_regions.py OUT.png <tags...>`.

The verdict tables of sections 70-73 come from the stage folders: `../gate_2026-10-27/` (cover limit and production:
`cover_wood_check.py`, `elimination_by_year.py`, `wood_traj_protocol.py`, `gate_cover.py`), `../audit_2026-10-26/`
(`wood_budget_chunks.py`, `closure_any.py`, `water_balance.py`), `../soil_2026-10-28/` (`soil_ref_bands.py`,
`soil_pools_by_group.py`, `soil_traj_protocol.py`) and `../methane_2026-10-28/ch4_bands_production.py`.

Outputs (lab NAS, not in the repository): the reading folder `ibis_local_document/figures/final_2026-10-26/` (figures,
`tables/`, `scripts/`). Its `tables/scorecard_s31n_s31t.txt` is the scorecard of the 28 October candidate (methane 170 Tg
before the hourly refit and the WAD2M fix); the release's numbers are in RELEASE_3.0.md section 5 and come from the
scoring chain on `final_rel30` (CALIBRATION.md 84).
