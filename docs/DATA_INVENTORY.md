# Data and code inventory

Everything needed to run and modify IBIS-JAX, for someone starting from
nothing on the chenlab box. All paths below are world-readable (`drwxrwxrwx`),
so no permission requests are needed.

Read `docs/RUNNING.md` next for how to run, and `docs/ENVIRONMENT.md` for the
Python environments. This file is only the *where*.

**Total data footprint: ~350 GB.** Nothing here should be copied — it is on
shared storage and everyone reads the same paths. The only thing you clone is
the code.

---

## 1. Code

| what | where |
|---|---|
| Repository | `git@github.com:UW-GCRL/ibis.git` |
| **Branch to use** | **`fire-test`** |
| Working copy on the box | `/shared/nas2/Hangkai/ibis-port` (66 MB) |

`main` does **not** have this work -- it lacks the grid fix, the carbon
correction, the fire and methane modules, and the calibration. Use
`fire-test`.

    git clone git@github.com:UW-GCRL/ibis.git
    cd ibis && git checkout fire-test

Layout:

| path | lines | what |
|---|---|---|
| `python-port/ibis/` | 16,284 across 67 files | the model |
| `python-port/scripts/` | | `run_global.py` is the entry point |
| `python-port/configs/` | | run configurations |
| `python-port/tests/` | | pytest suite, incl. the wiring guards |
| `verify/` | | diagnostics and evaluation scripts |
| `docs/` | ~6,100 | why things are the way they are -- read these |
| `fortran/ibis_2.6b4/` | | the original F77, for checking faithfulness |

---

## 2. Climate forcing

| dataset | path | size | notes |
|---|---|---|---|
| **GSWP3** | `/shared/nas2/Min/GSWP3` | **276 GB** | 1901-2010, 3-hourly, 0.5 deg, 110 decade files |

Variables: `tas tasmax tasmin pr rsds rlds huss ps rhs wind`. `ps` reaches the canopy and soil physics only since 2026-09-21 (before that the physics ran at 101325 Pa everywhere; DEFECTS_FOR_REVIEW.md "sea-level pressure").
Files are decade blocks, e.g. `tas_gswp3_1901_1910.nc4`.

**It ends in 2010.** That is the binding constraint on any evaluation against
recent observations -- see `docs/CALIBRATION.md` 8. GSWP3-W5E5 exists at
`/shared/nas2/Chuanfeng/gswp3-w5e5` and runs to 2019, but carries only
`tas/pr/hurs/sfcwind` -- no radiation -- so it cannot drive IBIS without
downloading more variables.

---

## 3. Boundary and input data

| dataset | path | size | used for |
|---|---|---|---|
| **Land mask** | `/shared/nas2/Min/HWSD/surta.nc` | 37 KB | `landfrac`, 95,333 land cells, 165.0 Mkm2 |
| **Soil texture** | `/shared/nas2/Min/HWSD/soita.nc` | 2.6 MB | sand/clay by layer |
| **Vegetation type** | `/shared/nas2/Min/ESACCI_LC/ibis_vegtype` | 56 MB | cold-start PFT init, 29 years (1992-2020) |
| ESACCI raw | `/shared/nas2/Min/ESACCI_LC` | 63 GB | 300 m land cover, source for the above |
| **Wetland extent** | `/shared/nas2/Hangkai/ibis/inputs/methane/wad2m` | 571 MB | WAD2M `Fw`, monthly 2000-2020 -- methane forcing |
| **Learned wetland area** | `/shared/nas2/Hangkai/ibis/inputs/methane/wetland_learned/wetland_learned_v2_s31t.npz` | 15 MB | the release's wetland-area model (`wetland_source: learned`; METHANE.md 11): per-band XGBoost models as JSON, per-cell WAD2M climatology, cell maximum, emulator climatology and static predictors; trained by `verify/methane/wetland_learned_train.py` on `final_s31t_ch4h` (2001-2010). The first, rejected file is kept beside it (`*.REJECTED_*`) |
| **Topography for the wetland training** | `/shared/nas2/Hangkai/ELM Edge Effects/inputdata/surfdata_0.5x0.5_CH_v1_all_forests.nc` | | CLM 0.5-degree surface data: `FMAX`, `SLOPE`, `STD_ELEV`, `TOPO`, `PCT_WETLAND`, `PCT_LAKE`, `binfl`, read only by the training script (the runtime takes them from the learned file) |
| **Köppen-Geiger classes** | `/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz` | 0.1 MB | Beck et al. 2023 (`/shared/nas/Public/Climate_Biome/Koppen_Beck2023`, 0.5-degree per period 1901-30, 31-60, 61-90, 91-2020), packed by `python-port/scripts/koppen_prep.py`; `classes` (4, 360, 720) uint8 N->S, 0 = none; mask on PFT existence (CALIBRATION.md 44) |
| **Soil organic carbon, bulk density** | `/shared/nas2/Hangkai/ibis/inputs/soil/soita_hwsd2_organic.nc` | ~8 MB | HWSD2 ORG_CARBON (%), BULK (g cm-3) plus sand and clay on the six IBIS layers, 0.5 degree N->S, by `python-port/scripts/hwsd2_organic_to_nc.py` (Min's mapping-unit rule); `physics.soil_organic` (CALIBRATION.md 44) |
| **Population** | `/shared/nas2/Hangkai/ibis/inputs/fire/popdens` | 252 MB | CLM population stream -- human ignition (OFF by default) |
| Burned area | `/shared/nas2/Hangkai/ibis/inputs/fire/gfed5` | 307 MB | GFED5 -- fire *target*, not forcing |
| Emission factors | `/shared/nas2/Hangkai/ibis/inputs/fire/emission_factors` | 36 KB | Andreae, per-species |

The `inputs/` tree has its own `README.md` per module stating provenance and
what was verified on download. `inputs/wiemip` (8.4 GB) belongs to a separate
protocol and is not needed here.

---

## 4. Evaluation data (targets, never forcing)

| dataset | path | size | constrains |
|---|---|---|---|
| **FLUXCOM GPP** | `/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc` | 80 MB | gridded GPP. **Units are g m-2 day-1** -- multiply by 365.25/1000 |
| **MODIS ET (MOD16-derived)** | `/shared/nas2/Hangkai/ibis/inputs/eval/et/MODIS_et_0.5x0.5.nc` | 29 MB | ET target used first (`et_vs_modis.py`); known high bias in arid land (ET > P there). `et` in `kg/m2/s`, 2000-01..2013-12 monthly, 0.5° grid, **S->N, must be flipped** |
| MOD16A2 ET | `/shared/nas2/Hangkai/ibis/inputs/eval/et/MOD16A2_et.nc` | 373 MB | the newer MOD16 collection, cross-check. `et` in `kg/m2/s`, 2000-01..2014-12 monthly, 0.5° grid, **S->N, must be flipped** |
| GLEAM v3.3a ET | `/shared/nas2/Hangkai/ibis/inputs/eval/et/GLEAMv3.3a_et.nc` | 970 MB | satellite-forced Priestley-Taylor, independent of MOD16. `et` in `kg/m2/s`, 1980-01..2018-12 monthly, 0.5° grid, **S->N, must be flipped** |
| DOLCE ET | `/shared/nas2/Hangkai/ibis/inputs/eval/et/DOLCE_et.nc` | 91 MB | tower-weighted ensemble of gridded ET products. `hfls` in `W m-2`, 2000-01..2009-12 monthly, 0.5° grid, **S->N, must be flipped** |
| FLUXCOM LE | `/shared/nas2/Hangkai/ibis/inputs/eval/et/FLUXCOM_le.nc` | 871 MB | latent heat from the same machine-learning family as the GPP target; LE / 2.45e6 J/kg -> mm. `le` in `watt/m2`, 1980-01..2014-12 monthly, 0.5° grid, **S->N, must be flipped** |
| WECANN LE | `/shared/nas2/Hangkai/ibis/inputs/eval/et/WECANN_hfls.nc` | 13 MB | solar-induced-fluorescence-constrained latent heat, cross-check. `hfls` in `W/m^2`, 2007-01..2015-12 monthly, 1.0° grid |
| **ET ensemble (derived)** | `/shared/nas2/Hangkai/ibis/inputs/eval/et/et_ensemble_0.5deg.npz` | 28 MB | the five 0.5-degree products above reduced to 2001-09 annual means (mm/yr) and monthly climatologies (mm/day) on the IBIS N->S grid, plus ensemble mean/min/max and product count; built by `verify/calibration/et_obs.py`. Score on cells with `ens_n == 5` |
| **MODIS PFT LAI** | `/shared/nas2/Hangkai/ELM Edge Effects/inputdata/MODISPFTLAI_0.5x0.5_v1_all_forests.nc` | 2.5 GB | leaf area; needs the surfdata file below for PFT weights |
| **CLM surfdata (tree fraction)** | `.../surfdata_0.5x0.5_CH_v1_all_forests.nc` | 2.4 GB | `PCT_NAT_PFT`, `PCT_NATVEG`. **S->N, must be flipped.** Also the fourth calibration observation: PFTs 1-8 of `PCT_NAT_PFT` = tree fraction of the natural-vegetation part; this is the no-crop `all_forests` variant (`PCT_CROP` = 0), i.e. potential vegetation like IBIS's own -- `fit_carbon.tree_at`, `fu_vs_tree.py`, CALIBRATION.md 14 |
| ILAMB fields | `python-port/data/ilamb_obs/` | small | pre-processed `gpp/lai/cveg/csoil` means |
| **FLUXNET2015** | `/shared/nas2/Min/FLUXNET2015/fluxnet_compact` | 810 files | ~200 sites, GPP/RECO/LE/NEE at HH/DD/MM/YY |
| **FLUXNET-CH4** | `/shared/nas/Chenchen/AmeriFLuxet/combined` | 65 sites | Chenchen's processing: RF-gapfilled FCH4 + uncertainty |
| **ICOS Warm Winter 2020** | `/shared/nas2/Hangkai/ibis/inputs/sites/ICOS_WW2020` | 31 GB | 73 stations 1989-2020, FLUXNET format, CC-BY-4.0, doi:10.18160/2G60-ZHAK; 39 new, 34 extend FLUXNET2015 sites; acquired 2026-10-13 |
| **ICOS ETC L2 archive 2026-1** | `/shared/nas2/Hangkai/ibis/inputs/sites/ICOS_L2` | 23 GB | 93 stations, CC-BY-4.0, doi:10.18160/S6TB-567H; 83 with ONEFlux daily + half-hourly, 10 urban/water half-hourly only (`halfhourly_L2_only/`) |
| **FLUXNET v1.3 (ONEFlux, FLUXNET Shuttle)** | `/shared/nas2/Hangkai/ibis/inputs/sites/FLUXNET_v1.3` | 102 GB | 512 sites, 4,365 site-years (ICOS, EuroFlux, JapanFlux, ChinaFlux, KoFlux, SAEON, 140 AmeriFlux incl. NEON Alaska, 19 TERN), CC-BY-4.0, no login; FLUXNET2015 column names; no AUXNEE/AUXMETEO; half-hourly extracted for the 301 sites of `halfhourly_sites.txt` |
| **New-tower inventory** | `/shared/nas2/Hangkai/ibis/inputs/sites/new_sites_inventory.csv` | 679 rows | one row per site x source, 524 distinct sites, 344 new; years with 50 % NEE, NEE variable, Köppen class/zone/PFT type, per-site citation; report `ibis_local_document/logs/tower_acquisition_2026-10-13.md` |
| FLUXNET-CH4 Europe | `/shared/nas/Chenchen/Eurofluxnet/EuroFlux` | 43 sites | |
| Site coordinates | `/shared/nas/Chenchen/hangkai/combined_site_coords.csv` | 108 sites | lat/lon/network for the above |

The methane tower data is **unpublished** (Chenchen's). Check with them before
using it in anything that leaves the group.

---

### Leaf-area reference products (2026-10-17, CALIBRATION.md 48)
`/shared/nas2/Hangkai/ibis/inputs/eval/lai/`
- `raw_glass_v61/<year>/GLASS01B01.V61.A<year><doy>.2025248.hdf` -- GLASS LAI V61 (MODIS-based), 0.05 deg, 8-day,
  2001-2010, 460 files, 1.2 GB, from https://glass.hku.hk/archive/LAI/MODIS/0.05D/ (public). Byte, LAI x 10, fill 255.
- `raw_ilamb_gimms_lai4g/cao2023_lai.nc` -- GIMMS LAI4g (Cao et al. 2023, doi:10.5194/essd-15-4877-2023), ILAMB's
  0.5-degree monthly copy, 1982-2020, 970 MB, from https://www.ilamb.org/ILAMB-Data/DATA/lai/GIMMS_LAI4g/.
- `raw_ilamb_modis/lai_0.5x0.5.nc` -- ILAMB "MODIS" LAI = NCAR's 2014 Collection-5 monthly climatology (not C6/C6.1).
- `glass_v61_0.5deg_monthly_2001-2010.npz`, `gimms_lai4g_0.5deg_monthly_2001-2010.npz` -- both on the model grid
  (north-up, 360 x 720), lai (10 years, 12 months, lat, lon); built by `verify/calibration/lai_products_prep.py`
  (run with the system Python, `/usr/bin/python3 -I`, which has GDAL with HDF4).

## 5. Model states you can start from

Spinning up from cold costs ~4.5 h on the GPU. Use a restart instead.

| state | restart file | what it is |
|---|---|---|
| **v11 transient** | `/shared/nas2/Hangkai/ibis/verify_outputs/v11_transient/restart/ibis_restart_2010.npz` | 55 MB. **Current reference.** IBIS default parameters, cold start -> 300 yr spin-up at 296 ppm -> transient 1901-2010 with rising CO2. GPP 232 PgC/yr (high, uncalibrated) |
| v11 spin-up | `.../v11_spinup/restart/ibis_restart_1900.npz` | end of the 300-yr spin-up, 296 ppm |
| both_fix | `.../both_fix/restart/ibis_restart_2200.npz` | 54 MB. **Superseded.** Carries `vmax x0.54` and soil-OM `x0.67`; GPP 138 PgC/yr but the Arctic is empty and restarts clobbered its climate normals |

Loading a restart needs a template with the **matching** `nbedrock`:

    create_ibis_state(GridConfig(npoi=95333, nbedrock=3))

`nbedrock=0` gives 257 leaves against the file's 258 and fails.

Full output trees (`verify_outputs/`, 302 GB total) hold yearly netCDF for
every run; you usually want only the restart.

---

## 6. Environment

| | |
|---|---|
| CPU (analysis, tests) | `/shared/nas2/Hangkai/ibis-env/ibis-python` |
| **GPU (model runs)** | `/home/hangkai/.venvs/ibis-gpu/bin/python` -- jax 0.11.0 + CUDA, RTX 4090 |
| Pinned versions | `ibis-env/requirements.lock` |

The GPU is shared. Check `nvidia-smi` before launching and set
`XLA_PYTHON_CLIENT_MEM_FRACTION` to what is actually free, with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

A model run at 95,333 cells costs **~53 s per simulated year** on the GPU and
is roughly 60x slower on CPU.

---

## 7. Where to start

1. `docs/RUNNING.md` -- how to run, and what to check before trusting output.
2. `docs/DEFECTS_FOR_REVIEW.md` -- what has been broken and how it was found.
   Most of it is the same pattern: a flag or field that is declared,
   documented, read, and never reaches the physics.
3. `python-port/tests/test_no_dead_wiring.py` -- the guards against that
   pattern. Run the suite before and after any change.
4. `docs/CALIBRATION.md` -- the current state of the calibration, including
   which conclusions have been withdrawn and why.

**One habit worth copying:** every quantity compared against observations here
states its normalising area and temporal convention. Eight wrong conclusions
in this project came from comparing a 165 Mkm2 domain mean against a 130 Mkm2
published range, an annual mean against an annual maximum, a g/m2/day product
against a kgC/m2/yr model field, or one year against a twenty-year mean.
