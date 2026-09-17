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

Variables: `tas tasmax tasmin pr rsds rlds huss ps rhs wind`.
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
| **MODIS PFT LAI** | `/shared/nas2/Hangkai/ELM Edge Effects/inputdata/MODISPFTLAI_0.5x0.5_v1_all_forests.nc` | 2.5 GB | leaf area; needs the surfdata file below for PFT weights |
| **CLM surfdata (tree fraction)** | `.../surfdata_0.5x0.5_CH_v1_all_forests.nc` | 2.4 GB | `PCT_NAT_PFT`, `PCT_NATVEG`. **S->N, must be flipped.** Also the fourth calibration observation: PFTs 1-8 of `PCT_NAT_PFT` = tree fraction of the natural-vegetation part; this is the no-crop `all_forests` variant (`PCT_CROP` = 0), i.e. potential vegetation like IBIS's own -- `fit_carbon.tree_at`, `fu_vs_tree.py`, CALIBRATION.md 14 |
| ILAMB fields | `python-port/data/ilamb_obs/` | small | pre-processed `gpp/lai/cveg/csoil` means |
| **FLUXNET2015** | `/shared/nas2/Min/FLUXNET2015/fluxnet_compact` | 810 files | ~200 sites, GPP/RECO/LE/NEE at HH/DD/MM/YY |
| **FLUXNET-CH4** | `/shared/nas/Chenchen/AmeriFLuxet/combined` | 65 sites | Chenchen's processing: RF-gapfilled FCH4 + uncertainty |
| FLUXNET-CH4 Europe | `/shared/nas/Chenchen/Eurofluxnet/EuroFlux` | 43 sites | |
| Site coordinates | `/shared/nas/Chenchen/hangkai/combined_site_coords.csv` | 108 sites | lat/lon/network for the above |

The methane tower data is **unpublished** (Chenchen's). Check with them before
using it in anything that leaves the group.

---

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
