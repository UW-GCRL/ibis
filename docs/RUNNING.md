# Running IBIS-JAX

How to actually run things. For paths, pinned versions and the CIFS storage
constraints, see [ENVIRONMENT.md](ENVIRONMENT.md); for known defects, see
[DEFECTS_FOR_REVIEW.md](DEFECTS_FOR_REVIEW.md).

**Read [§0](#0-before-you-trust-any-output) before using any archived output.**

---

## 0. Before you trust any output

Run this first. It verifies the one blessed model state end to end — no NaN,
physical ranges, convergence — and prints what produced it:

```bash
cd /shared/nas2/Hangkai/ibis-port
PYTHONPATH=python-port:. /shared/nas2/Hangkai/ibis-env/ibis-python \
    verify/reference_run.py
```

Exit code 0 means pass. Add `--skill` to also recompute the GFED5 fire metrics.

**Do not start from any other archived spin-up.** Four 200-year runs on this
NAS have `totlit = totcsoi = 0`, climate normals of exactly zero, and `exist`
either 1.000 (all 12 PFTs everywhere, tropical evergreen in Siberia) or 0.000
(no PFT anywhere); two contain NaN. Table in DEFECTS_FOR_REVIEW.md §1.

---

## 1. Two environments, and which to use

| | path | use for |
|---|---|---|
| **CPU** | `/shared/nas2/Hangkai/ibis-env/ibis-python` | tests, analysis, anything needing `jax_enable_x64` |
| **GPU** | `/home/hangkai/.venvs/ibis-gpu/bin/python` | spin-ups and multi-month runs (CUDA jax) |

Always set `PYTHONPATH=python-port:.` from the repo root — `verify/` and
`python-port/` are separate roots.

```bash
cd /shared/nas2/Hangkai/ibis-port
PYTHONPATH=python-port:. /home/hangkai/.venvs/ibis-gpu/bin/python -u <script>
```

The CPU env has **no** CUDA jaxlib and prints a warning to that effect; that is
expected, not a fault.

---

## 2. The canonical reference state

**Since 2026-09-11 the reference is the v11 pair** (CALIBRATION.md 12). The
state described under 2a below, `both_fix`, was the reference before that and
is superseded.
```
/shared/nas2/Hangkai/ibis/verify_outputs/v11_spinup/      cold start 1601-1900 on IBIS DEFAULT parameters,
                                                            1901-1930 GSWP3 cycled, 296 ppm  (cfg.yaml there)
/shared/nas2/Hangkai/ibis/verify_outputs/v11_transient/   1901-2010 real years, transient CO2
  restart/ibis_restart_2010.npz                           the state EVERY fit, validation and site run starts from
```
It carries no empirical correction (no vmax x0.54, no soil-OM x0.67). Its
totals on IBIS defaults are the S0 corrected baseline of CALIBRATION.md 20
(GPP 183 PgC on true land, vegetation C 580, soil C 13.8). Since 2026-09-21
the model has two forcing fixes (phenology clock, surface pressure) that every
state saved before that date lacks; states are not migrated -- the first hourly
step overwrites the affected fields.

### 2a. The previous reference state (both_fix, 2026-09-02) -- superseded
```
/shared/nas2/Hangkai/ibis/verify_outputs/both_fix/
  restart/ibis_restart_2200.npz     the blessed state
  out/ibis_yearly_{1901..2200}.nc
  params.yaml  cfg.yaml             (repo copies: configs/params_corrected_carbon.yaml,
                                     configs/spinup_corrected_carbon.yaml)
```

300 years from `spinup_cold_v10` with the corrected carbon cycle -- `vmax`
x0.54 and soil-OM loss rates x0.67, both **empirical corrections** documented in
docs/GPP.md. 0 NaN, soil-carbon drift +0.003 %/yr.

| quantity | value | observed |
|---|---|---|
| GPP | 138.2 PgC/yr | 120-130 |
| NPP | 63.8 | 55-60 |
| Rh | 57.9 | 49-58 |
| ET | 513 mm/yr | 500-550 |
| soil C | 12.14 kgC/m2 | 10-15 |
| litter | 0.60 | 0.4-0.8 |
| vegetation C | 481 PgC | 450-650 |

GPP is 1.01x FLUXCOM on matched cells, bought with the two empirical
corrections; **superseded by v11 for carbon, water and methane** (kept as a
record; the September PI deck used its numbers).

**Do NOT use it for fire without re-calibrating.** Burned area falls
0.90x -> 0.78x GFED5 because the fire parameters were fitted against a fuel
load ~1.7x too high. Fire work should start from `spinup_cold_v10` until that
re-calibration is done, and should not treat the old 0.92x agreement as
validation.

### 2b. The reference before that (spinup_cold_v10, 2026-08-31) -- superseded

```
/shared/nas2/Hangkai/ibis/verify_outputs/spinup_cold_v10/
  restart/ibis_restart_2200.npz    the blessed state (year 300)
  output/ibis_yearly_{1901..2200}.nc
  cold_v10.yaml                    the config that produced it
  run.log
  output_olddiag/, restart_olddiag/  superseded -- see README.txt there
```

300-year cold start, global 0.5 deg, 95 333 land points, fire on
(seasonlength + VPD + combustion completeness), 12 m thermal column
(`nbedrock=3`), `nsoilcall=10`. 258 min on the RTX 4090. **0 NaN, soil carbon
drift +0.017 %/yr over the last 50 years.** Reproduce with
`configs/spinup_cold_v10.yaml`.

**Use this rather than `spinup_fire_v8`.** v8 and its whole v7 ancestry were
produced through `verify/`'s grid builder, which duplicated 75% of its cells
and gave an effective 1 deg resolution behind a 0.5 deg label
(docs/DEFECTS_FOR_REVIEW.md). Their global totals are not inflated by that, but
their spatial detail is coarser than labelled, and they predate the NEE and
total-vs-above-ground-NPP fixes. They also cannot be loaded any more: `npoi`
changed 88 484 -> 95 333.

### 2c. The reference before those (spinup_fire_v8, August) -- superseded, not loadable

```
/shared/nas2/Hangkai/ibis/verify_outputs/spinup_fire_v8/
  checkpoints/state_latest.pkl     the blessed state (step 200)
  checkpoints/state_step{025..200}.pkl
  spinup_log.csv                   per-year biomass, exist, disturbf, C pools
  run.log
```

200 years cycling GSWP3 1901–2000, fire on (seasonlength + VPD + combustion
completeness + per-PFT mortality), `nsoilcall=10`. 208 min on the RTX 4090.
**0 NaN, converged, and it carries its PRNG key** — so a restart reproduces
exactly, which no earlier checkpoint does.

Load it:

```python
import pickle
ck = pickle.load(open(".../state_latest.pkl", "rb"))
state, key, step = ck["jax_state"], ck["key"], ck["step"]
```

---

## 3. Common tasks

### Run the test suite
```bash
cd python-port
PYTHONPATH=. /shared/nas2/Hangkai/ibis-env/ibis-python -m pytest tests/ -q
```
~4.5 min, 624 tests. `tests/test_no_dead_code.py` fails if a module or config
flag stops being reachable from the model — see [AUDIT.md](AUDIT.md).

### A new spin-up
```bash
PYTHONPATH=python-port:. nohup /home/hangkai/.venvs/ibis-gpu/bin/python -u \
  verify/timestepping/pipeline/spinup_fire_v8.py \
  --nyears 200 --cycle 1901 2000 --nsoilcall 10 --ckpt-every 25 --isimfire 1 \
  > run.log 2>&1 &
```
~3.5 h. Use `nohup` — it outlives your shell. It halts and names the cell if a
NaN appears, rather than logging `nan` for the rest of the run.

**`--ckpt-every 25` is deliberate.** Earlier runs wrote 200 checkpoints each and
consumed 250 GB; only the last is needed to restart, and `per_month_stats/`
already holds the trajectory.

### Fire evaluation against GFED5
```bash
PYTHONPATH=python-port:. /shared/nas2/Hangkai/ibis-env/ibis-python \
  verify/fire/evaluate_gfed5.py              # add --calibrate to fit
```
Reference values on `both_fix` (FIRE.md §9): burned area 3.59% against GFED5
4.60% -- inside the 2.38-4.60% product spread -- rank +0.742 all / +0.376
burning, NMSE 0.754, mean |lat| 10.9 against GFED5's 10.8. The earlier ×0.92
figure rested on a 1.7× fuel-load bias and is retired.

**`both_fix` is superseded by v11d** (CALIBRATION.md §8-12). On v11d, no fire
parameter changed: burned area **3.10%** (inside the 2.38-4.60% spread),
rank +0.759 all / +0.365 burning, NMSE 0.814, mean |lat| 10.1 against 10.8.

### Methane evaluation against the Global Methane Budget
```bash
PYTHONPATH=python-port:. /home/hangkai/.venvs/ibis-gpu/bin/python \
  verify/methane/evaluate_global.py --config all
```
~12 min. On `both_fix` with `--config biome_tref` (the production config,
METHANE.md §5q): 182 Tg/yr against GMB 158 [119-203], shape error 11.0,
latitudinal split 72.7 / 19.8 / 1.0 % against 70 / 21 / 8. On **v11d**
(CALIBRATION.md §12, METHANE.md §5r): `biome_tref` 251 Tg, split
53.9 / 30.6 / 10.3, peak 7; `biome_noq10` now has the best shape (10.3).
The Arctic deficit was `vmax x0.54`, not structure. Magnitude tracks Rh
(x1.4); do not fit `r_CH4` to it.

---

## 4. Configuration that actually does something

`ibis/config.py` `RunConfig`. Flags verified live (AUDIT.md):

| flag | default | effect |
|---|---|---|
| `isimveg` | 1 | 0 = static vegetation, 1 = dynamic (`dynaveg`) |
| `isimfire` | **0** | 0 = no fire, disturbance fixed at 0.005. **Fire is off by default** |
| `fire_scheme` | `seasonlength` | vs `legacy` (F77-exact) |
| `enable_fire_emissions` | False | trace gases; needs `isimfire=1` |
| `isimco2` | 0 | fixed vs varying CO₂ |
| `enable_permafrost` | False | **unwired** — the module exists but nothing reads the flag |
| `forcing.koppen_file` | "" | path of the Köppen npz (`scripts/koppen_prep.py`): mask on PFT existence by climate class and period (CALIBRATION.md 44) |
| `physics.larch_gdd_fix` | False | PFT 8 (larch) existence on the degree-day limit the F77 comment intends; False = the F77 code (never true) |
| `physics.soil_organic` | False | mix the Lawrence & Slater 2008 organic end-member by the HWSD2 organic fraction; needs `orgcpct`/`bulkdens` in `soil_file` (`scripts/hwsd2_organic_to_nc.py`) |
| `pft.tree_stress_min` (params yaml, not RunConfig) | -1 (off) | >= 0 activates the water-limited tree-establishment gate (CALIBRATION.md 14): the state gains the 3-leaf `StressAccum`, `dynaveg` gets a survival factor, and the run log says `Tree-establishment gate ACTIVE` or `inactive` -- read that line to know which model produced a state |
| `stomata.vmax_mix` (params yaml) | false (F77) | true: each canopy big leaf takes the displayed-LAI-weighted mean vmax of its PFTs instead of ONE PFT's by existence priority; F77 leaves four `vmax_pft` entries dead (CALIBRATION.md 17). On in every fitted configuration from stage (b2) |
| `stomata.scale_leaf_only` (params yaml) | false (F77) | true: canopy scaling divides by k(LAI+SAI) instead of kLAI, so stems attenuate light but do not assimilate (old scale x LAI/(LAI+SAI)); F77 credits stem-absorbed PAR to the leaves, +25 % upper-canopy GPP and no response to leaf drop (CALIBRATION.md 16). Takes global GPP 168 -> 131 PgC on the (b1) parameters |
| `stomata.acclim_light`, `frost_on`, `frost_t0`, `frost_t1` (params yaml) | false, false, -5, 5 | acclim_light: the acclimation factor of each big leaf also scales its light-limited rate (quantum yield), not only Vmax; frost_on: a scalar on the day's mean air temperature, 0 at <= frost_t0 C and 1 at >= frost_t1 C, on the acclimated PFTs (CALIBRATION.md 53). Off: bit-identical. On in `params_31d.yaml` |
| `pft.<pft>.lai_max`, `crop.lai_max`, `types.<id>.lai_max` (params yaml) | unset (F77, no cap) | per-type maximum canopy LAI per unit of the layer's cover: each year a layer's leaf pools are held to its leaf-area-weighted mean cap, the excess leaf carbon goes to wood and roots (awood:aroot); `crop.lai_max` is the cap on the cropland share of the lower canopy (unset: the grass cap). `verify/calibration/laicap_yaml.py IN OUT` writes the literature set (CALIBRATION.md 46) |
| `physics.wetland_dynamic` + `forcing.wetland_clim_file`, `wetland_area:` (params yaml) | false (WAD2M climatology) | dynamic inundated fraction from the model's own root-zone water (METHANE.md 7): f = clip(f_clim (1 + beta A), f_min, f_max), A the standardised anomaly against the climatology file (`verify/methane/wetland_clim_prep.py` on a reference hindcast), bounds the cell's WAD2M range, 0 where the top layer is frozen, beta per band (`wetland_area: {tropical, temperate, boreal}`; 0 = the bounded climatology). Needs `enable_methane`. Writes `finund` in the monthly files |

`FireParams` defaults: VPD danger **on**, combustion completeness **on**,
per-PFT mortality **on**, human ignition off, fine fuel off.

---

## 5. Gotchas that have cost real time

* **`imonthout: 1` costs ~4x yearly output -- and the first year misleads.**
  Measured on a 95,333-cell run: steady state ~16 s per monthly file, ~3 min
  per model-year, against ~50 s without monthly output; ten years ~35 min.
  The first 12 months took ~12 min and, read alone, suggested 15x -- that was
  JIT, and an earlier version of this note said 2.7 h for ten years on the
  strength of it. For a seasonal diagnostic that does not need the full
  field, `verify/calibration/seasonal_baseline_sample.py` gets monthly GPP
  from `one_year` on the stratified sample with no writer.

**`surta.nc` is a BINARY mask: 95,333 cells at landfrac 1.0 = 165 Mkm2 of "land"** (Earth: 149). It was cut as "any vegetation fraction > 0" from `vegtype_frac_1992.nc`, whose 15 IBIS-type fractions sum to the true land fraction (45,651 partial cells; open-water archipelago cells with a speck of island count as full land). Weight any global total by `verify/calibration/landmask.true_landfrac()` (143.7 Mkm2), or read `state_report`'s "TRUE land" rows; the per-m2 fluxes are fine (CALIBRATION.md 18).

**Surface pressure reaches the physics only since 2026-09-21.** The GSWP3
driver always read `ps`, but the hourly atmosphere update wrote every
forcing field except `psurf`, so the canopy and soil physics ran at the
creation value 101325 Pa on every cell (every restart before that date has
`leaf_110` = 101325 everywhere). High cells condensed instead of
evaporating (Himalaya -200 mm/month), plateaus ran with stomata too wide.
Fixed in `loops_gswp3._step_body`; nothing to migrate, the first hourly step
overwrites the value. Runs and fits from before the fix are on the old
pressure -- DEFECTS_FOR_REVIEW.md "sea-level pressure".

**Restart leaf counts: 258 without the tree gate, 261 with it.** `load_state`
migrates both ways with a warning -- it fills the three trailing accumulator
leaves from the template, or drops them when a gated restart meets an
ungated template (the fire, methane and report tools). A gated run restarted
without the gate simply loses its accumulator.

**Checkpoints predating F1b** lack `nfirestep_vpd`/`sdry_vpd`. `dynaveg` raises
a message naming the migration rather than a bare `AttributeError`.

**`global_gswp3_test` enables x64 at import.** `run_spinup_gswp3` runs float32,
and `soilbgc` would return float64 into the `lax.scan` carry. Turn x64 off
*after* that import — every pipeline script does this already.

**Cell ordering.** Build the grid with `build_global_cell_grid(0.5, 0.5)`;
do not slice `surta.nc` by index. They disagree for 88,180 of 88,484 cells.

**CIFS has no chmod**, so `git push` prints `chmod on .git/config.lock failed`.
Harmless — verify with `git ls-remote`.

**419 `._*.py` AppleDouble files** sit in the tree (macOS resource forks,
untracked). Inert, but they hide from bash globs since `*` skips dotfiles, and
they break `Path.read_text()` without an explicit encoding.

**Storage.** `/shared/nas2` is shared and ~87 % full. A 200-year run is ~1.5 GB
at `--ckpt-every 25`.

---

## 6. Where the science is written down

| file | contents |
|---|---|
| [FIRE.md](FIRE.md) | fire module: design, provenance audits, defects, GFED5 results |
| [METHANE.md](METHANE.md) | methane: tiered design, literature survey, benchmarks, M1–M2 results |
| [AUDIT.md](AUDIT.md) | dead-code audit and its resolution |
| [DEFECTS_FOR_REVIEW.md](DEFECTS_FOR_REVIEW.md) | **for collaborators** — which archived outputs are unusable and why |
| [ENVIRONMENT.md](ENVIRONMENT.md) | paths, pinned versions, storage constraints |
