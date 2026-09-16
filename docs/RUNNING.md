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

GPP is 1.01x FLUXCOM on matched cells. **Use this for carbon, water and
methane.**

**Do NOT use it for fire without re-calibrating.** Burned area falls
0.90x -> 0.78x GFED5 because the fire parameters were fitted against a fuel
load ~1.7x too high. Fire work should start from `spinup_cold_v10` until that
re-calibration is done, and should not treat the old 0.92x agreement as
validation.

### 2a. The previous reference state

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

### 2a. The superseded reference state

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

`FireParams` defaults: VPD danger **on**, combustion completeness **on**,
per-PFT mortality **on**, human ignition off, fine fuel off.

---

## 5. Gotchas that have cost real time

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
