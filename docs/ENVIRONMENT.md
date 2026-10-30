# Working environment

Single source of truth for where the code, the data and the Python environment
live, and for the constraints of the storage they live on. Referenced by
`docs/FIRE.md` and `docs/METHANE.md`.

**Last updated:** 2026-08-12

---

## Layout

| What | Path | Notes |
|---|---|---|
| **Git working copy** | `/shared/nas2/Hangkai/ibis-port` | **Authoritative.** Branches `main`, `calibration-s1` (the IBIS 3.0 release), `fire-test`, `methane-test`. |
| **Python environment** | `/shared/nas2/Hangkai/ibis-env` | Launcher `ibis-python`, `requirements.lock`, `site-packages/` (755 M). |
| **Run tree / outputs** | `/shared/nas2/Hangkai/ibis` | 374 G `verify_outputs/`, 8.4 G `inputs/` (Ramankutty & Foley 1999, WIEMIP), WIEMIP protocol. Not a git repo. |
| Older git copy | `/shared/nas/Chenchen/hangkai/ibis-port` | **Superseded** — identical content as of 2026-08-12 but will drift. `nas` is 97 % full. |
| GitHub remote | `git@github.com:UW-GCRL/ibis.git` | `main`, `calibration-s1`, `fire-test`, `methane-test`. |

Run anything with:

```sh
cd /shared/nas2/Hangkai/ibis-port/python-port
/shared/nas2/Hangkai/ibis-env/ibis-python -m pytest tests/ -q
```

---

## Storage constraints (both `/shared/nas` and `/shared/nas2`)

These CIFS mounts **cannot create symlinks and cannot change file modes**.
Exec permission is granted by default (everything is effectively 777), and
hardlinks work. This breaks a surprising number of standard tools:

| Tool | Failure | Workaround used |
|---|---|---|
| `python -m venv` | `[Errno 61] No data available: 'lib' -> '.../lib64'` | none possible — see below |
| `conda create --prefix --copy` | `[Errno 61] ... 'cacert.pem' -> '.../ssl/cert.pem'` | none possible |
| `pip install --target <nas2>` | `[Errno 1] Operation not permitted` (`shutil.copy2` preserves modes) | install to local disk, then `rsync --no-perms` |
| `git clone` | `chmod on .git/config.lock failed` | `rsync` the repo, then edit `.git/config` by hand |
| `git config` | same lockfile `chmod` failure | edit `.git/config` directly |
| `cp -a` / `rsync -p` | `Operation not permitted` setting perms | `--no-perms --no-owner --no-group --no-times` |
| `chmod` | `Operation not permitted` | set the mode in git's index: `git update-index --chmod=-x <file>` |
| pytest cache | cannot create `.pytest_cache` | `-p no:cacheprovider` (the launcher adds it automatically) |

**Consequence:** a self-contained interpreter cannot exist on this storage.
`ibis-env/site-packages` holds the packages; the *interpreter* must come from
somewhere else (default `/home/hangkai/miniconda3/bin/python3.12`, override with
`IBIS_BASE_PYTHON`). Packages are compiled for CPython 3.12 and the launcher
refuses any other minor version.

`core.fileMode` is set to `false` in both git copies. Without it, git reports
~444 phantom modified files from permission bits alone, which blocks branch
switching and pollutes commits.

---

## Pinned versions

```
jax 0.11.0 · jaxlib 0.11.0 · equinox 0.13.8 · jaxtyping 0.3.11
numpy 2.5.2 · scipy 1.18.0 · xarray 2026.7.0 · netCDF4 1.7.4
optax 0.2.8 · pyyaml 6.0.3 · pytest 9.1.1
```

Full pins in `/shared/nas2/Hangkai/ibis-env/requirements.lock`. CPU JAX only —
the RTX 4090 runs in `verify/STATUS.md` used a CUDA jaxlib, which would need
`jax[cuda12]==0.11.0` staged and rsynced the same way.

---

## ⚠ The verification caveat

These versions are **much newer than the ones the port was verified against**.
The 46-year F77↔JAX lockstep and the 200-year spin-up (`verify/STATUS.md`, work
ending 2026-06-11) ran under a `verify/.venv` that did not survive the move to
shared storage, so those versions are **not recoverable from disk**.

Suite status on the release branch (2026-10-30): the full suite passes, no failures (CALIBRATION 84). The seven
failures below were the 2026-08-12 state and no longer occur; the caveat on the lost verification environment stands.

Suite status in this environment on 2026-08-12 (identical on local disk
and on nas2):

```
465 passed · 7 failed · 6 skipped · 6 xfailed · 3 xpassed
```

The 7 failures are **pre-existing and unrelated to the fire/methane work** —
confirmed by stashing all changes and reproducing the identical set on a clean
tree:

- `test_canopy.py::TestTurvap::test_temps_reasonable`
- `test_canopy.py::TestTurvap::test_snow_temp_capped`
- `test_f77_verification.py::TestCanopyExchange::test_impexp_phase_change_detection`
- `test_f77_verification.py::TestStomata::test_colimit_quadratic`
- `test_f77_verification.py::TestWeatherGeneration::test_daily_temperature_ordering`
- `test_f77_verification.py::TestBiogeochem::test_carbon_conservation`
- `test_stomata.py::TestStomata::test_jit_compatible`

They look like library drift rather than physics — e.g.
`TypeError: unhashable type: 'jaxlib._jax.ArrayImpl'` from newer Equinox.

**Open action before any numerical claim is made from this environment:** either
bisect the library versions until the 7 pass, or re-run the lockstep and
re-baseline `verify/STATUS.md` against these versions. Until then, treat
absolute numbers from this environment as provisional; relative comparisons
made *within* one environment (such as the F0 bit-exactness test) are still valid.

## Learned wetland area: xgboost (2026-10-29)

`physics.wetland_source: learned` (and `observed_learned`) evaluates gradient-boosted trees each month (`ibis.bgc.wetland_learned`),
so the runtime needs `xgboost` (>= 3.0; and scipy for the neighbourhood filter), the `wetland` optional dependency group of
`python-port/pyproject.toml`. Installed on this box in the GPU venv (`/home/hangkai/.venvs/ibis-gpu`, xgboost 3.4.1 with
scikit-learn 1.9.1) and the CPU environment (xgboost 3.3.0). The pip wheel reports "No visible GPU" and trains on the CPU;
training the release file takes about 4 minutes on 24 threads, so the GPU build was not pursued. The trees load from the
file's JSON in any xgboost >= 3.0. Default runs (`wetland_source: climatology`) do not import xgboost.

