# Working environment

> Documentation edition, 2 November 2026: dated release results remain historical. Descriptions of later corrections refer to the reviewed development implementation; this publication does not update model code on main. See [the edition note](technote/EARLY_EDITION.md).


Single source of truth for where the code, the data and the Python environment
live, and for the constraints of the storage they live on. Referenced by
`docs/FIRE.md` and `docs/METHANE.md`.

**Last updated:** 2026-11-01 (the GPU environment of the release run); the other sections as dated in them, else 2026-08-12

---

## Layout

| What | Path | Notes |
|---|---|---|
| **Git working copy** | `/shared/nas2/Hangkai/ibis-port` | **Authoritative.** Branches `main` (the IBIS 3.0 release), `calibration-s1` (its calibration history), `fire-test`, `methane-test`. |
| **Python environment** | `/shared/nas2/Hangkai/ibis-env` | Launcher `ibis-python`, `requirements.lock`, `site-packages/` (755 M). CPU JAX. |
| **GPU Python environment** | `/home/hangkai/.venvs/ibis-gpu` (local disk of the chenlab machine) | The interpreter of the GPU runs, the release run among them; see "GPU environment of the release run". |
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

Full pins in `/shared/nas2/Hangkai/ibis-env/requirements.lock`. CPU JAX only.
The GPU runs, the release run among them, use the separate environment of the
next section. (The June GPU benchmark of `verify/STATUS.md` used yet another
one, `verify/.venv_gpu` with `jax[cuda12]` 0.10.0, which is no longer in either
working copy, `/home/hangkai/ibis-port` or `/shared/nas2/Hangkai/ibis-port`.)

---

## GPU environment of the release run (checked 2026-11-01)

The release run `final_rel30` ran with `/home/hangkai/.venvs/ibis-gpu/bin/python`, the default `PY` of
`verify/calibration/final_validation.sh` (line 25): the `run.log` of each of its three stages reports `JAX backend: gpu`
and prints warnings from this environment's site-packages. It is a venv of `/home/hangkai/miniconda3` (Python 3.12.7)
on the local disk, where `python -m venv` works (unlike on the NAS; storage constraints above). Its packages, from
`pip list`:

```
jax 0.11.0 · jaxlib 0.11.0 · jax-cuda12-plugin 0.11.0 · jax-cuda12-pjrt 0.11.0
equinox 0.13.8 · jaxtyping 0.3.11 · ml_dtypes 0.6.0 · numpy 2.5.2 · scipy 1.18.0
xarray 2026.7.0 · netCDF4 1.7.4 · optax 0.2.8 · PyYAML 6.0.3 · xgboost 3.4.1
CUDA libraries from pip wheels: nvidia-cuda-runtime-cu12 12.9.79 · nvidia-cuda-nvcc-cu12 12.9.86
  nvidia-cuda-nvrtc-cu12 12.9.86 · nvidia-cublas-cu12 12.9.2.10 · nvidia-cudnn-cu12 9.24.0.43
  nvidia-cufft-cu12 11.4.1.4 · nvidia-cusolver-cu12 11.7.5.82 · nvidia-cusparse-cu12 12.5.10.65
  nvidia-nvjitlink-cu12 12.9.86 · nvidia-nccl-cu12 2.31.2 · nvidia-cuda-cupti-cu12 12.9.79
  nvidia-cuda-cccl-cu12 12.9.27 · nvidia-nvshmem-cu12 3.7.2 (and nvidia-nccl-cu13 2.32.3, with xgboost)
NVIDIA driver 575.57.08 (nvidia-smi: CUDA Version 12.9) · GeForce RTX 4090, 24564 MiB
```

What was checked, and how:
- Versions: `pip list` in the environment; `jax.__version__` and `jaxlib.__version__` read 0.11.0, and JAX reports the
  backend `gpu` with one device and double precision off (the release run computed in single precision; technote 17.5).
- Install dates: the dist-info folders of jax, jaxlib, the CUDA plugin and pjrt, equinox, numpy and the nvidia wheels
  are dated 2026-08-23, those of xgboost and nccl-cu13 2026-10-29; none of these is later than the release run of
  2026-10-30. These dates are the only evidence that the packages did not change between the run and this check.
- Driver: `nvidia-smi` and `/proc/driver/nvidia/version` give 575.57.08; the machine was last booted on 2026-09-05
  (`uptime -s`), so this driver was loaded for the release run.
- CUDA libraries: a JAX process of this environment maps cuBLAS, cuDNN, cuFFT, cuSOLVER, NCCL, nvJitLink and the CUDA
  runtime from the pip wheels in its site-packages and only `libcuda.so.575.57.08` from the system (`/proc/self/maps`
  after a matrix product on the GPU); the CUDA toolkits under `/usr/local` (12.4, 12.6, 12.8) are not used.

The JAX, jaxlib, equinox and numpy versions equal the CPU environment's pins above, so the technote's statements about
the versions hold for both environments.

**Deterministic reductions (2026-11-02, technote review D83).** Since this date `python-port/scripts/run_global.py` adds
`--xla_gpu_exclude_nondeterministic_ops=true` to `XLA_FLAGS` before JAX starts, unless `XLA_FLAGS` already names the
option or `IBIS_NONDETERMINISTIC=1` is set. Existing unrelated XLA flags are preserved; setting the environment variable
does not remove an option already supplied explicitly. In the same-configuration GPU trial, running year 2001 twice
with the option produced identical yearly fields, at 2–8 % more time (reported year-step times of 559 and 594 s,
against 550 s without it; technote 17.8). This observation does not establish bitwise agreement across devices, library versions or
other execution paths. The release run `final_rel30` used XLA's default.

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

**Historical follow-up proposed in August 2026:** bisect the library versions or rerun the lockstep and update
`verify/STATUS.md`. The seven failures subsequently cleared, as the October suite record above notes. That later
suite result does not recover the lost environment of the original lockstep experiment. Reproducing a historical
numerical result still requires stating the actual code, settings and environment used; agreement within one
configuration does not establish agreement across configurations.

## Learned wetland area: xgboost (2026-10-29)

`physics.wetland_source: learned` (and `observed_learned`) evaluates gradient-boosted trees each month (`ibis.bgc.wetland_learned`),
so the runtime needs `xgboost` (>= 3.0; and scipy for the neighbourhood filter), the `wetland` optional dependency group of
`python-port/pyproject.toml`. Installed on this box in the GPU venv (`/home/hangkai/.venvs/ibis-gpu`, xgboost 3.4.1 with
scikit-learn 1.9.1) and the CPU environment (xgboost 3.3.0). The pip wheel reports "No visible GPU" and trains on the CPU;
training the release file takes about 4 minutes on 24 threads, so the GPU build was not pursued. The trees load from the
file's JSON in any xgboost >= 3.0. Default runs (`wetland_source: climatology`) do not import xgboost.

