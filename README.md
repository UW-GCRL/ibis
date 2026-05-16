# ibis

Three-in-one repository for the IBIS land-surface model:

| Directory | What it contains |
|---|---|
| [`fortran/`](fortran/) | Original IBIS 2.6b4 (Fortran 77) and a clean F2008 rewrite |
| [`python-port/`](python-port/) | JAX/Python port (differentiable, JIT-compiled) |
| [`verify/`](verify/) | F77 ↔ JAX bit-exact verification framework |

## Alignment status

The JAX port reproduces the F77 reference **at machine precision** across every test.
All tests pass `|F77 − JAX|` below the threshold; the worst observed drift is shown.

| Test | Fields × cells | Worst |Δ| (temp) | Worst |Δ| (water) | Verdict |
|---|---|---|---|---|
| Single-step base (5 climate cells, NPOI=5) | 28 × 5 | 8.5×10⁻⁹ K | 5×10⁻¹³ fraction | ✅ bit-exact |
| 24-step base (1 day, hourly) | 9 × 5 | 2.5×10⁻⁵ K | 1.2×10⁻⁸ fraction | ✅ |
| 24-step perturbed (10 random seeds) | 18 × 5 | each seed PASS | each seed PASS | ✅ |
| 7-day diurnal base | 9 × 5 | 6.6×10⁻⁴ K | 1.1×10⁻⁵ fraction | ✅ |
| 7-day cold snap (Ta −20 K) | 9 × 5 | 7.9×10⁻⁶ K | 4.7×10⁻⁸ fraction | ✅ |
| 7-day heat wave (Ta +15 K) | 9 × 5 | 4.4×10⁻⁵ K | 1.9×10⁻⁷ fraction | ✅ |
| 7-day monsoon (10× rain days 3-4) | 9 × 5 | 1.1×10⁻³ K | 1.9×10⁻⁵ fraction | ✅ |
| 7-day polar night (no sun, Ta −10 K) | 9 × 5 | 9.6×10⁻⁸ K | 8.0×10⁻¹¹ fraction | ✅ |
| 7-day freeze-thaw oscillation | 9 × 5 | 1.5×10⁻⁵ K | 1.3×10⁻⁷ fraction | ✅ |
| **30-day diurnal base** | 9 × 5 | **6.6×10⁻⁴ K** | **1.1×10⁻⁵ fraction** | ✅ |
| **30-day global GSWP3 (~120 land cells)** | 20 × 117 | **~10⁻⁶ K** | **~10⁻⁶ fraction** | ✅ |

### Key bug found and fixed during verification — finding #16

The most subtle bug, found via the [diagnostic pipeline](verify/timestepping/pipeline/):
the variable unpacking in `timestepping/loops.py:272` was **swapped** between
upper and lower canopy stress. Before fix vs after fix, JAX-vs-F77 drift in the
7-day heat-wave scenario:

| Quantity | Pre-fix | Post-fix | Reduction |
|---|---|---|---|
| chaotic test (167 timesteps) | 1.365×10⁻³ wsoi | **2.93×10⁻⁸ wsoi** | **47,000×** |
| 7-day heat wave (worst temp) | 2.6×10⁻² K | **4.4×10⁻⁵ K** | **600×** |
| 30-day base (worst wsoi) | 2.0×10⁻³ fraction | **1.1×10⁻⁵ fraction** | **180×** |

See [`verify/STATUS.md`](verify/STATUS.md) for the full list of 16 verification findings.

## Layout

### `fortran/` — Original IBIS
- `fortran/ibis_2.6b4/` — the historical Fortran-77 sources (original code from Foley et al.)
- `fortran/src/` — a Fortran 2008 rewrite by Min (modular, validated against F77 at loose tolerance)
- `fortran/tests/` — F90 regression test infrastructure

### `python-port/` — JAX port
- `ibis/` — the port itself, organized by physics module (radiation, soil, snow, canopy, stomata, …)
- `tests/f2py/` — Python wrappers around the F77 sources for f2py-style direct calls
- `configs/` — model configuration YAMLs

### `verify/` — Verification framework
- `verify/<section>/` — per-section bit-exact tests (thermo, radiation, soil, snow, canopy, …)
- `verify/timestepping/` — full-timestep integration tests including the global GSWP3 multi-month verification
- `verify/timestepping/pipeline/` — diagnostic pipeline that found the [stresstu unpack-order bug](verify/STATUS.md) (finding #16)
- `verify/STATUS.md` — comprehensive verification status with all 16 findings

## Quick start

### Build the F2008 reference (optional)
```bash
cd fortran && make
```

### Build the F77 f2py wrapper for testing
```bash
cd python-port/tests/f2py && python3 build_f77.py
```

### Set up the JAX environment
```bash
cd verify && python3 -m venv .venv && source .venv/bin/activate
pip install -e ../python-port  # install JAX port
pip install netCDF4 jax equinox jaxtyping numpy
```

### Run the F77↔JAX integration verifier
```bash
PYTHONPATH=python-port verify/.venv/bin/python -m verify.timestepping.verify_lsxmain
```

### Run the global GSWP3 verification (requires GSWP3 data on NAS)
```bash
PYTHONPATH=python-port verify/.venv/bin/python -m verify.timestepping.pipeline.global_gswp3_test
```

## Data availability

The global verification uses two external datasets not in this repo:

- **GSWP3 climate forcing** (`/shared/nas2/Min/GSWP3/`) — 0.5° daily atmospheric forcing 1901-2010 (tas, pr, rsds, rlds, huss, wind, ps)
- **HWSD land mask** (`/shared/nas2/Min/HWSD/surta.nc`) — global land-fraction grid

Both are available on the Chen Lab NAS at UW–Madison.

## Citation

If you use this code, please cite the original IBIS model:

- Foley, J.A. et al. (1996), "An integrated biosphere model of land surface processes, terrestrial carbon balance, and vegetation dynamics", _Global Biogeochemical Cycles_ 10(4): 603-628.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Hangkai You · fji7@wisc.edu · University of Wisconsin–Madison · [UW-GCRL](https://github.com/UW-GCRL)
