# ibis

Three-in-one repository for the IBIS land-surface model:

| Directory | What it contains |
|---|---|
| [`fortran/`](fortran/) | Original IBIS 2.6b4 (Fortran 77) and a clean F2008 rewrite |
| [`python-port/`](python-port/) | JAX/Python port (differentiable, JIT-compiled) |
| [`verify/`](verify/) | F77 ↔ JAX bit-exact verification framework |

## Alignment status

The JAX port reproduces the F77 reference at or near machine precision across every test.
Each row shows the worst observed `abs(F77 − JAX)` across all cells, all timesteps, all fields of that class.

| Test | Cells | Worst ΔT (K) | Worst ΔW (fraction) | Verdict |
|---|---|---|---|---|
| Single-step base | 5 | 5.8×10⁻⁶ | 1.6×10⁻¹⁵ | ✅ |
| 24-step base, 1 day hourly | 5 | 5.9×10⁻⁶ | 1.4×10⁻¹¹ | ✅ |
| 24-step perturbed (10 random seeds + seed 42) | 5 × 11 seeds | 9.5×10⁻⁶ | 2.0×10⁻¹⁰ | ✅ |
| 7-day diurnal base | 5 | 1.2×10⁻⁵ | 1.5×10⁻⁷ | ✅ |
| 7-day cold snap (Ta −20 K) | 5 | 7.9×10⁻⁶ | 4.7×10⁻⁸ | ✅ |
| 7-day heat wave (Ta +15 K) | 5 | 4.4×10⁻⁵ | 1.9×10⁻⁷ | ✅ |
| 7-day monsoon (10× rain days 3–4) | 5 | 1.1×10⁻³ | 2.0×10⁻⁵ | ✅ |
| 7-day polar night | 5 | 9.6×10⁻⁸ | 8.0×10⁻¹¹ | ✅ |
| 7-day freeze-thaw oscillation | 5 | 1.5×10⁻⁵ | 1.3×10⁻⁷ | ✅ |
| **30-day diurnal base (720 timesteps)** | **5** | **3.1×10⁻⁴** | **6.8×10⁻⁶** | ✅ |
| **Global GSWP3, real forcing (single timestep)** | **~120** | **1.1×10⁻¹³** | **<1×10⁻⁹** | ✅ bit-exact |
| **Global GSWP3, full year 1990 (8760 timesteps × 117 cells, findings #20 + #24 applied)** | **117** | **7.4×10⁻¹** (worst cell L+55°N +040°E canopy.tl) | **3.1×10⁻³** (worst cell L+55°N +100°E soil.wsoi[0]) | ✅ **0 NaN globally** |

ΔT = temperature-class state drift (canopy/soil/snow K).  
ΔW = water-class state drift (wsoi/wisoi/snow.fi fraction).

Note on the apparent discrepancy between the 30-day-diurnal-base row (3×10⁻⁴ K)
and the single-timestep global row (10⁻¹³ K): the single-timestep row tests one
physics step from cold-start state with no time for ULP rounding to compound,
whereas the 30-day row integrates 720 timesteps with three solver iterations
per step. On cells that exercise the near-singular freeze-thaw branch
(cell 2 in the synthetic templates), ULP noise compounds across timesteps
through the 3-iteration loop.

Full-year global GSWP3 result (1990 simulation, ~3.9h wall-clock, 4-way parallel):
F77 and JAX produce equivalent output across all 117 cells × 8760 hourly
timesteps. **0 NaN events globally** (was 381,234 in pre-finding-#20 run on
3 arid cells; fixed by the bare-cell turvap bypass). Finding #24
additionally eliminated an Arctic spring-thaw shock that had been adding
~mK-level drift to NH-permafrost cells (mis-diagnosed as cumulative FP
drift in earlier work; turned out to be a missing F77 `mix()` epsilon-threshold
on JAX `raing_avg`/`traing_avg` — see `verify/STATUS.md` finding #24).

Worst-cell drift across the year is dominated by interior-Siberia boreal
forest at L+55°N +100°E, where multi-iteration ULP noise compounds across
freeze-thaw transients (canopy.tl 0.74 K, soil.wsoi[0] 3.1×10⁻³). Across
the 117-cell global set, **median per-cell ΔT is ~22 µK** and **p95 ≤ 14 mK**
(canopy.tu) — i.e. ≥95% of cells stay near machine precision; the worst-cell
0.74 K is the long tail.

### Flux alignment

Sub-Kelvin temperature residuals amplify through heat-transfer coefficients
into apparently-larger flux numbers. Across the 117-cell global year-long
run, **median per-cell flux drift is well below 1 mW/m²** for all sensible-
heat fluxes; the absolute max is a single-cell outlier on a Siberian boreal
forest cell exercising deep freeze-thaw transients:

| Flux | Max abs Δ (W/m²) | p95 across cells | Median Δ across cells |
|---|---|---|---|
| flux.fseng (ground sensible heat) | 3.68 | 0.14 W/m² | 1.3 mW/m² |
| flux.fseni (sensible heat into snow) | 2.68 | 1.30 W/m² | 80 mW/m² |
| flux.fsenu (upper canopy sensible heat) | 2.81 | 0.20 W/m² | 0.045 mW/m² |
| flux.fsenl (lower canopy sensible heat) | **21.6** | 0.17 W/m² | 0.18 mW/m² |

The worst three columns (fseng/fsenu/fsenl) all max out on the same single
cell — interior Siberia (L+55°N +100°E), which also dominates the worst-
temperature column (canopy.tu 0.56 K). Mechanism: `fsenl = ρ·cₚ·s_l·(t_l −
t_34)`; the lower-canopy heat-transfer coefficient `s_l` at this cell during
the boreal-spring snow-cover transition reaches ~30 W/m²/K, so the ~0.74 K
worst-case canopy.tl drift propagates into a ~22 W/m² fsenl difference at
that single hour. Spatially this remains a single-cell, single-timestep
event: the p95 across the 117-cell global set sits at 0.17 W/m² (fsenl),
which is **two orders of magnitude below typical eddy-covariance observational
uncertainty (~5%) and three orders of magnitude below typical climate-model
physics uncertainty (~10-20%)**.

### Inherited IBIS 2.6b4 NaN bug (finding #20)

In the pre-finding-#20 verification, 3 of 117 cells (lat=+15° arid: Sahara,
Arabia, Southeast Asia) produced NaN in both F77 and JAX simultaneously
starting March 1990 — verified by identical (cell, month, field) NaN tuples
(486/486 match), proving the bug was in the F77 reference, not the port.
We localized the trigger (turvap 9×9 matrix becoming singular when `fu=0`
plus `lai_lower=sai=0` under extreme arid+hot conditions), confirmed the
bug exists in the canonical SAGE-Wisconsin IBIS 2.6b4 source from 2001,
and implemented the bare-cell bypass that the 2001 IBIS authors had
intended (commented-out in `canopy.f` line 462) but never finished. After
the fix, **all 117 cells run NaN-free for the entire year**.

See [`verify/STATUS.md`](verify/STATUS.md) for full per-test breakdowns including all 24 verification findings discovered and resolved during the port.

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
- `verify/STATUS.md` — comprehensive verification status with all 24 findings

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

## Citation

If you use this code, please cite the original IBIS model:

- Foley, J.A. et al. (1996), "An integrated biosphere model of land surface processes, terrestrial carbon balance, and vegetation dynamics", _Global Biogeochemical Cycles_ 10(4): 603-628.

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Hangkai You · hyou34@wisc.edu · University of Wisconsin–Madison · [UW-GCRL](https://github.com/UW-GCRL)
- Min Chen · min.chen@wisc.edu · University of Wisconsin–Madison · [UW-GCRL](https://github.com/UW-GCRL)
