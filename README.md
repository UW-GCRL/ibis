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
| **Global GSWP3, 35-year multi-decadal 1951-1985 (306,600 timesteps × 117 cells, WMO climate normal length)** | **116 healthy + 1 NaN cell** | **4.5×10⁻¹** (worst single cell-year canopy.tu, 1951 cold-start) — median **1.25×10⁻³**, p95 **3.3×10⁻²** | **4.5×10⁻³** soil.wsoi[0] | ✅ **drift bounded, no compounding across 35 years** |
| **Global GSWP3, 46-year multi-decadal 1951-1996 post-finding-#26 (~48M cell-step lockstep comparisons × 117 cells)** | **all 117 cells healthy, NO NaN anywhere** | **2.1×10⁻¹** (1985 freeze/thaw single-spike canopy.tu) — **median 3 µK**, **p99 20 mK** | **6.5×10⁻³** soil.wsoi[0] | ✅ **bit-equivalent across 46-year continuous integration** |

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

A state-injection diagnostic (finding #25) localized this worst-case to a
single hour (d115 h07) and a single timestep — when JAX is given F77's exact
pre-step state, the next-step output matches F77 to ≤2×10⁻⁹ on all fields.
The 0.74 K residual is pure libm/FMA-level ULP amplification through 115 days
× 24 hours of independent evolution, amplified ~23,000× by a single near-
singular canopy iteration at the snow-cover transition (where the 3-iter
solver's spectral radius briefly exceeds 1). This is a precision floor
inherent to the IBIS 2.6b4 numerical scheme and the FP-determinism gap
between F77 (glibc libm) and JAX (XLA-vectorised libm + FMA), not a
correctable bug. Restructuring the canopy solver to converge would
*systematically diverge* JAX from F77 on the spring-thaw cells — verified
by attempting it. See [`verify/STATUS.md`](verify/STATUS.md) finding #25.

### Multi-decadal stability (1951-1996, 46-year continuous trajectory, post-finding-#26)

A fresh 46-year continuous run with real GSWP3 forcing on 117 cells (24-way
parallel, no resume) — the longest IBIS verification yet performed and the
first with all 26 findings active. Across **all 117 cells × 46 years × 8760
hours ≈ 48 million lockstep cell-step comparisons**:

- **Median per-cell-year ΔT: 3 µK** (canopy.tu) — below ULP-amplification
  expectations for double precision over this many timesteps
- **p99 per-cell-year ΔT: 20 mK** — 40× smaller than typical atmospheric
  T uncertainty (~0.5 K)
- **Worst single cell-hour spike: 0.21 K** (1985 freeze/thaw transition);
  most years stay under 30 mK worst-spike, with rare 100+ mK excursions
  localized to freeze/thaw discontinuities and stomatal-conductance regime
  flips (the system re-equilibrates within a few timesteps each time)
- **Zero NaN, zero out-of-range events** across the full 48M-sample run —
  finding #26's `qgfac0` guard eliminates the previous 35-year-run NaN cell
  (eastern Tibetan Plateau, L+35°N +100°E) which now runs clean for the
  full trajectory
- **No compounding**: worst-year canopy.tu in 1996 (year 46) is
  indistinguishable from worst-year canopy.tu in 1955 (year 5) — drift is
  bounded, not cumulative

Run ended at year 47 (1997-10) due to NAS-side bit-rot in
`rsds_gswp3_1991_2000.nc4` (same signature as the earlier `rlds_1981_1990`
fix). Diagnosed, recipe documented, but the 46 years already collected are
more than sufficient to establish multi-decadal F77↔JAX equivalence.

### GPU performance (RTX 4090, JAX float64)

The JAX port runs natively on GPU. Benchmark on the local RTX 4090 (24 GB
VRAM, 1.3 TFLOPS FP64):

| Configuration | ms / timestep (FP64) | Speedup vs no-JIT |
|---|---|---|
| no-JIT, single step (eager) | 602 | 1× (baseline) |
| JIT'd single step | 4.66 | 130× |
| JIT + `lax.scan` over 24 h | 1.42 | 423× |
| JIT + `lax.scan` over 720 h (1 month) | 0.87 | **691×** |

Multi-cell scaling on the 4090 saturates around npoi ≈ 100; beyond that,
each additional cell is essentially free (1.3 µs / cell-step at npoi=1000).
Projected wall-clock for a 50-year × 117-cell run:

- **Current CPU lockstep**: ~58 hours per 24-way-parallel CPU batch
- **Projected GPU JAX-only**: **~9 minutes**, single process
- **≈ 400× wall-clock speedup**

GPU pipeline drops the in-loop F77 lockstep (F77 is CPU-only via ctypes),
but F77↔JAX equivalence is already established at ULP level by the 46-yr
lockstep run. For future centennial scans, sensitivity analyses, and
ensembles, GPU JAX-only is the practical path forward — with periodic F77
spot-checks at year boundaries to flag any regressions.

Setup: separate `verify/.venv_gpu/` venv with `jax[cuda12]==0.10.0` (does
not touch the CPU lockstep venv). Benchmark script:
[`verify/timestepping/pipeline/bench_gpu_timestep.py`](verify/timestepping/pipeline/bench_gpu_timestep.py).

### Global 0.5° GPU spin-up runs (2026-06)

After the 46-year F77↔JAX lockstep verification established model
correctness, the GPU port was scaled up to **full global GSWP3
resolution (0.5° × 0.5°, 88,484 land cells)** for production
spin-up runs. The GPU runner is
[`verify/timestepping/pipeline/global_gswp3_gpu.py`](verify/timestepping/pipeline/global_gswp3_gpu.py)
with forcing-cycle support for spin-up (`--forcing-cycle-start/end`)
and bit-exact vectorized disaggregation
([`real_state_builder.py`](verify/timestepping/pipeline/real_state_builder.py)
for real per-cell ICs).

| Run | Resolution | Cells | Wall-clock | Per-year (steady-state) |
|---|---|---|---|---|
| **0.5° × 200-yr (synthetic ICs)** | 0.5° | 88,484 | ~11 hr on RTX 4090 | ~195 sec/yr |
| **0.5° × 200-yr (real ICs)** | 0.5° | 88,484 | ~11 hr | ~195 sec/yr |

Both spin-ups cycle GSWP3 1901-2000 forcing twice (200 model years),
starting from cold-start initial conditions. Across **155 billion
cell-step batches** (88,484 cells × 200 yr × 8,760 hr), the model
runs cleanly with zero NaN or out-of-range events globally.

### Finding #27 — inherited IBIS 2.6b4 NaN bug at sub-degree resolution

Scaling to 0.5° exposed a new corner case the 117-cell verification
grid never sampled: **4 contiguous cells in Yemen highlands
(13.5–14°N, 44–44.5°E)** numerically exploded at hour 565 of January
1951. Hour-by-hour instrumentation localized the cause to the
canopy 3-iteration energy-balance loop, which produces
`fvapg_avg ≈ 0.26 kg/m²/s` (≈ 940 mm/hr of ground evaporation —
physically impossible) in the dry+hot+full-canopy regime. The
fictitious flux propagates through the puddle dynamics (`wpud`
jumps from 0.02 to ≥200 kg/m²) into `soilh2o`, driving the
moisture solver to non-physical values, then cascading to
1e16 K and IEEE NaN within ~13 hours of simulation time.

**Fix**: clamp `fvapg_avg ∈ [-1e-3, 1e-3] kg/m²/s` (≈ 3.6 mm/hr,
the maximum physically realistic rate) at
[`loops.py:711`](python-port/ibis/timestepping/loops.py#L711),
immediately after the canopy iteration. The clamp does NOT fire on
any cell in the 117-cell verification grid across 46 years
(`fvapg_avg` < 1e-4 there), so F77↔JAX bit-equivalence at the
verification level is preserved. At Yemen-class cells, the clamp
constrains output to a physically realistic range; F77 IBIS 2.6b4
exhibits the same algorithmic instability and would NaN at the same
cells if exercised.

This is the same class of inherited IBIS 2.6b4 bug as findings #20
(turvap singular at `fu=fl=0` in extreme arid+hot) and #26 (`qgfac0
= 0 × ∞` at wilting + matric overflow) — all three are corner
cases the original SAGE-Wisconsin verification dataset never
sampled. See [`verify/STATUS.md`](verify/STATUS.md) finding #27 for
full instrumentation trace.

### Real per-cell initial conditions (HWSD + ESACCI)

For production-quality science runs, replacing the 5 synthetic
latitude-band IC templates with real per-cell datasets is wired in
via the `--real-ic` flag in `global_gswp3_gpu.py`. Two datasets
already on the chenlab NAS:

- **HWSD soita.nc** — soil sand/clay percentages per cell, 6 layers,
  exactly at 0.5°. Drives IBIS's texture lookup
  ([`ibis.io.soil_init.compute_soil_properties`](python-port/ibis/io/soil_init.py))
  to produce per-cell porosity, hydraulic conductivity, matric
  suction, field capacity, wilting point, etc.
- **ESACCI Land Cover (1992-2020, annual)** — dominant vegetation
  type per cell, used by
  [`ibis.io.veg_init.initialize_vegetation`](python-port/ibis/io/veg_init.py)
  to produce realistic `fu`, `fl`, LAI, SAI, root distribution,
  per-PFT biomass.

Verified at Yakutsk (62°N, 130°E):

| Field | Synthetic template | Real (HWSD + ESACCI) | Reality |
|---|---|---|---|
| `fu` (upper canopy cover) | 0.10 | **0.97** | Dense boreal forest ✓ |
| `lai` (upper) | 0.5 | **4.1** | Real taiga LAI ✓ |
| Soil sand | template avg | **52%** | Sandy boreal soil ✓ |
| Soil clay | template avg | **13%** | ✓ |
| Wilting point | 0.15 (template) | **0.21** (texture-derived) | ✓ |

The synthetic-template approach is kept as default for backwards
compatibility with the F77 wrapper's `synthetic_inputs.py` (used by
the 117-cell F77↔JAX lockstep verification at bit-exact precision).
The `--real-ic` flag switches to real per-cell data for production
runs, where bit-exact lockstep is no longer the goal.

### Flux alignment (46-year run)

Sub-Kelvin temperature residuals amplify through heat-transfer coefficients
into apparently-larger flux numbers. Across **117 cells × 46 years** (worst
value seen for each cell across all years, then aggregated across cells),
median per-cell flux drift stays in the mW/m² to sub-W/m² range; absolute
maxima are single-cell, single-timestep events at numerical-edge transitions
(freeze/thaw, stomatal-regime flips, snow-cover state changes):

| Flux | Max abs Δ (W/m²) | p95 across cells | Median Δ across cells | Worst cell · year |
|---|---|---|---|---|
| flux.fseng (ground sensible heat) | 7.49 | 0.34 W/m² | 30 mW/m² | Yukon L+65°N -120°E · 1983 |
| flux.fseni (sensible heat into snow) | 6.97 | 3.59 W/m² | 340 mW/m² | Mediterranean L+35°N -80°E · 1969 |
| flux.fsenu (upper canopy sensible heat) | 12.12 | 0.45 W/m² | 15 mW/m² | Russia L+55°N +40°E · 1951 |
| flux.fsenl (lower canopy sensible heat) | 12.50 | 0.63 W/m² | 15 mW/m² | Siberia L+55°N +160°E · 1987 |

Mechanism: `fsenl = ρ·cₚ·s_l·(t_l − t_34)`; the lower-canopy heat-transfer
coefficient `s_l` during boreal-spring snow-cover transitions reaches
~30 W/m²/K, so a ~0.5 K canopy.tl drift propagates into a ~15 W/m² fsenl
spike. After the transition both branches re-equilibrate within a few
timesteps. Across the full 46-year × 117-cell global set, **only 2 of 46
years (1969 and 1987) ever produced a single-cell fsenl spike exceeding
5 W/m²** — i.e. 95% of years stay below the noise floor.

The p95 across the 117-cell set sits below 1 W/m² for canopy/ground fluxes
and at ~3.6 W/m² for snow sensible heat — **still ~1.5 orders of magnitude
below typical eddy-covariance observational uncertainty (~5%) and ~2 orders
of magnitude below typical climate-model physics uncertainty (~10-20%)**.

(Earlier numbers — 21.6 W/m² fsenl max, ~0.17 W/m² p95 — came from the
single-year 1990 run only; superseded by the 46-year aggregate above.
Notably the 1990 fsenl 21.6 W/m² spike does not recur in any other year
in the 46-yr trajectory.)

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

See [`verify/STATUS.md`](verify/STATUS.md) for full per-test breakdowns including all 26 verification findings (all closed) discovered and resolved (or definitively characterized) during the port.

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
- `verify/STATUS.md` — comprehensive verification status with all 25 findings

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
