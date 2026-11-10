# Port verification against IBIS 2.6b4: summary and historical record

Moved here unchanged from the README on 10 November 2026. The current account of the verification, with its
revised settings and the unresolved differences, is [Chapter 17 of the technical note](technote/17_numerics_verification.md);
the working record is [verify/STATUS.md](../verify/STATUS.md).

The port was compared with IBIS 2.6b4 (F77) using shared inputs, first routine by routine and then hour by hour.
The May–June 2026 time-step comparisons used fixed vegetation and simplified test soils. They show close agreement
in many tested conditions, with larger differences at some hours. They do not establish bit-for-bit equivalence,
complete model correctness or validation of the processes added in IBIS 3.0.

The current interpretation, including the revised test settings and unresolved differences, is in
[Chapter 17](technote/17_numerics_verification.md), especially sections 17.9–17.12. The historical numerical
tables below are retained for provenance; they are not results from a rerun of the current code.

## Historical time-step comparisons

The following values were reported in the early verification summary. Their aggregation differs among rows:
short tests report maxima over the sampled cells and steps, while the multi-year rows also report distributions.
ΔT denotes temperature differences in K and ΔW water or snow fractions. A finite run is not necessarily an
accurate or physically valid run, and a nonzero difference is not bit-exact agreement.

| Test | Cells | Reported ΔT (K) | Reported ΔW (fraction) | Scope |
|---|---|---|---|---|
| Single-step base | 5 | 5.8×10⁻⁶ | 1.6×10⁻¹⁵ | historical comparison; see qualifications below |
| 24-step base, 1 day hourly | 5 | 5.9×10⁻⁶ | 1.4×10⁻¹¹ | historical comparison; see qualifications below |
| 24-step perturbed (10 random seeds + seed 42) | 5 × 11 seeds | 9.5×10⁻⁶ | 2.0×10⁻¹⁰ | historical comparison; see qualifications below |
| 7-day diurnal base | 5 | 1.2×10⁻⁵ | 1.5×10⁻⁷ | historical comparison; see qualifications below |
| 7-day cold snap (Ta −20 K) | 5 | 7.9×10⁻⁶ | 4.7×10⁻⁸ | historical comparison; see qualifications below |
| 7-day heat wave (Ta +15 K) | 5 | 4.4×10⁻⁵ | 1.9×10⁻⁷ | historical comparison; see qualifications below |
| 7-day monsoon (10× rain days 3–4) | 5 | 1.1×10⁻³ | 2.0×10⁻⁵ | historical comparison; see qualifications below |
| 7-day polar night | 5 | 9.6×10⁻⁸ | 8.0×10⁻¹¹ | historical comparison; see qualifications below |
| 7-day freeze-thaw oscillation | 5 | 1.5×10⁻⁵ | 1.3×10⁻⁷ | historical comparison; see qualifications below |
| **30-day diurnal base (720 timesteps)** | **5** | **3.1×10⁻⁴** | **6.8×10⁻⁶** | historical comparison; see qualifications below |
| **Global GSWP3, real forcing (single timestep)** | **~120** | **1.1×10⁻¹³** | **<1×10⁻⁹** | historical comparison; see qualifications below |
| **Global GSWP3, full year 1990 (8760 timesteps × 117 cells, findings #20 + #24 applied)** | **117** | **7.4×10⁻¹** (worst cell L+55°N +040°E canopy.tl) | **3.1×10⁻³** (worst cell L+55°N +100°E soil.wsoi[0]) | historical comparison; see qualifications below |
| **Global GSWP3, 35-year multi-decadal 1951-1985 (306,600 timesteps × 117 cells, WMO climate normal length)** | **116 finite + 1 NaN cell** | **4.5×10⁻¹** (worst single cell-year canopy.tu, 1951 cold-start) — median **1.25×10⁻³**, p95 **3.3×10⁻²** | **4.5×10⁻³** soil.wsoi[0] | historical comparison; see qualifications below |
| **Global GSWP3, 46-year multi-decadal 1951-1996 post-finding-#26 (~48M cell-step comparisons in total)** | **117 finite, no NaN** | **2.1×10⁻¹** (1985 freeze/thaw single-spike canopy.tu) — **median 3 µK**, **p99 20 mK** | **6.5×10⁻³** soil.wsoi[0] | historical comparison; see qualifications below |

The 46-year row's 0.21 K value is retained as an earlier reported event in 1985, not as the largest difference over
the complete record. The later audit of the same archived outputs gives a largest upper-canopy difference of
0.45 K in 1951 and a largest lower-canopy difference of 0.87 K. The upper-canopy median and 99th percentile over
5,382 cell-years are 2.6 µK and 17 mK, rounded in the early summary to 3 µK and 20 mK. These are distributions of
each cell-year's largest hourly difference; they are not errors of annual means. See Chapter 17, Table 17.8.

The old test soils had a negative saturated suction that kept capillary diffusion at its floor in both codes.
The settings also differed from IBIS 2.6b4's own soil, snow and vegetation parameters. Since November 2026 the
verification harness uses those original settings. Revised one-step checks then agree to about 10⁻¹² K in the
compared temperatures; the lower-canopy sensible-heat diagnostic now differs by design. The 46-year run has not
been repeated with the revised harness or current code.

## What the differences do and do not show

Small numerical differences can grow when freezing or melting sends the two calculations through different
branches. A diagnostic at 55°N, 100°E on 25 April of the separate 1990 comparison tested one such event: starting
the port from the Fortran state just before that hour reduced the next-step differences to at most 2 × 10⁻⁹ in
the recorded quantities. This supports accumulated rounding as the cause of that event. It does not isolate a
particular math library or fused operation, or show that every discrepancy has the same cause.

Two later cases remain unexplained: a spring canopy event at 55°N, 40°E and the fourth-day event in the synthetic
monsoon test. Giving the port the Fortran state immediately before those events does not remove their differences.
Chapter 17.9.3 and its developer notes distinguish these cases from the rounding-sensitive cases.

## The 46-year comparison, 1951–1996

The two codes ran continuously at 117 cells, about 48 million cell-step comparisons, with vegetation fixed and
the old synthetic surface settings. Neither produced a non-finite value. The temperature differences did not show
steadily increasing drift across those years, but this observation is limited to that configuration.

The original statement that there were no out-of-range events was incorrect. The archived files contain
1,117,261 step-level exceedances of the harness's plausibility windows; the flagged cell-months occurred in both
codes. These include Antarctic temperatures below 200 K and large sensible-heat diagnostics, some at snow-free
surfaces where the snow flux does not enter the energy balance. They are described in Chapter 17.9.4.

The run stopped during 1997 after an input-file read failure, recorded for `rsds_gswp3_1991_2000.nc4`.
The completed 46 years are evidence about numerical agreement under the test conditions, not proof of general
equivalence or of the correctness of a future global simulation.

## Historical GPU benchmark (RTX 4090, double precision)

The early benchmark compared ways of running the same ported time step on an RTX 4090:

| Configuration | ms / timestep (FP64) | Speedup vs no-JIT |
|---|---|---|
| no-JIT, single step (eager) | 602 | 1× (baseline) |
| JIT'd single step | 4.66 | 130× |
| JIT + `lax.scan` over 24 h | 1.42 | 423× |
| JIT + `lax.scan` over 720 h (1 month) | 0.87 | **691×** |

The record reports 1.3 µs per cell-step at 1,000 cells. It projected about 9 minutes for a 50-year, 117-cell
GPU-only calculation, against about 58 hours for the CPU lockstep workflow, a roughly 400-fold wall-clock ratio.
That was a projection between different workflows, not a measured speedup of the full IBIS 3.0 protocol.
The GPU workflow omits the in-loop Fortran comparison, which runs on the CPU.

These benchmarks used the historical `verify/.venv_gpu/` environment with `jax[cuda12]==0.10.0` and
[bench_gpu_timestep.py](../verify/timestepping/pipeline/bench_gpu_timestep.py). Use
[ENVIRONMENT.md](ENVIRONMENT.md) for the current pinned environments, and Chapter 17.8 for the tested
scope of deterministic GPU reductions.

## Early global GPU spin-ups (June 2026)

The historical physics runner was also exercised at 0.5° resolution with 88,484 land cells. This was a larger
numerical-stability test, distinct from the later release protocol and its 95,333-cell land mask. The runner is
[global_gswp3_gpu.py](../verify/timestepping/pipeline/global_gswp3_gpu.py), with forcing-cycle options and the
initialization helpers in [real_state_builder.py](../verify/timestepping/pipeline/real_state_builder.py).

| Run | Resolution | Cells | Wall-clock | Per-year (steady-state) |
|---|---|---|---|---|
| **0.5° × 200-yr (synthetic ICs)** | 0.5° | 88,484 | ~11 hr on RTX 4090 | ~195 sec/yr |
| **0.5° × 200-yr (real ICs)** | 0.5° | 88,484 | ~11 hr | ~195 sec/yr |

Both recorded runs cycled the 1901–2000 forcing twice, starting from cold-start initial conditions. The early
record reports that the resulting approximately 155 billion cell-steps per run stayed finite after safeguards
were applied. Finite output alone does not verify energy and water conservation, realism or equivalence with the
Fortran at every global cell. These runs were not repeated for the documentation review.

## Ground-evaporation safeguard (finding 27)

The higher-resolution tests exposed four Yemen-highland cells near 13.5–14°N, 44–44.5°E that the 117-cell grid had
not sampled. The historical trace records an unstable canopy calculation producing ground evaporation near
0.26 kg m⁻² s⁻¹, about 940 mm h⁻¹, followed by nonphysical stores and non-finite values.

The port bounds cell-mean ground evaporation to ±10⁻³ kg m⁻² s⁻¹, equivalent to ±3.6 mm h⁻¹. This is a numerical
safeguard, not a universal physical upper limit. The current `bounded_ground_evaporation` function in
[loops.py](../python-port/ibis/timestepping/loops.py) supplies the same bounded rate to the water calculation,
ground latent heat and ET output; the release bounded only the water. Chapter 17.4 describes the bounds and
their diagnostic counts. They can make the two codes differ when active.

This should not be grouped with finding 26 as an established inherited-soil defect. Finding 26's overflow came
from the negative suction of the old synthetic harness; its classification as an inherited bug was withdrawn.
The guard remains in both codes. The bare-canopy singularity of finding 20 is a separate case, discussed below.

## Historical input-based initialization example

The early runner's `--real-ic` option replaces its synthetic latitude-band templates with soil texture from
HWSD and vegetation classes from ESA CCI. The associated initial states are computed by
[soil_init.py](../python-port/ibis/io/soil_init.py) and [veg_init.py](../python-port/ibis/io/veg_init.py).
The following Yakutsk example (62°N, 130°E) compares the two initializations; it is not an independent validation
against measured canopy structure or soil properties.

| Field | Synthetic template | Real (HWSD + ESACCI) | Interpretation of the input-based initialization |
|---|---|---|---|
| `fu` (upper canopy cover) | 0.10 | **0.97** | high initialized canopy cover |
| `lai` (upper) | 0.5 | **4.1** | initialized leaf area |
| Soil sand | template avg | **52%** | input texture |
| Soil clay | template avg | **13%** | — |
| Wilting point | 0.15 (template) | **0.21** (texture-derived) | — |

The synthetic option was retained for comparisons using the historical harness. The current production driver
and release initialization are described in [Chapter 2](technote/02_surface_characterization.md).

## Historical flux differences in the 46-year run

For each cell, the early flux summary took the largest hourly absolute difference across all 46 years, then
reported the distribution over cells. This aggregation differs from the distribution over cell-years in
Chapter 17, Table 17.8, so their percentiles should not be compared directly.

| Flux | Max abs Δ (W/m²) | p95 across cells | Median Δ across cells | Worst cell · year |
|---|---|---|---|---|
| flux.fseng (ground sensible heat) | 7.49 | 0.34 W/m² | 30 mW/m² | Yukon L+65°N -120°E · 1983 |
| flux.fseni (sensible heat into snow) | 6.97 | 3.59 W/m² | 340 mW/m² | L+35°N -80°E · 1969 |
| flux.fsenu (upper canopy sensible heat) | 12.12 | 0.45 W/m² | 15 mW/m² | Russia L+55°N +40°E · 1951 |
| flux.fsenl (lower canopy sensible heat) | 12.50 | 0.63 W/m² | 15 mW/m² | Siberia L+55°N +160°E · 1987 |

Temperature differences can produce larger flux differences through the transfer coefficients, particularly
near phase changes. The table alone does not establish the cause of each extreme or a general observational
noise threshold. The earlier 1990-only comparison reported a lower-canopy sensible-heat maximum of 21.6 W m⁻²;
it belongs to a separate trajectory and is not the maximum of this 46-year trajectory. Also, the current code's
lower-canopy heat diagnostic uses the lower canopy's own air temperature, a deliberate correction that must be
accounted for in any new Fortran comparison.

## Bare-canopy singularity (finding 20) and remaining work

In an earlier one-year test, three bare arid cells developed non-finite values in both codes. The investigation
traced this to a singular canopy system with vanishing vegetation area. A bare-canopy branch was added to both
the port and the reference Fortran; the repeated 117-cell year then stayed finite. This outcome establishes the
effect of that correction in the tested cases, not that all numerical issues are closed.

The detailed record is [verify/STATUS.md](../verify/STATUS.md). For its later corrections, the harness scope and the
unresolved canopy cases, use [Chapter 17](technote/17_numerics_verification.md). New scientific processes,
driver behavior, calibration and numerical repeatability require their own checks in addition to the historical
Fortran comparisons.

