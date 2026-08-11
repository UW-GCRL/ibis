# IBIS-JAX fire module — design & build log

**Status:** design agreed, not yet implemented
**Owner:** Hangkai
**Created:** 2026-08-10
**Scope:** replace IBIS 2.6b4's annual disturbance-fraction fire scheme with a
Glob-FIRM-class scheme that produces *burned area*, adds a human term, and emits
trace gases — without leaving the "light model" band.

---

## 0. Design constraint

IBIS's value is that it is small, fast and legible. This module is bound by:

- **No new prognostic PDE and no new implicit solver.**
- **At most one new input dataset** (population density).
- **≤ 20 calibratable parameters**, all exposed for gradient-based fitting.
- **Runtime cost < 2%** of the timestep budget.
- The legacy F77-equivalent path stays reachable and **bit-exact** so the
  46-year F77↔JAX lockstep in `verify/STATUS.md` remains valid.

Justification for staying light: FireMIP's nine-model evaluation found no model
best across fire variables, and that *process-based models performed no better
than empirical ones* (Hantson et al. 2020). Burned area across those models spanned
39–536 Mha against an observed 345–468 Mha. Complexity has not bought skill, so we
buy the cheap structural fixes instead and spend the saved effort on calibration.

---

## 1. Build status

| Stage | Description | Status |
|---|---|---|
| F0 | Legacy path frozen behind flag, regression test | ☐ not started |
| F1 | Fire season length replaces annual-mean `firefac` | ☐ not started |
| F2 | Fine-fuel term (`totalit` + live leaf) | ☐ not started |
| F3 | Human ignition/suppression term | ☐ not started |
| F4 | PFT-specific fire mortality | ☐ not started |
| F5 | Burned area output + trace-gas emissions (Andreae 2019) | ☐ not started |
| F6 | Differentiable calibration against GFED5 | ☐ not started |

Legend: ☐ not started · ◐ in progress · ☑ done & verified

---

## 2. What IBIS has today

### 2.1 The scheme

F77 `fortran/ibis_2.6b4/vegetation.f:591`, called annually from `dynaveg` when
`isimfire == 1`:

```fortran
burn        = firefac(i) * min (1.0, totlit(i) / 0.200)
disturbf(i) = 1.0 - exp(-0.5 * burn)
disturbf(i) = max (0.0, min (1.0, disturbf(i)))
```

`firefac` is accumulated in `fortran/ibis_2.6b4/stats.f:1180`:

```fortran
water     = wisoi(i,1) + (1. - wisoi(i,1)) * wsoi(i,1)
waterfrac = (water - swilt(i,1)) / (1. - swilt(i,1))
fueldry   = max (0.0, min (1.0, -2.0 * (waterfrac - 0.5)))
firefac(i) = ((nytimes-1) * firefac(i) + fueldry) * rwork      ! running annual MEAN
```

JAX port: `python-port/ibis/physics/vegetation_dynamics.py:22` (`fire()`), called
from `dynaveg()` at ~line 399. When fire is off, `disturbf` falls back to a
constant 0.005 yr⁻¹ background disturbance.

### 2.2 Defects, ranked by impact

| # | Defect | Why it matters | Fixed by |
|---|---|---|---|
| 1 | `firefac` is an **annual mean** of daily dryness | A cell with 6 bone-dry + 6 wet months scores identically to one permanently half-dry. Fire is threshold-driven; averaging is the wrong operator. This is the single largest error. | F1 |
| 2 | Fuel is `totlit` — **total** litter | `totlit = totalit + totrlit`. Root litter cannot burn. `totalit` (above-ground litter) already exists in `CarbonState` and is the correct variable. Live fine fuel (grass, leaves) is ignored entirely. | F2 |
| 3 | No human term | Population density drives both ignition (low density) and suppression (high density). Its absence is the leading explanation for the Africa bias and the missed post-2000 declining trend across CMIP6 models. | F3 |
| 4 | Uniform mortality across PFTs | `disturbf` hits savanna trees and boreal conifers identically, so the fire→vegetation feedback — the whole reason fire is in a DGVM — is wrong. | F4 |
| 5 | Output is not burned area | `disturbf` has no observational counterpart, so no version of this module can be validated or falsified. | F5 |
| 6 | No trace-gas emissions | No CO₂/CH₄/N₂O from fire; nothing to hand to the methane budget. | F5 |
| 7 | `bgc/fire_emissions.py` is dead scaffolding | Never called (only a dormant `enable_fire_emissions=False`); Andreae & Merlet (2001) 4-class factors; state field documented gC m⁻² yr⁻¹, code returns kgC m⁻² yr⁻¹, `nee.py` consumes it as gC m⁻² d⁻¹. | F5 (rewrite, do not extend) |

### 2.3 To check before implementing

- [ ] `vegetation_dynamics.py:408` puts `cbior` (fine roots) into `cdisturb`.
      Confirm whether `cdisturb` is treated as **combusted** (emitted) or as
      **mortality transferred to litter**. If combusted, roots must be excluded
      from the emission calculation even if they stay in the mortality flux.
- [ ] Whether `firefac` is used anywhere besides `fire()`.
- [ ] Whether the annual `dynaveg` call site can accept a daily-accumulated input
      without breaking the verified `dynaveg` residuals.

---

## 3. What we take from other models

| Model | Structure | Take | Leave |
|---|---|---|---|
| **Glob-FIRM** (Thonicke et al. 2001), LPJ | Burned fraction from simulated **fire season length** + minimum annual fuel load | The fire-season-length concept — this is exactly what IBIS lacks and it is nearly free | Its lack of any ignition/human term |
| **Li et al. 2012/2013** → CLM4.5/5, ELM | Burned area = fire counts × mean fire size; lightning + population ignition; GDP suppression; 4 fire types | The **human ignition/suppression shape** only | Fire counts, elliptical spread, 4 fire types, lightning climatology, GDP — ~30 params, too heavy |
| **SPITFIRE** (Thonicke et al. 2010) | Rothermel rate of spread, Nesterov index, dead-fuel size classes, fire intensity | Nothing structural. Note Nesterov as an *option* for fuel moisture if soil-moisture dryness proves inadequate | Everything else: needs sub-daily wind, fuel bulk density and size classes IBIS does not carry |
| **SIMFIRE / INFERNO** | Regression of burned area on a few drivers | Confidence that a low-complexity form is competitive | — |
| **ELM2.1-XGBfire** (2025) | XGBoost burned area coupled into ELM | Cautionary: 0.98 spatial correlation offline collapsed to **0.59** once coupled, as fire–vegetation feedback accumulated ML error | Do not put an unconstrained ML predictor in the feedback loop |
| **Andreae (2019)** | 370+ studies, per-biome emission factors incl. peat | Emission factors (F5) | — |

---

## 3b. Reference implementation survey (code-level, 2026-08-10)

Source was fetched and measured, not just read about. This is the honest cost of
each option:

| Implementation | Files / lines | Tunable params | Required forcing | Notes |
|---|---|---|---|---|
| **IBIS 2.6b4** `vegetation.f:591` | **11 lines** | 2 | none | Where we are |
| **CTSM `CNFireLi2014Mod.F90`** + `CNFireBaseMod.F90` | 1500 + 1365 = **2865 lines** | ~25 (in `cnfire_const`) | lightning, population density, GDP | `need_lightning_and_popdens()` is an explicit method — it will not run without them |
| **CTSM `CNFireLi2021Mod.F90`** | **670 lines** | subset | same | Newer, leaner variant — worth reading before implementing |
| **FATES SPITFIRE** `SFMainMod` + `SFParamsMod` + `SFNesterovMod` | 645 + 278 + 109 = **1032 lines** | **33** | daily T, RH, wind, precip | The most compact real SPITFIRE. 9 subroutines. |
| **LPJmL SPITFIRE** `src/spitfire/` | **30 files, ~120 KB C** | many | + wind, ignition maps | Full-fat |
| **This proposal** | ~250 lines | 21 | population density only | |

### What is actually cheap inside the expensive models

Reading the source rather than the papers changed two things:

**1. SPITFIRE's cost is in rate-of-spread, not in fire danger.** The Nesterov fire
danger index is 81 lines in LPJmL (`firedangerindex.c`) and reduces to one line
with one PFT parameter:

```c
d_fdi = max(0, 1 - (1/char_moist_factor) * exp(-alpha_fuelp * nesterov_accum))
```

The expensive part is `rateofspread.c` + `area_burnt.c`, which need the Rothermel
equations, fuel bulk density and an elliptical fire geometry:

```c
base       = 1 - exp(-0.03 * windsp)
lb_tree    = fpc_tree * (1 + 8.729 * base^2.155)     /* length:breadth ratio */
dbf        = (ros_backward + ros_forward) * fire_duration
area_burnt = num_fires * (pi/4)/lb * dbf^2 * 1e-4
```

**So we can borrow a real fire-danger index without buying Rothermel.** This is a
better option than the soil-moisture proxy if F1 does not fix seasonality.

**2. Li's occurrence structure is piecewise-linear ramps, not physics.** The
expensive part of CLM's fire is the *four fire types*, not the core. `cnfire_const`
shows the core is a set of thresholds:

- `lfuel`, `ufuel` — lower/upper fuel-mass thresholds (gC m⁻²), a ramp
- `rh_low`, `rh_hgh` — relative humidity ramp
- `bt_min`, `bt_max` — root-zone wetness (`btran`) ramp
- `pot_hmn_ign_counts_alpha` — human ignition, per person per month
- population suppression ~ `(hdm/450)^0.5`; GDP suppression ~ `f(gdp/7)`

Then `cli_scale` (deforestation), `cropfire_a1` (cropland), `boreal_peatfire_c` /
`non_boreal_peatfire_c` (peat) add the three extra fire types — **that is where the
2865 lines go.** A single-fire-type Li core is cheap.

**Revised borrowing decision:** take Li's *ramp structure* for fuel availability and
combustibility and its population terms, with **one fire type only**; keep IBIS's
existing exponential burned-fraction closure instead of fire-counts × fire-size.

---

## 4. Proposed scheme

### 4.1 Daily accumulation (new)

Runs in the existing daily statistics step. Reuses variables already computed:

```
water_1   = wisoi[:,0] + (1 - wisoi[:,0]) * wsoi[:,0]
waterfrac = (water_1 - swilt[:,0]) / (1 - swilt[:,0])
fueldry_d = clip(-2 * (waterfrac - w_mid), 0, 1)          # unchanged form

# NEW — replace the running mean with a threshold counter
is_fireday = fueldry_d > d_crit
nfireday  += is_fireday                                    # integer accumulator
sdry      += fueldry_d * is_fireday                        # intensity-weighted
```

Cost: two extra `[npoi]` accumulators, one comparison per day. Negligible.

### 4.2 Annual burned area

Revised after the code survey (§3b) to use Li-style ramps, which are cheaper and
better-tested than the ad-hoc products in the first draft:

```
fseason   = nfireday / ndaypy                              # fire season fraction
dry_mean  = sdry / max(nfireday, 1)                        # mean severity in season

# --- fuel availability: Li-style ramp instead of min(1, fuel/ref)
fine_fuel = totalit + Σ_pft ( w_leaf * cbiol[:,pft] * frac[:,pft] )
f_fuel    = clip( (fine_fuel - lfuel) / (ufuel - lfuel), 0, 1 )

# --- human: ignition saturates, suppression follows CLM's sqrt form
f_ign     = 1 - exp(-a_ig * popd)
f_sup     = 1 - c_sup * (popd / 450)**0.5                  # CLM Li2014 form, clipped ≥ 0
f_human   = f_ign * max(f_sup, f_min)

BA        = 1 - exp( -k_b * fseason^p * dry_mean * f_fuel * f_human )
disturbf  = BA                                             # same variable feeds dynaveg
```

`lfuel`/`ufuel` replace `fuel_ref` (2 params instead of 1, but a ramp rather than a
kink, and directly comparable to CLM's calibrated values). The `(popd/450)^0.5`
suppression form is lifted from `CNFireLi2014Mod.F90` so its exponent and scale
have an external reference rather than being fitted from nothing.

**Fallback if F1 does not fix seasonality:** swap the soil-moisture dryness for a
Nesterov index, using LPJmL's one-line FDI closure (§3b). Cost is one accumulator
and one PFT parameter — still inside budget. Recorded as a planned contingency, not
a v1 feature.

`h_norm` normalises the hump to peak at 1 so `k_b` stays interpretable.
`BA` is reported as `burntFractionAll` (the CMIP7 priority-1 variable).

### 4.3 Fire mortality and emissions

```
C_comb[pft] = BA * mort[pft] * ( cc_leaf * cbiol[:,pft] + cc_wood * cbiow[:,pft] )
C_lit       = BA * cc_lit * totalit
C_total     = Σ_pft C_comb[pft] * frac[:,pft] + C_lit

DM          = C_total / C_frac_DM                          # C_frac_DM = 0.45
E_x         = DM * EF_x[biome]                             # x ∈ {CO2, CO, CH4, N2O}
```

Fine roots are excluded from combustion (they may still die — see §2.3).
`EF_x` from Andreae (2019) by biome class, not the 2001 four-class table.
**One declared unit convention: gC m⁻² yr⁻¹ for carbon species, gN m⁻² yr⁻¹ for N₂O.**
`E_CH4` is handed to the methane module's budget (see `METHANE.md` §7).

### 4.4 Parameters

| Symbol | Meaning | Init | Calibrate? | Source of init |
|---|---|---|---|---|
| `w_mid` | dryness midpoint | 0.5 | no | F77 value |
| `d_crit` | fire-day dryness threshold | 0.5 | **yes** | guess |
| `k_b` | burned-area scale | 0.5 | **yes** | F77 `-0.5` exponent |
| `p` | fire-season exponent | 1.0 | **yes** | guess |
| `lfuel` | lower fuel threshold (gC m⁻²) | 155 | **yes** | CLM `cnfire_const%lfuel` |
| `ufuel` | upper fuel threshold (gC m⁻²) | 1050 | **yes** | CLM `cnfire_const%ufuel` |
| `w_leaf` | live-leaf fine-fuel weight | 0.5 | **yes** | guess |
| `a_ig` | human ignition saturation rate | — | **yes** | fit; cf. CLM `pot_hmn_ign_counts_alpha` |
| `c_sup` | population suppression scale | — | **yes** | CLM Li2014 `(hdm/450)^0.5` form |
| `f_min` | suppression floor | 0.05 | no | prevents total suppression |
| `mort[12]` | PFT fire mortality | 0.5 grass-high, tree-low | **yes** | literature priors |
| `cc_leaf` | leaf combustion completeness | 0.9 | maybe | literature |
| `cc_wood` | wood combustion completeness | 0.2 | maybe | literature |
| `cc_lit` | litter combustion completeness | 0.5 | maybe | literature |

**Scalars: 9. With `mort[12]`: 21.** Slightly over the 20 budget; acceptable
because `mort` is a physically-interpretable vector with strong priors, and it can
be frozen during calibration if it proves ill-conditioned.

### 4.5 New inputs

| Dataset | Resolution | Purpose | On NAS? |
|---|---|---|---|
| Population density (HYDE 3.3 or GPWv4) | 0.5°, time-varying | `f_human` | ☐ to check |
| GFED5 burned area | 0.25°, 1997– | calibration target | ☐ to check |
| FireCCI51 burned area | 250 m → 0.5° | independent check | ☐ to check |

Note: GFED5 reports 93% more burned area than MCD64A1 and 61% more than GFED4s
(small-fire inclusion). **Which product we calibrate against will move our
parameters more than most structural choices** — decide and document it, and
report sensitivity to the choice.

---

## 5. Implementation plan

| Stage | Files touched | Notes |
|---|---|---|
| **F0** | `config.py`, `tests/regression/` | Add `fire_scheme: str = "legacy"`. `"legacy"` must reproduce the current `fire()` bit-exactly. Regression test asserts this. |
| **F1** | new `physics/fire.py`; `state/vegetation.py`; `timestepping/loops.py` | Add `nfireday`, `sdry` to `VegetationState`. Hook accumulator into the daily stats block. Keep `firefac` alive for the legacy path. |
| **F2** | `physics/fire.py` | Switch to `totalit`; add live-leaf term. |
| **F3** | `physics/fire.py`, `io/` | Read population density; add `f_human`. |
| **F4** | `physics/fire.py`, `physics/vegetation_dynamics.py` | PFT-resolved mortality applied inside `dynaveg`'s disturbance block. |
| **F5** | rewrite `bgc/fire_emissions.py`; `state/model_state.py` | Populate the existing empty `IBISState.fire` slot. Andreae 2019 factors. |
| **F6** | new `verify/fire/calibrate_fire.py` | `jax.grad` through the annual loop against GFED5. |

`IBISState` already carries an optional `fire` slot and `config.py` already has
`isimfire`, so no structural surgery is needed.

---

## 6. Verification plan

There is no F77 counterpart for the new scheme, so the two-layer pattern from
`verify/STATUS.md` changes shape:

- **Layer 0 — regression.** `fire_scheme="legacy"` reproduces current `fire()`
  bit-exactly. Non-negotiable; protects the 46-year lockstep.
- **Layer 1 — algorithmic + conservation.** NumPy transcription cross-check.
  Assert: `0 ≤ BA ≤ 1`; every gC leaving a biomass pool appears in litter, in
  `cdisturb`, or in an emitted species; no negative pools; emissions zero when
  `BA = 0`.
- **Layer 2 — observational.**
  - Global annual burned area vs GFED5 (obs range 345–468 Mha).
  - Fire carbon emissions vs GFED5 (FireMIP model range 0.91–4.75 PgC yr⁻¹).
  - Spatial pattern correlation (FireMIP: 7/9 models pass this).
  - **Seasonality and fire season length** — where nearly every model fails, and
    where F1 should show its value. This is the headline metric.
  - Interannual variability (also a known universal weakness).

Append a row to `verify/results/section_summary.csv` and a dashboard row to
`verify/STATUS.md` when each stage lands.

---

## 7. Open questions

1. **Domain — global 0.5° or CONUS?** CONUS makes higher-resolution burned-area
   products usable and simplifies the human term. Global is what the port already
   runs. *Undecided.*
2. **Which burned-area product is the calibration target?** GFED5 vs FireCCI51
   differ by ~60%. *Undecided — must be documented, not defaulted.*
3. Does `f_human` need to be time-varying (HYDE) or is a static map enough for the
   target period?
4. Keep soil-moisture-based dryness, or switch to Nesterov? Soil moisture is
   already computed and physically coupled; Nesterov is the SPITFIRE standard.
   *Leaning: keep soil moisture, revisit if seasonality stays poor.*
5. Should `mort[12]` be calibrated or fixed from literature priors?

---

## 8. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-10 | Extend the existing scheme rather than adopt Li et al. or SPITFIRE | FireMIP: process complexity does not improve skill. Li needs lightning + population + GDP + 4 fire types (~30 params); SPITFIRE needs fuel size classes and sub-daily wind IBIS lacks. |
| 2026-08-10 | Output burned area, not just `disturbf` | Without it the module cannot be validated against anything. |
| 2026-08-10 | Rewrite `bgc/fire_emissions.py` rather than extend | Three conflicting unit conventions and superseded emission factors. |
| 2026-08-10 | After code survey: borrow Li's **ramp structure** and population terms, one fire type only | Source inspection showed CLM's 2865 lines are dominated by the 3 extra fire types (crop, deforestation, peat), not by the core. A single-type Li core is genuinely light. |
| 2026-08-10 | Keep IBIS's exponential burned-fraction closure; do **not** adopt fire-counts × fire-size | `area_burnt.c` needs Rothermel rate-of-spread + elliptical geometry + wind. That is where SPITFIRE's cost lives. |
| 2026-08-10 | Nesterov FDI recorded as a **contingency**, not v1 | LPJmL's `firedangerindex.c` shows the FDI itself is one line and one parameter — cheap to add later if soil-moisture dryness fails on seasonality. |

---

## 9. Build log

> Append one entry per work session: what changed, what was measured, what broke.

_(empty — first entry goes here)_

---

## 10. References

1. Hantson et al. (2020), *Quantitative assessment of fire and vegetation properties in simulations with fire-enabled vegetation models from FireMIP*, Geosci. Model Dev. 13, 3299. https://gmd.copernicus.org/articles/13/3299/2020/
2. *The Fire Modeling Intercomparison Project (FireMIP) for CMIP7*, Geosci. Model Dev. 19, 3989 (2026). https://gmd.copernicus.org/articles/19/3989/2026/
3. Rabin et al. (2017), *FireMIP phase 1: protocols and model descriptions*, Geosci. Model Dev. 10, 1175. https://gmd.copernicus.org/articles/10/1175/2017/gmd-10-1175-2017.pdf
4. Thonicke et al. (2001), *The role of fire disturbance for global vegetation dynamics* (Glob-FIRM), Global Ecol. Biogeogr. 10, 661.
5. Thonicke et al. (2010), *The influence of vegetation, fire spread and fire behaviour on biomass burning and trace gas emissions* (SPITFIRE), Biogeosciences 7, 1991. https://bg.copernicus.org/articles/7/1991/2010/
6. *Sources of uncertainty in the SPITFIRE global fire model: LPJmL-SPITFIRE1.9*, Geosci. Model Dev. 18, 2021 (2025). https://gmd.copernicus.org/articles/18/2021/2025/
7. *ELM2.1-XGBfire1.0*, Geosci. Model Dev. 18, 4103 (2025). https://gmd.copernicus.org/articles/18/4103/2025/
8. Andreae (2019), *Emission of trace gases and aerosols from biomass burning — an updated assessment*, Atmos. Chem. Phys. 19, 8523. https://acp.copernicus.org/articles/19/8523/2019/
9. Chen et al. (2023), *Multi-decadal trends and variability in burned area from GFED5*, Earth Syst. Sci. Data 15, 5227. https://essd.copernicus.org/articles/15/5227/2023/
