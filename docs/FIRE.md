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
| F0 | Legacy path frozen behind flag, regression test | ☑ **done & verified** (2026-08-11) |
| F1 | Fire season length replaces annual-mean `firefac` | ☑ **done & verified** (2026-08-18) |
| **F1b** | **VPD fire danger × vegetation density** (replaces soil-moisture dryness) | ☑ **done & verified** (2026-08-24) |
| F2 | Fine-fuel term (`totalit` + live leaf) | ☑ **done & verified** (2026-08-18) |
| F3 | Human ignition/suppression term | ☑ **done & verified** (2026-08-18) |
| F4 | PFT-specific fire mortality | ☑ **done & verified** (2026-08-18) |
| F5 | Burned area output + trace-gas emissions (Andreae 2019) | ☑ **done & verified** (2026-08-19) |
| F6 | Differentiable calibration against GFED5 | ☑ **global fit done** (2026-08-24); see the entry below -- the honest result is that the a-priori parameters are already competitive |

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
| **0** | ✅ **FIXED 2026-08-18** — `firefac` was never computed in the JAX port; fire was dead code | Discovered 2026-08-16. `firefac` is declared (`state/vegetation.py:70`), initialised to zero (`:124`), read by `fire()` (`vegetation_dynamics.py:408`) — and **never assigned anywhere in `ibis/`**. It is absent from restart I/O. The only assignments in the whole repo are in *tests* and in `verify/vegetation/verify_dynaveg.py:71`, which injects `rng.uniform` values. Consequence: in any real run `firefac ≡ 0`, so `burn = 0` and `disturbf = 1 − exp(0) = **0**`. **With `isimfire=1` the port produces exactly zero fire disturbance.** F77's `stats.f:1180` accumulator was simply never ported. This is why verification passed: the verifier tests `fire()` in isolation with injected inputs, never the accumulation chain feeding it. | **F1** |
| 1 | `firefac` is an **annual mean** of daily dryness (in F77) | A cell with 6 bone-dry + 6 wet months scores identically to one permanently half-dry. Fire is threshold-driven; averaging is the wrong operator. Applies to the F77 side; the port has no accumulator at all (defect 0). | F1 |
| 2 | Fuel is `totlit` — **total** litter | ✅ Verified at `biogeochem.f:924`, commented *"total above and belowground litter"*: `totlit = clitlm+clitls+clitll + clitrm+clitrs+clitrl + clitwm+clitws+clitwl` — the middle three are **root** litter, which cannot burn. `totalit` (`:930`) is the above-ground sum and already exists in `CarbonState`. Live fine fuel (grass, leaves) is ignored entirely. | F2 |
| 3 | No human term | Population density drives both ignition (low density) and suppression (high density). Its absence is the leading explanation for the Africa bias and the missed post-2000 declining trend across CMIP6 models. | F3 |
| 4 | Uniform mortality across PFTs | `disturbf` hits savanna trees and boreal conifers identically, so the fire→vegetation feedback — the whole reason fire is in a DGVM — is wrong. | F4 |
| 5 | Output is not burned area | `disturbf` has no observational counterpart, so no version of this module can be validated or falsified. | F5 |
| 6 | No trace-gas emissions | No CO₂/CH₄/N₂O from fire; nothing to hand to the methane budget. | F5 |
| 7 | `bgc/fire_emissions.py` is dead scaffolding | Never called (only a dormant `enable_fire_emissions=False`); Andreae & Merlet (2001) 4-class factors; state field documented gC m⁻² yr⁻¹, code returns kgC m⁻² yr⁻¹, `nee.py` consumes it as gC m⁻² d⁻¹. | F5 (rewrite, do not extend) |

### 2.3 Resolved 2026-08-12 (second audit)

- [x] **`cdisturb` is emitted to the atmosphere, roots included, and none of it
      reaches litter.** `vegetation.f:433` accumulates
      `(cbiol + cbiow + cbior) × (disturbf + disturbo)`, and `vegetation.f:481`
      does `ayneetot(i) = ayneetot(i) - cdisturb(i)` — carbon leaves the ecosystem.
      Sitting directly above that code is an unresolved author TODO:

      ```
      **** DTP 2000/04/22 QUESTION:
      * Shouldn't a portion of the destroyed material be added to litter fall?
      ```

      So IBIS has **combusted fine roots for 26 years**, and the original authors
      knew it was questionable. Consequences: (a) F5 emissions must exclude
      `cbior`; (b) routing roots to litter instead would change `ayneetot` and the
      whole carbon balance, so it must sit behind `fire_scheme`, never in the
      legacy path; (c) this is an *inherited* defect, worth reporting as a finding
      rather than silently fixing.

- [x] **`firefac` timing.** `sumyear` is called every timestep
      (`main.f:437`), `nytimes` increments per call, `rwork = 1/float(nytimes)`
      (`stats.f:960,966`). So `firefac` is a running mean over **all ~17 520
      timesteps in the year**, not over days. §2.1's earlier wording
      ("annual mean of daily dryness") was imprecise.
      → **F1 must state its accumulation frequency explicitly.** `sumday`
      (`stats.f:300`) is the natural daily hook; counting "fire days" at the
      timestep level would need a ÷`nsteps_per_day` conversion and would
      double-count within a day.

- [ ] Whether the annual `dynaveg` call site (`main.f:500`) can accept a
      daily-accumulated input without perturbing verified `dynaveg` residuals.
      *Still open — resolve at F1.*

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

**2. Li's occurrence core is a handful of closed-form factors; the bulk of the code
is the other three fire types.** Verified line by line 2026-08-12 (§4.4b):

```fortran
fb     = max(0, min(1, (fuelc - lfuel)/(ufuel - lfuel)))            ! :582  ramp
fire_m = exp(-pi*(m/0.69)**2)                                       ! :584  GAUSSIAN in moisture
         * (1 - max(0, min(1, (rh - rh_low)/(rh_hgh - rh_low))))    ! :585  x RH ramp
lpop   = 0.2 + 0.8*exp(-pi*(hdm/450)**0.5)                          ! :440  suppression, floor 0.2
lgdp   = 0.1 + 0.9*exp(-pi*(gdp/8)**0.5)                            ! :436  GDP, shrub/grass
nfire  = ig/secsphr * fb * fire_m * lgdp                            ! :591
```

with calibrated constants `lfuel=75`, `ufuel=650` gC m⁻², `rh_low=30`, `rh_hgh=80` %,
`bt_min=0.3`, `bt_max=0.7`, `pot_hmn_ign_counts_alpha=0.0035` /person/month
(`CNFireBaseMod.F90:55-64`).

⚠ Correction to the first draft of this section: only **fuel availability** is a
ramp. **Combustibility is a Gaussian in fuel moisture** times an RH ramp, and the
suppression terms are **exponentials with built-in floors**, not linear ramps.

Then `cli_scale` (deforestation), `cropfire_a1` (cropland), `boreal_peatfire_c` /
`non_boreal_peatfire_c` (peat) add the three extra fire types — **that is where the
2865 lines go.** A single-fire-type core is cheap.

**Revised borrowing decision:** take `fb` and `lpop` **verbatim with CLM's own
constants**, plus linear human ignition, with **one fire type only**; keep IBIS's
existing exponential burned-fraction closure instead of fire-counts × fire-size.
Skip `fire_m` for now — IBIS has no prognostic RH at the fire timestep (§4.2).

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

Revised twice: first after the code survey (§3b), then again after verifying every
borrowed expression against CTSM source on 2026-08-12 (§4.5). **The first draft of
this section was wrong in two places** — corrected below.

```
fseason   = nfireday / ndaypy                              # fire season fraction
dry_mean  = sdry / max(nfireday, 1)                        # mean severity in season

# --- fuel availability: VERBATIM from CLM CNFireLi2014Mod.F90:582
#     !! UNITS: IBIS litter pools are kg_C m-2 (comveg.h:132,137); CLM's
#     !! lfuel/ufuel are gC m-2. Convert CLM's values; do NOT mix (1000x error).
#     !! NO frac weighting -- cbiol is ALREADY per grid-cell area (see 4.4d)
fine_fuel = totalit + Σ_pft ( w_leaf[pft] * cbiol[:,pft] )            # kgC m-2
f_fuel    = clip( (fine_fuel - lfuel) / (ufuel - lfuel), 0, 1 )       # lfuel/ufuel in kgC m-2

# --- human ignition: linear in population density, as CLM (alpha per person/month)
ign       = ig0 + a_ig * popd

# --- human suppression: VERBATIM from CLM CNFireLi2014Mod.F90:439-440
f_sup     = 0.2 + 0.8 * exp( -pi * (popd / 450)**0.5 )     # bounded [0.2, 1.0] by construction

BA        = 1 - exp( -k_b * fseason^p * dry_mean * f_fuel * ign * f_sup )
disturbf  = BA                                             # same variable feeds dynaveg
```

`f_fuel` is CLM's `fb` exactly. `f_sup` is CLM's `lpop` exactly — note it carries
its **own floor of 0.2**, so the ad-hoc `f_min` in the first draft is unnecessary
and has been dropped. Ignition is linear in population density, matching CLM's
`pot_hmn_ign_counts_alpha`, with `ig0` standing in for the lightning term IBIS has
no data for.

`BA` is reported as `burntFractionAll` (the CMIP7 priority-1 variable).

**CLM's fuel combustibility — now a live option, not excluded.** CLM computes

```fortran
fire_m = exp(-pi*(m/0.69)**2) * (1 - clip((rh - rh_low)/(rh_hgh - rh_low), 0, 1))
```

a Gaussian in fuel moisture times an RH ramp (`CNFireLi2014Mod.F90:584-585`).

⚠ **Reversed on 2026-08-12.** The first draft excluded this on the grounds that
"IBIS has no prognostic relative humidity". **That was false.** IBIS carries
`rh(npoi)` — relative humidity in % — in common block `comatm3`
(`comatm.h:25`), and the JAX port carries it as `AtmState.rh`
(`state/atmosphere.py:41`). Wind `ua(npoi)` is there too (`comatm.h:21`).

So `fire_m` is implementable with **no new inputs**. It is held back from v1 only
because `fseason^p · dry_mean` is the term F1 exists to introduce and the two
should not be conflated in the same step. **Add `fire_m` at F1b and A/B test it
against the season-length term** — the comparison is now cheap and worth doing.

**Fallback if F1 does not fix seasonality:** swap the soil-moisture dryness for a
Nesterov index, using LPJmL's one-line FDI closure (§3b). Cost is one accumulator
and one PFT parameter — still inside budget. A planned contingency, not a v1 feature.

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
| `lfuel` | lower fuel threshold (**kgC m⁻²**) | **0.075** | **yes** | ✅ CTSM `CNFireBaseMod.F90:55` = 75 gC m⁻², **converted** |
| `ufuel` | upper fuel threshold (**kgC m⁻²**) | **0.650** | **yes** | ✅ CTSM `CNFireBaseMod.F90:56` = 650 gC m⁻², **converted** |
| `w_leaf` | live-leaf fine-fuel weight | 0.5 | **yes** | guess |
| `ig0` | background (lightning) ignition | — | **yes** | fit; IBIS has no lightning data |
| `a_ig` | human ignition rate (/person/month) | **0.0035** | **yes** | ✅ CTSM `pot_hmn_ign_counts_alpha`, `CNFireBaseMod.F90:64` |
| `mort[12]` | PFT fire mortality | 0.5 grass-high, tree-low | **yes** | literature priors; cf. FATES `bark_scaler` |
| `cc_leaf` | leaf combustion completeness | 0.9 | maybe | literature |
| `cc_wood` | wood combustion completeness | 0.2 | maybe | literature |
| `cc_lit` | litter combustion completeness | 0.5 | maybe | literature |

`f_sup` has **no free parameters** — the 0.2/0.8/450 constants are CLM's calibrated
values, used verbatim. `c_sup` and `f_min` from the first draft are deleted.

**Scalars: 9. With `mort[12]`: 21.** Slightly over the 20 budget; acceptable because
`mort` is a physically-interpretable vector with strong priors, and can be frozen
during calibration if ill-conditioned.

Other CTSM constants verified in passing, available if needed:
`rh_low = 30.0 %`, `rh_hgh = 80.0 %`, `bt_min = 0.3`, `bt_max = 0.7`
(`CNFireBaseMod.F90:58-61`).

### 4.4b Provenance audit (2026-08-12)

Every borrowed expression re-checked against primary source. Recording what was
verified and what was not, so nothing enters a paper on the strength of a summary.

| Claim | Status |
|---|---|
| `fb = clip((fuel - lfuel)/(ufuel - lfuel), 0, 1)` | ✅ verbatim, `CNFireLi2014Mod.F90:582` |
| `lpop = 0.2 + 0.8·exp(−π·(hdm/450)^0.5)` | ✅ verbatim, `CNFireLi2014Mod.F90:439-440` — **first draft had this wrong** |
| `lfuel=75`, `ufuel=650`, `alpha=0.0035` | ✅ `CNFireBaseMod.F90:55,56,64` — **first draft said 155 / 1050** |
| `fire_m` Gaussian in moisture × RH ramp | ✅ `CNFireLi2014Mod.F90:584-585` — first draft mischaracterised it as a ramp |
| IBIS `totlit` includes root litter | ✅ `biogeochem.f:924` — "total above and belowground litter", sums `clitrm/clitrs/clitrl`. `totalit` (`:930`) is above-ground only. F2 confirmed. |
| FATES per-PFT fire resistance | ✅ `SFMainMod.F90:570` `CalculatePostFireMortality` → `CambialMortality(bark_scaler)` + crown scorch |
| Nesterov FDI one-liner | ✅ LPJmL `firedangerindex.c` (81 lines) |
| Glob-FIRM's exact burned-fraction equation | ⚠ **NOT verified** — Thonicke et al. (2001) is paywalled (Wiley) and secondary sources describe it only as "an empirical relationship between fire season length and area burned". **We take the *concept* (season length, not mean dryness); the closure is IBIS's own exponential.** Do not cite a Glob-FIRM formula we have not read. |
| Andreae (2019) emission factors | ✅ **Verified 2026-08-18** — Table 1 downloaded (`acp-19-8523-2019-t01.xlsx`, open access) and extracted to `inputs/fire/emission_factors/andreae2019_ef.csv`. The abstract-derived figures were right: CH₄ 2.7 g/kg savanna → 9.1 g/kg peat. Full set below. |

### 4.4e Andreae (2019) Table 1 — open-vegetation fire (g species per kg dry matter)

| species | savanna/grass | tropical forest | temperate forest | boreal forest | peat | agri. residue |
|---|---|---|---|---|---|---|
| CO₂ | 1660 | 1620 | 1570 | 1530 | 1590 | 1430 |
| CO | 69 | 104 | 113 | 121 | 260 | 76 |
| CH₄ | 2.7 | 6.5 | 5.2 | 5.5 | 9.1 | 5.7 |
| N₂O | 0.17 | 0.20 | 0.25 | 0.24 | **no data** | 0.09 |
| NOx (as NO) | 2.5 | 2.8 | 3.0 | 1.18 | 1.2 | 2.4 |

Stored with SD and study counts in `andreae2019_ef.csv`. The spreadsheet's
second block (row 135 on) covers biofuels, dung, charcoal, garbage and lab
studies — **not** open vegetation fire, and deliberately not extracted.

⚠ **N₂O for peat is absent from the source**, marked `–`. It must be handled
explicitly at F5, not silently filled with zero or a neighbouring biome's value.

Mapping to IBIS's 12 PFTs is an F5 decision: the four vegetation categories map
naturally onto tropical / temperate / boreal trees and grass, but IBIS has no
peat or cropland representation, so those two columns have no home yet.

### 4.4c Second audit — assumptions about IBIS itself (2026-08-12)

The first audit checked *borrowed* expressions. This one checked what we assumed
about **IBIS**, which is where a wrong assumption actually breaks the build. Four
more errors found:

| # | Assumption | Reality | Impact |
|---|---|---|---|
| 1 | `lfuel`/`ufuel` usable as-is | **IBIS litter is kg_C m⁻²** (`comveg.h:132,137`); CLM's are gC m⁻². | **1000× error** if implemented literally. Converted to 0.075 / 0.650 kgC m⁻². Sanity check: IBIS's own threshold 0.200 kgC m⁻² = 200 gC m⁻² sits inside CLM's 75–650 range ✓ |
| 2 | "IBIS has no relative humidity" | **False.** `rh(npoi)` in `comatm.h:25`, `AtmState.rh` in the port. Wind `ua(npoi)` too. | Reversed the decision to skip `fire_m`; now an F1b A/B test |
| 3 | `firefac` = mean of *daily* dryness | Mean over **every timestep** (`sumyear` per timestep, `main.f:437`) | F1 must declare its accumulation frequency; `sumday` is the right hook |
| 4 | `cdisturb` might be mortality-to-litter | It is **subtracted from NEE** (`vegetation.f:481`) — fully emitted, roots included, with a 2000-era author TODO questioning it | F5 must exclude roots; any fix changes the carbon balance and must stay behind the flag |

Confirmed correct on re-check: soil layer thicknesses 0.10/0.15/0.25/0.50/1.00/2.00 m
(= 4.0 m), read from `fortran/params.soi` via `readpars.f:221` — as stated.

### 4.4d Third audit — variable semantics and data availability (2026-08-12)

| # | Assumption | Reality | Impact |
|---|---|---|---|
| 1 | Fine fuel = `Σ w_leaf · cbiol · frac` | ❌ **`frac` weighting is wrong.** `cbiol(npoi,npft)` is **already per grid-cell area**. Proof: `totlaiu(i) = plai(i,1)+…+plai(i,8)` is a plain sum with no weighting (`vegetation.f:494`), and `plai = cbiol × specla`; conversely `vegetation.f:551` *divides* by `fu` to reach canopy area. Note `frac` is documented as "fraction of **canopy** occupied by each PFT" (`comveg.h:197`) — a canopy fraction, not a grid fraction. | Multiplying by `frac` would have under-counted fuel by roughly the canopy fraction. **`frac` removed from §4.2.** |
| 2 | Background disturbance value | ✅ `vegetation.f:403-404` sets `disturbf = 0.005` **and** `disturbo = 0.005` unconditionally; `fire()` then overwrites `disturbf` only when `isimfire==1`. JAX port faithful. | ⚠ But `cdisturb` accumulates `(disturbf + disturbo)`. **Fire emissions must use `disturbf` alone**, or background mortality gets attributed to fire. §4.3 already uses `BA` only ✓ — now stated explicitly. |
| 3 | `ndaypy` exists for `fseason` | ✅ `compar.h:126`, plus `ndaypm(12)`. | `fseason = nfireday/ndaypy` is valid as written. |
| 4 | Population density must be downloaded | ⚠ **Mixed.** `/shared/nas2/Hangkai/ibis/inputs/wiemip/population/` exists **but is empty** (0 files, created 2026-06-15). `crujra`, `ndep`, `ozone` are likewise empty; `co2` (10 files) and `luh2_gcb2025` have data. | The WIEMIP protocol already reserves the right slot for it. **F3 is blocked on filling that directory**, not on deciding where it goes. |
| 5 | No burned-area product available | ✅ **Found: 19 GB of MCD64A1 v061** at `/shared/nas2/Chuanfeng/fire_data/data/burned_area/` — `ori_data/` HDF tiles (h07–h11, v04–v05 = **CONUS**) from 2009, plus `preprocessed/*.npy` monthlies. | **F5/F6 have a calibration target on the same NAS, no download.** ⚠ But GFED5 reports **93 % more** burned area than MCD64A1 — calibrating against MCD64A1 gives a systematically different answer. Choose and document (§7 Q2). Note the tiles are CONUS-only, which bears on the domain question (§7 Q1). |

### 4.5 New inputs

| Dataset | Resolution | Purpose | On NAS? (checked 2026-08-12) |
|---|---|---|---|
| Population density (HYDE 3.3 or GPWv4) | 0.5°, time-varying | `f_human` (F3) | ⚠ **slot exists, empty** — `inputs/wiemip/population/`, 0 files |
| **MCD64A1 v061** | 500 m HDF tiles + monthly `.npy` | **calibration target, available now** | ✅ **19 GB**, `/shared/nas2/Chuanfeng/fire_data/data/burned_area/` — CONUS tiles h07–h11/v04–v05, from 2009 |
| GFED5 burned area | 0.25°, 1997– | preferred global target | ☐ not present — ~93% higher than MCD64A1 |
| FireCCI51 burned area | 250 m → 0.5° | independent check | ☐ not present |

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
| 2026-08-12 | Use CLM's `fb` and `lpop` **verbatim with CLM's constants** rather than re-deriving similar forms | Provenance audit (§4.4b) found the first draft had invented a suppression form and mis-stated `lfuel`/`ufuel` by ~2×. Copying the calibrated expression removes two free parameters and gives every constant an external reference. |
| 2026-08-12 | Do **not** cite a Glob-FIRM equation | Thonicke et al. (2001) is paywalled and could not be verified. We take the concept (season length replaces mean dryness) and use IBIS's own closure. Claiming an unread formula is how errors enter papers. |
| 2026-08-23 | **F1b promoted from contingency to priority; calibration paused** | A parameter-free rank test showed the failure is not calibration: fire-season length correlates **−0.11** with observed burned area across cells (Spearman), and the model's most fire-prone cells sit at mean \|lat\| 39.8° while GFED5's sit at 10.7°. `k_b`/`p` are monotonic and cannot reorder cells, so no fit repairs a 30° displacement. Drüke et al. (2019) hit the same failure in LPJmL4-SPITFIRE and fixed it with VPD: South America NMSE 1.80→0.82, Caatinga 327.8→15.2. Their diagnosis of Nesterov — "a strong but very localized maximum for high temperatures and low humidity" — describes our soil-moisture dryness exactly. |
| 2026-08-23 | Test VPD **offline from GSWP3 first**, before changing the model | GSWP3 carries `tas` and `rhs`, so VPD-based fire danger can be rank-tested against GFED5 without running IBIS at all. Three rounds of calibration fixes each repaired a real defect and left the outcome unchanged; the rank test costs seconds and would have redirected that effort. Cheap decisive tests come first. |
| 2026-08-12 | ~~Skip CLM's `fire_m` combustibility in v1~~ **REVERSED same day** | The stated reason ("IBIS lacks RH") was false — `comatm.h:25` carries `rh(npoi)`. `fire_m` needs no new inputs. Now scheduled as F1b, A/B tested against the season-length term. |
| 2026-08-12 | Carry `lfuel`/`ufuel` in **kgC m⁻²**, not CLM's gC m⁻² | IBIS litter pools are kg_C m⁻². Mixing them is a silent 1000× error — the single most likely way this module would have shipped broken. |
| 2026-08-12 | F1 must declare accumulation frequency; hook `sumday`, not `sumyear` | `firefac` is a per-*timestep* mean, not per-day. Counting "fire days" at timestep resolution would over-count by ~48×. |
| 2026-08-12 | Treat combusted fine roots as an **inherited finding**, not a silent fix | `cdisturb` is subtracted from NEE with roots included, and carries an unresolved 2000-era author TODO. Fixing it changes the carbon balance, so it stays behind `fire_scheme` and gets reported. |

---

## 9. Build log

> Append one entry per work session: what changed, what was measured, what broke.

### 2026-08-11 — F0 complete

**Changed**

- New `python-port/ibis/physics/fire.py`: `fire_legacy()` (verbatim transcription
  of `vegetation.f:591`, with the two magic numbers named `LEGACY_LITTER_REF =
  0.200` and `LEGACY_BURN_COEF = 0.5`), plus a `compute_disturbf(scheme, ...)`
  dispatcher and an `AVAILABLE_SCHEMES` registry.
- `physics/vegetation_dynamics.py`: `fire()` kept as a thin delegating wrapper so
  existing callers and verification scripts are untouched; `dynaveg` unchanged.
- `config.py`: added `fire_scheme: str = "legacy"`.
- New `tests/test_fire_legacy.py` — 17 tests.

**Measured**

- **17/17 pass.** The refactor is bit-exact: 0 of 20 007 sampled values differ
  from a verbatim copy of the pre-F0 inline expression (commit 605800c).
- Agreement with a pure-Python `math` transcription of the Fortran is to
  **machine precision but not bit-exact** — JAX's `exp` and glibc's `exp` differ
  by up to 1 ULP (max |diff| 2.2e-16 over 200 k samples; 85% of values identical).
  The test asserts ≤ 4 eps. Do not claim more than that anywhere.
- Full suite: **465 passed, 7 failed**. All 7 reproduce identically on a clean
  stashed tree, so they are **pre-existing and unrelated to F0** (`test_canopy` ×2,
  `test_f77_verification` ×4, `test_stomata::test_jit_compatible`). Cause is
  library drift, not physics — e.g. `TypeError: unhashable type:
  'jaxlib._jax.ArrayImpl'` from newer Equinox.

**Environment** *(relocated to nas2, 2026-08-12 — see `docs/ENVIRONMENT.md`)*

The `verify/.venv` referenced in `verify/STATUS.md` did not survive the move to
shared storage. Rebuilt at **`/shared/nas2/Hangkai/ibis-env`**. CPU JAX only:

```
jax 0.11.0 | jaxlib 0.11.0 | equinox 0.13.8 | numpy 2.5.2 | pytest 9.1.1
```

⚠ These are **much newer than the versions the port was verified against** and are
the likely cause of the 7 pre-existing failures. Before trusting any numerical
result from this environment, pin to the versions the 46-year lockstep used, or
re-run the lockstep and re-baseline. Run tests with:

```
cd /shared/nas2/Hangkai/ibis-port/python-port
/shared/nas2/Hangkai/ibis-env/ibis-python -m pytest tests/ -q
```

**Also fixed:** `core.fileMode = false` in `.git/config`, removing 444 phantom
modified files caused by NAS permission bits. `git status` is clean and branch
switching works again.

**Not done / deferred**

- No `dynaveg`-level regression test. F0 does not change `dynaveg` — the call site
  is byte-identical and `fire()` is proven bit-exact — so a full-state test adds
  cost without coverage. **It becomes mandatory at F1**, when behaviour changes.
- `verify/vegetation/verify_dynaveg.py` not run: needs the f2py Fortran wrappers,
  which were not rebuilt on the NAS.

### 2026-08-24 — F6 global fit, and a defect in the loss function

First global fit on the live VPD accumulators: 88 484 cells, GFED5 `Norm`
2001–2005, `use_vpd_danger=True`, `use_fine_fuel=False`.

**A defect found first.** The fit cut the loss from 7.97 to 3.71 — and took
predicted burned area from 3.28 % to **0.34 %** against an observed 3.00 %,
with the rank correlation *unchanged* (+0.746 → +0.743). It was buying a 10×
magnitude collapse, not skill.

The cause is the objective, not the parameters. Log-space MSE is minimised
where the prediction sits at the **geometric** mean of the target; burned
fraction is strongly right-skewed, so its geometric mean lies far below its
arithmetic mean. Setting `balance_zeros=False` changes nothing (0.335 % →
0.308 %) — the skew is the cause, not the zeros, and at 46 % burning cells the
zeros were never rare enough to matter here.

`fire_loss` now carries a `total_weight` term: a squared log error on the
area-mean burned fraction, which pins the global total while the per-cell term
shapes the pattern. Anchoring the global total against GFED is standard in
fire-model calibration. `total_weight=0.0` recovers the old pure-pattern loss.
Sweeping it recovers the magnitude monotonically:

| anchor weight | 0.0 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|
| global bias | ×0.11 | ×0.29 | ×0.54 | ×0.77 |

**The fit, on real skill metrics.** NMSE is the Drüke (2019) metric; area
weighting is `cos(lat)`, so this is burned *area*, not a mean of fractions.

| configuration | NMSE | log-RMSE | rank all/burning | area-wtd burned |
|---|---|---|---|---|
| **a priori** (`k_b`=0.5, `p`=1.0) | 0.731 | 2.833 | **+0.746**/+0.401 | 4.77 % (**×1.03**) |
| fitted, anchor 1.0 | 0.770 | **2.146** | +0.720/+0.415 | 2.50 % (×0.54) |
| fitted, anchor 3.0 | 0.728 | 2.261 | +0.711/+0.417 | 3.61 % (×0.78) |
| **fitted + human (F4), anchor 3.0** | **0.687** | 2.190 | +0.725/+0.412 | 3.71 % (×0.80) |

```
best fit  k_b 0.328   p 2.233   lai_ref_vd 7.446   a_ig 0.326   ign_max 9.843
```

**The honest reading.** Calibration buys much less than expected. The a-priori
parameters — never fitted to anything — already give the best global rank
(+0.746) and a near-perfect global burned area (×1.03). Fitting improves NMSE
only once human ignition is switched on (0.731 → 0.687, ~6 %) and improves
log-RMSE substantially (2.833 → 2.190, ~23 %, i.e. better relative accuracy in
the low-burning boreal cells), but it costs global rank and leaves total burned
area 20 % low.

This is consistent with FireMIP (Hantson 2020): fire-model skill is dominated
by structure, not by parameter values. The large win this month came from
replacing soil-moisture dryness with VPD (+0.258 → +0.630 rank), not from the
optimiser.

**Therefore the shipped defaults stay a priori.** The fitted set is recorded
here as an alternative for anyone who wants minimum log-RMSE and is willing to
accept a 20 % low global total. `use_human=True` is worth enabling either way —
it is the only change that improves NMSE.

**Two API corrections made while doing this.**

- `ign_max` must **not** be added to `PARAM_BOUNDS`. It is already reachable
  through the `ign_span` reparameterisation, which guarantees `ign_max > ig0`;
  a direct bound would let the fit invert that and make fire *decrease* with
  population. A test now pins this.
- `predict_burned_fraction` applies vegetation density whenever `use_vpd_danger`
  is set, so `lai_ref_vd` is fittable (the fit moves it 4.0 → 7.4).

### 2026-08-24 — F1b confirmed with a live run, and one finding reversed

The earlier rank test used VPD computed offline from GSWP3 plus stored
vegetation. This is the same test against the **in-model accumulator**, from a
5-year run (2001–2005, 88 484 cells) off the spun-up carbon state:

| driver | all | burning | (offline) |
|---|---|---|---|
| soil-moisture burn index (F1) | +0.315 | +0.274 | +0.258 |
| in-model VPD `fseason` | +0.316 | +0.355 | +0.254 |
| in-model VPD × severity | +0.322 | +0.354 | +0.243 |
| **in-model VPD × VD** | **+0.630** | **+0.403** | +0.590 |
| in-model VPD × VD × fuel | +0.618 | +0.329 | +0.604 |

```
mean |lat| of top-2000 cells
  GFED5 observed        10.7
  soil-moisture (F1)    39.8
  in-model VPD x VD     14.2
```

**Confirmed and slightly better than the proxy** (+0.630 vs +0.590). The
latitude displacement falls from 29° to ~3.5°.

**Two corrections to what F1b's entry claims.**

1. Its "known limits" said VPD fixes *where* fire happens far better than *how
   much*. Live, burning-cell skill improves **0.274 → 0.403**, not to the 0.32
   the proxy suggested. That caveat was too pessimistic.
2. It called `VD` "largely redundant" with the F2 fuel term. Live, adding fuel
   on top of VD is **actively harmful**: 0.630 → 0.618 overall and 0.403 →
   0.329 among burning cells. **`use_vpd_danger` and `use_fine_fuel` must not
   both be on**, and VD is the better vegetation control. `predict_burned_fraction`
   now applies VD whenever VPD danger is active, and F6 should fit one or the
   other, never both.

**Consequences for F6.** `vpd_ref` and `d_crit_vpd` join `d_crit` in
`FIXED_PARAMS`: they enter only through the accumulators, which the model run
precomputes, so their gradients are identically zero unless the model is re-run
per proposal. `lai_ref_vd` **is** fittable, because vegetation density is
evaluated annually from `plai` rather than accumulated.

### 2026-08-24 — F1b complete: VPD fire danger replaces soil-moisture dryness

**The measurement that drove this.** Parameter-free Spearman rank correlation
against GFED5 2001–2005 over 62 176 cells. `k_b`/`p` are monotonic and cannot
change rank, so this is *driver* skill, independent of any calibration:

| driver | all cells | burning only |
|---|---|---|
| soil-moisture `fseason` (F1) | **−0.112** | +0.166 |
| soil-moisture burn index | +0.258 | +0.222 |
| VPD mean | +0.217 | +0.302 |
| **VPD × vegetation density** | **+0.590** | +0.323 |
| VPD season × VD × fuel | +0.604 | +0.301 |

Latitude of the most fire-prone cells:

```
GFED5 top-2000 burning       |lat| 10.7 deg
soil moisture (F1)           |lat| 39.8 deg   <- 29 deg displaced
VPD x vegetation density     |lat| 11.6 deg   <- within 1 deg
```

Soil-moisture dryness correlates **negatively** with burned area and put the
model's fire in the subtropical dry belt rather than tropical savanna.
Perennially dry subtropical cells score high all year but have nothing to burn;
tropical savannas cure seasonally and burn. **VPD alone is not enough (+0.22) —
the jump to +0.59 comes from multiplying by vegetation density.** Neither
factor separates the two regions; the product does.

This is Drüke et al. (2019)'s fix for LPJmL4-SPITFIRE, where replacing Nesterov
with VPD took South America from NMSE 1.80 → 0.82 and Caatinga from 327.8 →
15.2. Their diagnosis of Nesterov — "a strong but very localized maximum for
high temperatures and low humidity" — describes our soil-moisture index exactly.

**Changed**

- `physics/fire.py`: `vpd_from_state`, `flammability_vpd`,
  `vegetation_density`, `update_fire_accumulators_vpd`; `FireParams` gains
  `vpd_ref`, `d_crit_vpd`, `lai_ref_vd`, static `use_vpd_danger`;
  `fire_seasonlength` takes an optional `vd` multiplier.
- `state/vegetation.py`: `nfirestep_vpd`, `sdry_vpd`.
- `loops.py`: VPD accumulators advanced every timestep **alongside** the
  soil-moisture pair, so one run yields both and the A/B costs no extra
  simulation. Reset annually with the rest.
- `vegetation_dynamics.dynaveg`: `use_vpd_danger` selects the source.
- `tests/test_fire_vpd.py` — 11 tests. Suite **540 → 551**, 0 failures.

**Two implementation points**

VPD comes from `ta`/`qa`/`psurf`, **not** `AtmState.rh` — the GSWP3 driver never
populates `rh`, so using it would have silently returned the initial 70 %
everywhere. And it is built on IBIS's own `thermo.esat`, a bit-exact
transcription of `comsat.h`, rather than a separate Tetens curve, so the model
does not carry two inconsistent saturation curves. A test pins that.

The accumulator is **purely meteorological**; vegetation density is applied
annually in `fire_seasonlength`. VD varies slowly so the product is nearly
identical either way, and this keeps weather and vegetation separable.

**Known limits.** Burning-cell rank skill improves much less (0.22 → 0.32), so
this fixes *where* fire happens far better than *how much* burns where it
already does — likely the fuel-limited/moisture-limited switch at
~800 mm yr⁻¹ (Alvarado et al. 2020), which a single multiplicative form cannot
represent. `VD` is largely redundant with the F2 fuel term (+0.590 vs +0.604
with both), so calibration must not be given free rein over both. Off by
default. The rank test used offline VPD from GSWP3 plus stored vegetation, not a
live run with the in-model accumulator, so the implemented skill still needs a
real run to confirm.

### 2026-08-19 — F6 machinery: gradient-based calibration

**Changed**

- `io/gfed5.py` (new): GFED5 reader — conservative 0.25°→0.5° aggregation,
  coordinate-aligned, returns annual burned *fraction* on the model grid.
- `ml/fire_calibration.py` (new): `FireBatch`, `predict_burned_fraction`,
  `fire_loss`, `fit_fire_params` (optax Adam).
- `tests/test_fire_calibration.py` — 11 tests.

**Gradients reach every fitted parameter.** 11 trainable leaves (10 scalars plus
`mort[12]`), all finite; the structural `use_*` switches are partitioned out via
`eqx.is_inexact_array_like` and provably not differentiated.

**⚠ `mort` has zero gradient with respect to burned area — and that is
correct.** Per-PFT mortality controls what fire does to vegetation and how much
carbon is emitted, not how much of a cell burns. It is therefore **not
identifiable from burned area alone** and needs a vegetation or emissions
target. Pinned as a test so the zero is never mistaken for a broken gradient.

**Synthetic recovery works.** Observations generated from `k_b = 1.7` are fitted
back to 1.7 within 2%. A multi-parameter fit reduces the loss by >50× and
reproduces the observations to 2e-3.

**Loss is area-weighted MSE in log space.** Burned fraction spans orders of
magnitude; a plain MSE is dominated by a handful of savanna cells and learns
nothing about the boreal zone. `eps = 1e-4` keeps zero-burn cells finite.

### GFED5 as the target — one finding that matters

GFED5 splits burned area by cause, and `Total = Norm + Peat + Defo + Crop`
exactly (verified on a sample month). **The default target is `Norm`, not
`Total`**, because IBIS represents neither cropland, nor deforestation fire, nor
peatland. Calibrating against `Total` would force natural-vegetation parameters
to absorb fire the model structurally cannot produce.

That is not a rounding decision: for 2015, Total = **744 Mha** and Norm =
**658 Mha**, so the excluded components are **~12%** of global burned area,
overwhelmingly cropland.

Annual totals read from the archive — 2001 916, 2010 758, 2015 744, 2020 723 Mha
— bracket GFED5's published ~774 Mha yr⁻¹ mean and reproduce its declining
trend. (An earlier note in this log cited 5.4–7.4 Mkm² as the expected range;
that figure is from the 1901–2020 *reconstruction* paper, not GFED5 itself.)

Burned area and burnable area are aggregated conservatively over 2×2 blocks and
only then divided, giving a true area-weighted fraction rather than a mean of
fractions.

**Not done — and this is the honest boundary.** The *machinery* is verified; the
*global fit* is not, because it needs multi-year model runs producing the
per-cell accumulators to calibrate against. That is a production activity, not a
code change. Until it runs, no fitted parameters exist and every default in
`FireParams` remains an uncalibrated prior.

### 2026-08-19 — F5 complete: burned area and emissions

**The module is now falsifiable.** Until this stage IBIS reported `disturbf`, a
disturbance fraction with no observational counterpart — nothing in the world
could confirm or contradict it. F5 reports burned fraction (CMIP7
`burntFractionAll`) and the trace gases from the carbon fire consumes, so F6 can
score the model against GFED5.

**Changed**

- `bgc/fire_emissions.py` **rewritten** (the prototype was quarantined to
  `bgc/_legacy_prototype/`, not extended — §2.2 defect 7).
- `bgc/__init__.py` updated for the new API.
- `tests/test_fire_emissions.py` — 16 tests.

**Emission factors are the verified Andreae (2019) values**, embedded as
constants for speed with the extracted CSV as the source of record. A test
cross-checks constants against that CSV and **fails if they drift apart** — the
kind of silent divergence this module has already suffered twice.

Only the four vegetation categories are used. Peat and cropland are in the
source table but IBIS represents neither, so they have no PFT to map onto; N₂O
for peat is absent from the source anyway. A test asserts they stay out.

PFT → category: 0–1 tropical · 2–4 temperate · 5–7 boreal · 8–11
savanna/grassland. Each cell takes an **LAI-weighted blend** of its PFTs'
factors rather than attributing litter to a dominant PFT, which would be
arbitrary. Verified: a single-PFT cell reproduces that category's row exactly;
a 50/50 cell gives the mean; a bare cell yields no NaN.

**Fine roots are excluded — structurally.** `fire_emissions()` has no `cbior`
argument, so roots cannot be combusted by accident. This is the audit finding
that IBIS has emitted fine-root carbon since 2000 (`vegetation.f:433`, `:481`)
with an unresolved author TODO above the code. F5 fixes the *emission
diagnostic*; the legacy carbon balance is unchanged and stays behind
`fire_scheme`.

**One declared unit convention**, stated once on the state class and not
re-derived: burned fraction 0–1, carbon kgC m⁻² yr⁻¹, species g m⁻² yr⁻¹. The
quarantined prototype carried three conflicting conventions for the same number.
A test pins the conversion chain `species = (carbon_burned / 0.45) × EF`.

**Not done.** `cc_leaf`/`cc_wood`/`cc_litter` (0.90/0.20/0.50) are
literature-typical, uncalibrated. Emissions are a pure diagnostic of state —
they do not feed back into the carbon balance, which still runs through
`cdisturb` unchanged.

### 2026-08-18 — F4 complete: per-PFT fire mortality

**Changed**

- `physics/fire.py`: `DEFAULT_PFT_MORTALITY` (12 values, one per IBIS PFT),
  `FireParams.mort` as a **traced array leaf** so F6 gradients can reach it,
  plus a static `use_pft_mortality` flag.
- `vegetation_dynamics.dynaveg`: the disturbance loop scales the **fire part
  only** by `mort[j]`; `disturbo` stays uniform because it is not fire.
  Fractional cover takes the LAI-weighted mean mortality, since cover is a
  cell-level quantity.
- `tests/test_fire_mortality.py` — 9 tests.

**Priors** (params.veg:27-38 ordering), lowest to highest vulnerability:
temperate conifer and tropical drought-deciduous 0.35 (thick bark, adapted to
frequent low-intensity surface fire) · tropical/temperate broadleaf 0.50 ·
boreal broadleaf 0.70 · boreal conifer deciduous 0.75 · boreal conifer
evergreen 0.80 (thin bark, stand-replacing crown fire) · shrubs 0.85 ·
**grasses 1.00** (entirely top-killed by any fire that passes).

⚠ These are **literature-informed priors, not measured values**. The tests pin
the *ordering* — boreal conifer above temperate conifer, savanna tree below
tropical evergreen, every woody PFT below grass — because the ordering is the
defensible part. The exact numbers are F6 calibration targets and are marked as
such in the source.

Precedent is FATES, which resolves this as crown scorch plus cambial damage
scaled by a per-PFT `bark_scaler` (`SFMainMod.F90:570`
`CalculatePostFireMortality` → `CambialMortality`). F4 collapses that to one
coefficient per PFT — the light-model version of the same idea. A future
refinement could split leaf from wood mortality, as FATES does.

**One design point worth recording:** `mort` scales `disturbf` but not
`disturbo`. Background (non-fire) disturbance should not inherit fire
resistance, and a test asserts that with `disturbf = 0` every PFT sees exactly
the background rate.

**Not done.** Off by default. Uncalibrated. Mortality applies equally to leaf,
wood and root pools; the second audit's finding that roots are combusted and
emitted (`vegetation.f:481`) is F5's problem, not F4's.

### 2026-08-18 — F3 complete: human ignition and suppression

**Changed**

- `physics/fire.py`: `human_suppression()` (verbatim CLM), `human_ignition()`
  (deliberately *not* CLM — see below); `FireParams` gains `ig0`, `a_ig`,
  `ign_max` and a static `use_human`. `fire_seasonlength()` takes optional
  `popd`.
- `io/popdens.py` (new): reader for the CLM 0.5° population stream.
- `vegetation_dynamics.dynaveg`: takes `popd`; raises if `use_human` is set
  without it, rather than silently running without the term.
- `tests/test_fire_human.py` — 9 tests.

**Suppression is CLM verbatim** and pinned against CLM's own expression to
4 eps: `lpop = 0.2 + 0.8·exp(−π·(hdm/450)^0.5)`
(`CNFireLi2014Mod.F90:439-440`). A test asserts it has **no free parameters** —
the 0.2/0.8/450 constants are CLM's calibrated values and `FireParams` must not
be able to shift them.

**Ignition deliberately departs from CLM.** CLM's human ignition is *linear*
in population density. That is safe there because CLM's fuel term collapses
toward zero in dense urban cells, capping the product. IBIS has no such urban
fuel collapse, and the linear form was measured giving a **7× increase** in
burning at 10 000 ind/km² — the opposite of the observed decline at high
density. Replaced with a saturating form bounded in `[ig0, ign_max]`:

```
ign = ig0 + (ign_max − ig0)·(1 − exp(−a_ig·popd))
```

Net human factor is then hump-shaped and bounded in **[0.60, 1.00]** at default
parameters: 1.00 where nobody lives, dipping to ~0.84 at 20 ind/km², settling
at 0.60 in dense population. `a_ig` and `ign_max` are calibration targets, **not
borrowed constants — do not cite them as CLM's.**

**The reader aligns on coordinates, not indices.** `latindex`/`lonindex` come
from whatever ordering the *surta* land-mask file uses
(`netcdf_io.read_input_data`), while the population stream stores latitude
south-to-north. Assuming they agree by index would put Siberia's population in
the Sahara. `model_lat`/`model_lon` are therefore **required arguments**, and
alignment is by nearest coordinate value. Verified: reversing `model_lat`
changes the returned values, so the alignment is doing real work rather than
passing through.

Reader checked against the real file: 251 years (1850–2100), 360×720, year-2000
global mean 9.81 ind/km², max 6939; 1850→2000 shows the expected growth.

**Not done.** `use_human` defaults off. `ig0`, `a_ig`, `ign_max` uncalibrated
(F6). Lightning is still folded into the constant `ig0`; a 0.5° lightning
climatology sits in the same CLM directory if we ever want it, at the cost of a
second input dataset.

### 2026-08-18 — F2 complete: fuel that can actually burn

**Changed**

- `physics/fire.py`: `legacy_fuel_fraction()` (verbatim legacy, kept for the
  legacy path) and `fine_fuel_fraction()`; `FireParams` gains `lfuel`, `ufuel`,
  `w_leaf_woody`, `w_leaf_grass` and a static `use_fine_fuel` flag.
  `fire_seasonlength()` takes an optional precomputed `fuel`, so F1's
  behaviour and tests are untouched.
- `vegetation_dynamics.dynaveg`: computes the fine-fuel term when
  `use_fine_fuel` is set. Default `False` keeps legacy fuel.
- `tests/test_fire_finefuel.py` — 9 tests.

**Both audit findings are now enforced by test, not just documented:**

- **Root litter excluded.** Fuel is built from `totalit` (`biogeochem.f:930`),
  not `totlit` (`:924`, *"total above and belowground litter"*). LPJmL
  independently corroborates — `firepft.c` burns only `agtop` litter.
- **No `frac` weighting.** A test splits the same total leaf carbon across
  eight PFTs and asserts the answer is unchanged; a `frac` weighting would
  change it. `cbiol` is already per grid-cell area (`vegetation.f:494` sums
  `plai` unweighted; `:551` *divides* by `fu` to reach canopy area).
- **Units guarded.** A test asserts `lfuel == 0.075` and `ufuel == 0.650`
  **kgC m⁻²** — the converted CLM values — and that a realistic 0.3 kgC m⁻²
  litter load lands strictly inside the ramp. If anyone pastes CLM's raw
  75/650 gC m⁻² back in, `f_fuel` clips to zero everywhere and fire switches
  off globally and silently. That is the failure this test exists to prevent.

**Behavioural change worth knowing.** Legacy saturates at `totlit = 0.2`
kgC m⁻², so it treats a modest litter layer as the maximum possible fuel —
fuel was effectively never limiting. The CLM ramp does not reach 1 until
0.65 kgC m⁻². Measured at 0.30 kgC m⁻² litter with no live leaf: legacy fuel
term `1.000`, fine-fuel term `0.217`. Fuel limitation is now a real control
rather than a saturated constant.

PFT grouping follows IBIS's own `vegmap`: 0–9 woody (8–9 shrubs), 10–11 grass.
Grass leaf carbon is weighted higher than woody (defaults 1.0 vs 0.3) because
cured grass carries fire; both are calibration targets for F6.

**Not done.** `w_leaf_woody`, `w_leaf_grass`, `lfuel`, `ufuel` are uncalibrated
(F6). `use_fine_fuel` defaults off until F6 can fit the thresholds — turning it
on changes disturbance magnitudes substantially, as the numbers above show.

### 2026-08-18 — F1 complete: fire season length

**Changed**

- `physics/fire.py`: `FireParams(d_crit=0.5, k_b=0.5, p=1.0)`,
  `update_fire_accumulators()`, `fire_seasonlength()`, scheme `"seasonlength"`
  registered and dispatchable.
- `state/vegetation.py`: `nfirestep`, `sdry` per grid cell.
- `timestepping/loops.py`: accumulators advanced each timestep; all three
  annual counters reset in `run_year`; `SimParams.fire` added.
- `physics/vegetation_dynamics.py`: `dynaveg` takes `fire_scheme` /
  `fire_params`, defaulting to legacy so every existing call is unchanged.
- `tests/test_fire_seasonlength.py` — 10 tests.

**The open design question from §2.3 is resolved.** Rather than counting
"fire days" — which would have over-counted ~48x if hooked at `sumyear`'s
per-timestep cadence — F1 measures the **fraction of the year spent above
`d_crit`**, sampled every timestep. That is what fire-season length means, it
needs no day-boundary hook, and it is finer-grained than daily counting.

**Measured**

- **Reduction to legacy holds.** With `d_crit` admitting every timestep and
  `p = 1`, `fseason = 1` and `dry_mean` is the all-timestep mean — i.e.
  `firefac`. Agreement ≤ 8 eps (5 of 6 sample cells bit-identical). Not
  bit-exact by construction: legacy accumulates an incremental running mean
  while F1 divides a sum by a count, so the two round differently. The
  mathematics is identical; only the summation order differs.
- **The discrimination F1 exists for is demonstrated.** Two cells with
  *identical annual-mean dryness* (0.2): one dry at 0.8 for a quarter of the
  year, one mildly dry at 0.2 throughout. Legacy burns them identically — that
  is the defect. F1 burns the first and gives the second exactly zero.
- `fseason` recovers prescribed season fractions (0.25 / 0.50 / 1.00) to
  within one timestep; disturbance is monotonic in season length at equal
  severity; zero season gives zero disturbance regardless of fuel load.
- Legacy path re-checked untouched; jit and grad clean.
- Fire tests **36 passed** (10 F1 + 17 F0 + 9 defect-0).

**Note on `p`.** At `p = 1` the product `fseason · dry_mean` equals the mean
dryness *restricted to fire-prone timesteps* — a threshold-filtered mean. Season
length and severity only separate as independent controls once `p ≠ 1`. `p` is
therefore one of the more interesting knobs for F6 calibration, not a formality.

**Not done.** The fuel term is still legacy `min(1, totlit/0.2)`, including root
litter — F2 replaces it. `d_crit`, `k_b` and `p` remain uncalibrated guesses.

### 2026-08-18 — defect 0 fixed: the fuel-dryness accumulator is ported

Ported F77 `stats.f:1174-1180`, which had no JAX counterpart.

**Added** (`physics/fire.py`):
- `fueldry_from_soil(wsoi, wisoi, swilt)` — verbatim `stats.f:1174-1178`. Uses
  the correct water combination `wisoi + (1-wisoi)*wsoi`, not `wsoi + wisoi`.
- `update_firefac(firefac, nytimes, ...)` — verbatim `stats.f:1180` running
  annual mean, `((n-1)*firefac + fueldry)/n`.

**Wired** (`timestepping/loops.py`):
- `timestep_body` now advances `nytimes` and `firefac` every timestep, matching
  F77 where `main.f:437` calls `sumyear` every timestep.
- `run_year` resets `nytimes` to 0 at the year boundary, matching `stats.f:956`.
  `firefac` is deliberately *not* cleared — at `nytimes == 1` the running-mean
  formula returns `fueldry` exactly, which is how F77 re-initialises it.

**State** (`state/vegetation.py`): added scalar `nytimes`, mirroring F77's saved
counter in `sumyear`.

**Tests** (`tests/test_fire_firefac.py`, 9 tests): agreement with a stdlib
transcription of the Fortran to ≤4 eps over 20 000 samples; endpoints and
bounds; the ice combination pinned against the plausible-but-wrong
`wsoi + wisoi`; `nytimes == 1` returns `fueldry` exactly; N updates equal the
arithmetic mean of the N values; jit and grad clean. **The last test asserts the
wiring itself** — that `timestep_body` actually advances `firefac` — which is
the test whose absence let defect 0 survive.

Suite: **472 → 481 passed, 0 failed.**

Two incidental findings: `create_ibis_state` alone is not physically runnable
(`poros == 0` → NaN), so the wiring test reuses the cold-start fixtures from
`tests/test_integration.py`; and `default_sim_params` takes only `npoi`.

**Still to do:** F1 proper (fire season length replacing the annual mean). The
accumulator now exists and is F77-faithful, so F1 becomes what it was always
meant to be — an improvement on a working baseline rather than building one.

### 2026-08-16 — environment green, and a port defect found

**Test suite repaired: 465 passed / 7 failed → 472 passed / 0 failed.**
All seven were **test-vs-source drift**. None was a physics bug and none needed
version pinning — the earlier "library drift" hypothesis was wrong. Two of them
(`test_canopy` turvap) were literally *asserting the bug that verification
finding #13 fixed*. Details in commit `15b1a3c`. The environment at
`/shared/nas2/Hangkai/ibis-env` is therefore trustworthy for judging numerical
differences, and F1 can proceed without re-baselining.

**🔴 Then, looking for F1's accumulator hook: `firefac` is never computed in the
JAX port.** See defect 0 in §2.2. Declared, zeroed, read — never assigned, and
not in restart I/O. In any production run `firefac ≡ 0`, so `disturbf = 0` and
**fire has no effect at all**, even with `isimfire=1`. The F77 `stats.f:1180`
accumulator was never ported.

Why nobody noticed: `verify/vegetation/verify_dynaveg.py:71` injects
`rng.uniform(0,1)` for `firefac`, so it verifies `fire()` as a *function* against
F77 — correctly, at machine precision — while never exercising the chain that
supplies its input. A unit-verified component wired to nothing.

This does **not** invalidate F0: `fire_legacy` still reproduces the F77 formula
bit-exactly *given* `firefac`. The defect is upstream of it.

**Consequences for F1:**

- F1 grows in scope: it must **build** the accumulator, not merely improve it.
  The good news is there is no port-side legacy behaviour to preserve — zero is
  not a behaviour worth protecting — so F1 is less constrained than expected.
- The F77 side still has the real accumulator, so the *reference* semantics are
  defined; we implement to those, then improve.
- The `dynaveg` regression question from §2.3 partly dissolves: any nonzero
  `disturbf` is already a change from the port's status quo.
- **Scope of impact — narrower than first stated.** Checked 2026-08-18: no
  config in `python-port/configs/` (27 files) sets `isimfire`, and the default
  is `0`. **Fire was never switched on in any run**, so no published result
  silently contained broken fire — the runs simply had no fire, which is a
  configuration choice rather than a hidden error. An earlier note in this log
  claimed this "affects any published run where fire was believed active";
  that was an overstatement and is retracted.
- **What it really was: a landmine for whoever first set `isimfire=1`.** In
  `dynaveg`, `disturbf` defaults to 0.005 and the fire branch *replaces* it:

  ```python
  disturbf_val  = 0.005
  disturbf_fire = where(isimfire == 1, fire(firefac, totlit), disturbf_val)
  ```

  With `firefac = 0`, `fire()` returns exactly 0, so enabling fire would have
  **halved** total disturbance (0.005 + 0.005 = 0.010/yr → 0.000 + 0.005 =
  0.005/yr). The likely reading would have been "fire works but is weak",
  not "the input is dead". Background disturbance itself was always correct
  and matches F77's fixed regime (`vegetation.f:403-404`).

### 2026-08-12 — third audit: variable semantics and data

Checked what the remaining variables actually *mean*, and whether the inputs exist.
One more design error, and two useful discoveries.

1. **`frac` weighting was wrong.** `cbiol` is already per grid-cell area — proved by
   `totlaiu = Σ_j plai(i,j)` summing PFTs unweighted (`vegetation.f:494`) while
   `vegetation.f:551` *divides* by `fu` to get canopy area. `frac` is a **canopy**
   fraction (`comveg.h:197`), not a grid fraction. Multiplying by it would have
   under-counted fine fuel. Removed from §4.2.
2. **`disturbf` and `disturbo` are both 0.005** (`vegetation.f:403-404`) and
   `cdisturb` sums them. Fire emissions must use `disturbf` alone or background
   mortality is attributed to fire. §4.3 was already correct; now stated.
3. **Population density has a home but no data.** The WIEMIP input tree already
   reserves `inputs/wiemip/population/` — **empty**. So does `crujra`, `ndep`,
   `ozone`. F3 is blocked on filling a directory that already exists.
4. **A burned-area product is already on the NAS.** 19 GB of MCD64A1 v061 in
   `/shared/nas2/Chuanfeng/fire_data/data/burned_area/`, CONUS tiles from 2009,
   with monthly `.npy` preprocessing done. F5/F6 need no download — but MCD64A1
   runs ~93 % below GFED5, so the choice of target must be explicit.

Confirmed: `ndaypy` exists (`compar.h:126`); the JAX port faithfully reproduces
F77's disturbance defaults.

**Scheduling consequence:** F1 and F2 need no new data and are unblocked. F3
onward needs the population directory filled. F5/F6 can use MCD64A1 today.

### 2026-08-12 — second audit: assumptions about IBIS itself

The first audit checked borrowed expressions; this one checked what the design
assumes about IBIS. **Four more errors, and the worst of the whole exercise.**

1. **Unit mismatch, 1000×.** IBIS litter pools are `kg_C m-2`; CLM's `lfuel`/
   `ufuel` are `gC m-2`. The design carried CLM's raw 75/650 into an IBIS
   expression. Implemented literally, `f_fuel` would have been ~0 everywhere and
   fire would have switched off globally — plausibly without an obvious crash.
   Now 0.075/0.650 kgC m⁻², with the conversion commented in the equation block.
2. **"IBIS has no relative humidity" was false.** `comatm.h:25` carries
   `rh(npoi)` in `comatm3`; the port has `AtmState.rh`. Wind is there too. The
   decision to skip CLM's `fire_m` was made on a false premise and is reversed —
   now F1b with an A/B test.
3. **`firefac` is a per-timestep mean, not per-day.** `sumyear` runs every
   timestep (`main.f:437`). An F1 "fire day" counter hooked there would
   over-count by the number of steps per day (~48).
4. **`cdisturb` is fully emitted, roots included** (`vegetation.f:481`
   `ayneetot -= cdisturb`), with an unresolved author TODO from 2000 asking
   whether some should go to litter instead.

Confirmed correct: soil thicknesses 0.10/0.15/0.25/0.50/1.00/2.00 m from
`params.soi`; `totalit`/`totlit` distinction; everything from the first audit.

Still no code written. Two audits have now changed the design; that is the
argument for having run them before F1 rather than after.

### 2026-08-12 — provenance audit before implementing F1

Re-checked every borrowed expression and constant against primary source before
writing any code. Full result table in §4.4b. **Three errors found in the first
draft of the design, all now corrected:**

1. `lfuel`/`ufuel` were written as 155 / 1050 gC m⁻² "from CLM". Actual CTSM
   defaults are **75 / 650** (`CNFireBaseMod.F90:55-56`) — off by roughly 2×, and
   attributed to a source that had not been opened.
2. The population suppression term was invented: `1 − c·(pop/450)^0.5` with an
   ad-hoc floor. CLM's actual form is
   **`lpop = 0.2 + 0.8·exp(−π·(hdm/450)^0.5)`** (`CNFireLi2014Mod.F90:439-440`),
   which carries its own floor of 0.2. Adopting it verbatim **removes two free
   parameters** (`c_sup`, `f_min`).
3. Li's occurrence core was described as "piecewise-linear ramps". Only fuel
   availability is a ramp; combustibility is a **Gaussian in fuel moisture**
   (`exp(−π(m/0.69)²)`) times an RH ramp.

**Confirmed correct:** the `totlit` defect (`biogeochem.f:924` is explicit — "total
above and belowground litter", summing the three `clitr*` root pools), FATES
per-PFT fire resistance via `CambialMortality(bark_scaler)`, and the LPJmL Nesterov
one-liner.

**Left unverified, and marked as such:** Glob-FIRM's exact burned-fraction equation
(paywalled) and the Andreae 2019 per-biome emission-factor table (abstract only).
Neither is on the F1 critical path; both must be resolved before they appear in a
manuscript.

No code was written this session — the audit changed the design, which is why it
ran first.

### 2026-08-12 — relocated to nas2

Working copy and environment moved from `nas` (97 % full) to `nas2` (17 T free),
next to the 374 G run tree at `/shared/nas2/Hangkai/ibis`. Details and the full
list of CIFS constraints are in `docs/ENVIRONMENT.md`.

- Repo → `/shared/nas2/Hangkai/ibis-port`. `git clone` fails on this mount
  (`chmod on .git/config.lock`), so the tree was rsynced. Verified: `git fsck`
  clean, all three branch tips identical to the source, origin still GitHub,
  `core.fileMode=false` carried over, working tree clean.
- Environment → `/shared/nas2/Hangkai/ibis-env`. Neither `venv` nor `conda` can
  build here (no symlink support), so packages were pip-installed on local disk
  and rsynced, with an `ibis-python` launcher supplying `PYTHONPATH`. Versions
  pinned in `requirements.lock`, identical to the local venv.
- **Re-verified end to end on nas2:** F0 17/17 pass; full suite 465 passed /
  7 failed — byte-identical outcome to local disk, so the relocation changed
  nothing.
- ⚠ The older copy at `/shared/nas/Chenchen/hangkai/ibis-port` is now
  **superseded**. It is identical as of this commit but will drift. Delete it
  once you are satisfied — not removed automatically.

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
