# IBIS-JAX methane module — design & build log

**Status:** built, calibrated and in the IBIS 3.0 release (2026-10-30): the flux every time step (§9) with band
parameters refitted at 48 wetland towers (CALIBRATION.md 80), the learned wetland area (§11) and the carbon partition (§12).
Sections 0-5s are the original design and its findings; D6-D10 the original plan, decision and build logs; 6-12 the later builds.
**Owner:** Hangkai
**Created:** 2026-08-10
**Scope:** add wetland CH₄ emissions to IBIS, which currently has none. Written
from scratch, borrowing structure from CLM5-CH₄ and the two existing IBIS-based
CH₄ models, but deliberately avoiding CLM5's numerics.

---

## 0. Design constraint

Same "light model" budget as `FIRE.md`:

- **No new prognostic PDE and no new implicit solver.** Bulk two-zone, not a
  resolved reaction–diffusion column.
- **At most one new input dataset** (inundated fraction).
- **≤ 15 calibratable parameters.**
- **Runtime cost < 5%** of the timestep budget.
- Defaults off (`enable_methane = False`) so the 46-year F77↔JAX lockstep in
  `verify/STATUS.md` is untouched.

The strategic argument for staying light: global wetland CH₄ is ~153 Tg CH₄ yr⁻¹
with a 136–168 Tg spread across models, and **that spread is dominated by wetland
extent, not by flux parameterisation**. Investing in a 4× more expensive
biogeochemistry while inheriting the same extent uncertainty buys almost nothing.

---

## 1. Build status

| Stage | Description | Status |
|---|---|---|
| M0 | Read TRIPLEX-GHG and IBIS-CH4 papers; extract their parameter values | ☑ **done** (2026-08-21) — see §4.11 |
| M1 | Inundated fraction + water table depth diagnostic | ☑ **done** (2026-08-25) — `io/wad2m.py`, `bgc/methane.py`, 18 tests |
| M2 | **M-lite⁺**: `f_inund · r_CH4 · Rh · Q10^((T−T_ref)/10) · exp(−z/τ)` | ☑ **built and wired** (2026-08-28), 30 tests. Runs inside the timestep, `enable_methane` read by `run_global`. Uncalibrated: 353 Tg/yr and too tropical at the best configuration — §5i–5k |
| M3 | Evaluate M-lite: methane towers + global total | ☑ **done** — tower fits CALIBRATION.md 38, 80; global total CALIBRATION.md 81-84 |
| M4 | **microbial growth/dormancy** (2 params) — the only added process with a replicated skill gain; *only if M3 shows seasonal bias* | ☐ not started |
| M5 | ~~ebullition, plant transport, diffusion~~ — **removed from the plan** (§5f); reinstate only for a question needing pathway partitioning | ✗ dropped |
| M6 | Global evaluation vs Global Methane Budget 2025 | ☑ **done** — 152 Tg CH4 yr⁻¹ against the bottom-up 153 (116-189), 2001-2010 (CALIBRATION.md 84) |
| M7 | Calibration of `r_CH4`, `Q10p`, `z_tau`, `tau_oxid` | ☑ **done** — per-tower CMA-ES fits, band medians (CALIBRATION.md 80), not by gradients |

Gate between tiers: do not start the next tier until the previous one is evaluated
and the comparison is written into §D9.

Legend: ☐ not started · ◐ in progress · ☑ done & verified

---

## 2. Starting point

### 2.1 F77 IBIS 2.6b4

**Nothing.** `grep -i 'methane\|ch4'` over the entire Fortran source returns zero
hits. This module is greenfield — no legacy path to preserve, no bit-exactness
constraint. That is a freedom the fire module does not have.

### 2.2 `python-port/ibis/bgc/_legacy_unwired/wetlands.py` — quarantine, do not extend

250 lines, present since the initial monorepo commit, **never called** (only a
dormant `enable_wetlands: bool = False` in `config.py`), never verified, absent
from `verify/STATUS.md`. Its defects are structural rather than cosmetic:

| Line | Defect |
|---|---|
| `:82` | Docstring claims TOPMODEL; the code is a linear ramp on mean saturation of the top 3 layers. `cs_topo` is declared and never used. No topographic input exists. |
| `:109` | `wtd = -0.3 * (1 - sat)` pins the water table inside the top 30 cm by construction, regardless of hydrology. |
| `:138` | `prod = base * f_T * fwet * (1 + 0.1*Rh)` — Rh is an additive nudge on a prescribed base rate. Every published model makes production *proportional* to anaerobic Rh. |
| `:166` | Oxidation scaled by `(1 - fwet)`: **the wetter the cell, the less oxidation.** Backwards — methanotrophy occurs in the aerobic zone *above the water table within* the wetland. |
| `:190` | `prod` already carries `fwet`; plant transport multiplies by `fwet` again, so that pathway scales as `fwet²`. |
| `:68` | Ebullition threshold 500 gC m⁻² against a base production of 0.15 gC m⁻² d⁻¹ — roughly a decade of unoxidised accumulation to trigger. Effectively dead code. |

**Action:** move to `bgc/_legacy_prototype/`. Keep the `eqx.Module` state-class
layout and the JAX idiom; discard every parameterisation.

### 2.3 What IBIS gives us for free

| Available | Where | Use |
|---|---|---|
| Heterotrophic respiration | `physics/biogeochem.py:235` → `tco2mic` (mol CO₂ m⁻² s⁻¹) | **The substrate for production.** CENTURY soilbgc, verified to 3e-8. |
| Soil temperature profile | `SoilState.tsoi[npoi, 6]` | Q₁₀ for production and oxidation, freeze gating |
| Liquid / ice saturation | `SoilState.wsoi`, `wisoi` | Water table diagnostic, aeration |
| Porosity, layer depths | `SoilState.poros`, `hsoi` | Air-filled porosity, diffusion |
| LAI, root fraction | `VegetationState.lai`, `froot` | Plant-mediated transport |
| Empty state slot | `IBISState.wetland` | Plug-in point, already exists |

### 2.4 What IBIS does **not** have — the real constraints

| Missing | Consequence |
|---|---|
| **Water table** — lower BC is free drainage via `bperm` | The central design problem. See §4.1. |
| Lateral flow / topography | No TOPMODEL index available without a new dataset. |
| Peat / organic soil carbon pool | **Northern peatlands will be structurally underrepresented.** LPJ-WHyMe needed explicit peat hydrology to do them well. Must be stated as a known bias, not hidden. |
| Wetland PFTs (sedge, *Sphagnum*) | Plant-mediated transport must be parameterised from generic LAI rather than aerenchyma-bearing PFT cover. |
| Soil depth beyond 4 m | Acceptable — most CH₄ production is shallow. |
| Soil pH map | `f_pH` optional, default off. |

---

## 3. What we take from other models

| Model | Take | Leave |
|---|---|---|
| **CLM5-CH₄ / CLM4Me** (Riley et al. 2011) | The **equations**: production as `Rh × f_CH4 × f_T × f_pH × f_pE × S`; Michaelis-Menten oxidation; Henry's-law ebullition at 15% partial pressure; aerenchyma form. Parameter values as priors. | The **numerics**: a reaction–diffusion PDE for CH₄ *and* O₂ in *both* inundated and non-inundated fractions = 4 Crank-Nicolson tridiagonal solves per timestep. This is the expensive part and it mostly buys vertical profile detail we cannot validate at 0.5°. |
| **TRIPLEX-GHG** (Zhu et al. 2014) — *built on IBIS* | Direct precedent. Reports that **the CH₄:CO₂ ratio `r` and `Q10` dominate the sensitivity** out of the whole parameter set → these become our primary calibration knobs. Calibrated at 19 sites. Production gated to 0–45 °C. | Its full water-table module, at least initially. |
| **IBIS-CH4** (Xiao et al. 2024) | Second IBIS precedent; 26-site evaluation set; defines reviewer expectations. | Explicit CH₄/CO₂/H₂/O₂ microbial pathways — too heavy for our budget. |
| **LPJ-WHyMe** (Wania et al. 2010) | The lesson that peatlands need dedicated hydrology + PFTs. Informs our stated limitation. | Peat hydrology, permafrost module, two new PFTs. |
| **WETMETH** (2021) | Precedent that a deliberately reduced model is publishable for ESM use. | — |
| **2024 wetland-model review** (16 models) | **Only 25% represent ebullition adequately** (vs 50% production/oxidation, 44% transport). Doing ebullition properly is a cheap differentiator. | — |

---

## 3b. Reference implementation survey (code-level, 2026-08-10)

Source fetched and measured:

| Implementation | Lines | Tunable params | Structure | Forcing |
|---|---|---|---|---|
| **CTSM `ch4Mod.F90`** | **4410** | **35** (`params_type`) | 20 subroutines: `ch4_prod`, `ch4_oxid`, `ch4_aere`, `ch4_ebul`, `ch4_tran` (tridiagonal), plus lake CH4, restart, history, balance checks | soil C/N state, `finundated` **stream file**, pH map, atm CH₄ |
| **JULES / Gedney et al. 2004** | small | **~3** | `CH4 = k · substrate · Q10^((T-T0)/10) · f_wet` | substrate (soil C, NPP or Rh), T, TOPMODEL `f_wet` |
| **WetCHARTs** (Bloom et al. 2017) | data product | **3 ensemble dims** | `CH4 = scale · Rh · q10^(T) · wetland_extent` | Rh from a C model, T, extent maps |
| **This proposal (M-full)** | ~300 | 14 | two-zone bulk | Rh, T, `f_inund` |
| **This proposal (M-lite)** | ~80 | 4 | Gedney/WetCHARTs form | Rh, T, `f_inund` |

### Three findings that change the design

**1. Even CLM prescribes inundation.** `ch4Mod.F90` imports
`ch4FInundatedStreamType` — CLM ships a stream-file path for `finundated`
alongside the prognostic one. Our Phase 1 decision (§4.1) is not a shortcut, it is
what the reference implementation also offers.

**2. The 35 parameters are mostly not the core.** Reading `params_type`: the
production/oxidation core needs `f_ch4`, `q10ch4`, `vmax_ch4_oxid`, `k_m`,
`q10_ch4oxid`, `f_sat` — six. The rest are lake CH₄ (`lake_decomp_fact`,
`q10lakebase`), redox lag (`redoxlag`, `redoxlag_vertical`), sensitivity-test scale
factors (`scale_factor_aere`, `scale_factor_gasdiff`, `scale_factor_liqdiff`,
`cnscalefactor`, `om_frac_sf`), and aerenchyma geometry (`rob`, `porosmin`,
`nongrassporosratio`, `unsat_aere_ratio`). **A 6-parameter core is defensible.**

**3. The very light schemes are still in service and still competitive.**
Gedney et al. (2004) — substrate × Q10 × wetland fraction, ~3 parameters — remains
the JULES default and was still being evaluated against satellite observations in
2022. WetCHARTs uses essentially the same form and reports that **carbon
decomposition rate and wetland extent together account for >80% of the uncertainty**
in wetland CH₄ magnitude, timing and seasonality. Neither oxidation kinetics nor
transport partitioning is in that 80%.

**Consequence:** building the full 14-parameter version first would be spending
effort on the 20%. Restructured below as a tiered design.

---

## 4. Proposed scheme

### 4.0 Tiered design — build M-lite first

| Tier | Form | Params | Purpose |
|---|---|---|---|
| **M-lite** | `F = f_inund · r_CH4 · Rh · Q10p^((T-T_ref)/10)` | 4 | **The v1 default.** Gedney/WetCHARTs form. Establishes the global total and seasonality baseline, and gives calibration something to bite on immediately. |
| **M-mid** | + oxidation, + upland sink | 8 | Adds the CH₄ sink terms. Only build if M-lite's site-level bias is systematic rather than random. |
| **M-full** | + ebullition, plant transport, diffusion partition | 14 | Only build if the science question needs pathway partitioning (isotopes, episodicity, wetland-type contrasts). |

Each tier must be evaluated before the next is built, and the comparison recorded
in §D9. If M-lite matches M-full's global total to within the observational spread,
that is a publishable result in itself and we stop.

The subsections below specify M-full; M-lite is its first term.

Two zones split by water table depth, evaluated on the existing 6 layers.
Daily timestep. No implicit solve — weighted sums only.

### 4.1 Inundation and water table — decide this first

**Phase 1 (default): prescribe extent, diagnose depth.**

```
f_inund  = prescribed from satellite (WAD2M / GIEMS-2 / SWAMPS), per grid cell, monthly

# water table depth WITHIN the inundated fraction, from IBIS's own moisture profile
sat[k]   = wsoi[:,k] * (1 - wisoi[:,k])
deficit  = Σ_k hsoi[k] * max(0, sat_crit - sat[k]) / sat_crit
zwt      = min(deficit, z_max)                       # m below surface, ≥ 0
```

Rationale: extent is the dominant global uncertainty. Prescribing it **isolates
our biogeochemistry so its skill is measurable** — we can state "given observed
extent, our fluxes are X", which is a cleaner scientific claim than co-tuning both.

**Phase 2 (optional): prognostic TOPMODEL-lite** — needed only if the science
question requires future projections where satellite extent does not exist.
Requires a compound topographic index dataset we do not currently have.

### 4.2 Production

```
f_frz[k]  = 0 if tsoi[k] < 273.15 else 1            # and 0 above 318.15 K (TRIPLEX-GHG)
anaer[k]  = fraction of layer k below zwt           # 0..1, linear within the layer
w_rh[k]   = depth weight for Rh (from soil C / root profile), Σ w_rh = 1

P = r_CH4 * Rh * Σ_k [ w_rh[k] * anaer[k] * f_frz[k] * Q10p^((tsoi[k]-T_ref)/10) ]
```

`Rh` is `tco2mic` converted to gC m⁻² d⁻¹. `r_CH4` is the CH₄:CO₂ ratio — CLM5's
`f_CH4` baseline is 0.2. Production occurs **only below the water table**.

Optional, default off: `f_pH = 10^(-0.2235·pH² + 2.7727·pH - 8.6)` (Dunfield et al. 1993).

### 4.3 Oxidation

Single Michaelis-Menten in CH₄, with an aeration factor standing in for the O₂
profile. This is the main deliberate simplification versus CLM5's double-MM:

```
f_aer = clip(zwt / z_aer_ref, 0, 1)                 # thicker aerobic zone as WT drops
O     = Vmax * CH4_aq / (Km + CH4_aq) * f_aer * Q10o^((T_aer - T_ref_o)/10)
```

Dropping the prognostic O₂ profile removes the second PDE entirely.

### 4.4 Transport

Three pathways, as partitioned fluxes rather than resolved physics. Ebullition
is done properly because it is where most models are weak:

```
# Ebullition — Henry's law solubility threshold (after CLM5: bubbles at 15% of p)
k_H(T)   = Henry coefficient, temperature-dependent
C_sat    = 0.15 * p_atm * k_H(T)
E        = max(0, CH4_aq - C_sat) / tau_eb

# Plant-mediated — scaled by LAI, with rhizosphere oxidation loss
Tp       = k_p * (lai / lai_ref) * CH4_aq * (1 - f_ox_rhizo)

# Diffusion — conductance to surface, gated by snow/ice
theta_a  = poros * (1 - sat)                        # air-filled porosity
g_soil   = D0 * theta_a^(10/3) / poros^2 / z_eff    # Millington-Quirk
D        = g_soil * (CH4_aq - CH4_air) * gate_snow_ice
```

### 4.5 Upland sink

Cheap, and a real global term (order 30 Tg CH₄ yr⁻¹):

```
U = k_up * CH4_air / (Km_up + CH4_air) * f_dry * Q10u^(...)     # where zwt is deep
```

### 4.6 Grid-cell flux and pool update

```
dCH4/dt  = P - O - (E + Tp + D)
F_CH4    = f_inund * (E + Tp + D) - (1 - f_inund) * U
```

Explicit Euler at daily timestep with a non-negativity clamp. If the clamp ever
activates in a production run, that is a bug signal, not a fix — log it.

Fire CH₄ from `FIRE.md` §4.3 is added at the budget level in `bgc/nee.py`, not
inside this module.

### 4.7 Parameters

| Symbol | Meaning | Init | Calibrate? | Source |
|---|---|---|---|---|
| `r_CH4` | **gross** CH₄:CO₂ production ratio | **0.23** (bounds **0.10–0.30**) | **yes — primary** | ✅ TRIPLEX-GHG calibrated mean, 19 sites — **restored as gross** now that M-lite⁺ carries an explicit `exp(−z/τ)` oxidation term (§5f). Consistency check: 0.23 × ~0.30 surviving fraction ⇒ net ≈ 0.07, matching CLASSIC's net 0.061 and CLM's 0.20 × 0.30 = 0.06 independently. |
| `Q10p` | **additional** production Q₁₀ (Rh already carries respiration's) | **1.60** | **yes — primary** | ✅ TRIPLEX-GHG calibrated mean (range 1.6–4.5), 19 sites — and independently corroborated by FLUXNET-CH₄'s annual-flux Q₁₀ of **2.57**. Keep. Note three models (JULES, WETMETH, CLASSIC) make Q₁₀ *decline* with temperature and Zhu et al. 2025 reports constant-Q₁₀ forms bias against FLUXNET-CH₄; a temperature-dependent form is an M7 test, not an M2 default. |
| `T_ref` | production reference T | 295.15 K | no | CLM5 `T_B = 22 °C` |
| `sat_crit` | saturation defining "wet" | 0.95 | maybe | guess |
| `z_max` | max diagnosed WT depth | 4.0 m | no | soil column depth |
| `Vmax` | max oxidation rate | 1.25e-5 mol m⁻³ s⁻¹ | **yes — primary** | CLM5 |
| `Km` | MM half-saturation, CH₄ | 5e-3 mol m⁻³ | maybe | CLM5 |
| `Q10o` | oxidation Q₁₀ | 2.0 | maybe | CLM5 (base 12 °C) |
| `z_aer_ref` | aerobic zone scale | 0.1 m | maybe | guess |
| `tau_eb` | ebullition timescale | 1 d | maybe | guess |
| `k_p` | plant transport rate | — | **yes** | fit |
| `lai_ref` | reference LAI | 4.0 | no | guess |
| `f_ox_rhizo` | rhizosphere oxidation loss | 0.5 | maybe | literature |
| `k_up` | upland sink rate | — | **yes** | fit to ~30 Tg yr⁻¹ global |

**Total: 14.** Within budget. Four flagged as primary calibration targets.

### 4.8 New inputs

| Dataset | Purpose | On NAS? |
|---|---|---|
| **WAD2M v2.0** inundated fraction | `f_inund` (Phase 1) | ✅ **acquired 2026-08-21, reader written 2026-08-25** — `inputs/methane/wad2m/`, 499 MB, `Fw(time=252, lat=360, lon=720)`, monthly 2000–2020, **already on IBIS's 0.5° grid** (lat −89.75→89.75, lon −179.75→179.75). Read by `ibis/io/wad2m.py`; the half-cell centre-vs-corner offset is asserted, not assumed |
| FLUXNET-CH4 | site evaluation | ☑ v1.0, 79 sites / 293 site-yr, CC-BY-4.0, fluxnet.org — **not yet downloaded**; ~42 freshwater sites, only ~8 below 30°N |
| Global Methane Budget 2025 ensemble | global benchmark | ☑ Saunois et al. 2025, ESSD 17, 1873 — wetlands 159 [119–203] Tg/yr BU, 165 [145–214] TD (2010–2019); ensemble 157 ± 24 |
| Soil pH map (optional) | `f_pH` | ☐ to check |
| Compound topographic index (Phase 2 only) | TOPMODEL | ☐ not needed yet |

### 4.9 Provenance audit (2026-08-12)

Same audit applied to `FIRE.md` §4.4b. Three tiers of confidence — **the tiering
decision in §4.0 currently rests on the weakest tier**, which must be fixed before
it is defended anywhere.

**✅ Verified against CTSM source** (`ch4Mod.F90`, fetched and read):

| Claim | Where |
|---|---|
| `vmax_ch4_oxid = 45e-6·1000/3600` = 1.25e-5 mol m⁻³ s⁻¹ | `:1406`, hardcoded |
| `k_m = 5e-6·1000` = 5e-3 mol m⁻³ | `:1418`, hardcoded |
| 35 parameters in `params_type` | declaration block |
| Production/oxidation/aeration/ebullition/transport are separate routines | `ch4_prod :2324`, `ch4_oxid :2679`, `ch4_aere :2832`, `ch4_ebul :3202`, `ch4_tran :3345` |
| CLM ships a **prescribed-inundation stream** | `use ch4FInundatedStreamType` `:41` |
| `f_sat = 0.95`, `atmch4 = 1.7e-6`, `rob = 3` | values given in declaration comments |

⚠ **Most CLM CH₄ parameters are *not* in the source** — they are read from a netCDF
parameters file (`readParams`, `:1325`). So `f_ch4`, `q10ch4`, `q10_ch4oxid` and
others come to us from the tech note, not from code.

**⚠ Secondary source only — CLM5 tech note web page**, not verified against the
params file: `f_ch4 = 0.2`; `q10ch4 = 2.0` at `T_B = 22 °C`; the pH factor
`10^(−0.2235pH² + 2.7727pH − 8.6)`; aerenchyma porosity 0.3 grass / 0.1 woody;
ebullition trigger at 15 % of local pressure; the seasonal inundation factor `S`
with `β = 0.2`. Reasonable — a tech note documents the shipped defaults — but they
should be confirmed against `ctsm_paramfile` before publication.

**⚠⚠ Search-summary only — NOT yet read in the primary paper.** These are load
bearing and the weakest link in the whole design:

| Claim | Used to justify | Risk if wrong |
|---|---|---|
| WetCHARTs: decomposition rate + wetland extent = **>80 % of uncertainty** | **The entire M-lite-first tiering and the stop rule (§4.0)** | If oxidation/transport matter more than stated, M-lite is the wrong starting point |
| TRIPLEX-GHG: `r` (CH₄:CO₂) and `Q10` are the **dominant sensitivities** | Choice of the two primary calibration knobs (§4.7) | We would calibrate the wrong parameters |
| Gedney et al. 2004 is a **~3-parameter** scheme still used as the JULES default | The claim that a very light scheme is competitive | Weakens the light-model argument |
| 16-model review: only **25 %** represent ebullition adequately | Decision to "do ebullition properly" (§4.4) | Minor — the cost is ~10 lines either way |
| Global wetland CH₄ **153 (136–168) Tg yr⁻¹** | The global evaluation target (§D6) | Wrong benchmark |

**Action before M1:** read Bloom et al. (2017) and Zhu et al. (2014) in full and
replace the two top rows with verified numbers. This is the M0 stage already in the
build plan — the audit confirms M0 is not optional.

**Not applicable:** no IBIS Fortran counterpart exists to verify against (§2.1),
and no public JAX/Python implementation of any wetland CH₄ scheme was found.

### 4.11 M0 — load-bearing claims verified against the primary papers (2026-08-21)

Both claims the design rests on were checked in the published articles, not in
search summaries. **Both hold**, and the second returned better priors than we had.

**WetCHARTs — verified verbatim.** Bloom et al. (2017), *Geosci. Model Dev.* 10,
2141, abstract:

> "uncertainties in carbon decomposition rates and the wetland extent together
> account for more than 80 % of the dominant uncertainty in the timing,
> magnitude and seasonal variability in wetland CH₄ emissions"

This is the sentence the whole **M-lite-first tiering and stop rule** (§4.0)
depend on. It stands as quoted. Their ensemble varies satellite-derived surface
water extent, precipitation reanalyses, **nine** heterotrophic-respiration
simulations and **three** temperature-dependence parameterisations — i.e.
exactly the two terms M-lite already contains.

**TRIPLEX-GHG — verified, and it upgraded our parameters.** Zhu et al. (2014),
*Geosci. Model Dev.* 7, 981, abstract:

> "the most sensitive parameters to evaluate CH₄ emission processes from
> wetlands are *r* (defined as the CH₄ to CO₂ release ratio) and *Q*₁₀ in the
> CH₄ production process"

and it gives **calibrated values**, which our first draft did not have:

| parameter | calibrated mean | range | basis |
|---|---|---|---|
| `r` (CH₄:CO₂) | **0.23** | 0.1 – 0.7 | 19 sites, ~35 studies |
| `Q₁₀` | **2.48** | 1.6 – 4.5 | 19 sites, ~35 studies |

§4.7 now uses these instead of CLM5's `f_CH4 = 0.2` and a guessed `Q10 = 3.0`.
**TRIPLEX-GHG is built on IBIS**, so its calibration is the closer prior — and
the published ranges give F6-style optimisation sensible bounds rather than
arbitrary ones.

Still not read in full: the mechanism of TRIPLEX-GHG's water-table module, and
IBIS-CH4 (Xiao et al. 2024) — paywalled at Wiley, though reachable through the
UW–Madison library. Neither is on M2's critical path now that the priors are in
hand.

### 4.10 Second audit — assumptions about IBIS itself (2026-08-12)

Ran alongside `FIRE.md` §4.4c, which found a 1000× unit error in the fire design.
The same class of error is the main risk here, so units were checked first.

| # | Assumption | Reality | Impact |
|---|---|---|---|
| 1 | `tco2mic` may only be an annual diagnostic (§D7 open question 3) | ✅ **Resolved — it is a per-timestep field.** `VegetationState.tco2mic` (`state/vegetation.py:66`), computed in `physics/biogeochem.py:235`. | **M2 needs no plumbing work.** The open question is closed. |
| 2 | Rh units | `tco2mic` is **mol CO₂ m⁻² s⁻¹**. `timestepping/loops_gswp3.py:414-415` already converts: `rh_day = tco2mic × 12.0e-3 × 86400` → **kgC m⁻² d⁻¹**. | §4.2 said "gC m⁻² d⁻¹". **Pick one and state it in code.** Reuse the existing conversion rather than writing a second one — the fire module's 1000× near-miss came from exactly this. |
| 3 | Per-cell root profile for plant transport (§4.4 `f_root`) | ⚠ **`froot` has no `npoi` dimension**: `Float[Array, "nsoilay 2"]`, shape (6, 2) (`state/vegetation.py:44`). Root fraction is **spatially uniform** in the port, upper/lower canopy only. | Plant-mediated transport cannot vary by cell through `froot`. Either scale by `lai` (which *is* per-cell) or accept a uniform profile. **Decide at M3, do not assume.** |
| 4 | Soil layers 0.10/0.15/0.25/0.50/1.00/2.00 m, 4 m total | ✅ Confirmed — `fortran/params.soi:33-38`, read via `readpars.f:221`. | Layer weighting `w_rh[k]` can be built on these as stated. |

| 5 | `sat[k] = wsoi·(1−wisoi)` is the liquid saturation | ✅ **Correct, but subtle — document the reasoning.** `wsoi` and `wisoi` are both described as "fraction of soil pore space" (`comsoi.h:41-42`), which reads as if they simply add. They do not. IBIS's own combination is `water = wisoi + (1−wisoi)·wsoi` (`stats.f:1175`), i.e. **`wsoi` is the liquid fraction of the *ice-free* pore space.** So liquid-of-total-pore-space is `(1−wisoi)·wsoi` ✓ and total water is `wisoi + (1−wisoi)·wsoi`. | Writing `wsoi + wisoi` or `wsoi` alone would be wrong and would look right. We deliberately use **liquid only** (ice neither supports methanogenesis nor moves), with freeze gating `f_frz` on top — but this must be stated, not left implicit. |

Also inherited from the fire audit and relevant here: IBIS **does** carry
`rh(npoi)` and `ua(npoi)` (`comatm.h:21,25`) — more forcing is available than the
first draft assumed, should any CH₄ term want it.

**Data availability (checked 2026-08-12):** none of WAD2M / GIEMS-2 / SWAMPS /
FLUXNET-CH4 is present anywhere under `/shared/nas2`, `/shared/nas` or
`/home/hangkai`. The WIEMIP input tree reserves empty slots for `crujra`, `ndep`,
`ozone` and `population` but not for inundation. **M1 is blocked on acquiring an
inundation product** — unlike the fire module, nothing usable is already on disk.
(For contrast, fire found 19 GB of MCD64A1 burned area on the same NAS; see
`FIRE.md` §4.4d.)

---

## 5. Implementation plan

| Stage | Files | Notes |
|---|---|---|
| **M0** | — | Read Zhu et al. 2014 and Xiao et al. 2024 in full. Extract their `r` and `Q10` values as priors. Do not write code first. |
| **M1** | new `bgc/methane.py`; `io/` | `f_inund` reader + `zwt` diagnostic. Standalone-testable. |
| **M2** | `bgc/methane.py` | Production + oxidation. Requires `tco2mic` plumbed out of `soilbgc` — check it is exposed, not just a local diagnostic. |
| **M3** | `bgc/methane.py` | Three transport pathways. |
| **M4** | `bgc/methane.py` | Upland sink. |
| **M5–M6** | `verify/methane/` | Site then global evaluation. |
| **M7** | `verify/methane/calibrate_ch4.py` | `jax.grad` on `r_CH4`, `Q10p`, `Vmax`, `k_p`. |
| all | `config.py`, `state/model_state.py` | `enable_methane: bool = False`; populate the existing `IBISState.wetland` slot. |

---

## 5a. Benchmarks and the WAD2M area question — RESOLVED (2026-08-27)

### The open question from M1 is closed: our numbers were right

M1 flagged that our reader gives a global wetland area of 3.06 Mkm² (Jan 2010)
against Zhang et al.'s published 7.5–13 Mkm², and that methane flux scales
linearly with it. **It is a definitional difference, not an error.**

WAD2M's headline figures are **per-grid-cell statistics summed globally**, not
the global sum in any month:

* `MAmin` / `MAmax` — each cell's own annual minimum / maximum, then summed
* `MALt` — each cell's maximum over the whole record, then summed

Different cells peak in different months (Amazon May–Jun, Congo Nov–Dec, West
Siberia Jun–Jul, Sudd Sep), so those exceed any single month's global total by
about 1.6–1.7×. The WAD2M v2.0 technical update (Zhang & Poulter, Oct 2021)
Fig. 2a plots the *global monthly aggregate* on a 3.0–5.5 Mkm² axis — which is
where our numbers sit.

Computing their metrics their way, from our own file:

| metric | ours (v2.0, 0.5°) | published |
|---|---|---|
| `MAmax` | 7.10 | 7.51 (v1.0, 0.25°); 7.9 v2.0 |
| `MALt` | 12.07 | 13.02 (v1.0); 13.6 v2.0 |
| `MAmin` | **1.72** | **3.51** ← unexplained |
| global monthly total | 2.54 – 4.76 | Fig 2a spans 3.0 – 5.5 |

`MAmax` within 5 % and `MALt` within 7 % across a version and a resolution
change is good agreement, and it validates `io/wad2m.py` end to end.

**`MAmin` at half the published value is not explained.** Coarsening 0.25° →
0.5° should *raise* a per-cell temporal minimum, not halve it. Recorded as open;
it does not affect M-lite, which uses monthly `Fw` directly — the quantity that
does validate.

**No scaling, masking, or add-back is needed.** `Fw` is a fraction of grid-cell
area, so `Fw × cell_area` summed is correct. Rice paddies (0.59 Mkm²) and
lakes/rivers (2.52 Mkm²) are deliberately excluded so wetland CH₄ does not
double-count the inland-water budget.

**Known data defect, and it constrains our evaluation period.** GMB 2025 warns
that SWAMPS v3.2 "has had discontinuity issues over a few tropical hotspots
since 2015", which propagates into WAD2M. Zhang et al. (2025, BG 22, 305)
restricted their diagnostic runs for this reason. **M3 should evaluate on
2000–2014 and treat post-2015 tropical trends as suspect.**

### The number M6 has to hit

Global Methane Budget 2025 (Saunois et al., ESSD 17, 1873–1958), wetlands,
Tg CH₄ yr⁻¹:

| period | bottom-up | top-down |
|---|---|---|
| 2000–2009 | 153 [116–189] | 158 [145–172] |
| **2010–2019** | **159 [119–203]** | **165 [145–214]** |
| 2020 | 161 [131–198] | 175 [151–229] |

The 16-model GCP-CH₄ ensemble gives **157 ± 24** Tg CH₄ yr⁻¹ (prognostic range
117–195). Zhang et al. (2025) report **158 ± 24** for 2010–2020.

**What that spread means for us.** The ensemble 1σ is ±15 %, and the bottom-up
range spans a factor of 1.7. A new model landing anywhere in 119–203 is inside
the existing spread — so hitting the global total is a **necessary but very weak
test**. M3 and M6 must lean on seasonality, latitudinal partitioning and
site-level comparison, not on the global number alone.

## 5c. Evaluation, benchmarks, and the complexity question (research 2026-08-27)

### The headline: methane has its own FireMIP result

**Liu et al. 2026, ACP 26, 1229** (26 Jan 2026) is the direct methane analogue
of Hantson et al. (2020) for fire. 11 prognostic + 16 diagnostic GCP models over
Arctic/boreal North America, fluxes pushed through STILT footprints and compared
against 10 wetland-dominated tower sites:

> "GCP models such as LPJ-wsl, SDGVM, and JULES produce smaller flux magnitudes,
> and each of these models uses simple approaches to simulate CH₄ fluxes."

> "models such as VISIT, JSBACH, and ISAM have the largest flux magnitudes, and
> these models employ more complex equations that include multiple components of
> CH₄ fluxes."

> "in data-sparse regions, added process detail could potentially amplify input
> and parameter uncertainty and enlarge the flux spread."

The simple models both **agree better with atmospheric observations and have
higher R²**. The GCP ensemble "likely overestimates wetland fluxes by a factor
of two or more across Canada and Alaska", and the overshoot sits in the *complex*
models. Independently, **Chang et al. 2023** (GCB) benchmarked 14 bottom-up
models against 42 FLUXNET-CH₄ sites and found "the performance of BU models may
not be determined by the complexity of the biogeochemical process
representations", and that "models with finer grids do not necessarily show an
improved model performance".

**This is direct evidence for the tiered design in §4.0** — stronger than the
WetCHARTs argument M0 verified, because it is a head-to-head skill comparison
rather than an uncertainty attribution.

**And it validates prescribing extent (§4.1): for 7 of 11 models the diagnostic
run — shared, prescribed WAD2M — beat that model's own prognostic run.**

**The honest counter-evidence, recorded so we do not overclaim.** WETCHIMP-WSL
(Bohn et al. 2015) found the opposite over West Siberia: models "that lacked
either soil thermal physics appropriate to the high latitudes or realistic
emissions from unsaturated peatlands tended to be dominated by a single
environmental driver". The defensible statement is *added complexity does not
reliably buy skill and can hurt where the driving data cannot constrain it* —
not "simpler is always better".

### What M3/M6 must actually hit

| target | value | source |
|---|---|---|
| global wetland CH₄ | **158 ± 24** Tg yr⁻¹ (range 119–203) | GCP 16-model, GMB 2025 |
| **IBIS-CH4 precedent** | **152.67** (135.72–167.57) | Xiao et al. 2024 |
| tropics 30°S–30°N | 110 [67–144] — **~70 %** | GCP prognostic 2010–2019 |
| 30–60°N | 34 [17–64] — ~21 % | " |
| 60–90°N | 12 [4–30] — ~8 % | " |
| boreal/Arctic seasonal peak | **July–August**; 60–70 % of May–Oct flux in Jun–Aug | Liu et al. 2026 |
| interannual variability | 4.7 ± 1.5 Tg yr⁻¹ (2010–2020) | Zhang et al. 2025 |
| temperature sensitivity | 4.6 Tg yr⁻¹ °C⁻¹ (range −0.4 to 9.0) | " |

**A spring peak is a known failure mode with names attached: CLM4.5 and CLM5
both peak too early and fail against atmospheric data.** Worth an explicit test.

### Realistic skill expectations — set these before evaluating, not after

| metric | published range | what counts as good |
|---|---|---|
| site-level daily/monthly R² or NSE | **0.10–0.50** process-based; 0.45–0.73 ML | ≥ 0.5 |
| mean seasonal cycle, extratropics | 0.5–0.65 | ≥ 0.6 |
| **mean seasonal cycle, tropics** | **~0 or negative** | near-universal failure |
| site-mean / spatial R² | 0.25–0.54 | ≥ 0.5 |
| **interannual anomalies** | **NSE < 0 universally** | any positive NSE is a result |
| RMSE, northern wetlands | 10–35 nmol m⁻² s⁻¹ | < 30 |

Anchors: **UpCH4** (the ML state of the art) gets site-mean NSE 0.54 but
**tropical R² 0.08 / NSE −0.12**, and **interannual monthly NSE −5.11**.
**X-MethaneWet** reports R² ≈ 0.92 emulating a process model but **0.03–0.26
with sites held out** — spatial generalisation is unsolved by anyone.

**Consequence for our evaluation design.** Hold out *sites*, not observations —
pooled cross-validation leaks site identity. Report IAV separately. Expect the
tropics to fail and say so rather than hiding it in a global mean.

### Evaluation data

**FLUXNET-CH₄ v1.0** (Delwiche et al. 2021, ESSD 13, 3607): 79 sites, 293
site-years, 37°S–69°N, **CC-BY-4.0**, free with registration via
fluxnet.org / AmeriFlux. **The sampling bias must be stated in any paper**: ~42
freshwater-wetland sites, of which only ~8 lie south of 30°N — while ~70 % of
global wetland CH₄ is tropical.

In-dataset benchmarks worth testing against: median annual flux 9.5 gC m⁻² yr⁻¹;
annual-flux **Q₁₀ = 2.57** (close to TRIPLEX-GHG's 2.48, an independent
corroboration of our prior); and a warning for process modellers — "the timing of
peak summer CH₄ emissions does not correlate with the timing for either peak
summer temperature or peak GPP."

### Two cautions about benchmarking against inversions

1. **Top-down "wetlands" silently includes inland freshwaters.** GMB 2025 says
   so explicitly. On a like-for-like basis the 2010–2019 gap is BU 248 vs TD 165,
   not the ~6 Tg the wetland rows imply. Never compare a wetland-only model to TD
   "wetlands" without stating this.
2. **The inversion constraint is weakest where wetlands are**: < ±5 % globally
   but **> ±20 % at 60–90°N**, and wetland emissions often barely move from the
   prior (3 degrees of freedom out of 155 in one TROPOMI inversion).

### Sourcing caveat

Two items are unverified against full text because the publishers block
automated access: the FUMEMIP emergent-constraint figures (Nature Geoscience)
and Xiao et al.'s site-level R²/RMSE (Wiley). The Xiao numbers matter most to us
— it is the other IBIS methane implementation — and should be retrieved manually
through the UW library before being cited.

## 5d. Wetland extent: the definition *is* the uncertainty

Global wetland area spans a **factor of eight** across published products, and
almost all of it is definitional rather than observational:

| definition tier | products | range (Mkm²) |
|---|---|---|
| satellite inundation **incl.** permanent water + rice, mean annual max | GIEMS-1 5.86, GIEMS-2 6.18–6.80, SWAMPS v3.2 8.26 | **5.9 – 8.3** |
| **vegetated wetland only**, mean annual max | GIEMS-MC_ISW 3.80, **WAD2M v1.0 7.51, v2.0 7.9** | **3.8 – 7.9** |
| long-term / intermittent-inclusive max | GIEMS-2 11.7, WAD2M 13.0–13.6, GIEMS-D15 17.3, GLWD v2 18.2 | **11.7 – 18.2** |
| composite incl. groundwater-driven | Tootchi 27.5–29, Hu ~30, Lane 30.5 | **27.5 – 30.5** |

Stripping permanent open water and rice out of GIEMS-2 nearly halves it —
MAmax 6.80 → 3.80 Mkm². The same physical planet, two defensible definitions,
a factor of 1.8.

**Three consequences for us.**

1. **We are on the community-standard product.** GMB 2025 prescribes WAD2M
   v2.0 for its diagnostic protocol — the same file we read, one resolution
   finer. Our extent choice needs no defence beyond naming it.
2. **Our definition must be stated explicitly in any paper**, because WAD2M
   excludes lakes, rivers, reservoirs, rice (0.59 Mkm²), open water (2.52 Mkm²)
   and coastal vegetated systems. GMB's wetland definition matches: "inundated
   or saturated soils or peats where anaerobic conditions below the water table
   lead to CH₄ production… excludes exposed water surfaces… rice agriculture…
   coastal vegetated ecosystems". A model calibrated against WAD2M extent is
   *not* comparable to one calibrated against GLWD v2's 18.2 Mkm².
3. **Prognostic extent is not a solved alternative.** Even tuned to satellite
   flooded area, GMB's prognostic models spread ±25 % (1σ) on peak wetland area.
   Combined with Liu et al. 2026's finding that the diagnostic run beat the
   prognostic run for 7 of 11 models, Phase 2 (TOPMODEL-lite) should stay
   deferred until there is a reason beyond completeness.

GMB 2025's stated **priority #1** for the whole field is "producing a global,
high-resolution map of water-saturated soils and inundated areas emitting CH₄" —
extent is the acknowledged bottleneck, not process detail. That is the same
conclusion §5c reaches from the skill side.

**Residual contamination to be aware of:** SWAMPS retains desert false-positives
(Sahel, Australia) that propagate into WAD2M, on top of the post-2015 tropical
discontinuity already noted.

## 5e. Model structures — and a correction to our own prior

### The finding that changes M2

**`f_CH4` values are not comparable across models unless you check whether
oxidation sits inside or outside the ratio.**

| model | ratio | oxidation |
|---|---|---|
| CLM4Me/CLM5/ELM/JSBACH | 0.20 | **outside** (double Michaelis–Menten) |
| LPJ-WHyMe | 0.10 | outside (O₂ stoichiometry, f_oxid 0.5) |
| LPJ-GUESS (14-site MCMC posterior) | **0.14 ± 0.007** | outside |
| ORCHIDEE-PCH4 | 0.10 (`q_MG`=10) | outside |
| **TRIPLEX-GHG / IBIS-CH4** | **0.23** | **outside** (single MM in CH₄) |
| **CLASSIC** | δ_s 0.135 × α_w 0.45 = **0.061 net** (code: 0.029–0.046) | **INSIDE — net emission factor** |
| JULES | folded into `k_CH4` | **inside** |
| LPX-Bern tropics | 0.024–0.0415 | inside (net) |
| VISIT | 0.50 | outside |

**M-lite has no oxidation term.** So the ratio it needs is a *net* one, and
**TRIPLEX-GHG's 0.23 is gross**. Using 0.23 in a production-only scheme would
roughly double the flux, because ~50 % of gross production is typically
oxidised before it escapes.

**This corrects the prior M0 established.** M0 was right that 0.23 is
TRIPLEX-GHG's calibrated value and right that TRIPLEX-GHG is the closest
precedent — but 0.23 belongs to a scheme that *then removes methane by
oxidation*. The structurally correct comparators for M-lite are the **net**
formulations: CLASSIC 0.135 and JULES. M2 should start from a net ratio near
**0.1–0.15**, with 0.23 recoverable only once M4 adds explicit oxidation.

Recorded as a correction rather than a silent edit, because the M0 commit
message states 0.23 as the value to use.

### A second convergence worth knowing

**0.1 is where independent groups land** for the anoxic:oxic decomposition rate
ratio — LPJ-WHyMe fitted 0.10, ORCHIDEE `q_MG` = 10 ⇒ 0.1, LPJmL6 `r_k` = 0.1.
CLM's 0.2 and VISIT's 0.5 are the outliers. Our M-lite prior sitting near 0.1
is therefore the consensus position, not a low-balled guess.

### Q10: the value is agreed, the *sign of its temperature derivative* is not

| model | production Q₁₀ |
|---|---|
| **TRIPLEX-GHG calibrated** | **2.48** (1.6–4.5) |
| **FLUXNET-CH₄ annual flux** | **2.57** — independent corroboration |
| CLM/ELM | 2, but **additional** to decomposition's 1.5 ⇒ effective ≈ **3.0** |
| JULES | 3.7 at 273.15 K, **declining** as 3.7^(T₀/T) |
| WETMETH | 4.2, **declining** as 1.7 + 2.5·tanh[0.1(308.15−T)] |
| CLASSIC | **1.44 + 0.56·tanh[0.075(46−T)]** ≈ **2.0** below 20 °C |
| LPJ-WHyMe / LPJ-GUESS | **none** — Lloyd & Taylor on decomposition only |

Three models make Q₁₀ *decline* with temperature; Zhu et al. (2023, L&O Letters)
argues on thermodynamic grounds that CH₄:CO₂ should *rise* with it; and Zhu et
al. (2025, ERL) reports that constant-Q₁₀ formulations "produce large biases
across FLUXNET-CH4 because the real temperature response is hysteretic".

**Our 2.48 is well supported and we should keep it for M2** — it is calibrated
on the closest model and independently matched by FLUXNET-CH₄'s 2.57. But a
constant Q₁₀ is a known, named limitation, and M7 should test whether a
temperature-dependent form earns its place.

### Correction to the correction (same day) — the ratio has *three* denominators

The follow-up research corrected two things in what is written above, and one of
them corrects a number I had already written into §4.7 an hour earlier. Recorded
in full because this is now the **second** time ratio comparability has bitten.

**1. "CH₄:CO₂ ratio" is not one axis. It has at least three denominators:**

| model | what the ratio is *of* | conversion |
|---|---|---|
| CLM, LPJ-WHyMe, CLASSIC | **molar CH₄:CO₂** | direct |
| **JSBACH `f_CH4`** | fraction of anaerobically mineralised **carbon atoms** | 0.35 ⇒ CH₄:CO₂ ≈ **0.54**; 0.24 ⇒ ≈0.32 |
| HIMMELI `f_m` | fraction of **anoxic respiration** | — |
| WETMETH `r` | specific rate on the **carbon pool** | not on this axis at all |

So JSBACH is not 3–5× CLASSIC as a naive reading of the table above suggests.
**Roughly half the published `f_ch4` values need transformation before they can
be compared at all** — once for the denominator, once for gross-vs-net.

**2. CLASSIC's net ratio is 0.061, not 0.135.** The formulation is
`E_w = R_h · f_w · α_w · δ_s`, so the effective net ratio is
**δ_s 0.135 × α_w 0.45 = 0.061** (paper). The shipped code uses δ_s = 0.125 and
α_w = 0.23/0.37/0.23 by latitude band ⇒ **0.029–0.046**, undocumented in any
paper. Calibration history: 0.25 → 0.16 → 0.125.

**3. CLASSIC's production Q₁₀ is ≈2.0, not 3.22−0.046·T** — that expression is
CTEM's *maintenance respiration* Q₁₀. The heterotrophic one is
`1.44 + 0.56·tanh[0.075(46−T)]`.

**4. JULES has ~10–12 parameters, not 3**, and a `ch4_substrate` switch that
resolves the 3.7-vs-2 Q₁₀ confusion: with soil carbon as substrate (default) the
pool is time-invariant so the whole sensitivity (3.7) sits on the CH₄ term; with
soil respiration as substrate the substrate already carries Q₁₀ = 2.0 and only
1.5 is added. **The options are not temperature-interchangeable.**

**Consequence for §4.7.** My earlier entry set M-lite's net `r_CH4` = 0.13 by
treating CLASSIC's 0.135 as a net ratio. It is not — the net value is 0.061.
The defensible net comparators are:

```
CLASSIC (paper)        0.061
CLASSIC (code)         0.029 - 0.046
LPX-Bern tropics       0.024 - 0.0415
gross 0.10-0.23 with ~50 % oxidation  ->  0.05 - 0.12
```

**A net prior near 0.06–0.08 with wide bounds, not 0.13.** And the honest
framing is that `r_CH4` in M-lite is an *effective* parameter absorbing
production, oxidation and transport at once — it is the primary calibration
target precisely because no published value transfers cleanly into this form.

### Structural landscape

| complexity | models | params |
|---|---|---|
| **bulk flux, no transport, no explicit oxidation** | **JULES 3, CLASSIC ~4, WETMETH 6** | 3–6 |
| ratio + MM oxidation + 3-pathway transport | TRIPLEX-GHG 13, ORCHIDEE-PCH4 7, LPJ-WHyMe 7, VISIT ~10 | 7–13 |
| full vertical PDE, double MM | **CLM/ELM/JSBACH 35, LPJmL6 ~50** | 35–50 |

A **10× spread in parameter count**, and §5c shows the light end wins on skill.
Note also that much apparent independence is shared ancestry: the Walter &
Heimann (2000) diffusivity polynomials, Schmidt numbers, Bunsen coefficients and
the 750 µmol L⁻¹ ebullition threshold are reused near-verbatim across LPJ,
LPJmL6, CLM, TRIPLEX-GHG, VISIT and ORCHIDEE.

### 2023–2026 developments that bear on the design

* **Moser et al. 2026** (BG 23, 605) — argues the fixed CO₂:CH₄ ratio is itself
  a primary uncertainty and that three pathways (fermentation, acetoclastic,
  hydrogenotrophic) are the minimum to capture ratio dynamics. Only 3 of 40
  models reviewed resolve two pathways. **This is the strongest argument
  against our M-lite form** and should be cited as a stated limitation.
* **LPJmL6** (GMD 19, 7615, 2026) — **abandons the CH₄:CO₂ ratio entirely** for
  competing oxic/anoxic decomposition, and uses Arrhenius rather than Q₁₀.
  Uncalibrated as published.
* **Kallingal et al. 2024/2025** (GMD 17, 2299; BG 22, 4061) — first rigorous
  MCMC constraint on wetland CH₄ parameters; the tightest published posterior on
  CH₄:CO₂ anywhere (0.14 ± 0.007). **Directly relevant to M7**, and the obvious
  comparison for a gradient-based alternative.

### Two implementation traps in CLM, if we ever borrow from it

1. **CLM's `f_ch4` is not 0.2 in effect** — the code computes
   `f_ch4_adj = f_ch4 · q10ch4^((T−22)/10)` and then divides out the
   moisture/N limitation (`/fphr`). At 10 °C the effective ratio is ≈**0.09**.
2. **CLM's Michaelis–Menten constants are hardcoded in current CTSM master** —
   the netCDF reads for `k_m`, `vmax_ch4_oxid` etc. are commented out, so
   editing the params file silently does nothing.

## 5f. DECISION (2026-08-27): M-lite becomes M-lite⁺, and the tiers are re-cut

### The decisive experiment

**Chadburn et al. (2020), GBC 34, e2020GB006678** is a controlled ablation, not
a model intercomparison — which makes it worth more than everything in §5c.
They ran CLM5/CLM4Me (explicit O₂ transport, Michaelis–Menten oxidation
co-limited by O₂ and CH₄, ebullition, aerenchyma, aqueous and gaseous diffusion)
against **CLM-prod.only** — CLM's own production, with JULES's single
`exp(−τz)` factor substituted for *all* of that machinery:

> "The correlation coefficient of the weekly mean methane emissions in CLM and
> CLM-prod.only is between **0.94 and 0.98 (R² 0.89–0.97)**. Thus, we conclude
> that including transport and oxidation processes … **does not significantly
> improve the predictive power of the model on a seasonal time scale**"

and from the abstract: **"a more complex scheme used in recent Earth system
models does not add predictive power."**

One scalar reproduces CLM's entire kinetic scheme to R² ≥ 0.89. Corroborated by
McNorton et al. (2016, GRL), by King et al. (1998) — aerenchyma changes
*magnitude* but not *dynamics* — and by LPJ-Bern/LPJ-WSL, which ignore oxidation
and transport entirely and were not WETCHIMP outliers.

### The design this implies, and how it reconciles the ratio mess

```
F = f_inund · r_CH4 · Rh · Q10p^((T − T_ref)/10) · exp(−z_oxic / τ_oxid)
```

Gedney's production form with WETMETH's one-parameter exponential oxidation.
**Four fitted parameters**: `r_CH4`, `Q10p`, `τ_oxid`, and `lai_ref`-style
scaling if needed. `z_oxic` comes from M1's water-table diagnostic, which is
already built and tested.

**This resolves the CH₄:CO₂ confusion that bit twice today.** With an explicit
oxidation factor, `r_CH4` is a **gross** ratio again — so TRIPLEX-GHG's 0.23 is
the correct prior after all, and the net comparators become a *consistency
check* rather than the target:

```
gross 0.23  x  exp(-z/tau) at Chadburn's tau = 6.5  (~70 % oxidised)
   ->  net ~0.07
   vs CLASSIC net 0.061, LPX-Bern tropics 0.024-0.0415     <- agrees
   vs CLM gross 0.20 x 0.30 = 0.06                          <- agrees
```

Four independent formulations land on a net ratio of 0.06–0.07. **That
agreement is the strongest single piece of evidence in this whole review**, and
it only becomes visible once oxidation is separated out. §4.7 reverts to
`r_CH4` = 0.23 gross, with the net value as a derived diagnostic to check.

### Re-cut tiers

| was | now | why |
|---|---|---|
| M2 = production only | **M2 = M-lite⁺: production × Q₁₀ × f_inund × exp(−z/τ)** | one extra parameter buys CLM-equivalent skill (Chadburn R² 0.89–0.97) |
| M4 = oxidation kinetics + upland sink | **M4 = microbial growth/dormancy** | the *only* added process with a documented, replicated skill gain: seasonal-cycle RMSE improves at **6 of 7 sites** (Chadburn), and it reconciles the seasonal-vs-spatial Q₁₀ discrepancy. JULES-microbe adds just **2** fitted parameters |
| M5 = ebullition, plant transport, diffusion | **M5 = deferred indefinitely** | Chadburn, McNorton 2016 and King 1998 all say these change magnitude, not dynamics — and magnitude is absorbed by a scalar we calibrate anyway. Zhang et al. 2023 shows the sub-15-day band where ebullition lives is unresolved by *all* models regardless of complexity |
| M7 = differentiable calibration | **M7 = unchanged, and it is the novel contribution** | see below |

Michaelis–Menten oxidation kinetics, multi-layer O₂ diffusion, aerenchyma and
ebullition are **removed from the plan**, not merely deferred. Reinstating any of
them now requires a specific science question that needs pathway partitioning.

### The novelty claim, stated narrowly enough to survive review

Systematic search (OpenAlex plus targeted review reading) found:

* `differentiable AND wetland` — **0 relevant works**
* `"differentiable parameter learning"` — **47 works, 100 % hydrology**
* of the **432 works citing Shen et al. (2023)**, the field's authoritative
  differentiable-modelling review, a methane keyword search returns **10 hits,
  none of which is a methane study** (water quality, groundwater, CARDAMOM, a
  lake model, river nitrogen, DSSAT). *Stated precisely: no methane-focused work
  has picked up the differentiable-modelling framework. An earlier draft of this
  section said "zero mention methane", which overstated what the query ran.*
* Raoult et al. (2025, JAMES 17), the current review of LSM parameter
  estimation, lists methane only as a *future* data-stream opportunity

**The telling data point**: ORCHIDAS has a TAF-generated tangent linear and
normally uses L-BFGS-B for carbon-cycle assimilation — but when the same group
calibrated **methane**, they switched to a gradient-free **genetic algorithm**
(Salmon et al. 2022, GMD 15, 2813; 7 parameters, 100 iterations).

Defensible claim: **first end-to-end differentiable land-surface model used for
gradient-based calibration of methane biogeochemistry parameters.** Not "first
differentiable land model" — JAX-CanVeg, NoahPy, DifferLand, ADELM and
clm-ml-jax all landed in 2025–26, and every one stops at water/energy/carbon.
Not "first gradient-based BGC calibration" — CCDAS adjoints date to 2005.

**The baselines are weak**: Kallingal et al. (2025, BG 22, 4061) needed
**100 000 MCMC iterations ≈ 480 CPU-hours** for the LPJ-GUESS CH₄ module across
14 sites. Tsai et al. (2021) report dPL reaching the same RMSE as SCE-UA in
**9.9 min vs 253 min**, and full-CONUS calibration going from a 100-core cluster
for 2–3 days to ~40 min on one GPU.

**And there is a badly-conditioned direction that gradients are made for**:
WETMETH's global emission changes **±37 % for a ±10 % change in τ_oxid**, and
±77–120 % for ±30 %. Random search and perturbed-parameter ensembles handle that
regime badly; a Jacobian handles it trivially. Our M-lite⁺ has exactly this
parameter.

**A controlled precedent in our favour**: Ahmad et al. (2025, GRL 52) found
differentiable calibration of Noah-MP *via an LSTM surrogate* gave only marginal
gains and parameters that did not transfer back to the real model. That is a
**surrogate-gap** failure. A truly end-to-end differentiable IBIS is the
controlled version of that experiment — we differentiate the model itself.

### Open problem handed to us verbatim

McNicol et al. (2023), the UpCH4 paper:

> "To date, process-based models have yet to use networked data, for instance
> FLUXNET-CH₄, for multi-site calibration, in part because this approach is more
> technically challenging than for machine learning models."

Multi-site joint calibration through a differentiable model is exactly that, and
the tooling gap they name is the one the JAX port has already closed.

### Sensitivity analysis independently picks our two parameters

**Chinta et al. (2024, JAMES)** ran Sobol sensitivity on **19 ELM CH₄
parameters across 14 FLUXNET-CH₄ sites**:

> "The production parameters Q₁₀ and f_CH4 emerged as sensitive parameters for
> production **at all sites**."

The top 5–6 parameters account for **> 90 % of emission variance**, and
**13–14 are consistently negligible**. Their published sampling ranges are
tighter than the ones we had:

| parameter | ELM default | Chinta range | ours |
|---|---|---|---|
| Q₁₀ | 2 | **[1.5, 4]** | 2.48, bounds 1.6–4.5 |
| f_CH4 | 0.2 | **[0.1, 0.3]** | 0.23, bounds 0.10–0.70 |

Two independent lines — TRIPLEX-GHG's own sensitivity analysis (M0) and a Sobol
decomposition of a *different* model — select the same two parameters. That is
about as good a justification for a two-parameter primary calibration as this
field offers.

**We should tighten `r_CH4`'s upper bound from 0.70 to ~0.30.** TRIPLEX-GHG's
0.1–0.7 is the spread of its per-site fits; ELM's [0.1, 0.3] is a considered
sampling range, and the multi-model convergence (below) sits at 0.10–0.23.

### Two constraints on Q₁₀ worth recording

**Walter & Heimann's original production Q₁₀ is 6**, and it is the root of the
whole LPJ/CLM/TRIPLEX/VISIT/ORCHIDEE family. **Ringeval et al. dropped it to ~3**
on the explicit argument that 6 conflates substrate production with
methanogenesis, and that **methanogenesis-only studies report 1.2–3.5**.

Our 2.48 sits comfortably inside that range, and is independently matched by
FLUXNET-CH₄'s annual-flux Q₁₀ of 2.57. Ringeval's per-band optima are worth
keeping for a future latitude-varying test: **2.60 boreal / 3.20 temperate /
1.20 tropical**.

### The 0.1 convergence, confirmed and slightly corrected

Salmon et al. compare `q_MG` (CO₂:CH₄, so the inverse) across models directly:
LPJ-WHyMe 10, **CLM4Me 5**, Khvorostyanov 9, Morel 10 — i.e. CH₄:CO₂ of 0.10,
**0.20**, 0.11, 0.10. ORCHIDEE-PCH4's own multi-site posterior is
**q_MG = 9.64 ± 0.8 ⇒ f_ch4 = 0.104**. So the clustering at ~0.1 is real and
CLM's 0.2 is genuinely the outlier — which makes TRIPLEX-GHG's 0.23 the *high*
end of the credible gross range, not the centre.

### Corrections to §5e's table

* **ORCHIDEE-MICT and ORCHIDEE-PEAT have no CH₄ module at all** — verified by
  full-text search. Qiu et al. 2019: "CH₄ and dissolved organic carbon (DOC) are
  not yet included in the model." ORCHIDEE-PCH4 is a *branch* of PEAT.
* **There are two non-overlapping ORCHIDEE CH₄ lineages and GCP uses the older
  one** — every GCP release cites Ringeval et al. (2011), the Walter–Heimann
  lineage, which has **no CH₄:CO₂ ratio at all**. Only PCH4 has one.
* **GCP's headline configuration changed between budgets**: Saunois et al. 2020
  reported the **diagnostic** ensemble at 148 [101–179]; Saunois et al. 2025
  reports the **prognostic** ensemble at 157 [117–195]. When comparing against
  "the GCP bottom-up wetland number", state which.
* Walter & Heimann (2000) itself is paywalled; its constants above come from
  Part 2 and from restatements in TRIPLEX-GHG and PEATLAND-VU — mutually
  consistent but **second-hand**.

### TRIPLEX-GHG's parameters resolve by biome — and they anti-correlate

M0 recorded only the all-site means. The biome breakdown matters more:

| | `r` (CH₄:CO₂) | `Q10p` |
|---|---|---|
| Boreal | 0.18 | 2.03 |
| Temperate | **0.32** | 2.66 |
| **Tropical** | **0.15** | **3.75** |
| all-site mean | 0.23 | 2.48 |

**`r` and `Q10p` move in opposite directions across biomes** — the tropics have
the *lowest* ratio and the *highest* Q₁₀. Two consequences:

1. **Expect the gradient fit to find this trade-off as a ridge.** A joint
   optimisation of `r_CH4` and `Q10p` will be poorly conditioned along the
   direction that trades one against the other. That is worth anticipating in
   M7 rather than discovering as non-convergence — and it is precisely the
   regime where a Jacobian beats random search.
2. **The all-site mean may be the wrong global default.** ~70 % of global flux
   is tropical, where TRIPLEX-GHG fits r = 0.15 and Q₁₀ = 3.75 — not 0.23 and
   2.48. Worth testing a biome-varying pair against the single global pair at
   M3, and it is the natural first application of dPL-style regionalisation
   (attributes → parameters) at M7.

Site extremes for context: Mer Bleue / Ruovesi / Quebec 0.10; Sanjiang 0.45;
Loch Vale, Colorado 0.70.

### Substrate proxy: three families, not one

An earlier draft of §5e said "R_h as substrate proxy is dominant". Too clean.
There are three, and **only the first exposes a parameter directly comparable to
a CH₄:CO₂ ratio**:

1. **R_h fraction** — CLM5, ELM, JSBACH, LPJ-WHyMe, LPJ-GUESS, CLASSIC,
   **TRIPLEX-GHG**, VISIT-A, SDGVM
2. **potential rate × multipliers** — TEM, ORCHIDEE (`a₀·C_labile`), VISIT-B,
   WETMETH (`r·C`), JULES (`k·C_eff`)
3. **substrate-pool Michaelis–Menten** — DLEM (DOC), **IBIS-CH4** (DOC →
   acetate/H₂), ELM-SPRUCE

Families 2 and 3 carry dimensional rate constants that are **not**
interchangeable with a ratio. Of ~15 models, only about half can be compared on
this axis at all — the third distinct comparability trap found today.

**We are in family 1**, with `Rh` from `tco2mic`, alongside TRIPLEX-GHG. Good:
our closest precedent is structurally identical on the axis that matters.

Canonical taxonomy citation for all of this: **Xu et al. 2016, BG 13, 3735**,
"Four decades of modeling methane cycling in terrestrial ecosystems", Table 3.

### One more note on VISIT

VISIT's `FP = 0.5` is a **gross** production fraction with oxidation applied
separately at **60–90 %** (Scheme A, `O = P·(0.60 + 0.30·GPP/GPP_max)`), so its
*net* factor is ~0.05–0.20 — in line with everyone else. The apparent 0.5
outlier in §5e's table dissolves once oxidation placement is accounted for,
which is the same lesson for the third time.

### Provenance of this section — what is verified and what is not

This project has been burned repeatedly by claims that turned out to be search
summaries rather than sources, so the split is recorded explicitly.

**First-hand, read from the paper text or a publisher API:**

* **Everything the decision rests on** — Chadburn et al. 2020 (r 0.94–0.98,
  R² 0.89–0.97, τ = 6.5, the per-site RMSE table, the "does not add predictive
  power" quote), Melton et al. 2013 (ρ = 0.898, the equifinality passage), Riley
  et al. 2011 (factor 4/10 regional/grid-cell, the tuning caveat), Zhang et al.
  2023 wavelet, the Zhu et al. 2025 ERL quotes.
* All the scheme equations and parameter values in §5e — Gedney/JULES, WETMETH,
  WetCHARTs, SatWetCH4, Bloom 2012.
* All UpCH4 numbers — six predictors, 26 clusters, NSE 0.54/0.53, tropical
  R² 0.08 / NSE −0.12, IAV NSE −5.11, 146 ± 42.7 → 71.8 ± 22.6 on GIEMS-2, the
  Sahel 3×, and the "yet to use networked data" quote.
* 13 identifiers spot-checked against Crossref/arXiv — all resolve, including
  the 2026 ones.

**Second-hand — real papers, numbers not personally read from the PDF:**

* **Essentially all the quantitative claims in the novelty argument above**:
  Tsai et al. 9.9 vs 253 min, Feng et al. 0.732 vs 0.748, Aboelyazeed et al.
  0.763/0.579, Chinta et al.'s 0.72 ms and five-parameter shortlist, Kallingal
  et al.'s 100 000 iterations / 480 CPU-hours, Ahmad et al.'s RMSE,
  clm-ml-jax's convergence, Raoult et al.'s adjoint-version claims.
* X-MethaneWet's 0.92 / 0.55 / 0.09 split; Yuan et al. 2022; Irvin et al. 2021;
  Peltola et al. 2019's NSE 0.25 and Q₁₀ 1.90; Chang et al. 2021's 77 %.

**Consequence.** The M-lite⁺ decision is safe — it rests entirely on first-hand
material. **The novelty and speed claims must be verified against the primary
PDFs before they appear in any manuscript**, exactly as M0 did for WetCHARTs and
TRIPLEX-GHG.

### Two risks to design around

1. **Upstream fields are load-bearing.** Zhang et al. (2023) wavelet benchmark
   finds the **production substrate proxy** is what discriminates models, not the
   transport machinery. M-lite⁺ inherits IBIS's soil temperature, soil carbon and
   hydrology wholesale — so M3 failures may be IBIS failures, not CH₄ failures,
   and must be diagnosed as such.
2. **FLUXNET-CH₄ has a tropical hole**: 6 sites between 20°S and 20°N, ~9 % of
   training data south of 23°N, against ~70 % of global flux being tropical.
   UpCH4 assigns the Amazon and Congo — zero towers — to the constituencies of
   two SE Asian peat-swamp towers, and is ~3× high in the Sahel with the
   semi-arid→humid gradient inverted. **Gradient efficiency does not create
   information where there are no towers.** The dPL regionalisation answer (a
   network mapping cell attributes → parameters, rather than per-site fitting) is
   the structural mitigation, and is what per-site JULES-HIMMELI calibration
   failed to do.

## 5g. Consolidated priors — the table M2 and M7 should actually use

Every value below converted to a **common basis: gross molar CH₄:CO₂, oxidation
excluded**. Half the published numbers needed transforming to get here.

| model | as published | basis | **as gross molar CH₄:CO₂** |
|---|---|---|---|
| **TRIPLEX-GHG** (our lineage) | 0.23 | gross molar | **0.23** — boreal 0.18, temperate 0.32, tropical 0.15 |
| CLM5 / ELM | 0.20 [0.1–0.3] | gross molar | 0.20 — *but effective ≈0.09 at 10 °C, see below* |
| LPJ-GUESS (14-site MCMC) | 0.14 ± 0.007 | gross molar | **0.14** — tightest published posterior |
| ORCHIDEE-PCH4 (multi-site) | q_MG 9.64 | gross molar (inverse) | 0.104 |
| LPJ-WHyMe | 0.10 | gross molar | 0.10 |
| CLASSIC | δ_s 0.135 × α_w 0.45 | **net** | ≈0.20 gross at 70 % oxidation |
| VISIT scheme A | FP 0.50 | gross, 60–90 % oxidised | 0.50 → net 0.05–0.20 |
| JSBACH / de Vrese / ICON-XPP | 0.35 / 0.40 / 0.24 | **carbon-atom fraction** | **÷ ~0.65** ⇒ 0.54 / 0.62 / 0.37 |

**Consensus gross ratio: 0.10–0.23.** Our 0.23 is the top of it, which is
defensible only because TRIPLEX-GHG is the closest precedent — same host model,
same substrate family. Flagged for M7 to move.

### Pre-fit checklist — three traps, each of which bit once today

1. **Gross vs net of oxidation.** CLM 0.20 gross, CLASSIC 0.135 net, VISIT 0.50
   gross-with-60–90 %-oxidation are three different quantities.
2. **Molar ratio vs carbon-atom fraction.** JSBACH's `f_CH4` is the latter;
   divide by ~0.65.
3. **Substrate family.** Only R_h-fraction models expose a comparable ratio at
   all; potential-rate and substrate-pool-MM models carry dimensional constants
   that are not interchangeable.

### Two implementation warnings, if we ever port from CLM

* **CLM's effective `f_ch4` is not 0.20.** The code applies
  `f_ch4 · q10ch4^((T−22)/10)` then divides out the moisture/N limitation — at
  10 °C the effective ratio is **≈0.09**.
* **CLM's Michaelis–Menten constants are hardcoded in current CTSM master.**
  The netCDF reads are commented out; editing the params file silently does
  nothing.

### Which IBIS precedent we are following

**TRIPLEX-GHG and IBIS-CH4 are two different schemes on the same host DGVM**, and
we follow the former:

* **TRIPLEX-GHG** (Zhu et al. 2014) — R_h-fraction, 13 params. **Our lineage.**
* **IBIS-CH4** (Song et al. 2020; Xiao et al. 2024) — substrate-pool
  Michaelis–Menten with prognostic microbial biomass, DOC → acetate/H₂,
  double-MM oxidation, 30 params. Its closest thing to a ratio is a
  stoichiometric yield (0.5 mol CH₄ per mol acetate), **not** an R_h fraction —
  so its parameters are *not* transferable to M-lite⁺.

Xiao et al.'s global total of **152.7 [135.7–167.6] Tg yr⁻¹ on prescribed
WAD2M** remains the single most comparable benchmark, since it is the same host
model on the same extent product.

### Remaining holes, both paywall-driven

* **Walter & Heimann (2000)** constants are second-hand — via Part 2 (NASA NTRS)
  and restatements in TRIPLEX-GHG and PEATLAND-VU. Mutually consistent, not
  primary.
* **VISIT's biome-specific P₀** is not published in any accessible source.
* **Xiao et al.'s site-level R²/RMSE** need manual retrieval through the UW
  library (Wiley blocks automated access) before being cited.

## 5h. Challenges to the §5f decision — recorded, not buried

The last research thread returned material that pushes against parts of §5f.
Written down so the decision is held with the right confidence.

### 1. Chinta's seasonal result, which I missed on first reading

§5g reported that Q₁₀ and `f_CH4` are the top sensitivity parameters. True of
the **20-year annual mean**. The seasonal breakdown says something else:

> Q₁₀ and f_CH4 dominate only in **cold months** (Dec–Mar at CH-Cha, Nov–Jun at
> SE-Deg); in the remaining months sensitivity shifts to **`f_D0` (diffusion
> multiplier) and `K_O2`** — neither of which is dominant in the annual mean.
> **Annual averaging hides them.**

`f_D0` is diffusion, which §5f removed from the plan. Two readings:

* **Against the decision**: dropping transport may cost more in the warm season
  than Chadburn's seasonal-mean correlation reveals.
* **For it, and I think this is right**: M-lite⁺ has no resolved diffusion, so
  **`τ_oxid` plays exactly the role `f_D0`/`K_O2` play in ELM** — it is the
  single knob controlling how much produced CH₄ escapes. That it dominates in
  warm months is *consistent* with WETMETH's finding that `τ_oxid` is its most
  sensitive parameter by a wide margin (±10 % → ±37 % global).

**Actionable consequence**: evaluate M3 **by season, not on annual means**, and
expect `τ_oxid` to carry the warm-season skill. If a single constant `τ_oxid`
underperforms in summer, that — not ebullition — is the first thing to relax.

Also: Chinta's method was **Gaussian process regression, not gradient boosting**,
on **14 sites × 190 LHS simulations**, not 20 sites × 1024 members. §5g's
description came from a secondary source and was wrong on both.

### 2. The fixed CH₄:CO₂ ratio is under sustained, explicit attack

Three independent lines, all 2025–26:

* **Moser et al. (2026, BG 23, 605)** — observed ratios span orders of
  magnitude: Knoblauch et al. CO₂:CH₄ = 0.92 ± 0.18 in permafrost incubations
  (after a multi-year lag while methanogen communities establish); Heslop et al.
  13–134 by depth; Galera et al. in-situ medians of **12 and 373** by tundra
  type. Minimum defensible representation: fermentation + acetoclastic +
  hydrogenotrophic methanogenesis.
* **Yvon-Durocher et al. (2014)** — methanogenesis activation energy **0.98 eV**
  vs general respiration **0.65 eV**. Any model applying one Q₁₀ to R_h and
  taking CH₄ as a fixed fraction has this wrong by construction.
* **SatWetCH4 (2025)** deliberately uses **no ratio at all**, and the ES&T 2025
  synthesis finds a **median 24.8-day lag** between GPP and CH₄ — a lag an
  instantaneous `R_h × f_CH4` form cannot produce.

**This does not overturn M-lite⁺** — a fixed ratio remains what most operational
models use, and our M4 tier (microbial dormancy) is the cheap partial answer.
But it is the **strongest stated limitation of the design** and belongs in any
manuscript's discussion, not just its appendix.

### 3. A globally uniform Q₁₀ is not defensible

**Basso et al. (2026, BG 23, 2815)** — JSBACH + Jena CarboScope inversion over
the Arctic. Best pan-Arctic Q₁₀ = 1.8, **but Western Siberia is best at 1.4 and
Europe / northern Canada at 2.2**. Their conclusion is explicit: regionally
specific parameters are required.

This corroborates TRIPLEX-GHG's biome split (§5g: boreal 2.03 / temperate 2.66 /
tropical 3.75) from a completely different model and method. **Two independent
lines now say a single global Q₁₀ is wrong** — which strengthens the case for
dPL-style regionalisation at M7 rather than a single global fit.

### 4. Extent may matter less than §5d implies

Measured sensitivity of the global total to the extent *product* is roughly
**5 Tg** (GMB diagnostic ↔ prognostic) against an **~80 Tg inter-model range**.
And GMB 2025 **switched its headline from the diagnostic (prescribed WAD2M) to
the prognostic ensemble**, precisely because of the post-2015 tropical
discontinuity.

That is in tension with §5d's framing and with WetCHARTs' ">80 % is extent +
carbon". The reconciliation: **within one model**, swapping extent products
moves the total less than **across models**, structural differences do. Both
statements are true; they answer different questions. WetCHARTs varied extent
*within a fixed structure*, which is our situation.

**Our Phase-1 prescribed-extent decision still holds** — Liu et al. 2026 found
the diagnostic run beat the prognostic for 7 of 11 models — but the justification
is "prognostic hydrology is not better and costs a module", not "extent is the
dominant uncertainty".

### 5. A genuinely open niche, if we want one

> Nobody has run **one** model across WAD2M v2 / GIEMS-MC_ISW / GIEMS-MC_ISW+P /
> GLWD v2-scaled over the same period and reported ΔTg.

**GIEMS-MC** (Bernard et al. 2025, ESSD 17, 2985) is new and well suited: monthly
**1992–2020** (29 years vs WAD2M's 21), and it ships **two** variables —
inundated+saturated (MAmax 3.80 Mkm²) and +peatland (7.75 Mkm²). Its spatial
correlation with WAD2M is only **0.61**, so the two disagree regionally while
agreeing globally — Africa differs by a factor of 2.4.

A differentiable model makes that experiment cheap: refit `r_CH4`, `Q10p`,
`τ_oxid` under each extent product and report both ΔTg *and* Δparameters. **The
parameter shift is the more interesting number**, and it is exactly what
gradient calibration makes affordable.

### The recurring theme in every thread

**The tropics.** FLUXNET-CH₄ has 6–7 sub/tropical sites; UpCH4's tropical NSE is
−0.12; SWAMPS is discontinuous there since 2015; GIEMS-MC misses the Congo by an
order of magnitude; SatWetCH4's RMSD is 5× worse there than boreal. Every
methodological family fails in the same place, and ~70 % of the flux is there.

**Do not report a global total without a tropical caveat.** It is the one thing
every paper in this literature has in common.

## 5i. M2 first global run — two bugs found, both now fixed

Built M-lite⁺ and ran it on the validated fire spin-up state. Working end to
end. The first run found two real defects, which is what a trial is for.

### Bug 1: the water table was read from the wrong array

M1's `water_table_depth` integrates the unsaturated deficit of the whole column
from IBIS's **grid-cell-mean** moisture. But a cell that is 15 % wetland is 85 %
upland, and the mean describes the upland. §4.1 specified "water table depth
**within the inundated fraction**"; M1 implemented it against the cell mean.

Measured on the spun-up state:

```
wetland cells      zwt      top-layer saturation    fraction escaping
30S-30N           0.85 m          0.655                  0.205
30-60N            2.57 m          0.123                  0.0006
60-90N            3.60 m          0.001                  0.00000
```

A 3.6 m water table annihilates the flux through `exp(−zwt/τ)`. **Result: 99.8 %
of global emission in the tropics and exactly zero north of 60°.**

Two fixes, in `wetland_water_table`: bound the answer to a wetland-plausible
range (0 to `z_wt_max` = 0.30 m), and **count ice as water** — a frozen bog is
still saturated, its water is simply solid. Arctic wetland cells read liquid
saturation 0.001 while being frozen ground; total pore fill (IBIS's own `zwsoi`)
reads them correctly. Production is gated below freezing, so this cannot
manufacture winter emissions.

### Bug 2: temperature was counted twice

`Rh` **already carries a temperature dependence** — that is why Arctic wetland
Rh is 0.086 gC m⁻² d⁻¹ against the tropics' 1.923, a 22× drop. Multiplying by
`Q10 = 2.48` applied temperature sensitivity a *second* time, a further 19×.
Product: a **420× tropical:Arctic gradient**, which is the 93/4.7/0.5 split.

The literature says this explicitly and I walked into it anyway. WetCHARTs:
because Rh already carries temperature, their q₁₀ "parameterises only the
temperature dependence of the CH₄:C **ratio**" — hence 1, 2, 3, not 3.7. CLM's
source: `q10ch4` is "additional Q10 … **ABOVE** the soil decomposition
temperature relationship".

Yvon-Durocher et al. (2014) gives the right value directly:

```
Ea 0.98 eV (methanogenesis)  -> Q10 = 3.94
Ea 0.65 eV (respiration)     -> Q10 = 2.48     <- already inside Rh
Ea 0.33 eV (the difference)  -> Q10 = 1.59     <- what belongs on the CH4 term
```

**0.65 eV gives 2.48 — exactly TRIPLEX-GHG's calibrated value.** That confirms
2.48 is the *respiration* Q₁₀ and is already in our substrate. `q10p` is now
**1.60**.

### Where it stands

```
                     before fixes    after     target
global Tg CH4/yr           545.5     458.7     158 +- 24
30S-30N                    93.3%     87.4%     70%
30-60N                      4.7%      8.8%     21%
60-90N                      0.5%      1.5%      8%
seasonal peak                Jul       Jul     Jul-Aug
```

Every axis improved. **Two issues remain, and they are different in kind.**

**Magnitude, 2.9× high — calibration, by design.** Mean surviving fraction is
0.55, giving a net ratio of 0.23 × 0.55 = 0.127 against the 0.06–0.07 that four
independent formulations agree on (§5f). `r_CH4` and `τ_oxid` are the primary
calibration targets precisely because no literature value transfers into this
form. This is M7's job, not a defect.

**Distribution, still too tropical — and it is not ours.** Wetland *area* is
well distributed (55/28/12 % against GCP's 70/21/8 % emission shares), so WAD2M
is fine. The residual traces to **IBIS's own Rh gradient**: 1.923 / 0.444 /
0.086 gC m⁻² d⁻¹, a 22× tropics-to-Arctic drop. Real peatland Rh is roughly
1000–1500 gC m⁻² yr⁻¹ tropical against 100–300 boreal — a 5–10× ratio, not 22×.

**IBIS's Arctic soil respiration looks too low, and that is a carbon-cycle
problem, not a methane one.** Exactly the risk §5f flagged: "M-lite⁺ inherits
IBIS's soil temperature, soil carbon and hydrology, so M3 failures may be IBIS
failures and must be diagnosed as such." First real instance.

Investigate before calibrating — fitting `r_CH4` against a biased substrate
would bake the bias into the parameter.

## 5j. Diagnosing the latitudinal bias — and correcting myself twice

§5i blamed IBIS's Arctic soil respiration. **That was wrong, and wrong for the
same reason as an earlier error in this project: I measured one month and
generalised.**

### Correction 1 — there is no Rh problem

§5i reported a 22× tropics-to-Arctic Rh gradient from the December checkpoint.
Run through a full year, the **annual** gradient is:

```
              30S-30N   30-60N   60-90N   ratio
annual Rh      1.747    1.270    0.365    4.8x
```

**4.8×, inside the observed peatland range of 5–10×.** The seasonal cycle is
right too: Arctic wetland Rh peaks at 0.995 gC m⁻² d⁻¹ in July and falls to
0.027 in January — textbook active-layer behaviour. Soil carbon is also right:
8.94 / 22.37 / 20.26 kgC m⁻², boreal correctly denser than tropical, global
2351 PgC against an observed 1500–2400.

**IBIS's carbon cycle was not the problem.** The claim in §5i is withdrawn.

### A real but negligible bug, found on the way

`loops_gswp3.py` computed the decomposition moisture factor with
`wice_top = 0`, commented "ice fraction approx 0". F77 `stats.f` says the
opposite explicitly: *"wsoi is relative to pore space not occupied by ice and
water, thus **must include** the ice fraction"*, i.e.
`wfps = (1 − wisoi)·wsoi·100`.

Fixed. **A control run shows it accounts for almost nothing** — Arctic Rh 0.383
(without) vs 0.365 (with), 4.6× vs 4.8×. Kept as a fidelity fix, not a
performance one. Recorded because the whole apparent improvement in §5i came
from correcting my sampling, not from this.

### Correction 2 — I reproduced the very bug I had just diagnosed

To fix a temperature gate, I made production scale with `f_thawed`, the thawed
fraction of the anaerobic zone. That **double-counts**: `Rh` already collapses
when soil freezes. Measured contribution: a further **2.6×** tropics-to-Arctic
suppression on top of Rh's own 3.4×. Removed — `thawed_anaerobic_zone` now only
selects *which layers set the temperature*, and does not scale the flux.

This is the same error as the Q10 double-count in §5i, made an hour after
diagnosing it.

### The measured decomposition

Each multiplicative term, by band, averaged over a year of wetland cells:

| band | area % | Rh | Q₁₀ | f_thaw | zwt | surviving |
|---|---|---|---|---|---|---|
| 30S-30N | 55.0 | 2.248 | 1.347 | 1.000 | 0.116 | 0.532 |
| 30-60N | 28.4 | 1.398 | 0.542 | 0.846 | 0.122 | 0.488 |
| 60-90N | 12.1 | 0.652 | 0.327 | 0.390 | 0.101 | 0.551 |
| **ratio** | **4.5×** | **3.4×** | **4.1×** | **2.6×** | — | **~1.0×** |

`surviving_fraction` is flat across bands, so the water table is not
implicated. The suppression is entirely temperature, applied up to three times.

### Where it lands, and the finding that matters

```
                   first run   after fixes   target
global Tg CH4/yr       545.5         473.6   158 +- 24
30S-30N                93.3%         84.7%   70%
30-60N                  4.7%          9.7%   21%
60-90N                  0.5%          3.4%    8%
```

**Area × Rh alone predicts 70 / 26 / 3** against GCP's 70/21/8 — the tropics
land exactly and 30-60N is close. Every temperature term added on top pushes the
extratropics *down*, away from the target.

**A single global `(r_CH4, Q10p)` pair cannot reproduce the latitudinal
distribution.** Calibrating `r_CH4` scales all bands equally, so it can fix the
3× magnitude error and cannot touch the shape.

That is not a defeat — **it independently reproduces a published result.** Basso
et al. (2026) found no single Q₁₀ fits the Arctic (W Siberia 1.4, Europe/N
Canada 2.2), and TRIPLEX-GHG's own calibration splits by biome (boreal
0.18/2.03, temperate 0.32/2.66, tropical 0.15/3.75) with `r` and `Q10` moving in
*opposite* directions. We have now hit the same wall from a third direction.

**It is also the concrete argument for M7's regionalisation**: parameters that
vary with cell attributes, learned by gradient descent, rather than one global
pair. That was previously a methodological preference; it is now a measured
requirement.

### What not to do next

**Do not calibrate `r_CH4` against the global total yet.** It would absorb the
3× magnitude error into a single scalar while leaving the shape wrong, and bake
a compensating bias into the one parameter we most need to be meaningful.

## 5k. Regionalisation test — it works, and imported parameters do not

One run, four parameter configurations evaluated offline against the same
cached monthly fields. Bands used as a crude biome proxy.

| configuration | Tg/yr | 30S-30N | 30-60N | 60-90N | shape err |
|---|---|---|---|---|---|
| **GCP target** | 158 | 70.0 % | 21.0 % | 8.0 % | — |
| global `r`=0.23, `Q10`=1.6 | 473.6 | 84.5 % | 9.7 % | 3.4 % | 30.4 |
| **biome `r` (TRIPLEX), `Q10`=1.6** | **353.1** | **73.9 %** | **18.1 %** | **3.6 %** | **11.2** |
| biome `r` + biome `Q10` (TRIPLEX) | 474.5 | 88.1 % | 7.7 % | 1.7 % | 37.6 |
| biome `r`, `Q10`=1.0 | 376.3 | 54.2 % | 31.7 % | 8.5 % | 27.0 |

### Three findings

**1. Regionalisation works.** Biome-varying `r` alone cuts the shape error by
almost 3× and improves the magnitude at the same time (474 → 353 Tg/yr).
Tropics land at 73.9 % against a target of 70, temperate at 18.1 % against 21.
This confirms §5j's conclusion empirically rather than by inference.

**2. Imported parameters actively hurt.** Adding TRIPLEX-GHG's *own* biome Q₁₀
values (3.75 / 2.66 / 2.03) makes the shape **worse than the global default**.
They were calibrated inside a scheme whose temperature structure differs from
ours, so they double-count — the same trap that produced the `Q10` and
`f_thawed` errors in §5i and §5j, now for the third time in one module.

**The rule this establishes: parameters transfer only with their structure.**
A calibrated value from another model is evidence about *that* model, not a
prior for ours, unless the substrate and the temperature terms match.

**3. The boreal share is entirely Q₁₀-controlled.** At `Q10` = 1.0 the 60-90N
band hits **8.5 % against a target of 8.0 %**, while the tropics overcorrect to
54 %. So the best fit needs `Q10` varying regionally — near 1.6 in the tropics
and near 1.0 in the boreal — which no published table gives us.

### What this means for M7

The regionalisation is no longer a methodological preference or an inference
from Basso et al.; it is a measured requirement with a **baseline to beat:
shape error 11.2, global 353 Tg/yr**. A learned attribute→parameter map has to
improve on that to justify itself.

It also means the primary calibration cannot be a single global scalar. Fitting
`r_CH4` alone would scale all bands equally and leave the shape at 11.2.

### Still open, and not fixable by parameters

Magnitude remains **2.2× high** at the best configuration. The net ratio
implied by biome `r` with mean surviving fraction ~0.52 is 0.15 × 0.52 ≈ 0.08
in the tropics, against the 0.06–0.07 that four independent formulations agree
on (§5f) — so roughly a third of the excess is the ratio and the rest is
elsewhere. Not diagnosed yet.

## 5l. M2 wired — the module is now part of the model

M1 shipped tested and uncalled, which is the failure docs/AUDIT.md is about.
M2 is wired end to end:

* **`MethaneState`** in `IBISState.wetland` — monthly flux, running annual
  total, and the water table depth used (kept because it is the term that broke
  first, §5i).
* **`methane_step`** called at month end in `run_month_gswp3`, and threaded
  through `run_year_gswp3` → `run_spinup_gswp3`. Monthly, because that is
  WAD2M's resolution — a daily flux would imply a precision the forcing lacks.
* **`enable_methane`** read by `scripts/run_global.py`, which loads the WAD2M
  climatology when set. `forcing.wad2m_file` gives the path.

Verified through the runner rather than in isolation: `state.wetland` goes from
`None` to a populated state, the flux is finite and non-negative, and July
(5.55 mg CH₄ m⁻² d⁻¹ mean) exceeds January (3.08) as it should.

**A dead flag caught in the act.** The first version keyed the wiring off
`f_inund` being present and left `enable_methane` read by nothing — the exact
pattern this project keeps repeating, created an hour after finishing the audit
that removed twelve instances of it. Wiring it into `run_global` fixed it, and
`test_enable_methane_flag_is_actually_read` now pins it.

**Still diagnostic, deliberately.** The carbon is not removed from the soil
pool. At ~7 % of Rh it is a small term, and coupling it before either the
methane or the carbon parameters are calibrated would bake one bias into the
other. Revisit after M3.

### M3 is blocked on data we do not have

FLUXNET-CH₄ is **not on the NAS**. `/shared/nas2/Min/FLUXNET2015` holds 810
sites of the CO₂/energy/water product, which carries **no CH₄ columns** —
FLUXNET-CH₄ (Delwiche et al. 2021, ESSD 13, 3607) is a separate community
product. It is free under CC-BY-4.0 but needs a fluxnet.org registration.

Until it is downloaded, the only available test is the global total against
GMB — and §5c already established that is a **weak** test, since the published
range 119–203 Tg yr⁻¹ spans a factor of 1.7.

**FLUXNET2015 is still worth using**, for a different purpose: validating
IBIS's Rh and GPP directly. §5j showed the substrate is what drives the
latitudinal shape, and 810 sites of CO₂ flux is exactly the data to check it
against. That is a carbon-cycle evaluation, not a methane one, and it does not
need the CH₄ product.

## 5m. Two structural fixes to the magnitude — and where I stopped

### The term M-lite⁺ was missing

`r_CH4` is a CH₄:CO₂ ratio for decomposition **below the water table** — CLM
defines its `f_ch4` in exactly those words. IBIS's `soilbgc` is a bulk model, so
`tco2mic` is whole-column Rh, and M-lite⁺ applied the anaerobic ratio to *all*
of it. That counts the aerobic soil above the water table as methanogenic.

`anaerobic_substrate_fraction` = `exp(−zwt / z_tau)` now weights the substrate.
`z_tau` = 0.50 m is CLM's default with a published range [0.1, 0.8] — one of the
five parameters carrying >90 % of ELM's CH₄ variance (Chinta et al. 2024).

Not double-counting the moisture limitation already inside Rh: that sets how
*fast* carbon decomposes, this sets what share does so *without oxygen*.

### The oxidation default was outside the observed range

`tau_oxid` = 0.15 m gave only **47 % oxidised** at a typical water table.
Chadburn et al. (2020) fitted a value giving **~70 %**, and observations give
40–70 % (Bridgham et al. 2013) and 60–90 % (Popp et al. 2000). **0.10 m gives
70 %** and is inside every one of those.

### Effect

| | Tg CH₄/yr | shape err |
|---|---|---|
| before both fixes | 353.1 | 11.2 |
| + anaerobic substrate weighting | 309.5 | 13.1 |
| + `tau_oxid` 0.15 → 0.10 | **255.8** | 16.2 |
| GMB 2025 target | **158 [119–203]** | — |

**Still 1.6× high, and the shape got slightly worse.** Both fixes are justified
independently of the target — the first is structurally required, the second
moves a default into the observed range — so they stand regardless.

### Why I stopped here

The next lever would be fitting `r_CH4` or `tau_oxid` to hit 158. **§5k
established that a global scalar cannot fix the shape**, so doing that would
absorb a structural error into a parameter that then means nothing physically.
It is also how the three double-counting bugs in §5i–5k got in: adjusting a
term without first establishing what it should be.

A useful check on where the remaining excess is *not*: with the anaerobic term
in, the tropical net ratio is 0.15 × 0.79 × 0.30 ≈ **0.035**, now *below* the
0.06–0.07 that four formulations agree on. So the ratio chain is no longer the
excess — meaning the residual lives in Rh × area × Q₁₀, and per §5j the Rh
gradient and the WAD2M area are both independently validated.

**That points at the Q₁₀ term with biome-varying `r`.** TRIPLEX-GHG's biome `r`
values were each calibrated *at that biome's own temperature*, so multiplying
them by `Q10^((T−T_ref)/10)` applies the biome temperature difference a second
time — the same trap as §5i and §5k, in a fourth guise. Testing that properly
needs a target the global total cannot provide.

**Which is the real blocker: M3 needs FLUXNET-CH₄.** Site-level fluxes
discriminate between these hypotheses; a global total whose published range
spans a factor of 1.7 does not.

## 5b. Build log

### 2026-08-25 — M1 done

`ibis/io/wad2m.py` and `ibis/bgc/methane.py`, 18 tests.

**Grid alignment, checked not assumed.** WAD2M labels cell *centres*
(−89.75 … 89.75); the IBIS land mask labels the lower-left corner
(−90.0 … 89.5). Both are 360×720 south-to-north, so cell *i* is cell *i* — the
0.25° difference is a labelling convention, not an offset. `_check_grid`
asserts the uniform half-cell offset rather than trusting index order.

Verified against geography, by regional mean Fw for July 2010:

| region | mean Fw |
|---|---|
| West Siberian Lowland | 0.229 |
| Hudson Bay Lowlands | 0.218 |
| Congo Cuvette | 0.150 |
| Amazon basin | 0.130 |
| US Great Plains | 0.0023 |
| Sahara | 0.0005 |

Wetlands are ~100× the dry regions and land where they should. The first
version of that test used single cells and **failed** — 57.0 °N sits exactly
between WAD2M's 56.75 and 57.25 centres, so a knife-edge point says nothing
about orientation. Regional means, not points.

**Ice is not water.** `wsoi` is the liquid share of the *ice-free* pore space,
so saturation is `wsoi · (1 − wisoi)`. Treating a frozen column as saturated
would put the water table at the surface across the whole winter Arctic and
make it the model's largest methane source. A test pins this.

**Pre-satellite years are explicit.** WAD2M starts in 2000; a spin-up cycling
1901–2000 has no record for 99 of those years. `read_wad2m_month` raises unless
`clamp_to_range=True` is passed, and `read_wad2m_climatology` gives a mean
seasonal cycle for spin-up. Silently returning zeros would delete every wetland
before 2000.

**~~Open question for M2.~~ CLOSED 2026-08-30 — definitional, not an error.**
Global wetland area through this reader is 3.06 Mkm² in January 2010 and
4.64 Mkm² in July, against the larger figures in Zhang et al. (2021). Settled
by `verify/methane/wad2m_area_audit.py`; see §5o.

## 5n. The Q10 double-count hypothesis: right about the boreal, not a fix

§5m ended by naming the leading suspect for the remaining excess -- `t_ref` is a
single global 295.15 K (22 C, a tropical value) while `r_ch4` varies by biome.
TRIPLEX-GHG's biome ratios were each calibrated AT that biome's own
temperature, so referencing them all to a tropical `T_ref` applies the biome
temperature difference a second time. That is the same trap as 5i, 5j and 5k,
in a fourth guise.

§5m also said testing it needed FLUXNET-CH4. **That was wrong -- it is testable
against the latitudinal shape**, because the double-count is a *gradient* error
and the shape is what a gradient error distorts.

### The prediction, made before running

The suppression is `1.60^((275 - 295.15)/10) = 0.388`, i.e. boreal production
is cut **2.58x too much**. Boreal share was 3.7% against a target of 8.0%, so
it needed roughly **2.2x more**. Those two numbers were derived independently
and agree.

### The test

`verify/methane/evaluate_global.py --config all` gained `biome_tref`, which
sets each biome's `T_ref` to that biome's own production-weighted anaerobic
temperature, taken from the model's own fields rather than fitted. Measured:
30S-30N 300.2 K, 30-60N 280.4 K, 60-90N 275.6 K.

| configuration | Tg/yr | 30S-30N | 30-60N | 60-90N | shape err |
|---|---|---|---|---|---|
| GMB 2025 target | 158 | 70.0% | 21.0% | 8.0% | -- |
| `biome` (current default) | 256.3 | 76.7% | 16.4% | 3.7% | **15.5** |
| `biome_tref` | 279.2 | 55.5% | 30.1% | 8.5% | 24.2 |
| `biome_noq10` | 269.6 | 57.1% | 29.7% | 8.9% | 22.5 |

Deviations from target:

| configuration | 30S-30N | 30-60N | 60-90N |
|---|---|---|---|
| `biome` | +6.7 | -4.6 | **-4.3** |
| `biome_tref` | **-14.5** | +9.1 | **+0.5** |

### What this establishes

**The hypothesis is right about the thing it predicted.** The boreal deficit
was the specific claim, and removing the double-count closes it almost exactly:
3.7% -> 8.5% against a target of 8.0%, a deviation of -4.3 -> +0.5. A
prediction made from the Q10 arithmetic alone landed within half a point.

**It is not a fix.** Aggregate shape error gets *worse*, 15.5 -> 24.2, because
the tropics then fall from +6.7 to -14.5. Removing the boreal suppression also
removes the tropical amplification, and the tropics were being held up by it.
So a second error opposes the first, and the current default is partly right
for the wrong reason -- a Q10 double-count compensating for something that
inflates the tropics independently.

**`biome_tref` is NOT adopted as the default.** It trades a known error for a
larger one. It is kept as a diagnostic configuration because it isolates the
boreal term cleanly.

### Where this leaves M3

The target sits between the two configurations in every band, which invites
fitting a partial correction. §5k already established why not: a parameter that
absorbs two opposing structural errors means nothing physically, and that is
exactly how the bugs in 5i-5k got in.

What this narrows: the remaining excess is **not** uniformly distributed. It is
concentrated in the tropics, and it is large enough to have been masking the
boreal double-count. Candidates, none testable against a global total:
tropical wetland extent in WAD2M, tropical Rh, and the tropical `r_ch4` of 0.15.

Site-level fluxes still discriminate; a global total whose published range
spans a factor of 1.7 does not. **M3 remains the blocker** -- but the question
it now has to answer is sharper than in 5m: not "where is the excess" but "what
inflates the tropics by enough to hide a 2.6x boreal suppression".

---

## 5o. Wetland area is not the problem — the M2 open question, closed

§5l left this blocking M3: our area is 3.06 Mkm² in January 2010 and 4.64 in
July, Zhang et al. (2021) quote larger figures, and *"whether that reflects a
different definition or an error here is not yet established. Total flux scales
linearly with this, so it must be settled before any M3 evaluation is
believed."* §5n then narrowed the excess to the tropics, which made area the
obvious first suspect.

**It is definitional, and the distinction is easy to miss.** WAD2M's headline
`MAmin` / `MAmax` / `MALt` are **per-grid-cell** statistics summed globally,
not the extremes of the global sum. Different cells peak in different months,
so sum-of-per-cell-maxima is far larger than max-of-global-sum. Reduced the
paper's way, our file agrees:

| statistic | ours (Ver 2.0, 0.5°) | Zhang et al. 2021 (Ver 1.0, 0.25°) | ratio |
|---|---|---|---|
| MAmin | 1.71 | 3.5 | 0.49 |
| **MAmax** | **7.09** | **7.5** | **0.95** |
| **MALt** | **12.04** | **13.0** | **0.93** |

MAmax and MALt agree to 5–7%, which the version difference and the 0.25°→0.5°
bilinear regrid comfortably explain. MAmin is the outlier and is the least
robust of the three: a per-cell *minimum* is destroyed by smoothing, since any
month a coarsened cell reads dry drives that cell's minimum to near zero.

Two further checks:

* the `Fw` variable's own metadata reads **"Fraction inundated"**, confirming
  it is inundated area, not inundated-plus-peatland;
* computing the January/July 2010 totals on the **full WAD2M grid** gives 3.05
  and 4.63 Mkm² against the reader's 3.06 and 4.64, so the reader loses
  essentially nothing to the IBIS land mask.

### Consequence for 5n

Mean monthly inundated area splits **55.6% tropics / 28.3% (30–60N) / 12.0%
(60–90N)**, while our methane flux is **76.7%** tropical. The tropics are
emitting far out of proportion to their wetland area, and the area itself is
faithful to the product. **So the tropical excess in §5n is not extent** — it
lives in `Rh × Q10 × r_ch4`, and the remaining candidates are tropical Rh and
the tropical `r_ch4` of 0.15.

That does not make the inundation product *correct* — §D7 open question 2 stands,
since WAD2M, GIEMS-2 and SWAMPS disagree substantially and that remains the
largest single lever on the global total. What is now established is narrower
and sufficient for M3: **we are reproducing WAD2M faithfully, so our error is
not a reader bug.**

---

## 5p. Rh is ~1.5x too high — and methane scales linearly with it

§5o eliminated wetland extent, leaving `Rh` and `r_ch4`. Measured from the
`spinup_bedrock_v9_3lay` state over one simulated year
(`verify/methane/rh_audit.py`):

| | IBIS | published |
|---|---|---|
| global Rh | **80.2 PgC/yr** | **49–58 PgC/yr** |

Published range spans Potter & Klooster (1998), Bond-Lamberty & Thomson (2010),
Hashimoto et al. (2015) at 51–57; Yao et al. (2020) at 48.8 ± 0.9; and the
partitioning of a 91–102 PgC/yr total soil respiration at an Rh:Rs ratio of
0.56–0.63. We are **1.4–1.6x high**.

**Methane scales linearly with Rh in M-lite⁺, and our methane is 1.6x high**
(256.3 against a GMB target of 158). The two ratios coincide closely enough
that Rh is now the leading explanation of the magnitude error, ahead of
`r_ch4`.

### What was checked before believing it

* **Sampling.** `evaluate_global.py` samples `tco2mic` at each month's end and
  scales by days in month -- a snapshot method, and every methane number in
  this document rests on it. `summary.ayco2mic` accumulates Rh *daily*, so it
  is the true annual integral. Both give 80.2 PgC/yr. The snapshot method is
  sound here. (`ayco2mic` had to be differenced against its starting value --
  see the reset bug below.)
* **Equilibrium.** Soil carbon drifts −0.07 %/yr over the final spin-up year,
  so the state is converged and Rh ≈ carbon input rather than a transient.
* **Units.** Per-area values are 0.644 kgC/m²/yr in the tropics and 0.469 in
  30–60N -- the right order, not a factor-of-1000 slip.

### What could NOT be checked, and why it matters

**Whether this is a decomposition problem or a productivity one.** At
equilibrium Rh ≈ carbon input, so an Rh of 80 implies an NPP of 80 -- also well
above the published 55–60. If NPP is equally high, the cause is upstream in
productivity and `soilbgc` is faithfully respiring what it is given; if NPP is
~57 and Rh is 80, decomposition is genuinely too fast.

That check is unavailable: **`tgpp`, `tnpp`, `tgpptot`, `tnpptot` and
`tco2root` are dead** -- exactly zero in the checkpoint and never assigned
anywhere in production code (docs/DEFECTS_FOR_REVIEW.md). The model has no
usable GPP or NPP diagnostic. **Wiring one is the next step for M3**, and it is
worth more than any further methane parameter work.

### The tropical chain

| stage | tropical share |
|---|---|
| inundated area (§5o) | 55.8% |
| wetland-weighted Rh | 66.3% |
| CH₄ flux (§5n) | 76.7% |
| GMB target | 70.0% |

Each stage amplifies the tropics. Wetland Rh is 0.824 kgC/m²/yr in the tropics
against 0.514 (30–60N) and 0.491 (60–90N). So the tropical excess is *partly*
Rh and partly the Q₁₀/`r_ch4` step, consistent with §5n finding that removing
the Q₁₀ double-count alone overshoots.

### Bug found while doing this

Seven annual accumulators were never reset and grew across the whole run
(`ayco2mic` reached 921.9 kgC/m², `ayprcp` 3.2×10⁶ mm), while being written to
netCDF labelled `/yr`. No physics reads them, so simulations are unaffected and
only reported output was wrong. Fixed, with a guard test; see
docs/DEFECTS_FOR_REVIEW.md.

---

## 5q. Re-validated on the corrected carbon cycle -- and 5n is vindicated

The GPP investigation (docs/GPP.md) ended with two empirical corrections --
`vmax` x0.54 and soil-OM loss rates x0.67 -- that brought Rh from 89.3 to
**57.9 PgC/yr**, inside the published 49-58. Since CH4 scales linearly with Rh,
the methane evaluation had to be repeated on that state.

**Like-for-like on the same grid** (both from `run_global` restarts, 95 333
points, N->S ordering -- the 256 Tg figure in 5m came from the OLD duplicated
88 484-cell v9 checkpoint and is not comparable):

| configuration | v10 baseline | **both-fix** | GMB 2025 |
|---|---|---|---|
| `biome` | 458.0 Tg, shape 37.2 | 328.3, shape 48.2 | 158 [119-203] |
| **`biome_tref`** | 290.3, shape 19.3 | **182.0, shape 11.0** | **in range** |

Latitudinal split of `biome_tref` on the corrected state: **72.7 / 19.8 / 1.0 %**
against a target of 70 / 21 / 8.

### Three findings

**1. Magnitude is essentially solved, without touching a methane parameter.**
182 Tg sits inside the GMB range, and the improvement came entirely from fixing
the carbon cycle. `r_ch4` was never fitted -- 5m explicitly refused to -- and
that restraint is what let the correction propagate cleanly instead of being
double-counted.

**2. The tropical and mid-latitude shares are now nearly right** (72.7 vs 70,
19.8 vs 21). The residual is the high latitudes at 1.0 % against 8.0 %.

**3. 5n was right and its test was confounded.** 5n introduced `biome_tref` --
per-biome reference temperatures, removing the Q10 double-count -- and rejected
it because aggregate shape error rose 15.5 -> 24.2. **That test ran on a model
with a 1.6x carbon-cycle bias and a duplicated grid.** On the corrected state
`biome_tref` is decisively the better configuration: 182 vs 328 Tg and shape
error 11.0 vs 48.2. The physical reasoning in 5n was sound; the evaluation it
was judged against was not.

That is a general caution for this project: **a parameter change tested against
a biased baseline can be rejected for the baseline's faults.** Anything
evaluated before the carbon-cycle correction deserves re-testing before it is
treated as settled.

### All five configurations, re-ranked on the corrected state

5k ranked four parameter configurations against a model carrying a 1.6x carbon
bias on a duplicated grid. Re-run on `both_fix`:

| configuration | Tg/yr | 30S-30N | 30-60N | 60-90N | peak | shape err |
|---|---|---|---|---|---|---|
| GMB 2025 target | 158 | 70.0% | 21.0% | 8.0% | 7-8 | -- |
| `global` | 271.1 | 92.5% | 5.4% | 0.4% | 3 | 45.7 |
| `biome` | 328.3 | 93.6% | 3.9% | 0.4% | 3 | 48.2 |
| `biome_q10` | 412.8 | 97.0% | 1.6% | 0.1% | 3 | 54.3 |
| `biome_noq10` | 283.1 | 87.5% | 8.1% | 1.2% | 3 | 37.3 |
| **`biome_tref`** | **182.0** | **72.7%** | **19.8%** | 1.0% | **6** | **11.0** |

`biome_tref` wins on magnitude and shape simultaneously, by a wide margin --
shape error 11.0 against 37.3 for the next best. **On the biased model `biome`
looked best; here it is second-worst.** The ranking inverted, which is the
clearest demonstration in this project of why anything evaluated before the
carbon-cycle correction had to be re-tested.

**Seasonality also separates them.** Every configuration except `biome_tref`
peaks in **March**; `biome_tref` peaks in **June** against a target of July-
August. The March peak is the Q10 double-count driving southern-hemisphere wet
season emissions: with a single tropical `T_ref`, the Q10 term amplifies warm
tropical wetlands so strongly that they dominate the annual cycle. Removing the
double-count moves the peak five months and leaves a residual one to two months
early -- a real, separate discrepancy, and now the sharpest open question in
this module after the high-latitude deficit.

### Seasonal phase: the deficit is NOT the cause

The obvious hypothesis was that the early peak follows from the high-latitude
deficit -- boreal wetlands peak in late summer, so if they contribute 1%
instead of 8% the global peak is pulled toward the tropical cycle. Tested with
`verify/methane/seasonal_phase.py`, and **refuted**.

Monthly CH4, Tg/yr-equivalent per month, `biome_tref` on `both_fix`:

| band | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | peak | share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30S-30N | 11.1 | 10.4 | 12.7 | 11.6 | 12.4 | 11.7 | 12.2 | 11.0 | 9.8 | 9.7 | 9.7 | 10.0 | Mar | 77.8% |
| 30-60N | 2.1 | 2.0 | 2.8 | 3.6 | 3.9 | **4.9** | 4.4 | 3.4 | 2.8 | 2.6 | 1.6 | 1.9 | **Jun** | 21.1% |
| 60-90N | 0.0 | 0.0 | 0.0 | 0.1 | 0.3 | **0.4** | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 | 0.0 | **Jun** | 1.1% |
| GLOBAL | 13.2 | 12.4 | 15.5 | 15.2 | 16.5 | **17.0** | **17.0** | 14.7 | 12.9 | 12.4 | 11.3 | 11.9 | Jun/Jul | |

Scaling 60-90N from 1.1% to its observed 8% moves the global peak **earlier**,
to June -- because **both northern bands already peak in June**. Amplifying a
band that peaks too early cannot fix a peak that is too early.

**Two corrections to the earlier framing:**

1. **The global phase error is smaller than 5q implied.** June and July are
   tied at 17.0, so the peak is on the edge of the Jul-Aug target rather than
   one to two months early. `evaluate_global.py` reports "peak 6" because
   `argmax` breaks the tie toward the earlier month.
2. **The phase error is in the NORTHERN bands, not the global aggregate.**
   30-60N and 60-90N both peak in June against an observed July-August. That is
   a real, localised discrepancy and it is independent of the high-latitude
   magnitude deficit -- the two open questions do not collapse into one.

A one-to-two month early peak in northern wetlands is the classic signature of
insufficient soil thermal inertia: soil temperature tracking air temperature
too closely, so the anaerobic zone warms before it should. Worth noting that
the 12 m bedrock column (docs/BEDROCK.md) increased that inertia and this run
already includes it, so whatever remains is not the shallow-column error.

### The high-latitude deficit, diagnosed

`verify/methane/highlat_decompose.py` breaks M-lite+ into its factors, weighted
the way the flux weights them. Inundated-area-weighted means by band:

| band | area % | **Rh** | f_anaer | Q10 | unfrozen | f_surv | flux % |
|---|---|---|---|---|---|---|---|
| 30S-30N | 58.2 | **0.0019** | 0.807 | 1.304 | 1.000 | 0.431 | 94.1 |
| 30-60N | 29.6 | 0.0008 | 0.764 | 0.570 | 0.998 | 0.288 | 5.5 |
| **60-90N** | **12.3** | **0.0003** | 0.795 | 0.417 | 0.985 | 0.344 | **0.4** |

(Default `MethaneParams`, so the Q10 column still shows the double-count that
`biome_tref` removes.)

**Wetland extent is not the problem** -- 12.3% of inundated area is north of
60N, matching WAD2M. **The freeze gate is not the problem** -- it reads 0.985,
barely firing. `f_anaer` is uniform. **Rh carries the deficit**: Arctic wetland
respiration is 6x lower per unit area than the tropics.

### Which traces to Arctic soil carbon

| band | model soil C | observed |
|---|---|---|
| **60-90N** | **4.33 kgC/m2** (85 PgC over 19.7 Mkm2) | NCSCD **~1035 PgC** in 0-3 m over ~18 Mkm2, ~55 kgC/m2 |
| 30-60N | 16.37 | -- |
| 30S-30N | 12.55 | -- |

The model holds **~12x too little Arctic soil carbon**, and less there than the
global mean (12.16) -- when the Arctic in fact has the highest soil carbon
density on Earth.

### Two factors, each ~3.5x

`pool = input x residence_time`, and both terms are short:

| band | soil C | Rh (= input at equilibrium) | residence yr | expected |
|---|---|---|---|---|
| **60-90N** | 4.33 | **0.042** | **103** | centuries to millennia |
| 30-60N | 16.37 | 0.315 | 52 | ~30-60 |
| 30S-30N | 12.55 | 0.507 | 25 | ~10-20 |

The temperature limitation IS working -- Arctic residence time is 4x the
tropics, and the mid-latitude and tropical values are about right. But reaching
55 kgC/m2 needs `input x residence = 55`; we have `0.042 x 103 = 4.33`. Both
terms are roughly 3.5x too low.

### Why this is structural, not a parameter

Even with a correct residence time, **a 300-year spin-up cannot build a pool
that took the Holocene to accumulate.** Northern peat carbon formed over
~10 000 years under frozen protection, waterlogging and burial -- mechanisms
IBIS does not represent. Lengthening the decomposition timescale would slow the
approach without changing the destination, because the model's equilibrium is
set by modern input and modern decomposition.

### It is a documented, near-universal ESM failure

Literature search, 2026-09-03. Both halves of the problem are known:

**Soil carbon.** Evaluating soil carbon in CMIP6 Earth system models
(https://bg.copernicus.org/articles/19/4671/2022/) finds the
*"underestimation of northern high-latitude soil carbon stocks"* to be
systematic -- **all ESMs largely underestimate the stocks except CESM2-WACCM**,
and it is a common problem in CMIP5 as well as CMIP6. The named cause is that
most models *"lack critical permafrost-specific features in their soil carbon
modules"*:

| mechanism | present in IBIS? |
|---|---|
| vertically resolved decomposition | **no** -- CENTURY here is a bulk scheme (§D7 Q4) |
| freezing-induced decomposition inhibition | **no** explicit term; only the generic `ftemp` |
| cryoturbation-driven SOC mixing | **no** |
| SOC thermal insulation | **no** -- `FORGANIC` is a constant 0.010, not prognostic |
| deep carbon (~10 m peat, ~20 m Yedoma) | **no** -- 4 m hydrological column |

IBIS lacks all five, which is why our deficit (~12x) exceeds the CMIP6
ensemble's. CESM2-WACCM, the exception, runs CLM5's vertically resolved soil
carbon.

**Methane.** Model comparisons against atmospheric observations over Arctic and
boreal North America (https://acp.copernicus.org/articles/26/1229/2026/) find
Global Carbon Project models average **a factor of 1.5 smaller across Canada
and Alaska**. That is the same GCP ensemble our 158 Tg target comes from -- so
the benchmark itself is low in the north, and our 8x deficit is a severe case
of a field-wide bias rather than a defect unique to this port.

Recent Boreal-Arctic work (https://www.nature.com/articles/s41558-024-01933-3)
also reports GCP models failing to capture the observed +8.9% emission trend.

**What this changes.** Closing our high-latitude gap is not a calibration
exercise; it requires mechanisms neither IBIS nor most ESMs have. The honest
options remain prescribing northern soil carbon, or reporting high-latitude CH4
as a known underestimate -- now with the qualification that the GCP benchmark
shares the bias, so matching it would not mean being right.

### Recommendation

1. **Adopt `biome_tref`** as the default methane configuration -- 182 Tg,
   inside the GMB range, shape error 11.0. **Done 2026-09-04**:
   `ibis.bgc.methane.biome_methane_params(lat)` builds it, and `run_global`
   now passes it.

   This closed a real gap. `run_global` set `enable_methane`, loaded the WAD2M
   climatology, and passed `f_inund` -- but **never passed `methane_params`**,
   so every production run silently used `MethaneParams()`, the "global"
   configuration worth **458 Tg** on the reference state. The flag worked, the
   forcing loaded, and the parameters did not. `tests/
   test_methane_production_config.py` now guards both the constants and the
   wiring.

   The per-biome `t_ref` values (299.6 / 283.0 / 277.9 K) are fixed constants
   taken from the corrected reference state, not recomputed per run --
   deriving them from the state being simulated would make the parameters
   depend on the answer.
2. **Do not tune methane parameters to close the high-latitude gap.** The gap
   is in the carbon pool that feeds them; a methane parameter fitted to
   compensate would be absorbing a soil-carbon error, which is the trap
   documented throughout 5i-5n.
3. The tractable options are **prescribing northern soil carbon from NCSCD**,
   or accepting the deficit and reporting high-latitude CH4 as a known
   underestimate. The former is a data question, not a modelling one.

---

## D6. Verification plan (the original design)

No F77 counterpart exists, so `verify/STATUS.md`'s two-layer pattern is replaced by:

- **Layer 1 — conservation and algorithmic correctness.**
  - Mass balance closes every step: `ΔCH4_pool = P - O - (E + Tp + D)` to machine precision.
  - No negative pools; clamp activations counted and logged, not silently absorbed.
  - `P = 0` when `zwt = 0` (fully drained) and when `tsoi < 273.15`.
  - `O → 0` as `zwt → 0`; monotonic in `f_aer`. **Explicitly regression-test the sign**, since the prototype had it backwards.
  - NumPy transcription cross-check of the full daily step.
- **Layer 2 — site.** FLUXNET-CH4 towers: magnitude, seasonal phase, and the
  production/oxidation partition where flux-partitioning data exist.
- **Layer 3 — global.** Total against Global Methane Budget 2025 (~153 Tg CH₄ yr⁻¹,
  range 136–168). **Report the extent-driven and flux-driven error separately** —
  that separation is what the field currently lacks, and prescribing extent in
  Phase 1 is what makes it possible for us.

Known bias to state up front in any writeup: no peat carbon pool and a 4 m soil
column ⇒ northern peatlands underrepresented. Either scope to "wetlands excluding
deep peat" or quantify the bias. Do not leave it implicit.

---

## D7. Open questions (the original design)

1. **Domain — global 0.5° or CONUS?** CONUS largely removes the peatland problem
   and has denser FLUXNET-CH4 coverage; global is what the field expects.
   *Undecided.*
2. Which inundation product? WAD2M, GIEMS-2 and SWAMPS disagree substantially and
   this is the largest single lever on our global total.
3. ~~Is `tco2mic` exposed at the timestep level?~~ **CLOSED 2026-08-12** — it is a
   per-timestep field on `VegetationState` (`state/vegetation.py:66`) and an
   existing unit conversion lives at `loops_gswp3.py:415`. No plumbing needed.
4. Depth weighting `w_rh[k]` — derive from IBIS's soil C profile, from `froot`,
   or fit? IBIS's CENTURY does not resolve soil C by layer, which may force a
   prescribed exponential profile.
5. Do we need `f_pE` (redox, 30-day e-folding after inundation)? CLM5 defaults it
   to 1.0. *Leaning: omit in v1.*
6. Timestep — daily, or sub-daily to capture ebullition episodicity?

---

## D8. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-10 | Write our own module rather than port CLM5-CH₄ wholesale | CLM5's cost is in its numerics (4 tridiagonal solves/step), which buy vertical detail unvalidatable at 0.5°. Its equations are worth taking; its solver is not. |
| 2026-08-10 | Two-zone bulk split by water table, not a resolved column | Keeps within the no-new-solver constraint. |
| 2026-08-10 | Single-MM oxidation with an aeration factor, not double-MM | Removes the prognostic O₂ profile and the second PDE. |
| 2026-08-10 | Prescribe inundated fraction in Phase 1 | Extent dominates global uncertainty; prescribing it makes our own flux skill measurable instead of confounded. |
| 2026-08-10 | Quarantine `bgc/wetlands.py` rather than extend | Backwards oxidation sign, `fwet²` double-counting, unreachable ebullition threshold, fictitious TOPMODEL. |
| 2026-08-10 | Do ebullition properly despite the light budget | Only 25% of 16 reviewed models represent it adequately; cost is ~10 lines. |
| 2026-08-10 | **Restructured as tiered M-lite → M-mid → M-full; build M-lite first** | Code survey: CLM's 35 params reduce to a 6-param core (the rest are lake CH₄, redox lag and sensitivity scale factors). WetCHARTs reports >80% of wetland CH₄ uncertainty comes from decomposition rate + extent — neither oxidation nor transport is in that 80%. Building M-full first spends effort on the 20%. |
| 2026-08-10 | Phase-1 prescribed inundation confirmed as standard practice, not a shortcut | `ch4Mod.F90` imports `ch4FInundatedStreamType` — CLM ships a stream-file path for `finundated` alongside the prognostic one. |
| 2026-08-10 | Stop rule added: if M-lite matches M-full's global total within observational spread, do not build M-full | Prevents complexity creep; the negative result is itself publishable. |

---

## D9. Build log

> Append one entry per work session: what changed, what was measured, what broke.

- **2026-10-20 -- the dynamic wetland-area switch (section 7) built.** `ibis.bgc.wetland_area`: `root_zone_water` (top
  metre, poros x wsoi x hsoi, frozen layers 0 -- the regression study's Wrz) and `dynamic_fraction`
  (f = clip(f_clim (1 + beta A), f_min, f_max), A standardised against a monthly climatology, A = 0 under a frozen top
  layer -- the first version zeroed f itself there, which removed boreal shoulder-month wetland the WAD2M climatology
  has (-21 % boreal CH4 in a global year at beta 0.06) and broke "beta 0 = the climatology"; a West-Siberian box test had
  not caught it because that domain has no frozen month with nonzero climatology -- invariants are checked globally);
  `ibis.io.wad2m.read_wad2m_bounds` (per-cell range over 2000-2020); `verify/methane/wetland_clim_prep.py` (mu, sd per
  cell and month from a hindcast's monthly wsoi/tsoi with the model's own porosity; s7 2001-2010 ->
  `inputs/methane/wetland_clim/wetland_clim_s7_2001-2010.npz`, W 0.137 +- 0.018 m); run_global: `physics.wetland_dynamic`
  + `forcing.wetland_clim_file` + the yaml `wetland_area:` betas by band (W01/W02/W03 when the 3.0 types are on, else the
  latitude bands), the fraction written as `finund`. Off by default; beta 0 is the bounded climatology. Betas from the
  regression: tropical 0.13 (Wrz), boreal 0.06 (W lag 0); the temperate storage beta was not tabulated (the band's best
  predictor was precipitation), 0.07 provisional. Not done: the water-table factor held at climatology inside f_inund
  (the design's last clause) -- the module diagnoses the water table from the same soil water, so the wet anomaly is
  counted twice; to be measured on a global run before deciding whether it matters. Tests: `tests/test_wetland_dynamic.py`;
  wiring checks: a three-arm cold-start box on West Siberia (off == beta 0 bit-for-bit; beta moves every cell within
  its bounds in summer) and two global years from the s7 2000 state beside the s31a run (beta 0 against the cover run's
  2001 as the off control, and beta 0.13 / 0.07 / 0.06). Results: beta 0 leaves GPP, ET, soil water, soil temperature
  and LAI exactly as the off run and CH4 within 3.7e-10 kgC m-2 (1.6e-6 relative, 6 cells: float32 roundoff of the
  methane step); the betas change CH4 in 43,905 cells, wetland CH4 2001 +0.4 % globally (+0.6 % tropical, +1.1 %
  temperate, -3.0 % boreal) at an unchanged annual-mean fraction (0.0168): the small effect the design expected for a
  single year; the interannual behaviour is for a multi-year run. Full suite 728 passed with the switch in place.

---

## D10. References

1. Riley et al. (2011), *Barriers to predicting changes in global terrestrial methane fluxes: analyses using CLM4Me*, Biogeosciences 8, 1925. https://bg.copernicus.org/articles/8/1925/2011/bg-8-1925-2011.html
2. *CLM5.0 Technical Note — Methane Model*. https://escomp.github.io/CTSM/tech_note/Methane/CLM50_Tech_Note_Methane.html
3. Zhu et al. (2014), *Modelling methane emissions from natural wetlands by development and application of the TRIPLEX-GHG model*, Geosci. Model Dev. 7, 981. **Built on IBIS.** https://gmd.copernicus.org/articles/7/981/2014/
4. Xiao et al. (2024), *Global wetland methane emissions from 2001 to 2020: magnitude, dynamics and controls*, Earth's Future. **IBIS-CH4.** https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024EF004794
5. Wania et al. (2010), *Implementation and evaluation of a new methane model within a DGVM: LPJ-WHyMe v1.3.1*, Geosci. Model Dev. 3, 565. https://gmd.copernicus.org/articles/3/565/2010/gmd-3-565-2010.html
6. *Review and synthesis: peatland and wetland models simulating CH₄ production, oxidation and transport pathways*, EGUsphere preprint (2024). https://egusphere.copernicus.org/preprints/2024/egusphere-2024-3852/
7. Xu et al. (2016), *Four decades of modeling methane cycling in terrestrial ecosystems*, Biogeosciences 13, 3735. https://bg.copernicus.org/articles/13/3735/2016/
8. *WETMETH 1.0: a new wetland methane model for implementation in Earth system models*, Geosci. Model Dev. 14, 6215 (2021). https://gmd.copernicus.org/articles/14/6215/2021/
9. *Ensemble estimates of global wetland methane emissions over 2000–2020*, Biogeosciences 22, 305 (2025). https://bg.copernicus.org/articles/22/305/2025/
10. Dunfield et al. (1993) — pH dependence of methanogenesis (source of `f_pH` form).
11. Gedney et al. (2004), *Climate feedback from wetland methane emissions*, Geophys. Res. Lett. 31. **The ~3-parameter scheme still used as the JULES default.** https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004GL020919
12. Bloom et al. (2017), *A global wetland methane emissions and uncertainty dataset for atmospheric chemical transport models (WetCHARTs v1.0)*, Geosci. Model Dev. See also Parker et al. (2020), *Exploring constraints on WetCHARTs using GOSAT observations*, Biogeosciences 17, 5669. https://bg.copernicus.org/articles/17/5669/2020/
13. *Evaluation of wetland CH₄ in JULES using satellite observations*, Biogeosciences 19, 5779 (2022). https://bg.copernicus.org/articles/19/5779/2022/

### Code consulted directly

- `ESCOMP/CTSM` — `src/biogeochem/ch4Mod.F90` (4410 lines, 35 params)
- `NGEET/fates` — `fire/SFMainMod.F90`, `SFParamsMod.F90`, `SFNesterovMod.F90`
- `PIK-LPJmL/LPJmL` — `src/spitfire/` (30 files)
- `ESCOMP/CTSM` — `src/biogeochem/CNFireLi2014Mod.F90`, `CNFireLi2021Mod.F90`, `CNFireBaseMod.F90`

Not publicly available: JULES (MOSRS), ORCHIDEE (forge registration), LPJ-GUESS,
LPJ-WHyMe, TEM, ecosys. No JAX or Python reference implementation of either
scheme exists — searched, nothing credible. **We would be first on that.**

## 5r. The high-latitude deficit was `vmax x0.54`, not structure (2026-09-16)

5q diagnosed the 1.0% Arctic share (target 8%) as structural: no
vertically resolved decomposition, no cryoturbation, no deep carbon, Arctic
soil carbon 4.33 against NCSCD ~55 kgC m-2. Those gaps are real. They were
not the cause.

The calibration work (docs/CALIBRATION.md 4, 8) found that `vmax x0.54` --
the global scalar behind both_fix's GPP agreement -- had put nine of twelve
PFTs below their physiological minimum and suppressed high-latitude
productivity below viability: 60-70N LAI 0.48 against MODIS 1.40, NPP at
70-90N 0.003 kgC m-2 yr-1. With no carbon input there was nothing to
decompose, and no vertical scheme would have changed that.

On the v11d state (default-physics spin-up, per-PFT calibration, no scalar),
with **no methane parameter changed**, `biome_tref` gives a 60-90N share of
**10.3%** and every configuration's Arctic share rose 4-12x. Full table in
CALIBRATION.md 12. The 5q conclusion is withdrawn; the module was right.

Two consequences. Magnitude is now 251 Tg for `biome_tref` -- x1.38 on
182 -- and Rh is x1.36: the excess is the carbon cycle's, and 5m's refusal to
fit `r_CH4` stands. And the ranking inverted a second time: `biome_noq10`
(Q10 = 1) now has the best latitudinal shape (10.3) while `biome_tref` keeps
the better seasonal peak. A production choice between them belongs on
FLUXNET-CH4 sites, not on another global re-rank.

The general caution from 5q applies to 5q itself: a structural diagnosis
made on a biased carbon state can be wrong for the state's faults.

## 5s. The global driver read a latitude-mirrored wetland map (found 2026-10-09)

`python-port/scripts/run_global.py` passed its GSWP3 row indices (north-to-south: GSWP3's latitude runs 89.75 to
-89.75) into `read_wad2m_climatology`, whose array runs south-to-north (`_check_grid` compares the file to the
south-to-north `mlat` the caller supplies and passes). Every land point therefore read the inundated fraction of its
mirror latitude: 0.00 Mkm2 north of 60 N (the Antarctic rows), 0.04 in 30-60 N, 1.14 in the tropics, against the map's
0.43 / 1.04 / 2.20. In the section-37 transient the wetland CH4 came out 64 / 3 / 0 Tg by band. `verify/methane/
evaluate_global.py` was never affected (it builds `wli` from the land latitudes), so the 5i-5r evaluations on states
stand; any methane number taken from a run_global log or output before this date is not usable. The symptom was seen
once before and attributed to the water-table diagnostic (`wetland_water_table`'s docstring: "exactly zero north of
60 degrees"); that fix was real but was not the cause of the zero. Fixed by indexing the climatology from the land
points' own coordinates (commit 80c48d9); the mean inundated fraction of the land points went from 0.0046 to 0.0168.
The yearly `aych4` written by the driver also accumulated across years until the same day (no `reset_annual` call);
both fixes precede the section-39 hindcast.


## 6. Dynamic wetland area: what precipitation can and cannot do (2026-10-12)

Min asked for a switch, off by default, that scales the WAD2M climatology with precipitation for years without
observations. Before any code, two things were done (CALIBRATION.md 44): a regression of WAD2M 2000-2010 on the
model's own GSWP3 precipitation per band (`verify/calibration/diagnostics/wad2m_precip_regression.py`, outputs in
`verify_outputs/wad2m_regression/`), and a literature review. Both reports are kept in full in
`ibis_local_document/logs/` (`wad2m_precip_regression_2026-10-12.md`, `wetland_area_review_2026-10-12.md`); the
review follows.

### Interannual variability of wetland area: literature support for an empirical precipitation-driven extent mode in IBIS-JAX

Target equation: `f_inund(m) = f_clim(m) x g(precipitation anomaly)`. Numbers are quoted from the papers; "not reported" means no number was given in the accessible text. My own inferences are labelled.

#### 1. Observational products and the precipitation share of their IAV

**GIEMS** (Prigent et al. 2007, JGR 112, doi:10.1029/2006JD007847; Papa et al. 2010, JGR 115, doi:10.1029/2009JD012674; Prigent et al. 2012, GRL 39, doi:10.1029/2012GL051276). 1993-2007, 0.25 deg. Global mean annual maximum fell ~5.7 % over 1993-2004 (6 % to 2007), mostly tropical. The precipitation link is only qualitative: "in the tropics, surface water is essentially related to precipitation", with "much worse agreement at higher latitudes"; Prigent 2007 documents a rain-to-inundation hysteresis. No slope or r is given. Quantified relations come from users: Ringeval et al. 2012 (GMD 5, 941, doi:10.5194/gmd-5-941-2012) find for tropical South America a zero-lag seasonal inundation-rainfall r = 0.03 rising to 0.60 at 3-month lag, lags of 0-5 months globally, and interannual r between GIEMS anomalies and a precipitation-driven saturated fraction from -0.30 (tropical Asia) to 0.85 (temperate Eurasia), < 0.1 in several regions. Papa et al. 2013 (JGR 118, doi:10.1002/2013JD020500): Amazon surface-water volume vs basin rain, lagged r = 0.86, rain leading by 2 months.

**GIEMS-2** (Prigent et al. 2020, JGR 125, doi:10.1029/2019JD030711): 1992-2020; "comparisons with precipitation estimates show good agreement" (no r). The 30-year assessment (Frontiers in Remote Sensing 2024, doi:10.3389/frsen.2024.1399234): mean annual max 6.18 Mkm2; interannual-anomaly r vs river discharge 0.60 (Amazon), 0.66 (Congo), > 0.6 in 7 of 9 basins, inundation leading discharge by 1 month (Amazon, Congo, Mekong, La Plata) to 2 months (Mississippi, Yangtze). Precipitation/GRACE: not reported.

**SWAMPS** (Schroeder et al. 2015, Remote Sens. 7, 16688, doi:10.3390/rs71215843; Jensen and McDonald 2019, IEEE GRSL 16, 1402): 1992-present; R2 = 0.91-0.94 vs MODIS/SRTM open water. Precipitation regression: not reported.

**WAD2M** (Zhang et al. 2021, ESSD 13, 2001, doi:10.5194/essd-13-2001-2021): long-term max 13.0 Mkm2 = 3.5 (annual min) + 4.0 (seasonal) + 5.5 (intermittent) Mkm2, so long-term max is 1.7x the mean annual max. Tropics give 68 % of global annual variation; no significant trend 2000-2018; captures the 2010-11 La Nina peak and 2015-16 Amazon decline; basin anomalies agree with GRACE and ESA-CCI soil moisture (figure only). Drivers stated: tropics "primarily driven by precipitation"; high latitudes "surface water runoff from snowmelt, not direct precipitation, contributes towards the lower correlation between inundation extent and precipitation". No slope.

**GIEMS-MC** (Bernard et al. 2025, ESSD 17, 2985, doi:10.5194/essd-17-2985-2025): 1992-2020; inundated+saturated MAmax 3.80 Mkm2. Boreal trends +10e3 km2/yr (May-Jun) and +24e3 km2/yr (Sep-Oct), linked to snow cover. Precipitation: not reported.

**GRACE as proxy** (Bloom et al. 2010, Science 327, 322, doi:10.1126/science.1175176): tropical CH4 variability "largely described" by GRACE water height, extratropical by temperature; variance fraction not given. Feng et al. 2024 (Nat. Commun. 15, doi:10.1038/s41467-024-55266-y): 2020-21 inversion anomalies correlate with GRACE-FO storage, while "correlation with temperature or precipitation changes was not statistically significant across regions"; storage leads emissions by ~1 month.

**IAV amplitude check** (Comm. Earth Environ. 6, 450, 2025, doi:10.1038/s43247-025-02438-3): CYGNSS tropical inundation has 25.7 % and 11.9 % higher relative SD than SWAMPS and WAD2M (Pantanal 5.8x / 3.6x; Sudd annual range 11.4 % vs 2.5 / 4.1 %); inundation vs ONI r = 0.77 (Congo 0.85, Pantanal 0.73, Sudd -0.74); vs GRACE-FO r = 0.81-0.90. The microwave climatology understates tropical IAV.

**Boreal** (Watts et al. 2014, ERL 9, 075001, doi:10.1088/1748-9326/9/7/075001): AMSR-E 2003-11; driest year -6 %, wettest +3 % (N. America) / +7 % (Eurasia) of the mean extent.

#### 2. Empirical extent schemes in CH4 budget work

- **Hodson et al. 2011** (GRL 38, L08810, doi:10.1029/2011GL046861): extent 1950-2005 from linear fits of GIEMS 1993-2000 area to LPJ-simulated runoff, per cell and month or per TransCom region; "each of the fitting methods captured approximately 50 % of the inundation variability"; boreal uses the GIEMS climatology because its IAV is small. Tropical CH4 IAV: R2_inundation = 0.65 vs R2_Rh = 0.39 (boreal 0.16 vs 0.77); tropical emissions vs -MEI R2 = 0.56 (global 0.39) with MEI lagging precipitation 3 months; El Nino -9 +/- 3, La Nina +8 +/- 4 Tg CH4/yr.
- **Bousquet et al. 2006** (Nature 443, 439, doi:10.1038/nature05132): wetlands dominate CH4 IAV; bottom-up used ORCHIDEE with Prigent inundation; any precipitation extrapolation: not retrievable.
- **Bousquet et al. 2011** (ACP 11, 3689, doi:10.5194/acp-11-3689-2011): ORCHIDEE-TOPMODEL saturated-area anomalies (relative to its 1993-2000 mean) are added to the satellite 1993-2000 climatology; no precipitation regression. Sensitivity runs: tropical emissions "mostly driven by precipitations through the wetland extent"; removing precipitation IAV flips the 2006 anomaly; tropical land precipitation +3 % (2007), +10 % (2008). Warning: the model "may overstate the control ... by precipitations" since floodplains are absent and "modelled wetland extent increases continuously with precipitations".
- **WetCHARTs** (Bloom et al. 2017, GMD 10, 2141, doi:10.5194/gmd-10-2141-2017), the only budget product with an explicit precipitation-scaled extent: A_ij(t,x) = w_i(x) h_ij(t,x); w = static GLOBCOVER or GLWD maximum; h = SWAMPS or monthly ERA-Interim precipitation normalised so its mean (GLOBCOVER) or maximum (GLWD) is 1; A = min{w h, 1}. Linear in same-month precipitation, elasticity 1, no lag; skill vs satellite IAV not reported; precipitation "does not provide a direct constraint on the wetland ... extent".
- **Poulter et al. 2017** (ERL 12, 094013, doi:10.1088/1748-9326/aa8391): SWAMPS-GLWD prescribed; global mean area -2 % between 2000-06 and 2007-12 (tropics -3.4 %, boreal +1.8 %); ensemble CH4 IAV partial r: area 0.64, temperature 0.37, precipitation 0.09.
- **Zhang et al. 2017** (PNAS 114, doi:10.1073/pnas.1618765114) and **2018** (ERL 13, doi:10.1088/1748-9326/aac939): prognostic LPJ-wsl TOPMODEL calibrated to SWAMPS-GLWD; "precipitation is the dominant climatic variable regulating interannual variability", temperature the trend; growth rate vs MEI r = -0.35 to -0.51, peak at 3-month lag. **Zhu et al. 2017** (GCB 23, doi:10.1111/gcb.13726): ENSO explains 49 % of tropical CH4 IAV, ~8-month lag. **Zhu et al. 2015** (GEB 24, doi:10.1111/geb.12307): extent scheme not retrieved.
- **GMB protocol** (Saunois et al. 2020, ESSD 12, doi:10.5194/essd-12-1561-2020; 2025, ESSD 17, doi:10.5194/essd-17-1873-2025; Zhang et al. 2025, BG 22, 305, doi:10.5194/bg-22-305-2025): diagnostic runs prescribe WAD2M (v1 2000-2017, v2 2000-2020), prognostic runs use own hydrology; no precipitation scaling. Ensemble max 8.0 +/- 2.0 Mkm2, seasonal amplitude 4.7 +/- 2.0; no trend in prognostic annual-mean area 2000-2020; +5.2 Tg (2010 La Nina), -4.6 Tg (2015 El Nino). The 2000-2025 extension (Li et al. 2026, ESSD preprint, doi:10.5194/essd-2026-176) emulates emissions directly from ERA5 soil temperature and precipitation, bypassing extent.

#### 3. Process-based (TOPMODEL-type) schemes

- **Gedney and Cox 2003** (J. Hydrometeorol. 4, 1265) / **Gedney et al. 2004** (GRL 31, doi:10.1029/2004GL020919): wetland fraction = share of the topographic-index distribution with local water table at the surface; inputs: index statistics, simulated moisture deficit. IAV skill: not reported.
- **Ringeval et al. 2010** (GBC 24, doi:10.1029/2008GB003354), GIEMS prescribed: area variation gives ~30 % of annual flux for boreal bogs, 40 %/66 % of temperate/tropical seasonal amplitude; interannually area is "the dominant cause". **Ringeval 2012**: ORCHIDEE-TOP seasonal figure-of-merit 64-92 %, interannual r -0.30 to 0.85; the modelled fraction tracks precipitation (r > 0.54 outside boreal; 0.81 tropical Asia, 0.94 S. Africa) while observed inundation does not, because floodplain routing is missing.
- **DYPTOP** (Stocker et al. 2014, GMD 7, 3089, doi:10.5194/gmd-7-3089-2014): per-cell asymmetric sigmoid f = [1 + v exp(-k(Gamma - q))]^(-1/v) fitted to the CTI distribution, Gamma = monthly water-table index from soil water + runoff, CTImin caps f_max; Amazon seasonal max 13 % vs 15 % observed; regional totals/timing agree with GIEMS, spatial heterogeneity not; retreat too fast where routing matters. IAV r: not reported.
- **LPJ-wsl** (Zhang et al. 2016, BG 13, 1387, doi:10.5194/bg-13-1387-2016): F_max calibrated to SWAMPS-GLWD; Spearman r of interannual anomalies vs SWAMPS-GLWD (GIEMS): boreal N. America 0.77 (0.38), boreal Eurasia 0.79 (0.51), tropical S. America 0.72 (0.84), tropical Asia 0.69 (0.67), S. Africa 0.08 (0.74), temperate Eurasia 0.74 (0.65), temperate N. America 0.36 (0.14), Europe 0.60 (0.09). Global annual max 5.17 Mkm2 (GIEMS 4.47, SWAMPS-GLWD 6.62); IAV sigma of annual max 0.11 / 0.20 / 0.18 Mkm2 (2.7-4.5 % of max).
- **JSBACH** (Kleinen et al. 2020, Clim. Past 16, 575, doi:10.5194/cp-16-575-2020): z_i = z_bar + (1/f)(chi_i - chi_bar), f = 2.6, chi_min = 8.5, saturation threshold 0.95; tropics 1.2 vs 0.8 Mkm2 observed; boreal peak phase-shifted. IAV: not evaluated.
- **CLM4Me** (Riley et al. 2011, BG 8, 1925, doi:10.5194/bg-8-1925-2011): f_s = p1 exp(-z_w/p2) + p3 Q_r, three parameters per cell inverted against Prigent 2007 (+ IGBP where < 10 %); "consistent ... at large spatial scales"; IAV r: not reported.
- **Diagnostic TOPMODEL on reanalysis soil moisture** (Xi et al. 2022, Sci. Data 9, doi:10.1038/s41597-022-01460-w): same sigmoid, 3 parameters per cell; global interannual r = 0.79 (ERA5) / 0.77 (ERA5-Land) vs GIEMS-2, 0.70 / 0.66 vs WAD2M; boreal 0.59-0.88, temperate 0.36-0.90 vs GIEMS-2. Best documented IAV skill of any scheme (same approach: Xi et al. 2021, Nat. Clim. Change 11, 45; Hardouin et al. 2024, Earth's Future 12, doi:10.1029/2024EF004553).
- **WETCHIMP** (Melton et al. 2013, BG 10, 753, doi:10.5194/bg-10-753-2013): +3.9 % uniform precipitation gave "a consistent small positive response in CH4 fluxes and wetland area" (magnitude not retrieved); none of the models reproduced the GIEMS 1990s decline.

#### 4. Dominant driver by climate zone

- **Tropics**: precipitation integrated over 1-3 months and routed through rivers/floodplains. Amazon storage vs rain r = 0.86 at 2-month lag (Papa 2013); inundation vs rain r 0.03 -> 0.60 at 3-month lag (Ringeval 2012); GIEMS-2 anomalies vs discharge r = 0.60-0.66 (2024 assessment); inundation vs GRACE r = 0.81-0.90 (2025 CEE); floodplain area is a function of river stage, not local rain (Hamilton et al. 2002, JGR 107, 8038, doi:10.1029/2000JD000306; regressions not retrieved). ENSO integrates this: tropical CH4 vs MEI R2 = 0.56 (Hodson), 49 % with 8-month lag (Zhu 2017), inundation vs ONI r = 0.77 (2025); the 2011 La Nina gave +4.9 mm/month tropical land precipitation, SWAMPS area expansion and >= 5 % higher tropical emissions (Pandey et al. 2017, Sci. Rep. 7, 45759, doi:10.1038/srep45759). Caveats: Feng 2024 finds storage, not precipitation, significant; the Sudd has the opposite ENSO sign.
- **Boreal/Arctic**: snowmelt timing, thaw and frozen-soil drainage. WAD2M and Papa 2010 both report weak precipitation correlation; Hodson used a climatology (R2_inund = 0.16); Watts IAV +/-3-7 %; Zhang et al. 2024 (Nat. Clim. Change 14, doi:10.1038/s41558-024-01933-3) find boreal-Arctic CH4 IAV dominated by temperature (42 % of cells) and GPP (24 %), water-related drivers 18 %, no WAD2M area trend; GIEMS-MC trends sit in the melt and freeze-up months; prognostic models mis-phase the boreal peak (Kleinen 2020; Stocker 2014).
- **Temperate**: weakest and human-affected: TOPMODEL vs observation r 0.09-0.65 (Zhang 2016); GIEMS declines co-locate with population growth (Prigent 2012). (Prigent 2007 and Papa 2010 statements are from abstracts; full texts were not accessible.)

#### 5. Synthesis

**Form.** A bounded multiplicative anomaly: f = min(f_clim(m) x g, f_max), g = 1 + beta (P_w - P_w,clim)/P_w,clim or (P_w/P_w,clim)^beta, P_w over window w. Precedents: WetCHARTs (linear, beta = 1, same month), Hodson (linear in runoff per cell/month), Bousquet 2011 (anomaly transplant). TOPMODEL theory and the fitted sigmoids (Stocker 2014; Xi 2022) imply a saturating response, so g must be capped and a power form with beta < 1 for wet anomalies is more defensible than unbounded linear (Bousquet 2011 flags the overshoot).

**Window.** Tropics: 2-3 months, lagged 1-2 months (Papa 2013; Ringeval 2012; Hodson's 3-month lag). Boreal: no precipitation window is supported; use thaw/melt indicators or g = 1. Temperate: 1-3 months (not quantified).

**Coefficients.** No paper reports an elasticity of area to precipitation. Inference (mine): observed IAV of global annual-max area is 2.7-4.5 % (Zhang 2016) against tropical land precipitation anomalies of ~3-10 % in anomalous years (Bousquet 2011), consistent with beta ~ 0.5-1 in the tropics (WetCHARTs' beta = 1 is the upper end; the CYGNSS finding that microwave products damp tropical IAV by 12-26 % argues against < 0.7). Boreal: beta ~ 0-0.3. Temperate: ~0.5, poorly constrained. Fit them: regress WAD2M or GIEMS-2 monthly anomalies (1992/2000-2020) on the chosen P_w anomaly per climate band.

**Bound.** Cap per cell at the WAD2M long-term maximum (1.7x mean annual max globally) or GIEMS-2 MAmax; Zhang 2016's calibrated F_max (10.3 vs 5.2 Mkm2) implies a similar ~2x ceiling. Floor at the WAD2M annual minimum, not zero.

**Expected skill.** Precipitation-only schemes reach roughly half of the IAV, and only in the tropics: Hodson ~50 % (runoff-fitted); Ringeval 2012 precipitation-driven vs observed r < 0.1 to 0.85; Feng 2024 and the 2025 CEE paper find storage, not rain, significant. Schemes with memory and routing do better: Xi 2022 r = 0.79 global vs GIEMS-2 (~60 % of variance), Zhang 2016 r 0.7-0.8 boreal/tropics; GRACE storage is the strongest single predictor (Bloom 2010; Feng 2024). Nothing shows a precipitation-only scheme above ~50 % outside the tropics; in the boreal zone the area signal is small and thaw-driven.

#### Summary table

| Product / scheme | Driver | Lag / window | Variance explained or slope | Region | Reference |
|---|---|---|---|---|---|
| GIEMS | precipitation | hysteresis; 0-5 mo | qualitative; seasonal r 0.03 -> 0.60 at 3 mo; IAV r -0.30 to 0.85 vs precip-driven model | tropics / regions | Papa 2010; Prigent 2007, 2012; Ringeval 2012 |
| GIEMS Amazon volume | precipitation | 2 mo | r = 0.86 | Amazon | Papa 2013 |
| GIEMS-2 | river discharge | leads 1-2 mo | r_ano 0.60 Amazon, 0.66 Congo | 9 basins | Frontiers RS 2024 |
| WAD2M | precip (tropics), snowmelt (boreal) | - | tropics 68 % of annual variation; ENSO captured; no slope | global | Zhang 2021 |
| GIEMS-MC | snow cover | - | boreal +10e3 / +24e3 km2 yr-1 | boreal | Bernard 2025 |
| CYGNSS | ENSO, GRACE | 3-mo means | r 0.77 vs ONI; 0.81-0.90 vs GRACE-FO; RSD 12-26 % > WAD2M/SWAMPS | tropics | CEE 2025 |
| GRACE | storage | ~1 mo lead | "largely describes" tropical CH4; precip n.s. | tropics | Bloom 2010; Feng 2024 |
| Hodson runoff fit | LPJ runoff | per cell/month | ~50 % of inundation IAV | global | Hodson 2011 |
| WetCHARTs | monthly precipitation | 0 | linear, elasticity 1, capped; skill n.r. | global | Bloom 2017 |
| ORCHIDEE transplant | TOPMODEL/precip | - | precip IAV flips 2006 sign; +3/+10 % precip 2007/08 | tropics | Bousquet 2011 |
| LPJ-wsl TOPMODEL | soil water | - | IAV r 0.7-0.8 boreal/tropics, 0.1-0.6 temperate | regions | Zhang 2016 |
| Diagnostic TOPMODEL | reanalysis soil moisture | - | r 0.79 (GIEMS-2), 0.70 (WAD2M) | global | Xi 2022 |

#### Recommendation

Implement the optional mode as `f = clip(f_clim(m) x (1 + beta_band x A_w), f_min, f_max)`, where A_w is the relative anomaly of a 3-month running mean of IBIS-simulated total runoff or soil-water storage (not raw precipitation), lagged one month in the tropics; fit beta per climate band by regressing WAD2M or GIEMS-2 anomalies on the same IBIS variable over the satellite period (expect ~0.7-1 tropics, ~0.5 temperate, 0-0.3 boreal, where the mode should probably stay off); set f_max = WAD2M long-term maximum and f_min = WAD2M annual minimum per cell. Using the model's own storage anomaly is what Hodson (runoff), Riley (water table + runoff) and the GRACE studies all point to, costs nothing extra in IBIS, and is the only route the literature suggests can approach the ~60 % of interannual variance that soil-moisture TOPMODEL diagnostics reach; a precipitation-only version will plausibly capture about half of the tropical IAV and little elsewhere, and should be documented as such.

## 7. Dynamic wetland area driven by the model's own water: what others did and what our data say (2026-10-16)

Decision (Hangkai, 2026-10-16): the switchable wetland area of Min's point 4 is driven by the model's own runoff or
soil-water anomaly, after background research. Two studies, kept in full in `ibis_local_document/logs/`:

* `wetland_storage_review_2026-10-16.md` -- how Hodson 2011 (LPJ runoff regression), CLM4Me/CLM5 (water table + runoff;
  total water storage), ORCHIDEE and JULES (TOPMODEL), DYPTOP and LPJ-wsl (water-table index + sigmoid), JSBACH and the
  Xi et al. 2022 diagnostic product drive extent. Lessons: tropical floodplains need accumulated runoff with a ~3-month
  lag and local soil-water schemes fail there; boreal interannual variability is small (Hodson used the climatology) and
  unfrozen soil water does best; every working scheme bounds the area by the cell's observed range, handles snow and
  frozen soil explicitly and works with anomalies against the model's own climatology; WAD2M carries almost no tropical
  interannual signal (Xi 2022: r <= 0 in 60S-30N), so a beta fitted to it will be small there; the water-table term must
  not count the same wet anomaly a second time.
* `wad2m_storage_regression_2026-10-16.md` -- WAD2M 2001-2010 against the IBIS 3.0 hindcast's own variables at the same
  19,278 wetland cells: no model variable explains WAD2M's interannual variability better than precipitation; column or
  root-zone soil water has the best seasonal shape (seasonal r 0.85; boreal 0.91) and is no worse interannually; the
  model's runoff output was unusable (the diagnostic defect of DEFECTS_FOR_REVIEW, fixed the same day); a fitted
  f = clip(f_clim (1 + beta A), f_min, f_max) reproduces 5-20 % of the observed interannual amplitude with beta per band
  0.06-0.13 (95 % CIs above zero) and little hold-out skill except the temperate band.

Design adopted for the switch (off by default, as Min asked): soil-water storage of the unfrozen layers of the top
metre, as a standardised anomaly against the model's own monthly climatology, applied to the seasonal part of the WAD2M
climatology, bounded by each cell's WAD2M minimum and maximum, zero under snow or frozen soil, one beta per band fitted
on 2000-2010 and checked on later years, and the CH4 water-table factor held at its monthly climatology inside f_inund
in that mode. The expected effect is small; the switch exists for the transient and future runs.

## 8. Can the model's own hydrology carry the wetland area? The feasibility test (2026-10-29)

Hangkai (10-29): the wetland area must be dynamic, seasonally and from year to year. Before any build, the question the
literature of sections 6-7 left open was measured on the final run: how much of the WAD2M monthly inundated fraction
does a model-side driver explain, cell by cell? `verify/methane/wetland_fit_diag.py` and `wetland_fit_drivers.py` fit,
per land cell, f = f_max sigmoid(k (X_lag - X0)) with f_max the cell's WAD2M maximum, X a model driver from the s31t
hindcast (2001-2010 monthly: top-metre liquid soil water W, 3-month runoff, 3-month precipitation, runoff + W) at the
best lag 0-3 months, in-sample and held-out (fit 2001-05, test 2006-10); tables `wetland_fit_diag_s31t.txt`,
`wetland_fit_drivers_s31t.txt` in figures/final_2026-10-26/tables.

| band | monthly R2 of the WAD2M climatology | best model driver, R2 in-sample / held-out | IAV r of annual area, held-out | seasonal amplitude fit / obs, Mkm2 |
|---|---|---|---|---|
| 90S-30N | 0.57 | W 0.26 / 0.14 | +0.22 (W); runoff +0.61 held-out but +0.01 in-sample | 0.16 / 0.40 |
| 30-60N | 0.59 | W 0.45 / 0.43 | +0.07 (W); negative for runoff and precipitation | 0.55 / 0.31 |
| 60-90N | 0.89 | W 0.65 / 0.65 | +0.71 (W) | 0.76 / 0.96 |
| global | 0.74 | W 0.54 / 0.52 | +0.52 (W) | 1.37 / 1.53 |

Only 10 % of the wet area (35 % north of 60 N) has a cell fit with R2 above 0.5; the best lag is 0 for 78 % of the area.
The band-level anomaly mode of section 7 (f_clim x (1 + beta A)) with beta fitted on the band's annual area reaches IAV r
+0.16 (tropics), -0.40 (30-60 N), +0.30 (60-90 N) with W. The observed interannual range of the annual area is small:
0.24 Mkm2 of 3.7 globally (6 %), 0.15 in the tropics.

**Reading.** The model's local water does not carry the inundation signal where most of the methane is: tropical
floodplains fill from river routing and lagged basin-scale rain, which a column model without routing cannot see, and the
temperate wetlands are managed. A fully prognostic extent from IBIS's own hydrology would halve the seasonal skill of the
climatology in the tropics and add noise interannually; it has real skill only north of 60 N. What the data support now:
the observed monthly WAD2M fields for the years they exist (2000-2020), which give the hindcast both the seasonal cycle
and the year-to-year variation from observation, with the climatology plus a boreal soil-water anomaly outside the record;
a prognostic extent needs river routing (the IBIS-THMB lineage) and belongs after the release.

**Built (2026-10-29): the `physics.wetland_source` switch.** `'climatology'` (default, bit-identical to before) or
`'observed'`: for calendar years inside the WAD2M record (2000-2020, `wad2m_in_record`) the month's inundated fraction
is the observed field (`read_wad2m_month`), written to the monthly output as `finund`; outside the record the climatology
is used, with the `wetland_dynamic` anomaly on top if that switch is on. Tests: `tests/test_wetland_dynamic.py`. Run on
the final state as `verify_outputs/final_s31t_wadobs` (2001-2010 hindcast from the s31t 2000 restart); evaluation
`verify/methane/wadobs_eval.py`, results below.

**Result of the observed-area run (2026-10-29, `wadobs_eval_s31t.txt`).** Against the climatology run, 2001-2010:
global 175.3 against 170.0 Tg CH4 yr-1, all of it tropical (157.4 against 152.1; 30-60 N and 60-90 N unchanged to 0.1 Tg).
The difference grows through the decade, +1.4 Tg in 2001 to +11.0 in 2010, because the observed tropical inundated area
rises from 1.93 to 2.09 Mkm2 over these years (the 2000-2020 climatology sits at the decade's start). The interannual
standard deviation of the global total rises from 4.3 to 6.8 Tg yr-1; the annual anomalies of the two runs correlate at
0.99, so the year-to-year signal the model already had (soil temperature and respiration on a fixed area) is amplified
rather than replaced. The seasonal cycle is unchanged in shape in every band (30-60 N max/min 10.4 -> 10.2). Observed
area used: tropics 2.03 Mkm2 (sd of annual means 0.046), 30-60 N 0.95 (0.037), 60-90 N 0.38 (0.008), global 3.36 (0.063).

**A learned wetland fraction (Hangkai's proposal, 2026-10-29; `verify/methane/wetland_ml_feasibility.py`,
`wetland_ml_variants.py`; tables `wetland_emulator_*_s31t.txt`).** Gradient-boosted trees (scikit-learn
HistGradientBoosting, 400 iterations, 63 leaves, monotone increasing in every water variable), one model per wetland band
(W01/W02/W03), trained on the 49,545 wet cells x 120 months of the s31t hindcast against the WAD2M monthly fraction, no
month-of-year and no climatology as inputs; held out by year (odd -> even and back, so every month is predicted out of
year) and by cell (a random half). Predictor sets: forcing only (precipitation with lags and 5x5-cell neighbourhood means,
surface temperature, frozen soil), model state only (soil water by layer, top-metre liquid water with lags, runoff,
drainage, ET, LAI, soil temperature), both, both with the state as standardised anomalies; static CLM topography (FMAX,
slope, elevation spread, elevation, percent wetland and lake, VIC b), sand, clay, latitude and the cell's WAD2M range.

| predictors | tropics: monthly R2 (clim 0.58) / IAV r | temperate: R2 (0.49) / IAV r | boreal: R2 (0.81) / IAV r | global IAV r |
|---|---|---|---|---|
| both (state + forcing) | 0.34 / +0.85 | 0.34 / +0.09 | 0.74 / +0.59 | +0.59 |
| forcing only | 0.28 / +0.36 | 0.25 / -0.11 | 0.67 / +0.58 | +0.22 |
| state only | 0.30 / +0.77 | 0.33 / -0.13 | 0.74 / +0.51 | +0.53 |
| both, state as anomalies | 0.32 / +0.81 | 0.33 / +0.14 | 0.74 / +0.47 | +0.41 |
| hybrid: observed seasonal climatology + emulator anomaly (both) | 0.59 / +0.85 | 0.47 / +0.09 | 0.80 / +0.61 | +0.58 |

Reading. (1) No predictor set beats the observed seasonal climatology month by month; the emulator's seasonal cycle has
the right shape (r 0.91-1.00) but 60-70 % of the amplitude. (2) The emulator carries the year-to-year variation the
climatology lacks, strongly in the tropics (r 0.85 on ten held-out years) and the boreal zone (0.6), not in the temperate
band (managed wetlands; every scheme in the literature fails there). (3) Forcing alone loses the tropical signal (0.36):
the model's soil water is what carries it, so "both" is the answer to the bias question, and the anomaly form costs
nothing in skill. (4) Without the cell's WAD2M range as a static input the tropics collapse (R2 -0.49), as every
calibrated scheme keeps a satellite maximum. (5) The hybrid, observed climatology plus the emulator's anomaly, keeps the
climatology's seasonal skill exactly and adds the interannual skill; its anomalies are damped (sd ratio 0.30 tropics,
0.66 boreal), which a per-band inflation fitted on the training years corrects. (6) The "bounded" column is misleading:
WAD2M has missing months that the truth zero-fills and the range ignores, so the floor raises winter zeros.
Design taken forward: `wetland_source: learned` = the observed seasonal climatology scaled by the emulator's standardised
anomaly, inflated per band, bounded to [0, cell maximum], with the emulator an input file evaluated host-side each month.

## 9. Hourly methane (2026-10-29)

Hangkai: "I want the methane not monthly, at least daily, better to be hourly! ... If the methane module is called once at
the end of each month with the end-of-month snapshot ... it has large bias and it is 100 % not what I want to see."
Built: `physics.methane_hourly` (off by default, bit-identical off). In the time-step loop (`_run_day_inner`, `loops_gswp3`)
the wetland CH4 flux is evaluated every step from the hourly soil temperature, water and ice with the day's heterotrophic
respiration rate (`tco2mic`, a daily rate because the soil carbon model runs once a day, spread over the steps as the
day/night NEE diagnostic does), accumulated to a day mean, carried through the day scan as a fourth daily output and
averaged over the month; `methane_step_from_mean` books the month (`MethaneState.flux` = the mean, `annual` += mean x days;
the water-table diagnostic stays the end-of-month value). The monthly inundated fraction is unchanged (WAD2M's resolution
is the area's, not the flux's). The site harness takes the same switch (`site_fit.py --methane-hourly`,
`site_transient.py`), so the tower fits can be redone at the hourly cadence. Tests: `tests/test_methane_hourly.py`; the
methane, wetland and wiring test files pass (72). First run on the final state: `verify_outputs/final_s31t_ch4hourly`
(2001-2010 hindcast from the s31t 2000 restart, production bands, climatology area), results below.

**Result, hourly against the month-end snapshot with the production parameters and the climatology area
(`final_s31t_ch4hourly` vs `final_s31t`, 2001-2010; `tables/ch4_hourly_vs_snapshot_s31t.txt`, `ch4_bands_s31t_ch4hourly.txt`).**
Totals barely move: 169.3 against 170.0 Tg CH4 yr-1 (tropics 151.8 vs 152.1, 30-60 N 11.5 vs 11.6, 60-90 N 6.1 vs 6.3);
the annual anomalies correlate at 0.95-1.00 and the interannual spread is 2-30 % smaller in the north. What moves is the
season in the north: with the snapshot, 30-60 N emitted 1.4 Tg in May and 0.8 in October; hourly it emits 0.9 in May and
1.1 in October, 1.7 in September against 1.3, 0.5 in November against 0.3 -- the peak stays in July-August but the curve
shifts two to four weeks later and broadens, because the month-end soil state ran ahead of the month's mean in the
thawing spring and behind it in the freezing autumn. North of 60 N the same: May 0.8 -> 0.3, September 0.8 -> 1.1,
October 0.3 -> 0.5. The GCP prior's cycle in 30-60 N peaks July-August with September at 3.1 and October at 1.7 Tg, so
the hourly shape is the prior's shape; the amplitude gap of the north (0.36-0.52 of the budget) is untouched by the
cadence and stays with the parameters, which the hourly refit of section 80 (CALIBRATION) now re-estimates. The tropics
are unchanged in every month (max/min 1.2). The snapshot's cost was therefore the timing of the northern season, not the
amount; the production parameters had absorbed the amount.

## 10. Monthly means of the soil state, not month-end snapshots (2026-10-29)

The monthly output files carry the soil temperature, liquid water and surface temperature of the LAST time step of the
month (`write_monthly`: "end-of-month model state"); precipitation, runoff, ET and the carbon fluxes are monthly totals.
The wetland-area tests of section 8 used those month-end soil fields as predictors, so they carry the same snapshot error
section 9 removed from the methane flux. Built: `output.soil_means` (off by default). The time-step loop's accumulator
(the one the hourly CH4 uses, generalised) also sums tsoi, wsoi, wisoi by layer and tg over every step; `run_month_gswp3`
returns the month means, and the monthly files gain `tsoi_mean`, `wsoi_mean`, `wisoi_mean` (by layer) and `tg_mean` beside
the month-end fields. Test (`tests/test_soil_means_hourly_ch4.py`, real GSWP3 forcing, three cells, January 2001):
with `methane_hourly` and `soil_means` on, every leaf of the model state except the wetland diagnostics equals the
default run to float32 rounding (bit for bit when the test runs alone; inside the full suite the extra outputs change
XLA's fusion and the soil temperature differs by ~1e-6 relative, as on the GPU, CALIBRATION 82); the means are physical and differ from the month-end state; the hourly CH4 is booked as the month's
mean. The section-8 emulator is retrained on the month means before any switch is built on it.

## 11. The learned wetland area as a switch (2026-10-29; validated in CALIBRATION.md 81-84, the release setting)

`physics.wetland_source: learned` (and `observed_learned`: the observed WAD2M field inside 2000-2020, learned outside),
with `forcing.wetland_learned_file`; needs `methane_hourly`. Off by default. Module `ibis/bgc/wetland_learned.py`:
- `FeatureBuilder`: the ONE construction of the predictors, stepped month by month in calendar order with explicit
  buffers (lags 1-3, 3-month means, 5x5-cell land-only neighbourhood means of 3-month precipitation and runoff), from the
  month-MEAN soil state (section 10) and the month's water totals. The training script
  (`verify/methane/wetland_learned_train.py`) builds its predictors through this class over the reference hindcast's 120
  months, so training and runtime features are identical by construction.
- `LearnedWetland`: the trained file (one XGBoost model per wetland band, monotone increasing in every water variable,
  stored as JSON; per-cell emulator climatology Pbar, WAD2M climatology, cell maximum, static predictors; per-band
  inflation lambda cross-fitted on out-of-year anomalies). f = clip(f_clim[m] + lambda_band (P - Pbar[m]), 0, f_max).
- Runner: in a learned month the methane runs with a unit inundated fraction, the hourly accumulator integrating the
  flux per unit of wetland; after the month the fraction is computed from that month's means and the month's flux and the
  annual total are scaled by it (exact: the flux is linear in the fraction, clip(f, 0, 1)). The month's fraction is written
  as `finund`. No end-of-month state enters anywhere. The predictor buffers (lags, 3-month means) are not in the restart: a
  run started from a restart builds them afresh, as the training did from 2001, so a transient split at 2000 and a
  continuous run differ in the first three months' predictors. The mode needs the full global grid (the file's cells are
  global rows and columns; the run stops on a sub-domain).
Tests: `tests/test_wetland_learned.py` (the builder against the explicit array definitions on a random series; the hybrid's
bounds and its climatology limit; a file applies to its own cells only and rejects a feature list that differs).

**Training on the month means (2026-10-29 21:55; `tables/wetland_learned_v2_training.txt`).** Reference hindcast
`final_s31t_ch4h` (soil means on); XGBoost, 400 rounds, depth 8, monotone; out-of-year (odd <-> even years) against WAD2M:

| band | monthly R2, climatology | monthly R2, hybrid | IAV r of annual area, hybrid | annual anomaly sd, hybrid / observed |
|---|---|---|---|---|
| tropical | 0.52 | 0.56 | +0.90 | 0.23 |
| temperate | 0.46 | 0.48 | +0.48 | 0.28 |
| boreal | 0.80 | 0.81 | +0.58 | 0.77 |
| global | 0.72 | 0.73 | +0.65 | 0.41 |

The hybrid beats the fixed climatology in every band on held-out years and carries the year-to-year timing the climatology
cannot; its seasonal cycle matches the observations month for month (boreal January 0.16 against 0.15 Mkm2). Its
anomalies are damped (a quarter of the observed size in the tropics). Inflation was tested and rejected as the default:
matching the annual variance (lambda 4.4 tropics) costs all tropical monthly skill (R2 -0.18); the monthly-skill optimum is
lambda 0.6-0.7, barely different from 1. The file stores lambda = 1 (`--lambda one`; `r2` and `var` are options).
Two fixes on the way: WAD2M's NaN on land are the frozen months (none in the tropics, 95-99 % of cell-months north of 60 N
from November to April); the model's climatology (`read_wad2m_climatology`) fills them with zero and so must the training's
(the first training averaged with NaN skipped, which put boreal January at 0.62 Mkm2); and the runner's learned block had
swallowed the dynamic-wetland log line, which crashed the first learned run at start-up (no output).
File: `inputs/methane/wetland_learned/wetland_learned_v2_s31t.npz` (the rejected first file kept beside it, renamed).

## 12. Carbon partition of the wetland CH4 (2026-10-30)

Until now the module was diagnostic in carbon: the soil model books all decomposed carbon as heterotrophic respiration
(CO2, `summary.ayco2mic`) and the CH4 was computed from part of it, so the emitted carbon was counted twice by 0.11 PgC
yr-1. `physics.methane_carbon_partition` (off by default; on in the release): after each month's methane is final (after
the learned area's scaling), the month's net CH4 emission is taken out of the month's Rh booked as CO2, never more than
that Rh. Accumulators only, the soil pools having lost this carbon in soilbgc already; nothing the physics reads changes.
Monthly `amco2mic` / `amneetot` and the yearly `ayneetot` are then CO2-only; the yearly output gains `ayrh_ch4c` (the
carbon moved), and `closure_any.py` adds it back when it closes the soil budget. Verified on 2001 against the same
configuration without the partition: Rh_CO2 + CH4-C equals the old Rh to 2e-6 per cell, NEE changes by exactly the moved
carbon (111.5 TgC, the CH4 emitted), 27 cells were clipped at their month Rh with negligible carbon, and every other field
differs only at the level a repeat of the identical run differs (1-4 float32 units in the last place; the GPU does not
repeat bit for bit between launches).

