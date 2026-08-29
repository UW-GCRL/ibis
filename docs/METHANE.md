# IBIS-JAX methane module — design & build log

**Status:** design agreed, not yet implemented
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
| M3 | Evaluate M-lite: FLUXNET-CH4 + global total | ☐ not started |
| M4 | **microbial growth/dormancy** (2 params) — the only added process with a replicated skill gain; *only if M3 shows seasonal bias* | ☐ not started |
| M5 | ~~ebullition, plant transport, diffusion~~ — **removed from the plan** (§5f); reinstate only for a question needing pathway partitioning | ✗ dropped |
| M6 | Global evaluation vs Global Methane Budget 2025 | ☐ not started |
| M7 | Differentiable calibration of `r_CH4`, `Q10p` (+ `Vmax` if M-mid) | ☐ not started |

Gate between tiers: do not start the next tier until the previous one is evaluated
and the comparison is written into §9.

Legend: ☐ not started · ◐ in progress · ☑ done & verified

---

## 2. Starting point

### 2.1 F77 IBIS 2.6b4

**Nothing.** `grep -i 'methane\|ch4'` over the entire Fortran source returns zero
hits. This module is greenfield — no legacy path to preserve, no bit-exactness
constraint. That is a freedom the fire module does not have.

### 2.2 `python-port/ibis/bgc/wetlands.py` — quarantine, do not extend

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
in §9. If M-lite matches M-full's global total to within the observational spread,
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
| Global wetland CH₄ **153 (136–168) Tg yr⁻¹** | The global evaluation target (§6) | Wrong benchmark |

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
| 1 | `tco2mic` may only be an annual diagnostic (§7 open question 3) | ✅ **Resolved — it is a per-timestep field.** `VegetationState.tco2mic` (`state/vegetation.py:66`), computed in `physics/biogeochem.py:235`. | **M2 needs no plumbing work.** The open question is closed. |
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

**Open question for M2.** Global wetland area through this reader is 3.06 Mkm²
in January 2010 and 4.64 Mkm² in July. Zhang et al. (2021) quote larger
figures, but whether that reflects a different definition (inundated-only vs
including non-inundated peatland) or an error here is **not yet established**.
Total flux scales linearly with this, so it must be settled against the paper
before any M3 evaluation is believed.

## 6. Verification plan

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

## 7. Open questions

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

## 8. Decision log

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

## 9. Build log

> Append one entry per work session: what changed, what was measured, what broke.

_(empty — first entry goes here)_

---

## 10. References

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
