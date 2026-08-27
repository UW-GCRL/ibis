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
| M2 | **M-lite**: production only (`f_inund · r_CH4 · Rh · Q10`) | ☐ not started |
| M3 | Evaluate M-lite: FLUXNET-CH4 + global total | ☐ not started |
| M4 | **M-mid**: + oxidation, + upland sink — *only if M3 shows systematic bias* | ☐ not started |
| M5 | **M-full**: + ebullition, plant transport, diffusion — *only if pathway partitioning is needed* | ☐ not started |
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
| `r_CH4` | CH₄:CO₂ production ratio | **0.23** | **yes — primary** | ✅ TRIPLEX-GHG calibrated mean (range 0.1–0.7) across 19 sites. Supersedes CLM5's 0.20 — TRIPLEX-GHG is built on IBIS, so its calibration is the closer prior. |
| `Q10p` | production Q₁₀ | **2.48** | **yes — primary** | ✅ TRIPLEX-GHG calibrated mean (range 1.6–4.5) across 19 sites. Supersedes the earlier guess of 3.0. |
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
| FLUXNET-CH4 | site evaluation | ☐ to check |
| Global Methane Budget 2025 ensemble | global benchmark | ☐ to check |
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
