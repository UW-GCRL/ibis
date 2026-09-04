# Dead-code audit, 2026-08-28

**Why this exists.** Five separate defects this month had the same shape:
something written, tested, exported, and called by nothing — the dead
`firefac`; `fire_scheme` and `enable_fire_emissions` read nowhere; `soil.csoi`
stored and never read; the whole `bgc/` package reachable only from tests. Each
was found *by accident* while looking for something else. This is the
deliberate sweep.

A green test suite certifies that code is **correct**. It says nothing about
whether anything **calls** it.

## Method

Whole repo, 140 production files against 59 test files. "Production" excludes
tests, `_legacy/`, and `__init__.py` — **a re-export is not a caller.** A module
is dead if no production file outside it calls any of its public symbols.

The first version of this script was itself wrong (a Python chained-comparison
bug emptied the production file list, so *everything* looked dead, including
`soil_physics` and `stomata`). Recorded because a broken audit is worse than
none.

## Result: 4 dead config flags, 13 of 55 modules with no production caller

### Config flags a user can set that do nothing

| flag | readers |
|---|---|
| `enable_bvoc` | **0** |
| `enable_permafrost` | **0** |
| `enable_soil_n2o` | **0** |
| `enable_wetlands` | **0** |
| `enable_fire_emissions` | 1 ✅ *(wired 2026-08-24)* |
| `fire_scheme` | 6 ✅ *(wired 2026-08-24)* |
| `isimfire` / `isimveg` / `isimco2` | 6 / 4 / 1 ✅ |

### Modules, in three groups — they are not the same problem

**A. Dead model components.** Written, tested, unreachable. Each has a dead
config flag pointing at it.

| module | symbols | test calls |
|---|---|---|
| `ibis/bgc/bvoc.py` | 6 | 7 |
| `ibis/bgc/nee.py` | 3 | 4 |
| `ibis/bgc/permafrost.py` | 8 | 8 |
| `ibis/bgc/soil_gases.py` | 8 | 9 |
| `ibis/bgc/wetlands.py` | 9 | 13 |

**B. Work in progress.** Expected.

| module | symbols | note |
|---|---|---|
| `ibis/bgc/methane.py` | 13 | M1+M2 built; wiring into the runner is M2's remaining task |

**C. Tools whose callers were never committed.** *This is the most actionable
group.* These are correct, useful, and used — by throwaway scripts in `/tmp`.
Every methane and fire evaluation run this month used them, and **none of that
analysis is reproducible by anyone else.**

| module | symbols | committed caller |
|---|---|---|
| `ibis/io/gfed5.py` | 2 | none |
| `ibis/io/popdens.py` | 3 | none |
| `ibis/io/wad2m.py` | 4 | none |
| `ibis/ml/base.py`, `registry.py`, `wrappers.py` | 10 | none |
| `ibis/ml/fire_calibration.py` | 7 | none |

## Triage

| group | action |
|---|---|
| **A** — dead components | **Decide per module: wire, quarantine, or delete.** `wetlands.py` is already documented as quarantined (docs/METHANE.md §2.2). `permafrost.py` is on the roadmap. `bvoc`, `nee`, `soil_gases` are on no current path — quarantine to `_legacy/` unless wanted. |
| **B** — methane | Wire during M2 completion. |
| **C** — uncommitted callers | **Commit the analysis scripts.** Highest value for a lab tool: it converts one person's `/tmp` into something the group can rerun. |

## Guard

`tests/test_no_dead_code.py` fails if a new dead flag or module appears, and
*also* fails if a listed one is fixed without updating the list — so the debt
can only shrink. The allowlists are debt, not permission.


---

# Resolution, 2026-08-28

Worked through one at a time; every step verified before the next.

| # | fix | dead modules after |
|---|---|---|
| — | audit baseline | **13** |
| 1 | `verify/methane/evaluate_global.py` committed → `io/wad2m.py`, `bgc/methane.py` live | 11 |
| 2 | `verify/fire/evaluate_gfed5.py` committed → `io/gfed5.py`, `io/popdens.py`, `ml/fire_calibration.py` live | 8 |
| 3 | `ml/{base,registry,wrappers}.py` → `ml/_legacy_unwired/` | 5 |
| 4 | `bgc/{bvoc,nee,soil_gases,wetlands}.py` → `bgc/_legacy_unwired/` | **1** |
| 5 | `enable_wetlands`, `enable_soil_n2o`, `enable_bvoc` removed from `RunConfig` | 1 |

**Final state: 1 of 48 modules unwired, 1 dead config flag — both
`permafrost`, both deliberate and tracked.**

## What each fix actually bought

**Fixes 1–2 were the valuable ones.** The evaluation scripts existed only in
`/tmp`. Every fire and methane number in FIRE.md and METHANE.md came from code
nobody else could run. Both committed scripts were verified to reproduce their
documented results exactly — the fire metrics (×0.92, rank +0.749/+0.410, NMSE
0.723, |lat| 11.9) and all four methane configurations (473.6 / 353.1 / 474.5 /
376.3 Tg CH₄/yr), including the calibration null result.

**Fixes 3–4 removed traps rather than adding capability.** A newcomer reading
`ibis.ml` would reasonably assume `NeuralPhysics` was wired; reading `ibis.bgc`
would assume `wetland_step` ran. Neither was true. `training.py` and
`fire_calibration.py` stayed — they have real callers.

**Fix 5** removed flags a user could set that could no longer do anything.

## Two incidental findings

**My first audit script was wrong**, and its output was absurd enough to catch
(`soil_physics` and `stomata` reported dead). A Python chained-comparison bug,
`"_legacy" in str(p) is False`, emptied the production file list.

**419 macOS AppleDouble files** (`._*.py`) sit in the working tree — resource
forks from a Mac copy. **None is tracked in git.** They are inert, but they
broke a `read_text()` and they hid from a bash glob, since `*` skips dotfiles.
Worth deleting; left in place pending a decision.

## The guard

`tests/test_no_dead_code.py` fails when new dead code appears **and** when a
listed item is fixed without updating the list. The allowlists shrank from 4
flags + 13 modules to 1 + 1 across this session, which is the behaviour
intended.


---

# Closed, 2026-08-29

**0 dead config flags, 0 of 48 modules without a production caller.**
`KNOWN_DEAD_MODULES` is now empty; `KNOWN_DEAD_FLAGS` holds only the two
selectors for the quarantined ML framework.

| # | fix | dead modules |
|---|---|---|
| 1–5 | (see above) | 13 → 1 |
| 6 | `bgc/permafrost.py` wired; `enable_permafrost` read | **0** |
| — | 419 AppleDouble files deleted (already `.gitignore`d at lines 57–60) | — |

## Wiring permafrost found two bugs in it

Not a formality: it was wired, tested against observations, and produced wrong
numbers twice before it was right.

**1. A hardcoded layer thickness.** `diagnose_frozen_ground` used
`layer_thickness = 0.5  # approximate` while IBIS's layers are
`[0.10, 0.15, 0.25, 0.50, 1.00, 2.00]`. A first-frozen layer at index 2 reported
1.0 m where the truth is 0.25 m — 4× — and the frozen fraction counted the
0.10 m surface layer equally with the 2.00 m bottom one. Now thickness-weighted,
with `hsoi` threaded through.

**2. A unit error in the Stefan equation, worth a factor of 294.** `k` is
W m⁻¹ K⁻¹ = J s⁻¹ m⁻¹ K⁻¹, so `2·k·TDD / (ρ·L_f·θ)` has units of m² only if TDD
is in degree-**seconds**. `gdd0` is degree-**days**, passed straight in.
Measured: **0.0039 m** against an observed Arctic active layer of 0.3–2 m; with
the ×86400, **1.31 m**.

**3. And a wiring bug of mine.** I first called `permafrost_step` annually. It
reads the *instantaneous* temperature profile, so a year-end call reads December
— winter, nothing thawed, active layer 0. It is now monthly, which lets
`thaw_depth`'s internal running maximum accumulate the annual thaw.

## Result against observations

```
                              ours     observed
NH permafrost extent        23.2 Mkm2  21.8 (Obu et al. 2019)   +6%
NH annual max thaw depth     1.14 m    0.3-2 m                  in range
seasonal peak                August    Aug-Sep                  correct
active layer, Jan -> Aug     0.025 -> 1.059 m
```

**A comparison error of mine, for the record.** I first reported extent as
"43 % of land, 35.9 Mkm² against an observed 15 %, 22 Mkm²" and treated it as a
model failure. Obu et al.'s figure is **Northern Hemisphere excluding ice
sheets**; ours was global including Antarctica's 12.5 Mkm². Like for like it is
23.2 vs 21.8. Third time this session I compared against the wrong reference.

## Known limitation, unchanged by any of this

IBIS's soil column is **4.0 m** and the annual damping depth in wet soil is
**~3.2 m**, so the bottom boundary still sees ~28 % of the surface annual
amplitude. `active_layer` and `frozen_ground` are seasonal quantities and read
correctly; **`talik` and deep `thaw_depth` are boundary-controlled and should
not be trusted.** Fixing that means 10–15 layers to ~30 m, with a longer
spin-up and a migration of every checkpoint — deliberately not done here.


---

## The defect-0 pattern, and the guard that now covers it (2026-09-04)

`test_no_dead_code.py` catches two shapes: a module with no production caller,
and a config field never read outside its own declaration. A third shape kept
slipping through, and has now been found **six times** in this project:

| instance | shape | found in |
|---|---|---|
| `firefac` | computed, never used | docs/FIRE.md |
| `soilcspin` | declared, read nowhere | docs/GPP.md |
| `stressfac` | loaded into `StomataParams`, never passed to `drystress`, so the hardcoded -5.0 always won | docs/GPP.md |
| `methane_params` | never passed to `run_month_gswp3`, so production used the 458 Tg "global" CH4 config instead of the validated 182 Tg one | docs/METHANE.md 5q |
| `tgpp`/`tnpp`/`tco2root` | state fields never assigned | docs/DEFECTS_FOR_REVIEW.md |
| `upsoil`/`upsoiu` | state fields never assigned (physics live via locals) | below |

**The common shape: the NAME appears.** It is declared, imported, loaded into a
params object, threaded through a signature -- so any test that greps for the
name passes, while the VALUE never reaches the physics. Every one of these was
found by accident, while investigating something else.

`tests/test_no_dead_wiring.py` checks reachability instead:

1. **Config-carrying keyword parameters must be passed by some caller.** Any
   optional kwarg matching `*_params`, `stressfac`, `nsoilcall`, `f_inund`,
   `fire_scheme` or `n_hydro` must be passed explicitly somewhere in
   `ibis/`, `scripts/` or `verify/`. If nothing passes it, its default is the
   only behaviour the model can have -- which is exactly what happened to
   `methane_params` and `stressfac`.
2. **Every state field must be written somewhere.** A field that is only ever
   read stays at its initial value, which for a zero-initialised state means
   the quantity silently reads zero everywhere.

Both carry explicit allowlists with a reason per entry, so suppressing an
entry requires stating why.

### It immediately found a seventh

`upsoil`/`upsoiu` -- canopy water uptake per soil layer. **The physics is
live**: `loops.py:953-954` computes `upsoiu_k`/`upsoil_k` as locals and uses
them to remove water (`:956`) and heat (`:973-974`) from each layer. But the
results are never stored back onto the state, so transpiration extraction is
correct while `state.soil.upsoil` reads zero. Anyone inspecting it expecting a
flux gets nothing. Allowlisted with that distinction recorded, because the fix
is a diagnostic write, not a physics change.
