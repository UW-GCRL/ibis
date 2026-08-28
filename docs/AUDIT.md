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
