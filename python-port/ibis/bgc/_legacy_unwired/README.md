# Unwired biogeochemistry components

Moved here 2026-08-28 by docs/AUDIT.md, which found them reachable only from
tests. Each had a config flag (`enable_bvoc`, `enable_soil_n2o`,
`enable_wetlands`) that was read nowhere.

| module | why it is here |
|---|---|
| `wetlands.py` | The full prognostic CH₄ scheme — TOPMODEL wetland fraction, methanogenesis, methanotrophy, ebullition, plant transport, diffusion. **Superseded.** docs/METHANE.md §2.2 already said "quarantine, do not extend", and §5f *removed* ebullition and transport from the plan outright on the evidence of Chadburn et al. (2020): substituting one `exp(−τz)` scalar for all of CLM's equivalent machinery reproduced it at R² 0.89–0.97. `ibis/bgc/methane.py` is the live scheme. |
| `nee.py` | Aggregates NEE_CO2 / NEE_CH4 / NEE_N2O from `wetland_CH4`, `soil_N2O` and `fire_*`. An aggregator for an architecture that was never assembled — and IBIS already accumulates `ayneetot` natively in `vegetation_dynamics`. |
| `bvoc.py` | Biogenic VOC emissions. On no current path. |
| `soil_gases.py` | Soil N₂O / NO. On no current path; would need the nitrogen cycle first. |

**`permafrost.py` deliberately stayed in `ibis/bgc/`.** It is unwired too, but it
is on the roadmap (active-layer depth for Arctic CH₄), so quarantining it would
misrepresent the intent. It remains in `KNOWN_DEAD_MODULES` until wired.

To revive any of these: move the file back, restore its `ibis/bgc/__init__.py`
export, wire the config flag into the timestep, and drop the entry from
`KNOWN_DEAD_MODULES` in `tests/test_no_dead_code.py`.
