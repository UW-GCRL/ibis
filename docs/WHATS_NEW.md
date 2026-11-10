# What changed from IBIS 2.6b4

The components of IBIS 3.0 against the original Fortran model (moved here from the README on 10 November
2026, with three corrections: the time step, the fire terms that are off in the release, and burned area in the
calibration row). The science of each is in the [technical note](technote/README.md).

| Component | IBIS 2.6b4 | IBIS 3.0 | Kind |
|---|---|---|---|
| Numerics | Fortran 77, serial, hourly | JAX on GPU, float32 with float64 verification, an hourly time step as in 2.6b4 (the time step's gradients not yet finite) | port |
| Reference and numerical corrections | real*4 physiology constants; bare-cell NaN; wet-day chain; the soil-water pump of `soilh2o` | corrections and scope documented in Chapter 17; the old wilting-point overflow was a synthetic-soil test artifact, not an established Fortran soil defect | corrections |
| Forcing in the port | — | surface pressure, phenology clock, solar latitude, grid duplication, normals on restart, accumulators | port fixes |
| Canopy photosynthesis | one big leaf, the stems' area counted in the leaves' capacity, vmax by existence priority | sunlit/shaded two big leaves, capacity e^(−Kn x), vmax mixed by leaf area, acclimation | new |
| Phenology | hard-coded thresholds, temperature-only senescence, inert drought trigger | parameters; photoperiod senescence; soil-stress drought trigger; absolute cold threshold | new |
| Plant types | 12 PFTs, one parameter set each | 40 Köppen-zone types in the 12 slots, each square taking its zone's types; a Köppen mask on existence | new |
| Cover and establishment | linear grass cover, 25 % floors, no water limit on trees | saturating cover, floors as parameters, a water-limited maximum tree cover per zone | new |
| Fire | annual-mean dryness, never switched on in the port | fire-season length, VPD × density danger, fine fuel and human terms (both off in the release), per-type mortality, closed-canopy damping, emissions as a diagnostic (no feedback), combustion completeness | new |
| Methane | none | one-scalar wetland module evaluated every time step; wetland area = WAD2M seasonal climatology + a learned year-to-year anomaly from the model's own water; per-band parameters fitted at towers at the hourly cadence | new |
| Cropland | none (crops become forest) | managed grass on a sub-grid share: harvest, crop capacity factor, calibrated at cropland towers | new |
| Soil column | 4 m, zero flux | +8 m bedrock, permafrost diagnostic | new |
| Respiration and allocation | hard-coded constants | per-type parameters, NEE by day and night, ET components; slow/passive soil turnover per zone group | parameters |
| Calibration | none shipped | tower-first: 356 towers one by one, then one set per type; 48 Fluxnet-Methane_V2.0 wetland towers and 16 cropland towers; then cover, wood, soil carbon and burned area (GFED5 zone means) against the gridded products inside the full protocol | calibration |

Most new processes sit behind a switch that is off by default; the details of the fire scheme are on by default (technote Table 13.1) and some additions act through parameter values (technote 10.6). The documents are listed in the [README](../README.md#documentation).
