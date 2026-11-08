# IBIS 3.0 Technical Note

**Early edition · updated 7 November 2026.** This dated update records the completed refits, the first full protocol
run of the revised model and its evaluation, the equilibrium analysis of the spin-up and the production protocol now
running. It retains the published release benchmark tables. Chapter 16 records the completed work and what remains;
the production run's results are not yet available. The model code on GitHub main remains the 30 October release;
this publication updates documentation only.

This note describes the Integrated Biosphere Simulator, version 3.0: the equations the model solves, their
parameters and the data the model reads. Its chapter structure follows the CLM5 Technical Note (Lawrence et al.,
2018). The calibration record is [`../CALIBRATION.md`](../CALIBRATION.md); the release state is
[`../RELEASE_3.0.md`](../RELEASE_3.0.md).

The documentation review of 2 November 2026 uses code at `c6698fd`. Chapters distinguish the published release,
the current starting parameters and exploratory results. In that reviewed implementation, the selected temperature-sum floors are 700 for the boreal
types and 975 for temperate conifer; fire counts eligible steps throughout the day; and the global driver requests
deterministic GPU reductions by default. These settings are not assigned retrospectively to older experiments.
Remaining scientific decisions, refits and final validation are described in Chapter 16.

**Development progress, 7 November 2026.** All tower refits are complete: natural-tower GPP and ET (356 fits), crops
(46), day and night NEE (358) and wetland methane (48). Their type and band medians form the assembled parameter set
([Chapter 16](16_calibration.md#16145-development-progress-7-november-2026)). A first full protocol run with that set
(5-6 November) was scored against the same products as the release: it lost most of the subarctic conifer cover and
raised wetland methane above the budget range. Controlled spin-ups traced the conifer loss to the refitted values of
the two subarctic tree types, which the production set returns to their base values; the tropical methane band is
under review. A pool-level analysis found the release-protocol spin-up only about 78 % of the way to soil-carbon
equilibrium, so the production run uses a chained spin-up of more than 1,290 years ([Chapter 12](12_soil_biogeochemistry.md#127-spin-up)),
in progress; its ramp to the normal rate exposed a fire-driven dependence of sparse-cell vegetation on the acceleration,
and a corrective stage is under decision. The canopy water-accounting correction is part of the production source
([Chapter 7](07_soil_hydrology.md)). The release candidate retains the G06 timing threshold of
[Chapter 10](10_phenology.md) and leaves the crop-allocation override unset ([Chapter 15](15_cropland.md)). These
development results do not change the released benchmark tables or the model code published on GitHub main.

The chapters describe the equations, inputs and parameter choices in detail. Implementation history and unresolved
questions appear in each chapter's closing Notes for developers.

## Chapters

| Chapter | Title | Edition |
|---|---|---|
| 1 | [Introduction](01_introduction.md) | early edition; updated 7 November |
| 2 | [Surface Characterization and Plant Functional Types](02_surface_characterization.md) | early edition |
| 3 | [Atmospheric Forcing and the Diurnal Weather Generator](03_forcing_weather_generator.md) | early edition |
| 4 | [Canopy Radiative Transfer](04_radiation.md) | early edition |
| 5 | [Surface Energy, Momentum and Water Vapour Fluxes; Canopy Interception](05_surface_fluxes.md) | early edition; status updated 7 November |
| 6 | [Soil and Bedrock Temperature](06_soil_bedrock_temperature.md) | early edition |
| 7 | [Soil Hydrology](07_soil_hydrology.md) | early edition; updated 7 November |
| 8 | [Snow](08_snow.md) | early edition |
| 9 | [Photosynthesis and Stomatal Conductance](09_photosynthesis_stomata.md) | early edition; status updated 7 November |
| 10 | [Phenology](10_phenology.md) | early edition; status updated 7 November |
| 11 | [Autotrophic Respiration, Allocation and Vegetation Dynamics](11_vegetation_dynamics.md) | early edition; updated 7 November |
| 12 | [Soil Biogeochemistry and Spin-up](12_soil_biogeochemistry.md) | early edition; spin-up updated 7 November |
| 13 | [Fire](13_fire.md) | early edition; updated 7 November |
| 14 | [Wetland Methane](14_methane.md) | early edition; updated 7 November |
| 15 | [Cropland](15_cropland.md) | early edition; status updated 7 November |
| 16 | [Calibration](16_calibration.md) | early edition; progress updated 7 November |
| 17 | [Numerical Implementation and Verification against IBIS 2.6b4](17_numerics_verification.md) | early edition; updated 7 November |

Chapter 5 continues in [Part 2: energy balance and evapotranspiration](05_surface_fluxes_part2.md).

## Conventions

Sections, equations, tables and figures are numbered by chapter (section 4.2, equation (4.7), Table 4.1). Every symbol
is defined, with its units, in the sentence that follows the equation that introduces it. Symbols that name model
variables give the variable's name in the code in the form `module.variable` where the two differ. Parameter values
are those of the release parameter file, `python-port/configs/params/params_31s_ch4h.yaml`, unless a chapter states
otherwise. Later starting parameters are identified explicitly. Figures use the documented code, input data and
run or verification outputs, with their provenance stated in captions and developer notes. They are generated by
[`figures/make_figures.py`](figures/make_figures.py) (Chapters 1-3) and `figures/make_chNN.py` (Chapter NN). Each chapter
has undergone content and figure review, with confirmed corrections applied. This early edition also includes a
separate language edit. Completion of these reviews does not imply that the pending scientific validation is complete.

The figure PNGs are included with this documentation edition. Reproducing them requires the generation scripts and
the corresponding source data described in the chapters.

## References

Lawrence, D. M., et al., 2018: Technical Description of version 5.0 of the Community Land Model (CLM). National Center
for Atmospheric Research, Boulder, CO.
