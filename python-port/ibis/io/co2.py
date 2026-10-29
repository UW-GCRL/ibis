"""Historical atmospheric CO2 for transient runs.

`run_global` sets `atm.co2conc` once, at state creation, from `co2init` --
so every run this project has done so far, including the 300-year `both_fix`
spin-up and Min's `calibration_transient`, held CO2 fixed. For a spin-up that
is correct. For a historical run validated against eddy-covariance towers it
is not: 1901-2010 CO2 rose from ~296 to ~390 ppm, and photosynthesis responds
directly to it.

**Provenance, stated plainly.** The knots below are published global annual
means entered by hand: Mauna Loa (Keeling) from 1959, Law Dome / firn-air
before it, as compiled for the CMIP6 historical forcing (Meinshausen et al.,
GMD 10, 2057, 2017). Values between knots are linearly interpolated. This is
a coarse stand-in for reading the CMIP6 forcing file, adequate for a decadal
mean but NOT for interannual attribution -- it carries no seasonal cycle and
no year-to-year growth-rate variability. `co2_series_from_file` is provided so
the table can be replaced by real data without touching callers.

Outside the table CO2 is clamped to the end values rather than extrapolated. 2021-2024 (NOAA GML Mauna Loa annual
means) were added on 2026-10-13 for the IBIS 3.0 tower windows that run past 2020 (CALIBRATION.md 45); the run
years up to 2020 are unchanged.
"""
from __future__ import annotations

import numpy as np

#: (year, ppm) global annual-mean CO2 knots. See module docstring.
CO2_KNOTS: tuple[tuple[int, float], ...] = (
    (1850, 285.2), (1860, 286.1), (1870, 287.4), (1880, 289.0),
    (1890, 293.4), (1900, 295.7), (1910, 299.2), (1920, 303.0),
    (1930, 306.3), (1940, 310.5), (1950, 311.3), (1959, 315.98),
    (1960, 316.91), (1965, 320.04), (1970, 325.68), (1975, 331.13),
    (1980, 338.76), (1985, 346.12), (1990, 354.39), (1995, 360.82),
    (2000, 369.55), (2005, 379.80), (2010, 389.85), (2015, 400.83),
    (2020, 414.24), (2021, 416.41), (2022, 418.53), (2023, 421.08), (2024, 424.61),
)

PREINDUSTRIAL_PPM = 285.2


def co2_ppm(year: int | float) -> float:
    """Global annual-mean CO2 in ppm, linearly interpolated between knots."""
    yr = np.array([k[0] for k in CO2_KNOTS], dtype=np.float64)
    pp = np.array([k[1] for k in CO2_KNOTS], dtype=np.float64)
    return float(np.interp(float(year), yr, pp))


def co2_molmol(year: int | float) -> float:
    """CO2 as a mole fraction, the unit `atm.co2conc` carries."""
    return co2_ppm(year) * 1.0e-6


def co2_series_from_file(path: str) -> dict[int, float]:
    """Read a two-column `year,ppm` CSV, so the table can be replaced."""
    a = np.genfromtxt(path, delimiter=",", names=True)
    names = a.dtype.names
    ycol, ccol = names[0], names[1]
    return {int(y): float(c) for y, c in zip(a[ycol], a[ccol])}


def apply_co2(state, year: int | float, series: dict[int, float] | None = None):
    """Return `state` with `atm.co2conc` set to `year`'s value.

    Kept as a whole-state operation rather than a parameter so that a caller
    which forgets it fails loudly in comparison against observations, not
    silently at 350 ppm.
    """
    import equinox as eqx

    if series is not None:
        yrs = np.array(sorted(series))
        val = float(series[int(yrs[np.argmin(np.abs(yrs - year))])]) * 1.0e-6
    else:
        val = co2_molmol(year)
    return eqx.tree_at(lambda s: s.atm.co2conc, state, val)
