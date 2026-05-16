"""Calendar utilities for IBIS-JAX.

Month lengths, leap year logic, Julian day computation.
All functions are pure and JIT-compatible.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float, Array

# Days per month (non-leap year), 0-indexed: Jan=0
NDAYPM = jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=jnp.int32)
NDAYPM_LEAP = jnp.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=jnp.int32)

# Cumulative days before each month (non-leap), 0-indexed
# Jan=0, Feb=31, Mar=59, ...
_CUM_DAYS = jnp.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=jnp.int32)
_CUM_DAYS_LEAP = jnp.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335], dtype=jnp.int32)


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year (Gregorian)."""
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def days_in_month(imonth: int, leap: bool = False) -> int:
    """Return number of days in month (0-indexed: Jan=0)."""
    table = NDAYPM_LEAP if leap else NDAYPM
    return int(table[imonth])


def days_in_year(leap: bool = False) -> int:
    """Return 365 or 366."""
    return 366 if leap else 365


def julian_day(imonth, iday, leap: bool = False):
    """Return Julian day (1-based) from 0-indexed month and 0-indexed day.

    imonth: 0=Jan, 11=Dec (may be a JAX tracer inside lax.scan)
    iday: 0-based day within month (may be a JAX tracer)
    Returns: 1-based Julian day (1..365 or 366), as int or jnp scalar
    """
    cum = _CUM_DAYS_LEAP if leap else _CUM_DAYS
    return cum[imonth] + iday + 1


def timesteps_per_day(dtime: float) -> int:
    """Number of physics timesteps per day."""
    return int(86400.0 / dtime)
