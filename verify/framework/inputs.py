"""Input generators for verification.

This first version provides three sources, in order of rigor:

  1. fixed_scenarios -- a small reproducible set of physically distinct
     points (carries forward Min's idea of tropical / arctic / etc.).
  2. linspace_sweep  -- dense sweep across one dimension for plots.
  3. random_uniform  -- many random samples inside physical bounds.

Edge cases (T = tmelt, very dry, fully saturated) are produced by
fixed_scenarios; random_uniform covers the bulk distribution. Hypothesis-
based fuzzing comes later if these don't surface enough bugs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float


def fixed_scenarios_T_K() -> np.ndarray:
    """Physically distinct temperature points in Kelvin.

    Hits both branches of esat (liquid/ice), the tmelt boundary, and
    extreme cold/warm. Includes the clamp endpoints of tsatl/tsati
    (t = 273.16 - 60 = 213.16 K and t = 273.16 + 100 = 373.16 K).
    """
    return np.array([
        180.0,             # extreme cold (below ice clamp)
        213.16,            # ice clamp lower edge
        220.0,             # deep winter
        250.0,             # arctic
        273.15,            # freezing - 0.01 (liquid branch active by one ulp)
        273.16,            # tmelt exactly
        273.17,            # tmelt + 0.01
        283.0,             # cool temperate
        293.0,             # 20 C
        300.0,             # warm tropics
        315.0,             # hot desert
        333.0,             # extreme heat
        373.16,            # liquid clamp upper edge
        400.0,             # above liquid clamp
    ], dtype=np.float64)


def linspace_sweep(rng: Range, n: int = 200) -> np.ndarray:
    return np.linspace(rng.lo, rng.hi, n, dtype=np.float64)


def random_uniform(rng: Range, n: int, *, seed: int = 0) -> np.ndarray:
    """n uniform samples in [lo, hi]. Reproducible from seed."""
    return np.random.default_rng(seed).uniform(rng.lo, rng.hi, n).astype(np.float64)
