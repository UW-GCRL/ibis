"""Koppen-Geiger climate classes as a mask on PFT existence (CALIBRATION.md 44).

IBIS decides which PFTs may exist in a cell from four temperature normals
(`ibis.physics.climate_analysis.existence`): nothing in the envelope says how
dry the cell is, so the dynamic vegetation grows trees on steppes and, with the
calibrated set, onto the tundra (CALIBRATION.md 40, 43). The Koppen-Geiger
classification carries the moisture axis (B = arid, s/w = dry season) and the
treeline (E = polar). This module turns a class map (Beck et al. 2023, Sci. Data
10:724, 1 km, 30-year periods 1901-2099; the 0.5-degree product shipped with it)
into a (npoi, 12) mask that `existence` multiplies into its envelope: the
envelope still decides by temperature, the mask removes the upper-canopy PFTs a
class does not carry. Lower-canopy PFTs (shrubs, grasses) are never masked.

Which trees a class allows (upper-canopy PFT indices, 0-based):
  0 tropical broadleaf evergreen   4 temperate broadleaf cold-deciduous
  1 tropical broadleaf drought-dec 5 boreal conifer evergreen
  2 warm-temperate broadleaf everg 6 boreal broadleaf cold-deciduous
  3 temperate conifer evergreen    7 boreal conifer cold-deciduous (larch)

  A  Af, Am (rainforest, monsoon)        0, 1
     Aw (savanna)                        1          the evergreen rainforest tree is what grew forest on savanna
  B  BWh, BWk (desert), BSk (cold steppe) none      the phantom-tree bands of CALIBRATION.md 13-15
     BSh (hot steppe: Sahel, Deccan)     1          acacia savanna is real woodland
  C  Cs (Mediterranean)                  2, 3, 4
     Cw, Cfa (subtropical, monsoon)      0, 1, 2, 3, 4   the tropical trees' own Tmin > 0 rule decides
     Cfb, Cfc (oceanic)                  2, 3, 4, 5, 6, 7
  D  all continental classes             3, 4, 5, 6, 7
  E  ET (tundra), EF (ice)               none       the treeline
  0  no class (ocean in Beck, land here) unconstrained

The map is prescribed by period (1901-1930 for the spin-up and the early
transient, ..., 1991-2020 for the hindcast); a later step can derive the class
from the model's own normals so it moves with the climate.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

NPFT = 12

KOPPEN_LEGEND = {
    1: "Af", 2: "Am", 3: "Aw", 4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc", 11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc", 17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd", 25: "Dfa", 26: "Dfb", 27: "Dfc", 28: "Dfd",
    29: "ET", 30: "EF",
}

_TROP = (0, 1)
_TROP_DRY = (1,)
_NONE = ()
_MED = (2, 3, 4)
_SUBTROP = (0, 1, 2, 3, 4)
_OCEANIC = (2, 3, 4, 5, 6, 7)
_CONT = (3, 4, 5, 6, 7)

#: Upper-canopy PFTs allowed per Koppen class.
KOPPEN_TREES = {
    1: _TROP, 2: _TROP, 3: _TROP_DRY,
    4: _NONE, 5: _NONE, 6: _TROP_DRY, 7: _NONE,
    8: _MED, 9: _MED, 10: _MED,
    11: _SUBTROP, 12: _SUBTROP, 13: _SUBTROP, 14: _SUBTROP,
    15: _OCEANIC, 16: _OCEANIC,
    17: _CONT, 18: _CONT, 19: _CONT, 20: _CONT,
    21: _CONT, 22: _CONT, 23: _CONT, 24: _CONT,
    25: _CONT, 26: _CONT, 27: _CONT, 28: _CONT,
    29: _NONE, 30: _NONE,
}


def _build_table() -> np.ndarray:
    t = np.ones((31, NPFT), dtype=np.float32)       # class 0: unconstrained
    for c, trees in KOPPEN_TREES.items():
        t[c, :8] = 0.0
        for j in trees:
            t[c, j] = 1.0
    return t


#: (31, 12): row = Koppen class (0 = none), column = PFT; 1 = may exist.
MASK_TABLE = _build_table()


def koppen_exist_mask(classes) -> jnp.ndarray:
    """(npoi,) class codes -> (npoi, 12) existence mask."""
    c = np.clip(np.asarray(classes, dtype=np.int64), 0, 30)
    return jnp.asarray(MASK_TABLE[c])


def load_koppen(path: str, latindex, lonindex) -> dict:
    """Read the npz of scripts/koppen_prep.py at the land points.

    Returns classes (nperiod, npoi) uint8, period_start, period_end (nperiod,).
    `latindex` are the GSWP3 row indices (north to south), the same convention
    as the cropland map, so the (360, 720) north-to-south array is indexed
    directly.
    """
    d = np.load(path)
    cl = d["classes"]
    if cl.shape[1:] != (360, 720):
        raise ValueError(f"koppen classes shape {cl.shape}, expected (nperiod, 360, 720)")
    return {
        "classes": cl[:, latindex, lonindex],
        "period_start": np.asarray(d["period_start"], dtype=int),
        "period_end": np.asarray(d["period_end"], dtype=int),
    }


def period_index(year: int, period_start, period_end) -> int:
    """Period holding `year`; years before the first period take the first,
    after the last take the last, gaps take the period that ended last."""
    ps = np.asarray(period_start); pe = np.asarray(period_end)
    if year <= ps[0]:
        return 0
    if year >= ps[-1]:
        return len(ps) - 1
    i = int(np.searchsorted(ps, year, side="right") - 1)
    return i


def describe(classes, mask=None) -> str:
    """One line per class with the land-point count, for the run log."""
    c = np.asarray(classes).astype(int)
    parts = []
    for k in range(31):
        n = int((c == k).sum())
        if n:
            parts.append(f"{KOPPEN_LEGEND.get(k, 'none')}:{n}")
    s = " ".join(parts)
    if mask is not None:
        m = np.asarray(mask)
        s += f" | land points with no tree allowed: {int((m[:, :8].sum(1) == 0).sum())} of {m.shape[0]}"
    return s
