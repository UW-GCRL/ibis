"""Dynamic wetland (inundated) area from the model's own soil water (docs/METHANE.md 7; Min's point 4).

Off by default. When `physics.wetland_dynamic` is on, the inundated fraction the methane scheme sees each month is

    f = clip(f_clim[m] * (1 + beta * A), f_min, f_max),      A = (W - mu[m]) / sd[m],     A = 0 where the top layer is frozen

W is the liquid soil water of the top metre (the regression study's Wrz: sum over the top ROOT_LAYERS layers of
poros * wsoi * hsoi, in m), mu and sd its monthly climatology per cell from a reference hindcast
(verify/methane/wetland_clim_prep.py), f_clim the WAD2M monthly climatology, f_min / f_max the cell's WAD2M range over
the record (ibis.io.wad2m.read_wad2m_bounds), beta per band (yaml `wetland_area:`; the 2026-10-16 regression gave
tropical 0.13, boreal 0.06 per standard deviation of W). A frozen top layer suppresses the anomaly, not the climatology
(WAD2M's own frozen months are already 0 in f_clim), so beta = 0 reproduces the climatology path exactly. Host-side
numpy: the fraction is a forcing of the month, not a state.
"""
from __future__ import annotations
import numpy as np

#: layers of the top metre in IBIS's 6-layer column (0.10, 0.25, 0.25, 0.40, 1.0, 2.0 m)
ROOT_LAYERS = 4
T_FREEZE = 273.16


def root_zone_water(wsoi, poros, hsoi, tsoi=None, nlay: int = ROOT_LAYERS) -> np.ndarray:
    """Liquid water of the top `nlay` layers, m: sum(poros * wsoi * hsoi); a frozen layer (tsoi <= 0 C) counts 0."""
    w = np.asarray(wsoi, np.float64)[:, :nlay] * np.asarray(poros, np.float64)[:, :nlay] * np.asarray(hsoi, np.float64)[None, :nlay]
    if tsoi is not None:
        w = np.where(np.asarray(tsoi, np.float64)[:, :nlay] > T_FREEZE, w, 0.0)
    return w.sum(axis=1)


def dynamic_fraction(f_clim, W, mu, sd, beta, f_min, f_max, frozen=None) -> np.ndarray:
    """The month's inundated fraction, (npoi,), from the climatology `f_clim` (npoi,), the storage `W` and its monthly
    climatology `mu`, `sd` (npoi,), `beta` (scalar or (npoi,)), the cell bounds, and an optional frozen mask (anomaly 0)."""
    f_clim, W, mu, sd = (np.asarray(x, np.float64) for x in (f_clim, W, mu, sd))
    A = np.where(sd > 1e-9, (W - mu) / np.maximum(sd, 1e-9), 0.0)
    if frozen is not None:
        A = np.where(np.asarray(frozen, bool), 0.0, A)
    f = np.clip(f_clim * (1.0 + np.asarray(beta, np.float64) * A), np.asarray(f_min, np.float64), np.asarray(f_max, np.float64))
    return np.clip(f, 0.0, 1.0)
