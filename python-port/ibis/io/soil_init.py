"""Soil property initialization from texture data.

Ported from initial.f90: inisoil + textcls + inpoly.
Computes porosity, field capacity, wilting point, Campbell b exponent,
suction, and hydraulic conductivity from sand/clay percentages using
USDA textural classification and Rawls et al. (1992) lookup tables.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---- Layer thicknesses (m) from params.soi ----
HSOI_LAYERS = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])


# ---- Rawls et al. (1992) soil property lookup tables ----
# Index 0..10 corresponds to USDA classes: Sand, Loamy Sand, Sandy Loam,
# Loam, Silt Loam, Sandy Clay Loam, Clay Loam, Silty Clay Loam,
# Sandy Clay, Silty Clay, Clay
_POROS   = np.array([0.437, 0.437, 0.453, 0.463, 0.501, 0.398, 0.464, 0.471, 0.430, 0.479, 0.475])
_SFIELD  = np.array([0.091, 0.125, 0.207, 0.270, 0.330, 0.255, 0.318, 0.366, 0.339, 0.387, 0.396])
_SWILT   = np.array([0.033, 0.055, 0.095, 0.117, 0.133, 0.148, 0.197, 0.208, 0.239, 0.250, 0.272])
_BEX     = np.array([1.7,   2.1,   3.1,   4.5,   4.7,   4.0,   5.2,   6.6,   6.0,   7.9,   7.6])
_SUCTION = np.array([0.07,  0.09,  0.15,  0.11,  0.21,  0.28,  0.26,  0.33,  0.29,  0.34,  0.37])
_HYDRAUL = np.array([5.833e-5, 1.6972e-5, 7.1944e-6, 3.6667e-6, 1.8889e-6,
                     1.1944e-6, 6.3889e-7, 4.1667e-7, 3.3333e-7, 2.5e-7, 1.6667e-7])

# ---- USDA textural triangle polygons (sand%, clay%) ----
# Each polygon: array of (npoints, 2) with columns [sand, clay]
_SANDY           = np.array([[85,0],[90,10],[100,0]])
_LOAMY_SAND      = np.array([[70,0],[85,15],[90,10],[85,0]])
_SANDY_LOAM      = np.array([[50,0],[43,7],[52,7],[52,20],[80,20],[85,15],[70,0]])
_LOAM            = np.array([[43,7],[23,27],[45,27],[52,20],[52,7]])
_SILTY_LOAM      = np.array([[0,0],[0,27],[23,27],[50,0]])
_SANDY_CLAY_LOAM = np.array([[52,20],[45,27],[45,35],[65,35],[80,20]])
_CLAY_LOAM       = np.array([[20,27],[20,40],[45,40],[45,27]])
_SILTY_CLAY_LOAM = np.array([[0,27],[0,40],[20,40],[20,27]])
_SANDY_CLAY      = np.array([[45,35],[45,55],[65,35]])
_SILTY_CLAY      = np.array([[0,40],[0,60],[20,40]])
_CLAYEY          = np.array([[20,40],[0,60],[0,100],[45,55],[45,40]])

_POLYGONS = [
    _SANDY, _LOAMY_SAND, _SANDY_LOAM, _LOAM, _SILTY_LOAM,
    _SANDY_CLAY_LOAM, _CLAY_LOAM, _SILTY_CLAY_LOAM,
    _SANDY_CLAY, _SILTY_CLAY, _CLAYEY,
]


def _point_in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    """Test if point (x, y) is inside polygon using ray-casting."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _build_texture_lookup() -> np.ndarray:
    """Build a 101×101 lookup table mapping (sand%, clay%) → class index.

    Runs once at import time (~10k iterations). All subsequent calls use
    O(1) numpy integer indexing instead of per-point polygon tests.
    """
    lut = np.full((101, 101), 4, dtype=np.int8)  # default: silt loam
    for s in range(101):
        for c in range(101):
            if s + c > 100:
                continue
            if s == 0 and c == 0:
                continue  # keep default
            for idx, poly in enumerate(_POLYGONS):
                if _point_in_polygon(poly, float(s), float(c)):
                    lut[s, c] = idx
                    break
    return lut


_TEXTURE_LOOKUP = _build_texture_lookup()


#: Organic fraction of the soil solid phase. A placeholder in IBIS 2.6b4 too
#: (`initial.f:711`), not a map -- but it is the single place both the mineral
#: density and the specific heat are derived from, so they cannot drift apart.
FORGANIC = 0.010

#: Density of the soil solid phase, kg m-3 (Campbell & Norman 1998).
RHOSOI_VAL = 2650.0 * (1.0 - FORGANIC) + 1300.0 * FORGANIC      # 2636.5

#: Specific heat of the soil solid phase, J kg-1 K-1 (Campbell & Norman 1998).
CSOI_VAL = 870.0 * (1.0 - FORGANIC) + 1920.0 * FORGANIC          # 880.5


def compute_soil_properties(
    sand: np.ndarray,
    clay: np.ndarray,
) -> dict:
    """Compute soil hydraulic properties from sand/clay percentages.

    Parameters
    ----------
    sand : (npoi, nsoilay) percent sand
    clay : (npoi, nsoilay) percent clay

    Returns
    -------
    dict with keys: poros, sfield, swilt, bex, suction, hydraul, albsav, albsan,
                    csoi, rhosoi, z0soi, hsoi
    """
    npoi, nsoilay = sand.shape
    forganic = FORGANIC

    ms = np.clip(np.round(sand).astype(np.int32), 0, 100)
    mc = np.clip(np.round(clay).astype(np.int32), 0, 100)
    lmin = _TEXTURE_LOOKUP[ms, mc]  # (npoi, nsoilay)

    poros   = _POROS[lmin]
    sfield  = _SFIELD[lmin] / _POROS[lmin]
    swilt   = _SWILT[lmin] / _POROS[lmin]
    bex     = _BEX[lmin]
    suction = _SUCTION[lmin]
    hydraul = _HYDRAUL[lmin]

    # Soil mineral density and heat capacity (Campbell & Norman 1998), matching
    # F77 `initial.f:715-724`. Uniform today because `forganic` is a placeholder
    # constant, but kept per-layer so a real organic-fraction map can be dropped
    # in without touching the physics -- which the methane work will want, since
    # peat is exactly where this correction stops being negligible.
    rhosoi = np.full((npoi, nsoilay), RHOSOI_VAL)
    csoi   = np.full((npoi, nsoilay), CSOI_VAL)

    # Surface albedo from top-layer texture
    fsand_top = 0.01 * np.round(sand[:, 0]).astype(int)
    fclay_top = 0.01 * np.round(clay[:, 0]).astype(int)
    fsilt_top = 1.0 - fsand_top - fclay_top
    albsav = fsand_top * 0.120 + fsilt_top * 0.085 + fclay_top * 0.050
    albsan = 2.0 * albsav

    # Roughness
    z0soi = np.full(npoi, 0.005)

    # Layer thicknesses
    hsoi_lay = HSOI_LAYERS[:nsoilay]
    # Interface thicknesses: hsoi has nsoilay+1 entries
    # hsoi[0..nsoilay-1] = layer thicknesses, hsoi[nsoilay] = 0 (bottom)
    hsoi = np.zeros(nsoilay + 1)
    hsoi[:nsoilay] = hsoi_lay

    return {
        'poros': jnp.array(poros),
        'sfield': jnp.array(sfield),
        'swilt': jnp.array(swilt),
        'bex': jnp.array(bex),
        'suction': jnp.array(suction),
        'hydraul': jnp.array(hydraul),
        'csoi': jnp.array(csoi),
        'rhosoi': jnp.array(rhosoi),
        'albsav': jnp.array(albsav),
        'albsan': jnp.array(albsan),
        'z0soi': jnp.array(z0soi),
        'hsoi': jnp.array(hsoi),
        'sand': jnp.array(sand),
        'clay': jnp.array(clay),
    }
