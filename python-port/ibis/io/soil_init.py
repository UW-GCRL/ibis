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

# ---- Organic end-member (physics.soil_organic) ----
# Lawrence & Slater (2008, Clim. Dyn. 30:145, Table 1), the CLM4/5 treatment: the organic-matter
# fraction of the solids f_om = rho_om / RHO_OM_MAX mixes these linearly into the mineral values.
# rho_om = organic carbon (mass %) x OM_PER_OC x bulk density, from HWSD2
# (scripts/hwsd2_organic_to_nc.py). Only used when an organic-carbon map is supplied.
RHO_OM_MAX = 130.0          # kg m-3, bulk density of pure organic soil (peat)
OM_PER_OC = 1.724           # organic matter per organic carbon (van Bemmelen)
ORGANIC_POROS = 0.9
ORGANIC_BEX = 2.7
ORGANIC_SUCTION = 0.0103    # m (10.3 mm)
ORGANIC_HYDRAUL = 1.0e-4    # m s-1 (0.1 mm s-1)
# Volumetric water at field capacity (-33 kPa = 3.37 m) and wilting (-1500 kPa = 153 m) on the
# organic Campbell curve theta = poros * (suction / psi) ** (1 / b): 0.106 and 0.026.
ORGANIC_SFIELD = ORGANIC_POROS * (ORGANIC_SUCTION / 3.37) ** (1.0 / ORGANIC_BEX)
ORGANIC_SWILT = ORGANIC_POROS * (ORGANIC_SUCTION / 153.0) ** (1.0 / ORGANIC_BEX)
# End-members by decomposition state (Letts et al. 2000, Atmos.-Ocean 38:141, Table 2; poros, b, psi_sat m,
# k_sat m s-1). "ls2008" is the Lawrence & Slater set above (fibric-like), the CLM default. The fibric drainage
# (k_sat 1e-4 m s-1 with b 2.7 at porosity 0.9) made IBIS's hourly soil-water solver produce NaN in ~1 % of
# cells, pure-peat cells first (CALIBRATION.md 44, run o1); the hemic and sapric sets drain 50-1000x slower.
PERC_THRESHOLD = 0.5        # organic fraction above which organic matter percolates (Lawrence & Slater 2008)
PERC_BETA = 0.139
ORGANIC_END_MEMBERS = {
    'ls2008': (ORGANIC_POROS, ORGANIC_BEX, ORGANIC_SUCTION, ORGANIC_HYDRAUL),
    'fibric': (0.93, 2.7, 0.0103, 2.8e-4),
    'hemic':  (0.88, 6.1, 0.0102, 2.0e-6),
    'sapric': (0.83, 12.0, 0.0101, 1.0e-7),
}
ORGANIC_CONDRY = 0.05       # W m-1 K-1, dry organic conductivity (soil_physics.setsoi)
ORGANIC_RHO = 1300.0        # kg m-3 and J kg-1 K-1 of the organic solids, as F77 initial.f:715-724
ORGANIC_CSOI = 1920.0



def organic_ksat(k_min: np.ndarray, k_om: float, fo: np.ndarray,
                 f_threshold: float = PERC_THRESHOLD, beta: float = PERC_BETA) -> np.ndarray:
    """Saturated conductivity of a mineral-organic mix by percolation theory
    (Lawrence & Slater 2008 eq. 6-8; CLM4 Tech. Note 7.4). Below the
    percolation threshold the organic matter is in series with the mineral
    soil (harmonic mean, so a third organic barely changes k_sat); above it a
    connected fraction f_perc = ((f_om - f_thr) / (1 - f_thr)) ** beta * f_om
    conducts in parallel at the organic rate. A linear mix (the first o1 run,
    CALIBRATION.md 44) raised k_sat a hundredfold at f_om = 0.3 and the hourly
    soil-water solver returned NaN in 1 % of the cells.
    """
    fo = np.asarray(fo, dtype=np.float64)
    n = np.clip((fo - f_threshold) / (1.0 - f_threshold), 0.0, 1.0)
    f_perc = np.where(fo > f_threshold, n ** beta * fo, 0.0)
    f_uncon = 1.0 - f_perc
    k_uncon = f_uncon / np.maximum((1.0 - fo) / k_min + (fo - f_perc) / k_om, 1e-30)
    k_uncon = np.where(f_uncon > 0.0, k_uncon, 0.0)
    return (1.0 - f_perc) * k_uncon + f_perc * k_om


def compute_soil_properties(
    sand: np.ndarray,
    clay: np.ndarray,
    orgc: np.ndarray | None = None,
    bulk: np.ndarray | None = None,
    end_member: str = 'ls2008',
) -> dict:
    """Compute soil hydraulic properties from sand/clay percentages.

    Parameters
    ----------
    sand : (npoi, nsoilay) percent sand
    clay : (npoi, nsoilay) percent clay
    orgc, bulk : (npoi, nsoilay), optional
        Organic carbon (% of dry mass) and bulk density (g cm-3) from HWSD2.
    end_member : 'ls2008' (default, Lawrence & Slater 2008), 'fibric', 'hemic' or 'sapric'
        (Letts et al. 2000): the organic hydraulic properties mixed in.
        Given together, the organic-matter fraction of the solids
        f_om = min(1, orgc/100 x OM_PER_OC x bulk x 1000 / RHO_OM_MAX)
        mixes the organic end-member (module constants) linearly into the
        porosity, Campbell b, saturated suction and conductivity, field
        capacity, wilting point, and the solids' density and heat capacity
        (Lawrence & Slater 2008). Without them every value is the F77 texture
        lookup, `forganic` is zero and rhosoi/csoi keep the F77 constant
        correction FORGANIC.

    Returns
    -------
    dict with keys: poros, sfield, swilt, bex, suction, hydraul, albsav, albsan,
                    csoi, rhosoi, z0soi, hsoi, sand, clay, forganic
    """
    npoi, nsoilay = sand.shape

    ms = np.clip(np.round(sand).astype(np.int32), 0, 100)
    mc = np.clip(np.round(clay).astype(np.int32), 0, 100)
    lmin = _TEXTURE_LOOKUP[ms, mc]  # (npoi, nsoilay)

    poros_m   = _POROS[lmin]
    sfield_m  = _SFIELD[lmin]
    swilt_m   = _SWILT[lmin]
    bex_m     = _BEX[lmin]
    suction_m = _SUCTION[lmin]
    hydraul_m = _HYDRAUL[lmin]

    if orgc is not None and bulk is not None:
        oc = np.clip(np.nan_to_num(np.asarray(orgc, dtype=np.float64), nan=0.0), 0.0, 100.0)
        bd = np.asarray(bulk, dtype=np.float64)
        bd = np.where(np.isfinite(bd) & (bd > 0.0), bd, 1.3)
        fo = np.clip(0.01 * oc * OM_PER_OC * bd * 1000.0 / RHO_OM_MAX, 0.0, 1.0)
        o_poros, o_bex, o_suction, o_hydraul = ORGANIC_END_MEMBERS[end_member]
        o_sfield = o_poros * (o_suction / 3.37) ** (1.0 / o_bex)
        o_swilt = o_poros * (o_suction / 153.0) ** (1.0 / o_bex)
        poros   = (1.0 - fo) * poros_m + fo * o_poros
        sfield  = ((1.0 - fo) * sfield_m + fo * o_sfield) / poros
        swilt   = ((1.0 - fo) * swilt_m + fo * o_swilt) / poros
        bex     = (1.0 - fo) * bex_m + fo * o_bex
        suction = (1.0 - fo) * suction_m + fo * o_suction
        hydraul = organic_ksat(hydraul_m, o_hydraul, fo)
        rhosoi  = 2650.0 * (1.0 - fo) + ORGANIC_RHO * fo
        csoi    = 870.0 * (1.0 - fo) + ORGANIC_CSOI * fo
        forganic = fo
    else:
        poros   = poros_m
        sfield  = sfield_m / poros_m
        swilt   = swilt_m / poros_m
        bex     = bex_m
        suction = suction_m
        hydraul = hydraul_m
        # Soil mineral density and heat capacity (Campbell & Norman 1998), matching
        # F77 `initial.f:715-724` with its constant organic fraction FORGANIC.
        rhosoi = np.full((npoi, nsoilay), RHOSOI_VAL)
        csoi   = np.full((npoi, nsoilay), CSOI_VAL)
        forganic = np.zeros((npoi, nsoilay))

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
        'forganic': jnp.array(forganic),
    }
