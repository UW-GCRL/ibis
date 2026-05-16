"""Vegetation initialization from vegetation type map.

Ported from initial.f: inisurf vegetation cold-start logic.
Maps the 15-class vegetation type map to 12 PFT LAI values,
then computes canopy structure (fu, fl, lai, sai, ztop, zbot)
and root distribution (froot).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


# ---- plai_init lookup table (4 groups x 15 veg types) ----
# Rows: [evgreen_tree, decid_tree, shrub, grass]
# Columns: veg types 1-15 (0-indexed: 0-14)
# Types 13-14 (tundra, desert, polar desert) have no vegetation data
# in the original Fortran params.veg (only 12 types listed), so we
# pad with zeros for types 13-15.
PLAI_INIT = np.array([
    # evgr   deci  shrub  grass
    [5.00,  1.00,  0.25,  0.25],   # 1  trop evgr forest
    [1.00,  5.00,  0.25,  0.25],   # 2  trop decid forest
    [4.00,  1.00,  0.25,  0.25],   # 3  temp evgr broadleaf
    [3.00,  1.00,  0.25,  0.25],   # 4  temp evgr conifer
    [1.00,  3.00,  0.25,  0.25],   # 5  temp decid forest
    [3.00,  1.00,  0.25,  0.25],   # 6  boreal evgr forest
    [1.00,  3.00,  0.25,  0.25],   # 7  boreal decid forest
    [2.00,  2.00,  0.25,  0.25],   # 8  mixed forest
    [0.50,  1.00,  0.50,  2.00],   # 9  savanna
    [0.25,  0.25,  0.50,  2.50],   # 10 grassland
    [0.10,  0.10,  1.00,  0.50],   # 11 dense shrubland
    [0.00,  0.00,  0.25,  0.25],   # 12 open shrubland
    [0.00,  0.00,  0.10,  0.10],   # 13 tundra
    [0.00,  0.00,  0.00,  0.00],   # 14 desert
    [0.00,  0.00,  0.00,  0.00],   # 15 polar desert/rock/ice
], dtype=np.float32)

# PFT groupings
IEVGR = 4    # number of evergreen tree PFTs (1,3,4,6)
IDECI = 4    # number of deciduous tree PFTs (2,5,7,8)
ISHRUB = 2   # number of shrub PFTs (9,10)
IGRASS = 2   # number of grass PFTs (11,12)

# PFT-to-group mapping (0-indexed PFT → group index in plai_init)
# PFTs 0-7 are trees (upper canopy), PFTs 8-11 are lower canopy
_PFT_GROUP = np.array([0, 1, 0, 0, 1, 0, 1, 1, 2, 2, 3, 3])
_PFT_COUNT = np.array([
    IEVGR, IDECI, IEVGR, IEVGR, IDECI, IEVGR, IDECI, IDECI,
    ISHRUB, ISHRUB, IGRASS, IGRASS,
], dtype=np.float32)

# specla (specific leaf area, m2/kg) per PFT — from params.can
SPECLA = np.array([
    25.0, 25.0, 25.0, 12.5, 25.0, 12.5, 25.0, 25.0,
    12.5, 25.0, 20.0, 20.0,
], dtype=np.float32)

# Carbon allocation fractions per PFT — from params.can
ALEAF = np.array([0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
                   0.45, 0.45, 0.45, 0.45], dtype=np.float32)
AROOT = np.array([0.20, 0.20, 0.20, 0.40, 0.20, 0.40, 0.20, 0.20,
                   0.40, 0.35, 0.55, 0.55], dtype=np.float32)
AWOOD = np.array([0.50, 0.50, 0.50, 0.30, 0.50, 0.30, 0.50, 0.50,
                   0.15, 0.20, 0.00, 0.00], dtype=np.float32)

XMINLAI = 0.01
SAPFRAC_INIT = 0.1
WOODNORM = 7.5
BETA1 = 0.950   # root profile param, lower canopy (grass/shrub)
BETA2 = 0.975   # root profile param, upper canopy (trees)

# Layer thicknesses (m) — must match soil_init.py
HSOI_LAYERS = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])


def compute_root_distribution(nsoilay: int = 6) -> np.ndarray:
    """Compute root fraction per soil layer for lower (col 0) and upper (col 1) canopy.

    Returns (nsoilay, 2) array.
    """
    hsoi = HSOI_LAYERS[:nsoilay]

    # Cumulative depth in cm
    depth = np.cumsum(hsoi) * 100.0
    totdepth = depth[-1]

    frootnorm1 = 1.0 - BETA1 ** totdepth
    frootnorm2 = 1.0 - BETA2 ** totdepth

    froot = np.zeros((nsoilay, 2))
    for k in range(nsoilay):
        if k == 0:
            froot[k, 0] = 1.0 - BETA1 ** depth[k]
            froot[k, 1] = 1.0 - BETA2 ** depth[k]
        else:
            froot[k, 0] = (1.0 - BETA1 ** depth[k]) - (1.0 - BETA1 ** depth[k - 1])
            froot[k, 1] = (1.0 - BETA2 ** depth[k]) - (1.0 - BETA2 ** depth[k - 1])
        froot[k, 0] /= frootnorm1
        froot[k, 1] /= frootnorm2

    return froot


def initialize_vegetation(
    vegtype: np.ndarray,
    tw: np.ndarray | None = None,
    npft: int = 12,
    nsoilay: int = 6,
) -> dict:
    """Initialize vegetation state from vegetation type map.

    Parameters
    ----------
    vegtype : (npoi,) int or float
        Vegetation type classification (1-15). Values <=0 or >15 treated as
        bare ground.
    tw : (npoi,) float, optional
        Warmest month temperature (C). Used for C3/C4 grass split in
        savanna/grassland. If None, defaults to 25C (tropical assumption).
    npft : int
        Number of plant functional types (default 12).
    nsoilay : int
        Number of soil layers (default 6).

    Returns
    -------
    dict with keys:
        plai     : (npoi, npft) per-PFT leaf area index
        cbiol    : (npoi, npft) leaf carbon
        cbior    : (npoi, npft) root carbon
        cbiow    : (npoi, npft) wood carbon
        biomass  : (npoi, npft) total biomass
        exist    : (npoi, npft) PFT existence (all 1.0 for cold start)
        vegtype0 : (npoi,) vegetation type
        sapfrac  : (npoi,) sapwood fraction
        fu       : (npoi,) upper canopy fractional cover
        fl       : (npoi,) lower canopy fractional cover
        lai      : (npoi, 2) canopy LAI [lower, upper]
        sai      : (npoi, 2) canopy SAI [lower, upper]
        ztop     : (npoi, 2) canopy top height [lower, upper]
        zbot     : (npoi, 2) canopy bottom height [lower, upper]
        totlaiu  : (npoi,) total upper canopy LAI
        totlail  : (npoi,) total lower canopy LAI
        totbiou  : (npoi,) total upper canopy biomass
        totbiol  : (npoi,) total lower canopy biomass
        froot    : (nsoilay, 2) root fraction per layer
        specla   : (npft,) specific leaf area
        aleaf    : (npft,) leaf allocation fraction
        aroot    : (npft,) root allocation fraction
        awood    : (npft,) wood allocation fraction
    """
    vegtype = np.asarray(vegtype)
    npoi = len(vegtype)

    if tw is None:
        tw = np.full(npoi, 25.0)
    else:
        tw = np.asarray(tw)

    # Round vegtype to nearest integer (1-15)
    inveg = np.clip(np.round(vegtype).astype(int), 0, 15)

    # Existence: all 1.0 for cold start (climatic constraints applied later)
    exist = np.ones((npoi, npft), dtype=np.float32)

    # Set plai from vegtype using plai_init lookup
    plai = np.zeros((npoi, npft), dtype=np.float32)

    for i in range(npoi):
        iv = inveg[i]
        if iv < 1 or iv > 15:
            continue  # bare ground / invalid

        # 0-indexed into PLAI_INIT
        lai_evgr = PLAI_INIT[iv - 1, 0]
        lai_deci = PLAI_INIT[iv - 1, 1]
        lai_shrub = PLAI_INIT[iv - 1, 2]
        lai_grass = PLAI_INIT[iv - 1, 3]

        # Evergreen trees: PFTs 0,2,3,5 (1-indexed: 1,3,4,6)
        for j in [0, 2, 3, 5]:
            plai[i, j] = exist[i, j] / IEVGR * lai_evgr

        # Deciduous trees: PFTs 1,4,6,7 (1-indexed: 2,5,7,8)
        for j in [1, 4, 6, 7]:
            plai[i, j] = exist[i, j] / IDECI * lai_deci

        # Shrubs: PFTs 8,9 (1-indexed: 9,10)
        for j in [8, 9]:
            plai[i, j] = exist[i, j] / ISHRUB * lai_shrub

        # Grasses: PFTs 10,11 (1-indexed: 11,12)
        # Special case for savanna (9) and grassland (10): C3/C4 split
        # Uses sigmoid transition centered at 22°C (width 3°C) instead of
        # hard binary cutoff — eliminates sharp boundary artifacts.
        if iv == 9 or iv == 10:
            c4_frac = 0.80 / (1.0 + np.exp(-(tw[i] - 22.0) / 1.5))  # 0→0.8 sigmoid
            c3_frac = 1.0 - c4_frac
            plai[i, 10] = exist[i, 10] * c4_frac * lai_grass  # C4
            plai[i, 11] = exist[i, 11] * c3_frac * lai_grass  # C3
        else:
            for j in [10, 11]:
                plai[i, j] = exist[i, j] / IGRASS * lai_grass

    # Enforce minimum LAI
    for j in range(npft):
        plai[:, j] = np.maximum(plai[:, j], exist[:, j] * XMINLAI)

    # Biomass from plai
    cbiol = plai / SPECLA[np.newaxis, :]
    cbior = 0.5 * cbiol
    cbiow = np.zeros_like(plai)
    # Wood biomass for tree PFTs only (0-7)
    for j in range(8):
        cbiow[:, j] = plai[:, j] * 10.0 / 6.0
    biomass = cbiol + cbior + cbiow

    # Sapwood fraction
    sapfrac = np.full(npoi, SAPFRAC_INIT, dtype=np.float32)

    # Canopy totals
    totlaiu = plai[:, :8].sum(axis=1)   # upper canopy = trees (PFTs 0-7)
    totlail = plai[:, 8:].sum(axis=1)   # lower canopy = shrubs+grasses (PFTs 8-11)
    totbiou = biomass[:, :8].sum(axis=1)
    totbiol = biomass[:, 8:].sum(axis=1)

    # SAI
    sai = np.zeros((npoi, 2), dtype=np.float32)
    sai[:, 0] = 0.050 * totlail   # lower canopy
    sai[:, 1] = 0.250 * totlaiu   # upper canopy

    # Fractional cover
    wood = cbiow.sum(axis=1)
    fu = (1.0 - np.exp(-wood)) / (1.0 - np.exp(-WOODNORM))
    fl = totlail / 1.0
    fu = np.clip(fu, 0.25, 0.975)
    fl = np.clip(fl, 0.25, 0.975)

    # Canopy LAI (per unit canopy area)
    lai = np.zeros((npoi, 2), dtype=np.float32)
    lai[:, 0] = np.where(fl > 0, totlail / fl, 0.0)
    lai[:, 1] = np.where(fu > 0, totlaiu / fu, 0.0)

    # Canopy heights
    ztop = np.zeros((npoi, 2), dtype=np.float32)
    zbot = np.zeros((npoi, 2), dtype=np.float32)

    zbot[:, 0] = 0.05
    ztop[:, 0] = np.maximum(0.25, lai[:, 0] * 0.25)
    zbot[:, 1] = ztop[:, 0] + 1.0
    ztop[:, 1] = np.maximum(
        zbot[:, 1] + 1.0,
        np.where(fu > 0, 2.50 * totbiou / fu * 0.75, zbot[:, 1] + 1.0),
    )

    # Root distribution
    froot = compute_root_distribution(nsoilay)

    # vegtype0 as float
    vegtype0 = vegtype.astype(np.float32)

    return {
        'plai': jnp.array(plai),
        'cbiol': jnp.array(cbiol),
        'cbior': jnp.array(cbior),
        'cbiow': jnp.array(cbiow),
        'biomass': jnp.array(biomass),
        'exist': jnp.array(exist),
        'vegtype0': jnp.array(vegtype0),
        'sapfrac': jnp.array(sapfrac),
        'fu': jnp.array(fu.astype(np.float32)),
        'fl': jnp.array(fl.astype(np.float32)),
        'lai': jnp.array(lai),
        'sai': jnp.array(sai),
        'ztop': jnp.array(ztop),
        'zbot': jnp.array(zbot),
        'totlaiu': jnp.array(totlaiu.astype(np.float32)),
        'totlail': jnp.array(totlail.astype(np.float32)),
        'totbiou': jnp.array(totbiou.astype(np.float32)),
        'totbiol_agg': jnp.array(totbiol.astype(np.float32)),
        'froot': jnp.array(froot.astype(np.float32)),
        'specla': jnp.array(SPECLA),
        'aleaf': jnp.array(ALEAF),
        'aroot': jnp.array(AROOT),
        'awood': jnp.array(AWOOD),
    }
