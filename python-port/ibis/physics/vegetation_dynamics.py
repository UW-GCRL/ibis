"""Vegetation dynamics routines for IBIS-JAX.

Ported from vegetation.f90: dynaveg, fire, vegmap subroutines.

Annual dynamic vegetation update: updates biomass pools, applies
disturbances, updates fractional cover, canopy height, SAI, sapwood.
Called once per year.
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from ibis.state.vegetation import VegetationState
from ibis.state.carbon import CarbonState
from ibis.state.summary import SummaryState
from ibis.io.parameters import PFTParams


def fire(
    firefac: Float[Array, "npoi"],
    totlit: Float[Array, "npoi"],
) -> Float[Array, "npoi"]:
    """Calculate fire disturbance probability from litter loading.

    Parameters
    ----------
    firefac : (npoi,)
        Fire dryness factor (0-1).
    totlit : (npoi,)
        Total litter C (kgC/m2).

    Returns
    -------
    disturbf : (npoi,)
        Fire disturbance fraction.
    """
    burn = firefac * jnp.minimum(1.0, totlit / 0.200)
    return jnp.minimum(1.0, jnp.maximum(0.0, 1.0 - jnp.exp(-0.5 * burn)))


def vegmap(
    plai: Float[Array, "npoi npft"],
    totlaiu: Float[Array, "npoi"],
    totlail: Float[Array, "npoi"],
    gdd5: Float[Array, "npoi"],
    gdd0: Float[Array, "npoi"],
    epsilon: float,
) -> Float[Array, "npoi"]:
    """Classify each grid cell into one of 15 IBIS vegetation types.

    Parameters
    ----------
    plai : (npoi, 12)
        Per-PFT LAI.
    totlaiu, totlail : (npoi,)
        Total upper/lower canopy LAI.
    gdd5, gdd0 : (npoi,)
        Growing degree days.
    epsilon : float
        Small number.

    Returns
    -------
    vegtype0 : (npoi,)
        Vegetation type classification (1-15).
    """
    treelai = totlaiu
    shrublai = plai[:, 8] + plai[:, 9]
    grasslai = plai[:, 10] + plai[:, 11]
    totlai = jnp.maximum(0.01, totlail + totlaiu)

    # Find dominant tree type (PFTs 1-8, indices 0-7)
    tree_plai = plai[:, :8]
    domtree = jnp.argmax(tree_plai, axis=1)  # 0-indexed
    maxlai = jnp.max(tree_plai, axis=1)
    # If no trees, domtree = -1
    has_trees = maxlai > 0.0
    domtree_1based = jnp.where(has_trees, domtree + 1, 0)

    # Start with default unknown
    vegtype0 = jnp.full_like(treelai, -999.99)

    # --- Tropical broadleaf evergreen (domtree=1) ---
    is_dt1 = domtree_1based == 1
    vt_dt1 = jnp.where(treelai > 2.5, 1.0,
             jnp.where(treelai > 0.5, 9.0,
             jnp.where(shrublai >= grasslai, 11.0, 10.0)))
    vegtype0 = jnp.where(is_dt1, vt_dt1, vegtype0)

    # --- Tropical broadleaf drought-deciduous (domtree=2) ---
    is_dt2 = domtree_1based == 2
    vt_dt2 = jnp.where(treelai > 2.5, 2.0,
             jnp.where(treelai > 0.5, 9.0,
             jnp.where(shrublai >= grasslai, 11.0, 10.0)))
    vegtype0 = jnp.where(is_dt2, vt_dt2, vegtype0)

    # --- Warm-temperate broadleaf evergreen (domtree=3) ---
    is_dt3 = domtree_1based == 3
    vt_dt3 = jnp.where(treelai > 2.5, 3.0,
             jnp.where(treelai > 0.5, 9.0,
             jnp.where(shrublai >= grasslai, 11.0, 10.0)))
    vegtype0 = jnp.where(is_dt3, vt_dt3, vegtype0)

    # --- Temperate conifer evergreen (domtree=4) ---
    is_dt4 = domtree_1based == 4
    vt_dt4 = jnp.where(treelai > 1.5, 4.0,
             jnp.where(treelai > 0.5, 9.0,
             jnp.where(shrublai >= grasslai, 11.0, 10.0)))
    vegtype0 = jnp.where(is_dt4, vt_dt4, vegtype0)

    # --- Temperate broadleaf cold-deciduous (domtree=5) ---
    is_dt5 = domtree_1based == 5
    vt_dt5 = jnp.where(treelai > 1.5, 5.0,
             jnp.where(treelai > 0.5, 9.0,
             jnp.where(shrublai >= grasslai, 11.0, 10.0)))
    vegtype0 = jnp.where(is_dt5, vt_dt5, vegtype0)

    # --- Boreal types (domtree=6,7,8) ---
    is_dt6 = domtree_1based == 6
    vegtype0 = jnp.where(is_dt6, 6.0, vegtype0)

    is_dt7 = domtree_1based == 7
    vegtype0 = jnp.where(is_dt7, 7.0, vegtype0)

    is_dt8 = domtree_1based == 8
    vegtype0 = jnp.where(is_dt8, 7.0, vegtype0)

    # --- Mixed forest check (domtree 4-8) ---
    is_dt48 = (domtree_1based >= 4) & (domtree_1based <= 8)
    decid_lai = plai[:, 4] + plai[:, 6] + plai[:, 7]
    total_tempboreal = plai[:, 3] + plai[:, 4] + plai[:, 5] + plai[:, 6] + plai[:, 7]
    ratio = decid_lai / jnp.maximum(total_tempboreal, epsilon)
    is_mixed = is_dt48 & (treelai > 1.0) & (ratio > 0.45) & (ratio < 0.55)
    vegtype0 = jnp.where(is_mixed, 8.0, vegtype0)

    # Open woodland for domtree 4-5 with low tree LAI
    is_open45 = is_dt48 & (domtree_1based <= 5) & (treelai <= 1.0)
    vegtype0 = jnp.where(is_open45 & (shrublai >= grasslai), 11.0,
               jnp.where(is_open45 & (grasslai > shrublai), 10.0, vegtype0))

    # --- No dominant tree ---
    is_dt0 = domtree_1based == 0
    vt_dt0 = jnp.where(treelai > 1.0, 9.0,
             jnp.where(shrublai >= grasslai, 11.0, 10.0))
    vegtype0 = jnp.where(is_dt0, vt_dt0, vegtype0)

    # --- Override by total LAI ---
    vegtype0 = jnp.where(totlai < 1.0, 12.0, vegtype0)
    vegtype0 = jnp.where(totlai <= 0.4, 14.0, vegtype0)

    # --- Override by climate ---
    vegtype0 = jnp.where((gdd5 < 350.0) & (totlai >= 0.4), 13.0, vegtype0)
    vegtype0 = jnp.where((gdd5 < 350.0) & (totlai < 0.4), 15.0, vegtype0)
    vegtype0 = jnp.where(gdd0 < 100.0, 15.0, vegtype0)

    return vegtype0


def biomass_update(
    veg: VegetationState,
    carbon: CarbonState,
    summary: SummaryState,
    pft_params: PFTParams,
    epsilon: float,
) -> tuple[VegetationState, CarbonState, SummaryState]:
    """Annual biomass and LAI update from NPP allocation.

    ALWAYS called (even with isimveg=0) to couple LAI to the carbon cycle:
      NPP → allocation (aleaf/aroot/awood) → biomass pools → LAI = cbiol × specla

    Does NOT change PFT composition (frac, exist, vegtype, fu, fl).
    Does NOT apply disturbances (fire, background mortality).
    For disturbances and PFT reclassification, call dynaveg() with isimveg=1.
    """
    npft = 12
    tauleaf = pft_params.tauleaf
    tauroot = pft_params.tauroot
    tauwood0 = pft_params.tauwood0
    aleaf = pft_params.aleaf
    aroot = pft_params.aroot
    awood = pft_params.awood
    specla = pft_params.specla
    xminlai = pft_params.xminlai

    cbiol = veg.cbiol
    cbiow = veg.cbiow
    cbior = veg.cbior
    exist = veg.exist
    raw_aynpp = summary.aynpp

    falll = jnp.zeros_like(veg.fu)
    fallr = jnp.zeros_like(veg.fu)
    fallw = jnp.zeros_like(veg.fu)
    local_ayanpp = jnp.zeros_like(raw_aynpp)

    for j in range(npft):
        aynpp_j = exist[:, j] * raw_aynpp[:, j]
        ayanpp_j = (aleaf[j] + awood[j]) * aynpp_j
        local_ayanpp = local_ayanpp.at[:, j].set(ayanpp_j)

        # Use tauwood0 (no exist-dependent acceleration — that's for dynaveg)
        tauwood_j = tauwood0[j]

        # Litterfall
        falll = falll + cbiol[:, j] / tauleaf[j]
        fallr = fallr + cbior[:, j] / tauroot[j]
        fallw = fallw + cbiow[:, j] / tauwood_j

        # Biomass pool update: exponential relaxation toward NPP-driven equilibrium
        exp_leaf = jnp.exp(-1.0 / tauleaf[j])
        cbiol_j = (cbiol[:, j] * exp_leaf +
                   aleaf[j] * tauleaf[j] * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_leaf))
        cbiol = cbiol.at[:, j].set(cbiol_j)

        exp_wood = jnp.exp(-1.0 / tauwood_j)
        cbiow_j = (cbiow[:, j] * exp_wood +
                   awood[j] * tauwood_j * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_wood))
        cbiow = cbiow.at[:, j].set(cbiow_j)

        exp_root = jnp.exp(-1.0 / tauroot[j])
        cbior_j = (cbior[:, j] * exp_root +
                   aroot[j] * tauroot[j] * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_root))
        cbior = cbior.at[:, j].set(cbior_j)

    # Update plai (prognostic LAI) and biomass from new cbiol
    plai = veg.plai
    biomass = veg.biomass
    for j in range(npft):
        cbiol_min = exist[:, j] * xminlai / specla[j]
        cbiol = cbiol.at[:, j].set(jnp.maximum(cbiol_min, cbiol[:, j]))
        plai = plai.at[:, j].set(cbiol[:, j] * specla[j])
        biomass = biomass.at[:, j].set(cbiol[:, j] + cbiow[:, j] + cbior[:, j])

    # Update canopy fractions from wood biomass
    woodnorm = pft_params.woodnorm
    wood = jnp.full_like(veg.fu, 0.001)
    for j in range(8):  # upper canopy PFTs only
        wood = wood + jnp.maximum(0.0, cbiow[:, j])

    fu = (1.0 - jnp.exp(-wood)) / (1.0 - jnp.exp(-woodnorm))
    fu = jnp.maximum(0.25, jnp.minimum(0.975, fu))

    totlaiu = jnp.sum(plai[:, :8], axis=1)
    totlail = jnp.sum(plai[:, 8:], axis=1)

    fl = jnp.maximum(0.25, jnp.minimum(0.975, totlail))

    # Canopy-level LAI (per unit canopy area)
    lai = veg.lai
    lai = lai.at[:, 1].set(totlaiu / jnp.maximum(fu, epsilon))
    lai = lai.at[:, 0].set(totlail / jnp.maximum(fl, epsilon))
    lai = jnp.minimum(lai, 12.0)

    totbiou = jnp.sum(biomass[:, :8], axis=1)
    totbiol_agg = jnp.sum(biomass[:, 8:], axis=1)

    # SAI update from LAI
    sai = veg.sai
    sai = sai.at[:, 0].set(0.050 * totlail)
    sai = sai.at[:, 1].set(0.250 * totlaiu)

    ayanpptot = jnp.sum(local_ayanpp, axis=1)

    # Update carbon state (litterfall)
    new_carbon = eqx.tree_at(
        lambda c: (c.falll, c.fallr, c.fallw),
        carbon,
        (falll, fallr, fallw),
    )

    # Update vegetation state (biomass, LAI, canopy fractions — NOT frac, vegtype)
    new_veg = eqx.tree_at(
        lambda v: (v.cbiol, v.cbiow, v.cbior, v.plai, v.biomass,
                   v.fu, v.fl, v.lai, v.sai,
                   v.totlaiu, v.totlail, v.totbiou, v.totbiol_agg),
        veg,
        (cbiol, cbiow, cbior, plai, biomass,
         fu, fl, lai, sai,
         totlaiu, totlail, totbiou, totbiol_agg),
    )

    # Update summary (ayanpp derived, preserve raw aynpp)
    new_summary = eqx.tree_at(
        lambda s: (s.ayanpp, s.ayanpptot),
        summary,
        (local_ayanpp, ayanpptot),
    )

    return new_veg, new_carbon, new_summary


def dynaveg(
    veg: VegetationState,
    carbon: CarbonState,
    summary: SummaryState,
    pft_params: PFTParams,
    isimfire: int,
    epsilon: float,
) -> tuple[VegetationState, CarbonState, SummaryState]:
    """Annual dynamic vegetation update.

    Updates biomass pools, applies disturbances, updates fractional
    cover, canopy height, SAI, sapwood fraction, and calls vegmap.

    Called once per year.

    Parameters
    ----------
    veg : VegetationState
        Current vegetation state.
    carbon : CarbonState
        Current C pool state.
    summary : SummaryState
        Summary accumulators.
    pft_params : PFTParams
        PFT parameters.
    isimfire : int
        Fire switch (1 = fire active).
    epsilon : float
        Small number.

    Returns
    -------
    (VegetationState, CarbonState, SummaryState)
    """
    npft = 12
    woodnorm = pft_params.woodnorm

    tauleaf = pft_params.tauleaf
    tauroot = pft_params.tauroot
    tauwood0 = pft_params.tauwood0
    aleaf = pft_params.aleaf
    aroot = pft_params.aroot
    awood = pft_params.awood
    specla = pft_params.specla
    xminlai = pft_params.xminlai

    cbiol = veg.cbiol
    cbiow = veg.cbiow
    cbior = veg.cbior
    exist = veg.exist

    # Use LOCAL copies of NPP — do NOT overwrite summary accumulators
    # (summary.aynpp contains the raw accumulated NPP for yearly output)
    raw_aynpp = summary.aynpp
    local_ayanpp = jnp.zeros_like(raw_aynpp)

    # --- Update NPP and biomass pools for each PFT ---
    falll = jnp.zeros_like(veg.fu)
    fallr = jnp.zeros_like(veg.fu)
    fallw = jnp.zeros_like(veg.fu)
    cdisturb = jnp.zeros_like(veg.fu)
    wood = jnp.full_like(veg.fu, 0.001)

    for j in range(npft):
        aynpp_j = exist[:, j] * raw_aynpp[:, j]
        ayanpp_j = (aleaf[j] + awood[j]) * aynpp_j
        local_ayanpp = local_ayanpp.at[:, j].set(ayanpp_j)

        taufin = tauwood0[j] / 2.0
        tauwood_j = tauwood0[j] - (tauwood0[j] - taufin) * (1.0 - exist[:, j])

        falll = falll + cbiol[:, j] / tauleaf[j]
        fallr = fallr + cbior[:, j] / tauroot[j]
        fallw = fallw + cbiow[:, j] / tauwood_j

        exp_leaf = jnp.exp(-1.0 / tauleaf[j])
        cbiol_j = (cbiol[:, j] * exp_leaf +
                   aleaf[j] * tauleaf[j] * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_leaf))
        cbiol = cbiol.at[:, j].set(cbiol_j)

        exp_wood = jnp.exp(-1.0 / tauwood_j)
        cbiow_j = (cbiow[:, j] * exp_wood +
                   awood[j] * tauwood_j * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_wood))
        cbiow = cbiow.at[:, j].set(cbiow_j)

        exp_root = jnp.exp(-1.0 / tauroot[j])
        cbior_j = (cbior[:, j] * exp_root +
                   aroot[j] * tauroot[j] * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_root))
        cbior = cbior.at[:, j].set(cbior_j)

        # Accumulate wood for upper canopy PFTs (1-8)
        wood = jnp.where(j < 8, wood + jnp.maximum(0.0, cbiow_j), wood)

    # --- Apply disturbances ---
    disturbf_val = jnp.full_like(veg.fu, 0.005)
    disturbo_val = jnp.full_like(veg.fu, 0.005)

    # Fire disturbance (if enabled)
    disturbf_fire = jnp.where(
        isimfire == 1,
        fire(veg.firefac, carbon.totlit),
        disturbf_val,
    )
    disturbf_val = disturbf_fire

    for j in range(npft):
        disturb_total = disturbf_val + disturbo_val
        cdisturb = cdisturb + (cbiol[:, j] + cbiow[:, j] + cbior[:, j]) * disturb_total

        cbiol = cbiol.at[:, j].set(jnp.maximum(0.0, cbiol[:, j] * (1.0 - disturb_total)))
        cbiow = cbiow.at[:, j].set(jnp.maximum(0.0, cbiow[:, j] * (1.0 - disturb_total)))
        cbior = cbior.at[:, j].set(jnp.maximum(0.0, cbior[:, j] * (1.0 - disturb_total)))

    # --- Maintain minimum biomass and update plai / biomass ---
    plai = veg.plai
    biomass = veg.biomass
    for j in range(npft):
        cbiol_min = exist[:, j] * xminlai / specla[j]
        cbiol = cbiol.at[:, j].set(jnp.maximum(cbiol_min, cbiol[:, j]))
        plai = plai.at[:, j].set(cbiol[:, j] * specla[j])
        biomass = biomass.at[:, j].set(cbiol[:, j] + cbiow[:, j] + cbior[:, j])

    # --- Update annual summary ---
    ayneetot = summary.ayneetot - cdisturb
    ayanpptot = jnp.sum(local_ayanpp, axis=1)

    totlaiu = jnp.sum(plai[:, :8], axis=1)
    totlail = jnp.sum(plai[:, 8:], axis=1)
    totbiou = jnp.sum(biomass[:, :8], axis=1)
    totbiol_agg = jnp.sum(biomass[:, 8:], axis=1)

    # --- Update fractional cover ---
    disturb_total = disturbf_val + disturbo_val
    fu = (1.0 - jnp.exp(-wood)) / (1.0 - jnp.exp(-woodnorm))
    fu = fu * (1.0 - disturb_total)
    fu = jnp.maximum(0.25, jnp.minimum(0.975, fu))

    fl = totlail / 1.0
    fl = fl * (1.0 - disturb_total)
    fl = jnp.maximum(0.25, jnp.minimum(0.975, fl))

    # --- Update upper canopy height ---
    zbot = veg.zbot.at[:, 1].set(3.0)
    ztop_upper = jnp.maximum(zbot[:, 1] + 1.0,
                             2.50 * totbiou / jnp.maximum(fu, epsilon) * 0.75)
    ztop = veg.ztop.at[:, 1].set(ztop_upper)

    # --- SAI and sapwood fraction ---
    sai = veg.sai
    sai = sai.at[:, 0].set(0.050 * totlail)
    sai = sai.at[:, 1].set(0.250 * totlaiu)

    sapspeed = 25.0      # m/day
    trans = 0.0025       # m/day transpiration
    saparea = trans / sapspeed
    sapvolume = saparea * ztop_upper * 0.75
    denswood = 400.0     # kg/m3
    sapfrac = jnp.minimum(0.50, jnp.maximum(0.05,
                          sapvolume * denswood / jnp.maximum(wood, epsilon)))

    # --- Vegetation type classification ---
    vegtype0 = vegmap(plai, totlaiu, totlail, veg.gdd5, veg.gdd0, epsilon)

    # --- Update carbon state ---
    new_carbon = eqx.tree_at(
        lambda c: (c.falll, c.fallr, c.fallw, c.cdisturb),
        carbon,
        (falll, fallr, fallw, cdisturb),
    )

    # --- Update vegetation state ---
    new_veg = eqx.tree_at(
        lambda v: (v.cbiol, v.cbiow, v.cbior, v.plai, v.biomass,
                   v.frac, v.fu, v.fl, v.lai, v.sai, v.ztop, v.zbot,
                   v.sapfrac, v.disturbf, v.disturbo,
                   v.totlaiu, v.totlail, v.totbiou, v.totbiol_agg,
                   v.vegtype0),
        veg,
        (cbiol, cbiow, cbior, plai, biomass,
         veg.frac, fu, fl, veg.lai, sai, ztop, zbot,
         sapfrac, disturbf_val, disturbo_val,
         totlaiu, totlail, totbiou, totbiol_agg,
         vegtype0),
    )

    # --- Update summary ---
    # Only update ayanpp (derived) and ayanpptot; preserve raw aynpp untouched
    new_summary = eqx.tree_at(
        lambda s: (s.ayanpp, s.ayneetot, s.ayanpptot),
        summary,
        (local_ayanpp, ayneetot, ayanpptot),
    )

    return new_veg, new_carbon, new_summary
