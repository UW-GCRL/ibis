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
from ibis.physics.fire import (
    LEGACY, SEASONLENGTH, FireParams, default_fire_params, fine_fuel_fraction,
    fire_legacy, fire_seasonlength, vegetation_density, canopy_damping,
)


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

    Notes
    -----
    The implementation now lives in :mod:`ibis.physics.fire` so that additional
    schemes can be added without editing :func:`dynaveg`. This wrapper is kept
    because existing callers and verification scripts import ``fire`` from this
    module; it delegates and is bit-identical to the previous inline code.
    """
    return fire_legacy(firefac, totlit)


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


def cap_leaf_area(cbiol, cbiow, cbior, pft_params, fu, fl_of, crop_c=None, crop_lai_max=None):
    """Per-type maximum canopy leaf area (docs/CALIBRATION.md 46), applied to the year's leaf pools.

    A layer's LAI per unit of its cover (upper: sum plai[:, :8] / fu; lower: sum plai[:, 8:] / fl) may not exceed the
    leaf-area-weighted mean lai_max of its PFTs. The layer's leaf pools are scaled down to that, and the removed leaf
    carbon goes to each PFT's wood and roots in its awood:aroot ratio (roots when both are 0), so biomass is conserved.
    The upper cover fu comes from wood and is held fixed (the moved carbon can only raise it). The lower cover fl_of(L)
    rises with the lower LAI L, so the lower limit is the largest L with L <= cap x fl_of(L), found by bisection
    (L / fl_of(L) never falls as L rises). Returns the new leaf, wood and root pools and the wood added to PFTs 0-7.
    """
    n = cbiol.shape[0]
    cap = jnp.broadcast_to(pft_params.lai_max, cbiol.shape)
    plai = cbiol * jnp.broadcast_to(pft_params.specla, cbiol.shape)

    def layer(sl):
        w = plai[:, sl]
        tot = jnp.sum(w, axis=1)
        c = jnp.sum(jnp.where(w > 0.0, w * cap[:, sl], 0.0), axis=1) / jnp.maximum(tot, 1e-12)
        return tot, jnp.where(tot > 0.0, c, jnp.inf)

    lu, cu = layer(slice(0, 8))
    ll, cl = layer(slice(8, 12))
    if crop_c is not None and crop_lai_max is not None:
        cl = (1.0 - crop_c) * cl + crop_c * crop_lai_max
    su = jnp.where(lu > cu * fu, cu * fu / jnp.maximum(lu, 1e-12), 1.0)
    if bool(pft_params.lai_cap_lower_ground):
        # the lower cap on the ground leaf area itself (CALIBRATION.md 54)
        lo = jnp.minimum(ll, cl); sl = jnp.where(ll > cl, cl / jnp.maximum(ll, 1e-12), 1.0)
    else:
        lo, hi = jnp.zeros_like(ll), ll
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            ok = mid <= cl * fl_of(mid)
            lo, hi = jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)
        sl = jnp.where(ll > cl * fl_of(ll), lo / jnp.maximum(ll, 1e-12), 1.0)
    scale = jnp.concatenate([jnp.broadcast_to(su[:, None], (n, 8)), jnp.broadcast_to(sl[:, None], (n, 4))], axis=1)
    # Sharing of the excess among the layer's PFTs (cap_share_power; CALIBRATION.md 54): proportional (1) is the
    # per-layer scale above; otherwise each PFT gives up excess x plai^p / sum(plai^p), bounded by its own leaf area.
    pw = jnp.asarray(pft_params.cap_share_power, dtype=plai.dtype)
    def shared(sl_, tot, target):
        w = jnp.where(plai[:, sl_] > 0.0, jnp.power(jnp.maximum(plai[:, sl_], 1e-12), pw), 0.0)
        ex = jnp.maximum(tot - target, 0.0)[:, None] * w / jnp.maximum(jnp.sum(w, axis=1, keepdims=True), 1e-12)
        ex = jnp.minimum(ex, plai[:, sl_])
        return jnp.where(plai[:, sl_] > 0.0, 1.0 - ex / jnp.maximum(plai[:, sl_], 1e-12), 1.0)
    scale_p = jnp.concatenate([shared(slice(0, 8), lu, jnp.minimum(lu, cu * fu)), shared(slice(8, 12), ll, jnp.minimum(ll, lo) if bool(pft_params.lai_cap_lower_ground) else jnp.where(ll > cl * fl_of(ll), lo, ll))], axis=1)
    scale = jnp.where(pw == 1.0, scale, scale_p)
    excess = cbiol * (1.0 - scale)
    aw = jnp.broadcast_to(pft_params.awood, cbiol.shape)
    ar = jnp.broadcast_to(pft_params.aroot, cbiol.shape)
    to_wood = excess * jnp.where(aw + ar > 0.0, aw / jnp.maximum(aw + ar, 1e-12), 0.0)
    dt = cbiol.dtype                        # keep the pools' dtype (the cover and caps may be float64 under x64)
    return ((cbiol * scale).astype(dt), (cbiow + to_wood).astype(dt), (cbior + (excess - to_wood)).astype(dt),
            jnp.sum(to_wood[:, :8], axis=1).astype(dt))


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
        ayanpp_j = (aleaf[..., j] + awood[..., j]) * aynpp_j
        local_ayanpp = local_ayanpp.at[:, j].set(ayanpp_j)

        # Use tauwood0 (no exist-dependent acceleration — that's for dynaveg)
        tauwood_j = tauwood0[..., j]

        # Litterfall
        falll = falll + cbiol[:, j] / tauleaf[..., j]
        fallr = fallr + cbior[:, j] / tauroot[..., j]
        fallw = fallw + cbiow[:, j] / tauwood_j

        # Biomass pool update: exponential relaxation toward NPP-driven equilibrium
        exp_leaf = jnp.exp(-1.0 / tauleaf[..., j])
        cbiol_j = (cbiol[:, j] * exp_leaf +
                   aleaf[..., j] * tauleaf[..., j] * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_leaf))
        cbiol = cbiol.at[:, j].set(cbiol_j)

        exp_wood = jnp.exp(-1.0 / tauwood_j)
        cbiow_j = (cbiow[:, j] * exp_wood +
                   awood[..., j] * tauwood_j * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_wood))
        cbiow = cbiow.at[:, j].set(cbiow_j)

        exp_root = jnp.exp(-1.0 / tauroot[..., j])
        cbior_j = (cbior[:, j] * exp_root +
                   aroot[..., j] * tauroot[..., j] * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_root))
        cbior = cbior.at[:, j].set(cbior_j)

    # Update plai (prognostic LAI) and biomass from new cbiol
    plai = veg.plai
    biomass = veg.biomass
    for j in range(npft):
        cbiol_min = exist[:, j] * xminlai / specla[..., j]
        cbiol = cbiol.at[:, j].set(jnp.maximum(cbiol_min, cbiol[:, j]))

    # Canopy fractions: upper from wood biomass; lower from lower LAI
    # (cover_k = 0 keeps F77's linear rule, cover = LAI; > 0 saturates).
    woodnorm = pft_params.woodnorm

    def _fu_of(wood):
        fu = (1.0 - jnp.exp(-wood / pft_params.fu_k)) / (1.0 - jnp.exp(-woodnorm / pft_params.fu_k))
        return jnp.maximum(pft_params.fu_min, jnp.minimum(0.975, fu))

    def _fl_of(totlail):
        fl_raw = jnp.where(pft_params.cover_k > 0.0,
                           1.0 - jnp.exp(-pft_params.cover_k * totlail), totlail)
        return jnp.maximum(pft_params.fl_min, jnp.minimum(0.975, fl_raw))

    if pft_params.lai_max is not None:      # per-type maximum leaf area (CALIBRATION.md 46)
        cbiol, cbiow, cbior, _ = cap_leaf_area(cbiol, cbiow, cbior, pft_params,
                                               _fu_of(0.001 + jnp.sum(jnp.maximum(0.0, cbiow[:, :8]), axis=1)), _fl_of)

    for j in range(npft):
        plai = plai.at[:, j].set(cbiol[:, j] * specla[..., j])
        biomass = biomass.at[:, j].set(cbiol[:, j] + cbiow[:, j] + cbior[:, j])

    wood = jnp.full_like(veg.fu, 0.001)
    for j in range(8):  # upper canopy PFTs only
        wood = wood + jnp.maximum(0.0, cbiow[:, j])
    fu = _fu_of(wood)

    totlaiu = jnp.sum(plai[:, :8], axis=1)
    totlail = jnp.sum(plai[:, 8:], axis=1)
    fl = _fl_of(totlail)

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
    fire_scheme: str = LEGACY,
    fire_params: FireParams | None = None,
    popd: Float[Array, "npoi"] | None = None,
    tree_surv=None,
    cropfrac=None,
    crop_params=None,
    diag=None,
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

    # Cropland as managed grass (CALIBRATION.md 41, 43): the share c of the cell that is cropland is grass only. The
    # shrubs (PFTs 8, 9) grow on the natural share: their NPP is scaled by (1 - c), so their pools equilibrate at
    # (1 - c) of the natural density; the trees get the sub-grid cover rule below; the grasses are the crops.
    _crop_c = None if cropfrac is None else jnp.clip(cropfrac, 0.0, 1.0)
    for j in range(npft):
        aynpp_j = exist[:, j] * raw_aynpp[:, j]
        if _crop_c is not None and j in (8, 9):
            aynpp_j = aynpp_j * (1.0 - _crop_c)
        ayanpp_j = (aleaf[..., j] + awood[..., j]) * aynpp_j
        local_ayanpp = local_ayanpp.at[:, j].set(ayanpp_j)

        taufin = tauwood0[..., j] / 2.0
        tauwood_j = tauwood0[..., j] - (tauwood0[..., j] - taufin) * (1.0 - exist[:, j])

        falll = falll + cbiol[:, j] / tauleaf[..., j]
        fallr = fallr + cbior[:, j] / tauroot[..., j]
        fallw = fallw + cbiow[:, j] / tauwood_j

        exp_leaf = jnp.exp(-1.0 / tauleaf[..., j])
        cbiol_j = (cbiol[:, j] * exp_leaf +
                   aleaf[..., j] * tauleaf[..., j] * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_leaf))
        cbiol = cbiol.at[:, j].set(cbiol_j)

        exp_wood = jnp.exp(-1.0 / tauwood_j)
        cbiow_j = (cbiow[:, j] * exp_wood +
                   awood[..., j] * tauwood_j * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_wood))
        cbiow = cbiow.at[:, j].set(cbiow_j)

        exp_root = jnp.exp(-1.0 / tauroot[..., j])
        cbior_j = (cbior[:, j] * exp_root +
                   aroot[..., j] * tauroot[..., j] * jnp.maximum(0.0, aynpp_j) *
                   (1.0 - exp_root))
        cbior = cbior.at[:, j].set(cbior_j)

        # Accumulate wood for upper canopy PFTs (1-8)
        wood = jnp.where(j < 8, wood + jnp.maximum(0.0, cbiow_j), wood)

    # Water-limited tree establishment (docs/CALIBRATION.md 14): an annual
    # survival factor on the upper-canopy PFTs from the year's mean soil-water
    # stress. None (default) leaves the pools exactly as F77 computes them.
    if tree_surv is not None:
        ts = tree_surv[:, None]
        m8 = (jnp.arange(npft) < 8)[None, :]
        # The removed share is dead trees: with tree_kill_to_litter (default) it joins this year's litterfall, which
        # soilbgc decomposes next year like any other. Before 2026-10-26 it left the model unbooked -- neither litter
        # nor cdisturb -- 3-5 PgC yr-1 in s31f / s31g (CALIBRATION.md 60-61).
        if getattr(pft_params, 'tree_kill_to_litter', False):
            kill = jnp.where(m8, 1.0 - ts, 0.0)
            falll = falll + jnp.sum(cbiol * kill, axis=1)
            fallw = fallw + jnp.sum(cbiow * kill, axis=1)
            fallr = fallr + jnp.sum(cbior * kill, axis=1)
        cbiol = jnp.where(m8, cbiol * ts, cbiol)
        cbiow = jnp.where(m8, cbiow * ts, cbiow)
        cbior = jnp.where(m8, cbior * ts, cbior)
        wood = wood * tree_surv

    # --- Apply disturbances ---
    disturbf_val = jnp.full_like(veg.fu, 0.005)
    disturbo_val = jnp.full_like(veg.fu, 0.005)

    # Fire disturbance (if enabled)
    # `fire_scheme` is a static Python string from RunConfig, so this branch
    # resolves at trace time and costs nothing under jit.
    if fire_scheme == SEASONLENGTH:
        _fp = fire_params or default_fire_params()
        # F2: above-ground litter + live fine fuel, on a CLM-style ramp.
        # `use_fine_fuel` is a static field, so this branch resolves at trace
        # time. Default False keeps the legacy fuel term.
        _fuel = (fine_fuel_fraction(carbon.totalit, veg.cbiol, _fp)
                 if _fp.use_fine_fuel else None)
        # F3: human ignition/suppression. Requires popd; without it the
        # scheme stays purely climate-driven.
        _popd = popd if (_fp.use_human and popd is not None) else None
        if _fp.use_human and popd is None:
            raise ValueError("fire params set use_human=True but popd was not supplied")
        # F1b: select the danger source. VPD ranks cells far better than
        # soil moisture (Spearman 0.59 vs 0.26 against GFED5) and puts fire at
        # |lat| 11.6 rather than 39.8, against an observed 10.7.
        if _fp.use_vpd_danger:
            # Checkpoints written before F1b have no VPD accumulators. Say so
            # plainly rather than letting an AttributeError surface from deep
            # inside the disturbance branch.
            if not hasattr(veg, "nfirestep_vpd"):
                raise AttributeError(
                    "fire params set use_vpd_danger=True (the default since "
                    "2026-08-24) but this VegetationState predates F1b and has "
                    "no nfirestep_vpd / sdry_vpd. Add them zero-initialised to "
                    "migrate the checkpoint, or pass "
                    "FireParams(use_vpd_danger=False) to keep the "
                    "soil-moisture path."
                )
            _nfs, _sd = veg.nfirestep_vpd, veg.sdry_vpd
            _vd = vegetation_density(veg.plai, _fp)
        else:
            _nfs, _sd = veg.nfirestep, veg.sdry
            _vd = None
        # F7: closed-canopy damping (CALIBRATION.md 59). Static switch, off by default.
        if _fp.use_canopy_damping:
            _cd = canopy_damping(veg.fu, _fp, veg.plai)
            _vd = _cd if _vd is None else _vd * _cd
        _disturbf = fire_seasonlength(
            _nfs, _sd, veg.nytimes, carbon.totlit, _fp,
            fuel=_fuel, popd=_popd, vd=_vd,
        )
    elif fire_scheme == LEGACY:
        _disturbf = fire(veg.firefac, carbon.totlit)
    else:
        raise ValueError(f"unknown fire_scheme {fire_scheme!r}")

    disturbf_fire = jnp.where(isimfire == 1, _disturbf, disturbf_val)
    disturbf_val = disturbf_fire

    # F4: per-PFT fire mortality. IBIS applies `disturbf` identically to every
    # PFT, so savanna trees and boreal conifers lose the same fraction -- which
    # makes the fire->vegetation feedback, the whole reason fire lives in a
    # DGVM, wrong. `mort[j]` scales the FIRE part only; background disturbance
    # (`disturbo`) stays uniform because it is not fire.
    _use_mort = fire_scheme == SEASONLENGTH and (fire_params or default_fire_params()).use_pft_mortality
    _mort = (fire_params or default_fire_params()).mort

    # Combustion completeness. IBIS sends every gram of killed biomass straight
    # to NEE (`vegetation.f:433,481`), which was harmless while `disturbf` was
    # a constant 0.005 but is not once the seasonlength scheme produces 30-50 %
    # burned fractions. Wood is ~97 % of live biomass in the spun-up global
    # state, and `cc_wood = 0.20`, so the old path emitted roughly five times
    # the carbon that `bgc.fire_emissions` reports as burned -- and the other
    # 80 % was neither emitted nor left as litter. It simply vanished.
    #
    # Only the FIRE share is treated this way. Background disturbance
    # (`disturbo`) keeps IBIS's original all-to-NEE behaviour: it is not fire,
    # and changing it is not this module's business.
    _fp_cc = fire_params or default_fire_params()
    _use_cc = fire_scheme == SEASONLENGTH and _fp_cc.use_combustion_completeness

    for j in range(npft):
        if _use_mort:
            disturb_total = disturbf_val * _mort[..., j] + disturbo_val
        else:
            disturb_total = disturbf_val + disturbo_val

        if _use_cc:
            _f = disturbf_val * (_mort[..., j] if _use_mort else 1.0)
            kill_l, kill_w, kill_r = (cbiol[:, j] * _f, cbiow[:, j] * _f,
                                      cbior[:, j] * _f)
            # Fine roots do not combust at all -- they are underground.
            combusted = _fp_cc.cc_leaf * kill_l + _fp_cc.cc_wood * kill_w
            cdisturb = cdisturb + combusted + (
                (cbiol[:, j] + cbiow[:, j] + cbior[:, j]) * disturbo_val)
            # What fire killed but did not burn becomes litter, on the same
            # pools as ordinary litterfall, so soilbgc decomposes it.
            falll = falll + (1.0 - _fp_cc.cc_leaf) * kill_l
            fallw = fallw + (1.0 - _fp_cc.cc_wood) * kill_w
            fallr = fallr + kill_r
        else:
            cdisturb = cdisturb + (
                (cbiol[:, j] + cbiow[:, j] + cbior[:, j]) * disturb_total)

        cbiol = cbiol.at[:, j].set(jnp.maximum(0.0, cbiol[:, j] * (1.0 - disturb_total)))
        cbiow = cbiow.at[:, j].set(jnp.maximum(0.0, cbiow[:, j] * (1.0 - disturb_total)))
        cbior = cbior.at[:, j].set(jnp.maximum(0.0, cbior[:, j] * (1.0 - disturb_total)))

    # --- Maintain minimum biomass and update plai / biomass ---
    # --- Cropland as managed grass (docs/CALIBRATION.md 41, 43) ---
    # The cropland share c of a cell carries crops (grass with the crop capacity, harvested); the trees live on the
    # non-cropland share 1 - c at their natural wood density. Cover is sub-grid: the cell's upper-canopy cover is
    # (1 - c) x the cover that the wood DENSITY ON THE NATURAL SHARE (wood / (1 - c)) gives, applied in the fu formula
    # below. The trees' GPP scales with their cover, so their pools equilibrate on their share without any clearing.
    # (The first version capped the cell cover at 1 - c and scaled the pools to it; because fu saturates near 1 for any
    # wood above ~3 kgC m-2, that cleared every forest with a few percent of cropland -- 12 PgC yr-1, section 43.)
    # The cropland share of the grass leaf turnover leaves the field as harvest instead of becoming litter.
    if cropfrac is not None:
        from ibis.io.parameters import default_crop_params as _dcp
        _cp = crop_params if crop_params is not None else _dcp()
        _harvest = jnp.zeros_like(_crop_c)
        for j in (10, 11):
            _harvest = _harvest + _crop_c * _cp.harvest_index * veg.cbiol[:, j] / tauleaf[..., j]  # from the pre-growth leaf pool, as falll was
        falll = jnp.maximum(falll - _harvest, 0.0)
        cdisturb = cdisturb + _harvest
        if diag is not None:
            diag["harvest"] = _harvest; diag["clearing"] = jnp.zeros_like(_harvest)

    plai = veg.plai
    biomass = veg.biomass
    for j in range(npft):
        cbiol_min = exist[:, j] * xminlai / specla[..., j]
        if tree_surv is not None and j < 8:
            cbiol_min = cbiol_min * (tree_surv > 0.0)
        cbiol = cbiol.at[:, j].set(jnp.maximum(cbiol_min, cbiol[:, j]))

    # --- Fractional cover rules (used by the leaf-area cap and below) ---
    # Cover is a cell-level quantity, so it takes the LAI-weighted mean
    # mortality rather than any single PFT's.
    if _use_mort:
        _w = jnp.maximum(veg.plai, 0.0)
        _wsum = jnp.maximum(jnp.sum(_w, axis=1), 1e-12)
        # (12,) from FireParams, or (npoi, 12) once the Köppen types have gathered it per cell (pft_types.apply_types);
        # the runner passes params.fire since 2026-10-25 (CALIBRATION.md 59), so both reach here.
        _mort_mean = jnp.sum(_w * (_mort if _mort.ndim == 2 else _mort[None, :]), axis=1) / _wsum
        disturb_total = disturbf_val * _mort_mean + disturbo_val
    else:
        disturb_total = disturbf_val + disturbo_val

    def _fu_of(wood):
        if _crop_c is not None:                                   # sub-grid cropland: cover of the trees' own share, times that share
            _wn = wood / jnp.maximum(1.0 - _crop_c, 0.01)
            fu = (1.0 - _crop_c) * (1.0 - jnp.exp(-_wn / pft_params.fu_k)) / (1.0 - jnp.exp(-woodnorm / pft_params.fu_k))
        else:
            fu = (1.0 - jnp.exp(-wood / pft_params.fu_k)) / (1.0 - jnp.exp(-woodnorm / pft_params.fu_k))
        fu = fu * (1.0 - disturb_total)
        return jnp.maximum(pft_params.fu_min, jnp.minimum(0.975, fu))

    def _fl_of(totlail):
        fl = jnp.where(pft_params.cover_k > 0.0,
                       1.0 - jnp.exp(-pft_params.cover_k * totlail), totlail / 1.0)
        fl = fl * (1.0 - disturb_total)
        return jnp.maximum(pft_params.fl_min, jnp.minimum(0.975, fl))

    if pft_params.lai_max is not None:      # per-type maximum leaf area (CALIBRATION.md 46)
        _clm = None if (_crop_c is None or crop_params is None) else crop_params.lai_max
        cbiol, cbiow, cbior, _dw = cap_leaf_area(cbiol, cbiow, cbior, pft_params, _fu_of(wood), _fl_of,
                                                 crop_c=_crop_c, crop_lai_max=_clm)
        wood = wood + _dw

    for j in range(npft):
        plai = plai.at[:, j].set(cbiol[:, j] * specla[..., j])
        biomass = biomass.at[:, j].set(cbiol[:, j] + cbiow[:, j] + cbior[:, j])

    # --- Update annual summary ---
    ayneetot = summary.ayneetot - cdisturb
    ayanpptot = jnp.sum(local_ayanpp, axis=1)

    totlaiu = jnp.sum(plai[:, :8], axis=1)
    totlail = jnp.sum(plai[:, 8:], axis=1)
    totbiou = jnp.sum(biomass[:, :8], axis=1)
    totbiol_agg = jnp.sum(biomass[:, 8:], axis=1)

    # --- Update fractional cover ---
    fu = _fu_of(wood)
    fl = _fl_of(totlail)

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
