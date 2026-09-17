"""Leaf phenology routines for IBIS-JAX.

Ported from vegetation.f90: pheno subroutine.

Updates leaf phenology indices (cold/drought triggers) and derives
LAI, SAI, canopy height, and PFT fractions for each grid cell.
Called once per day.
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from ibis.state.vegetation import VegetationState
from ibis.state.summary import SummaryState


class PhenoParams(eqx.Module):
    """Leaf-phenology thresholds. Defaults are the IBIS 2.6b4 literals from
    vegetation.f (pheno), which were hard-coded inside `pheno()` until the
    carbon recalibration made seasonal pattern a target (docs/CALIBRATION.md).
    Bit-identical to the previous behaviour at these defaults -- guarded.

    Temperatures in degrees C (added to tmelt inside pheno); GDD in K-days.
    """
    #: upper-canopy (trees) cold phenology: leaves drop below tc + offset
    tthresh_u_offset: float = 5.0
    #: upper-canopy GDD accumulation starts above this air temperature
    gthresh_u_c: float = 0.0
    #: budburst GDD threshold and ramp width, upper canopy
    gdd_thresh_u: float = 100.0
    gdd_ramp_u: float = 50.0
    #: lower-canopy (grasses/shrubs) thresholds
    tthresh_l_c: float = 0.0
    gthresh_l_c: float = -5.0
    gdd_thresh_l: float = 150.0
    gdd_ramp_l: float = 50.0
    #: drought phenology: 10-day mean net assimilation below this drops leaves
    drop_thresh: float = 0.0
    #: minimum retained leaf fraction under drought
    drop_floor: float = 0.1
    #: e-folding days for leaf drop/regrowth
    ddays: float = 15.0
    #: Soil-water-stress dormancy trigger (docs/CALIBRATION.md 14). Leaves of
    #: the drought-deciduous PFTs also drop while the canopy's soil-water
    #: stress factor (drystress, 1 = unstressed) is below this. IBIS 2.6b4
    #: triggers drought phenology only on NEGATIVE 10-day net assimilation,
    #: which a leafed canopy in daylight essentially never reaches -- in 60
    #: semi-arid cells it fired in no month while stresstl sat at 0.3. The
    #: default -1.0 can never be reached, so default behaviour is unchanged.
    drop_stress_thresh: float = -1.0
    #: Absolute upper-canopy leaf-drop temperature (C). IBIS drops
    #: cold-deciduous tree leaves when the 10-day mean falls below
    #: `tc + tthresh_u_offset`, RELATIVE to the local coldest month: a
    #: continental aspen keeps leaves to -15 C, and in mild-winter cells the
    #: threshold sits inside the winter range so `tempu` dithers at ~0.5 all
    #: year (docs/CALIBRATION.md 14). Real temperate deciduous senescence is
    #: near an absolute 5-10 C. The threshold becomes max(relative, absolute);
    #: default -100 C never binds, so default behaviour is unchanged.
    tthresh_u_abs: float = -100.0


def pheno(
    veg: VegetationState,
    summary: SummaryState,
    td: Float[Array, "npoi"],
    epsilon: float,
    params: PhenoParams | None = None,
    stresstu=None,
    stresstl=None,
) -> VegetationState:
    """Update leaf phenology and derive canopy structure.

    Called once per day. Updates:
      - Cold-phenology triggers (tempu, templ)
      - GDD accumulators (agddu, agddl)
      - Drought triggers (dropu, dropls, dropl4, dropl3)
      - PFT fractions (frac)
      - Canopy LAI (lai)
      - Canopy height (ztop, zbot for lower canopy)

    Parameters
    ----------
    veg : VegetationState
        Current vegetation state.
    summary : SummaryState
        10-day running means needed for phenology.
    td : (npoi,)
        Daily mean temperature (K).
    epsilon : float
        Small number to avoid zero-divides.

    Returns
    -------
    VegetationState
        Updated vegetation state.
    """
    pp = params if params is not None else PhenoParams()
    tmelt = 273.16
    ddfac = 1.0 / pp.ddays

    a10td = summary.a10td
    # Soil-water stress per canopy (1 = unstressed). When not supplied, a
    # value above any admissible threshold keeps the stress trigger inert.
    s_u = stresstu if stresstu is not None else jnp.full_like(a10td, 2.0)
    s_l = stresstl if stresstl is not None else jnp.full_like(a10td, 2.0)
    a10ancub = summary.a10ancub
    a10ancls = summary.a10ancls
    a10ancl4 = summary.a10ancl4
    a10ancl3 = summary.a10ancl3

    # --- Upper canopy winter phenology ---
    tthreshold_u = jnp.maximum(jnp.maximum(0.0 + tmelt, veg.tc + pp.tthresh_u_offset + tmelt),
                               pp.tthresh_u_abs + tmelt)
    gthreshold_u = pp.gthresh_u_c + tmelt

    agddu = jnp.where(
        a10td < gthreshold_u,
        0.0,
        veg.agddu + td - gthreshold_u,
    )

    tempu = jnp.where(
        a10td < tthreshold_u,
        jnp.maximum(0.0, veg.tempu - ddfac),
        jnp.minimum(1.0, jnp.maximum(0.0, agddu - pp.gdd_thresh_u) / pp.gdd_ramp_u),
    )

    # --- Lower canopy winter phenology ---
    tthreshold_l = pp.tthresh_l_c + tmelt
    gthreshold_l = pp.gthresh_l_c + tmelt

    agddl = jnp.where(
        a10td < gthreshold_l,
        0.0,
        veg.agddl + td - gthreshold_l,
    )

    templ = jnp.where(
        a10td < tthreshold_l,
        jnp.maximum(0.0, veg.templ - ddfac),
        jnp.minimum(1.0, jnp.maximum(0.0, agddl - pp.gdd_thresh_l) / pp.gdd_ramp_l),
    )

    # --- Drought phenology ---
    dropu = jnp.where(
        (a10ancub < pp.drop_thresh) | (s_u < pp.drop_stress_thresh),
        jnp.maximum(pp.drop_floor, veg.dropu - ddfac),
        jnp.minimum(1.0, veg.dropu + ddfac),
    )

    dropls = jnp.where(
        (a10ancls < pp.drop_thresh) | (s_l < pp.drop_stress_thresh),
        jnp.maximum(pp.drop_floor, veg.dropls - ddfac),
        jnp.minimum(1.0, veg.dropls + ddfac),
    )

    dropl4 = jnp.where(
        (a10ancl4 < pp.drop_thresh) | (s_l < pp.drop_stress_thresh),
        jnp.maximum(pp.drop_floor, veg.dropl4 - ddfac),
        jnp.minimum(1.0, veg.dropl4 + ddfac),
    )

    dropl3 = jnp.where(
        (a10ancl3 < pp.drop_thresh) | (s_l < pp.drop_stress_thresh),
        jnp.maximum(pp.drop_floor, veg.dropl3 - ddfac),
        jnp.minimum(1.0, veg.dropl3 + ddfac),
    )

    # --- Update LAI and PFT fractions ---
    plai = veg.plai

    # Upper canopy: weighted average LAI
    # PFTs 1-8 (0-7): evergreen (no phenology), deciduous (dropu), cold-deciduous (tempu)
    avglaiu = (
        plai[:, 0]             +  # PFT 1: trop broadleaf evergreen
        plai[:, 1] * dropu     +  # PFT 2: trop broadleaf drought-deciduous
        plai[:, 2]             +  # PFT 3: warm-temp broadleaf evergreen
        plai[:, 3]             +  # PFT 4: temp conifer evergreen
        plai[:, 4] * tempu     +  # PFT 5: temp broadleaf cold-deciduous
        plai[:, 5]             +  # PFT 6: boreal conifer evergreen
        plai[:, 6] * tempu     +  # PFT 7: boreal broadleaf cold-deciduous
        plai[:, 7] * tempu        # PFT 8: boreal conifer cold-deciduous
    )

    avglaiu_safe = jnp.maximum(avglaiu, epsilon)

    frac = veg.frac
    frac = frac.at[:, 0].set(plai[:, 0] / avglaiu_safe)
    frac = frac.at[:, 1].set(plai[:, 1] * dropu / avglaiu_safe)
    frac = frac.at[:, 2].set(plai[:, 2] / avglaiu_safe)
    frac = frac.at[:, 3].set(plai[:, 3] / avglaiu_safe)
    frac = frac.at[:, 4].set(plai[:, 4] * tempu / avglaiu_safe)
    frac = frac.at[:, 5].set(plai[:, 5] / avglaiu_safe)
    frac = frac.at[:, 6].set(plai[:, 6] * tempu / avglaiu_safe)
    frac = frac.at[:, 7].set(plai[:, 7] * tempu / avglaiu_safe)

    # Lower canopy: PFTs 9-12 (8-11)
    avglail = (
        plai[:, 8]                                      +  # PFT 9: evergreen shrubs
        plai[:, 9]  * jnp.minimum(templ, dropls)        +  # PFT 10: deciduous shrubs
        plai[:, 10] * jnp.minimum(templ, dropl4)        +  # PFT 11: C4 grasses
        plai[:, 11] * jnp.minimum(templ, dropl3)           # PFT 12: C3 grasses
    )

    avglail_safe = jnp.maximum(avglail, epsilon)

    frac = frac.at[:, 8].set(plai[:, 8] / avglail_safe)
    frac = frac.at[:, 9].set(plai[:, 9] * jnp.minimum(templ, dropls) / avglail_safe)
    frac = frac.at[:, 10].set(plai[:, 10] * jnp.minimum(templ, dropl4) / avglail_safe)
    frac = frac.at[:, 11].set(plai[:, 11] * jnp.minimum(templ, dropl3) / avglail_safe)

    # LAI per canopy level (capped at 12)
    lai_lower = jnp.minimum(avglail / jnp.maximum(veg.fl, epsilon), 12.0)
    lai_upper = jnp.minimum(avglaiu / jnp.maximum(veg.fu, epsilon), 12.0)

    lai = veg.lai
    lai = lai.at[:, 0].set(lai_lower)
    lai = lai.at[:, 1].set(lai_upper)

    # --- Canopy height (lower canopy only; upper set by dynaveg) ---
    zbot = veg.zbot.at[:, 0].set(0.05)
    ztop_lower = jnp.maximum(0.25, lai_lower * 0.25)
    ztop_lower = jnp.minimum(ztop_lower, veg.zbot[:, 1] - 0.5)
    ztop = veg.ztop.at[:, 0].set(ztop_lower)

    # --- Pack updated fields ---
    return eqx.tree_at(
        lambda v: (v.tempu, v.templ, v.agddu, v.agddl,
                   v.dropu, v.dropls, v.dropl3, v.dropl4,
                   v.frac, v.lai, v.ztop, v.zbot),
        veg,
        (tempu, templ, agddu, agddl,
         dropu, dropls, dropl3, dropl4,
         frac, lai, ztop, zbot),
    )
