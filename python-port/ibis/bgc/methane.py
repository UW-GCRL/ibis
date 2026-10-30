"""Methane scheme for IBIS-JAX -- M1: inundation and water table.

Built in tiers (docs/METHANE.md 4.0): M-lite first, and each later tier only if
evaluation shows the simpler one is systematically wrong. This module currently
holds M1 alone -- the water table diagnostic and the inundated fraction it is
paired with. Production, oxidation and transport are M2 onward.

`ibis/bgc/wetlands.py` is quarantined and must not be extended
(docs/METHANE.md 2.2); this is a separate module deliberately.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class MethaneParams(eqx.Module):
    """Parameters for the methane scheme. See docs/METHANE.md 4.7.

    Only the M1 subset is defined so far; adding a field before the tier that
    uses it invites the dead-parameter problem the fire module ran into, where
    `firefac` was computed, verified, and fed to nothing for 26 years.
    """

    #: Liquid saturation at or above which a layer counts as saturated. Below
    #: it, the layer contributes its unsaturated share to the water table
    #: depth. 0.95 rather than 1.0 because a real soil retains trapped air.
    sat_crit: float = 0.95
    #: Deepest water table the diagnostic will report, m. IBIS's soil column
    #: is 4 m, so nothing below that is resolved and reporting more would be
    #: invention.
    z_max: float = 4.0

    # --- M2: production ----------------------------------------------------
    #: **Gross** molar CH4:CO2 production ratio below the water table.
    #: TRIPLEX-GHG's calibrated mean over 19 sites (Zhu et al. 2014, GMD 7,
    #: 981). Gross, not net: oxidation is removed separately by
    #: :func:`surviving_fraction`, so this must NOT be a net emission factor.
    #: Published gross values cluster 0.10-0.23 (LPJ-WHyMe 0.10, ORCHIDEE-PCH4
    #: 0.104, LPJ-GUESS 14-site MCMC posterior 0.14, CLM5 0.20); ours sits at
    #: the top of that, defensible only because TRIPLEX-GHG is built on IBIS.
    #: Primary calibration target -- see docs/METHANE.md 5g.
    r_ch4: float = 0.23
    #: Production Q10. TRIPLEX-GHG calibrated 2.48 (range 1.6-4.5), and
    #: FLUXNET-CH4's annual-flux Q10 of 2.57 corroborates it independently.
    #: A *constant* Q10 is a known limitation: three models make it decline
    #: with temperature and Basso et al. (2026) show no single global value
    #: fits (W Siberia 1.4, Europe/N Canada 2.2). Second calibration target.
    q10p: float = 1.60
    #: Reference temperature for the Q10, K. CLM5's T_B = 22 degC.
    t_ref: float = 295.15

    # --- M2: oxidation -----------------------------------------------------
    #: Deepest the water table is allowed to sit *inside* a wetland, m.
    #: A wetland is wet by definition -- that is what makes it a wetland -- so
    #: its water table is shallow even when the surrounding grid cell is dry.
    #: See :func:`wetland_water_table` for why this exists.
    z_wt_max: float = 0.30
    #: Depth over which soil wetness is averaged to place the water table, m.
    #: The top half-metre: below that, moisture says little about where the
    #: surface water table sits.
    z_wt_ref: float = 0.50

    #: e-folding depth of decomposition with soil depth, m. Sets what
    #: fraction of Rh happens *below* the water table and is therefore
    #: anaerobic. CLM's `z_tau`, default 0.5 m, published range [0.1, 0.8]
    #: (Chinta et al. 2024, Table 2 -- one of the five parameters carrying
    #: >90 % of ELM's CH4 variance).
    z_tau: float = 0.50

    #: e-folding depth of the oxic zone, m. Methane produced below the water
    #: table must cross the aerobic soil above it, where methanotrophs consume
    #: most of it; a deeper water table means a thicker oxic zone and less
    #: escapes.
    #:
    #: This single parameter replaces the whole oxidation-and-transport
    #: apparatus -- Michaelis-Menten kinetics, O2 co-limitation, ebullition,
    #: aerenchyma, vertical diffusion. That is not a shortcut: Chadburn et al.
    #: (2020, GBC 34) substituted exactly this factor for all of CLM's
    #: machinery and the two agreed at **r = 0.94-0.98 (R^2 0.89-0.97)**,
    #: concluding "a more complex scheme used in recent Earth system models
    #: does not add predictive power".
    #:
    #: 0.15 m puts ~70 % oxidation at a typical wetland water table of ~0.18 m,
    #: matching Chadburn's fitted value and the observed 40-90 % range
    #: (Bridgham et al. 2013; Popp et al. 2000). WETMETH reports its emissions
    #: change +-37 % for a +-10 % change in this parameter, so it is both the
    #: most sensitive knob and the one gradients handle best.
    tau_oxid: float = 0.10


def layer_saturation(
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
) -> Float[Array, "npoi nsoilay"]:
    """Liquid-water saturation of each layer, in [0, 1].

    `wsoi` is the liquid fraction of the pore space *not* occupied by ice, so
    the liquid share of the whole pore space is `wsoi * (1 - wisoi)`. Ice
    counts as unavailable, not as water: a frozen layer holds no water a
    methanogen can live in, and treating it as saturated would put a water
    table at the surface across the entire winter Arctic.
    """
    return jnp.clip(wsoi * (1.0 - wisoi), 0.0, 1.0)


def water_table_depth(
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    params: MethaneParams | None = None,
) -> Float[Array, "npoi"]:
    """Water table depth below the surface, m, from IBIS's own moisture profile.

    IBIS has no water table: it carries a moisture profile and nothing else
    (docs/METHANE.md 2.4). This integrates the *unsaturated* thickness from the
    top down, so a column saturated to the surface gives 0 and a bone-dry
    column gives the full depth:

        deficit = sum_k hsoi[k] * max(0, sat_crit - sat[k]) / sat_crit

    This is a diagnostic, not a prognostic water table -- it says how much
    unsaturated pore space sits above the wet zone, which is what the anaerobic
    fraction in M2 needs. It deliberately does not model lateral flow, perched
    tables, or drainage, none of which IBIS represents.

    Returns
    -------
    (npoi,) depth in metres, clipped to [0, z_max].
    """
    p = params or MethaneParams()
    sat = layer_saturation(wsoi, wisoi)
    deficit = jnp.sum(
        hsoi[None, :] * jnp.maximum(0.0, p.sat_crit - sat) / p.sat_crit, axis=1)
    return jnp.clip(deficit, 0.0, p.z_max)


def anaerobic_fraction(
    zwt: Float[Array, "npoi"],
    hsoi: Float[Array, "nsoilay"],
) -> Float[Array, "npoi nsoilay"]:
    """Fraction of each layer lying below the water table, in [0, 1].

    Linear within a layer, so a water table halfway through a layer makes it
    half anaerobic rather than snapping to 0 or 1. M2 multiplies production by
    this: methane is produced only below the water table.
    """
    top = jnp.concatenate([jnp.zeros(1), jnp.cumsum(hsoi)[:-1]])   # layer tops
    bot = jnp.cumsum(hsoi)
    below = bot[None, :] - jnp.maximum(zwt[:, None], top[None, :])
    return jnp.clip(below / hsoi[None, :], 0.0, 1.0)


def wetland_flux_area(
    f_inund: Float[Array, "npoi"],
    fixed_fraction: float = 0.0,
) -> Float[Array, "npoi"]:
    """The cell fraction methane production applies over, in [0, 1].

    Kept separate from the reader so the model never assumes a particular
    observational product, and so a sensitivity test can hold extent fixed.
    """
    return jnp.clip(jnp.maximum(f_inund, fixed_fraction), 0.0, 1.0)


# ---------------------------------------------------------------------------
# M2 -- M-lite+ : production and oxidation
# ---------------------------------------------------------------------------


def anaerobic_mean_temperature(
    tsoi: Float[Array, "npoi nsoilay"],
    zwt: Float[Array, "npoi"],
    hsoi: Float[Array, "nsoilay"],
) -> Float[Array, "npoi"]:
    """Soil temperature averaged over the layers below the water table.

    Methanogenesis happens in the anaerobic zone, so the temperature that
    drives it is the temperature *there* -- not the surface, and not a whole-
    column mean. Weighted by each layer's anaerobic fraction (M1), which is
    linear within the layer straddling the water table.

    Falls back to the top-layer temperature where nothing is anaerobic; the
    flux is multiplied by zero in that case anyway, but a NaN would poison the
    global mean the way one Taymyr cell poisoned a 200-year spin-up.
    """
    w = anaerobic_fraction(zwt, hsoi) * hsoi[None, :]
    tot = jnp.sum(w, axis=1)
    return jnp.where(tot > 1e-12,
                     jnp.sum(w * tsoi, axis=1) / jnp.maximum(tot, 1e-12),
                     tsoi[:, 0])


def thawed_anaerobic_zone(
    tsoi: Float[Array, "npoi nsoilay"],
    zwt: Float[Array, "npoi"],
    hsoi: Float[Array, "nsoilay"],
    tmelt: float = 273.15,
) -> tuple[Float[Array, "npoi"], Float[Array, "npoi"]]:
    """Temperature of, and fraction of, the anaerobic zone that is thawed.

    Methanogenesis needs soil that is **both** below the water table **and**
    unfrozen. Averaging temperature over the whole anaerobic zone conflates
    those: in permafrost the zone runs from a water table near the surface down
    through metres of frozen ground, so the mean sits below freezing all year
    and a hard `T >= tmelt` gate switches the Arctic off entirely — even in
    July, when a half-metre active layer is thawed and emitting.

    That is what put 87 % of the first global run's emission in the tropics and
    1.5 % north of 60, against GCP's 70/8. Annual `Rh` alone predicts 70/26/3,
    so the substrate was never the problem (docs/METHANE.md 5j).

    Returns
    -------
    (t_thawed, f_thawed)
        Thickness-weighted mean temperature of the thawed anaerobic soil, and
        the fraction of the anaerobic zone that is thawed. Production scales
        with the second: a thin active layer emits from a thin active layer.
    """
    anaer = anaerobic_fraction(zwt, hsoi) * hsoi[None, :]
    thawed = anaer * (tsoi >= tmelt).astype(tsoi.dtype)
    tot_a = jnp.sum(anaer, axis=1)
    tot_t = jnp.sum(thawed, axis=1)
    t_thawed = jnp.where(tot_t > 1e-12,
                         jnp.sum(thawed * tsoi, axis=1) / jnp.maximum(tot_t, 1e-12),
                         tsoi[:, 0])
    f_thawed = jnp.where(tot_a > 1e-12, tot_t / jnp.maximum(tot_a, 1e-12), 0.0)
    return t_thawed, f_thawed


def wetland_water_table(
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    params: MethaneParams | None = None,
) -> Float[Array, "npoi"]:
    """Water table depth **inside the inundated fraction**, m.

    This is not the same quantity as :func:`water_table_depth`, and the
    difference matters enough that the first global M-lite+ run put 99.8 % of
    emissions in the tropics and exactly zero north of 60 degrees.

    `water_table_depth` integrates the unsaturated deficit of the whole soil
    column from IBIS's **grid-cell-mean** moisture. But a cell that is 15 %
    wetland is 85 % upland, and the mean describes the upland. Asking it where
    the water table sits inside a bog is asking the wrong array: measured on
    the spun-up state, it returned **2.57 m in 30-60N wetlands and 3.60 m
    north of 60**, which annihilates the flux through `exp(-zwt/tau)`.

    Two things fix it.

    **Bound the answer to a wetland-plausible range.** The water table inside a
    wetland lies between the surface and `z_wt_max`; relative wetness places it
    within that range rather than setting it in absolute terms.

    **Count ice as water.** A frozen bog is still saturated -- its water is
    simply solid. Using liquid saturation alone reported 0.001 in Arctic
    wetland cells and drove the water table to the bottom of the column. Total
    pore fill `wsoi*(1-wisoi) + wisoi` is IBIS's own `zwsoi` from `setsoi`, and
    it reads such a cell as the saturated ground it is. Production is gated off
    below freezing in :func:`gross_production`, so treating frozen soil as wet
    here does not manufacture winter emissions.
    """
    p = params or MethaneParams()
    theta = jnp.clip(wsoi * (1.0 - wisoi) + wisoi, 0.0, 1.0)   # liquid + ice
    # thickness weights over the top z_wt_ref of soil, the layer straddling
    # the cutoff counted in proportion
    bot = jnp.cumsum(hsoi)
    top = jnp.concatenate([jnp.zeros(1), bot[:-1]])
    w = jnp.clip(p.z_wt_ref - top, 0.0, hsoi)
    tot = jnp.maximum(jnp.sum(w), 1e-12)
    theta_top = jnp.sum(theta * w[None, :], axis=1) / tot
    wetness = jnp.clip(theta_top / jnp.maximum(p.sat_crit, 1e-6), 0.0, 1.0)
    return p.z_wt_max * (1.0 - wetness)


def anaerobic_substrate_fraction(
    zwt: Float[Array, "npoi"],
    params: MethaneParams | None = None,
) -> Float[Array, "npoi"]:
    """Fraction of heterotrophic respiration occurring below the water table.

    **The term M-lite+ was missing.** `r_CH4` is a CH4:CO2 ratio for anaerobic
    decomposition -- CLM defines its `f_ch4` as the "ratio between CH4 and CO2
    production *below the water table*". IBIS's `soilbgc` is a bulk model, so
    `tco2mic` is whole-column Rh, and applying the anaerobic ratio to all of it
    counts the aerobic soil above the water table as methanogenic.

    Decomposition falls off with depth as `exp(-z / z_tau)`, so the share below
    a water table at `zwt` is `exp(-zwt / z_tau)`. Not double-counting the
    moisture limitation already inside Rh: that sets how *fast* carbon
    decomposes, this sets what fraction does so *without oxygen*.
    """
    p = params or MethaneParams()
    return jnp.exp(-jnp.maximum(zwt, 0.0) / jnp.maximum(p.z_tau, 1e-6))


def gross_production(
    rh: Float[Array, "npoi"],
    t_anaer: Float[Array, "npoi"],
    params: MethaneParams | None = None,
) -> Float[Array, "npoi"]:
    """Gross CH4 production below the water table, kgC m-2 d-1.

    `rh` is heterotrophic respiration in kgC m-2 d-1 -- the carbon the soil is
    decomposing. Because `r_ch4` is a *molar* CH4:CO2 ratio and both gases
    carry one carbon atom, the ratio applies unchanged in carbon units.

    Production is clamped at zero below freezing: methanogenesis effectively
    stops in frozen soil, and the Q10 would otherwise keep producing methane
    at 250 K.
    """
    p = params or MethaneParams()
    q10 = p.q10p ** ((t_anaer - p.t_ref) / 10.0)
    unfrozen = (t_anaer >= 273.15).astype(rh.dtype)
    return jnp.maximum(rh, 0.0) * p.r_ch4 * q10 * unfrozen


def surviving_fraction(
    zwt: Float[Array, "npoi"],
    params: MethaneParams | None = None,
) -> Float[Array, "npoi"]:
    """Fraction of produced CH4 that escapes oxidation, in (0, 1].

    `exp(-zwt / tau_oxid)`. A water table at the surface leaves no aerobic
    soil to cross, so everything escapes; a deep water table means a thick
    oxic zone and almost nothing does.

    See :attr:`MethaneParams.tau_oxid` for why one parameter is enough.
    """
    p = params or MethaneParams()
    return jnp.exp(-jnp.maximum(zwt, 0.0) / jnp.maximum(p.tau_oxid, 1e-6))


def methane_flux(
    f_inund: Float[Array, "npoi"],
    rh: Float[Array, "npoi"],
    tsoi: Float[Array, "npoi nsoilay"],
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    params: MethaneParams | None = None,
    zwt_obs=None,
) -> Float[Array, "npoi"]:
    """Net wetland CH4 emission, kgC m-2 d-1. This is M-lite+ entire.

        F = f_inund * r_ch4 * Rh * Q10^((T_anaer - T_ref)/10) * exp(-zwt/tau)

    Gedney et al. (2004)'s production form with WETMETH's one-parameter
    oxidation. Four fitted parameters. Costs one exponential, one power and
    four multiplies per cell per day -- measured at 0.05 % of IBIS's runtime.

    Deliberately absent, each with evidence in docs/METHANE.md 5f: ebullition,
    plant-mediated transport, resolved vertical diffusion, Michaelis-Menten
    oxidation kinetics, O2 co-limitation. They change magnitude rather than
    dynamics, and magnitude is absorbed by parameters we calibrate anyway.

    Purely diagnostic: the carbon is *not* removed from the soil pool. At
    ~7 % of Rh that is a small term, and making it prognostic would couple the
    methane parameters to the carbon balance before either is calibrated.
    Revisit once M3 has evaluated the flux.
    """
    p = params or MethaneParams()
    # The water table INSIDE the wetland, not the column-mean deficit -- see
    # wetland_water_table for why that distinction is load-bearing.
    zwt = wetland_water_table(wsoi, wisoi, hsoi, p)
    if zwt_obs is not None:                        # an observed water-table depth (m below the surface, >= 0; NaN = use the model's), CALIBRATION.md 42
        zwt = jnp.where(jnp.isfinite(zwt_obs), jnp.clip(zwt_obs, 0.0, p.z_max), zwt)
    # Only THAWED anaerobic soil sets the production temperature -- but do not
    # multiply by the thawed fraction as well. `Rh` already collapses when the
    # soil freezes (Arctic wetland Rh is 0.027 gC/m2/d in January against 0.995
    # in July), so scaling by `f_thawed` would apply the freeze constraint a
    # second time. Measured: doing so contributed a 2.6x tropics-to-Arctic
    # suppression on top of Rh's own 3.4x. Same error as the Q10 double-count
    # (docs/METHANE.md 5j).
    t_thawed, _f_thawed = thawed_anaerobic_zone(tsoi, zwt, hsoi)
    # Only the share of Rh happening below the water table is anaerobic, and
    # only that share can make methane.
    rh_anaerobic = jnp.maximum(rh, 0.0) * anaerobic_substrate_fraction(zwt, p)
    prod = gross_production(rh_anaerobic, t_thawed, p)
    return wetland_flux_area(f_inund) * prod * surviving_fraction(zwt, p)


#: kgC m-2 d-1 -> mg CH4 m-2 d-1. 16.04 g CH4 per 12.011 g C.
KGC_TO_MG_CH4 = 1.0e6 * 16.04 / 12.011

#: kgC m-2 d-1 -> nmol CH4 m-2 s-1, the FLUXNET-CH4 convention.
KGC_TO_NMOL_CH4 = 1.0e12 / 12.011 / 86400.0


def to_mg_ch4(flux_kgc: Float[Array, "npoi"]) -> Float[Array, "npoi"]:
    """kgC m-2 d-1 -> mg CH4 m-2 d-1, for comparison with the literature."""
    return flux_kgc * KGC_TO_MG_CH4


def to_nmol_ch4(flux_kgc: Float[Array, "npoi"]) -> Float[Array, "npoi"]:
    """kgC m-2 d-1 -> nmol CH4 m-2 s-1, the FLUXNET-CH4 convention."""
    return flux_kgc * KGC_TO_NMOL_CH4


# ---------------------------------------------------------------------------
# State, so methane is part of the model rather than a script that reads it
# ---------------------------------------------------------------------------


class MethaneState(eqx.Module):
    """Wetland CH4 diagnostics carried in `IBISState.wetland`.

    Diagnostic, not prognostic: the methane does not feed back on the soil pools
    (docs/METHANE.md 5f). Its carbon has already left them as part of Rh; with
    `methane_carbon_partition` the driver books it as CH4 instead of CO2
    (`ch4_carbon_from_rh`, docs/METHANE.md 12).
    """

    #: Most recent monthly-mean emission, kgC m-2 d-1.
    flux: Float[Array, "npoi"]
    #: Running annual total, kgC m-2 yr-1. Cleared by `reset_annual`.
    annual: Float[Array, "npoi"]
    #: Water table depth used for the most recent flux, m -- kept because it is
    #: the term that broke first (docs/METHANE.md 5i) and is worth watching.
    zwt: Float[Array, "npoi"]


def create_methane_state(npoi: int) -> MethaneState:
    z = jnp.zeros((npoi,))
    return MethaneState(flux=z, annual=z, zwt=z)


def methane_step(
    mstate: MethaneState,
    f_inund: Float[Array, "npoi"],
    rh: Float[Array, "npoi"],
    tsoi: Float[Array, "npoi nsoilay"],
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    ndays: int,
    params: MethaneParams | None = None,
    zwt_obs=None,
) -> MethaneState:
    """Advance the methane diagnostics by one month.

    Monthly because that is WAD2M's resolution -- computing daily would imply a
    precision in `f_inund` that the forcing does not have.

    `rh` is heterotrophic respiration in kgC m-2 d-1 (`tco2mic * 12e-3 * 86400`).
    """
    p = params or MethaneParams()
    flux = methane_flux(f_inund, rh, tsoi, wsoi, wisoi, hsoi, p, zwt_obs=zwt_obs)
    zwt = wetland_water_table(wsoi, wisoi, hsoi, p)
    if zwt_obs is not None: zwt = jnp.where(jnp.isfinite(zwt_obs), jnp.clip(zwt_obs, 0.0, p.z_max), zwt)
    return MethaneState(flux=flux, annual=mstate.annual + flux * ndays, zwt=zwt)


def methane_step_from_mean(
    mstate: MethaneState,
    flux_month: Float[Array, "npoi"],
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    ndays: int,
    params: MethaneParams | None = None,
    zwt_obs=None,
) -> MethaneState:
    """Advance the diagnostics by one month from the month's MEAN emission (kgC m-2 d-1), as the hourly evaluation in
    the time-step loop produces it (`loops_gswp3` with `methane_hourly`; docs/METHANE.md 9). The water-table diagnostic
    stays the end-of-month value, as before."""
    p = params or MethaneParams()
    zwt = wetland_water_table(wsoi, wisoi, hsoi, p)
    if zwt_obs is not None: zwt = jnp.where(jnp.isfinite(zwt_obs), jnp.clip(zwt_obs, 0.0, p.z_max), zwt)
    return MethaneState(flux=flux_month, annual=mstate.annual + flux_month * ndays, zwt=zwt)


def reset_annual(mstate: MethaneState) -> MethaneState:
    """Clear the annual accumulator. Call at year end, after it is consumed."""
    return MethaneState(flux=mstate.flux, annual=jnp.zeros_like(mstate.annual),
                        zwt=mstate.zwt)


def scale_month(mstate: MethaneState, annual_before, f) -> MethaneState:
    """Scale a month run at unit inundated fraction to the fraction `f` (the learned wetland area, docs/METHANE.md 11):
    the month's flux and its share of the annual total, both linear in the fraction. `annual_before` is the annual
    total at the start of the month (None = zero)."""
    a0 = jnp.zeros_like(mstate.annual) if annual_before is None else annual_before
    f = jnp.asarray(f, jnp.float32)
    return MethaneState(flux=mstate.flux * f, annual=a0 + (mstate.annual - a0) * f, zwt=mstate.zwt)


def ch4_carbon_from_rh(annual, annual_before, rh, rh_before):
    """The month's CH4 carbon to book out of Rh as CO2 (docs/METHANE.md 12), kgC m-2: the month's net methane emission,
    never more than the month's Rh. All four are running annual totals, at the end and at the start of the month
    (`annual_before` None = zero)."""
    w0 = jnp.zeros_like(annual) if annual_before is None else annual_before
    return jnp.minimum(jnp.maximum(annual - w0, 0.0), jnp.maximum(rh - rh_before, 0.0))


# ---------------------------------------------------------------------------
# The validated production configuration
# ---------------------------------------------------------------------------

#: Biome-varying CH4:CO2 ratios, TRIPLEX-GHG (docs/METHANE.md 5k).
_R_CH4_BIOME = (0.15, 0.32, 0.18)          # tropical, temperate, boreal

#: Per-biome reference temperatures for the Q10 term, K.
#:
#: TRIPLEX-GHG's `r_ch4` values were each calibrated AT their own biome's
#: temperature, so referencing them all to one tropical `T_ref` applies the
#: biome temperature difference a second time -- the double-count diagnosed in
#: METHANE.md 5n. These are the production-weighted mean anaerobic-zone
#: temperatures of the corrected reference state (`both_fix`, 2026-09-02), so
#: the Q10 factor is ~1 in each biome's normal state and modulates only
#: departures from it.
#:
#: They are fixed constants rather than recomputed per run: deriving them from
#: the state being simulated would make the parameters depend on the answer.
_T_REF_BIOME = (299.6, 283.0, 277.9)       # tropical, temperate, boreal


def biome_methane_params(lat_deg, base: MethaneParams | None = None, bands: dict | None = None,
                         band_index=None) -> MethaneParams:
    """MethaneParams with per-biome `r_ch4` and `t_ref`, keyed on latitude.

    This is the configuration validated in docs/METHANE.md 5q -- **182 Tg/yr,
    inside the GMB 2025 range of 158 [119-203], shape error 11.0** -- and is
    what a production run should use. `MethaneParams()` alone is the "global"
    configuration, which gives 458 Tg on the same state.

    Latitude is a crude biome proxy, matching how TRIPLEX-GHG and the GCP both
    report. Note the asymmetry: only `lat >= 60` is treated as boreal, so
    southern high latitudes take temperate values. That follows the evaluation
    this was validated against, and matters little because there is almost no
    land south of 60 S outside Antarctica.
    """
    p = base or MethaneParams()
    lat = jnp.asarray(lat_deg)
    trop = (lat > -30.0) & (lat < 30.0)
    boreal = lat >= 60.0
    if band_index is not None:
        # IBIS 3.0 (CALIBRATION.md 45): the wetland types W01/W02/W03 by Köppen zone -- 0 tropical and subtropical,
        # 1 temperate, 2 boreal and arctic; -1 (unclassified cells) keeps the latitude rule
        bi = jnp.asarray(band_index)
        trop = jnp.where(bi >= 0, bi == 0, trop); boreal = jnp.where(bi >= 0, bi == 2, boreal)

    def band(vals):
        t, m, b = vals
        return jnp.where(trop, t, jnp.where(boreal, b, m))

    if bands is not None:
        # Per-band overrides of any field from a parameter file's `methane_bands` section
        # ({tropical|temperate|boreal: {field: value}}; CALIBRATION.md 38: the tower-class medians by band). A band
        # without an entry keeps `base`'s value; nothing else is changed, in particular t_ref stays at `base`'s value
        # unless given, because the tower fits held it fixed and their r_ch4 / Q10 absorb the band's temperature.
        fields = sorted({k for v in bands.values() for k in v}); bad = [k for k in fields if k not in MethaneParams.__dataclass_fields__]
        if bad: raise KeyError(f'unknown methane fields in methane_bands: {bad}')
        vals = [band(tuple(float(bands.get(b, {}).get(k, getattr(p, k))) for b in ('tropical', 'temperate', 'boreal'))) for k in fields]
        return eqx.tree_at(lambda q: tuple(getattr(q, k) for k in fields), p, tuple(vals))

    return eqx.tree_at(
        lambda q: (q.r_ch4, q.t_ref), p,
        (band(_R_CH4_BIOME), band(_T_REF_BIOME)),
    )
