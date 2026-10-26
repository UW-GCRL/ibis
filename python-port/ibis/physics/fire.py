"""Fire disturbance schemes for IBIS-JAX.

This module owns the fire parameterisation. It exists so that new schemes can
be added without touching `vegetation_dynamics.dynaveg`, and so that the
original IBIS 2.6b4 behaviour stays reachable and provably unchanged.

Schemes
-------
``"legacy"``
    Exact transcription of IBIS 2.6b4 ``fortran/ibis_2.6b4/vegetation.f:591``
    (``subroutine fire``). Annual timestep, driven by an annual-mean fuel
    dryness factor and total litter load. This is the default and must never
    change: the 46-year F77<->JAX lockstep verification depends on it.

See ``docs/FIRE.md`` for the design of the planned replacement schemes and for
why the legacy scheme is being extended rather than replaced wholesale.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

LEGACY = "legacy"
SEASONLENGTH = "seasonlength"

#: Schemes accepted by :func:`compute_disturbf`. Extended as new stages land.
AVAILABLE_SCHEMES: tuple[str, ...] = (LEGACY, SEASONLENGTH)


# ---------------------------------------------------------------------------
# Legacy scheme (IBIS 2.6b4)
# ---------------------------------------------------------------------------

# Litter load at which the fuel term saturates (kgC/m2).
# F77: `min (1.0, totlit(i) / 0.200)` -- vegetation.f:609
LEGACY_LITTER_REF: float = 0.200

# Exponential decay coefficient converting burn index to disturbed fraction.
# F77: `1.0 - exp(-0.5 * burn)` -- vegetation.f:611
LEGACY_BURN_COEF: float = 0.5


def fire_legacy(
    firefac: Float[Array, "npoi"],
    totlit: Float[Array, "npoi"],
) -> Float[Array, "npoi"]:
    """IBIS 2.6b4 fire disturbance fraction.

    Transcribes ``fortran/ibis_2.6b4/vegetation.f:591`` exactly::

        burn        = firefac(i) * min (1.0, totlit(i) / 0.200)
        disturbf(i) = 1.0 - exp(-0.5 * burn)
        disturbf(i) = max (0.0, min (1.0, disturbf(i)))

    Note that ``totlit`` is *total* litter, including root litter which cannot
    physically burn. This is faithful to F77 and is deliberately preserved
    here; the correction to above-ground litter belongs to stage F2 in
    ``docs/FIRE.md``, not to this function.

    Parameters
    ----------
    firefac : (npoi,)
        Annual-mean fuel dryness factor, 0-1. Accumulated in F77
        ``stats.f:1180`` as a running mean of daily ``fueldry``.
    totlit : (npoi,)
        Total litter carbon (kgC/m2).

    Returns
    -------
    disturbf : (npoi,)
        Fire disturbance fraction, clipped to [0, 1].
    """
    burn = firefac * jnp.minimum(1.0, totlit / LEGACY_LITTER_REF)
    return jnp.minimum(1.0, jnp.maximum(0.0, 1.0 - jnp.exp(-LEGACY_BURN_COEF * burn)))


# ---------------------------------------------------------------------------
# Fuel dryness accumulator (F77 stats.f) -- restores defect 0
# ---------------------------------------------------------------------------

def fueldry_from_soil(
    wsoi_top: Float[Array, "npoi"],
    wisoi_top: Float[Array, "npoi"],
    swilt_top: Float[Array, "npoi"],
) -> Float[Array, "npoi"]:
    """Instantaneous fuel dryness from top-layer soil water.

    Verbatim transcription of ``fortran/ibis_2.6b4/stats.f:1174-1178``::

        water     = wisoi(i,1) + (1. - wisoi(i,1)) * wsoi(i,1)
        waterfrac = (water - swilt(i,1)) / (1. - swilt(i,1))
        fueldry   = max (0.0, min (1.0, -2.0 * (waterfrac - 0.5)))

    Note the water combination: `wsoi` is the liquid fraction of the *ice-free*
    pore space, so total water is ``wisoi + (1-wisoi)*wsoi`` -- not
    ``wsoi + wisoi``. See docs/FIRE.md 4.4c.

    Returns 0 (no dryness) when the soil is wet, 1 when it is at or below the
    midpoint between wilting point and saturation.
    """
    water = wisoi_top + (1.0 - wisoi_top) * wsoi_top
    waterfrac = (water - swilt_top) / (1.0 - swilt_top)
    return jnp.minimum(1.0, jnp.maximum(0.0, -2.0 * (waterfrac - 0.5)))


def update_firefac(
    firefac: Float[Array, "npoi"],
    nytimes: Float[Array, ""],
    wsoi_top: Float[Array, "npoi"],
    wisoi_top: Float[Array, "npoi"],
    swilt_top: Float[Array, "npoi"],
) -> Float[Array, "npoi"]:
    """Advance the running annual mean of fuel dryness by one timestep.

    Verbatim transcription of ``fortran/ibis_2.6b4/stats.f:1180``::

        rwork      = 1. / float(nytimes)
        firefac(i) = ((nytimes-1) * firefac(i) + fueldry) * rwork

    ``nytimes`` is the **1-based count of timesteps elapsed in the current
    year**, matching F77's counter in ``sumyear``, which is reset to 0 at
    ``istep==1 .and. iday==1 .and. imonth==1`` and incremented once per
    timestep (``main.f:437`` calls ``sumyear`` every timestep, so this is a
    per-timestep mean, *not* a per-day mean -- see docs/FIRE.md 4.4c).

    At ``nytimes == 1`` this returns ``fueldry`` exactly, which is how F77
    effectively re-initialises the mean each year without clearing ``firefac``.

    This function did not exist in the JAX port before 2026-08-18. Its absence
    left ``firefac`` pinned at its zero initial value, so ``fire()`` returned
    ``disturbf = 0`` in every production run -- defect 0 in docs/FIRE.md.
    """
    fueldry = fueldry_from_soil(wsoi_top, wisoi_top, swilt_top)
    n = jnp.maximum(nytimes, 1.0)
    return ((n - 1.0) * firefac + fueldry) / n


# ---------------------------------------------------------------------------
# F1: fire season length (docs/FIRE.md stage F1)
# ---------------------------------------------------------------------------

#: Fire mortality by IBIS PFT (params.veg lines 27-38 give the ordering):
#:
#:   0  tropical broadleaf evergreen tree     6  boreal broadleaf cold-decid. tree
#:   1  tropical broadleaf drought-decid.     7  boreal conifer cold-decid. tree
#:   2  warm-temperate broadleaf evergreen    8  evergreen shrub
#:   3  temperate conifer evergreen           9  cold-deciduous shrub
#:   4  temperate broadleaf cold-deciduous   10  warm (C4) grass
#:   5  boreal conifer evergreen             11  cool (C3) grass
#:
#: These are **literature-informed priors, not measured values**, and are
#: calibration targets for F6. The ordering of magnitudes is the defensible
#: part; the exact numbers are not.
#:
#: Reasoning: grasses are entirely top-killed by any fire that passes, so 1.0.
#: Shrubs are nearly so. Boreal conifers carry thin bark and support
#: stand-replacing crown fire, so mortality is high. Temperate conifers and
#: tropical drought-deciduous trees are the fire-adapted end -- thick bark,
#: frequent low-intensity surface fire -- so they sit lowest. Tropical
#: evergreen forest burns rarely but destructively when it does.
#:
#: Precedent: FATES resolves this as crown scorch plus cambial damage scaled by
#: a per-PFT `bark_scaler` (SFMainMod.F90:570 CalculatePostFireMortality ->
#: CambialMortality). We collapse that to one coefficient per PFT, which is the
#: light-model version of the same idea.
#: Combustion completeness by pool -- the fraction of *killed* carbon that a
#: fire actually volatilises, as opposed to killing and leaving behind. Wood
#: burns far less completely than foliage: a crown fire kills a tree in minutes
#: but consumes only its fine branches, and the bole becomes coarse woody
#: debris that decomposes over decades.
#:
#: These are the single definition used by BOTH the emission diagnostic
#: (`ibis.bgc.fire_emissions`) and the carbon balance
#: (`ibis.physics.vegetation_dynamics`). Keeping two copies is exactly how the
#: two came to disagree by a factor of five -- see docs/FIRE.md.
CC_LEAF = 0.90
CC_WOOD = 0.20
CC_LITTER = 0.50

DEFAULT_PFT_MORTALITY: tuple[float, ...] = (
    0.50,  #  0 tropical broadleaf evergreen
    0.35,  #  1 tropical broadleaf drought-deciduous (fire-adapted savanna tree)
    0.50,  #  2 warm-temperate broadleaf evergreen
    0.35,  #  3 temperate conifer evergreen (thick bark, surface-fire adapted)
    0.50,  #  4 temperate broadleaf cold-deciduous
    0.80,  #  5 boreal conifer evergreen (thin bark, crown fire)
    0.70,  #  6 boreal broadleaf cold-deciduous
    0.75,  #  7 boreal conifer cold-deciduous (larch)
    0.85,  #  8 evergreen shrub
    0.85,  #  9 cold-deciduous shrub
    1.00,  # 10 warm (C4) grass -- fully top-killed
    1.00,  # 11 cool (C3) grass
)


class FireParams(eqx.Module):
    """Tunable parameters for the season-length scheme.

    All are exposed for the gradient-based calibration planned in F6.
    """

    #: Dryness above which a timestep counts as fire-prone. `fueldry` is in
    #: [0, 1]; 0.5 is the midpoint between wilting point and the wet limit.
    d_crit: float = 0.5
    #: Overall burned-fraction scale. 0.5 reproduces the legacy coefficient.
    k_b: float = 0.5
    #: Exponent on fire-season fraction. 1.0 makes the scheme reduce exactly
    #: to legacy when `d_crit` admits every timestep (see tests).
    p: float = 1.0

    # --- F2: fine fuel -----------------------------------------------------
    #: Lower fuel threshold (kgC m-2). CTSM CNFireBaseMod.F90:55 gives
    #: lfuel = 75 gC m-2; IBIS carries litter in kg_C m-2 (comveg.h:132,137),
    #: so CLM's value is CONVERTED, not copied. Mixing the two is a 1000x error.
    lfuel: float = 0.075
    #: Upper fuel threshold (kgC m-2). CTSM CNFireBaseMod.F90:56 = 650 gC m-2.
    ufuel: float = 0.650
    #: Fraction of live woody-PFT leaf carbon acting as fine fuel (PFTs 0-9).
    w_leaf_woody: float = 0.3
    #: Same for grass PFTs (10-11). Grass cures and carries fire readily, so
    #: its leaf carbon counts for more than a tree's.
    w_leaf_grass: float = 1.0
    #: Select the F2 fuel term. False keeps the legacy min(1, totlit/0.2).
    use_fine_fuel: bool = eqx.field(static=True, default=False)

    # --- F3: human ignition and suppression --------------------------------
    #: Baseline (non-human) ignition factor, dimensionless. At popd = 0 the
    #: ignition term is `ig0`, so 1.0 leaves the scheme unchanged where nobody
    #: lives. IBIS carries no lightning data, so natural ignition is folded
    #: into this constant rather than resolved (docs/FIRE.md 4.2).
    ig0: float = 1.0
    #: Rate at which human ignition saturates, per ind/km2.
    a_ig: float = 0.01
    #: Ceiling on the ignition factor. Human ignition saturates rather than
    #: growing without bound -- see :func:`human_ignition`.
    ign_max: float = 3.0
    #: Select the F3 human term. False leaves fire purely climate-driven.
    use_human: bool = eqx.field(static=True, default=False)

    # --- F4: PFT-specific fire mortality -----------------------------------
    #: Fraction of a PFT's live biomass killed per unit burned fraction, one
    #: value per IBIS PFT (see DEFAULT_PFT_MORTALITY for the ordering and the
    #: reasoning). 1.0 = fully killed, lower = more fire-resistant.
    mort: Float[Array, "npft"] = eqx.field(
        default_factory=lambda: jnp.asarray(DEFAULT_PFT_MORTALITY)
    )
    #: Select the F4 per-PFT mortality. False keeps IBIS's uniform mortality,
    #: which kills a fire-adapted savanna tree and a thin-barked boreal conifer
    #: at exactly the same rate and makes the fire->vegetation feedback -- the
    #: whole reason fire belongs in a DGVM -- wrong in a way no calibration can
    #: repair. Default True since 2026-08-26: the v8 spin-up put burned area
    #: within 8 % of GFED5 while uniform mortality was still on, which left
    #: mortality as the largest remaining structural bias. Only applies under
    #: "seasonlength"; "legacy" is untouched.
    use_pft_mortality: bool = eqx.field(static=True, default=True)

    # --- F1b: VPD-based fire danger ----------------------------------------
    #: VPD (kPa) at which flammability saturates.
    vpd_ref: float = 3.0
    #: Flammability above which a timestep counts as fire-prone. Compare
    #: `d_crit`, which plays the same role for the soil-moisture path.
    d_crit_vpd: float = 0.33
    #: Total LAI at which vegetation density saturates. Coupling flammability
    #: to vegetation density is what lifted rank skill from 0.22 to 0.59 --
    #: see :func:`vegetation_density`.
    lai_ref_vd: float = 4.0
    #: Use the VPD accumulators instead of the soil-moisture ones. Default
    #: True: measured in-model against GFED5 2001-2005 over 88 484 cells, the
    #: VPD path ranks +0.630 against the soil-moisture path's +0.315, and puts
    #: fire at |lat| 14.2 rather than 39.8 against an observed 10.7. This only
    #: applies under the "seasonlength" scheme; "legacy" is untouched.
    use_vpd_danger: bool = eqx.field(static=True, default=True)

    # --- F7: closed-canopy damping (CALIBRATION.md 59) ----------------------
    #: Upper-canopy cover `fu` at which the closed-forest damping halves the
    #: burn. Savanna fire spreads through a continuous grass layer and stops
    #: where the tree canopy closes and the fuel under it stays moist (the
    #: fire-spread threshold of Staver et al. 2011, Archibald et al. 2009).
    fu_half: float = 0.6
    #: Width of the logistic in `fu` (0.08: from 0.9 to 0.1 of the undamped
    #: burn over fu_half -/+ 0.18).
    fu_width: float = 0.08
    #: Multiply the burn by :func:`canopy_damping`. Without it the fuel and
    #: vegetation-density terms are saturated in rainforest and cancel the
    #: weather contrast: the humid Amazon, Congo and Southeast Asia burned
    #: 7-10 % a year against 1-4 % in GFED5, African savannas 10 % against 42 %.
    #: Default False keeps every earlier run bit-identical.
    use_canopy_damping: bool = eqx.field(static=True, default=False)
    #: Weight the damping by the broadleaf share of the upper-canopy leaf area
    #: (positions 0, 1, 2, 4, 6), so closed conifer stands keep their fire. A
    #: closed broadleaf canopy holds moist, compact litter and does not carry
    #: surface fire (tropical rainforest; deciduous stands as fire breaks in the
    #: boreal zone), while closed conifer forest burns as crown fire. Damping
    #: every closed canopy took boreal fire from 1.27 to ~0.5 x GFED5 in the
    #: 2003-2005 offline fit (CALIBRATION.md 59).
    damp_broadleaf_only: bool = eqx.field(static=True, default=True)
    #: Köppen zones whose climate supports a closed broadleaf canopy whatever the cell's current cover (CALIBRATION.md
    #: 65): there the damping is applied as for a closed canopy (cover at the 0.975 cap). Without it a cell that starts
    #: open burns, stays open and never closes -- the fire trap of the cold start (8 % of the humid Amazon carried 52 %
    #: of its fire at 15 % a year). Precedent: CLM5 excludes natural fire from tropical closed forest altogether; the
    #: forest-savanna bistability sits at 1000-2000 mm (Staver 2011), below the Af / Am climate. Names from
    #: pft_types.ZONES, e.g. ('A_humid',); `apply_types` turns them into `closed_canopy_cell`.
    closed_canopy_zones: tuple = eqx.field(static=True, default=())
    #: The upper-canopy cover the damping is evaluated at in those cells: 0.975 = dynaveg's cover cap, the floor of
    #: the logistic (0.08 at width 0.05); a lower value leaves more fire in the humid zone (0.93 -> 0.17, 0.90 -> 0.27)
    #: at the price of a weaker guard against the trap. Fittable with the fire refit.
    closed_canopy_fu: float = 0.975
    #: (npoi,) 1 where the cell's zone is in `closed_canopy_zones`; None until the types are applied.
    closed_canopy_cell: Float[Array, "npoi"] | None = None

    # --- combustion completeness -------------------------------------------
    #: Fraction of *killed* leaf / wood / above-ground litter carbon that is
    #: actually combusted. The remainder is killed but not burned, and becomes
    #: litter -- standing dead wood decomposes over decades, it does not
    #: volatilise in a fire. See :data:`CC_LEAF` for provenance.
    cc_leaf: float = CC_LEAF
    cc_wood: float = CC_WOOD
    cc_litter: float = CC_LITTER
    #: Route the non-combusted share of fire-killed biomass to litter instead
    #: of emitting all of it. IBIS 2.6b4 `vegetation.f:433,481` adds the whole
    #: killed pool to `cdisturb` and subtracts it from NEE, which was harmless
    #: while `disturbf` was a constant 0.005 but is not once the scheme
    #: produces 30-50 % burned fractions: wood is ~97 % of live biomass in the
    #: spun-up global state, and emitting 100 % of it contradicts this module's
    #: own `cc_wood = 0.20`. Only applies under "seasonlength".
    use_combustion_completeness: bool = eqx.field(static=True, default=True)


def default_fire_params() -> FireParams:
    return FireParams()


def update_fire_accumulators(
    nfirestep: Float[Array, "npoi"],
    sdry: Float[Array, "npoi"],
    wsoi_top: Float[Array, "npoi"],
    wisoi_top: Float[Array, "npoi"],
    swilt_top: Float[Array, "npoi"],
    d_crit: float,
) -> tuple[Float[Array, "npoi"], Float[Array, "npoi"]]:
    """Advance the fire-season accumulators by one timestep.

    Counts timesteps whose fuel dryness exceeds ``d_crit`` and sums their
    dryness. Sampling every timestep (rather than once per day) measures the
    **fraction of the year spent in fire-prone conditions** directly, which is
    what "fire season length" means. It also sidesteps the trap noted in
    docs/FIRE.md 4.4c: F77's `sumyear` runs every timestep, so a naive
    "fire day" counter hooked there would over-count by ~48x.

    Returns
    -------
    (nfirestep, sdry)
        Updated counters.
    """
    fueldry = fueldry_from_soil(wsoi_top, wisoi_top, swilt_top)
    is_fire = (fueldry > d_crit).astype(fueldry.dtype)
    return nfirestep + is_fire, sdry + fueldry * is_fire


def legacy_fuel_fraction(totlit: Float[Array, "npoi"]) -> Float[Array, "npoi"]:
    """Legacy fuel availability: `min(1, totlit/0.2)` from vegetation.f:609.

    Uses `totlit`, which `biogeochem.f:924` defines as *"total above and
    belowground litter"* -- it includes the three `clitr*` root pools, which
    cannot burn. Preserved verbatim for the legacy path; :func:`fine_fuel_fraction`
    is the corrected version.
    """
    return jnp.minimum(1.0, totlit / LEGACY_LITTER_REF)


def fine_fuel_fraction(
    totalit: Float[Array, "npoi"],
    cbiol: Float[Array, "npoi npft"],
    params: FireParams,
) -> Float[Array, "npoi"]:
    """Fuel availability from above-ground litter plus live fine fuel.

    Two corrections over :func:`legacy_fuel_fraction`, both from the docs/FIRE.md
    audits:

    1. **`totalit`, not `totlit`.** `totalit` (biogeochem.f:930) is the
       above-ground litter sum; `totlit` (`:924`) also carries root litter,
       which cannot burn. LPJmL corroborates: `firepft.c` burns only
       `litter->item[p].agtop.*`.
    2. **No `frac` weighting.** `cbiol` is *already per grid-cell area* --
       `totlaiu = sum_j plai(i,j)` sums PFTs unweighted (vegetation.f:494) and
       vegetation.f:551 *divides* by `fu` to reach canopy area. `frac` is a
       **canopy** fraction (comveg.h:197), not a grid fraction; multiplying by
       it would under-count fuel.

    Live leaf carbon is split woody (PFTs 0-9) / grass (10-11), following IBIS's
    own grouping in `vegmap` where shrubs are 8-9 and grasses 10-11.

    The ramp replaces legacy's kink and is CLM's `fb`
    (CNFireLi2014Mod.F90:582), with thresholds converted to kgC m-2.
    """
    woody = jnp.sum(cbiol[:, :10], axis=1)
    grass = jnp.sum(cbiol[:, 10:], axis=1)
    fine = totalit + params.w_leaf_woody * woody + params.w_leaf_grass * grass
    span = jnp.maximum(params.ufuel - params.lfuel, 1e-12)
    return jnp.clip((fine - params.lfuel) / span, 0.0, 1.0)


def human_suppression(
    popd: Float[Array, "npoi"],
    params: FireParams | None = None,
) -> Float[Array, "npoi"]:
    """Fire suppression by people. **Verbatim CLM.**

    `lpop = 0.2 + 0.8 * exp(-pi * (hdm/450)**0.5)`
    -- CTSM `CNFireLi2014Mod.F90:439-440`, constants included.

    Bounded to [0.2, 1.0] by construction: dense population suppresses at most
    80% of fire, and empty land is unsuppressed. That built-in floor is why the
    ad-hoc `f_min` in the first draft of the design was dropped (docs/FIRE.md
    4.4b) -- copying CLM's expression removed two free parameters.

    `popd` is in ind/km2, matching the units of the CLM population stream
    (`hdm`, docs at inputs/fire/README.md).
    """
    del params  # no free parameters: the constants are CLM's
    return 0.2 + 0.8 * jnp.exp(-jnp.pi * jnp.sqrt(jnp.maximum(popd, 0.0) / 450.0))


def human_ignition(
    popd: Float[Array, "npoi"],
    params: FireParams,
) -> Float[Array, "npoi"]:
    """Ignition factor: a background term plus a *saturating* human term.

    `ign = ig0 + (ign_max - ig0) * (1 - exp(-a_ig * popd))`

    bounded in [`ig0`, `ign_max`]. At `popd = 0` it is `ig0` -- fire still
    happens where nobody lives, because lightning does.

    **Deliberate deviation from CLM.** CLM's human ignition is *linear* in
    population density (`pot_hmn_ign_counts_alpha`, CNFireBaseMod.F90:64). That
    is safe there because CLM's fuel term collapses toward zero in dense urban
    cells, capping the product. IBIS has no such urban fuel collapse, so a
    linear term is unbounded: at 10 000 ind/km2 it produced a 7x *increase* in
    burning, which is the opposite of the observed decline at high density.
    Saturating the term keeps the population response hump-shaped -- rising
    where people provide ignition, falling where they provide suppression --
    which is what burned-area observations show.

    `a_ig` and `ign_max` are calibration targets (F6), not borrowed constants.
    Do not cite them as CLM's.

    IBIS carries no lightning data, so natural ignition is the constant `ig0`
    rather than a resolved field. A 0.5-degree lightning climatology sits in
    the same CLM stream directory if we later want it, at the cost of a second
    input dataset (docs/FIRE.md 0).
    """
    x = params.a_ig * jnp.maximum(popd, 0.0)
    return params.ig0 + (params.ign_max - params.ig0) * (1.0 - jnp.exp(-x))


def vpd_from_state(
    ta: Float[Array, "npoi"],
    qa: Float[Array, "npoi"],
    psurf: Float[Array, "npoi"],
) -> Float[Array, "npoi"]:
    """Vapour pressure deficit (kPa) from temperature, humidity and pressure.

    Uses IBIS's own `thermo.esat`, which is a transcription of the F77
    statement functions in `comsat.h` and verified bit-exact, rather than a
    separate Tetens formula -- the model should not carry two different
    saturation curves.

    Computed from `qa`/`psurf` rather than `AtmState.rh` because the GSWP3
    driver populates the former and leaves `rh` at its initial value.
    """
    from ibis.physics.thermo import esat
    e_sat = esat(ta)                                        # Pa
    e_act = qa * psurf / (0.622 + 0.378 * qa)               # Pa
    return jnp.maximum(e_sat - e_act, 0.0) / 1000.0         # kPa


def flammability_vpd(
    ta: Float[Array, "npoi"],
    qa: Float[Array, "npoi"],
    psurf: Float[Array, "npoi"],
    params: FireParams,
) -> Float[Array, "npoi"]:
    """Atmospheric flammability in [0, 1], after Drüke et al. (2019).

    **Why this replaces soil-moisture dryness.** Measured on 62 176 cells
    against GFED5 2001-2005, as parameter-free Spearman rank correlation
    (monotonic parameters cannot change rank, so this is driver skill, not
    calibration):

        soil-moisture burn index         +0.258
        VPD mean                         +0.217
        VPD x vegetation density         **+0.590**

    and the latitude of the most fire-prone cells moves from 39.8 deg (wrong --
    the subtropical dry belt) to 11.6 deg, against GFED5's 10.7 deg.

    Soil moisture fails the way Drüke et al. describe the Nesterov index: "a
    strong but very localized maximum for high temperatures and low humidity".
    Perennially dry subtropical cells score high all year but have little to
    burn; tropical savannas cure seasonally and burn. VPD separates them once
    multiplied by vegetation density.
    """
    return jnp.clip(vpd_from_state(ta, qa, psurf) / params.vpd_ref, 0.0, 1.0)


def vegetation_density(
    plai: Float[Array, "npoi npft"], params: FireParams
) -> Float[Array, "npoi"]:
    """Vegetation density in [0, 1] from total leaf area.

    Drüke et al.'s `VD` term. On its own VPD gives rank skill +0.22; multiplied
    by this it gives +0.59. Note it is largely redundant with the F2 fuel term
    (+0.604 with both), so calibration should not be given free rein over both.
    """
    return jnp.clip(jnp.sum(jnp.maximum(plai, 0.0), axis=1) / params.lai_ref_vd,
                    0.0, 1.0)


#: dynaveg's upper-canopy cover cap; the default of FireParams.closed_canopy_fu.
CLOSED_CANOPY_FU = 0.975

#: Upper-canopy broadleaf positions (0-based): tropical evergreen, tropical
#: drought-deciduous, warm-temperate evergreen, temperate and boreal deciduous.
#: Zone types keep the leaf form of their position (pft_types.zone_slots).
BROADLEAF_POSITIONS = (0, 1, 2, 4, 6)


def broadleaf_share(plai: Float[Array, "npoi npft"]) -> Float[Array, "npoi"]:
    """Broadleaf share of the upper-canopy (positions 0-7) leaf area; 0 where there is no tree leaf (a leafless stand
    keeps its wood and cover for decades and must not be damped as a closed broadleaf canopy)."""
    up = jnp.maximum(plai[:, :8], 0.0)
    tot = jnp.sum(up, axis=1)
    bl = jnp.sum(up[:, jnp.asarray(BROADLEAF_POSITIONS)], axis=1)
    return jnp.where(tot > 1e-9, bl / jnp.maximum(tot, 1e-9), 0.0)   # no tree leaf: no closed broadleaf canopy to damp


def canopy_damping(
    fu: Float[Array, "npoi"], params: FireParams,
    plai: Float[Array, "npoi npft"] | None = None,
    closed: Float[Array, "npoi"] | None = None,
) -> Float[Array, "npoi"]:
    """Closed-canopy damping in (0, 1): 1 under open canopies, ~0 under closed broadleaf forest.

    A logistic in the upper-canopy cover `fu`, 0.5 at ``params.fu_half``. With
    ``params.damp_broadleaf_only`` (default) and `plai` given, it acts on the
    broadleaf share of the canopy only: d = 1 - b (1 - logistic), so a closed
    conifer stand keeps its fire. Applied annually with vegetation density
    (F7, CALIBRATION.md 59).
    """
    if closed is not None:                      # closed-canopy climate: the cover the climate supports, not today's
        fu = jnp.where(closed > 0.5, params.closed_canopy_fu, fu)
    d = 1.0 / (1.0 + jnp.exp((jnp.clip(fu, 0.0, 1.0) - params.fu_half)
                            / jnp.maximum(params.fu_width, 1e-6)))
    if params.damp_broadleaf_only and plai is not None:
        d = 1.0 - broadleaf_share(plai) * (1.0 - d)
    return d


def zero_fire_accumulators(veg):
    """The vegetation state with the annual fire counters cleared.

    F77 clears `nytimes` at the first step of each year (stats.f:956) and the
    season-length counters follow it. `loops.run_year` did this; the GSWP3
    production driver (`scripts/run_global.py`) did not, so before 2026-10-25
    every year's fire read the counters accumulated since the first spin-up
    year (CALIBRATION.md 59).
    """
    return eqx.tree_at(
        lambda v: (v.nytimes, v.nfirestep, v.sdry, v.nfirestep_vpd, v.sdry_vpd),
        veg,
        (jnp.zeros_like(veg.nytimes), jnp.zeros_like(veg.nfirestep),
         jnp.zeros_like(veg.sdry), jnp.zeros_like(veg.nfirestep_vpd),
         jnp.zeros_like(veg.sdry_vpd)),
    )


def update_fire_accumulators_vpd(
    nfirestep_vpd: Float[Array, "npoi"],
    sdry_vpd: Float[Array, "npoi"],
    ta: Float[Array, "npoi"],
    qa: Float[Array, "npoi"],
    psurf: Float[Array, "npoi"],
    params: FireParams,
) -> tuple[Float[Array, "npoi"], Float[Array, "npoi"]]:
    """Advance the VPD fire-season accumulators by one timestep.

    Deliberately kept purely meteorological: vegetation density is applied
    annually in :func:`fire_seasonlength`, not mixed into the weather counter.
    VD varies slowly, so the product is nearly unchanged either way, and this
    keeps weather and vegetation separable.

    Accumulated alongside the soil-moisture pair rather than replacing it, so a
    single run yields both and the A/B comparison costs no extra simulation.
    """
    f = flammability_vpd(ta, qa, psurf, params)
    is_fire = (f > params.d_crit_vpd).astype(f.dtype)
    return nfirestep_vpd + is_fire, sdry_vpd + f * is_fire


def fire_seasonlength(
    nfirestep: Float[Array, "npoi"],
    sdry: Float[Array, "npoi"],
    nytimes: Float[Array, ""],
    totlit: Float[Array, "npoi"],
    params: FireParams,
    fuel: Float[Array, "npoi"] | None = None,
    popd: Float[Array, "npoi"] | None = None,
    vd: Float[Array, "npoi"] | None = None,
) -> Float[Array, "npoi"]:
    """Fire disturbance fraction from fire-season length and severity.

    Replaces the legacy annual *mean* of dryness. The mean cannot distinguish
    a cell with a severe dry season from one that is mildly dry all year --
    they average identically -- yet only the first burns. Fire is
    threshold-driven, so averaging is the wrong operator. This is the insight
    Glob-FIRM (Thonicke et al. 2001) was built on; the closure below is IBIS's
    own exponential, not Glob-FIRM's equation (which we have not read -- see
    docs/FIRE.md 4.4b).

    ``fseason``  fraction of the year spent above ``d_crit``
    ``dry_mean`` mean dryness *within* the fire season (severity)

    Reduces exactly to :func:`fire_legacy` when ``d_crit`` admits every
    timestep and ``p == 1``: then ``fseason == 1`` and ``dry_mean`` is the
    all-timestep mean, which is precisely ``firefac``.

    The fuel term is unchanged from legacy at this stage; F2 replaces it.
    """
    fseason = nfirestep / jnp.maximum(nytimes, 1.0)
    dry_mean = sdry / jnp.maximum(nfirestep, 1.0)
    fuel = legacy_fuel_fraction(totlit) if fuel is None else fuel
    burn = params.k_b * fseason ** params.p * dry_mean * fuel
    if vd is not None:
        burn = burn * vd
    if popd is not None:
        burn = burn * human_ignition(popd, params) * human_suppression(popd)
    return jnp.minimum(1.0, jnp.maximum(0.0, 1.0 - jnp.exp(-burn)))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compute_disturbf(
    scheme: str,
    *,
    firefac: Float[Array, "npoi"],
    totlit: Float[Array, "npoi"],
    nfirestep: Float[Array, "npoi"] | None = None,
    sdry: Float[Array, "npoi"] | None = None,
    nytimes: Float[Array, ""] | None = None,
    params: "FireParams | None" = None,
) -> Float[Array, "npoi"]:
    """Dispatch to the configured fire scheme.

    ``scheme`` is a static Python string taken from ``RunConfig.fire_scheme``;
    it is never a traced value, so this branch is resolved at trace time and
    costs nothing under ``jit``.

    Parameters
    ----------
    scheme : str
        One of :data:`AVAILABLE_SCHEMES`.
    firefac, totlit
        See :func:`fire_legacy`.

    Returns
    -------
    disturbf : (npoi,)
        Fire disturbance fraction, clipped to [0, 1].

    Raises
    ------
    ValueError
        If ``scheme`` is not a known scheme name.
    """
    if scheme == LEGACY:
        return fire_legacy(firefac, totlit)
    if scheme == SEASONLENGTH:
        if nfirestep is None or sdry is None or nytimes is None:
            raise ValueError(
                "fire_scheme='seasonlength' needs nfirestep, sdry and nytimes"
            )
        return fire_seasonlength(
            nfirestep, sdry, nytimes, totlit, params or default_fire_params()
        )
    raise ValueError(
        f"unknown fire_scheme {scheme!r}; available: {AVAILABLE_SCHEMES}"
    )
