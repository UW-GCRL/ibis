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
    #: Select the F4 per-PFT mortality. False keeps IBIS's uniform mortality.
    use_pft_mortality: bool = eqx.field(static=True, default=False)


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


def fire_seasonlength(
    nfirestep: Float[Array, "npoi"],
    sdry: Float[Array, "npoi"],
    nytimes: Float[Array, ""],
    totlit: Float[Array, "npoi"],
    params: FireParams,
    fuel: Float[Array, "npoi"] | None = None,
    popd: Float[Array, "npoi"] | None = None,
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
