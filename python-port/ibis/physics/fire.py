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


def fire_seasonlength(
    nfirestep: Float[Array, "npoi"],
    sdry: Float[Array, "npoi"],
    nytimes: Float[Array, ""],
    totlit: Float[Array, "npoi"],
    params: FireParams,
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
    fuel = jnp.minimum(1.0, totlit / LEGACY_LITTER_REF)
    burn = params.k_b * fseason ** params.p * dry_mean * fuel
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
