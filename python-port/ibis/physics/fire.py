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

import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

LEGACY = "legacy"

#: Schemes accepted by :func:`compute_disturbf`. Extended as new stages land.
AVAILABLE_SCHEMES: tuple[str, ...] = (LEGACY,)


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
# Dispatcher
# ---------------------------------------------------------------------------

def compute_disturbf(
    scheme: str,
    *,
    firefac: Float[Array, "npoi"],
    totlit: Float[Array, "npoi"],
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
    raise ValueError(
        f"unknown fire_scheme {scheme!r}; available: {AVAILABLE_SCHEMES}"
    )
