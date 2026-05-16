"""Thermodynamic functions for IBIS-JAX.

Ported from mod_thermo_funcs.f90.  All functions are pure, JIT-compatible,
and differentiable.

Saturation vapour pressure uses Lowe (1977) polynomial approximation,
identical to the Fortran original.

Key insight from the Fortran: esat evaluates BOTH liquid and ice polynomials
simultaneously, using clamped temperature offsets so only one branch
contributes non-trivially.  The ice branch uses ti = clip(t-273.16, -60, 0),
which is NEGATIVE for T < freezing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ibis.constants import PhysicalConstants

# ---------------------------------------------------------------------------
# Lowe (1977) SVP polynomial coefficients.
# asat: over liquid water (T >= 273.16 K); arg = t - 273.16, clamped [0,100]
# bsat: over ice (T < 273.16 K); arg = t - 273.16, clamped [-60,0]
# Note: bsat[0] is the base constant, selected via merge/where.
# ---------------------------------------------------------------------------

# Liquid-water coefficients (ascending powers of tl)
_ASAT0 = 6.1078000
_ASAT1 = 4.4365185e-1
_ASAT2 = 1.4289458e-2
_ASAT3 = 2.6506485e-4
_ASAT4 = 3.0312404e-6
_ASAT5 = 2.0340809e-8
_ASAT6 = 6.1368209e-11

# Ice coefficients (ascending powers of ti, where ti is NEGATIVE)
_BSAT0 = 6.1091780
_BSAT1 = 5.0346990e-1
_BSAT2 = 1.8860134e-2
_BSAT3 = 4.1762237e-4
_BSAT4 = 5.8247203e-6
_BSAT5 = 4.8388032e-8
_BSAT6 = 1.8388269e-10

# d(SVP)/dT coefficients — liquid (csat) and ice (dsat)
_CSAT0 = 4.4381000e-1
_CSAT1 = 2.8570026e-2
_CSAT2 = 7.9380540e-4
_CSAT3 = 1.2152151e-5
_CSAT4 = 1.0365614e-7
_CSAT5 = 3.5324218e-10
_CSAT6 = -7.0902448e-13

_DSAT0 = 5.0303052e-1
_DSAT1 = 3.7732550e-2
_DSAT2 = 1.2679954e-3
_DSAT3 = 2.4775631e-5
_DSAT4 = 3.0056931e-7
_DSAT5 = 2.1585425e-9
_DSAT6 = 7.1310977e-12


def esat(t: jnp.ndarray) -> jnp.ndarray:
    """Saturation vapour pressure (Pa).

    Uses Lowe (1977) 6th-order polynomial, matching the Fortran exactly.
    Both liquid and ice branches are evaluated simultaneously; clamped
    temperature offsets ensure only the active branch contributes.

    Parameters
    ----------
    t : array
        Temperature (K).

    Returns
    -------
    array
        Saturation vapour pressure (Pa).
    """
    # Liquid branch: tl in [0, 100] (positive degrees above freezing)
    tl = jnp.clip(t - 273.16, 0.0, 100.0)
    # Ice branch: ti in [-60, 0] (negative degrees below freezing)
    ti = jnp.clip(t - 273.16, -60.0, 0.0)

    # Select base constant
    a0 = jnp.where(t >= 273.16, _ASAT0, _BSAT0)

    # Horner evaluation of both polynomials (excluding the base constant)
    liq_poly = tl * (_ASAT1 + tl * (_ASAT2 + tl * (_ASAT3
               + tl * (_ASAT4 + tl * (_ASAT5 + tl * _ASAT6)))))
    ice_poly = ti * (_BSAT1 + ti * (_BSAT2 + ti * (_BSAT3
               + ti * (_BSAT4 + ti * (_BSAT5 + ti * _BSAT6)))))

    # hPa -> Pa
    return 100.0 * (a0 + liq_poly + ice_poly)


def desat(t: jnp.ndarray) -> jnp.ndarray:
    """Derivative of saturation vapour pressure w.r.t. temperature (Pa/K).

    Uses Lowe (1977) d(SVP)/dT polynomials (csat and dsat series),
    NOT the analytical derivative of the esat polynomial.

    Parameters
    ----------
    t : array
        Temperature (K).

    Returns
    -------
    array
        d(esat)/dT (Pa/K).
    """
    tl = jnp.clip(t - 273.16, 0.0, 100.0)
    ti = jnp.clip(t - 273.16, -60.0, 0.0)

    c0 = jnp.where(t >= 273.16, _CSAT0, _DSAT0)

    liq_poly = tl * (_CSAT1 + tl * (_CSAT2 + tl * (_CSAT3
               + tl * (_CSAT4 + tl * (_CSAT5 + tl * _CSAT6)))))
    ice_poly = ti * (_DSAT1 + ti * (_DSAT2 + ti * (_DSAT3
               + ti * (_DSAT4 + ti * (_DSAT5 + ti * _DSAT6)))))

    return 100.0 * (c0 + liq_poly + ice_poly)


def qsat(esat_val: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Saturation specific humidity (kg/kg).

    Matches Fortran: qs = 0.622 * e / max(p - 0.378*e, 0.622*e)

    Parameters
    ----------
    esat_val : array
        Saturation vapour pressure (Pa).
    p : array
        Atmospheric pressure (Pa).

    Returns
    -------
    array
        Saturation specific humidity (kg/kg).
    """
    return 0.622 * esat_val / jnp.maximum(
        p - (1.0 - 0.622) * esat_val,
        0.622 * esat_val,
    )


def dqsat(t: jnp.ndarray, q1: jnp.ndarray,
          esat_val: jnp.ndarray | None = None) -> jnp.ndarray:
    """Derivative of saturation specific humidity w.r.t. temperature (K-1).

    Matches Fortran: dqs = desat(t) * q1 * (1 + q1*(1/0.622 - 1)) / esat(t)

    Parameters
    ----------
    t : array
        Temperature (K).
    q1 : array
        Saturation specific humidity (kg/kg).
    esat_val : array, optional
        Pre-computed esat(t).  Avoids a redundant esat call when the
        caller has already computed it (e.g. to derive q1 via qsat).

    Returns
    -------
    array
        d(qsat)/dT (K-1).
    """
    e = esat_val if esat_val is not None else esat(t)
    return desat(t) * q1 * (1.0 + q1 * (1.0 / 0.622 - 1.0)) / e


def hvapf(t: jnp.ndarray, tair: jnp.ndarray, tmelt: float,
          hvap: float, ch2o: float, cvap: float) -> jnp.ndarray:
    """Temperature-adjusted latent heat of vaporisation (J/kg).

    Matches Fortran: hvapf(t, tair) = hvap + cvap*(tair-tmelt) - ch2o*(t-tmelt)

    Parameters
    ----------
    t : array
        Surface temperature (K), e.g. soil or leaf.
    tair : array
        Air temperature (K).
    tmelt : float
        Freezing point (K).
    hvap, ch2o, cvap : float
        Physical constants.

    Returns
    -------
    array
        Adjusted latent heat (J/kg).
    """
    return hvap + cvap * (tair - tmelt) - ch2o * (t - tmelt)


def hsubf(t: jnp.ndarray, tair: jnp.ndarray, tmelt: float,
          hsub: float, cvap: float, cice: float) -> jnp.ndarray:
    """Temperature-adjusted latent heat of sublimation (J/kg).

    Matches Fortran: hsubf(t, tair) = hsub + cvap*(tair-tmelt) - cice*(t-tmelt)

    Parameters
    ----------
    t : array
        Surface temperature (K), e.g. soil or snow.
    tair : array
        Air temperature (K).
    tmelt : float
        Freezing point (K).
    hsub, cvap, cice : float
        Physical constants.

    Returns
    -------
    array
        Adjusted latent heat (J/kg).
    """
    return hsub + cvap * (tair - tmelt) - cice * (t - tmelt)
