"""Biogenic VOC emissions (MEGAN 2.1) for IBIS-JAX.

Implements isoprene (light + temperature dependent) and monoterpene
(temperature dependent) emissions following MEGAN 2.1 algorithm.
Basal emission rates by PFT.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class BVOCState(eqx.Module):
    """BVOC emission diagnostics."""

    isoprene: Float[Array, "npoi"]      # Isoprene emission (ugC/m2/hr)
    monoterpene: Float[Array, "npoi"]   # Monoterpene emission (ugC/m2/hr)


def create_bvoc_state(npoi: int) -> BVOCState:
    z = jnp.zeros((npoi,))
    return BVOCState(isoprene=z, monoterpene=z)


# ---------------------------------------------------------------------------
# Basal emission rates by PFT (ugC/m2leaf/hr at standard conditions)
# from MEGAN 2.1, Table 3 (Guenther et al., 2012)
# PFT order: [0-3] tropical trees, [4-5] temperate trees,
#            [6-7] boreal trees, [8] shrub, [9-10] C3/C4 grass, [11] crop
# ---------------------------------------------------------------------------

ISOP_BASAL = jnp.array([
    24.0, 24.0, 24.0, 24.0,  # tropical broadleaf
    45.0, 8.0,                 # temperate deciduous, conifer
    8.0, 8.0,                  # boreal deciduous, conifer
    16.0,                      # shrub
    12.0, 12.0,               # C3, C4 grass
    5.0,                       # crop
])

MONO_BASAL = jnp.array([
    0.8, 0.8, 0.8, 0.8,       # tropical broadleaf
    0.8, 2.0,                   # temperate deciduous, conifer
    2.0, 2.0,                   # boreal deciduous, conifer
    0.6,                        # shrub
    0.4, 0.4,                  # C3, C4 grass
    0.2,                       # crop
])


# ---------------------------------------------------------------------------
# MEGAN activity factors
# ---------------------------------------------------------------------------

# Standard conditions
T_STANDARD = 303.0  # K (30C)
PAR_STANDARD = 1000.0  # umol/m2/s

# Isoprene temperature parameters (from MEGAN 2.1)
CT1 = 95000.0   # J/mol
CT2 = 230000.0  # J/mol
T_OPT = 313.0   # K
R = 8.314        # J/mol/K


def gamma_t_isop(tleaf):
    """Temperature activity factor for isoprene (MEGAN 2.1 Eq. 5).

    Parameters
    ----------
    tleaf : temperature (K)

    Returns
    -------
    gamma_T : dimensionless, ~1 at T_STANDARD
    """
    x = (1.0 / T_OPT - 1.0 / tleaf) / R
    num = jnp.exp(CT1 * x)
    denom = 1.0 + jnp.exp(CT2 * (tleaf * x - 0.0))
    # Simplified: Arrhenius-like response
    eopt = jnp.exp(CT1 * (tleaf - T_STANDARD) / (R * T_STANDARD * tleaf))
    gamma = eopt / (1.0 + jnp.exp(CT2 * (tleaf - T_OPT) / (R * T_OPT * tleaf)))
    return jnp.clip(gamma, 0.0, 10.0)


def gamma_par_isop(par):
    """Light activity factor for isoprene (MEGAN 2.1 Eq. 4).

    Parameters
    ----------
    par : PAR (umol/m2/s)

    Returns
    -------
    gamma_PAR : dimensionless, ~1 at PAR_STANDARD
    """
    alpha = 0.0027
    cl1 = 1.066
    gamma = alpha * cl1 * par / jnp.sqrt(1.0 + alpha**2 * par**2)
    return jnp.clip(gamma, 0.0, 5.0)


def gamma_t_mono(tleaf, beta=0.09):
    """Temperature activity factor for monoterpenes.

    Exponential T dependence: exp(beta * (T - Ts))

    Parameters
    ----------
    tleaf : temperature (K)
    beta : temperature sensitivity (1/K)

    Returns
    -------
    gamma_T : dimensionless
    """
    return jnp.exp(beta * (tleaf - T_STANDARD))


def bvoc_emissions(
    tleaf, par, lai, frac,
    isop_basal=ISOP_BASAL,
    mono_basal=MONO_BASAL,
):
    """Compute BVOC emissions using MEGAN 2.1.

    Parameters
    ----------
    tleaf : (npoi,) canopy temperature (K)
    par : (npoi,) photosynthetically active radiation (umol/m2/s)
    lai : (npoi, npft) LAI per PFT
    frac : (npoi, npft) PFT area fractions
    isop_basal : (npft,) basal isoprene emission rates
    mono_basal : (npft,) basal monoterpene emission rates

    Returns
    -------
    BVOCState with isoprene and monoterpene emissions
    """
    npoi = tleaf.shape[0]
    npft = lai.shape[1]

    # Trim basal rates to match npft
    isop = isop_basal[:npft]
    mono = mono_basal[:npft]

    # Activity factors (scalar per grid point)
    gt_isop = gamma_t_isop(tleaf)       # (npoi,)
    gp_isop = gamma_par_isop(par)       # (npoi,)
    gt_mono = gamma_t_mono(tleaf)       # (npoi,)

    # Emission = sum over PFTs of: basal * gamma_T * gamma_PAR * LAI * frac
    # Isoprene
    isop_pft = isop[None, :] * lai * frac  # (npoi, npft)
    isoprene = jnp.sum(isop_pft, axis=1) * gt_isop * gp_isop

    # Monoterpene (no light dependence)
    mono_pft = mono[None, :] * lai * frac  # (npoi, npft)
    monoterpene = jnp.sum(mono_pft, axis=1) * gt_mono

    return BVOCState(isoprene=isoprene, monoterpene=monoterpene)
