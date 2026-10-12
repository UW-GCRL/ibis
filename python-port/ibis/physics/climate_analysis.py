"""Climate analysis routines for PFT existence.

Ported from climate.f90: climanl, climanl2, existence.

These routines determine which PFTs can exist in each grid cell
based on climate criteria (GDD, coldest/warmest month temperatures).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float, Array

from ibis.io.parameters import PFTParams


def existence(
    tcmin: Float[Array, "npoi"],
    tw: Float[Array, "npoi"],
    gdd0: Float[Array, "npoi"],
    gdd5: Float[Array, "npoi"],
    pft_params: PFTParams,
    exist_mask=None,
    larch_gdd_fix: bool = False,
) -> Float[Array, "npoi 12"]:
    """Determine which PFTs can exist in each grid cell.

    `exist_mask` (npoi, 12), optional: multiplied into the result -- the
    Koppen-Geiger mask of `ibis.io.koppen` (CALIBRATION.md 44). None leaves
    the bioclimatic envelope exactly as F77 computes it.

    `larch_gdd_fix`: the F77 rule for PFT 8 (boreal conifer cold-deciduous,
    larch) reads `gdd5 < TminL(4)` (= -45) where its own comment says
    `gdd5 < GDD(4)` (= 1200) -- `climate.f:398-400`. The comparison can never
    be true, so larch exists only where the coldest month is below -45 C.
    False (default) reproduces the Fortran; True uses the degree-day limit
    the comment intends.

    Uses bioclimatic limits from pft_params to set existence flags.
    All logic uses jnp.where for JIT compatibility.

    Parameters
    ----------
    tcmin : (npoi,)
        Coldest month temperature adjusted by delta (deg C).
    tw : (npoi,)
        Warmest month temperature (deg C).
    gdd0 : (npoi,)
        Growing degree days above 0C.
    gdd5 : (npoi,)
        Growing degree days above 5C.
    pft_params : PFTParams
        PFT bioclimatic limit parameters.

    Returns
    -------
    exist : (npoi, 12)
        PFT existence flags (0.0 or 1.0).
    """
    TminL = pft_params.TminL
    TminU = pft_params.TminU
    Twarm = pft_params.Twarm
    GDD = pft_params.GDD

    npoi = tcmin.shape[0]
    exist = jnp.zeros((npoi, 12))

    # PFT 1: tropical broadleaf evergreen (tcmin > TminL(1))
    exist = exist.at[:, 0].set(jnp.where(tcmin > TminL[0], 1.0, 0.0))

    # PFT 2: tropical broadleaf drought-deciduous
    exist = exist.at[:, 1].set(jnp.where(tcmin > TminL[1], 1.0, 0.0))

    # PFT 3: warm-temperate broadleaf evergreen
    exist = exist.at[:, 2].set(jnp.where(
        (tcmin < TminU[2]) & (tcmin > TminL[2]), 1.0, 0.0))

    # PFT 4: temperate conifer evergreen
    exist = exist.at[:, 3].set(jnp.where(
        (tcmin < TminU[3]) & (tcmin > TminL[3]) & (gdd5 > GDD[3]),
        1.0, 0.0))

    # PFT 5: temperate broadleaf cold-deciduous
    exist = exist.at[:, 4].set(jnp.where(
        (tcmin < TminU[4]) & (tcmin > TminL[4]) & (gdd5 > GDD[4]),
        1.0, 0.0))

    # PFT 6: boreal conifer evergreen
    exist = exist.at[:, 5].set(jnp.where(
        ((tcmin < TminU[5]) | (gdd5 < GDD[3])) &
        (tcmin > TminL[5]) & (gdd5 > GDD[5]),
        1.0, 0.0))

    # PFT 7: boreal broadleaf cold-deciduous
    exist = exist.at[:, 6].set(jnp.where(
        ((tcmin < TminU[6]) | (gdd5 < GDD[4])) &
        (tcmin > TminL[6]) & (gdd5 > GDD[6]),
        1.0, 0.0))

    # PFT 8: boreal conifer cold-deciduous (see `larch_gdd_fix` above)
    _gdd_lim8 = GDD[3] if larch_gdd_fix else TminL[3]
    exist = exist.at[:, 7].set(jnp.where(
        ((tcmin < TminU[7]) | (gdd5 < _gdd_lim8)) &
        (gdd5 > GDD[7]),
        1.0, 0.0))

    # PFT 9: evergreen shrubs
    exist = exist.at[:, 8].set(jnp.where(gdd0 > GDD[8], 1.0, 0.0))

    # PFT 10: deciduous shrubs
    exist = exist.at[:, 9].set(jnp.where(gdd0 > GDD[9], 1.0, 0.0))

    # PFT 11: warm (C4) grasses
    exist = exist.at[:, 10].set(jnp.where(
        (tw > Twarm[10]) & (gdd0 > GDD[10]), 1.0, 0.0))

    # PFT 12: cool (C3) grasses
    exist = exist.at[:, 11].set(jnp.where(gdd0 > GDD[11], 1.0, 0.0))

    if exist_mask is not None:
        exist = exist * jnp.asarray(exist_mask, dtype=exist.dtype)
    return exist


def climanl(
    xint: Float[Array, "npoi 12"],
    deltat: Float[Array, "npoi"],
    ndaypm: tuple,
) -> tuple[
    Float[Array, "npoi"],   # tc
    Float[Array, "npoi"],   # tw
    Float[Array, "npoi"],   # tcmin
    Float[Array, "npoi"],   # gdd0
    Float[Array, "npoi"],   # gdd5
]:
    """Initialise climate accumulators from climatological monthly means.

    Called once at the start of the simulation (year 0).

    Parameters
    ----------
    xint : (npoi, 12)
        Monthly mean temperature (deg C).
    deltat : (npoi,)
        Temperature correction for tcmin.
    ndaypm : tuple of 12 ints
        Days per month.

    Returns
    -------
    tc, tw, tcmin, gdd0, gdd5
    """
    npoi = xint.shape[0]

    tc = jnp.min(xint, axis=1)
    tw = jnp.max(xint, axis=1)
    tcmin = tc + deltat

    gdd0 = jnp.zeros((npoi,))
    gdd5 = jnp.zeros((npoi,))

    # Interpolate monthly to daily and accumulate GDD
    for k in range(12):
        ndays_k = ndaypm[k]
        rwork = 1.0 / ndays_k
        for lda in range(ndays_k):
            lda_f = lda + 1.0  # 1-based day
            half = (ndays_k + 1.0) * 0.5

            # Determine interpolation indices and weight
            in_first_half = lda_f < half
            it1w = jnp.where(in_first_half, (k - 1) % 12, k)
            it2w = jnp.where(in_first_half, k, (k + 1) % 12)
            dt = jnp.where(
                in_first_half,
                (lda_f - 0.5) * rwork + 0.5,
                (lda_f - 0.5) * rwork - 0.5,
            )

            # This is a scalar loop, fine for initialization
            it1w_int = int(it1w)
            it2w_int = int(it2w)
            dtemp = xint[:, it1w_int] + dt * (xint[:, it2w_int] - xint[:, it1w_int])
            gdd0 = gdd0 + jnp.maximum(0.0, dtemp)
            gdd5 = gdd5 + jnp.maximum(0.0, dtemp - 5.0)

    return tc, tw, tcmin, gdd0, gdd5


def climanl2(
    tc: Float[Array, "npoi"],
    tw: Float[Array, "npoi"],
    tcmin: Float[Array, "npoi"],
    gdd0: Float[Array, "npoi"],
    gdd5: Float[Array, "npoi"],
    tcthis: Float[Array, "npoi"],
    twthis: Float[Array, "npoi"],
    gdd0this: Float[Array, "npoi"],
    gdd5this: Float[Array, "npoi"],
    deltat: Float[Array, "npoi"],
) -> tuple[
    Float[Array, "npoi"],   # tc
    Float[Array, "npoi"],   # tw
    Float[Array, "npoi"],   # tcmin
    Float[Array, "npoi"],   # gdd0
    Float[Array, "npoi"],   # gdd5
]:
    """Update 30-year e-folding running averages of climate variables.

    Called once per year after the annual summary.

    Parameters
    ----------
    tc, tw, tcmin, gdd0, gdd5 : (npoi,)
        Previous running averages.
    tcthis, twthis, gdd0this, gdd5this : (npoi,)
        This year's values.
    deltat : (npoi,)
        Temperature correction for tcmin.

    Returns
    -------
    tc, tw, tcmin, gdd0, gdd5 : updated running averages
    """
    zweig = jnp.exp(-1.0 / 30.0)
    rwork = 1.0 - zweig

    tc_new = zweig * tc + rwork * tcthis
    tw_new = zweig * tw + rwork * twthis
    tcmin_new = tc_new + deltat
    gdd0_new = zweig * gdd0 + rwork * gdd0this
    gdd5_new = zweig * gdd5 + rwork * gdd5this

    return tc_new, tw_new, tcmin_new, gdd0_new, gdd5_new
