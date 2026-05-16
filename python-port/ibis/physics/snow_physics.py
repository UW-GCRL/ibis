"""Snow-layer dynamics and heat conduction for IBIS-JAX.

Ported from snow.f90.  All functions are pure, JIT-compatible, and
differentiable.  Dynamic indexing (indsno/npn) is replaced by boolean
masks and jnp.where.

Functions:
  vadapt    -- re-adapt snow layer thicknesses (conserves heat)
  snowheat  -- implicit snow heat conduction (Thomas algorithm)
  snow_step -- main snow timestep controller
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Float, Array

from ibis.physics.solvers import tridia_batched


# ---------------------------------------------------------------------------
# vadapt: re-adapt snow layer profile
# ---------------------------------------------------------------------------

def vadapt(
    hcur: Float[Array, "npoi nlay"],
    tcur: Float[Array, "npoi nlay"],
    htop: float,
    has_snow: Float[Array, "npoi"],
    epsilon: float,
):
    """Re-adapt snow layer thicknesses, conserving heat (temperature * thickness).

    Matches Fortran vadapt().  The top layer gets thickness htop; the
    remaining layers are equal.  Temperature is redistributed to
    conserve the vertical integral of temperature * thickness.

    Parameters
    ----------
    hcur : array [npoi, nlay]
        Current layer thicknesses (m).
    tcur : array [npoi, nlay]
        Current temperatures (K).
    htop : float
        Prescribed top-layer thickness (m).
    has_snow : array [npoi]
        Boolean mask (1 where snow exists).

    Returns
    -------
    hcur_new, tcur_new : arrays
        Updated thicknesses and temperatures.
    """
    npoi, nlay = hcur.shape

    # Cumulative old layer boundaries from surface
    zold = jnp.zeros((npoi, nlay + 1))
    for k in range(nlay):
        zold = zold.at[:, k + 1].set(zold[:, k] + hcur[:, k])

    ht = zold[:, nlay]  # total thickness

    # New layer thicknesses
    rwork = nlay * htop
    h1 = jnp.where(ht >= rwork,
                   (ht - htop) / (nlay - 1),
                   jnp.maximum(ht / nlay, epsilon))

    hnew = jnp.broadcast_to(h1[:, None], (npoi, nlay))
    hnew = hnew.at[:, 0].set(jnp.where(ht >= rwork, htop, h1))

    # Integrate old temperature over each new layer
    # New layer boundaries
    znew = jnp.zeros((npoi, nlay + 1))
    for k in range(nlay):
        znew = znew.at[:, k + 1].set(znew[:, k] + hnew[:, k])

    tnew = jnp.zeros((npoi, nlay))
    for k in range(nlay):
        za = znew[:, k]
        zb = znew[:, k + 1]
        zheat = jnp.zeros(npoi)
        for ko in range(nlay):
            overlap = (jnp.minimum(zold[:, ko + 1], zb)
                       - jnp.maximum(zold[:, ko], za))
            overlap = jnp.maximum(overlap, 0.0)
            zheat = zheat + tcur[:, ko] * overlap
        tnew = tnew.at[:, k].set(zheat / jnp.maximum(hnew[:, k], epsilon))

    # Apply only to snow points
    hcur_new = jnp.where(has_snow[:, None], hnew, hcur)
    tcur_new = jnp.where(has_snow[:, None], tnew, tcur)

    return hcur_new, tcur_new


# ---------------------------------------------------------------------------
# snowheat: implicit snow heat conduction
# ---------------------------------------------------------------------------

class SnowheatResult(NamedTuple):
    tsno: Float[Array, "npoi nsnolay"]
    tlsub: Float[Array, "npoi"]
    sflo: Float[Array, "npoi nsnolay_p2"]


def snowheat(
    tsno: Float[Array, "npoi nsnolay"],
    tlsub: Float[Array, "npoi"],
    fhtop: Float[Array, "npoi"],
    xl: Float[Array, "npoi"],
    chl: float,
    fi: Float[Array, "npoi"],
    hsno: Float[Array, "npoi nsnolay"],
    rhos: float,
    consno: float,
    hsoi_1: float,
    consoi_1: Float[Array, "npoi"],
    tsoi_1: Float[Array, "npoi"],
    dtime: float,
    cice: float,
):
    """Implicit snow heat conduction with buried lower-veg layer.

    Matches Fortran snowheat().  Solves nsnolay+1 layers: nsnolay snow
    layers plus one buried lower-vegetation layer.

    Returns updated tsno, tlsub, and downward heat flux profile sflo.
    """
    npoi, nsnolay = tsno.shape
    RIMP = 1.0
    CONBUR = 2.0   # conduction coeff for buried veg (W m-2 K-1)
    HFAKE = 0.01   # minimum thickness for zero-snow points

    ntot = nsnolay + 1  # total layers (snow + buried veg)

    # Combined temperature array: [snow layers, buried-veg layer]
    temp = jnp.concatenate([tsno, tlsub[:, None]], axis=1)

    # --- Conduction coefficients between layers ---
    con = jnp.zeros((npoi, nsnolay + 2))

    # Top: zero
    # Between snow layers
    rwork_sno = 0.5 / consno
    for k in range(1, nsnolay):
        con_k = 1.0 / (jnp.maximum(hsno[:, k - 1], HFAKE) * rwork_sno
                        + jnp.maximum(hsno[:, k], HFAKE) * rwork_sno)
        con = con.at[:, k].set(con_k)

    # Snow-to-buried-veg interface
    con_sv = 1.0 / (jnp.maximum(hsno[:, nsnolay - 1], HFAKE) * rwork_sno
                     + 0.5 * xl / CONBUR)
    con = con.at[:, nsnolay].set(con_sv)

    # Buried-veg-to-soil interface
    rwork_bur = 0.5 / CONBUR
    con_vs = 1.0 / (xl * rwork_bur + 0.5 * hsoi_1 / jnp.maximum(consoi_1, 1e-12))
    con = con.at[:, nsnolay + 1].set(con_vs)

    # --- Tridiagonal system ---
    d1 = jnp.zeros((npoi, ntot))
    d2 = jnp.zeros((npoi, ntot))
    d3 = jnp.zeros((npoi, ntot))
    rhs_arr = jnp.zeros((npoi, ntot))

    rwork_dt = dtime / (rhos * cice)

    # Snow layers
    for k in range(nsnolay):
        km1 = max(k - 1, 0)
        kp1 = min(k + 1, ntot - 1)
        dt = rwork_dt / jnp.maximum(hsno[:, k], HFAKE)

        d1_k = -dt * RIMP * con[:, k]
        d2_k = 1.0 + dt * RIMP * (con[:, k] + con[:, k + 1])
        d3_k = -dt * RIMP * con[:, k + 1]
        rhs_k = (temp[:, k]
                 + dt * ((1.0 - RIMP) * con[:, k] * (temp[:, km1] - temp[:, k])
                         + (1.0 - RIMP) * con[:, k + 1] * (temp[:, kp1] - temp[:, k])))
        if k == 0:
            rhs_k = rhs_k + dt * fhtop

        d1 = d1.at[:, k].set(d1_k)
        d2 = d2.at[:, k].set(d2_k)
        d3 = d3.at[:, k].set(d3_k)
        rhs_arr = rhs_arr.at[:, k].set(rhs_k)

    # Buried-veg layer (k = nsnolay)
    k = nsnolay
    km1 = k - 1
    dti = xl * chl / dtime
    d1_k = -RIMP * con[:, k]
    d2_k = dti + RIMP * (con[:, k] + con[:, k + 1])
    d3_k = jnp.zeros(npoi)
    rhs_k = (dti * temp[:, k]
             + (1.0 - RIMP) * con[:, k] * (temp[:, km1] - temp[:, k])
             + con[:, k + 1] * (tsoi_1 - (1.0 - RIMP) * temp[:, k]))

    d1 = d1.at[:, k].set(d1_k)
    d2 = d2.at[:, k].set(d2_k)
    d3 = d3.at[:, k].set(d3_k)
    rhs_arr = rhs_arr.at[:, k].set(rhs_k)

    # Solve
    temp_new = tridia_batched(d1, d2, d3, rhs_arr)

    # --- Deduce downward heat fluxes ---
    sflo = jnp.zeros((npoi, nsnolay + 2))
    sflo = sflo.at[:, 0].set(fhtop)

    rwork_heat = rhos * cice / dtime
    for k in range(nsnolay):
        sflo = sflo.at[:, k + 1].set(
            sflo[:, k] - rwork_heat * hsno[:, k] * (temp_new[:, k] - tsno[:, k])
        )
    # Buried-veg layer
    sflo = sflo.at[:, nsnolay + 1].set(
        sflo[:, nsnolay] - xl * (chl / dtime) * (temp_new[:, nsnolay] - tlsub)
    )

    # Apply only where snow exists
    has_snow = fi > 0.0
    tsno_new = jnp.where(has_snow[:, None], temp_new[:, :nsnolay], tsno)
    tlsub_new = jnp.where(has_snow, temp_new[:, nsnolay], tlsub)

    return SnowheatResult(tsno=tsno_new, tlsub=tlsub_new, sflo=sflo)


# ---------------------------------------------------------------------------
# snow_melt: melt snow from any layer
# ---------------------------------------------------------------------------

def snow_melt(
    hsno: Float[Array, "npoi nsnolay"],
    tsno: Float[Array, "npoi nsnolay"],
    fi: Float[Array, "npoi"],
    rhos: float,
    cice: float,
    hfus: float,
    tmelt: float,
):
    """Melt snow from any layer where temperature exceeds melting point.

    Returns updated (hsno, tsno, zmelt, zheat).
    """
    xh = rhos * hsno * cice * jnp.maximum(tsno - tmelt, 0.0)
    xm = jnp.minimum(rhos * hsno, xh / hfus)
    hsno_new = hsno - xm / rhos
    tsno_new = jnp.minimum(tsno, tmelt)
    zmelt = jnp.sum(fi[:, None] * xm, axis=1)
    zheat = jnp.sum(fi[:, None] * (xh - hfus * xm), axis=1)

    return hsno_new, tsno_new, zmelt, zheat


# ---------------------------------------------------------------------------
# snow_coverage: adjust fi based on snow volume
# ---------------------------------------------------------------------------

def snow_coverage(
    hsno: Float[Array, "npoi nsnolay"],
    fi: Float[Array, "npoi"],
    hsnomin: float,
    hsnoruf: Float[Array, "npoi"],
    fimax: float,
    epsilon: float,
):
    """Adjust fractional snow cover based on total thickness.

    Uses the quadratic volume-coverage relation from the Fortran code.
    Returns updated (fi, hsno).
    """
    ht = jnp.sum(hsno, axis=1)
    # Only substitute hsnomin when snow is truly absent (Fortran: if ht==0)
    ht = jnp.where(ht < 1e-10, hsnomin, ht)

    zhh = 0.5 * hsnomin
    zdh = hsnoruf - hsnomin
    zdh = jnp.maximum(zdh, epsilon)

    finew = (-zhh + jnp.sqrt(zhh ** 2 + zdh * fi * ht)) / zdh
    finew = jnp.clip(finew, 0.0, fimax)

    scale = fi / jnp.maximum(finew, epsilon)
    hsno_new = hsno * scale[:, None]

    return finew, hsno_new
