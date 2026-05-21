"""Soil thermal and hydraulic model for IBIS-JAX.

Ported from soil.f90.  All functions are pure, JIT-compatible, and
differentiable.  Mutable Fortran state is replaced by explicit
inputs/outputs.

Functions:
  setsoi     -- diagnostic soil conductivity, evap fractions, latent heat
  wadjust    -- bounds-enforcement for wsoi and wisoi
  soilh2o    -- implicit soil moisture solver (Thomas algorithm)
  soilheat   -- implicit soil heat conduction solver (Thomas algorithm)
  soilctl    -- main soil timestep controller
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Float, Array

from ibis.physics.solvers import tridia_batched
from ibis.physics.thermo import hvapf, hsubf


# ---------------------------------------------------------------------------
# setsoi: diagnostic soil quantities
# ---------------------------------------------------------------------------

class SetsoiResult(NamedTuple):
    consoi: Float[Array, "npoi nsoilay"]
    qglif: Float[Array, "npoi 4"]
    hvasug: Float[Array, "npoi"]
    hvasui: Float[Array, "npoi"]


def setsoi(
    sand: Float[Array, "npoi nsoilay"],
    clay: Float[Array, "npoi nsoilay"],
    poros: Float[Array, "npoi nsoilay"],
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    tsoi: Float[Array, "npoi nsoilay"],
    wpud: Float[Array, "npoi"],
    wipud: Float[Array, "npoi"],
    wpudmax: float,
    zwpmax: float,
    tsno_1: Float[Array, "npoi"],
    ta: Float[Array, "npoi"],
    tmelt: float,
    hvap: float,
    hsub: float,
    ch2o: float,
    cvap: float,
    cice: float,
    epsilon: float,
) -> SetsoiResult:
    """Set diagnostic soil conductivity, evap fractions, latent heat.

    Matches Fortran setsoi().
    """
    npoi = sand.shape[0]

    # --- Thermal conductivity per layer ---
    fsand = 0.01 * sand
    fclay = 0.01 * clay
    fsilt = 1.0 - fsand - fclay

    powliq = poros * wsoi * (1.0 - wisoi)
    powice = poros * wisoi

    zcondry = fsand * 0.300 + fsilt * 0.265 + fclay * 0.250
    consoi = zcondry * (56.0 ** powliq) * (224.0 ** powice)

    # --- Evaporation source fractions (qglif) ---
    # zwpud: fraction of surface covered by puddle
    pud_total = wpud + wipud
    zwpud = jnp.clip(zwpmax * pud_total / jnp.maximum(wpudmax, epsilon), 0.0, zwpmax)
    zwsoi = (1.0 - wisoi[:, 0]) * wsoi[:, 0] + wisoi[:, 0]

    has_soil_water = zwsoi >= epsilon
    has_puddle = zwpud >= epsilon
    rwork1 = 1.0 / jnp.maximum(zwsoi, epsilon)
    rwork2 = 1.0 / jnp.maximum(pud_total, epsilon)

    # Case 1: soil water present, puddle present
    q1_sp = (1.0 - zwpud) * (1.0 - wisoi[:, 0]) * wsoi[:, 0] * rwork1
    q2_sp = (1.0 - zwpud) * wisoi[:, 0] * rwork1
    q3_sp = zwpud * wpud * rwork2
    q4_sp = zwpud * wipud * rwork2

    # Case 2: soil water present, no puddle
    q1_s = (1.0 - wisoi[:, 0]) * wsoi[:, 0] * rwork1
    q2_s = wisoi[:, 0] * rwork1
    q3_s = jnp.zeros(npoi)
    q4_s = jnp.zeros(npoi)

    # Case 3: no soil water, puddle present
    q1_p = jnp.zeros(npoi)
    q2_p = jnp.zeros(npoi)
    q3_p = zwpud * wpud * rwork2
    q4_p = zwpud * wipud * rwork2

    # Case 4: no soil water, no puddle
    # finding #21: eps-band to avoid ULP-level branch flip at tmelt
    above_melt = tsoi[:, 0] >= tmelt - 1.0e-9
    q1_n = jnp.zeros(npoi)
    q2_n = jnp.zeros(npoi)
    q3_n = jnp.where(above_melt, 1.0, 0.0)
    q4_n = jnp.where(above_melt, 0.0, 1.0)

    # Combine cases
    q1 = jnp.where(has_soil_water,
                   jnp.where(has_puddle, q1_sp, q1_s),
                   jnp.where(has_puddle, q1_p, q1_n))
    q2 = jnp.where(has_soil_water,
                   jnp.where(has_puddle, q2_sp, q2_s),
                   jnp.where(has_puddle, q2_p, q2_n))
    q3 = jnp.where(has_soil_water,
                   jnp.where(has_puddle, q3_sp, q3_s),
                   jnp.where(has_puddle, q3_p, q3_n))
    q4 = jnp.where(has_soil_water,
                   jnp.where(has_puddle, q4_sp, q4_s),
                   jnp.where(has_puddle, q4_p, q4_n))

    qglif = jnp.stack([q1, q2, q3, q4], axis=1)

    # --- Latent heat at soil and snow surfaces ---
    # hvapf(t, tair) matches Fortran: hvap + cvap*(tair-tmelt) - ch2o*(t-tmelt)
    zvap = hvapf(tsoi[:, 0], ta, tmelt, hvap, ch2o, cvap)
    zsub = hsubf(tsoi[:, 0], ta, tmelt, hsub, cvap, cice)
    hvasug = (q1 + q3) * zvap + (q2 + q4) * zsub
    hvasui = hsubf(tsno_1, ta, tmelt, hsub, cvap, cice)

    return SetsoiResult(consoi=consoi, qglif=qglif, hvasug=hvasug, hvasui=hvasui)


# ---------------------------------------------------------------------------
# wadjust: bounds enforcement
# ---------------------------------------------------------------------------

def wadjust(
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    poros: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    rhow: float,
    dtime: float,
    epsilon: float,
):
    """Enforce bounds on wsoi and wisoi, return adjusted arrays and gadjust.

    Matches Fortran wadjust().
    """
    npoi, nsoilay = wsoi.shape

    # Total water before
    ztot0 = hsoi[None, :] * poros * rhow * ((1.0 - wisoi) * wsoi + wisoi)

    # Clamp small values to zero
    wsoi_new = jnp.where(wsoi < epsilon, 0.0, wsoi)
    wisoi_new = jnp.where(wisoi < epsilon, 0.0, wisoi)

    # Clamp to [0, 1]
    wsoi_new = jnp.minimum(1.0, wsoi_new)
    wisoi_new = jnp.minimum(1.0, wisoi_new)

    # If ice fills pore space, liquid must be zero
    wsoi_new = jnp.where(wisoi_new >= 1.0 - epsilon, 0.0, wsoi_new)

    # Total water after
    ztot1 = hsoi[None, :] * poros * rhow * ((1.0 - wisoi_new) * wsoi_new + wisoi_new)

    # Mass adjustment rate
    gadjust = jnp.sum(ztot1 - ztot0, axis=1) / dtime

    return wsoi_new, wisoi_new, gadjust


# ---------------------------------------------------------------------------
# soilh2o: implicit soil moisture solver
# ---------------------------------------------------------------------------

class Soilh2oResult(NamedTuple):
    wsoi: Float[Array, "npoi nsoilay"]
    wpud: Float[Array, "npoi"]
    wflo: Float[Array, "npoi nsoilay_p1"]
    fsqueez: Float[Array, "npoi"]


def soilh2o(
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    poros: Float[Array, "npoi nsoilay"],
    porosflo: Float[Array, "npoi nsoilay"],
    hydraul: Float[Array, "npoi nsoilay"],
    suction: Float[Array, "npoi nsoilay"],
    bex: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    fwtop: Float[Array, "npoi"],
    fwpud: Float[Array, "npoi"],
    wpud: Float[Array, "npoi"],
    bperm: float,
    dtime: float,
    rhow: float,
    epsilon: float,
    ftrans: Float[Array, "npoi nsoilay"] | None = None,
) -> Soilh2oResult:
    """Implicit soil moisture solver.

    Sets up and solves tridiagonal moisture diffusion equation.
    Matches Fortran soilh2o().

    ftrans : (npoi, nsoilay) transpiration water extraction rate (kg/m2/s)
        per soil layer, distributed by root fraction.  When None, no
        transpiration is withdrawn (legacy behaviour).
    """
    npoi, nsoilay = wsoi.shape
    DMIN = 1e-9
    RIMP = 1.0

    owsoi = wsoi.copy()

    # --- Interpolation weights and boundary moistures ---
    # hsoim[k] = distance between layer centers
    # We compute e, f, g flux terms at layer interfaces (nsoilay+1 total)

    # Interface 0 (top): zero flux
    e = jnp.zeros((npoi, nsoilay + 1))
    f = jnp.zeros((npoi, nsoilay + 1))
    g = jnp.zeros((npoi, nsoilay + 1))

    # Middle interfaces (k=1..nsoilay-1 in 0-indexed)
    for k in range(1, nsoilay):
        hsoim_k = 0.5 * (hsoi[k - 1] + hsoi[k])
        weip = 0.5 * hsoi[k] / hsoim_k
        weim = 1.0 - weip

        a_k = weim * hydraul[:, k - 1] + weip * hydraul[:, k]
        b_k = (weim * hydraul[:, k - 1] * suction[:, k - 1] * bex[:, k - 1]
               + weip * hydraul[:, k] * suction[:, k] * bex[:, k])
        zbex = weim * bex[:, k - 1] + weip * bex[:, k]
        # F77 nint() is round-half-away-from-zero; jnp.round is banker's.
        # Use jnp.floor(x + 0.5) for F77-nint-compatible behavior on positive x.
        m_k = 2 * jnp.floor(zbex + 0.5).astype(jnp.int32) + 3
        n_k = jnp.floor(zbex + 0.5).astype(jnp.int32) + 2

        wsoim = weim * wsoi[:, k - 1] + weip * wsoi[:, k]
        wsoia = jnp.minimum(wsoim, 1.0)
        wsoib = jnp.minimum(wsoim, 1.0)

        bwn1 = b_k * (wsoib ** (n_k - 1))
        bwn = bwn1 * wsoib
        bwn1 = jnp.where(bwn < DMIN, 0.0, bwn1)
        bwn = jnp.maximum(bwn, DMIN)

        dw = (wsoi[:, k] - wsoi[:, k - 1]) / hsoim_k

        e_k = ((-1.0 + RIMP * m_k) * a_k * (wsoia ** m_k)
               + ((1.0 - RIMP) * bwn - RIMP * n_k * bwn1 * wsoib) * dw)
        f_k = (-RIMP * m_k * a_k * (wsoia ** (m_k - 1))
               + RIMP * n_k * bwn1 * dw)
        g_k = RIMP * bwn

        # Ice barrier
        z_ice = jnp.maximum(0.0, 1.0 - jnp.maximum(wisoi[:, k - 1], wisoi[:, k])) ** 2
        e = e.at[:, k].set(z_ice * e_k)
        f = f.at[:, k].set(z_ice * f_k)
        g = g.at[:, k].set(z_ice * g_k)

    # Bottom interface: free drainage or impermeable
    k = nsoilay
    wsoia_bot = jnp.minimum(wsoi[:, nsoilay - 1], 1.0)
    a_bot = hydraul[:, nsoilay - 1]
    m_bot = 2 * jnp.floor(bex[:, nsoilay - 1] + 0.5).astype(jnp.int32) + 3
    e_bot = -a_bot * (wsoia_bot ** m_bot) * bperm
    z_ice_bot = jnp.maximum(0.0, 1.0 - wisoi[:, nsoilay - 1]) ** 2
    e = e.at[:, k].set(z_ice_bot * e_bot)

    # --- Build tridiagonal system ---
    d1 = jnp.zeros((npoi, nsoilay))
    d2 = jnp.zeros((npoi, nsoilay))
    d3 = jnp.zeros((npoi, nsoilay))
    rhs_arr = jnp.zeros((npoi, nsoilay))

    for k in range(nsoilay):
        dt = dtime / jnp.maximum(porosflo[:, k] * hsoi[k], 1e-30)
        if k == 0:
            hsoim_k = 0.5 * hsoi[0]
        else:
            hsoim_k = 0.5 * hsoi[k - 1] + 0.5 * hsoi[k]
        hsoim_k = jnp.maximum(hsoim_k, 1e-30)

        km1 = max(k - 1, 0)  # Python int, always OK
        d1_k = dt * (f[:, k] * 0.5 * hsoi[k] / hsoim_k
                     - g[:, k] / hsoim_k)
        rhs_k = wsoi[:, k] + dt * (e[:, k + 1] - e[:, k])

        if k == 0:
            rhs_k = rhs_k + dt * (fwtop + fwpud) / rhow
            d1_k = jnp.zeros(npoi)

        # Transpiration extraction: remove water from each layer by root uptake
        if ftrans is not None:
            rhs_k = rhs_k - dt * ftrans[:, k] / rhow

        if k < nsoilay - 1:
            hsoim_kp1 = jnp.maximum(0.5 * (hsoi[k] + hsoi[k + 1]), 1e-30)
            d2_k = (1.0 + dt * (-f[:, k + 1] * 0.5 * hsoi[k + 1] / hsoim_kp1
                                + f[:, k] * 0.5 * hsoi[km1] / hsoim_k
                                + g[:, k + 1] / hsoim_kp1
                                + g[:, k] / hsoim_k))
            d3_k = dt * (-f[:, k + 1] * 0.5 * hsoi[k] / hsoim_kp1
                         - g[:, k + 1] / hsoim_kp1)
        else:
            d2_k = (1.0 + dt * (-f[:, k + 1]
                                + f[:, k] * 0.5 * hsoi[km1] / hsoim_k
                                + g[:, k] / hsoim_k))
            d3_k = jnp.zeros(npoi)

        d1 = d1.at[:, k].set(d1_k)
        d2 = d2.at[:, k].set(d2_k)
        d3 = d3.at[:, k].set(d3_k)
        rhs_arr = rhs_arr.at[:, k].set(rhs_k)

    # Solve
    wsoi_new = tridia_batched(d1, d2, d3, rhs_arr)

    # --- Squeeze excess and compute water fluxes ---
    wflo = jnp.zeros((npoi, nsoilay + 1))
    wflo = wflo.at[:, nsoilay].set(-rhow * e[:, nsoilay])

    fsqueez = jnp.zeros(npoi)
    for k in range(nsoilay - 1, -1, -1):
        zz = rhow * poros[:, k] * jnp.maximum(epsilon, 1.0 - wisoi[:, k]) * hsoi[k]
        wsoi_new_k = wsoi_new[:, k] + dtime * fsqueez / zz
        fsqueez_new = jnp.maximum(wsoi_new_k - 1.0, 0.0) * zz / dtime
        wsoi_new_k = jnp.minimum(wsoi_new_k, 1.0)
        wflo_k = wflo[:, k + 1] + (wsoi_new_k - owsoi[:, k]) * zz / dtime
        wsoi_new = wsoi_new.at[:, k].set(wsoi_new_k)
        wflo = wflo.at[:, k].set(wflo_k)
        fsqueez = fsqueez_new

    # Update puddle
    wpud_new = wpud + (fsqueez - fwpud) * dtime
    wflo = wflo.at[:, 0].set(wflo[:, 0] + fsqueez - fwpud)

    return Soilh2oResult(wsoi=wsoi_new, wpud=wpud_new, wflo=wflo, fsqueez=fsqueez)


# ---------------------------------------------------------------------------
# soilheat: implicit soil heat conduction
# ---------------------------------------------------------------------------

class SoilheatResult(NamedTuple):
    tsoi: Float[Array, "npoi nsoilay"]
    hflo: Float[Array, "npoi nsoilay_p1"]


def soilheat(
    tsoi: Float[Array, "npoi nsoilay"],
    otsoi: Float[Array, "npoi nsoilay"],
    owsoi: Float[Array, "npoi nsoilay"],
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    poros: Float[Array, "npoi nsoilay"],
    consoi: Float[Array, "npoi nsoilay"],
    csoi_mineral: Float[Array, "npoi nsoilay"],
    rhosoi: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    c0pud: Float[Array, "npoi nsoilay"],
    c1pud: Float[Array, "npoi nsoilay"],
    fhtop: Float[Array, "npoi"],
    wflo: Float[Array, "npoi nsoilay_p1"],
    ch2o: float,
    cice: float,
    rhow: float,
    dtime: float,
) -> SoilheatResult:
    """Implicit soil heat conduction solver.

    Matches Fortran soilheat().
    """
    npoi, nsoilay = tsoi.shape
    RIMP = 1.0

    # --- Conduction coefficients ---
    con = jnp.zeros((npoi, nsoilay + 1))
    whflo = jnp.zeros((npoi, nsoilay + 1))

    # Top interface: zero conduction
    whflo = whflo.at[:, 0].set(wflo[:, 0] * ch2o * tsoi[:, 0])

    # Middle interfaces
    for k in range(1, nsoilay):
        con_k = 1.0 / (0.5 * (hsoi[k - 1] / jnp.maximum(consoi[:, k - 1], 1e-12)
                               + hsoi[k] / jnp.maximum(consoi[:, k], 1e-12)))
        t_avg = ((hsoi[k] * tsoi[:, k - 1] + hsoi[k - 1] * tsoi[:, k])
                 / (hsoi[k - 1] + hsoi[k]))
        whflo_k = wflo[:, k] * ch2o * t_avg
        con = con.at[:, k].set(con_k)
        whflo = whflo.at[:, k].set(whflo_k)

    # Bottom interface: zero conduction
    whflo = whflo.at[:, nsoilay].set(wflo[:, nsoilay] * ch2o * tsoi[:, nsoilay - 1])

    # --- Heat capacity terms ---
    rwork1 = (1.0 - poros) * csoi_mineral * rhosoi
    rwork2 = poros * (1.0 - wisoi) * ch2o * rhow
    rwork3 = poros * wisoi * cice * rhow

    c0 = c0pud + (rwork1 + rwork2 * owsoi + rwork3) * hsoi[None, :]
    c1 = c1pud + (rwork1 + rwork2 * wsoi + rwork3) * hsoi[None, :]

    # --- Tridiagonal matrix ---
    d1 = jnp.zeros((npoi, nsoilay))
    d2 = jnp.zeros((npoi, nsoilay))
    d3 = jnp.zeros((npoi, nsoilay))
    rhs_arr = jnp.zeros((npoi, nsoilay))

    for k in range(nsoilay):
        km1 = max(k - 1, 0)
        kp1 = min(k + 1, nsoilay - 1)
        rwork = dtime / c1[:, k]

        d1_k = -rwork * RIMP * con[:, k]
        d2_k = 1.0 + rwork * RIMP * (con[:, k] + con[:, k + 1])
        d3_k = -rwork * RIMP * con[:, k + 1]

        rhs_k = ((c0[:, k] / c1[:, k]) * tsoi[:, k]
                 + rwork * ((1.0 - RIMP) * con[:, k] * (tsoi[:, km1] - tsoi[:, k])
                            + (1.0 - RIMP) * con[:, k + 1] * (tsoi[:, kp1] - tsoi[:, k])
                            + whflo[:, k] - whflo[:, k + 1]))
        if k == 0:
            rhs_k = rhs_k + (dtime / c1[:, k]) * fhtop

        d1 = d1.at[:, k].set(d1_k)
        d2 = d2.at[:, k].set(d2_k)
        d3 = d3.at[:, k].set(d3_k)
        rhs_arr = rhs_arr.at[:, k].set(rhs_k)

    # Solve
    tsoi_new = tridia_batched(d1, d2, d3, rhs_arr)

    # --- Deduce heat fluxes ---
    hflo = jnp.zeros((npoi, nsoilay + 1))
    hflo = hflo.at[:, 0].set(fhtop)
    for k in range(nsoilay):
        hflo = hflo.at[:, k + 1].set(
            hflo[:, k] - (c1[:, k] * tsoi_new[:, k] - c0[:, k] * otsoi[:, k]) / dtime
        )

    return SoilheatResult(tsoi=tsoi_new, hflo=hflo)


# ---------------------------------------------------------------------------
# freeze_thaw: heat-conserving liquid/ice phase change
# ---------------------------------------------------------------------------

def freeze_thaw(
    tsoi: Float[Array, "npoi nsoilay"],
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    poros: Float[Array, "npoi nsoilay"],
    csoi_mineral: Float[Array, "npoi nsoilay"],
    rhosoi: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    c1pud: Float[Array, "npoi nsoilay"],
    ch2o: float,
    cice: float,
    rhow: float,
    hfus: float,
    tmelt: float,
    epsilon: float,
):
    """Heat-conserving freeze/thaw adjustment for soil layers.

    Matches the freeze/thaw loop in Fortran soilctl().
    Returns updated (tsoi, wsoi, wisoi, porosflo).
    """
    npoi, nsoilay = tsoi.shape

    for k in range(nsoilay):
        zporos = jnp.maximum(poros[:, k], epsilon)
        rwork = (c1pud[:, k] / (rhow * hsoi[k])
                 + (1.0 - zporos) * csoi_mineral[:, k] * rhosoi[:, k] / rhow)

        chav = (rwork
                + zporos * (1.0 - wisoi[:, k]) * wsoi[:, k] * ch2o
                + zporos * wisoi[:, k] * cice)

        # Freeze liquid below melting point
        zwsoi = jnp.minimum(1.0, wsoi[:, k])
        dh = chav * (tmelt - tsoi[:, k])
        dw = jnp.minimum(
            zporos * (1.0 - wisoi[:, k]) * zwsoi,
            jnp.maximum(0.0, dh / hfus),
        )
        wisoi_k = wisoi[:, k] + dw / zporos
        wsoi_k = wsoi[:, k] - (dw / zporos) * (1.0 - zwsoi) / jnp.maximum(epsilon, 1.0 - wisoi_k)

        chav = (rwork
                + zporos * (1.0 - wisoi_k) * wsoi_k * ch2o
                + zporos * wisoi_k * cice)
        tsoi_k = tmelt - (dh - hfus * dw) / chav

        # Melt ice above melting point
        dh = chav * (tsoi_k - tmelt)
        dw = jnp.minimum(zporos * wisoi_k, jnp.maximum(0.0, dh / hfus))
        wisoi_k = wisoi_k - dw / zporos
        wsoi_k = wsoi_k + (dw / zporos) * (1.0 - wsoi_k) / jnp.maximum(epsilon, 1.0 - wisoi_k)

        chav = (rwork
                + zporos * (1.0 - wisoi_k) * wsoi_k * ch2o
                + zporos * wisoi_k * cice)
        tsoi_k = tmelt + (dh - hfus * dw) / chav

        tsoi = tsoi.at[:, k].set(tsoi_k)
        wsoi = wsoi.at[:, k].set(wsoi_k)
        wisoi = wisoi.at[:, k].set(wisoi_k)

    porosflo = poros * jnp.maximum(epsilon, 1.0 - wisoi)
    return tsoi, wsoi, wisoi, porosflo


# ---------------------------------------------------------------------------
# puddle_freeze_thaw: freeze/thaw puddle water (Fortran soil.f lines 485-514)
# ---------------------------------------------------------------------------

def puddle_freeze_thaw(
    tsoi_0: Float[Array, "npoi"],
    wpud: Float[Array, "npoi"],
    wipud: Float[Array, "npoi"],
    poros_0: Float[Array, "npoi"],
    wsoi_0: Float[Array, "npoi"],
    wisoi_0: Float[Array, "npoi"],
    csoi_mineral_0: Float[Array, "npoi"],
    rhosoi_0: Float[Array, "npoi"],
    hsoi_0: float,
    ch2o: float,
    cice: float,
    rhow: float,
    hfus: float,
    tmelt: float,
):
    """Freeze/thaw puddle water based on top soil layer temperature.

    Matches Fortran soil.f lines 485-514.  The soil temperature tsoi(1)
    has already been adjusted by soil freeze_thaw so it is consistent
    with wsoi/wisoi; this routine additionally equilibrates puddle water.
    """
    # Soil first-layer heat capacity (without puddle)
    rwork = ((1.0 - poros_0) * csoi_mineral_0 * rhosoi_0
             + poros_0 * (1.0 - wisoi_0) * wsoi_0 * ch2o * rhow
             + poros_0 * wisoi_0 * cice * rhow) * hsoi_0

    # Freeze puddle liquid below tmelt
    chav = ch2o * wpud + cice * wipud + rwork
    dh = chav * (tmelt - tsoi_0)
    dw = jnp.minimum(wpud, jnp.maximum(0.0, dh / hfus))
    wipud = wipud + dw
    wpud = wpud - dw
    chav = ch2o * wpud + cice * wipud + rwork
    tsoi_0 = tmelt - (dh - hfus * dw) / chav

    # Melt puddle ice above tmelt
    dh2 = chav * (tsoi_0 - tmelt)
    dw2 = jnp.minimum(wipud, jnp.maximum(0.0, dh2 / hfus))
    wipud = wipud - dw2
    wpud = wpud + dw2
    chav = ch2o * wpud + cice * wipud + rwork
    tsoi_0 = tmelt + (dh2 - hfus * dw2) / chav

    return tsoi_0, wpud, wipud
