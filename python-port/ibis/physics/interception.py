"""Canopy water interception, cascade, and wet-fraction physics for IBIS-JAX.

Ported from canopy.f90 (fwetcal, cascade, steph2o, cascad2, steph2o2,
mix, noveg, twet3).  All functions are pure and JIT-compatible.

Functions:
  fwetcal   -- fractional wet areas and liquid fraction
  mix       -- calorimetric mixing of masses
  twet3     -- wet bulb temperature (iterative)
  steph2o   -- step intercepted water for one canopy component
  cascade   -- precipitation cascade through canopy
  steph2o2  -- evaporation removal + phase correction
  cascad2   -- end-of-timestep evaporation cascade
  noveg     -- set temperatures when no vegetation exists
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Float, Array

from ibis.physics.thermo import esat, qsat, hvapf, hsubf


# ---------------------------------------------------------------------------
# Interception parameters
# ---------------------------------------------------------------------------

class InterceptionParams(NamedTuple):
    """Static interception parameters from params.can."""
    wliqumax: float = 0.20    # max liquid on upper leaves (kg/m2)
    wliqsmax: float = 0.40    # max liquid on upper stems
    wliqlmax: float = 0.20    # max liquid on lower canopy
    wsnoumax: float = 2.00    # max snow on upper leaves
    wsnosmax: float = 4.00    # max snow on upper stems
    wsnolmax: float = 2.00    # max snow on lower canopy
    tdripu: float = 7200.0    # drip timescale, upper leaves (s)
    tdrips: float = 7200.0    # drip timescale, upper stems
    tdripl: float = 7200.0    # drip timescale, lower canopy
    tblowu: float = 43200.0   # blowoff timescale, upper leaves (s)
    tblows: float = 43200.0   # blowoff timescale, upper stems
    tblowl: float = 43200.0   # blowoff timescale, lower canopy
    fmax: float = 0.25        # maximum wetted fraction


# ---------------------------------------------------------------------------
# fwetcal: fractional wet areas
# ---------------------------------------------------------------------------

class FwetResult(NamedTuple):
    """Wet fractions and liquid proportions."""
    fwetu: Float[Array, "npoi"]
    fwets: Float[Array, "npoi"]
    fwetl: Float[Array, "npoi"]
    rliqu: Float[Array, "npoi"]
    rliqs: Float[Array, "npoi"]
    rliql: Float[Array, "npoi"]


def fwetcal(
    tu, ts, tl,
    wliqu, wliqs, wliql,
    wsnou, wsnos, wsnol,
    wliqumax, wliqsmax, wliqlmax,
    wsnoumax, wsnosmax, wsnolmax,
    fmax, tmelt, epsilon,
):
    """Calculate fractional wet areas and liquid proportion.

    Returns FwetResult(fwetu, fwets, fwetl, rliqu, rliqs, rliql).
    """
    def _calc_fwet(tveg, wliq, wsno, wliqmax, wsnomax):
        xliq = wliq / jnp.maximum(wliqmax, 0.01)
        xtot = xliq + wsno / jnp.maximum(wsnomax, 0.01)
        fwet = jnp.minimum(fmax, xtot)
        rliq = xliq / jnp.maximum(xtot, epsilon)
        # When fwet==0, set rliq based on temperature
        rliq = jnp.where(
            fwet == 0.0,
            jnp.where(tveg < tmelt, 0.0, 1.0),
            rliq,
        )
        return fwet, rliq

    fwetu, rliqu = _calc_fwet(tu, wliqu, wsnou, wliqumax, wsnoumax)
    fwets, rliqs = _calc_fwet(ts, wliqs, wsnos, wliqsmax, wsnosmax)
    fwetl, rliql = _calc_fwet(tl, wliql, wsnol, wliqlmax, wsnolmax)

    return FwetResult(
        fwetu=fwetu, fwets=fwets, fwetl=fwetl,
        rliqu=rliqu, rliqs=rliqs, rliql=rliql,
    )


# ---------------------------------------------------------------------------
# mix: calorimetric mixing
# ---------------------------------------------------------------------------

def mix(x1, t1, x2, t2, x3, t3, epsilon):
    """Calorimetrically mix three masses with temperatures.

    Returns (xm, tm) where xm = x1+x2+x3 and tm is mass-weighted temp.
    """
    xm = x1 + x2 + x3
    ytmp = jnp.sign(xm) * jnp.maximum(jnp.abs(xm), epsilon)
    tm = jnp.where(
        jnp.abs(xm) >= epsilon,
        (t1 * x1 + t2 * x2 + t3 * x3) / ytmp,
        0.0,
    )
    xm = jnp.where(jnp.abs(xm) >= epsilon, xm, 0.0)
    return xm, tm


# ---------------------------------------------------------------------------
# twet3: wet bulb temperature
# ---------------------------------------------------------------------------

def twet3(tak, q, p, tmelt, cair, hvap, ch2o, cvap):
    """Approximate wet bulb temperature (K).

    Uses iterative approach from Molling (2000), 20 iterations max.
    For JAX compatibility, always runs all 20 iterations.
    """
    e = esat(tak)
    qs0 = qsat(e, p)
    ta = tak - tmelt
    twk = ta * q / jnp.maximum(qs0, 1e-12) + tmelt

    def _body(twk, _):
        twold = twk - tmelt
        hv = hvapf(twk, tak, tmelt, hvap, ch2o, cvap)
        e_tw = esat(twk)
        qs_tw = qsat(e_tw, p)
        twb = ta - (hv / cair) * (qs_tw - q)
        diff = twb - twold
        twk = twold + 0.2 * diff + tmelt
        return twk, None

    # Unroll 20 iterations
    for _ in range(20):
        twk, _ = _body(twk, None)

    return twk


# ---------------------------------------------------------------------------
# steph2o: step intercepted water for one canopy component
# ---------------------------------------------------------------------------

class StepH2OResult(NamedTuple):
    """Result from one canopy component interception step."""
    wliq: Float[Array, "npoi"]
    wsno: Float[Array, "npoi"]
    pflux: Float[Array, "npoi"]
    rain: Float[Array, "npoi"]
    train: Float[Array, "npoi"]
    snow: Float[Array, "npoi"]
    tsnow: Float[Array, "npoi"]


def steph2o(
    tveg, wliq, wsno, xai, pflux,
    rain, train, snow, tsnow,
    tdrip, tblow, wliqmax, wsnomax, wliqmin, wsnomin,
    dtime, ch2o, cice, tmelt, hfus, epsilon,
):
    """Step intercepted water for one canopy component.

    Handles interception, drip, blowoff, and phase change.
    Returns StepH2OResult.
    """
    # Intercepted fraction per unit leaf/stem area
    fint = jnp.where(xai >= epsilon,
                     (1.0 - jnp.exp(-0.5 * xai)) / xai,
                     0.5)

    rwork = 1.0 / dtime

    # Liquid
    drip = xai * wliq / tdrip
    wliq = wliq * (1.0 - dtime / tdrip)
    wliq = wliq + dtime * rain * fint
    pflux = rain * fint * (tveg - train) * ch2o
    rain = rain * (1.0 - xai * fint)
    x = wliq
    wliq = jnp.minimum(wliq, wliqmax)
    wliq = jnp.where(wliq < wliqmin, 0.0, wliq)
    drip = drip + xai * (x - wliq) * rwork

    # Snow
    blow = xai * wsno / tblow
    wsno = wsno * (1.0 - dtime / tblow)
    wsno = wsno + dtime * snow * fint
    pflux = pflux + snow * fint * (tveg - tsnow) * cice
    snow = snow * (1.0 - xai * fint)
    x = wsno
    wsno = jnp.minimum(wsno, wsnomax)
    wsno = jnp.where(wsno < wsnomin, 0.0, wsno)
    blow = blow + xai * (x - wsno) * rwork

    # Phase change: liquid below freezing
    dw_freeze = jnp.where(tveg < tmelt, wliq, 0.0)
    rwork2 = ch2o - cice
    pflux = pflux + dw_freeze * (rwork2 * (tmelt - tveg) - hfus) * rwork
    wliq = wliq - dw_freeze
    wsno = wsno + dw_freeze

    # Phase change: snow above freezing
    dw_melt = jnp.where(tveg > tmelt, wsno, 0.0)
    pflux = pflux + dw_melt * (rwork2 * (tveg - tmelt) + hfus) * rwork
    wsno = wsno - dw_melt
    wliq = wliq + dw_melt

    # Adjust rainfall/snowfall for interception, drip, blowoff
    vzero = jnp.zeros_like(rain)
    rain, train = mix(rain, train, drip, tveg, vzero, vzero, epsilon)
    snow, tsnow = mix(snow, tsnow, blow, tveg, vzero, vzero, epsilon)

    return StepH2OResult(
        wliq=wliq, wsno=wsno, pflux=pflux,
        rain=rain, train=train, snow=snow, tsnow=tsnow,
    )


# ---------------------------------------------------------------------------
# cascade: precipitation interception cascade through canopy
# ---------------------------------------------------------------------------

class CascadeResult(NamedTuple):
    """Result from cascade through all canopy levels."""
    wliqu: Float[Array, "npoi"]
    wliqs: Float[Array, "npoi"]
    wliql: Float[Array, "npoi"]
    wsnou: Float[Array, "npoi"]
    wsnos: Float[Array, "npoi"]
    wsnol: Float[Array, "npoi"]
    pfluxu: Float[Array, "npoi"]
    pfluxs: Float[Array, "npoi"]
    pfluxl: Float[Array, "npoi"]
    raing: Float[Array, "npoi"]
    traing: Float[Array, "npoi"]
    snowg: Float[Array, "npoi"]
    tsnowg: Float[Array, "npoi"]
    rainl: Float[Array, "npoi"]     # rain below upper canopy / above lower
    trainl: Float[Array, "npoi"]    # temperature of rainl
    snowl: Float[Array, "npoi"]     # snow below upper canopy / above lower
    tsnowl: Float[Array, "npoi"]    # temperature of snowl


def cascade(
    tu, ts, tl, t12, t34,
    wliqu, wliqs, wliql, wsnou, wsnos, wsnol,
    lai, sai, fu, fl,
    ta, qa, psurf, raina, snowa,
    iparams,
    dtime, tmelt, ch2o, cice, hfus, epsilon,
    cair, hvap, cvap,
):
    """Step intercepted water through all canopy levels.

    Returns CascadeResult.
    """
    npoi = tu.shape[0]
    vzero = jnp.zeros(npoi)

    # Minimum interception thresholds
    wliqmin = 0.001 * (dtime / 3600.0) * (iparams.wliqumax / 0.2)
    wsnomin = 0.001 * (dtime / 3600.0) * (iparams.wsnoumax / 2.0)

    # --- Adjust rainfall at above-tree level ---
    rainu = raina
    # Wet bulb temp for rain temperature
    twetbulb = jnp.where(
        ta > tmelt,
        twet3(ta, qa, psurf, tmelt, cair, hvap, ch2o, cvap),
        tmelt,
    )
    trainu = jnp.maximum(twetbulb, tmelt)
    rainu, trainu = mix(rainu, trainu, vzero, jnp.maximum(t12, tmelt),
                        vzero, vzero, epsilon)

    snowu = snowa
    tsnowu = jnp.minimum(ta, tmelt)
    snowu, tsnowu = mix(snowu, tsnowu, vzero, jnp.minimum(t12, tmelt),
                        vzero, vzero, epsilon)

    # --- Upper leaves ---
    xai = 2.0 * lai[:, 1]
    res_u = steph2o(
        tu, wliqu, wsnou, xai, vzero,
        rainu, trainu, snowu, tsnowu,
        iparams.tdripu, iparams.tblowu,
        iparams.wliqumax, iparams.wsnoumax, wliqmin, wsnomin,
        dtime, ch2o, cice, tmelt, hfus, epsilon,
    )
    rain, train = res_u.rain, res_u.train
    snow, tsnow = res_u.snow, res_u.tsnow

    # --- Upper stems ---
    xai = 2.0 * sai[:, 1]
    res_s = steph2o(
        ts, wliqs, wsnos, xai, vzero,
        rain, train, snow, tsnow,
        iparams.tdrips, iparams.tblows,
        iparams.wliqsmax, iparams.wsnosmax, wliqmin, wsnomin,
        dtime, ch2o, cice, tmelt, hfus, epsilon,
    )
    rain, train = res_s.rain, res_s.train
    snow, tsnow = res_s.snow, res_s.tsnow

    # --- Adjust rainfall at below-tree level ---
    rainl, trainl = mix(fu * rain, train,
                        (1.0 - fu) * rainu, trainu,
                        vzero, jnp.maximum(t34, tmelt), epsilon)
    snowl, tsnowl = mix(fu * snow, tsnow,
                        (1.0 - fu) * snowu, tsnowu,
                        vzero, jnp.minimum(t34, tmelt), epsilon)

    # --- Lower vegetation ---
    xai = 2.0 * (lai[:, 0] + sai[:, 0])
    res_l = steph2o(
        tl, wliql, wsnol, xai, vzero,
        rainl, trainl, snowl, tsnowl,
        iparams.tdripl, iparams.tblowl,
        iparams.wliqlmax, iparams.wsnolmax, wliqmin, wsnomin,
        dtime, ch2o, cice, tmelt, hfus, epsilon,
    )

    # --- Adjust rainfall at soil level ---
    raing, traing = mix(fl * res_l.rain, res_l.train,
                        (1.0 - fl) * rainl, trainl,
                        vzero, vzero, epsilon)
    snowg, tsnowg = mix(fl * res_l.snow, res_l.tsnow,
                        (1.0 - fl) * snowl, tsnowl,
                        vzero, vzero, epsilon)

    return CascadeResult(
        wliqu=res_u.wliq, wliqs=res_s.wliq, wliql=res_l.wliq,
        wsnou=res_u.wsno, wsnos=res_s.wsno, wsnol=res_l.wsno,
        pfluxu=res_u.pflux, pfluxs=res_s.pflux, pfluxl=res_l.pflux,
        raing=raing, traing=traing, snowg=snowg, tsnowg=tsnowg,
        rainl=rainl, trainl=trainl,
        snowl=snowl, tsnowl=tsnowl,
    )


# ---------------------------------------------------------------------------
# steph2o2: evaporation removal + phase correction
# ---------------------------------------------------------------------------

def steph2o2(
    tveg, wliq, wsno, fveg, xai, rliq, fvapw,
    cveg, dtime, ch2o, cice, hfus, tmelt,
    ta, fvapa, fsena, hvap_c, hsub_c, cvap_c,
):
    """Remove evaporation from intercepted water and do phase correction.

    Returns updated (tveg, wliq, wsno, fvapa, fsena).
    """
    # Step intercepted water due to evaporation/sublimation
    wliq = wliq - dtime * rliq * fvapw
    wsno = wsno - dtime * (1.0 - rliq) * fvapw

    # Fix negative water: transfer deficit to atmosphere
    has_veg = fveg * xai > 0.0
    neg_liq = jnp.maximum(-wliq, 0.0)
    zm_liq = jnp.where(has_veg, neg_liq * fveg * xai / dtime, 0.0)
    fvapa = fvapa + zm_liq
    fsena = fsena - zm_liq * hvapf(tveg, ta, tmelt, hvap_c, ch2o, cvap_c)
    wliq = jnp.maximum(wliq, 0.0)

    neg_sno = jnp.maximum(-wsno, 0.0)
    zm_sno = jnp.where(has_veg, neg_sno * fveg * xai / dtime, 0.0)
    fvapa = fvapa + zm_sno
    fsena = fsena - zm_sno * hsubf(tveg, ta, tmelt, hsub_c, cvap_c, cice)
    wsno = jnp.maximum(wsno, 0.0)

    # Final heat-conserving phase correction
    rwork = 1.0 / hfus
    chav = cveg + ch2o * wliq + cice * wsno

    # Liquid below freezing → freeze
    below = (tveg < tmelt) & (wliq > 0.0)
    dh = chav * (tmelt - tveg)
    dw = jnp.minimum(wliq, jnp.maximum(0.0, dh * rwork))
    dw = jnp.where(below, dw, 0.0)
    wliq = wliq - dw
    wsno = wsno + dw
    chav_new = cveg + ch2o * wliq + cice * wsno
    tveg = jnp.where(below, tmelt - (dh - hfus * dw) / chav_new, tveg)

    # Snow above freezing → melt
    above = (tveg > tmelt) & (wsno > 0.0)
    chav = cveg + ch2o * wliq + cice * wsno
    dh = chav * (tveg - tmelt)
    dw = jnp.minimum(wsno, jnp.maximum(0.0, dh * rwork))
    dw = jnp.where(above, dw, 0.0)
    wsno = wsno - dw
    wliq = wliq + dw
    chav_new = cveg + ch2o * wliq + cice * wsno
    tveg = jnp.where(above, tmelt + (dh - hfus * dw) / chav_new, tveg)

    return tveg, wliq, wsno, fvapa, fsena


# ---------------------------------------------------------------------------
# cascad2: end-of-timestep evaporation cascade
# ---------------------------------------------------------------------------

def cascad2(
    tu, ts, tl,
    wliqu, wliqs, wliql, wsnou, wsnos, wsnol,
    rliqu, rliqs, rliql,
    fvapuw, fvaps, fvaplw,
    fu, fl, fi, lai, sai,
    chu, chs, chl,
    dtime, ch2o, cice, hfus, tmelt,
    ta, fvapa, fsena, hvap_c, hsub_c, cvap_c,
):
    """Remove evaporation from all canopy components.

    Returns updated (tu,ts,tl, wliqu..wsnol, fvapa, fsena).
    """
    # Upper leaves
    fveg_u = fu
    xai_u = 2.0 * lai[:, 1]
    tu, wliqu, wsnou, fvapa, fsena = steph2o2(
        tu, wliqu, wsnou, fveg_u, xai_u, rliqu, fvapuw,
        chu, dtime, ch2o, cice, hfus, tmelt,
        ta, fvapa, fsena, hvap_c, hsub_c, cvap_c,
    )

    # Upper stems
    xai_s = 2.0 * sai[:, 1]
    ts, wliqs, wsnos, fvapa, fsena = steph2o2(
        ts, wliqs, wsnos, fveg_u, xai_s, rliqs, fvaps,
        chs, dtime, ch2o, cice, hfus, tmelt,
        ta, fvapa, fsena, hvap_c, hsub_c, cvap_c,
    )

    # Lower veg
    fveg_l = (1.0 - fi) * fl
    xai_l = 2.0 * (lai[:, 0] + sai[:, 0])
    tl, wliql, wsnol, fvapa, fsena = steph2o2(
        tl, wliql, wsnol, fveg_l, xai_l, rliql, fvaplw,
        chl, dtime, ch2o, cice, hfus, tmelt,
        ta, fvapa, fsena, hvap_c, hsub_c, cvap_c,
    )

    return tu, ts, tl, wliqu, wliqs, wliql, wsnou, wsnos, wsnol, fvapa, fsena


# ---------------------------------------------------------------------------
# noveg: set temperatures when no vegetation
# ---------------------------------------------------------------------------

def noveg(tu, ts, tl, wliqu, wliqs, wliql, wsnou, wsnos, wsnol,
          lai, sai, fu, fl, fi, tg, ti):
    """Set vegetation temperatures to surface average where no veg exists."""
    tav = (1.0 - fi) * tg + fi * ti

    # Upper leaves: no LAI or no fu
    no_upper_leaf = (lai[:, 1] == 0.0) | (fu == 0.0)
    tu = jnp.where(no_upper_leaf, tav, tu)
    wliqu = jnp.where(no_upper_leaf, 0.0, wliqu)
    wsnou = jnp.where(no_upper_leaf, 0.0, wsnou)

    # Upper stems: no SAI or no fu
    no_upper_stem = (sai[:, 1] == 0.0) | (fu == 0.0)
    ts = jnp.where(no_upper_stem, tav, ts)
    wliqs = jnp.where(no_upper_stem, 0.0, wliqs)
    wsnos = jnp.where(no_upper_stem, 0.0, wsnos)

    # Lower veg: no leaf+stem area or no effective cover
    x = 2.0 * (lai[:, 0] + sai[:, 0])
    y = fl * (1.0 - fi)
    no_lower = (x == 0.0) | (y == 0.0)
    tl = jnp.where(no_lower, tav, tl)
    wliql = jnp.where(no_lower, 0.0, wliql)
    wsnol = jnp.where(no_lower, 0.0, wsnol)

    return tu, ts, tl, wliqu, wliqs, wliql, wsnou, wsnos, wsnol
