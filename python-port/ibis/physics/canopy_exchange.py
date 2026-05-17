"""Canopy energy balance and turbulent exchange for IBIS-JAX.

Ported from canopy.f90.  All functions are pure, JIT-compatible, and
differentiable.  The 9x9 implicit system is solved with a JAX port of
F77's no-pivot Gaussian elimination (math.f:linsolve) — see _f77_linsolve.

Functions:
  canini   -- initialise aerodynamic geometry (heights, log profiles)
  fstrat   -- stratification correction (Louis 1979)
  turcof   -- wind speeds and transfer coefficients
  impexp   -- implicit/explicit fraction for phase-change near tmelt
  turvap   -- build and solve 9x9 implicit heat/moisture system
  canopy_solve -- top-level 3-iteration controller
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Float, Array

from ibis.physics.thermo import esat, qsat, dqsat, hvapf, hsubf


def _f77_linsolve(arr, rhs):
    """No-pivot Gaussian elimination matching F77 math.f:linsolve loop order.

    JAX's jnp.linalg.solve dispatches to LAPACK gesv (partial pivoting).
    LAPACK and F77's bare no-pivot Gauss produce mathematically equivalent
    results but in different FP operation orders. For most cells these
    coincide; for the near-singular freeze-thaw matrices (snow exactly at
    tmelt + rain), the pivoting choice introduces ~5 mK divergence in
    turvap state. Using the same loop order as F77 brings divergence down
    to ~10 μK — see verify/STATUS.md finding #17.

    F77 also uses a sparsity template (mplate) to skip row updates on
    positions that start as zero in the 9×9 turvap matrix. Skipping is
    mathematically equivalent in IEEE arithmetic (A[i,j] − f·0 = A[i,j]
    exactly) and was tested in finding #19; it does not reduce the
    L+25p0_+100p0 residual, and the vectorised whole-row update below is
    faster under XLA than per-column conditional updates would be.

    arr: (npoi, nd, nd), rhs: (npoi, nd)  →  vec: (npoi, nd)
    """
    npoi, nd, _ = arr.shape
    A = arr
    b = rhs
    for id_ in range(nd - 1):
        diag = A[:, id_, id_]
        for i in range(id_ + 1, nd):
            f = A[:, i, id_] / diag
            A = A.at[:, i, :].set(A[:, i, :] - f[:, None] * A[:, id_, :])
            b = b.at[:, i].set(b[:, i] - f * b[:, id_])
    vec = jnp.zeros_like(b)
    for id_ in range(nd - 1, -1, -1):
        s = jnp.zeros(npoi)
        for j in range(id_ + 1, nd):
            s = s + A[:, id_, j] * vec[:, j]
        vec = vec.at[:, id_].set((b[:, id_] - s) / A[:, id_, id_])
    return vec


# ---------------------------------------------------------------------------
# CanopyParams: static canopy/aerodynamic parameters
# ---------------------------------------------------------------------------

class CanopyParams(NamedTuple):
    """Static canopy parameters read from params.can."""
    cleaf: float = 0.01       # leaf-air transfer coefficient
    cstem: float = 0.01       # stem-air transfer coefficient
    cgrass: float = 0.01      # lower canopy-air transfer coefficient
    dleaf: tuple = (0.10, 0.10)   # leaf dimension (lower, upper) (m)
    dstem: tuple = (0.10, 0.10)   # stem dimension (lower, upper) (m)
    alaiml: float = 8.0       # lower canopy max LAI+SAI (2-sided)
    alaimu: float = 8.0       # upper canopy max LAI+SAI (2-sided)
    chu: float = 8436.0       # heat capacity, upper leaves (J/m2/K)
    chs: float = 2.109e5      # heat capacity, upper stems (J/m2/K)
    chl: float = 8436.0       # heat capacity, lower canopy (J/m2/K)


# ---------------------------------------------------------------------------
# Aerodynamic initialisation result
# ---------------------------------------------------------------------------

class AeroState(NamedTuple):
    """Aerodynamic geometry computed by canini."""
    rhoa: Float[Array, "npoi"]
    cp: Float[Array, "npoi"]
    za: Float[Array, "npoi"]
    tfac: float
    z1: Float[Array, "npoi"]
    z2: Float[Array, "npoi"]
    z3: Float[Array, "npoi"]
    z4: Float[Array, "npoi"]
    z12: Float[Array, "npoi"]
    z34: Float[Array, "npoi"]
    bdl: Float[Array, "npoi"]
    bdu: Float[Array, "npoi"]
    dil: Float[Array, "npoi"]
    diu: Float[Array, "npoi"]
    exphl: Float[Array, "npoi"]
    expl: Float[Array, "npoi"]
    exphu: Float[Array, "npoi"]
    expu: Float[Array, "npoi"]
    displ: Float[Array, "npoi"]
    dispu: Float[Array, "npoi"]
    alogav: Float[Array, "npoi"]
    alog1: Float[Array, "npoi"]
    alog2: Float[Array, "npoi"]
    alog3: Float[Array, "npoi"]
    alog4: Float[Array, "npoi"]
    aloga: Float[Array, "npoi"]
    alogu: Float[Array, "npoi"]
    alogl: Float[Array, "npoi"]
    u2: Float[Array, "npoi"]


# ---------------------------------------------------------------------------
# Transfer coefficient result
# ---------------------------------------------------------------------------

class TransferCoeffs(NamedTuple):
    """Wind speeds and transfer coefficients from turcof."""
    cu: Float[Array, "npoi"]
    cl: Float[Array, "npoi"]
    su: Float[Array, "npoi"]
    ss: Float[Array, "npoi"]
    sl: Float[Array, "npoi"]
    sg: Float[Array, "npoi"]
    si: Float[Array, "npoi"]
    u2: Float[Array, "npoi"]
    alogu: Float[Array, "npoi"]
    alogl: Float[Array, "npoi"]
    straml: Float[Array, "npoi"]
    strahl: Float[Array, "npoi"]
    stramu: Float[Array, "npoi"]
    strahu: Float[Array, "npoi"]


# ---------------------------------------------------------------------------
# Canopy solver result
# ---------------------------------------------------------------------------

class CanopyResult(NamedTuple):
    """Output from the 9x9 canopy solve (turvap)."""
    # Updated temperatures and humidities
    tu: Float[Array, "npoi"]
    ts: Float[Array, "npoi"]
    tl: Float[Array, "npoi"]
    t12: Float[Array, "npoi"]
    t34: Float[Array, "npoi"]
    q12: Float[Array, "npoi"]
    q34: Float[Array, "npoi"]
    tg: Float[Array, "npoi"]
    ti: Float[Array, "npoi"]
    # Sensible heat fluxes
    fsena: Float[Array, "npoi"]
    fsenu: Float[Array, "npoi"]
    fsens: Float[Array, "npoi"]
    fsenl: Float[Array, "npoi"]
    fseng: Float[Array, "npoi"]
    fseni: Float[Array, "npoi"]
    # Moisture fluxes
    fvapa: Float[Array, "npoi"]
    fvapuw: Float[Array, "npoi"]
    fvaput: Float[Array, "npoi"]
    fvaps: Float[Array, "npoi"]
    fvaplw: Float[Array, "npoi"]
    fvaplt: Float[Array, "npoi"]
    fvapg: Float[Array, "npoi"]
    fvapi: Float[Array, "npoi"]
    # IR adjustments
    firb: Float[Array, "npoi"]
    firg: Float[Array, "npoi"]
    firi: Float[Array, "npoi"]
    # Aggregated fluxes
    ginvap: Float[Array, "npoi"]
    gsuvap: Float[Array, "npoi"]
    gtrans: Float[Array, "npoi"]


# ---------------------------------------------------------------------------
# canini: initialise aerodynamic geometry
# ---------------------------------------------------------------------------

def canini(
    ta, qa, psurf, ua,
    fi, z0soi, z0sno,
    fl, fu, lai, sai, ztop, zbot,
    alaiml, alaimu,
    grav, rair, rvap, cair, cvap, cappa,
):
    """Initialise aerodynamic geometry for canopy exchange.

    Returns an AeroState with heights, displacement heights,
    exponential profile parameters, and log-layer variables.
    """
    siga = 0.999
    tfac = 1.0 / (siga ** cappa)

    pa = psurf * siga
    rhoa = pa / (rair * ta * (1.0 + (rvap / rair - 1.0) * qa))
    cp = cair * (1.0 + (cvap / cair - 1.0) * qa)
    za = (psurf - pa) / (rhoa * grav)
    za = jnp.maximum(za, ztop[:, 1] + 1.0)

    # --- Lower story aerodynamics ---
    x = fl * (1.0 - fi) * 2.0 * (lai[:, 0] + sai[:, 0]) / alaiml
    x = jnp.minimum(x, 3.0)
    x1 = jnp.minimum(x, 1.0)

    rwork = jnp.maximum(ztop[:, 0] - zbot[:, 0], 0.01)
    cvegl = (0.4 / rwork) * jnp.maximum(1e-5, x)
    dvegl = (0.1 * rwork) / jnp.maximum(1e-5, jnp.maximum(x, x ** 2))
    bvegl = jnp.sqrt(2.0 * cvegl / dvegl)

    bdl = 0.5 * bvegl * dvegl
    dil = 1.0 / dvegl

    rwork = (1.0 - x1) * (jnp.maximum(z0soi, z0sno) + 0.01)
    z3 = x1 * ztop[:, 0] + rwork
    z4 = x1 * zbot[:, 0] + rwork
    z34 = 0.5 * (z3 + z4)

    exphl = jnp.exp(0.5 * bvegl * (z3 - z4))
    expl = exphl ** 2
    displ = x1 * 0.7 * z3

    # --- Upper story aerodynamics ---
    x = fu * 2.0 * (lai[:, 1] + sai[:, 1]) / alaimu
    x = jnp.minimum(x, 3.0)
    x1 = jnp.minimum(x, 1.0)

    rwork = jnp.maximum(ztop[:, 1] - zbot[:, 1], 0.01)
    cvegu = (0.4 / rwork) * jnp.maximum(1e-5, x)
    dvegu = (0.1 * rwork) / jnp.maximum(1e-5, jnp.maximum(x, x ** 2))
    rwork = 1.0 / dvegu
    bvegu = jnp.sqrt(2.0 * cvegu * rwork)
    bdu = 0.5 * bvegu * dvegu
    diu = rwork

    rwork = (1.0 - x1) * (z3 + 0.01)
    z1 = x1 * ztop[:, 1] + rwork
    z2 = x1 * zbot[:, 1] + rwork
    z12 = 0.5 * (z1 + z2)

    exphu = jnp.exp(0.5 * bvegu * (z1 - z2))
    expu = exphu ** 2
    dispu = x1 * 0.7 * z1 + (1.0 - x1) * displ

    # --- Mixing-length logarithms ---
    alogg = jnp.log(z0soi)
    alogi = jnp.log(z0sno)
    alogav = (1.0 - fi) * alogg + fi * alogi

    alog4 = jnp.log(jnp.maximum(z4, jnp.maximum(1.1 * z0soi, 1.1 * z0sno)))
    alog3 = jnp.log(z3 - displ)
    alog2 = jnp.log(z2 - displ)
    alog1 = jnp.log(z1 - dispu)
    aloga = jnp.log(za - dispu)

    u2 = ua / exphu
    alogu = jnp.log(jnp.maximum(0.01, 0.1 * (z1 - z2)))
    alogl = jnp.log(jnp.maximum(0.01, 0.1 * (z3 - z4)))

    return AeroState(
        rhoa=rhoa, cp=cp, za=za, tfac=tfac,
        z1=z1, z2=z2, z3=z3, z4=z4, z12=z12, z34=z34,
        bdl=bdl, bdu=bdu, dil=dil, diu=diu,
        exphl=exphl, expl=expl, exphu=exphu, expu=expu,
        displ=displ, dispu=dispu, alogav=alogav,
        alog1=alog1, alog2=alog2, alog3=alog3, alog4=alog4,
        aloga=aloga, alogu=alogu, alogl=alogl, u2=u2,
    )


# ---------------------------------------------------------------------------
# fstrat: Louis (1979) stratification correction
# ---------------------------------------------------------------------------

def fstrat(
    tb, tt, ttfac, qb, qt,
    zb, zt, albm, albh, alt, u,
    grav, vonk, iter_num,
    rich_prev, stram_prev, strah_prev,
):
    """Stratification correction factors (Louis 1979).

    Returns (rich, stram, strah).
    """
    zht = tt * ttfac * (1.0 + 0.622 * qt)
    zhb = tb * (1.0 + 0.622 * qb)
    rich = grav * jnp.maximum(zt - zb, 0.0) * (zht - zhb) / (
        0.5 * (zht + zhb) * u ** 2
    )
    rich = jnp.clip(rich, -2.0, 1.0)

    # Unstable (Ri <= 0)
    xm = jnp.maximum(alt - albm, 0.5)
    xh = jnp.maximum(alt - albh, 0.5)
    rwork = jnp.sqrt(jnp.maximum(-rich, 0.0))
    ym = (vonk / xm) ** 2 * jnp.exp(0.5 * xm) * rwork
    yh = (vonk / xh) ** 2 * jnp.exp(0.5 * xh) * rwork
    stram_unstable = 1.0 - 10.0 * rich / (1.0 + 75.0 * ym)
    strah_unstable = 1.0 - 15.0 * rich / (1.0 + 75.0 * yh)

    # Stable (Ri > 0)
    z = jnp.sqrt(1.0 + 5.0 * jnp.maximum(rich, 0.0))
    stram_stable = 1.0 / (1.0 + 10.0 * jnp.maximum(rich, 0.0) / z)
    strah_stable = 1.0 / (1.0 + 15.0 * jnp.maximum(rich, 0.0) * z)

    stram_new = jnp.where(rich <= 0.0, stram_unstable, stram_stable)
    strah_new = jnp.where(rich <= 0.0, strah_unstable, strah_stable)

    # Weight with previous iteration (except first)
    is_first = (iter_num == 0)
    stram = jnp.where(is_first, stram_new, 0.5 * stram_new + 0.5 * stram_prev)
    strah = jnp.where(is_first, strah_new, 0.5 * strah_new + 0.5 * strah_prev)

    return rich, stram, strah


# ---------------------------------------------------------------------------
# turcof: wind speeds and transfer coefficients
# ---------------------------------------------------------------------------

def turcof(
    aero, ta, qa, ua,
    t12, t34, q12, q34, fi,
    vonk, dtime,
    cleaf, cstem, cgrass, dleaf, dstem,
    iter_num,
    richl_prev, straml_prev, strahl_prev,
    richu_prev, stramu_prev, strahu_prev,
    grav,
):
    """Compute wind speeds and aerodynamic transfer coefficients.

    Returns TransferCoeffs.
    """
    # Stratification corrections
    richl, straml, strahl = fstrat(
        t34, t12, 1.0, q34, q12,
        aero.z3, aero.z2, aero.alogl, aero.alogl, aero.alog2, aero.u2,
        grav, vonk, iter_num, richl_prev, straml_prev, strahl_prev,
    )
    richu, stramu, strahu = fstrat(
        t12, ta, aero.tfac, q12, qa,
        aero.z1, aero.za, aero.alogu, aero.alogu, aero.aloga, ua,
        grav, vonk, iter_num, richu_prev, stramu_prev, strahu_prev,
    )

    # Lower-story effective roughness
    x = ((aero.alog4 - aero.alogav) / vonk) ** 2 * aero.bdl
    rwork = 1.0 / aero.expl
    yl = ((x + 1) * aero.expl + (x - 1) * rwork) / (
         (x + 1) * aero.expl - (x - 1) * rwork)
    alogl = aero.alog3 - vonk * jnp.sqrt(yl / aero.bdl)

    # Upper-story effective roughness
    x = ((aero.alog2 - alogl) / vonk) ** 2 * aero.bdu / straml
    rwork = 1.0 / aero.expu
    yu = ((x + 1) * aero.expu + (x - 1) * rwork) / (
         (x + 1) * aero.expu - (x - 1) * rwork)
    alogu = aero.alog1 - vonk * jnp.sqrt(yu / aero.bdu)

    cdmax = 300.0 / (2.0 * dtime)

    # Wind speeds at various levels
    tauu = (ua * vonk / (aero.aloga - alogu)) ** 2 * stramu
    a = 0.5 * tauu * (yu + 1) / aero.bdu
    b = 0.5 * tauu * (yu - 1) / aero.bdu
    taul = aero.bdu * (a / aero.expu - b * aero.expu)
    c = 0.5 * taul * (yl + 1) / aero.bdl
    d = 0.5 * taul * (yl - 1) / aero.bdl

    _usafe = lambda v: jnp.maximum(0.01, jnp.sqrt(jnp.maximum(0.0, v)))
    u1 = _usafe(a + b)
    u12 = _usafe(a / aero.exphu + b * aero.exphu)
    u2 = _usafe(a / aero.expu + b * aero.expu)
    u3 = _usafe(c + d)
    u34 = _usafe(c / aero.exphl + d * aero.exphl)
    u4 = _usafe(c / aero.expl + d * aero.expl)

    # Air-air transfer coefficients
    ca = ua * strahu * vonk ** 2 / (
        (aero.aloga - alogu) * (aero.aloga - aero.alog1))
    ca = jnp.minimum(cdmax, ca / (1.0 + ca * 1e-20))
    cai = 1.0 / (aero.rhoa * ca)

    cbi = aero.diu * (aero.z1 - aero.z12) / (
        aero.rhoa * 0.5 * (u1 + u12))
    cci = aero.diu * (aero.z12 - aero.z2) / (
        aero.rhoa * 0.5 * (u12 + u2))
    cdi = (aero.alog2 - alogl) * (aero.alog2 - aero.alog3) / (
        aero.rhoa * u2 * strahl * vonk ** 2)
    cei = aero.dil * (aero.z3 - aero.z34) / (
        aero.rhoa * 0.5 * (u3 + u34))
    cfi = aero.dil * (aero.z34 - aero.z4) / (
        aero.rhoa * 0.5 * (u34 + u4))

    cu = 1.0 / (cai + cbi)
    cl = 1.0 / (cci + cdi + cei)

    # Air-solid transfer coefficients
    su = aero.rhoa * cleaf * jnp.sqrt(u12 / dleaf[1])
    ss = aero.rhoa * cstem * jnp.sqrt(u12 / dstem[1])
    sl = aero.rhoa * cgrass * jnp.sqrt(u34 / dleaf[0])

    # Soil and snow
    sg0 = aero.rhoa * (0.004 + 0.012 * u4)
    si0 = aero.rhoa * (0.003 + 0.010 * u4)
    sg = 1.0 / (cfi + 1.0 / sg0)
    si = 1.0 / (cfi + 1.0 / si0)

    return TransferCoeffs(
        cu=cu, cl=cl, su=su, ss=ss, sl=sl, sg=sg, si=si,
        u2=u2, alogu=alogu, alogl=alogl,
        straml=straml, strahl=strahl, stramu=stramu, strahu=strahu,
    )


# ---------------------------------------------------------------------------
# impexp: implicit/explicit fraction
# ---------------------------------------------------------------------------

def impexp(tveg, ch, wliq, wsno, tmelt, hfus, epsilon, wimp_prev, is_first):
    """Compute implicit/explicit blending weight for vegetation surfaces."""
    h = ch * (tveg - tmelt)
    z = jnp.maximum(jnp.abs(h), epsilon)
    winew = jnp.where(
        h > epsilon,
        1.0 - jnp.minimum(1.0, hfus * wsno / z),
        jnp.where(h < -epsilon,
                  1.0 - jnp.minimum(1.0, hfus * wliq / z),
                  1.0)
    )
    return jnp.where(is_first, 1.0, 0.5 * (wimp_prev + winew))


def impexp2(t, told, tmelt, epsilon, wimp_prev, is_first):
    """Compute implicit/explicit fraction for snow skin temperature."""
    winew = jnp.where(
        (t - told) > epsilon,
        (tmelt - told) / (t - told),
        wimp_prev,
    )
    winew = jnp.clip(winew, 0.0, 1.0)
    return jnp.where(is_first, 1.0, winew)


# ---------------------------------------------------------------------------
# turvap: build and solve 9x9 implicit heat/moisture system
# ---------------------------------------------------------------------------

def turvap(
    # Current temperatures and humidity
    tu, ts, tl, t12, t34, q12, q34, tg, ti,
    # Saved original temperatures
    tuold, tsold, tlold, tgold, tiold,
    # Transfer coefficients
    cu, cl, su, ss, sl, sg, si,
    cp, rhoa, tfac,
    # Canopy structure weights
    xu, xs, xl, fi, fu, fl, lai, sai, frac,
    # Heat capacities
    chux, chsx, chlx,
    # Intercepted water state
    wliqu, wliqs, wliql, wsnou, wsnos, wsnol,
    rliqu, rliqs, rliql,
    fwetu, fwets, fwetl,
    # Stomatal conductance (aggregated)
    totcondub, totconduc, totcondls, totcondl3, totcondl4,
    # Ground thermal properties
    chgx, cog, coi, zirg, ziri,
    # Soil moisture for qgfac
    wpud, wipud, wpudmax, wisoi_1, wsoi_1,
    swilt_1, suction_1, bex_1,
    hvasug, hvasui,
    # Lower boundary temperatures
    tsoi_1, tsno_1,
    # Radiation fluxes
    solu, sols, soll, solg, soli,
    firu, firs, firl, firg, firi, firb,
    pfluxu, pfluxs, pfluxl,
    # Atmospheric forcing
    ta, qa, psurf,
    # Implicit weights
    wu, ws, wl, wg, wi,
    # Physical constants
    dtime, tmelt, hvap_c, hsub_c, ch2o_c, cice_c,
    cvap_c, stef_c, rvap_c, grav_c, epsilon_c,
):
    """Build and solve 9x9 implicit canopy system.

    Variables: tu(0), ts(1), tl(2), t12(3), t34(4), q12(5), q34(6), tg(7), ti(8)

    Returns updated 9 variables plus diagnostic fluxes.
    """
    npoi = tu.shape[0]

    # Adjust temperatures with implicit/explicit blend
    tu = wu * tu + (1.0 - wu) * tuold
    ts = ws * ts + (1.0 - ws) * tsold
    tl = wl * tl + (1.0 - wl) * tlold
    tg = wg * tg + (1.0 - wg) * tgold
    ti = wi * ti + (1.0 - wi) * tiold

    # Save pre-solve values for final flux calculation
    tupre, tspre, tlpre, tgpre, tipre = tu, ts, tl, tg, ti

    # --- Saturation humidities and derivatives ---
    eu = esat(tu);  qu = qsat(eu, psurf);  dqu = jnp.minimum(dqsat(tu, qu, eu), qu * 0.1)
    es = esat(ts);  qs = qsat(es, psurf);  dqs = jnp.minimum(dqsat(ts, qs, es), qs * 0.1)
    el = esat(tl);  ql = qsat(el, psurf);  dql = jnp.minimum(dqsat(tl, ql, el), ql * 0.1)
    eg = esat(tg);  qg = qsat(eg, psurf);  dqg = jnp.minimum(dqsat(tg, qg, eg), qg * 0.1)
    ei = esat(ti);  qi = qsat(ei, psurf);  dqi = jnp.minimum(dqsat(ti, qi, ei), qi * 0.1)

    # --- Soil surface humidity factor qgfac ---
    zwpud = jnp.clip(0.5 * (wpud + wipud) / jnp.maximum(wpudmax, epsilon_c), 0.0, 0.5)
    zwsoi = (1.0 - wisoi_1) * wsoi_1 + wisoi_1
    zwtot = zwpud + (1.0 - zwpud) * zwsoi

    psig = -grav_c * suction_1 * (zwtot ** (-bex_1))
    hfac = jnp.exp(psig / (rvap_c * tg))
    hfac2 = q34 / jnp.maximum(qg, epsilon_c)

    betaw = jnp.clip((zwtot - swilt_1) / jnp.maximum(1.0 - swilt_1, epsilon_c), 0.0, 1.0)

    emisoil = 0.95
    e34 = esat(t34)
    qs1 = qsat(e34, psurf)
    dqs1 = dqsat(t34, qs1)
    xnumer = hvap_c * dqs1
    xdenom = cp + (4.0 * emisoil * stef_c * t34 ** 3) / sg
    betafac = xnumer / xdenom
    betas = betaw / (1.0 + betafac * (1.0 - betaw))
    qgfac0 = betas * hfac + (1.0 - betas) * hfac2
    qgfac = jnp.where(q34 > qg, 1.0, qgfac0)

    # --- Wet-fraction modifiers ---
    fwetux = jnp.where(q12 > qu, 1.0, fwetu)
    fwetsx = jnp.where(q12 > qs, 1.0, fwets)
    fwetlx = jnp.where(q34 > ql, 1.0, fwetl)

    # --- Absorbed radiation per unit leaf/stem area ---
    fradu = jnp.where(lai[:, 1] > epsilon_c,
                      (solu + firu) / (2.0 * lai[:, 1]), 0.0)
    frads = jnp.where(sai[:, 1] > epsilon_c,
                      (sols + firs) / (2.0 * sai[:, 1]), 0.0)
    fradl = jnp.where((lai[:, 0] + sai[:, 0]) > epsilon_c,
                      (soll + firl) / (2.0 * (lai[:, 0] + sai[:, 0])), 0.0)

    # --- Moisture transfer coefficients ---
    suw = jnp.minimum(
        fwetux * su,
        0.8 * (wliqu + wsnou) / jnp.maximum(dtime * (qu - q12), epsilon_c))
    sut = jnp.maximum(0.0, (1.0 - fwetux) * 0.5 * (
        totcondub * frac[:, 0] + totcondub * frac[:, 1] +
        totcondub * frac[:, 2] + totconduc * frac[:, 3] +
        totcondub * frac[:, 4] + totconduc * frac[:, 5] +
        totcondub * frac[:, 6] + totcondub * frac[:, 7]))

    suh = (suw * (rliqu * hvapf(tu, ta, tmelt, hvap_c, ch2o_c, cvap_c)
                  + (1.0 - rliqu) * hsubf(tu, ta, tmelt, hsub_c, cvap_c, cice_c))
           + sut * hvapf(tu, ta, tmelt, hvap_c, ch2o_c, cvap_c))

    ssw = jnp.minimum(
        fwetsx * ss,
        0.8 * (wliqs + wsnos) / jnp.maximum(dtime * (qs - q12), epsilon_c))
    ssh = ssw * (rliqs * hvapf(ts, ta, tmelt, hvap_c, ch2o_c, cvap_c)
                 + (1.0 - rliqs) * hsubf(ts, ta, tmelt, hsub_c, cvap_c, cice_c))

    slw = jnp.minimum(
        fwetlx * sl,
        0.8 * (wliql + wsnol) / jnp.maximum(dtime * (ql - q34), epsilon_c))
    slt0 = jnp.maximum(0.0, (1.0 - fwetlx) * 0.5 * (
        totcondls * frac[:, 8] + totcondls * frac[:, 9] +
        totcondl4 * frac[:, 10] + totcondl3 * frac[:, 11]))
    slt = slt0 * lai[:, 0] / jnp.maximum(lai[:, 0] + sai[:, 0], epsilon_c)
    slh = (slw * (rliql * hvapf(tl, ta, tmelt, hvap_c, ch2o_c, cvap_c)
                  + (1.0 - rliql) * hsubf(tl, ta, tmelt, hsub_c, cvap_c, cice_c))
           + slt * hvapf(tl, ta, tmelt, hvap_c, ch2o_c, cvap_c))

    # --- Build 9x9 coefficient matrix and RHS ---
    arr = jnp.zeros((npoi, 9, 9))
    rhs = jnp.zeros((npoi, 9))
    rwork = 1.0 / dtime

    # Row 0: tu
    rwork2 = su * cp
    arr = arr.at[:, 0, 0].set(chux * rwork + wu * rwork2 + wu * suh * dqu)
    arr = arr.at[:, 0, 3].set(-rwork2)
    arr = arr.at[:, 0, 5].set(-suh)
    rhs = rhs.at[:, 0].set(tuold * chux * rwork - (1.0 - wu) * rwork2 * tu
                            - suh * (qu - wu * dqu * tu) + fradu - pfluxu)

    # Row 1: ts
    rwork2 = ss * cp
    arr = arr.at[:, 1, 1].set(chsx * rwork + ws * rwork2 + ws * ssh * dqs)
    arr = arr.at[:, 1, 3].set(-rwork2)
    arr = arr.at[:, 1, 5].set(-ssh)
    rhs = rhs.at[:, 1].set(tsold * chsx * rwork - (1.0 - ws) * rwork2 * ts
                            - ssh * (qs - ws * dqs * ts) + frads - pfluxs)

    # Row 2: tl
    rwork2 = sl * cp
    arr = arr.at[:, 2, 2].set(chlx * rwork + wl * rwork2 + wl * slh * dql)
    arr = arr.at[:, 2, 4].set(-rwork2)
    arr = arr.at[:, 2, 6].set(-slh)
    rhs = rhs.at[:, 2].set(tlold * chlx * rwork - (1.0 - wl) * rwork2 * tl
                            - slh * (ql - wl * dql * tl) + fradl - pfluxl)

    # Row 3: t12
    rwork_u = xu * su
    rwork_s = xs * ss
    arr = arr.at[:, 3, 0].set(-wu * rwork_u)
    arr = arr.at[:, 3, 1].set(-ws * rwork_s)
    arr = arr.at[:, 3, 3].set(cu + cl + rwork_u + rwork_s)
    arr = arr.at[:, 3, 4].set(-cl)
    rhs = rhs.at[:, 3].set(cu * ta * tfac
                            + (1.0 - wu) * rwork_u * tu
                            + (1.0 - ws) * rwork_s * ts)

    # Row 4: t34
    rwork_l = xl * sl
    rwork_i = fi * si
    arr = arr.at[:, 4, 2].set(-wl * rwork_l)
    arr = arr.at[:, 4, 3].set(-cl)
    arr = arr.at[:, 4, 4].set(cl + rwork_l + (1.0 - fi) * sg + rwork_i)
    arr = arr.at[:, 4, 7].set(-wg * (1.0 - fi) * sg)
    arr = arr.at[:, 4, 8].set(-wi * rwork_i)
    rhs = rhs.at[:, 4].set((1.0 - wl) * rwork_l * tl
                            + (1.0 - wg) * (1.0 - fi) * sg * tg
                            + (1.0 - wi) * rwork_i * ti)

    # Row 5: q12
    rwork_uq = xu * (suw + sut)
    rwork_sq = xs * ssw
    arr = arr.at[:, 5, 0].set(-wu * rwork_uq * dqu)
    arr = arr.at[:, 5, 1].set(-ws * rwork_sq * dqs)
    arr = arr.at[:, 5, 5].set(cu + cl + rwork_uq + rwork_sq)
    arr = arr.at[:, 5, 6].set(-cl)
    rhs = rhs.at[:, 5].set(cu * qa + rwork_uq * (qu - wu * dqu * tu)
                            + rwork_sq * (qs - ws * dqs * ts))

    # Row 6: q34
    rwork_lq = xl * (slw + slt)
    rwork_gq = (1.0 - fi) * sg
    arr = arr.at[:, 6, 2].set(-wl * rwork_lq * dql)
    arr = arr.at[:, 6, 5].set(-cl)
    arr = arr.at[:, 6, 6].set(cl + rwork_lq + rwork_gq + fi * si)
    arr = arr.at[:, 6, 7].set(-wg * rwork_gq * qgfac * dqg)
    arr = arr.at[:, 6, 8].set(-wi * fi * si * dqi)
    rhs = rhs.at[:, 6].set(rwork_lq * (ql - wl * dql * tl)
                            + rwork_gq * qgfac * (qg - wg * dqg * tg)
                            + fi * si * (qi - wi * dqi * ti))

    # Row 7: tg
    rwork_sg = sg * cp
    rwork_sg2 = sg * hvasug
    arr = arr.at[:, 7, 4].set(-rwork_sg)
    arr = arr.at[:, 7, 6].set(-rwork_sg2)
    arr = arr.at[:, 7, 7].set(rwork_sg + rwork_sg2 * qgfac * dqg + cog + zirg)
    rhs = rhs.at[:, 7].set(-rwork_sg2 * qgfac * (qg - dqg * tg)
                            + cog * tsoi_1 + solg + firg + zirg * tgold)

    # Row 8: ti
    rwork_si = si * cp
    rwork_si2 = si * hvasui
    arr = arr.at[:, 8, 4].set(-rwork_si)
    arr = arr.at[:, 8, 6].set(-rwork_si2)
    arr = arr.at[:, 8, 8].set(rwork_si + rwork_si2 * dqi + coi + ziri)
    rhs = rhs.at[:, 8].set(-rwork_si2 * (qi - dqi * ti)
                            + coi * tsno_1 + soli + firi + ziri * tiold)

    # --- Solve the 9x9 system using F77-style no-pivot Gauss (finding #17) ---
    vec = _f77_linsolve(arr, rhs)

    tu_new, ts_new, tl_new = vec[:, 0], vec[:, 1], vec[:, 2]
    t12_new, t34_new = vec[:, 3], vec[:, 4]
    q12_new, q34_new = vec[:, 5], vec[:, 6]
    tg_new, ti_new = vec[:, 7], vec[:, 8]

    # --- Flux diagnostics ---
    # Sensible heat
    fsena = cp * cu * (ta * tfac - t12_new)
    fsenu = cp * su * (wu * tu_new + (1.0 - wu) * tupre - t12_new)
    fsens = cp * ss * (ws * ts_new + (1.0 - ws) * tspre - t12_new)
    fsenl = cp * sl * (wl * tl_new + (1.0 - wl) * tlpre - t12_new)
    fseng = cp * sg * (wg * tg_new + (1.0 - wg) * tgpre - t34_new)
    fseni = cp * si * (wi * ti_new + (1.0 - wi) * tipre - t34_new)

    # Moisture
    fvapa = cu * (qa - q12_new)
    quav = qu + wu * dqu * (tu_new - tupre)
    fvapuw = suw * (quav - q12_new)
    fvaput = jnp.maximum(0.0, sut * (quav - q12_new))
    qsav = qs + ws * dqs * (ts_new - tspre)
    fvaps = ssw * (qsav - q12_new)
    qlav = ql + wl * dql * (tl_new - tlpre)
    fvaplw = slw * (qlav - q34_new)
    fvaplt = jnp.maximum(0.0, slt0 * (qlav - q34_new))
    qgav = qg + wg * dqg * (tg_new - tgpre)
    fvapg = sg * (qgfac * qgav - q34_new)
    qiav = qi + wi * dqi * (ti_new - tipre)
    fvapi = si * (qiav - q34_new)

    # IR adjustments
    firg_adj = firg - wg * zirg * (tg_new - tgold)
    firi_adj = firi - wi * ziri * (ti_new - tiold)
    firb_adj = (firb + (1.0 - fi) * wg * zirg * (tg_new - tgold)
                     + fi * wi * ziri * (ti_new - tiold))

    # NOTE: F77 turvap.f line 1249 clamps ti = min(ti, tmelt) ONLY at the
    # last iter (after `if iter.lt.niter return`). To match F77 exactly,
    # the iter loop in timestep_body applies the clamp AFTER all iters;
    # turvap leaves ti unclamped. (Removing the per-iter clamp dropped
    # cell 2 freezethaw temperature divergence from 0.07K to near-zero.)

    # Net evaporation
    ginvap = fvapuw * xu + fvaps * xs + fvaplw * xl
    gsuvap = fvapg * (1.0 - fi) + fvapi * fi
    gtransu = fvaput * 2.0 * lai[:, 1] * fu
    gtransl = fvaplt * 2.0 * lai[:, 0] * fl * (1.0 - fi)
    gtrans = gtransu + gtransl

    return CanopyResult(
        tu=tu_new, ts=ts_new, tl=tl_new,
        t12=t12_new, t34=t34_new, q12=q12_new, q34=q34_new,
        tg=tg_new, ti=ti_new,
        fsena=fsena, fsenu=fsenu, fsens=fsens,
        fsenl=fsenl, fseng=fseng, fseni=fseni,
        fvapa=fvapa, fvapuw=fvapuw, fvaput=fvaput,
        fvaps=fvaps, fvaplw=fvaplw, fvaplt=fvaplt,
        fvapg=fvapg, fvapi=fvapi,
        firb=firb_adj, firg=firg_adj, firi=firi_adj,
        ginvap=ginvap, gsuvap=gsuvap, gtrans=gtrans,
    )
