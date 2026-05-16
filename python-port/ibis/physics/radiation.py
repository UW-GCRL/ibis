"""Two-stream solar and infrared radiation for IBIS-JAX.

Ported from radiation.f90.  All functions are pure, JIT-compatible,
and differentiable.  Dynamic indexing (indsol/nsol) is replaced by
boolean sunlit masks and jnp.where.

Functions:
  twoset       -- set two-stream optical parameters
  twostr       -- solve two-stream RT for one vegetation layer
  solsur       -- soil and snow surface albedos
  solalb       -- effective canopy albedos (calls twostr twice)
  solarf       -- absorbed solar fluxes per component
  irrad        -- absorbed infrared fluxes per component
  compute_radiation -- top-level entry point

Reference: Sellers et al. (1996); Lowe (1977); Pollard & Thompson (1995).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


# ---------------------------------------------------------------------------
# Vegetation optical parameters (read from params.can in full model)
# ---------------------------------------------------------------------------

class VegRadParams(eqx.Module):
    """Vegetation optical properties for radiation calculations.

    Attributes:
        rhoveg: Leaf/stem reflectance [nband, 2] (band, story).
        tauveg: Leaf/stem transmittance [nband, 2].
        oriev:  Vertical leaf orientation fraction [2] (story).
        orieh:  Horizontal leaf orientation fraction [2] (story).
    """
    rhoveg: Float[Array, "nband 2"]
    tauveg: Float[Array, "nband 2"]
    oriev: Float[Array, "2"]
    orieh: Float[Array, "2"]


def default_veg_rad_params() -> VegRadParams:
    """Default optical params from params.can (Sellers et al. 1996)."""
    return VegRadParams(
        rhoveg=jnp.array([[0.10, 0.10],    # VIS: lower, upper
                          [0.60, 0.40]]),   # NIR: lower, upper
        tauveg=jnp.array([[0.07, 0.05],     # VIS
                          [0.25, 0.20]]),    # NIR
        oriev=jnp.array([0.5, 0.0]),        # lower: max(-chiflz,0)=0.5; upper: max(-chifuz,0)=0
        orieh=jnp.array([0.0, 0.0]),        # lower: max(chiflz,0)=0;  upper: max(chifuz,0)=0
    )


# ---------------------------------------------------------------------------
# Snow optical constants
# ---------------------------------------------------------------------------
_OMEGASNO = jnp.array([0.9, 0.7])   # snow single-scatter albedo [VIS, NIR]
_BETADSNO = 0.5                       # snow upscatter fraction, direct
_BETAISNO = 0.5                       # snow upscatter fraction, diffuse

# Emissivities
_EMISOIL = 0.95
_EMISNOW = 0.95


# ---------------------------------------------------------------------------
# Two-stream parameter setup
# ---------------------------------------------------------------------------

def twoset(
    coszen: Float[Array, "npoi"],
    lai_iv: Float[Array, "npoi"],
    sai_iv: Float[Array, "npoi"],
    fwet_leaf: Float[Array, "npoi"],
    rliq_leaf: Float[Array, "npoi"],
    temp_leaf: Float[Array, "npoi"],
    rhoveg_ib: float,
    tauveg_ib: float,
    oriev_iv: float,
    orieh_iv: float,
    omegasno_ib: float,
    tmelt: float,
    iv: int,
    sunlit: Float[Array, "npoi"],
    fwet_stem: Float[Array, "npoi"] | None = None,
    rliq_stem: Float[Array, "npoi"] | None = None,
    temp_stem: Float[Array, "npoi"] | None = None,
):
    """Set two-stream optical parameters for one band and one story.

    Matches Fortran twoset().  Returns (omega, betad, betai, avmu, gdir).

    For the upper story (iv=2), pass separate leaf (fwet_leaf/rliq_leaf/temp_leaf)
    and stem (fwet_stem/rliq_stem/temp_stem) values.  The snow adjustment uses
    proper weighted products matching the Fortran formula:
        y = x*(fwetu*(1-rliqu)) + (1-x)*(fwets*(1-rliqs))
        o = (x*clamp(tmelt-tu) + (1-x)*clamp(tmelt-ts)) * omegasno

    For the lower story (iv=1), stem values are unused.
    """
    orand = 1.0 - oriev_iv - orieh_iv
    omega0 = rhoveg_ib + tauveg_ib

    # Direct upscatter fraction from lookup-table equivalent.
    # The Fortran tablemu is exactly: 0.5 - coszen/3.
    ztab = 0.5 - coszen / 3.0
    rwork = 1.0 / jnp.maximum(omega0, 1e-12)

    betad = (oriev_iv * 0.5 * (rhoveg_ib + tauveg_ib)
             + orieh_iv * rhoveg_ib
             + orand * ((1.0 - ztab) * rhoveg_ib + ztab * tauveg_ib)) * rwork

    betai = (oriev_iv * 0.5 * (rhoveg_ib + tauveg_ib)
             + orieh_iv * rhoveg_ib
             + orand * ((2.0 / 3.0) * rhoveg_ib
                        + (1.0 / 3.0) * tauveg_ib)) * rwork

    gdir = (oriev_iv * (2.0 / 3.1415927)
            * jnp.sqrt(jnp.maximum(0.0, 1.0 - coszen ** 2))
            + orieh_iv * coszen
            + orand * 0.5)

    avmu = jnp.ones_like(coszen)

    # Adjust for intercepted snow on canopy
    if iv == 2 and fwet_stem is not None:
        # Upper story: weighted sum of products (not product of averages)
        # Fortran: y = x*(fwetu*(1-rliqu)) + (1-x)*(fwets*(1-rliqs))
        x_up = lai_iv / jnp.maximum(lai_iv + sai_iv, 1e-7)
        y = x_up * (fwet_leaf * (1.0 - rliq_leaf)) + (1.0 - x_up) * (fwet_stem * (1.0 - rliq_stem))
        # Fortran: o = (x*clamp(tmelt-tu) + (1-x)*clamp(tmelt-ts)) * omegasno
        snow_leaf = jnp.clip((tmelt - temp_leaf) / 0.1, 0.6, 1.0)
        snow_stem = jnp.clip((tmelt - temp_stem) / 0.1, 0.6, 1.0)
        o = (x_up * snow_leaf + (1.0 - x_up) * snow_stem) * omegasno_ib
    else:
        # Lower story: single fwet/rliq/temp
        y = fwet_leaf * (1.0 - rliq_leaf)
        snow_factor = 0.6 + 0.4 * jnp.clip((tmelt - temp_leaf) / 1.0, 0.0, 1.0)
        o = omegasno_ib * snow_factor

    rwork_sno = y * o
    omega = (1.0 - y) * omega0 + rwork_sno
    omega = jnp.maximum(omega, 1e-12)
    betad = ((1.0 - y) * omega0 * betad + rwork_sno * _BETADSNO) / omega
    betai = ((1.0 - y) * omega0 * betai + rwork_sno * _BETAISNO) / omega

    # Zero out where not sunlit
    omega = jnp.where(sunlit, omega, 0.0)
    betad = jnp.where(sunlit, betad, 0.0)
    betai = jnp.where(sunlit, betai, 0.0)
    gdir = jnp.where(sunlit, gdir, 0.0)
    avmu = jnp.where(sunlit, avmu, 1.0)

    return omega, betad, betai, avmu, gdir


# ---------------------------------------------------------------------------
# Two-stream solver
# ---------------------------------------------------------------------------

class TwostrResult(NamedTuple):
    """Output of the two-stream solver."""
    abvegd: Float[Array, "npoi"]   # absorbed flux, direct
    abvegi: Float[Array, "npoi"]   # absorbed flux, diffuse
    refld: Float[Array, "npoi"]    # reflected, direct
    refli: Float[Array, "npoi"]    # reflected, diffuse
    fbeldd: Float[Array, "npoi"]   # below-layer direct (per unit direct)
    fbeldi: Float[Array, "npoi"]   # always 0
    fbelid: Float[Array, "npoi"]   # below-layer diffuse (per unit direct)
    fbelii: Float[Array, "npoi"]   # below-layer diffuse (per unit diffuse)
    term: Float[Array, "npoi 7"]   # canopy scaling terms (VIS only)


def twostr(
    coszen: Float[Array, "npoi"],
    lai_iv: Float[Array, "npoi"],
    sai_iv: Float[Array, "npoi"],
    asurd_ib: Float[Array, "npoi"],
    asuri_ib: Float[Array, "npoi"],
    omega: Float[Array, "npoi"],
    betad: Float[Array, "npoi"],
    betai: Float[Array, "npoi"],
    avmu: Float[Array, "npoi"],
    gdir: Float[Array, "npoi"],
    epsilon: float,
    sunlit: Float[Array, "npoi"],
    ib: int,
) -> TwostrResult:
    """Solve two-stream RT for one vegetation layer.

    Matches Fortran twostr() exactly.  Returns TwostrResult.
    """
    npoi = coszen.shape[0]

    b = 1.0 - omega * (1.0 - betai)
    c = omega * betai
    tmp0 = b * b - c * c

    q = jnp.sqrt(jnp.maximum(0.0, tmp0))
    k = gdir / jnp.maximum(coszen, 0.01)
    p = avmu * k

    # Perturb p away from q to avoid singular denominator
    # Fortran sign(0.001, p-q) returns +0.001 when p==q; jnp.sign(0)=0 would fail
    sign_pq = jnp.where(p >= q, 1.0, -1.0)
    p = jnp.where(jnp.abs(p - q) < 0.001 * p,
                  (1.0 + sign_pq * 0.001) * p, p)

    c0 = omega * p
    d = c0 * betad
    f = c0 * (1.0 - betad)
    h = q / avmu
    sigma = p * p - tmp0

    # Avoid division by zero in surface albedo terms
    asurd_safe = jnp.maximum(asurd_ib, 1e-12)
    asuri_safe = jnp.maximum(asuri_ib, 1e-12)

    ud1 = b - c / asurd_safe
    ui1 = b - c / asuri_safe
    ud2 = b - c * asurd_ib
    ui2 = b - c * asuri_ib
    ud3 = f + c * asurd_ib

    xai = jnp.maximum(lai_iv + sai_iv, epsilon)
    s1 = jnp.exp(-h * xai)
    s2 = jnp.exp(-k * xai)
    p1 = b + q
    p2 = b - q
    p3 = b + p
    p4 = b - p
    rwork = 1.0 / jnp.maximum(s1, 1e-30)

    dd1 = p1 * (ud1 - q) * rwork - p2 * (ud1 + q) * s1
    di1 = p1 * (ui1 - q) * rwork - p2 * (ui1 + q) * s1
    dd2 = (ud2 + q) * rwork - (ud2 - q) * s1
    di2 = (ui2 + q) * rwork - (ui2 - q) * s1

    # Prevent zero denominators
    dd1 = jnp.where(jnp.abs(dd1) < 1e-30, 1e-30, dd1)
    di1 = jnp.where(jnp.abs(di1) < 1e-30, 1e-30, di1)
    dd2 = jnp.where(jnp.abs(dd2) < 1e-30, 1e-30, dd2)
    di2 = jnp.where(jnp.abs(di2) < 1e-30, 1e-30, di2)
    sigma = jnp.where(jnp.abs(sigma) < 1e-30, 1e-30, sigma)

    h1 = -d * p4 - c * f
    rwork2 = s2 * (d - c - h1 * (ud1 + p) / sigma)
    h2 = (1.0 / dd1) * ((d - h1 * p3 / sigma) * (ud1 - q) / jnp.maximum(s1, 1e-30) - p2 * rwork2)
    h3 = -(1.0 / dd1) * ((d - h1 * p3 / sigma) * (ud1 + q) * s1 - p1 * rwork2)
    h4 = -f * p3 - c * d
    rwork3 = s2 * (ud3 - h4 * (ud2 - p) / sigma)
    h5 = -(1.0 / dd2) * (h4 * (ud2 + q) / (sigma * jnp.maximum(s1, 1e-30)) + rwork3)
    h6 = (1.0 / dd2) * (h4 * s1 * (ud2 - q) / sigma + rwork3)
    h7 = c * (ui1 - q) / (di1 * jnp.maximum(s1, 1e-30))
    h8 = -c * s1 * (ui1 + q) / di1
    h9 = (ui2 + q) / (di2 * jnp.maximum(s1, 1e-30))
    h10 = -s1 * (ui2 - q) / di2

    fbeldd = s2
    fbeldi = jnp.zeros(npoi)
    fbelid = h4 / sigma * s2 + h5 * s1 + h6 / jnp.maximum(s1, 1e-30)
    fbelii = h9 * s1 + h10 / jnp.maximum(s1, 1e-30)

    refld = h1 / sigma + h2 + h3
    refli = h7 + h8

    absurd = (1.0 - asurd_ib) * fbeldd + (1.0 - asuri_ib) * fbelid
    absuri = (1.0 - asuri_ib) * fbelii
    abvegd = jnp.maximum(0.0, 1.0 - refld - absurd)
    abvegi = jnp.maximum(0.0, 1.0 - refli - absuri)

    # Zero out where no vegetation
    no_veg = xai < epsilon
    abvegd = jnp.where(no_veg, 0.0, abvegd)
    abvegi = jnp.where(no_veg, 0.0, abvegi)

    # Canopy scaling terms (VIS band only, otherwise zeros)
    term = jnp.zeros((npoi, 7))
    if ib == 0:
        term = term.at[:, 0].set(k * (1.0 + (h4 - h1) / sigma))
        term = term.at[:, 1].set(h * (h5 - h2))
        term = term.at[:, 2].set(h * (h9 - h7))
        term = term.at[:, 3].set(h * (h3 - h6))
        term = term.at[:, 4].set(h * (h8 - h10))
        term = term.at[:, 5].set(k)
        term = term.at[:, 6].set(h)

    # Zero out non-sunlit points
    abvegd = jnp.where(sunlit, abvegd, 0.0)
    abvegi = jnp.where(sunlit, abvegi, 0.0)
    refld = jnp.where(sunlit, refld, 0.0)
    refli = jnp.where(sunlit, refli, 0.0)
    fbeldd = jnp.where(sunlit, fbeldd, 0.0)
    fbelid = jnp.where(sunlit, fbelid, 0.0)
    fbelii = jnp.where(sunlit, fbelii, 0.0)

    return TwostrResult(
        abvegd=abvegd, abvegi=abvegi,
        refld=refld, refli=refli,
        fbeldd=fbeldd, fbeldi=fbeldi,
        fbelid=fbelid, fbelii=fbelii,
        term=term,
    )


# ---------------------------------------------------------------------------
# Surface albedos
# ---------------------------------------------------------------------------

def solsur(
    coszen: Float[Array, "npoi"],
    wsoi_1: Float[Array, "npoi"],
    wisoi_1: Float[Array, "npoi"],
    tsno_1: Float[Array, "npoi"],
    albsav: Float[Array, "npoi"],
    albsan: Float[Array, "npoi"],
    tmelt: float,
    sunlit: Float[Array, "npoi"],
    ib: int,
):
    """Compute soil and snow surface albedos for one waveband.

    Parameters
    ----------
    ib : int
        Waveband index (0=VIS, 1=NIR).  Note: 0-indexed unlike Fortran.

    Returns
    -------
    albsod, albsoi, albsnd, albsni : arrays
        Soil (direct, diffuse) and snow (direct, diffuse) albedos.
    """
    # Soil albedo depends on surface moisture
    zw = wsoi_1 * (1.0 - wisoi_1)
    dinc = 1.0 + jnp.clip(1.0 - zw / 0.50, 0.0, 1.0)
    alb_base = jnp.where(ib == 0, albsav, albsan)
    albsod = jnp.minimum(alb_base * dinc, 0.80)
    albsoi = albsod

    # Snow albedo depends on snow temperature
    # VIS: hi=0.70, lo=0.90;  NIR: hi=0.40, lo=0.60
    a_hi = jnp.where(ib == 0, 0.70, 0.40)
    a_lo = jnp.where(ib == 0, 0.90, 0.60)
    t_shi = tmelt
    t_slo = tmelt - 15.0

    x = (a_hi * (tsno_1 - t_slo) + a_lo * (t_shi - tsno_1)) / (t_shi - t_slo)
    x = jnp.clip(x, a_hi, a_lo)
    zfac = jnp.maximum(0.0, 1.5 / (1.0 + 4.0 * coszen) - 0.5)
    albsnd = jnp.minimum(0.99, x + (1.0 - x) * zfac)
    albsni = jnp.minimum(1.0, x)

    # Zero out non-sunlit
    albsod = jnp.where(sunlit, albsod, 0.0)
    albsoi = jnp.where(sunlit, albsoi, 0.0)
    albsnd = jnp.where(sunlit, albsnd, 0.0)
    albsni = jnp.where(sunlit, albsni, 0.0)

    return albsod, albsoi, albsnd, albsni


# ---------------------------------------------------------------------------
# Canopy albedos (solalb)
# ---------------------------------------------------------------------------

def solalb(
    coszen, lai, sai, fu, fl, fi,
    albsod, albsoi, albsnd, albsni,
    fwetu, fwets, fwetl, rliqu, rliqs, rliql,
    tu, ts, tl,
    veg_params, tmelt, epsilon, ib, sunlit,
):
    """Compute effective canopy albedos for one waveband.

    Calls twostr twice: once for lower canopy, once for upper.
    Returns (asurd_ib, asuri_ib, twostr_lower, twostr_upper).
    """
    npoi = coszen.shape[0]
    rhoveg_ib_lo = veg_params.rhoveg[ib, 0]
    tauveg_ib_lo = veg_params.tauveg[ib, 0]
    rhoveg_ib_up = veg_params.rhoveg[ib, 1]
    tauveg_ib_up = veg_params.tauveg[ib, 1]
    oriev_lo = veg_params.oriev[0]
    orieh_lo = veg_params.orieh[0]
    oriev_up = veg_params.oriev[1]
    orieh_up = veg_params.orieh[1]
    omegasno_ib = _OMEGASNO[ib]

    # --- (a) Lower canopy two-stream ---
    # Surface below lower canopy = bare soil
    asurd_lo = albsod
    asuri_lo = albsoi

    omega_lo, betad_lo, betai_lo, avmu_lo, gdir_lo = twoset(
        coszen, lai[:, 0], sai[:, 0],
        fwetl, rliql, tl,
        rhoveg_ib_lo, tauveg_ib_lo, oriev_lo, orieh_lo,
        omegasno_ib, tmelt, 1, sunlit,
    )
    ts_lo = twostr(
        coszen, lai[:, 0], sai[:, 0], asurd_lo, asuri_lo,
        omega_lo, betad_lo, betai_lo, avmu_lo, gdir_lo,
        epsilon, sunlit, ib,
    )

    # --- (b) Areal average surface albedo (lower veg + soil + snow) ---
    asurd_ib = (fl * (1.0 - fi) * ts_lo.refld
                + (1.0 - fl) * (1.0 - fi) * albsod
                + fi * albsnd)
    asuri_ib = (fl * (1.0 - fi) * ts_lo.refli
                + (1.0 - fl) * (1.0 - fi) * albsoi
                + fi * albsni)

    # --- (c) Upper canopy two-stream ---
    # Pass separate leaf/stem values for correct weighted products
    omega_up, betad_up, betai_up, avmu_up, gdir_up = twoset(
        coszen, lai[:, 1], sai[:, 1],
        fwetu, rliqu, tu,
        rhoveg_ib_up, tauveg_ib_up, oriev_up, orieh_up,
        omegasno_ib, tmelt, 2, sunlit,
        fwet_stem=fwets, rliq_stem=rliqs, temp_stem=ts,
    )
    ts_up = twostr(
        coszen, lai[:, 1], sai[:, 1], asurd_ib, asuri_ib,
        omega_up, betad_up, betai_up, avmu_up, gdir_up,
        epsilon, sunlit, ib,
    )

    # --- (d) Overall albedos ---
    asurd_ib = fu * ts_up.refld + (1.0 - fu) * asurd_ib
    asuri_ib = fu * ts_up.refli + (1.0 - fu) * asuri_ib

    # Zero non-sunlit
    asurd_ib = jnp.where(sunlit, asurd_ib, 0.0)
    asuri_ib = jnp.where(sunlit, asuri_ib, 0.0)

    return asurd_ib, asuri_ib, ts_lo, ts_up


# ---------------------------------------------------------------------------
# Absorbed solar fluxes (solarf)
# ---------------------------------------------------------------------------

def solarf(
    solad_ib, solai_ib,
    lai, sai, fu, fl, fi, epsilon,
    ts_lo, ts_up,
    albsod, albsoi, albsnd, albsni,
    sunlit, ib,
):
    """Compute absorbed solar fluxes for one waveband.

    Parameters
    ----------
    ts_lo, ts_up : TwostrResult
        Two-stream results for lower and upper canopy.

    Returns
    -------
    solu, sols, soll, solg, soli : absorbed fluxes (W/m2)
    topparu, topparl : top-of-canopy PAR (W/m2), VIS band only
    scalcoefu, scalcoefl : PAR scaling coefficients [npoi, 4], VIS band only
    """
    npoi = solad_ib.shape[0]

    # (f) Absorbed by upper leaves/stems; fluxes below upper canopy
    x = solad_ib * ts_up.abvegd + solai_ib * ts_up.abvegi
    y = lai[:, 1] / jnp.maximum(lai[:, 1] + sai[:, 1], epsilon)
    solu = x * y
    sols = x * (1.0 - y)
    sol2d = solad_ib * ts_up.fbeldd
    sol2i = solad_ib * ts_up.fbelid + solai_ib * ts_up.fbelii

    # (g) Areally average fluxes to lower canopy
    sol3d = fu * sol2d + (1.0 - fu) * solad_ib
    sol3i = fu * sol2i + (1.0 - fu) * solai_ib

    # (h) Absorbed by lower canopy
    soll = sol3d * ts_lo.abvegd + sol3i * ts_lo.abvegi

    # (i) Absorbed by soil (snow-free)
    xd = (fl * ts_lo.fbeldd + 1.0 - fl) * sol3d
    xi = (fl * (sol3d * ts_lo.fbelid + sol3i * ts_lo.fbelii)
          + (1.0 - fl) * sol3i)
    solg = (1.0 - albsod) * xd + (1.0 - albsoi) * xi

    # (j) Absorbed by snow
    soli = ((1.0 - albsnd) * sol3d + (1.0 - albsni) * sol3i)

    # PAR scaling (VIS band only)
    topparu_ib = jnp.zeros(npoi)
    topparl_ib = jnp.zeros(npoi)
    scalcoefu_ib = jnp.zeros((npoi, 4))
    scalcoefl_ib = jnp.zeros((npoi, 4))

    if ib == 0:
        # Upper canopy
        xaiu = jnp.maximum(lai[:, 1] + sai[:, 1], epsilon)
        termu = ts_up.term
        sc0 = termu[:, 0] * solad_ib
        sc1 = termu[:, 1] * solad_ib + termu[:, 2] * solai_ib
        sc2 = termu[:, 3] * solad_ib + termu[:, 4] * solai_ib
        sc3 = sc0 + sc1 + sc2
        scalcoefu_ib = jnp.stack([sc0, sc1, sc2, sc3], axis=-1)
        topparu_ib = sc3 * lai[:, 1] / xaiu

        # Lower canopy
        xail = jnp.maximum(lai[:, 0] + sai[:, 0], epsilon)
        terml = ts_lo.term
        sc0l = terml[:, 0] * sol3d
        sc1l = terml[:, 1] * sol3d + terml[:, 2] * sol3i
        sc2l = terml[:, 3] * sol3d + terml[:, 4] * sol3i
        sc3l = sc0l + sc1l + sc2l
        scalcoefl_ib = jnp.stack([sc0l, sc1l, sc2l, sc3l], axis=-1)
        topparl_ib = sc3l * lai[:, 0] / xail

    # Zero non-sunlit
    solu = jnp.where(sunlit, solu, 0.0)
    sols = jnp.where(sunlit, sols, 0.0)
    soll = jnp.where(sunlit, soll, 0.0)
    solg = jnp.where(sunlit, solg, 0.0)
    soli = jnp.where(sunlit, soli, 0.0)

    return solu, sols, soll, solg, soli, topparu_ib, topparl_ib, scalcoefu_ib, scalcoefl_ib


# ---------------------------------------------------------------------------
# Infrared radiation (irrad)
# ---------------------------------------------------------------------------

def irrad(
    fira: Float[Array, "npoi"],
    lai: Float[Array, "npoi 2"],
    sai: Float[Array, "npoi 2"],
    fu: Float[Array, "npoi"],
    fl: Float[Array, "npoi"],
    fi: Float[Array, "npoi"],
    tu: Float[Array, "npoi"],
    ts: Float[Array, "npoi"],
    tl: Float[Array, "npoi"],
    tg: Float[Array, "npoi"],
    ti: Float[Array, "npoi"],
    stef: float,
):
    """Compute absorbed IR fluxes for all components.

    Matches Fortran irrad().  Returns (firu, firs, firl, firg, firi, firb).
    """
    avmuir = 1.0

    emu = 1.0 - jnp.exp(-lai[:, 1] / avmuir)
    ems = 1.0 - jnp.exp(-sai[:, 1] / avmuir)
    eml = 1.0 - jnp.exp(-(lai[:, 0] + sai[:, 0]) / avmuir)

    emg = _EMISOIL
    emi = _EMISNOW

    tu4 = stef * tu ** 4
    ts4 = stef * ts ** 4
    tl4 = stef * tl ** 4
    tg4 = stef * tg ** 4
    ti4 = stef * ti ** 4

    # Downward flux below upper canopy
    fdown = ((1.0 - fu) * fira
             + fu * ((1.0 - emu) * (1.0 - ems) * fira
                     + emu * (1.0 - ems) * tu4
                     + ems * ts4))

    # Downward flux below lower canopy
    fdowng = (1.0 - eml) * fdown + eml * tl4

    # Upward from soil/snow
    fupg = (1.0 - emg) * fdowng + emg * tg4
    fupgb = (1.0 - emg) * fdown + emg * tg4
    fupi = (1.0 - emi) * fdown + emi * ti4

    # Upward flux from surface system
    fup = ((1.0 - fi)
           * (fl * (eml * tl4 + (1.0 - eml) * fupg)
              + (1.0 - fl) * fupgb)
           + fi * fupi)

    # Total upward IR (for atmospheric boundary)
    firb = ((1.0 - fu) * fup
            + fu * ((1.0 - emu) * (1.0 - ems) * fup
                    + emu * tu4
                    + ems * (1.0 - emu) * ts4))

    # Net absorbed by each component
    firu = (emu * ems * ts4
            + emu * (1.0 - ems) * fup
            + emu * fira
            - 2.0 * emu * tu4)

    firs = (ems * emu * tu4
            + ems * fup
            + ems * (1.0 - emu) * fira
            - 2.0 * ems * ts4)

    firl = eml * fdown + eml * fupg - 2.0 * eml * tl4

    firg = fl * (fdowng - fupg) + (1.0 - fl) * (fdown - fupgb)

    firi = fdown - fupi

    return firu, firs, firl, firg, firi, firb


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

class RadiationResult(NamedTuple):
    """All radiation outputs for one timestep."""
    solu: Float[Array, "npoi"]
    sols: Float[Array, "npoi"]
    soll: Float[Array, "npoi"]
    solg: Float[Array, "npoi"]
    soli: Float[Array, "npoi"]
    firu: Float[Array, "npoi"]
    firs: Float[Array, "npoi"]
    firl: Float[Array, "npoi"]
    firg: Float[Array, "npoi"]
    firi: Float[Array, "npoi"]
    firb: Float[Array, "npoi"]
    asurd: Float[Array, "npoi nband"]
    asuri: Float[Array, "npoi nband"]
    topparu: Float[Array, "npoi"]
    topparl: Float[Array, "npoi"]
    scalcoefu: Float[Array, "npoi 4"]
    scalcoefl: Float[Array, "npoi 4"]
    # Two-stream k and h terms for stomata canopy scaling (VIS band only)
    termu6: Float[Array, "npoi"]      # Upper canopy extinction k
    termu7: Float[Array, "npoi"]      # Upper canopy extinction h
    terml6: Float[Array, "npoi"]      # Lower canopy extinction k
    terml7: Float[Array, "npoi"]      # Lower canopy extinction h


def compute_radiation(
    coszen, solad, solai, fira,
    lai, sai, fu, fl,
    tu, ts, tl, tg, ti,
    fi, tsno_1,
    wsoi_1, wisoi_1,
    albsav, albsan,
    fwetu, fwets, fwetl,
    rliqu, rliqs, rliql,
    veg_params: VegRadParams,
    stef: float,
    tmelt: float,
    epsilon: float,
) -> RadiationResult:
    """Compute all radiation fluxes for one timestep.

    This is the top-level entry point combining solsur, solalb, solarf,
    and irrad.  Loops over 2 wavebands (VIS=0, NIR=1).
    """
    npoi = coszen.shape[0]
    nband = 2
    sunlit = coszen > 0.0

    # Accumulators
    solu_tot = jnp.zeros(npoi)
    sols_tot = jnp.zeros(npoi)
    soll_tot = jnp.zeros(npoi)
    solg_tot = jnp.zeros(npoi)
    soli_tot = jnp.zeros(npoi)
    asurd_out = jnp.zeros((npoi, nband))
    asuri_out = jnp.zeros((npoi, nband))
    topparu_out = jnp.zeros(npoi)
    topparl_out = jnp.zeros(npoi)
    scalcoefu_out = jnp.zeros((npoi, 4))
    scalcoefl_out = jnp.zeros((npoi, 4))
    termu6_out = jnp.zeros(npoi)
    termu7_out = jnp.zeros(npoi)
    terml6_out = jnp.zeros(npoi)
    terml7_out = jnp.zeros(npoi)

    for ib in range(nband):
        # Surface albedos
        albsod, albsoi, albsnd, albsni = solsur(
            coszen, wsoi_1, wisoi_1, tsno_1,
            albsav, albsan, tmelt, sunlit, ib,
        )

        # Canopy albedos
        asurd_ib, asuri_ib, ts_lo, ts_up = solalb(
            coszen, lai, sai, fu, fl, fi,
            albsod, albsoi, albsnd, albsni,
            fwetu, fwets, fwetl, rliqu, rliqs, rliql,
            tu, ts, tl,
            veg_params, tmelt, epsilon, ib, sunlit,
        )

        # Absorbed solar fluxes
        (solu_ib, sols_ib, soll_ib, solg_ib, soli_ib,
         topparu_ib, topparl_ib, scalcoefu_ib, scalcoefl_ib) = solarf(
            solad[:, ib], solai[:, ib],
            lai, sai, fu, fl, fi, epsilon,
            ts_lo, ts_up,
            albsod, albsoi, albsnd, albsni,
            sunlit, ib,
        )

        # Accumulate
        solu_tot += solu_ib
        sols_tot += sols_ib
        soll_tot += soll_ib
        solg_tot += solg_ib
        soli_tot += soli_ib
        asurd_out = asurd_out.at[:, ib].set(asurd_ib)
        asuri_out = asuri_out.at[:, ib].set(asuri_ib)
        topparu_out += topparu_ib
        topparl_out += topparl_ib
        scalcoefu_out += scalcoefu_ib
        scalcoefl_out += scalcoefl_ib

        # Extract two-stream k, h terms (VIS band only, ib=0)
        if ib == 0:
            termu6_out = ts_up.term[:, 5]
            termu7_out = ts_up.term[:, 6]
            terml6_out = ts_lo.term[:, 5]
            terml7_out = ts_lo.term[:, 6]

    # Infrared radiation (band-independent)
    firu, firs, firl, firg, firi, firb = irrad(
        fira, lai, sai, fu, fl, fi,
        tu, ts, tl, tg, ti, stef,
    )

    return RadiationResult(
        solu=solu_tot, sols=sols_tot, soll=soll_tot,
        solg=solg_tot, soli=soli_tot,
        firu=firu, firs=firs, firl=firl, firg=firg, firi=firi,
        firb=firb,
        asurd=asurd_out, asuri=asuri_out,
        topparu=topparu_out, topparl=topparl_out,
        scalcoefu=scalcoefu_out, scalcoefl=scalcoefl_out,
        termu6=termu6_out, termu7=termu7_out,
        terml6=terml6_out, terml7=terml7_out,
    )
