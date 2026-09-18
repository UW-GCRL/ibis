"""Photosynthesis and stomatal conductance for IBIS-JAX.

Ported from physiology.f90.  Implements Farquhar C3/C4 photosynthesis
with Ball-Berry stomatal conductance, plus soil moisture stress.

All functions are pure, JIT-compatible, and differentiable.

Functions:
  colimit        -- co-limitation quadratic solver
  farquhar_c3    -- C3 leaf-level photosynthesis (Farquhar et al. 1980)
  farquhar_c4    -- C4 leaf-level photosynthesis
  ball_berry_step -- one implicit relaxation step for cs, gs, ci
  canopy_scaling -- PAR extinction-based canopy scaling
  drystress      -- soil moisture stress factor
  stomata        -- full stomata calculation for all 5 PFT groups
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array

from ibis.physics.thermo import esat, qsat


# ---------------------------------------------------------------------------
# StomataParams: all PFT physiological parameters
# ---------------------------------------------------------------------------

class StomataParams(eqx.Module):
    """PFT-specific physiological parameters.

    Default values from params.can.  Ball-Berry parameter arrays are ordered:
    [broadleaf, conifer, shrub, C4grass, C3grass] — matching params.can order.

    Uses jnp.ndarray for PFT-indexed fields so they are JAX-traceable and
    differentiable for gradient-based calibration.
    """
    # Universal photosynthesis constants
    tau15: float = 4500.0          # CO2/O2 specificity at 15C
    kc15: float = 1.5e-4           # CO2 Michaelis-Menten at 15C (mol/mol)
    ko15: float = 0.25             # O2 Michaelis-Menten at 15C (mol/mol)
    cimax: float = 2000.0e-6       # Max internal CO2 (mol/mol)
    o2conc: float = 0.209          # Atmospheric O2 (mol/mol)

    # C3 parameters
    alpha3: float = 0.080          # Quantum efficiency
    theta3: float = 0.950          # Light/Rubisco co-limitation curvature
    beta3: float = 0.990           # TPU co-limitation curvature

    # C4 parameters
    alpha4: float = 0.050          # Quantum efficiency
    theta4: float = 0.970          # Light/Rubisco co-limitation curvature
    beta4: float = 0.800           # CO2 co-limitation curvature

    # Ball-Berry: [broadleaf, conifer, shrub, C4grass, C3grass]
    # F77 defaults from params.can (matches wrap_init_extras / readpars.f
    # comments in fortran/ibis_2.6b4/physiology.f lines 95-122). Required for
    # bit-exact F77 lsxmain ↔ JAX timestep_body verification.
    # Prior JAX-only CMA-ES-optimized values (kept for reference, not used):
    #   coefm=[10.9, 9.0, 5.8, 3.0, 6.0], coefb=[0.022, 0.011, 0.010, 0.020, 0.010]
    # Explicit float64 dtype: default_factory runs at field-init time which
    # may precede jax.config.update('jax_enable_x64', True); without dtype
    # these arrays are float32 and JAX weak-type promotion preserves that
    # through gsc_ub → totcondub → suh → arr[tu, q12] which then introduces
    # ~1e-6 matrix error and mK state divergence on freeze-thaw cells.
    gamma: jnp.ndarray = eqx.field(
        default_factory=lambda: jnp.array([0.015, 0.015, 0.015, 0.030, 0.015],
                                          dtype=jnp.float64))
    coefm: jnp.ndarray = eqx.field(
        default_factory=lambda: jnp.array([10.0, 6.0, 9.0, 4.0, 9.0],
                                          dtype=jnp.float64))
    coefb: jnp.ndarray = eqx.field(
        default_factory=lambda: jnp.array([0.010, 0.010, 0.010, 0.040, 0.010],
                                          dtype=jnp.float64))
    gsmin: jnp.ndarray = eqx.field(
        default_factory=lambda: jnp.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
                                          dtype=jnp.float64))

    # Vmax per PFT (12 PFTs, 0-indexed)
    vmax_pft: jnp.ndarray = eqx.field(
        default_factory=lambda: jnp.array([
            65e-6, 65e-6, 40e-6, 30e-6, 30e-6, 25e-6,
            30e-6, 30e-6, 27.5e-6, 27.5e-6, 15e-6, 25e-6],
            dtype=jnp.float64))

    # Soil moisture stress factor
    stressfac: float = -5.0

    # Canopy vmax by composition. F77 (False): each big leaf takes ONE PFT's
    # vmax by existence priority, so the deciduous-tree and deciduous-shrub
    # entries of vmax_pft are never read (CALIBRATION.md 17). True: the
    # displayed-LAI-weighted mean over the PFTs in that big leaf, priority
    # pick where no leaves exist. Static switch; set `stomata: vmax_mix: true`.
    vmax_mix: bool = False


# ---------------------------------------------------------------------------
# StomataResult: all outputs from the stomata calculation
# ---------------------------------------------------------------------------

class StomataResult(NamedTuple):
    """Outputs from the stomata calculation."""
    # Updated per-PFT states
    ci_upper: Float[Array, "npoi 2"]
    gs_upper: Float[Array, "npoi 2"]
    cs_upper: Float[Array, "npoi 2"]
    an_upper: Float[Array, "npoi 2"]       # Canopy-scaled, dry-fraction adjusted
    ag_upper: Float[Array, "npoi 2"]
    ci_lower: Float[Array, "npoi 3"]
    gs_lower: Float[Array, "npoi 3"]
    cs_lower: Float[Array, "npoi 3"]
    an_lower: Float[Array, "npoi 3"]
    ag_lower: Float[Array, "npoi 3"]
    # Total conductance for water vapour (used by turvap)
    totcondub: Float[Array, "npoi"]
    totconduc: Float[Array, "npoi"]
    totcondls: Float[Array, "npoi"]
    totcondl3: Float[Array, "npoi"]
    totcondl4: Float[Array, "npoi"]
    # Updated 10-day running means
    a10scalparamu: Float[Array, "npoi"]
    a10daylightu: Float[Array, "npoi"]
    a10scalparaml: Float[Array, "npoi"]
    a10daylightl: Float[Array, "npoi"]


# ---------------------------------------------------------------------------
# Helper: co-limitation solver
# ---------------------------------------------------------------------------

def colimit(j1, j2, theta):
    """Solve the co-limitation quadratic: theta*x² - (j1+j2)*x + j1*j2 = 0.

    Returns the smaller root (the co-limited rate).

    Parameters
    ----------
    j1, j2 : array
        Two rate-limiting terms.
    theta : float
        Curvature parameter (0 < theta <= 1).

    Returns
    -------
    array
        Co-limited rate (smaller root).
    """
    a = theta
    b = j1 + j2
    c = j1 * j2
    disc = jnp.maximum(b ** 2 - 4.0 * a * c, 0.0)
    # q = 0.5*(b + sqrt(disc)) + tiny offset to avoid division by zero
    q = 0.5 * (b + jnp.sqrt(disc)) + 1.0e-15
    return jnp.minimum(q / a, c / q)


# ---------------------------------------------------------------------------
# C3 photosynthesis (Farquhar et al. 1980)
# ---------------------------------------------------------------------------

def farquhar_c3(par, ci, vmax, gamstar, kc, ko, o2conc, rdark, params):
    """C3 leaf-level photosynthesis.

    Parameters
    ----------
    par : array [npoi]
        Top-of-canopy PAR (W/m2).
    ci : array [npoi]
        Intercellular CO2 (mol/mol).
    vmax : array [npoi]
        Temperature- and stress-adjusted Vmax (mol/m2/s).
    gamstar : array [npoi]
        CO2 compensation point (mol/mol).
    kc, ko : array [npoi]
        Michaelis-Menten constants for CO2 and O2.
    o2conc : float
        Atmospheric O2 concentration.
    rdark : array [npoi]
        Dark respiration (mol/m2/s).
    params : StomataParams

    Returns
    -------
    ag, an : arrays [npoi]
        Gross and net photosynthesis (mol/m2/s).
    """
    # Light-limited rate (RuBP regeneration)
    je = par * 4.59e-06 * params.alpha3 * (ci - gamstar) / (ci + 2.0 * gamstar)
    # Rubisco-limited rate
    jc = vmax * (ci - gamstar) / (ci + kc * (1.0 + o2conc / ko))
    # Co-limit light and Rubisco
    jp = colimit(je, jc, params.theta3)
    # TPU-limited rate (sink limitation)
    js = vmax / 2.2
    # Co-limit with sink
    ag = colimit(jp, js, params.beta3)
    an = ag - rdark
    return ag, an


# ---------------------------------------------------------------------------
# C4 photosynthesis
# ---------------------------------------------------------------------------

def farquhar_c4(par, ci, vmax, rdark, params):
    """C4 leaf-level photosynthesis.

    Parameters
    ----------
    par : array [npoi]
        Top-of-canopy PAR (W/m2).
    ci : array [npoi]
        Intercellular CO2 (mol/mol).
    vmax : array [npoi]
        Temperature- and stress-adjusted Vmax (mol/m2/s).
    rdark : array [npoi]
        Dark respiration (mol/m2/s).
    params : StomataParams

    Returns
    -------
    ag, an : arrays [npoi]
        Gross and net photosynthesis (mol/m2/s).
    """
    # Light-limited
    je = par * 4.59e-06 * params.alpha4
    # Rubisco-limited
    jc = vmax
    # Co-limit light and Rubisco
    jp = colimit(je, jc, params.theta4)
    # CO2-limited
    ji = 18.0e3 * vmax * ci
    # Co-limit with CO2
    ag = colimit(jp, ji, params.beta4)
    an = ag - rdark
    return ag, an


# ---------------------------------------------------------------------------
# Ball-Berry stomatal conductance: one implicit relaxation step
# ---------------------------------------------------------------------------

def ball_berry_step(an, cs_old, gs_old, ci_old, co2conc, gbco2, rh,
                    coefm, coefb, stress, gsmin, gamstar_min, cimax,
                    is_c4=False):
    """One implicit relaxation step for surface CO2, stomatal conductance, ci.

    Uses the Fortran 0.5*(old + new) averaging for stability.

    Parameters
    ----------
    an : array [npoi]
        Net photosynthesis at current ci (mol/m2/s).
    cs_old, gs_old, ci_old : arrays [npoi]
        Previous-step surface CO2, stomatal conductance, intercellular CO2.
    co2conc : float
        Atmospheric CO2 (mol/mol).
    gbco2 : array [npoi]
        Boundary layer conductance for CO2 (mol/m2/s).
    rh : array [npoi]
        Relative humidity at canopy air space (0-1).
    coefm, coefb : float
        Ball-Berry slope and intercept.
    stress : array [npoi]
        Soil moisture stress factor (0-1).
    gsmin : float
        Minimum stomatal conductance (mol/m2/s).
    gamstar_min : array [npoi]
        1.05 * gamstar, minimum ci/cs threshold.
    cimax : float
        Maximum ci for stability.
    is_c4 : bool
        If True, use C4-specific cs/ci lower bounds.

    Returns
    -------
    cs, gs, ci : arrays [npoi]
        Updated surface CO2, stomatal conductance, intercellular CO2.
    """
    # Surface CO2: relaxation
    cs = 0.5 * (cs_old + co2conc - an / gbco2)
    cs_min = jnp.where(is_c4, 1.0e-8, gamstar_min)
    cs = jnp.maximum(cs_min, cs)

    # Stomatal conductance: Ball-Berry with relaxation
    gs = 0.5 * (gs_old + coefm * an * rh / cs + coefb * stress)
    gs = jnp.maximum(gs, jnp.maximum(gsmin, coefb * stress))

    # Intercellular CO2: relaxation
    ci = 0.5 * (ci_old + cs - 1.6 * an / gs)
    ci_min = jnp.where(is_c4, 0.0, gamstar_min)
    ci = jnp.clip(ci, ci_min, cimax)

    return cs, gs, ci


# ---------------------------------------------------------------------------
# Canopy PAR scaling
# ---------------------------------------------------------------------------

def canopy_scaling(toppar, term6, term7, scalcoef, lai_canopy, sai_canopy,
                   a10scalp, a10daylight, dtime, epsilon):
    """Compute canopy-average scaling factor from PAR extinction.

    Parameters
    ----------
    toppar : array [npoi]
        Top-of-canopy PAR (W/m2).
    term6, term7 : arrays [npoi]
        Radiation transmission terms from two-stream RT.
    scalcoef : array [npoi, 4]
        PAR scaling coefficients from radiation module.
    lai_canopy, sai_canopy : arrays [npoi]
        Canopy LAI and SAI (one canopy level).
    a10scalp, a10daylight : arrays [npoi]
        10-day running mean accumulators.
    dtime : float
        Timestep length (s).
    epsilon : float
        Small number to avoid division by zero.

    Returns
    -------
    scale, a10scalp_new, a10daylight_new : arrays [npoi]
        Canopy scaling factor and updated running means.
    """
    # Effective extinction parameter
    extpar = (term6 * scalcoef[:, 0] +
              term7 * scalcoef[:, 1] -
              term7 * scalcoef[:, 2]) / jnp.maximum(scalcoef[:, 3], epsilon)
    extpar = jnp.clip(extpar, 0.1, 10.0)

    # Plant area indices
    pxai = extpar * (lai_canopy + sai_canopy)
    plai = extpar * lai_canopy

    # 10-day running mean decay weight
    zweight = jnp.exp(-1.0 / (10.0 * 86400.0 / dtime))

    # Daytime: update running means; nighttime: use stored mean
    has_lai = plai > 0.0
    is_daytime = toppar > 10.0
    scale_day = (1.0 - jnp.exp(-pxai)) / jnp.maximum(plai, epsilon)

    # Update running means (only during daytime with LAI)
    update = has_lai & is_daytime
    a10scalp_new = jnp.where(
        update,
        zweight * a10scalp + (1.0 - zweight) * scale_day * toppar,
        a10scalp)
    a10daylight_new = jnp.where(
        update,
        zweight * a10daylight + (1.0 - zweight) * toppar,
        a10daylight)

    # Scale factor: daytime uses current, nighttime uses 10-day mean
    scale_night = a10scalp / jnp.maximum(a10daylight, epsilon)
    scale = jnp.where(has_lai,
                      jnp.where(is_daytime, scale_day, scale_night),
                      0.0)

    return scale, a10scalp_new, a10daylight_new


# ---------------------------------------------------------------------------
# Soil moisture stress
# ---------------------------------------------------------------------------

def drystress(wsoi, wisoi, swilt, sfield, froot, stressfac=-5.0):
    """Compute soil moisture stress factors for upper and lower canopy.

    Parameters
    ----------
    wsoi : array [npoi, nsoilay]
        Liquid water fraction of pore space.
    wisoi : array [npoi, nsoilay]
        Ice fraction of pore space.
    swilt : array [npoi, nsoilay]
        Wilting point (fraction of pore space).
    sfield : array [npoi, nsoilay]
        Field capacity (fraction of pore space).
    froot : array [nsoilay, 2]
        Root fraction per soil layer, columns = [lower, upper].
    stressfac : float
        Sigmoidal stress shape parameter (default -5.0).

    Returns
    -------
    stresstl, stresstu : arrays [npoi]
        Total moisture stress for lower and upper canopy (0-1).
    stressl, stressu : arrays [npoi, nsoilay]
        Per-layer moisture stress, lower and upper canopy.
    """
    znorm = 1.0 - jnp.exp(stressfac)

    # Plant-available water content (fraction of field-capacity range)
    awc = jnp.clip(
        (wsoi * (1.0 - wisoi) - swilt) / jnp.maximum(sfield - swilt, 1e-10),
        0.0, 1.0)

    # Sigmoidal response to dryness
    zwilt = (1.0 - jnp.exp(stressfac * awc)) / znorm
    zwilt = jnp.clip(zwilt, 0.0, 1.0)

    # Per-layer stress weighted by root fraction
    # froot: [nsoilay, 2], zwilt: [npoi, nsoilay]
    stressl = zwilt * froot[:, 0][jnp.newaxis, :]   # [npoi, nsoilay]
    stressu = zwilt * froot[:, 1][jnp.newaxis, :]

    # Total stress = sum over layers
    stresstl = jnp.sum(stressl, axis=1)              # [npoi]
    stresstu = jnp.sum(stressu, axis=1)

    return stresstl, stresstu, stressl, stressu


# ---------------------------------------------------------------------------
# Vmax selection based on PFT existence
# ---------------------------------------------------------------------------

BROADLEAF_PFTS = (0, 1, 2, 4, 6)     # upper-canopy broadleaf big leaf
CONIFER_PFTS = (3, 5, 7)             # upper-canopy conifer big leaf
SHRUB_PFTS = (8, 9)                  # lower-canopy shrub big leaf


def _mix_vmax(frac, vp, idx, fallback, eps=1e-6):
    """Displayed-LAI-weighted mean of vp over the PFTs in `idx`; `fallback`
    where those PFTs have no leaves (frac sums to ~0)."""
    ix = jnp.array(idx)
    w = frac[:, ix]                                   # [npoi, len(idx)]
    tot = jnp.sum(w, axis=1)
    mixed = jnp.sum(w * vp[ix][jnp.newaxis, :], axis=1) / jnp.maximum(tot, eps)
    return jnp.where(tot > eps, mixed, fallback)


def _select_vmax_upper(exist, params, frac=None):
    """Nominal Vmax of the two upper-canopy big leaves.

    F77 (params.vmax_mix False, or no `frac`): ONE PFT's vmax each, by
    existence priority -- broadleaf: tropical (PFT 0) > warm-temperate (2)
    > temperate deciduous (4); conifer: temperate (3) > boreal (5). The
    deciduous entries 1, 6, 7 are then never read (CALIBRATION.md 17).
    Composition (vmax_mix True): the displayed-LAI-weighted mean over the
    PFTs in each big leaf, with the priority pick where no leaves exist.

    Parameters
    ----------
    exist : array [npoi, npft]
        PFT existence (>0.5 = present).
    params : StomataParams
    frac : array [npoi, npft] or None
        Displayed-LAI share of each PFT within its canopy (pheno).

    Returns
    -------
    vmaxub, vmaxuc : arrays [npoi]
        Nominal Vmax for the broadleaf and conifer big leaves.
    """
    vp = params.vmax_pft
    vmaxub = jnp.where(exist[:, 0] > 0.5, vp[0],
             jnp.where(exist[:, 2] > 0.5, vp[2], vp[4]))
    vmaxuc = jnp.where(exist[:, 3] > 0.5, vp[3], vp[5])
    if params.vmax_mix and frac is not None:
        vmaxub = _mix_vmax(frac, vp, BROADLEAF_PFTS, vmaxub)
        vmaxuc = _mix_vmax(frac, vp, CONIFER_PFTS, vmaxuc)
    return vmaxub, vmaxuc


def _select_vmax_shrub(params, frac=None):
    """Nominal Vmax of the lower-canopy shrub big leaf: F77 reads the
    evergreen shrub's (PFT 8) for both shrub PFTs; composition mixes 8 and 9."""
    vp = params.vmax_pft
    base = jnp.broadcast_to(vp[8], frac.shape[:1]) if frac is not None else vp[8]
    if params.vmax_mix and frac is not None:
        return _mix_vmax(frac, vp, SHRUB_PFTS, base)
    return vp[8]


# ---------------------------------------------------------------------------
# Temperature-dependent kinetic parameters
# ---------------------------------------------------------------------------

def _temp_dep_params(tleaf_k, params):
    """Compute temperature-dependent photosynthesis parameters.

    Parameters
    ----------
    tleaf_k : array [npoi]
        Leaf temperature (K).
    params : StomataParams

    Returns
    -------
    kc, ko, tau, gamstar, tempvm_c3, tempvm_c4 : arrays [npoi]
    """
    rwork = 3.47e-03 - 1.0 / tleaf_k
    tau = params.tau15 * jnp.exp(-4500.0 * rwork)
    kc = params.kc15 * jnp.exp(6000.0 * rwork)
    ko = params.ko15 * jnp.exp(1500.0 * rwork)

    gamstar = params.o2conc / (2.0 * tau)

    tleaf_c = tleaf_k - 273.16
    base = jnp.exp(3500.0 * rwork)
    cold_c3 = (1.0 + jnp.exp(0.40 * (5.0 - tleaf_c)))
    cold_c4 = (1.0 + jnp.exp(0.40 * (10.0 - tleaf_c)))
    hot = (1.0 + jnp.exp(0.40 * (tleaf_c - 50.0)))
    tempvm_c3 = base / (cold_c3 * hot)
    tempvm_c4 = base / (cold_c4 * hot)

    return kc, ko, tau, gamstar, tempvm_c3, tempvm_c4


# ---------------------------------------------------------------------------
# Total conductance for water vapour
# ---------------------------------------------------------------------------

def _total_conductance(su, gsc):
    """Compute total conductance for water vapour (series resistance).

    Match F77 physiology.f:515-517 EXACT FP operation order:

        rwork = 1. / su
        dump  = 1. / 0.029           (precomputed scalar)
        totcond = 1. / (rwork + dump / gsc)

    Mathematically: (1/0.029) / gsc = 1 / (0.029 * gsc) -- but the two
    differ at the ULP level (0.029 is not exact in binary). F77's order
    yields totcond ≈1e-12 different from the algebraic rewrite, which
    propagates through suh = suw*hvapf to ~1e-6 in the turvap matrix
    and amplifies through the 3-iter solve to mK on freeze-thaw cells.
    See finding #20.

    Parameters
    ----------
    su : array [npoi]
        Aerodynamic transfer coefficient (heat/momentum).
    gsc : array [npoi]
        Canopy-average stomatal conductance (mol/m2/s).

    Returns
    -------
    totcond : array [npoi]
        Total conductance for water vapour.
    """
    dump = 1.0 / 0.029
    return 1.0 / (1.0 / su + dump / gsc)


# ---------------------------------------------------------------------------
# Main stomata function
# ---------------------------------------------------------------------------

def stomata(tu, tl, t12, t34, q12, q34, su, sl, psurf,
            topparu, topparl,
            termu6, termu7, terml6, terml7,
            scalcoefu, scalcoefl,
            ci_upper, gs_upper, cs_upper,
            ci_lower, gs_lower, cs_lower,
            fwetu, fwetl,
            lai, sai, exist,
            stresstu, stresstl,
            a10scalparamu, a10daylightu,
            a10scalparaml, a10daylightl,
            co2conc, dtime, params, epsilon=1.0e-7, frac=None):
    """Full stomata calculation for all 5 PFT groups.

    Computes Farquhar photosynthesis + Ball-Berry stomatal conductance
    for upper canopy (broadleaf, conifer) and lower canopy (shrub, C3 grass,
    C4 grass), with canopy scaling and dry-fraction adjustment.

    Parameters
    ----------
    tu, tl : arrays [npoi]
        Upper and lower canopy leaf temperatures (K).
    t12, t34 : arrays [npoi]
        Air temperatures at z12 and z34 levels (K).
    q12, q34 : arrays [npoi]
        Specific humidities at z12 and z34 levels (kg/kg).
    su, sl : arrays [npoi]
        Aerodynamic transfer coefficients, upper and lower canopy.
    psurf : array [npoi]
        Surface pressure (Pa).
    topparu, topparl : arrays [npoi]
        Top-of-canopy PAR for upper and lower canopy (W/m2).
    termu6, termu7, terml6, terml7 : arrays [npoi]
        Radiation transmission terms from two-stream RT.
    scalcoefu, scalcoefl : arrays [npoi, 4]
        PAR scaling coefficients from radiation module.
    ci_upper : array [npoi, 2]
        Intercellular CO2, upper [broadleaf, conifer] (mol/mol).
    gs_upper : array [npoi, 2]
        Stomatal conductance, upper (mol/m2/s).
    cs_upper : array [npoi, 2]
        Surface CO2, upper (mol/mol).
    ci_lower : array [npoi, 3]
        Intercellular CO2, lower [shrub, C3grass, C4grass].
    gs_lower : array [npoi, 3]
        Stomatal conductance, lower.
    cs_lower : array [npoi, 3]
        Surface CO2, lower.
    fwetu, fwetl : arrays [npoi]
        Wetted fractions, upper and lower canopy.
    lai, sai : arrays [npoi, 2]
        LAI and SAI, [lower, upper] canopy levels.
    exist : array [npoi, npft]
        PFT existence (>0.5 = present).
    stresstu, stresstl : arrays [npoi]
        Soil moisture stress, upper and lower canopy (0-1).
    a10scalparamu, a10daylightu : arrays [npoi]
        10-day running mean PAR scaling accumulators, upper canopy.
    a10scalparaml, a10daylightl : arrays [npoi]
        10-day running mean PAR scaling accumulators, lower canopy.
    co2conc : float
        Atmospheric CO2 (mol/mol).
    dtime : float
        Timestep length (s).
    params : StomataParams
    epsilon : float
        Small number to avoid division by zero.

    Returns
    -------
    StomataResult
    """
    # ===================================================================
    # Upper canopy physiology
    # ===================================================================

    # Temperature-dependent parameters at upper leaf temperature
    kc_u, ko_u, _, gamstar_u, tempvm_c3_u, _ = _temp_dep_params(tu, params)
    gamstar_min_u = 1.05 * gamstar_u

    # Constrain ci
    ci_ub = jnp.clip(ci_upper[:, 0], gamstar_min_u, params.cimax)
    ci_uc = jnp.clip(ci_upper[:, 1], gamstar_min_u, params.cimax)

    # Boundary layer conductance for CO2
    gbco2u = jnp.clip(su * 25.5, 0.1, 10.0)

    # Relative humidity in upper canopy air space
    esat12 = esat(t12)
    qsat12 = qsat(esat12, psurf)
    rh12 = jnp.maximum(0.30, q12 / jnp.maximum(qsat12, epsilon))

    # Select Vmax per PFT
    vmaxub_nom, vmaxuc_nom = _select_vmax_upper(exist, params, frac)

    # --- Broadleaf (ub) ---
    vmax_ub = vmaxub_nom * tempvm_c3_u * stresstu
    rdark_ub = params.gamma[0] * vmaxub_nom * tempvm_c3_u
    ag_ub, an_ub = farquhar_c3(
        topparu, ci_ub, vmax_ub, gamstar_u, kc_u, ko_u,
        params.o2conc, rdark_ub, params)
    cs_ub, gs_ub, ci_ub = ball_berry_step(
        an_ub, cs_upper[:, 0], gs_upper[:, 0], ci_ub,
        co2conc, gbco2u, rh12,
        params.coefm[0], params.coefb[0], stresstu,
        params.gsmin[0], gamstar_min_u, params.cimax)

    # --- Conifer (uc) ---
    vmax_uc = vmaxuc_nom * tempvm_c3_u * stresstu
    rdark_uc = params.gamma[1] * vmaxuc_nom * tempvm_c3_u
    ag_uc, an_uc = farquhar_c3(
        topparu, ci_uc, vmax_uc, gamstar_u, kc_u, ko_u,
        params.o2conc, rdark_uc, params)
    cs_uc, gs_uc, ci_uc = ball_berry_step(
        an_uc, cs_upper[:, 1], gs_upper[:, 1], ci_uc,
        co2conc, gbco2u, rh12,
        params.coefm[1], params.coefb[1], stresstu,
        params.gsmin[1], gamstar_min_u, params.cimax)

    # --- Upper canopy scaling ---
    scale_u, a10sp_u, a10dl_u = canopy_scaling(
        topparu, termu6, termu7, scalcoefu,
        lai[:, 1], sai[:, 1],
        a10scalparamu, a10daylightu, dtime, epsilon)

    # Scale to canopy level
    agc_ub = ag_ub * scale_u
    agc_uc = ag_uc * scale_u
    anc_ub = an_ub * scale_u
    anc_uc = an_uc * scale_u

    # Canopy-average surface CO2 and stomatal conductance
    csc_ub = jnp.maximum(gamstar_min_u, co2conc - anc_ub / gbco2u)
    csc_uc = jnp.maximum(gamstar_min_u, co2conc - anc_uc / gbco2u)
    gsc_ub = jnp.maximum(
        jnp.maximum(params.gsmin[0], params.coefb[0] * stresstu),
        params.coefm[0] * anc_ub * rh12 / csc_ub + params.coefb[0] * stresstu)
    gsc_uc = jnp.maximum(
        jnp.maximum(params.gsmin[1], params.coefb[1] * stresstu),
        params.coefm[1] * anc_uc * rh12 / csc_uc + params.coefb[1] * stresstu)

    # Total conductance for water vapour
    totcondub = _total_conductance(su, gsc_ub)
    totconduc = _total_conductance(su, gsc_uc)

    # Adjust for wet fraction (wet leaves don't photosynthesize)
    dry_u = 1.0 - fwetu
    agc_ub = agc_ub * dry_u
    agc_uc = agc_uc * dry_u
    anc_ub = anc_ub * dry_u
    anc_uc = anc_uc * dry_u

    # ===================================================================
    # Lower canopy physiology
    # ===================================================================

    # Temperature-dependent parameters at lower leaf temperature
    kc_l, ko_l, _, gamstar_l, tempvm_c3_l, tempvm_c4_l = _temp_dep_params(
        tl, params)
    gamstar_min_l = 1.05 * gamstar_l

    # Constrain ci
    ci_ls = jnp.clip(ci_lower[:, 0], gamstar_min_l, params.cimax)
    ci_l3 = jnp.clip(ci_lower[:, 1], gamstar_min_l, params.cimax)
    ci_l4 = jnp.clip(ci_lower[:, 2], 0.0, params.cimax)

    # Boundary layer conductance for CO2
    gbco2l = jnp.clip(sl * 25.5, 0.1, 10.0)

    # Relative humidity in lower canopy air space
    esat34 = esat(t34)
    qsat34 = qsat(esat34, psurf)
    rh34 = jnp.maximum(0.30, q34 / jnp.maximum(qsat34, epsilon))

    # --- Shrub (ls) - C3 ---
    vmaxls_nom = _select_vmax_shrub(params, frac)
    vmax_ls = vmaxls_nom * tempvm_c3_l * stresstl
    rdark_ls = params.gamma[2] * vmaxls_nom * tempvm_c3_l
    ag_ls, an_ls = farquhar_c3(
        topparl, ci_ls, vmax_ls, gamstar_l, kc_l, ko_l,
        params.o2conc, rdark_ls, params)
    cs_ls, gs_ls, ci_ls = ball_berry_step(
        an_ls, cs_lower[:, 0], gs_lower[:, 0], ci_ls,
        co2conc, gbco2l, rh34,
        params.coefm[2], params.coefb[2], stresstl,
        params.gsmin[2], gamstar_min_l, params.cimax)

    # --- C3 grass (l3) ---
    vmaxl3_nom = params.vmax_pft[11]
    vmax_l3 = vmaxl3_nom * tempvm_c3_l * stresstl
    rdark_l3 = params.gamma[4] * vmaxl3_nom * tempvm_c3_l
    ag_l3, an_l3 = farquhar_c3(
        topparl, ci_l3, vmax_l3, gamstar_l, kc_l, ko_l,
        params.o2conc, rdark_l3, params)
    cs_l3, gs_l3, ci_l3 = ball_berry_step(
        an_l3, cs_lower[:, 1], gs_lower[:, 1], ci_l3,
        co2conc, gbco2l, rh34,
        params.coefm[4], params.coefb[4], stresstl,
        params.gsmin[4], gamstar_min_l, params.cimax)

    # --- C4 grass (l4) ---
    vmaxl4_nom = params.vmax_pft[10]
    vmax_l4 = vmaxl4_nom * tempvm_c4_l * stresstl
    rdark_l4 = params.gamma[3] * vmaxl4_nom * tempvm_c4_l
    ag_l4, an_l4 = farquhar_c4(
        topparl, ci_l4, vmax_l4, rdark_l4, params)
    cs_l4, gs_l4, ci_l4 = ball_berry_step(
        an_l4, cs_lower[:, 2], gs_lower[:, 2], ci_l4,
        co2conc, gbco2l, rh34,
        params.coefm[3], params.coefb[3], stresstl,
        params.gsmin[3], gamstar_min_l, params.cimax,
        is_c4=True)

    # --- Lower canopy scaling ---
    scale_l, a10sp_l, a10dl_l = canopy_scaling(
        topparl, terml6, terml7, scalcoefl,
        lai[:, 0], sai[:, 0],
        a10scalparaml, a10daylightl, dtime, epsilon)

    # Scale to canopy level
    agc_ls = ag_ls * scale_l
    agc_l3 = ag_l3 * scale_l
    agc_l4 = ag_l4 * scale_l
    anc_ls = an_ls * scale_l
    anc_l3 = an_l3 * scale_l
    anc_l4 = an_l4 * scale_l

    # Canopy-average surface CO2 and stomatal conductance
    csc_ls = jnp.maximum(gamstar_min_l, co2conc - anc_ls / gbco2l)
    csc_l3 = jnp.maximum(gamstar_min_l, co2conc - anc_l3 / gbco2l)
    csc_l4 = jnp.maximum(1.0e-8, co2conc - anc_l4 / gbco2l)

    gsc_ls = jnp.maximum(
        jnp.maximum(params.gsmin[2], params.coefb[2] * stresstl),
        params.coefm[2] * anc_ls * rh34 / csc_ls + params.coefb[2] * stresstl)
    gsc_l3 = jnp.maximum(
        jnp.maximum(params.gsmin[4], params.coefb[4] * stresstl),
        params.coefm[4] * anc_l3 * rh34 / csc_l3 + params.coefb[4] * stresstl)
    gsc_l4 = jnp.maximum(
        jnp.maximum(params.gsmin[3], params.coefb[3] * stresstl),
        params.coefm[3] * anc_l4 * rh34 / csc_l4 + params.coefb[3] * stresstl)

    # Total conductance for water vapour
    totcondls = _total_conductance(sl, gsc_ls)
    totcondl3 = _total_conductance(sl, gsc_l3)
    totcondl4 = _total_conductance(sl, gsc_l4)

    # Adjust for wet fraction
    dry_l = 1.0 - fwetl
    agc_ls = agc_ls * dry_l
    agc_l3 = agc_l3 * dry_l
    agc_l4 = agc_l4 * dry_l
    anc_ls = anc_ls * dry_l
    anc_l3 = anc_l3 * dry_l
    anc_l4 = anc_l4 * dry_l

    # ===================================================================
    # Pack results
    # ===================================================================

    ci_upper_out = jnp.stack([ci_ub, ci_uc], axis=-1)
    gs_upper_out = jnp.stack([gs_ub, gs_uc], axis=-1)
    cs_upper_out = jnp.stack([cs_ub, cs_uc], axis=-1)
    an_upper_out = jnp.stack([anc_ub, anc_uc], axis=-1)
    ag_upper_out = jnp.stack([agc_ub, agc_uc], axis=-1)

    ci_lower_out = jnp.stack([ci_ls, ci_l3, ci_l4], axis=-1)
    gs_lower_out = jnp.stack([gs_ls, gs_l3, gs_l4], axis=-1)
    cs_lower_out = jnp.stack([cs_ls, cs_l3, cs_l4], axis=-1)
    an_lower_out = jnp.stack([anc_ls, anc_l3, anc_l4], axis=-1)
    ag_lower_out = jnp.stack([agc_ls, agc_l3, agc_l4], axis=-1)

    return StomataResult(
        ci_upper=ci_upper_out,
        gs_upper=gs_upper_out,
        cs_upper=cs_upper_out,
        an_upper=an_upper_out,
        ag_upper=ag_upper_out,
        ci_lower=ci_lower_out,
        gs_lower=gs_lower_out,
        cs_lower=cs_lower_out,
        an_lower=an_lower_out,
        ag_lower=ag_lower_out,
        totcondub=totcondub,
        totconduc=totconduc,
        totcondls=totcondls,
        totcondl3=totcondl3,
        totcondl4=totcondl4,
        a10scalparamu=a10sp_u,
        a10daylightu=a10dl_u,
        a10scalparaml=a10sp_l,
        a10daylightl=a10dl_l,
    )
