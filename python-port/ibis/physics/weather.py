"""Weather generation for IBIS-JAX.

Ported from weather.f90.  Implements:
  - Richardson-type stochastic weather generator (daily)
  - Diurnal disaggregation to sub-daily timesteps
  - Monthly precipitation pre-generation (gamma distribution)

All functions are pure, JIT-compatible, and differentiable (where applicable).
Random number generation uses JAX PRNGKey instead of Fortran's ran2.  The
random sequences will differ from the Fortran, but statistical properties
(mean, variance, wet-day frequency) are preserved.

Functions:
  generate_month_precip  -- pre-generate daily precip for a month
  daily                  -- generate daily weather from monthly climatology
  diurnal                -- disaggregate daily to sub-daily timestep
  compute_precip_timing  -- precipitation event start/end times
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ibis.physics.thermo import esat, qsat

# ---------------------------------------------------------------------------
# Richardson autocorrelation matrices (from weather.f, column-major → row-major)
# ---------------------------------------------------------------------------

_A = jnp.array([[0.600, 0.010, 0.020],
                [0.500, 0.250, 0.125],
                [0.005, 0.005, 0.250]], dtype=jnp.float32)

_B = jnp.array([[ 0.500, 0.000, 0.000],
                [ 0.250, 0.500, 0.000],
                [-0.250, 0.250, 0.500]], dtype=jnp.float32)

# In the Fortran, r(j) = sum_k B(j,k) * ee(j), i.e. ee(j) not ee(k).
# This effectively reduces to r = B_rowsums * ee (element-wise).
_B_ROWSUMS = jnp.sum(_B, axis=1)  # [0.5, 0.75, 0.5]

# Days per month (non-leap)
_NDAYPM = jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Derived constants
_GRAV_OVER_RAIR_LAPSE = 9.80616 / 287.04 / 0.0065   # hypsometric exponent
_PI = jnp.pi
_STEF = 5.67e-8


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class ClimateData(NamedTuple):
    """Monthly climatological input data.

    All arrays indexed as [npoi, 12] where 12 = months (0-indexed).
    """
    xint: jnp.ndarray       # Monthly mean temperature (C) [npoi, 12]
    xinq: jnp.ndarray       # Monthly mean relative humidity (%) [npoi, 12]
    xinprec: jnp.ndarray    # Monthly mean precipitation rate (mm/day) [npoi, 12]
    xinwind: jnp.ndarray    # Monthly mean wind speed (m/s) [npoi, 12]
    xincld: jnp.ndarray     # Monthly mean cloud cover (%) [npoi, 12]
    xinwet: jnp.ndarray     # Mean wet days per month [npoi, 12]
    xintrng: jnp.ndarray    # Monthly diurnal temperature range (C) [npoi, 12]
    xintopo: jnp.ndarray    # Topographic elevation (m) [npoi]


class MonthlyPrecip(NamedTuple):
    """Pre-generated daily precipitation for a month."""
    iwetday: jnp.ndarray    # Wet/dry flags [npoi, 31]
    precipday: jnp.ndarray  # Daily amounts (mm) [npoi, 31]


class DailyWeather(NamedTuple):
    """Daily weather output."""
    td: jnp.ndarray         # Daily mean temperature (K) [npoi]
    tmax: jnp.ndarray       # Daily max temperature (K) [npoi]
    tmin: jnp.ndarray       # Daily min temperature (K) [npoi]
    cloud: jnp.ndarray      # Cloud fraction (0-1) [npoi]
    precip: jnp.ndarray     # Daily precipitation (mm) [npoi]
    psurf: jnp.ndarray      # Surface pressure (Pa) [npoi]
    qd: jnp.ndarray         # Daily mean specific humidity (kg/kg) [npoi]
    ud: jnp.ndarray         # Daily mean wind speed (m/s) [npoi]
    iwet: jnp.ndarray       # Wet-day flag (0 or 1) [npoi]
    xstore: jnp.ndarray     # Updated AR(1) state [npoi, 3]


class HourlyWeather(NamedTuple):
    """Sub-daily weather output."""
    coszen: jnp.ndarray     # Cosine of solar zenith angle [npoi]
    ta: jnp.ndarray         # Air temperature (K) [npoi]
    qa: jnp.ndarray         # Specific humidity (kg/kg) [npoi]
    ua: jnp.ndarray         # Wind speed (m/s) [npoi]
    rh: jnp.ndarray         # Relative humidity (%) [npoi]
    fira: jnp.ndarray       # Downward longwave radiation (W/m2) [npoi]
    raina: jnp.ndarray      # Rainfall rate (kg/m2/s) [npoi]
    snowa: jnp.ndarray      # Snowfall rate (kg/m2/s) [npoi]
    solad: jnp.ndarray      # Direct solar radiation [npoi, nband]
    solai: jnp.ndarray      # Diffuse solar radiation [npoi, nband]


# ---------------------------------------------------------------------------
# Monthly precipitation generation
# ---------------------------------------------------------------------------

def generate_month_precip(key, climate, imonth):
    """Generate daily precipitation for the entire month.

    Uses a Markov chain for wet/dry day sequence and gamma distribution
    for precipitation amounts, matching the Fortran weather generator's
    statistical properties.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    climate : ClimateData
        Monthly climatological input.
    imonth : int
        Month index (0-based).

    Returns
    -------
    key : PRNGKey
        Updated random key.
    monthly_precip : MonthlyPrecip
        Pre-generated daily precipitation.
    """
    ndaypm = _NDAYPM[imonth]
    npoi = climate.xinprec.shape[0]

    xinprec_m = jnp.maximum(0.0, climate.xinprec[:, imonth])
    xinwet_m = jnp.maximum(1.0, climate.xinwet[:, imonth])

    # Wet day transition probabilities
    pwet = xinwet_m / ndaypm
    pwd = 0.75 * pwet                       # P(wet | dry)
    pww = 0.25 + pwd                        # P(wet | wet)

    # Expected rain per wet day
    rainpwd = xinprec_m * ndaypm / jnp.maximum(0.1, xinwet_m)

    # Gamma distribution parameters (Ahrens-Dieter parameterisation)
    beta_val = jnp.maximum(1.0, -2.16 + 1.83 * rainpwd)
    alpha_val = jnp.maximum(0.01, rainpwd / beta_val)

    # Pre-generate random numbers
    key, k_wet, k_gamma = jax.random.split(key, 3)
    rn_wet = jax.random.uniform(k_wet, (31, npoi))      # [31, npoi]

    # Gamma deviates for all days, vectorised over grid points
    gamma_keys = jax.random.split(k_gamma, npoi)
    gamma_raw = jax.vmap(
        lambda k, a: jax.random.gamma(k, a, (31,))
    )(gamma_keys, alpha_val)                              # [npoi, 31]
    gamma_vals = gamma_raw * beta_val[:, None]

    # Markov chain for wet/dry (sequential over days, parallel over npoi)
    def scan_fn(iwet_prev, rn_day):
        # rn_day: [npoi]
        p_wet = jnp.where(iwet_prev == 0, pwd, pww)
        iwet_new = (rn_day <= p_wet).astype(jnp.int32)
        return iwet_new, iwet_new

    _, iwetday = jax.lax.scan(
        scan_fn, jnp.zeros(npoi, dtype=jnp.int32), rn_wet)
    iwetday = iwetday.T                                   # [npoi, 31]

    # Apply wet/dry mask and clip amounts
    precipday = jnp.where(
        iwetday == 1, jnp.clip(gamma_vals, 0.25, 150.0), 0.0)

    # Mask days beyond month length
    day_mask = jnp.arange(31) < ndaypm
    precipday = precipday * day_mask[None, :]
    iwetday = iwetday * day_mask[None, :].astype(jnp.int32)

    # Scale to match monthly total
    total = jnp.sum(precipday, axis=1, keepdims=True)
    target = (xinprec_m * ndaypm)[:, None]
    scale = target / jnp.maximum(total, 0.01)
    precipday = precipday * scale

    # Handle degenerate case: no wet days but positive monthly precip
    key, k_fix = jax.random.split(key)
    fix_day = jax.random.randint(k_fix, (npoi,), 0, jnp.maximum(ndaypm, 1))
    needs_fix = (total.squeeze() <= 0.0) & (xinprec_m > 0.0)
    fix_amount = xinprec_m * ndaypm
    precipday = jnp.where(
        needs_fix[:, None] & (jnp.arange(31)[None, :] == fix_day[:, None]),
        fix_amount[:, None], precipday)
    iwetday = jnp.where(
        needs_fix[:, None] & (jnp.arange(31)[None, :] == fix_day[:, None]),
        1, iwetday)

    return key, MonthlyPrecip(iwetday=iwetday, precipday=precipday)


# ---------------------------------------------------------------------------
# Daily weather generation
# ---------------------------------------------------------------------------

def _interp_weights(imonth, iday, ndaypm):
    """Compute monthly interpolation indices and weight.

    Returns
    -------
    it1w, it2w : int
        Source month indices (0-based).
    dt : float
        Interpolation weight.
    """
    day1 = iday + 1.0  # 1-based
    mid = (ndaypm + 1.0) * 0.5
    first_half = day1 < mid
    it1w = jnp.where(first_half, (imonth - 1) % 12, imonth).astype(jnp.int32)
    it2w = jnp.where(first_half, imonth, (imonth + 1) % 12).astype(jnp.int32)
    dt = jnp.where(first_half,
                   (day1 - 0.5) / ndaypm + 0.5,
                   (day1 - 0.5) / ndaypm - 0.5)
    return it1w, it2w, dt


def _interp_climate(arr, it1w, it2w, dt):
    """Linearly interpolate a [npoi, 12] climate array between two months."""
    return arr[:, it1w] + dt * (arr[:, it2w] - arr[:, it1w])


def daily(key, imonth, iday, climate, monthly_precip, xstore):
    """Generate daily weather from monthly climatology.

    Uses a Richardson-type multivariate AR(1) weather generator to
    perturb temperature, cloud cover, and humidity around their
    monthly-mean expected values.

    Parameters
    ----------
    key : PRNGKey
    imonth : int
        Month index (0-based).
    iday : int
        Day index within month (0-based).
    climate : ClimateData
    monthly_precip : MonthlyPrecip
    xstore : array [npoi, 3]
        AR(1) autocorrelation state.

    Returns
    -------
    key : PRNGKey
    daily_weather : DailyWeather
    """
    npoi = climate.xint.shape[0]
    ndaypm = _NDAYPM[imonth]

    # Read today's precipitation from pre-generated monthly data
    iwet = monthly_precip.iwetday[:, iday]
    precip = monthly_precip.precipday[:, iday]

    # Interpolation weights
    it1w, it2w, dt = _interp_weights(imonth, iday, ndaypm)

    # Wet day probability
    xinwet_m = jnp.maximum(1.0, climate.xinwet[:, imonth])
    pwet = xinwet_m / ndaypm

    # Omega parameters (EPIC weather generator style)
    omcloud = 0.90
    omqd = 0.50
    omtmax = 0.75

    # --- Temperature ---
    tdm = _interp_climate(climate.xint, it1w, it2w, dt) + 273.16
    trngm = _interp_climate(climate.xintrng, it1w, it2w, dt)

    tmaxm = tdm + 0.56 * trngm
    tminm = tdm - 0.44 * trngm

    # Wet/dry adjustment for max temperature
    tmaxd = tmaxm + pwet * omtmax * trngm
    tmaxw = tmaxd - omtmax * trngm
    # Use tmaxm if pwet==0 to avoid nan
    tmaxd = jnp.where(pwet > 0.0, tmaxd, tmaxm)
    tmaxw = jnp.where(pwet > 0.0, tmaxw, tmaxm)

    tmaxe = jnp.where(iwet == 0, tmaxd, tmaxw)
    tmine = tminm

    # Temperature standard deviations
    tmaxs = jnp.maximum(1.0, -0.0713 * (tdm - 273.16) + 4.89)
    tmins = jnp.maximum(1.0, -0.1280 * (tdm - 273.16) + 5.73)

    # --- Cloud cover ---
    xincld_it1 = jnp.clip(climate.xincld[:, it1w], 0.0, 100.0)
    xincld_it2 = jnp.clip(climate.xincld[:, it2w], 0.0, 100.0)
    cloudm = (xincld_it1 + dt * (xincld_it2 - xincld_it1)) / 100.0

    cloudd = jnp.clip(
        (cloudm - pwet * omcloud) / jnp.maximum(1.0 - pwet * omcloud, 1e-6),
        0.0, 1.0)
    cloudw = (cloudm - (1.0 - pwet) * cloudd) / jnp.maximum(pwet, 1e-6)
    cloudd = jnp.where(pwet > 0.0, cloudd, cloudm)
    cloudw = jnp.where(pwet > 0.0, cloudw, cloudm)

    cloude = jnp.where(iwet == 0, cloudd, cloudw)
    clouds = jnp.where(iwet == 0, 0.24 * cloude, 0.48 * cloude)

    # --- Richardson AR(1) perturbation ---
    # Generate 3 truncated normal variates per grid point
    key, k_normal = jax.random.split(key)
    ee = jax.random.truncated_normal(k_normal, -2.5, 2.5, (npoi, 3))

    # r = B_rowsums * ee (Fortran uses ee(j) not ee(k) — see weather.f)
    r = _B_ROWSUMS[None, :] * ee                      # [npoi, 3]
    # rr = A @ xstore
    rr = xstore @ _A.T                                # [npoi, 3]
    x = r + rr
    xstore_new = x

    # Perturbed daily temperatures
    tmax_out = tmaxe + tmaxs * x[:, 0]
    tmin_out = tmine + tmins * x[:, 1]
    # Ensure tmin <= tmax
    swap = tmin_out > tmax_out
    tmax_final = jnp.where(swap, tmin_out, tmax_out)
    tmin_final = jnp.where(swap, tmax_out, tmin_out)
    td = 0.44 * tmax_final + 0.56 * tmin_final

    # Perturbed cloud
    cloud = jnp.clip(cloude + clouds * x[:, 2], 0.0, 1.0)

    # --- Surface pressure (hypsometric formula) ---
    psurf = 101325.0 * (td / (td + 0.0065 * climate.xintopo)) \
            ** _GRAV_OVER_RAIR_LAPSE

    # --- Humidity ---
    xinq_it1 = jnp.clip(climate.xinq[:, it1w], 0.0, 100.0)
    xinq_it2 = jnp.clip(climate.xinq[:, it2w], 0.0, 100.0)
    qdm = (xinq_it1 + dt * (xinq_it2 - xinq_it1)) / 100.0

    # Wet/dry adjustment
    qdd_raw = (qdm - pwet * omqd) / jnp.maximum(1.0 - pwet * omqd, 1e-6)
    qdd = jnp.clip(qdd_raw, 0.2, 1.0)
    # Fortran only updates qdm when qdd was clamped below 0.2
    qdm = jnp.where(qdd_raw < 0.2, jnp.maximum(qdm, 0.2), qdm)
    qdw = (qdm - (1.0 - pwet) * qdd) / jnp.maximum(pwet, 1e-6)
    qdd = jnp.where(pwet > 0.0, qdd, qdm)
    qdw = jnp.where(pwet > 0.0, qdw, qdm)
    qde = jnp.where(iwet == 0, qdd, qdw)

    # Triangular distribution perturbation
    qdup = qde + (1.0 - qde) * jnp.exp(qde - 1.0)
    qdlow = qde * (1.0 - jnp.exp(-qde))

    key, k_qd = jax.random.split(key)
    rn_qd = jax.random.uniform(k_qd, (npoi,))
    y = 2.0 / jnp.maximum(qdup - qdlow, 1e-10)
    b3 = qde - qdlow
    b2 = qdup - qde
    b1 = rn_qd / y
    x1 = y * b3 / 2.0

    qd_raw = jnp.where(
        rn_qd > x1,
        qdup - jnp.sqrt(jnp.maximum(b2**2 - 2.0 * b2 * (b1 - 0.5 * b3), 0.0)),
        qdlow + jnp.sqrt(jnp.maximum(2.0 * b1 * b3, 0.0)))

    amn = (qdup + qde + qdlow) / 3.0
    qd_frac = jnp.clip(qd_raw * qde / jnp.maximum(amn, 1e-10), 0.30, 0.99)
    qd = qd_frac * qsat(esat(td), psurf)

    # --- Wind ---
    key, k_wind = jax.random.split(key)
    eud = _interp_climate(climate.xinwind, it1w, it2w, dt)
    rn_wind = jax.random.uniform(k_wind, (npoi,))
    ud = jnp.clip(
        1.13989 * eud * (-jnp.log(jnp.maximum(rn_wind, 1e-10))) ** 0.30,
        2.5, 10.0)

    return key, DailyWeather(
        td=td, tmax=tmax_final, tmin=tmin_final,
        cloud=cloud, precip=precip, psurf=psurf,
        qd=qd, ud=ud, iwet=iwet, xstore=xstore_new)


# ---------------------------------------------------------------------------
# Precipitation timing
# ---------------------------------------------------------------------------

def compute_precip_timing(key, precip, dtime):
    """Compute precipitation event timing for one day.

    Determines the start and end times and duration of the precipitation
    event, ensuring the rate doesn't exceed 15 mm per timestep.

    Parameters
    ----------
    key : PRNGKey
    precip : array [npoi]
        Daily precipitation (mm).
    dtime : float
        Timestep length (s).

    Returns
    -------
    key, plens, startp, endp : arrays [npoi]
        Event duration (s), start time (s), end time (s).
    """
    niter = (86400.0 / dtime).astype(jnp.int32) if hasattr(dtime, 'astype') \
            else int(86400.0 / dtime)
    plenmin = 1 + int((4.0 * 3600.0 - 1.0) / dtime)
    plenmax = max(int(24.0 * 3600.0 / dtime), plenmin)

    # Random event duration between plenmin and plenmax (Fortran main.f lines 407-408)
    key, k_plen = jax.random.split(key)
    plen_random = plenmin + (jax.random.uniform(k_plen, precip.shape)
                             * (plenmax - plenmin + 1)).astype(jnp.int32)
    plen_random = jnp.clip(plen_random, plenmin, plenmax)
    # Still extend if rate would exceed 15 mm/timestep
    plen_needed = jnp.ceil(precip / 15.0).astype(jnp.int32)
    plen = jnp.maximum(plen_random, plen_needed)
    plen = jnp.clip(plen, plenmin, plenmax)

    plens = dtime * plen.astype(jnp.float32)

    key, k_start = jax.random.split(key)
    max_start = jnp.maximum(0, niter - plen)
    start_idx = jnp.minimum(
        (jax.random.uniform(k_start, precip.shape) *
         (max_start + 1).astype(jnp.float32)).astype(jnp.int32),
        max_start)

    startp = dtime * start_idx.astype(jnp.float32)
    endp = startp + plens

    return key, plens, startp, endp


# ---------------------------------------------------------------------------
# Diurnal disaggregation
# ---------------------------------------------------------------------------

def effective_solar_constant(jday):
    """Solar constant times the orbital factor (Hartmann), W m-2; the expression `diurnal` has always used."""
    orbit = 2.0 * _PI * jday / 365.2425
    return 1370.0 * (1.000110
                     + 0.034221 * jnp.cos(orbit)
                     + 0.001280 * jnp.sin(orbit)
                     + 0.000719 * jnp.cos(2.0 * orbit)
                     + 0.000077 * jnp.sin(2.0 * orbit))


def observed_shortwave(sw_obs, use, coszen, jday, solad_gen, solai_gen, nband=2):
    """An observed hourly shortwave total in place of the generator's diurnal cycle (site runs; CALIBRATION.md 27).

    Where `use` is set, the observed total `sw_obs` (W m-2, instantaneous at the step's hour) is split into direct and
    diffuse with the hourly relation of Spitters et al. (1986) on the clearness index kt = sw_obs / (S0 cos z):
    fd = 1 (kt <= 0.22); 1 - 6.4 (kt - 0.22)^2 (kt <= 0.35); 1.47 - 1.66 kt (kt <= K); R above, with
    R = 0.847 - 1.61 sin(beta) + 1.04 sin(beta)^2 and K = (1.47 - R)/1.66; all diffuse when the sun is down. Then into
    the visible and near-infrared bands with the generator's fractions (0.46, 0.54). Elsewhere the generator's
    `solad_gen`, `solai_gen` are returned unchanged, so a mask that is off is bit-identical.
    """
    kt = jnp.clip(sw_obs / jnp.maximum(effective_solar_constant(jday) * coszen, 1.0), 0.0, 1.0)
    r = 0.847 - 1.61 * coszen + 1.04 * coszen ** 2
    kk = (1.47 - r) / 1.66
    fd = jnp.where(kt <= 0.22, 1.0,
                   jnp.where(kt <= 0.35, 1.0 - 6.4 * (kt - 0.22) ** 2,
                             jnp.where(kt <= kk, 1.47 - 1.66 * kt, r)))
    fd = jnp.where(coszen <= 0.0, 1.0, jnp.clip(fd, 0.0, 1.0))
    frac = jnp.array([0.46, 0.54])[:nband]
    sw = jnp.maximum(sw_obs, 0.0)
    solad = sw[:, None] * frac[None, :] * (1.0 - fd)[:, None]
    solai = sw[:, None] * frac[None, :] * fd[:, None]
    return (jnp.where(use[:, None], solad, solad_gen).astype(solad_gen.dtype),
            jnp.where(use[:, None], solai, solai_gen).astype(solai_gen.dtype))


def diurnal(time, jday, daily_weather, latitude, plens, startp, endp,
            key, dtime, nband=2):
    """Disaggregate daily weather to sub-daily timestep.

    Computes solar radiation, temperature, humidity, wind, IR, and
    precipitation at the current model timestep using Fourier-series
    diurnal cycles and standard solar geometry.

    Parameters
    ----------
    time : float
        Current time of day (s since midnight).
    jday : int
        Day of year (1-based).
    daily_weather : DailyWeather
        Daily weather statistics.
    latitude : array [npoi]
        Latitude of each grid point (degrees).
    plens : array [npoi]
        Precipitation event length (s).
    startp, endp : arrays [npoi]
        Precipitation start and end times (s).
    key : PRNGKey
    dtime : float
        Timestep length (s).
    nband : int
        Number of radiation wavebands (default 2: VIS, NIR).

    Returns
    -------
    key : PRNGKey
    hourly_weather : HourlyWeather
    """
    npoi = daily_weather.td.shape[0]
    rtime = time / 3600.0

    # Earth's orbital angle and solar declination (Hartmann)
    orbit = 2.0 * _PI * jday / 365.2425
    xdecl = (0.006918
             - 0.399912 * jnp.cos(orbit)
             + 0.070257 * jnp.sin(orbit)
             - 0.006758 * jnp.cos(2.0 * orbit)
             + 0.000907 * jnp.sin(2.0 * orbit)
             - 0.002697 * jnp.cos(3.0 * orbit)
             + 0.001480 * jnp.sin(3.0 * orbit))

    # Solar hour angle
    angle = 2.0 * _PI * (rtime - 12.0) / 24.0

    # Effective solar constant
    sw = effective_solar_constant(jday)

    # Solar zenith angle
    xlat = latitude * _PI / 180.0
    coszen = jnp.maximum(
        0.0,
        jnp.sin(xlat) * jnp.sin(xdecl) +
        jnp.cos(xlat) * jnp.cos(xdecl) * jnp.cos(angle))

    # Atmospheric transmissivity (Friend, A.)
    cloud = daily_weather.cloud
    trans = 0.251 + 0.509 * (1.0 - cloud)
    fdiffuse = jnp.where(
        trans > 0.75,
        0.166,
        1.0045 + 0.0435 * trans - 3.5227 * trans**2 + 2.6313 * trans**3)

    # Direct and diffuse solar by waveband
    frac_arr = jnp.array([0.46, 0.54])[:nband]    # VIS=0.46, NIR=0.54
    solad = sw * coszen[:, None] * frac_arr[None, :] * \
            trans[:, None] * (1.0 - fdiffuse[:, None])
    solai = sw * coszen[:, None] * frac_arr[None, :] * \
            trans[:, None] * fdiffuse[:, None]

    # Hourly air temperature (Campbell & Norman Fourier series)
    gamma = (0.44
             - 0.46 * jnp.sin(_PI / 12.0 * rtime + 0.9)
             + 0.11 * jnp.sin(2.0 * _PI / 12.0 * rtime + 0.9))
    ta = daily_weather.tmax * gamma + daily_weather.tmin * (1.0 - gamma)

    # Hourly specific humidity (preserve daily mean, cap RH at 99%)
    psurf = daily_weather.psurf
    qmin = jnp.minimum(
        daily_weather.qd,
        0.99 * qsat(esat(daily_weather.tmin), psurf))
    qmax = (daily_weather.qd - 0.56 * qmin) / 0.44
    esat_ta = esat(ta)
    qsa = 0.99 * qsat(esat_ta, psurf)
    qa = jnp.minimum(qsa, qmax * gamma + qmin * (1.0 - gamma))

    # Relative humidity (%)
    rh = 100.0 * qa / jnp.maximum(qsat(esat_ta, psurf), 1e-10)

    # Hourly wind speed (EPIC weather generator)
    key, k_wind = jax.random.split(key)
    rn_wind = jax.random.uniform(k_wind, (npoi,))
    ua = jnp.clip(
        1.13989 * daily_weather.ud *
        (-jnp.log(jnp.maximum(rn_wind, 1e-10))) ** 0.30,
        2.5, 10.0)

    # Downward IR flux (Idso 1981)
    emb = 0.01 * (psurf * qa / (0.622 + qa))
    ea = 0.70 + 5.95e-5 * emb * jnp.exp(1500.0 / ta)
    ec = 0.950
    fira = ((1.0 - cloud) * ea * _STEF * ta**4 +
            cloud * ec * _STEF * ta**4)

    # Precipitation partitioning: rain vs snow
    in_event = (time >= startp) & (time < endp)
    warm = (ta - 273.15) > 2.5
    rate = daily_weather.precip / jnp.maximum(plens, dtime)
    raina = jnp.where(in_event & warm, rate, 0.0)
    snowa = jnp.where(in_event & ~warm, rate, 0.0)

    return key, HourlyWeather(
        coszen=coszen, ta=ta, qa=qa, ua=ua, rh=rh,
        fira=fira, raina=raina, snowa=snowa,
        solad=solad, solai=solai)
