"""Convert GSWP3 daily arrays to DailyWeather objects for IBIS-JAX.

This module bridges the GSWP3 transient forcing data (from GSWP3Reader)
to the DailyWeather namedtuple consumed by the IBIS-JAX timestepping loops.

Unit conversions applied:
    pr      : kg m-2 s-1 → mm day-1  (× 86400)
    rhs     : %          → fraction  (/ 100) [stored in cloud derivation]

Cloud fraction derivation from rsds
------------------------------------
The IBIS diurnal disaggregator uses:
    trans  = 0.251 + 0.509 * (1 - cloud)          [transmissivity]
    solrad = SW_eff * coszen * trans               [instantaneous solar]

We invert this to get cloud from the observed daily-mean rsds:
    trans  = rsds / (SW_eff * mean_coszen_24h)
    cloud  = clip(1 - (trans - 0.251) / 0.509, 0, 1)

where mean_coszen_24h is the 24-hour integral of max(0, coszen) / 86400,
computed analytically from latitude and solar declination. The same orbital
and declination formulae as weather.py (diurnal) are used for consistency.

SW_eff accounts for Earth-Sun distance variation (same formula as weather.py).
At polar night (mean_coszen_24h ≈ 0) cloud defaults to 0.5.

Longwave (fira)
---------------
GSWP3 provides observed rlds (W m-2). Since the existing diurnal() function
computes fira from cloud and temperature, we store rlds separately and
override fira in run_day after the diurnal call (see run_year_gswp3 in
ibis/timestepping/loops_gswp3.py).
"""

from __future__ import annotations

import numpy as np

from ibis.physics.weather import DailyWeather

# ---------------------------------------------------------------------------
# Constants (must match weather.py diurnal)
# ---------------------------------------------------------------------------
_PI = np.pi
_S0 = 1370.0                    # Solar constant (W m-2)
_PREC_THRESH = 0.1              # mm/day threshold for wet-day flag


# ---------------------------------------------------------------------------
# Solar geometry helpers
# ---------------------------------------------------------------------------

def _orbital_angle(jday: int) -> float:
    """Earth orbital angle (radians) for julian day (1-based)."""
    return 2.0 * _PI * jday / 365.2425


def _declination(orbit: float) -> float:
    """Solar declination (radians) — Hartmann formula, matches weather.py."""
    return (0.006918
            - 0.399912 * np.cos(orbit)
            + 0.070257 * np.sin(orbit)
            - 0.006758 * np.cos(2.0 * orbit)
            + 0.000907 * np.sin(2.0 * orbit)
            - 0.002697 * np.cos(3.0 * orbit)
            + 0.001480 * np.sin(3.0 * orbit))


def _sw_eff(orbit: float) -> float:
    """Effective solar constant (W m-2) accounting for Earth-Sun distance."""
    return _S0 * (1.000110
                  + 0.034221 * np.cos(orbit)
                  + 0.001280 * np.sin(orbit)
                  + 0.000719 * np.cos(2.0 * orbit)
                  + 0.000077 * np.sin(2.0 * orbit))


def daily_mean_coszen(lat_rad: np.ndarray, jday: int) -> np.ndarray:
    """24-hour mean cosine of solar zenith angle for each land point.

    Computed analytically:
        mean_coszen_24h = (sin(lat)*sin(decl)*h_s
                           + cos(lat)*cos(decl)*sin(h_s)) / pi

    where h_s is the sunset hour angle (radians).

    Parameters
    ----------
    lat_rad : np.ndarray, shape (npoi,)
        Latitude in radians (positive = N).
    jday : int
        Julian day (1-based).

    Returns
    -------
    np.ndarray, shape (npoi,)  — values in [0, 1].
    """
    orbit = _orbital_angle(jday)
    decl = _declination(orbit)

    cos_h_s = -np.tan(lat_rad) * np.tan(decl)
    h_s = np.arccos(np.clip(cos_h_s, -1.0, 1.0))  # sunset hour angle

    mean_cz = (np.sin(lat_rad) * np.sin(decl) * h_s
               + np.cos(lat_rad) * np.cos(decl) * np.sin(h_s)) / _PI
    return np.maximum(0.0, mean_cz)


def rsds_to_cloud(
    rsds: np.ndarray,
    lat_rad: np.ndarray,
    jday: int,
    min_coszen: float = 0.01,
) -> np.ndarray:
    """Derive cloud fraction from observed daily-mean rsds.

    Inverts the IBIS transmissivity formula:
        trans = rsds / (SW_eff * mean_coszen_24h)
        cloud = clip(1 - (trans - 0.251) / 0.509, 0, 1)

    At polar night (mean_coszen < min_coszen) defaults to 0.5.

    Parameters
    ----------
    rsds : np.ndarray, shape (npoi,)
        Daily mean downwelling shortwave (W m-2).
    lat_rad : np.ndarray, shape (npoi,)
        Latitude in radians.
    jday : int
        Julian day (1-based).
    min_coszen : float
        Minimum mean coszen to attempt cloud derivation.

    Returns
    -------
    np.ndarray, shape (npoi,)  — cloud fraction in [0, 1].
    """
    orbit = _orbital_angle(jday)
    sw = _sw_eff(orbit)
    mcz = daily_mean_coszen(lat_rad, jday)

    # Transmissivity (where solar radiation is non-negligible)
    polar_night = mcz < min_coszen
    denom = np.where(polar_night, 1.0, sw * mcz)
    trans = np.clip(rsds / denom, 0.0, 1.0)

    # Invert: cloud = 1 - (trans - 0.251) / 0.509
    cloud = 1.0 - (trans - 0.251) / 0.509
    cloud = np.clip(cloud, 0.0, 1.0)

    # Default 0.5 at polar night
    cloud = np.where(polar_night, 0.5, cloud)
    return cloud.astype(np.float32)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def gswp3_day_to_daily_weather(
    gswp3_month: dict[str, np.ndarray],
    iday: int,
    lat_rad: np.ndarray,
    jday: int,
    xstore: np.ndarray,
) -> tuple[DailyWeather, np.ndarray]:
    """Convert one day of GSWP3 data to a DailyWeather namedtuple.

    Parameters
    ----------
    gswp3_month : dict
        Output of GSWP3Reader.read_month(); arrays are (ndays, npoi).
    iday : int
        Day index within the month (0-based).
    lat_rad : np.ndarray, shape (npoi,)
        Latitude of each land point in radians.
    jday : int
        Julian day of the year (1-based).
    xstore : np.ndarray, shape (npoi, 3)
        AR(1) state (not used in transient mode; passed through unchanged).

    Returns
    -------
    dw : DailyWeather
    rlds : np.ndarray, shape (npoi,)
        Actual observed downwelling longwave (W m-2), to be used to
        override the cloud-based fira in the diurnal disaggregator.
    """
    npoi = lat_rad.shape[0]

    def _get(key: str) -> np.ndarray:
        arr = gswp3_month[key][iday]      # (npoi,)
        # Replace NaN with a reasonable fallback (should not occur on land)
        arr = np.where(np.isnan(arr), 0.0, arr)
        return arr.astype(np.float32)

    tas    = _get('tas')      # K
    tasmax = _get('tasmax')   # K
    tasmin = _get('tasmin')   # K
    pr     = _get('pr')       # kg m-2 s-1
    rsds   = _get('rsds')     # W m-2
    rlds   = _get('rlds')     # W m-2
    huss   = _get('huss')     # kg kg-1
    wind   = _get('wind')     # m s-1
    ps     = _get('ps')       # Pa

    # --- Unit conversions ---
    precip_mm = pr * 86400.0              # mm day-1
    iwet = (precip_mm > _PREC_THRESH).astype(np.int32)

    # --- Cloud fraction from rsds ---
    cloud = rsds_to_cloud(rsds, lat_rad, jday)

    dw = DailyWeather(
        td=tas,
        tmax=tasmax,
        tmin=tasmin,
        cloud=cloud,
        precip=precip_mm,
        psurf=ps,
        qd=huss,
        ud=wind,
        iwet=iwet,
        xstore=xstore,
    )
    return dw, rlds


def make_monthly_daily_weather(
    gswp3_month: dict[str, np.ndarray],
    lat_rad: np.ndarray,
    year: int,
    month: int,
    xstore: np.ndarray,
) -> tuple[list[DailyWeather], list[np.ndarray]]:
    """Build DailyWeather objects for every day in a GSWP3 month.

    Parameters
    ----------
    gswp3_month : dict
        Output of GSWP3Reader.read_month().
    lat_rad : np.ndarray, shape (npoi,)
    year, month : int
        Year and 0-based month.
    xstore : np.ndarray, shape (npoi, 3)
        Initial AR(1) state (passed through each day).

    Returns
    -------
    daily_list : list of DailyWeather, length ndays
    rlds_list  : list of np.ndarray (npoi,), length ndays
    """
    from ibis.timestepping.calendar import julian_day as _julian_day, is_leap_year

    leap = is_leap_year(year)
    ndays = gswp3_month['ndays']

    daily_list: list[DailyWeather] = []
    rlds_list: list[np.ndarray] = []

    for iday in range(ndays):
        jday = _julian_day(month, iday, leap)
        dw, rlds = gswp3_day_to_daily_weather(
            gswp3_month, iday, lat_rad, jday, xstore,
        )
        daily_list.append(dw)
        rlds_list.append(rlds)

    return daily_list, rlds_list
