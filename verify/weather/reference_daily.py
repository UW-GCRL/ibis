"""Reference Python port of F77 `daily` from weather.f.

Literal line-by-line transcription of the F77 weather generator's daily
routine, using our bit-exact Python ran2 and pure float64 scalar math.
Goal: bit-exact (or ~1-2 ULP from libm) match against F77 daily, so that
this serves as a deterministic reference for Min's vectorized JAX port.

The function operates point-by-point in the same order as F77 and consumes
ran2 in the same order. State that F77 keeps in common blocks is passed in
and out explicitly.

Caveats vs F77:
  - libm `exp/log/pow/sqrt/sin/cos` may differ by <=1 ULP from gfortran's,
    so some float outputs may show 0-2 ULP residual, not strict 0 ULP.
  - We only support the jdaily=0 (weather-generator) path. The jdaily=1
    branch (reading observed daily data) is not used here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from verify.weather.ran2 import Ran2


# ---------------------------------------------------------------------------
# Constants matching F77 daily/comsat.h (all float64).
# ---------------------------------------------------------------------------

_NDAYPM = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)

# Richardson AR(1) matrices. F77 'data a /...,...,/' fills column-major.
# Row j, column k.
_A = np.array([
    [0.600, 0.010, 0.020],
    [0.500, 0.250, 0.125],
    [0.005, 0.005, 0.250],
], dtype=np.float64)
_B = np.array([
    [ 0.500, 0.000, 0.000],
    [ 0.250, 0.500, 0.000],
    [-0.250, 0.250, 0.500],
], dtype=np.float64)

# Lowe (1977) polynomial coefficients (comsat.h).
_ASAT = (6.1078, 4.4365185e-1, 1.4289458e-2, 2.6506485e-4,
         3.0312404e-6, 2.0340809e-8, 6.1368209e-11)
_BSAT = (6.109178, 5.034699e-1, 1.8860134e-2, 4.1762237e-4,
         5.8247203e-6, 4.8388032e-8, 1.8388269e-10)


def _esat(t: float) -> float:
    """Saturation vapor pressure (N/m^2) at temperature t (K). Lowe poly."""
    tsatl = min(100.0, max(t - 273.16, 0.0))
    tsati = max(-60.0, min(t - 273.16, 0.0))
    if t >= 273.16:
        c0, cs = _ASAT[0], _ASAT
    else:
        c0, cs = _BSAT[0], _BSAT
    # Horner-style nested poly identical to F77.
    poly_l = tsatl * (_ASAT[1] + tsatl * (_ASAT[2] + tsatl * (_ASAT[3]
             + tsatl * (_ASAT[4] + tsatl * (_ASAT[5] + tsatl * _ASAT[6])))))
    poly_i = tsati * (_BSAT[1] + tsati * (_BSAT[2] + tsati * (_BSAT[3]
             + tsati * (_BSAT[4] + tsati * (_BSAT[5] + tsati * _BSAT[6])))))
    return 100.0 * (c0 + poly_l + poly_i)


def _qsat(e1: float, p1: float) -> float:
    return 0.622 * e1 / max(p1 - (1.0 - 0.622) * e1, 0.622 * e1)


def _nint(x: float) -> int:
    """Fortran nint: round-half-away-from-zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


# ---------------------------------------------------------------------------
# State / inputs.
# ---------------------------------------------------------------------------

class Climate(NamedTuple):
    xint:    np.ndarray  # [n, 12] C
    xinq:    np.ndarray  # [n, 12] %
    xinprec: np.ndarray  # [n, 12] mm/day
    xinwind: np.ndarray  # [n, 12] m/s
    xincld:  np.ndarray  # [n, 12] %
    xinwet:  np.ndarray  # [n, 12] days/month
    xintrng: np.ndarray  # [n, 12] C
    xintopo: np.ndarray  # [n]


@dataclass
class DailyState:
    """Mutable per-grid-point state daily reads/writes via common blocks."""
    xstore:       np.ndarray  # [n, 3]
    iwetday:      np.ndarray  # [n, 31] int
    precipday:    np.ndarray  # [n, 31] float64
    iwetdaysum:   np.ndarray  # [n] int
    precipdaysum: np.ndarray  # [n]
    tcthis:       np.ndarray  # [n]
    twthis:       np.ndarray  # [n]
    gdd0this:     np.ndarray  # [n]
    gdd5this:     np.ndarray  # [n]
    a10td:        np.ndarray  # [n] (read-only by daily)

    @classmethod
    def fresh(cls, n: int) -> "DailyState":
        return cls(
            xstore       = np.zeros((n, 3), dtype=np.float64),
            iwetday      = np.zeros((n, 31), dtype=np.int64),
            precipday    = np.zeros((n, 31), dtype=np.float64),
            iwetdaysum   = np.zeros(n, dtype=np.int64),
            precipdaysum = np.zeros(n, dtype=np.float64),
            tcthis       = np.full(n,  50.0, dtype=np.float64),
            twthis       = np.full(n, -50.0, dtype=np.float64),
            gdd0this     = np.zeros(n, dtype=np.float64),
            gdd5this     = np.zeros(n, dtype=np.float64),
            a10td        = np.full(n, 288.0, dtype=np.float64),
        )

    def copy(self) -> "DailyState":
        return DailyState(
            xstore=self.xstore.copy(),
            iwetday=self.iwetday.copy(),
            precipday=self.precipday.copy(),
            iwetdaysum=self.iwetdaysum.copy(),
            precipdaysum=self.precipdaysum.copy(),
            tcthis=self.tcthis.copy(),
            twthis=self.twthis.copy(),
            gdd0this=self.gdd0this.copy(),
            gdd5this=self.gdd5this.copy(),
            a10td=self.a10td.copy(),
        )


class DailyOut(NamedTuple):
    td:     np.ndarray
    tmax:   np.ndarray
    tmin:   np.ndarray
    cloud:  np.ndarray
    precip: np.ndarray
    qd:     np.ndarray
    ud:     np.ndarray
    psurf:  np.ndarray
    iwet:   np.ndarray


# ---------------------------------------------------------------------------
# Reference daily.
# ---------------------------------------------------------------------------

class DailyAnomaly(NamedTuple):
    """Observed daily anomalies / fractions (jdaily=1 inputs)."""
    xintd:    np.ndarray   # [n] daily temperature anomaly (C)
    xinqd:    np.ndarray   # [n] daily relative humidity fraction
    xinprecd: np.ndarray   # [n] daily precip fraction
    xinwindd: np.ndarray   # [n] daily wind speed fraction
    xincldd:  np.ndarray   # [n] daily cloud cover anomaly (%)
    xintrngd: np.ndarray   # [n] daily temperature range fraction


def daily_jdaily1(
    imonth: int,
    iday: int,
    climate: Climate,
    anomaly: DailyAnomaly,
    *,
    grav: float = 9.80616,
    rair: float = 287.04,
) -> DailyOut:
    """F77 `daily` with jdaily=1 — use observed daily climatology.

    weather.f lines 824-891 (the `else` branch of `if (jdaily.eq.0)`).
    Pure arithmetic; no RNG. Mirrors F77 line-by-line.
    """
    n = climate.xint.shape[0]
    rwork = grav / rair / 0.0065
    im0 = imonth - 1

    td     = np.empty(n, dtype=np.float64)
    tmax   = np.empty(n, dtype=np.float64)
    tmin   = np.empty(n, dtype=np.float64)
    cloud  = np.empty(n, dtype=np.float64)
    precip = np.empty(n, dtype=np.float64)
    qd     = np.empty(n, dtype=np.float64)
    ud     = np.empty(n, dtype=np.float64)
    psurf  = np.empty(n, dtype=np.float64)

    NDAYPM_MONTH = float(_NDAYPM[im0])

    for i in range(n):
        # daily total precipitation
        precip[i] = (climate.xinprec[i, im0] * NDAYPM_MONTH) * anomaly.xinprecd[i]

        # daily average temperatures
        td[i] = climate.xint[i, im0] + 273.16 + anomaly.xintd[i]
        trngm = min(44.0, climate.xintrng[i, im0] * anomaly.xintrngd[i])
        tmax[i] = td[i] + 0.56 * trngm
        tmin[i] = td[i] - 0.44 * trngm

        # daily average cloud cover
        cloud[i] = (climate.xincld[i, im0] + anomaly.xincldd[i]) * 0.01
        cloud[i] = min(cloud[i], 1.0)
        cloud[i] = max(0.0, cloud[i])

        # surface pressure (hypsometric)
        psurf[i] = 101325.0 * (td[i] / (td[i] + 0.0065 * climate.xintopo[i])) ** rwork

        # daily average specific humidity
        humidfrac = climate.xinq[i, im0] / 100.0
        sphumid = humidfrac * _qsat(_esat(td[i]), psurf[i])
        qd[i] = sphumid * anomaly.xinqd[i]

        # daily average wind speed
        ud[i] = climate.xinwind[i, im0] * anomaly.xinwindd[i]

    return DailyOut(
        td=td, tmax=tmax, tmin=tmin, cloud=cloud, precip=precip,
        qd=qd, ud=ud, psurf=psurf, iwet=np.zeros(n, dtype=np.int64),
    )


def daily(
    iseed: int,
    imonth: int,           # 1..12
    iday: int,             # 1..ndaypm(imonth)
    climate: Climate,
    state: DailyState,     # mutated in place (matches F77 common-block style)
    *,
    grav: float = 9.80616,
    rair: float = 287.04,
) -> DailyOut:
    """One call of F77 `daily` (jdaily=0 path)."""

    # Per-call ran2 init mirrors wrap_daily's seed=-iseed.
    rng = Ran2(iseed)

    # Per-call deterministic xinwet copy — F77 may mutate it in the goto-9000
    # retry. We mutate the climate.xinwet array in place to match.
    xinwet = climate.xinwet

    n = climate.xint.shape[0]
    rwork = grav / rair / 0.0065
    omcloud = 0.90
    omqd    = 0.50
    omtmax  = 0.75

    # --- monthly interpolation indices (1-based, like F77) ---
    fday = float(iday)
    half = float(_NDAYPM[imonth - 1] + 1) / 2.0
    if fday < half:
        it1w = imonth - 1
        it2w = imonth
        dt = (fday - 0.5) / _NDAYPM[imonth - 1] + 0.5
    else:
        it1w = imonth
        it2w = imonth + 1
        dt = (fday - 0.5) / _NDAYPM[imonth - 1] - 0.5
    if it1w < 1:
        it1w = 12
    if it2w > 12:
        it2w = 1

    if iday == 1 and imonth == 1:
        state.tcthis.fill(100.0)
        state.twthis.fill(-100.0)
        state.gdd0this.fill(0.0)
        state.gdd5this.fill(0.0)

    # Allocate outputs.
    td     = np.empty(n, dtype=np.float64)
    tmax   = np.empty(n, dtype=np.float64)
    tmin   = np.empty(n, dtype=np.float64)
    cloud  = np.empty(n, dtype=np.float64)
    precip = np.empty(n, dtype=np.float64)
    qd     = np.empty(n, dtype=np.float64)
    ud     = np.empty(n, dtype=np.float64)
    psurf  = np.empty(n, dtype=np.float64)
    iwet   = np.empty(n, dtype=np.int64)

    # 0-based index helpers
    im0 = imonth - 1
    id0 = iday - 1

    for i in range(n):
        # === (1) Monthly precip generation on iday==1 with goto-9000 retry. ===
        pwet = max(1.0, xinwet[i, im0]) / _NDAYPM[im0]
        pwd  = 0.75 * pwet
        pww  = 0.25 + pwd

        if iday == 1:
            climate.xinprec[i, im0] = max(0.0, climate.xinprec[i, im0])
            xinwet[i, im0]          = max(1.0, xinwet[i, im0])

            retry = True
            while retry:
                retry = False
                state.iwetdaysum[i]   = 0
                state.precipdaysum[i] = 0.0
                state.iwetday[i, :]   = 0
                state.precipday[i, :] = 0.0

                ndpm = int(_NDAYPM[im0])
                for j in range(1, ndpm + 1):  # j is 1-based, matching F77
                    rndnum = rng.uniform()
                    j0 = j - 1
                    if j == 1:
                        if rndnum <= pwd:
                            state.iwetday[i, j0] = 1
                            state.iwetdaysum[i] += 1
                        # else: already 0
                    else:
                        # Corrected Markov chain (matches the bug-fixed F77,
                        # finding #5 was patched 2026-05-14 in weather.f).
                        if state.iwetday[i, j0 - 1] == 0:
                            if rndnum <= pwd:
                                state.iwetday[i, j0] = 1
                                state.iwetdaysum[i] += 1
                        else:
                            if rndnum <= pww:
                                state.iwetday[i, j0] = 1
                                state.iwetdaysum[i] += 1

                    if state.iwetday[i, j0] == 1:
                        rainpwd = (climate.xinprec[i, im0] * _NDAYPM[im0]
                                   / max(0.1, xinwet[i, im0]))
                        beta  = max(1.0, -2.16 + 1.83 * rainpwd)
                        alpha = rainpwd / beta
                        aa = 1.0 / alpha
                        ab = 1.0 / (1.0 - alpha)
                        tr1 = math.exp(-18.42 / aa)
                        tr2 = math.exp(-18.42 / ab)
                        # Rejection loop: re-draw rn1, rn2 until s12 <= 1.
                        while True:
                            rn1 = rng.uniform()
                            rn2 = rng.uniform()
                            s1 = 0.0 if (rn1 - tr1) <= 0 else rn1 ** aa
                            s2 = 0.0 if (rn2 - tr2) <= 0 else rn2 ** ab
                            s12 = s1 + s2
                            if (s12 - 1.0) <= 0:
                                break
                        z = s1 / s12
                        rn3 = rng.uniform()
                        state.precipday[i, j0] = -z * math.log(rn3) * beta
                        state.precipday[i, j0] = max(state.precipday[i, j0], 0.25)
                        state.precipday[i, j0] = min(state.precipday[i, j0], 150.0)

                    state.precipdaysum[i] += state.precipday[i, j0]

                # Fallback when month has 0 precip but climate says wet.
                if state.precipdaysum[i] == 0 and climate.xinprec[i, im0] > 0.0:
                    rndnum = 1.0 + (float(_NDAYPM[im0]) - 1.0) * rng.uniform()
                    k0 = _nint(rndnum) - 1
                    state.iwetday[i, k0] = 1
                    state.precipday[i, k0] = (
                        climate.xinprec[i, im0] * float(_NDAYPM[im0]))
                    state.precipdaysum[i] = state.precipday[i, k0]
                    state.iwetdaysum[i] = 1

                precipfac = ((climate.xinprec[i, im0] * float(_NDAYPM[im0]))
                             / max(0.01, state.precipdaysum[i]))

                # Rescale + check for >360 mm anywhere.
                exceeded = False
                for j in range(1, ndpm + 1):
                    j0 = j - 1
                    state.precipday[i, j0] *= precipfac
                    if state.precipday[i, j0] > 360.0:
                        if xinwet[i, im0] < _NDAYPM[im0]:
                            xinwet[i, im0] += 1
                            # pwet/pwd/pww update happens implicitly via reload
                            # at the top of the while loop.
                            pwet = max(1.0, xinwet[i, im0]) / _NDAYPM[im0]
                            pwd  = 0.75 * pwet
                            pww  = 0.25 + pwd
                            retry = True
                            exceeded = True
                            break
                        else:
                            # F77 also goto 9000 in this branch, but with no
                            # capacity left to grow xinwet, it would loop
                            # forever — in practice this is just a warning
                            # path. We still retry once to match line-for-line.
                            retry = True
                            exceeded = True
                            break

                if not retry:
                    # Verification recount that F77 does.
                    state.iwetdaysum[i] = 0
                    state.precipdaysum[i] = 0.0
                    for j in range(1, ndpm + 1):
                        j0 = j - 1
                        state.precipdaysum[i] += state.precipday[i, j0]
                        state.iwetdaysum[i]   += state.iwetday[i, j0]

        iwet[i]   = int(state.iwetday[i, id0])
        precip[i] = state.precipday[i, id0]

        # === (3) expected min/max temperatures ===
        tdm = (climate.xint[i, it1w - 1]
               + dt * (climate.xint[i, it2w - 1] - climate.xint[i, it1w - 1])
               + 273.16)
        trngm = (climate.xintrng[i, it1w - 1]
                 + dt * (climate.xintrng[i, it2w - 1]
                         - climate.xintrng[i, it1w - 1]))
        tmaxm = tdm + 0.56 * trngm
        tminm = tdm - 0.44 * trngm

        if pwet != 0.0:
            tmaxd = tmaxm + pwet * omtmax * trngm
            tmaxw = tmaxd - omtmax * trngm
        else:
            tmaxd = tmaxm
            tmaxw = tmaxm

        tmaxe = tmaxd if iwet[i] == 0 else tmaxw
        tmine = tminm

        tmaxs = max(1.0, -0.0713 * (tdm - 273.16) + 4.89)
        tmins = max(1.0, -0.1280 * (tdm - 273.16) + 5.73)

        # === (4) cloud cover ===
        if iday == 1:
            climate.xincld[i, it1w - 1] = max(0.0,   climate.xincld[i, it1w - 1])
            climate.xincld[i, it1w - 1] = min(100.0, climate.xincld[i, it1w - 1])
            climate.xincld[i, it2w - 1] = max(0.0,   climate.xincld[i, it2w - 1])
            climate.xincld[i, it2w - 1] = min(100.0, climate.xincld[i, it2w - 1])

        cloudm = (climate.xincld[i, it1w - 1]
                  + dt * (climate.xincld[i, it2w - 1]
                          - climate.xincld[i, it1w - 1]))
        cloudm /= 100.0

        if pwet != 0.0:
            cloudd = (cloudm - pwet * omcloud) / (1.0 - pwet * omcloud)
            cloudd = min(1.0, max(0.0, cloudd))
            cloudw = (cloudm - (1.0 - pwet) * cloudd) / pwet
        else:
            cloudd = cloudm
            cloudw = cloudm

        cloude = cloudd if iwet[i] == 0 else cloudw
        clouds = 0.24 * cloude if iwet[i] == 0 else 0.48 * cloude

        # === (5) AR(1) update ===
        ee = np.zeros(3, dtype=np.float64)
        for j in range(3):
            while True:
                rn1 = rng.uniform()
                rn2 = rng.uniform()
                v = math.sqrt(-2.0 * math.log(rn1)) * math.cos(6.283185 * rn2)
                if abs(v) <= 2.5:
                    break
            ee[j] = v

        r  = np.zeros(3, dtype=np.float64)
        rr = np.zeros(3, dtype=np.float64)
        # F77 has: r(j) = sum_k b(j,k) * ee(j) -- note ee(j) not ee(k).
        # So r(j) = (sum_k b(j,k)) * ee(j).
        for j in range(3):
            for k in range(3):
                r[j]  += _B[j, k] * ee[j]
                rr[j] += _A[j, k] * state.xstore[i, k]

        x_pert = r + rr
        state.xstore[i, :] = x_pert

        tmax[i] = tmaxe + tmaxs * x_pert[0]
        tmin[i] = tmine + tmins * x_pert[1]
        if tmin[i] > tmax[i]:
            tmin[i], tmax[i] = tmax[i], tmin[i]

        td[i] = 0.44 * tmax[i] + 0.56 * tmin[i]

        cloud[i] = cloude + clouds * x_pert[2]
        cloud[i] = max(0.0, min(1.0, cloud[i]))

        # === (6) surface pressure ===
        psurf[i] = 101325.0 * (td[i] / (td[i] + 0.0065 * climate.xintopo[i])) ** rwork

        # === (7) humidity ===
        if iday == 1:
            climate.xinq[i, it1w - 1] = max(0.0,   climate.xinq[i, it1w - 1])
            climate.xinq[i, it1w - 1] = min(100.0, climate.xinq[i, it1w - 1])
            climate.xinq[i, it2w - 1] = max(0.0,   climate.xinq[i, it2w - 1])
            climate.xinq[i, it2w - 1] = min(100.0, climate.xinq[i, it2w - 1])

        qdm = (climate.xinq[i, it1w - 1]
               + dt * (climate.xinq[i, it2w - 1] - climate.xinq[i, it1w - 1]))
        qdm /= 100.0

        if pwet != 0.0:
            qdd = (qdm - pwet * omqd) / (1.0 - pwet * omqd)
            if qdd < 0.2:
                qdd = 0.2
                if qdd > qdm:
                    qdm = qdd
            qdd = min(1.0, qdd)
            qdw = (qdm - (1.0 - pwet) * qdd) / pwet
        else:
            qdd = qdm
            qdw = qdm

        qde = qdd if iwet[i] == 0 else qdw

        qdup  = qde + (1.0 - qde) * math.exp(qde - 1.0)
        qdlow = qde * (1.0 - math.exp(-qde))

        rn = rng.uniform()
        y_tri = 2.0 / (qdup - qdlow)
        b3 = qde  - qdlow
        b2 = qdup - qde
        b1 = rn / y_tri
        x1 = y_tri * b3 / 2.0
        if rn > x1:
            qd_val = qdup  - math.sqrt(b2 * b2 - 2.0 * b2 * (b1 - 0.5 * b3))
        else:
            qd_val = qdlow + math.sqrt(2.0 * b1 * b3)

        amn = (qdup + qde + qdlow) / 3.0
        qd_val = qd_val * qde / amn
        qd_val = max(0.30, qd_val)
        qd_val = min(0.99, qd_val)
        qd[i] = qd_val * _qsat(_esat(td[i]), psurf[i])

        # === (8) wind ===
        eud = (climate.xinwind[i, it1w - 1]
               + dt * (climate.xinwind[i, it2w - 1]
                       - climate.xinwind[i, it1w - 1]))
        ud[i] = 1.13989 * eud * (-math.log(rng.uniform())) ** 0.30
        ud[i] = max(2.5, min(10.0, ud[i]))

        # === final running min/max/GDD updates ===
        state.tcthis[i]   = min(state.tcthis[i], (state.a10td[i] - 273.16))
        state.twthis[i]   = max(state.twthis[i], (state.a10td[i] - 273.16))
        state.gdd0this[i] = state.gdd0this[i] + max(0.0, (td[i] - 273.16))
        state.gdd5this[i] = state.gdd5this[i] + max(0.0, (td[i] - 278.16))

    return DailyOut(
        td=td, tmax=tmax, tmin=tmin, cloud=cloud, precip=precip,
        qd=qd, ud=ud, psurf=psurf, iwet=iwet,
    )
