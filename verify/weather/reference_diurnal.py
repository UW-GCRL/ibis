"""Reference Python port of F77 `diurnal` from weather.f.

Literal transcription of the F77 sub-daily disaggregator, using our bit-exact
ran2 port. Same semantics as reference_daily: loop over grid points in F77
order so ran2 is consumed identically.

Inputs / outputs match `wrap_diurnal`.

Caveats vs F77:
  - Same libm-noise caveat as reference_daily — but F77 with -O0 + gfortran
    libm and Python's math both call glibc, so results match bit-exactly in
    practice.
  - The `goto 9001` retry triggered by precip/plen > 15 mm/timestep on the
    first timestep of the day is implemented as a Python while-loop. F77
    restarts the grid loop with new plens/startp/endp; we do the same.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from verify.weather.ran2 import Ran2


# F77 compile-time parameters.
_PI    = 3.1415927     # F77 parameter (compar.h, line 88)
_STEF  = 5.67e-8
_NBAND = 2


class DiurnalInputs(NamedTuple):
    cloud:  np.ndarray   # [n] fraction
    tmax:   np.ndarray   # [n] K
    tmin:   np.ndarray   # [n] K
    qd:     np.ndarray   # [n] kg/kg
    ud:     np.ndarray   # [n] m/s daily-mean wind
    psurf:  np.ndarray   # [n] Pa
    precip: np.ndarray   # [n] mm/day
    lat:    np.ndarray   # [n] degrees


class DiurnalOut(NamedTuple):
    coszen: np.ndarray   # [n]
    ta:     np.ndarray   # [n]
    qa:     np.ndarray   # [n]
    ua:     np.ndarray   # [n]
    rh:     np.ndarray   # [n] percent
    fira:   np.ndarray   # [n] W/m2
    raina:  np.ndarray   # [n] kg/m2/s
    snowa:  np.ndarray   # [n] kg/m2/s
    solad:  np.ndarray   # [n, 2]
    solai:  np.ndarray   # [n, 2]
    plens:  float
    startp: float
    endp:   float


# Reuse esat/qsat from reference_daily (identical algebra; we re-declare to
# avoid a circular import while keeping each file self-contained).
_ASAT = (6.1078, 4.4365185e-1, 1.4289458e-2, 2.6506485e-4,
         3.0312404e-6, 2.0340809e-8, 6.1368209e-11)
_BSAT = (6.109178, 5.034699e-1, 1.8860134e-2, 4.1762237e-4,
         5.8247203e-6, 4.8388032e-8, 1.8388269e-10)


def _esat(t: float) -> float:
    tsatl = min(100.0, max(t - 273.16, 0.0))
    tsati = max(-60.0, min(t - 273.16, 0.0))
    c0 = _ASAT[0] if t >= 273.16 else _BSAT[0]
    poly_l = tsatl * (_ASAT[1] + tsatl * (_ASAT[2] + tsatl * (_ASAT[3]
             + tsatl * (_ASAT[4] + tsatl * (_ASAT[5] + tsatl * _ASAT[6])))))
    poly_i = tsati * (_BSAT[1] + tsati * (_BSAT[2] + tsati * (_BSAT[3]
             + tsati * (_BSAT[4] + tsati * (_BSAT[5] + tsati * _BSAT[6])))))
    return 100.0 * (c0 + poly_l + poly_i)


def _qsat(e1: float, p1: float) -> float:
    return 0.622 * e1 / max(p1 - (1.0 - 0.622) * e1, 0.622 * e1)


def diurnal(
    iseed: int,
    time: float,         # s into the day
    jday: int,           # 1..365
    plens: float, startp: float, endp: float,   # precip event timing (s)
    dtime: float,        # timestep length (s)
    inputs: DiurnalInputs,
) -> DiurnalOut:
    """One call of F77 `diurnal` (Layer-1 reference)."""
    rng = Ran2(iseed)
    n = inputs.cloud.shape[0]

    # --- calendar/orbital (per call, not per gridpoint) ---
    rtime = time / 3600.0
    orbit = 2.0 * _PI * float(jday) / 365.2425
    angle = 2.0 * _PI * (rtime - 12.0) / 24.0
    xdecl = (0.006918
             - 0.399912 * math.cos(orbit)
             + 0.070257 * math.sin(orbit)
             - 0.006758 * math.cos(2.0 * orbit)
             + 0.000907 * math.sin(2.0 * orbit)
             - 0.002697 * math.cos(3.0 * orbit)
             + 0.001480 * math.sin(3.0 * orbit))
    sw = 1370.0 * (1.000110
                   + 0.034221 * math.cos(orbit)
                   + 0.001280 * math.sin(orbit)
                   + 0.000719 * math.cos(2.0 * orbit)
                   + 0.000077 * math.sin(2.0 * orbit))

    coszen = np.empty(n, dtype=np.float64)
    ta     = np.empty(n, dtype=np.float64)
    qa     = np.empty(n, dtype=np.float64)
    ua     = np.empty(n, dtype=np.float64)
    rh     = np.empty(n, dtype=np.float64)
    fira   = np.empty(n, dtype=np.float64)
    raina  = np.zeros(n, dtype=np.float64)
    snowa  = np.zeros(n, dtype=np.float64)
    solad  = np.empty((n, _NBAND), dtype=np.float64)
    solai  = np.empty((n, _NBAND), dtype=np.float64)

    # F77 has `9001 continue` outside the grid loop. Implement as outer while.
    while True:
        restart = False
        for i in range(n):
            xlat = inputs.lat[i] * _PI / 180.0
            coszen[i] = max(0.0, (math.sin(xlat) * math.sin(xdecl)
                                 + math.cos(xlat) * math.cos(xdecl)
                                 * math.cos(angle)))
            trans = 0.251 + 0.509 * (1.0 - inputs.cloud[i])
            fdiffuse = (1.0045 + 0.0435 * trans
                        - 3.5227 * trans ** 2
                        + 2.6313 * trans ** 3)
            if trans > 0.75:
                fdiffuse = 0.166

            for ib in range(_NBAND):
                frac = 0.46 + 0.08 * float(ib)
                solad[i, ib] = sw * coszen[i] * frac * trans * (1.0 - fdiffuse)
                solai[i, ib] = sw * coszen[i] * frac * trans * fdiffuse

            # Temperature disaggregation.
            gamma = (0.44
                     - 0.46 * math.sin(      _PI / 12.0 * rtime + 0.9)
                     + 0.11 * math.sin(2.0 * _PI / 12.0 * rtime + 0.9))
            ta[i] = inputs.tmax[i] * gamma + inputs.tmin[i] * (1.0 - gamma)

            # Humidity.
            qmin_i = min(inputs.qd[i],
                         0.99 * _qsat(_esat(inputs.tmin[i]), inputs.psurf[i]))
            qmax_i = (inputs.qd[i] - 0.56 * qmin_i) / 0.44
            qsa = 0.99 * _qsat(_esat(ta[i]), inputs.psurf[i])
            qa[i] = min(qsa, qmax_i * gamma + qmin_i * (1.0 - gamma))
            rh[i] = 100.0 * qa[i] / _qsat(_esat(ta[i]), inputs.psurf[i])

            # Wind disaggregation (ran2 call).
            ua[i] = 1.13989 * inputs.ud[i] * (-math.log(rng.uniform())) ** 0.30
            ua[i] = max(2.5, min(10.0, ua[i]))

            # IR flux.
            emb = 0.01 * (inputs.psurf[i] * qa[i] / (0.622 + qa[i]))
            ea  = 0.70 + 5.95e-5 * emb * math.exp(1500.0 / ta[i])
            ec  = 0.950
            dtair   = 0.0
            dtcloud = 0.0
            fira[i] = ((1.0 - inputs.cloud[i]) * ea * _STEF * (ta[i] - dtair  ) ** 4
                       +    inputs.cloud[i]   * ec * _STEF * (ta[i] - dtcloud) ** 4)

            snowa[i] = 0.0
            raina[i] = 0.0

            niter = int(86400.0 / dtime)

            if time < dtime:
                # F77 uses integer division for plen.
                plen = int(plens / dtime)
                plenmin = 1 + int((4.0 * 3600.0 - 1.0) / dtime)
                plenmax = max(int(24.0 * 3600.0 / dtime), plenmin)
                checkP = 0
                while ((inputs.precip[i] / plen) > 15.0) and (plen < plenmax):
                    plen = plen + 1
                    checkP = 1

                if checkP == 1:
                    plens  = dtime * plen
                    rn_val = rng.uniform()
                    startp = dtime * min(niter - plen,
                                         int(rn_val * (niter - plen + 1)))
                    endp   = startp + plen * dtime
                    restart = True
                    break    # exit i-loop, redo from 9001

            if (time >= startp) and (time < endp):
                if (ta[i] - 273.15) > 2.5:
                    raina[i] = inputs.precip[i] / plens
                else:
                    snowa[i] = inputs.precip[i] / plens

        if not restart:
            break

    return DiurnalOut(
        coszen=coszen, ta=ta, qa=qa, ua=ua, rh=rh,
        fira=fira, raina=raina, snowa=snowa,
        solad=solad, solai=solai,
        plens=plens, startp=startp, endp=endp,
    )
