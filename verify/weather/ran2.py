"""Bit-exact Python port of Numerical Recipes ran2 (the RNG used by IBIS F77).

Reference: Press et al., Numerical Recipes in Fortran 77, 2nd ed., 1992,
section 7.1. Also reproduced in `fortran/ibis_2.6b4/math.f`.

The original F77:
    real function ran2 (idum)
      ... integer state idum, idum2, iv(32), iy
      ... L'Ecuyer's combined generator with Bays-Durham shuffle

Mapping notes:
  - All integer math fits in int32 (no overflow). We use numpy int64 internally
    to be safe, but every intermediate result stays well inside int32 range.
  - The state (idum, idum2, iv, iy) is kept in an instance, not module-level,
    so multiple independent streams are possible.
  - Float conversion: `ran2 = min(am*iy, rnmx)`. With F77 `-fdefault-real-8`,
    am = 1.0/im1 is float64, am*iy is float64, eps = 1.0e-7 is float64.
    We match exactly using Python float (= IEEE 754 float64).

Public API:
    rng = Ran2(seed)            # seed > 0; internally calls with idum=-seed
    x   = rng.uniform()         # next uniform deviate in (0, 1-eps]
    xs  = rng.uniforms(n)       # n successive deviates
"""
from __future__ import annotations

import numpy as np


_IM1   = 2147483563
_IM2   = 2147483399
_IMM1  = _IM1 - 1
_IA1   = 40014
_IA2   = 40692
_IQ1   = 53668
_IQ2   = 52774
_IR1   = 12211
_IR2   = 3791
_NTAB  = 32
_NDIV  = 1 + _IMM1 // _NTAB    # 67108862 = 1 + 2147483562/32 (integer division)
_AM    = 1.0 / _IM1            # float64
_EPS   = 1.0e-7                # float64 under -fdefault-real-8
_RNMX  = 1.0 - _EPS


class Ran2:
    """Stateful Numerical Recipes ran2.

    Constructor takes a positive int seed. The first call triggers the same
    initialization sequence as F77 when called with idum=-seed.
    """

    __slots__ = ("idum", "idum2", "iv", "iy")

    def __init__(self, seed: int):
        if seed <= 0:
            raise ValueError("ran2 seed must be positive (we pass -seed internally).")
        # Reproduce: if (idum.le.0) then idum=max(-idum,1); idum2=idum; ...
        idum = max(seed, 1)
        idum2 = idum
        iv = [0] * _NTAB
        # do 10 j = ntab+8, 1, -1
        for j in range(_NTAB + 8, 0, -1):
            k = idum // _IQ1
            idum = _IA1 * (idum - k * _IQ1) - k * _IR1
            if idum < 0:
                idum += _IM1
            if j <= _NTAB:
                iv[j - 1] = idum     # F77 1-based -> 0-based
        iy = iv[0]
        self.idum  = idum
        self.idum2 = idum2
        self.iv    = iv
        self.iy    = iy

    def uniform(self) -> float:
        """Return next uniform deviate in (0, 1-eps] as float64."""
        idum  = self.idum
        idum2 = self.idum2
        iv    = self.iv
        iy    = self.iy

        k = idum // _IQ1
        idum = _IA1 * (idum - k * _IQ1) - k * _IR1
        if idum < 0:
            idum += _IM1

        k = idum2 // _IQ2
        idum2 = _IA2 * (idum2 - k * _IQ2) - k * _IR2
        if idum2 < 0:
            idum2 += _IM2

        j = iy // _NDIV       # F77: j = 1 + iy/ndiv (1-based); 0-based here
        iy = iv[j] - idum2
        iv[j] = idum
        if iy < 1:
            iy += _IMM1

        self.idum  = idum
        self.idum2 = idum2
        # iv mutated in place
        self.iy    = iy

        val = _AM * iy
        return val if val < _RNMX else _RNMX

    def uniforms(self, n: int) -> np.ndarray:
        """Generate n successive deviates, returned as float64 array."""
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self.uniform()
        return out
