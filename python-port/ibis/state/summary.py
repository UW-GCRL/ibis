"""Time-averaged diagnostic accumulators.

Ported from mod_summary.f90.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class SummaryState(eqx.Module):
    """Accumulators for daily/monthly/annual output diagnostics."""

    # --- 10-day running means (used by phenology & stomata) ---
    a10td: Float[Array, "npoi"]            # 10-day avg daily temperature (K)
    a10ancub: Float[Array, "npoi"]         # 10-day avg photosynthesis, broadleaf upper
    a10ancuc: Float[Array, "npoi"]         # 10-day avg photosynthesis, conifer upper
    a10ancls: Float[Array, "npoi"]         # 10-day avg photosynthesis, shrub
    a10ancl3: Float[Array, "npoi"]         # 10-day avg photosynthesis, C3 grass
    a10ancl4: Float[Array, "npoi"]         # 10-day avg photosynthesis, C4 grass
    a10scalparamu: Float[Array, "npoi"]    # 10-day scaling parameter, upper
    a10scalparaml: Float[Array, "npoi"]    # 10-day scaling parameter, lower

    # --- Daily accumulators ---
    adrain: Float[Array, "npoi"]           # Daily rainfall (mm)
    adsnow: Float[Array, "npoi"]           # Daily snowfall (mm)
    adaet: Float[Array, "npoi"]            # Daily AET (mm)
    adneetot: Float[Array, "npoi"]         # Daily NEE (kgC/m2)

    # --- Annual accumulators ---
    ayanpptot: Float[Array, "npoi"]        # Annual ABOVE-GROUND NPP (kgC/m2/yr)
    #: Annual TOTAL NPP (kgC/m2/yr). Distinct from ayanpptot, exactly as in
    #: F77 (`comsum.h:117-118`: "annual above-ground npp" vs "annual total
    #: npp"). The port previously accumulated total NPP into ayanpptot while
    #: dynaveg overwrote the same field with the above-ground sum, so the two
    #: quantities collided and the published field silently became ANPP.
    aynpptot: Float[Array, "npoi"]
    ayneetot: Float[Array, "npoi"]         # Annual NEE (kgC/m2/yr)
    ayprcp: Float[Array, "npoi"]           # Annual precipitation (mm/yr)
    ayaet: Float[Array, "npoi"]            # Annual AET (mm/yr)

    # --- Per-PFT annual NPP accumulators (for vegetation dynamics) ---
    aynpp: Float[Array, "npoi npft"]       # Annual NPP per PFT (kgC/m2/yr)
    ayanpp: Float[Array, "npoi npft"]      # Annual above-ground NPP per PFT

    # --- N and leaching accumulators ---
    storedn: Float[Array, "npoi"]          # Stored N from mineralisation
    yrleach: Float[Array, "npoi"]          # Annual C leaching (kgC/m2/yr)
    ynleach: Float[Array, "npoi"]          # Annual N leaching (kgN/m2/yr)

    # --- Annual GPP, Rh, runoff accumulators (for ILAMB/WIEMIP) ---
    # NOTE: placed AFTER ynleach so old restart files (without these fields)
    # can be loaded via backward-compatible leaf appending.
    aygpptot: Float[Array, "npoi"]         # Annual total GPP (kgC/m2/yr)
    ayco2mic: Float[Array, "npoi"]         # Annual heterotrophic resp Rh (kgC/m2/yr)
    ayco2root: Float[Array, "npoi"]        # Annual root resp Ra_root (kgC/m2/yr)
    aytrunof: Float[Array, "npoi"]         # Annual surface runoff (mm/yr)
    aytdrain: Float[Array, "npoi"]         # Annual drainage (mm/yr)

    # --- Time counters (fields with defaults must come last) ---
    ndtimes: int = 0                       # Daily averaging counter
    nmtimes: int = 0                       # Monthly averaging counter
    nytimes: int = 0                       # Annual averaging counter


def create_summary_state(npoi: int, npft: int = 12) -> SummaryState:
    """Create a zero-initialised SummaryState."""
    z = jnp.zeros((npoi,))
    return SummaryState(
        a10td=z, a10ancub=z, a10ancuc=z, a10ancls=z,
        a10ancl3=z, a10ancl4=z,
        a10scalparamu=z, a10scalparaml=z,
        adrain=z, adsnow=z, adaet=z, adneetot=z,
        ayanpptot=z, aynpptot=z, ayneetot=z, ayprcp=z, ayaet=z,
        aynpp=jnp.zeros((npoi, npft)),
        ayanpp=jnp.zeros((npoi, npft)),
        aygpptot=z, ayco2mic=z, ayco2root=z, aytrunof=z, aytdrain=z,
        storedn=z, yrleach=z, ynleach=z,
    )
