"""Seasonal states for two optional formulations (docs/CALIBRATION.md 21-22).

`acclim_x` -- state of acclimation of the photosynthetic apparatus (deg C): the first-order delay of daily mean air
temperature of Makela et al. (2004, Tree Physiology 24, 369), in the discrete daily form PRELES uses
(Peltoniemi et al. 2015; Rpreles src/gpp.c `fS_model`):

    S <- S + (T - S) / tau
    f_S = clip((S - S0) / Smax, 0, 1)

and `f_S` scales the photosynthetic capacity (vmax, hence dark respiration) of the PFTs in
`StomataParams.acclim_pfts`. Evergreen conifers down-regulate photosynthesis completely in winter while they stay
green; without this IBIS's capacity is constant per PFT with an instantaneous temperature response, and no
parameter could remove the winter photosynthesis the towers do not see (CALIBRATION.md 20-21).

`ssen_u`, `ssen_l` -- senescence states (deg C day) of Delpierre et al. (2009, Agric. For. Meteorol. 149, 938; the
equations as restated in Vitasse et al. 2011, eqs 11-15) for the upper and lower canopy, with x = 1, y = 0: while
day length is decreasing and below Pstart, cold degree-days below Tb accumulate,

    S_sen <- S_sen + max(Tb - T, 0)

and leaf fall starts when S_sen >= Ycrit. One deliberate departure: the state resets whenever day length is not
decreasing (at the winter solstice) rather than only when it climbs back above Pstart in spring. Delpierre's reset
would keep cold-deciduous PFTs dormant for ever wherever day length never exceeds Pstart (|lat| below ~30 deg), and
it would make Pstart a spring parameter as well; with the solstice reset leaf-out stays the model's degree-day rule.

A state without the module keeps its leaf count. With it, create_ibis_state also attaches StressAccum and ETParts,
so the trailing-leaf layouts form one nested chain (258 / 261 / 265 / 268) and load_state migrates along it. The
module is inert unless `stomata: acclim_on` or `pheno: sen_photoperiod_u / sen_photoperiod_l` is set: its states
advance, but nothing reads them.
"""
from __future__ import annotations
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class SeasonalState(eqx.Module):
    acclim_x: Float[Array, "npoi"]   # state of acclimation, deg C (Makela 2004 / PRELES)
    ssen_u: Float[Array, "npoi"]     # Delpierre senescence state, upper canopy, deg C day
    ssen_l: Float[Array, "npoi"]     # Delpierre senescence state, lower canopy, deg C day


FIELDS = ("acclim_x", "ssen_u", "ssen_l")


def create_seasonal(npoi: int, acclim_x0: float = 0.0) -> SeasonalState:
    """Zero senescence states; the acclimation state starts at `acclim_x0` deg C and relaxes to the local
    temperature with the time constant tau (about 11 days) -- within any spin-up."""
    z = jnp.zeros((npoi,))
    return SeasonalState(acclim_x=z + acclim_x0, ssen_u=z, ssen_l=z)


def daylength_hours(lat_rad, doy):
    """Astronomical day length (h) from latitude (radians) and day of year (1-based): solar declination
    23.44 deg * sin(2 pi (doy - 81) / 365) and the sunrise hour-angle equation, clipped for polar day/night."""
    decl = 0.40910518 * jnp.sin(2.0 * jnp.pi * (doy - 81.0) / 365.0)
    x = jnp.clip(-jnp.tan(lat_rad) * jnp.tan(decl), -1.0, 1.0)
    return 24.0 / jnp.pi * jnp.arccos(x)
