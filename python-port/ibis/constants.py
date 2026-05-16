"""Physical constants for IBIS-JAX.

Values are taken verbatim from the F77 original (ibis_2.6b4/initial.f,
subroutine inicon).  Do NOT change these without a full regression test.

This module is the JAX equivalent of mod_constants.f90.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class PhysicalConstants(eqx.Module):
    """Immutable container of physical constants.

    Registered as a JAX pytree so it can be passed through jit/vmap/grad.
    All fields are static (frozen); none are updated during the simulation.
    """

    # Stefan-Boltzmann constant (W m-2 K-4)
    stef: float = 5.67e-8
    # von Karman constant (dimensionless)
    vonk: float = 0.4
    # Gravitational acceleration (m s-2)
    grav: float = 9.80616
    # Freezing point of water (K)
    tmelt: float = 273.16
    # Latent heat of vaporisation (J kg-1)
    hvap: float = 2.5104e6
    # Latent heat of fusion (J kg-1)
    hfus: float = 0.3336e6
    # Latent heat of sublimation (J kg-1) -- derived: hvap + hfus
    hsub: float = 2.5104e6 + 0.3336e6
    # Specific heat of liquid water (J kg-1 K-1)
    ch2o: float = 4.218e3
    # Specific heat of ice (J kg-1 K-1)
    cice: float = 2.106e3
    # Specific heat of dry air at constant pressure (J kg-1 K-1)
    cair: float = 1.00464e3
    # Specific heat of water vapour (J kg-1 K-1)
    cvap: float = 1.81e3
    # Gas constant for dry air (J kg-1 K-1)
    rair: float = 287.04
    # Gas constant for water vapour (J kg-1 K-1)
    rvap: float = 461.0
    # rair / cair (dimensionless) -- derived
    cappa: float = 287.04 / 1.00464e3
    # Density of liquid water (kg m-3)
    rhow: float = 1.0e3
    # Small number to avoid zero-divides
    epsilon: float = 1.0e-7

    # Pi -- preserved from F77 original (3.1415927, NOT high-precision)
    pi: float = 3.1415927

    # Days per month (non-leap default)
    ndaypm: tuple = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

    # ----- Photosynthesis parameters (from physiology.f90) -----
    # Rubisco specificity factor at 15C
    tau15: float = 2600.0
    # Michaelis-Menten for CO2 at 15C (mol/mol)
    kc15: float = 1.5e-4
    # Michaelis-Menten for O2 at 15C (mol/mol)
    ko15: float = 0.25
    # Quantum efficiency, C3
    alpha3: float = 0.06
    # Co-limitation curvature, C3
    theta3: float = 0.95
    # Joint limitation curvature, C3
    beta3: float = 0.98
    # C4 quantum efficiency
    alpha4: float = 0.04

    # ----- BVOC emission factors (MEGAN 2.1 defaults, placeholder) -----
    # Will be overridden from parameter files when BVOC is enabled.


def get_constants() -> PhysicalConstants:
    """Return the default IBIS physical constants."""
    return PhysicalConstants()
