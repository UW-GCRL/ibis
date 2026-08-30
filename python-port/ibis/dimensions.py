"""Grid and state-space dimension configuration for IBIS-JAX.

JAX equivalent of mod_dimensions.f90.  Unlike Fortran compile-time
parameters, these are runtime configuration passed as a frozen dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridConfig:
    """Grid and state dimensions.

    All fields are static integers/floats used to size arrays.
    Not a JAX pytree — these are Python-side metadata.
    """

    # Horizontal grid
    nlon: int = 720
    nlat: int = 360
    npoi: int = 1718       # Number of active land points
    xres: float = 0.50     # Longitude resolution (degrees)
    yres: float = 0.50     # Latitude resolution (degrees)

    # State-space dimensions
    nband: int = 2         # Solar radiation wavebands (VIS, NIR)
    nsoilay: int = 6       # Number of soil layers
    #: Thermal-only bedrock layers below the soil column. 0 reproduces the
    #: Fortran configuration (zero-flux at 4 m); 3 gives the 12 m column that
    #: keeps the annual wave from reflecting into the deepest soil node.
    #: See ibis/physics/bedrock.py.
    nbedrock: int = 0
    nsnolay: int = 3       # Number of snow layers
    npft: int = 12         # Number of plant functional types


# Predefined grid configurations
SOUTH_ASIA = GridConfig()

GLOBAL_HALF_DEG = GridConfig(
    nlon=720, nlat=360, npoi=58920, xres=0.5, yres=0.5,
)

GLOBAL_1DEG = GridConfig(
    nlon=360, nlat=180, npoi=14545, xres=1.0, yres=1.0,
)

GLOBAL_2DEG = GridConfig(
    nlon=180, nlat=90, npoi=3728, xres=2.0, yres=2.0,
)

TEST_10DEG = GridConfig(
    nlon=36, nlat=18, npoi=147, xres=10.0, yres=10.0,
)
