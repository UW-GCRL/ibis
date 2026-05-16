"""Shared test fixtures for IBIS-JAX."""

import pytest
from ibis.dimensions import GridConfig
from ibis.constants import PhysicalConstants


# Small test grid (avoids allocating 1718-point arrays in unit tests)
TEST_GRID = GridConfig(
    nlon=36, nlat=18, npoi=4,
    xres=10.0, yres=10.0,
    nband=2, nsoilay=6, nsnolay=3, npft=12,
)


@pytest.fixture
def grid():
    """Small test grid."""
    return TEST_GRID


@pytest.fixture
def consts():
    """Default physical constants."""
    return PhysicalConstants()
