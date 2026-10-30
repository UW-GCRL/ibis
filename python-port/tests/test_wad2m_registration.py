"""WAD2M registration (DEFECTS_FOR_REVIEW "WAD2M registration", 2026-10-29): a 0.5-degree cell centre must map to the WAD2M
cell whose centre it is. np.rint on (lat + 90) / 0.5 rounds the exact .5 of every centre half to even and put every other
row and column one cell off (75 % of land points)."""
import os
import numpy as np
import pytest
from ibis.io.wad2m import wad2m_cell_index

WAD2M = "/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc"


def test_every_cell_centre_maps_to_its_own_cell():
    lat_c = -89.75 + 0.5 * np.arange(360); lon_c = -179.75 + 0.5 * np.arange(720)
    li, _ = wad2m_cell_index(lat_c, np.zeros(360)); _, lo = wad2m_cell_index(np.zeros(720), lon_c)
    assert np.array_equal(li, np.arange(360)) and np.array_equal(lo, np.arange(720))


def test_any_point_maps_to_the_cell_that_contains_it():
    rng = np.random.default_rng(0); lat = rng.uniform(-89.99, 89.99, 5000); lon = rng.uniform(-179.99, 179.99, 5000)
    li, lo = wad2m_cell_index(lat, lon)
    assert np.all((-90 + 0.5 * li <= lat) & (lat < -90 + 0.5 * (li + 1)))
    assert np.all((-180 + 0.5 * lo <= lon) & (lon < -180 + 0.5 * (lo + 1)))


@pytest.mark.skipif(not os.path.exists(WAD2M), reason="WAD2M not on this machine")
def test_the_file_cell_centre_equals_the_point():
    import xarray as xr
    ds = xr.open_dataset(WAD2M); wlat = np.asarray(ds["lat"].values); wlon = np.asarray(ds["lon"].values)
    lat_c = -89.75 + 0.5 * np.arange(360); lon_c = -179.75 + 0.5 * np.arange(720)
    li, _ = wad2m_cell_index(lat_c, np.zeros(360)); _, lo = wad2m_cell_index(np.zeros(720), lon_c)
    assert np.allclose(wlat[li], lat_c) and np.allclose(wlon[lo], lon_c)
