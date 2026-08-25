"""M1: inundated fraction and water table depth. See docs/METHANE.md 4.1."""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest

from ibis.bgc.methane import (
    MethaneParams, anaerobic_fraction, layer_saturation, water_table_depth,
    wetland_flux_area,
)

WAD2M = ("/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/"
         "WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc")
HSOI = jnp.asarray([0.05, 0.10, 0.20, 0.30, 0.50, 1.00])   # 2.15 m column


# --- water table -----------------------------------------------------------


def test_saturated_column_has_the_water_table_at_the_surface():
    n = 4
    wsoi = jnp.ones((n, 6))
    zwt = water_table_depth(wsoi, jnp.zeros((n, 6)), HSOI)
    assert np.allclose(np.asarray(zwt), 0.0)


def test_dry_column_puts_it_at_the_bottom_of_the_column():
    n = 3
    zwt = water_table_depth(jnp.zeros((n, 6)), jnp.zeros((n, 6)), HSOI)
    assert np.allclose(np.asarray(zwt), float(jnp.sum(HSOI)))


def test_ice_does_not_count_as_water():
    """A frozen column must NOT read as saturated.

    `wsoi` is the liquid share of the ice-free pore space, so wsoi=1 with
    wisoi=1 is fully frozen, not fully wet. Treating it as wet would put a
    water table at the surface across the whole winter Arctic and make it the
    model's largest methane source.
    """
    n = 2
    frozen = water_table_depth(jnp.ones((n, 6)), jnp.ones((n, 6)), HSOI)
    assert np.allclose(np.asarray(frozen), float(jnp.sum(HSOI)))


def test_depth_increases_as_the_column_dries():
    n = 1
    prev = -1.0
    for s in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        z = float(water_table_depth(jnp.full((n, 6), s), jnp.zeros((n, 6)), HSOI)[0])
        assert z > prev
        prev = z


def test_sat_crit_means_nearly_saturated_still_counts_as_wet():
    n = 1
    p = MethaneParams(sat_crit=0.95)
    z = float(water_table_depth(jnp.full((n, 6), 0.95), jnp.zeros((n, 6)), HSOI, p)[0])
    assert z == pytest.approx(0.0, abs=1e-6)


def test_depth_is_clipped_to_z_max():
    p = MethaneParams(z_max=0.5)
    z = water_table_depth(jnp.zeros((2, 6)), jnp.zeros((2, 6)), HSOI, p)
    assert float(jnp.max(z)) <= 0.5


def test_layer_saturation_stays_in_range():
    rng = np.random.default_rng(0)
    s = layer_saturation(jnp.asarray(rng.uniform(0, 1, (50, 6))),
                         jnp.asarray(rng.uniform(0, 1, (50, 6))))
    a = np.asarray(s)
    assert a.min() >= 0.0 and a.max() <= 1.0


# --- anaerobic fraction ----------------------------------------------------


def test_water_table_at_surface_makes_every_layer_anaerobic():
    a = anaerobic_fraction(jnp.zeros(3), HSOI)
    assert np.allclose(np.asarray(a), 1.0)


def test_water_table_below_the_column_makes_none_anaerobic():
    a = anaerobic_fraction(jnp.full((3,), 10.0), HSOI)
    assert np.allclose(np.asarray(a), 0.0)


def test_a_layer_straddling_the_water_table_is_partly_anaerobic():
    """Layer 2 spans 0.15-0.35 m; a table at 0.25 m leaves it half anaerobic."""
    a = np.asarray(anaerobic_fraction(jnp.asarray([0.25]), HSOI))[0]
    assert a[0] == pytest.approx(0.0)      # 0.00-0.05, above the table
    assert a[1] == pytest.approx(0.0)      # 0.05-0.15, above
    assert a[2] == pytest.approx(0.5)      # 0.15-0.35, straddling
    assert a[3] == pytest.approx(1.0)      # 0.35-0.65, below


def test_anaerobic_fraction_is_monotone_in_depth():
    tot = [float(jnp.sum(anaerobic_fraction(jnp.asarray([z]), HSOI)))
           for z in (0.0, 0.2, 0.5, 1.0, 2.0)]
    assert all(b <= a for a, b in zip(tot, tot[1:]))


# --- flux area -------------------------------------------------------------


def test_flux_area_respects_a_fixed_floor():
    f = wetland_flux_area(jnp.asarray([0.0, 0.3, 0.9]), fixed_fraction=0.1)
    assert np.allclose(np.asarray(f), [0.1, 0.3, 0.9])


def test_flux_area_is_bounded():
    f = wetland_flux_area(jnp.asarray([-0.5, 0.5, 2.0]))
    a = np.asarray(f)
    assert a.min() >= 0.0 and a.max() <= 1.0


# --- WAD2M reader ----------------------------------------------------------

pytestmark_wad2m = pytest.mark.skipif(
    not os.path.exists(WAD2M), reason="WAD2M not on this machine")


@pytestmark_wad2m
def test_wad2m_geography_is_the_right_way_up():
    """The check that would have caught Siberia's wetlands in the Sahara.

    Regional means, not single cells. A single point on a region boundary sits
    on a knife edge -- 57.0 N is exactly between WAD2M's 56.75 and 57.25 cell
    centres -- and which side it lands on says nothing about orientation.
    """
    from ibis.io.wad2m import read_wad2m_month

    mlat = np.arange(-90.0, 90.0, 0.5)
    mlon = np.arange(-180.0, 180.0, 0.5)

    def box(la0, la1, lo0, lo1):
        la = np.arange(la0, la1, 0.5)
        lo = np.arange(lo0, lo1, 0.5)
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        li = np.rint((LA.ravel() - mlat[0]) / 0.5).astype(int)
        lj = np.rint((LO.ravel() - mlon[0]) / 0.5).astype(int)
        return float(read_wad2m_month(WAD2M, 2010, 7, li, lj, mlat, mlon).mean())

    hudson = box(51, 59, -96, -80)
    siberia = box(55, 65, 60, 90)
    congo = box(-4, 2, 15, 25)
    sahara = box(20, 30, 5, 25)
    plains = box(35, 45, -105, -95)

    for wet, name in ((hudson, "hudson"), (siberia, "siberia"), (congo, "congo")):
        assert wet > 0.05, f"{name} should be wet, got {wet}"
        assert wet > 20 * max(sahara, plains), f"{name} vs dry: {wet}"
    assert sahara < 0.01 and plains < 0.01


@pytestmark_wad2m
def test_wad2m_values_are_fractions():
    from ibis.io.wad2m import read_wad2m_month

    mlat = np.arange(-90.0, 90.0, 0.5)
    mlon = np.arange(-180.0, 180.0, 0.5)
    li = np.arange(0, 360, 7)
    lo = np.arange(0, 720, 14)[:len(li)]
    fw = read_wad2m_month(WAD2M, 2005, 6, li, lo, mlat, mlon)
    assert fw.min() >= 0.0 and fw.max() <= 1.0
    assert not np.isnan(fw).any()          # ocean is masked to 0, not NaN


@pytestmark_wad2m
def test_wad2m_rejects_a_mismatched_model_grid():
    from ibis.io.wad2m import read_wad2m_month

    with pytest.raises(ValueError, match="does not match|regridding"):
        read_wad2m_month(WAD2M, 2005, 6, np.array([0]), np.array([0]),
                         np.arange(-90.0, 90.0, 1.0), np.arange(-180.0, 180.0, 1.0))


@pytestmark_wad2m
def test_pre_satellite_years_must_be_asked_for_explicitly():
    """A 1901-2000 spin-up has no satellite record; silence would be wrong."""
    from ibis.io.wad2m import read_wad2m_month

    mlat = np.arange(-90.0, 90.0, 0.5)
    mlon = np.arange(-180.0, 180.0, 0.5)
    li, lo = np.array([200]), np.array([400])
    with pytest.raises(ValueError, match="WAD2M covers"):
        read_wad2m_month(WAD2M, 1950, 6, li, lo, mlat, mlon,
                         clamp_to_range=False)
    ok = read_wad2m_month(WAD2M, 1950, 6, li, lo, mlat, mlon,
                          clamp_to_range=True)
    assert 0.0 <= float(ok[0]) <= 1.0


@pytestmark_wad2m
def test_wad2m_month_index_tracks_the_seasonal_cycle():
    """Northern wetlands peak in boreal summer, not January."""
    from ibis.io.wad2m import read_wad2m_month

    mlat = np.arange(-90.0, 90.0, 0.5)
    mlon = np.arange(-180.0, 180.0, 0.5)
    li = np.arange(300, 340)                       # 60-80 N
    lo = np.full_like(li, int(round((75.0 + 180.0) / 0.5)))
    jan = read_wad2m_month(WAD2M, 2010, 1, li, lo, mlat, mlon).mean()
    jul = read_wad2m_month(WAD2M, 2010, 7, li, lo, mlat, mlon).mean()
    assert jul > jan
