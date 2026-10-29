"""Dynamic wetland area (docs/METHANE.md 7): the pure fraction function, and the switch's default state."""
import numpy as np
from ibis.bgc.wetland_area import root_zone_water, dynamic_fraction, ROOT_LAYERS, T_FREEZE


def test_root_zone_water_counts_the_top_metre_and_skips_frozen_layers():
    hsoi = np.array([0.10, 0.25, 0.25, 0.40, 1.0, 2.0, 0.0])
    poros = np.full((2, 6), 0.5); wsoi = np.full((2, 6), 0.8)
    w = root_zone_water(wsoi, poros, hsoi)
    assert np.allclose(w, 0.5 * 0.8 * hsoi[:ROOT_LAYERS].sum())               # 0.4 m of the 1 m column
    tsoi = np.full((2, 6), 280.0); tsoi[1, 0] = T_FREEZE - 1.0                # cell 1: frozen top layer
    w2 = root_zone_water(wsoi, poros, hsoi, tsoi)
    assert np.isclose(w2[0], w[0]) and np.isclose(w2[1], w[1] - 0.5 * 0.8 * 0.10)


def test_fraction_is_bounded_climatology_at_zero_beta_and_responds_to_the_anomaly():
    f_clim = np.array([0.3, 0.3, 0.0, 0.5]); mu = np.array([0.2, 0.2, 0.2, 0.2]); sd = np.array([0.05, 0.05, 0.05, 0.0])
    fmin = np.array([0.1, 0.1, 0.0, 0.4]); fmax = np.array([0.6, 0.6, 0.0, 0.5])
    W = np.array([0.3, 0.1, 0.3, 0.3])                                          # +2 sd, -2 sd, (no wetland), (sd 0)
    f0 = dynamic_fraction(f_clim, W, mu, sd, 0.0, fmin, fmax)
    assert np.allclose(f0, np.clip(f_clim, fmin, fmax))
    f = dynamic_fraction(f_clim, W, mu, sd, 0.1, fmin, fmax)
    assert np.isclose(f[0], 0.3 * 1.2) and np.isclose(f[1], 0.3 * 0.8) and f[2] == 0.0 and np.isclose(f[3], 0.5)
    fb = dynamic_fraction(f_clim, W, mu, sd, 5.0, fmin, fmax)                  # a huge beta stays inside the cell's range
    assert np.isclose(fb[0], 0.6) and np.isclose(fb[1], 0.1)
    ff = dynamic_fraction(f_clim, W, mu, sd, 0.1, fmin, fmax, frozen=np.array([True, False, False, False]))
    assert np.isclose(ff[0], f0[0]) and np.isclose(ff[1], f[1])                 # frozen: the anomaly is 0, the climatology stays
    assert (dynamic_fraction(f_clim, W, mu, sd, np.array([0.1, 0.0, 0.1, 0.1]), fmin, fmax)[1] == np.clip(0.3, 0.1, 0.6))


def test_switch_is_off_by_default():
    from ibis.io.config_reader import PhysicsSection, ForcingSection
    assert PhysicsSection().wetland_dynamic is False and ForcingSection().wetland_clim_file == ''


def test_wetland_source_default_is_climatology_and_record_helper():
    """docs/METHANE.md 8: the default source is the climatology (bit-identical to before); the observed source is
    only taken for calendar years with a WAD2M field."""
    from ibis.io.config_reader import PhysicsSection
    from ibis.io.wad2m import wad2m_in_record, WAD2M_YEARS
    assert PhysicsSection().wetland_source == 'climatology'
    y0, y1 = WAD2M_YEARS
    assert wad2m_in_record(y0) and wad2m_in_record(y1) and wad2m_in_record(2010)
    assert not wad2m_in_record(y0 - 1) and not wad2m_in_record(y1 + 1) and not wad2m_in_record(1901)
