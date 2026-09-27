"""The weather generator's latitude (DEFECTS_FOR_REVIEW.md "solar latitude", 2026-09-27).

SimParams.latitude is in radians (build_all_params; daylength_hours reads it so). `diurnal` converts from DEGREES, as
F77's `xlat = latscale * pi / 180` does. Both day loops must convert at the call, otherwise every cell's solar
geometry is that of the equator: 11-13 h days all year and the noon sun overhead in December."""
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]


def _daylight_hours(lat_value, jday):
    from ibis.physics.weather import diurnal, DailyWeather
    n = 1; z = jnp.zeros(n)
    dw = DailyWeather(td=jnp.full(n, 290.0), tmax=jnp.full(n, 295.0), tmin=jnp.full(n, 285.0), cloud=jnp.full(n, 0.3), precip=z,
                      psurf=jnp.full(n, 101325.0), qd=jnp.full(n, 0.008), ud=jnp.full(n, 3.0), iwet=jnp.zeros(n, jnp.int32), xstore=jnp.zeros((n, 3)))
    key = jax.random.PRNGKey(0); lat = jnp.full(n, lat_value)
    cz = np.array([float(diurnal(h * 3600.0, jnp.int32(jday), dw, lat, z, z, z, key, 3600.0)[1].coszen[0]) for h in range(24)])
    return int((cz > 0).sum()), cz[12]


def test_diurnal_takes_degrees_and_the_parameters_hold_radians():
    from ibis.io.physics_params import build_all_params
    from ibis.timestepping.loops_gswp3 import latitude_degrees
    sp, _, _ = build_all_params(2, np.deg2rad(np.array([60.0, 42.5], np.float32)), None)
    np.testing.assert_allclose(np.asarray(sp.latitude), np.deg2rad([60.0, 42.5]), rtol=1e-6)     # radians stored
    np.testing.assert_allclose(np.asarray(latitude_degrees(sp)), [60.0, 42.5], rtol=1e-5)          # degrees for diurnal
    h_jun, noon_jun = _daylight_hours(60.0, 172); h_dec, noon_dec = _daylight_hours(60.0, 355)
    assert 18 <= h_jun <= 20 and 4 <= h_dec <= 7, (h_jun, h_dec)                                   # 60 N in degrees: real seasons
    assert noon_jun > 0.75 and noon_dec < 0.15
    h_bug, noon_bug = _daylight_hours(float(np.deg2rad(60.0)), 355)
    assert 10 <= h_bug <= 13 and noon_bug > 0.85, (h_bug, noon_bug)                                 # the radians value read as degrees: the equator


def test_both_day_loops_convert_at_the_call():
    for f, needle in (("ibis/timestepping/loops_gswp3.py", "time, jday, dw, latitude_degrees(params),"),
                      ("ibis/timestepping/loops.py", "time, jday, daily_weather, params.latitude * (180.0 / jnp.pi),")):
        src = (ROOT / f).read_text()
        assert needle in src, (f, needle)
        assert "diurnal(\n            time, jday, dw, params.latitude,\n" not in src and "time, jday, daily_weather, params.latitude,\n" not in src, f
