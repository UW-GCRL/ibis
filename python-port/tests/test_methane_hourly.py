"""Hourly wetland CH4 (docs/METHANE.md 9): the switch is off by default (the month-end snapshot path is untouched), and the
month step from the mean of the hourly fluxes books the mean and the annual total correctly."""
import numpy as np, jax.numpy as jnp
from ibis.bgc.methane import MethaneParams, MethaneState, methane_step_from_mean, wetland_water_table, create_methane_state


def test_switch_is_off_by_default():
    from ibis.io.config_reader import PhysicsSection
    assert PhysicsSection().methane_hourly is False


def test_month_step_from_mean_books_the_mean_and_the_annual_total():
    npoi, nlay = 3, 6; p = MethaneParams()
    wsoi = jnp.full((npoi, nlay), 0.9); wisoi = jnp.zeros((npoi, nlay)); hsoi = jnp.array([0.10, 0.25, 0.25, 0.40, 1.0, 2.0])
    m0 = create_methane_state(npoi); flux = jnp.array([0.0, 1e-5, 2e-5])
    m1 = methane_step_from_mean(m0, flux, wsoi, wisoi, hsoi, 30, p)
    assert np.allclose(np.asarray(m1.flux), np.asarray(flux)) and np.allclose(np.asarray(m1.annual), np.asarray(flux) * 30)
    m2 = methane_step_from_mean(m1, flux, wsoi, wisoi, hsoi, 31, p)
    assert np.allclose(np.asarray(m2.annual), np.asarray(flux) * 61)
    assert np.allclose(np.asarray(m2.zwt), np.asarray(wetland_water_table(wsoi, wisoi, hsoi, p)))


def test_day_loop_accepts_the_hourly_argument_in_its_signature():
    import inspect
    from ibis.timestepping.loops_gswp3 import _run_day_inner, run_month_gswp3
    assert 'ch4' in inspect.signature(_run_day_inner).parameters
    assert 'methane_hourly' in inspect.signature(run_month_gswp3).parameters
