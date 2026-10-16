"""Column water balance of one timestep, and the runoff/drainage diagnostics (CALIBRATION.md 45.2, 2026-10-16).

The port added the squeeze-out of a saturated column (fsqueez) to the surface runoff diagnostic although soilh2o
already returns it to the puddle -- the F77 `soil.f` keeps it in the puddle (line 845) and never in grunof (lines
256-280) -- so reported runoff counted the same water every hour it re-infiltrated (2.2 m/yr of 'runoff' on 1.3 m of
rain in the wetland cells). And gdrain was hard-wired to zero, while F77 reports the flow out of the bottom layer
(`gdrain = wflo(nsoilay+1)`, line 387). The physics was right; the diagnostics were not. This test closes the water
balance of the soil column and puddle with the stored diagnostics:

    d(soil water + puddle) = (rain reaching the soil - soil evaporation - transpiration - runoff - drainage) dt
"""
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import pytest

from tests.test_integration import consts, params, state, NPOI, DTIME   # noqa: F401  (fixtures)
from ibis.timestepping.loops import timestep_body


def _column_water(st):
    s = st.soil
    rhow = 1000.0
    hs = jnp.asarray(s.hsoi)[:6]
    w = rhow * s.poros * hs[None, :] * ((1.0 - s.wisoi) * s.wsoi + s.wisoi)
    return np.asarray(w.sum(axis=1) + s.wpud + s.wipud, dtype=np.float64)


def _wet(state, wsoi, rain):
    soil = eqx.tree_at(lambda s: s.wsoi, state.soil, jnp.full_like(state.soil.wsoi, wsoi))
    atm = eqx.tree_at(lambda a: a.raina, state.atm, jnp.full_like(state.atm.raina, rain))
    return eqx.tree_at(lambda s: (s.soil, s.atm), state, (soil, atm))


@pytest.mark.parametrize('wsoi,rain', [(0.6, 0.0), (0.98, 2.0e-3), (1.0, 5.0e-3)])
def test_column_water_balance_closes_with_the_stored_fluxes(state, consts, params, wsoi, rain):
    st = _wet(state, wsoi, rain)
    w0 = _column_water(st)
    new = timestep_body(st, consts, params, DTIME)
    w1 = _column_water(new)
    h = new.hydrology
    into = np.asarray(h.raing_avg, np.float64)
    out = np.asarray(h.gsuvap + h.gtrans + h.grunof + h.gdrain, np.float64)
    resid = (w1 - w0) - (into - out) * DTIME
    scale = np.maximum(np.abs(into) + np.abs(out), 1e-6) * DTIME
    assert np.all(np.abs(resid) <= 0.02 * scale + 0.05), (resid, into * DTIME, out * DTIME, np.asarray(h.grunof) * DTIME, np.asarray(h.gdrain) * DTIME)


def test_drainage_is_reported_and_runoff_excludes_the_squeeze(state, consts, params):
    st = _wet(state, 1.0, 5.0e-3)
    new = timestep_body(st, consts, params, DTIME)
    assert float(np.max(np.asarray(new.hydrology.gdrain))) > 0.0      # a saturated column drains through the bottom
    assert np.all(np.asarray(new.hydrology.grunof) <= 5.0e-3 + 1e-9 + np.asarray(st.soil.wpud) / DTIME)   # no more than rain + standing puddle
