"""Seasonal invariants under REAL forcing at the REAL cadence (review 2026-09-22, point 5).

The two forcing defects of September 2026 -- a 240-day phenology clock and sea-level
pressure on every cell -- passed every unit and lockstep test because those tests never
ran a season of real GSWP3 forcing through the production loop. This test does: twelve
months of 2001 on three cells (Himalayan slope at ~580 hPa, sea-level Amazon, Kazakh
steppe) from the canonical 2010 state, default parameters, and checks two things a
correct driver must satisfy:
  1. the pressure the physics sees is the forcing pressure (not the creation value);
  2. the ten-day running mean of air temperature carries the annual cycle (range >= 0.8x
     the range of monthly air temperature; the 240-day clock gave 0.24x).
Skipped where the GSWP3 forcing or the 2010 state is not available.  ~2-3 min on CPU.
"""
import os
import numpy as np
import pytest

GSWP3 = "/shared/nas2/Min/GSWP3"
STATE = "/shared/nas2/Hangkai/ibis/verify_outputs/v11_transient/restart/ibis_restart_2010.npz"
MASK = "/shared/nas2/Min/HWSD/surta.nc"
NPOI_FULL = 95333
CELLS = {"Himalaya 31.25N 78.75E": (117, 517), "Amazon 3.25S 60.25W": (186, 239), "Kazakh 50.25N 70.25E": (79, 500)}

pytestmark = pytest.mark.skipif(not (os.path.isdir(GSWP3) and os.path.exists(STATE) and os.path.exists(MASK)),
                                reason="GSWP3 forcing / 2010 state not available on this machine")


def _cells():
    import xarray as xr
    m = np.array(xr.open_dataset(MASK).landfrac.values) > 0
    fl = np.where(m.ravel())[0]; posmap = {g: p for p, g in enumerate(fl)}
    lat = np.asarray(xr.open_dataset(MASK).lat.values); lon = np.asarray(xr.open_dataset(MASK).lon.values)
    ij = list(CELLS.values())
    for (i, j) in ij:
        assert i * 720 + j in posmap, (i, j)
    return np.array([posmap[i * 720 + j] for i, j in ij]), np.array([i for i, _ in ij], np.int32), np.array([j for _, j in ij], np.int32), lat


def test_year_of_real_forcing_carries_pressure_and_a_ten_day_clock():
    import jax, jax.numpy as jnp, equinox as eqx
    from ibis.dimensions import GridConfig
    from ibis.io.restart import load_state
    from ibis.state.model_state import create_ibis_state
    from ibis.io.physics_params import build_all_params, load_physics_params
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.constants import PhysicalConstants
    from ibis.timestepping.loops_gswp3 import run_month_gswp3, _cast_tree_f32
    from pathlib import Path
    sel, i, j, lat = _cells()
    full, _ = load_state(STATE, create_ibis_state(GridConfig(npoi=NPOI_FULL, nbedrock=3)))
    cut = lambda x: x[sel] if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL else x
    st = _cast_tree_f32(jax.tree.map(cut, full))
    lat_rad = np.deg2rad(lat[i]).astype(np.float32)
    root = Path(__file__).resolve().parents[1]
    sp, pft, _ = build_all_params(len(sel), lat_rad, load_physics_params(str(root / "configs" / "params" / "default.yaml")))
    rd = GSWP3Reader(GSWP3); consts = PhysicalConstants(); key = jax.random.PRNGKey(0)
    assert np.allclose(np.asarray(st.atm.psurf), 101325.0)            # the creation value, before any forcing
    ps_month = []; ta_month = []; a10 = []
    for im in range(12):
        raw = rd.read_month(2001, im, i, j)
        ps_month.append(np.nanmean(raw["ps"], axis=0)); ta_month.append(np.nanmean(raw["tas"], axis=0))
        st, key, _ = run_month_gswp3(st, consts, sp, rd, i, j, lat_rad, 2001, im, key, 3600.0, leap=False, nband=2, nsoilcall=1)
        a10.append(np.asarray(st.summary.a10td).copy())
        if im == 0:
            got = np.asarray(st.atm.psurf)
            # 1. the physics saw the forcing pressure: within 3 % of the month's mean ps, and NOT sea level on the slope
            assert np.all(np.abs(got / ps_month[0] - 1.0) < 0.03), (got, ps_month[0])
            assert got[0] < 70000.0, got                                   # Himalayan cell ~ 580 hPa
            assert 95000.0 < got[1] < 103000.0, got                          # Amazon near sea level
    a10 = np.array(a10); ta = np.array(ta_month)
    # 2. the ten-day clock carries the season: monthly a10td range vs monthly air-temperature range, cell by cell
    ratio = (a10.max(0) - a10.min(0)) / np.maximum(ta.max(0) - ta.min(0), 1.0)
    assert np.all(ratio > 0.8), ratio                                       # exp(-1/240) per day gave ~0.24
    assert np.all(np.abs(a10[-1] - ta[-1]) < 6.0), (a10[-1], ta[-1])          # December mean within a few K of the month's air temperature
