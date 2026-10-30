"""Hourly wetland CH4 and the monthly soil-state means under REAL forcing (docs/METHANE.md 9-10).

One month (January 2001) of the production loop on the three cells of test_forcing_invariants, run twice from the same
state: (a) the default path with the month-end CH4 snapshot; (b) `methane_hourly` + `soil_means`. Checks:
  1. the accumulators only read the state: every leaf of the model state except the wetland diagnostics agrees
     between (a) and (b) to float32 rounding (bit for bit when run alone; see the comment at the assertion);
  2. the monthly soil means are physical (temperature in range, water and ice fractions in [0, 1]) and differ from the
     month-end state somewhere (they are means, not copies of the last step);
  3. the hourly CH4 month mean is finite and non-negative, and the monthly flux is booked as the mean.
Skipped where the GSWP3 forcing or the 2010 state is not available. ~3 min on CPU.
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


@pytest.fixture
def production_f32():
    import jax
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


def _setup():
    import xarray as xr, jax, jax.numpy as jnp
    from pathlib import Path
    from ibis.dimensions import GridConfig
    from ibis.io.restart import load_state
    from ibis.state.model_state import create_ibis_state
    from ibis.io.physics_params import build_all_params, load_physics_params
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.constants import PhysicalConstants
    from ibis.timestepping.loops_gswp3 import _cast_tree_f32
    m = np.array(xr.open_dataset(MASK).landfrac.values) > 0
    fl = np.where(m.ravel())[0]; posmap = {g: p for p, g in enumerate(fl)}
    lat = np.asarray(xr.open_dataset(MASK).lat.values)
    i = np.array([a for a, _ in CELLS.values()], np.int32); j = np.array([b for _, b in CELLS.values()], np.int32)
    sel = np.array([posmap[a * 720 + b] for a, b in zip(i, j)])
    full, _ = load_state(STATE, create_ibis_state(GridConfig(npoi=NPOI_FULL, nbedrock=3)))
    cut = lambda x: x[sel] if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL else x
    st = _cast_tree_f32(jax.tree.map(cut, full))
    lat_rad = np.deg2rad(lat[i]).astype(np.float32)
    root = Path(__file__).resolve().parents[1]
    sp, _, _ = build_all_params(len(sel), lat_rad, load_physics_params(str(root / "configs" / "params" / "default.yaml")))
    return st, sp, GSWP3Reader(GSWP3), PhysicalConstants(), i, j, lat_rad


def test_accumulators_leave_the_state_alone_and_give_physical_means(production_f32):
    import jax, jax.numpy as jnp
    from ibis.timestepping.loops_gswp3 import run_month_gswp3
    from ibis.bgc.methane import MethaneParams
    st0, sp, rd, consts, i, j, lat_rad = _setup()
    f_in = np.ones(len(i), np.float32); mp = MethaneParams(); key = jax.random.PRNGKey(0)
    a, _, dd_a = run_month_gswp3(st0, consts, sp, rd, i, j, lat_rad, 2001, 0, key, 3600.0, leap=False, nband=2, nsoilcall=1,
                                 f_inund=f_in, methane_params=mp)
    b, _, dd_b = run_month_gswp3(st0, consts, sp, rd, i, j, lat_rad, 2001, 0, key, 3600.0, leap=False, nband=2, nsoilcall=1,
                                 f_inund=f_in, methane_params=mp, methane_hourly=True, soil_means=True)
    # 1. the state outside the wetland diagnostics is bit-identical
    import equinox as eqx
    a_ = eqx.tree_at(lambda s: s.wetland, a, None, is_leaf=lambda x: x is None)
    b_ = eqx.tree_at(lambda s: s.wetland, b, None, is_leaf=lambda x: x is None)
    fa = jax.tree_util.tree_leaves(a_); fb = jax.tree_util.tree_leaves(b_)
    assert len(fa) == len(fb)
    # Equal to float32 rounding, not bit for bit: the extra accumulator outputs can change how XLA fuses the step's
    # arithmetic (bit-identical when the test runs alone; ~1e-6 relative inside the full suite, CALIBRATION.md 82).
    # A feedback of the accumulators into the state would show at the level of kelvins and whole water fractions.
    # The measure is each field's difference relative to that field's own scale (its largest magnitude): rounding drift
    # over a month stays below 1e-3 of the scale, a feedback of the accumulators would be percent-level or more.
    def scaled_diff(x, y):
        x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); ok = np.isfinite(x) & np.isfinite(y)
        assert np.array_equal(np.isfinite(x), np.isfinite(y))
        scale = max(float(np.max(np.abs(x[ok]))) if ok.any() else 0.0, 1e-30)
        return float(np.max(np.abs(x[ok] - y[ok]))) / scale if ok.any() else 0.0
    # hydrology.gadjust (the last step's water change from the bound clamp in wadjust, ztot1 - ztot0 over the step) is
    # float32 rounding of a column total of hundreds of kg/m2 when nothing is clamped: its own scale is that rounding, so
    # it is compared in water per step instead, below 1e-3 kg/m2.
    worst = 0.0; where = ''
    pa = jax.tree_util.tree_flatten_with_path(a_)[0]
    for (path, x), y in zip(pa, fb):
        if jax.tree_util.keystr(path).endswith('.gadjust'):
            dw = float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)))) * 3600.0
            assert dw < 1e-3, f'gadjust differs by {dw:.2e} kg/m2 per step'
        elif isinstance(x, jnp.ndarray) and np.issubdtype(np.asarray(x).dtype, np.floating):
            d = scaled_diff(x, y)
            if d > worst: worst, where = d, jax.tree_util.keystr(path) + f' a={np.asarray(x).ravel()[:6]} b={np.asarray(y).ravel()[:6]}'
        elif isinstance(x, jnp.ndarray):
            assert np.array_equal(np.asarray(x), np.asarray(y)), jax.tree_util.keystr(path)
    assert len(dd_a) == 3 and len(dd_b) == 4
    for k_, (x, y) in enumerate(zip(dd_a, dd_b[:3])):
        d = scaled_diff(x, y)
        if d > worst: worst, where = d, f'daily output {k_}'
    assert worst < 1e-3, f'a field differs by {worst:.2e} of its scale with the accumulators on: {where}'
    # 2. the soil means are physical and are means
    sm = {k: np.asarray(v) for k, v in dd_b[3].items()}
    assert sm['tsoi_mean'].shape == np.asarray(b.soil.tsoi).shape and sm['tg_mean'].shape == np.asarray(b.soil.tg).shape
    assert np.all((sm['tsoi_mean'] > 200.0) & (sm['tsoi_mean'] < 330.0)) and np.all((sm['tg_mean'] > 200.0) & (sm['tg_mean'] < 340.0))
    for k in ('wsoi_mean', 'wisoi_mean'):
        assert np.all((sm[k] >= 0.0) & (sm[k] <= 1.0 + 1e-6)), k
    assert np.max(np.abs(sm['tg_mean'] - np.asarray(b.soil.tg))) > 0.1        # a mean, not the month-end state
    assert np.max(np.abs(sm['tsoi_mean'][:, 0] - np.asarray(b.soil.tsoi)[:, 0])) > 0.01
    # 3. the hourly CH4 mean is finite, non-negative, and booked as the month's flux
    fh = np.asarray(b.wetland.flux)
    assert np.all(np.isfinite(fh)) and np.all(fh >= 0.0)
    assert np.allclose(np.asarray(b.wetland.annual), fh * 31, rtol=1e-6, atol=1e-12)
