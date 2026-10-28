"""CALIBRATION.md 72: the slow and passive SOM pools' own temperature sensitivity (som_lt_e0, anchored at som_t_ref).
Off by default and bit-exact; on, the SOM outflows scale by the ratio and carbon is conserved."""
from __future__ import annotations
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ibis.io.parameters import default_bgc_params
from ibis.physics.biogeochem import soilbgc, som_climate_ratio
from ibis.state.carbon import create_carbon_state
from ibis.state.summary import create_summary_state
from ibis.state.vegetation import create_vegetation_state


def test_ratio_is_one_when_off_and_at_the_anchor_and_follows_lloyd_taylor_when_on():
    T = np.array([275.0, 283.0, 298.16, 303.0]); tconst = np.full(4, 369.0)
    np.testing.assert_allclose(np.asarray(som_climate_ratio(T, tconst, 0.0, 298.16)), 1.0)          # off
    r = np.asarray(som_climate_ratio(T, tconst, 520.0, 298.16))
    assert abs(r[2] - 1.0) < 1e-5                                                                   # anchor: unchanged
    inv = lambda t: 1.0 / (t - 227.13)
    f_lit = lambda t: np.minimum(np.exp(369.0 * (inv(288.16) - inv(t))), 10.0)
    want = np.exp(520.0 * (inv(298.16) - inv(T))) * f_lit(298.16) / f_lit(T)
    np.testing.assert_allclose(r, want, rtol=1e-4)
    assert r[0] < r[1] < 1.0 < r[3]                                                                  # colder: slower SOM
    assert 0.25 < r[0] < 0.5                                                                          # boreal soils: a third to a half


def _state(npoi=3):
    carbon = create_carbon_state(npoi)
    carbon = eqx.tree_at(lambda c: (c.csoislon, c.csoislop, c.csoipas, c.clitlm, c.decompl, c.decomps), carbon,
                         (jnp.full((npoi,), 5.0), jnp.full((npoi,), 3.0), jnp.full((npoi,), 10.0), jnp.full((npoi,), 0.2),
                          jnp.full((npoi,), 1.0), jnp.full((npoi,), 1.0)))
    veg = create_vegetation_state(npoi, 12, 6); summary = create_summary_state(npoi, 12)
    hsoi = jnp.asarray([0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 0.0]); sand = jnp.full((npoi, 6), 40.0); clay = jnp.full((npoi, 6), 20.0)
    return carbon, veg, summary, hsoi, sand, clay


def test_default_off_is_bit_exact_and_the_factor_scales_only_the_som_pools():
    bgc = default_bgc_params(); assert bgc.som_lt_e0 == 0.0 and abs(bgc.som_t_ref - 298.16) < 1e-6
    carbon, veg, summary, hsoi, sand, clay = _state()
    c0, v0, _ = soilbgc(carbon, veg, summary, hsoi, sand, clay, bgc, 365)
    c1, v1, _ = soilbgc(carbon, veg, summary, hsoi, sand, clay, bgc, 365, som_factor=jnp.ones((3,)))
    for k in ('csoislon', 'csoislop', 'csoipas', 'clitlm', 'totcsoi'):
        np.testing.assert_array_equal(np.asarray(getattr(c0, k)), np.asarray(getattr(c1, k)))
    np.testing.assert_array_equal(np.asarray(v0.tco2mic), np.asarray(v1.tco2mic))
    ch, vh, _ = soilbgc(carbon, veg, summary, hsoi, sand, clay, bgc, 365, som_factor=jnp.full((3,), 0.5))
    loss_full = float(carbon.csoislon[0] - c0.csoislon[0]); loss_half = float(carbon.csoislon[0] - ch.csoislon[0])
    assert loss_full > 0 and abs(loss_half - 0.5 * loss_full) < 1e-3 * loss_full                     # slow pool: half the outflow (float32)
    np.testing.assert_array_equal(np.asarray(c0.clitlm), np.asarray(ch.clitlm))                        # litter untouched
    # conservation: what the pools keep, the air does not get (kgC m-2 day-1 from mol CO2 m-2 s-1)
    tot = lambda c: float(np.asarray(c.totcsoi)[0])
    d_pool = tot(ch) - tot(c0); d_resp = float(np.asarray(v0.tco2mic)[0] - np.asarray(vh.tco2mic)[0]) * 12.0e-3 * 86400.0
    assert d_pool > 0 and abs(d_pool - d_resp) < 0.05 * d_pool


def test_loader_and_wiring():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert "'som_lt_e0'" in (root / "ibis" / "io" / "physics_params.py").read_text()
    lp = (root / "ibis" / "timestepping" / "loops_gswp3.py").read_text()
    assert "som_factor=som_factor" in lp and "som_climate_ratio" in lp


def test_som_turnover_factor_per_zone_through_the_types():
    """CALIBRATION.md 72: a zone's types may carry som_k_scale; apply_types makes it a per-cell array on the bgc
    parameters (base 1 elsewhere), and the loop multiplies the SOM climate factor by it."""
    from ibis.io import pft_types as P
    from ibis.io.physics_params import build_all_params
    params, pp, _ = build_all_params(3, np.zeros(3))
    z = np.array([P.ZI['D_subarctic'], P.ZI['A_humid'], P.ZI['D_temperate']])
    over = P.parse_type_overrides({'T17': {'som_k_scale': 0.35}, 'T18': {'som_k_scale': 0.35}, 'T16': {'som_k_scale': 0.5}})
    p2, _ = P.apply_types(params, pp, z, over)
    np.testing.assert_allclose(np.asarray(p2.bgc_params.som_k_scale), [0.35, 1.0, 0.5], rtol=1e-6)
    assert np.ndim(np.asarray(P.apply_types(params, pp, z, {})[0].bgc_params.som_k_scale)) == 0
    from pathlib import Path
    lp = (Path(__file__).resolve().parents[1] / "ibis" / "timestepping" / "loops_gswp3.py").read_text()
    assert "params.bgc_params.som_k_scale" in lp
    assert "'som_k_scale'" in (Path(__file__).resolve().parents[1] / "ibis" / "io" / "physics_params.py").read_text()
