"""The production runner must use the VALIDATED methane configuration.

`run_global` never passed `methane_params` to `run_month_gswp3`, so every
production run silently used `MethaneParams()` -- the "global" configuration,
458 Tg/yr on the reference state -- while the configuration actually validated
in docs/METHANE.md 5q gives 182 Tg, inside the GMB 2025 range. The flag was
wired, the climatology was loaded, and the parameters were not.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from ibis.bgc.methane import MethaneParams, biome_methane_params


def test_biome_params_match_the_validated_values():
    """Guards the constants the 182 Tg result was obtained with."""
    lat = jnp.array([-70.0, -45.0, -10.0, 0.0, 20.0, 45.0, 70.0, 85.0])
    p = biome_methane_params(lat)
    r = np.asarray(p.r_ch4)
    t = np.asarray(p.t_ref)

    trop = (np.asarray(lat) > -30) & (np.asarray(lat) < 30)
    boreal = np.asarray(lat) >= 60
    temp = ~trop & ~boreal

    assert np.allclose(r[trop], 0.15) and np.allclose(t[trop], 299.6)
    assert np.allclose(r[temp], 0.32) and np.allclose(t[temp], 283.0)
    assert np.allclose(r[boreal], 0.18) and np.allclose(t[boreal], 277.9)
    # q10p is deliberately global -- the biome dependence lives in t_ref
    assert float(p.q10p) == float(MethaneParams().q10p)


def test_per_biome_t_ref_is_not_the_global_default():
    """The whole point of 5n: one tropical t_ref double-counts the Q10 term."""
    lat = jnp.array([-70.0, 0.0, 45.0])
    p = biome_methane_params(lat)
    assert not np.allclose(np.asarray(p.t_ref), float(MethaneParams().t_ref)), (
        "per-biome t_ref must differ from the global 295.15 K default")


def test_run_global_passes_methane_params():
    """The wiring itself -- this is what was missing."""
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "run_global.py").read_text()
    assert "biome_methane_params(" in src, (
        "run_global must build the validated biome configuration")
    assert "methane_params=ch4_params" in src, (
        "run_global must pass methane_params to run_month_gswp3; without it "
        "the model silently falls back to the 'global' configuration")
