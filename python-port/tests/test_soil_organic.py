"""Organic-soil mixing from the HWSD2 organic carbon map (CALIBRATION.md 44).

Off (no map): every property is the F77 texture lookup. On: the organic end-member of
Lawrence & Slater (2008) mixes in by the organic-matter fraction of the solids, and the
dry thermal conductivity in setsoi follows.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from ibis.io.soil_init import (compute_soil_properties, RHOSOI_VAL, CSOI_VAL, ORGANIC_POROS, ORGANIC_BEX,
                               ORGANIC_SUCTION, ORGANIC_HYDRAUL, ORGANIC_RHO, ORGANIC_CSOI, ORGANIC_SFIELD,
                               ORGANIC_SWILT, ORGANIC_CONDRY, RHO_OM_MAX, OM_PER_OC)
from ibis.io.physics_params import build_all_params
from ibis.physics.soil_physics import setsoi
from ibis.timestepping.loops import SimParams


def _tex(n=2, ns=6):
    return np.full((n, ns), 40.0), np.full((n, ns), 25.0)


def test_without_a_map_nothing_changes():
    s, c = _tex()
    a = compute_soil_properties(s, c)
    assert np.allclose(np.asarray(a["rhosoi"]), RHOSOI_VAL) and np.allclose(np.asarray(a["csoi"]), CSOI_VAL)
    assert np.all(np.asarray(a["forganic"]) == 0.0)
    b = compute_soil_properties(s, c, orgc=None, bulk=None)
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def test_pure_peat_is_the_organic_end_member():
    s, c = _tex()
    oc = np.full(s.shape, 58.0); bd = np.full(s.shape, RHO_OM_MAX / (0.58 * OM_PER_OC * 1000.0))   # f_om = 1
    p = compute_soil_properties(s, c, orgc=oc, bulk=bd)
    assert np.allclose(np.asarray(p["forganic"]), 1.0, atol=1e-6)
    assert np.allclose(np.asarray(p["poros"]), ORGANIC_POROS, atol=1e-5)
    assert np.allclose(np.asarray(p["bex"]), ORGANIC_BEX, atol=1e-5)
    assert np.allclose(np.asarray(p["suction"]), ORGANIC_SUCTION, atol=1e-7)
    assert np.allclose(np.asarray(p["hydraul"]), ORGANIC_HYDRAUL, rtol=1e-4)
    assert np.allclose(np.asarray(p["rhosoi"]), ORGANIC_RHO, atol=1e-2)
    assert np.allclose(np.asarray(p["csoi"]), ORGANIC_CSOI, atol=1e-2)
    assert np.allclose(np.asarray(p["sfield"]) * np.asarray(p["poros"]), ORGANIC_SFIELD, atol=1e-5)
    assert np.allclose(np.asarray(p["swilt"]) * np.asarray(p["poros"]), ORGANIC_SWILT, atol=1e-5)


def test_zero_carbon_is_mineral_and_mixing_is_monotonic():
    s, c = _tex()
    m = compute_soil_properties(s, c)
    z = compute_soil_properties(s, c, orgc=np.zeros(s.shape), bulk=np.full(s.shape, 1.4))
    for k in ("poros", "bex", "suction", "hydraul", "sfield", "swilt"):
        assert np.allclose(np.asarray(z[k]), np.asarray(m[k])), k
    h = compute_soil_properties(s, c, orgc=np.full(s.shape, 2.0), bulk=np.full(s.shape, 1.3))   # forest topsoil
    f = float(np.asarray(h["forganic"])[0, 0])
    assert 0.3 < f < 0.5
    assert float(m["poros"][0, 0]) < float(h["poros"][0, 0]) < ORGANIC_POROS
    assert float(h["bex"][0, 0]) < float(m["bex"][0, 0])
    assert float(h["rhosoi"][0, 0]) == pytest.approx(2650.0 * (1 - f) + ORGANIC_RHO * f, rel=1e-5)


def test_missing_values_do_not_poison():
    s, c = _tex()
    oc = np.full(s.shape, np.nan); bd = np.full(s.shape, -7.0)
    p = compute_soil_properties(s, c, orgc=oc, bulk=bd)
    assert np.all(np.isfinite(np.asarray(p["poros"]))) and np.all(np.asarray(p["forganic"]) == 0.0)


def _setsoi(forganic):
    n, ns = 2, 6
    sand = jnp.full((n, ns), 40.0); clay = jnp.full((n, ns), 25.0); poros = jnp.full((n, ns), 0.46)
    z = jnp.zeros((n, ns)); t = jnp.full((n, ns), 283.0)
    return setsoi(sand, clay, poros, z, z, t, jnp.zeros(n), jnp.zeros(n), 10.0, 0.5,
                  jnp.full(n, 283.0), jnp.full(n, 285.0),
                  273.15, 2.5e6, 2.83e6, 4218.0, 1810.0, 2106.0, 1e-12, forganic=forganic)


def test_setsoi_dry_conductivity_follows_the_organic_fraction():
    mineral = 0.40 * 0.300 + 0.35 * 0.265 + 0.25 * 0.250
    r0 = _setsoi(None)
    assert np.allclose(np.asarray(r0.consoi), mineral, atol=1e-6)
    r1 = _setsoi(jnp.ones((2, 6)))
    assert np.allclose(np.asarray(r1.consoi), ORGANIC_CONDRY, atol=1e-6)
    rh = _setsoi(jnp.full((2, 6), 0.5))
    assert np.allclose(np.asarray(rh.consoi), 0.5 * mineral + 0.5 * ORGANIC_CONDRY, atol=1e-6)


def test_forganic_rides_on_sim_params():
    p, _, _ = build_all_params(3, np.zeros(3))
    assert p.forganic is None
    q, _, _ = build_all_params(3, np.zeros(3), forganic=np.full((3, 6), 0.2))
    assert q.forganic.shape == (3, 6) and float(q.forganic[0, 0]) == pytest.approx(0.2)
    assert SimParams.__dataclass_fields__["forganic"].default is None
