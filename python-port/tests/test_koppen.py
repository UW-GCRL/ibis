"""Koppen-Geiger mask on PFT existence, and the larch-rule switch (CALIBRATION.md 44)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from ibis.io.koppen import (MASK_TABLE, KOPPEN_TREES, KOPPEN_LEGEND, koppen_exist_mask,
                            load_koppen, period_index)
from ibis.io.physics_params import build_all_params
from ibis.physics.climate_analysis import existence


def _pft_params(n=1):
    return build_all_params(n, np.zeros(n))[1]


def test_table_shape_and_lower_canopy_never_masked():
    assert MASK_TABLE.shape == (31, 12)
    assert np.all(MASK_TABLE[:, 8:] == 1.0)
    assert np.all(MASK_TABLE[0] == 1.0)                 # no class: unconstrained
    assert set(KOPPEN_TREES) == set(range(1, 31)) == set(KOPPEN_LEGEND)


def test_treeless_and_tree_classes():
    for c in (4, 5, 7, 29, 30):                          # BWh, BWk, BSk, ET, EF
        assert MASK_TABLE[c, :8].sum() == 0, KOPPEN_LEGEND[c]
    assert MASK_TABLE[1, 0] == 1 and MASK_TABLE[1, 1] == 1          # Af: both tropical trees
    assert MASK_TABLE[3, 0] == 0 and MASK_TABLE[3, 1] == 1          # Aw: drought-deciduous only
    assert MASK_TABLE[27, 5] == 1 and MASK_TABLE[27, 0] == 0        # Dfc: boreal, no tropical
    assert MASK_TABLE[15, 7] == 1                                   # Cfb allows larch by envelope


def test_mask_multiplies_into_existence():
    p = _pft_params(3)
    tcmin = jnp.array([5.0, 5.0, 5.0]); tw = jnp.array([28.0] * 3)
    gdd0 = jnp.array([6000.0] * 3); gdd5 = jnp.array([5000.0] * 3)
    base = existence(tcmin, tw, gdd0, gdd5, p)
    assert float(base[0, 0]) == 1.0 and float(base[0, 1]) == 1.0
    mask = koppen_exist_mask(np.array([1, 3, 29]))      # Af, Aw, ET
    out = existence(tcmin, tw, gdd0, gdd5, p, exist_mask=mask)
    assert np.allclose(np.asarray(out), np.asarray(base) * np.asarray(mask))
    assert float(out[0, 0]) == 1.0                      # Af keeps the evergreen tree
    assert float(out[1, 0]) == 0.0 and float(out[1, 1]) == 1.0
    assert float(out[2, :8].sum()) == 0.0 and float(out[2, 11]) == 1.0
    same = existence(tcmin, tw, gdd0, gdd5, p, exist_mask=None)
    assert np.array_equal(np.asarray(same), np.asarray(base))


def test_larch_rule_switch():
    """F77 compares gdd5 with TminL(4) = -45 (never true); the fix uses GDD(4) = 1200."""
    p = _pft_params(1)
    tcmin = jnp.array([-30.0]); tw = jnp.array([15.0]); gdd0 = jnp.array([1500.0]); gdd5 = jnp.array([800.0])
    f77 = existence(tcmin, tw, gdd0, gdd5, p)
    fix = existence(tcmin, tw, gdd0, gdd5, p, larch_gdd_fix=True)
    assert float(f77[0, 7]) == 0.0
    assert float(fix[0, 7]) == 1.0
    assert np.array_equal(np.asarray(f77[0, :7]), np.asarray(fix[0, :7]))
    assert np.array_equal(np.asarray(f77[0, 8:]), np.asarray(fix[0, 8:]))
    # where the coldest month is below -45 the two rules agree
    cold = existence(jnp.array([-50.0]), tw, gdd0, gdd5, p)
    assert float(cold[0, 7]) == 1.0


def test_period_index():
    ps = np.array([1901, 1931, 1961, 1991]); pe = np.array([1930, 1960, 1990, 2020])
    assert [period_index(y, ps, pe) for y in (1701, 1901, 1930, 1931, 1960, 1990, 1991, 2010, 2050)] == \
           [0, 0, 0, 1, 1, 2, 3, 3, 3]


def test_load_koppen_indexes_north_to_south(tmp_path):
    cl = np.zeros((2, 360, 720), np.uint8)
    cl[0, 10, 20] = 29; cl[1, 10, 20] = 27                 # row 10 from the north
    f = tmp_path / "k.npz"
    np.savez(f, classes=cl, period_start=np.array([1901, 1991]), period_end=np.array([1930, 2020]))
    d = load_koppen(str(f), np.array([10, 300]), np.array([20, 5]))
    assert d["classes"].shape == (2, 2)
    assert d["classes"][0, 0] == 29 and d["classes"][1, 0] == 27 and d["classes"][0, 1] == 0
    m = koppen_exist_mask(d["classes"][period_index(2005, d["period_start"], d["period_end"])])
    assert m.shape == (2, 12) and float(m[0, 5]) == 1.0 and float(m[0, 0]) == 0.0
