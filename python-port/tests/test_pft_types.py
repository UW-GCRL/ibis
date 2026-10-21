"""IBIS 3.1 Köppen-zone types in the 12 IBIS positions (ibis.io.pft_types; CALIBRATION.md 45.1)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from ibis.io import pft_types as P
from ibis.io.physics_params import build_all_params
from ibis.physics.climate_analysis import existence
from ibis.physics.stomata import capacity_shares


def test_forty_types_and_no_t20():
    assert P.NTYPE == 40 and 'T20' not in P.TI and len(set(P.IDS)) == 40


def test_every_type_is_placed_in_every_zone_it_lists():
    for i, (tid, _, form, zones, parent) in enumerate(P.TYPES):
        if form in ('wetland', 'crop'):
            continue
        for z in zones:
            row = P.ZONE_SLOTS[P.ZI[z]]
            assert i in row, (tid, z)
            pos = int(np.where(row == i)[0][0])
            if form in ('tree_ne', 'tree_nd'): assert pos in P.CONIFER_POS, (tid, z, pos)
            elif form.startswith('tree'): assert pos in P.BROADLEAF_POS, (tid, z, pos)
            elif form == 'shrub_e': assert pos == 8
            elif form == 'shrub_d': assert pos == 9
            elif form == 'grass_c4': assert pos == 10
            elif form == 'grass_c3': assert pos == 11


def test_moved_types_keep_their_habit():
    hab = P.ZONE_HABIT[P.ZI['A_savanna']]; row = P.ZONE_SLOTS[P.ZI['A_savanna']]
    assert P.IDS[row[0]] == 'T03' and hab[0] == P.DD          # savanna tree in position 1, drought-deciduous
    assert P.IDS[row[1]] == 'T02' and hab[1] == P.DD
    row = P.ZONE_SLOTS[P.ZI['B_hot']]; hab = P.ZONE_HABIT[P.ZI['B_hot']]
    assert P.IDS[row[0]] == 'T05' and hab[0] == P.DD


def test_unclassified_cells_are_the_12_pft_model_and_ice_has_no_trees():
    assert np.all(P.ZONE_SLOTS[P.ZI['none']] == P.GENERIC)
    assert np.all(P.ZONE_SLOTS[P.ZI['E_ice']][:8] == P.OFF)
    assert tuple(P.ZONE_HABIT[P.ZI['none']]) == P.POSITION_HABIT


def test_gather_uses_the_parent_and_the_overrides():
    base = np.arange(1.0, 13.0)                                # PFT k (1-based) has value k
    z = np.array([P.ZI['A_savanna'], P.ZI['D_subarctic'], P.ZI['none']])
    s = P.slot_map(z)
    g = P.gather(base, s)
    assert g.shape == (3, 12)
    assert g[0, 0] == 2.0                                       # T03 in position 1 takes parent 2's value
    assert g[1, 5] == 6.0 and g[1, 7] == 8.0                    # T17 and T19
    assert np.array_equal(g[2], base.astype(np.float32))        # unclassified: the 12-PFT values
    g2 = P.gather(base, s, {'T03': 99.0})
    assert g2[0, 0] == 99.0 and g2[0, 1] == 2.0                 # T02 untouched
    with pytest.raises(KeyError):
        P.gather(base, s, {'T20': 1.0})


def test_existence_takes_the_parent_envelope_and_the_gate():
    pp = build_all_params(2, np.zeros(2))[1]
    tcmin = jnp.array([5.0, 5.0]); tw = jnp.array([28.0, 28.0]); gdd0 = jnp.array([6000.0] * 2); gdd5 = jnp.array([5000.0] * 2)
    z = np.array([P.ZI['A_savanna'], P.ZI['none']]); s = P.slot_map(z)
    e = existence(tcmin, tw, gdd0, gdd5, pp, slot_parent=P.slot_parent(s), slot_gate=P.slot_gate(s))
    base = existence(tcmin, tw, gdd0, gdd5, pp)
    assert float(e[0, 0]) == float(base[0, 1]) == 1.0           # T03 in position 1 with PFT 2's envelope
    assert float(e[0, 2]) == 0.0 and float(e[0, 3]) == 0.0      # no type there in Aw
    assert np.array_equal(np.asarray(e[1]), np.asarray(base[1]))


def test_typed_positions_exist_by_zone_not_by_the_parents_temperature_band():
    # CALIBRATION.md 52: a subtropical cell (coldest month +5 C) -- the F77 bands forbid the temperate PFTs 3/4/5
    # (they need tcmin < 0), the zone type must exist there; a tundra cell below the degree-day floor stays treeless.
    pp = build_all_params(3, np.zeros(3))[1]
    tcmin = jnp.array([5.0, -30.0, 5.0]); tw = jnp.array([28.0, 8.0, 28.0]); gdd0 = jnp.array([6000.0, 300.0, 6000.0]); gdd5 = jnp.array([5000.0, 200.0, 5000.0])
    z = np.array([P.ZI['C_subtrop'], P.ZI['E_tundra'], P.ZI['none']]); s = P.slot_map(z)
    kw = dict(slot_parent=P.slot_parent(s), slot_gate=P.slot_gate(s), slot_typed=P.slot_typed(s))
    e = existence(tcmin, tw, gdd0, gdd5, pp, **kw); base = existence(tcmin, tw, gdd0, gdd5, pp)
    assert float(base[0, 2]) == 0.0 and float(base[0, 3]) == 0.0 and float(base[0, 4]) == 0.0
    assert [float(e[0, k]) for k in (2, 3, 4)] == [1.0, 1.0, 1.0]        # T06 / T07 / T08 exist in Cfa
    assert [float(e[0, k]) for k in (0, 1, 5, 6, 7)] == [0.0] * 5       # OFF positions stay off
    assert float(e[1, 5]) == 0.0                                          # T17 in tundra below gdd5 350: no
    old = existence(tcmin, tw, gdd0, gdd5, pp, slot_parent=P.slot_parent(s), slot_gate=P.slot_gate(s))
    assert float(old[0, 3]) == 0.0                                        # the parent's band alone forbade it
    assert np.array_equal(np.asarray(e[2]), np.asarray(base[2]))         # unclassified cell: the 12-PFT envelope


def test_capacity_shares_conserve_the_big_leaf_and_favour_capacity():
    frac = jnp.array([[0.5, 0.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0]])
    same = capacity_shares(frac, jnp.full((12,), 40e-6), (0, 1, 2, 4, 6))
    assert np.allclose(np.asarray(same), np.asarray(frac[:, jnp.array([0, 1, 2, 4, 6])]))
    vp = jnp.full((12,), 40e-6).at[4].set(80e-6)
    sh = capacity_shares(frac, vp, (0, 1, 2, 4, 6))
    assert float(sh.sum()) == pytest.approx(float(frac[:, jnp.array([0, 1, 2, 4, 6])].sum()), rel=1e-6)
    assert float(sh[0, 3]) > 0.2                                # position 5 has twice the vmax
    empty = capacity_shares(jnp.zeros((1, 12)), vp, (3, 5, 7))
    assert np.all(np.asarray(empty) == 0.0)


def test_apply_types_shapes_and_unclassified_identity():
    params, pp, _ = build_all_params(3, np.zeros(3))
    z = np.full(3, P.ZI['none'])
    p2, pp2 = P.apply_types(params, pp, z)
    assert pp2.specla.shape == (3, 12) and p2.stomata_params.vmax_pft.shape == (3, 12)
    assert np.allclose(np.asarray(pp2.specla), np.asarray(pp.specla)[None, :])
    assert np.allclose(np.asarray(p2.stomata_params.vmax_pft), np.asarray(params.stomata_params.vmax_pft)[None, :])
    assert np.allclose(np.asarray(p2.fire.mort), np.asarray(params.fire.mort)[None, :])
    assert p2.slot_habit.shape == (3, 12) and float(p2.slot_gate.min()) == 1.0
    z2 = np.array([P.ZI['D_subarctic'], P.ZI['A_savanna'], P.ZI['C_med']])
    over = P.parse_type_overrides({'T19': {'vmax': 55e-6, 'specla': 20.0}})
    p3, pp3 = P.apply_types(params, pp, z2, over, base=(params, pp))
    assert float(p3.stomata_params.vmax_pft[0, 7]) == pytest.approx(55e-6)
    assert float(pp3.specla[0, 7]) == pytest.approx(20.0)
    assert float(p3.slot_gate[1, 3]) == 0.0                    # no conifer in Aw
    with pytest.raises(KeyError):
        P.parse_type_overrides({'T19': {'vmaxx': 1.0}})
    with pytest.raises(ValueError):
        P.apply_types(p3, pp3, z2)                              # already per cell without a base


def test_phenology_habit_matches_the_index_path():
    from ibis.physics.phenology import pheno
    from ibis.state.vegetation import create_vegetation_state
    import inspect
    n = 4
    veg = create_vegetation_state(n, 12, 6)
    rng = np.random.default_rng(0)
    veg = veg.__class__(**{**{k: getattr(veg, k) for k in veg.__dataclass_fields__},
                          'plai': jnp.asarray(rng.uniform(0.1, 3.0, (n, 12)), dtype=jnp.float32),
                          'fu': jnp.full((n,), 0.6), 'fl': jnp.full((n,), 0.7),
                          'tempu': jnp.full((n,), 0.4), 'dropu': jnp.full((n,), 0.7)})
    from ibis.state.summary import create_summary_state
    sm = create_summary_state(n)
    td = jnp.full((n,), 290.0)
    a = pheno(veg, sm, td, 1e-12)
    b = pheno(veg, sm, td, 1e-12, habit=jnp.broadcast_to(jnp.array(P.POSITION_HABIT), (n, 12)))
    assert np.allclose(np.asarray(a.frac), np.asarray(b.frac), rtol=1e-6, atol=1e-7)
    assert np.allclose(np.asarray(a.lai), np.asarray(b.lai), rtol=1e-6)
    h = np.array(P.POSITION_HABIT); h[0] = P.DD
    c = pheno(veg, sm, td, 1e-12, habit=jnp.broadcast_to(jnp.array(h), (n, 12)))
    assert float(c.frac[0, 0]) < float(a.frac[0, 0])            # position 1 now drops its leaves with dropu


def test_table_matches_the_design_script():
    import ast, pathlib
    f = pathlib.Path(__file__).resolve().parents[2] / 'verify' / 'calibration' / 'pft_koppen_design.py'
    tree = ast.parse(f.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'PFTS')
    d = {p[0]: (p[2], tuple(p[3]), p[4]) for p in ast.literal_eval(node.value)}
    m = {t[0]: (t[2], tuple(t[3]), t[4]) for t in P.TYPES}
    assert d == m


def test_per_zone_scalars_follow_the_zone_types():
    params, pp, _ = build_all_params(3, np.zeros(3))
    z = np.array([P.ZI['D_temperate'], P.ZI['A_humid'], P.ZI['D_subarctic']])
    over = P.parse_type_overrides({'T15': {'gdd_thresh_u': 200.0}, 'G06': {'gdd_thresh_l': 120.0},
                                   'T16': {'stressfac': -3.0}, 'T18': {'gdd_thresh_u': 80.0}, 'T19': {'gdd_thresh_u': 100.0}})
    p, _ = P.apply_types(params, pp, z, over)
    gu = np.asarray(p.pheno.gdd_thresh_u); gl = np.asarray(p.pheno.gdd_thresh_l); sf = np.asarray(p.stomata_params.stressfac)
    assert gu[0] == pytest.approx(200.0)                                   # T15, the zone's cold-deciduous tree
    assert gu[1] == pytest.approx(float(params.pheno.gdd_thresh_u))        # no cold-deciduous tree in A_humid: base
    assert gu[2] == pytest.approx(90.0)                                    # median of T18 and T19
    assert gl[0] == pytest.approx(120.0) and gl[2] == pytest.approx(float(params.pheno.gdd_thresh_l))
    assert sf[0] == pytest.approx(-3.0) and sf[1] == pytest.approx(float(params.stomata_params.stressfac))


def test_drystress_per_cell_stressfac_matches_the_scalar():
    from ibis.physics.stomata import drystress
    rng = np.random.default_rng(1)
    w = jnp.asarray(rng.uniform(0.2, 0.9, (3, 6))); wi = jnp.zeros((3, 6))
    sw = jnp.full((3, 6), 0.3); sfc = jnp.full((3, 6), 0.7); fr = jnp.asarray(np.tile(np.linspace(0.3, 0.05, 6)[:, None], (1, 2)))
    per = drystress(w, wi, sw, sfc, fr, stressfac=jnp.array([-5.0, -3.0, -8.0]))
    for k, s in enumerate((-5.0, -3.0, -8.0)):
        one = drystress(w, wi, sw, sfc, fr, stressfac=s)
        for a, b in zip(per, one):
            assert np.allclose(np.asarray(a)[k], np.asarray(b)[k], rtol=1e-6)
