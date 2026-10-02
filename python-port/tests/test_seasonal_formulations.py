"""Guards for the two seasonal formulations of CALIBRATION.md 21-22: temperature acclimation of photosynthetic
capacity (Makela 2004 / PRELES) and photoperiod-gated senescence (Delpierre 2009 structure).

Each must be bit-identical when switched off, reach the physics through every path when switched on, extend the
optional-module chain so restarts migrate, and round-trip through the parameter file."""
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_daylength_is_astronomically_right():
    from ibis.state.seasonal import daylength_hours
    eq = np.asarray(daylength_hours(jnp.array([0.0]), jnp.arange(1, 366)))
    assert np.all(np.abs(eq - 12.0) < 0.05)                                       # equator: 12 h all year
    n45 = float(daylength_hours(jnp.deg2rad(45.0), 172)); s45 = float(daylength_hours(jnp.deg2rad(45.0), 355))
    assert 15.2 < n45 < 15.6 and 8.4 < s45 < 8.8                                   # 45 N solstices (geometric, no refraction)
    assert abs(float(daylength_hours(jnp.deg2rad(-45.0), 172)) - s45) < 0.1       # hemispheres mirror
    assert float(daylength_hours(jnp.deg2rad(80.0), 172)) == pytest.approx(24.0) and float(daylength_hours(jnp.deg2rad(80.0), 355)) == pytest.approx(0.0)


def test_state_chain_and_restart_migration(tmp_path):
    from ibis.state.model_state import create_ibis_state
    from ibis.io.restart import save_state, load_state
    from ibis.dimensions import GridConfig
    g = GridConfig(npoi=4, nbedrock=3)
    n = [len(jax.tree.leaves(create_ibis_state(g, **kw))) for kw in ({}, {"with_stress": True}, {"with_etparts": True}, {"with_seasonal": True}, {"with_neeparts": True})]
    from ibis.io.restart import CHAIN_LEAVES
    assert (n[1] - n[0], n[2] - n[1], n[3] - n[2], n[4] - n[3]) == (3, 4, 3, 2) == CHAIN_LEAVES      # one nested chain
    s = create_ibis_state(g, with_seasonal=True)
    assert s.stress is not None and s.etparts is not None and s.seasonal is not None   # seasonal implies the others
    s5 = create_ibis_state(g, with_neeparts=True)
    assert s5.seasonal is not None and s5.neeparts is not None                        # neeparts implies all of them
    save_state(s5, f"{tmp_path}/n.npz")
    down, _ = load_state(f"{tmp_path}/n.npz", create_ibis_state(g, with_seasonal=True))   # 270 -> 268: drop 2
    assert down.neeparts is None and down.seasonal is not None
    upn, _ = load_state(f"{tmp_path}/n.npz", create_ibis_state(g, with_neeparts=True))    # 270 -> 270
    assert upn.neeparts is not None and float(upn.neeparts.night.sum()) == 0.0
    fu = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=s.vegetation.fu.dtype)
    s = eqx.tree_at(lambda z: (z.vegetation.fu, z.seasonal.acclim_x), s, (fu, jnp.full_like(s.seasonal.acclim_x, 7.0)))
    save_state(s, f"{tmp_path}/s.npz")
    back, _ = load_state(f"{tmp_path}/s.npz", create_ibis_state(g))                    # 268 -> 258: drop 10
    assert back.seasonal is None and np.array_equal(np.asarray(back.vegetation.fu), np.asarray(fu))
    save_state(create_ibis_state(g, with_etparts=True), f"{tmp_path}/e.npz")
    up, _ = load_state(f"{tmp_path}/e.npz", create_ibis_state(g, with_seasonal=True))  # 265 -> 268: fill 3
    assert up.seasonal is not None and float(up.seasonal.ssen_u.sum()) == 0.0
    save_state(create_ibis_state(g, with_stress=True), f"{tmp_path}/s3.npz")        # 261 -> 265: a legal move along the chain
    mid, _ = load_state(f"{tmp_path}/s3.npz", create_ibis_state(g, with_etparts=True))
    assert mid.etparts is not None and float(mid.etparts.trans.sum()) == 0.0
    z = dict(np.load(f"{tmp_path}/s3.npz", allow_pickle=True)); z["num_leaves"] = np.array(int(z["num_leaves"]) - 2)
    np.savez(f"{tmp_path}/bad.npz", **z)                                               # 2 leaves short: on no chain position
    with pytest.raises(ValueError):
        load_state(f"{tmp_path}/bad.npz", create_ibis_state(g, with_stress=True))


def test_parameter_file_round_trip(tmp_path):
    import yaml
    from ibis.io.physics_params import build_all_params, load_physics_params
    base = yaml.safe_load(open(ROOT / "configs" / "params" / "default.yaml"))
    base.setdefault("stomata", {}).update(acclim_on=True, acclim_pfts=[3, 5], acclim_tau=8.0, acclim_s0=-4.0, acclim_smax=15.0)
    base.setdefault("pheno", {}).update(sen_photoperiod_u=True, sen_pstart_u=13.5, sen_tb_u=18.0, sen_ycrit_u=250.0)
    f = tmp_path / "p.yaml"; yaml.safe_dump(base, open(f, "w"))
    sp, pft, _ = build_all_params(2, np.zeros(2, np.float32), load_physics_params(str(f)))
    st = sp.stomata_params; ph = sp.pheno
    assert st.acclim_on is True and st.acclim_pfts == (3, 5) and (st.acclim_tau, st.acclim_s0, st.acclim_smax) == (8.0, -4.0, 15.0)
    assert ph.sen_photoperiod_u is True and ph.sen_photoperiod_l is False and isinstance(ph.sen_photoperiod_u, bool)
    assert (ph.sen_pstart_u, ph.sen_tb_u, ph.sen_ycrit_u) == (13.5, 18.0, 250.0)
    sp0, _, _ = build_all_params(2, np.zeros(2, np.float32), load_physics_params(str(ROOT / "configs" / "params" / "default.yaml")))
    assert sp0.stomata_params.acclim_on is False and sp0.pheno.sen_photoperiod_u is False and sp0.pheno.sen_photoperiod_l is False


def test_vmax_tables_scale_only_the_acclimating_pfts():
    from ibis.physics.stomata import StomataParams, _select_vmax_upper, _select_vmax_shrub
    p = StomataParams(vmax_mix=True)
    npoi = 3; frac = jnp.zeros((npoi, 12)).at[:, 3].set(0.5).at[:, 4].set(0.5).at[:, 8].set(1.0)
    exist = jnp.ones((npoi, 12))
    ub0, uc0 = _select_vmax_upper(exist, p, frac); sh0 = _select_vmax_shrub(p, frac)
    ones = jnp.ones((npoi, 12))
    ub1, uc1 = _select_vmax_upper(exist, p, frac, vp=p.vmax_pft[None, :] * ones); sh1 = _select_vmax_shrub(p, frac, vp=p.vmax_pft[None, :] * ones)
    assert np.array_equal(np.asarray(ub0), np.asarray(ub1)) and np.array_equal(np.asarray(uc0), np.asarray(uc1)) and np.array_equal(np.asarray(sh0), np.asarray(sh1))
    fs = jnp.array([0.0, 0.5, 1.0]); mask = jnp.zeros(12).at[jnp.array([3, 5, 8])].set(1.0)
    vfac = 1.0 - mask[None, :] + mask[None, :] * fs[:, None]
    ub2, uc2 = _select_vmax_upper(exist, p, frac, vp=p.vmax_pft[None, :] * vfac); sh2 = _select_vmax_shrub(p, frac, vp=p.vmax_pft[None, :] * vfac)
    assert np.allclose(np.asarray(ub2), np.asarray(ub0))                            # broadleaf: not acclimating
    assert np.allclose(np.asarray(uc2), np.asarray(uc0) * np.asarray(fs), rtol=1e-6)  # conifer: scaled by f_S
    assert np.allclose(np.asarray(sh2), np.asarray(sh0) * np.asarray(fs), rtol=1e-6)  # evergreen shrub: scaled


def test_pheno_senescence_switch_is_bit_identical_when_off_and_drops_leaves_when_on():
    from ibis.physics.phenology import pheno, PhenoParams
    from ibis.state.model_state import create_ibis_state
    from ibis.dimensions import GridConfig
    s = create_ibis_state(GridConfig(npoi=3, nbedrock=3))
    v = eqx.tree_at(lambda q: (q.tempu, q.agddu, q.plai), s.vegetation, (jnp.ones(3), jnp.full((3,), 900.0), s.vegetation.plai.at[:, 4].set(4.0)))
    summ = eqx.tree_at(lambda q: q.a10td, s.summary, jnp.full((3,), 288.0))         # warm: old rule keeps leaves
    td = jnp.full((3,), 288.0)
    a = pheno(v, summ, td, 1e-7, PhenoParams())
    b = pheno(v, summ, td, 1e-7, PhenoParams(), sen_on_u=jnp.zeros(3, bool), sen_on_l=jnp.zeros(3, bool))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)): assert np.array_equal(np.asarray(x), np.asarray(y))
    c = pheno(v, summ, td, 1e-7, PhenoParams(), sen_on_u=jnp.array([True, False, True]))
    assert np.allclose(np.asarray(c.tempu), [1.0 - 1.0 / 15.0, 1.0, 1.0 - 1.0 / 15.0])   # decline at 1/ddays
    assert np.allclose(np.asarray(c.agddu)[[0, 2]], 0.0) and float(c.agddu[1]) > 0      # next leaf-out sum held at 0


def test_every_path_is_wired():
    lg = (ROOT / "ibis" / "timestepping" / "loops_gswp3.py").read_text()
    assert "sen_on_u=sen_on_u, sen_on_l=sen_on_l" in lg and "_sea.acclim_x + (_tc - _sea.acclim_x) / _sp.acclim_tau" in lg
    assert "_dec & (_dl < _pp.sen_pstart_u)" in lg and "jnp.maximum(_pp.sen_tb_u - _tc, 0.0)" in lg
    lp = (ROOT / "ibis" / "timestepping" / "loops.py").read_text()
    assert "frac=veg.frac, vfac=acclim_vfac," in lp and "state.seasonal.acclim_x - sp.acclim_s0) / sp.acclim_smax" in lp
    rg = (ROOT / "scripts" / "run_global.py").read_text()
    assert rg.count("with_seasonal=seas_on") == 2
    st = (ROOT / "ibis" / "physics" / "stomata.py").read_text()
    assert "vmaxl3_nom = params.vmax_pft[11] if vp_eff is None else vp_eff[:, 11]" in st and "vmaxl4_nom = params.vmax_pft[10] if vp_eff is None else vp_eff[:, 10]" in st


def test_acclimation_recursion_matches_preles():
    # PRELES fS_model: S = S + (T - S)/tau; fS = clip((S - S0)/Smax, 0, 1)
    tau, s0, smax = 10.93, -3.063, 17.72
    S = 0.0; T = np.r_[np.full(40, -15.0), np.full(60, 20.0)]
    for t in T: S = S + (t - S) / tau
    fs = min(max((S - s0) / smax, 0.0), 1.0)
    assert fs == 1.0 and S > 19.0                                                   # summer: full capacity
    S = 0.0
    for t in np.full(60, -15.0): S = S + (t - S) / tau
    assert min(max((S - s0) / smax, 0.0), 1.0) == 0.0                               # frozen winter: shut down
