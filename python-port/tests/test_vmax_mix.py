"""Composition-weighted canopy vmax (CALIBRATION.md 17).

F77 picks ONE PFT's vmax per big leaf by existence priority, so four of the
twelve vmax_pft entries never reach photosynthesis. With vmax_mix the
displayed-LAI shares decide; every entry must be reachable."""
import numpy as np, jax.numpy as jnp
from ibis.physics.stomata import (StomataParams, _select_vmax_upper, _select_vmax_shrub,
                                  BROADLEAF_PFTS, CONIFER_PFTS, SHRUB_PFTS)

VP = jnp.arange(1.0, 13.0) * 1e-6          # distinct value per PFT


def _params(mix):
    return StomataParams(vmax_pft=VP, vmax_mix=mix)


def test_default_is_the_f77_priority_pick_bit_for_bit():
    rng = np.random.default_rng(0)
    exist = jnp.asarray(rng.random((50, 12)) > 0.5, dtype=jnp.float32)
    frac = jnp.asarray(rng.random((50, 12)), dtype=jnp.float32)
    ub0, uc0 = _select_vmax_upper(exist, _params(False))
    ub1, uc1 = _select_vmax_upper(exist, _params(False), frac)      # frac ignored when off
    exp_ub = jnp.where(exist[:, 0] > 0.5, VP[0], jnp.where(exist[:, 2] > 0.5, VP[2], VP[4]))
    exp_uc = jnp.where(exist[:, 3] > 0.5, VP[3], VP[5])
    assert np.array_equal(ub0, exp_ub) and np.array_equal(uc0, exp_uc)
    assert np.array_equal(ub1, exp_ub) and np.array_equal(uc1, exp_uc)
    assert float(_select_vmax_shrub(_params(False), frac)) == float(VP[8])


def test_every_vmax_entry_is_reachable_with_mix():
    """One-hot canopies: the big leaf takes exactly that PFT's vmax -- the
    guard that no entry of vmax_pft is dead once the switch is on."""
    p = _params(True)
    exist = jnp.ones((1, 12))
    for j in range(12):
        frac = jnp.zeros((1, 12)).at[0, j].set(1.0)
        ub, uc = _select_vmax_upper(exist, p, frac)
        ls = _select_vmax_shrub(p, frac)
        got = {**{k: float(ub[0]) for k in BROADLEAF_PFTS}, **{k: float(uc[0]) for k in CONIFER_PFTS},
               **{k: float(ls[0]) for k in SHRUB_PFTS}}
        if j in got:
            assert abs(got[j] - float(VP[j])) < 1e-12, (j, got[j], float(VP[j]))
    # grasses (10, 11) are read directly as vmax_pft[10], vmax_pft[11] in stomata()


def test_mix_is_the_lai_weighted_mean_and_falls_back_without_leaves():
    p = _params(True)
    exist = jnp.ones((2, 12))
    frac = jnp.zeros((2, 12)).at[0, 0].set(0.25).at[0, 1].set(0.75)      # cell 0: 25 % PFT1, 75 % PFT2
    ub, uc = _select_vmax_upper(exist, p, frac)
    assert abs(float(ub[0]) - (0.25 * float(VP[0]) + 0.75 * float(VP[1]))) < 1e-12
    assert abs(float(ub[1]) - float(VP[0])) < 1e-12          # cell 1: no leaves -> priority pick (PFT 0 exists)
    assert abs(float(uc[0]) - float(VP[3])) < 1e-12          # no conifer leaves -> priority (PFT 3 exists)


def test_call_site_threads_frac_and_yaml_reads_the_switch():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert "frac=veg.frac" in (root / "ibis" / "timestepping" / "loops.py").read_text()
    assert "vmax_mix=bool(s.get('vmax_mix'" in (root / "ibis" / "io" / "physics_params.py").read_text()


# ---------------------------------------------------------------------------
# Canopy scaling denominator (CALIBRATION.md 16, lever test 11)
# ---------------------------------------------------------------------------

def test_scale_leaf_only_is_old_scale_times_lai_over_lai_plus_sai():
    """Default reproduces F77's (1-exp(-k(L+S)))/kL bit for bit; the switch
    gives exactly old x L/(L+S) in daytime, so stems stop assimilating."""
    from ibis.physics.stomata import canopy_scaling
    rng = np.random.default_rng(1); n = 40
    toppar = jnp.full(n, 400.0); term6 = jnp.asarray(rng.uniform(0.4, 1.5, n)); term7 = jnp.asarray(rng.uniform(0.3, 1.0, n))
    scalcoef = jnp.asarray(rng.uniform(0.1, 1.0, (n, 4))); lai = jnp.asarray(rng.uniform(0.2, 4.0, n)); sai = 0.25 * lai
    a10s = jnp.ones(n); a10d = jnp.ones(n)
    s_old, _, _ = canopy_scaling(toppar, term6, term7, scalcoef, lai, sai, a10s, a10d, 3600.0, 1e-7)
    s_def, _, _ = canopy_scaling(toppar, term6, term7, scalcoef, lai, sai, a10s, a10d, 3600.0, 1e-7, leaf_only=False)
    s_new, _, _ = canopy_scaling(toppar, term6, term7, scalcoef, lai, sai, a10s, a10d, 3600.0, 1e-7, leaf_only=True)
    assert np.array_equal(np.asarray(s_old), np.asarray(s_def))
    np.testing.assert_allclose(np.asarray(s_new), np.asarray(s_old) * np.asarray(lai / (lai + sai)), rtol=1e-6)


def test_scale_leaf_only_switch_reaches_both_canopies_and_the_yaml():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    st = (root / "ibis" / "physics" / "stomata.py").read_text()
    assert st.count("leaf_only=params.scale_leaf_only") == 2, "upper and lower canopy_scaling calls"
    assert "scale_leaf_only=bool(s.get('scale_leaf_only'" in (root / "ibis" / "io" / "physics_params.py").read_text()
    assert StomataParams().scale_leaf_only is False
