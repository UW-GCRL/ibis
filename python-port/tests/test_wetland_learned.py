"""Learned wetland fraction (docs/METHANE.md 8, 10, 11): the month-by-month feature builder reproduces the explicit array
definitions the predictors were designed with; the hybrid stays inside the cell's bounds and is the climatology when the
anomaly is zero; a trained file applies to its own cells only."""
import numpy as np
import pytest
from scipy.ndimage import uniform_filter
from ibis.bgc.wetland_learned import FeatureBuilder, DYN_FEATS, STAT_FEATS, hybrid_fraction, ROOT_LAYERS, NB


def _rm3(X):
    o = X.copy()
    for t in range(len(X)): o[t] = X[max(0, t - 2):t + 1].mean(0)
    return o


def _lag(X, L):
    o = np.roll(X, L, axis=0); o[:L] = X[:L]; return o


def test_builder_matches_the_array_definitions():
    rng = np.random.default_rng(1); nlat, nlon = 12, 20
    I, J = np.where(rng.random((nlat, nlon)) < 0.6); n = len(I); nl = 6; NM = 9
    poros = rng.uniform(0.3, 0.6, (n, nl)); hsoi = np.array([0.10, 0.15, 0.25, 0.50, 1.0, 2.0])
    ws = rng.uniform(0, 1, (NM, n, nl)); wi = rng.uniform(0, 0.3, (NM, n, nl)); ts = rng.uniform(250, 300, (NM, n, nl)); tg = rng.uniform(250, 300, (NM, n))
    pr = rng.uniform(0, 200, (NM, n)); et = rng.uniform(0, 100, (NM, n)); ro = rng.uniform(0, 80, (NM, n))
    fb = FeatureBuilder(I, J, poros, hsoi, nlat, nlon); out = {k: [] for k in DYN_FEATS}
    for t in range(NM):
        f = fb.step(ts[t], ws[t], wi[t], tg[t], pr[t], et[t], ro[t])
        assert set(f) == set(DYN_FEATS)
        for k in DYN_FEATS: out[k].append(f[k])
    out = {k: np.array(v) for k, v in out.items()}
    W = (ws[:, :, :ROOT_LAYERS] * poros[None, :, :ROOT_LAYERS] * hsoi[None, None, :ROOT_LAYERS]).sum(2)
    lm = np.zeros((nlat, nlon)); lm[I, J] = 1; lf = uniform_filter(lm, NB, mode='wrap')
    def grid(v): g = np.zeros((nlat, nlon)); g[I, J] = v; return g
    def nb(G): return np.stack([(uniform_filter(g, NB, mode='wrap') / np.maximum(lf, 1e-6))[I, J] for g in G])
    P3 = _rm3(pr); R3 = _rm3(ro); NP3 = nb(_rm3(np.stack([grid(v) for v in pr]))); NR3 = nb(_rm3(np.stack([grid(v) for v in ro])))
    want = {'W': W, 'W_l1': _lag(W, 1), 'W_l2': _lag(W, 2), 'W_l3': _lag(W, 3), 'W_3mo': _rm3(W), 'prcp_3mo': P3, 'prcp_3mo_l1': _lag(P3, 1),
            'prcp_3mo_l3': _lag(P3, 3), 'runoff_3mo': R3, 'runoff_3mo_l1': _lag(R3, 1), 'runoff_3mo_l3': _lag(R3, 3), 'nb_prcp_3mo': NP3,
            'nb_prcp_3mo_l1': _lag(NP3, 1), 'nb_prcp_3mo_l2': _lag(NP3, 2), 'nb_runoff_3mo': NR3, 'nb_runoff_3mo_l1': _lag(NR3, 1),
            'nb_runoff_3mo_l2': _lag(NR3, 2), 'prcp': pr, 'runoff': ro, 'aet': et, 'tg': tg, 'ws1': ws[:, :, 0], 'ws6': ws[:, :, 5], 'wi2': wi[:, :, 1],
            't14': (ts[:, :, :ROOT_LAYERS] * hsoi[None, None, :ROOT_LAYERS]).sum(2) / hsoi[:ROOT_LAYERS].sum(),
            'frz1': wi[:, :, 0] / np.maximum(ws[:, :, 0] + wi[:, :, 0], 1e-6)}
    for k, v in want.items():
        assert np.allclose(out[k], v, rtol=1e-10, atol=1e-10), k


def test_hybrid_is_bounded_and_is_the_climatology_without_anomaly():
    fclim = np.array([0.0, 0.1, 0.5, 0.9]); fmax = np.array([0.05, 0.2, 0.6, 0.95]); P = np.array([0.3, 0.9, 0.0, 1.0]); Pb = np.array([0.1, 0.2, 0.5, 0.5])
    f = hybrid_fraction(fclim, P, Pb, 3.0, fmax)
    assert np.all(f >= 0) and np.all(f <= fmax)
    assert np.allclose(hybrid_fraction(fclim, Pb, Pb, 3.0, fmax), np.minimum(fclim, fmax))
    assert np.allclose(hybrid_fraction(fclim, P, Pb, 0.0, fmax), np.minimum(fclim, fmax))


def test_trained_file_applies_to_its_own_cells_only(tmp_path):
    xgb = pytest.importorskip('xgboost')
    rng = np.random.default_rng(2); nf = len(DYN_FEATS) + len(STAT_FEATS); models = []
    for b in range(3):
        X = rng.random((400, nf)).astype(np.float32); y = np.clip(X[:, 0], 0, 1)
        models.append(xgb.train({'max_depth': 2, 'eta': 0.3}, xgb.DMatrix(X, label=y), 5).save_raw('json').decode())
    cI = np.array([1, 2, 9]); cJ = np.array([3, 4, 9])                      # (9, 9) is not a land point of the run below
    np.savez(tmp_path / 'f.npz', dyn_feats=np.array(DYN_FEATS), stat_feats=np.array(STAT_FEATS), cells_I=cI, cells_J=cJ, band=np.array([0, 1, 2]),
             pbar=np.full((12, 3), 0.2), fclim=np.full((12, 3), 0.3), fmax=np.array([0.5, 0.5, 0.5]), lam=np.array([2.0, 1.0, 1.5]),
             stat=rng.random((3, len(STAT_FEATS))), models=np.array(models))
    from ibis.bgc.wetland_learned import LearnedWetland
    I = np.array([0, 1, 2, 5]); J = np.array([0, 3, 4, 6])                  # land points; cells (1,3) -> 1, (2,4) -> 2
    lw = LearnedWetland(str(tmp_path / 'f.npz'), I, J)
    assert list(lw.pts) == [1, 2] and lw.n_dropped == 1
    feats = {k: rng.random(4) for k in DYN_FEATS}; base = np.array([0.01, 0.02, 0.03, 0.04])
    f = lw.fraction(5, feats, base)
    assert f[0] == base[0] and f[3] == base[3]                             # non-cells keep the base fraction
    assert np.all((f[1:3] >= 0) & (f[1:3] <= 0.5))
    with pytest.raises(ValueError):
        bad = dict(np.load(tmp_path / 'f.npz')); bad['dyn_feats'] = np.array(list(DYN_FEATS)[::-1]); np.savez(tmp_path / 'g.npz', **bad)
        LearnedWetland(str(tmp_path / 'g.npz'), I, J)
