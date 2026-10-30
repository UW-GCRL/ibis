"""Learned wetland (inundated) fraction (docs/METHANE.md 8, 10, 11; Hangkai 2026-10-29).

Off by default. With `physics.wetland_source: learned` the month's inundated fraction the methane scheme sees is

    f = clip(f_clim[m] + lambda_band * (P - Pbar[m]), 0, f_max_cell)

the observed WAD2M seasonal climatology plus the anomaly of a learned model P against its own climatology Pbar, inflated
per wetland band. P is a gradient-boosted tree model per band (W01 / W02 / W03), monotone increasing in every water variable,
trained on the WAD2M monthly fraction of 2001-2010 from a reference hindcast's MONTH-MEAN soil state (output.soil_means),
its monthly water totals, their lags, 3-month means and 5x5-cell neighbourhood means, and static topography, texture,
latitude and the cell's WAD2M range (verify/methane/wetland_learned_train.py). No month-of-year and no climatology among the
predictors: the anomaly comes from the model's own hydrology.

This module holds the ONE feature construction both the training script and the runtime use, stepped month by month in
calendar order with explicit buffers, so lags and running means cannot differ between them. Host-side numpy; the trees
are evaluated with xgboost (imported only when the mode is on).

Semantics of the buffers at the start of a series (as the training's arrays): a lag of L months before L months exist is
the current month's value; a 3-month mean over fewer months is the mean of those that exist.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter

#: Monthly predictors, in model order (the trained file stores its own order and is checked against this).
DYN_FEATS = ('W', 'Wice', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6', 'wi1', 'wi2', 't1', 't14', 'tg', 'frz1', 'prcp', 'aet', 'runoff',
             'W_l1', 'W_l2', 'W_l3', 'W_3mo', 'prcp_3mo', 'prcp_3mo_l1', 'prcp_3mo_l3', 'runoff_3mo', 'runoff_3mo_l1', 'runoff_3mo_l3',
             'nb_prcp_3mo', 'nb_prcp_3mo_l1', 'nb_prcp_3mo_l2', 'nb_runoff_3mo', 'nb_runoff_3mo_l1', 'nb_runoff_3mo_l2')
STAT_FEATS = ('FMAX_clm', 'SLOPE', 'STD_ELEV', 'TOPO', 'PCT_WETLAND', 'PCT_LAKE', 'binfl', 'sand', 'clay', 'lat', 'fmin_w', 'fmax_w')
#: Monotone increasing in these (more water -> more wetland, never less).
MONO = ('W', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6', 'W_l1', 'W_l2', 'W_l3', 'W_3mo', 'prcp', 'runoff', 'prcp_3mo', 'prcp_3mo_l1', 'prcp_3mo_l3',
        'runoff_3mo', 'runoff_3mo_l1', 'runoff_3mo_l3', 'nb_prcp_3mo', 'nb_prcp_3mo_l1', 'nb_prcp_3mo_l2', 'nb_runoff_3mo', 'nb_runoff_3mo_l1', 'nb_runoff_3mo_l2')
ROOT_LAYERS = 4          # the top metre of IBIS's column (0.10, 0.15, 0.25, 0.50 m)
NB = 5                   # neighbourhood width, cells (5 x 5 at 0.5 degree ~ 250 km)


class FeatureBuilder:
    """Month-by-month predictors for every land point of a run, in calendar order.

    I, J: the land points' rows and columns on the (nlat, nlon) output grid; poros (npoi, nlayer) and hsoi (nlayer,) the
    model's porosity and layer thicknesses. Call `step` once per month with that month's MEAN soil state and its totals;
    it returns {name: (npoi,) array} for DYN_FEATS.
    """

    def __init__(self, I, J, poros, hsoi, nlat: int = 360, nlon: int = 720):
        self.I = np.asarray(I, int); self.J = np.asarray(J, int); self.nlat, self.nlon = nlat, nlon
        self.p4 = np.asarray(poros, np.float64)[:, :ROOT_LAYERS]; self.h4 = np.asarray(hsoi, np.float64)[:ROOT_LAYERS]
        lm = np.zeros((nlat, nlon), np.float64); lm[self.I, self.J] = 1.0
        self.lf = uniform_filter(lm, NB, mode='wrap')
        self.buf = {k: [] for k in ('W', 'prcp', 'runoff', 'GP', 'GR', 'W3', 'P3', 'R3', 'NP3', 'NR3')}
        self.t = 0                                                       # months stepped

    def _grid(self, v):
        g = np.zeros((self.nlat, self.nlon), np.float64); g[self.I, self.J] = v; return g

    def _nb(self, g):
        return (uniform_filter(g, NB, mode='wrap') / np.maximum(self.lf, 1e-6))[self.I, self.J]

    @staticmethod
    def _mean3(lst):
        return np.mean(lst[-3:], axis=0)

    @staticmethod
    def _lag(lst, L):
        return lst[-1 - L] if len(lst) > L else lst[-1]

    def _push(self, k, v, keep=4):
        self.buf[k].append(v)
        if len(self.buf[k]) > keep: self.buf[k].pop(0)

    def step(self, tsoi_mean, wsoi_mean, wisoi_mean, tg_mean, prcp, aet, runoff) -> dict:
        ws = np.asarray(wsoi_mean, np.float64); wi = np.asarray(wisoi_mean, np.float64); ts = np.asarray(tsoi_mean, np.float64)
        W = (ws[:, :ROOT_LAYERS] * self.p4 * self.h4[None, :]).sum(1)
        Wice = (wi[:, :ROOT_LAYERS] * self.p4 * self.h4[None, :]).sum(1)
        prcp = np.asarray(prcp, np.float64); runoff = np.asarray(runoff, np.float64)
        f = {'W': W, 'Wice': Wice, **{f'ws{l + 1}': ws[:, l] for l in range(6)}, 'wi1': wi[:, 0], 'wi2': wi[:, 1], 't1': ts[:, 0],
             't14': (ts[:, :ROOT_LAYERS] * self.h4[None, :]).sum(1) / self.h4.sum(), 'tg': np.asarray(tg_mean, np.float64),
             'frz1': wi[:, 0] / np.maximum(ws[:, 0] + wi[:, 0], 1e-6), 'prcp': prcp, 'aet': np.asarray(aet, np.float64), 'runoff': runoff}
        self._push('W', W); self._push('prcp', prcp, 3); self._push('runoff', runoff, 3)
        self._push('GP', self._grid(prcp), 3); self._push('GR', self._grid(runoff), 3)
        self._push('W3', self._mean3(self.buf['W'])); self._push('P3', self._mean3(self.buf['prcp'])); self._push('R3', self._mean3(self.buf['runoff']))
        self._push('NP3', self._nb(self._mean3(self.buf['GP'])), 3); self._push('NR3', self._nb(self._mean3(self.buf['GR'])), 3)
        b = self.buf
        f.update({'W_l1': self._lag(b['W'], 1), 'W_l2': self._lag(b['W'], 2), 'W_l3': self._lag(b['W'], 3), 'W_3mo': b['W3'][-1],
                  'prcp_3mo': b['P3'][-1], 'prcp_3mo_l1': self._lag(b['P3'], 1), 'prcp_3mo_l3': self._lag(b['P3'], 3),
                  'runoff_3mo': b['R3'][-1], 'runoff_3mo_l1': self._lag(b['R3'], 1), 'runoff_3mo_l3': self._lag(b['R3'], 3),
                  'nb_prcp_3mo': b['NP3'][-1], 'nb_prcp_3mo_l1': self._lag(b['NP3'], 1), 'nb_prcp_3mo_l2': self._lag(b['NP3'], 2),
                  'nb_runoff_3mo': b['NR3'][-1], 'nb_runoff_3mo_l1': self._lag(b['NR3'], 1), 'nb_runoff_3mo_l2': self._lag(b['NR3'], 2)})
        self.t += 1
        return f


def hybrid_fraction(fclim_m, P, Pbar_m, lam, fmax):
    """f = clip(fclim[m] + lam (P - Pbar[m]), 0, fmax), element-wise."""
    return np.clip(np.asarray(fclim_m, np.float64) + np.asarray(lam, np.float64) * (np.asarray(P, np.float64) - np.asarray(Pbar_m, np.float64)),
                   0.0, np.asarray(fmax, np.float64))


class LearnedWetland:
    """The trained file (wetland_learned_train.py --save) applied to a run's land points.

    I, J: the run's land points on the output grid. Cells of the file that are not land points of the run are dropped.
    """

    def __init__(self, path: str, I, J):
        import xgboost as xgb
        z = np.load(path, allow_pickle=False)
        dyn = tuple(str(x) for x in z['dyn_feats']); stat = tuple(str(x) for x in z['stat_feats'])
        if dyn != DYN_FEATS or stat != STAT_FEATS:
            raise ValueError(f'{path}: feature list differs from ibis.bgc.wetland_learned (file {dyn + stat})')
        pos = {(int(i), int(j)): p for p, (i, j) in enumerate(zip(np.asarray(I), np.asarray(J)))}
        keep = np.array([(int(i), int(j)) in pos for i, j in zip(z['cells_I'], z['cells_J'])])
        self.pts = np.array([pos[(int(i), int(j))] for i, j, k in zip(z['cells_I'], z['cells_J'], keep) if k], int)
        self.band = np.asarray(z['band'])[keep]; self.pbar = np.asarray(z['pbar'], np.float64)[:, keep]
        self.fclim = np.asarray(z['fclim'], np.float64)[:, keep]; self.fmax = np.asarray(z['fmax'], np.float64)[keep]
        self.lam = np.asarray(z['lam'], np.float64); self.stat = np.asarray(z['stat'], np.float64)[keep]
        self.models = []
        for js in z['models']:
            bst = xgb.Booster(); bst.load_model(bytearray(str(js), 'utf-8')); self.models.append(bst)
        self._xgb = xgb
        self.n_dropped = int((~keep).sum())

    def predict(self, feats: dict) -> np.ndarray:
        """P for the file's cells (len(self.pts),) from the month's DYN feature dict over all land points."""
        X = np.empty((len(self.pts), len(DYN_FEATS) + len(STAT_FEATS)), np.float32)
        for c, k in enumerate(DYN_FEATS): X[:, c] = feats[k][self.pts]
        X[:, len(DYN_FEATS):] = self.stat
        P = np.full(len(self.pts), np.nan)
        for b, bst in enumerate(self.models):
            sel = self.band == b
            if sel.any(): P[sel] = np.clip(bst.predict(self._xgb.DMatrix(X[sel])), 0.0, 1.0)
        return P

    def fraction(self, imonth: int, feats: dict, f_base) -> np.ndarray:
        """The month's inundated fraction over all land points: the file's cells from the hybrid, every other land point
        from `f_base` (the WAD2M climatology of the month, as the default path)."""
        P = self.predict(feats)
        f = np.array(f_base, np.float64, copy=True)
        f[self.pts] = hybrid_fraction(self.fclim[imonth], P, self.pbar[imonth], self.lam[self.band], self.fmax)
        return np.clip(f, 0.0, 1.0)
