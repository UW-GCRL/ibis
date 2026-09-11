#!/usr/bin/env python3
"""Stratified global cell sample for calibration.

Calibrating on the 171 FLUXNET cells would fit the tower NETWORK's biome and
climate distribution, not the world's: the towers are 24% temperate conifer
and 3% tropical, against a land surface that is the other way round, and they
sit in productive patches inside their cells (docs/CALIBRATION.md 1). A
stratified sample of the model's own domain avoids importing that bias.

Strata are latitude band x dominant PFT, sampled proportionally to land area
with a floor per stratum so rare biomes are not lost. Cells are drawn once
with a fixed seed and written to disk, so every parameter evaluation and every
arm of every experiment sees the identical sample -- otherwise a change in
score could be a change in the sample, which has already caught this project
out once (docs/CALIBRATION.md 3).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import xarray as xr

REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port")

MASK = "/shared/nas2/Min/HWSD/surta.nc"
BANDS = [(-90, -30), (-30, 0), (0, 30), (30, 50), (50, 60), (60, 70), (70, 90)]


def build_sample(n_target, restart, npoi_full=95333, nbedrock=3, seed=20260911,
                 floor=20):
    """Return indices into the land-point vector, stratified by band x PFT."""
    from ibis.dimensions import GridConfig
    from ibis.io.restart import load_state
    from ibis.state.model_state import create_ibis_state

    st, _ = load_state(restart, create_ibis_state(
        GridConfig(npoi=npoi_full, nbedrock=nbedrock)))
    plai = np.asarray(st.vegetation.plai)
    dom = np.argmax(plai, axis=1)
    dom[plai.sum(axis=1) < 0.05] = -1          # effectively bare

    m = np.array(xr.open_dataset(MASK).landfrac.values)
    fl = np.where(m.ravel() > 0)[0]
    lat = np.arange(89.75, -90, -0.5)[fl // 720]

    rng = np.random.default_rng(seed)
    strata = {}
    for lo, hi in BANDS:
        for p in range(-1, 12):
            idx = np.where((lat >= lo) & (lat < hi) & (dom == p))[0]
            if len(idx):
                strata[(lo, hi, p)] = idx

    total = sum(len(v) for v in strata.values())
    picked = []
    for k, idx in strata.items():
        want = max(floor, int(round(n_target * len(idx) / total)))
        want = min(want, len(idx))
        picked.append(rng.choice(idx, size=want, replace=False))
    sel = np.unique(np.concatenate(picked))
    return sel, lat[sel], dom[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default=f"{REPO}/verify_out_stage1/calib_cells.npz")
    a = ap.parse_args()
    sel, lat, dom = build_sample(a.n, a.restart)
    np.savez(a.out, idx=sel, lat=lat, dom=dom)
    print(f"sampled {len(sel)} cells -> {a.out}")
    print(f"{'band':>10} {'n':>6}   dominant-PFT counts")
    for lo, hi in BANDS:
        s = (lat >= lo) & (lat < hi)
        if not s.any():
            continue
        c = np.bincount(dom[s] + 1, minlength=13)
        top = ", ".join(f"{i-1}:{c[i]}" for i in np.argsort(c)[::-1][:4] if c[i])
        print(f"{f'{lo},{hi}':>10} {int(s.sum()):>6}   {top}")


if __name__ == "__main__":
    main()
