#!/usr/bin/env python3
"""Is the bare Arctic a cold-start failure, or can the model not support it?

At the dead high-latitude cells `plai` sits at EXACTLY 0.010 per PFT -- the
`xminlai` floor -- with no real leaf area. `both_fix` was a cold start, so
vegetation had to bootstrap from that floor: warm cells escaped, cold cells
never did. The vmax experiment showed the floor is escapable when
productivity is raised (RU-Che LAI 0.07 -> 1.01), which points at a bistable
vegetation state rather than a hard climatic limit.

This tests it directly. Seed every cell with MODIS annual-maximum LAI, keep
the UNCHANGED parameters, and run 20 years with dynaveg:

  persists -> the equilibrium supports Arctic vegetation and the bare state
              is a cold-start artifact; the fix is initialisation, not physics
  collapses -> the model genuinely cannot hold vegetation there, and no
              vmax/specla calibration will fix the Arctic carbon input

Reported as the LAI trajectory, because a slow decay and a fast collapse mean
different things.

Usage:
  ibis-python verify/calibration/arctic_seed_test.py --nyear 20
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import xarray as xr

REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port")
sys.path.insert(0, f"{REPO}/verify/calibration")

import jax
import jax.numpy as jnp

from ibis.dimensions import GridConfig
from ibis.io.restart import load_state
from ibis.physics.prescribed_lai import apply_prescribed_lai
from ibis.state.model_state import create_ibis_state

from stage1_baseline import load_sites, modis_canopy_lai, MASK
from site_transient import build_context, one_year

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart",
                    default=f"{REPO}/verify_out_stage1/site_transient_1996.npz")
    ap.add_argument("--nyear", type=int, default=20)
    ap.add_argument("--year0", type=int, default=1997)
    ap.add_argument("--out", default=f"{REPO}/verify_out_stage1")
    a = ap.parse_args()

    df = load_sites(1990, 2010)
    m = np.array(xr.open_dataset(MASK).landfrac.values)
    fl = np.where(m.ravel() > 0)[0]
    pos = {g: p for p, g in enumerate(fl)}
    df["pos"] = [pos.get(i * 720 + j, -1) for i, j in zip(df.i, df.j)]
    df = df[df.pos >= 0].reset_index(drop=True)
    ctx = build_context(df)
    tmpl = create_ibis_state(GridConfig(npoi=len(df), nbedrock=3))

    state, _ = load_state(a.restart, tmpl)
    lai_modis = modis_canopy_lai(df.i.values, df.j.values)
    lai_before = np.asarray(state.vegetation.plai).sum(axis=1)
    state = apply_prescribed_lai(state, jnp.array(lai_modis),
                                 ctx["pft"].specla)
    lai_seeded = np.asarray(state.vegetation.plai).sum(axis=1)
    log.info("seeded LAI: model %.2f -> MODIS %.2f (mean over %d cells)",
             lai_before.mean(), lai_seeded.mean(), len(df))

    ARCTIC = df.lat > 66.0
    BOREAL = (df.lat > 55) & (df.lat <= 66)
    log.info("  arctic cells (>66N): %d   boreal (55-66N): %d",
             int(ARCTIC.sum()), int(BOREAL.sum()))

    rows = []
    key = jax.random.PRNGKey(0)
    for n in range(a.nyear):
        yr = a.year0 + (n % 10)
        state, key, d = one_year(state, key, yr, ctx, 1)
        lai = d["lai"]
        gpp = d["aygpptot"]
        rows.append(dict(n=n + 1,
                         lai_arctic=float(np.nanmean(lai[ARCTIC.values])),
                         lai_boreal=float(np.nanmean(lai[BOREAL.values])),
                         lai_all=float(np.nanmean(lai)),
                         gpp_arctic=float(np.nanmean(gpp[ARCTIC.values])),
                         gpp_all=float(np.nanmean(gpp))))
        log.info("  yr %2d: LAI arctic=%.3f boreal=%.2f all=%.2f | "
                 "GPP arctic=%.4f", n + 1, rows[-1]["lai_arctic"],
                 rows[-1]["lai_boreal"], rows[-1]["lai_all"],
                 rows[-1]["gpp_arctic"])

    t = pd.DataFrame(rows)
    t.to_csv(f"{a.out}/arctic_seed_trajectory.csv", index=False)

    out = df.copy()
    out["lai_modis"] = lai_modis
    out["lai_before"] = lai_before
    out["lai_final"] = d["lai"]
    out["gpp_final"] = d["aygpptot"]
    out.to_csv(f"{a.out}/arctic_seed_sites.csv", index=False)

    print(f"\n{'='*70}\nARCTIC (>66N) per cell: seeded -> after {a.nyear} yr")
    q = out[ARCTIC.values].sort_values("lat")
    print(q[["site", "lat", "gpp_obs", "lai_modis", "lai_before",
             "lai_final", "gpp_final"]].round(3).to_string(index=False))
    print(f"\nTRAJECTORY (mean LAI)")
    print(t[["n", "lai_arctic", "lai_boreal", "lai_all", "gpp_arctic"]]
          .round(3).to_string(index=False))
    a0 = t.lai_arctic.iloc[0]; a1 = t.lai_arctic.iloc[-1]
    seed = float(np.nanmean(lai_seeded[ARCTIC.values]))
    print(f"\n  Arctic LAI: seeded {seed:.3f} -> yr1 {a0:.3f} -> "
          f"yr{a.nyear} {a1:.3f}   ({100*(a1-seed)/max(seed,1e-9):+.0f}%)")
    print("  VERDICT: " + ("collapses back toward the xminlai floor"
                           if a1 < 0.25 * seed else
                           "persists -- bare state was a cold-start artifact"
                           if a1 > 0.75 * seed else "partially retained"))


if __name__ == "__main__":
    main()
