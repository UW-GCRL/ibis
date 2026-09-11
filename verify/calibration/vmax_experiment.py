#!/usr/bin/env python3
"""Does restoring vmax to IBIS defaults fix the high-latitude GPP deficit?

`both_fix` applied a global `vmax x0.54` to bring global GPP onto FLUXCOM.
Checked against the calibration harness's own literature bounds, that pushed
NINE of twelve PFTs below their physical minimum -- boreal conifer to 13.5
against a 15-40 umol range. Meanwhile GPP north of 60N is 3-8x too low and
LAI is 0.33x (60-70N) / 0.13x (70-90N) of MODIS.

This runs two arms from the 1996 transient restart, cycling 1997-2006 forcing
so leaf area re-equilibrates to the new vmax (tauleaf is 1-2.5 yr, so ~20
years is ample for LAI; woody pools are not expected to converge and that is
stated rather than hidden):

  base     -- vmax as `both_fix` left it
  restore  -- vmax / 0.54, i.e. the IBIS defaults

Reported by latitude band against towers AND FLUXCOM, because the two
disagree by ~0.75x at tower cells for representativeness reasons and only the
gridded comparison constrains the global mean.

Usage:
  ibis-python verify/calibration/vmax_experiment.py --cycles 2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port")
sys.path.insert(0, f"{REPO}/verify/calibration")

import equinox as eqx
import jax
import jax.numpy as jnp

from ibis.dimensions import GridConfig
from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state

from stage1_baseline import load_sites, LAT, MASK
from site_transient import build_context, one_year

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
FLUXCOM = "/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc"


def scale_vmax(ctx, factor):
    sp = ctx["sp"]
    new = sp.stomata_params.vmax_pft * factor
    ctx = dict(ctx)
    ctx["sp"] = eqx.tree_at(lambda p: p.stomata_params.vmax_pft, sp, new)
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart",
                    default=f"{REPO}/verify_out_stage1/site_transient_1996.npz")
    ap.add_argument("--year0", type=int, default=1997)
    ap.add_argument("--nyear", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--out", default=f"{REPO}/verify_out_stage1")
    a = ap.parse_args()

    df = load_sites(1990, 2010)
    m = np.array(xr.open_dataset(MASK).landfrac.values)
    fl = np.where(m.ravel() > 0)[0]
    pos = {g: p for p, g in enumerate(fl)}
    df["pos"] = [pos.get(i * 720 + j, -1) for i, j in zip(df.i, df.j)]
    df = df[df.pos >= 0].reset_index(drop=True)
    ctx0 = build_context(df)
    tmpl = create_ibis_state(GridConfig(npoi=len(df), nbedrock=3))

    o = xr.open_dataset(FLUXCOM)
    gmm = np.asarray(o["gpp"].mean(dim="time")) * 365.25 / 1000.0
    ola = np.asarray(o["lat"]); olo = np.asarray(o["lon"])
    df["fluxcom"] = gmm[[np.argmin(np.abs(ola - x)) for x in df.lat],
                        [np.argmin(np.abs(olo - x)) for x in df.lon]]

    res = {}
    for tag, fac in (("base", 1.0), ("restore", 1.0 / 0.54)):
        ctx = scale_vmax(ctx0, fac)
        v = np.asarray(ctx["sp"].stomata_params.vmax_pft) * 1e6
        log.info("=== ARM %s  vmax boreal_con=%.1f  c3grass=%.1f umol ===",
                 tag, v[5], v[11])
        state, _ = load_state(a.restart, tmpl)
        key = jax.random.PRNGKey(0)
        for c in range(a.cycles):
            for n in range(a.nyear):
                state, key, d = one_year(state, key, a.year0 + n, ctx, 1)
            log.info("  cycle %d: GPP=%.3f LAI=%.2f", c + 1,
                     float(np.nanmean(d["aygpptot"])), float(np.nanmean(d["lai"])))
        acc = []
        for n in range(a.nyear):
            state, key, d = one_year(state, key, a.year0 + n, ctx, 1)
            acc.append(d)
        res[tag] = {k: np.mean([x[k] for x in acc], axis=0)
                    for k in ("aygpptot", "ayaet", "lai", "ayco2mic")}

    out = df.copy()
    for tag in res:
        out[f"gpp_{tag}"] = res[tag]["aygpptot"]
        out[f"lai_{tag}"] = res[tag]["lai"]
        out[f"et_{tag}"] = res[tag]["ayaet"]
    out.to_csv(f"{a.out}/vmax_experiment.csv", index=False)

    out["band"] = pd.cut(out.lat, [-90, 30, 50, 60, 70, 90],
                         labels=["<30N", "30-50N", "50-60N", "60-70N", "70-90N"])
    v = out[(out.gpp_obs > 0.05) & np.isfinite(out.fluxcom) & (out.fluxcom > 0)]
    print(f"\n{'='*84}\nGPP BY BAND   (n, tower, FLUXCOM, model base, model restore)")
    print(f"{'band':>8}{'n':>4}{'tower':>8}{'FLXCOM':>8}{'base':>8}{'restore':>9}"
          f"{'b/tow':>7}{'r/tow':>7}{'b/flx':>7}{'r/flx':>7}")
    for b, g in v.groupby("band", observed=True):
        print(f"{str(b):>8}{len(g):>4}{g.gpp_obs.mean():>8.3f}{g.fluxcom.mean():>8.3f}"
              f"{g.gpp_base.mean():>8.3f}{g.gpp_restore.mean():>9.3f}"
              f"{(g.gpp_base/g.gpp_obs).median():>7.2f}"
              f"{(g.gpp_restore/g.gpp_obs).median():>7.2f}"
              f"{(g.gpp_base/g.fluxcom).median():>7.2f}"
              f"{(g.gpp_restore/g.fluxcom).median():>7.2f}")
    print(f"{'ALL':>8}{len(v):>4}{v.gpp_obs.mean():>8.3f}{v.fluxcom.mean():>8.3f}"
          f"{v.gpp_base.mean():>8.3f}{v.gpp_restore.mean():>9.3f}"
          f"{(v.gpp_base/v.gpp_obs).median():>7.2f}"
          f"{(v.gpp_restore/v.gpp_obs).median():>7.2f}"
          f"{(v.gpp_base/v.fluxcom).median():>7.2f}"
          f"{(v.gpp_restore/v.fluxcom).median():>7.2f}")
    print(f"\nLAI BY BAND      base -> restore")
    for b, g in v.groupby("band", observed=True):
        print(f"  {str(b):>8}  {g.lai_base.mean():.2f} -> {g.lai_restore.mean():.2f}")
    print(f"\nBOREAL/ARCTIC SITES (>55N), per site:")
    bs = v[v.lat > 55].sort_values("lat")
    print(bs[["site", "igbp", "lat", "gpp_obs", "fluxcom", "gpp_base",
              "gpp_restore", "lai_base", "lai_restore"]].round(2).to_string(index=False))
    print(f"\nWrote {a.out}/vmax_experiment.csv")


if __name__ == "__main__":
    main()
