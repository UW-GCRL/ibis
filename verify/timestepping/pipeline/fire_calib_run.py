"""Produce fire accumulators over the global cell grid, for F6 calibration.

Reuses the GSWP3 machinery proven by the 46-year F77<->JAX verification
(`global_gswp3_test.py`) but runs JAX only -- this is not a verification run,
it exists to generate the per-cell inputs that `ml.fire_calibration` fits
against GFED5.

Usage:
    PYTHONPATH=. /shared/nas2/Hangkai/ibis-env/ibis-python -u -m \
        verify.timestepping.pipeline.fire_calib_run --years 2001 2002 --out FILE
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python-port"))

from verify.timestepping.pipeline.global_gswp3_test import (
    build_global_cell_grid, diurnal_disaggregate, make_batch_state,
    read_gswp3_month,
)
from verify.timestepping.verify_lsxmain import setup_jax_state
from ibis.constants import PhysicalConstants
from ibis.timestepping.loops import default_sim_params, timestep_body

DTIME = 3600.0            # the pipeline steps hourly
NSTEP_DAY = 24
NPOI = 5


def run_batch_accumulators(cells, years):
    """Run one batch of cells through `years`; return last year's accumulators."""
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = PhysicalConstants()
    params = default_sim_params(len(cells))
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    lats = np.array([c[1] for c in cells])
    step = eqx.filter_jit(timestep_body)

    for year in years:
        # F77 stats.f:956 resets the annual counters at the year boundary.
        state = eqx.tree_at(
            lambda s: (s.vegetation.nytimes, s.vegetation.nfirestep,
                       s.vegetation.sdry),
            state,
            (jnp.zeros(()), jnp.zeros_like(state.vegetation.nfirestep),
             jnp.zeros_like(state.vegetation.sdry)),
        )
        jday = 0
        for month in range(1, 13):
            gd = read_gswp3_month(year, month, cells)
            ndays = np.asarray(gd["tas"]).shape[0]
            for d in range(ndays):
                jday += 1
                day_data = {k: np.asarray(gd[k])[d, :] for k in gd}
                for h in range(NSTEP_DAY):
                    atm = diurnal_disaggregate(day_data, float(h), jday, lats)
                    state = eqx.tree_at(
                        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                                   s.atm.coszen, s.atm.solad, s.atm.solai,
                                   s.atm.fira, s.atm.raina, s.atm.snowa),
                        state,
                        tuple(jnp.asarray(atm[k]) for k in
                              ("ta", "qa", "ua", "psurf", "coszen", "solad",
                               "solai", "fira", "raina", "snowa")),
                    )
                    state = step(state, consts, params, DTIME)

    v, c = state.vegetation, state.carbon
    return dict(
        nfirestep=np.asarray(v.nfirestep), sdry=np.asarray(v.sdry),
        nytimes=float(v.nytimes), firefac=np.asarray(v.firefac),
        totlit=np.asarray(c.totlit), totalit=np.asarray(c.totalit),
        cbiol=np.asarray(v.cbiol), plai=np.asarray(v.plai),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2001, 2002])
    ap.add_argument("--batch-start", type=int, default=0)
    ap.add_argument("--batch-end", type=int, default=None)
    ap.add_argument("--out", default="/tmp/fire_calib_accum.npz")
    a = ap.parse_args()

    cells = build_global_cell_grid(lat_step=10.0, lon_step=20.0)
    nb = (len(cells) + NPOI - 1) // NPOI
    be = a.batch_end if a.batch_end is not None else nb
    print(f"{len(cells)} cells, {nb} batches, running [{a.batch_start},{be}) "
          f"years {a.years}", flush=True)

    out, t0 = {k: [] for k in
               ("nfirestep", "sdry", "firefac", "totlit", "totalit")}, time.time()
    out["cbiol"], out["plai"], names, lat, lon = [], [], [], [], []
    nyt = None

    for b in range(a.batch_start, be):
        bc = cells[b * NPOI:(b + 1) * NPOI]
        if not bc:
            continue
        # `make_batch_state` always builds NPOI=5 cells from 5 synthetic
        # templates, so a short final batch (117 = 23*5 + 2) produced a
        # 5-vs-2 broadcast error. Pad, run, then trim back.
        ncell = len(bc)
        bc_run = bc + [bc[-1]] * (NPOI - ncell)
        r = run_batch_accumulators(bc_run, a.years)
        if ncell < NPOI:
            r = {k: (v[:ncell] if isinstance(v, np.ndarray) and v.ndim >= 1
                     else v) for k, v in r.items()}
        nyt = r["nytimes"]
        for k in ("nfirestep", "sdry", "firefac", "totlit", "totalit",
                  "cbiol", "plai"):
            out[k].append(r[k])
        names += [c[0] for c in bc]
        lat += [c[1] for c in bc]
        lon += [c[2] for c in bc]
        el = time.time() - t0
        print(f"  batch {b+1}/{be}  {el/60:.1f} min elapsed  "
              f"fseason={np.round(r['nfirestep']/max(r['nytimes'],1),3)}", flush=True)
        # Checkpoint every batch: the first attempt at this run crashed on the
        # final batch and discarded 40 minutes of completed work.
        np.savez(a.out, nytimes=nyt, cell=np.array(names),
                 lat=np.array(lat), lon=np.array(lon),
                 **{k: np.concatenate(v, axis=0) for k, v in out.items()})

    np.savez(a.out, nytimes=nyt, cell=np.array(names),
             lat=np.array(lat), lon=np.array(lon),
             **{k: np.concatenate(v, axis=0) for k, v in out.items()})
    print(f"saved -> {a.out}  ({len(names)} cells, {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
