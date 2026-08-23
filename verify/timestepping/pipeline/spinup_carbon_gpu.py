"""Carbon spin-up with the COMPLETE chain (docs/FIRE.md; see run_spinup_gswp3).

The archived 200-year spin-ups ran `biomass_update` but never `soilbgc`, so
litter and soil carbon are identically zero in all of them. This run starts
from that state -- its biomass is genuinely spun up -- and fills the pools that
were never computed, using `loops_gswp3.run_spinup_gswp3`, which contains both
halves of the chain.

Cell order comes from `build_global_cell_grid`, verified to reproduce the
checkpoint's 88484 cells exactly. Do NOT derive cells from surta.nc directly:
the mask has 95333 points in north->south order while the state is
south->north, and truncating it misplaces 99.7% of cells silently.

Usage:
    PYTHONPATH=. /home/hangkai/.venvs/ibis-gpu/bin/python -u -m \
        verify.timestepping.pipeline.spinup_carbon_gpu --years 1901 1950
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python-port"))

from verify.timestepping.pipeline.global_gswp3_test import (
    build_global_cell_grid, latlon_to_grid_idx,
)
# `global_gswp3_test` enables x64 at import time. `run_spinup_gswp3` runs in
# float32 (it casts the state internally), and soilbgc would then return
# float64 and break the lax.scan carry with a dtype mismatch. Turn x64 back off
# AFTER that import, before any array is built.
jax.config.update("jax_enable_x64", False)

from ibis.constants import PhysicalConstants
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.io.physics_params import build_sim_params
from ibis.state.vegetation import VegetationState
from ibis.timestepping.loops_gswp3 import run_spinup_gswp3

CKPT_IN = ("/shared/nas2/Hangkai/ibis/verify_outputs/"
           "spinup_200yr_0p5deg_v6_flux_clamp/checkpoints/gpu_state_after_200.pkl")
OUT = Path("/shared/nas2/Hangkai/ibis/verify_outputs/spinup_carbon_v7_soilbgc")


def migrate(st, npoi):
    """Add the F1 accumulator fields absent from the pre-F1 checkpoint."""
    v = st.vegetation
    if hasattr(v, "nytimes"):
        return st
    f = {fd.name: getattr(v, fd.name)
         for fd in VegetationState.__dataclass_fields__.values() if hasattr(v, fd.name)}
    f["nytimes"] = jnp.zeros(())
    f["nfirestep"] = jnp.zeros((npoi,), dtype=jnp.float32)
    f["sdry"] = jnp.zeros((npoi,), dtype=jnp.float32)
    return eqx.tree_at(lambda s: s.vegetation, st, VegetationState(**f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[1901, 1950])
    ap.add_argument("--nsoilcall", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=10)
    a = ap.parse_args()

    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"backend {jax.default_backend()} {jax.devices()}", flush=True)

    t0 = time.time()
    with open(CKPT_IN, "rb") as f:
        st = pickle.load(f)["jax_state"]
    npoi = int(np.asarray(st.carbon.totlit).shape[0])
    st = migrate(st, npoi)
    st = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, "dtype") and x.dtype == jnp.float64 else x, st)
    print(f"loaded+migrated {npoi} cells in {time.time()-t0:.0f}s", flush=True)

    cells = build_global_cell_grid(lat_step=0.5, lon_step=0.5)
    assert len(cells) == npoi, f"cell grid {len(cells)} != state {npoi}"
    lat = np.array([c[1] for c in cells], dtype=np.float64)
    lon = np.array([c[2] for c in cells], dtype=np.float64)
    idx = [latlon_to_grid_idx(la, lo) for la, lo in zip(lat, lon)]
    li = np.array([i[0] for i in idx]); lo_i = np.array([i[1] for i in idx])
    print(f"cells verified against build_global_cell_grid; lat {lat.min()}..{lat.max()}",
          flush=True)

    params = build_sim_params(npoi, jnp.asarray(np.deg2rad(lat).astype(np.float32)), None)
    consts = PhysicalConstants()
    reader = GSWP3Reader("/shared/nas2/Min/GSWP3")

    log = OUT / "carbon_spinup_log.csv"
    with open(log, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["year", "totalit", "totlit", "totcsoi", "falll", "elapsed_s"])

    t0 = time.time()

    def on_year(year, s):
        c = s.carbon
        row = [year, float(c.totalit.mean()), float(c.totlit.mean()),
               float(c.totcsoi.mean()), float(c.falll.mean()), round(time.time()-t0)]
        with open(log, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        print(f"  y{year} totalit={row[1]:.4f} totlit={row[2]:.4f} "
              f"totcsoi={row[3]:.4f} falll={row[4]:.4f}  [{row[5]}s]", flush=True)
        # Checkpoint periodically, and always keep the latest: an earlier run
        # in this project crashed at the end and discarded 40 minutes of work.
        if year % a.ckpt_every == 0:
            with open(OUT / "checkpoints" / f"state_y{year}.pkl", "wb") as fp:
                pickle.dump({"year": year, "jax_state": s}, fp)
        with open(OUT / "checkpoints" / "state_latest.pkl", "wb") as fp:
            pickle.dump({"year": year, "jax_state": s}, fp)

    years = list(range(a.years[0], a.years[1] + 1))
    print(f"running {len(years)} years {years[0]}-{years[-1]}, "
          f"nsoilcall={a.nsoilcall}", flush=True)
    st, _ = run_spinup_gswp3(st, consts, params, reader, li, lo_i,
                             np.deg2rad(lat).astype(np.float32),
                             jax.random.PRNGKey(0), 3600.0, years,
                             nband=2, nsoilcall=a.nsoilcall, on_year_end=on_year)
    print(f"DONE {len(years)} years in {(time.time()-t0)/60:.1f} min -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
