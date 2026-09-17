#!/usr/bin/env python3
"""Seasonal baseline on the fit's own 4,331 cells via one_year -- no netCDF.

The global monthly-output run is writer-bound (~16 min per model-year against
50 s without monthly output). The stratified sample matches the domain's band
shares within 1% (docs/CALIBRATION.md 10), and one_year now returns monthly
GPP directly, so this gives the same diagnostic in ~5 min on the GPU -- and
exercises exactly the code path the three-criterion fit will score.
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx
REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port"); sys.path.insert(0, f"{REPO}/verify/calibration")
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing
import site_transient; from site_transient import one_year
from seasonal_baseline import report
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32

ap = argparse.ArgumentParser()
ap.add_argument("--restart", required=True); ap.add_argument("--params", required=True)
ap.add_argument("--sel", required=True, help="best.npz carrying the sample indices")
ap.add_argument("--co2", type=float, default=389.8); ap.add_argument("--nyear", type=int, default=10)
ap.add_argument("--label", default="model")
a = ap.parse_args()

sel = np.load(a.sel)["sel"]
m = np.array(xr.open_dataset(MASK).landfrac.values) > 0
fl = np.where(m.ravel())[0]; i = (fl[sel] // 720).astype(np.int32); j = (fl[sel] % 720).astype(np.int32)
lat_rad = np.deg2rad(LATV[i]).astype(np.float32)
full, _ = load_state(a.restart, create_ibis_state(GridConfig(npoi=NPOI_FULL, nbedrock=3)))
cut = lambda x: x[sel] if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL else x
st = _cast_tree_f32(jax.tree.map(cut, full)); del full
sp, pft, _ = build_all_params(len(sel), lat_rad, load_physics_params(a.params))
years = list(range(2001, 2011))
ctx = dict(i=i, j=j, lat_rad=lat_rad, sp=sp, pft=pft, consts=PhysicalConstants(),
           reader=CachedForcing(GSWP3Reader(GSWP3), years, i, j))
site_transient.apply_co2 = lambda s, y, series=None: eqx.tree_at(lambda q: q.atm.co2conc, s, a.co2 * 1e-6)
key = jax.random.PRNGKey(0); acc = []; t0 = time.time()
for n in range(a.nyear):
    st, key, d = one_year(st, key, years[n % 10], ctx, 1)
    acc.append(d["gpp_monthly"])
    if n == 0: print(f"  year 1 done in {time.time()-t0:.0f}s; monthly GPP shape {d['gpp_monthly'].shape}", flush=True)
mod_m = np.mean(acc, axis=0)
print(f"{a.nyear} years, {len(sel)} cells, CO2 {a.co2} ppm  [{time.time()-t0:.0f}s]")
report(mod_m, i, j, a.label)
