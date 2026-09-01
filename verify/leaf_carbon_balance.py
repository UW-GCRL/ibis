"""Is leaf carbon consistent with allocation? The fork in docs/GPP.md.

LAI is 2.7x MODIS and `plai = cbiol * specla` holds exactly, so one of the two
factors is too large. They pull in opposite directions on plausibility: leaf
carbon of 0.147 kgC/m2 is high but publishable, and specla of 20 m2/kgC is if
anything LOW. This decides which.

At equilibrium the leaf pool satisfies

    cbiol = aleaf * aynpp * tauleaf

If that holds, allocation is self-consistent and `specla` is the term carrying
the error. If cbiol comes out ABOVE the prediction, the leaf pool is being fed
or retained beyond what allocation and turnover allow, and the fix is upstream
-- in which case tuning specla would bury it.

`aynpp` cannot be read from a restart: run_global resets the accumulators at
line 700 and writes the restart at line 724, so it is always zero there. This
runs a year forward from the restart and inspects the accumulator directly,
before dynaveg consumes it.

    python verify/leaf_carbon_balance.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python-port"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp  # noqa: E402

from scripts.run_global import (  # noqa: E402
    _load_land_mask_from_file, _subset_land_mask, _load_soil, _apply_soil,
)
from ibis.constants import PhysicalConstants  # noqa: E402
from ibis.dimensions import GridConfig  # noqa: E402
from ibis.io.gswp3_reader import GSWP3Reader  # noqa: E402
from ibis.io.physics_params import build_all_params  # noqa: E402
from ibis.io.restart import load_state  # noqa: E402
from ibis.state.model_state import create_ibis_state  # noqa: E402
from ibis.timestepping.loops_gswp3 import run_month_gswp3  # noqa: E402
from ibis.timestepping.calendar import is_leap_year  # noqa: E402

RESTART = ("/shared/nas2/Hangkai/ibis/verify_outputs/spinup_cold_v10/"
           "restart/ibis_restart_2200.npz")
R = 6371.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", default=RESTART)
    ap.add_argument("--year", type=int, default=1901)
    ap.add_argument("--nsoilcall", type=int, default=10)
    a = ap.parse_args()

    # Replicate run_global's grid EXACTLY: it enumerates land cells with
    # argwhere(mask > 0) in row-major order from the north, which is a
    # different ordering from verify/'s build_global_cell_grid.
    mask, lat, lon = _load_land_mask_from_file("/shared/nas2/Min/HWSD/surta.nc")
    (li_sub, lo_sub, lat_sub, lon_sub,
     gli, glo) = _subset_land_mask(mask, lat, lon, 90.0, -90.0, -180.0, 180.0)
    npoi = len(gli)
    lat_deg = lat_sub[li_sub].astype(np.float32)
    print(f"  grid: {npoi} land points")

    grid = GridConfig(nlon=720, nlat=360, npoi=npoi, xres=0.5, yres=0.5,
                      nbedrock=3)
    st, meta = load_state(a.restart, create_ibis_state(grid))
    print(f"  restart: {meta}")
    st = _apply_soil(st, _load_soil("/shared/nas2/Min/HWSD/soita.nc", gli, glo))

    lat_rad = jnp.asarray(np.deg2rad(lat_deg))
    params, pft, _ = build_all_params(npoi, lat_rad, None)
    reader = GSWP3Reader("/shared/nas2/Min/GSWP3")
    key = jax.random.PRNGKey(0)
    leap = is_leap_year(a.year)

    aynpp0 = np.asarray(st.summary.aynpp).copy()
    for m in range(12):
        st, key, _ = run_month_gswp3(st, PhysicalConstants(), params, reader,
                                     gli, glo, lat_deg, a.year, m, key, 3600.0,
                                     leap, 2, a.nsoilcall)
        print(f"    month {m+1:2d}/12", flush=True)

    aynpp = np.asarray(st.summary.aynpp) - aynpp0     # kgC/m2/yr, per PFT
    cbiol = np.asarray(st.vegetation.cbiol)
    plai = np.asarray(st.vegetation.plai)
    aleaf = np.asarray(pft.aleaf)
    tauleaf = np.asarray(pft.tauleaf)
    specla = np.asarray(pft.specla)

    h = np.deg2rad(0.5)
    area = (R**2 * np.deg2rad(0.5)
            * (np.sin(np.deg2rad(lat_deg)+h/2)
               - np.sin(np.deg2rad(lat_deg)-h/2))) * 1e6

    pred = aleaf[None, :] * np.maximum(aynpp, 0.0) * tauleaf[None, :]
    live = plai > 0.01

    def aw(x):        # area-weighted mean over live PFT-cells
        return float((x * area[:, None] * live).sum() / (area[:, None]*live).sum())

    print(f"\n  annual aynpp summed over PFTs: "
          f"{float((aynpp.sum(1)*area).sum()/1e12):.1f} PgC/yr")
    print(f"\n  {'quantity':40} {'value':>10}")
    print(f"  {'cbiol actual (kgC/m2, per live PFT-cell)':40} {aw(cbiol):10.4f}")
    print(f"  {'cbiol predicted = aleaf*aynpp*tauleaf':40} {aw(pred):10.4f}")
    r = aw(cbiol) / max(aw(pred), 1e-12)
    print(f"  {'ratio actual / predicted':40} {r:10.3f}")
    print(f"\n  {'implied LAI from predicted cbiol':40} "
          f"{aw(pred*specla[None,:]):10.3f}")
    print(f"  {'actual LAI (plai)':40} {aw(plai):10.3f}")
    print(f"  MODIS annual-max LAI, natural veg: 1.69")

    print("""
  READ: ratio ~1 means allocation and turnover explain the leaf pool, so specla
  carries the LAI error. Ratio >> 1 means the pool exceeds what allocation
  allows and the fix is upstream of specla.""")


if __name__ == "__main__":
    main()
