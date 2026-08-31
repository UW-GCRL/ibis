"""Is IBIS's tropical heterotrophic respiration too high?

METHANE.md 5o eliminated wetland extent as the source of the tropical methane
excess: inundated area is 55.6% tropical while our flux is 76.7% tropical, and
the area is faithful to WAD2M. That leaves Rh and r_ch4.

Two different quantities matter and are easy to conflate:

  * GLOBAL Rh -- comparable to published global soil-respiration budgets;
  * WETLAND Rh, weighted by inundated fraction -- what the methane flux
    actually integrates. A model can have a fine global Rh and still put too
    much of it under tropical water.

Both are reported here, by latitude band.

    python verify/methane/rh_audit.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-port"))

from verify.timestepping.pipeline.global_gswp3_test import (  # noqa: E402
    build_global_cell_grid, latlon_to_grid_idx,
)

import jax  # noqa: E402
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp  # noqa: E402

from ibis.constants import PhysicalConstants  # noqa: E402
from ibis.io.gswp3_reader import GSWP3Reader  # noqa: E402
from ibis.io.physics_params import build_sim_params  # noqa: E402
from ibis.io.wad2m import read_wad2m_climatology  # noqa: E402
from ibis.timestepping.loops_gswp3 import run_month_gswp3  # noqa: E402

DEFAULT_STATE = ("/shared/nas2/Hangkai/ibis/verify_outputs/"
                 "spinup_bedrock_v9_3lay/checkpoints/state_latest.pkl")
DEFAULT_WAD2M = ("/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/"
                 "WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc")
NDAYPM = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--wad2m", default=DEFAULT_WAD2M)
    ap.add_argument("--gswp3", default="/shared/nas2/Min/GSWP3")
    ap.add_argument("--year", type=int, default=1901)
    ap.add_argument("--nsoilcall", type=int, default=10)
    a = ap.parse_args()

    ck = pickle.load(open(a.state, "rb"))
    st = ck["jax_state"]
    key = (jnp.asarray(ck["key"], dtype=jnp.uint32) if ck.get("key") is not None
           else jax.random.PRNGKey(0))

    cells = build_global_cell_grid(lat_step=0.5, lon_step=0.5)
    lat = np.array([c[1] for c in cells])
    lon = np.array([c[2] for c in cells])
    idx = [latlon_to_grid_idx(la, lo) for la, lo in zip(lat, lon)]
    li = np.array([i[0] for i in idx]); lo = np.array([i[1] for i in idx])
    latr = np.deg2rad(lat).astype(np.float32)

    mlat = np.arange(-90.0, 90.0, 0.5); mlon = np.arange(-180.0, 180.0, 0.5)
    wli = np.rint((lat - mlat[0]) / 0.5).astype(int).clip(0, 359)
    wlo = np.rint((lon - mlon[0]) / 0.5).astype(int).clip(0, 719)
    clim = read_wad2m_climatology(a.wad2m, wli, wlo, mlat, mlon)

    params = build_sim_params(len(cells), jnp.asarray(latr), None)
    reader = GSWP3Reader(a.gswp3)
    R = 6371.0; h = np.deg2rad(0.5)
    area = (R ** 2 * np.deg2rad(0.5)
            * (np.sin(np.deg2rad(lat) + h / 2)
               - np.sin(np.deg2rad(lat) - h / 2))) * 1e6      # m2

    # summary.ayco2mic accumulates Rh EVERY DAY across the year, so it is the
    # true annual integral. Sampling tco2mic at each month's end and scaling by
    # days in month -- which is what evaluate_global.py does, and hence what
    # every methane number in the docs rests on -- is a snapshot method. Both
    # are computed here so the difference is measured rather than assumed.
    ay0 = np.asarray(st.summary.ayco2mic).copy()
    print(f"  ayco2mic at start: max {ay0.max():.3e} (should be ~0 after the "
          f"year-end reset)", flush=True)
    rh_m, npp_m, gpp_m = [], [], []
    for m in range(12):
        st, key, _ = run_month_gswp3(st, PhysicalConstants(), params, reader,
                                     li, lo, latr, a.year, m, key, 3600.0,
                                     False, 2, a.nsoilcall)
        rh_m.append(np.asarray(st.vegetation.tco2mic) * 12.0e-3 * 86400.0)
        npp_m.append(np.asarray(st.vegetation.tnpptot) * 12.0e-3 * 86400.0)
        gpp_m.append(np.asarray(st.vegetation.tgpptot) * 12.0e-3 * 86400.0)
        print(f"  month {m+1:2d}/12", flush=True)
    rh = np.stack(rh_m)                                        # kgC/m2/d
    npp = np.stack(npp_m); gpp = np.stack(gpp_m)
    ay = np.asarray(st.summary.ayco2mic) - ay0                 # kgC/m2/yr

    bands = {"30S-30N": np.abs(lat) <= 30, "30-60N": (lat > 30) & (lat <= 60),
             "60-90N": lat > 60, "30-60S": (lat < -30) & (lat >= -60)}

    # --- whole-cycle context ------------------------------------------------
    # Rh cannot be judged alone. Soil carbon is at equilibrium in this state
    # (drift -0.07 %/yr over the last spin-up year), so Rh ~= carbon input, and
    # an Rh that is too high with a matching NPP means the productivity is the
    # cause rather than the decomposition.
    pg = lambda x: ((x * NDAYPM[:, None]).sum(axis=0) * area).sum() / 1e12
    g, n, r = pg(gpp), pg(npp), pg(rh)
    r_true = (ay * area).sum() / 1e12
    print(f"\n  ANNUAL Rh, two ways")
    print(f"    daily accumulation (summary.ayco2mic)  {r_true:7.1f} PgC/yr")
    print(f"    month-end snapshots x days in month    {r:7.1f} PgC/yr"
          f"   ({(r/max(r_true,1e-9)-1)*100:+.1f}%)")
    print(f"    The second is what evaluate_global.py samples.")
    print(f"\n  GLOBAL CARBON CYCLE, PgC/yr")
    print(f"  {'flux':6} {'IBIS':>8} {'published':>22}")
    print(f"  {'GPP':6} {g:8.1f} {'120-130 (FLUXCOM/TRENDY)':>22}")
    print(f"  {'NPP':6} {n:8.1f} {'55-60':>22}")
    print(f"  {'Rh':6} {r:8.1f} {'49-58':>22}")
    print(f"\n  NPP/GPP unavailable: tnpp, tgpp, tnpptot, tgpptot and "
          f"tco2root are\n  never assigned anywhere in production code -- "
          f"dead diagnostic fields, so\n  Rh cannot be cross-checked against "
          f"the model's own productivity here.")

    # --- global Rh, PgC/yr -------------------------------------------------
    ann = (rh * NDAYPM[:, None]).sum(axis=0)                   # kgC/m2/yr
    tot = (ann * area).sum() / 1e12                            # PgC/yr
    print(f"\n  GLOBAL Rh {tot:.1f} PgC/yr")
    print(f"  {'band':10} {'PgC/yr':>9} {'share':>8} {'kgC/m2/yr':>11}")
    for b, m in bands.items():
        v = (ann[m] * area[m]).sum() / 1e12
        print(f"  {b:10} {v:9.2f} {v/tot*100:7.1f}% "
              f"{(ann[m]*area[m]).sum()/area[m].sum():11.3f}")

    # --- wetland-weighted Rh: what methane actually sees -------------------
    wrh = (rh * clim * NDAYPM[:, None]).sum(axis=0)            # kgC/m2/yr * f
    wtot = (wrh * area).sum() / 1e12
    warea = (clim.mean(axis=0) * area).sum()
    print(f"\n  WETLAND-WEIGHTED Rh {wtot:.2f} PgC/yr over "
          f"{warea/1e12:.2f} Mkm2 mean inundated area")
    print(f"  {'band':10} {'PgC/yr':>9} {'share':>8} {'area share':>11} "
          f"{'kgC/m2/yr':>11}")
    for b, m in bands.items():
        v = (wrh[m] * area[m]).sum() / 1e12
        af = (clim[:, m].mean(axis=0) * area[m]).sum()
        print(f"  {b:10} {v:9.3f} {v/wtot*100:7.1f}% {af/warea*100:10.1f}% "
              f"{(wrh[m]*area[m]).sum()/max(af,1e-9):11.3f}")

    print("""
  Read the two share columns together. If the wetland-Rh share is far above the
  area share in the tropics, tropical wetland carbon is respiring faster than
  tropical wetland extent alone would give -- which is what would carry the
  76.7% tropical methane share reported in METHANE.md 5n.""")


if __name__ == "__main__":
    main()
