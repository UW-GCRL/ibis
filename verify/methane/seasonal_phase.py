"""Why does global CH4 peak in June when observations say July-August?

Tests one hypothesis before any others: the early peak may be a CONSEQUENCE of
the high-latitude deficit rather than an independent error. Boreal wetlands
peak in late summer; if they contribute 1% instead of 8%, the global peak is
pulled toward the tropical cycle.

Reports the monthly cycle per band, each band's own peak month, and what the
global peak would be if the high latitudes were scaled to their observed 8%
share -- which separates "phase is wrong" from "the late-peaking band is
missing".
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-port"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp  # noqa: E402

from ibis.bgc.methane import methane_flux, biome_methane_params  # noqa: E402
from ibis.constants import PhysicalConstants  # noqa: E402
from ibis.io.gswp3_reader import GSWP3Reader  # noqa: E402
from ibis.io.physics_params import build_sim_params  # noqa: E402
from ibis.io.wad2m import read_wad2m_climatology  # noqa: E402
from ibis.io.restart import load_state  # noqa: E402
from ibis.state.model_state import create_ibis_state  # noqa: E402
from ibis.dimensions import GridConfig  # noqa: E402
from ibis.timestepping.loops_gswp3 import run_month_gswp3  # noqa: E402
from scripts.run_global import _load_land_mask_from_file, _subset_land_mask  # noqa: E402

NDAYPM = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
KGC_TO_TG = 16.04 / 12.011 / 1e9
R = 6371.0
MON = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="/shared/nas2/Hangkai/ibis/verify_outputs/"
                                       "both_fix/restart/ibis_restart_2200.npz")
    ap.add_argument("--year", type=int, default=1901)
    ap.add_argument("--nsoilcall", type=int, default=10)
    a = ap.parse_args()

    mask, glat, glon = _load_land_mask_from_file("/shared/nas2/Min/HWSD/surta.nc")
    _, _, _, _, li, lo = _subset_land_mask(mask, glat, glon, 90., -90., -180., 180.)
    npoi = len(li); lat = glat[li]; lon = glon[lo]
    latr = np.deg2rad(lat).astype(np.float32)
    st, _ = load_state(a.state, create_ibis_state(
        GridConfig(nlon=720, nlat=360, npoi=npoi, xres=.5, yres=.5, nbedrock=3)))

    mlat = np.arange(-90., 90., .5); mlon = np.arange(-180., 180., .5)
    clim = read_wad2m_climatology(
        "/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/"
        "WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc",
        np.rint((lat-mlat[0])/.5).astype(int).clip(0,359),
        np.rint((lon-mlon[0])/.5).astype(int).clip(0,719), mlat, mlon)

    params = build_sim_params(npoi, jnp.asarray(latr), None)
    p = biome_methane_params(lat)
    hsoi = jnp.asarray(np.asarray(st.soil.hsoi)[:6])
    reader = GSWP3Reader("/shared/nas2/Min/GSWP3")
    key = jax.random.PRNGKey(0)
    h = np.deg2rad(.5)
    area = (R**2*np.deg2rad(.5)*(np.sin(np.deg2rad(lat)+h/2)
                                 - np.sin(np.deg2rad(lat)-h/2)))*1e6

    bands = {"30S-30N": np.abs(lat) <= 30,
             "30-60N": (lat > 30) & (lat <= 60),
             "60-90N": lat > 60}
    monthly = {b: np.zeros(12) for b in bands}
    for m in range(12):
        st, key, _ = run_month_gswp3(st, PhysicalConstants(), params, reader,
                                     li, lo, latr, a.year, m, key, 3600.,
                                     False, 2, a.nsoilcall)
        f = np.asarray(methane_flux(
            jnp.asarray(clim[m]),
            jnp.asarray(np.asarray(st.vegetation.tco2mic)*12.0e-3*86400.0),
            jnp.asarray(st.soil.tsoi), jnp.asarray(st.soil.wsoi),
            jnp.asarray(st.soil.wisoi), hsoi, p))
        for b, msk in bands.items():
            monthly[b][m] = (f[msk]*area[msk]).sum()*NDAYPM[m]*KGC_TO_TG
        print(f"    month {m+1:2d}/12", flush=True)

    tot = sum(monthly[b] for b in bands)
    print(f"\n  Monthly CH4, Tg/yr-equivalent per month\n")
    print(f"  {'band':9} " + " ".join(f"{x:>5}" for x in MON) + f" {'peak':>5} {'share':>6}")
    for b in bands:
        v = monthly[b]
        print(f"  {b:9} " + " ".join(f"{x:5.1f}" for x in v)
              + f" {MON[int(np.argmax(v))]:>5} {v.sum()/tot.sum()*100:5.1f}%")
    print(f"  {'GLOBAL':9} " + " ".join(f"{x:5.1f}" for x in tot)
          + f" {MON[int(np.argmax(tot))]:>5}")

    # what if the high latitudes carried their observed 8% share?
    hi = monthly["60-90N"]; cur = hi.sum()/tot.sum()
    if cur > 0:
        scaled = tot - hi + hi*(0.08/cur)
        print(f"\n  If 60-90N were scaled from {cur*100:.1f}% to its observed 8%:")
        print(f"  {'GLOBAL':9} " + " ".join(f"{x:5.1f}" for x in scaled)
              + f" {MON[int(np.argmax(scaled))]:>5}")
        print(f"\n  observed global peak: Jul-Aug")


if __name__ == "__main__":
    main()
