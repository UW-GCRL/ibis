"""Why is high-latitude CH4 1.0% of the global flux against a target of 8.0%?

M-lite+ is a product of factors:

    F = f_inund * r_CH4 * Rh * f_anaer * Q10 * unfrozen_gate * f_surv

so the deficit must live in one of them. This runs a year from a restart and
reports each factor's band mean, weighted the way the flux actually weights it,
so the responsible term is identified rather than guessed.

    python verify/methane/highlat_decompose.py --state <restart.npz>
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-port"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp  # noqa: E402

from ibis.bgc.methane import (  # noqa: E402
    MethaneParams, wetland_water_table, thawed_anaerobic_zone,
    anaerobic_substrate_fraction, surviving_fraction, wetland_flux_area)
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
R = 6371.0


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
    grid = GridConfig(nlon=720, nlat=360, npoi=npoi, xres=0.5, yres=0.5, nbedrock=3)
    st, _ = load_state(a.state, create_ibis_state(grid))

    mlat = np.arange(-90., 90., .5); mlon = np.arange(-180., 180., .5)
    wli = np.rint((lat - mlat[0]) / .5).astype(int).clip(0, 359)
    wlo = np.rint((lon - mlon[0]) / .5).astype(int).clip(0, 719)
    clim = read_wad2m_climatology(
        "/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/"
        "WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc", wli, wlo, mlat, mlon)

    params = build_sim_params(npoi, jnp.asarray(latr), None)
    reader = GSWP3Reader("/shared/nas2/Min/GSWP3")
    hsoi = jnp.asarray(np.asarray(st.soil.hsoi)[:6])
    p = MethaneParams()
    key = jax.random.PRNGKey(0)

    snap = []
    for m in range(12):
        st, key, _ = run_month_gswp3(st, PhysicalConstants(), params, reader,
                                     li, lo, latr, a.year, m, key, 3600.,
                                     False, 2, a.nsoilcall)
        snap.append(dict(rh=np.asarray(st.vegetation.tco2mic)*12.0e-3*86400.0,
                         tsoi=np.asarray(st.soil.tsoi),
                         wsoi=np.asarray(st.soil.wsoi),
                         wisoi=np.asarray(st.soil.wisoi)))
        print(f"    month {m+1:2d}/12", flush=True)

    bands = {"30S-30N": np.abs(lat) <= 30,
             "30-60N": (lat > 30) & (lat <= 60),
             "60-90N": lat > 60}
    h = np.deg2rad(0.5)
    area = (R**2*np.deg2rad(.5)*(np.sin(np.deg2rad(lat)+h/2)
                                 - np.sin(np.deg2rad(lat)-h/2)))*1e6

    # accumulate each factor, weighted by inundated area x days
    terms = {b: {k: 0.0 for k in
                 ("w","rh","f_anaer","q10","unfroz","f_surv","flux")}
             for b in bands}
    for m in range(12):
        s = snap[m]
        zwt = wetland_water_table(jnp.asarray(s["wsoi"]), jnp.asarray(s["wisoi"]), hsoi, p)
        tth, fth = thawed_anaerobic_zone(jnp.asarray(s["tsoi"]), zwt, hsoi)
        fa = np.asarray(anaerobic_substrate_fraction(zwt, p))
        fs = np.asarray(surviving_fraction(zwt, p))
        tth = np.asarray(tth)
        q10 = p.q10p ** ((tth - p.t_ref) / 10.0)
        unf = (tth >= 273.15).astype(float)
        rh = np.maximum(s["rh"], 0.0)
        w = np.asarray(wetland_flux_area(jnp.asarray(clim[m]))) * area * NDAYPM[m]
        flux = w * rh * fa * q10 * unf * fs * p.r_ch4
        for b, msk in bands.items():
            W = w[msk].sum(); t = terms[b]
            t["w"] += W
            t["rh"] += (rh[msk]*w[msk]).sum()
            t["f_anaer"] += (fa[msk]*w[msk]).sum()
            t["q10"] += (q10[msk]*w[msk]).sum()
            t["unfroz"] += (unf[msk]*w[msk]).sum()
            t["f_surv"] += (fs[msk]*w[msk]).sum()
            t["flux"] += flux[msk].sum()

    tot = sum(terms[b]["flux"] for b in bands)
    print(f"\n  Inundated-area-weighted MEAN of each factor, by band\n")
    print(f"  {'band':9} {'area%':>7} {'Rh':>9} {'f_anaer':>8} {'Q10':>7} "
          f"{'unfroz':>7} {'f_surv':>7} {'flux%':>7}")
    Wtot = sum(terms[b]["w"] for b in bands)
    for b in bands:
        t = terms[b]; W = max(t["w"], 1e-30)
        print(f"  {b:9} {t['w']/Wtot*100:6.1f}% {t['rh']/W:9.4f} "
              f"{t['f_anaer']/W:8.3f} {t['q10']/W:7.3f} {t['unfroz']/W:7.3f} "
              f"{t['f_surv']/W:7.3f} {t['flux']/tot*100:6.1f}%")
    print("\n  target flux shares: 30S-30N 70%, 30-60N 21%, 60-90N 8%")
    print("  Rh is kgC/m2/d; f_anaer, Q10, unfroz, f_surv are dimensionless [0,1]")


if __name__ == "__main__":
    main()
