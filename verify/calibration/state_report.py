#!/usr/bin/env python3
"""One table for the state of a run: fluxes, pools and LAI against observations.

Written so v11 can be judged against both_fix on identical definitions. Every
quantity here has been got wrong at least once in this project by a
normalisation or convention slip, so each carries its normalising area and
temporal convention explicitly:

* totals are integrated over the model's own 165.0 Mkm2 domain (surta.nc
  landfrac), which is larger than Earth's ~149 Mkm2 of land and much larger
  than the ~130 Mkm2 ice-free area some published per-area ranges use -- so
  per-area means and PgC totals can disagree about whether a pool is "low";
* FLUXCOM is g m-2 day-1 and is converted with x365.25/1000 (read as
  kgC m-2 yr-1 it gives a nonsense 2x, which has happened twice);
* MODIS LAI is the annual MAXIMUM, diluted by PCT_NATVEG to a cell mean,
  because IBIS carries phenology in `frac` rather than in `lai`;
* fluxes are averaged over the last `--nyear` years, not read off one year --
  a single-year read produced a spurious -13.6% drift once.

Usage:
  ibis-python verify/calibration/state_report.py \
      --run both_fix=/path/out:/path/restart/ibis_restart_2200.npz \
      --run v11=/path/out:/path/restart/ibis_restart_2010.npz
"""
from __future__ import annotations

import argparse
import glob
import re
import sys

import numpy as np
import netCDF4 as nc
import xarray as xr

REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port")

MASK = "/shared/nas2/Min/HWSD/surta.nc"
FLUXCOM = "/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc"
MODIS_BASE = "/shared/nas2/Hangkai/ELM Edge Effects/inputdata"
R = 6371.0
BANDS = [(-30, 30, "30S-30N"), (30, 50, "30-50N"), (50, 60, "50-60N"),
         (60, 70, "60-70N"), (70, 90, "70-90N")]
TARGET = {"GPP PgC/yr": "120-130", "NPP PgC/yr": "55-60", "Rh PgC/yr": "49-58",
          "ET mm/yr": "500-550", "soil C kgC/m2": "10-15",
          "litter kgC/m2": "0.4-0.8", "veg C PgC": "450-650"}


def cell_area():
    lat = np.arange(89.75, -90, -0.5)
    dl = np.deg2rad(0.5)
    a = (R ** 2) * dl * dl * np.cos(np.deg2rad(lat))     # km2 per cell
    return np.repeat(a[:, None], 720, axis=1) * 1.0e6    # m2


def load_run(outdir, nyear):
    fs = sorted(glob.glob(f"{outdir}/ibis_yearly_*.nc"),
                key=lambda p: int(re.search(r"(\d+)\.nc$", p).group(1)))[-nyear:]
    acc = {}
    for f in fs:
        with nc.Dataset(f) as ds:
            for v in ("aygpptot", "aynpptot", "ayco2mic", "ayaet", "cveg",
                      "laiu", "lail", "fu", "fl"):
                x = np.nan_to_num(np.asarray(ds.variables[v][:], dtype=np.float64))
                acc[v] = acc.get(v, 0.0) + x
    for k in acc:
        acc[k] /= len(fs)
    acc["lai"] = acc["laiu"] * acc["fu"] + acc["lail"] * acc["fl"]
    return acc, len(fs), fs[-1]


def pools(restart):
    from ibis.dimensions import GridConfig
    from ibis.io.restart import load_state
    from ibis.state.model_state import create_ibis_state
    st, _ = load_state(restart, create_ibis_state(
        GridConfig(npoi=95333, nbedrock=3)))
    return (np.asarray(st.carbon.totcsoi, dtype=np.float64),
            np.asarray(st.carbon.totlit, dtype=np.float64))


def modis_lai():
    ds = xr.open_dataset(f"{MODIS_BASE}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc")
    sd = xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc")
    pv = np.asarray(sd["PCT_NATVEG"].values) / 100.0
    pp = np.asarray(sd["PCT_NAT_PFT"].values) / 100.0
    nt = ds.sizes["time"]
    mon = np.zeros((12,) + pv.shape)
    for m in range(12):
        sel = [t for t in range(nt) if t % 12 == m]
        tot = np.zeros(pv.shape)
        for k in range(1, 17):
            v = f"LAI_{k}"
            if v in ds:
                tot += np.nan_to_num(ds[v].values[sel].mean(axis=0)) * pp[k]
        mon[m] = tot * pv
    g = mon.max(axis=0)
    if float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]):
        g = g[::-1, :]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="name=outdir[:restart.npz]")
    ap.add_argument("--nyear", type=int, default=20)
    a = ap.parse_args()

    m = np.array(xr.open_dataset(MASK).landfrac.values) > 0
    area = cell_area()
    A = float(area[m].sum())
    o = xr.open_dataset(FLUXCOM)
    fc = np.asarray(o["gpp"].mean(dim="time")) * 365.25 / 1000.0
    mod = modis_lai()
    lat = np.arange(89.75, -90, -0.5)
    LA = np.repeat(lat[:, None], 720, axis=1)

    print(f"Domain: {A/1e12:.1f} Mkm2 (surta.nc landfrac), "
          f"{int(m.sum())} cells. Fluxes = last {a.nyear} yr mean.\n")
    runs = {}
    for spec in a.run:
        name, rest = spec.split("=", 1)
        outdir, _, rst = rest.partition(":")
        d, n, last = load_run(outdir, a.nyear)
        runs[name] = (d, rst, last)

    rows = ["GPP PgC/yr", "NPP PgC/yr", "Rh PgC/yr", "ET mm/yr",
            "veg C PgC", "soil C kgC/m2", "litter kgC/m2", "LAI mean"]
    print(f"{'quantity':<16}" + "".join(f"{k:>14}" for k in runs) +
          f"{'observed':>14}")
    vals = {}
    for name, (d, rst, _) in runs.items():
        v = {}
        v["GPP PgC/yr"] = float((d["aygpptot"] * area)[m].sum()) / 1e12
        v["NPP PgC/yr"] = float((d["aynpptot"] * area)[m].sum()) / 1e12
        v["Rh PgC/yr"] = float((d["ayco2mic"] * area)[m].sum()) / 1e12
        v["ET mm/yr"] = float(np.average(d["ayaet"][m], weights=area[m]))
        v["veg C PgC"] = float((d["cveg"] * area)[m].sum()) / 1e12
        v["LAI mean"] = float(np.average(d["lai"][m], weights=area[m]))
        if rst:
            cs, lt = pools(rst)
            fl = np.where(m.ravel())[0]
            w = area.ravel()[fl]
            v["soil C kgC/m2"] = float(np.average(cs, weights=w))
            v["litter kgC/m2"] = float(np.average(lt, weights=w))
        vals[name] = v
    for r in rows:
        print(f"{r:<16}" + "".join(
            f"{vals[k].get(r, float('nan')):>14.3f}" for k in runs) +
            f"{TARGET.get(r, ''):>14}")

    print(f"\nLAI BY BAND vs MODIS annual-max (cell mean)")
    print(f"{'band':<10}" + "".join(f"{k:>12}" for k in runs) +
          f"{'MODIS':>10}{'ratio':>9}")
    for lo, hi, nm in BANDS:
        s = m & (LA >= lo) & (LA < hi)
        obs = float(np.average(mod[s], weights=area[s]))
        line = f"{nm:<10}"
        for k, (d, _, _) in runs.items():
            line += f"{float(np.average(d['lai'][s], weights=area[s])):>12.2f}"
        last = list(runs)[-1]
        r = float(np.average(runs[last][0]["lai"][s], weights=area[s])) / max(obs, 1e-9)
        print(line + f"{obs:>10.2f}{r:>9.2f}")

    print(f"\nGPP BY BAND vs FLUXCOM (kgC/m2/yr)")
    print(f"{'band':<10}" + "".join(f"{k:>12}" for k in runs) +
          f"{'FLUXCOM':>10}{'ratio':>9}")
    for lo, hi, nm in BANDS:
        s = m & (LA >= lo) & (LA < hi) & np.isfinite(fc) & (fc > 0)
        if not s.any():
            continue
        obs = float(np.average(fc[s], weights=area[s]))
        line = f"{nm:<10}"
        for k, (d, _, _) in runs.items():
            line += f"{float(np.average(d['aygpptot'][s], weights=area[s])):>12.3f}"
        last = list(runs)[-1]
        r = float(np.average(runs[last][0]["aygpptot"][s], weights=area[s])) / obs
        print(line + f"{obs:>10.3f}{r:>9.2f}")


if __name__ == "__main__":
    main()
