"""Global wetland CH4 evaluation against the Global Methane Budget.

Runs M-lite+ over one year from a spun-up state, prescribing wetland extent
from WAD2M, and reports the global total, the latitudinal split and the
seasonal peak against GMB 2025 (Saunois et al., ESSD 17, 1873).

Supports the parameter configurations compared in docs/METHANE.md 5k, including
biome-varying (r_CH4, Q10p) -- which is the configuration that currently fits
the latitudinal shape best.

    python verify/methane/evaluate_global.py --config biome
    python verify/methane/evaluate_global.py --config all --year 1901

Existed as a throwaway script in /tmp until 2026-08-28; every methane number in
docs/METHANE.md came from it, and none of it was reproducible by anyone else
(docs/AUDIT.md group C).
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

from ibis.bgc.methane import MethaneParams, methane_flux  # noqa: E402
from ibis.constants import PhysicalConstants  # noqa: E402
from ibis.io.gswp3_reader import GSWP3Reader  # noqa: E402
from ibis.io.physics_params import build_sim_params  # noqa: E402
from ibis.io.wad2m import read_wad2m_climatology  # noqa: E402
from ibis.timestepping.loops_gswp3 import run_month_gswp3  # noqa: E402

#: GMB 2025 (Saunois et al., ESSD 17, 1873), wetlands 2010-2019.
GMB_TOTAL, GMB_RANGE = 158.0, (119.0, 203.0)
#: GCP prognostic ensemble latitudinal shares, 2010-2019.
GMB_SHARES = {"30S-30N": 70.0, "30-60N": 21.0, "60-90N": 8.0}

DEFAULT_STATE = ("/shared/nas2/Hangkai/ibis/verify_outputs/spinup_fire_v8/"
                 "checkpoints/state_latest.pkl")
DEFAULT_WAD2M = ("/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/"
                 "WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc")
NDAYPM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#: kgC -> Tg CH4
KGC_TO_TG_CH4 = 16.04 / 12.011 / 1e9


def _band_parameter(lat, tropical, temperate, boreal):
    """Per-cell parameter from a latitude-band biome proxy.

    A crude proxy -- a real implementation should key on climate normals or
    PFT. Used because TRIPLEX-GHG and GCP both report by these bands.
    """
    a = np.full(lat.shape, temperate, dtype=np.float32)
    a[(lat > -30) & (lat < 30)] = tropical
    a[lat >= 60] = boreal
    return jnp.asarray(a)


def configurations(lat):
    """The parameter sets compared in docs/METHANE.md 5k."""
    g = lambda v: _band_parameter(lat, v, v, v)  # noqa: E731
    return {
        "global": (g(0.23), g(1.60)),
        "biome": (_band_parameter(lat, 0.15, 0.32, 0.18), g(1.60)),
        "biome_q10": (_band_parameter(lat, 0.15, 0.32, 0.18),
                      _band_parameter(lat, 3.75, 2.66, 2.03)),
        "biome_noq10": (_band_parameter(lat, 0.15, 0.32, 0.18), g(1.0)),
    }


def run(state_path, wad2m_path, gswp3_dir, year, nsoilcall, configs):
    ck = pickle.load(open(state_path, "rb"))
    st = ck["jax_state"] if isinstance(ck, dict) else ck
    key = jnp.asarray(ck.get("key"), dtype=jnp.uint32) if isinstance(ck, dict) \
        and ck.get("key") is not None else jax.random.PRNGKey(0)

    cells = build_global_cell_grid(lat_step=0.5, lon_step=0.5)
    lat = np.array([c[1] for c in cells])
    lon = np.array([c[2] for c in cells])
    idx = [latlon_to_grid_idx(a, b) for a, b in zip(lat, lon)]
    li = np.array([i[0] for i in idx])
    lo = np.array([i[1] for i in idx])
    latr = np.deg2rad(lat).astype(np.float32)

    mlat = np.arange(-90.0, 90.0, 0.5)
    mlon = np.arange(-180.0, 180.0, 0.5)
    wli = np.rint((lat - mlat[0]) / 0.5).astype(int).clip(0, 359)
    wlo = np.rint((lon - mlon[0]) / 0.5).astype(int).clip(0, 719)
    clim = read_wad2m_climatology(wad2m_path, wli, wlo, mlat, mlon)

    params = build_sim_params(len(cells), jnp.asarray(latr), None)
    reader = GSWP3Reader(gswp3_dir)
    hsoi = jnp.asarray(np.asarray(st.soil.hsoi)[:6])
    area = (111.32 * 0.5) ** 2 * np.cos(np.deg2rad(lat)) * 1e6

    # One model run; every configuration is then evaluated on the same fields.
    snap = []
    for m in range(12):
        st, key, _ = run_month_gswp3(st, PhysicalConstants(), params, reader,
                                     li, lo, latr, year, m, key, 3600.0,
                                     False, 2, nsoilcall)
        snap.append(dict(
            rh=np.asarray(st.vegetation.tco2mic) * 12.0e-3 * 86400.0,
            tsoi=np.asarray(st.soil.tsoi), wsoi=np.asarray(st.soil.wsoi),
            wisoi=np.asarray(st.soil.wisoi)))
        print(f"  month {m + 1:2d}/12", flush=True)

    bands = {"30S-30N": (lat > -30) & (lat < 30),
             "30-60N": (lat >= 30) & (lat < 60),
             "60-90N": lat >= 60}
    print(f"\n  {'configuration':16} {'Tg/yr':>8} "
          + " ".join(f"{b:>9}" for b in bands) + f" {'peak':>5} {'shape err':>10}")
    print(f"  {'GMB 2025 target':16} {GMB_TOTAL:8.0f} "
          + " ".join(f"{GMB_SHARES[b]:8.1f}%" for b in bands) + f" {'7-8':>5}")

    results = {}
    for name, (r_ch4, q10p) in configs.items():
        p = MethaneParams(r_ch4=r_ch4, q10p=q10p)
        tot = 0.0
        per_band = {b: 0.0 for b in bands}
        monthly = []
        for m in range(12):
            s = snap[m]
            f = np.asarray(methane_flux(
                jnp.asarray(clim[m]), jnp.asarray(s["rh"]),
                jnp.asarray(s["tsoi"]), jnp.asarray(s["wsoi"]),
                jnp.asarray(s["wisoi"]), hsoi, p))
            tg = (f * area).sum() * NDAYPM[m] * KGC_TO_TG_CH4
            tot += tg
            monthly.append(tg)
            for b, msk in bands.items():
                per_band[b] += (f[msk] * area[msk]).sum() * NDAYPM[m] * KGC_TO_TG_CH4
        share = {b: v / max(tot, 1e-9) * 100 for b, v in per_band.items()}
        err = sum(abs(share[b] - GMB_SHARES[b]) for b in bands)
        peak = int(np.argmax(monthly)) + 1
        results[name] = dict(total=tot, share=share, peak=peak, shape_err=err)
        print(f"  {name:16} {tot:8.1f} "
              + " ".join(f"{share[b]:8.1f}%" for b in bands)
              + f" {peak:5d} {err:10.1f}")

    print(f"\n  GMB range {GMB_RANGE[0]:.0f}-{GMB_RANGE[1]:.0f} Tg CH4/yr. "
          f"Landing inside it is a WEAK test -- the range spans a factor of "
          f"1.7. Shape and seasonality discriminate; the total does not.")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--wad2m", default=DEFAULT_WAD2M)
    ap.add_argument("--gswp3", default="/shared/nas2/Min/GSWP3")
    ap.add_argument("--year", type=int, default=1901)
    ap.add_argument("--nsoilcall", type=int, default=10)
    ap.add_argument("--config", default="biome",
                    help="global | biome | biome_q10 | biome_noq10 | all")
    a = ap.parse_args()

    cells = build_global_cell_grid(lat_step=0.5, lon_step=0.5)
    lat = np.array([c[1] for c in cells])
    all_cfg = configurations(lat)
    cfg = all_cfg if a.config == "all" else {a.config: all_cfg[a.config]}
    run(a.state, a.wad2m, a.gswp3, a.year, a.nsoilcall, cfg)


if __name__ == "__main__":
    main()
