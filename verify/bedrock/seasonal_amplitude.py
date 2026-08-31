"""Seasonal amplitude of soil temperature: the prediction the change was made for.

Runs one forcing year from each arm's final state, sampling soil temperature at
the end of every month, and reports the seasonal amplitude at each soil node.

This is the headline prediction of docs/BEDROCK.md. A zero-flux base at 4 m
reflects the annual wave, so the deepest soil node over-swings; moving the
boundary to 12 m should damp it. The idealised 1-D driver predicted a ~52%
amplitude error at the 3.0 m node in the Arctic, so the bedrock arm should show
a clearly SMALLER swing there, with the shallow nodes essentially unchanged.

Nothing here is a snapshot statistic: amplitude needs the seasonal cycle, which
is exactly why the final-state comparison in compare_ab.py cannot test it.

    python verify/bedrock/seasonal_amplitude.py --year 1901
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
from ibis.physics.bedrock import BedrockParams  # noqa: E402
from ibis.timestepping.loops_gswp3 import run_month_gswp3  # noqa: E402
from ibis.timestepping.calendar import is_leap_year  # noqa: E402

ROOT = Path("/shared/nas2/Hangkai/ibis/verify_outputs")
ARMS = [("control (4 m)", ROOT / "spinup_bedrock_v9_control"),
        ("bedrock (12 m)", ROOT / "spinup_bedrock_v9_3lay")]
MID = np.array([0.05, 0.175, 0.375, 0.75, 1.50, 3.00])   # soil node midpoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=1901)
    a = ap.parse_args()

    cells = build_global_cell_grid(lat_step=0.5, lon_step=0.5)
    lat = np.array([c[1] for c in cells])
    lon = np.array([c[2] for c in cells])
    idx = [latlon_to_grid_idx(la, lo) for la, lo in zip(lat, lon)]
    li = np.array([i[0] for i in idx])
    loi = np.array([i[1] for i in idx])
    lat_rad = np.deg2rad(lat).astype(np.float32)
    reader = GSWP3Reader("/shared/nas2/Min/GSWP3")
    leap = is_leap_year(a.year)

    bands = {"30S-30N": np.abs(lat) <= 30,
             "30-60N": (lat > 30) & (lat <= 60),
             "60-90N": lat > 60}

    res = {}
    for name, d in ARMS:
        st = pickle.load(open(d / "checkpoints" / "state_latest.pkl", "rb"))["jax_state"]
        npoi = int(np.asarray(st.carbon.totlit).shape[0])
        params = build_sim_params(npoi, jnp.asarray(lat_rad), None)
        if getattr(st.soil, "tsoi_deep", None) is not None:
            import equinox as eqx
            params = eqx.tree_at(lambda p: p.bedrock, params, BedrockParams(),
                                 is_leaf=lambda x: x is None)
        key = jax.random.PRNGKey(0)
        monthly = []
        for im in range(12):
            st, key, _ = run_month_gswp3(
                st, PhysicalConstants(), params, reader, li, loi, lat_rad,
                a.year, im, key, 3600.0, leap=leap, nband=2, nsoilcall=10)
            monthly.append(np.asarray(st.soil.tsoi))
            print(f"  {name}: month {im+1:2d} done", flush=True)
        res[name] = np.stack(monthly)          # (12, npoi, nsoilay)

    # --- Permafrost, from the seasonal cycle rather than a snapshot ---------
    # A final-state test cannot distinguish permafrost from seasonal frost: in
    # a December snapshot most of the boreal zone is frozen. Ground that stays
    # below freezing in EVERY month of the year is a defensible proxy. (The
    # strict definition needs two consecutive years; this is one year, so read
    # it as an upper bound on the perennially-frozen area.)
    R, dl = 6371.0, np.deg2rad(0.5)
    area = (R ** 2 * dl * (np.sin(np.deg2rad(lat) + dl / 2)
                           - np.sin(np.deg2rad(lat) - dl / 2)))
    nh = lat > 0
    hs = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])
    top = np.concatenate([[0.0], np.cumsum(hs)[:-1]])
    print(f"\n  Perennially frozen ground in {a.year} "
          f"(frozen in all 12 months), NH only\n")
    print(f"  {'arm':16} {'extent Mkm2':>12} {'active layer m':>15}")
    for name, _ in ARMS:
        always = (res[name] < 273.16).all(axis=0)          # (npoi, nsoilay)
        cell = always.any(axis=1)
        ext = float((area * cell * nh).sum()) / 1e6
        first = np.argmax(always, axis=1)
        alt = np.where(cell, top[first], np.nan)
        m = cell & nh
        print(f"  {name:16} {ext:12.2f} "
              f"{float(np.nanmean(alt[m])) if m.any() else float('nan'):15.3f}")
    print("  Obu (2021) NH permafrost extent: 21.8 Mkm2")

    print(f"\n  Seasonal amplitude (max-min)/2 of soil T over {a.year}, K\n")
    for band, m in bands.items():
        print(f"  --- {band} ({m.sum()} cells)")
        print(f"  {'node depth':>12} {'control':>9} {'bedrock':>9} {'change':>9}")
        for k, z in enumerate(MID):
            amps = {}
            for name, _ in ARMS:
                s = res[name][:, m, k]
                amps[name] = float(np.mean((s.max(0) - s.min(0)) / 2))
            c, b = amps["control (4 m)"], amps["bedrock (12 m)"]
            rel = (b - c) / c * 100 if c > 1e-6 else float("nan")
            print(f"  {z:11.3f}m {c:9.3f} {b:9.3f} {rel:+8.1f}%")
        print()


if __name__ == "__main__":
    main()
