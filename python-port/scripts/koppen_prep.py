#!/usr/bin/env python3
"""Pack the Beck et al. (2023) Koppen-Geiger 0.5-degree maps into one npz for `ibis.io.koppen`.

    python3 python-port/scripts/koppen_prep.py <Koppen_Beck2023 dir> <out.npz>

Reads <dir>/<start>_<end>/koppen_geiger_0p5.tif for the historical periods (1901_1930, 1931_1960,
1961_1990, 1991_2020): uint8 (360, 720), north to south, -180..180, 0 = no class (ocean), 1-30 the
classes of <dir>/legend.txt. Needs numpy and PIL only.
"""
import os, sys
import numpy as np
from PIL import Image

PERIODS = [(1901, 1930), (1931, 1960), (1961, 1990), (1991, 2020)]

def main(src, out):
    maps = []
    for a, b in PERIODS:
        f = os.path.join(src, f"{a}_{b}", "koppen_geiger_0p5.tif")
        arr = np.array(Image.open(f))
        assert arr.shape == (360, 720) and arr.dtype == np.uint8, (f, arr.shape, arr.dtype)
        assert arr.max() <= 30
        maps.append(arr)
    classes = np.stack(maps)
    # orientation checks (row from the north): Amazon Af, Sahara BWh, Antarctica EF
    def rc(lat, lon): return int((90 - lat) / 0.5), int((lon + 180) / 0.5)
    assert classes[3][rc(-2, -62)] == 1 and classes[3][rc(23, 10)] == 4 and classes[3][rc(-80, 0)] == 30
    lat = 89.75 - 0.5 * np.arange(360); lon = -179.75 + 0.5 * np.arange(720)
    np.savez_compressed(out, classes=classes, period_start=np.array([a for a, _ in PERIODS]),
                        period_end=np.array([b for _, b in PERIODS]), lat=lat, lon=lon,
                        source="Beck et al. 2023 Sci. Data 10:724, koppen_geiger_0p5.tif per period")
    for (a, b), m in zip(PERIODS, maps):
        land = m > 0
        print(f"{a}-{b}: {int(land.sum())} classified cells; ET {int((m == 29).sum())}, "
              f"B {int(((m >= 4) & (m <= 7)).sum())}, D {int(((m >= 17) & (m <= 28)).sum())}")
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
