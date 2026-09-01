"""MODIS-derived leaf area index on the IBIS grid, for prescribed-LAI runs.

Builds a grid-cell total LAI from the MODIS PFT LAI that ships with CLM's
0.5 degree surface dataset (Peter Lawrence, NCAR), weighted by the matching
surfdata's cover fractions:

    LAI_total = sum_i LAI_i * PCT_NAT_PFT_i/100 * PCT_NATVEG/100

Validated in docs/GPP.md: the Amazon box (0-10S, 55-70W) gives 5.44 against an
observed ~5-6, with 97% natural vegetation and 0% bare soil, and the global
maximum per grid cell is 6.30.

Two conventions have to be handled and each produced a wrong answer first:

  * CLM surface datasets run SOUTH->NORTH (`LATIXY` -89.75 -> 89.75) while IBIS
    output runs NORTH->SOUTH. Unflipped, the field is latitude-mirrored -- it
    put Antarctic values over the Arctic and reported LAI = 0.00 at 60-90N.
  * The ANNUAL MAXIMUM is the right quantity, not a monthly or annual-mean
    value. IBIS holds seasonality in `frac` (`plai * dropu`), not in `plai`, so
    prescribing anything but the full-leaf LAI applies leaf drop twice.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

DEFAULT_BASE = "/shared/nas2/Hangkai/ELM Edge Effects/inputdata"
DEFAULT_LAI = f"{DEFAULT_BASE}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc"
DEFAULT_SURF = f"{DEFAULT_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc"


def read_modis_lai_max(latindex, lonindex, lai_file=DEFAULT_LAI,
                       surf_file=DEFAULT_SURF) -> np.ndarray:
    """Annual-maximum total LAI at the model's land points, shape (npoi,).

    `latindex`/`lonindex` index the full 360x720 GSWP3 grid (lat 89.75 -> -89.75),
    which is what `run_global` carries as `gswp3_latindex`/`gswp3_lonindex`.
    """
    ds = xr.open_dataset(lai_file)
    sd = xr.open_dataset(surf_file)
    pctveg = np.asarray(sd["PCT_NATVEG"].values, dtype=np.float64) / 100.0
    pctpft = np.asarray(sd["PCT_NAT_PFT"].values, dtype=np.float64) / 100.0
    nt = ds.sizes["time"]

    monthly = np.zeros((12,) + pctveg.shape)
    for m in range(12):
        sel = [t for t in range(nt) if t % 12 == m]
        tot = np.zeros(pctveg.shape)
        for i in range(1, 17):                    # natpft 0 is bare soil
            v = f"LAI_{i}"
            if v in ds:
                tot += np.nan_to_num(ds[v].values[sel].mean(axis=0)) * pctpft[i]
        monthly[m] = tot * pctveg
    grid = monthly.max(axis=0)

    if float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]):
        grid = grid[::-1, :]                      # S->N  =>  N->S

    return np.nan_to_num(grid[latindex, lonindex]).astype(np.float32)
