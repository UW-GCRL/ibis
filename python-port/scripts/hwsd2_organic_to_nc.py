#!/usr/bin/env python3
"""HWSD2 organic carbon and bulk density on the IBIS 0.5-degree grid and layers (CALIBRATION.md 44).

The texture file soita.nc (Min's hwsd2_to_soita.py) carries sand and clay only. This script applies the
same rule -- per soil-mapping unit, SHARE x depth-overlap-weighted layer means on the six IBIS layers,
then the pixel-count mean of the 60 x 60 30-arcsecond pixels of each 0.5-degree cell -- to ORG_CARBON
(% of dry mass) and BULK (g cm-3), and recomputes sand and clay with it so the four fields are
consistent. Two steps, because the pure-Python Access reader lives in the system python and netCDF4 in
the model environment:

    python3 hwsd2_organic_to_nc.py build <HWSD2.mdb local copy> <HWSD2.bil> <out.npz>
    <ibis-python> hwsd2_organic_to_nc.py nc <out.npz> <out.nc> [<Min soita.nc to compare>]

Output nc: sandpct, claypct, orgcpct, bulkdens (layer=6, lat=360 north to south, lon=720), NaN where
HWSD2 has no data (the model's loader falls back to its defaults there). HWSD2 layers D1-D7 reach
200 cm; IBIS layer 6 (200-400 cm) copies layer 5, as in the texture file.
"""
import sys, time
import numpy as np

IB = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 400)]   # cm
NROWS, NCOLS, NODATA, PIX = 21600, 43200, 65535, 60
VARS = ("SAND", "CLAY", "ORG_CARBON", "BULK")


def build(mdb, bil, out):
    import pandas as pd
    from access_parser import AccessParser
    t0 = time.time()
    L = pd.DataFrame(AccessParser(mdb).parse_table("HWSD2_LAYERS"))
    for v in VARS + ("SHARE", "TOPDEP", "BOTDEP"):
        L[v] = pd.to_numeric(L[v], errors="coerce")
    for v in VARS:
        L[v] = L[v].where(L[v] >= 0)                       # -7 / -9 = missing
    L = L[(L.BOTDEP > L.TOPDEP) & (L.SHARE > 0) & (L.HWSD2_SMU_ID > 0)]
    smu = L.HWSD2_SMU_ID.values.astype(np.int64); nmax = int(smu.max()) + 1
    print(f"{len(L):,} layer rows, max SMU {nmax - 1:,}, read in {time.time() - t0:.0f} s", flush=True)
    lut = {v: np.full((nmax, 6), np.nan, np.float32) for v in VARS}
    top = L.TOPDEP.values.astype(float); bot = L.BOTDEP.values.astype(float); share = L.SHARE.values.astype(float)
    for k, (a, b) in enumerate(IB):
        ov = np.clip(np.minimum(bot, b) - np.maximum(top, a), 0, None)
        w = share * ov
        for v in VARS:
            x = L[v].values.astype(float); ok = np.isfinite(x) & (w > 0)
            num = np.bincount(smu[ok], weights=(w * x)[ok], minlength=nmax)
            den = np.bincount(smu[ok], weights=w[ok], minlength=nmax)
            with np.errstate(invalid="ignore", divide="ignore"):
                lut[v][:, k] = np.where(den > 0, num / den, np.nan)
    for v in VARS:                                          # fill a missing layer from the nearest layer with data
        a = lut[v]
        for k in range(6):
            miss = ~np.isfinite(a[:, k])
            for d in range(1, 6):
                for kk in (k - d, k + d):
                    if 0 <= kk < 6:
                        src = miss & np.isfinite(a[:, kk]); a[src, k] = a[src, kk]; miss = ~np.isfinite(a[:, k])
    print("lookup built", {v: float(np.nanmean(lut[v][:, 0])) for v in VARS}, flush=True)
    raster = np.memmap(bil, dtype="<u2", mode="r", shape=(NROWS, NCOLS))
    grids = {v: np.full((6, 360, 720), np.nan, np.float32) for v in VARS}
    for irow in range(360):
        blk = np.asarray(raster[irow * PIX:(irow + 1) * PIX, :]).astype(np.int64)      # (60, 43200)
        valid = (blk != NODATA) & (blk < nmax)
        blk = np.where(valid, blk, 0)
        for v in VARS:
            vals = lut[v][blk]                                                         # (60, 43200, 6)
            vals = np.where(valid[..., None], vals, np.nan)
            vals = vals.reshape(PIX, 720, PIX, 6)
            with np.errstate(invalid="ignore"):
                grids[v][:, irow, :] = np.nanmean(vals, axis=(0, 2)).T
        if irow % 60 == 0:
            print(f"  row {irow}/360  {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(out, **{v.lower(): grids[v] for v in VARS})
    print("wrote", out, {v: float(np.nanmean(grids[v][0])) for v in VARS})


def nc(npz, out, ref=None):
    import netCDF4
    d = np.load(npz)
    names = {"sand": "sandpct", "clay": "claypct", "org_carbon": "orgcpct", "bulk": "bulkdens"}
    units = {"sandpct": "percent", "claypct": "percent", "orgcpct": "percent of dry mass", "bulkdens": "g cm-3"}
    with netCDF4.Dataset(out, "w") as ds:
        ds.createDimension("layer", 6); ds.createDimension("lat", 360); ds.createDimension("lon", 720)
        la = ds.createVariable("lat", "f4", ("lat",)); la[:] = 89.75 - 0.5 * np.arange(360)
        lo = ds.createVariable("lon", "f4", ("lon",)); lo[:] = -179.75 + 0.5 * np.arange(720)
        ly = ds.createVariable("layer", "i4", ("layer",)); ly[:] = np.arange(1, 7)
        for k, nm in names.items():
            v = ds.createVariable(nm, "f4", ("layer", "lat", "lon"), zlib=True, fill_value=np.nan)
            v[:] = d[k]; v.units = units[nm]
        ds.title = "IBIS-JAX soil texture, organic carbon and bulk density from HWSD2 (scripts/hwsd2_organic_to_nc.py)"
        ds.grid = "Global 0.5-degree; lat N->S (89.75 to -89.75); lon W->E; IBIS layers 0-10, 10-25, 25-50, 50-100, 100-200, 200-400 cm"
        ds.aggregation = "per SMU: SHARE x depth-overlap-weighted layer means; per cell: pixel-count mean of the 60x60 HWSD2 30-arcsec pixels"
    print("wrote", out)
    if ref:
        with netCDF4.Dataset(ref) as r:
            for a, b in (("sandpct", "sand"), ("claypct", "clay")):
                x = np.asarray(r[a][:]); y = d[b]; ok = np.isfinite(y) & np.isfinite(x)
                print(f"  {a} vs {ref}: layer-1 mean abs diff {np.nanmean(np.abs(x[0][ok[0]] - y[0][ok[0]])):.3f} %, "
                      f"r {np.corrcoef(x[0][ok[0]], y[0][ok[0]])[0, 1]:.4f}, cells with data here/there {int(ok[0].sum())}/{int(np.isfinite(x[0]).sum())}")
    oc = d["org_carbon"]; bd = d["bulk"]
    fo = np.clip(0.01 * oc * 1.724 * bd * 1000 / 130.0, 0, 1)
    print("  f_om layer 1: mean %.3f, >0.5 in %d cells, >0.9 in %d; layer 4 mean %.3f" % (np.nanmean(fo[0]), int(np.nansum(fo[0] > 0.5)), int(np.nansum(fo[0] > 0.9)), np.nanmean(fo[3])))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:5])
    else:
        nc(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
