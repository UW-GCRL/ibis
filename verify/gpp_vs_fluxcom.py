"""Model GPP against FLUXCOM, globally, by latitude band and by aridity.

The aridity binning is the diagnostic that matters: docs show the IBIS GPP
bias is strongly precipitation-dependent (worst in semi-arid, near-correct in
the wettest bins), which is the signature of soil-moisture stress being too
weak rather than a uniform photosynthesis scaling.

    python verify/gpp_vs_fluxcom.py --dirs run_a=/path/out_a run_b=/path/out_b
"""
import argparse, glob, re
import numpy as np, netCDF4 as nc, xarray as xr

FLUXCOM = "/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc"
R = 6371.0


def load_model(d, nyear=30):
    fs = sorted(glob.glob(f"{d}/ibis_yearly_*.nc"),
                key=lambda p: int(re.search(r"(\d+)\.nc$", p).group(1)))[-nyear:]
    if not fs:
        return None, None
    gpp = np.zeros((360, 720)); prcp = np.zeros((360, 720))
    for f in fs:
        with nc.Dataset(f) as ds:
            gpp += np.nan_to_num(np.asarray(ds.variables["aygpptot"][:], dtype=np.float64))
            prcp += np.nan_to_num(np.asarray(ds.variables["ayprcp"][:], dtype=np.float64))
    return gpp / len(fs), prcp / len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--nyear", type=int, default=30)
    a = ap.parse_args()

    o = xr.open_dataset(FLUXCOM)
    lat = o.lat.values
    obs = o.gpp.mean("time").values * 365.25 / 1000.0     # g/m2/d -> kgC/m2/yr
    obs = np.where(obs > 0, obs, np.nan)
    h = np.deg2rad(0.5)
    A = (R**2 * np.deg2rad(0.5)
         * (np.sin(np.deg2rad(lat)+h/2) - np.sin(np.deg2rad(lat)-h/2)))[:, None] * 1e6

    runs = {}
    for spec in a.dirs:
        label, path = spec.split("=", 1)
        g, p = load_model(path, a.nyear)
        if g is None:
            print(f"  {label}: no output in {path}")
            continue
        runs[label] = (g, p)
    if not runs:
        return

    ref_p = next(iter(runs.values()))[1]
    print(f"\n  GLOBAL, cells where FLUXCOM is valid")
    print(f"  {'run':10} {'model':>8} {'FLUXCOM':>8} {'ratio':>7}")
    for label, (g, p) in runs.items():
        v = np.isfinite(obs) & (g > 0)
        Pm = (g*A*v).sum()/1e12; Po = (np.nan_to_num(obs)*A*v).sum()/1e12
        print(f"  {label:10} {Pm:8.1f} {Po:8.1f} {Pm/Po:6.2f}x")

    print(f"\n  BY ANNUAL PRECIPITATION -- ratio to FLUXCOM")
    bins = [(0,200),(200,400),(400,700),(700,1000),(1000,1500),(1500,2500),(2500,99999)]
    hdr = "  " + f"{'precip':>12} " + " ".join(f"{l:>8}" for l in runs)
    print(hdr)
    for lo, hi in bins:
        m = (ref_p >= lo) & (ref_p < hi) & np.isfinite(obs)
        if m.sum() < 50:
            continue
        cells = []
        for label, (g, p) in runs.items():
            v = m & (g > 0)
            r = (g*A*v).sum() / max((np.nan_to_num(obs)*A*v).sum(), 1e-9)
            cells.append(f"{r:7.2f}x")
        print(f"  {f'{lo}-{hi}':>12} " + " ".join(cells))


if __name__ == "__main__":
    main()
