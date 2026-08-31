"""Convergence and global budget of the cold-start spin-up.

Reads the yearly netCDF series and reports area-weighted global means and
totals, so "has soil carbon equilibrated" is answered from the record rather
than from the last year alone.

    python verify/check_spinup_convergence.py --dir <output dir>
"""
import argparse, glob, os, re
import numpy as np, netCDF4 as nc

R = 6371.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/shared/nas2/Hangkai/ibis/verify_outputs/"
                                     "spinup_cold_v10/output")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "ibis_yearly_*.nc")),
                   key=lambda p: int(re.search(r"(\d+)\.nc$", p).group(1)))
    print(f"  {len(files)} yearly files\n")

    with nc.Dataset(files[0]) as ds:
        lat = np.asarray(ds.variables["lat"][:])
    h = np.deg2rad(0.5)
    area = (R**2 * np.deg2rad(0.5)
            * (np.sin(np.deg2rad(lat)+h/2) - np.sin(np.deg2rad(lat)-h/2)))[:, None] * 1e6

    def gm(v, land):           # area-weighted mean over land
        return float((np.where(land, v, 0)*area).sum() / (land*area).sum())
    def pg(v, land):           # PgC/yr total
        return float((np.where(land, v, 0)*area).sum() / 1e12)

    rows = []
    for f in files:
        yr = int(re.search(r"(\d+)\.nc$", f).group(1))
        with nc.Dataset(f) as ds:
            g = lambda n: np.asarray(ds.variables[n][:], dtype=np.float64)
            csoi, lit, veg = g("totcsoi"), g("totlit"), g("cveg")
            rh, prcp = g("ayco2mic"), g("ayprcp")
            land = np.isfinite(prcp) & (prcp != 0)
            rows.append((yr, gm(csoi, land), gm(lit, land), gm(veg, land),
                         pg(rh, land), gm(g("laiu")+g("lail"), land),
                         int(np.isnan(csoi[land]).sum())))
    a_ = np.array([r[:6] for r in rows]); nan = sum(r[6] for r in rows)

    print(f"  {'year':>6} {'totcsoi':>9} {'totlit':>8} {'cveg':>8} "
          f"{'Rh PgC/yr':>10} {'LAI':>6}")
    for i in list(range(0, 60, 20)) + list(range(60, len(rows), 40)) + [len(rows)-1]:
        if i < len(rows):
            r = a_[i]
            print(f"  {int(r[0]):6d} {r[1]:9.3f} {r[2]:8.3f} {r[3]:8.3f} "
                  f"{r[4]:10.2f} {r[5]:6.2f}")

    print(f"\n  NaN in totcsoi across all years: {nan}")
    last50 = a_[-50:]
    drift = (last50[-1,1]-last50[0,1]) / 50
    print(f"  soil carbon drift over the last 50 yr: {drift:+.5f} kgC/m2/yr "
          f"({drift/last50[-1,1]*100:+.3f} %/yr)")
    print(f"  Rh over the last 50 yr: mean {last50[:,4].mean():.2f} "
          f"+- {last50[:,4].std():.2f} PgC/yr")
    print(f"\n  published global Rh 49-58 PgC/yr")

if __name__ == "__main__":
    main()
