"""Model LAI against MODIS-derived LAI, on the same 0.5 deg grid.

docs/GPP.md established that the GPP overestimate is a LEAF-AREA problem:
photosynthesis per unit leaf area is roughly right, while LAI appeared
1.46-1.68x high. But those LAI reference values were biome-typical estimates
typed into a diagnostic script, not a dataset -- so the conclusion was blocked
pending real data.

This uses the MODIS PFT LAI that ships with CLM's 0.5 deg surface dataset
(Peter Lawrence, NCAR), weighted to a grid-cell total by the matching
surfdata's PFT cover fractions:

    LAI_total = sum_i LAI_i * PCT_NAT_PFT_i/100 * PCT_NATVEG/100

**Month matching matters.** IBIS's yearly netCDF is written after the December
loop, so `laiu`/`lail` are a year-END snapshot, not an annual mean. Comparing
that against an annual-mean MODIS LAI would flatter the model in the northern
winter and penalise it in the south. December is compared against December.

    python verify/lai_vs_modis.py
"""
import argparse, glob, re
import numpy as np, netCDF4 as nc, xarray as xr

BASE = "/shared/nas2/Hangkai/ELM Edge Effects/inputdata"
MODIS = f"{BASE}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc"
SURF = f"{BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc"
R = 6371.0


def modis_lai_total(mode="max"):
    """Grid-cell total LAI: annual MAXIMUM (default) or annual mean.

    Annual maximum is the right target. IBIS carries seasonality in `frac`
    (`frac = plai * dropu / avglaiu`, vegetation.f:150), NOT in `lai`, so
    `laiu`/`lail` are the FULL-LEAF leaf area index rather than the current
    month's. Comparing them against a December or annual-mean MODIS LAI
    penalises the model for deciduous dormancy it does not represent in that
    variable -- a first pass at this reported 8.6x at 30-60N for exactly that
    reason.
    """
    ds = xr.open_dataset(MODIS)
    sd = xr.open_dataset(SURF)
    pctveg = sd["PCT_NATVEG"].values / 100.0          # (lat, lon)
    pctpft = sd["PCT_NAT_PFT"].values / 100.0         # (natpft, lat, lon)
    nt = ds.sizes["time"]
    monthly = np.zeros((12,) + pctveg.shape)
    for m in range(12):
        sel = [t for t in range(nt) if t % 12 == m]
        tot = np.zeros(pctveg.shape)
        for i in range(1, 17):                        # natpft 0 = bare soil
            v = f"LAI_{i}"
            if v not in ds:
                continue
            tot += np.nan_to_num(ds[v].values[sel].mean(axis=0)) * pctpft[i]
        monthly[m] = tot * pctveg
    out = monthly.max(axis=0) if mode == "max" else monthly.mean(axis=0)
    landfrac = np.asarray(sd["LANDFRAC_PFT"].values)
    natveg = np.asarray(sd["PCT_NATVEG"].values)
    # CLM surface datasets run SOUTH->NORTH (LATIXY -89.75 -> 89.75); IBIS
    # output runs NORTH->SOUTH. Without this flip the comparison comes out
    # latitude-mirrored -- Arctic model cells against Antarctic MODIS cells,
    # which is what made a first pass report MODIS LAI = 0.00 at 60-90N.
    if float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]):
        out = out[::-1, :]
        landfrac = landfrac[::-1, :]
        natveg = natveg[::-1, :]
    return out, landfrac, natveg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/shared/nas2/Hangkai/ibis/verify_outputs/"
                                     "spinup_cold_v10/output")
    ap.add_argument("--mode", default="max", choices=["max", "mean"],
                    help="annual maximum (default) or annual mean MODIS LAI")
    ap.add_argument("--min-natveg", type=float, default=90.0,
                    help="only compare cells at least this %% natural "
                         "vegetation. The MODIS total weights PCT_NAT_PFT by "
                         "PCT_NATVEG, so cropland contributes zero LAI while "
                         "IBIS -- which has no crop PFT -- grows vegetation "
                         "there. PCT_NATVEG averages 67.6%% over land, so "
                         "comparing everywhere inflates the ratio.")
    a = ap.parse_args()

    obs, landfrac, natveg = modis_lai_total(a.mode)

    fs = sorted(glob.glob(f"{a.dir}/ibis_yearly_*.nc"),
                key=lambda p: int(re.search(r"(\d+)\.nc$", p).group(1)))[-1]
    with nc.Dataset(fs) as ds:
        g = lambda n: np.nan_to_num(np.asarray(ds.variables[n][:], dtype=np.float64))
        lat = np.asarray(ds.variables["lat"][:])
        mod = g("laiu") * g("fu") + g("lail") * g("fl")
        prcp = g("ayprcp")

    h = np.deg2rad(0.5)
    A = (R**2 * np.deg2rad(0.5)
         * (np.sin(np.deg2rad(lat)+h/2) - np.sin(np.deg2rad(lat)-h/2)))[:, None]
    valid = ((prcp > 0) & (landfrac > 0.5) & np.isfinite(obs)
             & (natveg >= a.min_natveg))

    print(f"  MODIS PFT LAI x surfdata cover, annual {a.mode.upper()}, "
          f"vs model full-leaf LAI")
    print(f"  restricted to cells >= {a.min_natveg:.0f}% natural vegetation")
    print(f"  cells compared: {int(valid.sum())}\n")
    wm = lambda x, m: float((x*A*m).sum() / (A*m).sum())
    print(f"  GLOBAL   model {wm(mod, valid):.2f}   MODIS {wm(obs, valid):.2f}"
          f"   ratio {wm(mod, valid)/wm(obs, valid):.2f}x\n")

    print(f"  {'precip mm/yr':>13} {'cells':>7} {'model':>7} {'MODIS':>7} {'ratio':>7}")
    for lo, hi in [(0,200),(200,400),(400,700),(700,1000),(1000,1500),
                   (1500,2500),(2500,99999)]:
        m = valid & (prcp >= lo) & (prcp < hi)
        if m.sum() < 50:
            continue
        mm, oo = wm(mod, m), wm(obs, m)
        print(f"  {f'{lo}-{hi}':>13} {int(m.sum()):7d} {mm:7.2f} {oo:7.2f} "
              f"{mm/max(oo,1e-9):6.2f}x")

    print(f"\n  {'lat band':>13} {'model':>7} {'MODIS':>7} {'ratio':>7}")
    for nm, mb in (("60-90N", lat > 60), ("30-60N", (lat > 30) & (lat <= 60)),
                   ("0-30N", (lat > 0) & (lat <= 30)),
                   ("30S-0", (lat > -30) & (lat <= 0)),
                   ("60-30S", (lat > -60) & (lat <= -30))):
        m = valid & mb[:, None]
        if m.sum() < 50:
            continue
        mm, oo = wm(mod, m), wm(obs, m)
        print(f"  {nm:>13} {mm:7.2f} {oo:7.2f} {mm/max(oo,1e-9):6.2f}x")


if __name__ == "__main__":
    main()
