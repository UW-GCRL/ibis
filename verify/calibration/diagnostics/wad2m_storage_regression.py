#!/usr/bin/env python
"""
wad2m_storage_regression.py -- diagnostic only (no model code touched).

How much of the interannual variability of the satellite wetland (inundated)
fraction (WAD2M v2.0, 0.5 deg, monthly) is explained by the land model's OWN
runoff and soil-water storage, per climate band and Koppen group?  Companion of
wad2m_precip_regression.py (GSWP3 precipitation, 2000-2010), whose conventions
are reused: WAD2M orientation and fill (-9999 = frozen / no data -> NaN for the
regressions, 0 for areas and climatologies), the SAME analysis cells (model land,
WAD2M 2000-2010 mean >= 0.02 with fill = 0, >= 24 valid months), wetland-area
weights (cell area x that mean), bands |lat| < 30 / 30-60 / >= 60, Koppen groups
A-E (Beck 2023), relative anomalies against the same-month climatology, the
freeze-mask fixed-season variant.

Model: IBIS 3.0 final run s7 (final_s7_crop/hindcast/out), monthly 2001-2010,
north-to-south.  No 2000 monthly output exists (transient output is yearly), so
the common monthly sample is Jul 2001 - Dec 2010 (114 months): the longest
accumulation window (7 months) needs 6 lead months.  Climatologies (WAD2M and
predictors) are taken over that same sample.

Water storage: wsoi is the liquid fraction of pore space at the END of the month
(run_global.py writes state.soil.wsoi after the month); porosity per layer from
/shared/nas2/Min/HWSD/soita.nc with the USDA texture-class table of
ibis/io/soil_init.py (copied, no JAX; the run has soil_organic off).
W = sum_k poros_k * wsoi_k * hsoi_k (mm, 4 m column), Wrz = top 4 layers (1 m),
s1 = wsoi of the top layer (0.10 m).  Storage lags 0, 1, 2, 3 months plus
"lag0.5" = mean of the end-of-month states of t and t-1 (the in-month mean).

Predictors (anomalies vs the same-month climatology; relative and standardized):
  runoff   amtrunof + amtdrain (amtdrain is identically 0 in this run, see checks)
  drainage amtdrain
  precip   amprcp (GSWP3, same data as the precipitation analysis)
  pme      amprcp - amaet (net water input; water-conserving, added because the
           runoff diagnostic does not close the water balance, see checks)
  fluxes summed over windows of 1, 2, 3, 4, 6, 7 months ending in the month;
  W, Wrz, s1 at lags 0, 0.5, 1, 2, 3.

Outputs (under OUT):
  checks.json                     data / orientation / water-balance checks, variables + units
  per_cell_results.csv            per cell: r and slopes per predictor, best predictor, seasonal lags
  summary_band_predictor.csv      group x predictor (x window / lag): weighted median [IQR] r, slopes, shares
  band_timeseries.csv             annual wetland area and predictor values per group (long format)
  band_timeseries_stats.csv       group x predictor: annual r, slope, detrended r, fixed-season variants
  seasonal_lag.csv                seasonal-cycle lag of Fw behind each base predictor
  scheme_fit.csv                  fitted beta per band x candidate, uncertainty, train / hold-out skill
  scheme_band_monthly.csv         band monthly wetland area: obs, climatology, scheme (2001-2010)
  regional_seasonality.csv        floodplain / peatland regions: peaks, lags, expectations
  regional_climatology.csv        their 12-month climatologies (long format)
  fig_best_predictor_r_map.png    map of r at the best predictor + its family
  fig_band_timeseries.png         annual wetland area vs best predictor and vs precipitation
  model_cells_2001-2010.npz       cache of the model monthly fields at the analysis cells

Run:  nice -n 10 /shared/nas2/Hangkai/ibis-env/ibis-python wad2m_storage_regression.py
"""
import os
import time
import json
import resource
import warnings

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from scipy import stats, optimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = "/shared/nas2/Hangkai/ibis/verify_outputs/wad2m_regression/storage"
PREV = "/shared/nas2/Hangkai/ibis/verify_outputs/wad2m_regression"
WAD = "/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc"
HIND = "/shared/nas2/Hangkai/ibis/verify_outputs/final_s7_crop/hindcast/out"
SOITA = "/shared/nas2/Min/HWSD/soita.nc"
KOPPEN = "/shared/nas/Public/Climate_Biome/Koppen_Beck2023/1991_2020/koppen_geiger_0p5.tif"

Y0, Y1 = 2001, 2010
NYR = Y1 - Y0 + 1
NMON = NYR * 12
S0 = 6                         # first month of the common sample (Jul 2001)
TRAIN_LAST = (2006 - Y0) * 12 + 11   # Dec 2006
HSOI = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])
NRZ = 4                        # root zone = top 4 layers = 1.0 m
WINDOWS = {"1mo": 1, "2mo": 2, "3mo": 3, "4mo": 4, "6mo": 6, "7mo": 7}
LAGS = {"lag0": 0, "lag0.5": 0.5, "lag1": 1, "lag2": 2, "lag3": 3}
MINFW = 0.02
MIN_CLIM_FW = 0.01
MIN_CLIM_FLUX_PER_MONTH = 2.0  # mm per window-month (as for P in the precipitation analysis)
MIN_PAIRS = 24
MIN_YEARS_CLIM = 6
N_NULL = 10                    # year-shuffle permutations for the best-of-predictors null
N_BOOT = 1000

os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time()-T0:7.1f}s, maxRSS {rss:5.2f} GB] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 0. USDA texture classes -> porosity (copied from ibis/io/soil_init.py)
# ----------------------------------------------------------------------------
_POROS = np.array([0.437, 0.437, 0.453, 0.463, 0.501, 0.398, 0.464, 0.471, 0.430, 0.479, 0.475])
_POLYGONS = [
    np.array([[85, 0], [90, 10], [100, 0]]),                                   # sand
    np.array([[70, 0], [85, 15], [90, 10], [85, 0]]),                          # loamy sand
    np.array([[50, 0], [43, 7], [52, 7], [52, 20], [80, 20], [85, 15], [70, 0]]),  # sandy loam
    np.array([[43, 7], [23, 27], [45, 27], [52, 20], [52, 7]]),               # loam
    np.array([[0, 0], [0, 27], [23, 27], [50, 0]]),                            # silt loam
    np.array([[52, 20], [45, 27], [45, 35], [65, 35], [80, 20]]),             # sandy clay loam
    np.array([[20, 27], [20, 40], [45, 40], [45, 27]]),                        # clay loam
    np.array([[0, 27], [0, 40], [20, 40], [20, 27]]),                          # silty clay loam
    np.array([[45, 35], [45, 55], [65, 35]]),                                  # sandy clay
    np.array([[0, 40], [0, 60], [20, 40]]),                                    # silty clay
    np.array([[20, 40], [0, 60], [0, 100], [45, 55], [45, 40]]),              # clay
]


def _point_in_polygon(poly, x, y):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _build_texture_lookup():
    lut = np.full((101, 101), 4, dtype=np.int8)   # default: silt loam
    for s in range(101):
        for c in range(101):
            if s + c > 100:
                continue
            if s == 0 and c == 0:
                continue
            for idx, poly in enumerate(_POLYGONS):
                if _point_in_polygon(poly, float(s), float(c)):
                    lut[s, c] = idx
                    break
    return lut


TEXTURE_LUT = _build_texture_lookup()

# ----------------------------------------------------------------------------
# 1. WAD2M (whole 2000-2020 record) and the analysis cells
# ----------------------------------------------------------------------------
checks = {}
ds = nc.Dataset(WAD)
lat = np.asarray(ds.variables["lat"][:])
lon = np.asarray(ds.variables["lon"][:])
assert lat[0] < lat[-1] and abs(lat[0] + 89.75) < 1e-6, "WAD2M expected south-to-north"
FwAll = np.ma.filled(ds.variables["Fw"][:].astype(np.float32), np.nan)      # (252,360,720)
ds.close()
NREC = FwAll.shape[0]
assert NREC == 252
fin = np.isfinite(FwAll)
checks["wad2m"] = dict(shape=list(FwAll.shape), lat0=float(lat[0]), lat1=float(lat[-1]), lon0=float(lon[0]),
                       min_valid=float(np.nanmin(FwAll)), max_valid=float(np.nanmax(FwAll)))
assert np.nanmin(FwAll) >= 0 and np.nanmax(FwAll) <= 1.0 + 1e-6

# land mask of the model (north-to-south output -> flip)
h = nc.Dataset(os.path.join(HIND, "ibis_monthly_2001_01.nc"))
mlat = np.asarray(h.variables["lat"][:]); mlon = np.asarray(h.variables["lon"][:])
assert mlat[0] > mlat[-1] and abs(mlat[0] - 89.75) < 1e-6 and np.allclose(mlon, lon), "model grid N->S, same lon"
var_info = {k: dict(dims=list(v.dimensions), units=getattr(v, "units", ""), long_name=getattr(v, "long_name", ""))
            for k, v in h.variables.items() if v.ndim >= 2}
land_mask = np.isfinite(np.ma.filled(h.variables["amprcp"][:].astype(np.float64), np.nan))[::-1, :]
h.close()
checks["model_monthly_variables"] = var_info

# analysis cells: identical definition (and period) to the precipitation analysis
F11 = FwAll[0:132]
nvalid11 = np.isfinite(F11).sum(0)
mean11 = np.where(np.isfinite(F11), F11, 0.0).mean(0)
cells = land_mask & (nvalid11 >= MIN_PAIRS) & (mean11 >= MINFW)
ii, jj = np.where(cells)
N = len(ii)
log(f"analysis cells: {N}")
checks["n_cells"] = int(N)

R_E = 6371.0088
edges = np.deg2rad(np.arange(-90, 90.5, 0.5))
area_row = (R_E ** 2) * np.deg2rad(0.5) * (np.sin(edges[1:]) - np.sin(edges[:-1]))
clat = lat[ii]; clon = lon[jj]
carea = area_row[ii]
cmean = mean11[ii, jj]
wet_area = carea * cmean
band = np.where(np.abs(clat) < 30, "tropical", np.where(np.abs(clat) < 60, "temperate",
                np.where(clat >= 60, "boreal", "antarctic")))
hemi = np.where(clat >= 0, "N", "S")

import rasterio
with rasterio.open(KOPPEN) as src:
    kop = src.read(1)[::-1, :]
kcode = kop[ii, jj].astype(int)
kg = np.full(N, "", dtype=object)
for g, (a, b) in {"A": (1, 3), "B": (4, 7), "C": (8, 16), "D": (17, 28), "E": (29, 30)}.items():
    kg[(kcode >= a) & (kcode <= b)] = g

# WAD2M at the cells: 2001-2010 (regressions), whole record (f_min, f_max, model climatology)
Frec = FwAll[:, ii, jj].astype(np.float64)                                  # (252,N)
del FwAll, F11, fin
Frec0 = np.where(np.isfinite(Frec), Frec, 0.0)
fmin_c = Frec0.min(0); fmax_c = Frec0.max(0)
fclim20 = np.clip(Frec0.reshape(21, 12, N).mean(0), 0, 1)                   # what ibis/io/wad2m.py prescribes
i0 = (Y0 - 2000) * 12
Fw_c = Frec[i0:i0 + NMON]                                                    # (120,N), NaN = fill
Fw0_c = Frec0[i0:i0 + NMON]
del Frec, Frec0
log("WAD2M extracted")

# ----------------------------------------------------------------------------
# 2. Model monthly output at the cells (cached)
# ----------------------------------------------------------------------------
rows_m = 359 - ii                                                           # model row (N->S) of WAD2M row i
cache = os.path.join(OUT, "model_cells_2001-2010.npz")
if os.path.exists(cache):
    z = np.load(cache)
    assert z["ii"].shape[0] == N and (z["ii"] == ii).all() and (z["jj"] == jj).all()
    RUN, DRN, PRC, AET, WS, TS1 = (z[k] for k in ["RUN", "DRN", "PRC", "AET", "WS", "TS1"])
    log("model cells loaded from cache")
else:
    RUN = np.full((NMON, N), np.nan); DRN = np.full((NMON, N), np.nan)
    PRC = np.full((NMON, N), np.nan); AET = np.full((NMON, N), np.nan)
    WS = np.full((NMON, N, 6), np.nan, dtype=np.float32); TS1 = np.full((NMON, N), np.nan, dtype=np.float32)
    for t in range(NMON):
        y, m = Y0 + t // 12, t % 12 + 1
        h = nc.Dataset(os.path.join(HIND, f"ibis_monthly_{y}_{m:02d}.nc"))
        assert int(h.year) == y and int(h.month) == m
        rd = lambda k: np.ma.filled(h.variables[k][:].astype(np.float64), np.nan)
        RUN[t] = rd("amtrunof")[rows_m, jj]; DRN[t] = rd("amtdrain")[rows_m, jj]
        PRC[t] = rd("amprcp")[rows_m, jj]; AET[t] = rd("amaet")[rows_m, jj]
        WS[t] = rd("wsoi")[rows_m, jj, :]
        TS1[t] = rd("tsoi")[rows_m, jj, 0]
        h.close()
        if t % 12 == 11:
            log(f"  read {y}")
    np.savez(cache, ii=ii, jj=jj, RUN=RUN, DRN=DRN, PRC=PRC, AET=AET, WS=WS, TS1=TS1)
    log("model cells cached")
for nm, X in [("amtrunof", RUN), ("amtdrain", DRN), ("amprcp", PRC), ("amaet", AET), ("wsoi", WS)]:
    assert np.isfinite(X).all(), f"{nm} has NaN at analysis cells"
WS = WS.astype(np.float64)

# precipitation agrees with the GSWP3 cache of the precipitation analysis?
pc_prev = os.path.join(PREV, "monthly_gswp3_pr_1999-06_2010-12.npy")
if os.path.exists(pc_prev):
    G = np.load(pc_prev, mmap_mode="r")                                     # (139,360,720) S->N, from Jun 1999
    off = 7 + 12                                                            # Jan 2001
    gP = np.stack([np.asarray(G[off + t])[ii, jj] for t in range(NMON)]).astype(np.float64)
    checks["amprcp_vs_gswp3_cache"] = dict(max_abs_diff_mm=float(np.abs(gP - PRC).max()),
                                           mean_diff_mm=float((PRC - gP).mean()))
    del G, gP

# porosity
sd = nc.Dataset(SOITA)
slat = np.asarray(sd.variables["lat"][:])
assert slat[0] > slat[-1] and abs(slat[0] - 89.75) < 1e-6
sand = np.ma.filled(sd.variables["sandpct"][:].astype(np.float64), np.nan)[:, rows_m, jj].T   # (N,6)
clay = np.ma.filled(sd.variables["claypct"][:].astype(np.float64), np.nan)[:, rows_m, jj].T
sd.close()
n_tex_missing = int((~np.isfinite(sand) | ~np.isfinite(clay)).sum())
# the model casts NaN through np.round(...).astype(int32) and clips to 0 -> class lut[0,0] = silt loam
ms = np.clip(np.round(np.nan_to_num(sand, nan=0.0)).astype(np.int32), 0, 100)
mc = np.clip(np.round(np.nan_to_num(clay, nan=0.0)).astype(np.int32), 0, 100)
tex = TEXTURE_LUT[ms, mc]
poros = _POROS[tex]                                                          # (N,6)
W = (WS * poros[None] * HSOI[None, None] * 1000.0).sum(-1)                    # (120,N) mm
Wrz = (WS[:, :, :NRZ] * poros[None, :, :NRZ] * HSOI[None, None, :NRZ] * 1000.0).sum(-1)
S1 = WS[:, :, 0]
Wsat = (poros * HSOI[None] * 1000).sum(-1)                                   # pore volume of the column
checks["porosity"] = dict(n_cell_layers_texture_missing=n_tex_missing,
                          poros_wmedian_layer1=float(np.median(poros[:, 0])),
                          class_share_layer1={int(k): float((tex[:, 0] == k).mean()) for k in range(11)},
                          W_mean_mm=float(W.mean()), Wsat_mean_mm=float(Wsat.mean()),
                          W_over_Wsat_mean=float((W.mean(0) / Wsat).mean()),
                          share_cell_months_s1_ge_0p98=float((S1 >= 0.98).mean()))
log(f"porosity done (texture-missing cell-layers {n_tex_missing}); W mean {W.mean():.0f} mm of {Wsat.mean():.0f} mm pore volume")

# ----------------------------------------------------------------------------
# 3. Water-balance checks (does the runoff diagnostic close the budget?)
# ----------------------------------------------------------------------------
annP = PRC.reshape(NYR, 12, N).sum(1); annE = AET.reshape(NYR, 12, N).sum(1)
annR = RUN.reshape(NYR, 12, N).sum(1); annD = DRN.reshape(NYR, 12, N).sum(1)
# storage change Dec 2001 -> Dec 2010 against the 2002-2010 flux sums
dW = W[-1] - W[11]
resid = (annP[1:] - annE[1:] - annR[1:] - annD[1:]).sum(0) - dW               # mm over 9 years, = unaccounted
mP, mE, mR = annP.mean(0), annE.mean(0), annR.mean(0)
wb = {}
for b in ["all", "tropical", "temperate", "boreal"]:
    s = np.ones(N, bool) if b == "all" else band == b
    w = wet_area[s]
    wm = lambda v: float((v[s] * w).sum() / w.sum())
    over = (mR > (mP - mE))[s]
    wb[b] = dict(P_mm_yr=wm(mP), AET_mm_yr=wm(mE), runoff_mm_yr=wm(mR), drainage_mm_yr=wm(annD.mean(0)),
                 P_minus_AET_minus_runoff_mm_yr=wm(mP - mE - mR), dW_per_yr_2001_2010=wm(dW / 9.0),
                 unaccounted_mm_yr=wm(resid / 9.0),
                 share_wetland_area_runoff_gt_P_minus_AET=float(w[over].sum() / w.sum()),
                 share_wetland_area_runoff_gt_P=float(w[(mR > mP)[s]].sum() / w.sum()),
                 s1_mean_where_runoff_gt_PmE=float(S1.mean(0)[s][over].mean()) if over.any() else None,
                 s1_mean_elsewhere=float(S1.mean(0)[s][~over].mean()) if (~over).any() else None,
                 median_runoff_over_PmE=float(np.median((mR / np.maximum(mP - mE, 1.0))[s])))
checks["water_balance_wetland_cells"] = wb
checks["drainage_identically_zero"] = bool((DRN == 0).all())
# whole model land, from the yearly files
gl = []
for y in range(Y0, Y1 + 1):
    h = nc.Dataset(os.path.join(HIND, f"ibis_yearly_{y}.nc"))
    rd = lambda k: np.ma.filled(h.variables[k][:].astype(np.float64), np.nan)
    gr, gd, gp, ge = rd("aytrunof"), rd("aytdrain"), rd("ayprcp"), rd("ayaet")
    aw = np.repeat(area_row[::-1][:, None], 720, axis=1)                      # N->S area
    ok = np.isfinite(gr) & (np.abs(mlat)[:, None] < 60)
    gl.append(dict(year=y, runoff=float((gr * aw)[ok].sum() / aw[ok].sum()), drainage=float(np.nanmax(gd)),
                   P=float((gp * aw)[ok].sum() / aw[ok].sum()), AET=float((ge * aw)[ok].sum() / aw[ok].sum()),
                   share_cells_R_gt_P=float((gr > gp)[ok].mean())))
    h.close()
gl = pd.DataFrame(gl)
checks["model_land_60S_60N_yearly_mean"] = dict(P_mm_yr=float(gl.P.mean()), AET_mm_yr=float(gl.AET.mean()),
                                                runoff_mm_yr=float(gl.runoff.mean()), max_drainage=float(gl.drainage.max()),
                                                share_cells_runoff_gt_P=float(gl.share_cells_R_gt_P.mean()))
log("water balance: " + json.dumps(wb["all"]))

# ----------------------------------------------------------------------------
# 4. Predictors, anomalies
# ----------------------------------------------------------------------------
t_idx = np.arange(NMON)
smask = t_idx >= S0


def window_sum(X, n):
    C = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)], axis=0)
    out = np.full_like(X, np.nan)
    out[n - 1:] = C[n:] - C[:-n]
    return out


def lagged(X, L):
    out = np.full_like(X, np.nan)
    if L == 0:
        out[:] = X
    elif L == 0.5:
        out[1:] = 0.5 * (X[1:] + X[:-1])
    else:
        out[L:] = X[:-L]
    return out


FAMILY = {"runoff": "runoff", "drainage": "drainage", "precip": "precip", "pme": "pme",
          "W": "storage", "Wrz": "storage", "s1": "storage"}
PRED = []
for v, X in [("runoff", RUN + DRN), ("drainage", DRN), ("precip", PRC), ("pme", PRC - AET)]:
    wins = {"1mo": 1} if v == "drainage" else WINDOWS
    for wl, n in wins.items():
        PRED.append(dict(name=f"{v}_{wl}", var=v, family=FAMILY[v], kind="window", label=wl, months=n,
                         X=window_sum(X, n), floor=MIN_CLIM_FLUX_PER_MONTH * n))
for v, X, fl in [("W", W, 1.0), ("Wrz", Wrz, 1.0), ("s1", S1, 0.01)]:
    for ll, L in LAGS.items():
        PRED.append(dict(name=f"{v}_{ll}", var=v, family="storage", kind="lag", label=ll, months=L, X=lagged(X, L), floor=fl))
PNAMES = [p["name"] for p in PRED]
log(f"{len(PRED)} predictors")


def anomalies(X, floor, mask):
    Xs = np.where(mask[:, None], X, np.nan)
    Xy = Xs.reshape(NYR, 12, -1)
    clim = np.nanmean(Xy, axis=0)
    an = (Xy - clim[None]).reshape(NMON, -1)
    den = np.where(clim >= floor, clim, np.nan)
    rel = an / np.tile(den, (NYR, 1))
    sdv = np.nanstd(an, axis=0)
    sdv = np.where(sdv > 0, sdv, np.nan)
    return clim, an, rel, an / sdv[None]


def fw_anomalies(mask, min_years=MIN_YEARS_CLIM):
    Fs = np.where(mask[:, None], Fw_c, np.nan)
    Fy = Fs.reshape(NYR, 12, N)
    nyv = np.isfinite(Fy).sum(0)
    clim = np.nanmean(Fy, axis=0)
    clim[nyv < min_years] = np.nan
    den = np.where(clim >= MIN_CLIM_FW, clim, np.nan)
    return ((Fy - clim[None]) / den[None]).reshape(NMON, N)


def nan_corr_slope(x, y, min_n=MIN_PAIRS):
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(0)
    xz = np.where(ok, x, 0.0); yz = np.where(ok, y, 0.0)
    mx = xz.sum(0) / n; my = yz.sum(0) / n
    dx = np.where(ok, x - mx, 0.0); dy = np.where(ok, y - my, 0.0)
    sxx = (dx * dx).sum(0); syy = (dy * dy).sum(0); sxy = (dx * dy).sum(0)
    r = sxy / np.sqrt(sxx * syy)
    slope = sxy / sxx
    bad = (n < min_n) | ~np.isfinite(r)
    r[bad] = np.nan; slope[bad] = np.nan
    return r, slope, n


rel_Fw = fw_anomalies(smask)
REL, ZST, CLIM = {}, {}, {}
res = {}
for p in PRED:
    clim, an, rel, zz = anomalies(p["X"], p["floor"], smask)
    CLIM[p["name"]] = clim
    REL[p["name"]] = rel.astype(np.float32); ZST[p["name"]] = zz.astype(np.float32)
    r_rel, s_rel, n_rel = nan_corr_slope(rel, rel_Fw)
    r_z, s_z, _ = nan_corr_slope(zz, rel_Fw)
    res[p["name"]] = dict(r=r_rel, slope=s_rel, n=n_rel, r_z=r_z, slope_z=s_z)
log("per-cell regressions done; median r: " + ", ".join(f"{k}={np.nanmedian(v['r']):.3f}" for k, v in res.items() if k.endswith(("_1mo", "_3mo", "lag0"))))

CAND = [p["name"] for p in PRED if p["family"] != "drainage"]
CAND_FAM = np.array([FAMILY[n.split("_")[0]] for n in CAND])


def best_of(names, rdict):
    Rm = np.stack([rdict[n] for n in names])
    Rf = np.where(np.isfinite(Rm), Rm, -np.inf)
    k = np.argmax(Rf, axis=0)
    has = np.isfinite(Rm).any(0)
    return np.where(has, np.array(names)[k], ""), np.where(has, Rf[k, np.arange(N)], np.nan), k, has


best_name, best_r, best_k, has_best = best_of(CAND, {n: res[n]["r"] for n in CAND})
best_slope = np.where(has_best, np.stack([res[n]["slope"] for n in CAND])[best_k, np.arange(N)], np.nan)
best_slope_z = np.where(has_best, np.stack([res[n]["slope_z"] for n in CAND])[best_k, np.arange(N)], np.nan)
best_family = np.array([FAMILY[n.split("_")[0]] if n else "" for n in best_name])
fam_best = {}
for fam in ["runoff", "storage", "precip", "pme"]:
    names = [n for n in CAND if FAMILY[n.split("_")[0]] == fam]
    nm, rr, kk, hh = best_of(names, {n: res[n]["r"] for n in names})
    fam_best[fam] = dict(name=nm, r=rr,
                         slope=np.where(hh, np.stack([res[n]["slope"] for n in names])[kk, np.arange(N)], np.nan),
                         slope_z=np.where(hh, np.stack([res[n]["slope_z"] for n in names])[kk, np.arange(N)], np.nan))

# null: shuffle whole years of the WAD2M anomalies against the predictors
rng = np.random.default_rng(20261016)
null_best = []
null_fam = {}
for k in range(N_NULL):
    while True:
        perm = rng.permutation(NYR)
        if (perm != np.arange(NYR)).all():
            break
    rf = rel_Fw.reshape(NYR, 12, N)[perm].reshape(NMON, N)
    rr = {n: nan_corr_slope(REL[n].astype(np.float64), rf)[0] for n in CAND}
    null_best.append(best_of(CAND, rr)[1])
    for fam in ["runoff", "storage", "precip", "pme"]:
        names = [n for n in CAND if FAMILY[n.split("_")[0]] == fam]
        null_fam.setdefault(fam, []).append(best_of(names, rr)[1])
null_best = np.stack(null_best)                                              # (N_NULL, N)
null_fam = {k: np.stack(v) for k, v in null_fam.items()}
log("year-shuffle null done")

# ----------------------------------------------------------------------------
# 5. Summary per group x predictor
# ----------------------------------------------------------------------------
def wquantiles(v, w, qs=(0.25, 0.5, 0.75)):
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return [np.nan] * len(qs)
    v = v[ok]; w = w[ok]
    o = np.argsort(v); v = v[o]; w = w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return list(np.interp(qs, cw, v))


GROUPS = [("band", "all", np.ones(N, bool))]
for b in ["tropical", "temperate", "boreal"]:
    GROUPS.append(("band", b, band == b))
for g in ["A", "B", "C", "D", "E"]:
    GROUPS.append(("koppen", g, kg == g))

pinfo = {p["name"]: p for p in PRED}
rows = []
for gtype, gname, sel in GROUPS:
    w = wet_area[sel]
    entries = [(n, pinfo[n]["var"], pinfo[n]["family"], pinfo[n]["label"], res[n]["r"], res[n]["slope"], res[n]["r_z"], res[n]["slope_z"]) for n in PNAMES]
    entries.append(("best", "best", "any", "best", best_r, best_slope, np.full(N, np.nan), best_slope_z))
    for fam, d in fam_best.items():
        entries.append((f"best_{fam}", fam, fam, "best", d["r"], d["slope"], np.full(N, np.nan), d["slope_z"]))
    entries.append(("best_null_yearshuffle", "null", "any", "best", np.nanmean(null_best, 0), np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)))
    for fam in null_fam:
        entries.append((f"best_{fam}_null_yearshuffle", "null", fam, "best", np.nanmean(null_fam[fam], 0), np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)))
    for (n, var, fam, lab, r, s, rz, sz) in entries:
        r = r[sel]; s = s[sel]; rz = rz[sel]; sz = sz[sel]
        ok = np.isfinite(r)
        rq = wquantiles(r, w); sq = wquantiles(s, w); rzq = wquantiles(rz, w); szq = wquantiles(sz, w)
        nb = np.isfinite(best_r[sel]).sum()
        row = dict(group_type=gtype, group=gname, predictor=n, variable=var, family=fam, window_or_lag=lab,
                   n_cells=int(sel.sum()), n_cells_fit=int(ok.sum()), wetland_area_Mkm2=w.sum() / 1e6,
                   r_wmedian=rq[1], r_q25=rq[0], r_q75=rq[2],
                   share_area_r_gt_0p5=float(w[ok][r[ok] > 0.5].sum() / w[ok].sum()) if ok.any() else np.nan,
                   share_cells_r_gt_0p5=float((r[ok] > 0.5).mean()) if ok.any() else np.nan,
                   slope_pct_per_pct_wmedian=sq[1], slope_q25=sq[0], slope_q75=sq[2],
                   r_std_wmedian=rzq[1], r_std_q25=rzq[0], r_std_q75=rzq[2],
                   slope_pct_per_sd_wmedian=100 * szq[1] if np.isfinite(szq[1]) else np.nan,
                   slope_pct_per_sd_q25=100 * szq[0] if np.isfinite(szq[0]) else np.nan,
                   slope_pct_per_sd_q75=100 * szq[2] if np.isfinite(szq[2]) else np.nan,
                   share_cells_best=float((best_name[sel] == n).sum() / max(nb, 1)) if n in CAND else np.nan,
                   share_area_best=float(w[best_name[sel] == n].sum() / w.sum()) if n in CAND else
                   (float(w[best_family[sel] == fam].sum() / w.sum()) if n.startswith("best_") and fam in fam_best
                    and not n.endswith("null_yearshuffle") else np.nan))
        if n.endswith("null_yearshuffle"):
            nb_all = (null_best if n == "best_null_yearshuffle" else null_fam[fam])[:, sel]
            row["share_area_r_gt_0p5"] = float(np.mean([w[np.isfinite(x)][x[np.isfinite(x)] > 0.5].sum() / w[np.isfinite(x)].sum() for x in nb_all]))
            row["r_wmedian"] = float(np.mean([wquantiles(x, w)[1] for x in nb_all]))
        rows.append(row)
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "summary_band_predictor.csv"), index=False, float_format="%.4g")
log("summary written")

# ----------------------------------------------------------------------------
# 6. Seasonal cycle: lag of Fw behind each base predictor
# ----------------------------------------------------------------------------
def xcorr_lags(a, b):
    out = np.full((12,) + a.shape[1:], np.nan)
    a0 = a - a.mean(0); sa = np.sqrt((a0 ** 2).sum(0))
    for L in range(12):
        bs = np.roll(b, L, axis=0)
        b0 = bs - bs.mean(0); sb = np.sqrt((b0 ** 2).sum(0))
        out[L] = (a0 * b0).sum(0) / (sa * sb)
    return out


clim_Fw0 = Fw0_c.reshape(NYR, 12, N).mean(0)
BASE = {"runoff": (RUN + DRN, 5.0), "precip": (PRC, 10.0), "pme": (PRC - AET, 10.0), "AET": (AET, 10.0),
        "W": (W, 5.0), "Wrz": (Wrz, 5.0), "s1": (S1, 0.02)}
seas = {}
lag_rows = []
for v, (X, amp_min) in BASE.items():
    cX = X.reshape(NYR, 12, N).mean(0)
    xc = xcorr_lags(clim_Fw0, cX)
    ok_amp = ((clim_Fw0.max(0) - clim_Fw0.min(0)) >= 0.01) & ((cX.max(0) - cX.min(0)) >= amp_min)
    xcf = np.where(np.isfinite(xc), xc, -np.inf)
    li = np.argmax(xcf, axis=0)
    lagv = np.where(li > 6, li - 12, li).astype(float)
    rbest = xcf[li, np.arange(N)]; r0 = xc[0].copy()
    lagv[~ok_amp] = np.nan; rbest[~ok_amp] = np.nan; r0[~ok_amp] = np.nan
    seas[v] = dict(lag=lagv, r=rbest, r0=r0, clim=cX)
    for gtype, gname, sel in GROUPS:
        w = wet_area[sel]; lg = lagv[sel]; ok = np.isfinite(lg)
        lq = wquantiles(lg, w)
        lag_rows.append(dict(group_type=gtype, group=gname, variable=v, scope="per-cell", n_cells=int(ok.sum()),
                             lag_wmedian=lq[1], lag_q25=lq[0], lag_q75=lq[2],
                             r_at_best_lag_wmedian=wquantiles(rbest[sel], w)[1], r_lag0_wmedian=wquantiles(r0[sel], w)[1]))
        for hh in ["N", "S"]:
            s2 = sel & (hemi == hh)
            if s2.sum() < 20:
                continue
            Fb = (clim_Fw0[:, s2] * carea[s2]).sum(1)
            Xb = (cX[:, s2] * wet_area[s2]).sum(1) / wet_area[s2].sum()
            xcb = xcorr_lags(Fb[:, None], Xb[:, None])[:, 0]
            L = int(np.argmax(xcb)); L = L - 12 if L > 6 else L
            lag_rows.append(dict(group_type=gtype, group=gname, variable=v, scope=f"aggregate-{hh}", n_cells=int(s2.sum()),
                                 lag_wmedian=L, r_at_best_lag_wmedian=float(xcb.max()), r_lag0_wmedian=float(xcb[0]),
                                 Fw_area_peak_month=int(np.argmax(Fb)) + 1, predictor_peak_month=int(np.argmax(Xb)) + 1))
lagdf = pd.DataFrame(lag_rows)
lagdf.to_csv(os.path.join(OUT, "seasonal_lag.csv"), index=False, float_format="%.4g")
log("seasonal lags written")

# ----------------------------------------------------------------------------
# 7. Per-cell CSV
# ----------------------------------------------------------------------------
pc = pd.DataFrame(dict(lat=clat, lon=clon, band=band, koppen_group=kg, koppen_code=kcode, cell_area_km2=carea,
                       mean_Fw_fill0_2000_2010=cmean, wetland_area_km2=wet_area, n_valid_months_2001_2010=np.isfinite(Fw_c).sum(0),
                       f_min_2000_2020=fmin_c, f_max_2000_2020=fmax_c,
                       P_mm_yr=mP, AET_mm_yr=mE, runoff_mm_yr=mR, runoff_over_P_minus_AET=mR / np.maximum(mP - mE, 1.0),
                       W_mean_mm=W.mean(0), W_pore_volume_mm=Wsat, s1_mean=S1.mean(0), poros_layer1=poros[:, 0],
                       best_predictor=best_name, best_family=best_family, best_r=best_r,
                       best_slope_pct_per_pct=best_slope, best_slope_pct_per_sd=100 * best_slope_z,
                       best_r_null_yearshuffle=np.nanmean(null_best, 0)))
for fam, d in fam_best.items():
    pc[f"best_{fam}_predictor"] = d["name"]; pc[f"best_{fam}_r"] = d["r"]
for v in BASE:
    pc[f"seas_lag_vs_{v}"] = seas[v]["lag"]; pc[f"seas_r_vs_{v}"] = seas[v]["r"]
for n in PNAMES:
    pc[f"r_{n}"] = res[n]["r"]; pc[f"slope_{n}"] = res[n]["slope"]
    pc[f"rstd_{n}"] = res[n]["r_z"]; pc[f"slopestd_{n}"] = 100 * res[n]["slope_z"]
pc.to_csv(os.path.join(OUT, "per_cell_results.csv"), index=False, float_format="%.5g")
log("per_cell_results.csv written")

# ----------------------------------------------------------------------------
# 8. Annual group totals
# ----------------------------------------------------------------------------
years = np.arange(Y0, Y1 + 1)
annFw0 = Fw0_c.reshape(NYR, 12, N).mean(1)
nyr_valid = np.isfinite(Fw_c.reshape(NYR, 12, N)).sum(0)
fixed = (nyr_valid == NYR)                                                   # (12,N) months valid in all years
nfix = fixed.sum(0)
annFw_fixed = np.where(nfix > 0, (Fw0_c.reshape(NYR, 12, N) * fixed[None]).sum(1) / np.maximum(nfix, 1), np.nan)


def annual_of(X, months_mask=None):
    Xy = X.reshape(NYR, 12, N)
    full = np.isfinite(Xy).all(1)
    if months_mask is None:
        a = Xy.mean(1)
    else:
        a = np.where(nfix > 0, (np.nan_to_num(Xy) * months_mask[None]).sum(1) / np.maximum(nfix, 1), np.nan)
    return np.where(full, a, np.nan)


detrend = lambda v, x: v - np.polyval(np.polyfit(x, v, 1), x)
ANN = {n: (annual_of(pinfo[n]["X"]), annual_of(pinfo[n]["X"], fixed)) for n in CAND}
ts_rows, st_rows = [], []
group_best = {}
for gtype, gname, sel in GROUPS:
    a = carea[sel]; w = wet_area[sel]
    WA = (annFw0[:, sel] * a).sum(1)
    okf = nfix[sel] > 0
    WAf = (annFw_fixed[:, sel][:, okf] * a[okf]).sum(1)
    wa_pct = 100 * (WA / WA.mean() - 1); waf_pct = 100 * (WAf / WAf.mean() - 1)
    for k, y in enumerate(years):
        ts_rows.append(dict(group_type=gtype, group=gname, year=int(y), predictor="wetland_area_km2", value=WA[k], anom_pct=wa_pct[k]))
        ts_rows.append(dict(group_type=gtype, group=gname, year=int(y), predictor="wetland_area_fixedseason_km2", value=WAf[k], anom_pct=waf_pct[k]))
    sub = summ[(summ.group_type == gtype) & (summ.group == gname) & summ.predictor.isin(CAND)]
    group_best[(gtype, gname)] = sub.sort_values("r_wmedian", ascending=False).predictor.iloc[0]
    for n in CAND:
        A_, Af_ = ANN[n]
        Xb = np.array([(A_[k, sel] * w).sum() / w.sum() if np.isfinite(A_[k, sel]).all() else np.nan for k in range(NYR)])
        Xbf = np.array([(Af_[k, sel][okf] * w[okf]).sum() / w[okf].sum() if np.isfinite(Af_[k, sel][okf]).all() else np.nan for k in range(NYR)])
        yok = np.isfinite(Xb)
        x_pct = 100 * (Xb / np.nanmean(Xb) - 1); xf_pct = 100 * (Xbf / np.nanmean(Xbf) - 1)
        for k, y in enumerate(years):
            ts_rows.append(dict(group_type=gtype, group=gname, year=int(y), predictor=n, value=Xb[k], anom_pct=x_pct[k]))
        yy = years[yok]; xx = x_pct[yok]; ww = wa_pct[yok]
        # anomalies recentred on the years used
        xx = xx - xx.mean(); ww = ww - ww.mean()
        r, p = stats.pearsonr(xx, ww); sl = np.polyfit(xx, ww, 1)[0]
        rd_, pd_ = stats.pearsonr(detrend(xx, yy), detrend(ww, yy)); sld = np.polyfit(detrend(xx, yy), detrend(ww, yy), 1)[0]
        wf = waf_pct[yok] - waf_pct[yok].mean()
        rf, pf = stats.pearsonr(xx, wf); slf = np.polyfit(xx, wf, 1)[0]
        xfm = xf_pct[yok] - xf_pct[yok].mean()
        rfm, pfm = stats.pearsonr(xfm, wf); slfm = np.polyfit(xfm, wf, 1)[0]
        st_rows.append(dict(group_type=gtype, group=gname, predictor=n, family=pinfo[n]["family"], n_years=int(yok.sum()),
                            first_year=int(yy[0]), mean_wetland_area_Mkm2=WA.mean() / 1e6,
                            sd_wetland_area_pct=ww.std(ddof=1), sd_predictor_pct=xx.std(ddof=1),
                            r=r, p=p, slope_pct_per_pct=sl, slope_pct_per_sd=sl * xx.std(ddof=1),
                            r_detrended=rd_, p_detrended=pd_, slope_detrended=sld,
                            r_fixedseason=rf, p_fixedseason=pf, slope_fixedseason=slf,
                            r_fixedseason_matched=rfm, p_fixedseason_matched=pfm, slope_fixedseason_matched=slfm,
                            trend_wetland_pct_per_yr=np.polyfit(yy, ww, 1)[0], trend_predictor_pct_per_yr=np.polyfit(yy, xx, 1)[0]))
ts = pd.DataFrame(ts_rows); ts.to_csv(os.path.join(OUT, "band_timeseries.csv"), index=False, float_format="%.6g")
st = pd.DataFrame(st_rows); st.to_csv(os.path.join(OUT, "band_timeseries_stats.csv"), index=False, float_format="%.4g")
log("annual group series written; monthly-best per group: " + json.dumps({f"{a}:{b}": v for (a, b), v in group_best.items()}))

# ----------------------------------------------------------------------------
# 9. Fitted scheme f = clip(f_clim (1 + beta A), f_min, f_max), per band
# ----------------------------------------------------------------------------
train = (t_idx >= S0) & (t_idx <= TRAIN_LAST)
test = t_idx > TRAIN_LAST
rel_Fw_train = fw_anomalies(train, min_years=4)    # 5-6 training years per calendar month
fcT = np.tile(fclim20, (NYR, 1))                                             # (120,N)
blk_all = (np.floor((clat + 90) / 10) * 36 + np.floor((clon + 180) / 10)).astype(int)


def scheme_A(n, form="rel", sel=None):
    """rel: (X - clim) / max(clim, floor) (the specified form); std: (X - clim) / sd (per cell);
    band: one relative anomaly per month for the whole band, sum_c w (X - clim) / sum_c w max(clim, floor)."""
    if form == "std":
        A = ZST[n].astype(np.float64)
    elif form == "band":
        clim = CLIM[n]; X = pinfo[n]["X"]; w = wet_area[sel]
        an = (X.reshape(NYR, 12, N) - clim[None])[:, :, sel]
        den = np.maximum(clim, pinfo[n]["floor"])[:, sel]
        Ab = ((an * w).sum(-1) / (den * w).sum(-1)[None]).reshape(NMON)
        A = np.zeros((NMON, N)); A[:, sel] = Ab[:, None]
    else:
        clim = CLIM[n]
        X = pinfo[n]["X"]
        den = np.maximum(clim, pinfo[n]["floor"])
        A = ((X.reshape(NYR, 12, N) - clim[None]) / den[None]).reshape(NMON, N)
    return np.where(np.isfinite(A), A, 0.0)


def fit_beta(A, sel):
    a = carea[sel]
    m = np.where(train)[0]
    At, Ft, Ot = A[m][:, sel], fcT[m][:, sel], Fw0_c[m][:, sel]
    lo, hi = fmin_c[sel], fmax_c[sel]
    yr = m // 12
    starts = np.r_[0, np.where(np.diff(yr) > 0)[0] + 1]
    ny = len(starts)
    blk = np.unique(blk_all[sel], return_inverse=True)[1]
    nb = blk.max() + 1
    act = At[Ft > 0]
    p1, p99 = np.percentile(act, [1, 99])
    sdA = float(np.std(np.clip(act, p1, p99)))                               # scale of A without its extreme tail
    sdA = sdA if sdA > 0 else 1.0

    def sse_grid(grid):
        S = np.zeros((len(grid), ny, nb))
        for k, b in enumerate(grid):
            E = a * (np.clip(Ft * (1 + b * At), lo, hi) - Ot) ** 2
            Ey = np.add.reduceat(E, starts, axis=0)
            for y in range(ny):
                S[k, y] = np.bincount(blk, weights=Ey[y], minlength=nb)
        return S

    span = 3.0
    for _ in range(4):
        grid = np.linspace(-span, span, 241) / sdA
        S = sse_grid(grid)
        tot = S.sum((1, 2))
        k0 = int(np.nanargmin(tot))
        if 0 < k0 < len(grid) - 1:
            break
        span *= 4
    obj = lambda b: float((a * (np.clip(Ft * (1 + b * At), lo, hi) - Ot) ** 2).sum())
    kk = min(max(k0, 1), len(grid) - 2)
    opt = optimize.minimize_scalar(obj, bounds=(grid[kk - 1], grid[kk + 1]), method="bounded", options=dict(xatol=1e-6 / sdA))
    # the objective is not smooth (clipping, heavy-tailed A): keep the best of the refined point, the grid minimum and 0
    trial = {float(opt.x): float(opt.fun), float(grid[k0]): obj(float(grid[k0])), 0.0: obj(0.0)}
    beta = min(trial, key=trial.get)
    sse_best = trial[beta]

    def argmin_curve(c):
        k = int(np.argmin(c))
        if 0 < k < len(c) - 1:
            y0, y1, y2 = c[k - 1], c[k], c[k + 1]
            d = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / d if d > 0 else 0.0
            return grid[k] + off * (grid[1] - grid[0])
        return grid[k]

    rb = np.random.default_rng(7)
    S = np.where(np.isfinite(S), S, np.inf)
    Sy = S.sum(2); Sb = S.sum(1)
    by = np.array([argmin_curve(Sy @ np.bincount(rb.integers(0, ny, ny), minlength=ny)) for _ in range(N_BOOT)])
    bb = np.array([argmin_curve(Sb @ np.bincount(rb.integers(0, nb, nb), minlength=nb)) for _ in range(N_BOOT)])
    return dict(beta=beta, sdA=sdA, sse=sse_best, sse_clim=obj(0.0), boot_year=by, boot_block=bb, n_blocks=int(nb),
                grid_edge=bool(k0 in (0, len(grid) - 1)), grid_step=float(grid[1] - grid[0]))


def evaluate(A, beta, sel, months):
    a = carea[sel]
    m = np.where(months)[0]
    At, Ft, Ot = A[m][:, sel], fcT[m][:, sel], Fw0_c[m][:, sel]
    Pt = np.clip(Ft * (1 + beta * At), fmin_c[sel], fmax_c[sel])
    Fb, Cb, Ob = (Pt * a).sum(1), (Ft * a).sum(1), (Ot * a).sum(1)
    r_band = stats.pearsonr(Fb - Cb, Ob - Cb)[0] if np.std(Fb - Cb) > 0 else np.nan
    skill_band = 1 - ((Fb - Ob) ** 2).sum() / ((Cb - Ob) ** 2).sum()
    rc, _, _ = nan_corr_slope(Pt - Ft, Ot - Ft, min_n=12)
    pooled = 1 - (a * (Pt - Ot) ** 2).sum() / (a * (Ft - Ot) ** 2).sum()
    yr = m // 12
    full_years = [y for y in np.unique(yr) if (yr == y).sum() == 12]
    Fy = np.array([Fb[yr == y].sum() for y in full_years]); Oy = np.array([Ob[yr == y].sum() for y in full_years])
    Cy = np.array([Cb[yr == y].sum() for y in full_years])
    r_ann = stats.pearsonr(Fy, Oy)[0] if len(full_years) >= 3 and np.std(Fy) > 0 else np.nan
    skill_ann = 1 - ((Fy - Oy) ** 2).sum() / ((Cy - Oy) ** 2).sum()
    return dict(r_band_monthly_anom=r_band, skill_band_monthly=skill_band, r_cell_anom_wmedian=wquantiles(rc, wet_area[sel])[1],
                skill_pooled_cells=pooled, r_band_annual=r_ann, skill_band_annual=skill_ann, n_years_annual=len(full_years),
                bias_pct=100 * (Fb.sum() - Ob.sum()) / Ob.sum(), clim_bias_pct=100 * (Cb.sum() - Ob.sum()) / Ob.sum(),
                obs_band_annual_sd_pct=100 * np.std(Oy, ddof=1) / Oy.mean() if len(Oy) > 1 else np.nan,
                pred_band_annual_sd_pct=100 * np.std(Fy, ddof=1) / Fy.mean() if len(Fy) > 1 else np.nan), Fb, Cb, Ob


fit_rows = []
series_rows = []
scheme_choice = {}
for b in ["tropical", "temperate", "boreal"]:
    sel = band == b
    w = wet_area[sel]
    cands = []
    for fam in ["runoff", "storage", "precip", "pme"]:
        names = [n for n in CAND if FAMILY[n.split("_")[0]] == fam]
        med = {}
        for n in names:
            rel = np.where(train[:, None], REL[n][:, sel].astype(np.float64), np.nan)
            med[n] = wquantiles(nan_corr_slope(rel, rel_Fw_train[:, sel])[0], w)[1]
        nbest = max(med, key=lambda k: med[k] if np.isfinite(med[k]) else -9)
        cands.append((fam, nbest, med[nbest]))
    fits = {}
    for (fam, n, rtrain), form in [(c, fm) for c in cands for fm in ["rel", "std", "band"]]:
        A = scheme_A(n, form, sel)
        f = fit_beta(A, sel)
        ev_tr, _, _, _ = evaluate(A, f["beta"], sel, train)
        ev_te, Fb, Cb, Ob = evaluate(A, f["beta"], sel, test)
        fits[(n, form)] = (f, A)
        row = dict(band=b, family=fam, predictor=n, form=form, r_train_cell_wmedian=rtrain, beta=f["beta"],
                   beta_se_year_boot=np.std(f["boot_year"], ddof=1),
                   beta_ci95_year_lo=np.percentile(f["boot_year"], 2.5), beta_ci95_year_hi=np.percentile(f["boot_year"], 97.5),
                   beta_se_block_boot=np.std(f["boot_block"], ddof=1),
                   beta_ci95_block_lo=np.percentile(f["boot_block"], 2.5), beta_ci95_block_hi=np.percentile(f["boot_block"], 97.5),
                   n_blocks=f["n_blocks"], sd_A_robust=f["sdA"], beta_times_sdA_pct=100 * f["beta"] * f["sdA"],
                   train_sse_reduction_pct=100 * (1 - f["sse"] / f["sse_clim"]), grid_edge=f["grid_edge"])
        row.update({f"train_{k}": v for k, v in ev_tr.items()})
        row.update({f"test_{k}": v for k, v in ev_te.items()})
        fit_rows.append(row)
        log(f"scheme {b} {n} {form}: beta={f['beta']:.4g} (year-boot se {row['beta_se_year_boot']:.3g}, block-boot se {row['beta_se_block_boot']:.3g}); "
            f"train SSE -{row['train_sse_reduction_pct']:.2f}%; test r_band={ev_te['r_band_monthly_anom']:.2f} skill={ev_te['skill_band_monthly']:.3f}")
    fb = pd.DataFrame([r for r in fit_rows if r["band"] == b and r["form"] == "rel"])
    chosen = fb.sort_values("train_sse_reduction_pct", ascending=False).predictor.iloc[0]
    scheme_choice[b] = chosen
    f, A = fits[(chosen, "rel")]
    for months, lab in [(train, "train"), (test, "test")]:
        _, Fb, Cb, Ob = evaluate(A, f["beta"], sel, months)
        for k, t in enumerate(np.where(months)[0]):
            series_rows.append(dict(band=b, predictor=chosen, period=lab, year=Y0 + t // 12, month=t % 12 + 1,
                                    obs_km2=Ob[k], clim_km2=Cb[k], scheme_km2=Fb[k]))
fitdf = pd.DataFrame(fit_rows)
fitdf["chosen_by_train_sse"] = [(scheme_choice[b] == p) and (fm == "rel") for b, p, fm in zip(fitdf.band, fitdf.predictor, fitdf.form)]
fitdf.to_csv(os.path.join(OUT, "scheme_fit.csv"), index=False, float_format="%.5g")
pd.DataFrame(series_rows).to_csv(os.path.join(OUT, "scheme_band_monthly.csv"), index=False, float_format="%.6g")
log("scheme fits written: " + json.dumps(scheme_choice))

# ----------------------------------------------------------------------------
# 10. Regions: floodplains and peatlands
# ----------------------------------------------------------------------------
REGIONS = {
    "Amazon central floodplain": ((-4.5, -1.0, -70.0, -54.0),
                                  "local rain peak Feb-Apr; mainstem discharge (Obidos) peaks May-Jun, low Oct-Nov; "
                                  "floodplain inundation max May-Jul (lags local rain/runoff by ~2-3 months)"),
    "Congo Cuvette Centrale": ((-3.0, 2.5, 15.5, 21.0),
                               "bimodal rain (Mar-May, Sep-Nov); Congo at Brazzaville peaks Nov-Dec (secondary Apr-May); "
                               "inundation max Oct-Dec"),
    "Pantanal": ((-20.5, -15.5, -58.5, -55.0),
                 "rain Nov-Mar (peak Jan); Paraguay at Ladario peaks ~May-Jul; inundation max Mar-Jun "
                 "(north) to Jun-Jul (south), i.e. 2-5 months after the rain peak"),
    "Sudd": ((6.0, 10.0, 29.0, 32.5),
             "rain May-Oct (peak Jul-Aug); White Nile inflow (Mongalla) peaks Aug-Oct; extent max Sep-Nov"),
    "Hudson Bay Lowlands": ((50.5, 57.5, -95.0, -80.0),
                            "snowmelt freshet May (Albany/Attawapiskat/Moose); runoff low Jan-Mar under frozen ground; "
                            "inundation from thaw (May) through Oct"),
    "West Siberian Lowland": ((55.0, 66.0, 63.0, 85.0),
                              "snowmelt peak May-Jun in local rivers (Ob at Salekhard peaks Jun, recession to Aug); "
                              "inundation peaks May-Jul after melt"),
}
reg_rows, regclim_rows = [], []
MN = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
for rname, ((la0, la1, lo0, lo1), expect) in REGIONS.items():
    s = (clat >= la0) & (clat <= la1) & (clon >= lo0) & (clon <= lo1)
    if s.sum() < 5:
        continue
    w = wet_area[s]; a = carea[s]
    Fa = (clim_Fw0[:, s] * a).sum(1)
    cl = {v: (seas[v]["clim"][:, s] * w).sum(1) / w.sum() for v in ["runoff", "precip", "pme", "AET", "W", "s1"]}
    for mth in range(12):
        regclim_rows.append(dict(region=rname, month=mth + 1, Fw_area_km2=Fa[mth], **{f"{v}": cl[v][mth] for v in cl}))
    row = dict(region=rname, box=f"{la0}..{la1}N, {lo0}..{lo1}E", n_cells=int(s.sum()), wetland_area_km2=w.sum(),
               Fw_area_peak=MN[np.argmax(Fa)], Fw_area_min=MN[np.argmin(Fa)],
               Fw_seasonal_amplitude_pct=100 * (Fa.max() - Fa.min()) / Fa.mean(), expectation=expect)
    for v in ["runoff", "precip", "pme", "W", "s1"]:
        xc = xcorr_lags(Fa[:, None], cl[v][:, None])[:, 0]
        L = int(np.argmax(xc)); L = L - 12 if L > 6 else L
        row[f"{v}_peak"] = MN[np.argmax(cl[v])]
        row[f"lag_Fw_behind_{v}"] = L; row[f"r_at_lag_{v}"] = float(xc.max()); row[f"r_lag0_{v}"] = float(xc[0])
    row["runoff_mm_yr"] = float((mR[s] * w).sum() / w.sum()); row["P_mm_yr"] = float((mP[s] * w).sum() / w.sum())
    row["AET_mm_yr"] = float((mE[s] * w).sum() / w.sum())
    row["runoff_over_P_minus_AET"] = row["runoff_mm_yr"] / max(row["P_mm_yr"] - row["AET_mm_yr"], 1.0)
    row["share_months_s1_ge_0p98"] = float((S1[:, s] >= 0.98).mean())
    row["W_over_pore_volume"] = float((W[:, s].mean(0) / Wsat[s]).mean())
    # interannual: annual area vs annual runoff / W / P (n = 10), and cell r of the band-best predictors
    WAr = (annFw0[:, s] * a).sum(1)
    for v, X in [("runoff", RUN + DRN), ("W", W), ("precip", PRC)]:
        Xa = (X.reshape(NYR, 12, N).mean(1)[:, s] * w).sum(1) / w.sum()
        row[f"annual_r_vs_{v}"] = stats.pearsonr(Xa, WAr)[0]
    for fam in ["runoff", "storage", "precip"]:
        row[f"cell_best_{fam}_r_wmedian"] = wquantiles(fam_best[fam]["r"][s], w)[1]
    reg_rows.append(row)
regdf = pd.DataFrame(reg_rows)
regdf.to_csv(os.path.join(OUT, "regional_seasonality.csv"), index=False, float_format="%.4g")
pd.DataFrame(regclim_rows).to_csv(os.path.join(OUT, "regional_climatology.csv"), index=False, float_format="%.5g")
log("regional tables written")

# ----------------------------------------------------------------------------
# 11. Figures
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#9a9a95", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "axes.titlecolor": "#0b0b0b", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
landbg = np.where(land_mask, 1.0, np.nan)
rmap = np.full((360, 720), np.nan); rmap[ii, jj] = best_r
FAMS3 = {"runoff": 0, "storage": 1, "precip": 2, "pme": 2}
fmap = np.full((360, 720), np.nan); fmap[ii, jj] = [FAMS3.get(f, np.nan) for f in best_family]
fig, axes = plt.subplots(2, 1, figsize=(11, 9.5))
ax = axes[0]
ax.pcolormesh(lon, lat, landbg, cmap=ListedColormap(["#e8e7e3"]), vmin=0, vmax=1, shading="auto")
div = LinearSegmentedColormap.from_list("div", [(0, "#8f1d1d"), (0.25, "#e34948"), (0.5, "#f0efec"), (0.75, "#2a78d6"), (1, "#0d366b")])
pm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(rmap), cmap=div, vmin=-1, vmax=1, shading="auto")
cb = fig.colorbar(pm, ax=ax, orientation="vertical", fraction=0.025, pad=0.01); cb.set_label("Pearson r, best predictor")
ax.set_title("WAD2M relative inundation anomaly vs the IBIS s7 hindcast's own water variables, Jul 2001-Dec 2010:\n"
             f"r at the cell's best of {len(CAND)} predictors (runoff / storage / P / P-AET, windows and lags); "
             f"weighted median {wquantiles(best_r, wet_area)[1]:.2f}, year-shuffle null {wquantiles(np.nanmean(null_best, 0), wet_area)[1]:.2f}",
             loc="left", fontsize=10)
ax.set_ylim(-60, 85); ax.set_xlim(-180, 180); ax.set_ylabel("latitude"); ax.grid(color="#dcdbd6", lw=0.4)
ax = axes[1]
ax.pcolormesh(lon, lat, landbg, cmap=ListedColormap(["#e8e7e3"]), vmin=0, vmax=1, shading="auto")
fcm = ListedColormap(["#2a78d6", "#1baf7a", "#eb6834"])
pm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(fmap), cmap=fcm, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), shading="auto")
cb = fig.colorbar(pm, ax=ax, orientation="vertical", fraction=0.025, pad=0.01, ticks=[0, 1, 2])
cb.set_ticklabels(["runoff", "soil water\n(W, root zone, top layer)", "water input\n(P, P-AET)"]); cb.set_label("family of the best predictor")
ax.set_title("Which kind of model variable explains the cell best", loc="left", fontsize=10)
ax.set_ylim(-60, 85); ax.set_xlim(-180, 180); ax.set_ylabel("latitude"); ax.set_xlabel("longitude"); ax.grid(color="#dcdbd6", lw=0.4)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_best_predictor_r_map.png"), dpi=150); plt.close(fig)

panels = [("band", "all"), ("band", "tropical"), ("band", "temperate"), ("band", "boreal"),
          ("koppen", "A"), ("koppen", "C"), ("koppen", "D"), ("koppen", "E")]
fig, axes = plt.subplots(2, 4, figsize=(17, 8.8), sharex=True)
zsc = lambda v: (v - np.nanmean(v)) / np.nanstd(v, ddof=1)
group_best_storage = {}
for ax, (gt, gname) in zip(axes.ravel(), panels):
    d = ts[(ts.group_type == gt) & (ts.group == gname)]
    sub = summ[(summ.group_type == gt) & (summ.group == gname) & (summ.family == "storage") & summ.predictor.isin(CAND)]
    bs = sub.sort_values("r_wmedian", ascending=False).predictor.iloc[0]
    group_best_storage[(gt, gname)] = bs
    series = [("wetland_area_km2", "wetland area (WAD2M)", "#2a78d6", "wetland"),
              ("runoff_1mo", "model runoff (annual total)", "#eda100", "runoff"),
              (bs, "model soil water (group's best storage variable)", "#1baf7a", bs),
              ("precip_1mo", "precipitation (amprcp)", "#eb6834", "precip")]
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    for key, lab, col, txt in series:
        q = d[d.predictor == key].sort_values("year")
        v = zsc(q.anom_pct.values)
        ax.plot(q.year, v, color=col, lw=2, marker="o", ms=4.5, label=lab)
        if np.isfinite(v[-1]):
            ax.text(Y1 + 0.15, v[-1], txt, color="#52514e", va="center", fontsize=7.5)
    S_ = lambda n: st[(st.group_type == gt) & (st.group == gname) & (st.predictor == n)].iloc[0]
    sr, sb, sp = S_("runoff_1mo"), S_(bs), S_("precip_1mo")
    label = gname if gt == "band" else f"Koppen {gname}"
    ax.set_title(f"{label}: {sr.mean_wetland_area_Mkm2:.2f} Mkm2, sd of annual area {sr.sd_wetland_area_pct:.1f} %\n"
                 f"runoff r {sr.r:+.2f} (p {sr.p:.2f}), detr. {sr.r_detrended:+.2f}\n"
                 f"{bs} r {sb.r:+.2f} (p {sb.p:.2f}), detr. {sb.r_detrended:+.2f}\n"
                 f"precip r {sp.r:+.2f} (p {sp.p:.2f}), detr. {sp.r_detrended:+.2f}", loc="left", fontsize=8.2)
    ax.set_xlim(Y0 - 0.3, Y1 + 2.4); ax.grid(color="#e4e3df", lw=0.5)
for ax in axes[:, 0]:
    ax.set_ylabel("standardized annual anomaly (z)")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="lower center", ncol=4, fontsize=8.5)
fig.suptitle("Annual wetland area (cell area x WAD2M fraction, fill = 0) vs the model's own runoff and soil water and vs precipitation, "
             "2001-2010, wetland-area-weighted group means; each series standardized (r and p on the % anomalies)",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(os.path.join(OUT, "fig_band_timeseries.png"), dpi=150); plt.close(fig)
log("figures written")

checks["sample"] = dict(monthly_sample="Jul 2001 - Dec 2010 (114 months)", train="Jul 2001 - Dec 2006", test="Jan 2007 - Dec 2010",
                        f_clim="WAD2M 2000-2020 monthly climatology, fill = 0 (ibis/io/wad2m.read_wad2m_climatology)",
                        thresholds=dict(MINFW=MINFW, MIN_CLIM_FW=MIN_CLIM_FW, MIN_CLIM_FLUX_PER_MONTH=MIN_CLIM_FLUX_PER_MONTH,
                                        MIN_PAIRS=MIN_PAIRS, MIN_YEARS_CLIM=MIN_YEARS_CLIM))
checks["group_best_predictor"] = {f"{a}:{b}": v for (a, b), v in group_best.items()}
checks["group_best_storage_predictor"] = {f"{a}:{b}": v for (a, b), v in group_best_storage.items()}
checks["scheme_choice"] = scheme_choice
with open(os.path.join(OUT, "checks.json"), "w") as fh:
    json.dump(checks, fh, indent=1, default=str)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
cols = ["group", "predictor", "n_cells_fit", "r_wmedian", "r_q25", "r_q75", "share_area_r_gt_0p5", "slope_pct_per_pct_wmedian",
        "r_std_wmedian", "slope_pct_per_sd_wmedian", "share_area_best"]
print("\n=== summary ===")
print(summ[cols].to_string(index=False))
print("\n=== annual stats (selected) ===")
print(st[st.predictor.isin(["runoff_1mo", "runoff_3mo", "precip_1mo", "pme_1mo", "W_lag0", "Wrz_lag0", "s1_lag0", "W_lag0.5"])]
      [["group", "predictor", "n_years", "r", "p", "slope_pct_per_pct", "slope_pct_per_sd", "r_detrended", "r_fixedseason", "r_fixedseason_matched"]].to_string(index=False))
print("\n=== scheme ===")
print(fitdf.to_string(index=False))
print("\n=== regions ===")
print(regdf.drop(columns=["expectation"]).T.to_string())
print("\n=== seasonal lag (aggregates) ===")
print(lagdf[lagdf.scope != "per-cell"].to_string(index=False))
print("\n=== checks ===")
print(json.dumps({k: v for k, v in checks.items() if k != "model_monthly_variables"}, indent=1, default=str))
log("done")
