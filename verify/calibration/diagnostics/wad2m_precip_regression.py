#!/usr/bin/env python
"""
wad2m_precip_regression.py -- diagnostic only (no model code touched).

How much of the interannual variability of the satellite-observed wetland
(inundated) fraction (WAD2M v2.0, 0.5 deg, monthly) is explained by
precipitation, per climate band?  Precipitation is the model's own forcing,
GSWP3 (daily on this box), aggregated to monthly by streaming one month at a
time; it is identical to the hindcast's `amprcp` (checked, max |diff| < 1e-3 mm)
but also covers 2000, which the hindcast monthly output does not.

Overlap window: 2000-2010 (11 years).  Analysis cells: model land cells where
the WAD2M 2000-2010 mean fraction (frozen/fill months counted as 0) >= 0.02.

Outputs (under OUT):
  per_cell_results.csv          per-cell r / slope per window (pr and tas)
  summary_band_window.csv       wetland-area-weighted stats per band x window
  band_timeseries.csv           annual wetland area and P / T anomalies per band
  band_timeseries_stats.csv     r, slope, p of the band annual series
  seasonal_lag.csv              seasonal-cycle lag of Fw behind P per band
  fig_best_window_r_map.png     map of r at the best window (+ best window)
  fig_band_timeseries.png       band annual wetland area vs annual P anomaly
  precip_source_check.txt       GSWP3-monthly vs hindcast amprcp agreement
  monthly_gswp3_{pr,tas}_*.npy  cache of the streamed monthly forcing

Run:  /shared/nas2/Hangkai/ibis-env/ibis-python wad2m_precip_regression.py
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from scipy import stats

OUT = "/shared/nas2/Hangkai/ibis/verify_outputs/wad2m_regression"
WAD = "/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc"
GSWP3 = "/shared/nas2/Min/GSWP3"
HIND = "/shared/nas2/Hangkai/ibis/verify_outputs/final_s7_crop/hindcast/out"
KOPPEN = "/shared/nas/Public/Climate_Biome/Koppen_Beck2023/1991_2020/koppen_geiger_0p5.tif"

Y0, Y1 = 2000, 2010
NYR = Y1 - Y0 + 1
NMON = NYR * 12
PRE = 7                      # lead months before Jan 2000 (longest window = 7 months)
# window name -> number of months summed (current month + previous N)
WINDOWS = {"w0": 1, "w1": 2, "w2": 3, "w3": 4, "w6": 7}
MINFW = 0.02                 # analysis-cell threshold on the 2000-2010 mean fraction
MIN_CLIM_FW = 0.01           # months with a smaller Fw climatology are excluded from relative anomalies
MIN_CLIM_P_PER_MONTH = 2.0   # mm per window-month: floor for relative P anomalies
MIN_PAIRS = 24               # minimum valid months for a per-cell regression
MIN_YEARS_CLIM = 6           # calendar month needs >= 6 of 11 valid years for a climatology

os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 1. Monthly forcing from GSWP3 daily files (streamed one month at a time)
# ----------------------------------------------------------------------------
def month_list():
    """(year, month) tuples from Jun 1999 (PRE months before Jan 2000) to Dec 2010."""
    out = []
    y, m = Y0, 1
    for _ in range(PRE):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    while (y, m) <= (Y1, 12):
        out.append((y, m))
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


MONTHS = month_list()
assert MONTHS[PRE] == (Y0, 1) and len(MONTHS) == NMON + PRE


def stream_monthly(var, reducer):
    """Monthly aggregate of a GSWP3 daily variable, returned south-to-north float32.

    reducer: 'sum_mm' (kg m-2 s-1 -> mm/month) or 'mean'.
    """
    cache = os.path.join(OUT, f"monthly_gswp3_{var}_{MONTHS[0][0]}-{MONTHS[0][1]:02d}_{Y1}-12.npy")
    if os.path.exists(cache):
        log(f"{var}: loading cache {cache}")
        return np.load(cache)
    out = np.full((len(MONTHS), 360, 720), np.nan, dtype=np.float32)
    files = {}
    for (y, m) in MONTHS:
        dec = f"{(y-1)//10*10+1}_{(y-1)//10*10+10}"      # 1991_2000, 2001_2010
        files.setdefault(dec, []).append((y, m))
    for dec, yms in files.items():
        fn = os.path.join(GSWP3, f"{var}_gswp3_{dec}.nc4")
        log(f"{var}: streaming {fn} ({len(yms)} months)")
        ds = nc.Dataset(fn)
        tv = ds.variables["time"]
        dates = nc.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))
        ym = np.array([d.year * 100 + d.month for d in dates])
        lat = ds.variables["lat"][:]
        flip = lat[0] > lat[-1]
        v = ds.variables[var]
        for (y, m) in yms:
            idx = np.where(ym == y * 100 + m)[0]
            assert len(idx) >= 28, (y, m, len(idx))
            block = np.asarray(v[idx[0]:idx[-1] + 1, :, :], dtype=np.float64)
            if reducer == "sum_mm":
                agg = block.sum(0) * 86400.0
            else:
                agg = block.mean(0)
            if flip:
                agg = agg[::-1, :]
            out[MONTHS.index((y, m))] = agg
        ds.close()
    assert np.isfinite(out).all()
    np.save(cache, out)
    log(f"{var}: cached to {cache}")
    return out


P = stream_monthly("pr", "sum_mm")      # (PRE+NMON, 360, 720) mm/month, south-to-north
TAS = stream_monthly("tas", "mean")     # K
log(f"forcing loaded: P {P.shape}, TAS {TAS.shape}")

# cross-check against the hindcast's own monthly precipitation (north-to-south)
with open(os.path.join(OUT, "precip_source_check.txt"), "w") as fh:
    fh.write("GSWP3 daily pr summed to monthly (mm) vs hindcast amprcp (final_s7_crop, mm)\n")
    land_mask = None
    for (y, m) in [(2001, 1), (2003, 7), (2006, 4), (2010, 12)]:
        fn = os.path.join(HIND, f"ibis_monthly_{y}_{m:02d}.nc")
        h = nc.Dataset(fn)
        hp = np.ma.filled(h.variables["amprcp"][:].astype(np.float64), np.nan)[::-1, :]
        h.close()
        if land_mask is None:
            land_mask = np.isfinite(hp)
        g = P[MONTHS.index((y, m))].astype(np.float64)
        d = (hp - g)[land_mask]
        fh.write(f"{y}-{m:02d}: n_land={land_mask.sum()} mean_diff={d.mean():.2e} mm "
                 f"max|diff|={np.abs(d).max():.2e} mm  r={np.corrcoef(hp[land_mask], g[land_mask])[0,1]:.6f}\n")
    fh.write("Conclusion: the hindcast amprcp is the GSWP3 monthly sum; GSWP3 used because it also covers 2000.\n")
log("precip source check written")

# ----------------------------------------------------------------------------
# 2. WAD2M
# ----------------------------------------------------------------------------
ds = nc.Dataset(WAD)
lat = np.asarray(ds.variables["lat"][:])
lon = np.asarray(ds.variables["lon"][:])
assert lat[0] < lat[-1] and abs(lat[0] + 89.75) < 1e-6, "WAD2M expected south-to-north"
Fw = np.ma.filled(ds.variables["Fw"][0:NMON, :, :].astype(np.float32), np.nan)   # (132,360,720)
ds.close()
valid = np.isfinite(Fw)
nvalid = valid.sum(0)
Fw0 = np.where(valid, Fw, 0.0).astype(np.float32)           # frozen / fill -> 0
meanFw_fill0 = Fw0.mean(0)
meanFw_valid = np.where(nvalid > 0, Fw0.sum(0) / np.maximum(nvalid, 1), np.nan)

# orientation checks (printed into the log and the report)
checks = {}
for name, la, lo in [("HudsonBayLowlands", 55, -85), ("WestSiberianLowland", 60, 75),
                     ("CongoCuvette", -1, 18), ("Pantanal", -17, -57), ("Sahara", 23, 10)]:
    i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon - lo)))
    checks[name] = dict(lat=float(lat[i]), lon=float(lon[j]),
                        meanFw_valid=None if not np.isfinite(meanFw_valid[i, j]) else round(float(meanFw_valid[i, j]), 3),
                        meanFw_fill0=round(float(meanFw_fill0[i, j]), 3), n_valid_months=int(nvalid[i, j]),
                        P_mm_yr=round(float(P[PRE:, i, j].sum() / NYR), 0),
                        T_C=round(float(TAS[PRE:, i, j].mean() - 273.15), 1))
log("orientation checks: " + json.dumps(checks))

# cell area (km2), full cell
R = 6371.0088
edges = np.deg2rad(np.arange(-90, 90.5, 0.5))
area_row = (R ** 2) * np.deg2rad(0.5) * (np.sin(edges[1:]) - np.sin(edges[:-1]))
AREA = np.repeat(area_row[:, None], 720, axis=1)

# Koppen (north-to-south uint8 -> flip)
import rasterio
with rasterio.open(KOPPEN) as src:
    kop = src.read(1)[::-1, :]
kgroup = np.full(kop.shape, "", dtype=object)
for g, (a, b) in {"A": (1, 3), "B": (4, 7), "C": (8, 16), "D": (17, 28), "E": (29, 30)}.items():
    kgroup[(kop >= a) & (kop <= b)] = g

# analysis cells
cells = land_mask & (nvalid >= MIN_PAIRS) & (meanFw_fill0 >= MINFW)
n_outside_land = int(((nvalid >= MIN_PAIRS) & (meanFw_fill0 >= MINFW) & ~land_mask).sum())
ii, jj = np.where(cells)
N = len(ii)
log(f"analysis cells: {N} (WAD2M cells >= {MINFW} outside the model land mask: {n_outside_land})")

clat = lat[ii]; clon = lon[jj]
carea = AREA[ii, jj]
cmean = meanFw_fill0[ii, jj]
cmean_valid = meanFw_valid[ii, jj]
cnvalid = nvalid[ii, jj]
wet_area = carea * cmean                      # km2 of mean wetland per cell (weights)
band = np.where(np.abs(clat) < 30, "tropical", np.where(np.abs(clat) < 60, "temperate",
                np.where(clat >= 60, "boreal", "antarctic")))
kg = np.array([kgroup[i, j] for i, j in zip(ii, jj)])
hemi = np.where(clat >= 0, "N", "S")

Fw_c = Fw[:, ii, jj].astype(np.float64)            # (132, N) with NaN for fill
Fw0_c = Fw0[:, ii, jj].astype(np.float64)
P_c = P[:, ii, jj].astype(np.float64)               # (139, N)
T_c = TAS[:, ii, jj].astype(np.float64)
del Fw, Fw0, valid
log("cell arrays extracted")

# ----------------------------------------------------------------------------
# 3. Climatologies, anomalies, windows
# ----------------------------------------------------------------------------
Fw_y = Fw_c.reshape(NYR, 12, N)
nyr_valid = np.isfinite(Fw_y).sum(0)                                   # (12,N)
with np.errstate(invalid="ignore"):
    clim_Fw = np.nanmean(Fw_y, axis=0)                                  # (12,N)
clim_Fw[nyr_valid < MIN_YEARS_CLIM] = np.nan
anom_Fw = (Fw_y - clim_Fw[None]).reshape(NMON, N)
rel_den = np.where(clim_Fw >= MIN_CLIM_FW, clim_Fw, np.nan)
rel_Fw = (Fw_y - clim_Fw[None]).reshape(NMON, N) / np.tile(rel_den, (NYR, 1))

clim_Fw0 = Fw0_c.reshape(NYR, 12, N).mean(0)                           # fill->0 climatology (seasonal cycle)


def window_sum(X, n):
    """Sum of the n months ending at each month of 2000-2010 (X has PRE lead months)."""
    C = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)], axis=0)
    t = np.arange(PRE, PRE + NMON)
    return C[t + 1] - C[t + 1 - n]


def nan_corr_slope(x, y, min_n=MIN_PAIRS):
    """Pearson r and OLS slope of y on x, column-wise, ignoring NaN pairs."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(0)
    xz = np.where(ok, x, 0.0); yz = np.where(ok, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = xz.sum(0) / n; my = yz.sum(0) / n
        dx = np.where(ok, x - mx, 0.0); dy = np.where(ok, y - my, 0.0)
        sxx = (dx * dx).sum(0); syy = (dy * dy).sum(0); sxy = (dx * dy).sum(0)
        r = sxy / np.sqrt(sxx * syy)
        slope = sxy / sxx
    bad = (n < min_n) | ~np.isfinite(r)
    r[bad] = np.nan; slope[bad] = np.nan
    return r, slope, n


res = {}
for wname, nmon in WINDOWS.items():
    S = window_sum(P_c, nmon)                                            # (132,N) mm
    clim_S = S.reshape(NYR, 12, N).mean(0)
    den = np.where(clim_S >= MIN_CLIM_P_PER_MONTH * nmon, clim_S, np.nan)
    rel_P = (S.reshape(NYR, 12, N) - clim_S[None]).reshape(NMON, N) / np.tile(den, (NYR, 1))
    abs_P = (S.reshape(NYR, 12, N) - clim_S[None]).reshape(NMON, N)
    r_rel, s_rel, n_rel = nan_corr_slope(rel_P, rel_Fw)
    r_abs, s_abs, _ = nan_corr_slope(abs_P, anom_Fw)
    # temperature: mean over the window, anomaly in K
    TS = window_sum(T_c, nmon) / nmon
    clim_T = TS.reshape(NYR, 12, N).mean(0)
    anom_T = (TS.reshape(NYR, 12, N) - clim_T[None]).reshape(NMON, N)
    r_tas, s_tas, _ = nan_corr_slope(anom_T, rel_Fw)
    res[wname] = dict(r=r_rel, slope=s_rel, n=n_rel, r_abs=r_abs, slope_abs=s_abs, r_tas=r_tas, slope_tas=s_tas)
    log(f"window {wname} ({nmon} mo): median r_rel={np.nanmedian(r_rel):.3f} r_abs={np.nanmedian(r_abs):.3f} r_tas={np.nanmedian(r_tas):.3f}")

# annual-scale per-cell regression (n = 11 years): relative annual anomalies (fill -> 0 annual mean)
annFw0_c = Fw0_c.reshape(NYR, 12, N).mean(1)
annP_c = window_sum(P_c, 1).reshape(NYR, 12, N).sum(1)
annT_c = window_sum(T_c, 1).reshape(NYR, 12, N).mean(1)
relA_Fw = annFw0_c / annFw0_c.mean(0) - 1
relA_P = annP_c / annP_c.mean(0) - 1
r_ann, s_ann, _ = nan_corr_slope(relA_P, relA_Fw, min_n=NYR)
r_ann_T, s_ann_T, _ = nan_corr_slope(annT_c - annT_c.mean(0), relA_Fw, min_n=NYR)
res["annual"] = dict(r=r_ann, slope=s_ann, n=np.full(N, NYR), r_abs=np.full(N, np.nan), slope_abs=np.full(N, np.nan),
                     r_tas=r_ann_T, slope_tas=s_ann_T)
log(f"annual scale (n={NYR}): median r={np.nanmedian(r_ann):.3f} r_tas={np.nanmedian(r_ann_T):.3f}")

wnames = list(WINDOWS)
R_all = np.stack([res[w]["r"] for w in wnames])                          # (5,N)
R_all_f = np.where(np.isfinite(R_all), R_all, -np.inf)
best_idx = np.argmax(R_all_f, axis=0)
has_any = np.isfinite(R_all).any(0)
best_w = np.where(has_any, np.array(wnames)[best_idx], "")
best_r = np.where(has_any, R_all_f[best_idx, np.arange(N)], np.nan)
best_slope = np.where(has_any, np.stack([res[w]["slope"] for w in wnames])[best_idx, np.arange(N)], np.nan)
RT_all = np.stack([res[w]["r_tas"] for w in wnames])
RT_f = np.where(np.isfinite(RT_all), RT_all, -np.inf)
best_t_idx = np.argmax(np.abs(np.where(np.isfinite(RT_all), RT_all, 0)), axis=0)
best_tas_w = np.where(np.isfinite(RT_all).any(0), np.array(wnames)[best_t_idx], "")
best_tas_r = np.where(np.isfinite(RT_all).any(0), RT_all[best_t_idx, np.arange(N)], np.nan)

# ----------------------------------------------------------------------------
# 4. Seasonal lag per cell (circular cross-correlation of 12-month climatologies)
# ----------------------------------------------------------------------------
clim_P1 = window_sum(P_c, 1).reshape(NYR, 12, N).mean(0)                 # (12,N)


def xcorr_lags(a, b):
    """r(lag) for lag 0..11 where b is shifted: corr(a[m], b[m-lag]); a,b (12,N)."""
    out = np.full((12,) + a.shape[1:], np.nan)
    a0 = a - a.mean(0); sa = np.sqrt((a0 ** 2).sum(0))
    for L in range(12):
        bs = np.roll(b, L, axis=0)
        b0 = bs - bs.mean(0); sb = np.sqrt((b0 ** 2).sum(0))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[L] = (a0 * b0).sum(0) / (sa * sb)
    return out


xc = xcorr_lags(clim_Fw0, clim_P1)                                         # (12,N)
amp_ok = ((clim_Fw0.max(0) - clim_Fw0.min(0)) >= 0.01) & ((clim_P1.max(0) - clim_P1.min(0)) >= 10.0)
xcf = np.where(np.isfinite(xc), xc, -np.inf)
lag_idx = np.argmax(xcf, axis=0)
seas_lag = np.where(lag_idx > 6, lag_idx - 12, lag_idx).astype(float)      # -5..6, + = Fw lags P
seas_r = xcf[lag_idx, np.arange(N)]
seas_r0 = xc[0]
seas_lag[~amp_ok] = np.nan; seas_r[~amp_ok] = np.nan

# ----------------------------------------------------------------------------
# 5. Per-cell CSV
# ----------------------------------------------------------------------------
pc = pd.DataFrame(dict(lat=clat, lon=clon, band=band, koppen_group=kg, koppen_code=[int(kop[i, j]) for i, j in zip(ii, jj)],
                       cell_area_km2=carea, mean_Fw_fill0=cmean, mean_Fw_validmonths=cmean_valid,
                       n_valid_months=cnvalid, wetland_area_km2=wet_area,
                       best_window=best_w, best_r=best_r, best_slope=best_slope,
                       seasonal_lag_months=seas_lag, seasonal_r_at_lag=seas_r, seasonal_r_lag0=seas_r0,
                       best_tas_window=best_tas_w, best_tas_r=best_tas_r))
for w in wnames:
    pc[f"r_{w}"] = res[w]["r"]; pc[f"slope_{w}"] = res[w]["slope"]; pc[f"n_{w}"] = res[w]["n"]
    pc[f"rabs_{w}"] = res[w]["r_abs"]
    pc[f"rtas_{w}"] = res[w]["r_tas"]; pc[f"slopetas_{w}"] = res[w]["slope_tas"]
pc["r_annual"] = res["annual"]["r"]; pc["slope_annual"] = res["annual"]["slope"]
pc["rtas_annual"] = res["annual"]["r_tas"]; pc["slopetas_annual"] = res["annual"]["slope_tas"]
pc.to_csv(os.path.join(OUT, "per_cell_results.csv"), index=False, float_format="%.5g")
log("per_cell_results.csv written")

# ----------------------------------------------------------------------------
# 6. Summary per band x window (wetland-area-weighted)
# ----------------------------------------------------------------------------
def wquantiles(v, w, qs=(0.25, 0.5, 0.75)):
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return [np.nan] * len(qs)
    v = v[ok]; w = w[ok]
    o = np.argsort(v); v = v[o]; w = w[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return list(np.interp(qs, cw, v))


groups = [("band", "all", np.ones(N, bool))]
for b in ["tropical", "temperate", "boreal"]:
    groups.append(("band", b, band == b))
for g in ["A", "B", "C", "D", "E"]:
    groups.append(("koppen", g, kg == g))

rows = []
for gtype, gname, sel in groups:
    w = wet_area[sel]
    for var in ["pr", "tas"]:
        for wname in wnames + ["annual", "best"]:
            if var == "pr":
                r = (best_r if wname == "best" else res[wname]["r"])[sel]
                s = (best_slope if wname == "best" else res[wname]["slope"])[sel]
                rabs = (np.full(N, np.nan) if wname == "best" else res[wname]["r_abs"])[sel]
                share_best = np.nan if wname in ("best", "annual") else float(((best_w[sel] == wname) & np.isfinite(best_r[sel])).sum() / max(np.isfinite(best_r[sel]).sum(), 1))
            else:
                r = (best_tas_r if wname == "best" else res[wname]["r_tas"])[sel]
                s = (np.full(N, np.nan) if wname == "best" else res[wname]["slope_tas"])[sel]
                rabs = np.full(sel.sum(), np.nan)
                share_best = np.nan
            ok = np.isfinite(r)
            rq = wquantiles(r, w); sq = wquantiles(s, w); raq = wquantiles(rabs, w)
            rows.append(dict(group_type=gtype, group=gname, variable=var, window=wname,
                             window_months=(WINDOWS.get(wname, np.nan)), n_cells=int(sel.sum()), n_cells_fit=int(ok.sum()),
                             wetland_area_Mkm2=w.sum() / 1e6,
                             r_wmedian=rq[1], r_q25=rq[0], r_q75=rq[2],
                             r2_wmedian=rq[1] ** 2 if np.isfinite(rq[1]) else np.nan,
                             slope_wmedian=sq[1], slope_q25=sq[0], slope_q75=sq[2],
                             share_cells_r_gt_0p5=float((r[ok] > 0.5).mean()) if ok.any() else np.nan,
                             share_area_r_gt_0p5=float(w[ok][r[ok] > 0.5].sum() / w[ok].sum()) if ok.any() else np.nan,
                             share_cells_r_gt_0p7=float((r[ok] > 0.7).mean()) if ok.any() else np.nan,
                             share_cells_best_window=share_best,
                             r_abs_wmedian=raq[1], r_abs_q25=raq[0], r_abs_q75=raq[2]))
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "summary_band_window.csv"), index=False, float_format="%.4g")
log("summary_band_window.csv written")

# ----------------------------------------------------------------------------
# 7. Band annual time series: wetland area vs annual P (and T) anomaly
# ----------------------------------------------------------------------------
years = np.arange(Y0, Y1 + 1)
annFw0 = Fw0_c.reshape(NYR, 12, N).mean(1)                                   # (11,N) fill->0
fixed_season = (nyr_valid == NYR)                                            # calendar months valid in every year
with np.errstate(invalid="ignore"):
    annFw_fixed = np.where(fixed_season.sum(0) > 0,
                           (Fw0_c.reshape(NYR, 12, N) * fixed_season[None]).sum(1) / np.maximum(fixed_season.sum(0), 1), np.nan)
annP = window_sum(P_c, 1).reshape(NYR, 12, N).sum(1)                         # mm/yr
annT = window_sum(T_c, 1).reshape(NYR, 12, N).mean(1)                        # K

ts_rows, st_rows = [], []
for gtype, gname, sel in groups:
    if sel.sum() == 0:
        continue
    a = carea[sel]; w = wet_area[sel]
    WA = (annFw0[:, sel] * a).sum(1)                                           # km2
    okf = np.isfinite(annFw_fixed[:, sel]).all(0)
    WAf = (annFw_fixed[:, sel][:, okf] * a[okf]).sum(1) if okf.any() else np.full(NYR, np.nan)
    Pb = (annP[:, sel] * w).sum(1) / w.sum()                                   # wetland-area-weighted mm/yr
    Tb = (annT[:, sel] * w).sum(1) / w.sum()
    wa_pct = 100 * (WA / WA.mean() - 1); p_pct = 100 * (Pb / Pb.mean() - 1); t_anom = Tb - Tb.mean()
    waf_pct = 100 * (WAf / np.nanmean(WAf) - 1)
    detrend = lambda v: v - np.polyval(np.polyfit(years, v, 1), years)
    wa_d, p_d, t_d = detrend(wa_pct), detrend(p_pct), detrend(t_anom)
    rPd, pPd = stats.pearsonr(p_d, wa_d); slPd = np.polyfit(p_d, wa_d, 1)[0]
    rTd, pTd = stats.pearsonr(t_d, wa_d)
    trend_wa = np.polyfit(years, wa_pct, 1)[0]; trend_p = np.polyfit(years, p_pct, 1)[0]
    for k, y in enumerate(years):
        ts_rows.append(dict(group_type=gtype, group=gname, year=int(y), wetland_area_km2=WA[k],
                            wetland_area_fixedseason_km2=WAf[k], P_mm=Pb[k], T_K=Tb[k],
                            wetland_area_anom_pct=wa_pct[k], wetland_area_fixedseason_anom_pct=waf_pct[k],
                            P_anom_pct=p_pct[k], T_anom_K=t_anom[k]))
    rP, pP = stats.pearsonr(p_pct, wa_pct); slP = np.polyfit(p_pct, wa_pct, 1)[0]
    rT, pT = stats.pearsonr(t_anom, wa_pct); slT = np.polyfit(t_anom, wa_pct, 1)[0]
    # previous-year and two-year-mean precipitation
    rP1 = stats.pearsonr(p_pct[:-1], wa_pct[1:])[0]
    p2 = 0.5 * (p_pct[1:] + p_pct[:-1]); rP2, pP2 = stats.pearsonr(p2, wa_pct[1:]); slP2 = np.polyfit(p2, wa_pct[1:], 1)[0]
    if np.isfinite(waf_pct).all():
        rPf, pPf = stats.pearsonr(p_pct, waf_pct); slPf = np.polyfit(p_pct, waf_pct, 1)[0]
    else:
        rPf = pPf = slPf = np.nan
    # multiple regression on P and T
    X = np.column_stack([np.ones(NYR), p_pct, t_anom]); beta, *_ = np.linalg.lstsq(X, wa_pct, rcond=None)
    r2_PT = 1 - ((wa_pct - X @ beta) ** 2).sum() / ((wa_pct - wa_pct.mean()) ** 2).sum()
    st_rows.append(dict(group_type=gtype, group=gname, n_cells=int(sel.sum()), mean_wetland_area_Mkm2=WA.mean() / 1e6,
                        sd_wetland_area_pct=wa_pct.std(ddof=1), sd_P_pct=p_pct.std(ddof=1), sd_T_K=t_anom.std(ddof=1),
                        r_P_sameyear=rP, p_P_sameyear=pP, slope_pct_per_pct=slP, r2_P_sameyear=rP ** 2,
                        r_P_prevyear=rP1, r_P_2yrmean=rP2, slope_2yrmean_pct_per_pct=slP2,
                        r_P_fixedseason=rPf, slope_fixedseason_pct_per_pct=slPf,
                        r_T=rT, p_T=pT, slope_pct_per_K=slT, r2_P_plus_T=r2_PT,
                        trend_wetland_pct_per_yr=trend_wa, trend_P_pct_per_yr=trend_p,
                        r_P_detrended=rPd, p_P_detrended=pPd, slope_detrended_pct_per_pct=slPd, r_T_detrended=rTd, p_T_detrended=pTd))
ts = pd.DataFrame(ts_rows); ts.to_csv(os.path.join(OUT, "band_timeseries.csv"), index=False, float_format="%.6g")
st = pd.DataFrame(st_rows); st.to_csv(os.path.join(OUT, "band_timeseries_stats.csv"), index=False, float_format="%.4g")
log("band time series written")

# ----------------------------------------------------------------------------
# 8. Seasonal lag per band
# ----------------------------------------------------------------------------
lag_rows = []
for gtype, gname, sel in groups:
    if sel.sum() == 0:
        continue
    w = wet_area[sel]; lg = seas_lag[sel]; rr = seas_r[sel]; r0 = seas_r0[sel]
    ok = np.isfinite(lg)
    lq = wquantiles(lg, w)
    hist = {f"share_lag_{L:+d}": float(w[ok][lg[ok] == L].sum() / w[ok].sum()) if ok.any() else np.nan for L in range(-5, 7)}
    row = dict(group_type=gtype, group=gname, scope="per-cell", n_cells=int(ok.sum()),
               lag_wmedian=lq[1], lag_q25=lq[0], lag_q75=lq[2],
               r_at_best_lag_wmedian=wquantiles(rr, w)[1], r_lag0_wmedian=wquantiles(r0, w)[1])
    row.update(hist); lag_rows.append(row)
    # band aggregate by hemisphere
    for hh in ["N", "S"]:
        s2 = sel & (hemi == hh)
        if s2.sum() < 20:
            continue
        Fb = (clim_Fw0[:, s2] * carea[s2]).sum(1)                              # km2 per month
        Pb = (clim_P1[:, s2] * wet_area[s2]).sum(1) / wet_area[s2].sum()
        xcb = xcorr_lags(Fb[:, None], Pb[:, None])[:, 0]
        L = int(np.argmax(xcb)); L = L - 12 if L > 6 else L
        lag_rows.append(dict(group_type=gtype, group=gname, scope=f"aggregate-{hh}", n_cells=int(s2.sum()),
                             lag_wmedian=L, r_at_best_lag_wmedian=float(xcb.max()), r_lag0_wmedian=float(xcb[0]),
                             Fw_area_peak_month=int(np.argmax(Fb)) + 1, P_peak_month=int(np.argmax(Pb)) + 1))
pd.DataFrame(lag_rows).to_csv(os.path.join(OUT, "seasonal_lag.csv"), index=False, float_format="%.4g")
log("seasonal_lag.csv written")

# ----------------------------------------------------------------------------
# 9. Figures
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#9a9a95", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "axes.titlecolor": "#0b0b0b", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
LON, LAT = np.meshgrid(lon, lat)
rmap = np.full((360, 720), np.nan); rmap[ii, jj] = best_r
wmap = np.full((360, 720), np.nan); wmap[ii, jj] = [wnames.index(w) if w else np.nan for w in best_w]
landbg = np.where(land_mask, 1.0, np.nan)

fig, axes = plt.subplots(2, 1, figsize=(11, 9.5))
ax = axes[0]
ax.pcolormesh(lon, lat, landbg, cmap=ListedColormap(["#e8e7e3"]), vmin=0, vmax=1, shading="auto")
div = LinearSegmentedColormap.from_list("div", [(0, "#8f1d1d"), (0.25, "#e34948"), (0.5, "#f0efec"), (0.75, "#2a78d6"), (1, "#0d366b")])
pm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(rmap), cmap=div, vmin=-1, vmax=1, shading="auto")
cb = fig.colorbar(pm, ax=ax, orientation="vertical", fraction=0.025, pad=0.01)
cb.set_label("Pearson r at the best window")
ax.set_title("WAD2M relative inundation anomaly vs relative GSWP3 precipitation anomaly, 2000-2010: r at the cell's best window", loc="left", fontsize=10)
ax.set_ylim(-60, 85); ax.set_xlim(-180, 180); ax.set_ylabel("latitude"); ax.grid(color="#dcdbd6", lw=0.4)
ax = axes[1]
ax.pcolormesh(lon, lat, landbg, cmap=ListedColormap(["#e8e7e3"]), vmin=0, vmax=1, shading="auto")
ocm = ListedColormap(["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
pm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(wmap), cmap=ocm, norm=BoundaryNorm(np.arange(-0.5, 5.5, 1), 5), shading="auto")
cb = fig.colorbar(pm, ax=ax, orientation="vertical", fraction=0.025, pad=0.01, ticks=range(5))
cb.set_ticklabels(["1 mo", "2 mo", "3 mo", "4 mo", "7 mo"]); cb.set_label("best accumulation window")
ax.set_title("Accumulation window (current month + previous N) that maximises r", loc="left", fontsize=10)
ax.set_ylim(-60, 85); ax.set_xlim(-180, 180); ax.set_ylabel("latitude"); ax.set_xlabel("longitude"); ax.grid(color="#dcdbd6", lw=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_best_window_r_map.png"), dpi=150)
plt.close(fig)

panels = [("band", "all"), ("band", "tropical"), ("band", "temperate"), ("band", "boreal"),
          ("koppen", "A"), ("koppen", "C"), ("koppen", "D"), ("koppen", "B")]
fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
for ax, (gt, gname) in zip(axes.ravel(), panels):
    d = ts[(ts.group_type == gt) & (ts.group == gname)]
    s_ = st[(st.group_type == gt) & (st.group == gname)].iloc[0]
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.plot(d.year, d.wetland_area_anom_pct, color="#2a78d6", lw=2, marker="o", ms=5, label="wetland area (WAD2M)")
    ax.plot(d.year, d.P_anom_pct, color="#eb6834", lw=2, marker="o", ms=5, label="annual precipitation (GSWP3)")
    ax.text(d.year.iloc[-1] + 0.15, d.wetland_area_anom_pct.iloc[-1], "wetland", color="#52514e", va="center", fontsize=8)
    ax.text(d.year.iloc[-1] + 0.15, d.P_anom_pct.iloc[-1], "precip", color="#52514e", va="center", fontsize=8)
    label = gname if gt == "band" else f"Koppen {gname}"
    ax.set_title(f"{label}: {s_.mean_wetland_area_Mkm2:.2f} Mkm2 wetland\n"
                 f"r = {s_.r_P_sameyear:+.2f} (p = {s_.p_P_sameyear:.2f}), slope = {s_.slope_pct_per_pct:+.2f} % per %\n"
                 f"detrended r = {s_.r_P_detrended:+.2f}, slope = {s_.slope_detrended_pct_per_pct:+.2f} % per %", loc="left", fontsize=8.5)
    ax.set_xlim(Y0 - 0.3, Y1 + 1.7); ax.grid(color="#e4e3df", lw=0.5)
for ax in axes[:, 0]:
    ax.set_ylabel("anomaly vs 2000-2010 mean (%)")
axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle("Annual wetland area (cell area x WAD2M fraction, fill = 0) vs wetland-area-weighted annual precipitation, 2000-2010", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_band_timeseries.png"), dpi=150)
plt.close(fig)
log("figures written")

with open(os.path.join(OUT, "orientation_checks.json"), "w") as fh:
    json.dump(dict(checks=checks, n_cells=N, n_cells_outside_model_land=n_outside_land,
                   windows=WINDOWS, thresholds=dict(MINFW=MINFW, MIN_CLIM_FW=MIN_CLIM_FW, MIN_CLIM_P_PER_MONTH=MIN_CLIM_P_PER_MONTH,
                                                    MIN_PAIRS=MIN_PAIRS, MIN_YEARS_CLIM=MIN_YEARS_CLIM)), fh, indent=1)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("\n=== summary (pr) ===")
print(summ[summ.variable == "pr"][["group", "window", "n_cells", "wetland_area_Mkm2", "r_wmedian", "r_q25", "r_q75", "slope_wmedian", "slope_q25", "slope_q75", "share_cells_r_gt_0p5", "share_cells_best_window", "r_abs_wmedian"]].to_string(index=False))
print("\n=== summary (tas) ===")
print(summ[summ.variable == "tas"][["group", "window", "r_wmedian", "r_q25", "r_q75", "slope_wmedian", "share_cells_r_gt_0p5"]].to_string(index=False))
print("\n=== band time series ===")
print(st.to_string(index=False))
print("\n=== seasonal lag ===")
print(pd.DataFrame(lag_rows)[["group", "scope", "n_cells", "lag_wmedian", "lag_q25", "lag_q75", "r_at_best_lag_wmedian", "r_lag0_wmedian"]].to_string(index=False))
log("done")
