"""Per-tower diagnostic figure from site_diag_year records (CALIBRATION.md 22-26): GPP with the variables that make it.

    python verify/calibration/diagnostics/site_diag_plot.py <out_dir> <label>=<record.npz> [<label>=<record.npz> ...]

All records must be for the same towers and year. One figure per tower (<out_dir>/<site>_<year>.png), rows against day
of year, 7-day running means for daily series:
  1 GPP: tower (mean of NT and DT; dots = days with NEE QC >= 0.8) and each run
  2 LAI: upper-canopy leaf area per ground area (lai x fu) of each run, the site's measured maximum (ESSD 2025 site
    attributes, where the tower has one), MODIS at the 0.5-degree cell (monthly)
  3 vcmax of the upper big leaf: nominal (PFT value x acclimation) and effective (x leaf-temperature response x soil
    stress), daylight means
  4 what limits photosynthesis: share of daylight hours in which the canopy rate is capacity-limited (Rubisco or sink),
    at canopy level (F77: the top leaf's limit; capacity scaling: limits scaled by their own profiles)
  5 canopy scaling: effective leaf area for capacity (f_cap x L) and for light (f_light x L), daylight means
  6 radiation: tower incoming shortwave and the model's top-of-canopy PAR (daily means)
  7 temperature and air dryness: tower air temperature, model leaf temperature (daylight), tower VPD
  8 water: model soil-water stress factor (1 = unstressed) and wet-leaf fraction; tower soil water content
  9 ET: tower LE_CORR and each run (mm/day)
 10 monthly ET partition of each run (transpiration, interception, soil evaporation) with the tower's monthly ET
 11 monthly NEE (tower NEE_VUT_REF; model -(NPP - Rh)) and model heterotrophic respiration
"""
import sys, os, glob, numpy as np, pandas as pd, xarray as xr, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path[:0] = ['python-port', 'verify/calibration']
from site_attributes import prescribe_from_attributes
FX = '/shared/nas2/Hangkai/ibis/inputs/sites/FLUXNET2015_FULLSET/daily_monthly_yearly'
MODIS_BASE = None
try:
    from fit_carbon import MODIS_BASE
except Exception:
    pass

out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
RUNS = [(x.split('=', 1)[0], np.load(x.split('=', 1)[1], allow_pickle=True)) for x in sys.argv[2:]]
R0 = RUNS[0][1]; sites = [str(s) for s in R0['sites']]; year = int(R0['year'])
for lab, R in RUNS[1:]:
    assert [str(s) for s in R['sites']] == sites and int(R['year']) == year, lab
COL = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:purple']
sm = lambda x, w=7: pd.Series(x).rolling(w, center=True, min_periods=1).mean().values


def modis_monthly(i, j):
    if MODIS_BASE is None: return np.full(12, np.nan)
    ds = xr.open_dataset(f"{MODIS_BASE}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc"); sd = xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc")
    pv = np.asarray(sd["PCT_NATVEG"].values) / 100.0; pp = np.asarray(sd["PCT_NAT_PFT"].values) / 100.0; nt = ds.sizes["time"]
    flip = float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]); ny = pv.shape[0]; ii = ny - 1 - i if flip else i
    mon = np.zeros(12)
    for m in range(12):
        sel = [t for t in range(nt) if t % 12 == m]; tot = 0.0
        for k in range(1, 17):
            v = f"LAI_{k}"
            if v in ds: tot += float(np.nan_to_num(ds[v].values[sel, ii, j].mean())) * pp[k, ii, j]
        mon[m] = tot * pv[ii, j]
    return mon


def field(R, name, n):
    return R['R'][:, list(R['fields']).index(name), n].astype(float)


def daily(x, how='mean', mask=None):
    x = np.asarray(x, float).reshape(-1, 24)
    if mask is not None:
        m = np.asarray(mask).reshape(-1, 24); x = np.where(m, x, np.nan)
    with np.errstate(all='ignore'):
        return np.nanmean(x, 1) if how == 'mean' else np.nansum(x, 1)


for n, s in enumerate(sites):
    igbp = str(R0['igbp']); leaf = 'uc' if igbp == 'ENF' else 'ub'
    f = glob.glob(f'{FX}/FLX_{s}_FLUXNET2015_FULLSET_DD_*.csv'); d = pd.read_csv(f[0]); t = pd.to_datetime(d['TIMESTAMP'].astype(str), format='%Y%m%d')
    d = d[t.dt.year == year].reset_index(drop=True); doy_t = (pd.to_datetime(d['TIMESTAMP'].astype(str), format='%Y%m%d') - pd.Timestamp(f'{year}-01-01')).dt.days.values + 1
    w = lambda c: d[c].where(d[c] > -9000).values if c in d else np.full(len(d), np.nan)
    gpp_t = np.nanmean(np.vstack([w('GPP_NT_VUT_REF'), w('GPP_DT_VUT_REF')]), 0); qc = w('NEE_VUT_REF_QC')
    le_src = 'LE_CORR' if np.isfinite(w('LE_CORR')).sum() > 30 else 'LE_F_MDS'                    # some towers have no corrected LE
    le_t = w(le_src) * 86400.0 / 2.45e6; sw_t = w('SW_IN_F'); ta_t = w('TA_F'); vpd_t = w('VPD_F'); swc_t = w('SWC_F_MDS_1'); nee_t = w('NEE_VUT_REF')
    mo_t = pd.to_datetime(d['TIMESTAMP'].astype(str), format='%Y%m%d').dt.month.values
    nee_mon_t = np.array([np.nansum(nee_t[mo_t == m]) if np.isfinite(nee_t[mo_t == m]).sum() > 20 else np.nan for m in range(1, 13)])
    le_mon_t = np.array([np.nansum(le_t[mo_t == m]) if np.isfinite(le_t[mo_t == m]).sum() > 20 else np.nan for m in range(1, 13)])
    at = prescribe_from_attributes(s); lai_site = at['lai_max'] if at is not None and np.isfinite(at['lai_max']) else np.nan
    mod = modis_monthly(int(R0['i'][n]), int(R0['j'][n])); mid = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    fig, ax = plt.subplots(11, 1, figsize=(12, 30), sharex=False); plt.rcParams.update({'font.size': 9})
    nd = len(R0['day_gpp']); days = np.arange(1, nd + 1)
    # 1 GPP
    a = ax[0]; a.plot(doy_t, sm(gpp_t), 'k', lw=1.6, label='tower (mean NT, DT)'); good = qc >= 0.8; a.plot(doy_t[good], gpp_t[good], 'k.', ms=2, alpha=0.4)
    for k, (lab, R) in enumerate(RUNS): a.plot(days, sm(np.asarray(R['day_gpp'])[:, n] * 1000.0), color=COL[k], lw=1.3, label=lab)
    a.set_ylabel('GPP gC m-2 d-1'); a.legend(fontsize=8, ncol=4, loc='upper left'); a.set_title(f'{s} ({igbp}) {year}: GPP and what makes it', fontsize=11)
    # 2 LAI
    a = ax[1]
    for k, (lab, R) in enumerate(RUNS):
        a.plot(days, daily(field(R, 'lai_u', n)) * float(R['fu'][n]), color=COL[k], lw=1.3, label=f'{lab}: lai_u x fu ({str(R["lai_src"][n])})')
    if np.isfinite(lai_site): a.axhline(lai_site, color='k', ls='--', lw=1, label=f'site measured max {lai_site:.1f}')
    a.plot(mid, mod, 'ks', ms=4, label='MODIS, 0.5-deg cell (monthly)'); a.set_ylabel('LAI m2 m-2'); a.legend(fontsize=7, loc='upper left')
    # 3 vcmax
    a = ax[2]
    for k, (lab, R) in enumerate(RUNS):
        day = field(R, 'par_u', n) > 10.0
        a.plot(days, daily(field(R, f'vnom_{leaf}', n) * 1e6, mask=day), color=COL[k], lw=1.0, ls='--', label=f'{lab}: nominal (x acclimation)')
        a.plot(days, sm(daily(field(R, f'vmax_{leaf}', n) * 1e6, mask=day)), color=COL[k], lw=1.4, label=f'{lab}: effective (x leaf T x stress)')
    a.set_ylabel('vcmax umol m-2 s-1'); a.legend(fontsize=7, ncol=2, loc='upper left')
    # 4 limitation at canopy level
    a = ax[3]
    for k, (lab, R) in enumerate(RUNS):
        day = field(R, 'par_u', n) > 10.0; fl = field(R, 'fl_u', n); fc = field(R, 'fc_u', n)
        je, jc, js = field(R, f'je_{leaf}', n) * fl, field(R, f'jc_{leaf}', n) * fc, field(R, f'js_{leaf}', n) * fc
        cap = (np.minimum(jc, js) < je).astype(float)
        a.plot(days, sm(daily(cap, mask=day)), color=COL[k], lw=1.3, label=f'{lab}')
    a.set_ylim(-0.02, 1.02); a.set_ylabel('share of daylight hours\ncapacity-limited'); a.legend(fontsize=8, loc='lower left')
    # 5 canopy scaling
    a = ax[4]
    for k, (lab, R) in enumerate(RUNS):
        day = field(R, 'par_u', n) > 10.0; Lu = field(R, 'lai_u', n)
        a.plot(days, sm(daily(field(R, 'fc_u', n) * Lu, mask=day)), color=COL[k], lw=1.4, label=f'{lab}: capacity f_cap x L')
        a.plot(days, sm(daily(field(R, 'fl_u', n) * Lu, mask=day)), color=COL[k], lw=0.9, ls=':', label=f'{lab}: light f_light x L')
    a.set_ylabel('effective leaf area\nm2 leaf m-2'); a.legend(fontsize=7, ncol=2, loc='upper left')
    # 6 radiation
    a = ax[5]; a.plot(doy_t, sm(sw_t), 'k', lw=1.3, label='tower SW_IN (daily mean)')
    for k, (lab, R) in enumerate(RUNS[:1]):
        a.plot(days, sm(daily(field(R, 'par_u', n))), color='tab:gray', lw=1.2, label="model PAR absorbed by the top leaf (per leaf area; 0 without leaves) -- not incident")
    a.set_ylabel('W m-2'); a.legend(fontsize=8, loc='upper left')
    # 7 temperature, VPD
    a = ax[6]; a.plot(doy_t, sm(ta_t), 'k', lw=1.3, label='tower air T (24-h mean)')
    for k, (lab, R) in enumerate(RUNS):
        day = field(R, 'par_u', n) > 10.0; a.plot(days, sm(daily(field(R, 'tu', n), mask=day)), color=COL[k], lw=1.0, label=f'{lab}: leaf T (daylight)')
    a.set_ylabel('C'); a.legend(fontsize=7, ncol=2, loc='upper left'); b = a.twinx(); b.plot(doy_t, sm(vpd_t), color='tab:brown', lw=1.0, alpha=0.8); b.set_ylabel('tower VPD hPa', color='tab:brown')
    # 8 water
    a = ax[7]
    for k, (lab, R) in enumerate(RUNS):
        a.plot(days, sm(daily(field(R, 'stress_u', n))), color=COL[k], lw=1.3, label=f'{lab}: soil stress')
        a.plot(days, sm(daily(field(R, 'fwet_u', n))), color=COL[k], lw=0.8, ls=':', label=f'{lab}: wet fraction')
    a.set_ylim(-0.02, 1.05); a.set_ylabel('0-1'); a.legend(fontsize=7, ncol=2, loc='lower left')
    if np.isfinite(swc_t).any(): b = a.twinx(); b.plot(doy_t, sm(swc_t), color='tab:cyan', lw=1.0); b.set_ylabel('tower SWC %', color='tab:cyan')
    # 9 ET
    a = ax[8]; a.plot(doy_t, sm(le_t), 'k', lw=1.4, label=f'tower {le_src}')
    for k, (lab, R) in enumerate(RUNS): a.plot(days, sm(np.asarray(R['day_aet'])[:, n]), color=COL[k], lw=1.2, label=lab)
    a.set_ylabel('ET mm d-1'); a.legend(fontsize=8, ncol=4, loc='upper left')
    # 10 ET partition, monthly
    a = ax[9]; wbar = 0.8 / len(RUNS); mm = np.arange(1, 13)
    for k, (lab, R) in enumerate(RUNS):
        tr, it, so = (np.asarray(R[f'mon_{q}'])[:, n] for q in ('trans', 'interc', 'soilev')); x = mm - 0.4 + wbar * (k + 0.5)
        a.bar(x, tr, wbar, color=COL[k], alpha=0.9); a.bar(x, it, wbar, bottom=tr, color=COL[k], alpha=0.45); a.bar(x, so, wbar, bottom=tr + it, color=COL[k], alpha=0.2, hatch='//')
    a.plot(mm, le_mon_t, 'k-o', ms=3, lw=1.2, label=f'tower monthly ET ({le_src})'); a.set_xticks(mm); a.set_ylabel('ET mm month-1')
    a.set_title('per run: transpiration (dark) / interception (medium) / soil evaporation (hatched)', fontsize=8); a.legend(fontsize=8, loc='upper left')
    # 11 NEE monthly
    a = ax[10]; a.plot(mm, nee_mon_t, 'k-o', ms=3, lw=1.4, label='tower NEE'); a.axhline(0, color='0.6', lw=0.6)
    for k, (lab, R) in enumerate(RUNS):
        npp, rh = (np.asarray(R[f'mon_{q}'])[:, n] * 1000.0 for q in ('npp', 'rh'))
        a.plot(mm, -(npp - rh), color=COL[k], lw=1.3, marker='.', label=f'{lab}: NEE'); a.plot(mm, rh, color=COL[k], lw=0.8, ls=':', label=f'{lab}: Rh')
    a.set_xticks(mm); a.set_ylabel('gC m-2 month-1'); a.legend(fontsize=7, ncol=3, loc='lower left')
    for a in ax[:9]: a.set_xlim(1, nd)
    for a in ax[:9]: a.set_xlabel('day of year', fontsize=8)
    ax[9].set_xlabel('month'); ax[10].set_xlabel('month')
    fig.tight_layout(); p = f'{out_dir}/{s}_{year}.png'; fig.savefig(p, dpi=105); plt.close(fig); print('figure', p)
    # numbers behind the panels
    rows = []
    for lab, R in RUNS:
        day = field(R, 'par_u', n) > 10.0; g = np.asarray(R['day_gpp'])[:, n] * 1000.0; jja = slice(151, 243)
        rows.append(dict(run=lab, gpp_year=g.sum(), gpp_jja_mean=g[jja].mean(), tower_jja_mean=np.nanmean(gpp_t[(doy_t > 151) & (doy_t <= 243)]),
                         lai_peak=float(np.nanmax(daily(field(R, 'lai_u', n)) * float(R['fu'][n]))),
                         vmax_eff_jja=np.nanmean(daily(field(R, f'vmax_{leaf}', n) * 1e6, mask=day)[jja]),
                         cap_limited_jja=np.nanmean(daily(((np.minimum(field(R, f'jc_{leaf}', n) * field(R, 'fc_u', n), field(R, f'js_{leaf}', n) * field(R, 'fc_u', n)) < field(R, f'je_{leaf}', n) * field(R, 'fl_u', n))).astype(float), mask=day)[jja]),
                         eff_leaf_cap_jja=np.nanmean(daily(field(R, 'fc_u', n) * field(R, 'lai_u', n), mask=day)[jja]),
                         stress_jja=np.nanmean(daily(field(R, 'stress_u', n))[jja]), et_year=np.asarray(R['day_aet'])[:, n].sum()))
    T = pd.DataFrame(rows).set_index('run'); pd.set_option('display.width', 220)
    print(f'{s} {year}: tower GPP year {np.nansum(gpp_t):.0f} gC m-2, tower ET year {np.nansum(le_t):.0f} mm ({le_src}), site LAI max {lai_site}, MODIS cell LAI max {np.nanmax(mod):.1f}'); print(T.round(3).to_string())
    T.to_csv(f'{out_dir}/{s}_{year}_numbers.csv')
