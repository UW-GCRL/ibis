"""Methane towers (Chenchen's gap-filled AmeriFlux + EuroFlux sets) in the FLUXNET2015 file layout the site harness reads
(CALIBRATION.md 38). One directory with three roles, selected with the IBIS_FLUX_* environment variables of tower_forcing:

    <out>/compact/     FLX_<s>_FLUXNET2015_FULLSET_DD_* (daily meteorology), _HH_* (compact: TA, SW), _MM_*, _YY_*, FLUXNET_siteinfo.csv
    <out>/halfhourly/  FLX_<s>_FLUXNET2015_FULLSET_HH_* (full: NIGHT, NEE, FCH4 and its gap-fill + uncertainty)
    <out>/monthly/     FLX_<s>_FLUXNET2015_FULLSET_MM_* (same as compact; site_fit.tower_obs reads them from its own FX)
    <out>/ch4_targets.npz   monthly CH4 (gC-CH4 m-2 month-1, + = emission) per site and year, sigma, measured share

    python verify/calibration/ch4_tower_prep.py <out dir> [SITE ...]

Meteorology is the FLUXNET-format half-hourly record of each site (ERA-filled *_F columns, so there are no gaps and the
GSWP3 fallback is not exercised); the daily file carries the FLUXNET daily names the reader expects. GPP/NEE/LE are the
FLUXNET partitioning columns of the same files (umol -> gC m-2 d-1 monthly means in the MM file). CH4: FCH4_F of the
random-forest gap-fill at the USTAR50 threshold (nmol CH4 m-2 s-1), with FCH4_F_UNCERTAINTY; budgets in gC-CH4 m-2 yr-1.
Sites without coordinates or without a USTAR50 budget are skipped; croplands are written but flagged (IBIS has no crop PFT).
"""
import sys, os, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
out = sys.argv[1]; want = sys.argv[2:]
ROOTS = {'AmeriFlux': '/shared/nas/Chenchen/AmeriFLuxet/combined', 'EuroFlux': '/shared/nas/Chenchen/Eurofluxnet/combined'}
META = pd.read_csv('/shared/nas/Chenchen/FLUXNET-CH4/v2/v2_site_metadata.csv').drop_duplicates('SITE_ID').set_index('SITE_ID')
COORDS = pd.read_csv('/shared/nas/Chenchen/hangkai/combined_site_coords.csv').set_index('Site')
GC = 12.011e-6 * 86400.0                                      # umol m-2 s-1 -> gC m-2 d-1
for sub in ('compact', 'halfhourly', 'monthly'): os.makedirs(f'{out}/{sub}', exist_ok=True)
MET = ['TA_F', 'SW_IN_F', 'LW_IN_F', 'VPD_F', 'PA_F', 'P_F', 'WS_F']
COLS = ['TIMESTAMP_START', 'SW_IN_POT', 'NEE_VUT_REF', 'NEE_VUT_REF_QC', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'LE_F_MDS', 'LE_F_MDS_QC', 'FCH4'] + MET
info = []; T = {}
for net, root in ROOTS.items():
    for d in sorted(glob.glob(f'{root}/*/')):
        s = os.path.basename(d.rstrip('/'))
        if want and s not in want: continue
        gf = f'{d}gapfilled/rf_predictors_laplace_FCH4_CUT_USTAR50.csv'; bf = f'{d}gapfilled/rf_predictors_laplace_FCH4_CUT_USTAR50_budget.csv'
        if not (os.path.exists(gf) and os.path.exists(bf) and os.path.exists(f'{d}raw.csv')): print(f'{s}: no USTAR50 gap-fill, skipped', flush=True); continue
        lat = META.LAT.get(s, COORDS.Lat.get(s, np.nan)); lon = META.LON.get(s, COORDS.Lon.get(s, np.nan))
        if not np.isfinite(lat): print(f'{s}: no coordinates, skipped', flush=True); continue
        igbp = META.IGBP.get(s, '?')
        r = pd.read_csv(f'{d}raw.csv', usecols=lambda c: c in COLS, na_values=[-9999, -9999.0]); t = pd.DatetimeIndex(pd.to_datetime(r['TIMESTAMP_START'].astype(str), format='%Y%m%d%H%M')); r.index = t
        step = (t[1] - t[0]).total_seconds()                                                    # 1800 or 3600 s
        g = pd.read_csv(gf, usecols=['TIMESTAMP_START', 'FCH4_F', 'FCH4_F_UNCERTAINTY'], na_values=[-9999]); g.index = pd.to_datetime(g['TIMESTAMP_START'].astype(str), format='%Y%m%d%H%M')
        r = r.join(g[['FCH4_F', 'FCH4_F_UNCERTAINTY']], how='left')
        for c in COLS:
            if c not in r: r[c] = np.nan                                                        # a few sites lack a partitioning column
        y0, y1 = int(t.year.min()), int(t.year.max()); tag = f'{y0}-{y1}'
        # --- daily meteorology (compact DD) ---
        day = r.index.floor('D')
        dd = pd.DataFrame({'TA_F_MDS': r.TA_F.groupby(day).mean(), 'P_F': r.P_F.groupby(day).sum(min_count=1), 'SW_IN_F_MDS': r.SW_IN_F.groupby(day).mean(),
                           'LW_IN_F_MDS': r.LW_IN_F.groupby(day).mean(), 'VPD_F_MDS': r.VPD_F.groupby(day).mean(), 'PA_F': r.PA_F.groupby(day).mean(), 'WS_F': r.WS_F.groupby(day).mean()})
        dd.insert(0, 'TIMESTAMP', dd.index.strftime('%Y%m%d')); dd.fillna(-9999.0).to_csv(f'{out}/compact/FLX_{s}_FLUXNET2015_FULLSET_DD_{tag}.csv', index=False, float_format='%.4f')
        # --- compact half-hourly: TA and SW (diurnal range, hourly light) ---
        hh = pd.DataFrame({'TIMESTAMP': r.index.strftime('%Y-%m-%d %H:%M:%S'), 'TA_F_MDS': r.TA_F.values, 'SW_IN_F_MDS': r.SW_IN_F.values})
        hh.to_csv(f'{out}/compact/FLX_{s}_FLUXNET2015_FULLSET_HH_{tag}.csv', index=False, float_format='%.3f')
        # --- full half-hourly: NEE day/night and CH4 ---
        full = pd.DataFrame({'TIMESTAMP_START': r['TIMESTAMP_START'].values, 'NIGHT': (r.SW_IN_POT.fillna(0) <= 0).astype(int).values,
                             'NEE_VUT_REF': r.NEE_VUT_REF.values, 'NEE_VUT_REF_QC': r.NEE_VUT_REF_QC.values, 'SW_IN_F_MDS': r.SW_IN_F.values, 'TA_F_MDS': r.TA_F.values,
                             'FCH4': r.FCH4.values, 'FCH4_F': r.FCH4_F.values, 'FCH4_F_UNCERTAINTY': r.FCH4_F_UNCERTAINTY.values})
        full.fillna(-9999).to_csv(f'{out}/halfhourly/FLX_{s}_FLUXNET2015_FULLSET_HH_{tag}.csv', index=False, float_format='%.4f')
        # --- monthly carbon / energy (MM) and yearly (YY) ---
        mon = r.index.to_period('M')
        mm = pd.DataFrame({'GPP_NT_VUT_REF': r.GPP_NT_VUT_REF.groupby(mon).mean() * GC, 'GPP_DT_VUT_REF': r.GPP_DT_VUT_REF.groupby(mon).mean() * GC,
                           'NEE_VUT_REF': r.NEE_VUT_REF.groupby(mon).mean() * GC, 'NEE_VUT_REF_QC': (r.NEE_VUT_REF_QC <= 1).groupby(mon).mean(),
                           'LE_F_MDS': r.LE_F_MDS.groupby(mon).mean(), 'LE_F_MDS_QC': (r.LE_F_MDS_QC <= 1).groupby(mon).mean(), 'TA_F_MDS': r.TA_F.groupby(mon).mean()})
        mm.insert(0, 'TIMESTAMP', [p.strftime('%Y%m') for p in mm.index]); mm = mm.fillna(-9999.0)
        for sub in ('compact', 'monthly'): mm.to_csv(f'{out}/{sub}/FLX_{s}_FLUXNET2015_FULLSET_MM_{tag}.csv', index=False, float_format='%.4f')
        yy = pd.DataFrame({'TA_F_MDS': r.TA_F.groupby(r.index.year).mean(), 'P_F': r.P_F.groupby(r.index.year).sum(min_count=1)}); yy.index = yy.index.astype(int); yy.insert(0, 'TIMESTAMP', yy.index.astype(str))
        yy.fillna(-9999.0).to_csv(f'{out}/compact/FLX_{s}_FLUXNET2015_FULLSET_YY_{tag}.csv', index=False, float_format='%.4f')
        # --- CH4 targets: monthly sums of the gap-filled flux (nmol m-2 s-1 -> gC-CH4 m-2 month-1) ---
        conv = 12.011e-9 * step
        ch4 = pd.DataFrame({'sum': r.FCH4_F.groupby(mon).sum(min_count=1) * conv, 'n': r.FCH4_F.groupby(mon).count(), 'meas': r.FCH4.notna().groupby(mon).mean(),
                            'sig': np.sqrt((r.FCH4_F_UNCERTAINTY ** 2).groupby(mon).sum(min_count=1)) * conv, 'ndays': r.FCH4_F.groupby(mon).count() * step / 86400.0})
        ch4 = ch4[ch4.ndays >= 20]                                                                   # months with at least 20 days of record
        T[s] = ch4
        bud = pd.read_csv(bf)
        info.append(dict(SITE_ID=s, SITE_NAME=META.SITE_NAME.get(s, s), FLUXNET2015='', **{'FLUXNET-CH4': net}, LOCATION_LAT=lat, LOCATION_LONG=lon, LOCATION_ELEV=META.ELEV.get(s, np.nan),
                         IGBP=igbp, MAT=round(float(r.TA_F.mean()), 2), MAP=round(float(yy.P_F[yy.P_F > 0].mean()), 0) if (yy.P_F > 0).any() else np.nan,
                         years=tag, nyr_ch4=len(bud), budget_gC=round(float(bud.budget_mean.mean()), 2), step_s=int(step), ch4_months=len(ch4), meas_share=round(float(ch4.meas.mean()), 2)))
        print(f'{s:7s} {net:9s} {igbp:4s} {lat:7.2f} {lon:8.2f} {tag} step {int(step)} s  CH4 months {len(ch4):3d} measured {ch4.meas.mean():.2f}  budget {bud.budget_mean.mean():6.2f} gC m-2 yr-1', flush=True)
I = pd.DataFrame(info); I.to_csv(f'{out}/compact/FLUXNET_siteinfo.csv', index=False); I.to_csv(f'{out}/ch4_sites.csv', index=False)
sites = list(T); years = list(range(min(int(T[s].index.min().year) for s in sites), max(int(T[s].index.max().year) for s in sites) + 1))
A = {k: np.full((len(years), 12, len(sites)), np.nan) for k in ('sum', 'sig', 'n', 'meas', 'ndays')}
for n, s in enumerate(sites):
    for p, row in T[s].iterrows():
        for k in A: A[k][years.index(p.year), p.month - 1, n] = row[k]
np.savez(f'{out}/ch4_targets.npz', sites=np.array(sites), years=np.array(years), **{f'{k}_ch4': v for k, v in A.items()})
print(f'\n{len(sites)} sites, years {years[0]}-{years[-1]}; {I.IGBP.value_counts().to_dict()}; wrote {out}/ch4_targets.npz and ch4_sites.csv')
