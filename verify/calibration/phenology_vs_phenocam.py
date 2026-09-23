"""Leaf-out and senescence timing at towers: model daily GPP vs PhenoCam greenness vs tower daily GPP (CALIBRATION.md 20, stage S1.4 target).

    ibis-python verify/calibration/phenology_vs_phenocam.py <site_runs dir with gpp_daily> [<out png>]

Transition dates follow the PhenoCam convention: the day the 15-day smoothed seasonal curve crosses 25 % (and 50 %) of its annual
amplitude on the way up (rising, spring) and on the way down (falling, autumn), per year, then averaged over the run years.
Cameras: the tower's matched ROIs (phenocam_tower_matches.csv) whose type fits the tower class (DB/EN/GR/SH; MF takes DB and EN),
`gcc_90` transition dates from the 3-day product. Tower GPP: GPP_NT_VUT_REF from the FLUXNET2015 FULLSET daily file, same smoothing.
Northern-hemisphere towers only (the crossing search assumes a mid-year maximum).
"""
import sys, glob, os, json, numpy as np, pandas as pd
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; C = '/shared/nas2/Hangkai/ibis/inputs/sites/PhenoCam'; FX = '/shared/nas2/Hangkai/ibis/inputs/sites/FLUXNET2015_FULLSET/daily_monthly_yearly'
ROI_FOR = {'DBF': ['DB'], 'ENF': ['EN'], 'GRA': ['GR'], 'OSH': ['SH'], 'CSH': ['SH'], 'MF': ['DB', 'EN'], 'WSA': ['DB', 'GR'], 'SAV': ['DB', 'GR'], 'EBF': ['EB', 'DB'], 'DNF': ['DN']}


def smooth(x, w=15):
    x = np.asarray(x, float); k = np.ones(w) / w; xp = np.pad(x, (w // 2, w // 2), mode='edge'); return np.convolve(xp, k, mode='valid')[:len(x)]


def crossings(daily, frac, ref=None):
    """(rising DOY, falling DOY) where a smoothed daily series with a mid-year maximum crosses `frac` of its annual amplitude; NaN if flat.

    `ref=(lo, hi)` times the run against a FIXED level instead of its own amplitude. Sensitivity runs must use the base run's
    (lo, hi), otherwise a perturbation that only changes the magnitude moves the self-referential level and fakes a date shift."""
    y = smooth(np.nan_to_num(daily, nan=0.0)); lo, hi = (np.nanpercentile(y, 5), np.nanmax(y)) if ref is None else ref
    if hi - lo <= 0 or hi < 0.5: return np.nan, np.nan            # < 0.5 gC/m2/d peak: no season to time
    lev = lo + frac * (hi - lo); imax = int(np.argmax(y))
    if y[imax] < lev: return np.nan, np.nan                        # never reaches the fixed level
    above = y >= lev
    r = next((d for d in range(imax, -1, -1) if not above[d]), -1) + 1           # last below-level day before the peak, +1
    f = next((d for d in range(imax, len(y)) if not above[d]), len(y)) - 1      # last above-level day after the peak
    return float(r + 1), float(f + 1)                                            # 1-based DOY


def cam_dates(tower, igbp, years, matches, all_years=False):
    rows = matches[(matches['tower'] == tower) & (matches['roitype'].isin(ROI_FOR.get(igbp, ['DB', 'EN', 'GR', 'SH'])))]
    out = {'r25': [], 'f25': [], 'r50': [], 'f50': []}; n_roi = 0
    for _, r in rows.iterrows():
        f = f"{C}/{r['roi']}_3day_transition_dates.csv"
        if not os.path.exists(f): continue
        d = pd.read_csv(f, comment='#'); d = d[d['gcc_value'] == 'gcc_90']; n_roi += 1
        for _, x in d.iterrows():
            t25 = pd.to_datetime(x['transition_25']); t50 = pd.to_datetime(x['transition_50'])
            if not all_years and t25.year not in years: continue
            if x['direction'] == 'rising': out['r25'].append(t25.dayofyear); out['r50'].append(t50.dayofyear)
            else: out['f25'].append(t25.dayofyear); out['f50'].append(t50.dayofyear)
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in out.items()}, n_roi, len(out['r25'])


def tower_gpp_dates(site, years):
    f = glob.glob(f'{FX}/FLX_{site}_FLUXNET2015_FULLSET_DD_*.csv')
    if not f: return {}, 0
    d = pd.read_csv(f[0], usecols=['TIMESTAMP', 'GPP_NT_VUT_REF']); t = pd.to_datetime(d['TIMESTAMP'].astype(str), format='%Y%m%d'); d['y'] = t.dt.year
    g = d['GPP_NT_VUT_REF'].where(d['GPP_NT_VUT_REF'] > -9000)
    acc = {'r25': [], 'f25': [], 'r50': [], 'f50': []}; n = 0
    for y in years:
        s = g[d['y'] == y].values
        if len(s) < 360 or np.isnan(s).mean() > 0.2: continue
        r25, f25 = crossings(s, 0.25); r50, f50 = crossings(s, 0.50)
        if np.isfinite(r25): acc['r25'].append(r25); acc['f25'].append(f25); acc['r50'].append(r50); acc['f50'].append(f50); n += 1
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in acc.items()}, n


def main():
    run = sys.argv[1]; out_png = sys.argv[2] if len(sys.argv) > 2 else f'{run}/phenology_vs_phenocam.png'
    z = np.load(f'{run}/site_runs.npz', allow_pickle=True); meta = json.load(open(f'{run}/site_runs.json'))
    sites = [str(s) for s in z['sites']]; years = [int(y) for y in z['years']]; gd = z['gpp_daily']; lat = None
    matches = pd.read_csv(f'{C}/phenocam_tower_matches.csv')
    rows = []
    for n, s in enumerate(sites):
        igbp = meta['notes'][s]['igbp']
        acc = {'r25': [], 'f25': [], 'r50': [], 'f50': []}
        for yi, y in enumerate(years):
            r25, f25 = crossings(gd[yi, :, n], 0.25); r50, f50 = crossings(gd[yi, :, n], 0.50)
            if np.isfinite(r25): acc['r25'].append(r25); acc['f25'].append(f25); acc['r50'].append(r50); acc['f50'].append(f50)
        mod = {k: (float(np.mean(v)) if v else np.nan) for k, v in acc.items()}
        cam, n_roi, n_cam_yrs = cam_dates(s, igbp, set(years), matches); cam_src = 'run years'
        if n_cam_yrs < 2:
            cam, n_roi, n_cam_yrs = cam_dates(s, igbp, set(years), matches, all_years=True); cam_src = 'all camera years' if n_cam_yrs else 'none'
        tow, n_tow = tower_gpp_dates(s, years)
        # winter-growing (Mediterranean) sites: the model's annual minimum falls in June-August; the mid-year-peak crossing search does not apply
        gm = np.nanmean(np.nan_to_num(gd[:, :365, n]), axis=0); winter_growing = 152 <= int(np.argmin(smooth(gm))) <= 243
        # ... or the camera or the tower says so (the model may fail to be Mediterranean where the site is: Vaira, Tonzi)
        if np.isfinite(cam.get('r25', np.nan)) and cam['r25'] > 180: winter_growing = True
        if np.isfinite(tow.get('r25', np.nan)) and (tow['r25'] < 40 or tow['r25'] > 180): winter_growing = True
        rows.append(dict(site=s, igbp=igbp, years=f'{years[0]}-{years[-1]}', cam_rois=n_roi, cam_years=n_cam_yrs, cam_src=cam_src, tower_years=n_tow, winter_growing=winter_growing,
                         leafout25_model=mod['r25'], leafout25_cam=cam['r25'], leafout25_towerGPP=tow.get('r25', np.nan),
                         senesc25_model=mod['f25'], senesc25_cam=cam['f25'], senesc25_towerGPP=tow.get('f25', np.nan),
                         leafout50_model=mod['r50'], leafout50_cam=cam['r50'], senesc50_model=mod['f50'], senesc50_cam=cam['f50']))
    df = pd.DataFrame(rows)
    for k in ('leafout25', 'senesc25'):
        df[f'{k}_model_minus_cam'] = df[f'{k}_model'] - df[f'{k}_cam']; df[f'{k}_model_minus_towerGPP'] = df[f'{k}_model'] - df[f'{k}_towerGPP']; df[f'{k}_towerGPP_minus_cam'] = df[f'{k}_towerGPP'] - df[f'{k}_cam']
    df.to_csv(f'{run}/phenology_vs_phenocam.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['site', 'igbp', 'cam_years', 'cam_src', 'winter_growing', 'leafout25_cam', 'leafout25_towerGPP', 'leafout25_model', 'leafout25_model_minus_cam', 'senesc25_cam', 'senesc25_towerGPP', 'senesc25_model', 'senesc25_model_minus_cam']
    print(df[cols].to_string(index=False, float_format=lambda x: f'{x:.0f}'))
    ok = df.dropna(subset=['leafout25_cam', 'leafout25_model']); ok = ok[~ok['winter_growing']]
    print(f"\nwinter-growing towers excluded from the means: {list(df[df['winter_growing']]['site'])}")
    g = ok.groupby('igbp').agg(n=('site', 'count'), leafout_model_minus_cam=('leafout25_model_minus_cam', 'mean'), senesc_model_minus_cam=('senesc25_model_minus_cam', 'mean'), leafout_model_minus_towerGPP=('leafout25_model_minus_towerGPP', 'mean'), senesc_model_minus_towerGPP=('senesc25_model_minus_towerGPP', 'mean'), leafout_towerGPP_minus_cam=('leafout25_towerGPP_minus_cam', 'mean'), senesc_towerGPP_minus_cam=('senesc25_towerGPP_minus_cam', 'mean'))
    print('\nby class (days; positive = model later than the camera):'); print(g.round(0).to_string())
    print(f"\nsummer-growing towers with a camera date: n {len(ok)}; leaf-out model minus camera {ok['leafout25_model_minus_cam'].mean():+.0f} d, model minus tower GPP {ok['leafout25_model_minus_towerGPP'].mean():+.0f} d (tower GPP minus camera {ok['leafout25_towerGPP_minus_cam'].mean():+.0f} d); senescence model minus camera {ok['senesc25_model_minus_cam'].mean():+.0f} d, model minus tower GPP {ok['senesc25_model_minus_towerGPP'].mean():+.0f} d (tower GPP minus camera {ok['senesc25_towerGPP_minus_cam'].mean():+.0f} d)")
    allsg = df[~df['winter_growing']].dropna(subset=['leafout25_towerGPP', 'leafout25_model'])
    print(f"summer-growing towers with tower GPP (camera or not): n {len(allsg)}; leaf-out model minus tower GPP {allsg['leafout25_model_minus_towerGPP'].mean():+.0f} d; senescence {allsg['senesc25_model_minus_towerGPP'].mean():+.0f} d")
    print(allsg.groupby('igbp').agg(n=('site','count'), leafout_model_minus_towerGPP=('leafout25_model_minus_towerGPP','mean'), senesc_model_minus_towerGPP=('senesc25_model_minus_towerGPP','mean')).round(0).to_string())
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
        for ax, k, ttl in ((axs[0], 'leafout25', 'leaf-out: 25 % of amplitude, rising'), (axs[1], 'senesc25', 'senescence: 25 % of amplitude, falling')):
            for igbp, grp in ok.groupby('igbp'):
                ax.scatter(grp[f'{k}_cam'], grp[f'{k}_model'], s=28, label=f'{igbp} (n={len(grp)})'); ax.scatter(grp[f'{k}_cam'], grp[f'{k}_towerGPP'], s=18, marker='x', color='0.4')
            lo, hi = (60, 200) if k == 'leafout25' else (220, 365); ax.plot([lo, hi], [lo, hi], 'k', lw=0.7); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel('PhenoCam transition date (day of year)'); ax.set_ylabel('model (filled) / tower GPP (x) transition date'); ax.set_title(ttl, fontsize=10, loc='left'); ax.legend(fontsize=8, frameon=False)
        fig.suptitle(f'Phenology timing at {len(ok)} PhenoCam towers, {years[0]}-{years[-1]}, {os.path.basename(meta["params"])}: model daily GPP vs camera greenness (and tower GPP)', fontsize=10, x=0.01, ha='left')
        fig.tight_layout(); fig.savefig(out_png, dpi=140); print('figure', out_png)
    except ModuleNotFoundError: print('no matplotlib in this interpreter; figure skipped')


if __name__ == '__main__':
    main()
