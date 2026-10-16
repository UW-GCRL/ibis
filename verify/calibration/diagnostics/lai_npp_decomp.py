"""Where the leaf-area excess comes from: the NPP it is grown from, or the leaf parameters (CALIBRATION.md 47).

    python verify/calibration/diagnostics/lai_npp_decomp.py <run out dir> [--years 2001-2010] [--cue-obs 0.5] [--modis-ground 0.72]

IBIS's leaf pool relaxes to aleaf x tauleaf x NPP, and LAI = pool x specla, so the peak leaf area factors as
    LAI_model / LAI_obs = (GPP_model / GPP_obs) x (CUE_model / CUE_obs) x (LAI per NPP)_model / (LAI per NPP)_obs
with CUE = NPP / GPP. The last factor is what the leaf parameters (aleaf, tauleaf, specla) and the phenology own;
the first two are the productivity the leaves are grown from. Observations: GPP FLUXCOM RS+METEO; peak LAI MODIS (the
CLM surface-data PFT LAI, as site_fit uses it), raw and divided by the MODIS/ground ratio at the towers (0.72,
logs/modis_vs_ground_lai.csv); CUE_obs a literature value (default 0.5; there is no gridded NPP product on disk).
Area-weighted over true land north of 60 S, by latitude band and Koppen group (Beck 2023, 1991-2020).
"""
import sys, glob, re, numpy as np, xarray as xr
sys.path.insert(0, 'verify/calibration'); from landmask import true_landfrac
a = sys.argv[1:]; run = a[0]
def opt(k, d): return type(d)(a[a.index(k) + 1]) if k in a else d
y0, y1 = (int(v) for v in opt('--years', '2001-2010').split('-')); cue_obs = opt('--cue-obs', 0.5); mg = opt('--modis-ground', 0.72)
LAT = np.arange(89.75, -90, -0.5); lf = true_landfrac()
area = np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * lf
land = (lf > 0.05) & (LAT[:, None] > -60)
yrs = range(y0, y1 + 1)
gpp = np.mean([np.nan_to_num(xr.open_dataset(f'{run}/ibis_yearly_{y}.nc')['aygpptot'].values.astype(float)) for y in yrs], axis=0)
npp = np.mean([np.nan_to_num(xr.open_dataset(f'{run}/ibis_yearly_{y}.nc')['aynpptot'].values.astype(float)) for y in yrs], axis=0)
lmax = []
for y in yrs:
    m = []
    for k in range(1, 13):
        with xr.open_dataset(f'{run}/ibis_monthly_{y}_{k:02d}.nc') as x:
            m.append(np.nan_to_num((x['fu'] * x['laiu'] + x['fl'] * x['lail']).values.astype(float)))
    lmax.append(np.max(m, axis=0))
lai = np.mean(lmax, axis=0)
with xr.open_dataset('/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc') as fc:
    ys = np.array([int(str(t)[:4]) for t in fc['time'].values]); sel = (ys >= y0) & (ys <= y1)
    gpp_o = np.nanmean(fc['gpp'].values[sel].astype(float), axis=0) * 365.25 / 1000.0
    if float(fc['lat'].values[0]) < float(fc['lat'].values[-1]): gpp_o = gpp_o[::-1]
MB = '/shared/nas2/Hangkai/ELM Edge Effects/inputdata'
with xr.open_dataset(f'{MB}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc') as ds, xr.open_dataset(f'{MB}/surfdata_0.5x0.5_CH_v1_all_forests.nc') as sd:
    pv = sd['PCT_NATVEG'].values / 100.0; pp = sd['PCT_NAT_PFT'].values / 100.0; nt = ds.sizes['time']
    mm = np.array([sum(np.nan_to_num(ds[f'LAI_{k}'].values[[t for t in range(nt) if t % 12 == mo]].mean(axis=0)) * pp[k]
                       for k in range(1, 17) if f'LAI_{k}' in ds) * pv for mo in range(12)])
    if float(sd['LATIXY'].values[0, 0]) < float(sd['LATIXY'].values[-1, 0]): mm = mm[:, ::-1]
lai_o = mm.max(axis=0)
kz = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); kc = kz['classes'][list(kz['period_start']).index(1991)].astype(int)
if float(kz['lat'][0]) < float(kz['lat'][-1]): kc = kc[::-1]
grp = np.select([(kc >= 1) & (kc <= 3), (kc >= 4) & (kc <= 7), (kc >= 8) & (kc <= 16), (kc >= 17) & (kc <= 28), (kc >= 29)], list('ABCDE'), '-')
ok = land & np.isfinite(gpp_o) & (lai_o > 0)
regions = {'global': ok, '90S-30N': ok & (LAT[:, None] < 30), '30N-60N': ok & (LAT[:, None] >= 30) & (LAT[:, None] < 60), '60N-90N': ok & (LAT[:, None] >= 60)}
regions.update({f'Koppen {g}': ok & (grp == g) for g in 'ABCDE'})
W = lambda f, m: float((f[m] * area[m]).sum() / area[m].sum())
print(f'{run}, {y0}-{y1}; CUE_obs {cue_obs}; MODIS/ground {mg}\n')
print(f"{'region':10s} {'LAI x MODIS':>11s} {'(x ground)':>10s} = {'GPP x FLX':>9s} * {'CUE x obs':>9s} * {'LAI/NPP x obs':>13s} {'(x ground)':>10s}   model CUE, LAI/NPP; obs LAI/NPP (ground)")
for r, m in regions.items():
    rl = W(lai, m) / W(lai_o, m); rg = W(gpp, m) / W(gpp_o, m); cue = W(npp, m) / W(gpp, m); rc = cue / cue_obs
    lpn_m = W(lai, m) / W(npp, m); lpn_o = W(lai_o, m) / (W(gpp_o, m) * cue_obs)
    print(f'{r:10s} {rl:11.2f} {rl * mg:10.2f} = {rg:9.2f} * {rc:9.2f} * {rl / (rg * rc):13.2f} {rl * mg / (rg * rc):10.2f}   {cue:.2f}, {lpn_m:.1f}; {lpn_o:.1f} ({lpn_o / mg:.1f})')
