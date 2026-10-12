"""Global totals of a transient run against the budget products (CALIBRATION.md 39; decades 2001-2010 style and period means since 2026-10-12, CALIBRATION.md 44 -- before that the last "decade" row was the single year 2010): land sink by decade vs the Global
Carbon Budget 2024 and the TRENDY-v13 ensemble, wetland CH4 vs the Global Methane Budget 2025 (global and by band), GPP,
and the carbon pools.

    python verify/calibration/diagnostics/mip_compare.py <out csv> <yearly dir> [<yearly dir> ...] [--years 1991-2010]

Yearly files `ibis_yearly_YYYY.nc` (lat, lon) from python-port/scripts/run_global.py; several directories are read as one
record (transient + hindcast). Area: cell area x the land fraction of surta.nc. NEE = NPP - Rh (+ = sink); the run has no
land-use change, so it is compared with GCB's S_LAND (the TRENDY S2 sink), not with the net land flux.
"""
import sys, re, glob, os, numpy as np, pandas as pd, xarray as xr
out = sys.argv[1]; args = sys.argv[2:]; yrs = (1991, 2010)
if '--years' in args: k = args.index('--years'); yrs = tuple(int(v) for v in args[k + 1].split('-')); args = args[:k] + args[k + 2:]
CUM = '--ch4-cumulative' in args                                                            # runs before the aych4 reset fix: difference successive years
if CUM: args.remove('--ch4-cumulative')
dirs = args; MIP = '/shared/nas2/Hangkai/ibis/inputs/eval/mip'
sys.path.insert(0, 'verify/calibration'); from landmask import true_landfrac
LAT = np.arange(89.75, -90, -0.5); R = 6371.0
area = (R ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * 1.0e6 * true_landfrac()   # m2 of TRUE land per cell (state_report.py: the binary surta mask overstates land by ~20 %)
bands = {'90S-30N': LAT < 30, '30N-60N': (LAT >= 30) & (LAT < 60), '60N-90N': LAT >= 60}
files = {}
for d in dirs:
    for f in glob.glob(f'{d}/ibis_yearly_*.nc'): files[int(re.search(r'(\d+)\.nc$', f).group(1))] = f
rows = []; prev_ch4 = None
for y in sorted(files):
    if y < yrs[0] - 1 or y > yrs[1]: continue                                           # the year before the range gives the first year's dC/dt
    with xr.open_dataset(files[y]) as d:
        if not np.allclose(d['lat'].values[:3], LAT[:3]): raise SystemExit(f'{files[y]}: latitude runs {d["lat"].values[:2]}, expected north to south')
        g = lambda v: np.nan_to_num(np.asarray(d[v].values, float)) if v in d else None
        tot = lambda a: float(np.sum(a * area)) * 1e-12                                     # kgC m-2 -> PgC
        r = dict(year=y, gpp=tot(g('aygpptot')), npp=tot(g('aynpptot')), rh=tot(g('ayco2mic')), nee=tot(g('aynpptot') - g('ayco2mic')),
                 cveg=tot(g('cveg')), csoil=tot(g('totcsoi')), clit=tot(g('totlit')) if g('totlit') is not None else np.nan)
        if g('aych4') is not None:
            a = g('aych4')
            if CUM: a, prev_ch4 = (a - prev_ch4 if prev_ch4 is not None else None), a
            if a is not None:
                ch4 = a * area * 1e-9 * (16.043 / 12.011)                                  # kgC m-2 yr-1 -> Tg CH4 yr-1 per cell
                r['ch4'] = float(ch4.sum()); r.update({f'ch4_{b}': float(ch4[mask].sum()) for b, mask in bands.items()})
        rows.append(r)
D = pd.DataFrame(rows)
if D.empty: raise SystemExit('no yearly files in the range')
# net land flux as the change of the stored carbon (wood + litter + soil; leaves and fine roots are not in the yearly files),
# which is what the budget's land sink measures; NPP - Rh also carries fire and disturbance losses, so it is larger
D['ctot'] = D.cveg + D.csoil + D.clit.fillna(0.0); D['nbp'] = D.ctot.diff().where(D.year.diff() == 1)
D = D[D.year >= yrs[0]].reset_index(drop=True)
D.to_csv(out, index=False)
pd.set_option('display.width', 220); dec = D.assign(decade=((D.year - 1) // 10) * 10 + 1).groupby('decade').mean(numeric_only=True).drop(columns='year')
per = D.mean(numeric_only=True)
print(f'model totals by decade (labelled by first year, 2001 = 2001-2010; {yrs[0]}-{yrs[1]}; PgC yr-1, pools PgC, CH4 Tg CH4 yr-1; nbp = change of wood + litter + soil carbon):'); print(dec.drop(columns=['ctot']).round(2).to_string())
# --- GCB 2024: land sink (S_LAND) and the TRENDY-v13 DGVM decadal means
G = pd.read_csv(f'{MIP}/gcb2024/GCB2024_v1.1_GlobalCarbonBudget_1959-2023.csv'); G['decade'] = ((G.Year - 1) // 10) * 10 + 1
sl = G.groupby('decade')['land sink'].agg(['mean', 'std'])
T = pd.read_csv(f'{MIP}/gcb2024/GCB2024_v1.1_TerrestrialSink_DGVM_decadal_means.csv', index_col=0)
print('\nland sink (PgC yr-1, + = uptake): model NEE vs GCB S_LAND (decadal mean, interannual sd) and the TRENDY-v13 models (2001-2010):')
for dcd in dec.index:
    gl = f"GCB {sl.loc[dcd, 'mean']:.2f} (sd {sl.loc[dcd, 'std']:.2f})" if dcd in sl.index else 'GCB n/a'
    print(f"  {dcd}-{dcd + 9}: model dC/dt {dec.loc[dcd, 'nbp']:+.2f} (NPP - Rh {dec.loc[dcd, 'nee']:+.2f})   {gl}")
c = '2001-2010' if '2001-2010' in T.columns else T.columns[0]; mm = T[c].drop([i for i in T.index if 'mean' in str(i).lower() or 'sd' in str(i).lower() or 'spread' in str(i).lower()], errors='ignore')
print(f"  TRENDY-v13 {c}: multi-model mean {mm.mean():.2f}, range {mm.min():.2f} ({mm.idxmin()}) to {mm.max():.2f} ({mm.idxmax()}), IBIS-v13 {T.loc['IBIS', c] if 'IBIS' in T.index else float('nan'):.2f}")
# --- GMB 2025: bottom-up wetlands
for per in ('2000-2009', '2010-2019'):
    f = f'{MIP}/gmb2024/GMB2024_BU_wetlands_{per}_ensemble_stats.csv'
    if os.path.exists(f):
        E = pd.read_csv(f, index_col=0)
        print(f"\nGMB 2025 bottom-up wetlands {per} (Tg CH4 yr-1): global {E.loc['mean', 'Global']:.0f} [{E.loc['min', 'Global']:.0f}-{E.loc['max', 'Global']:.0f}]; "
              + ', '.join(f"{b} {E.loc['mean', b]:.0f} [{E.loc['min', b]:.0f}-{E.loc['max', b]:.0f}]" for b in bands))
if 'ch4' in D:
    for dcd in dec.index:
        print(f"  model {dcd}-{dcd + 9}: global {dec.loc[dcd, 'ch4']:.0f}; " + ', '.join(f"{b} {dec.loc[dcd, f'ch4_{b}']:.0f}" for b in bands))
else: print('  (no aych4 in these yearly files: methane was not enabled in this run)')
print(f"\nGPP: model {per['gpp']:.0f} PgC yr-1 ({yrs[0]}-{yrs[1]} mean; stored-carbon sink {per['nbp']:+.2f} PgC yr-1) (TRENDY-v13 models ~115-135; FLUXCOM 2001-2010 ~120); vegetation C {per['cveg']:.0f} PgC (above-ground wood; literature total 450-650), soil C {per['csoil']:.0f} PgC (literature 1500-2400 to 1 m)")
