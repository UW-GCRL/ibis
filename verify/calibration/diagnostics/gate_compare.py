"""Two global runs from the same start, compared on GPP, tree cover, leaf area and ET (CALIBRATION.md 47).

    python verify/calibration/diagnostics/gate_compare.py <control run dir> <test run dir> [--years 2006-2010]

GPP against FLUXCOM RS+METEO, ET against the five-product ensemble, cell peak LAI (fu x laiu + fl x lail, monthly)
against MODIS (CLM surface-data PFT LAI), the upper-canopy cover fu (July) on the natural share against the CLM/MODIS
tree share; by latitude band and by observed land type; true land north of 60 S, area-weighted. Also the yearly global
GPP of both runs over their whole length, to see whether the test has settled.
"""
import sys, glob, re, numpy as np, xarray as xr
sys.path.insert(0, 'verify/calibration'); from landmask import true_landfrac
a = sys.argv[1:]; ctl, tst = a[0], a[1]
y0, y1 = (int(v) for v in (a[a.index('--years') + 1] if '--years' in a else '2006-2010').split('-')); YRS = range(y0, y1 + 1)
LAT = np.arange(89.75, -90, -0.5); lf = true_landfrac()
A = 6371e3 ** 2 * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * lf
land = (lf > 0.05) & (LAT[:, None] > -60)
rd = lambda f, v: np.nan_to_num(np.asarray(xr.open_dataset(f)[v].values, float))

def load(run):
    o = {}
    o['gpp'] = np.mean([rd(f'{run}/out/ibis_yearly_{y}.nc', 'aygpptot') for y in YRS], 0)
    o['npp'] = np.mean([rd(f'{run}/out/ibis_yearly_{y}.nc', 'aynpptot') for y in YRS], 0)
    o['et'] = np.mean([rd(f'{run}/out/ibis_yearly_{y}.nc', 'ayaet') for y in YRS], 0)
    lm = []; fu = []
    for y in YRS:
        m = [rd(f'{run}/out/ibis_monthly_{y}_{k:02d}.nc', 'fu') * rd(f'{run}/out/ibis_monthly_{y}_{k:02d}.nc', 'laiu')
             + rd(f'{run}/out/ibis_monthly_{y}_{k:02d}.nc', 'fl') * rd(f'{run}/out/ibis_monthly_{y}_{k:02d}.nc', 'lail') for k in range(1, 13)]
        lm.append(np.max(m, 0)); fu.append(rd(f'{run}/out/ibis_monthly_{y}_07.nc', 'fu'))
    o['lai'] = np.mean(lm, 0); o['fu'] = np.mean(fu, 0)
    yrs = sorted(int(re.search(r'(\d{4})\.nc$', f).group(1)) for f in glob.glob(f'{run}/out/ibis_yearly_*.nc'))
    o['series'] = {y: float((rd(f'{run}/out/ibis_yearly_{y}.nc', 'aygpptot') * A)[land].sum() * 1e-12) for y in yrs}
    return o
C, T = load(ctl), load(tst)
with xr.open_dataset('/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc') as fc:
    ys = np.array([int(str(t)[:4]) for t in fc['time'].values])
    gfx = np.nanmean(fc['gpp'].values[(ys >= y0) & (ys <= y1)].astype(float), 0) * 365.25 / 1000
z = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/et/et_ensemble_0.5deg.npz', allow_pickle=True); etr = z['ens_ann_mean']
if float(z['lat'][0]) < float(z['lat'][-1]): etr = etr[::-1]
MB = '/shared/nas2/Hangkai/ELM Edge Effects/inputdata'
with xr.open_dataset(f'{MB}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc') as ds, xr.open_dataset(f'{MB}/surfdata_0.5x0.5_CH_v1_all_forests.nc') as sd:
    pv = sd['PCT_NATVEG'].values / 100; pp = sd['PCT_NAT_PFT'].values / 100; nt = ds.sizes['time']
    mm = np.array([sum(np.nan_to_num(ds[f'LAI_{k}'].values[[t for t in range(nt) if t % 12 == mo]].mean(0)) * pp[k] for k in range(1, 17) if f'LAI_{k}' in ds) * pv for mo in range(12)])
    flip = float(sd['LATIXY'].values[0, 0]) < float(sd['LATIXY'].values[-1, 0])
    lmod = mm.max(0); tree = pp[1:9].sum(0); shrub = pp[9:12].sum(0); grass = pp[12:15].sum(0); bare = pp[0]
    if flip: lmod, tree, shrub, grass, bare = (x[::-1] for x in (lmod, tree, shrub, grass, bare))
cf = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz')['cropfrac'].clip(0, 1)
ok = land & np.isfinite(gfx)
P = lambda f, m: float((f * A)[m].sum() * 1e-12)                  # PgC yr-1 for kgC m-2 yr-1
M = lambda f, m: float((f * A)[m].sum() / A[m].sum())
print(f'control {ctl}\ntest    {tst}\nmeans {y0}-{y1}\n')
print('global GPP by year, PgC yr-1 (true land): ' + '  '.join(f'{y} {C["series"].get(y, np.nan):.0f}/{T["series"].get(y, np.nan):.0f}' for y in sorted(T['series'])))
bands = {'global': ok, '90S-30N': ok & (LAT[:, None] < 30), '30N-60N': ok & (LAT[:, None] >= 30) & (LAT[:, None] < 60), '60N-90N': ok & (LAT[:, None] >= 60)}
print(f"\n{'':10s} {'GPP ctl':>8s} {'test':>6s} {'FLX':>6s} {'x FLX':>11s}   {'NPP ctl/test':>12s}   {'ET x ens':>11s}   {'peak LAI x MODIS':>16s}")
for b, m in bands.items():
    e = m & np.isfinite(etr)
    print(f'{b:10s} {P(C["gpp"], m):8.1f} {P(T["gpp"], m):6.1f} {P(gfx, m):6.1f} {P(C["gpp"], m) / P(gfx, m):5.2f}->{P(T["gpp"], m) / P(gfx, m):4.2f}   '
          f'{P(C["npp"], m):5.1f}/{P(T["npp"], m):5.1f}   {M(C["et"], e) / M(etr, e):5.2f}->{M(T["et"], e) / M(etr, e):4.2f}   '
          f'{M(C["lai"], m) / M(lmod, m):5.2f}->{M(T["lai"], m) / M(lmod, m):4.2f}')
cls = {'forest (tree>=60%)': (cf < .5) & (bare < .5) & (tree >= .6), 'mixed tree (10-60%)': (cf < .5) & (bare < .5) & (tree >= .1) & (tree < .6),
       'grassland (tree<10%)': (cf < .5) & (bare < .5) & (tree < .1) & (shrub < grass), 'shrubland (tree<10%)': (cf < .5) & (bare < .5) & (tree < .1) & (shrub >= grass),
       'bare/sparse (bare>=50%)': (cf < .5) & (bare >= .5), 'cropland (>=50%)': cf >= .5}
ex_c = P(C['gpp'] - gfx, ok); ex_t = P(T['gpp'] - gfx, ok)
print(f"\nby observed land type      GPP x FLX     excess PgC (ctl {ex_c:.0f}, test {ex_t:.0f})   tree cover on the natural share (obs)")
for k, msk in cls.items():
    m = ok & msk; fn = lambda r: r['fu'] / np.maximum(1 - cf, 0.01)
    print(f'{k:26s} {P(C["gpp"], m) / P(gfx, m):5.2f}->{P(T["gpp"], m) / P(gfx, m):4.2f}   {P(C["gpp"] - gfx, m):6.1f} -> {P(T["gpp"] - gfx, m):6.1f}            '
          f'{M(np.clip(fn(C), 0, 1), m):.2f} -> {M(np.clip(fn(T), 0, 1), m):.2f}  ({M(tree, m):.2f})')
nat = ok & (cf < .5) & np.isfinite(tree)
for tag, r in (('control', C), ('test', T)):
    f = np.clip(r['fu'] / np.maximum(1 - cf, 0.01), 0, 1)
    print(f'{tag:8s} natural land: tree cover {M(f, nat):.3f} (obs {M(tree, nat):.3f}), phantom (obs<0.15, model>0.4) {100 * M(((tree < .15) & (f > .4)).astype(float), nat):.1f}%, '
          f'forest lost (obs>0.6, model<0.3) {100 * M(((tree > .6) & (f < .3)).astype(float), nat):.1f}%, r {np.corrcoef(f[nat], tree[nat])[0, 1]:.2f}')
for tag, run in (('control', ctl), ('test', tst)):
    f = f'{run}/out/ibis_yearly_{y1}.nc'
    with xr.open_dataset(f) as d:
        if 'aynppl' in d:
            u = np.nan_to_num(d['aynppu'].values.astype(float)); l = np.nan_to_num(d['aynppl'].values.astype(float))
            print(f'{tag:8s} {y1}: lower-canopy share of NPP {P(np.maximum(l, 0), ok) / max(P(np.maximum(u, 0) + np.maximum(l, 0), ok), 1e-9):.2f} '
                  f'(in observed forest {P(np.maximum(l, 0), ok & cls["forest (tree>=60%)"]) / max(P(np.maximum(u, 0) + np.maximum(l, 0), ok & cls["forest (tree>=60%)"]), 1e-9):.2f})')
