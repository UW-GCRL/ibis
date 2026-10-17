"""Leaf area per unit GPP, model against observed, per dominant PFT: the leaf-area stage's factor (CALIBRATION.md 48).

    python verify/calibration/diagnostics/leaf_per_gpp.py <run dir> [--years 2006-2010] [--params P.yaml --out OUT.yaml]
                                                          [--route tauleaf:pft,pft] [--skip pft,pft]

k = peak-month cell LAI / annual GPP. Model: fu x laiu + fl x lail (monthly means; the peak of the climatology) and
`aygpptot`; observed: the mean of GLASS V61 and GIMMS LAI4g (GLASS alone where GIMMS is empty) and FLUXCOM RS+METEO.
Cells are assigned to the PFT holding >= 50 % of the cell's leaf area (the run's restarts over the years) and taken
only where the ESA CCI cropland share is < 30 % (crops share the grass PFTs; reported separately). The factor per PFT is
f = sum(LAI_mod A) / sum(GPP_mod A) / (sum(LAI_obs A) / sum(GPP_obs A)) over its cells. With --params/--out, writes a
copy of the parameter file with specla / f for every PFT with >= 200 cells (others unchanged), f clipped to [0.5, 2];
--route tauleaf:... applies f to the leaf residence time instead (tauleaf / f: the same equilibrium leaf area, for PFTs whose
specla would leave its trait range), --skip leaves PFTs unchanged.
"""
import sys, dataclasses, numpy as np, xarray as xr
sys.path.insert(0, 'verify/calibration'); sys.path.insert(0, 'python-port')
from landmask import true_landfrac
from ibis.state.vegetation import VegetationState
a = sys.argv[1:]; run = a[0]
opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
y0, y1 = (int(v) for v in opt('--years', '2006-2010').split('-')); YR = list(range(y0, y1 + 1))
NAMES = ['tropical_broadleaf_evergreen', 'tropical_broadleaf_deciduous', 'warm_temperate_broadleaf_evergreen', 'temperate_conifer_evergreen',
         'temperate_broadleaf_deciduous', 'boreal_conifer_evergreen', 'boreal_broadleaf_deciduous', 'boreal_conifer_deciduous',
         'evergreen_shrub', 'deciduous_shrub', 'c4_grass', 'c3_grass']
B = '/shared/nas2/Hangkai/ibis/inputs/eval/lai'; yi = [y - 2001 for y in YR]
G = np.nanmean(np.load(f'{B}/glass_v61_0.5deg_monthly_2001-2010.npz')['lai'][yi], 0)
M = np.nanmean(np.load(f'{B}/gimms_lai4g_0.5deg_monthly_2001-2010.npz')['lai'][yi], 0)
lai_o = np.nanmax(np.where(np.isfinite(M), 0.5 * (G + M), G), 0)
LAT = np.arange(89.75, -90, -0.5); lf = true_landfrac(); A = np.cos(np.deg2rad(LAT))[:, None] * np.ones((360, 720)) * lf
rd = lambda f, v: np.nan_to_num(np.asarray(xr.open_dataset(f)[v].values, float))
mon = np.zeros((12, 360, 720))
for y in YR:
    for k in range(12):
        f = f'{run}/out/ibis_monthly_{y}_{k + 1:02d}.nc'; mon[k] += rd(f, 'fu') * rd(f, 'laiu') + rd(f, 'fl') * rd(f, 'lail')
lai_m = (mon / len(YR)).max(0)
gpp_m = np.mean([rd(f'{run}/out/ibis_yearly_{y}.nc', 'aygpptot') for y in YR], 0)
with xr.open_dataset('/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc') as fc:
    ys = np.array([int(str(t)[:4]) for t in fc['time'].values])
    gpp_o = np.nanmean(fc['gpp'].values[(ys >= y0) & (ys <= y1)].astype(float), 0) * 365.25 / 1000
# dominant PFT from the run's restarts
with xr.open_dataset(f'{run}/out/ibis_monthly_{y0}_07.nc') as o: ij = np.argwhere(np.isfinite(o['fu'].values))
vf = [f.name for f in dataclasses.fields(VegetationState)]; pl = []
for y in YR:
    r = np.load(f'{run}/restart/ibis_restart_{y}.npz'); tr = str(r['treedef_repr']); vo = tr[:tr.find('VegetationState')].count('*')
    pl.append(r[f'leaf_{vo + vf.index("plai")}'])
pl = np.mean(pl, 0); share = pl / np.maximum(pl.sum(1, keepdims=True), 1e-9)
dom = np.full((360, 720), -1); top = share.argmax(1); dom[ij[:, 0], ij[:, 1]] = np.where(share.max(1) >= 0.5, top, -1)
cf = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz')['cropfrac'].clip(0, 1)
ok = (lf > 0.05) & (LAT[:, None] > -60) & np.isfinite(lai_o) & np.isfinite(gpp_o) & (gpp_o > 0.05) & (lai_o > 0.1) & (gpp_m > 0)
S = lambda x, m: float((x * A)[m].sum())
fac = {}
print(f'{run}, {y0}-{y1}: leaf area (peak) per unit GPP, model / observed, by dominant PFT (>= 50 % of cell LAI), cropland < 30 %\n')
print(f"{'PFT':36s} {'cells':>6s} {'LAI mod/obs':>11s} {'GPP mod/obs':>11s} {'k mod/obs':>9s} {'cell median':>11s}")
for j, n in enumerate(NAMES):
    m = ok & (dom == j) & (cf < 0.3)
    if m.sum() < 30: print(f'{n:36s} {int(m.sum()):6d}   (too few cells)'); continue
    rl = S(lai_m, m) / S(lai_o, m); rg = S(gpp_m, m) / S(gpp_o, m); f = rl / rg
    med = float(np.median((lai_m / gpp_m)[m] / (lai_o / gpp_o)[m]))
    fac[n] = (f, int(m.sum())); print(f'{n:36s} {int(m.sum()):6d} {rl:11.2f} {rg:11.2f} {f:9.2f} {med:11.2f}')
mc = ok & (cf >= 0.5)
print(f"{'cropland (>= 50 %), all PFTs':36s} {int(mc.sum()):6d} {S(lai_m, mc) / S(lai_o, mc):11.2f} {S(gpp_m, mc) / S(gpp_o, mc):11.2f} {S(lai_m, mc) / S(lai_o, mc) / (S(gpp_m, mc) / S(gpp_o, mc)):9.2f}")
mall = ok & (cf < 0.3)
print(f"{'all natural cells':36s} {int(mall.sum()):6d} {S(lai_m, mall) / S(lai_o, mall):11.2f} {S(gpp_m, mall) / S(gpp_o, mall):11.2f} {S(lai_m, mall) / S(lai_o, mall) / (S(gpp_m, mall) / S(gpp_o, mall)):9.2f}")
if opt('--params') and opt('--out'):
    import yaml
    d = yaml.safe_load(open(opt('--params'))); changes = []
    via_tau = set(opt('--route', 'tauleaf:').split(':', 1)[1].split(',')) - {''}; skip = set(opt('--skip', '').split(',')) - {''}
    for n, (f, nc) in fac.items():
        if nc < 200 or n in skip: continue
        f = float(np.clip(f, 0.5, 2.0)); key = 'tauleaf' if n in via_tau else 'specla'
        old = float(d['pft'][n][key]); d['pft'][n][key] = old / f; changes.append(f'{n}: {key} {old:.3f} -> {old / f:.3f} (factor {f:.2f})')
    with open(opt('--out'), 'w') as fo:
        fo.write(f'# {opt("--params").split("/")[-1]} with the leaf area per unit NPP divided by (model / observed leaf area per unit GPP) '
                 f'from {run} {y0}-{y1}: specla, or tauleaf for {sorted(via_tau) or "none"} (CALIBRATION.md 48; diagnostics/leaf_per_gpp.py)\n')
        yaml.safe_dump(d, fo, sort_keys=False)
    print('\nwrote', opt('--out')); print('\n'.join('  ' + c for c in changes))
