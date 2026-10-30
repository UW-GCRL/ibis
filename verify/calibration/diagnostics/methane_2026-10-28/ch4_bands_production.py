"""Wetland CH4 of a production hindcast, decomposed by GMB band, GCP region and wetland type, against the Global Methane
Budget 2025 bottom-up wetland ensemble (Saunois et al. 2025; files under inputs/eval/mip/gmb2024).

Per month and cell, from the hindcast's own monthly output (amch4, amco2mic, tsoi, wsoi) and the WAD2M climatology:
    F = f_inund x Rh x f_anaer(zwt) x r_ch4 x Q10^((T_thawed - T_ref)/10) x unfrozen x f_surv(zwt)
with the band parameters the run used (params yaml: methane + methane_bands, band = wetland type W01/W02/W03 of the
Köppen zone, latitude rule where the zone has no wetland type). The water table is recomputed from the monthly liquid
moisture (no ice in the monthly files: exact in thawed months, which carry the flux); the reconstruction is checked
against the model's amch4 per band. Read-only.
    python -I ch4_bands_production.py RUN_HINDCAST_DIR [--years 2001-2010] [--params params.yaml]"""
import sys, re, ast, glob, numpy as np, xarray as xr, yaml, pandas as pd, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
from ibis.io.pft_types import ZONES, wetland_band
from ibis.io.wad2m import read_wad2m_climatology
S = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work'; M = f'{S}/meeting'
G = '/shared/nas2/Hangkai/ibis/inputs/eval/mip/gmb2024'; WAD = '/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc'
args = [a for a in sys.argv[1:] if not a.startswith('--')]; run = args[0].rstrip('/')
yrs = next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('--years=')), '2001-2010'); y0, y1 = map(int, yrs.split('-'))
pfile = next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('--params=')), None)
if pfile is None: pfile = yaml.safe_load(open(f'{run}/cfg.yaml'))['params_file']
P = yaml.safe_load(open(pfile)); base = P.get('methane', {}); bands_p = P.get('methane_bands', {})
DEF = dict(r_ch4=0.23, q10p=1.60, t_ref=295.15, z_tau=0.50, tau_oxid=0.10, z_wt_max=0.30, z_wt_ref=0.50, sat_crit=None)
def par(k, band):
    v = bands_p.get(band, {}).get(k, base.get(k, DEF[k])); return float(v) if v is not None else None
NDAY = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]); KGC_TG = 16.043 / 12.011 * 1e-9
# --- grid, area, zones, bands
d0 = xr.open_dataset(sorted(glob.glob(f'{run}/out/ibis_monthly_{y0}_01.nc'))[0]); lat = d0['lat'].values; lon = d0['lon'].values
LAT, LON = np.meshgrid(lat, lon, indexing='ij'); lf = true_landfrac(); gi = np.rint((89.75 - LAT) / 0.5).astype(int); gj = np.rint((LON + 179.75) / 0.5).astype(int)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(LAT)) * np.nan_to_num(lf[gi, gj])          # m2 of land
ref = np.load(f'{M}/glob_refs.npz'); zone = ref['zone'][gi, gj]; bi = wetland_band(np.clip(zone, 0, len(ZONES) - 1)); bi = np.where(zone < 0, -1, bi)
trop = (LAT > -30) & (LAT < 30); boreal = LAT >= 60
bidx = np.where(bi >= 0, bi, np.where(trop, 0, np.where(boreal, 2, 1)))                                   # 0 tropical 1 temperate 2 boreal
BN = ('tropical', 'temperate', 'boreal'); pc = {k: np.choose(bidx, [par(k, b) for b in BN]) for k in ('r_ch4', 'q10p', 'z_tau', 'tau_oxid')}
t_ref = float(base.get('t_ref', DEF['t_ref'])); z_wt_max = float(base.get('z_wt_max', DEF['z_wt_max'])); z_wt_ref = float(base.get('z_wt_ref', DEF['z_wt_ref']))
# sat_crit: the module's default lives in MethaneParams; read it
from ibis.bgc.methane import MethaneParams; mp = MethaneParams(); sat_crit = float(base.get('sat_crit', mp.sat_crit))
# --- WAD2M climatology on the output grid
mlat = np.arange(-90.0, 90.0, 0.5); mlon = np.arange(-180.0, 180.0, 0.5)
wli = np.floor((LAT.ravel() - mlat[0]) / 0.5 + 1e-9).astype(int).clip(0, 359); wlo = np.floor((LON.ravel() - mlon[0]) / 0.5 + 1e-9).astype(int).clip(0, 719)
clim = np.asarray(read_wad2m_climatology(WAD, wli, wlo, mlat, mlon)).reshape(12, *LAT.shape); clim = np.nan_to_num(clim)
# --- soil layers
st = eg._load(f'{run}/restart/ibis_restart_{y1}.npz'); hsoi = np.asarray(st.soil.hsoi, float).ravel()[:6]
bot = np.cumsum(hsoi); top = np.concatenate([[0.0], bot[:-1]])
def zwt_from(wsoi):                                   # wetland_water_table with wisoi = 0 (monthly files carry liquid moisture only)
    theta = np.clip(wsoi, 0, 1); w = np.clip(z_wt_ref - top, 0, hsoi); th = (theta * w[None, None, :]).sum(-1) / max(w.sum(), 1e-12)
    return z_wt_max * (1.0 - np.clip(th / max(sat_crit, 1e-6), 0, 1))
def t_thawed(tsoi, zwt):                              # thawed_anaerobic_zone: thickness-weighted T of the thawed soil below the water table
    frac = np.clip((bot[None, None, :] - zwt[..., None]) / hsoi[None, None, :], 0, 1) * hsoi[None, None, :]
    th = frac * (tsoi >= 273.15); tt = th.sum(-1)
    return np.where(tt > 1e-12, (th * tsoi).sum(-1) / np.maximum(tt, 1e-12), tsoi[..., 0])
# --- accumulate
keys = ('w', 'rh', 'fa', 'q10', 'unf', 'fs', 'zwt', 'flux_model', 'flux_rec')
acc = {k: np.zeros(LAT.shape) for k in keys}; mon_model = np.zeros((12, *LAT.shape)); ny = 0
for y in range(y0, y1 + 1):
    for m in range(12):
        f = f'{run}/out/ibis_monthly_{y}_{m + 1:02d}.nc'
        if not glob.glob(f): continue
        d = xr.open_dataset(f); ch4 = np.nan_to_num(d['amch4'].values) * NDAY[m]; rh = np.nan_to_num(d['amco2mic'].values) / NDAY[m]   # kgC/m2/yr-cell ; kgC/m2/d
        tsoi = np.nan_to_num(d['tsoi'].values, nan=273.0); wsoi = np.nan_to_num(d['wsoi'].values)
        zwt = zwt_from(wsoi); tt = t_thawed(tsoi, zwt); fa = np.exp(-zwt / pc['z_tau']); fs = np.exp(-zwt / pc['tau_oxid'])
        q10 = pc['q10p'] ** ((tt - t_ref) / 10.0); unf = (tt >= 273.15).astype(float); fin = clim[m]
        wgt = fin * area * NDAY[m]                                                                              # inundated area x days
        rec = fin * np.maximum(rh, 0) * fa * pc['r_ch4'] * q10 * unf * fs * NDAY[m]
        for k, v in (('w', wgt), ('rh', rh * wgt), ('fa', fa * wgt), ('q10', q10 * wgt), ('unf', unf * wgt), ('fs', fs * wgt), ('zwt', zwt * wgt)): acc[k] += v
        acc['flux_model'] += ch4 * area; acc['flux_rec'] += rec * area; mon_model[m] += ch4 * area
    ny += 1
for k in keys: acc[k] /= ny
mon_model /= ny
# --- masks
GB = {'90S-30N': LAT < 30, '30N-60N': (LAT >= 30) & (LAT < 60), '60N-90N': LAT >= 60}
SUB = {'30S-30N': (LAT > -30) & (LAT < 30), '30-45N': (LAT >= 30) & (LAT < 45), '45-60N': (LAT >= 45) & (LAT < 60), '60-70N': (LAT >= 60) & (LAT < 70), '>70N': LAT >= 70, '<30S': LAT <= -30}
WT = {'W01 tropical type': bidx == 0, 'W02 temperate type': bidx == 1, 'W03 boreal type': bidx == 2}
rm = xr.open_dataset(f'{G}/gridded/GCP2023_regions_mask.nc'); regs = rm['regions'].values; names = [str(n) for n in rm['regions_name'].values]
ri = np.clip(np.floor(LAT + 90).astype(int), 0, 179); rj = np.clip(np.floor(LON + 180).astype(int), 0, 359); reg05 = regs[ri, rj]
def gmb(period):
    E = pd.read_csv(f'{G}/GMB2024_BU_wetlands_{period}_ensemble_stats.csv', index_col=0); return E
E09, E19 = gmb('2000-2009'), gmb('2010-2019')
def col(E, name):
    for c in E.columns:
        if c.replace(' ', '_').lower() == name.replace(' ', '_').lower(): return c
    return None
W = acc['w'] / 365.25                                                                                           # mean inundated area, m2
def row(name, m, gkey):
    wet = W[m].sum() / 1e12; fm = acc['flux_model'][m].sum() * KGC_TG; fr = acc['flux_rec'][m].sum() * KGC_TG
    w = max(acc['w'][m].sum(), 1e-9); fac = {k: acc[k][m].sum() / w for k in ('rh', 'fa', 'q10', 'unf', 'fs', 'zwt')}
    c = col(E09, gkey) if gkey else None
    g09 = f"{E09.loc['mean', c]:6.1f} [{E09.loc['min', c]:5.1f}-{E09.loc['max', c]:5.1f}]" if c else ' ' * 20
    g19 = f"{E19.loc['mean', c]:6.1f}" if c else ' ' * 6
    ratio = f"{fm / E09.loc['mean', c]:5.2f}" if c and E09.loc['mean', c] > 0.5 else '   --'
    per_m = fm * 1e12 / max(wet * 1e12, 1e-9) if wet > 0.005 else float('nan'); per_g = (E09.loc['mean', c] * 1e12 / max(wet * 1e12, 1e-9)) if (c and wet > 0.005) else float('nan')
    return (f"{name:28s} {wet:6.2f} | {fm:6.1f} {fr:6.1f} | {g09} {g19} {ratio} | {per_m:6.1f} {per_g:6.1f} | "
            f"{fac['rh'] * 1e3:6.2f} {fac['fa']:5.2f} {fac['q10']:5.2f} {fac['unf']:5.2f} {fac['fs']:5.2f} {fac['zwt']:5.2f}")
hdr = (f"{'group':28s} {'wetMkm2':>6s} | {'model':>6s} {'recon':>6s} | {'GMB 2000-09 mean [min-max]':26s} {'10-19':>6s} {'x':>5s} | {'gCH4/m2/yr: mod':>15s} {'GMB':>6s} | "
       f"{'Rh gC/m2/d':>6s} {'f_an':>5s} {'Q10f':>5s} {'unfr':>5s} {'f_sv':>5s} {'zwt m':>5s}")
print(f'{run}: wetland CH4 {y0}-{y1} (Tg CH4 yr-1); params {pfile}')
print(f"  band parameters (r_ch4, q10p, z_tau, tau_oxid): " + '; '.join(f"{b} {par('r_ch4', b):.3f} {par('q10p', b):.2f} {par('z_tau', b):.2f} {par('tau_oxid', b):.3f}" for b in BN) + f"; t_ref {t_ref:.2f} K, z_wt_max {z_wt_max}, sat_crit {sat_crit}")
print(f"  cells by band rule: W01 {int((bidx == 0).sum())} W02 {int((bidx == 1).sum())} W03 {int((bidx == 2).sum())} (latitude rule on {int((bi < 0).sum())} cells)")
print('  wetMkm2 = mean inundated area (WAD2M climatology on land); recon = the factor product (checks the decomposition); factors are inundated-area-weighted means')
print(hdr)
land = area > 0
for n, m in GB.items(): print(row(n, m & land, n))
print(row('GLOBAL', land, 'Global'))
print('--- sub-bands'); [print(row(n, m & land, None)) for n, m in SUB.items()]
print('--- wetland type (parameter band) within 30N-60N'); [print(row(n, m & GB['30N-60N'] & land, None)) for n, m in WT.items()]
print('--- GCP regions (GMB regional ensemble)')
for k, n in enumerate(names):
    m = (reg05 == k) & land
    if m.sum() == 0 or n.lower().startswith('ocean'): continue
    print(row(n, m, n))
# --- seasonal cycle by band: model vs the GCP prior wetland climatology
pr = xr.open_dataset(f'{G}/gridded/GCP_Prior_CH4_fluxes.nc', decode_times=False); fw = pr['flux_ch4_wetlands'].values   # kg/m2/s (12, 180, 360)
plat = pr['lat'].values; plon = pr['lon'].values; PL, _ = np.meshgrid(plat, plon, indexing='ij'); a1 = (np.deg2rad(1.0) * 6.371e6) ** 2 * np.cos(np.deg2rad(PL))
print('--- seasonal cycle, Tg CH4 per month (model | GCP prior climatology), peak month and max/min ratio')
for n, m in GB.items():
    mm = m & land; mo = np.array([mon_model[k][mm].sum() * KGC_TG for k in range(12)])
    pm = {'90S-30N': PL < 30, '30N-60N': (PL >= 30) & (PL < 60), '60N-90N': PL >= 60}[n]
    gp = np.array([np.nansum(fw[k][pm] * a1[pm]) * 86400 * NDAY[k] * 1e-9 for k in range(12)])
    print(f"  {n:8s} model " + ' '.join(f'{v:5.1f}' for v in mo) + f" | peak {int(mo.argmax()) + 1:2d} max/min {mo.max() / max(mo.min(), 1e-3):5.1f}")
    print(f"  {'':8s} prior " + ' '.join(f'{v:5.1f}' for v in gp) + f" | peak {int(gp.argmax()) + 1:2d} max/min {gp.max() / max(gp.min(), 1e-3):5.1f}  (prior total {gp.sum():.0f})")
