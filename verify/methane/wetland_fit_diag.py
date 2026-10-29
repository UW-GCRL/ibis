"""Feasibility of a wetland area driven by the model's own soil water: per cell, fit f = f_max * sigmoid(k (W_lag - W0)) between
the WAD2M monthly inundated fraction (2001-2010) and the model's top-metre liquid water W (frozen layers 0) from the s31t
hindcast, lag 0-3 months (best per cell). Reports, by latitude band: observed vs climatology vs fitted inundated area, the
seasonal-cycle and interannual skill of the fitted series (in-sample and held-out 2006-2010 from a 2001-2005 fit), against
the two existing modes (climatology; climatology x (1 + beta A)). Read-only.   python -I wetland_fit_diag.py"""
import sys, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R]
from ibis.bgc.wetland_area import root_zone_water
from ibis.io.soil_init import compute_soil_properties
V = '/shared/nas2/Hangkai/ibis/verify_outputs/final_s31t/hindcast/out'; SOIL = '/shared/nas2/Min/HWSD/soita.nc'
WAD = '/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc'
Y0, Y1 = 2001, 2010; NM = (Y1 - Y0 + 1) * 12
m0 = xr.open_dataset(f'{V}/ibis_monthly_{Y0}_01.nc'); mlat = np.asarray(m0['lat'].values, float); mlon = np.asarray(m0['lon'].values, float); nlay = m0.sizes['soilay']
land = np.isfinite(m0['fu'].values); ij = np.argwhere(land); I, J = ij[:, 0], ij[:, 1]; n = len(I)
sd_ = xr.open_dataset(SOIL); sand = np.asarray(sd_['sandpct'].values, float); clay = np.asarray(sd_['claypct'].values, float); slat = np.asarray(sd_['lat'].values, float)
if (slat[0] < slat[-1]) != (mlat[0] < mlat[-1]): sand, clay = sand[:, ::-1], clay[:, ::-1]
sand = np.where(np.isfinite(sand), sand, 45.0); clay = np.where(np.isfinite(clay), clay, 25.0)
props = compute_soil_properties(sand[:, I, J].T, clay[:, I, J].T); poros = np.asarray(props['poros'])[:, :nlay]; hsoi = np.asarray(props['hsoi'])
W = np.zeros((NM, n)); k = 0
for y in range(Y0, Y1 + 1):
    for m in range(12):
        with xr.open_dataset(f'{V}/ibis_monthly_{y}_{m + 1:02d}.nc') as d:
            W[k] = root_zone_water(np.asarray(d['wsoi'].values, float)[I, J], poros, hsoi, np.asarray(d['tsoi'].values, float)[I, J]); k += 1
print('W loaded', W.shape, f'mean {W.mean():.3f} m', flush=True)
wd = xr.open_dataset(WAD); t = wd['time'].dt.year.values; sel = (t >= Y0) & (t <= Y1); Fw = np.asarray(wd['Fw'].values[sel], float)
if float(wd['lat'].values[0]) < float(wd['lat'].values[-1]): Fw = Fw[:, ::-1]
Fw = np.nan_to_num(Fw)[:, I, J]; FMAX = np.nan_to_num(np.nanmax(np.where(np.asarray(wd['lat'].values)[0] < 0, np.asarray(wd['Fw'].values, float)[:, ::-1], np.asarray(wd['Fw'].values, float)), 0))[I, J]
print('WAD2M loaded', Fw.shape, flush=True)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(mlat[I]))   # per land cell; true land fraction is not applied here (same for obs and model)
lat = mlat[I]; BANDS = {'90S-30N': lat < 30, '30-60N': (lat >= 30) & (lat < 60), '60-90N': lat >= 60, 'GLOBAL': np.ones(n, bool)}
wet = FMAX > 0.005
fclim = np.array([Fw[m::12].mean(0) for m in range(12)]); fclim_series = np.tile(fclim, (Y1 - Y0 + 1, 1))
def fit(Wx, F, fmax, train):
    """per cell logit-linear fit of F/fmax on Wx over the train months; returns k, W0 and the fitted series for all months"""
    ff = np.clip(F / np.maximum(fmax, 1e-6), 0.02, 0.98); z = np.log(ff / (1 - ff))
    x = Wx[train]; zz = z[train]; xm = x.mean(0); zm = zz.mean(0); sxx = ((x - xm) ** 2).sum(0); sxz = ((x - xm) * (zz - zm)).sum(0)
    kk = np.where(sxx > 1e-12, sxz / np.maximum(sxx, 1e-12), 0.0); kk = np.clip(kk, 0, 200); b = zm - kk * xm
    pred = fmax / (1 + np.exp(-np.clip(kk * Wx + b, -30, 30)))
    return kk, b, pred
def lagged(W, L):
    out = np.roll(W, L, axis=0)
    if L: out[:L] = W[:L]      # the first months have no earlier data: use the same month
    return out
def r2(F, P, msk):
    ss = ((F[:, msk] - P[:, msk]) ** 2).sum(); st = ((F[:, msk] - F[:, msk].mean(0)) ** 2).sum(); return 1 - ss / max(st, 1e-12)
def band_series(X, msk): return (X[:, msk] * area[msk]).sum(1) / 1e12       # Mkm2
def iav(a, b): a = a.reshape(-1, 12).mean(1); b = b.reshape(-1, 12).mean(1); return float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
def seas(a, b): a = a.reshape(-1, 12).mean(0); b = b.reshape(-1, 12).mean(0); return float(np.corrcoef(a, b)[0, 1]), (a.max() - a.min()), (b.max() - b.min())
months = np.arange(NM); train_all = np.ones(NM, bool); train_h = months < 60
best = {}
for name, train in (('in-sample', train_all), ('held-out 2006-10', train_h)):
    r2best = np.full(n, -np.inf); P = np.zeros_like(Fw); L = np.zeros(n, int)
    for lag in range(4):
        kk, b, pred = fit(lagged(W, lag), Fw, FMAX, train)
        ev = ~train if name.startswith('held') else train_all
        res = ((Fw[ev] - pred[ev]) ** 2).sum(0); tot = ((Fw[ev] - Fw[ev].mean(0)) ** 2).sum(0); r2c = 1 - res / np.maximum(tot, 1e-12)
        better = r2c > r2best; r2best = np.where(better, r2c, r2best); P[:, better] = pred[:, better]; L[better] = lag
    best[name] = (P, L, r2best)
# the existing anomaly mode for comparison: f = fclim (1 + beta A), beta per band from the 2026-10-16 regression
mu = np.array([W[m::12].mean(0) for m in range(12)]); sdv = np.array([W[m::12].std(0) for m in range(12)]); A = (W - np.tile(mu, (10, 1))) / np.maximum(np.tile(sdv, (10, 1)), 1e-9)
beta = np.where(lat < 30, 0.13, np.where(lat < 60, 0.07, 0.06)); Fanom = np.clip(fclim_series * (1 + beta * A), 0, FMAX)
print('\nInundated area by band, 2001-2010 (Mkm2, land-cell area): WAD2M | climatology mode | anomaly mode | sigmoid fit (in-sample) | sigmoid (fit 2001-05, shown 2006-10 vs WAD2M 2006-10)')
print(f"{'band':9s} {'obs':>6s} {'clim':>6s} {'anom':>6s} {'fit':>6s} | seasonal r (clim/anom/fit) amp obs/fit | IAV r of annual area (anom / fit in / fit held-out) | R2 monthly cells (clim / anom / fit / fit held-out) | wet cells R2>0.5 share of area | best lag share 0/1/2/3")
for bn, msk in BANDS.items():
    m2 = msk & wet; o = band_series(Fw, m2); c = band_series(fclim_series, m2); a_ = band_series(Fanom, m2); f = band_series(best['in-sample'][0], m2); fh = band_series(best['held-out 2006-10'][0], m2)
    ev = ~train_h; sr = seas(o, f); 
    r2cells = lambda P, evm=train_all: r2(Fw[evm], P[evm], m2)
    share = (area[m2] * (best['in-sample'][2][m2] > 0.5)).sum() / area[m2].sum(); lags = [(area[m2] * (best['in-sample'][1][m2] == l)).sum() / area[m2].sum() for l in range(4)]
    print(f"{bn:9s} {o.mean():6.2f} {c.mean():6.2f} {a_.mean():6.2f} {f.mean():6.2f} | {seas(o, c)[0]:.2f}/{seas(o, a_)[0]:.2f}/{sr[0]:.2f} amp {sr[1]:.2f}/{sr[2]:.2f} | {iav(o, a_):+.2f} / {iav(o, f):+.2f} / {iav(o[ev], fh[ev]):+.2f} | {r2cells(fclim_series):.2f} / {r2cells(Fanom):.2f} / {r2cells(best['in-sample'][0]):.2f} / {r2(Fw[ev], best['held-out 2006-10'][0][ev], m2):.2f} | {share:.2f} | " + '/'.join(f'{l:.2f}' for l in lags))
print('\nObserved WAD2M interannual range of annual band area (max-min of the 10 annual means, Mkm2):', {bn: round(float(np.ptp(band_series(Fw, msk & wet).reshape(-1, 12).mean(1))), 3) for bn, msk in BANDS.items()})
np.savez_compressed('/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad/wetland/fit_diag.npz', I=I, J=J, lat=lat, FMAX=FMAX, lag=best['in-sample'][1], r2=best['in-sample'][2], W=W.astype(np.float32), Fw=Fw.astype(np.float32))
print('saved fit_diag.npz')
