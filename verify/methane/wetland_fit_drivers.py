"""Second feasibility pass: which model variable carries the WAD2M inundation signal? Drivers per cell: W (top-metre liquid
water), 3-month running means of total runoff (surface + drainage) and of precipitation, each at lags 0-3; the same per-cell
sigmoid fit and band metrics as wetland_fit_diag.py; plus the band-level anomaly mode (f_clim x (1 + beta A)) with beta
fitted on the band's annual area. Uses fit_diag.npz (W, Fw, cells) and re-reads runoff and precipitation."""
import sys, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad/wetland'; V = '/shared/nas2/Hangkai/ibis/verify_outputs/final_s31t/hindcast/out'
z = np.load(f'{S}/fit_diag.npz'); I, J, lat, FMAX, W, Fw = z['I'], z['J'], z['lat'], z['FMAX'], z['W'].astype(float), z['Fw'].astype(float); n = len(I); NM = 120
RO = np.zeros((NM, n)); PR = np.zeros((NM, n)); k = 0
for y in range(2001, 2011):
    for m in range(12):
        with xr.open_dataset(f'{V}/ibis_monthly_{y}_{m + 1:02d}.nc') as d:
            RO[k] = np.nan_to_num(np.asarray(d['amtrunof'].values, float)[I, J]) + np.nan_to_num(np.asarray(d['amtdrain'].values, float)[I, J]); PR[k] = np.nan_to_num(np.asarray(d['amprcp'].values, float)[I, J]); k += 1
def rm3(X):
    out = X.copy()
    for t in range(NM): out[t] = X[max(0, t - 2):t + 1].mean(0)
    return out
DR = {'W (top metre)': W, 'runoff 3-mo': rm3(RO), 'precip 3-mo': rm3(PR), 'runoff 3-mo + W': None}
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)); wet = FMAX > 0.005
BANDS = {'90S-30N': lat < 30, '30-60N': (lat >= 30) & (lat < 60), '60-90N': lat >= 60, 'GLOBAL': np.ones(n, bool)}
fclim = np.array([Fw[m::12].mean(0) for m in range(12)]); fcs = np.tile(fclim, (10, 1)); months = np.arange(NM); tr = months < 60
def fit(Xx, train):
    ff = np.clip(Fw / np.maximum(FMAX, 1e-6), 0.02, 0.98); zz = np.log(ff / (1 - ff)); x = Xx[train]; zt = zz[train]; xm = x.mean(0); zm = zt.mean(0)
    sxx = ((x - xm) ** 2).sum(0); kk = np.clip(np.where(sxx > 1e-12, ((x - xm) * (zt - zm)).sum(0) / np.maximum(sxx, 1e-12), 0.0), 0, 1e4); b = zm - kk * xm
    return FMAX / (1 + np.exp(-np.clip(kk * Xx + b, -30, 30)))
def lagged(X, L): o = np.roll(X, L, axis=0); o[:L] = X[:L] if L else o[:L]; return o
def band_series(X, msk): return (X[:, msk] * area[msk]).sum(1) / 1e12
def iav(a, b): a = a.reshape(-1, 12).mean(1); b = b.reshape(-1, 12).mean(1); return float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
def seas_amp(a): a = a.reshape(-1, 12).mean(0); return a.max() - a.min()
def r2cells(P, msk, ev): ss = ((Fw[ev][:, msk] - P[ev][:, msk]) ** 2).sum(); st = ((Fw[ev][:, msk] - Fw[ev][:, msk].mean(0)) ** 2).sum(); return 1 - ss / max(st, 1e-12)
print('Per-cell sigmoid fit of the WAD2M monthly fraction on a model driver (best lag 0-3 per cell). Columns per band: monthly R2 over wet cells (in-sample | held-out 2006-10 from a 2001-05 fit), IAV r of annual band area (in | held-out), seasonal amplitude fit/obs')
print(f"{'driver':18s} " + ' '.join(f'{b:>42s}' for b in BANDS))
for name, X in DR.items():
    if X is None:  # two-driver linear combination: fit on standardised runoff and W jointly (lag 0)
        a1 = (rm3(RO) - rm3(RO).mean(0)) / np.maximum(rm3(RO).std(0), 1e-9); a2 = (W - W.mean(0)) / np.maximum(W.std(0), 1e-9); X2 = a1 + a2
        rows = []
        for ev_name, train in (('in', np.ones(NM, bool)), ('held', tr)):
            P = fit(X2, train); rows.append((P, train))
    else:
        rows = []
        for ev_name, train in (('in', np.ones(NM, bool)), ('held', tr)):
            best = np.full(n, -np.inf); P = np.zeros_like(Fw)
            for L in range(4):
                p = fit(lagged(X, L), train); ev = ~train if ev_name == 'held' else np.ones(NM, bool)
                r = 1 - ((Fw[ev] - p[ev]) ** 2).sum(0) / np.maximum(((Fw[ev] - Fw[ev].mean(0)) ** 2).sum(0), 1e-12); better = r > best; best = np.where(better, r, best); P[:, better] = p[:, better]
            rows.append((P, train))
    (Pin, _), (Ph, trh) = rows; ev = ~trh; line = f'{name:18s} '
    for bn, msk in BANDS.items():
        m2 = msk & wet; o = band_series(Fw, m2); fi = band_series(Pin, m2); fh = band_series(Ph, m2)
        line += f' R2 {r2cells(Pin, m2, np.ones(NM, bool)):.2f}|{r2cells(Ph, m2, ev):.2f} IAV {iav(o, fi):+.2f}|{iav(o[ev], fh[ev]):+.2f} amp {seas_amp(fi):.2f}/{seas_amp(o):.2f}'
    print(line)
print(f"{'climatology':18s} " + ' '.join(f' R2 {r2cells(fcs, msk & wet, np.ones(NM, bool)):.2f}|{r2cells(fcs, msk & wet, ~tr):.2f} IAV  nan|  nan amp {seas_amp(band_series(fcs, msk & wet)):.2f}/{seas_amp(band_series(Fw, msk & wet)):.2f}' for bn, msk in BANDS.items()))
print('\nBand-level anomaly mode f = f_clim (1 + beta A_driver), beta fitted on the band annual-area anomaly (in-sample), IAV r and the beta:')
for name, X in DR.items():
    if X is None: continue
    line = f'{name:18s} '
    for bn, msk in BANDS.items():
        m2 = msk & wet; o = band_series(Fw, m2).reshape(-1, 12).mean(1); oa = o - o.mean()
        mu = np.array([X[m::12].mean(0) for m in range(12)]); sd = np.array([X[m::12].std(0) for m in range(12)]); A = (X - np.tile(mu, (10, 1))) / np.maximum(np.tile(sd, (10, 1)), 1e-9)
        dA = band_series(fcs * A, m2).reshape(-1, 12).mean(1); dA = dA - dA.mean()           # area response per unit beta
        beta = float((dA * oa).sum() / max((dA ** 2).sum(), 1e-12)); r = float(np.corrcoef(dA, oa)[0, 1]) if dA.std() > 0 else float('nan')
        line += f' IAV r {r:+.2f} beta {beta:+.3f} (obs range {np.ptp(o):.3f} Mkm2)   '
    print(line)
