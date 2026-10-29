"""Can a constrained tree ensemble predict the WAD2M monthly inundated fraction from the model's own state and static
topography? (Hangkai's idea, 2026-10-29.) Features: the s31t hindcast's monthly soil water by layer, top-metre liquid water
and its lags, soil and ground temperature, frozen flag, precipitation / runoff / drainage / ET with lags and 3-month means,
5x5-cell neighbourhood means of 3-month precipitation and runoff (the stand-in for upstream supply), LAI; static CLM
topography (FMAX, slope, elevation spread, elevation, percent wetland and lake, VIC b), sand and clay, latitude, and the
cell's WAD2M range (fmin, fmax; variant B leaves these out). No month-of-year, no WAD2M climatology. One gradient-boosted
model per wetland band (the methane bands), monotone increasing in the water variables, output clipped to the cell's range.
Held out: (A) odd years trained -> even years predicted and the reverse (out-of-fold series for all 120 months);
(B) a random half of the cells (all years) -> the other half. Scored against the climatology and the sigmoid of the first
pass. Output: tables + rf_oof.npz (out-of-fold monthly fraction).     python -I wetland_ml_feasibility.py [--quick]"""
import sys, time, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingRegressor
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R]
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'; V = '/shared/nas2/Hangkai/ibis/verify_outputs/final_s31t/hindcast/out'
QUICK = '--quick' in sys.argv; t0 = time.time()
z = np.load(f'{S}/wetland/fit_diag.npz'); I, J, lat, FMAXW, W, Fw = z['I'], z['J'], z['lat'], z['FMAX'], z['W'].astype(np.float32), z['Fw'].astype(np.float32); n = len(I); NM = 120
ref = np.load(f'{S}/meeting/glob_refs.npz'); zone = ref['zone'][I, J]
from ibis.io.pft_types import wetland_band
band = np.asarray(wetland_band(zone)); band = np.where(band >= 0, band, np.where(np.abs(lat) < 30, 0, np.where(lat >= 60, 2, 1)))   # 0 tropical/W01, 1 temperate/W02, 2 boreal/W03
# --- monthly fields
names = ['wsoi1', 'wsoi2', 'wsoi3', 'wsoi4', 'wsoi5', 'wsoi6', 'tsoi1', 'tsoi14', 'tg', 'prcp', 'aet', 'runoff', 'lai']
D = np.zeros((NM, n, len(names)), np.float32); grid_pr = np.zeros((NM, 360, 720), np.float32); grid_ro = np.zeros((NM, 360, 720), np.float32); k = 0
for y in range(2001, 2011):
    for m in range(12):
        with xr.open_dataset(f'{V}/ibis_monthly_{y}_{m + 1:02d}.nc') as d:
            ws = np.asarray(d['wsoi'].values, np.float32)[I, J]; ts = np.asarray(d['tsoi'].values, np.float32)[I, J]
            pr = np.nan_to_num(np.asarray(d['amprcp'].values, np.float32)); ro = np.nan_to_num(np.asarray(d['amtrunof'].values, np.float32)) + np.nan_to_num(np.asarray(d['amtdrain'].values, np.float32))
            D[k, :, 0:6] = ws; D[k, :, 6] = ts[:, 0]; D[k, :, 7] = ts[:, :4].mean(1); D[k, :, 8] = np.nan_to_num(np.asarray(d['tg'].values, np.float32)[I, J])
            D[k, :, 9] = pr[I, J]; D[k, :, 10] = np.nan_to_num(np.asarray(d['amaet'].values, np.float32)[I, J]); D[k, :, 11] = ro[I, J]
            D[k, :, 12] = np.nan_to_num(np.asarray(d['laiu'].values, np.float32)[I, J] * np.asarray(d['fu'].values, np.float32)[I, J]) + np.nan_to_num(np.asarray(d['lail'].values, np.float32)[I, J] * np.asarray(d['fl'].values, np.float32)[I, J])
            grid_pr[k] = pr; grid_ro[k] = ro
        k += 1
print(f'monthly fields loaded {time.time() - t0:.0f}s', flush=True)
land = np.zeros((360, 720), np.float32); land[I, J] = 1; lf = uniform_filter(land, 5, mode='wrap')
def nb(g): return uniform_filter(g, 5, mode='wrap') / np.maximum(lf, 1e-6)                 # land-only 5x5 mean
def rm3(X):
    o = X.copy()
    for t in range(NM): o[t] = X[max(0, t - 2):t + 1].mean(0)
    return o
def lagged(X, L): o = np.roll(X, L, axis=0); o[:L] = X[:L]; return o
pr3 = rm3(D[:, :, 9]); ro3 = rm3(D[:, :, 11]); W3 = rm3(W)
nbp = np.stack([nb(g)[I, J] for g in rm3(grid_pr)]); nbr = np.stack([nb(g)[I, J] for g in rm3(grid_ro)])
dyn = {**{nm: D[:, :, i] for i, nm in enumerate(names)}, 'W': W, 'W_l1': lagged(W, 1), 'W_l2': lagged(W, 2), 'W_l3': lagged(W, 3), 'W_3mo': W3,
       'prcp_3mo': pr3, 'prcp_3mo_l1': lagged(pr3, 1), 'prcp_3mo_l3': lagged(pr3, 3), 'runoff_3mo': ro3, 'runoff_3mo_l1': lagged(ro3, 1), 'runoff_3mo_l3': lagged(ro3, 3),
       'nb_prcp_3mo': nbp, 'nb_prcp_3mo_l1': lagged(nbp, 1), 'nb_prcp_3mo_l2': lagged(nbp, 2), 'nb_runoff_3mo': nbr, 'nb_runoff_3mo_l1': lagged(nbr, 1), 'nb_runoff_3mo_l2': lagged(nbr, 2),
       'frozen': (D[:, :, 6] <= 273.16).astype(np.float32)}
# --- static fields
sd = xr.open_dataset('/shared/nas2/Hangkai/ELM Edge Effects/inputdata/surfdata_0.5x0.5_CH_v1_all_forests.nc'); flip = float(sd['LATIXY'].values[0, 0]) < float(sd['LATIXY'].values[-1, 0])
def st(v): a = np.asarray(sd[v].values, np.float32); a = a[::-1] if flip else a; return np.nan_to_num(a[I, J])
soil = xr.open_dataset('/shared/nas2/Min/HWSD/soita.nc'); sl = np.asarray(soil['lat'].values, float); sflip = sl[0] < sl[-1]
def so(v): a = np.asarray(soil[v].values, np.float32); a = a[0] if a.ndim == 3 else a; a = a[::-1] if sflip else a; return np.nan_to_num(a[I, J], nan=35.0)
wd = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc'); FW = np.asarray(wd['Fw'].values, np.float32); FW = FW[:, ::-1] if float(wd['lat'].values[0]) < float(wd['lat'].values[-1]) else FW
fmin_w = np.nan_to_num(np.nanmin(FW, 0))[I, J]; fmax_w = np.nan_to_num(np.nanmax(FW, 0))[I, J]
static = {'FMAX_clm': st('FMAX'), 'SLOPE': st('SLOPE'), 'STD_ELEV': st('STD_ELEV'), 'TOPO': st('TOPO'), 'PCT_WETLAND': st('PCT_WETLAND'), 'PCT_LAKE': st('PCT_LAKE'), 'binfl': st('binfl'),
          'sand': so('sandpct'), 'clay': so('claypct'), 'lat': lat.astype(np.float32), 'fmin_w': fmin_w, 'fmax_w': fmax_w}
print(f'features built {time.time() - t0:.0f}s', flush=True)
wet = fmax_w > 0.005; cells = np.where(wet)[0]
if QUICK: cells = cells[::4]
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat))
MONO_POS = {'wsoi1', 'wsoi2', 'wsoi3', 'wsoi4', 'wsoi5', 'wsoi6', 'W', 'W_l1', 'W_l2', 'W_l3', 'W_3mo', 'prcp', 'prcp_3mo', 'prcp_3mo_l1', 'prcp_3mo_l3', 'runoff', 'runoff_3mo', 'runoff_3mo_l1', 'runoff_3mo_l3', 'nb_prcp_3mo', 'nb_prcp_3mo_l1', 'nb_prcp_3mo_l2', 'nb_runoff_3mo', 'nb_runoff_3mo_l1', 'nb_runoff_3mo_l2'}
def table(variant):
    feats = list(dyn) + [k for k in static if variant == 'A' or k not in ('fmin_w', 'fmax_w')]
    X = np.empty((NM * len(cells), len(feats)), np.float32)
    for c, f in enumerate(feats):
        X[:, c] = (dyn[f][:, cells] if f in dyn else np.broadcast_to(static[f][cells], (NM, len(cells)))).reshape(-1)
    y = Fw[:, cells].reshape(-1); months = np.repeat(np.arange(NM), len(cells)); cid = np.tile(np.arange(len(cells)), NM); bd = band[cells][cid]
    mono = [1 if f in MONO_POS else 0 for f in feats]
    return feats, X, y, months, cid, bd, mono
def fit_predict(X, y, tr, te, mono):
    m = HistGradientBoostingRegressor(max_iter=150 if QUICK else 400, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0, monotonic_cst=mono, random_state=0)
    m.fit(X[tr], y[tr]); return m.predict(X[te]), m
def band_series(F, sel): return (F[:, sel] * area[cells][sel]).sum(1) / 1e12
def iav(a, b): a = a.reshape(-1, 12).mean(1); b = b.reshape(-1, 12).mean(1); return float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
def seas(a, b): a = a.reshape(-1, 12).mean(0); b = b.reshape(-1, 12).mean(0); return float(np.corrcoef(a, b)[0, 1]), float(np.ptp(a)), float(np.ptp(b))
Fo = Fw[:, cells]; fclim = np.array([Fo[m::12].mean(0) for m in range(12)]); Fc = np.tile(fclim, (10, 1))
BN = {'tropical (W01)': 0, 'temperate (W02)': 1, 'boreal (W03)': 2}
out = {}
for variant in ('A', 'B'):
    feats, X, y, months, cid, bd, mono = table(variant); yr = months // 12
    print(f'\n=== variant {variant}: {len(feats)} features ({"with" if variant == "A" else "without"} the cell WAD2M range), {len(cells)} wet cells, {len(y) / 1e6:.1f} M rows', flush=True)
    # (A) temporal 2-fold: odd years -> even years and back
    P_oof = np.full((NM, len(cells)), np.nan, np.float32); imp = {}
    for b_name, b in BN.items():
        for fold in (0, 1):
            tr = (bd == b) & (yr % 2 == fold); te = (bd == b) & (yr % 2 != fold)
            p, m = fit_predict(X, y, tr, te, mono); P_oof[months[te], cid[te]] = p
        print(f'  {b_name}: temporal folds done {time.time() - t0:.0f}s', flush=True)
    P_oof = np.clip(P_oof, 0, 1); P_oofB = np.clip(P_oof, np.broadcast_to(fmin_w[cells], P_oof.shape), np.broadcast_to(fmax_w[cells], P_oof.shape))
    # (B) spatial half split, all years
    rng = np.random.default_rng(0); half = rng.random(len(cells)) < 0.5; P_sp = np.full((NM, len(cells)), np.nan, np.float32)
    for b_name, b in BN.items():
        tr = (bd == b) & half[cid]; te = (bd == b) & ~half[cid]
        p, m = fit_predict(X, y, tr, te, mono); P_sp[months[te], cid[te]] = p
        imp[b_name] = m
    P_sp = np.clip(P_sp, 0, 1)
    print(f'  spatial half-split done {time.time() - t0:.0f}s', flush=True)
    hdr = f"{'band':16s} {'obs area':>8s} | monthly R2 on wet cells: clim | ML out-of-year | ML out-of-year bounded | ML unseen cells || seasonal r (ML) amp obs/ML | IAV r annual area: ML out-of-year | ML bounded | ML unseen cells (obs vs pred on those cells)"
    print(hdr)
    def r2(F, P, sel): ok = np.isfinite(P[:, sel]); return 1 - np.nansum((F[:, sel] - P[:, sel]) ** 2) / max(np.nansum((F[:, sel] - F[:, sel].mean(0)) ** 2), 1e-12)
    for b_name, b in list(BN.items()) + [('GLOBAL', None)]:
        sel = np.ones(len(cells), bool) if b is None else band[cells] == b; selu = sel & ~half
        o = band_series(Fo, sel); p = band_series(P_oof, sel); pb = band_series(P_oofB, sel); ou = band_series(Fo, selu); pu = band_series(np.nan_to_num(P_sp), selu)
        sr = seas(o, p)
        print(f"{b_name:16s} {o.mean():8.2f} | {r2(Fo, Fc, sel):.2f} | {r2(Fo, P_oof, sel):.2f} | {r2(Fo, P_oofB, sel):.2f} | {r2(Fo, P_sp, selu):.2f} || {sr[0]:.2f} amp {sr[1]:.2f}/{sr[2]:.2f} | {iav(o, p):+.2f} | {iav(o, pb):+.2f} | {iav(ou, pu):+.2f}")
    out[variant] = dict(P_oof=P_oofB, P_sp=P_sp)
np.savez_compressed(f'{S}/wetland/rf_oof.npz', cells=cells, I=I[cells], J=J[cells], band=band[cells], Fw=Fo, P_oof_A=out['A']['P_oof'], P_oof_B=out['B']['P_oof'], P_sp_A=out['A']['P_sp'], months=np.arange(NM))
print(f'\nsaved rf_oof.npz; total {time.time() - t0:.0f}s')
