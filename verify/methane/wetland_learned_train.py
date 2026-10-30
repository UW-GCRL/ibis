"""Learned wetland fraction, v2 (docs/METHANE.md 8, 10): gradient-boosted trees (XGBoost, hist, monotone increasing in every
water variable), one model per wetland band (W01/W02/W03), trained on the WAD2M monthly inundated fraction from a hindcast's
MONTH-MEAN soil state (output.soil_means: tsoi_mean, wsoi_mean, wisoi_mean, tg_mean, averaged over every time step) and its
monthly totals (precipitation, ET, runoff + drainage), with lags, 3-month means and 5x5-cell neighbourhood means, plus static
CLM topography, soil texture, latitude and the cell's WAD2M range. No month-of-year, no climatology among the predictors.
The switch the model feeds is the HYBRID: f = clip(f_clim[m] + lambda_band (P - Pbar[m]), 0, f_max_cell), the observed WAD2M
seasonal climatology plus the emulator's anomaly against its own climatology, inflated per band.
Validation: odd years -> even years and back (every month predicted out of year); lambda per band cross-fitted (estimated on
the other fold's out-of-year anomalies, by matching the sd of the annual band-area anomalies); scored against WAD2M by band.
    python wetland_learned_train.py <hindcast out with soil means> [--save <model.npz>]      (gpu venv; CPU threads)"""
import sys, time, json, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
import xgboost as xgb
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R]
from ibis.io.soil_init import compute_soil_properties
from ibis.io.pft_types import wetland_band
S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'
H = sys.argv[1]; SAVE = sys.argv[sys.argv.index('--save') + 1] if '--save' in sys.argv else ''
Y0, Y1 = 2001, 2010; NY = Y1 - Y0 + 1; NM = NY * 12; t0 = time.time(); NTH = 24
m0 = xr.open_dataset(f'{H}/ibis_monthly_{Y0}_01.nc'); assert 'wsoi_mean' in m0, 'hindcast has no soil means (output.soil_means)'
mlat = np.asarray(m0['lat'].values, float); land = np.isfinite(m0['fu'].values); ij = np.argwhere(land); I, J = ij[:, 0], ij[:, 1]; n = len(I); nl = m0.sizes['soilay']
# porosity and layer thickness, the model's own (as wetland_clim_prep.py)
sd_ = xr.open_dataset('/shared/nas2/Min/HWSD/soita.nc'); sand = np.asarray(sd_['sandpct'].values, float); clay = np.asarray(sd_['claypct'].values, float); slat = np.asarray(sd_['lat'].values, float)
if (slat[0] < slat[-1]) != (mlat[0] < mlat[-1]): sand, clay = sand[:, ::-1], clay[:, ::-1]
sand = np.where(np.isfinite(sand), sand, 45.0); clay = np.where(np.isfinite(clay), clay, 25.0)
props = compute_soil_properties(sand[:, I, J].T, clay[:, I, J].T); poros = np.asarray(props['poros'])[:, :nl]; hsoi = np.asarray(props['hsoi'])[:nl]
# --- monthly predictors through the runtime's own feature builder (ibis.bgc.wetland_learned), month by month in order
from ibis.bgc.wetland_learned import FeatureBuilder, DYN_FEATS, STAT_FEATS, MONO as MONO_T
fb = FeatureBuilder(I, J, poros, hsoi); DYN = {k: np.zeros((NM, n), np.float32) for k in DYN_FEATS}; k = 0
for y in range(Y0, Y1 + 1):
    for m in range(12):
        with xr.open_dataset(f'{H}/ibis_monthly_{y}_{m + 1:02d}.nc') as d:
            g = lambda v: np.asarray(d[v].values, np.float64)[I, J]
            ro = np.nan_to_num(g('amtrunof')) + np.nan_to_num(g('amtdrain'))
            f = fb.step(g('tsoi_mean'), g('wsoi_mean'), g('wisoi_mean'), g('tg_mean'), np.nan_to_num(g('amprcp')), np.nan_to_num(g('amaet')), ro)
        for kk in DYN_FEATS: DYN[kk][k] = f[kk]
        k += 1
print(f'monthly predictors built (runtime feature builder) {time.time() - t0:.0f}s', flush=True)
MONO = set(MONO_T)
# --- static fields
sf = xr.open_dataset('/shared/nas2/Hangkai/ELM Edge Effects/inputdata/surfdata_0.5x0.5_CH_v1_all_forests.nc'); flip = float(sf['LATIXY'].values[0, 0]) < float(sf['LATIXY'].values[-1, 0])
st = lambda v: np.nan_to_num((np.asarray(sf[v].values, np.float32)[::-1] if flip else np.asarray(sf[v].values, np.float32))[I, J])
wd = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc'); FWall = np.asarray(wd['Fw'].values, np.float32)
FWall = FWall[:, ::-1] if float(wd['lat'].values[0]) < float(wd['lat'].values[-1]) else FWall
fmin_w = np.nan_to_num(np.nanmin(FWall, 0))[I, J]; fmax_w = np.nan_to_num(np.nanmax(FWall, 0))[I, J]
t = wd['time'].dt.year.values; Fw = np.nan_to_num(FWall[(t >= Y0) & (t <= Y1)])[:, I, J]
FCLIM = np.array([np.nan_to_num(np.nanmean(FWall[m::12], 0))[I, J] for m in range(12)], np.float32)       # 2000-2020 climatology, as the model's
STAT = {'FMAX_clm': st('FMAX'), 'SLOPE': st('SLOPE'), 'STD_ELEV': st('STD_ELEV'), 'TOPO': st('TOPO'), 'PCT_WETLAND': st('PCT_WETLAND'), 'PCT_LAKE': st('PCT_LAKE'), 'binfl': st('binfl'),
        'sand': sand[0, I, J].astype(np.float32), 'clay': clay[0, I, J].astype(np.float32), 'lat': mlat[I].astype(np.float32), 'fmin_w': fmin_w, 'fmax_w': fmax_w}
zone = np.load(f'{S}/meeting/glob_refs.npz')['zone'][I, J]; band = np.asarray(wetland_band(zone)); lat = mlat[I]
band = np.where(band >= 0, band, np.where(np.abs(lat) < 30, 0, np.where(lat >= 60, 2, 1)))
assert tuple(STAT) == STAT_FEATS, list(STAT)
wet = fmax_w > 0.005; cells = np.where(wet)[0]; FEATS = list(DYN_FEATS) + list(STAT_FEATS); nc = len(cells)
print(f'features: {len(FEATS)}; wet cells {nc}; {time.time() - t0:.0f}s', flush=True)
def table(rows_t):   # rows_t: month indices -> X (len(rows_t) * nc, nfeat)
    X = np.empty((len(rows_t) * nc, len(FEATS)), np.float32)
    for c, f in enumerate(FEATS): X[:, c] = (DYN[f][rows_t][:, cells] if f in DYN else np.broadcast_to(STAT[f][cells], (len(rows_t), nc))).reshape(-1)
    return X
mono = '(' + ','.join('1' if f in MONO else '0' for f in FEATS) + ')'
PAR = dict(tree_method='hist', max_depth=8, eta=0.08, min_child_weight=200, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8, nthread=NTH, monotone_constraints=mono, objective='reg:squarederror', seed=0)
NROUND = 400
def fit(rows_t, b):
    X = table(rows_t); y = Fw[rows_t][:, cells].reshape(-1); bm = np.tile(band[cells] == b, len(rows_t))
    return xgb.train(PAR, xgb.DMatrix(X[bm], label=y[bm]), num_boost_round=NROUND)
def predict(model, rows_t, b):
    sel = band[cells] == b; X = table(rows_t); out = np.full((len(rows_t), nc), np.nan, np.float32)
    p = model.predict(xgb.DMatrix(X[np.tile(sel, len(rows_t))])); out[:, sel] = np.clip(p.reshape(len(rows_t), sel.sum()), 0, 1); return out
yr = np.arange(NM) // 12; folds = [np.where(yr % 2 == 0)[0], np.where(yr % 2 == 1)[0]]
P = np.full((NM, nc), np.nan, np.float32)
for b in (0, 1, 2):
    for k_ in (0, 1):
        tr, te = folds[k_], folds[1 - k_]; mdl = fit(tr, b); p = predict(mdl, te, b); sel = band[cells] == b; P[np.ix_(te, np.where(sel)[0])] = p[:, sel]
    print(f'  band {b}: out-of-year folds done {time.time() - t0:.0f}s', flush=True)
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat[cells])); Fo = Fw[:, cells]; Fc = np.tile(FCLIM[:, cells], (NY, 1)); fmx = fmax_w[cells]
def bs(X, sel): return (np.nan_to_num(X[:, sel]) * area[sel]).sum(1) / 1e12
def ann(a): return a.reshape(-1, 12).mean(1)
def anom_hybrid(Pf, rows_fold, lam):
    """hybrid on rows_fold, with the emulator climatology Pbar from the same out-of-year predictions of that fold"""
    Pb = np.array([np.nanmean(Pf[rows_fold][m::12], 0) for m in range(12)])
    H = Fc[rows_fold] + lam * (Pf[rows_fold] - np.tile(Pb, (len(rows_fold) // 12, 1))); return np.clip(H, 0, fmx[None, :])
# cross-fitted inflation: lambda_b for fold k from fold (1-k)'s out-of-year anomalies
LAM = {}
for b in (0, 1, 2):
    sel = band[cells] == b; lam_k = []
    for k_ in (0, 1):
        r = folds[k_]; o = ann(bs(Fo[r], sel)); h = ann(bs(anom_hybrid(P, r, 1.0), sel)); lam_k.append(float(np.clip(np.std(o - o.mean()) / max(np.std(h - h.mean()), 1e-9), 1.0, 5.0)))
    LAM[b] = lam_k
Hy = np.zeros_like(P); Hy1 = np.zeros_like(P)
for b in (0, 1, 2):
    sel = np.where(band[cells] == b)[0]
    for k_ in (0, 1):
        r = folds[k_]; Hy[np.ix_(r, sel)] = anom_hybrid(P, r, LAM[b][1 - k_])[:, sel]; Hy1[np.ix_(r, sel)] = anom_hybrid(P, r, 1.0)[:, sel]
def r2(F_, Q, sel): return 1 - np.nansum((F_[:, sel] - Q[:, sel]) ** 2) / np.nansum((F_[:, sel] - F_[:, sel].mean(0)) ** 2)
def iav(a, b_): a = ann(a); b_ = ann(b_); return float(np.corrcoef(a - a.mean(), b_ - b_.mean())[0, 1]) if np.std(b_) > 0 else float('nan')
def sdr(a, b_): a = ann(a); b_ = ann(b_); return float(np.std(b_ - b_.mean()) / max(np.std(a - a.mean()), 1e-12))
print('\nOut-of-year scores against WAD2M 2001-2010 (month-mean features). Columns: monthly R2 over wet cells | IAV r of the annual band area | sd ratio of the annual anomalies (model/obs)')
print(f"{'band':10s} {'area':>5s} | {'clim':>5s} {'emul':>5s} {'hyb1':>5s} {'hybL':>5s} | {'emul':>6s} {'hyb1':>6s} {'hybL':>6s} | {'hyb1':>5s} {'hybL':>5s} | lambda folds")
for nm, b in (('tropical', 0), ('temperate', 1), ('boreal', 2), ('GLOBAL', None)):
    sel = np.ones(nc, bool) if b is None else band[cells] == b; o = bs(Fo, sel)
    print(f"{nm:10s} {o.mean():5.2f} | {r2(Fo, Fc, sel):5.2f} {r2(Fo, P, sel):5.2f} {r2(Fo, Hy1, sel):5.2f} {r2(Fo, Hy, sel):5.2f} | {iav(o, bs(P, sel)):+6.2f} {iav(o, bs(Hy1, sel)):+6.2f} {iav(o, bs(Hy, sel)):+6.2f} | {sdr(o, bs(Hy1, sel)):5.2f} {sdr(o, bs(Hy, sel)):5.2f} | " + (str([round(v, 2) for v in LAM[b]]) if b is not None else ''))
print('\nseasonal cycle of band area (Mkm2), obs | hybrid-lambda (decade means by calendar month):')
for nm, b in (('tropical', 0), ('temperate', 1), ('boreal', 2)):
    sel = band[cells] == b; o = bs(Fo, sel).reshape(-1, 12).mean(0); h = bs(Hy, sel).reshape(-1, 12).mean(0)
    print(f'  {nm:10s} obs ' + ' '.join(f'{v:5.2f}' for v in o) + f'\n  {"":10s} hyb ' + ' '.join(f'{v:5.2f}' for v in h))
np.savez_compressed(f'{S}/wetland/learned_v2_oof.npz', cells=cells, band=band[cells], Fo=Fo, P=P, Hy=Hy, Fc=Fc)
if SAVE:
    print('\ntraining the final models on all ten years ...', flush=True)
    allr = np.arange(NM); models = {}; Pin = np.full((NM, nc), np.nan, np.float32)
    for b in (0, 1, 2):
        mdl = fit(allr, b); models[b] = mdl; p = predict(mdl, allr, b); sel = band[cells] == b; Pin[:, sel] = p[:, sel]
    PBAR = np.array([np.nanmean(Pin[m::12], 0) for m in range(12)], np.float32)
    lam_final = {b: float(np.mean(LAM[b])) for b in (0, 1, 2)}
    np.savez_compressed(SAVE, feats=np.array(FEATS), dyn_feats=np.array(DYN_FEATS), stat_feats=np.array(STAT_FEATS), mono=mono,
                        cells_I=I[cells], cells_J=J[cells], band=band[cells], pbar=PBAR, fclim=FCLIM[:, cells], fmax=fmax_w[cells], fmin=fmin_w[cells],
                        lam=np.array([lam_final[0], lam_final[1], lam_final[2]], np.float32), stat=np.stack([STAT[f][cells] for f in STAT], 1),
                        models=np.array([models[b].save_raw('json').decode() for b in (0, 1, 2)]), params=json.dumps({k: (v if not isinstance(v, np.generic) else v.item()) for k, v in PAR.items()}),
                        nround=NROUND, years=np.array([Y0, Y1]), hindcast=H, xgboost=xgb.__version__,
                        note='docs/METHANE.md 8/10: hybrid f = clip(fclim[m] + lam[band] (P - pbar[m]), 0, fmax); P = XGBoost model of the band on the month-mean features')
    print('saved', SAVE, 'lambda', lam_final)
print(f'total {time.time() - t0:.0f}s')
