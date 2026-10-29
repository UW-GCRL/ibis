"""Extract 2006-2010 means of a protocol run (and, with --refs, the references) onto 0.5-degree grids for the meeting figures.
    python extract_global.py <tag> [--refs]"""
import sys, os, dataclasses, numpy as np, xarray as xr, warnings
warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [R, f'{R}/python-port', f'{R}/verify/calibration']
from landmask import true_landfrac
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; M = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad/meeting'
LAT = np.arange(89.75, -90, -0.5); LON = np.arange(-179.75, 180, 0.5); YRS = range(2006, 2011)
tag = sys.argv[1]
if tag != 'refs':
    run = f'{V}/final_{tag}/hindcast'; o = {}
    rdy = lambda y, v: np.asarray(xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc')[v].values, float)
    rdm = lambda y, k, v: np.nan_to_num(np.asarray(xr.open_dataset(f'{run}/out/ibis_monthly_{y}_{k:02d}.nc')[v].values, float))
    o['gpp'] = np.mean([rdy(y, 'aygpptot') for y in YRS], 0) * 1e3
    o['npp'] = np.mean([rdy(y, 'aynpptot') for y in YRS], 0) * 1e3
    o['nee'] = np.mean([rdy(y, 'ayneetot') for y in YRS], 0) * 1e3          # IBIS sign: + = uptake (NPP - Rh - disturbance)
    o['rh'] = np.mean([rdy(y, 'ayco2mic') for y in YRS], 0) * 1e3
    o['et'] = np.mean([rdy(y, 'ayaet') for y in YRS], 0)
    o['ch4'] = np.mean([rdy(y, 'aych4') for y in YRS], 0) * 1e3 * 16.0 / 12.0  # g CH4 m-2 yr-1
    o['gpp_m'] = np.array([np.mean([rdm(y, k, 'amgpptot') for y in YRS], 0) for k in range(1, 13)]) * 1e3
    o['lai_peak'] = np.array([np.mean([rdm(y, k, 'laiu') * rdm(y, k, 'fu') + rdm(y, k, 'lail') * rdm(y, k, 'fl') for y in YRS], 0) for k in range(1, 13)]).max(0)
    o['fu'] = rdm(2010, 7, 'fu'); o['cveg'] = rdy(2010, 'cveg'); o['csoil'] = rdy(2010, 'totcsoi')
    from ibis.state import VegetationState
    r = np.load(f'{run}/restart/ibis_restart_2010.npz'); tr = str(r['treedef_repr']); vo = tr[:tr.find('VegetationState')].count('*'); vf = [f.name for f in dataclasses.fields(VegetationState)]
    ij = np.argwhere(np.isfinite(xr.open_dataset(f'{V}/final_s31f/hindcast/out/ibis_monthly_2010_07.nc')['fu'].values)); I, J = ij[:, 0], ij[:, 1]   # land-point order (s31f has every land cell finite)
    def grid(v): g = np.full((360, 720), np.nan); g[I, J] = v; return g
    cbiow = np.asarray(r[f'leaf_{vo + vf.index("cbiow")}'], float)
    o['wood'] = grid(cbiow[:, :8].sum(1)); o['wood_dec'] = grid(cbiow[:, [1, 4, 6, 7]].sum(1)); o['wood_ev'] = grid(cbiow[:, [0, 2, 3, 5]].sum(1))
    o['disturbf'] = grid(np.asarray(r[f'leaf_{vo + vf.index("disturbf")}'], float)) if 'disturbf' in vf else np.full((360, 720), np.nan)
    np.savez_compressed(f'{M}/glob_{tag}.npz', **o); print('saved', tag, {k: float(np.nanmean(v)) for k, v in o.items() if v.ndim == 2})
else:
    o = {'lat': LAT, 'lon': LON, 'lf': true_landfrac()}
    f = xr.open_dataset('/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc'); ys = f['time'].dt.year.values; ms = f['time'].dt.month.values
    dim = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]); flip = float(f['lat'].values[0]) < float(f['lat'].values[-1])
    gm = np.array([np.nanmean(f['gpp'].values[(ys >= 2006) & (ys <= 2010) & (ms == k)].astype(float), 0) * dim[k - 1] for k in range(1, 13)])
    gm = gm[:, ::-1] if flip else gm; gm = np.where(np.abs(gm) > 1e5, np.nan, gm); o['gpp_m'] = gm; o['gpp'] = gm.sum(0)
    e = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/et/et_ensemble_0.5deg.npz'); et = np.asarray(e['ens_ann_mean'], float)
    o['et'] = et[::-1] if np.asarray(e['lat'])[0] < np.asarray(e['lat'])[-1] else et
    z = np.load(f'{M}/../s31b_maps.npz'); o['nee_gcb'] = z['nee_gcb']; o['nee_fluxcom'] = -z['nee_fluxcom'] if np.nanmean(z['nee_fluxcom']) > 0 else z['nee_fluxcom']; o['wad'] = z['wad']
    o['nee_fluxcom'] = np.where(np.abs(z['nee_fluxcom']) > 1e5, np.nan, -z['nee_fluxcom'])   # stored as release-positive; + = uptake here
    g = np.nanmean(np.load('/shared/nas2/Hangkai/ibis/inputs/eval/lai/glass_v61_0.5deg_monthly_2001-2010.npz')['lai'][5:10], 0).max(0)
    mm = np.asarray(np.load('/shared/nas2/Hangkai/ibis/inputs/eval/lai/gimms_lai4g_0.5deg_monthly_2001-2010.npz')['lai'], float); mm = (mm if mm.ndim == 4 else mm.reshape(-1, 12, 360, 720))[5:10]; mm = np.nanmean(mm, 0).max(0)
    o['lai_peak'] = np.where(np.isfinite(mm), 0.5 * (g + mm), g)
    x = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/eval/biomass/ilamb_dl/XuSaatchi2021_XuSaatchi.nc'); xs = x['time'].dt.year.values
    xa = np.nanmean(x['biomass'].values[(xs >= 2006) & (xs <= 2010)].astype(float), 0) * 0.1; o['wood'] = xa[::-1] if float(x['lat'].values[0]) < float(x['lat'].values[-1]) else xa
    sd = xr.open_dataset('/shared/nas2/Hangkai/ELM Edge Effects/inputdata/surfdata_0.5x0.5_CH_v1_all_forests.nc'); P = sd['PCT_NAT_PFT'].values.astype(float) / 100.0
    if float(sd['LATIXY'].values[0, 0]) < float(sd['LATIXY'].values[-1, 0]): P = P[:, ::-1]
    o['tree'] = P[1:9].sum(0); o['tree_dec'] = P[[3, 6, 7, 8]].sum(0)
    o['crop'] = np.load('/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz')['cropfrac'].clip(0, 1)
    k = np.load('/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz'); c = np.asarray(k['classes'], int)[-1]
    if np.asarray(k['lat'], float)[0] < np.asarray(k['lat'], float)[-1]: c = c[::-1]
    from ibis.io.pft_types import zones_from_classes; o['zone'] = zones_from_classes(c).astype(np.int16)
    # GFED5 burned fraction 2001-2005 on the model's land points (the protocol's fire evaluation)
    sys.path.insert(0, f'{R}/verify/fire'); import evaluate_gfed5 as eg
    st = f'{V}/final_s31e/hindcast/restart/ibis_restart_2010.npz'; lat, lon, li, lo, mlat, mlon = eg._grid(st)
    obs = eg.observed(eg.DEFAULT_GFED5, li, lo, mlat, mlon, range(2001, 2006)); gg = np.full((360, 720), np.nan)
    gi = np.clip(np.round((89.75 - np.asarray(lat)) / 0.5).astype(int), 0, 359); gj = np.clip(np.round((np.asarray(lon) + 179.75) / 0.5).astype(int), 0, 719); gg[gi, gj] = obs
    o['burn_gfed5'] = gg
    np.savez_compressed(f'{M}/glob_refs.npz', **o); print('saved refs')
