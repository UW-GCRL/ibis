"""Amazon diagnostics (2026-10-25, Hangkai: "fire is overestimated in Amazon, and the total carbon in Amazon looks strange").
2010 state of each protocol run: fire disturbance fraction (disturbf), wood / leaf / root carbon of the trees (cbiow, cbiol, cbior,
positions 1-8), litter and soil carbon, tree cover fu, plus 2006-2010 GPP / NPP and the yearly NEE terms. Saved on the 0.5-degree grid.
    python amazon_extract.py <run tag> [...]"""
import sys, dataclasses, numpy as np, xarray as xr, warnings
warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [R, f'{R}/python-port']
from ibis.state import VegetationState, CarbonState
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; OUT = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work/nee_cycle/amazon'
import os; os.makedirs(OUT, exist_ok=True)
ref_mask = xr.open_dataset(f'{V}/final_s31f/hindcast/out/ibis_monthly_2010_07.nc')['fu'].values
for tag in sys.argv[1:]:
    run = f'{V}/final_{tag}/hindcast'
    y10 = xr.open_dataset(f'{run}/out/ibis_yearly_2010.nc')
    ij = np.argwhere(np.isfinite(ref_mask))   # s31f land order: every land cell is finite there (road B has one NaN cell)
    r = np.load(f'{run}/restart/ibis_restart_2010.npz'); tr = str(r['treedef_repr'])
    def block(cls, name):
        off = tr[:tr.find(cls.__name__)].count('*'); fl = [f.name for f in dataclasses.fields(cls)]
        return np.asarray(r[f'leaf_{off + fl.index(name)}'], float) if name in fl else None
    o = {}
    npt = len(ij)
    def grid(v):
        g = np.full((360, 720), np.nan)
        if v is None: return g
        v = np.asarray(v);
        if v.shape[0] != npt: print(tag, 'shape mismatch', v.shape, npt); return g
        g[ij[:, 0], ij[:, 1]] = v; return g
    cbiow, cbiol, cbior = block(VegetationState, 'cbiow'), block(VegetationState, 'cbiol'), block(VegetationState, 'cbior')
    o['wood'] = grid(cbiow[:, :8].sum(1)); o['leaf_tree'] = grid(cbiol[:, :8].sum(1)); o['root_tree'] = grid(cbior[:, :8].sum(1))
    o['leaf_low'] = grid(cbiol[:, 8:].sum(1)); o['root_low'] = grid(cbior[:, 8:].sum(1))
    o['disturbf'] = grid(block(VegetationState, 'disturbf'))
    for k in ('totlit', 'totcsoi', 'totalit', 'totrlit'):
        b = block(CarbonState, k)
        if b is None: b = block(VegetationState, k)
        o[k] = grid(b)
    for k in ('cveg', 'totcsoi', 'aygpptot', 'aynpptot', 'ayneetot', 'ayco2mic', 'fu'):
        if k in y10: o[f'y_{k}'] = np.asarray(y10[k].values, float)
    yrs = range(2006, 2011)
    o['gpp'] = np.mean([np.asarray(xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc')['aygpptot'].values, float) for y in yrs], 0) * 1e3
    o['npp'] = np.mean([np.asarray(xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc')['aynpptot'].values, float) for y in yrs], 0) * 1e3
    np.savez_compressed(f'{OUT}/{tag}.npz', **o); print(tag, 'saved', sorted(o))
