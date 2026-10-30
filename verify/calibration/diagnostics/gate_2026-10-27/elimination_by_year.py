"""Tree elimination under the cover limit, by year and zone: cells whose upper-canopy wood goes from > 0.3 kgC m-2 to
< 0.01 between consecutive restarts, the wood they held (PgC), and the zone's mean annual stress that year.
usage: python -I elimination_by_year.py RUN_DIR y0 y1 [RUN_DIR y0 y1 ...]   (restart/ibis_restart_YYYY.npz and out/ibis_yearly_YYYY.nc)"""
import sys, re, ast, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire', f'{R}/verify/calibration']
import evaluate_gfed5 as eg; from landmask import true_landfrac
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
S = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work'; Z = np.load(f'{S}/meeting/glob_refs.npz')['zone']
args = sys.argv[1:]
for i in range(0, len(args), 3):
    run, y0, y1 = args[i], int(args[i + 1]), int(args[i + 2])
    print(f'===== {run}')
    prev = None
    for y in range(y0, y1 + 1):
        p = f'{run}/restart/ibis_restart_{y}.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p)
        if prev is None:
            lf = true_landfrac(); gi = np.rint((89.75 - lat) / 0.5).astype(int); gj = np.rint((lon + 179.75) / 0.5).astype(int)
            area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat)) * np.nan_to_num(lf[gi, gj]); zone = Z[gi, gj]
        wood = np.asarray(st.vegetation.cbiow, float)[:, :8].sum(1); fu = np.asarray(st.vegetation.fu, float)
        try:
            u = np.asarray(xr.open_dataset(f'{run}/out/ibis_yearly_{y}.nc')['aystressu'].values, float)[gi, gj]
        except Exception: u = np.full_like(wood, np.nan)
        if prev is not None:
            elim = (prev > 0.3) & (wood < 0.01)
            parts = []
            for zn in ('A_savanna', 'B_hot', 'B_cold', 'C_med', 'C_subtrop', 'C_oceanic', 'D_temperate', 'D_subarctic'):
                m = elim & (zone == ZONES.index(zn))
                if m.sum(): parts.append(f'{zn} {int(m.sum())} cells {np.nansum(prev[m] * area[m]) / 1e12:.2f} PgC (u {np.nanmean(u[m]):.2f})')
            print(f'  {y}: eliminated {int(elim.sum())} cells, {np.nansum(prev[elim] * area[elim]) / 1e12:.2f} PgC of wood | ' + '; '.join(parts))
            for zn in ('D_temperate', 'C_med', 'A_savanna'):
                m = zone == ZONES.index(zn)
                print(f'        {zn:12s} mean u {np.nanmean(u[m]):.3f}  fu {np.average(fu[m], weights=area[m]):.3f}  wood {np.nansum(wood[m] * area[m]) / 1e12:.1f} PgC')
        prev = wood
