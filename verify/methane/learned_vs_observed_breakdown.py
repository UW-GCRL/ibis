"""Learned vs observed wetland area (and the CH4 it gives), 2001-2010, broken down: by year, by calendar month, every month of
the series, and in space (cell-level agreement, how much area moves between cells, by Koppen group). Read-only.
    python -I learned_vs_observed_breakdown.py <learned out> <observed out>"""
import sys, re, ast, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/calibration']
from landmask import true_landfrac
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', open(f'{R}/python-port/ibis/io/pft_types.py').read()).group(1))
L, O = sys.argv[1], sys.argv[2]; S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'
NDAY = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
m0 = xr.open_dataset(f'{L}/ibis_monthly_2001_01.nc'); lat = np.asarray(m0['lat'].values); I, J = np.where(np.isfinite(m0['fu'].values)); la = lat[I]
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(la)) * np.nan_to_num(true_landfrac()[I, J])
def load(d, v):
    return np.array([np.nan_to_num(np.asarray(xr.open_dataset(f'{d}/ibis_monthly_{y}_{m:02d}.nc')[v].values, float))[I, J] for y in range(2001, 2011) for m in range(1, 13)])
fl, fo = load(L, 'finund'), load(O, 'finund'); cl, co = load(L, 'amch4'), load(O, 'amch4')
ndays = np.array([NDAY[k % 12] for k in range(120)], float)
A = lambda f, m: (f[:, m] * area[m]).sum(1) / 1e12                                          # Mkm2 per month
C = lambda c, m: (c[:, m] * area[m]).sum(1) * ndays * 16.0 / 12.0 / 1e9                      # Tg CH4 per month
zone = np.load(f'{S}/meeting/glob_refs.npz')['zone'][I, J]
zn = lambda *nm: np.isin(zone, [ZONES.index(n) for n in nm])
BANDS = {'90S-30N': la < 30, '30-60N': (la >= 30) & (la < 60), '60-90N': la >= 60, 'GLOBAL': np.ones(len(la), bool)}
print('1. ANNUAL (decade mean and each year): wetland area, Mkm2 (annual mean) and CH4, Tg yr-1; learned | observed | learned - observed (%)')
for b, m in BANDS.items():
    al, ao = A(fl, m).reshape(10, 12).mean(1), A(fo, m).reshape(10, 12).mean(1); cL, cO = C(cl, m).reshape(10, 12).sum(1), C(co, m).reshape(10, 12).sum(1)
    print(f'  {b:8s} area {al.mean():.3f} | {ao.mean():.3f} | {100 * (al.mean() / ao.mean() - 1):+5.1f} %   CH4 {cL.mean():6.1f} | {cO.mean():6.1f} | {100 * (cL.mean() / cO.mean() - 1):+5.1f} %')
    print(f'  {"":8s} by year, area diff %: ' + ' '.join(f'{100 * (x / y - 1):+5.1f}' for x, y in zip(al, ao)) + '\n' + f'  {"":8s} by year, CH4 diff %:  ' + ' '.join(f'{100 * (x / y - 1):+5.1f}' for x, y in zip(cL, cO)))
print('\n2. MONTHLY, decade mean by calendar month: area diff % (learned / observed - 1), and the range over all 120 months')
for b, m in list(BANDS.items())[:3]:
    al, ao = A(fl, m), A(fo, m); d = 100 * (al / np.maximum(ao, 1e-9) - 1); dc = d.reshape(10, 12).mean(0)
    print(f'  {b:8s} ' + ' '.join(f'{v:+5.1f}' for v in dc) + f'   | 120 months: median |diff| {np.median(np.abs(d)):.1f} %, max |diff| {np.max(np.abs(d)):.1f} % ({2001 + int(np.argmax(np.abs(d))) // 12}-{int(np.argmax(np.abs(d))) % 12 + 1:02d})')
print('\n3. SPATIAL (decade-mean maps): cell correlation of the area maps, share of area that sits in different cells (0.5 x sum|diff| / total), by band and Koppen group')
G = {'humid tropics': zn('A_humid'), 'savanna + hot dry': zn('A_savanna', 'B_hot'), 'subtrop/Med/highland': zn('C_med', 'C_subtrop', 'C_highland'),
     'temperate': zn('C_oceanic', 'D_temperate'), 'boreal': zn('D_subarctic', 'D_extreme'), 'tundra + cold dry': zn('E_tundra', 'B_cold')}
ml, mo = fl.mean(0), fo.mean(0)
for nm, m in list(BANDS.items()) + list(G.items()):
    w = m & ((ml > 0) | (mo > 0)); tot = (mo[w] * area[w]).sum()
    if tot <= 0: continue
    moved = 0.5 * (np.abs(ml[w] - mo[w]) * area[w]).sum() / tot; r = np.corrcoef(ml[w], mo[w])[0, 1]
    print(f'  {nm:22s} area {(ml[w] * area[w]).sum() / 1e12:.3f} vs {tot / 1e12:.3f} Mkm2 | cell r {r:.3f} | area in different cells {100 * moved:.1f} %')
print('\n4. MONTH BY MONTH IN SPACE: for each of the 120 months, the share of that month\'s area in different cells (learned vs observed)')
for b, m in list(BANDS.items()):
    sh = [0.5 * (np.abs(fl[k, m] - fo[k, m]) * area[m]).sum() / max((fo[k, m] * area[m]).sum(), 1e-9) for k in range(120)]
    print(f'  {b:8s} median {100 * np.median(sh):.1f} %, range {100 * np.min(sh):.1f}-{100 * np.max(sh):.1f} %')
