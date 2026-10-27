"""Cold open-land boxes, 2010 hindcast: what the vegetation is in the cells where road B's rerun collapsed or ran away."""
import sys, glob, re, ast, numpy as np, xarray as xr
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; S = sys.argv[1]
src = open(glob.glob('/shared/nas2/Hangkai/ibis-port/python-port/**/pft_types.py', recursive=True)[0]).read()
ZONES = ast.literal_eval(re.search(r'\nZONES = (\([^)]*\))', src).group(1))
R = np.load(f'{S}/meeting/glob_refs.npz'); lat = np.asarray(R['lat'], float); lf = np.asarray(R['lf'], float); zone = np.asarray(R['zone'])
LAT = (lat[:, None] * np.ones((1, 720))) if lat.ndim == 1 else lat
lon = np.asarray(R['lon'], float); LON = (np.ones((360, 1)) * lon[None, :]) if lon.ndim == 1 else lon
area = (111.2e3 * 0.5) ** 2 * np.cos(np.deg2rad(LAT)) * lf
gfx = np.asarray(R['gpp'], float); laiobs = np.asarray(R['lai_peak'], float); tree = np.asarray(R['tree'], float)
def year(run, y=2010):
    d = xr.open_dataset(f'{V}/final_{run}/hindcast/out/ibis_yearly_{y}.nc'); flip = float(d['lat'].values[0]) < float(d['lat'].values[-1])
    g = lambda v: (np.asarray(d[v].values, float)[::-1] if flip else np.asarray(d[v].values, float))
    return {k: g(k) for k in ('aygpptot', 'aynppu', 'aynppl', 'laiu', 'lail', 'fu', 'fl', 'aystressu', 'cveg', 'totcsoi')}
zi = {z: i for i, z in enumerate(ZONES)}
E = np.isin(zone, [i for z, i in zi.items() if z.startswith('E')]) if any(z.startswith('E') for z in ZONES) else np.zeros_like(zone, bool)
boxes = {
  'Tibetan Plateau (28-40N, 78-105E, polar zone)': (LAT >= 28) & (LAT < 40) & (LON >= 78) & (LON < 105) & E,
  'High Arctic >70N': LAT >= 70,
  '60-70N': (LAT >= 60) & (LAT < 70),
  'Tierra del Fuego (52-56S, 75-65W)': (LAT <= -52) & (LAT > -56) & (LON >= -75) & (LON < -65),
  'all polar-zone cells': E,
}
print('zones:', ZONES)
def am(x, m): w = area[m]; v = x[m]; ok = np.isfinite(v); return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))
runs = sys.argv[2:]; Y = {r: year(r) for r in runs}
for nm, m in boxes.items():
    m = m & (lf > 0.05)
    print(f'\n### {nm}: {int(m.sum())} cells, {np.sum(area[m]) / 1e12:.2f} Mkm2; obs FLUXCOM GPP {am(gfx, m):.0f} gC/m2/yr, MODIS peak LAI {am(laiobs, m):.2f}, tree cover {am(tree, m):.2f}')
    print(f'{"run":6s} {"GPP":>5s} {"xFLX":>5s} {"NPPu":>5s} {"NPPl":>5s} {"laiu":>5s} {"lail":>5s} {"fu":>5s} {"fl":>5s} {"stressu":>7s} {"cveg":>5s} {"csoil":>5s}  GPP PgC')
    for r in runs:
        y = Y[r]; g = 1000 * am(y["aygpptot"], m)
        print(f'{r:6s} {g:5.0f} {g / max(am(gfx, m), 1):5.2f} {1000 * am(y["aynppu"], m):5.0f} {1000 * am(y["aynppl"], m):5.0f} {am(y["laiu"], m):5.2f} {am(y["lail"], m):5.2f} {am(y["fu"], m):5.2f} {am(y["fl"], m):5.2f} {am(y["aystressu"], m):7.2f} {am(y["cveg"], m):5.2f} {am(y["totcsoi"], m):5.1f}  {np.nansum(y["aygpptot"][m] * area[m]) / 1e12:.3f}')
