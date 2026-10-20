"""Root-zone soil-water climatology for the dynamic wetland area (docs/METHANE.md 7; ibis.bgc.wetland_area).

    python verify/methane/wetland_clim_prep.py <hindcast out dir> <soil_file> <out.npz> [--years 2001-2010]

W = sum over the top metre (ROOT_LAYERS layers) of poros * wsoi * hsoi, m, from the hindcast's monthly files (wsoi,
tsoi by layer at the end of the month; a frozen layer counts 0), porosity per layer from the soil file through
ibis.io.soil_init.compute_soil_properties (the model's own), layer thicknesses the model's. Output: mu, sd
(12, 360, 720) north-up on the model grid (NaN off land), n_years, and the layer table.
"""
import sys, numpy as np, xarray as xr
sys.path.insert(0, 'python-port')
from ibis.bgc.wetland_area import root_zone_water, ROOT_LAYERS
from ibis.io.soil_init import compute_soil_properties
a = sys.argv[1:]; hind, soil_file, out = a[:3]
y0, y1 = (int(v) for v in (a[a.index('--years') + 1] if '--years' in a else '2001-2010').split('-'))
with xr.open_dataset(soil_file) as sd:
    sand = np.asarray(sd[[v for v in ('sandpct', 'sand') if v in sd][0]].values, float); clay = np.asarray(sd[[v for v in ('claypct', 'clay') if v in sd][0]].values, float)
# the soil file is (layer, lat, lon); flip to the model's north-up rows if its latitude runs the other way
with xr.open_dataset(f'{hind}/ibis_monthly_{y0}_01.nc') as m0:
    mlat = np.asarray(m0['lat'].values, float); nlay = m0.sizes['soilay']
with xr.open_dataset(soil_file) as sd:
    slat = np.asarray(sd['lat'].values, float)
flip = (slat[0] < slat[-1]) != (mlat[0] < mlat[-1])
if sand.ndim == 3 and flip: sand, clay = sand[:, ::-1], clay[:, ::-1]
sand = np.where(np.isfinite(sand), sand, 45.0); clay = np.where(np.isfinite(clay), clay, 25.0)
land = np.isfinite(xr.open_dataset(f'{hind}/ibis_monthly_{y0}_01.nc')['fu'].values)
ij = np.argwhere(land); I, J = ij[:, 0], ij[:, 1]
props = compute_soil_properties(sand[:, I, J].T if sand.ndim == 3 else np.repeat(sand[I, J][:, None], nlay, 1),
                                clay[:, I, J].T if clay.ndim == 3 else np.repeat(clay[I, J][:, None], nlay, 1))
poros = np.asarray(props['poros'])[:, :nlay]; hsoi = np.asarray(props['hsoi'])
acc = np.zeros((12, len(I))); acc2 = np.zeros((12, len(I))); n = np.zeros(12)
for y in range(y0, y1 + 1):
    for m in range(12):
        with xr.open_dataset(f'{hind}/ibis_monthly_{y}_{m + 1:02d}.nc') as d:
            ws = np.asarray(d['wsoi'].values, float)[I, J]; ts = np.asarray(d['tsoi'].values, float)[I, J]
        w = root_zone_water(ws, poros, hsoi, ts); acc[m] += w; acc2[m] += w * w; n[m] += 1
    print(y, flush=True)
mu = acc / n[:, None]; sd_ = np.sqrt(np.maximum(acc2 / n[:, None] - mu ** 2, 0.0))
MU = np.full((12, 360, 720), np.nan); SD = MU.copy(); MU[:, I, J] = mu; SD[:, I, J] = sd_
np.savez_compressed(out, mu=MU, sd=SD, lat=mlat, n_years=y1 - y0 + 1, hsoi=hsoi, root_layers=ROOT_LAYERS,
                    note=f'root-zone liquid water (m) climatology from {hind} {y0}-{y1}, soil {soil_file} (wetland_clim_prep.py)')
print('wrote', out, f'land points {len(I)}, W mean {np.nanmean(mu):.3f} m, sd mean {np.nanmean(sd_):.3f} m')
