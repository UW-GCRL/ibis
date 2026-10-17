"""Leaf-area reference products on the model grid (0.5 deg, north-up, monthly 2001-2010) for the leaf-area stage
(CALIBRATION.md 48).

    /usr/bin/python3 -I verify/calibration/lai_products_prep.py      (the system Python: GDAL with HDF4)

GLASS LAI V61 (MODIS-based, 0.05 deg, 8-day; glass.hku.hk/archive/LAI/MODIS/0.05D): byte LAI x 10, fill 255; 10 x 10
block means over valid pixels (a 0.5-degree cell with fewer than 25 % valid pixels is NaN), each 8-day composite
spread over the months by the days it covers. GIMMS LAI4g (Cao et al. 2023, doi:10.5194/essd-15-4877-2023; ILAMB's
0.5-degree monthly resampling of the Zenodo release): the 2001-2010 months, flipped north-up if needed.
Output: inputs/eval/lai/{glass_v61,gimms_lai4g}_0.5deg_monthly_2001-2010.npz with lai (10, 12, 360, 720), lat, lon.
"""
import glob, os, re, datetime, numpy as np
from osgeo import gdal
B = '/shared/nas2/Hangkai/ibis/inputs/eval/lai'
LAT = np.arange(89.75, -90, -0.5); LON = np.arange(-179.75, 180, 0.5); YRS = range(2001, 2011)


def glass():
    out = np.zeros((10, 12, 360, 720)); wts = np.zeros((10, 12, 360, 720))
    for yi, y in enumerate(YRS):
        for f in sorted(glob.glob(f'{B}/raw_glass_v61/{y}/GLASS01B01.V61.A{y}*.hdf')):
            doy = int(re.search(r'\.A\d{4}(\d{3})\.', f).group(1))
            g = gdal.Open(f); a = g.GetRasterBand(1).ReadAsArray().astype(np.float32); g = None   # hold the dataset while reading its band
            assert a.shape == (3600, 7200), a.shape
            v = np.where(a <= 100, a * 0.1, np.nan).reshape(360, 10, 720, 10)
            n = np.isfinite(v).sum(axis=(1, 3)); m = np.where(n >= 25, np.nansum(v, axis=(1, 3)) / np.maximum(n, 1), np.nan)
            d0 = datetime.date(y, 1, 1) + datetime.timedelta(doy - 1)
            for k in range(8):                                      # the composite's days, clipped to the year
                d = d0 + datetime.timedelta(k)
                if d.year != y: break
                ok = np.isfinite(m); out[yi, d.month - 1][ok] += m[ok]; wts[yi, d.month - 1][ok] += 1
        print(y, 'done', flush=True)
    lai = np.where(wts > 0, out / np.maximum(wts, 1), np.nan).astype(np.float32)
    np.savez_compressed(f'{B}/glass_v61_0.5deg_monthly_2001-2010.npz', lai=lai, lat=LAT, lon=LON,
                        note='GLASS LAI V61 0.05 deg 8-day -> 0.5 deg monthly (lai_products_prep.py)')


def gimms():
    ds = gdal.Open(f'NETCDF:"{B}/raw_ilamb_gimms_lai4g/cao2023_lai.nc":lai')
    a = ds.ReadAsArray().astype(np.float32)                         # (time, lat, lon) as stored
    md = gdal.Open(f'{B}/raw_ilamb_gimms_lai4g/cao2023_lai.nc').GetMetadata()
    gt = ds.GetGeoTransform()                                       # GDAL presents NetCDF north-up when it can
    nt = a.shape[0]; assert nt == 468 and a.shape[1:] == (360, 720), a.shape
    if gt[5] > 0: a = a[:, ::-1]                                    # make north-up
    a = np.where((a >= 0) & (a < 15), a, np.nan)
    start = (2001 - 1982) * 12
    lai = a[start:start + 120].reshape(10, 12, 360, 720)
    np.savez_compressed(f'{B}/gimms_lai4g_0.5deg_monthly_2001-2010.npz', lai=lai, lat=LAT, lon=LON,
                        note='GIMMS LAI4g (Cao et al. 2023) via ILAMB 0.5 deg monthly, 2001-2010 (lai_products_prep.py); geotransform ' + str(gt))
    print('gimms done', gt)


if __name__ == '__main__':
    import sys
    if 'glass' not in sys.argv[1:]: gimms()
    if 'gimms' not in sys.argv[1:]: glass()
