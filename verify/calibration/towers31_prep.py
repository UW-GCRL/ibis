#!/usr/bin/env python3
"""Towers of the IBIS 3.1 calibration (CALIBRATION.md 45) in the FLUXNET2015 layout the site harness reads.

    python3 verify/calibration/towers31_prep.py <out dir> [--sites A,B,...] [--workers 6]

For every tower on disk -- FLUXNET2015 Tier 1 (205 sites), ICOS Warm Winter 2020, the ICOS L2 archive and FLUXNET
v1.3 (new_sites_inventory.csv) -- it (1) counts the good NEE years of each record (a year is good with >= 240 daily
values and a mean NEE_VUT_REF_QC >= 0.5; NEE_CUT_REF where a record has no VUT), (2) takes the record with the most
good years (ties: v1.3 > ICOS L2 > Warm Winter > FLUXNET2015, the newer processing), (3) chooses the calibration
window -- 5 consecutive years (4 or 3 for short records) within 1996-2024, first year warm-up, last held out; the
held-out year and at least one fitted year good; most good years, then the latest -- and (4) writes

    <out>/compact/FLX_<s>_FLUXNET2015_FULLSET_{DD,MM,YY}_<a>-<b>_31.csv   the record's files, renamed (NEE_VUT_* copied
                                                                         from NEE_CUT_* where VUT is missing)
    <out>/compact/FLX_<s>_FLUXNET2015_FULLSET_HH_<y0>-<y1>_31.csv        compact half-hourly (or hourly) file of the
                    window: TIMESTAMP, TIMESTAMP_START, TA_F_MDS, SW_IN_F_MDS, SW_IN_F, NIGHT, NEE_VUT_REF, NEE_VUT_REF_QC
    <out>/compact/FLUXNET_siteinfo.csv                                    one row per tower with the record, window, Köppen
                                                                         class, zone and IBIS 3.1 type
Point IBIS_FLUX_DIR, IBIS_FLUX_HH_DIR and IBIS_FLUX_MM_DIR at <out>/compact. Needs numpy, pandas, PIL (Köppen map).
"""
import sys, os, glob, re, zipfile, io, csv, argparse
import numpy as np, pandas as pd
from multiprocessing import Pool

B = '/shared/nas2/Hangkai/ibis/inputs/sites'
F15 = f'{B}/FLUXNET2015_FULLSET'
PRIO = {'FLUXNET_v1.3': 3, 'ICOS_L2': 2, 'ICOS_WW2020': 1, 'FLUXNET2015': 0}
Y_MIN, Y_MAX = 1996, 2024


def records(site):
    """[(source, dd, mm, yy, hh_path or (zip, member) or None)] for a site."""
    out = []
    def first(p):
        g = sorted(glob.glob(p), key=lambda x: -os.path.getsize(x)); return g[0] if g else None
    dd = first(f'{F15}/daily_monthly_yearly/FLX_{site}_FLUXNET2015_FULLSET_DD_*.csv')
    if dd:
        hh = first(f'{F15}/halfhourly/FLX_{site}_FLUXNET2015_FULLSET_H[HR]_*.csv')
        if not hh:
            z = f'{F15}/_zips/fluxnet_original/{site}.zip'
            if os.path.exists(z):
                try:
                    m = [n for n in zipfile.ZipFile(z).namelist() if re.search(r'_FULLSET_H[HR]_', n)]
                    hh = (z, m[0]) if m else None
                except zipfile.BadZipFile: hh = None
        out.append(('FLUXNET2015', dd, dd.replace('_DD_', '_MM_'), dd.replace('_DD_', '_YY_'), hh))
    dd = first(f'{B}/ICOS_WW2020/daily_monthly_yearly/FLX_{site}_FLUXNET2015_FULLSET_DD_*.csv')
    if dd:
        out.append(('ICOS_WW2020', dd, dd.replace('_DD_', '_MM_'), dd.replace('_DD_', '_YY_'),
                    first(f'{B}/ICOS_WW2020/halfhourly/FLX_{site}_FLUXNET2015_FULLSET_H[HR]_*.csv')))
    dd = first(f'{B}/ICOS_L2/daily_monthly_yearly/ICOSETC_{site}_FLUXNET_DD_L2.csv')
    if dd:
        out.append(('ICOS_L2', dd, dd.replace('_DD_', '_MM_'), dd.replace('_DD_', '_YY_'),
                    first(f'{B}/ICOS_L2/halfhourly/ICOSETC_{site}_FLUXNET_H[HR]_L2.csv')))
    dd = first(f'{B}/FLUXNET_v1.3/daily_monthly_yearly/*_{site}_FLUXNET_FLUXMET_DD_*.csv')
    if dd:
        hh = first(f'{B}/FLUXNET_v1.3/halfhourly/*_{site}_FLUXNET_FLUXMET_H[HR]_*.csv')
        if not hh:
            for z in sorted(glob.glob(f'{B}/FLUXNET_v1.3/_zips/*_{site}_FLUXNET_*.zip')):
                try:
                    m = [n for n in zipfile.ZipFile(z).namelist() if re.search(r'_FLUXMET_H[HR]_', n)]
                except zipfile.BadZipFile: continue
                if m: hh = (z, m[0]); break
        out.append(('FLUXNET_v1.3', dd, dd.replace('_DD_', '_MM_'), dd.replace('_DD_', '_YY_'), hh))
    return out


def good_years(dd_path):
    d = pd.read_csv(dd_path, na_values=[-9999, -9999.0])
    var = 'NEE_VUT_REF' if 'NEE_VUT_REF' in d and d['NEE_VUT_REF'].notna().sum() > 0 else ('NEE_CUT_REF' if 'NEE_CUT_REF' in d else None)
    if var is None: return var, {}, d
    y = d['TIMESTAMP'].astype(str).str[:4].astype(int)
    qc = d.get(f'{var}_QC', pd.Series(1.0, index=d.index))
    g = {}
    for yr, idx in d.groupby(y).groups.items():
        n = int(d.loc[idx, var].notna().sum()); q = float(qc.loc[idx].where(d.loc[idx, var].notna()).mean()) if n else 0.0
        g[int(yr)] = (n >= 240 and q >= 0.5)            # 240: high-latitude records leave the winter unfilled (RU-SkP)
    return var, g, d


def window(g):
    yrs = sorted(y for y in g if Y_MIN <= y <= Y_MAX)
    best = None
    for L in (5, 4, 3):
        for y0 in yrs:
            y1 = y0 + L - 1
            if y1 not in g or y1 > Y_MAX or any(y not in g for y in range(y0, y1 + 1)): continue
            fit = [g[y] for y in range(y0 + 1, y1)]
            if not g[y1] or not any(fit): continue
            key = (sum(fit) + 1, L, y1)
            if best is None or key > best[0]: best = (key, y0, y1)
        if best is not None and best[0][1] == L: break
    return (best[1], best[2], best[0][0]) if best else None


def read_hh(src):
    cols = ('TIMESTAMP_START', 'TA_F_MDS', 'TA_F', 'SW_IN_F_MDS', 'SW_IN_F', 'NIGHT', 'NEE_VUT_REF', 'NEE_VUT_REF_QC', 'NEE_CUT_REF', 'NEE_CUT_REF_QC')
    if isinstance(src, tuple):
        with zipfile.ZipFile(src[0]) as z, z.open(src[1]) as f:
            return pd.read_csv(io.TextIOWrapper(f), usecols=lambda c: c in cols, na_values=[-9999, -9999.0])
    return pd.read_csv(src, usecols=lambda c: c in cols, na_values=[-9999, -9999.0])


def prep(args):
    site, meta, out = args
    try:
        recs = records(site)
        if not recs: return {'site': site, 'status': 'no daily file'}
        scored = []
        for r in recs:
            var, g, _ = good_years(r[1])
            scored.append((sum(g.values()), PRIO[r[0]], r, var, g))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        ngood, _, rec, var, g = scored[0]
        src, dd, mm, yy, hh = rec
        row = dict(meta); row.update(SITE_ID=site, SOURCE=src, NEE_VAR=var or '', N_GOOD=ngood,
                                     Y0_REC=min(g) if g else '', Y1_REC=max(g) if g else '')
        w = window(g) if g else None
        row.update(WIN_Y0=w[0] if w else '', WIN_Y1=w[1] if w else '', WIN_SCORE=w[2] if w else 0)
        d = pd.read_csv(dd)
        a, b = str(d['TIMESTAMP'].iloc[0])[:4], str(d['TIMESTAMP'].iloc[-1])[:4]
        tag = f'{a}-{b}_31'
        t = pd.read_csv(dd, na_values=[-9999, -9999.0])
        row.setdefault('MAT', np.nan); row.setdefault('MAP', np.nan)
        if 'TA_F' in t: row['MAT'] = round(float(t['TA_F'].mean()), 2)
        if 'P_F' in t: row['MAP'] = round(float(t['P_F'].mean() * 365.25), 1)
        for res, p in (('DD', dd), ('MM', mm), ('YY', yy)):
            if not os.path.exists(p): continue
            x = pd.read_csv(p)
            if var == 'NEE_CUT_REF':                                                    # the harness reads the VUT names
                for c in list(x.columns):
                    if 'CUT_REF' in c and c.replace('CUT_REF', 'VUT_REF') not in x: x[c.replace('CUT_REF', 'VUT_REF')] = x[c]
            x.to_csv(f'{out}/FLX_{site}_FLUXNET2015_FULLSET_{res}_{tag}.csv', index=False)
        if hh is not None and w:
            h = read_hh(hh)
            ts = pd.to_datetime(h['TIMESTAMP_START'].astype(np.int64).astype(str), format='%Y%m%d%H%M')
            keep = (ts.dt.year >= w[0]) & (ts.dt.year <= w[1]); h = h[keep].copy(); ts = ts[keep]
            if var == 'NEE_CUT_REF' or 'NEE_VUT_REF' not in h:
                h['NEE_VUT_REF'] = h.get('NEE_CUT_REF'); h['NEE_VUT_REF_QC'] = h.get('NEE_CUT_REF_QC')
            if 'TA_F_MDS' not in h: h['TA_F_MDS'] = h.get('TA_F')
            else: h['TA_F_MDS'] = h['TA_F_MDS'].where(h['TA_F_MDS'].notna(), h.get('TA_F'))
            o = pd.DataFrame({'TIMESTAMP': ts.dt.strftime('%Y-%m-%d %H:%M:%S').values, 'TIMESTAMP_START': h['TIMESTAMP_START'].astype(np.int64).values})
            for c in ('TA_F_MDS', 'SW_IN_F_MDS', 'SW_IN_F', 'NIGHT', 'NEE_VUT_REF', 'NEE_VUT_REF_QC'):
                o[c] = h[c].values if c in h else np.nan
            o.to_csv(f'{out}/FLX_{site}_FLUXNET2015_FULLSET_HH_{w[0]}-{w[1]}_31.csv', index=False, float_format='%.3f', na_rep='-9999')
            row['HH'] = 'yes'
        else:
            row['HH'] = 'no'
        row['status'] = 'ok'
        return row
    except Exception as e:
        return {'site': site, 'status': f'error {type(e).__name__}: {e}'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--sites', default=''); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args(); out = f'{a.out}/compact'; os.makedirs(out, exist_ok=True)
    meta = {}
    for r in csv.DictReader(open('/shared/nas2/Min/FLUXNET2015/fluxnet_compact/FLUXNET_siteinfo.csv', encoding='utf-8-sig')):
        meta[r['SITE_ID']] = dict(SITE_NAME=r['SITE_NAME'], FLUXNET2015=r['FLUXNET2015'], LOCATION_LAT=r['LOCATION_LAT'],
                                  LOCATION_LONG=r['LOCATION_LONG'], LOCATION_ELEV=r['LOCATION_ELEV'], IGBP=r['IGBP'])
    for r in csv.DictReader(open(f'{B}/new_sites_inventory.csv')):
        if r['site'] not in meta:
            meta[r['site']] = dict(SITE_NAME=r['site_name'], FLUXNET2015='', LOCATION_LAT=r['lat'], LOCATION_LONG=r['lon'],
                                   LOCATION_ELEV='', IGBP=r['igbp'])
    sites = a.sites.split(',') if a.sites else sorted(meta)
    with Pool(a.workers) as p:
        rows = p.map(prep, [(s, meta.get(s, {}), out) for s in sites], chunksize=1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pft_koppen_design as D
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(D.KOP))
    def kclass(lat, lon):
        i = min(int((90.0 - lat) * 120), img.shape[0] - 1); j = min(int((lon + 180.0) * 120), img.shape[1] - 1); v = int(img[i, j])
        if v == 0:
            w = img[max(i - 6, 0):i + 7, max(j - 6, 0):j + 7]; vals = w[w > 0]; v = int(np.bincount(vals).argmax()) if vals.size else 0
        return D.LEG.get(v, 'none')
    ok = [r for r in rows if r.get('status') == 'ok']
    for r in ok:
        k = kclass(float(r['LOCATION_LAT']), float(r['LOCATION_LONG'])); z = D.ZONE.get(k, 'none')
        t = D.assign(r['IGBP'], z, r['SITE_ID']); r.update(KOPPEN=k, ZONE=z, TYPE=' '.join(t))
    cols = ['SITE_ID', 'SITE_NAME', 'FLUXNET2015', 'FLUXNET-CH4', 'LOCATION_LAT', 'LOCATION_LONG', 'LOCATION_ELEV', 'IGBP', 'MAT', 'MAP',
            'SOURCE', 'NEE_VAR', 'Y0_REC', 'Y1_REC', 'N_GOOD', 'WIN_Y0', 'WIN_Y1', 'WIN_SCORE', 'HH', 'KOPPEN', 'ZONE', 'TYPE']
    pd.DataFrame([{c: r.get(c, '') for c in cols} for r in ok]).to_csv(f'{out}/FLUXNET_siteinfo.csv', index=False)
    bad = [r for r in rows if r.get('status') != 'ok']
    print(f'{len(ok)} towers written, {len(bad)} skipped: {[(r["site"], r["status"]) for r in bad][:20]}')
    cal = [r for r in ok if r.get('WIN_SCORE', 0) and int(r['WIN_SCORE']) >= 2 and r.get('HH') == 'yes']
    print(f'calibration-grade (window with a good held-out year and >= 1 good fitted year, half-hourly file): {len(cal)}')


if __name__ == '__main__':
    main()
