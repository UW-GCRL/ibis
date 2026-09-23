"""Prescribed composition from the flux-tower site-attribute dataset (Shi, Yuan et al. 2025, ESSD 17, 117; Zenodo 12596218).

    ibis-python verify/calibration/site_attributes.py            # writes the table for all 90 sites

`prescribe_from_attributes(site)` -> dict(pfts=[IBIS indices], frac={idx: fraction of vegetated cover}, fu, fl, canopy_height, lai_max)
or None when the site has no file or no PFT cover. CLM 16-PFT cover (percent of the cell) is mapped to the 12 IBIS PFTs:

    CLM 1 NET temperate -> IBIS 3 temp_con_ever      CLM 2 NET boreal -> 5 bor_con_ever     CLM 3 NDT boreal -> 7 bor_con_dec
    CLM 4 BET tropical  -> 0 trop_bl_ever            CLM 5 BET temperate -> 2 warm_bl_ever  CLM 6 BDT tropical -> 1 trop_bl_dec
    CLM 7 BDT temperate -> 4 temp_bl_dec             CLM 8 BDT boreal -> 6 bor_bl_dec
    CLM 9 BES temperate -> 8 ever_shrub              CLM 10/11 BDS temperate/boreal -> 9 dec_shrub
    CLM 12 C3 arctic grass, 13 C3 grass -> 11 c3_grass    CLM 14 C4 grass -> 10 c4_grass     CLM 0 bare, 15 crop -> not vegetation

Upper-canopy cover `fu` = summed tree cover (CLM 1-8), lower cover `fl` = summed shrub + grass cover (9-14), both as fractions of the
cell; the IBIS upper canopy shades the lower one, so `fl` is the cover the lower canopy would have under the trees. PFTs below 5 % of the
vegetated cover are dropped (they would be seeded at 5 % otherwise).
"""
import glob, os, numpy as np, netCDF4 as nc, pandas as pd
A = '/shared/nas2/Hangkai/ibis/inputs/sites/SiteAttributes_ESSD2025'
CLM_TO_IBIS = {1: 3, 2: 5, 3: 7, 4: 0, 5: 2, 6: 1, 7: 4, 8: 6, 9: 8, 10: 9, 11: 9, 12: 11, 13: 11, 14: 10}
TREES = {0, 1, 2, 3, 4, 5, 6, 7}
IBIS_NAMES = ['trop_bl_ever', 'trop_bl_dec', 'warm_bl_ever', 'temp_con_ever', 'temp_bl_dec', 'bor_con_ever', 'bor_bl_dec', 'bor_con_dec', 'ever_shrub', 'dec_shrub', 'c4_grass', 'c3_grass']


def prescribe_from_attributes(site, min_share=0.05):
    f = glob.glob(f'{A}/{site}_*.nc')
    if not f: return None
    with nc.Dataset(f[0]) as d:
        pct = np.asarray(d['PCT_PFT'][:]).ravel() if 'PCT_PFT' in d.variables else None
        lai = float(np.asarray(d['LAI_Max'][:]).ravel()[0]) if 'LAI_Max' in d.variables else np.nan
        ch = float(np.asarray(d['Canopy_height'][:]).ravel()[0]) if 'Canopy_height' in d.variables else np.nan
        igbp = int(np.asarray(d['IGBP'][:]).ravel()[0]) if 'IGBP' in d.variables else -1
    if pct is None or not np.isfinite(pct).any() or np.nansum(pct) <= 0: return None
    pct = np.nan_to_num(pct); n = len(pct)
    ibis = np.zeros(12)
    for k in range(n):
        if k in CLM_TO_IBIS: ibis[CLM_TO_IBIS[k]] += pct[k]
    veg = ibis.sum(); crop = pct[15] if n > 15 else 0.0; bare = pct[0] if n > 0 else 0.0
    if veg <= 0: return None
    share = ibis / veg; keep = share >= min_share
    share = np.where(keep, share, 0.0); share = share / share.sum()
    pfts = [int(k) for k in np.where(keep)[0]]
    fu = float(sum(ibis[k] for k in pfts if k in TREES) / 100.0); fl = float(sum(ibis[k] for k in pfts if k not in TREES) / 100.0)
    return dict(pfts=pfts, frac={int(k): float(share[k]) for k in pfts}, fu=round(fu, 3), fl=round(fl, 3), canopy_height=ch, lai_max=lai, crop_pct=float(crop), bare_pct=float(bare), igbp=igbp)


if __name__ == '__main__':
    rows = []
    for f in sorted(glob.glob(f'{A}/*.nc')):
        s = os.path.basename(f).split('_')[0]; r = prescribe_from_attributes(s)
        if r is None: rows.append(dict(site=s, pfts='(no PFT cover)', fu=np.nan, fl=np.nan, canopy_height=np.nan, lai_max=np.nan, crop_pct=np.nan)); continue
        rows.append(dict(site=s, pfts=' + '.join(f"{IBIS_NAMES[k]} {r['frac'][k]:.2f}" for k in r['pfts']), fu=r['fu'], fl=r['fl'], canopy_height=r['canopy_height'], lai_max=r['lai_max'], crop_pct=r['crop_pct'], bare_pct=r['bare_pct']))
    df = pd.DataFrame(rows); df.to_csv(f'{A}/ibis_prescription_from_attributes.csv', index=False)
    pd.set_option('display.width', 220); print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
