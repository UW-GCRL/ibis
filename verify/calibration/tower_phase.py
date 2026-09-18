"""Tropical savanna / grassland seasonality at FLUXNET2015 towers: tower GPP vs FLUXCOM vs the model.

    ibis-python verify/calibration/tower_phase.py [<monthly output dir with ibis_monthly_*.nc>]

Sites: IGBP SAV / WSA / GRA / OSH / CSH with |lat| < 30 and monthly
GPP_NT_VUT_REF and P_F in the compact FLUXNET2015 files. Each site is
aligned to its own wettest month; FLUXCOM and the model are read at the
tower's 0.5-degree cell. Reports per site the amplitude and the GPP-peak
offset from the rain peak for tower / FLUXCOM / model, then the composite
cycles. Negative (night-time-partitioned) tower months are clipped at zero
before normalising. CALIBRATION.md 16: towers confirm FLUXCOM's phase and
dispute its dry-season depth (FLUXCOM 0.6x towers), model +1.1 months late.
"""
import sys, glob, re, numpy as np, pandas as pd, xarray as xr, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, LONV, MASK, fluxcom_monthly_at
F='/shared/nas2/Min/FLUXNET2015/fluxnet_compact'; V='/shared/nas2/Hangkai/ibis/verify_outputs'
meta=pd.read_csv(f'{F}/FLUXNET_siteinfo.csv',encoding='utf-8-sig').set_index('SITE_ID')
igbp_col=[c for c in meta.columns if 'IGBP' in c.upper()]
OUT=sys.argv[1] if len(sys.argv)>1 else f'{V}/val_v12a/monthly/out'
fs=sorted(glob.glob(f'{OUT}/ibis_monthly_*.nc')); mon=np.array([int(re.search(r'_(\d\d)\.nc$',f).group(1)) for f in fs])
G=np.zeros((12,360,720)); n=np.zeros(12)
for f,mm in zip(fs,mon):
    with xr.open_dataset(f) as d: G[mm-1]+=np.nan_to_num(np.asarray(d['amgpptot'].values,float)); n[mm-1]+=1
G/=n[:,None,None]
m=np.array(xr.open_dataset(MASK).landfrac.values)>0
rows=[]
for path in sorted(glob.glob(f'{F}/FLX_*_FULLSET_MM_*.csv')):
    site=path.split('FLX_')[1].split('_')[0]
    if site not in meta.index: continue
    lat=float(meta.loc[site,'LOCATION_LAT']); lon=float(meta.loc[site,'LOCATION_LONG'])
    if abs(lat)>30: continue
    veg=str(meta.loc[site,igbp_col[0]]) if igbp_col else '?'
    if igbp_col and veg not in ('SAV','WSA','GRA','OSH','CSH'): continue
    df=pd.read_csv(path); 
    if 'GPP_NT_VUT_REF' not in df or 'P_F' not in df: continue
    df=df[(df['GPP_NT_VUT_REF']>-9000)&(df['P_F']>-9000)].copy(); ts=pd.to_datetime(df['TIMESTAMP'].astype(str).str[:10]); df['mo']=ts.dt.month; df['yr']=ts.dt.year
    if len(df)<24: continue
    g=np.maximum(df.groupby("mo")["GPP_NT_VUT_REF"].mean().reindex(range(1,13)).values,0.0); p=df.groupby('mo')['P_F'].mean().reindex(range(1,13)).values
    if np.isnan(g).any() or np.isnan(p).any() or g.sum()<=0: continue
    i=int(np.argmin(np.abs(LATV-lat))); j=int(np.argmin(np.abs(LONV-lon)))
    if not m[i,j]: continue
    fc=fluxcom_monthly_at(np.array([i]),np.array([j]))[:,0]; md=G[:,i,j]
    wet=int(np.argmax(p)); roll=lambda x: np.roll(x,-wet)
    gt,gf,gm,pp=roll(g/g.sum()),roll(fc/fc.sum()),roll(md/md.sum()),roll(p/p.sum())
    circ=lambda k:(k+6)%12-6
    rows.append(dict(site=site,veg=veg,lat=lat,yrs=df['yr'].nunique(),
                     amp_t=g.max()/max(g.min(),1e-9),amp_f=fc.max()/max(fc.min(),1e-9),amp_m=md.max()/max(md.min(),1e-9),
                     pk_t=circ(np.argmax(gt)),pk_f=circ(np.argmax(gf)),pk_m=circ(np.argmax(gm)),gt=gt,gf=gf,gm=gm,pp=pp))
print(f"\n{len(rows)} tropical/subtropical savanna-grassland-shrub tower sites (|lat|<30, monthly GPP_NT_VUT_REF & P_F)")
print(f"{'site':7s}{'veg':5s}{'lat':>6s}{'yrs':>4s} | amplitude tower/FLUXCOM/model | GPP peak minus rain peak (months): tower / FLUXCOM / model")
for r in rows: print(f"{r['site']:7s}{r['veg']:5s}{r['lat']:6.1f}{r['yrs']:4d} | {min(r['amp_t'],30):5.1f} {min(r['amp_f'],30):5.1f} {min(r['amp_m'],30):5.1f}           | {r['pk_t']:+3d} / {r['pk_f']:+3d} / {r['pk_m']:+3d}")
if rows:
    W=np.array([r['pp'] for r in rows]).mean(0); T=np.array([r['gt'] for r in rows]).mean(0); Fx=np.array([r['gf'] for r in rows]).mean(0); M=np.array([r['gm'] for r in rows]).mean(0)
    print("\ncomposite, months from the tower's wettest:  "+" ".join(f"{t:5d}" for t in range(12)))
    for nm,x in (("precip frac",W),("TOWER GPP frac",T),("FLUXCOM frac",Fx),("MODEL frac",M)): print(f"  {nm:16s}"+" ".join(f"{v:5.3f}" for v in x))
    print(f"  model/tower     "+" ".join(f"{v:5.2f}" for v in M/np.maximum(T,1e-9)))
    print(f"  FLUXCOM/tower   "+" ".join(f"{v:5.2f}" for v in Fx/np.maximum(T,1e-9)))
    print(f"\nmean GPP-peak offset from rain: tower {np.mean([r['pk_t'] for r in rows]):+.2f}, FLUXCOM {np.mean([r['pk_f'] for r in rows]):+.2f}, model {np.mean([r['pk_m'] for r in rows]):+.2f} months; mean amplitude tower {np.mean([min(r['amp_t'],30) for r in rows]):.1f} FLUXCOM {np.mean([min(r['amp_f'],30) for r in rows]):.1f} model {np.mean([min(r['amp_m'],30) for r in rows]):.1f}")
