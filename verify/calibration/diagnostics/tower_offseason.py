"""Is FLUXCOM's near-zero off-season GPP in semi-arid cells real? Ask the towers."""
import sys, glob, os, numpy as np, pandas as pd, xarray as xr, jax, jax.numpy as jnp, equinox as eqx
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, LONV, MASK, GSWP3, NPOI_FULL, CachedForcing, fluxcom_monthly_at
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
F='/shared/nas2/Min/FLUXNET2015/fluxnet_compact'; V='/shared/nas2/Hangkai/ibis/verify_outputs'
SITES=["US-SRM","US-SRC","US-SRG","US-Wkg","US-Whs","US-Cop","AU-Cpr","AU-ASM","AU-Stp","AU-DaS","SD-Dem","CN-Cng","CN-Du2","CN-Sw2","ES-LJu","IT-Noe","ZA-Kru","US-Ton","US-Var","US-Me6"]
meta=pd.read_csv(f'{F}/FLUXNET_siteinfo.csv',encoding='utf-8-sig').set_index('SITE_ID')
rows=[]
for s in SITES:
    fs=glob.glob(f'{F}/FLX_{s}_*_MM_*.csv')
    if not fs or s not in meta.index: continue
    d=pd.read_csv(fs[0]); ts=d['TIMESTAMP'].astype(str).str.replace('-','').str[:6]; d['t']=pd.to_datetime(ts,format='%Y%m',errors='coerce'); d=d.dropna(subset=['t'])
    g=pd.to_numeric(d['GPP_NT_VUT_REF'],errors='coerce'); g[g<-9000]=np.nan
    clim=g.groupby(d['t'].dt.month).mean()                       # gC/m2/day
    if clim.notna().sum()<12: continue
    la,lo=float(meta.loc[s,'LOCATION_LAT']),float(meta.loc[s,'LOCATION_LONG'])
    rows.append(dict(site=s,igbp=meta.loc[s,'IGBP'],lat=la,lon=lo,i=int(np.argmin(np.abs(LATV-la))),j=int(np.argmin(np.abs(LONV-lo))),tower=clim.values*np.array([31,28.25,31,30,31,30,31,31,30,31,30,31])/1000.0))  # kgC/m2/mo
T=pd.DataFrame(rows); print(f"{len(T)} semi-arid towers with 12-month GPP climatology")
i=T.i.values.astype(np.int32); j=T.j.values.astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32)
fc=fluxcom_monthly_at(i,j)                                      # (12,n) kgC/m2/mo
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]; pos={g:p for p,g in enumerate(fl)}
sel=np.array([pos.get(a*720+b,-1) for a,b in zip(i,j)]); ok=sel>=0; T=T[ok].reset_index(drop=True); sel=sel[ok]; i=i[ok]; j=j[ok]; lat_rad=lat_rad[ok]; fc=fc[:,ok]
full,_=load_state(f'{V}/val_v11d/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st=_cast_tree_f32(jax.tree.map(cut,full)); del full
sp,pft,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v11d/params_fit_v11d.yaml'))
ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=PhysicalConstants(),reader=CachedForcing(GSWP3Reader(GSWP3),[2001,2002,2003],i,j))
site_transient.apply_co2=lambda s_,y,series=None: eqx.tree_at(lambda q:q.atm.co2conc,s_,389.8e-6)
key=jax.random.PRNGKey(0); acc=[]
for n in range(4):
    st,key,d=one_year(st,key,2001+(n%3),ctx,1)
    if n>=1: acc.append(d['gpp_monthly'])
mod=np.mean(acc,axis=0)
tw=np.stack(T.tower.values,axis=1)                               # (12,n)
def season(x,lat):  # growing = 4 warmest-obs months per site by tower climatology; off = other 8
    return x
print(f"\n{'site':<7}{'igbp':<5}{'lat':>6} | annual kgC/m2: {'tower':>6}{'FLXCOM':>7}{'model':>6} | off-season (8 lowest tower months) mean kgC/m2/mo: {'tower':>6}{'FLXCOM':>7}{'model':>6} | model/tower {'ann':>5}{'off':>5}")
agg=[]
for k,r in T.iterrows():
    order=np.argsort(tw[:,k]); off=order[:8]
    ta,fa,ma=tw[:,k].sum(),fc[:,k].sum(),mod[:,k].sum(); to,fo,mo=tw[off,k].mean(),fc[off,k].mean(),mod[off,k].mean()
    agg.append((ta,fa,ma,to,fo,mo))
    print(f"{r.site:<7}{r.igbp:<5}{r.lat:6.1f} |                {ta:6.2f}{fa:7.2f}{ma:6.2f} |                                                {to:6.3f}{fo:7.3f}{mo:6.3f} |             {ma/ta:5.2f}{mo/max(to,1e-6):5.2f}")
A=np.array(agg); print(f"\nMEAN over {len(T)} towers:  annual tower {A[:,0].mean():.2f}  FLUXCOM {A[:,1].mean():.2f}  model {A[:,2].mean():.2f}   |  off-season tower {A[:,3].mean():.3f}  FLUXCOM {A[:,4].mean():.3f}  model {A[:,5].mean():.3f}")
print(f"  FLUXCOM/tower: annual {A[:,1].mean()/A[:,0].mean():.2f}  off-season {A[:,4].mean()/A[:,3].mean():.2f}      model/tower: annual {A[:,2].mean()/A[:,0].mean():.2f}  off-season {A[:,5].mean()/A[:,3].mean():.2f}")
print("  (tower = point in a productive patch, FLUXCOM/model = 0.5-deg cell mean: expect tower > cell; the OFF-SEASON RATIO is the question)")
