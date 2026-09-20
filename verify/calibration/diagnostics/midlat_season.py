"""Criterion 3 residual: the 30-60N seasonal amplitude (0.6). Which months carry the excess, by dominant PFT class, v12g monthly output."""
import sys, glob, re, numpy as np, xarray as xr, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, MODIS_BASE, NPOI_FULL, fluxcom_at, fluxcom_monthly_at, PFT_NAMES
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state; from ibis.state.model_state import create_ibis_state
V='/shared/nas2/Hangkai/ibis/verify_outputs'
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; i,j=np.where(m); lat=LATV[i]; a=np.cos(np.deg2rad(lat))
fs=sorted(glob.glob(f'{V}/val_v12g/monthly/out/ibis_monthly_*.nc')); mon=np.array([int(re.search(r'_(\d\d)\.nc$',f).group(1)) for f in fs])
G=np.zeros((12,len(i))); L=np.zeros((12,len(i))); n=np.zeros(12)
for f,mm in zip(fs,mon):
    with xr.open_dataset(f) as d: G[mm-1]+=np.nan_to_num(np.asarray(d['amgpptot'].values,float))[i,j]; L[mm-1]+=(np.nan_to_num(np.asarray(d['laiu'].values,float))+np.nan_to_num(np.asarray(d['lail'].values,float)))[i,j]; n[mm-1]+=1
G/=n[:,None]; L/=n[:,None]
om=fluxcom_monthly_at(i,j); go=fluxcom_at(i,j)
st,_=load_state(f'{V}/val_v12g/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3),with_stress=True,with_etparts=True))
plai=np.asarray(st.vegetation.plai); dom=np.argmax(plai,1); dom[plai.sum(1)<0.05]=-1
band=(lat>=30)&(lat<60)&np.isfinite(go)&(go>0.2)&np.isfinite(om).all(0)
print(f"30-60N cells with FLUXCOM>0.2: {band.sum()}")
def comp(k,lbl):
    w=a[k]*go[k]; w=w/w.sum(); gm=(G[:,k]*w).sum(1); ob=(om[:,k]*w).sum(1); lai=(L[:,k]*w).sum(1)
    amp_m=gm.max()/max(gm.min(),1e-9); amp_o=ob.max()/max(ob.min(),1e-9)
    print(f"\n{lbl:40s} n={k.sum():5d}  annual model/obs {gm.sum()/ob.sum():.2f}  amplitude model {amp_m:5.1f} obs {amp_o:5.1f}  (ratio {amp_m/amp_o:.2f})")
    print("   month        : "+" ".join(f"{q+1:5d}" for q in range(12)))
    print("   model/obs    : "+" ".join(f"{x:5.2f}" for x in gm/np.maximum(ob,1e-9)))
    print("   model GPP    : "+" ".join(f"{x:5.3f}" for x in gm)); print("   FLUXCOM      : "+" ".join(f"{x:5.3f}" for x in ob))
    print("   model LAI    : "+" ".join(f"{x:5.2f}" for x in lai))
comp(band,"ALL 30-60N")
for p_,nm in ((3,"temp_con_ever dominant"),(4,"temp_bl_dec dominant"),(5,"boreal_con_ever dominant"),(11,"c3_grass dominant"),(10,"c4_grass dominant"),(8,"ever_shrub dominant"),(9,"dec_shrub dominant")):
    k=band&(dom==p_)
    if k.sum()>=100: comp(k,nm)
