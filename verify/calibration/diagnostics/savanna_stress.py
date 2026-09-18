"""Savanna cells: plant-available water and root-weighted stress by month, from the wsoi climatology and the cells' soil properties."""
import sys, glob, re, numpy as np, xarray as xr, jax.numpy as jnp, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, NPOI_FULL, fluxcom_at, fluxcom_monthly_at
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state; from ibis.state.model_state import create_ibis_state
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v12c/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]
ia=fl[sel_all]//720; ja=fl[sel_all]%720
gobs=fluxcom_at(ia,ja); om_all=fluxcom_monthly_at(ia,ja); amp=om_all.max(0)/np.maximum(om_all.min(0),1e-9)
trop=(np.abs(LATV[ia])<23.5)&np.isfinite(gobs)&(gobs>0.5)
rng=np.random.default_rng(1); idx=np.where(trop&(amp>=3))[0]; sel=sel_all[rng.choice(idx,60,replace=False)]
i=fl[sel]//720; j=fl[sel]%720; a=np.cos(np.deg2rad(LATV[i])); w=a/a.sum()
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
swilt=np.asarray(full.soil.swilt)[sel]; sfield=np.asarray(full.soil.sfield)[sel]; poros=np.asarray(full.soil.poros)[sel] if hasattr(full.soil,'poros') else None
froot=np.asarray(full.vegetation.froot)          # (nsoilay, 2): [:,0] lower, [:,1] upper
print("froot lower canopy by layer:", np.round(froot[:,0],3), " upper:", np.round(froot[:,1],3))
print("swilt (frac sat) mean by layer:", np.round(swilt.mean(0),3), " sfield:", np.round(sfield.mean(0),3))
fs=sorted(glob.glob(f'{V}/val_v12a/monthly/out/ibis_monthly_*.nc')); mon=np.array([int(re.search(r'_(\d\d)\.nc$',f).group(1)) for f in fs])
W=np.zeros((12,6,len(sel))); P=np.zeros((12,len(sel))); n=np.zeros(12)
for f,mm in zip(fs,mon):
    with xr.open_dataset(f) as d: W[mm-1]+=np.nan_to_num(np.asarray(d['wsoi'].values,float))[i,j,:].T; P[mm-1]+=np.nan_to_num(np.asarray(d['amprcp'].values,float))[i,j]
    n[mm-1]+=1
W/=n[:,None,None]; P/=n[:,None]; wet=np.argmax(P,axis=0)
def comp(x):
    out=np.zeros(x.shape[0]) if x.ndim==2 else np.zeros((12,)+x.shape[1:-1])
    for k in range(len(sel)): out+=w[k]*np.roll(x[...,k],-wet[k],axis=0)
    return out
awc=np.clip((W-swilt.T[None])/np.maximum(sfield.T[None]-swilt.T[None],1e-6),0,1)     # (12,6,n)
print("months from wettest      : "+" ".join(f"{t:5d}" for t in range(12)))
for L in range(6): print(f"awc layer {L+1}              : "+" ".join(f"{x:5.2f}" for x in comp(awc[:,L,:])))
for sf in (-8.27,-5.0,-2.5,-0.5):
    z=(1-np.exp(sf*awc))/(1-np.exp(sf))
    stl=(z*froot[:,0][None,:,None]).sum(1); stu=(z*froot[:,1][None,:,None]).sum(1)
    print(f"stressfac {sf:5.2f}: lower canopy stress : "+" ".join(f"{x:5.2f}" for x in comp(stl))+"   | upper: "+" ".join(f"{x:5.2f}" for x in comp(stu)))
