"""Savanna cells: model soil water by layer and ET vs the ET ensemble, aligned to the wettest month (v12a monthly output)."""
import sys, glob, re, numpy as np, xarray as xr, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, fluxcom_at, fluxcom_monthly_at
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v12c/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]
ia=fl[sel_all]//720; ja=fl[sel_all]%720
gobs=fluxcom_at(ia,ja); om_all=fluxcom_monthly_at(ia,ja); amp=om_all.max(0)/np.maximum(om_all.min(0),1e-9)
trop=(np.abs(LATV[ia])<23.5)&np.isfinite(gobs)&(gobs>0.5)
rng=np.random.default_rng(1); idx=np.where(trop&(amp>=3))[0]; sel=sel_all[rng.choice(idx,60,replace=False)]   # same 60 cells
i=(fl[sel]//720); j=(fl[sel]%720); a=np.cos(np.deg2rad(LATV[i])); w=a/a.sum()
fs=sorted(glob.glob(f'{V}/val_v12a/monthly/out/ibis_monthly_*.nc')); mon=np.array([int(re.search(r'_(\d\d)\.nc$',f).group(1)) for f in fs])
with xr.open_dataset(fs[0]) as d0: print("wsoi dims:", d0['wsoi'].dims, d0['wsoi'].shape, " amaet units:", d0['amaet'].attrs.get('units','?'), " amprcp units:", d0['amprcp'].attrs.get('units','?'))
acc={}; n=np.zeros(12)
for f,mm in zip(fs,mon):
    with xr.open_dataset(f) as d:
        for k in ('wsoi','amaet','amprcp','amgpptot'):
            x=np.asarray(d[k].values,float)
            x=x[i,j,:].T if x.ndim==3 else x[i,j]          # wsoi is (lat, lon, soilay) -> (soilay, n)
            acc.setdefault(k,np.zeros((12,)+x.shape))[mm-1]+=np.nan_to_num(x)
    n[mm-1]+=1
for k in acc: acc[k]/=n.reshape((12,)+(1,)*(acc[k].ndim-1))
P=acc['amprcp']; wet=np.argmax(P,axis=0)
def comp(x):   # x (12, n) -> area-weighted composite aligned to wettest
    out=np.zeros(12)
    for k in range(len(sel)): out+=w[k]*np.roll(x[:,k],-wet[k])
    return out
e=np.load(f'{V.replace("verify_outputs","inputs")}/eval/et/et_ensemble_0.5deg.npz')
eto=e['ens_clim_mean'][:,i,j]*30.44                       # mm/day -> mm/month
etmn=e['ens_clim_min'][:,i,j]*30.44; etmx=e['ens_clim_max'][:,i,j]*30.44
etm=acc['amaet']; pm=P; gm=acc['amgpptot']; om=fluxcom_monthly_at(i,j)
print(f"savanna n={len(sel)}; annual: precip {comp(pm).sum():.0f} mm, ET model {comp(etm).sum():.0f}, ET ensemble {comp(eto).sum():.0f} [{comp(etmn).sum():.0f}-{comp(etmx).sum():.0f}] mm")
print("months from wettest   : "+" ".join(f"{t:6d}" for t in range(12)))
print("precip mm/mo          : "+" ".join(f"{x:6.0f}" for x in comp(pm)))
print("ET model mm/mo        : "+" ".join(f"{x:6.0f}" for x in comp(etm)))
print("ET ensemble mean      : "+" ".join(f"{x:6.0f}" for x in comp(eto)))
print("ET ensemble min..max  : "+" ".join(f"{x:6.0f}" for x in comp(etmn))+"\n                        "+" ".join(f"{x:6.0f}" for x in comp(etmx)))
print("ET model/ensemble     : "+" ".join(f"{x:6.2f}" for x in comp(etm)/np.maximum(comp(eto),1e-9)))
print("GPP model/FLUXCOM shape: "+" ".join(f"{x:6.2f}" for x in (comp(gm)/comp(gm).sum())/np.maximum(comp(om)/comp(om).sum(),1e-9)))
ws=acc['wsoi']
if ws.ndim==3:
    for L in range(ws.shape[1]):
        print(f"wsoi layer {L+1} (frac sat): "+" ".join(f"{x:6.2f}" for x in comp(ws[:,L,:])))
else:
    print("wsoi (frac sat)       : "+" ".join(f"{x:6.2f}" for x in comp(ws)))
