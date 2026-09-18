"""Criterion 3 diagnosis: where and why the tropical GPP cycle lags FLUXCOM (v12a monthly output)."""
import sys, glob, re, numpy as np, xarray as xr, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, LONV, MASK, fluxcom_at, fluxcom_monthly_at
V='/shared/nas2/Hangkai/ibis/verify_outputs'; OUT=sys.argv[1] if len(sys.argv)>1 else f'{V}/val_v12a/monthly/out'
fs=sorted(glob.glob(f'{OUT}/ibis_monthly_*.nc')); assert len(fs)>=24, len(fs)
mon=np.array([int(re.search(r'_(\d\d)\.nc$',f).group(1)) for f in fs])
def clim(var):
    acc=np.zeros((12,360,720)); n=np.zeros(12)
    for f,m in zip(fs,mon):
        with xr.open_dataset(f) as d: acc[m-1]+=np.nan_to_num(np.asarray(d[var].values,float)); n[m-1]+=1
    return acc/n[:,None,None]
G=clim('amgpptot'); P=clim('amprcp')
m=np.array(xr.open_dataset(MASK).landfrac.values)>0
i,j=np.where(m); trop=np.abs(LATV[i])<23.5; i,j=i[trop],j[trop]
gobs_ann=fluxcom_at(i,j); ok=np.isfinite(gobs_ann)&(gobs_ann>0.5)
i,j=i[ok],j[ok]; gobs_ann=gobs_ann[ok]; a=np.cos(np.deg2rad(LATV[i]))
om=fluxcom_monthly_at(i,j)                       # (12,n) obs monthly totals
gm=G[:,i,j]; pm=P[:,i,j]                          # model monthly GPP, precip
def frac(x): s=x.sum(axis=0); return x/np.where(s>0,s,np.nan)
of,gf,pf=frac(om),frac(gm),frac(pm)
def peak(x): return np.argmax(x,axis=0)
def circ(d): return (d+6)%12-6
off=circ(peak(gf)-peak(of)); off_p_obs=circ(peak(of)-peak(pf)); off_p_mod=circ(peak(gf)-peak(pf))
amp=lambda x: x.max(axis=0)/np.maximum(x.min(axis=0),1e-9)
ao,am,ap=amp(om),amp(gm),amp(pm)
def lagcorr(x,y):   # best circular lag of x behind y, per cell, from monthly fractions
    best=np.zeros(x.shape[1]); bc=np.full(x.shape[1],-9.0)
    for L in range(-5,7):
        c=np.array([np.corrcoef(np.roll(x[:,k],-L),y[:,k])[0,1] if np.isfinite(x[:,k]).all() and np.isfinite(y[:,k]).all() and x[:,k].std()>0 and y[:,k].std()>0 else np.nan for k in range(x.shape[1])])
        better=np.nan_to_num(c,nan=-9)>bc; best[better]=L; bc[better]=np.nan_to_num(c,nan=-9)[better]
    return best,bc
lag_obs,c_obs=lagcorr(of,pf); lag_mod,c_mod=lagcorr(gf,pf)
print(f"{OUT}\ntropical land cells (|lat|<23.5, FLUXCOM>0.5): {len(i)}; {len(fs)} monthly files")
classes={"rainforest (obs amp<1.5)":ao<1.5,"seasonal (1.5-3)":(ao>=1.5)&(ao<3),"savanna/dry (>=3)":ao>=3}
for nm,k in classes.items():
    w=a[k]/a[k].sum()
    print(f"\n== {nm}: n={k.sum()}  obs GPP {np.sum(w*gobs_ann[k]):.2f}  model/obs annual {np.sum(w*gm[:,k].sum(0))/np.sum(w*om[:,k].sum(0)):.2f}")
    print(f"   peak offset model-obs: mean {np.sum(w*off[k]):+.2f} mo, |off|>=2: {100*np.sum(w*(np.abs(off[k])>=2)):.0f}%   amplitude obs {np.sum(w*np.minimum(ao[k],20)):.1f} model {np.sum(w*np.minimum(am[k],20)):.1f} precip {np.sum(w*np.minimum(ap[k],20)):.1f}")
    print(f"   GPP peak relative to PRECIP peak: obs {np.sum(w*off_p_obs[k]):+.2f} mo, model {np.sum(w*off_p_mod[k]):+.2f} mo;  best lag behind rain: obs {np.sum(w*lag_obs[k]):+.2f} (r {np.sum(w*c_obs[k]):.2f}), model {np.sum(w*lag_mod[k]):+.2f} (r {np.sum(w*c_mod[k]):.2f})")
    # composite cycles aligned to each cell's precipitation peak (month 0 = wettest)
    def comp(x):
        out=np.zeros(12)
        for kk,idx in enumerate(np.where(k)[0]): out+=w[kk]*np.roll(x[:,idx],-peak(pf)[idx])
        return out
    co,cg,cp=comp(of),comp(gf),comp(pf)
    print("   months from wettest:  "+" ".join(f"{t:5d}" for t in range(12)))
    print("   precip frac         : "+" ".join(f"{x:5.3f}" for x in cp))
    print("   obs GPP frac        : "+" ".join(f"{x:5.3f}" for x in co))
    print("   model GPP frac      : "+" ".join(f"{x:5.3f}" for x in cg))
    print("   model/obs           : "+" ".join(f"{x:5.2f}" for x in cg/np.maximum(co,1e-9)))
print("\n== by hemisphere/continent (all tropical cells) ==")
for nm,k in {"S America":(LONV[j]<-30),"Africa":(LONV[j]>=-30)&(LONV[j]<60),"Asia/Aus":LONV[j]>=60,"N hemisphere":LATV[i]>0,"S hemisphere":LATV[i]<=0}.items():
    w=a[k]/a[k].sum(); print(f"  {nm:14s} n={k.sum():5d} offset {np.sum(w*off[k]):+.2f} mo  |off|>=2 {100*np.sum(w*(np.abs(off[k])>=2)):.0f}%  model-vs-rain lag {np.sum(w*lag_mod[k]):+.2f} obs-vs-rain {np.sum(w*lag_obs[k]):+.2f}")
