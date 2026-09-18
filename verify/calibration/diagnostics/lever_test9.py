"""Savanna dry-season GPP by canopy component: counterfactual removals on the (b1) structure (CALIBRATION.md 16)."""
import sys, glob, re, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at, fluxcom_monthly_at, PFT_NAMES
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v12c/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]
ia=fl[sel_all]//720; ja=fl[sel_all]%720
gobs=fluxcom_at(ia,ja); om_all=fluxcom_monthly_at(ia,ja); amp=om_all.max(0)/np.maximum(om_all.min(0),1e-9)
trop=(np.abs(LATV[ia])<23.5)&np.isfinite(gobs)&(gobs>0.5)
rng=np.random.default_rng(1); idx=np.where(trop&(amp>=3))[0]; sel=sel_all[rng.choice(idx,60,replace=False)]   # same 60 as lever test 7
fs=sorted(glob.glob(f'{V}/val_v12a/monthly/out/ibis_monthly_*.nc')); mon=np.array([int(re.search(r'_(\d\d)\.nc$',f).group(1)) for f in fs])
P=np.zeros((12,360,720)); n=np.zeros(12)
for f,mm in zip(fs,mon):
    with xr.open_dataset(f) as d: P[mm-1]+=np.nan_to_num(np.asarray(d['amprcp'].values,float)); n[mm-1]+=1
P/=n[:,None,None]
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
consts=PhysicalConstants(); reader=GSWP3Reader(GSWP3)
site_transient.apply_co2=lambda s_,y,series=None: eqx.tree_at(lambda q:q.atm.co2conc,s_,389.8e-6)
real_exist=site_transient.existence
i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32)
obs_m=fluxcom_monthly_at(i,j); area=np.cos(np.deg2rad(LATV[i])); w=area/area.sum(); wet=np.argmax(P[:,i,j],axis=0)
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st0=_cast_tree_f32(jax.tree.map(cut,full)); st0=eqx.tree_at(lambda s_: s_.stress, st0, create_stress_accum(len(sel)), is_leaf=lambda x: x is None)
sp0,pft0,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v12c/params_fit_v12c.yaml'))
v=np.asarray(sp0.stomata_params.vmax_pft); s_=np.asarray(pft0.specla); rd=CachedForcing(reader,[2001,2002,2003],i,j)
of=obs_m/np.maximum(obs_m.sum(0),1e-9); co=np.zeros(12)
for k in range(len(sel)): co+=w[k]*np.roll(of[:,k],-wet[k])
print(f"SAVANNA n={len(sel)}  obs GPP {np.sum(w*fluxcom_at(i,j)):.2f}  mean |lat| {np.mean(np.abs(LATV[i])):.0f}")
print("months from wettest   : "+" ".join(f"{t:5d}" for t in range(12)))
print("obs GPP frac          : "+" ".join(f"{x:5.3f}" for x in co), flush=True)

REC=[]; real_rm=site_transient.run_month_gswp3
def rec_rm(state,*a,**k):
    out=real_rm(state,*a,**k); st=out[0]
    REC.append((np.asarray(st.vegetation.lai).copy(), np.asarray(st.vegetation.dropu).copy(), np.asarray(st.vegetation.dropl4).copy(),
                np.asarray(st.soil.stresstu).copy(), np.asarray(st.soil.stresstl).copy()))
    return out
site_transient.run_month_gswp3=rec_rm
cases={"as fitted (trigger off)":{}, "trigger 0.7":{"drop_stress_thresh":0.7}, "trigger 0.7 + stressfac -2.5":{"drop_stress_thresh":0.7,"stressfac":-2.5}, "trigger 0.9":{"drop_stress_thresh":0.9}}
def cmp12(x):   # x (12, n) -> composite by month from wettest
    out=np.zeros(12)
    for k in range(len(sel)): out+=w[k]*np.roll(x[:,k],-wet[k])
    return out
for nm,ex in cases.items():
    REC.clear(); st=st0; sp,pft=apply_params(sp0,pft0,v,s_,ex); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
    key=jax.random.PRNGKey(0); acc=[]
    for n_ in range(6):
        st,key,d=one_year(st,key,2001+(n_%3),ctx,1)
        if n_>=3: acc.append(d['gpp_monthly'])
    mm=np.mean(acc,axis=0); gf=mm/np.maximum(mm.sum(0),1e-9)
    R=REC[-36:]; get=lambda q: np.mean([np.stack([R[y*12+m_][q] for m_ in range(12)]) for y in range(3)],axis=0)   # (12, n[, 2])
    lai=get(0); dropu=get(1); dropl4=get(2); stu=get(3); stl=get(4)
    ann=np.sum(area*mm.sum(0))/np.sum(area*obs_m.sum(0))
    print(f"\n{nm}   [annual model/obs {ann:4.2f}  amplitude {(cmp12(gf).max()/cmp12(gf).min()):4.1f} (obs {(co.max()/co.min()):4.1f})]")
    print("  months from wettest : "+" ".join(f"{t:5d}" for t in range(12)))
    print("  GPP model/obs shape : "+" ".join(f"{x:5.2f}" for x in cmp12(gf)/np.maximum(co,1e-9)))
    print("  displayed LAI upper : "+" ".join(f"{x:5.2f}" for x in cmp12(lai[:,:,1])))
    print("  displayed LAI lower : "+" ".join(f"{x:5.2f}" for x in cmp12(lai[:,:,0])))
    print("  dropu (trop decid)  : "+" ".join(f"{x:5.2f}" for x in cmp12(dropu)))
    print("  dropl4 (C4 grass)   : "+" ".join(f"{x:5.2f}" for x in cmp12(dropl4)))
    print("  stress upper/lower  : "+" ".join(f"{x:5.2f}" for x in cmp12(stu))+"  |  "+" ".join(f"{x:5.2f}" for x in cmp12(stl)), flush=True)
