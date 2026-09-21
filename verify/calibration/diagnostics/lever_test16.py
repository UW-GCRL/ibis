"""Lever test 16: the tropical dry-season composite on the FIXED phenology clock (CALIBRATION.md 16): joint-fit parameters, dormancy trigger as fitted (0.277) and at 0.5 / 0.7."""
import sys, glob, re, numpy as np, xarray as xr, netCDF4 as nc, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, LONV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at, fluxcom_monthly_at
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v12g/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]
ia=fl[sel_all]//720; ja=fl[sel_all]%720
gobs=fluxcom_at(ia,ja); om_all=fluxcom_monthly_at(ia,ja); amp=om_all.max(0)/np.maximum(om_all.min(0),1e-9)
trop=(np.abs(LATV[ia])<23.5)&np.isfinite(gobs)&(gobs>0.5)
rng=np.random.default_rng(1)
def pick(mask,n): idx=np.where(mask)[0]; return sel_all[rng.choice(idx,min(n,len(idx)),replace=False)]
SETS={"SAVANNA (amp>=3)":pick(trop&(amp>=3),60),"SEASONAL (1.5-3)":pick(trop&(amp>=1.5)&(amp<3),40),"RAINFOREST (<1.5)":pick(trop&(amp<1.5),30)}
# wettest month per cell from the v12a monthly precipitation climatology (fixed forcing)
fs=sorted(glob.glob(f'{V}/val_v12a/monthly/out/ibis_monthly_*.nc')); mon=np.array([int(re.search(r'_(\d\d)\.nc$',f).group(1)) for f in fs])
P=np.zeros((12,360,720)); n=np.zeros(12)
for f,mm in zip(fs,mon):
    with xr.open_dataset(f) as d: P[mm-1]+=np.nan_to_num(np.asarray(d['amprcp'].values,float)); n[mm-1]+=1
P/=n[:,None,None]
cases={"joint fit as fitted (trigger 0.277)":{}, "trigger 0.5":{"drop_stress_thresh":0.5}, "trigger 0.7":{"drop_stress_thresh":0.7}}
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
rd_years=[2001,2002,2003]; consts=PhysicalConstants(); reader=GSWP3Reader(GSWP3)
site_transient.apply_co2=lambda s_,y,series=None: eqx.tree_at(lambda q:q.atm.co2conc,s_,389.8e-6)
for setname,sel in SETS.items():
    i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32)
    obs_m=fluxcom_monthly_at(i,j); area=np.cos(np.deg2rad(LATV[i])); w=area/area.sum(); wet=np.argmax(P[:,i,j],axis=0)
    cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
    st0=_cast_tree_f32(jax.tree.map(cut,full)); st0=eqx.tree_at(lambda s_: s_.stress, st0, create_stress_accum(len(sel)), is_leaf=lambda x: x is None)
    sp0,pft0,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v12g/params_fit_v12g.yaml'))
    v=np.asarray(sp0.stomata_params.vmax_pft); s_=np.asarray(pft0.specla); rd=CachedForcing(reader,rd_years,i,j)
    of=obs_m/np.maximum(obs_m.sum(0),1e-9); co=np.zeros(12)
    for k in range(len(sel)): co+=w[k]*np.roll(of[:,k],-wet[k])
    print(f"\n################ {setname}  n={len(sel)}  obs GPP {np.sum(w*gobs[np.isin(sel_all,sel)][np.argsort(np.argsort(sel))]) if False else np.sum(w*fluxcom_at(i,j)):.2f} ################")
    print("months from wettest   : "+" ".join(f"{t:5d}" for t in range(12)))
    print("obs GPP frac          : "+" ".join(f"{x:5.3f}" for x in co), flush=True)
    for nm,ex in cases.items():
        sp,pft=apply_params(sp0,pft0,v,s_,ex); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
        st=st0; key=jax.random.PRNGKey(0); acc=[]
        for n_ in range(6):
            st,key,d=one_year(st,key,2001+(n_%3),ctx,1)
            if n_>=3: acc.append(d['gpp_monthly'])
        mm=np.mean(acc,axis=0); gf=mm/np.maximum(mm.sum(0),1e-9); cg=np.zeros(12)
        for k in range(len(sel)): cg+=w[k]*np.roll(gf[:,k],-wet[k])
        ann=np.sum(area*mm.sum(0))/np.sum(area*obs_m.sum(0)); vg=st.vegetation
        print(f"\n{nm}   [annual model/obs {ann:4.2f}  amp obs {(co.max()/co.min()):4.1f} model {(cg.max()/cg.min()):4.1f}  fu {np.asarray(vg.fu).mean():.2f} fl {np.asarray(vg.fl).mean():.2f}  dropu {np.asarray(vg.dropu).mean():.2f} dropl4 {np.asarray(vg.dropl4).mean():.2f}]")
        print("  model GPP frac      : "+" ".join(f"{x:5.3f}" for x in cg))
        print("  model/obs (shape)   : "+" ".join(f"{x:5.2f}" for x in cg/np.maximum(co,1e-9)), flush=True)
