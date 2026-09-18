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

def flags(sp_, leaf_only, mix):
    sp_=eqx.tree_at(lambda q: q.stomata_params.scale_leaf_only, sp_, leaf_only)
    return eqx.tree_at(lambda q: q.stomata_params.vmax_mix, sp_, mix)
NAMES12=list(PFT_NAMES); DEAD={"trop_bl_dec":"trop_bl_ever","boreal_bl_dec":"temp_bl_dec","boreal_con_dec":"boreal_con_ever","dec_shrub":"ever_shrub"}
vmix=v.copy()
for dd,ll in DEAD.items(): vmix[NAMES12.index(dd)]=vmix[NAMES12.index(ll)]
cases={"as fitted (b1)":(False,{}), "scale_leaf_only":(True,{}), "leaf_only + trigger 0.7":(True,{"drop_stress_thresh":0.7}),
       "leaf_only + trigger 0.9":(True,{"drop_stress_thresh":0.9}), "leaf_only + trigger 0.7 + stressfac -5":(True,{"drop_stress_thresh":0.7,"stressfac":-5.0})}
base=None
for nm,(lo,ex) in cases.items():
    sp,pft=apply_params(sp0,pft0,vmix,s_,ex); sp=flags(sp,lo,True); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
    st=st0; key=jax.random.PRNGKey(0); acc=[]; lais=[]
    for n_ in range(6):
        st,key,d=one_year(st,key,2001+(n_%3),ctx,1)
        if n_>=3: acc.append(d['gpp_monthly']); lais.append(d['lai'])
    mm=np.mean(acc,axis=0); gf=mm/np.maximum(mm.sum(0),1e-9); cg=np.zeros(12); cabs=np.zeros(12)
    for k in range(len(sel)): cg+=w[k]*np.roll(gf[:,k],-wet[k]); cabs+=w[k]*np.roll(mm[:,k],-wet[k])
    if base is None: base=cabs
    ann=np.sum(area*mm.sum(0))/np.sum(area*obs_m.sum(0)); vg=st.vegetation; pl=np.asarray(vg.plai)
    print(f"\n{nm}   [annual model/obs {ann:4.2f}  amplitude {(cg.max()/cg.min()):4.1f} (obs {(co.max()/co.min()):4.1f})  LAI {np.mean(lais):.2f}  fu {np.asarray(vg.fu).mean():.2f}  upper {pl[:,:8].sum(1).mean():.2f} lower {pl[:,8:].sum(1).mean():.2f}]")
    print("  model/obs (shape)   : "+" ".join(f"{x:5.2f}" for x in cg/np.maximum(co,1e-9)))
    print("  GPP vs as-fitted    : "+" ".join(f"{x:5.2f}" for x in cabs/np.maximum(base,1e-9)), flush=True)
