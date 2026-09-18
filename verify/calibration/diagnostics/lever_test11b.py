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

# Lever test 11: GPP response to displayed LAI alone. Run to the end of a dry-season month, save the state,
# then run the next month from that state with veg.lai scaled (frac, plai, cover unchanged).
from ibis.timestepping.loops_gswp3 import run_month_gswp3
from ibis.io.co2 import apply_co2
sp,pft=apply_params(sp0,pft0,v,s_,{}); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
st=st0; key=jax.random.PRNGKey(0)
for n_ in range(3): st,key,d=one_year(st,key,2001+(n_%3),ctx,1)          # 3 adjust years
# the median cell's dry season: months from wettest 5-7. Pick calendar month = (wet + 6) % 12 per cell; use the modal one.
target=int(np.bincount((wet+6)%12).argmax()); print(f"dry-season test month (0-based, modal (wettest+6)): {target}; cells with that dry month: {(((wet+6)%12)==target).sum()} of {len(sel)}")
st=eqx.tree_at(lambda q:q.atm.co2conc,st,389.8e-6); yr=2001
for im in range(target): st,key,dd=run_month_gswp3(st,consts,sp,rd,i,j,lat_rad,yr,im,key,3600.0,leap=False,nband=2,nsoilcall=1)
saved=st; k_in=(((wet+6)%12)==target)
lai0=np.asarray(saved.vegetation.lai); print(f"scaling plai (lai is rebuilt daily from plai); saved state: displayed LAI upper {lai0[k_in,1].mean():.2f} lower {lai0[k_in,0].mean():.2f} (cells in their dry season), fu {np.asarray(saved.vegetation.fu)[k_in].mean():.2f}")
res={}
for f in (1.0,0.5,0.25,0.1,2.0):
    stf=eqx.tree_at(lambda q:q.vegetation.plai, saved, saved.vegetation.plai*f)     # plai: pheno rebuilds lai from it daily; sai held
    _,_,dd=run_month_gswp3(stf,consts,sp,rd,i,j,lat_rad,yr,target,key,3600.0,leap=False,nband=2,nsoilcall=1)
    g=np.asarray(dd[0]).sum(0); e=np.asarray(dd[1]).sum(0); res[f]=(g,e)
    gk=np.sum(area[k_in]*g[k_in])/area[k_in].sum(); ek=np.sum(area[k_in]*e[k_in])/area[k_in].sum(); g0=np.sum(area[k_in]*res[1.0][0][k_in])/area[k_in].sum()
    print(f"  LAI x{f:<4}: month GPP {gk:7.4f} kgC/m2 (ratio to x1 {gk/g0:5.3f}; dlnGPP/dlnLAI {np.log(gk/g0)/np.log(f) if f!=1 else float('nan'):5.2f})   ET {ek:5.1f} mm   per-leaf GPP relative {gk/g0/f:5.2f}", flush=True)
