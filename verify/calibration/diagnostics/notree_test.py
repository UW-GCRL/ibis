"""Leverage ceiling: what if the sparse cells had no upper canopy at all?"""
import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, netCDF4 as nc, glob, re
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, fluxcom_at, fluxcom_monthly_at
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v11d/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]
fs=sorted(glob.glob(f'{V}/val_v11d/fit/out/ibis_yearly_*.nc'),key=lambda p:int(re.search(r'(\d+)\.nc$',p).group(1)))[-5:]
g=np.zeros(len(fl))
for f in fs:
    with nc.Dataset(f) as d: g+=np.nan_to_num(np.asarray(d['aygpptot'][:],float)).ravel()[fl]/5
ia=fl[sel_all]//720; ja=fl[sel_all]%720; gobs=fluxcom_at(ia,ja)
ratio=g[sel_all]/np.maximum(gobs,1e-9); pick=np.where((ratio>=2)&(gobs>0.05)&(np.abs(LATV[ia])<66))[0]
rng=np.random.default_rng(0); pick=rng.choice(pick,size=min(60,len(pick)),replace=False); sel=sel_all[pick]
i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32)
obs_m=fluxcom_monthly_at(i,j); area=np.cos(np.deg2rad(LATV[i])); NH=LATV[i]>0
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st0=_cast_tree_f32(jax.tree.map(cut,full)); del full
sp,pft,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v11d/params_fit_v11d.yaml'))
rd=CachedForcing(GSWP3Reader(GSWP3),[2001,2002,2003],i,j)
site_transient.apply_co2=lambda s_,y,series=None: eqx.tree_at(lambda q:q.atm.co2conc,s_,389.8e-6)
real_exist=site_transient.existence
def run(no_trees):
    st=st0
    if no_trees:
        v=st.vegetation; z8=jnp.zeros_like(v.plai[:,:8])
        st=eqx.tree_at(lambda s_:(s_.vegetation.plai,s_.vegetation.cbiol,s_.vegetation.cbiow,s_.vegetation.cbior,s_.vegetation.exist), st,
                       (v.plai.at[:,:8].set(0.0), v.cbiol.at[:,:8].set(0.0), v.cbiow.at[:,:8].set(0.0), v.cbior.at[:,:8].set(0.0), v.exist.at[:,:8].set(0.0)))
        site_transient.existence=lambda *a,**k: real_exist(*a,**k).at[:,:8].set(0.0)   # trees may not re-establish
    else:
        site_transient.existence=real_exist
    ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=PhysicalConstants(),reader=rd); key=jax.random.PRNGKey(0); acc=[]
    for n in range(5):
        st,key,d=one_year(st,key,2001+(n%3),ctx,1)
        if n>=2: acc.append(d['gpp_monthly'])
    mm=np.mean(acc,axis=0); v=st.vegetation
    out={h:((mm[:,k]*area[k]).sum(axis=1)/np.maximum((obs_m[:,k]*area[k]).sum(axis=1),1e-9), float((mm[:,k]*area[k]).sum()/(obs_m[:,k]*area[k]).sum())) for h,k in (("NH",NH),("SH",~NH))}
    return out, float(np.asarray(v.fu).mean()), float(np.asarray(v.fl).mean()), float(np.asarray(v.plai)[:,:8].sum(axis=1).mean()), float(np.asarray(v.plai)[:,8:].sum(axis=1).mean())
print(f"60 sparse-excess cells, {NH.sum()} NH / {(~NH).sum()} SH; years 3-5 of 2001-03 cycled at 389.8 ppm")
for nm,nt in (("DEFAULT (v11d)",False),("NO UPPER CANOPY (trees removed, cannot re-establish)",True)):
    out,fu_,fl_,up,lo=run(nt); print(f"\n{nm}   [fu {fu_:.2f} fl {fl_:.2f}  upper LAI {up:.2f}  lower LAI {lo:.2f}]")
    for h in ("NH","SH"):
        r,ann=out[h]; print(f"  {h} annual {ann:5.2f}   "+" ".join(f"{x:5.2f}" for x in r), flush=True)
print("                       J     F     M     A     M     J     J     A     S     O     N     D")
