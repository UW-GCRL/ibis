"""Can the stage-(a) levers move the sparse cells' autumn GPP excess at all?"""
import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, netCDF4 as nc, glob, re
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at, fluxcom_monthly_at
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v11d/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]
# sparse-excess cells: domain-run annual GPP >= 2x FLUXCOM (last 5 yr of val_v11d/fit)
fs=sorted(glob.glob(f'{V}/val_v11d/fit/out/ibis_yearly_*.nc'),key=lambda p:int(re.search(r'(\d+)\.nc$',p).group(1)))[-5:]
g=np.zeros(len(fl))
for f in fs:
    with nc.Dataset(f) as d: g+=np.nan_to_num(np.asarray(d['aygpptot'][:],float)).ravel()[fl]/5
ia=fl[sel_all]//720; ja=fl[sel_all]%720; gobs=fluxcom_at(ia,ja)
ratio=g[sel_all]/np.maximum(gobs,1e-9); pick=np.where((ratio>=2)&(gobs>0.05)&(np.abs(LATV[ia])<66))[0]
rng=np.random.default_rng(0); pick=rng.choice(pick,size=min(60,len(pick)),replace=False); sel=sel_all[pick]
i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32)
obs_m=fluxcom_monthly_at(i,j); area=np.cos(np.deg2rad(LATV[i]))
print(f"{len(sel)} sparse-excess cells (annual ratio>=2, |lat|<66), mean lat {np.mean(np.abs(LATV[i])):.1f}, mean ratio {ratio[pick].mean():.2f}")
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st0=_cast_tree_f32(jax.tree.map(cut,full)); del full
sp0,pft0,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v11d/params_fit_v11d.yaml'))
v=np.asarray(sp0.stomata_params.vmax_pft); s=np.asarray(pft0.specla)
rd=CachedForcing(GSWP3Reader(GSWP3),[2001,2002,2003],i,j)
site_transient.apply_co2=lambda s_,y,series=None: eqx.tree_at(lambda q:q.atm.co2conc,s_,389.8e-6)
def run(extras, nyr=4):
    sp,pft=apply_params(sp0,pft0,v,s,extras); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=PhysicalConstants(),reader=rd)
    st=st0; key=jax.random.PRNGKey(0); acc=[]
    for n in range(nyr):
        st,key,d=one_year(st,key,2001+(n%3),ctx,1)
        if n>=1: acc.append(d['gpp_monthly'])
    mm=np.mean(acc,axis=0); r=(mm*area).sum(axis=1)/np.maximum((obs_m*area).sum(axis=1),1e-9)
    return r, float((mm*area).sum()/(obs_m*area).sum())
cases={
 "default (v11d, trigger inert)":     {},
 "drop_stress_thresh = 0.3":           {"drop_stress_thresh":0.3},
 "drop_stress_thresh = 0.5":           {"drop_stress_thresh":0.5},
 "drop_stress_thresh = 0.7":           {"drop_stress_thresh":0.7},
 "0.5 + drop_floor 0.02, ddays 10":    {"drop_stress_thresh":0.5,"drop_floor":0.02,"ddays":10.0},
}
print(f"{'case':<48}{'annual':>7}   J     F     M     A     M     J     J     A     S     O     N     D")
for nm,ex in cases.items():
    r,ann=run(ex); print(f"{nm:<48}{ann:>7.2f}  "+" ".join(f"{x:5.2f}" for x in r), flush=True)
