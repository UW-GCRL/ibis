import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, netCDF4 as nc, glob, re
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at, modis_at
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32, run_month_gswp3
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
modis=modis_at(i,j)
full,_=load_state(f'{V}/val_v11d/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st0=_cast_tree_f32(jax.tree.map(cut,full)); del full
v=st0.vegetation; plai=np.asarray(v.plai); sai=np.asarray(v.sai); lai=np.asarray(v.lai); fu=np.asarray(v.fu); flw=np.asarray(v.fl)
print(f"60 cells: model plai total {plai.sum(axis=1).mean():.2f}   MODIS annual-max LAI {modis.mean():.2f}  (model/MODIS {np.median(plai.sum(axis=1)/np.maximum(modis,0.05)):.1f}x median)")
print(f"  MODIS LAI distribution: <0.5: {np.mean(modis<0.5):.0%}  0.5-1: {np.mean((modis>=0.5)&(modis<1)):.0%}  1-2: {np.mean((modis>=1)&(modis<2)):.0%}  >2: {np.mean(modis>=2):.0%}")
print(f"  stem area sai: upper {sai[:,1].mean():.2f}  lower {sai[:,0].mean():.2f}    canopy-level lai: upper {lai[:,1].mean():.2f} (fu {fu.mean():.2f})  lower {lai[:,0].mean():.2f} (fl {flw.mean():.2f})")
print(f"  => per-canopy optical load lai+sai: upper {(lai[:,1]+sai[:,1]).mean():.2f}, lower {(lai[:,0]+sai[:,0]).mean():.2f}")
# monthly absorbed PAR fraction proxy: use the daily GPP response to leaf removal already measured; here report displayed lai+sai by month for default vs 0.7
sp0,pft0,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v11d/params_fit_v11d.yaml'))
vv=np.asarray(sp0.stomata_params.vmax_pft); ss=np.asarray(pft0.specla)
rd=CachedForcing(GSWP3Reader(GSWP3),[2001],i,j); consts=PhysicalConstants()
print(f"\n{'mon':>3} | {'DEFAULT lai_u+sai_u':>19} {'lai_l+sai_l':>11} {'GPP':>6} | {'THRESH 0.7 lai_u+sai_u':>22} {'lai_l+sai_l':>11} {'GPP':>6} | {'dGPP/dLAI elasticity':>20}")
def run(extras):
    sp,pft=apply_params(sp0,pft0,vv,ss,extras); st=eqx.tree_at(lambda q:q.atm.co2conc,st0,389.8e-6); key=jax.random.PRNGKey(0); out=[]
    for im in range(12):
        st,key,dd=run_month_gswp3(st,consts,sp,rd,i,j,lat_rad,2001,im,key,3600.0,leap=False,nband=2,nsoilcall=1)
        vg=st.vegetation; L=np.asarray(vg.lai); Sa=np.asarray(vg.sai)
        out.append((float((L[:,1]+Sa[:,1]).mean()),float((L[:,0]+Sa[:,0]).mean()),float(L.sum(axis=1).mean()),float(np.asarray(dd[0]).sum(axis=0).mean())))
    return out
A=run({}); B=run({"drop_stress_thresh":0.7})
for im in range(12):
    a=A[im]; b=B[im]; el=(np.log(b[3]/a[3])/np.log(b[2]/a[2])) if abs(np.log(b[2]/a[2]))>1e-3 else float('nan')
    print(f"{im+1:>3} | {a[0]:19.2f} {a[1]:11.2f} {a[3]:6.3f} | {b[0]:22.2f} {b[1]:11.2f} {b[3]:6.3f} | {el:20.2f}")
print("\nelasticity = dlnGPP/dlnLAI between the two runs (Beer-Lambert at LAI 2-3 would be ~0.4-0.5; docs/CALIBRATION.md 5 measured 0.09 globally)")
