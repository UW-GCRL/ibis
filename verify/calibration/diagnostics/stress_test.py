import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, netCDF4 as nc, glob, re
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, fluxcom_at, fluxcom_monthly_at
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
lat=LATV[i]; nh=lat>0
print(f"60 cells: {nh.sum()} NH / {(~nh).sum()} SH;  lat range {lat.min():.0f}..{lat.max():.0f}")
full,_=load_state(f'{V}/val_v11d/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st=_cast_tree_f32(jax.tree.map(cut,full)); del full
sp,pft,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v11d/params_fit_v11d.yaml'))
rd=CachedForcing(GSWP3Reader(GSWP3),[2001],i,j); consts=PhysicalConstants(); key=jax.random.PRNGKey(0)
st=eqx.tree_at(lambda q:q.atm.co2conc,st,389.8e-6)
obs=fluxcom_monthly_at(i,j)
print(f"\n{'mon':>3} | {'stresstl':>8} {'stresstu':>8} {'frac tl<.5':>10} | {'wsoi L1':>7} {'wsoi L2':>7} | {'prcp mm':>7} | {'GPP mod':>7} {'GPP obs':>7} {'ratio':>6}")
for im in range(12):
    p0=float(np.asarray(st.summary.ayprcp).mean())
    st,key,dd=run_month_gswp3(st,consts,sp,rd,i,j,lat_rad,2001,im,key,3600.0,leap=False,nband=2,nsoilcall=1)
    stl=np.asarray(st.soil.stresstl); stu=np.asarray(st.soil.stresstu); w=np.asarray(st.soil.wsoi)
    gm=np.asarray(dd[0]).sum(axis=0).mean(); go=obs[im].mean()
    print(f"{im+1:>3} | {stl.mean():8.3f} {stu.mean():8.3f} {np.mean(stl<0.5):10.2f} | {w[:,0].mean():7.3f} {w[:,1].mean():7.3f} | {float(np.asarray(st.summary.ayprcp).mean())-p0:7.1f} | {gm:7.3f} {go:7.3f} {gm/max(go,1e-9):6.2f}")
print("\nstresst* = soil-water stress factor entering stomata/drystress (1 = unstressed, 0 = fully stressed); wsoi = liquid fraction of pore space")
