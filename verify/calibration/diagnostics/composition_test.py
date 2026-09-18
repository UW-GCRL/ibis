import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, netCDF4 as nc, glob, re
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at, fluxcom_monthly_at
import site_transient
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32, run_month_gswp3
from ibis.timestepping.calendar import is_leap_year
V='/shared/nas2/Hangkai/ibis/verify_outputs'
NM=("tropBLev","tropBLdec","warmBLev","tempCONev","tempBLdec","borCONev","borBLdec","borCONdec","evShrub","decShrub","C4grass","C3grass")
sel_all=np.load(f'{V}/fit_carbon_v11d/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]
fs=sorted(glob.glob(f'{V}/val_v11d/fit/out/ibis_yearly_*.nc'),key=lambda p:int(re.search(r'(\d+)\.nc$',p).group(1)))[-5:]
g=np.zeros(len(fl))
for f in fs:
    with nc.Dataset(f) as d: g+=np.nan_to_num(np.asarray(d['aygpptot'][:],float)).ravel()[fl]/5
ia=fl[sel_all]//720; ja=fl[sel_all]%720; gobs=fluxcom_at(ia,ja)
ratio=g[sel_all]/np.maximum(gobs,1e-9); pick=np.where((ratio>=2)&(gobs>0.05)&(np.abs(LATV[ia])<66))[0]
rng=np.random.default_rng(0); pick=rng.choice(pick,size=min(60,len(pick)),replace=False); sel=sel_all[pick]   # SAME 60 cells as lever_test
i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32)
full,_=load_state(f'{V}/val_v11d/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st=_cast_tree_f32(jax.tree.map(cut,full)); del full
sp,pft,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v11d/params_fit_v11d.yaml'))
plai=np.asarray(st.vegetation.plai); tot=plai.sum(axis=1)
print("=== WHO HOLDS THE LEAF AREA in the 60 sparse-excess cells (v11d 2060 state) ===")
share=(plai.sum(axis=0)/plai.sum()); dom=np.argmax(plai,axis=1)
for k in np.argsort(-share)[:6]:
    if share[k]>0.005: print(f"  {NM[k]:<10} share of total LAI {share[k]:5.1%}   dominant in {int((dom==k).sum()):2d}/60 cells   mean plai {plai[:,k].mean():.2f}")
print(f"  total LAI mean {tot.mean():.2f}; upper canopy (PFT 0-7) {plai[:,:8].sum(axis=1).mean():.2f}, lower (8-11) {plai[:,8:].sum(axis=1).mean():.2f}")
ex=np.asarray(st.vegetation.exist); print("  exist allowed (fraction of cells):", {NM[k]:round(float(ex[:,k].mean()),2) for k in range(12) if ex[:,k].mean()>0.3})
# one year, month by month: drought/cold triggers
rd=CachedForcing(GSWP3Reader(GSWP3),[2001],i,j); consts=PhysicalConstants(); key=jax.random.PRNGKey(0)
st=eqx.tree_at(lambda q:q.atm.co2conc,st,389.8e-6)
print("\n=== MONTH-BY-MONTH TRIGGERS (mean over 60 cells; frac<0 = fraction of cells with negative 10-day net assimilation) ===")
print(f"{'mon':>3} {'a10ancls':>9} {'frac<0':>7} {'dropls':>7} | {'a10ancl3':>9} {'frac<0':>7} {'dropl3':>7} | {'templ':>6} {'tempu':>6} | {'a10td C':>8} | {'GPP mo':>7}")
for im in range(12):
    st,key,dd=run_month_gswp3(st,consts,sp,rd,i,j,lat_rad,2001,im,key,3600.0,leap=False,nband=2,nsoilcall=1)
    v=st.vegetation; sm=st.summary
    a_ls=np.asarray(sm.a10ancls); a_l3=np.asarray(sm.a10ancl3)
    print(f"{im+1:>3} {a_ls.mean():9.2e} {np.mean(a_ls<0):7.2f} {np.asarray(v.dropls).mean():7.3f} | {a_l3.mean():9.2e} {np.mean(a_l3<0):7.2f} {np.asarray(v.dropl3).mean():7.3f} | {np.asarray(v.templ).mean():6.3f} {np.asarray(v.tempu).mean():6.3f} | {np.asarray(sm.a10td).mean()-273.16:8.1f} | {np.asarray(dd[0]).sum(axis=0).mean():7.3f}")
print("\nFLUXCOM monthly (mean over cells, kgC/m2/mo):", " ".join(f"{x:.3f}" for x in fluxcom_monthly_at(i,j).mean(axis=1)))
