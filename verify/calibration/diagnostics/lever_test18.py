"""Lever test 18: winter GPP vs DAILY air temperature on the 60 steppe cells (fixed clock, joint-fit params): where do the -3 C months' assimilation days come from?"""
import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum; from ibis.state.et_parts import create_et_parts
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32, run_month_gswp3
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v12g/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]; ia=fl[sel_all]//720; ja=fl[sel_all]%720
stg,_=load_state(f'{V}/val_v12g/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3),with_stress=True,with_etparts=True))
plai_g=np.asarray(stg.vegetation.plai)[sel_all]; dom=np.argmax(plai_g,1); dom[plai_g.sum(1)<0.05]=-1
go=fluxcom_at(ia,ja); k=(dom==10)&(LATV[ia]>=40)&(LATV[ia]<55)&np.isfinite(go)&(go>0.2)
rng=np.random.default_rng(5); pos=rng.choice(np.where(k)[0],min(60,k.sum()),replace=False); sel=sel_all[pos]
i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32); area=np.cos(np.deg2rad(LATV[i])); w=area/area.sum()
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st=_cast_tree_f32(jax.tree.map(cut,full)); st=eqx.tree_at(lambda s: s.stress, st, create_stress_accum(len(sel)), is_leaf=lambda x: x is None); st=eqx.tree_at(lambda s: s.etparts, st, create_et_parts(len(sel)), is_leaf=lambda x: x is None)
sp,pft,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v12g/params_fit_v12g.yaml'))
rd=CachedForcing(GSWP3Reader(GSWP3),[2001,2002,2003],i,j); consts=PhysicalConstants()
site_transient.apply_co2=lambda q,y,series=None: eqx.tree_at(lambda z:z.atm.co2conc,q,389.8e-6)
ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd); key=jax.random.PRNGKey(0)
for n in range(5): st,key,d=one_year(st,key,2001+(n%3),ctx,1)          # 5 adjust years on the fixed clock
# year 6: month by month, daily GPP from dd[0] and daily air T from the forcing
st=eqx.tree_at(lambda q:q.atm.co2conc,st,389.8e-6); yr=2003
bins=[(-40,-10),(-10,-5),(-5,0),(0,5),(5,10),(10,15),(15,20),(20,40)]
acc={b:[0.0,0] for b in bins}; win_acc={b:[0.0,0] for b in bins}; month_from_bins=np.zeros(12)
print("year-6 daily GPP binned by DAILY mean air temperature (60 steppe cells, fixed clock):")
for im in range(12):
    raw=rd.read_month(yr,im,i,j); tas=np.asarray(raw['tas']); ndays=tas.shape[0]//24; tday=tas[:ndays*24].reshape(ndays,24,-1).mean(1)-273.16     # (ndays, n)
    st,key,dd=run_month_gswp3(st,consts,sp,rd,i,j,lat_rad,yr,im,key,3600.0,leap=False,nband=2,nsoilcall=1)
    g=np.asarray(dd[0])[:ndays]                                                                                                     # (ndays, n) kgC/m2/day
    for b in bins:
        msk=(tday>=b[0])&(tday<b[1]); acc[b][0]+=float((g*msk*w[None,:]).sum()); acc[b][1]+=int(msk.sum())
        if im in (0,1,10,11): win_acc[b][0]+=float((g*msk*w[None,:]).sum()); win_acc[b][1]+=int(msk.sum())
tot=sum(v[0] for v in acc.values()); wtot=sum(v[0] for v in win_acc.values())
print(f"{'daily T bin (C)':16s} {'cell-days':>9} {'share of days':>13} | {'GPP gC/m2/day (mean over those days)':>36} {'share of annual GPP':>19} | winter (NDJF): {'cell-days':>9} {'GPP share':>9}")
for b in bins:
    n_=acc[b][1]; gsum=acc[b][0]; nw=win_acc[b][1]; gw=win_acc[b][0]
    print(f"{f'{b[0]:+d}..{b[1]:+d}':16s} {n_:9d} {n_/sum(v[1] for v in acc.values()):13.2f} | {1000*gsum*len(sel)/max(n_,1):36.2f} {gsum/tot:19.2f} | {nw:24d} {gw/max(wtot,1e-9):9.2f}")
