import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, netCDF4 as nc, glob, re
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at, fluxcom_monthly_at
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
full,_=load_state(f'{V}/val_v11d/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st0=_cast_tree_f32(jax.tree.map(cut,full)); del full
sp0,pft0,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v11d/params_fit_v11d.yaml'))
v=np.asarray(sp0.stomata_params.vmax_pft); s=np.asarray(pft0.specla)
rd=CachedForcing(GSWP3Reader(GSWP3),[2001],i,j); consts=PhysicalConstants()
plai=np.asarray(st0.vegetation.plai)
def year(extras):
    sp,pft=apply_params(sp0,pft0,v,s,extras); st=eqx.tree_at(lambda q:q.atm.co2conc,st0,389.8e-6); key=jax.random.PRNGKey(0); rows=[]
    for im in range(12):
        st,key,dd=run_month_gswp3(st,consts,sp,rd,i,j,lat_rad,2001,im,key,3600.0,leap=False,nband=2,nsoilcall=1)
        sm=st.summary; vg=st.vegetation
        # canopy-type 10-day net assimilation (mol/m2 leaf/s) x displayed leaf area -> relative carbon by type
        ub=np.asarray(sm.a10ancub); uc=np.asarray(sm.a10ancuc); ls=np.asarray(sm.a10ancls); l3=np.asarray(sm.a10ancl3); l4=np.asarray(sm.a10ancl4)
        tempu=np.asarray(vg.tempu); templ=np.asarray(vg.templ); dropu=np.asarray(vg.dropu); dropls=np.asarray(vg.dropls); dropl3=np.asarray(vg.dropl3); dropl4=np.asarray(vg.dropl4)
        lai_ub=plai[:,0]+plai[:,1]*dropu+plai[:,2]+plai[:,4]*tempu+plai[:,6]*tempu     # upper broadleaf (ever + decid)
        lai_uc=plai[:,3]+plai[:,5]+plai[:,7]*tempu                                     # upper conifer
        lai_ls=plai[:,8]+plai[:,9]*np.minimum(templ,dropls)                             # shrubs (evergreen + decid)
        lai_l3=plai[:,11]*np.minimum(templ,dropl3); lai_l4=plai[:,10]*np.minimum(templ,dropl4)
        c={"upper BL":(ub*lai_ub).mean(),"upper conif":(uc*lai_uc).mean(),"shrub":(ls*lai_ls).mean(),"C3":(l3*lai_l3).mean(),"C4":(l4*lai_l4).mean()}
        tot=sum(c.values()); rows.append(dict(mon=im+1,gpp=float(np.asarray(dd[0]).sum(axis=0).mean()),**{k:v_/tot for k,v_ in c.items()},
                 tempu=tempu.mean(),templ=templ.mean(),dropu=dropu.mean(),dropls=dropls.mean(),dropl3=dropl3.mean(),dropl4=dropl4.mean(),
                 lai_ub=lai_ub.mean(),lai_uc=lai_uc.mean(),lai_ls=lai_ls.mean(),lai_gr=(lai_l3+lai_l4).mean()))
    return rows
for nm,ex in (("DEFAULT (trigger inert)",{}),("drop_stress_thresh = 0.7",{"drop_stress_thresh":0.7})):
    R=year(ex); print(f"\n=== {nm} ===")
    print(f"{'mon':>3} {'GPP':>6} | share of assimilation:  {'upBL':>5} {'upCon':>5} {'shrub':>5} {'C3':>5} {'C4':>5} | displayed LAI: {'upBL':>5} {'upCon':>5} {'shrub':>5} {'grass':>5} | {'tempu':>5} {'templ':>5} {'dropu':>5} {'dropls':>6} {'dropl4':>6}")
    for r in R: print(f"{r['mon']:>3} {r['gpp']:6.3f} |                          {r['upper BL']:5.2f} {r['upper conif']:5.2f} {r['shrub']:5.2f} {r['C3']:5.2f} {r['C4']:5.2f} |                {r['lai_ub']:5.2f} {r['lai_uc']:5.2f} {r['lai_ls']:5.2f} {r['lai_gr']:5.2f} | {r['tempu']:5.2f} {r['templ']:5.2f} {r['dropu']:5.2f} {r['dropls']:6.2f} {r['dropl4']:6.2f}")
