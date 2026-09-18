"""Lever test 6: water-limited tree establishment on sparse, dense and tower cells."""
import sys, glob, re, numpy as np, pandas as pd, xarray as xr, netCDF4 as nc, jax, jax.numpy as jnp, equinox as eqx
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, LONV, MASK, GSWP3, NPOI_FULL, MODIS_BASE, CachedForcing, apply_params, fluxcom_at, fluxcom_monthly_at
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.state.stress_accum import create_stress_accum
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V='/shared/nas2/Hangkai/ibis/verify_outputs'; F='/shared/nas2/Min/FLUXNET2015/fluxnet_compact'
# ---- cell sets
sel_all=np.load(f'{V}/fit_carbon_v11d/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]; posmap={g:p for p,g in enumerate(fl)}
fs=sorted(glob.glob(f'{V}/val_v11d/fit/out/ibis_yearly_*.nc'),key=lambda p:int(re.search(r'(\d+)\.nc$',p).group(1)))[-5:]
g=np.zeros(len(fl))
for f in fs:
    with nc.Dataset(f) as d: g+=np.nan_to_num(np.asarray(d['aygpptot'][:],float)).ravel()[fl]/5
ia=fl[sel_all]//720; ja=fl[sel_all]%720; gobs_all=fluxcom_at(ia,ja); ratio=g[sel_all]/np.maximum(gobs_all,1e-9)
rng=np.random.default_rng(0)
sparse=np.where((ratio>=2)&(gobs_all>0.05)&(np.abs(LATV[ia])<66))[0]; sparse=sel_all[rng.choice(sparse,60,replace=False)]
dense=np.where((ratio>=0.8)&(ratio<1.25)&(gobs_all>=np.quantile(gobs_all[gobs_all>0.05],0.75)))[0]; dense=sel_all[rng.choice(dense,60,replace=False)]
meta=pd.read_csv(f'{F}/FLUXNET_siteinfo.csv',encoding='utf-8-sig').set_index('SITE_ID')
TS=["US-SRM","US-Wkg","US-Whs","US-Cop","AU-Cpr","AU-ASM","AU-Stp","AU-DaS","SD-Dem","CN-Cng","CN-Du2","CN-Sw2","ES-LJu","IT-Noe","ZA-Kru","US-Ton","US-Me6","US-Ha1","BR-Sa1","FI-Hyy"]
tw=[(s,int(np.argmin(np.abs(LATV-float(meta.loc[s,'LOCATION_LAT'])))),int(np.argmin(np.abs(LONV-float(meta.loc[s,'LOCATION_LONG']))))) for s in TS if s in meta.index]
tw=[(s,a,b) for s,a,b in tw if a*720+b in posmap]; TOWER_NAMES=[x[0] for x in tw]; tower=np.array([posmap[a*720+b] for _,a,b in tw])
sd=xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc"); pp=np.asarray(sd["PCT_NAT_PFT"].values); pv=np.asarray(sd["PCT_NATVEG"].values)/100.0
if float(sd["LATIXY"].values[0,0])<float(sd["LATIXY"].values[-1,0]): pp=pp[:,::-1,:]; pv=pv[::-1,:]
TREE=pp[1:9].sum(axis=0)/100.0*pv
SETS={"SPARSE (ratio>=2)":sparse,"DENSE well-fitted":dense,"TOWER cells":tower}
cases={"default (gate inert)":{},"tree_stress_min 0.45":{"tree_stress_min":0.45},"tree_stress_min 0.55":{"tree_stress_min":0.55},"tree_stress_min 0.65":{"tree_stress_min":0.65},
       "0.55 + cover_k 0.5 + stress trigger 0.7":{"tree_stress_min":0.55,"cover_k":0.5,"drop_stress_thresh":0.7}}
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
rd_years=[2001,2002,2003]; consts=PhysicalConstants(); reader=GSWP3Reader(GSWP3)
site_transient.apply_co2=lambda s_,y,series=None: eqx.tree_at(lambda q:q.atm.co2conc,s_,389.8e-6)
print("months: J F M A M J J A S O N D")
for setname,sel in SETS.items():
    i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32); NH=LATV[i]>0
    obs_m=fluxcom_monthly_at(i,j); area=np.cos(np.deg2rad(LATV[i]))
    cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
    st0=_cast_tree_f32(jax.tree.map(cut,full)); st0=eqx.tree_at(lambda s_: s_.stress, st0, create_stress_accum(len(sel)), is_leaf=lambda x: x is None)
    sp0,pft0,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v11d/params_fit_v11d.yaml'))
    v=np.asarray(sp0.stomata_params.vmax_pft); s_=np.asarray(pft0.specla); rd=CachedForcing(reader,rd_years,i,j)
    print(f"\n################ {setname}  n={len(sel)}  {NH.sum()} NH / {(~NH).sum()} SH  mean|lat| {np.mean(np.abs(LATV[i])):.0f} ################")
    for nm,ex in cases.items():
        sp,pft=apply_params(sp0,pft0,v,s_,ex); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
        st=st0; key=jax.random.PRNGKey(0); acc=[]
        for n in range(6):
            st,key,d=one_year(st,key,2001+(n%3),ctx,1)
            if n>=3: acc.append(d['gpp_monthly'])
        mm=np.mean(acc,axis=0); vg=st.vegetation
        fu=np.asarray(vg.fu); fl_=np.asarray(vg.fl); up=np.asarray(vg.plai)[:,:8].sum(axis=1); lo=np.asarray(vg.plai)[:,8:].sum(axis=1)
        print(f"\n{nm}   [fu {fu.mean():.2f}  fl {fl_.mean():.2f}  upper LAI {up.mean():.2f}  lower LAI {lo.mean():.2f}]")
        for h,k in (("NH",NH),("SH",~NH)):
            if k.sum()==0: continue
            r=(mm[:,k]*area[k]).sum(axis=1)/np.maximum((obs_m[:,k]*area[k]).sum(axis=1),1e-9); ann=(mm[:,k]*area[k]).sum()/(obs_m[:,k]*area[k]).sum()
            print(f"  {h} annual {ann:5.2f}   "+" ".join(f"{x:5.2f}" for x in r), flush=True)
        if setname.startswith("TOWER"):
            print("   fu(model)/tree(MODIS): "+"  ".join(f"{nm_}:{fu[k_]:.2f}/{TREE[i[k_],j[k_]]:.2f}" for k_,nm_ in enumerate(TOWER_NAMES)))
