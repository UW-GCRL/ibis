"""Lever test 22 (fixed clock): is the steppe summer deficit COVER or LAI? Lower-canopy cover floor fl_min 0.25 (fitted) / 0.5 / 0.75 / 1.0 and cover_k, with the light partition (PAR to lower canopy vs soil) on the 60 steppe cells."""
import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at, fluxcom_monthly_at, PFT_NAMES
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum; from ibis.state.et_parts import create_et_parts
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v12g/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]; ia=fl[sel_all]//720; ja=fl[sel_all]%720
stg,_=load_state(f'{V}/val_v12g/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3),with_stress=True,with_etparts=True))
plai_g=np.asarray(stg.vegetation.plai)[sel_all]; dom=np.argmax(plai_g,1); dom[plai_g.sum(1)<0.05]=-1
go=fluxcom_at(ia,ja); k=(dom==10)&(LATV[ia]>=40)&(LATV[ia]<55)&np.isfinite(go)&(go>0.2)
rng=np.random.default_rng(5); pos=rng.choice(np.where(k)[0],min(60,k.sum()),replace=False); sel=sel_all[pos]
i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32); area=np.cos(np.deg2rad(LATV[i])); w=area/area.sum()
om=fluxcom_monthly_at(i,j); g_o=fluxcom_at(i,j)
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st0=_cast_tree_f32(jax.tree.map(cut,full)); st0=eqx.tree_at(lambda s: s.stress, st0, create_stress_accum(len(sel)), is_leaf=lambda x: x is None); st0=eqx.tree_at(lambda s: s.etparts, st0, create_et_parts(len(sel)), is_leaf=lambda x: x is None)
sp0,pft0,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v12g/params_fit_v12g.yaml'))
v=np.asarray(sp0.stomata_params.vmax_pft); s_=np.asarray(pft0.specla); rd=CachedForcing(GSWP3Reader(GSWP3),[2001,2002,2003],i,j); consts=PhysicalConstants()
site_transient.apply_co2=lambda q,y,series=None: eqx.tree_at(lambda z:z.atm.co2conc,q,389.8e-6)
tw=np.asarray(full.vegetation.tw)[sel]; tc=np.asarray(full.vegetation.tc)[sel]; tw=tw-273.16 if tw.mean()>100 else tw; tc=tc-273.16 if tc.mean()>100 else tc
print(f"{len(sel)} C4-dominated 40-55N steppe cells: tw {tw.mean():.1f} C (range {tw.min():.1f}-{tw.max():.1f}), tc {tc.mean():.1f} C; FLUXCOM GPP {np.sum(w*g_o):.2f} kgC/m2/yr")
print("obs monthly shape  : "+" ".join(f"{x:5.3f}" for x in (om*w).sum(1)/(om*w).sum()))



REC=[]; real_rm=site_transient.run_month_gswp3
def rec_rm(state,*a,**k):
    out=real_rm(state,*a,**k); st=out[0]; c=st.canopy; vg=st.vegetation
    REC.append(dict(fl=np.asarray(vg.fl).copy(), lai_l=np.asarray(vg.lai[:,0]).copy(), soll=np.asarray(c.soll).copy(), solg=np.asarray(c.solg).copy(), solu=np.asarray(c.solu).copy(), topparl=np.asarray(c.topparl).copy()))
    return out
site_transient.run_month_gswp3=rec_rm
cases={"as fitted (fl_min 0.25, cover_k 0.20)":{}, "fl_min 0.50":{"fl_min":0.50}, "fl_min 0.75":{"fl_min":0.75}, "fl_min 1.0 (full cover)":{"fl_min":1.0}, "cover_k 1.0 (fl = 1-exp(-LAI))":{"cover_k":1.0}}
print(f"\n{'case':38s} | {'annual':>6} | {'Jun-Jul':>7} {'Nov-Feb':>7} | {'amp':>5} (obs {(om*w).sum(1).max()/(om*w).sum(1).min():.0f}) | Jul: fl  LAI_l | SW absorbed Jul (W/m2, month-end step): lower  soil  upper")
for nm,ex in cases.items():
    REC.clear(); st=st0
    sp,pft=apply_params(sp0,pft0,v,s_,ex); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
    key=jax.random.PRNGKey(0); GM=[]
    for n in range(8):
        st,key,d=one_year(st,key,2001+(n%3),ctx,1)
        if n>=5: GM.append(d['gpp_monthly'])
    gm=np.mean(GM,0); R=REC[-36:]; jul=[R[y*12+6] for y in range(3)]
    W=lambda key_: float(np.mean([np.sum(w*r_[key_]) for r_ in jul]))
    comp=(gm*w).sum(1); co=(om*w).sum(1); r=comp/np.maximum(co,1e-9)
    print(f"{nm:38s} | {comp.sum()/co.sum():6.2f} | {comp[5:7].sum()/co[5:7].sum():7.2f} {comp[[10,11,0,1]].sum()/co[[10,11,0,1]].sum():7.2f} | {comp.max()/max(comp.min(),1e-9):5.1f} | {W('fl'):4.2f} {W('lai_l'):5.2f} | {W('soll'):6.0f} {W('solg'):5.0f} {W('solu'):6.0f}", flush=True)
    print("   model/obs by month: "+" ".join(f"{x:5.2f}" for x in r))
