"""Lever test 17: why does the steppe still assimilate in winter on the fixed clock? GPP per unit displayed LAI by month; C4 vs the temperature response; the lower-canopy floor fl_min."""
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
om=fluxcom_monthly_at(i,j)
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st0=_cast_tree_f32(jax.tree.map(cut,full)); st0=eqx.tree_at(lambda s: s.stress, st0, create_stress_accum(len(sel)), is_leaf=lambda x: x is None); st0=eqx.tree_at(lambda s: s.etparts, st0, create_et_parts(len(sel)), is_leaf=lambda x: x is None)
sp0,pft0,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v12g/params_fit_v12g.yaml'))
v=np.asarray(sp0.stomata_params.vmax_pft); s_=np.asarray(pft0.specla); rd=CachedForcing(GSWP3Reader(GSWP3),[2001,2002,2003],i,j); consts=PhysicalConstants()
site_transient.apply_co2=lambda q,y,series=None: eqx.tree_at(lambda z:z.atm.co2conc,q,389.8e-6)
REC=[]; real_rm=site_transient.run_month_gswp3
def rec_rm(state,*a,**k):
    out=real_rm(state,*a,**k); st=out[0]; vg=st.vegetation
    REC.append(dict(lai_l=np.asarray(vg.lai[:,0]).copy(), lai_u=np.asarray(vg.lai[:,1]).copy(), templ=np.asarray(vg.templ).copy(), a10td=np.asarray(st.summary.a10td).copy()-273.16, fl=np.asarray(vg.fl).copy()))
    return out
site_transient.run_month_gswp3=rec_rm
cases={"as fitted (fixed clock)":{}, "fl_min 0.05 (floor lowered)":{"fl_min":0.05}, "tthresh_l_c +8 C":{"tthresh_l_c":8.0}, "C4 vmax x0.5":"c4half", "no lower canopy (plai 9-12 = 0)":"nolower"}
real_exist=site_transient.existence
for nm,ex in cases.items():
    REC.clear(); st=st0; vv=v.copy(); e={} if not isinstance(ex,dict) else ex
    if ex=="c4half": vv[10]*=0.5
    if ex=="nolower":
        vg=st.vegetation; ix=jnp.array([8,9,10,11]); st=eqx.tree_at(lambda s: (s.vegetation.plai,s.vegetation.cbiol,s.vegetation.exist), st, (vg.plai.at[:,ix].set(0.0), vg.cbiol.at[:,ix].set(0.0), vg.exist.at[:,ix].set(0.0)))
        site_transient.existence=lambda *a,**k: real_exist(*a,**k).at[:,ix].set(0.0)
    sp,pft=apply_params(sp0,pft0,vv,s_,e); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
    key=jax.random.PRNGKey(0); GM=[]
    for n in range(8):
        st,key,d=one_year(st,key,2001+(n%3),ctx,1)
        if n>=5: GM.append(d['gpp_monthly'])
    site_transient.existence=real_exist
    gm=np.mean(GM,0); R=REC[-36:]; M=lambda key_: np.mean([np.stack([R[y*12+q][key_] for q in range(12)]) for y in range(3)],axis=0)
    W=lambda x: (x*w).sum(1); comp=(gm*w).sum(1); co=(om*w).sum(1); lai_l=W(M('lai_l')); lai_u=W(M('lai_u'))
    print(f"\n== {nm} ==  annual {comp.sum()/co.sum():.2f}   Nov-Feb {comp[[10,11,0,1]].sum()/co[[10,11,0,1]].sum():.2f}   Jun-Jul {comp[[5,6]].sum()/co[[5,6]].sum():.2f}   amp {comp.max()/max(comp.min(),1e-9):.1f} (obs {co.max()/co.min():.0f})")
    print("  month              : "+" ".join(f"{q+1:6d}" for q in range(12)))
    print("  GPP model/obs      : "+" ".join(f"{x:6.2f}" for x in comp/np.maximum(co,1e-9)))
    print("  GPP kgC/m2/mo      : "+" ".join(f"{x:6.3f}" for x in comp))
    print("  displayed LAI low/up: "+" ".join(f"{a:3.1f}/{b:3.1f}" for a,b in zip(lai_l,lai_u)))
    print("  GPP per LAI (g/m2 leaf/mo): "+" ".join(f"{1000*x/max(a+b,0.05):6.1f}" for x,a,b in zip(comp,lai_l,lai_u)))
    print("  a10td / templ      : "+" ".join(f"{a:4.0f}/{b:3.2f}" for a,b in zip(W(M('a10td')),W(M('templ')))), flush=True)
