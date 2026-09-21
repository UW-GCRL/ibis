"""Lever test 21 (fixed clock): can C3 grass win the steppes if its vmax is not at the floor? C3 vmax 16.2 (fitted floor) / 25 (F77) / 30 / 35, C4 as fitted, on the 60 C4-dominated 40-55N cells."""
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


print(f"\n{'case':30s} | dominant: {'C4':>3} {'C3':>3} {'evs':>3} {'dcs':>3} | {'annual':>6} | {'Nov-Feb':>7} {'May-Aug':>7} {'Jun-Jul':>7} | {'amp':>5} (obs {(om*w).sum(1).max()/(om*w).sum(1).min():.0f}) | LAI C4 / C3")
for c3v in (16.2, 25.0, 30.0, 35.0):
    vv=v.copy(); vv[11]=c3v*1e-6
    st=st0; vg=st.vegetation
    # seed C3 with a small leaf pool so the competition can start (it is ~0 in the joint-fit state)
    st=eqx.tree_at(lambda s:(s.vegetation.plai,s.vegetation.cbiol), st, (vg.plai.at[:,11].set(jnp.maximum(vg.plai[:,11],0.3)), vg.cbiol.at[:,11].set(jnp.maximum(vg.cbiol[:,11],0.3/float(s_[11])))))
    sp,pft=apply_params(sp0,pft0,vv,s_,{}); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
    key=jax.random.PRNGKey(0); GM=[]
    for n in range(10):
        st,key,d=one_year(st,key,2001+(n%3),ctx,1)
        if n>=7: GM.append(d['gpp_monthly'])
    gm=np.mean(GM,0); pl=np.asarray(st.vegetation.plai); dm=np.argmax(pl,1); dm[pl.sum(1)<0.05]=-1
    comp=(gm*w).sum(1); co=(om*w).sum(1); r=comp/np.maximum(co,1e-9)
    print(f"C3 vmax {c3v:4.1f} (seeded 0.3 LAI)   | {int((dm==10).sum()):3d} {int((dm==11).sum()):3d} {int((dm==8).sum()):3d} {int((dm==9).sum()):3d} | {comp.sum()/co.sum():6.2f} | {comp[[10,11,0,1]].sum()/co[[10,11,0,1]].sum():7.2f} {comp[4:8].sum()/co[4:8].sum():7.2f} {comp[5:7].sum()/co[5:7].sum():7.2f} | {comp.max()/max(comp.min(),1e-9):5.1f} | {pl[:,10].mean():.2f} / {pl[:,11].mean():.2f}", flush=True)
    print("   model/obs by month: "+" ".join(f"{x:5.2f}" for x in r))
