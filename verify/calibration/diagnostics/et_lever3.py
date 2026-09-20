"""ET lever test 3: interception capacities and fmax -- transpiration / interception / soil evaporation / snow sublimation by class and month, (b3) parameters, vs the ET ensemble total."""
import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at, tree_at, et_at, et_monthly_at
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum; from ibis.state.et_parts import create_et_parts
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v12e/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]
ia=fl[sel_all]//720; ja=fl[sel_all]%720; lat=LATV[ia]
go=fluxcom_at(ia,ja); tr=tree_at(ia,ja); eo=et_at(ia,ja)
ok=np.isfinite(go)&(go>0.3)&np.isfinite(eo)&np.isfinite(tr)
rng=np.random.default_rng(3)
def pick(k,n): idx=np.where(k&ok)[0]; return rng.choice(idx,min(n,len(idx)),replace=False)
cls={"forest (tree>0.7, |lat|<55)":pick((tr>0.7)&(np.abs(lat)<55),25),"open (tree<0.3, |lat|<40)":pick((tr<0.3)&(np.abs(lat)<40),25),"boreal (lat>50)":pick(lat>50,20)}
pos=np.concatenate(list(cls.values())); sel=sel_all[pos]
edges=np.cumsum([0]+[len(v_) for v_ in cls.values()]); cls_idx={k:np.arange(edges[n],edges[n+1]) for n,k in enumerate(cls)}   # positional ranges in `sel`
i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32); area=np.cos(np.deg2rad(LATV[i]))
g_o=fluxcom_at(i,j); e_o=et_at(i,j); e_om=et_monthly_at(i,j)
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st0=_cast_tree_f32(jax.tree.map(cut,full)); st0=eqx.tree_at(lambda s_: s_.stress, st0, create_stress_accum(len(sel)), is_leaf=lambda x: x is None); st0=eqx.tree_at(lambda s_: s_.etparts, st0, create_et_parts(len(sel)), is_leaf=lambda x: x is None)
sp0,pft0,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v12e/params_fit_v12e.yaml'))
v=np.asarray(sp0.stomata_params.vmax_pft); s_=np.asarray(pft0.specla); rd=CachedForcing(GSWP3Reader(GSWP3),[2001,2002,2003],i,j); consts=PhysicalConstants()
site_transient.apply_co2=lambda q,y,series=None: eqx.tree_at(lambda z:z.atm.co2conc,q,389.8e-6)


names=("transp","interc","soilev","snow")
print("\n(b3) parameters, years 4-6 of a 2001-03 cycle from the v11 state; obs = five-product ensemble (total). Annual, area-weighted mm/yr")
print(f"{'case':22s} | {'class':30s} | {'ET mod':>6} {'obs':>5} {'ratio':>5} | {'T/ET':>5} {'Ei/ET':>5} {'Es/ET':>5} | {'GPP/obs':>7} {'ET amp mod/obs':>14}")
CASES={"F77 interception":{}, "capacities x0.5":{"wliqumax":0.10,"wliqsmax":0.20,"wliqlmax":0.10}, "capacities x2":{"wliqumax":0.40,"wliqsmax":0.80,"wliqlmax":0.40}, "fmax 0.15":{"fmax":0.15}, "fmax 0.40":{"fmax":0.40}}
for p_,ex in CASES.items():
    sp,pft=apply_params(sp0,pft0,v,s_,ex); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
    st=st0; key=jax.random.PRNGKey(0); EPM=[]; EM=[]; G=[]
    for n in range(6):
        st,key,d=one_year(st,key,2001+(n%3),ctx,1)
        if n>=3: EPM.append(d['et_parts_monthly']); EM.append(d['aet_monthly']); G.append(d['aygpptot'])
    epm=np.mean(EPM,0); em=np.mean(EM,0); g=np.mean(G,0)
    for k,ix in cls_idx.items():
        w=area[ix]/area[ix].sum(); ann=epm[:,:,ix].sum(0); T=float(np.sum(w*em[:,ix].sum(0))); O=float(np.sum(w*e_o[ix]))
        parts=[float(np.sum(w*ann[q])) for q in range(4)]
        amp=np.average(em[:,ix].max(0)/np.maximum(em[:,ix].min(0),1),weights=w)/np.average(e_om[:,ix].max(0)/np.maximum(e_om[:,ix].min(0),1),weights=w)
        print(f"{p_:22s} | {k:30s} | {T:6.0f} {O:5.0f} {T/O:5.2f} | {parts[0]/T:5.2f} {parts[1]/T:5.2f} {parts[2]/T:5.2f} | {float(np.sum(w*g[ix]))/float(np.sum(w*g_o[ix])):7.3f} {amp:14.2f}", flush=True)
