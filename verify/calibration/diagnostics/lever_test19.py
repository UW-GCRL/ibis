"""Lever test 19: the winter canopy energy balance, hour by hour. One January and one July day on 20 steppe cells (fixed clock, joint-fit params):
air T, lower-canopy leaf T, tempvm_c4(leaf T), PAR at the lower canopy, and the C4 net assimilation per hour."""
import sys, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, GSWP3, NPOI_FULL, CachedForcing, fluxcom_at
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum; from ibis.state.et_parts import create_et_parts
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
from ibis.timestepping.loops import timestep_body
from ibis.physics.weather import DailyWeather, compute_precip_timing, diurnal
from ibis.physics.stomata import _temp_dep_params
V='/shared/nas2/Hangkai/ibis/verify_outputs'
sel_all=np.load(f'{V}/fit_carbon_v12g/best.npz')['sel']
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]; ia=fl[sel_all]//720; ja=fl[sel_all]%720
stg,_=load_state(f'{V}/val_v12g/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3),with_stress=True,with_etparts=True))
plai_g=np.asarray(stg.vegetation.plai)[sel_all]; dom=np.argmax(plai_g,1); dom[plai_g.sum(1)<0.05]=-1
go=fluxcom_at(ia,ja); k=(dom==10)&(LATV[ia]>=40)&(LATV[ia]<55)&np.isfinite(go)&(go>0.2)
rng=np.random.default_rng(5); pos=rng.choice(np.where(k)[0],min(60,k.sum()),replace=False)[:20]; sel=sel_all[pos]
i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32)
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st=_cast_tree_f32(jax.tree.map(cut,full)); st=eqx.tree_at(lambda s: s.stress, st, create_stress_accum(len(sel)), is_leaf=lambda x: x is None); st=eqx.tree_at(lambda s: s.etparts, st, create_et_parts(len(sel)), is_leaf=lambda x: x is None)
sp,pft,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v12g/params_fit_v12g.yaml'))
rd=CachedForcing(GSWP3Reader(GSWP3),[2001,2002,2003],i,j); consts=PhysicalConstants()
site_transient.apply_co2=lambda q,y,series=None: eqx.tree_at(lambda z:z.atm.co2conc,q,389.8e-6)
ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd); key=jax.random.PRNGKey(0)
for n in range(5): st,key,d=one_year(st,key,2001+(n%3),ctx,1)
st=eqx.tree_at(lambda q:q.atm.co2conc,st,389.8e-6); dtime=3600.0
def one_day(st, key, year, im, day_idx, label):
    raw=rd.read_month(year,im,i,j)
    f=lambda k_: jnp.asarray(np.asarray(raw[k_])[day_idx], dtype=jnp.float32)
    td,tmax,tmin,pr,rsds,rlds,huss,wind,ps=(f(k_) for k_ in ('tas','tasmax','tasmin','pr','rsds','rlds','huss','wind','ps'))
    # cloud from rsds as loops_gswp3 does: reuse its helper if exposed, else a crude proxy (cloudiness only shapes the diurnal split)
    try:
        from ibis.timestepping.loops_gswp3 import _cloud_from_rsds; cloud=_cloud_from_rsds(rsds, lat_rad, jnp.int32(day_idx+1+31*im))
    except Exception: cloud=jnp.clip(1.0-rsds/300.0,0.0,1.0)
    dw=DailyWeather(td=td,tmax=tmax,tmin=tmin,cloud=cloud,precip=pr*86400.0 if float(pr.max())<1e-2 else pr,psurf=ps,qd=huss,ud=wind,iwet=(pr>0).astype(jnp.float32),xstore=jnp.zeros((len(sel),3),jnp.float32))
    key,kt,kd=jax.random.split(key,3); _,plens,startp,endp=compute_precip_timing(kt,dw.precip,dtime)
    jday=jnp.int32(sum([31,28,31,30,31,30,31,31,30,31,30,31][:im])+day_idx+1)
    rows=[]
    for h in range(24):
        kd,sub=jax.random.split(kd); _,hr=diurnal(h*dtime,jday,dw,sp.latitude,plens,startp,endp,sub,dtime,2)
        c=lambda x: jnp.asarray(x,dtype=st.atm.ta.dtype)
        atm=eqx.tree_at(lambda a:(a.ta,a.qa,a.ua,a.coszen,a.solad,a.solai,a.fira,a.raina,a.snowa), st.atm, (c(hr.ta),c(hr.qa),c(hr.ua),c(hr.coszen),c(hr.solad),c(hr.solai),c(rlds),c(hr.raina),c(hr.snowa)))
        st2=eqx.tree_at(lambda s:s.atm, st, atm)
        st2=timestep_body(st2,consts,sp,dtime)
        c2=st2.canopy; an4=np.asarray(c2.an_lower[:,2]); ag4=np.asarray(c2.ag_lower[:,2])          # C4 leaf-level net / gross (mol CO2 m-2 leaf s-1)
        tl=np.asarray(c2.tl); ta=np.asarray(st2.atm.ta); t34=np.asarray(c2.t34); tg=np.asarray(st2.soil.tsoi[:,0])
        _,_,_,_,_,t4=_temp_dep_params(jnp.asarray(tl),sp.stomata_params)
        rows.append((h, ta.mean()-273.16, tl.mean()-273.16, t34.mean()-273.16, tg.mean()-273.16, float(np.mean(np.asarray(t4))),
                     float(np.asarray(c2.topparl).mean()), float(np.asarray(c2.soll).mean()), float(np.asarray(c2.firl).mean()), float(np.asarray(c2.fsenl).mean()),
                     an4.mean()*1e6, ag4.mean()*1e6, float(np.asarray(st2.vegetation.fl).mean()), float(np.asarray(st2.vegetation.lai[:,0]).mean())))
        st=st2
    print(f"\n== {label}: day mean air {float(td.mean())-273.16:5.1f} C, tmax {float(tmax.mean())-273.16:5.1f}, tmin {float(tmin.mean())-273.16:5.1f}; displayed lower LAI {float(np.asarray(st.vegetation.lai[:,0]).mean()):.2f}, fl {float(np.asarray(st.vegetation.fl).mean()):.2f} ==")
    print(f"{'h':>2} {'air':>6} {'leaf_l':>6} {'t34':>6} {'tsoi1':>6} {'tempvm4':>8} {'PARlow':>7} {'soll':>6} {'firl':>7} {'fsenl':>7} | {'An4':>6} {'Ag4':>6} umol/m2leaf/s")
    for r in rows: print(f"{r[0]:2d} {r[1]:6.1f} {r[2]:6.1f} {r[3]:6.1f} {r[4]:6.1f} {r[5]:8.3f} {r[6]:7.0f} {r[7]:6.0f} {r[8]:7.0f} {r[9]:7.0f} | {r[10]:6.2f} {r[11]:6.2f}")
    lai_l=rows[-1][13]; fl_=rows[-1][12]
    print(f"   daily leaf-level Ag C4 (positive part): {sum(max(r[11],0) for r in rows)*3600*12e-6:.3f} gC per m2 LEAF; canopy-scale ~ x LAI_l {lai_l:.2f} x fl {fl_:.2f} x2 = {sum(max(r[11],0) for r in rows)*3600*12e-6*lai_l*fl_*2:.3f} gC/m2 ground/day (crude, no canopy scaling)")
    return st, key
st,key=one_day(st,key,2003,0,14,"15 January 2003")
st,key=one_day(st,key,2003,6,14,"15 July 2003")
