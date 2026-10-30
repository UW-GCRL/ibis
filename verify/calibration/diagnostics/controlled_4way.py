"""Controlled comparison (review 2026-09-22, step 2): SAME cells, SAME forcing years (2001-2005), SAME start state (v11 2010 restart),
SAME parameters; four model variants built from the current loops_gswp3 source by substituting the two fixed lines back:
  no fix / clock fix only / pressure fix only / both fixes.  Two parameter sets: IBIS defaults (switches off) and the joint fit v12g.
Pass 1: dynamic vegetation on (the model as calibrated).  Pass 2: fixed composition (isimveg=0) for process diagnosis.
Scores = mean of years 4-5 (states adjusted to the variant) against FLUXCOM monthly GPP and the 5-product ET ensemble."""
import sys, types, json, numpy as np, xarray as xr, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, LONV, MASK, GSWP3, NPOI_FULL, CachedForcing, fluxcom_at, fluxcom_monthly_at, et_monthly_at, tree_at
import site_transient; from site_transient import one_year
import ibis.timestepping.loops_gswp3 as LG
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum; from ibis.state.et_parts import create_et_parts
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
V='/shared/nas2/Hangkai/ibis/verify_outputs'; S='/shared/nas2/Hangkai/ibis/verify_outputs/diag_work'; R='/shared/nas2/Hangkai/ibis-port'
# ---------------- four variants of the forcing loop, by source substitution ----------------
src=open(f'{R}/python-port/ibis/timestepping/loops_gswp3.py').read()
CLOCK=("zweight = jnp.exp(-1.0 / 10.0)", "zweight = jnp.exp(-1.0 / 240.0)")
PRES=("_cast(jnp.broadcast_to(dw.psurf, st.atm.psurf.shape))", "_cast(jnp.full(st.atm.psurf.shape, 101325.0))")
assert src.count(CLOCK[0])==1 and src.count(PRES[0])==1, "anchor lines changed"
def variant(name, clock_fixed, pres_fixed):
    s=src if clock_fixed else src.replace(*CLOCK); s=s if pres_fixed else s.replace(*PRES)
    m=types.ModuleType(f"loops_gswp3_{name}"); m.__file__=LG.__file__; exec(compile(s, f"<loops_gswp3_{name}>", "exec"), m.__dict__); return m
VARIANTS={"no fix":variant("none",False,False),"clock fix only":variant("clock",True,False),"pressure fix only":variant("pres",False,True),"both fixes":variant("both",True,True)}
# ---------------- cells ----------------
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]; posmap={g:p for p,g in enumerate(fl)}
sel_all=np.load(f'{V}/fit_carbon_v12g/best.npz')['sel']; ia=fl[sel_all]//720; ja=fl[sel_all]%720
stg,_=load_state(f'{V}/val_v12g/fit/restart/ibis_restart_2060.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3),with_stress=True,with_etparts=True))
plai_g=np.asarray(stg.vegetation.plai)[sel_all]; dom=np.argmax(plai_g,1); dom[plai_g.sum(1)<0.05]=-1
go=fluxcom_at(ia,ja); om_all=fluxcom_monthly_at(ia,ja); amp=np.nanmax(om_all,0)/np.maximum(np.nanmin(om_all,0),1e-9)
rng=np.random.default_rng(5); groups={}
def pick(mask,n): k=np.where(mask&np.isfinite(go))[0]; return sel_all[rng.choice(k,min(n,len(k)),replace=False)]
groups["steppe C4 40-55N"]=pick((dom==10)&(LATV[ia]>=40)&(LATV[ia]<55)&(go>0.2),30)
groups["boreal forest 55-70N"]=pick(np.isin(dom,[5,6,7])&(LATV[ia]>=55)&(LATV[ia]<70)&(go>0.3),20)
groups["temperate forest 35-55N"]=pick(np.isin(dom,[3,4])&(LATV[ia]>=35)&(LATV[ia]<55)&(go>0.8),20)
groups["tropical savanna"]=pick((np.abs(LATV[ia])<23)&(amp>=3)&(go>0.5),20)
groups["tropical rainforest"]=pick((np.abs(LATV[ia])<15)&(go>2.5),20)
worst=[tuple(x) for x in np.load(f'{S}/neg_et_cells.npz')['worst'][:6]]
ctrl={"Mongolia":(47.25,105.25),"Great Basin":(40.25,-116.25),"Tibet":(33.25,90.25),"Kenya":(0.25,37.25),"Amazon":(-3.25,-60.25),"Kazakh":(50.25,70.25),"Iowa":(42.25,-93.25)}
fixed=[(f"Himalaya#{k+1}",i,j) for k,(i,j) in enumerate(worst)]+[(nm,int(np.argmin(np.abs(LATV-la))),int(np.argmin(np.abs(LONV-lo)))) for nm,(la,lo) in ctrl.items()]
groups["Himalaya worst 6"]=np.array([posmap[i*720+j] for nm,i,j in fixed[:6]]); groups["plateau + sea-level controls"]=np.array([posmap[i*720+j] for nm,i,j in fixed[6:]])
sel=np.concatenate(list(groups.values())); gid=np.concatenate([[k]*len(v) for k,v in enumerate(groups.values())]); gnames=list(groups)
i=(fl[sel]//720).astype(np.int32); j=(fl[sel]%720).astype(np.int32); lat_rad=np.deg2rad(LATV[i]).astype(np.float32); area=np.cos(np.deg2rad(LATV[i]))
om=fluxcom_monthly_at(i,j); g_o=fluxcom_at(i,j); em=et_monthly_at(i,j); tree_o=tree_at(i,j)
print(f"{len(sel)} cells in {len(groups)} groups: "+", ".join(f"{k} ({len(v)})" for k,v in groups.items()), flush=True)
# ---------------- start state, forcing, parameter sets ----------------
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st0=LG._cast_tree_f32(jax.tree.map(cut,full)); st0=eqx.tree_at(lambda s: s.stress, st0, create_stress_accum(len(sel)), is_leaf=lambda x: x is None); st0=eqx.tree_at(lambda s: s.etparts, st0, create_et_parts(len(sel)), is_leaf=lambda x: x is None)
YEARS=[2001,2002,2003,2004,2005]; rd=CachedForcing(GSWP3Reader(GSWP3),YEARS,i,j); consts=PhysicalConstants()
site_transient.apply_co2=lambda q,y,series=None: eqx.tree_at(lambda z:z.atm.co2conc,q,389.8e-6)
PARAMS={"IBIS defaults":f'{R}/python-port/configs/params/default.yaml',"joint fit v12g":f'{R}/python-port/configs/params/params_fit_v12g.yaml'}
REC=[]
def make_rec(fn):
    def rec(state,*a,**k):
        out=fn(state,*a,**k); st=out[0]; vg=st.vegetation
        REC.append(dict(lai_l=np.asarray(vg.lai[:,0]).copy(), lai_u=np.asarray(vg.lai[:,1]).copy(), fl=np.asarray(vg.fl).copy(), fu=np.asarray(vg.fu).copy(),
                        templ=np.asarray(vg.templ).copy(), tempu=np.asarray(vg.tempu).copy(), a10td=np.asarray(st.summary.a10td).copy(), psurf=np.asarray(st.atm.psurf).copy()))
        return out
    return rec
def wsum(x,k): return float(np.sum(x[k]*area[k]))
def ratio(mod,obs,k,months=None):
    mm=range(12) if months is None else months
    num=sum(wsum(mod[t],k) for t in mm); den=sum(wsum(np.nan_to_num(obs[t]),k) for t in mm); return num/max(den,1e-9)
RESULTS={}
for isimveg,passname in ((1,"dynamic vegetation"),(0,"fixed composition")):
    print(f"\n##################  PASS: {passname} (isimveg={isimveg})  ##################", flush=True)
    for pname,yaml_ in PARAMS.items():
        sp,pft,_=build_all_params(len(sel),lat_rad,load_physics_params(yaml_)); ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
        for vname,mod in VARIANTS.items():
            if isimveg==0 and vname not in ('no fix','both fixes'): continue      # pass 2 trimmed: endpoints only (compile-bound)
            site_transient.run_month_gswp3=make_rec(mod.run_month_gswp3); REC.clear()
            st=st0; key=jax.random.PRNGKey(0); GM=[]; EM=[]; EP=[]
            for n,yr in enumerate(YEARS):
                st,key,d=one_year(st,key,yr,ctx,isimveg)
                if n>=3: GM.append(d['gpp_monthly']); EM.append(d['aet_monthly']); EP.append(d['et_parts'])
            gm=np.mean(GM,0); etm=np.mean(EM,0); ep=np.mean(EP,0); R24=REC[-24:]
            month_state={k_:np.mean([r[k_] for r in R24],0) for k_ in ('fu','fl','psurf')}
            jja=[r for n_,r in enumerate(R24) if n_%12 in (5,6,7)]; lai_l=np.mean([r['lai_l']*r['fl'] for r in jja],0); lai_u=np.mean([r['lai_u']*r['fu'] for r in jja],0)   # grid-mean displayed LAI, JJA
            a10=np.array([np.mean([R24[y*12+mo]['a10td'] for y in range(2)],0) for mo in range(12)])
            tag=f"{passname}|{pname}|{vname}"; RESULTS[tag]=dict(gpp=gm,et=etm,etparts=ep,lai_l=lai_l,lai_u=lai_u,fu=month_state['fu'],fl=month_state['fl'],a10td=a10,psurf=month_state['psurf'])
            print(f"\n=== {tag} ===  (psurf seen by physics: {np.min(month_state['psurf'])/100:.0f}-{np.max(month_state['psurf'])/100:.0f} hPa)", flush=True)
            print(f"{'group':30s} | {'GPP ann':>7} {'Jun-Jul':>7} {'Nov-Feb':>7} {'amp m/o':>7} | {'ET ann':>6} {'ET JJA':>6} | {'LAIl':>5} {'LAIu':>5} {'fu':>5} {'tree':>5} | {'a10td Jan/Jul':>13}")
            for g,gname in enumerate(gnames):
                k=gid==g; NH=LATV[i[k]].mean()>=0
                jj=[5,6] if NH else [11,0]; ndjf=[10,11,0,1] if NH else [4,5,6,7]
                comp=np.array([wsum(gm[t],k) for t in range(12)]); co=np.array([wsum(np.nan_to_num(om[t]),k) for t in range(12)])
                ampm=comp.max()/max(comp.min(),1e-9); ampo=co.max()/max(co.min(),1e-9)
                ek=k&np.all(np.isfinite(em),0)
                et_ann=ratio(etm,em,ek) if ek.sum() else np.nan; et_jja=ratio(etm,em,ek,[5,6,7]) if ek.sum() else np.nan
                tr=np.nanmean(tree_o[k]) if np.isfinite(tree_o[k]).any() else np.nan
                print(f"{gname:30s} | {ratio(gm,om,k):7.2f} {ratio(gm,om,k,jj):7.2f} {ratio(gm,om,k,ndjf):7.2f} {ampm/ampo:7.2f} | {et_ann:6.2f} {et_jja:6.2f} | {np.mean(lai_l[k]):5.2f} {np.mean(lai_u[k]):5.2f} {np.mean(month_state['fu'][k]):5.2f} {tr:5.2f} | {np.mean(a10[0][k])-273.15:6.1f}/{np.mean(a10[6][k])-273.15:6.1f}", flush=True)
            k=gid==gnames.index("Himalaya worst 6"); kc=gid==gnames.index("plateau + sea-level controls")
            print("   Himalaya worst 6 -- ET mm/yr per cell: "+" ".join(f"{etm[:,c].sum():5.0f}" for c in np.where(k)[0])+"   | controls "+" ".join(f"{fixed[6+n_][0]}:{etm[:,c].sum():.0f}" for n_,c in enumerate(np.where(kc)[0])), flush=True)
            print("   Himalaya worst 6 -- GPP model/obs per cell: "+" ".join(f"{gm[:,c].sum()/max(g_o[c],1e-9):5.2f}" for c in np.where(k)[0])+"   | controls "+" ".join(f"{fixed[6+n_][0]}:{gm[:,c].sum()/max(g_o[c],1e-9):.2f}" for n_,c in enumerate(np.where(kc)[0])), flush=True)
    np.savez(f'{S}/controlled_4way_{isimveg}.npz', sel=sel, gid=gid, gnames=np.array(gnames), i=i, j=j, om=om, em=em, g_o=g_o, tree_o=tree_o, tags=np.array(list(RESULTS)), **{f"{n_}_{k_}":v_ for n_,(t_,r_) in enumerate(RESULTS.items()) for k_,v_ in r_.items()})
    print(f"saved controlled_4way_{isimveg}.npz", flush=True)
