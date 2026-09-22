"""Surface-pressure probe: after the fix (atm.psurf carries GSWP3 ps), run the worst negative-ET cells of the v12g closing hindcast plus plateau and sea-level controls for 2 years and read monthly ET, GPP, ground temperature and the pressure the physics saw."""
import sys, glob, re, numpy as np, xarray as xr, netCDF4 as nc, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, LONV, MASK, GSWP3, NPOI_FULL, CachedForcing, apply_params, fluxcom_at, fluxcom_monthly_at
import site_transient; from site_transient import one_year
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum; from ibis.state.et_parts import create_et_parts
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V='/shared/nas2/Hangkai/ibis/verify_outputs'; S='/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'
m=np.array(xr.open_dataset(MASK).landfrac.values)>0; fl=np.where(m.ravel())[0]; posmap={g:p for p,g in enumerate(fl)}
worst=[tuple(x) for x in np.load(f'{S}/neg_et_cells.npz')['worst'][:6]]
ctrl={"Mongolia 47N 105E (1300 m)":(47.25,105.25),"Great Basin 40N 116W (1600 m)":(40.25,-116.25),"Tibet 33N 90E (4800 m)":(33.25,90.25),"Kenya 0N 37E (1700 m)":(0.25,37.25),
      "Amazon 3S 60W (sea level)":(-3.25,-60.25),"Kazakh steppe 50N 70E (300 m)":(50.25,70.25),"Iowa 42N 93W (300 m)":(42.25,-93.25)}
cells=[(f"worst{k+1} {LATV[i]:.1f}N {LONV[j]:.1f}E",i,j) for k,(i,j) in enumerate(worst)]
for nm,(la,lo) in ctrl.items():
    i=int(np.argmin(np.abs(LATV-la))); j=int(np.argmin(np.abs(LONV-lo)))
    if i*720+j in posmap: cells.append((nm,i,j))
    else: print("not land:",nm)
names=[c[0] for c in cells]; i=np.array([c[1] for c in cells],np.int32); j=np.array([c[2] for c in cells],np.int32); sel=np.array([posmap[a*720+b] for a,b in zip(i,j)])
lat_rad=np.deg2rad(LATV[i]).astype(np.float32)
# hindcast (before the fix) monthly ET for the same cells: Aug and annual mean, 2001-2010
fs=sorted(glob.glob(f'{V}/final_v12g/hindcast/out/ibis_monthly_*.nc'))
ET=np.array([np.asarray(nc.Dataset(f)['amaet'][:],float)[i,j] for f in fs]); TG=np.array([np.asarray(nc.Dataset(f)['tg'][:],float)[i,j] for f in fs]); ET=np.where(np.abs(ET)>1e30,np.nan,ET)
mon=np.array([int(re.search(r'_(\d\d)\.nc$',f).group(1)) for f in fs])
before_aug=np.nanmean(ET[mon==8],0); before_ann=np.nanmean(ET,0)*12; before_tg_aug=np.nanmean(TG[mon==8],0)
full,_=load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz',create_ibis_state(GridConfig(npoi=NPOI_FULL,nbedrock=3)))
cut=lambda x: x[sel] if isinstance(x,jnp.ndarray) and x.ndim>=1 and x.shape[0]==NPOI_FULL else x
st=_cast_tree_f32(jax.tree.map(cut,full)); st=eqx.tree_at(lambda s: s.stress, st, create_stress_accum(len(sel)), is_leaf=lambda x: x is None); st=eqx.tree_at(lambda s: s.etparts, st, create_et_parts(len(sel)), is_leaf=lambda x: x is None)
sp,pft,_=build_all_params(len(sel),lat_rad,load_physics_params(f'{V}/fit_carbon_v12g/params_fit_v12g.yaml'))
rd=CachedForcing(GSWP3Reader(GSWP3),[2001,2002],i,j); consts=PhysicalConstants()
site_transient.apply_co2=lambda q,y,series=None: eqx.tree_at(lambda z:z.atm.co2conc,q,389.8e-6)
ctx=dict(i=i,j=j,lat_rad=lat_rad,sp=sp,pft=pft,consts=consts,reader=rd)
print("psurf in the start state (Pa):", np.unique(np.asarray(st.atm.psurf)))
key=jax.random.PRNGKey(0)
for n,yr in enumerate((2001,2002)):
    st,key,d=one_year(st,key,yr,ctx,1)
ps=np.asarray(st.atm.psurf); aet=d['aet_monthly']; gpp=d['gpp_monthly']; tgm=None
go=fluxcom_at(i,j)
print("psurf after 2 years (Pa) = the forcing pressure the physics now sees\n")
print(f"{'cell':34s} {'ps hPa':>7} | {'ET Aug':>7} {'ET yr':>7} | before: {'Aug':>7} {'yr':>7} {'tgAug':>6} | {'GPP yr':>7} {'obs':>6} | monthly ET 2002 (mm)")
for k,nm in enumerate(names):
    print(f"{nm:34s} {ps[k]/100:7.0f} | {aet[7,k]:7.1f} {aet[:,k].sum():7.0f} | {before_aug[k]:7.1f} {before_ann[k]:7.0f} {before_tg_aug[k]-273.15:6.1f} | {gpp[:,k].sum():7.2f} {go[k]:6.2f} | "+" ".join(f"{x:5.0f}" for x in aet[:,k]))
print("\nET components 2002 (mm): trans interc soilev snowsub")
ep=d['et_parts']
for k,nm in enumerate(names): print(f"{nm:34s} "+" ".join(f"{x:7.0f}" for x in ep[:,k]))
