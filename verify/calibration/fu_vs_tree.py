"""Upper-canopy cover `fu` against the CLM tree fraction, globally.

    ibis-python verify/calibration/fu_vs_tree.py <validation out dir with ibis_yearly_*.nc>

Averages the last 5 yearly files. The tree fraction is PFTs 1-8 of
`PCT_NAT_PFT` in the no-crop `all_forests` surfdata -- a potential-vegetation
field like IBIS's own -- on cells with >= 70 % natural vegetation (drops ice
and lakes). Reports fu vs tree by tree-fraction bin and latitude band, with
the GPP mod/obs ratio and each class's share of the global positive GPP
excess. CALIBRATION.md 14: the fourth calibration criterion (the
tree-establishment gate's direct target). v11d baseline: fu 0.66 vs 0.34.
"""
import sys, glob, re, numpy as np, xarray as xr
sys.path.insert(0,'python-port'); sys.path.insert(0,'verify/calibration')
from fit_carbon import LATV, MASK, MODIS_BASE, fluxcom_at
V='/shared/nas2/Hangkai/ibis/verify_outputs'
m=np.array(xr.open_dataset(MASK).landfrac.values)>0
OUT=sys.argv[1] if len(sys.argv)>1 else f'{V}/val_v11d/fit/out'
fs=sorted(glob.glob(f'{OUT}/ibis_yearly_*.nc'),key=lambda p:int(re.search(r'(\d+)\.nc$',p).group(1)))[-5:]
acc={k:0.0 for k in ('fu','fl','laiu','lail','aygpptot')}
for f in fs:
    d=xr.open_dataset(f)
    for k in acc: acc[k]=acc[k]+np.nan_to_num(np.asarray(d[k].values,float))/len(fs)
sd=xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc"); pp=np.asarray(sd["PCT_NAT_PFT"].values); pv=np.asarray(sd["PCT_NATVEG"].values)/100.0
if float(sd["LATIXY"].values[0,0])<float(sd["LATIXY"].values[-1,0]): pp=pp[:,::-1,:]; pv=pv[::-1,:]
tree_nat=pp[1:9].sum(axis=0)/100.0
i,j=np.where(m); a=np.cos(np.deg2rad(LATV[i])); fu=acc['fu'][i,j]; fl=acc['fl'][i,j]; g=acc['aygpptot'][i,j]; gobs=fluxcom_at(i,j)
tn=tree_nat[i,j]; nat=pv[i,j]>=0.7; okg=np.isfinite(gobs)&(gobs>0.02)&np.isfinite(g)
def wm(x,w): return float(np.sum(x*w)/np.sum(w))
def row(lbl,k):
    k2=k&okg
    if k.sum()==0: return
    print(f"{lbl:34s} n={k.sum():5d} area%={100*a[k].sum()/a[nat].sum():5.1f}  fu={wm(fu[k],a[k]):.2f} tree={wm(tn[k],a[k]):.2f}  fl={wm(fl[k],a[k]):.2f}  GPP mod/obs={np.sum(a[k2]*g[k2])/np.sum(a[k2]*gobs[k2]):.2f}  excess%={100*np.sum(a[k2]*np.maximum(g[k2]-gobs[k2],0))/EX:.0f}")
EX=np.sum(a[okg]*np.maximum(g[okg]-gobs[okg],0))
print(f"{OUT}\nyears {fs[0][-7:-3]}-{fs[-1][-7:-3]}; land cells {m.sum()}, natural-veg>=70%: {nat.sum()} ({100*a[nat].sum()/a.sum():.0f}% of land area); positive GPP excess EX={EX:.1f} (area-wtd, all land)")
print(f"ALL natural cells: area-wtd fu={wm(fu[nat],a[nat]):.3f} vs tree={wm(tn[nat],a[nat]):.3f}; MAD(tree term)={wm(np.abs(fu[nat]-tn[nat]),a[nat]):.3f}; RMSE={np.sqrt(wm((fu[nat]-tn[nat])**2,a[nat])):.3f}; r={np.corrcoef(fu[nat],tn[nat])[0,1]:.2f}; fu floor 0.25 -> min fu={fu[nat].min():.2f}")
print("\n-- by CLM tree fraction (natural cells) --")
for lo,hi in ((0,0.05),(0.05,0.15),(0.15,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.01)): row(f"tree {lo:.2f}-{hi:.2f}", nat&(tn>=lo)&(tn<hi))
print("\n-- by latitude (natural cells) --")
for lo,hi in ((60,90),(45,60),(30,45),(15,30),(0,15),(-15,0),(-30,-15),(-60,-30)): row(f"lat {lo:+d}..{hi:+d}", nat&(LATV[i]>=lo)&(LATV[i]<hi))
print("\n-- phantom / missing trees (natural cells) --")
row("PHANTOM: tree<0.15 & fu>0.40", nat&(tn<0.15)&(fu>0.40))
row("PHANTOM: tree<0.15 & fu>0.60", nat&(tn<0.15)&(fu>0.60))
row("MISSING: tree>0.50 & fu<0.30", nat&(tn>0.50)&(fu<0.30))
row("agree: |fu-tree|<0.15", nat&(np.abs(fu-tn)<0.15))
print("\n-- crop-dominated cells (natveg<0.3), for reference --"); row("natveg<0.3", (pv[i,j]<0.3))
