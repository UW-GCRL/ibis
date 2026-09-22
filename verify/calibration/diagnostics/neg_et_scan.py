"""Scan a monthly hindcast for negative total ET (amaet): how many cells and cell-months, where (latitude bands), when (months), and the worst cells with their co-located state. Run 2026-09-21 on final_v12g/hindcast; it located the surface-pressure defect (DEFECTS_FOR_REVIEW.md "sea-level pressure")."""
import sys, glob, re, collections, numpy as np, netCDF4 as nc
OUT = sys.argv[1] if len(sys.argv) > 1 else '/shared/nas2/Hangkai/ibis/verify_outputs/final_v12g/hindcast/out'
fs = sorted(glob.glob(f'{OUT}/ibis_monthly_*.nc'), key=lambda p: int(re.search(r'(\d+)_(\d+)\.nc$', p).group(1)) * 12 + int(re.search(r'(\d+)_(\d+)\.nc$', p).group(2)))
lat = None; ET = []; P = []; FU = []; FL = []; LL = []; TG = []; W1 = []
for f in fs:
    with nc.Dataset(f) as d:
        if lat is None: lat = np.asarray(d['lat'][:]); lon = np.asarray(d['lon'][:])
        ET.append(np.asarray(d['amaet'][:], float)); P.append(np.asarray(d['amprcp'][:], float)); FU.append(np.asarray(d['fu'][:], float)); FL.append(np.asarray(d['fl'][:], float))
        LL.append(np.asarray(d['lail'][:], float)); TG.append(np.asarray(d['tg'][:], float)); W1.append(np.asarray(d['wsoi'][:, :, 0], float))
ET = np.array(ET); P = np.array(P); FU = np.array(FU); FL = np.array(FL); LL = np.array(LL); TG = np.array(TG); W1 = np.array(W1)
ET = np.where(np.abs(ET) > 1e30, np.nan, ET)
print(f"{len(fs)} monthly files; mean |amaet| over land {np.nanmean(np.abs(ET)):.1f} mm/month")
neg = ET < 0; anyneg = np.nanmin(ET, axis=0) < 0
print("cells with ANY negative month:", anyneg.sum(), "of", np.isfinite(ET[0]).sum(), " cells with < -1 mm:", (np.nanmin(ET, 0) < -1).sum(), " < -3 mm:", (np.nanmin(ET, 0) < -3).sum())
print(f"negative cell-months: {neg.sum()} of {np.isfinite(ET).sum()} ({100 * neg.sum() / np.isfinite(ET).sum():.2f} %)")
ii, jj = np.where(anyneg)
print("15-degree latitude bands of cells with any negative month:", collections.Counter((lat[ii] // 15 * 15).astype(int)).most_common())
mon = np.arange(len(fs)) % 12 + 1
print("month histogram of negative cell-months:", collections.Counter(mon[np.where(neg)[0]]).most_common())
mn = np.nanmin(ET, axis=0); mn = np.where(np.isfinite(mn), mn, 0); order = np.argsort(mn.ravel())[:12]
print("\nworst cells (min monthly amaet):")
print(f"{'i':>4}{'j':>4} {'lat':>6}{'lon':>7} {'minET':>7} {'mon':>4} {'prcp':>6} {'fu':>5}{'fl':>5}{'lail':>5} {'tg':>6} {'wsoi1':>6}  mean monthly ET  months<0")
for o in order:
    i, j = divmod(o, ET.shape[2]); k = np.nanargmin(ET[:, i, j])
    print(f"{i:4d}{j:4d} {lat[i]:6.1f}{lon[j]:7.1f} {ET[k, i, j]:7.2f} {mon[k]:4d} {P[k, i, j]:6.2f} {FU[k, i, j]:5.2f}{FL[k, i, j]:5.2f}{LL[k, i, j]:5.2f} {TG[k, i, j]:6.1f} {W1[k, i, j]:6.2f}  {np.nanmean(ET[:, i, j]):6.2f}  {int(neg[:, i, j].sum())}/{len(fs)}")
