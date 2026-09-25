"""Monthly GPP climatology of two site_fit baselines against the towers, and the cold-season share (CALIBRATION.md 22).

    python verify/calibration/diagnostics/site_ab_monthly.py <dir_on> <dir_off>

Each directory is a `site_fit.py --only-baseline` output (best_series.npz). Prints, per tower, the 2009-2011 monthly
climatology of tower GPP (mean of the NT and DT partitions, QC >= 0.5), of the `off` run and of the `on` run, and the
share of annual GPP in November-February; then the median share over towers with at least 10 observed months.
Produced the conifer A/B of section 22 (smoke_ENF with default_seasonal.yaml, smoke_ENF_off with default.yaml).
"""
import sys, os, numpy as np
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
from site_fit import tower_obs

on = np.load(os.path.join(sys.argv[1], 'best_series.npz'), allow_pickle=True)
off = np.load(os.path.join(sys.argv[2], 'best_series.npz'), allow_pickle=True)
sites = [str(s) for s in on['sites']]; yrs = [1, 2, 3]
print("monthly GPP climatology 2009-2011 (gC/m2/mo), prescribed stands, fixed leaf area")
print(f"{'tower':8s} {'':5s} " + " ".join(f"{m:>4s}" for m in 'J F M A M J J A S O N D'.split()) + "  | cold share (NDJF)")
agg = {'off': [], 'on': [], 'obs': []}
cs = lambda x: np.nansum(x[[10, 11, 0, 1]]) / max(np.nansum(x), 1e-6)
for n, s in enumerate(sites):
    o = tower_obs(s); ob = np.nanmean(o['gpp'][yrs], 0); a = off['gpp'][yrs, :, n].mean(0); b = on['gpp'][yrs, :, n].mean(0)
    for lab, x in (('obs', ob), ('off', a), ('on', b)):
        print(f"{s if lab == 'obs' else '':8s} {lab:5s} " + " ".join(f"{v:4.0f}" if np.isfinite(v) else "   ." for v in x) + f"  | {cs(x):.2f}")
        if np.isfinite(ob).sum() >= 10: agg[lab].append(cs(x))
print(f"\nmedian cold-season share over towers: tower {np.median(agg['obs']):.3f}  model off {np.median(agg['off']):.3f}  "
      f"model on {np.median(agg['on']):.3f}  (n={len(agg['obs'])})")
