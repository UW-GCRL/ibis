"""Stage B ranking (CALIBRATION.md 22-23): combine site_sensitivity sweeps and rank the perturbations per metric and class.

    python verify/calibration/diagnostics/stageB_rank.py <sweep_dir> [<sweep_dir> ...]

Each directory is a `site_sensitivity.py` output (sensitivity_by_class.csv, sensitivity_runs.csv). The sweeps must share the
base (same towers, parameter file and harness mode); the base of the first is printed, and the others are checked against it.
Prints: the base by class; for every metric, the perturbations ranked by the largest class-mean change (with the class);
and, for the forest classes, the perturbations that move GPP, the cold-season share and the season dates the most.
"""
import sys, os, numpy as np, pandas as pd

dirs = sys.argv[1:]
by = pd.concat([pd.read_csv(os.path.join(d, 'sensitivity_by_class.csv')).assign(sweep=os.path.basename(d.rstrip('/'))) for d in dirs], ignore_index=True)
runs = [pd.read_csv(os.path.join(d, 'sensitivity_runs.csv')) for d in dirs]
base = runs[0][runs[0]['pert'] == 'base'].set_index('site')
num = [c for c in base.columns if c not in ('pert', 'igbp') and pd.api.types.is_numeric_dtype(base[c])]
for d, r in zip(dirs[1:], runs[1:]):                      # the sweeps must share one base
    b = r[r['pert'] == 'base'].set_index('site').loc[base.index, num]
    dev = float(np.nanmax(np.abs(b.values - base[num].values)))
    print(f'base of {os.path.basename(d)} vs {os.path.basename(dirs[0])}: max abs difference {dev:.2e}')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
cols = ['gpp_ratio', 'junjul_ratio', 'winter_share', 'leafout_doy', 'senesc_doy', 'lai_on_month', 'lai_off_month', 'et_ratio', 'lai_summer']
print(f"\nBASE by class ({len(base)} towers):"); print(base.groupby('igbp')[[c for c in cols if c in base]].mean().round(3).to_string())
metrics = [c for c in by.columns if c.startswith('d_')]
print(f"\n{len(by['pert'].unique())} perturbations from {len(dirs)} sweeps")
for m in metrics:
    g = by.assign(a=by[m].abs()).sort_values('a', ascending=False).drop_duplicates('pert')
    print(f"\n== {m}: largest class-mean change per perturbation (top 14)")
    print(g.head(14)[['pert', 'igbp', m]].round(3).to_string(index=False))
for c in ('ENF', 'DBF', 'MF', 'GRA'):
    sub = by[by['igbp'] == c]
    if not len(sub): continue
    print(f"\n== {c}: top perturbations by |d_gpp_ratio|, with the cold-season share and dates")
    k = [x for x in ('pert', 'd_gpp_ratio', 'd_junjul_ratio', 'd_winter_share', 'd_leafout_days', 'd_senesc_days', 'd_lai_on_month', 'd_lai_off_month', 'd_et_ratio') if x in sub]
    print(sub.assign(a=sub['d_gpp_ratio'].abs()).sort_values('a', ascending=False).head(12)[k].round(3).to_string(index=False))
    for m in ('d_winter_share', 'd_senesc_days', 'd_leafout_days'):
        if m in sub:
            t = sub.assign(a=sub[m].abs()).sort_values('a', ascending=False).head(5)
            print(f"   {m:16s}: " + '; '.join(f"{p} {v:+.3f}" for p, v in zip(t['pert'], t[m])))
