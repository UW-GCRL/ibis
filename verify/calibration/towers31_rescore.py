"""Re-score every campaign tower under the IBIS 3.0 per-type set (CALIBRATION.md 50): per-tower yamls and commands.

    IBIS_FLUX_DIR=<towers31>/compact python verify/calibration/towers31_rescore.py <params_31a.yaml> <base.yaml> \\
        <s2 cmds> <open refit cmds> <yaml dir> <cmds out> [--out-root DIR] [--prefix /Users/hyou/nas2]

A tower runs its types at their parent positions (section 45.2), so the 3.0 values it would see in the global run are
its types' `types:` entries written into the 12-position base: vmax (stomata.vmax, long names), coefm / acclim
(stomata.coefm_pft, acclim_s0_pft, acclim_smax_pft, short names), lt_e0 / rgrowth / rroot (bgc, short names) per type at
its parent; the scalars (gdd_thresh_u, gdd_ramp_u, sen_ycrit_u from the tree types; gdd_thresh_l, tthresh_l_c from the
lower types; drop_stress_thresh, stressfac from all) as the mean over the tower's types that carry them. Each command is
the tower's own fit line (the open refit's for GRA/OSH/CSH/SAV/WSA, else the S2 line) with the per-tower yaml, no start
point, `--only-baseline --workers 1`: site_fit then writes best.json (the loss of the 3.0 type set on the tower's objective)
and best_series.npz (its monthly GPP, ET, NEE). --prefix rewrites the NAS prefix for a mirror (the Mac).
"""
import sys, os, re, yaml, numpy as np, pandas as pd
sys.path[:0] = ['python-port', 'verify/calibration']
from ibis.io import pft_types as P
from ibis.io.physics_params import PFT_NAMES as LONG, PFT_SHORT_NAMES as SHORT
a = sys.argv[1:]; types_yaml, base_yaml, s2_cmds, open_cmds, ydir, cmds_out = a[:6]
opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
out_root = opt('--out-root', '/shared/nas2/Hangkai/ibis/verify_outputs/site_fit/t31_rescore'); prefix = opt('--prefix', '')
si = pd.read_csv(f"{os.environ['IBIS_FLUX_DIR']}/FLUXNET_siteinfo.csv", encoding='utf-8-sig').set_index('SITE_ID')
T = yaml.safe_load(open(types_yaml))['types']; base = yaml.safe_load(open(base_yaml))
PARENT = {t[0]: t[4] - 1 for t in P.TYPES}; FORM = {t[0]: t[2] for t in P.TYPES}
os.makedirs(ydir, exist_ok=True)
lines = {}
for f, tag in ((s2_cmds, 's2'), (open_cmds, 'open')):
    for l in open(f):
        l = l.split('   #')[0].rstrip()
        if l and not l.startswith('#'): lines.setdefault(l.split()[0], {})[tag] = l
n = 0; out = []
for s, d in lines.items():
    if s not in si.index: continue
    own = [t for t in str(si.loc[s, 'TYPE']).split() if t in T]
    if not own: continue
    y = yaml.safe_load(open(base_yaml)); st = y['stomata']; bg = y['bgc']; ph = y['pheno']
    for key in ('coefm_pft', 'acclim_s0_pft', 'acclim_smax_pft'): st.setdefault(key, {})
    for key in ('lt_e0', 'rgrowth', 'rroot'): bg.setdefault(key, {})
    scal = {}
    for t in own:
        p = PARENT[t]; v = T[t]
        if 'vmax' in v: st['vmax'][LONG[p]] = float(v['vmax'])
        for key in ('coefm', 'acclim_s0', 'acclim_smax'):
            if key in v: st[key + '_pft'][SHORT[p]] = float(v[key])
        for key in ('lt_e0', 'rgrowth', 'rroot'):
            if key in v: bg[key][SHORT[p]] = float(v[key])
        for key in ('gdd_thresh_u', 'gdd_ramp_u', 'sen_ycrit_u', 'gdd_thresh_l', 'tthresh_l_c', 'drop_stress_thresh', 'stressfac'):
            if key in v: scal.setdefault(key, []).append(float(v[key]))
    for key, vals in scal.items():
        (st if key == 'stressfac' else ph)[key] = float(np.mean(vals))
    yp = f'{ydir}/{s}.yaml'
    with open(yp, 'w') as fo:
        fo.write(f'# {s}: types {" ".join(own)} at their parent positions on {os.path.basename(base_yaml)} (towers31_rescore.py)\n'); yaml.safe_dump(y, fo, sort_keys=False)
    line = d.get('open', d['s2'])
    for o in ('--params', '--start', '--out', '--n-evals', '--popsize', '--workers', '--sigma'): line = re.sub(rf' {o} [^ ]+', '', line)
    line = line.replace(' --robust-auto', '')
    line += f' --params {yp} --only-baseline --workers 1 --out {out_root}/{s}'
    if prefix: line = line.replace('/shared/nas2', prefix)
    out.append(line); n += 1
open(cmds_out, 'w').write('\n'.join(out) + '\n')
print(f'{n} towers; yamls in {ydir}; commands {cmds_out}; from the open refit: {sum(1 for s, d in lines.items() if "open" in d and s in si.index)}')
