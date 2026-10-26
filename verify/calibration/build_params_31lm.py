"""params_31l (road A) / params_31m (road B) = params_31j / params_31k + the closed-canopy climate rule for fire (CALIBRATION.md
65) and, on road B only, the deciduous shrub's leaf allocation x0.5 (the generic position-9 shrub and the tundra type S06): with the
corrected snow its far-north leaf area went 0.19 -> 1.45 (60-90 N), the whole of road B's run-away there; LAI ~ k^2.3 at
low leaf area (section 55), so 0.2^(1/2.3) = 0.5. The grasses and the evergreen shrub did not move and are left alone.
    python verify/calibration/build_params_31lm.py"""
import re, yaml
D = '/shared/nas2/Hangkai/ibis-port/python-port/configs/params'; V = lambda x: f'{x:.15g}'
for src, dst, road in (('params_31j', 'params_31l', 'A'), ('params_31k', 'params_31m', 'B')):
    t = open(f'{D}/{src}.yaml').read(); y = yaml.safe_load(t)
    def sub1(old, new):
        global t; assert t.count(old) == 1, (dst, old[:60], t.count(old)); t = t.replace(old, new)
    sub1('  use_canopy_damping: true\n', '  use_canopy_damping: true\n  closed_canopy_zones: [A_humid]   # 2026-10-26: humid-tropics climate damps fire as a closed canopy whatever the cover (CALIBRATION.md 65)\n')
    if road == 'B':
        base = float(y['pft']['deciduous_shrub']['aleaf'])
        blk = re.search(r'(\n  deciduous_shrub:\n(?:    .*\n)+)', t); assert blk; b = blk.group(1)
        m = re.search(r'\n    aleaf: ([^\n#]+)', b); assert m
        t = t.replace(b, b.replace(f'\n    aleaf: {m.group(1)}', f'\n    aleaf: {V(0.5 * base)}   # 2026-10-26: x0.5, the far-north run-away under the corrected snow (CALIBRATION.md 65) (was {m.group(1).strip()})', 1), 1)
        mr = re.search(r'\n    aroot: ([^\n#]+)', b); assert mr   # the halved leaf share goes to roots: allocation must sum to one
        # types in the deciduous-shrub position with their own aleaf/awood keep their own fit: pin aroot to the parent's old value
        for code in ('S04',):
            i = t.index(f'\n  {code}:\n'); j = t.index('\n  ', t.index('\n    awood:', i) + 1); t = t[:j] + f'\n    aroot: {mr.group(1).strip()}   # 2026-10-26: pinned to its own fit so {code} keeps summing to one' + t[j:]
        t = t.replace(f'\n    aroot: {mr.group(1)}', f'\n    aroot: {V(float(mr.group(1)) + 0.5 * base)}   # 2026-10-26: takes the halved leaf share so the allocation sums to one (was {mr.group(1).strip()})', 1)
        blk = re.search(r'(\n  S06:\n(?:    .*\n)+)', t); assert blk; b = blk.group(1)
        t = t.replace(b, b.rstrip('\n') + f'\n    aleaf: {V(0.5 * base)}   # 2026-10-26: x0.5 of the parent deciduous shrub ({base}), far north under the corrected snow (CALIBRATION.md 65)\n', 1)
    t = t.replace('+ the audit fixes of CALIBRATION.md 62-63:', '+ the audit fixes of CALIBRATION.md 62-63 + the closed-canopy climate rule of 65:', 1)
    open(f'{D}/{dst}.yaml', 'w').write(t)
    z = yaml.safe_load(open(f'{D}/{dst}.yaml')); assert len(z['types']) == 35
    print(dst, 'fire zones', z['fire']['closed_canopy_zones'], '| pft.deciduous_shrub.aleaf', z['pft']['deciduous_shrub']['aleaf'], '| S06 aleaf', z['types']['S06'].get('aleaf'), '| G07 aleaf', z['types']['G07'].get('aleaf'))
