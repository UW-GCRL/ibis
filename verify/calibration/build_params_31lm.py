"""params_31l (road A) / params_31m (road B) = params_31j / params_31k + the closed-canopy climate rule for fire (CALIBRATION.md
65) and, on road B only, the cold open-land types' leaf allocation x0.77 (G07, G08, S05, S06: their far north ran to LAI
1.9x the products under the corrected snow; LAI ~ k^2.3 at low leaf area, section 55, so 0.55^(1/2.3) = 0.77).
    python verify/calibration/build_params_31lm.py"""
import re, yaml
D = '/shared/nas2/Hangkai/ibis-port/python-port/configs/params'; V = lambda x: f'{x:.15g}'
for src, dst, road in (('params_31j', 'params_31l', 'A'), ('params_31k', 'params_31m', 'B')):
    t = open(f'{D}/{src}.yaml').read(); y = yaml.safe_load(t)
    def sub1(old, new):
        global t; assert t.count(old) == 1, (dst, old[:60], t.count(old)); t = t.replace(old, new)
    sub1('  use_canopy_damping: true\n', '  use_canopy_damping: true\n  closed_canopy_zones: [A_humid]   # 2026-10-26: humid-tropics climate damps fire as a closed canopy whatever the cover (CALIBRATION.md 65)\n')
    if road == 'B':
        for tid, parent in (('G07', 'c3_grass'), ('G08', 'c3_grass'), ('S05', 'evergreen_shrub'), ('S06', 'deciduous_shrub')):
            blk = re.search(rf'(\n  {tid}:\n(?:    .*\n)+)', t); assert blk, tid; b = blk.group(1)
            m = re.search(r'\n    aleaf: ([^\n#]+)', b)
            if m:
                old_v = float(m.group(1)); b2 = b.replace(f'\n    aleaf: {m.group(1)}', f'\n    aleaf: {V(0.77 * old_v)}   # 2026-10-26: x0.77, far north under the corrected snow (CALIBRATION.md 65) (was {m.group(1).strip()})', 1)
            else:
                base = float(y['pft'][parent]['aleaf']); b2 = b.rstrip('\n') + f'\n    aleaf: {V(0.77 * base)}   # 2026-10-26: x0.77 of the parent {parent} ({base}), far north under the corrected snow (CALIBRATION.md 65)\n'
            t = t.replace(b, b2, 1)
    t = t.replace('+ the audit fixes of CALIBRATION.md 62-63:', '+ the audit fixes of CALIBRATION.md 62-63 + the closed-canopy climate rule of 65:', 1)
    open(f'{D}/{dst}.yaml', 'w').write(t)
    z = yaml.safe_load(open(f'{D}/{dst}.yaml')); assert len(z['types']) == 35
    print(dst, 'fire zones', z['fire']['closed_canopy_zones'], '| aleaf G07 %.3f G08 %.3f S05 %s S06 %s' % (z['types']['G07'].get('aleaf', float('nan')), z['types']['G08'].get('aleaf', float('nan')), z['types']['S05'].get('aleaf'), z['types']['S06'].get('aleaf')))
