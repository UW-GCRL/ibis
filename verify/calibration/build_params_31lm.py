"""params_31l (road A) / params_31m (road B) = params_31j / params_31k + the closed-canopy climate rule for fire (CALIBRATION.md
65) and, on road B only, the deciduous shrub's leaf allocation x0.5 (the generic position-9 shrub and the tundra type S06): with the
corrected snow its far-north leaf area went 0.19 -> 1.45 (60-90 N), the whole of road B's run-away there; LAI ~ k^2.3 at
low leaf area (section 55), so 0.2^(1/2.3) = 0.5. The grasses and the evergreen shrub did not move and are left alone.
    python verify/calibration/build_params_31lm.py"""
import re, sys, yaml
D = '/shared/nas2/Hangkai/ibis-port/python-port/configs/params'; OUT = sys.argv[1] if len(sys.argv) > 1 else D; V = lambda x: f'{x:.15g}'
CLOSED_FU = 0.90   # 2026-10-26 (CALIBRATION.md 66): humid floor 0.27; the sweep against GFED5 put the Amazon at 1.1x, trap test check10_31l_fu090
sys.path.insert(0, '/shared/nas2/Hangkai/ibis-port/python-port'); from ibis.io.pft_types import TYPES
for src, dst, road in (('params_31j', 'params_31l', 'A'), ('params_31k', 'params_31m', 'B')):
    t = open(f'{D}/{src}.yaml').read(); y = yaml.safe_load(t)
    def sub1(old, new):
        global t; assert t.count(old) == 1, (dst, old[:60], t.count(old)); t = t.replace(old, new)
    sub1('  use_canopy_damping: true\n', '  use_canopy_damping: true\n  closed_canopy_zones: [A_humid]   # 2026-10-26: humid-tropics climate damps fire as a closed canopy whatever the cover (CALIBRATION.md 65)\n'
         f'  closed_canopy_fu: {CLOSED_FU}   # 2026-10-26: humid damping floor {1 / (1 + 2.718281828 ** ((CLOSED_FU - 0.85) / 0.05)):.2f} (0.08 at the 0.975 cover cap burned the humid zone at a third of GFED5; CALIBRATION.md 66)\n')
    if road == 'B':
        base = float(y['pft']['deciduous_shrub']['aleaf'])
        blk = re.search(r'(\n  deciduous_shrub:\n(?:    .*\n)+)', t); assert blk; b = blk.group(1)
        m = re.search(r'\n    aleaf: ([^\n#]+)', b); assert m
        mr = re.search(r'\n    aroot: ([^\n#]+)', b); assert mr   # the halved leaf share goes to roots: allocation must sum to one
        nb = b.replace(f'\n    aleaf: {m.group(1)}', f'\n    aleaf: {V(0.5 * base)}   # 2026-10-26: x0.5, the far-north run-away under the corrected snow (CALIBRATION.md 65) (was {m.group(1).strip()})', 1)
        nb = nb.replace(f'\n    aroot: {mr.group(1)}', f'\n    aroot: {V(float(mr.group(1)) + 0.5 * base)}   # 2026-10-26: takes the halved leaf share so the allocation sums to one (was {mr.group(1).strip()})', 1)
        t = t.replace(b, nb, 1)
        # types in the deciduous-shrub position with their own aleaf/awood keep their own fit: pin aroot to the parent's old value
        for code in ('S04',):
            i = t.index(f'\n  {code}:\n'); j = t.index('\n  ', t.index('\n    awood:', i) + 1); t = t[:j] + f'\n    aroot: {mr.group(1).strip()}   # 2026-10-26: pinned to its own fit (the parent deciduous shrub now carries {V(float(mr.group(1)) + 0.5 * base)}) so {code} keeps summing to one' + t[j:]
        blk = re.search(r'(\n  S06:\n(?:    .*\n)+)', t); assert blk; b = blk.group(1)
        t = t.replace(b, b.rstrip('\n') + f'\n    aleaf: {V(0.5 * base)}   # 2026-10-26: x0.5 of the parent deciduous shrub ({base}), far north under the corrected snow (CALIBRATION.md 65)\n', 1)
    t = t.replace('+ the audit fixes of CALIBRATION.md 62-63:', '+ the audit fixes of CALIBRATION.md 62-63 + the closed-canopy climate rule of 65 (humid floor of 66):', 1)
    open(f'{OUT}/{dst}.yaml', 'w').write(t)
    z = yaml.safe_load(open(f'{OUT}/{dst}.yaml')); assert len(z['types']) == 35
    pft = {k: v for k, v in z['pft'].items() if isinstance(v, dict) and 'aleaf' in v}; names = list(pft)   # every allocation sums to one
    bad = [k for k, v in pft.items() if abs(v['aleaf'] + v['aroot'] + v['awood'] - 1) > 1e-6]
    for tt in TYPES:
        spec = z['types'].get(tt[0], {}); bb = pft[names[tt[4] - 1]]
        if abs(sum(float(spec.get(k, bb[k])) for k in ('aleaf', 'aroot', 'awood')) - 1) > 1e-6: bad.append(tt[0])
    assert not bad, (dst, bad)
    print(dst, 'fire zones', z['fire']['closed_canopy_zones'], 'closed_canopy_fu', z['fire']['closed_canopy_fu'], '| pft.deciduous_shrub aleaf/aroot', z['pft']['deciduous_shrub']['aleaf'], z['pft']['deciduous_shrub']['aroot'], '| S06 aleaf', z['types']['S06'].get('aleaf'), '| S04 aroot', z['types']['S04'].get('aroot'), '| allocation sums ok')
