"""params_31j (road A) / params_31k (road B) = params_31h / params_31i + the audit fixes of CALIBRATION.md 62-63.
    python verify/calibration/build_params_31jk.py
Text edits on the source files (comments kept), then a yaml round trip to verify."""
import re, yaml
D = '/shared/nas2/Hangkai/ibis-port/python-port/configs/params'; V = lambda x: f'{x:.15g}'
for src, dst, road in (('params_31h', 'params_31j', 'road A'), ('params_31i', 'params_31k', 'road B')):
    t = open(f'{D}/{src}.yaml').read(); y = yaml.safe_load(t); T = y['types']; base = y['stomata']['vmax']
    def sub1(old, new):
        global t; assert t.count(old) == 1, (dst, old[:60], t.count(old)); t = t.replace(old, new)
    sub1('  coefm_mix: true\n', "  coefm_mix: true\n  vmax_mix: true          # 2026-10-26: each type's leaves at its own capacity (CALIBRATION.md 63)\n"
         "  gpp_split_by_capacity: true   # 2026-10-26: each PFT's GPP share = leaf area x its own capacity, so capacity competes (CALIBRATION.md 63)\n")
    sub1('  bperm: 0.04\n', '  bperm: 0.04\n  h2o_consistent_weights: true   # 2026-10-26: soilh2o linearisation pump fixed (CALIBRATION.md 62)\n')
    sub1(f"  tree_stress_ramp: {y['pft']['tree_stress_ramp']}\n", f"  tree_stress_ramp: {y['pft']['tree_stress_ramp']}\n  exact_litterfall: true         # 2026-10-26: carbon-conserving litterfall booking (CALIBRATION.md 63)\n")
    sub1('  fu_width: 0.10\n', '  fu_width: 0.05                 # 2026-10-26: damping floor 0.08 at the 0.975 cover cap (was 0.22 at 0.10)\n')
    def set_type(tid, key, val, note):
        global t
        blk = re.search(rf'(\n  {tid}:\n(?:    .*\n)+)', t); assert blk, tid
        b = blk.group(1); m = re.search(rf'\n    {key}: ([^\n#]+)', b); assert m, (tid, key)
        t = t.replace(b, b.replace(f'\n    {key}: {m.group(1)}', f'\n    {key}: {V(val)}   # 2026-10-26: {note} (was {m.group(1).strip()})', 1), 1)
    # types whose parent position was dead at their towers (the big leaf read another position's vmax): the value their
    # leaves actually ran at -- behaviour-preserving (section 17's rule) -- instead of the flat fit. Respiration fits stay.
    set_type('T02', 'vmax', T['T01']['vmax'], "T01's; dead position 1 at its EBF towers, whose leaves ran at the fitted evergreen capacity; T02 now A_savanna only")
    set_type('T03', 'vmax', base['temperate_broadleaf_deciduous'], 'base temp_bl_dec, what its leaves ran at (SAV/WSA towers prescribe no position 0/2/4)')
    set_type('T18', 'vmax', base['temperate_broadleaf_deciduous'], 'base temp_bl_dec (dead position 6 at its tower)')
    set_type('T19', 'vmax', T['T17']['vmax'], "T17's (larch leaves ran at the picked conifer vmax; dead position 7)")
    for s in ('S02', 'S04', 'S06'): set_type(s, 'vmax', base['evergreen_shrub'], 'generic evergreen shrub (the shrub big leaf reads position 8; dead position 9)')
    t = t.replace(f'# F7 fire fix (CALIBRATION.md 59, 2026-10-26): {road}', f'# F7 fire fix (CALIBRATION.md 59, 2026-10-26) + the audit fixes of CALIBRATION.md 62-63: {road}', 1)
    open(f'{D}/{dst}.yaml', 'w').write(t)
    z = yaml.safe_load(open(f'{D}/{dst}.yaml')); assert len(z['types']) == 35
    print(dst, 'vmax_mix', z['stomata']['vmax_mix'], 'split', z['stomata']['gpp_split_by_capacity'], 'h2o', z['soil']['h2o_consistent_weights'], 'exact_lf', z['pft']['exact_litterfall'], 'fu_width', z['fire']['fu_width'],
          '| T02', {k: round(z['types']['T02'][k] * (1e6 if k == 'vmax' else 1), 3) for k in ('vmax', 'rgrowth', 'rroot', 'lt_e0')},
          '| vmax umol T03 %.1f T18 %.1f T19 %.1f S02 %.1f S04 %.1f S06 %.1f' % tuple(z['types'][k]['vmax'] * 1e6 for k in ('T03', 'T18', 'T19', 'S02', 'S04', 'S06')))
