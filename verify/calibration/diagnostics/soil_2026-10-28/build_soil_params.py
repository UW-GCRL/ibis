"""Build a soil-stage parameter file from a base file: som_k_scale per zone group on the zones' tree types (CALIBRATION.md 72).
usage: python -I build_soil_params.py BASE.yaml OUT.yaml strong|mild"""
import sys, re, yaml
sys.path[:0] = ['/shared/nas2/Hangkai/ibis-port/python-port']
from ibis.io import pft_types as P
base, out, variant = sys.argv[1:4]
FACT = {'strong': {'temperate': 0.5, 'boreal': 0.35, 'tundra_cold': 0.6, 'savanna_hot': 0.7, 'subtrop': 0.9, 'humid': 1.0},
        'mild':   {'temperate': 0.6, 'boreal': 0.45, 'tundra_cold': 0.75, 'savanna_hot': 0.8, 'subtrop': 1.0, 'humid': 1.0}}[variant]
# zone group -> (types that carry the value; one per zone is enough, 'all' takes the median over those that carry one)
CARRY = {'temperate': ['T16', 'T12'], 'boreal': ['T18', 'T19'], 'tundra_cold': ['T17', 'T21'], 'savanna_hot': ['T03', 'T05'], 'subtrop': ['T06', 'T04', 'T09'], 'humid': ['T01']}
# T17 is in D_subarctic, D_extreme AND E_tundra: it carries the tundra value, T18/T19 the boreal one (zone median of the carriers: subarctic 0.35, tundra 0.6)
s = open(base).read()
s = f"# {out.split('/')[-1]} ({variant} soil-stage variant, CALIBRATION.md 72): base {base.split('/')[-1]} + som_k_scale per zone group on the zones' tree types.\n" + s
def type_block(tid):
    m = re.search(rf"\n  {tid}:\n((?:    .*\n)+)", s); assert m, tid; return m
for grp, tids in CARRY.items():
    f = FACT[grp]
    if f == 1.0: continue
    for tid in tids:
        m = type_block(tid); s = s.replace(m.group(0), m.group(0) + f"    som_k_scale: {f}    # 2026-10-28 soil stage ({variant}): SOM turnover factor of the {grp} group (CALIBRATION.md 72)\n")
open(out, 'w').write(s)
y = yaml.safe_load(open(out)); over = P.parse_type_overrides(y['types'])
print(f'{out}: zone som_k_scale (None = 1):')
for z in P.ZONES:
    if z in ('none', 'E_ice'): continue
    print(f'  {z:14s}', P.zone_scalar(z, over.get('som_k_scale', {}), 'all'))
