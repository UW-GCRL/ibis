"""Approximate capacity correction from the towers' own leaf area (CALIBRATION.md 49): a test, not the refit.

    python verify/calibration/diagnostics/capacity_pixel.py IN.yaml OUT.yaml [--eps 0.29] [--eta 0.75]

Each S2 fit matched its tower's GPP at the leaf area it held (logs/lai_tower_pixel.csv: `held`); the GLASS 0.05-degree
pixel holding the tower has r = pixel / held times as much. At GPP ~ LAI^eps x vmax^eta, holding the pixel's leaf area
would have needed vmax x r^(-eps/eta) for the same GPP: eps 0.29 is dlnGPP/dlnLAI over the towers' range (section 47's
chain, F/L), eta 0.75 dlnGPP/dlnVmax (both_fix, section 4). Per PFT (the parent of each tower's type) the median factor
over its towers (>= 5 towers, else 1), r clipped to [0.5, 2]; written into stomata.vmax of a copy of IN.yaml. The proper
version refits capacity per type at the towers with the pixel leaf area.
"""
import sys, numpy as np, pandas as pd, yaml
sys.path.insert(0, 'python-port')
from ibis.io import pft_types as T
from ibis.io.physics_params import PFT_NAMES
a = sys.argv[1:]; src, out = a[0], a[1]
eps = float(a[a.index('--eps') + 1]) if '--eps' in a else 0.29; eta = float(a[a.index('--eta') + 1]) if '--eta' in a else 0.75
d = pd.read_csv('/shared/nas2/Hangkai/ibis_local_document/logs/lai_tower_pixel.csv')
TI = {t[0]: t for t in T.TYPES}
d['pft'] = [TI[t][4] - 1 if t in TI else -1 for t in d['type'].astype(str).str.split().str[0]]
d = d[(d.pft >= 0) & np.isfinite(d.glass_pixel_peak) & (d.held > 0) & (d.glass_pixel_peak > 0.05)]
d['r'] = (d.glass_pixel_peak / d.held).clip(0.5, 2.0); d['f'] = d.r ** (-eps / eta)
p = yaml.safe_load(open(src)); vm = p['stomata']['vmax']; lines = []
for j, n in enumerate(PFT_NAMES):
    x = d[d.pft == j]
    f = float(x.f.median()) if len(x) >= 5 else 1.0
    old = float(vm[n]); vm[n] = old * f
    lines.append(f'{n:36s} towers {len(x):3d}  r median {x.r.median() if len(x) else float("nan"):5.2f}  vmax x {f:.3f}  ({old * 1e6:.1f} -> {old * f * 1e6:.1f} umol)')
with open(out, 'w') as fo:
    fo.write(f'# {src.split("/")[-1]} with stomata.vmax x (tower pixel / held leaf area)^(-{eps}/{eta}), median per PFT '
             f'(CALIBRATION.md 49; diagnostics/capacity_pixel.py) -- an approximate test of the capacity refit\n')
    yaml.safe_dump(p, fo, sort_keys=False)
print('\n'.join(lines)); print('wrote', out)
