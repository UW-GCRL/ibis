"""Add the per-type maximum leaf area (docs/CALIBRATION.md 46) to a parameter file.

    python verify/calibration/laicap_yaml.py IN.yaml OUT.yaml [--crop 5.0]

The cap is the canopy LAI per unit of the layer's cover (m2 m-2), held on the year's leaf pool, so it bounds the
seasonal peak. Each value is the midpoint of ORCHIDEE's lai_max (Krinner et al. 2005; ORCHIDEE-MICT source) and
JULES-TRIFFID's Lmax (Clark et al. 2011, Harper et al. 2016) for the matching PFT -- the generous end of the published
caps, since MODIS reads low over needleleaf, crops and dense canopies (Yan et al. 2016; Brown et al. 2020). Shrubs
have no ORCHIDEE PFT and take JULES's 4. The cropland share of the lower canopy takes ORCHIDEE's crop cap of 5
(CLM5 crops: 5-7). Types in the file's `types:` section inherit their parent PFT's cap unless they set `lai_max`.
"""
import argparse
import yaml

#            PFT                                  ORCHIDEE  JULES
LIT = {'tropical_broadleaf_evergreen':        (7.0, 9.0),
       'tropical_broadleaf_deciduous':        (5.0, 7.0),
       'warm_temperate_broadleaf_evergreen':  (4.0, 7.0),
       'temperate_conifer_evergreen':         (5.0, 7.0),
       'temperate_broadleaf_deciduous':       (5.0, 7.0),
       'boreal_conifer_evergreen':            (3.5, 7.0),
       'boreal_broadleaf_deciduous':          (4.0, 7.0),
       'boreal_conifer_deciduous':            (3.0, 6.0),
       'evergreen_shrub':                     (4.0, 4.0),
       'deciduous_shrub':                     (4.0, 4.0),
       'c4_grass':                            (2.0, 3.0),
       'c3_grass':                            (2.5, 3.0)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--crop', type=float, default=5.0, help='cap on the cropland share of the lower canopy')
    a = ap.parse_args()
    d = yaml.safe_load(open(a.inp))
    pft = d.setdefault('pft', {})
    for name, (orc, jul) in LIT.items():
        pft.setdefault(name, {})['lai_max'] = 0.5 * (orc + jul)
    d.setdefault('crop', {})['lai_max'] = a.crop
    with open(a.out, 'w') as f:
        f.write(f'# {a.inp.split("/")[-1]} + per-type maximum leaf area (docs/CALIBRATION.md 46; verify/calibration/laicap_yaml.py)\n')
        yaml.safe_dump(d, f, sort_keys=False)
    print(a.out, {k: pft[k]['lai_max'] for k in LIT}, 'crop', a.crop)
