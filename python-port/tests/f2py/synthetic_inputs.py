"""Synthetic test data generators for process-by-process verification.

Each scenario covers a distinct physical regime. NPOI=5 grid cells:
  [0] Warm humid tropical (300K, wet soil, no snow, dense canopy)
  [1] Cold arctic tundra  (250K, icy soil, deep snow, sparse veg)
  [2] Freeze/thaw edge    (273.16K, mixed soil, thin snow, moderate veg)
  [3] Hot arid desert      (315K, very dry soil, no snow, bare ground)
  [4] Saturated wetland    (285K, saturated soil, no snow, moderate canopy)
"""
from __future__ import annotations

import numpy as np

NPOI = 5
NSOILAY = 6
NSNOLAY = 3
NPFT = 12
NBAND = 2

# Default soil layer thicknesses (m) matching Fortran initial.f
HSOI = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00, 0.0])

# Physical constants matching Fortran inisurf
TMELT = 273.16
HVAP = 2.5104e6
HFUS = 0.3336e6
HSUB = HVAP + HFUS
CH2O = 4.218e3
CICE = 2.106e3
CAIR = 1.00464e3
CVAP = 1.81e3
RHOW = 1.0e3
EPSILON = 1.0e-7


def make_soil_inputs() -> dict:
    """Soil state for all 5 scenarios."""
    tsoi = np.zeros((NPOI, NSOILAY))
    wsoi = np.zeros((NPOI, NSOILAY))
    wisoi = np.zeros((NPOI, NSOILAY))
    sand = np.zeros((NPOI, NSOILAY))
    clay = np.zeros((NPOI, NSOILAY))

    # [0] Warm humid: warm soil, moderate moisture
    tsoi[0] = 295.0; wsoi[0] = 0.6; sand[0] = 30.0; clay[0] = 30.0
    # [1] Arctic: cold, icy
    tsoi[1] = 255.0; wsoi[1] = 0.1; wisoi[1] = 0.5; sand[1] = 60.0; clay[1] = 10.0
    # [2] Freeze/thaw
    tsoi[2] = 273.16; wsoi[2] = 0.3; wisoi[2] = 0.3; sand[2] = 40.0; clay[2] = 20.0
    # [3] Hot arid
    tsoi[3] = 320.0; wsoi[3] = 0.05; sand[3] = 90.0; clay[3] = 5.0
    # [4] Saturated
    tsoi[4] = 283.0; wsoi[4] = 0.95; sand[4] = 20.0; clay[4] = 40.0

    poros = np.full((NPOI, NSOILAY), 0.45)
    hydraul = np.full((NPOI, NSOILAY), 1e-5)
    suction = np.full((NPOI, NSOILAY), -0.15)
    bex = np.full((NPOI, NSOILAY), 5.0)
    sfield = np.full((NPOI, NSOILAY), 0.35)
    swilt = np.full((NPOI, NSOILAY), 0.15)
    rhosoi = np.full((NPOI, NSOILAY), 2650.0)

    wpud = np.array([0.0, 0.0, 2.0, 0.0, 5.0])
    wipud = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    tg = np.array([295.0, 255.0, 273.16, 320.0, 283.0])
    ti = np.array([273.16, 255.0, 270.0, 273.16, 273.16])

    return dict(
        tsoi=tsoi, wsoi=wsoi, wisoi=wisoi, sand=sand, clay=clay,
        poros=poros, hydraul=hydraul, suction=suction, bex=bex,
        sfield=sfield, swilt=swilt, rhosoi=rhosoi,
        wpud=wpud, wipud=wipud, tg=tg, ti=ti, hsoi=HSOI,
    )


def make_snow_inputs() -> dict:
    """Snow state for all 5 scenarios."""
    fi = np.array([0.0, 0.8, 0.3, 0.0, 0.0])
    tsno = np.full((NPOI, NSNOLAY), TMELT)
    tsno[1] = 250.0  # arctic: cold snow
    tsno[2] = 270.0  # freeze/thaw: near-melting
    hsno = np.zeros((NPOI, NSNOLAY))
    hsno[1] = [0.05, 0.20, 0.50]  # arctic: deep snow
    hsno[2] = [0.05, 0.05, 0.05]  # thin snow

    return dict(fi=fi, tsno=tsno, hsno=hsno)


def make_atm_inputs() -> dict:
    """Atmospheric forcing for all 5 scenarios."""
    ta = np.array([300.0, 250.0, 273.16, 315.0, 285.0])
    qa = np.array([0.015, 0.001, 0.004, 0.003, 0.006])
    ua = np.array([3.0, 5.0, 3.0, 2.0, 2.0])
    raina = np.array([5e-5, 0.0, 2e-5, 0.0, 1e-4])
    snowa = np.array([0.0, 1e-5, 1e-5, 0.0, 0.0])
    coszen = np.array([0.6, 0.2, 0.4, 0.8, 0.3])
    fira = np.array([380.0, 200.0, 300.0, 350.0, 320.0])
    psurf = np.full(NPOI, 101325.0)
    solad = np.zeros((NPOI, NBAND))
    solai = np.zeros((NPOI, NBAND))
    # VIS and NIR direct + diffuse
    solad[:, 0] = [200.0, 50.0, 100.0, 300.0, 80.0]   # VIS direct
    solad[:, 1] = [150.0, 30.0, 80.0, 250.0, 60.0]    # NIR direct
    solai[:, 0] = [60.0, 20.0, 40.0, 80.0, 30.0]      # VIS diffuse
    solai[:, 1] = [40.0, 10.0, 30.0, 60.0, 20.0]      # NIR diffuse

    return dict(
        ta=ta, qa=qa, ua=ua, raina=raina, snowa=snowa,
        coszen=coszen, fira=fira, psurf=psurf,
        solad=solad, solai=solai,
    )


def make_veg_inputs() -> dict:
    """Vegetation state for all 5 scenarios."""
    # Temperatures
    tu = np.array([298.0, 252.0, 273.16, 315.0, 284.0])
    ts = np.array([298.0, 252.0, 273.16, 315.0, 284.0])
    tl = np.array([298.0, 252.0, 273.16, 315.0, 284.0])
    tlsub = np.array([273.16, 252.0, 270.0, 273.16, 273.16])
    t12 = np.array([299.0, 251.0, 273.16, 315.0, 284.5])
    t34 = np.array([299.0, 251.0, 273.16, 315.0, 284.5])
    q12 = np.array([0.014, 0.001, 0.004, 0.003, 0.006])
    q34 = np.array([0.014, 0.001, 0.004, 0.003, 0.006])

    # Intercepted water
    wliqu = np.array([0.05, 0.0, 0.02, 0.0, 0.03])
    wliqs = np.array([0.03, 0.0, 0.01, 0.0, 0.02])
    wliql = np.array([0.04, 0.0, 0.02, 0.0, 0.02])
    wsnou = np.array([0.0, 0.5, 0.1, 0.0, 0.0])
    wsnos = np.array([0.0, 0.3, 0.05, 0.0, 0.0])
    wsnol = np.array([0.0, 0.2, 0.05, 0.0, 0.0])

    # Cover fractions
    fu = np.array([0.8, 0.1, 0.4, 0.0, 0.5])   # upper canopy
    fl = np.array([0.5, 0.3, 0.4, 0.05, 0.6])  # lower canopy

    # LAI/SAI [upper, lower]
    lai = np.zeros((NPOI, 2))
    lai[:, 0] = [4.0, 0.5, 2.0, 0.0, 3.0]  # upper
    lai[:, 1] = [2.0, 0.3, 1.5, 0.1, 2.0]  # lower
    sai = np.zeros((NPOI, 2))
    sai[:, 0] = [1.0, 0.2, 0.5, 0.0, 0.5]  # upper
    sai[:, 1] = [0.5, 0.1, 0.3, 0.0, 0.3]  # lower

    # Canopy heights
    ztop = np.zeros((NPOI, 2))
    ztop[:, 0] = [30.0, 5.0, 15.0, 0.0, 20.0]  # upper top
    ztop[:, 1] = [0.5, 0.2, 0.3, 0.1, 0.4]     # lower top
    zbot = np.zeros((NPOI, 2))
    zbot[:, 0] = [10.0, 1.0, 5.0, 0.0, 8.0]
    zbot[:, 1] = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Root distribution
    froot = np.array([
        [0.30, 0.25],
        [0.25, 0.25],
        [0.20, 0.20],
        [0.15, 0.15],
        [0.07, 0.10],
        [0.03, 0.05],
    ])

    # PFT fractions (12 PFTs - only a few nonzero)
    frac = np.zeros((NPOI, NPFT))
    # Tropical: broadleaf evergreen + C4 grass
    frac[0, 0] = 0.7; frac[0, 10] = 0.3
    # Arctic: C3 grass
    frac[1, 11] = 1.0
    # Temperate: broadleaf deciduous + C3 grass
    frac[2, 2] = 0.5; frac[2, 11] = 0.5
    # Desert: bare/C4 grass
    frac[3, 10] = 0.1
    # Wetland: broadleaf evergreen + shrub
    frac[4, 0] = 0.5; frac[4, 8] = 0.5

    return dict(
        tu=tu, ts=ts, tl=tl, tlsub=tlsub,
        t12=t12, t34=t34, q12=q12, q34=q34,
        wliqu=wliqu, wliqs=wliqs, wliql=wliql,
        wsnou=wsnou, wsnos=wsnos, wsnol=wsnol,
        fu=fu, fl=fl, lai=lai, sai=sai,
        ztop=ztop, zbot=zbot, froot=froot, frac=frac,
    )


def setup_f77_full(f77):
    """Set up all state in the F77 model using synthetic inputs."""
    soil = make_soil_inputs()
    snow = make_snow_inputs()
    atm = make_atm_inputs()
    veg = make_veg_inputs()

    f77.set_dtime(3600.0)
    f77.set_ndaypy(365)
    f77.set_soil(**soil, wpudmax=4.5, zwpmax=0.5)
    f77.set_snow(**snow)
    f77.set_atm(**atm)
    f77.set_veg(**{k: veg[k] for k in [
        'tu', 'ts', 'tl', 'tlsub', 't12', 't34', 'q12', 'q34',
        'wliqu', 'wliqs', 'wliql', 'wsnou', 'wsnos', 'wsnol',
        'fu', 'fl', 'lai', 'sai', 'ztop', 'zbot', 'froot', 'frac',
    ]})
    f77.set_veg_params()
    f77.call_init_extras()

    return soil, snow, atm, veg
