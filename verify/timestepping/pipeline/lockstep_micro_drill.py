"""Hour-by-hour drill from day 80 through 90 — tracks ALL fields, no skip,
print everything to identify when temp/flux fields first diverge.

Excludes flux.fvapg from "worst field" (apples-to-oranges between F77's
areal-averaged and JAX's per-area). Focuses on tu/ts/tl/tg/tsoi/fseng/tsno
which are stored consistently.
"""
from __future__ import annotations
import sys
from datetime import date
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import ctypes
from verify.timestepping.pipeline.global_gswp3_test import (
    diurnal_disaggregate, read_gswp3_month, make_batch_state,
    soil_template_for_lat,
)

# Direct ctypes access to F77 common block /com1d3/ to read variables
# that the F77 wrapper doesn't expose (rainl, snowl, trainl, tsnowl).
# Layout in com1d3 (gfortran -fdefault-real-8: each var is npoi=5 real*8):
# 0:firb, 1:firg, 2:firi, 3:firl, 4:firs, 5:firu, 6:fsena, 7:fseng,
# 8:fseni, 9:fsenl, 10:fsens, 11:fsenu, 12:fvapa, 13:fvapg, 14:fvapi,
# 15:fvaplt, 16:fvaplw, 17:fvaps, 18:fvaput, 19:fvapuw,
# 20:soli, 21:solg, 22:soll, 23:sols, 24:solu,
# 25:raing, 26:rainl, 27:rainu, 28:snowg, 29:snowl, 30:snowu,
# 31:traing, 32:trainl, 33:trainu, 34:tsnowg, 35:tsnowl, 36:tsnowu
_F77_SO = "python-port/tests/f2py/ibis_f77.so"
_NPOI_C = 5
_lib_handle = ctypes.CDLL(_F77_SO)
_com1d3_addr = ctypes.addressof(ctypes.c_double.in_dll(_lib_handle, "com1d3_"))
_com1d3_data = (ctypes.c_double * (37 * _NPOI_C)).from_address(_com1d3_addr)

# wflo_dump common block: wflo_save(npoi, nsoilay+1) + fhtop_save(npoi)
# Layout: nsoilay+1 = 7 layers, each npoi=5. Total wflo_save = 35 doubles.
# Then fhtop_save = 5 doubles. Total = 40 doubles.
_wflo_addr = ctypes.addressof(ctypes.c_double.in_dll(_lib_handle, "wflo_dump_"))
_wflo_data = (ctypes.c_double * (7 * _NPOI_C + _NPOI_C)).from_address(_wflo_addr)

def _f77_wflo(layer):
    # wflo_save(i, k) in Fortran is column-major: index = (k-1)*npoi + (i-1)
    # We want cell 0 (i=1 in Fortran), layer k (1-indexed).
    return _wflo_data[layer * _NPOI_C]  # k 0-indexed, i=0

def _f77_fhtop():
    return _wflo_data[7 * _NPOI_C]  # First fhtop_save entry, i=0

def _f77_rainl(): return _com1d3_data[26 * _NPOI_C : 27 * _NPOI_C][0]
def _f77_snowl(): return _com1d3_data[29 * _NPOI_C : 30 * _NPOI_C][0]
def _f77_trainl(): return _com1d3_data[32 * _NPOI_C : 33 * _NPOI_C][0]
def _f77_tsnowl(): return _com1d3_data[35 * _NPOI_C : 36 * _NPOI_C][0]
from verify.timestepping.verify_lsxmain import (
    setup_f77_state, setup_jax_state, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body
from tests.f2py.ibis_f77_wrapper import NPOI


FIELDS = [
    # Canopy state
    ("canopy.tu",      lambda f,j: (float(f.get_veg()["tu"][0]),       float(j.canopy.tu[0]))),
    ("canopy.ts",      lambda f,j: (float(f.get_veg()["ts"][0]),       float(j.canopy.ts[0]))),
    ("canopy.tl",      lambda f,j: (float(f.get_veg()["tl"][0]),       float(j.canopy.tl[0]))),
    ("canopy.t12",     lambda f,j: (float(f.get_veg()["t12"][0]),      float(j.canopy.t12[0]))),
    ("canopy.t34",     lambda f,j: (float(f.get_veg()["t34"][0]),      float(j.canopy.t34[0]))),
    ("canopy.q12",     lambda f,j: (float(f.get_veg()["q12"][0]),      float(j.canopy.q12[0]))),
    ("canopy.q34",     lambda f,j: (float(f.get_veg()["q34"][0]),      float(j.canopy.q34[0]))),
    ("canopy.tlsub",   lambda f,j: (float(f.get_veg()["tlsub"][0]),    float(j.snow.tlsub[0]))),
    ("canopy.wliqu",   lambda f,j: (float(f.get_veg()["wliqu"][0]),    float(j.canopy.wliqu[0]))),
    ("canopy.wsnou",   lambda f,j: (float(f.get_veg()["wsnou"][0]),    float(j.canopy.wsnou[0]))),
    ("canopy.wliql",   lambda f,j: (float(f.get_veg()["wliql"][0]),    float(j.canopy.wliql[0]))),
    ("canopy.wsnol",   lambda f,j: (float(f.get_veg()["wsnol"][0]),    float(j.canopy.wsnol[0]))),
    # Soil state
    ("soil.tg",        lambda f,j: (float(f.get_soil()["tg"][0]),      float(j.soil.tg[0]))),
    ("soil.tsoi[0]",   lambda f,j: (float(f.get_soil()["tsoi"][0,0]),  float(j.soil.tsoi[0,0]))),
    ("soil.tsoi[1]",   lambda f,j: (float(f.get_soil()["tsoi"][0,1]),  float(j.soil.tsoi[0,1]))),
    ("soil.wsoi[0]",   lambda f,j: (float(f.get_soil()["wsoi"][0,0]),  float(j.soil.wsoi[0,0]))),
    ("soil.wisoi[0]",  lambda f,j: (float(f.get_soil()["wisoi"][0,0]), float(j.soil.wisoi[0,0]))),
    ("soil.heatg",     lambda f,j: (float(f.get_soil()["heatg"][0]),   float(j.soil.heatg[0]))),
    ("soil.hvasug",    lambda f,j: (float(f.get_soil()["hvasug"][0]),  float(j.soil.hvasug[0]))),
    # Snow state
    ("snow.fi",        lambda f,j: (float(f.get_snow()["fi"][0]),      float(j.snow.fi[0]))),
    ("snow.tsno[0]",   lambda f,j: (float(f.get_snow()["tsno"][0,0]),  float(j.snow.tsno[0,0]))),
    ("snow.tsno[1]",   lambda f,j: (float(f.get_snow()["tsno"][0,1]),  float(j.snow.tsno[0,1]))),
    ("snow.hsno[0]",   lambda f,j: (float(f.get_snow()["hsno"][0,0]),  float(j.snow.hsno[0,0]))),
    # Radiation (input to heat balance)
    ("rad.solg",       lambda f,j: (float(f.get_radiation()["solg"][0]),  float(j.canopy.solg[0]))),
    ("rad.firg",       lambda f,j: (float(f.get_radiation()["firg"][0]),  float(j.canopy.firg[0]))),
    ("rad.firi",       lambda f,j: (float(f.get_radiation()["firi"][0]),  float(j.canopy.firi[0]))),
    ("rad.soli",       lambda f,j: (float(f.get_radiation()["soli"][0]),  float(j.canopy.soli[0]))),
    # Fluxes from turvap (post-areal-averaging in F77)
    ("flux.fseng",     lambda f,j: (float(f.get_fluxes()["fseng"][0]), float(j.canopy.fseng[0]))),
    ("flux.fsenu",     lambda f,j: (float(f.get_fluxes()["fsenu"][0]), float(j.canopy.fsenu[0]))),
    ("flux.fsenl",     lambda f,j: (float(f.get_fluxes()["fsenl"][0]), float(j.canopy.fsenl[0]))),
    ("flux.fseni",     lambda f,j: (float(f.get_fluxes()["fseni"][0]), float(j.canopy.fseni[0]))),
    # NOTE: fvapg / fvapi / solg accessor mismatch: F77 stores AREAL-AVG (after snow.f),
    # JAX canopy state is PER-AREA. Compare apples-to-apples by scaling JAX by (1-fi)
    # or fi as appropriate.
    ("flux.fvapg_aa",  lambda f,j: (float(f.get_fluxes()["fvapg"][0]),
                                     float((1.0 - j.snow.fi[0]) * j.canopy.fvapg[0]))),
    ("flux.fvapi_aa",  lambda f,j: (float(f.get_fluxes()["fvapi"][0]),
                                     float(j.snow.fi[0] * j.canopy.fvapi[0]))),
    ("rad.solg_aa",    lambda f,j: (float(f.get_radiation()["solg"][0]),
                                     float((1.0 - j.snow.fi[0]) * j.canopy.solg[0]))),
    # Hydrology — surface fluxes. F77's raing and traing in get_fluxes are AREAL-AVG
    # (after snow.f line 287-290 mix with snowmelt). JAX's raing_avg is computed
    # locally and stored as part of hydrology updates; here we use grunof which IS
    # stored consistently.
    ("hydro.grunof",   lambda f,j: (float(f.get_hydro()["grunof"][0]),
                                     float(j.hydrology.grunof[0]))),
    # NEW: directly compare F77 post-snow.f raing/traing vs JAX's raing_avg/traing_avg.
    # F77 raing is what flows into soilctl's fhtop computation.
    ("raing_avg",      lambda f,j: (float(f.get_fluxes()["raing"][0]),
                                     float(j.hydrology.raing_avg[0]))),
    ("traing_avg",     lambda f,j: (float(f.get_fluxes()["traing"][0]),
                                     float(j.hydrology.traing_avg[0]))),
    # fhtop_soil: F77 computes locally in soilctl, not directly exposed.
    # Compare JAX's value to expected F77 value by reconstructing the formula:
    #   F77_fhtop = heatg + raing*ch2o*(traing-tsoi[0]) - grunof*ch2o*max(tmelt-tsoi[0],0)
    ("fhtop_soil",     lambda f,j: (
        float(f.get_soil()["heatg"][0]
              + f.get_fluxes()["raing"][0] * 4218.0 * (f.get_fluxes()["traing"][0] - f.get_soil()["tsoi"][0,0])
              - f.get_hydro()["grunof"][0] * 4218.0 * max(273.16 - f.get_soil()["tsoi"][0,0], 0.0)),
        float(j.hydrology.fhtop_soil[0]))),
    # Puddle and soil thermal conductivity (inputs to soilheat)
    ("soil.wpud",      lambda f,j: (float(f.get_soil()["wpud"][0]),    float(j.soil.wpud[0]))),
    ("soil.wipud",     lambda f,j: (float(f.get_soil()["wipud"][0]),   float(j.soil.wipud[0]))),
    ("soil.consoi[0]", lambda f,j: (float(f.get_soil()["consoi"][0,0]), float(j.soil.consoi[0,0]))),
    ("soil.consoi[1]", lambda f,j: (float(f.get_soil()["consoi"][0,1]), float(j.soil.consoi[0,1]))),
    # hflo: heat flux at layer interfaces. F77 stores in soilheat output;
    # JAX stores in soil.hflo. Compare interface 0 (= fhtop_soil) and 1.
    ("soil.hflo[0]",   lambda f,j: (0.0, float(j.soil.hflo[0,0]))),  # F77 doesn't expose, will compute later
    # wsoi at deeper layers
    ("soil.wsoi[1]",   lambda f,j: (float(f.get_soil()["wsoi"][0,1]),  float(j.soil.wsoi[0,1]))),
    ("soil.wsoi[2]",   lambda f,j: (float(f.get_soil()["wsoi"][0,2]),  float(j.soil.wsoi[0,2]))),
    ("soil.wisoi[1]",  lambda f,j: (float(f.get_soil()["wisoi"][0,1]), float(j.soil.wisoi[0,1]))),
    # qglif components (evap partitioning) — affects wisoi and wsoi updates
    ("soil.qglif[0]",  lambda f,j: (float(f.get_soil()["qglif"][0,0]), float(j.soil.qglif[0,0]))),
    ("soil.qglif[1]",  lambda f,j: (float(f.get_soil()["qglif"][0,1]), float(j.soil.qglif[0,1]))),
    ("soil.qglif[3]",  lambda f,j: (float(f.get_soil()["qglif"][0,3]), float(j.soil.qglif[0,3]))),
    # tsoi at deeper layers
    ("soil.tsoi[2]",   lambda f,j: (float(f.get_soil()["tsoi"][0,2]),  float(j.soil.tsoi[0,2]))),
    ("soil.tsoi[3]",   lambda f,j: (float(f.get_soil()["tsoi"][0,3]),  float(j.soil.tsoi[0,3]))),
]


def main():
    lat, lon = 65.0, -160.0
    YEAR = 1990
    print(f"=== Micro-drill day 83-86 on lat={lat} lon={lon} ===", flush=True)

    template = soil_template_for_lat(lat)
    cells = [("L+65", lat, lon, template, template)] * NPOI
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    lats = np.full(NPOI, lat)
    import sys as _s; _s.stdout.reconfigure(line_buffering=True)

    step = 0
    PRINT_START = 84
    PRINT_END = 85

    for month in [1, 2, 3, 4]:
        gswp3 = read_gswp3_month(YEAR, month, cells)
        ndays = 31 if month in (1,3) else (28 if month==2 else 30)
        for d_in_m in range(ndays):
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + d_in_m + 1
            day_data = {k: gswp3[k][d_in_m, :] for k in gswp3}
            day_global = step // 24 + 1
            for h in range(24):
                if day_global % 20 == 0 and h == 0:
                    print(f"  progress: day {day_global}", flush=True)
                atm = diurnal_disaggregate(day_data, float(h), jday, lats)
                f77.set_atm(**atm)
                state_atm = eqx.tree_at(
                    lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                               s.atm.coszen, s.atm.solad, s.atm.solai,
                               s.atm.fira, s.atm.raina, s.atm.snowa),
                    state,
                    tuple(jnp.asarray(atm[k]) for k in
                          ("ta","qa","ua","psurf","coszen","solad","solai",
                           "fira","raina","snowa")),
                )
                f77.call_lsxmain()
                state = timestep_body(state_atm, consts, params, DTIME)

                if PRINT_START <= day_global <= PRINT_END:
                    diffs = []
                    for fname, getter in FIELDS:
                        f77v, jv = getter(f77, state)
                        d = abs(f77v - jv)
                        diffs.append((fname, d, f77v, jv))
                    # Sort and find top 8 nonzero
                    diffs.sort(key=lambda x: -x[1])
                    top = diffs[:8]
                    parts = []
                    for fname, d, f77v, jv in top:
                        if d > 1e-12:
                            parts.append(f"{fname}={d:.2e}")
                    parts_str = "  ".join(parts) if parts else "all<1e-12"
                    tu = next(x for x in diffs if x[0]=="canopy.tu")
                    tg = next(x for x in diffs if x[0]=="soil.tg")
                    wsnou = next(x for x in diffs if x[0]=="canopy.wsnou")
                    wliqu = next(x for x in diffs if x[0]=="canopy.wliqu")
                    heatg = next(x for x in diffs if x[0]=="soil.heatg")
                    # Get F77 raing (post-snow-areal-avg) for diagnosis
                    raing_f77 = float(f77.get_fluxes()["raing"][0])
                    # JAX intermediate state for shock investigation
                    j_rainl = float(state.hydrology.rainl[0])
                    j_snowl = float(state.hydrology.snowl[0])
                    j_sflo_bot = float(state.hydrology.sflo_bot[0])
                    j_fhtop_snow = float(state.hydrology.fhtop_snow[0])
                    j_raing_avg = float(state.hydrology.raing_avg[0])
                    # Direct F77 access via ctypes (rainl/snowl not exposed by wrapper)
                    f_rainl = _f77_rainl()
                    f_snowl = _f77_snowl()
                    f_trainl = _f77_trainl()
                    f_tsnowl = _f77_tsnowl()
                    j_trainl = float(state.hydrology.trainl[0])
                    j_tsnowl = float(state.hydrology.tsnowl[0])
                    # Read F77 wflo[0,1,2] (top 3 layer fluxes) and fhtop_soil via ctypes
                    f_wflo0 = _f77_wflo(0)  # top boundary flux
                    f_wflo1 = _f77_wflo(1)
                    f_wflo2 = _f77_wflo(2)
                    f_fhtop = _f77_fhtop()
                    # JAX hflo (heat flux) is already in state. JAX wflo NOT stored;
                    # but JAX raing_avg / fhtop_soil are.
                    j_fhtop_soil = float(state.hydrology.fhtop_soil[0])
                    j_raing_avg = float(state.hydrology.raing_avg[0])
                    print(f"  d{day_global:3d} h{h:2d}  "
                          f"tu={tu[2]:.5f}  wliqu={wliqu[2]:.2e}  "
                          f"f.wflo[0,1,2]=({f_wflo0:.3e},{f_wflo1:.3e},{f_wflo2:.3e})  "
                          f"fhtop(f77={f_fhtop:.3f}/jax={j_fhtop_soil:.3f})  "
                          f"jraing_avg={j_raing_avg:.2e}  "
                          f"|  {parts_str}", flush=True)
                step += 1
                if day_global > PRINT_END:
                    return


if __name__ == "__main__":
    main()
