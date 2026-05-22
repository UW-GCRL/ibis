"""Tight drill on the d115 h05-h12 burst window for L+55°N +100°E.

The previous broad drill identified the explosion at d115 h07:
  Δmax = 21.6 W/m² on flux.fsenl   (matches year-long worst)
  canopy.tu jumps by 0.56 K        (matches year-long worst)

Conditions: ta near 273.16 K (freezing point), snow.fi=0 but tiny atmospheric
snow input (~2e-6 kg/m²/s). Looks like a freeze-thaw or snow-precipitation
branch flip.

This drill seeds silently through d115 h04, then dumps the FULL post-step
diff for EVERY field h05-h12 so we can see exactly which field flips first.
"""
from __future__ import annotations
from datetime import date
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

from verify.timestepping.pipeline.global_gswp3_test import (
    diurnal_disaggregate, read_gswp3_month, make_batch_state,
    soil_template_for_lat,
)
from verify.timestepping.verify_lsxmain import (
    setup_f77_state, setup_jax_state, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body
from tests.f2py.ibis_f77_wrapper import NPOI


FIELDS = [
    ("canopy.tu",      lambda f,j: (float(f.get_veg()["tu"][0]),       float(j.canopy.tu[0]))),
    ("canopy.ts",      lambda f,j: (float(f.get_veg()["ts"][0]),       float(j.canopy.ts[0]))),
    ("canopy.tl",      lambda f,j: (float(f.get_veg()["tl"][0]),       float(j.canopy.tl[0]))),
    ("canopy.t12",     lambda f,j: (float(f.get_veg()["t12"][0]),      float(j.canopy.t12[0]))),
    ("canopy.t34",     lambda f,j: (float(f.get_veg()["t34"][0]),      float(j.canopy.t34[0]))),
    ("canopy.q12",     lambda f,j: (float(f.get_veg()["q12"][0]),      float(j.canopy.q12[0]))),
    ("canopy.q34",     lambda f,j: (float(f.get_veg()["q34"][0]),      float(j.canopy.q34[0]))),
    ("canopy.wliqu",   lambda f,j: (float(f.get_veg()["wliqu"][0]),    float(j.canopy.wliqu[0]))),
    ("canopy.wsnou",   lambda f,j: (float(f.get_veg()["wsnou"][0]),    float(j.canopy.wsnou[0]))),
    ("canopy.wliqs",   lambda f,j: (float(f.get_veg()["wliqs"][0]),    float(j.canopy.wliqs[0]))),
    ("canopy.wsnos",   lambda f,j: (float(f.get_veg()["wsnos"][0]),    float(j.canopy.wsnos[0]))),
    ("canopy.wliql",   lambda f,j: (float(f.get_veg()["wliql"][0]),    float(j.canopy.wliql[0]))),
    ("canopy.wsnol",   lambda f,j: (float(f.get_veg()["wsnol"][0]),    float(j.canopy.wsnol[0]))),
    ("canopy.fwetu",   lambda f,j: (float(f.get_veg()["fwetu"][0]),    float(j.canopy.fwetu[0]))),
    ("canopy.fwetl",   lambda f,j: (float(f.get_veg()["fwetl"][0]),    float(j.canopy.fwetl[0]))),
    ("soil.tg",        lambda f,j: (float(f.get_soil()["tg"][0]),      float(j.soil.tg[0]))),
    ("soil.tsoi[0]",   lambda f,j: (float(f.get_soil()["tsoi"][0,0]),  float(j.soil.tsoi[0,0]))),
    ("soil.wsoi[0]",   lambda f,j: (float(f.get_soil()["wsoi"][0,0]),  float(j.soil.wsoi[0,0]))),
    ("snow.fi",        lambda f,j: (float(f.get_snow()["fi"][0]),      float(j.snow.fi[0]))),
    ("snow.tsno[0]",   lambda f,j: (float(f.get_snow()["tsno"][0,0]),  float(j.snow.tsno[0,0]))),
    ("snow.tsno[1]",   lambda f,j: (float(f.get_snow()["tsno"][0,1]),  float(j.snow.tsno[0,1]))),
    ("snow.tsno[2]",   lambda f,j: (float(f.get_snow()["tsno"][0,2]),  float(j.snow.tsno[0,2]))),
    ("snow.hsno[0]",   lambda f,j: (float(f.get_snow()["hsno"][0,0]),  float(j.snow.hsno[0,0]))),
    ("snow.hsno[1]",   lambda f,j: (float(f.get_snow()["hsno"][0,1]),  float(j.snow.hsno[0,1]))),
    ("snow.hsno[2]",   lambda f,j: (float(f.get_snow()["hsno"][0,2]),  float(j.snow.hsno[0,2]))),
    ("flux.fseng",     lambda f,j: (float(f.get_fluxes()["fseng"][0]), float(j.canopy.fseng[0]))),
    ("flux.fsenu",     lambda f,j: (float(f.get_fluxes()["fsenu"][0]), float(j.canopy.fsenu[0]))),
    ("flux.fsenl",     lambda f,j: (float(f.get_fluxes()["fsenl"][0]), float(j.canopy.fsenl[0]))),
    ("flux.fseni",     lambda f,j: (float(f.get_fluxes()["fseni"][0]), float(j.canopy.fseni[0]))),
    ("flux.fvapg",     lambda f,j: (float(f.get_fluxes()["fvapg"][0]), float(j.canopy.fvapg[0]))),
]


def main():
    target = "L+55p0_+100p0"
    lat, lon = 55.0, 100.0
    YEAR = 1990
    BURST_START = 115
    BURST_END = 115
    BURST_HOUR_LO = 5
    BURST_HOUR_HI = 12
    print(f"=== Siberia d{BURST_START}-h{BURST_HOUR_LO}-{BURST_HOUR_HI} burst drill on {target} ===")

    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    lats = np.full(NPOI, lat)

    step = 0
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    print(f"  init complete, NPOI={NPOI}", flush=True)

    for month in [1, 2, 3, 4, 5]:
        gswp3 = read_gswp3_month(YEAR, month, cells)
        if month in (1, 3, 5):
            ndays = 31
        elif month == 2:
            ndays = 28
        else:
            ndays = 30
        for d_in_m in range(ndays):
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + d_in_m + 1
            day_data = {k: gswp3[k][d_in_m, :] for k in gswp3}
            day_global = step // 24 + 1
            for h in range(24):
                if h == 0 and day_global % 20 == 0 and day_global < BURST_START:
                    print(f"  seeding: day {day_global}", flush=True)

                atm = diurnal_disaggregate(day_data, float(h), jday, lats)

                in_burst = (BURST_START <= day_global <= BURST_END
                            and BURST_HOUR_LO <= h <= BURST_HOUR_HI)
                if in_burst:
                    pre_vals = {fname: getter(f77, state) for fname, getter in FIELDS}

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

                if in_burst:
                    post_vals = {fname: getter(f77, state) for fname, getter in FIELDS}
                    diffs = {fname: abs(post_vals[fname][0] - post_vals[fname][1])
                             for fname in post_vals}
                    pre_diffs = {fname: abs(pre_vals[fname][0] - pre_vals[fname][1])
                                 for fname in pre_vals}
                    delta = {f: diffs[f] - pre_diffs[f] for f in diffs}

                    ta_v   = float(atm["ta"][0])
                    raina_v = float(atm["raina"][0])
                    snowa_v = float(atm["snowa"][0])
                    print(f"\n=== d{day_global:3d} h{h:02d}  ta={ta_v:.4f}  rain={raina_v:.3e}  snow={snowa_v:.3e} ===")
                    # Sort fields by Δ-diff magnitude
                    rows = sorted(delta.items(), key=lambda x: -abs(x[1]))
                    print(f"  {'field':18s}  {'pre|Δ|':>11s}  {'post|Δ|':>11s}  {'Δ-jump':>11s}  {'F77 post':>14s}  {'JAX post':>14s}")
                    for f, dj in rows:
                        if abs(dj) < 1e-12 and diffs[f] < 1e-10:
                            continue
                        fp, jp = post_vals[f]
                        print(f"  {f:18s}  {pre_diffs[f]:>11.3e}  {diffs[f]:>11.3e}  {dj:+11.3e}  {fp:>14.6e}  {jp:>14.6e}")
                step += 1
                if day_global > BURST_END:
                    return


if __name__ == "__main__":
    main()
