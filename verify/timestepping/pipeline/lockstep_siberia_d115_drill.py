"""Day-115 snow-disappearance drill for L+55°N +100°E (Siberian boreal).

Tracks the same field set as lockstep_day84_drill.py, but for the Siberian
boreal cell whose post-finding-#24 worst-year drift (0.56 K canopy.tu) lands
on day 115 (Apr 25) — the exact day snow.fi crosses from ~0.1 to 0.0.

Approach: integrate days 0-114 silently to seed F77/JAX state, then drill
hourly from day 110-118 (window around the snow-disappearance event) with
full per-hour diff dump for all 27 fields. Flag the first hour any post-step
diff exceeds 1 mK so we can see WHICH branch fired differently.
"""
from __future__ import annotations
import sys
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
    ("soil.wisoi[0]",  lambda f,j: (float(f.get_soil()["wisoi"][0,0]), float(j.soil.wisoi[0,0]))),
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
    ("flux.fvapg",     lambda f,j: (float(f.get_fluxes()["fvapg"][0]), float(j.canopy.fvapg[0]))),
]


def main():
    target = "L+55p0_+100p0"
    lat, lon = 55.0, 100.0
    YEAR = 1990
    DRILL_START = 110   # day-of-year — start verbose dump
    DRILL_END   = 120   # last day to dump
    print(f"=== Siberia day-{DRILL_START}-{DRILL_END} drill on {target} ===")

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
    shock_logged = False
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    print(f"  init complete, NPOI={NPOI}", flush=True)

    for month in [1, 2, 3, 4, 5]:
        gswp3 = read_gswp3_month(YEAR, month, cells)
        if month in (1, 3, 5):
            ndays = 31
        elif month == 2:
            ndays = 28
        else:  # 4
            ndays = 30
        for d_in_m in range(ndays):
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + d_in_m + 1
            day_data = {k: gswp3[k][d_in_m, :] for k in gswp3}
            day_global = step // 24 + 1
            for h in range(24):
                if h == 0 and day_global % 20 == 0 and day_global < DRILL_START:
                    print(f"  seeding: day {day_global}", flush=True)

                atm = diurnal_disaggregate(day_data, float(h), jday, lats)

                verbose = (DRILL_START <= day_global <= DRILL_END)
                if verbose:
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

                if verbose:
                    post_vals = {fname: getter(f77, state) for fname, getter in FIELDS}
                    diffs = {fname: abs(post_vals[fname][0] - post_vals[fname][1])
                             for fname in post_vals}
                    max_diff = max(diffs.values())
                    worst_field = max(diffs, key=diffs.get)

                    pre_diffs = {fname: abs(pre_vals[fname][0] - pre_vals[fname][1])
                                 for fname in pre_vals}
                    delta_diffs = {f: diffs[f] - pre_diffs.get(f, 0.0)
                                   for f in diffs}
                    sorted_jumps = sorted(delta_diffs.items(),
                                          key=lambda x: -abs(x[1]))
                    top_jumps = [(f, d, diffs[f]) for f, d in sorted_jumps[:10]
                                 if abs(d) > 1e-12]

                    if not shock_logged and max_diff > 1e-4:
                        print(f"\nSHOCK FIRST DETECTED: day={day_global} hour={h}")
                        print(f"   post-step max diff: {max_diff:.3e}  ({worst_field})")
                        print(f"\n   Top jumps:")
                        for f, jump, total in top_jumps:
                            f77_pre, jax_pre = pre_vals[f]
                            f77_post, jax_post = post_vals[f]
                            print(f"      {f:18s}  Δdiff={jump:+.3e}  total={total:.3e}")
                            print(f"        pre:   F77={f77_pre:.10e}  JAX={jax_pre:.10e}")
                            print(f"        post:  F77={f77_post:.10e}  JAX={jax_post:.10e}")
                        shock_logged = True

                    tu = post_vals["canopy.tu"][0]
                    fi = post_vals["snow.fi"][0]
                    snowamt = post_vals["snow.hsno[0]"][0] + post_vals["snow.hsno[1]"][0] + post_vals["snow.hsno[2]"][0]
                    raina_v = float(atm["raina"][0])
                    snowa_v = float(atm["snowa"][0])
                    ta_v   = float(atm["ta"][0])
                    print(f"  d{day_global:3d} h{h:02d}  Δmax={max_diff:.3e}  {worst_field:14s}  "
                          f"tu={tu:.3f}  fi={fi:.4f}  hsno={snowamt:.4f}  ta={ta_v:.2f}  "
                          f"rain={raina_v:.2e}  snow={snowa_v:.2e}",
                          flush=True)
                step += 1
                if day_global > DRILL_END:
                    return


if __name__ == "__main__":
    main()
