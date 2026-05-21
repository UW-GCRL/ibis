"""Day-84 spring-thaw shock drill for L+65°N -160°E.

Tracks 30+ fields hourly to identify exact trigger of the mK-level shock
that appears suddenly on day 84 (March 25, Arctic spring thaw).

Approach: run F77 and JAX in lockstep, snapshot every hour from day 80-90,
compute diff in canopy water (wliqu, wsnou, wliqs, wsnos, wliql, wsnol),
temperatures (tu, ts, tl, t12, t34), and flag the FIRST hour where any
field's diff exceeds a tight threshold. Print the diff snapshot at that
moment so we can see WHICH branch fired differently.
"""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path
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


# Detailed field accessors: each returns (f77_val, jax_val) scalars for cell 0.
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
    ("canopy.rliqu",   lambda f,j: (float(f.get_veg()["rliqu"][0]),    float(j.canopy.rliqu[0]))),
    ("soil.tg",        lambda f,j: (float(f.get_soil()["tg"][0]),      float(j.soil.tg[0]))),
    ("soil.tsoi[0]",   lambda f,j: (float(f.get_soil()["tsoi"][0,0]),  float(j.soil.tsoi[0,0]))),
    ("soil.wsoi[0]",   lambda f,j: (float(f.get_soil()["wsoi"][0,0]),  float(j.soil.wsoi[0,0]))),
    ("soil.wisoi[0]",  lambda f,j: (float(f.get_soil()["wisoi"][0,0]), float(j.soil.wisoi[0,0]))),
    ("snow.fi",        lambda f,j: (float(f.get_snow()["fi"][0]),      float(j.snow.fi[0]))),
    ("snow.tsno[0]",   lambda f,j: (float(f.get_snow()["tsno"][0,0]),  float(j.snow.tsno[0,0]))),
    ("snow.hsno[0]",   lambda f,j: (float(f.get_snow()["hsno"][0,0]),  float(j.snow.hsno[0,0]))),
    ("flux.fseng",     lambda f,j: (float(f.get_fluxes()["fseng"][0]), float(j.canopy.fseng[0]))),
    ("flux.fsenu",     lambda f,j: (float(f.get_fluxes()["fsenu"][0]), float(j.canopy.fsenu[0]))),
    ("flux.fsenl",     lambda f,j: (float(f.get_fluxes()["fsenl"][0]), float(j.canopy.fsenl[0]))),
    ("flux.fvapg",     lambda f,j: (float(f.get_fluxes()["fvapg"][0]), float(j.canopy.fvapg[0]))),
]


def main():
    target = "L+65p0_-160p0"
    lat, lon = 65.0, -160.0
    YEAR = 1990
    print(f"=== Day-84 drill on {target} ===")

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
    prev_state = None
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    print(f"  init complete, NPOI={NPOI}", flush=True)

    for month in [1, 2, 3, 4]:
        gswp3 = read_gswp3_month(YEAR, month, cells)
        if month in (1, 3): ndays = 31
        elif month == 2: ndays = 28
        else: ndays = 30
        for d_in_m in range(ndays):
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + d_in_m + 1
            day_data = {k: gswp3[k][d_in_m, :] for k in gswp3}
            day_global = step // 24 + 1
            for h in range(24):
                # Progress heartbeat
                if h == 0 and day_global % 10 == 0:
                    print(f"  progress: day {day_global}", flush=True)
                # Skip days before 80 to save time
                if day_global < 80:
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
                    step += 1
                    continue

                atm = diurnal_disaggregate(day_data, float(h), jday, lats)
                # Capture pre-step F77/JAX state
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

                # Post-step diffs
                post_vals = {fname: getter(f77, state) for fname, getter in FIELDS}
                diffs = {fname: abs(post_vals[fname][0] - post_vals[fname][1])
                         for fname in post_vals}
                max_diff = max(diffs.values())
                worst_field = max(diffs, key=diffs.get)

                # Pre-step diffs (state at start of this step)
                pre_diffs = {fname: abs(pre_vals[fname][0] - pre_vals[fname][1])
                             for fname in pre_vals}
                pre_max = max(pre_diffs.values())

                # Find fields that JUMPED this step (delta in diff > 1e-9)
                delta_diffs = {f: diffs[f] - pre_diffs.get(f, 0.0)
                               for f in diffs}
                # Sort by absolute jump
                sorted_jumps = sorted(delta_diffs.items(),
                                       key=lambda x: -abs(x[1]))
                top_jumps = [(f, d, diffs[f]) for f, d in sorted_jumps[:8]
                             if abs(d) > 1e-11]

                if not shock_logged and max_diff > 1e-6:
                    print(f"\n🔥 SHOCK FIRST DETECTED: day={day_global} hour={h}")
                    print(f"   pre-step max diff: {pre_max:.3e}")
                    print(f"   post-step max diff: {max_diff:.3e}  ({worst_field})")
                    print(f"\n   Top fields with biggest JUMP this step (Δ in diff):")
                    for f, jump, total in top_jumps:
                        f77_pre, jax_pre = pre_vals[f]
                        f77_post, jax_post = post_vals[f]
                        print(f"      {f:18s}  Δdiff={jump:+.3e}  total={total:.3e}")
                        print(f"        pre:   F77={f77_pre!r:>22s}  JAX={jax_pre!r:>22s}")
                        print(f"        post:  F77={f77_post!r:>22s}  JAX={jax_post!r:>22s}")
                    shock_logged = True

                # Hourly summary
                if h == 12 and day_global <= 90:
                    tu = post_vals["canopy.tu"][0]
                    wliqu = post_vals["canopy.wliqu"][0]
                    wsnou = post_vals["canopy.wsnou"][0]
                    print(f"  day {day_global:3d} h12  max_d={max_diff:.3e}  worst={worst_field:18s}  "
                          f"tu={tu:.4f}  wliqu={wliqu:.2e}  wsnou={wsnou:.2e}",
                          flush=True)
                step += 1
                if day_global > 90:
                    return


if __name__ == "__main__":
    main()
