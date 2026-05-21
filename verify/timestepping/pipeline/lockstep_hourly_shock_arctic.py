"""Hour-by-hour lockstep on L+65°N -160°E to find the remaining shock source.

Day 84 shock observed in earlier test. This drills hourly with detailed
field tracking through Jan-Mar to pin down the exact hour and field
where the shock first appears.
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


def main():
    target = "L+65p0_-160p0"
    lat, lon = 65.0, -160.0
    YEAR = 1990
    print(f"=== Hourly lockstep on {target} for Jan-Mar 1990 ===")

    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    lats = np.full(NPOI, lat)

    fields = [
        ("canopy.tu",      lambda f,j: (f.get_veg()["tu"], j.canopy.tu)),
        ("canopy.tl",      lambda f,j: (f.get_veg()["tl"], j.canopy.tl)),
        ("canopy.t12",     lambda f,j: (f.get_veg()["t12"], j.canopy.t12)),
        ("canopy.t34",     lambda f,j: (f.get_veg()["t34"], j.canopy.t34)),
        ("soil.tg",        lambda f,j: (f.get_soil()["tg"], j.soil.tg)),
        ("soil.tsoi[0]",   lambda f,j: (f.get_soil()["tsoi"][:, 0], j.soil.tsoi[:, 0])),
        ("soil.tsoi[1]",   lambda f,j: (f.get_soil()["tsoi"][:, 1], j.soil.tsoi[:, 1])),
        ("soil.wsoi[0]",   lambda f,j: (f.get_soil()["wsoi"][:, 0], j.soil.wsoi[:, 0])),
        ("soil.wisoi[0]",  lambda f,j: (f.get_soil()["wisoi"][:, 0], j.soil.wisoi[:, 0])),
        ("snow.tsno[0]",   lambda f,j: (f.get_snow()["tsno"][:, 0], j.snow.tsno[:, 0])),
        ("snow.fi",        lambda f,j: (f.get_snow()["fi"], j.snow.fi)),
        ("snow.hsno[0]",   lambda f,j: (f.get_snow()["hsno"][:, 0], j.snow.hsno[:, 0])),
        ("flux.fseng",     lambda f,j: (f.get_fluxes()["fseng"], j.canopy.fseng)),
    ]

    print(f"\n  {'day':>4s} {'hr':>3s}  {'ΔT_max':>11s}  {'biggest':>16s}  "
          f"{'tu':>9s}  {'soil.tg':>9s}  {'tsoi[0]':>9s}  {'fi':>7s}")
    print("-" * 110)

    step = 0
    prev_diffs = {f: 0.0 for f, _ in fields}
    for month in [1, 2, 3]:
        gswp3 = read_gswp3_month(YEAR, month, cells)
        if month == 1: ndays = 31
        elif month == 2: ndays = 28
        else: ndays = 31
        for d_in_m in range(ndays):
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + d_in_m + 1
            day_data = {k: gswp3[k][d_in_m, :] for k in gswp3}
            day_global = step // 24 + 1
            for h in range(24):
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

                hr_diffs = {}
                for fname, getter in fields:
                    fv, jv = getter(f77, state)
                    d = abs(float(np.asarray(fv).flat[0]) - float(np.asarray(jv).flat[0]))
                    hr_diffs[fname] = d
                step += 1
                if step % 24 == 0:
                    # End of day — log
                    hour_max = max(hr_diffs.values())
                    biggest = max(hr_diffs, key=hr_diffs.get)
                    tu_v = float(np.asarray(f77.get_veg()["tu"]).flat[0])
                    tg_v = float(np.asarray(f77.get_soil()["tg"]).flat[0])
                    tsoi_v = float(np.asarray(f77.get_soil()["tsoi"][:, 0]).flat[0])
                    fi_v = float(np.asarray(f77.get_snow()["fi"]).flat[0])
                    print(f"  {day_global:>4d} {h:>3d}  {hour_max:>11.3e}  {biggest:>16s}  "
                          f"{tu_v:>9.3f}  {tg_v:>9.3f}  {tsoi_v:>9.3f}  {fi_v:>7.3f}",
                          flush=True)
                # Big spike detection: log immediately
                hour_max = max(hr_diffs.values())
                if hour_max > 1e-6:
                    biggest = max(hr_diffs, key=hr_diffs.get)
                    if hr_diffs[biggest] > prev_diffs.get(biggest, 0) * 10 or hr_diffs[biggest] > 1e-5:
                        tu_v = float(np.asarray(f77.get_veg()["tu"]).flat[0])
                        tg_v = float(np.asarray(f77.get_soil()["tg"]).flat[0])
                        tsoi_v = float(np.asarray(f77.get_soil()["tsoi"][:, 0]).flat[0])
                        fi_v = float(np.asarray(f77.get_snow()["fi"]).flat[0])
                        snowt_v = float(np.asarray(f77.get_snow()["tsno"][:, 0]).flat[0])
                        print(f"  🔥 SPIKE  day={day_global} h={h:>3d}  "
                              f"{biggest}={hr_diffs[biggest]:.3e}  "
                              f"tu={tu_v:.4f} tg={tg_v:.4f} tsoi[0]={tsoi_v:.4f} "
                              f"snow.fi={fi_v:.4f} tsno[0]={snowt_v:.4f}", flush=True)
                prev_diffs = hr_diffs


if __name__ == "__main__":
    main()
