"""Continuous 365-day lockstep on a single target cell.

Differs from lockstep_inject_cell.py which only does one month.
Reads all 12 months of GSWP3 sequentially with state carried forward,
and prints daily max ΔT and the field/source. Goal: find when drift
accelerates over the year for the worst-drift Antarctic cell.

Usage:
    PYTHONPATH=python-port verify/.venv/bin/python -m \
        verify.timestepping.pipeline.lockstep_year_cell L-85p0_-160p0
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


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "L-85p0_-160p0"
    n_days_total = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    YEAR = 1990

    lat_str, lon_str = target.split("_", 1)
    lat = float(lat_str[1:].replace("p", "."))
    lon = float(lon_str.replace("p", "."))
    print(f"=== Year-long lockstep on {target} (lat={lat}, lon={lon}) "
          f"for {n_days_total} days starting {YEAR}-01-01 ===")

    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI

    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    lats = np.full(NPOI, lat)

    fields_track = [
        ("canopy.tu",      lambda f,j: (f.get_veg()["tu"], j.canopy.tu)),
        ("canopy.tl",      lambda f,j: (f.get_veg()["tl"], j.canopy.tl)),
        ("canopy.t12",     lambda f,j: (f.get_veg()["t12"], j.canopy.t12)),
        ("canopy.t34",     lambda f,j: (f.get_veg()["t34"], j.canopy.t34)),
        ("soil.tg",        lambda f,j: (f.get_soil()["tg"], j.soil.tg)),
        ("soil.tsoi[0]",   lambda f,j: (f.get_soil()["tsoi"][:, 0], j.soil.tsoi[:, 0])),
        ("soil.wsoi[0]",   lambda f,j: (f.get_soil()["wsoi"][:, 0], j.soil.wsoi[:, 0])),
        ("snow.tsno[0]",   lambda f,j: (f.get_snow()["tsno"][:, 0], j.snow.tsno[:, 0])),
        ("snow.fi",        lambda f,j: (f.get_snow()["fi"], j.snow.fi)),
        ("flux.fseni",     lambda f,j: (f.get_fluxes()["fseni"], j.canopy.fseni)),
        ("flux.fseng",     lambda f,j: (f.get_fluxes()["fseng"], j.canopy.fseng)),
    ]

    print(f"\n{'day':>4s} {'date':>10s}  {'ΔT_max':>11s}  {'biggest field':>16s} "
          f"{'tu':>9s} {'snow.fi':>9s} {'coszen':>8s}")
    print("-" * 90)

    day_global = 0
    cumulative_worst = 0.0
    for month in range(1, 13):
        if month == 12:
            next_m = date(YEAR + 1, 1, 1)
        else:
            next_m = date(YEAR, month + 1, 1)
        n_days_month = (next_m - date(YEAR, month, 1)).days
        if day_global >= n_days_total:
            break

        gswp3 = read_gswp3_month(YEAR, month, cells)

        for day_in_month in range(n_days_month):
            if day_global >= n_days_total:
                break
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + day_in_month + 1
            day_data = {k: gswp3[k][day_in_month, :] for k in gswp3}

            day_max = 0.0
            day_field = ""
            day_coszen_max = 0.0
            for hour in range(24):
                atm = diurnal_disaggregate(day_data, float(hour), jday, lats)
                day_coszen_max = max(day_coszen_max, float(atm["coszen"][0]))
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
                for fname, getter in fields_track:
                    fv, jv = getter(f77, state)
                    if fname.startswith("flux."):
                        continue
                    d = abs(float(np.asarray(fv).flat[0])
                            - float(np.asarray(jv).flat[0]))
                    if d > day_max:
                        day_max = d
                        day_field = fname

            day_global += 1
            if day_max > cumulative_worst:
                cumulative_worst = day_max
            tu_jax = float(state.canopy.tu[0])
            fi_jax = float(state.snow.fi[0])
            print(f"{day_global:>4d} {YEAR}-{month:02d}-{day_in_month+1:02d}  "
                  f"{day_max:>11.3e}  {day_field:>16s} "
                  f"{tu_jax:>9.2f} {fi_jax:>9.4f} {day_coszen_max:>8.4f}",
                  flush=True)

    print(f"\nMax ΔT over {n_days_total} days: {cumulative_worst:.3e}")


if __name__ == "__main__":
    main()
