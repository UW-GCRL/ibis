"""Find when and where NaN first appears in a problem cell's simulation.

Used to investigate the year-long GSWP3 test finding: 3 cells at lat=+15°N
arid (Sahara, Arabia, Southeast Asia) produced NaN in both F77 and JAX
starting March 1990 — a shared IBIS 2.6b4 instability.

Runs continuous Jan-Mar lockstep on the target cell with NPOI=5 replicas,
checks for NaN every hour, reports the FIRST NaN with full state context.

Usage:
    PYTHONPATH=python-port verify/.venv/bin/python -m \
        verify.timestepping.pipeline.lockstep_nan_hunt L+15p0_+020p0
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


def is_nan(x):
    try:
        return np.any(np.isnan(np.asarray(x)))
    except Exception:
        return False


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "L+15p0_+020p0"
    lat_str, lon_str = target.split("_", 1)
    lat = float(lat_str[1:].replace("p", "."))
    lon = float(lon_str.replace("p", "."))
    YEAR = 1990
    n_months = 4  # Jan-Apr to catch March onset

    print(f"=== NaN hunt on {target} (lat={lat}, lon={lon}) "
          f"for {n_months} months starting {YEAR}-01-01 ===")

    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    lats = np.full(NPOI, lat)

    f77_fields = [
        ("soil.tg",        lambda f: f.get_soil()["tg"]),
        ("soil.tsoi[0]",   lambda f: f.get_soil()["tsoi"][:, 0]),
        ("soil.wsoi[0]",   lambda f: f.get_soil()["wsoi"][:, 0]),
        ("soil.wisoi[0]",  lambda f: f.get_soil()["wisoi"][:, 0]),
        ("snow.fi",        lambda f: f.get_snow()["fi"]),
        ("snow.tsno[0]",   lambda f: f.get_snow()["tsno"][:, 0]),
        ("snow.hsno[0]",   lambda f: f.get_snow()["hsno"][:, 0]),
        ("canopy.tu",      lambda f: f.get_veg()["tu"]),
        ("canopy.ts",      lambda f: f.get_veg()["ts"]),
        ("canopy.tl",      lambda f: f.get_veg()["tl"]),
        ("canopy.t12",     lambda f: f.get_veg()["t12"]),
        ("canopy.t34",     lambda f: f.get_veg()["t34"]),
        ("canopy.q12",     lambda f: f.get_veg()["q12"]),
        ("canopy.q34",     lambda f: f.get_veg()["q34"]),
        ("canopy.wliqu",   lambda f: f.get_veg()["wliqu"]),
        ("canopy.wsnou",   lambda f: f.get_veg()["wsnou"]),
        ("stresstu",       lambda f: f.get_stomata()["stresstu"]),
        ("flux.fseng",     lambda f: f.get_fluxes()["fseng"]),
        ("flux.fvapg",     lambda f: f.get_fluxes()["fvapg"]),
        ("flux.fseni",     lambda f: f.get_fluxes()["fseni"]),
    ]

    first_nan_step = None
    first_nan_field = None
    prev_state = {}

    print(f"\n{'day':>4s} {'hr':>3s}  status")
    print("-" * 80)

    step = 0
    for month in range(1, n_months + 1):
        if month == 12:
            next_m = date(YEAR + 1, 1, 1)
        else:
            next_m = date(YEAR, month + 1, 1)
        n_days_month = (next_m - date(YEAR, month, 1)).days

        gswp3 = read_gswp3_month(YEAR, month, cells)

        for day_in_month in range(n_days_month):
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + day_in_month + 1
            day_data = {k: gswp3[k][day_in_month, :] for k in gswp3}

            for hour in range(24):
                atm = diurnal_disaggregate(day_data, float(hour), jday, lats)

                # Record state BEFORE this step
                prev_state = {name: float(np.asarray(get(f77)).flat[0])
                              for name, get in f77_fields}

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

                # Check every field for NaN
                for name, get in f77_fields:
                    if is_nan(get(f77)):
                        first_nan_step = step
                        first_nan_field = name
                        print(f"\n🔥 FIRST NaN detected at step {step} "
                              f"(month={month} day={day_in_month+1} hour={hour}, "
                              f"jday={jday})")
                        print(f"   First field to NaN: {name}")
                        print(f"\nState BEFORE this step (last valid):")
                        for n in sorted(prev_state.keys()):
                            v = prev_state[n]
                            marker = " 🔥" if np.isnan(v) else ""
                            print(f"  {n:>16s}: {v:>+15.6e}{marker}")
                        print(f"\nAtm forcing at NaN step:")
                        for k in ("ta","qa","ua","psurf","coszen","fira","raina","snowa"):
                            v = float(np.asarray(atm[k]).flat[0])
                            print(f"  {k:>8s}: {v:>+15.6e}")
                        print(f"\nState AFTER this step (NaN here):")
                        for n, get in f77_fields:
                            v = float(np.asarray(get(f77)).flat[0])
                            marker = " 🔥 NaN" if np.isnan(v) else ""
                            print(f"  {n:>16s}: {v:>+15.6e}{marker}")
                        return

                step += 1

            if day_in_month % 7 == 0 and hour == 23:
                print(f" {day_in_month:>3d} {hour:>3d}  ok (step {step})", flush=True)

    print(f"\nNo NaN encountered in {n_months} months")


if __name__ == "__main__":
    main()
