"""Hour-by-hour lockstep covering days 1-20 of 1990 to pin down the day-16
freeze-thaw shock event on L-85°S -160°E.

Logs every hour where |F77-JAX| > prev or above a tiny threshold, with
all major state vars. Outputs CSV for analysis.

Usage:
    PYTHONPATH=python-port verify/.venv/bin/python -m \
        verify.timestepping.pipeline.lockstep_hourly_shock
"""
from __future__ import annotations
import sys
import csv
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
    target = "L-85p0_-160p0"
    lat, lon = -85.0, -160.0
    n_days = 20
    YEAR = 1990

    print(f"=== Hour-by-hour lockstep on {target} for {n_days} days from {YEAR}-01-01 ===")

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
        ("canopy.tu",       lambda f,j: (f.get_veg()["tu"], j.canopy.tu)),
        ("canopy.tl",       lambda f,j: (f.get_veg()["tl"], j.canopy.tl)),
        ("canopy.t12",      lambda f,j: (f.get_veg()["t12"], j.canopy.t12)),
        ("canopy.t34",      lambda f,j: (f.get_veg()["t34"], j.canopy.t34)),
        ("soil.tg",         lambda f,j: (f.get_soil()["tg"], j.soil.tg)),
        ("soil.tsoi[0]",    lambda f,j: (f.get_soil()["tsoi"][:, 0], j.soil.tsoi[:, 0])),
        ("soil.tsoi[1]",    lambda f,j: (f.get_soil()["tsoi"][:, 1], j.soil.tsoi[:, 1])),
        ("soil.tsoi[2]",    lambda f,j: (f.get_soil()["tsoi"][:, 2], j.soil.tsoi[:, 2])),
        ("soil.wsoi[0]",    lambda f,j: (f.get_soil()["wsoi"][:, 0], j.soil.wsoi[:, 0])),
        ("soil.wisoi[0]",   lambda f,j: (f.get_soil()["wisoi"][:, 0], j.soil.wisoi[:, 0])),
        ("snow.tsno[0]",    lambda f,j: (f.get_snow()["tsno"][:, 0], j.snow.tsno[:, 0])),
        ("snow.fi",         lambda f,j: (f.get_snow()["fi"], j.snow.fi)),
        ("snow.hsno[0]",    lambda f,j: (f.get_snow()["hsno"][:, 0], j.snow.hsno[:, 0])),
    ]

    csv_path = Path("/tmp/hourly_shock.csv")
    csv_fh = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_fh)
    csv_w.writerow(["day", "hour", "field", "F77", "JAX", "abs_diff",
                    "tsoi0_F", "tsoi0_J", "tg_F", "tg_J",
                    "snowfi_F", "snowfi_J"])

    print(f"\n  {'day':>4s} {'hr':>3s}  {'ΔT_max':>11s}  {'biggest':>18s} "
          f"   {'tu':>10s} {'ts':>10s} {'tl':>10s} {'tg':>10s}  Δtu Δts Δtl Δtg")
    print("-" * 100)

    last_dmax = 0.0
    for day_global in range(n_days):
        month = 1
        day_in_month = day_global
        jday = (date(YEAR, 1, 1) - date(YEAR, 1, 1)).days + day_in_month + 1

        if day_in_month == 0:
            gswp3 = read_gswp3_month(YEAR, 1, cells)
        day_data = {k: gswp3[k][day_in_month, :] for k in gswp3}

        for hour in range(24):
            atm = diurnal_disaggregate(day_data, float(hour), jday, lats)
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

            # Collect all field diffs
            hour_max = 0.0
            hour_field = "canopy.tu"
            field_diffs = {}
            for fname, getter in fields_track:
                fv, jv = getter(f77, state)
                fv = float(np.asarray(fv).flat[0]); jv = float(np.asarray(jv).flat[0])
                d = abs(fv - jv)
                field_diffs[fname] = (fv, jv, d)
                if d > hour_max:
                    hour_max = d
                    hour_field = fname

            tsoi0_f, tsoi0_j, _ = field_diffs.get("soil.tsoi[0]", (0,0,0))
            tg_f, tg_j, _ = field_diffs.get("soil.tg", (0,0,0))
            snowfi_f, snowfi_j, _ = field_diffs.get("snow.fi", (0,0,0))

            tu_f, tu_j, dtu = field_diffs.get("canopy.tu", (0,0,0))
            ts_f, ts_j, dts = field_diffs.get("canopy.ts", (0,0,0))
            tl_f, tl_j, dtl = field_diffs.get("canopy.tl", (0,0,0))
            tg_f, tg_j, dtg = field_diffs.get("soil.tg", (0,0,0))

            # Log every hour of day 16 to capture shock detail
            if day_global+1 == 16 or hour_max > last_dmax * 1.1 or hour_max > 1e-10 or hour == 0:
                print(f"  {day_global+1:>4d} {hour:>3d}  {hour_max:>11.3e}  "
                      f"{hour_field:>18s}    "
                      f"{tu_f:>10.6f} {ts_f:>10.6f} {tl_f:>10.6f} {tg_f:>10.6f}  "
                      f"Δ {dtu:.1e} {dts:.1e} {dtl:.1e} {dtg:.1e}",
                      flush=True)
                csv_w.writerow([day_global+1, hour, hour_field,
                               f"{field_diffs[hour_field][0]:.16e}",
                               f"{field_diffs[hour_field][1]:.16e}",
                               f"{hour_max:.3e}",
                               f"{tsoi0_f:.16e}", f"{tsoi0_j:.16e}",
                               f"{tg_f:.16e}", f"{tg_j:.16e}",
                               f"{snowfi_f:.10e}", f"{snowfi_j:.10e}"])
                csv_fh.flush()
            last_dmax = max(last_dmax, hour_max)

    csv_fh.close()
    print(f"\nFull log → {csv_path}")


if __name__ == "__main__":
    main()
