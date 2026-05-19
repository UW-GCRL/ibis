"""Test all 3 previously-NaN cells together for 2 months (Jan-Feb 1990).

Cells: L+15p0_+020p0 (Sahara), L+15p0_+080p0 (Arabia/India), L+15p0_+120p0 (SE Asia)
All used to NaN starting Feb-Mar 1990 in unpatched F77.
With finding #20 bare-cell bypass, expect 0 NaN throughout.
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


def main():
    # 3 NaN cells + 2 padding (replicas of first cell)
    tmpl = soil_template_for_lat(15.0)
    cells = [
        ("L+15p0_+020p0", 15.0,  20.0, tmpl, tmpl),  # Sahara
        ("L+15p0_+080p0", 15.0,  80.0, tmpl, tmpl),  # Arabia/India
        ("L+15p0_+120p0", 15.0, 120.0, tmpl, tmpl),  # SE Asia
        ("L+15p0_+020p0", 15.0,  20.0, tmpl, tmpl),  # padding (replica)
        ("L+15p0_+020p0", 15.0,  20.0, tmpl, tmpl),  # padding
    ]
    assert len(cells) == NPOI

    YEAR = 1990
    print(f"=== Testing 3 previously-NaN cells for 2 months (Jan-Feb {YEAR}) ===")
    for c in cells[:3]:
        print(f"    {c[0]}  lat={c[1]}  lon={c[2]}")

    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    lats = np.array([c[1] for c in cells])

    nan_seen = {c[0]: False for c in cells[:3]}
    first_nan_step = {c[0]: None for c in cells[:3]}

    step = 0
    for month in (1, 2):
        n_days_month = 31 if month == 1 else 28
        gswp3 = read_gswp3_month(YEAR, month, cells)
        for day_in_month in range(n_days_month):
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + day_in_month + 1
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
                # NaN check on each cell
                tg = np.asarray(f77.get_soil()["tg"])
                for i, c in enumerate(cells[:3]):
                    if np.isnan(tg[i]) and not nan_seen[c[0]]:
                        nan_seen[c[0]] = True
                        first_nan_step[c[0]] = step
                step += 1
            if day_in_month % 7 == 0 and hour == 23:
                print(f"  month {month} day {day_in_month}: step {step} ok", flush=True)

    print("\n=== Results ===")
    any_nan = False
    for c in cells[:3]:
        if nan_seen[c[0]]:
            print(f"  🔥 {c[0]}: NaN at step {first_nan_step[c[0]]}")
            any_nan = True
        else:
            print(f"  ✅ {c[0]}: no NaN in 2 months ({step} steps)")
    print()
    if any_nan:
        print("FAIL: some cell still NaN'd")
    else:
        print("PASS: all 3 previously-NaN cells now stable")


if __name__ == "__main__":
    main()
