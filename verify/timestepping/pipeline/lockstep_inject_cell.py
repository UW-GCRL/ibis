"""Lockstep + state-injection diagnostic for a specific problem cell.

Reproduces the bug-finding pipeline that localized finding #16, but for a
single cell from the global GSWP3 run (the one with worst temperature
drift). Targets `L-35p0_+140p0` (southern Australia, ~35°S, 140°E) by
default.

Strategy:
  1. Run F77 + JAX in lockstep for ~5 days under GSWP3 forcing for this
     cell (padded to 5 cells for F77 NPOI=5).
  2. At each step, capture both states and compare. Track the EARLIEST
     step where any field's diff exceeds a small threshold (~1e-9).
  3. At step N (just before divergence), inject F77 state into JAX
     (forced alignment), then run one step.
  4. Compare the one-step output. If diff > input diff, we've localized
     to step-N physics.

Usage:
    PYTHONPATH=python-port verify/.venv/bin/python -m \
        verify.timestepping.pipeline.lockstep_inject_cell L-35p0_+140p0
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

# Reuse the global test infrastructure
from verify.timestepping.pipeline.global_gswp3_test import (
    diurnal_disaggregate, read_gswp3_month, make_batch_state,
    soil_template_for_lat,
)
from verify.timestepping.verify_lsxmain import (
    setup_f77_state, setup_jax_state, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body
from tests.f2py.ibis_f77_wrapper import NPOI, NSOILAY


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "L-35p0_+140p0"
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    # Parse lat/lon from label
    lat_str, lon_str = target.split("_", 1)
    lat = float(lat_str[1:].replace("p", "."))
    lon = float(lon_str.replace("p", "."))
    print(f"=== Lockstep diagnostic on {target} (lat={lat}, lon={lon}) ===")
    print(f"    Running {n_days} days = {n_days*24} timesteps")

    # Build batch: target cell repeated 5x to satisfy NPOI=5
    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI

    YEAR, MONTH = 1990, 6
    gswp3 = read_gswp3_month(YEAR, MONTH, cells)
    # gswp3 has shape (ndays, ncells)

    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    lats = np.full(NPOI, lat)
    from datetime import date
    jday_offset = (date(YEAR, MONTH, 1) - date(YEAR, 1, 1)).days

    print(f"\n{'step':>4s} {'day':>3s} {'hr':>3s}  {'ΔT_max':>11s}  {'Δsoi_max':>11s}  {'biggest field':>20s}")
    print("-" * 80)

    fields_track = [
        ("canopy.tu",      lambda f,j: (f.get_veg()["tu"], j.canopy.tu)),
        ("canopy.tl",      lambda f,j: (f.get_veg()["tl"], j.canopy.tl)),
        ("canopy.t12",     lambda f,j: (f.get_veg()["t12"], j.canopy.t12)),
        ("canopy.t34",     lambda f,j: (f.get_veg()["t34"], j.canopy.t34)),
        ("canopy.q12",     lambda f,j: (f.get_veg()["q12"], j.canopy.q12)),
        ("soil.tg",        lambda f,j: (f.get_soil()["tg"], j.soil.tg)),
        ("soil.tsoi[:,0]", lambda f,j: (f.get_soil()["tsoi"][:, 0], j.soil.tsoi[:, 0])),
        ("soil.wsoi[:,0]", lambda f,j: (f.get_soil()["wsoi"][:, 0], j.soil.wsoi[:, 0])),
        ("snow.fi",        lambda f,j: (f.get_snow()["fi"], j.snow.fi)),
        ("stresstu",       lambda f,j: (f.get_stomata()["stresstu"], j.soil.stresstu)),
    ]
    first_diverge_step = None
    for step in range(n_days * 24):
        day_idx = step // 24
        hour = float(step % 24)
        jday = jday_offset + day_idx + 1
        day_data = {k: gswp3[k][day_idx, :] for k in gswp3}
        atm = diurnal_disaggregate(day_data, hour, jday, lats)

        f77.set_atm(**atm)
        state_atm = eqx.tree_at(
            lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                       s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                       s.atm.raina, s.atm.snowa),
            state,
            (jnp.asarray(atm["ta"]), jnp.asarray(atm["qa"]),
             jnp.asarray(atm["ua"]), jnp.asarray(atm["psurf"]),
             jnp.asarray(atm["coszen"]),
             jnp.asarray(atm["solad"]), jnp.asarray(atm["solai"]),
             jnp.asarray(atm["fira"]),
             jnp.asarray(atm["raina"]), jnp.asarray(atm["snowa"])),
        )
        f77.call_lsxmain()
        state = timestep_body(state_atm, consts, params, DTIME)

        # Compare fields for cell 0 (target cell)
        max_t_diff = 0.0
        max_w_diff = 0.0
        biggest_field = ""
        for fname, getter in fields_track:
            fv, jv = getter(f77, state)
            d = abs(float(np.asarray(fv)[0]) - float(np.asarray(jv)[0]))
            is_w = "wsoi" in fname or "wisoi" in fname or "fi" in fname
            if is_w and d > max_w_diff:
                max_w_diff = d
            elif not is_w and d > max_t_diff:
                max_t_diff = d
                biggest_field = fname

        if max_t_diff > 1e-9 or max_w_diff > 1e-11:
            if first_diverge_step is None:
                first_diverge_step = step

        if step % 12 == 0 or step == first_diverge_step or max_t_diff > 1e-4:
            cz = atm["coszen"][0]
            sun = "☀️" if cz > 0.2 else ("🌅" if cz > 0 else "🌙")
            print(f"{step:>4d} {day_idx:>3d} {int(hour):>3d}  "
                  f"{max_t_diff:>11.3e}  {max_w_diff:>11.3e}  "
                  f"{biggest_field:>20s} {sun}")

    print()
    if first_diverge_step is not None:
        print(f"  First step with ΔT > 1e-9 or Δwsoi > 1e-11: step {first_diverge_step}")
        print(f"  (day {first_diverge_step//24}, hour {first_diverge_step%24})")
    print(f"\nDone. To drill further, modify this script to inject F77 state at the")
    print(f"first-diverge step and isolate per-stage.")


if __name__ == "__main__":
    main()
