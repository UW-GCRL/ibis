"""Per-step amplification diagnostic for heat-wave wsoi drift on cell 0.

For each step, capture F77 vs JAX values of state quantities and ratios
of successive Δs. The sub-quantity whose Δ amplifies fastest is the source
of accumulated divergence.

Quantities tracked (all are persisted state fields, no extra plumbing):
  wsoi[:,0] (input)  — start of step
  stresstu          — drystress output, persisted in soil state
  fvapg, fsenu      — last-iter canopy fluxes
  tg, tu, t12, t34  — temperatures
  wsoi[:,0] (out)   — after lsxmain
"""
from __future__ import annotations
import numpy as np
from verify.timestepping.verify_lsxmain import (
    setup_inputs, setup_f77_state, setup_jax_state, diurnal_atm,
    update_jax_atm, heat_wave_forcing, NPOI, CELL_LABELS, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body
import equinox as eqx

CELL = 0


def cell(name, f77_val, jax_val):
    a = float(np.asarray(f77_val).ravel()[CELL])
    b = float(np.asarray(jax_val).ravel()[CELL])
    return name, a, b, abs(a - b)


def main():
    soil_i, snow_i, atm_i, veg_i = setup_inputs()
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    n_steps = 100
    print(f"Cell {CELL} ({CELL_LABELS[CELL]}) over {n_steps} steps")
    print(f"{'step':>4s} {'hr':>4s} {'wsoi_Δ':>11s} {'stresstu_Δ':>11s} "
          f"{'tg_Δ':>11s} {'tu_Δ':>11s} {'t12_Δ':>11s} "
          f"{'fvapg_Δ':>11s} {'fsenu_Δ':>11s} {'amp':>7s}")
    print("-" * 110)

    prev_wsoi_d = 0.0
    for step in range(n_steps):
        hour = float(step % 24)
        fk = heat_wave_forcing(step, n_steps)
        new_atm = diurnal_atm(atm_i, hour, **fk)
        f77.set_atm(**new_atm)
        state = update_jax_atm(state, new_atm)
        f77.call_lsxmain()
        state = timestep_body(state, consts, params, DTIME)

        f_soil = f77.get_soil()
        f_veg = f77.get_veg()
        f_flux = f77.get_fluxes()
        f_stom = f77.get_stomata()

        _, _, _, wsoi_d = cell("wsoi", f_soil["wsoi"][:, 0], state.soil.wsoi[:, 0])
        _, _, _, stresstu_d = cell("stresstu", f_stom["stresstu"], state.soil.stresstu)
        _, _, _, tg_d = cell("tg", f_soil["tg"], state.soil.tg)
        _, _, _, tu_d = cell("tu", f_veg["tu"], state.canopy.tu)
        _, _, _, t12_d = cell("t12", f_veg["t12"], state.canopy.t12)
        _, _, _, fvapg_d = cell("fvapg", f_flux["fvapg"], state.canopy.fvapg)
        _, _, _, fsenu_d = cell("fsenu", f_flux["fsenu"], state.canopy.fsenu)

        amp = wsoi_d / max(prev_wsoi_d, 1e-30)
        prev_wsoi_d = wsoi_d
        print(f"{step:>4d} {hour:>4.1f} "
              f"{wsoi_d:>11.3e} {stresstu_d:>11.3e} "
              f"{tg_d:>11.3e} {tu_d:>11.3e} {t12_d:>11.3e} "
              f"{fvapg_d:>11.3e} {fsenu_d:>11.3e} {amp:>7.2f}")


if __name__ == "__main__":
    main()
