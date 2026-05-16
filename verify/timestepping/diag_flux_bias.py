"""Pinpoint which flux carries the systematic F77-JAX bias.

Track signed differences (F77 - JAX) per step for:
  fvapg     soil evap (kg/m2/s)
  fvapi     snow evap
  fvapa     atmospheric total vapour flux
  gtrans    total transpiration (m/s)
  gtransu   upper canopy transpiration
  gtransl   lower canopy transpiration
  ginvap    interception evap
  gsuvap    surface evap
  grunof    surface runoff
  gdrain    deep drainage
  gadjust   mass-conservation adjustment
  stresstu  upper canopy soil moisture stress
  totcondub upper broadleaf total conductance

Whichever has a CONSISTENTLY signed drift (not zero-mean noise) over the run
is the source.
"""
from __future__ import annotations
import numpy as np
import equinox as eqx

from verify.timestepping.verify_lsxmain import (
    setup_inputs, setup_f77_state, setup_jax_state, diurnal_atm,
    update_jax_atm, heat_wave_forcing, NPOI, CELL_LABELS, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body


CELL = 0


def signed_diff(f77_arr, jax_arr):
    a = float(np.asarray(f77_arr).ravel()[CELL])
    b = float(np.asarray(jax_arr).ravel()[CELL])
    return a - b


def main():
    soil_i, snow_i, atm_i, veg_i = setup_inputs()
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    fields = ["fvapg", "fvapi", "fvapa", "gtrans", "gtransu", "gtransl",
              "ginvap", "gsuvap", "grunof", "gdrain", "gadjust",
              "stresstu", "wsoi[:,0]"]

    n_steps = 100
    print(f"Tracking signed F77-minus-JAX diffs on cell {CELL} ({CELL_LABELS[CELL]})")
    print(f"Heat wave forcing, {n_steps} steps. Negative means JAX is bigger.")
    print()
    print("step  " + "  ".join(f"{f:>11s}" for f in fields))
    print("-" * (6 + 13 * len(fields)))

    cum_signed = {f: 0.0 for f in fields}
    for step in range(n_steps):
        hour = float(step % 24)
        fk = heat_wave_forcing(step, n_steps)
        new_atm = diurnal_atm(atm_i, hour, **fk)
        f77.set_atm(**new_atm)
        state = update_jax_atm(state, new_atm)
        f77.call_lsxmain()
        state = timestep_body(state, consts, params, DTIME)

        f_flux = f77.get_fluxes()
        f_hydro = f77.get_hydro()
        f_stom = f77.get_stomata()
        f_soil = f77.get_soil()

        # JAX equivalents — try to map best we can from state
        diffs = {
            "fvapg":     signed_diff(f_flux["fvapg"], state.canopy.fvapg),
            "fvapi":     signed_diff(f_flux["fvapi"], state.canopy.fvapi),
            "fvapa":     signed_diff(f_flux["fvapa"], state.canopy.fvapg
                                       + state.canopy.fvapi),  # approx
            "gtrans":    signed_diff(f_hydro["gtrans"], state.canopy.fvapg * 0.0),  # placeholder
            "gtransu":   signed_diff(f_hydro["gtransu"], state.canopy.fvapg * 0.0),
            "gtransl":   signed_diff(f_hydro["gtransl"], state.canopy.fvapg * 0.0),
            "ginvap":    signed_diff(f_hydro["ginvap"], state.canopy.fvapg * 0.0),
            "gsuvap":    signed_diff(f_hydro["gsuvap"], state.canopy.fvapg * 0.0),
            "grunof":    signed_diff(f_hydro["grunof"], state.canopy.fvapg * 0.0),
            "gdrain":    signed_diff(f_hydro["gdrain"], state.canopy.fvapg * 0.0),
            "gadjust":   signed_diff(f_hydro["gadjust"], state.canopy.fvapg * 0.0),
            "stresstu":  signed_diff(f_stom["stresstu"], state.soil.stresstu),
            "wsoi[:,0]": signed_diff(f_soil["wsoi"][:, 0], state.soil.wsoi[:, 0]),
        }
        for f, v in diffs.items():
            cum_signed[f] += v

        if step % 6 == 0 or step >= 65 and step <= 75:
            cells = "  ".join(f"{diffs[f]:>+11.3e}" for f in fields)
            print(f"{step:>4d}  {cells}")

    print()
    print("Cumulative signed sum (sign indicates systematic direction):")
    for f in fields:
        s = cum_signed[f]
        direction = ("JAX>F77" if s < 0 else "F77>JAX") if abs(s) > 1e-20 else "~0"
        print(f"  {f:>11s}: {s:>+12.3e}   ({direction})")


if __name__ == "__main__":
    main()
