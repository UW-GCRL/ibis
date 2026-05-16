"""Mass balance diagnostic: does F77 vs JAX conserve total soil water?

If both lose the same total over n steps, the drift is just vertical
redistribution (root-uptake / drainage profile differs). If they lose
DIFFERENT total mass, one has a non-conservative bug (numerical sink).

Track cell 0 (the only one that drifts under heat wave).
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
HSOI = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])  # m per layer
POROS = 0.45  # uniform poros
RHOW = 1000.0  # kg/m^3 water density


def total_water_kg_m2(wsoi, wisoi):
    """Compute total soil water column (kg/m²) for cell."""
    poros = POROS
    # wsoi is liquid water fraction of pore space, wisoi is ice fraction
    # mass = sum over layers: poros * hsoi * rho * (wsoi * (1-wisoi) + wisoi*ratio)
    # We track only LIQUID equivalent (wsoi * (1-wisoi)) + ICE (wisoi)
    layer_liquid = poros * HSOI * RHOW * (1.0 - wisoi) * wsoi
    layer_ice = poros * HSOI * RHOW * wisoi * 0.917  # ice density ratio
    return float(np.sum(layer_liquid + layer_ice))


def main():
    soil_i, snow_i, atm_i, veg_i = setup_inputs()
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    n_steps = 168

    # Initial state
    f_soil0 = f77.get_soil()
    w_f77_init = total_water_kg_m2(f_soil0["wsoi"][CELL], f_soil0["wisoi"][CELL])
    w_jax_init = total_water_kg_m2(np.asarray(state.soil.wsoi[CELL]),
                                     np.asarray(state.soil.wisoi[CELL]))

    cum = {}
    for k in ("fvapg_f77","fvapg_jax","raing_f77","raing_jax",
              "gtrans_f77","gtrans_jax","gtransu_f77","gtransu_jax",
              "gtransl_f77","gtransl_jax","gdrain_f77","gdrain_jax",
              "grunof_f77","grunof_jax","gadjust_f77","gadjust_jax",
              "ginvap_f77","ginvap_jax","gsuvap_f77","gsuvap_jax"):
        cum[k] = 0.0

    for step in range(n_steps):
        hour = float(step % 24)
        new_atm = diurnal_atm(atm_i, hour, **heat_wave_forcing(step, n_steps))
        f77.set_atm(**new_atm)
        state = update_jax_atm(state, new_atm)
        f77.call_lsxmain()
        state = timestep_body(state, consts, params, DTIME)

        f_flux = f77.get_fluxes()
        f_hydro = f77.get_hydro()
        cum["fvapg_f77"] += float(f_flux["fvapg"][CELL]) * DTIME
        cum["fvapg_jax"] += float(state.canopy.fvapg[CELL]) * DTIME
        cum["raing_f77"] += float(f_flux["raing"][CELL]) * DTIME
        cum["raing_jax"] += float(state.canopy.raing[CELL]) * DTIME
        cum["gtrans_f77"]  += float(f_hydro["gtrans"][CELL]) * DTIME
        cum["gtrans_jax"]  += float(state.hydrology.gtrans[CELL]) * DTIME
        cum["gtransu_f77"] += float(f_hydro["gtransu"][CELL]) * DTIME
        cum["gtransu_jax"] += float(state.hydrology.gtransu[CELL]) * DTIME
        cum["gtransl_f77"] += float(f_hydro["gtransl"][CELL]) * DTIME
        cum["gtransl_jax"] += float(state.hydrology.gtransl[CELL]) * DTIME
        cum["gdrain_f77"]  += float(f_hydro["gdrain"][CELL]) * DTIME
        cum["gdrain_jax"]  += float(state.hydrology.gdrain[CELL]) * DTIME
        cum["grunof_f77"]  += float(f_hydro["grunof"][CELL]) * DTIME
        cum["grunof_jax"]  += float(state.hydrology.grunof[CELL]) * DTIME
        cum["gadjust_f77"] += float(f_hydro["gadjust"][CELL]) * DTIME
        cum["gadjust_jax"] += float(state.hydrology.gadjust[CELL]) * DTIME
        cum["ginvap_f77"]  += float(f_hydro["ginvap"][CELL]) * DTIME
        cum["ginvap_jax"]  += float(state.hydrology.ginvap[CELL]) * DTIME
        cum["gsuvap_f77"]  += float(f_hydro["gsuvap"][CELL]) * DTIME
        cum["gsuvap_jax"]  += float(state.hydrology.gsuvap[CELL]) * DTIME

    # Final state
    f_soil = f77.get_soil()
    w_f77_final = total_water_kg_m2(f_soil["wsoi"][CELL], f_soil["wisoi"][CELL])
    w_jax_final = total_water_kg_m2(np.asarray(state.soil.wsoi[CELL]),
                                      np.asarray(state.soil.wisoi[CELL]))

    print(f"Mass balance for cell {CELL} ({CELL_LABELS[CELL]}) over {n_steps} steps, heat wave forcing")
    print()
    print(f"Initial water (kg/m^2):  F77 = {w_f77_init:10.4f}   JAX = {w_jax_init:10.4f}")
    print(f"Final water (kg/m^2):    F77 = {w_f77_final:10.4f}   JAX = {w_jax_final:10.4f}")
    print(f"Water lost (Δ):          F77 = {w_f77_init - w_f77_final:+10.4f}   JAX = {w_jax_init - w_jax_final:+10.4f}")
    print(f"Discrepancy (F77 - JAX): {(w_f77_init - w_f77_final) - (w_jax_init - w_jax_final):+10.4f} kg/m^2")
    print()
    print(f"{'flux':>10s}  {'F77':>12s}  {'JAX':>12s}  {'F77-JAX':>12s}")
    for f in ("fvapg", "gtrans", "gtransu", "gtransl",
              "gdrain", "grunof", "gadjust", "ginvap", "gsuvap", "raing"):
        a = cum[f"{f}_f77"]; b = cum[f"{f}_jax"]
        print(f"{f:>10s}  {a:+12.6f}  {b:+12.6f}  {a-b:+12.6e}")


if __name__ == "__main__":
    main()
