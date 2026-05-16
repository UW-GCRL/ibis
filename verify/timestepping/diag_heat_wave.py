"""Diagnostic: pinpoint when/where wsoi drift starts under heat wave forcing.

Runs the heat-wave scenario step-by-step and prints per-cell wsoi values
for both F77 and JAX every 12 steps. Goal: see which cell drifts first,
correlate with the absolute wsoi value (near swilt? near 0? near sfield?),
and identify the timestep at which divergence becomes >1e-4.
"""
from __future__ import annotations
import sys
import numpy as np

# Reuse the verifier's setup, mitigations, and helpers
from verify.timestepping.verify_lsxmain import (
    setup_inputs, setup_f77_state, setup_jax_state, diurnal_atm,
    update_jax_atm, heat_wave_forcing, NPOI, CELL_LABELS, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body
import equinox as eqx
import jax.numpy as jnp


def main():
    soil_i, snow_i, atm_i, veg_i = setup_inputs()
    print("Heat wave diagnostic — per-cell wsoi[0] over 168 steps")
    print(f"Initial wsoi[:,0] = {soil_i['wsoi'][:, 0]}")
    print(f"Soil swilt        = ~0.15 (default)")
    print(f"Soil sfield       = ~0.35 (default)")
    print(f"Cells: " + ", ".join(f"[{i}]{lab}" for i, lab in enumerate(CELL_LABELS)))
    print()

    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    print(f"{'step':>4s} {'hour':>4s} {'ta_off':>6s}  "
          f"{'cell0 F77/JAX':>22s}  {'cell1 F77/JAX':>22s}  "
          f"{'cell2 F77/JAX':>22s}  {'cell3 F77/JAX':>22s}  "
          f"{'cell4 F77/JAX':>22s}  {'maxΔ':>9s}")
    print("-" * 160)

    first_significant_step = None
    for step in range(168):
        hour = float(step % 24)
        fk = heat_wave_forcing(step, 168)
        new_atm = diurnal_atm(atm_i, hour, **fk)
        f77.set_atm(**new_atm)
        state = update_jax_atm(state, new_atm)
        f77.call_lsxmain()
        state = timestep_body(state, consts, params, DTIME)
        if step % 12 == 0 or step >= 100:
            f_soil = f77.get_soil()
            f77_w = np.asarray(f_soil["wsoi"][:, 0])
            jax_w = np.asarray(state.soil.wsoi[:, 0])
            diffs = np.abs(f77_w - jax_w)
            mdiff = float(diffs.max())
            cells = " ".join(
                f"{f77_w[i]:.4f}/{jax_w[i]:.4f} ({diffs[i]:.0e})"
                for i in range(NPOI))
            ta_off = fk.get("ta_offset", 0.0)
            print(f"{step:>4d} {hour:>4.1f} {ta_off:>+6.2f}  {cells}  {mdiff:.2e}")
            if first_significant_step is None and mdiff > 1e-4:
                first_significant_step = step

    print()
    if first_significant_step is not None:
        print(f"First step with max wsoi |Δ| > 1e-4: step {first_significant_step}")
        print(f"That's ~hour {first_significant_step} ({first_significant_step / 24.0:.1f} days)")
    else:
        print("Never exceeded 1e-4 (false alarm?)")

    # Also report other soil layers
    print()
    print("End-of-run state by soil layer:")
    f_soil = f77.get_soil()
    for lay in range(6):
        f77_l = np.asarray(f_soil["wsoi"][:, lay])
        jax_l = np.asarray(state.soil.wsoi[:, lay])
        d = np.abs(f77_l - jax_l)
        print(f"  layer {lay}: F77={f77_l}, JAX={jax_l}, |Δ|={d}")


if __name__ == "__main__":
    main()
