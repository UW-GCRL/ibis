"""Prove (or refute) that wsoi drift is chaotic-IC sensitivity, not a code bug.

If we run JAX TWICE with the same forcing but initial wsoi differing by 1e-8,
how do they drift? Compare to F77-vs-JAX drift over the same scenario.

If JAX-vs-JAX-perturbed drift matches F77-vs-JAX drift in shape and magnitude,
the system has natural Lyapunov-style amplification, NOT a port bug.
"""
from __future__ import annotations
import numpy as np
import equinox as eqx
import jax.numpy as jnp

from verify.timestepping.verify_lsxmain import (
    setup_inputs, setup_jax_state, setup_f77_state, diurnal_atm,
    update_jax_atm, heat_wave_forcing, NPOI, CELL_LABELS, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body


def run_jax_only(state, n_steps, atm_i, forcing, params, consts):
    """Run JAX for n_steps under heat-wave forcing, return per-step wsoi[0,0]."""
    history = []
    for step in range(n_steps):
        hour = float(step % 24)
        new_atm = diurnal_atm(atm_i, hour, **forcing(step, n_steps))
        state = update_jax_atm(state, new_atm)
        state = timestep_body(state, consts, params, DTIME)
        history.append(float(state.soil.wsoi[0, 0]))
    return np.array(history)


def run_f77_only(soil_i, snow_i, atm_i, veg_i, n_steps, forcing):
    """Run F77 for n_steps under heat-wave forcing, return per-step wsoi[0,0]."""
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    history = []
    for step in range(n_steps):
        hour = float(step % 24)
        new_atm = diurnal_atm(atm_i, hour, **forcing(step, n_steps))
        f77.set_atm(**new_atm)
        f77.call_lsxmain()
        history.append(float(f77.get_soil()["wsoi"][0, 0]))
    return np.array(history)


def main():
    soil_i, snow_i, atm_i, veg_i = setup_inputs()
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    n_steps = 168

    # Baseline JAX
    print(f"Running JAX baseline ({n_steps} steps, heat wave forcing) ...")
    state_a = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    wsoi_jax_a = run_jax_only(state_a, n_steps, atm_i, heat_wave_forcing,
                               params, consts)

    # JAX with tiny perturbation to cell-0 initial wsoi
    print("Running JAX with wsoi[0,0] perturbed by +1e-8 ...")
    state_b = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    # Add 1e-8 to cell 0 wsoi[0,0]
    new_wsoi = state_b.soil.wsoi.at[0, 0].add(1e-8)
    state_b = eqx.tree_at(lambda s: s.soil.wsoi, state_b, new_wsoi)
    wsoi_jax_b = run_jax_only(state_b, n_steps, atm_i, heat_wave_forcing,
                               params, consts)

    # F77
    print("Running F77 baseline ...")
    wsoi_f77 = run_f77_only(soil_i, snow_i, atm_i, veg_i, n_steps,
                             heat_wave_forcing)

    # Compute drifts at each step
    drift_jax_vs_f77 = np.abs(wsoi_jax_a - wsoi_f77)
    drift_jax_vs_jaxperturbed = np.abs(wsoi_jax_a - wsoi_jax_b)

    print()
    print("Per-step drift on cell 0 wsoi[:,0] over heat-wave scenario:")
    print(f"{'step':>4s} {'JAX_a':>11s} {'JAX_b':>11s} {'F77':>11s} "
          f"{'JAX_a-F77':>10s} {'JAX_a-JAX_b':>11s}")
    print("-" * 80)
    for step in [0, 11, 23, 47, 59, 67, 68, 69, 71, 95, 119, 143, 167]:
        print(f"{step:>4d} {wsoi_jax_a[step]:>11.7f} "
              f"{wsoi_jax_b[step]:>11.7f} {wsoi_f77[step]:>11.7f} "
              f"{drift_jax_vs_f77[step]:>10.3e} "
              f"{drift_jax_vs_jaxperturbed[step]:>11.3e}")

    print()
    print(f"Final step ({n_steps - 1}):")
    print(f"  JAX_a vs F77:           drift = {drift_jax_vs_f77[-1]:.3e}")
    print(f"  JAX_a vs JAX_b (1e-8):  drift = {drift_jax_vs_jaxperturbed[-1]:.3e}")
    print()
    print("Interpretation:")
    print("  If drifts have similar magnitude/growth shape, the system is")
    print("  chaotic and the F77-JAX drift is IC-sensitivity, not a code bug.")
    print("  If JAX-JAX_perturbed drift is much smaller than JAX-F77 drift,")
    print("  the code has a real systematic bias to find.")


if __name__ == "__main__":
    main()
