"""Check whether F77's stresstu equals 1.0 exactly when soil is unstressed.

If yes, F77's stressu/stresstu reduces to froot[k] exactly, and OLD JAX
(raw froot) matches F77. Then my "fix" is wrong because F77 happens to
behave as raw froot in this case.

Print stresstu, stressu values for cell 0 at step 0 (initial wet soil)
in both F77 and JAX, comparing bit-by-bit.
"""
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from verify.timestepping.verify_lsxmain import (
    setup_inputs, setup_f77_state, setup_jax_state, NPOI,
)
from ibis.physics.stomata import drystress


def hexrepr(x):
    return float(x).hex()


def main():
    soil_i, snow_i, atm_i, veg_i = setup_inputs()

    # F77 path: run drystress alone, read stresstu
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    f77.call_drystress()
    f_stom = f77.get_stomata()

    # JAX path: call drystress with same initial state
    jax_state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    s = jax_state.soil
    stresstl, stresstu, stressl, stressu = drystress(
        s.wsoi, s.wisoi, s.swilt, s.sfield, jax_state.vegetation.froot
    )

    print("Cell 0 (tropical, wsoi=0.6 uniform, wisoi=0):")
    print(f"  F77 stresstu = {float(f_stom['stresstu'][0]):.17e}")
    print(f"      hex      = {hexrepr(f_stom['stresstu'][0])}")
    print(f"  JAX stresstu = {float(stresstu[0]):.17e}")
    print(f"      hex      = {hexrepr(stresstu[0])}")
    print(f"  diff = {float(f_stom['stresstu'][0]) - float(stresstu[0]):.3e}")
    print()

    # Per-layer stressu
    print("  Per-layer stressu (F77 vs JAX):")
    for k in range(6):
        a = float(f_stom.get("stressu_k", [None]*6)[k]) if "stressu_k" in f_stom else None
        b = float(stressu[0, k])
        print(f"    layer {k}: JAX = {b:.17e}  hex={hexrepr(b)}")

    print()
    print("  sum manually (left-to-right F77-style):")
    froot_col1 = np.asarray(jax_state.vegetation.froot)[:, 1]
    s_seq = np.float64(0.0)
    for v in froot_col1:
        s_seq += np.float64(v)
    print(f"    Sequential sum = {s_seq:.17e}  hex={s_seq.hex()}")
    print(f"    Equals 1.0 exactly? {s_seq == 1.0}")

    s_jnp = float(jnp.sum(jnp.asarray(froot_col1)))
    print(f"    jnp.sum        = {s_jnp:.17e}  hex={float(s_jnp).hex()}")
    print(f"    Equals 1.0 exactly? {s_jnp == 1.0}")


if __name__ == "__main__":
    main()
