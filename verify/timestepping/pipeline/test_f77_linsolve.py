"""Test the hypothesis: the canopy 3-iter divergence is caused by F77 using
no-pivot Gaussian elimination (linsolve) while JAX uses LAPACK gesv (with
partial pivoting). Different solvers produce different rounding errors.

Implement F77's no-pivot Gaussian elimination in JAX and check if turvap
using it produces bit-exact agreement with F77.

If yes → root cause confirmed; we have a fix (swap jnp.linalg.solve for
        this f77_linsolve in canopy_exchange.turvap).
If no  → there's something else in turvap, not just the solver.
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
from ibis.timestepping.loops import default_sim_params
from tests.f2py.ibis_f77_wrapper import NPOI


def f77_linsolve(arr, rhs):
    """JAX implementation of F77's no-pivot Gaussian elimination from
    fortran/ibis_2.6b4/math.f:73 (linsolve).

    arr shape: (npoi, nd, nd), rhs shape: (npoi, nd)
    Returns: vec shape (npoi, nd)

    Operations match F77's loop order:
      forward elimination: for id in 0..nd-1:
        for i in id+1..nd:
          f = arr[i,id] / arr[id,id]
          arr[i,j] -= f * arr[id,j]    for j in id..nd
          rhs[i]   -= f * rhs[id]
      back-substitution: for id in nd-1..0:
        f = sum_j>id arr[id,j] * vec[j]
        vec[id] = (rhs[id] - f) / arr[id,id]

    Sparsity (mplate) is ignored because 0 * x - 0 = 0 in IEEE FP, so
    dense version matches sparse version bit-exact.
    """
    npoi, nd, _ = arr.shape
    # Convert to Python list of (npoi, nd) for clean .at[] semantics
    # Use jax.lax.fori_loop? No — nd is small (9), so static unroll is fine.

    # Make mutable copies
    A = arr
    b = rhs
    # Forward elimination
    for id_ in range(nd - 1):
        diag = A[:, id_, id_]
        for i in range(id_ + 1, nd):
            f = A[:, i, id_] / diag
            # arr[i, j] -= f * arr[id_, j]  for j in id_..nd-1
            update_row_A = A[:, i, :] - f[:, None] * A[:, id_, :]
            A = A.at[:, i, :].set(update_row_A)
            b = b.at[:, i].set(b[:, i] - f * b[:, id_])
    # Back substitution
    vec = jnp.zeros_like(b)
    for id_ in range(nd - 1, -1, -1):
        s = jnp.zeros(npoi)
        for j in range(id_ + 1, nd):
            s = s + A[:, id_, j] * vec[:, j]
        vec = vec.at[:, id_].set((b[:, id_] - s) / A[:, id_, id_])
    return vec


# Monkey-patch jnp.linalg.solve usage inside canopy_exchange to use our solver
import ibis.physics.canopy_exchange as ce

# Save original
_orig_solve = jnp.linalg.solve

# Replace it inside the canopy_exchange module — turvap calls
# jnp.linalg.solve(arr, rhs[..., None]).squeeze(-1)
def _patched_solve(arr, rhs_2d):
    # rhs_2d has shape (npoi, nd, 1); strip last axis, call f77 solver, re-add
    rhs = rhs_2d.squeeze(-1)
    vec = f77_linsolve(arr, rhs)
    return vec[..., None]


def main():
    target = "L-35p0_+140p0"
    lat, lon = -35.0, 140.0
    tmpl = soil_template_for_lat(lat)
    cells = [(target, lat, lon, tmpl, tmpl)] * NPOI
    print(f"=== Test F77-style linsolve in JAX turvap: {target} ===")

    YEAR, MONTH = 1990, 6
    gswp3 = read_gswp3_month(YEAR, MONTH, cells)
    lats = np.full(NPOI, lat)
    jday = (date(YEAR, MONTH, 1) - date(YEAR, 1, 1)).days + 1
    day_data = {k: gswp3[k][0, :] for k in gswp3}
    atm_d = diurnal_disaggregate(day_data, 0.0, jday, lats)

    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)

    # Apply forcing
    f77.set_atm(**atm_d)

    # Run F77 full timestep
    f77.call_lsxmain()
    fv = f77.get_veg(); fs = f77.get_soil(); ffx = f77.get_fluxes()
    f77_t12 = float(fv["t12"][0]); f77_t34 = float(fv["t34"][0])
    f77_q12 = float(fv["q12"][0]); f77_q34 = float(fv["q34"][0])
    f77_tg  = float(fs["tg"][0])
    f77_fseng = float(ffx["fseng"][0])
    f77_fseni = float(ffx["fseni"][0])

    # =========================================
    # JAX with STOCK linalg.solve (the bug case)
    # =========================================
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    state = eqx.tree_at(
        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                   s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                   s.atm.raina, s.atm.snowa),
        state,
        tuple(jnp.asarray(atm_d[k]) for k in
              ("ta","qa","ua","psurf","coszen","solad","solai","fira","raina","snowa")),
    )
    from ibis.timestepping.loops import timestep_body
    stock = timestep_body(state, consts, params, DTIME)

    # =========================================
    # JAX with F77-STYLE linsolve patched in
    # =========================================
    # Monkey-patch turvap's solver call by editing canopy_exchange.py's
    # jnp.linalg.solve reference. Since turvap uses `vec = jnp.linalg.solve(...)`,
    # we replace jnp.linalg.solve at the module level.
    saved_solve = ce.jnp.linalg.solve
    # Patch ce.jnp.linalg.solve — but this won't work because it's a frozen attribute.
    # Instead, monkey-patch jax.numpy.linalg.solve globally (impacts only this proc).
    import jax.numpy.linalg as jlinalg
    jlinalg.solve = _patched_solve  # type: ignore

    # Reload? jit caches may have captured the original. Let's force a fresh
    # invocation by NOT using jit. Easiest: call timestep_body with new state.
    state2 = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    state2 = eqx.tree_at(
        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                   s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                   s.atm.raina, s.atm.snowa),
        state2,
        tuple(jnp.asarray(atm_d[k]) for k in
              ("ta","qa","ua","psurf","coszen","solad","solai","fira","raina","snowa")),
    )
    patched = timestep_body(state2, consts, params, DTIME)

    # Restore
    jlinalg.solve = saved_solve  # type: ignore

    def show(label, t12, t34, q12, q34, tg, fseng, fseni):
        dt12 = abs(t12 - f77_t12); dt34 = abs(t34 - f77_t34)
        dq12 = abs(q12 - f77_q12); dq34 = abs(q34 - f77_q34)
        dtg = abs(tg - f77_tg)
        dfg = abs(fseng - f77_fseng); dfi = abs(fseni - f77_fseni)
        print(f"\n  {label}")
        print(f"    Δt12={dt12:.3e}  Δt34={dt34:.3e}  Δq12={dq12:.3e}")
        print(f"    Δtg={dtg:.3e}  Δfseng={dfg:.3e}  Δfseni={dfi:.3e}")

    def grab(x): return float(np.asarray(x).flat[0])

    show("STOCK JAX (jnp.linalg.solve)",
         grab(stock.canopy.t12), grab(stock.canopy.t34),
         grab(stock.canopy.q12), grab(stock.canopy.q34),
         grab(stock.soil.tg),
         grab(stock.canopy.fseng), grab(stock.canopy.fseni))

    show("PATCHED JAX (F77-style no-pivot)",
         grab(patched.canopy.t12), grab(patched.canopy.t34),
         grab(patched.canopy.q12), grab(patched.canopy.q34),
         grab(patched.soil.tg),
         grab(patched.canopy.fseng), grab(patched.canopy.fseni))


if __name__ == "__main__":
    main()
