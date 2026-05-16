"""Generate reference data for regression testing.

Runs IBIS-JAX for 1 month with a fixed seed and saves key state variables
as .npz files in tests/regression/reference_data/.

Usage:
    python scripts/generate_reference.py [--npoi 4] [--months 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from ibis.constants import get_constants
from ibis.dimensions import GridConfig
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.loops import (
    default_sim_params, run_month, run_year,
)
from ibis.physics.weather import ClimateData


REF_DIR = Path("tests/regression/reference_data")


def make_state(npoi: int):
    """Create deterministic test state."""
    grid = GridConfig(
        nlon=2, nlat=max(npoi // 2, 1), npoi=npoi,
        xres=180.0, yres=180.0 / max(npoi // 2, 1),
        nband=2, nsoilay=6, nsnolay=3, npft=12,
    )
    st = create_ibis_state(grid)

    new_veg = eqx.tree_at(
        lambda v: (v.lai, v.sai, v.fu, v.fl, v.ztop, v.zbot, v.exist),
        st.vegetation,
        (
            jnp.full((npoi, 2), 2.0),
            jnp.full((npoi, 2), 0.5),
            jnp.full((npoi,), 0.8),
            jnp.full((npoi,), 0.5),
            jnp.broadcast_to(jnp.array([5.0, 20.0]), (npoi, 2)),
            jnp.broadcast_to(jnp.array([0.5, 2.0]), (npoi, 2)),
            jnp.ones((npoi, 12)),
        ),
    )
    new_soil = eqx.tree_at(
        lambda s: (s.poros, s.hydraul, s.suction, s.bex,
                   s.sfield, s.swilt, s.sand, s.clay,
                   s.hsoi, s.z0soi, s.wsoi),
        st.soil,
        (
            jnp.full((npoi, 6), 0.45),
            jnp.full((npoi, 6), 1e-5),
            jnp.full((npoi, 6), -0.15),
            jnp.full((npoi, 6), 5.0),
            jnp.full((npoi, 6), 0.35),
            jnp.full((npoi, 6), 0.15),
            jnp.full((npoi, 6), 50.0),
            jnp.full((npoi, 6), 20.0),
            jnp.array([0.10, 0.15, 0.30, 0.50, 1.00, 2.00, 4.00]),
            jnp.full((npoi,), 0.01),
            jnp.full((npoi, 6), 0.40),
        ),
    )
    new_atm = eqx.tree_at(
        lambda a: (a.ta, a.qa, a.ua, a.psurf, a.coszen,
                   a.solad, a.solai, a.fira, a.raina, a.snowa),
        st.atm,
        (
            jnp.full((npoi,), 293.0),
            jnp.full((npoi,), 0.008),
            jnp.full((npoi,), 3.0),
            jnp.full((npoi,), 101325.0),
            jnp.full((npoi,), 0.8),
            jnp.full((npoi, 2), 200.0),
            jnp.full((npoi, 2), 50.0),
            jnp.full((npoi,), 320.0),
            jnp.full((npoi,), 0.0),
            jnp.full((npoi,), 0.0),
        ),
    )
    new_can = eqx.tree_at(
        lambda c: (c.tu, c.ts, c.tl, c.t12, c.t34, c.q12, c.q34),
        st.canopy,
        tuple(jnp.full((npoi,), 293.0) for _ in range(5))
        + tuple(jnp.full((npoi,), 0.008) for _ in range(2)),
    )
    return eqx.tree_at(
        lambda s: (s.vegetation, s.soil, s.atm, s.canopy),
        st,
        (new_veg, new_soil, new_atm, new_can),
    )


def make_climate(npoi: int):
    return ClimateData(
        xint=jnp.full((npoi, 12), 25.0),
        xinq=jnp.full((npoi, 12), 70.0),
        xinprec=jnp.full((npoi, 12), 5.0),
        xinwind=jnp.full((npoi, 12), 2.5),
        xincld=jnp.full((npoi, 12), 50.0),
        xinwet=jnp.full((npoi, 12), 15.0),
        xintrng=jnp.full((npoi, 12), 8.0),
        xintopo=jnp.full((npoi,), 200.0),
    )


def save_state_snapshot(state, name: str):
    """Save key state variables as reference data."""
    data = {
        "tg": np.asarray(state.soil.tg),
        "tsoi": np.asarray(state.soil.tsoi),
        "wsoi": np.asarray(state.soil.wsoi),
        "wisoi": np.asarray(state.soil.wisoi),
        "tu": np.asarray(state.canopy.tu),
        "tl": np.asarray(state.canopy.tl),
        "t12": np.asarray(state.canopy.t12),
        "fi": np.asarray(state.snow.fi),
    }
    path = REF_DIR / f"{name}.npz"
    np.savez_compressed(str(path), **data)
    print(f"  Saved: {path} ({sum(v.nbytes for v in data.values()) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Generate reference data")
    parser.add_argument("--npoi", type=int, default=4)
    parser.add_argument("--months", type=int, default=1,
                        help="Number of months to run (1 or 12)")
    args = parser.parse_args()

    REF_DIR.mkdir(parents=True, exist_ok=True)

    npoi = args.npoi
    state = make_state(npoi)
    consts = get_constants()
    params = default_sim_params(npoi)
    climate = make_climate(npoi)
    key = jax.random.PRNGKey(42)
    dtime = 3600.0

    print(f"Generating reference data (npoi={npoi}, months={args.months})")
    print(f"Platform: {jax.devices()[0].platform}")

    if args.months == 1:
        state, key = run_month(state, consts, params, climate, 0, key, dtime, False, 2)
        save_state_snapshot(state, "month_jan")
    elif args.months == 12:
        state, key = run_year(state, consts, params, climate, key, dtime, 2000, 2)
        save_state_snapshot(state, "year_end")
    else:
        for m in range(args.months):
            state, key = run_month(state, consts, params, climate, m % 12, key, dtime, False, 2)
            print(f"  Month {m+1} done")
        save_state_snapshot(state, f"month_{args.months}")

    print("Done.")


if __name__ == "__main__":
    main()
