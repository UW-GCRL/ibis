"""Performance benchmarking for IBIS-JAX.

Measures:
  - JIT compilation time
  - Single timestep execution time (CPU)
  - Full-day scan execution time
  - Memory footprint
  - Scaling with grid size

Usage:
    python scripts/benchmark.py [--npoi 100] [--platform cpu|gpu]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import equinox as eqx

from ibis.constants import get_constants
from ibis.dimensions import GridConfig
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.loops import (
    default_sim_params, timestep_body, run_month,
)
from ibis.physics.weather import ClimateData


def make_state(npoi: int):
    """Create test state for benchmarking."""
    grid = GridConfig(
        nlon=max(npoi, 2), nlat=1, npoi=npoi,
        xres=360.0 / max(npoi, 2), yres=180.0,
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


def benchmark_jit_compilation(npoi: int):
    """Measure JIT compilation time for timestep_body."""
    print(f"\n--- JIT Compilation (npoi={npoi}) ---")
    state = make_state(npoi)
    consts = get_constants()
    params = default_sim_params(npoi)
    dtime = 3600.0

    @jax.jit
    def _step(st):
        return timestep_body(st, consts, params, dtime)

    t0 = time.perf_counter()
    st1 = _step(state)
    st1.canopy.tu.block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"  First call (compile + run): {t_compile:.2f} s")

    t0 = time.perf_counter()
    st2 = _step(st1)
    st2.canopy.tu.block_until_ready()
    t_run = time.perf_counter() - t0
    print(f"  Second call (run only):     {t_run*1000:.2f} ms")

    return t_compile, t_run


def benchmark_timestep_throughput(npoi: int, nsteps: int = 24):
    """Measure throughput for multiple timesteps."""
    print(f"\n--- Timestep Throughput (npoi={npoi}, nsteps={nsteps}) ---")
    state = make_state(npoi)
    consts = get_constants()
    params = default_sim_params(npoi)
    dtime = 3600.0

    @jax.jit
    def _step(st):
        return timestep_body(st, consts, params, dtime)

    # Warmup
    st = _step(state)
    st.canopy.tu.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(nsteps):
        st = _step(st)
    st.canopy.tu.block_until_ready()
    t_total = time.perf_counter() - t0

    print(f"  {nsteps} steps: {t_total:.3f} s")
    print(f"  Per step:   {t_total/nsteps*1000:.2f} ms")
    print(f"  Throughput: {npoi * nsteps / t_total:.0f} point-steps/s")
    return t_total


def benchmark_month(npoi: int):
    """Measure time for a full month (weather gen + day loop)."""
    print(f"\n--- Full Month (npoi={npoi}) ---")
    state = make_state(npoi)
    consts = get_constants()
    params = default_sim_params(npoi)
    climate = make_climate(npoi)
    key = jax.random.PRNGKey(42)
    dtime = 3600.0

    t0 = time.perf_counter()
    st_out, _ = run_month(state, consts, params, climate, 0, key, dtime, False, 2)
    st_out.canopy.tu.block_until_ready()
    t_month = time.perf_counter() - t0

    print(f"  January (31 days x 24 steps): {t_month:.2f} s")
    print(f"  Per day:  {t_month/31:.3f} s")
    return t_month


def benchmark_memory(npoi: int):
    """Estimate memory footprint of state pytree."""
    print(f"\n--- Memory Footprint (npoi={npoi}) ---")
    state = make_state(npoi)
    leaves = jax.tree.leaves(state)
    total_bytes = sum(leaf.nbytes for leaf in leaves)
    print(f"  State leaves:     {len(leaves)}")
    print(f"  Total state size: {total_bytes / 1024:.1f} KB ({total_bytes / 1024**2:.2f} MB)")
    return total_bytes


def benchmark_scaling(npoi_list=None):
    """Measure scaling with grid size."""
    if npoi_list is None:
        npoi_list = [4, 16, 64, 256]
    print("\n--- Scaling ---")
    print(f"  {'npoi':>8}  {'compile(s)':>12}  {'step(ms)':>10}  {'memory(KB)':>12}")
    for npoi in npoi_list:
        try:
            t_compile, t_run = benchmark_jit_compilation(npoi)
            total_bytes = sum(
                leaf.nbytes for leaf in jax.tree.leaves(make_state(npoi))
            )
            print(f"  {npoi:>8}  {t_compile:>12.2f}  {t_run*1000:>10.2f}  {total_bytes/1024:>12.1f}")
        except Exception as e:
            print(f"  {npoi:>8}  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="IBIS-JAX Benchmarks")
    parser.add_argument("--npoi", type=int, default=16, help="Number of grid points")
    parser.add_argument("--platform", choices=["cpu", "gpu", "tpu"], default=None)
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    args = parser.parse_args()

    if args.platform:
        jax.config.update("jax_platform_name", args.platform)

    print(f"JAX platform: {jax.devices()[0].platform}")
    print(f"JAX version:  {jax.__version__}")

    if args.scaling:
        benchmark_scaling()
    else:
        npoi = args.npoi
        benchmark_memory(npoi)
        benchmark_jit_compilation(npoi)
        benchmark_timestep_throughput(npoi)
        benchmark_month(npoi)


if __name__ == "__main__":
    main()
