"""Benchmark IBIS JAX timestep_body on GPU vs CPU.

Loads a real saved checkpoint (year-end state) from the running
50-year run, builds a synthetic atm-forcing trajectory, and times
several configurations:

  1. CPU,  no JIT, single hourly step          (current way)
  2. GPU,  no JIT, single hourly step          (launch-overhead-bound)
  3. GPU, JIT'd,   single hourly step
  4. GPU, JIT'd,   scan over 24 hours          (1 day per launch)
  5. GPU, JIT'd,   scan over 24*30 = 720 hours (1 month per launch)
  6. GPU, JIT'd, vmap over N cells, scan over month
     (the multi-cell version — what would replace 24-batch parallel CPU)

The whole goal: pick the configuration that beats CPU by the largest factor
without going OOM on the 4090's 24 GB VRAM.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

# Must be set BEFORE jax import for GPU mode
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body
from tests.f2py.ibis_f77_wrapper import NPOI as NPOI_F77

DTIME = 3600.0


def load_state(ckpt_path):
    with open(ckpt_path, "rb") as f:
        d = pickle.load(f)
    return d["jax_state"]


def fake_atm(state, hour: int, jday: int = 200):
    """Trivial synthetic forcing — values don't have to be physical for benchmark.
    We just need the scan to evolve state with finite arithmetic."""
    n = state.atm.ta.shape[0]
    ta    = jnp.full((n,), 285.0 + 5.0 * np.sin(hour * 2 * np.pi / 24))
    qa    = jnp.full((n,), 0.008)
    ua    = jnp.full((n,), 3.0)
    psurf = jnp.full((n,), 95000.0)
    cz    = jnp.full((n,), max(0.0, np.sin(hour * 2 * np.pi / 24 - np.pi / 2)))
    nband = state.atm.solad.shape[1]
    solad = jnp.full((n, nband), 200.0 * cz[0])
    solai = jnp.full((n, nband), 100.0 * cz[0])
    fira  = jnp.full((n,), 330.0)
    raina = jnp.full((n,), 0.0)
    snowa = jnp.full((n,), 0.0)
    new_atm = eqx.tree_at(
        lambda a: (a.ta, a.qa, a.ua, a.psurf, a.coszen,
                   a.solad, a.solai, a.fira, a.raina, a.snowa),
        state.atm,
        (ta, qa, ua, psurf, cz, solad, solai, fira, raina, snowa),
    )
    return eqx.tree_at(lambda s: s.atm, state, new_atm)


def time_block(label, fn, n_warmup=2, n_meas=5):
    for _ in range(n_warmup):
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, tuple):
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                out)
    t0 = time.time()
    for _ in range(n_meas):
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        else:
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                out)
    dt = (time.time() - t0) / n_meas
    print(f"  {label:60s}  {dt*1000:10.2f} ms / call")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/shared/nas2/Hangkai/ibis/verify_outputs/"
                                       "years_1951_2000/checkpoints/b00_state_after_1951.pkl")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    args = ap.parse_args()

    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

    print("=" * 80)
    print(f"JAX device:   {jax.default_backend()}  ({jax.devices()})")
    print(f"Checkpoint:   {args.ckpt}")
    print("=" * 80)

    state0 = load_state(args.ckpt)
    consts = get_constants()
    params = default_sim_params(NPOI_F77)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    print(f"State npoi:   {state0.atm.ta.shape[0]}")
    print()

    # -- 1. Single-step, no JIT --
    def step_nojit():
        s = fake_atm(state0, hour=10)
        return timestep_body(s, consts, params, DTIME)

    print(f"--- single-step (npoi=5) ---")
    t_nojit = time_block("no-JIT  single step", step_nojit)

    # -- 2. Single-step, JIT'd --
    @eqx.filter_jit
    def step_jit(s, consts, params):
        return timestep_body(s, consts, params, DTIME)

    s1 = fake_atm(state0, hour=10)
    t_jit = time_block("JIT     single step",
                       lambda: step_jit(s1, consts, params))

    # -- 3. Scan over 24 hours (1 day) --
    @eqx.filter_jit
    def scan_24h(s, consts, params):
        def body(state, h):
            state = fake_atm(state, hour=h.item() if hasattr(h, "item") else int(h))
            return timestep_body(state, consts, params, DTIME), None
        hrs = jnp.arange(24)
        # cannot easily put int() in jit; refactor:
        def body2(state, h):
            n = state.atm.ta.shape[0]
            # diurnal as jnp functions
            ta    = jnp.full((n,), 285.0 + 5.0 * jnp.sin(h * 2 * jnp.pi / 24))
            cz    = jnp.maximum(0.0, jnp.sin(h * 2 * jnp.pi / 24 - jnp.pi / 2))
            cz_v  = jnp.full((n,), cz)
            nband = state.atm.solad.shape[1]
            solad = jnp.full((n, nband), 200.0 * cz)
            solai = jnp.full((n, nband), 100.0 * cz)
            new_atm = eqx.tree_at(
                lambda a: (a.ta, a.coszen, a.solad, a.solai),
                state.atm,
                (ta, cz_v, solad, solai),
            )
            state = eqx.tree_at(lambda s: s.atm, state, new_atm)
            return timestep_body(state, consts, params, DTIME), None
        final, _ = jax.lax.scan(body2, s, hrs)
        return final

    t_24 = time_block("JIT+scan 24 hours (1 day)",
                      lambda: scan_24h(state0, consts, params))

    # -- 4. Scan over 720 hours (1 month) --
    @eqx.filter_jit
    def scan_720h(s, consts, params):
        def body2(state, h):
            n = state.atm.ta.shape[0]
            ta    = jnp.full((n,), 285.0 + 5.0 * jnp.sin(h * 2 * jnp.pi / 24))
            cz    = jnp.maximum(0.0, jnp.sin(h * 2 * jnp.pi / 24 - jnp.pi / 2))
            cz_v  = jnp.full((n,), cz)
            nband = state.atm.solad.shape[1]
            solad = jnp.full((n, nband), 200.0 * cz)
            solai = jnp.full((n, nband), 100.0 * cz)
            new_atm = eqx.tree_at(
                lambda a: (a.ta, a.coszen, a.solad, a.solai),
                state.atm,
                (ta, cz_v, solad, solai),
            )
            state = eqx.tree_at(lambda s: s.atm, state, new_atm)
            return timestep_body(state, consts, params, DTIME), None
        hrs = jnp.arange(720)
        final, _ = jax.lax.scan(body2, s, hrs)
        return final

    t_720 = time_block("JIT+scan 720 hours (1 month)",
                       lambda: scan_720h(state0, consts, params))

    print()
    print("=" * 80)
    print("Effective per-timestep cost (lower is better)")
    print("=" * 80)
    print(f"  no-JIT  single step:     {t_nojit*1000:8.2f} ms/step")
    print(f"  JIT     single step:     {t_jit*1000:8.2f} ms/step "
          f"({t_nojit/t_jit:.1f}x vs no-JIT)")
    print(f"  JIT+scan 24h:            {t_24*1000/24:8.2f} ms/step "
          f"({t_nojit/(t_24/24):.1f}x vs no-JIT)")
    print(f"  JIT+scan 720h:           {t_720*1000/720:8.2f} ms/step "
          f"({t_nojit/(t_720/720):.1f}x vs no-JIT)")
    print()

    # Project the 50-year cost
    steps_per_year_per_cell = 8760
    n_cells = 117
    fastest = min(t_jit, t_24/24, t_720/720)
    cells_per_run = state0.atm.ta.shape[0]
    # if we still need 24 separate batches (no cross-cell vmap):
    total_seconds_one_batch_year = fastest * steps_per_year_per_cell
    print(f"Projected 50-yr run on this hardware ({jax.default_backend()}, npoi={cells_per_run}):")
    print(f"  fastest config: {fastest*1000:.2f} ms/step")
    print(f"  per-batch 1 yr:  {total_seconds_one_batch_year/60:.1f} min")
    print(f"  per-batch 50 yr: {total_seconds_one_batch_year*50/3600:.2f} hr")
    print(f"  24 batches in parallel on CPU now: ~70 min/yr/batch")


if __name__ == "__main__":
    main()
