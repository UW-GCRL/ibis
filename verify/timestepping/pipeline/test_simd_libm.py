"""Definitive test: does JAX vectorize transcendentals (SIMD libm)?

Hypothesis: XLA-CPU uses a vectorized exp/log/sqrt implementation for
length-N arrays (polynomial approximation, last-bit-imprecise vs scalar
glibc). Scalar JAX (one element at a time) uses different code path.

Test:
  1. Build a length-5 array of typical turcof inputs (ta, qa, etc.)
  2. Compute jnp.exp(arr) — batched
  3. Compute [jnp.exp(jnp.array(x)) for x in arr] — scalar JAX
  4. Compute [math.exp(x) for x in arr] — CPython scalar libm
  5. ULP-compare all three

If batched differs from scalar JAX by even 1 ULP → SIMD libm confirmed.
"""
from __future__ import annotations
import math
import struct
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def ulps(a, b):
    if math.isnan(a) or math.isnan(b): return float('nan')
    if a == b: return 0
    ai = struct.unpack('q', struct.pack('d', a))[0]
    bi = struct.unpack('q', struct.pack('d', b))[0]
    if ai < 0: ai = -ai + (1 << 63)
    if bi < 0: bi = -bi + (1 << 63)
    return abs(ai - bi)


def main():
    # Realistic turcof inputs: 5 cells with the GSWP3 forcing values
    # All 5 cells in our test are identical (target cell × 5), so same input
    # But we also try varied inputs to see batch behavior
    print("=== Test 1: identical 5 elements (matches our diagnostic) ===")
    ta = jnp.array([280.873, 280.873, 280.873, 280.873, 280.873])
    print(f"  Input: ta = {ta}")

    batched = jnp.exp(ta * 0.001)
    scalar_jax = jnp.array([float(jnp.exp(jnp.array(float(x) * 0.001))) for x in ta])
    scalar_math = jnp.array([math.exp(float(x) * 0.001) for x in ta])

    print(f"  Batched exp:    {batched[0]:.15e}")
    print(f"  Scalar JAX exp: {scalar_jax[0]:.15e}")
    print(f"  CPython math:   {scalar_math[0]:.15e}")
    print(f"  ULPs batched vs scalar JAX:   {ulps(float(batched[0]), float(scalar_jax[0]))}")
    print(f"  ULPs batched vs CPython math: {ulps(float(batched[0]), float(scalar_math[0]))}")
    print(f"  ULPs scalar JAX vs math:      {ulps(float(scalar_jax[0]), float(scalar_math[0]))}")

    print("\n=== Test 2: varied 5 elements ===")
    arr = jnp.array([280.873, 300.0, 250.0, 273.16, 286.83])
    print(f"  Inputs: {arr.tolist()}")

    # Test exp
    batched = jnp.exp(arr * 0.001)
    scalar_jax = jnp.array([float(jnp.exp(jnp.array(float(x) * 0.001))) for x in arr])
    scalar_math = jnp.array([math.exp(float(x) * 0.001) for x in arr])

    print(f"\n  exp(x * 0.001):")
    print(f"  {'idx':>4s} {'batched':>22s} {'scalar JAX':>22s} {'CPython math':>22s}  {'b-s':>4s} {'b-m':>4s}")
    for i in range(5):
        b = float(batched[i]); s = float(scalar_jax[i]); m = float(scalar_math[i])
        print(f"  {i:>4d} {b:>22.15e} {s:>22.15e} {m:>22.15e}  {ulps(b,s):>4d} {ulps(b,m):>4d}")

    # Test log
    print(f"\n  log(x):")
    batched_log = jnp.log(arr)
    scalar_jax_log = jnp.array([float(jnp.log(jnp.array(float(x)))) for x in arr])
    scalar_math_log = jnp.array([math.log(float(x)) for x in arr])
    print(f"  {'idx':>4s} {'batched':>22s} {'scalar JAX':>22s} {'CPython math':>22s}  {'b-s':>4s} {'b-m':>4s}")
    for i in range(5):
        b = float(batched_log[i]); s = float(scalar_jax_log[i]); m = float(scalar_math_log[i])
        print(f"  {i:>4d} {b:>22.15e} {s:>22.15e} {m:>22.15e}  {ulps(b,s):>4d} {ulps(b,m):>4d}")

    # Test sqrt
    print(f"\n  sqrt(x):")
    batched_sq = jnp.sqrt(arr)
    scalar_jax_sq = jnp.array([float(jnp.sqrt(jnp.array(float(x)))) for x in arr])
    scalar_math_sq = jnp.array([math.sqrt(float(x)) for x in arr])
    print(f"  {'idx':>4s} {'batched':>22s} {'scalar JAX':>22s} {'CPython math':>22s}  {'b-s':>4s} {'b-m':>4s}")
    for i in range(5):
        b = float(batched_sq[i]); s = float(scalar_jax_sq[i]); m = float(scalar_math_sq[i])
        print(f"  {i:>4d} {b:>22.15e} {s:>22.15e} {m:>22.15e}  {ulps(b,s):>4d} {ulps(b,m):>4d}")

    print("\n=== Test 3: numpy (which also uses SIMD libm) ===")
    arr_np = np.array([280.873, 300.0, 250.0, 273.16, 286.83])
    np_exp = np.exp(arr_np * 0.001)
    print(f"  numpy.exp(arr * 0.001):")
    print(f"  {'idx':>4s} {'numpy':>22s} {'CPython math':>22s}  {'ulps':>5s}")
    for i in range(5):
        n = float(np_exp[i]); m = math.exp(float(arr_np[i]) * 0.001)
        print(f"  {i:>4d} {n:>22.15e} {m:>22.15e}  {ulps(n,m):>5d}")

    print("\n=== Test 4: XLA HLO output for exp (debug) ===")
    @jax.jit
    def f(x):
        return jnp.exp(x)
    hlo = jax.jit(f).lower(arr).compile().as_text() if hasattr(jax.jit(f).lower(arr).compile(), 'as_text') else "(unavailable)"
    print(hlo[:500] if isinstance(hlo, str) else "(unavailable)")


if __name__ == "__main__":
    main()
