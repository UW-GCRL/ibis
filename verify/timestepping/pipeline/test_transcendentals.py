"""Compare F77 vs JAX transcendental function results bit-by-bit.

Hypothesis: residual divergence in turcof/turvap comes from different
implementations of exp(), log(), sqrt(), pow(x,y) etc. between
gfortran's libm and JAX/XLA's libm.

Test: pass same float64 to F77 wrapper and JAX, compare ULP-level diffs.

F77 doesn't expose individual math functions, but we can build a tiny
test using values that exercise the same exp/log/sqrt paths as turcof.
"""
from __future__ import annotations
import math
import struct
import numpy as np
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


def ulps(a, b):
    """Count ULPs between two float64 values."""
    if math.isnan(a) or math.isnan(b):
        return float('nan')
    if a == b:
        return 0
    # Convert to integer representation
    ai = struct.unpack('q', struct.pack('d', a))[0]
    bi = struct.unpack('q', struct.pack('d', b))[0]
    # Handle sign-magnitude across zero
    if ai < 0:
        ai = -ai + (1 << 63)
    if bi < 0:
        bi = -bi + (1 << 63)
    return abs(ai - bi)


def main():
    # Compare math.X (CPython libm) — likely same as gfortran libm on Linux
    # vs jnp.X (JAX-compiled XLA libm)
    test_vals = [
        0.5, 1.0, 2.71828, 3.14159, 10.0, 100.0,
        273.16, 280.87, 286.83,  # typical Ta values
        0.001, 0.006, 6.37e-3,   # typical qa values
        -0.5, -1.0,
    ]
    funcs = [
        ("exp", math.exp,  lambda x: float(jnp.exp(jnp.array(x)))),
        ("log", math.log,  lambda x: float(jnp.log(jnp.array(x)))),
        ("sqrt", math.sqrt, lambda x: float(jnp.sqrt(jnp.array(x)))),
        ("sin", math.sin,   lambda x: float(jnp.sin(jnp.array(x)))),
        ("cos", math.cos,   lambda x: float(jnp.cos(jnp.array(x)))),
    ]

    print(f"{'func':>6s} {'x':>15s}  {'CPython math':>20s}  {'JAX':>20s}  {'|Δ|':>11s}  {'ULPs':>5s}")
    print("-" * 100)
    any_diff = False
    for fname, f_py, f_jax in funcs:
        for x in test_vals:
            if fname == "log" and x <= 0:
                continue
            if fname == "sqrt" and x < 0:
                continue
            v_py = f_py(x)
            v_jax = f_jax(x)
            d = abs(v_py - v_jax)
            u = ulps(v_py, v_jax)
            flag = " 🔥" if u > 1 else (" ⚠" if u > 0 else "")
            print(f"{fname:>6s} {x:>15.6e}  {v_py:>20.13e}  {v_jax:>20.13e}  {d:>11.3e}  {u:>5d}{flag}")
            if u > 0:
                any_diff = True

    # Composite computations that appear in turcof
    print(f"\n=== Composite ops (turcof-style) ===")
    # Example: rich = grav * (zt - zb) * (zht - zhb) / (0.5 * (zht + zhb) * u**2)
    grav, zt, zb = 9.81, 1.0, 0.5
    zht, zhb = 280.0, 285.0
    u = 4.5

    # Python
    rich_py = grav * (zt - zb) * (zht - zhb) / (0.5 * (zht + zhb) * u**2)
    # JAX
    rich_jax = float(9.81 * (1.0 - 0.5) * (280.0 - 285.0) /
                     (0.5 * (280.0 + 285.0) * jnp.array(4.5)**2))
    print(f"  rich:  py={rich_py:.13e}  jax={rich_jax:.13e}  ulps={ulps(rich_py, rich_jax)}")

    # ym = (vonk/xm)**2 * exp(0.5*xm) * sqrt(-rich)
    vonk, xm = 0.4, 1.0
    rich_test = -0.005
    ym_py = (vonk/xm)**2 * math.exp(0.5*xm) * math.sqrt(-rich_test)
    ym_jax = float((0.4/1.0)**2 * jnp.exp(jnp.array(0.5)) * jnp.sqrt(jnp.array(0.005)))
    print(f"    ym:  py={ym_py:.13e}  jax={ym_jax:.13e}  ulps={ulps(ym_py, ym_jax)}")

    if not any_diff:
        print(f"\n✅ All scalar transcendentals BIT-EXACT (0 ULPs) between CPython libm and JAX/XLA")
    else:
        print(f"\n⚠ Some scalar transcendentals differ — possible source of canopy iter residual")


if __name__ == "__main__":
    main()
