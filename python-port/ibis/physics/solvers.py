"""Linear algebra solvers for IBIS-JAX.

Provides batched tridiagonal solver (Thomas algorithm) matching the
Fortran tridia() subroutine.  The solver is unrolled over layers for
static JIT tracing and supports batched systems (one per grid point).

Functions:
  tridia_batched -- solve npoi independent tridiagonal systems
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float, Array


def tridia_batched(
    a: Float[Array, "npoi nlay"],
    b: Float[Array, "npoi nlay"],
    c: Float[Array, "npoi nlay"],
    rhs: Float[Array, "npoi nlay"],
) -> Float[Array, "npoi nlay"]:
    """Solve npoi independent tridiagonal systems via Thomas algorithm.

    Matches Fortran tridia(ns, nd, ne, a, b, c, y, x, alpha, gamma).

    The tridiagonal system for each grid point j is::

        b[j,0]*x[j,0] + c[j,0]*x[j,1]                           = rhs[j,0]
        a[j,k]*x[j,k-1] + b[j,k]*x[j,k] + c[j,k]*x[j,k+1]     = rhs[j,k]
        a[j,N]*x[j,N-1] + b[j,N]*x[j,N]                         = rhs[j,N]

    Parameters
    ----------
    a : array [npoi, nlay]
        Sub-diagonal (a[:,0] unused).
    b : array [npoi, nlay]
        Main diagonal.
    c : array [npoi, nlay]
        Super-diagonal (c[:,-1] unused).
    rhs : array [npoi, nlay]
        Right-hand side.

    Returns
    -------
    x : array [npoi, nlay]
        Solution vectors.
    """
    nlay = a.shape[1]
    nm1 = nlay - 1

    # Forward sweep: LU decomposition + forward substitution
    # alpha[k] = 1 / (b[k] - a[k] * gamma[k-1])
    # gamma[k] = c[k] * alpha[k]
    # x[k] = (rhs[k] - a[k] * x[k-1]) * alpha[k]

    # Layer 0
    alpha_0 = 1.0 / b[:, 0]
    gamma_prev = c[:, 0] * alpha_0
    x_prev = rhs[:, 0] * alpha_0

    # Collect results in lists (will stack at end)
    gamma_list = [gamma_prev]
    x_list = [x_prev]

    # Layers 1..nlay-2
    for k in range(1, nm1):
        alpha_k = 1.0 / (b[:, k] - a[:, k] * gamma_prev)
        gamma_prev = c[:, k] * alpha_k
        x_prev = (rhs[:, k] - a[:, k] * x_prev) * alpha_k
        gamma_list.append(gamma_prev)
        x_list.append(x_prev)

    # Last layer (nlay-1)
    x_last = ((rhs[:, nm1] - a[:, nm1] * x_prev)
              / (b[:, nm1] - a[:, nm1] * gamma_prev))
    x_list.append(x_last)

    # Back substitution
    for k in range(nm1 - 1, -1, -1):
        x_list[k] = x_list[k] - gamma_list[k] * x_list[k + 1]

    return jnp.stack(x_list, axis=1)
