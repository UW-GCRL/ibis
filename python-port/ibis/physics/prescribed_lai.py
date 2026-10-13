"""Prescribe leaf area index from observations instead of growing it.

This is the configuration behind the published, validated IBIS GPP result.
Liu et al. (2014, PLoS ONE 9(11) e110407), whose IBIS gives 107.5 PgC/yr
globally, state in their methods:

    "To decrease model uncertainty, we used the satellite-based leaf area
     index (LAI) from the Moderate Resolution Imaging Spectroradiometer
     (MODIS) as a model input."

IBIS never computed LAI in that study. Our prognostic 300-year spin-up
equilibrates 2.7x above the same MODIS field (docs/GPP.md), and GPP follows at
1.6-1.8x. Prescribing LAI therefore reproduces the published configuration and
isolates how much of our carbon-cycle bias is leaf area rather than physiology.

**What is prescribed and what is not.** Only the FULL-LEAF leaf area index is
replaced. IBIS carries seasonality in `frac` (`frac = plai * dropu / avglaiu`,
`vegetation.f:150`), not in `plai`, so the target must be an annual MAXIMUM and
the model's own phenology still modulates it. Prescribing a monthly LAI here
would apply leaf drop twice.

**What this breaks, deliberately.** `plai = cbiol * specla` is kept exact by
back-solving `cbiol`, so the leaf pool stays consistent with the leaf area. But
that leaf carbon no longer comes from `aleaf * aynpp * tauleaf` -- the leaf pool
is decoupled from allocation, which is the point of prescribing it. Carbon
totals from a prescribed-LAI run are therefore diagnostic, not a closed budget,
and must not be compared against a prognostic run's biomass or soil carbon.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


def apply_prescribed_lai(
    state,
    lai_target: Float[Array, "npoi"],
    specla: Float[Array, "npft"],
    epsilon: float = 1.0e-12,
):
    """Rescale per-PFT LAI so the grid-cell total matches `lai_target`.

    The model's own PFT composition is preserved -- only the total is set --
    so which PFTs occupy a cell remains a model result rather than an input.
    Cells where the model carries no leaf area at all are left untouched:
    there is no basis for distributing an observed LAI across PFTs there, and
    inventing one would put vegetation where the model says none exists.
    """
    veg = state.vegetation
    plai = veg.plai
    tot = jnp.sum(plai, axis=1)

    have = tot > epsilon
    scale = jnp.where(have, lai_target / jnp.maximum(tot, epsilon), 1.0)
    scale = jnp.maximum(scale, 0.0)

    plai_new = plai * scale[:, None]
    # Keep plai = cbiol * specla exact by back-solving the leaf pool.
    cbiol_new = plai_new / jnp.maximum(specla if jnp.ndim(specla) == 2 else specla[None, :], epsilon)

    totlaiu = jnp.sum(plai_new[:, :8], axis=1)
    totlail = jnp.sum(plai_new[:, 8:], axis=1)

    # fu is set by WOOD biomass and is unaffected by a leaf-area change; fl
    # follows the lower-canopy leaf area, exactly as dynaveg computes it.
    fl = jnp.maximum(0.25, jnp.minimum(0.975, totlail))
    lai = veg.lai
    lai = lai.at[:, 1].set(totlaiu / jnp.maximum(veg.fu, epsilon))
    lai = lai.at[:, 0].set(totlail / jnp.maximum(fl, epsilon))

    return eqx.tree_at(
        lambda s: (s.vegetation.plai, s.vegetation.cbiol,
                   s.vegetation.lai, s.vegetation.fl),
        state, (plai_new, cbiol_new, lai, fl),
    )
