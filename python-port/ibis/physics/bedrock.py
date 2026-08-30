"""Thermal-only bedrock layers beneath the IBIS soil column.

IBIS 2.6b4 solves heat on 6 soil layers totalling 4.0 m with a zero-flux base.
That boundary is too shallow: a zero-flux base reflects the annual temperature
wave, so the deepest soil node carries a large amplitude error (+53% in the
Arctic against a finely-resolved reference).  The error is confined to the
deepest node -- the active layer, and hence methane, is barely affected -- but
it makes deep soil temperature and interannual thermal memory unreliable.

The fix used by essentially every land-surface model that faced this is a
hydrologically-inactive thermal slab below the soil (CLM4+, JULES `l_bedrock`,
LPJmL4, LPJ-GUESS, CLASSIC).  No water, carbon or roots exist there; only heat
conducts.  That keeps the cost to a slightly taller tridiagonal solve and adds
no new spin-up burden.

Layer thicknesses matter more than total depth.  LPJmL4's single 10 m slab
reaches 14 m but leaves +28% error at our deepest soil node, because the
1 m -> 10 m jump distorts the flux at that interface.  Three layers of
(2, 2, 4) m reach only 12 m yet cut the error to 1.5% (Arctic) and 0.3%
(ice-rich, doubled diffusivity), by keeping every thickness ratio at or below
two.  Extending past 12 m buys nothing measurable.

Bedrock properties follow CLM: 3.0 W/m/K and 2.0e6 J/m3/K.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array

from ibis.physics.soil_physics import soilheat, SoilheatResult


class BedrockParams(eqx.Module):
    """Thermal-only layers below the soil column."""

    #: Layer thicknesses, m. With IBIS's 4.0 m soil column this gives 12 m.
    hsoi: tuple = (2.0, 2.0, 4.0)
    #: Thermal conductivity (W/m/K). CLM `thk_bedrock`.
    consoi: float = 3.0
    #: Volumetric heat capacity (J/m3/K). CLM `csol_bedrock`, split into a
    #: granite density and specific heat so it enters soilheat's (1-poros)*c*rho.
    rhosoi: float = 2700.0
    csoi: float = 740.74074074074076   # 2700 * 740.74... = 2.0e6
    #: Gradient used only to initialise new layers, K/m. Default 0.0.
    #:
    #: Under a zero-flux base the steady state of the diffusion equation is
    #: ISOTHERMAL: d2T/dz2 = 0 makes T linear, and zero gradient at the base
    #: forces the slope to zero. So isothermal continuation -- not a geothermal
    #: gradient -- is the equilibrium seed for OUR boundary condition, and a
    #: seeded gradient would simply decay away over the diffusive timescale.
    #:
    #: The geothermal-gradient form of downward continuation belongs to models
    #: that impose a basal heat flux (CryoGrid, GIPL, UVic). We deliberately do
    #: not: prescribing one at shallow depth injects spurious heat. Set this
    #: non-zero only alongside such a flux.
    dtdz: float = 0.0

    @property
    def nlayer(self) -> int:
        return len(self.hsoi)

    @property
    def volumetric_heat_capacity(self) -> float:
        return self.rhosoi * self.csoi


def initialize_bedrock_temperature(
    tsoi_bottom: Float[Array, "npoi"],
    hsoi_soil: Float[Array, "nsoilay"],
    params: BedrockParams,
) -> Float[Array, "npoi nbedrock"]:
    """Seed bedrock temperatures by downward continuation.

    New nodes are continued downward from the deepest soil node. With the
    default `dtdz = 0.0` this is isothermal, which IS the equilibrium profile
    under a zero-flux base, so the column starts settled rather than spending
    decades relaxing a seeded gradient away. See `BedrockParams.dtdz`.
    """
    h = jnp.asarray(params.hsoi, dtype=tsoi_bottom.dtype)
    # Depth of each bedrock node below the centre of the deepest soil layer.
    soil_bottom_offset = 0.5 * jnp.asarray(hsoi_soil, dtype=h.dtype)[-1]
    depth = soil_bottom_offset + jnp.cumsum(h) - 0.5 * h
    return tsoi_bottom[:, None] + params.dtdz * depth[None, :]


def soilheat_with_bedrock(
    tsoi: Float[Array, "npoi nsoilay"],
    tsoi_deep: Float[Array, "npoi nbedrock"],
    otsoi: Float[Array, "npoi nsoilay"],
    otsoi_deep: Float[Array, "npoi nbedrock"],
    owsoi: Float[Array, "npoi nsoilay"],
    wsoi: Float[Array, "npoi nsoilay"],
    wisoi: Float[Array, "npoi nsoilay"],
    poros: Float[Array, "npoi nsoilay"],
    consoi: Float[Array, "npoi nsoilay"],
    csoi_mineral: Float[Array, "npoi nsoilay"],
    rhosoi: Float[Array, "npoi nsoilay"],
    hsoi: Float[Array, "nsoilay"],
    c0pud: Float[Array, "npoi nsoilay"],
    c1pud: Float[Array, "npoi nsoilay"],
    fhtop: Float[Array, "npoi"],
    wflo: Float[Array, "npoi nsoilay_p1"],
    ch2o: float,
    cice: float,
    rhow: float,
    dtime: float,
    params: BedrockParams,
):
    """Solve soil heat conduction with thermal-only layers appended.

    Returns `(SoilheatResult, tsoi_deep)`. The returned SoilheatResult is
    trimmed back to the soil layers, so callers see exactly the arrays they saw
    before; `hflo` at the soil base is now the real conductive flux into
    bedrock rather than the zero imposed by the old boundary.
    """
    npoi, nsoilay = tsoi.shape
    nbed = params.nlayer
    if tsoi_deep.shape[1] != nbed:
        raise ValueError(
            f"state carries {tsoi_deep.shape[1]} bedrock layers but "
            f"BedrockParams defines {nbed} ({params.hsoi}). Set GridConfig."
            f"nbedrock and SimParams.bedrock consistently.")
    dt = tsoi.dtype
    z = jnp.zeros((npoi, nbed), dtype=dt)
    cat = lambda a, b: jnp.concatenate([a, b], axis=1)

    hsoi_ext = jnp.concatenate(
        [jnp.asarray(hsoi, dtype=dt), jnp.asarray(params.hsoi, dtype=dt)])

    res = soilheat(
        tsoi=cat(tsoi, tsoi_deep),
        otsoi=cat(otsoi, otsoi_deep),
        # Bedrock holds no water: zero moisture and zero porosity make its heat
        # capacity fall through to (1-0) * csoi * rhosoi = 2.0e6 J/m3/K.
        owsoi=cat(owsoi, z), wsoi=cat(wsoi, z), wisoi=cat(wisoi, z),
        poros=cat(poros, z),
        consoi=cat(consoi, jnp.full((npoi, nbed), params.consoi, dtype=dt)),
        csoi_mineral=cat(csoi_mineral, jnp.full((npoi, nbed), params.csoi, dtype=dt)),
        rhosoi=cat(rhosoi, jnp.full((npoi, nbed), params.rhosoi, dtype=dt)),
        hsoi=hsoi_ext,
        c0pud=cat(c0pud, z), c1pud=cat(c1pud, z),
        fhtop=fhtop,
        # wflo is defined on interfaces; the bedrock ones carry no water. The
        # drainage sink at the soil base is preserved by soilheat's n_hydro.
        wflo=jnp.concatenate([wflo, jnp.zeros((npoi, nbed), dtype=dt)], axis=1),
        ch2o=ch2o, cice=cice, rhow=rhow, dtime=dtime,
        n_hydro=nsoilay,
    )
    return (SoilheatResult(tsoi=res.tsoi[:, :nsoilay],
                           hflo=res.hflo[:, :nsoilay + 1]),
            res.tsoi[:, nsoilay:])
