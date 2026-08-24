"""Burned area and fire trace-gas emissions (docs/FIRE.md stage F5).

Two things this stage delivers, both absent from IBIS before it:

**Burned area.** IBIS reports `disturbf`, a disturbance fraction with no
observational counterpart -- nothing in the world can confirm or contradict it.
Reporting burned fraction makes the module falsifiable against GFED5 and
FireCCI51, and it is the CMIP7 priority-1 fire variable (`burntFractionAll`).

**Emissions.** Trace gases from the carbon actually consumed, using Andreae
(2019) Table 1 emission factors -- verified from the published spreadsheet, not
from a summary. See `docs/FIRE.md` §4.4e and
`inputs/fire/emission_factors/andreae2019_ef.csv`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ibis.physics.fire import CC_LEAF, CC_LITTER, CC_WOOD

# ---------------------------------------------------------------------------
# Emission factors -- Andreae (2019) Table 1, g species per kg dry matter
# ---------------------------------------------------------------------------
# Source of record: inputs/fire/emission_factors/andreae2019_ef.csv, extracted
# from acp-19-8523-2019-t01.xlsx. `tests/test_fire_emissions.py` cross-checks
# these constants against that CSV when it is present, so code and data cannot
# drift apart silently.
#
# Only the four *vegetation* categories are used. The table also carries `peat`
# and `agricultural residue`, but IBIS represents neither peatland nor cropland,
# so those columns have no PFT to map onto (docs/FIRE.md §4.4e). Note also that
# N2O for peat is ABSENT from the source -- another reason not to reach for it.

#: Species order used throughout this module.
SPECIES: tuple[str, ...] = ("CO2", "CO", "CH4", "N2O", "NOx")

#: (4 categories x 5 species), g species per kg dry matter.
#: Rows: savanna/grassland, tropical forest, temperate forest, boreal forest.
EF_TABLE = jnp.asarray([
    [1660.0,  69.0, 2.7, 0.17, 2.50],   # savanna and grassland
    [1620.0, 104.0, 6.5, 0.20, 2.80],   # tropical forest
    [1570.0, 113.0, 5.2, 0.25, 3.00],   # temperate forest
    [1530.0, 121.0, 5.5, 0.24, 1.18],   # boreal forest
])

EF_CATEGORIES: tuple[str, ...] = (
    "savanna_grassland", "tropical_forest", "temperate_forest", "boreal_forest",
)

#: IBIS PFT (params.veg:27-38) -> row of EF_TABLE.
#:   0,1  tropical trees          -> tropical forest
#:   2-4  warm/temperate trees    -> temperate forest
#:   5-7  boreal trees            -> boreal forest
#:   8-11 shrubs and grasses      -> savanna and grassland
PFT_TO_EF_CATEGORY: tuple[int, ...] = (1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0)

#: Carbon fraction of dry matter (kgC per kg DM). Andreae's factors are per kg
#: of dry matter, so combusted carbon must be converted before applying them.
C_FRACTION_OF_DM: float = 0.45


class FireEmissionParams(eqx.Module):
    """Combustion completeness by pool. All are F6 calibration targets.

    The values come from :mod:`ibis.physics.fire` rather than being restated
    here. They used to be restated, and the carbon balance in
    `vegetation_dynamics` used none of them -- it emitted 100 % of killed
    biomass while this module reported 20 % of killed wood as burned. One
    definition, imported in both places, is what stops that recurring.
    """

    #: Fraction of killed leaf carbon that is combusted rather than left as litter.
    cc_leaf: float = CC_LEAF
    #: Wood combusts far less completely than foliage.
    cc_wood: float = CC_WOOD
    #: Above-ground litter.
    cc_litter: float = CC_LITTER


class FireEmissions(eqx.Module):
    """Annual fire diagnostics for one grid cell vector.

    Units are declared once, here, and not re-derived anywhere else -- the
    quarantined prototype carried three conflicting conventions for the same
    number.
    """

    burned_fraction: Float[Array, "npoi"]   # 0-1, CMIP7 `burntFractionAll`
    carbon_burned: Float[Array, "npoi"]     # kgC m-2 yr-1, combusted only
    co2: Float[Array, "npoi"]               # g CO2  m-2 yr-1
    co: Float[Array, "npoi"]                # g CO   m-2 yr-1
    ch4: Float[Array, "npoi"]               # g CH4  m-2 yr-1
    n2o: Float[Array, "npoi"]               # g N2O  m-2 yr-1
    nox: Float[Array, "npoi"]               # g NO   m-2 yr-1


def cell_emission_factors(plai: Float[Array, "npoi npft"]) -> Float[Array, "npoi 5"]:
    """LAI-weighted mean emission factor per grid cell.

    A cell holds several PFTs mapping to different Andreae categories, and its
    litter is a mixture of all of them. Rather than attribute litter to a
    dominant PFT -- an arbitrary choice -- the cell takes one blended factor,
    weighted by each PFT's leaf area.

    Cells with no vegetation fall back to savanna/grassland, which is
    immaterial: they have no fuel, so emissions are zero regardless.
    """
    ef_by_pft = EF_TABLE[jnp.asarray(PFT_TO_EF_CATEGORY)]        # (npft, 5)
    w = jnp.maximum(plai, 0.0)                                   # (npoi, npft)
    wsum = jnp.sum(w, axis=1, keepdims=True)
    blended = (w @ ef_by_pft) / jnp.maximum(wsum, 1e-12)
    return jnp.where(wsum > 0.0, blended, EF_TABLE[0][None, :])


def fire_emissions(
    burned_fraction: Float[Array, "npoi"],
    cbiol: Float[Array, "npoi npft"],
    cbiow: Float[Array, "npoi npft"],
    totalit: Float[Array, "npoi"],
    plai: Float[Array, "npoi npft"],
    mort: Float[Array, "npft"],
    params: FireEmissionParams | None = None,
) -> FireEmissions:
    """Trace-gas emissions from the carbon fire actually consumes.

    Combusted carbon is leaf plus wood plus above-ground litter, each scaled by
    its combustion completeness and by the burned fraction. Woody pools are
    additionally scaled by the per-PFT mortality from F4.

    **Fine roots are excluded.** `vegetation.f:433` adds `cbior` to `cdisturb`
    and `:481` subtracts the total from NEE, so IBIS has emitted fine-root
    carbon to the atmosphere for 26 years -- with an unresolved author TODO from
    2000/04/22 sitting directly above the code asking whether some should go to
    litter instead. Roots do not combust; they are excluded here. Note this
    changes only the *emission* diagnostic, not the legacy carbon balance, which
    stays behind `fire_scheme` (docs/FIRE.md §2.3).

    Parameters
    ----------
    burned_fraction : (npoi,)
        Fire-disturbed fraction of the cell this year, i.e. `disturbf` from the
        fire scheme -- **not** including background disturbance `disturbo`,
        which is not fire.
    cbiol, cbiow : (npoi, npft)
        Live leaf and wood carbon, kgC m-2. Already per grid-cell area, so no
        `frac` weighting (docs/FIRE.md §4.4d).
    totalit : (npoi,)
        Above-ground litter carbon, kgC m-2. Not `totlit`, which includes roots.
    plai : (npoi, npft)
        Per-PFT leaf area, used to blend emission factors.
    mort : (npft,)
        Per-PFT fire mortality from F4.
    """
    p = params or FireEmissionParams()

    live = jnp.sum(
        (p.cc_leaf * cbiol + p.cc_wood * cbiow) * mort[None, :], axis=1
    )
    litter = p.cc_litter * totalit
    carbon_burned = burned_fraction * (live + litter)          # kgC m-2 yr-1

    # kgC -> kg dry matter -> g species, via Andreae's per-kg-DM factors.
    dm = carbon_burned / C_FRACTION_OF_DM                      # kg DM m-2 yr-1
    ef = cell_emission_factors(plai)                           # g species / kg DM
    e = dm[:, None] * ef                                       # g species m-2 yr-1

    return FireEmissions(
        burned_fraction=burned_fraction,
        carbon_burned=carbon_burned,
        co2=e[:, 0], co=e[:, 1], ch4=e[:, 2], n2o=e[:, 3], nox=e[:, 4],
    )


def create_fire_emissions(npoi: int) -> FireEmissions:
    """Zero-initialised diagnostics."""
    z = jnp.zeros((npoi,))
    return FireEmissions(burned_fraction=z, carbon_burned=z,
                         co2=z, co=z, ch4=z, n2o=z, nox=z)


def fire_emissions_from_state(
    veg,
    carbon,
    mort: Float[Array, "npft"] | None = None,
    params: FireEmissionParams | None = None,
) -> FireEmissions:
    """Convenience wrapper: emissions from post-`dynaveg` model state.

    `dynaveg` writes the fire disturbance fraction back to `veg.disturbf`, so
    emissions are a pure diagnostic of state -- nothing needs to be threaded
    through the timestep.

    Exists mainly so the correct wiring is obvious at the call site. The
    module's own history is the argument for that: defect 0 was a
    bit-exact-verified function that nothing ever fed, and the quarantined
    prototype was a whole file that nothing ever called.
    """
    from ibis.physics.fire import DEFAULT_PFT_MORTALITY

    if mort is None:
        mort = jnp.asarray(DEFAULT_PFT_MORTALITY)
    return fire_emissions(
        burned_fraction=veg.disturbf,
        cbiol=veg.cbiol, cbiow=veg.cbiow,
        totalit=carbon.totalit, plai=veg.plai,
        mort=mort, params=params,
    )
