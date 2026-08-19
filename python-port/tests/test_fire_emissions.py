"""Stage F5: burned area and trace-gas emissions.

Before F5, IBIS reported `disturbf` -- a disturbance fraction with no
observational counterpart, so no version of the fire module could be confirmed
or contradicted by anything. F5 reports burned fraction (CMIP7
`burntFractionAll`) and the trace gases from the carbon fire actually consumes.
"""

import csv
import os

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.bgc.fire_emissions import (
    C_FRACTION_OF_DM, EF_CATEGORIES, EF_TABLE, PFT_TO_EF_CATEGORY, SPECIES,
    FireEmissionParams, cell_emission_factors, create_fire_emissions,
    fire_emissions,
)
from ibis.physics.fire import DEFAULT_PFT_MORTALITY

EPS = float(np.finfo(np.float64).eps)
NPOI, NPFT = 4, 12
MORT = jnp.asarray(DEFAULT_PFT_MORTALITY)

EF_CSV = ("/shared/nas2/Hangkai/ibis/inputs/fire/emission_factors/"
          "andreae2019_ef.csv")


# ---------------------------------------------------------------------------
# The constants must match the published table
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(EF_CSV), reason="EF csv not present")
def test_ef_constants_match_the_published_spreadsheet():
    """Cross-check the hard-coded factors against Andreae (2019) Table 1.

    The constants are embedded for speed, but the CSV extracted from the
    published spreadsheet is the source of record. This test is what stops the
    two drifting apart silently.
    """
    want = {}
    with open(EF_CSV) as f:
        for row in csv.DictReader(f):
            if row["ef_g_per_kg_dm"]:
                want[(row["species"], row["category"])] = float(row["ef_g_per_kg_dm"])

    csv_name = {"CO2": "CO2", "CO": "CO", "CH4": "CH4", "N2O": "N2O",
                "NOx": "NOx (as NO)"}
    for i, cat in enumerate(EF_CATEGORIES):
        for j, sp in enumerate(SPECIES):
            expected = want[(csv_name[sp], cat)]
            assert float(EF_TABLE[i, j]) == pytest.approx(expected, rel=1e-12), (
                f"{sp}/{cat}: constant {float(EF_TABLE[i, j])} != "
                f"published {expected}")


def test_only_vegetation_categories_are_used():
    """Peat and cropland are in the source table but unusable here.

    IBIS represents neither peatland nor cropland, so those columns have no PFT
    to map onto -- and N2O for peat is absent from the source entirely.
    """
    assert EF_CATEGORIES == ("savanna_grassland", "tropical_forest",
                             "temperate_forest", "boreal_forest")
    assert EF_TABLE.shape == (4, len(SPECIES))
    assert "peat" not in EF_CATEGORIES and "agricultural_residue" not in EF_CATEGORIES


def test_pft_mapping_covers_every_pft_and_is_in_range():
    assert len(PFT_TO_EF_CATEGORY) == NPFT
    assert set(PFT_TO_EF_CATEGORY) <= set(range(len(EF_CATEGORIES)))
    # tropical trees -> tropical, boreal trees -> boreal, grass -> savanna
    assert PFT_TO_EF_CATEGORY[0] == EF_CATEGORIES.index("tropical_forest")
    assert PFT_TO_EF_CATEGORY[5] == EF_CATEGORIES.index("boreal_forest")
    assert PFT_TO_EF_CATEGORY[10] == EF_CATEGORIES.index("savanna_grassland")
    assert PFT_TO_EF_CATEGORY[3] == EF_CATEGORIES.index("temperate_forest")


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def test_single_pft_cell_gets_that_category_exactly():
    for pft, cat in ((0, 1), (5, 3), (10, 0), (3, 2)):
        plai = jnp.zeros((1, NPFT)).at[0, pft].set(4.0)
        got = np.asarray(cell_emission_factors(plai))[0]
        assert np.allclose(got, np.asarray(EF_TABLE[cat]), atol=8 * EPS)


def test_mixed_cell_blends_between_its_categories():
    plai = jnp.zeros((1, NPFT)).at[0, 0].set(1.0).at[0, 10].set(1.0)
    got = np.asarray(cell_emission_factors(plai))[0]
    trop = np.asarray(EF_TABLE[1]); sav = np.asarray(EF_TABLE[0])
    assert np.allclose(got, 0.5 * (trop + sav), atol=8 * EPS)


def test_bare_cell_does_not_produce_nan():
    got = np.asarray(cell_emission_factors(jnp.zeros((2, NPFT))))
    assert np.all(np.isfinite(got))


# ---------------------------------------------------------------------------
# What burns, and what does not
# ---------------------------------------------------------------------------

def _call(**kw):
    base = dict(
        burned_fraction=jnp.full((NPOI,), 0.25),
        cbiol=jnp.full((NPOI, NPFT), 0.04),
        cbiow=jnp.full((NPOI, NPFT), 0.40),
        totalit=jnp.full((NPOI,), 0.30),
        plai=jnp.zeros((NPOI, NPFT)).at[:, 10].set(2.0),
        mort=MORT,
    )
    base.update(kw)
    return fire_emissions(**base)


def test_fine_roots_are_not_combusted():
    """Roots do not burn.

    `vegetation.f:433` adds `cbior` to `cdisturb` and `:481` subtracts it from
    NEE, so IBIS has emitted fine-root carbon for 26 years -- with an author
    TODO from 2000 questioning it. The emission diagnostic excludes roots, and
    the signature makes that structural: there is no `cbior` argument to pass.
    """
    import inspect
    assert "cbior" not in inspect.signature(fire_emissions).parameters


def test_zero_burned_area_gives_zero_emissions():
    e = _call(burned_fraction=jnp.zeros(NPOI))
    for v in (e.carbon_burned, e.co2, e.co, e.ch4, e.n2o, e.nox):
        assert np.allclose(np.asarray(v), 0.0, atol=8 * EPS)


def test_emissions_scale_linearly_with_burned_area():
    a = _call(burned_fraction=jnp.full((NPOI,), 0.10))
    b = _call(burned_fraction=jnp.full((NPOI,), 0.20))
    assert np.allclose(np.asarray(b.ch4), 2.0 * np.asarray(a.ch4), rtol=1e-12)


def test_carbon_to_species_conversion_is_self_consistent():
    """species = (carbon_burned / 0.45) * EF, with one declared unit convention."""
    e = _call()
    plai = jnp.zeros((NPOI, NPFT)).at[:, 10].set(2.0)
    ef = np.asarray(cell_emission_factors(plai))
    dm = np.asarray(e.carbon_burned) / C_FRACTION_OF_DM
    for k, sp in enumerate(SPECIES):
        got = np.asarray(getattr(e, sp.lower() if sp != "NOx" else "nox"))
        assert np.allclose(got, dm * ef[:, k], rtol=1e-12)


def test_species_ratios_follow_the_emission_factors():
    e = _call()
    ratio = np.asarray(e.ch4) / np.asarray(e.co2)
    expected = float(EF_TABLE[0, 2] / EF_TABLE[0, 0])   # savanna cell
    assert np.allclose(ratio, expected, rtol=1e-10)


def test_wood_burns_less_completely_than_leaves():
    p = FireEmissionParams()
    assert p.cc_wood < p.cc_leaf
    lo = _call(cbiow=jnp.full((NPOI, NPFT), 0.40))
    hi = _call(cbiow=jnp.full((NPOI, NPFT), 0.80))
    assert np.all(np.asarray(hi.carbon_burned) > np.asarray(lo.carbon_burned))


def test_resistant_pfts_contribute_less_carbon():
    all_dead = jnp.ones((NPFT,))
    resistant = jnp.full((NPFT,), 0.3)
    assert np.all(np.asarray(_call(mort=resistant).carbon_burned)
                  < np.asarray(_call(mort=all_dead).carbon_burned))


def test_burned_fraction_is_reported_unchanged():
    """The CMIP7 `burntFractionAll` output must be the input fraction itself."""
    bf = jnp.asarray([0.0, 0.13, 0.5, 1.0])
    assert np.array_equal(np.asarray(_call(burned_fraction=bf).burned_fraction),
                          np.asarray(bf))


def test_state_helper_and_zero_constructor():
    z = create_fire_emissions(NPOI)
    assert np.allclose(np.asarray(z.ch4), 0.0)
    assert np.asarray(z.burned_fraction).shape == (NPOI,)


def test_jittable_and_differentiable():
    f = lambda tl: fire_emissions(
        jnp.full((NPOI,), 0.2), jnp.full((NPOI, NPFT), 0.04),
        jnp.full((NPOI, NPFT), 0.4), tl,
        jnp.zeros((NPOI, NPFT)).at[:, 10].set(2.0), MORT).ch4.sum()
    tl = jnp.full((NPOI,), 0.3)
    assert np.isfinite(float(eqx.filter_jit(f)(tl)))
    g = jax.grad(f)(tl)
    assert np.all(np.isfinite(np.asarray(g))) and float(g[0]) > 0.0
