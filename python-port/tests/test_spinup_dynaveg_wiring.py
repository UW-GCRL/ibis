"""The spin-up runner must apply disturbance, and must apply it exactly once.

`run_spinup_gswp3` called `biomass_update`, whose own docstring says it "does
NOT apply disturbances (fire, background mortality)". Every result from that
path -- including the 50-year GPU carbon spin-up -- therefore had no fire
feedback at all, whatever `isimfire` said.

`scripts/run_global.py` had the opposite defect: it called `biomass_update`
*and* `dynaveg`, which perform the same allocation from the same uncleared
`summary.aynpp`, so each dynamic-vegetation year advanced the pools twice.
"""

from __future__ import annotations

import inspect
import pathlib

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.io.parameters import default_pft_params
from ibis.physics.fire import LEGACY
from ibis.physics.vegetation_dynamics import biomass_update, dynaveg
from ibis.state.carbon import create_carbon_state
from ibis.state.summary import create_summary_state
from ibis.state.vegetation import create_vegetation_state
from ibis.timestepping.loops_gswp3 import (
    climate_stats_from_monthly, run_spinup_gswp3,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _set(o, **kw):
    for k, v in kw.items():
        o = eqx.tree_at(lambda x, _k=k: getattr(x, _k), o, v)
    return o


def _growing_state(n=8, npft=12):
    v = create_vegetation_state(n, npft, 6)
    v = _set(v, cbiol=jnp.full((n, npft), 0.2), cbiow=jnp.full((n, npft), 1.0),
             cbior=jnp.full((n, npft), 0.2), exist=jnp.ones((n, npft)),
             plai=jnp.full((n, npft), 0.3), gdd0=jnp.full((n,), 3000.0),
             gdd5=jnp.full((n,), 2000.0))
    su = _set(create_summary_state(n, npft), aynpp=jnp.full((n, npft), 0.05))
    return v, create_carbon_state(n), su


def test_calling_both_updates_advances_biomass_twice():
    """The defect run_global had. Pins why exactly one may be called."""
    v, c, su = _growing_state()
    pf, e = default_pft_params(), 1e-12

    def total(veg):
        return float(jnp.sum(veg.cbiol + veg.cbiow + veg.cbior))

    start = total(v)
    once, _, _ = dynaveg(v, c, su, pf, 0, e, fire_scheme=LEGACY)
    bv, bc, bs = biomass_update(v, c, su, pf, e)
    twice, _, _ = dynaveg(bv, bc, bs, pf, 0, e, fire_scheme=LEGACY)

    d_once, d_twice = total(once) - start, total(twice) - start
    assert abs(d_once) > 1e-6
    # Two applications of one year's update, not one.
    assert abs(d_twice) > abs(d_once) * 1.2


def test_spinup_runner_offers_dynamic_vegetation_and_defaults_to_it():
    sig = inspect.signature(run_spinup_gswp3).parameters
    assert sig["dynamic_vegetation"].default is True
    assert sig["isimfire"].default == 0        # fire stays the modeller's call


def test_spinup_runner_calls_dynaveg_not_biomass_update():
    src = (ROOT / "ibis" / "timestepping" / "loops_gswp3.py").read_text()
    body = src[src.index("def run_spinup_gswp3"):]
    assert "dynaveg(" in body
    # and maintains the climate normals dynaveg's PFT reclassification needs
    for needed in ("annual_climate_stats", "climanl2", "existence("):
        assert needed in body, needed


def test_run_global_applies_exactly_one_annual_update():
    src = (ROOT / "scripts" / "run_global.py").read_text()
    year_loop = src[src.index("Annual vegetation update"):]
    assert year_loop.count("= dynaveg(") == 1
    assert year_loop.count("= biomass_update(") == 1
    assert "if p.isimveg > 0:" in year_loop     # they are the two branches


def test_climate_stats_are_computed_once_and_shared():
    """run_global had its own copy of the gdd arithmetic."""
    src = (ROOT / "scripts" / "run_global.py").read_text()
    assert "climate_stats_from_monthly" in src
    assert "gdd5this_np +=" not in src         # the inline copy is gone


def test_climate_stats_arithmetic():
    """A flat 10 degC year: gdd0 = 3650, gdd5 = 1825, tc = tw = 10."""
    xint = np.full((4, 12), 10.0, dtype=np.float32)
    tc, tw, gdd0, gdd5 = climate_stats_from_monthly(xint, leap=False)
    assert tc == pytest.approx(np.full(4, 10.0))
    assert tw == pytest.approx(np.full(4, 10.0))
    assert gdd0 == pytest.approx(np.full(4, 3650.0))
    assert gdd5 == pytest.approx(np.full(4, 1825.0))


def test_climate_stats_handle_leap_years():
    xint = np.full((2, 12), 10.0, dtype=np.float32)
    _, _, gdd0_no, _ = climate_stats_from_monthly(xint, leap=False)
    _, _, gdd0_yes, _ = climate_stats_from_monthly(xint, leap=True)
    assert gdd0_yes == pytest.approx(gdd0_no + 10.0)   # one extra day at 10 degC


def test_a_freezing_year_accumulates_no_growing_degree_days():
    xint = np.full((3, 12), -20.0, dtype=np.float32)
    tc, tw, gdd0, gdd5 = climate_stats_from_monthly(xint, leap=False)
    assert gdd0 == pytest.approx(np.zeros(3))
    assert gdd5 == pytest.approx(np.zeros(3))
    assert tc == pytest.approx(np.full(3, -20.0))


# --- climate normals must be seeded, not crawled up to ----------------------


def test_first_year_seeds_normals_instead_of_averaging_from_zero():
    """`climanl2` is a 30-year running mean; from zero it is wrong for decades.

    Not hypothetical: every climate normal in the v7 spin-up checkpoint is
    exactly 0.0, because no runner ever maintained them, which left `exist` at
    1.000 -- all 12 PFTs permitted everywhere, tropical evergreen included in
    Siberia. Seeding fixed `exist` at 0.341 from year one; without it, it
    climbed 0.226 -> 0.335 over three years and was wrong throughout.
    """
    src = (ROOT / "ibis" / "timestepping" / "loops_gswp3.py").read_text()
    body = src[src.index("def run_spinup_gswp3"):]
    assert "climanl(" in body and "climanl2(" in body
    # seeded when the normals are unset, averaged thereafter
    assert "jnp.abs(v.gdd5)" in body


def test_annual_climate_stats_returns_the_monthly_means_for_seeding():
    import inspect

    from ibis.timestepping.loops_gswp3 import annual_climate_stats
    doc = inspect.getdoc(annual_climate_stats)
    assert "xint" in doc          # climanl needs (npoi, 12), not the 4 stats


def test_seeding_and_averaging_agree_on_a_steady_climate():
    """A constant climate: the seed is already the equilibrium of the mean."""
    from ibis.physics.climate_analysis import climanl, climanl2

    n = 5
    xint = jnp.asarray(np.tile(np.linspace(-5.0, 25.0, 12), (n, 1)))
    ndaypm = tuple(int(d) for d in np.array([31, 28, 31, 30, 31, 30, 31, 31,
                                             30, 31, 30, 31]))
    tc, tw, tcmin, gdd0, gdd5 = climanl(xint, jnp.zeros(n), ndaypm)
    tcthis, twthis, gdd0this, gdd5this = climate_stats_from_monthly(
        np.asarray(xint), leap=False)
    # Feeding the same year back through the running mean must not move it.
    out = climanl2(tc, tw, tcmin, gdd0, gdd5,
                   jnp.asarray(tcthis), jnp.asarray(twthis),
                   jnp.asarray(gdd0this), jnp.asarray(gdd5this),
                   jnp.zeros(n))
    for seeded, averaged in zip((tc, tw, tcmin, gdd0, gdd5), out):
        assert np.allclose(np.asarray(seeded), np.asarray(averaged), rtol=1e-3)


def test_all_annual_accumulators_are_reset_each_year():
    """Every ay* field is labelled "/yr"; none may survive into the next year.

    Only the three dynaveg consumes were cleared, so the rest accumulated over
    the whole run -- ayco2mic reached 921.9 kgC/m2 and ayprcp 3.2e6 mm after
    200 years, both written to netCDF as per-year quantities. Guards the whole
    set, so a new accumulator cannot quietly join the unreset ones.
    """
    import re
    from pathlib import Path
    from ibis.state.summary import SummaryState

    # ayanpp is derived, not accumulated: dynaveg builds it from zeros out of
    # aynpp and overwrites it wholesale every year (vegetation_dynamics.py:207,
    # 588-593). Clearing it here would zero the value dynaveg had just written.
    OVERWRITTEN = {"ayanpp"}
    annual = {f for f in SummaryState.__dataclass_fields__
              if f.startswith("ay")} - OVERWRITTEN
    assert annual, "no ay* fields found -- has the naming convention changed?"

    loops = Path(
        __import__("ibis.timestepping.loops_gswp3", fromlist=["x"]).__file__
    ).read_text()
    # the year-end clear block
    block = loops.split("# --- clear the annual accumulators")[1].split(")\n")[0]
    cleared = set(re.findall(r"s\.summary\.(ay\w+)", block))
    missing = annual - cleared
    assert not missing, (
        f"annual accumulators never reset: {sorted(missing)}. They are labelled "
        f"per-year in summary.py and diagnostics.py but would accumulate across "
        f"the whole run.")


def test_nee_is_npp_minus_rh_with_a_sink_positive_sign():
    """NEE must be formed from both halves, with F77's sign.

    The port accumulated `ayneetot - npp_step` every timestep and never added
    Rh, reporting -106 PgC/yr for a spin-up whose soil carbon had converged.
    That also inverted the sign against dynaveg's `ayneetot - cdisturb`
    (vegetation.f:481), so biomass burning increased the apparent sink.

    F77 (stats.f:1200): ayneetot(i) = aynpptot(i) - ayco2mic(i), POSITIVE =
    sink, formed once at year end.
    """
    from pathlib import Path
    import ibis.timestepping.loops_gswp3 as L

    src = Path(L.__file__).read_text()
    assert "state.summary.aynpptot - state.summary.ayco2mic" in src, (
        "NEE must be aynpptot - ayco2mic, formed at year end as in F77")
    assert "sm.ayneetot  - npp_step" not in src, (
        "NEE must not be accumulated from NPP alone -- that omits Rh entirely")


def test_total_and_aboveground_npp_are_separate_fields():
    """aynpptot (total) and ayanpptot (above-ground) must not collide.

    F77 comsum.h:117-118 declares them separately: "annual above-ground npp"
    and "annual total npp". The port accumulated TOTAL npp into ayanpptot while
    dynaveg overwrote the same field with the ABOVE-GROUND sum, so whichever
    wrote last won and the published field silently became ANPP.
    """
    from pathlib import Path
    from ibis.state.summary import SummaryState
    import ibis.timestepping.loops_gswp3 as L
    import ibis.physics.vegetation_dynamics as V

    for f in ("aynpptot", "ayanpptot"):
        assert f in SummaryState.__dataclass_fields__, f"{f} missing"

    loops = Path(L.__file__).read_text()
    assert "sm.aynpptot  + npp_step" in loops, (
        "the timestepper must accumulate TOTAL npp into aynpptot")
    assert "sm.ayanpptot + npp_step" not in loops, (
        "the timestepper must not write above-ground npp -- dynaveg owns it")

    veg = Path(V.__file__).read_text()
    assert "ayanpptot = jnp.sum(local_ayanpp, axis=1)" in veg, (
        "dynaveg must own ayanpptot, the above-ground sum")
    assert "aynpptot" not in veg.replace("ayanpptot", ""), (
        "dynaveg must not touch total npp")
