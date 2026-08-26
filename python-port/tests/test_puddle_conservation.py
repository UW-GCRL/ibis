"""A puddle cannot supply more water than it holds.

`setsoi` follows F77 (`soil.f:135-160`) and assigns ALL ground evaporation to
the puddle when the surface is 100 % dry, qglif(3) = 1. F77's own comment
concedes this "could lead to negative puddle depths", assuming it will not
matter because "for a 100% dry soil with small puddles, evaporation is likely
to be very small or less than zero (condensation)".

That assumption is written for dew. It is violated wherever a bone-dry surface
meets strong evaporative demand -- measured at cell 85004 (75.0 N, 91.0 E) on
5 August 1915, where `wpud` reached -3e-3 kg/m2 and the next timestep read the
negative puddle back as a spurious CONDENSATION flux: +339 W/m2 of ground heat
flux at night with zero insolation, and tsoi[0] up 8.7 K in one hour. Repeated,
it drove the thermal solve to 324.7 K on CPU and NaN on GPU.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.constants import get_constants
from ibis.dimensions import GridConfig
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.loops import default_sim_params, timestep_body

NPOI = 4
GRID = GridConfig(nlon=2, nlat=2, npoi=NPOI, xres=90.0, yres=90.0,
                  nband=2, nsoilay=6, nsnolay=3, npft=12)
DTIME = 3600.0


def _dry_surface_state(wsoi_val=0.0, wpud=0.0):
    """A bone-dry surface with an empty puddle under strong evaporative demand.

    This is the configuration F77's comment assumes cannot happen with a large
    positive flux: dry soil, no puddle, hot dry air, full sun.
    """
    st = create_ibis_state(GRID)
    st = eqx.tree_at(
        lambda s: (s.vegetation.lai, s.vegetation.sai, s.vegetation.fu,
                   s.vegetation.fl, s.vegetation.ztop, s.vegetation.zbot,
                   s.vegetation.exist, s.vegetation.frac, s.vegetation.cbior,
                   s.vegetation.cbiow, s.vegetation.sapfrac),
        st,
        (jnp.full((NPOI, 2), 0.2), jnp.full((NPOI, 2), 0.1),
         jnp.full((NPOI,), 0.25), jnp.full((NPOI,), 0.25),
         jnp.array([[1.0, 2.0]] * NPOI), jnp.array([[0.1, 0.5]] * NPOI),
         jnp.ones((NPOI, 12)), jnp.full((NPOI, 12), 1.0 / 12.0),
         jnp.full((NPOI, 12), 0.05), jnp.full((NPOI, 12), 0.2),
         jnp.full((NPOI,), 0.1)),
    )
    st = eqx.tree_at(
        lambda s: (s.soil.poros, s.soil.hydraul, s.soil.suction, s.soil.bex,
                   s.soil.sfield, s.soil.swilt, s.soil.sand, s.soil.clay,
                   s.soil.hsoi, s.soil.z0soi, s.soil.wsoi, s.soil.wisoi,
                   s.soil.albsav, s.soil.albsan, s.soil.wpud, s.soil.wipud,
                   s.soil.tsoi, s.soil.tg),
        st,
        (jnp.full((NPOI, 6), 0.463), jnp.full((NPOI, 6), 1e-5),
         jnp.full((NPOI, 6), -0.11), jnp.full((NPOI, 6), 4.5),
         jnp.full((NPOI, 6), 0.35), jnp.full((NPOI, 6), 0.15),
         jnp.full((NPOI, 6), 50.0), jnp.full((NPOI, 6), 20.0),
         jnp.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00, 4.00]),
         jnp.full((NPOI,), 0.01),
         jnp.full((NPOI, 6), wsoi_val), jnp.zeros((NPOI, 6)),
         jnp.full((NPOI,), 0.10), jnp.full((NPOI,), 0.20),
         jnp.full((NPOI,), wpud), jnp.zeros((NPOI,)),
         jnp.full((NPOI, 6), 290.0), jnp.full((NPOI,), 300.0)),
    )
    st = eqx.tree_at(
        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf, s.atm.coszen,
                   s.atm.solad, s.atm.solai, s.atm.fira, s.atm.raina,
                   s.atm.snowa),
        st,
        (jnp.full((NPOI,), 300.0), jnp.full((NPOI,), 0.002),   # hot, dry air
         jnp.full((NPOI,), 5.0), jnp.full((NPOI,), 99000.0),
         jnp.full((NPOI,), 0.9),
         jnp.full((NPOI, 2), 400.0), jnp.full((NPOI, 2), 80.0),
         jnp.full((NPOI,), 330.0),
         jnp.zeros((NPOI,)), jnp.zeros((NPOI,))),
    )
    st = eqx.tree_at(
        lambda s: (s.canopy.tu, s.canopy.ts, s.canopy.tl, s.canopy.t12,
                   s.canopy.t34, s.canopy.q12, s.canopy.q34),
        st,
        tuple([jnp.full((NPOI,), 300.0)] * 5)
        + (jnp.full((NPOI,), 0.002), jnp.full((NPOI,), 0.002)),
    )
    return st, get_constants(), default_sim_params(NPOI), DTIME


def test_puddle_never_goes_negative_on_a_dry_surface():
    """The invariant. Negative puddle water is not a physical state."""
    st, consts, params, dtime = _dry_surface_state()
    for _ in range(12):
        st = timestep_body(st, consts, params, dtime)
        wpud = np.asarray(st.soil.wpud)
        assert np.all(wpud >= 0.0), f"negative puddle water: {wpud.min()}"
        assert np.all(np.asarray(st.soil.wipud) >= 0.0)


def test_a_dry_surface_does_not_manufacture_a_nighttime_heat_source():
    """The consequence: heatg must stay a sink when there is no sunlight.

    The negative puddle was read back as condensation, releasing latent heat
    into the ground -- +339 W/m2 at night in the failing cell.
    """
    st, consts, params, dtime = _dry_surface_state()
    for _ in range(6):
        st = timestep_body(st, consts, params, dtime)
    # switch the sun off; nothing should be pumping heat into the ground
    st = eqx.tree_at(
        lambda s: (s.atm.solad, s.atm.solai, s.atm.coszen),
        st, (jnp.zeros((NPOI, 2)), jnp.zeros((NPOI, 2)), jnp.zeros((NPOI,))),
    )
    for _ in range(6):
        st = timestep_body(st, consts, params, dtime)
        heatg = np.asarray(st.soil.heatg)
        assert np.all(heatg < 200.0), f"nighttime ground heat flux {heatg.max()}"


def test_dry_surface_stays_finite_and_physical():
    """No NaN, and no soil layer hotter than boiling."""
    st, consts, params, dtime = _dry_surface_state()
    for _ in range(24):
        st = timestep_body(st, consts, params, dtime)
    tsoi = np.asarray(st.soil.tsoi)
    assert np.all(np.isfinite(tsoi))
    assert tsoi.max() < 340.0, f"runaway soil temperature {tsoi.max()}"
    assert np.all(np.isfinite(np.asarray(st.soil.tg)))


def test_a_real_puddle_still_evaporates():
    """The floor must not stop a puddle that actually holds water."""
    st, consts, params, dtime = _dry_surface_state(wpud=2.0)
    start = float(np.asarray(st.soil.wpud)[0])
    for _ in range(6):
        st = timestep_body(st, consts, params, dtime)
    end = float(np.asarray(st.soil.wpud)[0])
    assert end < start, "a wet puddle under hot dry air should lose water"
    assert end >= 0.0
