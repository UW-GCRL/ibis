"""The soil solid-phase properties must be read, not silently replaced.

`io/soil_init.py` computed the specific heat and density faithfully from F77
(`initial.f:715-724`, Campbell & Norman 1998), `scripts/run_global.py` stored
`csoi` into the state -- and the physics read neither, substituting flat
constants that dropped the organic correction. `rhosoi` had no state field at
all. Same class of defect as the dead `firefac` and the three dead config
flags: computed, stored, verified, and fed to nothing.

There were three separate copies of the two constants (`soil_init`,
`SimParams`, `physics_params`), two of which omitted the correction.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ibis.io.soil_init import CSOI_VAL, FORGANIC, RHOSOI_VAL, compute_soil_properties
from ibis.timestepping.loops import SimParams, _solid_phase, default_sim_params


def test_the_constants_have_one_definition():
    p = default_sim_params(4)
    assert p.csoi_mineral_val == CSOI_VAL
    assert p.rhosoi_val == RHOSOI_VAL
    assert SimParams.__dataclass_fields__["csoi_mineral_val"].default == CSOI_VAL
    assert SimParams.__dataclass_fields__["rhosoi_val"].default == RHOSOI_VAL


def test_the_constants_match_the_f77_formula():
    """`initial.f:715-724`, Campbell & Norman 1998."""
    assert CSOI_VAL == pytest.approx(870.0 * (1 - FORGANIC) + 1920.0 * FORGANIC)
    assert RHOSOI_VAL == pytest.approx(2650.0 * (1 - FORGANIC) + 1300.0 * FORGANIC)
    # and they are NOT the bare mineral values the physics used to assume
    assert CSOI_VAL != 870.0
    assert RHOSOI_VAL != 2650.0


def test_soil_init_emits_both_properties():
    sand = np.full((5, 6), 40.0, dtype=np.float32)
    clay = np.full((5, 6), 25.0, dtype=np.float32)
    props = compute_soil_properties(sand, clay)
    assert np.allclose(np.asarray(props["csoi"]), CSOI_VAL)
    assert np.allclose(np.asarray(props["rhosoi"]), RHOSOI_VAL)


class _Soil(eqx.Module):
    tsoi: jnp.ndarray
    csoi: jnp.ndarray
    rhosoi: jnp.ndarray


class _SoilNoRhosoi(eqx.Module):
    tsoi: jnp.ndarray
    csoi: jnp.ndarray


def test_populated_state_is_used_not_the_constant():
    """A spatially varying organic map must reach the physics when supplied."""
    n, ns = 3, 6
    soil = _Soil(tsoi=jnp.full((n, ns), 280.0),
                 csoi=jnp.full((n, ns), 1500.0),
                 rhosoi=jnp.full((n, ns), 1200.0))
    csoi, rho = _solid_phase(soil, default_sim_params(n))
    assert np.allclose(np.asarray(csoi), 1500.0)
    assert np.allclose(np.asarray(rho), 1200.0)


def test_unset_state_falls_back_to_the_constant():
    """Checkpoints written before this fix carry csoi == 0.

    Zero heat capacity would be catastrophic rather than merely wrong, so an
    unset state must take the parameter.
    """
    n, ns = 3, 6
    soil = _Soil(tsoi=jnp.full((n, ns), 280.0),
                 csoi=jnp.zeros((n, ns)), rhosoi=jnp.zeros((n, ns)))
    csoi, rho = _solid_phase(soil, default_sim_params(n))
    assert np.allclose(np.asarray(csoi), CSOI_VAL)
    assert np.allclose(np.asarray(rho), RHOSOI_VAL)


def test_state_without_a_rhosoi_field_still_works():
    """Older checkpoints predate the field entirely."""
    n, ns = 3, 6
    soil = _SoilNoRhosoi(tsoi=jnp.full((n, ns), 280.0),
                         csoi=jnp.full((n, ns), 1500.0))
    csoi, rho = _solid_phase(soil, default_sim_params(n))
    assert np.allclose(np.asarray(csoi), 1500.0)
    assert np.allclose(np.asarray(rho), RHOSOI_VAL)


def test_partially_populated_state_mixes_per_cell():
    """A land mask can leave ocean cells at zero; those take the constant."""
    n, ns = 4, 6
    csoi_in = jnp.asarray(np.tile([[1500.0], [0.0], [900.0], [0.0]], (1, ns)))
    soil = _Soil(tsoi=jnp.full((n, ns), 280.0), csoi=csoi_in,
                 rhosoi=jnp.zeros((n, ns)))
    csoi, _ = _solid_phase(soil, default_sim_params(n))
    got = np.asarray(csoi)[:, 0]
    assert got[0] == pytest.approx(1500.0)
    assert got[1] == pytest.approx(CSOI_VAL)
    assert got[2] == pytest.approx(900.0)
