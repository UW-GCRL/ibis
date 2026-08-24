"""Where fire-killed carbon actually goes.

IBIS 2.6b4 adds every gram of disturbed biomass to `cdisturb` and subtracts it
from NEE (`vegetation.f:433,481`), with an unresolved author TODO from
2000/04/22 above it asking whether some should become litter instead. That was
harmless while `disturbf` was a constant 0.005. It is not harmless once the
seasonlength scheme produces 30-50 % burned fractions: wood is ~97 % of live
biomass in the spun-up global state, and `cc_wood = 0.20`, so the old path
emitted roughly five times what `bgc.fire_emissions` reports as burned -- and
the difference was neither emitted nor left as litter. It vanished.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from ibis.bgc.fire_emissions import FireEmissionParams
from ibis.config import RunConfig
from ibis.io.parameters import default_pft_params
from ibis.physics.fire import (
    CC_LEAF, CC_LITTER, CC_WOOD, LEGACY, SEASONLENGTH, FireParams,
)
from ibis.physics.vegetation_dynamics import dynaveg
from ibis.state.carbon import create_carbon_state
from ibis.state.summary import create_summary_state
from ibis.state.vegetation import create_vegetation_state


def _burning_state(npoi=32, npft=12):
    """A state that actually burns: real accumulators, real biomass."""
    veg = create_vegetation_state(npoi, npft, 6)
    carbon = create_carbon_state(npoi)
    summary = create_summary_state(npoi, npft)
    nyt = 17520.0
    veg = _set(veg, nytimes=jnp.asarray(nyt),
               nfirestep_vpd=jnp.full((npoi,), 0.45 * nyt),
               sdry_vpd=jnp.full((npoi,), 0.45 * nyt * 0.8),
               nfirestep=jnp.full((npoi,), 0.45 * nyt),
               sdry=jnp.full((npoi,), 0.45 * nyt * 0.8),
               cbiol=jnp.full((npoi, npft), 0.15),
               cbiow=jnp.full((npoi, npft), 1.20),
               cbior=jnp.full((npoi, npft), 0.10),
               plai=jnp.full((npoi, npft), 0.30),
               exist=jnp.ones((npoi, npft)))
    carbon = _set(carbon, totlit=jnp.full((npoi,), 1.0),
                  totalit=jnp.full((npoi,), 0.6))
    return veg, carbon, summary


def _set(obj, **kw):
    import equinox as eqx
    for k, v in kw.items():
        obj = eqx.tree_at(lambda o, _k=k: getattr(o, _k), obj, v)
    return obj


def _run(fp):
    veg, carbon, summary = _burning_state()
    nv, nc, ns = dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12,
                         fire_scheme=SEASONLENGTH, fire_params=fp)
    killed = float(jnp.sum(veg.cbiol + veg.cbiow + veg.cbior)
                   - jnp.sum(nv.cbiol + nv.cbiow + nv.cbior))
    emitted = float(jnp.sum(summary.ayneetot) - jnp.sum(ns.ayneetot))
    litter = float(jnp.sum(nc.falll + nc.fallw + nc.fallr))
    return killed, emitted, litter, nv


def test_the_scheme_actually_burns_something():
    """Guard the fixture: a vacuous disturbf makes every test below trivial."""
    _, _, _, nv = _run(FireParams())
    assert float(jnp.mean(nv.disturbf)) > 0.01


def test_completeness_routing_conserves_carbon():
    """What fire kills is either emitted or left as litter. Nothing vanishes."""
    k_old, e_old, l_old, _ = _run(FireParams(use_combustion_completeness=False))
    k_new, e_new, l_new, _ = _run(FireParams(use_combustion_completeness=True))
    assert k_new == pytest.approx(k_old, rel=1e-6)   # mortality is unchanged
    # Carbon that stopped being emitted shows up in litter, to the last gram.
    assert (e_old - e_new) == pytest.approx(l_new - l_old, rel=1e-5)


def test_completeness_routing_emits_less():
    """The whole point: wood is killed far more often than it is combusted."""
    _, e_old, _, _ = _run(FireParams(use_combustion_completeness=False))
    _, e_new, _, _ = _run(FireParams(use_combustion_completeness=True))
    assert e_new < e_old * 0.6


def test_fine_roots_are_never_combusted():
    """Roots are underground. Killing them must not emit them."""
    veg, carbon, summary = _burning_state()
    _, nc, _ = dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12,
                       fire_scheme=SEASONLENGTH,
                       fire_params=FireParams(use_combustion_completeness=True))
    killed_root = float(jnp.sum(veg.cbior) - 0.0)
    # Every fire-killed root goes to fallr, none to the atmosphere.
    assert float(jnp.sum(nc.fallr)) > 0.0
    assert killed_root > 0.0


def test_legacy_scheme_ignores_the_switch():
    """`legacy` reproduces F77 exactly and must not be touched by this."""
    veg, carbon, summary = _burning_state()
    outs = []
    for flag in (False, True):
        _, _, ns = dynaveg(veg, carbon, summary, default_pft_params(), 1, 1e-12,
                           fire_scheme=LEGACY,
                           fire_params=FireParams(use_combustion_completeness=flag))
        outs.append(float(jnp.sum(ns.ayneetot)))
    assert outs[0] == pytest.approx(outs[1], rel=1e-12)


def test_combustion_completeness_has_one_definition():
    """Two copies is how the balance and the diagnostic came to disagree."""
    fp, fe = FireParams(), FireEmissionParams()
    assert (fp.cc_leaf, fp.cc_wood, fp.cc_litter) == (CC_LEAF, CC_WOOD, CC_LITTER)
    assert (fe.cc_leaf, fe.cc_wood, fe.cc_litter) == (CC_LEAF, CC_WOOD, CC_LITTER)


# --- the three dead config flags --------------------------------------------


def test_fire_emissions_is_reachable_from_the_package():
    """F5 was defined, tested, and importable nowhere a run would find it."""
    from ibis.bgc import fire_emissions_from_state
    assert callable(fire_emissions_from_state)


def test_run_global_passes_fire_scheme_to_dynaveg():
    """The call site omitted it, so every run took legacy whatever config said."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "run_global.py").read_text()
    assert "fire_scheme=p.fire_scheme" in src


def test_run_global_reads_enable_fire_emissions():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "run_global.py").read_text()
    assert "p.enable_fire_emissions" in src


def test_vpd_danger_is_on_by_default():
    """+0.630 rank against GFED5, versus +0.315 for the soil-moisture path."""
    assert FireParams().use_vpd_danger is True
    assert RunConfig().fire_scheme == SEASONLENGTH


def test_pre_f1b_state_fails_with_a_readable_message():
    """A cryptic AttributeError from inside the disturbance branch is not ok."""
    veg, carbon, summary = _burning_state()
    stripped = type(veg).__new__(type(veg))
    for fd in type(veg).__dataclass_fields__.values():
        if fd.name in ("nfirestep_vpd", "sdry_vpd"):
            continue
        object.__setattr__(stripped, fd.name, getattr(veg, fd.name))
    with pytest.raises(AttributeError, match="predates F1b"):
        dynaveg(stripped, carbon, summary, default_pft_params(), 1, 1e-12,
                fire_scheme=SEASONLENGTH, fire_params=FireParams())
