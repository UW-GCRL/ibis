"""Guards the defect-0 pattern: a feature that appears wired and does nothing.

`test_no_dead_code.py` catches two shapes -- a module with no production
caller, and a config field never read outside its declaration. It missed a
third, which has now bitten this project five times:

  * `firefac`                      computed, never used (docs/FIRE.md)
  * `soilcspin`                    declared, read nowhere
  * `stressfac`                    loaded into StomataParams, never passed to
                                   drystress, so the hardcoded -5.0 always won
  * `methane_params`               never passed to run_month_gswp3, so every
                                   production run used the "global" CH4
                                   configuration (458 Tg) instead of the
                                   validated one (182 Tg)
  * `tgpp`/`tnpp`/`tco2root`       state fields never assigned

The common shape is that the NAME appears -- it is declared, imported, loaded
into a params object -- so a grep-for-the-name test passes while the VALUE
never reaches the physics. These two tests check reachability instead.
"""

import ast
import collections
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _production_sources():
    out = {}
    for base in (ROOT / "ibis", ROOT / "scripts", REPO / "verify"):
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            sp = str(p)
            if "_legacy" in sp or "/tests/" in sp or "__pycache__" in sp:
                continue
            try:
                out[p] = p.read_text()
            except OSError:
                continue
    return out


#: Optional keyword parameters that carry configuration into the physics.
#: If nothing passes one, its default silently becomes the only behaviour.
_CONFIG_KWARG = re.compile(r"(_params$|^stressfac$|^nsoilcall$|^f_inund$|"
                           r"^fire_scheme$|^n_hydro$)")

#: Declared for callers outside the scanned trees, or deliberately defaulted.
KNOWN_UNPASSED_KWARGS = {
    # run_spinup_gswp3's isimfire is passed by verify/ spin-up drivers; it is
    # not reachable from run_global, which has its own year loop.
}


def test_config_carrying_kwargs_are_passed_by_some_caller():
    src = _production_sources()
    declared = collections.defaultdict(list)
    passed = collections.defaultdict(set)

    for path, text in src.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                names = ([x.arg for x in a.args[len(a.args) - len(a.defaults):]]
                         + [x.arg for x in a.kwonlyargs])
                for nm in names:
                    if _CONFIG_KWARG.search(nm):
                        declared[nm].append(f"{path.name}:{node.name}")
            elif isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg:
                        passed[kw.arg].add(path.name)

    dead = {nm: where for nm, where in declared.items()
            if not passed.get(nm) and nm not in KNOWN_UNPASSED_KWARGS}
    assert not dead, (
        "config-carrying parameter(s) that no caller ever passes, so the "
        "default is the only behaviour the model can have:\n" +
        "\n".join(f"  {nm}  declared in {w}" for nm, w in sorted(dead.items())))


#: State fields that are legitimately never written by the model.
KNOWN_UNWRITTEN_STATE_FIELDS = {
    # Diagnostic fields IBIS declares but never computes. Documented in
    # docs/DEFECTS_FOR_REVIEW.md -- wiring them means deciding where GPP/NPP
    # are summed, which is a physics change rather than a reporting fix.
    "tgpp", "tnpp", "tgpptot", "tnpptot", "tco2root", "tneetot",
    # Allocation fractions duplicated onto state; dynaveg reads them from
    # pft_params instead.
    "aleaf", "aroot", "awood", "specla",
    # Canopy water uptake per soil layer. The PHYSICS is live -- loops.py
    # computes upsoiu_k/upsoil_k locally and uses them to remove water
    # (loops.py:956) and heat (973-974) from each layer -- but the results are
    # never stored back onto the state. So transpiration extraction is
    # correct, and only the diagnostic reads zero. Anyone inspecting
    # state.soil.upsoil expecting a flux will get nothing.
    "upsoil", "upsoiu",
}


def test_state_fields_are_written_somewhere():
    """Every prognostic/diagnostic state field must be assigned in production.

    A field that is only ever read stays at its initial value -- which for a
    zero-initialised state means the quantity silently reads zero everywhere,
    exactly as tgpp/tnpp did.
    """
    src = _production_sources()
    state_dir = ROOT / "ibis" / "state"
    fields = {}
    for p in state_dir.glob("*.py"):
        for nm in re.findall(r"^\s{4}(\w+)\s*:\s*(?:Float|Int|jnp|Array)",
                             p.read_text(), re.M):
            fields[nm] = p.name
    assert fields, "no state fields found -- has ibis/state moved?"

    unwritten = []
    for nm, owner in fields.items():
        if nm in KNOWN_UNWRITTEN_STATE_FIELDS:
            continue
        hit = False
        for path, text in src.items():
            if path.name == owner and "state" in str(path):
                continue
            # assigned directly, or named as a tree_at target, or passed by kw
            if (re.search(rf"\b{nm}\s*=", text)
                    or re.search(rf"\.{nm}\b", text)):
                hit = True
                break
        if not hit:
            unwritten.append(f"{nm} (declared in state/{owner})")
    assert not unwritten, (
        "state field(s) never touched by production code -- they will read "
        "their initial value forever:\n" + "\n".join(f"  {u}" for u in unwritten))


def test_seed_lai_flag_reaches_the_state():
    """`seed_lai` must actually change leaf area, not just parse.

    Added because this project has now found thirteen flags, parameters and
    state fields that were declared, documented and read but never reached
    the physics -- `firefac`, `stressfac`, `soilcspin`, `methane_params`,
    `fire_scheme`, `specla` in the FLUXNET harness, `tco2root`, and the seven
    unreset annual accumulators. A new flag gets a reachability test before it
    gets a run.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "run_global.py"
    text = src.read_text()

    # the flag is read
    assert 'getattr(p, "seed_lai", False)' in text, "seed_lai never read"

    # ... and the value it loads is passed to apply_prescribed_lai
    block = text.split('getattr(p, "seed_lai", False)')[1]
    block = block.split('getattr(p, "prescribe_lai"')[0]
    assert "apply_prescribed_lai" in block, (
        "seed_lai is read but never applied -- defect-0 pattern")
    assert "lai_seed" in block, "seeded field not passed through"

    # and it must NOT be re-imposed inside the annual loop: that is
    # `prescribe_lai`, a different experiment.
    after = text.split("for iyear_offset in range")[1]
    assert "lai_seed" not in after, (
        "seed_lai must apply once, not every year -- that is prescribe_lai")


def test_seed_lai_is_declared_in_config():
    """Read-but-undeclared flags raised AttributeError on every dynamic run."""
    from ibis.io.config_reader import PhysicsSection
    assert hasattr(PhysicsSection(), "seed_lai")


def test_climate_normals_not_reseeded_on_restart():
    """Restarts must keep the normals the run accumulated, not clobber them.

    The reseed is derived from a SINGLE forcing year and `existence` reads
    tcmin/tw/gdd0/gdd5 directly, so running it on a restart flipped PFT
    existence in 6.3% of cells (3,173 of them north of 50N) against the
    both_fix year-2200 checkpoint. Inverse of the defect-0 pattern: a value
    written where it should not be.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "run_global.py"
    text = src.read_text()

    marker = "s.vegetation.tc, s.vegetation.tw, s.vegetation.tcmin"
    assert marker in text, "climate-normal write not found"
    before = text.split(marker)[0]
    # the nearest enclosing guard must be a cold-start test
    guard = before.rfind("if t.irestart == 0:")
    treeat = before.rfind("eqx.tree_at(")
    assert guard != -1 and guard < treeat, (
        "climate normals are written without a cold-start guard")


def test_transient_co2_reaches_the_state():
    """`transient_co2` must set atm.co2conc each year, not merely parse."""
    from pathlib import Path
    from ibis.io.config_reader import PhysicsSection

    assert hasattr(PhysicsSection(), "transient_co2")
    src = Path(__file__).resolve().parents[1] / "scripts" / "run_global.py"
    text = src.read_text()
    assert 'getattr(p, "transient_co2", False)' in text
    loop = text.split("for iyear_offset in range")[1]
    assert "apply_co2(state, year)" in loop, (
        "transient_co2 read but CO2 never applied inside the annual loop")


def test_co2_trajectory_matches_known_values():
    """Guard the hand-entered table against silent corruption."""
    from ibis.io.co2 import co2_ppm
    # Mauna Loa annual means
    assert abs(co2_ppm(1960) - 316.91) < 0.01
    assert abs(co2_ppm(1990) - 354.39) < 0.01
    assert abs(co2_ppm(2010) - 389.85) < 0.01
    # monotonic across the historical period
    v = [co2_ppm(y) for y in range(1901, 2011)]
    assert all(b >= a for a, b in zip(v, v[1:])), "CO2 must be non-decreasing"


def test_fit_carbon_has_no_dead_co2_flag():
    """The fit must not carry a CO2 flag that `one_year` silently overrides."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "verify" / "calibration"
           / "fit_carbon.py")
    if not src.exists():
        return
    text = src.read_text()
    assert "--co2-year" not in text, (
        "fit_carbon declares --co2-year, but one_year calls apply_co2(state, "
        "year) and overrides it")


def test_run_global_logs_which_parameters_it_uses():
    """A run must say whether it is on a params_file or on defaults."""
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "scripts" / "run_global.py").read_text()
    assert 'logger.info("Physics parameters: %s", config.params_file)' in text
    assert "IBIS DEFAULTS (no params_file)" in text


def test_pheno_params_defaults_are_the_f77_literals():
    """PhenoParams was lifted out of literals inside pheno(); its defaults must
    stay the IBIS 2.6b4 values (vegetation.f) so default behaviour is unchanged.
    A bit-identity probe on 54 cells proved this at the time of the lift; this
    pins it against drift."""
    from ibis.physics.phenology import PhenoParams
    p = PhenoParams()
    assert (p.tthresh_u_offset, p.gthresh_u_c, p.gdd_thresh_u, p.gdd_ramp_u) == (5.0, 0.0, 100.0, 50.0)
    assert (p.tthresh_l_c, p.gthresh_l_c, p.gdd_thresh_l, p.gdd_ramp_l) == (0.0, -5.0, 150.0, 50.0)
    assert (p.drop_thresh, p.drop_floor, p.ddays) == (0.0, 0.1, 15.0)


def test_pheno_params_reach_pheno_through_the_gswp3_loop():
    """The lifted parameters must actually be passed -- SimParams.pheno into
    pheno() inside loops_gswp3 -- or they are the fifteenth declared-but-dead
    knob in this project."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "ibis"
    loop = (root / "timestepping" / "loops_gswp3.py").read_text()
    assert "params.pheno)" in loop, "loops_gswp3 calls pheno() without SimParams.pheno"
    sim = (root / "timestepping" / "loops.py").read_text()
    assert "pheno: PhenoParams" in sim
    ph = (root / "physics" / "phenology.py").read_text()
    for name in ("pp.gdd_thresh_u", "pp.gdd_thresh_l", "pp.drop_floor", "pp.ddays",
                 "pp.tthresh_u_offset", "pp.gthresh_l_c"):
        assert name in ph, f"{name} not used inside pheno()"



def test_pheno_yaml_section_round_trips_and_rejects_typos(tmp_path):
    """A fitted phenology must travel through the params yaml into SimParams,
    and a misspelt key must fail rather than silently default."""
    import numpy as np, yaml
    from ibis.io.physics_params import build_all_params, load_physics_params
    good = {"stomata": {"stressfac": -7.5},
            "pheno": {"tthresh_l_c": 3.5, "gthresh_l_c": -1.0, "drop_floor": 0.2, "ddays": 30.0}}
    p = tmp_path / "p.yaml"; p.write_text(yaml.safe_dump(good))
    sp, _, _ = build_all_params(4, np.zeros(4, np.float32), load_physics_params(str(p)))
    assert float(sp.stomata_params.stressfac) == -7.5
    assert (float(sp.pheno.tthresh_l_c), float(sp.pheno.gthresh_l_c),
            float(sp.pheno.drop_floor), float(sp.pheno.ddays)) == (3.5, -1.0, 0.2, 30.0)
    assert float(sp.pheno.gdd_thresh_l) == 150.0          # untouched field keeps its default
    bad = {"pheno": {"ttresh_l_c": 3.5}}                   # typo
    q = tmp_path / "q.yaml"; q.write_text(yaml.safe_dump(bad))
    import pytest
    with pytest.raises(KeyError):
        build_all_params(4, np.zeros(4, np.float32), load_physics_params(str(q)))
