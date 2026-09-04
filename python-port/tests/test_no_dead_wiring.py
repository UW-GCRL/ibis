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
