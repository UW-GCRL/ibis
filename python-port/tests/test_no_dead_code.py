"""Guard against the failure mode this codebase keeps hitting.

Five separate defects this month were the same shape: something written,
tested, exported, and called by nothing — the dead `firefac`, `fire_scheme`
and `enable_fire_emissions` read nowhere, `soil.csoi` stored and never read,
and the whole `bgc/` package reachable only from tests. Each was found by
accident while looking for something else.

A green test suite certifies that code is *correct*. It says nothing about
whether anything *calls* it. These tests close that gap.

The allowlists below are debt, not permission. They must shrink.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _kind(p: pathlib.Path) -> str:
    if "tests" in p.parts:
        return "test"
    if any("_legacy" in s for s in p.parts):
        return "legacy"
    if p.name == "__init__.py":
        return "reexport"        # a re-export is not a caller
    return "prod"


def _sources():
    out = {}
    for p in ROOT.rglob("*.py"):
        if ".git" in p.parts:
            continue
        try:
            out[p] = p.read_text(errors="replace")
        except OSError:
            pass
    return out


#: Config flags with no production reader. Every entry is a switch a user can
#: set that does nothing. Shrink this; do not grow it.
KNOWN_DEAD_FLAGS = {
    "enable_bvoc",
    "enable_permafrost",
    "enable_soil_n2o",
    "enable_wetlands",
}

#: Modules with no production call site. Three groups:
#:   * dead model components  — bvoc, nee, permafrost, soil_gases, wetlands
#:   * work in progress       — methane (M2 wiring pending)
#:   * tools whose callers are not committed — the io readers and ml/*, used
#:     only by throwaway analysis scripts. Committing those callers is the fix.
KNOWN_DEAD_MODULES = {
    "ibis/bgc/bvoc.py", "ibis/bgc/methane.py", "ibis/bgc/nee.py",
    "ibis/bgc/permafrost.py", "ibis/bgc/soil_gases.py", "ibis/bgc/wetlands.py",
    "ibis/io/gfed5.py", "ibis/io/popdens.py", "ibis/io/wad2m.py",
    "ibis/ml/base.py", "ibis/ml/fire_calibration.py", "ibis/ml/registry.py",
    "ibis/ml/wrappers.py",
}


def test_no_new_dead_config_flags():
    """Every enable_*/isim* flag must be read somewhere outside config.py."""
    src = _sources()
    prod = {p: s for p, s in src.items() if _kind(p) == "prod"}
    cfg = (ROOT / "python-port" / "ibis" / "config.py").read_text()
    flags = set(re.findall(r"^\s{4}(enable_\w+|isim\w+|\w+_scheme)\s*:", cfg, re.M))

    dead = {
        f for f in flags
        if not any(p.name != "config.py" and re.search(rf"\b{f}\b", s)
                   for p, s in prod.items())
    }
    new = dead - KNOWN_DEAD_FLAGS
    assert not new, (
        f"new dead config flag(s): {sorted(new)}. A flag a user can set that "
        f"nothing reads is the defect-0 pattern. Wire it or remove it."
    )
    fixed = KNOWN_DEAD_FLAGS - dead
    assert not fixed, (
        f"{sorted(fixed)} now have readers — remove them from "
        f"KNOWN_DEAD_FLAGS so the guard keeps its teeth."
    )


def test_no_new_modules_without_a_production_caller():
    src = _sources()
    prod = {p: s for p, s in src.items() if _kind(p) == "prod"}

    dead = set()
    for mod in (ROOT / "python-port" / "ibis").rglob("*.py"):
        if _kind(mod) != "prod":
            continue
        try:
            tree = ast.parse(src[mod])
        except (SyntaxError, KeyError):
            continue
        names = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and not n.name.startswith("_")]
        if not names:
            continue
        called = any(
            p != mod and re.search(rf"\b{n}\s*\(", s)
            for n in names for p, s in prod.items()
        )
        if not called:
            dead.add(str(mod.relative_to(ROOT / "python-port")))

    new = dead - KNOWN_DEAD_MODULES
    assert not new, (
        f"module(s) with no production caller: {sorted(new)}. Tests calling it "
        f"is not the same as the model calling it — see the module docstring."
    )
    fixed = KNOWN_DEAD_MODULES - dead
    assert not fixed, (
        f"{sorted(fixed)} now have production callers — remove them from "
        f"KNOWN_DEAD_MODULES."
    )
