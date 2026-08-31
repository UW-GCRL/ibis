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
    # selectors for the component-swapping framework quarantined to
    # ibis/ml/_legacy_unwired/ -- nothing reads them
    "ml_stomata",
    "ml_canopy",
    # Surfaced 2026-08-31 when this guard was widened to scan
    # io/config_reader.py and to count every scalar field rather than only
    # names matching enable_*/isim*/*_scheme/ml_*. All three are output/input
    # options a user can set that nothing reads. Recorded rather than wired
    # so the guard keeps its teeth; wire or remove them.
    "idailyout",    # OutputSection: daily output switch
    "iyrdaily",     # OutputSection: first year of daily output
    "param_dir",    # InputSection: directory for parameter files
}

#: Modules with no production call site. Three groups:
#:   * dead model components  — bvoc, nee, permafrost, soil_gases, wetlands
#:   * work in progress       — methane (M2 wiring pending)
#:   * tools whose callers are not committed — the io readers and ml/*, used
#:     only by throwaway analysis scripts. Committing those callers is the fix.
#: Empty. Every module in ibis/ has a production caller as of 2026-08-29.
KNOWN_DEAD_MODULES: set[str] = set()


def test_no_new_dead_config_flags():
    """Every user-settable config field must be read somewhere outside its declaration.

    This used to scan only ibis/config.py, and only for names matching
    enable_*/isim*/*_scheme/ml_*. `soilcspin` in io/config_reader.py --
    documented as "0=no spinup, 1=acceleration" -- was therefore invisible on
    both counts, and appeared exactly once in the whole codebase: its own
    declaration. Both config modules are now scanned, and every scalar field
    counts, not just the ones matching a name pattern.
    """
    src = _sources()
    prod = {p: s for p, s in src.items() if _kind(p) == "prod"}
    decl_files = ["config.py", "io/config_reader.py"]
    flags = {}
    for rel in decl_files:
        path = ROOT / "python-port" / "ibis" / rel
        if not path.exists():
            continue
        # any dataclass field with an annotation and a default -- covers
        # Optional[str]/`x | None` as well as bare int/float/str/bool
        for name in re.findall(
                r"^\s{4}(\w+)\s*:\s*[\w\[\]|., ]+?\s*=", path.read_text(), re.M):
            flags[name] = path.name

    dead = {
        f for f, owner in flags.items()
        if not any(p.name != owner and re.search(rf"\b{f}\b", s)
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
