"""Verify the JAX thermo module against the F77 statement functions.

This is the *template* application of the verify/ framework. The structure
applied to thermo here is what will be repeated for every section:

  1. Build / import the F77 wrapper for the section.
  2. Build a list of inputs (fixed scenarios + dense sweep + random).
  3. Run F77 reference and JAX candidate on each input.
  4. Compute residual stats per output variable; judge against tolerance.
  5. Print a table, save residuals, return PASS/FAIL.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=.:python-port verify/.venv/bin/python -m verify.thermo.verify_thermo
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # match Fortran -fdefault-real-8

# Ensure the compiled F77 wrapper directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fortran_wrappers"))

import thermo_wrap as f77  # the f2py module we just built

from ibis.physics.thermo import esat as jax_esat
from ibis.physics.thermo import desat as jax_desat
from ibis.physics.thermo import qsat as jax_qsat
from ibis.physics.thermo import dqsat as jax_dqsat
from ibis.physics.thermo import hvapf as jax_hvapf
from ibis.physics.thermo import hsubf as jax_hsubf

from verify.framework.harness import Report, verify_variable
from verify.framework.inputs import (
    Range, fixed_scenarios_T_K, linspace_sweep, random_uniform,
)
from verify.thermo.tolerance import SPECS


# Physical constants -- must exactly match ibis.constants.PhysicalConstants.
TMELT = 273.16
HVAP  = 2.5104e6
HFUS  = 0.3336e6
HSUB  = HVAP + HFUS
CH2O  = 4.218e3
CICE  = 2.106e3
CVAP  = 1.81e3


def make_temperatures() -> np.ndarray:
    """Concatenate fixed + sweep + random samples."""
    return np.concatenate([
        fixed_scenarios_T_K(),
        linspace_sweep(Range(180.0, 400.0), n=441),    # 0.5 K step
        random_uniform(Range(200.0, 350.0), n=5000, seed=1),
    ]).astype(np.float64)


def make_pressures() -> np.ndarray:
    """Surface pressure samples spanning sea-level to high-altitude."""
    return np.concatenate([
        np.array([101325.0, 95000.0, 80000.0, 60000.0, 40000.0]),
        linspace_sweep(Range(30000.0, 105000.0), n=200),
        random_uniform(Range(30000.0, 105000.0), n=5000, seed=2),
    ]).astype(np.float64)


def verify_section() -> Report:
    report = Report(section="thermo")

    # --- esat / desat: scalar in T ---
    T = make_temperatures()
    ref = f77.wrap_esat(T)
    cand = np.asarray(jax_esat(jnp.array(T)))
    report.add(verify_variable("esat", ref, cand, SPECS["esat"]))

    ref = f77.wrap_desat(T)
    cand = np.asarray(jax_desat(jnp.array(T)))
    report.add(verify_variable("desat", ref, cand, SPECS["desat"]))

    # --- qsat: needs (esat_value, pressure) on equal-length arrays ---
    # Use a paired sweep: same length, varied T to derive esat, varied P.
    rng = np.random.default_rng(3)
    Tq = rng.uniform(220.0, 320.0, 5000).astype(np.float64)
    Pq = rng.uniform(40000.0, 105000.0, 5000).astype(np.float64)
    eq_ref = f77.wrap_esat(Tq)
    ref = f77.wrap_qsat(eq_ref, Pq)
    cand = np.asarray(jax_qsat(jnp.array(eq_ref), jnp.array(Pq)))
    report.add(verify_variable("qsat", ref, cand, SPECS["qsat"],
                                abs_floor=1e-6))  # qsat is tiny (~kg/kg)

    # --- dqsat: needs (T, q1) ---
    q1_ref = np.asarray(jax_qsat(jnp.array(eq_ref), jnp.array(Pq)))
    ref = f77.wrap_dqsat(Tq, q1_ref)
    cand = np.asarray(jax_dqsat(jnp.array(Tq), jnp.array(q1_ref)))
    report.add(verify_variable("dqsat", ref, cand, SPECS["dqsat"],
                                abs_floor=1e-6))

    # --- hvapf / hsubf: paired (t, tair) over the temperature sweep ---
    rng = np.random.default_rng(4)
    Th = rng.uniform(220.0, 320.0, 5000).astype(np.float64)
    Tah = rng.uniform(220.0, 320.0, 5000).astype(np.float64)

    ref = f77.wrap_hvapf(Th, Tah, HVAP, CVAP, CH2O)
    cand = np.asarray(jax_hvapf(jnp.array(Th), jnp.array(Tah),
                                 TMELT, HVAP, CH2O, CVAP))
    report.add(verify_variable("hvapf", ref, cand, SPECS["hvapf"]))

    ref = f77.wrap_hsubf(Th, Tah, HSUB, CVAP, CICE)
    cand = np.asarray(jax_hsubf(jnp.array(Th), jnp.array(Tah),
                                 TMELT, HSUB, CVAP, CICE))
    report.add(verify_variable("hsubf", ref, cand, SPECS["hsubf"]))

    return report


if __name__ == "__main__":
    report = verify_section()
    report.print_table()
    out = report.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")
    sys.exit(0 if report.passed else 1)
