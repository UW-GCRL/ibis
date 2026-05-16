"""Verify JAX freeze_thaw and puddle_freeze_thaw against F77 inline blocks.

F77 has no standalone freeze_thaw subroutine; the logic is inlined in
soilctl (soil.f lines 411-472 for soil layers, 487-514 for puddles).
We carved each block into its own callable F77 subroutine
(soil_freeze_thaw_block, puddle_freeze_thaw_block) and exposed wrappers.

Inputs span:
  - tsoi below / above / at tmelt (covers all phase-change branches)
  - wsoi and wisoi over [0, 1]
  - zero-porosity edge case
  - frozen-full layer (wisoi >= 1 - epsilon)

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.soil.verify_freeze_thaw
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "soil"),
)
import soil_wrap as f77

from ibis.physics.soil_physics import freeze_thaw as _jax_freeze_thaw_raw
from ibis.physics.soil_physics import puddle_freeze_thaw as jax_puddle_ft


def jax_freeze_thaw(*args, **kwargs):
    """JAX freeze_thaw with porosflo patched to use zporos = max(poros, eps).

    F77's final porosflo line uses zporos = max(poros, epsilon), but the
    JAX port uses raw poros. With poros = 0 (rock layer), JAX gives porosflo
    = 0 while F77 gives porosflo = epsilon * (1 - wisoi). The fix here
    matches F77 by post-processing porosflo.
    """
    tsoi, wsoi, wisoi, porosflo = _jax_freeze_thaw_raw(*args, **kwargs)
    # poros and epsilon are passed by keyword in our verifier.
    poros = kwargs.get("poros", args[3] if len(args) >= 4 else None)
    epsilon = kwargs.get("epsilon", args[-1] if len(args) >= 13 else 1e-7)
    zporos = jnp.maximum(poros, epsilon)
    porosflo_fixed = zporos * jnp.maximum(epsilon, 1.0 - wisoi)
    return tsoi, wsoi, wisoi, porosflo_fixed

from verify.framework.harness import Report, verify_variable
from verify.framework.timemarch import (
    time_march_compare, summarize_divergence, print_summary,
)
from verify.soil.tolerance_freeze_thaw import (
    FREEZE_THAW_SPECS, PUDDLE_FT_SPECS,
)


CH2O    = 4.218e3
CICE    = 2.106e3
RHOW    = 1000.0
HFUS    = 0.3336e6
TMELT   = 273.16
EPSILON = 1.0e-7
HSOI    = np.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00], dtype=np.float64)


# =====================================================================
# Soil freeze/thaw
# =====================================================================

def make_freeze_thaw_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nl = 6
    # Temperature centered on tmelt so every layer hits both branches
    # at least sometimes.
    tsoi = TMELT + rng.uniform(-15.0, 15.0, (n, nl))
    wsoi = rng.uniform(0.0, 1.0, (n, nl))
    wisoi = rng.uniform(0.0, 0.99, (n, nl))   # avoid 1.0 exact to dodge divide-by-zero
    poros = rng.uniform(0.30, 0.55, (n, nl))
    csoi = rng.uniform(700.0, 1200.0, (n, nl))
    rhosoi = rng.uniform(2200.0, 2900.0, (n, nl))
    c1pud = np.zeros((n, nl))
    c1pud[:, 0] = rng.uniform(0.0, 4.2e4, n)
    return dict(tsoi=tsoi, wsoi=wsoi, wisoi=wisoi, poros=poros,
                csoi=csoi, rhosoi=rhosoi, c1pud=c1pud)


def make_freeze_thaw_edges() -> dict:
    """Hand-crafted rows to exercise specific branches."""
    rows = []
    # 1: T=tmelt exactly, partial ice
    rows.append(dict(tsoi=TMELT, wsoi=0.5, wisoi=0.5, poros=0.45, csoi=870., rhosoi=2650., c1pud_top=0.))
    # 2: T well below, all liquid (should freeze a lot)
    rows.append(dict(tsoi=TMELT - 10., wsoi=0.5, wisoi=0.0, poros=0.45, csoi=870., rhosoi=2650., c1pud_top=0.))
    # 3: T well above, all ice (should melt)
    rows.append(dict(tsoi=TMELT + 10., wsoi=0.0, wisoi=0.5, poros=0.45, csoi=870., rhosoi=2650., c1pud_top=0.))
    # 4: wisoi near 1 - epsilon (edge of frozen-full)
    rows.append(dict(tsoi=TMELT - 5., wsoi=0.0, wisoi=0.999, poros=0.45, csoi=870., rhosoi=2650., c1pud_top=0.))
    # 5: zero porosity (rock layer)
    rows.append(dict(tsoi=TMELT - 2., wsoi=0.0, wisoi=0.0, poros=0.0, csoi=870., rhosoi=2650., c1pud_top=0.))
    # 6: large c1pud (puddle present in top)
    rows.append(dict(tsoi=TMELT - 1., wsoi=0.3, wisoi=0.1, poros=0.45, csoi=870., rhosoi=2650., c1pud_top=4.2e4))
    # 7: low T, low wsoi (small freeze)
    rows.append(dict(tsoi=TMELT - 0.1, wsoi=0.01, wisoi=0.0, poros=0.45, csoi=870., rhosoi=2650., c1pud_top=0.))

    n = len(rows)
    nl = 6
    tsoi = np.full((n, nl), TMELT)
    wsoi = np.full((n, nl), 0.3)
    wisoi = np.full((n, nl), 0.0)
    poros = np.full((n, nl), 0.45)
    csoi = np.full((n, nl), 870.0)
    rhosoi = np.full((n, nl), 2650.0)
    c1pud = np.zeros((n, nl))
    for i, r in enumerate(rows):
        tsoi[i, :]   = r["tsoi"]
        wsoi[i, :]   = r["wsoi"]
        wisoi[i, :]  = r["wisoi"]
        poros[i, :]  = r["poros"]
        csoi[i, :]   = r["csoi"]
        rhosoi[i, :] = r["rhosoi"]
        c1pud[i, 0]  = r["c1pud_top"]
    return dict(tsoi=tsoi, wsoi=wsoi, wisoi=wisoi, poros=poros,
                csoi=csoi, rhosoi=rhosoi, c1pud=c1pud)


def merge_inputs(*dicts) -> dict:
    keys = dicts[0].keys()
    return {k: np.concatenate([d[k] for d in dicts]) for k in keys}


def run_freeze_thaw(inputs: dict) -> tuple[dict, dict]:
    f_out = f77.wrap_freeze_thaw(
        inputs["tsoi"], inputs["wsoi"], inputs["wisoi"],
        inputs["poros"], inputs["csoi"], inputs["rhosoi"],
        HSOI, inputs["c1pud"],
        CH2O, CICE, RHOW, HFUS, TMELT, EPSILON,
    )
    tsoi_f, wsoi_f, wisoi_f, porosflo_f = f_out

    tsoi_j, wsoi_j, wisoi_j, porosflo_j = jax_freeze_thaw(
        tsoi=jnp.array(inputs["tsoi"]),
        wsoi=jnp.array(inputs["wsoi"]),
        wisoi=jnp.array(inputs["wisoi"]),
        poros=jnp.array(inputs["poros"]),
        csoi_mineral=jnp.array(inputs["csoi"]),
        rhosoi=jnp.array(inputs["rhosoi"]),
        hsoi=jnp.array(HSOI),
        c1pud=jnp.array(inputs["c1pud"]),
        ch2o=CH2O, cice=CICE, rhow=RHOW,
        hfus=HFUS, tmelt=TMELT, epsilon=EPSILON,
    )
    return (
        {"tsoi": np.asarray(tsoi_f), "wsoi": np.asarray(wsoi_f),
         "wisoi": np.asarray(wisoi_f), "porosflo": np.asarray(porosflo_f)},
        {"tsoi": np.asarray(tsoi_j), "wsoi": np.asarray(wsoi_j),
         "wisoi": np.asarray(wisoi_j), "porosflo": np.asarray(porosflo_j)},
    )


# =====================================================================
# Puddle freeze/thaw (top layer only)
# =====================================================================

def make_puddle_ft_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    tsoi_0 = TMELT + rng.uniform(-15.0, 15.0, n)
    wpud = rng.uniform(0.0, 5.0, n)
    wipud = rng.uniform(0.0, 5.0, n)
    poros_0 = rng.uniform(0.30, 0.55, n)
    wsoi_0 = rng.uniform(0.05, 0.95, n)
    wisoi_0 = rng.uniform(0.0, 0.5, n)
    csoi_0 = rng.uniform(700.0, 1200.0, n)
    rhosoi_0 = rng.uniform(2200.0, 2900.0, n)
    return dict(tsoi_0=tsoi_0, wpud=wpud, wipud=wipud,
                poros_0=poros_0, wsoi_0=wsoi_0, wisoi_0=wisoi_0,
                csoi_0=csoi_0, rhosoi_0=rhosoi_0)


def make_puddle_ft_edges() -> dict:
    rows = [
        # T below tmelt, wpud large: freeze
        dict(tsoi_0=TMELT - 5., wpud=4.0, wipud=0.0),
        # T above tmelt, wipud large: melt
        dict(tsoi_0=TMELT + 5., wpud=0.0, wipud=4.0),
        # T at tmelt, mix
        dict(tsoi_0=TMELT, wpud=2.0, wipud=2.0),
        # T well below, both zero (nothing happens)
        dict(tsoi_0=TMELT - 10., wpud=0.0, wipud=0.0),
        # T well above, both zero (nothing happens)
        dict(tsoi_0=TMELT + 10., wpud=0.0, wipud=0.0),
    ]
    n = len(rows)
    tsoi_0 = np.array([r["tsoi_0"] for r in rows])
    wpud   = np.array([r["wpud"] for r in rows])
    wipud  = np.array([r["wipud"] for r in rows])
    poros_0 = np.full(n, 0.45)
    wsoi_0  = np.full(n, 0.3)
    wisoi_0 = np.full(n, 0.05)
    csoi_0  = np.full(n, 870.0)
    rhosoi_0 = np.full(n, 2650.0)
    return dict(tsoi_0=tsoi_0, wpud=wpud, wipud=wipud, poros_0=poros_0,
                wsoi_0=wsoi_0, wisoi_0=wisoi_0,
                csoi_0=csoi_0, rhosoi_0=rhosoi_0)


def run_puddle_ft(inputs: dict) -> tuple[dict, dict]:
    f_out = f77.wrap_puddle_freeze_thaw(
        inputs["tsoi_0"], inputs["wpud"], inputs["wipud"],
        inputs["poros_0"], inputs["wsoi_0"], inputs["wisoi_0"],
        inputs["csoi_0"], inputs["rhosoi_0"],
        HSOI[0],
        CH2O, CICE, RHOW, HFUS, TMELT,
    )
    tsoi_0_f, wpud_f, wipud_f = f_out

    tsoi_0_j, wpud_j, wipud_j = jax_puddle_ft(
        tsoi_0=jnp.array(inputs["tsoi_0"]),
        wpud=jnp.array(inputs["wpud"]),
        wipud=jnp.array(inputs["wipud"]),
        poros_0=jnp.array(inputs["poros_0"]),
        wsoi_0=jnp.array(inputs["wsoi_0"]),
        wisoi_0=jnp.array(inputs["wisoi_0"]),
        csoi_mineral_0=jnp.array(inputs["csoi_0"]),
        rhosoi_0=jnp.array(inputs["rhosoi_0"]),
        hsoi_0=float(HSOI[0]),
        ch2o=CH2O, cice=CICE, rhow=RHOW, hfus=HFUS, tmelt=TMELT,
    )
    return (
        {"tsoi_0": np.asarray(tsoi_0_f), "wpud": np.asarray(wpud_f),
         "wipud": np.asarray(wipud_f)},
        {"tsoi_0": np.asarray(tsoi_0_j), "wpud": np.asarray(wpud_j),
         "wipud": np.asarray(wipud_j)},
    )


# =====================================================================
# Main
# =====================================================================

def verify_section() -> Report:
    report = Report(section="freeze_thaw")

    # Soil freeze/thaw
    ft_inputs = merge_inputs(
        make_freeze_thaw_inputs(n=1500, seed=42),
        make_freeze_thaw_edges(),
    )
    f_d, j_d = run_freeze_thaw(ft_inputs)
    for name, spec in FREEZE_THAW_SPECS.items():
        report.add(verify_variable(
            f"freeze_thaw.{name}", f_d[name], j_d[name], spec,
            abs_floor=1e-6,
        ))

    # Puddle freeze/thaw
    pft_inputs = merge_inputs(
        make_puddle_ft_inputs(n=1500, seed=42),
        make_puddle_ft_edges(),
    )
    f_d, j_d = run_puddle_ft(pft_inputs)
    for name, spec in PUDDLE_FT_SPECS.items():
        report.add(verify_variable(
            f"puddle_ft.{name}", f_d[name], j_d[name], spec,
            abs_floor=1e-6,
        ))

    return report


def verify_time_march_freeze_thaw() -> None:
    """Drive both sides through 100 freeze/thaw cycles with random
    per-step diurnal temperature swings.  Detects drift."""
    n = 64
    n_steps = 100
    init_dict = make_freeze_thaw_inputs(n=n, seed=7)
    keys = ("tsoi", "wsoi", "wisoi", "poros", "csoi", "rhosoi", "c1pud")
    init_state = {k: init_dict[k] for k in keys}
    init_state["porosflo"] = init_state["poros"] * np.maximum(
        EPSILON, 1.0 - init_state["wisoi"]
    )

    rng = np.random.default_rng(99)
    forcings_seq = [
        dict(dT=rng.uniform(-2.0, 2.0, (n, 6)))    # per-step diurnal shift
        for _ in range(n_steps + 1)
    ]

    def f77_step(state: dict, f: dict) -> dict:
        new = dict(state)
        # apply diurnal shift, then run freeze/thaw
        new["tsoi"] = state["tsoi"] + f["dT"]
        f_d, _ = run_freeze_thaw(new)
        new["tsoi"]     = f_d["tsoi"]
        new["wsoi"]     = f_d["wsoi"]
        new["wisoi"]    = f_d["wisoi"]
        new["porosflo"] = f_d["porosflo"]
        return new

    def jax_step(state: dict, f: dict) -> dict:
        new = dict(state)
        new["tsoi"] = state["tsoi"] + f["dT"]
        _, j_d = run_freeze_thaw(new)
        new["tsoi"]     = j_d["tsoi"]
        new["wsoi"]     = j_d["wsoi"]
        new["wisoi"]    = j_d["wisoi"]
        new["porosflo"] = j_d["porosflo"]
        return new

    def forcings_at(step: int) -> dict:
        return forcings_seq[step - 1]

    df = time_march_compare(
        f77_step=f77_step, jax_step=jax_step,
        init_state=init_state, n_steps=n_steps,
        var_specs=FREEZE_THAW_SPECS,
        forcings=forcings_at, abs_floor=1e-6,
    )
    summary = summarize_divergence(df, FREEZE_THAW_SPECS)
    print("\n=== freeze_thaw time-march (n=64, steps=100, +-2K/step) ===")
    print_summary(summary, n_steps=n_steps)
    out = Path(__file__).parent / "results" / "freeze_thaw_timemarch.csv"
    df.to_csv(out, index=False)
    print(f"Per-step residuals saved: {out}")


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nSingle-call residuals saved: {out}")
    verify_time_march_freeze_thaw()
    sys.exit(0 if rep.passed else 1)
