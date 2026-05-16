"""Verify JAX drystress against F77 drystress.

drystress computes soil moisture stress for upper and lower canopy from
per-layer wsoi/wisoi vs swilt (wilting point) and sfield (field capacity),
weighted by per-layer root fraction.

Note: JAX guards `(sfield - swilt)` with a 1e-10 floor; F77 doesn't.  For
typical IBIS parameters sfield > swilt always, so no divergence.  We test
realistic inputs only.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.physiology.verify_drystress
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
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "physiology"),
)
import physiology_wrap as f77

from ibis.physics.stomata import drystress as jax_drystress

from verify.framework.harness import Report, verify_variable
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


DRYSTRESS_SPECS = {
    "stresstl": ToleranceSpec("stresstl", TolClass.TIGHT,
                               abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                               why="Sum over layers of clamped sigmoidal awc response."),
    "stresstu": ToleranceSpec("stresstu", TolClass.TIGHT,
                               abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                               why="Same as stresstl, upper canopy."),
    "stressl": ToleranceSpec("stressl", TolClass.TIGHT,
                              abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                              why="Per-layer stress, root-fraction-weighted."),
    "stressu": ToleranceSpec("stressu", TolClass.TIGHT,
                              abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13,
                              why="Per-layer stress, upper canopy."),
}


def make_inputs(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nl = 6
    wsoi = rng.uniform(0.0, 1.0, (n, nl))
    wisoi = rng.uniform(0.0, 0.5, (n, nl))
    swilt = rng.uniform(0.10, 0.20, (n, nl))
    # Ensure sfield > swilt (realistic constraint)
    sfield = swilt + rng.uniform(0.05, 0.25, (n, nl))
    # Root fraction: monotonically decreasing with depth, sum near 1.
    fr_template = np.array([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])
    froot = np.stack([fr_template, fr_template], axis=1)  # (nl, 2)
    return dict(wsoi=wsoi, wisoi=wisoi, swilt=swilt, sfield=sfield, froot=froot)


def make_edge_cases() -> dict:
    """Hit boundary cases: dry to wilting, saturated, frozen, etc."""
    rows = []
    # 1: at wilting point exactly -> awc=0 -> stress=0
    rows.append(dict(wsoi=0.15, wisoi=0.0, swilt=0.15, sfield=0.35))
    # 2: at field capacity -> awc=1 -> stress=1
    rows.append(dict(wsoi=0.35, wisoi=0.0, swilt=0.15, sfield=0.35))
    # 3: below wilting -> awc clipped to 0
    rows.append(dict(wsoi=0.05, wisoi=0.0, swilt=0.15, sfield=0.35))
    # 4: above field capacity -> awc clipped to 1
    rows.append(dict(wsoi=0.50, wisoi=0.0, swilt=0.15, sfield=0.35))
    # 5: half-ice, modest liquid
    rows.append(dict(wsoi=0.50, wisoi=0.5, swilt=0.15, sfield=0.35))
    # 6: fully frozen -> awc clipped to 0
    rows.append(dict(wsoi=0.00, wisoi=1.0, swilt=0.15, sfield=0.35))

    n = len(rows)
    nl = 6
    wsoi = np.zeros((n, nl))
    wisoi = np.zeros((n, nl))
    swilt = np.zeros((n, nl))
    sfield = np.zeros((n, nl))
    for i, r in enumerate(rows):
        wsoi[i, :]   = r["wsoi"]
        wisoi[i, :]  = r["wisoi"]
        swilt[i, :]  = r["swilt"]
        sfield[i, :] = r["sfield"]
    fr_template = np.array([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])
    froot = np.stack([fr_template, fr_template], axis=1)
    return dict(wsoi=wsoi, wisoi=wisoi, swilt=swilt, sfield=sfield, froot=froot)


def run_drystress(inputs: dict) -> tuple[dict, dict]:
    f_out = f77.wrap_drystress(
        inputs["wsoi"], inputs["wisoi"],
        inputs["swilt"], inputs["sfield"], inputs["froot"],
    )
    stresstl_f, stresstu_f, stressl_f, stressu_f = f_out

    stresstl_j, stresstu_j, stressl_j, stressu_j = jax_drystress(
        wsoi=jnp.array(inputs["wsoi"]),
        wisoi=jnp.array(inputs["wisoi"]),
        swilt=jnp.array(inputs["swilt"]),
        sfield=jnp.array(inputs["sfield"]),
        froot=jnp.array(inputs["froot"]),
    )

    return (
        {"stresstl": np.asarray(stresstl_f), "stresstu": np.asarray(stresstu_f),
         "stressl": np.asarray(stressl_f), "stressu": np.asarray(stressu_f)},
        {"stresstl": np.asarray(stresstl_j), "stresstu": np.asarray(stresstu_j),
         "stressl": np.asarray(stressl_j), "stressu": np.asarray(stressu_j)},
    )


def verify_section() -> Report:
    report = Report(section="drystress")
    random_in = make_inputs(n=1500, seed=42)
    edges = make_edge_cases()
    inputs = {k: np.concatenate([random_in[k], edges[k]])
              if k != "froot" else random_in[k]
              for k in random_in}
    f_d, j_d = run_drystress(inputs)
    for name, spec in DRYSTRESS_SPECS.items():
        report.add(verify_variable(
            f"drystress.{name}", f_d[name], j_d[name], spec, abs_floor=1e-6,
        ))
    return report


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")
    sys.exit(0 if rep.passed else 1)
