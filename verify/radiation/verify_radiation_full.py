"""Verify JAX compute_radiation against F77 chain (solset + solsur*nband +
solalb*nband + solarf*nband + irrad).

twostr/twoset are already verified bit-exact standalone. This test exercises
the orchestration that calls them.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.radiation.verify_radiation_full
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
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "radiation"),
)
import radiation_wrap as f77

import ibis.physics.radiation as rad_mod
from ibis.physics.radiation import default_veg_rad_params, compute_radiation
from ibis.physics.radiation import _BETADSNO, _BETAISNO

# ---------------------------------------------------------------------------
# Monkey-patch JAX twoset to use the F77 tablemu lookup. This eliminates
# finding #1 (analytic vs lookup) so the chain can be verified at narrow
# (PRECISION) tolerance.
#
# F77 tablemu values from radiation.f lines 756-769: 101 entries, indexed
# by nint(coszen*100 + 1). Each entry is 0.5 - 0.01*(k-1)/3, i.e. the
# analytic value sampled at coszen = 0.00, 0.01, ..., 1.00.
# ---------------------------------------------------------------------------
# The F77 tablemu literals are stored at 4-decimal precision, e.g. 0.4967
# (not 0.49666666...). Match by rounding to 4 decimals.
_F77_TABLEMU = jnp.asarray(
    np.array([round(0.5 - 0.01 * k / 3.0, 4) for k in range(101)],
             dtype=np.float64)
)


def _ztab_f77(coszen):
    """F77 tablemu lookup: ztab = tablemu[nint(coszen*100)] (0-indexed)."""
    # F77 nint() rounds half-away-from-zero (not banker's rounding!). For
    # coszen >= 0 this is equivalent to floor(x + 0.5). Using jnp.round
    # here is the gotcha-#3 mistake: jnp.round(0.5) = 0, but F77 nint(0.5) = 1.
    idx = jnp.clip(jnp.floor(coszen * 100.0 + 0.5).astype(jnp.int32), 0, 100)
    return _F77_TABLEMU[idx]


def _twoset_f77_ztab(coszen, lai_iv, sai_iv, fwet_leaf, rliq_leaf, temp_leaf,
                     rhoveg_ib, tauveg_ib, oriev_iv, orieh_iv,
                     omegasno_ib, tmelt, iv, sunlit,
                     fwet_stem=None, rliq_stem=None, temp_stem=None):
    """JAX twoset with ztab replaced by the F77 tablemu lookup."""
    orand = 1.0 - oriev_iv - orieh_iv
    omega0 = rhoveg_ib + tauveg_ib

    ztab = _ztab_f77(coszen)        # <-- the only change from JAX twoset
    rwork = 1.0 / jnp.maximum(omega0, 1e-12)

    betad = (oriev_iv * 0.5 * (rhoveg_ib + tauveg_ib)
             + orieh_iv * rhoveg_ib
             + orand * ((1.0 - ztab) * rhoveg_ib + ztab * tauveg_ib)) * rwork
    betai = (oriev_iv * 0.5 * (rhoveg_ib + tauveg_ib)
             + orieh_iv * rhoveg_ib
             + orand * ((2.0 / 3.0) * rhoveg_ib
                        + (1.0 / 3.0) * tauveg_ib)) * rwork

    gdir = (oriev_iv * (2.0 / 3.1415927)
            * jnp.sqrt(jnp.maximum(0.0, 1.0 - coszen ** 2))
            + orieh_iv * coszen
            + orand * 0.5)
    avmu = jnp.ones_like(coszen)

    if iv == 2 and fwet_stem is not None:
        x_up = lai_iv / jnp.maximum(lai_iv + sai_iv, 1e-7)
        y = x_up * (fwet_leaf * (1.0 - rliq_leaf)) + (1.0 - x_up) * (fwet_stem * (1.0 - rliq_stem))
        snow_leaf = jnp.clip((tmelt - temp_leaf) / 0.1, 0.6, 1.0)
        snow_stem = jnp.clip((tmelt - temp_stem) / 0.1, 0.6, 1.0)
        o = (x_up * snow_leaf + (1.0 - x_up) * snow_stem) * omegasno_ib
    else:
        y = fwet_leaf * (1.0 - rliq_leaf)
        snow_factor = 0.6 + 0.4 * jnp.clip((tmelt - temp_leaf) / 1.0, 0.0, 1.0)
        o = omegasno_ib * snow_factor

    rwork_sno = y * o
    omega = (1.0 - y) * omega0 + rwork_sno
    omega = jnp.maximum(omega, 1e-12)
    betad = ((1.0 - y) * omega0 * betad + rwork_sno * _BETADSNO) / omega
    betai = ((1.0 - y) * omega0 * betai + rwork_sno * _BETAISNO) / omega

    omega = jnp.where(sunlit, omega, 0.0)
    betad = jnp.where(sunlit, betad, 0.0)
    betai = jnp.where(sunlit, betai, 0.0)
    gdir  = jnp.where(sunlit, gdir,  0.0)
    avmu  = jnp.where(sunlit, avmu,  1.0)

    return omega, betad, betai, avmu, gdir


# Install the patched twoset into the module namespace so that
# compute_radiation -> solalb -> twoset resolves to our patched version.
rad_mod.twoset = _twoset_f77_ztab

from verify.framework.harness import Report, verify_variable
from verify.framework.tolerance_classes import ToleranceSpec, TolClass


STEF    = 5.67e-8
TMELT   = 273.16
EPSILON = 1.0e-12

VP = default_veg_rad_params()
RHOVEG = np.asarray(VP.rhoveg, dtype=np.float64)   # [nband, 2]
TAUVEG = np.asarray(VP.tauveg, dtype=np.float64)
ORIEV  = np.asarray(VP.oriev,  dtype=np.float64)   # [2]
ORIEH  = np.asarray(VP.orieh,  dtype=np.float64)


# With the JAX twoset monkey-patched to use the F77 tablemu lookup,
# all radiation chain outputs should now be at machine precision.
SPECS = {
    "solu":  ToleranceSpec("solu",  TolClass.PRECISION, abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
    "sols":  ToleranceSpec("sols",  TolClass.PRECISION, abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
    "soll":  ToleranceSpec("soll",  TolClass.PRECISION, abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
    "solg":  ToleranceSpec("solg",  TolClass.PRECISION, abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
    "soli":  ToleranceSpec("soli",  TolClass.PRECISION, abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
    "asurd": ToleranceSpec("asurd", TolClass.PRECISION, abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14),
    "asuri": ToleranceSpec("asuri", TolClass.PRECISION, abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14),
    "topparu":  ToleranceSpec("topparu",  TolClass.PRECISION, abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
    "topparl":  ToleranceSpec("topparl",  TolClass.PRECISION, abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
    "scalcoefu": ToleranceSpec("scalcoefu", TolClass.PRECISION, abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
    "scalcoefl": ToleranceSpec("scalcoefl", TolClass.PRECISION, abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
    "firu":  ToleranceSpec("firu",  TolClass.PRECISION, abs_tol=1e-9,  rel_tol=1e-12, bias_tol=1e-10),
    "firs":  ToleranceSpec("firs",  TolClass.PRECISION, abs_tol=1e-9,  rel_tol=1e-12, bias_tol=1e-10),
    "firl":  ToleranceSpec("firl",  TolClass.PRECISION, abs_tol=1e-9,  rel_tol=1e-12, bias_tol=1e-10),
    "firg":  ToleranceSpec("firg",  TolClass.PRECISION, abs_tol=1e-9,  rel_tol=1e-12, bias_tol=1e-10),
    "firi":  ToleranceSpec("firi",  TolClass.PRECISION, abs_tol=1e-9,  rel_tol=1e-12, bias_tol=1e-10),
    "firb":  ToleranceSpec("firb",  TolClass.PRECISION, abs_tol=1e-9,  rel_tol=1e-12, bias_tol=1e-10),
}


def make_inputs(n: int, seed: int, coszen_floor: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    coszen = rng.uniform(coszen_floor, 1.0, n)
    return dict(
        coszen=coszen,
        solad=rng.uniform(0.0, 500.0, (n, 2)),
        solai=rng.uniform(0.0, 200.0, (n, 2)),
        fira=rng.uniform(200.0, 400.0, n),
        lai=rng.uniform(0.0, 6.0, (n, 2)),
        sai=rng.uniform(0.0, 3.0, (n, 2)),
        fu=rng.uniform(0.1, 0.95, n),
        fl=rng.uniform(0.1, 0.95, n),
        fi=rng.uniform(0.0, 0.7, n),
        tu=rng.uniform(265.0, 305.0, n),
        ts=rng.uniform(265.0, 305.0, n),
        tl=rng.uniform(265.0, 305.0, n),
        tg=rng.uniform(265.0, 305.0, n),
        ti=rng.uniform(255.0, 273.16, n),
        tsno1=rng.uniform(240.0, 273.16, n),
        wsoi1=rng.uniform(0.05, 0.6, n),
        wisoi1=rng.uniform(0.0, 0.3, n),
        albsav=rng.uniform(0.05, 0.25, n),
        albsan=rng.uniform(0.15, 0.40, n),
        fwetu=rng.uniform(0.0, 0.25, n),
        fwets=rng.uniform(0.0, 0.25, n),
        fwetl=rng.uniform(0.0, 0.25, n),
        rliqu=rng.uniform(0.0, 1.0, n),
        rliqs=rng.uniform(0.0, 1.0, n),
        rliql=rng.uniform(0.0, 1.0, n),
    )


def call_f77(inp: dict) -> dict:
    out = f77.wrap_radiation_full(
        inp["coszen"], inp["solad"], inp["solai"], inp["fira"],
        inp["lai"], inp["sai"], inp["fu"], inp["fl"], inp["fi"],
        inp["tu"], inp["ts"], inp["tl"], inp["tg"], inp["ti"],
        inp["tsno1"],
        inp["wsoi1"], inp["wisoi1"], inp["albsav"], inp["albsan"],
        inp["fwetu"], inp["fwets"], inp["fwetl"],
        inp["rliqu"], inp["rliqs"], inp["rliql"],
        RHOVEG, TAUVEG, ORIEV, ORIEH,
        STEF, TMELT, EPSILON,
    )
    keys = ("solu", "sols", "soll", "solg", "soli",
            "firu", "firs", "firl", "firg", "firi", "firb",
            "asurd", "asuri", "topparu", "topparl",
            "scalcoefu", "scalcoefl")
    return {k: np.asarray(v) for k, v in zip(keys, out)}


def call_jax(inp: dict) -> dict:
    res = compute_radiation(
        coszen=jnp.asarray(inp["coszen"]),
        solad=jnp.asarray(inp["solad"]), solai=jnp.asarray(inp["solai"]),
        fira=jnp.asarray(inp["fira"]),
        lai=jnp.asarray(inp["lai"]), sai=jnp.asarray(inp["sai"]),
        fu=jnp.asarray(inp["fu"]), fl=jnp.asarray(inp["fl"]),
        tu=jnp.asarray(inp["tu"]), ts=jnp.asarray(inp["ts"]),
        tl=jnp.asarray(inp["tl"]), tg=jnp.asarray(inp["tg"]),
        ti=jnp.asarray(inp["ti"]),
        fi=jnp.asarray(inp["fi"]),
        tsno_1=jnp.asarray(inp["tsno1"]),
        wsoi_1=jnp.asarray(inp["wsoi1"]),
        wisoi_1=jnp.asarray(inp["wisoi1"]),
        albsav=jnp.asarray(inp["albsav"]),
        albsan=jnp.asarray(inp["albsan"]),
        fwetu=jnp.asarray(inp["fwetu"]),
        fwets=jnp.asarray(inp["fwets"]),
        fwetl=jnp.asarray(inp["fwetl"]),
        rliqu=jnp.asarray(inp["rliqu"]),
        rliqs=jnp.asarray(inp["rliqs"]),
        rliql=jnp.asarray(inp["rliql"]),
        veg_params=VP, stef=STEF, tmelt=TMELT, epsilon=EPSILON,
    )
    return dict(
        solu=np.asarray(res.solu), sols=np.asarray(res.sols),
        soll=np.asarray(res.soll), solg=np.asarray(res.solg),
        soli=np.asarray(res.soli),
        firu=np.asarray(res.firu), firs=np.asarray(res.firs),
        firl=np.asarray(res.firl), firg=np.asarray(res.firg),
        firi=np.asarray(res.firi), firb=np.asarray(res.firb),
        asurd=np.asarray(res.asurd), asuri=np.asarray(res.asuri),
        topparu=np.asarray(res.topparu), topparl=np.asarray(res.topparl),
        scalcoefu=np.asarray(res.scalcoefu),
        scalcoefl=np.asarray(res.scalcoefl),
    )


def verify_one(report: Report, label: str, n: int, seed: int,
               coszen_floor: float = 0.05) -> None:
    inp = make_inputs(n, seed, coszen_floor)
    f = call_f77(inp); j = call_jax(inp)
    for name, spec in SPECS.items():
        report.add(verify_variable(
            f"rad.{name}.{label}", f[name], j[name], spec, abs_floor=1e-6,
        ))


def main() -> int:
    rep = Report(section="radiation_chain")
    verify_one(rep, "daytime", n=1500, seed=42, coszen_floor=0.05)
    verify_one(rep, "highsun", n=1500, seed=99, coszen_floor=0.5)
    verify_one(rep, "lowsun",  n=1500, seed=7,  coszen_floor=0.05)

    # Edge case: all coszen == 0 (night) — exercises solset's early-return
    # branch (`if (nsol.eq.0) return`) and irrad's no-sun path.
    inp_night = make_inputs(100, seed=11, coszen_floor=0.05)
    inp_night["coszen"] = np.zeros_like(inp_night["coszen"])
    f = call_f77(inp_night); j = call_jax(inp_night)
    # IR outputs are independent of coszen; solar should be all zero on both.
    for name in ("firu", "firs", "firl", "firg", "firi", "firb"):
        rep.add(verify_variable(
            f"rad.{name}.night", f[name], j[name], SPECS[name], abs_floor=1e-6,
        ))

    rep.print_table()
    rep.save(Path(__file__).parent / "results")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
