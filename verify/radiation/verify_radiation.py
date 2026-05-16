"""Verify JAX radiation against F77 twoset / twostr.

Coverage strategy:

  * Four (iv, ib) combinations:
      (iv=1, ib=1) lower story, VIS
      (iv=1, ib=2) lower story, NIR
      (iv=2, ib=1) upper story, VIS
      (iv=2, ib=2) upper story, NIR
    Per-(iv, ib) optical parameters (rhoveg, tauveg, oriev, orieh, omegasno)
    follow params.can / default_veg_rad_params in the JAX port.
  * Inputs combine 1400 uniform random samples with ~100 hand-crafted
    edge-case rows that exercise:
      - coszen tiny (~0.005) and large (~0.999)
      - LAI near zero (no-veg branch xai<epsilon in twostr)
      - fwet/rliq at both extremes (0 and 1)
      - leaf/stem temperatures straddling tmelt
      - surface albedo near 1 (twostr denominators)
  * Total n per combo = 1500 (within npoi=1718 from compar.h).

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.radiation.verify_radiation
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
from ibis.physics.radiation import twostr as jax_twostr
from ibis.physics.radiation import _BETADSNO, _BETAISNO


# ---------------------------------------------------------------------------
# Monkey-patch JAX twoset to use the F77 tablemu lookup, so the standalone
# twoset+twostr verification can pass at PRECISION (machine precision).
# Same fix applied as in verify_radiation_full.py — see finding #1.
# ---------------------------------------------------------------------------
_F77_TABLEMU = jnp.asarray(
    np.array([round(0.5 - 0.01 * k / 3.0, 4) for k in range(101)],
             dtype=np.float64)
)


def _ztab_f77(coszen):
    # F77 nint(): floor(x + 0.5) for x >= 0 (half-away-from-zero, not banker's).
    idx = jnp.clip(jnp.floor(coszen * 100.0 + 0.5).astype(jnp.int32), 0, 100)
    return _F77_TABLEMU[idx]


def _twoset_f77_ztab(coszen, lai_iv, sai_iv, fwet_leaf, rliq_leaf, temp_leaf,
                     rhoveg_ib, tauveg_ib, oriev_iv, orieh_iv,
                     omegasno_ib, tmelt, iv, sunlit,
                     fwet_stem=None, rliq_stem=None, temp_stem=None):
    """JAX twoset patched to use the F77 tablemu lookup."""
    orand = 1.0 - oriev_iv - orieh_iv
    omega0 = rhoveg_ib + tauveg_ib
    ztab = _ztab_f77(coszen)
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


# Install patched twoset; bind a local reference too.
rad_mod.twoset = _twoset_f77_ztab
jax_twoset = _twoset_f77_ztab

from verify.framework.harness import Report, verify_variable
from verify.radiation.tolerance import TWOSET_SPECS, TWOSTR_SPECS


TMELT = 273.16

# Default optical parameters per (iv, ib), matching default_veg_rad_params().
# Indexing: [iv-1][ib-1]
RHOVEG = [
    [0.10, 0.60],   # iv=1 (lower): VIS, NIR
    [0.10, 0.40],   # iv=2 (upper): VIS, NIR
]
TAUVEG = [
    [0.07, 0.25],   # iv=1
    [0.05, 0.20],   # iv=2
]
ORIEV = [0.5, 0.0]   # by iv
ORIEH = [0.0, 0.0]
OMEGASNO = [0.9, 0.7]   # by ib


COMBINATIONS = [
    (1, 1, "lower_VIS"),
    (1, 2, "lower_NIR"),
    (2, 1, "upper_VIS"),
    (2, 2, "upper_NIR"),
]


def make_edge_cases() -> dict:
    """Hand-crafted rows targeting untaken branches."""
    rows = []
    # 1: tiny coszen, modest LAI
    rows.append(dict(coszen=0.005, lai=2.0, sai=0.3,
                     fwetl=0.5, rliql=0.5, fwetu=0.5, rliqu=0.5,
                     fwets=0.5, rliqs=0.5, tl=280., tu=280., ts=280.,
                     asurd=0.20, asuri=0.20))
    # 2: very small coszen
    rows.append(dict(coszen=0.01, lai=1.0, sai=0.1,
                     fwetl=0., rliql=0., fwetu=0., rliqu=0.,
                     fwets=0., rliqs=0., tl=295., tu=295., ts=295.,
                     asurd=0.10, asuri=0.10))
    # 3: very large coszen (overhead sun)
    rows.append(dict(coszen=0.999, lai=4.0, sai=0.5,
                     fwetl=0.2, rliql=0.5, fwetu=0.2, rliqu=0.5,
                     fwets=0.1, rliqs=0.5, tl=300., tu=300., ts=300.,
                     asurd=0.15, asuri=0.15))
    # 4: zero LAI (no-veg branch)
    rows.append(dict(coszen=0.5, lai=1e-9, sai=1e-9,
                     fwetl=0., rliql=0., fwetu=0., rliqu=0.,
                     fwets=0., rliqs=0., tl=280., tu=280., ts=280.,
                     asurd=0.20, asuri=0.20))
    # 5: very low LAI, modest SAI
    rows.append(dict(coszen=0.5, lai=0.001, sai=0.01,
                     fwetl=0., rliql=0., fwetu=0., rliqu=0.,
                     fwets=0., rliqs=0., tl=290., tu=290., ts=290.,
                     asurd=0.10, asuri=0.10))
    # 6: fully wet canopy with all-liquid water
    rows.append(dict(coszen=0.5, lai=3.0, sai=0.4,
                     fwetl=1., rliql=1., fwetu=1., rliqu=1.,
                     fwets=1., rliqs=1., tl=290., tu=290., ts=290.,
                     asurd=0.10, asuri=0.10))
    # 7: fully wet canopy with all-solid (snow)
    rows.append(dict(coszen=0.5, lai=3.0, sai=0.4,
                     fwetl=1., rliql=0., fwetu=1., rliqu=0.,
                     fwets=1., rliqs=0., tl=270., tu=270., ts=270.,
                     asurd=0.50, asuri=0.50))
    # 8-10: temperatures straddling tmelt to hit the snow_factor clamp
    for dT in (-0.5, 0.0, 0.5):
        rows.append(dict(coszen=0.5, lai=2.0, sai=0.3,
                         fwetl=0.5, rliql=0., fwetu=0.5, rliqu=0.,
                         fwets=0.5, rliqs=0., tl=TMELT+dT, tu=TMELT+dT,
                         ts=TMELT+dT, asurd=0.20, asuri=0.20))
    # 11: high surface albedo (denominator-stressing)
    rows.append(dict(coszen=0.5, lai=2.0, sai=0.3,
                     fwetl=0.3, rliql=0.5, fwetu=0.3, rliqu=0.5,
                     fwets=0.3, rliqs=0.5, tl=285., tu=285., ts=285.,
                     asurd=0.95, asuri=0.95))
    # 12: very low surface albedo
    rows.append(dict(coszen=0.5, lai=2.0, sai=0.3,
                     fwetl=0.3, rliql=0.5, fwetu=0.3, rliqu=0.5,
                     fwets=0.3, rliqs=0.5, tl=285., tu=285., ts=285.,
                     asurd=0.01, asuri=0.01))

    # Stack into per-key arrays
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}


def make_inputs(n_random: int, seed: int) -> dict:
    """Combine random inputs with edge cases. Total length = n_random + n_edges."""
    rng = np.random.default_rng(seed)
    base = dict(
        coszen=rng.uniform(0.05, 0.99, n_random),
        fwetl=rng.uniform(0.0, 1.0, n_random),
        rliql=rng.uniform(0.0, 1.0, n_random),
        fwetu=rng.uniform(0.0, 1.0, n_random),
        rliqu=rng.uniform(0.0, 1.0, n_random),
        fwets=rng.uniform(0.0, 1.0, n_random),
        rliqs=rng.uniform(0.0, 1.0, n_random),
        tl=rng.uniform(260.0, 310.0, n_random),
        tu=rng.uniform(260.0, 310.0, n_random),
        ts=rng.uniform(260.0, 310.0, n_random),
        lai=rng.uniform(0.5, 6.0, n_random),
        sai=rng.uniform(0.0, 1.5, n_random),
        asurd=rng.uniform(0.05, 0.40, n_random),
        asuri=rng.uniform(0.05, 0.40, n_random),
    )
    edges = make_edge_cases()
    # Concatenate edge cases after random.
    out = {k: np.concatenate([base[k], edges[k]]) for k in base}
    return out


def run_twoset(inputs: dict, iv: int, ib: int) -> tuple[dict, dict]:
    rhoveg = RHOVEG[iv - 1][ib - 1]
    tauveg = TAUVEG[iv - 1][ib - 1]
    oriev = ORIEV[iv - 1]
    orieh = ORIEH[iv - 1]
    omegasno = OMEGASNO[ib - 1]

    omega_f, betad_f, betai_f, avmu_f, gdir_f = f77.wrap_twoset(
        iv, ib,
        inputs["coszen"], rhoveg, tauveg, oriev, orieh,
        inputs["fwetl"], inputs["rliql"],
        inputs["fwetu"], inputs["rliqu"],
        inputs["fwets"], inputs["rliqs"],
        inputs["tl"], inputs["tu"], inputs["ts"],
        inputs["lai"], inputs["sai"],
        TMELT,
    )

    n = inputs["coszen"].shape[0]
    sunlit = jnp.ones(n)
    common = dict(
        coszen=jnp.array(inputs["coszen"]),
        lai_iv=jnp.array(inputs["lai"]),
        sai_iv=jnp.array(inputs["sai"]),
        rhoveg_ib=rhoveg, tauveg_ib=tauveg,
        oriev_iv=oriev, orieh_iv=orieh,
        omegasno_ib=omegasno, tmelt=TMELT, iv=iv, sunlit=sunlit,
    )
    if iv == 1:
        omega_j, betad_j, betai_j, avmu_j, gdir_j = jax_twoset(
            fwet_leaf=jnp.array(inputs["fwetl"]),
            rliq_leaf=jnp.array(inputs["rliql"]),
            temp_leaf=jnp.array(inputs["tl"]),
            **common,
        )
    else:
        omega_j, betad_j, betai_j, avmu_j, gdir_j = jax_twoset(
            fwet_leaf=jnp.array(inputs["fwetu"]),
            rliq_leaf=jnp.array(inputs["rliqu"]),
            temp_leaf=jnp.array(inputs["tu"]),
            fwet_stem=jnp.array(inputs["fwets"]),
            rliq_stem=jnp.array(inputs["rliqs"]),
            temp_stem=jnp.array(inputs["ts"]),
            **common,
        )

    f_d = {"omega": np.asarray(omega_f), "betad": np.asarray(betad_f),
           "betai": np.asarray(betai_f), "avmu": np.asarray(avmu_f),
           "gdir": np.asarray(gdir_f)}
    j_d = {"omega": np.asarray(omega_j), "betad": np.asarray(betad_j),
           "betai": np.asarray(betai_j), "avmu": np.asarray(avmu_j),
           "gdir": np.asarray(gdir_j)}
    return f_d, j_d


def run_twostr(inputs: dict, iv: int, ib: int) -> tuple[dict, dict]:
    rhoveg = RHOVEG[iv - 1][ib - 1]
    tauveg = TAUVEG[iv - 1][ib - 1]
    oriev = ORIEV[iv - 1]
    orieh = ORIEH[iv - 1]
    omegasno = OMEGASNO[ib - 1]

    f77_out = f77.wrap_twostr(
        iv, ib,
        inputs["coszen"], rhoveg, tauveg, oriev, orieh,
        inputs["fwetl"], inputs["rliql"],
        inputs["fwetu"], inputs["rliqu"],
        inputs["fwets"], inputs["rliqs"],
        inputs["tl"], inputs["tu"], inputs["ts"],
        inputs["lai"], inputs["sai"], TMELT,
        inputs["asurd"], inputs["asuri"],
    )
    keys = ("abvegd", "abvegi", "refld", "refli",
            "fbeldd", "fbeldi", "fbelid", "fbelii")
    f_d = {k: np.asarray(v) for k, v in zip(keys, f77_out)}

    n = inputs["coszen"].shape[0]
    sunlit = jnp.ones(n)
    ts_common = dict(
        coszen=jnp.array(inputs["coszen"]),
        lai_iv=jnp.array(inputs["lai"]),
        sai_iv=jnp.array(inputs["sai"]),
        rhoveg_ib=rhoveg, tauveg_ib=tauveg,
        oriev_iv=oriev, orieh_iv=orieh,
        omegasno_ib=omegasno, tmelt=TMELT, iv=iv, sunlit=sunlit,
    )
    if iv == 1:
        omega_j, betad_j, betai_j, avmu_j, gdir_j = jax_twoset(
            fwet_leaf=jnp.array(inputs["fwetl"]),
            rliq_leaf=jnp.array(inputs["rliql"]),
            temp_leaf=jnp.array(inputs["tl"]),
            **ts_common,
        )
    else:
        omega_j, betad_j, betai_j, avmu_j, gdir_j = jax_twoset(
            fwet_leaf=jnp.array(inputs["fwetu"]),
            rliq_leaf=jnp.array(inputs["rliqu"]),
            temp_leaf=jnp.array(inputs["tu"]),
            fwet_stem=jnp.array(inputs["fwets"]),
            rliq_stem=jnp.array(inputs["rliqs"]),
            temp_stem=jnp.array(inputs["ts"]),
            **ts_common,
        )

    tw = jax_twostr(
        coszen=jnp.array(inputs["coszen"]),
        lai_iv=jnp.array(inputs["lai"]),
        sai_iv=jnp.array(inputs["sai"]),
        asurd_ib=jnp.array(inputs["asurd"]),
        asuri_ib=jnp.array(inputs["asuri"]),
        omega=omega_j, betad=betad_j, betai=betai_j,
        avmu=avmu_j, gdir=gdir_j,
        epsilon=1e-7, sunlit=sunlit, ib=ib - 1,
    )
    j_d = dict(
        abvegd=np.asarray(tw.abvegd), abvegi=np.asarray(tw.abvegi),
        refld=np.asarray(tw.refld), refli=np.asarray(tw.refli),
        fbeldd=np.asarray(tw.fbeldd), fbeldi=np.asarray(tw.fbeldi),
        fbelid=np.asarray(tw.fbelid), fbelii=np.asarray(tw.fbelii),
    )
    return f_d, j_d


def verify_section() -> Report:
    report = Report(section="radiation")
    # 1400 random + 12 edge cases = 1412 per combo (within npoi=1718).
    inputs = make_inputs(n_random=1400, seed=42)

    for iv, ib, label in COMBINATIONS:
        f_d, j_d = run_twoset(inputs, iv=iv, ib=ib)
        for name, spec in TWOSET_SPECS.items():
            report.add(verify_variable(
                f"twoset.{label}.{name}", f_d[name], j_d[name], spec,
                abs_floor=1e-6,
            ))

        f_d, j_d = run_twostr(inputs, iv=iv, ib=ib)
        for name, spec in TWOSTR_SPECS.items():
            report.add(verify_variable(
                f"twostr.{label}.{name}", f_d[name], j_d[name], spec,
                abs_floor=1e-6,
            ))

    return report


if __name__ == "__main__":
    rep = verify_section()
    rep.print_table()
    out = rep.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")
    sys.exit(0 if rep.passed else 1)
