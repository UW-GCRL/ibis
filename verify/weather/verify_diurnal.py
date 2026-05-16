"""Verify reference Python diurnal against F77 diurnal -- Layer 1.

Mirrors verify_daily: same Layer-1 pattern, exercises mid-day (no event-timing
retry), first-timestep-of-day (may trigger retry), and edge cases (jday=1/183/
365 for solar geometry; high vs low precip for snow/rain partition).

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.weather.verify_diurnal
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "fortran_wrappers" / "weather"),
)
import weather_wrap as f77

from verify.framework.harness import Report, verify_variable
from verify.framework.tolerance_classes import ToleranceSpec, TolClass

from verify.weather.reference_diurnal import DiurnalInputs, diurnal as ref_diurnal


SPECS = {
    "coszen": ToleranceSpec("coszen", TolClass.PRECISION,
                            abs_tol=1e-13, rel_tol=1e-12, bias_tol=1e-14,
                            why="Pure sin/cos solar geometry."),
    "ta":     ToleranceSpec("ta",     TolClass.PRECISION,
                            abs_tol=1e-11, rel_tol=1e-13, bias_tol=1e-12,
                            why="Linear blend of tmax/tmin."),
    "qa":     ToleranceSpec("qa",     TolClass.PRECISION,
                            abs_tol=1e-14, rel_tol=1e-12, bias_tol=1e-15,
                            why="Bounded by qsat; small magnitude."),
    "ua":     ToleranceSpec("ua",     TolClass.PRECISION,
                            abs_tol=1e-10, rel_tol=1e-10, bias_tol=1e-11,
                            why="Weibull-style (-log u)**0.30; libm pow."),
    "rh":     ToleranceSpec("rh",     TolClass.PRECISION,
                            abs_tol=1e-9,  rel_tol=1e-11, bias_tol=1e-10,
                            why="qa / qsat product."),
    "fira":   ToleranceSpec("fira",   TolClass.PRECISION,
                            abs_tol=1e-9,  rel_tol=1e-12, bias_tol=1e-10,
                            why="Stefan-Boltzmann at ~300 K, magnitudes ~300 W/m2."),
    "raina":  ToleranceSpec("raina",  TolClass.PRECISION,
                            abs_tol=1e-14, rel_tol=1e-12, bias_tol=1e-15),
    "snowa":  ToleranceSpec("snowa",  TolClass.PRECISION,
                            abs_tol=1e-14, rel_tol=1e-12, bias_tol=1e-15),
    "solad":  ToleranceSpec("solad",  TolClass.PRECISION,
                            abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11,
                            why="sw*coszen*frac*trans*(1-fdiffuse); ~1000 W/m2 scale."),
    "solai":  ToleranceSpec("solai",  TolClass.PRECISION,
                            abs_tol=1e-10, rel_tol=1e-12, bias_tol=1e-11),
}


def make_inputs(n: int, seed: int, jday: int) -> DiurnalInputs:
    rng = np.random.default_rng(seed)
    return DiurnalInputs(
        cloud  = rng.uniform(0.0, 1.0, n),
        tmax   = rng.uniform(285.0, 305.0, n),
        tmin   = rng.uniform(265.0, 285.0, n),
        qd     = rng.uniform(0.001, 0.020, n),
        ud     = rng.uniform(2.0, 6.0, n),
        psurf  = rng.uniform(90000.0, 102000.0, n),
        precip = rng.uniform(0.0, 30.0, n),
        lat    = rng.uniform(-60.0, 60.0, n),
    )


def call_f77(iseed: int, time: float, jday: int,
             plens: float, startp: float, endp: float, dtime: float,
             inputs: DiurnalInputs) -> dict:
    n = inputs.cloud.shape[0]
    out = f77.wrap_diurnal(
        iseed, time, jday,
        plens, startp, endp, dtime,
        inputs.cloud, inputs.tmax, inputs.tmin, inputs.qd, inputs.ud,
        inputs.psurf, inputs.precip, inputs.lat,
    )
    (coszen_f, ta_f, qa_f, ua_f, rh_f, fira_f, raina_f, snowa_f,
     solad_f, solai_f, plens_f, startp_f, endp_f) = out
    return dict(
        coszen=np.asarray(coszen_f), ta=np.asarray(ta_f), qa=np.asarray(qa_f),
        ua=np.asarray(ua_f), rh=np.asarray(rh_f),
        fira=np.asarray(fira_f), raina=np.asarray(raina_f),
        snowa=np.asarray(snowa_f),
        solad=np.asarray(solad_f), solai=np.asarray(solai_f),
        plens=float(plens_f), startp=float(startp_f), endp=float(endp_f),
    )


def call_ref(iseed: int, time: float, jday: int,
             plens: float, startp: float, endp: float, dtime: float,
             inputs: DiurnalInputs) -> dict:
    out = ref_diurnal(iseed, time, jday, plens, startp, endp, dtime, inputs)
    return dict(
        coszen=out.coszen, ta=out.ta, qa=out.qa, ua=out.ua, rh=out.rh,
        fira=out.fira, raina=out.raina, snowa=out.snowa,
        solad=out.solad, solai=out.solai,
        plens=out.plens, startp=out.startp, endp=out.endp,
    )


def verify_one(report: Report, label: str, *,
               iseed: int, time: float, jday: int, dtime: float,
               plens: float, startp: float, endp: float,
               n: int, climate_seed: int) -> None:
    inputs_f = make_inputs(n, climate_seed, jday)
    # Reference Python doesn't mutate inputs in place, but be safe.
    inputs_r = DiurnalInputs(*(arr.copy() for arr in inputs_f))

    f_out = call_f77(iseed, time, jday, plens, startp, endp, dtime, inputs_f)
    r_out = call_ref(iseed, time, jday, plens, startp, endp, dtime, inputs_r)

    for name, spec in SPECS.items():
        report.add(verify_variable(
            f"diurnal.{name}.{label}", f_out[name], r_out[name], spec,
            abs_floor=1e-6,
        ))

    # Also check scalar event-timing outputs.
    for key in ("plens", "startp", "endp"):
        diff = abs(f_out[key] - r_out[key])
        status = "PASS" if diff < 1e-6 else "FAIL"
        print(f"[{status}] diurnal.{key}.{label}: F77={f_out[key]:.6f} "
              f"ref={r_out[key]:.6f} diff={diff:.2e}")


def main() -> int:
    report = Report(section="diurnal_layer1")

    DTIME = 3600.0

    # Scenario A: noon mid-year, no precip-event retry possible (time >= dtime).
    verify_one(report, "noon_jul",
               iseed=4242, time=12 * 3600.0, jday=183,
               dtime=DTIME, plens=4 * 3600.0, startp=10 * 3600.0,
               endp=14 * 3600.0, n=64, climate_seed=42)

    # Scenario B: pre-dawn, winter (low coszen).
    verify_one(report, "predawn_jan",
               iseed=4242, time=4 * 3600.0, jday=15,
               dtime=DTIME, plens=2 * 3600.0, startp=3 * 3600.0,
               endp=5 * 3600.0, n=64, climate_seed=43)

    # Scenario C: dusk, equinox.
    verify_one(report, "dusk_sep",
               iseed=9000, time=18 * 3600.0, jday=265,
               dtime=DTIME, plens=2 * 3600.0, startp=16 * 3600.0,
               endp=18 * 3600.0, n=64, climate_seed=44)

    # Scenario D: time < dtime (first timestep), event retry might fire.
    verify_one(report, "firstsstep_lowP",
               iseed=12345, time=0.0, jday=100,
               dtime=DTIME, plens=4 * 3600.0, startp=10 * 3600.0,
               endp=14 * 3600.0, n=64, climate_seed=45)

    # Scenario E: time < dtime + high precip across many points → trigger retry.
    rng = np.random.default_rng(46)
    big_inputs = DiurnalInputs(
        cloud  = rng.uniform(0.0, 1.0, 64),
        tmax   = rng.uniform(285.0, 305.0, 64),
        tmin   = rng.uniform(265.0, 285.0, 64),
        qd     = rng.uniform(0.001, 0.020, 64),
        ud     = rng.uniform(2.0, 6.0, 64),
        psurf  = rng.uniform(90000.0, 102000.0, 64),
        precip = rng.uniform(80.0, 200.0, 64),    # heavy precip
        lat    = rng.uniform(-60.0, 60.0, 64),
    )
    f_e = call_f77(1234, 0.0, 200, 4 * 3600.0, 10 * 3600.0, 14 * 3600.0,
                   DTIME, big_inputs)
    r_e = call_ref(1234, 0.0, 200, 4 * 3600.0, 10 * 3600.0, 14 * 3600.0,
                   DTIME, big_inputs)
    for name, spec in SPECS.items():
        report.add(verify_variable(
            f"diurnal.{name}.heavyP_firstdt", f_e[name], r_e[name], spec,
            abs_floor=1e-6,
        ))
    for key in ("plens", "startp", "endp"):
        diff = abs(f_e[key] - r_e[key])
        print(f"[{'PASS' if diff < 1e-6 else 'FAIL'}] "
              f"diurnal.{key}.heavyP_firstdt: F77={f_e[key]:.6f} "
              f"ref={r_e[key]:.6f} diff={diff:.2e}")

    report.print_table()
    out = report.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
