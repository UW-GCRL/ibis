"""Verify reference Python daily against F77 daily — Layer 1.

Layer 1: the reference Python `daily` (which uses our bit-exact ran2 port and
is a literal transcription of F77 weather.f) should match F77 daily up to a
few ULP of libm noise (exp/log/pow/sqrt/sin/cos can differ ~1 ULP between
gfortran/glibc and Python's math module).

This catches any transcription bug in our reference Python *and* documents
the F77 algorithm's actual behavior (including the wet/dry Markov bug).

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.weather.verify_daily
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

from verify.weather.ran2 import Ran2
from verify.weather.reference_daily import (
    Climate, DailyState, daily as ref_daily,
)


SPECS = {
    "td":     ToleranceSpec("td",    TolClass.PRECISION,
                            abs_tol=1e-9,  rel_tol=1e-11, bias_tol=1e-10,
                            why="Daily mean temperature; transcends one sqrt/log call."),
    "tmax":   ToleranceSpec("tmax",  TolClass.PRECISION,
                            abs_tol=1e-9,  rel_tol=1e-11, bias_tol=1e-10),
    "tmin":   ToleranceSpec("tmin",  TolClass.PRECISION,
                            abs_tol=1e-9,  rel_tol=1e-11, bias_tol=1e-10),
    "cloud":  ToleranceSpec("cloud", TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-12, bias_tol=1e-13),
    "precip": ToleranceSpec("precip", TolClass.PRECISION,
                            abs_tol=1e-10, rel_tol=1e-10, bias_tol=1e-11,
                            why="Gamma rejection sampler; involves pow/log."),
    "qd":     ToleranceSpec("qd",    TolClass.PRECISION,
                            abs_tol=1e-12, rel_tol=1e-11, bias_tol=1e-13,
                            why="Triangular CDF inverse with sqrt; involves esat/qsat."),
    "ud":     ToleranceSpec("ud",    TolClass.PRECISION,
                            abs_tol=1e-10, rel_tol=1e-10, bias_tol=1e-11,
                            why="Weibull deviate: (-log u)**0.30, libm pow noise."),
    "psurf":  ToleranceSpec("psurf", TolClass.PRECISION,
                            abs_tol=1e-6,  rel_tol=1e-11, bias_tol=1e-7,
                            why="Hypsometric: T^rwork with rwork~5.26."),
    "iwet":   ToleranceSpec("iwet",  TolClass.BIT_EXACT,
                            why="Integer flag from rn <=> pwd compare; must match exactly."),
}


def make_climate(n: int, seed: int) -> Climate:
    rng = np.random.default_rng(seed)
    return Climate(
        xint    = rng.uniform(-10.0, 35.0, (n, 12)),     # C
        xinq    = rng.uniform(20.0, 95.0, (n, 12)),      # %
        xinprec = rng.uniform(0.1, 8.0,   (n, 12)),      # mm/day
        xinwind = rng.uniform(1.0, 6.0,   (n, 12)),      # m/s
        xincld  = rng.uniform(10.0, 90.0, (n, 12)),      # %
        xinwet  = rng.uniform(3.0, 25.0,  (n, 12)),      # days
        xintrng = rng.uniform(4.0, 18.0,  (n, 12)),      # C
        xintopo = rng.uniform(0.0, 2000.0, n),           # m
    )


def make_initial_state(n: int) -> DailyState:
    s = DailyState.fresh(n)
    # Use mid-range a10td so tcthis/twthis updates produce reasonable values.
    s.a10td.fill(288.0)
    return s


def call_f77(climate: Climate, state: DailyState,
             iseed: int, imonth: int, iday: int) -> dict:
    out = f77.wrap_daily(
        iseed, imonth, iday,
        climate.xint, climate.xinq, climate.xinprec, climate.xinwind,
        climate.xincld, climate.xinwet, climate.xintrng, climate.xintopo,
        state.xstore, state.iwetday, state.precipday,
        state.iwetdaysum, state.precipdaysum,
        state.tcthis, state.twthis, state.gdd0this, state.gdd5this,
        state.a10td, 9.80616, 287.04,
    )
    (td_f, tmax_f, tmin_f, cloud_f, precip_f,
     qd_f, ud_f, psurf_f, iwet_f,
     xstore_f, iwetday_f, precipday_f,
     iwetdaysum_f, precipdaysum_f,
     tcthis_f, twthis_f, gdd0this_f, gdd5this_f,
     xinwet_f) = out
    return dict(
        td=np.asarray(td_f), tmax=np.asarray(tmax_f), tmin=np.asarray(tmin_f),
        cloud=np.asarray(cloud_f), precip=np.asarray(precip_f),
        qd=np.asarray(qd_f), ud=np.asarray(ud_f),
        psurf=np.asarray(psurf_f), iwet=np.asarray(iwet_f),
        xstore=np.asarray(xstore_f),
        iwetday=np.asarray(iwetday_f), precipday=np.asarray(precipday_f),
        iwetdaysum=np.asarray(iwetdaysum_f),
        precipdaysum=np.asarray(precipdaysum_f),
    )


def call_ref(climate: Climate, state: DailyState,
             iseed: int, imonth: int, iday: int) -> dict:
    out = ref_daily(iseed, imonth, iday, climate, state)
    return dict(
        td=out.td, tmax=out.tmax, tmin=out.tmin,
        cloud=out.cloud, precip=out.precip,
        qd=out.qd, ud=out.ud, psurf=out.psurf, iwet=out.iwet,
        xstore=state.xstore.copy(),
        iwetday=state.iwetday.copy(), precipday=state.precipday.copy(),
        iwetdaysum=state.iwetdaysum.copy(),
        precipdaysum=state.precipdaysum.copy(),
    )


def verify_one(report: Report, iseed: int, imonth: int, iday: int, n: int,
               climate_seed: int = 42) -> None:
    """Run F77 and reference Python with identical setup; verify."""
    climate_f = make_climate(n, climate_seed)
    climate_r = Climate(*(arr.copy() for arr in climate_f))
    state_f = make_initial_state(n)
    state_r = make_initial_state(n)

    f_out = call_f77(climate_f, state_f, iseed, imonth, iday)
    r_out = call_ref(climate_r, state_r, iseed, imonth, iday)

    label = f"m{imonth}d{iday}_s{iseed}"
    for name, spec in SPECS.items():
        report.add(verify_variable(
            f"daily.{name}.{label}", f_out[name], r_out[name], spec,
            abs_floor=1e-6,
        ))


def verify_sequence(report: Report, iseed: int, imonth: int, n: int,
                    n_days: int = 10, climate_seed: int = 42) -> None:
    """Step F77 and reference Python through n_days days threading state."""
    climate_f = make_climate(n, climate_seed)
    climate_r = Climate(*(arr.copy() for arr in climate_f))
    state_f = make_initial_state(n)
    state_r = make_initial_state(n)

    for iday in range(1, n_days + 1):
        f_out = call_f77(climate_f, state_f, iseed + iday, imonth, iday)
        r_out = call_ref(climate_r, state_r, iseed + iday, imonth, iday)

        # Update state_f from F77 outputs (state_r is already mutated in place).
        state_f.xstore = f_out["xstore"]
        state_f.iwetday = f_out["iwetday"]
        state_f.precipday = f_out["precipday"]
        state_f.iwetdaysum = f_out["iwetdaysum"]
        state_f.precipdaysum = f_out["precipdaysum"]

        label = f"seq_m{imonth}_d{iday}_s{iseed+iday}"
        for name, spec in SPECS.items():
            report.add(verify_variable(
                f"daily.{name}.{label}", f_out[name], r_out[name], spec,
                abs_floor=1e-6,
            ))


def make_extreme_climate(n: int, seed: int) -> Climate:
    """Climate designed to trigger F77's goto-9000 retry (precip > 360 mm)."""
    rng = np.random.default_rng(seed)
    return Climate(
        xint    = rng.uniform(15.0, 30.0, (n, 12)),
        xinq    = rng.uniform(40.0, 90.0, (n, 12)),
        xinprec = rng.uniform(20.0, 50.0, (n, 12)),   # very heavy
        xinwind = rng.uniform(1.0, 6.0,   (n, 12)),
        xincld  = rng.uniform(10.0, 90.0, (n, 12)),
        xinwet  = rng.uniform(3.0, 8.0,   (n, 12)),    # few wet days
        xintrng = rng.uniform(4.0, 18.0,  (n, 12)),
        xintopo = rng.uniform(0.0, 500.0, n),
    )


def verify_extreme(report: Report, iseed: int, imonth: int, n: int,
                   climate_seed: int) -> None:
    """Extreme rainfall scenario to exercise F77's goto-9000 retry path
    (lines 556-567 in weather_subset.f). Reference Python mirrors F77."""
    climate_f = make_extreme_climate(n, climate_seed)
    climate_r = Climate(*(arr.copy() for arr in climate_f))
    state_f = make_initial_state(n)
    state_r = make_initial_state(n)

    f_out = call_f77(climate_f, state_f, iseed, imonth, 1)
    r_out = call_ref(climate_r, state_r, iseed, imonth, 1)

    label = f"extreme_m{imonth}_s{iseed}"
    for name, spec in SPECS.items():
        report.add(verify_variable(
            f"daily.{name}.{label}", f_out[name], r_out[name], spec,
            abs_floor=1e-6,
        ))


def main() -> int:
    report = Report(section="daily_layer1")

    # Single-call scenarios.
    scenarios = [
        (1234, 1,  1, 50),
        (1234, 1,  5, 50),
        (5678, 6, 15, 50),
        (4242, 7,  1, 50),
        (4242, 12, 31, 50),
        (99,   3,  1, 100),
    ]
    for iseed, imonth, iday, n in scenarios:
        verify_one(report, iseed, imonth, iday, n)

    # Extreme-precip scenarios to exercise F77 retry path (lines 556-567)
    # and gamma rejection edge (line 453). Reference Python mirrors F77.
    verify_extreme(report, iseed=12345, imonth=7, n=200, climate_seed=55)
    verify_extreme(report, iseed=999,   imonth=8, n=200, climate_seed=77)

    # Sequential-day stress: thread state across iday=1..10.
    verify_sequence(report, iseed=12345, imonth=4, n=64, n_days=10)
    verify_sequence(report, iseed=999,   imonth=8, n=64, n_days=10)

    report.print_table()
    out = report.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
