"""Verify F77 `daily` jdaily=1 path vs reference Python.

The jdaily=1 path (weather.f lines 824-891) is purely deterministic — uses
observed daily anomalies/fractions rather than the weather generator. We
verify F77 ↔ reference Python at machine precision. JAX has no equivalent
path, so this is F77 ↔ ref only.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.weather.verify_daily_jdaily1
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

from verify.weather.reference_daily import (
    Climate, DailyAnomaly, daily_jdaily1 as ref_jdaily1,
)


SPEC = ToleranceSpec("v", TolClass.PRECISION,
                     abs_tol=1e-9, rel_tol=1e-12, bias_tol=1e-10)


def make_climate(n: int, seed: int) -> Climate:
    rng = np.random.default_rng(seed)
    return Climate(
        xint    = rng.uniform(-10.0, 35.0, (n, 12)),
        xinq    = rng.uniform(20.0, 95.0, (n, 12)),
        xinprec = rng.uniform(0.1, 8.0,   (n, 12)),
        xinwind = rng.uniform(1.0, 6.0,   (n, 12)),
        xincld  = rng.uniform(10.0, 90.0, (n, 12)),
        xinwet  = rng.uniform(3.0, 25.0,  (n, 12)),
        xintrng = rng.uniform(4.0, 18.0,  (n, 12)),
        xintopo = rng.uniform(0.0, 2000.0, n),
    )


def make_anomaly(n: int, seed: int) -> DailyAnomaly:
    rng = np.random.default_rng(seed + 9999)
    return DailyAnomaly(
        xintd    = rng.uniform(-10.0, 10.0, n),
        xinqd    = rng.uniform(0.5, 1.5, n),
        xinprecd = rng.uniform(0.0, 3.0, n),
        xinwindd = rng.uniform(0.5, 1.5, n),
        xincldd  = rng.uniform(-30.0, 30.0, n),
        xintrngd = rng.uniform(0.5, 1.5, n),
    )


def verify_one(report: Report, label: str, n: int, seed: int,
               imonth: int, iday: int) -> None:
    climate = make_climate(n, seed)
    anomaly = make_anomaly(n, seed)

    f_out = f77.wrap_daily_jdaily1(
        imonth, iday,
        climate.xint, climate.xinq, climate.xinprec, climate.xinwind,
        climate.xincld, climate.xinwet, climate.xintrng, climate.xintopo,
        anomaly.xintd, anomaly.xinqd, anomaly.xinprecd, anomaly.xinwindd,
        anomaly.xincldd, anomaly.xintrngd,
        9.80616, 287.04,
    )
    td_f, tmax_f, tmin_f, cloud_f, precip_f, qd_f, ud_f, psurf_f = (
        np.asarray(x) for x in f_out
    )

    r_out = ref_jdaily1(imonth, iday, climate, anomaly)

    pairs = [
        ("td",     td_f,     r_out.td),
        ("tmax",   tmax_f,   r_out.tmax),
        ("tmin",   tmin_f,   r_out.tmin),
        ("cloud",  cloud_f,  r_out.cloud),
        ("precip", precip_f, r_out.precip),
        ("qd",     qd_f,     r_out.qd),
        ("ud",     ud_f,     r_out.ud),
        ("psurf",  psurf_f,  r_out.psurf),
    ]
    for name, fv, rv in pairs:
        report.add(verify_variable(
            f"jd1.{name}.{label}", fv, rv, SPEC, abs_floor=1e-6,
        ))


def main() -> int:
    rep = Report(section="daily_jdaily1")
    verify_one(rep, "jan_d1",  n=1500, seed=42, imonth=1,  iday=1)
    verify_one(rep, "jul_d15", n=1500, seed=99, imonth=7,  iday=15)
    verify_one(rep, "dec_d31", n=1500, seed=7,  imonth=12, iday=31)
    rep.print_table()
    rep.save(Path(__file__).parent / "results")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
