"""Verify Python ran2 against F77 ran2 bit-exact.

Layer-1 anchor: if our Python port matches F77 to 0 ULP over many seeds and
long sequences, we have a deterministic foundation for everything downstream
in the weather generator that consumes ran2 output.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.weather.verify_ran2
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
from verify.framework.stats import _ulp_distance
from verify.framework.tolerance_classes import ToleranceSpec, TolClass

from verify.weather.ran2 import Ran2


def compare_seed(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (f77_seq, py_seq, max_ulp) for one seed."""
    f_seq = np.asarray(f77.wrap_ran2_seq(seed, n))
    rng = Ran2(seed)
    p_seq = rng.uniforms(n)
    ulps = _ulp_distance(f_seq, p_seq)
    return f_seq, p_seq, int(ulps.max())


def main() -> int:
    seeds = [1, 2, 3, 17, 42, 99, 123, 1234, 12345, 987654, 2147483]
    n = 10_000

    report = Report(section="ran2")

    all_max_ulp = 0
    all_max_abs = 0.0

    print(f"=== ran2 bit-exact check (n={n} per seed, {len(seeds)} seeds) ===")
    print(f"{'seed':>10}  {'max_abs':>10}  {'max_ulp':>8}  {'in_range':>9}")
    for seed in seeds:
        f_seq, p_seq, max_ulp = compare_seed(seed, n)
        max_abs = float(np.max(np.abs(f_seq - p_seq)))
        in_range = bool(np.all((p_seq > 0) & (p_seq <= 1.0 - 1e-7)))
        print(f"{seed:>10d}  {max_abs:>10.2e}  {max_ulp:>8d}  {str(in_range):>9}")
        all_max_ulp = max(all_max_ulp, max_ulp)
        all_max_abs = max(all_max_abs, max_abs)

    print()
    print(f"Worst max_abs across all seeds: {all_max_abs:.3e}")
    print(f"Worst max_ulp across all seeds: {all_max_ulp}")

    # Use the framework on one representative seed for the canonical report row
    f_seq, p_seq, _ = compare_seed(42, n)
    spec = ToleranceSpec(
        "ran2", TolClass.BIT_EXACT,
        why="Pure integer state + float64 conversion; must be bit-exact.",
    )
    report.add(verify_variable("ran2.uniform", f_seq, p_seq, spec))
    report.print_table()
    out = report.save(Path(__file__).parent / "results")
    print(f"\nResiduals saved: {out}")

    passed = (all_max_ulp == 0)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
