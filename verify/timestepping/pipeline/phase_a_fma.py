"""Phase A: FMA elimination test.

Hypothesis: F77 and JAX produce different rounding because XLA uses FMA
(fused multiply-add) while F77 with -O0 does not. The accumulated effect
over many timesteps biases JAX toward consistently slightly different
values, producing the multi-day drift.

Test:
  1. Rebuild F77 with -ffp-contract=off (explicit no-FMA).
  2. Set JAX to use highest precision and disable XLA fast-math.
  3. Re-run the heat-wave chaotic test (which measures JAX-vs-F77 drift).
  4. Compare drift magnitude to baseline.

If drift drops dramatically (e.g., from 1.4e-3 to <1e-7) — FMA was the
source. If drift unchanged — FMA is not the cause.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=python-port verify/.venv/bin/python -m verify.timestepping.pipeline.phase_a_fma
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path


def rebuild_f77_no_fma() -> None:
    """Rebuild F77 with -ffp-contract=off."""
    print("=" * 72)
    print("Rebuilding F77 with -ffp-contract=off (no FMA)...")
    print("=" * 72)

    build_script = Path("/home/hangkai/ibis-port/python-port/tests/f2py/build_f77.py")
    # We'll patch FFLAGS by setting env var that build_f77 honors, OR pass
    # via a wrapper. Simplest: patch the file temporarily.
    text = build_script.read_text()
    original_fflags = '"-ffixed-line-length-132 -O0 -fdefault-real-8 -fdefault-double-8 -fPIC -w".split()'
    new_fflags = '"-ffixed-line-length-132 -O0 -ffp-contract=off -fdefault-real-8 -fdefault-double-8 -fPIC -w".split()'

    if "-ffp-contract=off" in text:
        print("  build_f77.py already patched.")
        already_patched = True
    else:
        already_patched = False
        text_new = text.replace(original_fflags, new_fflags)
        if text_new == text:
            print("  WARN: could not locate FFLAGS to patch.")
            print("  Run rebuild manually.")
            return False
        build_script.write_text(text_new)
        print("  Patched build_f77.py with -ffp-contract=off")

    # Clean and rebuild
    print("  Running build...")
    r = subprocess.run(
        ["verify/.venv/bin/python", "python-port/tests/f2py/build_f77.py", "--clean"],
        cwd="/home/hangkai/ibis-port",
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  Clean failed: {r.stderr[-500:]}")
    r = subprocess.run(
        ["verify/.venv/bin/python", "python-port/tests/f2py/build_f77.py"],
        cwd="/home/hangkai/ibis-port",
        capture_output=True, text=True,
        timeout=300,
    )
    if r.returncode != 0:
        print(f"  Build failed: {r.stderr[-1000:]}")
        if not already_patched:
            build_script.write_text(text)
        return False
    print("  F77 rebuilt with no-FMA.")
    return True


def run_chaotic_test() -> tuple[float, float]:
    """Run the chaotic test and parse drift values."""
    env = os.environ.copy()
    env["XLA_FLAGS"] = (
        "--xla_cpu_enable_fast_math=false "
        "--xla_cpu_fast_math_honor_infs=true "
        "--xla_cpu_fast_math_honor_nans=true "
        "--xla_allow_excess_precision=false"
    )
    env["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

    env["PYTHONPATH"] = "/home/hangkai/ibis-port/python-port"
    r = subprocess.run(
        ["verify/.venv/bin/python", "-m", "verify.timestepping.diag_chaotic_test"],
        cwd="/home/hangkai/ibis-port",
        env=env,
        capture_output=True, text=True,
        timeout=600,
    )
    if r.returncode != 0:
        print(f"  Test failed: {r.stderr[-1000:]}")
        return float("nan"), float("nan")

    jax_vs_f77 = float("nan")
    jax_vs_perturbed = float("nan")
    for line in r.stdout.split("\n"):
        if "JAX_a vs F77:" in line:
            jax_vs_f77 = float(line.split("=")[-1].strip())
        if "JAX_a vs JAX_b" in line:
            jax_vs_perturbed = float(line.split("=")[-1].strip())
    return jax_vs_f77, jax_vs_perturbed


def main():
    print("PHASE A: FMA elimination test")
    print()
    print("Baseline (current build, current JAX defaults):")
    print("  Expected drift: ~1.4e-3 (JAX vs F77 at step 167 of heat wave)")
    print()

    ok = rebuild_f77_no_fma()
    if not ok:
        print("Aborting — F77 rebuild failed.")
        return 1

    print()
    print("=" * 72)
    print("Running chaotic test with no-FMA F77 + JAX high-precision flags...")
    print("=" * 72)

    drift_vs_f77, drift_vs_perturbed = run_chaotic_test()
    print()
    print(f"  JAX vs F77 (no-FMA both):     {drift_vs_f77:.3e}")
    print(f"  JAX vs JAX_a perturbed 1e-8:  {drift_vs_perturbed:.3e}")
    print()
    print("Interpretation:")
    if drift_vs_f77 < 1e-7:
        print("  FMA was the source — drift collapsed to machine precision.")
    elif drift_vs_f77 < 1e-4:
        print("  FMA was PART of the source — drift reduced 10×+. Investigate")
        print("  remaining FP-rounding sources.")
    else:
        print("  FMA is NOT the source — drift unchanged.")
        print("  The bias has a structural origin (formula / iteration / solver).")
        print("  Proceed to Phase B (per-stage bisection).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
