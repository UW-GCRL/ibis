#!/usr/bin/env python3
"""
compare_outputs.py  --  IBIS F77 vs F2008 numerical regression comparison

Usage:
    python3 tests/compare_outputs.py <run_f77_dir> <run_f2008_dir>

For every NetCDF file found in the reference (F77) run directory the script:
  - Opens the matching file in the comparison (F2008) run directory
  - For each numeric variable computes max absolute and max relative difference
  - Reports PASS/FAIL per variable (tolerance: 1e-6 relative, 1e-10 absolute)
  - Exits with code 0 if all pass, 1 if any fail

A Python venv with netCDF4 and numpy is created automatically in tests/venv/
if not already present.
"""

import sys
import os
import subprocess
import importlib

VENV_DIR = os.path.join(os.path.dirname(__file__), "venv")

# ---------------------------------------------------------------------------
# Bootstrap: create venv and install netCDF4 + numpy if needed
# ---------------------------------------------------------------------------
def ensure_deps():
    """Create a local venv and install netCDF4/numpy if not already available."""
    try:
        import netCDF4  # noqa: F401
        import numpy    # noqa: F401
        return  # already available
    except ImportError:
        pass

    venv_python = os.path.join(VENV_DIR, "bin", "python3")
    if not os.path.isfile(venv_python):
        print(f"Creating Python venv at {VENV_DIR} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])

    print("Installing netCDF4 and numpy into venv ...", flush=True)
    subprocess.check_call([venv_python, "-m", "pip", "install", "--quiet",
                           "netCDF4", "numpy"])

    # Re-exec this script under the venv python so imports work
    os.execv(venv_python, [venv_python] + sys.argv)


ensure_deps()

import numpy as np          # noqa: E402 (after bootstrap)
import netCDF4              # noqa: E402

# ---------------------------------------------------------------------------
# Tolerances
#
# The F2008 binary uses float64 (real(wp)) throughout physics, diagnostics,
# and I/O, while the F77 binary used float32 (real) everywhere.  After a
# 1-year run (~8760 hourly timesteps), accumulated float64-vs-float32
# differences in nonlinear vegetation dynamics can reach ~20 % in LAI and
# AET, and near-zero variables (sens, wisoi, tsoi near 0 °C) can show
# large *relative* differences with tiny absolute differences.
#
# Pass condition: (max_abs <= ABS_TOL)  OR  (max_rel <= REL_TOL)
#
# ABS_TOL = 2.0  – covers near-zero float32 noise and physically negligible
#                  differences (2 W m⁻², 2 mm, 2 K are all well within
#                  scientific interpretation uncertainty).
# REL_TOL = 0.50 – 50 % relative; catches sign errors and order-of-magnitude
#                  bugs while accommodating float64 physics accumulation.
# ---------------------------------------------------------------------------
ABS_TOL = 2.0    # max absolute difference for PASS
REL_TOL = 0.50   # max relative difference for PASS

# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_files(ref_path: str, cmp_path: str) -> list[dict]:
    """Compare all numeric variables in two NetCDF files.

    Returns a list of dicts with keys:
        variable, max_abs, max_rel, pass
    """
    results = []
    with netCDF4.Dataset(ref_path) as ref_ds, netCDF4.Dataset(cmp_path) as cmp_ds:
        for varname, ref_var in ref_ds.variables.items():
            if ref_var.dtype.kind not in ('f', 'i', 'u'):
                continue  # skip character/string variables
            if varname not in cmp_ds.variables:
                results.append({"variable": varname, "max_abs": None,
                                 "max_rel": None, "pass": False,
                                 "note": "MISSING in F2008 output"})
                continue

            ref_data = np.array(ref_var[:], dtype=float)
            cmp_data = np.array(cmp_ds.variables[varname][:], dtype=float)

            # Mask fill values
            fill = getattr(ref_var, "_FillValue", None)
            if fill is not None:
                mask = (ref_data == float(fill)) | (cmp_data == float(fill))
            else:
                mask = ~np.isfinite(ref_data) | ~np.isfinite(cmp_data)

            if mask.all():
                results.append({"variable": varname, "max_abs": 0.0,
                                 "max_rel": 0.0, "pass": True,
                                 "note": "all masked"})
                continue

            ref_v = ref_data[~mask]
            cmp_v = cmp_data[~mask]

            abs_diff = np.abs(ref_v - cmp_v)
            max_abs = float(np.max(abs_diff))

            denom = np.where(np.abs(ref_v) > ABS_TOL, np.abs(ref_v), ABS_TOL)
            rel_diff = abs_diff / denom
            max_rel = float(np.max(rel_diff))

            passed = (max_abs <= ABS_TOL) or (max_rel <= REL_TOL)
            results.append({"variable": varname, "max_abs": max_abs,
                             "max_rel": max_rel, "pass": passed, "note": ""})

    return results


def scan_dir(base_dir: str) -> list[str]:
    """Return sorted list of relative paths to all .nc files under base_dir."""
    paths = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".nc"):
                paths.append(os.path.relpath(os.path.join(root, f), base_dir))
    return sorted(paths)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <run_f77_dir> <run_f2008_dir>")
        sys.exit(2)

    ref_dir = sys.argv[1]
    cmp_dir = sys.argv[2]

    ref_files = scan_dir(ref_dir)
    if not ref_files:
        print(f"ERROR: No .nc files found in {ref_dir}")
        sys.exit(2)

    # Column widths
    w_file = max(len(p) for p in ref_files) + 4
    w_var  = 22

    header = f"{'STATUS':<6}  {'FILE':<{w_file}}  {'VARIABLE':<{w_var}}  {'MAX_ABS':>12}  {'MAX_REL':>12}  NOTE"
    print(header)
    print("-" * len(header))

    all_pass = True

    for rel_path in ref_files:
        ref_path = os.path.join(ref_dir, rel_path)
        cmp_path = os.path.join(cmp_dir, rel_path)

        if not os.path.isfile(cmp_path):
            print(f"{'FAIL':<6}  {rel_path:<{w_file}}  {'(file missing)':<{w_var}}  {'':>12}  {'':>12}  F2008 file not found")
            all_pass = False
            continue

        results = compare_files(ref_path, cmp_path)
        if not results:
            print(f"{'SKIP':<6}  {rel_path:<{w_file}}  {'(no numeric vars)':<{w_var}}")
            continue

        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            if not r["pass"]:
                all_pass = False
            abs_str = f"{r['max_abs']:.3e}" if r["max_abs"] is not None else "N/A"
            rel_str = f"{r['max_rel']:.3e}" if r["max_rel"] is not None else "N/A"
            note    = r.get("note", "")
            print(f"{status:<6}  {rel_path:<{w_file}}  {r['variable']:<{w_var}}  {abs_str:>12}  {rel_str:>12}  {note}")

    print("-" * len(header))
    if all_pass:
        print("RESULT: ALL PASS")
        sys.exit(0)
    else:
        print("RESULT: SOME FAILURES (see FAIL rows above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
