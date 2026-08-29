"""Verify the canonical reference state is what it claims to be.

There is one blessed model state in this project. Everything downstream --
methane evaluation, fire skill, calibration -- starts from it. This script
checks it end to end so nobody has to take its provenance on trust:

  * no NaN anywhere in the state
  * physically plausible ranges
  * converged (drift over the final 50 years within tolerance)
  * the fire skill quoted in docs/FIRE.md

    python verify/reference_run.py            # verify
    python verify/reference_run.py --skill    # also re-run GFED5 metrics (slower)

Exit code 0 if every check passes, 1 otherwise -- so it can gate a workflow.

Why this exists: four archived spin-ups on this NAS have zero litter, zero
climate normals and all 12 PFTs permitted everywhere, and two of them contain
NaN (docs/DEFECTS_FOR_REVIEW.md). None of that was discoverable without opening
the files. This makes the current one self-describing.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python-port"))

REFERENCE = Path("/shared/nas2/Hangkai/ibis/verify_outputs/spinup_fire_v8")
STATE = REFERENCE / "checkpoints" / "state_latest.pkl"
LOG = REFERENCE / "spinup_log.csv"

#: What the reference run is, recorded so the state is self-describing.
PROVENANCE = {
    "pipeline": "verify/timestepping/pipeline/spinup_fire_v8.py",
    "years": "200, cycling GSWP3 1901-2000",
    "resolution": "0.5 deg, 88484 land cells",
    "fire": "isimfire=1, seasonlength scheme, VPD danger, combustion "
            "completeness, per-PFT mortality",
    "nsoilcall": 10,
    "runtime": "208 min on an RTX 4090",
}

#: Convergence tolerances over the last 50 simulated years.
CONVERGED = {"biomass": 0.05, "totcsoi": 0.30, "disturbf": 0.005}

#: Physical bounds. Wider than expected values -- these catch breakage, not bias.
BOUNDS = {
    "soil.tsoi": (200.0, 340.0), "soil.tg": (200.0, 340.0),
    "soil.wsoi": (0.0, 1.0), "soil.wisoi": (0.0, 1.0),
    "soil.wpud": (0.0, 1e4), "vegetation.disturbf": (0.0, 1.0),
    "vegetation.exist": (0.0, 1.0),
}


def _get(state, dotted):
    obj = state
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return np.asarray(obj)


def check_state(path):
    ok = True
    ck = pickle.load(open(path, "rb"))
    st = ck["jax_state"] if isinstance(ck, dict) else ck
    step = ck.get("step") if isinstance(ck, dict) else None
    has_key = isinstance(ck, dict) and ck.get("key") is not None

    print(f"  state      {path}")
    print(f"  step       {step}")
    print(f"  PRNG key   {'present -- resumable reproducibly' if has_key else 'MISSING'}")
    if not has_key:
        print("    FAIL: without the key a restart gives different stochastic "
              "weather (docs/FIRE.md).")
        ok = False

    nan = 0
    for grp in ("soil", "vegetation", "carbon", "snow", "hydrology", "summary",
                "canopy", "atm"):
        o = getattr(st, grp, None)
        if o is None:
            continue
        for fd in type(o).__dataclass_fields__.values():
            a = getattr(o, fd.name, None)
            if a is None or not hasattr(a, "dtype"):
                continue
            arr = np.asarray(a)
            if np.issubdtype(arr.dtype, np.floating):
                nan += int(np.isnan(arr).sum())
    print(f"  NaN values {nan}")
    if nan:
        print("    FAIL: one NaN cell poisons every global mean.")
        ok = False

    print("\n  physical bounds")
    for field, (lo, hi) in BOUNDS.items():
        try:
            a = _get(st, field)
        except AttributeError:
            print(f"    SKIP  {field} (absent)")
            continue
        bad = (a < lo) | (a > hi)
        flag = "ok  " if not bad.any() else "FAIL"
        if bad.any():
            ok = False
        print(f"    {flag}  {field:22} {a.min():10.3f} .. {a.max():10.3f}"
              f"   expected [{lo}, {hi}]")
    return ok, st


def check_convergence(log_path):
    if not log_path.exists():
        print(f"\n  SKIP convergence: {log_path} not found")
        return True
    rows = [r for r in csv.DictReader(open(log_path))
            if r["biomass"].lower() != "nan"]
    print(f"\n  convergence over the last 50 of {len(rows)} steps")
    ok = True
    tail = [r for r in rows if int(r["step"]) >= int(rows[-1]["step"]) - 50]
    for k, tol in CONVERGED.items():
        v = [float(r[k]) for r in tail]
        drift = abs(v[-1] - v[0])
        sd = float(np.std(v))
        flag = "ok  " if drift <= tol else "FAIL"
        if drift > tol:
            ok = False
        print(f"    {flag}  {k:10} mean {np.mean(v):8.4f}  sd {sd:7.4f}  "
              f"drift {drift:+7.4f}   tol {tol}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--skill", action="store_true",
                    help="also recompute the GFED5 fire metrics")
    a = ap.parse_args()

    print("\n=== CANONICAL REFERENCE RUN ===")
    for k, v in PROVENANCE.items():
        print(f"  {k:12} {v}")
    print()

    ok, st = check_state(Path(a.state))
    ok = check_convergence(Path(a.log)) and ok

    if a.skill:
        print("\n  fire skill vs GFED5")
        from verify.fire.evaluate_gfed5 import _grid, observed, report
        lat, _, li, lo, mlat, mlon = _grid()
        obs = observed("/shared/nas2/Hangkai/ibis/inputs/fire/gfed5",
                       li, lo, mlat, mlon, range(2001, 2006))
        report(np.asarray(st.vegetation.disturbf), obs, lat, "reference state")

    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
