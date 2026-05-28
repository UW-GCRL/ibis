"""Per-stage F77 isolation drill to pinpoint which subroutine first NaN.

Loads a snapshot (saved by lockstep_tibet_nan_drill.py) at a specific
hour, then runs the F77 lsxmain stages one by one, checking NaN after each:

    1. setsoi      (soil thermodynamic params)
    2. fwetcal     (canopy wetness)
    3. radiation   (shortwave + longwave splits)
    4. cascade     (canopy interception)
    5. fwetcal     (re-call with post-cascade water)
    6. drystress   (water stress)
    7. canopy      (3-iter loop: turcof + stomata + turvap)
    8. cascad2     (final phase correction)
    9. noveg       (no-veg handling)
   10. soilctl     (soil column thermodynamics + hydrology)
   11. snow        (snow physics)

The first stage that introduces NaN is the bug source.

Usage:
    python -m verify.timestepping.pipeline.lockstep_tibet_stage_isolate \\
        --snapshot d04_h15_pre.pkl
"""
from __future__ import annotations
import argparse
import math
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

from verify.timestepping.pipeline.global_gswp3_test import (
    diurnal_disaggregate, read_gswp3_month, make_batch_state,
    soil_template_for_lat,
)
from verify.timestepping.verify_lsxmain import (
    setup_f77_state, setup_jax_state, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params
from tests.f2py.ibis_f77_wrapper import NPOI


SNAPSHOT_DIR = Path("/shared/nas2/Hangkai/ibis/verify_outputs/tibet_nan_snapshots")


def has_nan(f77, label, *, check_extra=False):
    """Quick scan: which F77 fields are NaN right now?"""
    fields_to_check = [
        ("veg.tu",       lambda: f77.get_veg()["tu"]),
        ("veg.ts",       lambda: f77.get_veg()["ts"]),
        ("veg.tl",       lambda: f77.get_veg()["tl"]),
        ("veg.t12",      lambda: f77.get_veg()["t12"]),
        ("veg.t34",      lambda: f77.get_veg()["t34"]),
        ("veg.q12",      lambda: f77.get_veg()["q12"]),
        ("veg.q34",      lambda: f77.get_veg()["q34"]),
        ("veg.wliqu",    lambda: f77.get_veg()["wliqu"]),
        ("veg.wsnou",    lambda: f77.get_veg()["wsnou"]),
        ("veg.wliql",    lambda: f77.get_veg()["wliql"]),
        ("veg.wsnol",    lambda: f77.get_veg()["wsnol"]),
        ("veg.fwetu",    lambda: f77.get_veg()["fwetu"]),
        ("veg.fwetl",    lambda: f77.get_veg()["fwetl"]),
        ("soil.tg",      lambda: f77.get_soil()["tg"]),
        ("soil.tsoi[0]", lambda: f77.get_soil()["tsoi"][:, 0]),
        ("soil.wsoi[0]", lambda: f77.get_soil()["wsoi"][:, 0]),
        ("soil.wisoi[0]",lambda: f77.get_soil()["wisoi"][:, 0]),
        ("snow.fi",      lambda: f77.get_snow()["fi"]),
        ("snow.tsno[0]", lambda: f77.get_snow()["tsno"][:, 0]),
        ("snow.hsno[0]", lambda: f77.get_snow()["hsno"][:, 0]),
        ("flux.fseng",   lambda: f77.get_fluxes()["fseng"]),
        ("flux.fsenu",   lambda: f77.get_fluxes()["fsenu"]),
        ("flux.fsenl",   lambda: f77.get_fluxes()["fsenl"]),
        ("flux.fseni",   lambda: f77.get_fluxes()["fseni"]),
        ("flux.fvapg",   lambda: f77.get_fluxes()["fvapg"]),
        ("stom.gsub",    lambda: f77.get_stomata()["gsub"]),
        ("stom.gsls",    lambda: f77.get_stomata()["gsls"]),
    ]
    nans = []
    for name, getter in fields_to_check:
        try:
            val = getter()
            arr = np.asarray(val)
            if np.any(np.isnan(arr)):
                # how many cells / which?
                if arr.ndim == 1:
                    nan_cells = np.where(np.isnan(arr))[0].tolist()
                else:
                    nan_cells = np.where(np.isnan(arr).any(axis=tuple(range(1, arr.ndim))))[0].tolist()
                nans.append((name, nan_cells))
        except Exception as e:
            nans.append((name, f"<get failed: {e}>"))
    return nans


def restore_state(f77, snap):
    """Restore F77 state from a snapshot — best-effort across exposed fields.
    Sets veg/soil/snow via the wrapper's set_* methods. Not every common-block
    field is settable; this is approximate."""
    veg = snap["veg"]
    soil = snap["soil"]
    snow = snap["snow"]

    # set_veg needs structural fields too (fu/fl/lai/sai/...). The snapshot
    # captured everything from get_veg(); fill in via veg dict.
    veg_kwargs = {}
    for k in ("tu","ts","tl","tlsub","t12","t34","q12","q34",
              "wliqu","wliqs","wliql","wsnou","wsnos","wsnol"):
        if k in veg:
            veg_kwargs[k] = veg[k]
    # Veg structure: fu/fl/lai/sai/ztop/zbot/froot/frac — may not be in
    # get_veg() output; use get_veg_structure() if available.
    if hasattr(f77, "get_veg_structure"):
        try:
            vs = f77.get_veg_structure()
            for k in ("fu", "fl", "lai", "sai", "ztop", "zbot", "froot", "frac"):
                if k in vs:
                    veg_kwargs[k] = vs[k]
        except Exception:
            pass
    if all(k in veg_kwargs for k in ("fu","fl","lai","sai","ztop","zbot","froot","frac")):
        f77.set_veg(**veg_kwargs)

    # set_soil
    soil_kwargs = {}
    for k in ("tsoi","wsoi","wisoi","tg","ti","wpud","wipud"):
        if k in soil:
            soil_kwargs[k] = soil[k]
    # Soil structure: sand, clay, poros, hydraul, suction, bex, sfield, swilt, rhosoi, hsoi
    # These don't change during a run; need to load from elsewhere or skip.
    # For now, skip set_soil if missing structure args.

    # set_snow
    snow_kwargs = {}
    for k in ("fi","tsno","hsno"):
        if k in snow:
            snow_kwargs[k] = snow[k]
    if all(k in snow_kwargs for k in ("fi","tsno","hsno")):
        f77.set_snow(**snow_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, default="d04_h15_pre.pkl",
                        help="Snapshot file in tibet_nan_snapshots/")
    parser.add_argument("--year", type=int, default=1960)
    parser.add_argument("--month", type=int, default=3)
    parser.add_argument("--day", type=int, default=4)
    parser.add_argument("--hour", type=int, default=15)
    args = parser.parse_args()

    snap_path = SNAPSHOT_DIR / args.snapshot
    if not snap_path.exists():
        print(f"Snapshot not found: {snap_path}", file=sys.stderr)
        print(f"Available: {[p.name for p in SNAPSHOT_DIR.glob('*.pkl')]}",
              file=sys.stderr)
        sys.exit(1)

    print(f"=== Per-stage isolation drill ===")
    print(f"  Snapshot: {snap_path}")
    print(f"  Target: {args.year}-{args.month:02d}-{args.day:02d} h{args.hour:02d}")

    target = "L+35p0_+100p0"
    lat, lon = 35.0, 100.0
    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)

    # Load snapshot
    with open(snap_path, "rb") as fh:
        snap = pickle.load(fh)
    print(f"  Snapshot loaded; restoring F77 state...")
    restore_state(f77, snap)
    state = snap["jax_state"]
    print(f"  Restoration complete.")

    # Build atm forcing for the target hour
    gswp3 = read_gswp3_month(args.year, args.month, cells)
    jday = (date(args.year, args.month, 1) - date(args.year, 1, 1)).days + args.day
    day_data = {k: gswp3[k][args.day - 1, :] for k in gswp3}
    lats = np.full(NPOI, lat)
    atm = diurnal_disaggregate(day_data, float(args.hour), jday, lats)
    f77.set_atm(**atm)

    print(f"\n  ATM forcing: ta={float(atm['ta'][0]):.3f}  qa={float(atm['qa'][0]):.4e}  "
          f"ua={float(atm['ua'][0]):.3f}  psurf={float(atm['psurf'][0]):.1f}")

    # Pre-step NaN check
    nans = has_nan(f77, "pre")
    if nans:
        print(f"\n  PRE-STAGE NaNs already present: {nans}")
        return

    # Run F77 stages one by one with NaN check
    stages = [
        ("setsoi",     f77.call_setsoi),
        ("fwetcal_pre",f77.call_fwetcal),
        ("radiation",  f77.call_radiation),
        ("cascade",    f77.call_cascade),
        ("fwetcal_post",f77.call_fwetcal),
        ("drystress",  f77.call_drystress),
        ("canopy",     f77.call_canopy),
        ("cascad2",    f77.call_cascad2),
        ("noveg",      f77.call_noveg),
        ("soilctl",    f77.call_soilctl),
        ("snow",       f77.call_snow),
    ]
    print(f"\n  Running F77 stages...")
    for sname, sfunc in stages:
        try:
            sfunc()
        except Exception as e:
            print(f"  {sname}: EXCEPTION — {e}")
            return
        nans = has_nan(f77, f"after {sname}")
        if nans:
            print(f"\n*** FIRST NaN appears AFTER {sname.upper()} ***")
            for name, cells in nans:
                print(f"    {name}: NaN in cells {cells}")
            return
        print(f"    {sname:14s} → OK")

    print(f"\n  All stages OK — F77 did NOT produce NaN at this hour with restored state.")
    print(f"  (May indicate restoration was incomplete or NaN trigger condition not reproduced.)")


if __name__ == "__main__":
    main()
