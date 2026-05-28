"""NaN-onset drill for finding #26 — L+35°N +100°E (Tibetan Plateau cell).

Cell ran cleanly from 1951-01 through 1960-02 in the 35-year multi-year
verification, then BOTH F77 and JAX went NaN simultaneously starting
March 1960. Same diagnostic signature as finding #20.

This script:
  1. Replicates the target cell NPOI=5 times so all batch slots are the
     same cell (lockstep convention).
  2. Seeds silently from 1951-01 through 1960-02 (9 years 2 months).
  3. From 1960-03-01 h00 forward, runs HOURLY with full state checks:
     for each timestep, before and after the step, scan every tracked
     field for NaN. Print a wide diff dump when the FIRST NaN appears.
  4. Stops at the first NaN-producing timestep with full state context.

Output: identifies the (year, month, day, hour) of first NaN, the field
that first went NaN, and the pre-step state that triggered it. Then we
can map this to the F77 subroutine producing the NaN.
"""
from __future__ import annotations
from datetime import date
import math
import os
import pickle
import sys
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
from ibis.timestepping.loops import default_sim_params, timestep_body
from tests.f2py.ibis_f77_wrapper import NPOI


# Comprehensive field set to monitor — includes state vars, fluxes, and
# intermediate solver variables when accessible.
FIELDS_F77 = [
    # canopy temperatures
    ("canopy.tu",    lambda f: f.get_veg()["tu"][0]),
    ("canopy.ts",    lambda f: f.get_veg()["ts"][0]),
    ("canopy.tl",    lambda f: f.get_veg()["tl"][0]),
    ("canopy.t12",   lambda f: f.get_veg()["t12"][0]),
    ("canopy.t34",   lambda f: f.get_veg()["t34"][0]),
    ("canopy.q12",   lambda f: f.get_veg()["q12"][0]),
    ("canopy.q34",   lambda f: f.get_veg()["q34"][0]),
    ("canopy.wliqu", lambda f: f.get_veg()["wliqu"][0]),
    ("canopy.wsnou", lambda f: f.get_veg()["wsnou"][0]),
    ("canopy.wliqs", lambda f: f.get_veg()["wliqs"][0]),
    ("canopy.wsnos", lambda f: f.get_veg()["wsnos"][0]),
    ("canopy.wliql", lambda f: f.get_veg()["wliql"][0]),
    ("canopy.wsnol", lambda f: f.get_veg()["wsnol"][0]),
    ("canopy.fwetu", lambda f: f.get_veg()["fwetu"][0]),
    ("canopy.fwetl", lambda f: f.get_veg()["fwetl"][0]),
    # soil
    ("soil.tg",      lambda f: f.get_soil()["tg"][0]),
    ("soil.tsoi[0]", lambda f: f.get_soil()["tsoi"][0, 0]),
    ("soil.tsoi[1]", lambda f: f.get_soil()["tsoi"][0, 1]),
    ("soil.tsoi[2]", lambda f: f.get_soil()["tsoi"][0, 2]),
    ("soil.tsoi[3]", lambda f: f.get_soil()["tsoi"][0, 3]),
    ("soil.wsoi[0]", lambda f: f.get_soil()["wsoi"][0, 0]),
    ("soil.wsoi[1]", lambda f: f.get_soil()["wsoi"][0, 1]),
    ("soil.wisoi[0]",lambda f: f.get_soil()["wisoi"][0, 0]),
    ("soil.wisoi[1]",lambda f: f.get_soil()["wisoi"][0, 1]),
    ("soil.wpud",    lambda f: f.get_soil()["wpud"][0]),
    ("soil.wipud",   lambda f: f.get_soil()["wipud"][0]),
    # snow
    ("snow.fi",      lambda f: f.get_snow()["fi"][0]),
    ("snow.tsno[0]", lambda f: f.get_snow()["tsno"][0, 0]),
    ("snow.tsno[1]", lambda f: f.get_snow()["tsno"][0, 1]),
    ("snow.tsno[2]", lambda f: f.get_snow()["tsno"][0, 2]),
    ("snow.hsno[0]", lambda f: f.get_snow()["hsno"][0, 0]),
    ("snow.hsno[1]", lambda f: f.get_snow()["hsno"][0, 1]),
    ("snow.hsno[2]", lambda f: f.get_snow()["hsno"][0, 2]),
    # fluxes
    ("flux.fseng",   lambda f: f.get_fluxes()["fseng"][0]),
    ("flux.fsenu",   lambda f: f.get_fluxes()["fsenu"][0]),
    ("flux.fsenl",   lambda f: f.get_fluxes()["fsenl"][0]),
    ("flux.fseni",   lambda f: f.get_fluxes()["fseni"][0]),
    ("flux.fvapa",   lambda f: f.get_fluxes()["fvapa"][0]),
    ("flux.fvapg",   lambda f: f.get_fluxes()["fvapg"][0]),
    # stomata
    ("gsub",         lambda f: f.get_stomata()["gsub"][0]),
    ("gsls",         lambda f: f.get_stomata()["gsls"][0]),
    ("ciub",         lambda f: f.get_stomata()["ciub"][0]),
    ("cils",         lambda f: f.get_stomata()["cils"][0]),
    ("totcondub",    lambda f: f.get_stomata()["totcondub"][0]),
    ("totcondls",    lambda f: f.get_stomata()["totcondls"][0]),
    ("stresstu",     lambda f: f.get_stomata()["stresstu"][0]),
    ("stresstl",     lambda f: f.get_stomata()["stresstl"][0]),
]


def has_nan_f77(f77):
    """Check if any tracked F77 field is NaN. Returns list of NaN fields."""
    nan_fields = []
    for name, getter in FIELDS_F77:
        try:
            v = float(getter(f77))
            if math.isnan(v):
                nan_fields.append((name, v))
        except (KeyError, IndexError):
            pass
    return nan_fields


def has_nan_jax(state):
    """Check if any JAX state field is NaN. Returns list of NaN fields."""
    checks = [
        ("canopy.tu", state.canopy.tu[0]),
        ("canopy.ts", state.canopy.ts[0]),
        ("canopy.tl", state.canopy.tl[0]),
        ("canopy.t12", state.canopy.t12[0]),
        ("canopy.t34", state.canopy.t34[0]),
        ("canopy.q12", state.canopy.q12[0]),
        ("canopy.q34", state.canopy.q34[0]),
        ("canopy.wliqu", state.canopy.wliqu[0]),
        ("canopy.wsnou", state.canopy.wsnou[0]),
        ("canopy.wliqs", state.canopy.wliqs[0]),
        ("canopy.wsnos", state.canopy.wsnos[0]),
        ("canopy.wliql", state.canopy.wliql[0]),
        ("canopy.wsnol", state.canopy.wsnol[0]),
        ("soil.tg", state.soil.tg[0]),
        ("soil.tsoi[0]", state.soil.tsoi[0, 0]),
        ("soil.wsoi[0]", state.soil.wsoi[0, 0]),
        ("snow.fi", state.snow.fi[0]),
        ("snow.tsno[0]", state.snow.tsno[0, 0]),
        ("snow.hsno[0]", state.snow.hsno[0, 0]),
    ]
    return [(n, float(v)) for n, v in checks if math.isnan(float(v))]


def dump_state(f77, state, label, hour, day, month, atm):
    """Print a full F77 vs JAX state comparison."""
    print(f"\n=== {label} at {month:02d}-{day:02d} h{hour:02d} ===")
    print(f"  ta={float(atm['ta'][0]):.3f}  qa={float(atm['qa'][0]):.4e}  "
          f"ua={float(atm['ua'][0]):.3f}  psurf={float(atm['psurf'][0]):.1f}")
    solad_total = float(np.asarray(atm['solad'][0]).sum())
    solai_total = float(np.asarray(atm['solai'][0]).sum())
    print(f"  coszen={float(atm['coszen'][0]):.4f}  solad_total={solad_total:.1f}  "
          f"solai_total={solai_total:.1f}  fira={float(atm['fira'][0]):.1f}  "
          f"raina={float(atm['raina'][0]):.3e}  snowa={float(atm['snowa'][0]):.3e}")
    print(f"  {'field':18s} {'F77':>14s} {'JAX':>14s} {'|Δ|':>10s}")
    for name, getter in FIELDS_F77:
        try:
            fv = float(getter(f77))
        except (KeyError, IndexError):
            fv = float('nan')
        # JAX equivalent (only for fields with direct counterparts)
        jv_map = {
            "canopy.tu": state.canopy.tu[0], "canopy.ts": state.canopy.ts[0],
            "canopy.tl": state.canopy.tl[0], "canopy.t12": state.canopy.t12[0],
            "canopy.t34": state.canopy.t34[0], "canopy.q12": state.canopy.q12[0],
            "canopy.q34": state.canopy.q34[0],
            "canopy.wliqu": state.canopy.wliqu[0], "canopy.wsnou": state.canopy.wsnou[0],
            "canopy.wliqs": state.canopy.wliqs[0], "canopy.wsnos": state.canopy.wsnos[0],
            "canopy.wliql": state.canopy.wliql[0], "canopy.wsnol": state.canopy.wsnol[0],
            "soil.tg": state.soil.tg[0], "soil.tsoi[0]": state.soil.tsoi[0, 0],
            "soil.wsoi[0]": state.soil.wsoi[0, 0], "soil.wisoi[0]": state.soil.wisoi[0, 0],
            "snow.fi": state.snow.fi[0], "snow.tsno[0]": state.snow.tsno[0, 0],
            "snow.hsno[0]": state.snow.hsno[0, 0],
        }
        if name in jv_map:
            jv = float(jv_map[name])
            d = abs(fv - jv) if not (math.isnan(fv) or math.isnan(jv)) else float('nan')
            marker = " NaN!" if math.isnan(fv) or math.isnan(jv) else ""
            print(f"  {name:18s} {fv:>14.6e} {jv:>14.6e} {d:>10.3e}{marker}")
        else:
            marker = " NaN!" if math.isnan(fv) else ""
            print(f"  {name:18s} {fv:>14.6e}                              {marker}")


SNAPSHOT_DIR = Path("/shared/nas2/Hangkai/ibis/verify_outputs/tibet_nan_snapshots")


def save_snapshot(f77, state, label):
    """Save F77 + JAX state to NAS for future fast re-investigation."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap = {
        "veg":  {k: np.asarray(v).copy() for k, v in f77.get_veg().items()},
        "soil": {k: np.asarray(v).copy() for k, v in f77.get_soil().items()},
        "snow": {k: np.asarray(v).copy() for k, v in f77.get_snow().items()},
        "stom": {k: np.asarray(v).copy() for k, v in f77.get_stomata().items()},
        "jax_state": state,
    }
    with open(SNAPSHOT_DIR / f"{label}.pkl", "wb") as fh:
        pickle.dump(snap, fh)


def main():
    target = "L+35p0_+100p0"
    lat, lon = 35.0, 100.0
    YEAR_START = 1951
    YEAR_DRILL = 1960
    MONTH_DRILL = 3  # March 1960
    print(f"=== NaN-onset drill for {target} (finding #26) ===")
    print(f"    Seeding {YEAR_START}-01 → {YEAR_DRILL}-02, then verbose")
    print(f"    {YEAR_DRILL}-{MONTH_DRILL:02d} forward")
    print(f"    Snapshots → {SNAPSHOT_DIR}")

    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    lats = np.full(NPOI, lat)

    sys.stdout.reconfigure(line_buffering=True)

    # Snapshot save lets future analyses reload state without re-seeding 9 years.
    # (Fast-load support is left for a future revision; the F77 wrapper's
    # set_* methods don't expose all common-block state, so a full restart
    # requires more careful design.)

    step = 0
    found_nan = False
    for year in range(YEAR_START, YEAR_DRILL + 1):
        for month in range(1, 13):
            if year == YEAR_DRILL and month > MONTH_DRILL:
                # Stop after the drill month (don't run further)
                if found_nan:
                    return
                print(f"\n  Reached end of {YEAR_DRILL}-{MONTH_DRILL:02d}; no NaN detected.")
                return

            # Save snapshot at start of March 1960 (so we never re-seed)
            if year == YEAR_DRILL and month == MONTH_DRILL:
                print(f"\n  SAVE: writing Feb 1960 end snapshot...", flush=True)
                save_snapshot(f77, state, "feb_1960_end")
                print(f"  SAVE: snapshot saved → {SNAPSHOT_DIR}/feb_1960_end.pkl",
                      flush=True)

            if month == 12:
                next_m = date(year + 1, 1, 1)
            else:
                next_m = date(year, month + 1, 1)
            n_days = (next_m - date(year, month, 1)).days

            gswp3 = read_gswp3_month(year, month, cells)

            verbose = (year == YEAR_DRILL and month == MONTH_DRILL)
            if month == 1 and year % 1 == 0 and not verbose:
                print(f"  seeding {year}-{month:02d}…", flush=True)

            for day_idx in range(n_days):
                jday = (date(year, month, 1) - date(year, 1, 1)).days + day_idx + 1
                day_data = {k: gswp3[k][day_idx, :] for k in gswp3}
                for hour in range(24):
                    atm = diurnal_disaggregate(day_data, float(hour), jday, lats)

                    # Pre-step NaN check (state could already be NaN)
                    if verbose:
                        pre_nan_f = has_nan_f77(f77)
                        pre_nan_j = has_nan_jax(state)
                        if (pre_nan_f or pre_nan_j) and not found_nan:
                            print(f"\n*** PRE-STEP NaN at {year}-{month:02d}-"
                                  f"{day_idx+1:02d} h{hour:02d} (before this step ran) ***")
                            print(f"  F77 NaN fields: {pre_nan_f}")
                            print(f"  JAX NaN fields: {pre_nan_j}")
                            found_nan = True
                            dump_state(f77, state, "PRE-STEP STATE",
                                       hour, day_idx+1, month, atm)
                            return

                    f77.set_atm(**atm)
                    state_atm = eqx.tree_at(
                        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                                   s.atm.coszen, s.atm.solad, s.atm.solai,
                                   s.atm.fira, s.atm.raina, s.atm.snowa),
                        state,
                        tuple(jnp.asarray(atm[k]) for k in
                              ("ta","qa","ua","psurf","coszen","solad","solai",
                               "fira","raina","snowa")),
                    )
                    # Capture pre-step F77 state for the dump
                    if verbose:
                        # cheap pre-step snapshot of key fields
                        pre_tu = float(f77.get_veg()["tu"][0])
                        pre_tg = float(f77.get_soil()["tg"][0])
                        pre_fi = float(f77.get_snow()["fi"][0])

                    # Hourly snapshot during the drill — lets future drills replay
                    # any moment without re-seeding 9 years.
                    if verbose and not found_nan:
                        save_snapshot(f77, state,
                                      f"d{day_idx+1:02d}_h{hour:02d}_pre")

                    # At the known NaN-onset hour, do per-stage isolation
                    # instead of call_lsxmain so we identify WHICH F77
                    # subroutine first produces NaN. Sequence matches the
                    # F77 lsxmain composition exactly (verified equivalent).
                    if verbose and year == YEAR_DRILL and month == MONTH_DRILL \
                       and day_idx + 1 == 4 and hour == 15:
                        print(f"\n*** Per-stage isolation at d04 h15 ***", flush=True)
                        import ctypes
                        _lib = ctypes.CDLL(
                            "/home/hangkai/ibis-port/python-port/tests/f2py/ibis_f77.so"
                        )
                        def call_compute_heatg_heati():
                            _lib.wrap_call_compute_heatg_heati_()
                        def ct_turcof(iter_num):
                            arg = ctypes.c_int(iter_num)
                            _lib.wrap_call_turcof_(ctypes.byref(arg))
                        def ct_turvap(iter_num, niter):
                            a1 = ctypes.c_int(iter_num); a2 = ctypes.c_int(niter)
                            _lib.wrap_call_turvap_(ctypes.byref(a1), ctypes.byref(a2))

                        # Run pre-canopy stages normally; then break canopy
                        # apart into drystress + 3-iter (turcof/stomata/turvap)
                        # to identify the EXACT sub-stage and iter producing NaN.
                        pre_canopy_stages = [
                            ("setsoi",            f77.call_setsoi),
                            ("fwetcal_pre",       f77.call_fwetcal),
                            ("radiation",         f77.call_radiation),
                            ("cascade",           f77.call_cascade),
                            ("fwetcal_post",      f77.call_fwetcal),
                        ]
                        for sname, sfunc in pre_canopy_stages:
                            sfunc()
                            f_nan = has_nan_f77(f77)
                            print(f"  after {sname:20s} → F77 NaN fields: "
                                  f"{[n for n,_ in f_nan]}", flush=True)
                            if f_nan:
                                print(f"\n>>> NaN AFTER {sname.upper()} (pre-canopy) <<<")
                                with open(SNAPSHOT_DIR / "nan_stage_report.txt", "w") as fh:
                                    fh.write(f"NaN appeared after F77 stage: {sname}\n")
                                return

                        # Snapshot just before canopy
                        save_snapshot(f77, state, "d04_h15_before_canopy")
                        print(f"  >>> PRE-CANOPY snapshot saved", flush=True)

                        # Decompose canopy: drystress + 3 iters
                        f77.call_drystress()
                        f_nan = has_nan_f77(f77)
                        print(f"  after drystress           → F77 NaN: "
                              f"{[n for n,_ in f_nan]}", flush=True)
                        if f_nan:
                            print(f"\n>>> FIRST NaN APPEARS AFTER DRYSTRESS <<<")
                            with open(SNAPSHOT_DIR / "nan_stage_report.txt", "w") as fh:
                                fh.write(f"NaN appeared after F77 stage: drystress\n")
                            return

                        niter = 3
                        for it in range(1, niter + 1):
                            f77.call_turcof(it)
                            f_nan = has_nan_f77(f77)
                            print(f"  iter {it} after turcof    → F77 NaN: "
                                  f"{[n for n,_ in f_nan]}", flush=True)
                            if f_nan:
                                print(f"\n>>> FIRST NaN: iter {it} TURCOF <<<")
                                save_snapshot(f77, state,
                                              f"d04_h15_iter{it}_turcof_nan")
                                with open(SNAPSHOT_DIR / "nan_stage_report.txt", "w") as fh:
                                    fh.write(f"NaN appeared at: iter {it} turcof\n")
                                    for n, v in f_nan:
                                        fh.write(f"  {n}\n")
                                return

                            f77.call_stomata()
                            f_nan = has_nan_f77(f77)
                            print(f"  iter {it} after stomata   → F77 NaN: "
                                  f"{[n for n,_ in f_nan]}", flush=True)
                            if f_nan:
                                print(f"\n>>> FIRST NaN: iter {it} STOMATA <<<")
                                save_snapshot(f77, state,
                                              f"d04_h15_iter{it}_stomata_nan")
                                with open(SNAPSHOT_DIR / "nan_stage_report.txt", "w") as fh:
                                    fh.write(f"NaN appeared at: iter {it} stomata\n")
                                    for n, v in f_nan:
                                        fh.write(f"  {n}\n")
                                return

                            f77.call_turvap(it, niter)
                            f_nan = has_nan_f77(f77)
                            print(f"  iter {it} after turvap    → F77 NaN: "
                                  f"{[n for n,_ in f_nan]}", flush=True)
                            if f_nan:
                                print(f"\n>>> FIRST NaN: iter {it} TURVAP <<<")
                                save_snapshot(f77, state,
                                              f"d04_h15_iter{it}_turvap_nan")
                                with open(SNAPSHOT_DIR / "nan_stage_report.txt", "w") as fh:
                                    fh.write(f"NaN appeared at: iter {it} turvap\n")
                                    for n, v in f_nan:
                                        fh.write(f"  {n}\n")
                                return

                        print(f"  All canopy sub-stages OK — no NaN. Unexpected.")
                        return

                    f77.call_lsxmain()
                    state = timestep_body(state_atm, consts, params, DTIME)
                    step += 1

                    if verbose:
                        post_nan_f = has_nan_f77(f77)
                        post_nan_j = has_nan_jax(state)
                        if (post_nan_f or post_nan_j) and not found_nan:
                            print(f"\n*** FIRST NaN at {year}-{month:02d}-"
                                  f"{day_idx+1:02d} h{hour:02d} (after this step) ***")
                            print(f"  PRE-STEP scalars: tu={pre_tu:.4f}  "
                                  f"tg={pre_tg:.4f}  fi={pre_fi:.4f}")
                            print(f"  F77 NaN fields ({len(post_nan_f)}): {[n for n,_ in post_nan_f]}")
                            print(f"  JAX NaN fields ({len(post_nan_j)}): {[n for n,_ in post_nan_j]}")
                            found_nan = True
                            try:
                                dump_state(f77, state, "POST-STEP STATE (NaN onset)",
                                           hour, day_idx+1, month, atm)
                            except Exception as e:
                                print(f"  (dump_state failed: {e})")
                            # Save NaN-onset snapshot too
                            save_snapshot(f77, state,
                                          f"d{day_idx+1:02d}_h{hour:02d}_post_nan")
                            return

                        # Per-hour heartbeat for the drill month
                        if hour in (0, 6, 12, 18):
                            tu = float(f77.get_veg()["tu"][0])
                            tg = float(f77.get_soil()["tg"][0])
                            fi = float(f77.get_snow()["fi"][0])
                            hsno = float(f77.get_snow()["hsno"][0, 0])
                            ta = float(atm["ta"][0])
                            print(f"  d{day_idx+1:02d} h{hour:02d}  "
                                  f"ta={ta:.2f}  tu={tu:.3f}  tg={tg:.3f}  "
                                  f"fi={fi:.4f}  hsno={hsno:.4f}",
                                  flush=True)


if __name__ == "__main__":
    main()
