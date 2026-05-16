"""Phases E+B combined: lockstep to step 68, then state-inject and bisect step 69.

Goal: identify WHICH lsxmain stage first introduces the F77↔JAX divergence
that causes the wsoi[:,0] for cell 0 to jump 5.5e-8 → 4.7e-5 in one step.

Strategy:
  1. Run F77 + JAX in lockstep for 68 steps under heat-wave forcing.
  2. After step 68: capture both states. They should differ by ~5.5e-8 in
     wsoi[:,0] (the accumulated drift up to that point).
  3. INJECT F77's state into JAX (force perfect alignment). Verify by
     re-reading.
  4. Run step 69 in BOTH F77 and JAX from the IDENTICAL aligned state.
  5. After step 69: compare every observable field. The diverging fields
     are caused by step-69 physics on identical inputs.

If JAX matches F77 bit-exactly after step 69 with aligned inputs → bug is
in state CARRY (some field that's not injected, or a hidden internal
state). Need broader state alignment.

If JAX still diverges → bug is in step-physics for this specific regime.
Drill down into individual lsxmain stages using wrap_call_X.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=python-port verify/.venv/bin/python -m verify.timestepping.pipeline.phase_eb_lockstep_per_stage
"""
from __future__ import annotations
import sys
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from verify.timestepping.verify_lsxmain import (
    setup_inputs, setup_f77_state, setup_jax_state, diurnal_atm,
    update_jax_atm, heat_wave_forcing, NPOI, CELL_LABELS, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body


def capture_f77_full_state(f77) -> dict:
    """Capture all F77 state that affects subsequent timesteps."""
    soil = f77.get_soil()
    snow = f77.get_snow()
    veg = f77.get_veg()
    stom = f77.get_stomata()
    rad = f77.get_radiation()
    flux = f77.get_fluxes()
    hydro = f77.get_hydro()
    aero = f77.get_aerodynamic()
    return dict(soil=soil, snow=snow, veg=veg, stom=stom,
                rad=rad, flux=flux, hydro=hydro, aero=aero)


def inject_f77_state_into_jax(jax_state, f77_state: dict, params):
    """Force JAX state to match F77 state for major fields."""
    soil_f = f77_state["soil"]
    snow_f = f77_state["snow"]
    veg_f = f77_state["veg"]
    stom_f = f77_state["stom"]

    new_soil = eqx.tree_at(
        lambda s: (s.wsoi, s.wisoi, s.tsoi, s.tg, s.wpud, s.wipud,
                   s.stresstu, s.stresstl, s.consoi, s.qglif, s.hvasug),
        jax_state.soil,
        (jnp.asarray(soil_f["wsoi"]),
         jnp.asarray(soil_f["wisoi"]),
         jnp.asarray(soil_f["tsoi"]),
         jnp.asarray(soil_f["tg"]),
         jnp.asarray(soil_f["wpud"]),
         jnp.asarray(soil_f["wipud"]),
         jnp.asarray(stom_f["stresstu"]),
         jnp.asarray(stom_f["stresstl"]),
         jnp.asarray(soil_f["consoi"]),
         jnp.asarray(soil_f["qglif"]),
         jnp.asarray(soil_f["hvasug"])),
    )
    new_snow = eqx.tree_at(
        lambda s: (s.tsno, s.hsno, s.fi, s.tlsub, s.hvasui),
        jax_state.snow,
        (jnp.asarray(snow_f["tsno"]),
         jnp.asarray(snow_f["hsno"]),
         jnp.asarray(snow_f["fi"]),
         jnp.asarray(veg_f["tlsub"]),
         jnp.asarray(soil_f["hvasui"])),
    )
    new_canopy = eqx.tree_at(
        lambda c: (c.tu, c.ts, c.tl, c.t12, c.t34, c.q12, c.q34,
                   c.wliqu, c.wliqs, c.wliql, c.wsnou, c.wsnos, c.wsnol,
                   c.fwetu, c.fwets, c.fwetl, c.rliqu, c.rliqs, c.rliql),
        jax_state.canopy,
        (jnp.asarray(veg_f["tu"]), jnp.asarray(veg_f["ts"]),
         jnp.asarray(veg_f["tl"]),
         jnp.asarray(veg_f["t12"]), jnp.asarray(veg_f["t34"]),
         jnp.asarray(veg_f["q12"]), jnp.asarray(veg_f["q34"]),
         jnp.asarray(veg_f["wliqu"]), jnp.asarray(veg_f["wliqs"]),
         jnp.asarray(veg_f["wliql"]),
         jnp.asarray(veg_f["wsnou"]), jnp.asarray(veg_f["wsnos"]),
         jnp.asarray(veg_f["wsnol"]),
         jnp.asarray(veg_f["fwetu"]), jnp.asarray(veg_f["fwets"]),
         jnp.asarray(veg_f["fwetl"]),
         jnp.asarray(veg_f["rliqu"]), jnp.asarray(veg_f["rliqs"]),
         jnp.asarray(veg_f["rliql"])),
    )
    return eqx.tree_at(
        lambda s: (s.soil, s.snow, s.canopy),
        jax_state,
        (new_soil, new_snow, new_canopy),
    )


def diff_states(label, f77_state: dict, jax_state, *, cells=(0, 1, 2, 3, 4),
                threshold=1e-12) -> list:
    """Compare F77 state vs JAX state across many fields. Return diff list."""
    diffs = []

    def add(name, f_val, j_val, *, layer=None):
        f = np.asarray(f_val).astype(np.float64)
        j = np.asarray(j_val).astype(np.float64)
        if layer is not None and f.ndim > 1:
            f = f[:, layer]; j = j[:, layer]
        if f.shape != j.shape:
            return
        for c in cells:
            if c >= f.size:
                continue
            d = abs(float(f.ravel()[c]) - float(j.ravel()[c]))
            if d > threshold:
                diffs.append((name, c, d, float(f.ravel()[c]), float(j.ravel()[c])))

    f_soil = f77_state["soil"]
    f_snow = f77_state["snow"]
    f_veg  = f77_state["veg"]
    f_stom = f77_state["stom"]
    f_rad  = f77_state["rad"]
    f_flux = f77_state["flux"]
    f_hydro= f77_state["hydro"]

    # Soil state (per-layer)
    for layer in range(6):
        add(f"wsoi[:,{layer}]",  f_soil["wsoi"],  jax_state.soil.wsoi,  layer=layer)
        add(f"wisoi[:,{layer}]", f_soil["wisoi"], jax_state.soil.wisoi, layer=layer)
        add(f"tsoi[:,{layer}]",  f_soil["tsoi"],  jax_state.soil.tsoi,  layer=layer)
    add("wpud",   f_soil["wpud"],   jax_state.soil.wpud)
    add("wipud",  f_soil["wipud"],  jax_state.soil.wipud)
    add("tg",     f_soil["tg"],     jax_state.soil.tg)
    add("heatg",  f_soil["heatg"],  jax_state.soil.heatg)
    add("hvasug", f_soil["hvasug"], jax_state.soil.hvasug)
    add("hvasui", f_soil["hvasui"], jax_state.snow.hvasui)
    add("qglif[:,0]", f_soil["qglif"], jax_state.soil.qglif, layer=0)
    add("qglif[:,1]", f_soil["qglif"], jax_state.soil.qglif, layer=1)
    add("consoi[:,0]", f_soil["consoi"], jax_state.soil.consoi, layer=0)

    # Snow state
    add("snow.fi",    f_snow["fi"],    jax_state.snow.fi)
    add("snow.tsno[:,0]", f_snow["tsno"], jax_state.snow.tsno, layer=0)
    add("snow.hsno[:,0]", f_snow["hsno"], jax_state.snow.hsno, layer=0)

    # Canopy temps + humidities
    for fname, jname in [("tu", "tu"), ("ts", "ts"), ("tl", "tl"),
                          ("t12", "t12"), ("t34", "t34"),
                          ("q12", "q12"), ("q34", "q34")]:
        add(f"canopy.{fname}", f_veg[fname], getattr(jax_state.canopy, jname))

    # Intercepted water
    for fname in ("wliqu", "wliqs", "wliql", "wsnou", "wsnos", "wsnol"):
        add(f"canopy.{fname}", f_veg[fname], getattr(jax_state.canopy, fname))

    # Wet fractions
    for fname in ("fwetu", "fwets", "fwetl", "rliqu", "rliqs", "rliql"):
        add(f"canopy.{fname}", f_veg[fname], getattr(jax_state.canopy, fname))

    # Radiation
    for fname in ("solu", "sols", "soll", "solg", "soli",
                   "firu", "firs", "firl", "firg", "firi", "firb",
                   "topparu", "topparl"):
        add(f"rad.{fname}", f_rad[fname], getattr(jax_state.canopy, fname))
    for band in range(2):
        add(f"asurd[:,{band}]", f_rad["asurd"], jax_state.canopy.asurd, layer=band)
        add(f"asuri[:,{band}]", f_rad["asuri"], jax_state.canopy.asuri, layer=band)

    # Fluxes
    for fname in ("fsenu", "fsens", "fsenl", "fseng", "fseni"):
        add(f"flux.{fname}", f_flux[fname], getattr(jax_state.canopy, fname))
    for fname in ("fvapg", "fvapi"):
        add(f"flux.{fname}", f_flux[fname], getattr(jax_state.canopy, fname))
    add("flux.raing", f_flux["raing"], jax_state.canopy.raing)
    add("flux.snowg", f_flux["snowg"], jax_state.canopy.snowg)

    # Stomata state
    for fname in ("ci_upper", "gs_upper", "cs_upper", "ci_lower", "gs_lower", "cs_lower"):
        f_arr = f_stom[fname.replace("_upper","ub").replace("_lower","ls")
                       if False else f"{fname[:2]}{'ub' if 'upper' in fname else 'ls'}"]
        # Awkward — skip detailed stomata for now
        pass

    # Soil stress
    add("stresstu", f_stom["stresstu"], jax_state.soil.stresstu)
    add("stresstl", f_stom["stresstl"], jax_state.soil.stresstl)

    # Hydrology
    add("hydro.gtrans",  f_hydro["gtrans"],  jax_state.hydrology.gtrans)
    add("hydro.gtransu", f_hydro["gtransu"], jax_state.hydrology.gtransu)
    add("hydro.gtransl", f_hydro["gtransl"], jax_state.hydrology.gtransl)
    add("hydro.ginvap",  f_hydro["ginvap"],  jax_state.hydrology.ginvap)
    add("hydro.gsuvap",  f_hydro["gsuvap"],  jax_state.hydrology.gsuvap)

    return diffs


def report_diffs(label: str, diffs: list, top_n=30):
    print(f"\n=== {label} ===")
    if not diffs:
        print("  All fields match within threshold.")
        return
    diffs.sort(key=lambda x: -x[2])
    print(f"  Total fields with diff > threshold: {len(diffs)}")
    print(f"  Top {min(top_n, len(diffs))}:")
    print(f"  {'field':25s} {'cell':>6s} {'abs_diff':>12s}  {'F77':>15s}  {'JAX':>15s}")
    for name, cell, d, f, j in diffs[:top_n]:
        print(f"  {name:25s} {cell:>6d} {d:>12.3e}  {f:>+15.7e}  {j:>+15.7e}")


def main():
    soil_i, snow_i, atm_i, veg_i = setup_inputs()

    print("=" * 72)
    print("LOCKSTEP PHASE — running F77 + JAX in parallel for 68 steps")
    print("=" * 72)

    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    for step in range(68):
        hour = float(step % 24)
        new_atm = diurnal_atm(atm_i, hour, **heat_wave_forcing(step, 168))
        f77.set_atm(**new_atm)
        state = update_jax_atm(state, new_atm)
        f77.call_lsxmain()
        state = timestep_body(state, consts, params, DTIME)
    print(f"After 68 steps: cell 0 wsoi[0] F77={float(f77.get_soil()['wsoi'][0,0]):.10f} "
          f"JAX={float(state.soil.wsoi[0,0]):.10f}")

    # Capture state at step 68 end
    f77_state_68 = capture_f77_full_state(f77)
    state_68_jax_orig = state

    # === DIFF before injection (natural drift) ===
    diffs_before = diff_states("before injection", f77_state_68, state_68_jax_orig,
                                threshold=1e-15)
    report_diffs("STATE AFTER STEP 68 — F77 vs JAX (natural drift)",
                 diffs_before, top_n=50)

    # === Inject F77 state into JAX ===
    print()
    print("=" * 72)
    print("INJECTING F77 state INTO JAX (forced alignment at step 68 end)")
    print("=" * 72)
    state_injected = inject_f77_state_into_jax(state_68_jax_orig, f77_state_68, params)

    # Sanity check: confirm injection worked
    diffs_post_inj = diff_states("post-injection sanity", f77_state_68, state_injected,
                                  threshold=1e-15)
    print(f"\nPost-injection diff count: {len(diffs_post_inj)}")
    if diffs_post_inj:
        # Group by field name, show ALL
        from collections import defaultdict
        by_field = defaultdict(list)
        for name, cell, d, f, j in diffs_post_inj:
            by_field[name].append((cell, d, f, j))
        print("  Fields not aligned by injection (groups by field name):")
        for name in sorted(by_field.keys()):
            entries = by_field[name]
            max_d = max(e[1] for e in entries)
            n = len(entries)
            print(f"    {name:25s} ({n} cells, max diff {max_d:.3e})")

    # === Run step 69 from aligned state ===
    print()
    print("=" * 72)
    print("RUNNING STEP 69 from aligned state")
    print("=" * 72)

    hour_69 = float(68 % 24)
    new_atm_69 = diurnal_atm(atm_i, hour_69, **heat_wave_forcing(68, 168))
    f77.set_atm(**new_atm_69)
    state_aligned = update_jax_atm(state_injected, new_atm_69)

    # Pre-step-69 sanity: print key inputs to drystress
    print()
    print("PRE-step-69 sanity (drystress inputs for cell 2):")
    f_pre = f77.get_soil()
    print(f"  F77  wsoi[2,:] = {f_pre['wsoi'][2]}")
    print(f"  JAX  wsoi[2,:] = {np.asarray(state_aligned.soil.wsoi[2])}")
    print(f"  F77  wisoi[2,:] = {f_pre['wisoi'][2]}")
    print(f"  JAX  wisoi[2,:] = {np.asarray(state_aligned.soil.wisoi[2])}")

    # Run drystress on JAX side EXPLICITLY to capture its output
    from ibis.physics.stomata import drystress as _ds
    js_stresstl, js_stresstu, js_stressl, js_stressu = _ds(
        state_aligned.soil.wsoi, state_aligned.soil.wisoi,
        state_aligned.soil.swilt, state_aligned.soil.sfield,
        state_aligned.vegetation.froot,
    )
    print(f"  JAX drystress-alone output stresstu[2] = {float(js_stresstu[2]):.10f}")
    print(f"  JAX drystress-alone output stresstl[2] = {float(js_stresstl[2]):.10f}")

    f77.call_lsxmain()
    state_69_jax = timestep_body(state_aligned, consts, params, DTIME)
    print(f"  F77 after lsxmain stresstu[2] = {float(f77.get_stomata()['stresstu'][2]):.10f}")
    print(f"  F77 after lsxmain stresstl[2] = {float(f77.get_stomata()['stresstl'][2]):.10f}")
    print(f"  JAX after lsxmain stresstu[2] = {float(state_69_jax.soil.stresstu[2]):.10f}")
    print(f"  JAX after lsxmain stresstl[2] = {float(state_69_jax.soil.stresstl[2]):.10f}")

    f77_state_69 = capture_f77_full_state(f77)
    diffs_69 = diff_states("step 69 from aligned state",
                            f77_state_69, state_69_jax, threshold=1e-12)
    report_diffs("STATE AFTER STEP 69 — F77 vs JAX (FROM ALIGNED INPUT)",
                 diffs_69, top_n=50)

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    big_count = sum(1 for _, _, d, _, _ in diffs_69 if d > 1e-6)
    if big_count == 0:
        print("  Cell 0 wsoi[0] and other fields match at machine precision after")
        print("  step 69 from aligned state. The bug is in MULTI-STEP CARRY:")
        print("  some hidden state field is not being matched between F77 and JAX,")
        print("  accumulating drift. Look at fields with small but non-zero diff")
        print("  in 'natural drift' that aren't reset by injection.")
    else:
        print(f"  {big_count} fields show diff > 1e-6 after step 69 from aligned input.")
        print("  This is the per-step-physics bug. Drill down with per-stage bisection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
