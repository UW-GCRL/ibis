"""Fine-grained bisection diagnostic.

Track per-step F77-vs-JAX signed differences of ALL observable cell-0
quantities, then rank by per-step growth rate. The quantity that
introduces the largest systematic bias per step is the source.

Focus on cell 0 (tropical, actively transpiring — the only cell that
drifts). Run under base diurnal forcing for 100 steps so we can watch
the slow accumulation (heat-wave kicks in late and is too dramatic for
finding small per-step biases).
"""
from __future__ import annotations
import numpy as np
import equinox as eqx

from verify.timestepping.verify_lsxmain import (
    setup_inputs, setup_f77_state, setup_jax_state, diurnal_atm,
    update_jax_atm, base_forcing, heat_wave_forcing, NPOI, CELL_LABELS, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body


CELL = 0
LAYER = 0  # focus on top layer for layer-specific fields


def sd(name, f77_val, jax_val, *, layer=None):
    a = np.asarray(f77_val)
    b = np.asarray(jax_val)
    if layer is not None:
        a = a[:, layer] if a.ndim == 2 else a
        b = b[:, layer] if b.ndim == 2 else b
    if a.ndim == 0 or a.size == 1:
        af = float(a)
        bf = float(b)
    else:
        af = float(a.ravel()[CELL])
        bf = float(b.ravel()[CELL])
    return af - bf


def main():
    soil_i, snow_i, atm_i, veg_i = setup_inputs()
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    n_steps = 100
    print(f"Per-step F77 minus JAX diffs for cell {CELL} ({CELL_LABELS[CELL]})")
    print(f"HEAT WAVE forcing, {n_steps} steps")
    print()
    # Only fields where JAX has a matched state field — avoids placeholder bugs
    fields = [
        ("wsoi[:,0]", "wsoi top"),
        ("tsoi[:,0]", "tsoi top"),
        ("tg",        "tg"),
        ("solg",      "solg sun→ground"),
        ("solu",      "solu sun→up"),
        ("soll",      "soll sun→lo"),
        ("soli",      "soli sun→snow"),
        ("firg",      "firg ir→ground"),
        ("firu",      "firu ir→up"),
        ("firl",      "firl ir→lo"),
        ("asurd[:,0]","asurd VIS"),
        ("asurd[:,1]","asurd NIR"),
        ("asuri[:,0]","asuri VIS"),
        ("asuri[:,1]","asuri NIR"),
        ("topparu",   "topparu"),
        ("topparl",   "topparl"),
        ("tu",        "tu"),
        ("tl",        "tl"),
        ("ts",        "ts"),
        ("t12",       "t12"),
        ("t34",       "t34"),
        ("q12",       "q12"),
        ("q34",       "q34"),
        ("fvapg",     "fvapg"),
        ("fsenu",     "fsenu"),
        ("fsenl",     "fsenl"),
        ("fseng",     "fseng"),
        ("stresstu",  "stresstu"),
        ("stresstl",  "stresstl"),
    ]

    history = {f[0]: [] for f in fields}

    for step in range(n_steps):
        hour = float(step % 24)
        new_atm = diurnal_atm(atm_i, hour, **heat_wave_forcing(step, n_steps))
        f77.set_atm(**new_atm)
        state = update_jax_atm(state, new_atm)
        f77.call_lsxmain()
        state = timestep_body(state, consts, params, DTIME)

        f_soil = f77.get_soil()
        f_veg = f77.get_veg()
        f_flux = f77.get_fluxes()
        f_hydro = f77.get_hydro()
        f_stom = f77.get_stomata()

        f_rad = f77.get_radiation()
        diffs = {
            "wsoi[:,0]":  sd("", f_soil["wsoi"][:, 0], state.soil.wsoi[:, 0]),
            "tsoi[:,0]":  sd("", f_soil["tsoi"][:, 0], state.soil.tsoi[:, 0]),
            "tg":         sd("", f_soil["tg"], state.soil.tg),
            "solg":       sd("", f_rad["solg"], state.canopy.solg),
            "solu":       sd("", f_rad["solu"], state.canopy.solu),
            "soll":       sd("", f_rad["soll"], state.canopy.soll),
            "soli":       sd("", f_rad["soli"], state.canopy.soli),
            "firg":       sd("", f_rad["firg"], state.canopy.firg),
            "firu":       sd("", f_rad["firu"], state.canopy.firu),
            "firl":       sd("", f_rad["firl"], state.canopy.firl),
            "asurd[:,0]": sd("", f_rad["asurd"][:, 0], state.canopy.asurd[:, 0]),
            "asurd[:,1]": sd("", f_rad["asurd"][:, 1], state.canopy.asurd[:, 1]),
            "asuri[:,0]": sd("", f_rad["asuri"][:, 0], state.canopy.asuri[:, 0]),
            "asuri[:,1]": sd("", f_rad["asuri"][:, 1], state.canopy.asuri[:, 1]),
            "topparu":    sd("", f_rad["topparu"], state.canopy.topparu),
            "topparl":    sd("", f_rad["topparl"], state.canopy.topparl),
            "tu":         sd("", f_veg["tu"], state.canopy.tu),
            "tl":         sd("", f_veg["tl"], state.canopy.tl),
            "ts":         sd("", f_veg["ts"], state.canopy.ts),
            "t12":        sd("", f_veg["t12"], state.canopy.t12),
            "t34":        sd("", f_veg["t34"], state.canopy.t34),
            "q12":        sd("", f_veg["q12"], state.canopy.q12),
            "q34":        sd("", f_veg["q34"], state.canopy.q34),
            "fvapg":      sd("", f_flux["fvapg"], state.canopy.fvapg),
            "fsenu":      sd("", f_flux["fsenu"], state.canopy.fsenu),
            "fsenl":      sd("", f_flux["fsenl"], state.canopy.fsenl),
            "fseng":      sd("", f_flux["fseng"], state.canopy.fseng),
            "stresstu":   sd("", f_stom["stresstu"], state.soil.stresstu),
            "stresstl":   sd("", f_stom["stresstl"], state.soil.stresstl),
        }
        for k, v in diffs.items():
            history[k].append(v)

    # Print final-step diffs sorted by magnitude
    print(f"Final step ({n_steps - 1}) signed F77-JAX diffs for cell {CELL}:")
    last = [(name, label, history[name][-1]) for name, label in fields]
    last.sort(key=lambda x: -abs(x[2]))
    for name, label, v in last:
        marker = " >>" if abs(v) > 1e-6 else ""
        print(f"  {label:12s} {name:12s} = {v:+12.3e}{marker}")
    print()

    # Print per-step growth pattern for the top divergent fields
    print("Per-step trajectory (selected fields, in scientific notation):")
    top_fields = [t[0] for t in last[:8] if abs(t[2]) > 1e-12]
    sample_steps = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    header = f"{'step':>4s}  " + "  ".join(f"{f[:11]:>11s}" for f in top_fields)
    print(header)
    print("-" * len(header))
    for step in sample_steps:
        cells = "  ".join(f"{history[f][step]:>+11.2e}" for f in top_fields)
        print(f"{step:>4d}  {cells}")


if __name__ == "__main__":
    main()
