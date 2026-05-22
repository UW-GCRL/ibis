"""State-injection test at d115 h07 for L+55°N +100°E.

Procedure:
  1. Run lockstep F77+JAX through d115 h06 (each evolves its own state).
  2. INJECT F77 state into JAX state at end of h06.
  3. Run F77 lsxmain and JAX timestep_body at h07 with IDENTICAL inputs.
  4. Compare h07 outputs.

Hypothesis test:
  - If h07 outputs match within ~ULP (~1e-12 K): the 0.23 K residual is
    pure ULP amplification across 115 days, and there's no hidden bug
    in the single-timestep physics.
  - If h07 outputs STILL diverge by mK or larger: there's a hidden bug
    in the single timestep at d115 h07 conditions.
"""
from __future__ import annotations
from datetime import date
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


def grab(a, idx=0):
    return float(np.asarray(a).flat[idx])


def inject_f77_state_into_jax(f77, jax_state):
    """Copy F77 state into JAX state for all major canopy/soil/snow fields."""
    veg = f77.get_veg()
    soil = f77.get_soil()
    snow = f77.get_snow()

    # Canopy
    jax_state = eqx.tree_at(
        lambda s: (
            s.canopy.tu, s.canopy.ts, s.canopy.tl,
            s.canopy.t12, s.canopy.t34,
            s.canopy.q12, s.canopy.q34,
            s.canopy.wliqu, s.canopy.wsnou,
            s.canopy.wliqs, s.canopy.wsnos,
            s.canopy.wliql, s.canopy.wsnol,
        ),
        jax_state,
        (
            jnp.asarray(veg["tu"]),  jnp.asarray(veg["ts"]),  jnp.asarray(veg["tl"]),
            jnp.asarray(veg["t12"]), jnp.asarray(veg["t34"]),
            jnp.asarray(veg["q12"]), jnp.asarray(veg["q34"]),
            jnp.asarray(veg["wliqu"]), jnp.asarray(veg["wsnou"]),
            jnp.asarray(veg["wliqs"]), jnp.asarray(veg["wsnos"]),
            jnp.asarray(veg["wliql"]), jnp.asarray(veg["wsnol"]),
        ),
    )

    # Soil
    jax_state = eqx.tree_at(
        lambda s: (
            s.soil.tg, s.soil.tsoi, s.soil.wsoi, s.soil.wisoi,
            s.soil.wpud, s.soil.wipud,
        ),
        jax_state,
        (
            jnp.asarray(soil["tg"]), jnp.asarray(soil["tsoi"]),
            jnp.asarray(soil["wsoi"]), jnp.asarray(soil["wisoi"]),
            jnp.asarray(soil["wpud"]), jnp.asarray(soil["wipud"]),
        ),
    )

    # Snow (tlsub lives in get_veg(), not get_snow())
    jax_state = eqx.tree_at(
        lambda s: (s.snow.fi, s.snow.tsno, s.snow.hsno, s.snow.ti),
        jax_state,
        (
            jnp.asarray(snow["fi"]), jnp.asarray(snow["tsno"]),
            jnp.asarray(snow["hsno"]), jnp.asarray(veg["tlsub"]),
        ),
    )

    # Stomata carry: F77 stores per-cell scalars (active PFT only); JAX stores
    # (npoi, npft) arrays. Inject F77's ciub/gsub/csub into ci_upper[:, 0:2]
    # (broadleaf + conifer slots get same F77 value), similarly for lower PFTs.
    stoma = f77.get_stomata()
    ciub = jnp.asarray(stoma["ciub"])
    gsub = jnp.asarray(stoma["gsub"])
    csub = jnp.asarray(stoma["csub"])
    cils = jnp.asarray(stoma["cils"])
    gsls = jnp.asarray(stoma["gsls"])
    csls = jnp.asarray(stoma["csls"])
    # Broadcast to (npoi, npft) matching JAX layout
    ci_u_2d = jnp.stack([ciub, ciub], axis=-1)
    gs_u_2d = jnp.stack([gsub, gsub], axis=-1)
    cs_u_2d = jnp.stack([csub, csub], axis=-1)
    ci_l_2d = jnp.stack([cils, cils, cils], axis=-1)
    gs_l_2d = jnp.stack([gsls, gsls, gsls], axis=-1)
    cs_l_2d = jnp.stack([csls, csls, csls], axis=-1)
    jax_state = eqx.tree_at(
        lambda s: (
            s.canopy.ci_upper, s.canopy.gs_upper, s.canopy.cs_upper,
            s.canopy.ci_lower, s.canopy.gs_lower, s.canopy.cs_lower,
        ),
        jax_state,
        (ci_u_2d, gs_u_2d, cs_u_2d, ci_l_2d, gs_l_2d, cs_l_2d),
    )

    return jax_state


def diff_report(label, f77, jax_state):
    """Print field-by-field diff between F77 and JAX state."""
    fields = [
        ("canopy.tu",    grab(f77.get_veg()["tu"]),    grab(jax_state.canopy.tu)),
        ("canopy.ts",    grab(f77.get_veg()["ts"]),    grab(jax_state.canopy.ts)),
        ("canopy.tl",    grab(f77.get_veg()["tl"]),    grab(jax_state.canopy.tl)),
        ("canopy.t12",   grab(f77.get_veg()["t12"]),   grab(jax_state.canopy.t12)),
        ("canopy.t34",   grab(f77.get_veg()["t34"]),   grab(jax_state.canopy.t34)),
        ("canopy.q12",   grab(f77.get_veg()["q12"]),   grab(jax_state.canopy.q12)),
        ("canopy.q34",   grab(f77.get_veg()["q34"]),   grab(jax_state.canopy.q34)),
        ("canopy.wliqu", grab(f77.get_veg()["wliqu"]), grab(jax_state.canopy.wliqu)),
        ("canopy.wsnou", grab(f77.get_veg()["wsnou"]), grab(jax_state.canopy.wsnou)),
        ("canopy.wliqs", grab(f77.get_veg()["wliqs"]), grab(jax_state.canopy.wliqs)),
        ("canopy.wsnos", grab(f77.get_veg()["wsnos"]), grab(jax_state.canopy.wsnos)),
        ("canopy.wliql", grab(f77.get_veg()["wliql"]), grab(jax_state.canopy.wliql)),
        ("canopy.wsnol", grab(f77.get_veg()["wsnol"]), grab(jax_state.canopy.wsnol)),
        ("soil.tg",      grab(f77.get_soil()["tg"]),    grab(jax_state.soil.tg)),
        ("soil.tsoi[0]", grab(f77.get_soil()["tsoi"][0]), grab(jax_state.soil.tsoi[0, 0])),
        ("soil.wsoi[0]", grab(f77.get_soil()["wsoi"][0]), grab(jax_state.soil.wsoi[0, 0])),
        ("soil.wisoi[0]",grab(f77.get_soil()["wisoi"][0]),grab(jax_state.soil.wisoi[0, 0])),
        ("snow.fi",      grab(f77.get_snow()["fi"]),    grab(jax_state.snow.fi)),
        ("snow.tsno[0]", grab(f77.get_snow()["tsno"][0]),grab(jax_state.snow.tsno[0, 0])),
        ("snow.hsno[0]", grab(f77.get_snow()["hsno"][0]),grab(jax_state.snow.hsno[0, 0])),
    ]
    print(f"\n=== {label} ===")
    print(f"  {'field':18s} {'F77':>14s} {'JAX':>14s} {'|Δ|':>10s}")
    for name, fv, jv in fields:
        d = abs(fv - jv)
        marker = " <<<" if d > 1e-9 else ""
        print(f"  {name:18s} {fv:>14.6e} {jv:>14.6e} {d:>10.3e}{marker}")


def main():
    target = "L+55p0_+100p0"
    lat, lon = 55.0, 100.0
    YEAR = 1990
    INJECT_DAY = 115
    INJECT_HOUR = 7  # at start of h07, inject; then run h07
    print(f"=== State-injection test on {target} at d{INJECT_DAY} h{INJECT_HOUR} ===")

    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    lats = np.full(NPOI, lat)

    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    print(f"  init complete, NPOI={NPOI}", flush=True)

    step = 0
    for month in [1, 2, 3, 4, 5]:
        gswp3 = read_gswp3_month(YEAR, month, cells)
        if month in (1, 3, 5):
            ndays = 31
        elif month == 2:
            ndays = 28
        else:
            ndays = 30
        for d_in_m in range(ndays):
            jday = (date(YEAR, month, 1) - date(YEAR, 1, 1)).days + d_in_m + 1
            day_data = {k: gswp3[k][d_in_m, :] for k in gswp3}
            day_global = step // 24 + 1
            for h in range(24):
                if h == 0 and day_global % 30 == 0 and day_global < INJECT_DAY:
                    print(f"  seeding: day {day_global}", flush=True)

                if day_global == INJECT_DAY and h == INJECT_HOUR:
                    # === INJECTION POINT ===
                    print(f"\n>>> Reached d{day_global} h{h} (start of h{INJECT_HOUR}) <<<")
                    diff_report("PRE-INJECTION: F77 vs JAX (115 days of lockstep drift)",
                                f77, state)
                    state = inject_f77_state_into_jax(f77, state)
                    diff_report("POST-INJECTION: should be zero", f77, state)

                    # Now run h07 step with identical input state
                    atm = diurnal_disaggregate(day_data, float(h), jday, lats)
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
                    f77.call_lsxmain()
                    state_new = timestep_body(state_atm, consts, params, DTIME)

                    diff_report(
                        f"POST-STEP h{INJECT_HOUR}: F77 vs JAX (identical pre-step state)",
                        f77, state_new,
                    )

                    print("\n=== INTERPRETATION ===")
                    print("If all post-step diffs are <1e-9 K: pure ULP amplification,")
                    print("the 0.23 K Siberian residual at h07 is from 115 days of state drift.")
                    print("If any post-step diff is >1e-6 K: HIDDEN BUG in single timestep at h07.")
                    return

                # Normal lockstep advance
                atm = diurnal_disaggregate(day_data, float(h), jday, lats)
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
                f77.call_lsxmain()
                state = timestep_body(state_atm, consts, params, DTIME)
                step += 1


if __name__ == "__main__":
    main()
