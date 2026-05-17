"""Single-cell, single-step drill into fseng/fsenu divergence.

All 87 problem cells in the global GSWP3 run had their worst diff in
flux.fseng (76) or flux.fsenu (11) — sensible heat flux. 9 of them
diverged at step=0, meaning the bug fires before any iteration history
can accumulate. The biggest step=0 case is L-35p0_+140p0 (S. Australia,
|Δfseng|=0.138 W/m²).

This script:
  1. Sets up F77 and JAX with the same per-cell soil/veg template
  2. Applies the GSWP3 atm forcing for June 1 1990, hour 0
  3. Runs ONE lockstep timestep
  4. Dumps EVERY accessible state + flux value side-by-side, ranked by |Δ|

Goal: localize whether the fseng spike comes from
  - aerodynamic conductances (sg, su, ss, sl)  →  turcof bug
  - solver outputs (tg, t34)                   →  turvap bug
  - the flux formula itself                    →  diagnostic bug

Usage:
    PYTHONPATH=python-port verify/.venv/bin/python -m \
        verify.timestepping.pipeline.drill_fseng_step0 [cell_label]
"""
from __future__ import annotations
import sys
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


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "L-35p0_+140p0"
    lat_str, lon_str = target.split("_", 1)
    lat = float(lat_str[1:].replace("p", "."))
    lon = float(lon_str.replace("p", "."))
    print(f"=== Single-step drill: {target} (lat={lat}, lon={lon}) ===")

    template = soil_template_for_lat(lat)
    cells = [(target, lat, lon, template, template)] * NPOI

    YEAR, MONTH = 1990, 6
    gswp3 = read_gswp3_month(YEAR, MONTH, cells)

    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    lats = np.full(NPOI, lat)
    jday_offset = (date(YEAR, MONTH, 1) - date(YEAR, 1, 1)).days
    day_idx, hour = 0, 0.0
    jday = jday_offset + day_idx + 1
    day_data = {k: gswp3[k][day_idx, :] for k in gswp3}
    atm = diurnal_disaggregate(day_data, hour, jday, lats)

    print()
    print("Forcing applied:")
    print(f"  ta={atm['ta'][0]:.4f} K  qa={atm['qa'][0]:.6e}  ua={atm['ua'][0]:.4f}")
    print(f"  psurf={atm['psurf'][0]:.2f}  coszen={atm['coszen'][0]:.4f}")
    sol = np.asarray(atm['solad'])
    print(f"  solad.shape={sol.shape}  fira={atm['fira'][0]:.4f}")
    print(f"  solad[0]={sol.flat[0]:.4f}  solad.sum_cell0={sol[..., 0].sum() if sol.ndim==3 else sol[0,:].sum():.4f}")
    print(f"  raina={atm['raina'][0]:.4e}  snowa={atm['snowa'][0]:.4e}")

    # Apply same forcing to both
    f77.set_atm(**atm)
    state_atm = eqx.tree_at(
        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                   s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                   s.atm.raina, s.atm.snowa),
        state,
        (jnp.asarray(atm["ta"]), jnp.asarray(atm["qa"]),
         jnp.asarray(atm["ua"]), jnp.asarray(atm["psurf"]),
         jnp.asarray(atm["coszen"]),
         jnp.asarray(atm["solad"]), jnp.asarray(atm["solai"]),
         jnp.asarray(atm["fira"]),
         jnp.asarray(atm["raina"]), jnp.asarray(atm["snowa"])),
    )

    # Run one step
    f77.call_lsxmain()
    new_state = timestep_body(state_atm, consts, params, DTIME)

    # Collect every comparable field
    f77_aero = f77.get_aerodynamic()
    f77_veg = f77.get_veg()
    f77_soil = f77.get_soil()
    f77_snow = f77.get_snow()
    f77_flux = f77.get_fluxes()
    f77_stom = f77.get_stomata()

    def grab(arr, idx=0):
        return float(np.asarray(arr).flat[idx])

    cn = new_state.canopy  # JAX stores fluxes here

    rows = []
    # Canopy state (only fields present in both)
    for k in ("tu","tl","ts","t12","t34","q12","q34"):
        fv = grab(f77_veg[k])
        jv = grab(getattr(new_state.canopy, k))
        rows.append((f"canopy.{k}", fv, jv))

    # Soil/snow heat surfaces
    rows.append(("soil.tg", grab(f77_soil["tg"]), grab(new_state.soil.tg)))
    rows.append(("snow.tsno[0]", grab(f77_snow["tsno"][:, 0]),
                 grab(new_state.snow.tsno[:, 0])))
    rows.append(("snow.hsno[0]", grab(f77_snow["hsno"][:, 0]),
                 grab(new_state.snow.hsno[:, 0])))
    rows.append(("soil.tsoi[0]", grab(f77_soil["tsoi"][:, 0]),
                 grab(new_state.soil.tsoi[:, 0])))
    rows.append(("soil.wsoi[0]", grab(f77_soil["wsoi"][:, 0]),
                 grab(new_state.soil.wsoi[:, 0])))
    rows.append(("snow.fi", grab(f77_snow["fi"]), grab(new_state.snow.fi)))

    # Stomata stress
    rows.append(("stresstu", grab(f77_stom["stresstu"]),
                 grab(new_state.soil.stresstu)))
    rows.append(("stresstl", grab(f77_stom["stresstl"]),
                 grab(new_state.soil.stresstl)))

    # Fluxes — sensible heat suspects + neighbors
    for k in ("fsenu","fsens","fsenl","fseng","fseni","fvapg","fvapi"):
        fv = grab(f77_flux[k])
        jv = grab(getattr(cn, k))
        rows.append((f"flux.{k}", fv, jv))

    print()
    print(f"{'field':>16s} {'F77':>15s} {'JAX':>15s} {'|Δ|':>13s} {'rel':>10s}")
    print("-" * 75)
    rows_with_diff = []
    for name, fv, jv in rows:
        d = abs(fv - jv)
        rel = d / max(abs(fv), 1e-30)
        rows_with_diff.append((name, fv, jv, d, rel))
    rows_with_diff.sort(key=lambda r: -r[3])
    for name, fv, jv, d, rel in rows_with_diff:
        marker = " ⚠" if d > 1e-6 and not name.startswith("flux.") else ""
        marker += " 🔥" if d > 1e-3 else ""
        print(f"{name:>16s} {fv:>15.6e} {jv:>15.6e} {d:>13.3e} {rel:>10.2e}{marker}")

    print()
    # JAX doesn't persist aero state; only get sg from F77 side. Print
    # the key fseng inputs we can see.
    print("fseng formula suspects:")
    print(f"  aero.sg (F77 only): {grab(f77_aero['sg']):.6e}")
    print(f"  canopy.t34: F77={grab(f77_veg['t34']):.6e}  JAX={grab(new_state.canopy.t34):.6e}  Δ={abs(grab(f77_veg['t34']) - grab(new_state.canopy.t34)):.3e}")
    print(f"  soil.tg:    F77={grab(f77_soil['tg']):.6e}  JAX={grab(new_state.soil.tg):.6e}  Δ={abs(grab(f77_soil['tg']) - grab(new_state.soil.tg)):.3e}")
    print(f"  flux.fseng: F77={grab(f77_flux['fseng']):.6e}  JAX={grab(cn.fseng):.6e}  Δ={abs(grab(f77_flux['fseng']) - grab(cn.fseng)):.3e}")


if __name__ == "__main__":
    main()
