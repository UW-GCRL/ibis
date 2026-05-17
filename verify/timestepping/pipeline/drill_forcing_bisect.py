"""Bisect which GSWP3 forcing variable triggers the 5 mK step-0 divergence.

Same target cell + soil template as drill_fseng_step0, but vary the atm
forcing:
  - run_synthetic:    use 5-cell synthetic atm (template 2) — expect 8.5e-9
  - run_gswp3:        use real June-1 hour-0 GSWP3 — expect 5e-3
  - run_bisect_X:     start from synthetic, replace X with GSWP3 — find culprit
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
from tests.f2py.synthetic_inputs import make_atm_inputs


def run_one_step(atm_dict, cells):
    """Run one F77 + JAX step with given forcing, return (ΔT_max, ΔW_max, diffs)."""
    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

    f77.set_atm(**atm_dict)
    state_atm = eqx.tree_at(
        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                   s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                   s.atm.raina, s.atm.snowa),
        state,
        tuple(jnp.asarray(atm_dict[k]) for k in
              ("ta","qa","ua","psurf","coszen","solad","solai","fira","raina","snowa")),
    )
    f77.call_lsxmain()
    new_state = timestep_body(state_atm, consts, params, DTIME)

    fv = f77.get_veg()
    fs = f77.get_soil()
    fc = f77.get_fluxes()
    def grab(arr): return float(np.asarray(arr).flat[0])

    diffs = {}
    for k in ("tu","tl","ts","t12","t34"):
        diffs[f"can.{k}"] = abs(grab(fv[k]) - grab(getattr(new_state.canopy, k)))
    diffs["soil.tg"] = abs(grab(fs["tg"]) - grab(new_state.soil.tg))
    diffs["soil.wsoi0"] = abs(grab(fs["wsoi"][:, 0]) - grab(new_state.soil.wsoi[:, 0]))
    for k in ("fseng","fsenu","fsens","fsenl","fseni"):
        diffs[f"flx.{k}"] = abs(grab(fc[k]) - grab(getattr(new_state.canopy, k)))

    return diffs


def main():
    target = "L-35p0_+140p0"
    lat, lon = -35.0, 140.0
    template = soil_template_for_lat(lat)  # = 2 (temperate)
    cells = [(target, lat, lon, template, template)] * NPOI

    # Synthetic atm (5-cell, template-2 = idx 2)
    syn = make_atm_inputs()  # has shape (5,...) — replicate idx 2 across NPOI
    syn_atm = {}
    for k, v in syn.items():
        v = np.asarray(v)
        if v.ndim == 1 and v.shape[0] == NPOI:
            syn_atm[k] = np.full(NPOI, v[template])
        elif v.ndim == 2 and v.shape[0] == NPOI:
            # solad/solai (NPOI, NBAND)
            syn_atm[k] = np.tile(v[template:template+1, :], (NPOI, 1))
        else:
            syn_atm[k] = v.copy()

    # GSWP3 atm
    YEAR, MONTH = 1990, 6
    gswp3 = read_gswp3_month(YEAR, MONTH, cells)
    lats = np.full(NPOI, lat)
    jday = (date(YEAR, MONTH, 1) - date(YEAR, 1, 1)).days + 1
    day_data = {k: gswp3[k][0, :] for k in gswp3}
    gs_atm = diurnal_disaggregate(day_data, 0.0, jday, lats)

    def report(name, atm):
        d = run_one_step(atm, cells)
        max_t = max(v for k, v in d.items() if k.startswith("can.") or k=="soil.tg")
        max_f = max(v for k, v in d.items() if k.startswith("flx."))
        worst_state_field = max((k for k in d if k.startswith("can.") or k=="soil.tg"),
                                key=lambda k: d[k])
        print(f"  {name:>28s}  ΔT_max={max_t:.3e}K  Δflux_max={max_f:.3e}W/m²  worst={worst_state_field}")
        return d

    print("\n=== Baseline: synthetic vs GSWP3 forcing on same cell ===\n")
    print(f"  Cell: {target}  template={template}  one-step ΔT comparison")
    print(f"  {'config':>28s}  {'ΔT_max':>10s}  {'Δflux_max':>13s}\n")

    print("  Synthetic atm (template-2 idx values):")
    print(f"    ta={syn_atm['ta'][0]:.2f}  qa={syn_atm['qa'][0]:.4e}  raina={syn_atm['raina'][0]:.2e}  coszen={syn_atm['coszen'][0]:.3f}")
    d_syn = report("synthetic", syn_atm)

    print("\n  GSWP3 atm (June-1 hour-0):")
    print(f"    ta={gs_atm['ta'][0]:.2f}  qa={gs_atm['qa'][0]:.4e}  raina={gs_atm['raina'][0]:.2e}  coszen={gs_atm['coszen'][0]:.3f}")
    d_gs  = report("gswp3", gs_atm)

    print()
    print("=== Bisect: swap one GSWP3 field at a time into synthetic ===\n")
    for key in ("ta","qa","ua","psurf","coszen","fira","raina","snowa","solad","solai"):
        mixed = {k: v.copy() if hasattr(v, 'copy') else v for k, v in syn_atm.items()}
        mixed[key] = gs_atm[key].copy() if hasattr(gs_atm[key], 'copy') else gs_atm[key]
        report(f"syn + gswp3.{key}", mixed)


if __name__ == "__main__":
    main()
