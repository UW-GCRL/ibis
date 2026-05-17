"""Test if FMA suppression eliminates the tropical 70 μK residual.

Three configurations to try (set via env var XLA_FLAGS in shell wrapper):
  A. Default JAX (current behavior)
  B. XLA_FLAGS="--xla_cpu_enable_fast_math=false"
  C. + jax_default_matmul_precision='highest'

Cells tested: 4 cells spanning template 0/1/2.
This script reads the CURRENT env-set XLA_FLAGS and reports it, runs the
4-cell test, prints diffs. Compare outputs across runs with different
XLA_FLAGS settings.
"""
from __future__ import annotations
import os
from datetime import date
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

# Try high-precision matmul (may affect FMA in scalar ops too)
if os.environ.get("USE_HIGHEST_PRECISION"):
    jax.config.update("jax_default_matmul_precision", "highest")
    print(f"[config] jax_default_matmul_precision = highest")

print(f"[env] XLA_FLAGS = {os.environ.get('XLA_FLAGS', '<unset>')}")

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


def parse_cell(label):
    lat_s, lon_s = label.split("_", 1)
    return float(lat_s[1:].replace("p", ".")), float(lon_s.replace("p", "."))


def run_one(target, lat, lon):
    tmpl = soil_template_for_lat(lat)
    cells = [(target, lat, lon, tmpl, tmpl)] * NPOI
    gswp3 = read_gswp3_month(1990, 6, cells)
    lats = np.full(NPOI, lat)
    jday = (date(1990, 6, 1) - date(1990, 1, 1)).days + 1
    day_data = {k: gswp3[k][0, :] for k in gswp3}
    atm_d = diurnal_disaggregate(day_data, 0.0, jday, lats)

    soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
    f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
    f77.set_atm(**atm_d)
    f77.call_lsxmain()
    fv = f77.get_veg(); fs = f77.get_soil(); fx = f77.get_fluxes()

    state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
    state = eqx.tree_at(
        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                   s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                   s.atm.raina, s.atm.snowa),
        state,
        tuple(jnp.asarray(atm_d[k]) for k in
              ("ta","qa","ua","psurf","coszen","solad","solai","fira","raina","snowa")),
    )
    consts = get_constants()
    params = default_sim_params(NPOI)
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))
    new_state = timestep_body(state, consts, params, DTIME)

    def g(x): return float(np.asarray(x).flat[0])
    return {
        "t12": abs(float(fv["t12"][0]) - g(new_state.canopy.t12)),
        "t34": abs(float(fv["t34"][0]) - g(new_state.canopy.t34)),
        "tg":  abs(float(fs["tg"][0])  - g(new_state.soil.tg)),
        "fseng": abs(float(fx["fseng"][0]) - g(new_state.canopy.fseng)),
        "fsenu": abs(float(fx["fsenu"][0]) - g(new_state.canopy.fsenu)),
        "fseni": abs(float(fx["fseni"][0]) - g(new_state.canopy.fseni)),
    }


def main():
    sample = [
        ("L-35p0_+140p0", 2),  # the one that solver fix helped
        ("L-05p0_-080p0", 0),  # tropical
        ("L+55p0_-080p0", 1),  # boreal
        ("L+25p0_+100p0", 2),  # NH summer mid-lat template 2
    ]
    print(f"\n{'cell':>17s} {'tmpl':>4s} | "
          f"{'Δt12':>11s} {'Δt34':>11s} {'Δtg':>11s} | "
          f"{'Δfseng':>11s} {'Δfsenu':>11s} {'Δfseni':>11s}")
    print("-" * 110)
    for cell, tmpl in sample:
        lat, lon = parse_cell(cell)
        d = run_one(cell, lat, lon)
        print(f"{cell:>17s} {tmpl:>4d} | "
              f"{d['t12']:>11.3e} {d['t34']:>11.3e} {d['tg']:>11.3e} | "
              f"{d['fseng']:>11.3e} {d['fsenu']:>11.3e} {d['fseni']:>11.3e}")


if __name__ == "__main__":
    main()
