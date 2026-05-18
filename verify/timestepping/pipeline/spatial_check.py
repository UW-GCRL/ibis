"""Spatial-biome reasonableness check: one timestep on all 117 global cells.

For each cell, runs F77 + JAX single timestep with June 1 1990 hour 0
GSWP3 forcing, then verifies BIOME-LEVEL patterns make physical sense:
  - Tropical regions should be warmer than polar
  - Tropical regions should have lower snow cover
  - Tropical (high LAI, hot) should have higher photosynthesis than boreal
  - Polar SH (austral winter) should be coldest in June
  - Tropics never have snow
"""
from __future__ import annotations
from datetime import date
import collections
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

from verify.timestepping.pipeline.global_gswp3_test import (
    diurnal_disaggregate, read_gswp3_month, make_batch_state,
    soil_template_for_lat, build_global_cell_grid,
)
from verify.timestepping.verify_lsxmain import (
    setup_f77_state, setup_jax_state, DTIME,
)
from ibis.constants import get_constants
from ibis.timestepping.loops import default_sim_params, timestep_body
from tests.f2py.ibis_f77_wrapper import NPOI


def biome(lat):
    """Lat-based biome classification (June = NH summer, SH winter)."""
    if lat > 60:  return "boreal_NH"     # NH summer growing
    if lat > 30:  return "midlat_NH"     # NH summer hot
    if lat > -30: return "tropical"      # Always warm
    if lat > -60: return "midlat_SH"     # SH winter cold
    return "polar_SH"                    # Antarctica winter (no sun)


def main():
    CELLS = build_global_cell_grid(lat_step=10.0, lon_step=20.0)
    print(f"Biome-pattern check: {len(CELLS)} cells, single-step, June 1 1990 12:00 UTC")

    YEAR, MONTH = 1990, 6
    consts = get_constants()
    n_batches = (len(CELLS) + NPOI - 1) // NPOI

    biome_data = collections.defaultdict(lambda: collections.defaultdict(list))
    nan_count = 0

    for b in range(n_batches):
        batch_cells = CELLS[b*NPOI:(b+1)*NPOI]
        original_n = len(batch_cells)
        while len(batch_cells) < NPOI:
            batch_cells.append(batch_cells[-1])

        gswp3 = read_gswp3_month(YEAR, MONTH, batch_cells)
        lats = np.array([c[1] for c in batch_cells])
        day_data = {k: gswp3[k][0, :] for k in gswp3}
        # Hour 12 UTC — gives non-zero coszen for tropical/SH-summer cells
        atm = diurnal_disaggregate(day_data, 12.0, 152, lats)

        soil_i, snow_i, atm_i, veg_i = make_batch_state(batch_cells)
        f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
        state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
        params = default_sim_params(NPOI)
        params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

        f77.set_atm(**atm)
        state = eqx.tree_at(
            lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                       s.atm.coszen, s.atm.solad, s.atm.solai,
                       s.atm.fira, s.atm.raina, s.atm.snowa),
            state,
            tuple(jnp.asarray(atm[k]) for k in
                  ("ta","qa","ua","psurf","coszen","solad","solai",
                   "fira","raina","snowa")),
        )
        f77.call_lsxmain()
        new_state = timestep_body(state, consts, params, DTIME)

        fs = f77.get_soil(); fv = f77.get_veg()
        fn = f77.get_snow(); ff = f77.get_fluxes()
        fst = f77.get_stomata()

        for ci in range(original_n):
            cell = batch_cells[ci]
            lat = cell[1]
            b_label = biome(lat)
            # Extract key fields (F77 values for biome-pattern check)
            vals = {
                "ta_atm":        float(atm["ta"][ci]),
                "coszen":        float(atm["coszen"][ci]),
                "soil.tg":       float(fs["tg"][ci]),
                "snow.fi":       float(fn["fi"][ci]),
                "canopy.tu":     float(fv["tu"][ci]),
                "canopy.t12":    float(fv["t12"][ci]),
                "stresstu":      float(fst["stresstu"][ci]),
                "totcondub":     float(fst["totcondub"][ci]),     # upper canopy conductance ~ photosynth
                "flux.fseng":    float(ff["fseng"][ci]),
                "flux.fvapa":    float(ff["fvapa"][ci]),
            }
            # JAX comparison
            jvals = {
                "ta_atm":        float(atm["ta"][ci]),
                "soil.tg":       float(new_state.soil.tg[ci]),
                "snow.fi":       float(new_state.snow.fi[ci]),
                "canopy.tu":     float(new_state.canopy.tu[ci]),
                "canopy.t12":    float(new_state.canopy.t12[ci]),
                "stresstu":      float(new_state.soil.stresstu[ci]),
                "flux.fseng":    float(new_state.canopy.fseng[ci]),
            }

            for k, v in vals.items():
                if np.isnan(v): nan_count += 1
                biome_data[b_label][k].append(v)
            for k, v in jvals.items():
                key = f"jax_{k}"
                biome_data[b_label][key].append(v)

    print(f"\nNaN count: {nan_count}\n")
    print(f"=== Biome-level F77 outputs (June 1 1990 12 UTC, one step) ===")
    print(f"{'biome':>12s} {'N':>3s}  {'ta_atm':>8s} {'soil.tg':>8s} {'canopy.tu':>10s} "
          f"{'snow.fi':>8s} {'stresstu':>10s} {'totcondub':>11s} {'fseng':>8s}")
    print("-" * 100)
    biome_order = ("polar_SH", "midlat_SH", "tropical", "midlat_NH", "boreal_NH")
    for b in biome_order:
        d = biome_data[b]
        if not d: continue
        n = len(d["ta_atm"])
        def mm(k):
            v = d[k]
            return f"{np.mean(v):>7.2f}"
        print(f"{b:>12s} {n:>3d}  "
              f"{mm('ta_atm'):>8s} {mm('soil.tg'):>8s} {mm('canopy.tu'):>10s} "
              f"{np.mean(d['snow.fi']):>8.3f} {np.mean(d['stresstu']):>10.3f} "
              f"{np.mean(d['totcondub']):>11.3e} {np.mean(d['flux.fseng']):>8.2f}")

    # ----- Biome-pattern assertions -----
    print(f"\n=== Biome-pattern sanity (F77 mean values) ===")
    means = {b: {k: np.mean(biome_data[b][k]) for k in biome_data[b]}
             for b in biome_order if biome_data[b]}

    checks = []
    if "tropical" in means and "polar_SH" in means:
        checks.append(("Tropical canopy warmer than polar SH (Antarctic winter)",
                       means['tropical']['canopy.tu'] > means['polar_SH']['canopy.tu']))
        checks.append(("Polar SH has more snow than tropical",
                       means['polar_SH']['snow.fi'] > means['tropical']['snow.fi']))
        checks.append(("Tropical canopy conductance > boreal (photosynth proxy)",
                       means['tropical']['totcondub'] > means.get('boreal_NH', {}).get('totcondub', 0)))
        checks.append(("Tropical soil tg > polar tg",
                       means['tropical']['soil.tg'] > means['polar_SH']['soil.tg']))
    if "midlat_NH" in means and "polar_SH" in means:
        checks.append(("NH midlat (summer) warmer than SH polar (winter)",
                       means['midlat_NH']['canopy.tu'] > means['polar_SH']['canopy.tu']))
    if "tropical" in means:
        checks.append(("Tropical snow fraction near zero",
                       means['tropical']['snow.fi'] < 0.1))

    all_pass = True
    for desc, ok in checks:
        mark = "✅" if ok else "❌"
        print(f"  {mark} {desc}")
        if not ok: all_pass = False

    print(f"\n{'✅ All biome patterns physical' if all_pass else '❌ Some biome patterns failed'}")

    # ----- F77 vs JAX spatial alignment (max |Δ| per biome) -----
    print(f"\n=== F77 vs JAX max |Δ| per biome (single timestep) ===")
    print(f"{'biome':>12s}  {'Δsoil.tg':>11s} {'Δcanopy.tu':>12s} {'Δsnow.fi':>11s} "
          f"{'Δstresstu':>11s} {'Δfseng':>11s}")
    print("-" * 80)
    for b in biome_order:
        d = biome_data[b]
        if not d: continue
        def maxd(f77k, jaxk):
            f = d[f77k]; j = d[jaxk]
            return max(abs(a-b) for a,b in zip(f,j))
        print(f"{b:>12s}  "
              f"{maxd('soil.tg','jax_soil.tg'):>11.3e} "
              f"{maxd('canopy.tu','jax_canopy.tu'):>12.3e} "
              f"{maxd('snow.fi','jax_snow.fi'):>11.3e} "
              f"{maxd('stresstu','jax_stresstu'):>11.3e} "
              f"{maxd('flux.fseng','jax_flux.fseng'):>11.3e}")


if __name__ == "__main__":
    main()
