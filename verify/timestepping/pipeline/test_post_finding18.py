"""Sweep all 87 original problem cells, report step-0 ΔT and Δfseng post-finding-#18.

Efficient: reads GSWP3 ONCE for all cells, then runs in batches of NPOI=5.
Counts: bit-exact (≤1e-9 K), near-exact (≤1e-6 K), residual (>1e-6 K).
"""
from __future__ import annotations
import csv
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


def parse_cell(label):
    a, b = label.split("_", 1)
    return float(a[1:].replace("p", ".")), float(b.replace("p", "."))


def main():
    # Load problem cells
    with open("/shared/nas2/Hangkai/ibis/verify_outputs/problem_cells.csv") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows)} problem cells. Reading GSWP3 once...")

    cells_all = [(r["cell_label"], *parse_cell(r["cell_label"]),
                  soil_template_for_lat(parse_cell(r["cell_label"])[0]),
                  soil_template_for_lat(parse_cell(r["cell_label"])[0]))
                 for r in rows]
    gswp3_all = read_gswp3_month(1990, 6, cells_all)
    print(f"Read complete. Running {len(rows)} cells in batches of {NPOI}...")
    print()
    print(f"{'cell':<20s} {'ΔT_max (K)':>13s} {'Δfseng':>13s} {'Δfseni':>13s}")
    print("-" * 65)

    consts = get_constants()

    bit_exact, near_exact, residual = 0, 0, 0
    worst_residual = ("", 0.0)
    rows_residual = []

    # Process one cell at a time but reuse GSWP3 data
    for idx, r in enumerate(rows):
        target = r["cell_label"]
        lat, lon = parse_cell(target)
        tmpl = soil_template_for_lat(lat)
        cells = [(target, lat, lon, tmpl, tmpl)] * NPOI

        # Use this cell's slice of GSWP3
        day_data = {k: np.full(NPOI, gswp3_all[k][0, idx]) for k in gswp3_all}
        lats = np.full(NPOI, lat)
        jday = (date(1990, 6, 1) - date(1990, 1, 1)).days + 1
        atm = diurnal_disaggregate(day_data, 0.0, jday, lats)

        soil_i, snow_i, atm_i, veg_i = make_batch_state(cells)
        f77 = setup_f77_state(soil_i, snow_i, atm_i, veg_i)
        state = setup_jax_state(soil_i, snow_i, atm_i, veg_i)
        params = default_sim_params(NPOI)
        params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), params, (4.5, 0.5))

        f77.set_atm(**atm)
        state = eqx.tree_at(
            lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                       s.atm.coszen, s.atm.solad, s.atm.solai, s.atm.fira,
                       s.atm.raina, s.atm.snowa),
            state,
            tuple(jnp.asarray(atm[k]) for k in
                  ("ta","qa","ua","psurf","coszen","solad","solai","fira","raina","snowa")),
        )
        f77.call_lsxmain()
        new_state = timestep_body(state, consts, params, DTIME)

        fv = f77.get_veg(); fs = f77.get_soil(); fc = f77.get_fluxes()
        def g(x): return float(np.asarray(x).flat[0])
        dT_max = max(
            abs(g(fv["t12"]) - g(new_state.canopy.t12)),
            abs(g(fv["t34"]) - g(new_state.canopy.t34)),
            abs(g(fs["tg"])  - g(new_state.soil.tg)),
            abs(g(fv["tu"])  - g(new_state.canopy.tu)),
        )
        df_fseng = abs(g(fc["fseng"]) - g(new_state.canopy.fseng))
        df_fseni = abs(g(fc["fseni"]) - g(new_state.canopy.fseni))

        marker = ""
        if dT_max < 1e-9:
            bit_exact += 1; marker = " ✅"
        elif dT_max < 1e-6:
            near_exact += 1; marker = " ✓"
        else:
            residual += 1; marker = " 🔥"
            rows_residual.append((target, dT_max, df_fseng, df_fseni))
            if dT_max > worst_residual[1]:
                worst_residual = (target, dT_max)
        print(f"{target:<20s} {dT_max:>13.3e} {df_fseng:>13.3e} {df_fseni:>13.3e}{marker}")

    print()
    print(f"Summary across {len(rows)} originally-problematic cells:")
    print(f"  Bit-exact (ΔT < 1e-9 K):    {bit_exact:>3d}")
    print(f"  Near-exact (ΔT < 1e-6 K):   {near_exact:>3d}")
    print(f"  Residual (ΔT > 1e-6 K):     {residual:>3d}")
    if residual > 0:
        print(f"  Worst residual: {worst_residual[0]} ΔT={worst_residual[1]:.3e} K")
        print()
        print(f"  Residual cells:")
        for c, dT, dg, di in sorted(rows_residual, key=lambda x: -x[1]):
            print(f"    {c:<20s} ΔT={dT:.3e}  Δfseng={dg:.3e}  Δfseni={di:.3e}")


if __name__ == "__main__":
    main()
