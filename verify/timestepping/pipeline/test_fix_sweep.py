"""Sweep N problem cells with stock vs patched solver to see fix impact."""
from __future__ import annotations
from datetime import date
import csv
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


def f77_linsolve(arr, rhs):
    """F77-style no-pivot Gaussian elimination matching math.f:73."""
    npoi, nd, _ = arr.shape
    A = arr; b = rhs
    for id_ in range(nd - 1):
        diag = A[:, id_, id_]
        for i in range(id_ + 1, nd):
            f = A[:, i, id_] / diag
            A = A.at[:, i, :].set(A[:, i, :] - f[:, None] * A[:, id_, :])
            b = b.at[:, i].set(b[:, i] - f * b[:, id_])
    vec = jnp.zeros_like(b)
    for id_ in range(nd - 1, -1, -1):
        s = jnp.zeros(npoi)
        for j in range(id_ + 1, nd):
            s = s + A[:, id_, j] * vec[:, j]
        vec = vec.at[:, id_].set((b[:, id_] - s) / A[:, id_, id_])
    return vec


def _patched_solve(arr, rhs_2d):
    return f77_linsolve(arr, rhs_2d.squeeze(-1))[..., None]


def parse_cell(label):
    lat_s, lon_s = label.split("_", 1)
    return float(lat_s[1:].replace("p", ".")), float(lon_s.replace("p", "."))


def run_one(target, lat, lon, use_patch):
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
    f77_vals = dict(
        t12=float(fv["t12"][0]), t34=float(fv["t34"][0]),
        tg=float(fs["tg"][0]),
        fseng=float(fx["fseng"][0]), fsenu=float(fx["fsenu"][0]),
        fsens=float(fx["fsens"][0]), fseni=float(fx["fseni"][0]),
    )

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

    if use_patch:
        import jax.numpy.linalg as jlinalg
        saved = jlinalg.solve
        jlinalg.solve = _patched_solve
        try:
            new_state = timestep_body(state, consts, params, DTIME)
        finally:
            jlinalg.solve = saved
    else:
        new_state = timestep_body(state, consts, params, DTIME)

    def g(x): return float(np.asarray(x).flat[0])
    jax_vals = dict(
        t12=g(new_state.canopy.t12), t34=g(new_state.canopy.t34),
        tg=g(new_state.soil.tg),
        fseng=g(new_state.canopy.fseng), fsenu=g(new_state.canopy.fsenu),
        fsens=g(new_state.canopy.fsens), fseni=g(new_state.canopy.fseni),
    )
    diffs = {k: abs(f77_vals[k] - jax_vals[k]) for k in f77_vals}
    return diffs


def main():
    # Diverse sample: 2 freeze-thaw, 2 tropical, 2 boreal, 2 arctic, 2 mid-lat
    sample = [
        "L-35p0_+140p0",  # template 2 — freeze-thaw (worst step=0)
        "L+25p0_+100p0",  # template 2 — China
        "L-05p0_-080p0",  # template 0 — Amazon
        "L-05p0_-060p0",  # template 0 — Brazil
        "L+55p0_-080p0",  # template 1 — Hudson Bay area
        "L+65p0_-100p0",  # template 1 — arctic
        "L+45p0_+120p0",  # template 2 — Mongolia
        "L+25p0_+020p0",  # template 2 — Sahara north
        "L-75p0_-020p0",  # template 1 — antarctic
        "L+05p0_+020p0",  # template 0 — central Africa
    ]
    print(f"{'cell':>17s} {'tmpl':>4s} | "
          f"{'stock Δfseng':>14s} {'patch Δfseng':>14s} {'ratio':>6s} | "
          f"{'stock Δt12':>11s} {'patch Δt12':>11s} {'ratio':>6s}")
    print("-" * 110)
    for cell in sample:
        lat, lon = parse_cell(cell)
        tmpl = soil_template_for_lat(lat)
        d_stock = run_one(cell, lat, lon, use_patch=False)
        d_patch = run_one(cell, lat, lon, use_patch=True)
        r_fseng = d_stock["fseng"] / max(d_patch["fseng"], 1e-30)
        r_t12   = d_stock["t12"] / max(d_patch["t12"], 1e-30)
        print(f"{cell:>17s} {tmpl:>4d} | "
              f"{d_stock['fseng']:>14.3e} {d_patch['fseng']:>14.3e} {r_fseng:>6.1f}x | "
              f"{d_stock['t12']:>11.3e} {d_patch['t12']:>11.3e} {r_t12:>6.1f}x")


if __name__ == "__main__":
    main()
