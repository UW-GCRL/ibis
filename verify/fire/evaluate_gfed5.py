"""Fire evaluation against GFED5 burned area, and optional calibration.

Reports the skill metrics quoted in docs/FIRE.md: area-weighted burned
fraction, Spearman rank correlation overall and among burning cells, NMSE
(the Druke et al. 2019 metric), and the mean absolute latitude of the most
fire-prone cells -- the diagnostic that exposed the 29-degree displacement the
VPD danger term fixed.

    python verify/fire/evaluate_gfed5.py
    python verify/fire/evaluate_gfed5.py --calibrate --steps 800

Existed only in /tmp until 2026-08-28 (docs/AUDIT.md group C): every fire number
in FIRE.md came from scripts nobody else could run.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-port"))

from verify.timestepping.pipeline.global_gswp3_test import (  # noqa: E402
    build_global_cell_grid,
)

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from ibis.io.gfed5 import read_gfed5_annual_fraction  # noqa: E402
from ibis.io.popdens import read_popdens_year  # noqa: E402

DEFAULT_STATE = ("/shared/nas2/Hangkai/ibis/verify_outputs/spinup_fire_v8/"
                 "checkpoints/state_latest.pkl")
DEFAULT_GFED5 = "/shared/nas2/Hangkai/ibis/inputs/fire/gfed5"
DEFAULT_POPD = ("/shared/nas2/Hangkai/ibis/inputs/fire/popdens/"
                "clmforc.Li_2018_SSP2_CMIP6_hdm_0.5x0.5_AVHRR_"
                "simyr1850-2100_c181205.nc")


def _grid(state_path=""):
    """Grid matching the state's own ordering.

    Two orderings coexist: verify/'s build_global_cell_grid walks latitude
    SOUTH->NORTH, while run_global enumerates land with argwhere on a
    NORTH->SOUTH mask. Indexing a state with the wrong one silently permutes
    every cell, so the format decides.
    """
    mlat = np.arange(-90.0, 90.0, 0.5)
    mlon = np.arange(-180.0, 180.0, 0.5)
    if str(state_path).endswith(".npz"):
        from scripts.run_global import (_load_land_mask_from_file,
                                        _subset_land_mask)
        mask, glat, glon = _load_land_mask_from_file(
            "/shared/nas2/Min/HWSD/surta.nc")
        _, _, _, _, gli, glo = _subset_land_mask(mask, glat, glon,
                                                 90.0, -90.0, -180.0, 180.0)
        lat = glat[gli]; lon = glon[glo]
    else:
        cells = build_global_cell_grid(lat_step=0.5, lon_step=0.5)
        lat = np.array([c[1] for c in cells])
        lon = np.array([c[2] for c in cells])
    li = np.rint((lat - mlat[0]) / 0.5).astype(int).clip(0, 359)
    lo = np.rint((lon - mlon[0]) / 0.5).astype(int).clip(0, 719)
    return lat, lon, li, lo, mlat, mlon


def _load(state_path):
    """Load either a verify/ pickle checkpoint or a run_global .npz restart."""
    if str(state_path).endswith(".npz"):
        from ibis.io.restart import load_state
        from ibis.state.model_state import create_ibis_state
        from ibis.dimensions import GridConfig
        from scripts.run_global import (_load_land_mask_from_file,
                                        _subset_land_mask)
        mask, glat, glon = _load_land_mask_from_file(
            "/shared/nas2/Min/HWSD/surta.nc")
        _, _, _, _, gli, _ = _subset_land_mask(mask, glat, glon,
                                               90.0, -90.0, -180.0, 180.0)
        grid = GridConfig(nlon=720, nlat=360, npoi=len(gli), xres=0.5,
                          yres=0.5, nbedrock=3)
        st, _meta = load_state(state_path, create_ibis_state(grid))
        return st
    st = pickle.load(open(state_path, "rb"))
    return st["jax_state"] if isinstance(st, dict) else st


def observed(gfed5_dir, li, lo, mlat, mlon, years, component="Norm"):
    return np.mean([read_gfed5_annual_fraction(gfed5_dir, y, li, lo, mlat,
                                               mlon, component=component)
                    for y in years], axis=0)


def report(pred, obs, lat, label="model"):
    """The metric set quoted in docs/FIRE.md."""
    aw = np.cos(np.deg2rad(lat))
    # v11 states carry one fully-NaN cell (68.75N -26.75E, Greenland margin;
    # docs/CALIBRATION.md 8). Unmasked, it turns the area-weighted burned
    # fraction and the all-cell rank into NaN. Drop non-finite cells first.
    ok = np.isfinite(pred) & np.isfinite(obs) & np.isfinite(aw)
    if not ok.all():
        print(f"    ({int((~ok).sum())} non-finite cell(s) dropped)")
        pred, obs, aw, lat = pred[ok], obs[ok], aw[ok], lat[ok]
    aw = aw / aw.sum()
    burning = obs > 0
    nmse = ((pred - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    top = np.argsort(-pred)[:2000]
    tgt = np.argsort(-obs)[:2000]
    print(f"  {label}")
    print(f"    area-weighted burned fraction  {float((pred * aw).sum()) * 100:6.3f}%"
          f"   GFED5 {float((obs * aw).sum()) * 100:6.3f}%"
          f"   x{(pred * aw).sum() / (obs * aw).sum():.2f}")
    print(f"    rank correlation               all {spearmanr(pred, obs).statistic:+.3f}"
          f"   burning {spearmanr(pred[burning], obs[burning]).statistic:+.3f}")
    print(f"    NMSE (lower better)            {nmse:.3f}")
    print(f"    mean |lat| of top-2000 cells   model {np.abs(lat[top]).mean():.1f}"
          f"   GFED5 {np.abs(lat[tgt]).mean():.1f}")
    return dict(nmse=nmse, rank=spearmanr(pred, obs).statistic)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--gfed5", default=DEFAULT_GFED5)
    ap.add_argument("--popdens", default=DEFAULT_POPD)
    ap.add_argument("--years", type=int, nargs=2, default=[2001, 2005])
    ap.add_argument("--calibrate", action="store_true",
                    help="fit k_b, p and lai_ref_vd by gradient descent")
    ap.add_argument("--steps", type=int, default=800)
    a = ap.parse_args()

    lat, lon, li, lo, mlat, mlon = _grid(a.state)
    st = _load(a.state)
    print(f"  state: {len(lat)} pts, "
          f"{'N->S argwhere' if str(a.state).endswith('.npz') else 'S->N cell-grid'} ordering")
    obs = observed(a.gfed5, li, lo, mlat, mlon,
                   range(a.years[0], a.years[1] + 1))
    pred = np.asarray(st.vegetation.disturbf)

    print(f"\n  GFED5 Norm {a.years[0]}-{a.years[1]}, {len(obs)} cells, "
          f"{int((obs > 0).sum())} burning\n")
    report(pred, obs, lat, "spun-up state (disturbf)")

    if a.calibrate:
        from ibis.ml.fire_calibration import (FireBatch, fit_fire_params,
                                              predict_burned_fraction)
        from ibis.physics.fire import FireParams
        popd = read_popdens_year(a.popdens, 2003, li, lo, mlat, mlon)
        v, c = st.vegetation, st.carbon
        batch = FireBatch(
            nfirestep=jnp.asarray(v.nfirestep_vpd), sdry=jnp.asarray(v.sdry_vpd),
            nytimes=jnp.asarray(float(v.nytimes)), totlit=jnp.asarray(c.totlit),
            totalit=jnp.asarray(c.totalit), cbiol=jnp.asarray(v.cbiol),
            plai=jnp.asarray(v.plai), popd=jnp.asarray(popd))
        p0 = FireParams(use_vpd_danger=True, use_fine_fuel=False)
        print(f"\n  calibrating k_b, p, lai_ref_vd ({a.steps} steps)...")
        res = fit_fire_params(p0, batch, jnp.asarray(obs), steps=a.steps,
                              learning_rate=3e-2,
                              fit_only=("k_b", "p", "lai_ref_vd"),
                              total_weight=3.0)
        print(f"    loss {res.loss_history[0]:.4f} -> {res.loss_history[-1]:.4f}")
        for k in ("k_b", "p", "lai_ref_vd"):
            print(f"      {k:11} = {float(getattr(res.params, k)):.4f}")
        report(np.asarray(predict_burned_fraction(res.params, batch)), obs, lat,
               "\n  after calibration")
        print("\n  NOTE: docs/FIRE.md records that the a-priori parameters beat "
              "the fitted\n  ones on global rank and total burned area. "
              "Calibration is not automatically\n  an improvement here -- "
              "compare before adopting.")


if __name__ == "__main__":
    main()
