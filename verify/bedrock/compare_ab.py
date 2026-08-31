"""Compare the bedrock A/B spin-ups: does the 12 m column do what it predicted?

Both arms start from the same `spinup_fire_v8` state, run the same 200 years of
GSWP3 with the same fire configuration, and receive the same `rhosoi`
migration. The ONLY difference is the soil column: 6 layers to 4 m (control)
versus 6 soil + 3 thermal-only layers to 12 m (treatment).

The predictions in docs/BEDROCK.md, restated as falsifiable claims:

  1. deep soil temperature seasonal amplitude FALLS -- the change is for this;
  2. NH permafrost extent changes LITTLE (v8: 23.2 Mkm2 vs Obu's 21.8);
  3. active layer depth changes LITTLE (v8: 1.14 m; measured bias there +2%);
  4. global methane changes LITTLE -- the bias was at 3.0 m, below the active
     layer.

If (4) fails, the reasoning that the error was confined to the base node is
wrong, and the change should not be trusted until that is understood.

    python verify/bedrock/compare_ab.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-port"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp  # noqa: E402

from verify.timestepping.pipeline.global_gswp3_test import (  # noqa: E402
    build_global_cell_grid,
)

ROOT = Path("/shared/nas2/Hangkai/ibis/verify_outputs")
ARMS = {"control (4 m)": ROOT / "spinup_bedrock_v9_control",
        "bedrock (12 m)": ROOT / "spinup_bedrock_v9_3lay"}
TMELT = 273.16
HSOI = jnp.array([0.10, 0.15, 0.25, 0.50, 1.00, 2.00])

def cell_area_km2(lat_deg, dlat=0.5, dlon=0.5):
    """Spherical cell area, so the extent total is not a cell count."""
    R = 6371.0
    lat = np.deg2rad(lat_deg)
    h = np.deg2rad(dlat)
    return (R ** 2 * np.deg2rad(dlon)
            * (np.sin(lat + h / 2) - np.sin(lat - h / 2)))


def load(arm_dir):
    p = arm_dir / "checkpoints" / "state_latest.pkl"
    if not p.exists():
        return None, None
    ck = pickle.load(open(p, "rb"))
    return ck["jax_state"], ck.get("step")


def main():
    cells = build_global_cell_grid(lat_step=0.5, lon_step=0.5)
    lat = np.array([c[1] for c in cells])
    area = cell_area_km2(lat)
    nh = lat > 0

    out = {}
    for name, d in ARMS.items():
        st, step = load(d)
        if st is None:
            print(f"  {name}: no checkpoint yet at {d}")
            continue
        tsoi = st.soil.tsoi
        # NO permafrost extent or active layer depth here. Both need the
        # seasonal cycle: in a final-state snapshot most of the boreal zone is
        # frozen, so "any frozen layer" counts seasonal frost and reports ~41
        # Mkm2 against a true ~23. Use verify/bedrock/seasonal_amplitude.py,
        # which tests "frozen in all 12 months" over a simulated year.
        out[name] = dict(
            step=step,
            tbase=float(jnp.mean(tsoi[:, -1])),
            tbase_nh=float(np.mean(np.asarray(tsoi[:, -1])[nh])),
            tdeep=(float(jnp.mean(st.soil.tsoi_deep[:, -1]))
                   if getattr(st.soil, "tsoi_deep", None) is not None else None),
            biomass=float(jnp.mean(jnp.sum(st.vegetation.biomass, axis=1))),
            totcsoi=float(jnp.mean(st.carbon.totcsoi)),
            nan=int(np.isnan(np.asarray(tsoi)).sum()),
        )

    if len(out) < 2:
        print("\n  Need both arms; rerun when the pair has finished.")
        return

    a, b = out["control (4 m)"], out["bedrock (12 m)"]
    print(f"\n  Bedrock A/B, step {a['step']} vs {b['step']}\n")
    print(f"  {'quantity':34} {'control':>11} {'bedrock':>11} {'delta':>11}")
    rows = [
        ("soil base T, global mean (K)", "tbase", 1.0),
        ("soil base T, NH mean (K)", "tbase_nh", 1.0),
        ("mean biomass (kgC/m2)", "biomass", 1.0),
        ("mean soil carbon (kgC/m2)", "totcsoi", 1.0),
    ]
    for label, k, _ in rows:
        d = b[k] - a[k]
        print(f"  {label:34} {a[k]:11.3f} {b[k]:11.3f} {d:+11.3f}")
    print(f"  {'bedrock base T (K)':34} {'--':>11} {b['tdeep']:11.3f} {'':>11}")
    print(f"  {'NaN in tsoi':34} {a['nan']:11d} {b['nan']:11d}")
    print("\n  Permafrost extent, active layer depth and seasonal amplitude "
          "all need the\n  seasonal cycle rather than this final state -- run "
          "verify/bedrock/seasonal_amplitude.py.")


if __name__ == "__main__":
    main()
