"""The global cell grid must map one row per land cell.

It used to sample latitudes/longitudes on cell BOUNDARIES (X.0 / X.5) while
`latlon_to_grid_idx` rounds against centres at X.25 / X.75. Every sample then
landed on an exact .5, and Python's round-half-to-even collapsed adjacent
samples in pairs -- in both axes, so four rows per cell. At 0.5 deg that gave
88 484 rows covering 22 301 distinct cells: 4x the compute for no extra
information, an effective 1 deg resolution behind a 0.5 deg label, and 77% of
land never simulated.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (REPO / "verify/timestepping/pipeline/global_gswp3_test.py").exists(),
    reason="verify/ tree not present")


def _grid(step):
    for p in (str(REPO), str(REPO / "python-port")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from verify.timestepping.pipeline.global_gswp3_test import (
        build_global_cell_grid, latlon_to_grid_idx)
    cells = build_global_cell_grid(step, step)
    idx = [latlon_to_grid_idx(c[1], c[2]) for c in cells]
    return cells, idx


@pytest.mark.parametrize("step", [0.5, 1.0, 2.0])
def test_no_duplicate_grid_cells(step):
    cells, idx = _grid(step)
    dup = len(idx) - len(set(idx))
    assert dup == 0, (
        f"{dup}/{len(idx)} rows at {step} deg map to a cell another row already "
        f"claims. Rows must sample cell centres (X.25/X.75), not boundaries.")


def test_half_degree_grid_covers_every_land_cell():
    """At the grid's own resolution, every land cell must appear exactly once."""
    from netCDF4 import Dataset
    for p in (str(REPO), str(REPO / "python-port")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from verify.timestepping.pipeline.global_gswp3_test import LAND_MASK_FILE
    if not Path(LAND_MASK_FILE).exists():
        pytest.skip("land mask not available")
    ds = Dataset(str(LAND_MASK_FILE))
    n_land = int((np.asarray(ds.variables["landfrac"][:]) > 0.5).sum())
    ds.close()
    cells, _ = _grid(0.5)
    assert len(cells) == n_land, (
        f"0.5 deg grid has {len(cells)} rows for {n_land} land cells -- it must "
        f"be exactly one row per land cell, no duplicates and none skipped.")
