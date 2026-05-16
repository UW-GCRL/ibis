#!/usr/bin/env python3
"""Convert HWSD2 data to IBIS-JAX soil texture file (soita.nc).

Input
-----
    HWSD2.bil  : 30 arc-second global raster of SMU IDs (43200 × 21600, uint16)
    HWSD2.mdb  : Access/Jet database with soil properties per SMU and layer
                 (mdb-export binary extracted from mdbtools deb)

Output
------
    soita.nc   : NetCDF4 with sandpct and claypct on the GSWP3 0.5° grid
                 dimensions: (layer=6, lat=360, lon=720)

IBIS soil layer depths (from soil_init.py HSOI_LAYERS):
    Layer 1:   0 –  10 cm
    Layer 2:  10 –  25 cm
    Layer 3:  25 –  50 cm
    Layer 4:  50 – 100 cm
    Layer 5: 100 – 200 cm  (HWSD2 has no data; deepest HWSD2 layer repeated)
    Layer 6: 200 – 400 cm  (same)

HWSD2_LAYERS has TOPDEP/BOTDEP (cm) and SHARE (%) per component, so each
IBIS layer is the share-weighted, depth-overlap-weighted mean of all
HWSD2 components that intersect that depth range.

Usage
-----
    python scripts/hwsd2_to_soita.py \
        --hwsd-dir /shared/nas2/Min/HWSD \
        --output-dir /path/to/input_data

    # Skip MDB export if CSV already exists:
    python scripts/hwsd2_to_soita.py \
        --hwsd-dir /shared/nas2/Min/HWSD \
        --layers-csv /path/to/HWSD2_LAYERS.csv \
        --output-dir /path/to/input_data
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# mdb-export binary (extracted from deb without sudo)
MDB_EXPORT = "/home/min/.local/mdbtools/usr/bin/mdb-export"
MDB_LIB    = "/home/min/.local/mdbtools/usr/lib/x86_64-linux-gnu"

# IBIS layer depth ranges [top, bottom] in cm
IBIS_LAYERS = [
    (0,   10),
    (10,  25),
    (25,  50),
    (50,  100),
    (100, 200),
    (200, 400),
]
N_IBIS_LAYERS = len(IBIS_LAYERS)

# HWSD2 maximum depth with data (cm) — HWSD2 has 7 depth layers to 200 cm
# (D1: 0-20, D2: 20-40, D3: 40-60, D4: 60-80, D5: 80-100, D6: 100-150, D7: 150-200)
HWSD2_MAX_DEPTH = 200

# GSWP3 / IBIS target grid
N_LAT, N_LON = 360, 720
LAT0, LON0, RES = 89.75, -179.75, 0.5

# HWSD2 BIL raster
HWSD_NROWS, HWSD_NCOLS = 21600, 43200
HWSD_RES  = 1.0 / 120.0      # 30 arc-second
HWSD_NODATA = 65535
PIXELS_PER_CELL = int(round(RES / HWSD_RES))   # 60 × 60 per 0.5° cell

# Defaults for ocean / missing cells
DEFAULT_SAND, DEFAULT_CLAY = 40.0, 20.0


# ---------------------------------------------------------------------------
# Step 1: Export HWSD2_LAYERS from MDB
# ---------------------------------------------------------------------------

def export_layers_csv(mdb_path: str, csv_path: str) -> None:
    """Export HWSD2_LAYERS table to CSV using the local mdb-export binary."""
    if not os.path.exists(MDB_EXPORT):
        print(f"ERROR: mdb-export not found at {MDB_EXPORT}")
        print("Run this first to install without sudo:")
        print("  cd /home/min/ibis-jax")
        print("  apt-get download mdbtools libmdb3t64")
        print("  mkdir -p /home/min/.local/mdbtools")
        print("  dpkg-deb -x libmdb3t64_*.deb /home/min/.local/mdbtools/")
        print("  dpkg-deb -x mdbtools_*.deb    /home/min/.local/mdbtools/")
        sys.exit(1)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = MDB_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    print(f"Exporting HWSD2_LAYERS from {mdb_path} ...")
    try:
        result = subprocess.run(
            [MDB_EXPORT, mdb_path, "HWSD2_LAYERS"],
            capture_output=True, text=True, check=True, env=env,
        )
        with open(csv_path, "w") as f:
            f.write(result.stdout)
        n_rows = result.stdout.count("\n") - 1
        print(f"  Exported {n_rows:,} rows to {csv_path}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR running mdb-export: {e.stderr}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2: Build SMU → sand/clay lookup per IBIS layer
# ---------------------------------------------------------------------------

def build_smu_lookup(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse HWSD2_LAYERS CSV and build per-SMU, per-IBIS-layer sand/clay arrays.

    Returns
    -------
    sand_lut : (max_smu+1, N_IBIS_LAYERS) float32   percent sand
    clay_lut : (max_smu+1, N_IBIS_LAYERS) float32   percent clay
    """
    print(f"Parsing {csv_path} ...")

    # First pass: find column indices and max SMU ID
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        print(f"  Columns: {header}")

        smu_col    = _find_col(header, ["HWSD2_SMU_ID", "SMU_ID"])
        share_col  = _find_col(header, ["SHARE"])
        topdep_col = _find_col(header, ["TOPDEP"])
        botdep_col = _find_col(header, ["BOTDEP"])
        sand_col   = _find_col(header, ["SAND"])
        clay_col   = _find_col(header, ["CLAY"])

        # Accumulate (smu_id, share, topdep, botdep, sand, clay)
        records: list[tuple[int, float, float, float, float, float]] = []
        max_smu = 0
        for row in reader:
            try:
                smu    = int(float(row[smu_col]))
                share  = _safe_float(row[share_col],  100.0)   # default 100%
                topdep = _safe_float(row[topdep_col], 0.0)
                botdep = _safe_float(row[botdep_col], 30.0)
                sand_raw = row[sand_col].strip()
                clay_raw = row[clay_col].strip()
                sand = _safe_pct(sand_raw, -1.0)
                clay = _safe_pct(clay_raw, -1.0)
                if botdep <= topdep or smu <= 0:
                    continue
                # Skip rows where HWSD2 uses -9 (or empty) sentinel for texture
                if sand < 0 or clay < 0:
                    continue
                records.append((smu, share, topdep, botdep, sand, clay))
                if smu > max_smu:
                    max_smu = smu
            except (ValueError, KeyError, TypeError):
                continue

    print(f"  {len(records):,} valid layer records, max SMU ID = {max_smu:,}")

    # Build lookup arrays: numerator and denominator for weighted mean
    sand_num = np.zeros((max_smu + 1, N_IBIS_LAYERS), dtype=np.float64)
    clay_num = np.zeros((max_smu + 1, N_IBIS_LAYERS), dtype=np.float64)
    weight   = np.zeros((max_smu + 1, N_IBIS_LAYERS), dtype=np.float64)

    for smu, share, topdep, botdep, sand, clay in records:
        for k, (ibis_top, ibis_bot) in enumerate(IBIS_LAYERS):
            # Clip to HWSD2 data range (0-100 cm) for layers 5-6 later
            eff_top = max(topdep, ibis_top)
            eff_bot = min(botdep, ibis_bot)
            overlap = eff_bot - eff_top
            if overlap <= 0:
                continue
            w = share * overlap   # share(%) × depth(cm)
            sand_num[smu, k] += sand * w
            clay_num[smu, k] += clay * w
            weight[smu, k]   += w

    # Compute weighted means; fill missing with defaults
    sand_lut = np.full((max_smu + 1, N_IBIS_LAYERS), DEFAULT_SAND, dtype=np.float32)
    clay_lut = np.full((max_smu + 1, N_IBIS_LAYERS), DEFAULT_CLAY, dtype=np.float32)

    has_data = weight > 0
    sand_lut[has_data] = (sand_num[has_data] / weight[has_data]).astype(np.float32)
    clay_lut[has_data] = (clay_num[has_data] / weight[has_data]).astype(np.float32)

    # For any layer still missing data: fill from the nearest layer that has data.
    # HWSD2 goes to 200 cm so layers 1-5 should mostly be covered.
    # Layer 6 (200-400 cm) has no HWSD2 data → copy layer 5.
    for smu in range(max_smu + 1):
        if weight[smu].sum() == 0:
            continue  # SMU entirely missing — keep defaults
        for k in range(N_IBIS_LAYERS):
            if has_data[smu, k]:
                continue
            # Search outward from k for nearest layer with data
            for delta in range(1, N_IBIS_LAYERS):
                for kk in (k - delta, k + delta):
                    if 0 <= kk < N_IBIS_LAYERS and has_data[smu, kk]:
                        sand_lut[smu, k] = sand_lut[smu, kk]
                        clay_lut[smu, k] = clay_lut[smu, kk]
                        has_data[smu, k] = True
                        break
                if has_data[smu, k]:
                    break

    # Final physical constraints on lookup table
    np.clip(sand_lut, 0.0, 100.0, out=sand_lut)
    np.clip(clay_lut, 0.0, 100.0, out=clay_lut)
    # sand + clay must not exceed 100% (silt = 100 - sand - clay >= 0)
    total = sand_lut + clay_lut
    exceed = total > 100.0
    if exceed.any():
        scale = np.where(exceed, 100.0 / total, 1.0)
        sand_lut *= scale
        clay_lut *= scale

    n_smu_with_data = int((weight.sum(axis=1) > 0).sum())
    print(f"  SMUs with data: {n_smu_with_data:,}")
    return sand_lut, clay_lut


def _find_col(header: list[str], candidates: list[str]) -> str:
    lower = {h.lower(): h for h in (header or [])}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"Cannot find any of {candidates} in {header}")


def _safe_float(val: str, default: float) -> float:
    try:
        return float(val) if val.strip() else default
    except (ValueError, TypeError):
        return default


def _safe_pct(val: str, default: float) -> float:
    """Parse a soil texture percentage; reject out-of-range values (e.g. -9 sentinel)."""
    try:
        v = float(val) if val.strip() else default
        return v if 0.0 <= v <= 100.0 else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Step 3: Aggregate HWSD2 BIL to 0.5° grid
# ---------------------------------------------------------------------------

def aggregate_raster(bil_path: str,
                     sand_lut: np.ndarray,
                     clay_lut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read HWSD2 BIL in row-chunks, aggregate SMU IDs to 0.5° grid.

    Each 0.5° cell = mean of up to 60×60 = 3600 HWSD2 pixels.
    Pixels with NODATA (65535) or SMU > lookup size are skipped.

    Returns
    -------
    sand_grid, clay_grid : (N_IBIS_LAYERS, N_LAT, N_LON) float32
    """
    from osgeo import gdal
    gdal.UseExceptions()

    max_smu = sand_lut.shape[0]
    ds = gdal.Open(bil_path)
    band = ds.GetRasterBand(1)

    sand_grid = np.full((N_IBIS_LAYERS, N_LAT, N_LON), DEFAULT_SAND, dtype=np.float32)
    clay_grid = np.full((N_IBIS_LAYERS, N_LAT, N_LON), DEFAULT_CLAY, dtype=np.float32)

    print(f"Aggregating {HWSD_NROWS}×{HWSD_NCOLS} BIL → {N_LAT}×{N_LON} ...")
    print(f"  ({PIXELS_PER_CELL}×{PIXELS_PER_CELL} HWSD pixels per 0.5° cell)")

    for irow in range(N_LAT):
        if irow % 36 == 0:
            pct = 100 * irow // N_LAT
            print(f"  {pct}% (row {irow}/{N_LAT})")

        bil_rows = band.ReadAsArray(0, irow * PIXELS_PER_CELL,
                                    HWSD_NCOLS, PIXELS_PER_CELL)
        # bil_rows: (60, 43200) uint16

        for icol in range(N_LON):
            cell = bil_rows[:, icol * PIXELS_PER_CELL:(icol + 1) * PIXELS_PER_CELL]
            valid_mask = (cell != HWSD_NODATA) & (cell < max_smu)
            smu_ids = cell[valid_mask]
            if len(smu_ids) == 0:
                continue   # ocean or missing

            # Mean sand/clay across all valid pixels in cell
            for k in range(N_IBIS_LAYERS):
                sand_grid[k, irow, icol] = np.mean(sand_lut[smu_ids, k])
                clay_grid[k, irow, icol] = np.mean(clay_lut[smu_ids, k])

    ds = None
    print("  100% done.")
    return sand_grid, clay_grid


# ---------------------------------------------------------------------------
# Step 4: Write soita.nc
# ---------------------------------------------------------------------------

def write_soita_nc(sand_grid: np.ndarray, clay_grid: np.ndarray,
                   output_path: str) -> None:
    import netCDF4 as nc

    lat = np.array([LAT0 - i * RES for i in range(N_LAT)], dtype=np.float64)
    lon = np.array([LON0 + j * RES for j in range(N_LON)], dtype=np.float64)
    layers = np.arange(1, N_IBIS_LAYERS + 1, dtype=np.int32)

    print(f"Writing {output_path} ...")
    ds = nc.Dataset(output_path, "w", format="NETCDF4")

    ds.createDimension("layer", N_IBIS_LAYERS)
    ds.createDimension("lat",   N_LAT)
    ds.createDimension("lon",   N_LON)

    v = ds.createVariable("lat", "f8", ("lat",))
    v[:] = lat; v.units = "degrees_north"; v.long_name = "Latitude"

    v = ds.createVariable("lon", "f8", ("lon",))
    v[:] = lon; v.units = "degrees_east";  v.long_name = "Longitude"

    v = ds.createVariable("layer", "i4", ("layer",))
    v[:] = layers
    v.long_name = "Soil layer index (1 = shallowest)"
    v.depth_ranges_cm = "0-10, 10-25, 25-50, 50-100, 100-200, 200-400"

    kw = dict(fill_value=-9999.0, zlib=True, complevel=4)

    v = ds.createVariable("sandpct", "f4", ("layer", "lat", "lon"), **kw)
    v[:] = sand_grid
    v.units = "%"; v.long_name = "Sand fraction by weight"
    v.source = "HWSD2 (FAO/IIASA/ISRIC/ISS-CAS/JRC, 2022)"

    v = ds.createVariable("claypct", "f4", ("layer", "lat", "lon"), **kw)
    v[:] = clay_grid
    v.units = "%"; v.long_name = "Clay fraction by weight"
    v.source = "HWSD2 (FAO/IIASA/ISRIC/ISS-CAS/JRC, 2022)"

    ds.title = "IBIS-JAX soil texture derived from HWSD2"
    ds.grid  = ("Global 0.5-degree; lat N→S (89.75 to -89.75); "
                "lon W→E (-179.75 to 179.75)")
    ds.aggregation = (f"Pixel-count mean of {PIXELS_PER_CELL}×{PIXELS_PER_CELL} "
                      "HWSD2 30-arcsec SMU pixels per 0.5° cell. "
                      "Per-SMU values are SHARE×depth-overlap-weighted means "
                      "of HWSD2_LAYERS rows (D1-D7, covers 0-200 cm). "
                      "Layer 6 (200-400 cm) extrapolated from layer 5. "
                      "Missing layers filled from nearest available depth.")
    ds.created = datetime.datetime.utcnow().isoformat() + " UTC"
    ds.close()

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Done → {output_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert HWSD2 to IBIS-JAX soita.nc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hwsd-dir",    default="/shared/nas2/Min/HWSD",
                        help="Directory with HWSD2.bil and HWSD2.mdb")
    parser.add_argument("--output-dir",  default="input",
                        help="Output directory for soita.nc")
    parser.add_argument("--layers-csv",  default=None,
                        help="Pre-exported HWSD2_LAYERS.csv (skips mdb-export)")
    parser.add_argument("--output-name", default="soita.nc")
    args = parser.parse_args()

    hwsd_dir    = Path(args.hwsd_dir)
    output_path = str(Path(args.output_dir) / args.output_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- CSV ---
    if args.layers_csv and Path(args.layers_csv).exists():
        csv_path = args.layers_csv
    else:
        mdb_path = hwsd_dir / "HWSD2.mdb"
        if not mdb_path.exists():
            print(f"ERROR: {mdb_path} not found"); sys.exit(1)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        csv_path = tmp.name; tmp.close()
        export_layers_csv(str(mdb_path), csv_path)

    # --- Lookup table ---
    sand_lut, clay_lut = build_smu_lookup(csv_path)

    # --- Aggregate BIL ---
    bil_path = hwsd_dir / "HWSD2.bil"
    if not bil_path.exists():
        print(f"ERROR: {bil_path} not found"); sys.exit(1)
    sand_grid, clay_grid = aggregate_raster(str(bil_path), sand_lut, clay_lut)

    # --- Final clamp on output grid ---
    np.clip(sand_grid, 0.0, 100.0, out=sand_grid)
    np.clip(clay_grid, 0.0, 100.0, out=clay_grid)
    total = sand_grid + clay_grid
    exceed = total > 100.0
    if exceed.any():
        scale = np.where(exceed, 100.0 / total, 1.0)
        sand_grid *= scale
        clay_grid *= scale

    # --- Sanity check ---
    land = sand_grid[0] != DEFAULT_SAND
    print(f"\nSanity check (layer 1, {land.sum():,} land cells):")
    print(f"  Sand: mean={sand_grid[0][land].mean():.1f}%  "
          f"range [{sand_grid[0][land].min():.0f}, {sand_grid[0][land].max():.0f}]%")
    print(f"  Clay: mean={clay_grid[0][land].mean():.1f}%  "
          f"range [{clay_grid[0][land].min():.0f}, {clay_grid[0][land].max():.0f}]%")

    # --- Write ---
    write_soita_nc(sand_grid, clay_grid, output_path)


if __name__ == "__main__":
    main()
