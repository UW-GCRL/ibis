#!/usr/bin/env python3
"""Convert old ibis-jax restart files to current pytree structure.

Loads the old restart with the backup code, then transfers each module
field-by-field to a new-format state using eqx.tree_at.

Usage:
    python scripts/convert_restart.py restart/spinup/ibis_restart_200.npz \
           restart/spinup/ibis_restart_200_new.npz
"""
import sys
import os
import importlib

# IMPORTANT: Must set x64 before ANY jax import
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

# ---- Step 1: Load with OLD code ----
# Temporarily use backup code to create old template and load
BACKUP_DIR = "/home/min/ibis-jax-backup-20260402"
sys.path.insert(0, BACKUP_DIR)

# Import old modules
from ibis.state.model_state import create_ibis_state as old_create_ibis_state
from ibis.dimensions import GridConfig
import netCDF4 as nc

# Get npoi from land mask
ds = nc.Dataset("/shared/nas2/Min/HWSD/surta.nc")
mask = np.array(ds.variables["landfrac"][:], dtype=np.float32).squeeze()
ds.close()
npoi = int(np.sum(mask > 0))
print(f"Land points: {npoi}")

grid = GridConfig(nlon=720, nlat=360, npoi=npoi, xres=0.5, yres=0.5,
                  nband=2, nsoilay=6, nsnolay=3, npft=12)

# Create old template
old_template = old_create_ibis_state(grid)
old_leaves, old_treedef = jax.tree.flatten(old_template)
print(f"Old template: {len(old_leaves)} leaves")

# Load restart
src_path = sys.argv[1] if len(sys.argv) > 1 else "restart/spinup/ibis_restart_200_orig.npz"
data = np.load(src_path)
num_leaves = int(data["num_leaves"])
print(f"Checkpoint: {num_leaves} leaves")

if num_leaves > len(old_leaves):
    raise ValueError(f"Checkpoint has MORE leaves ({num_leaves}) than old template ({len(old_leaves)})")

# Load each leaf from checkpoint, use template for extras
loaded_leaves = []
for i in range(len(old_leaves)):
    if i < num_leaves and f"leaf_{i}" in data:
        arr = data[f"leaf_{i}"]
        tmpl = old_leaves[i]
        if tmpl is not None and hasattr(tmpl, 'dtype'):
            loaded_leaves.append(jnp.array(arr, dtype=jnp.float64 if jnp.issubdtype(tmpl.dtype, jnp.floating) else tmpl.dtype))
        else:
            loaded_leaves.append(jnp.array(arr))
    else:
        loaded_leaves.append(old_leaves[i])

old_state = jax.tree.unflatten(old_treedef, loaded_leaves)

# Verify
print(f"Old state loaded: tsoi finite={100*np.isfinite(np.array(old_state.soil.tsoi)).mean():.0f}%")
print(f"  tg mean={float(jnp.nanmean(old_state.soil.tg)):.1f} K")
print(f"  fl shape={old_state.vegetation.fl.shape}, range=[{float(old_state.vegetation.fl.min()):.4f}, {float(old_state.vegetation.fl.max()):.4f}]")

# ---- Step 2: Clear old imports, load NEW code ----
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("ibis"):
        del sys.modules[mod_name]
sys.path.remove(BACKUP_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibis.state.model_state import create_ibis_state as new_create_ibis_state
from ibis.io.restart import save_state

# Create new template
new_template = new_create_ibis_state(grid)
new_leaves, _ = jax.tree.flatten(new_template)
print(f"New template: {len(new_leaves)} leaves")

# ---- Step 3: Copy module-by-module via eqx.tree_at ----
# All modules except Summary are structurally identical
new_state = eqx.tree_at(
    lambda s: (s.soil, s.snow, s.canopy, s.atm, s.vegetation, s.carbon, s.hydrology),
    new_template,
    (old_state.soil, old_state.snow, old_state.canopy, old_state.atm,
     old_state.vegetation, old_state.carbon, old_state.hydrology),
)

# Copy Summary field-by-field (skip new fields which stay at template defaults = 0)
old_sm = old_state.summary
new_state = eqx.tree_at(
    lambda s: (s.summary.a10td, s.summary.a10ancub, s.summary.a10ancuc,
               s.summary.a10ancls, s.summary.a10ancl3, s.summary.a10ancl4,
               s.summary.a10scalparamu, s.summary.a10scalparaml,
               s.summary.adrain, s.summary.adsnow, s.summary.adaet, s.summary.adneetot,
               s.summary.ayanpptot, s.summary.ayneetot, s.summary.ayprcp, s.summary.ayaet,
               s.summary.aynpp, s.summary.ayanpp,
               s.summary.storedn, s.summary.yrleach, s.summary.ynleach),
    new_state,
    (old_sm.a10td, old_sm.a10ancub, old_sm.a10ancuc,
     old_sm.a10ancls, old_sm.a10ancl3, old_sm.a10ancl4,
     old_sm.a10scalparamu, old_sm.a10scalparaml,
     old_sm.adrain, old_sm.adsnow, old_sm.adaet, old_sm.adneetot,
     old_sm.ayanpptot, old_sm.ayneetot, old_sm.ayprcp, old_sm.ayaet,
     old_sm.aynpp, old_sm.ayanpp,
     old_sm.storedn, old_sm.yrleach, old_sm.ynleach),
)

# ---- Step 4: Verify ----
print(f"\nNew state verification:")
print(f"  tsoi finite: {100*np.isfinite(np.array(new_state.soil.tsoi)).mean():.0f}%")
print(f"  tg mean: {float(jnp.nanmean(new_state.soil.tg)):.1f} K")
print(f"  fl shape: {new_state.vegetation.fl.shape}")
print(f"  fl range: [{float(new_state.vegetation.fl.min()):.4f}, {float(new_state.vegetation.fl.max()):.4f}]")
print(f"  froot shape: {new_state.vegetation.froot.shape}")
print(f"  aygpptot sum: {float(new_state.summary.aygpptot.sum()):.0f} (should be 0)")
print(f"  Total leaves: {len(jax.tree.leaves(new_state))}")

# Extract metadata
metadata = {}
for key in data.files:
    if key.startswith("meta_"):
        val = data[key]
        metadata[key[5:]] = val.item() if val.ndim == 0 else val

# ---- Step 5: Save ----
dst_path = sys.argv[2] if len(sys.argv) > 2 else src_path.replace(".npz", "_converted.npz")
save_state(new_state, dst_path, metadata=metadata)
print(f"\nSaved: {dst_path}")
