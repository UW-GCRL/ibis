"""Restart (checkpoint) save/restore for IBIS-JAX.

Serializes the full IBISState pytree to a single file using equinox's
serialization, which handles arbitrary JAX arrays and nested pytrees.
Falls back to a NumPy-based approach for portability.
"""

from __future__ import annotations

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx


def save_state(state, path: str, metadata: Optional[dict] = None) -> None:
    """Save IBISState to a checkpoint file.

    Uses equinox serialization for clean pytree save/restore.

    Parameters
    ----------
    state : IBISState
        Model state to save.
    path : str
        Output file path (will create directories as needed).
    metadata : dict, optional
        Extra metadata (year, month, day, step) saved alongside state.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Flatten the pytree into leaves + treedef
    leaves, treedef = jax.tree.flatten(state)

    save_dict = {
        "treedef_repr": repr(treedef),
        "num_leaves": len(leaves),
    }

    # Save each leaf as a numpy array
    for i, leaf in enumerate(leaves):
        if leaf is None:
            save_dict[f"leaf_{i}_is_none"] = True
        else:
            save_dict[f"leaf_{i}"] = np.asarray(leaf)

    # Save metadata
    if metadata is not None:
        for k, v in metadata.items():
            save_dict[f"meta_{k}"] = np.array(v)

    np.savez_compressed(path, **save_dict)


#: Leaf counts of the optional trailing modules, in chain order (StressAccum, ETParts, SeasonalState).
#: Guarded against the real modules by tests/test_seasonal_formulations.py.
CHAIN_LEAVES = (3, 4, 3, 2)      # StressAccum, ETParts, SeasonalState, NEEParts


def load_state(path: str, template_state):
    """Load IBISState from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to the checkpoint file (.npz).
    template_state : IBISState
        A template state with the correct structure (shapes, dtypes).
        The loaded arrays are placed into this structure.

    Returns
    -------
    state : IBISState
        Restored model state.
    metadata : dict
        Any metadata saved with the checkpoint.
    """
    if not path.endswith(".npz"):
        path = path + ".npz"

    data = np.load(path, allow_pickle=False)

    # Get template structure
    leaves_template, treedef = jax.tree.flatten(template_state)
    num_leaves = int(data["num_leaves"])

    n_fill = len(leaves_template) - num_leaves
    # The optional trailing modules form one nested chain: StressAccum (3 leaves), ETParts (4), SeasonalState (3),
    # NEEParts (2), each implying the ones before it, so a state carries 0, 3, 7, 10 or 12 trailing leaves
    # (258 / 261 / 265 / 268 / 270).
    # A checkpoint may load into a template anywhere on that chain; any other difference is a real mismatch.
    _chain = [0]
    for _n in CHAIN_LEAVES:
        _chain.append(_chain[-1] + _n)
    _off_t = sum(len(jax.tree.leaves(_m)) for _m in (getattr(template_state, "stress", None),
                                                     getattr(template_state, "etparts", None),
                                                     getattr(template_state, "seasonal", None),
                                                     getattr(template_state, "neeparts", None)) if _m is not None)
    _off_c = _off_t - n_fill
    if _off_t not in _chain or _off_c not in _chain:
        raise ValueError(
            f"Checkpoint has {num_leaves} leaves but template has {len(leaves_template)} "
            f"(trailing offsets {_off_c} and {_off_t} are not both on the optional-module chain {_chain}). "
            f"State structure mismatch. Use scripts/convert_restart.py to migrate old restarts.")
    if n_fill < 0:
        # The converse migration: a checkpoint written WITH the StressAccum
        # (a gated run) loading into a template without it. The trailing
        # leaves are the accumulator; an ungated run has no use for them.
        import logging
        logging.getLogger(__name__).warning(
            "Checkpoint has %d leaves, template %d: dropping the %d trailing "
            "leaves (optional trailing accumulators not in this template).",
            num_leaves, len(leaves_template), -n_fill)
        n_fill = 0
    if n_fill < 0 or n_fill > _chain[-1]:
        raise ValueError(
            f"Checkpoint has {num_leaves} leaves but template has "
            f"{len(leaves_template)}. State structure mismatch. "
            f"Use scripts/convert_restart.py to migrate old restarts."
        )
    if n_fill:
        # Trailing-leaf migration: a template with the optional StressAccum
        # (3 leaves, declared after every other field) may load a checkpoint
        # written without it; the accumulator starts from the template's zeros.
        import logging
        logging.getLogger(__name__).warning(
            "Checkpoint has %d leaves, template %d: filling the %d trailing "
            "leaves (optional accumulators) from the template.",
            num_leaves, len(leaves_template), n_fill)

    # Restore leaves
    new_leaves = []
    for i, tmpl in enumerate(leaves_template):
        if i >= num_leaves:
            new_leaves.append(tmpl)
            continue
        if f"leaf_{i}_is_none" in data:
            new_leaves.append(None)
        else:
            arr = data[f"leaf_{i}"]
            # Convert to JAX array with matching dtype
            if tmpl is not None:
                arr = jnp.array(arr, dtype=jnp.asarray(tmpl).dtype)
            else:
                arr = jnp.array(arr)
            new_leaves.append(arr)

    state = jax.tree.unflatten(treedef, new_leaves)

    # Extract metadata
    metadata = {}
    for key in data.files:
        if key.startswith("meta_"):
            meta_key = key[5:]  # strip "meta_"
            val = data[key]
            metadata[meta_key] = val.item() if val.ndim == 0 else val

    return state, metadata


def restart_path(restart_dir: str, year: int, prefix: str = "ibis") -> str:
    """Generate a standard restart file path.

    Parameters
    ----------
    restart_dir : str
        Directory for restart files.
    year : int
        Simulation year.
    prefix : str
        File prefix.

    Returns
    -------
    str
        Full path like ``restart_dir/ibis_restart_1961.npz``.
    """
    return os.path.join(restart_dir, f"{prefix}_restart_{year}.npz")
