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
CHAIN_LEAVES = (4, 4, 3, 2)      # StressAccum (4 since CALIBRATION.md 71: + u_smooth), ETParts, SeasonalState, NEEParts
#: the StressAccum had 3 leaves before 2026-10-28; such checkpoints get a zero (= unset) u_smooth leaf on loading
OLD_STRESS_LEAVES = 3


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
    num_leaves = int(data["num_leaves"])
    ck = [None if f"leaf_{i}_is_none" in data else data[f"leaf_{i}"] for i in range(num_leaves)]

    # Get template structure
    leaves_template, treedef = jax.tree.flatten(template_state)

    n_fill = len(leaves_template) - num_leaves
    # The optional trailing modules form one nested chain: StressAccum (3 leaves), ETParts (4), SeasonalState (3),
    # NEEParts (2), each implying the ones before it, so a state carries 0, 3, 7, 10 or 12 trailing leaves
    # (258 / 261 / 265 / 268 / 270).
    # A checkpoint may load into a template anywhere on that chain; any other difference is a real mismatch.
    _chain = [0]
    for _n in CHAIN_LEAVES:
        _chain.append(_chain[-1] + _n)
    _old_chain = [0]                     # offsets a checkpoint with the 3-leaf StressAccum can carry
    for _n in (OLD_STRESS_LEAVES,) + CHAIN_LEAVES[1:]:
        _old_chain.append(_old_chain[-1] + _n)
    _off_t = sum(len(jax.tree.leaves(_m)) for _m in (getattr(template_state, "stress", None),
                                                     getattr(template_state, "etparts", None),
                                                     getattr(template_state, "seasonal", None),
                                                     getattr(template_state, "neeparts", None)) if _m is not None)
    _off_c = _off_t - n_fill
    if _off_c not in _chain and _off_c not in _old_chain and ((_off_c - 3) in _chain or (_off_c - 3) in _old_chain) \
            and getattr(template_state, "wetland", None) is None:
        # The checkpoint carries the wetland MethaneState (3 leaves: flux, annual, zwt; declared before the trailing
        # chain, so it is not a trailing module). Attach one to the template so every leaf aligns, and load it: a
        # restart of a methane-enabled run keeps its water table and annual sums (CALIBRATION.md 38).
        import equinox as eqx, logging
        from ibis.bgc.methane import create_methane_state
        _npoi = int(jnp.asarray(template_state.soil.tg).shape[0])
        template_state = eqx.tree_at(lambda s: s.wetland, template_state, create_methane_state(_npoi), is_leaf=lambda x: x is None)
        leaves_template, treedef = jax.tree.flatten(template_state); n_fill = len(leaves_template) - num_leaves; _off_c = _off_t - n_fill
        logging.getLogger(__name__).warning("Checkpoint carries the wetland CH4 state (3 leaves): attached to the template and loaded.")
    _base_t = len(leaves_template) - _off_t          # leaves before the optional chain (the wetland state included)
    _trail_c = num_leaves - _base_t
    if getattr(template_state, "stress", None) is not None and _trail_c in _old_chain[1:] and _trail_c not in _chain:
        # Written with the 3-leaf StressAccum (before CALIBRATION.md 71): insert the smoothed-stress leaf as zeros
        # (= unset: the first year's mean seeds it). Every restart on disk keeps loading.
        import logging
        ck.insert(_base_t + OLD_STRESS_LEAVES, np.zeros_like(np.asarray(ck[_base_t])))
        num_leaves += 1; n_fill = len(leaves_template) - num_leaves; _off_c = _off_t - n_fill
        logging.getLogger(__name__).warning("Checkpoint carries the 3-leaf StressAccum: u_smooth leaf inserted as zeros (unset).")
    if _off_t not in _chain or (_off_c not in _chain and _off_c not in _old_chain):
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
    if n_fill < 0 or n_fill > _chain[-1]:                       # (an old-chain checkpoint longer than the template was dropped above)
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
        if ck[i] is None:
            new_leaves.append(None)
        else:
            arr = ck[i]
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
