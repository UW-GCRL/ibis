"""Residual statistics for verification.

Each comparison computes a row of stats; rows aggregate into a DataFrame
that is the durable verification artifact (residuals stay on disk for
later inspection). Tests then assert on these stats against tolerances.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualStats:
    """Stats over (reference, candidate) on a single variable."""
    n:           int     # number of compared elements (post-mask)
    max_abs:     float   # max |ref - cand|
    max_rel:     float   # max |ref - cand| / max(|ref|, abs_floor)
    rmse:        float   # sqrt(mean((ref - cand)^2))
    bias:        float   # mean(ref - cand) -- signed
    max_ulp:     float   # max distance in ULPs (float64)
    n_nan_ref:   int
    n_nan_cand:  int

    def as_dict(self) -> dict:
        return {
            "n":          self.n,
            "max_abs":    self.max_abs,
            "max_rel":    self.max_rel,
            "rmse":       self.rmse,
            "bias":       self.bias,
            "max_ulp":    self.max_ulp,
            "n_nan_ref":  self.n_nan_ref,
            "n_nan_cand": self.n_nan_cand,
        }


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between a and b in float64 ULPs.

    Uses the IEEE-754 trick: consecutive representable floats have
    consecutive integer encodings when the float bits are reinterpreted
    as ordered (sign-magnitude → biased-twos-complement) integers.
    """
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    ai = a64.view(np.int64).copy()
    bi = b64.view(np.int64).copy()
    # For negative floats, int view has the sign bit set so signed int64
    # is very negative. Map to ordered form via (INT64_MIN - x), which
    # keeps everything in int64 range and preserves adjacency.
    INT64_MIN = np.int64(np.iinfo(np.int64).min)
    ai = np.where(ai < 0, INT64_MIN - ai, ai)
    bi = np.where(bi < 0, INT64_MIN - bi, bi)
    return np.abs(ai - bi).astype(np.float64)


def compute(reference: np.ndarray, candidate: np.ndarray,
            *, abs_floor: float = 1.0) -> ResidualStats:
    """Compute residual stats. Skips elements where either side is NaN.

    abs_floor controls the denominator clamp for max_rel: values smaller
    than abs_floor use abs_floor as the denominator. Matches
    compare_outputs.py style so near-zero values are not penalized.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {cand.shape}")

    n_nan_ref = int(np.sum(~np.isfinite(ref)))
    n_nan_cand = int(np.sum(~np.isfinite(cand)))
    mask = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(mask):
        return ResidualStats(0, np.nan, np.nan, np.nan, np.nan, np.nan,
                             n_nan_ref, n_nan_cand)

    r = ref[mask]
    c = cand[mask]
    diff = r - c
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(r), abs_floor)

    return ResidualStats(
        n=int(r.size),
        max_abs=float(abs_diff.max()),
        max_rel=float((abs_diff / denom).max()),
        rmse=float(np.sqrt(np.mean(diff * diff))),
        bias=float(np.mean(diff)),
        max_ulp=float(_ulp_distance(r, c).max()),
        n_nan_ref=n_nan_ref,
        n_nan_cand=n_nan_cand,
    )
