"""Site-level validation against FLUXNET observations.

Compares IBIS-JAX output against FLUXNET eddy covariance data for
key fluxes: NEE, latent heat, sensible heat, soil heat flux.

Usage:
    python scripts/validate_fluxnet.py config.yaml fluxnet_data.csv [--site US-Ha1]

Input:
    config.yaml — IBIS-JAX run configuration
    fluxnet_data.csv — FLUXNET half-hourly or daily data file

Metrics:
    RMSE, bias, R^2, Nash-Sutcliffe efficiency for each flux variable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np


class ValidationMetrics(NamedTuple):
    """Validation metrics for a single variable."""
    name: str
    n: int
    rmse: float
    bias: float
    r2: float
    nse: float      # Nash-Sutcliffe efficiency


def compute_metrics(obs: np.ndarray, mod: np.ndarray, name: str = "") -> ValidationMetrics:
    """Compute validation metrics between observed and modelled time series.

    Parameters
    ----------
    obs : (N,) observed values
    mod : (N,) modelled values
    name : variable name

    Returns
    -------
    ValidationMetrics
    """
    # Remove NaN pairs
    mask = np.isfinite(obs) & np.isfinite(mod)
    obs = obs[mask]
    mod = mod[mask]
    n = len(obs)

    if n < 2:
        return ValidationMetrics(name=name, n=n, rmse=np.nan,
                                 bias=np.nan, r2=np.nan, nse=np.nan)

    residual = mod - obs
    bias = np.mean(residual)
    rmse = np.sqrt(np.mean(residual**2))

    # R^2 (coefficient of determination)
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    # Nash-Sutcliffe efficiency
    nse = 1.0 - ss_res / max(ss_tot, 1e-10)

    return ValidationMetrics(name=name, n=n, rmse=rmse, bias=bias, r2=r2, nse=nse)


def print_metrics(metrics: list[ValidationMetrics]):
    """Print a table of validation metrics."""
    print(f"\n{'Variable':<20} {'N':>6} {'RMSE':>10} {'Bias':>10} {'R2':>8} {'NSE':>8}")
    print("-" * 64)
    for m in metrics:
        print(f"{m.name:<20} {m.n:>6} {m.rmse:>10.3f} {m.bias:>10.3f} "
              f"{m.r2:>8.3f} {m.nse:>8.3f}")


def load_fluxnet_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load FLUXNET CSV file and extract key variables.

    Handles FLUXNET2015 format with common variable names:
      NEE_VUT_REF, LE_F_MDS, H_F_MDS, G_F_MDS, TA_F, SW_IN_F, etc.

    Parameters
    ----------
    path : path to FLUXNET CSV file

    Returns
    -------
    dict mapping variable names to numpy arrays
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for FLUXNET validation: pip install pandas")
        sys.exit(1)

    df = pd.read_csv(path)

    # Standard FLUXNET2015 variable mapping
    var_map = {
        "NEE_VUT_REF": "nee",
        "LE_F_MDS": "le",
        "H_F_MDS": "h",
        "G_F_MDS": "g",
        "TA_F": "ta",
        "SW_IN_F": "sw_in",
        "P_F": "precip",
        "TS_F_MDS_1": "tsoil",
        "SWC_F_MDS_1": "swc",
    }

    data = {}
    for fluxnet_name, our_name in var_map.items():
        if fluxnet_name in df.columns:
            arr = df[fluxnet_name].values.astype(np.float64)
            # FLUXNET uses -9999 as missing value
            arr[arr < -9998] = np.nan
            data[our_name] = arr

    # Timestamp
    for ts_col in ["TIMESTAMP_START", "TIMESTAMP", "Date"]:
        if ts_col in df.columns:
            data["timestamp"] = df[ts_col].values
            break

    return data


def validate_against_fluxnet(
    model_output: dict[str, np.ndarray],
    fluxnet_data: dict[str, np.ndarray],
) -> list[ValidationMetrics]:
    """Compare model output against FLUXNET observations.

    Parameters
    ----------
    model_output : dict with keys like 'nee', 'le', 'h', etc.
    fluxnet_data : dict from load_fluxnet_csv

    Returns
    -------
    list of ValidationMetrics
    """
    metrics = []

    var_pairs = [
        ("nee", "NEE (umol/m2/s)"),
        ("le", "Latent heat (W/m2)"),
        ("h", "Sensible heat (W/m2)"),
        ("g", "Ground heat flux (W/m2)"),
        ("tsoil", "Soil temperature (C)"),
        ("swc", "Soil water content (%)"),
    ]

    for var_key, var_label in var_pairs:
        if var_key in model_output and var_key in fluxnet_data:
            obs = fluxnet_data[var_key]
            mod = model_output[var_key]

            # Trim to matching length
            n = min(len(obs), len(mod))
            m = compute_metrics(obs[:n], mod[:n], var_label)
            metrics.append(m)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="FLUXNET Validation")
    parser.add_argument("config", help="IBIS-JAX config YAML")
    parser.add_argument("fluxnet_csv", help="FLUXNET data CSV file")
    parser.add_argument("--site", default="", help="Site name for reporting")
    parser.add_argument("--output", default=None, help="Output CSV for metrics")
    args = parser.parse_args()

    print(f"IBIS-JAX FLUXNET Validation")
    if args.site:
        print(f"Site: {args.site}")

    # Load FLUXNET data
    fluxnet_data = load_fluxnet_csv(args.fluxnet_csv)
    print(f"Loaded FLUXNET data: {list(fluxnet_data.keys())}")

    # TODO: Run IBIS-JAX simulation or load existing output
    # For now, just demonstrate the validation framework
    print("\nTo complete validation:")
    print("  1. Run IBIS-JAX with: python scripts/run_ibis.py", args.config)
    print("  2. Extract model output time series at the flux tower location")
    print("  3. Call validate_against_fluxnet(model_output, fluxnet_data)")

    # Save metrics if requested
    if args.output and len(fluxnet_data) > 0:
        print(f"\nMetrics would be saved to: {args.output}")


if __name__ == "__main__":
    main()
