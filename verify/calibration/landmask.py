"""The land fraction IBIS's binary mask threw away.

`surta.nc` (the IBIS-JAX land mask, 95,333 cells, 165 Mkm2) is `landfrac = 1`
wherever `vegtype_frac_1992.nc` has any vegetation fraction at all. That file's
15 IBIS vegetation-type fractions sum to the true land fraction of the cell
(ESA CCI Land Cover, 300 m aggregated to 0.5 degree): 49,682 cells are fully
land, 45,651 are fractional, and open-water cells with a speck of island
(sum 0.00-0.04 in the Sulu and Molucca seas) were made 100 % land. Every PgC
total weighted with the binary mask over-counts (CALIBRATION.md 18); the
model's per-m2 fluxes are unaffected.
"""
import numpy as np, xarray as xr

VEGTYPE_1992 = "/shared/nas2/Min/ESACCI_LC/ibis_vegtype/vegtype_frac_1992.nc"


def true_landfrac(path=VEGTYPE_1992):
    """(360, 720) land fraction on the IBIS N->S grid: sum of the vegtype fractions, clipped to [0, 1]."""
    d = xr.open_dataset(path)
    s = np.asarray(d["vegfrac"].values, dtype=float).sum(axis=-1)
    if float(d["lat"].values[0]) < float(d["lat"].values[-1]):
        s = s[::-1, :]
    return np.clip(np.nan_to_num(s), 0.0, 1.0)
