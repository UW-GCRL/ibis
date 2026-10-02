#!/usr/bin/env python3
"""Stage 1: joint per-PFT vmax + specla fit against FLUXCOM and MODIS.

Design follows what the diagnosis established (docs/CALIBRATION.md):

* **Magnitude comes from gridded products, never tower point values.** The
  model is 1.09x FLUXCOM at tower cells but 0.81x the tower value, and
  FLUXCOM -- trained on those towers -- is itself 0.75x them at their own
  cells. Fitting the tower gap would inflate global GPP ~1.4x while every
  site plot improved.

* **vmax and specla must move together.** GPP responds to vmax; LAI is set by
  `NPP x aleaf x tauleaf x specla` and cannot be corrected by initialisation
  (seeding decays back within 30 years). Fitting GPP alone drove LAI from
  5.01 to 8.17 in the vmax experiment, so LAI is a constraint here, not an
  afterthought.

* **Stratified global cells, not the tower network**, which is 24% temperate
  conifer against a land surface that is nothing like that.

* **Held-out cells.** A fit that only improves the cells it was fitted on is
  not a calibration.

* **Cells poleward of the exclusion latitude are dropped and declared.**
  Seeded Arctic vegetation collapses back to the `xminlai` floor and restored
  vmax does not reach it either, so leaving those cells in lets the optimiser
  compensate for a structural hole -- which is how `vmax x0.54` arose.

Usage:
  ibis-python verify/calibration/fit_carbon.py --restart <v11 transient> \
      --n-evals 200 --cells 4000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import xarray as xr

REPO = "/shared/nas2/Hangkai/ibis-port"
sys.path.insert(0, f"{REPO}/python-port")
sys.path.insert(0, f"{REPO}/verify/calibration")

import equinox as eqx
import jax
import jax.numpy as jnp

from ibis.constants import PhysicalConstants
from ibis.dimensions import GridConfig
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state
from ibis.timestepping.loops_gswp3 import _cast_tree_f32

from site_transient import one_year


class CachedForcing:
    """A GSWP3Reader that serves a fixed set of years from memory.

    The fit re-runs the SAME few forcing years for every parameter
    evaluation, and reading them back off disk each time dominated the cost:
    18 minutes per evaluation at 4,331 cells, against 53 s per model-year for
    all 95,333 cells in the v11 spin-up. The difference is not compute -- it
    is fancy-indexing a compressed netCDF 12 times a year, 150 times over.

    Exposes only `read_month`, which is all `run_month_gswp3` uses.
    """

    def __init__(self, reader, years, i, j):
        self._c = {}
        for y in years:
            for m in range(12):
                self._c[(y, m)] = reader.read_month(y, m, i, j)
        n = sum(v[next(iter(v))].nbytes for v in self._c.values()) / 1e9
        log.info("cached %d months of forcing (%.2f GB)", len(self._c), n)

    def read_month(self, year, imonth, latindex=None, lonindex=None):
        return self._c[(year, imonth)]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MASK = "/shared/nas2/Min/HWSD/surta.nc"
GSWP3 = "/shared/nas2/Min/GSWP3"
FLUXCOM = "/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc"
MODIS_BASE = "/shared/nas2/Hangkai/ELM Edge Effects/inputdata"
NPOI_FULL = 95333
LATV = np.arange(89.75, -90, -0.5)
LONV = np.arange(-179.75, 180, 0.5)

PFT_NAMES = ("trop_bl_ever", "trop_bl_dec", "warm_bl_ever", "temp_con_ever",
             "temp_bl_dec", "boreal_con_ever", "boreal_bl_dec",
             "boreal_con_dec", "ever_shrub", "dec_shrub", "c4_grass",
             "c3_grass")

# Literature bounds, umol m-2 s-1 (scripts/calibrate_fluxnet.py)
VMAX_LO = np.array([40, 40, 25, 15, 20, 15, 20, 15, 15, 15, 8, 15]) * 1e-6
VMAX_HI = np.array([100, 100, 70, 50, 60, 40, 50, 45, 45, 45, 30, 45]) * 1e-6
SPECLA_LO = np.array([10, 10, 10, 5, 12, 5, 12, 8, 5, 10, 10, 10], float)
SPECLA_HI = np.array([30, 30, 30, 15, 35, 15, 35, 25, 18, 30, 25, 25], float)


# --------------------------------------------------------------- observations
def fluxcom_at(i, j):
    o = xr.open_dataset(FLUXCOM)
    g = np.asarray(o["gpp"].mean(dim="time")) * 365.25 / 1000.0   # -> kgC/m2/yr
    la, lo = np.asarray(o["lat"]), np.asarray(o["lon"])
    ii = np.array([np.argmin(np.abs(la - LATV[a])) for a in i])
    jj = np.array([np.argmin(np.abs(lo - LONV[b])) for b in j])
    return g[ii, jj]


NDAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def fluxcom_monthly_at(i, j):
    """FLUXCOM 1980-2013 monthly climatology as MONTHLY TOTALS, (12, n) kgC m-2.

    FLUXCOM is a daily mean rate per month (g m-2 day-1); multiplied by days
    in month so it is a total, like the model's summed daily GPP. Small
    negative winter values are an ML-product artefact and are clipped at 0.
    """
    o = xr.open_dataset(FLUXCOM)
    clim = np.asarray(o["gpp"].groupby("time.month").mean("time"))     # (12, lat, lon)
    la, lo = np.asarray(o["lat"]), np.asarray(o["lon"])
    ii = np.array([np.argmin(np.abs(la - LATV[a])) for a in i])
    jj = np.array([np.argmin(np.abs(lo - LONV[b])) for b in j])
    c = np.clip(np.nan_to_num(clim[:, ii, jj]), 0.0, None)              # g/m2/day
    return c * NDAYS[:, None] / 1000.0                                   # kgC/m2/month


def seasonal_shape(gpp_m, obs_m):
    """Per-cell seasonal-shape error, scale-free.

    Each cell's 12 monthly totals are divided by its annual total so the
    vector is a fraction-of-year distribution; the error is the summed
    squared difference of those fractions. A perfect cycle scores 0; a
    one-month phase shift of a temperate cycle ~0.05-0.1; a flat model
    against a peaked observation ~0.1-0.2. Independent of magnitude, so it
    does not double-count the total term. Cells with negligible annual GPP
    (either side) return NaN and are excluded by the caller.
    """
    am, ao = gpp_m.sum(axis=0), obs_m.sum(axis=0)
    ok = (am > 0.02) & (ao > 0.02)
    fm = np.where(ok, gpp_m / np.maximum(am, 1e-9), np.nan)
    fo = np.where(ok, obs_m / np.maximum(ao, 1e-9), np.nan)
    return np.nansum((fm - fo) ** 2, axis=0), ok


def peak_offset(gpp_m, obs_m):
    """Circular peak-month offset in months, model minus obs, per cell."""
    d = np.argmax(gpp_m, axis=0) - np.argmax(obs_m, axis=0)
    return ((d + 6) % 12) - 6


def modis_at(i, j):
    ds = xr.open_dataset(f"{MODIS_BASE}/MODISPFTLAI_0.5x0.5_v1_all_forests.nc")
    sd = xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc")
    pv = np.asarray(sd["PCT_NATVEG"].values) / 100.0
    pp = np.asarray(sd["PCT_NAT_PFT"].values) / 100.0
    nt = ds.sizes["time"]
    mon = np.zeros((12,) + pv.shape)
    for m in range(12):
        sel = [t for t in range(nt) if t % 12 == m]
        tot = np.zeros(pv.shape)
        for k in range(1, 17):
            v = f"LAI_{k}"
            if v in ds:
                tot += np.nan_to_num(ds[v].values[sel].mean(axis=0)) * pp[k]
        mon[m] = tot * pv
    g = mon.max(axis=0)
    if float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]):
        g = g[::-1, :]
    return np.nan_to_num(g[i, j])


def tree_at(i, j):
    """CLM tree fraction of the natural-vegetation part of each cell (PFTs 1-8
    of PCT_NAT_PFT), NaN where natural vegetation is under 70 % of the cell
    (ice, lakes). The surfdata is the no-crop `all_forests` variant, so this
    is a potential-vegetation field like IBIS's own -- the fourth criterion,
    the tree-establishment gate's direct target (CALIBRATION.md 14)."""
    sd = xr.open_dataset(f"{MODIS_BASE}/surfdata_0.5x0.5_CH_v1_all_forests.nc")
    pp = np.asarray(sd["PCT_NAT_PFT"].values) / 100.0
    pv = np.asarray(sd["PCT_NATVEG"].values) / 100.0
    t = np.where(pv >= 0.7, pp[1:9].sum(axis=0), np.nan)
    if float(sd["LATIXY"].values[0, 0]) < float(sd["LATIXY"].values[-1, 0]):
        t = t[::-1, :]
    return t[i, j]


ET_ENS = "/shared/nas2/Hangkai/ibis/inputs/eval/et/et_ensemble_0.5deg.npz"


def et_at(i, j):
    """ET observation: five-product ensemble mean (MODIS, MOD16A2, GLEAM v3.3a,
    DOLCE, FLUXCOM LE; 2001-09), mm/yr, NaN unless all five cover the cell --
    coverage differs and the mean elsewhere depends on who covers it
    (CALIBRATION.md 15). Built by verify/calibration/et_obs.py."""
    e = np.load(ET_ENS)
    ann = np.where(e["ens_n"] == 5, e["ens_ann_mean"], np.nan)
    return ann[i, j]


def et_monthly_at(i, j):
    """(12, n) monthly ET climatology of the ensemble mean, mm/month, NaN off
    the all-five cells."""
    e = np.load(ET_ENS)
    clim = e["ens_clim_mean"] * NDAYS[:, None, None]            # mm/day -> mm/month
    clim = np.where((e["ens_n"] == 5)[None], clim, np.nan)
    return clim[:, i, j]


# ------------------------------------------------------------------ parameters
#: Scalar parameters that may join the vector after the 12+12 PFT blocks.
#: name -> (setter(sp, value) -> sp, lo, hi, log-encoded?). Bounds are
#: physically reasoned, not fitted.
#:
#: stressfac: drystress curvature. F77 -5.0; MORE NEGATIVE = LESS stress
#: (tests/test_stomata.py pins the sign). Dead until docs/GPP.md restored it;
#: the natural lever for the within-PFT sparse-cell excess (CALIBRATION.md 10).
#:
#: Phenology (PhenoParams, lifted from literals in pheno() -- CALIBRATION.md):
#: GDD thresholds and ramps log-encoded, temperature offsets raw. `drop_thresh`
#: is deliberately NOT here: it is a threshold on 10-day net assimilation in
#: mol m-2 s-1, O(1e-6) around zero, and fitting a near-zero threshold is
#: ill-conditioned; `drop_floor` and `ddays` carry the drought response.
def _set(path):
    # 0-d float32 ARRAY, not a Python float. loops_gswp3 compiles under
    # eqx.filter_jit, where Python scalars are static: a Python-float extra
    # would recompile the whole model on every evaluation (measured: 1695 s
    # per evaluation instead of ~240). An array leaf is traced once.
    return lambda sp, pft, v: (eqx.tree_at(path, sp, jnp.asarray(v, dtype=jnp.float32)), pft)


def _set_pft(path):
    return lambda sp, pft, v: (sp, eqx.tree_at(path, pft, jnp.asarray(v, dtype=jnp.float32)))


EXTRAS = {
    "stressfac":        (_set(lambda p: p.stomata_params.stressfac), -10.0, -2.5, False),
    # upper bound 1000 (was 250): at the deciduous towers leaf-out is ~30 d early and Stage B needs ~+300 C d to move
    # it 25 d (CALIBRATION.md 22); forcing sums at base 0 C run to several hundred C d without a chilling term
    "gdd_thresh_u":     (_set(lambda p: p.pheno.gdd_thresh_u),        50.0, 1000.0, True),
    "gdd_ramp_u":       (_set(lambda p: p.pheno.gdd_ramp_u),          20.0, 150.0, True),
    "gdd_thresh_l":     (_set(lambda p: p.pheno.gdd_thresh_l),        50.0, 300.0, True),
    "gdd_ramp_l":       (_set(lambda p: p.pheno.gdd_ramp_l),          20.0, 150.0, True),
    "tthresh_u_offset": (_set(lambda p: p.pheno.tthresh_u_offset),     0.0,  12.0, False),
    "gthresh_u_c":      (_set(lambda p: p.pheno.gthresh_u_c),         -5.0,   5.0, False),
    "tthresh_l_c":      (_set(lambda p: p.pheno.tthresh_l_c),         -5.0,   8.0, False),
    "gthresh_l_c":      (_set(lambda p: p.pheno.gthresh_l_c),        -12.0,   2.0, False),
    "drop_floor":       (_set(lambda p: p.pheno.drop_floor),          0.02,   0.5, True),
    "ddays":            (_set(lambda p: p.pheno.ddays),                5.0,  60.0, True),
    # soil-water-stress dormancy trigger; -1 (default) is inert. Raw-encoded.
    "drop_stress_thresh": (_set(lambda p: p.pheno.drop_stress_thresh),  0.05, 0.85, False),
    # absolute upper-canopy leaf-drop temperature (C); -100 (default) never binds
    "tthresh_u_abs":    (_set(lambda p: p.pheno.tthresh_u_abs),        -5.0,  15.0, False),
    # canopy fractional cover (PFTParams): cover_k 0 = F77 linear; floors 0.25 in F77
    "cover_k":          (_set_pft(lambda p: p.cover_k),                 0.2,   2.0, True),
    "fl_min":           (_set_pft(lambda p: p.fl_min),                 0.01,  0.25, True),
    "fu_min":           (_set_pft(lambda p: p.fu_min),                 0.01,  0.25, True),
    # wood-to-cover scale for fu; 1.0 = F77
    "fu_k":             (_set_pft(lambda p: p.fu_k),                    0.5,  10.0, True),
    # water-limited tree establishment: survival = clip((mean stresstu - min)/ramp)
    "tree_stress_min":  (_set_pft(lambda p: p.tree_stress_min),        0.10,  0.90, False),
    "tree_stress_ramp": (_set_pft(lambda p: p.tree_stress_ramp),       0.05,  0.50, True),
}
#: extras that need the StressAccum attached to the state
def _set_idx(path, k):
    """Setter for one element of a vector parameter (traced, dtype-preserving)."""
    def f(sp, pft, v):
        arr = path(sp)
        return eqx.tree_at(path, sp, arr.at[k].set(jnp.asarray(v, dtype=arr.dtype))), pft
    return f


#: Ball-Berry stomatal slope per canopy group (StomataParams.coefm; F77 10, 6,
#: 9, 4, 9 for broadleaf, conifer, shrub, C4 grass, C3 grass). The ET lever:
#: gs = coefm * An * hs / cs + coefb couples transpiration to assimilation, so
#: coefm moves ET (and WUE) at fixed GPP. Log-encoded, literature range.
for _k, _nm in enumerate(("coefm_bl", "coefm_con", "coefm_shrub", "coefm_c4", "coefm_c3")):
    EXTRAS[_nm] = (_set_idx(lambda p: p.stomata_params.coefm, _k), 3.0, 16.0, True)

#: Soil-evaporation control strength (CALIBRATION.md 15): exponent on the
#: wetness beta in turvap; 1 = F77. Log-encoded, [1, 6].
EXTRAS["soil_beta_power"] = (_set(lambda p: p.canopy.soil_beta_power), 1.0, 6.0, True)

#: Respiration constants (NEE stage, CALIBRATION.md 36): Lloyd-Taylor E0 of heterotrophic decomposition (F77 344 K;
#: Lloyd & Taylor 1994 308.56), the WFPS optimum of the moisture factor (F77 60 %), the growth-respiration fraction
#: (F77 0.30). Linear, [150, 600], [35, 85], [0.10, 0.50].
EXTRAS["bgc_e0"] = (_set(lambda p: p.bgc_params.lt_e0), 150.0, 600.0, False)
EXTRAS["wfps_opt"] = (_set(lambda p: p.bgc_params.wfps_opt), 35.0, 85.0, False)
EXTRAS["rgrowth"] = (_set(lambda p: p.bgc_params.rgrowth), 0.10, 0.50, False)
#: Per-site disequilibrium factor on the three SOM pools (csoislon, csoislop, csoipas) at the start of every
#: evaluation; a STATE scaling, applied by site_fit.run, not a parameter: the setter is a no-op. Log, [0.1, 3].
EXTRAS["som_scale"] = ((lambda sp, pft, v: (sp, pft)), 0.1, 3.0, True)

#: Interception (CALIBRATION.md 15): the water-holding capacities of upper
#: leaves / upper stems / lower canopy (kg m-2 per unit area index; F77 0.20,
#: 0.40, 0.20) and the maximum wetted fraction fmax (F77 0.25). Forest
#: interception evaporation is 33-41 % of ET on the (b3) state, the top of
#: the observed 10-30 %. Log-encoded capacities; fmax raw.
EXTRAS["wliqumax"] = (_set(lambda p: p.interception.wliqumax), 0.05, 0.60, True)
EXTRAS["wliqsmax"] = (_set(lambda p: p.interception.wliqsmax), 0.10, 1.20, True)
EXTRAS["wliqlmax"] = (_set(lambda p: p.interception.wliqlmax), 0.05, 0.60, True)
EXTRAS["fmax"] = (_set(lambda p: p.interception.fmax), 0.10, 0.60, False)

#: Temperature acclimation of photosynthetic capacity (CALIBRATION.md 21-22; StomataParams, active only with
#: `stomata: acclim_on: true` in the yaml): time constant (d), onset (C) and range to full capacity (C) of the
#: Makela (2004) / PRELES state of acclimation. PRELES 10-site conifer values 10.93, -3.063, 17.72.
EXTRAS["acclim_tau"] = (_set(lambda p: p.stomata_params.acclim_tau), 2.0, 30.0, True)
EXTRAS["acclim_s0"] = (_set(lambda p: p.stomata_params.acclim_s0), -10.0, 5.0, False)
EXTRAS["acclim_smax"] = (_set(lambda p: p.stomata_params.acclim_smax), 5.0, 30.0, True)
#: Capacity-profile coefficient of the canopy scaling (CALIBRATION.md 26; active with `stomata: cap_scale_u/_l`).
#: Lloyd et al. (2010): 0.13-0.23 for observed Vcmax; log-encoded, [0.05, 1.0].
EXTRAS["cap_kn"] = (_set(lambda p: p.stomata_params.cap_kn), 0.05, 1.0, True)
#: Photoperiod-gated senescence (Delpierre 2009 structure; PhenoParams, active only with
#: `pheno: sen_photoperiod_u / _l: true`): critical day length (h), base temperature (C), critical sum (C d).
for _c in ("u", "l"):
    EXTRAS[f"sen_pstart_{_c}"] = (_set(eval(f"lambda p: p.pheno.sen_pstart_{_c}")), 11.0, 16.0, False)
    EXTRAS[f"sen_tb_{_c}"] = (_set(eval(f"lambda p: p.pheno.sen_tb_{_c}")), 10.0, 28.0, False)
    EXTRAS[f"sen_ycrit_{_c}"] = (_set(eval(f"lambda p: p.pheno.sen_ycrit_{_c}")), 30.0, 1500.0, True)

NEEDS_STRESS_ACCUM = ("tree_stress_min", "tree_stress_ramp")
NPFT = 12


def _enc(name, v):
    return np.log(v) if EXTRAS[name][3] else float(v)


def _dec(name, x):
    return float(np.exp(x)) if EXTRAS[name][3] else float(x)


def extras_defaults(sp, names, pft=None):
    """Starting value of each extra, read from wherever it lives (the same
    routing the EXTRAS setters and finalize_fit.sh use)."""
    out = {}
    for n in names:
        out[n] = float(_extra_value(sp, pft, n))
    return out


#: Where each extra lives. Anything not listed is a PhenoParams field.
PFT_SCALARS = ("cover_k", "fl_min", "fu_min", "fu_k", "tree_stress_min", "tree_stress_ramp")
COEFM_GROUPS = {"coefm_bl": 0, "coefm_con": 1, "coefm_shrub": 2, "coefm_c4": 3, "coefm_c3": 4}
INTERCEPTION_FIELDS = ("wliqumax", "wliqsmax", "wliqlmax", "fmax")
CANOPY_FIELDS = ("soil_beta_power",)
BGC_FIELDS = {"bgc_e0": "lt_e0", "wfps_opt": "wfps_opt", "rgrowth": "rgrowth"}
STOMATA_FIELDS = ("acclim_tau", "acclim_s0", "acclim_smax", "cap_kn")
PFT_ARRAYS = {"coefm": (lambda p: p.stomata_params.coefm_pft, 3.0, 16.0, True), "acclim_s0": (lambda p: p.stomata_params.acclim_s0_pft, -10.0, 5.0, False),
              "acclim_smax": (lambda p: p.stomata_params.acclim_smax_pft, 5.0, 30.0, True)}          # per-PFT extras '<name>:<pft>' (CALIBRATION.md 32)
for _k, _nm in enumerate(PFT_NAMES):
    for _base, (_path, _lo, _hi, _log) in PFT_ARRAYS.items():
        EXTRAS[f"{_base}:{_nm}"] = (_set_idx(_path, _k), _lo, _hi, _log)


def _extra_value(sp, pft, n):
    if ":" in n and n.split(":")[0] in PFT_ARRAYS:
        base, pf = n.split(":"); return PFT_ARRAYS[base][0](sp)[PFT_NAMES.index(pf)]
    if n == "stressfac":
        return sp.stomata_params.stressfac
    if n in BGC_FIELDS:
        return getattr(sp.bgc_params, BGC_FIELDS[n])
    if n == "som_scale":
        return 1.0
    if n in PFT_SCALARS:
        return getattr(pft, n)
    if n in COEFM_GROUPS:
        return sp.stomata_params.coefm[COEFM_GROUPS[n]]
    if n in INTERCEPTION_FIELDS:
        return getattr(sp.interception, n)
    if n in CANOPY_FIELDS:
        return getattr(sp.canopy, n)
    if n in STOMATA_FIELDS:
        return getattr(sp.stomata_params, n)
    return getattr(sp.pheno, n)

def to_vec(vmax, specla, extras=None, names=()):
    parts = [np.log(vmax), np.log(specla)]
    if names:
        parts.append(np.array([_enc(n, extras[n]) for n in names], dtype=float))
    return np.concatenate(parts)


def from_vec(x, names=()):
    vmax, specla = np.exp(x[:NPFT]), np.exp(x[NPFT:2 * NPFT])
    extras = {n: _dec(n, x[2 * NPFT + k]) for k, n in enumerate(names)}
    return vmax, specla, extras


def extras_bounds(names):
    lo = np.array([_enc(n, EXTRAS[n][1]) for n in names], dtype=float)
    hi = np.array([_enc(n, EXTRAS[n][2]) for n in names], dtype=float)
    return lo, hi


def extras_in_bounds(extras, tol=1e-9):
    for n, v in extras.items():
        lo, hi = EXTRAS[n][1], EXTRAS[n][2]
        if not (lo - tol * abs(lo) - 1e-12 <= v <= hi + tol * abs(hi) + 1e-12):
            return False
    return True


def apply_params(sp, pft, vmax, specla, extras=None):
    sp = eqx.tree_at(lambda p: p.stomata_params.vmax_pft, sp,
                     jnp.array(vmax, dtype=jnp.float32))
    for n, v in (extras or {}).items():
        sp, pft = EXTRAS[n][0](sp, pft, v)
    pft = eqx.tree_at(lambda p: p.specla, pft,
                      jnp.array(specla, dtype=jnp.float32))
    return sp, pft


def enable_compile_cache():
    """Persistent XLA compilation cache: the same program (same cells, same yaml switches, traced extras) is compiled
    once and loaded by every later process -- a full set of month kernels takes ~30 min to compile on the GPU for
    17 towers (~2.5 min on the CPU). Location `IBIS_JAX_CACHE`, default ~/.cache/jax_ibis. Call before the first run."""
    jax.config.update('jax_compilation_cache_dir', os.environ.get('IBIS_JAX_CACHE', os.path.expanduser('~/.cache/jax_ibis')))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)


def trace_extras(sp, pft):
    """Every EXTRAS parameter as a 0-d float32 array at its current value.

    `_set` makes a fitted extra an array, but the parameters as built are Python floats, and eqx.filter_jit keys
    its cache on which leaves are arrays: the first run with a new set of array leaves recompiles every month
    kernel (up to 8; about 35 min for the set on the GPU for 17 towers). A spin-up on the built parameters followed
    by fitted evaluations compiled twice, and a sweep compiled once per perturbed extra. Converting all of them
    once gives every run the same pytree. Values are unchanged (float32 rounding of the same numbers)."""
    return apply_params(sp, pft, np.asarray(sp.stomata_params.vmax_pft), np.asarray(pft.specla),
                        {n: float(_extra_value(sp, pft, n)) for n in EXTRAS})


def bounds_penalty(vmax, specla):
    return float(
        np.sum(np.maximum(0, np.log(VMAX_LO / vmax)) ** 2) +
        np.sum(np.maximum(0, np.log(vmax / VMAX_HI)) ** 2) +
        np.sum(np.maximum(0, np.log(SPECLA_LO / specla)) ** 2) +
        np.sum(np.maximum(0, np.log(specla / SPECLA_HI)) ** 2))


# ------------------------------------------------------------------------ loss
ET_MIN_RATIO = 0.2          # model annual ET below this fraction of observed = anomaly, excluded from the ET terms
ET_EXCLUDED = [0]           # cells excluded by ET_MIN_RATIO at the last score() call (for the report)


def score(gpp, lai, gpp_obs, lai_obs, w_lai, area=None, w_total=1.0,
          gpp_m=None, obs_m=None, w_season=0.0,
          fu=None, tree_obs=None, w_tree=0.0,
          et=None, et_obs=None, w_et=0.0, et_m=None, et_obs_m=None, w_et_season=0.0):
    """Magnitude-honest loss. Returns (total, gpp_term, lai_term).

    The first fit used an unweighted per-cell mean of squared log-ratios and
    reported GPP 0.83x FLUXCOM at the sample while the full domain came out
    ~1.35x (docs/CALIBRATION.md 9). A sparse cell at 0.5x counted as much as
    a rainforest cell at 1.5x, so the optimiser satisfied the many low-flux
    cells and left the few that carry the global total high.

    Two changes. (1) Each cell's GPP log-ratio is weighted by its share of the
    observed flux, area x GPP_obs -- i.e. by what it contributes to the total
    being judged. LAI is weighted by area only, since LAI comparisons are
    area means. (2) An explicit term on the log-ratio of area-integrated
    totals, so the sum itself is a target and not just an emergent property.

    Three criteria, one loss. The GPP term carries the GLOBAL TOTAL (its
    total sub-term) and the SPATIAL PATTERN (its per-cell sub-term). The
    optional SEASONAL term is `seasonal_shape` -- each cell's 12-month GPP as
    fractions of its annual total against FLUXCOM's -- flux-weighted like
    the spatial term. It is scale-free, so it adds no second vote on
    magnitude.

    Optional FOURTH criterion (`w_tree`): upper-canopy cover `fu` against the
    CLM tree fraction, area-weighted mean |fu - tree| over cells with natural
    vegetation (`tree_obs` NaN elsewhere). Absolute, not log: the observation
    is zero on a third of the land. Without it a joint fit can satisfy GPP
    through vmax instead of through the trees -- the degeneracy behind the
    phantom trees (CALIBRATION.md 14).
    Optional ET terms (Stage 2, CALIBRATION.md 15): `w_et` weights the ET
    analogue of the GPP term -- ET-weighted per-cell log-ratio plus the
    log-ratio of area totals -- against the five-product ensemble mean on
    all-five cells (`et_obs` NaN elsewhere); `w_et_season` weights the ET
    seasonal shape. Returns (total, gpp_term, lai_term, season_term,
    tree_term, et_term, et_season_term).
    """
    ok = np.isfinite(gpp) & np.isfinite(lai) & (gpp_obs > 0.02) & (lai_obs > 0.05)
    if ok.sum() < 20:
        return 50.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    a = np.ones(ok.sum()) if area is None else np.asarray(area)[ok]
    g_m, g_o = np.maximum(gpp[ok], 1e-4), gpp_obs[ok]
    l_m, l_o = np.maximum(lai[ok], 1e-3), lai_obs[ok]

    wg = a * g_o; wg = wg / wg.sum()
    wl = a / a.sum()
    g_cell = float(np.sum(wg * np.log(g_m / g_o) ** 2))
    l_cell = float(np.sum(wl * np.log(l_m / l_o) ** 2))
    g_tot = float(np.log(np.sum(a * g_m) / np.sum(a * g_o)) ** 2)
    l_tot = float(np.log(np.sum(a * l_m) / np.sum(a * l_o)) ** 2)

    g_term = g_cell + w_total * g_tot
    l_term = l_cell + w_total * l_tot

    s_term = 0.0
    if gpp_m is not None and obs_m is not None and w_season > 0:
        err, sok = seasonal_shape(gpp_m[:, ok], obs_m[:, ok])
        if sok.sum() >= 20:
            ws = (a * g_o) * sok; ws = ws / ws.sum()
            s_term = float(np.sum(ws * np.nan_to_num(err)))
    t_term = 0.0
    if fu is not None and tree_obs is not None and w_tree > 0:
        tk = np.isfinite(tree_obs[ok]) & np.isfinite(fu[ok])
        if tk.sum() >= 20:
            wt = a * tk; wt = wt / wt.sum()
            t_term = float(np.sum(wt * np.abs(np.nan_to_num(fu[ok])
                                              - np.nan_to_num(tree_obs[ok]))))
    # ET terms. Model monthly ET can be NEGATIVE (net deposition) and on a
    # few subtropical cells reaches -80..-146 mm/month -- a model anomaly
    # logged in DEFECTS_FOR_REVIEW.md, not a shape to fit. Unguarded, those
    # cells gave the first joint-fit evaluation an ET-shape term 46x the GPP
    # one and a per-cell term of 1.1 (median cell 0.005). So: months clipped
    # at zero for the shape, and both ET terms restricted to cells whose
    # model annual ET is at least a fifth of the observed (ET_MIN_RATIO);
    # score() returns the excluded count through ET_EXCLUDED for the report.
    e_term = 0.0
    if et is not None and et_obs is not None and w_et > 0:
        ek = np.isfinite(et_obs[ok]) & np.isfinite(et[ok]) & (et_obs[ok] > 10.0)
        anom = ek & ~(et[ok] >= ET_MIN_RATIO * et_obs[ok])
        ET_EXCLUDED[0] = int(anom.sum()); ek = ek & ~anom
        if ek.sum() >= 20:
            e_m, e_o, ae = et[ok][ek], et_obs[ok][ek], a[ek]
            we = ae * e_o; we = we / we.sum()
            e_cell = float(np.sum(we * np.log(e_m / e_o) ** 2))
            e_tot = float(np.log(np.sum(ae * e_m) / np.sum(ae * e_o)) ** 2)
            e_term = e_cell + w_total * e_tot
    es_term = 0.0
    if et_m is not None and et_obs_m is not None and w_et_season > 0:
        etm_c = np.maximum(et_m[:, ok], 0.0)
        ek = np.isfinite(et_obs_m[:, ok]).all(axis=0) & np.isfinite(etm_c).all(axis=0)
        if et is not None and et_obs is not None:
            ek = ek & np.isfinite(et_obs[ok]) & (et[ok] >= ET_MIN_RATIO * et_obs[ok])
        if ek.sum() >= 20:
            err, sok = seasonal_shape(etm_c[:, ek], et_obs_m[:, ok][:, ek])
            if sok.sum() >= 20:
                we = (a[ek] * np.nansum(et_obs_m[:, ok][:, ek], axis=0)) * sok; we = we / we.sum()
                es_term = float(np.sum(we * np.nan_to_num(err)))
    return (g_term + w_lai * l_term + w_season * s_term + w_tree * t_term
            + w_et * e_term + w_et_season * es_term,
            g_term, l_term, s_term, t_term, e_term, es_term)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", required=True)
    ap.add_argument("--cells", type=int, default=4000)
    ap.add_argument("--n-evals", type=int, default=200)
    ap.add_argument("--adjust-years", type=int, default=10)
    ap.add_argument("--eval-years", type=int, default=3)
    ap.add_argument("--year0", type=int, default=2001)
    ap.add_argument("--force-years", type=int, default=10,
                    help="distinct forcing years cycled within an evaluation. "
                         "3 (2001-03) biased v11d low: that triplet holds the 2002 "
                         "US drought and 2003 European heat wave -- CALIBRATION.md 10")
    ap.add_argument("--exclude-poleward", type=float, default=68.0)
    ap.add_argument("--w-lai", type=float, default=0.5)
    ap.add_argument("--w-total", type=float, default=1.0,
                    help="weight on the log-ratio of area-integrated totals inside the GPP and LAI "
                         "terms. At 1 a 15 %% global bias costs ln(1.15)^2 = 0.02 against a per-cell "
                         "term of ~0.2, so criterion 1 was never pulled on (CALIBRATION.md 18); 25 "
                         "makes a 10 %% bias cost as much as the pattern")
    ap.add_argument("--representative", type=int, default=0,
                    help="1: weight each sampled cell by domain/sample stratum counts so the "
                         "loss terms estimate the DOMAIN totals and patterns (CALIBRATION.md 18); "
                         "0: plain area weights as in stages a-b3")
    ap.add_argument("--w-et", type=float, default=0.0,
                    help="Stage 2: weight on the ET term (five-product ensemble mean, "
                         "all-five cells; CALIBRATION.md 15); 0 = off")
    ap.add_argument("--w-et-season", type=float, default=0.0,
                    help="Stage 2: weight on the ET seasonal-shape term; 0 = off")
    ap.add_argument("--w-tree", type=float, default=0.0,
                    help="weight on the fourth criterion: area-weighted mean |fu - CLM "
                         "tree fraction| over natural-vegetation cells (CALIBRATION.md 14); "
                         "0 = off (stage a)")
    ap.add_argument("--init-params", default="",
                    help="params yaml to start from (e.g. the v11d fit); "
                         "default IBIS defaults")
    ap.add_argument("--freeze-pft", type=int, default=0,
                    help="1: hold vmax/specla at their starting values and fit "
                         "ONLY the extras -- the PFT blocks leave the CMA vector")
    ap.add_argument("--extras", default="stressfac",
                    help="comma list of scalar extras to fit (see EXTRAS); '' for none. "
                         "Default fits stressfac only; add phenology fields once the "
                         "seasonal baseline says which matter")
    ap.add_argument("--w-season", type=float, default=10.0,
                    help="weight on the seasonal-shape term. Flux-weighted shape "
                         "errors are O(0.02-0.04) against a GPP term of O(0.3-0.4) "
                         "(CALIBRATION.md 13); 10 makes the third criterion carry "
                         "comparable weight rather than ~5% of the loss")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--min-cells", type=int, default=30,
                    help="PFTs dominant in fewer cells are frozen at default")
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--out", default=f"{REPO}/verify_out_stage1/fit_carbon")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import cma
    from sample_cells import build_sample

    sel, lat, dom, w_rep = build_sample(a.cells, a.restart, return_weights=True)
    keep = np.abs(lat) < a.exclude_poleward
    log.info("sample %d cells; %d kept after |lat| < %.0f exclusion (%d dropped)",
             len(sel), keep.sum(), a.exclude_poleward, (~keep).sum())
    sel, lat, dom, w_rep = sel[keep], lat[keep], dom[keep], w_rep[keep]

    m = np.array(xr.open_dataset(MASK).landfrac.values)
    fl = np.where(m.ravel() > 0)[0]
    i = (fl[sel] // 720).astype(np.int32)
    j = (fl[sel] % 720).astype(np.int32)
    lat_rad = np.deg2rad(LATV[i]).astype(np.float32)

    gpp_obs = fluxcom_at(i, j)
    lai_obs = modis_at(i, j)
    tree_obs = tree_at(i, j)                    # CLM tree fraction, NaN off natural veg
    et_obs = et_at(i, j)                        # ET ensemble mean, NaN off all-five cells
    et_obs_m = et_monthly_at(i, j)              # (12, n) mm/month
    log.info("ET ensemble at sample: %.0f mm/yr on %d all-five cells",
             np.nanmean(et_obs), np.isfinite(et_obs).sum())
    obs_m = fluxcom_monthly_at(i, j)            # (12, n) monthly totals
    from landmask import true_landfrac
    area = np.cos(np.deg2rad(LATV[i])) * true_landfrac()[i, j]   # relative LAND area: the binary mask made partial cells full (CALIBRATION.md 18)
    log.info("cell weights carry the true land fraction: mean %.2f over the sample, %d cells under 0.5",
             float(np.mean(true_landfrac()[i, j])), int((true_landfrac()[i, j] < 0.5).sum()))
    if a.representative:
        # Inverse-probability weights: the stratified sample over-samples
        # rare strata (floor of 20 cells), so its plain-area totals are not
        # the domain's -- v12e read 0.95x on the sample against 1.31x on the
        # domain (CALIBRATION.md 18). Weighting by domain/sample stratum
        # counts makes every term a domain estimate.
        area = area * w_rep
        log.info("REPRESENTATIVE weights on: stratum weights min/median/max %.1f/%.1f/%.1f, "
                 "effective sample size %.0f of %d",
                 w_rep.min(), np.median(w_rep), w_rep.max(),
                 w_rep.sum() ** 2 / np.sum(w_rep ** 2), len(w_rep))
    else:
        log.info("plain area weights (sample-representative weighting OFF)")
    log.info("obs at sample: FLUXCOM GPP %.3f  MODIS LAI %.2f  CLM tree %.2f (%d cells)",
             np.nanmean(gpp_obs), np.nanmean(lai_obs), np.nanmean(tree_obs),
             np.isfinite(tree_obs).sum())

    counts = np.bincount(dom[dom >= 0], minlength=12)
    free = counts >= a.min_cells
    log.info("PFTs free to move (>= %d dominant cells): %s", a.min_cells,
             [PFT_NAMES[k] for k in range(12) if free[k]])
    log.info("PFTs FROZEN at their STARTING values (too few cells to identify): %s",
             [f"{PFT_NAMES[k]}({counts[k]})" for k in range(12) if not free[k]])

    rng = np.random.default_rng(a.seed)
    fit_mask = rng.random(len(sel)) > a.holdout
    log.info("fit on %d cells, hold out %d", fit_mask.sum(), (~fit_mask).sum())

    full, _ = load_state(a.restart, create_ibis_state(
        GridConfig(npoi=NPOI_FULL, nbedrock=3)))

    def cut(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL:
            return x[sel]
        return x
    state0 = _cast_tree_f32(jax.tree.map(cut, full))
    state0 = jax.tree.map(
        lambda x: jnp.nan_to_num(x, nan=0.0) if isinstance(x, jnp.ndarray)
        and jnp.issubdtype(x.dtype, jnp.floating) else x, state0)
    # The accumulator must exist whenever the gate can act: when its scalars
    # are in the vector, OR when the start yaml already activates it
    # (pft.tree_stress_min >= 0) with the gate held fixed. Attaching only on
    # the former would run a fixed gate gateless -- the production-path hole
    # of CALIBRATION.md 14 inside the harness.
    gate_fitted = any(n in NEEDS_STRESS_ACCUM for n in a.extras.split(","))
    _pd_gate = (load_physics_params(a.init_params) if a.init_params else None) or {}
    _tsm0 = float((_pd_gate.get("pft") or {}).get("tree_stress_min", -1.0))
    gate_fixed = _tsm0 >= 0.0
    if gate_fitted or gate_fixed:
        from ibis.state.stress_accum import create_stress_accum
        state0 = eqx.tree_at(lambda s: s.stress, state0, create_stress_accum(len(sel)),
                             is_leaf=lambda x: x is None)
        log.info("StressAccum attached: tree-establishment gate %s (start tree_stress_min=%.3f)",
                 "fitted" if gate_fitted else "FIXED from the start yaml", _tsm0)
    else:
        log.info("no StressAccum: tree-establishment gate inactive")
    # CO2 is NOT set here: `one_year` calls apply_co2(state, year) itself, so
    # it follows the evaluation year (2001-2005 -> 371-380 ppm). Setting it
    # here as well would have been a flag that parses and does nothing, which
    # is the defect this project has now found fourteen times.
    del full

    pdict0 = load_physics_params(a.init_params) if a.init_params else None
    log.info("starting parameters: %s", a.init_params or "IBIS defaults")
    sp0, pft0, _ = build_all_params(len(sel), lat_rad, pdict0)
    years = list(range(a.year0, a.year0 + a.force_years))
    cached = CachedForcing(GSWP3Reader(GSWP3), years, i, j)
    ctx0 = dict(i=i, j=j, lat_rad=lat_rad, sp=sp0, pft=pft0,
                consts=PhysicalConstants(), reader=cached)
    v0 = np.asarray(sp0.stomata_params.vmax_pft, dtype=np.float64)
    s0 = np.asarray(pft0.specla, dtype=np.float64)
    log.info("start vmax (umol): %s", np.round(v0 * 1e6, 1).tolist())
    log.info("start specla:      %s", np.round(s0, 1).tolist())

    hist = []
    best = [np.inf, None]
    ev = [0]

    def run(vmax, specla, extras=None):
        sp, pft = apply_params(sp0, pft0, vmax, specla, extras)
        ctx = dict(ctx0, sp=sp, pft=pft)
        st = state0
        key = jax.random.PRNGKey(0)
        for n in range(a.adjust_years):
            st, key, _ = one_year(st, key, years[n % len(years)], ctx, 1)
        gs, ls, ms, fs, es, ems = [], [], [], [], [], []
        for n in range(a.eval_years):
            st, key, d = one_year(st, key, years[n % len(years)], ctx, 1)
            gs.append(d["aygpptot"]); ls.append(d["lai"]); ms.append(d["gpp_monthly"])
            fs.append(np.asarray(st.vegetation.fu))
            es.append(d["ayaet"]); ems.append(d["aet_monthly"])
        return (np.mean(gs, axis=0), np.mean(ls, axis=0), np.mean(ms, axis=0),
                np.mean(fs, axis=0), np.mean(es, axis=0), np.mean(ems, axis=0))

    def objective(x):
        t0 = time.time()
        if a.freeze_pft:
            # x carries ONLY the extras; the PFT blocks are fixed at the start.
            x = np.concatenate([to_vec(v0, s0), np.asarray(x, dtype=float)])
        vmax, specla, extras = from_vec(x, names)
        # A parameter the sample cannot constrain must not move: with zero
        # dominant cells the loss is flat in it, so CMA-ES walks it to a
        # bound on noise. The first run did exactly that -- boreal_con_dec
        # (0 cells) went to 57 umol against a 45 ceiling.
        vmax = np.where(free, vmax, v0)
        specla = np.where(free, specla, s0)
        if not extras_in_bounds(extras):
            return 100.0
        # Tolerance matters: `specla` defaults sit EXACTLY on two bounds
        # (30.0 and 35.0), and the log/exp round-trip in to_vec/from_vec puts
        # them a float above, which rejected the baseline itself as
        # out-of-bounds and reported loss 100.
        tol = 1.0 + 1e-9
        if np.any(vmax < VMAX_LO / tol) or np.any(vmax > VMAX_HI * tol) \
                or np.any(specla < SPECLA_LO / tol) \
                or np.any(specla > SPECLA_HI * tol):
            return 100.0
        try:
            gpp, lai, gpp_m, fu_m, et_a, et_mm = run(vmax, specla, extras)
        except Exception as e:                       # noqa: BLE001
            log.warning("  eval %d failed: %s", ev[0], str(e)[:90])
            return 50.0
        def _score(msk):
            return score(gpp[msk], lai[msk], gpp_obs[msk], lai_obs[msk], a.w_lai,
                         area[msk], gpp_m=gpp_m[:, msk], obs_m=obs_m[:, msk],
                         w_season=a.w_season, fu=fu_m[msk], tree_obs=tree_obs[msk],
                         w_tree=a.w_tree, et=et_a[msk], et_obs=et_obs[msk], w_et=a.w_et,
                         w_total=a.w_total,
                         et_m=et_mm[:, msk], et_obs_m=et_obs_m[:, msk],
                         w_et_season=a.w_et_season)
        tot, gt, lt, st_, tt, et_, ets_ = _score(fit_mask)
        ho = _score(~fit_mask)[0]
        ev[0] += 1
        hist.append(dict(eval=ev[0], loss=tot, gpp_term=gt, lai_term=lt,
                         season_term=st_, tree_term=tt, et_term=et_, et_season_term=ets_,
                         holdout=ho,
                         gpp_mean=float(np.nanmean(gpp)),
                         lai_mean=float(np.nanmean(lai))))
        if tot < best[0]:
            best[0], best[1] = tot, (vmax.copy(), specla.copy(), dict(extras))
            log.info("  eval %3d  loss=%.4f (BEST)  gpp=%.4f lai=%.4f season=%.4f "
                     "tree=%.3f et=%.4f etseas=%.4f holdout=%.4f  GPP=%.3f LAI=%.2f  [%.0fs]",
                     ev[0], tot, gt, lt, st_, tt, et_, ets_, ho, hist[-1]["gpp_mean"],
                     hist[-1]["lai_mean"], time.time() - t0)
        elif ev[0] % 10 == 0:
            log.info("  eval %3d  loss=%.4f  holdout=%.4f  [%.0fs]",
                     ev[0], tot, ho, time.time() - t0)
        return tot

    names = tuple(n for n in a.extras.split(",") if n)
    for n in names:
        assert n in EXTRAS, f"unknown extra {n}; choose from {list(EXTRAS)}"
    ex0 = extras_defaults(sp0, names, pft0)
    log.info("extras fitted: %s", {n: (ex0[n], EXTRAS[n][1], EXTRAS[n][2]) for n in names} or "none")
    elo, ehi = extras_bounds(names)
    ex_x0 = np.array([_enc(n, ex0[n]) for n in names], dtype=float)
    if a.freeze_pft:
        assert names, "--freeze-pft with no --extras leaves nothing to fit"
        log.info("PFT blocks FROZEN at start values; CMA vector = %d extras", len(names))
        lo_v, hi_v = elo, ehi
        x0 = np.clip(ex_x0, lo_v, hi_v)
        off = 0
    else:
        lo_v = np.concatenate([to_vec(VMAX_LO, SPECLA_LO), elo])
        hi_v = np.concatenate([to_vec(VMAX_HI, SPECLA_HI), ehi])
        x0 = np.clip(to_vec(v0, s0, ex0, names), lo_v, hi_v)
        off = 2 * NPFT
    log.info("=== baseline (starting parameters) ===")
    b = objective(x0)
    log.info("baseline loss %.4f", b)
    # CMA-ES uses one sigma across coordinates. log-encoded coordinates span
    # ~1; raw-encoded extras span (hi-lo), so scale those to a comparable width.
    scales = np.ones(len(x0))
    for k, n in enumerate(names):
        if not EXTRAS[n][3]:
            scales[off + k] = (EXTRAS[n][2] - EXTRAS[n][1]) / 1.5
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {
        "maxfevals": a.n_evals, "seed": a.seed, "verb_disp": 0,
        "CMA_stds": scales,
        "bounds": [lo_v, hi_v]})
    while not es.stop() and ev[0] < a.n_evals:
        xs = es.ask()
        es.tell(xs, [objective(x) for x in xs])

    vmax, specla, extras = best[1]
    import pandas as pd, json
    pd.DataFrame(hist).to_csv(f"{a.out}/history.csv", index=False)
    np.savez(f"{a.out}/best.npz", vmax=vmax, specla=specla, sel=sel,
             extras_json=json.dumps(extras))
    if extras:
        print("\nEXTRAS (default -> fitted  [lo, hi]):")
        for n in names:
            print(f"  {n:<18} {ex0[n]:9.3f} -> {extras[n]:9.3f}   [{EXTRAS[n][1]}, {EXTRAS[n][2]}]")
    print(f"\n{'='*76}\nBEST PARAMETERS (loss {best[0]:.4f}, baseline {b:.4f})")
    print(f"{'PFT':<18}{'vmax0':>8}{'vmax':>8}{'lo':>6}{'hi':>6}"
          f"{'specla0':>9}{'specla':>8}")
    for k, n in enumerate(PFT_NAMES):
        print(f"{n:<18}{v0[k]*1e6:>8.1f}{vmax[k]*1e6:>8.1f}"
              f"{VMAX_LO[k]*1e6:>6.0f}{VMAX_HI[k]*1e6:>6.0f}"
              f"{s0[k]:>9.1f}{specla[k]:>8.1f}")
    gpp, lai, gpp_m, fu_m, et_a, et_mm = run(vmax, specla, extras)
    for nm, msk in (("fit", fit_mask), ("HELD OUT", ~fit_mask)):
        t, g, l, s_, tt_, et_, ets_ = score(
            gpp[msk], lai[msk], gpp_obs[msk], lai_obs[msk], a.w_lai,
            area[msk], gpp_m=gpp_m[:, msk], obs_m=obs_m[:, msk],
            w_season=a.w_season, fu=fu_m[msk], tree_obs=tree_obs[msk], w_tree=a.w_tree,
            et=et_a[msk], et_obs=et_obs[msk], w_et=a.w_et, w_total=a.w_total,
            et_m=et_mm[:, msk], et_obs_m=et_obs_m[:, msk], w_et_season=a.w_et_season)
        off = peak_offset(gpp_m[:, msk], obs_m[:, msk])
        # NaN-aware area weighting for the report only (one NaN cell poisons
        # np.average; score() masks it): weights are zeroed where x is NaN.
        def wavg(x, w):
            f = np.isfinite(x); return float(np.sum(np.where(f, x, 0.0) * w * f) / np.sum(w * f))
        print(f"  {nm:>9}: loss {t:.4f}  GPP term {g:.4f}  LAI term {l:.4f}  "
              f"season term {s_:.4f}  tree term {tt_:.3f}  ET term {et_:.4f} ETseas {ets_:.4f}  "
              f"peak offset mean {np.mean(off):+.2f} mo  "
              f"area-wtd GPP {wavg(gpp[msk], area[msk]):.3f} vs "
              f"{wavg(gpp_obs[msk], area[msk]):.3f}"
              f"  LAI {wavg(lai[msk], area[msk]):.2f} vs "
              f"{wavg(lai_obs[msk], area[msk]):.2f}"
              f"  fu {np.nanmean(fu_m[msk]):.2f} vs tree {np.nanmean(tree_obs[msk]):.2f}"
              f"  ET {wavg(et_a[msk], area[msk]):.0f} vs {wavg(et_obs[msk], area[msk]):.0f} mm/yr")
        # The comparable total: model and observation on the SAME cells the
        # loss scores (gpp_obs > 0.02, lai_obs > 0.05, both finite). The
        # line above averages the model over every fit cell and the
        # observation over cells with data -- it read 0.95x where the
        # domain read 1.16x (CALIBRATION.md 18).
        okm = msk & np.isfinite(gpp) & np.isfinite(gpp_obs) & (gpp_obs > 0.02) & (lai_obs > 0.05)
        print(f"  {nm:>9}: ET terms excluded {ET_EXCLUDED[0]} anomalous cells (model annual ET < {ET_MIN_RATIO:.1f} x observed)")
        print(f"  {nm:>9}: on the {okm.sum()} scored cells, area-weighted GPP model/obs = "
              f"{wavg(gpp[okm], area[okm]) / wavg(gpp_obs[okm], area[okm]):.3f}"
              f"  (ET model/obs on all-five cells = "
              f"{wavg(et_a[okm & np.isfinite(et_obs)], area[okm & np.isfinite(et_obs)]) / max(wavg(et_obs[okm & np.isfinite(et_obs)], area[okm & np.isfinite(et_obs)]), 1e-9):.3f})")
    print(f"\nWrote {a.out}/")


if __name__ == "__main__":
    main()
