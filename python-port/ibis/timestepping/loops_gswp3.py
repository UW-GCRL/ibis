"""GSWP3 transient-forcing time loops for IBIS-JAX.

Replaces the climatology-based run_year/run_month with a version that reads
actual daily GSWP3 observations and overrides cloud-based fira with observed
downwelling longwave radiation (rlds).

Architecture
------------
run_year_gswp3
  └─ run_month_gswp3  (Python loop over 12 months; reads one month from disk)
       └─ jax.lax.scan over days  (jday is a dynamic JAX scalar — compiled ONCE)
            └─ jax.lax.scan over timesteps  (nested; runs full physics)

Performance note
----------------
Both day and timestep loops use jax.lax.scan, so the entire month compiles to
a single GPU program. The Python boundary is crossed only 12 times per year
(once per month, for disk I/O), not 365 times. This avoids daily recompilation
that would occur if jday were a Python int captured in a scan closure.

LW override rationale
---------------------
GSWP3 provides daily-mean rlds (W m-2). The diurnal() function in weather.py
computes fira from cloud fraction and temperature using Idso (1981), which can
differ substantially from the observed value. Since downwelling LW varies
little over the diurnal cycle (<10 W m-2 amplitude), using the daily mean as a
constant for all timesteps is a standard approximation.
"""

from __future__ import annotations

import functools
import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from ibis.constants import PhysicalConstants
from ibis.state.model_state import IBISState
from ibis.timestepping.loops import SimParams, timestep_body, timesteps_per_day
from ibis.timestepping.calendar import (
    NDAYPM, NDAYPM_LEAP, julian_day, is_leap_year, timesteps_per_day,
)
from ibis.io.gswp3_reader import GSWP3Reader
from ibis.io.gswp3_forcing import make_monthly_daily_weather
from ibis.physics.weather import DailyWeather, compute_precip_timing, diurnal
from ibis.physics.phenology import pheno
from ibis.state.seasonal import daylength_hours
from ibis.physics.biogeochem import soilbgc

logger = logging.getLogger(__name__)


def _cast_tree_f32(tree):
    """Cast all float arrays in a pytree to float32.

    This prevents float64 promotion from leaking through lax.scan carry,
    which requires input and output dtypes to match exactly.
    """
    def _cast_leaf(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float32)
        return x
    return jax.tree.map(_cast_leaf, tree)


# ---------------------------------------------------------------------------
# _run_day_inner: single day body, called from the day-level scan
# ---------------------------------------------------------------------------

def _run_day_inner(
    state: IBISState,
    consts: PhysicalConstants,
    params: SimParams,
    dw: DailyWeather,
    rlds_daily: jax.Array,
    plens: jax.Array,
    startp: jax.Array,
    endp: jax.Array,
    jday: jax.Array,          # dynamic JAX scalar — NOT a Python int
    key: jax.Array,
    dtime: float,
    nband: int = 2,
    ndaypy: int = 365,
) -> tuple[IBISState, jax.Array]:
    """Run all timesteps for one day inside a lax.scan.

    jday must be a JAX scalar (jnp.int32 / jnp.int64) so that it traces as a
    dynamic value and does not trigger recompilation for each unique day.
    """
    nsteps = timesteps_per_day(dtime)

    def _step_body(carry, istep):
        st, k = carry
        k, subkey = jax.random.split(k)

        time = istep * dtime  # seconds since midnight

        _, hourly = diurnal(
            time, jday, dw, params.latitude,
            plens, startp, endp, subkey, dtime, nband,
        )

        # Override cloud-based fira with observed rlds
        # Cast forcing to match state dtype (float32 forcing → float64 state when x64 enabled)
        _cast = lambda x: jnp.asarray(x, dtype=st.atm.ta.dtype)
        # psurf: the forcing surface pressure (GSWP3 `ps`, daily) must reach
        # the physics.  Until 2026-09-21 only `diurnal` saw it (to cap qa at
        # saturation) while `atm.psurf` kept its creation value 101325 Pa on
        # every cell, so leaves and soil compared a humidity built at the
        # true pressure against saturation at sea level: 136 % "relative
        # humidity" and forced condensation (-200 mm/month) on the
        # Himalaya, negative ET on 18 % of cell-months, and wider stomata on
        # every plateau.  DEFECTS_FOR_REVIEW.md "surface pressure".
        new_atm = eqx.tree_at(
            lambda a: (a.ta, a.qa, a.ua, a.coszen,
                       a.solad, a.solai, a.fira,
                       a.raina, a.snowa, a.psurf),
            st.atm,
            (_cast(hourly.ta), _cast(hourly.qa), _cast(hourly.ua), _cast(hourly.coszen),
             _cast(hourly.solad), _cast(hourly.solai), _cast(rlds_daily),
             _cast(hourly.raina), _cast(hourly.snowa),
             _cast(jnp.broadcast_to(dw.psurf, st.atm.psurf.shape))),
        )
        st = eqx.tree_at(lambda s: s.atm, st, new_atm)

        st = timestep_body(st, consts, params, dtime)

        # ------------------------------------------------------------------
        # Accumulate per-timestep fluxes into daily / annual buckets
        # (daily: adrain, adsnow; a10ancub/ls/l3/l4 used as daily sums here,
        #  converted to proper 10-day means at end of day in _day_body)
        # ------------------------------------------------------------------
        can  = st.canopy
        veg  = st.vegetation
        sm   = st.summary
        an_ub = can.an_upper[:, 0]   # broadleaf upper net photosyn (mol CO2/m2 leaf/s)
        an_uc = can.an_upper[:, 1]   # conifer upper
        an_ls = can.an_lower[:, 0]   # shrub lower
        an_l3 = can.an_lower[:, 1]   # C3 grass lower
        an_l4 = can.an_lower[:, 2]   # C4 grass lower
        lai_u = veg.lai[:, 1]        # upper-story LAI (m2/m2)
        lai_l = veg.lai[:, 0]        # lower-story LAI
        frac  = veg.frac             # PFT fraction within canopy level (npoi, 12)

        # --- Issue 22: Snow mask for lower canopy (Fortran stats.f: smask = 1-fi) ---
        smask = 1.0 - st.snow.fi

        # --- Issue 20: NPP with maintenance + growth respiration ---
        # (Fortran stats.f sumnow: rwood, rroot, rgrowth)
        rwood   = 0.0125 / (ndaypy * 86400.0) * (1000.0 / 12.0)  # /s, mol C basis
        rroot   = 1.2500 / (ndaypy * 86400.0) * (1000.0 / 12.0)
        rgrowth = 0.30

        # Temperature functions (Lloyd-Taylor, 15°C base)
        stemtemp = can.ts
        roottemp = jnp.zeros_like(stemtemp)
        for kk in range(st.soil.tsoi.shape[1]):
            roottemp = roottemp + st.soil.tsoi[:, kk] * 0.5 * (
                veg.froot[kk, 0] + veg.froot[kk, 1])
        funca = jnp.exp(3500.0 * (1.0 / 288.16 - 1.0 / jnp.maximum(stemtemp, 200.0)))
        funcb = jnp.exp(3500.0 * (1.0 / 288.16 - 1.0 / jnp.maximum(roottemp, 200.0)))

        # Per-PFT GPP (mol CO2/m2 ground/s)
        c_fac   = 12.0e-3 * dtime   # mol CO2 → kgC × dtime
        aynpp   = sm.aynpp
        # Upper canopy: broadleaf
        for j in (0, 1, 2, 4, 6):
            gpp_j = an_ub * frac[:, j] * lai_u * veg.fu
            resp_j = rwood * veg.cbiow[:, j] * veg.sapfrac * funca + rroot * veg.cbior[:, j] * funcb
            npp_j = (1.0 - rgrowth) * (gpp_j - resp_j)
            aynpp = aynpp.at[:, j].add(npp_j * c_fac)
        # Upper canopy: conifer
        for j in (3, 5, 7):
            gpp_j = an_uc * frac[:, j] * lai_u * veg.fu
            resp_j = rwood * veg.cbiow[:, j] * veg.sapfrac * funca + rroot * veg.cbior[:, j] * funcb
            npp_j = (1.0 - rgrowth) * (gpp_j - resp_j)
            aynpp = aynpp.at[:, j].add(npp_j * c_fac)
        # Lower canopy: shrubs (no sapwood for PFTs 8,9 — but cbiow may be nonzero)
        for j in (8, 9):
            gpp_j = an_ls * frac[:, j] * lai_l * veg.fl * smask
            resp_j = rwood * veg.cbiow[:, j] * funca + rroot * veg.cbior[:, j] * funcb
            npp_j = (1.0 - rgrowth) * (gpp_j - resp_j)
            aynpp = aynpp.at[:, j].add(npp_j * c_fac)
        # C4 grass
        gpp_c4 = an_l4 * frac[:, 10] * lai_l * veg.fl * smask
        resp_c4 = rroot * veg.cbior[:, 10] * funcb
        npp_c4 = (1.0 - rgrowth) * (gpp_c4 - resp_c4)
        aynpp = aynpp.at[:, 10].add(npp_c4 * c_fac)
        # C3 grass
        gpp_c3 = an_l3 * frac[:, 11] * lai_l * veg.fl * smask
        resp_c3 = rroot * veg.cbior[:, 11] * funcb
        npp_c3 = (1.0 - rgrowth) * (gpp_c3 - resp_c3)
        aynpp = aynpp.at[:, 11].add(npp_c3 * c_fac)

        # Total canopy NPP this timestep (kgC/m2)
        npp_step = jnp.sum(aynpp - sm.aynpp, axis=1)  # increment from this step

        # GPP: gross photosynthesis (use ag instead of an)
        ag_ub = can.ag_upper[:, 0]  # broadleaf gross
        ag_uc = can.ag_upper[:, 1]  # conifer gross
        ag_ls = can.ag_lower[:, 0]  # shrub gross
        ag_l3 = can.ag_lower[:, 1]  # C3 grass gross
        ag_l4 = can.ag_lower[:, 2]  # C4 grass gross
        gpp_upper = (ag_ub * jnp.sum(frac[:, jnp.array([0,1,2,4,6])], axis=1)
                     + ag_uc * jnp.sum(frac[:, jnp.array([3,5,7])], axis=1)
                    ) * lai_u * veg.fu
        gpp_lower = (ag_ls * jnp.sum(frac[:, 8:10], axis=1)
                     + ag_l4 * frac[:, 10]
                     + ag_l3 * frac[:, 11]
                    ) * lai_l * veg.fl * smask
        gpp_step = (gpp_upper + gpp_lower) * c_fac  # kgC/m2

        # AET: canopy interception evap + soil evap + transpiration (kg/m2/s → mm)
        aet_step = (st.hydrology.ginvap + st.hydrology.gsuvap
                    + st.hydrology.gtrans) * jnp.float32(dtime)

        # Runoff (kg/m2/s → mm) — cast to float32 to match accumulator dtype
        _dt = jnp.float32(dtime)
        runof_step = st.hydrology.grunof * _dt
        drain_step = st.hydrology.gdrain * _dt

        # ET components (docs/CALIBRATION.md 15): the same fluxes aet_step
        # sums, kept apart. Static branch: free when the field is None.
        if st.etparts is not None:
            _snowsub = st.canopy.fvapi * st.snow.fi * _dt
            st = eqx.tree_at(
                lambda s: (s.etparts.trans, s.etparts.interc, s.etparts.soilev, s.etparts.snowsub), st,
                (st.etparts.trans + st.hydrology.gtrans * _dt,
                 st.etparts.interc + st.hydrology.ginvap * _dt,
                 st.etparts.soilev + st.hydrology.gsuvap * _dt - _snowsub,
                 st.etparts.snowsub + _snowsub))

        new_sm = eqx.tree_at(
            lambda s: (s.adrain, s.adsnow, s.adaet,
                       s.a10ancub, s.a10ancls, s.a10ancl3, s.a10ancl4,
                       s.aynpp, s.aynpptot,
                       s.aygpptot, s.aytrunof, s.aytdrain),
            sm,
            (sm.adrain + st.atm.raina * dtime,
             sm.adsnow + st.atm.snowa * dtime,
             sm.adaet  + aet_step,
             sm.a10ancub + an_ub * dtime,   # daily accumulator for broadleaf phot.
             sm.a10ancls + an_ls * dtime,
             sm.a10ancl3 + an_l3 * dtime,
             sm.a10ancl4 + an_l4 * dtime,
             aynpp,
             sm.aynpptot  + npp_step,       # annual TOTAL NPP (kgC/m2/yr)
             sm.aygpptot  + gpp_step,       # total annual GPP (kgC/m2/yr)
             sm.aytrunof  + runof_step,     # annual surface runoff (mm/yr)
             sm.aytdrain  + drain_step),    # annual drainage (mm/yr)
        )
        st = eqx.tree_at(lambda s: s.summary, st, new_sm)
        # Ensure all floats stay float32 so lax.scan carry types match
        st = _cast_tree_f32(st)
        return (st, k), None

    (state_out, key_out), _ = jax.lax.scan(
        _step_body,
        (state, key),
        jnp.arange(nsteps),
    )
    return state_out, key_out


# ---------------------------------------------------------------------------
# _run_month_scan: JIT-compiled day-level lax.scan for one month
#
# Extracted from run_month_gswp3 so that eqx.filter_jit compiles it exactly
# once per unique (ndays-shape, nsoilcall, ndaypy, dtime, nband) combination.
# Without this, lax.scan is called outside any jit context and JAX recompiles
# the XLA kernel each time the XLA compilation cache is evicted (~year 96).
#
# Cache entries (spinup, dtime=3600, nsoilcall=1, nband=2):
#   ndays ∈ {28, 29, 30, 31}  ×  ndaypy ∈ {365, 366}  →  8 unique kernels.
# ---------------------------------------------------------------------------

@functools.partial(
    eqx.filter_jit,
    # Python scalars (int, float) are static by default in eqx.filter_jit;
    # they appear in the cache key, so different values get separate kernels.
)
def _run_month_scan(
    state: IBISState,
    key: jax.Array,
    consts: PhysicalConstants,
    params: SimParams,
    xs: tuple,           # (td, tmax, tmin, cloud, precip, psurf, qd, ud, iwet, rlds, jdays)
    nsoilcall: int,
    ndaypy: int,
    dtime: float,
    nband: int,
) -> tuple[IBISState, jax.Array, tuple]:
    """JIT-compiled day scan for one month.

    xs must be a tuple of (ndays, npoi) JAX arrays in the order:
    td, tmax, tmin, cloud, precip, psurf, qd, ud, iwet, rlds, jdays.
    """
    (td, tmax, tmin, cloud, precip, psurf, qd, ud, iwet, rlds, jdays) = xs
    npoi = td.shape[1]
    xstore_zeros = jnp.zeros((npoi, 3), dtype=jnp.float32)
    day_sec = float(timesteps_per_day(dtime)) * dtime   # = 86400.0 for hourly

    def _day_body(carry, xs_d):
        st, k = carry
        (td_d, tmax_d, tmin_d, cloud_d, precip_d,
         psurf_d, qd_d, ud_d, iwet_d, rlds_d, jday_d) = xs_d

        # Snapshot annual accumulators before this day for daily diagnostics
        gpp_before = st.summary.aygpptot
        aet_before = st.summary.ayaet
        rh_before  = st.summary.ayco2mic

        dw = DailyWeather(
            td=td_d, tmax=tmax_d, tmin=tmin_d, cloud=cloud_d,
            precip=precip_d, psurf=psurf_d, qd=qd_d, ud=ud_d,
            iwet=iwet_d, xstore=xstore_zeros,
        )

        # Capture current 10-day means before zeroing the daily accumulators.
        # pheno reads these; _step_body then reuses the same fields as daily sums.
        pheno_summary = st.summary
        z = jnp.zeros_like(st.summary.a10ancub)
        st = eqx.tree_at(
            lambda s: (s.summary.adrain, s.summary.adsnow, s.summary.adaet,
                       s.summary.a10ancub, s.summary.a10ancls,
                       s.summary.a10ancl3, s.summary.a10ancl4),
            st,
            (jnp.zeros_like(st.summary.adrain),
             jnp.zeros_like(st.summary.adsnow),
             jnp.zeros_like(st.summary.adaet),
             z, z, z, z),
        )

        # Seasonal states (CALIBRATION.md 21-22), advanced once a day from the daily mean temperature before
        # phenology and physics read them: the state of acclimation (Makela 2004 / PRELES: S += (T - S)/tau)
        # and the Delpierre senescence sums (cold degree-days below Tb while day length is decreasing and
        # below Pstart; reset otherwise). Present only if the run enables a formulation; inert otherwise.
        sen_on_u = None
        sen_on_l = None
        if getattr(st, "seasonal", None) is not None:
            _pp = params.pheno; _sp = params.stomata_params
            _tc = td_d - 273.16
            _dl = daylength_hours(params.latitude, jday_d)
            _dec = _dl < daylength_hours(params.latitude, jday_d - 1)     # day length decreasing
            _sea = st.seasonal
            _x = _sea.acclim_x + (_tc - _sea.acclim_x) / _sp.acclim_tau
            _su = jnp.where(_dec & (_dl < _pp.sen_pstart_u), _sea.ssen_u + jnp.maximum(_pp.sen_tb_u - _tc, 0.0), 0.0)
            _sl = jnp.where(_dec & (_dl < _pp.sen_pstart_l), _sea.ssen_l + jnp.maximum(_pp.sen_tb_l - _tc, 0.0), 0.0)
            _cast_s = lambda a: jnp.asarray(a, dtype=_sea.acclim_x.dtype)
            st = eqx.tree_at(lambda s: (s.seasonal.acclim_x, s.seasonal.ssen_u, s.seasonal.ssen_l),
                             st, (_cast_s(_x), _cast_s(_su), _cast_s(_sl)))
            if _pp.sen_photoperiod_u:
                sen_on_u = _su >= _pp.sen_ycrit_u
            if _pp.sen_photoperiod_l:
                sen_on_l = _sl >= _pp.sen_ycrit_l

        # Phenology update (once per day, before timestep physics)
        new_veg = pheno(st.vegetation, pheno_summary, td_d, consts.epsilon,
                        params.pheno, st.soil.stresstu, st.soil.stresstl,
                        sen_on_u=sen_on_u, sen_on_l=sen_on_l)
        st = eqx.tree_at(lambda s: s.vegetation, st, new_veg)

        k, k_timing, k_day = jax.random.split(k, 3)
        _, plens, startp, endp = compute_precip_timing(k_timing, dw.precip, dtime)

        st, k_day = _run_day_inner(
            st, consts, params, dw,
            rlds_d, plens, startp, endp,
            jday_d, k_day, dtime, nband, ndaypy,
        )

        # End-of-day: convert daily accumulations → 10-day running means.
        # F77 (stats.f sumday) applies zweight = exp(-dtime / (10 * 86400))
        # at EVERY timestep -- niter_day times a day -- so one day's decay is
        # zweight**niter_day = exp(-1/10): a 10-day memory. This block applies
        # the update once a day and used the per-TIMESTEP weight, exp(-1/240)
        # for hourly steps: a 240-day memory that attenuated the annual cycle
        # of a10td to 24 % of its amplitude with a 77-day lag (January a10td
        # +7 C on steppe cells at -3 C), so the lower canopy never saw winter
        # and the drought triggers on a10anc* never saw a dry season
        # (docs/CALIBRATION.md 16, DEFECTS_FOR_REVIEW.md). Corrected to the
        # daily-equivalent weight.
        niter_day = timesteps_per_day(dtime)
        zweight = jnp.exp(-1.0 / 10.0)              # == exp(-1/(10*niter_day)) ** niter_day
        alpha10 = 1.0 - zweight
        new_a10td    = zweight * pheno_summary.a10td    + alpha10 * td_d
        new_a10ancub = zweight * pheno_summary.a10ancub + alpha10 * (st.summary.a10ancub / day_sec)
        new_a10ancls = zweight * pheno_summary.a10ancls + alpha10 * (st.summary.a10ancls / day_sec)
        new_a10ancl3 = zweight * pheno_summary.a10ancl3 + alpha10 * (st.summary.a10ancl3 / day_sec)
        new_a10ancl4 = zweight * pheno_summary.a10ancl4 + alpha10 * (st.summary.a10ancl4 / day_sec)

        # Add daily totals to annual accumulators
        new_ayprcp = st.summary.ayprcp + st.summary.adrain + st.summary.adsnow
        new_ayaet  = st.summary.ayaet  + st.summary.adaet

        st = eqx.tree_at(
            lambda s: (s.summary.a10td,
                       s.summary.a10ancub, s.summary.a10ancls,
                       s.summary.a10ancl3, s.summary.a10ancl4,
                       s.summary.ayprcp, s.summary.ayaet),
            st,
            (new_a10td,
             new_a10ancub, new_a10ancls, new_a10ancl3, new_a10ancl4,
             new_ayprcp, new_ayaet),
        )

        # --- Compute decomposition rate modifiers (Lloyd-Taylor + WFPS) ---
        # These scale litter and SOM decomposition in soilbgc.
        # decompl: based on top soil layer
        # decomps: root-profile weighted average of layers 1-4
        _tconst = jnp.float32(344.0)
        _btemp = jnp.float32(288.16)
        _tzero = jnp.float32(227.13)

        def _ftemp(tsoil):
            safe_t = jnp.maximum(tsoil, _tzero + jnp.float32(10.0))
            ft = jnp.exp(_tconst * (jnp.float32(1.0) / (_btemp - _tzero)
                                     - jnp.float32(1.0) / (safe_t - _tzero)))
            return jnp.minimum(ft, jnp.float32(10.0))

        def _fmoist(wsoi_val, wice_val):
            wfps = (jnp.float32(1.0) - wice_val) * wsoi_val * jnp.float32(100.0)
            fm_wet = (jnp.float32(0.000371) * wfps * wfps
                      - jnp.float32(0.0748) * wfps + jnp.float32(4.13))
            fm_dry = jnp.exp((wfps - jnp.float32(60.0)) ** 2 / jnp.float32(-800.0))
            return jnp.where(wfps >= jnp.float32(60.0), fm_wet, fm_dry)

        # decompl: top layer only.
        #
        # The ice fraction is REQUIRED here, not "approx 0" as this previously
        # assumed. F77 `stats.f` is explicit: "wsoi is relative to pore space
        # not occupied by ice and water, thus must include the ice fraction in
        # the calculation", giving wfps = (1 - wisoi) * wsoi * 100.
        #
        # Dropping it inflates water-filled pore space wherever soil is partly
        # frozen. That is not a small error in the wrong direction: the
        # moisture factor peaks at wfps = 60 and falls away on BOTH sides, so
        # an inflated wfps pushes wet boreal soils past the optimum and
        # suppresses decomposition exactly where the model already runs too
        # slow (docs/METHANE.md 5j).
        fl_top = _ftemp(st.soil.tsoi[:, 0]) * _fmoist(st.soil.wsoi[:, 0],
                                                      st.soil.wisoi[:, 0])
        decompl_new = jnp.clip(fl_top, jnp.float32(0.001), jnp.float32(10.0))

        # decomps: weighted average of top 4 layers
        # hsoi is (nsoilay+1,) — same for all grid points, not per-point
        hsoi = st.soil.hsoi
        h4 = hsoi[0] + hsoi[1] + hsoi[2] + hsoi[3]
        tsoil_avg = (st.soil.tsoi[:, 0] * hsoi[0] + st.soil.tsoi[:, 1] * hsoi[1] +
                     st.soil.tsoi[:, 2] * hsoi[2] + st.soil.tsoi[:, 3] * hsoi[3]) / h4
        wsoi_avg = (st.soil.wsoi[:, 0] * hsoi[0] + st.soil.wsoi[:, 1] * hsoi[1] +
                    st.soil.wsoi[:, 2] * hsoi[2] + st.soil.wsoi[:, 3] * hsoi[3]) / h4
        wice_avg = (st.soil.wisoi[:, 0] * hsoi[0] + st.soil.wisoi[:, 1] * hsoi[1] +
                    st.soil.wisoi[:, 2] * hsoi[2] + st.soil.wisoi[:, 3] * hsoi[3]) / h4
        fs_avg = _ftemp(tsoil_avg) * _fmoist(wsoi_avg, wice_avg)
        decomps_new = jnp.clip(fs_avg, jnp.float32(0.001), jnp.float32(10.0))

        # Update carbon state with new decomposition factors
        st = eqx.tree_at(
            lambda s: (s.carbon.decompl, s.carbon.decomps),
            st,
            (decompl_new, decomps_new),
        )

        # Soil BGC update (nsoilcall times per day for spinup acceleration)
        def _bgc_once(_, s):
            nc, nv, ns = soilbgc(
                s.carbon, s.vegetation, s.summary,
                s.soil.hsoi, s.soil.sand, s.soil.clay,
                params.bgc_params, ndaypy,
            )
            return eqx.tree_at(
                lambda x: (x.carbon, x.vegetation, x.summary),
                s, (nc, nv, ns),
            )
        st = jax.lax.fori_loop(0, nsoilcall, _bgc_once, st)

        # Accumulate Rh (heterotrophic resp) and root resp after BGC
        # tco2mic/tco2root are in mol CO2/m2/s → kgC/m2/day: ×12e-3 ×86400
        rh_day = st.vegetation.tco2mic * jnp.float32(12.0e-3 * 86400.0)
        rr_day = st.vegetation.tco2root * jnp.float32(12.0e-3 * 86400.0)
        new_sm = eqx.tree_at(
            lambda s: (s.summary.ayco2mic, s.summary.ayco2root),
            st,
            (st.summary.ayco2mic + rh_day, st.summary.ayco2root + rr_day),
        )
        st = new_sm
        # Annual-mean soil-water stress for the tree-establishment gate
        # (docs/CALIBRATION.md 14). Structure is static, so this branch is
        # resolved at trace time and costs nothing when the field is None.
        if st.stress is not None:
            # Growing-season days only (params.pheno.stress_accum_tmin, C;
            # -100 = every day). Frozen soil reads as fully stressed, so a
            # calendar mean cannot tell boreal forest from steppe (CALIBRATION.md 18).
            grow = (td_d > consts.tmelt + params.pheno.stress_accum_tmin).astype(st.stress.nday.dtype)
            st = eqx.tree_at(
                lambda s: (s.stress.aystressu, s.stress.aystressl, s.stress.nday), st,
                (st.stress.aystressu + st.soil.stresstu * grow,
                 st.stress.aystressl + st.soil.stresstl * grow,
                 st.stress.nday + grow))
        # Ensure all floats stay float32 so lax.scan carry types match
        st = _cast_tree_f32(st)

        # Daily increments for diagnostic output
        daily_gpp = st.summary.aygpptot - gpp_before
        daily_aet = st.summary.ayaet - aet_before
        daily_rh  = st.summary.ayco2mic - rh_before

        return (st, k), (daily_gpp, daily_aet, daily_rh)

    xs_all = (td, tmax, tmin, cloud, precip, psurf, qd, ud, iwet, rlds, jdays)
    (state_out, key_out), daily_diag = jax.lax.scan(_day_body, (state, key), xs_all)
    return state_out, key_out, daily_diag


# ---------------------------------------------------------------------------
# run_month_gswp3: one calendar month, day loop via lax.scan
# ---------------------------------------------------------------------------

def run_month_gswp3(
    state: IBISState,
    consts: PhysicalConstants,
    params: SimParams,
    reader: GSWP3Reader,
    latindex: np.ndarray,
    lonindex: np.ndarray,
    lat_rad: np.ndarray,
    year: int,
    imonth: int,
    key: jax.Array,
    dtime: float,
    leap: bool = False,
    nband: int = 2,
    nsoilcall: int = 1,
    f_inund=None,
    methane_params=None,
    enable_permafrost=False,
    permafrost_params=None,
) -> tuple[IBISState, jax.Array]:
    """Run one calendar month using GSWP3 daily forcing.

    Reads the month from disk (Python), stacks all daily forcing into
    (ndays, npoi) JAX arrays, then runs a single lax.scan over days.
    jday is a dynamic JAX variable so the month compiles once regardless
    of which days it contains.
    """
    # -- I/O: read one month from disk --
    gswp3_month = reader.read_month(year, imonth, latindex, lonindex)
    ndays = gswp3_month['ndays']

    xstore_np = np.zeros((len(latindex), 3), dtype=np.float32)
    daily_list, rlds_list = make_monthly_daily_weather(
        gswp3_month, lat_rad, year, imonth, xstore_np,
    )

    # -- Stack daily forcing into (ndays, npoi) JAX arrays --
    # Each DailyWeather field: (npoi,) → stack to (ndays, npoi)
    td     = jnp.stack([dw.td     for dw in daily_list])
    tmax   = jnp.stack([dw.tmax   for dw in daily_list])
    tmin   = jnp.stack([dw.tmin   for dw in daily_list])
    cloud  = jnp.stack([dw.cloud  for dw in daily_list])
    precip = jnp.stack([dw.precip for dw in daily_list])
    psurf  = jnp.stack([dw.psurf  for dw in daily_list])
    qd     = jnp.stack([dw.qd     for dw in daily_list])
    ud     = jnp.stack([dw.ud     for dw in daily_list])
    iwet   = jnp.stack([dw.iwet   for dw in daily_list])
    rlds   = jnp.stack(rlds_list)   # (ndays, npoi)

    # Julian days as a dynamic JAX array — avoids per-day recompilation
    jdays = jnp.array(
        [julian_day(imonth, d, leap) for d in range(ndays)],
        dtype=jnp.int32,
    )

    # Number of days in the year (used by soilbgc)
    ndaypy = 366 if leap else 365

    xs = (td, tmax, tmin, cloud, precip, psurf, qd, ud, iwet, rlds, jdays)
    # Ensure state is float32 before entering JIT-compiled scan
    # (restart files may contain float64 arrays from older runs)
    state = _cast_tree_f32(state)
    state_out, key_out, daily_diag = _run_month_scan(
        state, key, consts, params, xs,
        nsoilcall, ndaypy, dtime, nband,
    )

    # --- wetland CH4, monthly (docs/METHANE.md) ---------------------------
    # Monthly because that is WAD2M's resolution; a daily flux would imply a
    # precision in `f_inund` the forcing does not have. Diagnostic only -- the
    # carbon is not removed from the soil pool.
    #
    # Wired here rather than left to a script because M1 shipped tested and
    # uncalled, which is the failure this project keeps repeating
    # (docs/AUDIT.md).
    # --- permafrost, monthly (docs/AUDIT.md group A) ----------------------
    # MONTHLY, not annually: permafrost_step reads the *instantaneous* soil
    # temperature profile, so calling it once at year end reads December --
    # winter, nothing thawed, active layer 0. `thaw_depth` takes a running
    # maximum internally, so monthly calls accumulate the annual thaw correctly.
    #
    # LIMITATION: IBIS's soil column is 4.0 m and the annual damping depth in
    # wet soil is ~3.2 m, so the bottom boundary still sees ~28 % of the surface
    # amplitude. Read `active_layer` and `frozen_ground` as seasonal; treat
    # `talik` and deep `thaw_depth` as unresolved until the column is deepened.
    if enable_permafrost:
        from ibis.bgc.permafrost import (create_permafrost_state,
                                         default_permafrost_params,
                                         permafrost_step)
        pf = state_out.permafrost
        if pf is None:
            pf = create_permafrost_state(state_out.soil.tsoi.shape[0])
        pf = permafrost_step(pf, state_out.soil.tsoi, consts.tmelt,
                             state_out.vegetation.gdd0,
                             permafrost_params or default_permafrost_params(),
                             hsoi=state_out.soil.hsoi)
        state_out = eqx.tree_at(lambda s: s.permafrost, state_out, pf,
                                is_leaf=lambda x: x is None)

    if f_inund is not None:
        from ibis.bgc.methane import create_methane_state, methane_step
        so, veg = state_out.soil, state_out.vegetation
        rh_kgc_day = veg.tco2mic * jnp.float32(12.0e-3 * 86400.0)
        mstate = state_out.wetland
        if mstate is None:
            mstate = create_methane_state(rh_kgc_day.shape[0])
        mstate = methane_step(
            mstate, jnp.asarray(f_inund), rh_kgc_day,
            so.tsoi, so.wsoi, so.wisoi, so.hsoi[:so.tsoi.shape[1]],
            ndays, methane_params,
        )
        state_out = eqx.tree_at(lambda s: s.wetland, state_out, mstate,
                                is_leaf=lambda x: x is None)

    return state_out, key_out, daily_diag


# ---------------------------------------------------------------------------
# Annual climate statistics -- the input `dynaveg` needs and `biomass_update`
# does not
# ---------------------------------------------------------------------------

def annual_climate_stats(reader, year, latindex, lonindex, leap):
    """This-year `tcthis` / `twthis` / `gdd0this` / `gdd5this` from GSWP3.

    `dynaveg` reclassifies PFTs through `vegmap(.., gdd5, gdd0, ..)` and grows
    only PFTs that `existence()` permits, so running it against stale or
    zero-valued climate normals would silently rewrite the vegetation map. Both
    runners therefore have to maintain these, and both now call this rather
    than keeping their own copy of the arithmetic.

    Temperature is re-read from the reader rather than taken from the physics
    loop: file handles are cached, so this is a slice read of `tas` alone, and
    it keeps the statistic independent of the daily-weather generator.

    Returns
    -------
    (tcthis, twthis, gdd0this, gdd5this, xint) :
        The four (npoi,) statistics -- coldest and warmest monthly mean in
        degC, and growing degree-days above 0 degC and 5 degC -- plus the
        (npoi, 12) monthly means they came from, which `climanl` needs to seed
        the normals on the first year.
    """
    xint = monthly_tmean(reader, year, latindex, lonindex)
    return climate_stats_from_monthly(xint, leap) + (xint,)


def monthly_tmean(reader, year, latindex, lonindex):
    """(npoi, 12) monthly mean temperature in degC, from GSWP3 `tas`."""
    return np.stack(
        [np.nanmean(reader.read_month(year, im, latindex, lonindex)["tas"],
                    axis=0) - 273.16                              # K -> degC
         for im in range(12)], axis=1)


def climate_stats_from_monthly(xint_c, leap):
    """The arithmetic behind :func:`annual_climate_stats`, on monthly means.

    Split out so the caller that already has the monthly temperatures in hand
    (`scripts/run_global.py` reads them in its own month loop) does not need a
    second pass over the files, and so the formula exists exactly once. Two
    copies of a constant is how the combustion-completeness values came to
    disagree by a factor of five -- see docs/FIRE.md.

    Both this and `climanl` are monthly-mean approximations of what F77
    accumulates from daily temperature during the run; they now at least agree
    with each other.

    Parameters
    ----------
    xint_c : (npoi, 12) array
        Monthly mean temperature in degC.
    """
    from ibis.physics.climate_analysis import climanl

    ndaypm = tuple(int(d) for d in (NDAYPM_LEAP if leap else NDAYPM).tolist())
    x = jnp.asarray(np.nan_to_num(xint_c))
    # Delegate the growing-degree-day sum to `climanl` rather than repeating
    # it. `climanl` interpolates the monthly means to daily values before
    # accumulating, following F77 `climate.f`; a flat monthly step -- which is
    # what this used to do, and what run_global did inline -- disagrees with it
    # by ~0.15 % on a smooth annual cycle and more on a sharp one. That
    # mismatch sat exactly at the seam between seeding the normals and
    # averaging them, so the normals would step on the transition.
    _, _, _, gdd0, gdd5 = climanl(x, jnp.zeros(x.shape[0]), ndaypm)
    return (np.nanmin(xint_c, axis=1).astype(np.float32),
            np.nanmax(xint_c, axis=1).astype(np.float32),
            np.asarray(gdd0, dtype=np.float32),
            np.asarray(gdd5, dtype=np.float32))


# ---------------------------------------------------------------------------
# run_year_gswp3: full year, Python loop over 12 months
# ---------------------------------------------------------------------------

def run_year_gswp3(
    state: IBISState,
    consts: PhysicalConstants,
    params: SimParams,
    reader: GSWP3Reader,
    latindex: np.ndarray,
    lonindex: np.ndarray,
    lat_rad: np.ndarray,
    key: jax.Array,
    dtime: float,
    year: int,
    nband: int = 2,
    nsoilcall: int = 1,
    f_inund_clim=None,
    methane_params=None,
    enable_permafrost=False,
    permafrost_params=None,
) -> tuple[IBISState, jax.Array]:
    """Run a full year using GSWP3 transient daily forcing.

    Python loop over 12 months (each reads one month from disk).
    Within each month, a nested lax.scan over days × timesteps runs
    as a single compiled GPU program.

    Parameters
    ----------
    state : IBISState
        Model state at start of year.
    consts : PhysicalConstants
    params : SimParams
        Must have params.latitude set to land-point latitudes in radians.
    reader : GSWP3Reader
    latindex, lonindex : (npoi,) int arrays
        Land point indices into the GSWP3 grid (N-to-S row, W-to-E column).
    lat_rad : (npoi,) float array
        Land-point latitudes in radians.
    key : PRNGKey
    dtime : float
        Timestep length (seconds).
    year : int
        Calendar year to simulate.
    nband : int

    Returns
    -------
    (IBISState, PRNGKey)
    """
    leap = is_leap_year(year)

    for imonth in range(12):
        logger.info("  Month %d/12", imonth + 1)
        state, key, _ = run_month_gswp3(
            state, consts, params, reader,
            latindex, lonindex, lat_rad,
            year, imonth, key, dtime, leap, nband, nsoilcall,
            f_inund=None if f_inund_clim is None else f_inund_clim[imonth],
            methane_params=methane_params,
            enable_permafrost=enable_permafrost,
            permafrost_params=permafrost_params,
        )

    return state, key


# ---------------------------------------------------------------------------
# Complete carbon spin-up loop
# ---------------------------------------------------------------------------

def run_spinup_gswp3(
    state: IBISState,
    consts: PhysicalConstants,
    params: SimParams,
    reader: "GSWP3Reader",
    latindex: np.ndarray,
    lonindex: np.ndarray,
    lat_rad: np.ndarray,
    key: jax.Array,
    dtime: float,
    years,
    nband: int = 2,
    nsoilcall: int = 1,
    on_year_end=None,
    dynamic_vegetation: bool = True,
    isimfire: int = 0,
    fire_scheme: str = None,
    fire_params=None,
    popd=None,
    f_inund_clim=None,
    methane_params=None,
    enable_permafrost=False,
    permafrost_params=None,
):
    """Multi-year spin-up with the **complete** carbon chain.

    IBIS's carbon cycle is split across two cadences:

        biomass_update  ->  falll / fallr / fallw  ->  soilbgc  ->  litter, soil C
          (annual)              litterfall            (daily)

    Neither existing runner contained both halves, so litter and soil carbon
    were never produced:

    * `verify/.../global_gswp3_gpu.py` -- which generated the archived 200-year
      spin-ups -- calls `biomass_update` annually but never `soilbgc`.
    * `run_year_gswp3` calls `soilbgc` daily but never `biomass_update`, so it
      decomposes a litterfall flux that is identically zero.

    Confirmed empirically before this was written: three years at
    `nsoilcall=10` left `totlit == totalit == totcsoi == 0` exactly, with
    `falll == fallr == fallw == 0` as the cause. The archived 200-year
    checkpoints show the same -- realistic `cbiol`/`plai`, zero litter.

    Ordering follows F77: `soilbgc` runs daily against the litterfall set by
    the *previous* year's `dynaveg` (`main.f:500` calls it at year end), so
    the first simulated year legitimately has no litterfall. Annual NPP
    accumulators are cleared after `biomass_update` consumes them.

    Parameters
    ----------
    years : iterable of int
        Calendar years to run, in order.
    nsoilcall : int
        Times to advance `soilbgc` per simulated day. >1 accelerates soil
        carbon equilibration, the standard spin-up trick.
    on_year_end : callable(year, state) -> None, optional
        Hook for checkpointing or diagnostics. If it accepts a third argument
        it is also given the PRNG key. **Checkpoint the key.** The stochastic
        weather generator draws from it, so a checkpoint without it cannot be
        resumed reproducibly -- which is not academic: a NaN that appeared at
        year 173 of the v8 spin-up could not be reproduced from the year-150
        checkpoint, because restarting with a fresh key gave that cell entirely
        different weather.
    dynamic_vegetation : bool
        Call `dynaveg` rather than `biomass_update` at year end. `dynaveg` is a
        superset: it does the same NPP allocation and litterfall, and adds
        disturbance (fire, background mortality), fractional cover, canopy
        height, sapwood fraction and PFT reclassification.

        This runner called `biomass_update`, whose own docstring says it "does
        NOT apply disturbances (fire, background mortality)". So every result
        produced by this path -- including the 50-year GPU carbon spin-up --
        had **no fire feedback of any kind**, whatever `isimfire` said.

        Call one or the other, never both: they perform the *same* allocation
        from the *same* `summary.aynpp`, so calling both grows biomass twice.
    isimfire : int
        0 = no fire (IBIS's constant 0.005 background disturbance only),
        1 = fire enabled. Ignored unless `dynamic_vegetation`.
    fire_scheme : str, optional
        Defaults to `RunConfig.fire_scheme`. See docs/FIRE.md.
    fire_params : FireParams, optional
    popd : (npoi,) array, optional
        Population density, required only if `fire_params.use_human`.

    Returns
    -------
    (IBISState, PRNGKey)
    """
    from ibis.io.parameters import default_pft_params
    from ibis.physics.climate_analysis import climanl, climanl2, existence
    from ibis.physics.vegetation_dynamics import biomass_update, dynaveg

    if fire_scheme is None:
        from ibis.config import RunConfig
        fire_scheme = RunConfig().fire_scheme

    pft_params = default_pft_params()

    def _match(ref, new):
        """Keep the carry dtypes stable; this loop runs in float32."""
        return jax.tree_util.tree_map(
            lambda r, n: n.astype(r.dtype)
            if hasattr(r, "dtype") and hasattr(n, "dtype") and r.dtype != n.dtype
            else n,
            ref, new,
        )

    for year in years:
        state, key = run_year_gswp3(
            state, consts, params, reader, latindex, lonindex, lat_rad,
            key, dtime, year, nband=nband, nsoilcall=nsoilcall,
            f_inund_clim=f_inund_clim, methane_params=methane_params,
            enable_permafrost=enable_permafrost,
            permafrost_params=permafrost_params,
        )

        # --- annual: climate normals, then grow biomass and disturb it ---
        if dynamic_vegetation:
            # `dynaveg` reclassifies PFTs from gdd0/gdd5 and grows only PFTs
            # `existence()` permits, so the normals have to be current or the
            # vegetation map drifts on stale climate.
            leap = is_leap_year(year)
            tcthis, twthis, gdd0this, gdd5this, xint = annual_climate_stats(
                reader, year, latindex, lonindex, leap,
            )
            state = eqx.tree_at(
                lambda s: (s.vegetation.tcthis, s.vegetation.twthis,
                           s.vegetation.gdd0this, s.vegetation.gdd5this),
                state,
                (jnp.asarray(tcthis), jnp.asarray(twthis),
                 jnp.asarray(gdd0this), jnp.asarray(gdd5this)),
            )
            v = state.vegetation
            # `climanl2` is a 30-year running mean. Started from zero it needs
            # three decades before the normals mean anything, and `existence()`
            # is wrong the whole time -- which is not hypothetical: every
            # normal in the v7 spin-up checkpoint is exactly 0.0, because no
            # runner ever maintained them, leaving `exist` at 1.000 with all 12
            # PFTs permitted everywhere including tropical evergreen in
            # Siberia. Seed from this year's climate instead, as
            # `scripts/run_global.py` does at start-up, and only then start
            # averaging.
            if float(jnp.max(jnp.abs(v.gdd5))) == 0.0:
                ndaypm = tuple(int(d) for d in
                               (NDAYPM_LEAP if leap else NDAYPM).tolist())
                tc_n, tw_n, tcmin_n, gdd0_n, gdd5_n = climanl(
                    jnp.asarray(xint), jnp.zeros_like(v.tc), ndaypm,
                )
            else:
                tc_n, tw_n, tcmin_n, gdd0_n, gdd5_n = climanl2(
                    v.tc, v.tw, v.tcmin, v.gdd0, v.gdd5,
                    v.tcthis, v.twthis, v.gdd0this, v.gdd5this,
                    jnp.zeros_like(v.tc),
                )
            state = eqx.tree_at(
                lambda s: (s.vegetation.tc, s.vegetation.tw,
                           s.vegetation.tcmin, s.vegetation.gdd0,
                           s.vegetation.gdd5),
                state, (tc_n, tw_n, tcmin_n, gdd0_n, gdd5_n),
            )
            state = eqx.tree_at(
                lambda s: s.vegetation.exist, state,
                existence(state.vegetation.tcmin, state.vegetation.tw,
                          state.vegetation.gdd0, state.vegetation.gdd5,
                          pft_params),
            )
            # NEE = total NPP - Rh, formed once at year end from the two
            # annual accumulators, exactly as F77 does (`stats.f:1200`:
            # `ayneetot(i) = aynpptot(i) - ayco2mic(i)`). Sign follows F77:
            # POSITIVE = carbon sink.
            #
            # The port used to accumulate `ayneetot - npp_step` every timestep
            # and never add Rh at all, so it reported -106 PgC/yr for a spin-up
            # whose soil carbon had converged. That also inverted the sign
            # against dynaveg's `ayneetot - cdisturb` (copied from
            # `vegetation.f:481`), which made biomass burning INCREASE the
            # apparent sink instead of reducing it.
            state = eqx.tree_at(
                lambda s: s.summary.ayneetot, state,
                state.summary.aynpptot - state.summary.ayco2mic,
            )
            from ibis.state.stress_accum import tree_survival
            new_veg, new_carbon, new_summary = dynaveg(
                state.vegetation, state.carbon, state.summary,
                pft_params, isimfire, consts.epsilon,
                fire_scheme=fire_scheme, fire_params=fire_params, popd=popd,
                tree_surv=tree_survival(state, pft_params),
            )
        else:
            new_veg, new_carbon, new_summary = biomass_update(
                state.vegetation, state.carbon, state.summary,
                pft_params, consts.epsilon,
            )
        new_veg = _match(state.vegetation, new_veg)
        new_carbon = _match(state.carbon, new_carbon)
        new_summary = _match(state.summary, new_summary)
        state = eqx.tree_at(
            lambda s: (s.vegetation, s.carbon, s.summary),
            state, (new_veg, new_carbon, new_summary),
        )

        # --- clear the annual accumulators ---
        # Every ay* field is an annual total: summary.py and diagnostics.py both
        # label them "/yr". Only the three dynaveg consumes were being cleared,
        # so the rest accumulated across the whole run -- after 200 years
        # ayco2mic reached 921.9 kgC/m2 and ayprcp 3.2e6 mm, both written to
        # netCDF as per-year quantities. No physics reads them (dynaveg takes
        # only aynpp/ayanpptot/ayneetot, and the daily diagnostics difference
        # against a saved value), so this corrects the reported output rather
        # than the simulation. ayneetot IS consumed by dynaveg, so like aynpp it
        # is cleared here, after that consumption.
        sm = state.summary
        state = eqx.tree_at(
            lambda s: (s.summary.aynpp, s.summary.aygpptot, s.summary.ayanpptot,
                       s.summary.aynpptot,
                       s.summary.ayneetot, s.summary.ayco2mic,
                       s.summary.ayco2root, s.summary.ayprcp, s.summary.ayaet,
                       s.summary.aytrunof, s.summary.aytdrain),
            state,
            tuple(jnp.zeros_like(x) for x in
                  (sm.aynpp, sm.aygpptot, sm.ayanpptot, sm.aynpptot, sm.ayneetot,
                   sm.ayco2mic, sm.ayco2root, sm.ayprcp, sm.ayaet,
                   sm.aytrunof, sm.aytdrain)),
        )
        from ibis.state.stress_accum import reset_stress_accum
        from ibis.state.et_parts import reset_et_parts
        state = reset_stress_accum(state)
        state = reset_et_parts(state)

        if on_year_end is not None:
            try:
                on_year_end(year, state, key)
            except TypeError:
                on_year_end(year, state)      # older two-argument hooks

    return state, key
