"""Main time-stepping loops for IBIS-JAX.

Assembles the full timestep body (= Fortran lsxmain) and wraps it
with jax.lax.scan for day and month loops.

Timestep flow (from Fortran lsxmain):
  1. setsoi       — diagnostic soil conductivity, evap fractions
  2. fwetcal      — canopy wet fractions
  3. radiation    — compute_radiation (solar + IR)
  4. cascade      — precipitation through canopy
  5. fwetcal      — update wet fractions after cascade
  6. canopy loop  — canini + drystress + 3×{turcof + stomata + turvap}
  7. cascad2      — evaporation from intercepted water
  8. noveg        — set temps for unvegetated areas
  9. snow         — snowheat + snow_melt + vadapt + snow_coverage
 10. soil         — soilh2o + soilheat + freeze_thaw + wadjust
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from ibis.constants import PhysicalConstants
from ibis.state.model_state import IBISState
from ibis.state.canopy import CanopyState
from ibis.state.soil import SoilState
from ibis.state.snow import SnowState
from ibis.state.atmosphere import AtmState
from ibis.state.hydrology import HydrologyDiag
from ibis.dimensions import GridConfig

from ibis.physics.radiation import compute_radiation, VegRadParams, default_veg_rad_params
from ibis.physics.soil_physics import setsoi, soilh2o, soilheat, freeze_thaw, wadjust
from ibis.io.parameters import BGCParams, default_bgc_params
from ibis.physics.snow_physics import vadapt, snowheat, snow_melt, snow_coverage
from ibis.physics.canopy_exchange import (
    canini, turcof, turvap, impexp, impexp2,
    CanopyParams, AeroState, TransferCoeffs,
)
from ibis.physics.interception import (
    fwetcal, cascade, cascad2, noveg,
    InterceptionParams,
)
from ibis.physics.stomata import stomata, drystress, StomataParams
from ibis.physics.weather import (
    ClimateData, MonthlyPrecip, DailyWeather, HourlyWeather,
    generate_month_precip, daily, diurnal, compute_precip_timing,
)
from ibis.timestepping.calendar import (
    days_in_month, julian_day, timesteps_per_day,
)


# ---------------------------------------------------------------------------
# SimParams: all static simulation parameters
# ---------------------------------------------------------------------------

class SimParams(eqx.Module):
    """Static simulation parameters that don't change during a run.

    Bundles all sub-parameter containers needed by the timestep.
    Registered as a JAX pytree (via eqx.Module).
    """
    veg_rad: VegRadParams
    canopy: CanopyParams
    interception: InterceptionParams
    stomata_params: StomataParams
    latitude: Float[Array, "npoi"]    # Grid point latitudes (radians)

    # Soil constants
    wpudmax: float = 10.0             # Maximum puddle water (kg/m2)
    zwpmax: float = 0.5               # Maximum puddle coverage fraction
    bperm: float = 0.04               # Bottom permeability factor

    # Soil mineral properties (constant, independent of sand/clay)
    rhosoi_val: float = 2650.0        # Mineral soil density (kg/m3)
    csoi_mineral_val: float = 870.0   # Mineral specific heat (J/kg/K)

    # BGC decomposition parameters (needed by soilbgc, called daily)
    bgc_params: BGCParams = eqx.field(default_factory=default_bgc_params)


def default_sim_params(npoi: int) -> SimParams:
    """Create SimParams with default values for testing."""
    from ibis.io.physics_params import build_sim_params
    return build_sim_params(npoi, jnp.zeros((npoi,)), None)


# ---------------------------------------------------------------------------
# Intermediate values computed between physics calls
# ---------------------------------------------------------------------------

def _canopy_intermediates(
    lai, sai, fu, fl, fi, consoi_0, hsoi_0, consno, hsno_0, hsnotop,
    tgold, tiold, stef, epsilon,
    wpud, wipud, poros_0, wsoi_0, wisoi_0,
    csoi_mineral_0, rhosoi_0, ch2o, cice, rhow,
):
    """Compute intermediate values needed between turcof and turvap.

    Returns (xu, xs, xl, chgx, wlgx, wigx, cog, coi, zirg, ziri).
    """
    # Canopy area scaling factors
    xu = 2.0 * lai[:, 1] * fu
    xs = 2.0 * sai[:, 1] * fu
    xl = 2.0 * (lai[:, 0] + sai[:, 0]) * fl * (1.0 - fi)

    # --- Issue 12: coi without fi factor (Fortran canopy.f line 684) ---
    cog = consoi_0 / jnp.maximum(0.5 * hsoi_0, epsilon)
    coi = consno / jnp.maximum(0.5 * jnp.maximum(hsno_0, hsnotop), epsilon)

    # IR linearization coefficients (emissivity ≈ 0.95)
    emiss = 0.95
    zirg = 4.0 * emiss * stef * tgold ** 3
    ziri = 4.0 * emiss * stef * tiold ** 3

    # --- Issue 11: Ground heat capacity (Fortran canopy.f lines 664-668) ---
    rwork_poros = poros_0 * rhow
    chgx = (ch2o * wpud + cice * wipud
            + ((1.0 - poros_0) * csoi_mineral_0 * rhosoi_0
               + rwork_poros * (1.0 - wisoi_0) * wsoi_0 * ch2o
               + rwork_poros * wisoi_0 * cice) * hsoi_0)
    # Liquid and ice mass for ground impexp (Fortran canopy.f lines 670-676)
    wlgx = (wpud
            + rwork_poros * (1.0 - wisoi_0) * wsoi_0 * hsoi_0)
    wigx = (wipud
            + rwork_poros * wisoi_0 * hsoi_0)

    return xu, xs, xl, chgx, wlgx, wigx, cog, coi, zirg, ziri


# ---------------------------------------------------------------------------
# timestep_body: one physics timestep (= Fortran lsxmain)
# ---------------------------------------------------------------------------

def timestep_body(
    state: IBISState,
    consts: PhysicalConstants,
    params: SimParams,
    dtime: float,
) -> IBISState:
    """Execute one physics timestep.

    Assumes atmospheric forcing (state.atm.ta, qa, ua, coszen, solad,
    solai, fira, raina, snowa) has already been set for this timestep.

    Parameters
    ----------
    state : IBISState
        Full model state.
    consts : PhysicalConstants
        Physical constants.
    params : SimParams
        Static simulation parameters.
    dtime : float
        Timestep in seconds.

    Returns
    -------
    IBISState
        Updated model state after one timestep.
    """
    c = consts
    ip = params.interception
    cp = params.canopy
    sp = params.stomata_params

    # --- Promote atm forcing to float64 (defensive) ---
    # GSWP3 and many other forcing netCDFs store data as float32. JAX's
    # weak-type promotion preserves float32 precision through operations
    # like `psurf * siga` (with siga a Python float), which corrupts the
    # subtractive cancellation `psurf - psurf*siga` in canini → za drifts
    # by ~2e-4 → mK-level state divergence amplified by cp·sg to ~0.3 W/m²
    # in flux. See verify/STATUS.md finding #18.
    _f64 = lambda x: x.astype(jnp.float64) if x.dtype == jnp.float32 else x
    state = eqx.tree_at(
        lambda s: (s.atm.ta, s.atm.qa, s.atm.ua, s.atm.psurf,
                   s.atm.coszen, s.atm.solad, s.atm.solai,
                   s.atm.fira, s.atm.raina, s.atm.snowa),
        state,
        (_f64(state.atm.ta), _f64(state.atm.qa), _f64(state.atm.ua),
         _f64(state.atm.psurf), _f64(state.atm.coszen),
         _f64(state.atm.solad), _f64(state.atm.solai),
         _f64(state.atm.fira), _f64(state.atm.raina), _f64(state.atm.snowa)),
    )

    # --- Extract frequently used arrays ---
    soil = state.soil
    snow = state.snow
    can = state.canopy
    atm = state.atm
    veg = state.vegetation

    npoi = soil.tg.shape[0]
    nsoilay = soil.tsoi.shape[1]

    # =====================================================================
    # 1. Diagnostic soil properties
    # =====================================================================
    setsoi_res = setsoi(
        soil.sand, soil.clay, soil.poros,
        soil.wsoi, soil.wisoi, soil.tsoi,
        soil.wpud, soil.wipud,
        params.wpudmax, params.zwpmax,
        snow.tsno[:, 0], atm.ta,
        c.tmelt, c.hvap, c.hsub, c.ch2o, c.cvap, c.cice, c.epsilon,
    )

    # =====================================================================
    # 2. Canopy wet fractions
    # =====================================================================
    fwet = fwetcal(
        can.tu, can.ts, can.tl,
        can.wliqu, can.wliqs, can.wliql,
        can.wsnou, can.wsnos, can.wsnol,
        ip.wliqumax, ip.wliqsmax, ip.wliqlmax,
        ip.wsnoumax, ip.wsnosmax, ip.wsnolmax,
        ip.fmax, c.tmelt, c.epsilon,
    )

    # =====================================================================
    # 3. Radiation
    # =====================================================================
    rad = compute_radiation(
        atm.coszen, atm.solad, atm.solai, atm.fira,
        veg.lai, veg.sai, veg.fu, veg.fl,
        can.tu, can.ts, can.tl, soil.tg, snow.ti,
        snow.fi, snow.tsno[:, 0],
        soil.wsoi[:, 0], soil.wisoi[:, 0],
        soil.albsav, soil.albsan,
        fwet.fwetu, fwet.fwets, fwet.fwetl,
        fwet.rliqu, fwet.rliqs, fwet.rliql,
        params.veg_rad,
        c.stef, c.tmelt, c.epsilon,
    )

    # =====================================================================
    # 4. Precipitation cascade
    # =====================================================================
    casc = cascade(
        can.tu, can.ts, can.tl, can.t12, can.t34,
        can.wliqu, can.wliqs, can.wliql,
        can.wsnou, can.wsnos, can.wsnol,
        veg.lai, veg.sai, veg.fu, veg.fl,
        atm.ta, atm.qa, atm.psurf,
        atm.raina, atm.snowa,
        ip,
        dtime, c.tmelt, c.ch2o, c.cice, c.hfus, c.epsilon,
        c.cair, c.hvap, c.cvap,
    )

    # Update intercepted water from cascade
    wliqu = casc.wliqu
    wliqs = casc.wliqs
    wliql = casc.wliql
    wsnou = casc.wsnou
    wsnos = casc.wsnos
    wsnol = casc.wsnol

    # =====================================================================
    # 5. Recompute wet fractions after cascade
    # =====================================================================
    fwet2 = fwetcal(
        can.tu, can.ts, can.tl,
        wliqu, wliqs, wliql,
        wsnou, wsnos, wsnol,
        ip.wliqumax, ip.wliqsmax, ip.wliqlmax,
        ip.wsnoumax, ip.wsnosmax, ip.wsnolmax,
        ip.fmax, c.tmelt, c.epsilon,
    )

    # =====================================================================
    # 6. Canopy exchange (3 iterations)
    # =====================================================================

    # 6a. Aerodynamic initialization
    aero = canini(
        atm.ta, atm.qa, atm.psurf, atm.ua,
        snow.fi, soil.z0soi, snow.z0sno,
        veg.fl, veg.fu, veg.lai, veg.sai, veg.ztop, veg.zbot,
        cp.alaiml, cp.alaimu,
        c.grav, c.rair, c.rvap, c.cair, c.cvap, c.cappa,
    )

    # 6b. Soil moisture stress (once, outside iteration loop)
    # NOTE: drystress returns (stresstl, stresstu, stressl, stressu) — LOWER
    # canopy first. We must unpack in that order. Prior code unpacked as
    # (stresstu, stresstl, stressu, stressl) which silently swapped upper/
    # lower canopy stress through the entire timestep_body. This was the
    # root cause of the multi-day heat-wave drift (finding #16).
    stresstl, stresstu, stressl, stressu = drystress(
        soil.wsoi, soil.wisoi, soil.swilt, soil.sfield,
        veg.froot,
    )

    # 6c. Compute canopy intermediates
    xu, xs, xl, chgx, wlgx, wigx, cog, coi, zirg, ziri = _canopy_intermediates(
        veg.lai, veg.sai, veg.fu, veg.fl, snow.fi,
        setsoi_res.consoi[:, 0], soil.hsoi[0],
        snow.consno, snow.hsno[:, 0], snow.hsnotop,
        soil.tg, snow.ti,
        c.stef, c.epsilon,
        soil.wpud, soil.wipud, soil.poros[:, 0], soil.wsoi[:, 0], soil.wisoi[:, 0],
        jnp.full_like(soil.tg, params.csoi_mineral_val),
        jnp.full_like(soil.tg, params.rhosoi_val),
        c.ch2o, c.cice, c.rhow,
    )

    # Save old temperatures for implicit/explicit blending
    tuold, tsold, tlold = can.tu, can.ts, can.tl
    tgold, tiold = soil.tg, snow.ti

    # Working canopy state (updated each iteration)
    tu, ts, tl = can.tu, can.ts, can.tl
    t12, t34 = can.t12, can.t34
    q12, q34 = can.q12, can.q34
    tg, ti = soil.tg, snow.ti

    # Stratification correction initial values
    z = jnp.zeros((npoi,))
    ones = jnp.ones((npoi,))
    richl, richu = z, z
    straml, strahl = ones, ones
    stramu, strahu = ones, ones

    # Implicit/explicit fractions (start with 1.0)
    wu_prev, ws_prev, wl_prev = ones, ones, ones
    wg_prev, wi_prev = ones, ones

    # Stomata state
    ci_upper, gs_upper, cs_upper = can.ci_upper, can.gs_upper, can.cs_upper
    ci_lower, gs_lower, cs_lower = can.ci_lower, can.gs_lower, can.cs_lower
    totcondub = z
    totconduc = z
    totcondls = z
    totcondl3 = z
    totcondl4 = z
    a10sp_u, a10dl_u = can.a10scalparamu, can.a10daylightu
    a10sp_l, a10dl_l = can.a10scalparaml, can.a10daylightl
    an_upper, ag_upper = can.an_upper, can.ag_upper
    an_lower, ag_lower = can.an_lower, can.ag_lower

    canopy_result = None

    # 6d. Three-iteration loop (unrolled for JIT tracing).
    # F77 turcof updates u2/alogu/alogl in com1d each iter; subsequent
    # turcof calls read them back. JAX must carry them explicitly.
    for iter_num in range(3):
        is_first = (iter_num == 0)

        # Override aero.u2/alogu/alogl with the carry-forward values
        # (canini values at iter 0; turcof outputs thereafter).
        aero_iter = eqx.tree_at(
            lambda a: (a.u2, a.alogu, a.alogl),
            aero,
            (aero.u2 if is_first else u2_carry,
             aero.alogu if is_first else alogu_carry,
             aero.alogl if is_first else alogl_carry),
        )

        # --- Transfer coefficients ---
        tc = turcof(
            aero_iter, atm.ta, atm.qa, atm.ua,
            t12, t34, q12, q34, snow.fi,
            c.vonk, dtime,
            cp.cleaf, cp.cstem, cp.cgrass, cp.dleaf, cp.dstem,
            iter_num,
            richl, straml, strahl,
            richu, stramu, strahu,
            c.grav,
        )

        # --- Stomata ---
        stom = stomata(
            tu, tl, t12, t34, q12, q34,
            tc.su, tc.sl, atm.psurf,
            rad.topparu, rad.topparl,
            rad.termu6, rad.termu7,
            rad.terml6, rad.terml7,
            rad.scalcoefu, rad.scalcoefl,
            ci_upper, gs_upper, cs_upper,
            ci_lower, gs_lower, cs_lower,
            fwet2.fwetu, fwet2.fwetl,
            veg.lai, veg.sai, veg.exist,
            stresstu, stresstl,
            a10sp_u, a10dl_u,
            a10sp_l, a10dl_l,
            atm.co2conc, dtime, sp, c.epsilon,
        )
        ci_upper = stom.ci_upper
        gs_upper = stom.gs_upper
        cs_upper = stom.cs_upper
        ci_lower = stom.ci_lower
        gs_lower = stom.gs_lower
        cs_lower = stom.cs_lower
        an_upper = stom.an_upper
        ag_upper = stom.ag_upper
        an_lower = stom.an_lower
        ag_lower = stom.ag_lower
        totcondub = stom.totcondub
        totconduc = stom.totconduc
        totcondls = stom.totcondls
        totcondl3 = stom.totcondl3
        totcondl4 = stom.totcondl4
        a10sp_u = stom.a10scalparamu
        a10dl_u = stom.a10daylightu
        a10sp_l = stom.a10scalparaml
        a10dl_l = stom.a10daylightl

        # --- Issue 10: Vegetation heat capacities with intercepted water ---
        # (Fortran canopy.f lines 657-659)
        chux = cp.chu + c.ch2o * wliqu + c.cice * wsnou
        chsx = cp.chs + c.ch2o * wliqs + c.cice * wsnos
        chlx = cp.chl + c.ch2o * wliql + c.cice * wsnol

        # --- Implicit/explicit fractions ---
        wu = impexp(tu, chux, wliqu, wsnou, c.tmelt, c.hfus, c.epsilon,
                    wu_prev, is_first)
        ws = impexp(ts, chsx, wliqs, wsnos, c.tmelt, c.hfus, c.epsilon,
                    ws_prev, is_first)
        wl = impexp(tl, chlx, wliql, wsnol, c.tmelt, c.hfus, c.epsilon,
                    wl_prev, is_first)
        # --- Issue 9: Ground uses impexp (not impexp2) with full heat capacity ---
        wg = impexp(tg, chgx, wlgx, wigx, c.tmelt, c.hfus, c.epsilon,
                    wg_prev, is_first)
        wi = impexp2(ti, tiold, c.tmelt, c.epsilon, wi_prev, is_first)

        # --- 9x9 canopy solve ---
        canopy_result = turvap(
            tu, ts, tl, t12, t34, q12, q34, tg, ti,
            tuold, tsold, tlold, tgold, tiold,
            tc.cu, tc.cl, tc.su, tc.ss, tc.sl, tc.sg, tc.si,
            aero.cp, aero.rhoa, aero.tfac,
            xu, xs, xl, snow.fi, veg.fu, veg.fl,
            veg.lai, veg.sai, veg.frac,
            chux, chsx, chlx,
            wliqu, wliqs, wliql, wsnou, wsnos, wsnol,
            fwet2.rliqu, fwet2.rliqs, fwet2.rliql,
            fwet2.fwetu, fwet2.fwets, fwet2.fwetl,
            totcondub, totconduc, totcondls, totcondl3, totcondl4,
            chgx, cog, coi, zirg, ziri,
            soil.wpud, soil.wipud, params.wpudmax,
            soil.wisoi[:, 0], soil.wsoi[:, 0],
            soil.swilt[:, 0], soil.suction[:, 0], soil.bex[:, 0],
            setsoi_res.hvasug, setsoi_res.hvasui,
            soil.tsoi[:, 0], snow.tsno[:, 0],
            rad.solu, rad.sols, rad.soll, rad.solg, rad.soli,
            rad.firu, rad.firs, rad.firl, rad.firg, rad.firi, rad.firb,
            casc.pfluxu, casc.pfluxs, casc.pfluxl,
            atm.ta, atm.qa, atm.psurf,
            wu, ws, wl, wg, wi,
            dtime, c.tmelt, c.hvap, c.hsub, c.ch2o, c.cice,
            c.cvap, c.stef, c.rvap, c.grav, c.epsilon,
        )

        # Update working state for next iteration
        tu = canopy_result.tu
        ts = canopy_result.ts
        tl = canopy_result.tl
        t12 = canopy_result.t12
        t34 = canopy_result.t34
        q12 = canopy_result.q12
        q34 = canopy_result.q34
        tg = canopy_result.tg
        ti = canopy_result.ti

        # Carry forward stratification + implicit fractions + wind/log state.
        straml, strahl = tc.straml, tc.strahl
        stramu, strahu = tc.stramu, tc.strahu
        u2_carry = tc.u2
        alogu_carry = tc.alogu
        alogl_carry = tc.alogl
        wu_prev, ws_prev, wl_prev = wu, ws, wl
        wg_prev, wi_prev = wg, wi

    # F77 turvap clamps ti = min(ti, tmelt) ONLY at last iter (after the
    # post-solve flux block). Apply that here, after the 3-iter loop.
    # Note: canopy_result.ti is unclamped from the last iter; we rebuild
    # the result tuple with ti clamped.
    canopy_result = eqx.tree_at(
        lambda r: r.ti, canopy_result,
        jnp.minimum(canopy_result.ti, c.tmelt))
    ti = canopy_result.ti

    # =====================================================================
    # 7. Evaporation cascade (cascad2)
    # =====================================================================
    (tu2, ts2, tl2, wliqu2, wliqs2, wliql2,
     wsnou2, wsnos2, wsnol2, _fvapa2, _fsena2) = cascad2(
        tu, ts, tl,
        wliqu, wliqs, wliql, wsnou, wsnos, wsnol,
        fwet2.rliqu, fwet2.rliqs, fwet2.rliql,
        canopy_result.fvapuw, canopy_result.fvaps, canopy_result.fvaplw,
        veg.fu, veg.fl, snow.fi, veg.lai, veg.sai,
        cp.chu, cp.chs, cp.chl,
        dtime, c.ch2o, c.cice, c.hfus, c.tmelt,
        atm.ta, canopy_result.fvapa, canopy_result.fsena,
        c.hvap, c.hsub, c.cvap,
    )
    tu, ts, tl = tu2, ts2, tl2

    # =====================================================================
    # 8. No-vegetation handling
    # =====================================================================
    (tu_nv, ts_nv, tl_nv,
     wliqu_nv, wliqs_nv, wliql_nv,
     wsnou_nv, wsnos_nv, wsnol_nv) = noveg(
        tu, ts, tl,
        wliqu2, wliqs2, wliql2, wsnou2, wsnos2, wsnol2,
        veg.lai, veg.sai, veg.fu, veg.fl, snow.fi,
        tg, ti,
    )

    # Ground-reaching precipitation (from initial cascade)
    raing_total = casc.raing
    snowg_total = casc.snowg

    # =====================================================================
    # 9. Soil/snow surface heat fluxes
    # =====================================================================
    # === Hudson Bay Y192M8 fix ===
    # turvap's 9x9 canopy energy-balance solve can diverge in dry-cold-
    # transitioning permafrost conditions (e.g., Hudson Bay 61.5N -94.5
    # at Y192M8: layer 0 dry+frozen at -28C while layer 1 saturated liquid
    # at 0C — the implicit solve produced tg_canopy_out=2316 K, then
    # consistent fseng=+135k W/m² and firg=-64k W/m², driving fhtop_soil
    # to -200k W/m² and tsoi[0] to -1169 K within one substep).
    #
    # The downstream T-clamp at section 12 (IBIS_T_MAX_DEV) clamps the
    # IBIS state surface temperatures AFTER soilheat. It does NOT protect
    # heatg here, which uses the canopy solve's raw flux output.
    #
    # Fix: clamp the surface fluxes (firg, fseng) to physical maxima
    # (±2 kW/m²; extreme tropical noon sensible heat is ~1 kW/m²) before
    # building heatg. Same idea as the fvapg clamp (finding #27) but for
    # heat fluxes. No-op for healthy cells (fluxes always far below 2 kW).
    import os as _hb
    FLUX_MAX = float(_hb.environ.get("IBIS_FLUX_MAX", "2000.0"))
    firg_clamped = jnp.clip(canopy_result.firg, -FLUX_MAX, FLUX_MAX)
    fseng_clamped = jnp.clip(canopy_result.fseng, -FLUX_MAX, FLUX_MAX)
    firi_clamped = jnp.clip(canopy_result.firi, -FLUX_MAX, FLUX_MAX)
    fseni_clamped = jnp.clip(canopy_result.fseni, -FLUX_MAX, FLUX_MAX)
    heatg = (rad.solg + firg_clamped
             - fseng_clamped - setsoi_res.hvasug * canopy_result.fvapg)
    heati = (rad.soli + firi_clamped
             - fseni_clamped - setsoi_res.hvasui * canopy_result.fvapi)

    # =====================================================================
    # 10. Snow physics (matches Fortran snow.f)
    # =====================================================================
    nsnolay = snow.hsno.shape[1]
    fiold = snow.fi   # save fi at start of snow section

    # Precipitation rates reaching snow surface.
    # Fortran snow.f line 96 + 101 use rainl + snowl (both BELOW upper
    # canopy, ABOVE lower canopy). Previously this used casc.snowg (snow
    # reaching ground -- after lower canopy) which is wrong; that gave
    # ~21% rel error on hsno cell 2.
    rainl_rate = casc.rainl
    snowl_rate = casc.snowl    # snow below upper canopy (NOT snowg)
    trainl_temp = casc.trainl
    tsnowl_temp = casc.tsnowl  # temperature of snowl (was casc.tsnowg)

    # Heat flux into snow top from canopy solve + precipitation heat.
    # F77 snow.f line 96 uses trainl directly (no clamp); rain colder than
    # tmelt thus injects NEGATIVE sensible-heat (cools snowpack).
    fhtop_snow = (heati
                  + rainl_rate * (c.ch2o * (trainl_temp - c.tmelt)
                                  + c.hfus + c.cice * (c.tmelt - snow.tsno[:, 0]))
                  + snowl_rate * c.cice * (tsnowl_temp - snow.tsno[:, 0]))

    # Buried vegetation density (lower canopy LAI+SAI buried under snow)
    xl = veg.fl * 2.0 * (veg.lai[:, 0] + veg.sai[:, 0])

    # --- Issue 1: Update hsno BEFORE snowheat (Fortran snow.f lines 95-113) ---
    rwork_sno = dtime / snow.rhos
    hsno_with_precip = jnp.where(
        snow.fi[:, None] > 0.0,
        snow.hsno.at[:, 0].set(
            snow.hsno[:, 0] + (rainl_rate + snowl_rate - canopy_result.fvapi) * rwork_sno
        ),
        snow.hsno,
    )
    # F77 snow.f line 100-102 does NOT clamp hsno to >= 0; allows negative
    # temporarily. Clamping here breaks derivative match for cells near
    # freeze boundary with strong sublimation/condensation.

    # snowheat — uses updated hsno and persistent tlsub
    snheat = snowheat(
        snow.tsno, snow.tlsub,  # persistent buried-veg temperature
        fhtop_snow,
        xl,
        cp.chl,
        snow.fi, hsno_with_precip,  # use precip-updated thickness
        snow.rhos, snow.consno,
        soil.hsoi[0], setsoi_res.consoi[:, 0],
        soil.tsoi[:, 0],
        dtime, c.cice,
    )

    # F77 snow.f order (re-established to match exactly):
    # 1) snowheat (above)
    # 2) new-snow on snow-free fraction (dfi) + mix tsno + hinit
    # 3) snow_melt
    # 4) snow_coverage (sqrt fi-vs-hsnomin)
    # 5) vadapt
    # 6) fimin melt-all
    # ORDER MATTERS: F77 mixes new snowfall into tsno BEFORE snow_melt,
    # and runs fimin AFTER vadapt. Previous JAX order ran snow_melt
    # before dfi/mix and fimin before dfi -- caused 1mK divergence on
    # snow.tsno cell 2 freezethaw under cool-atm perturbations.

    # (2) new-snow on snow-free area + mix tsno + hinit for newly-created
    # snow cells. F77 snow.f line 137-152.
    ht_pre = jnp.sum(hsno_with_precip, axis=1)
    ht_safe_pre = jnp.where(ht_pre > 0.0, ht_pre, snow.hsnomin)
    dfi_new = (1.0 - snow.fi) * dtime * casc.snowg / (snow.rhos * ht_safe_pre)
    dfi_new = jnp.minimum(dfi_new, 1.0 - snow.fi)
    fi_with_new = snow.fi + dfi_new
    # Mix snowfall temperature into existing snow column (F77 line 142-146)
    tsno_mixed = jnp.where(
        (fi_with_new > 0.0)[:, None],
        (snheat.tsno * snow.fi[:, None] + casc.tsnowg[:, None] * dfi_new[:, None])
        / jnp.maximum(fi_with_new[:, None], c.epsilon),
        snheat.tsno,
    )
    # hinit for newly-created snow (fi was 0)
    hinit = jnp.full_like(hsno_with_precip,
                           (snow.hsnomin - snow.hsnotop) / jnp.maximum(nsnolay - 1, 1))
    hinit = hinit.at[:, 0].set(snow.hsnotop)
    hsno_pre_melt = jnp.where(
        (snow.fi < c.epsilon)[:, None] & (dfi_new > 0.0)[:, None],
        hinit,
        hsno_with_precip,
    )

    # (3) snow_melt — uses POST-dfi tsno_mixed and fi_with_new
    hsno_new, tsno_new, smelt, zheat = snow_melt(
        hsno_pre_melt, tsno_mixed,
        fi_with_new, snow.rhos, c.cice, c.hfus, c.tmelt,
    )

    # (4) snow_coverage (sqrt fi-vs-hsnomin)
    hsnoruf = jnp.minimum(0.70, jnp.maximum(snow.hsnomin + 0.05,
                                              veg.fl * veg.ztop[:, 0]))
    fi_cov, hsno_cov = snow_coverage(
        hsno_new, fi_with_new,
        snow.hsnomin, hsnoruf, snow.fimax, c.epsilon,
    )

    # (5) vadapt
    hsno_va, tsno_va = vadapt(
        hsno_cov, tsno_new,
        snow.hsnotop,
        (jnp.sum(hsno_cov, axis=1) > c.epsilon),
        c.epsilon,
    )

    # (6) fimin melt-all (F77 snow.f line 224-237)
    too_small = fi_cov < snow.fimin
    remaining_mass = snow.rhos * jnp.sum(hsno_va, axis=1) * fi_cov
    remaining_heat = -jnp.sum(
        snow.rhos * hsno_va * fi_cov[:, None]
        * (c.cice * (c.tmelt - tsno_va) + c.hfus),
        axis=1,
    )
    smelt = jnp.where(too_small, smelt + remaining_mass, smelt)
    zheat = jnp.where(too_small, zheat + remaining_heat, zheat)
    hsno_final = jnp.where(too_small[:, None], 0.0, hsno_va)
    tsno_va = jnp.where(too_small[:, None], c.tmelt, tsno_va)
    fi_new = jnp.where(too_small, 0.0, fi_cov)
    fi_after_melt = snow.fi  # for buried-veg block: fi BEFORE dfi/coverage
    # actually F77 uses fiold for buried-veg adjustment, so fi_after_melt
    # should be fiold value

    # --- Issue 3: Buried-vegetation fi-adjustment (Fortran snow.f lines 247-273) ---
    dfi_cover = fi_new - fi_after_melt
    tlsub_out = snheat.tlsub

    # fi increased: mix tlsub with tl
    tlsub_out = jnp.where(
        dfi_cover > 0,
        (snheat.tlsub * fi_after_melt + tl_nv * dfi_cover)
        / jnp.maximum(fi_new, c.epsilon),
        tlsub_out,
    )
    # fi increased: intercepted water energy → zheat; mass → bottom snow layer
    zheat = jnp.where(
        dfi_cover > 0,
        zheat + dfi_cover * xl * (
            wliql_nv * (c.ch2o * (tl_nv - c.tmelt) + c.hfus
                        + c.cice * (c.tmelt - tsno_va[:, -1]))
            + wsnol_nv * c.cice * (tl_nv - tsno_va[:, -1])
        ),
        zheat,
    )
    hsno_final = jnp.where(
        (dfi_cover > 0)[:, None],
        hsno_final.at[:, -1].add(
            dfi_cover * xl * (wliql_nv + wsnol_nv)
            / (snow.rhos * jnp.maximum(fi_new, c.epsilon))
        ),
        hsno_final,
    )
    # fi decreased: heat correction + redistribute intercepted water
    zheat = jnp.where(
        dfi_cover < 0,
        zheat - dfi_cover * xl * cp.chl * (snheat.tlsub - tl_nv),
        zheat,
    )
    rwork_dec = jnp.where(
        dfi_cover < 0,
        (1.0 - fi_after_melt) / jnp.maximum(1.0 - fi_new, c.epsilon),
        1.0,
    )
    wliql_nv = wliql_nv * rwork_dec
    wsnol_nv = wsnol_nv * rwork_dec

    # When fi drops to 0, reset tlsub to tl
    tlsub_out = jnp.where(fi_new > 0, tlsub_out, tl_nv)

    # --- Issue 4: Areal flux averaging (Fortran snow.f lines 278-295) ---
    rwork_inv_dt = 1.0 / dtime
    heatg_avg = ((1.0 - fiold) * heatg
                 + fiold * snheat.sflo[:, -1]
                 + zheat * rwork_inv_dt)
    solg_avg = (1.0 - fiold) * rad.solg
    fvapg_avg = (1.0 - fiold) * canopy_result.fvapg
    # === Finding #27 fix ===
    # Physical bound on ground vapor flux. fvapg_avg comes out of the
    # canopy 3-iteration energy-balance loop, which DIVERGES in the
    # dry+hot+full-canopy regime (e.g. Yemen highlands lat 13.5-14°N
    # lon 44-44.5°E, only visible at sub-degree resolution): the iteration
    # produces fvapg ≈ 0.26 kg/m²/s = ~940 mm/hr of evaporation, which is
    # physically impossible (max realistic evap ≈ 1 mm/hr ≈ 3e-4 kg/m²/s).
    # The fictitious flux drives wpud to -60 kg/m² then to +200 kg/m² in
    # subsequent timesteps, which through fwpud injects fictitious water
    # into the top soil layer and triggers the NaN cascade documented in
    # STATUS.md finding #27.
    #
    # This clamp guards the puddle/soil water balance against the upstream
    # divergence. Maximum physical rate ≈ 1e-3 kg/m²/s = 3.6 mm/hr (extreme
    # but possible for hot dry windy desert; condensation symmetric).
    # Verified at 117-cell verification grid: typical fvapg_avg < 1e-4,
    # so this clamp does NOT fire on any cell exercised by the 46-yr
    # F77↔JAX lockstep verification.
    FVAPG_PHYS_MAX = 1.0e-3  # kg/m²/s (3.6 mm/hr of evap or condensation)
    fvapg_avg = jnp.clip(fvapg_avg, -FVAPG_PHYS_MAX, FVAPG_PHYS_MAX)
    # Combined rain + snowmelt reaching soil
    raing_nf = (1.0 - fiold) * raing_total
    smelt_rate = smelt * rwork_inv_dt
    # Finding #24: F77 snow.f:293 applies a mix-style epsilon threshold
    # (`if abs(xtmp).lt.epsilon then xtmp=0, ttmp=0`) on raing/traing.
    # JAX previously used the smooth formula without this threshold, leaving
    # tiny sub-epsilon raing_avg values (~5e-8) that contribute
    # raing_avg*ch2o*(tmelt-tsoi) ≈ 3 mW/m² to fhtop_soil. At Arctic spring
    # thaw this caused the day-84 mK-level tsoi[0] shock.
    raing_unmix = raing_nf + smelt_rate
    raing_zero = raing_unmix < c.epsilon
    raing_avg = jnp.where(raing_zero, 0.0, raing_unmix)
    traing_avg = jnp.where(
        raing_zero,
        0.0,
        (raing_nf * casc.traing + smelt_rate * c.tmelt)
        / jnp.maximum(raing_unmix, c.epsilon),
    )

    # =====================================================================
    # 11. Soil physics (matches Fortran soil.f soilctl)
    # =====================================================================
    # Soil mineral properties (constant)
    csoi_mineral = jnp.full_like(soil.tsoi, params.csoi_mineral_val)
    rhosoi = jnp.full_like(soil.tsoi, params.rhosoi_val)

    # Puddle heat capacity terms
    c0pud = jnp.zeros_like(soil.tsoi)
    c0pud = c0pud.at[:, 0].set(soil.wpud * c.ch2o + soil.wipud * c.cice)

    # --- Issue 16: Puddle dynamics (Fortran soil.f lines 240-330) ---
    # (0) Immediately transfer excess puddle to runoff
    wpud_w = soil.wpud
    wipud_w = soil.wipud
    grunof = jnp.minimum(wpud_w, jnp.maximum(
        0.0, wpud_w + wipud_w - params.wpudmax)) / dtime
    wpud_w = wpud_w - grunof * dtime

    # (1) Apportion rain between puddle and runoff
    zfrez = jnp.clip((soil.tsoi[:, 0] - c.tmelt + 0.5) * 2.0, 0.0, 1.0)
    zrunf = zfrez * jnp.clip((wpud_w + wipud_w) / params.wpudmax, 0.0, 1.0)
    wpud_w = wpud_w + (1.0 - zrunf) * raing_avg * dtime
    grunof = grunof + zrunf * raing_avg

    # --- Issue 14: Evaporation partitioning via qglif (Fortran soil.f lines 281-309) ---
    is_evap = fvapg_avg >= 0.0
    evap_dt = fvapg_avg * dtime

    # Evaporation: split according to qglif fractions
    fwtop = jnp.where(is_evap, -setsoi_res.qglif[:, 0] * fvapg_avg, 0.0)
    wpud_w = jnp.where(is_evap,
                        wpud_w - setsoi_res.qglif[:, 2] * evap_dt,
                        wpud_w)
    wipud_w = jnp.where(is_evap,
                         wipud_w - setsoi_res.qglif[:, 3] * evap_dt,
                         wipud_w)
    # Ice sublimation from top soil layer
    wisoi_ice_sub = jnp.where(is_evap,
                               setsoi_res.qglif[:, 1] * evap_dt
                               / jnp.maximum(c.rhow * soil.poros[:, 0] * soil.hsoi[0], c.epsilon),
                               0.0)
    wisoi_0_new = jnp.maximum(0.0, soil.wisoi[:, 0] - wisoi_ice_sub)
    wsoi_0_adj = jnp.where(
        (1.0 - wisoi_0_new) > c.epsilon,
        soil.wsoi[:, 0] * (1.0 - soil.wisoi[:, 0]) / (1.0 - wisoi_0_new),
        soil.wsoi[:, 0],
    )

    # Condensation: all goes to puddles
    wpud_w = jnp.where(~is_evap,
                        wpud_w - (setsoi_res.qglif[:, 0] + setsoi_res.qglif[:, 2]) * evap_dt,
                        wpud_w)
    wipud_w = jnp.where(~is_evap,
                         wipud_w - (setsoi_res.qglif[:, 1] + setsoi_res.qglif[:, 3]) * evap_dt,
                         wipud_w)

    # (3) Puddle infiltration
    zdpud = c.rhow * dtime * jnp.maximum(1.0 - wisoi_0_new, 0.0) ** 2 * soil.hydraul[:, 0]
    fwpud = jnp.maximum(0.0, jnp.minimum(wpud_w, zdpud)) / dtime
    c0pud = c0pud.at[:, 0].set(c.ch2o * wpud_w + c.cice * wipud_w)

    # Apply wisoi/wsoi ice sublimation adjustment to soil state for this timestep
    wsoi_work = jnp.where(is_evap[:, None],
                           soil.wsoi.at[:, 0].set(wsoi_0_adj),
                           soil.wsoi)
    wisoi_work = jnp.where(is_evap[:, None],
                            soil.wisoi.at[:, 0].set(wisoi_0_new),
                            soil.wisoi)

    # Compute effective porosity (accounts for ice)
    porosflo = soil.poros * jnp.maximum(1.0 - wisoi_work, c.epsilon)
    porosflo = jnp.maximum(porosflo, c.epsilon)  # guard for poros=0 (glaciers)

    # Per-layer transpiration extraction (kg/m2/s)
    lai_u = veg.lai[:, 1]
    lai_l = veg.lai[:, 0]
    gtransu = canopy_result.fvaput * 2.0 * lai_u * veg.fu
    gtransl = canopy_result.fvaplt * 2.0 * lai_l * veg.fl * (1.0 - fiold)
    ftrans = (gtransu[:, None] * veg.froot[None, :, 1]
              + gtransl[:, None] * veg.froot[None, :, 0])

    # --- Issue 13: Transpiration heat correction (Fortran soil.f lines 367-376) ---
    # F77 canopy.f:1269-1274 distributes per-layer extraction by
    # stress[k]/max(stresst, epsilon) — see findings #15 in STATUS.md.
    # When all layers are unstressed (zwilt=1 for all k), `stress[k]/stresst
    # == froot[k]` exactly in IEEE because `sum_k(froot[k]) == 1.0` exactly.
    # Use raw froot weighting when stresstu == 1.0 (fully saturated) to
    # preserve the bit-exact F77 match in the unstressed regime; otherwise
    # apply F77's stress-weighted redistribution.
    safe_stresstu = jnp.maximum(stresstu, c.epsilon)
    safe_stresstl = jnp.maximum(stresstl, c.epsilon)
    weightu = jnp.where(
        stresstu[:, None] >= 1.0,
        veg.froot[None, :, 1],
        stressu / safe_stresstu[:, None],
    )
    weightl = jnp.where(
        stresstl[:, None] >= 1.0,
        veg.froot[None, :, 0],
        stressl / safe_stresstl[:, None],
    )
    wsoi_pre_trans = wsoi_work
    tsoi_pre_trans = soil.tsoi
    for k in range(nsoilay):
        upsoiu_k = gtransu * weightu[:, k]
        upsoil_k = gtransl * weightl[:, k]
        wsoi_pre_trans = wsoi_pre_trans.at[:, k].add(
            -dtime * (upsoiu_k + upsoil_k)
            / (c.rhow * porosflo[:, k] * soil.hsoi[k])
        )
        # Floor wsoi at 0 (mirror F77 wadjust). Transpiration demand can exceed
        # available soil water in a single layer; negative wsoi then enters
        # soilh2o where wsoia**m_k (m_k≈13-15) and the implicit tridia solve
        # produce wild coefficients → NaN. F77 escapes via float32 underflow;
        # JAX-64 preserves the negative value. Y192M8 Hudson Bay diagnosis.
        wsoi_pre_trans = wsoi_pre_trans.at[:, k].set(
            jnp.maximum(wsoi_pre_trans[:, k], 0.0)
        )
        cx_k = (c0pud[:, k]
                + ((1.0 - soil.poros[:, k]) * csoi_mineral[:, k] * rhosoi[:, k]
                   + soil.poros[:, k] * (1.0 - wisoi_work[:, k]) * wsoi_pre_trans[:, k] * c.ch2o * c.rhow
                   + soil.poros[:, k] * wisoi_work[:, k] * c.cice * c.rhow)
                * soil.hsoi[k])
        tsoi_pre_trans = tsoi_pre_trans.at[:, k].add(
            -dtime * c.ch2o * (upsoiu_k * (tu_nv - tsoi_pre_trans[:, k])
                                + upsoil_k * (tl_nv - tsoi_pre_trans[:, k]))
            / jnp.maximum(cx_k, c.epsilon)
        )

    # Save old soil state for soilheat (after transpiration adjustments)
    otsoi = tsoi_pre_trans
    owsoi = wsoi_pre_trans

    # --- Issue 17: fhtop with rain/runoff heat (Fortran soil.f lines 322-324) ---
    fhtop_soil = (heatg_avg
                  + raing_avg * c.ch2o * (traing_avg - otsoi[:, 0])
                  - grunof * c.ch2o * jnp.maximum(c.tmelt - otsoi[:, 0], 0.0))

    # soilh2o — transpiration already extracted, so don't pass ftrans
    h2o = soilh2o(
        wsoi_pre_trans, wisoi_work, soil.poros, porosflo,
        soil.hydraul, soil.suction, soil.bex, soil.hsoi[:nsoilay],
        fwtop, fwpud, wpud_w,
        params.bperm, dtime, c.rhow, c.epsilon,
    )

    # --- Issue 18: Update c1pud after soilh2o ---
    c1pud = c0pud.at[:, 0].set(c.ch2o * h2o.wpud + c.cice * wipud_w)

    # soilheat
    heat = soilheat(
        tsoi_pre_trans, otsoi, owsoi,
        h2o.wsoi, wisoi_work, soil.poros,
        setsoi_res.consoi, csoi_mineral, rhosoi,
        soil.hsoi[:nsoilay],
        c0pud, c1pud,
        fhtop_soil, h2o.wflo,
        c.ch2o, c.cice, c.rhow, dtime,
    )

    # wadjust before freeze_thaw
    wsoi_pre, wisoi_pre, gadjust1 = wadjust(
        h2o.wsoi, wisoi_work, soil.poros,
        soil.hsoi[:nsoilay], c.rhow, dtime, c.epsilon,
    )

    # freeze_thaw
    tsoi_ft, wsoi_ft, wisoi_ft, porosflo_ft = freeze_thaw(
        heat.tsoi, wsoi_pre, wisoi_pre, soil.poros,
        csoi_mineral, rhosoi, soil.hsoi[:nsoilay],
        c1pud, c.ch2o, c.cice, c.rhow, c.hfus, c.tmelt, c.epsilon,
    )

    # --- Issue 15: Puddle freeze/thaw (Fortran soil.f lines 485-514) ---
    from ibis.physics.soil_physics import puddle_freeze_thaw
    tsoi_ft_0, wpud_ft, wipud_ft = puddle_freeze_thaw(
        tsoi_ft[:, 0], h2o.wpud, wipud_w,
        soil.poros[:, 0], wsoi_ft[:, 0], wisoi_ft[:, 0],
        csoi_mineral[:, 0], rhosoi[:, 0], soil.hsoi[0],
        c.ch2o, c.cice, c.rhow, c.hfus, c.tmelt,
    )
    tsoi_ft = tsoi_ft.at[:, 0].set(tsoi_ft_0)

    # wadjust after freeze_thaw
    wsoi_adj, wisoi_adj, gadjust2 = wadjust(
        wsoi_ft, wisoi_ft, soil.poros,
        soil.hsoi[:nsoilay], c.rhow, dtime, c.epsilon,
    )
    gadjust = gadjust1 + gadjust2

    # =====================================================================
    # 12. Physical T bound: clamp surface temperatures within ΔT of air T
    # =====================================================================
    # Standard practice in CLM (Bonan, damped Newton step) and JSBACHv3.11
    # (canopy heat capacity), applied here as Min's T-bound idea: surface
    # temperature (tg, tu, tl, ts, ti) can not deviate from air T by more
    # than T_MAX_DEV per timestep. This catches finding-#28-class divergence
    # at high-LAI tropical cells (LAI=12, fu=0.975 → eigenvalue > 1 in the
    # canopy iteration) without disabling biomass evolution. For normal
    # cells, |T_surf - T_air| << T_MAX_DEV is a no-op.
    #
    # Physical justification: surface T is tightly coupled to air T via
    # sensible heat exchange. Extreme tropical day: |T_surf - T_air| ~ 25K.
    # Polar winter: T_surf can be colder than T_air by ~15K. T_MAX_DEV=40K
    # is well outside the realistic envelope.
    # Override via IBIS_T_MAX_DEV (default 40 K). Set to 1e9 to disable.
    import os as _tmd
    T_MAX_DEV = float(_tmd.environ.get("IBIS_T_MAX_DEV", "40.0"))
    ta = state.atm.ta
    def _clamp_T(T_surf):
        return jnp.clip(T_surf, ta - T_MAX_DEV, ta + T_MAX_DEV)
    tu_nv = _clamp_T(tu_nv)
    ts_nv = _clamp_T(ts_nv)
    tl_nv = _clamp_T(tl_nv)
    tg = _clamp_T(tg)
    # ti is snow surface (only when snow fraction > 0); skip for now.

    # =====================================================================
    # 13. Pack results back into IBISState
    # =====================================================================

    # Updated canopy state
    new_canopy = CanopyState(
        tu=tu_nv, ts=ts_nv, tl=tl_nv,
        t12=t12, t34=t34,
        q12=q12, q34=q34,
        wliqu=wliqu_nv, wliqs=wliqs_nv, wliql=wliql_nv,
        wsnou=wsnou_nv, wsnos=wsnos_nv, wsnol=wsnol_nv,
        ci_upper=ci_upper, gs_upper=gs_upper, cs_upper=cs_upper,
        an_upper=an_upper, ag_upper=ag_upper,
        ci_lower=ci_lower, gs_lower=gs_lower, cs_lower=cs_lower,
        an_lower=an_lower, ag_lower=ag_lower,
        fwetu=fwet2.fwetu, fwets=fwet2.fwets, fwetl=fwet2.fwetl,
        rliqu=fwet2.rliqu, rliqs=fwet2.rliqs, rliql=fwet2.rliql,
        solu=rad.solu, sols=rad.sols, soll=rad.soll,
        solg=rad.solg, soli=rad.soli,
        firu=rad.firu, firs=rad.firs, firl=rad.firl,
        firg=canopy_result.firg, firi=canopy_result.firi,
        firb=canopy_result.firb,
        asurd=rad.asurd, asuri=rad.asuri,
        topparu=rad.topparu, topparl=rad.topparl,
        scalcoefu=rad.scalcoefu, scalcoefl=rad.scalcoefl,
        fsenu=canopy_result.fsenu, fsens=canopy_result.fsens,
        fsenl=canopy_result.fsenl, fseng=canopy_result.fseng,
        fseni=canopy_result.fseni,
        fvapg=canopy_result.fvapg, fvapi=canopy_result.fvapi,
        raing=raing_total, snowg=snowg_total,
        a10scalparamu=a10sp_u, a10daylightu=a10dl_u,
        a10scalparaml=a10sp_l, a10daylightl=a10dl_l,
        rhoa=aero.rhoa,
    )

    # Updated soil state (with puddle freeze/thaw results)
    new_soil = eqx.tree_at(
        lambda s: (s.tsoi, s.wsoi, s.wisoi, s.tg,
                   s.wpud, s.wipud, s.consoi, s.hflo,
                   s.heatg, s.hvasug, s.qglif,
                   s.stresstu, s.stresstl, s.stressu, s.stressl,
                   s.porosflo),
        soil,
        (tsoi_ft, wsoi_adj, wisoi_adj, tg,
         wpud_ft, wipud_ft, setsoi_res.consoi, heat.hflo,
         heatg_avg, setsoi_res.hvasug, setsoi_res.qglif,
         stresstu, stresstl, stressu, stressl,
         porosflo_ft),
    )

    # Updated snow state (includes persistent tlsub)
    new_snow = eqx.tree_at(
        lambda s: (s.fi, s.tsno, s.hsno, s.ti, s.tlsub, s.heati, s.hvasui),
        snow,
        (fi_new, tsno_va, hsno_final, ti, tlsub_out, heati, setsoi_res.hvasui),
    )

    # Updated hydrology (grunof = puddle overflow + rain runoff + squeeze)
    new_hydro = HydrologyDiag(
        ginvap=canopy_result.ginvap,
        gsuvap=canopy_result.gsuvap,
        gtrans=canopy_result.gtrans,
        gtransu=gtransu,
        gtransl=gtransl,
        grunof=grunof + h2o.fsqueez,
        gdrain=jnp.zeros((npoi,)),    # Bottom drainage
        gadjust=gadjust,
        wtot=jnp.zeros((npoi,)),      # Computed in accounting
        # Diagnostic exposures for day-84 shock investigation
        raing_avg=raing_avg,
        traing_avg=traing_avg,
        fhtop_soil=fhtop_soil,
        rainl=casc.rainl,
        snowl=casc.snowl,
        trainl=casc.trainl,
        tsnowl=casc.tsnowl,
        sflo_bot=snheat.sflo[:, -1],
        fhtop_snow=fhtop_snow,
    )

    # Updated atmosphere (xstore, iwet unchanged within timestep)
    new_atm = atm

    # Assemble full state
    new_state = eqx.tree_at(
        lambda s: (s.canopy, s.soil, s.snow, s.hydrology, s.atm),
        state,
        (new_canopy, new_soil, new_snow, new_hydro, new_atm),
    )

    return new_state


# ---------------------------------------------------------------------------
# run_day: lax.scan over timesteps within a day
# ---------------------------------------------------------------------------

def run_day(
    state: IBISState,
    consts: PhysicalConstants,
    params: SimParams,
    daily_weather: DailyWeather,
    plens: Float[Array, "npoi"],
    startp: Float[Array, "npoi"],
    endp: Float[Array, "npoi"],
    jday: int,
    key: jax.Array,
    dtime: float,
    nband: int = 2,
) -> tuple[IBISState, jax.Array]:
    """Run all timesteps within one day.

    Uses jax.lax.scan over the timestep dimension.

    Parameters
    ----------
    state : IBISState
        State at start of day.
    consts, params : PhysicalConstants, SimParams
        Constants and parameters.
    daily_weather : DailyWeather
        Daily forcing from the weather generator.
    plens, startp, endp : arrays
        Precipitation timing (duration, start, end of event).
    jday : int
        Julian day (1-based).
    key : PRNGKey
        Random key for diurnal variation.
    dtime : float
        Timestep in seconds.
    nband : int
        Number of wavebands (default 2).

    Returns
    -------
    (IBISState, PRNGKey)
        Updated state at end of day and consumed key.
    """
    nsteps = timesteps_per_day(dtime)

    def _step_body(carry, istep):
        st, k = carry
        k, subkey = jax.random.split(k)

        time = istep * dtime  # seconds since midnight

        # Generate hourly forcing
        _, hourly = diurnal(
            time, jday, daily_weather, params.latitude,
            plens, startp, endp, subkey, dtime, nband,
        )

        # Update atmospheric forcing in state
        new_atm = eqx.tree_at(
            lambda a: (a.ta, a.qa, a.ua, a.coszen,
                       a.solad, a.solai, a.fira,
                       a.raina, a.snowa),
            st.atm,
            (hourly.ta, hourly.qa, hourly.ua, hourly.coszen,
             hourly.solad, hourly.solai, hourly.fira,
             hourly.raina, hourly.snowa),
        )
        st = eqx.tree_at(lambda s: s.atm, st, new_atm)

        # Run one timestep
        st = timestep_body(st, consts, params, dtime)
        return (st, k), None

    (state_out, key_out), _ = jax.lax.scan(
        _step_body,
        (state, key),
        jnp.arange(nsteps),
    )

    return state_out, key_out


# ---------------------------------------------------------------------------
# run_month: lax.scan over days within a month
# ---------------------------------------------------------------------------

def run_month(
    state: IBISState,
    consts: PhysicalConstants,
    params: SimParams,
    climate: ClimateData,
    imonth: int,
    key: jax.Array,
    dtime: float,
    leap: bool = False,
    nband: int = 2,
) -> tuple[IBISState, jax.Array]:
    """Run all days within one month.

    Uses jax.lax.scan over the day dimension.

    Parameters
    ----------
    state : IBISState
        State at start of month.
    consts, params : PhysicalConstants, SimParams
        Constants and parameters.
    climate : ClimateData
        Monthly climatology for weather generation.
    imonth : int
        Month index (0 = January).
    key : PRNGKey
        Random key.
    dtime : float
        Timestep in seconds.
    leap : bool
        Whether it's a leap year.
    nband : int
        Number of wavebands.

    Returns
    -------
    (IBISState, PRNGKey)
        Updated state at end of month and consumed key.
    """
    ndays = days_in_month(imonth, leap)

    # Generate monthly precipitation pattern
    key, mprecip = generate_month_precip(key, climate, imonth)

    def _day_body(carry, iday):
        st, k, xstore = carry
        k, dk1, dk2 = jax.random.split(k, 3)

        # Generate daily weather
        _, dw = daily(dk1, imonth, iday, climate, mprecip, xstore)

        # Precipitation timing
        _, plens, startp, endp = compute_precip_timing(dk2, dw.precip, dtime)

        # Julian day
        jday = julian_day(imonth, iday, leap)

        # Run all timesteps for this day
        k, dk3 = jax.random.split(k)
        st, dk3 = run_day(
            st, consts, params, dw,
            plens, startp, endp, jday, dk3, dtime, nband,
        )

        return (st, k, dw.xstore), None

    xstore_init = state.atm.xstore
    (state_out, key_out, xstore_out), _ = jax.lax.scan(
        _day_body,
        (state, key, xstore_init),
        jnp.arange(ndays),
    )

    # Store updated xstore in atmosphere state
    state_out = eqx.tree_at(lambda s: s.atm.xstore, state_out, xstore_out)

    return state_out, key_out


# ---------------------------------------------------------------------------
# run_year: Python loop over months
# ---------------------------------------------------------------------------

def run_year(
    state: IBISState,
    consts: PhysicalConstants,
    params: SimParams,
    climate: ClimateData,
    key: jax.Array,
    dtime: float,
    year: int = 2000,
    nband: int = 2,
) -> tuple[IBISState, jax.Array]:
    """Run all 12 months of one year.

    Uses a Python loop over months (only 12 iterations, allows I/O
    between months for data loading).

    Parameters
    ----------
    state : IBISState
        State at start of year.
    consts, params : PhysicalConstants, SimParams
        Constants and parameters.
    climate : ClimateData
        Monthly climatology.
    key : PRNGKey
        Random key.
    dtime : float
        Timestep in seconds.
    year : int
        Year (for leap year computation).
    nband : int
        Number of wavebands.

    Returns
    -------
    (IBISState, PRNGKey)
        Updated state and key at end of year.
    """
    from ibis.timestepping.calendar import is_leap_year
    leap = is_leap_year(year)

    for imonth in range(12):
        state, key = run_month(
            state, consts, params, climate, imonth, key,
            dtime, leap, nband,
        )

    return state, key
