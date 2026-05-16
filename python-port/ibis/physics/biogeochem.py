"""Soil biogeochemistry (CENTURY-like decomposition) for IBIS-JAX.

Ported from biogeochem.f90: soilbgc subroutine.

Advances soil carbon and nitrogen pools by one day using the modified
Verberne/CENTURY decomposition scheme.  Handles litter inputs, microbial
growth, turnover of slow and passive pools, leaching, and N mineralisation.
Called once per day.
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from ibis.state.carbon import CarbonState
from ibis.state.vegetation import VegetationState
from ibis.state.summary import SummaryState
from ibis.io.parameters import BGCParams


def soilbgc(
    carbon: CarbonState,
    veg: VegetationState,
    summary: SummaryState,
    hsoi: Float[Array, "nsoilay+1"],
    sand: Float[Array, "npoi nsoilay"],
    clay: Float[Array, "npoi nsoilay"],
    bgc: BGCParams,
    ndaypy: int,
) -> tuple[CarbonState, VegetationState, SummaryState]:
    """Advance soil C/N pools by one day.

    Parameters
    ----------
    carbon : CarbonState
        Current C pool state.
    veg : VegetationState
        Vegetation state (for tco2mic, tnmin diagnostics).
    summary : SummaryState
        Summary accumulators (for storedn, yrleach, ynleach).
    hsoi : (nsoilay+1,)
        Soil layer thicknesses (m).
    sand, clay : (npoi, nsoilay)
        Sand and clay percentages.
    bgc : BGCParams
        Biogeochemistry rate constants.
    ndaypy : int
        Number of days per year.

    Returns
    -------
    (CarbonState, VegetationState, SummaryState)
        Updated states.
    """
    totts = 1.0 / ndaypy

    # --- Litter split fractions (Verberne / Geoderma 1997) ---
    fracll = bgc.fmax * bgc.cnleaf**2 / (bgc.rconst + bgc.cnleaf**2)
    fracls = (1.0/bgc.cnleaf - fracll/bgc.cnr[4] - (1.0-fracll)/bgc.cnr[6]) / \
             (1.0/bgc.cnr[5] - 1.0/bgc.cnr[6])
    fraclm = 1.0 - fracll - fracls

    fracrl = bgc.fmax * bgc.cnroot**2 / (bgc.rconst + bgc.cnroot**2)
    fracrs = (1.0/bgc.cnroot - fracrl/bgc.cnr[4] - (1.0-fracrl)/bgc.cnr[6]) / \
             (1.0/bgc.cnr[5] - 1.0/bgc.cnr[6])
    fracrm = 1.0 - fracrl - fracrs

    fracwl = bgc.fmax * bgc.cnwood**2 / (bgc.rconst + bgc.cnwood**2)
    fracws = (1.0/bgc.cnwood - fracwl/bgc.cnr[4] - (1.0-fracwl)/bgc.cnr[6]) / \
             (1.0/bgc.cnr[7] - 1.0/bgc.cnr[6])
    fracwm = 1.0 - fracwl - fracws

    # --- Texture factors (top 1 m = 4 layers) ---
    rdepth = 1.0 / (hsoi[0] + hsoi[1] + hsoi[2] + hsoi[3])
    cfrac   = jnp.sum(0.01 * clay[:, :4] * hsoi[:4], axis=1) * rdepth
    texfact = jnp.sum(0.01 * sand[:, :4] * hsoi[:4], axis=1) * rdepth

    fbpom = 0.50  # protected fraction of decomposing biomass

    # --- Total litterfall input ---
    totcin = (carbon.falll + carbon.fallr + carbon.fallw) * totts

    # --- Total C before decomposition ---
    totcbegin = (
        carbon.clitlm + carbon.clitls + carbon.clitrm + carbon.clitrs +
        carbon.clitwm + carbon.clitws + carbon.csoislop + carbon.csoislon +
        carbon.csoipas + carbon.totcmic + carbon.clitll + carbon.clitrl + carbon.clitwl
    )

    # --- Microbial biomass split: protected / non-protected ---
    cmicmx = bgc.fbsom * totcbegin
    cmicp = jnp.minimum(carbon.totcmic, cmicmx)
    cmicn = jnp.maximum(0.0, carbon.totcmic - cmicmx)

    # --- Add daily litterfall to litter pools ---
    clitlm = carbon.clitlm + fraclm * carbon.falll * totts
    clitls = carbon.clitls + fracls * carbon.falll * totts
    clitll = carbon.clitll + fracll * carbon.falll * totts
    clitrm = carbon.clitrm + fracrm * carbon.fallr * totts
    clitrs = carbon.clitrs + fracrs * carbon.fallr * totts
    clitrl = carbon.clitrl + fracrl * carbon.fallr * totts
    clitwm = carbon.clitwm + fracwm * carbon.fallw * totts
    clitws = carbon.clitws + fracws * carbon.fallw * totts
    clitwl = carbon.clitwl + fracwl * carbon.fallw * totts

    decompl = carbon.decompl
    decomps = carbon.decomps

    # --- Decomposition outflows ---
    outcrs = jnp.minimum(decomps * bgc.krs * clitrs, clitrs)
    outcws = jnp.minimum(decompl * bgc.kws * clitws, clitws)
    outcls = jnp.minimum(decompl * bgc.kls * clitls, clitls)
    outclm = jnp.minimum(decompl * bgc.klm * clitlm, clitlm)
    outcrm = jnp.minimum(decomps * bgc.krm * clitrm, clitrm)
    outcwm = jnp.minimum(decompl * bgc.kwm * clitwm, clitwm)
    outcnb = jnp.minimum(decomps * bgc.knb * carbon.csoislon, carbon.csoislon)
    outcpb = jnp.minimum(decomps * bgc.kpb * carbon.csoislop, carbon.csoislop)
    outcsb = jnp.minimum(decomps * bgc.ksb * carbon.csoipas, carbon.csoipas)

    # Microbial biomass turnover
    outcbp = jnp.minimum(bgc.kbp * cmicp, cmicp) * bgc.effac
    outcbn = jnp.minimum(bgc.kbn * cmicn, cmicn) * bgc.effac

    # Net change in microbial biomass
    dbdt = (outcrs*bgc.yrs + outcws*bgc.yws + outcls*bgc.yls +
            outclm*bgc.ylm + outcrm*bgc.yrm + outcwm*bgc.ywm +
            outcnb*bgc.ynb + outcpb*bgc.ypb + outcsb*bgc.ysb
            - outcbp - outcbn)

    # Lignin outflows
    outcll = jnp.minimum(decompl * bgc.kll * clitll, clitll)
    outcrl = jnp.minimum(decomps * bgc.krl * clitrl, clitrl)
    outcwl = jnp.minimum(decompl * bgc.kwl * clitwl, clitwl)
    outcns = jnp.minimum(decomps * bgc.kns * carbon.csoislon, carbon.csoislon)

    # Change in non-protected slow OM
    dcndt = (bgc.lig_frac * (outcll*bgc.yll + outcrl*bgc.yrl + outcwl*bgc.ywl) +
             (1.0 - fbpom) * (bgc.ybn*outcbn + bgc.ybp*outcbp) -
             outcnb - outcns)

    # Change in protected slow OM
    outcps = jnp.minimum(decomps * bgc.kps * carbon.csoislop, carbon.csoislop)

    dcpdt = (bgc.lig_frac * (outcll*bgc.yll + outcrl*bgc.yrl + outcwl*bgc.ywl) +
             fbpom * (bgc.ybn*outcbn + bgc.ybp*outcbp) -
             outcpb - outcps)

    # Leaching
    h20 = 0.30e-03
    fleach = h20 / 18.0 * (0.01 + 0.04 * texfact)

    # Change in passive (stabilised) OM
    dcsdt = (bgc.yns*outcns + bgc.yps*outcps) - outcsb - fleach*carbon.csoipas

    cleach = fleach * (carbon.csoipas + carbon.csoislop + carbon.csoislon)

    ynleach = (summary.ynleach +
               fleach*carbon.csoipas/bgc.cnr[1] +
               fleach*carbon.csoislop/bgc.cnr[2] +
               fleach*carbon.csoislon/bgc.cnr[3])

    dcndt = dcndt - fleach * carbon.csoislon
    dcpdt = dcpdt - fleach * carbon.csoislop

    yrleach = summary.yrleach + cleach

    # --- N mineralisation / immobilisation ---
    # Helper: net N from decomposition of pool
    def _nmin_imm(y, cnr_from, cnr_to, outflux):
        net = (1.0/cnr_to - y/cnr_from) * outflux
        nmin = jnp.where(y/cnr_from > 1.0/cnr_to, 0.0, net)
        nimm = jnp.where(y/cnr_from > 1.0/cnr_to, net, 0.0)
        return nmin, nimm

    nminrs, nbiors = _nmin_imm(bgc.yrs, bgc.cnr[0], bgc.cnr[5], outcrs)
    nminls, nbiols = _nmin_imm(bgc.yls, bgc.cnr[0], bgc.cnr[5], outcls)
    nminws, nbiows = _nmin_imm(bgc.yws, bgc.cnr[0], bgc.cnr[7], outcws)
    nminwm, nbiowm = _nmin_imm(bgc.ywm, bgc.cnr[0], bgc.cnr[7], outcwm)
    nminlm, nbiolm = _nmin_imm(bgc.ylm, bgc.cnr[0], bgc.cnr[6], outclm)
    nminrm, nbiorm = _nmin_imm(bgc.yrm, bgc.cnr[0], bgc.cnr[6], outcrm)
    nminslon, nbioslon = _nmin_imm(bgc.ynb, bgc.cnr[0], bgc.cnr[3], outcnb)
    nminslop, nbioslop = _nmin_imm(bgc.ypb, bgc.cnr[0], bgc.cnr[2], outcpb)
    nminpas, nbiopas = _nmin_imm(bgc.ysb, bgc.cnr[0], bgc.cnr[1], outcsb)

    totimm = (nbiors + nbiols + nbiows + nbiowm +
              nbiolm + nbiorm + nbioslon + nbioslop + nbiopas)
    totmin = (nminrs + nminls + nminws + nminwm +
              nminlm + nminrm + nminslon + nminslop + nminpas)

    nrelps = outcps * (1.0/bgc.cnr[2] - 1.0/bgc.cnr[1])
    nrelns = outcns * (1.0/bgc.cnr[3] - 1.0/bgc.cnr[1])
    nrelbn = ((1.0-fbpom)*outcbn*(1.0/bgc.cnr[0] - 1.0/bgc.cnr[3]) +
              (1.0-fbpom)*outcbp*(1.0/bgc.cnr[0] - 1.0/bgc.cnr[3]))
    nrelbp = (fbpom*outcbp*(1.0/bgc.cnr[0] - 1.0/bgc.cnr[2]) +
              fbpom*outcbn*(1.0/bgc.cnr[0] - 1.0/bgc.cnr[2]))
    nrelll = (bgc.lig_frac*outcll*(1.0/bgc.cnr[4] - 1.0/bgc.cnr[2]) +
              bgc.lig_frac*outcll*(1.0/bgc.cnr[4] - 1.0/bgc.cnr[3]))
    nrelrl = (bgc.lig_frac*outcrl*(1.0/bgc.cnr[4] - 1.0/bgc.cnr[2]) +
              bgc.lig_frac*outcrl*(1.0/bgc.cnr[4] - 1.0/bgc.cnr[3]))
    nrelwl = (bgc.lig_frac*outcwl*(1.0/bgc.cnr[4] - 1.0/bgc.cnr[2]) +
              bgc.lig_frac*outcwl*(1.0/bgc.cnr[4] - 1.0/bgc.cnr[3]))

    totnrel = nrelps + nrelns + nrelbn + nrelbp + nrelll + nrelrl + nrelwl
    netmin = totmin + totimm + totnrel
    tnmin = jnp.maximum(0.0, netmin) / (86400.0 * 0.014)

    # --- Update C pools ---
    totcmic_new = jnp.maximum(0.0, carbon.totcmic + dbdt)
    csoislon_new = jnp.maximum(0.0, carbon.csoislon + dcndt)
    csoislop_new = jnp.maximum(0.0, carbon.csoislop + dcpdt)
    csoipas_new = jnp.maximum(0.0, carbon.csoipas + dcsdt)
    clitlm = jnp.maximum(0.0, clitlm - outclm)
    clitls = jnp.maximum(0.0, clitls - outcls)
    clitll = jnp.maximum(0.0, clitll - outcll)
    clitrm = jnp.maximum(0.0, clitrm - outcrm)
    clitrs = jnp.maximum(0.0, clitrs - outcrs)
    clitrl = jnp.maximum(0.0, clitrl - outcrl)
    clitwm = jnp.maximum(0.0, clitwm - outcwm)
    clitws = jnp.maximum(0.0, clitws - outcws)
    clitwl = jnp.maximum(0.0, clitwl - outcwl)

    # --- Litter and SOM summaries ---
    totlit = (clitlm + clitls + clitll +
              clitrm + clitrs + clitrl +
              clitwm + clitws + clitwl)
    totalit = clitlm + clitls + clitwm + clitll + clitws + clitwl
    totrlit = clitrm + clitrs + clitrl
    totcsoi = csoipas_new + csoislop_new + totcmic_new + csoislon_new
    totfall = carbon.falll + carbon.fallr + carbon.fallw

    # --- CO2 flux from decomposition ---
    totcend = totlit + totcsoi
    tco2mic = (totcbegin + totcin - totcend - cleach) / (86400.0 * 0.012)

    # --- Update stored N ---
    storedn = summary.storedn + tnmin

    # --- Pack results ---
    new_carbon = eqx.tree_at(
        lambda c: (c.clitlm, c.clitls, c.clitll,
                   c.clitrm, c.clitrs, c.clitrl,
                   c.clitwm, c.clitws, c.clitwl,
                   c.csoipas, c.csoislon, c.csoislop,
                   c.totcmic,
                   c.totlit, c.totalit, c.totrlit,
                   c.totcsoi, c.totfall),
        carbon,
        (clitlm, clitls, clitll,
         clitrm, clitrs, clitrl,
         clitwm, clitws, clitwl,
         csoipas_new, csoislon_new, csoislop_new,
         totcmic_new,
         totlit, totalit, totrlit,
         totcsoi, totfall),
    )

    new_veg = eqx.tree_at(
        lambda v: (v.tco2mic, v.tnmin),
        veg,
        (tco2mic, tnmin),
    )

    new_summary = eqx.tree_at(
        lambda s: (s.storedn, s.yrleach, s.ynleach),
        summary,
        (storedn, yrleach, ynleach),
    )

    return new_carbon, new_veg, new_summary
