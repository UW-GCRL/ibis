"""PFT and BGC parameter containers with default values.

Ported from readpars.f90 / mod_pft_params.f90 / mod_bgc_params.f90.
All parameters are stored as frozen equinox Modules so they can be
passed through jit/vmap/grad.

Default values are taken from the original IBIS parameter files
(params.can, params.veg, params.soi).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class PFTParams(eqx.Module):
    """Per-PFT bioclimatic and allocation parameters.

    All per-PFT arrays have shape (npft=12,).
    """

    # Bioclimatic limits for PFT existence (from params.veg)
    TminL: Float[Array, "npft"]     # Lower bound for coldest-month temperature
    TminU: Float[Array, "npft"]     # Upper bound for coldest-month temperature
    Twarm: Float[Array, "npft"]     # Warm-month threshold
    GDD: Float[Array, "npft"]       # Growing degree day threshold

    # Turnover time scales (from params.can)
    tauleaf: Float[Array, "npft"]   # Leaf turnover (years)
    tauroot: Float[Array, "npft"]   # Root turnover (years)
    tauwood0: Float[Array, "npft"]  # Wood turnover base (years)

    # Allocation fractions (from params.can)
    specla: Float[Array, "npft"]    # Specific leaf area (m2/kgC)
    aleaf: Float[Array, "npft"]     # C allocation to leaves
    aroot: Float[Array, "npft"]     # C allocation to roots
    awood: Float[Array, "npft"]     # C allocation to wood

    # Vmax (from params.can)
    vmax_pft: Float[Array, "npft"]  # Max carboxylation rate (mol/m2/s)

    # Minimum LAI
    xminlai: float = 0.01

    # Canopy closure normalization
    woodnorm: float = 7.5
    #: Canopy fractional cover (docs/CALIBRATION.md 14). IBIS 2.6b4 sets
    #: lower-canopy cover fl = totlail (one unit of grass LAI = 100% cover) and
    #: clips both fu and fl to [0.25, 0.975], so a sparse cell carries >= 25%
    #: cover in each canopy with no leaves at all. cover_k > 0 replaces the
    #: linear rule by fl = 1 - exp(-cover_k * totlail); 0.0 keeps F77.
    cover_k: float = 0.0
    fl_min: float = 0.25
    fu_min: float = 0.25
    #: Wood-to-cover scale for the upper canopy: fu = (1 - exp(-wood / fu_k)) /
    #: (1 - exp(-woodnorm / fu_k)). F77 is fu_k = 1.0, which puts trees with
    #: 0.9 kgC m-2 of wood over 60% of a cell (docs/CALIBRATION.md 14).
    fu_k: float = 1.0
    #: Water-limited tree establishment (docs/CALIBRATION.md 14): upper-canopy
    #: survival = clip((annual-mean stresstu - tree_stress_min) / tree_stress_ramp,
    #: 0, 1). -1.0 never binds: F77 has no water limit on where trees grow.
    tree_stress_min: float = -1.0
    tree_stress_ramp: float = 0.2


def default_pft_params() -> PFTParams:
    """Create PFTParams with hardcoded default values from IBIS params files."""
    return PFTParams(
        TminL=jnp.array([
            0.0, 0.0, -10.0, -45.0, -45.0, -57.5,
            -57.5, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
        ]),
        TminU=jnp.array([
            9999.0, 9999.0, 0.0, 0.0, 0.0, -45.0,
            -45.0, -45.0, 9999.0, 9999.0, 9999.0, 9999.0,
        ]),
        Twarm=jnp.array([
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 22.0, 9999.0,
        ]),
        GDD=jnp.array([
            9999.0, 9999.0, 9999.0, 1200.0, 1200.0, 350.0,
            350.0, 350.0, 100.0, 100.0, 100.0, 100.0,
        ]),
        tauleaf=jnp.array([
            1.01, 1.0, 1.0, 2.0, 1.0, 2.5,
            1.0, 1.0, 1.5, 1.0, 1.25, 1.5,
        ]),
        tauroot=jnp.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]),
        tauwood0=jnp.array([
            25.0, 25.0, 25.0, 50.0, 50.0, 100.0,
            100.0, 100.0, 5.0, 5.0, 999.0, 999.0,
        ]),
        specla=jnp.array([
            # SLA (m2/kgC) — CMA-ES optimized against ILAMB (GPP+LAI+ET+Reco)
            13.0, 21.6,   # PFT 0-1: tropical broadleaf
            30.0,         # PFT 2: warm-temperate broadleaf (capped from 33.4)
            10.8,         # PFT 3: temperate conifer evergreen
            35.0,         # PFT 4: temperate broadleaf deciduous (capped from 39.4)
            6.6,          # PFT 5: boreal conifer evergreen
            30.0, 10.4,   # PFT 6-7: boreal broadleaf dec / conifer dec
            9.7, 15.1,    # PFT 8-9: evergreen / deciduous shrub
            15.7, 18.6,   # PFT 10-11: C4 / C3 grass
        ]),
        aleaf=jnp.array([
            0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
            0.30, 0.30, 0.45, 0.45, 0.45, 0.45,
        ]),
        aroot=jnp.array([
            0.20, 0.20, 0.20, 0.40, 0.20, 0.40,
            0.20, 0.20, 0.40, 0.35, 0.55, 0.55,
        ]),
        awood=jnp.array([
            0.50, 0.50, 0.50, 0.30, 0.50, 0.30,
            0.50, 0.50, 0.15, 0.20, 0.00, 0.00,
        ]),
        vmax_pft=jnp.array([
            # Vmax (mol CO2/m2/s) — CMA-ES optimized against ILAMB
            # Manual corrections applied where CMA-ES hit unphysical values
            32.0e-6, 80.0e-6,   # PFT 0-1: trop evergreen/deciduous (CMA: 31.8, 121.8→capped 80)
            25.0e-6,             # PFT 2: warm-temperate broadleaf (CMA: 24.9)
            43.0e-6, 48.0e-6,   # PFT 3-4: temperate conifer/broadleaf (CMA: 43.3, 47.7)
            31.5e-6,             # PFT 5: boreal conifer (CMA: 31.5)
            18.0e-6, 15.0e-6,   # PFT 6-7: boreal broadleaf/conifer dec (CMA: 18.1, 11→15)
            13.3e-6, 36.0e-6,   # PFT 8-9: shrubs (CMA: 13.3, 36.2)
            10.0e-6, 15.0e-6,   # PFT 10-11: C4 / C3 grass (CMA: 4.5→10, 11→15)
        ]),
    )


class BGCParams(eqx.Module):
    """Biogeochemistry rate constants for CENTURY-like decomposition.

    All values from mod_bgc_params.f90 / params.soi.
    """

    # Litter chemistry
    lig_frac: float = 0.50       # Lignin split between protected/non-protected
    fbsom: float = 0.017         # Protected biomass fraction of total SOC
    effac: float = 0.40          # Microbial biomass re-incorporation efficiency

    # Verberne model constants
    fmax: float = 0.45           # Maximum resistant fraction
    rconst: float = 1200.0       # Verberne model constant

    # Litter C:N ratios
    cnleaf: float = 40.0         # Average C:N of leaf litterfall
    cnroot: float = 60.0         # Average C:N of root turnover
    cnwood: float = 200.0        # Average C:N of woody debris

    # C:N ratios for pools (8 pools)
    cnr: Float[Array, "8"] = eqx.field(
        default_factory=lambda: jnp.array([
            8.0,   # 1: microbial biomass
            15.0,  # 2: passive soil C
            10.0,  # 3: protected slow soil C
            15.0,  # 4: non-protected slow soil C
            100.0, # 5: resistant litter (lignin)
            150.0, # 6: structural plant litter
            6.0,   # 7: metabolic plant litter
            250.0, # 8: woody biomass
        ])
    )

    # Decay rate constants (per day)
    klm: float = 0.15       # leaf metabolic → microbial
    kls: float = 0.01       # leaf structural → microbial
    kll: float = 0.01       # leaf lignin → non/protected OM
    krm: float = 0.10       # root metabolic → microbial
    krs: float = 0.005      # root structural → microbial
    krl: float = 0.005      # root lignin → non/protected OM
    kwm: float = 0.001      # wood metabolic → microbial
    kws: float = 0.001      # wood structural → microbial
    kwl: float = 0.001      # wood lignin → non/protected OM
    kbn: float = 0.045      # biomass → non-protected OM
    kbp: float = 0.005      # biomass → protected OM
    knb: float = 0.001      # non-protected OM → biomass
    kns: float = 1.0e-6     # non-protected → stabilised
    kpb: float = 0.0001     # protected OM → biomass
    kps: float = 1.0e-6     # protected → stabilised
    ksb: float = 8.0e-7     # stabilised → biomass

    # Decomposition efficiencies
    ylm: float = 0.4
    yls: float = 0.3
    yll: float = 1.0
    yrm: float = 0.4
    yrs: float = 0.3
    yrl: float = 1.0
    ywm: float = 0.4
    yws: float = 0.3
    ywl: float = 1.0
    ybn: float = 1.0
    ybp: float = 1.0
    yps: float = 1.0
    yns: float = 1.0
    ysb: float = 0.20
    ypb: float = 0.20
    ynb: float = 0.25
    #: Respiration constants that F77 hard-codes (NEE stage, CALIBRATION.md 36); defaults = F77, bit-identical.
    lt_e0: float = 344.0        # Lloyd-Taylor activation parameter (K) of the heterotrophic decomposition factor; a scalar or a (12,) per-PFT vector (LAI-weighted per cell, CALIBRATION.md 37)
    wfps_opt: float = 60.0      # water-filled pore space (%) at the optimum of the moisture factor (shifts both branches)
    rgrowth: float = 0.30       # growth respiration, fraction of (GPP - maintenance respiration); scalar or (12,) per PFT (CALIBRATION.md 37)
    k_scale: float = 1.0        # multiplier on both decomposition climate factors (= on every litter and SOM rate); 1 = F77; may be an (npoi,) array
    rwood: float = 0.0125       # maintenance respiration of sapwood, fraction of sapwood carbon per year at 15 C (F77 stats.f); scalar or (12,) per PFT (CALIBRATION.md 37)
    rroot: float = 1.25         # maintenance respiration of fine roots, fraction of root carbon per year at 15 C (F77); scalar or (12,) per PFT (CALIBRATION.md 37)


class CropParams(eqx.Module):
    """Cropland as managed grass (docs/CALIBRATION.md 41). The cropland share of a cell (`SimParams.cropfrac`, from the
    ESA CCI map) is land that trees may not occupy and whose grass is harvested: each year the tree pools are scaled so the
    upper-canopy cover stays within the non-cropland share (the removed carbon is a clearing flux), and the cropland share
    of the grass leaf turnover leaves the field as harvest instead of becoming litter. The grasses on that share
    photosynthesise with their vmax times `vmax_scale_c3` / `vmax_scale_c4`, fitted at the C3 and the maize towers."""
    harvest_index: float = 0.45      # share of the cropland grass leaf turnover removed as harvest (grain + residue export)
    vmax_scale_c3: float = 1.0       # crop / natural-grass ratio of photosynthetic capacity on the cropland share, C3 crops (PFT 11)
    vmax_scale_c4: float = 1.0       # the same for C4 crops (maize, sorghum, sugar cane; PFT 10)
    #: Which crop a cell's cropland grows. -1 (the tower setting): the C4 scale on the C4 grass and the C3 scale on the
    #: C3 grass, so a tower prescribed as C4 grass is a maize field. 0..1 (the global setting): the share of the
    #: cropland that is C4 crops; both grasses on the cropland share get (1 - share) x C3 + share x C4, because the
    #: model's grass competition (C4 wherever the warmest month exceeds 22 C) says nothing about what is sown there.
    c4_share: float = -1.0


def default_crop_params() -> CropParams:
    return CropParams()


def default_bgc_params() -> BGCParams:
    """Create BGCParams with default values."""
    return BGCParams()
