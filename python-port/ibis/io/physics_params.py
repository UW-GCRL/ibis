"""Physics parameter loaders for IBIS-JAX.

Reads configs/params/default.yaml (or a user-supplied params_file) and
builds the parameter dataclasses used by the simulation.  All loaders
fall back to hardcoded defaults when a section or key is absent, so
existing configs without params_file continue to work unchanged.

Public API
----------
load_physics_params(path) -> dict
build_stomata_params(d)   -> StomataParams
build_veg_rad_params(d)   -> VegRadParams
build_canopy_params(d)    -> CanopyParams
build_interception_params(d) -> InterceptionParams
build_pft_params(d)       -> PFTParams
build_bgc_params(d)       -> BGCParams
build_sim_params(npoi, lat_rad, params_dict) -> SimParams
build_all_params(npoi, lat_rad, params_dict) -> (SimParams, PFTParams, BGCParams)
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from ibis.physics.radiation import VegRadParams, default_veg_rad_params
from ibis.physics.canopy_exchange import CanopyParams
from ibis.physics.interception import InterceptionParams
from ibis.physics.stomata import StomataParams
from ibis.io.parameters import PFTParams, BGCParams, default_pft_params, default_bgc_params
from ibis.io.soil_init import CSOI_VAL, RHOSOI_VAL


# ---------------------------------------------------------------------------
# Fixed name → index mappings (must match parameter dataclass order)
# ---------------------------------------------------------------------------

PFT_NAMES = [
    'tropical_broadleaf_evergreen',
    'tropical_broadleaf_deciduous',
    'warm_temperate_broadleaf_evergreen',
    'temperate_conifer_evergreen',
    'temperate_broadleaf_deciduous',
    'boreal_conifer_evergreen',
    'boreal_broadleaf_deciduous',
    'boreal_conifer_deciduous',
    'evergreen_shrub',
    'deciduous_shrub',
    'c4_grass',
    'c3_grass',
]

# Short PFT names used by the calibration harness (verify/calibration/fit_carbon.py PFT_NAMES); per-PFT yaml
# tables (coefm_pft, acclim_s0_pft, acclim_smax_pft) may use either spelling.
PFT_SHORT_NAMES = [
    'trop_bl_ever', 'trop_bl_dec', 'warm_bl_ever', 'temp_con_ever', 'temp_bl_dec', 'boreal_con_ever',
    'boreal_bl_dec', 'boreal_con_dec', 'ever_shrub', 'dec_shrub', 'c4_grass', 'c3_grass',
]
_PFT_INDEX = {**{n: i for i, n in enumerate(PFT_NAMES)}, **{n: i for i, n in enumerate(PFT_SHORT_NAMES)}}


def _pft_vector(table, defaults):
    """12-vector from a yaml table keyed by PFT name (long or short spelling); unknown names are an error."""
    out = list(defaults)
    for name, v in (table or {}).items():
        if name not in _PFT_INDEX:
            raise KeyError(f'unknown PFT name {name!r} in per-PFT table; use one of {PFT_NAMES} or {PFT_SHORT_NAMES}')
        out[_PFT_INDEX[name]] = float(v)
    return jnp.array(out)


STOMATA_GROUPS = ['broadleaf', 'conifer', 'shrub', 'c4_grass', 'c3_grass']
_PFT_SLOPE_GROUP = (0, 0, 0, 1, 0, 1, 0, 1, 2, 2, 3, 4)     # PFT -> Ball-Berry group (stomata.PFT_SLOPE_GROUP)

BGC_POOL_NAMES = [
    'microbial_biomass',
    'passive_soc',
    'protected_slow_soc',
    'nonprotected_slow_soc',
    'resistant_litter',
    'structural_litter',
    'metabolic_litter',
    'woody_biomass',
]

# Default vmax_pft values (same order as PFT_NAMES)
_DEFAULT_VMAX = (65e-6, 65e-6, 40e-6, 30e-6, 30e-6, 25e-6,
                 30e-6, 30e-6, 27.5e-6, 27.5e-6, 15e-6, 25e-6)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_physics_params(path: str) -> dict:
    """Load physics parameters from a YAML file.

    Parameters
    ----------
    path : str
        Path to the params YAML file (e.g. configs/params/default.yaml).

    Returns
    -------
    dict
        Raw parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    import yaml
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Physics params file not found: {path}\n"
            "  Create one or remove 'params_file' from your config to use defaults."
        )
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------------

def build_stomata_params(d: dict) -> StomataParams:
    """Build StomataParams from the 'stomata' section of a params dict."""
    s = d.get('stomata', {})

    # Ball-Berry per group
    bb = s.get('ball_berry', {})
    defaults_gamma  = StomataParams().gamma
    defaults_coefm  = StomataParams().coefm
    defaults_coefb  = StomataParams().coefb
    defaults_gsmin  = StomataParams().gsmin

    gamma = jnp.array([bb.get(g, {}).get('gamma', float(defaults_gamma[i]))
                       for i, g in enumerate(STOMATA_GROUPS)])
    coefm = jnp.array([bb.get(g, {}).get('coefm', float(defaults_coefm[i]))
                       for i, g in enumerate(STOMATA_GROUPS)])
    coefb = jnp.array([bb.get(g, {}).get('coefb', float(defaults_coefb[i]))
                       for i, g in enumerate(STOMATA_GROUPS)])
    gsmin = jnp.array([bb.get(g, {}).get('gsmin', float(defaults_gsmin[i]))
                       for i, g in enumerate(STOMATA_GROUPS)])

    # vmax per PFT
    vm = s.get('vmax', {})
    vmax_pft = jnp.array([vm.get(name, _DEFAULT_VMAX[i])
                          for i, name in enumerate(PFT_NAMES)])

    return StomataParams(
        tau15=s.get('tau15', 4500.0),
        kc15=s.get('kc15', 1.5e-4),
        ko15=s.get('ko15', 0.25),
        cimax=s.get('cimax', 2000.0e-6),
        o2conc=s.get('o2conc', 0.209),
        alpha3=s.get('alpha3', 0.080),
        theta3=s.get('theta3', 0.950),
        beta3=s.get('beta3', 0.990),
        alpha4=s.get('alpha4', 0.050),
        theta4=s.get('theta4', 0.970),
        beta4=s.get('beta4', 0.800),
        stressfac=s.get('stressfac', -5.0),
        vmax_mix=bool(s.get('vmax_mix', False)),      # F77: priority pick
        scale_leaf_only=bool(s.get('scale_leaf_only', False)),   # F77: stems credited to leaves
        acclim_on=bool(s.get('acclim_on', False)),              # CALIBRATION.md 21-22; off: bit-identical
        acclim_pfts=tuple(int(k) for k in s.get('acclim_pfts', (3, 5, 8))),
        acclim_tau=float(s.get('acclim_tau', 10.93)),
        acclim_s0=float(s.get('acclim_s0', -3.063)),
        acclim_smax=float(s.get('acclim_smax', 17.72)),
        cap_scale_u=bool(s.get('cap_scale_u', False)),         # CALIBRATION.md 26; off: bit-identical
        cap_scale_l=bool(s.get('cap_scale_l', False)),
        cap_kn=float(s.get('cap_kn', 0.15)),
        sunshade_u=bool(s.get('sunshade_u', False)),           # CALIBRATION.md 27; off: bit-identical
        coefm_mix=bool(s.get('coefm_mix', False)),             # CALIBRATION.md 32; off: bit-identical
        coefm_pft=_pft_vector(s.get('coefm_pft'), [float(coefm[_PFT_SLOPE_GROUP[i]]) for i in range(len(PFT_NAMES))]),
        acclim_per_pft=bool(s.get('acclim_per_pft', False)),
        acclim_s0_pft=_pft_vector(s.get('acclim_s0_pft'), [float(s.get('acclim_s0', -3.063))] * len(PFT_NAMES)),
        acclim_smax_pft=_pft_vector(s.get('acclim_smax_pft'), [float(s.get('acclim_smax', 17.72))] * len(PFT_NAMES)),
        sunshade_l=bool(s.get('sunshade_l', False)),
        gamma=gamma,
        coefm=coefm,
        coefb=coefb,
        gsmin=gsmin,
        vmax_pft=vmax_pft,
    )


def build_veg_rad_params(d: dict) -> VegRadParams:
    """Build VegRadParams from the 'veg_rad' section of a params dict."""
    s = d.get('veg_rad', {})
    if not s:
        return default_veg_rad_params()
    return VegRadParams(
        rhoveg=jnp.array(s.get('rhoveg', [[0.10, 0.10], [0.60, 0.40]]),
                         dtype=jnp.float32),
        tauveg=jnp.array(s.get('tauveg', [[0.07, 0.05], [0.25, 0.20]]),
                         dtype=jnp.float32),
        oriev=jnp.array(s.get('oriev', [0.5, 0.0]), dtype=jnp.float32),
        orieh=jnp.array(s.get('orieh', [0.0, 0.0]), dtype=jnp.float32),
    )


def build_canopy_params(d: dict) -> CanopyParams:
    """Build CanopyParams from the 'canopy' section of a params dict."""
    s = d.get('canopy', {})
    return CanopyParams(
        cleaf=s.get('cleaf', 0.01),
        cstem=s.get('cstem', 0.01),
        cgrass=s.get('cgrass', 0.01),
        dleaf=tuple(s.get('dleaf', [0.10, 0.10])),
        dstem=tuple(s.get('dstem', [0.10, 0.10])),
        alaiml=s.get('alaiml', 8.0),
        alaimu=s.get('alaimu', 8.0),
        chu=s.get('chu', 8436.0),
        chs=s.get('chs', 210900.0),
        chl=s.get('chl', 8436.0),
        soil_beta_power=s.get('soil_beta_power', 1.0),
    )


def build_interception_params(d: dict) -> InterceptionParams:
    """Build InterceptionParams from the 'interception' section."""
    s = d.get('interception', {})
    return InterceptionParams(
        wliqumax=s.get('wliqumax', 0.20),
        wliqsmax=s.get('wliqsmax', 0.40),
        wliqlmax=s.get('wliqlmax', 0.20),
        wsnoumax=s.get('wsnoumax', 2.00),
        wsnosmax=s.get('wsnosmax', 4.00),
        wsnolmax=s.get('wsnolmax', 2.00),
        tdripu=s.get('tdripu', 7200.0),
        tdrips=s.get('tdrips', 7200.0),
        tdripl=s.get('tdripl', 7200.0),
        tblowu=s.get('tblowu', 43200.0),
        tblows=s.get('tblows', 43200.0),
        tblowl=s.get('tblowl', 43200.0),
        fmax=s.get('fmax', 0.25),
    )


def build_pft_params(d: dict) -> PFTParams:
    """Build PFTParams from the 'pft' section of a params dict.

    Missing PFT entries or missing fields within an entry fall back to
    the values from default_pft_params().
    """
    s = d.get('pft', {})
    if not s:
        return default_pft_params()

    defaults = default_pft_params()

    def _arr(field_name: str) -> jnp.ndarray:
        default_arr = getattr(defaults, field_name)
        vals = []
        for i, name in enumerate(PFT_NAMES):
            pft_entry = s.get(name, {})
            if isinstance(pft_entry, dict):
                vals.append(pft_entry.get(field_name, float(default_arr[i])))
            else:
                vals.append(float(default_arr[i]))
        return jnp.array(vals)

    return PFTParams(
        TminL=_arr('TminL'),
        TminU=_arr('TminU'),
        Twarm=_arr('Twarm'),
        GDD=_arr('GDD'),
        tauleaf=_arr('tauleaf'),
        tauroot=_arr('tauroot'),
        tauwood0=_arr('tauwood0'),
        specla=_arr('specla'),
        aleaf=_arr('aleaf'),
        aroot=_arr('aroot'),
        awood=_arr('awood'),
        vmax_pft=_arr('vmax_pft'),
        xminlai=s.get('xminlai', defaults.xminlai),
        woodnorm=s.get('woodnorm', defaults.woodnorm),
        cover_k=float(s.get('cover_k', defaults.cover_k)),
        fl_min=float(s.get('fl_min', defaults.fl_min)),
        fu_min=float(s.get('fu_min', defaults.fu_min)),
        fu_k=float(s.get('fu_k', defaults.fu_k)),
        tree_stress_min=float(s.get('tree_stress_min', defaults.tree_stress_min)),
        tree_stress_ramp=float(s.get('tree_stress_ramp', defaults.tree_stress_ramp)),
    )


def build_bgc_params(d: dict) -> BGCParams:
    """Build BGCParams from the 'bgc' section of a params dict."""
    s = d.get('bgc', {})
    if not s:
        return default_bgc_params()

    defaults = default_bgc_params()

    # C:N ratios for 8 pools
    cnr_dict = s.get('cnr', {})
    cnr_defaults = list(defaults.cnr)
    cnr = jnp.array([
        cnr_dict.get(name, cnr_defaults[i])
        for i, name in enumerate(BGC_POOL_NAMES)
    ])

    return BGCParams(
        lig_frac=s.get('lig_frac', defaults.lig_frac),
        fbsom=s.get('fbsom', defaults.fbsom),
        effac=s.get('effac', defaults.effac),
        fmax=s.get('fmax', defaults.fmax),
        rconst=s.get('rconst', defaults.rconst),
        cnleaf=s.get('cnleaf', defaults.cnleaf),
        cnroot=s.get('cnroot', defaults.cnroot),
        cnwood=s.get('cnwood', defaults.cnwood),
        cnr=cnr,
        klm=s.get('klm', defaults.klm),
        kls=s.get('kls', defaults.kls),
        kll=s.get('kll', defaults.kll),
        krm=s.get('krm', defaults.krm),
        krs=s.get('krs', defaults.krs),
        krl=s.get('krl', defaults.krl),
        kwm=s.get('kwm', defaults.kwm),
        kws=s.get('kws', defaults.kws),
        kwl=s.get('kwl', defaults.kwl),
        kbn=s.get('kbn', defaults.kbn),
        kbp=s.get('kbp', defaults.kbp),
        knb=s.get('knb', defaults.knb),
        kns=s.get('kns', defaults.kns),
        kpb=s.get('kpb', defaults.kpb),
        kps=s.get('kps', defaults.kps),
        ksb=s.get('ksb', defaults.ksb),
        ylm=s.get('ylm', defaults.ylm),
        yls=s.get('yls', defaults.yls),
        yll=s.get('yll', defaults.yll),
        yrm=s.get('yrm', defaults.yrm),
        yrs=s.get('yrs', defaults.yrs),
        yrl=s.get('yrl', defaults.yrl),
        ywm=s.get('ywm', defaults.ywm),
        yws=s.get('yws', defaults.yws),
        ywl=s.get('ywl', defaults.ywl),
        ybn=s.get('ybn', defaults.ybn),
        ybp=s.get('ybp', defaults.ybp),
        yps=s.get('yps', defaults.yps),
        yns=s.get('yns', defaults.yns),
        ysb=s.get('ysb', defaults.ysb),
        ypb=s.get('ypb', defaults.ypb),
        ynb=s.get('ynb', defaults.ynb),
    )


def build_pheno_params(d: dict):
    """PhenoParams from the optional 'pheno' section.

    Unknown keys raise. A misspelt threshold that quietly fell back to its
    default would be the declared-but-dead pattern this project has met
    fifteen times (docs/DEFECTS_FOR_REVIEW.md), here on a parameter the
    seasonal-pattern criterion is fitting.
    """
    from ibis.physics.phenology import PhenoParams
    sec = dict(d.get('pheno', {}) or {})
    allowed = set(PhenoParams.__dataclass_fields__)
    bad = set(sec) - allowed
    if bad:
        raise KeyError(f"unknown pheno parameter(s) {sorted(bad)}; "
                       f"allowed: {sorted(allowed)}")
    _d = PhenoParams()
    return PhenoParams(**{k: (bool(v) if isinstance(getattr(_d, k), bool) else float(v)) for k, v in sec.items()})


def build_sim_params(
    npoi: int,
    lat_rad,
    params_dict: Optional[dict] = None,
):
    """Build SimParams from an optional params dict.

    Parameters
    ----------
    npoi : int
        Number of land points.
    lat_rad : array-like
        Land-point latitudes in radians, shape (npoi,).
    params_dict : dict, optional
        Parsed YAML content from load_physics_params().
        If None, all defaults are used.

    Returns
    -------
    SimParams
    """
    # Import here to avoid circular imports (loops imports from physics)
    from ibis.timestepping.loops import SimParams

    d = params_dict or {}
    ss = d.get('soil', {})

    return SimParams(
        veg_rad=build_veg_rad_params(d),
        canopy=build_canopy_params(d),
        interception=build_interception_params(d),
        stomata_params=build_stomata_params(d),
        pheno=build_pheno_params(d),
        latitude=jnp.asarray(lat_rad),
        wpudmax=ss.get('wpudmax', 10.0),
        zwpmax=ss.get('zwpmax', 0.5),
        bperm=ss.get('bperm', 0.04),
        # Defaults from ibis.io.soil_init, which derives both from FORGANIC so
        # the two cannot drift apart. They were 2650.0/870.0 here, which drops
        # the organic correction F77 applies in initial.f:715-724.
        rhosoi_val=ss.get('rhosoi_val', RHOSOI_VAL),
        csoi_mineral_val=ss.get('csoi_mineral_val', CSOI_VAL),
        bgc_params=build_bgc_params(d),
    )


def build_all_params(
    npoi: int,
    lat_rad,
    params_dict: Optional[dict] = None,
):
    """Build all parameter objects from an optional params dict.

    Parameters
    ----------
    npoi : int
        Number of land points.
    lat_rad : array-like
        Land-point latitudes in radians.
    params_dict : dict, optional
        Parsed YAML content. None → all defaults.

    Returns
    -------
    (SimParams, PFTParams, BGCParams)
    """
    d = params_dict or {}
    return (
        build_sim_params(npoi, lat_rad, d),
        build_pft_params(d),
        build_bgc_params(d),
    )
