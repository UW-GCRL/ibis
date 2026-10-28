"""IBIS 3.1 Köppen-zone plant functional types in the 12 IBIS positions (CALIBRATION.md 45, 45.1).

Forty types (21 trees minus the merged T20, 7 shrubs, 8 grasses, 3 wetland, 2 crop), each a growth form in a
Köppen zone, with a `parent` among the 12 IBIS PFTs whose parameters it starts from. Every cell has ONE zone,
and within a zone the tree, shrub and grass types fit the 12 positions of the model, so the 40 types do not
widen the state: each cell maps each position to the type that occupies it (`slot_map`), and every per-PFT
parameter becomes a per-cell array (npoi, 12) gathered from the type table (`gather`).

Placement (`zone_slots`): a tree type takes its parent's position if free, otherwise the first free position of
its leaf form (broadleaf 0, 2, 4, 6, 1; conifer 3, 5, 7 -- the groups of the big-leaf photosynthesis); tree
positions with no type in the zone are OFF. The zone's shrub type takes position 8 (evergreen) or 9
(deciduous), its grass 10 (C4) or 11 (C3); the other shrub and grass position stays GENERIC (the IBIS 3.0
parameters and envelope of that position). Unclassified cells ('none') keep every position GENERIC; ice
('E_ice') has the trees OFF. Wetland (W) and crop (C) types are not positions (methane sets and crop factors).

The leaf habit travels with the type (`slot_habit`): 0 evergreen, 1 drought-deciduous, 2 cold-deciduous, the
parent's habit; phenology reads it for the upper canopy. With no type map nothing here is used and the model is
the 12-PFT IBIS 3.0.
"""
from __future__ import annotations

import numpy as np

NSLOT = 12
OFF, GENERIC = -1, -2
EV, DD, CD = 0, 1, 2
#: habit of the 12 IBIS PFTs (0-based); lower positions keep their own phenology in `pheno`
POSITION_HABIT = (EV, DD, EV, EV, CD, EV, CD, CD, EV, CD, EV, EV)
BROADLEAF_POS = (0, 2, 4, 6, 1)
CONIFER_POS = (3, 5, 7)

ZONES = ('A_humid', 'A_savanna', 'B_hot', 'B_cold', 'C_med', 'C_subtrop', 'C_highland', 'C_oceanic',
         'D_temperate', 'D_subarctic', 'D_extreme', 'E_tundra', 'E_ice', 'none')
ZI = {z: i for i, z in enumerate(ZONES)}
KOPPEN_LEGEND = {1: 'Af', 2: 'Am', 3: 'Aw', 4: 'BWh', 5: 'BWk', 6: 'BSh', 7: 'BSk', 8: 'Csa', 9: 'Csb', 10: 'Csc',
                 11: 'Cwa', 12: 'Cwb', 13: 'Cwc', 14: 'Cfa', 15: 'Cfb', 16: 'Cfc', 17: 'Dsa', 18: 'Dsb', 19: 'Dsc',
                 20: 'Dsd', 21: 'Dwa', 22: 'Dwb', 23: 'Dwc', 24: 'Dwd', 25: 'Dfa', 26: 'Dfb', 27: 'Dfc', 28: 'Dfd',
                 29: 'ET', 30: 'EF'}
CLASS_ZONE = {'Af': 'A_humid', 'Am': 'A_humid', 'Aw': 'A_savanna', 'BWh': 'B_hot', 'BSh': 'B_hot', 'BWk': 'B_cold',
              'BSk': 'B_cold', 'Csa': 'C_med', 'Csb': 'C_med', 'Csc': 'C_med', 'Cwa': 'C_subtrop', 'Cfa': 'C_subtrop',
              'Cwb': 'C_highland', 'Cwc': 'C_highland', 'Cfb': 'C_oceanic', 'Cfc': 'C_oceanic',
              'Dfa': 'D_temperate', 'Dfb': 'D_temperate', 'Dwa': 'D_temperate', 'Dwb': 'D_temperate',
              'Dsa': 'D_temperate', 'Dsb': 'D_temperate', 'Dfc': 'D_subarctic', 'Dwc': 'D_subarctic',
              'Dsc': 'D_subarctic', 'Dfd': 'D_extreme', 'Dwd': 'D_extreme', 'Dsd': 'D_extreme', 'ET': 'E_tundra',
              'EF': 'E_ice'}
#: Köppen class code (0-30) -> zone index
CLASS_TO_ZONE = np.array([ZI['none']] + [ZI[CLASS_ZONE[KOPPEN_LEGEND[c]]] for c in range(1, 31)], dtype=np.int16)

# id, name, form, zones, parent IBIS PFT (1-based) -- the list of verify/calibration/pft_koppen_design.py
TYPES = (
    ('T01', 'tropical humid broadleaf evergreen tree', 'tree_be', ('A_humid',), 1),
    ('T02', 'tropical seasonal semi-deciduous tree', 'tree_bd', ('A_savanna',), 2),   # A_humid dropped 2026-10-26: its towers are Aw (CALIBRATION.md 63)
    ('T03', 'tropical savanna tree (drought-deciduous, fire-adapted)', 'tree_bd', ('A_savanna', 'B_hot'), 2),
    ('T04', 'tropical highland broadleaf evergreen tree', 'tree_be', ('C_highland',), 3),
    ('T05', 'hot-arid woodland tree (acacia, mulga)', 'tree_be', ('B_hot',), 2),
    ('T06', 'subtropical broadleaf evergreen tree', 'tree_be', ('C_subtrop',), 3),
    ('T07', 'subtropical conifer (southern pines)', 'tree_ne', ('C_subtrop', 'C_highland'), 4),   # A_humid dropped (CALIBRATION.md 53)
    ('T08', 'subtropical broadleaf deciduous tree', 'tree_bd', ('C_subtrop', 'C_highland'), 5),
    ('T09', 'Mediterranean sclerophyll evergreen tree', 'tree_be', ('C_med',), 3),
    ('T10', 'Mediterranean conifer', 'tree_ne', ('C_med',), 4),
    ('T11', 'Mediterranean broadleaf deciduous tree', 'tree_bd', ('C_med',), 5),
    ('T12', 'oceanic temperate broadleaf deciduous tree', 'tree_bd', ('C_oceanic',), 5),
    ('T13', 'oceanic temperate conifer evergreen tree', 'tree_ne', ('C_oceanic',), 4),
    ('T14', 'oceanic temperate broadleaf evergreen tree', 'tree_be', ('C_oceanic',), 3),
    ('T15', 'continental temperate broadleaf deciduous tree', 'tree_bd', ('D_temperate', 'B_cold'), 5),
    ('T16', 'continental temperate conifer evergreen tree', 'tree_ne', ('D_temperate',), 4),
    ('T17', 'subarctic conifer evergreen tree (closed and open taiga)', 'tree_ne', ('D_subarctic', 'D_extreme', 'E_tundra'), 6),
    ('T18', 'subarctic broadleaf deciduous tree (birch, aspen)', 'tree_bd', ('D_subarctic', 'D_extreme'), 7),
    ('T19', 'subarctic conifer deciduous tree (larch)', 'tree_nd', ('D_subarctic', 'D_extreme'), 8),
    ('T21', 'cold-steppe conifer woodland (juniper, pinyon)', 'tree_ne', ('B_cold',), 4),
    ('S01', 'hot-arid evergreen shrub', 'shrub_e', ('B_hot',), 9),
    ('S02', 'cold-desert and steppe shrub (sagebrush)', 'shrub_d', ('B_cold',), 10),
    ('S03', 'Mediterranean shrub (maquis, chaparral)', 'shrub_e', ('C_med',), 9),
    ('S04', 'temperate deciduous shrub', 'shrub_d', ('C_subtrop', 'C_oceanic', 'D_temperate', 'C_highland'), 10),
    ('S05', 'boreal evergreen shrub (ericaceous)', 'shrub_e', ('D_subarctic', 'D_extreme'), 9),
    ('S06', 'tundra dwarf shrub', 'shrub_d', ('E_tundra',), 10),
    ('S07', 'tropical shrub and understorey', 'shrub_e', ('A_humid', 'A_savanna'), 9),
    ('G01', 'tropical C4 grass', 'grass_c4', ('A_humid', 'A_savanna'), 11),
    ('G02', 'hot-arid C4 grass', 'grass_c4', ('B_hot',), 11),
    ('G03', 'subtropical C4 grass', 'grass_c4', ('C_subtrop', 'C_highland'), 11),
    ('G04', 'cold-steppe C3 grass', 'grass_c3', ('B_cold',), 12),
    ('G05', 'Mediterranean annual C3 grass', 'grass_c3', ('C_med',), 12),
    ('G06', 'temperate C3 grass', 'grass_c3', ('C_oceanic', 'D_temperate'), 12),
    ('G07', 'subarctic C3 grass and forb', 'grass_c3', ('D_subarctic', 'D_extreme'), 12),
    ('G08', 'tundra graminoid (sedge, moss)', 'grass_c3', ('E_tundra',), 12),
    ('W01', 'tropical and subtropical wetland vegetation', 'wetland', ('A_humid', 'A_savanna', 'B_hot', 'C_subtrop', 'C_highland'), 11),
    ('W02', 'temperate wetland graminoid', 'wetland', ('C_med', 'C_oceanic', 'D_temperate', 'B_cold'), 12),
    ('W03', 'boreal and arctic sedge-moss wetland', 'wetland', ('D_subarctic', 'D_extreme', 'E_tundra'), 12),
    ('C01', 'C3 crop (managed)', 'crop', ('*',), 12),
    ('C02', 'C4 crop (managed)', 'crop', ('*',), 11),
)
IDS = tuple(t[0] for t in TYPES)
TI = {t: i for i, t in enumerate(IDS)}
NTYPE = len(TYPES)


def _habit(t):
    """The type's leaf habit: its parent's (shrubs and grasses keep the position's phenology)."""
    return POSITION_HABIT[t[4] - 1]


def zone_slots(zone: str):
    """(12,) type index per position for one zone (OFF / GENERIC codes), and (12,) habit codes."""
    slot = np.full(NSLOT, GENERIC, dtype=np.int16)
    habit = np.array(POSITION_HABIT, dtype=np.int8)
    if zone == 'none':
        return slot, habit
    slot[:8] = OFF
    if zone == 'E_ice':
        return slot, habit
    for i, t in enumerate(TYPES):
        tid, _, form, zones, parent = t
        if zone not in zones or form in ('wetland', 'crop'):
            continue
        if form.startswith('tree'):
            home = parent - 1
            order = (home,) + tuple(p for p in (CONIFER_POS if form in ('tree_ne', 'tree_nd') else BROADLEAF_POS) if p != home)
            free = [p for p in order if slot[p] == OFF]
            if not free:
                raise ValueError(f'zone {zone}: no free position for {tid}')
            slot[free[0]] = i; habit[free[0]] = _habit(t)
        elif form.startswith('shrub'):
            p = 8 if form == 'shrub_e' else 9
            if slot[p] != GENERIC: raise ValueError(f'zone {zone}: two shrub types in position {p}')
            slot[p] = i
        elif form.startswith('grass'):
            p = 10 if form == 'grass_c4' else 11
            if slot[p] != GENERIC: raise ValueError(f'zone {zone}: two grass types in position {p}')
            slot[p] = i
    return slot, habit


ZONE_SLOTS = np.stack([zone_slots(z)[0] for z in ZONES])        # (nzone, 12)
ZONE_HABIT = np.stack([zone_slots(z)[1] for z in ZONES])        # (nzone, 12)


def zones_from_classes(classes) -> np.ndarray:
    """Köppen class codes (0-30, any shape) -> zone indices."""
    return CLASS_TO_ZONE[np.clip(np.asarray(classes, dtype=np.int64), 0, 30)]


def slot_map(zone_idx) -> np.ndarray:
    """(npoi,) zone indices -> (npoi, 12) type index per position (OFF = -1, GENERIC = -2)."""
    return ZONE_SLOTS[np.asarray(zone_idx, dtype=np.int64)]


def slot_habit(zone_idx) -> np.ndarray:
    """(npoi,) zone indices -> (npoi, 12) leaf habit per position (0 evergreen, 1 drought-, 2 cold-deciduous)."""
    return ZONE_HABIT[np.asarray(zone_idx, dtype=np.int64)]


def slot_parent(slots) -> np.ndarray:
    """(npoi, 12) type map -> 0-based parent position per position (the position itself where GENERIC/OFF)."""
    slots = np.asarray(slots)
    par = np.array([t[4] - 1 for t in TYPES], dtype=np.int16)
    own = np.broadcast_to(np.arange(NSLOT, dtype=np.int16), slots.shape)
    return np.where(slots >= 0, par[np.clip(slots, 0, None)], own)


def slot_gate(slots) -> np.ndarray:
    """(npoi, 12) 1.0 where the position may hold vegetation (a type or GENERIC), 0.0 where OFF."""
    return (np.asarray(slots) != OFF).astype(np.float32)


def slot_typed(slots) -> np.ndarray:
    """(npoi, 12) 1.0 where the position holds a zone TYPE (its existence is the zone's; climate_analysis.existence)."""
    return (np.asarray(slots) >= 0).astype(np.float32)


def gather(base, slots, overrides: dict | None = None) -> np.ndarray:
    """Per-cell (npoi, 12) parameter from a 12-entry base vector (IBIS PFT order) and per-type overrides.

    A position holding type t takes overrides[t] if given, else the base value of t's PARENT (so a type moved
    off its parent's position keeps its parent's numbers); GENERIC and OFF positions take the base value of the
    position itself.
    """
    base = np.broadcast_to(np.asarray(base, dtype=np.float64), (NSLOT,))
    slots = np.asarray(slots)
    val = np.empty(NTYPE, dtype=np.float64)
    for i, t in enumerate(TYPES):
        val[i] = base[t[4] - 1]
    for tid, v in (overrides or {}).items():
        if tid not in TI:
            raise KeyError(f'unknown type {tid}')
        val[TI[tid]] = float(v)
    out = np.broadcast_to(base, slots.shape).copy()
    m = slots >= 0
    out[m] = val[slots[m]]
    return out.astype(np.float32)


def describe_zones() -> str:
    """One line per zone: the type in each position (for the run log and the docs)."""
    lines = []
    for z, row, hab in zip(ZONES, ZONE_SLOTS, ZONE_HABIT):
        cells = []
        for p, s in enumerate(row):
            if s >= 0: cells.append(f'{p + 1}:{IDS[s]}' + ('' if hab[p] == POSITION_HABIT[p] else f'({"EV DD CD".split()[hab[p]]})'))
            elif s == GENERIC: cells.append(f'{p + 1}:gen')
        lines.append(f'{z:12s} ' + ' '.join(cells))
    return '\n'.join(lines)


#: yaml key -> (object, field) of the per-PFT parameters a type carries
TYPE_FIELDS = {
    'vmax': ('stomata', 'vmax_pft'), 'coefm': ('stomata', 'coefm_pft'),
    'acclim_s0': ('stomata', 'acclim_s0_pft'), 'acclim_smax': ('stomata', 'acclim_smax_pft'),
    'specla': ('pft', 'specla'), 'tauleaf': ('pft', 'tauleaf'), 'tauroot': ('pft', 'tauroot'),
    'tauwood0': ('pft', 'tauwood0'), 'aleaf': ('pft', 'aleaf'), 'aroot': ('pft', 'aroot'), 'awood': ('pft', 'awood'),
    'rgrowth': ('bgc', 'rgrowth'), 'rroot': ('bgc', 'rroot'), 'rwood': ('bgc', 'rwood'), 'lt_e0': ('bgc', 'lt_e0'),
    'fire_mort': ('fire', 'mort'), 'lai_max': ('pft', 'lai_max'),
}


#: per-cell scalars a type can carry (CALIBRATION.md 45): yaml key -> (object, field, which of the zone's types set it).
#: 'upper': the zone's cold-deciduous tree types; 'lower': the zone's grass type, else its shrub type; 'all': every
#: type of the zone. A zone with no such type with a value keeps the base scalar.
TYPE_SCALARS = {
    'gdd_thresh_u': ('pheno', 'gdd_thresh_u', 'upper'), 'gdd_ramp_u': ('pheno', 'gdd_ramp_u', 'upper'),
    'sen_ycrit_u': ('pheno', 'sen_ycrit_u', 'upper'),
    'gdd_thresh_l': ('pheno', 'gdd_thresh_l', 'lower'), 'tthresh_l_c': ('pheno', 'tthresh_l_c', 'lower'),
    'sen_ycrit_l': ('pheno', 'sen_ycrit_l', 'lower'), 'sen_pstart_l': ('pheno', 'sen_pstart_l', 'lower'),   # CALIBRATION.md 54
    'drop_stress_thresh': ('pheno', 'drop_stress_thresh', 'all'), 'stressfac': ('stomata', 'stressfac', 'all'),
    # cover-limit ramp per zone (CALIBRATION.md 71): fu_max = clip((u - min) / ramp, 0, 1) ** power, set on the zone's
    # tree types; a zone without values keeps the base (tropical) ramp
    'tree_cover_min': ('pft', 'tree_cover_min', 'all'), 'tree_cover_ramp': ('pft', 'tree_cover_ramp', 'all'),
    'tree_cover_power': ('pft', 'tree_cover_power', 'all'),
}


def zone_scalar(zone: str, values: dict, which: str):
    """Median of `values` ({type_id: v}) over the zone's types selected by `which`; None when none has a value."""
    row = ZONE_SLOTS[ZI[zone]]; hab = ZONE_HABIT[ZI[zone]]
    if which == 'upper':
        tids = [IDS[row[p]] for p in range(8) if row[p] >= 0 and hab[p] == CD]
    elif which == 'lower':
        g = [IDS[row[p]] for p in (10, 11) if row[p] >= 0]; s = [IDS[row[p]] for p in (8, 9) if row[p] >= 0]
        tids = [t for t in g if t in values] or [t for t in s if t in values]
    else:
        tids = [IDS[r] for r in row if r >= 0]
    v = [values[t] for t in tids if t in values]
    return float(np.median(v)) if v else None


def parse_type_overrides(section) -> dict:
    """{param: {type_id: value}} from the params-file section 'types' ({type_id: {param: value}}); unknown
    types or parameters are an error."""
    out = {}
    for tid, kv in (section or {}).items():
        if tid not in TI:
            raise KeyError(f"unknown type {tid!r} in 'types'; types are {IDS}")
        for k, v in (kv or {}).items():
            if k not in TYPE_FIELDS and k not in TYPE_SCALARS:
                raise KeyError(f"unknown type parameter {k!r} for {tid}; parameters are {sorted(TYPE_FIELDS) + sorted(TYPE_SCALARS)}")
            out.setdefault(k, {})[tid] = float(v)
    return out


def apply_types(params, pft_params, zone_idx, overrides: dict | None = None, base=None):
    """SimParams and PFTParams with every per-PFT parameter gathered per cell for the zones `zone_idx` (npoi,).

    `base` = (params, pft_params) holding the 12-entry vectors (the IBIS 3.0 set); default: the objects given.
    Keep the base pair and call again when the zone map changes (a new Köppen period): shapes stay (npoi, 12),
    so the compiled kernels are reused.
    """
    import equinox as eqx
    import jax.numpy as jnp
    bp, bf = base if base is not None else (params, pft_params)
    slots = slot_map(zone_idx); npoi = slots.shape[0]
    overrides = overrides or {}
    objs = {'stomata': bp.stomata_params, 'pft': bf, 'bgc': bp.bgc_params, 'fire': bp.fire}
    new = {}
    for key, (obj, field) in TYPE_FIELDS.items():
        b = getattr(objs[obj], field)
        if b is None:                # an optional field (lai_max): stays off unless a type sets it; then unset = no cap
            if not overrides.get(key):
                continue
            b = np.full(NSLOT, np.inf)
        b = np.asarray(b, dtype=np.float64)
        if b.ndim == 2:          # already per cell: take the 12-entry base of the first row is wrong -- refuse
            raise ValueError(f'{obj}.{field} is already per cell; pass the 12-entry base via `base=`')
        new[(obj, field)] = jnp.asarray(gather(np.broadcast_to(b, (NSLOT,)), slots, overrides.get(key)))
    sp = bp.stomata_params; bg = bp.bgc_params; fp = bp.fire
    zones_cc = tuple(getattr(fp, 'closed_canopy_zones', ()) or ())
    if zones_cc:                                      # closed-canopy climate flag per cell (CALIBRATION.md 65)
        bad = [z for z in zones_cc if z not in ZI]
        if bad: raise KeyError(f'unknown closed_canopy_zones {bad}; zones: {ZONES}')
        flag = np.isin(np.asarray(zone_idx, dtype=np.int64), [ZI[z] for z in zones_cc]).astype(np.float32)
        fp = eqx.tree_at(lambda o: o.closed_canopy_cell, fp, jnp.asarray(flag), is_leaf=lambda x: x is None)
    for (obj, field), v in new.items():
        if obj == 'stomata': sp = eqx.tree_at(lambda o: getattr(o, field), sp, v)
        elif obj == 'bgc': bg = eqx.tree_at(lambda o: getattr(o, field), bg, v, is_leaf=lambda x: x is None)
        elif obj == 'fire': fp = eqx.tree_at(lambda o: getattr(o, field), fp, v)
    pf = bf
    for (obj, field), v in new.items():
        if obj == 'pft': pf = eqx.tree_at(lambda o: getattr(o, field), pf, v, is_leaf=lambda x: x is None)
    # per-cell scalars by zone (phenology thresholds, drought trigger, stress factor, cover-limit ramp)
    zi = np.asarray(zone_idx, dtype=np.int64); ph = bp.pheno
    for key, (obj, field, which) in TYPE_SCALARS.items():
        if key not in overrides:
            continue
        base_v = float(np.asarray(getattr({'pheno': ph, 'stomata': bp.stomata_params, 'pft': bf}[obj], field)))
        per_zone = np.array([zone_scalar(z, overrides[key], which) if z not in ('none', 'E_ice') else None for z in ZONES], dtype=object)
        per_zone = np.array([base_v if v is None else v for v in per_zone], dtype=np.float64)
        val = jnp.asarray(per_zone[zi], dtype=jnp.float32)
        if obj == 'pheno': ph = eqx.tree_at(lambda o: getattr(o, field), ph, val)
        elif obj == 'pft': pf = eqx.tree_at(lambda o: getattr(o, field), pf, val)
        else: sp = eqx.tree_at(lambda o: getattr(o, field), sp, val)
    hab = jnp.asarray(slot_habit(zone_idx), dtype=jnp.int32)
    par = jnp.asarray(slot_parent(slots), dtype=jnp.int32)
    gate = jnp.asarray(slot_gate(slots), dtype=jnp.float32); typed = jnp.asarray(slot_typed(slots), dtype=jnp.float32)
    p = eqx.tree_at(lambda o: (o.stomata_params, o.bgc_params, o.fire, o.pheno), bp, (sp, bg, fp, ph))
    p = eqx.tree_at(lambda o: (o.slot_habit, o.slot_parent, o.slot_gate, o.slot_typed), p, (hab, par, gate, typed),
                    is_leaf=lambda x: x is None)
    return p, pf


def type_area_summary(zone_idx) -> str:
    """Land points per zone, for the run log."""
    z = np.asarray(zone_idx)
    return ' '.join(f'{ZONES[i]}:{int((z == i).sum())}' for i in range(len(ZONES)) if (z == i).any())


#: methane band of the wetland type of each zone (ibis.bgc.methane.biome_methane_params: 0 tropical, 1 temperate,
#: 2 boreal; -1 = no wetland type, the latitude rule applies)
def wetland_band(zone_idx) -> np.ndarray:
    band = {}
    for t in TYPES:
        if t[2] != 'wetland': continue
        b = {'W01': 0, 'W02': 1, 'W03': 2}[t[0]]
        for z in t[3]: band[z] = b
    lut = np.array([band.get(z, -1) for z in ZONES], dtype=np.int16)
    return lut[np.asarray(zone_idx, dtype=np.int64)]
