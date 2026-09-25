"""One-at-a-time parameter sensitivity at towers (stage S1.4 screening; CALIBRATION.md 20).

    ibis-python verify/calibration/site_sensitivity.py --out <dir> [--per-class 4] [--params default.yaml]

Towers: up to `--per-class` per IGBP class from the 2008-2012 batch (natural vegetation), composition prescribed as in
site_run.py, tower meteorology. Protocol: two spin cycles of 2008-2012 on the base parameters; then, from that state, each
perturbation runs one cycle and is scored on its last three years. Metrics per tower: annual GPP ratio to the tower, June-July
ratio, cold-season share of model GPP, leaf-out and senescence day (25 % crossings of the smoothed daily GPP), ET ratio, summer LAI.
Output: <out>/sensitivity_runs.csv (every tower x perturbation) and <out>/sensitivity_by_class.csv (mean change of each metric
per perturbation and class, relative to the base run), printed.
"""
import sys, os, glob, json, argparse, numpy as np, pandas as pd, xarray as xr, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
import jax, jax.numpy as jnp, equinox as eqx
from fit_carbon import LATV, LONV, MASK, GSWP3, NPOI_FULL, apply_params, trace_extras, _extra_value, modis_at
import site_transient; from site_transient import one_year
from site_run import prescribe, tower_monthly, tower_mat, enforce_prescription, META, SEED_LAI, SEED_WOOD
from phenology_vs_phenocam import crossings, smooth
from tower_forcing import TowerForcingReader
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum; from ibis.state.et_parts import create_et_parts; from ibis.state.seasonal import create_seasonal
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; S = '/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad'
CLASSES = ['ENF', 'DBF', 'GRA', 'MF', 'EBF', 'OSH', 'WSA', 'SAV']

# (label, kind, spec): kind 'extras' -> dict name: function(default) ; 'vmax'/'specla'/'tauleaf'/'aleaf' -> factor or delta on all 12 PFTs
PERTURBATIONS = [
    ('vmax x0.7', 'vmax', 0.7), ('vmax x1.4', 'vmax', 1.4),
    ('specla x0.7', 'specla', 0.7), ('specla x1.4', 'specla', 1.4),
    ('tauleaf x0.7', 'tauleaf', 0.7), ('tauleaf x1.4', 'tauleaf', 1.4),
    ('aleaf -0.10 (to wood)', 'aleaf', -0.10), ('aleaf +0.10 (from wood)', 'aleaf', +0.10),
    ('gdd_thresh_u +150', 'extras', {'gdd_thresh_u': lambda d: d + 150.0}), ('gdd_thresh_u -50', 'extras', {'gdd_thresh_u': lambda d: max(d - 50.0, 10.0)}),
    ('gdd_thresh_l +150', 'extras', {'gdd_thresh_l': lambda d: d + 150.0}), ('gdd_thresh_l -50', 'extras', {'gdd_thresh_l': lambda d: max(d - 50.0, 10.0)}),
    ('gdd_ramp_u x2', 'extras', {'gdd_ramp_u': lambda d: d * 2.0}), ('gdd_ramp_l x2', 'extras', {'gdd_ramp_l': lambda d: d * 2.0}),
    ('tthresh_u_offset +5 C', 'extras', {'tthresh_u_offset': lambda d: d + 5.0}), ('tthresh_u_offset -5 C', 'extras', {'tthresh_u_offset': lambda d: d - 5.0}),
    ('tthresh_l_c +5 C', 'extras', {'tthresh_l_c': lambda d: d + 5.0}), ('tthresh_l_c +10 C', 'extras', {'tthresh_l_c': lambda d: d + 10.0}),
    ('ddays x2 (slower drop)', 'extras', {'ddays': lambda d: d * 2.0}), ('ddays x0.5 (faster drop)', 'extras', {'ddays': lambda d: d * 0.5}),
    ('stressfac x0.7', 'extras', {'stressfac': lambda d: d * 0.7}), ('stressfac x1.4', 'extras', {'stressfac': lambda d: d * 1.4}),
    ('drop_stress_thresh 0.3', 'extras', {'drop_stress_thresh': lambda d: 0.3}), ('drop_stress_thresh 0.6', 'extras', {'drop_stress_thresh': lambda d: 0.6}),
    ('coefm all x0.7', 'extras', {k: (lambda d: d * 0.7) for k in ('coefm_bl', 'coefm_con', 'coefm_shrub', 'coefm_c4', 'coefm_c3')}), ('coefm all x1.4', 'extras', {k: (lambda d: d * 1.4) for k in ('coefm_bl', 'coefm_con', 'coefm_shrub', 'coefm_c4', 'coefm_c3')}),
    ('fmax x0.5', 'extras', {'fmax': lambda d: d * 0.5}), ('fmax x2', 'extras', {'fmax': lambda d: min(d * 2.0, 1.0)}),
    ('soil_beta_power 2', 'extras', {'soil_beta_power': lambda d: 2.0}), ('soil_beta_power 3', 'extras', {'soil_beta_power': lambda d: 3.0}),
    ('fl_min 0.5', 'extras', {'fl_min': lambda d: 0.5}), ('cover_k x2', 'extras', {'cover_k': lambda d: d * 2.0}),
]

# Wide-range phenology set. The upper-canopy senescence threshold is max(0 C, tc + tthresh_u_offset, tthresh_u_abs), so at
# cold-winter sites the 0 C floor pins it and the offset cannot move it: `tthresh_u_abs` is the working knob. Leaf-out needs
# hundreds of growing-degree-days to move weeks, so the range is widened accordingly.
PHENOLOGY_WIDE = [
    ('gdd_thresh_u +300', 'extras', {'gdd_thresh_u': lambda d: d + 300.0}), ('gdd_thresh_u +600', 'extras', {'gdd_thresh_u': lambda d: d + 600.0}), ('gdd_thresh_u +1000', 'extras', {'gdd_thresh_u': lambda d: d + 1000.0}),
    ('gdd_thresh_l +300', 'extras', {'gdd_thresh_l': lambda d: d + 300.0}), ('gdd_thresh_l +600', 'extras', {'gdd_thresh_l': lambda d: d + 600.0}),
    ('gthresh_u_c +5 C (slower GDD)', 'extras', {'gthresh_u_c': lambda d: d + 5.0}), ('gthresh_l_c +5 C', 'extras', {'gthresh_l_c': lambda d: d + 5.0}),
    ('tthresh_u_abs 5 C', 'extras', {'tthresh_u_abs': lambda d: 5.0}), ('tthresh_u_abs 8 C', 'extras', {'tthresh_u_abs': lambda d: 8.0}), ('tthresh_u_abs 12 C', 'extras', {'tthresh_u_abs': lambda d: 12.0}),
    ('tthresh_l_c 5 C', 'extras', {'tthresh_l_c': lambda d: 5.0}), ('tthresh_l_c 10 C', 'extras', {'tthresh_l_c': lambda d: 10.0}), ('tthresh_l_c 15 C', 'extras', {'tthresh_l_c': lambda d: 15.0}),
    ('ddays 5 (fast drop)', 'extras', {'ddays': lambda d: 5.0}), ('ddays 30 (slow drop)', 'extras', {'ddays': lambda d: 30.0}),
    ('tthresh_u_abs 8 C + gdd_thresh_u +600', 'extras', {'tthresh_u_abs': lambda d: 8.0, 'gdd_thresh_u': lambda d: d + 600.0}),
]

# Parameters of the two seasonal formulations (CALIBRATION.md 21-22), plus a wide leaf-out range, for a base yaml that
# switches the formulations on (python-port/configs/params/default_seasonal.yaml).
SEASONAL_SET = [
    ('acclim_tau x0.5', 'extras', {'acclim_tau': lambda d: d * 0.5}), ('acclim_tau x2', 'extras', {'acclim_tau': lambda d: d * 2.0}),
    ('acclim_s0 -3 C', 'extras', {'acclim_s0': lambda d: d - 3.0}), ('acclim_s0 +3 C', 'extras', {'acclim_s0': lambda d: d + 3.0}),
    ('acclim_smax x0.6', 'extras', {'acclim_smax': lambda d: d * 0.6}), ('acclim_smax x1.5', 'extras', {'acclim_smax': lambda d: d * 1.5}),
    ('sen_pstart_u -1 h', 'extras', {'sen_pstart_u': lambda d: d - 1.0}), ('sen_pstart_u +1 h', 'extras', {'sen_pstart_u': lambda d: d + 1.0}),
    ('sen_tb_u -4 C', 'extras', {'sen_tb_u': lambda d: d - 4.0}), ('sen_tb_u +4 C', 'extras', {'sen_tb_u': lambda d: d + 4.0}),
    ('sen_ycrit_u x0.5', 'extras', {'sen_ycrit_u': lambda d: d * 0.5}), ('sen_ycrit_u x2', 'extras', {'sen_ycrit_u': lambda d: d * 2.0}),
    ('sen_pstart_l -1 h', 'extras', {'sen_pstart_l': lambda d: d - 1.0}), ('sen_ycrit_l x0.5', 'extras', {'sen_ycrit_l': lambda d: d * 0.5}),
    ('sen_ycrit_l x2', 'extras', {'sen_ycrit_l': lambda d: d * 2.0}), ('sen_tb_l +4 C', 'extras', {'sen_tb_l': lambda d: d + 4.0}),
    ('gdd_thresh_u +300', 'extras', {'gdd_thresh_u': lambda d: d + 300.0}), ('gdd_thresh_u +600', 'extras', {'gdd_thresh_u': lambda d: d + 600.0}),
    ('gthresh_u_c +5 C', 'extras', {'gthresh_u_c': lambda d: d + 5.0}), ('gdd_thresh_l +300', 'extras', {'gdd_thresh_l': lambda d: d + 300.0}),
]


def metrics(gm, gd, etm, lai_l, lai_u, og, oe, NH, ref=None, lai_m=None):
    """Per-tower metrics from 3 evaluation years: gm (3,12,n) gC/m2/mo, gd (3,366,n) gC/m2/d, etm (3,12,n) mm, og/oe (12,n) tower climatology."""
    m = gm.mean(0); e = etm.mean(0); out = {}
    jj = [5, 6] if NH else [11, 0]; win = [10, 11, 0, 1] if NH else [4, 5, 6, 7]
    out['gpp_ratio'] = m.sum(0) / np.maximum(np.nansum(og, 0), 1e-6); out['junjul_ratio'] = m[jj].sum(0) / np.maximum(np.nansum(og[jj], 0), 1e-6)
    out['winter_share'] = m[win].sum(0) / np.maximum(m.sum(0), 1e-6); out['et_ratio'] = e.sum(0) / np.maximum(np.nansum(oe, 0), 1e-6)
    out['lai_summer'] = (lai_l + lai_u).mean(0)
    lo, se, amp = [], [], []
    for k in range(gm.shape[2]):
        rk = None if ref is None else (ref[0][k], ref[1][k])
        r = [crossings(gd[y, :, k], 0.25, ref=rk) for y in range(gd.shape[0])]; r = [x for x in r if np.isfinite(x[0])]
        lo.append(np.mean([x[0] for x in r]) if r else np.nan); se.append(np.mean([x[1] for x in r]) if r else np.nan)
        yk = np.nanmean(gd[:, :, k], axis=0); amp.append((np.nanpercentile(smooth(np.nan_to_num(yk)), 5), np.nanmax(smooth(np.nan_to_num(yk)))))
    out['leafout_doy'] = np.array(lo); out['senesc_doy'] = np.array(se)
    out['_amp_lo'] = np.array([a[0] for a in amp]); out['_amp_hi'] = np.array([a[1] for a in amp])
    if lai_m is not None:                                           # displayed-LAI season, independent of any GPP threshold
        L = np.nanmean(lai_m, axis=0)                               # (12, n)
        on, off, rng = [], [], []
        for k in range(L.shape[1]):
            v = L[:, k]; hi2, lo2 = np.nanmax(v), np.nanmin(v)
            if hi2 - lo2 < 0.2: on.append(np.nan); off.append(np.nan); rng.append(hi2 - lo2); continue
            lev2 = lo2 + 0.5 * (hi2 - lo2); im = int(np.nanargmax(v)); above = v >= lev2
            r0 = next((m for m in range(im, -1, -1) if not above[m]), -1) + 1
            f0 = next((m for m in range(im, 12) if not above[m]), 12) - 1
            on.append(r0 + 1); off.append(f0 + 1); rng.append(hi2 - lo2)
        out['lai_on_month'] = np.array(on, float); out['lai_off_month'] = np.array(off, float); out['lai_range'] = np.array(rng, float)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--per-class', type=int, default=4); ap.add_argument('--params', default='python-port/configs/params/default.yaml'); ap.add_argument('--only', default='', help='comma list of perturbation labels (smoke test)'); ap.add_argument('--set', default='', choices=['', 'phenology', 'seasonal'], help="'phenology': the wide-range phenology set; 'seasonal': the CALIBRATION.md 21-22 formulation parameters (run with a yaml that switches them on)"); ap.add_argument('--fixed-lai', default='', choices=['', 'attrs'], help="'attrs': hold leaf carbon at the site-attribute maximum LAI (fixed vegetation, stage C1)"); ap.add_argument('--dump-monthly', action='store_true', help='also save the monthly GPP and displayed-LAI series per perturbation')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    # towers: batch 1 (2008-2012), up to per-class per IGBP, ordered by record length
    b1 = open(f'{S}/site_batches.txt').readline().split()[1].split(','); inv = pd.read_csv(f'{S}/site_inventory.csv').set_index('site')
    sites = []
    for c in CLASSES:
        cand = [s for s in b1 if META.loc[s, 'IGBP'] == c]; cand = sorted(cand, key=lambda s: -inv.loc[s, 'nyr'] if s in inv.index else 0)
        sites += cand[:a.per_class]
    print(f'{len(sites)} towers: ' + ', '.join(f"{s}({META.loc[s,'IGBP']})" for s in sites), flush=True)
    m = np.array(xr.open_dataset(MASK).landfrac.values) > 0; fl = np.where(m.ravel())[0]; posmap = {g: p for p, g in enumerate(fl)}
    i = np.array([int(np.argmin(np.abs(LATV - META.loc[s, 'LOCATION_LAT']))) for s in sites], np.int32); j = np.array([int(np.argmin(np.abs(LONV - META.loc[s, 'LOCATION_LONG']))) for s in sites], np.int32)
    sel = np.array([posmap[ii * 720 + jj] for ii, jj in zip(i, j)]); lat_rad = np.deg2rad(LATV[i]).astype(np.float32); NH = LATV[i].mean() >= 0
    YEARS = [2008, 2009, 2010, 2011, 2012]; g = GSWP3Reader(GSWP3); rd = TowerForcingReader(sites, g)
    full, _ = load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz', create_ibis_state(GridConfig(npoi=NPOI_FULL, nbedrock=3)))
    cut = lambda x: x[sel] if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL else x
    st0 = _cast_tree_f32(jax.tree.map(cut, full)); st0 = eqx.tree_at(lambda s: s.stress, st0, create_stress_accum(len(sel)), is_leaf=lambda x: x is None); st0 = eqx.tree_at(lambda s: s.etparts, st0, create_et_parts(len(sel)), is_leaf=lambda x: x is None)
    st0 = eqx.tree_at(lambda s: s.seasonal, st0, _cast_tree_f32(create_seasonal(len(sel), 5.0)), is_leaf=lambda x: x is None)   # inert unless a formulation is on (CALIBRATION.md 21-22)
    sp0, pft0, _ = build_all_params(len(sel), lat_rad, load_physics_params(a.params)); sp0, pft0 = trace_extras(sp0, pft0)   # one compile for the base and every perturbation
    specla0 = np.asarray(pft0.specla); vmax0 = np.asarray(sp0.stomata_params.vmax_pft)
    mask = np.zeros((len(sel), 12), np.float32); fu = np.zeros(len(sel), np.float32); fl_ = np.zeros(len(sel), np.float32)
    cbiol = np.asarray(st0.vegetation.cbiol).copy(); cbiow = np.asarray(st0.vegetation.cbiow).copy(); cbior = np.asarray(st0.vegetation.cbior).copy()
    for n, s in enumerate(sites):
        igbp = META.loc[s, 'IGBP']; mat = META.loc[s, 'MAT'] if 'MAT' in META else np.nan
        if not (isinstance(mat, (int, float)) and np.isfinite(mat)): mat = tower_mat(s)
        pfts, fu[n], fl_[n], _ = prescribe(igbp, float(META.loc[s, 'LOCATION_LAT']), float(mat) if pd.notna(mat) else None)
        for p in range(12):
            if p in pfts:
                mask[n, p] = 1.0
                if cbiol[n, p] * specla0[p] < 0.3: cbiol[n, p] = SEED_LAI[p] / specla0[p]; cbiow[n, p] = SEED_WOOD[p]; cbior[n, p] = 0.5
            else: cbiol[n, p] = 0.0; cbiow[n, p] = 0.0; cbior[n, p] = 0.0
    st0 = eqx.tree_at(lambda s: (s.vegetation.cbiol, s.vegetation.cbiow, s.vegetation.cbior, s.vegetation.plai, s.vegetation.exist, s.vegetation.fu, s.vegetation.fl), st0,
                      tuple(jnp.asarray(x, dtype=jnp.float32) for x in (cbiol, cbiow, cbior, cbiol * specla0[None, :], mask, fu, fl_)))
    consts = PhysicalConstants(); site_transient.apply_co2 = lambda q, y, series=None: eqx.tree_at(lambda z: z.atm.co2conc, q, 370.0e-6)
    fix = None
    if a.fixed_lai == 'attrs':
        from site_attributes import prescribe_from_attributes
        fix = np.asarray(cbiol).copy()
        for n, s in enumerate(sites):
            at = prescribe_from_attributes(s)
            if at is not None and np.isfinite(at['lai_max']) and at['lai_max'] > 0:
                fix[n] = mask[n] / max(mask[n].sum(), 1) * at['lai_max'] / np.maximum(specla0, 1e-6)
        st0 = eqx.tree_at(lambda s: (s.vegetation.cbiol, s.vegetation.plai), st0, (jnp.asarray(fix, dtype=jnp.float32), jnp.asarray(fix * specla0[None, :], dtype=jnp.float32)))
        print('fixed vegetation: leaf carbon held (site-attribute LAI where available)', flush=True)
    hold = lambda s_: enforce_prescription(s_, mask, specla0, fix_cbiol=fix)
    REC = []
    def hook(fn):
        def rec(state, *aa, **kk):
            out = fn(state, *aa, **kk); vg = out[0].vegetation; REC.append((np.asarray(vg.lai[:, 0] * vg.fl), np.asarray(vg.lai[:, 1] * vg.fu))); return out
        return rec
    site_transient.run_month_gswp3 = hook(site_transient.run_month_gswp3)
    # observations
    og = np.full((12, len(sites)), np.nan); oe = np.full((12, len(sites)), np.nan)
    for n, s in enumerate(sites):
        tm = tower_monthly(s); tm = tm[[y in YEARS for y in tm.index.get_level_values(0)]]
        for mo in range(12):
            sub = tm[tm.index.get_level_values(1) == mo + 1]
            if len(sub): og[mo, n] = np.nanmean(np.maximum(sub['gpp'].values, 0)); oe[mo, n] = np.nanmean(sub['et'].values)
    def run_cycle(st, sp, pft, key, n_eval=3, ref=None):
        ctx = dict(i=i, j=j, lat_rad=lat_rad, sp=sp, pft=pft, consts=consts, reader=rd); GM = []; GD = []; EM = []; LL = []; LU = []; LM = []
        for k, yr in enumerate(YEARS):
            REC.clear(); st, key, d = one_year(st, key, yr, ctx, 0); st = hold(st)
            if k >= len(YEARS) - n_eval:
                GM.append(d['gpp_monthly'] * 1000); gd = np.full((366, len(sel)), np.nan, np.float32); gd[:d['gpp_daily'].shape[0]] = d['gpp_daily'] * 1000; GD.append(gd); EM.append(d['aet_monthly'])
                summer = [5, 6, 7] if NH else [11, 0, 1]; LL.append(np.mean([REC[mo][0] for mo in summer], 0)); LU.append(np.mean([REC[mo][1] for mo in summer], 0))
                LM.append(np.array([REC[mo][0] + REC[mo][1] for mo in range(12)]))          # (12, n) displayed LAI
        met = metrics(np.array(GM), np.array(GD), np.array(EM), np.array(LL), np.array(LU), og, oe, NH, ref=ref, lai_m=np.array(LM))
        met['_gm'] = np.array(GM).mean(0); met['_lm'] = np.array(LM).mean(0); met['_gd'] = np.array(GD).mean(0)
        return st, key, met
    # base: two spin cycles, then one scored cycle
    key = jax.random.PRNGKey(0); st = st0
    for c in range(2): st, key, _ = run_cycle(st, sp0, pft0, key)
    st_base = st; key_base = key
    _, _, base = run_cycle(st_base, sp0, pft0, key_base); REF = (base['_amp_lo'], base['_amp_hi'])
    _, _, base = run_cycle(st_base, sp0, pft0, key_base, ref=REF)                 # re-score the base at its own fixed level
    print('base done', flush=True)
    rows = []
    DUMP = {}
    def add(label, met):
        DUMP[label] = {'gpp': met['_gm'], 'lai': met['_lm'], 'gpp_daily': met['_gd']}
        for n, s in enumerate(sites): rows.append(dict(pert=label, site=s, igbp=META.loc[s, 'IGBP'], **{k: float(v[n]) for k, v in met.items() if not k.startswith('_')}))
    add('base', base)
    pool = PHENOLOGY_WIDE if a.set == 'phenology' else SEASONAL_SET if a.set == 'seasonal' else PERTURBATIONS
    todo = [p for p in pool if (not a.only or p[0] in a.only.split(','))]
    for label, kind, spec in todo:
        sp, pft = sp0, pft0; vm, sl = vmax0.copy(), specla0.copy(); ex = {}
        if kind == 'vmax': vm = vmax0 * spec
        elif kind == 'specla': sl = specla0 * spec
        elif kind == 'tauleaf': pft = eqx.tree_at(lambda q: q.tauleaf, pft0, jnp.asarray(np.asarray(pft0.tauleaf) * spec, dtype=pft0.tauleaf.dtype))
        elif kind == 'aleaf':
            al = np.clip(np.asarray(pft0.aleaf) + spec, 0.05, 0.9); aw = np.clip(np.asarray(pft0.awood) - spec, 0.0, 0.9); ar = 1.0 - al - aw
            pft = eqx.tree_at(lambda q: (q.aleaf, q.awood, q.aroot), pft0, tuple(jnp.asarray(x, dtype=pft0.aleaf.dtype) for x in (al, aw, ar)))
        elif kind == 'extras': ex = {k: f(float(_extra_value(sp0, pft0, k))) for k, f in spec.items()}
        sp, pft = apply_params(sp, pft, vm, sl, ex)
        _, _, met = run_cycle(st_base, sp, pft, key_base, ref=REF); add(label, met); print(f'{label}: done', flush=True)
        pd.DataFrame(rows).to_csv(f'{a.out}/sensitivity_runs.csv', index=False)
    df = pd.DataFrame(rows); df.to_csv(f'{a.out}/sensitivity_runs.csv', index=False)
    if a.dump_monthly: np.savez(f'{a.out}/monthly_series.npz', sites=np.array(sites), og=og, oe=oe, **{f'{lab}|{k}': v for lab, m in DUMP.items() for k, v in m.items()})
    base_df = df[df['pert'] == 'base'].set_index('site'); out = []
    for label in [p[0] for p in todo]:
        d = df[df['pert'] == label].set_index('site')
        for c, grp in d.groupby('igbp'):
            b = base_df.loc[grp.index]
            out.append(dict(pert=label, igbp=c, n=len(grp), d_gpp_ratio=float((grp['gpp_ratio'] - b['gpp_ratio']).mean()), d_junjul_ratio=float((grp['junjul_ratio'] - b['junjul_ratio']).mean()), d_winter_share=float((grp['winter_share'] - b['winter_share']).mean()),
                            d_leafout_days=float((grp['leafout_doy'] - b['leafout_doy']).mean()), d_senesc_days=float((grp['senesc_doy'] - b['senesc_doy']).mean()), d_lai_on_month=float((grp['lai_on_month'] - b['lai_on_month']).mean()), d_lai_off_month=float((grp['lai_off_month'] - b['lai_off_month']).mean()), d_et_ratio=float((grp['et_ratio'] - b['et_ratio']).mean()), d_lai_summer=float((grp['lai_summer'] - b['lai_summer']).mean())))
    sens = pd.DataFrame(out); sens.to_csv(f'{a.out}/sensitivity_by_class.csv', index=False)
    pd.set_option('display.width', 250); print('\nBASE by class:'); print(base_df.groupby('igbp')[['gpp_ratio', 'junjul_ratio', 'winter_share', 'leafout_doy', 'senesc_doy', 'lai_on_month', 'lai_off_month', 'lai_range', 'et_ratio', 'lai_summer']].mean().round(2).to_string())
    print('\nCHANGE from base, mean per class:'); print(sens.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
