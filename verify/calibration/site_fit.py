"""Per-class site calibration in fixed-vegetation mode (CALIBRATION.md 22, stage C1 of the site-first plan).

    python verify/calibration/site_fit.py --igbp ENF --params python-port/configs/params/default_seasonal.yaml \\
        --fit vmax:temp_con_ever,vmax:boreal_con_ever,acclim_tau,acclim_s0,acclim_smax --n-evals 200 --workers 4 --out <dir>

Towers: every tower of the IGBP class in the 2008-2012 batch (`site_batches.txt`, line 1). Composition from the site's
class (`site_run.prescribe`, IBIS's temperate/boreal rule); upper-canopy leaf area held at the tower's measured maximum
(ESSD 2025 site attributes) where one exists, else at the starting state; composition and leaf carbon enforced every
year (`site_run.enforce_prescription`), so allocation, turnover and stand survival are OUT of this stage (C2).
Split: a third of the towers held out (seeded); 2008 is warm-up, 2009-2011 train, 2012 is held out at every tower.

Objective: mean over training towers of the normalised squared monthly misfit
    GPP: target = mean of the night-time and day-time partitions, sigma = max(|NT - DT| / 2, 10 % of target, 5 gC m-2 mo-1)
    ET : target = energy-balance-corrected LE (LE_CORR), sigma = max(|LE_CORR - LE_F_MDS| / 2, 10 %, 3 mm mo-1)
    L = mean_towers [ mean_months(r_GPP^2) + w_et * mean_months(r_ET^2) ]
over months whose NEE (resp. LE) is at least half measured or good-quality gap-filled (MM `*_QC` >= 0.5).
NEE is reported, not fitted: in fixed-vegetation mode the soil carbon pools are the grid state's, not the site's, so an
NEE fit would tune respiration to pool disequilibrium (the literature adds a per-site soil-carbon scaling for that;
stage C2). CMA-ES on bounds-normalised coordinates; every evaluation logs the training loss, the held-out towers' loss
and the held-out year's loss to history.csv.
"""
import sys, os, glob, json, time, types, argparse, numpy as np, pandas as pd, xarray as xr, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
import jax, jax.numpy as jnp, equinox as eqx
from fit_carbon import LATV, LONV, MASK, GSWP3, NPOI_FULL, PFT_NAMES, modis_at, apply_params, trace_extras, enable_compile_cache, EXTRAS, _enc, _dec, _extra_value, VMAX_LO, VMAX_HI, STATE_EXTRAS
import site_transient; from site_transient import one_year
from site_run import prescribe, tower_mat, enforce_prescription, META, SEED_LAI, SEED_WOOD
from site_attributes import prescribe_from_attributes
from tower_forcing import TowerForcingReader
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum
from ibis.state.et_parts import create_et_parts; from ibis.state.seasonal import create_seasonal; from ibis.state.nee_parts import create_nee_parts
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32
V = '/shared/nas2/Hangkai/ibis/verify_outputs'; S = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site_data')   # tower batches and inventory
FX = '/shared/nas2/Hangkai/ibis/inputs/sites/FLUXNET2015_FULLSET/daily_monthly_yearly'
YEARS = [2008, 2009, 2010, 2011, 2012]; TRAIN_Y = [1, 2, 3]; HOLD_Y = [4]         # indices into YEARS; 2008 is warm-up (--years, --hold-last)


def configure_years(a):
    """YEARS / TRAIN_Y / HOLD_Y from --years A-B and --hold-last K: the first year is warm-up, the last K are held out.
    Module globals, so workers (which rebuild the setup from the same arguments) score the same years."""
    global YEARS, TRAIN_Y, HOLD_Y
    y0, y1 = (int(x) for x in getattr(a, 'years', '2008-2012').split('-')); k = int(getattr(a, 'hold_last', 1))
    YEARS = list(range(y0, y1 + 1)); n = len(YEARS)
    if n - 1 - k < 1: raise SystemExit(f'--years {a.years} with --hold-last {k} leaves no training year after the warm-up')
    TRAIN_Y = list(range(1, n - k)); HOLD_Y = list(range(n - k, n))


def tower_obs(site):
    """Monthly GPP target and sigma (gC m-2 mo-1), ET target and sigma (mm mo-1), NEE (gC m-2 mo-1), QC masks; (5, 12) each."""
    f = glob.glob(f'{FX}/FLX_{site}_FLUXNET2015_FULLSET_MM_*.csv')
    out = {k: np.full((len(YEARS), 12), np.nan) for k in ('gpp', 'gpp_sig', 'et', 'et_sig', 'nee')}; out['et_src'] = 'LE_CORR'
    out['season'] = [5, 6, 7] if float(META.loc[site, 'LOCATION_LAT']) >= 0 else [11, 0, 1] if site in META.index else [5, 6, 7]   # summer months for the magnitude term
    if NEE_T is not None and site in list(NEE_T['sites']):                              # day/night NEE targets on the simulated years (gC m-2 month-1, + = release)
        k = list(NEE_T['sites']).index(site); ty = [int(y) for y in NEE_T['years']]
        for tag in ('night', 'day'):
            arr = np.full((len(YEARS), 12), np.nan)
            for yi, y in enumerate(YEARS):
                if y in ty: arr[yi] = NEE_T[f'sum_{tag}'][ty.index(y), :, k]
            out[f'nee_{tag}'] = arr; out[f'nee_{tag}_sig'] = np.maximum(0.15 * np.abs(arr), 5.0)             # 15 % or 5 gC m-2 month-1, whichever is larger
    if not f: return out
    d = pd.read_csv(f[0]); t = pd.to_datetime(d['TIMESTAMP'].astype(str).str[:6], format='%Y%m'); d['y'] = t.dt.year; d['m'] = t.dt.month; nd = t.dt.days_in_month
    w = lambda c: d[c].where(d[c] > -9000) if c in d else pd.Series(np.nan, index=d.index)
    nt = w('GPP_NT_VUT_REF') * nd; dt = w('GPP_DT_VUT_REF') * nd; g = pd.concat([nt, dt], axis=1).mean(axis=1)
    gs = np.maximum.reduce([(nt - dt).abs() / 2.0, 0.1 * g.abs(), pd.Series(5.0, index=d.index)])
    k = 86400.0 / 2.45e6 * nd; le = w('LE_CORR') * k; lef = w('LE_F_MDS') * k
    es = np.maximum.reduce([(le - lef).abs() / 2.0, 0.1 * le.abs(), pd.Series(3.0, index=d.index)])
    if le.notna().sum() == 0 and lef.notna().sum() > 0:
        # no energy-balance-corrected LE at this tower (e.g. US-Ha1): the gap-filled LE, with a wider sigma since the
        # closure correction is unknown; fill-value years drop out through LE_F_MDS_QC below
        le = lef; es = np.maximum.reduce([0.15 * le.abs(), pd.Series(3.0, index=d.index)]); out['et_src'] = 'LE_F_MDS'
    qn = w('NEE_VUT_REF_QC'); ql = w('LE_F_MDS_QC'); nee = w('NEE_VUT_REF') * nd
    for yi, yr in enumerate(YEARS):
        for mo in range(12):
            r = (d['y'] == yr) & (d['m'] == mo + 1)
            if not r.any(): continue
            r = r.idxmax()
            if pd.notna(qn[r]) and qn[r] >= 0.5 and pd.notna(g[r]): out['gpp'][yi, mo] = max(g[r], 0.0); out['gpp_sig'][yi, mo] = gs[r]
            if pd.notna(ql[r]) and ql[r] >= 0.5 and pd.notna(le[r]): out['et'][yi, mo] = le[r]; out['et_sig'][yi, mo] = es[r]
            if pd.notna(nee[r]): out['nee'][yi, mo] = nee[r]
    return out


ROBUST = None      # set by main() from --robust: dict(cap_month, cap_tower, w_mag, tol) (CALIBRATION.md 34)
TARGET = 'gpp'     # 'gpp' (GPP + w_et ET) or 'nee' (night-time + w_day day-time NEE; CALIBRATION.md 36), set by main() from --target
NEE_T = None       # the tower day/night NEE targets (tower_nee_daynight.py npz), loaded by main() from --nee-targets


def misfit_nee(dn, obs, n, years, w_day):
    """NEE-stage objective (CALIBRATION.md 36): normalised squared monthly misfit of night-time NEE + w_day * day-time NEE over
    `years`, the same caps and annual-magnitude terms as the robust GPP objective (ln of the ratio of annual sums, same sign)."""
    rn = ((dn[years, :, 0, n] - obs['nee_night'][years]) / obs['nee_night_sig'][years]) ** 2
    rd = ((dn[years, :, 1, n] - obs['nee_day'][years]) / obs['nee_day_sig'][years]) ** 2
    if ROBUST: rn = np.minimum(rn, ROBUST['cap_month']); rd = np.minimum(rd, ROBUST['cap_month'])
    ln = np.nanmean(rn) if np.isfinite(rn).any() else np.nan; ld = np.nanmean(rd) if np.isfinite(rd).any() else np.nan
    if not np.isfinite(ln): return np.nan
    L = ln + (w_day * ld if np.isfinite(ld) else 0.0)
    if ROBUST:
        mag = 0.0
        for j, key in ((0, 'nee_night'), (1, 'nee_day')):
            g, t = dn[years, :, j, n], obs[key][years]; ok = np.isfinite(t)
            if ok.sum() >= 3 and np.nansum(t[ok]) * g[ok].sum() > 0:
                mag += min((np.log(abs(g[ok].sum()) / abs(np.nansum(t[ok]))) / ROBUST['tol']) ** 2, ROBUST['cap_month'])
            elif ok.sum() >= 3: mag += ROBUST['cap_month']                                              # opposite sign of the annual sum
        L = L + ROBUST['w_mag'] * mag; L = min(L, ROBUST['cap_tower'])
    return L


def misfit(gm, em, obs, n, years, w_et, ne=None):
    if TARGET == 'nee': return misfit_nee(ne, obs, n, years, w_et)
    """Normalised squared monthly misfit of one tower over `years` (indices), GPP + w_et * ET; NaN if no usable month.

    With ROBUST set (site_fit --robust): every month's squared misfit is capped at cap_month, the tower's annual and
    summer GPP ratios enter as (ln R / tol)^2 (each capped at cap_month) times w_mag, and the tower's total is capped
    at cap_tower, so no month and no tower can dominate the mean over towers (section 33)."""
    rg = ((gm[years, :, n] - obs['gpp'][years]) / obs['gpp_sig'][years]) ** 2
    re = ((em[years, :, n] - obs['et'][years]) / obs['et_sig'][years]) ** 2
    if ROBUST: rg = np.minimum(rg, ROBUST['cap_month']); re = np.minimum(re, ROBUST['cap_month'])
    lg = np.nanmean(rg) if np.isfinite(rg).any() else np.nan; le = np.nanmean(re) if np.isfinite(re).any() else np.nan
    if not np.isfinite(lg): return np.nan
    L = lg + (w_et * le if np.isfinite(le) else 0.0)
    if ROBUST:
        g, t = gm[years, :, n], obs['gpp'][years]; ok = np.isfinite(t); season = obs.get('season', [5, 6, 7])
        mag = 0.0
        for sel in (ok, ok & np.isin(np.arange(12)[None, :].repeat(len(years), 0), season)):
            if sel.sum() >= 3 and np.nansum(t[sel]) > 0:
                mag += min((np.log(max(g[sel].sum(), 1e-3) / np.nansum(t[sel])) / ROBUST['tol']) ** 2, ROBUST['cap_month'])
        L = L + ROBUST['w_mag'] * mag
        L = min(L, ROBUST['cap_tower'])
    return L


def _args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--igbp', required=True); ap.add_argument('--params', required=True); ap.add_argument('--fit', required=True)
    ap.add_argument('--n-evals', type=int, default=200); ap.add_argument('--sigma', type=float, default=0.2); ap.add_argument('--w-et', type=float, default=0.5)
    ap.add_argument('--holdout', type=float, default=1 / 3); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--out', required=True)
    ap.add_argument('--only-baseline', action='store_true', help='score the starting parameters and stop (smoke test)')
    ap.add_argument('--profile', type=int, default=0, help='with --start: one-at-a-time profile of the loss around the start, N points across each parameter\'s bounds (others held); writes profile.csv and stops')
    ap.add_argument('--years', default='2008-2012', help='simulated years A-B: the first is warm-up (default: the 2008-2012 batch)')
    ap.add_argument('--hold-last', type=int, default=1, help='hold out the last K simulated years (scored, not fitted)')
    ap.add_argument('--lai', default='', help='SITE=value[,SITE=value]: leaf area to hold where the site attributes have none')
    ap.add_argument('--co2', default='fixed', choices=['fixed', 'series'], help="'fixed' 385 ppm (the batch fits) or the historical annual series")
    ap.add_argument('--lai-mode', default='', choices=['', 'attrs-modis'], help="'attrs-modis': non-forest towers (OSH, CSH, SAV, WSA, GRA) held at the MODIS cell maximum LAI, forests at the attribute maximum (CALIBRATION.md 31); --lai still overrides per tower")
    ap.add_argument('--robust', default='', help="bounded objective 'cap_month,cap_tower,w_mag,tol' e.g. 25,30,1,0.15: capped monthly misfits, annual and summer GPP ratio terms (ln R / tol)^2, capped per-tower total (CALIBRATION.md 34)")
    ap.add_argument('--et-corr-only', action='store_true', help='ET term only at towers with energy-balance-corrected LE (LE_CORR)')
    ap.add_argument('--et-sites', default='', help='comma list: only these towers carry the ET term (towers without energy-balance-corrected LE get 0); default all')
    ap.add_argument('--pft', default='', help="composition overrides SITE=idx,idx;SITE=... (IBIS PFT indices) where the IGBP rule is wrong, e.g. ZM-Mon=1 (tropical deciduous), RU-Cok=9,11 (deciduous shrub tundra)")
    ap.add_argument('--vmax-bounds', default='', help="LO,HI in umol m-2 s-1 at 25 C for every fitted vmax (e.g. the TRY mean +- SD of the PFT); converted to IBIS's 15 C basis with its own temperature function (CALIBRATION.md 29)")
    ap.add_argument('--hourly-sw', action='store_true', help="drive radiation with the tower's hourly shortwave (Spitters diffuse split) instead of the weather generator's cycle (CALIBRATION.md 27)")
    ap.add_argument('--sites', default='', help='comma list: only these towers of the class (probes, single-tower tests)')
    ap.add_argument('--start', default='', help="best.json of an earlier fit: start from its parameters (with --only-baseline: a replay)")
    ap.add_argument('--target', default='gpp', choices=['gpp', 'nee'], help="'nee': fit night-time + w_et x day-time NEE (tower_nee_daynight.py targets via --nee-targets) instead of GPP + w_et x ET (CALIBRATION.md 36)")
    ap.add_argument('--nee-targets', default='', help='npz from tower_nee_daynight.py (required with --target nee)')
    ap.add_argument('--start-state', default='', help='state.npz from site_pool_spinup.py for the same towers: its carbon pools replace the grid pools before the warm-up cycle')
    ap.add_argument('--popsize', type=int, default=0, help='CMA-ES population (0: the default 4 + 3 ln n); a multiple of --workers wastes no round')
    ap.add_argument('--workers', type=int, default=1, help='evaluate each CMA-ES generation in this many spawned processes; '
                    'at a few towers the GPU is latency-bound, so concurrent evaluations cost little more than one')
    return ap.parse_args(argv)


def setup(a, spin=True, verbose=True):
    """Towers, the prescribed start state, parameter coding and observations; with `spin`, one spin-up cycle and `run`.
    Deterministic in `a`, so the parent and every worker build the same towers, split and coding."""
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    configure_years(a)
    global TARGET, NEE_T                                                                   # also in each spawned worker (setup is rebuilt there)
    TARGET = getattr(a, 'target', 'gpp')
    if getattr(a, 'nee_targets', ''): NEE_T = np.load(a.nee_targets)
    if TARGET == 'nee' and NEE_T is None: raise SystemExit('--target nee needs --nee-targets')
    b1 = open(f'{S}/site_batches.txt').readline().split()[1].split(',')
    sites = b1 if a.igbp == 'ALL' else [s for s in b1 if META.loc[s, 'IGBP'] == a.igbp]      # 'ALL': the joint fit across classes (CALIBRATION.md 32)
    if getattr(a, 'sites', ''):
        want = a.sites.split(','); bad = [s for s in want if s not in sites]
        if bad:                                                                        # towers outside the 2008-2012 batch: any tower of the class with a daily file
            other = [b for b in bad if b in META.index and (a.igbp == 'ALL' or META.loc[b, 'IGBP'] == a.igbp)]
            if len(other) < len(bad): raise SystemExit(f'{[b for b in bad if b not in other]} are not {a.igbp} towers')
            say(f'{other}: {a.igbp} towers outside the 2008-2012 batch, taken with their own years')
        sites = want
    rng = np.random.default_rng(a.seed); order = rng.permutation(len(sites)); n_hold = int(round(a.holdout * len(sites)))
    hold = sorted(sites[k] for k in order[:n_hold]); train = sorted(s for s in sites if s not in hold); sites = train + hold
    say(f'{a.igbp}: {len(sites)} towers; train {train}; hold-out {hold}')
    # --- start state at the tower cells, prescribed and fixed
    m = np.array(xr.open_dataset(MASK).landfrac.values) > 0; fl = np.where(m.ravel())[0]; posmap = {g: p for p, g in enumerate(fl)}
    i = np.array([int(np.argmin(np.abs(LATV - META.loc[s, 'LOCATION_LAT']))) for s in sites], np.int32); j = np.array([int(np.argmin(np.abs(LONV - META.loc[s, 'LOCATION_LONG']))) for s in sites], np.int32)
    sel = np.array([posmap[ii * 720 + jj] for ii, jj in zip(i, j)]); lat_rad = np.deg2rad(LATV[i]).astype(np.float32); ns = len(sites)
    full, _ = load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz', create_ibis_state(GridConfig(npoi=NPOI_FULL, nbedrock=3)))
    cut = lambda x: x[sel] if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL else x
    st = _cast_tree_f32(jax.tree.map(cut, full)); del full
    st = eqx.tree_at(lambda s: s.stress, st, create_stress_accum(ns), is_leaf=lambda x: x is None)
    st = eqx.tree_at(lambda s: s.etparts, st, create_et_parts(ns), is_leaf=lambda x: x is None)
    st = eqx.tree_at(lambda s: s.seasonal, st, _cast_tree_f32(create_seasonal(ns, 5.0)), is_leaf=lambda x: x is None)
    st = eqx.tree_at(lambda s: s.neeparts, st, _cast_tree_f32(create_nee_parts(ns)), is_leaf=lambda x: x is None)
    sp0, pft0, _ = build_all_params(ns, lat_rad, load_physics_params(a.params))
    if getattr(a, 'start', ''):
        # --start best.json: EVERY entry becomes part of the base parameter set (the fitted names are then re-decoded from
        # it as the CMA-ES start). Until 2026-10-04 only the fitted names were taken from it, so a fit of respiration
        # parameters started from a GPP/ET best.json ran on the yaml's (default) photosynthesis parameters.
        _prev = json.load(open(a.start))['params']; _vm = np.asarray(sp0.stomata_params.vmax_pft).copy(); _ex = {}
        for _nm, _v in _prev.items():
            if _nm.startswith('vmax:'): _vm[PFT_NAMES.index(_nm[5:])] = _v
            elif _nm in EXTRAS and _nm not in STATE_EXTRAS: _ex[_nm] = _v
        sp0, pft0 = apply_params(sp0, pft0, _vm, np.asarray(pft0.specla), _ex); say(f'base parameters from {a.start}: {len(_prev)} entries applied')
    sp0, pft0 = trace_extras(sp0, pft0)   # one compile for spin-up and every evaluation
    specla = np.asarray(pft0.specla); vmax0 = np.asarray(sp0.stomata_params.vmax_pft)
    mask = np.zeros((ns, 12), np.float32); fu = np.zeros(ns, np.float32); flw = np.zeros(ns, np.float32)
    cb = np.asarray(st.vegetation.cbiol).copy(); cw = np.asarray(st.vegetation.cbiow).copy(); cr = np.asarray(st.vegetation.cbior).copy(); lai_src = {}
    for n, s in enumerate(sites):
        mat = META.loc[s, 'MAT'] if 'MAT' in META else np.nan
        if not (isinstance(mat, (int, float)) and np.isfinite(mat)): mat = tower_mat(s)
        pfts, fu[n], flw[n], _ = prescribe(META.loc[s, 'IGBP'], float(META.loc[s, 'LOCATION_LAT']), float(mat) if pd.notna(mat) else None)
        over = {kv.split('=')[0]: [int(x) for x in kv.split('=')[1].split(',')] for kv in getattr(a, 'pft', '').split(';') if kv}
        if s in over: pfts = over[s]; say(f'{s}: composition overridden to PFTs {pfts}')
        for p in range(12):
            if p in pfts:
                mask[n, p] = 1.0
                if cb[n, p] * specla[p] < 0.3: cb[n, p] = SEED_LAI[p] / specla[p]; cw[n, p] = SEED_WOOD[p]; cr[n, p] = 0.5
            else: cb[n, p] = 0.0; cw[n, p] = 0.0; cr[n, p] = 0.0
        at = prescribe_from_attributes(s)
        if at is not None and np.isfinite(at['lai_max']) and at['lai_max'] > 0:
            cb[n] = mask[n] / max(mask[n].sum(), 1) * at['lai_max'] / np.maximum(specla, 1e-6); lai_src[s] = f"attributes {at['lai_max']:.1f}"
        else: lai_src[s] = f"state {float((cb[n] * specla).sum()):.1f}"
        if getattr(a, 'lai_mode', '') == 'attrs-modis' and META.loc[s, 'IGBP'] in ('OSH', 'CSH', 'SAV', 'WSA', 'GRA'):
            mm = float(np.asarray(modis_at(i, j))[n])
            if np.isfinite(mm) and mm > 0: cb[n] = mask[n] / max(mask[n].sum(), 1) * mm / np.maximum(specla, 1e-6); lai_src[s] = f'MODIS cell maximum {mm:.1f}'
        given = dict(kv.split('=') for kv in getattr(a, 'lai', '').split(',') if kv)
        if s in given:
            cb[n] = mask[n] / max(mask[n].sum(), 1) * float(given[s]) / np.maximum(specla, 1e-6); lai_src[s] = f"given {float(given[s]):.1f}"
    st = eqx.tree_at(lambda s: (s.vegetation.cbiol, s.vegetation.cbiow, s.vegetation.cbior, s.vegetation.plai, s.vegetation.exist, s.vegetation.fu, s.vegetation.fl),
                     st, tuple(jnp.asarray(x, dtype=jnp.float32) for x in (cb, cw, cr, cb * specla[None, :], mask, fu, flw)))
    say('leaf area: ' + ', '.join(f'{s} {lai_src[s]}' for s in sites))
    fix = cb.copy(); hold_fn = lambda s_: enforce_prescription(s_, mask, specla, fix_cbiol=fix, veg_params=pft0)
    obs = [tower_obs(s) for s in sites]
    # --- parameter vector: 'vmax:<pft>' or an EXTRAS name; bounds-normalised coordinates
    names = a.fit.split(',')
    for nm in names:                                                                  # fail before the spin-up, with the valid names
        if nm.startswith('vmax:') and nm[5:] not in PFT_NAMES: raise SystemExit(f"unknown PFT in {nm!r}; PFT names: {PFT_NAMES}")
        if not nm.startswith('vmax:') and nm not in EXTRAS: raise SystemExit(f"unknown parameter {nm!r}; EXTRAS: {sorted(EXTRAS)}")
    vlo, vhi = VMAX_LO.copy(), VMAX_HI.copy()
    if getattr(a, 'vmax_bounds', ''):
        from ibis.physics.stomata import _temp_dep_params
        tv = lambda t: float(_temp_dep_params(jnp.asarray([t + 273.16], jnp.float32), sp0.stomata_params)[4][0])
        r25 = tv(25.0) / tv(15.0)                                                     # IBIS vmax_pft is at 15 C
        for item in a.vmax_bounds.split(';'):                                        # 'lo,hi' for every vmax, or 'pft=lo,hi;pft=lo,hi'
            key, _, rng = item.rpartition('=') if '=' in item else ('', '', item)
            b_lo, b_hi = (float(x) * 1e-6 / r25 for x in rng.split(','))
            idx = [PFT_NAMES.index(key)] if key else range(12)
            for k in idx: vlo[k] = b_lo; vhi[k] = b_hi
            say(f'vmax bounds {key or "all"}: {rng} umol at 25 C = {b_lo * 1e6:.1f}-{b_hi * 1e6:.1f}e-6 at 15 C (ratio {r25:.4f})')
    def enc_bounds(nm):
        if nm.startswith('vmax:'):
            k = PFT_NAMES.index(nm[5:]); return np.log(vlo[k]), np.log(vhi[k])
        return _enc(nm, EXTRAS[nm][1]), _enc(nm, EXTRAS[nm][2])
    def start(nm):
        if nm.startswith('vmax:'): return np.log(vmax0[PFT_NAMES.index(nm[5:])])
        return _enc(nm, float(_extra_value(sp0, pft0, nm)))
    lo = np.array([enc_bounds(nm)[0] for nm in names]); hi = np.array([enc_bounds(nm)[1] for nm in names])
    if a.start:                                                                       # an earlier fit's parameters (natural units)
        prev = json.load(open(a.start))['params']
        start0 = start
        def start(nm):
            if nm not in prev: return start0(nm)
            return np.log(prev[nm]) if nm.startswith('vmax:') else _enc(nm, float(prev[nm]))
    z0 = np.clip((np.array([start(nm) for nm in names]) - lo) / (hi - lo), 0.0, 1.0)
    def decode(z):
        x = lo + np.clip(z, 0, 1) * (hi - lo); vm = vmax0.copy(); ex = {}
        for nm, xv in zip(names, x):
            if nm.startswith('vmax:'): vm[PFT_NAMES.index(nm[5:])] = np.exp(xv)
            else: ex[nm] = _dec(nm, xv)
        return vm, ex
    C = types.SimpleNamespace(sites=sites, train=train, hold=hold, ns=ns, names=names, z0=z0, decode=decode, obs=obs, lai_src=lai_src)
    if not spin: return C
    # --- spin once with the starting parameters (fixed vegetation: soil water/temperature and seasonal states settle)
    rd = TowerForcingReader(sites, GSWP3Reader(GSWP3), use_hourly_sw=getattr(a, 'hourly_sw', False)); consts = PhysicalConstants()
    from ibis.io.co2 import co2_molmol          # an ARRAY, not a Python float: a float is static under filter_jit and recompiles per value
    site_transient.apply_co2 = (lambda q, y, series=None: eqx.tree_at(lambda z: z.atm.co2conc, q, jnp.asarray(co2_molmol(y) if a.co2 == 'series' else 385.0e-6, jnp.float32)))
    ctx0 = dict(i=i, j=j, lat_rad=lat_rad, sp=sp0, pft=pft0, consts=consts, reader=rd); key0 = jax.random.PRNGKey(0); t0 = time.time()
    if getattr(a, 'start_state', ''):                                                                   # equilibrated pools (site_pool_spinup.py)
        from ibis.state.model_state import create_ibis_state as _cis
        _z = np.load(a.start_state, allow_pickle=True); ssites = [str(x) for x in _z['meta_sites']] if 'meta_sites' in _z.files else list(sites); nss = len(ssites)   # save_state stores metadata as meta_<key>
        tmpl = st if nss == ns else _cast_tree_f32(jax.tree.map(lambda x: (jnp.repeat(x[:1], nss, axis=0) if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == ns else x), st))
        eq, _ = load_state(a.start_state, tmpl); sel_s = np.array([ssites.index(s_) for s_ in sites])         # the fit's towers inside the spin-up set
        pick = lambda x: (x[sel_s] if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == nss else x)
        st = eqx.tree_at(lambda s: s.carbon, st, jax.tree.map(pick, eq.carbon)); say(f'carbon pools from {a.start_state} ({nss} towers, {ns} used)')
    for yr in YEARS: st, key0, _ = one_year(st, key0, yr, ctx0, 0); st = hold_fn(st)
    print(f'{"" if verbose else f"worker {os.getpid()}: "}spin-up cycle done in {time.time() - t0:.0f} s', flush=True)
    base_state, base_key = st, key0
    def run(vm, ex):
        sp, pft = apply_params(sp0, pft0, vm, specla, ex); ctx = dict(i=i, j=j, lat_rad=lat_rad, sp=sp, pft=pft, consts=consts, reader=rd)
        s_, k_ = base_state, base_key; GM = []; EM = []; NM = []; DN = []
        if 'som_scale' in ex:                                                                            # per-site disequilibrium factor on the SOM pools (CALIBRATION.md 36)
            f = jnp.float32(ex['som_scale']); s_ = eqx.tree_at(lambda s: (s.carbon.csoislon, s.carbon.csoislop, s.carbon.csoipas), s_, (s_.carbon.csoislon * f, s_.carbon.csoislop * f, s_.carbon.csoipas * f))
        for yr in YEARS:
            s_, k_, d = one_year(s_, k_, yr, ctx, 0); s_ = hold_fn(s_); GM.append(d['gpp_monthly'] * 1000); EM.append(d['aet_monthly']); NM.append(d['nee_monthly'] * 1000)
            if 'nee_daynight_monthly' in d: DN.append(d['nee_daynight_monthly'] * 1000)
        return np.array(GM), np.array(EM), (np.array(DN) if (TARGET == 'nee' and DN) else np.array(NM))
    C.run = run; C.base_state, C.base_key, C.ctx0, C.hold_fn = base_state, base_key, ctx0, hold_fn   # for probes
    return C


_W = None                               # a worker's setup (--workers > 1): built once per spawned process by _winit


def _winit(av):
    global _W
    warnings.filterwarnings('ignore'); enable_compile_cache(); _W = setup(argparse.Namespace(**av), spin=True, verbose=False)


def _wready():
    return os.getpid()


def _weval(z):
    t1 = time.time(); vm, ex = _W.decode(np.asarray(z)); gm, em, ne = _W.run(vm, ex)
    return gm, em, ne, time.time() - t1


def main():
    a = _args(); os.makedirs(a.out, exist_ok=True)
    import cma
    enable_compile_cache()
    C = setup(a, spin=True)      # with workers, this spin-up compiles and fills the persistent cache; theirs then load it
    names, train, hold, sites, ns, obs = C.names, C.train, C.hold, C.sites, C.ns, C.obs
    ntr = len(train)
    global ROBUST
    if getattr(a, 'robust', ''):
        cm, ct, wm, tol = (float(x) for x in a.robust.split(',')); ROBUST = dict(cap_month=cm, cap_tower=ct, w_mag=wm, tol=tol); print(f'robust objective: month cap {cm}, tower cap {ct}, magnitude weight {wm}, tolerance {tol}', flush=True)
    et_sites = [x for x in getattr(a, 'et_sites', '').split(',') if x]; w_site = [a.w_et if (not et_sites or s in et_sites) else 0.0 for s in sites]
    if getattr(a, 'et_corr_only', False): w_site = [w if obs[n].get('et_src') == 'LE_CORR' else 0.0 for n, w in enumerate(w_site)]; print(f'ET term at {[s for n, s in enumerate(sites) if w_site[n] > 0]} (energy-balance-corrected LE)', flush=True)
    if et_sites: print(f'ET term at {[s for s in sites if w_site[sites.index(s)] > 0]} only')
    def score(gm, em, ne=None):
        tr = [misfit(gm, em, obs[n], n, TRAIN_Y, w_site[n], ne) for n in range(ntr)]
        ho = [misfit(gm, em, obs[n], n, TRAIN_Y, w_site[n], ne) for n in range(ntr, ns)]
        hy = [misfit(gm, em, obs[n], n, HOLD_Y, w_site[n], ne) for n in range(ns)]
        f = lambda v: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        return f(tr), f(ho), f(hy), tr, ho
    def nee_rmse(ne, towers):
        # monthly NEE RMSE (gC m-2 mo-1) over the training years, mean over towers; tower NEE_VUT_REF, all months with a value
        ne = ne.sum(axis=2) if ne.ndim == 4 else ne
        r = [np.sqrt(np.nanmean((ne[TRAIN_Y, :, n] - obs[n]['nee'][TRAIN_Y]) ** 2)) for n in towers if np.isfinite(obs[n]['nee'][TRAIN_Y]).any()]
        return float(np.mean(r)) if r else np.nan
    pool = None
    if a.workers > 1 and not a.only_baseline:
        import multiprocessing as mp; from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=_winit, initargs=(vars(a),))
        for _ in range(a.workers): pool.submit(_wready)                  # spawn every worker now: they spin up during the baseline
        print(f'{a.workers} worker processes started; each spins up once (loading the cached kernels)', flush=True)
    def evaluate(Z, local=False):
        if pool is not None and not local: return list(pool.map(_weval, [np.asarray(z) for z in Z]))
        out = []
        for z in Z:
            t1 = time.time(); vm, ex = C.decode(z); gm, em, ne = C.run(vm, ex); out.append((gm, em, ne, time.time() - t1))
        return out
    hist = []; best = {'loss': np.inf}
    def objective_batch(Z, local=False):
        Ls = []
        for z, (gm, em, ne, secs) in zip(Z, evaluate(Z, local)):
            vm, ex = C.decode(z); L, Lh, Ly, tr, ho = score(gm, em, ne)
            ne_tr, ne_ho = nee_rmse(ne, range(ntr)), nee_rmse(ne, range(ntr, ns))            # reported, not in the loss
            rec = dict(eval=len(hist) + 1, loss=L, holdout_towers=Lh, holdout_year=Ly, nee_rmse_train=ne_tr, nee_rmse_hold=ne_ho, secs=round(secs, 1), **{nm: (float(vm[PFT_NAMES.index(nm[5:])]) if nm.startswith('vmax:') else float(ex[nm])) for nm in names})
            hist.append(rec); pd.DataFrame(hist).to_csv(f'{a.out}/history.csv', index=False)
            if np.isfinite(L) and L < best['loss']:
                best.update(loss=L, holdout_towers=Lh, holdout_year=Ly, eval=rec['eval'], params={k: rec[k] for k in names}, per_tower_train=dict(zip(train, map(float, tr))), per_tower_hold=dict(zip(hold, map(float, ho))))
                json.dump(best, open(f'{a.out}/best.json', 'w'), indent=1); np.savez(f'{a.out}/best_series.npz', sites=np.array(sites), gpp=gm, et=em, nee=ne, years=np.array(YEARS))
            print(f"eval {rec['eval']:4d}  loss {L:9.3f}  hold-out towers {Lh:9.3f}  hold-out year {Ly:9.3f}  ({rec['secs']:.0f} s)  " + ' '.join(f'{k}={rec[k]:.4g}' for k in names), flush=True)
            Ls.append(L if np.isfinite(L) else 1e6)
        return Ls
    try:
        L0 = objective_batch([C.z0], local=True)[0]; print(f'baseline loss {L0:.4f}', flush=True)       # in the parent, while workers start
        json.dump(dict(igbp=a.igbp, params=a.params, fit=names, train=train, hold=hold, lai=C.lai_src, w_et=a.w_et, years=YEARS, train_years=[YEARS[k] for k in TRAIN_Y], hold_years=[YEARS[k] for k in HOLD_Y], workers=a.workers, popsize=a.popsize, seed=a.seed, sigma=a.sigma, n_evals=a.n_evals, co2=a.co2, lai_given=a.lai, hourly_sw=getattr(a, 'hourly_sw', False), vmax_bounds=getattr(a, 'vmax_bounds', ''), pft_override=getattr(a, 'pft', ''), et_sites=getattr(a, 'et_sites', ''), robust=getattr(a, 'robust', '')), open(f'{a.out}/setup.json', 'w'), indent=1)
        if a.only_baseline: return
        if a.profile:                                                                # one-at-a-time profile around the start (CALIBRATION.md 32)
            fr = np.linspace(0.0, 1.0, a.profile); rows = []
            for k, nm in enumerate(names):
                Z = [np.where(np.arange(len(names)) == k, f, C.z0) for f in fr]
                for f, L in zip(fr, objective_batch(Z)): rows.append(dict(param=nm, frac=float(f), value=float(C.decode(np.where(np.arange(len(names)) == k, f, C.z0))[1][nm]) if not nm.startswith('vmax:') else float(C.decode(np.where(np.arange(len(names)) == k, f, C.z0))[0][PFT_NAMES.index(nm[5:])]), loss=float(L)))
                pd.DataFrame(rows).to_csv(f'{a.out}/profile.csv', index=False); print(f'profile {nm}: ' + ' '.join(f'{r["loss"]:.2f}' for r in rows[-a.profile:]), flush=True)
            return
        es = cma.CMAEvolutionStrategy(C.z0, a.sigma, {'bounds': [0, 1], 'maxfevals': a.n_evals, 'seed': a.seed, 'verbose': -9, **({'popsize': a.popsize} if a.popsize else {})})
        while not es.stop():
            Z = es.ask(); es.tell(Z, objective_batch(Z))
        print(f"\nBEST eval {best.get('eval')}: loss {best['loss']:.4f} (baseline {L0:.4f}); hold-out towers {best.get('holdout_towers', np.nan):.4f}; hold-out year {best.get('holdout_year', np.nan):.4f}", flush=True)
        print('BEST PARAMETERS ' + json.dumps(best.get('params', {})), flush=True)
    finally:
        if pool is not None: pool.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
