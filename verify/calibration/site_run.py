"""Site-level runs at FLUXNET2015 towers with tower meteorology and prescribed vegetation (CALIBRATION.md 20).

    ibis-python verify/calibration/site_run.py --sites US-Ha1,FI-Hyy --params python-port/configs/params/default.yaml \
        --cycles 3 --out /shared/nas2/Hangkai/ibis/verify_outputs/site_runs/default

Protocol per tower: start from the canonical 2010 state at the tower's 0.5-degree cell; prescribe the
composition from the site's IGBP class (`prescribe`), i.e. keep carbon only in the prescribed PFTs, seed
them if empty, fix upper/lower cover, and hold `exist` to the prescription (re-imposed every year, because
`one_year` recomputes it from climate); drive with the tower's own daily meteorology (`TowerForcingReader`,
gaps from GSWP3 at the cell); spin up by cycling the tower's years `--cycles` times with dynamic vegetation
OFF (biomass and LAI evolve, composition does not); then one evaluation pass over the tower years.
Compares monthly GPP (tower GPP_NT_VUT_REF) and ET (LE_F_MDS) and the model's summer displayed LAI with the
MODIS annual maximum at the cell.  Writes <out>/site_runs.npz, site_metrics.csv and one figure per site.
"""
import sys, os, glob, argparse, json, numpy as np, pandas as pd, xarray as xr, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
import jax, jax.numpy as jnp, equinox as eqx
from fit_carbon import LATV, LONV, MASK, GSWP3, NPOI_FULL, PFT_NAMES, modis_at
import site_transient; from site_transient import one_year
from tower_forcing import TowerForcingReader, F as FLUX_DIR
from ibis.dimensions import GridConfig; from ibis.io.restart import load_state
from ibis.state.model_state import create_ibis_state; from ibis.state.stress_accum import create_stress_accum; from ibis.state.et_parts import create_et_parts; from ibis.state.seasonal import create_seasonal
from ibis.io.physics_params import build_all_params, load_physics_params
from ibis.io.gswp3_reader import GSWP3Reader; from ibis.constants import PhysicalConstants
from ibis.timestepping.loops_gswp3 import _cast_tree_f32

V = '/shared/nas2/Hangkai/ibis/verify_outputs'
META = pd.read_csv(f'{FLUX_DIR}/FLUXNET_siteinfo.csv', encoding='utf-8-sig').set_index('SITE_ID')
# IBIS PFT indices: 0 trop_bl_ever 1 trop_bl_dec 2 warmtemp_bl_ever 3 temp_con_ever 4 temp_bl_dec 5 bor_con_ever
#                   6 bor_bl_dec 7 bor_con_dec 8 ever_shrub 9 dec_shrub 10 C4 grass 11 C3 grass
SEED_LAI = {**{k: 4.0 for k in range(8)}, 8: 1.5, 9: 1.5, 10: 2.0, 11: 2.0}
SEED_WOOD = {**{k: 6.0 for k in range(8)}, 8: 0.5, 9: 0.5, 10: 0.0, 11: 0.0}


def prescribe(igbp, lat, mat):
    """IGBP class + location -> (PFT indices, upper cover fu, lower cover fl, note). A first mapping; override per site with --pft."""
    boreal = lat >= 55 or (mat is not None and np.isfinite(mat) and mat < 3.0)
    tropical = abs(lat) < 23
    if igbp == 'ENF': return ([5] if boreal else [3]), 0.9, 0.1, 'boreal' if boreal else 'temperate'
    if igbp == 'DBF': return ([6] if boreal else [4]), 0.9, 0.1, 'boreal' if boreal else 'temperate'
    if igbp == 'DNF': return [7], 0.9, 0.1, ''
    if igbp == 'EBF': return ([0] if tropical else [2]), 0.95, 0.05, 'tropical' if tropical else 'warm-temperate'
    if igbp == 'MF':  return ([5, 6] if boreal else [3, 4]), 0.9, 0.1, 'boreal' if boreal else 'temperate'
    if igbp == 'GRA':
        c4 = (mat is not None and np.isfinite(mat) and mat > 15.0) and abs(lat) < 35
        return ([10] if c4 else [11]), 0.0, 1.0, ('C4 by MAT>15' if c4 else 'C3 by MAT') + ' -- UNCERTAIN, check site description'
    if igbp in ('SAV', 'WSA'):
        return ([1, 10] if tropical else [4, 10]), (0.3 if igbp == 'SAV' else 0.55), 0.8, 'tree + C4 grass'
    if igbp == 'OSH': return ([8, 11] if (mat is not None and np.isfinite(mat) and mat < 10) else [9, 10]), 0.0, 0.6, 'shrub + grass -- UNCERTAIN'
    if igbp == 'CSH': return [8, 11], 0.0, 0.9, 'shrub + grass -- UNCERTAIN'
    raise ValueError(f'{igbp}: not a natural-vegetation class handled here (WET -> methane stage, CRO not in IBIS)')


def enforce_prescription(state, mask, specla, fix_cbiol=None):
    """Re-impose the prescribed composition after a year.

    With `fix_cbiol` the leaf carbon is held at a prescribed value as well, so the stand cannot decay: the fixed-vegetation
    experiment that isolates photosynthesis, water and phenology from allocation and turnover (the phenology still modulates
    the DISPLAYED leaf area through the cold and drought triggers). Without it the leaf carbon evolves under the model's own
    allocation, and a stand whose parameters cannot sustain it dies -- reported by the `stand_died` flag.

    `one_year` recomputes `exist` from the climate at year end, BEFORE the biomass update allocates NPP, so PFTs the site does
    not have receive carbon every year. Their leaf area then enters the canopy regardless of `exist` (the displayed upper LAI
    sums evergreen PFTs unweighted), which turned prescribed deciduous stands into evergreen ones at mild sites. Setting `exist`
    alone is not enough: the carbon pools and `plai` of the non-prescribed PFTs have to be cleared too.
    """
    import jax.numpy as _jnp
    m = _jnp.asarray(mask, dtype=state.vegetation.cbiol.dtype)
    v = state.vegetation
    cbiol = v.cbiol * m if fix_cbiol is None else _jnp.asarray(fix_cbiol, dtype=v.cbiol.dtype)
    cbiow = v.cbiow * m; cbior = v.cbior * m
    plai = cbiol * _jnp.asarray(specla, dtype=cbiol.dtype)[None, :]
    return eqx.tree_at(lambda s: (s.vegetation.exist, s.vegetation.cbiol, s.vegetation.cbiow, s.vegetation.cbior, s.vegetation.plai),
                       state, (_jnp.asarray(mask, dtype=v.exist.dtype), cbiol, cbiow, cbior, plai))


def tower_mat(site):
    """Mean air temperature (C) over the tower record, from the yearly file."""
    f = glob.glob(f'{FLUX_DIR}/FLX_{site}_FLUXNET2015_FULLSET_YY_*.csv'); d = pd.read_csv(f[0])
    c = 'TA_F_MDS' if 'TA_F_MDS' in d else 'TA_F'; v = d[c].where(d[c] > -9000)
    return float(v.mean()) if v.notna().any() else np.nan


def tower_monthly(site):
    """Monthly tower GPP (gC m-2 month-1) and ET (mm month-1) as DataFrames indexed by (year, month)."""
    f = glob.glob(f'{FLUX_DIR}/FLX_{site}_FLUXNET2015_FULLSET_MM_*.csv'); d = pd.read_csv(f[0])
    t = pd.to_datetime(d['TIMESTAMP'].astype(str).str.replace('-', '', regex=False).str[:6], format='%Y%m'); d['year'] = t.dt.year; d['month'] = t.dt.month; d['days'] = t.dt.days_in_month
    g = d['GPP_NT_VUT_REF'].where(d['GPP_NT_VUT_REF'] > -9000) * d['days']
    le = d['LE_F_MDS'].where(d['LE_F_MDS'] > -9000); et = le * 86400 / 2.45e6 * d['days']
    return pd.DataFrame({'year': d['year'], 'month': d['month'], 'gpp': g.values, 'et': et.values}).set_index(['year', 'month'])


def compute_metrics(sites, YEARS, GM, EM, LL, LU, obs_g, obs_e, notes, i, modis):
    """Per-tower metrics from the saved arrays (also used by --recompute)."""
    rows = []; y0, y1 = YEARS[0], YEARS[-1]
    for n, s in enumerate(sites):
        og = np.maximum(np.nanmean(obs_g[:, :, n], 0), 0.0); mg = GM[:, :, n].mean(0); oe = np.nanmean(obs_e[:, :, n], 0); me = EM[:, :, n].mean(0)
        ok = np.isfinite(og); usable = bool(ok.any()) and np.nansum(og) > 100.0      # no tower months, or < 100 gC/m2/yr: ratios are meaningless
        r = np.corrcoef(mg[ok], og[ok])[0, 1] if ok.sum() > 3 else np.nan
        NH = LATV[i[n]] >= 0; summer = [5, 6, 7] if NH else [11, 0, 1]; winter = [10, 11, 0, 1] if NH else [4, 5, 6, 7]
        died = float((LL[:, summer, n] + LU[:, summer, n]).mean()) < 0.2          # the prescribed stand did not survive the spin-up
        rows.append(dict(site=s, igbp=notes[s]['igbp'], pfts='+'.join(notes[s]['pfts']), years=f'{y0}-{y1}', usable=bool(usable), stand_died=bool(died), gpp_model=mg.sum() / 1000, gpp_tower=np.nansum(og) / 1000, gpp_ratio=mg.sum() / np.nansum(og) if usable else np.nan,
                         gpp_r=r, gpp_rmse=float(np.sqrt(np.nanmean((mg - og) ** 2))), winter_share_model=float(mg[winter].sum() / max(mg.sum(), 1e-6)), winter_share_tower=float(np.nansum(og[winter]) / np.nansum(og)) if usable else np.nan, peak_model=int(np.argmax(mg) + 1), peak_tower=int(np.nanargmax(og) + 1) if ok.any() else -1,
                         et_model=me.sum(), et_tower=np.nansum(oe) if np.isfinite(oe).any() else np.nan, et_ratio=me.sum() / np.nansum(oe) if np.isfinite(oe).any() and np.nansum(oe) > 0 else np.nan,
                         lai_summer_model=float((LL[:, summer, n] + LU[:, summer, n]).mean()), lai_modis_max=float(modis[n]), fu=notes[s]['fu'], fl=notes[s]['fl'], note=notes[s]['note']))
    return pd.DataFrame(rows)


def recompute(out):
    """Metrics from a saved run directory (site_runs.npz + site_runs.json)."""
    z = np.load(f'{out}/site_runs.npz', allow_pickle=True); meta = json.load(open(f'{out}/site_runs.json'))
    sites = [str(x) for x in z['sites']]; YEARS = [int(y) for y in z['years']]; i = z['i']; j = z['j']
    df = compute_metrics(sites, YEARS, z['gpp'], z['et'], z['lai_l'], z['lai_u'], z['obs_gpp'], z['obs_et'], meta['notes'], i, modis_at(i, j))
    df.to_csv(f'{out}/site_metrics.csv', index=False); pd.set_option('display.width', 250); print(df.drop(columns=['note']).to_string(index=False, float_format=lambda x: f'{x:.2f}')); return df


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--recompute': recompute(sys.argv[2]); return
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', required=True); ap.add_argument('--params', required=True); ap.add_argument('--cycles', type=int, default=3)
    ap.add_argument('--out', required=True); ap.add_argument('--pft', default='', help='overrides: SITE=pft,pft;SITE=...'); ap.add_argument('--years', default='', help='SITE=y0-y1;... restrict tower years')
    ap.add_argument('--no-hh', action='store_true', help='ignore half-hourly files (diurnal range from GSWP3)')
    ap.add_argument('--hourly-sw', action='store_true', help="the tower's hourly shortwave instead of the weather generator's cycle (CALIBRATION.md 27)")
    ap.add_argument('--fixed-lai', default='', choices=['', 'state', 'attrs', 'attrs-modis'], help="hold leaf carbon: 'state' at the prescribed starting value, 'attrs' at the site-attribute maximum LAI (fixed-vegetation experiment)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sites = a.sites.split(','); over = {kv.split('=')[0]: [int(x) for x in kv.split('=')[1].split(',')] for kv in a.pft.split(';') if kv}
    yrs = {kv.split('=')[0]: tuple(int(x) for x in kv.split('=')[1].split('-')) for kv in a.years.split(';') if kv}
    m = np.array(xr.open_dataset(MASK).landfrac.values) > 0; fl = np.where(m.ravel())[0]; posmap = {g: p for p, g in enumerate(fl)}
    i = np.array([int(np.argmin(np.abs(LATV - META.loc[s, 'LOCATION_LAT']))) for s in sites], np.int32); j = np.array([int(np.argmin(np.abs(LONV - META.loc[s, 'LOCATION_LONG']))) for s in sites], np.int32)
    for s, ii, jj in zip(sites, i, j): assert ii * 720 + jj in posmap, f'{s}: tower cell not on the land mask'
    sel = np.array([posmap[ii * 720 + jj] for ii, jj in zip(i, j)]); lat_rad = np.deg2rad(LATV[i]).astype(np.float32)
    g = GSWP3Reader(GSWP3); rd = TowerForcingReader(sites, g, use_hh=not a.no_hh, use_hourly_sw=a.hourly_sw)
    years = {s: yrs.get(s, rd.years[s]) for s in sites}
    y0 = max(v[0] for v in years.values()); y1 = min(v[1] for v in years.values())       # common years (cells run together)
    y0 = max(y0, 1901)                                                                    # GSWP3 fallback covers 1901-2010; later years fall back to 2010
    assert y1 - y0 >= 1, f'no common tower years across {sites}: {years}'
    YEARS = list(range(y0, y1 + 1)); print(f'sites {sites}; common tower years {y0}-{y1} ({len(YEARS)}); cycles {a.cycles}', flush=True)
    # --- start state at the tower cells, composition prescribed
    full, _ = load_state(f'{V}/v11_transient/restart/ibis_restart_2010.npz', create_ibis_state(GridConfig(npoi=NPOI_FULL, nbedrock=3)))
    cut = lambda x: x[sel] if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == NPOI_FULL else x
    st = _cast_tree_f32(jax.tree.map(cut, full)); st = eqx.tree_at(lambda s: s.stress, st, create_stress_accum(len(sel)), is_leaf=lambda x: x is None); st = eqx.tree_at(lambda s: s.etparts, st, create_et_parts(len(sel)), is_leaf=lambda x: x is None)
    st = eqx.tree_at(lambda s: s.seasonal, st, _cast_tree_f32(create_seasonal(len(sel), 5.0)), is_leaf=lambda x: x is None)   # inert unless a formulation is on (CALIBRATION.md 21-22)
    sp, pft, _ = build_all_params(len(sel), lat_rad, load_physics_params(a.params)); specla = np.asarray(pft.specla)
    mask = np.zeros((len(sel), 12), np.float32); fu = np.zeros(len(sel), np.float32); fl_ = np.zeros(len(sel), np.float32); notes = {}
    cbiol = np.asarray(st.vegetation.cbiol).copy(); cbiow = np.asarray(st.vegetation.cbiow).copy(); cbior = np.asarray(st.vegetation.cbior).copy()
    for n, s in enumerate(sites):
        igbp = META.loc[s, 'IGBP']; mat = META.loc[s, 'MAT'] if 'MAT' in META else np.nan
        if not (isinstance(mat, (int, float)) and np.isfinite(mat)): mat = tower_mat(s)            # site table often lacks MAT: use the tower record
        pfts, fu[n], fl_[n], note = prescribe(igbp, float(META.loc[s, 'LOCATION_LAT']), float(mat) if pd.notna(mat) else None)
        if s in over: pfts = over[s]; note += f' (override {pfts})'
        notes[s] = dict(igbp=igbp, pfts=[PFT_NAMES[p] for p in pfts], fu=round(float(fu[n]), 2), fl=round(float(fl_[n]), 2), note=note)
        for p in range(12):
            if p in pfts:
                mask[n, p] = 1.0
                if cbiol[n, p] * specla[p] < 0.3:                          # empty or nearly: seed
                    cbiol[n, p] = SEED_LAI[p] / specla[p]; cbiow[n, p] = SEED_WOOD[p]; cbior[n, p] = 0.5
            else: cbiol[n, p] = 0.0; cbiow[n, p] = 0.0; cbior[n, p] = 0.0
        print(f'  {s}: {igbp} at {META.loc[s, "LOCATION_LAT"]:.2f},{META.loc[s, "LOCATION_LONG"]:.2f} -> {notes[s]["pfts"]} fu {notes[s]["fu"]} fl {notes[s]["fl"]}  [{note}]', flush=True)
    plai = cbiol * specla[None, :]
    st = eqx.tree_at(lambda s: (s.vegetation.cbiol, s.vegetation.cbiow, s.vegetation.cbior, s.vegetation.plai, s.vegetation.exist, s.vegetation.fu, s.vegetation.fl),
                     st, tuple(jnp.asarray(x, dtype=jnp.float32) for x in (cbiol, cbiow, cbior, plai, mask, fu, fl_)))
    consts = PhysicalConstants(); ctx = dict(i=i, j=j, lat_rad=lat_rad, sp=sp, pft=pft, consts=consts, reader=rd)
    site_transient.apply_co2 = lambda q, y, series=None: eqx.tree_at(lambda z: z.atm.co2conc, q, 370.0e-6)   # ~2000 level; tower years 1991-2014
    fix = None
    if a.fixed_lai:
        fix = np.asarray(cbiol).copy()
        if a.fixed_lai in ('attrs', 'attrs-modis'):
            from site_attributes import prescribe_from_attributes
            mod_max = np.asarray(modis_at(i, j)) if a.fixed_lai == 'attrs-modis' else None            # MODIS annual maximum LAI at the cell (per tower)
            for n, s in enumerate(sites):
                at = prescribe_from_attributes(s); igbp = META.loc[s, 'IGBP']
                if a.fixed_lai == 'attrs-modis' and igbp in ('OSH', 'CSH', 'SAV', 'WSA', 'GRA') and np.isfinite(mod_max[n]) and mod_max[n] > 0:
                    lai_max, src = float(mod_max[n]), f'MODIS cell maximum ({mod_max[n]:.1f}; attribute maximum held only at forests, CALIBRATION.md 31)'
                elif at is not None and np.isfinite(at['lai_max']) and at['lai_max'] > 0: lai_max, src = float(at['lai_max']), f"site attributes ({at['lai_max']:.1f})"
                else: continue
                share = mask[n] / max(mask[n].sum(), 1)                      # split the site's max LAI over its prescribed PFTs
                fix[n] = share * lai_max / np.maximum(specla, 1e-6)
                notes[s]['lai_max_source'] = src
        log_lai = (fix * specla[None, :]).sum(1)
        print('  fixed leaf area (m2/m2): ' + ', '.join(f'{s} {log_lai[n]:.1f}' for n, s in enumerate(sites)), flush=True)
    hold = lambda s_: enforce_prescription(s_, mask, specla, fix_cbiol=fix)
    key = jax.random.PRNGKey(0); REC = []
    def hook(fn):
        def rec(state, *aa, **kk):
            out = fn(state, *aa, **kk); vg = out[0].vegetation; REC.append((np.asarray(vg.lai[:, 0] * vg.fl), np.asarray(vg.lai[:, 1] * vg.fu))); return out
        return rec
    site_transient.run_month_gswp3 = hook(site_transient.run_month_gswp3)
    for c in range(a.cycles):
        for yr in YEARS:
            st, key, d = one_year(st, key, yr, ctx, 0); st = hold(st)
        print(f'  spin cycle {c + 1}/{a.cycles} done: annual GPP {np.round(d["aygpptot"], 2)} kgC/m2, LAI {np.round(np.asarray(st.vegetation.plai).sum(1), 2)}', flush=True)
    REC.clear(); GM = []; EM = []; LL = []; LU = []; GD = []
    for yr in YEARS:
        st, key, d = one_year(st, key, yr, ctx, 0); st = hold(st); GM.append(d['gpp_monthly'] * 1000); EM.append(d['aet_monthly'])
        gd = np.full((366, len(sel)), np.nan, np.float32); gd[:d['gpp_daily'].shape[0]] = d['gpp_daily'] * 1000; GD.append(gd)       # gC m-2 d-1
        LL.append(np.array([r[0] for r in REC[-12:]])); LU.append(np.array([r[1] for r in REC[-12:]]))
    GM = np.array(GM); EM = np.array(EM); LL = np.array(LL); LU = np.array(LU)             # (nyr, 12, nsite)
    # --- observations, saved with the run BEFORE any metric is computed
    rows = []; obs_g = np.full(GM.shape, np.nan); obs_e = np.full(GM.shape, np.nan); modis = modis_at(i, j)
    for n, s in enumerate(sites):
        tm = tower_monthly(s)
        for yi, yr in enumerate(YEARS):
            for mo in range(12):
                if (yr, mo + 1) in tm.index: obs_g[yi, mo, n] = tm.loc[(yr, mo + 1), 'gpp']; obs_e[yi, mo, n] = tm.loc[(yr, mo + 1), 'et']
    np.savez(f'{a.out}/site_runs.npz', sites=np.array(sites), years=np.array(YEARS), gpp=GM, et=EM, lai_l=LL, lai_u=LU, obs_gpp=obs_g, obs_et=obs_e, i=i, j=j, gpp_daily=np.array(GD))
    json.dump(dict(notes=notes, gaps={f'{k[0]}|{k[1]}': v for k, v in rd.gap_report().items()}, params=a.params, cycles=a.cycles), open(f'{a.out}/site_runs.json', 'w'), indent=1)
    df = compute_metrics(sites, YEARS, GM, EM, LL, LU, obs_g, obs_e, notes, i, modis); df.to_csv(f'{a.out}/site_metrics.csv', index=False)
    pd.set_option('display.width', 250); print(df.drop(columns=['note']).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    # --- figures (optional here; `site_plot.py <out>` draws them from the npz with an interpreter that has matplotlib)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f'matplotlib not available in this interpreter; run: ibis-python verify/calibration/site_plot.py {a.out}', flush=True); return
    for n, s in enumerate(sites):
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
        og = np.nanmean(obs_g[:, :, n], 0); mg = GM[:, :, n].mean(0); oe = np.nanmean(obs_e[:, :, n], 0); me = EM[:, :, n].mean(0)
        axs[0].fill_between(range(12), 0, og, color='0.85'); axs[0].plot(range(12), og, 'k', lw=1.5, label='tower GPP_NT'); axs[0].plot(range(12), mg, 'tab:green', lw=2, marker='o', ms=3, label='model'); axs[0].set_ylabel('GPP (gC m$^{-2}$ mo$^{-1}$)')
        axs[1].fill_between(range(12), 0, np.nan_to_num(oe), color='0.85'); axs[1].plot(range(12), oe, 'k', lw=1.5, label='tower LE as ET'); axs[1].plot(range(12), me, 'tab:blue', lw=2, marker='o', ms=3, label='model'); axs[1].set_ylabel('ET (mm mo$^{-1}$)')
        for ax in axs[:2]: ax.set_xticks(range(12)); ax.set_xticklabels(list('JFMAMJJASOND')); ax.legend(frameon=False, fontsize=8)
        t = np.arange(GM.shape[0] * 12); axs[2].plot(t, obs_g[:, :, n].ravel(), 'k', lw=1, label='tower'); axs[2].plot(t, GM[:, :, n].ravel(), 'tab:green', lw=1, label='model'); axs[2].set_xticks(np.arange(0, len(t), 12)); axs[2].set_xticklabels(YEARS, rotation=90, fontsize=7); axs[2].set_ylabel('monthly GPP'); axs[2].legend(frameon=False, fontsize=8)
        fig.suptitle(f'{s} ({notes[s]["igbp"]} -> {"+".join(notes[s]["pfts"])}, fu {notes[s]["fu"]}, fl {notes[s]["fl"]}), tower years {y0}-{y1}, {a.cycles} spin cycles; params {os.path.basename(a.params)}', fontsize=9, x=0.01, ha='left')
        fig.tight_layout(); fig.savefig(f'{a.out}/{s}.png', dpi=130); plt.close(fig)
    print(f'wrote {a.out}/site_metrics.csv, site_runs.npz, site_runs.json and {len(sites)} figures', flush=True)


if __name__ == '__main__':
    main()
