"""Full-year diagnostic record at the towers of a class (CALIBRATION.md 22-26): what the canopy does, hour by hour.

    python verify/calibration/diagnostics/site_diag_year.py <IGBP> <params.yaml> <year> <out.npz> [<site,site,...>]

Prescribed composition and fixed leaf area exactly as in site_fit (setup; two spin-up years 2008-2009, then <year> on
tower meteorology). Throughout <year>, `stomata` in the production step is wrapped: it runs unchanged and a
jax.debug.callback records, per tower and hour (the last of the three canopy iterations), the inputs and pieces of
every big leaf. Upper canopy (broadleaf ub, conifer uc) and lower canopy (shrub ls, C3 grass l3, C4 grass l4):
nominal and effective vmax, ci, the limiting rates (light je, Rubisco jc, sink js / C4 CO2 ji), the top-leaf gross rate,
the canopy rate per unit leaf, and per canopy the top PAR, leaf temperature, soil stress, wet fraction, LAI, SAI,
extinction k, the light-profile factor f_light and the capacity factor f_cap (= f_light on the F77 canopy). After every
month the state gives the ET components, NPP, Rh, root respiration, AET, soil water and temperature, plant LAI and
phenology. The raw arrays are written before any analysis; `site_diag_plot.py` draws them against the towers.
"""
import sys, numpy as np, jax, jax.numpy as jnp, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = ['python-port', 'verify/calibration']
import site_fit, site_transient
import ibis.timestepping.loops as LP
import ibis.physics.stomata as SM
from ibis.physics.thermo import esat, qsat
from ibis.timestepping.loops_gswp3 import run_month_gswp3

igbp, yml, year, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; only = sys.argv[5] if len(sys.argv) > 5 else ''
U = ['par_u', 'tu', 't12', 'rh12', 'stress_u', 'fwet_u', 'lai_u', 'sai_u', 'k_u', 'fl_u', 'fc_u', 'tempvm_u']
UL = ['vnom', 'vmax', 'ci', 'je', 'jc', 'js', 'agtop', 'agc', 'gs']
L = ['par_l', 'tl', 'stress_l', 'fwet_l', 'lai_l', 'sai_l', 'k_l', 'fl_l', 'fc_l', 'tempvm3_l', 'tempvm4_l']
LL = ['vmax', 'ci', 'je', 'jc', 'js', 'agtop', 'agc']
FIELDS = U + [f'{v}_{c}' for c in ('ub', 'uc') for v in UL] + L + [f'{v}_{c}' for c in ('ls', 'l3', 'l4') for v in LL]
REC = []; CNT = [0]
_orig = LP.stomata


def _host(*arrs):
    CNT[0] += 1
    if CNT[0] % 3 == 0:                                   # stomata runs in each of the 3 canopy iterations; the last sets the fluxes
        REC.append(np.stack([np.asarray(x, dtype=np.float32) for x in arrs]))


def _ext(t6, t7, sc, eps):
    return jnp.clip((t6 * sc[:, 0] + t7 * sc[:, 1] - t7 * sc[:, 2]) / jnp.maximum(sc[:, 3], eps), 0.1, 10.0)


def wrapped(tu, tl, t12, t34, q12, q34, su, sl, psurf, topparu, topparl, termu6, termu7, terml6, terml7, scalcoefu, scalcoefl,
            ci_upper, gs_upper, cs_upper, ci_lower, gs_lower, cs_lower, fwetu, fwetl, lai, sai, exist, stresstu, stresstl,
            a10scalparamu, a10daylightu, a10scalparaml, a10daylightl, co2conc, dtime, params, epsilon=1.0e-7, frac=None, vfac=None):
    out = _orig(tu, tl, t12, t34, q12, q34, su, sl, psurf, topparu, topparl, termu6, termu7, terml6, terml7, scalcoefu, scalcoefl,
                ci_upper, gs_upper, cs_upper, ci_lower, gs_lower, cs_lower, fwetu, fwetl, lai, sai, exist, stresstu, stresstl,
                a10scalparamu, a10daylightu, a10scalparaml, a10daylightl, co2conc, dtime, params, epsilon, frac=frac, vfac=vfac)
    p = params; vp = None if vfac is None else p.vmax_pft[jnp.newaxis, :] * vfac
    # upper canopy
    kc, ko, _, gs_, tv3, _ = SM._temp_dep_params(tu, p)
    rh12 = jnp.maximum(0.30, q12 / jnp.maximum(qsat(esat(t12), psurf), epsilon))
    Lu, Su = lai[:, 1], sai[:, 1]; ku = _ext(termu6, termu7, scalcoefu, epsilon)
    flu = (1.0 - jnp.exp(-ku * (Lu + Su))) / jnp.maximum(ku * ((Lu + Su) if p.scale_leaf_only else Lu), epsilon)
    fcu = SM._capacity_integral(Lu, p.cap_kn, epsilon) if p.cap_scale_u else flu
    rec = [topparu, tu - 273.16, t12 - 273.16, rh12, stresstu, fwetu, Lu, Su, ku, flu, fcu, tv3]
    vnom = SM._select_vmax_upper(exist, p, frac, vp=vp)
    for c in (0, 1):
        vmax = vnom[c] * tv3 * stresstu; ci = jnp.clip(ci_upper[:, c], 1.05 * gs_, p.cimax)
        je = topparu * 4.59e-06 * p.alpha3 * (ci - gs_) / (ci + 2.0 * gs_); jc = vmax * (ci - gs_) / (ci + kc * (1.0 + p.o2conc / ko)); js = vmax / 2.2
        ag, _ = SM.farquhar_c3(topparu, ci, vmax, gs_, kc, ko, p.o2conc, p.gamma[c] * vnom[c] * tv3, p)
        rec += [vnom[c], vmax, ci, je, jc, js, ag, out.ag_upper[:, c], out.gs_upper[:, c]]
    # lower canopy
    kcl, kol, _, gsl, tv3l, tv4l = SM._temp_dep_params(tl, p)
    Ll, Sl = lai[:, 0], sai[:, 0]; kl = _ext(terml6, terml7, scalcoefl, epsilon)
    fll = (1.0 - jnp.exp(-kl * (Ll + Sl))) / jnp.maximum(kl * ((Ll + Sl) if p.scale_leaf_only else Ll), epsilon)
    fcl = SM._capacity_integral(Ll, p.cap_kn, epsilon) if p.cap_scale_l else fll
    rec += [topparl, tl - 273.16, stresstl, fwetl, Ll, Sl, kl, fll, fcl, tv3l, tv4l]
    v_ls = SM._select_vmax_shrub(p, frac, vp=vp); v_l3 = p.vmax_pft[11] if vp is None else vp[:, 11]; v_l4 = p.vmax_pft[10] if vp is None else vp[:, 10]
    for c, vn, g_idx in ((0, v_ls, 2), (1, v_l3, 4)):
        vmax = vn * tv3l * stresstl; ci = jnp.clip(ci_lower[:, c], 1.05 * gsl, p.cimax)
        je = topparl * 4.59e-06 * p.alpha3 * (ci - gsl) / (ci + 2.0 * gsl); jc = vmax * (ci - gsl) / (ci + kcl * (1.0 + p.o2conc / kol)); js = vmax / 2.2
        ag, _ = SM.farquhar_c3(topparl, ci, vmax, gsl, kcl, kol, p.o2conc, p.gamma[g_idx] * vn * tv3l, p)
        rec += [jnp.broadcast_to(vmax, Ll.shape), ci, je, jc, js, ag, out.ag_lower[:, c]]
    vmax4 = v_l4 * tv4l * stresstl; ci4 = jnp.clip(ci_lower[:, 2], 0.0, p.cimax)
    je4 = topparl * 4.59e-06 * p.alpha4; jc4 = vmax4; ji4 = 18.0e3 * vmax4 * ci4
    ag4, _ = SM.farquhar_c4(topparl, ci4, vmax4, p.gamma[3] * v_l4 * tv4l, p)
    rec += [jnp.broadcast_to(vmax4, Ll.shape), ci4, je4, jc4, ji4, ag4, out.ag_lower[:, 2]]
    rec = [jnp.broadcast_to(jnp.asarray(r, dtype=jnp.float32), Lu.shape) for r in rec]
    jax.debug.callback(_host, *rec, ordered=True)
    return out


a = site_fit._args(['--igbp', igbp, '--params', yml, '--fit', 'vmax:temp_con_ever', '--out', '/tmp/unused', '--years', '2008-2009', '--hold-last', '0']
                   + (['--sites', only] if only else []) + sys.argv[6:])   # two spin-up years; extra site_fit options (e.g. --lai US-Ha1=4.7 --co2 series)
C = site_fit.setup(a, spin=True, verbose=True)
st, key, c = C.base_state, C.base_key, C.ctx0
st = site_transient.reset_annual(st, len(C.sites))                  # annual sums from 1 January: month differences are clean
st = site_transient.apply_co2(st, year)
LP.stomata = wrapped; jax.clear_caches()
MON = {k: [] for k in ('trans', 'interc', 'soilev', 'snowsub', 'npp', 'rh', 'rroot', 'gpp', 'aet', 'wsoi4', 'tsoi1', 'lai_u_state', 'lai_l_state', 'tempu', 'templ')}
DAY = {k: [] for k in ('gpp', 'aet', 'rh')}
cum = lambda s_: dict(trans=s_.etparts.trans, interc=s_.etparts.interc, soilev=s_.etparts.soilev, snowsub=s_.etparts.snowsub,
                      npp=s_.summary.aynpptot, rh=s_.summary.ayco2mic, rroot=s_.summary.ayco2root, gpp=s_.summary.aygpptot, aet=s_.summary.ayaet)
prev = {k: np.asarray(v, dtype=np.float64) for k, v in cum(st).items()}
for im in range(12):
    st, key, dd = run_month_gswp3(st, c['consts'], c['sp'], c['reader'], c['i'], c['j'], c['lat_rad'], year, im, key, 3600.0, leap=(year % 4 == 0), nband=2, nsoilcall=1)
    now = {k: np.asarray(v, dtype=np.float64) for k, v in cum(st).items()}
    for k in now: MON[k].append(now[k] - prev[k])
    prev = now
    h = np.asarray(st.soil.hsoi)[:4]; w = np.asarray(st.soil.wsoi)[:, :4]
    MON['wsoi4'].append((w * h).sum(1) / h.sum()); MON['tsoi1'].append(np.asarray(st.soil.tsoi)[:, 0] - 273.16)
    MON['lai_u_state'].append(np.asarray(st.vegetation.lai)[:, 1]); MON['lai_l_state'].append(np.asarray(st.vegetation.lai)[:, 0])
    MON['tempu'].append(np.asarray(st.vegetation.tempu)); MON['templ'].append(np.asarray(st.vegetation.templ))
    for k, q in zip(('gpp', 'aet', 'rh'), dd): DAY[k].append(np.asarray(q))
    print(f'  {year}-{im + 1:02d} recorded ({len(REC)} hours so far)', flush=True)
R = np.stack(REC)                                                     # (hours, fields, towers)
v = st.vegetation
np.savez(out, R=R, fields=np.array(FIELDS), sites=np.array(C.sites), igbp=igbp, params=yml, year=year,
         train=np.array(C.train), hold=np.array(C.hold), fu=np.asarray(v.fu), fl=np.asarray(v.fl), exist=np.asarray(v.exist),
         vmax_pft=np.asarray(c['sp'].stomata_params.vmax_pft), i=np.asarray(c['i']), j=np.asarray(c['j']), lai_src=np.array([C.lai_src[s] for s in C.sites]),
         **{f'mon_{k}': np.array(x) for k, x in MON.items()}, **{f'day_{k}': np.concatenate(x, 0) for k, x in DAY.items()})
print(f'wrote {out}: {R.shape[0]} hours x {R.shape[1]} fields x {R.shape[2]} towers', flush=True)
