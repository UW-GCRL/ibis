"""Hourly probe of upper-canopy photosynthesis at the towers of a class (CALIBRATION.md 22): what sets the summer plateau.

    python verify/calibration/diagnostics/canopy_plateau_probe.py <IGBP> <params.yaml> [<year> <month 1-12>] [--fixed-lai]

Prescribed composition and fixed leaf area exactly as in site_fit (setup, two spin-up years 2008-2009, then the months
of <year> up to <month> on tower meteorology). During <month>, `stomata` in the production step is wrapped: it runs
unchanged and a jax.debug.callback records, per tower and hour, the inputs and the pieces of the conifer (or broadleaf,
for deciduous classes) upper big leaf -- top-of-canopy PAR, ci, the temperature- and stress-adjusted vmax, the
light-limited rate je, the Rubisco-limited rate jc, the sink limit js, the top-leaf gross rate, the canopy scaling
factor (1 - exp(-k (L+S))) / (k L), the extinction k, the wet fraction, the soil stress, leaf temperature, LAI and SAI.
Prints, per tower, the daytime means and the share of daylight hours in which each limit binds, and the plateau
arithmetic: canopy GPP = top-leaf rate x scale x (1 - fwet) x LAI x fu.
"""
import sys, inspect, numpy as np, pandas as pd, jax, jax.numpy as jnp, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = ['python-port', 'verify/calibration']
import site_fit
import ibis.timestepping.loops as LP
import ibis.physics.stomata as SM
from ibis.timestepping.loops_gswp3 import run_month_gswp3

igbp, yml = sys.argv[1], sys.argv[2]; year = int(sys.argv[3]) if len(sys.argv) > 3 else 2010; month = int(sys.argv[4]) if len(sys.argv) > 4 else 7
col = 1 if igbp in ('ENF',) else 0                       # upper big leaf: 0 broadleaf, 1 conifer
REC = []
_orig = LP.stomata


def _host(*arrs):
    REC.append(np.stack([np.asarray(x, dtype=np.float64) for x in arrs]))


_SIG = inspect.signature(_orig)


def wrapped(*a_, **kw_):
    out = _orig(*a_, **kw_)
    g = _SIG.bind(*a_, **kw_); g.apply_defaults(); g = g.arguments
    tu, topparu, termu6, termu7, scalcoefu, ci_upper, lai, sai, exist, stresstu, fwetu, co2conc, params, epsilon, frac, vfac = (
        g[n] for n in ('tu', 'topparu', 'termu6', 'termu7', 'scalcoefu', 'ci_upper', 'lai', 'sai', 'exist', 'stresstu', 'fwetu', 'co2conc', 'params', 'epsilon', 'frac', 'vfac'))
    kc, ko, _, gamstar, tempvm, _ = SM._temp_dep_params(tu, params)
    vp_eff = None if vfac is None else params.vmax_pft[jnp.newaxis, :] * vfac
    vnom = SM._select_vmax_upper(exist, params, frac, vp=vp_eff)[col]
    vmax = vnom * tempvm * stresstu
    ci = jnp.clip(ci_upper[:, col], 1.05 * gamstar, params.cimax)
    je = topparu * 4.59e-06 * params.alpha3 * (ci - gamstar) / (ci + 2.0 * gamstar)
    jc = vmax * (ci - gamstar) / (ci + kc * (1.0 + params.o2conc / ko))
    js = vmax / 2.2
    ag_top, _ = SM.farquhar_c3(topparu, ci, vmax, gamstar, kc, ko, params.o2conc, params.gamma[col] * vnom * tempvm, params)
    k = jnp.clip((termu6 * scalcoefu[:, 0] + termu7 * scalcoefu[:, 1] - termu7 * scalcoefu[:, 2]) / jnp.maximum(scalcoefu[:, 3], epsilon), 0.1, 10.0)
    L, S = lai[:, 1], sai[:, 1]
    scale = (1.0 - jnp.exp(-k * (L + S))) / jnp.maximum(k * ((L + S) if params.scale_leaf_only else L), epsilon)
    jax.debug.callback(_host, topparu, ci, vmax, je, jc, js, ag_top, out.ag_upper[:, col], scale, k, fwetu, stresstu, tu - 273.16, L, S, co2conc * jnp.ones_like(tu), ordered=True)
    return out


a = site_fit._args(['--igbp', igbp, '--params', yml, '--fit', f"vmax:{'temp_con_ever' if col else 'temp_bl_dec'}", '--out', '/tmp/unused', '--years', '2008-2009', '--hold-last', '0'])
C = site_fit.setup(a, spin=True, verbose=True)
st, key, c = C.base_state, C.base_key, C.ctx0
for im in range(month - 1):
    st, key, _ = run_month_gswp3(st, c['consts'], c['sp'], c['reader'], c['i'], c['j'], c['lat_rad'], year, im, key, 3600.0, leap=(year % 4 == 0), nband=2, nsoilcall=1)
LP.stomata = wrapped; jax.clear_caches()
st, key, dd = run_month_gswp3(st, c['consts'], c['sp'], c['reader'], c['i'], c['j'], c['lat_rad'], year, month - 1, key, 3600.0, leap=(year % 4 == 0), nband=2, nsoilcall=1)
R = np.stack(REC)                                   # (3 x steps, fields, towers): stomata runs in each of the 3 canopy iterations
np.savez(f'/shared/nas2/Hangkai/ibis/verify_outputs/site_fit/plateau_probe_{igbp}_{year}{month:02d}.npz', R=R, sites=np.array(C.sites), fu=np.asarray(st.vegetation.fu), fields=np.array(['par', 'ci', 'vmax', 'je', 'jc', 'js', 'ag_top', 'ag_canopy_leaf', 'scale', 'k', 'fwet', 'stress', 'tleaf', 'lai', 'sai', 'co2']))   # raw record first
R = R.reshape(R.shape[0] // 3, 3, *R.shape[1:])[:, 2]   # the last iteration sets the step's fluxes
names = ['par', 'ci', 'vmax', 'je', 'jc', 'js', 'ag_top', 'ag_canopy_leaf', 'scale', 'k', 'fwet', 'stress', 'tleaf', 'lai', 'sai', 'co2']
print(f"\n{igbp} {year}-{month:02d}: {R.shape[0]} hourly steps (last canopy iteration) x {R.shape[2]} towers; upper big leaf column {col}")
fu = np.asarray(st.vegetation.fu); rows = []
for n, s in enumerate(C.sites):
    x = {nm: R[:, q, n] for q, nm in enumerate(names)}; day = x['par'] > 10.0
    jp = np.minimum(x['je'], x['jc'])
    lim = np.where(np.minimum(jp, x['js']) == x['js'], 'sink', np.where(x['jc'] < x['je'], 'rubisco', 'light'))
    g_hour = x['ag_canopy_leaf'] * x['lai'] * fu[n] * 12.0 * 3600.0                           # gC m-2 per hour (from mol m-2 s-1)
    rows.append(dict(site=s, day_hours=int(day.sum()), par=x['par'][day].mean(), vmax_umol=x['vmax'][day].mean() * 1e6, ci_ppm=x['ci'][day].mean() * 1e6,
                     tleaf=x['tleaf'][day].mean(), stress=x['stress'][day].mean(), fwet=x['fwet'][day].mean(),
                     light_lim=(lim[day] == 'light').mean(), rubisco_lim=(lim[day] == 'rubisco').mean(), sink_lim=(lim[day] == 'sink').mean(),
                     ag_top_umol=x['ag_top'][day].mean() * 1e6, k=x['k'][day].mean(), lai=x['lai'][day].mean(), sai=x['sai'][day].mean(),
                     scale=x['scale'][day].mean(), eff_leaf_area=(x['scale'] * x['lai'])[day].mean(), fu=fu[n],
                     gpp_model_gC_d=g_hour.sum() / (R.shape[0] / 24.0), gpp_tower_gC_d=np.nan))
T = pd.DataFrame(rows).set_index('site')
site_fit.YEARS = [2008, 2009, 2010, 2011, 2012]; obs = {s: site_fit.tower_obs(s) for s in C.sites}
T['gpp_tower_gC_d'] = [np.nan if not np.isfinite(obs[s]['gpp'][[2008, 2009, 2010, 2011, 2012].index(year), month - 1]) else
                      obs[s]['gpp'][[2008, 2009, 2010, 2011, 2012].index(year), month - 1] / (R.shape[0] / 24.0) for s in C.sites]
pd.set_option('display.width', 260); print(T.round(3).to_string())
print('\nmedians: ' + '  '.join(f"{c_} {T[c_].median():.3f}" for c_ in T.columns))
