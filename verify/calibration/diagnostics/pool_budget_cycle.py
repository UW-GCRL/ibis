"""One cycle of the tower years from a spin-up state, printing the yearly carbon budget per tower (NEE stage diagnosis).

    python verify/calibration/diagnostics/pool_budget_cycle.py <spin-up dir> <SITE[,SITE]> [site_fit options]

Per year: NPP (aynpptot), litter inputs at the start of the year (falll + fallr + fallw), Rh (ayco2mic), leaching (yrleach),
change of the litter, microbial and SOM pools, and the residual NPP - Rh - dPools - leach (0 if carbon is conserved).
"""
import sys, numpy as np, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = ['python-port', 'verify/calibration']
import site_fit
from site_transient import one_year
from ibis.io.restart import load_state
spin, which = sys.argv[1], sys.argv[2].split(','); rest = sys.argv[3:]
a = site_fit._args(['--igbp', 'ALL', '--fit', 'vmax:temp_con_ever', '--out', '/tmp/unused', '--hold-last', '0', '--sites', ','.join(which)] + rest)
C = site_fit.setup(a, spin=True, verbose=False); sites = C.sites; ns = len(sites)
z = np.load(f'{spin}/state.npz', allow_pickle=True); ssites = [str(s) for s in z['meta_sites']]; nss = len(ssites)
tmpl = C.base_state if nss == ns else jax.tree.map(lambda x: (jnp.repeat(x[:1], nss, axis=0) if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == ns else x), C.base_state)
eq, _ = load_state(f'{spin}/state.npz', tmpl); sel = np.array([ssites.index(s) for s in sites]); pick = lambda x: (x[sel] if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == nss else x)
st = eqx.tree_at(lambda s: (s.carbon, s.vegetation), C.base_state, (jax.tree.map(pick, eq.carbon), jax.tree.map(pick, eq.vegetation))); key = C.base_key; hold = C.hold_fn
POOLS = ('clitlm', 'clitls', 'clitll', 'clitrm', 'clitrs', 'clitrl', 'clitwm', 'clitws', 'clitwl', 'totcmic', 'csoislon', 'csoislop', 'csoipas')
tot = lambda s: sum(np.asarray(getattr(s.carbon, p)) for p in POOLS) * 1000
for yr in site_fit.YEARS:
    before = tot(st); fin = (np.asarray(st.carbon.falll) + np.asarray(st.carbon.fallr) + np.asarray(st.carbon.fallw)) * 1000; fl0 = np.asarray(st.carbon.falll) * 1000
    st, key, d = one_year(st, key, yr, C.ctx0, 0)
    npp = np.asarray(st.summary.aynpptot) * 1000; rh = np.asarray(st.summary.ayco2mic) * 1000; leach = np.asarray(st.summary.yrleach) * 1000; mid = tot(st); fl_after_bu = np.asarray(st.carbon.falll) * 1000
    st = hold(st); after = tot(st); fl_after_hold = np.asarray(st.carbon.falll) * 1000
    for n, s in enumerate(sites):
        print(f'{yr} {s}: NPP {npp[n]:6.0f}  inputs(start) {fin[n]:6.0f} [leaf {fl0[n]:.0f} -> after biomass_update {fl_after_bu[n]:.0f} -> after hold {fl_after_hold[n]:.0f}]  Rh {rh[n]:6.0f}  leach {leach[n]:5.1f}  dPools {mid[n] - before[n]:+7.1f}  residual(inputs - Rh - dPools - leach) {fin[n] - rh[n] - (mid[n] - before[n]) - leach[n]:+7.1f}  NEE {rh[n] - npp[n]:+6.0f}', flush=True)
