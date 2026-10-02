"""Per-tower soil-carbon pool state in equilibrium with the tower's own forcing and fitted GPP (NEE stage, CALIBRATION.md 36).

    python verify/calibration/site_pool_spinup.py <out dir> --params <yaml> [--start-dir <s2 root>] [--sites A,B] [site_fit options]

The grid restart gives every tower the 0.5-degree cell's pools (1.3x the HWSD2 site carbon at the median, > 2x at 17
temperate-forest towers; Rh/NPP 0.5-2.3 by class). This script replaces them by the cyclic steady state under the
tower's forcing and the tower's fitted parameters (`--start-dir`: s2/<SITE>/best.json per tower, else the yaml alone):
  1. fixed vegetation as in site_fit (leaf area held, composition prescribed);
  2. accelerated spin-up: the five slow rate constants (knb, kns, kpb, kps, ksb) are multiplied by A while the tower
     years are cycled; for this linear cascade with external litter input the steady state of each slow pool is then
     1/A of the true one with every flux unchanged, so after the cycles the three SOM pools are multiplied back by A
     (Thornton & Rosenbloom 2005 without the correction terms, exact here because the slow pools feed nothing but each
     other and the microbial pool through the same scaled fluxes);
  3. two normal cycles for the fast pools to settle; the drift of total soil carbon and the annual NEE over the last
     cycle are reported (both ~0 at equilibrium).
Writes <out dir>/pools.npz (sites, the nine litter pools, totcmic, csoislon, csoislop, csoipas before and after, drift,
annual NEE and NPP of the last cycle) and <out dir>/state.npz (the equilibrated IBISState of the tower set, loadable by
load_state with the same template) for the NEE fits to start from.
"""
import sys, os, json, argparse, numpy as np, jax, jax.numpy as jnp, equinox as eqx, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = ['python-port', 'verify/calibration']
import site_fit
from site_transient import one_year
from fit_carbon import apply_params, PFT_NAMES
from ibis.io.restart import save_state
out = sys.argv[1]; os.makedirs(out, exist_ok=True); rest = sys.argv[2:]
start_dir = rest[rest.index('--start-dir') + 1] if '--start-dir' in rest else ''
if start_dir: k = rest.index('--start-dir'); rest = rest[:k] + rest[k + 2:]
accel = float(rest[rest.index('--accel') + 1]) if '--accel' in rest else 20.0        # factor on the non-protected slow pool's rates; protected x10 that, passive x100 that
if '--accel' in rest: k = rest.index('--accel'); rest = rest[:k] + rest[k + 2:]
cycles_fast = int(rest[rest.index('--cycles') + 1]) if '--cycles' in rest else 6
kscale = float(rest[rest.index('--k-scale') + 1]) if '--k-scale' in rest else 1.0
if '--k-scale' in rest: k = rest.index('--k-scale'); rest = rest[:k] + rest[k + 2:]
if '--k-scale-table' in rest: k = rest.index('--k-scale-table'); _ktab = rest[k + 1]; rest = rest[:k] + rest[k + 2:]; rest_ktab = _ktab
else: rest_ktab = ''
if '--cycles' in rest: k = rest.index('--cycles'); rest = rest[:k] + rest[k + 2:]
a = site_fit._args(['--igbp', 'ALL', '--fit', 'vmax:temp_con_ever', '--out', out, '--hold-last', '0'] + rest)
C = site_fit.setup(a, spin=True, verbose=True); sites = C.sites; ns = len(sites)          # spin=True: one warm-up cycle, and the state/ctx/hold_fn the replay needs
# the per-tower fitted parameters, applied tower by tower (vmax and extras are per PFT / scalar: one set for the batch
# cannot hold 62 different values, so the towers are run in groups sharing a parameter set when --start-dir differs...)
# -> simplest correct route: one tower set per call. We run all towers with the yaml's set, then, where --start-dir has
# a best.json, override by re-running that tower alone (handled by the driver script, not here).
st, key, ctx = C.base_state, C.base_key, C.ctx0
if start_dir and ns == 1 and os.path.exists(f'{start_dir}/{sites[0]}/best.json'):
    prev = json.load(open(f'{start_dir}/{sites[0]}/best.json'))['params']; vm = np.asarray(ctx['sp'].stomata_params.vmax_pft).copy(); ex = {}
    for nm, v in prev.items():
        if nm.startswith('vmax:'): vm[PFT_NAMES.index(nm[5:])] = v
        else: ex[nm] = v
    sp, pft = apply_params(ctx['sp'], ctx['pft'], vm, np.asarray(ctx['pft'].specla), ex); ctx = dict(ctx, sp=sp, pft=pft); print(f'{sites[0]}: per-tower parameters from {start_dir}')
ktab = rest_ktab
if ktab:                                                                              # csv site,k_scale: one multiplier per tower (an (npoi,) array in BGCParams)
    import pandas as pd; kt = pd.read_csv(ktab).set_index('site')['k_scale']; karr = jnp.asarray([float(kt.get(s_, 1.0)) for s_ in sites], dtype=jnp.float32)
    ctx = dict(ctx, sp=eqx.tree_at(lambda p: p.bgc_params.k_scale, ctx['sp'], karr)); print('k_scale per tower:', ' '.join(f'{s_} {float(v):.2f}' for s_, v in zip(sites, karr)))
elif kscale != 1.0: ctx = dict(ctx, sp=eqx.tree_at(lambda p: p.bgc_params.k_scale, ctx['sp'], kscale)); print(f'k_scale {kscale}')
POOLS = ('clitlm', 'clitls', 'clitll', 'clitrm', 'clitrs', 'clitrl', 'clitwm', 'clitws', 'clitwl', 'totcmic', 'csoislon', 'csoislop', 'csoipas')
SLOW = ('knb', 'kns', 'kpb', 'kps', 'ksb')
# one factor per pool, applied to every outflow rate of that pool (so each pool's steady state is 1/factor of the
# true one with all fluxes unchanged); the daily clamp min(k f X, X) stays inactive: 0.001 x 20 x f(<=10) = 0.2 < 1
FAC = {'csoislon': accel, 'csoislop': accel * 10.0, 'csoipas': accel * 100.0}; KFAC = {'knb': FAC['csoislon'], 'kns': FAC['csoislon'], 'kpb': FAC['csoislop'], 'kps': FAC['csoislop'], 'ksb': FAC['csoipas']}
def pools(s): return {p: np.asarray(getattr(s.carbon, p)).copy() for p in POOLS}
def tot(s): return sum(pools(s)[p] for p in POOLS)
p0 = pools(st); print('start pools (kgC m-2):', ' '.join(f'{s} {tot(st)[n]:.1f}' for n, s in enumerate(sites)))
# accelerated cycles
sp_fast = ctx['sp']
for kname in SLOW: sp_fast = eqx.tree_at(lambda p, kname=kname: getattr(p.bgc_params, kname), sp_fast, float(getattr(ctx['sp'].bgc_params, kname)) * KFAC[kname])
ctx_fast = dict(ctx, sp=sp_fast); hold = C.hold_fn
# scale the SOM pools DOWN by A first so the accelerated system starts near its own steady state
st = eqx.tree_at(lambda s: (s.carbon.csoislon, s.carbon.csoislop, s.carbon.csoipas), st, tuple(getattr(st.carbon, p) / FAC[p] for p in ('csoislon', 'csoislop', 'csoipas')))
true_tot = lambda s: tot(s) + sum((FAC[p] - 1) * pools(s)[p] for p in ('csoislon', 'csoislop', 'csoipas'))
for cyc in range(cycles_fast):
    t_before = true_tot(st)
    for yr in site_fit.YEARS: st, key, d = one_year(st, key, yr, ctx_fast, 0); st = hold(st)
    t_after = true_tot(st)
    print(f'accelerated cycle {cyc + 1}/{cycles_fast} (x{accel:.0f}/{accel * 10:.0f}/{accel * 100:.0f}): true-scale total C ' + ' '.join(f'{s} {t_after[n]:.2f} (d {t_after[n] - t_before[n]:+.3f})' for n, s in enumerate(sites)), flush=True)
# scale back and settle
st = eqx.tree_at(lambda s: (s.carbon.csoislon, s.carbon.csoislop, s.carbon.csoipas), st, tuple(getattr(st.carbon, p) * FAC[p] for p in ('csoislon', 'csoislop', 'csoipas')))
for cyc in range(2):
    t_before = tot(st); nee = np.zeros(ns); npp = np.zeros(ns)
    for yr in site_fit.YEARS: st, key, d = one_year(st, key, yr, ctx, 0); st = hold(st); nee += np.asarray(d['nee_monthly']).sum(0) * 1000; npp += np.asarray(d['npp_monthly']).sum(0) * 1000
    ny = len(site_fit.YEARS); print(f'normal cycle {cyc + 1}/2: total C drift ' + ' '.join(f'{s} {tot(st)[n] - t_before[n]:+.3f} kgC; NEE {nee[n] / ny:+.0f} NPP {npp[n] / ny:.0f} gC m-2 yr-1' for n, s in enumerate(sites)), flush=True)
p1 = pools(st)
np.savez(f'{out}/pools.npz', sites=np.array(sites), years=np.array(site_fit.YEARS), accel=accel, **{f'{p}_before': p0[p] for p in POOLS}, **{f'{p}_after': p1[p] for p in POOLS}, drift_kgC=tot(st) - t_before, nee_gC=nee / ny, npp_gC=npp / ny)
save_state(st, f'{out}/state.npz', metadata=dict(sites=list(sites), years=list(site_fit.YEARS), accel=accel, params=a.params, start_dir=start_dir))
print('equilibrium pools (kgC m-2):', ' '.join(f'{s} {tot(st)[n]:.1f} (was {tot(C.base_state)[n]:.1f})' for n, s in enumerate(sites))); print('wrote', out)
