"""Single-column reproduction of the phantom drainage (CALIBRATION.md 62): the as-is soilh2o vs an in-memory copy with the
linearization point's interpolation weights put the matrix's way round (soil_physics.py wsoim line). Columns from the s31f 2010
restart; no rain, no evaporation, no transpiration; hourly steps; wadjust after each step as the daily loop does. Read-only."""
import sys, types, numpy as np, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R, f'{R}/verify/fire']
import jax, jax.numpy as jnp, evaluate_gfed5 as eg
import ibis.physics.soil_physics as SP
p = '/shared/nas2/Hangkai/ibis/verify_outputs/final_s31f/hindcast/restart/ibis_restart_2010.npz'; lat, lon, *_ = eg._grid(p); st = eg._load(p); s = st.soil
cells = [('Tarim', 41.75, 80.25), ('Turkmenistan', 37.75, 67.25), ('Texas', 33.75, -98.75), ('Atacama', -23.75, -69.75), ('Sahara pump', 23.75, 3.75), ('Sahara normal', 23.25, 10.25), ('Amazon', -3.25, -60.25)]
idx = [int(np.argmin(np.abs(lat - la) + np.abs(lon - lo))) for _, la, lo in cells]
take = lambda a: jnp.asarray(np.asarray(a)[idx], jnp.float32)
wsoi0, wisoi, poros, hyd, suc, bex = take(s.wsoi), take(s.wisoi), take(s.poros), take(s.hydraul), take(s.suction), take(s.bex); hsoi = jnp.asarray(np.asarray(s.hsoi), jnp.float32)
n = len(idx); eps, rhow, dt, bperm = 1e-7, 1000.0, 3600.0, 0.04
src = open(f'{R}/python-port/ibis/physics/soil_physics.py').read()
old = 'wsoim = weim * wsoi[:, k - 1] + weip * wsoi[:, k]'; assert src.count(old) == 1
mod = types.ModuleType('soil_physics_patched'); mod.__dict__['__file__'] = SP.__file__; exec(compile(src.replace(old, 'wsoim = weip * wsoi[:, k - 1] + weim * wsoi[:, k]'), 'soil_physics_patched', 'exec'), mod.__dict__)
def run(fn, nsteps=480):
    w = wsoi0; drain = np.zeros(n); created = np.zeros(n)
    store0 = np.asarray(jnp.sum(hsoi[None, :6] * poros * rhow * ((1 - wisoi) * w + wisoi), 1))
    porosflo = jnp.maximum(poros * jnp.maximum(1 - wisoi, eps), eps)
    for _ in range(nsteps):
        h = fn(w, wisoi, poros, porosflo, hyd, suc, bex, hsoi[:6], jnp.zeros(n), jnp.zeros(n), jnp.zeros(n), bperm, dt, rhow, eps)
        drain += np.asarray(h.wflo[:, -1]) * dt
        w2, _, gadj = SP.wadjust(h.wsoi, wisoi, poros, hsoi[:6], rhow, dt, eps)
        created += np.asarray(gadj) * dt; w = w2
    store1 = np.asarray(jnp.sum(hsoi[None, :6] * poros * rhow * ((1 - wisoi) * w + wisoi), 1))
    f = 8760.0 / nsteps
    return drain * f, created * f, (store1 - store0) * f, np.asarray(w)
for name, fn in (('as is', SP.soilh2o), ('weights corrected', mod.soilh2o)):
    d, c, ds, w = run(fn)
    print(f'\n{name}: 480 hourly steps, no inputs; per-year equivalents, mm')
    print(f"{'column':14s} {'drainage':>9s} {'created':>9s} {'dStore':>8s}  {'closure':>8s}   wsoi after")
    for i, (nm, *_) in enumerate(cells):
        print(f'{nm:14s} {d[i]:9.0f} {c[i]:9.0f} {ds[i]:8.0f}  {ds[i] + d[i] - c[i]:8.1f}   ' + ' '.join(f'{x:.2f}' for x in w[i]))
