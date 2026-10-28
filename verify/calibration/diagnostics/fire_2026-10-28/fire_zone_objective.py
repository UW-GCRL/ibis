"""Zone log-RMS objective (fire_f7_fit.py's) over lai_ref_vd x k_b on a run's 2003-2005 restarts; k_b per lai_ref_vd by golden
section on the objective. Read-only.  usage: python -I fire_zone_objective.py <run dir>"""
import sys, math, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open(__file__.replace('fire_zone_objective.py', 'fire_terms_regions.py')).read().split('# what-ifs')[0])
ZFIT = [z for z in ZONES if z not in ('E_ice', 'D_extreme', 'none')]
def zlog(pred): return float(np.sqrt(np.mean([np.log(max(wm(pred, zone == Z[z]), 1e-6) / max(wm(obs, zone == Z[z]), 1e-6)) ** 2 for z in ZFIT])))
def fit_kb(**kw):
    a_, b_ = math.log(0.1), math.log(3.0); g = (5 ** 0.5 - 1) / 2
    f = lambda lk: zlog(predict(P(k_b=math.exp(lk), **kw)))
    c_, d_ = b_ - g * (b_ - a_), a_ + g * (b_ - a_); fc, fd = f(c_), f(d_)
    for _ in range(14):
        if fc < fd: b_, d_, fd = d_, c_, fc; c_ = b_ - g * (b_ - a_); fc = f(c_)
        else: a_, c_, fc = c_, d_, fd; d_ = a_ + g * (b_ - a_); fd = f(d_)
    return math.exp((a_ + b_) / 2)
print(f"\n--- zone log-RMS objective: configuration, k_b, zlog, then model/GFED5 by region\n{'configuration':26s} {'k_b':>5s} {'zlog':>6s} " + ' '.join(f'{r[:9]:>9s}' for r in REG))
rows = [('current (k_b %.2f)' % float(y['k_b']), float(y['k_b']), {})]
for lr in (4.0, 3.5, 3.0, 2.5, 2.0):
    kw = dict(lai_ref_vd=lr); rows.append((f'lai_ref_vd {lr}', fit_kb(**kw), kw))
for cfu in (0.92, 0.88):
    kw = dict(closed_canopy_fu=cfu); rows.append((f'closed_canopy_fu {cfu}', fit_kb(**kw), kw))
for name, kb, kw in rows:
    pr = predict(P(k_b=kb, **kw))
    print(f'{name:26s} {kb:5.2f} {zlog(pr):6.3f} ' + ' '.join(f'{wm(pr, m) / max(wm(obs, m), 1e-9):9.2f}' for m in REG.values()))
print('\nzone means % per year (GFED5 | current | each row):')
preds = [predict(P(k_b=kb, **kw)) for _, kb, kw in rows]
for z in ZFIT:
    m = zone == Z[z]; print(f'  {z:12s} {100 * wm(obs, m):6.2f} | ' + ' '.join(f'{100 * wm(p_, m):5.2f}' for p_ in preds))
