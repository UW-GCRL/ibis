"""Layer 2: distributional check of Min's JAX weather vs reference Python.

Reference Python is now Layer-1-verified bit-exact against F77 (with the
documented Markov-bug finding #5). This script answers: does the JAX
vectorized port produce *distributionally* equivalent output to the
reference (algorithmic) Python?

Since JAX uses a different RNG (jax.random vs ran2) and a different
algorithmic decomposition (pre-generated monthly precip via gamma+scan,
truncated_normal vs Box-Muller rejection, per-grid-point event timing),
bit-exact is structurally impossible. We compare ensemble moments and
mass-balance quantities with SANITY tolerance.

Distributional checks:
  (A) Monthly precip total per grid point:  mean ≈ xinprec * ndaypm  (mass)
  (B) Wet-day fraction per grid point:      mean ≈ xinwet / ndaypm   (frequency)
  (C) Ensemble mean of tmax/tmin/cloud/qd/ud across N draws ≈ expected values
  (D) Ensemble std of tmax/tmin ≈ tmaxs/tmins regression
  (E) Daily-precip distribution KS-like check (max-CDF-distance)

Findings expected to be visible:
  - #5 (Markov bug): F77/ref wet-day clustering is broken (no storm
    persistence). JAX clusters correctly. Per-point wet-day frequency
    matches in expectation; per-point lag-1 wet-day autocorrelation does NOT.
  - Algorithmic finding candidate #6: JAX `diurnal` takes per-grid-point
    plens/startp/endp; F77 uses scalars. Same expectation but different
    spatial coupling.

Run:
    cd /home/hangkai/ibis-port
    PYTHONPATH=. verify/.venv/bin/python -m verify.weather.verify_layer2
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ibis.physics.weather import (
    ClimateData,
    generate_month_precip as jax_gen_precip,
    daily as jax_daily,
)
from verify.weather.reference_daily import (
    Climate, DailyState, daily as ref_daily,
)


# ---------------------------------------------------------------------------
# Climate setup shared by both sides.
# ---------------------------------------------------------------------------

def make_climate_pair(n: int, climate_seed: int):
    """Return (ref Climate, JAX ClimateData) with identical numerical values."""
    rng = np.random.default_rng(climate_seed)
    xint    = rng.uniform(-10.0, 35.0, (n, 12))
    xinq    = rng.uniform(20.0, 95.0, (n, 12))
    xinprec = rng.uniform(0.1, 8.0,   (n, 12))
    xinwind = rng.uniform(1.0, 6.0,   (n, 12))
    xincld  = rng.uniform(10.0, 90.0, (n, 12))
    xinwet  = rng.uniform(3.0, 25.0,  (n, 12))
    xintrng = rng.uniform(4.0, 18.0,  (n, 12))
    xintopo = rng.uniform(0.0, 2000.0, n)

    ref = Climate(xint=xint.copy(), xinq=xinq.copy(), xinprec=xinprec.copy(),
                  xinwind=xinwind.copy(), xincld=xincld.copy(),
                  xinwet=xinwet.copy(), xintrng=xintrng.copy(),
                  xintopo=xintopo.copy())
    jax_climate = ClimateData(
        xint=jnp.asarray(xint), xinq=jnp.asarray(xinq),
        xinprec=jnp.asarray(xinprec), xinwind=jnp.asarray(xinwind),
        xincld=jnp.asarray(xincld), xinwet=jnp.asarray(xinwet),
        xintrng=jnp.asarray(xintrng), xintopo=jnp.asarray(xintopo),
    )
    return ref, jax_climate


# ---------------------------------------------------------------------------
# Ensembles.
# ---------------------------------------------------------------------------

NDAYPM = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def ref_ensemble(ref_clim: Climate, imonth: int, n_draws: int,
                 seed_base: int = 1000):
    """Run reference Python for n_draws independent seeds.

    For each draw, we generate a full month (iday=1..ndaypm) so we get
    a full sample of daily values, then collect daily outputs.

    Returns dict with arrays shaped [n_draws, ndaypm, n_points] for daily
    outputs and [n_draws, n_points] for monthly totals.
    """
    n = ref_clim.xint.shape[0]
    nd = int(NDAYPM[imonth - 1])

    tmax = np.empty((n_draws, nd, n), dtype=np.float64)
    tmin = np.empty((n_draws, nd, n), dtype=np.float64)
    td   = np.empty((n_draws, nd, n), dtype=np.float64)
    cloud= np.empty((n_draws, nd, n), dtype=np.float64)
    qd   = np.empty((n_draws, nd, n), dtype=np.float64)
    ud   = np.empty((n_draws, nd, n), dtype=np.float64)
    precip = np.empty((n_draws, nd, n), dtype=np.float64)
    iwet = np.empty((n_draws, nd, n), dtype=np.int64)

    for k in range(n_draws):
        seed = seed_base + k
        # Each draw uses fresh state + a copy of climate (xinwet may mutate).
        clim = Climate(*(arr.copy() for arr in ref_clim))
        state = DailyState.fresh(n)
        for iday in range(1, nd + 1):
            out = ref_daily(seed, imonth, iday, clim, state)
            tmax[k, iday - 1]   = out.tmax
            tmin[k, iday - 1]   = out.tmin
            td  [k, iday - 1]   = out.td
            cloud[k, iday - 1]  = out.cloud
            qd  [k, iday - 1]   = out.qd
            ud  [k, iday - 1]   = out.ud
            precip[k, iday - 1] = out.precip
            iwet[k, iday - 1]   = out.iwet
    return dict(tmax=tmax, tmin=tmin, td=td, cloud=cloud, qd=qd, ud=ud,
                precip=precip, iwet=iwet,
                precip_monthly_total=precip.sum(axis=1),       # [k, n]
                iwet_monthly_total=iwet.sum(axis=1))           # [k, n]


def jax_ensemble(jax_clim: ClimateData, imonth: int, n_draws: int,
                 key_base: int = 1000):
    n = jax_clim.xint.shape[0]
    imonth0 = imonth - 1  # JAX is 0-indexed for imonth
    nd = int(NDAYPM[imonth0])

    tmax = np.empty((n_draws, nd, n), dtype=np.float64)
    tmin = np.empty((n_draws, nd, n), dtype=np.float64)
    td   = np.empty((n_draws, nd, n), dtype=np.float64)
    cloud= np.empty((n_draws, nd, n), dtype=np.float64)
    qd   = np.empty((n_draws, nd, n), dtype=np.float64)
    ud   = np.empty((n_draws, nd, n), dtype=np.float64)
    precip = np.empty((n_draws, nd, n), dtype=np.float64)
    iwet = np.empty((n_draws, nd, n), dtype=np.int64)

    for k in range(n_draws):
        key = jax.random.PRNGKey(key_base + k)
        key, monthly = jax_gen_precip(key, jax_clim, imonth0)
        xstore = jnp.zeros((n, 3))
        for iday0 in range(nd):
            key, out = jax_daily(key, imonth0, iday0, jax_clim, monthly, xstore)
            xstore = out.xstore
            tmax[k, iday0]   = np.asarray(out.tmax)
            tmin[k, iday0]   = np.asarray(out.tmin)
            td  [k, iday0]   = np.asarray(out.td)
            cloud[k, iday0]  = np.asarray(out.cloud)
            qd  [k, iday0]   = np.asarray(out.qd)
            ud  [k, iday0]   = np.asarray(out.ud)
            precip[k, iday0] = np.asarray(out.precip)
            iwet[k, iday0]   = np.asarray(out.iwet)
    return dict(tmax=tmax, tmin=tmin, td=td, cloud=cloud, qd=qd, ud=ud,
                precip=precip, iwet=iwet,
                precip_monthly_total=precip.sum(axis=1),
                iwet_monthly_total=iwet.sum(axis=1))


# ---------------------------------------------------------------------------
# Comparison helpers.
# ---------------------------------------------------------------------------

def _mean_diff_per_point(a: np.ndarray, b: np.ndarray) -> dict:
    """a,b shape [n_draws, n_points]. Compare per-point ensemble means."""
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    abs_diff = np.abs(ma - mb)
    return dict(
        ref_mean=float(ma.mean()), jax_mean=float(mb.mean()),
        max_abs_diff=float(abs_diff.max()),
        mean_abs_diff=float(abs_diff.mean()),
        rms_diff=float(np.sqrt(((ma - mb) ** 2).mean())),
    )


def _var_diff_per_point(a: np.ndarray, b: np.ndarray) -> dict:
    va = a.var(axis=0, ddof=1)
    vb = b.var(axis=0, ddof=1)
    ratio = vb / np.maximum(va, 1e-30)
    return dict(
        ref_std=float(np.sqrt(va.mean())),
        jax_std=float(np.sqrt(vb.mean())),
        var_ratio_mean=float(ratio.mean()),
        var_ratio_p05=float(np.percentile(ratio, 5)),
        var_ratio_p95=float(np.percentile(ratio, 95)),
    )


def _ks_max_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS-stat (max CDF distance). Flattened arrays."""
    af = np.sort(a.flatten())
    bf = np.sort(b.flatten())
    # Empirical CDFs on union grid.
    grid = np.unique(np.concatenate([af, bf]))
    ca = np.searchsorted(af, grid, side="right") / len(af)
    cb = np.searchsorted(bf, grid, side="right") / len(bf)
    return float(np.max(np.abs(ca - cb)))


def _lag1_autocorr_wet(iwet: np.ndarray) -> np.ndarray:
    """Compute lag-1 autocorrelation of wet/dry flags per grid point.

    iwet shape [n_draws, ndaypm, n_points] -> returns [n_points] mean lag-1
    autocorrelation averaged over draws.
    """
    n_draws, nd, n = iwet.shape
    if nd < 2:
        return np.zeros(n)
    a = iwet[:, :-1, :].astype(np.float64)
    b = iwet[:, 1:,  :].astype(np.float64)
    ma = a.mean(axis=1, keepdims=True)
    mb = b.mean(axis=1, keepdims=True)
    num = ((a - ma) * (b - mb)).mean(axis=1)
    da = np.sqrt(((a - ma) ** 2).mean(axis=1))
    db = np.sqrt(((b - mb) ** 2).mean(axis=1))
    denom = da * db
    corr = np.where(denom > 1e-12, num / np.maximum(denom, 1e-12), 0.0)
    return corr.mean(axis=0)


# ---------------------------------------------------------------------------
# Per-scenario report.
# ---------------------------------------------------------------------------

def run_scenario(label: str, n_points: int, imonth: int, n_draws: int,
                 climate_seed: int) -> dict:
    print(f"\n=== Layer 2 scenario: {label} (n_points={n_points}, "
          f"imonth={imonth}, n_draws={n_draws}) ===")
    ref_clim, jax_clim = make_climate_pair(n_points, climate_seed)

    ref_e = ref_ensemble(ref_clim, imonth, n_draws)
    jax_e = jax_ensemble(jax_clim, imonth, n_draws)

    results = {}

    # ---- (A) Monthly precip total per grid point (mass balance) ----
    # Both should target xinprec[:, m] * ndaypm exactly.
    target = ref_clim.xinprec[:, imonth - 1] * NDAYPM[imonth - 1]
    a_ref = ref_e["precip_monthly_total"]      # [n_draws, n]
    a_jax = jax_e["precip_monthly_total"]
    ref_to_target = float(np.abs(a_ref.mean(axis=0) - target).max())
    jax_to_target = float(np.abs(a_jax.mean(axis=0) - target).max())
    print(f"(A) Monthly precip total max-abs-bias from target:")
    print(f"    ref->target = {ref_to_target:.3e}    jax->target = {jax_to_target:.3e}")
    results["monthly_precip_total"] = dict(
        target_max_abs=target.max(), ref_to_target=ref_to_target,
        jax_to_target=jax_to_target,
    )

    # ---- (B) Wet-day fraction ----
    target_wet = ref_clim.xinwet[:, imonth - 1] / NDAYPM[imonth - 1]
    ref_wf = (ref_e["iwet_monthly_total"] / NDAYPM[imonth - 1]).mean(axis=0)
    jax_wf = (jax_e["iwet_monthly_total"] / NDAYPM[imonth - 1]).mean(axis=0)
    ref_to_t = float(np.abs(ref_wf - target_wet).max())
    jax_to_t = float(np.abs(jax_wf - target_wet).max())
    print(f"(B) Wet-day fraction max-abs vs target xinwet/ndaypm:")
    print(f"    ref->target = {ref_to_t:.3e}    jax->target = {jax_to_t:.3e}")
    results["wet_day_fraction"] = dict(ref_to_target=ref_to_t,
                                       jax_to_target=jax_to_t)

    # ---- (C) Lag-1 wet/dry autocorrelation -- finding #5 manifests here ----
    ref_ac = _lag1_autocorr_wet(ref_e["iwet"])
    jax_ac = _lag1_autocorr_wet(jax_e["iwet"])
    # Expected target lag-1 for Markov chain (pww - pwd, where p_wet=xinwet/ndpm)
    pwet = target_wet
    pwd  = 0.75 * pwet
    pww  = 0.25 + pwd
    target_ac = pww - pwd       # = 0.25 (constant if Markov is correct)
    print(f"(C) Lag-1 wet/dry autocorrelation (finding #5 indicator):")
    print(f"    ref (F77 with bug)  mean={ref_ac.mean():.4f}  expected if "
          f"correct={target_ac.mean():.4f}")
    print(f"    jax (correct algo)  mean={jax_ac.mean():.4f}")
    results["lag1_autocorr"] = dict(
        ref_mean=float(ref_ac.mean()), jax_mean=float(jax_ac.mean()),
        target_correct=float(target_ac.mean()),
    )

    # ---- (D) Ensemble mean / std of continuous outputs ----
    flat_keys = ("tmax", "tmin", "td", "cloud", "qd", "ud")
    for key in flat_keys:
        # Compare mean over all (draw, day, point) values flattened across day.
        # Take per-(day, point) ensemble mean, then check per-point match.
        rm = ref_e[key].mean(axis=0)         # [nd, n]
        jm = jax_e[key].mean(axis=0)         # [nd, n]
        diff = np.abs(rm - jm)
        rms = float(np.sqrt(((rm - jm) ** 2).mean()))
        results[f"{key}_mean"] = dict(
            ref_mean=float(rm.mean()), jax_mean=float(jm.mean()),
            max_abs=float(diff.max()), rms=rms,
        )
        # Std comparison: take per-(day, point) std over draws, then compare.
        rs = ref_e[key].std(axis=0, ddof=1)
        js = jax_e[key].std(axis=0, ddof=1)
        results[f"{key}_std"] = dict(
            ref_std_mean=float(rs.mean()), jax_std_mean=float(js.mean()),
            ratio_jax_over_ref=float(js.mean() / max(rs.mean(), 1e-30)),
        )
        print(f"(D) {key:>6}: mean ref={rm.mean():.4f} jax={jm.mean():.4f} "
              f"max|diff|={diff.max():.3e};  std ref={rs.mean():.4f} "
              f"jax={js.mean():.4f} ratio={js.mean()/max(rs.mean(),1e-30):.3f}")

    # ---- (E) Daily precip distribution KS ----
    ks_precip = _ks_max_dist(ref_e["precip"], jax_e["precip"])
    print(f"(E) Daily-precip KS (max CDF distance): {ks_precip:.4f}")
    results["precip_ks"] = ks_precip

    return results


def main() -> int:
    scenarios = [
        ("temperate_jul", 32, 7,  100, 42),
        ("dry_jan",       32, 1,  100, 43),
    ]
    all_results = {}
    for label, n_pts, imonth, n_draws, cseed in scenarios:
        all_results[label] = run_scenario(label, n_pts, imonth, n_draws, cseed)

    print("\n=========== Layer 2 summary ===========")
    print(f"{'scenario':>16}  {'ref_lag1':>9}  {'jax_lag1':>9}  "
          f"{'precip_KS':>10}  {'tmax_diff':>10}")
    for label, r in all_results.items():
        print(f"{label:>16}  {r['lag1_autocorr']['ref_mean']:>9.4f}  "
              f"{r['lag1_autocorr']['jax_mean']:>9.4f}  "
              f"{r['precip_ks']:>10.4f}  {r['tmax_mean']['max_abs']:>10.3e}")
    print(f"\nExpected if Markov algorithm were correct: lag1 ~ {0.25:.4f}")
    print("Expected for F77 buggy version: lag1 close to 0 or negative")

    # Dump to CSV-able form.
    out_path = Path(__file__).parent / "results" / "layer2_summary.csv"
    out_path.parent.mkdir(exist_ok=True)
    lines = ["scenario,metric,ref,jax,target_or_diff"]
    for label, r in all_results.items():
        lines.append(f"{label},monthly_precip_to_target,"
                     f"{r['monthly_precip_total']['ref_to_target']},"
                     f"{r['monthly_precip_total']['jax_to_target']},0.0")
        lines.append(f"{label},wet_day_fraction_to_target,"
                     f"{r['wet_day_fraction']['ref_to_target']},"
                     f"{r['wet_day_fraction']['jax_to_target']},0.0")
        lines.append(f"{label},lag1_autocorr,"
                     f"{r['lag1_autocorr']['ref_mean']},"
                     f"{r['lag1_autocorr']['jax_mean']},"
                     f"{r['lag1_autocorr']['target_correct']}")
        lines.append(f"{label},precip_KS,,,{r['precip_ks']}")
        for k in ("tmax", "tmin", "td", "cloud", "qd", "ud"):
            lines.append(f"{label},{k}_mean,"
                         f"{r[k+'_mean']['ref_mean']},"
                         f"{r[k+'_mean']['jax_mean']},"
                         f"{r[k+'_mean']['max_abs']}")
            lines.append(f"{label},{k}_std,"
                         f"{r[k+'_std']['ref_std_mean']},"
                         f"{r[k+'_std']['jax_std_mean']},"
                         f"{r[k+'_std']['ratio_jax_over_ref']}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nResults CSV: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
