"""Summer light response at towers, model vs tower (CALIBRATION.md 22-23): is the forest GPP deficit capacity or light use?

    python verify/calibration/diagnostics/light_response.py <site_runs.npz> <IGBP> [<figure.png>]

<site_runs.npz> is a `site_run.py` output with daily GPP (gpp_daily, years x 366 x towers; tower meteorology, so the model
sees the tower's shortwave). For the towers of the class: June-August days of the evaluation years 2009-2011, daily GPP
(tower: mean of the NT and DT partitions, days with NEE_VUT_REF_QC >= 0.8) binned by the tower's daily incoming shortwave.
A deficit that is the same fraction at every light level is light use (absorption or quantum efficiency); one that opens
up only at high light is capacity (vmax, or stomatal/water limitation at the plateau).
"""
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, 'python-port'); sys.path.insert(0, 'verify/calibration')
from site_run import META
FX = '/shared/nas2/Hangkai/ibis/inputs/sites/FLUXNET2015_FULLSET/daily_monthly_yearly'
npz, igbp = sys.argv[1], sys.argv[2]; fig_path = sys.argv[3] if len(sys.argv) > 3 else ''
R = np.load(npz, allow_pickle=True); sites = [str(s) for s in R['sites']]; years = [int(y) for y in R['years']]
EVAL = [y for y in years if 2009 <= y <= 2011]
BINS = np.arange(0, 401, 50)
rows = []; curves = {}
for n, s in enumerate(sites):
    if META.loc[s, 'IGBP'] != igbp: continue
    f = glob.glob(f'{FX}/FLX_{s}_FLUXNET2015_FULLSET_DD_*.csv')
    if not f: continue
    d = pd.read_csv(f[0]); t = pd.to_datetime(d['TIMESTAMP'].astype(str), format='%Y%m%d')
    w = lambda c: d[c].where(d[c] > -9000) if c in d else pd.Series(np.nan, index=d.index)
    g = pd.concat([w('GPP_NT_VUT_REF'), w('GPP_DT_VUT_REF')], axis=1).mean(axis=1); sw = w('SW_IN_F'); q = w('NEE_VUT_REF_QC')
    summer = t.dt.month.isin([6, 7, 8]) if META.loc[s, 'LOCATION_LAT'] >= 0 else t.dt.month.isin([12, 1, 2])
    mod, obs, rad = [], [], []
    for y in EVAL:
        k = years.index(y); sel = (t.dt.year == y) & summer & (q >= 0.8) & g.notna() & sw.notna()
        doy = (t[sel] - pd.Timestamp(f'{y}-01-01')).dt.days.values
        mod.append(R['gpp_daily'][k, doy, n]); obs.append(g[sel].values); rad.append(sw[sel].values)
    mod, obs, rad = map(np.concatenate, (mod, obs, rad))
    if len(rad) < 30: continue
    ib = np.digitize(rad, BINS) - 1; cm, co = [], []
    for b in range(len(BINS) - 1):
        m = ib == b; cm.append(np.nanmean(mod[m]) if m.sum() >= 5 else np.nan); co.append(np.nanmean(obs[m]) if m.sum() >= 5 else np.nan)
    curves[s] = (np.array(cm), np.array(co))
    lo, hi = rad < 150, rad >= 250
    rows.append(dict(site=s, days=len(rad), ratio_all=mod.sum() / obs.sum(), ratio_low_light=mod[lo].sum() / max(obs[lo].sum(), 1e-6) if lo.sum() >= 5 else np.nan,
                     ratio_high_light=mod[hi].sum() / max(obs[hi].sum(), 1e-6) if hi.sum() >= 5 else np.nan,
                     tower_gpp_per_sw=np.polyfit(rad, obs, 1)[0] * 100, model_gpp_per_sw=np.polyfit(rad, mod, 1)[0] * 100))
T = pd.DataFrame(rows).set_index('site').round(3)
print(f"{igbp}: summer (JJA / DJF south) days 2009-2011 with NEE QC >= 0.8, daily GPP gC m-2 d-1 vs tower SW_IN_F W m-2")
print("ratio_* = model / tower GPP over the days in the light class (low < 150, high >= 250 W m-2); *_per_sw = slope, gC m-2 d-1 per 100 W m-2")
print(T.to_string()); print('medians: ' + '  '.join(f"{c} {T[c].median():.3f}" for c in T.columns if c != 'days'))
if fig_path and curves:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    nc = 6; nr = -(-len(curves) // nc); fig, axs = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.0 * nr + 0.7), squeeze=False)
    xc = (BINS[:-1] + BINS[1:]) / 2
    for a in axs.ravel()[len(curves):]: a.set_visible(False)
    for a, (s, (cm, co)) in zip(axs.ravel(), curves.items()):
        a.plot(xc, co, 'k-o', ms=2.5, lw=1, label='tower'); a.plot(xc, cm, color='tab:red', lw=1.3, label='model'); a.set_title(s, fontsize=8)
        a.set_xlabel('SW_IN W m-2', fontsize=7)
    axs[0, 0].set_ylabel('GPP gC m-2 d-1'); h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=2, loc='lower center')
    fig.suptitle(f'{igbp} summer light response, daily GPP binned by incoming shortwave (tower met; {npz.split("/")[-2]})', fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig(fig_path, dpi=130); print('figure', fig_path)
