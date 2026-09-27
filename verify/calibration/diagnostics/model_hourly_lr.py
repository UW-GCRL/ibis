"""What the upper big leaf does hour by hour in July, from a site_diag_year record (CALIBRATION.md 27).

    python verify/calibration/diagnostics/model_hourly_lr.py <record.npz> <SITE_hourly.csv> [<label>]

<record.npz> is a site_diag_year output (one tower); <SITE_hourly.csv> the tower's hourly file from
tower_light_response_hourly.py. Prints, for the daylight hours of July: the two-stream extinction coefficient k (mean,
PAR-weighted, median, by hour of day), the light-profile factor f_light and the capacity factor f_cap with the effective
leaf area each implies, which of the three leaf-level limits (light, Rubisco, sink) is smallest at the top leaf and how
often, the top leaf's inputs (absorbed PAR, vmax at leaf temperature and stress, ci, stress, wet fraction, gs), the
canopy rate, and the canopy light response against the tower's incident PPFD (rectangular-hyperbola alpha and Amax,
200-umol bins, bright > 1500 and dim < 500 umol m-2 s-1 hours) next to the tower's. Writes <record>_july_lr.npz with the
hourly extract that canopy_reference.py integrates.
"""
import sys, numpy as np, pandas as pd; from scipy.optimize import curve_fit
npz, csv = sys.argv[1], sys.argv[2]; lab = sys.argv[3] if len(sys.argv) > 3 else npz.split('/')[-1]
R = np.load(npz, allow_pickle=True); F = [str(x) for x in R['fields']]; g = lambda n: R['R'][:, F.index(n), 0].astype(float)
fu = float(R['fu'][0]); year = int(R['year']); leap = year % 4 == 0
L, S, k, fl, fc, par = g('lai_u'), g('sai_u'), g('k_u'), g('fl_u'), g('fc_u'), g('par_u')
agc, agtop, je, jc, js = g('agc_ub'), g('agtop_ub'), g('je_ub'), g('jc_ub'), g('js_ub')
vmax, vnom, tu, ci, stress, fwet, tvm, gs = g('vmax_ub'), g('vnom_ub'), g('tu'), g('ci_ub'), g('stress_u'), g('fwet_u'), g('tempvm_u'), g('gs_ub')
h = np.arange(len(L)); doy = h // 24 + 1; hod = h % 24
jul = (doy >= 182 + leap) & (doy <= 212 + leap); day = jul & (par > 10)
gppc = agc * L * fu * 1e6                    # canopy gross, umol m-2 s-1 per ground
tw = pd.read_csv(csv); tw = tw[tw.year == year].set_index(['doy', 'hour'])
ppfd = np.array([tw.ppfd.get((int(d_), int(hh)), np.nan) for d_, hh in zip(doy, hod)])
gtow = np.array([tw.gpp.get((int(d_), int(hh)), np.nan) for d_, hh in zip(doy, hod)])
qc = np.array([tw.NEE_VUT_REF_QC.get((int(d_), int(hh)), np.nan) for d_, hh in zip(doy, hod)])
print(f'== {lab}: {year} July, LAI {L[day].mean():.2f} SAI {S[day].mean():.2f} fu {fu:.3f}; daylight hours {day.sum()}')
print('k_u: mean %.2f, PAR-weighted %.2f, median %.2f; by hour of day:' % (k[day].mean(), (k * par)[day].sum() / par[day].sum(), np.median(k[day])))
print('   ' + '  '.join(f'{hh:02d}h {np.median(k[day & (hod == hh)]):.2f}' for hh in range(5, 21) if (day & (hod == hh)).sum() > 3))
print('f_light %.3f -> effective leaf area f_light*L = %.2f ; f_cap %.3f -> f_cap*L = %.2f' % (fl[day].mean(), (fl * L)[day].mean(), fc[day].mean(), (fc * L)[day].mean()))
lim = np.argmin(np.stack([je, jc, js]), 0)
print('top-leaf smallest rate: light %.0f%%, Rubisco %.0f%%, sink %.0f%% of daylight hours' % tuple(100 * np.mean(lim[day] == i) for i in range(3)))
print('top leaf daylight means: absorbed par %.0f W/m2 (=%.0f umol), vmax(T,stress) %.1f, vnom %.1f, tempvm %.2f, T %.1f C, ci %.0f ppm, stress %.3f, fwet %.3f, gs %.3f' % (
    par[day].mean(), par[day].mean() * 4.59, vmax[day].mean() * 1e6, vnom[day].mean() * 1e6, tvm[day].mean(), tu[day].mean(), ci[day].mean() * 1e6, stress[day].mean(), fwet[day].mean(), gs[day].mean()))
print('top leaf je %.1f jc %.1f js %.1f agtop %.1f  -> canopy agc %.2f (per leaf), canopy GPP %.1f umol m-2 s-1' % (je[day].mean() * 1e6, jc[day].mean() * 1e6, js[day].mean() * 1e6, agtop[day].mean() * 1e6, agc[day].mean() * 1e6, gppc[day].mean()))
sel = jul & np.isfinite(ppfd) & (ppfd > 5) & (hod >= 5) & (hod <= 20)
rh = lambda I, a, amax: a * I * amax / (a * I + amax)
(a, amax), _ = curve_fit(rh, ppfd[sel], gppc[sel], p0=[0.04, 30], bounds=([0, 1], [0.3, 200]))
selt = sel & (qc == 0) & np.isfinite(gtow)
(at, amaxt), _ = curve_fit(rh, ppfd[selt], gtow[selt], p0=[0.04, 30], bounds=([0, 1], [0.3, 200]))
print('canopy light response vs tower PPFD_IN, July %d: model alpha %.4f Amax %.1f | tower alpha %.4f Amax %.1f' % (year, a, amax, at, amaxt))
BINS = np.arange(0, 2201, 200); ib = np.digitize(ppfd, BINS) - 1
mb = [gppc[sel & (ib == b)].mean() if (sel & (ib == b)).sum() >= 5 else np.nan for b in range(len(BINS) - 1)]
tb = [gtow[selt & (ib == b)].mean() if (selt & (ib == b)).sum() >= 5 else np.nan for b in range(len(BINS) - 1)]
print('bins  ' + ' '.join(f'{b:>5d}' for b in BINS[:-1])); print('model ' + ' '.join(f'{x:5.1f}' for x in mb)); print('tower ' + ' '.join(f'{x:5.1f}' for x in tb))
print('daily GPP July over hours with tower data: model %.2f, tower %.2f gC m-2 d-1' % (np.nansum(np.where(np.isfinite(gtow) & jul, gppc, 0)) * 3600 * 12e-6 / 31, np.nansum(np.where(jul, gtow, 0)) * 3600 * 12e-6 / 31))
mid = day & (ppfd > 1500); dim = day & (ppfd < 500)
print('bright hours (PPFD>1500): k %.2f, je_top %.1f jc_top %.1f js_top %.1f, agtop %.1f, canopy GPP %.1f, tower %.1f' % (k[mid].mean(), je[mid].mean() * 1e6, jc[mid].mean() * 1e6, js[mid].mean() * 1e6, agtop[mid].mean() * 1e6, gppc[mid].mean(), np.nanmean(gtow[mid & (qc == 0)])))
print('dim hours (PPFD<500):    k %.2f, je_top %.1f jc_top %.1f js_top %.1f, agtop %.1f, canopy GPP %.1f, tower %.1f' % (k[dim].mean(), je[dim].mean() * 1e6, jc[dim].mean() * 1e6, js[dim].mean() * 1e6, agtop[dim].mean() * 1e6, gppc[dim].mean(), np.nanmean(gtow[dim & (qc == 0)])))
extra = {}
if 'rad_solad' in R and R['rad_solad'].shape[0] == len(L):           # the model's own hourly light (visible direct, diffuse W m-2)
    extra = dict(solad=R['rad_solad'][:, 0][sel], solai=R['rad_solai'][:, 0][sel])
    m_ppfd = (R['rad_solad'][:, 0] + R['rad_solai'][:, 0]) * 4.59
    print('model incident PPFD vs tower PPFD, July daylight: mean %.0f vs %.0f umol m-2 s-1; hourly r %.3f; model diffuse fraction (PAR-weighted) %.2f' % (
        m_ppfd[day].mean(), np.nanmean(ppfd[day]), np.corrcoef(m_ppfd[sel], ppfd[sel])[0, 1], (R['rad_solai'][:, 0][day].sum() / max((R['rad_solad'][:, 0] + R['rad_solai'][:, 0])[day].sum(), 1e-6))))
np.savez(npz.replace('.npz', '_july_lr.npz'), ppfd=ppfd[sel], gppc=gppc[sel], hod=hod[sel], doy=doy[sel], fu=fu, k=k[sel], je=je[sel], jc=jc[sel], js=js[sel], par=par[sel], tu=tu[sel], ci=ci[sel],
         vmax=vmax[sel], L=L[sel], S=S[sel], fl=fl[sel], fc=fc[sel], gtow=gtow[sel], qc=qc[sel], **extra)
