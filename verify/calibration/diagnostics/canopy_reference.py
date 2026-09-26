"""Reference canopy integrations against the tower's hourly light response (CALIBRATION.md 27).

    python verify/calibration/diagnostics/canopy_reference.py <record_july_lr.npz> <lat> [<LAI> <SAI>] [--vmax-scan]

<record_july_lr.npz> is the hourly July extract of a site_diag_year record (model_hourly_lr.py): the tower's incident
PPFD, the big leaf's temperature, ci, temperature- and stress-adjusted top-leaf vmax, leaf and stem area, hour of day
and the tower GPP of the same hours. With IBIS's own leaf biochemistry (Farquhar/Collatz as in physiology.f: alpha 0.08
on absorbed PAR, theta 0.95, beta 0.99, js = vmax/2.2, kc/ko/gamma* at leaf temperature) the same hours are integrated
over the canopy in several ways, all with light penetrating leaves AND stems and only leaves assimilating:

  ml_kn0.15 ...  multilayer (60 layers), sunlit and shaded leaves in every layer (de Pury & Farquhar 1997 light model:
                 kb = 0.5/sin(beta), kd = 0.78, leaf scattering 0.15), capacity V(x) = V0 e^-Kn x, Kn per unit LAI
  ss_kn0.15 ...  the two-big-leaf sunlit/shaded closed form of de Pury & Farquhar with the same profiles
  ml_mean_kn.15  multilayer with the layer-MEAN light (no sunlit/shaded split): the error de Pury calls "averaging"
  bigleaf_kn ... one big leaf: light limit from canopy APAR, capacity from the Kn profile, co-limited at canopy level
                 (the `cap_scale` switch); Kn = 0.15, and Kn = k-bar (a big leaf whose capacity follows the time-mean
                 light extinction, the Sellers 1992 argument) and Kn = 0.5 (JULES-type)
  f77 .........  the record's own canopy rate (IBIS as ported: capacity follows the instantaneous two-stream extinction)

Direct/diffuse from the tower clearness index (Spitters et al. 1986). Prints alpha and Amax of a rectangular hyperbola
fitted to each against incident PPFD, the mean rate in bright (> 1500) and dim (< 500 umol m-2 s-1) hours, and the
July daily mean, next to the tower; with --vmax-scan, the multilayer reference at 0.5-2x the top-leaf vmax.
"""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit

npz = sys.argv[1]; lat = float(sys.argv[2]); args = [a for a in sys.argv[3:] if not a.startswith('--')]
LAI = float(args[0]) if args else None; SAI = float(args[1]) if len(args) > 1 else None; scan = '--vmax-scan' in sys.argv
R = np.load(npz); ppfd, hod, tu, ci, vmax, gtow, qc = R['ppfd'], R['hod'], R['tu'] + 273.16, R['ci'], R['vmax'], R['gtow'], R['qc']
doy = R['doy'] if 'doy' in R else np.full_like(hod, 197); f77 = R['gppc']; fu = float(R['fu']) if 'fu' in R else 0.975
L = LAI if LAI is not None else float(np.mean(R['L'])); S = SAI if SAI is not None else float(np.mean(R['S']))
ALPHA, THETA, BETA, SIGMA, KD, O2 = 0.08, 0.95, 0.99, 0.15, 0.78, 0.209
rwork = 3.47e-3 - 1.0 / tu; tau = 4500.0 * np.exp(-4500.0 * rwork); kc = 1.5e-4 * np.exp(6000.0 * rwork); ko = 0.25 * np.exp(1500.0 * rwork)
gam = O2 / (2.0 * tau); fci_e = (ci - gam) / (ci + 2.0 * gam); fci_c = (ci - gam) / (ci + kc * (1.0 + O2 / ko))


def colimit(a, b, th):
    q = 0.5 * (a + b + np.sqrt(np.maximum((a + b) ** 2 - 4 * th * a * b, 0))) + 1e-30
    return np.minimum(q / th, a * b / q)


def leaf(apar, v):                       # gross rate of leaf area with absorbed PAR apar (umol m-2 s-1) and capacity v (umol)
    je = ALPHA * apar * fci_e; jc = v * fci_c; js = v / 2.2
    return colimit(colimit(je, jc, THETA), js, BETA)


# solar elevation and the direct/diffuse split (Spitters et al. 1986, hourly)
dec = 23.45 * np.pi / 180 * np.sin(2 * np.pi * (284 + doy) / 365); phi = lat * np.pi / 180
ha = np.pi / 12 * (hod + 0.5 - 12.0)     # tower files are local standard time; the half-hour centres the hour
sinb = np.maximum(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha), 0.02)
ppfd_pot = 1.92 * 1367.0 * sinb * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))   # extraterrestrial shortwave in PPFD units (tower PPFD_IN/SW_IN = 1.92 umol J-1)
kt = np.clip(ppfd / np.maximum(ppfd_pot, 1e-3), 0, 1)             # clearness index
Rr = 0.847 - 1.61 * sinb + 1.04 * sinb ** 2; K = (1.47 - Rr) / 1.66
fd = np.where(kt <= 0.22, 1.0, np.where(kt <= 0.35, 1 - 6.4 * (kt - 0.22) ** 2, np.where(kt <= K, 1.47 - 1.66 * kt, Rr)))
fd = np.clip(fd, 0.1, 1.0); Ib0, Id0 = ppfd * (1 - fd), ppfd * fd
kb = 0.5 / sinb; kbp = 0.46 / sinb; kdp = 0.719
rho_h = (1 - np.sqrt(1 - SIGMA)) / (1 + np.sqrt(1 - SIGMA)); rho_cb = 1 - np.exp(-2 * rho_h * kb / (1 + kb)); rho_cd = 0.036
PAI = L + S; fleaf = L / PAI                       # leaves are this share of the plant area at every depth


def multilayer(kn, v0, nlay=60, split=True):
    """Canopy gross rate per ground area: layers of plant area, sunlit/shaded leaves, capacity V0 e^-Kn x_leaf."""
    x = (np.arange(nlay) + 0.5) * PAI / nlay; dx = PAI / nlay; xl = x * fleaf
    out = np.zeros_like(ppfd)
    for xi, xli in zip(x, xl):
        fsun = np.exp(-kb * xi)
        Idif = (1 - rho_cd) * Id0 * kdp * np.exp(-kdp * xi)                       # diffuse absorbed per unit plant area
        Isca = Ib0 * ((1 - rho_cb) * kbp * np.exp(-kbp * xi) - (1 - SIGMA) * kb * np.exp(-kb * xi))   # scattered beam
        Isha = np.maximum(Idif + Isca, 0.0); Isun = Isha + (1 - SIGMA) * kb * Ib0
        v = v0 * np.exp(-kn * xli)
        if split: a = fsun * leaf(Isun, v) + (1 - fsun) * leaf(Isha, v)
        else: a = leaf(fsun * Isun + (1 - fsun) * Isha, v)
        out += a * fleaf * dx
    return out


def sunshade(kn, v0):
    """de Pury & Farquhar (1997) two big leaves, plant area for light and leaf area for capacity."""
    Ic = (1 - rho_cb) * Ib0 * (1 - np.exp(-kbp * PAI)) + (1 - rho_cd) * Id0 * (1 - np.exp(-kdp * PAI))
    Isun = ((1 - SIGMA) * Ib0 * (1 - np.exp(-kb * PAI)) + (1 - rho_cd) * Id0 * (1 - np.exp(-(kdp + kb) * PAI)) * kdp / (kdp + kb)
            + Ib0 * ((1 - rho_cb) * (1 - np.exp(-(kbp + kb) * PAI)) * kbp / (kbp + kb) - (1 - SIGMA) * (1 - np.exp(-2 * kb * PAI)) / 2))
    Isha = np.maximum(Ic - Isun, 0.0)
    knp = kn * fleaf                                   # capacity profile per unit plant area
    Vsun = v0 * fleaf * (1 - np.exp(-(knp + kb) * PAI)) / (knp + kb); Vsha = v0 * fleaf * (1 - np.exp(-knp * PAI)) / max(knp, 1e-9) - Vsun
    return leaf(Isun * fleaf, Vsun) + leaf(Isha * fleaf, Vsha)     # leaves get their share of each big leaf's APAR


def bigleaf(kn, v0):
    """One big leaf: light limit from the leaves' canopy APAR, capacity from the Kn profile, co-limited at canopy level."""
    Ic = ((1 - rho_cb) * Ib0 * (1 - np.exp(-kbp * PAI)) + (1 - rho_cd) * Id0 * (1 - np.exp(-kdp * PAI))) * fleaf
    Vc = v0 * (1 - np.exp(-kn * L)) / kn
    return leaf(Ic, Vc)


rh = lambda I, a, amax: a * I * amax / (a * I + amax)
sel = ppfd > 5; selt = sel & (qc == 0) & np.isfinite(gtow); bright = sel & (ppfd > 1500); dim = sel & (ppfd < 500)
kbar = float(np.sum(kb * ppfd) / np.sum(ppfd)); kb_noon = float(np.median(kb[(hod >= 11) & (hod <= 13)]))
print(f'{npz.split("/")[-1]}: LAI {L:.2f} SAI {S:.2f} fu {fu:.3f}; hours {sel.sum()}; light-weighted beam k {kbar:.2f}, noon {kb_noon:.2f}; '
      f'diffuse fraction mean {fd[sel].mean():.2f}; top-leaf vmax(T) mean {vmax[sel].mean() * 1e6:.1f} umol; ci {ci[sel].mean() * 1e6:.0f} ppm')
v0 = vmax * 1e6
tvm = lambda t: np.exp(3500.0 * (3.47e-3 - 1.0 / t)) / ((1 + np.exp(0.4 * (5 - (t - 273.16)))) * (1 + np.exp(0.4 * ((t - 273.16) - 50))))
v0_bern = v0 * (np.exp(65330.0 / 8.314 * (1 / 298.15 - 1 / tu)) / (tvm(tu) / tvm(298.16)))   # same Vcmax25, Bernacchi (2001) temperature response
runs = {'tower': gtow, 'f77 record': f77,
        'ml_kn0.15 (reference)': multilayer(0.15, v0) * fu, 'ml_kn0.30': multilayer(0.30, v0) * fu,
        'ss_kn0.15': sunshade(0.15, v0) * fu, 'ml_mean_kn0.15 (no sun/shade)': multilayer(0.15, v0, split=False) * fu,
        'bigleaf_kn0.15 (cap_scale)': bigleaf(0.15, v0) * fu, f'bigleaf_kn{kbar:.2f} (Sellers, time-mean k)': bigleaf(kbar, v0) * fu,
        'bigleaf_kn0.50': bigleaf(0.50, v0) * fu, 'ml_kn0.15_bernacchiT': multilayer(0.15, v0_bern) * fu,
        'ml_kn0.15_vmax1.26 (TRY 57.7)': multilayer(0.15, v0 * 1.26) * fu, 'ml_kn0.15_TRY+bernacchiT': multilayer(0.15, v0_bern * 1.26) * fu}
rows = []
for name, g in runs.items():
    s = selt if name == 'tower' else sel; s = s & np.isfinite(g)
    try: (a, amax), _ = curve_fit(rh, ppfd[s], g[s], p0=[0.04, 30], bounds=([0, 1], [0.3, 400]))
    except Exception: a = amax = np.nan
    rows.append(dict(canopy=name, alpha=a, Amax=amax, bright=np.nanmean(g[s & bright]), dim=np.nanmean(g[s & dim]),
                     daily_gC=np.nansum(np.where(selt, g, 0)) * 3600 * 12e-6 / 31))
T = pd.DataFrame(rows).set_index('canopy'); pd.set_option('display.width', 200)
print(T.round(3).to_string()); print('(daily_gC over the hours the tower has; bright/dim: umol m-2 s-1 in hours > 1500 / < 500 umol m-2 s-1 incident)')
if scan:
    print('\nvmax scan, multilayer sunlit/shaded reference, Kn 0.15 (tower: bright %.1f, dim %.1f, daily %.2f):' % (T.loc['tower', 'bright'], T.loc['tower', 'dim'], T.loc['tower', 'daily_gC']))
    for f in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        g = multilayer(0.15, v0 * f) * fu
        (a, amax), _ = curve_fit(rh, ppfd[sel], g[sel], p0=[0.04, 30], bounds=([0, 1], [0.3, 400]))
        print(f'  vmax x {f:<4}: alpha {a:.4f} Amax {amax:6.1f} bright {np.mean(g[bright]):5.1f} dim {np.mean(g[dim]):5.1f} daily {np.sum(np.where(selt, g, 0)) * 3600 * 12e-6 / 31:5.2f}')
BINS = np.arange(0, 2201, 200); ib = np.digitize(ppfd, BINS) - 1
print('\nbinned by incident PPFD (umol m-2 s-1):  ' + ' '.join(f'{b:>5d}' for b in BINS[:-1]))
for name, g in runs.items():
    s_ = (selt if name == 'tower' else sel) & np.isfinite(g)
    print(f'{name[:34]:<36} ' + ' '.join(f'{np.mean(g[s_ & (ib == b)]):5.1f}' if (s_ & (ib == b)).sum() >= 5 else '    -' for b in range(len(BINS) - 1)))
np.savez(npz.replace('.npz', '_reference.npz'), ppfd=ppfd, hod=hod, fd=fd, **{k.split(' ')[0]: v for k, v in runs.items()})
