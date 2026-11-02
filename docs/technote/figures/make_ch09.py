"""Figures of Chapter 9 (photosynthesis and stomatal conductance) of the IBIS 3.0 technical note.

Run from the repository root with the project interpreter (CPU is enough):

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch09.py            # all figures
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch09.py 9.2        # one

Every number a figure shows is computed here: the leaf rates with the model's own `farquhar_c3`, `farquhar_c4` and
`_temp_dep_params`; the canopy integrals with the model's two-stream (`twoset`, `twostr`), `canopy_scaling`,
`sunshade_canopy` and `_sunshade_c3`, under the light of the model's weather generator (`diurnal`); the stress
function with the model's `drystress`; the acclimation state from the GSWP3 daily mean temperature with the
recursion of `timestepping/loops_gswp3.py` and the factor of `timestepping/loops.py`. Parameter values are those of
the release parameter file, gathered for a zone's types with `ibis.io.pft_types`. The script prints the numbers the
chapter quotes.

The code changes of the technote review (2026-11-01) do not change these figures: run on the revised code
(2026-11-02), the script writes PNG files byte-identical to those of the release code and prints the same numbers. The
review's changes to this chapter's processes (the acclimated capacity in the GPP split, the frost ramp's input, the
light factor of `acclim_light`, the larch transpiration, the two-stream window) act outside what the figures show.
Figure 9.1 was redrawn on 2026-11-02 for its layout only: in panel (c) the label of the c_i line of (b) moved to the
left of its line, where it can no longer be read as the label of the line of (a).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: E402,F401  (style, rcParams)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

PFILE = ROOT / "python-port/configs/params/params_31s_ch4h.yaml"
KOPPEN = "/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz"
GSWP3 = "/shared/nas2/Min/GSWP3"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
REF = "#8a8a86"
LEG = dict(frameon=False, fontsize=6.8, handlelength=2.0, borderaxespad=0.3)

CA = 380.0e-6          # atmospheric CO2 of the examples (mol mol-1), about the 2005 value of Table 3.2
CI3 = 0.7 * CA         # C3 intercellular CO2 of the examples
CI4 = 0.4 * CA         # C4 intercellular CO2 of the examples
TLEAF = 298.16         # leaf temperature of the examples (25 C)


def _style(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _title(ax, s):
    ax.set_title(s, loc="left", fontsize=8.5, weight="bold", color=INK)


def _release():
    """(release parameter dict, StomataParams of the release, type overrides)."""
    from ibis.io.physics_params import load_physics_params, build_stomata_params
    from ibis.io import pft_types as pt
    pd = load_physics_params(str(PFILE))
    return pd, build_stomata_params(pd), pt.parse_type_overrides(pd.get("types"))


def _zone_values(zone, field_base, key, over):
    """(12,) values of a per-PFT parameter in the positions of `zone` (equation 2.1)."""
    from ibis.io import pft_types as pt
    slots = pt.slot_map(np.array([pt.ZI[zone]]))
    return pt.gather(np.asarray(field_base, float), slots, over.get(key))[0].astype(float)


# ---------------------------------------------------------------------------------------------------------------
# Figure 9.1: the leaf model
# ---------------------------------------------------------------------------------------------------------------

def fig_9_1():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ibis.physics.stomata import farquhar_c3, farquhar_c4, _temp_dep_params
    pd, sp, over = _release()
    v3 = _zone_values("D_temperate", sp.vmax_pft, "vmax", over)[4]        # T15 in position 5
    v4 = _zone_values("A_savanna", sp.vmax_pft, "vmax", over)[10]         # G01 in position 11
    g3, g4 = float(sp.gamma[0]), float(sp.gamma[3])
    T = jnp.array([TLEAF])
    kc, ko, _, gs_, f3, f4 = (np.asarray(x)[0] for x in _temp_dep_params(T, sp))
    V3, V4 = v3 * f3, v4 * f4                                                # Vmax at 25 C, no water stress
    R3, R4 = g3 * v3 * f3, g4 * v4 * f4
    par = np.linspace(0.0, 400.0, 401)
    n = len(par); one = np.ones(n)
    ag3, _ = farquhar_c3(jnp.asarray(par), jnp.asarray(CI3 * one), jnp.asarray(V3 * one), gs_, kc, ko,
                         float(sp.o2conc), jnp.asarray(R3 * one), sp)
    ag4, _ = farquhar_c4(jnp.asarray(par), jnp.asarray(CI4 * one), jnp.asarray(V4 * one), jnp.asarray(R4 * one), sp)
    # the limiting rates, as coded in farquhar_c3 / farquhar_c4
    je3 = par * 4.59e-6 * float(sp.alpha3) * (CI3 - gs_) / (CI3 + 2.0 * gs_)
    jc3 = V3 * (CI3 - gs_) / (CI3 + kc * (1.0 + float(sp.o2conc) / ko)) * one
    js3 = V3 / 2.2 * one
    je4 = par * 4.59e-6 * float(sp.alpha4); jc4 = V4 * one; ji4 = 18.0e3 * V4 * CI4 * one
    u = 1e6
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.9), gridspec_kw={"hspace": 0.42, "wspace": 0.28})
    ax = axs[0, 0]
    ax.plot(par, np.asarray(ag3) * u, color=BLUE, lw=1.6, label="gross photosynthesis $A_g$")
    ax.plot(par, je3 * u, color=ORANGE, lw=0.9, ls="--", label="light limit $J_e$")
    ax.plot(par, jc3 * u, color=AQUA, lw=0.9, ls="--", label="Rubisco limit $J_c$")
    ax.plot(par, js3 * u, color=YELLOW, lw=0.9, ls="--", label="sink limit $J_s$")
    ax.set_ylim(0, 40); ax.set_xlim(0, 400)
    ax.set_xlabel("absorbed PAR (W m$^{-2}$)"); ax.set_ylabel("µmol CO$_2$ m$^{-2}$ s$^{-1}$")
    ax.legend(loc="lower right", **LEG); _title(ax, "(a) C3 leaf: light response")
    ax = axs[0, 1]
    ax.plot(par, np.asarray(ag4) * u, color=BLUE, lw=1.6, label="gross photosynthesis $A_g$")
    ax.plot(par, je4 * u, color=ORANGE, lw=0.9, ls="--", label="light limit $J_e$")
    ax.plot(par, jc4 * u, color=AQUA, lw=0.9, ls="--", label="Rubisco limit $J_c$")
    ax.plot(par, ji4 * u, color=YELLOW, lw=0.9, ls="--", label="CO$_2$ limit $J_i$")
    ax.set_ylim(0, 40); ax.set_xlim(0, 400)
    ax.set_xlabel("absorbed PAR (W m$^{-2}$)"); ax.set_ylabel("µmol CO$_2$ m$^{-2}$ s$^{-1}$")
    ax.legend(loc="center right", **LEG); _title(ax, "(b) C4 leaf: light response")
    # (c) CO2 response at 200 W m-2 absorbed PAR
    ci = np.linspace(1e-6, 800e-6, 400); m = len(ci); o = np.ones(m)
    gs_c = np.maximum(ci, 1.05 * gs_)                                        # the model keeps C3 ci >= 1.05 gamma*
    a3, _ = farquhar_c3(jnp.asarray(200.0 * o), jnp.asarray(gs_c), jnp.asarray(V3 * o), gs_, kc, ko,
                        float(sp.o2conc), jnp.asarray(R3 * o), sp)
    a4, _ = farquhar_c4(jnp.asarray(200.0 * o), jnp.asarray(ci), jnp.asarray(V4 * o), jnp.asarray(R4 * o), sp)
    ax = axs[1, 0]
    ax.plot(ci * 1e6, np.asarray(a3) * u, color=BLUE, lw=1.6, label="C3 leaf")
    ax.plot(ci * 1e6, np.asarray(a4) * u, color=ORANGE, lw=1.6, label="C4 leaf")
    ax.axvline(CI3 * 1e6, color=REF, lw=0.8, ls=":"); ax.axvline(CI4 * 1e6, color=REF, lw=0.8, ls=":")
    ax.text(CI3 * 1e6 + 8, 2, "$c_i$ of (a)", fontsize=6.5, color=INK2)
    # to the left of its own line: on the right it ran up to the line of (a) and could be read as its label
    ax.text(CI4 * 1e6 - 8, 36, "$c_i$ of (b)", fontsize=6.5, color=INK2, ha="right")
    ax.set_xlim(0, 800); ax.set_ylim(0, 40)
    ax.set_xlabel("intercellular CO$_2$ $c_i$ (µmol mol$^{-1}$)"); ax.set_ylabel("µmol CO$_2$ m$^{-2}$ s$^{-1}$")
    ax.legend(loc="lower right", **LEG); _title(ax, "(c) $A_g$ against $c_i$, absorbed PAR 200 W m$^{-2}$")
    # (d) temperature factors; f4 dashed on top of f3, because the two coincide at warm temperatures
    tc = np.linspace(-10.0, 55.0, 651)
    _, _, _, _, t3, t4 = (np.asarray(x) for x in _temp_dep_params(jnp.asarray(tc + 273.16), sp))
    ax = axs[1, 1]
    ax.plot(tc, t3, color=BLUE, lw=2.0, label="C3, $f_3(T)$", zorder=2)
    ax.plot(tc, t4, color=ORANGE, lw=1.3, ls=(0, (4, 2)), label="C4, $f_4(T)$", zorder=3)
    ax.axvline(15.0, color=REF, lw=0.8, ls=":"); ax.text(15.6, 0.05, "15 °C", fontsize=6.5, color=INK2)
    ax.set_xlim(-10, 55); ax.set_ylim(0, 3.0)
    ax.set_xlabel("leaf temperature (°C)"); ax.set_ylabel("factor on $V_m$")
    ax.legend(loc="upper left", **LEG); _title(ax, "(d) temperature factor of capacity")
    for a in axs.ravel():
        _style(a)
    fig.savefig(OUT / "fig_9_1_leaf_model.png")
    plt.close(fig)
    i3 = int(np.argmax(t3)); i4 = int(np.argmax(t4))
    print(f"9.1 vmax15 C3 (T15) {v3*1e6:.2f}, C4 (G01) {v4*1e6:.2f} umol; at 25 C: f3 {f3:.3f} f4 {f4:.3f}; Vmax {V3*1e6:.2f} / {V4*1e6:.2f};"
          f" gamma* {gs_*1e6:.1f} umol/mol; Kc {kc*1e6:.0f} Ko {ko:.3f}; Rd {R3*1e6:.3f} / {R4*1e6:.3f};"
          f" Ag at 400 W: C3 {float(ag3[-1])*u:.2f} C4 {float(ag4[-1])*u:.2f}; Jc3 {jc3[0]*u:.2f} Js3 {js3[0]*u:.2f} Ji4 {ji4[0]*u:.2f}")
    print(f"    f3 max {t3[i3]:.3f} at {tc[i3]:.1f} C, f4 max {t4[i4]:.3f} at {tc[i4]:.1f} C; f3(15) {t3[np.argmin(abs(tc-15))]:.3f};"
          f" f3(0) {t3[np.argmin(abs(tc-0))]:.3f} f3(5) {t3[np.argmin(abs(tc-5))]:.3f} f4(10) {t4[np.argmin(abs(tc-10))]:.3f}")
    rel = np.abs(t3 - t4) / t3
    print(f"    f3 and f4 differ by less than 1 % above {tc[np.where(rel >= 0.01)[0].max() + 1]:.1f} C"
          f" (0.1 % above {tc[np.where(rel >= 0.001)[0].max() + 1]:.1f} C)")


# ---------------------------------------------------------------------------------------------------------------
# Figure 9.4: the canopy integration, IBIS 2.6b4 big leaf against the sunlit/shaded big leaves
# ---------------------------------------------------------------------------------------------------------------

def _generator_light(clouds, lat=43.25, jday=196, hour=12.0):
    """Direct and diffuse VIS radiation and cos(zenith) of the weather generator (`diurnal`) at one place and hour,
    for an array of cloud fractions."""
    import jax
    import jax.numpy as jnp
    from ibis.physics.weather import diurnal, DailyWeather
    n = len(clouds); f = lambda v: jnp.full((n,), v)
    dw = DailyWeather(td=f(295.0), tmax=f(300.0), tmin=f(290.0), cloud=jnp.asarray(clouds), precip=f(0.0),
                      psurf=f(98000.0), qd=f(0.010), ud=f(3.0), iwet=f(0.0), xstore=jnp.zeros((n, 3)))
    z = jnp.zeros((n,))
    _, h = diurnal(hour * 3600.0, jday, dw, jnp.full((n,), lat), z, z, z, jax.random.PRNGKey(0), 3600.0, 2)
    return np.asarray(h.solad[:, 0]), np.asarray(h.solai[:, 0]), np.asarray(h.coszen)


def _canopy(coszen, solad, solai, L, S, vmax15, sp, asur=0.1):
    """Gross photosynthesis of a broadleaf upper canopy per unit ground area of the canopy (mol m-2 s-1) for the
    F77 big leaf and the sunlit/shaded big leaves, at the chapter's fixed ci and leaf temperature, no water stress;
    and the canopy capacity in top-leaf units of both schemes."""
    import jax.numpy as jnp
    import ibis.physics.radiation as RD
    from ibis.physics.stomata import (farquhar_c3, canopy_scaling, sunshade_canopy, _sunshade_c3, _temp_dep_params)
    n = len(coszen)
    zc = jnp.asarray(coszen); L = jnp.asarray(L) * jnp.ones(n); S = jnp.asarray(S) * jnp.ones(n)
    zeros = jnp.zeros(n); t = jnp.full((n,), TLEAF); sunlit = zc > 0.0
    from ibis.io.physics_params import build_veg_rad_params
    from ibis.io.physics_params import load_physics_params
    vr = build_veg_rad_params(load_physics_params(str(PFILE)))
    rho, tau = float(vr.rhoveg[0, 1]), float(vr.tauveg[0, 1])               # upper canopy, VIS
    om, bd, bi, av, gd = RD.twoset(zc, L, S, zeros, zeros, t, rho, tau, float(vr.oriev[1]), float(vr.orieh[1]),
                                   RD._OMEGASNO[0], 273.16, 2, sunlit, fwet_stem=zeros, rliq_stem=zeros, temp_stem=t)
    a = jnp.full((n,), asur)
    ts = RD.twostr(zc, L, S, a, a, om, bd, bi, av, gd, 1.0e-7, sunlit, 0)
    sd = jnp.asarray(solad); si = jnp.asarray(solai)
    A = ts.term[:, 0] * sd; B = ts.term[:, 1] * sd + ts.term[:, 2] * si; C = ts.term[:, 3] * sd + ts.term[:, 4] * si
    scal = jnp.stack([A, B, C, A + B + C], -1); k, h = ts.term[:, 5], ts.term[:, 6]
    dbeam = (1.0 - ts.omega) * k * sd                                        # as radiation.solarf
    toppar = scal[:, 3] * L / jnp.maximum(L + S, 1e-7)                       # as radiation.solarf
    kc, ko, _, gs_, f3, _ = _temp_dep_params(t, sp)
    vm = vmax15 * f3; rd = float(sp.gamma[0]) * vmax15 * f3; ci = jnp.full((n,), CI3)
    scale, _, _ = canopy_scaling(toppar, k, h, scal, L, S, zeros, zeros, 3600.0, 1.0e-7)
    ag_top, _ = farquhar_c3(toppar, ci, vm, gs_, kc, ko, float(sp.o2conc), rd, sp)
    f77 = ag_top * scale * L
    ss = sunshade_canopy(scal, dbeam, k, h, L, S, float(sp.cap_kn), 1.0e-7)
    ag_ss, _ = _sunshade_c3(ss, ci, vm, gs_, kc, ko, float(sp.o2conc), rd, L, sp)
    apar_sun, apar_sha, l_sun, l_sha, v_sun, v_sha = ss
    ag_sun, _ = farquhar_c3(apar_sun, ci, vm * v_sun, gs_, kc, ko, float(sp.o2conc), rd * v_sun, sp)
    ag_sha, _ = farquhar_c3(apar_sha, ci, vm * v_sha, gs_, kc, ko, float(sp.o2conc), rd * v_sha, sp)
    extpar = jnp.clip((k * A + h * B - h * C) / jnp.maximum(scal[:, 3], 1e-7), 0.1, 10.0)
    cap_f77 = scale * L
    scale_lo, _, _ = canopy_scaling(toppar, k, h, scal, L, S, zeros, zeros, 3600.0, 1.0e-7, leaf_only=True)
    # radiation absorbed by the plant (leaves and stems), from the profile (9.16) integrated over L + S
    e1 = lambda a: (1.0 - jnp.exp(-a * (L + S))) / a
    apar_plant = jnp.maximum(A * e1(k) + B * e1(h) + C * e1(-h), 0.0)
    return dict(f77=np.asarray(f77), ss=np.asarray(ag_ss * L), sun=np.asarray(ag_sun), sha=np.asarray(ag_sha),
                cap_f77=np.asarray(cap_f77), cap_sun=np.asarray(v_sun), cap_sha=np.asarray(v_sha),
                l_sun=np.asarray(l_sun), l_sha=np.asarray(l_sha), k=np.asarray(k), extpar=np.asarray(extpar),
                apar=np.asarray(apar_sun + apar_sha), apar_sun=np.asarray(apar_sun), toppar=np.asarray(toppar),
                light_f77=np.asarray(toppar * scale * L), light_lo=np.asarray(toppar * scale_lo * L),
                cap_lo=np.asarray(scale_lo * L), apar_plant=np.asarray(apar_plant))


def fig_9_4():
    import jax
    jax.config.update("jax_enable_x64", True)
    pd, sp, over = _release()
    v15 = _zone_values("D_temperate", sp.vmax_pft, "vmax", over)[4]       # T15, the D_temperate broadleaf big leaf
    u = 1e6
    # (a), (c): leaf area from 0.25 to 8 under clear and overcast skies at noon, 15 July, 43.25 N
    Ls = np.linspace(0.25, 8.0, 32)
    clear = _generator_light(np.zeros(1)); cloudy = _generator_light(np.full(1, 0.8))
    rep = lambda x: np.repeat(x, len(Ls))
    cc = _canopy(rep(clear[2]), rep(clear[0]), rep(clear[1]), Ls, 0.25 * Ls, v15, sp)
    co = _canopy(rep(cloudy[2]), rep(cloudy[0]), rep(cloudy[1]), Ls, 0.25 * Ls, v15, sp)
    # (b): cloud fraction 1 -> 0 at L = 5
    cl = np.linspace(1.0, 0.0, 401)
    sd, si, mu = _generator_light(cl)
    cb = _canopy(mu, sd, si, 5.0 * np.ones_like(cl), 1.25 * np.ones_like(cl), v15, sp)
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={"wspace": 0.40})
    # the dotted (cloud fraction 0.8) curves are drawn above the solid ones, which they partly overlie
    ax = axs[0]
    ax.plot(Ls, cc["cap_sun"] + cc["cap_sha"], color=BLUE, lw=1.6, zorder=2)
    ax.plot(Ls, cc["cap_sun"], color=AQUA, lw=1.0, ls="--", zorder=3)
    ax.plot(Ls, cc["cap_f77"], color=ORANGE, lw=1.6, zorder=2)
    ax.plot(Ls, co["cap_f77"], color=ORANGE, lw=1.3, ls=":", zorder=4)
    ax.set_xlim(0, 8); ax.set_ylim(0, 5.0)
    ax.set_xlabel("leaf area index $L$"); ax.set_ylabel("capacity (m$^2$ of top leaf per m$^2$)")
    _title(ax, "(a) canopy capacity")
    ax = axs[1]
    x = sd + si
    ax.plot(x, cb["ss"] * u, color=BLUE, lw=1.6, zorder=2)
    ax.plot(x, cb["sun"] * u, color=AQUA, lw=1.0, ls="--", zorder=3)
    ax.plot(x, cb["sha"] * u, color=YELLOW, lw=1.0, ls="--", zorder=3)
    ax.plot(x, cb["f77"] * u, color=ORANGE, lw=1.6, zorder=2)
    ax.set_xlim(100, 450); ax.set_ylim(0, 35)                                 # the generator's overcast minimum is 142 W m-2
    ax.set_xlabel("incident visible radiation (W m$^{-2}$)"); ax.set_ylabel("µmol CO$_2$ m$^{-2}$ s$^{-1}$")
    _title(ax, "(b) $A_g$ against light, $L$ = 5")
    ax = axs[2]
    ax.plot(Ls, cc["ss"] * u, color=BLUE, lw=1.6, zorder=2)
    ax.plot(Ls, co["ss"] * u, color=BLUE, lw=1.3, ls=":", zorder=4)
    ax.plot(Ls, cc["f77"] * u, color=ORANGE, lw=1.6, zorder=2)
    ax.plot(Ls, co["f77"] * u, color=ORANGE, lw=1.3, ls=":", zorder=4)
    ax.set_xlim(0, 8); ax.set_ylim(0, 35)
    ax.set_xlabel("leaf area index $L$"); ax.set_ylabel("µmol CO$_2$ m$^{-2}$ s$^{-1}$")
    _title(ax, "(c) $A_g$ against leaf area")
    for a in axs:
        _style(a)
    handles = [Line2D([], [], color=BLUE, lw=1.6, label="sunlit + shaded leaves (IBIS 3.0)"),
               Line2D([], [], color=ORANGE, lw=1.6, label="big-leaf scaling (IBIS 2.6b4 scheme)"),
               Line2D([], [], color=AQUA, lw=1.0, ls="--", label="sunlit leaves only"),
               Line2D([], [], color=YELLOW, lw=1.0, ls="--", label="shaded leaves only (b)"),
               Line2D([], [], color=INK2, lw=1.6, label="clear sky (a, c)"),
               Line2D([], [], color=INK2, lw=1.3, ls=":", label="cloud fraction 0.8 (a, c)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
               **{**LEG, "borderaxespad": 0.0})
    fig.savefig(OUT / "fig_9_4_canopy_integration.png")
    plt.close(fig)
    i5 = int(np.argmin(abs(Ls - 5.0)))
    print(f"9.4 noon 15 July 43.25N: coszen {float(clear[2][0]):.3f}; clear VIS direct {float(clear[0][0]):.1f} diffuse {float(clear[1][0]):.1f};"
          f" cloud 0.8 direct {float(cloudy[0][0]):.1f} diffuse {float(cloudy[1][0]):.1f}; vmax15 {v15*1e6:.2f}")
    print(f"    L=5: k {cc['k'][i5]:.3f} extpar clear {cc['extpar'][i5]:.3f} cloudy {co['extpar'][i5]:.3f};"
          f" capacity F77 clear {cc['cap_f77'][i5]:.3f} cloudy {co['cap_f77'][i5]:.3f}; sunshade {cc['cap_sun'][i5]+cc['cap_sha'][i5]:.3f}"
          f" (sunlit {cc['cap_sun'][i5]:.3f}); sunlit LAI clear {cc['l_sun'][i5]:.3f}; sunlit APAR share {cc['apar_sun'][i5]/cc['apar'][i5]:.3f}")
    print(f"    L=5 clear: Ag F77 {cc['f77'][i5]*u:.2f} sunshade {cc['ss'][i5]*u:.2f}; cloud 0.8: F77 {co['f77'][i5]*u:.2f} sunshade {co['ss'][i5]*u:.2f}")
    print(f"    L=8 clear: F77 {cc['f77'][-1]*u:.2f} ss {cc['ss'][-1]*u:.2f}; capacity F77 {cc['cap_f77'][-1]:.3f} ss {cc['cap_sun'][-1]+cc['cap_sha'][-1]:.3f}")
    print(f"    (b) range of incident VIS {x.min():.1f}-{x.max():.1f}; Ag at overcast: F77 {cb['f77'][0]*u:.2f} ss {cb['ss'][0]*u:.2f};"
          f" clear F77 {cb['f77'][-1]*u:.2f} ss {cb['ss'][-1]*u:.2f}")
    print(f"    (b) incident diffuse VIS largest ({si.max():.1f}) at incident VIS {x[np.argmax(si)]:.1f} (cloud {cl[np.argmax(si)]:.3f});"
          f" sunlit+shaded largest {cb['ss'].max()*u:.2f} at {x[np.argmax(cb['ss'])]:.1f} (cloud {cl[np.argmax(cb['ss'])]:.3f});"
          f" shaded largest at {x[np.argmax(cb['sha'])]:.1f} (cloud {cl[np.argmax(cb['sha'])]:.3f});"
          f" F77 monotonic in light: {bool(np.all(np.diff(cb['f77']) >= 0))};"
          f" last five ss {np.round(cb['ss'][-5:]*u, 2)} at {np.round(x[-5:], 1)}")
    print(f"    (a) sunlit capacity at L=5 clear {cc['cap_sun'][i5]:.3f} cloud 0.8 {co['cap_sun'][i5]:.3f};"
          f" max |diff| over L {np.abs(cc['cap_sun'] - co['cap_sun']).max():.2e}; F77 cloud 0.8 at L=5 {co['cap_f77'][i5]:.3f}")
    print(f"    (c) L=5 cloud 0.8 ss {co['ss'][i5]*u:.2f} vs clear F77 {cc['f77'][i5]*u:.2f}; L=8 {co['ss'][-1]*u:.2f} vs {cc['f77'][-1]*u:.2f}")
    for nm, c in (("clear", cc), ("cloud 0.8", co)):
        print(f"    light at L=5, S=1.25, {nm}: plant absorbs {c['apar_plant'][i5]:.1f}, leaves {c['apar'][i5]:.1f} W m-2;"
              f" F77 light-equivalent P_top*Lambda*L {c['light_f77'][i5]:.1f}, with scale_leaf_only {c['light_lo'][i5]:.1f};"
              f" capacity F77 {c['cap_f77'][i5]:.3f}, scale_leaf_only {c['cap_lo'][i5]:.3f}")
    import jax.numpy as jnp
    from ibis.physics.stomata import _temp_dep_params
    f3_25 = float(np.asarray(_temp_dep_params(jnp.array([TLEAF]), sp)[4])[0])
    rd = float(sp.gamma[0]) * v15 * f3_25 * u                               # top leaf's dark respiration at 25 C
    print(f"    canopy dark respiration at L=5, 25 C: release {rd*(cc['cap_sun'][i5]+cc['cap_sha'][i5]):.2f},"
          f" F77 clear {rd*cc['cap_f77'][i5]:.2f} umol m-2 s-1; ratio {(cc['cap_sun'][i5]+cc['cap_sha'][i5])/cc['cap_f77'][i5]:.2f}")
    # equation (3.6) at the switch to a constant diffuse fraction
    t75 = 0.75
    print(f"    (3.6) cubic at tau 0.75: {1.0045 + 0.0435*t75 - 3.5227*t75**2 + 2.6313*t75**3:.5f} (constant 0.166);"
          f" cloud fraction at tau 0.75: {1.0 - (0.75 - 0.251) / 0.509:.4f}")


# ---------------------------------------------------------------------------------------------------------------
# Figure 9.3: the soil-water stress factor of one layer
# ---------------------------------------------------------------------------------------------------------------

def fig_9_3():
    import jax.numpy as jnp
    from ibis.physics.stomata import drystress
    from ibis.io import pft_types as pt
    pd, sp, over = _release()
    # the zones with types; E_ice and cells without a zone keep the base value
    zone_sf = {z: pt.zone_scalar(z, over["stressfac"], "all") for z in pt.ZONES if z not in ("none", "E_ice")}
    zmin = min(zone_sf, key=zone_sf.get); zmax = max(zone_sf, key=zone_sf.get)
    awc = np.linspace(0.0, 1.0, 201); n = len(awc)
    one = jnp.ones((n, 1)); froot = jnp.array([[1.0, 1.0]])
    def stress(sf):                                       # one layer holding every root: the layer factor itself
        _, su, _, _ = drystress(jnp.asarray(awc)[:, None], 0.0 * one, 0.0 * one, one, froot, stressfac=sf)
        return np.asarray(su)
    base = float(pd["stomata"]["stressfac"])
    mn = lambda v: f"{v:.2f}".replace("-", "−")         # true minus sign, as on the axes
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ax.plot(awc, stress(zone_sf[zmax]), color=BLUE, lw=1.6, label=f"{zmax}, {mn(zone_sf[zmax])}")
    ax.plot(awc, stress(zone_sf[zmin]), color=ORANGE, lw=1.6, label=f"{zmin}, {mn(zone_sf[zmin])}")
    ax.plot(awc, stress(base), color=AQUA, lw=1.6, label=f"base, {mn(base)} (E_ice, no zone)")
    ax.plot(awc, stress(-5.0), color=REF, lw=1.0, ls="--", label="IBIS 2.6b4, −5")
    ax.plot(awc, awc, color=REF, lw=0.8, ls=":")
    ax.text(0.80, 0.70, "$\\beta_k = a_k$", fontsize=6.5, color=INK2, ha="left", va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("available water of the layer $a_k$"); ax.set_ylabel("stress factor of the layer $\\beta_k$")
    ax.legend(loc="lower right", title="$s$ of the cell", title_fontsize=6.8, **LEG)
    _style(ax)
    fig.savefig(OUT / "fig_9_3_water_stress.png")
    plt.close(fig)
    print("9.3 stressfac per zone:", {z: round(v, 3) for z, v in zone_sf.items()}, "base", round(base, 3))
    for sf in (zone_sf[zmax], zone_sf[zmin], base, -5.0):
        b = stress(sf)
        print(f"    s {sf:.3f}: beta(0.1) {b[20]:.3f} beta(0.3) {b[60]:.3f} beta(0.5) {b[100]:.3f}")


# ---------------------------------------------------------------------------------------------------------------
# Figure 9.2: the state of acclimation and the capacity factor at two cells in 2001
# ---------------------------------------------------------------------------------------------------------------

CELLS = (("43.25°N, 89.25°W", 93, 181), ("61.75°N, 24.25°E", 56, 408))


def fig_9_2():
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.io.koppen import load_koppen, period_index
    from ibis.io import pft_types as pt
    pd, sp, over = _release()
    tau = float(sp.acclim_tau)
    rd = GSWP3Reader(GSWP3)
    I = np.array([c[1] for c in CELLS]); J = np.array([c[2] for c in CELLS])
    kop = load_koppen(KOPPEN, I, J)
    zi = pt.zones_from_classes(kop["classes"][period_index(2001, kop["period_start"], kop["period_end"])])
    tas = []
    for yr in (2000, 2001):
        for im in range(12):
            tas.append(rd.read_month(yr, im, I, J)["tas"])
    tas = np.concatenate(tas, 0) - 273.16                     # daily mean air temperature, C (ndays, 2)
    nd0 = 366                                                  # 2000 is a leap year: its days spin the state up
    S = np.zeros_like(tas); s = np.zeros(tas.shape[1])         # the model's cold-start value is 0 C
    for d in range(tas.shape[0]):
        s = s + (tas[d] - s) / tau                             # loops_gswp3.py: X <- X + (T - X) / tau, once a day
        S[d] = s
    T1, S1 = tas[nd0:], S[nd0:]; day = np.arange(1, T1.shape[0] + 1)
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.4), sharex=True, gridspec_kw={"hspace": 0.32, "wspace": 0.26})
    for j, (name, _, _) in enumerate(CELLS):
        zone = pt.ZONES[int(zi[j])]
        s0 = _zone_values(zone, sp.acclim_s0_pft, "acclim_s0", over); sm = _zone_values(zone, sp.acclim_smax_pft, "acclim_smax", over)
        slots = pt.slot_map(np.array([pt.ZI[zone]]))[0]
        lab = lambda p: (pt.IDS[slots[p]] if slots[p] >= 0 else "base")
        ax = axs[0, j]
        ax.plot(day, T1[:, j], color=REF, lw=0.7, label="daily mean air temperature $T_d$")
        ax.plot(day, S1[:, j], color=BLUE, lw=1.6, label="state of acclimation $X$")
        ax.set_ylim(-35, 35); ax.set_ylabel("°C")
        ax.legend(loc="lower center", **LEG); _title(ax, f"({'ab'[j]}) {name}, {zone}")
        ax = axs[1, j]
        print(f"9.2 {name} zone {zone}:")
        for p, col in ((3, ORANGE), (5, ORANGE), (8, AQUA)):
            if slots[p] == pt.OFF:
                continue
            f = np.clip((S1[:, j] - s0[p]) / sm[p], 0.0, 1.0)  # loops.py: f_S = clip((X - S0) / Smax, 0, 1)
            ax.plot(day, f, color=col, lw=1.6,
                    label=f"position {p + 1}, {lab(p)}")
            print(f"    position {p+1} {lab(p)} s0 {s0[p]:.3f} smax {sm[p]:.3f}: f_S=1 above S {s0[p]+sm[p]:.2f} C;"
                  f" days f_S=1 {int((f >= 1).sum())}, f_S=0 {int((f <= 0).sum())}; mean f_S {f.mean():.3f};"
                  f" Jun-Aug mean {f[151:243].mean():.3f}; Dec-Feb mean {np.r_[f[:59], f[334:]].mean():.3f}")
        print(f"    T mean {T1[:, j].mean():.2f}, S range {S1[:, j].min():.2f}..{S1[:, j].max():.2f}")
        ax.set_ylim(0, 1.05); ax.set_ylabel("capacity factor $f_S$")
        ax.legend(loc="lower center", **LEG); _title(ax, f"({'cd'[j]}) acclimated positions")
        ax.set_xlabel("day of 2001")
    for a in axs.ravel():
        a.set_xlim(1, 365); a.set_xticks([1, 60, 121, 182, 244, 305, 365]); _style(a)
    fig.savefig(OUT / "fig_9_2_acclimation.png")
    plt.close(fig)


FIGURES = {"9.1": fig_9_1, "9.2": fig_9_2, "9.3": fig_9_3, "9.4": fig_9_4}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
        print("figure", key, "done")
