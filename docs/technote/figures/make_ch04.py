"""Figures of Chapter 4 (canopy radiative transfer) of the IBIS 3.0 technical note, drawn from the model's code
(`ibis/physics/radiation.py`) and the release's input and output files.

Run from the repository root with the project interpreter:

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch04.py            # all figures
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch04.py 4.2 4.4    # some
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch04.py checks     # numbers of the text

Every curve is computed by the model's own functions (twoset, twostr, solsur, compute_radiation) in double precision;
Figure 4.4 drives compute_radiation with the hourly radiation that the model's weather generator makes from one day of
GSWP3 forcing and with the vegetation and soil water of the release run, each step evaluated at the time of day the
model's `step_time` gives (the middle of the step under the default `weather.sun_midstep`; technote review D33). Nothing
is typed in by hand except the labels and the schematic's drawing. The numbers quoted in the chapter are printed.

Code shown: the two-stream window of section 4.4 as fixed after the release (K perturbed with p; review D26, commit
8416c20), the mid-step sun (D33, 8416c20) and the solar-time tower forcing (D34, 63fd7c0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: F401  (style, rcParams)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

import ibis.physics.radiation as RD                                   # noqa: E402
from types import SimpleNamespace                                     # noqa: E402
from ibis.timestepping.loops import SimParams, step_time              # noqa: E402

#: the model's default timing of a step's sun and generated weather (weather.sun_midstep; True = the middle of the step)
SUN = SimpleNamespace(sun_midstep=SimParams.__dataclass_fields__["sun_midstep"].default)


def step_hours(dt=3600.0):
    """The time of day (h) at which the model evaluates the sun of each of the day's steps (step_time)."""
    return np.array([float(step_time(k, dt, SUN)) for k in range(int(round(86400.0 / dt)))]) / 3600.0

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#6250d6", "#e34948")
TMELT = 273.16
VP = RD.default_veg_rad_params()                                      # = the veg_rad section of the release file
REL = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30"
GSWP3 = "/shared/nas2/Min/GSWP3"
LOAM_SAND, LOAM_CLAY = 40.0, 20.0                                      # the fill texture of the soil file (Fig. 2.4)


def _style(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _title(ax, s):
    ax.set_title(s, loc="left", fontsize=8.5, weight="bold", color=INK)


def _albsat(sand, clay):
    """Saturated soil albedo (visible, near-infrared) of the model, equation (2.10)."""
    fs, fc = 0.01 * np.floor(sand + 0.5), 0.01 * np.floor(clay + 0.5)
    v = 0.120 * fs + 0.085 * (1.0 - fs - fc) + 0.050 * fc
    return v, 2.0 * v


def one_layer(mu, lai, sai, iv, ib, asd, asi, tveg=290.0, fwet=0.0, rliq=0.0):
    """twoset + twostr of the model for one canopy (iv 1 lower, 2 upper) and one band, vectorized over the inputs."""
    mu, lai, sai, asd, asi, tveg, fwet, rliq = np.broadcast_arrays(*(np.asarray(x, float) for x in
                                                                      (mu, lai, sai, asd, asi, tveg, fwet, rliq)))
    z = jnp.asarray(mu.ravel()); L = jnp.asarray(lai.ravel()); S = jnp.asarray(sai.ravel())
    fw = jnp.asarray(fwet.ravel()); rl = jnp.asarray(rliq.ravel()); t = jnp.asarray(tveg.ravel())
    st = iv - 1
    kw = dict(fwet_stem=fw, rliq_stem=rl, temp_stem=t) if iv == 2 else {}
    om, bd, bi, av, gd = RD.twoset(z, L, S, fw, rl, t, float(VP.rhoveg[ib, st]), float(VP.tauveg[ib, st]),
                                   float(VP.oriev[st]), float(VP.orieh[st]), float(RD._OMEGASNO[ib]), TMELT, iv, z > 0,
                                   **kw)
    r = RD.twostr(z, L, S, jnp.asarray(asd.ravel()), jnp.asarray(asi.ravel()), om, bd, bi, av, gd, 1.0e-7, z > 0, ib)
    sh = mu.shape
    out = {k: np.asarray(getattr(r, k)).reshape(sh) for k in ("abvegd", "abvegi", "refld", "refli", "fbeldd", "fbelid",
                                                             "fbelii")}
    out["term"] = np.asarray(r.term).reshape(sh + (7,))
    out["omega"] = np.asarray(om).reshape(sh)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Figure 4.1: the radiative column (schematic)
# ---------------------------------------------------------------------------------------------------------------

def fig_4_1():
    fig, ax = plt.subplots(figsize=(8.4, 5.55))
    ax.set_xlim(-3.5, 112); ax.set_ylim(-10.5, 66); ax.axis("off")
    C_UP, C_LO, C_SOIL, C_SNOW = "#5d9a45", "#a9cf6a", "#c9a77c", "#f4f8fc"
    # ---- (a) solar -------------------------------------------------------------------------------------------
    ax.text(0, 65.5, "(a) solar radiation, each waveband", fontsize=9, weight="bold", color=INK, va="top")
    # cell; edge of the upper canopy; snow from xs (straddling the canopy edge: the ground system is the same beneath
    # the canopy and its gap, positions are schematic); lower canopy to xl
    x0, x1, xu, xs, xl = 3.0, 51.0, 35.0, 28.0, 20.0
    ax.add_patch(Rectangle((x0, 33), xu - x0, 8, fc=C_UP, ec="none", alpha=0.9))
    ax.text((x0 + xu) / 2, 37, "upper canopy  $L_u + S_u$", ha="center", va="center", fontsize=7.3, color="white",
            weight="bold")
    ax.text((xu + x1) / 2, 37, "gap", ha="center", va="center", fontsize=7.2, color=INK2, style="italic")
    ax.annotate("", (x0, 31.6), (xu, 31.6), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
    ax.text((x0 + xu) / 2, 30.4, "$f_u$", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.annotate("", (xu, 31.6), (x1, 31.6), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
    ax.text((xu + x1) / 2, 30.4, "$1 - f_u$", ha="center", va="top", fontsize=7.5, color=INK2)
    # ground system: lower canopy over soil on f_l of the snow-free part, bare soil, snow on f_i
    ax.add_patch(Rectangle((x0, 9), x1 - x0, 6, fc=C_SOIL, ec="none"))
    ax.add_patch(Rectangle((x0, 15), xl - x0, 5, fc=C_LO, ec="none"))
    ax.add_patch(Rectangle((xs, 15), x1 - xs, 2.2, fc=C_SNOW, ec="#b9c7d6", lw=0.5))
    ax.text((x0 + xl) / 2, 17.5, "lower canopy\n$L_l + S_l$", ha="center", va="center", fontsize=7.0, color=INK)
    ax.text((xl + xs) / 2, 12, "bare soil", ha="center", va="center", fontsize=7.0, color=INK)
    ax.text((xs + x1) / 2, 16.1, "snow", ha="center", va="center", fontsize=7.0, color=INK2)
    ax.text((x0 + xl) / 2, 12, "soil", ha="center", va="center", fontsize=7.0, color=INK)
    ax.annotate("", (x0, 7.4), (xs, 7.4), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
    ax.text((x0 + xs) / 2, 6.3, "$1 - f_i$", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.annotate("", (xs, 7.4), (x1, 7.4), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
    ax.text((xs + x1) / 2, 6.3, "$f_i$", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.annotate("", (x0, 21.6), (xl, 21.6), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
    ax.text((x0 + xl) / 2, 22.4, "$f_l$ of the snow-free area", ha="center", va="bottom", fontsize=7.0, color=INK2)
    # incident and reflected radiation
    for xa in (10.0, 22.0):
        ax.add_patch(FancyArrowPatch((xa - 4.5, 58), (xa, 42), arrowstyle="-|>", mutation_scale=7, color=YELLOW, lw=1.2))
    ax.text(5.0, 59.5, "direct $S^{\\mu}_{\\Lambda}$", fontsize=7.2, color=INK, va="bottom")
    for xa in (28.0, 33.0):
        ax.add_patch(FancyArrowPatch((xa, 58), (xa, 42), arrowstyle="-|>", mutation_scale=7, color=BLUE, lw=0.9,
                                     ls=(0, (3, 2))))
    ax.text(26.5, 59.5, "diffuse $S_{\\Lambda}$", fontsize=7.2, color=INK, va="bottom")
    ax.add_patch(FancyArrowPatch((43.5, 42), (43.5, 58), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.9))
    ax.text(45.0, 59.5, "reflected", fontsize=7.2, color=INK, va="bottom", ha="center")
    # order of the calculation (numbers in circles); 6 is the bracket over the whole column
    steps = [(1, 44.0, 12.0, "albedos of the soil and the snow, (4.14)-(4.15)"),
             (2, xl + 3.4, 17.6, "two-stream solution of the lower canopy over the soil, (4.1)-(4.13)"),
             (3, 27.5, 25.5, "area-mean albedos of the ground system, (4.16)"),
             (4, 19.0, 44.0, "two-stream solution of the upper canopy over the ground system, (4.1)-(4.13) with (4.16)"),
             (5, 43.5, 47.0, "albedos of the cell, (4.17)"),
             (6, -1.6, 25.0, "absorbed fluxes of all components (bracket), (4.18)-(4.22)")]
    ax.plot([-1.6, -1.6], [9.0, 41.0], color=INK2, lw=0.7)
    for yb in (9.0, 41.0):
        ax.plot([-1.6, 0.6], [yb, yb], color=INK2, lw=0.7)
    for k, xc, yc, _ in steps:
        ax.add_patch(Circle((xc, yc), 1.5, fc="white", ec=INK, lw=0.7, zorder=5))
        ax.text(xc, yc, str(k), ha="center", va="center", fontsize=6.8, color=INK, zorder=6, weight="bold")
    lines = [f"{k}  {s}" for k, _, _, s in steps]
    for n, line in enumerate(lines):
        ax.text(x0, 2.6 - 1.75 * n, line, fontsize=6.4, color=INK2, va="top")
    ax.text(x0, 2.6 - 1.75 * len(lines) - 0.5, "positions are schematic: the ground system is the same beneath the "
            "upper canopy and its gap", fontsize=6.4, color=INK2, va="top", style="italic")
    # ---- (b) longwave ----------------------------------------------------------------------------------------
    bx0, bx1 = 64.0, 92.0
    ax.text(bx0 - 4, 65.5, "(b) longwave radiation", fontsize=9, weight="bold", color=INK, va="top")
    planes = [(45.0, C_UP, "upper leaves", "$\\varepsilon_u = 1 - e^{-L_u}$", bx0, bx0 + 26.0),
              (39.0, "#7a5a3c", "upper stems", "$\\varepsilon_s = 1 - e^{-S_u}$", bx0, bx0 + 26.0),
              (24.0, C_LO, "lower canopy", "$\\varepsilon_l = 1 - e^{-(L_l + S_l)}$", bx0, bx0 + 10.0),
              (12.0, C_SOIL, "soil", "$\\varepsilon_g = 0.95$", bx0, bx0 + 21.0)]
    for y, c, name, em, xa, xb in planes:
        ax.add_patch(Rectangle((xa, y - 0.8), xb - xa, 1.6, fc=c, ec="none"))
        ax.text(bx1 + 1.0, y + 0.5, name, fontsize=7.2, color=INK, va="bottom")
        ax.text(bx1 + 1.0, y + 0.3, em, fontsize=7.2, color=INK2, va="top")
    ax.add_patch(Rectangle((bx0 + 21.0, 12.0 - 0.8), bx1 - bx0 - 21.0, 1.6, fc=C_SNOW, ec="#b9c7d6", lw=0.5))
    ax.text(bx1 + 1.0, 7.6, "snow", fontsize=7.2, color=INK, va="bottom")
    ax.text(bx1 + 1.0, 7.4, "$\\varepsilon_i = 0.95$", fontsize=7.2, color=INK2, va="top")
    ax.add_patch(FancyArrowPatch((bx1 + 0.6, 7.7), (bx1 - 2.0, 10.9), arrowstyle="-", color=INK2, lw=0.6))
    # through the lower canopy (left), onto the bare soil and the snow directly (right)
    arrows = [((bx0 + 2, 58), (bx0 + 2, 47), BLUE, "$L^{\\downarrow}$", 47.0 + 5.5),
              ((bx0 + 7, 47), (bx0 + 7, 58), INK2, "$F_{b}$", 47.0 + 5.5),
              ((bx0 + 2, 37), (bx0 + 2, 26), BLUE, "$F_{d}$", 31.5),
              ((bx0 + 7, 26), (bx0 + 7, 37), INK2, "$F_{u}$", 31.5),
              ((bx0 + 2, 22), (bx0 + 2, 13.5), BLUE, "$F_{dg}$", 17.75),
              ((bx0 + 7, 13.5), (bx0 + 7, 22), INK2, "$F_{ug}$", 17.75),
              ((bx0 + 13, 37), (bx0 + 13, 13.5), BLUE, "$F_{d}$", 30.0),
              ((bx0 + 17, 13.5), (bx0 + 17, 37), INK2, "$F_{ugb}$", 30.0),
              ((bx0 + 24, 13.5), (bx0 + 24, 37), INK2, "$F_{ui}$", 30.0)]
    for (p, q, c, lab, ylab) in arrows:
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=c, lw=0.9))
        ax.text(p[0] + 0.8, ylab, lab, fontsize=7.2, color=INK, va="center", ha="left")
    ax.text(bx0 - 4, 2.6, "each plane absorbs the fraction $\\varepsilon$ of the radiation that reaches it",
            fontsize=6.4, color=INK2, va="top")
    ax.text(bx0 - 4, 2.6 - 1.75, "and emits $\\varepsilon\\sigma_{SB} T^4$ from each side, (4.27)-(4.31)", fontsize=6.4,
            color=INK2, va="top")
    fig.savefig(OUT / "fig_4_1_radiation_column.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 4.2: the two-stream solution for one canopy layer
# ---------------------------------------------------------------------------------------------------------------

def fig_4_2():
    av, an = _albsat(LOAM_SAND, LOAM_CLAY)                         # saturated loam: 0.092, 0.184
    MU = 0.25                                                      # the beam of panels (a), (b), (d)
    X = np.linspace(0.0, 10.0, 201); L, S = X * 5.0 / 6.0, X / 6.0  # plant area with the stems at 1/6, cf. Fig. 4.4
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 6.2), gridspec_kw={"hspace": 0.42, "wspace": 0.3})
    leg = dict(frameon=False, fontsize=6.6, handlelength=2.0, borderaxespad=0.3)
    # (a) albedo of the layer + soil
    ax = axs[0, 0]
    for ib, a, c, band in ((0, av, BLUE, "visible"), (1, an, ORANGE, "near-infrared")):
        r = one_layer(MU, L, S, 2, ib, a, a)
        ax.plot(X, r["refld"], color=c, lw=1.6, label=f"{band}, direct ($\\mu$ = {MU})")
        ax.plot(X, r["refli"], color=c, lw=1.2, ls=(0, (4, 2)), label=f"{band}, diffuse")
        if ib == 0:
            print(f"4.2a VIS albedo at X=6: direct {np.interp(6, X, r['refld']):.3f} diffuse {np.interp(6, X, r['refli']):.3f}")
        else:
            print(f"4.2a NIR albedo at X=6: direct {np.interp(6, X, r['refld']):.3f} diffuse {np.interp(6, X, r['refli']):.3f}")
    ax.set_xlim(0, 10); ax.set_ylim(0, 0.42)
    ax.set_xlabel("plant area index $L + S$"); ax.set_ylabel("albedo")
    ax.legend(loc="upper right", ncol=1, **leg)
    _title(ax, "(a) albedo of the layer over a loam")
    # (b) fate of a unit direct visible beam
    ax = axs[0, 1]
    r = one_layer(MU, L, S, 2, 0, av, av)
    ground = (1 - av) * r["fbeldd"] + (1 - av) * r["fbelid"]
    ax.plot(X, r["abvegd"], color=BLUE, lw=1.6, label="absorbed by the layer")
    ax.plot(X, ground, color=ORANGE, lw=1.6, label="absorbed by the soil")
    ax.plot(X, r["refld"], color=AQUA, lw=1.6, label="reflected")
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.0)
    ax.set_xlabel("plant area index $L + S$"); ax.set_ylabel("fraction of the incident beam")
    ax.legend(loc="center right", **leg)
    _title(ax, f"(b) a visible beam, $\\mu$ = {MU}")
    print(f"4.2b VIS direct at X=6: layer {np.interp(6, X, r['abvegd']):.3f} soil {np.interp(6, X, ground):.3f}")
    # (c) the extinction coefficients
    ax = axs[1, 0]
    mu = np.linspace(0.05, 1.0, 200)
    ku = one_layer(mu, 5.0, 1.0, 2, 0, av, av)["term"][..., 5]
    kl = one_layer(mu, 5.0, 1.0, 1, 0, av, av)["term"][..., 5]
    hu = one_layer(mu, 5.0, 1.0, 2, 0, av, av)["term"][..., 6]
    hl = one_layer(mu, 5.0, 1.0, 1, 0, av, av)["term"][..., 6]
    ax.plot(mu, ku, color=BLUE, lw=1.6, label="$K$, upper canopy (random leaves)")
    ax.plot(mu, kl, color=ORANGE, lw=1.6, label="$K$, lower canopy (half vertical)")
    ax.plot(mu, hu, color=BLUE, lw=1.2, ls=(0, (4, 2)), label=f"$h$, upper canopy, visible ({hu[0]:.3f})")
    ax.plot(mu, hl, color=ORANGE, lw=1.2, ls=(0, (4, 2)), label=f"$h$, lower canopy, visible ({hl[0]:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 4.0)
    ax.set_xlabel("cosine of the solar zenith angle $\\mu$"); ax.set_ylabel("$K$ and $h$, per unit plant area")
    ax.legend(loc="upper right", **leg)
    _title(ax, "(c) beam extinction $K$, diffuse exponent $h$")
    print(f"4.2c K upper at mu=0.5 {np.interp(0.5, mu, ku):.3f}, lower {np.interp(0.5, mu, kl):.3f}; h upper vis {hu[0]:.4f} lower vis {hl[0]:.4f}")
    # (d) absorbed visible per unit plant area through a layer of plant area 6
    ax = axs[1, 1]
    Xc = 6.0; x = np.linspace(0, Xc, 301)
    r = one_layer(MU, 5.0, 1.0, 2, 0, av, av); t = r["term"]
    k, h = t[5], t[6]
    direct = t[0] * np.exp(-k * x) + t[1] * np.exp(-h * x) + t[3] * np.exp(h * x)
    beam = (1.0 - r["omega"]) * k * np.exp(-k * x)
    diffuse = t[2] * np.exp(-h * x) + t[4] * np.exp(h * x)
    ax.plot(direct, x, color=BLUE, lw=1.6, label=f"direct incidence ($\\mu$ = {MU})")
    ax.plot(beam, x, color=BLUE, lw=1.0, ls=(0, (1.5, 1.5)), label="  of which the unscattered beam")
    ax.plot(diffuse, x, color=ORANGE, lw=1.6, label="diffuse incidence")
    ax.set_ylim(Xc, 0); ax.set_xlim(0, 2.0)
    ax.set_ylabel("plant area above, $x$"); ax.set_xlabel("absorbed per unit plant area,\nper unit incident flux")
    ax.legend(loc="lower right", **leg)
    _title(ax, "(d) visible absorption profile, $L + S$ = 6")
    print(f"4.2d integral direct {np.trapezoid(direct, x):.4f} vs abvegd {float(r['abvegd']):.4f}; diffuse {np.trapezoid(diffuse, x):.4f} vs abvegi {float(r['abvegi']):.4f}")
    print(f"4.2d K {k:.3f} h {h:.3f}; top: direct {direct[0]:.3f} of which beam {beam[0]:.3f}; diffuse {diffuse[0]:.3f}")
    for a in axs.ravel():
        _style(a)
    fig.savefig(OUT / "fig_4_2_two_stream.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 4.3: albedos of soil and snow; intercepted snow on the canopy
# ---------------------------------------------------------------------------------------------------------------

def fig_4_3():
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.9), gridspec_kw={"wspace": 0.38})
    leg = dict(frameon=False, fontsize=6.6, handlelength=2.0, borderaxespad=0.3)
    av, an = _albsat(LOAM_SAND, LOAM_CLAY)
    # (a) soil albedo against the liquid water of the top layer
    ax = axs[0]
    zw = np.linspace(0.0, 1.0, 201); n = zw.size; one = jnp.ones(n)
    for ib, c, band in ((0, BLUE, "visible"), (1, ORANGE, "near-infrared")):
        a = RD.solsur(0.5 * one, jnp.asarray(zw), 0 * one, TMELT * one, av * one, an * one, TMELT, one > 0, ib)[0]
        ax.plot(zw, np.asarray(a), color=c, lw=1.6, label=band)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.42)
    ax.set_xlabel("liquid water of the top layer,\nfraction of the pore space"); ax.set_ylabel("albedo")
    ax.legend(loc="upper right", **leg)
    _title(ax, "(a) soil (loam)")
    # (b) snow albedo against the temperature of the top snow layer
    ax = axs[1]
    T = np.linspace(-20.0, 0.0, 201); n = T.size; one = jnp.ones(n)
    for ib, c, band in ((0, BLUE, "visible"), (1, ORANGE, "near-infrared")):
        for mu, ls, lw, lab in ((0.2, "-", 1.6, "direct, $\\mu$ = 0.2"), (0.6, (0, (4, 2)), 1.2, "diffuse")):
            _, _, snd, sni = RD.solsur(mu * one, 0.3 * one, 0 * one, jnp.asarray(T + TMELT), av * one, an * one, TMELT,
                                       one > 0, ib)
            y = snd if mu < 0.5 else sni
            ax.plot(T, np.asarray(y), color=c, lw=lw, ls=ls, label=f"{band}, {lab}")
    ax.set_xlim(-20, 0); ax.set_ylim(0.1, 1.0)
    ax.set_xlabel("$T_{sno} - T_m$ (K)"); ax.set_ylabel("albedo")
    ax.legend(loc="lower left", **leg)
    _title(ax, "(b) snow")
    # (c) the factor on the single-scattering albedo of intercepted snow
    ax = axs[2]
    T = np.linspace(-1.5, 0.5, 2001)
    ou = one_layer(0.5, 4.0, 1.0, 2, 0, 0.1, 0.1, tveg=T + TMELT, fwet=1.0, rliq=0.0)["omega"]
    ol = one_layer(0.5, 4.0, 1.0, 1, 0, 0.1, 0.1, tveg=T + TMELT, fwet=1.0, rliq=0.0)["omega"]
    w = float(RD._OMEGASNO[0])
    ax.plot(T, ou / w, color=BLUE, lw=1.6, label="upper canopy (leaves and stems)")
    ax.plot(T, ol / w, color=ORANGE, lw=1.6, label="lower canopy")
    ax.set_xlim(-1.5, 0.5); ax.set_ylim(0.5, 1.05)
    ax.set_xlabel("canopy temperature $T - T_m$ (K)"); ax.set_ylabel("factor $\\phi$ on $\\omega_{sno}$")
    ax.legend(loc="lower left", **leg)
    _title(ax, "(c) snow held on the canopy")
    i = np.argmin(np.abs(T + 0.5)); print(f"4.3c factor at -0.5 C: upper {ou[i]/w:.3f} lower {ol[i]/w:.3f}")
    for a in axs:
        _style(a)
    fig.savefig(OUT / "fig_4_3_ground_snow_albedo.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 4.4: one day at one forest cell with the release run's vegetation
# ---------------------------------------------------------------------------------------------------------------

def _release_state(I, J):
    """Leaf and stem area, cover, saturated soil albedo (restart at the end of 2000, the annual values of 2001), and
    the leaf area and top-layer soil water and ice of July 2001 (monthly output; leaf area at the end of the month,
    soil water the month's mean)."""
    import xarray as xr
    from ibis.dimensions import GridConfig
    from ibis.state.model_state import create_ibis_state
    land = _land_mask()
    p = int(np.where((np.argwhere(land)[:, 0] == I) & (np.argwhere(land)[:, 1] == J))[0][0])
    tmpl = create_ibis_state(GridConfig(npoi=2, nbedrock=3))
    paths = [jax.tree_util.keystr(q) for q, _ in jax.tree_util.tree_flatten_with_path(tmpl)[0]]
    z = np.load(f"{REL}/hindcast/restart/ibis_restart_2000.npz")
    leaf = lambda name: z[f"leaf_{paths.index(name)}"][p]
    st = {k: np.asarray(leaf(f".{k}")) for k in ("vegetation.sai", "vegetation.fu", "vegetation.fl", "soil.albsav",
                                                  "soil.albsan", "vegetation.plai")}
    ds = xr.open_dataset(f"{REL}/hindcast/out/ibis_monthly_2001_07.nc")
    st["lai"] = np.array([float(ds.lail.values[I, J]), float(ds.laiu.values[I, J])])
    st["wsoi1"] = float(ds.wsoi_mean.values[I, J, 0]); st["wisoi1"] = float(ds.wisoi_mean.values[I, J, 0])
    assert abs(float(ds.fu.values[I, J]) - float(st["vegetation.fu"])) < 1e-6        # cover is annual
    return st


def fig_4_4():
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.io.gswp3_forcing import gswp3_day_to_daily_weather
    from ibis.physics.weather import diurnal, compute_precip_timing
    from ibis.timestepping.calendar import julian_day, is_leap_year
    I, J = 94, 215                                                   # 42.75 N, 72.25 W (Harvard Forest)
    year, month, day = 2001, 6, 21                                   # 22 July 2001 (month and day 0-based)
    lat = 89.75 - 0.5 * I; lon = -179.75 + 0.5 * J
    st = _release_state(I, J)
    mo = GSWP3Reader(GSWP3).read_month(year, month, np.array([I]), np.array([J]))
    jd = julian_day(month, day, is_leap_year(year))
    dw, rlds = gswp3_day_to_daily_weather(mo, day, np.deg2rad([lat]).astype(np.float32), jd, np.zeros((1, 3), np.float32))
    dt = 3600.0
    key = jax.random.PRNGKey(7)
    key, plens, startp, endp = compute_precip_timing(key, jnp.asarray(dw.precip), dt)
    hs = []; hs_start = []
    for istep in range(24):
        key, sub = jax.random.split(key)
        # the step's sun at the time of day the model uses (step_time: the middle of the step by default, D33)
        _, h = diurnal(step_time(istep, dt, SUN), jd, dw, jnp.array([lat], jnp.float32), plens, startp, endp, sub, dt, 2)
        hs.append(h)
        # for the text only: the same step evaluated at its start, as the release code did
        _, h0 = diurnal(istep * dt, jd, dw, jnp.array([lat], jnp.float32), plens, startp, endp, sub, dt, 2)
        hs_start.append(h0)
    inc_start = sum(float(np.sum(h0.solad[0]) + np.sum(h0.solai[0])) for h0 in hs_start) * dt / 1e6
    print(f"4.4 step times (h): {step_hours(dt)[:3]} ...; incident with the sun at the step start (release code) "
          f"{inc_start:.2f} MJ m-2 d-1; the forcing's daily mean shortwave {float(mo['rsds'][day, 0]):.1f} W m-2 = "
          f"{float(mo['rsds'][day, 0]) * 86400.0 / 1e6:.2f} MJ m-2 d-1")
    cz = jnp.asarray(np.array([float(h.coszen[0]) for h in hs]), jnp.float64)
    sd = jnp.asarray(np.array([np.asarray(h.solad[0]) for h in hs]), jnp.float64)
    si = jnp.asarray(np.array([np.asarray(h.solai[0]) for h in hs]), jnp.float64)
    ta = jnp.asarray(np.array([float(h.ta[0]) for h in hs]), jnp.float64)
    n = 24; one = jnp.ones(n); zero = jnp.zeros(n)
    lai = jnp.asarray(np.tile(st["lai"], (n, 1))); sai = jnp.asarray(np.tile(st["vegetation.sai"], (n, 1)))
    fu = float(st["vegetation.fu"]) * one; fl = float(st["vegetation.fl"]) * one
    r = RD.compute_radiation(cz, sd, si, float(rlds[0]) * one, lai, sai, fu, fl, ta, ta, ta, ta, ta, zero, TMELT * one,
                             st["wsoi1"] * one, st["wisoi1"] * one, float(st["soil.albsav"]) * one,
                             float(st["soil.albsan"]) * one, zero, zero, zero, one, one, one, VP, 5.67e-8, TMELT, 1e-7)
    g = lambda a: np.asarray(a)
    inc = g(sd.sum(1) + si.sum(1))
    parts = [("upper-canopy leaves", g(fu * r.solu), BLUE), ("upper-canopy stems", g(fu * r.sols), ORANGE),
             ("lower canopy", g(fl * r.soll), AQUA), ("soil", g(r.solg), YELLOW),
             ("reflected", g((r.asurd * sd).sum(1) + (r.asuri * si).sum(1)), MAGENTA)]
    assert np.allclose(sum(p[1] for p in parts), inc, atol=1e-9)
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), gridspec_kw={"wspace": 0.3, "width_ratios": [1.25, 1]})
    leg = dict(frameon=False, fontsize=6.6, handlelength=1.4, borderaxespad=0.3)
    ax = axs[0]
    t = np.arange(24)
    bottom = np.zeros(24)
    for name, v, c in parts:
        ax.bar(t + 0.5, v, width=0.86, bottom=bottom, color=c, label=name, lw=0)
        bottom += v
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6)); ax.set_ylim(0, max(inc) * 1.32)
    ax.set_xlabel("local solar time (h)"); ax.set_ylabel("W m$^{-2}$ of the cell")
    ax.legend(loc="upper left", ncol=2, **leg)
    _title(ax, "(a) incident shortwave and where it goes")
    ax = axs[1]
    day_ = inc > 0
    tot = parts[-1][1][day_] / inc[day_]
    vis = g((r.asurd[:, 0] * sd[:, 0] + r.asuri[:, 0] * si[:, 0]))[day_] / g(sd[:, 0] + si[:, 0])[day_]
    nir = g((r.asurd[:, 1] * sd[:, 1] + r.asuri[:, 1] * si[:, 1]))[day_] / g(sd[:, 1] + si[:, 1])[day_]
    th = step_hours(dt)[day_]                # the solar time at which mu and the albedos of the step are evaluated
    ax.plot(th, nir, color=ORANGE, lw=1.6, marker="o", ms=3, label="near-infrared")
    ax.plot(th, tot, color=INK2, lw=1.6, marker="o", ms=3, label="all shortwave")
    ax.plot(th, vis, color=BLUE, lw=1.6, marker="o", ms=3, label="visible")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6)); ax.set_ylim(0, 0.05 * np.ceil(nir.max() * 1.3 / 0.05))
    ax.set_xlabel("local solar time (h)"); ax.set_ylabel("albedo")
    ax.legend(loc="upper center", ncol=3, columnspacing=0.8, **{**leg, "handlelength": 1.0})
    print(f"4.4b albedo range: NIR {nir.min():.3f}-{nir.max():.3f}, VIS {vis.min():.3f}-{vis.max():.3f}, all {tot.min():.3f}-{tot.max():.3f}")
    _title(ax, "(b) albedo of the cell")
    for a in axs:
        _style(a)
    fig.savefig(OUT / "fig_4_4_forest_day.png")
    plt.close(fig)
    E = {name: v.sum() * dt / 1e6 for name, v, _ in parts}; Ein = inc.sum() * dt / 1e6
    print(f"4.4 cell {lat} {lon} day {jd}: lai {st['lai']}, sai {st['vegetation.sai']}, fu {float(st['vegetation.fu']):.3f}"
          f" fl {float(st['vegetation.fl']):.3f}, albsav {float(st['soil.albsav']):.4f} wsoi1 {st['wsoi1']:.3f} wisoi1 {st['wisoi1']:.3f}")
    print(f"4.4 incident {Ein:.2f} MJ m-2 d-1; " + "; ".join(f"{k} {v:.3f} ({100 * v / Ein:.2f} %)" for k, v in E.items()))
    import xarray as xr
    lu = [float(xr.open_dataset(f"{REL}/hindcast/out/ibis_monthly_{year}_{m:02d}.nc").laiu.values[I, J]) for m in range(1, 13)]
    print(f"4.4 upper-canopy leaf area at the end of each month of {year}: {min(lu):.3f}-{max(lu):.3f}; leaf area by "
          f"position (restart 2000): {np.round(st['vegetation.plai'], 3)}")
    up = E["upper-canopy leaves"] + E["upper-canopy stems"]
    print(f"4.4 upper canopy {100 * up / Ein:.2f} % of the incident, stems' share {E['upper-canopy stems'] / up:.3f}")
    hrs = step_hours(dt); lit = list(np.where(day_)[0])
    for noon in sorted(int(k) for k in np.argsort(np.abs(hrs - 12.0))[:2]):      # the steps nearest solar noon
        i = lit.index(noon)
        print(f"4.4 step {noon} (sun at {hrs[noon]:.2f} h, mu {float(cz[noon]):.3f}): albedo all {tot[i]:.3f} vis {vis[i]:.3f}"
              f" nir {nir[i]:.3f}; diffuse fraction VIS {float(si[noon, 0] / (sd[noon, 0] + si[noon, 0])):.3f};"
              f" toppar upper {float(r.topparu[noon]):.1f} W m-2, K_u {float(r.termu6[noon]):.3f}"
              f" h_u {float(r.termu7[noon]):.3f}, dbeamu {float(r.dbeamu[noon]):.1f}; incident {inc[noon]:.1f} W m-2")
    print(f"4.4 first and last daylight steps {lit[0]} h and {lit[-1]} h (sun at {th[0]:.2f} and {th[-1]:.2f} h, mu "
          f"{float(cz[lit[0]]):.3f}, {float(cz[lit[-1]]):.3f}): albedo {tot[0]:.3f}, {tot[-1]:.3f}; {len(lit)} sunlit steps")
    print(f"4.4 day albedo (energy-weighted) all {E['reflected'] / Ein:.3f}")


# ---------------------------------------------------------------------------------------------------------------
# Numbers quoted in the text (run with the argument "checks"; not drawn by default)
# ---------------------------------------------------------------------------------------------------------------

def _bvp(om, bd, bi, mubar, G, mu, X, asd, asi, direct):
    """Exact solution of the two-stream equations (4.7) with the boundary conditions (4.8) by matrix exponentials,
    independent of twostr's closed form. y = [I_up, I_down]; y' = M y + s e^-Kx; y = expm(M x) a + v e^-Kx."""
    from scipy.linalg import expm
    b = 1 - om * (1 - bi); c = om * bi; K = G / max(mu, 0.01)
    d = om * mubar * K * bd; f = om * mubar * K * (1 - bd)
    src = 1.0 if direct else 0.0
    M = np.array([[b, -c], [c, -b]]) / mubar
    s = np.array([-d, f]) / mubar * src
    v = -np.linalg.solve(M + K * np.eye(2), s)
    E = expm(M * X); eK = np.exp(-K * X)
    ai = asd if direct else asi                                     # (4.8): the direct solution reflects with asd
    A = np.array([[0.0, 1.0], [E[0, 0] - ai * E[1, 0], E[0, 1] - ai * E[1, 1]]])
    rhs = np.array([(0.0 if direct else 1.0) - v[1], asd * src * eK - (v[0] - ai * v[1]) * eK])
    a = np.linalg.solve(A, rhs)
    y = lambda x: expm(M * x) @ a + v * np.exp(-K * x)
    dy = lambda x: M @ (expm(M * x) @ a) - K * v * np.exp(-K * x)
    return y, dy, K


def checks():
    # Table 4.1, derived columns
    for st, name in ((0, "lower"), (1, "upper")):
        for ib, band in ((0, "visible"), (1, "near-infrared")):
            r = one_layer(0.5, 4.0, 1.0, st + 1, ib, 0.1, 0.1)
            om, bd, bi, av, gd = RD.twoset(jnp.array([0.5]), jnp.array([4.0]), jnp.array([1.0]), jnp.zeros(1),
                                           jnp.zeros(1), jnp.array([290.0]), float(VP.rhoveg[ib, st]),
                                           float(VP.tauveg[ib, st]), float(VP.oriev[st]), float(VP.orieh[st]),
                                           float(RD._OMEGASNO[ib]), TMELT, st + 1, jnp.array([True]),
                                           **({"fwet_stem": jnp.zeros(1), "rliq_stem": jnp.zeros(1),
                                               "temp_stem": jnp.array([290.0])} if st == 1 else {}))
            print(f"Table 4.1 {name} {band}: omega0 {float(om[0]):.3f} beta {float(bi[0]):.4f} "
                  f"beta0(mu=0.5) {float(bd[0]):.4f} G(mu=0.5) {float(gd[0]):.4f}")
    # night: the terms handed to photosynthesis
    r = one_layer(0.0, 4.0, 1.0, 2, 0, 0.1, 0.1)
    print(f"night (mu = 0): K {r['term'][5]:.3f} h {r['term'][6]:.3f} omega {float(r['omega']):.3f}")
    # two-stream solution and absorption profile against the matrix-exponential solution
    worst = {}; ncomb = 0
    for iv in (1, 2):
        for ib in (0, 1):
            for mu in (0.05, 0.2, 0.5, 0.9):
                for X in (0.3, 2.0, 6.0):
                    for asd, asi in ((0.1, 0.1), (0.25, 0.3), (0.8, 0.7)):
                        ncomb += 1
                        st = iv - 1
                        z = jnp.array([mu]); L = jnp.array([0.8 * X]); S = jnp.array([0.2 * X]); z0 = jnp.zeros(1)
                        t = jnp.array([290.0])
                        kw = dict(fwet_stem=z0, rliq_stem=z0, temp_stem=t) if iv == 2 else {}
                        om, bd, bi, av, gd = RD.twoset(z, L, S, z0, z0, t, float(VP.rhoveg[ib, st]),
                                                       float(VP.tauveg[ib, st]), float(VP.oriev[st]),
                                                       float(VP.orieh[st]), float(RD._OMEGASNO[ib]), TMELT, iv,
                                                       z > 0, **kw)
                        r = RD.twostr(z, L, S, jnp.array([asd]), jnp.array([asi]), om, bd, bi, av, gd, 1e-7, z > 0,
                                      ib)
                        pr = [float(q[0]) for q in (om, bd, bi, av, gd)]
                        term = np.asarray(r.term)[0]
                        for direct in (True, False):
                            y, dy, K = _bvp(*pr, mu, X, asd, asi, direct)
                            refl, below = (r.refld, r.fbelid) if direct else (r.refli, r.fbelii)
                            worst["reflectance"] = max(worst.get("reflectance", 0), abs(float(refl[0]) - y(0.0)[0]))
                            worst["transmittance"] = max(worst.get("transmittance", 0), abs(float(below[0]) - y(X)[1]))
                            if ib == 0:
                                xs = np.linspace(0, X, 41)
                                absx = np.array([-((-K * np.exp(-K * q) if direct else 0.0) + dy(q)[1] - dy(q)[0])
                                                 for q in xs])
                                k, h = term[5], term[6]
                                prof = (term[0] * np.exp(-k * xs) + term[1] * np.exp(-h * xs)
                                        + term[3] * np.exp(h * xs)) if direct else \
                                       (term[2] * np.exp(-h * xs) + term[4] * np.exp(h * xs))
                                worst["profile"] = max(worst.get("profile", 0), np.max(np.abs(prof - absx)))
    print(f"two-stream vs matrix exponentials, {ncomb} combinations (both canopies and bands): largest differences "
          + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    # energy closure of compute_radiation at random states
    rng = np.random.default_rng(0); n = 4000
    u = lambda a, b: jnp.asarray(rng.uniform(a, b, n))
    coszen = u(0.02, 1.0); solad = jnp.stack([u(0, 400), u(0, 470)], 1); solai = jnp.stack([u(0, 200), u(0, 230)], 1)
    lai = jnp.stack([u(0, 6), u(0, 10)], 1); sai = jnp.stack([u(0, 0.5), u(0, 2.5)], 1)
    fu = u(0.05, 0.975); fl = u(0.05, 0.975); fi = jnp.where(u(0, 1) > 0.5, u(0, 1), 0.0)
    fira = u(150, 450)
    res = RD.compute_radiation(coszen, solad, solai, fira, lai, sai, fu, fl, u(250, 305), u(250, 305), u(250, 305),
                               u(250, 305), u(240, 273.16), fi, u(250, 273.16), u(0, 1), u(0, 0.5), u(0.05, 0.12),
                               u(0.1, 0.24), u(0, 0.25), u(0, 0.25), u(0, 0.25), u(0, 1), u(0, 1), u(0, 1), VP,
                               5.67e-8, TMELT, 1e-7)
    inc = solad.sum(1) + solai.sum(1)
    refl = (res.asurd * solad).sum(1) + (res.asuri * solai).sum(1)
    absb = fu * (res.solu + res.sols) + (1 - fi) * fl * res.soll + (1 - fi) * res.solg + fi * res.soli
    lw = fu * (res.firu + res.firs) + (1 - fi) * fl * res.firl + (1 - fi) * res.firg + fi * res.firi
    print(f"closure at {n} random states: solar {float(jnp.max(jnp.abs(inc - refl - absb))):.1e} W m-2 "
          f"(incident up to {float(inc.max()):.0f}), longwave {float(jnp.max(jnp.abs(fira - res.firb - lw))):.1e} W m-2")
    # the table of z in IBIS 2.6b4 against 1/2 - mu/3
    mu = np.linspace(1e-6, 1.0, 200001)
    tab = np.round(0.5 - np.arange(101) / 300.0, 4)
    dz = np.max(np.abs(tab[np.floor(mu * 100 + 1.5).astype(int) - 1] - (0.5 - mu / 3.0)))
    db = max((1 - float(VP.oriev[st]) - float(VP.orieh[st])) * abs(float(VP.tauveg[ib, st] - VP.rhoveg[ib, st]))
             / float(VP.tauveg[ib, st] + VP.rhoveg[ib, st]) * dz for st in (0, 1) for ib in (0, 1))
    print(f"z table of IBIS 2.6b4: largest difference in z {dz:.2e}, in beta0 {db:.2e}")
    # the direct-albedo boundary condition: (alpha_g - alpha_g^mu) T^mu_u per unit direct flux on the upper canopy
    def upper(mu, Lu, Su, Ll, Sl, fu_, fl_, fi_, tsno, tveg, albsav=0.092, wsoi=0.3):
        out = []
        for ib in (0, 1):
            one = jnp.ones(1); z = mu * one
            sod, soi, snd, sni = RD.solsur(z, wsoi * one, 0 * one, tsno * one, albsav * one, 2 * albsav * one, TMELT,
                                           z > 0, ib)
            _, _, lo, up = RD.solalb(z, jnp.array([[Ll, Lu]]), jnp.array([[Sl, Su]]), fu_ * one, fl_ * one, fi_ * one,
                                     sod, soi, snd, sni, 0 * one, 0 * one, 0 * one, one, one, one, tveg * one,
                                     tveg * one, tveg * one, VP, TMELT, 1e-7, ib, z > 0)
            gd = fl_ * (1 - fi_) * lo.refld + (1 - fl_) * (1 - fi_) * sod + fi_ * snd
            gi = fl_ * (1 - fi_) * lo.refli + (1 - fl_) * (1 - fi_) * soi + fi_ * sni
            out.append((float(gi[0] - gd[0]) * float(up.fbelid[0]), float(up.abvegd[0])))
        return out
    for name, a in (("forest of Fig. 4.4, snow-free, mu 0.9", (0.9, 5.85, 1.43, 0.53, 0.0015, 0.975, 0.05, 0.0, TMELT, 295.0, 0.1004, 0.459)),
                    ("same, mu 0.3", (0.3, 5.85, 1.43, 0.53, 0.0015, 0.975, 0.05, 0.0, TMELT, 295.0, 0.1004, 0.459)),
                    ("L_u 4, S_u 1 over snow at -5 C, mu 0.2", (0.2, 4.0, 1.0, 0.5, 0.05, 0.8, 0.05, 1.0, TMELT - 5, 268.0)),
                    ("L_u 4, S_u 1 over snow at -0.5 C, mu 0.1", (0.1, 4.0, 1.0, 0.5, 0.05, 0.8, 0.05, 1.0, TMELT - 0.5, 272.0)),
                    ("L_u 1, S_u 0.3 over snow at -0.5 C, mu 0.2", (0.2, 1.0, 0.3, 0.5, 0.05, 0.5, 0.05, 1.0, TMELT - 0.5, 272.0))):
        o = upper(*a)
        print(f"(alpha_g - alpha_g^mu) T^mu_u, {name}: " + "; ".join(
            f"{b} {d:+.5f} (direct absorption {ab:.3f})" for b, (d, ab) in zip(("VIS", "NIR"), o)))
    _check_window()
    _check_guards()
    _check_release_extremes()
    _check_tower_shortwave()


def _bvp_beam(om, bd, bi, G, mu, X, alb):
    """Exact direct-beam solution of (4.7)-(4.8) (mubar = 1) with the unscattered beam as a third state, so that it
    stays exact where K = h (there the particular solution of _bvp is singular). Returns rho^mu, T^mu, T^mumu."""
    from scipy.linalg import expm
    b = 1 - om * (1 - bi); c = om * bi; K = G / max(mu, 0.01)
    A = np.array([[b, -c, -om * K * bd], [c, -b, om * K * (1 - bd)], [0.0, 0.0, -K]])
    E = expm(A * X)
    z0, e0 = E @ np.array([0.0, 0.0, 1.0]), E @ np.array([1.0, 0.0, 0.0])   # Z(X) = z0 + I_up(0) e0
    g = lambda Z: Z[0] - alb * (Z[1] + Z[2])                                 # I_up(X) = alpha^mu [beam + I_down]
    iu0 = -g(z0) / g(e0)
    Z = z0 + iu0 * e0
    return iu0, Z[1], Z[2]


def _check_window():
    """The perturbation where |p - q| < 0.001 p (4.10)-(4.11). Since the review fix D26 (commit 8416c20) K is perturbed
    with p (k = p / avmu), so that s2, t1 and term 6 belong to the same beam; the release code (IBIS 2.6b4) perturbed p
    alone. Where the window lies (snow-free elements); inside it, the code's solution against _bvp_beam for the beam's
    true K and for the perturbed K (the beam the code now solves for); and how often the mu of the land cells falls in a
    window, at the middle of the step (step_time, the default) and at its start (the release code)."""
    mu = np.linspace(0.011, 1.0, 400001)
    wins = {}
    n = mu.size; one = jnp.ones(n); zero = jnp.zeros(n)
    for iv, cn in ((1, "lower"), (2, "upper")):
        st = iv - 1
        for ib, bn in ((0, "visible"), (1, "near-infrared")):
            om, bd, bi, av, G = (np.asarray(q) for q in RD.twoset(
                jnp.asarray(mu), 4.0 * one, one, zero, zero, 290.0 * one, float(VP.rhoveg[ib, st]),
                float(VP.tauveg[ib, st]), float(VP.oriev[st]), float(VP.orieh[st]), float(RD._OMEGASNO[ib]), TMELT, iv,
                one > 0, **({"fwet_stem": zero, "rliq_stem": zero, "temp_stem": 290.0 * one} if iv == 2 else {})))
            b, c = 1 - om * (1 - bi), om * bi
            p, q = av * G / np.maximum(mu, 0.01), np.sqrt(np.maximum(0.0, b * b - c * c))   # as in twostr (term is VIS only)
            w = np.abs(p - q) < 0.001 * p
            wins[(cn, bn)] = (mu[w].min(), mu[w].max())
    for (cn, bn), (m0, m1) in wins.items():
        iv = 1 if cn == "lower" else 2; ib = 0 if bn == "visible" else 1; st = iv - 1
        albs = (0.07, 0.1, 0.2, 0.3, 0.46) if iv == 1 else (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)   # soil; ground system
        m = np.concatenate([np.linspace(m0, m1, 41), [m0 - 5e-4, m1 + 5e-4]]); inside = np.arange(m.size) < 41
        n = m.size; one = jnp.ones(n); zero = jnp.zeros(n)
        eT = ea = 0.0; worst = None; eout = 0.0; eK = 0.0
        for X in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
            for alb in albs:
                r = one_layer(m, 0.8 * X, 0.2 * X, iv, ib, alb, alb)
                pr = [np.asarray(q) for q in RD.twoset(
                    jnp.asarray(m), 0.8 * X * one, 0.2 * X * one, zero, zero, 290.0 * one, float(VP.rhoveg[ib, st]),
                    float(VP.tauveg[ib, st]), float(VP.oriev[st]), float(VP.orieh[st]), float(RD._OMEGASNO[ib]),
                    TMELT, iv, one > 0, **({"fwet_stem": zero, "rliq_stem": zero, "temp_stem": 290.0 * one}
                                           if iv == 2 else {}))]
                for j in range(n):
                    om_, bd_, bi_, av_, G_ = (float(pr[i][j]) for i in range(5))
                    rho, Td, Tb = _bvp_beam(om_, bd_, bi_, G_, m[j], X, alb)
                    a_ex = 1 - rho - (1 - alb) * (Tb + Td)
                    dT, da = abs(r["fbelid"][j] - Td), abs(r["abvegd"][j] - a_ex)
                    if inside[j]:
                        if dT > eT: eT, worst = dT, (X, alb, round(float(m[j]), 5), float(r["fbelid"][j]), Td)
                        ea = max(ea, da)
                        # the beam the code solves for inside the window: K multiplied by 1.001 (p >= q) or 0.999
                        b_, c_ = 1 - om_ * (1 - bi_), om_ * bi_
                        p_, q_ = av_ * G_ / max(m[j], 0.01), np.sqrt(max(0.0, b_ * b_ - c_ * c_))
                        fac = 1.001 if p_ >= q_ else 0.999
                        rho2, Td2, Tb2 = _bvp_beam(om_, bd_, bi_, G_ * fac, m[j], X, alb)
                        a2 = 1 - rho2 - (1 - alb) * (Tb2 + Td2)
                        eK = max(eK, abs(r["fbelid"][j] - Td2), abs(r["abvegd"][j] - a2), abs(r["refld"][j] - rho2),
                                 abs(r["fbeldd"][j] - Tb2))
                    else:
                        eout = max(eout, dT, da)
        print(f"p ~ q window, {cn} canopy, {bn}: mu {m0:.4f}-{m1:.4f}; inside, largest |difference| from the exact "
              f"solution for the true K: T^mu {eT:.1e}, a^mu {ea:.1e} (largest T^mu at X, albedo, mu = {worst[:3]}: code "
              f"{worst[3]:.4f}, exact {worst[4]:.4f}); from the exact solution for the perturbed K {eK:.1e}; "
              f"0.0005 outside {eout:.1e}")
    # the mu of the land cells over a year (weather.py diurnal): at the step time of the model (the middle of the step
    # by default) and at the step start (the release code)
    land = _land_mask(); ul, nl = np.unique(89.75 - 0.5 * np.argwhere(land)[:, 0], return_counts=True)
    orbit = 2.0 * np.pi * np.arange(1, 366) / 365.2425
    decl = (0.006918 - 0.399912 * np.cos(orbit) + 0.070257 * np.sin(orbit) - 0.006758 * np.cos(2 * orbit)
            + 0.000907 * np.sin(2 * orbit) - 0.002697 * np.cos(3 * orbit) + 0.001480 * np.sin(3 * orbit))
    xl = np.deg2rad(ul)[:, None, None]
    wt = nl[:, None, None]
    for label, hours in (("at the model's step time", step_hours()), ("at the step start (release code)",
                                                                     np.arange(24.0))):
        ang = 2.0 * np.pi * (hours - 12.0) / 24.0
        cz = np.maximum(0.0, np.sin(xl) * np.sin(decl)[None, :, None]
                        + np.cos(xl) * np.cos(decl)[None, :, None] * np.cos(ang)[None, None, :])
        inw = np.zeros(cz.shape, bool)
        for m0, m1 in wins.values():
            inw |= (cz >= m0) & (cz <= m1)
        print(f"p ~ q windows, mu {label}: {100 * (inw * wt).sum() / ((cz > 0) * wt).sum():.2f} % of the sunlit hourly "
              f"steps of the {land.sum()} land cells over a year have their mu in a window")


def _check_guards():
    """Which of the port's guards in twostr bind (the same expressions as twostr, evaluated here): random daylight
    states against night-time points (mu = 0: twoset returns omega = gdir = 0 and solsur albedos 0)."""
    def q_(om, bd, bi, G, mu, X, a):
        b = 1 - om * (1 - bi); c = om * bi; t0 = b * b - c * c; q = np.sqrt(np.maximum(0, t0))
        p = G / np.maximum(mu, 0.01); p = np.where(np.abs(p - q) < 0.001 * p, (1 + np.where(p >= q, 1, -1) * 0.001) * p, p)
        s1 = np.exp(-q * X); u1 = b - c / np.maximum(a, 1e-12); u2 = b - c * a
        D1 = (b + q) * (u1 - q) / s1 - (b - q) * (u1 + q) * s1; D2 = (u2 + q) / s1 - (u2 - q) * s1
        return np.abs(D1), np.abs(D2), np.abs(p * p - t0), s1
    rng = np.random.default_rng(1); n = 1_000_000; u = lambda a, b: rng.uniform(a, b, n)
    D1, D2, sg, s1 = q_(u(0.1, 0.95), u(0.3, 0.7), u(0.45, 0.6), u(0.25, 0.85), u(1e-4, 1.0), u(1e-7, 15.0),
                        u(0.02, 0.99))
    print(f"guards, {n} random daylight states (omega 0.1-0.95, X <= 15, albedo >= 0.02): smallest |D1| {D1.min():.3f}, "
          f"|D2| {D2.min():.3f}, |sigma| {sg.min():.1e}, s1 {s1.min():.1e}")
    z = np.zeros(3); D1, D2, sg, s1 = q_(z, z, z, z, z, np.array([0.5, 5.0, 12.0]), z)
    print(f"guards at night (albedo 0): |D1| {D1}, |D2| {np.round(D2, 2)}, |sigma| {sg}")
    m = np.linspace(0.01, 1.0, 400)
    r = one_layer(m, 0.95 * 12.0, 0.05 * 12.0, 1, 0, 0.0703, 0.0703)
    print(f"smallest daylight albedo of the ground system's parts: lower-canopy visible reflectance over the darkest "
          f"saturated soil (0.0703), L + S = 12: direct {r['refld'].min():.3f}, diffuse {r['refli'].min():.3f}")


def _check_release_extremes():
    """Saturated soil albedo and the largest stem and plant area at the land cells (restart at the end of 2000)."""
    from ibis.dimensions import GridConfig
    from ibis.state.model_state import create_ibis_state
    tmpl = create_ibis_state(GridConfig(npoi=2, nbedrock=3))
    paths = [jax.tree_util.keystr(q) for q, _ in jax.tree_util.tree_flatten_with_path(tmpl)[0]]
    z = np.load(f"{REL}/hindcast/restart/ibis_restart_2000.npz")
    g = lambda name: z[f"leaf_{paths.index(name)}"]
    sai, lai, av, an = g(".vegetation.sai"), g(".vegetation.lai"), g(".soil.albsav"), g(".soil.albsan")
    print(f"release restart 2000, {sai.shape[0]} land cells: albsav {av.min():.4f}-{av.max():.4f}, albsan/albsav "
          f"{(an / av).min():.4f}-{(an / av).max():.4f}; largest L_l {lai[:, 0].max():.2f} L_u {lai[:, 1].max():.2f}, "
          f"S_l {sai[:, 0].max():.2f} S_u {sai[:, 1].max():.2f}, L + S {(lai + sai).max():.2f}")


def _check_tower_shortwave():
    """Tower runs: observed shortwave at steps whose mu is 0, which compute_radiation neither absorbs nor reflects.
    US-Ha1, 2001-2010, days with all 24 hours, mu as diurnal computes it at the latitude of the tower's 0.5-degree cell.

    As the tower reader builds the hourly shortwave now (review D34, commit 63fd7c0): the record's mean over each hour of
    local apparent solar time (tower_forcing.solar_hour_means; the tower's clock from tower_clock), with mu at the
    model's step time (step_time: the middle of the step under the default weather.sun_midstep, D33). For comparison the
    release code's rule, rebuilt from the same record: the instant at each hour of the tower's clock (the mean of the two
    intervals around it, as the release's _load_hourly_sw) taken as the solar hour, with mu at the start of the step."""
    import pandas as pd
    sys.path.insert(0, str(ROOT / "verify" / "calibration"))
    import tower_forcing as TF
    site = "US-Ha1"
    lat, lon = TF.site_location(site)
    d = TF._read_hh(site)
    u, note = TF.tower_clock(site, d, lon)
    rec = TF._sw_record(d)
    jd = pd.date_range("2001-01-01", "2010-12-31", freq="D").dayofyear.values
    orbit = 2.0 * np.pi * jd / 365.2425
    decl = (0.006918 - 0.399912 * np.cos(orbit) + 0.070257 * np.sin(orbit) - 0.006758 * np.cos(2 * orbit)
            + 0.000907 * np.sin(2 * orbit) - 0.002697 * np.cos(3 * orbit) + 0.001480 * np.sin(3 * orbit))
    xl = np.deg2rad(TF.cell_latitude(lat))
    hours_all = pd.date_range("2001-01-01", "2010-12-31 23:00", freq="h")
    night = np.isin(np.arange(24), (0, 1, 2, 3, 21, 22, 23))

    def lost_stats(label, series, hours):
        v = series.reindex(hours_all).values.reshape(-1, 24)
        ok = np.isfinite(v).all(1); v = np.where(ok[:, None], v, np.nan)        # days with all 24 hours, as the reader
        ang = 2.0 * np.pi * (hours - 12.0) / 24.0
        cz = np.maximum(0.0, np.sin(xl) * np.sin(decl)[:, None] + np.cos(xl) * np.cos(decl)[:, None] * np.cos(ang)[None, :])
        lost = (cz <= 0) & (v > 0)
        Lw = np.where(lost, v, 0.0)
        print(f"tower shortwave at mu = 0, {site} 2001-2010, {label}: {lost.sum()} of {np.isfinite(v).sum()} hours, "
              f"{100 * Lw.sum() / np.nansum(v):.3f} % of the shortwave, at most {Lw.max():.1f} W m-2, mean "
              f"{(Lw[lost].mean() if lost.any() else 0.0):.2f} W m-2; in the hours 21-03 {lost[:, night].sum()} hours, at "
              f"most {Lw[:, night].max():.1f} W m-2, {100 * Lw[:, night].sum() / max(Lw.sum(), 1e-30):.0f} % of the lost "
              f"shortwave; {ok.sum()} complete days")
        # the lost hours in which the sun is above the horizon for part of the solar hour [k, k+1) (sunrise or sunset
        # inside it), against those dark throughout (night-time readings)
        part = np.zeros_like(lost)
        for f in np.linspace(0.0, 1.0, 13):
            a = 2.0 * np.pi * (np.arange(24.0) + f - 12.0) / 24.0
            part |= (np.sin(xl) * np.sin(decl)[:, None] + np.cos(xl) * np.cos(decl)[:, None] * np.cos(a)[None, :]) > 0
        lp = lost & part
        ld = lost & ~part
        print(f"    of these, with the sun up for part of the hour [k, k+1): {lp.sum()} hours, "
              f"{100 * Lw[lp].sum() / max(Lw.sum(), 1e-30):.0f} % of the lost shortwave, mean {Lw[lp].mean():.2f}, at most "
              f"{Lw[lp].max():.1f} W m-2; dark throughout: {ld.sum()} hours, mean {Lw[ld].mean():.2f}, at most "
              f"{Lw[ld].max():.1f} W m-2")

    print(f"{site}: latitude {lat}, cell latitude {TF.cell_latitude(lat)}, longitude {lon}; clock: {note}")
    lost_stats("solar-hour means, sun at the step time (current code)",
               TF.solar_hour_means(d.index, rec.values, lon, u), step_hours())
    x = rec.clip(lower=0.0); inst = 0.5 * (x + x.shift(1)); inst = inst[inst.index.minute == 0]
    lost_stats("instant at the tower's hour, sun at the step start (release code)", inst, np.arange(24.0))


FIGURES = {"4.1": fig_4_1, "4.2": fig_4_2, "4.3": fig_4_3, "4.4": fig_4_4}
EXTRA = {"checks": checks}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        {**FIGURES, **EXTRA}[key]()
        print(key, "done")
