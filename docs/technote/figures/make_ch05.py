"""Figures of Chapter 5 (Surface energy, momentum and water vapor fluxes; canopy interception) of the IBIS 3.0
technical note, drawn from the model's code and the release run.

Run from the repository root with the project interpreter (CPU is enough):

    JAX_PLATFORMS=cpu PYTHONPATH=python-port /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch05.py
    JAX_PLATFORMS=cpu PYTHONPATH=python-port ... make_ch05.py 5.2 5.4     # some figures
    JAX_PLATFORMS=cpu PYTHONPATH=python-port ... make_ch05.py checks      # the numbers of the text that no figure shows
    JAX_PLATFORMS=cpu PYTHONPATH=python-port ... make_ch05.py snowsub     # the split of the snow-surface evaporation (2 min)
    JAX_PLATFORMS=cpu PYTHONPATH=python-port ... make_ch05.py variants    # what the energy-test variants do to c_u (seconds)
    JAX_PLATFORMS=cpu PYTHONPATH=python-port ... make_ch05.py wbal        # counters of the water-balance test year (seconds)

Every number a figure shows is computed here. Figure 5.1 is a schematic. Figure 5.2 calls the model's own functions
(`canini`, `turcof`, `fstrat`, `steph2o`, `esat`, `qsat`, `dqsat`) on prescribed inputs, with the default settings of
the parameter file (the energy-test variants `canopy.za_above_top`, `canopy.ca_cap` and `canopy.dew_on_dry` at their
defaults, which are IBIS 2.6b4's); its panel (f) evaluates the soil-surface humidity factor of `turvap`
(canopy_exchange.py), which is not a separate function, with the same expressions. Figure 5.3 reads the
evapotranspiration components of the release hindcast (final_rel30/hindcast, yearly files 2001-2010; release code
e54e342). Figure 5.4 runs the code of this checkout with the decided parameter file `params_32a_base.yaml` (the release
values plus the parameter decisions of the technote review, before the refits) at one cell from the release state at
the end of 2000 (final_rel30 hindcast restart of 2000) to 21 July 2001 and then steps through 22 and 23 July 2001 hour by
hour, recording the results of `canini`, `turcof`, `turvap`, `cascade`, `fwetcal` and `cascad2` inside `timestep_body`
and the fluxes the time step books (about 2 minutes on two CPU threads after compilation; set CH05_CACHE to a file path
to keep the hourly values between runs). The recorded days evaluate each step's sun and generated weather at the time
`step_time` gives, the middle of the step under the default `weather.sun_midstep` (technote review D33, commit
8416c20), as the model's own day loop does. `checks` evaluates the reference height, the canopy-free cells and the limit
of the transfer coefficient above the canopy on the release state; `snowsub` runs one year at a sample of 150 cells
(about 2 minutes) and compares the two ways of booking the snow-surface evaporation; `variants` evaluates the
energy-test options of the transfer above the canopy for the forest and grassland of Figure 5.2b; `wbal` reads the
counters of the water-balance module from one global test year (2001) of the revised code and the cells with a negative
yearly soil evaporation in that year and in the release run.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: E402,F401  (also sets rcParams)

import matplotlib                                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                     # noqa: E402
from matplotlib.patches import Rectangle, Ellipse                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

REL = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30"
RESTART_2000 = f"{REL}/hindcast/restart/ibis_restart_2000.npz"
#: the decided parameter set of the technote review: the release file params_31s_ch4h.yaml plus the G06 timing (D18),
#: without the unread solid-phase keys (D80) and with soil.evap_needs_store (D25); the base of the refits
PARAMS = str(ROOT / "python-port/configs/params/params_32a_base.yaml")
GSWP3 = "/shared/nas2/Min/GSWP3"
KOPPEN = "/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz"
CROPFRAC = "/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz"
NPOI = 95333
DT = 3600.0

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
LEG = dict(frameon=False, fontsize=6.6, handlelength=1.8, borderaxespad=0.3)


def _clean(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _title(ax, s):
    ax.set_title(s, loc="left", fontsize=8.5, weight="bold", color=INK)


# ---------------------------------------------------------------------------------------------------------------
# Figure 5.1: the column of the canopy exchange: levels, unknowns and transfer coefficients
# ---------------------------------------------------------------------------------------------------------------

def fig_5_1():
    fig, ax = plt.subplots(figsize=(7.8, 5.9))
    ax.set_xlim(0, 100); ax.set_ylim(-7, 100); ax.axis("off")
    XL, XS, XR = 14, 56, 72            # left edge of the cell, edge of the snowpack (snow cover f_i to its right), right edge
    ax.add_patch(Rectangle((XL, 88), XR - XL, 10, fc="#eaf2fb", ec="none"))
    ax.add_patch(Rectangle((XL, 58), XR - XL, 22, fc="#e6f0dd", ec="none"))
    ax.add_patch(Rectangle((XL, 26), XS - XL, 16, fc="#eef5df", ec="none"))              # lower canopy, snow-free part
    ax.add_patch(Rectangle((XL, -5), XR - XL, 19, fc="#d9c3a2", ec="none"))              # soil beneath the whole cell
    ax.add_patch(Rectangle((XS, 14), XR - XS, 32, fc="#f4f8fc", ec="#c9d6e3", lw=0.6))    # snowpack lying on the soil
    ax.add_patch(Rectangle((XS, 14), XR - XS, 3.2, fc="#a9c98a", ec="white", lw=0.8, hatch="////"))  # buried lower canopy
    ax.plot([XL, XR], [14, 14], color=INK2, lw=0.8)
    for y, lab in ((93, "$z_a$"), (80, "$z_1$"), (69, "$z_{12}$"), (58, "$z_2$"), (42, "$z_3$"), (34, "$z_{34}$"),
                   (26, "$z_4$"), (14, "ground")):
        ax.plot([11.5, 14], [y, y], color=INK2, lw=0.6)
        ax.text(11, y, lab, ha="right", va="center", fontsize=8, color=INK)
    it = dict(fontsize=7, color=INK2, style="italic", va="center")
    ax.text(36, 95.6, "reference level: forcing $T_a$, $q_a$, $u_a$, $p$", **it)
    ax.text(32, 77.6, "upper canopy (trees), cover $f_u$", **it)
    ax.text(15, 39.4, "lower canopy,\ncover $f_l$", linespacing=1.15, **it)
    ax.text(64, 52.5, "snowpack on the soil,\ncover $f_i$", ha="center", linespacing=1.15, **it)
    ax.text(64, 20.2, "buried lower canopy", ha="center", fontsize=6.4, color=INK2, style="italic", va="center")
    ax.text(29, 18.0, "soil surface, $1-f_i$", **it)
    ax.text(47, -1.5, "soil", **it)

    def node(x, y, lab, fc="white", r=3.3, fs=7.6):
        ax.add_patch(Ellipse((x, y), 2 * r * 1.15, 2 * r, fc=fc, ec=INK2, lw=0.8, zorder=3))
        ax.text(x, y, lab, ha="center", va="center", fontsize=fs, color=INK, zorder=4)

    def edge(p, q, lab, ha="left", dx=0.0, dy=0.0, at=0.5):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=INK2, lw=1.0, zorder=2)
        mx, my = p[0] + at * (q[0] - p[0]) + dx, p[1] + at * (q[1] - p[1]) + dy
        if lab:
            ax.text(mx, my, lab, ha=ha, va="center", fontsize=7.6, color=BLUE, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))

    A = (30, 93); A12 = (30, 69); A34 = (30, 34); U = (55, 73); S = (55, 64); Lw = (48, 34)
    G = (24, 14); I = (65, 46); G1 = (24, 1); I1 = (65, 34)
    edge(A, A12, "$c_u$", dx=1.2)
    edge(A12, A34, "$c_l$", dx=1.2)
    edge(U, A12, "$x_u s_u$", ha="center", dy=2.6)
    edge(S, A12, "$x_s s_s$", ha="center", dy=-2.6)
    edge(Lw, A34, "$x_l s_l$", ha="center", dy=-2.6)
    edge(G, A34, "$(1-f_i) s_g$", ha="right", dx=-1.6, dy=2.0)
    edge(I, A34, "$f_i s_i$", ha="center", dy=2.3, at=0.62)
    edge(G, G1, "", ha="right")
    edge(I, I1, "")
    ax.text(22.6, 7.5, "conduction", ha="right", va="center", fontsize=7.2, color=BLUE)
    ax.text(63.9, 40.0, "conduction", ha="right", va="center", fontsize=6.6, color=BLUE)
    node(*A, "$T_a$\n$q_a$", fc="#eaf2fb", r=3.6)
    node(*A12, "$T_{12}$\n$q_{12}$", r=3.6)
    node(*A34, "$T_{34}$\n$q_{34}$", r=3.6)
    node(*U, "$T_u$"); node(*S, "$T_s$"); node(*Lw, "$T_l$")
    node(*G, "$T_g$"); node(*I, "$T_i$")
    node(*G1, "$T_{soi,1}$", fc="#d9c3a2", r=3.1, fs=6.8); node(*I1, "$T_{sno,1}$", fc="#f4f8fc", r=3.1, fs=6.8)
    ax.text(60, 73, "upper leaves", fontsize=7, color=INK2, va="center")
    ax.text(60, 64, "upper stems", fontsize=7, color=INK2, va="center")
    ax.text(48, 29.0, "lower leaves\nand stems", fontsize=7, color=INK2, ha="center", va="top", linespacing=1.1)
    ax.text(28.5, 1, "top soil layer", fontsize=7, color=INK2, va="center")
    ax.text(64.5, 28.6, "top snow layer", fontsize=6.6, color=INK2, ha="center", va="center")
    kx = 75.5
    ax.text(kx, 97, "Unknowns of the implicit system", fontsize=8, weight="bold", color=INK, va="top")
    ax.text(kx, 92.6, "temperatures of the upper leaves,\nupper stems and lower canopy:\n"
                      "$T_u$, $T_s$, $T_l$\n\ncanopy-air temperatures and\nspecific humidities:\n"
                      "$T_{12}$, $T_{34}$, $q_{12}$, $q_{34}$\n\nsoil and snow surface temperatures:\n$T_g$, $T_i$",
            fontsize=7.2, color=INK2, va="top", linespacing=1.25)
    ax.text(kx, 44, "Transfer coefficients", fontsize=8, weight="bold", color=INK, va="top")
    ax.text(kx, 39.6, "$c_u$, $c_l$: air to air\n$s_u$, $s_s$, $s_l$: leaf or stem\nsurface to air\n"
                      "$s_g$, $s_i$: soil or snow\nsurface to air\n(kg m$^{-2}$ s$^{-1}$)\n\n"
                      "$x_u$, $x_s$, $x_l$: leaf and stem\nsurface per unit ground", fontsize=7.2, color=INK2,
            va="top", linespacing=1.25)
    fig.savefig(OUT / "fig_5_1_column.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 5.2: response curves of the aerodynamics, interception and soil-surface humidity
# ---------------------------------------------------------------------------------------------------------------

def _aero(ua, lai_u, sai_u, fu, lai_l, sai_l, fl, ztop, zbot, ta=293.15, qa=0.01, ps=1.0e5, fi=0.0, za_above_top=1.0,
          ca_cap=1.0):
    """canini + turcof for neutral stratification at the first iteration; also c_a of turcof before its limit. The
    keywords are the energy-test variants canopy.za_above_top and canopy.ca_cap (defaults: IBIS 2.6b4)."""
    import jax.numpy as jnp
    from ibis.constants import PhysicalConstants
    from ibis.physics.canopy_exchange import canini, turcof, CanopyParams
    c = PhysicalConstants(); cp = CanopyParams()
    n = max(np.size(ua), np.size(lai_u), np.size(lai_l))
    B = lambda x: jnp.broadcast_to(jnp.asarray(x, jnp.float32), (n,))
    lai = jnp.stack([B(lai_l), B(lai_u)], 1); sai = jnp.stack([B(sai_l), B(sai_u)], 1)
    zt = jnp.stack([B(ztop[0]), B(ztop[1])], 1); zb = jnp.stack([B(zbot[0]), B(zbot[1])], 1)
    a = canini(B(ta), B(qa), B(ps), B(ua), B(fi), B(0.005), 0.0005, B(fl), B(fu), lai, sai, zt, zb,
               cp.alaiml, cp.alaimu, c.grav, c.rair, c.rvap, c.cair, c.cvap, c.cappa, za_above_top=za_above_top)
    t12 = B(ta) * a.tfac; q12 = B(qa)                       # Ri = 0 in both regions
    z = jnp.zeros((n,)); o = jnp.ones((n,))
    tc = turcof(a, B(ta), B(qa), B(ua), t12, t12, q12, q12, B(fi), c.vonk, DT, cp.cleaf, cp.cstem, cp.cgrass,
                cp.dleaf, cp.dstem, 0, z, o, o, z, o, o, c.grav, ca_cap=ca_cap)
    ca = B(ua) * tc.strahu * c.vonk ** 2 / ((a.aloga - tc.alogu) * (a.aloga - a.alog1))
    return a, tc, np.asarray(ca)


FOREST = dict(fu=1.0, lai_l=1.0, sai_l=0.05, fl=0.5, ztop=(0.25, 20.0), zbot=(0.05, 3.0))
GRASS = dict(lai_u=0.0, sai_u=0.0, fu=0.0, sai_l=0.1, fl=1.0, ztop=(0.5, 4.0), zbot=(0.05, 3.0))


def fig_5_2():
    import jax.numpy as jnp
    from ibis.constants import PhysicalConstants
    from ibis.physics.canopy_exchange import fstrat
    from ibis.physics.interception import steph2o, InterceptionParams
    from ibis.physics.thermo import esat, qsat, dqsat
    from ibis.io.soil_init import _POROS, _SWILT, _BEX, _SUCTION
    c = PhysicalConstants(); ip = InterceptionParams()
    fig, axs = plt.subplots(2, 3, figsize=(10.2, 6.4), gridspec_kw={"hspace": 0.5, "wspace": 0.36})

    # (a) roughness length and displacement height of a 20 m forest against its leaf + stem area
    L = np.linspace(0.0, 8.0, 161)
    a, tc, _ = _aero(np.full(L.shape, 3.0), L, 0.25 * L, **FOREST)
    h = FOREST["ztop"][1]
    z0 = np.exp(np.asarray(tc.alogu)); d = np.asarray(a.dispu)
    ax = axs[0, 0]
    ax.plot(1.25 * L, d / h, color=BLUE, lw=1.4, label="displacement height $d_u$")
    ax.plot(1.25 * L, z0 / h, color=ORANGE, lw=1.4, label="roughness length $z_{0,u}$")
    ax.axvline(4.0, color=INK2, lw=0.7, ls=":")
    ax.text(4.15, 0.6, "$\\xi_u = 1$", fontsize=6.8, color=INK2)
    ax.set_xlim(0, 10); ax.set_ylim(0, 0.8)
    ax.set_xlabel("upper-canopy leaf + stem area index $L_u + S_u$")
    ax.set_ylabel("fraction of $z_{top,u}$ = 20 m")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.5), **{**LEG, "frameon": True, "facecolor": "white", "edgecolor": "none", "framealpha": 1.0}); _clean(ax)
    _title(ax, "(a) roughness of a forest")
    i4 = np.argmin(np.abs(1.25 * L - 4.0))
    print(f"5.2a forest z0/h max {np.max(z0 / h):.3f} (z0 {np.max(z0):.2f} m) at L+S={1.25 * L[np.argmax(z0)]:.2f}; "
          f"at L+S=4 z0 {z0[i4]:.2f} m d {d[i4]:.2f} m; at L+S=10 z0/h {z0[-1] / h:.3f} d/h {d[-1] / h:.3f}; za {float(a.za[0]):.2f}")

    # (b) the air-to-air coefficient above the canopy against the wind speed, with its limit
    U = np.linspace(2.5, 10.0, 751)                      # the range of the hourly wind speed of the generator, (3.13)
    af, tf, caf = _aero(U, 5.0, 1.25, **FOREST)
    ag, tg, cag = _aero(U, **dict(GRASS, lai_l=2.0))
    cmax = 300.0 / (2.0 * DT)
    ax = axs[0, 1]
    ax.plot(U, caf, color=BLUE, lw=1.4, label="$c_a$ before the limit, forest")
    ax.plot(U, cag, color=ORANGE, lw=1.4, label="$c_a$ before the limit, grassland")
    ax.plot(U, np.asarray(tf.cu / af.rhoa), color=BLUE, lw=1.1, ls="--", label="$c_u/\\rho_a$, forest")
    ax.axhline(cmax, color=INK2, lw=0.8, ls=":")
    ax.text(2.6, cmax * 1.18, "$c_{max} = 300/(2\\Delta t)$", fontsize=6.8, color=INK2, ha="left", va="bottom")
    ax.set_yscale("log"); ax.set_xlim(2.5, 10); ax.set_ylim(1e-2, 30); ax.set_xticks(np.arange(3, 11))
    ax.set_xlabel("wind speed $u_a$ (m s$^{-1}$)"); ax.set_ylabel("m s$^{-1}$")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.47), **LEG); _clean(ax)
    _title(ax, "(b) transfer above the canopy")
    gc = cag - cmax                                       # grassland: c_a is linear in u_a; interpolate the crossing
    k = int(np.argmax(gc >= 0)) if np.any(gc >= 0) else 0
    gx = U[k - 1] - gc[k - 1] * (U[k] - U[k - 1]) / (gc[k] - gc[k - 1]) if k > 0 else np.nan
    cur = np.asarray(tf.cu / af.rhoa)
    print(f"5.2b c_max {cmax:.4f}; forest c_a at u=2.5 {caf[0]:.3f}, at 10 {caf[-1]:.3f}; cu/rho at 2.5 {cur[0]:.5f}, at 10 "
          f"{cur[-1]:.5f}; grass c_a at 2.5 {cag[0]:.5f}, reaches c_max at u={gx:.3f}; grass cu/rho - min(c_a, c_max) max "
          f"{np.max(np.abs(np.asarray(tg.cu / ag.rhoa) - np.minimum(cag, cmax))):.2e}; forest z0 {float(np.exp(tf.alogu[0])):.2f} "
          f"d {float(af.dispu[0]):.2f} za {float(af.za[0]):.2f}; grass z0 {float(np.exp(tg.alogu[0])):.3f} d {float(ag.dispu[0]):.3f}"
          f" za {float(ag.za[0]):.2f}")

    # (c) the stratification factors of fstrat against the Richardson number
    n = 401
    ri_target = np.linspace(-2.0, 1.0, n)
    u = 3.0; zb, zt = 20.0, 30.0; tb = 290.0
    k = ri_target * u ** 2 / (c.grav * (zt - zb))          # top temperature that gives the target Ri, with q = 0
    tt = tb * (1 + 0.5 * k) / (1 - 0.5 * k)
    ax = axs[0, 2]
    for X, ls in ((2.0, "-"), (5.0, "--")):
        alt = jnp.full((n,), X); alb = jnp.zeros((n,))
        Z = jnp.zeros((n,)); O = jnp.ones((n,))
        rich, sm, sh = fstrat(jnp.full((n,), tb), jnp.asarray(tt), 1.0, Z, Z, jnp.full((n,), zb), jnp.full((n,), zt),
                              alb, alb, alt, jnp.full((n,), u), c.grav, c.vonk, 0, Z, O, O)
        ax.plot(np.asarray(rich), np.asarray(sm), color=BLUE, lw=1.3, ls=ls, label=f"momentum $F_m$, $X$ = {X:g}")
        ax.plot(np.asarray(rich), np.asarray(sh), color=ORANGE, lw=1.3, ls=ls, label=f"heat and water vapour $F_h$, $X$ = {X:g}")
        if X == 2.0:
            print(f"5.2c X=2: F_m, F_h at Ri=-2 {float(sm[0]):.3f} {float(sh[0]):.3f}; at Ri=1 {float(sm[-1]):.4f} {float(sh[-1]):.5f}")
        else:
            print(f"5.2c X=5: F_m, F_h at Ri=-2 {float(sm[0]):.3f} {float(sh[0]):.3f}")
    ax.set_yscale("log"); ax.set_xlim(-2, 1); ax.set_ylim(0.01, 30); ax.set_xticks([-2, -1, 0, 1])
    ax.set_xlabel("bulk Richardson number Ri"); ax.set_ylabel("stratification factor")
    ax.legend(loc="lower left", **LEG); _clean(ax)
    _title(ax, "(c) stratification factors")

    # (d, e) water held on the upper leaves and rain below them during and after 4 h of rain at 2 mm/h (no evaporation)
    nsteps = 12; rain = np.where(np.arange(nsteps) < 4, 2.0 / 3600.0, 0.0)      # kg m-2 s-1
    Ls = (1.0, 3.0, 6.0)
    wmin = 0.001 * (DT / 3600.0) * (ip.wliqumax / 0.2)
    snowmin = 0.001 * (DT / 3600.0) * (ip.wsnoumax / 2.0)
    store = np.zeros((len(Ls), nsteps + 1)); thru = np.zeros((len(Ls), nsteps))
    for i, Lu in enumerate(Ls):
        w = jnp.zeros((1,)); s = jnp.zeros((1,)); T = jnp.full((1,), 290.0)
        for t in range(nsteps):
            r = steph2o(T, w, s, jnp.full((1,), 2.0 * Lu), jnp.zeros((1,)), jnp.full((1,), rain[t]), jnp.full((1,), 289.0),
                        jnp.zeros((1,)), jnp.full((1,), 273.16), ip.tdripu, ip.tblowu, ip.wliqumax, ip.wsnoumax,
                        wmin, snowmin, DT, c.ch2o, c.cice, c.tmelt, c.hfus, c.epsilon)
            w, s = r.wliq, r.wsno
            store[i, t + 1] = float(w[0]) * 2.0 * Lu
            thru[i, t] = float(r.rain[0]) * DT
    hrs = np.arange(nsteps + 1)
    cols3 = (BLUE, ORANGE, AQUA)
    ax = axs[1, 0]
    ax.axvspan(0, 4, color="#eaf2fb", zorder=0)
    for i, (Lu, col) in enumerate(zip(Ls, cols3)):
        ax.plot(hrs, store[i], color=col, lw=1.4, marker="o", ms=2.5, label=f"$L_u$ = {Lu:g}")
        ax.axhline(2.0 * Lu * ip.wliqumax, color=col, lw=0.7, ls=":")
    ax.text(2, 2.62, "rain 2 mm h$^{-1}$", fontsize=6.8, color=INK2, ha="center")
    ax.set_xlim(0, nsteps); ax.set_ylim(0, 2.8); ax.set_xticks(range(0, nsteps + 1, 2))
    ax.set_xlabel("hour"); ax.set_ylabel("water held per unit cover (mm)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.8), **LEG); _clean(ax)
    _title(ax, "(d) interception and drip")
    ax = axs[1, 1]
    ax.axvspan(0, 4, color="#eaf2fb", zorder=0)
    ax.step(np.arange(nsteps + 1), np.r_[rain * DT, rain[-1] * DT], where="post", color=INK2, lw=1.0, ls="--",
            label="rain above the leaves")
    for i, (Lu, col) in enumerate(zip(Ls, cols3)):
        ax.step(np.arange(nsteps + 1), np.r_[thru[i], thru[i][-1]], where="post", color=col, lw=1.4, label=f"$L_u$ = {Lu:g}")
    ax.set_xlim(0, nsteps); ax.set_ylim(0, 2.6); ax.set_xticks(range(0, nsteps + 1, 2))
    ax.set_xlabel("hour"); ax.set_ylabel("rain below the leaves (mm h$^{-1}$)")
    ax.legend(loc="upper right", **LEG); _clean(ax)
    _title(ax, "(e) throughfall and drip")
    print("5.2d water held at the end of each hour (mm per unit cover)", np.round(store, 3),
          "\n  throughfall + drip (mm/h)", np.round(thru, 3), "\n  totals below the leaves over 12 h", np.round(thru.sum(1), 3),
          "of", rain.sum() * DT)

    # (f) the soil-surface humidity factor alpha of turvap against the top-layer saturation, for three textures
    T = 293.15; p = 1.0e5; q34_over_qg = 0.5; u4 = 1.0
    e34 = esat(jnp.float32(T)); qs1 = qsat(e34, jnp.float32(p)); dqs1 = dqsat(jnp.float32(T), qs1)
    q34 = q34_over_qg * float(qs1)                           # T_g = T_34 = T_a, q_a = q_34
    rhoa = p * 0.999 / (c.rair * T * (1.0 + (c.rvap / c.rair - 1.0) * q34))   # (5.6)
    cpm = c.cair * (1.0 + (c.cvap / c.cair - 1.0) * q34)                     # (5.6)
    sg = rhoa * (0.004 + 0.012 * u4)                        # s_g without the in-canopy resistance
    betafac = float(c.hvap * dqs1 / (cpm + 4.0 * 0.95 * c.stef * T ** 3 / sg))
    ax = axs[1, 2]
    for (cls, name), col in zip(((0, "sand"), (3, "loam"), (10, "clay")), cols3):
        swilt = _SWILT[cls] / _POROS[cls]
        z = np.linspace(0.02, 1.0, 491)
        psig = -c.grav * _SUCTION[cls] * z ** (-_BEX[cls])
        hfac = np.exp(psig / (c.rvap * T))
        betaw = np.clip((z - swilt) / (1.0 - swilt), 0.0, 1.0)
        betas = betaw / (1.0 + betafac * (1.0 - betaw))
        alpha = np.where(betas > 0, betas * hfac + (1.0 - betas) * q34_over_qg, q34_over_qg)
        ax.plot(z, alpha, color=col, lw=1.4, label=f"{name} ($s_{{w}}$ = {swilt:.2f})")
        print(f"5.2f {name}: s_w {swilt:.3f}, h1 at z=s_w {np.exp(-c.grav * _SUCTION[cls] * swilt ** (-_BEX[cls]) / (c.rvap * T)):.3f};"
              f" alpha at z=0.5 {np.interp(0.5, z, alpha):.3f}, at 0.8 {np.interp(0.8, z, alpha):.3f}")
    ax.axhline(q34_over_qg, color=INK2, lw=0.7, ls=":")
    ax.text(0.02, q34_over_qg - 0.012, "$h_2 = q_{34}/q_{sat}(T_g)$", fontsize=6.8, color=INK2, va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("top-layer water and ice $z_{tot}$"); ax.set_ylabel("humidity factor $\\alpha$")
    ax.legend(loc="lower right", **LEG); _clean(ax)
    _title(ax, "(f) soil-surface humidity")
    print(f"5.2f gamma (betafac) {betafac:.4f} at T {T} K, q_34 = q_a {q34 * 1e3:.3f} g/kg, c_p {cpm:.2f}, rho_a {rhoa:.4f},"
          f" s_g/rho {sg / rhoa:.4f} m/s")
    fig.savefig(OUT / "fig_5_2_response_curves.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Model runs at release cells
# ---------------------------------------------------------------------------------------------------------------

_CAP = {}


def _land_index(I, J):
    m = _land_mask()
    flat = np.where(m.ravel())[0]
    idx = np.searchsorted(flat, np.atleast_1d(I) * 720 + np.atleast_1d(J))
    assert np.all(flat[idx] == np.atleast_1d(I) * 720 + np.atleast_1d(J))
    return idx


def _cell_params(I, J, year):
    """SimParams of the parameter file PARAMS at cells (I, J): parameter file, cropland share, Köppen-zone types."""
    import jax.numpy as jnp
    from ibis.io.physics_params import load_physics_params, build_all_params
    from ibis.io import pft_types as pt
    from ibis.io.koppen import load_koppen, period_index
    I = np.atleast_1d(I); J = np.atleast_1d(J)
    pdict = load_physics_params(PARAMS)
    lat_rad = jnp.array(np.deg2rad(89.75 - 0.5 * I))
    cf = np.asarray(np.load(CROPFRAC)["cropfrac"][I, J], np.float32)
    params, pft_params, _ = build_all_params(I.size, lat_rad, pdict, cropfrac=cf)
    kop = load_koppen(KOPPEN, I, J)
    zones = pt.zones_from_classes(kop["classes"])
    k = period_index(year, kop["period_start"], kop["period_end"])
    over = pt.parse_type_overrides(pdict.get("types"))
    params, pft_params = pt.apply_types(params, pft_params, zones[k], over, base=(params, pft_params))
    return params, lat_rad, cf, [pt.ZONES[int(z)] for z in zones[k]]


def _release_state(idx):
    """The release state at the end of 2000 at the land cells idx."""
    import jax
    from ibis.dimensions import GridConfig
    from ibis.io.restart import load_state
    from ibis.io.co2 import apply_co2
    from ibis.state.model_state import create_ibis_state
    tmpl = create_ibis_state(GridConfig(npoi=NPOI, nbedrock=3), 296e-6, 0.209, with_stress=True, with_etparts=True,
                             with_seasonal=True)
    full, _ = load_state(RESTART_2000, tmpl)
    if idx is None:
        return full
    idx = np.asarray(idx)
    st = jax.tree.map(lambda x: x[idx] if (hasattr(x, "shape") and np.ndim(x) >= 1 and x.shape[0] == NPOI) else x, full)
    return apply_co2(st, 2001)


def _month_xs(reader, I, J, lat_rad, year, imonth, ndays_take=None):
    import jax.numpy as jnp
    from ibis.io.gswp3_forcing import make_monthly_daily_weather
    from ibis.timestepping.calendar import julian_day, is_leap_year
    mo = reader.read_month(year, imonth, np.atleast_1d(I), np.atleast_1d(J))
    dl, rl = make_monthly_daily_weather(mo, np.asarray(lat_rad), year, imonth, np.zeros((np.size(I), 3), np.float32))
    nd = mo["ndays"] if ndays_take is None else ndays_take
    st = lambda f: jnp.stack([getattr(dw, f) for dw in dl[:nd]])
    xs = tuple(st(f) for f in ("td", "tmax", "tmin", "cloud", "precip", "psurf", "qd", "ud", "iwet"))
    xs = xs + (jnp.stack(rl[:nd]), jnp.array([julian_day(imonth, d, is_leap_year(year)) for d in range(nd)], jnp.int32))
    return xs, dl, rl, mo


def _install_capture():
    """Wrap the canopy routines that timestep_body calls, to record their results (the model is unchanged)."""
    import ibis.timestepping.loops as L
    if getattr(L, "_ch05_wrapped", False):
        return
    orig = {k: getattr(L, k) for k in ("canini", "turcof", "turvap", "cascade", "fwetcal", "cascad2")}

    def wrap(name):
        def f(*a, **k):
            r = orig[name](*a, **k); _CAP[name] = r; return r
        return f
    for k in orig:
        setattr(L, k, wrap(k))
    L._ch05_wrapped = True


def _worked_days(I=94, J=215, year=2001, month=6, day=21, keys=(7, 8)):
    """Run the parameter file PARAMS at one cell from the release state at the end of 2000 to the start of the given
    day, then len(keys) days hour by hour with the canopy routines' results recorded. The month runs use the model's own
    day loop (phenology, decomposition); the recorded days run the 24 time steps of each day only, with the random
    draws of jax.random.PRNGKey(keys[d]) for day d, each step's sun and weather at the model's step_time (the middle of
    the step by default), as the model's day loop. Returns a dict of hourly arrays."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from ibis.constants import PhysicalConstants
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.physics.weather import diurnal, compute_precip_timing
    from ibis.timestepping.calendar import julian_day
    import ibis.timestepping.loops as L
    import ibis.timestepping.loops_gswp3 as LG
    consts = PhysicalConstants(); c = consts
    t0 = time.time()
    idx = _land_index(I, J)
    st = _release_state(idx)
    params, lat_rad, cf, zone = _cell_params(I, J, year)
    cp = params.canopy
    reader = GSWP3Reader(GSWP3)
    key = jax.random.PRNGKey(2001)
    for m in range(month):
        st, key, _ = LG.run_month_gswp3(st, consts, params, reader, np.array([I]), np.array([J]), np.asarray(lat_rad),
                                       year, m, key, DT, leap=False)
        print(f"  month {m + 1} done {time.time() - t0:.0f} s", flush=True)
    if day > 0:
        xs, _, _, _ = _month_xs(reader, I, J, lat_rad, year, month, ndays_take=day)
        st, key, _ = LG._run_month_scan(LG._cast_tree_f32(st), key, consts, params, xs, 1, 365, DT, 2)
    _, dl, rl, mo = _month_xs(reader, I, J, lat_rad, year, month)
    _install_capture()
    st = LG._cast_tree_f32(st)
    v0 = st.vegetation
    res = dict(lai=np.asarray(v0.lai)[0], sai=np.asarray(v0.sai)[0], fu=float(v0.fu[0]), fl=float(v0.fl[0]),
               ztop=np.asarray(v0.ztop)[0], zbot=np.asarray(v0.zbot)[0], frac=np.asarray(v0.frac)[0],
               zone=zone[0], cropfrac=float(cf[0]), cell=int(idx[0]))

    def run_day(st, dw, rlds, jd, k):
        k, plens, startp, endp = compute_precip_timing(k, jnp.asarray(dw.precip), DT)
        # the day's mean air temperature, as the model's day loop sets it (read only by stomata.frost_on)
        st = eqx.tree_at(lambda a: a.atm.td, st, jnp.asarray(dw.td, dtype=st.atm.td.dtype))

        def body(carry, istep):
            st, k = carry
            k, sub = jax.random.split(k)
            # the step's sun and generated weather at the time of day the model uses (step_time: the middle of the step
            # under the default weather.sun_midstep; the release code used the step start)
            _, h = diurnal(L.step_time(istep, DT, params), jd, dw, LG.latitude_degrees(params), plens, startp, endp, sub,
                           DT, 2, wind_keep_mean=params.wind_keep_mean)
            cc = lambda x: jnp.asarray(x, dtype=st.atm.ta.dtype)
            st = eqx.tree_at(lambda a: (a.atm.ta, a.atm.qa, a.atm.ua, a.atm.coszen, a.atm.solad, a.atm.solai, a.atm.fira,
                                        a.atm.raina, a.atm.snowa, a.atm.psurf), st,
                             (cc(h.ta), cc(h.qa), cc(h.ua), cc(h.coszen), cc(h.solad), cc(h.solai), cc(rlds), cc(h.raina),
                              cc(h.snowa), cc(jnp.broadcast_to(dw.psurf, st.atm.psurf.shape))))
            s0 = st
            st = LG._cast_tree_f32(L.timestep_body(st, consts, params, DT))
            ae, tc, cr, cs, fw, c2 = (_CAP[n] for n in ("canini", "turcof", "turvap", "cascade", "fwetcal", "cascad2"))
            v = s0.vegetation; fi = s0.snow.fi
            ca = s0.atm.ua * tc.strahu * c.vonk ** 2 / ((ae.aloga - tc.alogu) * (ae.aloga - ae.alog1))
            # heat stored in the canopy over the step: heat capacities with the water held after interception,
            # temperature change of the implicit solution (per unit ground area)
            chu = cp.chu + c.ch2o * cs.wliqu + c.cice * cs.wsnou
            chs = cp.chs + c.ch2o * cs.wliqs + c.cice * cs.wsnos
            chl = cp.chl + c.ch2o * cs.wliql + c.cice * cs.wsnol
            xu = 2.0 * v.lai[:, 1] * v.fu; xs_ = 2.0 * v.sai[:, 1] * v.fu
            xl = 2.0 * (v.lai[:, 0] + v.sai[:, 0]) * v.fl * (1.0 - fi)
            store_heat = (xu * chu * (cr.tu - s0.canopy.tu) + xs_ * chs * (cr.ts - s0.canopy.ts)
                          + xl * chl * (cr.tl - s0.canopy.tl)) / DT
            pflux = xu * cs.pfluxu + xs_ * cs.pfluxs + xl * cs.pfluxl
            cn = st.canopy; vn = st.vegetation
            water = (vn.fu * 2.0 * (vn.lai[:, 1] * (cn.wliqu + cn.wsnou) + vn.sai[:, 1] * (cn.wliqs + cn.wsnos))
                     + vn.fl * (1.0 - st.snow.fi) * 2.0 * (vn.lai[:, 0] + vn.sai[:, 0]) * (cn.wliql + cn.wsnol))
            c0 = s0.canopy
            water0 = (v.fu * 2.0 * (v.lai[:, 1] * (c0.wliqu + c0.wsnou) + v.sai[:, 1] * (c0.wliqs + c0.wsnos))
                      + v.fl * (1.0 - fi) * 2.0 * (v.lai[:, 0] + v.sai[:, 0]) * (c0.wliql + c0.wsnol))
            out = dict(ta=s0.atm.ta, qa=s0.atm.qa, ua=s0.atm.ua, rain=s0.atm.raina + s0.atm.snowa, fira=s0.atm.fira,
                       psurf=s0.atm.psurf,
                       swin=jnp.sum(s0.atm.solad + s0.atm.solai, axis=-1),
                       swabs=jnp.sum(s0.atm.solad * (1.0 - cn.asurd) + s0.atm.solai * (1.0 - cn.asuri), axis=-1),
                       za=ae.za, z1=ae.z1, dispu=ae.dispu, rhoa=ae.rhoa, cpair=ae.cp,
                       ca=ca, cu=tc.cu, cl=tc.cl, sg=tc.sg, su=tc.su, ss=tc.ss, stramu=tc.stramu, strahu=tc.strahu,
                       strahl=tc.strahl, z0u=jnp.exp(tc.alogu), u12=tc.u2,
                       tu=cr.tu, ts=cr.ts, tl=cr.tl, t12=cr.t12, t34=cr.t34, q12=cr.q12, q34=cr.q34, tg=cr.tg,
                       tu_end=cn.tu, ts_end=cn.ts,
                       fsena=cr.fsena, fvapa=cr.fvapa, fvapa_c2=c2[9], firb=cr.firb,
                       # the components the time step books (hydrology): the interception evaporation net of what a
                       # store's evaporation took beyond its water, the ground evaporation with its bound
                       ginvap=st.hydrology.ginvap, gsuvap=st.hydrology.gsuvap, gtrans=st.hydrology.gtrans,
                       ginvap_turvap=cr.ginvap, gsuvap_turvap=cr.gsuvap, fvapg=cr.fvapg, fseng=cr.fseng,
                       heatg=st.soil.heatg, raing=cs.raing, water=water, water0=water0, store_heat=store_heat, pflux=pflux,
                       fwetu=fw.fwetu, fwets=fw.fwets)
            return (st, k), out

        (st, k), outs = jax.lax.scan(body, (st, k), jnp.arange(24))
        return st, outs

    run_day_j = eqx.filter_jit(run_day)
    rows = []
    for d, kk in enumerate(keys):
        dw = dl[day + d]; rlds = jnp.asarray(rl[day + d]); jd = jnp.int32(julian_day(month, day + d, False))
        if d == 0:
            res.update(precip_days=[float(dl[day + e].precip[0]) for e in range(len(keys))],
                       cloud_days=[float(dl[day + e].cloud[0]) for e in range(len(keys))],
                       rsds_days=[float(mo["rsds"][day + e, 0]) for e in range(len(keys))],
                       ud_days=[float(dl[day + e].ud[0]) for e in range(len(keys))])
        st, outs = run_day_j(st, dw, rlds, jd, jax.random.PRNGKey(kk))
        rows.append({n: np.asarray(x).reshape(24, -1)[:, 0] for n, x in outs.items()})
        print(f"  day {day + d + 1} done {time.time() - t0:.0f} s", flush=True)
    for n in rows[0]:
        res[n] = np.concatenate([r[n] for r in rows])
    res["lai_end"] = np.asarray(st.vegetation.lai)[0]
    return res


# ---------------------------------------------------------------------------------------------------------------
# Figure 5.4: two days of the canopy exchange at a forest cell (Harvard Forest, 22-23 July 2001), from the release state
# ---------------------------------------------------------------------------------------------------------------

def fig_5_4():
    cache = os.environ.get("CH05_CACHE")
    if cache and Path(cache).exists():
        r = dict(np.load(cache, allow_pickle=True)); r = {k: (v.item() if v.ndim == 0 else v) for k, v in r.items()}
    else:
        r = _worked_days()
        if cache:
            np.savez(cache, **{k: np.asarray(v) for k, v in r.items()})
    from ibis.constants import PhysicalConstants
    c = PhysicalConstants()
    nh = r["ta"].size
    t = np.arange(nh + 1.0)
    st_ = lambda v: np.r_[v, v[-1]]
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True, gridspec_kw={"hspace": 0.3, "wspace": 0.28})
    K = 273.15
    ax = axs[0, 0]
    for key, lab, col, lw, ls, zo in (("ta", "air at $z_a$, $T_a$", INK2, 1.0, "--", 2),
                                      ("t12", "upper canopy air $T_{12}$", BLUE, 1.2, (0, (2.6, 1.4)), 4),
                                      ("tu", "upper leaves $T_u$", ORANGE, 1.2, "-", 3), ("ts", "upper stems $T_s$", AQUA, 1.2, "-", 2),
                                      ("t34", "lower canopy air $T_{34}$", YELLOW, 1.2, "-", 2),
                                      ("tg", "soil surface $T_g$", MAGENTA, 1.2, "-", 2)):
        ax.step(t, st_(r[key] - K), where="post", color=col, lw=lw, ls=ls, label=lab, zorder=zo)
    keys_t = ("ta", "t12", "tu", "ts", "t34", "tg")
    lo = min(r[k].min() for k in keys_t) - K; hi = max(r[k].max() for k in keys_t) - K
    ax.set_ylim(lo - 1, hi + 0.62 * (hi - lo)); ax.set_ylabel("°C")
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8, **LEG); _clean(ax)
    _title(ax, "(a) temperatures")
    ax = axs[0, 1]
    rn = r["swabs"] + r["fira"] - r["firb"]
    H = -r["fsena"]
    LE = -r["fvapa"] * c.hvap
    for v_, lab, col in ((rn, "net radiation", INK2), (H, "sensible heat $-H_a$", BLUE), (LE, "latent heat $-h_v E_a$", ORANGE),
                         (r["store_heat"], "heat stored in the canopy", AQUA), (r["heatg"], "heat into the soil $G$", YELLOW)):
        ax.step(t, st_(v_), where="post", color=col, lw=1.2 if col != INK2 else 1.3, label=lab)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_ylabel("W m$^{-2}$"); ax.legend(loc="upper left", ncol=2, columnspacing=0.8, **LEG); _clean(ax)
    lo = min(rn.min(), H.min(), r["store_heat"].min(), r["heatg"].min()); hi = max(rn.max(), LE.max())
    ax.set_ylim(lo - 30, hi + 0.55 * (hi - lo))
    _title(ax, "(b) energy fluxes of the cell")
    ax = axs[1, 0]
    mmh = 3600.0
    for v_, lab, col in ((r["gtrans"], "transpiration", BLUE), (r["ginvap"], "evaporation of intercepted water", ORANGE),
                         (r["gsuvap"], "soil evaporation", AQUA)):
        ax.step(t, st_(v_ * mmh), where="post", color=col, lw=1.2, label=lab)
    ax.axhline(0, color=INK2, lw=0.5)
    ev = np.r_[r["gtrans"], r["ginvap"], r["gsuvap"]] * mmh
    ax.set_ylim(min(ev.min(), 0) - 0.02, ev.max() * 1.5)
    ax.set_ylabel("mm h$^{-1}$"); ax.legend(loc="upper left", **LEG); _clean(ax)
    _title(ax, "(c) evapotranspiration")
    ax = axs[1, 1]
    P = np.r_[0, np.cumsum(r["rain"] * DT)]
    Gc = np.r_[0, np.cumsum(r["raing"] * DT)]
    Ei = np.r_[0, np.cumsum(r["ginvap"] * DT)]
    W = np.r_[r["water0"][0], r["water"]]
    ax.plot(t, P, color=INK2, lw=1.3, ls="--", label="precipitation")
    ax.plot(t, Gc, color=BLUE, lw=1.3, label="reaching the soil")
    ax.plot(t, Ei, color=ORANGE, lw=1.3, label="evaporated from the canopy")
    ax.plot(t, W, color=AQUA, lw=1.3, label="held on the canopy")
    ax.set_ylabel("mm since 00 h on 22 July"); ax.legend(loc="upper left", **LEG); _clean(ax)
    ax.set_ylim(0, P.max() * 1.45)
    _title(ax, "(d) precipitation and interception")
    for a in axs.ravel():
        a.set_xlim(0, nh); a.set_xticks(range(0, nh + 1, 6))
        a.set_xticklabels([f"{h % 24:02d}" for h in range(0, nh + 1, 6)])
        a.axvline(24, color=INK2, lw=0.6, ls=":")
    for a in axs[1]:
        a.set_xlabel("local solar time (h), 22-23 July 2001")
    fig.savefig(OUT / "fig_5_4_two_days.png")
    plt.close(fig)
    # numbers quoted in the text
    cmax = 300.0 / (2.0 * DT)
    capped = int(np.sum(r["ca"] >= cmax))
    print(f"5.4 cell {r['cell']} zone {r['zone']} cropfrac {r['cropfrac']:.3f}; lai {np.round(r['lai'], 3)} sai {np.round(r['sai'], 3)}"
          f" fu {r['fu']:.3f} fl {r['fl']:.3f} ztop {np.round(r['ztop'], 2)} zbot {np.round(r['zbot'], 2)}; lai at the end {np.round(r['lai_end'], 3)}")
    print(f"  frac {np.round(r['frac'], 3)}")
    print(f"  precip by day {np.round(r['precip_days'], 2)} mm; cloud {np.round(r['cloud_days'], 3)}; rsds {np.round(r['rsds_days'], 1)};"
          f" daily wind {np.round(r['ud_days'], 2)}; hourly wind range {r['ua'].min():.2f}-{r['ua'].max():.2f}")
    print(f"  za {r['za'].min():.2f}-{r['za'].max():.2f} z1 {r['z1'].mean():.2f} d {r['dispu'].mean():.2f} z0 {r['z0u'].min():.3f}-{r['z0u'].max():.3f};"
          f" c_a >= c_max in {capped} of {nh} h; c_a range {r['ca'].min():.3f}-{r['ca'].max():.3f};"
          f" cu/rho {np.round((r['cu'] / r['rhoa']).min(), 4)}-{np.round((r['cu'] / r['rhoa']).max(), 4)};"
          f" cl/rho {np.round((r['cl'] / r['rhoa']).min(), 5)}-{np.round((r['cl'] / r['rhoa']).max(), 5)};"
          f" su/rho {np.round((r['su'] / r['rhoa']).min(), 4)}-{np.round((r['su'] / r['rhoa']).max(), 4)};"
          f" sg/rho {np.round((r['sg'] / r['rhoa']).min(), 5)}-{np.round((r['sg'] / r['rhoa']).max(), 5)}")
    print(f"  strahu {r['strahu'].min():.3f}-{r['strahu'].max():.3f}; strahl {r['strahl'].min():.3f}-{r['strahl'].max():.3f}")
    for d in range(nh // 24):
        s = slice(24 * d, 24 * (d + 1))
        tr, ei, es = (r[k][s].sum() * DT for k in ("gtrans", "ginvap", "gsuvap"))
        res_e = rn[s] - H[s] - LE[s] - r["heatg"][s] - r["store_heat"][s]
        print(f"  day {d + 1}: P {r['rain'][s].sum() * DT:.2f} mm, to soil {r['raing'][s].sum() * DT:.2f}, max water held "
              f"{r['water'][s].max():.3f} mm; ET {tr + ei + es:.3f} mm = T {tr:.3f} + Ei {ei:.3f} + Es {es:.3f};"
              f" -E_a {-(r['fvapa'][s]).sum() * DT:.3f}; -E_a after cascad2 {-(r['fvapa_c2'][s]).sum() * DT:.3f};"
              f" store overshoot (turvap Ei - booked Ei) {(r['ginvap_turvap'][s] - r['ginvap'][s]).sum() * DT:.5f} mm;"
              f" ground bound (turvap Egs - booked Egs) {(r['gsuvap_turvap'][s] - r['gsuvap'][s]).sum() * DT:.5f} mm")
        print(f"    means W/m2: Rn {rn[s].mean():.1f} H {H[s].mean():.1f} LE {LE[s].mean():.1f} G {r['heatg'][s].mean():.1f}"
              f" storage {r['store_heat'][s].mean():.1f} pflux {r['pflux'][s].mean():.2f} residual {res_e.mean():.2f};"
              f" max |residual| {np.abs(res_e).max():.1f}; noon Rn {rn[24 * d + 12]:.0f} H {H[24 * d + 12]:.0f}"
              f" LE {LE[24 * d + 12]:.0f} storage {r['store_heat'][24 * d + 12]:.0f} G {r['heatg'][24 * d + 12]:.0f};"
              f" storage range {r['store_heat'][s].min():.0f} to {r['store_heat'][s].max():.0f}")
        print(f"    T_u - T_a max {np.max(r['tu'][s] - r['ta'][s]):.2f} min {np.min(r['tu'][s] - r['ta'][s]):.2f};"
              f" T_s - T_a max {np.max(r['ts'][s] - r['ta'][s]):.2f} min {np.min(r['ts'][s] - r['ta'][s]):.2f};"
              f" T_s range {r['ts'][s].min() - K:.2f}-{r['ts'][s].max() - K:.2f}; T_u range {r['tu'][s].min() - K:.2f}-{r['tu'][s].max() - K:.2f};"
              f" T_a range {r['ta'][s].min() - K:.2f}-{r['ta'][s].max() - K:.2f}; T_g range {r['tg'][s].min() - K:.2f}-{r['tg'][s].max() - K:.2f};"
              f" fwetu max {r['fwetu'][s].max():.3f}; hours with fwetu>0 {int((r['fwetu'][s] > 0).sum())}")
        night = slice(24 * d, 24 * d + 6)
        print(f"    00-06 h: H mean {H[night].mean():.0f} W/m2, storage {r['store_heat'][night].mean():.0f}, T_s - T_a mean "
              f"{np.mean(r['ts'][night] - r['ta'][night]):.2f} K; hour of max T_s - T_a {int(np.argmax(r['ts'][s] - r['ta'][s]))};"
              f" hours with rain {int((r['rain'][s] > 0).sum())}; fwetu min {r['fwetu'][s].min():.3f}; H max {H[s].max():.0f} at hour"
              f" {int(np.argmax(H[s]))}, H min {H[s].min():.0f} at hour {int(np.argmin(H[s]))}")
    print(f"  max |T_u - T_12| {np.max(np.abs(r['tu'] - r['t12'])):.2f} K (mean {np.mean(np.abs(r['tu'] - r['t12'])):.2f});"
          f" max |T_g - T_34| {np.max(np.abs(r['tg'] - r['t34'])):.2f} K (mean {np.mean(np.abs(r['tg'] - r['t34'])):.2f})")
    xs_cs = 2.0 * r["sai"][1] * r["fu"] * 2.109e5; xu_cu = 2.0 * r["lai"][1] * r["fu"] * 8436.0
    print(f"  heat capacity per unit cell area of the dry upper stems {xs_cs:.3g} and leaves {xu_cu:.3g} J m-2 K-1"
          f" ({xs_cs / 4218.0:.0f} and {xu_cu / 4218.0:.0f} kg of water)")
    W_end = r["water"][-1]
    print(f"  canopy water balance over the two days: P {P[-1]:.3f} = soil {Gc[-1]:.3f} + evaporated {Ei[-1]:.3f} + held "
          f"{W_end - W[0]:.3f} (+ residual {P[-1] - Gc[-1] - Ei[-1] - (W_end - W[0]):.4f}); held at the start {W[0]:.3f}")
    wet = r["fwetu"] > 0
    print(f"  hours with wet upper leaves: {int(wet.sum())}; transpiration in them {r['gtrans'][wet].sum() * DT:.3f} mm;"
          f" interception evaporation in them {r['ginvap'][wet].sum() * DT:.3f} mm")


# ---------------------------------------------------------------------------------------------------------------
# Figure 5.3: the components of evapotranspiration in the release hindcast, by latitude
# ---------------------------------------------------------------------------------------------------------------

def fig_5_3():
    import xarray as xr
    land = _land_mask()
    lat = np.linspace(89.75, -89.75, 360)
    V = ("ayaet", "aytrans", "ayinterc", "aysoilev", "aysnowsub", "ayprcp")
    acc = {v: np.zeros((360, 720)) for v in V}
    years = range(2001, 2011)
    for y in years:
        ds = xr.open_dataset(f"{REL}/hindcast/out/ibis_yearly_{y}.nc")
        for v in V:
            acc[v] += np.asarray(ds[v].values, float) / len(years)
        ds.close()
    sel = land & (lat[:, None] > -60.0)
    w = np.cos(np.deg2rad(lat))[:, None] * sel
    zon = {v: np.where(land.sum(1) > 0, np.nansum(acc[v] * land, 1) / np.maximum(land.sum(1), 1), np.nan) for v in V}
    glob = {v: float(np.nansum(acc[v] * w) / w.sum()) for v in V}
    wa = np.cos(np.deg2rad(lat))[:, None] * land
    glob_all = {v: float(np.nansum(acc[v] * wa) / wa.sum()) for v in V}
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"wspace": 0.34, "width_ratios": [1.3, 1]})
    ax = axs[0]
    def sm(x):                                    # three-band running mean over the bands that have land
        num = np.convolve(np.nan_to_num(x), np.ones(3), mode="same")
        den = np.convolve(np.isfinite(x).astype(float), np.ones(3), mode="same")
        return np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
    ok = land.sum(1) >= 5
    for v, lab, col in (("ayaet", "evapotranspiration", INK2), ("aytrans", "transpiration", BLUE),
                        ("ayinterc", "intercepted water", ORANGE), ("aysoilev", "soil", AQUA),
                        ("aysnowsub", "snow", YELLOW)):
        y = np.where(ok, sm(zon[v]), np.nan)
        ax.plot(lat, y, color=col, lw=1.6 if v == "ayaet" else 1.2, label=lab)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlim(-60, 90); ax.set_xticks(range(-60, 91, 30)); ax.set_ylim(-50, 1450)
    ax.set_xticklabels([f"{abs(x)}°{'S' if x < 0 else 'N' if x > 0 else ''}" for x in range(-60, 91, 30)])
    ax.set_xlabel("latitude"); ax.set_ylabel("mm yr$^{-1}$ (mean over the land cells of the band)")
    ax.legend(loc="upper right", **LEG); _clean(ax)
    _title(ax, "(a) zonal means, 2001-2010")
    ax = axs[1]
    parts = [("aytrans", "transpiration", BLUE), ("ayinterc", "intercepted water", ORANGE),
             ("aysoilev", "soil", AQUA), ("aysnowsub", "snow", YELLOW)]
    vals = [glob[v] for v, _, _ in parts]
    ax.barh(range(len(parts)), vals, color=[c_ for _, _, c_ in parts], height=0.6)
    for i, val in enumerate(vals):
        num = f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}"          # one decimal for the small snow term
        ax.text(max(val, 0) + 4, i, f"{num} mm yr$^{{-1}}$ ({100 * val / glob['ayaet']:.1f} %)", va="center",
                fontsize=7, color=INK2)
    ax.set_yticks(range(len(parts))); ax.set_yticklabels([lab for _, lab, _ in parts])
    ax.invert_yaxis(); ax.set_xlim(min(0, min(vals)) - 5, max(vals) * 1.75)
    ax.set_xlabel("mm yr$^{-1}$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _title(ax, f"(b) land north of 60°S: {glob['ayaet']:.0f} mm yr$^{{-1}}$")
    fig.savefig(OUT / "fig_5_3_et_components.png")
    plt.close(fig)
    d = acc["ayaet"] - (acc["aytrans"] + acc["ayinterc"] + acc["aysoilev"] + acc["aysnowsub"])
    print("5.3 land north of 60S (area-weighted, 2001-2010, mm/yr)", {k: round(v, 1) for k, v in glob.items()},
          "shares %", {v: round(100 * glob[v] / glob["ayaet"], 1) for v in ("aytrans", "ayinterc", "aysoilev", "aysnowsub")},
          "ET/P", round(glob["ayaet"] / glob["ayprcp"], 3))
    print("    all land", {k: round(v, 1) for k, v in glob_all.items()}, "max |ET - sum of parts| (mm/yr)",
          float(np.nanmax(np.abs(d[land]))))
    for lo_, hi_ in ((-60, -15), (-15, 15), (15, 45), (45, 60), (60, 90)):
        b = sel & (lat[:, None] >= lo_) & (lat[:, None] < hi_)
        wb = np.cos(np.deg2rad(lat))[:, None] * b
        g = {v: float(np.nansum(acc[v] * wb) / wb.sum()) for v in V}
        print(f"    {lo_} to {hi_}: ET {g['ayaet']:.0f}, shares T/I/Es/S %",
              [round(100 * g[k] / g["ayaet"], 1) for k in ("aytrans", "ayinterc", "aysoilev", "aysnowsub")])
    sub = acc["aysnowsub"]
    neg = (sub < 0) & sel
    low = np.argsort(np.where(sel, sub, np.inf).ravel())[:25]                    # the 25 lowest cells
    il, jl = np.unravel_index(low, sub.shape)
    print(f"    cells north of 60S with a negative 10-year mean snow sublimation: {int(neg.sum())} of {int(sel.sum())},"
          f" below -5 mm/yr {int((sel & (sub < -5)).sum())}; with a negative soil evaporation: "
          f"{int(((acc['aysoilev'] < 0) & sel).sum())}")
    print(f"    area-weighted mean of the positive and of the negative cell values of the snow sublimation:"
          f" {np.nansum(np.where(sub > 0, sub, 0) * w) / w.sum():.2f} and {np.nansum(np.where(sub < 0, sub, 0) * w) / w.sum():.2f}"
          f" mm/yr; the 25 lowest cells: {sub[il[0], jl[0]]:.1f} to {sub[il[-1], jl[-1]]:.1f} mm/yr, precipitation"
          f" {acc['ayprcp'][il, jl].min():.0f}-{acc['ayprcp'][il, jl].max():.0f} mm/yr, at (lat, lon)",
          sorted((round(float(lat[a]), 2), round(-179.75 + 0.5 * float(b), 2)) for a, b in zip(il, jl)))


# ---------------------------------------------------------------------------------------------------------------
# Numbers of the text that no figure shows
# ---------------------------------------------------------------------------------------------------------------

def checks():
    """The reference height, the cells without canopy surfaces and the limit of the coefficient above the canopy, on the
    release state at the end of 2000 (fu, fl, stem area, heights, snow) and the leaf area at the end of July 2001."""
    import jax.numpy as jnp
    import xarray as xr
    from ibis.constants import PhysicalConstants
    from ibis.physics.canopy_exchange import canini, turcof, CanopyParams
    c = PhysicalConstants(); cp = CanopyParams()
    st = _release_state(None)
    land = _land_mask()
    lat = np.repeat(np.linspace(89.75, -89.75, 360)[:, None], 720, 1)[land]
    w = np.cos(np.deg2rad(lat)); n60 = lat > -60.0
    aw = lambda m, sel=n60: float(np.sum(w * m * sel) / np.sum(w * sel))
    v = st.vegetation
    ztop = np.asarray(v.ztop, float); zbot = np.asarray(v.zbot, float); fu = np.asarray(v.fu, float); fl = np.asarray(v.fl, float)
    lai = np.asarray(v.lai, float); sai = np.asarray(v.sai, float); fi = np.asarray(st.snow.fi, float)
    z0soi = np.asarray(st.soil.z0soi, float)
    ds = xr.open_dataset(f"{REL}/hindcast/out/ibis_monthly_2001_07.nc")
    assert np.max(np.abs(np.asarray(ds.fu.values, float)[land] - fu)) == 0.0      # the same cells, in the same order
    laiu7 = np.asarray(ds.laiu.values, float)[land]; lail7 = np.asarray(ds.lail.values, float)[land]
    print("checks: z0soi", np.unique(z0soi), "z0sno", float(st.snow.z0sno), "zbot_u", np.unique(zbot[:, 1]))
    # reference height
    za = lambda T, q=0.0: (1 - 0.999) / 0.999 * c.rair * T * (1 + (c.rvap / c.rair - 1) * q) / c.grav
    print("  za(q=0): 273.15 K %.3f m, 288.15 K %.3f m, 303.15 K %.3f m; per K %.4f m" % (za(273.15), za(288.15), za(303.15),
                                                                                        za(289.15) - za(288.15)))
    hi = ztop[:, 1] + 1.0 > za(288.15)
    print("  ztop_u percentiles 0/5/25/50/75/95/99/100:", np.percentile(ztop[:, 1], [0, 5, 25, 50, 75, 95, 99, 100]).round(2),
          "; cells with ztop_u = 4 m:", int(np.sum(ztop[:, 1] == 4.0)), "; > 50 m:", int(np.sum(ztop[:, 1] > 50)),
          "; > 100 m:", int(np.sum(ztop[:, 1] > 100)))
    print("  ztop_u + 1 > za(288.15 K): %d of %d cells; area share north of 60S %.1f %%; with fu > 0.5: %d of %d" % (
        hi.sum(), hi.size, 100 * aw(hi), int((hi & (fu > 0.5)).sum()), int((fu > 0.5).sum())))
    # cells without canopy surfaces (the bypass of the canopy rows)
    xu = 2 * lai[:, 1] * fu; xs = 2 * sai[:, 1] * fu; xl = 2 * (lai[:, 0] + sai[:, 0]) * fl * (1 - fi)
    bare = (xu + xs + xl) < 1e-7
    print("  no canopy surface at the end of 2000: %d cells (%d south of 60S, %d fully snow-covered, %d with fu = 0)" % (
        bare.sum(), (bare & ~n60).sum(), int((fi[bare] >= 1).sum()), int((fu[bare] == 0).sum())))
    bare7 = (2 * laiu7 * fu + 2 * sai[:, 1] * fu + 2 * (lail7 + sai[:, 0]) * fl) < 1e-7
    print("  no canopy surface with the July 2001 leaf area and no snow: %d cells (%d south of 60S)" % (
        bare7.sum(), (bare7 & ~n60).sum()))
    # neutral c_a against c_max, with the July 2001 leaf area and no snow
    n = lat.size
    J = lambda x: jnp.asarray(np.broadcast_to(np.asarray(x, np.float32), (n,)))
    ztl = np.minimum(np.maximum(0.25, 0.25 * lail7), zbot[:, 1] - 0.5)                  # (10.13)
    LAI = jnp.stack([J(lail7), J(laiu7)], 1); SAI = jnp.stack([J(sai[:, 0]), J(sai[:, 1])], 1)
    ZT = jnp.stack([J(ztl), J(ztop[:, 1])], 1); ZB = jnp.stack([J(0.05), J(zbot[:, 1])], 1)
    ta, qa, ps = 288.15, 0.008, 1.0e5
    cmax = 300.0 / (2.0 * DT)
    z = jnp.zeros((n,)); o = jnp.ones((n,))
    print("  neutral, T 288.15 K, q 0.008, p 1000 hPa, July 2001 leaf area, no snow: area shares north of 60S with fu > 0.5:"
          " %.1f %%, fu < 0.1: %.1f %%" % (100 * aw(fu > 0.5), 100 * aw(fu < 0.1)))
    for u in (2.5, 4.12, 10.0):
        a = canini(J(ta), J(qa), J(ps), J(u), J(0.0), J(z0soi), 0.0005, J(fl), J(fu), LAI, SAI, ZT, ZB,
                   cp.alaiml, cp.alaimu, c.grav, c.rair, c.rvap, c.cair, c.cvap, c.cappa)
        t12 = J(ta) * a.tfac
        tc = turcof(a, J(ta), J(qa), J(u), t12, t12, J(qa), J(qa), J(0.0), c.vonk, DT, cp.cleaf, cp.cstem, cp.cgrass,
                    cp.dleaf, cp.dstem, 0, z, o, o, z, o, o, c.grav)
        ca = np.asarray(J(u) * tc.strahu * c.vonk ** 2 / ((a.aloga - tc.alogu) * (a.aloga - a.alog1)))
        lim = ca >= cmax
        z0 = np.exp(np.asarray(tc.alogu))
        print("  u = %5.2f m/s: c_a >= c_max on %.1f %% of the land north of 60S; fu > 0.5: %.1f %%; fu < 0.1: %.1f %%;"
              " median c_a %.3f m/s; median z0 %.3f m (fu > 0.5: %.2f m, median d %.2f m); median c_u/rho %.4f m/s" % (
                  u, 100 * aw(lim), 100 * aw(lim, n60 & (fu > 0.5)), 100 * aw(lim, n60 & (fu < 0.1)), np.median(ca[n60]),
                  np.median(z0[n60]), np.median(z0[n60 & (fu > 0.5)]), float(np.median(np.asarray(a.dispu)[n60 & (fu > 0.5)])),
                  float(np.median(np.asarray(tc.cu / a.rhoa)[n60]))))


# ---------------------------------------------------------------------------------------------------------------
# The split of the snow-surface evaporation between soil evaporation and snow sublimation in the ET components
# ---------------------------------------------------------------------------------------------------------------

def snowsub(n=150, seed=5, year=2001):
    """One year of the parameter file PARAMS at a random sample of land cells north of 30N, from the release state at
    the end of 2000, with the snow sublimation booked in both ways: with the snow cover f_i at the START of the step, with
    which the surface fluxes of the step are computed (the current code, `et_parts_step`; technote review D44), and with
    the cover at the END of the step (the release code e54e342). A wrapper of timestep_body adds E_i times the difference
    of the two covers times dt to the snow-sublimation accumulator, so that it holds the form the code does not book;
    the booked form then follows from ayaet - trans - interc - soilev, which the wrapper does not touch. On the release
    code the same function gives the comparison the other way round."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import xarray as xr
    from ibis.constants import PhysicalConstants
    from ibis.io.gswp3_reader import GSWP3Reader
    import ibis.timestepping.loops_gswp3 as LG
    consts = PhysicalConstants()
    t0 = time.time()
    land = _land_mask()
    flat = np.where(land.ravel())[0]
    rows, cols = np.divmod(flat, 720)
    rng = np.random.default_rng(seed)
    cand = np.where(89.75 - 0.5 * rows > 30.0)[0]
    sel = np.sort(rng.choice(cand, n, replace=False))
    I, J = rows[sel], cols[sel]
    st = _release_state(sel)
    params, lat_rad, cf, zones = _cell_params(I, J, year)
    orig = LG.timestep_body
    current = hasattr(LG, "et_parts_step")            # the current code books the start-of-step cover (D44)

    def wrapped(s, consts_, params_, dtime, **kw):
        fi0 = s.snow.fi
        new = orig(s, consts_, params_, dtime, **kw)
        nst = new[0] if isinstance(new, tuple) else new
        dfi = (nst.snow.fi - fi0) if current else (fi0 - nst.snow.fi)
        dsub = (nst.canopy.fvapi * dfi * dtime).astype(nst.etparts.snowsub.dtype)
        nst = eqx.tree_at(lambda q: q.etparts.snowsub, nst, nst.etparts.snowsub + dsub)
        return (nst,) + tuple(new[1:]) if isinstance(new, tuple) else nst
    LG.timestep_body = wrapped
    try:
        e0 = {k: np.asarray(getattr(st.etparts, k), float) for k in ("trans", "interc", "soilev", "snowsub")}
        a0 = np.asarray(st.summary.ayaet, float)
        reader = GSWP3Reader(GSWP3)
        key = jax.random.PRNGKey(2001)
        for m in range(12):
            st, key, _ = LG.run_month_gswp3(st, consts, params, reader, I, J, np.asarray(lat_rad), year, m, key, DT, leap=False)
            print(f"  snowsub: month {m + 1} done {time.time() - t0:.0f} s", flush=True)
    finally:
        LG.timestep_body = orig
    e = {k: np.asarray(getattr(st.etparts, k), float) - e0[k] for k in e0}
    aet = np.asarray(st.summary.ayaet, float) - a0
    booked = aet - e["trans"] - e["interc"] - e["soilev"]     # the form the code books
    if current:
        sub_start, sub_end = booked, e["snowsub"]              # sums of E_i f_i,start dt and of E_i f_i,end dt
    else:
        sub_start, sub_end = e["snowsub"], booked
    print(f"snowsub: the code books the cover at the {'start' if current else 'end'} of the step; parameters {PARAMS}")
    lat = 89.75 - 0.5 * I
    wc = np.cos(np.deg2rad(lat))
    ds = xr.open_dataset(f"{REL}/hindcast/out/ibis_yearly_{year}.nc")
    rel = {k: np.asarray(ds[k].values, float)[I, J] for k in ("ayaet", "aysnowsub", "aysoilev")}
    m_ = lambda x: float(np.sum(wc * x) / np.sum(wc))
    print(f"snowsub: {n} cells north of 30N, {year}; area-weighted means (mm/yr): ET {m_(aet):.1f} (release output {m_(rel['ayaet']):.1f});"
          f" soil + snow evaporation {m_(e['soilev'] + booked):.2f}; sublimation with the end-of-step cover {m_(sub_end):.2f}"
          f" (release output {m_(rel['aysnowsub']):.2f}); with the start-of-step cover {m_(sub_start):.2f};"
          f" difference {m_(sub_end - sub_start):.2f}")
    snowy = np.abs(sub_start) > 1.0
    print(f"  cells with |sublimation| > 1 mm/yr: {int(snowy.sum())}; their median relative difference "
          f"{np.median((sub_end - sub_start)[snowy] / np.abs(sub_start[snowy])) * 100:.1f} %; 10-90 % range "
          f"{np.percentile((sub_end - sub_start)[snowy] / np.abs(sub_start[snowy]) * 100, [10, 90]).round(1)} %;"
          f" largest |difference| {np.max(np.abs(sub_end - sub_start)):.2f} mm/yr")
    print(f"  done {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------------------------------------------
# The energy-test variants of the transfer above the canopy (technote review D5): what they do to c_u
# ---------------------------------------------------------------------------------------------------------------

def variants():
    """c_u / rho_a of the forest and the grassland of Figure 5.2b (neutral, first iteration) with IBIS 2.6b4's limit on
    c_a and reference level (canopy.ca_cap = 1, canopy.za_above_top = 1 m: the defaults), without the limit
    (ca_cap = 0), and without the limit with the reference level 10 m above the canopy top (za_above_top = 10 m, an
    example value), over the range of the generator's hourly wind and at its mean (section 3.6)."""
    U = np.array([2.5, 4.12, 10.0])
    for lab, kw in (("forest, L_u = 5", dict(FOREST, lai_u=5.0, sai_u=1.25)), ("grassland, L_l = 2", dict(GRASS, lai_l=2.0))):
        for cap, zat in ((1.0, 1.0), (0.0, 1.0), (0.0, 10.0), (1.0, 10.0)):
            a, tc, ca = _aero(U, za_above_top=zat, ca_cap=cap, **kw)
            print(f"variants {lab}: ca_cap {cap:g}, za_above_top {zat:g} m: z_a {float(a.za[0]):.2f} m; c_a before the"
                  f" limit {np.round(ca, 3)} m/s; c_u/rho_a {np.round(np.asarray(tc.cu / a.rhoa), 4)} m/s at u_a = {U} m/s")


# ---------------------------------------------------------------------------------------------------------------
# How often the bound on the ground evaporation and the overshoot of the canopy stores act, and the cells with a
# negative yearly soil evaporation
# ---------------------------------------------------------------------------------------------------------------

#: one global test year (2001) of the revised code with the decided parameters (params_32a_base.yaml, with
#: soil.evap_needs_store) and physics.water_balance on, from a copy of the release hindcast's state at the end of 2000
#: (tower-analysis step of the technote review; code b2d2cca, whose physics is that of this checkout)
WBAL_TEST = "/shared/nas2/Hangkai/ibis/verify_outputs/step2_wbal_2001_d25b/out/ibis_yearly_2001.nc"


def wbal(path=WBAL_TEST, year=2001):
    """The counters of the water-balance module (physics.water_balance) in the test year: steps on which the bound on
    the ground evaporation acted (n_bound), on which a canopy store's evaporation overshot its water (n_cdef, with the
    water ay_w_cdef) and with ground snowfall beyond what covers the snow-free part (n_snow); and the land cells with a
    negative yearly soil evaporation (aysoilev) in the test year and in the release run's output of the same year."""
    import xarray as xr
    land = _land_mask()
    lat = np.repeat(np.linspace(89.75, -89.75, 360)[:, None], 720, 1)
    lon = np.repeat(np.linspace(-179.75, 179.75, 720)[None, :], 360, 0)
    w = np.cos(np.deg2rad(lat)) * land
    n60 = land & (lat > -60.0)
    w60 = np.cos(np.deg2rad(lat)) * n60
    aw = lambda x, ww=w60: float(np.nansum(np.nan_to_num(x) * ww) / ww.sum())
    with xr.open_dataset(path) as ds:
        g = {k: np.asarray(ds[k].values, float) for k in ("n_bound", "ay_w_bound", "n_snow", "ay_w_snow", "n_cdef",
                                                          "ay_w_cdef", "aysoilev", "aysnowsub", "ayaet")}
    print(f"wbal: test year {year} ({path}); {int(land.sum())} land cells, {int(n60.sum())} north of 60S")
    for n in ("bound", "snow", "cdef"):
        cnt, wat = g["n_" + n], g["ay_w_" + n]
        on = land & (np.nan_to_num(cnt) > 0)
        print(f"  n_{n}: {int(on.sum())} cells with at least one step; {np.nansum(cnt[land]):.0f} steps in all;"
              f" median {np.median(cnt[on]) if on.any() else 0:.0f} steps in those cells; ay_w_{n} area mean north of 60S"
              f" {aw(wat):.5f} mm/yr (all land {aw(wat, w):.5f}), largest cell {np.nanmax(np.abs(wat[land])):.3f} mm/yr")
    with xr.open_dataset(f"{REL}/hindcast/out/ibis_yearly_{year}.nc") as ds:
        rel = np.asarray(ds["aysoilev"].values, float)
    for lab, se in (("test year", g["aysoilev"]), ("release run", rel)):
        neg = land & (se < 0)
        k = np.nanargmin(np.where(land, se, np.inf))
        i, j = np.unravel_index(k, se.shape)
        print(f"  {lab}: {int(neg.sum())} land cells with a negative yearly soil evaporation ({int((neg & (lat >= 60)).sum())}"
              f" north of 60N, {int((neg & (lat < -60)).sum())} south of 60S); lowest {se[i, j]:.2f} mm/yr at"
              f" {lat[i, j]:.2f}N {lon[i, j]:.2f}E")
    lo = land & (g["aysoilev"] < 0)
    print(f"  test year, those cells: snow sublimation {np.nanmin(g['aysnowsub'][lo]):.1f} to"
          f" {np.nanmax(g['aysnowsub'][lo]):.1f} mm/yr; negative in both years: {int((lo & (rel < 0)).sum())}")


FIGURES = {"5.1": fig_5_1, "5.2": fig_5_2, "5.3": fig_5_3, "5.4": fig_5_4}
EXTRA = {"checks": checks, "snowsub": snowsub, "variants": variants, "wbal": wbal}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        (FIGURES.get(key) or EXTRA[key])()
        print(key, "done", flush=True)
