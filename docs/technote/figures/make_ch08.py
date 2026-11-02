"""Figures of Chapter 8 (Snow) of the IBIS 3.0 technical note, drawn from the model's code and the release run.

Run from the repository root with the project interpreter (CPU is enough):

    JAX_PLATFORMS=cpu /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch08.py          # all figures
    JAX_PLATFORMS=cpu /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch08.py 8.2 8.3  # some

Every number a figure shows is computed here: the snow constants are read from `ibis.state.snow.SnowState`, the
response curves are the model's own functions (`snow_coverage`, `solsur`), the maps read the snow state of the release
run's restart of 31 December 2001 (final_rel30), and Figure 8.4 runs the model, with the code of the checkout the
script sits in, at one cell through 2001 from the release's state of 1 January 2001 with the release parameters (about
5 minutes on one CPU core) and prints the numbers quoted in section 8.15. The option `numbers` prints those of section
8.14 (release state of 31 December 2001, water balance of the deep packs over 2002-2010, and the single/double precision
test of the cover relation and re-division, under the release's re-division of every step and under the current
deep-pack rule). The precision test runs each precision in a process of its own (options `precision32`, with 64-bit
types disabled as in the release run, and `precision64`). The diagnostic-year numbers of section 8.14 come from
`make_ch07.py diagyear`. Figures 8.1-8.3 were first drawn with the release code and redrawn on 2026-11-02 with the code
of branch review-technote (byte-identical to the first drawing); Figure 8.4 was redrawn on 2026-11-02 with the code of
branch review-technote.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: E402  (also sets rcParams)

import matplotlib                                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                     # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm                          # noqa: E402
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch                  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

REL = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30"
PARAMS = str(ROOT / "python-port/configs/params/params_31s_ch4h.yaml")
GSWP3 = "/shared/nas2/Min/GSWP3"
KOPPEN = "/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz"
CROPFRAC = "/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz"
NPOI = 95333

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# sequential ramp of the maps: one hue, evenly stepped in OKLab lightness (adjacent steps about 8 apart) through the
# house blue #2a78d6; its lightest step stands clear of the grey of snow-free land (LAND_BG)
BLUES = ("#a9c9ee", "#84b1e8", "#5f98e0", "#377fd8", "#2468bb", "#1d5298", "#153e75", "#0e2a55")


def _snow_constants():
    from ibis.state.snow import create_snow_state
    s = create_snow_state(1, 3)
    return {k: float(getattr(s, k)) for k in ("z0sno", "rhos", "consno", "hsnotop", "hsnomin", "fimin", "fimax")}


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def _load_restart(year):
    from ibis.dimensions import GridConfig
    from ibis.state.model_state import create_ibis_state
    from ibis.io.restart import load_state
    tmpl = create_ibis_state(GridConfig(npoi=NPOI, nbedrock=3), 296e-6, 0.209,
                             with_stress=True, with_etparts=True, with_seasonal=True)
    st, _ = load_state(f"{REL}/hindcast/restart/ibis_restart_{year}.npz", tmpl)
    return st


# ---------------------------------------------------------------------------------------------------------------
# Figure 8.1: the snow model of a grid cell
# ---------------------------------------------------------------------------------------------------------------

def fig_8_1():
    c = _snow_constants()
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    ax.set_xlim(0, 112); ax.set_ylim(0, 72); ax.axis("off")
    xs0, xs1, xg1 = 24.0, 60.0, 84.0                         # snow-covered part, snow-free part
    yg = 20.0                                                # ground surface
    hb = 4.0                                                 # buried-vegetation layer (drawn thick, not to scale)
    lay = [(yg + hb, 8.0), (yg + hb + 8.0, 8.0), (yg + hb + 16.0, 4.0)]    # (bottom, height) of snow layers 3, 2, 1
    ytop = yg + hb + 20.0
    # upper canopy (drawn lightly: Chapter 5)
    for xc in np.linspace(xs0 + 3, xg1 - 3, 7):
        ax.add_patch(Rectangle((xc - 0.3, ytop + 14.0), 0.6, 6.0, fc="#b9a58f", ec="none"))
        ax.add_patch(Ellipse((xc, ytop + 22.5), 7.0, 6.0, fc="#cfe0c4", ec="none"))
    ax.text(xg1 + 1.5, ytop + 22.5, "upper canopy\n(Ch. 5)", fontsize=7.2, color=INK2, va="center")
    # soil
    ax.add_patch(Rectangle((xs0, yg - 6.0), xg1 - xs0, 6.0, fc="#c9a77c", ec="white", lw=0.8))
    ax.add_patch(Rectangle((xs0, yg - 11.0), xg1 - xs0, 5.0, fc="#dcc4a3", ec="white", lw=0.8))
    ax.text(xs0 - 1.0, yg - 3.0, "soil layer 1, 0.10 m", ha="right", va="center", fontsize=7.2, color=INK2)
    ax.text(xs0 - 1.0, yg - 8.5, "soil layers 2–6 (Ch. 6)", ha="right", va="center", fontsize=7.2, color=INK2)
    # buried lower vegetation and the snow layers
    ax.add_patch(Rectangle((xs0, yg), xs1 - xs0, hb, fc="#a9c98a", ec="white", lw=0.8, hatch="////"))
    for (yb, hh), fc in zip(lay, ("#e3edf7", "#edf3fa", "#f7fafd")):
        ax.add_patch(Rectangle((xs0, yb), xs1 - xs0, hh, fc=fc, ec="#9fb6cc", lw=0.6))
    labels = ["$h_3 = (H - h_1)/2$", "$h_2 = (H - h_1)/2$", f"$h_1 = h_{{top}}$ = {c['hsnotop']:g} m"]
    for (yb, hh), lab, tl in zip(lay, labels, ("$T_3$", "$T_2$", "$T_1$")):
        ax.text(xs0 - 1.0, yb + hh / 2, lab, ha="right", va="center", fontsize=7.2, color=INK2)
        ax.text((xs0 + xs1) / 2, yb + hh / 2, tl, ha="center", va="center", fontsize=8, color=INK)
    ax.text(xs0 - 1.0, ytop, "snow surface, $T_i$ (Ch. 5)", ha="right", va="center", fontsize=7.2, color=INK2)
    ax.text(xs0 - 1.0, yg + hb / 2, "buried lower vegetation", ha="right", va="center", fontsize=7.2, color=INK2)
    ax.text((xs0 + xs1) / 2, yg + hb / 2, "$T_b$", ha="center", va="center", fontsize=8, color=INK,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # conductances at the interfaces, (8.2) and (8.7)
    for yy, lab in ((ytop, "$k_i$"), (yg + hb + 16.0, "$k_{12}$"), (yg + hb + 8.0, "$k_{23}$"), (yg + hb, "$k_{3b}$"),
                    (yg, "$k_{bs}$")):
        ax.text(xs1 - 1.0, yy, lab, ha="right", va="center", fontsize=6.8, color=INK2,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    # the skin temperature and the fluxes at the top
    ax.plot([xs0, xs1], [ytop, ytop], color="#5e7f9e", lw=1.2)
    arr = dict(arrowstyle="-|>", mutation_scale=7, lw=0.9)
    for xa in (33.0, 36.0):
        ax.add_patch(FancyArrowPatch((xa, ytop + 7.5), (xa, ytop + 0.3), color="#4d7fb5", **arr))
    ax.text(31.5, ytop + 8.2, "rain, snow\n$P_{r,l}$, $P_{s,l}$", ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.add_patch(FancyArrowPatch((43.0, ytop + 0.3), (43.0, ytop + 7.5), color="#4d7fb5", **arr))
    ax.text(43.0, ytop + 8.2, "sublimation\n$E_i$", ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.add_patch(FancyArrowPatch((53.0, ytop + 7.5), (53.0, ytop + 0.3), color="#b0652f", **arr))
    ax.text(53.0, ytop + 8.2, "heat\n$F_{top}$, (8.4)", ha="center", va="bottom", fontsize=6.8, color=INK2)
    # the snow-free part: lower canopy, snowfall that extends the cover
    for xg in np.arange(xs1 + 0.8, xg1 - 0.2, 0.9):
        ax.plot([xg, xg + 0.3], [yg, yg + 5.5 + 0.8 * np.sin(xg)], color="#7fa64a", lw=0.8)
    ax.add_patch(Rectangle((xs1, yg), 4.0, ytop - yg, fc="none", ec="#7f9bb8", lw=0.8, ls=(0, (3, 2))))
    for xa in (75.0, 80.0):
        ax.add_patch(FancyArrowPatch((xa, ytop + 7.5), (xa, yg + 7.2), color="#4d7fb5", **arr))
    ax.text(77.5, ytop + 8.2, "snow reaching\nthe ground $P_{s,g}$", ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.add_patch(FancyArrowPatch((73.0, yg + 17.0), (64.4, yg + 17.0), color=INK2, **arr))
    ax.text(65.0, yg + 15.8, "new cover\n$\\Delta f_i$, (8.10)", ha="left", va="top", fontsize=6.8, color=INK2)
    ax.text(xg1 + 1.5, yg + 3.0, "lower canopy", ha="left", va="center", fontsize=7.2, color=INK2)
    # fluxes into the soil: beneath the pack, and on the snow-free part (8.21)-(8.22)
    ax.add_patch(FancyArrowPatch((32.0, yg - 0.2), (32.0, yg - 5.0), color="#b0652f", **arr))
    ax.text(32.8, yg - 3.6, "$F_s$, (8.9)", fontsize=6.8, color=INK, va="center")
    ax.add_patch(FancyArrowPatch((46.0, yg - 0.2), (46.0, yg - 5.0), color="#4d7fb5", **arr))
    ax.text(46.8, yg - 3.6, "meltwater at $T_f$", fontsize=6.8, color=INK, va="center")
    ax.add_patch(FancyArrowPatch((69.0, yg - 0.2), (69.0, yg - 5.0), color="#b0652f", **arr))
    ax.add_patch(FancyArrowPatch((71.0, yg - 0.2), (71.0, yg - 5.0), color="#4d7fb5", **arr))
    ax.text(71.8, yg - 3.6, "$G_g$, $R_g$", fontsize=6.8, color=INK, va="center")
    # fractions of the cell
    for x0, x1, lab in ((xs0, xs1, "$f_i$"), (xs1, xg1, "$1 - f_i$")):
        ax.annotate("", (x0, 6.0), (x1, 6.0), arrowprops=dict(arrowstyle="|-|", color=INK2, lw=0.7, mutation_scale=3))
        ax.text((x0 + x1) / 2, 4.6, lab, ha="center", va="top", fontsize=8, color=INK)
    ax.text(xs0, 0.3, "not to scale", fontsize=6.5, color=INK2, style="italic")
    # the constants
    txt = (f"$\\rho_s$ = {c['rhos']:g} kg m$^{{-3}}$\n$\\lambda_s$ = {c['consno']:g} W m$^{{-1}}$ K$^{{-1}}$\n"
           f"$h_{{min}}$ = {c['hsnomin']:g} m\n$H = h_{{min}} + (h_{{ruf}} - h_{{min}})\\,f_i$\nwhile $f_i$ < {c['fimax']:g}, (8.13)")
    ax.text(xg1 + 1.5, yg + 12.0, txt, fontsize=6.9, color=INK2, va="center", linespacing=1.45)
    fig.savefig(OUT / "fig_8_1_snow_column.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 8.2: response curves of the cover relation and of the snow albedo
# ---------------------------------------------------------------------------------------------------------------

def fig_8_2():
    import jax.numpy as jnp
    from ibis.physics.snow_physics import snow_coverage
    from ibis.physics.radiation import solsur
    from ibis.constants import PhysicalConstants
    c = _snow_constants(); k = PhysicalConstants()
    w = np.linspace(0.0, 260.0, 1301)[1:]                            # cell-mean snow water equivalent, kg m-2
    v = w / c["rhos"]                                                # cell-mean snow volume, m
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 6.2), gridspec_kw={"hspace": 0.48, "wspace": 0.3})
    rufs = (c["hsnomin"] + 0.05, 0.45, 0.70)
    for ruf, col in zip(rufs, (BLUE, ORANGE, AQUA)):
        # a pack of volume v given to the model's cover relation as full cover (fi = 1, thickness v)
        h = np.repeat((v / 3.0)[:, None], 3, axis=1)
        fi, hn = snow_coverage(jnp.asarray(h), jnp.ones(v.shape), c["hsnomin"], jnp.full(v.shape, ruf), c["fimax"], k.epsilon)
        fi = np.asarray(fi); H = np.asarray(hn).sum(1)
        assert np.allclose(fi * H, v, rtol=1e-4)
        lab = f"$h_{{ruf}}$ = {ruf:.2f} m" + (", the floor of (8.14)" if ruf == rufs[0] else "")
        axs[0, 0].plot(w, fi, color=col, lw=1.6, label=lab)
        axs[0, 1].plot(w, H, color=col, lw=1.6, label=lab)
    axs[0, 0].set_ylim(0, 1.05); axs[0, 0].set_ylabel("snow cover $f_i$")
    axs[0, 0].set_title("(a) cover", loc="left", fontsize=9, weight="bold", color=INK)
    axs[0, 0].legend(frameon=False, fontsize=6.8, loc="lower right")
    axs[0, 1].set_ylim(0, 1.15); axs[0, 1].set_ylabel("thickness on the covered part $H$ (m)")
    axs[0, 1].axhline(c["hsnomin"], color=INK2, lw=0.7, ls=":")
    axs[0, 1].text(255, c["hsnomin"] + 0.02, f"$h_{{min}}$ = {c['hsnomin']:g} m", ha="right", va="bottom", fontsize=6.8, color=INK2)
    axs[0, 1].set_title("(b) thickness", loc="left", fontsize=9, weight="bold", color=INK)
    axs[0, 1].legend(frameon=False, fontsize=6.8, loc="upper left")
    for a in axs[0]:
        a.set_xlim(0, 260); a.set_xlabel("cell-mean snow water equivalent $\\rho_s f_i H$ (kg m$^{-2}$)")
    # snow albedo against the top-layer temperature, and against the cosine of the zenith angle
    tt = np.linspace(-25.0, 0.0, 251) + k.tmelt
    one = np.ones(tt.shape)
    for ib, col, nm in ((0, BLUE, "visible"), (1, ORANGE, "near-infrared")):
        _, _, ad, ai = solsur(jnp.asarray(0.6 * one), jnp.zeros(tt.shape), jnp.zeros(tt.shape), jnp.asarray(tt),
                              jnp.zeros(tt.shape), jnp.zeros(tt.shape), k.tmelt, jnp.ones(tt.shape, bool), ib)
        assert np.allclose(np.asarray(ad), np.asarray(ai))           # no zenith-angle term at cos z >= 0.5
        axs[1, 0].plot(tt - k.tmelt, np.asarray(ai), color=col, lw=1.6, label=nm)
    axs[1, 0].set_xlim(-25, 0); axs[1, 0].set_ylim(0.1, 1.02)
    axs[1, 0].set_xlabel("top-layer temperature $T_1$ (°C)"); axs[1, 0].set_ylabel("snow albedo")
    axs[1, 0].set_title("(c) diffuse albedo (direct for $\\mu$ ≥ 0.5)", loc="left", fontsize=9, weight="bold", color=INK)
    axs[1, 0].legend(frameon=False, fontsize=6.8, loc="lower left")
    mu = np.linspace(0.005, 1.0, 400)
    onem = np.ones(mu.shape)
    for ib, col, nm in ((0, BLUE, "visible"), (1, ORANGE, "near-infrared")):
        for tsn, ls, tl in ((k.tmelt - 15.0, "-", "≤ −15 °C"), (k.tmelt, (0, (4, 2)), "= 0 °C")):
            _, _, ad, _ = solsur(jnp.asarray(mu), jnp.zeros(mu.shape), jnp.zeros(mu.shape), jnp.asarray(tsn * onem),
                                 jnp.zeros(mu.shape), jnp.zeros(mu.shape), k.tmelt, jnp.ones(mu.shape, bool), ib)
            axs[1, 1].plot(mu, np.asarray(ad), color=col, lw=1.4, ls=ls, label=f"{nm}, $T_1$ {tl}")
    axs[1, 1].set_xlim(0, 1); axs[1, 1].set_ylim(0.1, 1.02)
    axs[1, 1].set_xlabel("cosine of the solar zenith angle $\\mu$"); axs[1, 1].set_ylabel("snow albedo, direct beam")
    axs[1, 1].set_title("(d) direct-beam albedo", loc="left", fontsize=9, weight="bold", color=INK)
    axs[1, 1].legend(frameon=False, fontsize=6.6, loc="lower right", ncol=1)
    for a in axs.ravel():
        _clean(a)
    fig.savefig(OUT / "fig_8_2_cover_albedo.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 8.3: the snow of the release run on 31 December 2001
# ---------------------------------------------------------------------------------------------------------------

def fig_8_3():
    land = _land_mask()
    ii, jj = np.nonzero(land)
    st = _load_restart(2001)
    sn = st.snow
    fi = np.asarray(sn.fi, np.float64); H = np.asarray(sn.hsno, np.float64).sum(1)
    swe = float(sn.rhos) * fi * H
    grid_fi = np.full(land.shape, np.nan); grid_fi[ii, jj] = np.where(fi > 0, fi, np.nan)
    grid_w = np.full(land.shape, np.nan); grid_w[ii, jj] = np.where(fi > 0, swe, np.nan)
    # W = 2,500 kg m-2 is a pack of 10 m at full cover; every pack thicker than 10 m is fully covered, so the two
    # darkest classes of (b) are exactly the cells with H > 10 m
    wdeep = 2500.0
    assert int((swe > wdeep).sum()) == int((H > 10).sum())
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 8.4), gridspec_kw={"hspace": 0.32})
    specs = [
        (grid_fi, [1e-12, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999999, 1.000001],
         ["< 0.01", "0.01–0.1", "0.1–0.3", "0.3–0.5", "0.5–0.7", "0.7–0.9", "0.9–1", "1"],
         "snow cover $f_i$", f"(a) snow cover: {int((fi > 0).sum()):,} cells with snow, "
         f"{int((fi >= 1 - 1e-6).sum()):,} of them fully covered"),
        (grid_w, [1e-12, 10, 30, 100, 300, 1000, wdeep, 1e4, 1e6],
         ["< 10", "10–30", "30–100", "100–300", "300–10$^3$", "10$^3$–2,500", "2,500–10$^4$", "> 10$^4$"],
         "cell-mean snow water equivalent $\\rho_s f_i H$ (kg m$^{-2}$)",
         f"(b) snow water equivalent: {int((swe > wdeep).sum()):,} cells above 2,500 kg m$^{{-2}}$ ($H$ > 10 m)"),
    ]
    for ax, (g, bounds, ticklab, cbl, title) in zip(axs, specs):
        ax.imshow(np.where(land, 0.0, np.nan), extent=(-180, 180, -90, 90), cmap=ListedColormap([LAND_BG]),
                  interpolation="nearest")
        im = ax.imshow(g, extent=(-180, 180, -90, 90), cmap=ListedColormap(BLUES),
                       norm=BoundaryNorm(bounds, len(BLUES)), interpolation="nearest")
        _coast(ax, land); _map_axes(ax)
        ax.set_title(title, loc="left", fontsize=8.5, weight="bold", color=INK)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09, aspect=40)
        mids = [0.5 * (lo + hi) for lo, hi in zip(bounds[:-1], bounds[1:])]                    # one label per class
        cb.set_ticks(mids); cb.ax.set_xticklabels(ticklab); cb.ax.minorticks_off()
        cb.set_label(cbl, fontsize=8); cb.ax.tick_params(labelsize=7, length=0)
        cb.outline.set_linewidth(0.4)
    fig.savefig(OUT / "fig_8_3_snow_31dec2001.png")
    plt.close(fig)
    print(f"8.3 cells with snow {(fi > 0).sum()}, full {(fi >= 1 - 1e-6).sum()}, trace cover (0 < fi < 0.01) "
          f"{((fi > 0) & (fi < 0.01)).sum()}, 0 < fi < 0.1 {((fi > 0) & (fi < 0.1)).sum()}; H > 10 m {(H > 10).sum()}, "
          f"W > {wdeep:g} {(swe > wdeep).sum()}, max H {H.max():.1f}, max W {swe.max():.0f}")


# ---------------------------------------------------------------------------------------------------------------
# Figure 8.4: one cell through 2001, run from the release state of 1 January 2001
# ---------------------------------------------------------------------------------------------------------------

def _run_cell_year(I, J, year=2001):
    """Run the model at the land cell in row I, column J of the 0.5-degree grid (north to south, west to east) through
    `year`, from the release run's restart at the end of `year` - 1, with the release parameters, the cell's Koppen-zone
    types and cropland share, and the forcing of `year`. The model's month scan (`_run_month_scan`) is called one day at
    a time, so the snow state is read at the end of every day; methane, which does not act on the physics, is off.
    The cell's soil properties are those stored in the restart (the release run's)."""
    import jax
    import jax.numpy as jnp
    from ibis.io.physics_params import load_physics_params, build_all_params
    from ibis.io import pft_types as pt
    from ibis.io.koppen import load_koppen, period_index
    from ibis.io.co2 import apply_co2
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.io.gswp3_forcing import make_monthly_daily_weather
    from ibis.timestepping.loops_gswp3 import _run_month_scan
    from ibis.timestepping.calendar import julian_day, is_leap_year
    from ibis.constants import PhysicalConstants
    land = _land_mask()
    ii, jj = np.nonzero(land)
    idx = int(np.nonzero((ii == I) & (jj == J))[0][0])
    full = _load_restart(year - 1)
    st = jax.tree.map(lambda x: x[idx:idx + 1] if (hasattr(x, "shape") and x.ndim >= 1 and x.shape[0] == NPOI) else x, full)
    del full
    lat_rad = jnp.asarray(np.deg2rad(np.array([89.75 - 0.5 * I], np.float32)))
    pd = load_physics_params(PARAMS)
    cf = np.load(CROPFRAC)["cropfrac"]
    params, pft_params, _ = build_all_params(1, lat_rad, pd, cropfrac=np.asarray(cf[[I], [J]], np.float32))
    kop = load_koppen(KOPPEN, np.array([I]), np.array([J]))
    zones = pt.zones_from_classes(kop["classes"])
    k = period_index(year, kop["period_start"], kop["period_end"])
    params, pft_params = pt.apply_types(params, pft_params, zones[k], pt.parse_type_overrides(pd.get("types")),
                                        base=(params, pft_params))
    st = apply_co2(st, year)
    consts = PhysicalConstants(); reader = GSWP3Reader(GSWP3); leap = is_leap_year(year)
    key = jax.random.PRNGKey(42)
    rec = {k_: [] for k_ in ("fi", "hsno", "tsno", "tlsub", "td", "precip", "fl", "ztop_l")}
    for imonth in range(12):
        mo = reader.read_month(year, imonth, np.array([I]), np.array([J]))
        dl, rl = make_monthly_daily_weather(mo, np.asarray(lat_rad), year, imonth, np.zeros((1, 3), np.float32))
        stack = lambda f: jnp.stack([getattr(d, f) for d in dl])
        xs = (stack("td"), stack("tmax"), stack("tmin"), stack("cloud"), stack("precip"), stack("psurf"), stack("qd"),
              stack("ud"), stack("iwet"), jnp.stack(rl),
              jnp.array([julian_day(imonth, d, leap) for d in range(mo["ndays"])], jnp.int32))
        for d in range(mo["ndays"]):
            xs_d = tuple(x[d:d + 1] for x in xs)
            st, key, _ = _run_month_scan(st, key, consts, params, xs_d, 1, 366 if leap else 365, 3600.0, 2)
            rec["fi"].append(float(st.snow.fi[0])); rec["hsno"].append(np.asarray(st.snow.hsno[0], np.float64))
            rec["tsno"].append(np.asarray(st.snow.tsno[0], np.float64)); rec["tlsub"].append(float(st.snow.tlsub[0]))
            rec["td"].append(float(xs_d[0][0, 0])); rec["precip"].append(float(xs_d[4][0, 0]))
            rec["fl"].append(float(st.vegetation.fl[0])); rec["ztop_l"].append(float(st.vegetation.ztop[0, 0]))
    return {k_: np.array(v) for k_, v in rec.items()}, pt.ZONES[int(zones[k][0])], float(cf[I, J])


def fig_8_4():
    from ibis.constants import PhysicalConstants
    I, J = 93, 181                                                   # 43.25 N, 89.25 W, the cell of Figure 3.1
    c = _snow_constants(); tf = PhysicalConstants().tmelt
    r, zone, crop = _run_cell_year(I, J)
    nd = len(r["fi"]); day = np.arange(1, nd + 1)
    snow = r["fi"] > 0
    H = r["hsno"].sum(1); W = c["rhos"] * r["fi"] * H
    ruf = np.minimum(0.70, np.maximum(c["hsnomin"] + 0.05, r["fl"] * r["ztop_l"]))
    fig, axs = plt.subplots(5, 1, figsize=(7.4, 9.6), sharex=True,
                            gridspec_kw={"hspace": 0.42, "height_ratios": [1.25, 0.8, 1, 0.9, 0.9]})
    leg = dict(frameon=False, fontsize=6.8, handlelength=1.8, borderaxespad=0.2)
    # a day with snow between two days without leaves no line segment: mark such days with a dot
    prev = np.r_[False, snow[:-1]]; nxt = np.r_[snow[1:], False]
    lone = snow & ~prev & ~nxt
    dot = dict(ls="none", marker="o", ms=2.6, mew=0)
    ax = axs[0]
    ax.plot(day, r["td"] - tf, color="#8a8a86", lw=0.8, label="air, daily mean (forcing)")
    ax.plot(day, np.where(snow, r["tsno"][:, 0] - tf, np.nan), color=BLUE, lw=1.3, label="top layer $T_1$")
    ax.plot(day, np.where(snow, r["tsno"][:, 2] - tf, np.nan), color=ORANGE, lw=1.3, label="bottom layer $T_3$")
    ax.plot(day[lone], r["tsno"][lone, 0] - tf, color=BLUE, **dot)
    ax.plot(day[lone], r["tsno"][lone, 2] - tf, color=ORANGE, **dot)
    ax.axhline(0.0, color=INK2, lw=0.6)
    ax.set_ylabel("°C"); ax.legend(loc="lower center", ncol=3, **leg)
    ax.set_title("(a) temperature (snow layers at the end of each day)", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[1]
    ax.bar(day, r["precip"], width=1.0, color=BLUE, lw=0)
    ax.set_ylabel("mm d$^{-1}$")
    ax.set_title("(b) precipitation (forcing)", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[2]
    ax.plot(day, W, color=BLUE, lw=1.4)
    ax.set_ylabel("kg m$^{-2}$"); ax.set_ylim(bottom=0)
    ax.set_title("(c) cell-mean snow water equivalent $W$", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[3]
    ax.plot(day, r["fi"], color=BLUE, lw=1.4)
    ax.set_ylim(0, 1.08); ax.set_ylabel("$f_i$")
    ax.set_title("(d) snow cover", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[4]
    ax.plot(day, np.where(snow, H, np.nan), color=BLUE, lw=1.4, label="$H$")
    ax.plot(day[lone], H[lone], color=BLUE, **dot)
    ax.plot(day, ruf, color="#8a8a86", lw=0.9, ls=(0, (4, 2)), label="$h_{ruf}$, (8.14)")
    ax.axhline(c["hsnomin"], color="#8a8a86", lw=0.9, ls=":", label=f"$h_{{min}}$ = {c['hsnomin']:g} m")
    ax.set_ylim(0, max(np.nanmax(np.where(snow, H, np.nan)), ruf.max()) * 1.25); ax.set_ylabel("m")
    ax.legend(loc="upper center", ncol=3, **leg)
    ax.set_title("(e) thickness of the pack on the covered part", loc="left", fontsize=8.5, weight="bold", color=INK)
    import datetime as _dt
    starts = [(_dt.date(2001, m, 1) - _dt.date(2001, 1, 1)).days + 1 for m in range(1, 13)]
    for a in axs:
        a.set_xlim(1, nd + 1); a.set_xticks(starts)
        a.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        _clean(a)
    axs[-1].set_xlabel("2001")
    fig.savefig(OUT / "fig_8_4_snow_year_madison.png")
    plt.close(fig)
    on = np.nonzero(snow)[0]
    print(f"8.4 zone {zone}, cropland {crop:.3f}; days with snow {snow.sum()}; max W {W.max():.1f} on day {W.argmax() + 1};"
          f" days with 0 < fi < 1 {(snow & (r['fi'] < 1 - 1e-6)).sum()}; h_ruf {ruf.min():.3f}-{ruf.max():.3f};"
          f" min T1 {np.nanmin(np.where(snow, r['tsno'][:, 0], np.nan)) - tf:.1f} C; precip sum {r['precip'].sum():.0f} mm;"
          f" snow periods (first/last day) {on[:1] + 1}, {on[-1:] + 1}; single days with snow drawn as dots "
          f"{[(_dt.date(2001, 1, 1) + _dt.timedelta(int(i))).strftime('%d %b') for i in np.nonzero(lone)[0]]}")
    # the numbers quoted in section 8.15 (values at the end of each day; index 0 = 1 January)
    date = lambda i: (_dt.date(2001, 1, 1) + _dt.timedelta(int(i))).strftime("%d %b")
    full = r["fi"] >= 1 - 1e-6
    lf = int(np.nonzero(full[:120])[0].max())                                   # last day of the winter's full cover
    print(f"8.15 f_l {r['fl'].min():.3f}-{r['fl'].max():.3f}, f_l z_top,l at most {(r['fl'] * r['ztop_l']).max():.3f} m;"
          f" full cover until {date(lf)}: W {W[:lf + 1].min():.1f}-{W[:lf + 1].max():.1f} kg m-2 (max on {date(W.argmax())}),"
          f" H {H[:lf + 1].min():.3f}-{H[:lf + 1].max():.3f} m; largest daily precipitation of Jan-Feb "
          f"{r['precip'][:59].max():.1f} mm on {date(r['precip'][:59].argmax())}")
    i20 = lf + 13
    gone = lf + int(np.nonzero(~snow[lf:])[0][0])
    print(f"8.15 melt: fi {r['fi'][lf]:.3f} / H {H[lf]:.3f} m on {date(lf)} -> fi {r['fi'][i20]:.4f} / H {H[i20]:.3f} m on "
          f"{date(i20)}; first day without snow {date(gone)}")
    def runs(i0, i1):                                                           # spells of snow: first day, last day, max fi
        out, cur = [], None
        for i in range(i0, i1):
            if snow[i] and cur is None:
                cur = i
            if cur is not None and (not snow[i] or i == i1 - 1):
                last = i - 1 if not snow[i] else i
                out.append((date(cur), date(last), round(float(r["fi"][cur:last + 1].max()), 4))); cur = None
        return out
    print(f"8.15 spells after the melt, before June: {runs(gone, 151)}")
    print(f"8.15 spells from September: {runs(243, nd)}; cover on 31 Dec {r['fi'][-1]:.3f}")
    dT = np.where(snow, r["tsno"][:, 2] - r["tsno"][:, 0], np.nan)
    print(f"8.15 largest T3 - T1 {np.nanmax(dT):.2f} K on {date(np.nanargmax(dT))}")


# ---------------------------------------------------------------------------------------------------------------
# Numbers quoted in section 8.14 (not a figure): `make_ch08.py numbers`
# ---------------------------------------------------------------------------------------------------------------

def numbers():
    import xarray as xr
    land = _land_mask(); ii, jj = np.nonzero(land); lat = 89.75 - 0.5 * ii
    def stored(st):
        sn, so = st.snow, st.soil
        swe = float(sn.rhos) * np.asarray(sn.fi, np.float64) * np.asarray(sn.hsno, np.float64).sum(1)
        hs = np.asarray(so.hsoi, np.float64)[:6]
        por = np.asarray(so.poros, np.float64); w = np.asarray(so.wsoi, np.float64); wi = np.asarray(so.wisoi, np.float64)
        soilw = (1000.0 * por * ((1 - wi) * w + wi) * hs[None, :]).sum(1)
        return swe + soilw + np.asarray(so.wpud, np.float64) + np.asarray(so.wipud, np.float64)
    st1 = _load_restart(2001)
    fi = np.asarray(st1.snow.fi, np.float64); h = np.asarray(st1.snow.hsno, np.float64); H = h.sum(1)
    snow = fi > 0; part = snow & (fi < 1 - 1e-6)
    print(f"31 Dec 2001: snow {snow.sum()}, full {(fi >= 1 - 1e-6).sum()}, partial {part.sum()}; partial fi median "
          f"{np.median(fi[part]):.3f}, H 5-95 % {np.round(np.quantile(H[part], [0.05, 0.95]), 3)}; top layer "
          f"{h[snow, 0].min():.4f}-{h[snow, 0].max():.4f} m, smallest lower layer {h[snow, 1:].min():.5f} m")
    s60 = lat < -60
    print(f"H > 10 m: {(H > 10).sum()} (S of 60S {((H > 10) & s60).sum()}, N of 60N {((H > 10) & (lat > 60)).sum()});"
          f" land cells S of 60S {s60.sum()} ({(s60 & snow).sum()} with snow), median H over all of them "
          f"{np.median(H[s60]):.2f} m, over those with H > 10 m {np.median(H[s60 & (H > 10)]):.2f} m; max {H.max():.1f} m at "
          f"{lat[H.argmax()]:.2f}, {-179.75 + 0.5 * jj[H.argmax()]:.2f}")
    s1 = stored(st1); st10 = _load_restart(2010); s10 = stored(st10); fi10 = np.asarray(st10.snow.fi)
    flux = np.zeros(NPOI)
    for y in range(2002, 2011):
        ds = xr.open_dataset(f"{REL}/hindcast/out/ibis_yearly_{y}.nc")
        g = lambda v: np.asarray(ds[v].values)[ii, jj].astype(np.float64)
        flux += g("ayprcp") - g("ayaet") - g("aytrunof") - g("aytdrain")
    res = (s10 - s1 - flux) / 9.0
    perm = (fi >= 1 - 1e-6) & (fi10 >= 1 - 1e-6)
    sel = perm & (H <= 10)
    print(f"covered on both dates, H <= 10 m: {sel.sum()} cells, residual 5-95 % {np.round(np.quantile(res[sel], [0.05, 0.95]), 2)} mm/yr")
    area = 6.371e6 ** 2 * np.deg2rad(0.5) * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))   # m2
    deep = perm & (H > 10)
    print(f"covered on both dates, H > 10 m: {deep.sum()} cells, {area[deep].sum() / 1e12:.1f} million km2; P - ET - R - D "
          f"{(flux[deep] / 9 * area[deep]).sum() / 1e12:.0f} Gt/yr; residual net {(res[deep] * area[deep]).sum() / 1e12:+.1f}, "
          f"absolute {(np.abs(res[deep]) * area[deep]).sum() / 1e12:.0f} Gt/yr")
    for lo, hi in ((10, 30), (30, 100), (100, 300), (300, 1e9)):
        sel = perm & (H > lo) & (H <= hi)
        print(f"  H {lo}-{hi} m: {sel.sum()} cells, residual median {np.median(res[sel]):.2f} mm/yr, accumulation "
              f"(P - ET - R - D) median {np.median(flux[sel]) / 9:.1f} mm/yr")
    # one year of a fully covered pack in single and double precision, with the model's cover relation and
    # re-division. JAX fixes its arithmetic per process (jax_enable_x64), so each precision runs in a process of its
    # own: single precision with 64-bit types disabled, the arithmetic of the release run.
    import subprocess
    for mode in ("precision32", "precision64"):
        out = subprocess.run([sys.executable, str(Path(__file__).resolve()), mode], capture_output=True, text=True,
                             check=True)
        print("\n".join(ln for ln in out.stdout.splitlines() if ln.startswith("  pack")))


def _pack_precision(bits):
    """A fully covered pack of thickness H0 (top layer h_top, the two others (H0 - h_top)/2) receives 78 kg m-2 yr-1 in
    its top layer every hour, as in (8.5), and passes through the model's snow_coverage and vadapt every step for one
    year (8760 steps). Prints the water the pack kept, under two rules of vadapt: the release's (and IBIS 2.6b4's), which
    re-divides the pack every step (`deep=None`), and the current code's, which leaves a pack of SNOW_DEEP (10 m) or more
    as it is until its top layer has moved by half its thickness (technote section 8.8). Run in a fresh process
    (`make_ch08.py precision32|precision64`)."""
    import jax
    if bits == 64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ibis.physics.snow_physics import vadapt, snow_coverage, SNOW_DEEP
    dt = jnp.float64 if bits == 64 else jnp.float32
    assert bool(jax.config.jax_enable_x64) == (bits == 64)
    c_ = _snow_constants()
    def year(H0, acc, deep):
        dh = acc / c_["rhos"] / 8760
        h0 = jnp.array([[c_["hsnotop"], (H0 - c_["hsnotop"]) / 2, (H0 - c_["hsnotop"]) / 2]], dt)
        def step(cr, _):
            hh, tt, ff = cr
            hh = hh.at[:, 0].add(jnp.asarray(dh, dt))
            ff, hh = snow_coverage(hh, ff, jnp.asarray(c_["hsnomin"], dt), jnp.full((1,), 0.2, dt), jnp.asarray(1.0, dt), 1e-7)
            hh, tt = vadapt(hh, tt, jnp.asarray(c_["hsnotop"], dt), jnp.sum(hh, axis=1) > 1e-7, 1e-7, deep=deep)
            assert hh.dtype == dt and tt.dtype == dt and ff.dtype == dt          # no silent promotion or truncation
            return (hh, tt, ff), None
        (h1, _, _), _ = jax.lax.scan(step, (h0, jnp.full((1, 3), 250.0, dt), jnp.ones((1,), dt)), None, length=8760)
        return (np.asarray(h1, np.float64).sum() - np.asarray(h0, np.float64).sum()) * c_["rhos"]
    name = "double" if bits == 64 else "single"
    for H0 in (5.0, 20.0, 50.0, 90.0, 200.0, 500.0, 1500.0):
        print(f"  pack of {H0:g} m, 78 kg m-2 yr-1, {name} precision: kept {year(H0, 78.0, None):.3f} re-divided every "
              f"step (release), {year(H0, 78.0, SNOW_DEEP):.3f} with the deep-pack rule (current code)")


FIGURES = {"8.1": fig_8_1, "8.2": fig_8_2, "8.3": fig_8_3, "8.4": fig_8_4}
OTHERS = {"numbers": numbers, "precision32": lambda: _pack_precision(32), "precision64": lambda: _pack_precision(64)}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        {**FIGURES, **OTHERS}[key]()
        print(key, "done")
