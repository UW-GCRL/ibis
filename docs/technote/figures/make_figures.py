"""Figures of the IBIS 3.0 technical note (docs/technote), drawn from the model's code and the release's input files.

Run from the repository root with the project interpreter (it needs matplotlib, xarray and the `ibis` package):

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_figures.py            # all figures
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_figures.py 2.1 2.4    # some

The input paths are those of the release run (verify_outputs/final_rel30/*/cfg.yaml); docs/DATA_INVENTORY.md lists
the sources. Every number a figure shows is computed here from the code (root profile, C4 share, texture classes,
zone of each Köppen class, layer depths) or read from the input files; nothing is typed in by hand except the labels
and the schematic's drawing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Ellipse, Polygon, FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

MASK = "/shared/nas2/Min/HWSD/surta.nc"
SOIL = "/shared/nas2/Min/HWSD/soita.nc"
KOPPEN = "/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz"
CROPFRAC = "/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz"

INK, INK2, GRID = "#1f1f1f", "#55534e", "#d9d9d6"
LAND_BG, OCEAN = "#efefed", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _land_mask():
    import xarray as xr
    return np.asarray(xr.open_dataset(MASK).landfrac.values) > 0          # (360, 720), north to south


def _map_axes(ax):
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
    ax.set_xticklabels([f"{abs(x)}°{'W' if x < 0 else 'E' if x > 0 else ''}" for x in range(-180, 181, 60)])
    ax.set_yticklabels([f"{abs(y)}°{'S' if y < 0 else 'N' if y > 0 else ''}" for y in range(-90, 91, 30)])
    ax.tick_params(length=2, labelsize=7)
    for s in ax.spines.values():
        s.set_color(GRID)


def _coast(ax, land):
    lon = np.linspace(-179.75, 179.75, 720); lat = np.linspace(89.75, -89.75, 360)
    ax.contour(lon, lat, land.astype(float), levels=[0.5], colors=[INK2], linewidths=0.25)


# ---------------------------------------------------------------------------------------------------------------
# Figure 1.1: a shared grid-cell column and selected process connections
# ---------------------------------------------------------------------------------------------------------------

def fig_1_1():
    # The standalone drawing reads layer geometry without importing the model.
    from make_model_structure import make_model_structure
    make_model_structure(OUT, source_root=ROOT)


# ---------------------------------------------------------------------------------------------------------------
# Figure 2.1: climate zones
# ---------------------------------------------------------------------------------------------------------------

ZONE_COLORS = {
    "A_humid": "#1a4f9c", "A_savanna": "#69a9e6", "B_hot": "#b8282a", "B_cold": "#f08a5d", "C_med": "#f7d43a",
    "C_subtrop": "#8fd16f", "C_highland": "#1f8a70", "C_oceanic": "#1e5e2c", "D_temperate": "#86d4e3",
    "D_subarctic": "#7c56c4", "D_extreme": "#3c2470", "E_tundra": "#8c8c8c", "E_ice": "#e0e0e0", "none": "#262626",
}


def fig_2_1():
    from ibis.io.pft_types import ZONES, zones_from_classes
    land = _land_mask()
    k = np.load(KOPPEN)
    z = zones_from_classes(k["classes"]).astype(float)               # (4, 360, 720) zone index
    first, last = z[0], z[-1]
    cmap = ListedColormap([ZONE_COLORS[n] for n in ZONES]); norm = BoundaryNorm(np.arange(-0.5, len(ZONES)), cmap.N)
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 9.0), gridspec_kw={"hspace": 0.48})
    a = np.where(land, first, np.nan)
    axs[0].imshow(a, extent=(-180, 180, -90, 90), cmap=cmap, norm=norm, interpolation="nearest")
    _coast(axs[0], land); _map_axes(axs[0])
    p0, p1 = int(k["period_start"][0]), int(k["period_end"][0])
    axs[0].set_title(f"(a) zone of each land cell, {p0}–{p1}", loc="left", fontsize=9, weight="bold", color=INK)
    counts = {n: int((first[land] == i).sum()) for i, n in enumerate(ZONES)}
    handles = [Patch(fc=ZONE_COLORS[n], ec=GRID if n == "E_ice" else "none", lw=0.5,
                     label=f"{n if n != 'none' else 'no zone'} ({counts[n]:,})") for n in ZONES]
    axs[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=5, frameon=False,
                  fontsize=7, handlelength=1.2, columnspacing=1.0)
    changed = land & (first != last)
    b = np.where(land, np.where(changed, 1.0, 0.0), np.nan)
    axs[1].imshow(b, extent=(-180, 180, -90, 90), cmap=ListedColormap([LAND_BG, "#b8282a"]),
                  norm=BoundaryNorm([-0.5, 0.5, 1.5], 2), interpolation="nearest")
    _coast(axs[1], land); _map_axes(axs[1])
    q0, q1 = int(k["period_start"][-1]), int(k["period_end"][-1])
    axs[1].set_title(f"(b) cells whose zone in {q0}–{q1} differs from {p0}–{p1} ({int(changed.sum()):,} cells)",
                     loc="left", fontsize=9, weight="bold", color=INK)
    fig.savefig(OUT / "fig_2_1_climate_zones.png")
    plt.close(fig)
    from collections import Counter
    tr = Counter((ZONES[int(i)], ZONES[int(j)]) for i, j in zip(first[changed], last[changed]))
    print("2.1 zone counts", counts, "\n2.1 changed", int(changed.sum()), tr.most_common(6))


# ---------------------------------------------------------------------------------------------------------------
# Figure 2.2: C4 share of the grass leaf area at the cold start
# ---------------------------------------------------------------------------------------------------------------

def fig_2_2():
    from ibis.io.veg_init import initialize_vegetation
    from ibis.io.veg_init import PLAI_INIT, XMINLAI
    tw = np.linspace(10.0, 34.0, 481).astype(np.float32)
    share = 0.8 / (1.0 + np.exp(-(tw - 22.0) / 1.5))               # equation (2.4)
    # the cold-start module gives the same share to position 11 of a grassland cell (vegetation type 10)
    # wherever its leaf area is above the minimum XMINLAI
    v = initialize_vegetation(np.full(tw.shape, 10.0, np.float32), tw=tw)
    lai11 = np.asarray(v["plai"])[:, 10]; grass = PLAI_INIT[9, 3]
    ok = lai11 > XMINLAI
    assert np.allclose(lai11[ok] / grass, share[ok], rtol=1e-5, atol=1e-6)
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.plot(tw, share, color="#2a78d6", lw=2.0, label="IBIS 3.0, eq. (2.4)")
    ax.plot(tw, np.where(tw > 22.0, 0.8, 0.0), color="#9a9a96", lw=1.6, ls=(0, (4, 2)), drawstyle="steps-post",
            label="IBIS 2.6b4 (step at 22 °C)")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=2.4)
    ax.set_xlim(10, 34); ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("warmest-month temperature $T_w$ (°C)"); ax.set_ylabel("C4 share of the grass leaf area")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig_2_2_c4_share.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 2.3: root profiles
# ---------------------------------------------------------------------------------------------------------------

def fig_2_3():
    from ibis.io.veg_init import compute_root_distribution, BETA1, BETA2, HSOI_LAYERS
    fr = compute_root_distribution(6)                                # (6, 2): lower, upper
    bot = np.cumsum(HSOI_LAYERS); top = bot - HSOI_LAYERS
    d = np.linspace(0, 400.0, 801)                                   # cm
    cum = lambda b: (1.0 - b ** d) / (1.0 - b ** 400.0)
    col = {"upper": "#1e5e2c", "lower": "#8fbf4a"}
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1.1]})
    ax = axs[0]
    for b in bot[:-1]:
        ax.axhline(b, color=GRID, lw=0.6)
    ax.plot(cum(BETA2), d / 100.0, color=col["upper"], lw=2.0, label=f"upper canopy, $\\beta$ = {BETA2:.3f}")
    ax.plot(cum(BETA1), d / 100.0, color=col["lower"], lw=2.0, label=f"lower canopy, $\\beta$ = {BETA1:.3f}")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_ylim(4.0, 0); ax.set_xlim(0, 1.0)
    ax.set_xlabel("cumulative root fraction"); ax.set_ylabel("depth (m)")
    ax.set_title("(a) roots above a depth", loc="left", fontsize=9, weight="bold", color=INK)
    ax = axs[1]
    yy = np.arange(6)
    h = 0.38
    ax.barh(yy - h / 2, fr[:, 1], height=h, color=col["upper"], label="upper canopy")
    ax.barh(yy + h / 2, fr[:, 0], height=h, color=col["lower"], label="lower canopy")
    for k in range(6):
        ax.text(fr[k, 1] + 0.006, k - h / 2, f"{fr[k, 1]:.3f}", va="center", fontsize=6.8, color=INK2)
        ax.text(fr[k, 0] + 0.006, k + h / 2, f"{fr[k, 0]:.3f}", va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(yy); ax.set_yticklabels([f"{t:g}–{b:g} m" for t, b in zip(top, bot)])
    ax.set_ylabel("soil layer")
    ax.invert_yaxis(); ax.set_xlim(0, 0.5)
    ax.set_xlabel("fraction of the roots in the layer")
    ax.set_title("(b) root fraction of each soil layer", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for a in axs:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.savefig(OUT / "fig_2_3_root_profiles.png")
    plt.close(fig)
    print("2.3 root fractions upper", np.round(fr[:, 1], 3), "lower", np.round(fr[:, 0], 3))


# ---------------------------------------------------------------------------------------------------------------
# Figure 2.4: texture classes on the texture triangle, with the top soil layer of the land cells
# ---------------------------------------------------------------------------------------------------------------

TEXTURE_NAMES = ("sand", "loamy sand", "sandy loam", "loam", "silt loam", "sandy clay loam", "clay loam",
                 "silty clay loam", "sandy clay", "silty clay", "clay")
TEXTURE_COLORS = ("#f3e3b3", "#ead39b", "#e2c58f", "#cfd9a6", "#bfe0c9", "#e7b9a1", "#d6b8d6", "#b9d3ea",
                  "#eeb3b8", "#c7c4ec", "#d8cbb8")


def _tri_xy(sand, clay):
    """Sand 100 % at the lower left, silt 100 % at the lower right, clay 100 % at the top."""
    sand = np.asarray(sand, float); clay = np.asarray(clay, float)
    return 100.0 - sand - 0.5 * clay, clay * np.sqrt(3.0) / 2.0


def fig_2_4():
    import xarray as xr
    from ibis.io.soil_init import _TEXTURE_LOOKUP, _POLYGONS, texture_class
    # the class of every point of the triangle, as the model assigns it (texture_class: the rounding included)
    n = 801
    x = np.linspace(0, 100, n); yv = np.linspace(0, 100 * np.sqrt(3) / 2, n)
    X, Y = np.meshgrid(x, yv)
    clay = Y * 2 / np.sqrt(3); sand = 100.0 - X - 0.5 * clay
    inside = (sand >= -1e-9) & (clay >= -1e-9) & (sand + clay <= 100 + 1e-9)
    cls = np.where(inside, texture_class(np.clip(sand, 0, 100), np.clip(clay, 0, 100)), -1).astype(float)
    cls[~inside] = np.nan
    land = _land_mask()
    ds = xr.open_dataset(SOIL)
    sa = np.asarray(ds.sandpct.values[0])[land]; cl = np.asarray(ds.claypct.values[0])[land]
    top_cls = texture_class(sa, cl)
    fig, ax = plt.subplots(figsize=(6.6, 5.9))
    ax.imshow(cls, origin="lower", extent=(0, 100, 0, 100 * np.sqrt(3) / 2), interpolation="nearest",
              cmap=ListedColormap(TEXTURE_COLORS), norm=BoundaryNorm(np.arange(-0.5, 11), 11))
    for poly in _POLYGONS:
        px, py = _tri_xy(poly[:, 0], poly[:, 1])
        ax.add_patch(Polygon(np.c_[px, py], closed=True, fc="none", ec="white", lw=0.9))
    px, py = _tri_xy(sa, cl)
    ax.scatter(px, py, s=0.6, c=INK, alpha=0.12, lw=0, rasterized=True)
    for i, poly in enumerate(_POLYGONS):
        pts = np.array([(s_, c_) for s_ in range(101) for c_ in range(101 - s_) if _TEXTURE_LOOKUP[s_, c_] == i], float)
        cx, cy = _tri_xy(pts[:, 0].mean(), pts[:, 1].mean())
        if TEXTURE_NAMES[i] == "silt loam":
            cx, cy = _tri_xy(20, 10)
        if TEXTURE_NAMES[i] == "loam":                               # off the fill point at 40 % sand, 20 % clay
            cx, cy = _tri_xy(31, 23)
        if TEXTURE_NAMES[i] == "loamy sand":                         # clear of the sand label in the corner
            cx, cy = _tri_xy(79, 6)
        ax.text(cx, cy, f"{TEXTURE_NAMES[i]}\n{int((top_cls == i).sum()):,}", ha="center", va="center",
                fontsize=6.8, color=INK, linespacing=1.1,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    # the fill texture of the soil file (all cells south of 60 S and most ice cells): one point, many cells
    fill = (np.round(sa, 1) == 40.0) & (np.round(cl, 1) == 20.0)
    fx, fy = _tri_xy(40, 20)
    ax.plot(fx, fy, marker="x", ms=7, mew=1.6, color="#b8282a")
    ax.annotate(f"40 % sand, 20 % clay:\n{int(fill.sum()):,} cells", (fx, fy), xytext=(70, 66),
                fontsize=6.8, color="#8f1f21", ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color="#8f1f21", lw=0.6),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    tri = np.array([_tri_xy(100, 0), _tri_xy(0, 0), _tri_xy(0, 100), _tri_xy(100, 0)])
    ax.plot(tri[:, 0], tri[:, 1], color=INK2, lw=0.8)
    for v in range(0, 101, 20):                                     # axis ticks
        bx, by = _tri_xy(v, 0); ax.text(bx, by - 2.2, f"{v}", ha="center", va="top", fontsize=6.5, color=INK2)
        lx, ly = _tri_xy(100 - v, v); ax.text(lx - 1.5, ly, f"{v}", ha="right", va="center", fontsize=6.5, color=INK2)
        rx, ry = _tri_xy(0, 100 - v); ax.text(rx + 1.5, ry, f"{v}", ha="left", va="center", fontsize=6.5, color=INK2)
    ax.text(50, -8.5, "sand (%)  ←", ha="center", fontsize=8, color=INK)
    ax.text(17, 47, "clay (%)", ha="center", rotation=60, fontsize=8, color=INK)
    ax.text(83, 47, "silt (%)", ha="center", rotation=-60, fontsize=8, color=INK)
    ax.set_xlim(-8, 108); ax.set_ylim(-11, 92); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(OUT / "fig_2_4_texture_classes.png")
    plt.close(fig)
    print("2.4 top-layer cells per class", {TEXTURE_NAMES[i]: int((top_cls == i).sum()) for i in range(11)},
          "fill texture cells", int(fill.sum()))


# ---------------------------------------------------------------------------------------------------------------
# Figure 2.5: cropland fraction
# ---------------------------------------------------------------------------------------------------------------

def fig_2_5():
    land = _land_mask()
    cf = np.asarray(np.load(CROPFRAC)["cropfrac"], float)
    ramp = ["#fde3cb", "#f9c595", "#f2a25f", "#e07a2e", "#bd5714", "#8f3d0a", "#5e2806"]
    cmap = ListedColormap(ramp); bounds = [1e-6, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.imshow(np.where(land, 0.0, np.nan), extent=(-180, 180, -90, 90), cmap=ListedColormap([LAND_BG]),
              interpolation="nearest")
    a = np.where(land & (cf > 0), cf, np.nan)
    im = ax.imshow(a, extent=(-180, 180, -90, 90), cmap=cmap, norm=norm, interpolation="nearest")
    _coast(ax, land); _map_axes(ax)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09, aspect=40, ticks=bounds)
    cb.ax.set_xticklabels([">0", "0.05", "0.1", "0.2", "0.35", "0.5", "0.75", "1"])
    cb.set_label("cropland fraction of the cell", fontsize=8); cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    fig.savefig(OUT / "fig_2_5_cropland.png")
    plt.close(fig)
    print("2.5 land cells with cropland", int((land & (cf > 0)).sum()), ">= 0.5", int((land & (cf >= 0.5)).sum()))


# ---------------------------------------------------------------------------------------------------------------
# Figure 3.1: one day of the weather generator at one cell
# ---------------------------------------------------------------------------------------------------------------

GSWP3 = "/shared/nas2/Min/GSWP3"


def fig_3_1():
    import jax
    import jax.numpy as jnp
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.io.gswp3_forcing import gswp3_day_to_daily_weather
    from ibis.physics.weather import diurnal, compute_precip_timing
    from ibis.physics.thermo import esat, qsat
    from ibis.timestepping.calendar import julian_day, is_leap_year
    from ibis.timestepping.loops import SimParams, step_time
    # the model's time of each step: its middle with the code's default of weather.sun_midstep (D33), its start without
    _clock = type("Clock", (), {"sun_midstep": SimParams.__dataclass_fields__["sun_midstep"].default})
    I, J = 93, 181                                                   # 43.25 N, 89.25 W (Madison, Wisconsin)
    year, month, day = 2001, 6, 21                                   # 22 July 2001 (month and day 0-based)
    lat = 89.75 - 0.5 * I; lon = -179.75 + 0.5 * J
    mo = GSWP3Reader(GSWP3).read_month(year, month, np.array([I]), np.array([J]))
    jd = julian_day(month, day, is_leap_year(year))
    dw, rlds = gswp3_day_to_daily_weather(mo, day, np.deg2rad([lat]).astype(np.float32), jd, np.zeros((1, 3), np.float32))
    dt = 3600.0
    key = jax.random.PRNGKey(7)
    key, plens, startp, endp = compute_precip_timing(key, jnp.asarray(dw.precip), dt)
    hs = []
    for istep in range(24):
        key, sub = jax.random.split(key)
        _, h = diurnal(step_time(istep, dt, _clock), jd, dw, jnp.array([lat], jnp.float32), plens, startp, endp, sub, dt, 2)
        hs.append({k: float(np.asarray(v).sum()) for k, v in h._asdict().items()})
    g = {k: np.array([x[k] for x in hs]) for k in hs[0]}
    t = np.arange(25.0)
    st = lambda v: np.r_[v, v[-1]]                                   # hold each step's value for its hour
    C1, C2, REF = "#2a78d6", "#eb6834", "#8a8a86"
    fig, axs = plt.subplots(3, 2, figsize=(7.4, 7.6), sharex=True, gridspec_kw={"hspace": 0.38, "wspace": 0.25})
    leg = dict(frameon=False, fontsize=6.5, handlelength=1.8, borderaxespad=0.2)
    ax = axs[0, 0]
    ax.step(t, st(g["ta"] - 273.15), where="post", color=C1, lw=1.8, label="hourly")
    ax.axhline(float(dw.tmax[0]) - 273.15, color=REF, lw=0.9, ls="--", label="daily max. and min.")
    ax.axhline(float(dw.tmin[0]) - 273.15, color=REF, lw=0.9, ls="--")
    ax.axhline(float(dw.td[0]) - 273.15, color=REF, lw=0.9, ls=":", label="daily mean (forcing)")
    ax.set_ylim(float(dw.tmin[0]) - 273.15 - 0.6, float(dw.tmax[0]) - 273.15 + 2.8)
    ax.legend(loc="upper left", ncol=1, **leg)
    ax.set_ylabel("°C"); ax.set_title("(a) air temperature", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[0, 1]
    ax.step(t, st(g["qa"] * 1e3), where="post", color=C1, lw=1.8, label="hourly")
    cap = 0.99 * np.array([float(qsat(esat(jnp.float32(x)), dw.psurf[0])) for x in g["ta"]]) * 1e3
    ax.step(t, st(cap), where="post", color=REF, lw=0.9, ls="--", label="99 % of saturation")
    ax.axhline(float(dw.qd[0]) * 1e3, color=REF, lw=0.9, ls=":", label="daily mean (forcing)")
    ax.set_ylim(top=26.5); ax.legend(loc="upper left", **leg)
    ax.set_ylabel("g kg$^{-1}$"); ax.set_title("(b) specific humidity", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[1, 0]
    ax.step(t, st(g["solad"]), where="post", color=C1, lw=1.8, label="direct")
    ax.step(t, st(g["solai"]), where="post", color=C2, lw=1.8, label="diffuse")
    ax.axhline(float(mo["rsds"][day, 0]), color=REF, lw=0.9, ls=":", label="daily mean total (forcing)")
    ax.set_ylim(0, 420); ax.legend(loc="upper left", **leg)
    ax.set_ylabel("W m$^{-2}$")
    ax.set_title(f"(c) shortwave (cloud fraction {float(dw.cloud[0]):.2f})", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[1, 1]
    ax.step(t, st((g["raina"] + g["snowa"]) * 3600.0), where="post", color=C1, lw=1.8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("mm h$^{-1}$")
    ax.set_title(f"(d) precipitation ({float(dw.precip[0]):.1f} mm in the day)", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[2, 0]
    ax.step(t, st(g["ua"]), where="post", color=C1, lw=1.8, label="hourly")
    ax.axhline(float(dw.ud[0]), color=REF, lw=0.9, ls=":", label="daily mean (forcing)")
    ax.axhline(2.5, color=REF, lw=0.9, ls="--", label="limits (2.5, 10)")
    ax.axhline(10.0, color=REF, lw=0.9, ls="--")
    ax.set_ylim(0, 12.5); ax.legend(loc="center left", bbox_to_anchor=(0, 0.62), ncol=2, columnspacing=0.8, **leg)
    ax.set_ylabel("m s$^{-1}$"); ax.set_title("(e) wind speed", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[2, 1]
    ax.step(t, st(np.full(24, float(rlds[0]))), where="post", color=C1, lw=1.8, label="forcing (used)")
    ax.step(t, st(g["fira"]), where="post", color=REF, lw=0.9, ls="--", label="generator's estimate (not used)")
    ax.legend(loc="upper left", **leg)
    ax.set_ylim(min(g["fira"].min(), float(rlds[0])) - 3, max(g["fira"].max(), float(rlds[0])) + 12)
    ax.set_ylabel("W m$^{-2}$"); ax.set_title("(f) downward longwave", loc="left", fontsize=8.5, weight="bold", color=INK)
    for a in axs.ravel():
        a.set_xlim(0, 24); a.set_xticks(range(0, 25, 6)); a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    for a in axs[2]:
        a.set_xlabel("local solar time (h)")
    fig.savefig(OUT / "fig_3_1_weather_generator_day.png")
    plt.close(fig)
    gen_sw = (g["solad"] + g["solai"]).mean()
    print(f"3.1 step times {step_time(0, dt, _clock) / 3600:.1f}, {step_time(1, dt, _clock) / 3600:.1f}, ... h;"
          f" warmest step {np.argmax(g['ta'])} h, coolest {np.argmin(g['ta'])} h; max direct {g['solad'].max():.1f},"
          f" max diffuse {g['solai'].max():.1f} W m-2; qa min {g['qa'].min()*1e3:.2f} max {g['qa'].max()*1e3:.2f} g/kg")
    print(f"3.1 {lat} {lon} day {jd}: tmax {float(dw.tmax[0])-273.15:.1f} tmin {float(dw.tmin[0])-273.15:.1f} tas {float(dw.td[0])-273.15:.1f}"
          f" hourly mean {g['ta'].mean()-273.15:.2f}; q mean {g['qa'].mean()*1e3:.2f} vs {float(dw.qd[0])*1e3:.2f};"
          f" sw generated mean {gen_sw:.1f} vs {float(mo['rsds'][day,0]):.1f}; cloud {float(dw.cloud[0]):.3f};"
          f" precip {float(dw.precip[0]):.2f} mm, event {float(startp[0])/3600:.0f}-{float(endp[0])/3600:.0f} h,"
          f" rain sum {(g['raina']+g['snowa']).sum()*3600:.2f}; wind mean {g['ua'].mean():.2f} vs {float(dw.ud[0]):.2f}; rlds {float(rlds[0]):.1f}")


FIGURES = {"1.1": fig_1_1, "2.1": fig_2_1, "2.2": fig_2_2, "2.3": fig_2_3, "2.4": fig_2_4, "2.5": fig_2_5, "3.1": fig_3_1}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
        print("figure", key, "done")
