"""Figures of Chapter 15 (cropland) of the IBIS 3.0 technical note, drawn from the model's code, the release's input
files and the release run (verify_outputs/final_rel30, read only).

Run from the repository root with the project interpreter:

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch15.py            # all figures
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch15.py 15.2       # some

Every number a figure shows is computed here: the capacity factor by the model's `crop_vfac`, the tree cover by the
model's `dynaveg` on synthetic cells with the release parameters, and the release-run fields from its yearly output.
The text of Chapter 15 quotes the values this script prints; `T15.1` and `T15.3` print the numbers of Tables 15.1 and
15.3 (the cropland map and the release run by bin of the cropland fraction), and `N15` the numbers of sections 15.4 and
15.9 that no table holds (the sapwood fraction by cropland bin; the comparison with the parameter set of 12 October
2026), without drawing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: E402  (style, helpers)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
sys.path.insert(0, str(ROOT / "verify" / "calibration"))
OUT = Path(__file__).resolve().parent

PARAMS = ROOT / "python-port" / "configs" / "params" / "params_31s_ch4h.yaml"
CROPFRAC = "/shared/nas2/Hangkai/ibis/inputs/eval/landcover/cropfrac_esacci_2000_05deg.npz"
RUN = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30/hindcast/out"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                            "#008300", "#6250d6", "#e34948")


def _clean(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _release_params(npoi, cropfrac=None):
    from ibis.io.physics_params import load_physics_params, build_all_params
    d = load_physics_params(str(PARAMS))
    return build_all_params(npoi, np.zeros(npoi, np.float32), d, cropfrac=cropfrac)


def _cropfrac():
    return np.clip(np.nan_to_num(np.asarray(np.load(CROPFRAC)["cropfrac"], float)), 0.0, 1.0)


def _true_land_area():
    """Area of the true land in each cell (m2): cos-latitude cell area times the land fraction of the vegetation map,
    the weighting of the release scores (verify/calibration/landmask.py)."""
    from landmask import true_landfrac
    lat = np.arange(89.75, -90, -0.5)
    cell = (6371.0e3 ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((360, 720))
    return cell * true_landfrac()


# ---------------------------------------------------------------------------------------------------------------
# Figure 15.1: the cropland share of a grid cell (schematic)
# ---------------------------------------------------------------------------------------------------------------

def fig_15_1():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(7, 62.5); ax.axis("off")
    x0, x1, xc = 4.0, 64.0, 40.0                      # the cell; xc divides the natural share (left) from the cropland
    yg = 18.0                                          # ground
    ya0, ya1 = 53.5, 59.0                              # the atmosphere above the cell
    ax.add_patch(Rectangle((x0, yg - 9), x1 - x0, 9, fc="#d8c29c", ec="none"))          # soil
    ax.text(x0 + 1, yg - 4.5, "litter and soil carbon (Ch. 12)", fontsize=7.2, color=INK2, va="center")
    ax.add_patch(Rectangle((x0, ya0), x1 - x0, ya1 - ya0, fc="#eaf2fb", ec="none"))      # atmosphere
    ax.text(x0 + 1, (ya0 + ya1) / 2, "atmosphere", fontsize=7.5, color=INK2, va="center", style="italic")
    ax.plot([xc, xc], [yg - 9, ya0], color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text((x0 + xc) / 2, ya1 + 0.9, "natural share  $1-f_{crop}$", ha="center", fontsize=8.5, color=INK, weight="bold")
    ax.text((xc + x1) / 2, ya1 + 0.9, "cropland share  $f_{crop}$", ha="center", fontsize=8.5, color=INK, weight="bold")
    # trees on the natural share
    for xt, kind in ((9.0, "b"), (17.0, "n"), (25.0, "b"), (33.0, "n")):
        ax.add_patch(Rectangle((xt - 0.35, yg), 0.7, 14, fc="#7a5a3c", ec="none"))
        if kind == "b":
            ax.add_patch(Ellipse((xt, yg + 17.5), 6.0, 8.0, fc="#4f8f3a", ec="none"))
        else:
            ax.add_patch(Polygon([(xt - 2.8, yg + 11), (xt + 2.8, yg + 11), (xt, yg + 23)], closed=True, fc="#2f6b3a", ec="none"))
    # shrubs on the natural share; on the cropland share a shrub that is counted in the NPP but not grown (dashed)
    for xs in (13.0, 21.0, 29.0):
        ax.add_patch(Ellipse((xs, yg + 1.6), 3.4, 3.2, fc="#8db84a", ec="none"))
    xsc = 42.4
    ax.add_patch(Ellipse((xsc, yg + 1.6), 3.4, 3.2, fc="none", ec="#6f9a2e", lw=0.9, ls=(0, (2, 1.5))))
    # grass on both shares, one leaf area for the cell (the crops on the cropland share)
    for xg in np.arange(x0 + 0.6, x1 - 0.3, 0.8):
        if xc < xg < xsc + 2.0:
            continue                                   # leave the dashed shrub clear
        ax.plot([xg, xg + 0.3], [yg, yg + 2.4], color="#b5c95c" if xg < xc else "#c9a227", lw=0.8)
    # labels on the cell
    ax.text(21.0, yg + 25.5, "trees: cover $f_u$ from the wood\ndensity on this share (eq. 15.3)",
            ha="center", fontsize=7.2, color=INK2, va="bottom", linespacing=1.25)
    ax.text(52.6, yg + 7.0, "crops: grass positions\n11 and 12, capacity\n$\\times\\,g$ (eq. 15.1)",
            ha="center", fontsize=7.0, color=INK2, va="bottom", linespacing=1.25)
    # the shrub NPP of the cropland share: counted, not grown, removed
    ax.add_patch(FancyArrowPatch((xsc, yg + 3.6), (xsc, ya0 + 2.2), arrowstyle="-|>", mutation_scale=9, color=VIOLET, lw=1.4))
    ax.text(xsc + 1.1, 48.6, "shrub NPP of this share,\n$S$ (eq. 15.4): counted\nin the NPP, not grown", fontsize=7.0,
            color=INK, va="top", linespacing=1.2)
    # the grass leaf litterfall of the cropland share divides: a share h is the harvest, the rest goes to the litter
    xn, yn = 62.6, yg + 3.2                            # where the litterfall divides
    ax.plot([49.5, xn], [yn, yn], color=INK2, lw=1.1, solid_capstyle="butt")
    ax.plot([xn], [yn], marker="o", ms=2.6, color=INK2)
    ax.text(54.8, yn + 0.5, "leaf litterfall", fontsize=6.6, color=INK2, va="bottom", ha="center")
    ax.add_patch(FancyArrowPatch((xn, yn), (xn, ya0 + 2.2), arrowstyle="-|>", mutation_scale=9, color=ORANGE, lw=1.4,
                                 shrinkA=0))
    ax.text(xn - 0.9, 39.5, "harvest $H$ (eq. 15.6):\nshare $h$ of the\nleaf litterfall", fontsize=7.0, color=INK,
            va="top", ha="right", linespacing=1.2)
    ax.add_patch(FancyArrowPatch((xn, yn), (xn, yg - 3.6), arrowstyle="-|>", mutation_scale=8, color=INK2, lw=0.9,
                                 shrinkA=0))
    ax.text(xn - 0.9, yg - 2.2, "rest of the leaf litter", fontsize=6.6, color=INK2, va="center", ha="right")
    ax.text((xsc + xn) / 2, ya1 - 1.25, "removed carbon $E$ (eq. 15.6)", fontsize=7.2, color=INK, va="center",
            ha="center")
    # right-hand notes, with the release values of the crop section of the parameter file
    import yaml
    cp = yaml.safe_load(open(PARAMS))["crop"]
    q = min(max(float(cp["c4_share"]), 0.0), 1.0)
    sbar = (1.0 - q) * float(cp["vmax_scale_c3"]) + q * float(cp["vmax_scale_c4"])
    lx = 68.0
    notes = [
        (51.0, "One set of grass pools per cell", "the crops are the grass positions;\nno sowing, harvest date or fallow"),
        (42.0, "Capacity of the grasses", f"$g = 1 + f_{{crop}}(\\bar{{s}}-1)$, $\\bar{{s}}$ = {sbar:.3f}\nin the release"),
        (33.5, "Trees", "keep their wood; cover is $(1-f_{crop})$\ntimes the cover of their own density"),
        (25.0, "Shrubs", "NPP $\\times\\,(1-f_{crop})$ grows; the rest\nleaves the cell"),
        (16.5, "Harvest", f"share $h$ = {float(cp['harvest_index']):g} of the cropland share\nof the grass leaf litterfall"),
    ]
    for yy, head, body in notes:
        ax.text(lx, yy + 1.2, head, fontsize=7.8, color=INK, weight="bold", va="bottom")
        ax.text(lx, yy + 0.9, body, fontsize=7.0, color=INK2, va="top", linespacing=1.25)
    fig.savefig(OUT / "fig_15_1_cropland_cell.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 15.2: (a) the capacity factor of the grasses, (b) the tree cover with cropland
# ---------------------------------------------------------------------------------------------------------------

def fig_15_2():
    import jax.numpy as jnp
    import equinox as eqx
    from ibis.timestepping.loops import crop_vfac
    from ibis.io.parameters import CropParams
    from ibis.physics.vegetation_dynamics import dynaveg
    from ibis.state.model_state import create_ibis_state
    from ibis.dimensions import GridConfig

    # (a) capacity factor, by the model's crop_vfac with the release crop section
    c = np.linspace(0.0, 1.0, 101).astype(np.float32)
    sp, pft, _ = _release_params(c.size, cropfrac=c)
    g_rel = np.asarray(crop_vfac(sp, None, 12))                                     # c4_share 0.2: both grasses
    tow = eqx.tree_at(lambda p: p.crop, sp, CropParams(harvest_index=float(sp.crop.harvest_index),
                                                         vmax_scale_c3=float(sp.crop.vmax_scale_c3),
                                                         vmax_scale_c4=float(sp.crop.vmax_scale_c4), c4_share=-1.0))
    g_tow = np.asarray(crop_vfac(tow, None, 12))                                     # c4_share -1: per position
    assert np.allclose(g_rel[:, :10], 1.0) and np.allclose(g_rel[:, 10], g_rel[:, 11])
    sbar = float(g_rel[-1, 11])

    # (b) tree cover: dynaveg on synthetic cells (wood in position 5, no NPP, fire off, no cover limit)
    shares = (0.0, 0.25, 0.5, 0.75)
    D = 0.005 + 0.005                       # background disturbance + the constant of the legacy fire scheme (fire off)
    w0 = np.linspace(0.0, 8.4, 421).astype(np.float32)                 # pool before the update; W spans 0-8.2
    n = w0.size * len(shares)
    cf = np.repeat(np.asarray(shares, np.float32), w0.size)
    sp2, pft2, _ = _release_params(n, cropfrac=cf)
    st = create_ibis_state(GridConfig(npoi=n, nbedrock=0))
    cbiow = jnp.zeros((n, 12)).at[:, 4].set(jnp.asarray(np.tile(w0, len(shares))))
    st = eqx.tree_at(lambda s: (s.vegetation.cbiol, s.vegetation.cbiow, s.vegetation.cbior, s.vegetation.exist,
                                s.summary.aynpp),
                     st, (jnp.full((n, 12), 0.01), cbiow, jnp.full((n, 12), 0.01), jnp.ones((n, 12)),
                          jnp.zeros((n, 12))))
    v, _, _ = dynaveg(st.vegetation, st.carbon, st.summary, pft2, 0, 1e-7, cropfrac=jnp.asarray(cf),
                      crop_params=sp2.crop)
    # The wood W of (15.3), the `wood` that dynaveg's cover reads: summed after the year's decay (11.11) and before the
    # disturbance (11.18), which then takes the share D of the pools that dynaveg returns (as make_ch11.fig_11_4).
    W = 0.001 + np.asarray(v.cbiow[:, :8].sum(axis=1)) / (1.0 - D)
    fu = np.asarray(v.fu)
    wn, fk, fmin = float(pft2.woodnorm), float(pft2.fu_k), float(pft2.fu_min)
    phi = lambda W_, c_: (1 - c_) * (1 - np.exp(-W_ / (np.maximum(1 - c_, 0.01) * fk))) / (1 - np.exp(-wn / fk)) * (1 - D)
    eq = np.clip(phi(W, cf), fmin, 0.975)              # (15.3) with no maximum tree cover
    assert np.allclose(fu, eq, atol=2e-6), np.abs(fu - eq).max()

    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), gridspec_kw={"wspace": 0.32})
    ax = axs[0]
    ax.axhline(1.0, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.plot(c, g_tow[:, 10], color=ORANGE, lw=1.6, label=f"towers: C4 grass, $s_{{C4}}$ = {float(sp.crop.vmax_scale_c4):.3f}")
    ax.plot(c, g_rel[:, 10], color=BLUE, lw=1.6, label=f"release: both grasses, $\\bar{{s}}$ = {sbar:.3f}")
    ax.plot(c, g_tow[:, 11], color=AQUA, lw=1.6, label=f"towers: C3 grass, $s_{{C3}}$ = {float(sp.crop.vmax_scale_c3):.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.3)
    ax.set_xlabel("cropland fraction $f_{crop}$"); ax.set_ylabel("factor $g$ on the grass capacity")
    ax.set_title("(a) capacity of the grasses", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    _clean(ax)
    ax = axs[1]
    hs = []
    for k, (s, col) in enumerate(zip(shares, (BLUE, ORANGE, AQUA, YELLOW))):
        m = slice(k * w0.size, (k + 1) * w0.size)
        hs += ax.plot(W[m], fu[m], color=col, lw=1.6, label=f"$f_{{crop}}$ = {s:g}")
        ax.axhline((1.0 - s) * (1.0 - D), color=col, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.0)
    ax.set_xlabel("wood carbon of the tree positions $W$ (kg C m$^{-2}$)"); ax.set_ylabel("upper-canopy cover $f_u$")
    ax.set_title("(b) tree cover with cropland", loc="left", fontsize=9, weight="bold", color=INK)
    order = [hs[0], hs[2], hs[1], hs[3]]               # two columns, filled column-wise, that read 0, 0.25 / 0.5, 0.75
    ax.legend(order, [h.get_label() for h in order], frameon=False, fontsize=6.8, loc="lower right", ncol=2)
    _clean(ax)
    fig.savefig(OUT / "fig_15_2_capacity_cover.png")
    plt.close(fig)
    print("15.2 sbar %.4f; g at fcrop=1: release %.4f, towers C4 %.4f C3 %.4f; g at fcrop=0.2: release %.4f" % (
        sbar, g_rel[-1, 10], g_tow[-1, 10], g_tow[-1, 11], g_rel[20, 10]))
    for k, s in enumerate(shares):
        m = slice(k * w0.size, (k + 1) * w0.size)
        Wk, fuk = W[m], fu[m]
        final = float(fuk[-1])
        print(f"     fcrop {s}: final level {final:.3f} (W {Wk[-1]:.2f}); within 1 % of it from W {Wk[np.argmax(fuk >= 0.99 * final)]:.2f};"
              f" at the cap 0.975 from W {(Wk[np.argmax(fuk >= 0.975 - 1e-6)] if fuk.max() >= 0.975 - 1e-6 else np.nan):.2f};"
              f" at W = 1: fu {np.interp(1.0, Wk, fuk):.3f} ((15.3): {float(np.clip(phi(1.0, s), fmin, 0.975)):.3f});"
              f" at W = {Wk[0]:.3f}: fu {fuk[0]:.3f}; limit (1 - c)(1 - D) {(1 - s) * (1 - D):.3f}")


# ---------------------------------------------------------------------------------------------------------------
# Figure 15.3: the release run, 2001-2010: harvest map, cover and harvest by cropland fraction
# ---------------------------------------------------------------------------------------------------------------

def _release_means(names, years=range(2001, 2011)):
    import xarray as xr
    acc = {v: np.zeros((360, 720)) for v in names}
    for y in years:
        with xr.open_dataset(f"{RUN}/ibis_yearly_{y}.nc") as ds:
            for v in names:
                acc[v] += np.nan_to_num(np.asarray(ds[v].values, float))
    return {v: a / len(years) for v, a in acc.items()}


def fig_15_3():
    land = _land_mask(); c = _cropfrac(); A = _true_land_area() * land
    F = _release_means(("ayharvest", "aycropshrub", "fu", "fl"))
    h = F["ayharvest"] * 1e3                                           # g C m-2 of the cell yr-1
    fig = plt.figure(figsize=(8.0, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    ramp = ["#fde3cb", "#f9c595", "#f2a25f", "#e07a2e", "#bd5714", "#8f3d0a", "#5e2806"]
    bounds = [1e-3, 1, 2, 5, 10, 20, 40, max(80.0, float(np.ceil(h[land].max())))]
    cmap = ListedColormap(ramp); norm = BoundaryNorm(bounds, cmap.N)
    ax.imshow(np.where(land, 0.0, np.nan), extent=(-180, 180, -90, 90), cmap=ListedColormap([LAND_BG]), interpolation="nearest")
    im = ax.imshow(np.where(land & (h > 1e-3), h, np.nan), extent=(-180, 180, -90, 90), cmap=cmap, norm=norm,
                   interpolation="nearest")
    _coast(ax, land); _map_axes(ax)
    ax.set_ylim(-60, 90)
    ax.set_title("(a) harvest, 2001–2010 mean (g C per m$^2$ of the cell per year)", loc="left", fontsize=9,
                 weight="bold", color=INK)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.1, aspect=45, ticks=bounds)
    cb.ax.set_xticklabels(["0.001", "1", "2", "5", "10", "20", "40", f"{bounds[-1]:g}"])
    cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_linewidth(0.4)
    # bins of the cropland fraction, over the cells with cropland; each bin is drawn at its mean cropland fraction
    # (land-area weighted), where the line 1 - f_crop is the bin's mean natural share
    edges = np.round(np.arange(0.0, 1.0001, 0.1), 2)
    fu_b, fl_b, hc_b, sc_b, n_b, x_b = [], [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = land & (c > lo if lo == 0 else c >= lo) & ((c < hi) if hi < 1 else (c <= 1.0))
        a = A[m]; n_b.append(int(m.sum())); x_b.append(np.sum(c[m] * a) / a.sum())
        fu_b.append(np.sum(F["fu"][m] * a) / a.sum()); fl_b.append(np.sum(F["fl"][m] * a) / a.sum())
        ca = np.sum(c[m] * a)
        hc_b.append(1e3 * np.sum(F["ayharvest"][m] * a) / ca); sc_b.append(1e3 * np.sum(F["aycropshrub"][m] * a) / ca)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x_b, fu_b, color=BLUE, lw=1.6, marker="o", ms=3, label="upper canopy $f_u$")
    ax.plot(x_b, fl_b, color=AQUA, lw=1.6, marker="o", ms=3, label="lower canopy $f_l$")
    ax.plot([0, 1], [1, 0], color=INK2, lw=0.7, ls=(0, (3, 2)), label="natural share $1-f_{crop}$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("cropland fraction $f_{crop}$ (mean of each bin of 0.1)"); ax.set_ylabel("cover (fraction of the cell)")
    ax.set_title("(b) canopy cover", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    _clean(ax)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x_b, hc_b, color=ORANGE, lw=1.6, marker="o", ms=3, label="harvest $H$")
    ax.plot(x_b, sc_b, color=VIOLET, lw=1.6, marker="o", ms=3, label="shrub NPP removed $S$")
    ax.set_xlim(0, 1); ax.set_ylim(0, max(max(hc_b), max(sc_b)) * 1.3)
    ax.set_xlabel("cropland fraction $f_{crop}$ (mean of each bin of 0.1)"); ax.set_ylabel("g C per m$^2$ of cropland per year")
    ax.set_title("(c) carbon removed", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    _clean(ax)
    fig.savefig(OUT / "fig_15_3_release_cropland.png")
    plt.close(fig)
    Pg = lambda x: float(np.sum((x * A)[land])) * 1e-12
    print("15.3 harvest %.3f PgC/yr, shrub NPP removed %.3f PgC/yr (true land, 2001-2010); cropland %.2f Mkm2; per m2 cropland"
          " %.1f gC" % (Pg(F["ayharvest"]), Pg(F["aycropshrub"]), np.sum((c * A)[land]) / 1e12,
                        1e3 * Pg(F["ayharvest"]) * 1e12 / np.sum((c * A)[land])))
    grey = land & (c > 0) & ~(h > 1e-3)
    print("     harvest max %.1f gC/m2 cell; cells with cropland drawn grey (harvest <= 0.001 gC/m2): %d, max cropland fraction"
          " %.3f, cropland %.3f Mkm2; cells per bin" % (h[land].max(), grey.sum(), c[grey].max(), np.sum((c * A)[grey]) / 1e12), n_b)
    print("     mean f_crop", np.round(x_b, 3), "\n     fu", np.round(fu_b, 3), "\n     fl", np.round(fl_b, 3),
          "\n     H/crop", np.round(hc_b, 1), "\n     S/crop", np.round(sc_b, 1))


# ---------------------------------------------------------------------------------------------------------------
# Tables 15.1 and 15.3: the cropland map and the release run by bin of the cropland fraction (printed, not drawn)
# ---------------------------------------------------------------------------------------------------------------

BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0))


def _bin_mask(land, c, lo, hi):
    """Cells with lo < c <= hi (the first bin takes c > 0, the last c <= 1)."""
    return land & (c > lo) & (c <= hi)


def table_15_1():
    """The cropland map: cells and cropland area by bin, on the land fraction of each cell (Table 15.1)."""
    land = _land_mask(); c = _cropfrac(); A = _true_land_area() * land; Acell = _cell_area() * land
    print("Table 15.1  land cells %d; with cropland %d; >= 0.5: %d; = 1: %d" % (
        land.sum(), (land & (c > 0)).sum(), (land & (c >= 0.5)).sum(), (land & (c >= 0.9999)).sum()))
    print("  cropland area: %.2f Mkm2 on the land fraction, %.2f Mkm2 on the whole cells; land area %.1f / %.1f Mkm2" % (
        np.sum(c * A) / 1e12, np.sum(c * Acell) / 1e12, A.sum() / 1e12, Acell.sum() / 1e12))
    for lo, hi in BINS:
        m = _bin_mask(land, c, lo, hi)
        print(f"  {lo:.1f} < c <= {hi:.1f}: {int(m.sum()):6d} cells, cropland {np.sum((c * A)[m]) / 1e12:6.2f} Mkm2,"
              f" mean c {np.sum((c * A)[m]) / A[m].sum():.3f}")


def _cell_area():
    lat = np.arange(89.75, -90, -0.5)
    return (6371.0e3 ** 2) * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((360, 720))


def table_15_3():
    """The release run, 2001-2010 means, by bin of the cropland fraction (Table 15.3) and the totals of section 15.9."""
    import xarray as xr
    land = _land_mask(); c = _cropfrac(); A = _true_land_area() * land
    names = ("ayharvest", "aycropshrub", "ayclearing", "aycdisturb", "fu", "fl", "lail")
    F = _release_means(names)
    Pg = lambda x: float(np.sum((x * A)[land])) * 1e-12
    ca = np.sum((c * A)[land])
    print("Table 15.3 / section 15.9  2001-2010 means on the land fraction of the cells")
    print("  harvest %.3f PgC/yr (%.1f gC per m2 of cropland); shrub NPP removed %.3f PgC/yr; clearing %.4f PgC/yr;"
          " carbon removed in all (cdisturb) %.3f PgC/yr" % (Pg(F["ayharvest"]), 1e3 * Pg(F["ayharvest"]) * 1e12 / ca,
                                                             Pg(F["aycropshrub"]), Pg(F["ayclearing"]), Pg(F["aycdisturb"])))
    years = []
    for y in range(2001, 2011):
        with xr.open_dataset(f"{RUN}/ibis_yearly_{y}.nc") as ds:
            years.append((y, 1e-12 * float(np.sum((np.nan_to_num(np.asarray(ds["ayharvest"].values, float)) * A)[land]))))
    print("  harvest by year:", ", ".join(f"{y} {h:.3f}" for y, h in years))
    print("  bin | cells | cropland Mkm2 | fu | fl | 1-c | H gC/m2 cropland | S gC/m2 cropland")
    rows = [("all cropland cells", land & (c > 0))] + [(f"{lo:.1f} < c <= {hi:.1f}", _bin_mask(land, c, lo, hi)) for lo, hi in BINS]
    for name, m in rows:
        a = A[m]; cam = np.sum((c * A)[m])
        print(f"  {name:18s} | {int(m.sum()):6d} | {cam / 1e12:5.2f} | {np.sum(F['fu'][m] * a) / a.sum():.3f} |"
              f" {np.sum(F['fl'][m] * a) / a.sum():.3f} | {np.sum((1 - c[m]) * a) / a.sum():.3f} |"
              f" {1e3 * np.sum((F['ayharvest'] * A)[m]) / cam:5.1f} | {1e3 * np.sum((F['aycropshrub'] * A)[m]) / cam:5.1f}")
    # the floor of the upper-canopy cover against the natural share (section 15.4)
    with xr.open_dataset(f"{RUN}/ibis_yearly_2010.nc") as ds:
        fu = np.nan_to_num(np.asarray(ds["fu"].values, float))
    over = land & (c > 0) & (fu > (1.0 - c) + 1e-4)
    print("  2010: cells whose upper-canopy cover exceeds the natural share 1 - c: %d, all with 1 - c < fu_min = 0.05: %s"
          % (over.sum(), bool(np.all((1.0 - c)[over] < 0.05))))


# ---------------------------------------------------------------------------------------------------------------
# Numbers of sections 15.4 and 15.9 that no table holds (printed, not drawn)
# ---------------------------------------------------------------------------------------------------------------

RUN_OCT12 = "/shared/nas2/Hangkai/ibis/verify_outputs/final_s7_crop/hindcast/out"     # the parameter set of 12 October 2026


def numbers_15():
    """(1) Section 15.4: the sapwood fraction (11.28) of the trees by bin of the cropland fraction in the release state at
    the end of 2010 (the hindcast's restart), against the same formula with the wood density of the natural share.
    (2) Section 15.9: the release hindcast against that of the parameter set of 12 October 2026 on the cells with more
    than 10 % cropland, 2001-2010 means weighted by the cropland area (cover: by the land area)."""
    import xarray as xr
    from ibis.io.restart import load_state
    from ibis.state.model_state import create_ibis_state
    from ibis.dimensions import GridConfig
    land = _land_mask(); c = _cropfrac(); A = _true_land_area() * land
    # (1) the restart holds the land points in the row-major order of the land mask (run_global._subset_land_mask)
    st, _ = load_state(str(Path(RUN).parent / "restart" / "ibis_restart_2010.npz"),
                       create_ibis_state(GridConfig(npoi=int(land.sum()), nbedrock=3)))
    v = st.vegetation
    sf = np.asarray(v.sapfrac, float); cw = np.maximum(np.asarray(v.cbiow[:, :8], float), 0.0).sum(axis=1)
    cl = c[land]
    free = (sf > 0.05 + 1e-4) & (sf < 0.5 - 1e-4)          # not at the floor or the cap: (q/v) 0.75 z_top rho_w / W
    sf_nat = np.clip(sf * (1.0 - cl), 0.05, 0.5)            # the same with W' = W / (1 - c) in place of W
    print("15.4 sapwood fraction, release state at the end of 2010, cells with more than 0.5 kgC m-2 of tree wood")
    for name, m in [("no cropland", cl == 0)] + [(f"{lo:.1f} < c <= {hi:.1f}", (cl > lo) & (cl <= hi)) for lo, hi in BINS]:
        m = m & (cw > 0.5); mf = m & free
        print(f"  {name:14s}: {int(m.sum()):6d} cells, median {np.median(sf[m]):.3f}, at the cap 0.5 {np.mean(sf[m] >= 0.5 - 1e-4):.3f};"
              f" below the cap and above the floor ({int(mf.sum())} cells): median {np.median(sf[mf]):.3f}, with W' {np.median(sf_nat[mf]):.3f}")
    # (2) the two hindcasts
    print("15.9 cells with more than 10 % cropland, 2001-2010: 12 October set | release")
    m = land & (c > 0.1); a = A[m]; ca = c[m] * a
    out = {}
    for tag, run in (("12 October", RUN_OCT12), ("release", RUN)):
        acc = {k: np.zeros((360, 720)) for k in ("ayharvest", "aynpptot", "fl")}
        for y in range(2001, 2011):
            with xr.open_dataset(f"{run}/ibis_yearly_{y}.nc") as ds:
                for k in acc:
                    acc[k] += np.nan_to_num(np.asarray(ds[k].values, float)) / 10.0
        H = np.sum(acc["ayharvest"][m] * a) / ca.sum()                      # kgC per m2 of cropland
        N = np.sum(acc["aynpptot"][m] * ca) / ca.sum()                      # the cells' NPP, cropland-weighted
        out[tag] = (1e3 * H, H / N, N, np.sum(acc["fl"][m] * a) / a.sum(),
                    1e-12 * float(np.sum((acc["ayharvest"] * A)[land])))
    for i, name in enumerate(("harvest, gC per m2 of cropland", "harvest per m2 of cropland / cell NPP per m2",
                              "cell NPP, kgC m-2 yr-1", "lower-canopy cover", "harvest, PgC yr-1 (all land)")):
        r = out["release"][i] / out["12 October"][i]
        print(f"  {name:46s}: {out['12 October'][i]:8.4f} | {out['release'][i]:8.4f}  (ratio {r:.3f})")
    # (3) section 15.4: the cells whose upper-canopy cover the floor holds above the natural share, release run, 2010
    with xr.open_dataset(f"{RUN}/ibis_yearly_2010.nc") as ds:
        g = {k: np.nan_to_num(np.asarray(ds[k].values, float)) for k in ("fu", "aynppu", "ayharvest")}
    over = land & (c > 0) & (g["fu"] > (1.0 - c) + 1e-4)
    Pg = lambda x: 1e-12 * float(np.sum((x * A)[over]))
    print("15.4 2010, cells with an upper-canopy cover above 1 - c: %d (cropland fraction %.3f-%.3f), land %.2f Mkm2, cropland"
          " %.2f Mkm2; upper-canopy NPP %.3f PgC/yr, of which above the natural share (pro rata of the cover) %.3f;"
          " harvest %.3f PgC/yr; global upper-canopy NPP %.2f PgC/yr" % (
              over.sum(), c[over].min(), c[over].max(), A[over].sum() / 1e12, np.sum((c * A)[over]) / 1e12,
              Pg(g["aynppu"]), Pg(g["aynppu"] * np.clip(1 - (1 - c) / np.maximum(g["fu"], 1e-9), 0, 1)),
              Pg(g["ayharvest"]), 1e-12 * float(np.sum((g["aynppu"] * A)[land]))))


FIGURES = {"15.1": fig_15_1, "15.2": fig_15_2, "15.3": fig_15_3, "T15.1": table_15_1, "T15.3": table_15_3,
           "N15": numbers_15}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
        print("figure", key, "done")
