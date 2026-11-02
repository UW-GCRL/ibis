"""Figures of Chapter 14 (wetland methane) of the IBIS 3.0 technical note, drawn from the model's code and the release's
input and output files.

Run from the repository root with the project interpreter (it needs matplotlib, xarray, jax and the `ibis` package):

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch14.py            # all figures
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch14.py 14.2       # some

Every number a figure shows is computed here: the response curves from the model's own functions
(`ibis.bgc.methane`) with the band parameters of the release parameter file; the band map from the Köppen file and
`ibis.io.pft_types.wetland_band`; the inundated fractions from WAD2M (through `ibis.io.wad2m`, as the model reads it)
and from the release hindcast's monthly output (`finund`); the production temperature from the release hindcast's
month-mean soil state (`tsoi_mean`, `wsoi_mean`, `wisoi_mean`) through `ibis.bgc.methane`, by the rule of the current code
(weighted by the substrate profile, `MethaneParams.t_anaer_substrate`, technote review D13) and by the release run's rule
(thickness-weighted to 4 m), and the emission from its `amch4` and `aych4` (the release run's, made with the earlier
rule); the latitude-band totals of the Global Methane Budget from its bottom-up wetland table (inputs/eval/mip/gmb2024).
Band-level aggregates only; no wetland tower data are used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: E402  (style and helpers)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

PARAMS = ROOT / "python-port/configs/params/params_31s_ch4h.yaml"
KOPPEN = "/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz"
WAD2M = "/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc"
LEARNED = "/shared/nas2/Hangkai/ibis/inputs/methane/wetland_learned/wetland_learned_v2_s31t.npz"
HINDCAST = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30/hindcast/out"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"              # the palette, in order
BAND_NAMES = ("W01 tropical and subtropical", "W02 temperate", "W03 boreal and arctic")
BAND_COLORS = (BLUE, ORANGE, AQUA)


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def _band_params():
    """MethaneParams of each wetland band, as the driver builds them (biome_methane_params with the parameter file's
    `methane` section as the base and its `methane_bands` section as the per-band values)."""
    import yaml
    import jax.numpy as jnp
    from ibis.io.physics_params import build_methane_params
    from ibis.bgc.methane import biome_methane_params
    d = yaml.safe_load(open(PARAMS))
    p = biome_methane_params(jnp.zeros(3), base=build_methane_params(d), bands=d["methane_bands"],
                             band_index=jnp.arange(3))
    return p


def _land_points():
    land = _land_mask()
    I, J = np.where(land)
    lat = 89.75 - 0.5 * I; lon = -179.75 + 0.5 * J
    return land, I, J, lat, lon


def _bands(period: int, I, J, lat):
    """Methane band of every land point for the Köppen period, as run_global builds it: the wetland type of the zone,
    the latitude rule where the zone has none (E_ice, cells without a zone). Returns (band, from_zone)."""
    from ibis.io.pft_types import zones_from_classes, wetland_band
    k = np.load(KOPPEN)
    b = np.asarray(wetland_band(zones_from_classes(k["classes"][period])[I, J])).astype(int)
    lr = np.where((lat > -30.0) & (lat < 30.0), 0, np.where(lat >= 60.0, 2, 1))
    return np.where(b >= 0, b, lr), b >= 0


def _land_area(I, J, lat):
    """Cell area (m2) times the land fraction of the cell in the ESA CCI land cover of 1992 (the sum of the IBIS
    vegetation-type fractions, as verify/calibration/landmask.true_landfrac), the weighting of the release's totals."""
    sys.path.insert(0, str(ROOT / "verify/calibration"))
    from landmask import true_landfrac
    cell = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat))
    return cell * np.nan_to_num(true_landfrac()[I, J])


# ---------------------------------------------------------------------------------------------------------------
# Figure 14.1: the calculation in one wetland column and through one month (schematic)
# ---------------------------------------------------------------------------------------------------------------

def fig_14_1():
    from ibis.io.soil_init import HSOI_LAYERS
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.set_xlim(0, 116); ax.set_ylim(9, 74); ax.axis("off")
    ax.text(0, 73.6, "(a) the soil column of the inundated fraction", fontsize=9, weight="bold", color=INK, va="top")
    x0, x1 = 18.0, 38.0
    bot = np.cumsum(HSOI_LAYERS); top = bot - HSOI_LAYERS
    ytop = 62.0
    hdraw = (5.2, 5.2, 6.0, 7.0, 8.0, 9.0)                         # drawn heights, not to scale
    yb = ytop - np.cumsum(hdraw); yt = yb + np.array(hdraw)
    zw_draw = yt[1] - 0.45 * hdraw[1]                              # a water table inside layer 2 (drawn)
    thawed = (True, True, True, True, False, False)                # an example: thawed to 1 m
    for k in range(6):
        ax.add_patch(Rectangle((x0, yb[k]), x1 - x0, hdraw[k], fc="#c9a77c" if thawed[k] else "#d7e3ee", ec="white", lw=0.8))
        ax.text(x0 - 0.6, yb[k], f"{bot[k]:g} m", ha="right", va="center", fontsize=6.5, color=INK2)
    ax.text(x0 - 0.6, ytop, "0", ha="right", va="center", fontsize=6.5, color=INK2)
    # anaerobic zone below the water table: hatched
    ax.add_patch(Rectangle((x0, yb[-1]), x1 - x0, zw_draw - yb[-1], fc="none", ec="#6d6a63", lw=0, hatch="////", alpha=0.35))
    ax.plot([x0 - 1.5, x1 + 1.5], [zw_draw, zw_draw], color=BLUE, lw=1.6)
    ax.text(x1 + 2.0, zw_draw, "water table $z_w$", color=INK, fontsize=7.5, va="center")
    # top 0.5 m bracket
    xb = x0 - 7.0
    ax.plot([xb, xb], [yb[2], ytop], color=INK2, lw=0.8)
    for yy in (yb[2], ytop):
        ax.plot([xb, xb + 0.8], [yy, yy], color=INK2, lw=0.8)
    ax.text(xb - 0.7, (yb[2] + ytop) / 2, "pore fill $\\bar{\\theta}$ of\nthe top 0.5 m\nsets $z_w$", ha="right", va="center",
            fontsize=6.6, color=INK2, linespacing=1.2)
    # surface flux
    xm = 0.5 * (x0 + x1)
    ax.add_patch(FancyArrowPatch((xm, ytop + 0.3), (xm, ytop + 8.0), arrowstyle="-|>", mutation_scale=9, color=ORANGE, lw=1.4))
    ax.text(xm + 1.3, ytop + 5.6, "CH$_4$ flux $F$", color=INK, fontsize=8, va="center")
    lx = x1 + 2.0
    notes = [
        (yt[0] - 1.2, "above $z_w$: oxic soil; the share of the\nCH$_4$ that escapes oxidation is exp($-z_w/\\tau_{ox}$)"),
        (yt[3] + 1.0, "below $z_w$ (hatched): anaerobic; the share\nof $R_h$ decomposed there is exp($-z_w/z_\\tau$)"),
        (yb[3] - 0.6, "brown: thawed layers below $z_w$ set\nthe production temperature $T_a$, weighted\nas the decomposition, "
                      "by exp($-z/z_\\tau$);\nfrozen layers (blue) do not count;\nno thawed layer below $z_w$: no CH$_4$"),
    ]
    for yy, s in notes:
        ax.text(lx, yy, s, fontsize=6.8, color=INK2, va="top", linespacing=1.25)
    ax.text(x0, yb[-1] - 1.6, "six soil layers to 4 m, not to scale", fontsize=6.5, color=INK2, style="italic", va="top")
    # (b) one month
    bx0, bw = 76.0, 39.0
    ax.text(bx0 - 2.0, 73.6, "(b) one month", fontsize=9, weight="bold", color=INK, va="top")
    boxes = [
        ("every time step", "$F$ per unit of wetland from the step's\nsoil temperature, water and ice and the\n$R_h$ of the last decomposition call", "#e8f0fa"),
        ("day, then month", "mean of the steps, then of the days", "#eef3f8"),
        ("end of the month", "× inundated fraction $f$ of the month:\nWAD2M climatology + learned anomaly from\nthe month's mean soil state and water totals\n(climatology alone outside the trained cells)", "#f6efe0"),
        ("bookkeeping", "monthly and annual CH$_4$; the CH$_4$ carbon\nis taken out of the month's $R_h$ booked\nas CO$_2$ (carbon partition)", "#f3e6e6"),
    ]
    tops = (70.0, 53.0, 43.0, 24.5); hs = (13.0, 7.0, 14.5, 13.0)
    for (head, body, fc), tp, h in zip(boxes, tops, hs):
        ax.add_patch(Rectangle((bx0, tp - h), bw, h, fc=fc, ec="none"))
        ax.text(bx0 + 1.2, tp - 1.2, head, fontsize=7.8, weight="bold", color=INK, va="top")
        ax.text(bx0 + 1.2, tp - 4.0, body, fontsize=6.8, color=INK2, va="top", linespacing=1.3)
    for tp, h, nxt in zip(tops[:-1], hs[:-1], tops[1:]):
        ax.add_patch(FancyArrowPatch((bx0 + 8, tp - h), (bx0 + 8, nxt), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.8))
    fig.savefig(OUT / "fig_14_1_methane_scheme.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 14.2: response of the flux to the water table and to the temperature, by band
# ---------------------------------------------------------------------------------------------------------------

def fig_14_2():
    import jax.numpy as jnp
    from ibis.bgc.methane import (MethaneParams, wetland_water_table, anaerobic_substrate_fraction, surviving_fraction,
                                  gross_production)
    p = _band_params()
    hsoi = jnp.asarray([0.10, 0.15, 0.25, 0.50, 1.0, 2.0])
    # (a) water table from a uniform pore fill theta of the top 0.5 m (no ice: wsoi = theta, wisoi = 0)
    th = np.linspace(0.0, 1.0, 401)
    zw = np.asarray(wetland_water_table(jnp.asarray(np.repeat(th[:, None], 6, 1)), jnp.zeros((th.size, 6)), hsoi, MethaneParams()))
    # (b) the two depth factors, (c) the production per unit Rh, by band
    z = np.linspace(0.0, 0.30, 301)
    T = np.linspace(-5.0, 35.0, 801) + 273.15
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.9), gridspec_kw={"wspace": 0.5})
    ax = axs[0]
    ax.plot(th, zw, color=INK, lw=1.2)
    ax.axvline(0.95, color=INK2, lw=0.7, ls=":")
    ax.text(0.935, 0.30, "$\\theta_c$", ha="right", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0.32, -0.01)
    ax.set_xlabel("pore fill $\\bar{\\theta}$ of the top 0.5 m"); ax.set_ylabel("water table $z_w$ (m)")
    ax.set_title("(a) water table, (14.1)-(14.2)", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[1]
    for b in range(3):
        q = MethaneParams(z_tau=float(p.z_tau[b]), tau_oxid=float(p.tau_oxid[b]))
        fa = np.asarray(anaerobic_substrate_fraction(jnp.asarray(z), q)); fs = np.asarray(surviving_fraction(jnp.asarray(z), q))
        ax.plot(z, fa, color=BAND_COLORS[b], lw=1.2)
        ax.plot(z, fs, color=BAND_COLORS[b], lw=1.2, ls=(0, (4, 2)))
    ax.set_xlim(0, 0.30); ax.set_ylim(0, 1.02)
    ax.set_xlabel("water table $z_w$ (m)"); ax.set_ylabel("share")
    ax.set_title("(b) depth factors, (14.5), (14.7)", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.2, label="$R_h$ below $z_w$"),
                       Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label="CH$_4$ escaping oxidation")],
              frameon=False, fontsize=6.5, loc="lower left", handlelength=2.2)
    ax = axs[2]
    tref = float(np.asarray(p.t_ref).ravel()[0])                  # not banded in the release: the base value
    for b in range(3):
        q = MethaneParams(r_ch4=float(p.r_ch4[b]), q10p=float(p.q10p[b]), t_ref=tref)
        g = np.asarray(gross_production(jnp.ones(T.size), jnp.asarray(T), q))
        ax.plot(T - 273.15, g, color=BAND_COLORS[b], lw=1.2)
    ax.axvline(tref - 273.15, color=INK2, lw=0.7, ls=":")
    ax.text(tref - 273.15 - 0.6, 0.47, "$T_{ref}$", ha="right", fontsize=6.8, color=INK2)
    ax.set_xlim(-5, 35); ax.set_ylim(0, 0.5)
    ax.set_xlabel("production temperature $T_a$ (°C)")
    ax.set_ylabel("CH$_4$ produced per unit $R_h$ below $z_w$\n(kg C per kg C)")
    ax.set_title("(c) production, (14.6)", loc="left", fontsize=8.5, weight="bold", color=INK)
    for a in axs:
        _spines(a)
    fig.legend(handles=[Line2D([], [], color=c, lw=1.6, label=n) for c, n in zip(BAND_COLORS, BAND_NAMES)],
               loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.13))
    fig.savefig(OUT / "fig_14_2_response_curves.png")
    plt.close(fig)
    zt = np.asarray(p.z_tau); to = np.asarray(p.tau_oxid)
    print("14.2 at z_w 0.1 m: Rh share below", np.round(np.exp(-0.1 / zt), 3), "surviving", np.round(np.exp(-0.1 / to), 3),
          "r x Q10 at 10 / 20 / 30 C", [np.round(np.asarray(p.r_ch4) * np.asarray(p.q10p) ** ((t + 273.15 - tref) / 10.0), 3) for t in (10, 20, 30)])


# ---------------------------------------------------------------------------------------------------------------
# Figure 14.3: the wetland bands and the WAD2M climatology
# ---------------------------------------------------------------------------------------------------------------

def fig_14_3():
    from ibis.io.wad2m import wad2m_cell_index, read_wad2m_climatology
    land, I, J, lat, lon = _land_points()
    band, from_zone = _bands(3, I, J, lat)                          # the hindcast's period, 1991-2020
    li, lo = wad2m_cell_index(lat, lon)
    clim = read_wad2m_climatology(WAD2M, li, lo, np.arange(-90.0, 90.0, 0.5), np.arange(-180.0, 180.0, 0.5))
    ann = clim.mean(axis=0)
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 8.4), gridspec_kw={"hspace": 0.42})
    ax = axs[0]
    cat = np.where(from_zone, band, 3).astype(float)
    g = np.full(land.shape, np.nan); g[I, J] = cat
    cols = list(BAND_COLORS) + ["#bdbdb8"]
    ax.imshow(g, extent=(-180, 180, -90, 90), cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(-0.5, 4), 4),
              interpolation="nearest")
    _coast(ax, land); _map_axes(ax)
    ax.set_title("(a) wetland band of each land cell, Köppen map of 1991–2020", loc="left", fontsize=9, weight="bold", color=INK)
    lab = [f"{BAND_NAMES[b]} ({int(((band == b) & from_zone).sum()):,})" for b in range(3)]
    nlr = [int(((band == b) & ~from_zone).sum()) for b in range(3)]
    lab.append(f"no wetland type: band by latitude ({int((~from_zone).sum()):,})")
    ax.legend(handles=[Patch(fc=c, ec="none", label=l) for c, l in zip(cols, lab)], loc="upper center",
              bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.2)
    ax = axs[1]
    ramp = ["#dbe8f8", "#b4cff0", "#86b0e5", "#5a8fd8", "#2a78d6", "#1d5aa6", "#123c73"]
    bounds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]
    ax.imshow(np.where(land, 0.0, np.nan), extent=(-180, 180, -90, 90), cmap=ListedColormap(["#c9c9c5"]), interpolation="nearest")
    g = np.full(land.shape, np.nan); g[I, J] = np.where(ann >= bounds[0], ann, np.nan)
    im = ax.imshow(g, extent=(-180, 180, -90, 90), cmap=ListedColormap(ramp), norm=BoundaryNorm(bounds, len(ramp)),
                   interpolation="nearest")
    _coast(ax, land); _map_axes(ax)
    ax.set_title("(b) WAD2M inundated fraction, mean of the 2000–2020 climatology", loc="left", fontsize=9, weight="bold", color=INK)
    cb = fig.colorbar(im, cax=ax.inset_axes([0.08, -0.2, 0.84, 0.035]), orientation="horizontal", ticks=bounds)
    cb.ax.set_xticklabels(["0.001", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "1"])
    cb.set_label("annual-mean inundated fraction of the cell", fontsize=8); cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    fig.savefig(OUT / "fig_14_3_bands_wad2m.png")
    plt.close(fig)
    print("14.3 bands from zone", [int(((band == b) & from_zone).sum()) for b in range(3)], "latitude rule by band", nlr,
          "cells with annual-mean fraction >= 0.001", int((ann >= 0.001).sum()), "> 0", int((ann > 0).sum()),
          "mean over land", float(clim.mean()))


# ---------------------------------------------------------------------------------------------------------------
# Figure 14.4: the learned inundated area against the climatology and the observed months, by band, 2001-2010
# ---------------------------------------------------------------------------------------------------------------

def _area_series():
    import xarray as xr
    from ibis.io.wad2m import wad2m_cell_index, read_wad2m_climatology
    land, I, J, lat, lon = _land_points()
    band, _ = _bands(3, I, J, lat)
    area = _land_area(I, J, lat)
    li, lo = wad2m_cell_index(lat, lon)
    clim = read_wad2m_climatology(WAD2M, li, lo, np.arange(-90.0, 90.0, 0.5), np.arange(-180.0, 180.0, 0.5))
    wd = xr.open_dataset(WAD2M); yrs = wd["time"].dt.year.values
    A = {k: np.zeros((120, 3)) for k in ("obs", "clim", "learned")}
    n = 0
    for y in range(2001, 2011):
        i0 = int(np.where(yrs == y)[0][0])
        for m in range(12):
            fo = np.clip(np.nan_to_num(np.asarray(wd["Fw"].isel(time=i0 + m).values, float)), 0.0, 1.0)[li, lo]
            with xr.open_dataset(f"{HINDCAST}/ibis_monthly_{y}_{m + 1:02d}.nc") as d:
                fl = np.nan_to_num(np.asarray(d["finund"].values, float))[I, J]
            for b in range(3):
                s = band == b
                A["obs"][n, b] = (fo * area)[s].sum() / 1e12
                A["clim"][n, b] = (clim[m] * area)[s].sum() / 1e12
                A["learned"][n, b] = (fl * area)[s].sum() / 1e12
            n += 1
    return A


def fig_14_4():
    A = _area_series()
    t = 2001 + (np.arange(120) + 0.5) / 12.0
    yrs = np.arange(2001, 2011)
    fig, axs = plt.subplots(2, 3, figsize=(8.8, 5.0), gridspec_kw={"hspace": 0.5, "wspace": 0.34, "height_ratios": [1.15, 1]})
    sty = {"obs": dict(color=INK2, lw=0.9, ls=(0, (3, 1.5)), label="WAD2M, the month observed"),
           "clim": dict(color=ORANGE, lw=1.1, ls=(0, (1, 1.2)), label="WAD2M climatology 2000–2020"),
           "learned": dict(color=BLUE, lw=1.2, label="learned (release run)")}
    stats = []
    for b in range(3):
        ax = axs[0, b]
        for k in ("obs", "learned", "clim"):                       # the climatology last, dotted: it lies close to the learned area
            ax.plot(t, A[k][:, b], **sty[k])
        ax.set_xlim(2001, 2011); ax.set_xticks([2002, 2004, 2006, 2008, 2010])
        ax.set_title(f"({'abc'[b]}) {BAND_NAMES[b]}", loc="left", fontsize=8, weight="bold", color=INK)
        ax.set_ylabel("inundated area (Mkm$^2$)" if b == 0 else "")
        ax = axs[1, b]
        an = {k: A[k][:, b].reshape(10, 12).mean(axis=1) for k in A}
        for k in ("obs", "learned"):
            ax.plot(yrs, an[k] - an[k].mean(), marker="o", ms=3.0, **{**sty[k], "lw": 1.0})
        ax.axhline(0.0, color=INK2, lw=0.6)
        ax.set_xlim(2000.5, 2010.5); ax.set_xticks([2002, 2004, 2006, 2008, 2010])
        ax.set_title(f"({'def'[b]}) annual-mean anomaly", loc="left", fontsize=8, weight="bold", color=INK)
        ax.set_ylabel("Mkm$^2$" if b == 0 else "")
        r = float(np.corrcoef(an["obs"], an["learned"])[0, 1])
        stats.append((BAND_NAMES[b], an["obs"].mean(), an["clim"].mean(), an["learned"].mean(),
                      an["obs"].std(), an["learned"].std(), r))
    for a in axs.ravel():
        _spines(a); a.tick_params(labelsize=7)
    fig.legend(handles=[Line2D([], [], **{k_: v for k_, v in sty[k].items()}) for k in ("obs", "clim", "learned")],
               loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(OUT / "fig_14_4_learned_area.png")
    plt.close(fig)
    for s in stats:
        print("14.4 %s: annual-mean area obs %.3f clim %.3f learned %.3f Mkm2; sd of annual means obs %.4f learned %.4f; r %.2f" % s)


# ---------------------------------------------------------------------------------------------------------------
# Figure 14.5: production temperature against the near-surface soil temperature, and the emission, by month
# ---------------------------------------------------------------------------------------------------------------

C2CH4 = 16.043 / 12.011                    # kg CH4 per kg C, as the release scoring (verify/.../diagnostics/mip_compare.py)
MONTHS = "JFMAMJJASOND"


def _monthly_temperatures():
    """Northern-Hemisphere cells of the temperate (W02) and boreal (W03) bands, release hindcast 2001-2010, month by
    month: the production temperature T_a (14.4) of the month-mean soil state through the model's own functions
    (wetland_water_table, thawed_anaerobic_zone), by the rule of the current code (each thawed part below the water table
    weighted by the integral of exp(-z/z_tau) over it, z_tau of the cell's band; MethaneParams.t_anaer_substrate, the
    default since the technote review, D13) and by the release run's rule (thickness weights to 4 m), and the
    thickness-weighted temperature of the top 0.5 m, all weighted by finund x area x Rh over the cells with thawed soil
    below the water table (the same cells under both rules); the share of each rule's weight below 1 m; and the
    emission of the month (amch4, Tg CH4; the release run's). Returns arrays (12, 2) averaged over the ten years, the
    number of cells averaged in each calendar month (mean of the ten years), and, as a check on the changing set of cells,
    T_a over the fixed set of cells with thawed soil below the water table in all 120 months, with that set's share of
    the band's NH emission. Also, for every band and both hemispheres, the release emission rescaled by the Q10 factor of
    the change of T_a between the two rules (zero where no soil below the water table is thawed, D61): a first-order
    estimate, on month-mean states and at the release parameters, of what the new rule alone does to the emission."""
    import xarray as xr
    import jax.numpy as jnp
    from ibis.bgc.methane import MethaneParams, wetland_water_table, thawed_anaerobic_zone, anaerobic_fraction
    from ibis.io.soil_init import HSOI_LAYERS
    land, I, J, lat, lon = _land_points()
    band, _ = _bands(3, I, J, lat)
    area = _land_area(I, J, lat)
    pb = _band_params()
    zt_band = np.asarray(pb.z_tau, float); q10_band = np.asarray(pb.q10p, float)
    zt_cell = zt_band[band]                                              # the band's z_tau of each cell (Table 14.1)
    q10_cell = q10_band[band]
    hs = jnp.asarray(HSOI_LAYERS, jnp.float32)
    hsn = np.asarray(HSOI_LAYERS, float)
    bot = np.cumsum(hsn); top = bot - hsn
    deep = np.asarray(bot > 1.0 + 1e-9)                                 # layers below 1 m
    S = {k: np.zeros((12, 2)) for k in ("ta", "ta_rel", "t05", "w", "deep", "deep_rel", "ch4", "n")}
    cta = np.zeros((12, I.size)); cta_rel = np.zeros((12, I.size)); cw = np.zeros((12, I.size))
    nok = np.zeros(I.size); cch4 = np.zeros(I.size)
    E = np.zeros((3, 2))                                                 # per band: release emission, rescaled emission
    for y in range(2001, 2011):
        leap = (y % 4 == 0)
        for m in range(12):
            nd = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[m]
            with xr.open_dataset(f"{HINDCAST}/ibis_monthly_{y}_{m + 1:02d}.nc") as d:
                g2 = lambda v: np.nan_to_num(np.asarray(d[v].values, float))[I, J]
                def g3(v):
                    a = np.nan_to_num(np.asarray(d[v].values, float))
                    return a[I, J, :] if a.shape[-1] == 6 else a[:, I, J].T
                fin, ch4 = g2("finund"), g2("amch4")
                rh = (g2("amco2mic") + ch4 * nd) / nd                       # Rh as CO2 + the CH4 carbon (section 14.8)
                ts, ws, wi = g3("tsoi_mean"), g3("wsoi_mean"), g3("wisoi_mean")
            ts32 = jnp.asarray(ts, jnp.float32)
            zw = wetland_water_table(jnp.asarray(ws, jnp.float32), jnp.asarray(wi, jnp.float32), hs, MethaneParams())
            ta, fth = thawed_anaerobic_zone(ts32, zw, hs, z_tau=jnp.asarray(zt_cell, jnp.float32))      # current rule
            ta_rel, fth_rel = thawed_anaerobic_zone(ts32, zw, hs)                                     # release rule
            ta = np.asarray(ta, float); ta_rel = np.asarray(ta_rel, float); fth = np.asarray(fth, float)
            assert np.array_equal(fth, np.asarray(fth_rel, float))
            thaw = ts >= 273.15
            zwn = np.asarray(zw, float)[:, None]
            wk_rel = np.asarray(anaerobic_fraction(zw, hs), float) * hsn[None, :] * thaw          # thickness weights
            a_ = np.maximum(zwn, top[None, :]); b_ = np.maximum(zwn, bot[None, :])               # substrate weights
            wk = np.exp(-(a_ - zwn) / zt_cell[:, None]) * -np.expm1(-(b_ - a_) / zt_cell[:, None]) * thaw
            fdeep = wk[:, deep].sum(1) / np.maximum(wk.sum(1), 1e-30)
            fdeep_rel = wk_rel[:, deep].sum(1) / np.maximum(wk_rel.sum(1), 1e-12)
            t05 = (ts[:, :3] * HSOI_LAYERS[:3]).sum(1) / HSOI_LAYERS[:3].sum()
            okall = (fth > 0) & (fin > 0)
            wall = fin * area * np.maximum(rh, 0.0)
            cta[m] += np.where(okall, ta * wall, 0.0); cta_rel[m] += np.where(okall, ta_rel * wall, 0.0)
            cw[m] += np.where(okall, wall, 0.0); nok += okall
            em = ch4 * nd * area
            cch4 += em
            fac = np.where(fth > 0, q10_cell ** ((ta - ta_rel) / 10.0), 0.0)
            for b in range(3):
                E[b, 0] += em[band == b].sum(); E[b, 1] += (em * fac)[band == b].sum()
            for c, b in enumerate((1, 2)):
                ok = (band == b) & (lat > 0) & okall
                w = wall[ok]
                S["ta"][m, c] += (ta[ok] * w).sum(); S["ta_rel"][m, c] += (ta_rel[ok] * w).sum()
                S["t05"][m, c] += (t05[ok] * w).sum(); S["w"][m, c] += w.sum()
                S["deep"][m, c] += (fdeep[ok] * w).sum(); S["deep_rel"][m, c] += (fdeep_rel[ok] * w).sum()
                S["n"][m, c] += ok.sum() / 10.0
                S["ch4"][m, c] += em[(band == b) & (lat > 0)].sum() * C2CH4 / 1e9 / 10.0
    fixed = {}
    for c, b in enumerate((1, 2)):
        nh = (band == b) & (lat > 0); fx = nh & (nok == 120)
        fixed[c] = (cta[:, fx].sum(1) / cw[:, fx].sum(1) - 273.15, cta_rel[:, fx].sum(1) / cw[:, fx].sum(1) - 273.15,
                    int(fx.sum()), float(cch4[fx].sum() / cch4[nh].sum()))
    return {"ta": S["ta"] / S["w"] - 273.15, "ta_rel": S["ta_rel"] / S["w"] - 273.15, "t05": S["t05"] / S["w"] - 273.15,
            "ch4": S["ch4"], "deep": S["deep"].sum(0) / S["w"].sum(0), "deep_rel": S["deep_rel"].sum(0) / S["w"].sum(0),
            "n": S["n"], "fixed": fixed, "E": E, "z_tau": zt_band, "q10": q10_band}


def fig_14_5():
    R = _monthly_temperatures()
    x = np.arange(1, 13)
    cols = (ORANGE, AQUA); names = ("W02 temperate", "W03 boreal and arctic")
    fig, axs = plt.subplots(1, 3, figsize=(8.8, 3.1), gridspec_kw={"wspace": 0.36})
    for c in range(2):
        ax = axs[c]
        ax.plot(x, R["ta"][:, c], color=cols[c], lw=1.4, marker="o", ms=3.0)
        ax.plot(x, R["ta_rel"][:, c], color=cols[c], lw=1.0, ls=(0, (1, 1.3)))
        ax.plot(x, R["t05"][:, c], color=cols[c], lw=1.1, ls=(0, (4, 2)))
        ax.set_title(f"({'ab'[c]}) {names[c]}, NH", loc="left", fontsize=8.5, weight="bold", color=INK)
        ax.set_ylabel("temperature (°C)" if c == 0 else "")
    lo = min(R["ta"].min(), R["ta_rel"].min(), R["t05"].min()); hi = max(R["ta"].max(), R["ta_rel"].max(), R["t05"].max())
    for ax in axs[:2]:
        ax.set_ylim(np.floor(lo / 5.0) * 5.0, np.ceil(hi / 5.0) * 5.0)
    ax = axs[2]
    for c in range(2):
        ax.plot(x, R["ch4"][:, c], color=cols[c], lw=1.4)
    ax.set_ylim(0, None)
    ax.set_title("(c) emission, NH, release run", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax.set_ylabel("Tg CH$_4$ month$^{-1}$")
    for a in axs:
        a.set_xlim(0.5, 12.5); a.set_xticks(x); a.set_xticklabels(list(MONTHS)); _spines(a); a.tick_params(labelsize=7)
    fig.legend(handles=[Line2D([], [], color=INK2, lw=1.4, marker="o", ms=3.0,
                               label="production temperature $T_a$, (14.4), weighted by the decomposition profile"),
                        Line2D([], [], color=INK2, lw=1.0, ls=(0, (1, 1.3)),
                               label="$T_a$ by the release run's rule (layer thickness to 4 m)"),
                        Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 2)), label="soil temperature, top 0.5 m"),
                        Line2D([], [], color=ORANGE, lw=1.6, label=names[0]),
                        Line2D([], [], color=AQUA, lw=1.6, label=names[1])],
               loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(OUT / "fig_14_5_production_temperature.png")
    plt.close(fig)
    for c in range(2):
        for key, lab in (("ta", "T_a substrate-weighted (current)"), ("ta_rel", "T_a thickness-weighted (release)")):
            ta = R[key][:, c]
            print(f"14.5 {names[c]} NH: {lab} {ta.min():.3f} (m{ta.argmin() + 1}) to {ta.max():.3f} (m{ta.argmax() + 1}), "
                  f"range {ta.max() - ta.min():.3f} K; annual mean {ta.mean():.2f}; Q10 factor change over the range "
                  f"{100 * (R['q10'][c + 1] ** ((ta.max() - ta.min()) / 10.0) - 1):.1f} %")
        t5 = R["t05"][:, c]
        print(f"14.5 {names[c]} NH: top 0.5 m {t5.min():.1f} (m{t5.argmin() + 1}) to {t5.max():.1f} (m{t5.argmax() + 1}); "
              f"share of the weight below 1 m: current {R['deep'][c]:.4f}, release {R['deep_rel'][c]:.4f}; "
              f"emission by month {np.round(R['ch4'][:, c], 2)}, year {R['ch4'][:, c].sum():.2f} Tg, Dec-Mar "
              f"{R['ch4'][[11, 0, 1, 2], c].sum():.2f} Tg")
        print(f"14.5 {names[c]} NH: cells averaged by month {np.round(R['n'][:, c]).astype(int)}")
        tf, tfr, nf, sf = R["fixed"][c]
        print(f"14.5 {names[c]} NH, fixed set of {nf} cells thawed below z_w in all 120 months ({100 * sf:.0f} % of the NH "
              f"emission): T_a current {tf.min():.1f} (m{tf.argmin() + 1}) to {tf.max():.1f} (m{tf.argmax() + 1}); release "
              f"{tfr.min():.1f} (m{tfr.argmin() + 1}) to {tfr.max():.1f} (m{tfr.argmax() + 1})")
    for b in range(3):
        e0, e1 = R["E"][b]
        print(f"14.5 band {BAND_NAMES[b]} (both hemispheres): release emission x Q10^(dT_a/10) / release emission = "
              f"{e1 / e0:.4f} ({100 * (e1 / e0 - 1):+.1f} %), z_tau {R['z_tau'][b]:.4f}, Q10 {R['q10'][b]:.4f}")
    print(f"14.5 all bands: {100 * (R['E'][:, 1].sum() / R['E'][:, 0].sum() - 1):+.1f} %")


# ---------------------------------------------------------------------------------------------------------------
# Figure 14.6: the emission of the release hindcast, map, zonal profile and latitude bands against the GMB ensemble
# ---------------------------------------------------------------------------------------------------------------

GMB_WET = "/shared/nas2/Hangkai/ibis/inputs/eval/mip/gmb2024/GMB2024_BU_wetlands_2000-2009.csv"


def fig_14_6():
    import csv
    import xarray as xr
    land, I, J, lat, lon = _land_points()
    area = _land_area(I, J, lat)                                       # cell area x ESA CCI 1992 land fraction (14.10)
    cell = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat))  # whole cell
    years = range(2001, 2011)
    ann = []
    for y in years:
        with xr.open_dataset(f"{HINDCAST}/ibis_yearly_{y}.nc") as d:
            ann.append(np.nan_to_num(np.asarray(d["aych4"].values, float))[I, J])     # kg C m-2 yr-1
    ann = np.asarray(ann)
    a = ann.mean(0)
    gch4 = a * C2CH4 * 1e3                                             # g CH4 m-2 yr-1
    tot = lambda v, w, m=None: float(((v * w) if m is None else (v * w)[m]).sum() * C2CH4 / 1e9)
    lb = ((lat < 30.0), (lat >= 30.0) & (lat < 60.0), (lat >= 60.0))
    ibis_land = [tot(a, area, m) for m in lb]; ibis_cell = [tot(a, cell, m) for m in lb]
    yr_tot = np.array([tot(v, area) for v in ann])
    rows = list(csv.DictReader(open(GMB_WET)))
    cols_ = ("90S-30N", "30N-60N", "60N-90N")
    gmb = np.array([[float(r[k]) for k in cols_ + ("Global",)] for r in rows])          # (runs, 4)
    # --- the figure
    fig = plt.figure(figsize=(8.8, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.3, 0.95, 1.35], wspace=0.12)
    ext = (-180, 180, -90, 90)
    bounds = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]
    ramp = ["#fde3cb", "#f9c493", "#f49b5c", "#e3713a", "#c4521f", "#963a12", "#5e2408"]
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.where(land, 0.0, np.nan), extent=ext, cmap=ListedColormap([LAND_BG]), interpolation="nearest")
    g = np.full(land.shape, np.nan); g[I, J] = np.where(gch4 >= bounds[0], gch4, np.nan)
    im = ax.imshow(g, extent=ext, cmap=ListedColormap(ramp), norm=BoundaryNorm(bounds, len(ramp)), interpolation="nearest")
    _coast(ax, land); _map_axes(ax); ax.set_ylim(-60, 90); ax.set_yticks(range(-60, 91, 30))
    ax.set_yticklabels([f"{abs(t)}°{'S' if t < 0 else 'N' if t > 0 else ''}" for t in range(-60, 91, 30)])
    ax.set_title("(a) wetland CH$_4$ emission, 2001–2010", loc="left", fontsize=8.8, weight="bold", color=INK)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.1, aspect=45, ticks=bounds)
    cb.ax.set_xticklabels(["0.01", "0.1", "0.3", "1", "3", "10", "30", "100"]); cb.ax.tick_params(labelsize=6.8, length=2)
    cb.outline.set_linewidth(0.4); cb.set_label("g CH$_4$ m$^{-2}$ yr$^{-1}$ (cells below 0.01 grey)", fontsize=7.5)
    # (b) zonal sum per 2 degrees of latitude, the convention of the release totals
    edges = np.arange(-60, 91, 2.0); mid = 0.5 * (edges[:-1] + edges[1:])
    zon = np.array([tot(a, area, (lat >= edges[k]) & (lat < edges[k + 1])) for k in range(len(mid))])
    axz = fig.add_subplot(gs[0, 1])
    axz.plot(zon, mid, color=BLUE, lw=1.4)
    axz.set_ylim(-60, 90); axz.set_yticks(range(-60, 91, 30)); axz.set_yticklabels([])
    axz.set_xlim(0, None); axz.set_xlabel("Tg CH$_4$ yr$^{-1}$\nper 2° of latitude", fontsize=7.5)
    axz.set_title("(b) zonal sum", loc="left", fontsize=8.8, weight="bold", color=INK)
    _spines(axz); axz.tick_params(labelsize=7)
    # (c) the three latitude bands of the Global Methane Budget
    axb = fig.add_subplot(gs[0, 2])
    x = np.arange(3); wbar = 0.26
    axb.bar(x - wbar, ibis_land, wbar, color=BLUE, label="IBIS 3.0, land fraction (release)")
    axb.bar(x, ibis_cell, wbar, color="#9cc3ee", label="IBIS 3.0, whole cell area")
    gm = gmb[:, :3].mean(0)
    axb.bar(x + wbar, gm, wbar, color=ORANGE, label="GMB 2000–2009, 22 runs")
    axb.errorbar(x + wbar, gm, yerr=[gm - gmb[:, :3].min(0), gmb[:, :3].max(0) - gm], fmt="none", ecolor=INK2, elinewidth=0.8,
                 capsize=2)
    axb.set_xticks(x); axb.set_xticklabels(["90°S–\n30°N", "30–\n60°N", "60–\n90°N"], fontsize=7)
    axb.set_ylabel("Tg CH$_4$ yr$^{-1}$", fontsize=7.5); axb.yaxis.set_label_position("right"); axb.yaxis.tick_right()
    axb.set_title("(c) latitude bands", loc="left", fontsize=8.8, weight="bold", color=INK)
    axb.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.45, -0.3), handlelength=1.0)
    for s in ("top", "left"):
        axb.spines[s].set_visible(False)
    axb.grid(axis="y", color=GRID, lw=0.5); axb.set_axisbelow(True); axb.tick_params(labelsize=7)
    fig.canvas.draw()
    p = ax.get_position(); pz = axz.get_position(); pb = axb.get_position()
    axz.set_position([pz.x0, p.y0, pz.width, p.height]); axb.set_position([pb.x0 + 0.02, p.y0, pb.width, p.height])
    fig.savefig(OUT / "fig_14_6_release_emission.png")
    plt.close(fig)
    q = np.percentile(gch4[gch4 > 0], [10, 50, 90, 99])
    print(f"14.6 total {yr_tot.mean():.2f} Tg CH4/yr (years {yr_tot.min():.1f}-{yr_tot.max():.1f}, sd ddof0 {yr_tot.std():.2f}, "
          f"ddof1 {yr_tot.std(ddof=1):.2f}); whole cell {sum(ibis_cell):.2f}")
    print("14.6 latitude bands, land fraction", np.round(ibis_land, 2), "whole cell", np.round(ibis_cell, 2),
          "GMB mean", np.round(gm, 1), "min", np.round(gmb[:, :3].min(0), 1), "max", np.round(gmb[:, :3].max(0), 1),
          "GMB global mean %.1f [%.1f-%.1f], runs %d" % (gmb[:, 3].mean(), gmb[:, 3].min(), gmb[:, 3].max(), len(gmb)))
    print(f"14.6 cells with emission > 0: {int((gch4 > 0).sum())}; >= 0.01 g: {int((gch4 >= 0.01).sum())}; percentiles 10/50/90/99 of "
          f"the cells > 0: {np.round(q, 3)} g CH4 m-2 yr-1; largest {gch4.max():.1f}")
    pk = np.argsort(zon)[::-1][:3]
    print("14.6 zonal sum, largest 2-degree bands:", [(f"{edges[k]:.0f} to {edges[k + 1]:.0f}", round(float(zon[k]), 2)) for k in pk])


FIGURES = {"14.1": fig_14_1, "14.2": fig_14_2, "14.3": fig_14_3, "14.4": fig_14_4, "14.5": fig_14_5, "14.6": fig_14_6}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
        print("figure", key, "done")
