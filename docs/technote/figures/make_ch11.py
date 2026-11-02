"""Figures of Chapter 11 of the IBIS 3.0 technical note (autotrophic respiration, allocation, vegetation dynamics).

Run from the repository root with the project interpreter (it needs matplotlib, xarray and the `ibis` package):

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch11.py            # all figures
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch11.py 11.3       # some

Every number a figure shows is computed here: the pool trajectories by the model's own annual update
(`vegetation_dynamics.biomass_update`), the cover curves by `vegetation_dynamics.dynaveg` on synthetic cells, the
cover limit by `stress_accum.tree_cover_max` with the zone ramps that `pft_types.apply_types` gathers from the release
parameter file, and the maps from the release run's output (verify_outputs/final_rel30/hindcast/out). Typed in: the
labels, the drawing of the schematic, and the IBIS 2.6b4 reference curves of Figure 11.4, which are the Fortran
formulas (vegetation.f, lines 525-542) with the parameters of the release file where they coincide.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: E402  (also sets rcParams)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

PARAMS = str(ROOT / "python-port/configs/params/params_31s_ch4h.yaml")
KOPPEN = "/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz"
HINDCAST = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30/hindcast/out"
PAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def _release_params(npoi=1):
    from ibis.io.physics_params import load_physics_params, build_all_params
    d = load_physics_params(PARAMS)
    params, pfp, _ = build_all_params(npoi, np.zeros(npoi), d)
    return d, params, pfp


# ---------------------------------------------------------------------------------------------------------------
# Figure 11.1: the carbon of a plant functional type position through the year (schematic)
# ---------------------------------------------------------------------------------------------------------------

def fig_11_1():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.set_xlim(0, 120); ax.set_ylim(0, 70); ax.axis("off")

    def box(x, y, w, h, head, body="", fc="#f2f2f0", ec="none", hs=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 1.6, head, ha="center", va="top", fontsize=hs, weight="bold", color=INK)
        if body:
            ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=6.9, color=INK2, linespacing=1.25)

    def arrow(p, q, color=INK2, lw=0.9, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    # every time step
    ax.text(1, 69, "(a) every time step", fontsize=9, weight="bold", color=INK, va="top")
    box(1, 50, 30, 14, "net assimilation", "of the position's displayed leaves\n(leaf respiration included; Ch. 9)",
        fc="#e8f0fa")
    # living wood: the sapwood of the trees, all the wood of the shrubs; the grasses only their roots (11.3)
    box(1, 31, 30, 14, "maintenance respiration", "living wood and fine roots, with a\ntemperature factor (11.3)-(11.4)",
        fc="#f6efe0")
    # growth respiration only on a positive gain (bgc.rgrowth_positive_only, default true, as IBIS 2.6b4; review D1)
    box(1, 12, 30, 14, "growth respiration", "the share $r_g$ of a positive gain,\nassimilation less maintenance (11.5)",
        fc="#f6efe0")
    arrow((16, 50), (16, 45.4)); arrow((16, 31), (16, 26.4))
    ax.text(16, 7.0, "NPP of the position,\nsummed over the year", ha="center", va="top", fontsize=7.2, color=INK)
    arrow((16, 12), (16, 7.6))
    # once a year: the pools (allocation fractions a_l, a_w, a_r as in Table 11.4)
    ax.text(37, 69, "(b) once a year", fontsize=9, weight="bold", color=INK, va="top")
    xs, wp = (38.0, 57.5, 77.0), 16.5                               # pool boxes: left edges, width
    xn = xs[1] + wp / 2                                              # the NPP box sits over the wood pool
    box(xn - 11, 55, 22, 9, "annual NPP", "", fc="#eaf4e4")
    # the three pools in one frame: disturbance takes a share of every pool (11.18)
    ax.add_patch(FancyBboxPatch((xs[0] - 1.6, 31.4), xs[2] + wp - xs[0] + 3.2, 14.2,
                                boxstyle="round,pad=0.0,rounding_size=1.6", fc="none", ec="#a33a3a", lw=0.8, ls=(0, (3, 2))))
    for x, name in zip(xs, ("leaf", "wood", "fine root")):
        box(x, 33, wp, 11, name, "carbon pool", fc="#eaf4e4")
        arrow((xn, 55), (x + wp / 2, 44.5), color="#3d6f2c")
        arrow((x + wp / 2, 31.2), (x + wp / 2, 22.6))
        ax.text(x + wp / 2 + 1.0, 27.0, "turnover $1/\\tau$", fontsize=6.6, color=INK2, va="center")
    # allocation labels, each set off its arrow on the outer side
    ax.text(55.0, 51.7, "$a_l$", ha="center", va="center", fontsize=8, color="#3d6f2c")
    ax.text(xn + 1.2, 49.5, "$a_w$", ha="left", va="center", fontsize=8, color="#3d6f2c")
    ax.text(76.6, 51.8, "$a_r$", ha="center", va="center", fontsize=8, color="#3d6f2c")
    box(xs[0], 13, xs[2] + wp - xs[0], 9, "litterfall: leaf, wood and root litter (Ch. 12)", "", fc="#efe6d8")
    # disturbance: a share of every pool; the burned carbon and the background share leave the cell
    box(102, 55, 17, 9, "atmosphere", "", fc="#e8f0fa")
    box(102, 32, 17, 12, "disturbance", "fire (Ch. 13) and\nbackground", fc="#f3e6e6")
    arrow((xs[2] + wp + 1.7, 38.5), (101.6, 38.5), color="#a33a3a")
    arrow((110.5, 44.5), (110.5, 54.6), color="#a33a3a")
    ax.text(111.5, 49.5, "burned;\nbackground", fontsize=6.4, color=INK2, va="center")
    arrow((106, 31.6), (xs[2] + wp + 0.6, 18.5), color="#a33a3a", rad=-0.15)
    # the litter share exists only with fire on and the season-length scheme; otherwise all goes up (11.19, review D29)
    ax.text(107.5, 26.0, "killed,\nnot burned\n(fire on)", fontsize=6.4, color=INK2, va="center")
    ax.text(78, 9.6, "harvest: on the cropland share, a share of the grass leaf litter leaves the cell (§11.6, Ch. 15)",
            ha="center", va="top", fontsize=6.8, color=INK2)
    ax.text(78, 5.4, "which positions grow: existence (§11.3), tree-cover limit (§11.5), minimum leaf carbon (§11.7);\n"
            "the cover of the canopies and the height of the upper one follow from the new pools (§11.8, §11.9)",
            ha="center", va="top", fontsize=6.6, color=INK2, style="italic")
    fig.savefig(OUT / "fig_11_1_carbon_flow.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 11.2: the annual pool update and the two litterfall bookings, worked on one position
# ---------------------------------------------------------------------------------------------------------------

def _pool_run(exact, npp_series, j, pfp, years):
    """Run the model's biomass_update year after year on one cell with this position's NPP; return pools and fluxes."""
    import jax.numpy as jnp
    import equinox as eqx
    from ibis.state.vegetation import create_vegetation_state
    from ibis.state.carbon import create_carbon_state
    from ibis.state.summary import create_summary_state
    from ibis.physics.vegetation_dynamics import biomass_update
    pf = eqx.tree_at(lambda p: p.exact_litterfall, pfp, exact)
    veg = create_vegetation_state(1, 12, 6)
    veg = eqx.tree_at(lambda v: v.exist, veg, jnp.zeros((1, 12)).at[0, j].set(1.0))
    car = create_carbon_state(1); sm = create_summary_state(1, 12)
    out = {"cl": [], "cw": [], "cr": [], "fall": [], "npp": []}
    for y in range(years):
        npp = float(npp_series[y])
        sm = eqx.tree_at(lambda s: s.aynpp, sm, jnp.zeros((1, 12)).at[0, j].set(npp))
        c0 = float(veg.cbiol[0, j] + veg.cbiow[0, j] + veg.cbior[0, j])
        veg, car, sm = biomass_update(veg, car, sm, pf, 1e-6)
        out["cl"].append(float(veg.cbiol[0, j])); out["cw"].append(float(veg.cbiow[0, j]))
        out["cr"].append(float(veg.cbior[0, j]))
        out["fall"].append(float(car.falll[0] + car.fallw[0] + car.fallr[0])); out["npp"].append(npp)
        out.setdefault("c0", []).append(c0)
    return {k: np.array(v) for k, v in out.items()}


def fig_11_2():
    _, _, pfp = _release_params()
    j = 3                                                            # position 4, temperate conifer evergreen
    years = 120
    yr = np.arange(1, years + 1)
    npp = np.where(yr <= 80, 0.8, 0.4)                               # kg C m-2 yr-1: a step down after 80 years
    ex = _pool_run(True, npp, j, pfp, years)
    f7 = _pool_run(False, npp, j, pfp, years)
    par = {k: float(np.asarray(getattr(pfp, k))[j]) for k in ("awood", "tauwood0", "aleaf", "tauleaf", "aroot", "tauroot")}
    eqw = par["awood"] * par["tauwood0"] * npp; eql = par["aleaf"] * par["tauleaf"] * npp
    eqr = par["aroot"] * par["tauroot"] * npp
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.9), gridspec_kw={"wspace": 0.34, "hspace": 0.62})
    axs = axs.ravel()
    ax = axs[0]
    ax.plot(yr, ex["cw"], color=PAL[0], lw=1.6, label="wood (11.11)")
    ax.plot(yr, eqw, color=PAL[0], lw=0.8, ls="--", label=r"wood at equilibrium, $a_w\,\tau_w\,N$")
    ax.set_ylabel("carbon (kg C m$^{-2}$)")
    ax.set_title("(a) wood", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[1]
    ax.plot(yr, ex["cl"], color=PAL[1], lw=1.6, label="leaf")
    ax.plot(yr, ex["cr"], color=PAL[2], lw=1.6, label="fine root")
    ax.plot(yr, eql, color=PAL[1], lw=0.8, ls="--", label=r"equilibria, $a\,\tau\,N$")
    ax.plot(yr, eqr, color=PAL[2], lw=0.8, ls="--")
    ax.set_ylabel("carbon (kg C m$^{-2}$)")
    ax.set_title("(b) leaf and fine root", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax.set_ylim(0, 0.65)
    ax = axs[2]
    dfall = ex["fall"] - f7["fall"]                                  # the pools are the same; only the booking differs
    assert np.allclose(ex["cw"], f7["cw"]) and np.allclose(ex["cl"], f7["cl"]) and np.allclose(ex["cr"], f7["cr"])
    ax.plot(yr, dfall, color=PAL[3], lw=1.6, label="exact (11.12) less $C/\\tau$ (11.13)")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("kg C m$^{-2}$ yr$^{-1}$")
    ax.set_title("(c) litterfall, exact less IBIS 2.6b4", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[3]
    tot_ex = ex["cl"] + ex["cw"] + ex["cr"]; tot_f7 = f7["cl"] + f7["cw"] + f7["cr"]
    res_ex = np.cumsum(ex["npp"] - ex["fall"] - (tot_ex - ex["c0"]))
    res_f7 = np.cumsum(f7["npp"] - f7["fall"] - (tot_f7 - f7["c0"]))
    ax.plot(yr, res_ex, color=PAL[0], lw=1.6, label="exact booking (11.12)")
    ax.plot(yr, res_f7, color=PAL[1], lw=1.6, label="IBIS 2.6b4, $C/\\tau$ (11.13)")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("kg C m$^{-2}$")
    ax.set_title("(d) unbooked carbon, cumulative", loc="left", fontsize=8.5, weight="bold", color=INK)
    for a in axs:
        _spines(a); a.set_xlim(0, years); a.set_xlabel("year")
        a.legend(frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    fig.savefig(OUT / "fig_11_2_pool_update.png")
    plt.close(fig)
    big = np.abs(dfall[5:80]).max()
    print(f"11.2 position {j+1}: {par}; year 80: wood {ex['cw'][79]:.3f} (eq {eqw[0]:.3f}), leaf {ex['cl'][79]:.3f} "
          f"(eq {eql[0]:.3f}), root {ex['cr'][79]:.3f} (eq {eqr[0]:.3f}); litterfall exact - C/tau: year 1 {dfall[0]:.4f}, "
          f"years 6-80 max |.| {big:.4f}, year 81 {dfall[80]:.4f}; residual F77 year 80 {res_f7[79]:.3f}, year 120 "
          f"{res_f7[-1]:.3f}; exact max |residual| over {years} yr {np.abs(res_ex).max():.2e} (float32)")


# ---------------------------------------------------------------------------------------------------------------
# Figure 11.3: the water-limited maximum tree cover
# ---------------------------------------------------------------------------------------------------------------

def _zone_ramps():
    """Per-zone (min, ramp, power) of the cover limit as apply_types gathers them from the release file."""
    from ibis.io import pft_types as pt
    d, params, pfp = _release_params(len(pt.ZONES))
    p2, pf2 = pt.apply_types(params, pfp, np.arange(len(pt.ZONES)), pt.parse_type_overrides(d.get("types")),
                             base=(params, pfp))
    return pt.ZONES, pf2


def _fu_max(u, pf):
    from ibis.state.stress_accum import tree_cover_max
    import jax.numpy as jnp
    u = jnp.asarray(u, jnp.float32)
    st = SimpleNamespace(stress=SimpleNamespace(aystressu=u, nday=jnp.ones_like(u), u_smooth=jnp.zeros_like(u)))
    return np.asarray(tree_cover_max(st, pf))


def fig_11_3():
    import xarray as xr
    import equinox as eqx
    from ibis.io import pft_types as pt
    zones, pf2 = _zone_ramps()
    ramps = {}
    for zi, z in enumerate(zones):
        key = (round(float(pf2.tree_cover_min[zi]), 4), round(float(pf2.tree_cover_ramp[zi]), 4),
               round(float(pf2.tree_cover_power[zi]), 4))
        ramps.setdefault(key, []).append(z)
    u = np.linspace(0, 1, 401).astype(np.float32)
    fig = plt.figure(figsize=(7.6, 9.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 1.05], hspace=0.32)
    ax = fig.add_subplot(gs[0].subgridspec(1, 2, width_ratios=[0.56, 0.44], wspace=0.05)[0])
    order = sorted(ramps, key=lambda k: (k[0], k[1]))
    for i, key in enumerate(order):
        zs = [z for z in ramps[key] if z != "none"]
        pfz = eqx.tree_at(lambda p: (p.tree_cover_min, p.tree_cover_ramp, p.tree_cover_power), pf2,
                          (np.float32(key[0]), np.float32(key[1]), np.float32(key[2])))
        lab = ", ".join(zs[:3]) + ("," if len(zs) > 3 else "") + ("\n" + ", ".join(zs[3:]) if len(zs) > 3 else "") \
            + ("; no zone" if "none" in ramps[key] else "")
        ax.plot(u, _fu_max(u, pfz), color=PAL[i], lw=1.6,
                label=f"{lab}\n({key[0]:.2f}, {key[1]:.2f}, {key[2]:g})")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"smoothed stress factor $\tilde{u}$ (1 = unstressed)")
    ax.set_ylabel(r"maximum tree cover $f_u^{max}$")
    ax.set_title(r"(a) $f_u^{max}(\tilde{u})$ by zone", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", handlelength=1.6, bbox_to_anchor=(1.03, 1.02),
              title=r"zones ($u_0$, $\Delta u$, $p$)", title_fontsize=6.8, alignment="left", labelspacing=0.7)
    _spines(ax)
    # maps from the release hindcast, 2010
    land = _land_mask()
    ds = xr.open_dataset(f"{HINDCAST}/ibis_yearly_2010.nc")
    lat = np.asarray(ds.lat.values)
    flip = lat[0] < lat[-1]
    us = np.asarray(ds.aystressu_smooth.values, np.float64); fu = np.asarray(ds.fu.values, np.float64)
    if flip:
        us = us[::-1]; fu = fu[::-1]
    k = np.load(KOPPEN)
    per = int(np.nonzero((k["period_start"] <= 2010) & (k["period_end"] >= 2010))[0][0])
    zone = pt.zones_from_classes(k["classes"][per])
    li, lj = np.nonzero(land)
    import jax.numpy as jnp
    pfc = eqx.tree_at(lambda p: (p.tree_cover_min, p.tree_cover_ramp, p.tree_cover_power), pf2,
                      (jnp.asarray(np.asarray(pf2.tree_cover_min)[zone[li, lj]]),
                       jnp.asarray(np.asarray(pf2.tree_cover_ramp)[zone[li, lj]]),
                       jnp.asarray(np.asarray(pf2.tree_cover_power)[zone[li, lj]])))
    fmax_pts = _fu_max(us[li, lj], pfc)
    fmax = np.full(land.shape, np.nan); fmax[li, lj] = fmax_pts
    bounds = [0, 1e-6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975, 1.0001]
    blues = ["#d6d2c8", "#dde9f6", "#c6daf0", "#adc9ea", "#93b7e2", "#78a4d9", "#5e90cd", "#467bbd", "#3367a8",
             "#235390", "#163f75", "#0c2c58"]
    cmap = ListedColormap(blues); norm = BoundaryNorm(bounds, cmap.N)
    fu_min = float(pf2.fu_min) if np.ndim(pf2.fu_min) == 0 else float(np.asarray(pf2.fu_min).ravel()[0])
    at_floor = land & (np.abs(fu - fu_min) < 1e-6)                  # cover at the floor: no tree cover
    FLOOR_C = "#e3cf9f"
    for r, (fld, title) in enumerate(((fmax, r"(b) maximum tree cover $f_u^{max}$, 2010"),
                                      (np.where(land & ~at_floor, fu, np.nan), r"(c) upper-canopy cover $f_u$, 2010"))):
        ax = fig.add_subplot(gs[r + 1])
        ax.imshow(np.where(land, 0.0, np.nan), extent=(-180, 180, -90, 90), cmap=ListedColormap([LAND_BG]),
                  interpolation="nearest")
        im = ax.imshow(fld, extent=(-180, 180, -90, 90), cmap=cmap, norm=norm, interpolation="nearest")
        if r == 1:
            ax.imshow(np.where(at_floor, 0.0, np.nan), extent=(-180, 180, -90, 90), cmap=ListedColormap([FLOOR_C]),
                      interpolation="nearest")
        _coast(ax, land); _map_axes(ax); ax.set_ylim(-60, 90)
        ax.set_title(title, loc="left", fontsize=9, weight="bold", color=INK)
    cax = fig.add_axes([0.14, 0.065, 0.56, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.5e-6] + bounds[2:-1] + [1.0001])
    cb.ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.975", "1"],
                          fontsize=6.5)
    cb.set_label("fraction of the cell (grey in (b): exactly 0)", fontsize=7.5); cb.outline.set_linewidth(0.4)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=FLOOR_C, ec="none", label=f"(c) at the floor {fu_min:g}:\nno tree cover")],
               loc="center left", bbox_to_anchor=(0.73, 0.066), frameon=False, fontsize=7, handlelength=1.6)
    fig.savefig(OUT / "fig_11_3_tree_cover_limit.png")
    plt.close(fig)
    # numbers for the text
    lat_c = 89.75 - 0.5 * li
    w = np.cos(np.deg2rad(lat_c))
    fu_p = fu[li, lj]
    zero = fmax_pts <= 0.0
    binds = (~zero) & (np.abs(fu_p - fmax_pts) < 1e-5) & (fmax_pts < 0.975) & (fmax_pts > 0.05)
    print("11.3 ramps", {str(kk): vv for kk, vv in ramps.items()})
    print(f"11.3 2010: land cells {li.size}; fu_max = 0 on {zero.sum()} ({100*zero.mean():.1f} %, area "
          f"{100*w[zero].sum()/w.sum():.1f} %); fu = fu_max (binding, 0.05 < fu_max < 0.975) on {binds.sum()} "
          f"({100*binds.mean():.1f} %, area {100*w[binds].sum()/w.sum():.1f} %); fu_max >= 0.975 on "
          f"{int((fmax_pts >= 0.975).sum())}; fu at the floor 0.05 on {int((np.abs(fu_p-0.05)<1e-6).sum())}")
    south = lat_c < -60.0
    ice = zone[li, lj] == list(zones).index("E_ice")
    print(f"11.3 south of 60S (not on the maps): fu_max = 0 on {int((zero & south).sum())}, of them E_ice "
          f"{int((zero & south & ice).sum())}; E_ice zeros in all {int((zero & ice).sum())}")
    for zi, z in enumerate(zones):
        m = zone[li, lj] == zi
        if m.any():
            print(f"   {z:12s} cells {m.sum():6d}  fu_max=0 {zero[m].sum():6d} ({100*zero[m].mean():5.1f} %)  binding {binds[m].sum():6d} ({100*binds[m].mean():5.1f} %)"
                  f"  mean u {us[li, lj][m].mean():.3f}")


# ---------------------------------------------------------------------------------------------------------------
# Figure 11.4: fractional cover of the two canopies
# ---------------------------------------------------------------------------------------------------------------

def _dynaveg_cover(pfp, wood=None, lail=None, cropfrac=None):
    """Cover from the model's dynaveg on synthetic cells: wood in position 5 or leaf area in position 12, no NPP,
    fire off (background disturbance only), legacy fire scheme."""
    import jax.numpy as jnp
    import equinox as eqx
    from ibis.state.vegetation import create_vegetation_state
    from ibis.state.carbon import create_carbon_state
    from ibis.state.summary import create_summary_state
    from ibis.physics.vegetation_dynamics import dynaveg
    n = len(wood) if wood is not None else len(lail)
    veg = create_vegetation_state(n, 12, 6)
    veg = eqx.tree_at(lambda v: v.exist, veg, jnp.ones((n, 12)))
    if wood is not None:
        veg = eqx.tree_at(lambda v: v.cbiow, veg, jnp.zeros((n, 12)).at[:, 4].set(jnp.asarray(wood, jnp.float32)))
    if lail is not None:
        sla = float(np.asarray(pfp.specla)[11])
        veg = eqx.tree_at(lambda v: v.cbiol, veg, jnp.zeros((n, 12)).at[:, 11].set(jnp.asarray(lail, jnp.float32) / sla))
    v2, _, _ = dynaveg(veg, create_carbon_state(n), create_summary_state(n, 12), pfp, 0, 1e-6, fire_scheme="legacy",
                       cropfrac=None if cropfrac is None else jnp.full((n,), cropfrac, jnp.float32))
    return np.asarray(v2.fu), np.asarray(v2.fl), np.asarray(v2.cbiow[:, 4]), np.asarray(v2.totlail)   # totlail: all lower positions


def fig_11_4():
    _, _, pfp = _release_params()
    w = np.linspace(0.0, 8.4, 421).astype(np.float32)                 # pool before the update; W spans 0-8.2
    D = 0.005 + 0.005                                                # background + legacy fire constant (fire off)
    # leaf area at the start of the year, chosen so that after the year's turnover (no NPP) and the disturbance the
    # lower canopy spans 0-8 m2 m-2
    L = (np.linspace(0.0, 8.0, 401) / (np.exp(-1.0 / float(np.asarray(pfp.tauleaf)[11])) * (1 - D))).astype(np.float32)
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw={"wspace": 0.3})
    ax = axs[0]
    wn, fk = float(pfp.woodnorm), float(pfp.fu_k)
    for i, c in enumerate((0.0, 0.5)):
        fu, _, cw, _ = _dynaveg_cover(pfp, wood=w, cropfrac=c)
        W = 0.001 + cw / (1 - D)          # the wood the cover reads: after the year's decay, before the disturbance
        ax.plot(W, fu, color=PAL[i], lw=1.6, label=f"IBIS 3.0, cropland fraction {c:g}")
        # check against (11.23) with f_u^max = 1 (no cover limit is passed)
        eq = np.clip((1 - c) * (1 - np.exp(-W / (max(1 - c, 0.01) * fk))) / (1 - np.exp(-wn / fk)) * (1 - D),
                     float(pfp.fu_min), 0.975)
        assert np.allclose(fu, eq, atol=2e-6), (c, np.abs(fu - eq).max())
        if c == 0.0:
            W0, fu0 = W, fu
    # IBIS 2.6b4 (vegetation.f 525-530): wood scale 1 kg C m-2, floor 0.25
    ref = np.clip((1 - np.exp(-W0)) / (1 - np.exp(-wn)) * (1 - D), 0.25, 0.975)
    ax.plot(W0, ref, color=INK2, lw=0.9, ls="--", label="IBIS 2.6b4")
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.02)
    ax.set_xlabel("wood carbon of the tree positions $W$ (kg C m$^{-2}$)"); ax.set_ylabel("upper-canopy cover $f_u$")
    ax.set_title("(a) upper canopy", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right"); _spines(ax)
    ax = axs[1]
    _, fl, _, pl = _dynaveg_cover(pfp, lail=L)
    ax.plot(pl, fl, color=PAL[0], lw=1.6, label=f"IBIS 3.0, $k$ = {float(pfp.cover_k):g}")
    ax.plot(pl, np.clip(pl * (1 - D), 0.25, 0.975), color=INK2, lw=0.9, ls="--", label="IBIS 2.6b4")   # vegetation.f 542
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.02)
    ax.set_xlabel("leaf area of the lower positions $L_l$ (m$^2$ m$^{-2}$)"); ax.set_ylabel("lower-canopy cover $f_l$")
    ax.set_title("(b) lower canopy", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right"); _spines(ax)
    # check the lower curve against (11.25)
    eql = np.clip((1 - np.exp(-pfp.cover_k * pl)) * (1 - D), pfp.fl_min, 0.975)
    assert np.allclose(fl, eql, atol=2e-6), np.abs(fl - eql).max()
    fig.savefig(OUT / "fig_11_4_canopy_cover.png")
    plt.close(fig)
    print(f"11.4 fu reaches 0.975 at W = {W0[np.argmax(fu0 >= 0.975 - 1e-6)]:.2f}; fl at L = 8: {np.interp(8.0, pl, fl):.4f}"
          f" (max {fl.max():.4f}); fu at W=1: {np.interp(1.0, W0, fu0):.3f}; fl at L=1: {np.interp(1.0, pl, fl):.3f}; "
          f"fl at L=2: {np.interp(2.0, pl, fl):.3f}; fu floor up to W = {W0[fu0 <= 0.05 + 1e-6].max():.3f}")


FIGURES = {"11.1": fig_11_1, "11.2": fig_11_2, "11.3": fig_11_3, "11.4": fig_11_4}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
        print("figure", key, "done")
