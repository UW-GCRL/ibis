"""Figures of Chapter 6 (soil and bedrock temperature) of the IBIS 3.0 technical note, drawn from the model's code and the
release run.

Run from the repository root with the project interpreter (it needs matplotlib, netCDF4, xarray and the `ibis` package):

    PYTHONPATH=python-port /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch06.py          # all
    PYTHONPATH=python-port /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch06.py 6.3      # one

Every number a figure shows is computed here: the layer depths from `ibis.io.soil_init.HSOI_LAYERS` and
`ibis.physics.bedrock.BedrockParams`; the thermal conductivity from `ibis.physics.soil_physics.setsoi`; the heat
capacity by probing `ibis.physics.soil_physics.soilheat` with a known flux; the annual temperature wave by integrating
`soilheat` and `soilheat_with_bedrock`; the soil temperatures of Figure 6.4 from the monthly output of the release
hindcast (verify_outputs/final_rel30/hindcast). Nothing is typed in by hand except the labels and the schematic's drawing.

Code shown (checked 2026-11-02 against the release, commit e54e342): `ibis/physics/bedrock.py` is unchanged; in
`soil_physics.py` only the evaporation shares of `setsoi`'s third case changed (technote review D52), not the
conductivity, and `soilheat` is unchanged; `soil_init.py` changed the texture-class border rule, which leaves the
porosities of the three textures drawn here unchanged (0.437, 0.463, 0.475). So Figures 6.1-6.3 hold for the code after
the technote review; Figure 6.4 is the release run. The review's soil changes (the top-layer temperature in the heat
flux of (6.8), the order of the puddle phase change, the cold start at 278.13 K, the flux beneath deep snowpacks) are
in `timestepping/loops.py`, `state/soil.py` and `snow_physics.py`, which these figures do not use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: E402,F401  (style, rcParams)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

import jax                                                                            # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                               # noqa: E402

from ibis.constants import PhysicalConstants                                          # noqa: E402
from ibis.io.soil_init import HSOI_LAYERS, RHOSOI_VAL, CSOI_VAL, compute_soil_properties  # noqa: E402
from ibis.physics.bedrock import BedrockParams, soilheat_with_bedrock                  # noqa: E402
from ibis.physics.soil_physics import setsoi, soilheat                                 # noqa: E402

C = PhysicalConstants()
BP = BedrockParams()
PAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
HINDCAST = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30/hindcast/out/ibis_monthly_%d_%02d.nc"
# three textures for the response curves: sand and clay percentages inside the class (Table 2.6)
TEXTURES = (("sand", 92.0, 3.0), ("loam", 40.0, 20.0), ("clay", 20.0, 60.0))


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _title(ax, s):
    ax.set_title(s, loc="left", fontsize=9, weight="bold", color=INK)


def _depths():
    """Bottoms and node (mid-layer) depths of the soil and bedrock layers, m."""
    h = np.r_[HSOI_LAYERS, np.asarray(BP.hsoi)]
    bot = np.cumsum(h)
    return h, bot, bot - h / 2


# ---------------------------------------------------------------------------------------------------------------
# thermal properties from the model's functions
# ---------------------------------------------------------------------------------------------------------------

def conductivity(sand, clay, poros, wsoi, wisoi):
    """Thermal conductivity (W m-1 K-1) of one soil layer state, from ibis.physics.soil_physics.setsoi."""
    n = np.size(wsoi)
    one = jnp.ones((n, 6), jnp.float64)
    col = lambda v: jnp.asarray(np.broadcast_to(np.asarray(v, float)[..., None] if np.ndim(v) else v, (n, 6)))
    r = setsoi(sand * one, clay * one, poros * one, col(wsoi), col(wisoi), 280.0 * one, jnp.zeros(n), jnp.zeros(n),
               10.0, 0.5, jnp.full(n, 270.0), jnp.full(n, 280.0), C.tmelt, C.hvap, C.hsub, C.ch2o, C.cvap, C.cice,
               C.epsilon)
    return np.asarray(r.consoi[:, 0])


def heat_capacity(poros, wsoi, wisoi, csoi=CSOI_VAL, rhosoi=RHOSOI_VAL):
    """Volumetric heat capacity (J m-3 K-1), read off ibis.physics.soil_physics.soilheat: a single 1 m layer with no
    conduction and no water flow receives a flux F for one step, so that C = F dt / dT."""
    n = np.size(wsoi)
    col = lambda v: jnp.asarray(np.broadcast_to(np.asarray(v, float).reshape(-1, 1), (n, 1)))
    F, dt, T0 = 100.0, 3600.0, 280.0
    r = soilheat(jnp.full((n, 1), T0), jnp.full((n, 1), T0), col(wsoi), col(wsoi), col(wisoi), col(poros + 0 * np.asarray(wsoi)),
                 jnp.ones((n, 1)), col(csoi + 0 * np.asarray(wsoi)), col(rhosoi + 0 * np.asarray(wsoi)), jnp.ones(1),
                 jnp.zeros((n, 1)), jnp.zeros((n, 1)), jnp.full(n, F), jnp.zeros((n, 2)), C.ch2o, C.cice, C.rhow, dt)
    return F * dt / (np.asarray(r.tsoi[:, 0]) - T0)


def _poros(sand, clay):
    return float(compute_soil_properties(np.array([[sand] * 6]), np.array([[clay] * 6]))["poros"][0, 0])


# ---------------------------------------------------------------------------------------------------------------
# Figure 6.1: the soil and bedrock column of the heat solver
# ---------------------------------------------------------------------------------------------------------------

def fig_6_1():
    h, bot, mid = _depths()
    top = bot - h
    Y = lambda z: -np.sqrt(z)                                         # square-root depth scale
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.set_xlim(-2.6, 10.6); ax.set_ylim(Y(12.0) - 0.42, 0.62); ax.axis("off")
    x0, x1 = 0.0, 2.4
    browns = ("#c9a77c", "#bf9a6c", "#b28c5e", "#a57e52", "#987147", "#8a653e")
    greys = ("#a3a3a0", "#959592", "#878784")
    for k in range(9):
        fc = browns[k] if k < 6 else greys[k - 6]
        ax.add_patch(Rectangle((x0, Y(bot[k])), x1 - x0, Y(top[k]) - Y(bot[k]), fc=fc, ec="white", lw=0.8))
        ax.plot([(x0 + x1) / 2], [Y(mid[k])], marker="o", ms=4.2, color=INK, zorder=4)
        lab = f"$T_{{{k + 1}}}$"
        ax.text((x0 + x1) / 2 + 0.17, Y(mid[k]), lab, fontsize=7.5, color=INK, va="center")
    for z in np.r_[0.0, bot]:
        ax.text(x0 - 0.12, Y(z), f"{z:g} m", ha="right", va="center", fontsize=6.8, color=INK2)
    ax.text(x0 - 1.25, (Y(0) + Y(12)) / 2, "depth (square-root scale)", rotation=90, fontsize=7, color=INK2,
            ha="center", va="center")
    # top boundary
    ax.add_patch(FancyArrowPatch((1.2, 0.52), (1.2, 0.03), arrowstyle="-|>", mutation_scale=8, color="#b0462c", lw=1.1))
    ax.text(1.45, 0.43, r"$G_{top}$: heat flux of the snow-free and the snow-covered ground," "\n"
            r"heat of rain and meltwater, less the heat of runoff (6.8)", fontsize=7.2, color=INK, va="center")
    # conduction between nodes (one example) and water flow
    k = 2
    ax.annotate("", xy=(0.62, Y(mid[k + 1])), xytext=(0.62, Y(mid[k])),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=2, shrinkB=2))
    ax.text(0.52, (Y(mid[k]) + Y(mid[k + 1])) / 2, r"$\kappa_{4}$", fontsize=7.5, color=INK, ha="right",
            va="center")
    for kk in range(1, 6):
        ax.add_patch(FancyArrowPatch((x1 - 0.32, Y(top[kk]) + 0.06), (x1 - 0.32, Y(top[kk]) - 0.06), arrowstyle="-|>",
                                     mutation_scale=6, color="#2a78d6", lw=0.9))
    ax.add_patch(FancyArrowPatch((x1 - 0.32, Y(4.0) + 0.13), (x1 - 0.32, Y(4.0)), arrowstyle="-|>",
                                 mutation_scale=7, color="#2a78d6", lw=1.1))
    # right-hand labels
    lx = 2.9
    ax.text(lx, Y(0.17), "Soil, 6 layers to 4 m", fontsize=8, weight="bold", color=INK, va="bottom")
    ax.text(lx, Y(0.17) - 0.04, "conductivity from texture, liquid water and ice (6.1)-(6.2);\n"
            "heat capacity of solids, water, ice and the puddle (6.3)-(6.4);\n"
            r"conductance $\kappa_k$ between nodes (6.6); heat of the water flow (6.7);" "\n"
            "freezing and thawing at the melting point after each step (6.11)-(6.14)", fontsize=7.0, color=INK2,
            va="top", linespacing=1.3)
    ax.text(lx, Y(2.6), "Base of the IBIS 2.6b4 column", fontsize=8, weight="bold", color=INK, va="bottom")
    ax.text(lx, Y(2.6) - 0.04, "zero conductive flux at 4 m when grid.nbedrock = 0;\n"
            "drainage leaves here at $T_6$ in either column", fontsize=7.0, color=INK2, va="top", linespacing=1.3)
    ax.plot([x0 - 0.05, x1 + 0.35], [Y(4.0), Y(4.0)], color="#2a78d6", lw=1.0, ls=(0, (3, 2)))
    ax.text(lx, Y(6.4), "Bedrock, 3 layers to 12 m (release)", fontsize=8, weight="bold", color=INK, va="bottom")
    ax.text(lx, Y(6.4) - 0.04, f"heat conduction only; no water, ice or roots;\n"
            f"conductivity {BP.consoi:g} W m$^{{-1}}$ K$^{{-1}}$, heat capacity "
            f"{BP.volumetric_heat_capacity / 1e6:.1f} × 10$^6$ J m$^{{-3}}$ K$^{{-1}}$", fontsize=7.0, color=INK2,
            va="top", linespacing=1.3)
    ax.plot([x0 - 0.05, x1 + 0.35], [Y(12.0), Y(12.0)], color=INK, lw=1.4)
    ax.text(lx, Y(12.0), "zero heat flux at the base (12 m)", fontsize=7.2, color=INK, va="center")
    leg = [Line2D([], [], marker="o", ls="none", color=INK, ms=4, label="node: layer temperature at mid-depth"),
           Line2D([], [], color="#2a78d6", lw=1.0, marker=r"$\downarrow$", ms=7, label="water flow (Chapter 7)")]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.30, 0.0), frameon=False, fontsize=7)
    fig.savefig(OUT / "fig_6_1_column.png")
    plt.close(fig)
    print("6.1 nodes (m)", np.round(mid, 3), "bottoms", bot)


# ---------------------------------------------------------------------------------------------------------------
# Figure 6.2: conductivity, heat capacity and diffusivity against the water content of the pore space
# ---------------------------------------------------------------------------------------------------------------

def fig_6_2():
    s = np.linspace(0.0, 1.0, 101)
    fig, axs = plt.subplots(1, 3, figsize=(8.2, 3.1), gridspec_kw={"wspace": 0.38})
    out = {}
    for (name, sa, cl), col in zip(TEXTURES, PAL):
        po = _poros(sa, cl)
        lam_l = conductivity(sa, cl, po, s, 0.0 * s); lam_i = conductivity(sa, cl, po, 0.0 * s, s)
        c_l = heat_capacity(po, s, 0.0 * s); c_i = heat_capacity(po, 0.0 * s, s)
        out[name] = (po, lam_l, lam_i, c_l, c_i)
        for ax, yl, yi in ((axs[0], lam_l, lam_i), (axs[1], c_l / 1e6, c_i / 1e6),
                           (axs[2], lam_l / c_l * 1e7, lam_i / c_i * 1e7)):
            ax.plot(s, yl, color=col, lw=1.4)
            ax.plot(s, yi, color=col, lw=1.4, ls=(0, (4, 2)))
    for ax, yb in ((axs[0], BP.consoi), (axs[1], BP.volumetric_heat_capacity / 1e6),
                   (axs[2], BP.consoi / BP.volumetric_heat_capacity * 1e7)):
        ax.axhline(yb, color="#8a8a86", lw=0.9, ls=":")
    axs[0].text(0.02, BP.consoi + 0.06, "bedrock", fontsize=6.8, color=INK2, va="bottom")
    axs[1].text(0.02, BP.volumetric_heat_capacity / 1e6 + 0.03, "bedrock", fontsize=6.8, color=INK2, va="bottom")
    axs[2].text(0.02, BP.consoi / BP.volumetric_heat_capacity * 1e7 + 0.3, "bedrock", fontsize=6.8, color=INK2,
                va="bottom")
    axs[0].set_ylabel("W m$^{-1}$ K$^{-1}$"); axs[1].set_ylabel("10$^6$ J m$^{-3}$ K$^{-1}$")
    axs[2].set_ylabel("10$^{-7}$ m$^2$ s$^{-1}$")
    _title(axs[0], "(a) conductivity"); _title(axs[1], "(b) heat capacity"); _title(axs[2], "(c) diffusivity")
    axs[0].set_ylim(0, 3.6); axs[1].set_ylim(0, 3.6); axs[2].set_ylim(0, 18)
    for ax in axs:
        ax.set_xlim(0, 1); ax.set_xlabel("fraction of pores filled\nwith liquid water or ice"); _spines(ax)
    hand = [Line2D([], [], color=c, lw=1.4, label=f"{n} (porosity {out[n][0]:.3f})") for (n, _, _), c in zip(TEXTURES, PAL)]
    hand += [Line2D([], [], color=INK2, lw=1.4, label="liquid"), Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 2)), label="ice")]
    fig.legend(handles=hand, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.20),
               handlelength=2.2, columnspacing=1.2)
    fig.savefig(OUT / "fig_6_2_thermal_properties.png")
    plt.close(fig)
    for n, (po, ll, li, cl_, ci) in out.items():
        print(f"6.2 {n}: porosity {po:.3f}; dry {ll[0]:.3f}; saturated liquid {ll[-1]:.3f}, ice {li[-1]:.3f} W/m/K;"
              f" C dry {cl_[0]/1e6:.3f}, sat liquid {cl_[-1]/1e6:.3f}, sat ice {ci[-1]/1e6:.3f} MJ/m3/K;"
              f" kappa sat liquid {ll[-1]/cl_[-1]*1e7:.2f}, ice {li[-1]/ci[-1]*1e7:.2f}, max liquid {np.max(ll/cl_)*1e7:.2f}"
              f" at s={s[np.argmax(ll/cl_)]:.2f} (1e-7 m2/s)")


# ---------------------------------------------------------------------------------------------------------------
# Figure 6.3: the annual temperature wave in the 4 m and the 12 m column against a deep, finely resolved column
# ---------------------------------------------------------------------------------------------------------------

DT = 3600.0
YEAR = 365.0 * 86400.0
OMEGA = 2.0 * np.pi / YEAR


def _soil_arrays(n, w, wi, poros, lam):
    o = jnp.ones((1, n))
    return dict(owsoi=w * o, wsoi=w * o, wisoi=wi * o, poros=poros * o, consoi=lam * o, csoi_mineral=CSOI_VAL * o,
                rhosoi=RHOSOI_VAL * o, c0pud=0 * o, c1pud=0 * o)


def _with_rock(d, nr):
    z = jnp.zeros((1, nr)); o = jnp.ones((1, nr)); cat = lambda a, b: jnp.concatenate([a, b], 1)
    return dict(owsoi=cat(d["owsoi"], z), wsoi=cat(d["wsoi"], z), wisoi=cat(d["wisoi"], z), poros=cat(d["poros"], z),
                consoi=cat(d["consoi"], BP.consoi * o), csoi_mineral=cat(d["csoi_mineral"], BP.csoi * o),
                rhosoi=cat(d["rhosoi"], BP.rhosoi * o), c0pud=cat(d["c0pud"], z), c1pud=cat(d["c1pud"], z))


def annual_wave(kind, w, wi, poros, lam, nyear=12, h=None, n_soil=6, forcing="temperature"):
    """Complex amplitude of the annual cycle of each layer's temperature, from the model's soilheat ('4m': the IBIS 2.6b4
    column; 'ref': n_soil soil layers on rock, layers h) or soilheat_with_bedrock ('12m': the release column).
    Conduction only: no water flow, no phase change. forcing = 'temperature': a surface temperature A cos(wt), A = 1 K,
    coupled to the top node through the half-layer conductance 2 lambda_1 / dz_1, explicit in the top-layer
    temperature as the surface energy balance couples T_g to T_1 (Chapter 5); 'flux': a surface heat flux
    F0 cos(wt), F0 = 30 W m-2."""
    F0, A = 30.0, 1.0
    if kind in ("12m", "4m"):
        h = HSOI_LAYERS; sa = _soil_arrays(6, w, wi, poros, lam)
    else:
        sa = _with_rock(_soil_arrays(n_soil, w, wi, poros, lam), len(h) - n_soil)
    hj = jnp.asarray(h); n = len(h); nh = 6 if kind != "ref" else n_soil
    k_top = 2.0 * lam / float(h[0])

    def step(carry, t):
        T, Td = carry
        if forcing == "temperature":
            f = k_top * (A * jnp.cos(OMEGA * (t + DT)) - T[:, 0])
        else:
            f = jnp.full((1,), F0 * jnp.cos(OMEGA * t))
        if kind == "12m":
            r, Td2 = soilheat_with_bedrock(T, Td, T, Td, sa["owsoi"], sa["wsoi"], sa["wisoi"], sa["poros"], sa["consoi"],
                                           sa["csoi_mineral"], sa["rhosoi"], hj, sa["c0pud"], sa["c1pud"], f,
                                           jnp.zeros((1, 7)), C.ch2o, C.cice, C.rhow, DT, BP)
            return (r.tsoi, Td2), jnp.concatenate([r.tsoi, Td2], 1)[0]
        r = soilheat(T, T, sa["owsoi"], sa["wsoi"], sa["wisoi"], sa["poros"], sa["consoi"], sa["csoi_mineral"],
                     sa["rhosoi"], hj, sa["c0pud"], sa["c1pud"], f, jnp.zeros((1, n + 1)), C.ch2o, C.cice, C.rhow, DT,
                     n_hydro=nh)
        return (r.tsoi, Td), r.tsoi[0]

    nstep = int(round(YEAR / DT))
    ts = jnp.arange(nyear * nstep) * DT
    _, out = jax.lax.scan(jax.jit(step), (jnp.zeros((1, n)), jnp.zeros((1, 3))), ts)
    out = np.asarray(out)
    amps = []
    for yr in (nyear - 2, nyear - 1):                                  # the last two years: a check of the spin-up
        seg = out[yr * nstep:(yr + 1) * nstep]; tt = np.asarray(ts[yr * nstep:(yr + 1) * nstep])
        seg = seg - seg.mean(0)
        amps.append(2.0 * (seg * np.exp(-1j * OMEGA * tt)[:, None]).mean(0))
    assert np.max(np.abs(np.abs(amps[1]) - np.abs(amps[0])) / np.abs(amps[1][0])) < 1e-3, "annual wave not periodic yet"
    return amps[1]


def _reference_layers():
    """A deep reference column: 0.05 m layers to 4 m (soil), 0.1 m layers to 12 m and then 20 layers growing by 25 %
    (rock), 180 layers to 54.87 m; every boundary of the model's nine layers is a boundary of the reference."""
    hs = np.full(80, 0.05); hr = np.full(80, 0.10)
    g = 0.10 * 1.25 ** np.arange(1, 40); g = g[np.cumsum(g) <= 48.0]
    return np.r_[hs, hr, g], 80


def fig_6_3():
    states = (("unfrozen", 0.5, 0.0), ("frozen", 0.0, 0.5))
    sa, cl = TEXTURES[1][1], TEXTURES[1][2]
    po = _poros(sa, cl)
    href, nsoil = _reference_layers()
    zb_ref = np.cumsum(href); zt_ref = zb_ref - href; zm_ref = zb_ref - href / 2
    h, bot, mid = _depths(); topd = bot - h
    res = {}
    for name, w, wi in states:
        lam = float(conductivity(sa, cl, po, np.array([w]), np.array([wi]))[0])
        cv = float(heat_capacity(po, np.array([w]), np.array([wi]))[0])
        a4 = annual_wave("4m", w, wi, po, lam); a12 = annual_wave("12m", w, wi, po, lam)
        ar = annual_wave("ref", w, wi, po, lam, h=href, n_soil=nsoil)
        # the reference averaged over each of the model's layers (heat-capacity weighted: uniform within a layer here)
        ar_lay = np.array([np.sum(href[(zt_ref >= t - 1e-9) & (zb_ref <= b + 1e-9)] * ar[(zt_ref >= t - 1e-9) & (zb_ref <= b + 1e-9)])
                           / (b - t) for t, b in zip(topd, bot)])
        d = np.sqrt(2.0 * lam / cv / OMEGA)
        res[name] = dict(a4=a4, a12=a12, ar=ar, ar_lay=ar_lay, lam=lam, cv=cv, d=d)
    norm = {k: 1.0 for k in res}                                       # surface amplitude A = 1 K
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 4.0), gridspec_kw={"wspace": 0.42, "width_ratios": [1.0, 1.15]})
    Y = lambda z: np.sqrt(z)
    ax = axs[0]
    r = res["unfrozen"]
    ax.plot(np.abs(r["ar"]) / norm["unfrozen"], Y(zm_ref), color="#8a8a86", lw=1.0,
            label=f"deep reference ({zb_ref[-1]:.0f} m)")
    ax.plot(np.abs(r["a4"]) / norm["unfrozen"], Y(mid[:6]), ls="none", marker="s", ms=5, mfc="white", mec=PAL[1], mew=1.3,
            label="4 m column (IBIS 2.6b4)")
    ax.plot(np.abs(r["a12"]) / norm["unfrozen"], Y(mid), ls="none", marker="o", ms=4.2, color=PAL[0],
            label="12 m column (release)")
    for zl in (4.0, 12.0):                                             # base of the 4 m and of the 12 m column
        ax.axhline(Y(zl), color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_xscale("log"); ax.set_xlim(0.008, 1.3)
    ticks = [0, 0.25, 1, 2, 4, 8, 12]
    ax.set_yticks(Y(np.array(ticks, float))); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.set_ylim(Y(14.5), 0)
    ax.set_ylabel("depth (m, square-root scale)"); ax.set_xlabel("amplitude / surface amplitude")
    _title(ax, "(a) annual amplitude, unfrozen loam")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")      # the curve runs from top right to bottom left
    _spines(ax)
    ax = axs[1]
    yy = np.arange(9)
    off = {"unfrozen": (-0.24, -0.08), "frozen": (0.08, 0.24)}
    for name in ("unfrozen", "frozen"):
        r = res[name]
        q4 = np.abs(r["a4"]) / np.abs(r["ar_lay"][:6]); q12 = np.abs(r["a12"]) / np.abs(r["ar_lay"])
        filled = name == "unfrozen"
        ax.plot(q4, yy[:6] + off[name][0], ls="none", marker="s", ms=5, mec=PAL[1], mew=1.3,
                mfc=PAL[1] if filled else "white")
        ax.plot(q12, yy + off[name][1], ls="none", marker="o", ms=5, mec=PAL[0], mew=1.3,
                mfc=PAL[0] if filled else "white")
        r["q4"], r["q12"] = q4, q12
    ax.axvline(1.0, color=INK2, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([f"{t:g}–{b:g} m" for t, b in zip(topd, bot)])
    ax.invert_yaxis(); ax.set_xlim(0.8, 1.45)
    ax.set_xlabel("amplitude / reference amplitude of the layer")
    _title(ax, "(b) each layer's amplitude / reference")
    hand = [Line2D([], [], ls="none", marker="s", ms=5, mec=PAL[1], mfc=PAL[1], mew=1.3, label="4 m column"),
            Line2D([], [], ls="none", marker="o", ms=5, mec=PAL[0], mfc=PAL[0], mew=1.3, label="12 m column"),
            Line2D([], [], ls="none", marker="o", ms=5, mec=INK2, mfc=INK2, label="unfrozen"),
            Line2D([], [], ls="none", marker="o", ms=5, mec=INK2, mfc="white", mew=1.3, label="frozen")]
    ax.legend(handles=hand, frameon=False, fontsize=6.8, loc="lower right", ncol=1)
    _spines(ax)
    fig.savefig(OUT / "fig_6_3_annual_wave.png")
    plt.close(fig)
    for name, r in res.items():
        print(f"6.3 {name}: lambda {r['lam']:.3f} W/m/K, C {r['cv']/1e6:.3f} MJ/m3/K, kappa {r['lam']/r['cv']:.3e} m2/s,"
              f" damping depth {r['d']:.2f} m")
        print("    4 m / ref per soil layer ", np.round(r["q4"], 3))
        print("    12 m / ref per layer     ", np.round(r["q12"], 3))
        print("    ref amplitude of the layers / surface", np.round(np.abs(r["ar_lay"]), 4))
    # for the text: the same comparison under a prescribed surface heat flux instead of a surface temperature
    for name, w, wi in states:
        lam = res[name]["lam"]
        ar = annual_wave("ref", w, wi, po, lam, h=href, n_soil=nsoil, forcing="flux")
        ar_lay = np.array([np.sum(href[(zt_ref >= t - 1e-9) & (zb_ref <= b + 1e-9)] * ar[(zt_ref >= t - 1e-9) & (zb_ref <= b + 1e-9)])
                           / (b - t) for t, b in zip(topd, bot)])
        q4 = np.abs(annual_wave("4m", w, wi, po, lam, forcing="flux")) / np.abs(ar_lay[:6])
        q12 = np.abs(annual_wave("12m", w, wi, po, lam, forcing="flux")) / np.abs(ar_lay)
        print(f"6.3 flux forcing, {name}: 4 m / ref {np.round(q4, 3)}; 12 m / ref {np.round(q12, 3)}")
    # for the text: the reference method against the analytic half-space solution. The reference layers with the soil
    # continued below 4 m instead of rock, driven by the same surface temperature, against the layer means of the
    # half-space wave A exp(-(1 + i) z / d), A = 1 K.
    print(f"6.3 reference column: {len(href)} layers, bottom {zb_ref[-1]:.2f} m")
    for name, w, wi in states:
        r = res[name]
        ar = annual_wave("ref", w, wi, po, r["lam"], h=href, n_soil=len(href))
        s = (1.0 + 1.0j) / r["d"]
        an = (np.exp(-s * zt_ref) - np.exp(-s * zb_ref)) / (s * href)
        err = np.abs(np.abs(ar) / np.abs(an) - 1.0) * 100.0
        print(f"6.3 all-soil reference vs half-space, {name}: max |error| of the layer amplitude "
              f"{err[zb_ref <= 4.0 + 1e-9].max():.4f} % to 4 m, {err[zb_ref <= 6.0 + 1e-9].max():.4f} % to 6 m")


# ---------------------------------------------------------------------------------------------------------------
# Figure 6.4: soil temperature of the release hindcast at two cells
# ---------------------------------------------------------------------------------------------------------------

CELLS = (("Madison, Wisconsin", 43.25, -89.25), ("Yakutsk, Siberia", 62.25, 129.75))


def fig_6_4():
    import netCDF4 as nc
    ij = [(int(round((89.75 - la) / 0.5)), int(round((lo + 179.75) / 0.5))) for _, la, lo in CELLS]
    T = np.zeros((len(CELLS), 10, 12, 6)); G = np.zeros((len(CELLS), 10, 12))
    for iy, y in enumerate(range(2001, 2011)):
        for m in range(12):
            d = nc.Dataset(HINDCAST % (y, m + 1))
            for c, (I, J) in enumerate(ij):
                T[c, iy, m] = np.asarray(d.variables["tsoi_mean"][I, J, :], float) - C.tmelt
                G[c, iy, m] = float(d.variables["tg_mean"][I, J]) - C.tmelt
            d.close()
    clim = T.mean(1); gclim = G.mean(1)
    h, bot, mid = _depths()
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw={"wspace": 0.25})
    mo = np.arange(1, 13)
    for c, ax in enumerate(axs):
        ax.axhline(0.0, color=INK2, lw=0.7)
        ax.plot(mo, gclim[c], color="#8a8a86", lw=1.0, ls=(0, (4, 2)), label="snow-free soil surface $T_g$")
        for k in range(6):
            ax.plot(mo, clim[c, :, k], color=PAL[k], lw=1.5, label=f"{mid[k]:g} m")
        ax.set_xlim(1, 12); ax.set_xticks(mo); ax.set_xticklabels(list("JFMAMJJASOND"))
        name, la, lo = CELLS[c]
        _title(ax, f"({'ab'[c]}) {name}, {abs(la):g}°{'N' if la > 0 else 'S'} {abs(lo):g}°{'E' if lo > 0 else 'W'}")
        _spines(ax)
    axs[0].set_ylabel("monthly mean temperature (°C)")
    hand, labs = axs[1].get_legend_handles_labels()
    labs = [l if l == "snow-free soil surface $T_g$" else f"node at {l}" for l in labs]
    fig.legend(hand, labs, loc="lower center", ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.10),
               handlelength=2.0, columnspacing=1.1)
    fig.savefig(OUT / "fig_6_4_release_cells.png", bbox_inches="tight")
    plt.close(fig)
    for c, (name, la, lo) in enumerate(CELLS):
        amp = (clim[c].max(0) - clim[c].min(0)) / 2
        lag = [int(np.argmax(clim[c][:, k])) + 1 for k in range(6)]
        print(f"6.4 {name}: annual mean by layer {np.round(clim[c].mean(0), 2)}; half-range {np.round(amp, 2)};"
              f" warmest month {lag}; ground mean {gclim[c].mean():.2f}; min/max of 3 m node {clim[c][:, 5].min():.2f}/{clim[c][:, 5].max():.2f}")


FIGURES = {"6.1": fig_6_1, "6.2": fig_6_2, "6.3": fig_6_3, "6.4": fig_6_4}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
        print("figure", key, "done", flush=True)
