"""Figures of Chapter 7 (soil hydrology) of the IBIS 3.0 technical note, drawn from the model's code and the release run.

Run from the repository root with the project interpreter:

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch07.py            # all figures
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch07.py 7.2 7.3    # some
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch07.py balance    # numbers of section 7.11
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch07.py states     # numbers of 7.3, 7.4, 7.7
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch07.py diagyear   # numbers of 7.11, 8.14

Every number a figure shows is computed here: the hydraulic functions from the texture table of the code
(`ibis.io.soil_init`) and checked against the model's own solver (`ibis.physics.soil_physics.soilh2o`); the column test
of Figure 7.3 runs the model's `soilh2o` and `wadjust`; Figure 7.4 reads the monthly output of the release run
(`final_rel30`, hindcast 2001-2010). The parameter values are those of the release parameter file. The option
`diagyear` reads the yearly output of the diagnostic year 2001 of the revised code with the water-balance diagnostic
(`verify_outputs/step2_wbal_2001*`). Figures 7.1-7.3 were first drawn with the release code and redrawn on 2026-11-02
with the code of branch review-technote (the soil water code they show did not change after the release; 7.1 got a new
drainage arrow, 7.3 new legend labels); Figure 7.4 is the release run's output.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse, Polygon
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: F401  (style, rcParams)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

PARAMS = ROOT / "python-port/configs/params/params_31s_ch4h.yaml"
HINDCAST = Path("/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30/hindcast/out")
warnings.filterwarnings("ignore", message="Explicitly requested dtype")   # float64 defaults truncated to float32

PAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
TEXTURE_NAMES = ("sand", "loamy sand", "sandy loam", "loam", "silt loam", "sandy clay loam", "clay loam",
                 "silty clay loam", "sandy clay", "silty clay", "clay")


def _soil_params():
    """wpudmax, zwpmax, bperm and the linearisation switch as the release builds them."""
    from ibis.io.physics_params import load_physics_params, build_sim_params
    import jax.numpy as jnp
    sp = build_sim_params(1, jnp.zeros((1,)), load_physics_params(str(PARAMS)))
    return float(sp.wpudmax), float(sp.zwpmax), float(sp.bperm), bool(sp.h2o_consistent_weights)


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------------------------------------
# Figure 7.1: the water fluxes of the soil column, and the content of one layer
# ---------------------------------------------------------------------------------------------------------------

def fig_7_1():
    from ibis.io.soil_init import HSOI_LAYERS
    wpudmax, zwpmax, bperm, _ = _soil_params()
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.set_xlim(0, 122); ax.set_ylim(0, 80); ax.axis("off")
    blue, brown = "#2a78d6", "#8a5a2b"
    arrow = dict(arrowstyle="-|>", mutation_scale=8, lw=1.0)
    x0, x1 = 17.0, 51.0                                               # the column
    ys = 54.0                                                         # soil surface
    hl = 6.2                                                          # drawn layer height (not to scale)
    bots = np.cumsum(HSOI_LAYERS)
    browns = ("#d9bf98", "#cfb289", "#c4a57b", "#b9986d", "#ad8b60", "#a17e54")
    for k in range(6):
        ax.add_patch(Rectangle((x0, ys - (k + 1) * hl), x1 - x0, hl, fc=browns[k], ec="white", lw=0.8))
        ax.text(x1 + 0.8, ys - (k + 1) * hl, f"{bots[k]:g} m", ha="left", va="center", fontsize=6.6, color=INK2)
        ax.text(x0 + 2.4, ys - (k + 0.5) * hl, f"layer {k + 1}", ha="left", va="center", fontsize=6.6, color=INK)
    ax.text(x1 + 0.8, ys, "0 m", ha="left", va="center", fontsize=6.6, color=INK2)
    yb = ys - 6 * hl
    ax.add_patch(Rectangle((x0, yb - 6.5), x1 - x0, 6.5, fc="#c9c9c6", ec="white", lw=0.8, hatch="////"))
    ax.text(x0 + 2.4, yb - 3.25, "bedrock: heat only (Ch. 6)", ha="left", va="center", fontsize=6.6, color=INK,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    # roots in the upper layers, right part of the column
    for xr_, depth in ((46.0, 4.6), (48.0, 3.4), (50.0, 4.1)):
        ax.plot([xr_, xr_ - 0.8, xr_ + 0.5, xr_ - 0.4], [ys, ys - depth * hl * 0.35, ys - depth * hl * 0.7,
                ys - depth * hl], color=brown, lw=0.8)
    # puddle
    xp = 27.0
    ax.add_patch(Ellipse((xp, ys + 0.2), 11.0, 2.4, fc="#9cc3ec", ec=blue, lw=0.6))
    ax.text(xp + 6.3, ys + 1.0, "puddle $P_l$, $P_i$", ha="left", va="bottom", fontsize=7.0, color=INK)
    ax.text(1.0, 77.5, "(a) the water fluxes of a column", fontsize=9, weight="bold", color=INK, va="top")
    # rain and melt
    ax.add_patch(FancyArrowPatch((25.0, 73.0), (25.0, ys + 1.5), color=blue, **arrow))
    ax.text(24.2, 72.5, "rain and snowmelt\nreaching the soil,\n$R$ (7.2)", ha="right", va="top", fontsize=6.8,
            color=INK, linespacing=1.2)
    # evaporation
    ax.add_patch(FancyArrowPatch((30.0, ys + 1.5), (30.0, 73.0), color="#b07a3b", **arrow))
    ax.text(30.8, 72.5, "ground evaporation $E_g$ from soil\nliquid, soil ice, puddle liquid\nand puddle ice (7.6)-(7.8)",
            ha="left", va="top", fontsize=6.8, color=INK, linespacing=1.2)
    # transpiration from roots
    ax.add_patch(FancyArrowPatch((48.0, ys + 0.3), (48.0, 63.0), color="#3d7a2c", **arrow))
    ax.text(48.8, 61.0, "root uptake $U_{u,k}+U_{l,k}$\n(7.10)-(7.12)", ha="left", va="center", fontsize=6.8,
            color=INK, linespacing=1.2)
    # surface runoff
    ax.add_patch(FancyArrowPatch((xp - 5.6, ys + 0.2), (2.0, ys + 0.2), color=blue, **arrow))
    ax.text(2.0, ys + 1.2, "surface runoff\n$Q_s$ (7.3)-(7.5)", ha="left", va="bottom", fontsize=6.8, color=INK,
            linespacing=1.2)
    # infiltration
    ax.add_patch(FancyArrowPatch((28.0, ys - 0.9), (28.0, ys - 4.6), color=blue, **arrow))
    ax.text(28.8, ys - 3.1, "infiltration $F_p$ (7.9)", ha="left", va="center", fontsize=6.4, color=INK)
    # interface flows
    for k in range(1, 6):
        yi = ys - k * hl
        ax.add_patch(FancyArrowPatch((31.0, yi + 1.6), (31.0, yi - 1.6), arrowstyle="<|-|>", mutation_scale=7,
                                     color=INK, lw=0.9))
    ax.text(32.0, ys - 2.5 * hl, "vertical flow\n$q_k$ (7.13)-(7.21)", ha="left", va="center", fontsize=6.6,
            color=INK, linespacing=1.2)
    # squeeze: inside the column, near its left edge, up below the surface, then across into the puddle (it is not
    # runoff: section 7.8)
    ysq = ys - 1.7
    sq = MPath([(x0 + 1.0, ys - 4.5 * hl), (x0 + 1.0, ysq), (xp - 3.5, ysq), (xp - 3.5, ys - 0.2)],
               [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=sq, color="#5a7fa8", lw=0.9, arrowstyle="-|>", mutation_scale=7,
                                 ls=(0, (3, 2))))
    ax.text(x0 - 0.8, ys - 3.0 * hl, "saturation\nexcess,\nback to the\npuddle (7.22)", ha="right", va="center",
            fontsize=6.6, color=INK, linespacing=1.2)
    # drainage: it leaves the model at the base of the soil column, so the arrow leaves the column at its side just
    # above the 4 m line and passes beside the bedrock, which carries heat but no water (sections 7.1, 7.13)
    yd, xd = yb + 1.6, x1 + 6.5                                       # just above 4 m; right of the depth labels
    dr = MPath([(45.0, yb + 3.6), (45.0, yd), (xd, yd), (xd, yb - 8.0)],
               [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=dr, color=blue, **arrow))
    ax.text(xd - 1.2, yb - 8.2, f"drainage $q_7$ (7.20) leaves the model\nat the base of the soil column, 4 m;\n"
            f"$b_{{perm}}$ = {bperm:g}", ha="right", va="top", fontsize=6.8, color=INK, linespacing=1.2)
    ax.text(1.0, 1.0, "layers not to scale", fontsize=6.5, color=INK2, style="italic")
    # (b) one layer: the pore space
    bx, bw = 84.0, 12.0
    ax.text(70.0, 77.5, "(b) the content of one layer", fontsize=9, weight="bold", color=INK, va="top")
    phi, wi, w = 0.45, 0.30, 0.60                                    # illustrative values
    H = 50.0; ytop = 70.0
    parts = [("solids", 1 - phi, "#c4a57b"), ("ice", phi * wi, "#d5ecf7"), ("liquid", phi * (1 - wi) * w, "#6fa6e0"),
             ("air", phi * (1 - wi) * (1 - w), "#f4f4f2")]
    labels = {"solids": r"$1-\phi$", "ice": r"$\phi\,w^{i}$", "liquid": r"$\phi\,(1-w^{i})\,w$",
              "air": r"$\phi\,(1-w^{i})(1-w)$"}
    y = ytop
    for name, frac, col in parts:
        h = H * frac
        ax.add_patch(Rectangle((bx, y - h), bw, h, fc=col, ec=INK2, lw=0.5))
        ax.text(bx + bw + 1.5, y - h / 2, f"{name}: {labels[name]}", ha="left", va="center", fontsize=7.2, color=INK)
        y -= h
    yp0, yp1 = ytop - H * (1 - phi), ytop - H                        # bracket of the pore space
    ax.plot([bx - 2.0, bx - 2.0], [yp0, yp1], color=INK2, lw=0.7)
    ax.plot([bx - 2.0, bx - 1.0], [yp0, yp0], color=INK2, lw=0.7); ax.plot([bx - 2.0, bx - 1.0], [yp1, yp1], color=INK2, lw=0.7)
    ax.text(bx - 2.8, (yp0 + yp1) / 2, r"pores $\phi$", ha="right", va="center", fontsize=7, color=INK)
    ax.text(70.0, 14.0,
            "Volume fractions of a layer. $w$ (wsoi): liquid as a fraction\nof the pore space not filled by ice; $w^{i}$ (wisoi):"
            " ice as a\nfraction of the pore space. Drawn for $\\phi$ = 0.45,\n$w^{i}$ = 0.3, $w$ = 0.6.",
            ha="left", va="top", fontsize=7, color=INK2, linespacing=1.3)
    fig.savefig(OUT / "fig_7_1_column_fluxes.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 7.2: hydraulic conductivity and diffusivity of the texture classes, as the solver forms them
# ---------------------------------------------------------------------------------------------------------------

SHOW = (0, 2, 3, 4, 6, 10)                                           # sand, sandy loam, loam, silt loam, clay loam, clay


def fig_7_2():
    import jax.numpy as jnp
    from ibis.io.soil_init import _BEX, _SUCTION, _HYDRAUL, _POROS, HSOI_LAYERS
    from ibis.physics.soil_physics import soilh2o
    wpudmax, zwpmax, bperm, cw = _soil_params()
    w = np.linspace(0.02, 1.0, 491)
    nint = lambda x: np.floor(x + 0.5)                                # F77 nint, as the solver rounds the exponents
    DMIN = 1.0e-9
    K = {c: _HYDRAUL[c] * w ** (2 * nint(_BEX[c]) + 3) for c in SHOW}
    D = {c: np.maximum(_HYDRAUL[c] * _SUCTION[c] * _BEX[c] * w ** (nint(_BEX[c]) + 2), DMIN) for c in SHOW}
    # check against the model's solver (its own coefficients, no formula retyped): a uniform column drains
    # bperm * K(w) at its base; fed rho_w K(w) at the top and with a fully permeable base (bperm = 1) it is
    # stationary, and the flow across every interior interface is rho_w K(w)
    hs = jnp.asarray(HSOI_LAYERS)
    for c in SHOW:
        ww = jnp.asarray(np.repeat(w[::49][:, None], 6, axis=1))
        n = ww.shape[0]
        z = jnp.zeros((n, 6)); one = jnp.ones((n, 6))
        kk = _HYDRAUL[c] * w[::49] ** (2 * nint(_BEX[c]) + 3)
        args = (z, one * _POROS[c], one * _POROS[c], one * _HYDRAUL[c], one * _SUCTION[c], one * _BEX[c], hs)
        r = soilh2o(ww, *args, jnp.zeros(n), jnp.zeros(n), jnp.zeros(n), bperm, 3600.0, 1000.0, 1e-7,
                    consistent_weights=cw)
        assert np.allclose(np.asarray(r.wflo[:, -1]), 1000.0 * bperm * kk, rtol=1e-4, atol=1e-12)
        feed = jnp.asarray(1000.0 * kk)
        r = soilh2o(ww, *args, jnp.zeros(n), feed, feed * 3600.0 + 1.0, 1.0, 3600.0, 1000.0, 1e-7,
                    consistent_weights=cw)
        assert np.allclose(np.asarray(r.wflo[:, 1:]), 1000.0 * kk[:, None], rtol=1e-3, atol=1e-7), \
            (TEXTURE_NAMES[c], np.asarray(r.wflo[:, 1:]), 1000.0 * kk)
    # check of the diffusivity: non-uniform columns, one step of the solver (release weights); the interior flows it
    # returns must equal the linearised flux (7.18) formed here with the K and D of (7.16) at the solver's end-of-step
    # water. The D terms are checked to be much larger than the tolerance, so a wrong D would fail.
    dmax = 0.0
    for c in SHOW:
        prof = np.array([[0.95, 0.85, 0.75, 0.70, 0.65, 0.60], [0.55, 0.62, 0.70, 0.78, 0.86, 0.93],
                         [0.30, 0.50, 0.20, 0.90, 0.40, 0.80]])
        n = prof.shape[0]
        one = jnp.ones((n, 6))
        r = soilh2o(jnp.asarray(prof), jnp.zeros((n, 6)), one * _POROS[c], one * _POROS[c], one * _HYDRAUL[c],
                    one * _SUCTION[c], one * _BEX[c], hs, jnp.zeros(n), jnp.zeros(n), jnp.zeros(n), bperm, 3600.0,
                    1000.0, 1e-7, consistent_weights=True)
        wp = np.asarray(r.wsoi, float)
        m, nn = 2 * nint(_BEX[c]) + 3, nint(_BEX[c]) + 2
        Ks, B = _HYDRAUL[c], _HYDRAUL[c] * _SUCTION[c] * _BEX[c]
        hsn = np.asarray(HSOI_LAYERS, float)
        q, qd = np.zeros((n, 5)), np.zeros((n, 5))
        for k in range(1, 6):
            lam, dz = hsn[k] / (hsn[k - 1] + hsn[k]), 0.5 * (hsn[k - 1] + hsn[k])
            wh = np.minimum(lam * prof[:, k - 1] + (1 - lam) * prof[:, k], 1.0)     # (7.19), release
            wb = lam * wp[:, k - 1] + (1 - lam) * wp[:, k]                            # (7.17)
            Dk = np.maximum(B * wh ** nn, DMIN)
            Dp = np.where(B * wh ** nn < DMIN, 0.0, nn * B * wh ** (nn - 1))
            G = (prof[:, k] - prof[:, k - 1]) / dz
            qd[:, k - 1] = 1000.0 * (-Dp * G * (wb - wh) - Dk * (wp[:, k] - wp[:, k - 1]) / dz)
            q[:, k - 1] = 1000.0 * (Ks * wh ** m + m * Ks * wh ** (m - 1) * (wb - wh)) + qd[:, k - 1]
        assert np.allclose(np.asarray(r.wflo[:, 1:6]), q, rtol=2e-2, atol=5e-8), (TEXTURE_NAMES[c], np.asarray(r.wflo), q)
        dmax = max(dmax, float(np.abs(qd).max()))
        assert np.abs(qd).max() > 100 * 5e-8, TEXTURE_NAMES[c]
    print(f"7.2 checks passed: K at the base and the interior; (7.16)-(7.18) with D, largest D term {dmax:.2e} kg m-2 s-1")
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"wspace": 0.32})
    for i, c in enumerate(SHOW):
        lab = f"{TEXTURE_NAMES[c]} ($b$ = {_BEX[c]:g})"
        ls = (0, (4, 2)) if TEXTURE_NAMES[c] == "silt loam" else "-"     # loam and silt loam nearly coincide in (b)
        axs[0].semilogy(w, K[c], color=PAL[i], lw=1.4, ls=ls, label=lab)
        axs[1].semilogy(w, D[c], color=PAL[i], lw=1.4, ls=ls, label=lab)
    axs[0].set_ylim(1e-16, 1e-3)
    axs[0].set_ylabel("hydraulic conductivity $K$ (m s$^{-1}$)")
    axs[0].set_title("(a) conductivity, eqs. (7.13), (7.16)", loc="left", fontsize=9, weight="bold", color=INK)
    axs[1].axhline(DMIN, color=INK2, lw=0.8, ls="--")
    axs[1].text(0.03, DMIN * 0.62, "minimum $D_{min}$", fontsize=7, color=INK2, va="center")
    axs[1].set_ylim(2.5e-10, 3e-4)
    axs[1].set_ylabel("diffusivity $D$ (m$^{2}$ s$^{-1}$)")
    axs[1].set_title("(b) diffusivity, eqs. (7.13), (7.16)", loc="left", fontsize=9, weight="bold", color=INK)
    axs[1].legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=1.6)
    for a in axs:
        a.set_xlim(0, 1)
        a.set_xlabel("liquid content $w$")
        _clean(a)
    fig.savefig(OUT / "fig_7_2_hydraulic_functions.png")
    plt.close(fig)
    for c in SHOW:
        print(f"7.2 {TEXTURE_NAMES[c]}: K exponent {int(2 * nint(_BEX[c]) + 3)}, D exponent {int(nint(_BEX[c]) + 2)};"
              f" base drainage at w = 1: {bperm * _HYDRAUL[c] * 1000 * 86400 * 365:.0f} mm/yr")


# ---------------------------------------------------------------------------------------------------------------
# Figure 7.3: the linearisation point of the flow solver, a column with a dry layer over wet layers
# ---------------------------------------------------------------------------------------------------------------

def column_test(nsteps=480, cls=2, w0=(0.12, 0.10, 0.06, 0.0, 0.71, 0.86)):
    """Hourly steps of the model's soilh2o and wadjust on one column with no inputs, with the IBIS 2.6b4 form of the
    linearisation point (7.19) and with the release form, whose weights are those of the matrix
    (soil.h2o_consistent_weights). Single precision, as the global runs."""
    import jax
    import jax.numpy as jnp
    from ibis.io.soil_init import _BEX, _SUCTION, _HYDRAUL, _POROS, HSOI_LAYERS
    from ibis.physics.soil_physics import soilh2o, wadjust
    _, _, bperm, _ = _soil_params()
    f32 = jnp.float32
    hs = jnp.asarray(HSOI_LAYERS, f32)
    one = jnp.ones((1, 6), f32)
    poros, hyd, suc, bex = one * _POROS[cls], one * _HYDRAUL[cls], one * _SUCTION[cls], one * _BEX[cls]
    wi = jnp.zeros((1, 6), f32)
    pflo = jnp.maximum(poros * jnp.maximum(1.0 - wi, 1e-7), 1e-7)
    zero = jnp.zeros((1,), f32)
    out = {}
    for cw in (False, True):
        @jax.jit
        def step(w):
            h = soilh2o(w, wi, poros, pflo, hyd, suc, bex, hs, zero, zero, zero, bperm, 3600.0, 1000.0, 1e-7,
                        consistent_weights=cw)
            w2, _, gadj = wadjust(h.wsoi, wi, poros, hs, 1000.0, 3600.0, 1e-7)
            return w2, h.wflo[0, -1] * 3600.0, gadj[0] * 3600.0
        w = jnp.asarray([w0], f32)
        store0 = float(jnp.sum(hs * poros[0] * 1000.0 * w[0]))
        drain, made, store = [0.0], [0.0], [0.0]
        for _ in range(nsteps):
            w, d, g = step(w)
            drain.append(drain[-1] + float(d)); made.append(made[-1] + float(g))
            store.append(float(jnp.sum(hs * poros[0] * 1000.0 * w[0])) - store0)
        out[cw] = dict(drain=np.array(drain), made=np.array(made), store=np.array(store), w=np.asarray(w[0]))
    return out, bperm


def fig_7_3():
    from ibis.io.soil_init import HSOI_LAYERS
    w0 = (0.12, 0.10, 0.06, 0.0, 0.71, 0.86)
    out, bperm = column_test(w0=w0)
    t = np.arange(len(out[False]["drain"])) / 24.0
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.0), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 0.85]})
    lab = {False: "IBIS 2.6b4 form (7.19)", True: "release form (7.19)"}           # the names of section 7.7.2
    for i, cw in enumerate((False, True)):
        axs[0].plot(t, out[cw]["drain"], color=PAL[i], lw=1.4, label=lab[cw])
        axs[1].plot(t, out[cw]["made"], color=PAL[i], lw=1.4, label=lab[cw])
    axs[0].set_ylabel("cumulative drainage (mm)")
    axs[0].set_title("(a) drainage at the base", loc="left", fontsize=8.5, weight="bold", color=INK)
    axs[1].set_ylabel("cumulative water added (mm)")
    axs[1].set_title("(b) water added by the bounds", loc="left", fontsize=8.5, weight="bold", color=INK)
    axs[0].legend(frameon=False, fontsize=6.8, loc="upper left")
    for a in axs[:2]:
        a.set_xlabel("days"); a.set_xlim(0, t[-1]); _clean(a)
    ax = axs[2]
    bot = np.cumsum(HSOI_LAYERS); mid = bot - HSOI_LAYERS / 2
    ax.plot(w0, mid, color=INK2, lw=1.0, ls="--", marker="o", ms=3, label="start")
    for i, cw in enumerate((False, True)):
        ax.plot(out[cw]["w"], mid, color=PAL[i], lw=1.4, marker="o", ms=3, label=f"day {t[-1]:g}, {lab[cw].split(' (')[0]}")
    ax.set_ylim(4.0, 0); ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("$w$ at the layer midpoint"); ax.set_ylabel("depth (m)")
    ax.set_title("(c) profile", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left")
    _clean(ax)
    fig.savefig(OUT / "fig_7_3_linearisation_column.png")
    plt.close(fig)
    for cw in (False, True):
        o = out[cw]; f = 8760.0 / (len(o["drain"]) - 1)
        print(f"7.3 {'release form' if cw else 'IBIS 2.6b4 form'}: drainage {o['drain'][-1]:.2f} mm ({o['drain'][-1] * f:.0f} mm/yr),"
              f" added by bounds {o['made'][-1]:.2f} mm ({o['made'][-1] * f:.0f} mm/yr), store change {o['store'][-1]:.2f} mm,"
              f" closure {o['store'][-1] + o['drain'][-1] - o['made'][-1]:.3f} mm; final w {np.round(o['w'], 3)};"
              f" steps in which the bounds added water: {int((np.diff(o['made']) > 0).sum())} of {len(o['made']) - 1}")


# ---------------------------------------------------------------------------------------------------------------
# Figure 7.4: one cell of the release run, monthly water budget and soil water by layer
# ---------------------------------------------------------------------------------------------------------------

def fig_7_4(years=(2003, 2004, 2005, 2006)):
    import xarray as xr
    I, J = 93, 181                                                   # 43.25 N, 89.25 W (as Figure 3.1)
    rows = []
    for y in years:
        for m in range(1, 13):
            ds = xr.open_dataset(HINDCAST / f"ibis_monthly_{y}_{m:02d}.nc")
            pt = ds.isel(lat=I, lon=J)
            rows.append(dict(t=y + (m - 0.5) / 12.0, P=float(pt.amprcp), E=float(pt.amaet), R=float(pt.amtrunof),
                             D=float(pt.amtdrain), w=np.asarray(pt.wsoi_mean, float), wi=np.asarray(pt.wisoi_mean, float),
                             lat=float(ds.lat[I]), lon=float(ds.lon[J])))
            ds.close()
    t = np.array([r["t"] for r in rows])
    fig, axs = plt.subplots(3, 1, figsize=(7.4, 7.0), sharex=True, gridspec_kw={"hspace": 0.32})
    ax = axs[0]
    for i, (k, lab) in enumerate((("P", "precipitation"), ("E", "evapotranspiration"), ("R", "surface runoff"),
                                  ("D", "drainage"))):
        ax.plot(t, [r[k] for r in rows], color=PAL[i], lw=1.3, marker="o", ms=2.2, label=lab)
    ax.set_ylabel("mm month$^{-1}$")
    ax.set_title("(a) monthly water budget", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")
    ax.set_ylim(0, None)
    from ibis.io.soil_init import HSOI_LAYERS
    bot = np.cumsum(HSOI_LAYERS); top = bot - HSOI_LAYERS
    # layers in a sequential ramp, light (top) to dark (base), so that they are not read as the fluxes of panel (a)
    ramp = plt.get_cmap("YlGnBu")(np.linspace(0.38, 1.0, 6))
    for k in range(6):
        axs[1].plot(t, [r["w"][k] for r in rows], color=ramp[k], lw=1.3, label=f"{top[k]:g}–{bot[k]:g} m")
        axs[2].plot(t, [r["wi"][k] for r in rows], color=ramp[k], lw=1.3, label=f"{top[k]:g}–{bot[k]:g} m")
    axs[1].set_ylabel("$w$ (liquid)")
    axs[1].set_title("(b) liquid water by layer, month mean", loc="left", fontsize=8.5, weight="bold", color=INK)
    axs[1].set_ylim(0, 1.0)
    axs[2].set_ylabel("$w^{i}$ (ice)")
    axs[2].set_title("(c) ice by layer, month mean", loc="left", fontsize=8.5, weight="bold", color=INK)
    axs[2].set_ylim(0, None)
    axs[2].legend(frameon=False, fontsize=6.8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.32), columnspacing=1.2,
                  handlelength=1.6, title="soil layer (panels b and c)", title_fontsize=7)
    for a in axs:
        _clean(a)
    axs[2].set_xlim(years[0], years[-1] + 1)
    axs[2].set_xticks(range(years[0], years[-1] + 2))
    axs[2].set_xlabel("year")
    fig.savefig(OUT / "fig_7_4_madison_monthly.png")
    plt.close(fig)
    P = sum(r["P"] for r in rows); E = sum(r["E"] for r in rows); R = sum(r["R"] for r in rows); D = sum(r["D"] for r in rows)
    ny = len(years)
    print(f"7.4 cell {rows[0]['lat']} {rows[0]['lon']}, {years[0]}-{years[-1]} means (mm/yr): P {P / ny:.0f} E {E / ny:.0f}"
          f" R {R / ny:.0f} D {D / ny:.0f}; P-E-R-D {(P - E - R - D) / ny:.1f}; max ice by layer "
          f"{np.round(np.max([r['wi'] for r in rows], axis=0), 2)}; min w by layer {np.round(np.min([r['w'] for r in rows], axis=0), 2)}")
    frozen = np.array([r["wi"][0] > 0.1 for r in rows]); Rm = np.array([r["R"] for r in rows])
    print(f"7.4 runoff in months with top-layer ice > 0.1: {Rm[frozen].sum() / max(Rm.sum(), 1e-9):.3f} of the total"
          f" ({int(frozen.sum())} of {len(rows)} months); largest monthly runoff {Rm.max():.0f} mm in {years[0] + np.argmax(Rm) // 12}-{np.argmax(Rm) % 12 + 1:02d};"
          f" largest monthly drainage {max(r['D'] for r in rows):.0f} mm")
    Dm = np.array([r["D"] for r in rows]); Pm = np.array([r["P"] for r in rows])
    W = np.array([r["w"] for r in rows]); WI = np.array([r["wi"] for r in rows])
    ym = lambda i: f"{years[0] + i // 12}-{i % 12 + 1:02d}"
    print(f"7.4 largest drainage in {ym(int(np.argmax(Dm)))}, largest precipitation {Pm.max():.0f} mm in {ym(int(np.argmax(Pm)))};"
          f" max w by layer {np.round(W.max(0), 2)}; months with layer-4 ice > 0.01: {[ym(i) for i in np.where(WI[:, 3] > 0.01)[0]]};"
          f" months with layer-4 ice > 0: {[(ym(i), float(f'{WI[i, 3]:.2g}')) for i in np.where(WI[:, 3] > 0)[0]]};"
          f" months with top-layer w < 0.1: {[ym(i) for i in np.where(W[:, 0] < 0.1)[0]]}, their top-layer ice"
          f" {np.round(WI[W[:, 0] < 0.1, 0], 2)}")


# ---------------------------------------------------------------------------------------------------------------
# Numbers of section 7.11 (no figure): the water balance of the release hindcast, 2001-2010
# ---------------------------------------------------------------------------------------------------------------

def balance():
    """Storage change from the restarts of 2000 and 2010 (canopy water, puddle, soil liquid and ice, snow, as F77
    initial.f wtot) against the written-out precipitation, evapotranspiration, surface runoff and drainage. Slow: it
    loads two global restarts (run it alone: make_ch07.py balance)."""
    import xarray as xr
    from ibis.io.restart import load_state
    from ibis.state.model_state import create_ibis_state
    from ibis.dimensions import GridConfig
    sys.path.insert(0, str(ROOT))
    from scripts.run_global import _load_land_mask_from_file, _subset_land_mask
    run = HINDCAST.parent
    mask, glat, glon = _load_land_mask_from_file("/shared/nas2/Min/HWSD/surta.nc")
    _, _, _, _, gli, glo = _subset_land_mask(mask, glat, glon, 90.0, -90.0, -180.0, 180.0)
    lat = np.asarray(glat)[gli]; lon = np.asarray(glon)[glo]; npoi = len(gli)
    tmpl = create_ibis_state(GridConfig(nlon=720, nlat=360, npoi=npoi, xres=0.5, yres=0.5, nbedrock=3),
                             with_stress=True, with_etparts=True, with_seasonal=True)

    def storage(year):
        st, _ = load_state(str(run / f"restart/ibis_restart_{year}.npz"), tmpl)
        s, sn, c, v = st.soil, st.snow, st.canopy, st.vegetation
        A = lambda x: np.asarray(x, np.float64)
        hs = A(s.hsoi)[:6]; lai, sai, fi = A(v.lai), A(v.sai), A(sn.fi)
        soil = (1000.0 * A(s.poros) * hs[None, :] * ((1 - A(s.wisoi)) * A(s.wsoi) + A(s.wisoi))).sum(1)
        can = ((A(c.wliqu) + A(c.wsnou)) * A(v.fu) * 2 * lai[:, 1] + (A(c.wliqs) + A(c.wsnos)) * A(v.fu) * 2 * sai[:, 1]
               + (A(c.wliql) + A(c.wsnol)) * A(v.fl) * 2 * (lai[:, 0] + sai[:, 0]) * (1 - fi))
        snow = fi * float(sn.rhos) * A(sn.hsno).sum(1)
        return soil + A(s.wpud) + A(s.wipud) + snow + can, snow

    (W0, snow0), (W1, snow1) = storage(2000), storage(2010)
    dW = (W1 - W0) / 10.0
    F = {k: np.zeros(npoi) for k in ("P", "E", "R", "D", "T", "I", "S")}
    for y in range(2001, 2011):
        with xr.open_dataset(HINDCAST / f"ibis_yearly_{y}.nc") as ds:
            for k, v in (("P", "ayprcp"), ("E", "ayaet"), ("R", "aytrunof"), ("D", "aytdrain"), ("T", "aytrans"),
                         ("I", "ayinterc"), ("S", "aysoilev")):
                F[k] += np.asarray(ds[v].values, np.float64)[gli, glo] / 10.0
    res = dW - (F["P"] - F["E"] - F["R"] - F["D"])
    area = np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25))
    for name, m in (("all land", np.ones(npoi, bool)), ("60S-60N", (lat > -60) & (lat < 60))):
        wm = lambda x: float(np.sum(x[m] * area[m]) / np.sum(area[m]))
        print(f"balance {name}: P {wm(F['P']):.1f} E {wm(F['E']):.1f} R {wm(F['R']):.1f} D {wm(F['D']):.1f} mm/yr,"
              f" D/(R+D) {wm(F['D']) / (wm(F['R']) + wm(F['D'])):.3f}; dW {wm(dW):.2f}; residual {wm(res):.2f} mm/yr;"
              f" cells {int(m.sum())}, |res| <= 1 mm/yr: {int((np.abs(res[m]) <= 1).sum())}")
    band = (lat > -60) & (lat < 60); hy = band & (res > 5)
    print(f"balance 60S-60N res > 5: {int(hy.sum())} cells, lat {lat[hy].min()}..{lat[hy].max()}, lon {lon[hy].min()}.."
          f"{lon[hy].max()}, P < 2 mm/yr in {int((F['P'][hy] < 2).sum())}, median E {np.median(F['E'][hy]):.1f},"
          f" band max {res[band].max():.1f} min {res[band].min():.1f}")
    share = res[hy] / np.maximum(F["S"][hy], 1e-9)
    print(f"balance 60S-60N res > 5: median transpiration {np.median(F['T'][hy]):.3f}, interception"
          f" {np.median(F['I'][hy]):.3f}, ground evaporation {np.median(F['S'][hy]):.2f} mm/yr; residual / ground"
          f" evaporation median {np.median(share):.3f}, range {share.min():.3f}-{share.max():.3f}")
    big = np.abs(res) > 10
    print(f"balance |res| > 10: Antarctica {int((big & (lat < -60)).sum())}, north of 60N {int((big & (lat >= 60)).sum())}"
          f" (75W-10W: {int((big & (lat >= 60) & (lon > -75) & (lon < -10)).sum())}), max |res| {np.abs(res).max():.1f};"
          f" of these {int(big.sum())} cells, with snow at the end of 2000 and of 2010:"
          f" {int((big & (snow0 > 0) & (snow1 > 0)).sum())}; positive {int((res > 10).sum())}, negative {int((res < -10).sum())}")


# ---------------------------------------------------------------------------------------------------------------
# Numbers of sections 7.3, 7.4 and 7.7 (no figure): puddle ice, dry top layers under a puddle, texture changes
# ---------------------------------------------------------------------------------------------------------------

def states():
    """From the restarts of the release hindcast at the end of 2000 and of 2010: cells whose puddle exceeds wpudmax
    (the overflow removes only liquid), and cells in the evaporation-share case of `setsoi` with a top layer without
    water (shares sum to a_p, or all to the puddle). From the soil file: cells with a texture change between two
    adjacent layers (classes of the current code). Slow: it loads two global restarts."""
    import netCDF4 as nc
    from ibis.io.restart import load_state
    from ibis.state.model_state import create_ibis_state
    from ibis.dimensions import GridConfig
    from ibis.io.soil_init import texture_class
    sys.path.insert(0, str(ROOT))
    from scripts.run_global import _load_land_mask_from_file, _subset_land_mask
    wpudmax, zwpmax, _, _ = _soil_params()
    eps = 1.0e-7
    run = HINDCAST.parent
    mask, glat, glon = _load_land_mask_from_file("/shared/nas2/Min/HWSD/surta.nc")
    _, _, _, _, gli, glo = _subset_land_mask(mask, glat, glon, 90.0, -90.0, -180.0, 180.0)
    lat = np.asarray(glat)[gli]; lon = np.asarray(glon)[glo]; npoi = len(gli)
    tmpl = create_ibis_state(GridConfig(nlon=720, nlat=360, npoi=npoi, xres=0.5, yres=0.5, nbedrock=3),
                             with_stress=True, with_etparts=True, with_seasonal=True)
    from ibis.constants import PhysicalConstants
    cice = float(PhysicalConstants().cice)
    ice = {}
    for year in (2000, 2010):
        st, _ = load_state(str(run / f"restart/ibis_restart_{year}.npz"), tmpl)
        s = st.soil
        wp, wip = np.asarray(s.wpud, np.float64), np.asarray(s.wipud, np.float64)
        ice[year] = wip
        if year == 2010:                      # the heat capacity the puddle ice adds to the top layer (c0pud)
            from ibis.timestepping.loops import _solid_phase
            from ibis.io.physics_params import load_physics_params, build_sim_params
            import jax.numpy as jnp
            sp = build_sim_params(1, jnp.zeros((1,)), load_physics_params(str(PARAMS)))
            cs, rho = (np.asarray(x, np.float64)[:, 0] for x in _solid_phase(s, sp))
            ph, w1, wi1 = (np.asarray(x, np.float64)[:, 0] for x in (s.poros, s.wsoi, s.wisoi))
            csoil = ((1 - ph) * cs * rho + ph * (1 - wi1) * w1 * float(PhysicalConstants().ch2o) * 1000.0
                     + ph * wi1 * cice * 1000.0) * float(np.asarray(s.hsoi)[0])
            m = wip > 100.0
            g = (ice[2010] - ice[2000])[m] / 10.0
            print(f"states 2000-2010: of the {int(m.sum())} cells with puddle ice > 100 kg/m2 at the end of 2010, the ice"
                  f" grew in {int((g > 0).sum())}: median {np.median(g):.1f}, 90th percentile {np.percentile(g, 90):.1f},"
                  f" max {g.max():.1f} kg/m2/yr; its heat capacity cice*wipud: median {np.median(cice * wip[m]):.3g},"
                  f" max {(cice * wip[m]).max():.3g} J/m2/K, against that of the top soil layer (median"
                  f" {np.median(csoil[m]):.3g} J/m2/K); ratio median {np.median(cice * wip[m] / csoil[m]):.0f},"
                  f" max {np.max(cice * wip[m] / csoil[m]):.0f}")
        big = wp + wip > wpudmax
        print(f"states {year}: puddle > {wpudmax:g} kg/m2 in {int(big.sum())} cells (south of 60S"
              f" {int((big & (lat < -60)).sum())}, north of 60N {int((big & (lat > 60)).sum())}); puddle ice > 100 kg/m2"
              f" in {int((wip > 100).sum())}; max liquid {wp.max():.2f}, max ice {wip.max():.1f} kg/m2")
        w1, wi1 = np.asarray(s.wsoi, np.float64)[:, 0], np.asarray(s.wisoi, np.float64)[:, 0]
        dry = (1.0 - wi1) * w1 + wi1 < eps
        ap = np.clip(zwpmax * (wp + wip) / wpudmax, 0.0, zwpmax)
        c3 = dry & (ap >= eps)
        print(f"states {year}: top layer without water in {int(dry.sum())} cells, {int(c3.sum())} with a puddle"
              f" (shares sum to a_p) at lat {np.unique(lat[c3])}, lon {lon[c3].min() if c3.any() else None}.."
              f"{lon[c3].max() if c3.any() else None}, puddle {np.round(np.sort((wp + wip)[c3]), 2)} kg/m2")
    ds = nc.Dataset("/shared/nas2/Min/HWSD/soita.nc")
    sand = np.array(ds.variables["sandpct"][:], np.float64)[:, gli, glo].T
    clay = np.array(ds.variables["claypct"][:], np.float64)[:, gli, glo].T
    ds.close()
    cls = texture_class(sand, clay)
    chg = cls[:, 1:] != cls[:, :-1]
    print(f"states: texture change between adjacent layers in {int(chg.any(1).sum())} of {npoi} cells"
          f" ({100 * chg.any(1).mean():.1f} %); by interface {chg.sum(0)}")


# ---------------------------------------------------------------------------------------------------------------
# Numbers of sections 7.11 and 8.14 (no figure): the diagnostic year 2001 of the revised code
# ---------------------------------------------------------------------------------------------------------------

DIAG = Path("/shared/nas2/Hangkai/ibis/verify_outputs")
DIAG_RUNS = (("switch off", "step2_wbal_2001"), ("switch on", "step2_wbal_2001_d25"),
             ("switch on, dropped rain booked (current code)", "step2_wbal_2001_d25b"))


def diagyear():
    """The water-balance diagnostic (physics.water_balance) in one global year, 2001, of the revised code with
    params_32a_base.yaml, started from the release hindcast's state at the end of 2000: soil.evap_needs_store off
    (step2_wbal_2001), on (step2_wbal_2001_d25), and on with the rain and melt below epsilon booked (step2_wbal_2001_d25b,
    commit 75d8935, the current code). Prints the closure of (7.26) over land, the safeguard terms in the hyperarid
    region of the release balance, the event counters, and (current code) the cells with large residuals and the
    residuals of the deep snowpacks by thickness at the start of the year (release restart of 2000; technote 8.14).
    Slow: it loads a global restart."""
    import netCDF4 as nc
    land = _land_mask()
    ii, jj = np.nonzero(land)
    lat = 89.75 - 0.5 * ii; lon = -179.75 + 0.5 * jj
    area = 6.371e6 ** 2 * np.deg2rad(0.5) * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))   # m2
    box = (lat >= 16.25) & (lat <= 27.75) & (lon >= 28.75) & (lon <= 42.25)
    out = {}
    for name, run in DIAG_RUNS:
        ds = nc.Dataset(DIAG / run / "out/ibis_yearly_2001.nc")
        g = lambda k: np.ma.filled(ds[k][:].astype(np.float64), np.nan)[ii, jj]
        P, E, R, D, dW, A = (g(k) for k in ("ayprcp", "ayaet", "aytrunof", "aytdrain", "aydwtot", "ayadjust"))
        res = P - E - R - D - dW + A
        wm = lambda x: float(np.sum(x * area) / np.sum(area))
        print(f"diagyear {run} ({name}): closure P - E - R - D - dW + A over {len(res)} land cells: median {np.median(res):+.4f},"
              f" p01 {np.percentile(res, 1):+.3f}, p99 {np.percentile(res, 99):+.3f}, area mean {wm(res):+.4f},"
              f" max |r| {np.abs(res).max():.3f} mm/yr; land precipitation {wm(P):.1f} mm/yr")
        print(f"diagyear {run}: hyperarid region ({int(box.sum())} cells): ayadjust mean {A[box].mean():.4f} max {A[box].max():.3f};"
              f" wadjust mean {g('ay_adj_wadjust')[box].mean():.4f} max {g('ay_adj_wadjust')[box].max():.3f};"
              f" puddle floor mean {g('ay_adj_puddle')[box].mean():.4f} max {g('ay_adj_puddle')[box].max():.3f};"
              f" ET mean {E[box].mean():.4f} mm/yr")
        print(f"diagyear {run}: cells where the evaporation limit acted {int((g('n_bound') > 0).sum())}, with excess ground"
              f" snowfall {int((g('n_snow') > 0).sum())}")
        out[run] = res
        ds.close()
    res = out[DIAG_RUNS[-1][1]]
    from ibis.dimensions import GridConfig
    from ibis.state.model_state import create_ibis_state
    from ibis.io.restart import load_state
    tmpl = create_ibis_state(GridConfig(npoi=len(ii), nbedrock=3), 296e-6, 0.209,
                             with_stress=True, with_etparts=True, with_seasonal=True)
    st, _ = load_state(str(HINDCAST.parent / "restart/ibis_restart_2000.npz"), tmpl)
    fi0 = np.asarray(st.snow.fi, np.float64); H0 = np.asarray(st.snow.hsno, np.float64).sum(1)
    big = np.abs(res) > 1.0
    north = big & (lat > 59) & (lon > -75) & (lon < -10)
    print(f"diagyear current code: |r| > 1 mm/yr in {int(big.sum())} cells (max {np.abs(res).max():.2f}), south of 60S"
          f" {int((big & (lat < -60)).sum())}, north of 59N between 75W and 10W {int(north.sum())};"
          f" with H > 10 m at the start of the year {int((big & (H0 > 10)).sum())}; the others:"
          f" {[(float(a), float(b), round(float(h), 2), round(float(f), 3)) for a, b, h, f in zip(lat[big & (H0 <= 10)], lon[big & (H0 <= 10)], H0[big & (H0 <= 10)], fi0[big & (H0 <= 10)])]}")
    k = int(np.argmax(np.abs(res)))
    print(f"diagyear current code: largest |r| {res[k]:+.3f} mm/yr at {lat[k]:.2f}, {lon[k]:.2f}, pack {H0[k]:.1f} m")
    deep = H0 > 10
    print(f"diagyear current code: cells without a pack thicker than 10 m: {int((~deep).sum())}, residual median"
          f" {np.median(res[~deep]):+.4f}, max |r| {np.abs(res[~deep]).max():.3f} mm/yr")
    print(f"diagyear current code: {int(deep.sum())} cells with H > 10 m: residual net {(res[deep] * area[deep]).sum() / 1e12:+.2f},"
          f" magnitude {(np.abs(res[deep]) * area[deep]).sum() / 1e12:.2f} Gt/yr")
    for lo, hi in ((10, 30), (30, 100), (100, 300), (300, 1e9)):
        sel = (fi0 >= 1 - 1e-6) & (H0 > lo) & (H0 <= hi)
        print(f"  H {lo:g}-{hi:g} m: {int(sel.sum())} cells, residual median {np.median(res[sel]):+.3f},"
              f" 5-95 % {np.round(np.quantile(res[sel], [0.05, 0.95]), 3)} mm/yr")


FIGURES = {"7.1": fig_7_1, "7.2": fig_7_2, "7.3": fig_7_3, "7.4": fig_7_4}
EXTRA = {"balance": balance, "states": states, "diagyear": diagyear}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        {**FIGURES, **EXTRA}[key]()
        print(key, "done")
