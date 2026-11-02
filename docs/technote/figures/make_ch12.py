"""Figures of Chapter 12 of the IBIS 3.0 technical note (soil biogeochemistry and spin-up), drawn from the model's code,
the release parameter file and the release run (verify_outputs/final_rel30).

Run from the repository root with the project interpreter (CPU is enough):

    JAX_PLATFORMS=cpu PYTHONPATH=python-port /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch12.py
    ... make_ch12.py 12.2 12.3                                                          # some

Every number a figure shows is computed here: the pool constants and the litter split from the release parameters
(`ibis.io.physics_params`), the climate factors from the expressions of `ibis/timestepping/loops_gswp3.py` (checked
against the model's `decomposition_factor` and `som_climate_ratio`), the per-cell parameters from the model's own type
gathering (`ibis.io.pft_types.apply_types`) and its cell sensitivity (`loops_gswp3.cell_lt_e0`, the rule of the current
code, technote review D45), and the pools and fluxes from the release run's restarts and yearly output.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast  # noqa: E402  (also sets rcParams)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
PFILE = str(Path(__file__).resolve().parents[3] / "python-port/configs/params/params_31s_ch4h.yaml")
REL = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30"
KOPPEN = "/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz"
PAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
REF = "#8a8a86"
T_BASE, T_ZERO = 288.16, 227.13                       # loops_gswp3.py decomposition_factor: _btemp, _tzero


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def _release_params(period=0):
    """Release parameters gathered per land cell for the Köppen period `period` (as run_global does)."""
    from ibis.io.physics_params import load_physics_params, build_all_params
    from ibis.io import pft_types as P
    land = _land_mask(); ij = np.argwhere(land)
    lat = 89.75 - 0.5 * ij[:, 0]
    d = load_physics_params(PFILE)
    params, pft, _ = build_all_params(len(ij), np.deg2rad(lat), d)
    z = np.asarray(P.zones_from_classes(np.load(KOPPEN)["classes"])[period])[ij[:, 0], ij[:, 1]]
    p2, _ = P.apply_types(params, pft, z, P.parse_type_overrides(d.get("types")))
    return d, params, p2, z, ij


VEGTYPE_1992 = "/shared/nas2/Min/ESACCI_LC/ibis_vegtype/vegtype_frac_1992.nc"


def _eval_area(land):
    """(360, 720) area (m2) of each land cell times its land fraction, the convention of the release's global totals
    (verify/calibration/landmask.py, diagnostics/mip_compare.py): the land fraction is the sum of the fifteen IBIS
    vegetation-type fractions of the ESA CCI land cover of 1992; the model itself treats each cell as entirely land."""
    import xarray as xr
    d = xr.open_dataset(VEGTYPE_1992)
    lf = np.asarray(d["vegfrac"].values, dtype=float).sum(axis=-1)
    if float(d["lat"].values[0]) < float(d["lat"].values[-1]):
        lf = lf[::-1, :]
    lf = np.clip(np.nan_to_num(lf), 0.0, 1.0)
    lat = np.arange(89.75, -90, -0.5)
    a = 6371.0e3 ** 2 * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 720))
    return np.where(land, a * lf, 0.0)


def _restart_fields(path, fields):
    """Named leaves of a restart (.npz of the flattened IBISState); the leaf order is the field order of the state
    classes, checked against the leaf counts of the saved tree structure."""
    import jax
    from ibis.state.soil import create_soil_state
    from ibis.state.snow import create_snow_state
    from ibis.state.canopy import create_canopy_state
    from ibis.state.atmosphere import create_atm_state
    from ibis.state.vegetation import create_vegetation_state
    from ibis.state.carbon import create_carbon_state
    from ibis.state.hydrology import create_hydrology_diag
    from ibis.state.summary import create_summary_state
    from ibis.bgc.methane import create_methane_state
    from ibis.state.stress_accum import create_stress_accum
    from ibis.state.et_parts import create_et_parts
    from ibis.state.seasonal import create_seasonal
    z = np.load(path, allow_pickle=False)
    tdef = str(z["treedef_repr"])
    mods = [("soil", create_soil_state(2, 6, 3)), ("snow", create_snow_state(2, 3)), ("canopy", create_canopy_state(2)),
            ("atm", create_atm_state(2, 2)), ("vegetation", create_vegetation_state(2, 12, 6)),
            ("carbon", create_carbon_state(2)), ("hydrology", create_hydrology_diag(2)), ("summary", create_summary_state(2, 12))]
    if "MethaneState" in tdef:
        mods.append(("wetland", create_methane_state(2)))
    mods += [("stress", create_stress_accum(2)), ("etparts", create_et_parts(2)), ("seasonal", create_seasonal(2))]
    names, counts = [], []
    for pre, m in mods:
        n = [pre + "." + ".".join(str(getattr(p, "name", p)) for p in path_)
             for path_, _ in jax.tree_util.tree_flatten_with_path(m)[0]]
        names += n; counts.append(len(n))
    saved = [s.count("*") for s in re.findall(r"CustomNode\(\w+\[\(Missing,\)\], \[([^\]]*)\]\)", tdef)]
    assert saved == counts and int(z["num_leaves"]) == len(names), (saved, counts)
    return {f: z[f"leaf_{names.index(f)}"] for f in fields}


def f_temp(t, e0):
    """Temperature factor of decomposition, as in loops_gswp3.py `decomposition_factor`: Lloyd-Taylor, held at its
    237.13 K value below 237.13 K, cap 10."""
    t = np.maximum(t, T_ZERO + 10.0)
    return np.minimum(np.exp(e0 * (1.0 / (T_BASE - T_ZERO) - 1.0 / (t - T_ZERO))), 10.0)


def f_moist(wfps):
    """Moisture factor of decomposition, as in loops_gswp3.py `decomposition_factor` (wfps in %, wfps_opt 60 %)."""
    return np.where(wfps >= 60.0, 0.000371 * wfps ** 2 - 0.0748 * wfps + 4.13, np.exp((wfps - 60.0) ** 2 / -800.0))


def _split(b):
    """Litter split fractions of soilbgc (lignin, structural, metabolic) for leaf, root and wood litter."""
    cnr = np.asarray(b.cnr, np.float64); out = {}
    for name, cn, cs in (("leaf", b.cnleaf, cnr[5]), ("root", b.cnroot, cnr[5]), ("wood", b.cnwood, cnr[7])):
        fl = b.fmax * cn ** 2 / (b.rconst + cn ** 2)
        fs = (1.0 / cn - fl / cnr[4] - (1.0 - fl) / cnr[6]) / (1.0 / cs - 1.0 / cnr[6])
        out[name] = (float(fl), float(fs), float(1.0 - fl - fs))
    return out


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------------------------------------
# Figure 12.1: the litter and soil carbon pools and their flows
# ---------------------------------------------------------------------------------------------------------------

def fig_12_1():
    from matplotlib.lines import Line2D
    from ibis.io.physics_params import load_physics_params, build_bgc_params
    b = build_bgc_params(load_physics_params(PFILE))
    sp = _split(b)
    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    ax.set_xlim(0, 133.5); ax.set_ylim(0, 75); ax.axis("off")
    C_MS, C_LIG, C_MIC, C_SOM, C_OUT = PAL[0], PAL[1], PAL[2], PAL[6], REF
    FS = 6.6

    def box(x0, y0, w, h, fc, title, lines):
        ax.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec="none"))
        ax.text(x0 + 1.0, y0 + h - 1.0, title, fontsize=7.6, weight="bold", color=INK, va="top")
        for k, ln in enumerate(lines):
            ax.text(x0 + 1.0, y0 + h - 5.0 - 2.9 * k, ln, fontsize=FS, color=INK2, va="top")

    def arrow(p, q, c, ls="-", lw=0.9):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=c, lw=lw, ls=ls,
                                     shrinkA=0, shrinkB=0))

    def line(xs, ys, c, ls="-"):
        ax.plot(xs, ys, color=c, lw=0.9, ls=ls, solid_capstyle="butt")

    # litter: nine pools as a table (source x chemistry)
    LX, LY, LW, LH = 1.0, 17.0, 41.0, 42.0
    ax.add_patch(Rectangle((LX, LY), LW, LH, fc="#f3ead7", ec="none"))
    ax.text(LX + 1.0, LY + LH - 1.0, "litter: nine pools", fontsize=7.6, weight="bold", color=INK, va="top")
    cols = (LX + 19.0, LX + 26.5, LX + 34.0, LX + 39.8)
    yy = LY + LH - 6.0
    for c, h in zip(cols, ("leaf", "root", "wood", "yield")):
        ax.text(c, yy, h, fontsize=FS, color=INK, ha="right", va="top", style="italic")
    rows = [("metabolic", 2, (b.klm, b.krm, b.kwm), b.ylm), ("structural", 1, (b.kls, b.krs, b.kws), b.yls),
            ("lignin", 0, (b.kll, b.krl, b.kwl), b.yll)]
    yy -= 3.6
    for name, ix, ks, yv in rows:
        ax.text(LX + 1.0, yy, name, fontsize=FS, color=INK2, va="top")
        for c, src in zip(cols[:3], ("leaf", "root", "wood")):
            ax.text(c, yy, f"{100 * sp[src][ix]:.1f} %".replace("-", "−"), fontsize=FS, color=INK2, ha="right",
                    va="top")
        ax.text(cols[3], yy, f"{yv:g}", fontsize=FS, color=INK2, ha="right", va="top")
        yy -= 2.9
        ax.text(LX + 3.0, yy, "k (d$^{-1}$)", fontsize=FS, color=INK2, va="top")
        for c, kv in zip(cols[:3], ks):
            ax.text(c, yy, f"{kv:g}", fontsize=FS, color=INK2, ha="right", va="top")
        yy -= 3.9
    ax.text(LX + 1.0, yy, "climate factor", fontsize=FS, color=INK2, va="top")
    for c, fac in zip(cols[:3], ("$f_l$", "$f_s$", "$f_l$")):
        ax.text(c, yy, fac, fontsize=FS + 0.4, color=INK2, ha="right", va="top")
    ax.text(LX + 1.0, LY + 7.4, "first row: share of the litterfall; k: rate constant", fontsize=FS - 0.2, color=INK2, style="italic")
    ax.text(LX + 1.0, LY + 4.6, "$f_l$: top soil layer; $f_s$: layers 1-4", fontsize=FS - 0.2, color=INK2, style="italic")
    ax.text(LX + 1.0, LY + 1.8, "lignin goes to the slow pools", fontsize=FS - 0.2, color=INK2, style="italic")
    ax.text(LX + 1.0, 70.5, "litterfall (Ch. 11, 13, 15)", fontsize=7.0, color=INK, va="center")
    arrow((LX + 8.0, 68.6), (LX + 8.0, LY + LH), INK, lw=1.1)                           # input: black, not a loss

    # microbial biomass, slow and passive pools
    def fmt(v):
        if v >= 1e-3:
            return f"{v:g}"
        m, e = f"{v:.1e}".split("e")
        return f"${float(m):g}\\times10^{{{int(e)}}}$"

    MX, MY, MW, MH = 51.5, 24.0, 24.5, 27.0
    box(MX, MY, MW, MH, "#e3eefb", "microbial biomass",
        [f"protected: ≤ {100 * b.fbsom:.1f} % of all C", f"   k {b.kbp:g} d$^{{-1}}$", "non-protected: the rest",
         f"   k {b.kbn:g} d$^{{-1}}$", f"{100 * b.effac:.0f} % of the turnover leaves,", "   half to each slow pool",
         "no climate factor"])
    S1 = (82.0, 46.0, 24.0, 17.0); S2 = (82.0, 8.0, 24.0, 17.0); PZ = (110.0, 28.0, 22.8, 17.0)
    box(*S1, "#e6f2e6", "slow, non-protected",
        [f"to biomass: {fmt(b.knb)} d$^{{-1}}$", f"   yield {b.ynb:g}", f"to passive: {fmt(b.kns)} d$^{{-1}}$"])
    box(*S2, "#e6f2e6", "slow, protected",
        [f"to biomass: {fmt(b.kpb)} d$^{{-1}}$", f"   yield {b.ypb:g}", f"to passive: {fmt(b.kps)} d$^{{-1}}$"])
    box(*PZ, "#efe8f7", "passive", [f"to biomass: {fmt(b.ksb)} d$^{{-1}}$", f"   yield {b.ysb:g}"])
    for (x0, y0, w, h) in (S1, S2, PZ):
        ax.text(x0 + w - 1.0, y0 + 1.0, "× $f_s$ $s$", fontsize=FS, color=INK2, ha="right", va="bottom")
    ax.add_patch(Rectangle((MX, 62.0), MW, 5.6, fc="#efefed", ec="none"))
    ax.text(MX + MW / 2, 64.8, "CO$_2$", fontsize=7.6, color=INK, ha="center", va="center")

    # flows
    arrow((LX + LW, 37.5), (MX, 37.5), C_MS)                                          # metabolic + structural
    ax.text((LX + LW + MX) / 2, 39.0, "metabolic,\nstructural", fontsize=FS - 0.4, color=INK2, ha="center", va="bottom")
    jx = LX + LW - 4.0                                                               # lignin: over the top to S1,
    line([jx, jx], [LY + LH, 72.0], C_LIG); line([jx, S1[0] + 11.5], [72.0, 72.0], C_LIG)
    arrow((S1[0] + 11.5, 72.0), (S1[0] + 11.5, S1[1] + S1[3]), C_LIG)
    line([jx, jx], [LY, 2.5], C_LIG); line([jx, S2[0] + 11.5], [2.5, 2.5], C_LIG)     # and under the bottom to S2
    arrow((S2[0] + 11.5, 2.5), (S2[0] + 11.5, S2[1]), C_LIG)
    ax.text(66.0, 72.8, f"lignin: {100 * b.lig_frac:.0f} % to each slow pool", fontsize=FS - 0.2, color=INK2, ha="center",
            va="bottom")
    arrow((MX + MW, 49.0), (S1[0], 58.0), C_MIC)                                      # dead biomass
    arrow((MX + MW, 26.0), (S2[0], 14.0), C_MIC)
    arrow((S1[0], 51.0), (MX + MW, 44.0), C_SOM)                                     # organic matter to biomass
    arrow((S2[0], 21.0), (MX + MW, 31.0), C_SOM)
    arrow((PZ[0], 37.5), (MX + MW, 37.5), C_SOM)
    arrow((S1[0] + S1[2], 52.0), (PZ[0] + 6.0, PZ[1] + PZ[3]), C_SOM, ls=(0, (3, 2)))  # to passive
    arrow((S2[0] + S2[2], 19.0), (PZ[0] + 6.0, PZ[1]), C_SOM, ls=(0, (3, 2)))
    arrow((MX + MW / 2, MY + MH), (MX + MW / 2, 62.0), C_OUT, lw=1.1)                  # respiration
    ax.text(MX + MW / 2 + 1.2, 55.5, "1 − y of each flow\ninto the biomass", fontsize=FS - 0.4, color=INK2, va="center")
    # leaching, out of each of the three pools (soilbgc: fleach x csoislon, csoislop, csoipas)
    for (x0, y0), y1, va in (((S1[0] + S1[2] - 3.0, S1[1] + S1[3]), S1[1] + S1[3] + 4.6, "bottom"),
                             ((S2[0] + S2[2] - 3.0, S2[1]), S2[1] - 4.6, "top"),
                             ((PZ[0] + 15.0, PZ[1]), PZ[1] - 4.6, "top")):
        arrow((x0, y0), (x0, y1), C_OUT, lw=1.0)
        ax.text(x0 + 1.0, y1 + (0.3 if va == "bottom" else -0.3), "leaching", fontsize=FS - 0.4, color=INK2,
                ha="left", va=va)
    ax.text(PZ[0], 14.0, "$s$: turnover factor (12.7)", fontsize=FS - 0.4, color=INK2, style="italic")
    yb = {"ybn": b.ybn, "ybp": b.ybp, "yns": b.yns, "yps": b.yps}
    assert all(v == 1.0 for v in yb.values()), yb                                      # the legend's "yield 1"
    hs = [Line2D([], [], color=C_MS, lw=1.2, label="litter to biomass"),
          Line2D([], [], color=C_LIG, lw=1.2, label="lignin to the slow pools"),
          Line2D([], [], color=C_MIC, lw=1.2, label="dead biomass to the slow pools (yield 1)"),
          Line2D([], [], color=C_SOM, lw=1.2, label="organic matter to biomass"),
          Line2D([], [], color=C_SOM, lw=1.2, ls=(0, (3, 2)), label="slow pools to passive (yield 1)"),
          Line2D([], [], color=C_OUT, lw=1.2, label="loss: CO$_2$, leaching")]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, -0.075), ncol=3, frameon=False, fontsize=6.4,
              handlelength=2.2, columnspacing=1.4)
    fig.savefig(OUT / "fig_12_1_pools.png")
    plt.close(fig)
    print("12.1 split", {k: tuple(round(100 * x, 2) for x in v) for k, v in sp.items()})


# ---------------------------------------------------------------------------------------------------------------
# Figure 12.2: the temperature and moisture factors of decomposition
# ---------------------------------------------------------------------------------------------------------------

def fig_12_2():
    import jax.numpy as jnp
    from ibis.physics.biogeochem import som_climate_ratio
    from ibis.io.physics_params import load_physics_params
    from ibis.io import pft_types as P
    d = load_physics_params(PFILE)
    e0_types = np.array(sorted(P.parse_type_overrides(d.get("types"))["lt_e0"].values()))
    tc = np.linspace(-30.0, 45.0, 751); tk = tc + 273.15
    # the model's own temperature factor: with tref = T_BASE and a vanishing SOM E0, som_climate_ratio = 1 / f_litter(T)
    for e0 in (344.0, e0_types.min(), e0_types.max()):
        r = np.asarray(som_climate_ratio(jnp.asarray(tk, jnp.float32), jnp.full(tk.shape, e0, jnp.float32), 1e-9, T_BASE))
        assert np.allclose(1.0 / r, f_temp(tk, e0), rtol=2e-4), e0
    # and the model's decomposition factor (12.6) itself, k_scale 1: f_T x f_W clipped to 0.001-10 (technote review D2)
    from ibis.timestepping.loops_gswp3 import decomposition_factor
    for e0 in (344.0, e0_types.min(), e0_types.max()):
        for wf in (30.0, 60.0, 90.0):
            got = np.asarray(decomposition_factor(jnp.asarray(tk, jnp.float32), jnp.full(tk.shape, wf / 100.0, jnp.float32),
                                                  jnp.zeros(tk.shape, jnp.float32), jnp.float32(e0)))
            assert np.allclose(got, np.clip(f_temp(tk, e0) * f_moist(np.array(wf)), 0.001, 10.0), rtol=2e-4), (e0, wf)
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw={"wspace": 0.30, "width_ratios": [1.25, 1]})
    ax = axs[0]
    ax.plot(tc, f_temp(tk, 344.0), color=REF, lw=1.3, ls=(0, (4, 2)), label="E$_0$ = 344 K (IBIS 2.6b4)")
    for e0, c, lab in ((e0_types.min(), PAL[0], "lowest"), (float(np.median(e0_types)), PAL[1], "median"),
                       (e0_types.max(), PAL[2], "highest")):
        ax.plot(tc, f_temp(tk, e0), color=c, lw=1.4, label=f"E$_0$ = {e0:.0f} K ({lab} of the 35 types)")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 30); ax.set_xlim(-30, 45)
    ax.axhline(1.0, color=INK2, lw=0.5)
    ax.text(44, 0.62, "1 at 15 °C", fontsize=6.5, color=INK2, ha="right")
    ax.set_xlabel("soil temperature (°C)"); ax.set_ylabel("temperature factor $f_T$")
    ax.set_title("(a) temperature factor", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    _clean(ax)
    ax = axs[1]
    w = np.linspace(0.0, 100.0, 1001)
    lo, hi = w < 60.0, w >= 60.0
    ax.plot(w[lo], f_moist(w[lo]), color=PAL[0], lw=1.4)
    ax.plot(w[hi], f_moist(w[hi]), color=PAL[0], lw=1.4)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.08)
    ax.annotate(f"{float(f_moist(np.array(60.0))):.3f} at 60 %", xy=(60.5, float(f_moist(np.array(60.0)))),
                xytext=(66, 1.02), fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    f100 = float(f_moist(np.array(100.0)))
    ax.annotate(f"{f100:.2f} at 100 %", xy=(99.5, f100), xytext=(86, 0.20), fontsize=6.5, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.set_xlabel("water-filled pore space (%)"); ax.set_ylabel("moisture factor $f_W$")
    ax.set_title("(b) moisture factor", loc="left", fontsize=9, weight="bold", color=INK)
    _clean(ax)
    fig.savefig(OUT / "fig_12_2_climate_factors.png")
    plt.close(fig)
    print(f"12.2 type E0 min {e0_types.min():.1f} median {np.median(e0_types):.1f} max {e0_types.max():.1f};"
          f" f_T(0 C) at 344 {float(f_temp(273.15, 344.0)):.3f}; f_W(60) {float(f_moist(np.array(60.0))):.4f}"
          f" f_W(100) {float(f_moist(np.array(100.0))):.3f}; cap reached at 45 C for E0 >= "
          f"{np.log(10.0) / (1 / (T_BASE - T_ZERO) - 1 / (318.15 - T_ZERO)):.0f} K")


# ---------------------------------------------------------------------------------------------------------------
# Figure 12.3: the per-cell parameters of the release: SOM turnover factor and the cell's temperature sensitivity
# ---------------------------------------------------------------------------------------------------------------

def fig_12_3():
    land = _land_mask()
    _, _, p0, z0, ij = _release_params(0)
    sk = np.asarray(p0.bgc_params.som_k_scale, np.float64)
    _, _, p3, z3, _ = _release_params(3)
    lt = np.asarray(p3.bgc_params.lt_e0, np.float64)
    rs = _restart_fields(f"{REL}/hindcast/restart/ibis_restart_2010.npz", ["vegetation.plai"])
    # the cell's E0 by the model's own rule (loops_gswp3.cell_lt_e0, in the model's float32): leaf-area-weighted over the
    # positions, the positions' mean where the total leaf area is below 1e-6 (technote review D45)
    import jax.numpy as jnp
    from ibis.timestepping.loops_gswp3 import cell_lt_e0
    plai32 = jnp.asarray(rs["vegetation.plai"], jnp.float32)
    e0 = np.asarray(cell_lt_e0(plai32, jnp.asarray(lt, jnp.float32)), np.float64)
    ws = np.asarray(jnp.maximum(plai32, 0.0).sum(axis=1), np.float64)            # the model's total leaf area
    w = np.maximum(rs["vegetation.plai"].astype(np.float64), 0.0)
    e0_rel = np.where(ws > 0, (w * lt).sum(1) / np.maximum(ws, 1e-6), lt.mean(1))  # the release run's rule, for the log
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 8.8), gridspec_kw={"hspace": 0.42})
    # (a) SOM turnover factor
    vals = np.unique(np.round(sk, 4))
    blues = ["#0d3b73", "#1f5ea8", "#2a78d6", "#6ea3e6", "#a9c8f0", "#dbe7f7"][:len(vals)]
    grid = np.full(land.shape, np.nan); grid[ij[:, 0], ij[:, 1]] = np.searchsorted(vals, np.round(sk, 4))
    ax = axs[0]
    ax.imshow(grid, extent=(-180, 180, -90, 90), cmap=ListedColormap(blues),
              norm=BoundaryNorm(np.arange(-0.5, len(vals)), len(vals)), interpolation="nearest")
    _coast(ax, land); _map_axes(ax)
    ax.set_title("(a) turnover factor $\\sigma$ of the slow and passive pools, zones of 1901–1930", loc="left", fontsize=9,
                 weight="bold", color=INK)
    from matplotlib.patches import Patch
    hs = [Patch(fc=c, ec=GRID if c == blues[-1] else "none", lw=0.5,
                label=f"{v:g} ({int((np.round(sk, 4) == v).sum()):,} cells)") for v, c in zip(vals, blues)]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=6, frameon=False, fontsize=7,
              handlelength=1.2, columnspacing=1.0)
    # (b) cell E0
    ax = axs[1]
    bounds = [150, 200, 225, 250, 275, 300, 325, 350, 400, 530]
    oranges = ["#fde9dc", "#fbd0b5", "#f8b48c", "#f39764", "#eb6834", "#cf5222", "#a83f17", "#7a2c0e", "#4d1b08"]
    lowlai = ws < 1e-6                                  # no or vanishing leaf area: the mean of the twelve positions
    g2 = np.full(land.shape, np.nan); g2[ij[:, 0], ij[:, 1]] = np.where(lowlai, np.nan, e0)
    g0 = np.full(land.shape, np.nan); g0[ij[:, 0], ij[:, 1]] = np.where(lowlai, 0.0, np.nan)
    ax.imshow(g0, extent=(-180, 180, -90, 90), cmap=ListedColormap(["#c9c9c5"]),
              norm=BoundaryNorm([-0.5, 0.5], 1), interpolation="nearest")
    im = ax.imshow(g2, extent=(-180, 180, -90, 90), cmap=ListedColormap(oranges), norm=BoundaryNorm(bounds, len(oranges)),
                   interpolation="nearest")
    _coast(ax, land); _map_axes(ax)
    ax.set_title("(b) temperature sensitivity $E_0$ of the cell, release state of 2010", loc="left", fontsize=9,
                 weight="bold", color=INK)
    cax = ax.inset_axes([0.0, -0.17, 1.0, 0.035])     # inside the map's width, so both maps keep the same size
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=bounds)
    cb.set_label("$E_0$ (K), leaf-area-weighted over the positions", fontsize=8); cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    ax.legend(handles=[Patch(fc="#c9c9c5", ec="none",
                             label=f"total leaf area below 10$^{{-6}}$: mean of the twelve positions ({int(lowlai.sum()):,} cells)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.29), ncol=1, frameon=False, fontsize=7, handlelength=1.2)
    fig.savefig(OUT / "fig_12_3_cell_parameters.png")
    plt.close(fig)
    tiny = (ws > 0.0) & (ws < 1e-6)
    hi = ws >= 1e-6
    zt = np.asarray(z3)[tiny]
    print("12.3 som_k_scale cells", {float(v): int((np.round(sk, 4) == v).sum()) for v in vals},
          f"E0 2010 (current rule): cells with total leaf area >= 1e-6 {int(hi.sum())}: min {e0[hi].min():.1f} p5 "
          f"{np.percentile(e0[hi], 5):.1f} median {np.median(e0[hi]):.1f} p95 {np.percentile(e0[hi], 95):.1f} max {e0[hi].max():.1f};"
          f" below 1e-6 {int(lowlai.sum())}, {int((lowlai & (np.asarray(z3) == 12)).sum())} of them in E_ice (zero {int((ws <= 0).sum())};"
          f" positive but below 1e-6 {int(tiny.sum())}); position means of the low cells: min {e0[lowlai].min():.1f} median"
          f" {np.median(e0[lowlai]):.1f} max {e0[lowlai].max():.1f}")
    print(f"12.3 release rule on the same state: the {int(tiny.sum())} tiny cells had E0 max {e0_rel[tiny].max():.3g} K; now "
          f"{e0[tiny].min():.1f}-{e0[tiny].max():.1f} K; cells whose E0 changes: {int((np.abs(e0 - e0_rel) > 1e-3).sum())};"
          f" tiny cells by zone code {dict(zip(*np.unique(zt, return_counts=True)))}; lat "
          f"{(89.75 - 0.5 * ij[tiny][:, 0]).min():.2f}-{(89.75 - 0.5 * ij[tiny][:, 0]).max():.2f}")


# ---------------------------------------------------------------------------------------------------------------
# Figure 12.4: the release run's litter and soil carbon from the cold start to 2010
# ---------------------------------------------------------------------------------------------------------------

def fig_12_4():
    import netCDF4 as nc
    land = _land_mask()
    area = _eval_area(land)
    years = np.arange(1701, 2011)
    keys = ("totlit", "totcsoi", "aynpptot", "ayco2mic", "ayrh_ch4c")
    series = {k: np.zeros(len(years)) for k in keys}
    for n, y in enumerate(years):
        stage = "spinup" if y <= 1900 else ("transient" if y <= 2000 else "hindcast")
        ds = nc.Dataset(f"{REL}/{stage}/out/ibis_yearly_{y}.nc")
        for k in keys:
            if k not in ds.variables:                 # ayrh_ch4c: only with methane_carbon_partition (1901-2010)
                continue
            v = np.ma.filled(ds.variables[k][:].astype(np.float64), np.nan)
            series[k][n] = np.nansum(np.where(land, v, 0.0) * area) / 1e12          # PgC or PgC yr-1
        ds.close()
    rh = series["ayco2mic"] + series["ayrh_ch4c"]      # the carbon decomposed: Rh as CO2 plus the methane carbon
    fig, axs = plt.subplots(3, 1, figsize=(7.4, 7.4), sharex=True, gridspec_kw={"hspace": 0.32})
    for ax in axs:
        ax.axvspan(1701, 1850.999, color="#eef3fa", lw=0)
        for x in (1900.5, 2000.5):
            ax.axvline(x, color=INK2, lw=0.6, ls=":")
        _clean(ax)
    i = lambda y: int(y - 1701)
    ax = axs[0]
    ax.plot(years, series["totlit"], color=PAL[1], lw=1.2)
    ytop = 160.0
    ax.set_ylim(0, ytop); ax.set_ylabel("PgC")
    ypk = int(years[np.argmax(series["totlit"])]); vpk = series["totlit"].max()
    ax.annotate(f"{vpk:.0f} PgC in {ypk}", xy=(ypk, ytop), xytext=(ypk + 14, 140), fontsize=6.5, color=INK2,
                va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.set_title("(a) litter carbon", loc="left", fontsize=9, weight="bold", color=INK)
    for x, lab in ((1800, "spin-up, decomposition ×10"), (1875.5, "spin-up, ×1"), (1950.5, "transient")):
        ax.text(x, ytop * 0.96, lab, ha="center", va="top", fontsize=7, color=INK2)
    ax.text(2005.5, ytop * 0.96, "hindcast", ha="center", va="top", fontsize=6.5, color=INK2, rotation=90)
    ax = axs[1]
    ax.plot(years, series["totcsoi"], color=PAL[0], lw=1.2)
    ax.set_ylabel("PgC"); ax.set_ylim(0, None)
    ax.set_title("(b) soil carbon (microbial biomass, slow and passive pools)", loc="left", fontsize=9, weight="bold",
                 color=INK)
    ax = axs[2]
    ax.plot(years, series["aynpptot"], color=PAL[2], lw=1.2, label="net primary production")
    ax.plot(years, rh, color=PAL[6], lw=1.2, label="heterotrophic respiration (CO$_2$ and CH$_4$ carbon)")
    ax.set_ylabel("PgC yr$^{-1}$"); ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("(c) global carbon fluxes", loc="left", fontsize=9, weight="bold", color=INK)
    ax.set_xlim(1701, 2010); ax.set_xlabel("model year")
    fig.savefig(OUT / "fig_12_4_spinup.png")
    plt.close(fig)
    s = series
    print("12.4 litter 1701 {:.1f} peak {:.1f} in {} 1850 {:.1f} 1851 {:.1f} 1900 {:.1f} 2000 {:.1f} 2010 {:.1f}".format(
        s["totlit"][0], vpk, ypk, *[s["totlit"][i(y)] for y in (1850, 1851, 1900, 2000, 2010)]))
    print("12.4 soil 1750 {:.1f} 1800 {:.1f} 1850 {:.1f} 1851 {:.1f} 1900 {:.1f} 2000 {:.1f} 2010 {:.1f}".format(
        *[s["totcsoi"][i(y)] for y in (1750, 1800, 1850, 1851, 1900, 2000, 2010)]))
    print("12.4 soil drift 1891-1900 {:.3f} PgC/yr; 1841-1850 {:.3f}; 2001-2010 {:.3f}".format(
        (s["totcsoi"][i(1900)] - s["totcsoi"][i(1890)]) / 10, (s["totcsoi"][i(1850)] - s["totcsoi"][i(1840)]) / 10,
        (s["totcsoi"][i(2010)] - s["totcsoi"][i(2000)]) / 10))
    m = lambda k, a, b: s[k][i(a):i(b) + 1].mean() if k != "rh" else rh[i(a):i(b) + 1].mean()
    print("12.4 means npp / rh: 1841-1850 {:.2f} / {:.2f}; 1891-1900 {:.2f} / {:.2f}; 2001-2010 {:.2f} / {:.2f}"
          " (CO2 only {:.2f}, CH4 carbon {:.3f})".format(
              m("aynpptot", 1841, 1850), m("rh", 1841, 1850), m("aynpptot", 1891, 1900), m("rh", 1891, 1900),
              m("aynpptot", 2001, 2010), m("rh", 2001, 2010), m("ayco2mic", 2001, 2010), m("ayrh_ch4c", 2001, 2010)))


FIGURES = {"12.1": fig_12_1, "12.2": fig_12_2, "12.3": fig_12_3, "12.4": fig_12_4}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
        print("figure", key, "done")
