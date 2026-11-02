"""Figures of Chapter 16 (Calibration) of the IBIS 3.0 technical note, drawn from the calibration code, the per-tower
fit records and the release parameter file, and the numbers the chapter quotes that were computed for it.

Run from the repository root with the project interpreter (it needs matplotlib, pyyaml, pandas, xarray and the `ibis`
package):

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch16.py            # all figures
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch16.py 16.2       # some
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch16.py numbers    # the chapter's numbers

Sources. The per-tower fits are the records of the calibration campaign (`verify_outputs/site_fit/<stage>/<tower>/
best.json`, written by `verify/calibration/site_fit.py`); the per-type medians are rebuilt from them by the campaign's own
script (`verify/calibration/towers31_types_yaml.py`); the objective of Figure 16.2 is `site_fit.misfit` itself; Figure 16.4
bins the product tree cover by the stress of the protocol run the ramps were fitted on (s31n, 2008 and 2010) as
`ramp_fit_zones.py` does, refits the ramps and draws the release ramps resolved by `apply_types` from the release file.
The counts in Figure 16.1 are counted from the fit records and the release file when the script runs. Nothing is typed in
by hand except labels and the schematic's layout. `numbers` prints, block by block, the numbers of sections 16.4, 16.5,
16.6.1, 16.6.2 (Table 16.5), 16.8, 16.11, 16.12, 16.13 and 16.14 that are neither code constants nor read from a log or a
score file, each with what it is computed on; it writes nothing (counts and band medians only for the methane towers,
whose data are not public; the band tool's own output, which names the towers it leaves out, is not printed).

The figures show the calibration of the release, made with the calibration tools as they were then. Several tools changed
after the release (technote section 16.15): the tower reader aligns hourly shortwave with solar time (`tower_forcing.py`),
the day and night NEE targets are means (`tower_nee_daynight.py`), the C4
capacity bounds are converted with the C4 temperature factor (`site_fit.vmax_bounds_15c`), and the band medians of the
methane parameters leave out the towers with a net uptake (`ch4_class_yaml.py --band-csv`). `numbers` prints the
release's values and, where the change moves a quoted number, the current tool's.
"""
from __future__ import annotations

import os
import sys
import glob
import json
import calendar
import collections
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: F401  (style and rcParams)

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "python-port"), str(ROOT / "verify" / "calibration")]
OUT = Path(__file__).resolve().parent

REL = ROOT / "python-port/configs/params/params_31s_ch4h.yaml"
P31A = ROOT / "python-port/configs/params/params_31a.yaml"               # the type set re-scored at the towers on 20 October
BASE_S2 = ROOT / "python-port/configs/params/params_s7_crop.yaml"         # the base set of the campaign's tower fits
F77_CAN = ROOT / "fortran/params.can"                                    # IBIS 2.6b4 canopy and vegetation parameters
SITE_FIT = "/shared/nas2/Hangkai/ibis/verify_outputs/site_fit"
TOWERS31 = "/shared/nas2/Hangkai/ibis/inputs/sites/towers31/compact"
REL30 = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30"            # the release run
CMP_TOWERS = "/shared/nas2/Hangkai/ibis/verify_outputs/diag_work/nee_cycle/cmp_final_towers.csv"   # compare_sets.py, per tower
NEE_TARGETS = "/shared/nas2/Hangkai/ibis_local_document/data/tower_nee_daynight_t31.npz"           # tower_nee_daynight.py
CH4_LOGS = "/shared/nas2/Hangkai/ibis_local_document/logs"                # ch4_fit_report_hourly.csv, ch4_towers_zone.csv
CH4_SITES = "/shared/nas2/Hangkai/ibis/inputs/sites/methane_towers/ch4_sites.csv"   # ch4_tower_prep.py: budget_gC per tower
SURTA = "/shared/nas2/Min/HWSD/surta.nc"                                   # the binary land mask of the protocol
NATURAL = ("ENF", "EBF", "DBF", "DNF", "MF", "SAV", "WSA", "OSH", "CSH", "GRA")

# palette (fixed order)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                           "#e87ba4", "#008300", "#6250d6", "#e34948")


def _yaml(path):
    import yaml
    return yaml.safe_load(open(path))


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _fitted(stage):
    """(results, results with fitted parameters) of one stage of the campaign. A record without a fit is the 151-byte
    placeholder `{"loss": null, "params": {}, "note": ...}` (log section 45.2); the size test spares opening every file
    on the file server."""
    fs = glob.glob(f"{SITE_FIT}/{stage}/*/best.json")
    return len(fs), sum(1 for f in fs if os.path.getsize(f) > 200)


def _towers(*stages):
    return len({os.path.basename(os.path.dirname(f)) for s in stages for f in glob.glob(f"{SITE_FIT}/{s}/*/best.json")})


def _towers_fitted(*stages):
    """Distinct towers with a best.json that carries fitted parameters in any of `stages`."""
    return len({os.path.basename(os.path.dirname(f)) for s in stages for f in glob.glob(f"{SITE_FIT}/{s}/*/best.json")
                if os.path.getsize(f) > 200})


# ---------------------------------------------------------------------------------------------------------------
# Figure 16.1: the calibration chain
# ---------------------------------------------------------------------------------------------------------------

def fig_16_1():
    rel = _yaml(REL)
    n_s2_all, n_s2 = _fitted("t31_s2")
    _, n_open = _fitted("t31_open_lpj")
    _, n_relax = _fitted("t31_relax")
    _, n_nee = _fitted("t31_nee")
    n_ch4 = _towers_fitted("ch4_h1", "ch4_h1_gpu", "ch4_h1_fix")      # one tower refitted on repaired pools (log section 80)
    n_crop = _towers("crop2", "crop2_gpu")
    n_types = sum(1 for d in rel["types"].values() if "vmax" in d)
    n_bands = len(rel["methane_bands"])
    print(f"16.1 S2 {n_s2} of {n_s2_all}; open {n_open}; relaxed {n_relax}; NEE {n_nee}; CH4 {n_ch4}; crop {n_crop}; "
          f"types with tower values {n_types}; bands {n_bands}")

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")

    def box(x, y, w, h, head, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.7))
        ax.text(x + 0.9, y + h - 0.9, head, fontsize=7.6, color=INK, weight="bold", va="top")
        ax.text(x + 0.9, y + h - 3.6, body, fontsize=6.6, color=INK2, va="top", linespacing=1.25)

    def arrow(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.7,
                                     connectionstyle=f"arc3,rad={rad}"))

    TW, TB = "#eaf2fb", "#9cbfe6"          # tower stages
    GW, GB = "#fdf0e7", "#f2b48f"          # global stages
    ax.text(1, 68.6, "At the towers: sections 16.2-16.6, 16.11, 16.12", fontsize=8.2, color=INK, weight="bold")
    ax.text(53, 68.6, "In the global protocol: sections 16.7-16.10, 16.13", fontsize=8.2, color=INK, weight="bold")
    # tower column
    box(1, 56.0, 46, 10.6, f"1  GPP and ET  ({n_s2} towers)",
        "capacity and stomatal slope of the prescribed positions,\nacclimation, phenology thresholds, drought trigger,"
        " stress factor;\nrefits: open towers under the saturating lower cover "
        f"({n_open}),\ncapacities at a trait bound with the bounds widened ({n_relax})", TW, TB)
    box(1, 44.4, 46, 8.8, f"2  Night- and day-time NEE  ({n_nee} towers)",
        "carbon pools spun up per window; growth respiration,\nroot respiration, temperature sensitivity of decomposition,\n"
        "soil-pool scale (the last not carried over)", TW, TB)
    box(1, 32.2, 46, 9.4, f"3  One value per plant type  ({n_types} types)",
        "median over the towers of each type; positions without a type\nkeep the base vectors (class medians of the"
        " earlier 62-tower fits;\ncapacity of positions 2 and 8 from IBIS 2.6b4)", TW, TB)
    box(1, 17.6, 22, 11.8, "4  Wetland CH4",
        f"{n_ch4} towers; 4 parameters,\nflux every time step;\nmedian per wetland\nband ({n_bands} bands)", TW, TB)
    box(25, 17.6, 22, 11.8, "5  Cropland",
        f"{n_crop} towers; capacity\nscale of the C3 and of\nthe C4 crop; median\nover the towers", TW, TB)
    arrow((24, 56.0), (24, 53.6)); arrow((24, 44.4), (24, 42.0))
    # global column
    gl = [
        ("6  Leaf area and allocation", "leaf, wood and root shares, leaf lifetime, specific\nleaf area; against product"
         " leaf area and biomass maps"),
        ("7  Fire", "damping of fire under a closed broadleaf canopy;\nzone objective against GFED5 burned area"),
        ("8  Water-limited tree cover", "ramp of the maximum tree cover per zone,\nfitted to CLM / MODIS tree cover"),
        ("9  Wood", "wood share and residence time of subtropical, temperate\nand tropical trees; against the"
         " Xu et al. (2021) biomass map"),
        ("10  Soil carbon", "turnover factor of the slow and passive pools per\nzone group; two-member bracket"
         " against HWSD2"),
    ]
    y = 58.2
    for head, body in gl:
        box(53, y, 46, 8.4, head, body, GW, GB)
        if y < 58.2:
            arrow((76, y + 8.4 + 1.6), (76, y + 8.4 + 0.5))
        y -= 10.6
    # release
    box(53, 2.0, 46, 9.0, "11  The release run and its scores",
        "cold spin-up 1701-1900, transient 1901-2000, hindcast\n2001-2010 (final_rel30); scored against budgets,"
        "\ngridded products and the towers (Section 16.13)", "#eef7f1", "#8fd0b2")
    arrow((76, 15.8), (76, 11.6))
    arrow((47.4, 37.0), (52.6, 61.0), rad=0.12)                 # bows into the gap between the columns, clear of box 2
    arrow((12, 17.6), (52.6, 9.4), rad=0.0)
    arrow((36, 17.6), (52.6, 10.4), rad=0.0)
    ax.text(1, 9.2, "Each global stage runs through the full protocol (or a ten-year\ncheck from an earlier protocol"
            " state) and is judged by model /\nreference ratios by zone, band or region; the soil bracket also\nagainst"
            " the literature range of the global stock and the land\nsink, the fire scale fitted offline on yearly"
            " protocol states.\nNo objective function contains a global total.",
            fontsize=6.6, color=INK2, va="top", linespacing=1.25)
    fig.savefig(OUT / "fig_16_1_calibration_chain.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 16.2: the bounded tower objective as a function of the model / tower ratio
# ---------------------------------------------------------------------------------------------------------------

def fig_16_2():
    import site_fit as SF
    # a synthetic tower: one warm-up year and three fitted years with a temperate seasonal GPP cycle (gC m-2 month-1),
    # NT = DT partition (so the uncertainty is the 10 % / 5 g floor of equation 16.1), northern summer
    m = np.arange(12)
    g = np.maximum(180.0 * np.sin(np.pi * (m - 2.5) / 8.0), 0.0) + 5.0
    G = np.tile(g, (4, 1))
    obs = {"gpp": G.copy(), "gpp_sig": np.maximum.reduce([0.1 * G, np.full_like(G, 5.0)]), "et": np.full_like(G, np.nan),
           "et_sig": np.full_like(G, np.nan), "season": [5, 6, 7]}
    years = [1, 2, 3]
    R = np.exp(np.linspace(np.log(0.2), np.log(5.0), 301))
    def loss(robust):
        SF.ROBUST = robust; SF.TARGET = "gpp"
        return np.array([SF.misfit((r * G)[..., None], np.zeros_like(G)[..., None], obs, 0, years, 0.0) for r in R])
    plain = loss(None)
    bounded = loss(dict(cap_month=25.0, cap_tower=30.0, w_mag=1.0, tol=0.15))
    lifted = loss(dict(cap_month=400.0, cap_tower=1e9, w_mag=1.0, tol=0.15))
    SF.ROBUST = None
    # the two parts of the bounded objective, from the same function with the other part switched off
    monthly = loss(dict(cap_month=25.0, cap_tower=1e9, w_mag=0.0, tol=0.15))
    SF.ROBUST = None
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot(R, plain, color="#9a9a96", lw=1.2, ls=(0, (4, 2)), label="plain, eq. (16.5)")
    ax.plot(R, lifted, color=ORANGE, lw=1.4, label="caps lifted (400, none)")
    ax.plot(R, monthly, color=AQUA, lw=1.4, label="bounded: monthly part")
    ax.plot(R, bounded, color=BLUE, lw=2.0, label="bounded, eq. (16.6), caps 25 and 30")
    ax.axhline(30.0, color=INK2, lw=0.6, ls=":")
    ax.text(4.9, 31.2, "tower cap 30", ha="right", fontsize=6.8, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(0.2, 5.0); ax.set_ylim(0, 60)
    from matplotlib.ticker import NullFormatter, NullLocator
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.2, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 5]); ax.set_xticklabels(["0.2", "0.33", "0.5", "0.67", "1", "1.5", "2", "3", "5"])
    ax.set_xlabel("model / tower ratio $R$ of every month's GPP"); ax.set_ylabel("loss of one tower")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); _spines(ax)
    # the legend sits above the axes: every curve crosses the upper band of the plot
    ax.legend(loc="lower center", frameon=False, fontsize=6.8, ncol=2, handlelength=2.2, columnspacing=1.6,
              bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
    fig.savefig(OUT / "fig_16_2_tower_objective.png")
    plt.close(fig)
    i1 = np.argmin(abs(R - 1.0)); flat_hi = R[(bounded >= 29.999) & (R > 1)]; flat_lo = R[(bounded >= 29.999) & (R < 1)]
    print(f"16.2 loss at R=1: {bounded[i1]:.3f}; bounded at the tower cap for R >= {flat_hi.min():.2f} and R <= {flat_lo.max():.2f}; "
          f"plain at R=0.5/2: {plain[np.argmin(abs(R - 0.5))]:.0f} / {plain[np.argmin(abs(R - 2))]:.0f}; "
          f"lifted at R=0.5/2: {lifted[np.argmin(abs(R - 0.5))]:.1f} / {lifted[np.argmin(abs(R - 2))]:.1f}")


# ---------------------------------------------------------------------------------------------------------------
# Figure 16.3: the per-type values against the spread of their towers
# ---------------------------------------------------------------------------------------------------------------

def _type_table():
    """The per-type medians and quartiles of the campaign, rebuilt by the campaign's own script."""
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        # keep the interpreter's own PYTHONPATH (the project wrapper puts its site-packages there) in front of the port
        env = dict(os.environ, IBIS_FLUX_DIR=TOWERS31,
                   PYTHONPATH=os.pathsep.join(x for x in (os.environ.get("PYTHONPATH", ""), str(ROOT / "python-port")) if x))
        cmd = [sys.executable, str(ROOT / "verify/calibration/towers31_types_yaml.py"), f"{SITE_FIT}/t31_s2_merged_relax",
               f"{SITE_FIT}/t31_nee", str(BASE_S2), f"{tmp}/types.yaml", f"{tmp}/table.csv"]
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr[-2000:])
        print("16.3", r.stdout.strip().splitlines()[0])
        tab = pd.read_csv(f"{tmp}/table.csv")
    if os.environ.get("CH16_TABLE_OUT"):                       # a copy for the chapter's number checks
        tab.to_csv(os.environ["CH16_TABLE_OUT"], index=False)
    return tab


def _f77_vmax():
    txt = open(F77_CAN).read().splitlines()
    i0 = [i for i, l in enumerate(txt) if l.startswith("# vmax_pft  specla")][0]
    return np.array([float(l.split("!")[0].split()[0]) for l in txt[i0 + 2:i0 + 14]])


def _vmax_bounds15():
    """The capacity bounds of the campaign per position (1-12) at 15 C: the TRY ranges at 25 C of the command generator
    (`towers31_cmds.VMAX_BOUNDS`) converted as `site_fit` converted them in the campaign, with the C3 temperature factor of
    the base set for every position (equation 16.9), and the widened bounds of the bound-relaxed refit (x0.5 and x1.5 at
    25 C). Also the ratio of the C4 factor, which the campaign did not use and which `site_fit.vmax_bounds_15c` now uses
    for the C4 position (section 16.5)."""
    import jax.numpy as jnp
    from towers31_cmds import VMAX_BOUNDS, PFT_NAMES        # the short position names of fit_carbon, as in VMAX_BOUNDS
    from ibis.io.physics_params import build_all_params, load_physics_params
    from ibis.physics.stomata import _temp_dep_params
    sp, _, _ = build_all_params(1, np.zeros(1, np.float32), load_physics_params(str(BASE_S2)))
    tv = lambda t, k: float(_temp_dep_params(jnp.asarray([t + 273.16], jnp.float32), sp.stomata_params)[k][0])
    r3 = tv(25.0, 4) / tv(15.0, 4); r4 = tv(25.0, 5) / tv(15.0, 5)       # index 4: tempvm_c3 (used); 5: tempvm_c4
    b = {}
    for item in VMAX_BOUNDS.split(";"):
        key, rng = item.split("="); lo, hi = (float(x) for x in rng.split(","))
        b[PFT_NAMES.index(key) + 1] = (lo / r3, hi / r3, 0.5 * lo / r3, 1.5 * hi / r3)
    return b, r3, r4


def fig_16_3():
    from ibis.io import pft_types as P
    tab = _type_table()
    rel = _yaml(REL)["types"]
    bounds, r3, _ = _vmax_bounds15()
    ids = [t[0] for t in P.TYPES if t[2] not in ("wetland", "crop")]
    form = {t[0]: t[2] for t in P.TYPES}; parent = {t[0]: t[4] for t in P.TYPES}
    col = lambda t: BLUE if form[t].startswith("tree") else (ORANGE if form[t].startswith("shrub") else AQUA)
    f77 = _f77_vmax() * 1e6
    fig, axs = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True, gridspec_kw={"hspace": 0.12})
    x = np.arange(len(ids))
    diff = []
    for ax, key, scale, ylab in ((axs[0], "vmax", 1e6, "$V_{max}$ at 15 °C (µmol m$^{-2}$ s$^{-1}$)"),
                                 (axs[1], "rgrowth", 1.0, "growth respiration fraction")):
        for k, t in enumerate(ids):
            r = tab[(tab.type == t) & (tab.key == key)]
            if r.empty:
                continue
            r = r.iloc[0]
            if key == "vmax":                                   # the search bounds of the type's parent position
                lo, hi, wlo, whi = bounds[parent[t]]
                ax.bar(k, whi - wlo, bottom=wlo, width=0.62, color="#efefec", lw=0, zorder=0)
                ax.bar(k, hi - lo, bottom=lo, width=0.62, color="#dcdcd8", lw=0, zorder=0)
            ax.plot([k, k], [r["q25"] * scale, r["q75"] * scale], color=col(t), lw=1.0, alpha=0.55, solid_capstyle="butt")
            ax.plot(k, r["median"] * scale, "o", ms=4.2, color=col(t), mec="white", mew=0.5, zorder=3)
            v = rel.get(t, {}).get(key)
            if v is not None and not np.isclose(v, r["median"], rtol=1e-6):
                ax.plot(k, v * scale, "x", ms=5.0, color=INK, mew=1.0, zorder=4); diff.append((t, key))
            if key == "vmax":
                ax.plot([k - 0.32, k + 0.32], [f77[parent[t] - 1]] * 2, color="#7c7c78", lw=1.0, zorder=2)
                ax.text(k, 1.5, f"{int(r['n'])}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_ylabel(ylab, fontsize=7.6); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); _spines(ax)
        ax.tick_params(labelsize=6.8)
    axs[0].set_ylim(0, 100)
    axs[0].text(-0.6, 7.0, "towers:", ha="left", va="bottom", fontsize=6.5, color=INK2)
    axs[1].axhline(0.30, color="#7c7c78", lw=1.0)
    for b in (0.10, 0.50):                                      # the search bounds of rgrowth (Table 16.3)
        axs[1].axhline(b, color="#9a9a96", lw=0.8, ls=(0, (1.5, 2)))
    axs[1].set_ylim(0.0, 0.55)
    axs[1].set_xticks(x); axs[1].set_xticklabels(ids, rotation=90, fontsize=6.6)
    axs[1].set_xlim(-0.9, len(ids) - 0.1)
    h = [Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=5, label="tree types: median"),
         Line2D([], [], marker="o", ls="", color=ORANGE, mec="white", ms=5, label="shrub types: median"),
         Line2D([], [], marker="o", ls="", color=AQUA, mec="white", ms=5, label="grass types: median"),
         Line2D([], [], marker="|", ls="", color="#9a9a96", ms=9, mew=1.2, label="inter-quartile range of the towers"),
         Line2D([], [], marker="x", ls="", color=INK, ms=5, mew=1.0, label="release value, where not the median"),
         Line2D([], [], color="#7c7c78", lw=1.0, label="IBIS 2.6b4 (parent PFT; 0.30 for all)"),
         Patch(fc="#dcdcd8", ec="none", label="bounds of the parent position (Table 16.3)"),
         Patch(fc="#efefec", ec="none", label="widened bounds (bound-relaxed refit)"),
         Line2D([], [], color="#9a9a96", lw=0.8, ls=(0, (1.5, 2)), label="search bounds of (b)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, fontsize=6.6,
               handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(top=0.88)
    axs[0].set_title("(a) photosynthetic capacity", loc="left", fontsize=8.2, weight="bold", color=INK)
    axs[1].set_title("(b) growth respiration", loc="left", fontsize=8.2, weight="bold", color=INK)
    fig.savefig(OUT / "fig_16_3_type_values.png")
    plt.close(fig)
    print("16.3 release values that are not the tower median:", diff)
    at = []
    for t in ids:
        r = tab[(tab.type == t) & (tab.key == "vmax")]
        if r.empty:
            continue
        v = float(r.iloc[0]["median"]) * 1e6; lo, hi, wlo, whi = bounds[parent[t]]
        if v >= 0.99 * whi or v <= 1.01 * wlo: at.append(f"{t} {v:.1f} at the widened bound ({wlo:.1f}-{whi:.1f})")
        elif v > hi or v < lo: at.append(f"{t} {v:.1f} outside the TRY range ({lo:.1f}-{hi:.1f})")
    print(f"16.3 C3 factor ratio 25/15 C {r3:.4f}; type capacities at or beyond the bounds of the parent position:", at)


# ---------------------------------------------------------------------------------------------------------------
# Figure 16.4: the fit of the water-limited maximum tree cover by zone group (section 16.8)
# ---------------------------------------------------------------------------------------------------------------

S31N = "/shared/nas2/Hangkai/ibis/verify_outputs/final_s31n/hindcast/out"          # the protocol run the ramps were fitted on
GLOB_REFS = "/shared/nas2/Hangkai/ibis/verify_outputs/diag_work/meeting/glob_refs.npz"   # product tree cover, cropland, zones
RAMP_BINS = [0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]  # as ramp_fit_zones.py


def _ramp_data():
    """The stress of s31n (mean of the annual factors of 2008 and 2010), the product tree cover on the natural share of
    the cell, the land area, the cells used (natural land: cropland share below 0.3) and the zone of every cell."""
    import xarray as xr
    from landmask import true_landfrac
    ref = np.load(GLOB_REFS)
    zone, tree, crop = ref["zone"], ref["tree"], np.nan_to_num(ref["crop"])
    lat = np.asarray(ref["lat"], float)
    u = np.mean([np.asarray(xr.open_dataset(f"{S31N}/ibis_yearly_{y}.nc")["aystressu"].values, float) for y in (2008, 2010)], 0)
    area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat))[:, None] * true_landfrac()
    tree_nat = np.clip(tree / np.clip(1 - crop, 0.05, 1), 0, 1)
    ok = np.isfinite(u) & np.isfinite(tree) & (area > 0) & (crop < 0.3)
    return u, tree_nat, area, ok, zone


def _ramp_bins(u, cover, area, m):
    """Area-weighted mean stress, median and 75th percentile of the cover, and area, of each stress bin with >= 20 cells
    (verify/calibration/diagnostics/gate_2026-10-27/ramp_fit_zones.py)."""
    def pct(q, mm):
        xs, ws = cover[mm], area[mm]; o = np.argsort(xs); c = np.cumsum(ws[o]) / ws[o].sum()
        return float(xs[o][np.searchsorted(c, q / 100)])
    rows = []
    for lo, hi in zip(RAMP_BINS[:-1], RAMP_BINS[1:]):
        mm = m & (u >= lo) & (u < hi)
        if mm.sum() < 20:
            continue
        rows.append((float((u[mm] * area[mm]).sum() / area[mm].sum()), pct(50, mm), pct(75, mm), float(area[mm].sum())))
    return np.array(rows)


def _ramp_fit(x, y, w):
    """Area-weighted least squares on the grid of ramp_fit_zones.py (equation 16.12)."""
    best = None
    for a in np.arange(0.2, 0.8, 0.02):
        for b in np.arange(0.1, 0.7, 0.02):
            for pw in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
                e = float(np.sum(w * (np.clip((x - a) / b, 0, 1) ** pw - y) ** 2) / w.sum())
                if best is None or e < best[0]:
                    best = (e, a, b, pw)
    return best


def _ramp_near(x, y, w, tol=0.005):
    """The points of the same grid whose area-weighted rms error is below `tol`: how tightly the bins fix the ramp."""
    good = [(a, b, pw) for a in np.arange(0.2, 0.8, 0.02) for b in np.arange(0.1, 0.7, 0.02)
            for pw in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
            if np.sqrt(np.sum(w * (np.clip((x - a) / b, 0, 1) ** pw - y) ** 2) / w.sum()) < tol]
    return len(good), (min(g[0] for g in good), max(g[0] for g in good)) if good else None


def _bcold_bins(u, tree_nat, area, ok, zone):
    """The bins of B_cold alone (section 16.8): the product's median cover is 0 in every bin, its 75th percentile at most
    0.01, and no bin above 0.75 holds 20 cells."""
    from ibis.io import pft_types as P
    B = _ramp_bins(u, tree_nat, area, ok & (zone == P.ZONES.index("B_cold")))
    print("16.8 B_cold alone (mean stress, median cover, 75th percentile, share of the zone's binned area):",
          [(round(r[0], 2), round(r[1], 2), round(r[2], 2), round(r[3] / B[:, 3].sum(), 2)) for r in B])
    return B


def fig_16_4():
    from ibis.io import pft_types as P
    from ibis.io.physics_params import build_all_params, load_physics_params
    u, tree_nat, area, ok, zone = _ramp_data()
    zi = lambda *nm: np.isin(zone, [P.ZONES.index(n) for n in nm])
    # the release ramp of each zone, resolved by the model from the release file (apply_types, equation 2.2)
    zones = [z for z in P.ZONES if z != "none"]
    pdict = load_physics_params(str(REL))
    sp, pft, _ = build_all_params(len(zones), np.zeros(len(zones), np.float32), pdict)
    sp, pft = P.apply_types(sp, pft, np.array([P.ZONES.index(z) for z in zones], np.int32),
                            P.parse_type_overrides(pdict.get("types")), base=(sp, pft))
    rel = {z: (float(np.asarray(pft.tree_cover_min)[k]), float(np.asarray(pft.tree_cover_ramp)[k]),
               float(np.asarray(pft.tree_cover_power)[k])) for k, z in enumerate(zones)}
    base = (pdict["pft"]["tree_cover_min"], pdict["pft"]["tree_cover_ramp"], pdict["pft"]["tree_cover_power"])
    panels = [("(a) A_humid", ("A_humid",), "A_humid", 50),
              ("(b) A_savanna", ("A_savanna",), "A_savanna", 50),
              ("(c) C_med, C_subtrop, C_highland, C_oceanic", ("C_med", "C_subtrop", "C_highland", "C_oceanic"), "C_med", 50),
              ("(d) D_temperate", ("D_temperate",), "D_temperate", 50),
              ("(e) B_cold (fitted on the cold zones)", ("D_temperate", "D_subarctic", "D_extreme", "B_cold", "E_tundra"), "B_cold", 75),
              ("(f) B_hot (base ramp)", ("B_hot",), "B_hot", None)]
    fig, axs = plt.subplots(2, 3, figsize=(8.6, 5.4), sharex=True, sharey=True, gridspec_kw={"hspace": 0.34, "wspace": 0.10})
    uu = np.linspace(0, 1, 201)
    for ax, (title, zs, zrel, q) in zip(axs.ravel(), panels, ):
        m = ok & zi(*zs)
        B = _ramp_bins(u, tree_nat, area, m)
        s = 8 + 220 * B[:, 3] / B[:, 3].sum()                   # dot area grows with the bin's share of the panel's area
        ax.scatter(B[:, 0], B[:, 1], s=s, color=BLUE, alpha=0.85, lw=0, label="product cover: median of the bin", zorder=3,
                   clip_on=False)
        ax.scatter(B[:, 0], B[:, 2], s=s, facecolor="none", edgecolor=ORANGE, lw=0.9, label="product cover: 75th percentile",
                   zorder=3, clip_on=False)
        a, b, pw = rel[zrel]
        ax.plot(uu, np.clip((uu - a) / b, 0, 1) ** pw, color=INK, lw=1.4, label="release ramp of the zone", zorder=2)
        if not np.allclose((a, b, pw), base):
            ax.plot(uu, np.clip((uu - base[0]) / base[1], 0, 1) ** base[2], color="#9a9a96", lw=1.0, ls=(0, (4, 2)),
                    label="base ramp", zorder=1)
        txt = f"$u_0$ {a:.2f}, $\\Delta u$ {b:.2f}, $p$ {pw:g}"
        if q is not None:
            yq = B[:, 1] if q == 50 else B[:, 2]
            e, fa, fb, fp = _ramp_fit(B[:, 0], yq, B[:, 3])
            nn, u0r = _ramp_near(B[:, 0], yq, B[:, 3])
            print(f"16.4 {title}: refit to the {'median' if q == 50 else '75th percentile'}: u0 {fa:.2f} du {fb:.2f} p {fp} "
                  f"(rms {np.sqrt(e):.3f}); release {a:.2f} {b:.2f} {pw}; bins {len(B)}; cells {int(m.sum())}; "
                  f"lowest bin at {B[:, 0].min():.2f}, its value {yq[np.argmin(B[:, 0])]:.2f}; grid points with rms < 0.005: {nn}"
                  + (f" (u0 {u0r[0]:.2f}-{u0r[1]:.2f})" if u0r else ""))
        else:
            print(f"16.4 {title}: base ramp {a:.2f} {b:.2f} {pw}; bins {len(B)}; cells {int(m.sum())}")
        ax.text(0.03, 0.96, txt, transform=ax.transAxes, fontsize=6.8, color=INK2, va="top")
        ax.set_title(title, loc="left", fontsize=7.4, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.03)
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); _spines(ax); ax.tick_params(labelsize=6.8)
    for ax in axs[1]:
        ax.set_xlabel("model stress factor $\\bar u$ (2008 and 2010)", fontsize=7.2)
    for ax in axs[:, 0]:
        ax.set_ylabel("tree cover of the natural land", fontsize=7.2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 1.0), markerscale=0.8)
    fig.subplots_adjust(top=0.88)
    fig.savefig(OUT / "fig_16_4_cover_ramp_fit.png")
    plt.close(fig)
    _bcold_bins(u, tree_nat, area, ok, zone)


# ---------------------------------------------------------------------------------------------------------------
# The numbers of the chapter that were computed for it (not a figure; `make_ch16.py numbers`)
# ---------------------------------------------------------------------------------------------------------------

def numbers():
    import pandas as pd
    import xarray as xr
    rel = _yaml(REL)
    # --- section 16.6.1: the towers of the campaign, from the site table of towers31_prep.py
    si = pd.read_csv(f"{TOWERS31}/FLUXNET_siteinfo.csv", encoding="utf-8-sig")
    cal = si[(si.HH == "yes") & (si.WIN_SCORE.fillna(0) >= 2)]
    nat = cal[cal.IGBP.isin(NATURAL)]
    wl = (nat.WIN_Y1 - nat.WIN_Y0 + 1).astype(int)
    print(f"16.6.1 site table: {len(si)} towers, {len(cal)} calibration-grade, {len(nat)} in natural vegetation: "
          f"{nat.IGBP.value_counts().to_dict()}; sources {nat.SOURCE.value_counts().to_dict()}; window length "
          f"{wl.value_counts().sort_index().to_dict()}; windows within {int(nat.WIN_Y0.min())}-{int(nat.WIN_Y1.max())}")
    n_s2_all, n_s2 = _fitted("t31_s2"); _, n_open = _fitted("t31_open_lpj"); _, n_relax = _fitted("t31_relax"); _, n_nee = _fitted("t31_nee")
    print(f"16.6.1 fit records with parameters: S2 {n_s2} of {n_s2_all}, open-vegetation refit {n_open}, bound-relaxed refit "
          f"{n_relax}, NEE {n_nee}; methane towers {_towers_fitted('ch4_h1', 'ch4_h1_gpu', 'ch4_h1_fix')}; "
          f"cropland towers {_towers('crop2', 'crop2_gpu')}")
    # --- section 16.4: fits whose caps were lifted because the starting loss lay at the tower cap (site_fit --robust-auto)
    for st in ("t31_s2", "t31_open_lpj", "t31_relax"):
        logs = glob.glob(f"{SITE_FIT}/{st}/*/run.log")
        lifted = sum("robust-auto: baseline at the tower cap" in open(f, errors="replace").read() for f in logs)
        print(f"16.4 {st}: caps lifted in {lifted} of {len(logs)} fits")
    # --- section 16.6 and 16.6.1: the reduction to types, rebuilt by towers31_types_yaml.py
    tab = _type_table(); T = rel["types"]
    eq = sum(bool(np.isclose(T.get(r.type, {}).get(r.key, np.nan), r.median, rtol=1e-6)) for r in tab.itertuples())
    n = tab.groupby("type")["n"].max()
    print(f"16.6.1 reduction: {len(tab)} type values, {eq} equal to the release; towers per type (largest over its "
          f"parameters) {n.to_dict()}; one tower {sorted(n[n == 1].index)}; three or more {int((n >= 3).sum())} of {len(n)}")
    rg_t = tab[tab.key == "rgrowth"]
    print(f"16.14 type medians of rgrowth within 0.005 of the lower bound 0.10: "
          f"{[(r.type, round(r.median, 3), int(r.n)) for r in rg_t.itertuples() if r.median < 0.105]}; "
          f"types with the lower quartile within 0.005 of it: {int((rg_t.q25 < 0.105).sum())} of {len(rg_t)}")
    # --- section 16.6.2: the 20 October set (params_31a) against the release, tower-derived type values
    a = _yaml(P31A)["types"]
    d = [(t, k) for t in a for k, x in a[t].items() if k in T.get(t, {}) and not np.isclose(x, T[t][k])]
    print(f"16.6.2 type values of params_31a.yaml that differ from the release: {len(d)}; by parameter "
          f"{dict(collections.Counter(k for _, k in d))}")
    # --- Table 16.5: medians over the towers of the per-tower scores of compare_sets.py
    p = pd.read_csv(CMP_TOWERS)
    print(f"Table 16.5 per-tower file: {len(p)} towers; median tower NEE over the window {p['nee_obs'].median():.0f} gC m-2 yr-1")
    for s, nm in (("own", "own fits"), ("deck_pools", "shared set, 20 October"), ("B2", "shared set, 26 October")):
        out = []
        for k in ("gpp_ratio", "et_ratio", "reco_ratio", "gpp_r", "et_r", "nee_r", "nee_bias"):
            x = p[f"{s}_{k}"].dropna()
            out.append(f"{k} n {len(x)} {x.median():.2f}" + (f" ({100 * ((x >= 0.75) & (x <= 1.25)).mean():.0f} % within 0.75-1.25)"
                                                           if k.endswith("ratio") else ""))
        print(f"Table 16.5 {nm}: " + "; ".join(out))
    # --- section 16.5: the capacity bounds at 15 C and the C4 factor
    bounds, r3, r4 = _vmax_bounds15()
    print(f"16.5 ratio 25 C / 15 C of the C3 factor {r3:.4f} (used in the campaign), of the C4 factor {r4:.4f}; C4 bounds "
          f"10-75 at 25 C = {10 / r3:.1f}-{75 / r3:.1f} at 15 C with the C3 factor, {10 / r4:.1f}-{75 / r4:.1f} with the C4 factor")
    # the current tool's conversion (site_fit.vmax_bounds_15c), on the campaign's base set
    import site_fit as SF
    from towers31_cmds import VMAX_BOUNDS
    from ibis.io.physics_params import build_all_params, load_physics_params
    sp, _, _ = build_all_params(1, np.zeros(1, np.float32), load_physics_params(str(BASE_S2)))
    lo, hi = SF.vmax_bounds_15c(VMAX_BOUNDS, sp.stomata_params, np.zeros(12), np.zeros(12))
    c4 = SF.PFT_NAMES.index("c4_grass")
    print(f"16.5 site_fit.vmax_bounds_15c now: C4 position {lo[c4] * 1e6:.1f}-{hi[c4] * 1e6:.1f} at 15 C; the C3 positions as in "
          f"the campaign: {all(np.isclose(lo[k] * 1e6, bounds[k + 1][0], rtol=1e-4) for k in range(12) if k != c4)}")
    print("16.5 bounds at 15 C by position (TRY lo, hi; widened lo, hi):", {k: tuple(round(x, 1) for x in v) for k, v in bounds.items()})
    # --- section 16.8: the bins of B_cold alone
    _bcold_bins(*_ramp_data())
    # --- section 16.12 and 16.14: the cropland fits
    c3, c4, sw = [], [], {}
    for f in sorted(glob.glob(f"{SITE_FIT}/crop2/*/best.json")):
        s = os.path.basename(os.path.dirname(f)); pr = json.load(open(f)).get("params", {})
        sw[s] = json.load(open(f"{os.path.dirname(f)}/setup.json")).get("hourly_sw")
        if "crop_vmax_scale_c3" in pr: c3.append(pr["crop_vmax_scale_c3"])
        if "crop_vmax_scale_c4" in pr: c4.append(pr["crop_vmax_scale_c4"])
    print(f"16.12 C3 crop scales ({len(c3)} towers) {sorted(round(v, 3) for v in c3)}, median {np.median(c3):.3f}; "
          f"C4 ({len(c4)} towers) median {np.median(c4):.3f}; within 0.003 of the floor {sum(v <= 0.303 for v in c3)}, within 0.05 "
          f"{sum(v <= 0.35 for v in c3)}; fits without the tower's hourly radiation {sorted(s for s, v in sw.items() if not v)}")
    # --- section 16.14: the growth respiration fraction of the NEE fits at its bounds (0.10-0.50)
    rg = np.array([json.load(open(f)).get("params", {}).get("rgrowth", np.nan) for f in glob.glob(f"{SITE_FIT}/t31_nee/*/best.json")])
    rg = rg[np.isfinite(rg)]
    print(f"16.14 NEE fits {rg.size}: rgrowth < 0.101 at {int((rg < 0.101).sum())}, < 0.105 at {int((rg < 0.105).sum())}, "
          f"> 0.495 at {int((rg > 0.495).sum())}")
    # --- section 16.14: coverage of the day/night NEE targets of the campaign (the sum over the kept half-hours against
    # the full month), in the campaign's target file, built before the targets became means (section 16.3)
    z = np.load(NEE_TARGETS)
    yrs = [int(y) for y in z["years"]]
    nh = np.nan_to_num(z["n_night"]) + np.nan_to_num(z["n_day"])        # kept half-hours (counts in half-hour units); a missing part counts 0
    full = np.array([[calendar.monthrange(y, m + 1)[1] * 48 for m in range(12)] for y in yrs], float)[:, :, None]
    has = np.isfinite(z["sum_night"]) | np.isfinite(z["sum_day"])
    cov = np.where(has, nh / full, np.nan); c = cov[np.isfinite(cov)]
    per = np.nanmedian(cov.reshape(-1, cov.shape[2]), axis=0)
    print(f"16.14 NEE targets: {len(z['sites'])} towers, {c.size} tower-months; coverage median {np.median(c):.3f}, mean {c.mean():.3f}; "
          f"below 0.9 in {100 * (c < 0.9).mean():.1f} %, below 0.75 in {100 * (c < 0.75).mean():.1f} % of the tower-months; "
          f"towers with a median coverage below 0.9: {int((per < 0.9).sum())}, below 0.75: {int((per < 0.75).sum())}")
    # --- section 16.11: the methane towers by band and the number with a net uptake (counts only: the data are not public)
    rep = pd.read_csv(f"{CH4_LOGS}/ch4_fit_report_hourly.csv").set_index("site")
    band = pd.read_csv(f"{CH4_LOGS}/ch4_towers_zone.csv").set_index("site")["zone_band"].reindex(rep.index)
    fy = rep["tower_gC_yr"]                                    # the tower's mean annual flux over its fitted years (the targets)
    bud = pd.read_csv(CH4_SITES).set_index("SITE_ID")["budget_gC"].reindex(rep.index)   # the site table's budget (selection rule)
    print(f"16.11 methane towers by band {band.value_counts().to_dict()}; fitted-years mean below zero "
          f"{dict(band[fy < 0].value_counts())}, all under 1 gC m-2 yr-1 in magnitude: {bool((fy[fy < 0] > -1).all())}; "
          f"fitted-years |mean| < 0.5: {int((fy.abs() < 0.5).sum())}; site-table |budget| < 0.5: {int((bud.abs() < 0.5).sum())}, "
          f"budget below zero: {int((bud < 0).sum())}")
    # the band medians the band tool gives (ch4_class_yaml.py --band-csv) with every tower (--keep-all, how the release's
    # sections were built) and with its current rule, the towers with a net uptake over their fitted years left out; band
    # level only (the tool's own output names the towers it leaves out and is not printed)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in (os.environ.get("PYTHONPATH", ""), str(ROOT / "python-port")) if x))
    bands = {}
    with tempfile.TemporaryDirectory() as tmp:
        for tag, extra in (("every tower (--keep-all, the release)", ["--keep-all"]), ("net-uptake towers left out (current)", [])):
            cmd = [sys.executable, str(ROOT / "verify/calibration/ch4_class_yaml.py"), f"{CH4_LOGS}/ch4_fit_report_hourly.csv",
                   str(REL), f"{tmp}/out.yaml", "--band-csv", f"{CH4_LOGS}/ch4_towers_zone.csv"] + extra
            r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit("ch4_class_yaml.py failed (output withheld: it names methane towers)")
            bands[tag] = _yaml(f"{tmp}/out.yaml")["methane_bands"]
    for tag, mb in bands.items():
        print(f"16.11 band medians, {tag}: " + "; ".join(
            f"{b} r {p['r_ch4']:.4f} Q10 {p['q10p']:.2f} z_tau {p['z_tau']:.3f} tau_ox {p['tau_oxid']:.3f}" for b, p in mb.items()))
    relb = _yaml(REL)["methane_bands"]
    print("16.11 --keep-all reproduces the release's bands:", all(np.isclose(relb[b][k], v, rtol=1e-9) for b, p in
                                                                   bands["every tower (--keep-all, the release)"].items() for k, v in p.items()))
    # --- Table 16.11: the release run's budget file and the GPP change from the yearly output
    D = pd.read_csv(f"{REL30}/score/mip_compare.csv"); dd = D[(D.year >= 2001) & (D.year <= 2010)]
    print("16.13 mip_compare.csv 2001-2010 means:", dd[["gpp", "nee", "nbp", "ch4", "ch4_90S-30N", "ch4_30N-60N", "ch4_60N-90N"]].mean().round(2).to_dict())
    from landmask import true_landfrac
    lat = np.arange(89.75, -90, -0.5); cell = np.cos(np.deg2rad(lat))[:, None] * np.ones((360, 720)) * (0.5 * 111.32e3) ** 2
    with xr.open_dataset(SURTA) as sd: binary = (np.nan_to_num(sd["landfrac"].values) > 0).astype(float)
    tl = true_landfrac(); dec = {}
    for y in list(range(1901, 1911)) + list(range(2001, 2011)):
        with xr.open_dataset(f"{REL30}/{'transient' if y <= 2000 else 'hindcast'}/out/ibis_yearly_{y}.nc") as ds:
            g = np.nan_to_num(np.asarray(ds["aygpptot"].values, float))
        for nm, w in (("binary mask", binary), ("land fraction", tl)):
            dec.setdefault((nm, y // 10 * 10 if y % 10 else y - 10), []).append(float(np.sum(g * cell * w)) * 1e-12)
    for nm in ("binary mask", "land fraction"):
        a0, a1 = np.mean(dec[(nm, 1900)]), np.mean(dec[(nm, 2000)])
        print(f"16.13 GPP on the {nm}: 1901-1910 {a0:.1f}, 2001-2010 {a1:.1f} PgC yr-1, change {100 * (a1 / a0 - 1):+.1f} %")


FIGURES = {"16.1": fig_16_1, "16.2": fig_16_2, "16.3": fig_16_3, "16.4": fig_16_4}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        if key == "numbers":
            numbers()
        else:
            FIGURES[key]()
        print(key, "done")
