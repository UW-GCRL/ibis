"""Figures of Chapter 17 of the IBIS 3.0 technical note (numerical implementation and verification against IBIS 2.6b4),
and the numbers the chapter quotes.

Run from the repository root with the project interpreter (CPU):

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch17.py            # all figures
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch17.py 17.2       # one
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch17.py numbers    # the quoted numbers
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch17.py timestep   # section 17.9.3 re-run
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch17.py inject     # section 17.9.3, day 4

Inputs (intermediate files go to CH17_WORK, default the folder named below; they are rebuilt when missing):
  * Figure 17.1 reads the layer counts, the number of canopy iterations, the size of the canopy system and the steps
    per day from the model's code.
  * Figure 17.3 reads the per-year files of the 46-year F77/JAX lockstep
    (/shared/nas2/Hangkai/ibis/verify_outputs/years_1951_2000/per_year_stats, written by
    verify/timestepping/pipeline/global_gswp3_multiyear.py): for every cell, field and year, the largest |F77 - JAX|
    over the year's hourly steps (cached by `lockstep_cache`).
  * Figure 17.2 adds the hourly output of the model's own timestep_body at the same 117 cells for 1951, run in a
    subprocess once in double and once in single precision (`precision_run`), with the lockstep's surface templates,
    initial state, snow and soil constants (section 17.9.1) and forcing disaggregation and the GSWP3 forcing of 1951
    (`forcing_cache`). The harness of the 46-year lockstep (its synthetic soils with the negative suction, its snow
    constants and solids) was replaced on 2026-11-01 by IBIS 2.6b4's own settings (commit bed0ae7), so
    `precision_run` reads the lockstep's harness inputs from the commit before that one (`_lockstep_harness`, read-only
    `git show`). The published figure is the release code's (cached files of 2026-10-31 in the default CH17_WORK); a
    run with another CH17_WORK repeats the experiment on the code of the working tree.
  * `numbers` recomputes the other quantities of the chapter: the lockstep table, range exceedances and cells below
    200 K, the two-stream look-up table, the rounding of sand and clay, the single-precision pressure difference, the
    restart files (spin-up and hindcast) and state leaves (the release configuration, and with physics.water_balance),
    with the leaves that have no cell dimension, the single-against-double numbers of section 17.5 (and, when a second
    single-precision run `prec_f32_repeat.npz` exists, the CPU launch-to-launch comparison of section 17.8), the GPU
    launch-to-launch comparison of 2001 (the output fields, as relative differences of their unweighted sums over the
    land cells, and the end-of-2001 restart files), the two GPU launches with XLA's deterministic reductions
    (`numbers_d83`) and the counts of the water-balance diagnostic year (`numbers_wbal`).
  * `timestep` runs the time-step checks of verify/timestepping/verify_lsxmain.py, as checked in, on the code of the
    working tree (section 17.9.3), then the cold snap again with the bounds of Table 17.4 that act there lifted: needs
    the compiled Fortran (tests/f2py/ibis_f77.so, built after 2026-11-01, and its wrapper), the f2py module of
    verify/fortran_wrappers/canopy and double precision, so it runs in a subprocess with JAX_ENABLE_X64=1 (about 10
    minutes on one CPU). GCOV_PREFIX sends the coverage files of the canopy module to CH17_WORK/gcov.
  * `selfpert` and `inject` (section 17.9.3, the events on the fourth day of the 7-day diurnal and monsoon checks):
    the port against itself started 1e-12 K apart, and the port started from the Fortran's state after step 80
    (`_inject_f77_state`); same requirements as `timestep`.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: E402  (also sets rcParams)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

PER_YEAR = "/shared/nas2/Hangkai/ibis/verify_outputs/years_1951_2000/per_year_stats"
WORK = Path(os.environ.get("CH17_WORK",
                           "/tmp/claude-1007/-home-hangkai/28719488-0896-443d-8709-849a94fae0be/scratchpad/technote_wf/ch17"))
WORK.mkdir(parents=True, exist_ok=True)
LOCKSTEP_NPZ = WORK / "lockstep_46yr.npz"
FORCING_NPZ = WORK / "forcing_1951.npz"
PRECISION_DIR = WORK
GSWP3 = "/shared/nas2/Min/GSWP3"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------------------------------------
# Figure 17.1: program structure and the solves of one time step
# ---------------------------------------------------------------------------------------------------------------

def _code_counts():
    """The counts the schematic prints, read from the code."""
    from ibis.io.soil_init import HSOI_LAYERS
    from ibis.physics.bedrock import BedrockParams
    from ibis.dimensions import GridConfig
    from ibis.timestepping.calendar import timesteps_per_day
    src_loops = (ROOT / "python-port/ibis/timestepping/loops.py").read_text()
    src_can = (ROOT / "python-port/ibis/physics/canopy_exchange.py").read_text()
    src_int = (ROOT / "python-port/ibis/physics/interception.py").read_text()
    niter = int(re.search(r"for iter_num in range\((\d+)\)", src_loops).group(1))
    nd = int(re.search(r"arr = jnp\.zeros\(\(npoi, (\d+), \d+\)\)", src_can).group(1))
    ntwet = int(re.search(r"def twet3[\s\S]*?for _ in range\((\d+)\)", src_int).group(1))
    # the wet-bulb iteration of the interception step (cascade): relaxation factor and tolerance of twet3
    relax = re.search(r"def twet3[\s\S]*?twk_new = twold \+ ([0-9.]+) \* diff", src_int).group(1)
    tol = re.search(r"def twet3[\s\S]*?will_converge = jnp\.abs\([^)]*\) < ([0-9.]+)", src_int).group(1)
    return dict(nsoil=len(HSOI_LAYERS), nrock=len(BedrockParams().hsoi), nsnow=GridConfig().nsnolay,
                nsteps=int(timesteps_per_day(3600.0)), niter=niter, nd=nd, ntwet=ntwet, relax=relax, tol=tol)


def _box(ax, x, y, w, h, fc, ec="none", lw=0.0, r=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def _arrow(ax, p0, p1, color=INK2, lw=0.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw))


def fig_17_1():
    k = _code_counts()
    W_IN, H_IN = 8.6, 9.6                      # one axis unit = 0.086 in in both directions
    ymax = 100.0 * H_IN / W_IN
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    ax.set_xlim(0, 100); ax.set_ylim(0, ymax); ax.axis("off")
    # --- (a) driver and compiled month ---
    ax.text(0, ymax - 0.5, "(a) the program", fontsize=9, weight="bold", color=INK, va="top")
    top = ymax - 5.0; bot = top - 45.0
    _box(ax, 0, bot, 42, top - bot, "#f3f3f1")
    ax.text(1.5, top - 1.6, "driver: Python, run operation by operation", fontsize=8, weight="bold", color=INK,
            va="top")
    drv = [                                  # run_global.py, in its order
        (top - 7.0, "run start", "configuration, parameter file,\ninput maps; cold start or\nrestart file; random key 42"),
        (top - 16.0, "each year", "Köppen-period types; fire counters\nand climate statistics reset; CO$_2$"),
        (top - 23.5, "each month", "read the month's forcing;\ncall the compiled month;\n"
                                   "methane month end; learned\nwetland area; monthly output"),
        (top - 34.0, "year end", "climate means (30-year memory);\nexistence; vegetation update;\n"
                                 "yearly output; accumulators\nreset; restart file"),
    ]
    for yy, head, body in drv:
        ax.text(1.5, yy, head, fontsize=7.6, weight="bold", color=INK2, va="top")
        ax.text(15.0, yy, body, fontsize=7.2, color=INK2, va="top", linespacing=1.3)
    _box(ax, 46, bot, 54, top - bot, "#e8f0fa")
    ax.text(47.5, top - 1.6, "the compiled month: one XLA program", fontsize=8, weight="bold", color=INK, va="top")
    ax.text(47.5, top - 5.2, "scan over the days of the month (28–31 days)", fontsize=7.2, color=INK2, va="top")
    _box(ax, 48, bot + 2.0, 50.5, top - bot - 10.5, "#d6e5f6")
    day = [
        (top - 12.0, "day start", "seasonal states; leaf phenology;\nprecipitation event (key split)"),
        (top - 20.5, f"{k['nsteps']} steps", f"scan over the {k['nsteps']} hourly steps: weather\n"
                                             "generator at mid-step (key split), the\ntime step of panel (b); "
                                             "NPP, GPP, ET, CH$_4$ sums"),
        (top - 30.0, "day end", "10-day means; the year's climate\nstatistics; decomposition factors;\n"
                                "litter and soil carbon × nsoilcall;\nsoil-water stress sums"),
    ]
    for yy, head, body in day:
        ax.text(49.5, yy, head, fontsize=7.6, weight="bold", color=INK, va="top")
        ax.text(62.0, yy, body, fontsize=7.2, color=INK2, va="top", linespacing=1.3)
    ym = 0.5 * (top + bot)
    _arrow(ax, (42.3, ym + 2.0), (45.7, ym + 2.0))
    _arrow(ax, (45.7, ym - 2.0), (42.3, ym - 2.0))
    ax.text(44.0, ym + 3.4, "state", fontsize=6.8, color=INK2, ha="center")
    # --- (b) the time step: the iterative and implicit solutions, labelled with their step of Table 1.3 ---
    ax.text(0, bot - 3.0, "(b) one time step (Δt = 3600 s): its iterative and implicit solutions, by step of "
            "Table 1.3", fontsize=9, weight="bold", color=INK, va="top")
    steps = [
        ("4  interception", "temperature of the rain above the canopy: the wet-bulb temperature of the air (where the "
                            "air is above the\nmelting point), by a fixed-point iteration relaxed by "
                            f"{k['relax']}\n(at most {k['ntwet']} passes; each cell stops at |Δ| < {k['tol']} K)",
         "#eaf4e4", 3),
        ("6  canopy", f"{k['niter']} fixed passes of: transfer coefficients → stomatal conductance and photosynthesis →\n"
                      f"{k['nd']} × {k['nd']} linear system (7 canopy and canopy-air unknowns, soil surface, snow "
                      "surface)\nby Gaussian elimination without pivoting", "#e8f0fa", 3),
        ("7  canopy water", "evaporation of intercepted water and final heat adjustment of the canopy water "
                            "(no iteration)", "#eaf4e4", 1),
        ("9  snow", f"heat conduction in {k['nsnow']} snow layers and the buried lower canopy: implicit, tridiagonal\n"
                    f"({k['nsnow'] + 1} unknowns, Thomas algorithm); melt, layer adjustment, cover", "#eef3f9", 2),
        ("10  soil water", f"water flow in {k['nsoil']} layers: fluxes linearized about the start-of-step state, "
                           f"implicit,\ntridiagonal ({k['nsoil']} unknowns, Thomas algorithm)", "#f6efe0", 2),
        ("10  soil heat", f"conduction in {k['nsoil']} soil + {k['nrock']} bedrock layers: implicit, tridiagonal "
                          f"({k['nsoil'] + k['nrock']} unknowns);\nfreezing and thawing; water-content bounds; "
                          "surface temperatures bounded (section 17.4)", "#f3e6e6", 2),
    ]
    gap = 1.8; y = bot - 7.5
    for i, (head, body, fc, nlines) in enumerate(steps):
        h = 2.6 + 2.0 * nlines
        _box(ax, 0, y - h, 100, h, fc)
        ax.text(1.5, y - 1.2, head, fontsize=7.8, weight="bold", color=INK, va="top")
        ax.text(19.0, y - 1.2, body, fontsize=7.2, color=INK2, va="top", linespacing=1.3)
        if i < len(steps) - 1:
            _arrow(ax, (8.0, y - h - 0.1), (8.0, y - h - gap + 0.1))
        y -= h + gap
    fig.savefig(OUT / "fig_17_1_program_structure.png")
    plt.close(fig)
    print("17.1", k)


# ---------------------------------------------------------------------------------------------------------------
# Figure 17.3: the 46-year lockstep against IBIS 2.6b4
# ---------------------------------------------------------------------------------------------------------------

def lockstep_cache():
    """Reduce the lockstep's per-year files to an array worst[field, cell, year] (largest hourly |F77 - JAX|)."""
    files = sorted(glob.glob(PER_YEAR + "/*.json"))
    worst = {}
    for f in files:
        d = json.load(open(f))
        for kk, v in d["worst"].items():
            c, fld = kk.split("|")
            worst[(c, fld, int(d["year"]))] = v
    cells = sorted({kk[0] for kk in worst}); years = sorted({kk[2] for kk in worst}); fields = sorted({kk[1] for kk in worst})
    parse = lambda c: [float(x.replace("p", ".")) for x in c[1:].split("_")]
    arr = np.full((len(fields), len(cells), len(years)), np.nan)
    for (c, fld, y), v in worst.items():
        arr[fields.index(fld), cells.index(c), years.index(y)] = v
    np.savez(LOCKSTEP_NPZ, fields=np.array(fields), cells=np.array(cells), years=np.array(years),
             lat=np.array([parse(c)[0] for c in cells]), lon=np.array([parse(c)[1] for c in cells]), worst=arr)


def _lockstep():
    if not LOCKSTEP_NPZ.exists():
        lockstep_cache()
    z = np.load(LOCKSTEP_NPZ)
    return z, list(z["fields"])


def fig_17_3():
    z, fields = _lockstep()
    W = z["worst"]; years = z["years"]; lat = z["lat"]; lon = z["lon"]
    tu = W[fields.index("canopy.tu")]; ts = W[fields.index("soil.tsoi[0]")]
    # the cell's position: the GSWP3 cell the lockstep read (latlon_to_grid_idx of the sample point)
    jj = np.array([max(0, min(359, int(round((89.75 - a) / 0.5)))) for a in lat])
    ii = np.array([max(0, min(719, int(round((b + 179.75) / 0.5)))) for b in lon])
    clat = 89.75 - 0.5 * jj; clon = -179.75 + 0.5 * ii
    land = _land_mask()
    fig = plt.figure(figsize=(7.4, 7.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax.imshow(np.where(land, 0.0, np.nan), extent=(-180, 180, -90, 90), cmap=ListedColormap([LAND_BG]),
              interpolation="nearest")
    _coast(ax, land); _map_axes(ax)
    cmax = np.nanmax(tu, axis=1)
    # five classes that the data populate; the first and the last are open (the smallest cell maximum is ~5e-13 K,
    # the rounding level of double precision; the largest 0.45 K). One blue ramp (steps 300-700 of the dataviz
    # reference ramp), its light end 2.2:1 against the land background so that every class reads on the map.
    edges = [1e-13, 1e-6, 1e-3, 1e-2, 1e-1, 1.0]
    names = ["< 1 µK", "1 µK – 1 mK", "1–10 mK", "10–100 mK", "≥ 0.1 K"]
    ramp = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = ListedColormap(ramp); norm = BoundaryNorm(edges, cmap.N)
    sc = ax.scatter(clon, clat, c=np.clip(cmax, 1.01e-13, 0.99), cmap=cmap, norm=norm, s=22, edgecolors=INK2,
                    linewidths=0.3, zorder=3, clip_on=False)
    mids = [0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])]   # centre of each (uniformly drawn) class
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09, aspect=40, ticks=mids)
    cb.ax.set_xticklabels(names); cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=7, length=0); cb.outline.set_linewidth(0.4)
    cb.set_label("largest hourly |IBIS 2.6b4 − IBIS 3.0| of the upper-canopy temperature, 1951–1996", fontsize=8)
    ax.set_title(f"(a) the {len(lat)} cells of the lockstep", loc="left", fontsize=9, weight="bold", color=INK)
    ax = fig.add_subplot(gs[1])
    series = [(tu, "upper-canopy temperature", BLUE), (ts, "top soil-layer temperature", ORANGE)]
    for a, lab, col in series:
        ax.plot(years, np.nanmedian(a, axis=0), color=col, lw=1.4, label=f"{lab}, median cell")
        ax.plot(years, np.nanpercentile(a, 99, axis=0), color=col, lw=1.0, ls=(0, (4, 2)),
                label=f"{lab}, 99th percentile")
    ax.set_yscale("log"); ax.set_ylim(1e-8, 3.0); ax.set_xlim(years[0], years[-1])
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1.0]); ax.set_yticklabels(["10 nK", "1 µK", "0.1 mK", "10 mK", "1 K"])
    ax.set_ylabel("largest hourly |IBIS 2.6b4 − IBIS 3.0|\nin the year")
    ax.set_xlabel("year")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left", handlelength=2.2, columnspacing=1.2)
    ax.set_title("(b) each year: the median cell and the 99th percentile over the cells", loc="left", fontsize=9,
                 weight="bold", color=INK)
    _clean(ax)
    fig.savefig(OUT / "fig_17_3_lockstep_46yr.png")
    plt.close(fig)
    q = lambda a: np.nanpercentile(a, [50, 95, 99, 100])
    print("17.3 cells", len(lat), "years", years[0], years[-1], "tu cell-year q50/95/99/max", q(tu), "tsoi", q(ts),
          "cells with 46-yr max tu < 1 mK", int((cmax < 1e-3).sum()), ">= 0.1 K", int((cmax >= 0.1).sum()),
          "; cells per class", dict(zip(names, np.histogram(cmax, bins=edges)[0].tolist())),
          "; smallest and largest cell maximum", float(cmax.min()), float(cmax.max()))


# ---------------------------------------------------------------------------------------------------------------
# Figure 17.2: single against double precision, beside the F77 difference. The precision experiment: the lockstep's
# set-up, run by the model's timestep_body in double and in single precision
# ---------------------------------------------------------------------------------------------------------------

def _lockstep_grid_index(lat, lon):
    """The GSWP3 cell the lockstep read for a sample point (verify/.../global_gswp3_test.py latlon_to_grid_idx)."""
    j = np.array([max(0, min(359, int(round((89.75 - a) / 0.5)))) for a in lat])
    i = np.array([max(0, min(719, int(round((b + 179.75) / 0.5)))) for b in lon])
    return j, i


def forcing_cache(year=1951):
    """The GSWP3 daily forcing of `year` at the lockstep's 117 cells, read as the lockstep reads it."""
    from netCDF4 import Dataset
    z, _ = _lockstep()
    lat, lon = z["lat"], z["lon"]
    j, i = _lockstep_grid_index(lat, lon)
    tmpl = np.array([0 if abs(a) < 15 else (1 if abs(a) > 50 else (3 if abs(a) < 25 else 2)) for a in lat])
    res = dict(lat=lat, lon=lon, template=tmpl, year=year, j=j, i=i)
    dec = 1901 + ((year - 1901) // 10) * 10
    for v in ("tas", "pr", "rsds", "rlds", "huss", "wind", "ps"):
        ds = Dataset(f"{GSWP3}/{v}_gswp3_{dec}_{dec + 9}.nc4")
        var = ds.variables[v]
        for m in range(1, 13):
            s0 = date(year, m, 1); e0 = date(year + 1, 1, 1) if m == 12 else date(year, m + 1, 1)
            t0 = (s0 - date(dec, 1, 1)).days; nd = (e0 - s0).days
            g = var[t0:t0 + nd, :, :]
            g = np.asarray(g.filled(np.nan) if hasattr(g, "filled") else g, dtype=np.float64)
            res[f"{v}_{m}"] = g[:, j, i]; res[f"ndays_{m}"] = nd
        ds.close()
    np.savez(FORCING_NPZ, **res)


def _disaggregate(day, hour, jday, lat_deg):
    """The lockstep's disaggregation of the daily forcing (verify/.../global_gswp3_test.py diurnal_disaggregate)."""
    orbit = 2.0 * np.pi * jday / 365.2425
    decl = (0.006918 - 0.399912 * np.cos(orbit) + 0.070257 * np.sin(orbit) - 0.006758 * np.cos(2 * orbit)
            + 0.000907 * np.sin(2 * orbit) - 0.002697 * np.cos(3 * orbit) + 0.001480 * np.sin(3 * orbit))
    lr = np.radians(lat_deg); ha = (hour - 12.0) * (2 * np.pi / 24.0)
    cz = np.maximum(0.0, np.sin(lr) * np.sin(decl) + np.cos(lr) * np.cos(decl) * np.cos(ha))
    h0 = np.arccos(np.clip(-np.tan(lr) * np.tan(decl), -1.0, 1.0))
    mcz = np.maximum((h0 * np.sin(lr) * np.sin(decl) + np.cos(lr) * np.cos(decl) * np.sin(h0)) / np.pi, 1e-8)
    sw = np.maximum(0.0, day["rsds"] * cz / mcz); dfr = 0.3 + 0.4 * (1.0 - cz)
    solad = np.stack([0.5 * sw * (1 - dfr)] * 2, 1); solai = np.stack([0.5 * sw * dfr] * 2, 1)
    ta = day["tas"] + 5.0 * np.sin((hour - 8.0) * 2 * np.pi / 24.0)
    return dict(ta=ta, qa=np.maximum(day["huss"], 1e-4), ua=np.maximum(day["wind"], 0.5), psurf=day["ps"], coszen=cz,
                fira=day["rlds"], solad=solad, solai=solai, raina=np.where(ta >= 273.16, day["pr"], 0.0),
                snowa=np.where(ta < 273.16, day["pr"], 0.0))


PREC_FIELDS = ("canopy.tu", "canopy.tl", "soil.tg", "soil.tsoi[0]", "soil.wsoi[0]", "flux.fseng", "flux.fsenu",
               "snow.tsno[0]")

#: The harness inputs of the 46-year lockstep: python-port/tests/f2py/synthetic_inputs.py as it stood from the initial
#: commit (May 2026) until commit bed0ae7 (2026-11-01, IBIS 2.6b4's own settings in the harness), i.e. at bed0ae7's parent.
LOCKSTEP_HARNESS_REV = "bed0ae7~1"


def _lockstep_harness():
    """The module of the lockstep's harness inputs (synthetic soils with the negative suction, the snow and soil values
    the 46-year run gave both codes), read from git without touching the working tree and imported from CH17_WORK."""
    import importlib.util
    path = WORK / "synthetic_inputs_lockstep.py"
    if not path.exists():
        src = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(ROOT), "show",
                              f"{LOCKSTEP_HARNESS_REV}:python-port/tests/f2py/synthetic_inputs.py"],
                             check=True, capture_output=True, text=True).stdout
        path.write_text(src)
    spec = importlib.util.spec_from_file_location("synthetic_inputs_lockstep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def precision_run(out):
    """One year of timestep_body at the lockstep's cells, in the precision of this process (JAX_ENABLE_X64)."""
    import jax, jax.numpy as jnp, equinox as eqx
    _h = _lockstep_harness()
    make_soil_inputs, make_snow_inputs = _h.make_soil_inputs, _h.make_snow_inputs
    make_atm_inputs, make_veg_inputs = _h.make_atm_inputs, _h.make_veg_inputs
    from ibis.constants import get_constants
    from ibis.dimensions import GridConfig
    from ibis.state.model_state import create_ibis_state
    from ibis.timestepping.loops import default_sim_params, timestep_body
    x64 = bool(jax.config.jax_enable_x64); FT = np.float64 if x64 else np.float32
    fc = np.load(FORCING_NPZ); lat = fc["lat"]; n = len(lat); tmpl = fc["template"]
    reo = lambda d: {k: (np.asarray(v)[tmpl] if np.ndim(v) >= 1 and np.shape(v)[0] == 5 else np.asarray(v))
                     for k, v in d.items()}
    so, sn, ve, at = reo(make_soil_inputs()), reo(make_snow_inputs()), reo(make_veg_inputs()), reo(make_atm_inputs())
    so["sand"] = np.round(so["sand"]); so["clay"] = np.round(so["clay"])
    A = lambda x: jnp.asarray(np.asarray(x, dtype=FT))
    st = create_ibis_state(GridConfig(nlon=n, nlat=1, npoi=n, xres=1.0, yres=1.0))
    # the initial state of the lockstep (verify/timestepping/verify_lsxmain.py setup_jax_state)
    soil = eqx.tree_at(lambda s: (s.tsoi, s.wsoi, s.wisoi, s.sand, s.clay, s.poros, s.hydraul, s.suction, s.bex,
                                  s.sfield, s.swilt, s.wpud, s.wipud, s.tg, s.hsoi, s.albsav, s.albsan, s.z0soi), st.soil,
                       tuple(A(so[k]) for k in ("tsoi", "wsoi", "wisoi", "sand", "clay", "poros", "hydraul", "suction",
                                                "bex", "sfield", "swilt", "wpud", "wipud", "tg", "hsoi"))
                       + (A(np.full(n, 0.10)), A(np.full(n, 0.20)), A(np.full(n, 0.005))))
    # the snow constants and the solid-phase specific heat and density that the lockstep gave both codes (the F77
    # wrapper's set_snow defaults and the synthetic soil; verify_lsxmain.setup_jax_state with the port of June 2026)
    snow = eqx.tree_at(lambda s: (s.fi, s.tsno, s.hsno, s.ti, s.tlsub, s.hsnomin, s.fimax, s.z0sno, s.fimin), st.snow,
                       (A(sn["fi"]), A(sn["tsno"]), A(sn["hsno"]), A(so["ti"]), A(ve["tlsub"]), A(0.10), A(0.95),
                        A(0.005), A(0.02)))
    soil = eqx.tree_at(lambda s: (s.csoi, s.rhosoi), soil, (A(np.full((n, soil.tsoi.shape[1]), 870.0)),
                                                         A(np.full((n, soil.tsoi.shape[1]), 2650.0))))
    keys = ("ta", "qa", "ua", "psurf", "coszen", "solad", "solai", "fira", "raina", "snowa")
    atm = eqx.tree_at(lambda a: tuple(getattr(a, k) for k in keys), st.atm, tuple(A(at[k]) for k in keys))
    can = eqx.tree_at(lambda c: (c.tu, c.ts, c.tl, c.t12, c.t34, c.q12, c.q34, c.wliqu, c.wliqs, c.wliql, c.wsnou,
                                 c.wsnos, c.wsnol, c.gs_upper, c.gs_lower, c.a10scalparamu, c.a10daylightu,
                                 c.a10scalparaml, c.a10daylightl), st.canopy,
                      tuple(A(ve[k]) for k in ("tu", "ts", "tl", "t12", "t34", "q12", "q34", "wliqu", "wliqs", "wliql",
                                               "wsnou", "wsnos", "wsnol"))
                      + (A(np.full((n, 2), 0.5)), A(np.full((n, 3), 0.5)), A(np.full(n, 1.0)), A(np.full(n, 21600.0)),
                         A(np.full(n, 1.0)), A(np.full(n, 21600.0))))
    veg = eqx.tree_at(lambda v: (v.fu, v.fl, v.lai, v.sai, v.ztop, v.zbot, v.frac, v.froot, v.exist), st.vegetation,
                      tuple(A(ve[k]) for k in ("fu", "fl", "lai", "sai", "ztop", "zbot", "frac", "froot"))
                      + (A(np.zeros((n, 12))),))
    state = eqx.tree_at(lambda s: (s.soil, s.snow, s.atm, s.canopy, s.vegetation), st, (soil, snow, atm, can, veg))
    if not x64:                                   # as the GSWP3 driver: every floating leaf in single precision
        state = jax.tree.map(lambda x: x.astype(jnp.float32) if isinstance(x, jnp.ndarray)
                             and jnp.issubdtype(x.dtype, jnp.floating) else x, state)
    consts = get_constants()
    params = eqx.tree_at(lambda p: (p.wpudmax, p.zwpmax), default_sim_params(n), (4.5, 0.5))

    @eqx.filter_jit
    def run_day(state, atm_day):
        def body(s, a):
            s = eqx.tree_at(lambda q: tuple(getattr(q.atm, k) for k in keys), s, tuple(a))
            s = timestep_body(s, consts, params, 3600.0)
            return s, (s.canopy.tu, s.canopy.tl, s.soil.tg, s.soil.tsoi[:, 0], s.soil.wsoi[:, 0], s.canopy.fseng,
                       s.canopy.fsenu, s.snow.tsno[:, 0])
        return jax.lax.scan(body, state, atm_day)

    recs = {k: [] for k in PREC_FIELDS}; jday = 0
    for m in range(1, 13):
        for d in range(int(fc[f"ndays_{m}"])):
            jday += 1
            day = {v: fc[f"{v}_{m}"][d] for v in ("tas", "pr", "rsds", "rlds", "huss", "wind", "ps")}
            hs = [_disaggregate(day, float(h), jday, lat) for h in range(24)]
            state, rec = run_day(state, tuple(A(np.stack([h[k] for h in hs])) for k in keys))
            for k, r in zip(PREC_FIELDS, rec):
                recs[k].append(np.asarray(r, dtype=np.float64))
    np.savez(out, x64=x64, **{k: np.concatenate(v, 0) for k, v in recs.items()})


def _precision_outputs():
    """Run the two precisions in subprocesses (JAX's precision is fixed per process) when their outputs are missing."""
    if not FORCING_NPZ.exists():
        forcing_cache()
    for tag, x64 in (("f64", "1"), ("f32", "0")):
        out = PRECISION_DIR / f"prec_{tag}.npz"
        if not out.exists():
            env = dict(os.environ, JAX_ENABLE_X64=x64, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
            subprocess.run([sys.executable, __file__, "--precision-run", str(out)], env=env, check=True)
    return np.load(PRECISION_DIR / "prec_f64.npz"), np.load(PRECISION_DIR / "prec_f32.npz")


def fig_17_2():
    z, fields = _lockstep()
    yi = list(z["years"]).index(1951)
    p64, p32 = _precision_outputs()
    assert bool(p64["x64"]) and not bool(p32["x64"])
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"wspace": 0.3})
    out = {}
    for ax, (fld, lab) in zip(axs, (("canopy.tu", "(a) upper-canopy temperature"),
                                    ("soil.tsoi[0]", "(b) top soil-layer temperature"))):
        f77 = z["worst"][fields.index(fld)][:, yi]
        prec = np.nanmax(np.abs(p32[fld] - p64[fld]), axis=0)
        for v, name, col in ((f77, "IBIS 2.6b4 − IBIS 3.0, both double", BLUE),
                             (prec, "IBIS 3.0 single − double", ORANGE)):
            s = np.sort(v[np.isfinite(v)])
            ax.step(np.maximum(s, 1e-12), np.arange(1, s.size + 1) / s.size, where="post", color=col, lw=1.5,
                    label=name)
        ax.set_xscale("log"); ax.set_xlim(1e-12, 30); ax.set_ylim(0, 1.0)
        ax.set_xticks([1e-12, 1e-9, 1e-6, 1e-3, 1.0]); ax.set_xticklabels(["1 pK", "1 nK", "1 µK", "1 mK", "1 K"])
        ax.set_xlabel("largest hourly difference in 1951")
        ax.set_title(lab, loc="left", fontsize=9, weight="bold", color=INK)
        _clean(ax)
        out[fld] = (np.nanmedian(f77), np.nanmedian(prec), np.nanpercentile(prec, 90), np.nanmax(prec),
                    float(np.nanmedian(np.abs(p32[fld] - p64[fld]))))
    axs[0].set_ylabel("fraction of the cells")
    axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(OUT / "fig_17_2_precision.png")
    plt.close(fig)
    for fld, v in out.items():
        print(f"17.2 {fld}: median cell F77 {v[0]:.3g}; single-double median cell {v[1]:.3g}, 90th pct {v[2]:.3g}, "
              f"max {v[3]:.3g}; median hourly |single-double| {v[4]:.3g}")
    for fld in ("soil.wsoi[0]", "flux.fseng", "soil.tg"):
        d = np.nanmax(np.abs(p32[fld] - p64[fld]), axis=0)
        print(f"17.2 {fld}: single-double annual max, median cell {np.nanmedian(d):.3g}, max {np.nanmax(d):.3g};"
              f" annual mean of double, median cell {np.nanmedian(np.nanmean(p64[fld], axis=0)):.4g},"
              f" |mean single - mean double| median cell {np.nanmedian(np.abs(np.nanmean(p32[fld], 0) - np.nanmean(p64[fld], 0))):.3g}"
              f" max {np.nanmax(np.abs(np.nanmean(p32[fld], 0) - np.nanmean(p64[fld], 0))):.3g}")


# ---------------------------------------------------------------------------------------------------------------
# The other numbers of the chapter
# ---------------------------------------------------------------------------------------------------------------

LOCKSTEP_BOUNDS = {          # the plausibility windows of the lockstep (verify/.../global_gswp3_year.py TRACKED_*)
    "soil.tg": (200, 350), "soil.tsoi[0]": (200, 350), "snow.tsno[0]": (200, 350), "canopy.tu": (200, 350),
    "canopy.ts": (200, 350), "canopy.tl": (200, 350), "canopy.t12": (200, 350), "canopy.t34": (200, 350),
    "soil.wsoi[0]": (0, 1), "soil.wisoi[0]": (0, 1), "snow.fi": (0, 1), "snow.hsno[0]": (0, 5), "stresstu": (0, 1),
    "stresstl": (0, 1), "canopy.q12": (0, 0.04), "canopy.q34": (0, 0.04), "flux.fseng": (-500, 500),
    "flux.fsenu": (-500, 500), "flux.fsenl": (-500, 500), "flux.fseni": (-1500, 1500), "flux.fvapg": (-1e-3, 1e-3)}


def numbers_lockstep():
    """Table 17.8 and the range exceedances of the 46-year lockstep."""
    z, fields = _lockstep()
    for fld in ("canopy.tu", "canopy.tl", "soil.tg", "soil.tsoi[0]", "soil.wsoi[0]", "flux.fseng", "flux.fvapg"):
        a = z["worst"][fields.index(fld)]
        print(f"lockstep {fld}: cell-years {np.isfinite(a).sum()}, median / 95th / 99th / largest of D_cy",
              np.round(np.nanpercentile(a, [50, 95, 99, 100]), 10))
    import collections
    n = collections.Counter(); both = collections.Counter(); lo_ = collections.Counter(); hi_ = collections.Counter()
    cells = collections.defaultdict(set); ext = {}; nan = oor = 0
    for f in sorted(glob.glob(PER_YEAR + "/*.json")):
        d = json.load(open(f)); nan += d["nan_count"]; oor += d["out_of_range_count"]
        for kk, (lo, hi) in d["f77_range"].items():
            c, fld, _m = kk.split("|"); blo, bhi = LOCKSTEP_BOUNDS[fld]; jlo, jhi = d["jax_range"][kk]
            fo = lo < blo or hi > bhi; jo = jlo < blo or jhi > bhi
            n[(fld, "F77")] += fo; n[(fld, "JAX")] += jo; both[fld] += fo and jo
            if fo or jo:
                cells[fld].add(c); lo_[fld] += min(lo, jlo) < blo; hi_[fld] += max(hi, jhi) > bhi
                e = ext.get(fld, (np.inf, -np.inf)); ext[fld] = (min(e[0], lo, jlo), max(e[1], hi, jhi))
    print("lockstep non-finite values", nan, "; step-level range exceedances counted by the runner", oor)
    for fld in sorted(cells, key=lambda x: -len(cells[x])):
        la = collections.Counter(float(c[1:].split("_")[0].replace("p", ".")) for c in cells[fld])
        print(f"  {fld}: cell-months outside F77 {n[(fld, 'F77')]} JAX {n[(fld, 'JAX')]} both {both[fld]}"
              f" (below {lo_[fld]}, above {hi_[fld]}); cells {len(cells[fld])}, cells by latitude {dict(sorted(la.items()))};"
              f" values {ext[fld][0]:.4g}..{ext[fld][1]:.4g}")
    cold = set().union(*(cells[f] for f in cells if LOCKSTEP_BOUNDS[f][0] == 200)) if cells else set()
    la = collections.Counter(float(c[1:].split("_")[0].replace("p", ".")) for c in cold)
    print(f"  cells with any temperature below 200 K: {len(cold)}, by latitude {dict(sorted(la.items()))}")


def numbers_precision():
    """Section 17.5: single against double precision at the 117 cells in 1951 (the files of `_precision_outputs`), and
    section 17.8: two CPU launches of the single-precision run, when `prec_f32_repeat.npz` exists beside them."""
    z, fields = _lockstep(); yi = list(z["years"]).index(1951)
    p64, p32 = _precision_outputs()
    print("precision runs: x64", bool(p64["x64"]), bool(p32["x64"]), "hourly steps x cells", p64["canopy.tu"].shape,
          "non-finite", int(np.isnan(p32["canopy.tu"]).sum() + np.isnan(p64["canopy.tu"]).sum()))
    for f in PREC_FIELDS:
        d = np.abs(p32[f] - p64[f]); amax = np.nanmax(d, axis=0)
        dm = np.abs(np.nanmean(p32[f], 0) - np.nanmean(p64[f], 0))
        s = (f"  {f:13s} largest hourly |single - double| per cell: median {np.nanmedian(amax):.3g}, 90th pct "
             f"{np.nanpercentile(amax, 90):.3g}, max {np.nanmax(amax):.3g}; hourly median {np.nanmedian(d):.3g}; "
             f"|annual mean single - double|: median {np.nanmedian(dm):.3g}, max {np.nanmax(dm):.3g}")
        if f.startswith(("canopy.t", "soil.tg", "soil.tsoi", "snow.tsno")):
            s += f"; cells above 0.1 K {int((amax > 0.1).sum())} of {amax.size}"
        if f in fields:
            w = z["worst"][fields.index(f)][:, yi]
            s += (f"; lockstep F77 - port 1951: median {np.nanmedian(w):.3g}, max {np.nanmax(w):.3g};"
                  f" ratio of the median cells (precision / lockstep) {np.nanmedian(amax) / np.nanmedian(w):.0f}")
        # daily and monthly means of 1951 (365 days of 24 steps)
        a32, a64 = np.asarray(p32[f]), np.asarray(p64[f])
        dday = np.abs(a32.reshape(365, 24, -1).mean(1) - a64.reshape(365, 24, -1).mean(1))
        ed = np.cumsum([0] + [24 * (date(1951 + (m == 12), m % 12 + 1, 1) - date(1951, m, 1)).days for m in range(1, 13)])
        dmon = np.abs(np.stack([a32[ed[i]:ed[i + 1]].mean(0) - a64[ed[i]:ed[i + 1]].mean(0) for i in range(12)]))
        s += f"; |daily mean single - double| max {np.nanmax(dday):.3g}; |monthly mean| max {np.nanmax(dmon):.3g}"
        print(s)
    rep = PRECISION_DIR / "prec_f32_repeat.npz"
    if rep.exists():
        r = np.load(rep)
        for f in PREC_FIELDS:
            print(f"  CPU launch to launch, {f}: elements differing {int((p32[f] != r[f]).sum())} of {p32[f].size}")
    else:
        print("  (no prec_f32_repeat.npz: run `--precision-run <that path>` with JAX_ENABLE_X64=0 for section 17.8)")


def verify_timestep():
    """Section 17.9.3: every time-step check of verify/timestepping/verify_lsxmain.py, as checked in, on the code of the
    working tree. Since 2026-11-01 (commit bed0ae7) the verifier gives both codes IBIS 2.6b4's own settings from
    tests/f2py/synthetic_inputs.py (texture-class soils with positive suctions, the solids, snow constants, base
    permeability and puddle capacity of IBIS 2.6b4, existence flags from its climate rules), so no constant is matched
    here any more (until then this function gave the port the harness's snow constants and solids). Then the cold snap
    once more with the bounds of Table 17.4 that act in it lifted: the surface heat-flux and temperature bounds through
    their environment variables, the ground-evaporation bound through loops.FVAPG_PHYS_MAX. The water floors stay."""
    if not bool(__import__("jax").config.jax_enable_x64):
        env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"),
                   GCOV_PREFIX=os.environ.get("GCOV_PREFIX", str(WORK / "gcov")))
        subprocess.run([sys.executable, __file__, "timestep"], env=env, check=True); return
    sys.path.insert(0, str(ROOT))
    for k_ in ("IBIS_FLUX_MAX", "IBIS_T_MAX_DEV"):
        os.environ.pop(k_, None)
    import verify.timestepping.verify_lsxmain as V
    print("== the verifier as checked in (the bounds of Table 17.4 as the model runs)", flush=True)
    rc = V.main()
    print("verifier exit code", rc, flush=True)
    import ibis.timestepping.loops as L
    os.environ.update({"IBIS_FLUX_MAX": "1e9", "IBIS_T_MAX_DEV": "1e9"})
    L.FVAPG_PHYS_MAX = 1e9
    print("== cold snap with the surface heat-flux, surface-temperature and ground-evaporation bounds lifted",
          flush=True)
    soil_i, snow_i, atm_i, veg_i = V.setup_inputs()
    V.run_multi_day(soil_i, snow_i, atm_i, veg_i, 7, "cold snap, bounds lifted", forcing=V.cold_snap_forcing)


def verify_selfpert():
    """Section 17.9.3: are the day-4 events of the 7-day diurnal and monsoon checks rounding? Each scenario is run with
    the Fortran, the port, and the port started 1e-12 K away from itself (every temperature of the state); for every
    step with a temperature difference above 1e-8 K in either comparison, the largest of each, its field and its cell.
    An event that the perturbed port repeats, at the same step and of the same size, is rounding carried into a branch,
    not a difference of the codes. Double precision in a subprocess, as `timestep` (a few minutes on one CPU)."""
    if not bool(__import__("jax").config.jax_enable_x64):
        env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"),
                   GCOV_PREFIX=os.environ.get("GCOV_PREFIX", str(WORK / "gcov")))
        subprocess.run([sys.executable, __file__, "selfpert"], env=env, check=True); return
    import equinox as eqx
    sys.path.insert(0, str(ROOT))
    import verify.timestepping.verify_lsxmain as V
    from ibis.constants import get_constants
    get = {"canopy.tu": lambda f, s: (f[0]["tu"], s.canopy.tu), "canopy.tl": lambda f, s: (f[0]["tl"], s.canopy.tl),
           "canopy.t12": lambda f, s: (f[0]["t12"], s.canopy.t12), "canopy.t34": lambda f, s: (f[0]["t34"], s.canopy.t34),
           "soil.tg": lambda f, s: (f[1]["tg"], s.soil.tg),
           "soil.tsoi[0]": lambda f, s: (f[1]["tsoi"][:, 0], s.soil.tsoi[:, 0]),
           "snow.tsno[0]": lambda f, s: (f[2]["tsno"][:, 0], s.snow.tsno[:, 0])}
    tpath = lambda s: (s.soil.tsoi, s.soil.tg, s.canopy.tu, s.canopy.ts, s.canopy.tl, s.canopy.t12, s.canopy.t34,
                       s.snow.tsno)
    consts, params = get_constants(), V.harness_params()
    for label, forcing in (("diurnal", V.base_forcing), ("monsoon", V.monsoon_forcing)):
        si, sn, ai, vi = V.setup_inputs()
        f77 = V.setup_f77_state(si, sn, ai, vi)
        a = V.setup_jax_state(si, sn, ai, vi)
        b = V.setup_jax_state(si, sn, ai, vi)
        b = eqx.tree_at(tpath, b, tuple(x + 1e-12 for x in tpath(b)))
        for step in range(168):
            atm = V.diurnal_atm(ai, float(step % 24), **forcing(step, 168))
            f77.set_atm(**atm); a = V.update_jax_atm(a, atm); b = V.update_jax_atm(b, atm)
            f77.call_lsxmain()
            a = V.timestep_body(a, consts, params, V.DTIME); b = V.timestep_body(b, consts, params, V.DTIME)
            fo = (f77.get_veg(), f77.get_soil(), f77.get_snow())
            dfa = {k: np.abs(np.asarray(g(fo, a)[0]) - np.asarray(g(fo, a)[1])) for k, g in get.items()}
            dab = {k: np.abs(np.asarray(g(fo, b)[1]) - np.asarray(g(fo, a)[1])) for k, g in get.items()}
            kf = max(dfa, key=lambda k: dfa[k].max()); kb = max(dab, key=lambda k: dab[k].max())
            if dfa[kf].max() > 1e-8 and 72 <= step < 96:            # the fourth day, where the events are
                print(f"  {label} step {step + 1}: F77 - port {dfa[kf].max():.3e} ({kf}, cell {int(dfa[kf].argmax())});"
                      f" port' - port {dab[kb].max():.3e} ({kb}, cell {int(dab[kb].argmax())})", flush=True)


def _inject_f77_state(f77, s):
    """The Fortran's state copied into the port's state `s`: canopy temperatures, humidities and intercepted water; soil
    temperature, water, ice and puddles; snow cover, layer temperatures and thicknesses and the buried lower canopy's
    temperature; the stomatal carries slot by slot (upper: broadleaf, conifer; lower: shrub, C3, C4 grass)."""
    import jax.numpy as jnp, equinox as eqx
    veg, soil, snow, sto = f77.get_veg(), f77.get_soil(), f77.get_snow(), f77.get_stomata()
    A = jnp.asarray
    s = eqx.tree_at(lambda q: (q.canopy.tu, q.canopy.ts, q.canopy.tl, q.canopy.t12, q.canopy.t34, q.canopy.q12,
                               q.canopy.q34, q.canopy.wliqu, q.canopy.wsnou, q.canopy.wliqs, q.canopy.wsnos,
                               q.canopy.wliql, q.canopy.wsnol), s,
                    tuple(A(veg[k]) for k in ("tu", "ts", "tl", "t12", "t34", "q12", "q34", "wliqu", "wsnou",
                                              "wliqs", "wsnos", "wliql", "wsnol")))
    s = eqx.tree_at(lambda q: (q.soil.tg, q.soil.tsoi, q.soil.wsoi, q.soil.wisoi, q.soil.wpud, q.soil.wipud), s,
                    tuple(A(soil[k]) for k in ("tg", "tsoi", "wsoi", "wisoi", "wpud", "wipud")))
    s = eqx.tree_at(lambda q: (q.snow.fi, q.snow.tsno, q.snow.hsno, q.snow.tlsub), s,
                    (A(snow["fi"]), A(snow["tsno"]), A(snow["hsno"]), A(veg["tlsub"])))
    st = lambda *k: jnp.stack([A(sto[x]) for x in k], -1)
    return eqx.tree_at(lambda q: (q.canopy.ci_upper, q.canopy.gs_upper, q.canopy.cs_upper, q.canopy.ci_lower,
                                  q.canopy.gs_lower, q.canopy.cs_lower), s,
                       (st("ciub", "ciuc"), st("gsub", "gsuc"), st("csub", "csuc"), st("cils", "cil3", "cil4"),
                        st("gsls", "gsl3", "gsl4"), st("csls", "csl3", "csl4")))


def verify_inject():
    """Section 17.9.3: are the day-4 events of the 7-day diurnal and monsoon checks rounding? The Fortran and the port
    run in lockstep for 80 steps, as verify_lsxmain.run_multi_day does; then the Fortran's state is copied into a second
    port state (`_inject_f77_state`), and the Fortran and both port states advance to the end of day 4 (step 96). Step
    81 is 08:00-09:00, when the forcing puts the freeze-thaw cell's air at 273.16 K. An event that the port started
    from the Fortran's state does not show is rounding carried into a branch; one it shows unchanged is a difference of
    the codes at that step. Double precision in a subprocess, as `timestep` (about 10 minutes on one CPU)."""
    if not bool(__import__("jax").config.jax_enable_x64):
        env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"),
                   GCOV_PREFIX=os.environ.get("GCOV_PREFIX", str(WORK / "gcov")))
        subprocess.run([sys.executable, __file__, "inject"], env=env, check=True); return
    sys.path.insert(0, str(ROOT))
    import verify.timestepping.verify_lsxmain as V
    from ibis.constants import get_constants

    def tdiff(f77, s):
        fv, fs, fn = f77.get_veg(), f77.get_soil(), f77.get_snow()
        d = {"canopy.tu": fv["tu"] - s.canopy.tu, "canopy.ts": fv["ts"] - s.canopy.ts, "canopy.tl": fv["tl"] - s.canopy.tl,
             "canopy.t12": fv["t12"] - s.canopy.t12, "canopy.t34": fv["t34"] - s.canopy.t34, "soil.tg": fs["tg"] - s.soil.tg,
             "soil.tsoi[0]": fs["tsoi"][:, 0] - s.soil.tsoi[:, 0], "snow.tsno[0]": fn["tsno"][:, 0] - s.snow.tsno[:, 0]}
        d = {k: np.abs(np.asarray(v, float)) for k, v in d.items()}
        k = max(d, key=lambda q: d[q].max())
        return float(d[k].max()), k, int(d[k].argmax())

    consts, params = get_constants(), V.harness_params()
    for label, forcing in (("diurnal", V.base_forcing), ("monsoon", V.monsoon_forcing)):
        si, sn, ai, vi = V.setup_inputs()
        f77 = V.setup_f77_state(si, sn, ai, vi)
        a = V.setup_jax_state(si, sn, ai, vi); b = None
        pre = 0.0; worst = {"as run": (0.0,), "injected": (0.0,)}
        for step in range(96):
            if step == 80:
                b = _inject_f77_state(f77, a)
            atm = V.diurnal_atm(ai, float(step % 24), **forcing(step, 168))
            f77.set_atm(**atm); f77.call_lsxmain()
            a = V.timestep_body(V.update_jax_atm(a, atm), consts, params, V.DTIME)
            if b is None:
                pre = max(pre, tdiff(f77, a)[0]); continue
            b = V.timestep_body(V.update_jax_atm(b, atm), consts, params, V.DTIME)
            for name, s in (("as run", a), ("injected", b)):
                d = tdiff(f77, s)
                if d[0] > worst[name][0]:
                    worst[name] = d + (step + 1,)
                if step == 80:
                    print(f"  {label} step 81 (air of the freeze-thaw cell {float(atm['ta'][2]):.4f} K), Fortran - port "
                          f"{name}: {d[0]:.3e} K ({d[1]}, cell {d[2]})", flush=True)
        print(f"{label}: steps 1-80 at most {pre:.3e} K; steps 81-96, Fortran - port as it ran: largest "
              f"(K, field, cell, step) {worst['as run']}; Fortran - port started from the Fortran's state after step 80: "
              f"largest {worst['injected']}", flush=True)


def numbers_code():
    """The two-stream table, the sand and clay rounding, the single-precision pressure difference, the state leaves."""
    import xarray as xr
    tab = np.array([round(0.5 - 0.01 * k / 3.0, 4) for k in range(101)])      # F77 radiation.f tablemu
    mu = np.linspace(0, 1, 2000001)
    print("tablemu at nint(100 mu) against 0.5 - mu/3: largest difference %.4f"
          % np.abs(tab[np.floor(mu * 100 + 0.5).astype(int)] - (0.5 - mu / 3.0)).max())
    land = _land_mask(); ds = xr.open_dataset("/shared/nas2/Min/HWSD/soita.nc")
    sa = np.asarray(ds.sandpct.values)[:, land].astype(float); cl = np.asarray(ds.claypct.values)[:, land].astype(float)
    ok = np.isfinite(sa) & np.isfinite(cl); sa, cl = sa[ok], cl[ok]
    zc = lambda s_, c_: 0.01 * s_ * 0.300 + (1 - 0.01 * s_ - 0.01 * c_) * 0.265 + 0.01 * c_ * 0.250   # setsoi zcondry
    r = lambda x: np.floor(x + 0.5)
    rel = np.abs(zc(sa, cl) - zc(r(sa), r(cl))) / zc(r(sa), r(cl))
    print("dry conductivity, unrounded against rounded sand and clay: layers %d, not whole percent %d, largest "
          "relative difference %.2e" % (rel.size, int(((sa != r(sa)) | (cl != r(cl))).sum()), rel.max()))
    p = np.linspace(50000.0, 105000.0, 200001).astype(np.float32); pd_ = p.astype(np.float64)
    e = np.abs((p - p * np.float32(0.999)).astype(np.float64) - (pd_ - pd_ * 0.999)) / (pd_ - pd_ * 0.999)
    print("psurf - 0.999 psurf in single precision, 500-1050 hPa: relative error median %.2e, largest %.2e"
          % (np.median(e), e.max()))
    import jax, equinox as eqx
    from ibis.dimensions import GridConfig
    from ibis.state.model_state import create_ibis_state
    from ibis.bgc.methane import create_methane_state
    st = create_ibis_state(GridConfig(npoi=3, nbedrock=3), with_stress=True, with_etparts=True, with_seasonal=True)
    st = eqx.tree_at(lambda q: q.wetland, st, create_methane_state(3), is_leaf=lambda x: x is None)
    print("state leaves (release configuration):", {k: len(jax.tree.leaves(getattr(st, k))) for k in
          ("soil", "snow", "canopy", "atm", "vegetation", "carbon", "hydrology", "summary", "wetland", "stress",
           "etparts", "seasonal")}, "total", len(jax.tree.leaves(st)))
    stw = create_ibis_state(GridConfig(npoi=3, nbedrock=3), with_stress=True, with_etparts=True, with_seasonal=True,
                            with_wbal=True)
    stw = eqx.tree_at(lambda q: q.wetland, stw, create_methane_state(3), is_leaf=lambda x: x is None)
    print("state leaves with physics.water_balance:", {k: len(jax.tree.leaves(getattr(stw, k))) for k in
          ("neeparts", "wbal")}, "total", len(jax.tree.leaves(stw)))
    paths = jax.tree_util.tree_flatten_with_path(st)[0]
    rel = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30"
    for rp in (f"{rel}/spinup/restart/ibis_restart_1900.npz", f"{rel}/hindcast/restart/ibis_restart_2010.npz"):
        zz = np.load(rp); nl = int(zz["num_leaves"]); kinds = {}; nel = 0; bad = 0; nocell = []
        for k in range(nl):
            a = zz[f"leaf_{k}"]; kinds[str(a.dtype)] = kinds.get(str(a.dtype), 0) + 1; nel += a.size
            bad += int((~np.isfinite(a)).sum()) if np.issubdtype(a.dtype, np.floating) else 0
            if a.ndim == 0 or a.shape[0] != 95333:
                nocell.append(k)
        names = ([jax.tree_util.keystr(paths[k][0]) for k in nocell] if nl == len(paths) else "")
        print(f"release restart {os.path.basename(rp)}: {nl} leaves {kinds}, {nel} numbers, "
              f"{os.path.getsize(rp) / 1e6:.1f} MB, non-finite {bad}; leaves without the cell dimension {len(nocell)} "
              f"{names}")
    for stage in ("spinup", "transient", "hindcast"):
        sz = [os.path.getsize(f) / 1e6 for f in glob.glob(f"{rel}/{stage}/restart/ibis_restart_*.npz")]
        print(f"restart files of the {stage}: {len(sz)}, {min(sz):.1f}-{max(sz):.1f} MB")


def numbers_gpu():
    """Two GPU launches of 2001: the release hindcast against a second launch of the same code from the same restart."""
    from netCDF4 import Dataset
    A_ = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30/hindcast/out"
    B_ = "/shared/nas2/Hangkai/ibis/verify_outputs/check_control_1030/out"
    land = _land_mask()
    for fn in ["ibis_yearly_2001.nc"] + [f"ibis_monthly_2001_{m:02d}.nc" for m in range(1, 13)]:
        da, db = Dataset(os.path.join(A_, fn)), Dataset(os.path.join(B_, fn))
        for v in da.variables:
            if v in ("lat", "lon", "soilay") or v not in db.variables:
                continue
            a = np.asarray(da.variables[v][:], float); b = np.asarray(db.variables[v][:], float)
            if a.ndim == 3 and a.shape[:2] == (360, 720):
                a, b = np.moveaxis(a, -1, 0), np.moveaxis(b, -1, 0)
            if a.shape[-2:] != (360, 720):
                continue
            m = np.broadcast_to(land, a.shape) & np.isfinite(a) & np.isfinite(b)
            diff = (a != b) & m
            if not diff.any():
                continue
            ncell = int(np.any(diff.reshape(-1, 360, 720), axis=0).sum())
            rc = np.abs(a - b)[diff] / np.maximum(np.maximum(np.abs(a), np.abs(b))[diff], 1e-30)
            sa_, sb_ = a[m].sum(), b[m].sum()      # unweighted sum of the values per unit area over the land cells
            print(f"gpu {fn} {v}: cells differing {ncell}, relative difference of the sum over the land cells "
                  f"{abs(sa_ - sb_) / abs(sa_):.2e}, largest cell relative {rc.max():.2e}")
    # the restart files written at the end of 2001 by the two launches
    ra = np.load("/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30/hindcast/restart/ibis_restart_2001.npz")
    rb = np.load("/shared/nas2/Hangkai/ibis/verify_outputs/check_control_1030/restart/ibis_restart_2001.npz")
    nl = int(ra["num_leaves"]); dif = []
    for k in range(nl):
        a, b = ra[f"leaf_{k}"], rb[f"leaf_{k}"]
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            dif.append((k, a.shape, int((a != b).sum()), float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))))
    print(f"gpu restart 2001: {nl} leaves, differing (leaf, shape, cells, largest difference) {dif}")


D83_DIR = "/shared/nas2/Hangkai/ibis/verify_outputs/step2_d83"           # det1, det2: XLA's deterministic GPU mode
WBAL_RUN = "/shared/nas2/Hangkai/ibis/verify_outputs/step2_wbal_2001_d25b"  # diagnostic year of the revised code


def _nc_values(var):
    a = var[:]
    return np.asarray(a.filled(np.nan) if hasattr(a, "filled") else a)


def _compare_launches(A, B):
    """Every variable of every output file of two launches of 2001, and their end-of-2001 restarts: (variables
    compared, variables differing, restart leaves, restart leaves differing)."""
    from netCDF4 import Dataset
    nvar = ndiff = 0
    for fn in sorted(f for f in os.listdir(os.path.join(A, "out")) if f.endswith(".nc")):
        da, db = Dataset(os.path.join(A, "out", fn)), Dataset(os.path.join(B, "out", fn))
        for v in da.variables:
            a = _nc_values(da.variables[v]); b = _nc_values(db.variables[v]) if v in db.variables else None
            nvar += 1
            ndiff += int(b is None or a.shape != b.shape or not np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.floating)))
        da.close(); db.close()
    ra = np.load(os.path.join(A, "restart", "ibis_restart_2001.npz"))
    rb = np.load(os.path.join(B, "restart", "ibis_restart_2001.npz"))
    nl = int(ra["num_leaves"])
    nd = sum(1 for k in range(nl) if not np.array_equal(ra[f"leaf_{k}"], rb[f"leaf_{k}"], equal_nan=True))
    return nvar, ndiff, nl, nd


def numbers_d83():
    """Section 17.8: two GPU launches of 2001 with XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true (det1, det2) and
    one without it (plain), on the revised code from the release run's restart of 2000 (run_d83.sh in D83_DIR)."""
    for a, b in (("det1", "det2"), ("plain", "det1")):
        nvar, ndiff, nl, nd = _compare_launches(os.path.join(D83_DIR, a), os.path.join(D83_DIR, b))
        print(f"d83 {a} against {b}: output variables {nvar}, differing {ndiff}; restart leaves {nl}, differing {nd}")
    for r in ("det1", "det2", "plain"):
        t = [ln.strip() for ln in open(os.path.join(D83_DIR, r, "run.log")) if "complete in" in ln]
        print(f"d83 {r}: {t[-1] if t else 'no timing'}")


def numbers_wbal():
    """Sections 17.4 and 17.5: the water-balance diagnostic year 2001 of the revised code (physics.water_balance):
    how often the counted bounds act, the water the floors add, and the closure P - ET - R - D - dW + adj per cell."""
    from netCDF4 import Dataset
    d = Dataset(os.path.join(WBAL_RUN, "out", "ibis_yearly_2001.nc"))
    g = lambda n: np.asarray(_nc_values(d.variables[n]), dtype=float).squeeze()
    land = np.isfinite(g("ayprcp"))
    print("wbal land cells", int(land.sum()))
    for n in ("n_bound", "n_snow", "ay_adj_uptake", "ay_adj_puddle", "ay_adj_ice"):
        x = g(n)[land]
        print(f"wbal {n}: cells with a nonzero value {int((x != 0).sum())}, largest {np.nanmax(x):.3g}")
    r = g("ayprcp") - g("ayaet") - g("aytrunof") - g("aytdrain") - g("aydwtot") + g("ayadjust")
    LA, LO = np.meshgrid(g("lat"), g("lon"), indexing="ij")
    ice = land & ((LA < -60) | ((LA > 59.5) & (LA < 84) & (LO > -75) & (LO < -10)))   # Antarctica, the Greenland box
    big = land & (np.abs(r) > 1.0)
    print(f"wbal closure: median {np.nanmedian(r[land]):+.3f}, p01 {np.nanpercentile(r[land], 1):+.3f}, p99 "
          f"{np.nanpercentile(r[land], 99):+.3f} mm/yr; |residual| > 1 mm/yr in {int(big.sum())} cells, "
          f"{int((big & ice).sum())} of them south of 60 S or in 60-84 N, 10-75 W; elsewhere at most "
          f"{np.nanmax(np.abs(r[land & ~ice])):.3f}; on the ice {np.nanmin(r[ice]):+.3f} to {np.nanmax(r[ice]):+.3f}")


def numbers():
    numbers_lockstep(); numbers_code(); numbers_precision(); numbers_gpu(); numbers_d83(); numbers_wbal()


FIGURES = {"17.1": fig_17_1, "17.2": fig_17_2, "17.3": fig_17_3}   # 17.2 precision, 17.3 lockstep
EXTRA = {"numbers": numbers, "timestep": verify_timestep, "selfpert": verify_selfpert, "inject": verify_inject,
         "d83": numbers_d83, "wbal": numbers_wbal}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--precision-run"]:
        precision_run(sys.argv[2])
    else:
        for key in (sys.argv[1:] or FIGURES):
            {**FIGURES, **EXTRA}[key]()
            print(key, "done")
