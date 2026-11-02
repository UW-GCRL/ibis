"""Figures of Chapter 13 (Fire) of the IBIS 3.0 technical note, drawn from the model's code, the release run and GFED5.

Run from the repository root with the project interpreter, CPU only:

    JAX_PLATFORMS=cpu /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch13.py            # all
    JAX_PLATFORMS=cpu /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch13.py 13.2 13.3  # some
    JAX_PLATFORMS=cpu /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch13.py numbers    # 13.4's numbers only

13.1 is a schematic of the calculation as the current code does it. 13.2 evaluates the model's fire functions
(`ibis.physics.fire`) over their inputs with the release parameters (`fire:` section of params_31s_ch4h.yaml, read by
the model's own builder) and the current default rule for the fire season (`fire.daylight_only`, false). 13.3
replays the hourly fire weather of 2001 from the GSWP3 forcing through the model's forcing conversion and weather
generator at every land point, twice: with the release's rule (sun and weather at the start of each step, every step
counted), which must give the counters the release run stored for 2001, and with the optional daylight variant (sun and weather at
the middle of each step, `weather.sun_midstep`; only daylight steps counted, `fire.daylight_only`); it draws three cells
and prints the zone means of the fire season and the dryness under both rules (the zones of Table 13.5, land-area
weights). The other factors of the burned fraction come from the release run's restarts. 13.4 maps the release run's
fire (hindcast restarts 2000-2010, final_rel30) against GFED5 Norm; the same function prints the numbers quoted in
sections 13.4, 13.5 and 13.9 of the chapter (burned area, the fate of the killed carbon and its check against the run's
aycdisturb, the trace-gas emissions of the daylight variant applied to the release run's combusted carbon and those of the
release's own diagnostic, the factors by zone); `numbers` prints them without drawing the figure. The release run counted
every step, so its counters are evaluated with the release's rule (`daylight_only` false). Totals, means and bands are
weighted by the land area of each cell (cell area times the ESA CCI 1992 land fraction, verify/calibration/landmask.py),
the weighting of the release's totals; the cell-area means of the release fire score are printed as well. Nothing a
figure shows is typed in by hand except labels and the schematic's drawing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # also sets rcParams, sys.path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

OUT = Path(__file__).resolve().parent
PARAMS = Path(__file__).resolve().parents[3] / "python-port/configs/params/params_31s_ch4h.yaml"
RESTART = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30/hindcast/restart/ibis_restart_{}.npz"
KOPPEN = "/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz"
GFED5 = "/shared/nas2/Hangkai/ibis/inputs/fire/gfed5"
GSWP3 = "/shared/nas2/Min/GSWP3"
YEARLY = "/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30/hindcast/out/ibis_yearly_{}.nc"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                          "#008300", "#6250d6", "#e34948")
LIGHT_BLUE = "#b9d3f2"
SPECIES = ("co2", "co", "ch4", "n2o", "nox")


def _release_fire_params(zone_idx=None, daylight_only=None):
    """FireParams of the release: the yaml 'fire' section through the model's builder; with the zone of each cell,
    the closed-canopy flag that apply_types sets. `daylight_only` overrides the fire-season rule: the release file does
    not set it, so the builder gives the current default (false, every step), as in the release run."""
    import dataclasses
    import yaml
    import jax.numpy as jnp
    import equinox as eqx
    from ibis.io.physics_params import build_fire_params
    from ibis.io.pft_types import ZONES
    fp = build_fire_params(yaml.safe_load(open(PARAMS)))
    if daylight_only is not None:
        fp = dataclasses.replace(fp, daylight_only=bool(daylight_only))
    if zone_idx is not None:
        flag = np.isin(np.asarray(zone_idx), [ZONES.index(z) for z in fp.closed_canopy_zones]).astype(np.float32)
        fp = eqx.tree_at(lambda o: o.closed_canopy_cell, fp, jnp.asarray(flag), is_leaf=lambda x: x is None)
    return fp


def _land_points():
    """Latitude, longitude and grid indices of the land points in the order of the global driver (argwhere, N to S)."""
    land = _land_mask()
    ij = np.argwhere(land)
    return 89.75 - 0.5 * ij[:, 0], -179.75 + 0.5 * ij[:, 1], ij[:, 0], ij[:, 1], land


def _zones(lat, lon, period):
    from ibis.io.pft_types import zones_from_classes
    k = np.load(KOPPEN)
    klat = np.asarray(k["lat"]); klon = np.asarray(k["lon"])
    ki = np.rint((lat - klat[0]) / (klat[1] - klat[0])).astype(int)
    kj = np.rint((lon - klon[0]) / (klon[1] - klon[0])).astype(int)
    return zones_from_classes(np.asarray(k["classes"])[period][ki, kj])


def _areas(lat, I, J):
    """(cell area, land area) of the land points, m2. The land area is the cell area times the land fraction of the cell
    in the ESA CCI land cover of 1992 (verify/calibration/landmask.true_landfrac), the weighting of the release's totals
    (Chapters 14-16). The binary mask counts the whole of a coastal cell as land (165 Mkm2 against 144). The cell-area
    means are the weighting of the release fire score (verify/fire/evaluate_gfed5.py)."""
    cell = 6.371e6 ** 2 * np.deg2rad(0.5) * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verify/calibration"))
    from landmask import true_landfrac
    return cell, cell * np.nan_to_num(true_landfrac()[I, J])


def _restart_loader(npoi):
    from ibis.io.restart import load_state
    from ibis.state.model_state import create_ibis_state
    from ibis.dimensions import GridConfig
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    tmpl = create_ibis_state(GridConfig(nlon=720, nlat=360, npoi=npoi, xres=0.5, yres=0.5, nbedrock=3))
    return lambda year: load_state(RESTART.format(year), tmpl)[0]


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------------------------------------
# Figure 13.1: the calculation, from the hourly fire weather to the fate of the killed carbon
# ---------------------------------------------------------------------------------------------------------------

def fig_13_1():
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.set_xlim(0, 120); ax.set_ylim(0, 70); ax.axis("off")
    cols = ((1.0, 34.0, "every time step (1 h)", "#e8f0fa"), (42.0, 34.0, "end of the year", "#f6efe0"),
            (83.0, 36.0, "carbon of each position $j$;\ncover of the cell", "#eaf4e4"))
    for x0, w, head, fc in cols:
        ax.add_patch(FancyBboxPatch((x0, 1.0), w, 61.0, boxstyle="round,pad=0.0,rounding_size=1.2", fc=fc, ec="none"))
        ax.text(x0 + w / 2, 66.0, head, ha="center", va="center", fontsize=9, weight="bold", color=INK, linespacing=1.15)

    def box(x, y, w, h, text, fs=7.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.8", fc="white", ec=GRID, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35)

    def arrow(p, q, color=INK2, lw=0.9):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw))

    # column 1: the default fire weather counter includes day and night (fire.daylight_only false)
    box(3.0, 50.0, 30.0, 8.5, "air temperature $T_a$, humidity $q_a$,\npressure $p_s$ of the step (Ch. 3)")
    box(3.0, 37.0, 30.0, 8.5, "vapour pressure deficit\n$\\delta_e = e_{sat}(T_a) - e_a$  (13.1)")
    box(3.0, 24.0, 30.0, 8.5, "flammability\n$F = \\min(\\delta_e / 3\\,\\mathrm{kPa},\\, 1)$  (13.2)")
    box(3.0, 2.5, 30.0, 17.0, "fire-prone step if $F > 0.33$\n(day or night by default):\n"
                              "$N \\leftarrow N + 1$,  $S \\leftarrow S + F$  (13.3)\n"
                              "$n \\leftarrow n + 1$ every step;\nall three cleared on 1 January", fs=7.3)
    for y0, y1 in ((50.0, 45.5), (37.0, 32.5), (24.0, 19.5)):
        arrow((18.0, y0), (18.0, y1))
    # column 2: the burned fraction. The counters give s and D; the factors f, V and d come from the state (the litter
    # of the year end, the leaf area and the upper-canopy cover of the previous annual update), not from s and D, so
    # their box takes no arrow from the counters and s, D reach the burned fraction past it.
    box(44.0, 49.0, 30.0, 9.5, "fire season $s = N / \\max(n,1)$\ndryness $D = S / \\max(N,1)$  (13.4)")
    box(44.0, 25.0, 25.5, 19.0, "from the state (Ch. 11, 12):\n$f = \\min(1, L / 0.2)$  (13.5)\n"
                                "$V = \\min(1, \\Lambda / 4)$  (13.6)\n$d(f_u, b)$  (13.7)-(13.8)", fs=7.2)
    box(44.0, 6.0, 30.0, 14.0, "burned fraction\n$A = 1 - \\exp(-k_b\\, s\\, D\\, f\\, V\\, d)$\n$k_b = 0.44$  (13.9)", fs=8.0)
    arrow((33.6, 11.0), (43.4, 53.0))
    arrow((71.8, 49.0), (71.8, 20.6)); arrow((56.75, 25.0), (56.75, 20.6))
    # column 3: the pools of each position and the cover of the cell
    box(85.0, 48.0, 32.0, 10.5, "killed: the share $A\\,m_j$ of the leaf,\nwood and root carbon  (13.10)")
    box(85.0, 29.5, 15.0, 13.0, "to the\natmosphere:\n0.9 of leaf,\n0.2 of wood", fs=7.2)
    box(102.0, 29.5, 15.0, 13.0, "to litter\n(Ch. 12): the\nrest, and\nall roots", fs=7.2)
    box(85.0, 14.5, 32.0, 9.5, "background: 0.005 of every\npool to the atmosphere  (13.11)")
    box(85.0, 3.0, 32.0, 8.5, "cell cover $f_u$, $f_l$ reduced by\n$1 - A\\,\\bar m - 0.005$  (13.14)")
    arrow((74.6, 15.0), (84.4, 53.0)); arrow((74.6, 9.0), (84.4, 7.25))
    arrow((95.0, 48.0), (92.5, 43.1)); arrow((107.0, 48.0), (109.5, 43.1))
    fig.savefig(OUT / "fig_13_1_fire_scheme.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 13.2: response of each factor of the burned fraction, with the release parameters
# ---------------------------------------------------------------------------------------------------------------

def _daylight_share():
    """Section 13.3: the share of a year's hourly steps with the sun up (coszen > 0, as the fire counters test it), by
    latitude, from the model's own solar geometry (weather.diurnal) with the sun at the middle of each step (the default,
    weather.sun_midstep) and at its start; the season of the daylight rule divides by half of the year's steps."""
    import jax
    import jax.numpy as jnp
    from ibis.physics.weather import diurnal, DailyWeather
    lat = np.arange(89.75, -90.0, -0.5).astype(np.float32); n = lat.size
    one = jnp.ones((n,), jnp.float32); z = jnp.zeros((n,), jnp.float32)
    dw = DailyWeather(td=one * 290.0, tmax=one * 295.0, tmin=one * 285.0, cloud=one * 0.5, precip=z, psurf=one * 101325.0,
                      qd=one * 0.008, ud=one * 3.0, iwet=z, xstore=jnp.zeros((n, 3), jnp.float32))

    @jax.jit
    def lit(jd, off):
        return jnp.stack([diurnal(k * 3600.0 + off, jd, dw, jnp.asarray(lat), z, z, z, jax.random.PRNGKey(0), 3600.0, 2)[1].coszen
                          for k in range(24)]) > 0.0
    for off, tag in ((1800.0, "middle"), (0.0, "start")):
        share = sum(np.asarray(lit(jnp.asarray(jd), off)).sum(0) for jd in range(1, 366)) / (24.0 * 365)
        mid = np.abs(lat) <= 60.0
        print(f"13.3 daylight share of a 365-day year's steps, sun at the step's {tag}: {share[mid].min():.4f}-{share[mid].max():.4f}"
              f" between 60S and 60N, {share.min():.4f}-{share.max():.4f} at all latitudes")


def fig_13_2():
    import jax.numpy as jnp
    import equinox as eqx
    from ibis.physics.fire import (flammability_vpd, vpd_from_state, legacy_fuel_fraction, vegetation_density,
                                   canopy_damping, fire_seasonlength, DEFAULT_PFT_MORTALITY)
    from ibis.physics.thermo import esat
    fp = _release_fire_params()                                         # current default: every step
    fig, axs = plt.subplots(2, 3, figsize=(8.6, 5.6), gridspec_kw={"hspace": 0.55, "wspace": 0.38})
    ttl = dict(loc="left", fontsize=8.5, weight="bold", color=INK)
    # (a) flammability against VPD, built from temperature and humidity through the model's own VPD
    ta = jnp.full((601,), 303.15); ps = jnp.full((601,), 101325.0)
    ea = jnp.linspace(float(esat(jnp.float32(303.15))), 0.0, 601)               # Pa, from saturated to dry air
    qa = 0.622 * ea / (ps - 0.378 * ea)
    vpd = np.asarray(vpd_from_state(ta, qa, ps)); F = np.asarray(flammability_vpd(ta, qa, ps, fp))
    ax = axs[0, 0]
    ax.axvspan(fp.d_crit_vpd * fp.vpd_ref, float(vpd.max()), color="#fbe3d8", lw=0)
    ax.plot(vpd, F, color=BLUE, lw=1.6)
    ax.axhline(fp.d_crit_vpd, color=INK2, lw=0.8, ls="--")
    ax.text(fp.d_crit_vpd * fp.vpd_ref + 0.1, 0.06, "fire-prone\nday or night", fontsize=7, color=INK2, va="bottom")
    ax.text(float(vpd.max()) - 0.1, fp.d_crit_vpd + 0.03, f"threshold {fp.d_crit_vpd:g}", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xlim(0, float(vpd.max())); ax.set_ylim(0, 1.08)
    ax.set_xlabel("vapour pressure deficit (kPa)"); ax.set_ylabel("flammability $F$")
    ax.set_title("(a) fire weather, (13.2)", **ttl)
    # (b) litter factor
    L = jnp.linspace(0.0, 0.4, 401)
    ax = axs[0, 1]
    ax.plot(np.asarray(L), np.asarray(legacy_fuel_fraction(L)), color=BLUE, lw=1.6)
    ax.set_xlim(0, 0.4); ax.set_ylim(0, 1.08)
    ax.set_xlabel("total litter carbon $L$ (kg C m$^{-2}$)"); ax.set_ylabel("litter factor $f$")
    ax.set_title("(b) litter, (13.5)", **ttl)
    # (c) vegetation density
    lai = jnp.linspace(0.0, 6.0, 601)
    plai = jnp.zeros((601, 12)).at[:, 0].set(lai)
    ax = axs[0, 2]
    ax.plot(np.asarray(lai), np.asarray(vegetation_density(plai, fp)), color=BLUE, lw=1.6)
    ax.set_xlim(0, 6); ax.set_ylim(0, 1.08)
    ax.set_xlabel("total leaf area index $\\Lambda$ (m$^2$ m$^{-2}$)"); ax.set_ylabel("leaf-area factor $V$")
    ax.set_title("(c) leaf area, (13.6)", **ttl)
    # (d) canopy damping against the upper-canopy cover, for three broadleaf shares
    fu = jnp.linspace(0.0, 1.0, 501)
    ax = axs[1, 0]
    for b, c in ((1.0, BLUE), (0.5, ORANGE), (0.0, AQUA)):
        pl = jnp.zeros((501, 12)).at[:, 0].set(b).at[:, 3].set(1.0 - b)       # position 1 broadleaf, 4 conifer
        if b == 0.0:
            pl = pl.at[:, 3].set(1.0)
        ax.plot(np.asarray(fu), np.asarray(canopy_damping(fu, fp, pl)), color=c, lw=1.6, label=f"broadleaf share {b:g}")
    one = jnp.ones((1,)); bl = jnp.zeros((1, 12)).at[:, 0].set(1.0)
    d_closed = float(canopy_damping(jnp.asarray([0.3]), fp, bl, closed=one)[0])
    d_cap = float(canopy_damping(jnp.asarray([0.975]), fp, bl)[0])
    ax.plot([fp.closed_canopy_fu], [d_closed], "o", ms=4.5, color=INK, zorder=5)
    ax.annotate(f"A_humid cells:\n$d$ = {d_closed:.2f}", (fp.closed_canopy_fu, d_closed), xytext=(0.62, 0.17),
                fontsize=7, color=INK, va="center", ha="right", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.axvline(0.975, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.08)
    ax.set_xlabel("upper-canopy cover $f_u$"); ax.set_ylabel("damping $d$")
    ax.legend(frameon=False, fontsize=6.8, loc="center left", bbox_to_anchor=(0.0, 0.47), handlelength=1.6)
    ax.set_title("(d) closed canopy, (13.7)-(13.8)", **ttl)
    # (e) burned fraction against the fire season, all other factors at 1. With the daylight rule the counters hold
    # daylight steps and the season is their share of half the year's steps, so a season s is N = s n / 2 steps.
    s = jnp.linspace(0.0, 1.0, 201); nyt = jnp.asarray(8760.0)
    per = 0.5 if fp.daylight_only else 1.0
    ax = axs[1, 1]
    for D, c in ((1.0, BLUE), (0.75, ORANGE), (0.5, AQUA)):
        A = fire_seasonlength(s * per * nyt, s * per * nyt * D, nyt, jnp.ones_like(s), fp, vd=jnp.ones_like(s))
        ax.plot(np.asarray(s), 100 * np.asarray(A), color=c, lw=1.6, label=f"$D$ = {D:g}")
    amax = 100 * float(fire_seasonlength(per * nyt[None], per * nyt[None], nyt, jnp.ones((1,)), fp, vd=jnp.ones((1,)))[0])
    ax.text(0.98, amax + 1.2, f"{amax:.1f} %", fontsize=7, color=INK, ha="right")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 40)
    ax.set_xlabel("fire season $s$"); ax.set_ylabel("burned fraction $A$ (% yr$^{-1}$)")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=1.6)
    ax.set_title("(e) burned fraction, (13.9)", **ttl)
    # (f) the fire mortality of each position
    m = np.asarray(DEFAULT_PFT_MORTALITY)
    ax = axs[1, 2]
    pos = np.arange(1, 13)
    ax.bar(pos, m, width=0.7, color=[BLUE] * 8 + [AQUA] * 4)
    ax.set_xticks(pos); ax.set_xticklabels([str(p) for p in pos], fontsize=7)
    ax.set_xlim(0.3, 12.7); ax.set_ylim(0, 1.2); ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("PFT position $j$"); ax.set_ylabel("fire mortality $m_j$")
    for x0, x1, lab in ((0.75, 8.25, "trees"), (8.75, 12.25, "shrubs, grasses")):      # group brackets, in ink
        ax.plot([x0, x0, x1, x1], [1.02, 1.05, 1.05, 1.02], color=INK2, lw=0.7, clip_on=False)
        ax.text(0.5 * (x0 + x1), 1.07, lab, fontsize=6.6, color=INK2, ha="center", va="bottom")
    ax.set_title("(f) mortality, Table 13.3", **ttl)
    for a in axs.ravel():
        _spines(a)
    axs[1, 2].grid(axis="x", visible=False)
    fig.savefig(OUT / "fig_13_2_response_curves.png")
    plt.close(fig)
    print(f"13.2 daylight_only {fp.daylight_only}; F threshold at VPD {fp.d_crit_vpd * fp.vpd_ref:.2f} kPa; damping at the"
          f" A_humid cover {d_closed:.3f}, at the cover cap {d_cap:.3f}; maximum burned fraction {amax:.2f} %")
    _daylight_share()
    # section 13.5: the carbon that the CO2, CO and CH4 factors of each category carry, against C_FRACTION_OF_DM
    from ibis.bgc.fire_emissions import EF_TABLE, EF_CATEGORIES, C_FRACTION_OF_DM
    ef = np.asarray(EF_TABLE)
    cc = ef[:, 0] * 12.0 / 44.0 + ef[:, 1] * 12.0 / 28.0 + ef[:, 2] * 12.0 / 16.0
    print("13.5 carbon in CO2 + CO + CH4, g C per kg dry matter: " + ", ".join(
        f"{c} {v:.1f} ({v / (1000 * C_FRACTION_OF_DM):.3f} of the {1000 * C_FRACTION_OF_DM:.0f} g assumed)" for c, v in zip(EF_CATEGORIES, cc)))


# ---------------------------------------------------------------------------------------------------------------
# Figure 13.3: release rule compared with the optional daylight variant, 2001
# ---------------------------------------------------------------------------------------------------------------

CELLS = (("West African savanna", 9.25, -1.75), ("central Amazon", -3.25, -60.25), ("central Siberia", 60.25, 90.25))


def _replay(year, lat, I, J, fp, midstep, keep=None):
    """Replay the hourly fire weather of `year` at the land points (lat, I, J): the GSWP3 forcing through the model's
    forcing conversion and weather generator, with the sun and the weather at the middle of each step (`midstep`, the
    current default, weather.sun_midstep) or at its start (the release), and (13.1)-(13.3). Returns the counters over
    every step (N, S) and over the daylight steps (Nd, Sd; the sun up, coszen > 0, as the model tests it), the number of
    steps n, and for the columns `keep` the flammability and the cosine of the solar zenith angle of every step."""
    import jax
    import jax.numpy as jnp
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.io.gswp3_forcing import gswp3_day_to_daily_weather
    from ibis.physics.weather import diurnal
    from ibis.physics.fire import flammability_vpd
    from ibis.timestepping.calendar import julian_day, is_leap_year, days_in_month
    reader = GSWP3Reader(GSWP3)
    leap = is_leap_year(year); dt = 3600.0; off = 0.5 * dt if midstep else 0.0
    npt = len(lat); z = jnp.zeros((npt,), jnp.float32); la = jnp.asarray(lat, jnp.float32)
    lat_rad = np.deg2rad(lat).astype(np.float32); xs = np.zeros((npt, 3), np.float32)
    thr = np.float32(fp.d_crit_vpd)

    @jax.jit
    def day_weather(dw, jd):
        F, C = [], []
        for istep in range(24):
            _, h = diurnal(istep * dt + off, jd, dw, la, z, z, z, jax.random.PRNGKey(0), dt, 2)
            F.append(flammability_vpd(h.ta, h.qa, dw.psurf, fp)); C.append(h.coszen)
        return jnp.stack(F), jnp.stack(C)

    acc = {k: np.zeros(npt) for k in ("N", "S", "Nd", "Sd")}
    kF, kC = [], []
    for mon in range(12):
        mo = reader.read_month(year, mon, I, J)
        for d in range(days_in_month(mon, leap)):
            jd = julian_day(mon, d, leap)
            dw, _ = gswp3_day_to_daily_weather(mo, d, lat_rad, jd, xs)
            F, C = (np.asarray(x) for x in day_weather(dw, jnp.asarray(jd)))
            prone = F > thr; lit = C > 0.0
            acc["N"] += prone.sum(0); acc["S"] += np.where(prone, F, 0.0).astype(np.float64).sum(0)
            acc["Nd"] += (prone & lit).sum(0); acc["Sd"] += np.where(prone & lit, F, 0.0).astype(np.float64).sum(0)
            if keep is not None:
                kF.append(F[:, keep]); kC.append(C[:, keep])
    n = 24 * (366 if leap else 365)
    return acc, n, ((np.concatenate(kF), np.concatenate(kC)) if keep is not None else None)


def fig_13_3(year=2001):
    import jax.numpy as jnp
    from ibis.physics.fire import vegetation_density, canopy_damping, legacy_fuel_fraction, fire_seasonlength
    from ibis.io.pft_types import ZONES
    from ibis.timestepping.calendar import is_leap_year, days_in_month
    lat_all, lon_all, I_all, J_all, _ = _land_points()
    idx = np.array([int(np.where((lat_all == la) & (lon_all == lo))[0][0]) for _, la, lo in CELLS])
    zone = _zones(lat_all, lon_all, 3)                                 # 1991-2020 map: the zones of 2001
    fp_rel = _release_fire_params(zone, daylight_only=False)           # the release run's rule
    fp_now = _release_fire_params(zone, daylight_only=True)            # optional daylight variant, release parameters
    load = _restart_loader(len(lat_all))
    s0, s1 = load(year - 1), load(year)
    v0, v1, c1 = s0.vegetation, s1.vegetation, s1.carbon
    # the release run's own factors of this year (the fire of `year` reads the leaf area and cover of the previous year
    # end, the counters and litter of this year end)
    vd = np.asarray(vegetation_density(v0.plai, fp_rel), float)
    cd = np.asarray(canopy_damping(v0.fu, fp_rel, v0.plai, closed=fp_rel.closed_canopy_cell), float)
    fl = np.asarray(legacy_fuel_fraction(c1.totlit), float)
    A_run = np.asarray(v1.disturbf, float)
    N_run, S_run = np.asarray(v1.nfirestep_vpd, float), np.asarray(v1.sdry_vpd, float)
    A_re = np.asarray(fire_seasonlength(v1.nfirestep_vpd, v1.sdry_vpd, v1.nytimes, c1.totlit, fp_rel,
                                        vd=jnp.asarray(vd * cd, jnp.float32)), float)
    # the replays at every land point: the release's rule (hour start, every step) and the daylight variant (mid-step,
    # daylight steps)
    rel, n, keep_rel = _replay(year, lat_all, I_all, J_all, fp_rel, midstep=False, keep=idx)
    now, n2, keep_now = _replay(year, lat_all, I_all, J_all, fp_now, midstep=True, keep=idx)
    assert n == n2 == int(float(v1.nytimes))
    s_rel, D_rel = rel["N"] / n, rel["S"] / np.maximum(rel["N"], 1.0)
    s_now, D_now = np.minimum(1.0, now["Nd"] / (0.5 * n)), now["Sd"] / np.maximum(now["Nd"], 1.0)
    A_now = np.asarray(fire_seasonlength(jnp.asarray(now["Nd"], jnp.float32), jnp.asarray(now["Sd"], jnp.float32),
                                         jnp.asarray(float(n)), c1.totlit, fp_now, vd=jnp.asarray(vd * cd, jnp.float32)), float)
    # --- the figure: the three cells, from the current code's replay, the fire-prone hours of each day split by the sun
    Fh, Ch = keep_now
    prone = Fh > np.float32(fp_now.d_crit_vpd); lit = Ch > 0.0
    day_h = (prone & lit).reshape(-1, 24, len(idx)).sum(1); night_h = (prone & ~lit).reshape(-1, 24, len(idx)).sum(1)
    leap = is_leap_year(year)
    fig, axs = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True, gridspec_kw={"hspace": 0.80})
    days = np.arange(1, day_h.shape[0] + 1)
    for k, (name, la, lo) in enumerate(CELLS):
        ax = axs[k]; i = idx[k]
        ax.bar(days, day_h[:, k], width=1.0, color=BLUE, lw=0)
        ax.bar(days, night_h[:, k], bottom=day_h[:, k], width=1.0, color=LIGHT_BLUE, lw=0)
        ax.set_ylim(0, 24); ax.set_yticks([0, 6, 12, 18, 24]); ax.set_xlim(0.5, days[-1] + 0.5)
        ax.set_ylabel("fire-prone\nhours per day")
        ns, ew = ("N" if la > 0 else "S"), ("E" if lo > 0 else "W")
        ax.set_title(f"({'abc'[k]}) {name}, {abs(la)}°{ns} {abs(lo)}°{ew}, {ZONES[zone[i]]}, {year}", loc="left",
                     fontsize=8.5, weight="bold", color=INK)
        ax.text(0.0, -0.32 if k == 2 else -0.19,
                f"daylight variant (mid-step): $s$ = {s_now[i]:.2f}, $D$ = {D_now[i]:.2f};   "
                f"release (every hour, hour start): $s$ = {s_rel[i]:.2f}, $D$ = {D_rel[i]:.2f}\n"
                f"release run: $f$ = {fl[i]:.2f}, $V$ = {vd[i]:.2f}, $d$ = {cd[i]:.2f}  →  $A$ = {100 * A_run[i]:.1f} % in {year}",
                transform=ax.transAxes, fontsize=7.2, color=INK2, va="top", linespacing=1.5)
        _spines(ax); ax.grid(axis="x", visible=False)
    axs[0].legend(handles=[Patch(fc=BLUE, ec="none", label="daylight hours"),
                           Patch(fc=LIGHT_BLUE, ec="none", label="night hours (excluded by daylight variant)")],
                  loc="lower center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=7.2, handlelength=1.4)
    mstart = np.cumsum([0] + [days_in_month(m, leap) for m in range(11)]) + 1
    axs[2].set_xticks(mstart); axs[2].set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    axs[2].tick_params(axis="x", labelbottom=True)
    fig.savefig(OUT / "fig_13_3_fire_weather_year.png")
    plt.close(fig)
    # --- the numbers of sections 13.2 and 13.9
    print(f"13.3 release-rule replay against the counters the release run stored for {year}, all {len(lat_all)} land points:"
          f" largest |N - N_run| {np.abs(rel['N'] - N_run).max():.0f} steps, largest |S - S_run| {np.abs(rel['S'] - S_run).max():.3f}"
          f" (relative {np.max(np.abs(rel['S'] - S_run) / np.maximum(S_run, 1.0)):.1e}); recomputed A vs the run's: "
          f"largest |diff| {np.abs(A_re - A_run).max():.1e}")
    Fr, Cr = keep_rel
    prone_r = Fr > np.float32(fp_rel.d_crit_vpd)
    hours_r = prone_r.reshape(-1, 24, len(idx)).sum(1)
    mon_of_day = np.concatenate([np.full(days_in_month(m, leap), m) for m in range(12)])
    for k, (name, la, lo) in enumerate(CELLS):
        i = idx[k]
        print(f"13.3 {name} ({la}, {lo}) zone {ZONES[zone[i]]}: release rule N {rel['N'][i]:.0f} S {rel['S'][i]:.2f}"
              f" (run N {N_run[i]:.0f} S {S_run[i]:.2f}) s {s_rel[i]:.4f} D {D_rel[i]:.4f}; daylight variant N {now['Nd'][i]:.0f}"
              f" S {now['Sd'][i]:.2f} s {s_now[i]:.4f} D {D_now[i]:.4f}; f {fl[i]:.3f} V {vd[i]:.3f} d {cd[i]:.3f};"
              f" A run {A_run[i]:.4f}, recomputed {A_re[i]:.4f}, daylight variant with the release parameters {A_now[i]:.4f}")
        print(f"13.3 {name}: release rule, fire-prone share of the hours by month " +
              " ".join(f"{hours_r[mon_of_day == m, k].sum() / (24.0 * (mon_of_day == m).sum()):.2f}" for m in range(12)) +
              f"; days with all 24 hours fire-prone {int((hours_r[:, k] == 24).sum())} (by month " +
              " ".join(str(int((hours_r[mon_of_day == m, k] == 24).sum())) for m in range(12)) + ")")
        dl = lit[:, k].reshape(-1, 24).sum(1)                           # daylight hours of each day (mid-step sun)
        print(f"13.3 {name}: daylight variant, fire-prone share of the daylight hours by month " +
              " ".join(f"{day_h[mon_of_day == m, k].sum() / max(dl[mon_of_day == m].sum(), 1):.2f}" for m in range(12)) +
              f"; daylight hours a day {dl.min()}-{dl.max()}; night fire-prone hours no longer counted {int(night_h[:, k].sum())}"
              f" of {int(day_h[:, k].sum() + night_h[:, k].sum())} fire-prone hours (mid-step)")
    # zone means of the year's fire season and dryness under both rules, and the burned fraction each gives with the
    # release parameters and the release run's litter, leaf area and damping (the daylight variant before the refit)
    cell, area = _areas(lat_all, I_all, J_all)
    aw = lambda v, m: float(np.sum(v[m] * area[m]) / area[m].sum())
    for zi, zn in enumerate(ZONES):
        m = zone == zi
        if m.any() and zn not in ("E_ice", "none"):
            print(f"13.3 zone {zn:12s} {year}: s release {aw(s_rel, m):.3f}, daylight {aw(s_now, m):.3f}; D release"
                  f" {aw(D_rel, m):.3f}, daylight {aw(D_now, m):.3f}; A (release parameters) release rule"
                  f" {100 * aw(A_re, m):.2f} %, daylight variant {100 * aw(A_now, m):.2f} %")
    print(f"13.3 land, {year}: s release {aw(s_rel, area > 0):.3f}, daylight {aw(s_now, area > 0):.3f}; burned area with the"
          f" release parameters: release rule {np.sum(A_re * area) / 1e12:.2f} Mkm2, daylight variant {np.sum(A_now * area) / 1e12:.2f} Mkm2")


# ---------------------------------------------------------------------------------------------------------------
# Figure 13.4: the fire of the release run, 2001-2010, with GFED5; prints the chapter's numbers
# ---------------------------------------------------------------------------------------------------------------

def fig_13_4(years=range(2001, 2011), draw=True):
    import jax.numpy as jnp
    from ibis.physics.fire import (fire_seasonlength, vegetation_density, canopy_damping, legacy_fuel_fraction,
                                   DEFAULT_PFT_MORTALITY)
    from ibis.io.pft_types import ZONES, slot_map, gather
    from ibis.bgc.fire_emissions import fire_emissions_from_state, emissions_from_combusted
    from ibis.io.gfed5 import read_gfed5_annual_fraction
    lat, lon, I, J, land = _land_points()
    npoi = len(lat)
    cell, area = _areas(lat, I, J)
    zone = _zones(lat, lon, 3)                                         # 1991-2020: the zone map of 2001-2010
    fp = _release_fire_params(zone, daylight_only=False)               # the release run counted every step
    mort = gather(np.asarray(DEFAULT_PFT_MORTALITY), slot_map(zone))   # per cell; no type sets fire_mort in the release
    load = _restart_loader(npoi)
    acc = {k: [] for k in ("A", "s", "D", "f", "V", "d", "comb", "litter", "bg", "diag", "diag_lit")
           + tuple("e_" + sp for sp in SPECIES) + tuple("d_" + sp for sp in SPECIES)}
    prev = load(years[0] - 1); maxdiff = 0.0
    for y in years:
        st = load(y)
        v0, v1, c1 = prev.vegetation, st.vegetation, st.carbon
        vd = vegetation_density(v0.plai, fp); cd = canopy_damping(v0.fu, fp, v0.plai, closed=fp.closed_canopy_cell)
        A_re = np.asarray(fire_seasonlength(v1.nfirestep_vpd, v1.sdry_vpd, v1.nytimes, c1.totlit, fp, vd=vd * cd), float)
        A = np.asarray(v1.disturbf, float); maxdiff = max(maxdiff, float(np.abs(A_re - A).max()))
        N = np.asarray(v1.nfirestep_vpd, float)
        acc["A"].append(A); acc["s"].append(N / float(v1.nytimes)); acc["D"].append(np.asarray(v1.sdry_vpd, float) / np.maximum(N, 1))
        acc["f"].append(np.asarray(legacy_fuel_fraction(c1.totlit), float))
        acc["V"].append(np.asarray(vd, float)); acc["d"].append(np.asarray(cd, float))
        # the fate of the killed carbon from the post-fire pools: post = pre (1 - A m_j - 0.005), (13.10)-(13.11)
        Dj = A[:, None] * mort + 0.005
        pre = [np.asarray(x, float) / (1.0 - Dj) for x in (v1.cbiol, v1.cbiow, v1.cbior)]
        k_ = A[:, None] * mort
        phi = k_ * (fp.cc_leaf * pre[0] + fp.cc_wood * pre[1])         # the carbon combusted from each position, (13.11)
        acc["comb"].append(phi.sum(1))
        acc["litter"].append(np.sum(k_ * ((1 - fp.cc_leaf) * pre[0] + (1 - fp.cc_wood) * pre[1] + pre[2]), 1))
        acc["bg"].append(0.005 * np.sum(pre[0] + pre[1] + pre[2], 1))
        # the emissions of the current code (D36): each position's combusted carbon with its own emission factors,
        # through the model's own function, on the release run's carbon
        e = emissions_from_combusted(jnp.asarray(phi, jnp.float32), jnp.asarray(A, jnp.float32))
        for sp in SPECIES:
            acc["e_" + sp].append(np.asarray(getattr(e, sp), float))
        fe = fire_emissions_from_state(v1, c1)                         # the release's diagnostic, as the release run logged it
        acc["diag"].append(np.asarray(fe.carbon_burned, float)); acc["diag_lit"].append(A * 0.5 * np.asarray(c1.totalit, float))
        for sp in SPECIES:
            acc["d_" + sp].append(np.asarray(getattr(fe, sp), float))
        prev = st
    M = {k: np.mean(v, 0) for k, v in acc.items()}
    mlat = np.arange(-90.0, 90.0, 0.5); mlon = np.arange(-180.0, 180.0, 0.5)
    li = np.floor((lat - mlat[0]) / 0.5 + 1e-9).astype(int); lo = np.floor((lon - mlon[0]) / 0.5 + 1e-9).astype(int)
    obs = np.mean([read_gfed5_annual_fraction(GFED5, y, li, lo, mlat, mlon, component="Norm") for y in years], 0)
    aw = lambda v, m=None, w=area: float(np.sum(v[m] * w[m]) / w[m].sum()) if m is not None else float(np.sum(v * w) / w.sum())
    tot = lambda v: float(np.sum(v * area))
    zl = [z for z in ZONES if z not in ("E_ice", "none") and (zone == ZONES.index(z)).any()]
    edges = np.arange(-60, 91, 2.0); mid = 0.5 * (edges[:-1] + edges[1:])

    def band(v, w=area):
        out = np.full(len(mid), np.nan)
        for b in range(len(mid)):
            m = (lat >= edges[b]) & (lat < edges[b + 1])
            if m.any() and w[m].sum() > 0:
                out[b] = 100 * np.sum(v[m] * w[m]) / w[m].sum()
        return out
    bA, bO = band(M["A"]), band(obs)
    if draw:
        def grid(v):
            g = np.full(land.shape, np.nan); g[I, J] = v; return g
        fig = plt.figure(figsize=(8.6, 7.4))
        gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 1.0], height_ratios=[1, 1], hspace=0.20, wspace=0.12)
        map_axes = []
        ext = (-180, 180, -90, 90)
        maps = ((M["s"], "(a) fire season $s$, fraction of the year", [1e-9, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
                 ["#c4daf3", "#a8c8ee", "#8db6e7", "#729fde", "#5689d2", "#3d73c2", "#2c5fa8", "#1f4b89", "#133766"],
                 [">0", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8", "1"], "fraction of the year"),
                (100 * M["A"], "(b) burned fraction $A$", [1e-3, 0.5, 1, 2, 4, 6, 8, 12, 16, 25],
                 ["#fbd2ae", "#f9c194", "#f7ad78", "#f49b5c", "#eb7c3c", "#d9622a", "#bd4d1c", "#963a12", "#6b290b"],
                 [">0", "0.5", "1", "2", "4", "6", "8", "12", "16", "25"], "% of the cell per year"))
        for r, (v, title, bounds, ramp, labels, unit) in enumerate(maps):
            ax = fig.add_subplot(gs[r, 0])
            ax.imshow(np.where(land, 0.0, np.nan), extent=ext, cmap=ListedColormap([LAND_BG]), interpolation="nearest")
            im = ax.imshow(np.where(grid(v) > 0, grid(v), np.nan), extent=ext, cmap=ListedColormap(ramp),
                           norm=BoundaryNorm(bounds, len(ramp)), interpolation="nearest")
            _coast(ax, land); _map_axes(ax); ax.set_ylim(-60, 90); ax.set_yticks(range(-60, 91, 30))
            ax.set_yticklabels([f"{abs(t)}°{'S' if t < 0 else 'N' if t > 0 else ''}" for t in range(-60, 91, 30)])
            ax.set_title(title + f", {years[0]}–{years[-1]}", loc="left", fontsize=8.8, weight="bold", color=INK)
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, aspect=45, ticks=bounds)
            cb.ax.set_xticklabels(labels); cb.ax.tick_params(labelsize=6.8, length=2); cb.outline.set_linewidth(0.4)
            cb.set_label(unit, fontsize=7.5)
            map_axes.append(ax)
        # (c) the burned fraction of each climate zone, model and GFED5 Norm (the A and GFED5 columns of Table 13.5)
        zA = np.array([100 * aw(M["A"], zone == ZONES.index(z)) for z in zl])
        zO = np.array([100 * aw(obs, zone == ZONES.index(z)) for z in zl])
        yz = np.arange(len(zl))[::-1].astype(float)
        axz = fig.add_subplot(gs[0, 1])
        axz.barh(yz + 0.2, zA, height=0.38, color=BLUE, lw=0, label="IBIS 3.0")
        axz.barh(yz - 0.2, zO, height=0.38, color=ORANGE, lw=0, label="GFED5 (Norm)")
        axz.set_yticks(yz); axz.set_yticklabels(zl, fontsize=6.8)
        axz.yaxis.tick_right(); axz.tick_params(axis="y", length=0, pad=3)
        axz.set_ylim(-0.7, len(zl) - 0.3); axz.set_xlim(0, 28); axz.set_xticks([0, 10, 20])
        axz.tick_params(axis="x", labelsize=7)
        axz.set_xlabel("burned fraction (% yr$^{-1}$),\nland-area mean", fontsize=7.5)
        axz.set_title("(c) climate zones", loc="left", fontsize=8.8, weight="bold", color=INK)
        axz.legend(frameon=False, fontsize=6.8, loc="lower right", handlelength=1.2)
        # (d) zonal means: model and GFED5 Norm, weighted by the land area of the cells of 2-degree bands
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(bA, mid, color=BLUE, lw=1.3, label="IBIS 3.0")
        ax.plot(bO, mid, color=ORANGE, lw=1.3, label="GFED5 (Norm)")
        ax.set_ylim(-60, 90); ax.set_yticks(range(-60, 91, 30))
        ax.set_yticklabels([f"{abs(t)}°{'S' if t < 0 else 'N' if t > 0 else ''}" for t in range(-60, 91, 30)])
        ax.yaxis.tick_right(); ax.tick_params(axis="y", labelsize=7, length=2)
        ax.set_xlim(0, None); ax.set_xlabel("burned fraction (% yr$^{-1}$),\nland-area mean", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title("(d) 2° bands", loc="left", fontsize=8.8, weight="bold", color=INK)
        ax.legend(frameon=False, fontsize=6.8, loc="upper right", handlelength=1.4)
        for a in (ax, axz):
            for sp in ("top", "left"):
                a.spines[sp].set_visible(False)
            a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
        axz.grid(axis="y", visible=False)
        fig.canvas.draw()                                              # align (c) and (d) with the drawn boxes of the maps
        for a, m_ax in ((axz, map_axes[0]), (ax, map_axes[1])):
            pm, pa = m_ax.get_position(), a.get_position()
            a.set_position([pa.x0, pm.y0, pa.width, pm.height])
        fig.savefig(OUT / "fig_13_4_release_fire.png")
        plt.close(fig)
    # --- the numbers of the chapter (land-area weights unless stated)
    ba = [np.sum(a * area) / 1e12 for a in acc["A"]]
    print(f"13.4 recomputed disturbf vs the run (release rule), {years[0]}-{years[-1]}: max |diff| {maxdiff:.1e}")
    print(f"13.4 land area {area.sum() / 1e12:.2f} Mkm2 (cell area of the {npoi} mask cells {cell.sum() / 1e12:.2f} Mkm2);"
          f" burned {tot(M['A']) / 1e12:.2f} Mkm2 yr-1 (years {min(ba):.2f}-{max(ba):.2f}), {100 * aw(M['A']):.2f} % of the"
          f" land; GFED5 Norm over the same cells {100 * aw(obs):.2f} %")
    print(f"13.4 cell-area means (the weighting of the release fire score): model {100 * aw(M['A'], w=cell):.2f} %,"
          f" GFED5 Norm {100 * aw(obs, w=cell):.2f} %")
    print(f"13.4 cells with s > 0: {int((M['s'] > 0).sum())} of {npoi}; cells with A > 10 %: {int((M['A'] > 0.1).sum())};"
          f" largest A {100 * M['A'].max():.1f} %")
    print(f"13.4 carbon, PgC yr-1: fire, combusted {tot(M['comb']) / 1e12:.3f}; fire, killed to litter {tot(M['litter']) / 1e12:.3f};"
          f" background disturbance {tot(M['bg']) / 1e12:.3f}")
    # the run's own aycdisturb (yearly output) against the reconstruction of (13.12): combusted + background + harvest +
    # the shrub NPP of the cropland share
    import xarray as xr
    hv = []
    for y in years:
        with xr.open_dataset(YEARLY.format(y)) as ds:
            assert ds.lat.size == 360 and float(ds.lat[0]) == 89.75 and float(ds.lon[0]) == -179.75
            hv.append([np.nan_to_num(np.asarray(ds[v].values, float)[I, J]) for v in ("aycdisturb", "ayharvest", "aycropshrub")])
    cd_run, harv, cshrub = np.mean(np.asarray(hv), 0)
    recon = M["comb"] + M["bg"] + harv + cshrub
    print(f"13.4 aycdisturb of the run {tot(cd_run) / 1e12:.3f} PgC yr-1; reconstructed {tot(recon) / 1e12:.3f} (harvest"
          f" {tot(harv) / 1e12:.3f}, cropland shrub NPP {tot(cshrub) / 1e12:.3f}); largest cell difference"
          f" {np.abs(cd_run - recon).max():.1e} kgC m-2 yr-1")
    # section 13.5: the emissions of the current code on the release run's combusted carbon, and the carbon the
    # emitted CO2, CO and CH4 carry against the carbon combusted
    ce = tot(M["e_co2"] * 12.0 / 44.0 + M["e_co"] * 12.0 / 28.0 + M["e_ch4"] * 12.0 / 16.0) / 1e15        # PgC yr-1
    print(f"13.5 emissions of the current code (D36) on the release run's combusted carbon {tot(M['comb']) / 1e12:.3f} PgC yr-1:"
          + "; ".join(f" {sp.upper()} {tot(M['e_' + sp]) / 1e12:.2f} Tg yr-1" for sp in SPECIES)
          + f"; carbon in the CO2, CO and CH4 {ce:.3f} PgC yr-1, {ce / (tot(M['comb']) / 1e12):.3f} of the carbon combusted")
    print(f"13.5 the release's diagnostic (logged by the release run): carbon {tot(M['diag']) / 1e12:.3f} PgC yr-1, of which litter"
          f" {tot(M['diag_lit']) / 1e12:.3f};" + "; ".join(f" {sp.upper()} {tot(M['d_' + sp]) / 1e12:.2f} Tg yr-1" for sp in SPECIES))
    for zi, zn in enumerate(ZONES):
        m = zone == zi
        if m.any():
            print(f"13.9 {zn:12s} cells {int(m.sum()):6d} s {aw(M['s'], m):.2f} D {aw(M['D'], m):.2f} f {aw(M['f'], m):.2f}"
                  f" V {aw(M['V'], m):.2f} d {aw(M['d'], m):.2f} A {100 * aw(M['A'], m):.2f} GFED5 {100 * aw(obs, m):.2f}"
                  f" share of burned area {100 * tot(M['A'] * m) / tot(M['A']):.1f}")
    left = np.isin(zone, [ZONES.index(z) for z in ZONES if z not in zl])
    print(f"13.9 share of the burned area in E_ice and the cells without a zone: {100 * tot(M['A'] * left) / tot(M['A']):.3f} %")
    pk = lambda b, lo_, hi_: (lambda k: f"{b[k]:.1f} % at {mid[k]:+.0f}")(np.nanargmax(np.where((mid > lo_) & (mid < hi_), b, -1)))
    print(f"13.9 band peaks: model {pk(bA, 0, 20)}, {pk(bA, -20, 0)}; GFED5 {pk(bO, 0, 20)}, {pk(bO, -20, 0)}")


FIGURES = {"13.1": fig_13_1, "13.2": fig_13_2, "13.3": fig_13_3, "13.4": fig_13_4,
           "numbers": lambda: fig_13_4(draw=False)}

if __name__ == "__main__":
    for key in (sys.argv[1:] or [k for k in FIGURES if k != "numbers"]):
        FIGURES[key]()
        print("figure" if key != "numbers" else "", key, "done")
