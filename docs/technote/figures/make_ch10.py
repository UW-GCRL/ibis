"""Figures of Chapter 10 (Phenology) of the IBIS 3.0 technical note, drawn from the model's code, the release
parameter file, the GSWP3 forcing and the release run (final_rel30).

Run from the repository root with the project interpreter:

    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch10.py            # everything (~4 min)
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch10.py 10.1       # one figure
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch10.py table      # Table 10.4
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch10.py drought    # section 10.6, Table 10.2
    /shared/nas2/Hangkai/ibis-env/ibis-python docs/technote/figures/make_ch10.py tc         # numbers of section 10.13

(with PYTHONPATH=python-port and JAX_PLATFORMS=cpu). Figure 10.2 runs the model's own `pheno`
(ibis/physics/phenology.py) and `daylength_hours` (ibis/state/seasonal.py) once a day through 2001 at one cell, from
the release run's state at the end of 2000, with the per-cell parameters that `pft_types.apply_types` gathers for the
cell's zone, and checks the result against the release hindcast's month-end leaf area of both canopies. The update of
the senescence sums and of the ten-day mean temperature is the one of `loops_gswp3._run_month_scan._day_body`,
transcribed below. `drought` does the same for every land cell with the drought factors held at 1 and compares the
lower canopy with the hindcast; `tc` compares the coldest-month statistic of the release run (monthly means) with that
of IBIS 2.6b4 (ten-day means), which the current code uses by default (`pheno.climate_stats_daily`, technote review D9).
Figure 10.1 evaluates `daylength_hours` over latitude and day of year. Table 10.4 is printed by `zone_table`.

What the code changes of the technote review (2026-11-01) do to these outputs: none changes Figures 10.1 and 10.2 (the
drought factors are left out of the computed curves of Figure 10.2, and at its cell the upper leaf-drop temperature is
0 C under either coldest-month rule; run on the revised code on 2026-11-02, the PNG files are byte-identical to those of
the release code); the drought thresholds of Table 10.4 follow the new zone rule (D15), and `table` prints the release
run's values next to them; `drought` reproduces the release run's numbers (its thresholds by the release's rule).
Figure 10.2 was redrawn on 2026-11-02 for one label only: panel (e) is titled "display factors" (formerly
"cold-phenology display factors"), because its squares are the release run's min(phi_l, D_b), drought factors included.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, GRID, LAND_BG, _land_mask, _map_axes, _coast   # noqa: E402,F401  (style)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-port"))
OUT = Path(__file__).resolve().parent

PARAMS = ROOT / "python-port/configs/params/params_31s_ch4h.yaml"
KOPPEN = "/shared/nas2/Hangkai/ibis/inputs/koppen/koppen_beck2023_0.5deg.npz"
GSWP3 = "/shared/nas2/Min/GSWP3"
REL = Path("/shared/nas2/Hangkai/ibis/verify_outputs/final_rel30")
RESTART_2000 = REL / "transient/restart/ibis_restart_2000.npz"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
REF = "#8a8a86"


def _clean(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _release_params():
    import yaml
    return yaml.safe_load(open(PARAMS))


# ---------------------------------------------------------------------------------------------------------------
# Table 10.4: the per-cell phenology scalars of each zone
# ---------------------------------------------------------------------------------------------------------------

ZONE_KEYS = ("gdd_thresh_u", "gdd_ramp_u", "sen_ycrit_u", "tthresh_l_c", "gdd_thresh_l", "drop_stress_thresh")


def _zone_types(pt, z, which, values):
    """The zone's types that supply a per-cell scalar under the rule `which` of pft_types.zone_scalar (the types that
    carry a value), so that the table can name them next to the value."""
    row = pt.ZONE_SLOTS[pt.ZI[z]]; hab = pt.ZONE_HABIT[pt.ZI[z]]
    if which == "upper":
        tids = [pt.IDS[row[p]] for p in range(8) if row[p] >= 0 and hab[p] == pt.CD]
    elif which == "lower":
        g = [pt.IDS[row[p]] for p in (10, 11) if row[p] >= 0]; s = [pt.IDS[row[p]] for p in (8, 9) if row[p] >= 0]
        tids = [t for t in g if t in values] or [t for t in s if t in values]
    elif which == "drought":      # drought-deciduous tree types and the types of positions 10-12 (review D15)
        tids = ([pt.IDS[row[p]] for p in range(8) if row[p] >= 0 and hab[p] == pt.DD]
                + [pt.IDS[row[p]] for p in (9, 10, 11) if row[p] >= 0])
    else:
        tids = [pt.IDS[r] for r in row if r >= 0]
    return [t for t in tids if t in values] if z not in ("none", "E_ice") else []


def zone_table():
    """The value of each per-cell phenology scalar in each zone (pft_types.zone_scalar, the rule apply_types uses),
    with the types that supply it. The drought threshold is also given under the rule of the release run (the median
    over every type of the zone, 'all'), which the current code replaced by the median over the types whose positions
    read a drought factor ('drought'; technote review D15)."""
    from ibis.io import pft_types as pt
    from ibis.io.physics_params import build_pheno_params
    d = _release_params()
    base = build_pheno_params(d)
    over = pt.parse_type_overrides(d.get("types"))
    rows = {}
    for z in pt.ZONES:
        vals, src = [], []
        for k in ZONE_KEYS:
            which = pt.TYPE_SCALARS[k][2]
            v = pt.zone_scalar(z, over.get(k, {}), which) if (k in over and z not in ("none", "E_ice")) else None
            vals.append(float(getattr(base, k)) if v is None else v); src.append(_zone_types(pt, z, which, over.get(k, {})))
        k = "drop_stress_thresh"
        v_rel = pt.zone_scalar(z, over.get(k, {}), "all") if (k in over and z not in ("none", "E_ice")) else None
        rel = (float(getattr(base, k)) if v_rel is None else v_rel, _zone_types(pt, z, "all", over.get(k, {})))
        rows[z] = (vals, src, rel)
        print(f"10.4 {z:12s} " + " ".join(f"{k}={v:.3f}" for k, v in zip(ZONE_KEYS, vals))
              + "  from " + "; ".join(f"{k}:{'+'.join(s) or 'base'}" for k, s in zip(ZONE_KEYS, src))
              + f"  | release run's rule: drop_stress_thresh={rel[0]:.3f} from {'+'.join(rel[1]) or 'base'}")
    print("10.4 base values", {k: round(float(getattr(base, k)), 4) for k in base.__dataclass_fields__})
    print("10.4 zone placement and habits\n" + pt.describe_zones())
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Figure 10.2: one year of the daily phenology at one cell
# ---------------------------------------------------------------------------------------------------------------

def _restart_fields(path, fields, k):
    """Named fields of a release restart at land point k: the restart stores the IBISState leaves in tree order, so
    the field names follow from the dataclass fields of the modules it carries (checked against its leaf count)."""
    from ibis.state.soil import SoilState
    from ibis.state.snow import SnowState
    from ibis.state.canopy import CanopyState
    from ibis.state.atmosphere import AtmState
    from ibis.state.vegetation import VegetationState
    from ibis.state.carbon import CarbonState
    from ibis.state.hydrology import HydrologyDiag
    from ibis.state.summary import SummaryState
    from ibis.bgc.methane import MethaneState
    from ibis.state.stress_accum import StressAccum
    from ibis.state.et_parts import ETParts
    from ibis.state.seasonal import SeasonalState
    mods = [("soil", SoilState), ("snow", SnowState), ("canopy", CanopyState), ("atm", AtmState),
            ("vegetation", VegetationState), ("carbon", CarbonState), ("hydrology", HydrologyDiag),
            ("summary", SummaryState), ("wetland", MethaneState), ("stress", StressAccum), ("etparts", ETParts),
            ("seasonal", SeasonalState)]
    names = [f"{m}.{f.name}" for m, c in mods for f in dataclasses.fields(c) if not f.metadata.get("static", False)]
    r = np.load(path, allow_pickle=True)
    assert len(names) == int(r["num_leaves"]), (len(names), int(r["num_leaves"]))
    out = {f: r[f"leaf_{names.index(f)}"][k] for f in fields}
    assert np.ndim(out["vegetation.tc"]) == 0 and np.shape(out["vegetation.plai"]) == (12,)
    return out


def fig_10_2(I=93, J=181, year=2001):
    """The cell at 43.25 N, 89.25 W (Madison, as Figure 3.1), D_temperate in the Köppen map of 1991-2020."""
    import jax.numpy as jnp
    import equinox as eqx
    import xarray as xr
    from ibis.io import pft_types as pt
    from ibis.io.physics_params import build_all_params
    from ibis.io.koppen import load_koppen, period_index
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.physics.phenology import pheno
    from ibis.state.seasonal import daylength_hours
    from ibis.state.vegetation import create_vegetation_state
    from ibis.state.summary import create_summary_state
    from ibis.timestepping.calendar import julian_day, is_leap_year
    from ibis.constants import PhysicalConstants

    land = _land_mask()
    ij = np.argwhere(land)                                           # the land-point order of run_global
    k = int(np.where((ij[:, 0] == I) & (ij[:, 1] == J))[0][0])
    lat = 89.75 - 0.5 * I; lon = -179.75 + 0.5 * J
    vnames = ("tc", "plai", "fu", "fl", "tempu", "templ", "agddu", "agddl", "dropu", "dropls", "dropl3", "dropl4",
              "lai", "frac", "zbot", "ztop")
    snames = ("a10td", "a10ancub", "a10ancls", "a10ancl3", "a10ancl4")
    r = _restart_fields(RESTART_2000, [f"vegetation.{n}" for n in vnames] + [f"summary.{n}" for n in snames]
                        + ["seasonal.ssen_u", "seasonal.ssen_l", "soil.stresstl", "soil.stresstu", "soil.wisoi"], k)
    # the cell's parameters, as run_global gathers them for the Köppen period of the year
    d = _release_params()
    lat_rad = jnp.asarray([np.deg2rad(lat)], jnp.float32)
    params, pftp, _ = build_all_params(1, lat_rad, d)
    kop = load_koppen(KOPPEN, np.array([I]), np.array([J]))
    zone = pt.zones_from_classes(kop["classes"])[period_index(year, kop["period_start"], kop["period_end"])]
    params, pftp = pt.apply_types(params, pftp, zone, pt.parse_type_overrides(d.get("types")), base=(params, pftp))
    ph = params.pheno
    habit = np.asarray(params.slot_habit)[0]
    P = {n: float(np.asarray(getattr(ph, n)).ravel()[0]) for n in
         ("tthresh_u_offset", "gthresh_u_c", "gdd_thresh_u", "gdd_ramp_u", "tthresh_l_c", "gthresh_l_c",
          "gdd_thresh_l", "gdd_ramp_l", "sen_pstart_u", "sen_tb_u", "sen_ycrit_u", "sen_pstart_l", "sen_tb_l",
          "sen_ycrit_l", "tthresh_u_abs")}
    tmelt = 273.16
    thr_u = max(0.0, float(r["vegetation.tc"]) + P["tthresh_u_offset"], P["tthresh_u_abs"])   # deg C, eq. (10.3)
    thr_l = P["tthresh_l_c"]
    # state at the end of 2000 (float32, as the global run)
    veg = create_vegetation_state(1, 12, 6)
    veg = eqx.tree_at(lambda v: tuple(getattr(v, n) for n in vnames), veg,
                      tuple(jnp.asarray(np.asarray(r[f"vegetation.{n}"])[None], jnp.float32) for n in vnames))
    sm = create_summary_state(1)
    sm = eqx.tree_at(lambda s: tuple(getattr(s, n) for n in snames), sm,
                     tuple(jnp.asarray([r[f"summary.{n}"]], jnp.float32) for n in snames))
    su = jnp.asarray([r["seasonal.ssen_u"]], jnp.float32); sl = jnp.asarray([r["seasonal.ssen_l"]], jnp.float32)
    eps = PhysicalConstants().epsilon
    zweight = np.float32(np.exp(-1.0 / 10.0))
    reader = GSWP3Reader(GSWP3); leap = is_leap_year(year)
    rec = []; month_end = []
    for m in range(12):
        mo = reader.read_month(year, m, np.array([I]), np.array([J]))
        for dd in range(mo["ndays"]):
            jd = julian_day(m, dd, leap)
            td = jnp.asarray(np.nan_to_num(mo["tas"][dd]), jnp.float32)
            # loops_gswp3 _day_body: senescence sums from the day's mean temperature (deg C), before pheno
            tc_ = td - tmelt
            dl = daylength_hours(lat_rad, jd); dec = dl < daylength_hours(lat_rad, jd - 1)
            su = jnp.where(dec & (dl < ph.sen_pstart_u), su + jnp.maximum(ph.sen_tb_u - tc_, 0.0), 0.0)
            sl = jnp.where(dec & (dl < ph.sen_pstart_l), sl + jnp.maximum(ph.sen_tb_l - tc_, 0.0), 0.0)
            on_u = su >= ph.sen_ycrit_u; on_l = sl >= ph.sen_ycrit_l
            a10_prev = float(sm.a10td[0])
            # pheno with the ten-day means of the day before; the drought triggers are left inert here
            # (stress None -> unstressed, the net-assimilation means held at the end-2000 values)
            veg = pheno(veg, sm, td, eps, ph, None, None, sen_on_u=on_u, sen_on_l=on_l, habit=params.slot_habit)
            sm = eqx.tree_at(lambda s: s.a10td, sm, zweight * sm.a10td + (1 - zweight) * td)
            rec.append((jd, float(td[0]) - tmelt, a10_prev - tmelt, float(dl[0]), float(su[0]), float(sl[0]),
                        float(veg.agddu[0]), float(veg.agddl[0]), float(veg.tempu[0]), float(veg.templ[0]),
                        float(veg.lai[0, 1]), bool(on_u[0]), bool(on_l[0]), bool(dec[0])))
        month_end.append(len(rec) - 1)
    a = np.array(rec, dtype=float)
    jd, tdc, a10c, dl, su_, sl_, gu, gl, phu, phl, laiu = (a[:, i] for i in range(11))
    on_u_, on_l_, dec_ = a[:, 11] > 0, a[:, 12] > 0, a[:, 13] > 0
    # the release run's month-end upper-canopy leaf area per unit cover, and the display factor it implies
    plai = np.asarray(r["vegetation.plai"], float); fu = float(r["vegetation.fu"])
    ev = [j for j in range(8) if habit[j] == 0]; cd = [j for j in range(8) if habit[j] == 2]
    dd_ = [j for j in range(8) if habit[j] == 1]
    rel_lai, rel_lail = [], []
    for m in range(12):
        ds = xr.open_dataset(REL / f"hindcast/out/ibis_monthly_{year}_{m + 1:02d}.nc")
        rel_lai.append(float(ds.laiu.sel(lat=lat, lon=lon, method="nearest").values))
        rel_lail.append(float(ds.lail.sel(lat=lat, lon=lon, method="nearest").values))
    rel_lai = np.array(rel_lai); rel_lail = np.array(rel_lail)
    rel_phi = (rel_lai * fu - plai[ev].sum() - plai[dd_].sum() * float(r["vegetation.dropu"])) / plai[cd].sum()
    # lower canopy: the display factor of the deciduous lower positions 10-12 implied by the release's month-end leaf
    # area, (lail fl - L9) / (L10 + L11 + L12): the leaf-area-weighted mean of min(phi_l, D_b) over the three
    # positions (equal to min(phi_l, D) only where the three D are equal; at this cell L11 = 0 and the month-end
    # values are 0 or 1)
    fl = float(r["vegetation.fl"])
    rel_phl = (rel_lail * fl - plai[8]) / plai[9:12].sum()
    err = np.abs(laiu[month_end] - rel_lai)
    # key dates
    first = lambda c: int(jd[np.argmax(c)]) if c.any() else None
    d_out0 = first(phu > 0); d_out1 = first(phu >= 1.0)
    d_sen = first(on_u_); d_gone = first((jd > (d_sen or 400)) & (phu <= 0.0))
    d_lout0 = first(phl > 0); d_lout1 = first(phl >= 1.0); d_lsen_temp = first((jd > 182) & (a10c < thr_l))
    d_lsen = first(on_l_); d_lgone = first((jd > 182) & (phl <= 0.0))
    d_gu0 = first(gu > 0); d_win = first(dec_ & (dl < P["sen_pstart_u"])); d_s0 = first(su_ > 0)
    i_gone = int(np.argmax(jd == d_gone)); i_l1 = int(np.argmax(jd == d_lout1))
    print(f"10.2 G_u first > 0 on day {d_gu0} (ten-day mean read that day {a10c[np.argmax(jd == d_gu0)]:.2f} C); "
          f"window opens day {d_win}; S first > 0 day {d_s0} (T_d {tdc[np.argmax(jd == d_s0)]:.1f} C, "
          f"day before {tdc[np.argmax(jd == d_s0) - 1]:.1f} C); ten-day mean read on day {d_gone}: {a10c[i_gone]:.1f} C; "
          f"G_l on day {d_lout1 - 1}: {gl[i_l1 - 1]:.0f} K d, ten-day mean read on day {d_lout1 - 1} "
          f"{a10c[i_l1 - 1]:.2f} C and on day {d_lout1} {a10c[i_l1]:.2f} C")

    fig, axs = plt.subplots(5, 1, figsize=(7.4, 10.4), sharex=True, gridspec_kw={"hspace": 0.32})
    leg = dict(frameon=False, fontsize=6.6, handlelength=2.0, borderaxespad=0.3)
    mticks = [julian_day(m, 0, leap) for m in range(12)]
    ax = axs[0]
    ax.plot(jd, tdc, color=REF, lw=0.6, label="daily mean $T_d$")
    ax.plot(jd, a10c, color=INK, lw=1.3, label=r"ten-day mean $\bar{T}_{10}$")
    ax.axhline(thr_u, color=BLUE, lw=1.0, ls="--", label=f"leaf drop, upper: $T^*_u$ = {thr_u:.2f} °C")
    ax.axhline(thr_l, color=ORANGE, lw=1.0, ls="--", label=f"leaf drop, lower: $T^*_l$ = {thr_l:.2f} °C")
    mn = lambda x: f"{x:.0f}".replace("-", "\u2212")
    # at this cell T*_u = max(0, Tc + 5) = 0 C equals the upper reset temperature: the dotted line lies on the dashed one
    same_u = abs(thr_u - P["gthresh_u_c"]) < 1e-6
    ax.axhline(P["gthresh_u_c"], color=BLUE, lw=0.8, ls=":",
               label=f"degree-day reset, upper ({mn(P['gthresh_u_c'])} °C{'; here equal to $T^*_u$' if same_u else ''})")
    ax.axhline(P["gthresh_l_c"], color=ORANGE, lw=0.8, ls=":", label=f"degree-day reset, lower ({mn(P['gthresh_l_c'])} °C)")
    ax.set_ylabel("°C"); ax.set_ylim(min(tdc.min(), -12) - 4, tdc.max() + 17)
    ax.set_yticks(np.arange(-20, tdc.max() + 17, 10))
    ax.legend(loc="upper left", ncol=3, columnspacing=1.0, **leg)
    ax.set_title("(a) air temperature", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[1]
    ax.plot(jd, gu, color=BLUE, lw=1.4, label="upper canopy $G_u$")
    ax.plot(jd, gl, color=ORANGE, lw=1.4, label="lower canopy $G_l$")
    GCUT = 800.0                                                     # the axis is cut: the sums reach 3000-4400 K d
    for g0, rr, c, nm in ((P["gdd_thresh_u"], P["gdd_ramp_u"], BLUE, "u"), (P["gdd_thresh_l"], P["gdd_ramp_l"], ORANGE, "l")):
        ax.axhspan(g0, g0 + rr, color=c, alpha=0.18, lw=0)
        # labelled in late summer, where both sums are above the axis cut and before the senescence resets
        # (days 271 and 287); label text in ink, not in the series colours
        ax.text(262, g0 + rr + 12, f"$G^*_{nm}$ to $G^*_{nm} + R_{nm}$: {g0:.0f}–{g0 + rr:.0f} K d", fontsize=6.4,
                color=INK2, va="bottom", ha="right")
    ax.set_ylabel("K d"); ax.set_ylim(-40, GCUT)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), **leg)
    ax.set_title("(b) growing degree-days", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[2]
    win = dec_ & (dl < P["sen_pstart_u"])
    ax.fill_between(jd, 0, 24, where=win, color=BLUE, alpha=0.10, lw=0, step="mid",
                    label="day length decreasing and below $P_{start}$")
    ax.plot(jd, dl, color=INK, lw=1.3, label="day length $N$")
    ax.axhline(P["sen_pstart_u"], color=REF, lw=0.9, ls="--", label=f"$P_{{start}}$ = {P['sen_pstart_u']:.0f} h")
    ax.set_ylim(6, 20); ax.set_ylabel("h"); ax.legend(loc="upper left", ncol=3, columnspacing=1.0, **leg)
    ax.set_title("(c) day length", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[3]
    if np.allclose(su_, sl_):                                         # same P_start and T_b in both canopies
        ax.plot(jd, su_, color=INK, lw=1.4, label="$Y_u = Y_l$ (same $P_{start}$ and $T_b$)")
    else:
        ax.plot(jd, su_, color=BLUE, lw=1.4, label="upper canopy $Y_u$")
        ax.plot(jd, sl_, color=ORANGE, lw=1.4, label="lower canopy $Y_l$")
    ax.axhline(P["sen_ycrit_u"], color=BLUE, lw=0.9, ls="--", label=f"$Y_{{crit,u}}$ = {P['sen_ycrit_u']:.0f} K d")
    ax.axhline(P["sen_ycrit_l"], color=ORANGE, lw=0.9, ls="--", label=f"$Y_{{crit,l}}$ = {P['sen_ycrit_l']:.0f} K d")
    ax.set_ylabel("K d"); ax.set_ylim(0, max(su_.max(), sl_.max()) * 1.25)
    ax.legend(loc="upper left", ncol=2, **leg)
    ax.set_title("(d) senescence sums", loc="left", fontsize=8.5, weight="bold", color=INK)
    ax = axs[4]
    ax.plot(jd, phu, color=BLUE, lw=1.4, label=r"upper canopy $\phi_u$")
    ax.plot(jd, phl, color=ORANGE, lw=1.4, label=r"lower canopy $\phi_l$")
    # the two marker sets coincide where both factors are 0 or 1 (ten of the twelve month ends): open circles
    # large enough for the small squares to sit inside them
    ax.plot(jd[month_end], rel_phi, ls="none", marker="o", ms=6.0, mfc="none", mec=BLUE, mew=1.0,
            label=r"release run, upper: $\phi_u$")
    ax.plot(jd[month_end], rel_phl, ls="none", marker="s", ms=2.6, mfc=ORANGE, mec=ORANGE, mew=0.6,
            label=r"release run, lower: $\min(\phi_l, D_b)$, leaf-area mean")
    ax.set_ylim(-0.05, 1.45); ax.set_ylabel("display factor")
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, **leg)
    # "display factors", not "cold-phenology": the squares are the release run's min(phi_l, D_b), drought factors included
    ax.set_title("(e) display factors", loc="left", fontsize=8.5, weight="bold", color=INK)
    for a_ in axs:
        _clean(a_); a_.set_xlim(1, jd[-1] + 1)
        a_.set_xticks(mticks); a_.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    axs[-1].set_xlabel(f"{year}")
    fig.savefig(OUT / "fig_10_2_phenology_year.png")
    plt.close(fig)
    print(f"10.2 cell {k} ({lat}, {lon}) zone {pt.ZONES[int(zone[0])]}; habits {habit.tolist()}; "
          f"tc {float(r['vegetation.tc']):.2f} C -> T*_u {thr_u:.2f} C, T*_l {thr_l:.2f} C; params {P}")
    print(f"10.2 plai {np.round(plai, 3).tolist()} fu {fu:.4f}; EV {ev} CD {cd} DD {dd_}")
    print(f"10.2 end-2000 state: a10td {float(r['summary.a10td']) - tmelt:.2f} C, agddu {float(r['vegetation.agddu']):.1f}, "
          f"tempu {float(r['vegetation.tempu']):.3f}, templ {float(r['vegetation.templ']):.3f}, ssen_u {float(r['seasonal.ssen_u']):.1f}")
    print(f"10.2 end-2000 drought state: dropu {float(r['vegetation.dropu']):.3f} dropls {float(r['vegetation.dropls']):.3f} "
          f"dropl3 {float(r['vegetation.dropl3']):.3f} dropl4 {float(r['vegetation.dropl4']):.3f}; stresstl "
          f"{float(r['soil.stresstl']):.3f} stresstu {float(r['soil.stresstu']):.3f}; ice fraction by layer "
          f"{np.round(np.asarray(r['soil.wisoi'], float), 3).tolist()}; drop_stress_thresh "
          f"{float(np.asarray(ph.drop_stress_thresh).ravel()[0]):.3f}; fl {fl:.4f}")
    print(f"10.2 month-end laiu offline {np.round(laiu[month_end], 4).tolist()}\n     release        {np.round(rel_lai, 4).tolist()}"
          f"\n     max |diff| {err.max():.2e}")
    print(f"10.2 month-end phi_l offline {np.round(phl[month_end], 3).tolist()}\n     release min(phi_l, D) "
          f"{np.round(rel_phl, 3).tolist()}")
    print(f"10.2 upper: leaf-out starts day {d_out0}, full day {d_out1}; senescence sum reaches Ycrit day {d_sen}, "
          f"leaves gone day {d_gone}; lower: leaf-out starts {d_lout0}, full {d_lout1}; a10 below T*_l from day "
          f"{d_lsen_temp}; S_l reaches Ycrit day {d_lsen}; gone day {d_lgone}")
    print(f"10.2 max G_u {gu.max():.0f}, max G_l {gl.max():.0f}, max S_u {su_.max():.0f}, max S_l {sl_.max():.0f}")
    return dict(jd=jd, phu=phu, phl=phl)


# ---------------------------------------------------------------------------------------------------------------
# Figure 10.1: when the senescence sum can grow, by latitude and day of the year
# ---------------------------------------------------------------------------------------------------------------

def fig_10_1():
    import jax.numpy as jnp
    from ibis.state.seasonal import daylength_hours
    from ibis.io.physics_params import build_pheno_params
    ph = build_pheno_params(_release_params())
    pstart = float(ph.sen_pstart_u)
    lat = np.linspace(-89.75, 89.75, 360)
    doy = np.arange(1, 366)
    L, D = np.meshgrid(np.deg2rad(lat), doy, indexing="ij")
    dl = np.asarray(daylength_hours(jnp.asarray(L), jnp.asarray(D, jnp.float32)))
    dl_prev = np.asarray(daylength_hours(jnp.asarray(L), jnp.asarray(D - 1, jnp.float32)))
    win = (dl < dl_prev) & (dl < pstart)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.contourf(doy, lat, win.astype(float), levels=[0.5, 1.5], colors=[BLUE], alpha=0.22)
    ax.contour(doy, lat, dl, levels=[pstart], colors=[BLUE], linewidths=1.2)
    polar_night = (dl <= 0.0)
    ax.contour(doy, lat, polar_night.astype(float), levels=[0.5], colors=[INK2], linewidths=0.7, linestyles=":")
    ax.text(345, 80, "polar\nnight", fontsize=6.6, color=INK2, ha="center", va="center")
    ax.text(170, -80, "polar\nnight", fontsize=6.6, color=INK2, ha="center", va="center")
    ax.text(268, 40, "the sum grows", fontsize=7, color=INK, ha="center", va="center")
    ax.text(85, -40, "the sum grows", fontsize=7, color=INK, ha="center", va="center")
    leap = False
    from ibis.timestepping.calendar import julian_day
    ax.set_xticks([julian_day(m, 0, leap) for m in range(12)])
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(-90, 91, 30))
    ax.set_yticklabels([f"{abs(y)}°{'S' if y < 0 else 'N' if y > 0 else ''}" for y in range(-90, 91, 30)])
    ax.set_xlim(1, 365); ax.set_ylim(-90, 90)
    ax.set_xlabel("day of the year"); ax.set_ylabel("latitude")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(fc=BLUE, alpha=0.22, label=f"day length decreasing and below {pstart:.0f} h"),
                       Line2D([], [], color=BLUE, lw=1.2, label=f"day length {pstart:.0f} h ($P_{{start}}$)"),
                       Line2D([], [], color=INK2, lw=0.7, ls=":", label="edge of the polar night (day length 0)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig_10_1_senescence_window.png")
    plt.close(fig)
    # numbers for the text
    for la in (0.25, 10.25, 20.25, 30.25, 31.25, 40.25, 50.25, 60.25, 70.25):
        i = int(np.argmin(np.abs(lat - la)))
        w = win[i]
        print(f"10.1 lat {lat[i]:.2f}: max day length {dl[i].max():.2f} h, window days {int(w.sum())}, "
              f"first day {int(doy[np.argmax(w)]) if w.any() else None}, last {int(doy[len(w) - 1 - np.argmax(w[::-1])]) if w.any() else None}")
    lmax = lat[(dl.max(axis=1) < pstart)]
    print(f"10.1 latitudes whose longest day is shorter than {pstart} h: |lat| <= {np.abs(lmax).max():.2f}")


# ---------------------------------------------------------------------------------------------------------------
# Numbers of section 10.6: the lower-canopy drought factors in the release, by zone
# ---------------------------------------------------------------------------------------------------------------

def _restart_all(path, fields):
    """Named fields of a release restart at every land point (the leaf order of _restart_fields)."""
    from ibis.state.soil import SoilState
    from ibis.state.snow import SnowState
    from ibis.state.canopy import CanopyState
    from ibis.state.atmosphere import AtmState
    from ibis.state.vegetation import VegetationState
    from ibis.state.carbon import CarbonState
    from ibis.state.hydrology import HydrologyDiag
    from ibis.state.summary import SummaryState
    from ibis.bgc.methane import MethaneState
    from ibis.state.stress_accum import StressAccum
    from ibis.state.et_parts import ETParts
    from ibis.state.seasonal import SeasonalState
    mods = [("soil", SoilState), ("snow", SnowState), ("canopy", CanopyState), ("atm", AtmState),
            ("vegetation", VegetationState), ("carbon", CarbonState), ("hydrology", HydrologyDiag),
            ("summary", SummaryState), ("wetland", MethaneState), ("stress", StressAccum), ("etparts", ETParts),
            ("seasonal", SeasonalState)]
    names = [f"{m}.{f.name}" for m, c in mods for f in dataclasses.fields(c) if not f.metadata.get("static", False)]
    r = np.load(path, allow_pickle=True)
    assert len(names) == int(r["num_leaves"]), (len(names), int(r["num_leaves"]))
    return {f: r[f"leaf_{names.index(f)}"] for f in fields}


def drought_stats(year=2001):
    """(1) The release state at the end of 2000: by zone, the area share of the cells with deciduous lower-canopy leaf
    area (positions 10-12, > 0.01 m2 m-2) whose lower-canopy drought factors stand at the floor D_min, and of these the
    share whose stress factor is below the zone's threshold with ice in the top soil layer, or whose ten-day mean net
    assimilation of a lower big leaf is negative; the threshold is the one the release run used (the median over every
    type of the zone), not the current rule of review D15. These are release-run numbers: the release had no
    growing-season gate on the trigger (review D11). (2) The cold phenology of the lower canopy run with the model's `pheno`
    for every land cell through `year` from that state with the drought factors held at 1 (stress not supplied, the
    net-assimilation means 0), against the release hindcast's month-end displayed lower-canopy leaf area lail * fl:
    the ratio release / cold phenology alone, by zone and month (area-weighted sums)."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import xarray as xr
    from ibis.io import pft_types as pt
    from ibis.io.physics_params import build_all_params
    from ibis.io.koppen import load_koppen, period_index
    from ibis.io.gswp3_reader import GSWP3Reader
    from ibis.physics.phenology import pheno
    from ibis.state.seasonal import daylength_hours
    from ibis.state.vegetation import create_vegetation_state
    from ibis.state.summary import create_summary_state
    from ibis.timestepping.calendar import julian_day, is_leap_year
    land = _land_mask()
    ij = np.argwhere(land); I, J = ij[:, 0], ij[:, 1]; npoi = len(I)
    lat = 89.75 - 0.5 * I; area = np.cos(np.deg2rad(lat))
    vn = ("tc", "plai", "fu", "fl", "tempu", "templ", "agddu", "agddl", "dropu", "dropls", "dropl3", "dropl4", "lai",
          "frac", "zbot", "ztop")
    r = _restart_all(RESTART_2000, [f"vegetation.{n}" for n in vn]
                     + ["summary.a10td", "summary.a10ancls", "summary.a10ancl3", "summary.a10ancl4",
                        "seasonal.ssen_u", "seasonal.ssen_l", "soil.stresstl", "soil.wisoi"])
    d = _release_params()
    kop = load_koppen(KOPPEN, I, J)
    zone = pt.zones_from_classes(kop["classes"])[period_index(year, kop["period_start"], kop["period_end"])]
    lat_rad = jnp.asarray(np.deg2rad(lat), jnp.float32)
    params, pftp, _ = build_all_params(npoi, lat_rad, d)
    params, pftp = pt.apply_types(params, pftp, zone, pt.parse_type_overrides(d.get("types")), base=(params, pftp))
    ph = params.pheno
    # the threshold the release run used: the median over every type of the zone (rule 'all'); the current code takes
    # the median over the types whose positions read a drought factor (review D15), which is not what the release
    # state at the end of 2000 was run with
    from ibis.io.physics_params import build_pheno_params
    _over = pt.parse_type_overrides(d.get("types")).get("drop_stress_thresh", {})
    _base_thr = float(build_pheno_params(d).drop_stress_thresh)                  # the base value of the file
    _zv = [pt.zone_scalar(z, _over, "all") if (_over and z not in ("none", "E_ice")) else None for z in pt.ZONES]
    thr = np.array([_base_thr if v is None else float(v) for v in _zv])[np.asarray(zone, int)]
    plai = np.asarray(r["vegetation.plai"], float); low_dec = plai[:, 9:12].sum(1)
    dmin = np.minimum(np.minimum(r["vegetation.dropls"], r["vegetation.dropl3"]), r["vegetation.dropl4"])
    floor = dmin <= float(ph.drop_floor) + 1e-6
    frozen = np.asarray(r["soil.wisoi"])[:, 0] > 0.0
    stressed = np.asarray(r["soil.stresstl"]) < thr
    negan = (np.minimum(np.minimum(r["summary.a10ancls"], r["summary.a10ancl3"]), r["summary.a10ancl4"])
             < float(ph.drop_thresh))
    print("10.6 (1) end of 2000, cells with deciduous lower-canopy leaf area > 0.01: area share with D at the floor; "
          "of these: stress below threshold with ice in layer 1 / stress below threshold / negative ten-day net "
          "assimilation")
    for zi, z in enumerate(pt.ZONES):
        s = (zone == zi) & (low_dec > 0.01)
        if not s.any():
            continue
        a = area[s].sum(); f = s & floor; af = max(area[f].sum(), 1e-12)
        print(f"10.6   {z:12s} cells {s.sum():6d}  floor {area[f].sum() / a:.3f}  frozen+stressed "
              f"{area[f & stressed & frozen].sum() / af:.3f}  stressed {area[f & stressed].sum() / af:.3f}  "
              f"negative An {area[f & negan].sum() / af:.3f}")
    veg = create_vegetation_state(npoi, 12, 6)
    veg = eqx.tree_at(lambda v: tuple(getattr(v, n) for n in vn), veg,
                      tuple(jnp.asarray(np.asarray(r[f"vegetation.{n}"]), jnp.float32) for n in vn))
    veg = eqx.tree_at(lambda v: (v.dropls, v.dropl3, v.dropl4, v.dropu), veg,
                      tuple(jnp.ones(npoi, jnp.float32) for _ in range(4)))
    sm = create_summary_state(npoi)
    sm = eqx.tree_at(lambda s: s.a10td, sm, jnp.asarray(r["summary.a10td"], jnp.float32))
    su = jnp.asarray(r["seasonal.ssen_u"], jnp.float32); sl = jnp.asarray(r["seasonal.ssen_l"], jnp.float32)
    zw = np.float32(np.exp(-1.0 / 10.0))

    @jax.jit
    def day(veg, sm, su, sl, td, jd):                       # loops_gswp3._day_body, the parts the cold phenology reads
        tc_ = td - 273.16
        dl = daylength_hours(lat_rad, jd); dec = dl < daylength_hours(lat_rad, jd - 1)
        su = jnp.where(dec & (dl < ph.sen_pstart_u), su + jnp.maximum(ph.sen_tb_u - tc_, 0.0), 0.0)
        sl = jnp.where(dec & (dl < ph.sen_pstart_l), sl + jnp.maximum(ph.sen_tb_l - tc_, 0.0), 0.0)
        veg = pheno(veg, sm, td, 1e-7, ph, None, None, sen_on_u=su >= ph.sen_ycrit_u, sen_on_l=sl >= ph.sen_ycrit_l,
                    habit=params.slot_habit)
        sm = eqx.tree_at(lambda s: s.a10td, sm, zw * sm.a10td + (1 - zw) * td)
        return veg, sm, su, sl

    reader = GSWP3Reader(GSWP3); leap = is_leap_year(year)
    ratio = np.full((len(pt.ZONES), 12), np.nan); ratio_dec = np.full((len(pt.ZONES), 12), np.nan)
    for m in range(12):
        mo = reader.read_month(year, m, I, J)
        tas = np.where(np.isnan(mo["tas"]), 273.16, mo["tas"]).astype(np.float32)
        for dd in range(mo["ndays"]):
            veg, sm, su, sl = day(veg, sm, su, sl, jnp.asarray(tas[dd]), julian_day(m, dd, leap))
        dec = np.asarray(veg.templ, float) * low_dec                 # the deciduous positions 10-12 alone
        nod = plai[:, 8] + dec                                       # with the evergreen shrubs of position 9
        ds = xr.open_dataset(REL / f"hindcast/out/ibis_monthly_{year}_{m + 1:02d}.nc")
        rel = np.asarray(ds.lail.values)[I, J] * np.asarray(ds.fl.values)[I, J]
        for zi in range(len(pt.ZONES)):
            s = (zone == zi) & (low_dec > 0.01)
            if s.sum() >= 50 and (area[s] * nod[s]).sum() > 1e-6:
                ratio[zi, m] = (area[s] * rel[s]).sum() / (area[s] * nod[s]).sum()
                if (area[s] * dec[s]).sum() > 1e-6:
                    ratio_dec[zi, m] = (area[s] * (rel[s] - plai[s, 8])).sum() / (area[s] * dec[s]).sum()
    months = "J F M A M J J A S O N D".split()
    print(f"10.6 (2) {year}, month-end displayed lower-canopy leaf area, release / cold phenology alone")
    print("10.6   zone          " + " ".join(f"{mm:>5s}" for mm in months))
    for zi, z in enumerate(pt.ZONES):
        if np.isfinite(ratio[zi]).any():
            print(f"10.6   {z:12s} " + " ".join(f"{x:5.2f}" for x in ratio[zi]))
    print(f"10.6 (2b) the same for the deciduous positions 10-12 alone (position 9 taken out of both)")
    for zi, z in enumerate(pt.ZONES):
        if np.isfinite(ratio_dec[zi]).any():
            print(f"10.6   {z:12s} " + " ".join(f"{x:5.2f}" for x in ratio_dec[zi]))
    print("10.6 Table 10.2 (markdown, the ratios of (2) for the zones with at least 50 cells, E_ice and 'none' left out)")
    print("| Zone | " + " | ".join(months) + " |\n|---|" + "---|" * 12)
    for zi, z in enumerate(pt.ZONES):
        if np.isfinite(ratio[zi]).any() and z not in ("E_ice", "none"):
            print(f"| {z} | " + " | ".join(f"{x:.2f}" for x in ratio[zi]) + " |")


def tc_stats(year=2001):
    """Numbers of section 10.13: the year's coldest-month statistic of the release run (the lowest monthly mean of the
    daily forcing temperature) against that of IBIS 2.6b4 with daily forcing (the year's lowest ten-day running mean a10td,
    weather.f line 909), over all land cells, the ten-day mean started from the release state at the end of 2000; and
    the leaf-drop threshold max(0, Tc + 5) of the two on the cells whose zone holds a cold-deciduous tree type. The
    current code uses the IBIS 2.6b4 rule by default (pheno.climate_stats_daily, technote review D9)."""
    import xarray as xr
    from ibis.io import pft_types as pt
    from ibis.io.koppen import load_koppen, period_index
    from ibis.io.gswp3_reader import GSWP3Reader
    land = _land_mask()
    ij = np.argwhere(land); I, J = ij[:, 0], ij[:, 1]
    a10 = np.asarray(_restart_all(RESTART_2000, ["summary.a10td"])["summary.a10td"], np.float64)
    w = np.exp(-1.0 / 10.0)
    reader = GSWP3Reader(GSWP3)
    mon, a10min = [], np.full(len(I), np.inf)
    for m in range(12):
        tas = np.asarray(reader.read_month(year, m, I, J)["tas"], np.float64)
        mon.append(np.nanmean(tas, axis=0) - 273.16)
        for dd in range(tas.shape[0]):
            # `daily` (weather.f line 909) runs before the day's time steps, so it reads the ten-day mean as sumnow
            # left it at the end of the day before: the year's minimum is taken over the start-of-day values (on
            # 1 January the value of the restart), as in the Fortran, and the update follows
            a10min = np.minimum(a10min, a10)
            a10 = w * a10 + (1 - w) * np.nan_to_num(tas[dd], nan=273.16)
    tc3 = np.nanmin(np.stack(mon, axis=1), axis=1); tc2 = a10min - 273.16
    diff = tc3 - tc2
    print(f"10.13 {year}: coldest monthly mean minus lowest ten-day mean, {len(I)} land cells: median {np.median(diff):.2f} K, "
          f"5th-95th percentile {np.percentile(diff, 5):.2f} to {np.percentile(diff, 95):.2f} K")
    kop = load_koppen(KOPPEN, I, J)
    zone = pt.zones_from_classes(kop["classes"])[period_index(year, kop["period_start"], kop["period_end"])]
    hab = pt.slot_habit(zone); slots = pt.slot_map(zone)
    has_cd = ((hab[:, :8] == pt.CD) & (slots[:, :8] != pt.OFF)).any(axis=1)
    t3 = np.maximum(0.0, tc3 + 5.0); t2 = np.maximum(0.0, tc2 + 5.0)
    ch = has_cd & (np.abs(t3 - t2) > 0.01)                    # "differs": by more than 0.01 K (the text says so)
    print(f"10.13 cells with a cold-deciduous tree position: {has_cd.sum()}; leaf-drop threshold differs (> 0.01 K) on "
          f"{ch.sum()} ({100 * ch.sum() / has_cd.sum():.1f} %), higher under the release rule by {np.median((t3 - t2)[ch]):.2f} K "
          f"(median), {np.percentile((t3 - t2)[ch], 95):.2f} K (95th percentile); lower on {(ch & (t3 < t2)).sum()} cells; "
          f"exact inequality: differs on {(has_cd & (t3 != t2)).sum()}, lower on {(has_cd & (t3 < t2)).sum()}; "
          f"coldest monthly mean below the lowest ten-day mean on {(diff < 0).sum()} land cells")


FIGURES = {"10.1": fig_10_1, "10.2": fig_10_2, "table": zone_table, "drought": drought_stats, "tc": tc_stats}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
        print("figure", key, "done")
