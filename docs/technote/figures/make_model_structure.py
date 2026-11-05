"""Draw Figure 1.1: the implemented IBIS 3.0 structure, not a future architecture.

Run independently (no model execution, forcing files or GPU required)::

    python docs/technote/figures/make_model_structure.py

Only NumPy and Matplotlib are required. The layer geometry is read with AST
from the model source, without importing JAX. Outputs are an editable SVG and
a high-resolution PNG. Optional modules are shown in their integrated setting.

Scientific basis: technote 1.3-1.4, 2.3, 9.2, 11.1, 12.1/12.6, 13.1,
14.1/14.5/14.8 and 15.2. Arrows are selected connections, not a complete budget
or execution sequence. In particular, methane does not debit the carbon pools
a second time, and crops/wetlands do not introduce separate soil columns.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import (
    Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle,
)
import numpy as np

W, H = 1800, 1350
P = {
    "ink": "#193B37", "muted": "#526561", "line": "#D6E2DC",
    "green": "#28785D", "green_light": "#EAF3EB", "leaf": "#67A167",
    "blue": "#316E91", "blue_light": "#EAF3F8", "brown": "#92623F",
    "brown_light": "#F4EEE5", "orange": "#B75C32", "orange_light": "#FBEEE5",
    "gold": "#967127", "gold_light": "#F7F1DF", "purple": "#765988",
    "purple_light": "#F1ECF5", "panel": "#FBFCFA", "white": "#FFFFFF",
}


def geometry(source_root: Path):
    """Read the two small literal parameter arrays without importing the model."""
    soil_path = source_root / "python-port/ibis/io/soil_init.py"
    rock_path = source_root / "python-port/ibis/physics/bedrock.py"
    soil_tree = ast.parse(soil_path.read_text(encoding="utf-8"))
    rock_tree = ast.parse(rock_path.read_text(encoding="utf-8"))
    soil = None
    for node in soil_tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "HSOI_LAYERS" for t in node.targets):
            soil = ast.literal_eval(node.value.args[0])
    rock = None
    for node in rock_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BedrockParams":
            for field in node.body:
                if isinstance(field, ast.AnnAssign) and field.target.id == "hsoi":
                    rock = ast.literal_eval(field.value)
    if soil is None or rock is None:
        raise ValueError("Cannot read model layer geometry; review source before drawing.")
    assert len(soil) == 6 and len(rock) == 3, "Layer counts changed: revise the drawing."
    return np.cumsum(soil), sum(soil) + np.cumsum(rock)


def find_source_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "python-port/ibis/io/soil_init.py").is_file():
            return parent
    raise FileNotFoundError("Pass --source-root pointing to the IBIS repository.")


def make_model_structure(out: Path | None = None, source_root: Path | None = None,
                         preview: bool = False):
    out = Path(out or Path(__file__).resolve().parent)
    out.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_root) if source_root else find_source_root()
    soil_bot, rock_bot = geometry(source_root)
    previous_rc=plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": P["ink"],
        "svg.fonttype": "none", "svg.hashsalt": "ibis-30-structure-20261104",
        "savefig.facecolor": "white", "axes.unicode_minus": False,
        "savefig.bbox": None,
    })
    fig = plt.figure(figsize=(18, 13.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    texts = []

    def text(x, y, value, size=20, color=None, weight="normal", ha="left",
             va="top", linespacing=1.38, **kwargs):
        artist = ax.text(x, y, value, fontsize=size * .72, color=color or P["ink"],
                         weight=weight, ha=ha, va=va, linespacing=linespacing,
                         zorder=8, **kwargs)
        texts.append(artist)
        return artist

    def box(x, y, w, h, fill, edge=None, radius=14, lw=1.1, zorder=2):
        patch = FancyBboxPatch((x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill, edgecolor=edge or fill, linewidth=lw, zorder=zorder)
        ax.add_patch(patch)
        return patch

    def arrow(points, color, dashed=False, double=False, width=2.4, zorder=5):
        vertices = [(float(x), float(y)) for x, y in points]
        path = MplPath(vertices, [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices)-1))
        patch = FancyArrowPatch(path=path, arrowstyle="<|-|>" if double else "-|>",
                  mutation_scale=12, linewidth=width, color=color,
                  linestyle=(0, (4, 3)) if dashed else "solid", zorder=zorder,
                  capstyle="round", joinstyle="round")
        ax.add_patch(patch)

    def line(points, color, width=1.5, **kwargs):
        ax.plot(*zip(*points), color=color, lw=width, solid_capstyle="round", **kwargs)

    cards=[]
    def card(x, y, w, h, title, body, color, fill, body_size=23):
        box(x, y, w, h, fill, edge=color, lw=.8)
        ax.add_patch(Rectangle((x, y+16), 4, h-32, facecolor=color, edgecolor="none", zorder=3))
        tx = x+22
        title_obj=text(tx, y+20, title, size=24, color=color, weight="bold")
        body_obj=text(tx, y+58, body, size=body_size, color=P["muted"],linespacing=1.25)
        cards.append((x,y,w,h,title_obj,body_obj))

    # Header and a prescribed-input banner. No prognostic atmospheric state.
    text(48, 36, "IBIS 3.0", size=46, weight="bold")
    text(48, 96, "Model structure and process connections", size=27, color=P["muted"])
    box(48, 151, 1704, 84, P["blue_light"], radius=14)
    text(72, 167, "PRESCRIBED INPUTS", size=14, weight="bold", color=P["blue"])
    text(72, 192, "Weather + atmospheric CO₂   ·   Soil and land-cover maps   ·   Crop + wetland-area data   ·   Climate-zone parameters",
         size=23, color=P["ink"])

    # Two views of the same system, deliberately not two distinct model tiles.
    box(48, 263, 532, 768, P["panel"], edge=P["line"], radius=16)
    box(612, 263, 1140, 768, P["panel"], edge=P["line"], radius=16)
    text(72, 286, "A", size=19, color=P["green"], weight="bold")
    text(104, 285, "One shared land column", size=24, weight="bold")
    text(72, 322, "35 climate-zone types fill 12 canopy positions", size=18, color=P["muted"])
    text(636, 286, "B", size=19, color=P["green"], weight="bold")
    text(668, 285, "Processes within that column", size=24, weight="bold")

    # Spatial cross-section: physical structure and exact layer boundary labels.
    x0, x1 = 103, 302
    # Sun and a small cloud are schematic forcing symbols, not additional states.
    sunx, suny = 128, 365
    ax.add_patch(Circle((sunx, suny), 17, fc="#EBCB72", ec="none", zorder=3))
    for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
        line([(sunx+23*np.cos(a), suny+23*np.sin(a)),
              (sunx+29*np.cos(a), suny+29*np.sin(a))], "#D6AF55", width=1.8)
    for cx, cy, rx, ry in [(232,360,55,20),(214,354,31,25),(246,352,31,26)]:
        ax.add_patch(Ellipse((cx,cy),rx,ry,fc="#E5EEF2",ec="none",zorder=2))
    arrow([(128, 402), (156, 440)], "#D1AC50", width=1.8)
    for rx in (220,244,267):
        line([(rx,390),(rx-6,403)],P["blue"],width=1.5)

    def broadleaf(cx, ground, h, scale=1):
        top = ground-h
        trunkw=8*scale
        ax.add_patch(Rectangle((cx-trunkw/2,top+53*scale),trunkw,h-53*scale,
                              fc="#816549", ec="none",zorder=3))
        for dx,dy,rx,ry,c in [(-17,45,46,62,"#477B57"),(18,45,49,62,"#3E7650"),
                              (0,23,52,55,"#69965E"),(-14,28,35,46,"#7DAA68"),
                              (19,25,32,44,"#5B8A53")]:
            ax.add_patch(Ellipse((cx+dx*scale,top+dy*scale),rx*scale,ry*scale,
                                 fc=c,ec="none",zorder=4))

    def conifer(cx, ground, h):
        top=ground-h
        ax.add_patch(Rectangle((cx-4,top+30),8,h-30,fc="#79634B",ec="none",zorder=3))
        for yy,half,hh,c in [(15,23,65,"#3F7755"),(48,32,69,"#356B4C"),(78,40,70,"#2C6247")]:
            ax.add_patch(Polygon([(cx,top+yy),(cx-half,top+yy+hh),(cx+half,top+yy+hh)],
                                 closed=True,fc=c,ec="none",zorder=4))
        line([(cx,top+16),(cx-12,top+73)],"#7AA07B",width=2,zorder=5)

    broadleaf(153,588,162,.97)
    conifer(252,588,176)
    # Shrubs and grasses under trees: not disjoint canopy-cover tiles.
    for cx,cy in ((130,583),(174,579)):
        ax.add_patch(Ellipse((cx,cy),34,30,fc="#93B882",ec="none",zorder=5))
        ax.add_patch(Ellipse((cx-7,cy-3),17,19,fc="#ACC894",ec="none",zorder=6))
    for cx in (106,117,191,203,215,228,241,259,272):
        line([(cx,610),(cx-4,579)],"#71994D",width=1.6,zorder=6)
        line([(cx,610),(cx+6,583)],"#91AE62",width=1.6,zorder=6)
    # Surface with a partial snow patch, split into three layers only where present.
    line([(x0,618),(x1,618)],P["green"],width=2.8,zorder=5)
    for yy in (597,604,611):
        ax.add_patch(Rectangle((221,yy),81,7,fc="#E3EDF6",ec="#B6CDDE",lw=.65,zorder=7))

    # Six hydraulic / thermal layers; spacing is schematic rather than metric.
    soil_ys=[618,651,685,720,759,803,851]
    browns=["#D9C7A7","#CEB68E","#BFA17A","#B18E66","#A47D59","#98704F"]
    for k in range(6):
        ax.add_patch(Rectangle((x0,soil_ys[k]),x1-x0,soil_ys[k+1]-soil_ys[k],
                              fc=browns[k],ec="white",lw=1,zorder=2))
    # Roots end within soil; no roots, liquid water or carbon in bedrock.
    for verts in [
        [(153,617),(139,660),(163,699),(144,774)],
        [(153,617),(176,657),(160,710),(184,799)],
        [(252,617),(233,662),(254,715),(231,824)],
        [(252,617),(268,659),(250,697),(274,766)],
    ]:
        ax.add_patch(PathPatch(MplPath(verts,[MplPath.MOVETO]+[MplPath.CURVE4]*3),
                               fc="none",ec="#6D573D",lw=1.65,zorder=4))
    for a,b in [((149,680),(126,706)),((163,711),(185,733)),((240,688),(216,710)),
                ((247,731),(269,751)),((241,785),(221,806))]:
        line([a,b],"#6D573D",width=1.1,zorder=4)
    rock_ys=[851,892,933,977]
    for k,c in enumerate(["#BFC5C1","#AEB8B3","#9EAAA4"]):
        ax.add_patch(Rectangle((x0,rock_ys[k]),x1-x0,rock_ys[k+1]-rock_ys[k],
                              fc=c,ec="white",lw=1,zorder=2))
        for xx in (126,203,275):
            line([(xx,rock_ys[k]+16),(xx+12,rock_ys[k]+21),(xx+22,rock_ys[k]+14)],
                 "#86978C",width=.85,zorder=3)
    for yy,val in zip(soil_ys,[0,*soil_bot]):
        text(94,yy,f"{val:g}",size=17,color=P["muted"],ha="right",va="center")
    for yy,val in zip(rock_ys[1:],rock_bot):
        text(94,yy,f"{val:g}",size=17,color=P["muted"],ha="right",va="center")
    text(80,598,"m",size=16,color=P["muted"],ha="right")
    # Surface labels are deliberately short; process details live in panel B.
    text(328,365,"Sunlit + shaded",size=20,color=P["green"],weight="bold")
    text(328,394,"photosynthesis",size=19,color=P["muted"])
    text(328,448,"Upper canopy",size=21,weight="bold")
    text(328,479,"Trees · positions 1–8",size=18,color=P["muted"])
    text(328,531,"Lower canopy",size=21,weight="bold")
    text(328,561,"Shrubs + grasses",size=18,color=P["muted"])
    text(328,586,"Positions 9–12",size=18,color=P["muted"])
    line([(304,608),(319,608)],P["blue"],width=1.3)
    text(328,623,"Snow, when present",size=19,weight="bold")
    text(328,651,"3 layers · partial cover",size=17,color=P["muted"])
    text(328,716,"Soil · 6 layers",size=21,weight="bold")
    text(328,748,f"0–{soil_bot[-1]:g} m\nWater, ice, heat\nand roots",size=19,color=P["muted"])
    text(328,884,"Bedrock · 3 layers",size=21,weight="bold")
    text(328,918,f"{soil_bot[-1]:g}–{rock_bot[-1]:g} m · heat only\nWhen enabled",size=19,color=P["muted"])
    text(104,997,"Schematic geometry; not to scale",size=16,color=P["muted"])

    # Physical core, then slower carbon processes. Dashed arrows are controls.
    core=(642,366,1080,134)
    arrow([(1182,235),(1182,355)],P["blue"],dashed=True,width=2)
    card(*core,"Surface exchange + soil physics",
         "Radiation · Heat + water exchange · Photosynthesis + stomata\n"
         "Snow · Soil water + heat · Evaporation + root water uptake",
         P["blue"],P["blue_light"])
    veg=(642,600,490,155)
    som=(1232,600,490,155)
    card(*veg,"Vegetation growth & cover",
         "Leaf / wood / root carbon\n"
         "Respiration · Allocation · Turnover\n"
         "Phenology · Water-limited tree cover",
         P["green"],P["green_light"])
    card(*som,"Litter & soil carbon",
         "Decomposition + respiration\n"
         "Whole-cell pools; no vertical layers",
         P["brown"],P["brown_light"])
    arrow([(762,500),(762,591)],P["green"])
    text(775,540,"Carbon gain",size=18,color=P["green"],va="center")
    arrow([(915,500),(915,591)],P["blue"],dashed=True)
    text(928,540,"Water\nstress",size=17,color=P["blue"],va="center")
    arrow([(1026,600),(1026,509)],P["green"],dashed=True)
    text(1042,539,"Leaf area\n+ cover",size=18,color=P["green"],va="center")
    arrow([(1477,500),(1477,591)],P["blue"],dashed=True)
    text(1494,541,"Soil water\n+ temperature",size=18,color=P["blue"],va="center")
    arrow([(1132,680),(1223,680)],P["brown"])
    text(1182,651,"Litter",size=18,color=P["brown"],ha="center")

    # Three integrations are shown as different processes on the SAME land state.
    # Crop grass positions, fire removals and methane bookkeeping are not tiles.
    crop=(642,845,320,155)
    fire=(1006,845,320,155)
    methane=(1370,845,352,155)
    card(*crop,"Cropland & harvest",
         "C3/C4 managed grass\nAnnual carbon removal",
         P["gold"],P["gold_light"])
    card(*fire,"Fire & emissions",
         "Fire weather · Mortality\nBurning + trace gases",
         P["orange"],P["orange_light"])
    card(*methane,"Wetland methane",
         "Shared soil + respiration\nArea climatology + anomaly\nMonthly CH₄ scaling",
         P["purple"],P["purple_light"],body_size=22)
    arrow([(802,755),(802,836)],P["gold"])
    text(815,800,"Harvest",size=18,color=P["gold"],va="center")
    arrow([(680,845),(680,764)],P["gold"],dashed=True)
    text(694,800,"Crop\nshare",size=18,color=P["gold"],va="center")
    arrow([(1081,755),(1166,755),(1166,836)],P["orange"])
    text(1180,800,"Killed\nbiomass",size=18,color=P["orange"],va="center")
    arrow([(1300,845),(1300,785),(1360,785),(1360,764)],P["brown"],width=1.9)
    text(1374,800,"Unburned\nmortality",size=17,color=P["brown"],va="center")
    arrow([(1546,755),(1546,836)],P["purple"],dashed=True)
    text(1560,799,"Respiration\nsubstrate",size=18,color=P["purple"],va="center")

    # Arrow legend belongs to the process panel, away from transfer paths.
    arrow([(666,335),(705,335)],P["muted"],width=1.7)
    text(717,335,"Selected transfer",size=18,color=P["muted"],va="center")
    arrow([(932,335),(971,335)],P["muted"],dashed=True,width=1.7)
    text(983,335,"State or input control",size=18,color=P["muted"],va="center")
    text(1485,335,"Not an execution order",size=18,color=P["muted"],va="center")

    # Readable output and cadence strips replace the old dense time-scale panel.
    box(48,1055,1704,78,"#EDF3F0",radius=12)
    text(72,1070,"COMPUTED QUANTITIES + DIAGNOSTICS",size=14,color=P["green"],weight="bold")
    text(72,1095,"Heat + water fluxes   ·   GPP / NPP / NEE   ·   Runoff + drainage   ·   Fire gases   ·   Wetland CH₄   ·   Harvest",
         size=22)
    xstarts=[48,483,918,1353]
    timings=[
        ("HOURLY","Surface physics, fire weather, CH₄"),
        ("DAILY","Phenology + decomposition"),
        ("MONTHLY","Wetland area + output totals"),
        ("ANNUALLY","Growth, cover, fire + harvest"),
    ]
    for x,(head,body) in zip(xstarts,timings):
        box(x,1157,399,76,P["panel"],edge=P["line"],radius=10)
        text(x+16,1170,head,size=16,color=P["green"],weight="bold")
        text(x+16,1198,body,size=20,color=P["muted"])
    text(48,1260,"Scope",size=18,weight="bold")
    text(118,1260,"One shared column per cell · No lateral routing · Nitrogen is diagnostic; it does not limit growth",
         size=19,color=P["muted"])
    text(48,1295,"Integrated configuration shown. Optional processes depend on run settings; calculation details are in Tables 1.3–1.4.",
         size=17,color=P["muted"])

    # Bounding-box verification checks canvas clipping; human review checks layout.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    canvas=fig.bbox
    for obj in texts:
        bounds=obj.get_window_extent(renderer)
        if bounds.x0<canvas.x0-1 or bounds.y0<canvas.y0-1 or bounds.x1>canvas.x1+1 or bounds.y1>canvas.y1+1:
            outside.append(obj.get_text())
    assert not outside, f"Text outside canvas: {outside}"
    overfull=[]
    for x,y,w,h,title_obj,body_obj in cards:
        for obj in (title_obj,body_obj):
            b=obj.get_window_extent(renderer).transformed(ax.transData.inverted())
            if b.x1>x+w-12 or max(b.y0,b.y1)>y+h-10:
                overfull.append(obj.get_text())
    assert not overfull,f"Text outside its process card: {overfull}"
    stem=out/"fig_1_1_model_structure"
    description=("IBIS 3.0: two canopy layers over fractional snow, six soil layers to 4 m, "
                 "and optional heat-only bedrock to 12 m. Selected process links show "
                 "surface physics, vegetation, whole-cell litter/soil carbon, simple crop "
                 "harvest, fire and diagnostic wetland methane. Crops and wetlands share "
                 "the soil column. No lateral routing or active nitrogen limitation.")
    fig.savefig(stem.with_suffix(".svg"),metadata={"Date":None,"Title":"IBIS 3.0 model structure", "Description":description})
    # Matplotlib uses platform text line endings; keep repository SVGs stable.
    svg_path=stem.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())+"\n",
                        encoding="utf-8",newline="\n")
    fig.savefig(stem.with_suffix(".png"),dpi=220,metadata={"Title":"IBIS 3.0 model structure","Description":description})
    if preview:
        fig.savefig(out/"fig_1_1_preview.png",dpi=100)
    plt.close(fig)
    plt.rcParams.update(previous_rc)
    return {"soil_layer_bottoms_m":soil_bot.tolist(),"bedrock_layer_bottoms_m":rock_bot.tolist(),
            "canvas":[W,H],"png_pixels":[3960,2970],"physical_canopy_layers":2,
            "natural_pft_positions":12,"natural_parameter_types":35,
            "soil_carbon_vertically_resolved":False,"separate_crop_soil_column":False,
            "separate_wetland_soil_column":False,"active_nitrogen_limitation":False,
            "lateral_routing":False,"all_text_within_canvas":True}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--preview",action="store_true")
    args=parser.parse_args()
    print(json.dumps(make_model_structure(args.out,args.source_root,args.preview),indent=2))
