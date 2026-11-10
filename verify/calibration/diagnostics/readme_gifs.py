"""The README animations: a global run month by month beside a reference product, with the monthly global series
drawn as the year goes round.

    python3 verify/calibration/diagnostics/readme_gifs.py <run_dir> <out_dir> [gpp|et|ch4|all]

  gpp  GPP (amgpptot) against FLUXCOM RS+METEO (ILAMB, 2001-2010); curves: both totals on the land FLUXCOM covers
  et   ET (amaet) against the mean of the five-product ensemble (2001-2009 climatology); curves: land means on the mask
       of global_spatial.py (true land north of 60 S with an ensemble value)
  ch4  wetland CH4 (amch4) against the wetland prior of the Global Methane Budget 2022 inversions (monthly climatology,
       Saunois et al. 2020), on a square-root colour scale; curves: global totals

Model fields: the 2001-2010 monthly means of <run_dir>/hindcast/out/ibis_monthly_YYYY_MM.nc. Before drawing, the script
prints the annual totals; for the release run (final_rel30) they reproduce score/mip_compare.txt and
score/global_spatial.txt: GPP 147.6 PgC/yr on true land and 144 on FLUXCOM's land against FLUXCOM 109, ET 529 vs
508 mm/yr, CH4 152.5 Tg/yr.

Needs numpy, h5py, matplotlib, Pillow, the Inter typeface (SemiBold and Bold TTF in $IBIS_FONTS) and an ffmpeg with
palettegen/paletteuse ($FFMPEG, `ffmpeg` on the PATH, or the imageio-ffmpeg binary). Output: seasons_<key>.gif
(1040 x 510 px, 12 months x 6 frames at 10 frames per second, looping)."""
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import h5py
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from PIL import Image, ImageDraw, ImageFont

VEGTYPE_1992 = '/shared/nas2/Min/ESACCI_LC/ibis_vegtype/vegtype_frac_1992.nc'   # true land fraction (landmask.py)
FLUXCOM = '/shared/nas2/Chuanfeng/ilambdata/gppfluxcom_ilamb.nc'                # 1980-2013 monthly, g m-2 d-1
ETENS = '/shared/nas2/Hangkai/ibis/inputs/eval/et/et_ensemble_0.5deg.npz'        # 2001-2009, mm d-1
PRIOR = '/shared/nas2/Hangkai/ibis/inputs/eval/mip/gmb2024/gridded/GCP_Prior_CH4_fluxes.nc'
FONTS = os.environ.get('IBIS_FONTS', '/shared/nas2/Hangkai/ibis/inputs/fonts')
YRS = range(2001, 2011)
DIM = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
LAT = np.arange(89.75, -90, -0.5)


def rd(p, v):
    with h5py.File(p) as h:
        x = np.asarray(h[v][:], float)
    return np.where(np.abs(x) > 1e5, np.nan, x)


def true_landfrac():
    """landmask.true_landfrac without xarray (bit-identical): the vegetation-type fractions summed, N->S, in [0, 1]."""
    with h5py.File(VEGTYPE_1992) as h:
        s = np.asarray(h['vegfrac'][:], float).sum(axis=-1); lat = h['lat'][:]
    if lat[0] < lat[-1]:
        s = s[::-1]
    return np.clip(np.nan_to_num(s), 0.0, 1.0)


def prepare(run):
    lf = true_landfrac(); land = lf > 0.05
    band = 6371000.0 ** 2 * np.deg2rad(0.5) * (np.sin(np.deg2rad(LAT + 0.25)) - np.sin(np.deg2rad(LAT - 0.25)))
    area = np.repeat(band[:, None], 720, 1); larea = area * lf
    gpp = np.zeros((10, 12, 360, 720)); et = np.zeros_like(gpp); ch4 = np.zeros_like(gpp)
    for i, y in enumerate(YRS):
        for k in range(12):
            p = f'{run}/hindcast/out/ibis_monthly_{y}_{k + 1:02d}.nc'
            gpp[i, k] = rd(p, 'amgpptot')            # kgC m-2 per month, per m2 of land
            et[i, k] = rd(p, 'amaet')                # mm per month
            ch4[i, k] = rd(p, 'amch4')               # kgC m-2 d-1, per m2 of land
    gpp_tot = np.nansum(gpp * larea, (2, 3)) / 1e12                                             # PgC per month
    ch4_g = ch4 * 1e3 * 16.043 / 12.011                                                         # g CH4 m-2 d-1
    ch4_tot = np.nansum(ch4_g * DIM[None, :, None, None] * larea, (2, 3)) / 1e12                # Tg per month
    d = dict(lf=lf, land=land,
             gpp_m=np.nanmean(gpp, 0) * 1e3 / DIM[:, None, None], et_m=np.nanmean(et, 0) / DIM[:, None, None],
             ch4_m=np.nanmean(ch4_g, 0) * lf[None])                                             # per m2 of cell
    with h5py.File(FLUXCOM) as h:
        fx = np.asarray(h['gpp'][(YRS[0] - 1980) * 12:(YRS[-1] + 1 - 1980) * 12], float)
    d['gpp_ref'] = np.nanmean(np.where(np.abs(fx) > 1e5, np.nan, fx).reshape(10, 12, 360, 720), 0)
    z = np.load(ETENS); d['et_ref'] = np.asarray(z['ens_clim_mean'], float); et_ann = np.asarray(z['ens_ann_mean'], float)
    with h5py.File(PRIOR) as h:
        w = np.asarray(h['flux_ch4_wetlands'][:], float); plat = h['lat'][:]
    w = np.where(np.abs(w) > 1e10, np.nan, w) * 1e3 * 86400                                     # g CH4 m-2 d-1
    if plat[0] < plat[-1]:
        w = w[:, ::-1]
    d['ch4_ref'] = np.repeat(np.repeat(w, 2, 1), 2, 2)
    cg = land & np.all(np.isfinite(d['gpp_ref']), 0) & np.all(np.isfinite(d['gpp_m']), 0)
    gpp_mon = np.nanmean(gpp, 0)                                                                # kgC m-2 per month
    d['s_gpp'] = (np.array([np.nansum(np.nan_to_num(gpp_mon[k]) * larea * cg) / 1e12 for k in range(12)]),
                  np.array([np.nansum(np.nan_to_num(d['gpp_ref'][k]) * DIM[k] * larea * cg) / 1e15 for k in range(12)]))
    wgt = larea * (land & (LAT[:, None] > -60) & np.isfinite(et_ann) & np.all(np.isfinite(d['et_m']), 0))
    d['s_et'] = tuple(np.array([np.nansum(np.nan_to_num(f[k]) * wgt) / wgt.sum() for k in range(12)])
                      for f in (d['et_m'], d['et_ref']))
    d['s_ch4'] = (ch4_tot.mean(0), np.array([np.nansum(np.nan_to_num(d['ch4_ref'][k]) * DIM[k] * area) / 1e12
                                             for k in range(12)]))
    print('GPP  %.1f PgC/yr on true land, %.1f on the land FLUXCOM covers; FLUXCOM %.1f' % (
        gpp_tot.mean(0).sum(), d['s_gpp'][0].sum(), d['s_gpp'][1].sum()))
    print('ET   %.0f mm/yr; five-product mean %.0f' % tuple((s * DIM).sum() for s in d['s_et']))
    print('CH4  %.1f Tg/yr; GMB wetland prior %.1f' % (d['s_ch4'][0].sum(), d['s_ch4'][1].sum()))
    return d


# ---------------------------------------------------------------- drawing (the style of the launch-deck clips)
BG, INK, MUTED, GRID = '#0A0F1E', '#F4F6FA', '#AEB8CC', '#243049'
GREEN, REF = '#5DD39E', '#9FB3D9'
OCEAN, LANDBASE = '#0F1830', '#262F47'
ramp = lambda cols: LinearSegmentedColormap.from_list('r', cols)
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
rgb = lambda h: np.array(to_rgb(h))
S, W, H, M = 2, 1000, 470, 20                          # drawn at 2x; content size and margin in final pixels
fnt = lambda size, bold=False: ImageFont.truetype(f'{FONTS}/Inter-{"Bold" if bold else "SemiBold"}.ttf', int(size * S))
D, CLIPS = {}, {}


def clips(d):
    return {
        'gpp': dict(a=d['gpp_m'], b=d['gpp_ref'], cmap=ramp(['#17382A', '#1B7F4F', '#5DD39E', '#DFFFEF']), vmax=11.0,
                    sqrt=False, s=d['s_gpp'], title='Photosynthesis (GPP)', ref='FLUXCOM', cb='11 gC m⁻² d⁻¹',
                    period='2001–2010 average', stat='PgC per month', ylim=(0, 18), yticks=(5, 10, 15)),
        'et': dict(a=d['et_m'], b=d['et_ref'], cmap=ramp(['#16284A', '#1E5A9E', '#5DA9E9', '#E2F2FF']), vmax=5.0,
                   sqrt=False, s=d['s_et'], title='Evapotranspiration', ref='Five-product mean', cb='5 mm d⁻¹',
                   period='2001–2010 average (products 2001–2009)', stat='mm per day, land mean', ylim=(0, 2.5),
                   yticks=(0.5, 1, 1.5, 2)),
        'ch4': dict(a=d['ch4_m'], b=d['ch4_ref'], cmap=ramp(['#2A1430', '#7A2A7A', '#D46AD4', '#FFDFFF']), vmax=0.08,
                    sqrt=True, s=d['s_ch4'], title='Wetland methane', ref='Global Methane Budget wetland prior',
                    leg='GMB wetland prior', cb='0.08 g CH₄ m⁻² d⁻¹', cbnote='square-root colour scale',
                    period='2001–2010 average; prior: a climatology', stat='Tg CH₄ per month', ylim=(0, 20),
                    yticks=(5, 10, 15)),
    }


def colour(field, c, crop=300):
    """0.5-degree field -> RGB (rows 90N to 60S); low values fade into the land base colour."""
    f = np.nan_to_num(np.asarray(field, float)[:crop]); isl = D['land'][:crop]; vmax = c['vmax']
    if c['sqrt']:
        f, vmax = np.sqrt(np.maximum(f, 0)), np.sqrt(vmax)
    t = np.clip(f / vmax, 0, 1); w = np.clip(f / (vmax * 0.06), 0, 1)[..., None]
    landcol = rgb(LANDBASE) * (1 - w) + c['cmap'](t)[..., :3] * w
    return np.where(isl[..., None], landcol, rgb(OCEAN))


def lerp_month(stack, tau):
    k = int(np.floor(tau)) % 12; fr = tau - np.floor(tau)
    return np.nan_to_num(stack[k]) * (1 - fr) + np.nan_to_num(stack[(k + 1) % 12]) * fr


def frame(args):
    key, n, nfr = args; c = CLIPS[key]; tau = 12.0 * n / nfr
    P = lambda x, y: ((M + x) * S, (M + y) * S)
    im = Image.new('RGB', ((W + 2 * M) * S, (H + 2 * M) * S), BG); dr = ImageDraw.Draw(im)
    for x, f in ((0, c['a']), (510, c['b'])):                         # 720 x 300 cells -> 490 x 204 px
        pic = Image.fromarray((np.clip(colour(lerp_month(f, tau), c), 0, 1) * 255).astype(np.uint8))
        im.paste(pic.resize((490 * S, 204 * S), Image.BILINEAR), P(x, 30))
    dr.text(P(0, 0), 'IBIS 3.0', font=fnt(19), fill=GREEN, anchor='lt')
    dr.text(P(510, 0), c['ref'], font=fnt(19), fill=REF, anchor='lt')
    dr.text(P(0, 258), c['title'], font=fnt(21, True), fill=INK, anchor='lt')
    grad = (c['cmap'](np.linspace(0, 1, 250 * S))[:, :3] * 255).astype(np.uint8)
    im.paste(Image.fromarray(np.repeat(grad[None], 9 * S, 0)), P(0, 296))
    dr.text(P(0, 312), '0', font=fnt(17), fill=MUTED, anchor='lt')
    dr.text(P(250, 312), c['cb'], font=fnt(17), fill=MUTED, anchor='rt')
    if c.get('cbnote'):
        dr.text(P(0, 340), c['cbnote'], font=fnt(15), fill=MUTED, anchor='lt')
    dr.text(P(0, 392), c['period'], font=fnt(17), fill=MUTED, anchor='ls')
    dr.text(P(0, 462), MONTHS[int(np.floor(tau + 0.5)) % 12], font=fnt(56, True), fill=INK, anchor='ls')
    x0, y0, x1, y1 = 420, 316, 1000, 428
    X = lambda x: (M + x0 + x * (x1 - x0)) * S
    Y = lambda y: (M + y1 - (y - c['ylim'][0]) / (c['ylim'][1] - c['ylim'][0]) * (y1 - y0)) * S
    for yt in c['yticks']:
        dr.line([(X(0), Y(yt)), (X(1), Y(yt))], fill=GRID, width=2 * S)
        dr.text((X(0) - 12 * S, Y(yt)), f'{yt:g}', font=fnt(16), fill=MUTED, anchor='rm')
    dr.text(P(x0 - 40, 266), c['stat'], font=fnt(17), fill=MUTED, anchor='lt')
    for k in range(12):
        dr.text((X((k + 0.5) / 12), (M + y1 + 10) * S), 'JFMAMJJASOND'[k], font=fnt(16), fill=MUTED, anchor='mt')
    lx = x1
    for lab, col in ((c.get('leg', c['ref']), REF), ('IBIS 3.0', GREEN)):
        tw = dr.textlength(lab, font=fnt(17)) / S
        dr.text(P(lx - tw, 266), lab, font=fnt(17), fill=INK, anchor='lt')
        dr.rectangle([*P(lx - tw - 30, 274), *P(lx - tw - 8, 279)], fill=col)
        lx -= tw + 52
    # values at the month centres; the curve wraps through the Dec-Jan midpoint at both edges
    u = ((tau + 0.5) / 12.0) % 1.0
    for ys, col in ((c['s'][1], REF), (c['s'][0], GREEN)):
        mid = 0.5 * (ys[-1] + ys[0])
        xe = np.concatenate([[0.0], (np.arange(12) + 0.5) / 12, [1.0]]); ye = np.concatenate([[mid], ys, [mid]])
        pts = [(X(x), Y(y)) for x, y in zip(xe, ye)]
        faint = tuple(int(v * 255 * 0.35 + int(BG[i * 2 + 1:i * 2 + 3], 16) * 0.65) for i, v in enumerate(rgb(col)))
        dr.line(pts, fill=faint, width=3 * S, joint='curve')
        yv = float(np.interp(u, xe, ye)); m = int(np.searchsorted(xe, u, side='right'))
        dr.line(pts[:m] + [(X(u), Y(yv))], fill=col, width=5 * S, joint='curve')
        r = 7 * S; cx, cy = X(u), Y(yv)
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=BG, width=2 * S)
    return im.resize((W + 2 * M, H + 2 * M), Image.LANCZOS)


def ffmpeg():
    exe = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if not exe:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    return exe


def build(key, out, fpm=6, fps=10, nproc=6):
    nfr = 12 * fpm; gif = os.path.join(out, f'seasons_{key}.gif'); ff = ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        with Pool(nproc) as pool:
            for i, im in enumerate(pool.imap(frame, [(key, n, nfr) for n in range(nfr)], chunksize=2)):
                im.save(os.path.join(tmp, f'{i:03d}.png'))
        pal = os.path.join(tmp, 'palette.png'); src = os.path.join(tmp, '%03d.png')
        q = [ff, '-y', '-hide_banner', '-loglevel', 'error', '-framerate', str(fps), '-i', src]
        subprocess.run(q + ['-vf', 'palettegen=max_colors=256:stats_mode=full', pal], check=True)
        subprocess.run(q + ['-i', pal, '-lavfi', 'paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
                            '-loop', '0', gif], check=True)
    print(f'{key}: {nfr} frames at {fps} fps, {os.path.getsize(gif) / 1e6:.2f} MB -> {gif}')


if __name__ == '__main__':
    run, out = sys.argv[1], sys.argv[2]
    what = sys.argv[3] if len(sys.argv) > 3 else 'all'
    os.makedirs(out, exist_ok=True)
    D.update(prepare(run)); CLIPS.update(clips(D))                    # module globals, inherited by the forked pool
    for key in (CLIPS if what == 'all' else [what]):
        build(key, out)
