"""Learned wetland area of a hindcast (finund) against WAD2M 2001-2010, by band: monthly R2 over wet cells, IAV r and sd ratio of
the annual band area, seasonal amplitude; the fixed climatology alongside. Read-only.   python -I learned_eval.py <hindcast out>"""
import sys, numpy as np, xarray as xr, warnings; warnings.filterwarnings('ignore')
R = '/shared/nas2/Hangkai/ibis-port'; sys.path[:0] = [f'{R}/python-port', R]
from ibis.io.pft_types import wetland_band
H = sys.argv[1]; S = '/shared/nas2/Hangkai/ibis/verify_outputs/diag_work'
m0 = xr.open_dataset(f'{H}/ibis_monthly_2001_01.nc'); mlat = np.asarray(m0['lat'].values, float); land = np.isfinite(m0['fu'].values); I, J = np.where(land)
fin = np.array([np.asarray(xr.open_dataset(f'{H}/ibis_monthly_{y}_{m:02d}.nc')['finund'].values, float)[I, J] for y in range(2001, 2011) for m in range(1, 13)])
wd = xr.open_dataset('/shared/nas2/Hangkai/ibis/inputs/methane/wad2m/WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc'); FW = np.asarray(wd['Fw'].values, float)
FW = FW[:, ::-1] if float(wd['lat'].values[0]) < float(wd['lat'].values[-1]) else FW; t = wd['time'].dt.year.values
Fo = np.nan_to_num(FW[(t >= 2001) & (t <= 2010)])[:, I, J]; Fc = np.tile(np.clip(np.nan_to_num(FW).reshape(21, 12, *FW.shape[1:]).mean(0), 0, 1)[:, I, J], (10, 1))   # zero-filled, as the model's climatology
fmax = np.nan_to_num(FW).max(0)[I, J]; wet = fmax > 0.005
zone = np.load(f'{S}/meeting/glob_refs.npz')['zone'][I, J]; band = np.asarray(wetland_band(zone)); lat = mlat[I]
band = np.where(band >= 0, band, np.where(np.abs(lat) < 30, 0, np.where(lat >= 60, 2, 1)))
area = (np.deg2rad(0.5) * 6.371e6) ** 2 * np.cos(np.deg2rad(lat))
def bs(X, sel): return (X[:, sel] * area[sel]).sum(1) / 1e12
def ann(a): return a.reshape(-1, 12).mean(1)
def r2(Q, sel): return 1 - ((Fo[:, sel] - Q[:, sel]) ** 2).sum() / ((Fo[:, sel] - Fo[:, sel].mean(0)) ** 2).sum()
def iav(a, b): a = ann(a); b = ann(b); return float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1]) if b.std() > 0 else float('nan')
def sdr(a, b): a = ann(a); b = ann(b); return float((b - b.mean()).std() / max((a - a.mean()).std(), 1e-12))
print(f'{H}: finund (learned) vs WAD2M 2001-2010 on wet cells (in-sample years for the full model; the out-of-year scores are the training script\'s)')
print(f"{'band':10s} {'obs':>5s} {'model':>5s} | R2 clim {'':>1s} learned | IAV r learned | sd ratio | seasonal amp obs / learned")
for nm, b in (('tropical', 0), ('temperate', 1), ('boreal', 2), ('GLOBAL', None)):
    sel = wet & (np.ones(len(I), bool) if b is None else band == b); o = bs(Fo, sel); f = bs(fin, sel)
    print(f"{nm:10s} {o.mean():5.2f} {f.mean():5.2f} | {r2(Fc, sel):5.2f}  {r2(fin, sel):5.2f} | {iav(o, f):+.2f} | {sdr(o, f):.2f} | {np.ptp(o.reshape(-1, 12).mean(0)):.2f} / {np.ptp(f.reshape(-1, 12).mean(0)):.2f}")
