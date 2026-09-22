"""Tower-meteorology forcing for site-level runs (CALIBRATION.md 20, site-first calibration).

`TowerForcingReader` has the same `read_month(year, month0, latindex, lonindex)` contract as
`GSWP3Reader`: a dict of (ndays, npoi) float32 arrays -- tas, tasmax, tasmin (K), pr (kg m-2 s-1),
rsds, rlds (W m-2), huss (kg kg-1), wind (m s-1), ps (Pa) -- plus 'ndays', 'year', 'month'.  Each
column of the arrays is one tower, built from its FLUXNET2015 FULLSET daily file:

    TA_F_MDS (C) -> tas;  tasmax/tasmin from the half-hourly file where it exists, else the
    GSWP3 diurnal range at the tower's cell added to the tower mean;  P_F (mm d-1) -> pr;
    SW_IN_F_MDS -> rsds;  LW_IN_F_MDS -> rlds;  VPD_F_MDS (hPa) with TA and PA -> huss;
    PA_F (kPa) -> ps;  WS_F -> wind.

Gaps (-9999 in the tower file, or a missing variable) fall back to GSWP3 at the tower's
0.5-degree cell for that day and variable; `gap_report()` says how much of each month was
tower data.  Years outside the tower record raise, so a site run can only cover tower years
(spin-up cycles them).
"""
from __future__ import annotations
import glob, re
import numpy as np, pandas as pd

F = '/shared/nas2/Min/FLUXNET2015/fluxnet_compact'
_MISSING = -9000.0


def _dates(col):
    """FLUXNET timestamps come as 1991-01-01, 19910101 or 199101010000; keep the day."""
    return pd.to_datetime(col.astype(str).str.replace('-', '', regex=False).str.replace(' ', '', regex=False).str[:8], format='%Y%m%d')


def _esat_hpa(t_c):
    """Saturation vapour pressure over water (hPa), Magnus form, t in C."""
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def _huss_from_vpd(t_c, vpd_hpa, ps_pa):
    e = np.maximum(_esat_hpa(t_c) - vpd_hpa, 0.05) * 100.0          # actual vapour pressure, Pa
    return 0.622 * e / np.maximum(ps_pa - 0.378 * e, 1.0)


class TowerForcingReader:
    def __init__(self, sites, gswp3_reader, use_hh=True):
        """sites: list of FLUXNET SITE_IDs (one column each); gswp3_reader: GSWP3Reader for gaps
        and for the diurnal range where no half-hourly file exists."""
        self.sites = list(sites); self.g = gswp3_reader; self.variables = gswp3_reader.variables
        self.dd = {}; self.hh = {}; self.gap = {}
        for s in self.sites:
            f = glob.glob(f'{F}/FLX_{s}_FLUXNET2015_FULLSET_DD_*.csv')
            if not f: raise FileNotFoundError(f'no daily FLUXNET file for {s}')
            d = pd.read_csv(f[0]); d['t'] = _dates(d['TIMESTAMP'])
            self.dd[s] = d.set_index('t')
            self.hh[s] = None
            if use_hh:
                h = glob.glob(f'{F}/FLX_{s}_FLUXNET2015_FULLSET_HH_*.csv')
                if h:
                    hd = pd.read_csv(h[0], usecols=['TIMESTAMP', 'TA_F_MDS'])      # compact HH files: TIMESTAMP '1996-01-01 00:00:00', NaN gaps
                    hd['t'] = _dates(hd['TIMESTAMP'])
                    ta = hd['TA_F_MDS'].where(hd['TA_F_MDS'] > _MISSING)
                    self.hh[s] = pd.DataFrame({'tmax': ta.groupby(hd['t']).max(), 'tmin': ta.groupby(hd['t']).min()})
        self.years = {s: (int(self.dd[s].index.year.min()), int(self.dd[s].index.year.max())) for s in self.sites}

    def _col(self, s, name, idx):
        d = self.dd[s]
        if name not in d: return np.full(len(idx), np.nan)
        v = d[name].reindex(idx).values.astype(float); v[v <= _MISSING] = np.nan
        return v

    def read_month(self, year, month0, latindex, lonindex, fill_value=1e20):
        assert len(latindex) == len(self.sites), 'one tower per column'
        g = self.g.read_month(year, month0, latindex, lonindex, fill_value=fill_value)   # fallback + diurnal range
        ndays = int(g['ndays']); idx = pd.date_range(f'{year}-{month0 + 1:02d}-01', periods=ndays, freq='D')
        out = {k: np.array(g[k], dtype=np.float32) for k in ('tas', 'tasmax', 'tasmin', 'pr', 'rsds', 'rlds', 'huss', 'wind', 'ps')}
        out.update(ndays=ndays, year=year, month=month0)
        gap = {}
        for n, s in enumerate(self.sites):
            ta = self._col(s, 'TA_F_MDS', idx); p = self._col(s, 'P_F', idx); sw = self._col(s, 'SW_IN_F_MDS', idx); lw = self._col(s, 'LW_IN_F_MDS', idx)
            vpd = self._col(s, 'VPD_F_MDS', idx); pa = self._col(s, 'PA_F', idx) * 1000.0; ws = self._col(s, 'WS_F', idx)
            def put(key, val):
                ok = np.isfinite(val); out[key][ok, n] = val[ok].astype(np.float32); gap[(s, key)] = float(ok.mean())
            put('tas', ta + 273.15); put('pr', p / 86400.0); put('rsds', sw); put('rlds', lw); put('ps', pa); put('wind', ws)
            ps_eff = np.where(np.isfinite(pa), pa, out['ps'][:, n])
            put('huss', _huss_from_vpd(ta, vpd, ps_eff))
            # diurnal range: tower half-hourly extremes if present, else GSWP3's range around the tower mean
            if self.hh[s] is not None:
                tx = self.hh[s]['tmax'].reindex(idx).values + 273.15; tn = self.hh[s]['tmin'].reindex(idx).values + 273.15
            else:
                tx = np.full(ndays, np.nan); tn = np.full(ndays, np.nan)
            rng_hi = g['tasmax'][:, n] - g['tas'][:, n]; rng_lo = g['tas'][:, n] - g['tasmin'][:, n]
            tx = np.where(np.isfinite(tx), tx, out['tas'][:, n] + rng_hi); tn = np.where(np.isfinite(tn), tn, out['tas'][:, n] - rng_lo)
            out['tasmax'][:, n] = np.maximum(tx, out['tas'][:, n]).astype(np.float32); out['tasmin'][:, n] = np.minimum(tn, out['tas'][:, n]).astype(np.float32)
        self.gap[(year, month0)] = gap
        return out

    def gap_report(self):
        """Fraction of days that came from the tower, per site and variable, over all months read."""
        acc = {}
        for (y, m), gap in self.gap.items():
            for k, v in gap.items(): acc.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in acc.items()}
